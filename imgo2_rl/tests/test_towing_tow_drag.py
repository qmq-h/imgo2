"""P4 tow-drag entry-point tests; standard Python, no simulator.

拖曳入口只在 `main()` 内部导入 Isaac Lab，所以模块级可以离线加载。这里锁住三件容易
静默出错的事：① 初始布局算术必须与场景配置的挂点常量、`cart.urdf` 的实际挂点一致
（差一点初始绳张力就不对）；② 记录器的严格校验；③ 施力/步进顺序与「不改 locomotion」
「失败时显式带码退出」（CART-02 的教训）这些接口契约。
"""

import ast
import csv
import importlib.util
import math
import re
from pathlib import Path
import sys
import tempfile
import unittest

RL = Path(__file__).resolve().parents[1]
REPO = RL.parent
TOW_DRAG = RL / "scripts/towing/tow_drag.py"
SCENE_CFG = RL / "source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/towing_env_cfg.py"

sys.path.insert(0, str(RL / "scripts/tools"))
sys.path.insert(0, str(RL / "source/imgo2_rl/imgo2_rl/assets"))


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tow_drag = module_at("towing_tow_drag_test", TOW_DRAG)
recording = module_at("towing_recording_test",
                      RL / "source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/utils/recording.py")
from cart_model import read_cart_model  # noqa: E402


def scene_constants():
    """从场景配置里取出模块级常量（该模块 import isaaclab，离线不能直接导入）。"""
    tree = ast.parse(SCENE_CFG.read_text(encoding="utf-8"))
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    found[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    continue
    return found


class LayoutTests(unittest.TestCase):
    """初始挂点间距必须等于 L0 + slack，否则初始张力就错了。"""

    def setUp(self):
        self.constants = scene_constants()
        self.cart_model = read_cart_model(REPO / "imgo2_description/cart/cart.urdf")

    def test_scene_exposes_attachment_and_spawn_constants(self):
        self.assertIn("ROBOT_ATTACHMENT_OFFSET_M", self.constants)
        self.assertIn("ROBOT_SPAWN_HEIGHT_M", self.constants)
        offset = self.constants["ROBOT_ATTACHMENT_OFFSET_M"]
        self.assertEqual(len(offset), 3)
        # 后挂点必须在机体后表面之后：base 碰撞箱长 0.315 ⇒ 后表面 -0.1575
        self.assertLessEqual(offset[0], -0.1575)
        self.assertAlmostEqual(offset[1], 0.0)
        self.assertAlmostEqual(offset[2], 0.0)

    def _gap(self, length, slack):
        """按配置解出的布局，算两个挂点的**三维**距离。"""
        offset = self.constants["ROBOT_ATTACHMENT_OFFSET_M"]
        cart_offset = self.cart_model["attachment_position_m"]
        height = self.constants["ROBOT_SPAWN_HEIGHT_M"]
        cart_height = self.cart_model["resting_height_m"]
        cart_x = self._cart_x(length, slack)
        robot_point = (0.0 + offset[0], 0.0 + offset[1], height + offset[2])
        cart_point = (cart_x + cart_offset[0], 0.0 + cart_offset[1], cart_height + cart_offset[2])
        return math.dist(robot_point, cart_point)

    def test_initial_gap_equals_rope_length_minus_slack(self):
        """`--slack` 是松弛量：初始间距必须 = L0 − slack（小于 L0）。"""
        for length, slack in ((1.0, 0.05), (1.5, 0.0), (0.8, 0.2)):
            self.assertAlmostEqual(self._gap(length, slack), length - slack, places=9,
                                   msg=f"L0={length} slack={slack}")

    def test_initial_condition_leaves_the_rope_slack(self):
        """回归：t=0 的张力必须是 0。

        实测第一版把初始间距写成 L0 + slack，5 cm 就对应 200 N 预载；叠加「用刚体原点
        当挂点」的错误后实测 1955 N，0.1 s 内把机器人拽到 pitch −0.74 rad 并拽倒。
        """
        rope = module_at("towing_rope_gap_test",
                         RL / "source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/mdp/rope.py")
        for length, slack in ((1.0, 0.05), (1.0, 0.0), (1.2, 0.3)):
            gap = self._gap(length, slack)
            self.assertLessEqual(gap, length + 1e-9, "初始间距必须不大于绳长")
            tension = rope.rope_tension(gap, 0.0, rest_length=length,
                                        stiffness=4000.0, damping=100.0)
            self.assertEqual(tension, 0.0, f"L0={length} slack={slack} 初始张力应为 0")

    def test_rejects_geometry_that_cannot_hold_the_requested_slack(self):
        """高差大于目标距离时必须报错，而不是悄悄给出错的布局。"""
        offset = self.constants["ROBOT_ATTACHMENT_OFFSET_M"]
        cart_offset = self.cart_model["attachment_position_m"]
        with self.assertRaises(ValueError):
            tow_drag.initial_cart_x(1.0, 0.99, spawn_height=0.35, cart_height=0.15,
                                    robot_offset=offset, cart_offset=cart_offset)

    def test_cart_attachment_faces_the_robot(self):
        """小车的挂点在车体 +x；机器人在前、小车在后 ⇒ 该点朝向机器人，拖曳方向正确。"""
        self.assertGreater(self.cart_model["attachment_position_m"][0], 0.0)
        self.assertLess(self._cart_x(1.0, 0.0), 0.0)

    def _cart_x(self, length, slack):
        return tow_drag.initial_cart_x(
            length, slack, spawn_height=self.constants["ROBOT_SPAWN_HEIGHT_M"],
            cart_height=self.cart_model["resting_height_m"],
            robot_offset=self.constants["ROBOT_ATTACHMENT_OFFSET_M"],
            cart_offset=self.cart_model["attachment_position_m"])

    def test_sign_convention_is_robot_in_front_and_cart_behind(self):
        offset = self.constants["ROBOT_ATTACHMENT_OFFSET_M"][0]
        cart_offset = self.cart_model["attachment_position_m"][0]
        cart_x = tow_drag.initial_cart_x(1.0, 0.05, spawn_height=0.35, cart_height=0.15,
                                         robot_offset=self.constants["ROBOT_ATTACHMENT_OFFSET_M"],
                                         cart_offset=self.cart_model["attachment_position_m"])
        self.assertLess(cart_x, 0.0, "小车必须在机器人后方")
        self.assertLessEqual(offset - cart_offset, 0.0)


class ArgumentTests(unittest.TestCase):
    def test_defaults_match_the_plan(self):
        args = tow_drag.parse_args([])
        self.assertEqual(args.velocity, 0.5)       # 计划 P4 先试 0.5 m/s
        self.assertEqual(args.duration, 5.0)       # 计划要求跑 5 秒
        self.assertEqual(args.policy, "amp")

    def test_rejects_out_of_range_inputs(self):
        for argv in (["--velocity", "0"], ["--velocity", "-1"], ["--velocity", "3"],
                     ["--dt", "0.02"], ["--rope-length", "0"], ["--stiffness", "0"],
                     ["--damping", "-1"], ["--slack", "-0.1"], ["--settle-time", "0"],
                     ["--wheel-damping", "0.5"], ["--spawn-height", "-1"]):
            with self.assertRaises(SystemExit, msg=str(argv)):
                tow_drag.parse_args(argv)

    def test_accepts_zero_wheel_damping_and_zero_slack(self):
        args = tow_drag.parse_args(["--wheel-damping", "0", "--slack", "0"])
        self.assertEqual(args.wheel_damping, 0.0)
        self.assertEqual(args.slack, 0.0)


class TowRecorderTests(unittest.TestCase):
    def _row(self, time_s=0.005, **changes):
        row = {"time_s": time_s, "user_cmd_mps": 0.5, "ref_cmd_mps": 0.5,
               "robot_vx_mps": 0.5, "load_vx_mps": 0.5, "rope_tension_n": 10.0,
               "rope_distance_m": 1.1, "robot_x_m": 0.1, "load_x_m": -1.0,
               "robot_z_m": 0.30, "load_z_m": 0.15,
               "body_pitch_rad": 0.01, "body_pitch_rate_radps": 0.0}
        row.update(changes)
        return row

    def _run_directory(self, tmp):
        """按入口的方式独占创建运行目录（recorder 不再自己建目录）。"""
        directory = Path(tmp) / "run"
        directory.mkdir(parents=True, exist_ok=False)
        return directory

    def test_writes_header_and_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = self._run_directory(tmp)
            with recording.TowRecorder(directory, {"policy": "amp"}) as logger:
                logger.append(self._row(0.005))
                logger.append(self._row(0.010))
            with (directory / "tow.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(list(rows[0]), list(recording.TOW_FIELDS))

    def test_rejects_non_finite_and_unsorted_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            with recording.TowRecorder(self._run_directory(tmp), {}) as logger:
                logger.append(self._row(0.005))
                with self.assertRaises(ValueError):
                    logger.append(self._row(0.005))                       # 时间不递增
                with self.assertRaises(ValueError):
                    logger.append(self._row(0.010, rope_tension_n=float("nan")))
                with self.assertRaises(ValueError):
                    logger.append({**self._row(0.010), "extra": 1.0})     # 字段集不符

    def test_accepts_an_already_created_run_directory(self):
        """入口先用 mkdir(exist_ok=False) 独占建目录，recorder 必须能直接写进去。

        第一版 recorder 自己又建一次目录，实跑时报 FileExistsError（自己撞自己）；
        这条测试把「目录由入口创建」的集成契约固定下来。
        """
        with tempfile.TemporaryDirectory() as tmp:
            directory = self._run_directory(tmp)          # 模拟入口已建目录
            recording.TowRecorder(directory, {"policy": "amp"}).close()
            self.assertTrue((directory / "tow.csv").is_file())
            self.assertTrue((directory / "config.json").is_file())

    def test_requires_an_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                recording.TowRecorder(Path(tmp) / "missing", {})

    def test_refuses_to_overwrite_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = self._run_directory(tmp)
            recording.TowRecorder(directory, {}).close()
            with self.assertRaises(FileExistsError):
                recording.TowRecorder(directory, {})
            (directory / "tow.csv").unlink()
            with self.assertRaises(FileExistsError):          # config.json 仍在
                recording.TowRecorder(directory, {})


class InterfaceContractTests(unittest.TestCase):
    """施力/步进顺序与「不改 locomotion」「失败要带码退出」的源码契约。"""

    def setUp(self):
        self.source = TOW_DRAG.read_text(encoding="utf-8")
        self.tree = ast.parse(self.source)

    def test_rope_wrench_uses_position_application_point(self):
        self.assertIn("set_external_force_and_torque", self.source)
        self.assertIn("positions=", self.source)
        self.assertIn("quat_apply_inverse", self.source)

    def test_forces_are_written_before_the_step(self):
        # 文档字符串里也出现了这些名字，所以取最后一次出现（即主循环里的真实调用）
        write = self.source.rindex("scene.write_data_to_sim()")
        step = self.source.rindex("sim.step()")
        update = self.source.rindex("scene.update(dt)")
        self.assertLess(write, step, "必须先 write_data_to_sim() 再 sim.step()")
        self.assertLess(step, update, "必须 sim.step() 之后再 scene.update(dt)")
        self.assertLess(self.source.rindex("apply_rope_and_resistance(command)"), write,
                        "必须先把力写进缓冲再 write_data_to_sim()")

    def test_failure_path_exits_with_a_code_and_skips_app_close(self):
        """CART-02 的教训：Kit 关停会吞掉异常与退出码，失败路径必须显式 os._exit。"""
        failure = self.source.index("except (Exception, KeyboardInterrupt) as exc:")
        tail = self.source[failure:]
        self.assertIn("os._exit(2)", tail)
        # 只禁止「真的调用」application.close()，注释里提到它不算
        self.assertIsNone(re.search(r"^\s*application\.close\(\)", tail, re.MULTILINE),
                          "失败路径不得调用 application.close()，否则退出码会被吞掉")

    def test_does_not_modify_locomotion_configuration(self):
        """只允许改 prim_path 与初始位姿，不得动增益/动作缩放/观测/奖励。

        用 AST 判断而不是字符串匹配：注释或文档字符串里提到这些名字不应触发失败。
        """
        scene_source = SCENE_CFG.read_text(encoding="utf-8")
        self.assertIn("IMGO2_CFG.replace(prim_path=", scene_source)
        touched = set()
        for node in ast.walk(ast.parse(scene_source)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                touched.add(node.func.id)
            if isinstance(node, ast.Attribute):
                touched.add(node.attr)
        forbidden = {"DCMotorCfg", "action_scale", "observations", "rewards",
                     "ImplicitActuatorCfg", "actuators"}
        self.assertEqual(touched & forbidden, set(),
                         f"场景配置不应触碰 locomotion 的 {sorted(touched & forbidden)}")
        for imported in ("velocity_env_cfg", "amp_env_cfg"):
            self.assertNotIn(imported, self.source,
                             f"拖曳入口不应导入/修改 {imported}")

    def test_policy_is_loaded_through_the_contract_adapter(self):
        self.assertIn("FrozenLowLevelPolicy", self.source)
        self.assertIn("get_policy(", self.source)
        # 控制抽帧必须与物理 dt 对齐（否则策略频率与契约不符）
        self.assertIn("decimation", self.source)
        self.assertIn("control_dt / dt", self.source)

    def test_joint_order_is_reconciled_by_permutation_not_compared_directly(self):
        """PhysX 顺序 ≠ 策略顺序，所以只能比集合 + 建置换，不能直接比顺序。"""
        self.assertIn("robot.joint_names", self.source)
        self.assertIn("asset_permutation", self.source)
        self.assertIn("policy_to_asset", self.source)
        self.assertIn("asset_to_policy", self.source)
        # 关节量进观测前必须重排、动作下发前必须换回资产顺序
        self.assertIn("joint_pos[:, policy_to_asset]", self.source)
        self.assertIn("joint_vel[:, policy_to_asset]", self.source)
        self.assertIn("joint_targets[:, asset_to_policy]", self.source)
        # 不允许退回「直接比较关节名列表」的写法
        self.assertNotIn("robot.joint_names) != list(policy_cfg.joint_names", self.source)

    def test_verdict_logic_lives_in_the_stdlib_tool(self):
        """判读必须放在 scripts/tools 的标准库工具里，否则新判据无法用真实轨迹离线复算。"""
        self.assertIn("from summarize_tow import summarize_tow", self.source)
        tool = RL / "scripts/tools/summarize_tow.py"
        self.assertTrue(tool.is_file(), tool)
        tool_source = tool.read_text(encoding="utf-8")
        for imported in ("torch", "isaaclab", "numpy"):
            self.assertNotIn(f"import {imported}", tool_source,
                             f"离线判读工具不得依赖 {imported}")
        # 入口只负责采样，不再自己重算判据
        self.assertNotIn("failures.append(", self.source)

    def test_run_directory_is_created_once_by_the_entry_point(self):
        """目录归属：入口用 mkdir(exist_ok=False) 独占创建，recorder 只负责写入。

        第一版两边都建目录，实跑报 FileExistsError（自己撞自己）。这条把职责固定住。
        """
        self.assertIn("output.mkdir(parents=True, exist_ok=False)", self.source)
        self.assertIn("recorder = TowRecorder(output, config)", self.source)
        recorder_source = (RL / "source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/utils/recording.py"
                           ).read_text(encoding="utf-8")
        calls = []
        for node in ast.walk(ast.parse(recorder_source)):
            if isinstance(node, ast.ClassDef) and node.name == "TowRecorder":
                for inner in ast.walk(node):
                    if (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute)
                            and inner.func.attr in ("mkdir", "makedirs")):
                        calls.append(inner.func.attr)
        self.assertEqual(calls, [], "TowRecorder 不应自己创建目录（由入口独占创建）")

    def test_initial_pose_is_set_before_the_scene_is_built(self):
        """场景构造时就会按 init_state 摆资产，之后再改配置不会生效（--spawn-height 会失效）。"""
        assign = self.source.index("scene_cfg.robot.init_state.pos")
        build = self.source.index("scene = InteractiveScene(scene_cfg)")
        self.assertLess(assign, build,
                        "robot 的初始位姿必须在 InteractiveScene(...) 之前写进配置")
        self.assertIn("scene_cfg.robot.init_state.joint_pos", self.source)


if __name__ == "__main__":
    unittest.main()
