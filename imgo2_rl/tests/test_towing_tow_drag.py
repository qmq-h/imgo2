"""P4 tow-drag entry-point tests; standard Python, no simulator.

拖曳入口只在 `main()` 内部导入 Isaac Lab，所以模块级可以离线加载。这里锁住三件容易
静默出错的事：① 初始布局算术必须与场景配置的挂点常量、`cart.urdf` 的实际挂点一致
（差一点初始绳张力就不对）；② 记录器的严格校验；③ 施力/步进顺序与「不改 locomotion」
「失败时显式带码退出」（CART-02 的教训）这些接口契约。
"""

import ast
import contextlib
import csv
import io
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

    def test_gap_solver_is_relative_to_the_robot_position(self):
        """拖曳段开始时按机器人**当前位置**重摆小车，间距仍须等于 L0 − slack。

        站定阶段机器人会向前窜动；重摆必须相对它当时的实际位置解算，否则间距会带着
        窜动量一起偏。"""
        offset = self.constants["ROBOT_ATTACHMENT_OFFSET_M"]
        cart_offset = self.cart_model["attachment_position_m"]
        for robot_x in (0.0, 0.08, -0.05):
            cart_x = tow_drag.cart_x_for_attachment_gap(
                1.0, 0.2, robot_x=robot_x, robot_z=0.30,
                cart_height=self.cart_model["resting_height_m"],
                robot_offset=offset, cart_offset=cart_offset)
            robot_point = (robot_x + offset[0], offset[1], 0.30 + offset[2])
            cart_point = (cart_x + cart_offset[0], cart_offset[1],
                          self.cart_model["resting_height_m"] + cart_offset[2])
            self.assertAlmostEqual(math.dist(robot_point, cart_point), 1.0 - 0.2, places=9,
                                   msg=f"robot_x={robot_x}")
            self.assertLess(cart_x, robot_x, "小车必须在机器人后方")

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

    def test_stop_at_must_be_inside_the_command_window(self):
        args = tow_drag.parse_args(["--duration", "10", "--stop-at", "5"])
        self.assertEqual(args.stop_at, 5.0)
        self.assertIsNone(tow_drag.parse_args([]).stop_at)
        for argv in (["--duration", "5", "--stop-at", "5"],
                     ["--duration", "5", "--stop-at", "6"],
                     ["--stop-at", "0"], ["--stop-at", "-1"]):
            with self.assertRaises(SystemExit, msg=str(argv)):
                tow_drag.parse_args(argv)

    def test_accepts_zero_wheel_damping_and_zero_slack(self):
        args = tow_drag.parse_args(["--wheel-damping", "0", "--slack", "0"])
        self.assertEqual(args.wheel_damping, [0.0])
        self.assertEqual(args.slack, 0.0)

    def test_sweep_parameters_accept_multiple_values(self):
        """`--cart-mass` / `--wheel-damping` 可给多个，做笛卡尔积在**同一进程**内跑。"""
        args = tow_drag.parse_args(["--cart-mass", "5", "10", "20",
                                    "--wheel-damping", "0.016", "0.032"])
        self.assertEqual(args.cart_mass, [5.0, 10.0, 20.0])
        self.assertEqual(args.wheel_damping, [0.016, 0.032])
        self.assertEqual(len(tow_drag.sweep_cases(args.cart_mass, args.wheel_damping)), 6)
        # 单词形式仍然可用（旧命令不变）
        single = tow_drag.parse_args(["--cart-mass", "15", "--wheel-damping", "0.032"])
        self.assertEqual(single.cart_mass, [15.0])
        self.assertEqual(single.wheel_damping, [0.032])
        # 缺省：名义质量 × 0.016
        default = tow_drag.parse_args([])
        self.assertEqual(default.cart_mass, [None])
        self.assertEqual(default.wheel_damping, [0.016])

    def test_case_list_is_a_cartesian_product_in_stable_order(self):
        """三维修扫：绳索模型 × 质量 × 轮阻，顺序固定（同一进程内逐 case 跑）。"""
        self.assertEqual(tow_drag.sweep_cases([5.0, 10.0], [0.016]),
                         [(5.0, 0.016, "compliant"), (10.0, 0.016, "compliant")])
        self.assertEqual(tow_drag.sweep_cases([5.0], [0.016], ["compliant", "inextensible"]),
                         [(5.0, 0.016, "compliant"), (5.0, 0.016, "inextensible")])
        for bad in (([], [0.016]), ([5.0], []), ([5.0], [0.016], [])):
            with self.assertRaises(ValueError):
                tow_drag.sweep_cases(*bad)

    def test_case_label_names_the_rope_model_mass_and_damping(self):
        """模型名必须进目录名：两套模型共用同一串 case 编号，否则会互相覆盖。"""
        self.assertEqual(tow_drag.case_label(0, 5.0, 0.016, 10.0),
                         "case_00_compliant_m5_b0.016")
        self.assertEqual(tow_drag.case_label(3, None, 0.032, 10.0, "inextensible"),
                         "case_03_inextensible_m10_b0.032")


class StopAtArgumentTests(unittest.TestCase):
    """`--stop-at` 必须配够 `--duration`；报错要直接给出改法。

    2026-09-21 交付命令里只写了 `--stop-at 5`（没写 `--duration`，默认也是 5），
    实跑被参数校验拦下——报错只说「必须小于」，用户不知道要补 `--duration`。
    """

    def test_stop_at_requires_a_longer_duration(self):
        # `parser.error` 把消息写 stderr 再以码 2 退出，所以断言要抓 stderr 而不是异常文本
        buffer = io.StringIO()
        with contextlib.redirect_stderr(buffer), self.assertRaises(SystemExit) as caught:
            tow_drag.parse_args(["--stop-at", "5"])
        self.assertEqual(caught.exception.code, 2)
        message = buffer.getvalue()
        self.assertIn("--duration", message)          # 必须点出缺哪个参数
        self.assertIn("--duration 10", message)       # 并且给出可直接照抄的改法

    def test_stop_at_with_duration_plans_the_coast_phase(self):
        args = tow_drag.parse_args(["--duration", "10", "--stop-at", "5"])
        schedule = tow_drag.make_schedule(settle_steps=200, duration=args.duration,
                                         stop_at=args.stop_at, dt=0.005)
        self.assertEqual(schedule.tow_steps, 1000)     # 5 s 拖曳
        self.assertEqual(schedule.coast_steps, 1000)   # 5 s 滑行


class PlanTimeOrderingTests(unittest.TestCase):
    """`main()` 里的局部量不得「先用后赋值」。

    这个错误已经踩过两次：`stop_steps`（config 字典里提前引用）与 `scale`（预判里提前
    引用）——两者都只在**跑仿真时**才执行，所以离线测试全过、一跑就 `UnboundLocalError`，
    而且注释里出现同一个词会让朴素的字符串检查假阳性。这里用 AST 按源码顺序判断：
    对每个被监视的局部量，第一次 **Store** 必须不晚于第一次 **Load**。
    """

    WATCHED = ("scales", "cases", "predictions", "schedule", "decimation",
               "settle_steps", "command_steps", "nominal_masses", "nominal_inertias")

    def test_no_local_is_used_before_assignment_in_main(self):
        tree = ast.parse(TOW_DRAG.read_text(encoding="utf-8"))
        main = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "main")
        events = {}
        for node in ast.walk(main):
            if isinstance(node, ast.Name) and node.id in self.WATCHED:
                kind = "store" if isinstance(node.ctx, ast.Store) else "load"
                events.setdefault(node.id, []).append((node.lineno, node.col_offset, kind))
        for name, items in events.items():
            items.sort()
            kinds = [kind for _, _, kind in items]
            first_store = kinds.index("store") if "store" in kinds else None
            first_load = kinds.index("load") if "load" in kinds else None
            self.assertIsNotNone(first_store, f"{name} 在 main() 里从未被赋值")
            if first_load is not None:
                self.assertLess(first_store, first_load,
                                f"{name} 在赋值前就被引用（先用后赋值）")


class ScheduleTests(unittest.TestCase):
    """阶段划分是纯算术，必须离线可测。

    这段算术原本内联在 `main()` 里，于是 `stop_steps` 在 config 字典里被提前引用、
    赋值却在后面，实跑直接 `UnboundLocalError`——只有跑仿真才会暴露。抽成纯函数后
    这类顺序错误能在离线测试里挡住。
    """

    def test_three_phases(self):
        sched = tow_drag.make_schedule(settle_steps=200, duration=10.0, stop_at=5.0, dt=0.005)
        self.assertEqual((sched.station_steps, sched.tow_steps, sched.coast_steps), (200, 1000, 1000))
        self.assertEqual(sched.total_steps, 2200)
        self.assertAlmostEqual(sched.tow_phase_s, 5.0, places=9)
        self.assertAlmostEqual(sched.coast_phase_s, 5.0, places=9)

    def test_without_stop_at_there_is_no_coast(self):
        sched = tow_drag.make_schedule(settle_steps=200, duration=5.0, stop_at=None, dt=0.005)
        self.assertEqual((sched.tow_steps, sched.coast_steps), (1000, 0))
        self.assertEqual(sched.total_steps, 1200)

    def test_phase_boundaries(self):
        sched = tow_drag.make_schedule(settle_steps=200, duration=10.0, stop_at=5.0, dt=0.005)
        self.assertEqual(sched.phase_of(0), "station")
        self.assertEqual(sched.phase_of(199), "station")
        self.assertEqual(sched.phase_of(200), "tow")          # station→tow 边界
        self.assertEqual(sched.phase_of(1199), "tow")
        self.assertEqual(sched.phase_of(1200), "coast")       # tow→coast 边界（阶跃点）
        self.assertEqual(sched.phase_of(sched.total_steps - 1), "coast")
        self.assertEqual(sum(1 for i in range(sched.total_steps) if sched.phase_of(i) == "coast"),
                         sched.coast_steps)

    def test_rejects_degenerate_schedules(self):
        with self.assertRaises(ValueError):
            tow_drag.make_schedule(settle_steps=0, duration=0.0, stop_at=None, dt=0.005)
        with self.assertRaises(ValueError):
            tow_drag.make_schedule(settle_steps=0, duration=5.0, stop_at=0.0001, dt=0.005)
        with self.assertRaises(ValueError):
            tow_drag.make_schedule(settle_steps=-1, duration=5.0, stop_at=None, dt=0.005)


class CoastPredictionTests(unittest.TestCase):
    """停车后小车滑行距离的解析预测（决定 L0 / 速度 / 轮阻怎么配）。"""

    # P2 验收实跑（run 16885db6，2026-09-20）的实测停止距离
    P2_MEASURED = {
        (0.5, 0.008): 1.033949, (0.5, 0.016): 0.515817, (0.5, 0.032): 0.256703,
        (1.0, 0.008): 2.111118, (1.0, 0.016): 1.053125, (1.0, 0.032): 0.524179,
    }
    CART = dict(cart_mass_kg=10.0, wheel_inertia_kgm2=0.00128, wheel_radius_m=0.08)

    def _predict(self, v, b):
        return tow_drag.predicted_coast_distance(v, b, **self.CART)

    def test_matches_the_p2_acceptance_measurements(self):
        """解析式必须重现 P2 的 6 个实测点（否则不能拿它做参数选择）。"""
        for (speed, damping), measured in self.P2_MEASURED.items():
            got = self._predict(speed, damping)
            self.assertLess(abs(got - measured) / measured, 0.015,
                            f"v={speed} b={damping}: 预测 {got:.4f} 对实测 {measured:.4f}")

    def test_default_configuration_keeps_a_margin(self):
        """默认组合（L0=0.8、slack=0.40、v=0.5、b=0.016）必须留出余量。"""
        args = tow_drag.parse_args([])
        self.assertEqual(args.rope_length, 0.8)
        self.assertEqual(args.slack, 0.40)
        self.assertEqual(args.velocity, 0.5)
        gap_at_stop = args.rope_length - self._predict(args.velocity, args.wheel_damping[0])
        self.assertGreater(gap_at_stop, 0.2, f"停车后最小间距只有 {gap_at_stop:.3f} m")

    def test_rope_length_must_be_chosen_with_speed_and_damping(self):
        """记录这个耦合：v=1.0、b=0.016 时滑行 1.06 m，L0=0.8/1.0 都会追到机器人。

        所以「缩短绳子」不能在 1.0 m/s 那一步沿用，要么加长 L0、要么加大轮阻
        （b=0.032 时滑行 0.53 m）。
        """
        self.assertLess(0.8 - self._predict(1.0, 0.016), 0.0)
        self.assertLess(1.0 - self._predict(1.0, 0.016), 0.0)
        self.assertGreater(0.8 - self._predict(1.0, 0.032), 0.2)

    def test_mass_scale_factor(self):
        self.assertEqual(tow_drag.mass_scale_factor(None, 10.0), 1.0)
        self.assertEqual(tow_drag.mass_scale_factor(20.0, 10.0), 2.0)
        self.assertEqual(tow_drag.mass_scale_factor(5.0, 10.0), 0.5)
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                tow_drag.mass_scale_factor(bad, 10.0)
        with self.assertRaises(ValueError):
            tow_drag.mass_scale_factor(None, 0.0)

    def test_coast_distance_scales_with_mass(self):
        """质量与惯量同比例缩放 ⇒ m_eff 同比例 ⇒ 滑行距离也同比例。"""
        base = self._predict(0.5, 0.016)
        doubled = tow_drag.predicted_coast_distance(
            0.5, 0.016, cart_mass_kg=20.0, wheel_inertia_kgm2=0.00128 * 2,
            wheel_radius_m=0.08)
        self.assertAlmostEqual(doubled, base * 2, places=9)

    def test_heavy_cart_needs_longer_rope_or_more_damping(self):
        """重量扫描的耦合：20 kg 时默认 L0=0.8/b=0.016 会追到机器人。

        这决定扫描命令怎么写：要么加 --wheel-damping 0.032，要么加长 --rope-length。
        """
        def margin(mass, damping, rope=0.8):
            scale = mass / 10.0
            coast = tow_drag.predicted_coast_distance(
                0.5, damping, cart_mass_kg=10.0 * scale, wheel_inertia_kgm2=0.00128 * scale,
                wheel_radius_m=0.08)
            return rope - coast
        self.assertGreater(margin(5.0, 0.016), 0.5)
        self.assertGreater(margin(10.0, 0.016), 0.2)
        self.assertLess(margin(20.0, 0.016), 0.0)          # 会追尾
        self.assertGreater(margin(25.0, 0.032), 0.1)       # 加大轮阻后安全
        self.assertGreater(margin(25.0, 0.016, rope=1.5), 0.1)   # 或加长绳

    def test_below_stop_speed_predicts_zero(self):
        self.assertEqual(self._predict(0.02, 0.016), 0.0)

    def test_rejects_nonpositive_inputs(self):
        with self.assertRaises(ValueError):                      # 轮阻为 0 ⇒ 永远不停
            tow_drag.predicted_coast_distance(0.5, 0.0, **self.CART)
        for bad in ({"cart_mass_kg": 0.0}, {"wheel_radius_m": -1.0}, {"wheel_inertia_kgm2": 0.0}):
            with self.assertRaises(ValueError):
                tow_drag.predicted_coast_distance(0.5, 0.016, **{**self.CART, **bad})


class TowRecorderTests(unittest.TestCase):
    def _row(self, time_s=0.005, **changes):
        # 字段集从 TOW_FIELDS 派生，不手工罗列：手工列表在加列时会变成假失败，
        # 而这条测试要验的是「字段集必须严格一致」这个契约本身。
        row = {"phase": "tow", "time_s": time_s}
        row.update({name: 0.0 for name in recording.TOW_NUMERIC_FIELDS if name != "time_s"})
        row.update({"user_cmd_mps": 0.5, "ref_cmd_mps": 0.5, "robot_vx_mps": 0.5,
                    "load_vx_mps": 0.5, "rope_tension_n": 10.0, "rope_distance_m": 1.1,
                    "robot_x_m": 0.1, "load_x_m": -1.0, "robot_z_m": 0.30, "load_z_m": 0.15,
                    "body_pitch_rad": 0.01, "robot_quat_w": 1.0, "load_quat_w": 1.0})
        for leg in recording.LEGS:
            row[f"wheel_{leg}_omega_radps"] = 6.0
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
        self.assertLess(self.source.rindex("apply_rope_and_resistance(command, damping)"), write,
                        "必须先把力写进缓冲再 write_data_to_sim()")

    def test_success_path_also_exits_explicitly(self):
        """成功路径也必须显式退出。

        实测（2026-09-21 friction 扫描）：跑完后 summary.json/sweep.json 都写好了、进程却不返回，
        shell 的 for 循环进不到下一个值。所以成功路径也要 os._exit，不能依赖 application.close()。
        """
        tail = self.source[self.source.index('print(f"[SWEEP]'):]
        self.assertIn("os._exit(0 if all_valid else 2)", tail)
        # 只禁止**真的调用**（注释里提到它不算）——朴素的字符串匹配已被这类假阳性咬过三次
        self.assertIsNone(re.search(r"^\s*application\.close\(\)", tail.split("except ")[0],
                                    re.MULTILINE),
                          "成功路径不得依赖 application.close() 退出")

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

    def test_entry_fills_exactly_the_recorded_columns(self):
        """入口每步给出的字段集必须与 `TowRecorder` 的列集完全一致。

        这条契约以前只在实跑时才暴露（`TowRecorder.append` 会拒绝字段集不符），也就是要烧掉
        一次 GPU 运行才知道。这里用 AST 把入口那个字典的**键**离线算出来核对，
        新增列却忘了填的情况就能在提交前拦住。
        """
        tree = ast.parse(self.source)
        target = None
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "append" and node.args
                    and isinstance(node.args[0], ast.Dict)):
                target = node.args[0]
        self.assertIsNotNone(target, "没找到 recorder.append({...}) 的字典")
        assigned = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.DictComp):
                for name in node.targets:
                    if isinstance(name, ast.Name):
                        assigned[name.id] = node.value
        self.assertEqual(sorted(self._dict_keys(target, assigned)),
                         sorted(recording.TOW_FIELDS),
                         "入口写的字段与 TowRecorder 的列不一致（多写/漏写都会让实跑抛异常）")

    @staticmethod
    def _dict_keys(node, assigned):
        """求出一个 dict 节点的字符串键集合，支持本入口用到的三种形态：

        * 字面量键；`**some_dict`（引用局部变量）；
        * `{f"robot_jp_{i:02d}": ... for i in range(12)}`；
        * `{f"wheel_{leg}_omega_radps": ... for leg, jid in zip(("fl","fr","rl","rr"), ...)}`。
        """
        if isinstance(node, ast.Name):
            return InterfaceContractTests._dict_keys(assigned[node.id], assigned)
        if isinstance(node, ast.Dict):
            keys = []
            for key, value in zip(node.keys, node.values):
                if key is not None:
                    keys.append(key.value)
                elif (isinstance(value, ast.Call) and getattr(value.func, "id", "") == "joint_position_fields"):
                    # 直接调用**生产函数**取键（关节列名只有 recording 一处来源）
                    keys.extend(recording.joint_position_fields(
                        [0.0] * len(recording.ROBOT_JOINT_POSITION_FIELDS)))
                else:
                    keys.extend(InterfaceContractTests._dict_keys(value, assigned))
            return keys
        if isinstance(node, ast.DictComp):
            loop = node.generators[0]
            source = loop.iter
            if isinstance(source, ast.Call) and getattr(source.func, "id", "") == "range":
                values = list(range(source.args[0].value))
            elif isinstance(source, ast.Call) and getattr(source.func, "id", "") == "zip":
                values = [element.value for element in source.args[0].elts]
            else:
                raise AssertionError(f"不支持的推导式来源：{ast.dump(loop.iter)[:80]}")
            text = ast.unparse(node.key)
            # `for leg, jid in zip(...)` 这种元组目标是允许的：f-string 只用到第一个（腿名）
            name = loop.target.elts[0].id if isinstance(loop.target, ast.Tuple) else loop.target.id
            return [eval(text, {name: value}) for value in values]   # noqa: S307 - 只喂源码里的 f-string
        raise AssertionError(f"不支持的字段来源：{type(node).__name__}")

    def test_both_rope_models_are_wired_in(self):
        """规格要求两套模型都实现、可切换；入口必须同时支持并按 case 记录用的是哪一套。"""
        self.assertIn("make_rope_model(", self.source)
        self.assertIn('choices=("compliant", "inextensible")', self.source)
        self.assertIn('"model": rope_name', self.source)
        self.assertIn("world_inverse_inertia", self.source)

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

    def test_schedule_is_built_before_the_config_dict(self):
        """回归：config 字典引用了 schedule，若赋值在后面就会 UnboundLocalError。

        上一版正是这样：`stop_steps` 在 config 里被提前引用、赋值在其后，实跑直接崩。
        """
        built = self.source.index("schedule = make_schedule(")
        config = self.source.index("def case_config(")
        self.assertLess(built, config, "schedule 必须在 case_config（消费它的 config 字典）之前建好")
        self.assertIn("schedule.tow_phase_s", self.source)
        # 不允许再出现内联的局部阶段划分（phase_of 只应作为 PhaseSchedule 的方法存在一次）
        self.assertEqual(self.source.count("def phase_of"), 1)
        self.assertNotIn("stop_steps =", self.source)
        self.assertNotIn("total_steps = settle_steps", self.source)
        self.assertIn("schedule.phase_of(step)", self.source)

    def test_sweep_runs_cases_in_one_process(self):
        """多 case 必须同进程顺序跑（省掉每个组合重启 Isaac Sim），且每 case 前显式复位。

        `Articulation.reset()` 只清执行器与外力缓冲、**不写位姿**（isaaclab articulation.py:172），
        所以复位必须自己写 root pose/velocity 与关节状态。
        """
        self.assertIn("cases = sweep_cases(", self.source)
        self.assertIn("def reset_case(", self.source)
        self.assertIn("write_root_pose_to_sim", self.source)
        self.assertIn("write_root_velocity_to_sim", self.source)
        self.assertIn("write_joint_state_to_sim", self.source)
        self.assertIn("root[:, 7:] = 0.0", self.source)          # 速度清零
        self.assertIn("for case_index, ((mass_target, damping, rope_name)", self.source)
        # 每个 case 按名字重建模型：同一次扫描里可以混用两套绳索模型
        self.assertIn("rope_model = build_rope_model(rope_name)", self.source)
        self.assertIn('"sweep"', self.source)
        # 逐 case 的产物目录与汇总
        self.assertIn("case_dir.mkdir(parents=True, exist_ok=False)", self.source)
        self.assertIn('json_file(case_dir / "summary.json", summary)', self.source)
        self.assertIn('json_file(output / "sweep.json", sweep)', self.source)

    def test_no_teleport_inside_the_step_loop(self):
        """位姿重写只允许出现在 `reset_case()`（逐 case 的 episode reset），不能在步进循环里。

        曾用 reset_cart_for_tow() 在**拖曳段开始时**重摆小车，实跑证明有害：它按机器人当前
        位置摆，机器人窜 0.097 m 就把小车往前挪 0.081 m 并注入 −0.039 m/s。逐 case 的复位
        不一样——那是每个 case 的初始条件（`Articulation.reset()` 本身不写位姿，见
        isaaclab articulation.py:172），与 P1/P2 每个 case 前重写状态同一约定。
        """
        reset = self.source.index("def reset_case(")
        loop = self.source.index("for step in range(schedule.total_steps):")
        self.assertLess(reset, loop, "reset_case 必须定义在步进循环之前")
        body = self.source[reset:loop]
        loop_body = self.source[loop:self.source.index("recorder.close()", loop)]
        for call in ("write_root_pose_to_sim", "write_root_velocity_to_sim", "write_joint_state_to_sim"):
            self.assertIn(call, body, f"{call} 应出现在 reset_case 里")
            self.assertNotIn(call, loop_body, f"{call} 不应出现在步进循环里")
        self.assertNotIn("reset_cart_for_tow", self.source)

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
        self.assertIn("recorder = TowRecorder(case_dir,", self.source)
        self.assertIn("case_config(mass_target, damping, rope_name,", self.source)
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
