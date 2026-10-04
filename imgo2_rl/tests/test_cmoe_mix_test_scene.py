"""`Imgo2-basemove-rough-cmoe-mix-test`（`mix`-only 受控测试场景）的离线回归。

2026-10-04（用户）：新增一个**继承 play 任务**的测试场景 —— 场景里只有 `mix` 一种地形、速度指令
只给前进（默认恒定 **1.0 m/s**）、横向用 **PD 外环**、heading 保持 0；用来在受控条件下评估策略
通过复合障碍（`mix`）的能力，把横向/航向扰动从评测里剔除。**不是新策略任务** ⇒ 动作空间与观测
契约一字未改，既有 CMoE checkpoint 可直接加载。

写法沿用仓库既有做法（`test_masked_terrain_terms.py` / `test_track_geometry.py`）：
`CMoE_env_cfg.py` 与 `mdp/mix_test_command.py` 顶层都 `import isaaclab`（缺 `omni.log`）无法整模块
import，所以 ① 纯 PD 数学从**只依赖 torch** 的 `mdp/mix_test_pd.py` 直接按文件加载；
② 命令项类用 AST 抽出真实源码、在桩环境里 exec；③ 配置级与注册级断言读**真实源码**（AST）。

覆盖：PD 纯函数（零点/符号/四象限/夹取/默认增益）、命令项在桩 env 上的行为（恒定 vx、PD 写
`vel_command_b`、世界系镜像 `vel_command_w`、站立环境归零）、配置级（只 mix、1 列 20 行、
速度与 heading 范围、不重采样、20 环境、命令项用新类）、mix 地形参数与 `cmoe_terrains.py`
默认值**逐一相等**（防漂移）、任务注册、以及"观测/动作契约未改"。
"""

from __future__ import annotations

import ast
import importlib.util
import math
import re
import sys
import types
import unittest
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity"
MDP = PKG / "mdp"
BASE_MOVE = PKG / "base_move"
CMOE_CFG = BASE_MOVE / "CMoE_env_cfg.py"
CMOE_TERRAINS = BASE_MOVE / "cmoe_terrains.py"
TASK_INIT = BASE_MOVE / "__init__.py"
MDP_INIT = MDP / "__init__.py"
MIX_TEST_PD = MDP / "mix_test_pd.py"
MIX_TEST_COMMAND = MDP / "mix_test_command.py"

TASK_ID = "Imgo2-basemove-rough-cmoe-mix-test"
CFG_CLASS = "Imgo2CMoEMixTestEnvCfg"
PLAY_CLASS = "Imgo2CMoERoughPlayEnvCfg"

try:
    import torch
except ModuleNotFoundError as error:  # pragma: no cover - 本容器有 torch
    torch = None
    IMPORT_ERROR: object = error
else:
    IMPORT_ERROR = None


# --------------------------------------------------------------------------- AST 小工具
def _class_def(path: Path, name: str) -> ast.ClassDef | None:
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    return next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name), None)


def _post_init(path: Path, name: str) -> ast.FunctionDef | None:
    cls = _class_def(path, name)
    if cls is None:
        return None
    return next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__post_init__"), None)


def _assignments(fn: ast.AST) -> dict[str, ast.AST]:
    """`{target 源码: value 节点}`（单目标 `Assign`；同名后者覆盖＝Python 实际语义）。"""
    out: dict[str, ast.AST] = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            out[ast.unparse(node.targets[0])] = node.value
    return out


def _literal(node: ast.AST):
    try:
        return ast.literal_eval(node)
    except Exception:
        return ast.unparse(node)


def _kwargs(call: ast.Call) -> dict[str, ast.AST]:
    return {kw.arg: kw.value for kw in call.keywords}


def _dict_items(node: ast.AST) -> dict[str, ast.AST]:
    """把 `{...}` 字面量的键（字符串常量）映射到 value 节点。"""
    if not isinstance(node, ast.Dict):
        return {}
    return {ast.literal_eval(k): v for k, v in zip(node.keys, node.values)}


# `gym.register(...)` 的 kwargs 值里有 `f"{__name__}.x:Y"` 与 `f"{agents.__name__}.x:Y"`，
# 用一个最小命名空间把它们**求值成真实字符串**（比字符串匹配更严）。
_REG_NS = {
    "__name__": "imgo2_rl.tasks.manager_based.locomotion.velocity.base_move",
    "agents": types.SimpleNamespace(
        __name__="imgo2_rl.tasks.manager_based.locomotion.velocity.base_move.agents"
    ),
}


def _reg_value(node: ast.AST):
    try:
        return eval(compile(ast.Expression(node), "<gym.register>", "eval"), {"__builtins__": {}}, dict(_REG_NS))
    except Exception:
        return ast.unparse(node)


def _clear_calls(fn: ast.AST) -> list[str]:
    """找出 `....sub_terrains.clear()` 这类调用（返回被清空的目标源码）。"""
    found = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = ast.unparse(node.value.func)
            if func.endswith("sub_terrains.clear"):
                found.append(func)
    return found


def _load_pure_module():
    """按文件路径加载 `mix_test_pd.py`（只依赖 torch，不会拉起 isaaclab）。"""
    spec = importlib.util.spec_from_file_location("mix_test_pd_under_test", MIX_TEST_PD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# =========================================================================== ① PD 纯函数
@unittest.skipIf(torch is None, f"PyTorch unavailable: {IMPORT_ERROR}")
class TestLateralHeadingPD(unittest.TestCase):
    """PD 公式：`vy_cmd = clip(kp_y·(y_des − y) + kd_y·(0 − vy), ±vy_max)`，wz 同型。"""

    @classmethod
    def setUpClass(cls):
        cls.pd = _load_pure_module()

    def _t(self, value):
        return torch.tensor([float(value)])

    def _call(self, y=0.0, vy=0.0, yaw=0.0, wz=0.0, **kwargs):
        vy_cmd, wz_cmd = self.pd.lateral_heading_pd(
            self._t(y), self._t(vy), self._t(yaw), self._t(wz), **kwargs
        )
        return float(vy_cmd[0]), float(wz_cmd[0])

    def test_default_gains_match_the_spec(self):
        """默认增益＝用户给定值（1.0/0.3/0.6、1.5/0.3/1.0），目标都是 0。"""
        self.assertEqual(self.pd.DEFAULT_KP_Y, 1.0)
        self.assertEqual(self.pd.DEFAULT_KD_Y, 0.3)
        self.assertEqual(self.pd.DEFAULT_VY_MAX, 0.6)
        self.assertEqual(self.pd.DEFAULT_KP_H, 1.5)
        self.assertEqual(self.pd.DEFAULT_KD_H, 0.3)
        self.assertEqual(self.pd.DEFAULT_WZ_MAX, 1.0)
        self.assertEqual(self.pd.DEFAULT_Y_DES, 0.0)
        self.assertEqual(self.pd.DEFAULT_YAW_DES, 0.0)

    def test_zero_state_gives_zero_commands(self):
        """`y=0, vy=0, yaw=0, wz=0` ⇒ `vy_cmd=wz_cmd=0`（无静差起点）。"""
        self.assertEqual(self._call(), (0.0, 0.0))

    def test_positive_y_gives_negative_vy(self):
        """y > 0（偏左）⇒ 负横向指令（往右回中心线）。"""
        vy_cmd, _ = self._call(y=0.3)
        self.assertAlmostEqual(vy_cmd, -0.3, places=6)
        self.assertLess(vy_cmd, 0.0)

    def test_positive_vy_gives_negative_damping(self):
        """vy > 0 ⇒ 负阻尼项（`−kd_y·vy`），即使位置误差为 0 也输出负指令。"""
        vy_cmd, wz_cmd = self._call(vy=1.0)
        self.assertAlmostEqual(vy_cmd, -0.3, places=6)
        self.assertAlmostEqual(wz_cmd, 0.0, places=6)

    def test_positive_yaw_and_wz_give_negative_wz(self):
        """yaw > 0（偏转）⇒ 负航向指令；wz > 0 ⇒ 负阻尼项。"""
        self.assertAlmostEqual(self._call(yaw=0.5)[1], -1.5 * 0.5, places=6)
        self.assertAlmostEqual(self._call(wz=1.0)[1], -0.3, places=6)

    def test_four_quadrants(self):
        """四条象限各一例：位置项与阻尼项同号时叠加、反号时相消，且符号都对。"""
        cases = (
            # (y, vy, 期望 vy_cmd, 说明)
            (+0.4, +0.5, -(1.0 * 0.4 + 0.3 * 0.5), "偏左且继续往左 ⇒ 最大负向"),
            (+0.4, -0.5, -(1.0 * 0.4 - 0.3 * 0.5), "偏左但在往右回 ⇒ 负向变小"),
            (-0.4, +0.5, -(-1.0 * 0.4 + 0.3 * 0.5), "偏右且继续往右 ⇒ 正向"),
            (-0.4, -0.5, -(-1.0 * 0.4 - 0.3 * 0.5), "偏右但在往左回 ⇒ 正向变小"),
        )
        for y, vy, expected, note in cases:
            with self.subTest(y=y, vy=vy):
                vy_cmd, wz_cmd = self._call(y=y, vy=vy)
                self.assertAlmostEqual(vy_cmd, expected, places=6, msg=note)
                self.assertAlmostEqual(wz_cmd, 0.0, places=6)

    def test_quadrant_signs_are_correct(self):
        """象限符号的产品化表述：位置项与阻尼项**同号**时修正更大，**反号**时修正更小。"""
        # y > 0 且继续往左（vy > 0）⇒ 比"正在往右回"（vy < 0）更负
        self.assertLess(self._call(y=0.4, vy=0.5)[0], self._call(y=0.4, vy=-0.5)[0])
        # y < 0 且继续往右（vy < 0）⇒ 比"正在往左回"（vy > 0）更正
        self.assertGreater(self._call(y=-0.4, vy=-0.5)[0], self._call(y=-0.4, vy=+0.5)[0])
        # yaw > 0 且继续正转（wz > 0）⇒ 比"正在负转回"（wz < 0）更负
        self.assertLess(self._call(yaw=0.4, wz=0.5)[1], self._call(yaw=0.4, wz=-0.5)[1])

    def test_outputs_are_clamped(self):
        """输出必须被 `vy_max` / `wz_max` 夹住（大误差、小上限、以及负方向都要夹）。"""
        self.assertAlmostEqual(self._call(y=100.0)[0], -0.6, places=6)
        self.assertAlmostEqual(self._call(y=-100.0)[0], +0.6, places=6)
        self.assertAlmostEqual(self._call(yaw=100.0)[1], -1.0, places=6)
        self.assertAlmostEqual(self._call(yaw=-100.0)[1], +1.0, places=6)
        # 阻尼项单独也能顶到上限
        self.assertAlmostEqual(self._call(vy=100.0)[0], -0.6, places=6)
        self.assertAlmostEqual(self._call(wz=100.0)[1], -1.0, places=6)
        # 自定义更小的上限
        self.assertAlmostEqual(self._call(y=1.0, vy_max=0.05)[0], -0.05, places=6)
        self.assertAlmostEqual(self._call(yaw=1.0, wz_max=0.05)[1], -0.05, places=6)

    def test_custom_gains_are_honoured(self):
        """增益与目标都可覆盖（`params` 走 cfg 字段 ⇒ 这里验证公式真的读参数）。"""
        vy_cmd, _ = self._call(y=0.5, kp_y=2.0, kd_y=0.0, vy_max=10.0)
        self.assertAlmostEqual(vy_cmd, -1.0, places=6)
        _, wz_cmd = self._call(yaw=0.5, kp_h=4.0, kd_h=0.0, wz_max=10.0)
        self.assertAlmostEqual(wz_cmd, -2.0, places=6)
        vy_cmd, wz_cmd = self._call(y=0.0, yaw=0.0, y_des=0.7, yaw_des=-0.3, kd_y=0.0, kd_h=0.0,
                                    vy_max=10.0, wz_max=10.0)
        self.assertAlmostEqual(vy_cmd, +0.7, places=6)
        self.assertAlmostEqual(wz_cmd, -0.45, places=6)

    def test_rotate_base_to_world(self):
        """`vel_command_w` 用的旋转：yaw=0 恒等；yaw=π/2 时机体系 +x 落到世界系 +y。"""
        vx_w, vy_w = self.pd.rotate_base_to_world(self._t(0.0), self._t(1.0), self._t(0.0))
        self.assertAlmostEqual(float(vx_w[0]), 1.0, places=6)
        self.assertAlmostEqual(float(vy_w[0]), 0.0, places=6)
        vx_w, vy_w = self.pd.rotate_base_to_world(self._t(math.pi / 2), self._t(1.0), self._t(0.0))
        self.assertAlmostEqual(float(vx_w[0]), 0.0, places=6)
        self.assertAlmostEqual(float(vy_w[0]), 1.0, places=6)


# ===================================================================== ② 命令项（桩 env）
@unittest.skipIf(torch is None, f"PyTorch unavailable: {IMPORT_ERROR}")
class TestMixTestCommandTerm(unittest.TestCase):
    """用桩 env 跑**真实源码**的 `MixTestVelocityCommand`（AST 抽类 + 桩基类）。"""

    NUM_ENVS = 2

    @classmethod
    def setUpClass(cls):
        cls.command_class = cls._load_command_class()

    @staticmethod
    def _load_command_class():
        source = MIX_TEST_COMMAND.read_text(encoding="utf-8")
        tree = ast.parse(source)
        node = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MixTestVelocityCommand"
        )
        ns = {
            "torch": torch,
            "mdp": _StubMdp,
            "Sequence": Sequence,
            "lateral_heading_pd": _load_pure_module().lateral_heading_pd,
            "rotate_base_to_world": _load_pure_module().rotate_base_to_world,
        }
        exec(compile(ast.get_source_segment(source, node), str(MIX_TEST_COMMAND), "exec"), ns)  # noqa: S102
        return ns["MixTestVelocityCommand"]

    @staticmethod
    def _cfg(**overrides):
        cfg = types.SimpleNamespace(
            kp_y=1.0, kd_y=0.3, vy_max=0.6, y_des=0.0,
            kp_h=1.5, kd_h=0.3, wz_max=1.0, yaw_des=0.0,
            rel_heading_envs=1.0, rel_standing_envs=0.0,
            ranges=types.SimpleNamespace(
                lin_vel_x=(1.0, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0.0, 0.0)
            ),
        )
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg

    def _term(self, y=(0.0, 0.0), vy=(0.0, 0.0), yaw=(0.0, 0.0), wz=(0.0, 0.0), **cfg_kwargs):
        data = types.SimpleNamespace(
            root_pos_w=torch.tensor([[0.0, val, 0.3] for val in y]),
            root_lin_vel_w=torch.tensor([[0.0, val, 0.0] for val in vy]),
            heading_w=torch.tensor(list(yaw)),
            root_ang_vel_b=torch.tensor([[0.0, 0.0, val] for val in wz]),
        )
        env = types.SimpleNamespace(
            num_envs=len(y),
            device="cpu",
            step_dt=0.02,
            scene=_StubScene(data, torch.zeros(len(y), 3)),
        )
        return self.command_class(self._cfg(**cfg_kwargs), env)

    def test_resample_samples_vx_once_and_zeroes_lateral_heading(self):
        term = self._term()
        term._resample_command(torch.tensor([0, 1]))
        self.assertTrue(torch.allclose(term.vel_command_b[:, 0], torch.ones(self.NUM_ENVS)))
        self.assertTrue(torch.allclose(term.vel_command_b[:, 1], torch.zeros(self.NUM_ENVS)))
        self.assertTrue(torch.allclose(term.vel_command_b[:, 2], torch.zeros(self.NUM_ENVS)))
        self.assertTrue(term.is_heading_env.all())
        self.assertFalse(term.is_standing_env.any())

    def test_update_command_keeps_vx_and_writes_pd_outputs(self):
        """`_update_command` 不碰第 0 列；vy/wz 由 PD 每步写（y=0.2 ⇒ vy_cmd=−0.2）。"""
        term = self._term(y=(0.2, -0.2), vy=(0.0, 0.0), yaw=(0.1, -0.1), wz=(0.0, 0.0))
        term._resample_command(torch.tensor([0, 1]))
        term._update_command()
        self.assertTrue(torch.allclose(term.vel_command_b[:, 0], torch.ones(self.NUM_ENVS)),
                        "vx 必须由 resample 决定、_update_command 不得改动")
        self.assertAlmostEqual(float(term.vel_command_b[0, 1]), -0.2, places=6)
        self.assertAlmostEqual(float(term.vel_command_b[1, 1]), +0.2, places=6)
        self.assertAlmostEqual(float(term.vel_command_b[0, 2]), -1.5 * 0.1, places=6)
        self.assertAlmostEqual(float(term.vel_command_b[1, 2]), +1.5 * 0.1, places=6)

    def test_update_command_writes_world_frame_mirror(self):
        """`vel_command_w` 是机体系指令按 yaw 旋转后的镜像（yaw=0 时相等）。"""
        term = self._term(y=(0.0, 0.0), yaw=(0.0, math.pi / 2))
        term._resample_command(torch.tensor([0, 1]))
        term._update_command()
        self.assertTrue(torch.allclose(term.vel_command_w[0], term.vel_command_b[0], atol=1e-6))
        # yaw=π/2：机体系 (vx, vy) → 世界系 (−vy, vx)
        vx_b, vy_b = float(term.vel_command_b[1, 0]), float(term.vel_command_b[1, 1])
        self.assertAlmostEqual(float(term.vel_command_w[1, 0]), -vy_b, places=5)
        self.assertAlmostEqual(float(term.vel_command_w[1, 1]), vx_b, places=5)

    def test_outputs_respect_cfg_limits(self):
        term = self._term(y=(5.0, -5.0), yaw=(5.0, -5.0))
        term._resample_command(torch.tensor([0, 1]))
        term._update_command()
        self.assertAlmostEqual(float(term.vel_command_b[0, 1]), -0.6, places=6)
        self.assertAlmostEqual(float(term.vel_command_b[1, 1]), +0.6, places=6)
        self.assertAlmostEqual(float(term.vel_command_b[0, 2]), -1.0, places=6)
        self.assertAlmostEqual(float(term.vel_command_b[1, 2]), +1.0, places=6)

    def test_standing_envs_are_zeroed(self):
        """基类语义保留：`is_standing_env` 为真的环境整条指令归零（本任务 rel_standing_envs=0）。"""
        term = self._term(y=(0.3, 0.3))
        term._resample_command(torch.tensor([0, 1]))
        term.is_standing_env[0] = True
        term._update_command()
        self.assertTrue(torch.allclose(term.vel_command_b[0], torch.zeros(3)))
        self.assertAlmostEqual(float(term.vel_command_b[1, 1]), -0.3, places=6)

    def test_uses_env_origins_for_lateral_offset(self):
        """横向误差是**相对本环境出生原点**的偏移（`root_pos_w[:,1] − env_origins[:,1]`）。"""
        term = self._term(y=(0.2, 0.2))
        term._resample_command(torch.tensor([0, 1]))
        term._env.scene.env_origins[:, 1] = torch.tensor([0.5, 0.0])
        term._update_command()
        # env0: y = 0.2 − 0.5 = −0.3 ⇒ +0.3；env1: y = 0.2 ⇒ −0.2
        self.assertAlmostEqual(float(term.vel_command_b[0, 1]), +0.3, places=6)
        self.assertAlmostEqual(float(term.vel_command_b[1, 1]), -0.2, places=6)


# ================================================================== ③ 配置级（真实源码）
class TestMixTestEnvCfgSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = CMOE_CFG.read_text(encoding="utf-8-sig")
        cls.cls = _class_def(CMOE_CFG, CFG_CLASS)
        cls.fn = _post_init(CMOE_CFG, CFG_CLASS)
        cls.assign = _assignments(cls.fn)

    def test_class_exists(self):
        self.assertIsNotNone(self.cls, f"{CFG_CLASS} 不存在")
        self.assertIsNotNone(self.fn, f"{CFG_CLASS}.__post_init__ 不存在")

    def test_inherits_play_task(self):
        """**继承 play 类**（保留其确定性设定：pose_range 归零、关域随机化、关观测噪声）。"""
        bases = [ast.unparse(b) for b in self.cls.bases]
        self.assertEqual(bases, [PLAY_CLASS], f"应当且只应当继承 {PLAY_CLASS}，实测 {bases}")
        self.assertIn("self.scene.num_envs = 1", self.src, "play 类的单环境设定应保持不变")

    def test_play_determinism_is_inherited(self):
        """不改的事件/噪声设定仍留在 play 类里（本类不重新打开随机化）。"""
        self.assertNotIn("self.events.randomize_reset_base = ", self.src.split(f"class {CFG_CLASS}")[1])
        self.assertNotIn("enable_corruption = True", self.src.split(f"class {CFG_CLASS}")[1])

    # ---------------------------------------------------------------- 地形：只 mix
    def test_sub_terrains_are_cleared_first(self):
        self.assertTrue(_clear_calls(self.fn), "必须先 `sub_terrains.clear()` 再只放 mix")

    def test_mix_is_the_only_sub_terrain(self):
        names = [k for k in self.assign
                 if re.search(r"\.sub_terrains\[['\"]([a-z_0-9]+)['\"]\]$", k)]
        self.assertEqual(len(names), 1, f"只允许一处 sub_terrains[...] 赋值，实测 {names}")
        match = re.search(r"\.sub_terrains\[['\"]([a-z_0-9]+)['\"]\]$", names[0])
        self.assertEqual(match.group(1), "mix", "唯一地形必须是 mix")
        call = self.assign[names[0]]
        self.assertIsInstance(call, ast.Call)
        self.assertEqual(ast.unparse(call.func), "CMoETrackMixTerrainCfg")
        kwargs = _kwargs(call)
        self.assertAlmostEqual(_literal(kwargs["proportion"]), 1.0, places=9)

    def test_num_cols_one_and_num_rows_twenty(self):
        self.assertEqual(
            _literal(self.assign["self.scene.terrain.terrain_generator.num_cols"]), 1
        )
        self.assertEqual(
            _literal(self.assign["self.scene.terrain.terrain_generator.num_rows"]), 20
        )

    def test_num_envs_is_twenty(self):
        self.assertEqual(_literal(self.assign["self.scene.num_envs"]), 20)

    def test_mix_terrain_params_match_training_defaults(self):
        """mix 的地形参数**逐字沿用训练值**（＝ `cmoe_terrains.py` 的类默认值），防两处漂移。"""
        call = self.assign[
            next(k for k in self.assign if re.search(r"\.sub_terrains\['mix'\]$", k))
        ]
        written = {k: _literal(v) for k, v in _kwargs(call).items()}
        defaults = _mix_defaults()
        self.assertEqual(set(written) - {"proportion"}, set(defaults),
                         f"显式写出的字段应与类字段集合一致：写={sorted(written)} 默认={sorted(defaults)}")
        for field, expected in defaults.items():
            with self.subTest(field=field):
                self.assertAlmostEqual(float(written[field]), float(expected), places=9,
                                       msg=f"{field} 与训练默认值不一致")

    # ----------------------------------------------------------- 速度指令 + PD
    def test_command_item_uses_new_class(self):
        call = self.assign["self.commands.base_velocity"]
        self.assertIsInstance(call, ast.Call)
        self.assertEqual(ast.unparse(call.func), "mdp.MixTestVelocityCommandCfg",
                         "命令项必须换成 mdp.MixTestVelocityCommandCfg")

    def test_command_ranges(self):
        call = self.assign["self.commands.base_velocity"]
        ranges = _kwargs(_kwargs(call)["ranges"])
        self.assertEqual(_literal(ranges["lin_vel_x"]), (1.0, 1.0), "前进速度默认固定 1.0 m/s")
        self.assertEqual(_literal(ranges["lin_vel_y"]), (0.0, 0.0), "横向交给 PD（范围只作占位）")
        self.assertEqual(_literal(ranges["ang_vel_z"]), (0.0, 0.0), "航向交给 PD（范围只作占位）")
        self.assertEqual(_literal(ranges["heading"]), (0.0, 0.0), "heading 目标恒 0")

    def test_heading_command_on_with_zero_standing(self):
        kwargs = _kwargs(self.assign["self.commands.base_velocity"])
        self.assertIs(_literal(kwargs["heading_command"]), True)
        self.assertAlmostEqual(_literal(kwargs["rel_heading_envs"]), 1.0, places=9)
        self.assertAlmostEqual(_literal(kwargs["rel_standing_envs"]), 0.0, places=9)

    def test_no_command_resampling(self):
        kwargs = _kwargs(self.assign["self.commands.base_velocity"])
        self.assertEqual(_literal(kwargs["resampling_time_range"]), (1.0e9, 1.0e9),
                         "整回合指令恒定 ⇒ 不做指令重采样")

    # ----------------------------------------------------------- 契约不变
    def test_observations_and_actions_are_untouched(self):
        """**观测/动作契约不变**：新类不得改观测组、动作维度、扫描器或奖励/终止/课程。"""
        forbidden = (
            "self.observations", "self.actions", "self.rewards", "self.terminations",
            "self.curriculum", "self.scene.height_scanner", "self.scene.robot",
            "self.scene.terrain.terrain_generator.size", "self.decimation", "self.sim",
        )
        touched = sorted(t for t in self.assign if t.startswith(forbidden))
        self.assertEqual(touched, [], f"新类不应触碰观测/动作/奖励等契约：{touched}")

    def test_no_obs_or_action_class_terms_in_body(self):
        body = ast.get_source_segment(self.src, self.cls)
        for token in ("ObsGroup", "ObsTerm", "ActionCfg", "ObsGroupCfg"):
            self.assertNotIn(token, body, f"新类体里不应出现 {token}")

    def test_cmoe_77_ray_contract_still_declared(self):
        """77 维地形扫描（观测契约的一部分）仍在父类里被强制。"""
        self.assertIn("CMoE terrain scan must be 77 rays", self.src)

    def test_new_symbols_are_exported_by_mdp_package(self):
        self.assertIn("from .mix_test_command import *", MDP_INIT.read_text(encoding="utf-8"))


# ================================================================== ④ 注册级
class TestMixTestTaskRegistration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(TASK_INIT.read_text(encoding="utf-8-sig"))
        cls.registrations = {}
        for node in ast.walk(cls.tree):
            if not (isinstance(node, ast.Call) and ast.unparse(node.func) == "gym.register"):
                continue
            kwargs = _kwargs(node)
            task_id = _reg_value(kwargs["id"])
            cls.registrations[task_id] = {
                "entry_point": _reg_value(kwargs["entry_point"]),
                "kwargs": {k: _reg_value(v) for k, v in _dict_items(kwargs["kwargs"]).items()},
            }

    def test_task_is_registered(self):
        self.assertIn(TASK_ID, self.registrations, f"{TASK_ID} 未注册")

    def test_registration_points_at_new_cfg(self):
        entry = self.registrations[TASK_ID]
        self.assertTrue(entry["kwargs"]["env_cfg_entry_point"].endswith(
            f"CMoE_env_cfg:{CFG_CLASS}"), entry["kwargs"]["env_cfg_entry_point"])
        self.assertTrue(entry["kwargs"]["cmoe_rsl_rl_cfg"].endswith(
            "CMoE_rsl_rl_cfg:Imgo2CMoERoughRunnerCfg"), entry["kwargs"]["cmoe_rsl_rl_cfg"])

    def test_registered_like_the_play_family(self):
        """与 `-play` 任务同族：同一 entry_point、同一 agent cfg（checkpoint 可直接加载）。"""
        play = self.registrations["Imgo2-basemove-rough-cmoe-play"]
        mine = self.registrations[TASK_ID]
        self.assertEqual(mine["entry_point"], play["entry_point"])
        self.assertEqual(mine["kwargs"]["cmoe_rsl_rl_cfg"], play["kwargs"]["cmoe_rsl_rl_cfg"])
        self.assertEqual(mine["entry_point"], "rl_lab.envs:CMoEManagerBasedRLEnv")


# ============================================================ 工具：cmoe_terrains 默认值
def _mix_defaults() -> dict[str, float]:
    """从 `cmoe_terrains.py` 读出 `CMoETrackMixTerrainCfg` 的字段默认值（含模块级常量求值）。"""
    tree = ast.parse(CMOE_TERRAINS.read_text(encoding="utf-8-sig"))
    ns: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                ns[node.targets[0].id] = eval(  # noqa: S307 - 只求值仓库自己的常量表达式
                    compile(ast.Expression(node.value), str(CMOE_TERRAINS), "eval"), {}, ns
                )
            except Exception:
                continue
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "CMoETrackMixTerrainCfg")
    defaults: dict[str, float] = {}
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            try:
                defaults[node.target.id] = eval(  # noqa: S307
                    compile(ast.Expression(node.value), str(CMOE_TERRAINS), "eval"), {}, ns
                )
            except Exception:
                continue
    return defaults


# ======================================================================= 桩（命令项用）
class _StubScene:
    def __init__(self, data, origins):
        self._data = data
        self.env_origins = origins
        self._robot = types.SimpleNamespace(data=data)

    def __getitem__(self, name):
        if name != "robot":
            raise KeyError(name)
        return self._robot


class _BaseVelocityCommandStub:
    """`UniformVelocityCommand` 的桩：只保留本测试需要的缓冲与 `_resample_command` 语义。"""

    def __init__(self, cfg, env):
        self.cfg = cfg
        self._env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self.robot = env.scene["robot"]
        self.vel_command_b = torch.zeros(env.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(env.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)

    def _resample_command(self, env_ids):
        r = torch.empty(len(env_ids), device=self.device)
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
        self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs


class _StubMdp:
    UniformVelocityCommand = _BaseVelocityCommandStub


if __name__ == "__main__":
    sys.exit(unittest.main())
