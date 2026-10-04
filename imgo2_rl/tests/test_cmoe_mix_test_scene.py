"""`Imgo2-basemove-rough-cmoe-mix-test`（`mix`-only 受控测试场景）的离线回归。

2026-10-04（用户）：新增一个**继承 play 任务**的测试场景 —— 场景里只有 `mix` 一种地形、速度指令
只给前进（默认恒定 **1.0 m/s**）、横向用 **PD 外环**、heading 保持 0；用来在受控条件下评估策略
通过复合障碍（`mix`）的能力，把横向/航向扰动从评测里剔除。**不是新策略任务** ⇒ 动作空间与观测
契约一字未改，既有 CMoE checkpoint 可直接加载。
2026-10-04（第二批，用户："地形不要按照列排，放在行里面" ＋ "都固定到14难度"）：排布改成
**20 条并排的 `mix` 道**（`num_cols=20`，沿世界 Y）× **20 档难度**（`num_rows=20`，沿世界 X），
且全部环境**固定在第 14 行**（名义 `d = 0.70`），不再依赖 `--terrain_level`。

写法沿用仓库既有做法（`test_masked_terrain_terms.py` / `test_track_geometry.py`）：
`CMoE_env_cfg.py` 与 `mdp/mix_test_command.py` 顶层都 `import isaaclab`（缺 `omni.log`）无法整模块
import，所以 ① 纯 PD 数学从**只依赖 torch** 的 `mdp/mix_test_pd.py` 直接按文件加载；
② 命令项类用 AST 抽出真实源码、在桩环境里 exec；③ 配置级与注册级断言读**真实源码**（AST）。

覆盖：PD 纯函数（零点/符号/四象限/夹取/默认增益）、命令项在桩 env 上的行为（恒定 vx、PD 写
`vel_command_b`、世界系镜像 `vel_command_w`、站立环境归零）、配置级（只 mix、**20 条并排的 mix 道**
（`num_cols=20`）× 20 行难度、难度**固定在第 14 行**（cfg 内钉死 + 冻结课程，不依赖 CLI）、速度与
heading 范围、不重采样、20 环境、命令项用新类）、mix 地形参数与 `cmoe_terrains.py` 默认值**逐一相等**
（防漂移；**唯一有意偏离项**是 `pattern_spacing_scale=2.0`）、任务注册、以及"观测/动作契约未改"。

2026-10-04（第二批，用户："地形不要按照列排，放在行里面" ＋ "都固定到14难度" ＋
"mix 中每个地形间隔大一点 ×1.5~2.0"）：新增 **`TestMixTerrainGeometry`** —— 用桩 `isaaclab` ＋
**真 `trimesh`** 把 `cmoe_terrains.track_mix_terrain` **真跑起来**，断三件事：
① 默认 `pattern_spacing_scale=1.0` 时逐块几何与改动前（`b2caad1`）公式**逐位一致**（"训练不受影响"）；
② `=2.0` 时每块障碍**宽度/高度/顺序不变**、只是间距变大，且图案总长 **6.70 m ≤ 8 m**（含上限 2.40625）；
③ 超过瓦片长度时**直接 raise**（不静默溢出，对照问题表 CMOE-13）。
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

import numpy as np

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
PLAY_SCRIPT = ROOT / "scripts/rl_lab/cmoe/play.py"

TASK_ID = "Imgo2-basemove-rough-cmoe-mix-test"
CFG_CLASS = "Imgo2CMoEMixTestEnvCfg"
PLAY_CLASS = "Imgo2CMoERoughPlayEnvCfg"
IMPORTER_CLASS = "Imgo2CMoEMixTestTerrainImporter"

# mix-test 网格的期望值（与 `CMoE_env_cfg.py` 里的可读常量逐一对应）
MIX_TEST_LANES = 20
MIX_TEST_LEVELS = 20
MIX_TEST_PINNED_LEVEL = 14
MIX_TEST_PATTERN_SPACING_SCALE = 2.0
TILE_SIZE = (8.0, 4.0)
MIX_PATTERN_END_UNITS = 160.0
MIX_PATTERN_START_X = 0.30
MIX_X_UNIT = 0.02

try:
    import torch
except ModuleNotFoundError as error:  # pragma: no cover - 本容器有 torch
    torch = None
    IMPORT_ERROR: object = error
else:
    IMPORT_ERROR = None

try:
    import trimesh  # noqa: F401  （真几何测试需要它；缺了就 skip 那一组）
except ModuleNotFoundError as error:  # pragma: no cover - 本容器有 trimesh
    trimesh = None
    TRIMESH_ERROR: object = error
else:
    TRIMESH_ERROR = None


# --------------------------------------------------------------------------- AST 小工具
def _class_def(path: Path, name: str) -> ast.ClassDef | None:
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    return next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name), None)


def _post_init(path: Path, name: str) -> ast.FunctionDef | None:
    cls = _class_def(path, name)
    if cls is None:
        return None
    return next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__post_init__"), None)


def _method_def(path: Path, class_name: str, method_name: str) -> ast.FunctionDef | None:
    """取指定类里名为 `method_name` 的方法节点（子类钩子用）。"""
    cls = _class_def(path, class_name)
    if cls is None:
        return None
    return next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name), None)


def _module_constants(path: Path = CMOE_CFG) -> dict[str, object]:
    """`CMoE_env_cfg.py` 顶层 `NAME = <可求值字面量>` 常量（`num_cols` 等改用可读常量赋值）。"""
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    out: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = eval(  # noqa: S307 - 只求值仓库自己的常量表达式
                    compile(ast.Expression(node.value), str(path), "eval"), {}, dict(out)
                )
            except Exception:
                continue
    return out


def _class_source(name: str, path: Path = CMOE_CFG) -> str:
    """按 `ast` 精确取出某个类的源码段（比 `str.split("class X")` 可靠：不会被后续类污染）。"""
    source = Path(path).read_text(encoding="utf-8-sig")
    klass = _class_def(path, name)
    if klass is None:
        raise AssertionError(f"{path} 里找不到类 {name}")
    segment = ast.get_source_segment(source, klass)
    if segment is None:  # pragma: no cover - 正常文件不会走到
        raise AssertionError(f"取不到 {name} 的源码段")
    return segment


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

    def test_num_cols_twenty_and_num_rows_twenty(self):
        """**20 条并排的 mix 道**（列 ⇒ 沿世界 Y）× 20 档难度（行 ⇒ 沿世界 X）。"""
        self.assertEqual(
            _literal(self.assign["self.scene.terrain.terrain_generator.num_cols"]), "MIX_TEST_LANES"
        )
        self.assertEqual(
            _literal(self.assign["self.scene.terrain.terrain_generator.num_rows"]), "MIX_TEST_LEVELS"
        )
        # 常量本身必须是 20/20（可读常量只是名字，值要钉住）
        self.assertEqual(_module_constants()["MIX_TEST_LANES"], MIX_TEST_LANES)
        self.assertEqual(_module_constants()["MIX_TEST_LEVELS"], MIX_TEST_LEVELS)

    def test_lane_count_constant_is_documented_as_the_env_ceiling(self):
        """`num_cols` 的可读常量旁必须写明"道数 = 可同时评估的环境数上限"（用户要求）。"""
        index = self.src.index("\nMIX_TEST_LANES = ")
        block: list[str] = []
        for line in reversed(self.src[:index].splitlines()):
            if not line.startswith("#"):
                break
            block.append(line)
        comment = "\n".join(reversed(block))
        self.assertIn("可同时评估的环境数上限", comment, f"常量注释缺失：{comment!r}")

    def test_num_envs_is_twenty(self):
        self.assertEqual(
            _literal(self.assign["self.scene.num_envs"]), "MIX_TEST_LANES"
        )
        self.assertEqual(_module_constants()["MIX_TEST_LANES"], 20,
                         "默认环境数 = 道数 ⇒ 环境 i → 第 i 道（一条道一个环境）")

    # --------------------------------------------------- 难度固定 14（不依赖 CLI）
    def test_max_init_terrain_level_is_pinned_to_fourteen(self):
        self.assertEqual(
            _literal(self.assign["self.scene.terrain.max_init_terrain_level"]), "MIX_TEST_PINNED_LEVEL"
        )
        self.assertEqual(_module_constants()["MIX_TEST_PINNED_LEVEL"], MIX_TEST_PINNED_LEVEL)

    def test_terrain_importer_is_the_pinning_subclass(self):
        """`scene.terrain.class_type` 指向本仓的钉子类 ⇒ 等级固定 + 课程冻结（都不依赖 CLI）。"""
        self.assertEqual(
            ast.unparse(self.assign["self.scene.terrain.class_type"]), IMPORTER_CLASS
        )
        klass = _class_def(CMOE_CFG, IMPORTER_CLASS)
        self.assertIsNotNone(klass, f"{IMPORTER_CLASS} 不存在")
        bases = [ast.unparse(b) for b in klass.bases]
        self.assertEqual(bases, ["terrain_gen.TerrainImporter"],
                         f"必须继承 Isaac Lab 的 TerrainImporter，实测 {bases}")

    def test_importer_pins_levels_and_freezes_the_curriculum(self):
        """子类的两条关键语义：把 `terrain_levels` 全设成 `pinned_level`；`update_env_origins` 空操作。"""
        body = ast.get_source_segment(self.src, _class_def(CMOE_CFG, IMPORTER_CLASS))
        self.assertIn("self.terrain_levels[:] = int(self.pinned_level)", body)
        self.assertIn("pinned_level: int = MIX_TEST_PINNED_LEVEL", body)
        self.assertIsNotNone(_method_def(CMOE_CFG, IMPORTER_CLASS, "configure_env_origins"),
                             "子类必须重写 configure_env_origins")
        method = _method_def(CMOE_CFG, IMPORTER_CLASS, "update_env_origins")
        self.assertIsNotNone(method, "子类必须重写 update_env_origins（否则难度会在一局内漂走）")
        stmts = [n for n in method.body if not isinstance(n, ast.Expr)]
        self.assertEqual(len(stmts), 1, f"update_env_origins 应只有一条 return，实测 {len(stmts)} 条")
        self.assertIsInstance(stmts[0], ast.Return)

    def test_importer_warns_when_envs_exceed_lanes(self):
        """`num_envs > num_cols` 时 `terrain_types` 会重复（出生点重叠）⇒ 子类必须告警。"""
        body = ast.get_source_segment(self.src, _class_def(CMOE_CFG, IMPORTER_CLASS))
        self.assertIn("num_envs > num_cols", body)
        self.assertIn("[WARN]", body)


    def test_class_type_is_not_touched_for_train_or_play(self):
        """训练/play 的 `class_type` 仍是 Isaac Lab 的默认 `TerrainImporter`（本改动只影响评测 cfg）。"""
        for name, label in (("Imgo2CMoERoughEnvCfg", "训练"), (PLAY_CLASS, "play")):
            self.assertNotIn("class_type", _class_source(name), f"{label} 链不得改 class_type")
        self.assertIn("max_init_terrain_level = 5", _class_source(PLAY_CLASS),
                      "play 的初始等级上限必须保持 5")

    # --------------------------------------------------- 障碍间距乘子
    def test_pattern_spacing_scale_is_two(self):
        call = self.assign[
            next(k for k in self.assign if re.search(r"\.sub_terrains\['mix'\]$", k))
        ]
        kwargs = _kwargs(call)
        self.assertIn("pattern_spacing_scale", kwargs, "间距乘子必须显式写出")
        self.assertEqual(ast.unparse(kwargs["pattern_spacing_scale"]),
                         "MIX_TEST_PATTERN_SPACING_SCALE")
        self.assertEqual(
            _module_constants()["MIX_TEST_PATTERN_SPACING_SCALE"],
            MIX_TEST_PATTERN_SPACING_SCALE,
        )

    def test_default_spacing_scale_stays_one_for_training(self):
        """硬要求：`CMoETrackMixTerrainCfg` 的默认乘子必须是 **1.0**；训练实例化**不传**它。"""
        self.assertEqual(_mix_defaults()["pattern_spacing_scale"], 1.0)
        train = _class_source("Imgo2CMoERoughEnvCfg")
        call = next(
            node for node in ast.walk(ast.parse(train))
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "CMoETrackMixTerrainCfg"
        )
        self.assertNotIn("pattern_spacing_scale", _kwargs(call),
                         "训练侧不得传间距乘子（取默认 1.0 ⇒ 几何逐位不变）")

    def test_mix_pattern_fits_the_tile(self):
        """总长核算：`pattern_start_x + 160·x_unit·scale ≤ size[0]`（不静默溢出）。"""
        call = self.assign[
            next(k for k in self.assign if re.search(r"\.sub_terrains\['mix'\]$", k))
        ]
        kwargs = _kwargs(call)
        offset = float(_literal(kwargs["pattern_start_x"]))
        x_unit = float(_literal(kwargs["x_unit"]))
        scale = MIX_TEST_PATTERN_SPACING_SCALE
        size_x = TILE_SIZE[0]
        end = offset + MIX_PATTERN_END_UNITS * x_unit * scale
        self.assertAlmostEqual(end, 6.70, places=9, msg="scale=2.0 时图案末端应在 6.70 m")
        self.assertLess(end, size_x, f"图案总长 {end} m 必须装得进 {size_x} m 瓦片")
        self.assertAlmostEqual(size_x - end, 1.30, places=9, msg="尾部走廊应剩 1.30 m")
        max_scale = (size_x - offset) / (MIX_PATTERN_END_UNITS * x_unit)
        self.assertAlmostEqual(max_scale, 2.40625, places=9,
                               msg="该瓦片上乘子上限 = (8 − 0.30) / (160 × 0.02)")
        self.assertGreaterEqual(max_scale, scale, "选定乘子必须 ≤ 上限（否则 raise）")
        # 用户给的区间是 1.5~2.0；取的是上限
        self.assertGreaterEqual(scale, 1.5)
        self.assertLessEqual(scale, 2.0)

    def test_tile_size_is_unchanged(self):
        """`size=(8, 4)` 由父类设定，本类不得改（世界范围：X 20×8=160 m、Y 20×4=80 m）。"""
        self.assertNotIn("terrain_generator.size", _class_source(CFG_CLASS))
        train = _class_source("Imgo2CMoERoughEnvCfg")
        self.assertIn("self.scene.terrain.terrain_generator.size = (8.0, 4.0)", train)
        self.assertIn("self.scene.terrain.terrain_generator.num_rows = 20", train)
        self.assertIn("self.scene.terrain.terrain_generator.num_cols = 40", train)

    def test_mix_terrain_params_match_training_defaults(self):
        """mix 的地形参数**逐字沿用训练值**（＝ `cmoe_terrains.py` 的类默认值），防两处漂移。

        唯一两处**有意偏离**：`proportion=1.0`（本场景只有这一类）与
        `pattern_spacing_scale=2.0`（评测专用的间距乘子）—— 两者都单独断言。
        """
        call = self.assign[
            next(k for k in self.assign if re.search(r"\.sub_terrains\['mix'\]$", k))
        ]
        written = {k: _literal(v) for k, v in _kwargs(call).items()}
        defaults = _mix_defaults()
        deviations = {"proportion", "pattern_spacing_scale"}
        self.assertEqual(set(written) - deviations, set(defaults) - deviations,
                         f"显式写出的字段应与类字段集合一致：写={sorted(written)} 默认={sorted(defaults)}")
        self.assertAlmostEqual(float(_literal(_kwargs(call)["proportion"])), 1.0, places=9)
        self.assertEqual(ast.unparse(_kwargs(call)["pattern_spacing_scale"]),
                         "MIX_TEST_PATTERN_SPACING_SCALE")
        for field, expected in defaults.items():
            if field in deviations:
                continue
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


# ============================= ③.5 mix 图案的**真几何**（桩 isaaclab + 真 trimesh）
# 改动前（`b2caad1`）`track_mix_terrain` 里逐字使用的图案表：`(start_units, end_units, height_units)`，
# 其中坑的两个起点写成 `"72-k"`/`"123-k"`（= `round(gap_shrink_units·(1−d))` 的难度律）。本表当**参照**，
# 用来断言"默认乘子 1.0 时与改动前逐块几何一致"。
_MIX_PATTERN = (
    (0.0, 30.0, 0.0),
    (30.0, 36.0, 30.0),
    (36.0, 42.0, 60.0),
    (42.0, 48.0, 90.0),
    (48.0, 60.0, 120.0),
    ("72-k", 84.0, 120.0),
    (84.0, 86.0, 0.0),
    (86.0, 96.0, 96.0),
    (96.0, 99.0, 170.0),
    (99.0, 111.0, 120.0),
    ("123-k", 140.0, 120.0),
    (140.0, 160.0, 60.0),
)
_GAP_BASE_UNITS = {"72-k": 72.0, "123-k": 123.0}


def _load_cmoe_terrains() -> dict:
    """exec 真实 `cmoe_terrains.py`（只把 `isaaclab.terrains` / `isaaclab.utils` 换成桩）。

    `numpy` 与 `trimesh` 用**真的** ⇒ `track_mix_terrain` 生成真网格，可以按 `mesh.bounds` 逐块核对几何。
    桩只为绕开 `import isaaclab` 缺 `omni.log`（本容器无 Isaac 运行环境）。用完立刻还原 `sys.modules`。
    """
    isaaclab = types.ModuleType("isaaclab")
    terrains = types.ModuleType("isaaclab.terrains")
    utils = types.ModuleType("isaaclab.utils")

    class _SubTerrainBaseCfg:  # 只当基类占位（函数只用 cfg 的数据字段）
        pass

    terrains.SubTerrainBaseCfg = _SubTerrainBaseCfg
    utils.configclass = lambda cls: cls  # noqa: E731 - 桩装饰器，原样返回类
    isaaclab.terrains = terrains
    isaaclab.utils = utils
    names = ("isaaclab", "isaaclab.terrains", "isaaclab.utils")
    saved = {name: sys.modules.get(name) for name in names}
    sys.modules.update(dict(zip(names, (isaaclab, terrains, utils))))
    try:
        namespace = {"__name__": "cmoe_terrains_under_test", "__file__": str(CMOE_TERRAINS)}
        exec(  # noqa: S102 - 只跑仓库自己的源码
            compile(CMOE_TERRAINS.read_text(encoding="utf-8-sig"), str(CMOE_TERRAINS), "exec"),
            namespace,
        )
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return namespace


def _pattern_rows(difficulty: float, spacing: float, size=TILE_SIZE) -> list[tuple]:
    """**参照表**（改动前公式＋间距乘子）：`[(x0, x1, y0, y1, top), ...]`，顺序＝函数的返回顺序。"""
    defaults = _mix_defaults()
    offset = float(defaults["pattern_start_x"])
    x_unit, z_unit = float(defaults["x_unit"]), float(defaults["z_unit"])
    height_scale = float(defaults["height_scale"])
    gap_shrink = round(float(defaults["gap_shrink_units"]) * (1.0 - difficulty))
    corridor = float(defaults["corridor_width"])
    pit_depth = float(defaults["pit_depth"])
    diff = height_scale * difficulty
    y0, y1 = 0.5 * (size[1] - corridor), 0.5 * (size[1] + corridor)
    rows = [(0.0, size[0], 0.0, size[1], -pit_depth), (0.0, offset, y0, y1, 0.0)]
    for start, end, height_units in _MIX_PATTERN:
        if isinstance(start, str):
            start = _GAP_BASE_UNITS[start] - gap_shrink
        shift = (spacing - 1.0) * start * x_unit
        x0 = offset + start * x_unit + shift
        x1 = min(offset + end * x_unit + shift, size[0])
        if end <= start or x1 <= x0:
            continue
        rows.append((x0, x1, y0, y1, height_units * z_unit * diff))
    tail = min(
        offset + MIX_PATTERN_END_UNITS * x_unit
        + (spacing - 1.0) * MIX_PATTERN_END_UNITS * x_unit,
        size[0],
    )
    if tail < size[0]:
        rows.append((tail, size[0], y0, y1, 0.0))
    return rows


def _mesh_rows(meshes) -> list[tuple]:
    """把 `trimesh` 网格摊成 `(x0, x1, y0, y1, top)`（`top` = 包围盒上表面 z）。"""
    rows = []
    for mesh in meshes:
        low, high = mesh.bounds
        rows.append((float(low[0]), float(high[0]), float(low[1]), float(high[1]), float(high[2])))
    return rows


@unittest.skipIf(trimesh is None, f"trimesh unavailable: {TRIMESH_ERROR}")
class TestMixTerrainGeometry(unittest.TestCase):
    """**真跑** `track_mix_terrain`（桩 isaaclab ＋ 真 trimesh）核对间距乘子与溢出保护。"""

    @classmethod
    def setUpClass(cls):
        namespace = _load_cmoe_terrains()
        cls.function = staticmethod(namespace["track_mix_terrain"])
        cls.cfg_class = namespace["CMoETrackMixTerrainCfg"]

    def _build(self, spacing, difficulty, size=TILE_SIZE):
        cfg = self.cfg_class()
        cfg.size = size
        cfg.proportion = 1.0
        cfg.pattern_spacing_scale = spacing
        return self.function(difficulty, cfg)[0]

    def _assert_matches_reference(self, meshes, spacing, difficulty, size=TILE_SIZE):
        got = _mesh_rows(meshes)
        want = _pattern_rows(difficulty, spacing, size)
        self.assertEqual(len(got), len(want),
                         f"网格块数不一致（spacing={spacing}, d={difficulty}, size={size}）")
        for index, (row, ref) in enumerate(zip(got, want)):
            with self.subTest(index=index):
                for value, expected in zip(row, ref):
                    self.assertAlmostEqual(value, expected, places=9,
                                           msg=f"第 {index} 块 {row} != 参照 {ref}")

    # ------------------------------------------------------------ 默认 1.0 = 改动前
    def test_default_spacing_reproduces_the_pre_change_geometry(self):
        """**硬要求**：默认乘子 1.0 ⇒ 与改动前（`b2caad1` 的公式）**逐块几何一致**。"""
        for difficulty in (0.0, 0.35, 0.70, 0.75, 1.0):
            for size in ((8.0, 4.0), (8.0, 8.0)):
                with self.subTest(difficulty=difficulty, size=size):
                    self._assert_matches_reference(
                        self._build(1.0, difficulty, size), 1.0, difficulty, size
                    )

    def test_training_default_instance_uses_spacing_one(self):
        """不传字段（训练侧 `CMoETrackMixTerrainCfg(proportion=0.10)`）走的就是默认 1.0 那条路径。"""
        cfg = self.cfg_class()
        cfg.size = TILE_SIZE
        self.assertEqual(cfg.pattern_spacing_scale, 1.0)
        meshes = self.function(0.70, cfg)[0]
        self._assert_matches_reference(meshes, 1.0, 0.70)

    # ------------------------------------------------------------ 2.0 = 只放大间距
    def test_two_times_spacing_only_changes_the_gaps(self):
        """`=2.0`：每块障碍**宽度/顶面高度/顺序不变**，只有 X 位置后移（间距变大）。

        ⚠️ 唯一例外是**最后一块尾部走廊**：它的起点是"图案末端"（会后移）⇒ 宽度必然变短，属预期。
        """
        difficulty = 0.70
        base = _mesh_rows(self._build(1.0, difficulty))
        wide = _mesh_rows(self._build(2.0, difficulty))
        self.assertEqual(len(base), len(wide), "块数必须不变（不新增/不丢图案）")
        for index, (small, big) in enumerate(zip(base, wide)):
            with self.subTest(index=index):
                if index != len(base) - 1:  # 尾廊除外（它的起点就是图案末端）
                    self.assertAlmostEqual(big[1] - big[0], small[1] - small[0], places=9,
                                           msg=f"第 {index} 块宽度被改了")
                self.assertAlmostEqual(big[2], small[2], places=9, msg="y 下界被改了")
                self.assertAlmostEqual(big[3], small[3], places=9, msg="y 上界被改了")
                self.assertAlmostEqual(big[4], small[4], places=9, msg=f"第 {index} 块顶面高度被改了")
                self.assertGreaterEqual(big[0], small[0] - 1e-12, "X 位置只能后移")
        self._assert_matches_reference(self._build(2.0, difficulty), 2.0, difficulty)

    def test_two_times_spacing_keeps_the_block_order_and_no_overlap(self):
        """走廊块（跳过整宽坑底那一块）起点单调递增、互不重叠 ⇒ 图案顺序不变。"""
        wide = _mesh_rows(self._build(2.0, 0.70))[1:]
        starts = [row[0] for row in wide]
        self.assertEqual(starts, sorted(starts), "图案顺序必须不变（X 起点单调递增）")
        for before, after in zip(wide, wide[1:]):
            self.assertGreaterEqual(after[0] + 1e-12, before[1], "拉大间距后相邻块不得重叠")

    # ------------------------------------------------------------ 总长核算（不静默溢出）
    def test_total_length_accounting_at_two_times(self):
        """总长核算：图案末端 6.70 m（尾部走廊 6.70→8.00）；被拉开的空间是**深坑**（见 docs）。"""
        rows = _mesh_rows(self._build(2.0, 0.70))
        corridors = rows[1:]
        self.assertAlmostEqual(corridors[0][0], 0.00, places=9, msg="起步走廊起点不动")
        self.assertAlmostEqual(corridors[0][1], 0.30, places=9, msg="起步平台宽度 0.30 m 不变")
        self.assertAlmostEqual(corridors[-1][0], 6.70, places=9, msg="尾部走廊起点 = 图案末端 6.70 m")
        self.assertAlmostEqual(corridors[-1][1], 8.00, places=9, msg="尾廊铺到瓦片末端")
        wide_gaps = [after[0] - before[1] for before, after in zip(corridors, corridors[1:])]
        self.assertAlmostEqual(max(wide_gaps), 0.60, places=9,
                               msg=f"最大间距（深坑）应为 0.60 m：{wide_gaps}")
        base_corridors = _mesh_rows(self._build(1.0, 0.70))[1:]
        base_gaps = [after[0] - before[1] for before, after in zip(base_corridors, base_corridors[1:])]
        self.assertAlmostEqual(max(base_gaps), 0.18, places=9,
                               msg=f"改动前最大间距（深坑）是 0.18 m：{base_gaps}")
        # 被拉开的空间由**原有几何**填充：坑底整宽 -0.50 m ⇒ 只加宽（不加平地跑道）
        self.assertAlmostEqual(rows[0][4], -0.50, places=9, msg="全瓦片坑底顶面 -0.50 m 不变")

    def test_overflow_guard_raises_instead_of_silently_clipping(self):
        """溢出保护：超出瓦片长度**直接 raise**（并给出上限），不静默截断（对照 CMOE-13）。"""
        with self.assertRaises(ValueError) as ctx:
            self._build(2.5, 0.70)
        message = str(ctx.exception)
        self.assertIn("超出瓦片长", message)
        self.assertIn("2.4062", message, f"应给出该瓦片上的乘子上限：{message}")
        self.assertIn("8.3000", message, f"应给出实际总长：{message}")

    def test_max_fitting_spacing_is_the_boundary(self):
        """上限 2.40625 = (8 − 0.30) / (160 × 0.02)：取它不 raise，再大一点就 raise。"""
        max_spacing = (TILE_SIZE[0] - MIX_PATTERN_START_X) / (MIX_PATTERN_END_UNITS * MIX_X_UNIT)
        self.assertAlmostEqual(max_spacing, 2.40625, places=9)
        self._build(max_spacing, 0.70)  # 不抛异常
        with self.assertRaises(ValueError):
            self._build(max_spacing + 1.0e-4, 0.70)

    def test_guard_formula_is_present_in_the_source(self):
        """保护必须写在 `track_mix_terrain` 里（源码级锁定，防止被删掉后仍"看起来能跑"）。"""
        source = CMOE_TERRAINS.read_text(encoding="utf-8-sig")
        body = source[source.index("def track_mix_terrain"):source.index("class CMoETrackMixTerrainCfg")]
        self.assertIn("cfg.size[0]", body)
        self.assertIn("raise ValueError", body)
        self.assertIn("pattern_spacing_scale", body)
        # 默认值必须精确为 0 增量（逐位不变的关键）
        self.assertIn("(spacing - 1.0) * anchor_units * x_unit", body)
        self.assertIn("pattern_spacing_scale: float = 1.0", source)


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


# ================================================== ⑤ CLI：`--terrain_level` 的语义与文档
class TestMixTestTerrainLevelCli(unittest.TestCase):
    """`scripts/rl_lab/cmoe/play.py` 的 `--terrain_level=N`：确认**不夹到 9** 并更新了帮助文本。

    用户要求先查清它能否直接钉 14（其 help 原写 0–9）：实测 argparse 是 `type=int` + **无 choices**，
    代码是 `terrain.terrain_levels[:] = int(args_cli.terrain_level)` ⇒ **不做任何夹取/截断**，
    `num_rows=20` 时 14 合法（0..19）。本类把"帮助文本不得再写 0–9"钉住。
    """

    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(PLAY_SCRIPT.read_text(encoding="utf-8-sig"))

    def _terrain_level_call(self) -> ast.Call:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "parser.add_argument":
                if node.args and isinstance(node.args[0], ast.Constant) \
                        and node.args[0].value == "--terrain_level":
                    return node
        raise AssertionError("play.py 里找不到 --terrain_level 参数")

    def test_no_choices_and_int_type(self):
        call = self._terrain_level_call()
        kwargs = _kwargs(call)
        self.assertEqual(ast.unparse(kwargs["type"]), "int")
        self.assertNotIn("choices", kwargs, "有 choices 才会被截断；实测没有 ⇒ 可钉 14")
        self.assertIs(_literal(kwargs["default"]), None, "默认 None = 沿用任务配置")

    def test_help_no_longer_says_zero_to_nine(self):
        help_text = ast.literal_eval(_kwargs(self._terrain_level_call())["help"])
        self.assertNotIn("0–9", help_text, "帮助文本里的 0–9 是旧值（现在等级可达 num_rows−1=19）")
        self.assertIn("被夹到 9", help_text, "必须写明等级不会被夹到 9")
        self.assertIn("不会", help_text)
        self.assertIn("14", help_text, "应写明 mix-test 固定 14")

    def test_help_matches_the_source_behaviour(self):
        """帮助文本说"不夹取"，源码里就必须**没有** clip/choices 之类的夹取（双向一致）。"""
        source = PLAY_SCRIPT.read_text(encoding="utf-8")
        block = source[source.index("if args_cli.terrain_level is not None:"):]
        block = block[:block.index("\n    if args_cli.video:")]
        self.assertIn("terrain.terrain_levels[:] = level", block)
        self.assertNotIn("min(", block)
        self.assertNotIn("clip(", block)



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
