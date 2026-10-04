"""`Imgo2-basemove-rough-cmoe-mix-test`（`mix`-only 受控测试场景）的离线回归。

2026-10-04（用户）：新增一个**继承 play 任务**的测试场景 —— 场景里只有 `mix` 一种地形、速度指令
只给前进（默认恒定 **1.0 m/s**）、横向用 **PD 外环**、heading 保持 0；用来在受控条件下评估策略
通过复合障碍（`mix`）的能力，把横向/航向扰动从评测里剔除。**不是新策略任务** ⇒ 动作空间与观测
契约一字未改，既有 CMoE checkpoint 可直接加载。
2026-10-04（第二批，用户："地形不要按照列排，放在行里面" ＋ "都固定到14难度"）：排布改成
**20 条并排的 `mix` 道**（`num_cols=20`，沿世界 Y）× **20 档难度**（`num_rows=20`，沿世界 X），
且全部环境**固定在第 14 行**（名义 `d = 0.70`），不再依赖 `--terrain_level`。
2026-10-04（第三批，用户："分居了，但是我不需要还保持那么多行，我需要他们并列"）：**行数 20 → 1**
（19 行原本永远用不到），世界由 160 m(X) × 80 m(Y) 缩到 **8 m(X) × 80 m(Y)**；难度不再靠"把等级钉在
第 14 行"，改由 `terrain_generator.difficulty_range = (0.70, 0.70)` **精确固定**（课程公式里
"行内 `U(0,1)` 抖动"被 `upper − lower = 0` 消掉）；钉等级的子类 `Imgo2CMoEMixTestTerrainImporter`
**删除**（`num_rows = 1` ⇒ 课程天然只有 0 级，`max_init_terrain_level = 0`）；间距乘子 2.0 → **1.0**
（用户："各个难度间距先不要调整" ⇒ 评测几何与训练**逐位一致**）。

写法沿用仓库既有做法（`test_masked_terrain_terms.py` / `test_track_geometry.py`）：
`CMoE_env_cfg.py` 与 `mdp/mix_test_command.py` 顶层都 `import isaaclab`（缺 `omni.log`）无法整模块
import，所以 ① 纯 PD 数学从**只依赖 torch** 的 `mdp/mix_test_pd.py` 直接按文件加载；
② 命令项类用 AST 抽出真实源码、在桩环境里 exec；③ 配置级与注册级断言读**真实源码**（AST）。

覆盖：PD 纯函数（零点/符号/四象限/夹取/默认增益）、命令项在桩 env 上的行为（恒定 vx、PD 写
`vel_command_b`、世界系镜像 `vel_command_w`、站立环境归零）、配置级（只 mix、**20 条并列的 mix 道**
（`num_cols=20`）× **唯一一行**难度（`num_rows=1`）、难度**精确固定 0.70**（`difficulty_range=(0.70,0.70)`，
不依赖 CLI、也不再需要子类）、世界坐标 X ∈ [−10,+10] / Y ∈ [−40,+40]（第五批：瓦片 X 8 m → 20 m）、速度与 heading 范围、
不重采样、20 环境、命令项用新类）、mix 地形参数与 `cmoe_terrains.py` 默认值**逐一相等**
（防漂移；**唯一有意偏离项**是 `proportion=1.0`）、任务注册、以及"观测/动作契约未改"。

2026-10-04（第二批）：新增 **`TestMixTerrainGeometry`** —— 用桩 `isaaclab` ＋ **真 `trimesh`** 把
`cmoe_terrains.track_mix_terrain` **真跑起来**，断三件事：
① 默认 `pattern_spacing_scale=1.0` 时逐块几何与改动前（`b2caad1`）公式**逐位一致**（"训练不受影响"）；
② `=2.0` 时每块障碍**宽度/高度/顺序不变**、只是间距变大，且图案总长 **6.70 m ≤ 8 m**（含上限 2.40625）；
③ 超过瓦片长度时**直接 raise**（不静默溢出，对照问题表 CMOE-13）。
2026-10-04（第三批）：新增 **`TestMixTestDifficultyPinning`**（难度确实被固定在 0.70，含 Isaac Lab
源码级依据与"生成器会传进去的 difficulty"两条路径）与 **`TestMixTestSpawnGeometry`**（用真几何量化
"`scale=2.0` 时出生点紧贴深坑、`scale=1.0` 落在一块 0.60 m 平台上"⇒ 这就是评测场景取 1.0 的原因），
以及 **`TestTrainingSpawnHazard`**（把训练侧 `spawn_x=0.75` ＋ `pose_range` 的同类隐患量化并**钉住**，
**本轮不改训练**，见 README 问题表 CMOE-17）。

2026-10-04（**第四批**，用户原话：「mix 还是太小了…课程长度太短：让 mix 占满整条道」）：`mix` 的障碍
序列铺满整条 8 m 道，**但障碍自身的尺寸/高度一字不变**（只把障碍之间拉开），并把拉开出来的区间
**铺成 `height=0` 的可走面**（而不是留成 −0.50 m 深坑）。本文件为此新增：

* **`cmoe_terrains.CMoETrackMixTerrainCfg.fill_stretched_gaps: bool = False`** —— 默认 False ⇒
  **与改动前逐位相同**（训练侧不传；`TestMixTerrainGeometry.test_fill_flag_and_defaults_are_neutral`
  + 既有的 10 组逐块对照一起钉住）；True 且 `scale > 1` 时把"拉开多出来的空档"（段与段之间、尾段）
  铺平，**原有两处坑（d=0.70 各 0.18 m）原样保留**；
* **评测场景取反算的"刚好占满整条道"乘子 `2.25`**（`(8 − 0.30 − 0.50)/(160 × 0.02)`；≤ 溢出上限
  2.40625，保护**不放宽**）＋ `fill_stretched_gaps = True` ⇒ 图案末端 **7.50 m**（≥ 7.0 m 目标、
  ≥ 85 % 道长的 6.8 m），尾部平地 0.50 m；出生点前方实心地面 **1.95 m**（原 0.75 m）；
* 新增断言（`TestMixTerrainGeometry` / `TestMixTestFilledSpawnGeometry` /
  `TestMixTestEnvCfgSource::test_scene_fills_the_whole_lane`）：① 默认 False 与改动前逐位相同；
  ② True 时**图案区间内的可走面**随 `scale` 单调增、**坑总长恒为 0.36 m（不随 scale 增）**、补出的
  平地为 `3.2·(scale−1)`；③ **障碍自身尺寸与 scale 无关**（`d=0.70` 时栏高 0.2618 m、第一处坑宽
  0.18 m 在任意 scale 下不变）；④ 图案末端 ≥ 7.0 m；⑤ 出生点安全（`scale ∈ {1.0, 2.25}` 都落在实心
  区间内、前方 ≥ 1.0 m）。
  ⚠️ **一处与字面要求的冲突（如实记录）**：**整块瓦片**的可走面总长**不随 scale 变**（
  `可走面 + 坑 = size[0]` 恒等，坑总长恒定 ⇒ 整片可走面恒为 7.64 m；尾廊会吸收拉伸量）。
  因此"可走面总长随 scale 单调增"只能对**图案区间内**的可走面成立（2.84 → 6.84 m），本文按此断言。

2026-10-04（**第五批**，用户原话：「整体地形放大，原本是10m长就改成20m长，还是布满，但是障碍数量
不变，设置不变，只把间隔改大」）：**单块瓦片的 X 由 8 m 放大到 20 m**（`terrain_generator.size =
(20, 4)`，Y 仍 4 m），mix 课程继续**铺满整条 20 m**，而**障碍数量 / 障碍自身尺寸与高度 / 坑深 /
走廊宽 / 难度(0.70) / 图案顺序全部不变**，只把障碍之间的间隔拉大。本文件为此：

* 反算式只换 `size[0]`：`scale = (20 − 0.30 − 0.50)/(160 × 0.02) = 19.20/3.20 = 6.00`
  ⇒ 图案末端 **19.50 m**（占 97.5 %，尾部平地 0.50 m），仍严格小于溢出上限
  `(20 − 0.30)/(160 × 0.02) = 6.15625` ⇒ **溢出保护不放宽**；
* `TILE_SIZE` 8 m → **20 m**、世界 X ∈ **[−10, +10]**、出生点世界 X = `0.75 − 10 = −9.25`；
  所有与瓦片长度相关的期望值同步（坑总长仍 0.36 m、整片可走面 7.64 → **19.64 m**、
  图案区间可走面 2.84 → **18.84 m**、补出平地 4.00 → **16.00 m**）；
* **新增"训练/play 侧 `size` 未被改"的断言**（`size` 是 `terrain_generator` 的共享字段，
  `Imgo2CMoERoughEnvCfg` 里仍是 `(8.0, 4.0)`；play 类也不设它）；
* **新增单局时长断言**：道 20 m ÷ 1.0 m/s = 20 s ⇒ 评测 cfg 把继承来的 `episode_length_s = 20 s`
  提到 **35 s**（常量 `MIX_TEST_EPISODE_LENGTH_S`），并断言 ≥ 25 s；**训练/play 的时长一字未改**；
* `TestMixTestFilledSpawnGeometry.MEASURED` 与 docs §7 的 X 区间表**用真 trimesh 在 20 m 道上重跑**；
  出生点 `spawn_x = 0.75` 仍在实心区间 `[0.30, 0.90]`（顶面 0）内，**前方实心地面 1.95 → 5.55 m**
  （第一处坑 `[2.70, 2.88]` → `[6.30, 6.48]`）。**如实记录一处与用户预期的差异**：用户预期"~8 m 级"，
  实测 **5.55 m** —— 第一处坑按既定的铺平地规则**紧贴上游块末端**，落在 6.30 m 而不是"下游块起点"
  （8.58 m）之前；详见 `test_spawn_is_safe_at_both_scales` 的 docstring。
  ⚠️ `scale = 1.0`（训练几何）下"前方实心"仍只有 **0.75 m**（既有 `CMOE-17` 结论不变）。
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

# Isaac Lab 的课程地形生成器源码（核实 `difficulty_range` 的字段名与语义用；缺失则 skip 那一组）
ISAACLAB_TERRAINS = Path("/root/IsaacLab/source/isaaclab/isaaclab/terrains")
ISAAC_GENERATOR = ISAACLAB_TERRAINS / "terrain_generator.py"
ISAAC_GENERATOR_CFG = ISAACLAB_TERRAINS / "terrain_generator_cfg.py"
ISAAC_IMPORTER = ISAACLAB_TERRAINS / "terrain_importer.py"

TASK_ID = "Imgo2-basemove-rough-cmoe-mix-test"
CFG_CLASS = "Imgo2CMoEMixTestEnvCfg"
PLAY_CLASS = "Imgo2CMoERoughPlayEnvCfg"
TRAIN_CLASS = "Imgo2CMoERoughEnvCfg"
# 2026-10-04（第三批）：钉等级的子类已删除（`num_rows=1` ⇒ 课程天然冻结）
REMOVED_IMPORTER_CLASS = "Imgo2CMoEMixTestTerrainImporter"

# mix-test 网格的期望值（与 `CMoE_env_cfg.py` 里的可读常量逐一对应）
MIX_TEST_LANES = 20
MIX_TEST_LEVELS = 1
MIX_TEST_DIFFICULTY = 0.70
# 2026-10-04（第五批）：**单块瓦片 X 8 m → 20 m**（用户："整体地形放大，原本是10m长就改成20m长，
# 还是布满，但是障碍数量不变，设置不变，只把间隔改大"）⇒ 乘子按同一反算式重算为
#   scale = (size[0] − pattern_start_x − 尾部余量) / (160 · x_unit) = (20 − 0.30 − 0.50)/(160 × 0.02) = 6.00
#   ⇒ 图案末端 = 0.30 + 160 × 0.02 × 6.00 = 19.50 m（占 20 m 的 97.5 %），尾部平地 0.50 m。
MIX_TEST_PATTERN_SPACING_SCALE = 6.00
MIX_TEST_TAIL_MARGIN = 0.50
MIX_TEST_PATTERN_END_X = 19.50
MIX_TEST_FILL_STRETCHED_GAPS = True
# "占满整条道"的目标下界：图案末端 ≥ 19.0 m（＝≥ 95 % 的 20 m 道，随第五批的瓦片长度一起抬）
MIX_PATTERN_END_MIN_X = 19.0
TILE_SIZE = (20.0, 4.0)
# 训练/play 侧仍是 `(8, 4)`（`size` 是 `terrain_generator` 的共享字段 ⇒ 只有评测 cfg 覆盖它）
TRAIN_TILE_SIZE = (8.0, 4.0)
# 单局时长（第五批）：道 20 m ÷ 恒定 1.0 m/s = 20 s ⇒ 评测 cfg 必须 ≥ 25 s（当前取 35 s）。
MIX_TEST_EPISODE_LENGTH_S = 35.0
MIX_TEST_EPISODE_LENGTH_MIN_S = 25.0
MIX_TEST_FORWARD_SPEED = 1.0
# 世界范围 = 单块尺寸 × 网格（行沿 X、道沿 Y）；地形整体按 (-size0·rows/2, -size1·cols/2) 居中
WORLD_X = (-0.5 * TILE_SIZE[0] * MIX_TEST_LEVELS, 0.5 * TILE_SIZE[0] * MIX_TEST_LEVELS)
WORLD_Y = (-0.5 * TILE_SIZE[1] * MIX_TEST_LANES, 0.5 * TILE_SIZE[1] * MIX_TEST_LANES)
MIX_PATTERN_END_UNITS = 160.0
MIX_PATTERN_START_X = 0.30
MIX_X_UNIT = 0.02
# `track_mix_terrain` 的 mix 图案前 5 段（0→30→36→42→48→60 索引）是**首尾相接**的 ⇒ 第一处
# 原始深坑的左沿恒为 `pattern_start_x + 60 · x_unit = 1.50 m`（与 difficulty / scale 无关，scale=1.0）。
MIX_FIRST_PIT_X = MIX_PATTERN_START_X + 60.0 * MIX_X_UNIT
# 出生点（瓦片局部 x）与走廊半宽（`corridor_width=0.80` ⇒ ±0.40 m；走廊外是 -0.50 m 坑底）
MIX_SPAWN_X = 0.75
MIX_CORRIDOR_HALF_WIDTH = 0.40
# 训练侧 `Imgo2CMoERoughEnvCfg` 的 reset 平移范围（`pose_range`），用于量化出生点隐患
TRAIN_POSE_RANGE_X = (-0.5, 0.5)
TRAIN_POSE_RANGE_Y = (-0.5, 0.5)

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

    def test_num_cols_twenty_and_num_rows_one(self):
        """**20 条并列的 mix 道**（列 ⇒ 沿世界 Y）× **唯一一行**难度（行 ⇒ 沿世界 X）。"""
        self.assertEqual(
            _literal(self.assign["self.scene.terrain.terrain_generator.num_cols"]), "MIX_TEST_LANES"
        )
        self.assertEqual(
            _literal(self.assign["self.scene.terrain.terrain_generator.num_rows"]), "MIX_TEST_LEVELS"
        )
        # 常量本身必须是 20/1（可读常量只是名字，值要钉住）
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

    # --------------------------------------------------- 难度精确固定 0.70（不依赖 CLI）
    def test_difficulty_range_is_pinned_to_the_constant(self):
        """`terrain_generator.difficulty_range = (d, d)`（上下界同值 ⇒ 抖动被消掉）。"""
        node = self.assign["self.scene.terrain.terrain_generator.difficulty_range"]
        self.assertEqual(
            ast.unparse(node), "(MIX_TEST_DIFFICULTY, MIX_TEST_DIFFICULTY)",
            "必须是常量名对（不是字面量）以便一处改、两处生效",
        )
        self.assertAlmostEqual(_module_constants()["MIX_TEST_DIFFICULTY"], MIX_TEST_DIFFICULTY, places=9)

    def test_max_init_terrain_level_is_zero(self):
        """`num_rows = 1` ⇒ 0 是**唯一**合法等级；显式写 0 让"课程无从升级"自文档。"""
        self.assertEqual(_literal(self.assign["self.scene.terrain.max_init_terrain_level"]), 0)

    def test_terrain_importer_subclass_is_gone_and_class_type_untouched(self):
        """旧"钉第 14 行"的子类已删；`class_type` 不再被本类覆盖（沿用默认 `TerrainImporter`）。"""
        self.assertNotIn("class_type", self.assign, "本类不得再设 `scene.terrain.class_type`")
        self.assertIsNone(_class_def(CMOE_CFG, REMOVED_IMPORTER_CLASS),
                          f"{REMOVED_IMPORTER_CLASS} 必须已删除")
        # 注释里保留"已删除"的说明是可以的（`ast` 看不见注释）；但代码里不得再**引用**这个名字
        tree = ast.parse(self.src)
        referenced = [n.id for n in ast.walk(tree)
                      if isinstance(n, ast.Name) and n.id == REMOVED_IMPORTER_CLASS]
        referenced += [n.attr for n in ast.walk(tree)
                       if isinstance(n, ast.Attribute) and n.attr == REMOVED_IMPORTER_CLASS]
        self.assertEqual(referenced, [], f"代码里不得再引用旧子类：{referenced}")

    def test_single_row_makes_the_curriculum_inherently_frozen(self):
        """`num_rows = 1` 时课程**只有一个合法等级**，逐字复现 Isaac Lab 的两段逻辑来证明。

        * `_compute_env_origins_curriculum`（`terrain_importer.py:334-341`）：
          `max_init_level = min(max_init_terrain_level, num_rows−1) = 0` ⇒ `randint(0, 1) = 0`；
        * `update_env_origins`（`:308-323`）：`level += move_up − move_down` 后
          `where(level >= max_terrain_level(= num_rows = 1), randint_like(level, 1), clip(level, 0))`
          ⇒ **只会得到 0**（`randint_like(..., 1)` 的取值域是 [0, 1)）⇒ 下标 `terrain_origins[0, col]` 恒合法。
        """
        num_rows, max_init = MIX_TEST_LEVELS, 0
        max_init_level = min(max_init, num_rows - 1)
        self.assertEqual(max_init_level, 0)
        self.assertEqual(list(range(0, max_init_level + 1)), [0], "初始等级只能是 0（不是 0..19）")

        max_terrain_level = num_rows  # `self.max_terrain_level = num_rows`
        for level_after_move in (0, 1, 2):
            with self.subTest(level_after_move=level_after_move):
                if level_after_move >= max_terrain_level:
                    wrapped = list(range(0, max_terrain_level))  # randint_like(..., num_rows)
                else:
                    wrapped = [max(level_after_move, 0)]
                self.assertEqual(wrapped, [0], f"升级后仍必须是 0，实测 {wrapped}")
        self.assertLess(0, max_terrain_level, "唯一行下标 0 必须在 [0, num_rows) 内 ⇒ 不会越界")

    def test_isaac_level_logic_matches_the_source(self):
        """上面那条推理必须与**已安装的 Isaac Lab 源码**逐字一致（文件缺失则跳过）。"""
        if not ISAAC_IMPORTER.is_file():
            self.skipTest(f"未安装 Isaac Lab 源码：{ISAAC_IMPORTER}")
        source = ISAAC_IMPORTER.read_text(encoding="utf-8")
        self.assertIn("max_init_level = min(self.cfg.max_init_terrain_level, num_rows - 1)", source)
        self.assertIn("self.terrain_levels = torch.randint(0, max_init_level + 1,", source)
        self.assertIn("self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down", source)
        self.assertIn("self.terrain_levels[env_ids] >= self.max_terrain_level", source)
        self.assertIn("torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level)", source)

    def test_class_type_is_not_touched_for_train_or_play(self):
        """训练/play/评测的 `class_type` 都保持 Isaac Lab 的默认 `TerrainImporter`（本改动不引入子类）。"""
        for name, label in ((TRAIN_CLASS, "训练"), (PLAY_CLASS, "play"), (CFG_CLASS, "mix-test")):
            assigned = _assignments(_post_init(CMOE_CFG, name))
            touched = sorted(t for t in assigned if t.endswith("class_type"))
            self.assertEqual(touched, [], f"{label} 链不得改 class_type，实测 {touched}")
        self.assertIn("max_init_terrain_level = 5", _class_source(PLAY_CLASS),
                      "play 的初始等级上限必须保持 5")

    # --------------------------------------------------- 障碍间距乘子（第四/五批：占满整条道）
    def test_scene_fills_the_whole_lane(self):
        """评测场景：乘子 = **反算的"刚好占满整条 20 m 道"值 6.00** ＋ `fill_stretched_gaps = True`。

        算式（用户第四批："让 mix 占满整条道"；第五批只把 `size[0]` 8 → 20）：
        ``scale = (size[0] − pattern_start_x − 尾部余量) / (160 · x_unit)``
        ``= (20 − 0.30 − 0.50) / (160 × 0.02) = 19.20 / 3.20 = 6.00`` ⇒ 图案末端 **19.50 m**。
        """
        call = self.assign[
            next(k for k in self.assign if re.search(r"\.sub_terrains\['mix'\]$", k))
        ]
        kwargs = _kwargs(call)
        # 乘子：写常量名（一处改、两处生效），常量值 = 反算值
        self.assertIn("pattern_spacing_scale", kwargs)
        self.assertEqual(ast.unparse(kwargs["pattern_spacing_scale"]), "MIX_TEST_PATTERN_SPACING_SCALE")
        consts = _module_constants()
        self.assertEqual(consts["MIX_TEST_PATTERN_SPACING_SCALE"], MIX_TEST_PATTERN_SPACING_SCALE)
        self.assertEqual(MIX_TEST_PATTERN_SPACING_SCALE, 6.00)
        self.assertAlmostEqual(
            consts["MIX_TEST_TAIL_MARGIN"], MIX_TEST_TAIL_MARGIN, places=9
        )
        self.assertAlmostEqual(
            consts["MIX_TEST_PATTERN_END_X"], MIX_TEST_PATTERN_END_X, places=9
        )
        # 反算式自洽：乘子 == (size[0] − pattern_start_x − 尾部余量)/(160·x_unit)
        reverse = (TILE_SIZE[0] - MIX_PATTERN_START_X - MIX_TEST_TAIL_MARGIN) / (
            MIX_PATTERN_END_UNITS * MIX_X_UNIT
        )
        self.assertAlmostEqual(reverse, MIX_TEST_PATTERN_SPACING_SCALE, places=9,
                               msg="乘子必须等于「占满整条道」的反算值")
        # 尾部余量 / 图案末端自洽
        end = MIX_PATTERN_START_X + MIX_PATTERN_END_UNITS * MIX_X_UNIT * MIX_TEST_PATTERN_SPACING_SCALE
        self.assertAlmostEqual(end, MIX_TEST_PATTERN_END_X, places=9)
        self.assertAlmostEqual(TILE_SIZE[0] - end, MIX_TEST_TAIL_MARGIN, places=9)
        # **补空档**字段：写成常量名（教程/审计可读），且常量为 True
        self.assertIn("fill_stretched_gaps", kwargs)
        self.assertEqual(ast.unparse(kwargs["fill_stretched_gaps"]), "MIX_TEST_FILL_STRETCHED_GAPS")
        self.assertIs(_module_constants()["MIX_TEST_FILL_STRETCHED_GAPS"], True)
        self.assertIs(MIX_TEST_FILL_STRETCHED_GAPS, True)
        # 与训练默认值的**有意偏离**只有两处（缺一不可）：
        self.assertNotEqual(_mix_defaults()["pattern_spacing_scale"], MIX_TEST_PATTERN_SPACING_SCALE,
                            "本场景的乘子**不再**等于训练默认值（按用户要求占满整条道）")
        self.assertIs(_mix_defaults()["fill_stretched_gaps"], False,
                      "类默认必须是 False ⇒ 训练与既有测试不受影响")

    def test_default_spacing_and_fill_stay_neutral_for_training(self):
        """硬要求：`CMoETrackMixTerrainCfg` 默认 `pattern_spacing_scale=1.0`、`fill_stretched_gaps=False`；
        训练实例化**两个都不传**。"""
        self.assertEqual(_mix_defaults()["pattern_spacing_scale"], 1.0)
        self.assertIs(_mix_defaults()["fill_stretched_gaps"], False)
        train = _class_source(TRAIN_CLASS)
        call = next(
            node for node in ast.walk(ast.parse(train))
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "CMoETrackMixTerrainCfg"
        )
        written = _kwargs(call)
        self.assertNotIn("pattern_spacing_scale", written,
                         "训练侧不得传间距乘子（取默认 1.0 ⇒ 几何逐位不变）")
        self.assertNotIn("fill_stretched_gaps", written,
                         "训练侧不得传补空档开关（取默认 False ⇒ 几何逐位不变）")

    def test_mix_pattern_fits_the_tile(self):
        """总长核算：`pattern_start_x + 160·x_unit·scale ≤ size[0]`（不静默溢出，上限不放宽）。"""
        call = self.assign[
            next(k for k in self.assign if re.search(r"\.sub_terrains\['mix'\]$", k))
        ]
        kwargs = _kwargs(call)
        offset = float(_literal(kwargs["pattern_start_x"]))
        x_unit = float(_literal(kwargs["x_unit"]))
        scale = MIX_TEST_PATTERN_SPACING_SCALE
        size_x = TILE_SIZE[0]
        end = offset + MIX_PATTERN_END_UNITS * x_unit * scale
        self.assertAlmostEqual(end, 19.50, places=9, msg="占满值下图案末端应在 19.50 m")
        self.assertGreaterEqual(end, MIX_PATTERN_END_MIN_X, "图案末端目标 ≥ 19.0 m（≥ 95 % 的 20 m 道）")
        self.assertLess(end, size_x, f"图案总长 {end} m 必须装得进 {size_x} m 瓦片")
        self.assertAlmostEqual(size_x - end, 0.50, places=9, msg="尾部走廊应剩 0.50 m 平地")
        max_scale = (size_x - offset) / (MIX_PATTERN_END_UNITS * x_unit)
        self.assertAlmostEqual(max_scale, 6.15625, places=9,
                               msg="该瓦片上乘子上限 = (20 − 0.30) / (160 × 0.02) —— 未放宽")
        self.assertLess(scale, max_scale, "选定乘子必须严格小于上限（否则 raise）")
        # 旧的 8 m 道上的上限仍可复现（对照，证明"放宽"不是靠改公式实现的）
        old_max = (8.0 - offset) / (MIX_PATTERN_END_UNITS * x_unit)
        self.assertAlmostEqual(old_max, 2.40625, places=9)

    def test_world_extent_is_twenty_by_eighty(self):
        """世界范围 = 单块尺寸 × 网格：**20 m(X) × 80 m(Y)**（第三批 8 m，第五批放大到 20 m）。"""
        self.assertEqual(WORLD_X, (-10.0, 10.0), "唯一一行 × 20 m ⇒ X ∈ [−10, +10]")
        self.assertEqual(WORLD_Y, (-40.0, 40.0), "20 道 × 4 m ⇒ Y ∈ [−40, +40]")
        rows = _literal(self.assign["self.scene.terrain.terrain_generator.num_rows"])
        cols = _literal(self.assign["self.scene.terrain.terrain_generator.num_cols"])
        self.assertEqual((rows, cols), ("MIX_TEST_LEVELS", "MIX_TEST_LANES"))
        # 出生点在瓦片局部 x = spawn_x；瓦片被 `_get_terrain_mesh` 按 −size/2 居中、整片再按
        # −size·rows/2 居中 ⇒ 唯一一行的世界 X 起点 = −10 ⇒ 出生点 X = spawn_x − 10 = −9.25。
        self.assertAlmostEqual(MIX_SPAWN_X - 10.0, -9.25, places=9)

    def test_tile_size_is_20_for_the_test_and_8_for_training(self):
        """**第五批硬要求**：只有评测 cfg 把 `terrain_generator.size` 改成 `(20, 4)`；
        训练/play 两条链共享的 `size` **仍是 `(8, 4)`**（`size` 是 `terrain_generator` 的共享字段）。"""
        cfg_body = _class_source(CFG_CLASS)
        self.assertIn("self.scene.terrain.terrain_generator.size = MIX_TEST_TILE_SIZE", cfg_body,
                      "评测 cfg 必须显式把 size 覆盖成常量 MIX_TEST_TILE_SIZE")
        self.assertEqual(_module_constants()["MIX_TEST_TILE_SIZE"], TILE_SIZE)
        # 训练类：字面量 (8.0, 4.0)，且 20 行 × 40 列不变
        train = _class_source(TRAIN_CLASS)
        self.assertIn("self.scene.terrain.terrain_generator.size = (8.0, 4.0)", train)
        self.assertIn("self.scene.terrain.terrain_generator.num_rows = 20", train)
        self.assertIn("self.scene.terrain.terrain_generator.num_cols = 40", train)
        # play 类不碰 size（沿用训练值 8×4）
        play = _class_source(PLAY_CLASS)
        self.assertNotIn("terrain_generator.size", play, "play 链不得改 size")
        # 训练类也不得写评测用的 20 m
        self.assertNotIn("(20.0, 4.0)", train, "训练侧不得被评测的 20 m 带偏")
        train_assign = _assignments(_post_init(CMOE_CFG, TRAIN_CLASS))
        self.assertIn("self.scene.terrain.terrain_generator.size", train_assign,
                      "训练类必须显式设自己的 (8, 4)——size 是共享字段，训练侧的值不能被评测场景改掉")
        self.assertEqual(
            ast.literal_eval(train_assign["self.scene.terrain.terrain_generator.size"]),
            TRAIN_TILE_SIZE,
            "训练侧 size 必须仍是 (8, 4)",
        )
        self.assertEqual(TRAIN_TILE_SIZE, (8.0, 4.0))

    # --------------------------------------------- 单局时长（第五批：20 m 道 × 1.0 m/s ⇒ ≥ 25 s）
    def test_episode_length_is_long_enough_for_the_twenty_metre_lane(self):
        """⑤ 单局时长充足性：道 20 m ÷ 恒定 1.0 m/s = 20 s ⇒ 评测 cfg 必须把继承来的
        `episode_length_s`（父类默认 **20 s**，`velocity_env_cfg.py:716`）提到 **≥ 25 s**（当前 35 s）。"""
        consts = _module_constants()
        self.assertAlmostEqual(consts["MIX_TEST_EPISODE_LENGTH_S"], MIX_TEST_EPISODE_LENGTH_S, places=9)
        self.assertAlmostEqual(consts["MIX_TEST_EPISODE_LENGTH_MIN_S"], MIX_TEST_EPISODE_LENGTH_MIN_S,
                               places=9)
        self.assertAlmostEqual(consts["MIX_TEST_FORWARD_SPEED"], MIX_TEST_FORWARD_SPEED, places=9)
        self.assertEqual(MIX_TEST_EPISODE_LENGTH_S, 35.0)
        self.assertEqual(MIX_TEST_EPISODE_LENGTH_MIN_S, 25.0)
        # 常量层断言：≥ 25 s，且 ≥ 走完整条道所需的时间（20 m ÷ 1.0 m/s = 20 s）
        needed = TILE_SIZE[0] / MIX_TEST_FORWARD_SPEED
        self.assertAlmostEqual(needed, 20.0, places=9)
        self.assertGreaterEqual(MIX_TEST_EPISODE_LENGTH_S, MIX_TEST_EPISODE_LENGTH_MIN_S,
                                "单局时长必须 ≥ 25 s")
        self.assertGreaterEqual(MIX_TEST_EPISODE_LENGTH_S, needed, "单局时长必须够走完 20 m")
        self.assertGreater(MIX_TEST_EPISODE_LENGTH_S, needed, "还要留余量（不能刚好卡在走完那一刻）")
        # 赋值必须是常量名（一处改、两处生效）
        self.assertEqual(
            ast.unparse(self.assign["self.episode_length_s"]), "MIX_TEST_EPISODE_LENGTH_S",
            "评测 cfg 必须显式覆盖 episode_length_s（继承来的是 20 s，不够）",
        )
        # 前进速度的字面量与常量一致（`lin_vel_x` 仍写 1.0，常量只作自文档/核算）
        ranges = _kwargs(_kwargs(self.assign["self.commands.base_velocity"])["ranges"])
        self.assertEqual(_literal(ranges["lin_vel_x"]), (MIX_TEST_FORWARD_SPEED, MIX_TEST_FORWARD_SPEED))

    def test_training_and_play_episode_length_are_untouched(self):
        """**训练/play 的时长一字未改**：两条链的 `__post_init__` 都不得写 `self.episode_length_s`。"""
        for name, label in ((TRAIN_CLASS, "训练"), (PLAY_CLASS, "play")):
            with self.subTest(class_name=name):
                assigned = _assignments(_post_init(CMOE_CFG, name))
                self.assertNotIn("self.episode_length_s", assigned,
                                 f"{label} 链不得改单局时长（本批只改评测 cfg）")
                self.assertNotIn("episode_length_s", _class_source(name),
                                 f"{label} 链源码里不应出现 episode_length_s")
        # 继承来的默认值仍是 20 s（父类 `velocity_env_cfg.py`）
        velocity_cfg = (ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/"
                        "velocity_env_cfg.py").read_text(encoding="utf-8")
        self.assertIn("self.episode_length_s = 20.0", velocity_cfg,
                      "父类默认 20 s 是'不够用'的前提，必须仍是 20.0")

    def test_mix_terrain_params_match_training_defaults(self):
        """mix 的地形参数**逐字沿用训练值**（＝ `cmoe_terrains.py` 的类默认值），防两处漂移。

        唯一**有意偏离**的是 `proportion=1.0`（本场景只有这一类）与第四批的
        `pattern_spacing_scale = 2.25` ＋ `fill_stretched_gaps = True`（"占满整条道"＋把拉开的空档铺平）
        —— 逐项单独断言。其余字段（`x_unit`/`z_unit`/`height_scale`/`gap_shrink_units`/
        `corridor_width`/`pit_depth`/`pattern_start_x`/`spawn_x`）与训练默认值**逐一相等**
        ⇒ **障碍自身的几何与训练逐位相同**，评测与训练的差异只在"障碍之间的间距/平地"。
        """
        call = self.assign[
            next(k for k in self.assign if re.search(r"\.sub_terrains\['mix'\]$", k))
        ]
        written = {k: _literal(v) for k, v in _kwargs(call).items()}
        defaults = _mix_defaults()
        deviations = {
            "proportion", "pattern_spacing_scale", "fill_stretched_gaps",
        }
        self.assertEqual(set(written) - deviations, set(defaults) - deviations,
                         f"显式写出的字段应与类字段集合一致：写={sorted(written)} 默认={sorted(defaults)}")
        self.assertAlmostEqual(float(_literal(_kwargs(call)["proportion"])), 1.0, places=9)
        self.assertEqual(ast.unparse(_kwargs(call)["pattern_spacing_scale"]),
                         "MIX_TEST_PATTERN_SPACING_SCALE")
        self.assertEqual(ast.unparse(_kwargs(call)["fill_stretched_gaps"]),
                         "MIX_TEST_FILL_STRETCHED_GAPS")
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
        """**观测/动作契约不变**：新类不得改观测组、动作维度、扫描器或奖励/终止/课程。

        ⚠️ 唯一**有意**新增的非契约赋值是第五批的 `self.scene.terrain.terrain_generator.size`
        （瓦片 X 8 m → 20 m）与 `self.episode_length_s`（20 s → 35 s）—— 它们由
        `test_tile_size_is_20_for_the_test_and_8_for_training` /
        `test_episode_length_is_long_enough_for_the_twenty_metre_lane` 单独钉死，故不在这里的禁用前缀里。
        """
        forbidden = (
            "self.observations", "self.actions", "self.rewards", "self.terminations",
            "self.curriculum", "self.scene.height_scanner", "self.scene.robot",
            "self.decimation", "self.sim",
        )
        touched = sorted(t for t in self.assign if t.startswith(forbidden))
        self.assertEqual(touched, [], f"新类不应触碰观测/动作/奖励等契约：{touched}")
        # 新增的非契约赋值**只允许**这两处（防止以后顺手多改）
        allowed_extra = {
            "self.scene.terrain.terrain_generator.size",
            "self.episode_length_s",
        }
        overrides = {t for t in self.assign if t.startswith((
            "self.scene.terrain.terrain_generator.size", "self.episode_length_s",
        ))}
        self.assertEqual(overrides, allowed_extra,
                         f"非契约覆盖只允许 size / episode_length_s，实测 {sorted(overrides)}")

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


def _pattern_piece_rows(difficulty: float, spacing: float, size=TILE_SIZE) -> list[tuple]:
    """**图案自身**每一段的参照几何（不含领先段 / 尾廊 / 整宽坑底），顺序＝图案顺序。

    与 `_pattern_rows` 用的是同一套公式（`shift = (spacing−1)·start·x_unit`、宽度与 scale 无关）——
    单独抽出来是为了证明「`fill_stretched_gaps=True` 只加 height=0 的补块，图案自身一块不变」。
    """
    defaults = _mix_defaults()
    offset = float(defaults["pattern_start_x"])
    x_unit, z_unit = float(defaults["x_unit"]), float(defaults["z_unit"])
    height_scale = float(defaults["height_scale"])
    gap_shrink = round(float(defaults["gap_shrink_units"]) * (1.0 - difficulty))
    corridor = float(defaults["corridor_width"])
    diff = height_scale * difficulty
    y0, y1 = 0.5 * (size[1] - corridor), 0.5 * (size[1] + corridor)
    rows: list[tuple] = []
    for start, end, height_units in _MIX_PATTERN:
        if isinstance(start, str):
            start = _GAP_BASE_UNITS[start] - gap_shrink
        shift = (spacing - 1.0) * start * x_unit
        x0 = offset + start * x_unit + shift
        x1 = min(offset + end * x_unit + shift, size[0])
        if end <= start or x1 <= x0:
            continue
        rows.append((x0, x1, y0, y1, height_units * z_unit * diff))
    return rows


def _pattern_rows(difficulty: float, spacing: float, size=TILE_SIZE) -> list[tuple]:
    """**参照表**（改动前公式＋间距乘子）：`[(x0, x1, y0, y1, top), ...]`，顺序＝函数的返回顺序。"""
    defaults = _mix_defaults()
    offset = float(defaults["pattern_start_x"])
    x_unit = float(defaults["x_unit"])
    corridor = float(defaults["corridor_width"])
    pit_depth = float(defaults["pit_depth"])
    y0, y1 = 0.5 * (size[1] - corridor), 0.5 * (size[1] + corridor)
    rows = [(0.0, size[0], 0.0, size[1], -pit_depth), (0.0, offset, y0, y1, 0.0)]
    rows += _pattern_piece_rows(difficulty, spacing, size)
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


def _walkable_and_pits(meshes, pit_depth: float = 0.50):
    """把真网格切成**沿 X 的可走面区间**与**坑区间**。

    第 0 块是整宽坑底（`_platform(0, size[0], size[1], -pit_depth)`，顶面 = `-pit_depth`），其余块是
    `_corridor`（可走面，顶面 ≥ 0 或台阶高度）。返回 ``(walkable, pits)``：

    * ``walkable``：``[(x0, x1, top), ...]``，按 x0 升序（相邻可走块相接时仍各自成段）；
    * ``pits``：相邻可走块之间**空出来的** X 区间 ``[(x0, x1), ...]``（顶面 = `-pit_depth` 的坑）。
    """
    rows = sorted(_mesh_rows(meshes), key=lambda r: r[0])
    # 只有整宽坑底那一块的下表面/顶面在 -pit_depth（走道块的顶面恒 ≥ 0）⇒ 用它把两类分开
    walk = [r for r in rows if abs(r[4] + pit_depth) > 1.0e-9]
    walk.sort(key=lambda r: r[0])
    pits = []
    for before, after in zip(walk, walk[1:]):
        if after[0] - before[1] > 1.0e-9:
            pits.append((before[1], after[0]))
    return [(r[0], r[1], r[4]) for r in walk], pits


def _spawn_probe(meshes, spawn_x: float = MIX_SPAWN_X, pit_depth: float = 0.50):
    """出生点的真几何读数：``{on_solid, piece, solid_ahead, first_pit_x, walkable_total}``。"""
    walk, pits = _walkable_and_pits(meshes, pit_depth)
    holders = [w for w in walk if w[0] - 1.0e-9 <= spawn_x <= w[1] + 1.0e-9]
    next_pits = [p for p in pits if p[0] >= spawn_x - 1.0e-9]
    first_pit_x = min(p[0] for p in next_pits) if next_pits else None
    return {
        "on_solid": bool(holders),
        "piece": holders[0] if holders else None,
        "first_pit_x": first_pit_x,
        "solid_ahead": (first_pit_x - spawn_x) if first_pit_x is not None else TILE_SIZE[0] - spawn_x,
        "walkable_total": sum(b - a for a, b, _ in walk),
        "pit_total": sum(b - a for a, b in pits),
        "pits": pits,
        "walkable": walk,
    }


def _pattern_piece_segments(difficulty: float) -> list[tuple[float, float, float]]:
    """**图案参照表**里每一段 `(start, end, height)`（原始 units；`72−k` / `123−k` 按难度还原）。

    来源是手写的 `_MIX_PATTERN`（**独立于** `cmoe_terrains.py` 里那份 `segments`）⇒ 可以当分组与几何的
    第二来源（`TestMixTerrainGeometry.test_segment_table_matches_the_reference` 先钉住两张表一致）。
    """
    defaults = _mix_defaults()
    gap_shrink = round(float(defaults["gap_shrink_units"]) * (1.0 - difficulty))
    out: list[tuple[float, float, float]] = []
    for start, end, height in _MIX_PATTERN:
        if isinstance(start, str):
            start = _GAP_BASE_UNITS[start] - gap_shrink
        if end > start:
            out.append((float(start), float(end), float(height)))
    return out


def _pattern_piece_signature(difficulty: float) -> list[tuple[float, float]]:
    """图案 12 块按 `(宽度, 顶面高度)` 取整的签名（**由手写参照表 ＋ cfg 默认值算出**，与实现无关）。"""
    defaults = _mix_defaults()
    x_unit, z_unit = float(defaults["x_unit"]), float(defaults["z_unit"])
    diff = float(defaults["height_scale"]) * difficulty
    return sorted(
        (round((end - start) * x_unit, 9), round(height * z_unit * diff, 9))
        for start, end, height in _pattern_piece_segments(difficulty)
    )


def _contiguous_groups(segments) -> list[list[tuple[float, float, float]]]:
    """第六批的分组规则：**原始 units 上首尾相接**（`next.start == prev.end`）的连续段归为一组。"""
    groups: list[list[tuple[float, float, float]]] = []
    for segment in segments:
        if groups and abs(segment[0] - groups[-1][-1][1]) <= 1.0e-9:
            groups[-1].append(segment)
        else:
            groups.append([segment])
    return groups


def _grouped_layout(difficulty: float, spacing: float, size=TILE_SIZE):
    """第六批"组内连续、组间拉大"的**独立参照**：返回 `(rows, pits, fills)`。

    * `rows` ＝ 与 `_mesh_rows` 同格式的期望网格表（顺序＝函数的返回顺序）；
    * `pits` ＝ 期望保留的两处坑 `[(x0, x1)]`（**紧贴下游组起点**；宽度 `= 原始空档 · x_unit`，
      与 scale 无关）；
    * `fills` ＝ 期望补出的 `height=0` 可走面 `[(x0, x1)]`（**铺在坑的上游**：上游组末端 → 坑左沿；
      尾段 = 最后一段末端 → 图案末端）。规则与 `track_mix_terrain` docstring 的第六批段落逐条对应：
      `extra = 160 · x_unit · (spacing − 1)` 在 **两处坑所在空档 ＋ 尾段** 之间**均匀分配**。
    """
    defaults = _mix_defaults()
    offset = float(defaults["pattern_start_x"])
    x_unit, z_unit = float(defaults["x_unit"]), float(defaults["z_unit"])
    height_scale = float(defaults["height_scale"])
    corridor = float(defaults["corridor_width"])
    pit_depth = float(defaults["pit_depth"])
    diff = height_scale * difficulty
    y0, y1 = 0.5 * (size[1] - corridor), 0.5 * (size[1] + corridor)

    segments = _pattern_piece_segments(difficulty)
    groups = _contiguous_groups(segments)
    ends = [group[-1][1] for group in groups]
    gaps = [
        (index, group[0][0] - ends[index - 1])
        for index, group in enumerate(groups)
        if index and group[0][0] - ends[index - 1] > 1.0e-9
    ]
    gap_indexes = {index for index, _width in gaps}
    extra_each = (spacing - 1.0) * MIX_PATTERN_END_UNITS * x_unit / (len(gaps) + 1)

    shifts: dict[tuple[float, float], float] = {}
    shift = 0.0
    for index, group in enumerate(groups):
        if index in gap_indexes:
            shift += extra_each
        for start, end, _height in group:
            shifts[(start, end)] = shift

    rows: list[tuple] = [(0.0, size[0], 0.0, size[1], -pit_depth), (0.0, offset, y0, y1, 0.0)]
    for start, end, height in segments:
        x0 = offset + start * x_unit + shifts[(start, end)]
        x1 = min(offset + end * x_unit + shifts[(start, end)], size[0])
        if x1 <= x0:
            continue
        rows.append((x0, x1, y0, y1, height * z_unit * diff))
    tail_x0 = min(offset + MIX_PATTERN_END_UNITS * x_unit * spacing, size[0])
    if tail_x0 < size[0]:
        rows.append((tail_x0, size[0], y0, y1, 0.0))

    pits: list[tuple[float, float]] = []
    fills: list[tuple[float, float]] = []
    for index, width_units in gaps:
        previous_start, previous_end, _previous_height = groups[index - 1][-1]
        pit_upstream_x = offset + previous_end * x_unit + shifts[(previous_start, previous_end)]
        first_start, first_end, _first_height = groups[index][0]
        next_x0 = offset + first_start * x_unit + shifts[(first_start, first_end)]
        # 2026-10-04（第六批追加）：平地铺在**坑的上游**，坑紧贴**下游组起点**。
        pit_x0 = next_x0 - width_units * x_unit
        pits.append((pit_x0, next_x0))
        fills.append((pit_upstream_x, pit_x0))
    last_start, last_end, _last_height = groups[-1][-1]
    fills.append((offset + last_end * x_unit + shifts[(last_start, last_end)], tail_x0))
    for fill_x0, fill_x1 in fills:
        if fill_x1 - fill_x0 > 1.0e-12:
            rows.append((fill_x0, fill_x1, y0, y1, 0.0))
    return rows, pits, fills


def _fill_pieces(meshes, difficulty: float, spacing: float, size=TILE_SIZE):
    """真网格里"补出来的 `height=0` 平地"：与**参照补块表**逐一配对（1e-6 容差）后剩下的多余块。"""
    _rows, _pits, expected_fills = _grouped_layout(difficulty, spacing, size)
    walk, _pits_real = _walkable_and_pits(meshes)
    matched = [False] * len(walk)
    fills = []
    for fill_x0, fill_x1 in expected_fills:
        if fill_x1 - fill_x0 <= 1.0e-12:
            continue
        for index, (x0, x1, _top) in enumerate(walk):
            if matched[index]:
                continue
            if abs(x0 - fill_x0) < 1.0e-6 and abs(x1 - fill_x1) < 1.0e-6:
                matched[index] = True
                fills.append((x0, x1))
                break
        else:
            raise AssertionError(f"参照补块 [{fill_x0}, {fill_x1}] 在真网格里找不到")
    return fills


def _pattern_signature(walk) -> list[tuple[float, float]]:
    """可走块按 `(宽度, 顶面)` 取整的签名（用来核对"图案自身一块不缺、尺寸/高度不变"）。"""
    return sorted((round(x1 - x0, 9), round(top, 9)) for x0, x1, top in walk)


def _missing_signatures(walk, expected: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """`expected` 里有多少个签名在 `walk` 里找不到（多重集意义上的缺项）。"""
    remaining = list(_pattern_signature(walk))
    missing = []
    for item in expected:
        if item in remaining:
            remaining.remove(item)
        else:
            missing.append(item)
    return missing


def _pattern_region_walkable(meshes, spacing: float, size=TILE_SIZE) -> float:
    """**图案区间** `[pattern_start_x, 图案末端]` 内的可走面总长（＝随 scale 单调增的那个量）。

    整块瓦片的可走面总长 = `size[0] − 坑总长`，**与 scale 无关**（尾廊吸收拉伸量）；因此"拉开后
    可走面变多"只能在这个图案区间内度量。
    """
    probe = _spawn_probe(meshes)
    x0 = MIX_PATTERN_START_X
    x1 = min(MIX_PATTERN_START_X + MIX_PATTERN_END_UNITS * MIX_X_UNIT * spacing, size[0])
    total = 0.0
    for start, end, _top in probe["walkable"]:
        low, high = max(start, x0), min(end, x1)
        if high > low:
            total += high - low
    return total


@unittest.skipIf(trimesh is None, f"trimesh unavailable: {TRIMESH_ERROR}")
class TestMixTerrainGeometry(unittest.TestCase):
    """**真跑** `track_mix_terrain`（桩 isaaclab ＋ 真 trimesh）核对间距乘子与溢出保护。

    2026-10-04（第三批）：评测场景的乘子已设回 **1.0**（与训练同值），但 `pattern_spacing_scale`
    字段本体与 `=2.0` 的行为**保留并继续在这里锁定**（中性基础设施：默认 1.0 ⇒ 训练几何逐位不变）。
    """

    @classmethod
    def setUpClass(cls):
        namespace = _load_cmoe_terrains()
        cls.function = staticmethod(namespace["track_mix_terrain"])
        cls.cfg_class = namespace["CMoETrackMixTerrainCfg"]

    def _build(self, spacing, difficulty, size=TILE_SIZE, spawn_x=MIX_SPAWN_X,
               pattern_start_x=MIX_PATTERN_START_X, fill=False):
        cfg = self.cfg_class()
        cfg.size = size
        cfg.proportion = 1.0
        cfg.pattern_spacing_scale = spacing
        cfg.fill_stretched_gaps = fill
        cfg.spawn_x = spawn_x
        cfg.pattern_start_x = pattern_start_x
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
        self.assertIs(cfg.fill_stretched_gaps, False, "补空档开关默认必须是 False（训练不受影响）")
        meshes = self.function(0.70, cfg)[0]
        self._assert_matches_reference(meshes, 1.0, 0.70)

    # ------------------------------------- 第四批：fill_stretched_gaps（把拉开的空档铺成可走面）
    def test_fill_flag_and_defaults_are_neutral(self):
        """① `fill_stretched_gaps=False`（默认）与改动前**逐位相同**；`True` 在 `scale=1.0` 时也一样。

        `scale = 1.0` 没有可拉的余量 ⇒ 补块列表为空 ⇒ 开不开这个开关几何完全相同（浮点上也必须完全相同，
        因为补块的条件是 `fill_x1 − fill_x0 > 1e-12`）。
        """
        self.assertIs(_mix_defaults()["fill_stretched_gaps"], False)
        for difficulty in (0.0, 0.35, 0.70, 0.75, 1.0):
            for size in ((8.0, 4.0), (8.0, 8.0)):
                with self.subTest(difficulty=difficulty, size=size):
                    # 显式 False：与改动前参照逐块一致（＝硬要求）
                    self._assert_matches_reference(
                        self._build(1.0, difficulty, size, fill=False), 1.0, difficulty, size
                    )
                    # True + scale=1.0：与 False 完全相同
                    self.assertEqual(
                        _mesh_rows(self._build(1.0, difficulty, size, fill=True)),
                        _mesh_rows(self._build(1.0, difficulty, size, fill=False)),
                        "scale=1.0 时没有可拉开的余量 ⇒ 开关不应改变任何一块",
                    )

    # =============================== 第六批：把拉伸粒度从"每一段"改成"每个障碍组"
    def test_segment_table_matches_the_reference(self):
        """源码级：`track_mix_terrain` 里的 `segments` 表与本文手写的 `_MIX_PATTERN` 参照表必须逐段一致
        （分组与几何的"第二来源"；段数 12 ⇒ **障碍数量不变**）。"""
        tree = ast.parse(CMOE_TERRAINS.read_text(encoding="utf-8-sig"))
        function = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "track_mix_terrain"
        )
        table = next(
            node.value for node in ast.walk(function)
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "segments"
        )
        namespace = {"gap_shrink": round(10.0 * (1.0 - 0.70))}
        got = [
            tuple(eval(compile(ast.Expression(element), str(CMOE_TERRAINS), "eval"), {}, namespace))
            for element in table.elts
        ]
        expected = []
        for start, end, height in _MIX_PATTERN:
            if isinstance(start, str):
                start = _GAP_BASE_UNITS[start] - namespace["gap_shrink"]
            expected.append((float(start), float(end), float(height)))
        self.assertEqual(got, expected, "`segments` 表与 `_MIX_PATTERN` 参照表必须逐段一致")
        self.assertEqual(len(got), 12, "图案应有 12 段（障碍数量不变）")

    def test_grouping_is_implemented_in_the_source(self):
        """源码级：分组规则（原始 units 首尾相接）与"额外长度按空档数均匀分配"必须真写在函数里。"""
        source = CMOE_TERRAINS.read_text(encoding="utf-8-sig")
        body = source[source.index("def track_mix_terrain"):source.index("class CMoETrackMixTerrainCfg")]
        self.assertIn("abs(segment[0] - groups[-1][-1][1])", body, "分组规则必须写在源码里")
        self.assertIn("extra_each", body, "额外长度必须按空档数均匀分配（而非按段/按占比）")
        self.assertIn("stretched_fills", body)

    def test_contiguous_groups_are_three_and_include_the_staircase(self):
        """① **分组结果**：按"原始 units 上是否首尾相接"切成 **3 组**，组内逐段首尾相接；
        4 级楼梯（30:36 / 36:42 / 42:48 / 48:60）必须落在**同一组**里。

        ⚠️ 与用户预期（"5 组：楼梯组 30..60；69..111；120..160；0..30 起步段；可能的空段"）的差异：
        `0:30` 与 `30:36` 在 units 上同样首尾相接（`30 == 30`）⇒ 按定义属**同一组**（实测 3 组，
        没有"空段"：`d = 0.70` 时每段宽度都 > 0）。这句话**不影响几何**：即便人为把它俩拆成两组，
        二者之间的**原始空档宽度是 0** ⇒ 拿不到任何额外长度、生成结果一字不差
        （`test_first_obstacle_group_is_bitwise_scale_invariant` 锁定前 1.68 m 与 scale 无关）。
        """
        segments = _pattern_piece_segments(0.70)
        self.assertEqual(len(segments), 12)
        groups = _contiguous_groups(segments)
        spans = [(group[0][0], group[-1][1]) for group in groups]
        self.assertEqual(len(groups), 3, f"实测 3 组：{spans}")
        self.assertEqual(spans, [(0.0, 60.0), (69.0, 111.0), (120.0, 160.0)])
        for index, group in enumerate(groups):
            with self.subTest(group=index):
                for before, after in zip(group, group[1:]):
                    self.assertEqual(after[0], before[1], "组内必须首尾相接（否则又会把楼梯拆散）")
        stairs = [segment for segment in groups[0] if segment[0] >= 30.0]
        self.assertEqual([segment[0] for segment in stairs], [30.0, 36.0, 42.0, 48.0])
        self.assertEqual([segment[1] for segment in stairs], [36.0, 42.0, 48.0, 60.0])
        self.assertEqual([segment[2] for segment in stairs], [30.0, 60.0, 90.0, 120.0])

    def test_staircase_is_four_contiguous_levels_at_full_scale(self):
        """**楼梯连续（本轮核心断言，真 trimesh）**：`scale = 6.00` 下 4 级楼梯逐级相接、级间
        **没有平地**、顶面高度序列 = 0.0462 / 0.0924 / 0.1386 / 0.1848。

        对照（改动前第四/五批的"逐段"读数，也就是用户看到的"台阶似乎只有一级"）：
        `[3.90,4.02] 0.0462 / [4.62,4.74] 0.0924 / [5.34,5.46] 0.1386 / [6.06,6.30] 0.1848`，
        相邻级之间各夹 0.60 m 平地。
        """
        walk, _pits = _walkable_and_pits(
            self._build(MIX_TEST_PATTERN_SPACING_SCALE, 0.70, fill=True)
        )
        steps = [piece for piece in walk if piece[0] < 1.6 and piece[2] > 0.04]
        expected = [
            (0.90, 1.02, 0.0462), (1.02, 1.14, 0.0924), (1.14, 1.26, 0.1386), (1.26, 1.50, 0.1848),
        ]
        self.assertEqual(len(steps), 4, f"应有且只有 4 级台阶：{steps}")
        for got, want in zip(steps, expected):
            with self.subTest(want=want):
                self.assertAlmostEqual(got[0], want[0], places=9)
                self.assertAlmostEqual(got[1], want[1], places=9)
                self.assertAlmostEqual(got[2], want[2], places=9, msg="顶面高度序列必须不变")
        # 级与级**严丝合缝**（前一级 x1 ≈ 后一级 x0，只差 trimesh 包围盒的浮点回程）⇒ 级间没有平地
        for before, after in zip(steps, steps[1:]):
            self.assertAlmostEqual(before[1], after[0], places=9,
                                   msg="相邻两级必须首尾相接（级间不得有平地）")
        self.assertEqual(len(_pits), 2, "整条道上只应有原图案那两处坑")
        for before, after in zip(steps, steps[1:]):
            self.assertFalse(any(abs(before[1] - pit[0]) < 1.0e-9 for pit in _pits),
                             "楼梯级间不得出现坑/空档")
        self.assertAlmostEqual(steps[0][0], MIX_PATTERN_START_X + 30.0 * MIX_X_UNIT, places=12)

    def test_pit_width_is_constant_and_the_pit_abuts_the_downstream_group(self):
        """② **坑宽恒 0.18 m**（`d = 0.70`、两处、总 0.36 m 与 scale 无关）；坑**紧贴下游组起点**
        （第六批追加：补出的平地铺在**坑的上游**）；坑前平地 = 均匀分配的一份
        `160·x_unit·(scale−1)/3`（`scale = 6.00` ⇒ 5.3333 m，**≥ 0.5 m**）。"""
        for spacing in (1.0, 1.25, 1.5, 2.0, MIX_TEST_PATTERN_SPACING_SCALE):
            with self.subTest(spacing=spacing):
                meshes = self._build(spacing, 0.70, fill=True)
                walk, pits = _walkable_and_pits(meshes)
                self.assertEqual(len(pits), 2, "只有原图案那两处坑（拉开的空档全被铺平）")
                for pit in pits:
                    self.assertAlmostEqual(pit[1] - pit[0], 0.18, places=9, msg="原地坑宽不变")
                    # 坑右沿必须正好落在下游组第一块的左沿上 ⇒ 坑后没有平地、直接是下游抬高块
                    self.assertEqual(
                        len([piece for piece in walk if abs(piece[0] - pit[1]) < 1.0e-12]), 1,
                        "坑右沿必须紧贴下游组第一块",
                    )
                self.assertAlmostEqual(sum(b - a for a, b in pits), 0.36, places=9)
                # 第一处坑：上游平地 = 上游组（4 级楼梯）末端 1.50 → 坑左沿
                upstream = pits[0][0] - MIX_FIRST_PIT_X
                self.assertAlmostEqual(upstream, 160.0 * MIX_X_UNIT * (spacing - 1.0) / 3.0, places=9,
                                       msg="坑前平地 = 均匀分配的一份")
                if spacing == MIX_TEST_PATTERN_SPACING_SCALE:
                    self.assertGreaterEqual(upstream, 0.5, "本场景（scale=6.00）坑前平地 ≥ 0.5 m")
                if spacing > 1.0:
                    fills = _fill_pieces(meshes, 0.70, spacing)
                    self.assertAlmostEqual(fills[0][1], pits[0][0], places=9,
                                           msg="补块必须正好铺到坑的左沿（坑的上游）")
                    self.assertAlmostEqual(fills[0][0], MIX_FIRST_PIT_X, places=9,
                                           msg="补块从上游组（4 级楼梯）末端起铺")

    def test_slack_is_distributed_evenly_between_the_two_pit_gaps_and_the_tail(self):
        """③ **组间空档的分配**：`extra = 160·x_unit·(scale−1)` 在 **2 处坑所在空档 ＋ 尾段** 之间
        **均匀分配**（3 个空档）；`scale = 6.00` ⇒ 每处 **5.3333 m**，一律是 `height=0` 可走面，
        且**铺在坑的上游**（第六批追加）⇒ 每处坑前 5.3333 m 平地、坑后 0 m。

        为什么选"均匀"而不是"按原始空档宽度占比"：两处坑所在空档的原始宽度各只有 0.18 m，按占比
        分配等于把 16 m 几乎全塞进这两处（各 ~8 m）、尾段分到 0 ⇒ 长距离平地被挤在两处坑后面、
        最后一组紧贴图案末端；均匀分配让三处空档各承担等长平地，行程更均匀，且同样满足"坑宽不变 +
        图案末端 19.50 m"。
        """
        scale = MIX_TEST_PATTERN_SPACING_SCALE
        meshes = self._build(scale, 0.70, fill=True)
        fills = _fill_pieces(meshes, 0.70, scale)
        self.assertEqual(len(fills), 3, "空档数 = 2 处坑所在空档 ＋ 尾段 = 3")
        self.assertEqual([(round(a, 4), round(b, 4)) for a, b in fills],
                         [(1.5, 6.8333), (7.8533, 13.1867), (14.1667, 19.5)])
        lengths = [b - a for a, b in fills]
        for length in lengths:
            self.assertAlmostEqual(length, 16.0 / 3.0, places=9, msg="三处空档必须等长（均匀分配）")
        self.assertAlmostEqual(sum(lengths), 160.0 * MIX_X_UNIT * (scale - 1.0), places=9,
                               msg="补出的平地总长 = 160·x_unit·(scale−1) = 16.00 m")
        # 补块位置：前两处紧跟在**上游组末端**之后、正好铺到**坑的左沿**（＝平地铺在坑的上游）；
        # 第三处（尾段）紧跟在最后一段之后、铺到图案末端。
        walk, pits = _walkable_and_pits(meshes)
        for index, (fill_x0, fill_x1) in enumerate(fills):
            with self.subTest(index=index):
                if index < len(pits):
                    self.assertAlmostEqual(fill_x1, pits[index][0], places=9,
                                           msg="补块必须正好铺到坑的左沿（＝平地铺在坑的上游）")
                    self.assertAlmostEqual(fill_x0, 1.5 if index == 0 else 7.853333333333335, places=9,
                                           msg="补块从上游组末端起铺")
                    self.assertAlmostEqual(pits[index][0] - fill_x0, 16.0 / 3.0, places=9,
                                           msg="每处坑前的平地长度 = 5.3333 m（≥ 0.5 m）")
                else:
                    self.assertAlmostEqual(fill_x0, walk[-2][0], places=9,
                                           msg="尾段补块必须紧跟在最后一段原图案块之后")
        # 坑后落点平地 = 0（坑右沿就是下游组第一块的左沿）
        for pit in pits:
            self.assertTrue(any(abs(piece[0] - pit[1]) < 1.0e-12 for piece in walk))
        self.assertAlmostEqual(walk[-1][0], MIX_TEST_PATTERN_END_X, places=9, msg="尾廊起点 = 图案末端")

    def test_grouped_layout_matches_the_independent_reference(self):
        """④ 真几何 == **独立参照** `_grouped_layout`（组内连续、组间拉大、坑宽不变、均匀分配）：
        `(x0, x1, y0, y1, top)` 逐块相同（顺序也相同），坑区间与补块区间也逐项相同。"""
        for spacing in (1.25, 1.5, 2.0, MIX_TEST_PATTERN_SPACING_SCALE):
            with self.subTest(spacing=spacing):
                rows, pits, fills = _grouped_layout(0.70, spacing)
                meshes = self._build(spacing, 0.70, fill=True)
                got = _mesh_rows(meshes)
                self.assertEqual(len(got), len(rows), "块数必须与参照一致")
                for index, (row, ref) in enumerate(zip(got, rows)):
                    for value, expected in zip(row, ref):
                        self.assertAlmostEqual(value, expected, places=9,
                                               msg=f"第 {index} 块 {row} != 参照 {ref}")
                _walk, real_pits = _walkable_and_pits(meshes)
                self.assertEqual(len(real_pits), len(pits))
                for (a, b), (ra, rb) in zip(real_pits, pits):
                    self.assertAlmostEqual(a, ra, places=9)
                    self.assertAlmostEqual(b, rb, places=9)
                expected_fills = [(a, b) for a, b in fills if b - a > 1.0e-12]
                real_fills = _fill_pieces(meshes, 0.70, spacing)
                self.assertEqual(len(real_fills), len(expected_fills))
                for (a, b), (ra, rb) in zip(real_fills, expected_fills):
                    self.assertAlmostEqual(a, ra, places=9)
                    self.assertAlmostEqual(b, rb, places=9)

    def test_first_obstacle_group_is_bitwise_scale_invariant(self):
        """⑤ **组 1（起步平台 ＋ 4 级楼梯）与 scale 无关**：`scale = 1.0` 与 `6.00` 的前 6 块
        （整宽坑底 ＋ 起步走廊 ＋ 起步块 ＋ 4 级楼梯）逐块**逐位相同**（组 1 的平移量恒为 0）；
        `scale = 6.00` 上"多出来的长度"是从 `1.50 m`（4 级楼梯末端）开始的**坑前平地**。"""
        spacing = MIX_TEST_PATTERN_SPACING_SCALE
        base = _mesh_rows(self._build(1.0, 0.70, fill=True))
        wide = _mesh_rows(self._build(spacing, 0.70, fill=True))
        self.assertEqual(base[0], wide[0], "整宽坑底不随 scale 变")
        self.assertEqual(
            base[:7], wide[:7],
            "前 7 块（整宽坑底 ＋ 起步走廊 ＋ 起步块 ＋ 4 级楼梯）不得随 scale 变",
        )
        fills = _fill_pieces(self._build(spacing, 0.70, fill=True), 0.70, spacing)
        self.assertAlmostEqual(fills[0][0], 1.50, places=9, msg="补块从 4 级楼梯末端起铺（坑的上游）")
        self.assertAlmostEqual(fills[0][1] - fills[0][0], 16.0 / 3.0, places=9)
        self.assertGreater(fills[0][1], base[7][1] + 4.0, "多出来的长度是这段坑前平地")

    def test_fill_false_keeps_the_pre_change_per_segment_layout(self):
        """⑥ **`fill_stretched_gaps=False`（默认）路径逐位不变**：仍与改动前（`b2caad1` 的逐段公式）
        逐块相同 —— 包括 `scale = 6.00` 时"把楼梯拆散 + 空档留成整宽 −0.50 m 深坑"的旧行为。"""
        for spacing in (1.0, 2.0, MIX_TEST_PATTERN_SPACING_SCALE):
            with self.subTest(spacing=spacing):
                self._assert_matches_reference(self._build(spacing, 0.70, fill=False), spacing, 0.70)
        plain = self._build(MIX_TEST_PATTERN_SPACING_SCALE, 0.70, fill=False)
        steps = [
            piece for piece in _walkable_and_pits(plain)[0]
            if piece[0] < 7.0 and 0.04 < piece[2] < 0.185
        ]
        self.assertEqual([(round(a, 4), round(b, 4)) for a, b, _t in steps],
                         [(3.9, 4.02), (4.62, 4.74), (5.34, 5.46), (6.06, 6.3)],
                         "旧行为的读数（4 级楼梯被 0.60 m 平地拆开）—— 这是用户反馈的来源")

    def test_obstacle_geometry_is_scale_independent(self):
        """③ **障碍自身尺寸与 scale 无关**（真几何）：12 个图案块按 `(宽度, 顶面)` 的签名在任意 scale
        下都能在真网格里找到（补块只会额外多出来）；高栏顶面恒 0.2618 m、坑仍是两处 0.18 m。"""
        expected = _pattern_piece_signature(0.70)
        self.assertEqual(len(expected), 12, "图案应有 12 段")
        for spacing in (1.0, 1.25, 1.5, 2.0, MIX_TEST_PATTERN_SPACING_SCALE):
            with self.subTest(spacing=spacing):
                walk, pits = _walkable_and_pits(self._build(spacing, 0.70, fill=True))
                self.assertEqual(_missing_signatures(walk, expected), [],
                                 "图案自身的块一块都不能少（宽度/顶面必须与 scale 无关）")
                self.assertAlmostEqual(max(top for _a, _b, top in walk), 0.2618, places=9,
                                       msg="高栏顶面 = 170·z_unit·height_scale·d（与 scale 无关）")
                self.assertEqual(len(pits), 2, "只有原图案那两处坑（拉开的空档已铺平）")
                self.assertAlmostEqual(pits[0][1] - pits[0][0], 0.18, places=9, msg="第一处坑宽不变")
                self.assertAlmostEqual(sum(b - a for a, b in pits), 0.36, places=9, msg="坑总长不变")

    def test_fill_mode_walkable_grows_in_the_pattern_and_pits_stay_constant(self):
        """②（关键）"只拉间距、不多深坑"：**坑总长恒 0.36 m 不随 scale 增**，图案区间内的可走面
        随 scale **单调增**（增量全部来自补出的 `height=0` 平地）。

        ⚠️ **如实记录一条与字面要求的冲突**：**整块瓦片**的可走面总长恒为 `size[0] − 坑总长 = 19.64 m`
        —— 尾廊会吸收拉伸量（20 m 道：图案末端 3.50 → 19.50 m 时尾廊 16.50 → 0.50 m），所以"整片可走面
        随 scale 增"在几何上不可能成立。随 scale 单调增的是"图案区间内的可走面"与"补出的平地"。
        """
        spacings = (1.0, 1.25, 1.5, 2.0, MIX_TEST_PATTERN_SPACING_SCALE)
        pattern_walkable, pit_totals, fill_totals, lane_walkable = [], [], [], []
        signature = _pattern_piece_signature(0.70)
        for spacing in spacings:
            meshes = self._build(spacing, 0.70, fill=True)
            probe = _spawn_probe(meshes)
            walk, _pits = _walkable_and_pits(meshes)
            self.assertEqual(_missing_signatures(walk, signature), [], "图案自身的块一块都不能少")
            pattern_walkable.append(_pattern_region_walkable(meshes, spacing))
            pit_totals.append(probe["pit_total"])
            fill_totals.append(sum(b - a for a, b in _fill_pieces(meshes, 0.70, spacing)))
            lane_walkable.append(probe["walkable_total"])
        # 图案区间内的可走面单调增
        for smaller, bigger in zip(pattern_walkable, pattern_walkable[1:]):
            self.assertLess(smaller, bigger, f"图案区间内的可走面必须随 scale 单调增：{pattern_walkable}")
        # 坑总长不随 scale 增（恒为两处 0.18 m）
        self.assertEqual([round(v, 9) for v in pit_totals], [0.36] * len(spacings))
        # 补出的平地 = 160·x_unit·(scale−1)
        self.assertEqual(
            [round(v, 6) for v in fill_totals],
            [round(160.0 * MIX_X_UNIT * (s - 1.0), 6) for s in spacings],
        )
        self.assertAlmostEqual(pattern_walkable[0], 2.84, places=9, msg="scale=1.0：图案区间内 3.20 − 0.36")
        self.assertAlmostEqual(pattern_walkable[-1], 18.84, places=9,
                               msg="scale=6.00：图案区间内 19.20 − 0.36")
        self.assertAlmostEqual(fill_totals[-1], 16.00, places=9, msg="补出平地 = 3.2·(6.00−1)")
        # 整块瓦片的可走面总长恒定（尾廊吸收拉伸量）—— 如实锁定（20 m 道 ⇒ 20 − 0.36）
        self.assertEqual([round(v, 9) for v in lane_walkable], [19.64] * len(spacings))

    def test_pattern_ends_past_nineteen_metres(self):
        """④ 图案末端 ≥ 19.0 m（＝≥ 95 % 的 20 m 道）：反算式、常量、真几何三处一致。"""
        scale = MIX_TEST_PATTERN_SPACING_SCALE
        end = MIX_PATTERN_START_X + MIX_PATTERN_END_UNITS * MIX_X_UNIT * scale
        self.assertAlmostEqual(end, MIX_TEST_PATTERN_END_X, places=9)
        self.assertAlmostEqual(end, 19.50, places=9)
        self.assertGreaterEqual(end, MIX_PATTERN_END_MIN_X)
        self.assertGreaterEqual(end, 0.95 * TILE_SIZE[0])
        self.assertAlmostEqual(TILE_SIZE[0] - end, MIX_TEST_TAIL_MARGIN, places=9)
        # 反算式：scale = (size[0] − pattern_start_x − 尾部余量) / (160 · x_unit)
        self.assertAlmostEqual(
            (TILE_SIZE[0] - MIX_PATTERN_START_X - MIX_TEST_TAIL_MARGIN)
            / (MIX_PATTERN_END_UNITS * MIX_X_UNIT),
            scale, places=9,
        )
        # 真几何：尾廊从 19.50 铺到 20.00；被铺平的是尾段空档
        # （最后一段原图案块的末端 14.1667 → 图案末端 19.50，5.3333 m ＝ 16 m / 3 个空档）
        walk, pits = _walkable_and_pits(self._build(scale, 0.70, fill=True))
        self.assertAlmostEqual(walk[-1][0], MIX_TEST_PATTERN_END_X, places=9, msg="尾廊起点 = 图案末端")
        self.assertAlmostEqual(walk[-1][1], TILE_SIZE[0], places=9)
        self.assertAlmostEqual(walk[-3][1], MIX_TEST_PATTERN_END_X - 16.0 / 3.0, places=9,
                               msg="最后一段原图案块（140:160）的末端")
        self.assertAlmostEqual(walk[-3][2], 0.0924, places=9, msg="最后一段顶面（height 索引 60）")
        self.assertAlmostEqual(walk[-2][0], walk[-3][1], places=9, msg="尾段空档从这里开始")
        self.assertAlmostEqual(walk[-2][1], MIX_TEST_PATTERN_END_X, places=9, msg="尾段空档铺到图案末端")
        self.assertAlmostEqual(walk[-2][1] - walk[-2][0], 16.0 / 3.0, places=9,
                               msg="尾段分到 16/3 = 5.3333 m（与两处坑所在空档等长）")
        self.assertEqual(len(pits), 2)

    def test_overflow_guard_is_not_relaxed_by_fill_mode(self):
        """溢出保护**不放宽**：`fill=True` 时超限仍直接 raise，上限仍是 6.15625（20 m 道）。"""
        max_spacing = (TILE_SIZE[0] - MIX_PATTERN_START_X) / (MIX_PATTERN_END_UNITS * MIX_X_UNIT)
        self.assertAlmostEqual(max_spacing, 6.15625, places=9)
        self._build(max_spacing, 0.70, fill=True)  # 不抛
        with self.assertRaises(ValueError) as ctx:
            self._build(max_spacing + 1.0e-4, 0.70, fill=True)
        self.assertIn("超出瓦片长", str(ctx.exception))
        self.assertIn("6.1562", str(ctx.exception), "上限必须仍是 6.15625（未放宽）")
        with self.assertRaises(ValueError):
            self._build(7.0, 0.70, fill=True)

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
        """总长核算（20 m 道）：图案末端 6.70 m（尾部走廊 6.70→20.00）；被拉开的空间是**深坑**（见 docs）。"""
        rows = _mesh_rows(self._build(2.0, 0.70))
        corridors = rows[1:]
        self.assertAlmostEqual(corridors[0][0], 0.00, places=9, msg="起步走廊起点不动")
        self.assertAlmostEqual(corridors[0][1], 0.30, places=9, msg="起步平台宽度 0.30 m 不变")
        self.assertAlmostEqual(corridors[-1][0], 6.70, places=9, msg="尾部走廊起点 = 图案末端 6.70 m")
        self.assertAlmostEqual(corridors[-1][1], TILE_SIZE[0], places=9, msg="尾廊铺到瓦片末端")
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
            self._build(6.25, 0.70)
        message = str(ctx.exception)
        self.assertIn("超出瓦片长", message)
        self.assertIn("6.1562", message, f"应给出该瓦片上的乘子上限：{message}")
        self.assertIn("20.3000", message, f"应给出实际总长：{message}")

    def test_max_fitting_spacing_is_the_boundary(self):
        """上限 6.15625 = (20 − 0.30) / (160 × 0.02)：取它不 raise，再大一点就 raise。"""
        max_spacing = (TILE_SIZE[0] - MIX_PATTERN_START_X) / (MIX_PATTERN_END_UNITS * MIX_X_UNIT)
        self.assertAlmostEqual(max_spacing, 6.15625, places=9)
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


# ============================================ ③.6 难度"确实被固定在 0.70"（离线可核）
def _isaac_curriculum_difficulty(row: int, num_rows: int, difficulty_range, eta: float) -> float:
    """**逐字复现** `terrain_generator.py:255-257`（`_generate_curriculum_terrains`）的难度公式：

        lower, upper = self.cfg.difficulty_range
        difficulty = (sub_row + self.np_rng.uniform()) / self.cfg.num_rows
        difficulty = lower + (upper - lower) * difficulty
    """
    lower, upper = difficulty_range
    difficulty = (row + eta) / num_rows
    return lower + (upper - lower) * difficulty


class TestMixTestDifficultyPinning(unittest.TestCase):
    """难度**确实**被固定在 0.70：两条路径（生成器会传进去的 difficulty / 真跑 `track_mix_terrain`）。

    路径① ＝ 用 cfg 里的 `num_rows = 1` 与 `difficulty_range = (0.70, 0.70)` 走 Isaac Lab 的课程公式
    （把 `U(0,1)` 抽成变量扫描）⇒ 要求**逐块精确 0.70**；
    路径② ＝ 把该 difficulty 真的喂给 `cmoe_terrains.track_mix_terrain`（桩 isaaclab ＋ 真 trimesh），
    核对生成出的几何就是 d = 0.70 那一套（栏高 0.2618 m、第一处坑左沿 1.50 m、坑宽 0.18 m）。
    另有一条**源码级**断言：字段名与语义必须与已安装的 Isaac Lab 一致（缺失则 skip）。
    """

    @classmethod
    def setUpClass(cls):
        cls.assign = _assignments(_post_init(CMOE_CFG, CFG_CLASS))

    def _pinned_range(self):
        node = self.assign["self.scene.terrain.terrain_generator.difficulty_range"]
        self.assertEqual(ast.unparse(node), "(MIX_TEST_DIFFICULTY, MIX_TEST_DIFFICULTY)")
        d = _module_constants()["MIX_TEST_DIFFICULTY"]
        return (d, d)

    # ---------------------------------------------------------------- 路径 ①：公式
    def test_curriculum_formula_yields_exactly_the_constant(self):
        num_rows = MIX_TEST_LEVELS
        difficulty_range = self._pinned_range()
        for eta in (0.0, 1.0e-9, 0.25, 0.5, 0.75, 1.0 - 1.0e-12):
            with self.subTest(eta=eta):
                got = _isaac_curriculum_difficulty(0, num_rows, difficulty_range, eta)
                self.assertAlmostEqual(got, MIX_TEST_DIFFICULTY, places=15,
                                       msg=f"η={eta} 时难度必须精确 {MIX_TEST_DIFFICULTY}，实测 {got}")

    def test_random_branch_is_pinned_too(self):
        """另一条分支（`curriculum=False` 时走 `_generate_random_terrains`）同样精确：
        `U(lower, upper) = U(0.70, 0.70) = 0.70` ⇒ **两条生成路径都不会抖动**。"""
        lower, upper = self._pinned_range()
        self.assertEqual(lower, upper)
        self.assertAlmostEqual(float(np.random.default_rng(0).uniform(lower, upper)), 0.70, places=15)

    def test_formula_jitter_is_eliminated_by_the_zero_width_range(self):
        """抖动被消掉的**机制**：`upper − lower = 0` ⇒ η 对结果**完全没有影响**。"""
        num_rows = MIX_TEST_LEVELS
        difficulty_range = self._pinned_range()
        values = [_isaac_curriculum_difficulty(0, num_rows, difficulty_range, eta)
                  for eta in (0.0, 0.4999, 0.9999)]
        self.assertEqual(len(set(values)), 1, f"同一行内的 η 不应改变难度，实测 {values}")
        # 对照：旧方案（20 行 + 默认 (0,1) 的第 14 行）**有**抖动 ⇒ 实际 d ∈ [0.70, 0.75)
        old = [_isaac_curriculum_difficulty(14, 20, (0.0, 1.0), eta) for eta in (0.0, 0.9999)]
        self.assertAlmostEqual(old[0], 0.70, places=9)
        self.assertGreater(old[1], 0.70)
        self.assertLess(old[1], 0.75)

    def test_generator_difficulty_source_is_the_curriculum_formula(self):
        """**源码级依据**：字段名 `difficulty_range` 与两条分支的公式在已安装的 Isaac Lab 里逐字存在。"""
        if not (ISAAC_GENERATOR.is_file() and ISAAC_GENERATOR_CFG.is_file()):
            self.skipTest(f"未安装 Isaac Lab 源码：{ISAAC_GENERATOR}")
        cfg_src = ISAAC_GENERATOR_CFG.read_text(encoding="utf-8")
        gen_src = ISAAC_GENERATOR.read_text(encoding="utf-8")
        self.assertIn("difficulty_range: tuple[float, float] = (0.0, 1.0)", cfg_src,
                      "字段名/默认值必须还是 `difficulty_range`（本方案的前提）")
        self.assertIn("curriculum: bool = False", cfg_src, "`curriculum` 字段（本仓在 runtime 置 True）")
        self.assertIn("difficulty = (sub_row + self.np_rng.uniform()) / self.cfg.num_rows", gen_src,
                      "课程模式的逐行映射（num_rows=1 ⇒ 0/1 = 0）")
        self.assertIn("lower, upper = self.cfg.difficulty_range", gen_src)
        self.assertIn("difficulty = lower + (upper - lower) * difficulty", gen_src,
                      "上下界同值 ⇒ 这一项恒等于 lower ⇒ 行内 U(0,1) 抖动被消掉")
        self.assertIn("difficulty = self.np_rng.uniform(*self.cfg.difficulty_range)", gen_src,
                      "随机分支同样被 (0.70, 0.70) 钉住")
        # 本仓在 `terrain_levels` 课程存在时会把 `curriculum` 置 True ⇒ 走上面那条课程公式
        vel = (ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/"
               "velocity_env_cfg.py").read_text(encoding="utf-8")
        self.assertIn("self.scene.terrain.terrain_generator.curriculum = True", vel)
        self.assertIn('getattr(self.curriculum, "terrain_levels", None) is not None', vel)

    # ------------------------------------------------- 路径 ②：真跑 track_mix_terrain
    @unittest.skipIf(trimesh is None, f"trimesh unavailable: {TRIMESH_ERROR}")
    def test_track_mix_terrain_at_the_pinned_difficulty(self):
        """把"生成器会传进去的 difficulty"真喂给 `track_mix_terrain`，核对就是 d = 0.70 那套几何。"""
        namespace = _load_cmoe_terrains()
        function, cfg_class = namespace["track_mix_terrain"], namespace["CMoETrackMixTerrainCfg"]
        num_rows, difficulty_range = MIX_TEST_LEVELS, self._pinned_range()
        difficulty = _isaac_curriculum_difficulty(0, num_rows, difficulty_range, 0.5)

        def build(d):
            cfg = cfg_class()
            cfg.size = TILE_SIZE
            cfg.proportion = 1.0
            cfg.pattern_spacing_scale = MIX_TEST_PATTERN_SPACING_SCALE
            cfg.fill_stretched_gaps = MIX_TEST_FILL_STRETCHED_GAPS
            return function(d, cfg)[0]

        rows = sorted(_mesh_rows(build(difficulty)), key=lambda r: r[0])
        walkable = [r for r in rows if abs(r[4] + 0.50) > 1.0e-9]
        # d = 0.70 的签名：栏（170 索引）顶面 = 170 × 0.002 × 1.1 × 0.70 = 0.2618 m
        self.assertAlmostEqual(difficulty, 0.70, places=15)
        self.assertAlmostEqual(max(r[4] for r in walkable), 0.2618, places=9,
                               msg="高栏顶面 = 170·z_unit·height_scale·d")
        # 第一处原始深坑：前 5 段（含 4 级楼梯）首尾相接 ⇒ 上游组末端 = 1.50 m（与 scale 无关）；
        # d = 0.70 ⇒ gap_shrink = round(10·0.3) = 3 ⇒ 坑宽 = (72−3−60)·0.02 = 0.18 m（与 scale 无关）。
        self.assertAlmostEqual(MIX_FIRST_PIT_X, 1.50, places=9, msg="scale=1.0 时第一处坑左沿 1.50 m")
        raw_pits = _walkable_and_pits(build(difficulty))[1]
        first_pits = [(round(a, 4), round(b, 4)) for a, b in raw_pits]
        self.assertEqual(len(first_pits), 2, f"补空档后只应剩两处原图案坑：{first_pits}")
        for pit_x0, pit_x1 in raw_pits:
            self.assertAlmostEqual(pit_x1 - pit_x0, 0.18, places=9, msg=f"坑宽与 scale 无关：{first_pits}")
        # 第六批：补出的平地铺在**坑的上游** ⇒ 第一处坑 = 下游组起点 − 0.18，而下游组起点 =
        #   0.30 + 69×0.02 + (160×0.02×(6.00−1)/3) = 0.30 + 1.38 + 5.3333 = 7.0133 m ⇒ 坑 = [6.8333, 7.0133]
        self.assertAlmostEqual(
            raw_pits[0][0],
            MIX_PATTERN_START_X + 69.0 * MIX_X_UNIT
            + 160.0 * MIX_X_UNIT * (MIX_TEST_PATTERN_SPACING_SCALE - 1.0) / 3.0 - 0.18,
            places=9,
        )
        self.assertAlmostEqual(raw_pits[0][0], 6.833333333333333, places=9)
        self.assertAlmostEqual(raw_pits[1][0], 13.186666666666667, places=9)
        # 与 d = 0.0 / 1.0 明显不同 ⇒ "固定成 0.70"是**有意义**的（不是恰好都一样）
        self.assertNotAlmostEqual(max(r[4] for r in sorted(_mesh_rows(build(0.0)), key=lambda r: r[0])
                                      if abs(r[4] + 0.50) > 1.0e-9), 0.2618, places=9)
        self.assertNotAlmostEqual(max(r[4] for r in sorted(_mesh_rows(build(1.0)), key=lambda r: r[0])
                                      if abs(r[4] + 0.50) > 1.0e-9), 0.2618, places=9)


# ============== ③.7 出生点真几何（第三批的 `fill_stretched_gaps=False` 旧行为，**回归锁定**）
@unittest.skipIf(trimesh is None, f"trimesh unavailable: {TRIMESH_ERROR}")
class TestMixTestSpawnGeometry(unittest.TestCase):
    """用真网格量化 `spawn_x = 0.75` 的位置与"前方实心地面"（用户实测反馈："出生点在一个很大的 gap 上"）。

    ⚠️ 本类锁的是**第四批之前**的行为（`fill_stretched_gaps = False`，即被拉开的空档仍是整宽
    `-0.50 m` 深坑）；第四批之后**评测场景**已改用 `scale = 2.25 + fill_stretched_gaps = True`
    （见 `TestMixTestFilledSpawnGeometry`），这里的读数作为"不补空档会怎样"的**回归对照**保留。

    实测结论（`d = 0.70`，`size = (20,4)`（第五批；8 m 道上这些 X 读数相同、只有"整片可走面"变），
    `pattern_start_x = 0.30`，`fill_stretched_gaps = False`）：

    | scale | 出生点落在 | 前方实心地面 | 第一处坑 |
    |---|---|---|---|
    | 1.0 | `[0.30, 0.90]` 顶面 0.00 m | **0.75 m** | `[1.50, 1.68]`（宽 0.18） |
    | 1.5 | 同上 | 0.15 m | `[0.90, 1.20]`（宽 0.30） |
    | 2.0（第二批用过） | 同上 | 0.15 m | `[0.90, 1.50]`（宽 **0.60**） |

    即：乘子 2.0 时出生点**仍然踩在实心面上**（不是"悬在坑里"），但它离平台前缘只有 0.15 m ——
    机器人机身前缘/前足（约 +0.2 m）已经探进 0.60 m 的整宽深坑 ⇒ 观感与实测都像"出生在一个大 gap 上"。
    """

    @classmethod
    def setUpClass(cls):
        namespace = _load_cmoe_terrains()
        cls.function = staticmethod(namespace["track_mix_terrain"])
        cls.cfg_class = namespace["CMoETrackMixTerrainCfg"]

    def _spawn(self, spacing, difficulty=0.70, fill=False):
        cfg = self.cfg_class()
        cfg.size = TILE_SIZE
        cfg.proportion = 1.0
        cfg.pattern_spacing_scale = spacing
        cfg.fill_stretched_gaps = fill
        return _spawn_probe(self.function(difficulty, cfg)[0])

    def test_scale_one_keeps_the_spawn_on_a_flat_platform(self):
        """`scale = 1.0`（＝训练几何）：出生点在 0.60 m 平平台上，前方 0.75 m 才是第一处坑。"""
        probe = self._spawn(1.0)
        self.assertTrue(probe["on_solid"])
        self.assertAlmostEqual(probe["piece"][0], 0.30, places=9)
        self.assertAlmostEqual(probe["piece"][1], 0.90, places=9)
        self.assertAlmostEqual(probe["piece"][2], 0.0, places=9, msg="平台顶面 = 0（可走）")
        self.assertAlmostEqual(probe["first_pit_x"], MIX_FIRST_PIT_X, places=9)
        self.assertAlmostEqual(probe["solid_ahead"], 0.75, places=9)
        self.assertAlmostEqual(probe["pits"][0][1] - probe["pits"][0][0], 0.18, places=9)

    def test_scale_two_puts_the_spawn_at_the_lip_of_a_sixty_centimetre_pit(self):
        """scale = 2.0（fill=False 的旧行为；场景未用）：出生点仍在实心面上，但前方只剩 0.15 m、
        且第一处坑宽 0.60 m。"""
        probe = self._spawn(2.0)
        self.assertTrue(probe["on_solid"], "2.0 时出生点仍踩在 [0.30, 0.90] 的实心面上")
        self.assertAlmostEqual(probe["piece"][1], 0.90, places=9)
        self.assertAlmostEqual(probe["first_pit_x"], 0.90, places=9, msg="深坑从 0.90 m 就开始")
        self.assertAlmostEqual(probe["solid_ahead"], 0.15, places=9)
        self.assertAlmostEqual(probe["pits"][0][1] - probe["pits"][0][0], 0.60, places=9)
        # 0.15 m < 机器人半长（约 0.2 m）⇒ 前足/机身前缘已经探进坑里：这就是"出生点在一个大 gap 上"的读数
        self.assertLess(probe["solid_ahead"], 0.20)

    def test_scale_one_point_five_is_only_slightly_better(self):
        probe = self._spawn(1.5)
        self.assertAlmostEqual(probe["solid_ahead"], 0.15, places=9)
        self.assertAlmostEqual(probe["pits"][0][1] - probe["pits"][0][0], 0.30, places=9)

    def test_walkable_total_shrinks_and_pit_total_grows_with_scale(self):
        """乘子越大 ⇒ 可走面总长越小、坑总长越大（"拉开间距"实际是"更多/更长的深坑"，与 docs §6.2 一致）。

        20 m 道（第五批）：`可走面 + 坑 = size[0]` ⇒ 可走面 19.64 / 18.04 / 16.44 m，坑 0.36 / 1.96 / 3.56 m
        （坑总长与瓦片长度无关，可走面＝20 − 坑总长）。
        """
        probes = [self._spawn(s) for s in (1.0, 1.5, 2.0)]
        self.assertEqual([round(p["walkable_total"], 4) for p in probes], [19.64, 18.04, 16.44])
        self.assertEqual([round(p["pit_total"], 4) for p in probes], [0.36, 1.96, 3.56])
        for smaller, bigger in zip(probes, probes[1:]):
            self.assertLess(bigger["walkable_total"], smaller["walkable_total"])
            self.assertGreater(bigger["pit_total"], smaller["pit_total"])


# =========================== ③.7b 出生点真几何（第四/六批：分组铺平地之后的 `fill_stretched_gaps=True`）
@unittest.skipIf(trimesh is None, f"trimesh unavailable: {TRIMESH_ERROR}")
class TestMixTestFilledSpawnGeometry(unittest.TestCase):
    """规格：`spawn_x = 0.75` 必须落在**实心可走面**上（读数＝桩 `isaaclab` ＋ **真 `trimesh`** 真跑
    `track_mix_terrain`）。附上"前方（到第一处坑）实心"读数与第四/五批的对照。

    实测（`d = 0.70`、`size = (20,4)`、`pattern_start_x = 0.30`、`fill_stretched_gaps = True`）：

    | scale | 出生点落在 | 到第一级台阶 | 到第一处坑 | 第一处坑 | 坑总长 | 整片可走面 |
    |---|---|---|---|---|---|---|
    | 1.0 | `[0.30, 0.90]` 顶面 0.00 m | 0.15 m | 0.75 m | `[1.50, 1.68]`（宽 0.18） | 0.36 m | 19.64 m |
    | **6.00（本场景）** | `[0.30, 0.90]` 顶面 0.00 m | 0.15 m | **6.08 m** | `[6.8333, 7.0133]`（宽 0.18） | 0.36 m | 19.64 m |

    ⚠️ **第六批改动带来的读数变化（如实记录）**：第四/五批的"逐段"铺平地会把 `0→30` 起步块与
    `30→60` 楼梯之间的空档也铺成平地、并把楼梯整体推后 ⇒ 那时 `scale = 6.00` 的"第一处坑"在
    `6.30 m`（前方实心 5.55 m）。第六批改成"组内连续"后，**起步块与 4 级楼梯同属第 1 组**
    （`0:30` 与 `30:36` 在 units 上首尾相接）⇒ 楼梯不再被推开、前 6 块与 `scale = 1.0` **逐位相同**
    （＝评测的起步段与训练几何完全一致）；多出来的长度铺在**坑的上游**（追加要求）⇒ 第一处坑落在
    `1.50 + 5.3333 = 6.8333 m`、"到第一处坑"= **6.08 m**（仍 ≥ 1.0 m）。

    * **好处**：楼梯终于连成楼梯（本轮目标）、前 1.50 m 与 `scale = 1.0` 逐位相同、每处坑前有
      5.3333 m 平地助跑（追加要求），且"到第一处坑 ≥ 1.0 m"这条第四批规格**继续成立**；
    * **代价**：出生点到第一级台阶只有 **0.15 m**（`spawn_x = 0.75` → 台阶左沿 `0.90`），
      这与 `scale = 1.0`（＝训练几何本身）**完全一致**，不是第六批新引入的隐患（训练侧见 CMOE-17）。

    对照：同样 `scale = 6.00` 但**不补空档**（`fill_stretched_gaps=False`，＝第四批之前的旧路径）时
    空档是整宽 −0.50 m 深坑、第一处"坑"从 `0.90 m` 就开始（前方只剩 **0.15 m**）、整片可走面只有
    `3.64 m`；分组铺平地仍把这段救回来（`test_fill_mode_is_what_keeps_the_spawn_safe_at_the_fill_scale`）。
    """

    @classmethod
    def setUpClass(cls):
        namespace = _load_cmoe_terrains()
        cls.function = staticmethod(namespace["track_mix_terrain"])
        cls.cfg_class = namespace["CMoETrackMixTerrainCfg"]

    def _build(self, spacing, difficulty=0.70, fill=True):
        cfg = self.cfg_class()
        cfg.size = TILE_SIZE
        cfg.proportion = 1.0
        cfg.pattern_spacing_scale = spacing
        cfg.fill_stretched_gaps = fill
        return self.function(difficulty, cfg)[0]

    def _spawn(self, spacing, difficulty=0.70, fill=True):
        return _spawn_probe(self._build(spacing, difficulty, fill))

    # 真几何实测的 X 区间表（x0, x1, 顶面高度；保留 4 位小数 ⇒ 与上面的函数读数逐位对上）
    # 第六批：用真 trimesh 在 **20 m 道**上重跑（`size = (20, 4)`）。`scale = 1.0` 一侧与第五批读数
    # **完全一致**（这就是"默认路径逐位不变"的实测证据）；`scale = 6.00` 一侧按"组内连续、组间拉大、
    # 平地铺在坑的上游"重排：4 级楼梯连在 `0.90..1.50`、坑前平地 `1.50..6.8333`、
    # 第一处坑 `[6.8333, 7.0133]`（紧贴下游组）、第二处坑 `[13.1867, 13.3667]`、
    # 三处补块各 5.3333 m（两处坑前 ＋ 尾段）。
    MEASURED = {
        1.0: {
            "walkable": [
                (0.0, 0.3, 0.0), (0.3, 0.9, 0.0), (0.9, 1.02, 0.0462), (1.02, 1.14, 0.0924),
                (1.14, 1.26, 0.1386), (1.26, 1.5, 0.1848), (1.68, 1.98, 0.1848), (1.98, 2.02, 0.0),
                (2.02, 2.22, 0.1478), (2.22, 2.28, 0.2618), (2.28, 2.52, 0.1848), (2.7, 3.1, 0.1848),
                (3.1, 3.5, 0.0924), (3.5, 20.0, 0.0),
            ],
            "pits": [(1.5, 1.68), (2.52, 2.7)],
            "solid_ahead": 0.75,
        },
        6.00: {
            "walkable": [
                (0.0, 0.3, 0.0), (0.3, 0.9, 0.0), (0.9, 1.02, 0.0462), (1.02, 1.14, 0.0924),
                (1.14, 1.26, 0.1386), (1.26, 1.5, 0.1848), (1.5, 6.8333, 0.0), (7.0133, 7.3133, 0.1848),
                (7.3133, 7.3533, 0.0), (7.3533, 7.5533, 0.1478), (7.5533, 7.6133, 0.2618),
                (7.6133, 7.8533, 0.1848), (7.8533, 13.1867, 0.0), (13.3667, 13.7667, 0.1848),
                (13.7667, 14.1667, 0.0924), (14.1667, 19.5, 0.0), (19.5, 20.0, 0.0),
            ],
            "pits": [(6.8333, 7.0133), (13.1867, 13.3667)],
            "solid_ahead": 6.0833,
        },
    }

    def test_scene_spacing_is_the_reverse_computed_fill_value(self):
        """先钉住"反算值 + 保护不放宽 + 图案末端 ≥ 19.0 m"这三条前提。"""
        max_spacing = (TILE_SIZE[0] - MIX_PATTERN_START_X) / (MIX_PATTERN_END_UNITS * MIX_X_UNIT)
        self.assertAlmostEqual(max_spacing, 6.15625, places=9, msg="溢出上限未放宽")
        self.assertLess(MIX_TEST_PATTERN_SPACING_SCALE, max_spacing,
                        f"{MIX_TEST_PATTERN_SPACING_SCALE} 必须严格小于上限 {max_spacing}")
        self.assertAlmostEqual(
            (TILE_SIZE[0] - MIX_PATTERN_START_X - MIX_TEST_TAIL_MARGIN)
            / (MIX_PATTERN_END_UNITS * MIX_X_UNIT),
            MIX_TEST_PATTERN_SPACING_SCALE, places=9, msg="必须是反算出来的「占满整条道」值",
        )
        end = MIX_PATTERN_START_X + MIX_PATTERN_END_UNITS * MIX_X_UNIT * MIX_TEST_PATTERN_SPACING_SCALE
        self.assertGreaterEqual(end, MIX_PATTERN_END_MIN_X, "图案末端 ≥ 19.0 m")
        # 两张实测表必须覆盖 `{1.0, 反算值}` 两个 scale
        self.assertIn(1.0, self.MEASURED)
        self.assertIn(MIX_TEST_PATTERN_SPACING_SCALE, self.MEASURED)

    def test_spawn_is_safe_at_both_scales(self):
        """⑤ 出生点：`spawn_x = 0.75` 在**两个 scale** 下都落在实心区间 `[0.30, 0.90]`（顶面 0.00 m）
        内；到第一级台阶都是 **0.15 m**（＝0.90 − 0.75）；`scale = 6.00` 下到第一处坑 **6.08 m**
        （≥ 1.0 m，第四批规格继续成立），`scale = 1.0` 下到第一处坑 **0.75 m**（训练几何本身）。

        ⚠️ **如实记录**：`scale = 1.0`（＝训练几何）"到第一处坑"只有 0.75 m；该数字只由图案与
        `pattern_start_x` 决定，要改就会破坏"默认路径逐位不变"与"障碍尺寸不变"的硬要求 ⇒ **本轮不做**。
        训练侧同类隐患（`pose_range ±0.5` ⇒ 35% 的重置会落在台阶段上）见 CMOE-17。
        """
        for spacing, min_ahead in ((1.0, 0.75), (MIX_TEST_PATTERN_SPACING_SCALE, 1.0)):
            with self.subTest(spacing=spacing):
                probe = self._spawn(spacing)
                self.assertTrue(probe["on_solid"], "spawn_x=0.75 必须落在实心可走面上")
                self.assertAlmostEqual(probe["piece"][0], MIX_PATTERN_START_X, places=9)
                self.assertAlmostEqual(probe["piece"][1], 0.90, places=9)
                self.assertAlmostEqual(probe["piece"][2], 0.0, places=9, msg="顶面 = 0（可走）")
                self.assertLessEqual(probe["piece"][0], MIX_SPAWN_X)
                self.assertGreaterEqual(probe["piece"][1], MIX_SPAWN_X)
                self.assertGreaterEqual(probe["solid_ahead"], min_ahead)
        # 具体读数（真几何）
        self.assertAlmostEqual(self._spawn(1.0)["solid_ahead"], 0.75, places=9,
                               msg="scale=1.0（训练几何）：到第一处坑 0.75 m")
        filled = self._spawn(MIX_TEST_PATTERN_SPACING_SCALE)
        self.assertAlmostEqual(filled["solid_ahead"], 6.083333333333333, places=9,
                               msg="scale=6.00：坑前 5.3333 m 平地 ⇒ 第一处坑 6.8333 m（前方 6.08 m）")
        self.assertAlmostEqual(filled["first_pit_x"], MIX_FIRST_PIT_X + 16.0 / 3.0, places=9)
        # 到第一级台阶只剩 0.15 m（＝训练几何读数；"组 1 不动"的直接推论）
        steps = [
            piece for piece in _walkable_and_pits(self._build(MIX_TEST_PATTERN_SPACING_SCALE, 0.70, fill=True))[0]
            if piece[0] < 1.6 and piece[2] > 0.04
        ]
        self.assertAlmostEqual(steps[0][0] - MIX_SPAWN_X, 0.15, places=9)
        # 前 6 块（起步块 ＋ 4 级楼梯）与 scale=1.0 逐位相同
        base = _mesh_rows(self._build(1.0, 0.70, fill=True))
        wide = _mesh_rows(self._build(MIX_TEST_PATTERN_SPACING_SCALE, 0.70, fill=True))
        self.assertEqual(base[:6], wide[:6])
        self.assertAlmostEqual(filled["first_pit_x"], MIX_FIRST_PIT_X + 16.0 / 3.0, places=9,
                               msg="第一处坑 = 4 级楼梯末端 1.50 ＋ 坑前平地 5.3333")
        self.assertAlmostEqual(filled["pits"][0][1] - filled["pits"][0][0], 0.18, places=9)

    def test_fill_mode_is_what_keeps_the_spawn_safe_at_the_fill_scale(self):
        """同一 `scale = 6.00`：**不补空档**（第四批之前的旧路径）时第一处"坑"从 0.90 m 就开始、
        前方只剩 0.15 m、整片可走面只有 3.64 m；分组铺平地后第一处坑退到 6.8333 m（前方 6.08 m）、
        可走面 19.64 m、且每处坑前有 5.3333 m 平地助跑。"""
        without = self._spawn(MIX_TEST_PATTERN_SPACING_SCALE, fill=False)
        self.assertAlmostEqual(without["first_pit_x"], 0.90, places=9, msg="不补时深坑从 0.90 m 就开始")
        self.assertAlmostEqual(without["solid_ahead"], 0.15, places=9)
        self.assertLess(without["solid_ahead"], 0.20, "0.15 m < 机身半长（≈0.2 m）⇒ 前足已探进坑里")
        with_fill = self._spawn(MIX_TEST_PATTERN_SPACING_SCALE)
        self.assertGreater(with_fill["solid_ahead"], without["solid_ahead"] + 1.0,
                           "分组铺平地把第一处坑从 0.90 m 推到 6.8333 m（坑前平地）")
        self.assertAlmostEqual(with_fill["solid_ahead"], 6.083333333333333, places=9)
        self.assertGreater(with_fill["walkable_total"], 19.0, "铺平地后整片可走面 19.64 m")

    def test_measured_x_interval_table(self):
        """真 trimesh 实测的"可走面 / 坑"X 区间表（对照 `scale = 1.0` 与占满值 6.00，20 m 道）—— 回归锁定。"""
        for spacing, expected in self.MEASURED.items():
            with self.subTest(spacing=spacing):
                probe = self._spawn(spacing)
                got_walk = [(round(a, 4), round(b, 4), round(t, 4)) for a, b, t in probe["walkable"]]
                got_pits = [(round(a, 4), round(b, 4)) for a, b in probe["pits"]]
                self.assertEqual(got_walk, expected["walkable"])
                self.assertEqual(got_pits, expected["pits"])
                self.assertAlmostEqual(probe["solid_ahead"], expected["solid_ahead"], places=4)
                self.assertAlmostEqual(probe["pit_total"], 0.36, places=9, msg="坑总长与 scale 无关")
                self.assertAlmostEqual(probe["walkable_total"], 19.64, places=9,
                                       msg="整片可走面 = size[0] − 坑总长（尾廊吸收拉伸量）")


# ================================= ③.8 训练侧出生点隐患（**本轮不改训练**，只量化+登记）
class TestTrainingSpawnHazard(unittest.TestCase):
    """训练侧 mix 的**同类**出生点隐患：`spawn_x = 0.75` ＋ `pose_range`（默认 ±0.5）—— 待决，本轮不改。

    量化（结论来自真几何 + 训练 cfg 的 `pose_range`，见 README 问题表 CMOE-17）：

    * **X**：训练 `scale = 1.0` ⇒ 第一处坑左沿恒为 **1.50 m**（前 5 段首尾相接，与 d 无关）；
      `pose_range x = ±0.5` ⇒ 出生 x ∈ **[0.25, 1.25]** ⇒ **永远不会落在坑上**（越界阈值是
      `spawn_x > 1.00`，即现行 0.75 还差 0.25 m 余量）；但 **x > 0.90（占 35%）会落在抬高的台阶段上**
      （顶面 0.0462 / 0.0924 / 0.1386 m @ d=0.70，d=1.0 时最高 0.198 m），而 `reset_root_state_uniform`
      的 z 只用 `env_origins.z(=0) + pose_range.z(=0)`（**不看局部地形高度**）⇒ 足端会**嵌进台阶**；
      最坏情况 x = 1.25 距坑沿只剩 **0.25 m**（约等于机身半长）。
    * **Y**：mix 走廊只有 **0.80 m** 宽（半宽 0.40 m），而 `pose_range y = ±0.5` 超出半宽 ⇒
      **|Δy| > 0.40 的 20% 重置会让机器人基座横向落在 -0.50 m 的坑底上方**（下落 0.5 m）。
    """

    @classmethod
    def setUpClass(cls):
        cls.assign = _assignments(_post_init(CMOE_CFG, TRAIN_CLASS))

    def _train_pose_range(self):
        key = next(k for k in self.assign if k.endswith("params['pose_range']"))
        return _dict_items(self.assign[key])

    def test_training_pose_range_is_plus_minus_half_a_metre(self):
        ranges = {k: _literal(v) for k, v in self._train_pose_range().items()}
        self.assertEqual(ranges["x"], TRAIN_POSE_RANGE_X, "训练侧 x 抖动 ±0.5（隐患的来源）")
        self.assertEqual(ranges["y"], TRAIN_POSE_RANGE_Y, "训练侧 y 抖动 ±0.5 超出走廊半宽 0.40 m")
        self.assertEqual(ranges["z"], (0.0, 0.0), "z 不做抖动 ⇒ 出生高度完全由 env_origins.z(=0) 决定")

    def test_first_pit_x_is_independent_of_difficulty_at_scale_one(self):
        """第一处坑左沿 = `pattern_start_x + 60·x_unit = 1.50 m`（前 5 段首尾相接），与 d 无关。"""
        self.assertAlmostEqual(MIX_FIRST_PIT_X, 1.50, places=9)
        if trimesh is None:
            self.skipTest(f"trimesh unavailable: {TRIMESH_ERROR}")
        namespace = _load_cmoe_terrains()
        function, cfg_class = namespace["track_mix_terrain"], namespace["CMoETrackMixTerrainCfg"]
        for d in (0.0, 0.70, 1.0):
            with self.subTest(difficulty=d):
                cfg = cfg_class()
                cfg.size = TILE_SIZE
                cfg.proportion = 1.0
                walk, pits = _walkable_and_pits(function(d, cfg)[0])
                self.assertAlmostEqual(pits[0][0], MIX_FIRST_PIT_X, places=9)

    def test_randomised_x_never_reaches_the_pit_but_does_enter_raised_segments(self):
        """x ∈ [0.25, 1.25]：**到不了坑**（阈值 spawn_x > 1.00），但 35% 会落在台阶段上。"""
        max_x = MIX_SPAWN_X + TRAIN_POSE_RANGE_X[1]
        min_x = MIX_SPAWN_X + TRAIN_POSE_RANGE_X[0]
        self.assertAlmostEqual((min_x, max_x), (0.25, 1.25), places=9)
        self.assertLess(max_x, MIX_FIRST_PIT_X, "x 抖动到不了第一处坑（1.50 m）")
        self.assertAlmostEqual(MIX_FIRST_PIT_X - max_x, 0.25, places=9,
                               msg="最坏情况的余量只有 0.25 m（≈机身半长）")
        # 落在抬高段（x > 0.90，第一段 0→30 索引的右沿）上的比例
        frac_on_step = (max_x - (MIX_PATTERN_START_X + 30.0 * MIX_X_UNIT)) / (max_x - min_x)
        self.assertAlmostEqual(frac_on_step, 0.35, places=9)
        # 越界阈值：要真的出生在坑上，需要 spawn_x + 0.5 > 1.50 ⇒ spawn_x > 1.00
        self.assertAlmostEqual(MIX_FIRST_PIT_X - TRAIN_POSE_RANGE_X[1], 1.00, places=9)

    def test_randomised_y_can_leave_the_eighty_centimetre_corridor(self):
        """走廊半宽 0.40 m < y 抖动 0.5 m ⇒ 20% 的重置让基座横向落在 -0.50 m 坑底上方。"""
        half = MIX_CORRIDOR_HALF_WIDTH
        span = TRAIN_POSE_RANGE_Y[1] - TRAIN_POSE_RANGE_Y[0]
        frac_outside = 2.0 * (TRAIN_POSE_RANGE_Y[1] - half) / span
        self.assertAlmostEqual(half, 0.40, places=9)
        self.assertAlmostEqual(frac_outside, 0.20, places=9,
                               msg="P(|Δy| > 0.40) = 20% ⇒ 基座悬在坑底上方（下落 0.5 m）")
        self.assertGreater(TRAIN_POSE_RANGE_Y[1], half, "y 抖动必须超出走廊半宽才谈得上隐患")

    def test_mix_test_scene_disables_that_randomisation(self):
        """评测场景不继承这个隐患：play 链把 `pose_range` 全设成 0 ⇒ 出生点就是走廊中心。"""
        play = _class_source(PLAY_CLASS)
        self.assertIn('"x": (0.0, 0.0)', play)
        self.assertIn('"y": (0.0, 0.0)', play)
        self.assertIn('"z": (0.0, 0.0)', play)
        self.assertNotIn("(-0.5, 0.5)", play)


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
    2026-10-04（第三批）：mix-test 的难度改由 `difficulty_range` 固定（`num_rows=1` ⇒ `--terrain_level`
    只有 0 合法）⇒ 帮助文本同步改成"本参数对该任务**不要**再传"，并写明难度来源。
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
        # 2026-10-04（第三批）：mix-test 的难度不再靠这个参数
        self.assertIn("difficulty_range", help_text, "必须写明 mix-test 的难度来源")
        self.assertIn("0.70", help_text)
        self.assertIn("只有 0 合法", help_text, "num_rows=1 ⇒ 该任务只有等级 0 合法")

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
