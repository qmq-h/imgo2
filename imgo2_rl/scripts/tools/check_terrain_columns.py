#!/usr/bin/env python3
"""离线算出每个地形子类型**实际拿到几列**（只用标准库，不需要 Isaac Lab / GPU）。

为什么需要：地形是按 `proportion` **按累积和切列**分配的，而不是"比例 × 列数"那样四舍五入。
Isaac Lab 的规则是（`isaaclab/terrains/terrain_generator.py:240`）：

    for index in range(num_cols):
        sub_index = argmin_k ( index / num_cols + 0.001 < cumsum(proportions_normalized)[k] )

⇒ 0.05 在 `num_cols=20` 时正好 1 列，但在 `num_cols=10` 时**不一定是 0 或 1**；
改一个 `proportion` 也可能"看起来变了、列数没变"。另外**某个地形拿到 0 列时不会有任何报错**，
而引用它名字的奖励掩码（`free_terrain_names` / `bound_terrain_names` / `no_trot_terrain_names`）
会静默退化成"全都不命中"。这类"改了但没生效"只能靠离线算列数发现。

用法::

    python3 imgo2_rl/scripts/tools/check_terrain_columns.py            # 默认任务（cmoe-rough）当前的列数
    python3 imgo2_rl/scripts/tools/check_terrain_columns.py --cols 20 40
    python3 imgo2_rl/scripts/tools/check_terrain_columns.py --task cmoe-rough
    python3 imgo2_rl/scripts/tools/check_terrain_columns.py --task mix-test
    python3 imgo2_rl/scripts/tools/check_terrain_columns.py --task Imgo2-basemove-rough-cmoe-mix-test

**2026-10-04 起支持"按任务区分"**：新增 `--task`（默认 `cmoe-rough`＝训练/play 的 11 类按比例场景）。
`mix-test`（`Imgo2-basemove-rough-cmoe-mix-test`）是**只有 `mix` 一种地形**的受控测试场景：
上面那张 `MASKED_NAMES` 判据里的其它地形名在该场景**根本不存在** ⇒ 引用它们的掩码项在该场景
**恒为 0**，这是**预期**而不是配置错误。因此该任务走**单独的分支**：仍然要求"唯一地形 `mix` 必须
占满全部列"、`num_cols=20`（＝`MIX_TEST_LANES`，20 条**并列**的 mix 道）、`num_rows=1`（**唯一一行
难度**）、`difficulty_range=(0.70, 0.70)`、`sub_terrains.clear()`，但把"其它地形名 0 列"**明确标注为
预期**，而不是报 ❌。**原任务的判据一字未放宽**（`--task cmoe-rough` 仍要求每一项都 ≥1 列）。

比例来源：基类顺序与默认比例读**已安装的 Isaac Lab** `isaaclab/terrains/config/rough.py`
（`ROUGH_TERRAINS_CFG`）；读不到时退回下面记录的默认值。任务侧的覆盖从
`CMoE_env_cfg.py` 的 `sub_terrains[...]` 赋值里解析（含新增键，新增键按赋值顺序追加到末尾）。
**2026-10-04 起解析按类作用域**（只走指定类的 `__post_init__`）：否则同一文件里
`Imgo2CMoEMixTestEnvCfg` 的 `sub_terrains["mix"] = ...(proportion=1.0)` 会污染训练侧的解析结果。
**2026-10-04（第二批）起还会解析模块级常量**：`num_cols`/`num_rows` 现在写成可读常量
（`MIX_TEST_LANES`/`MIX_TEST_LEVELS`），工具用顶层 `NAME = <字面量>` 求值后再读。
**2026-10-04（第三批）**：mix-test 的网格由 20 行缩到 **1 行**（用户："不需要还保持那么多行"），
难度改由 `terrain_generator.difficulty_range = (0.70, 0.70)` **精确固定**（等价旧"第 14 行"的名义
0.70）⇒ 本工具对 mix-test 增加这一项的检查（`MIX_TEST_DIFFICULTY_EXPECTED`）。

**2026-10-04（第四批）**：mix-test 的 `mix` 图案要**占满整条道**（用户："mix 还是太小了…让 mix
占满整条道"）⇒ 本工具对 mix-test 再增加三项检查（**原任务 `cmoe-rough` 的判据一字未动**）：
① `MIX_TEST_PATTERN_SPACING_SCALE` 必须等于**反算值**（`(size[0] − pattern_start_x − 尾部余量)
/ (160·x_unit)`）；② mix 调用必须显式 `fill_stretched_gaps=True`（把拉开出来的空档铺成 `height=0`
可走面）；③ 图案末端 `pattern_start_x + 160·x_unit·scale` 必须 `≥` 目标下界且 `≤ size[0]`
（溢出保护在运行期仍会 raise，上限未放宽）。

**2026-10-04（第五批）**：用户："整体地形放大，原本是10m长就改成20m长，还是布满，但是障碍数量不变，
设置不变，只把间隔改大" ⇒ 单块瓦片 X **8 m → 20 m**（`terrain_generator.size = (20, 4)`），乘子按同一
反算式重算为 **6.00** ⇒ 图案末端 **19.50 m**（目标下界抬到 **19.0 m**）。因此本工具：
① `terrain_size()` 改成**按类作用域**解析（原来全文件扫描会先撞到训练类的 `(8, 4)`），现在
`terrain_size(MIX_TEST_CLASS) = (20, 4)`、`terrain_size(TRAIN_CLASS) = (8, 4)`；
② mix-test 分支新增"**训练侧 `size` 仍是 (8, 4)**"的核对（用户明确要求：`size` 是共享字段，只有评测
cfg 能改）；③ 期望乘子 2.25 → **6.00**、目标下界 7.0 → **19.0**、上限 2.40625 → **6.15625**；
④ 新增"单局时长"检查：道 20 m ÷ 1.0 m/s = 20 s ⇒ 评测 cfg 的 `episode_length_s` 必须 ≥ 25 s。
`--task cmoe-rough` 的判据与输出**一字未改、一字未放宽**。

**2026-10-04（第六批）**：mix 评测道的「台阶似乎只有一级」修复 ⇒ mix-test 分支的 mix 图案检查改成
「**第六批检查：分组语义**」（组内连续／组间拉大／坑宽 0.18 m／平地铺在坑的上游），另**只报告**
图案里其它「难组合」。**原任务 `cmoe-rough` 的判据与输出一字未改、一字未放宽**（上面的第四/五批
段落保留作历史记录，其对应检查在第七批被替换）。

**2026-10-05（第七批）**：用户说「是有些障碍分到一起了，我说的是障碍本身。gap 和突台也在一起。
我希望是和单独的那个几个场景类似的障碍放在这个整条的（道）上」＋「每个地形要间隔开，而不是混合到
一起放在一个位置」⇒ 评测场景的地形**由参考 `mix` 图案换成复合道**
（`cmoe_terrains.track_composite_terrain` / `CMoETrackCompositeTerrainCfg`；`sub_terrains` 的键名
是 `composite`）。因此本工具对 mix-test 的检查**整体换成复合道**：

* 障碍清单与 **X 区间表**（种类 / `[x0,x1]` / 宽度 / 尺寸参数 / 每段平地长度）；
* **逐对相邻障碍间隔表** + 断言「最小相邻间隔 ≥ 下界」⇒ 打印
  `✅ 最小相邻间隔 2.3636 m ≥ 下界 1.5000 m`（用户硬约束：**所有**相邻对，不只是"坑不紧贴抬高块"）；
* **楼梯上下紧贴**（上 4 级 + 下 4 级首尾相接、级间无平地）＋「楼梯与相邻障碍间隔 = X m ≥ 下界」；
* **占满**整条 20 m（最后一个障碍末端 ＋ 尾段平地 = `size[0]`）；
* **出生点**（`spawn_x = 0.75` 在实心平地上、到第一个障碍 ≥ 1.50 m）；
* 另有一段「**训练侧 `mix` 未受影响**」：`CMoETrackMixTerrainCfg` 默认仍是 `1.0`/`False`、
  12 段图案仍切成 3 组、4 级楼梯仍连续（评测场景已不再引用 mix 图案，相关拉间距几何继续由
  `tests/test_cmoe_mix_test_scene.py` 用真 trimesh 锁定）。
`--task cmoe-rough` 的判据与输出仍然**一字未改、一字未放宽**（原判据的负向对照仍在
`tests/test_check_terrain_columns.py` 里）。
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CMOE_CFG = REPO / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/CMoE_env_cfg.py"

# 上游 ROUGH_TERRAINS_CFG 的键顺序与默认比例（读不到安装文件时的兜底；2026-09-24 校对过）
DEFAULT_BASE = [
    ("pyramid_stairs", 0.2),
    ("pyramid_stairs_inv", 0.2),
    ("boxes", 0.2),
    ("random_rough", 0.2),
    ("hf_pyramid_slope", 0.1),
    ("hf_pyramid_slope_inv", 0.1),
]
ISAACLAB_ROUGH = Path("/root/IsaacLab/source/isaaclab/isaaclab/terrains/config/rough.py")

# 奖励掩码引用的地形名（改配方时同步；名单不存在 ⇒ 掩码静默失效）
MASKED_NAMES = {
    "joint_mirror.free_terrain_names": ("boxes",),
    "joint_mirror.bound_terrain_names": ("gap",),
    "feet_air_time.free_terrain_names": ("boxes", "gap"),
    "feet_height_body.free_terrain_names": ("boxes", "gap"),
    "feet_air_time_variance.free_terrain_names": ("boxes", "gap"),
    "feet_gait.free_terrain_names": ("boxes", "gap"),
    # 2026-09-30：`base_height_flat_l2`（−35，去重力门的高度罚）用的是**白名单**
    # `active_terrain_names=("flat",)` —— 只有平地上生效。同样是"名字写错 ⇒ 静默失效"
    # （`is_env_assigned_to_terrain` 对未登记的名字返回全 False），所以一并纳入本表。
    "base_height_flat_l2.active_terrain_names": ("flat",),
}
# 注意：`lin_pos_y` / `yaw_abs` 的 `terrain_names=()` 表示**全局生效**（2026-09-24 用户决定
# "所有场景都给脱离中心的惩罚"），因此不在上面这张"必须 ≥1 列"的名单里。

# ------------------------------------------------------------------ 2026-10-04：按任务区分
TRAIN_CLASS = "Imgo2CMoERoughEnvCfg"
# 2026-10-04（第五批）：play 类（评测场景的父类）也要能按类解析 `size`/`episode_length_s`
# ⇒ 工具补一个类名常量（原来只有训练类与评测类两个）。
PLAY_CLASS = "Imgo2CMoERoughPlayEnvCfg"
MIX_TEST_CLASS = "Imgo2CMoEMixTestEnvCfg"
MIX_TEST_TASK = "Imgo2-basemove-rough-cmoe-mix-test"
# 任务 id（或短名）→ 规范任务 id
TASK_ALIASES = {
    "cmoe": "cmoe-rough",
    "cmoe-rough": "cmoe-rough",
    "mix-test": MIX_TEST_TASK,
    MIX_TEST_TASK: MIX_TEST_TASK,
}
TASK_CLASS = {"cmoe-rough": TRAIN_CLASS, MIX_TEST_TASK: MIX_TEST_CLASS}
# mix-test 的期望网格（道沿 Y、难度行沿 X；顺序与 `CMoE_env_cfg.py` 的常量名一致，值在此**写死**
# 以便"常量被改错"能被工具发现；常量的字面值由 `tests/test_check_terrain_columns.py` 单独钉住）。
MIX_TEST_LANES_EXPECTED = 20
# 2026-10-04（第三批）：行数 20 → **1**（只留唯一一行难度；19 行原本永远用不到）。
MIX_TEST_LEVELS_EXPECTED = 1
# 唯一一行的难度：由 `terrain_generator.difficulty_range = (d, d)` 精确固定（等价旧"第 14 行"的
# 名义 14/20 = 0.70）。工具解析 `difficulty_range` 并要求上下界都等于该值。
MIX_TEST_DIFFICULTY_EXPECTED = 0.70
# 单块瓦片尺寸期望值（**按类作用域**解析）：评测 cfg = (20, 4)；训练/play = (8, 4) 且**不得被改**。
MIX_TEST_TILE_SIZE_EXPECTED = (20.0, 4.0)
TRAIN_TILE_SIZE_EXPECTED = (8.0, 4.0)
# 单局时长（第五批）：道 20 m ÷ 恒定 1.0 m/s = 20 s ⇒ 评测 cfg 的 `episode_length_s` 必须 ≥ 25 s
# （留余量）。`MIX_TEST_EPISODE_LENGTH_EXPECTED` 是当前选定值（35 s），下界用于判定"够不够"。
MIX_TEST_EPISODE_LENGTH_EXPECTED = 35.0
MIX_TEST_EPISODE_LENGTH_MIN_S = 25.0
MIX_TEST_FORWARD_SPEED = 1.0
# ======================================================================================
# 2026-10-05（第七批）：评测场景的地形由"参考 `mix` 图案"改成**复合道**
# （`cmoe_terrains.track_composite_terrain` ＋ `CMoETrackCompositeTerrainCfg`）⇒ 本工具对 mix-test
# 的检查改成：**障碍清单 / 逐对相邻间隔下界 / 占满整条道 / 出生点**，并打印 X 区间表。
# 下面这些期望值**写死**，以便"cfg 被改错"能被工具发现（常量的字面值由
# `tests/test_check_terrain_columns.py` 单独钉住）。原任务 `--task cmoe-rough` 的判据**一字未放宽**。
# ======================================================================================
# `sub_terrains` 里唯一一项的键名（原来叫 `mix`）。
MIX_TEST_SUB_TERRAIN_KEY = "composite"
# 障碍顺序（＝`cmoe_terrains.COMPOSITE_OBSTACLE_SEQUENCE`）：坑 → 楼梯 → 箱子/台阶块 → 栏 → 坑。
COMPOSITE_SEQUENCE_EXPECTED = ("gap", "stairs", "boxes", "hurdle", "gap")
# d = 0.70 时每个障碍单元的宽度（m）与总宽 —— 全部由各自独立地形的难度律算出：
#   坑 0.12 + 0.70×0.20 = 0.26；楼梯 2×4×0.30 = 2.40；箱子 2×0.44 + 1.30 = 2.18；
#   栏 2×0.096 + 0.64 = 0.832；坑 0.26 ⇒ Σ = 5.932。
COMPOSITE_WIDTHS_EXPECTED = (0.26, 2.40, 2.18, 0.832, 0.26)
COMPOSITE_TOTAL_WIDTH_EXPECTED = 5.932
# 出生平地：`first_obstacle_x = spawn_x + spawn_clearance = 0.75 + 1.50 = 2.25 m`（用户硬约束 ≥ 1.50 m）。
COMPOSITE_SPAWN_X_EXPECTED = 0.75
COMPOSITE_SPAWN_CLEARANCE_MIN = 1.50
COMPOSITE_FIRST_OBSTACLE_X_EXPECTED = 2.25
# 相邻障碍之间的**均匀平地间隔**（反算值）：`(20 − 2.25 − 5.932) / 5 = 11.818 / 5 = **2.3636 m**`
# （5 段＝每个障碍之后各一段，最后一段是尾段）⇒ 整条道刚好占满 20.00 m。
COMPOSITE_SPACING_EXPECTED = 2.3636
# 间隔下界（＝`cmoe_terrains.COMPOSITE_MIN_OBSTACLE_SPACING`，用户硬约束："任意两个相邻障碍之间都必须
# 有 ≥ obstacle_spacing 的平地"）。
COMPOSITE_MIN_SPACING_EXPECTED = 1.50
# 最后一个障碍末端与尾段长度（占满核算）：17.6364 + 2.3636 = 20.0000 m。
COMPOSITE_LAST_OBSTACLE_END_EXPECTED = 17.6364
# 楼梯：上 4 级 + 下 4 级**首尾相接**（中间没有平地），级高 0.05 + 0.70×0.15 = 0.155 m、步深 0.30 m
# ⇒ 楼梯段总长 8 × 0.30 = 2.40 m、峰高 4 × 0.155 = 0.62 m。
COMPOSITE_STAIR_LEVELS_EXPECTED = 4
COMPOSITE_STAIR_STEP_HEIGHT_EXPECTED = 0.155
COMPOSITE_STAIR_PEAK_EXPECTED = 0.62
# `track_mix_terrain` 的图案最后一个索引（**训练侧参考**：评测场景已不再用 mix 图案，这里保留供
# "训练侧未被改"的核算与既有回归测试使用）。
MIX_PATTERN_END_UNITS = 160.0
# `mix` 一侧的"占满 20 m 道"参考乘子（第五/六批的评测取值；**现只作训练侧几何参照**）。
MIX_WIDE_SCALE_EXPECTED = 6.00
# mix 图案的源码（第六批的分组/几何核算从它 AST 解析 `segments` 表与 cfg 默认值；只用标准库）
CMOE_TERRAINS = (REPO / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity"
                 "/base_move/cmoe_terrains.py")
# 2026-10-04（第六批）：分组语义的期望值（**写死**，以便"分组规则被改坏"能被工具发现）。
# 分组 = 把 `segments` 按"原始 units 上是否首尾相接"切开；d = 0.70 时实测 3 组：
#   0→60（起步平台 ＋ 4 级楼梯）、69→111、120→160。
MIX_CONTIGUOUS_GROUPS_EXPECTED = 3
MIX_GROUP_SPANS_EXPECTED = ((0.0, 60.0), (69.0, 111.0), (120.0, 160.0))
# 楼梯必须是同一组里 **首尾相接的 4 级**（级间不得有平地），顶面高度序列固定：
MIX_STAIR_LEVELS_EXPECTED = 4
MIX_STAIR_TOPS_EXPECTED = (0.0462, 0.0924, 0.1386, 0.1848)
# 可拉伸空档数 = 两处坑所在空档 ＋ 尾段 = 3；坑宽恒 0.18 m（d = 0.70）。
MIX_GAP_SLOTS_EXPECTED = 3
MIX_PIT_WIDTH_EXPECTED = 0.18
# 追加要求：补出的平地铺在**坑的上游** ⇒ 每处坑前的平地长度 = 均匀分配的一份（建议 ≥ 0.5 m）。
MIX_UPSTREAM_FLAT_MIN_S = 0.5
# 障碍种类的中文标签（打印 X 区间表/逐对相邻间隔表用）。
COMPOSITE_KIND_CN = {
    "gap": "坑（横向沟）",
    "stairs": "楼梯（上下紧贴）",
    "boxes": "箱子/台阶块",
    "hurdle": "栏（薄横栏）",
}


def _composite_detail(kind: str, params: dict) -> str:
    """把一个障碍单元的尺寸参数压成一行可读文本（打印 X 区间表用）。"""
    if kind == "gap":
        return f"坑宽 {float(params['gap_width']):.4f} m"
    if kind == "stairs":
        return (f"上/下各 {int(params['num_steps'])} 级，级高 {float(params['step_height']):.4f} m、"
                f"步深 {float(params['step_depth']):.2f} m、峰高 {float(params['peak_height']):.4f} m")
    if kind == "boxes":
        return (f"{int(params['count'])} 块，块高 {float(params['block_height']):.4f} m、块长 "
                f"{float(params['block_length']):.4f} m、块间距 {float(params['block_spacing']):.2f} m")
    if kind == "hurdle":
        lo, hi = params["height_range"]
        return (f"{int(params['count'])} 道，栏高 {float(params['bar_height']):.4f} m"
                f"（律区间 [{float(lo):.4f}, {float(hi):.4f}] 的中点）、厚 "
                f"{float(params['bar_length']):.4f} m、间距 {float(params['bar_spacing']):.2f} m")
    return ""

# 类作用域的目标匹配（`ast.unparse` 用单引号，正则同时接受两种引号）
_SUB_PROP = re.compile(r"sub_terrains\[['\"]([a-z_0-9]+)['\"]\]\.proportion$")
_SUB_TERRAIN = re.compile(r"sub_terrains\[['\"]([a-z_0-9]+)['\"]\]$")
_NUM_COLS = re.compile(r"terrain_generator\.num_cols$")
_NUM_ROWS = re.compile(r"terrain_generator\.num_rows$")
_DIFFICULTY_RANGE = re.compile(r"terrain_generator\.difficulty_range$")
_TERRAIN_SIZE = re.compile(r"terrain_generator\.size$")


def module_constants() -> dict[str, object]:
    """`CMoE_env_cfg.py` 顶层的 `NAME = <可求值字面量>` 常量表（2026-10-04 起 num_cols 用常量写）。"""
    return _top_level_constants(CMOE_CFG)


def literal(node: ast.AST, ns: dict[str, object]) -> object:
    """先按字面量求值；失败则用模块常量命名空间求值；再失败退回源码文本（便于报错时看得见原式）。"""
    try:
        return ast.literal_eval(node)
    except Exception:
        pass
    try:
        return eval(compile(ast.Expression(node), str(CMOE_CFG), "eval"), {}, dict(ns))  # noqa: S307
    except Exception:
        return ast.unparse(node)


def base_sub_terrains() -> list[tuple[str, float]]:
    """从已安装的 Isaac Lab 读基类顺序/比例；失败则用兜底值。"""
    if not ISAACLAB_ROUGH.is_file():
        return list(DEFAULT_BASE)
    # 该文件是纯声明式配置：抓 "name": <Ctor>( 与其后的 proportion=...
    src = ISAACLAB_ROUGH.read_text(encoding="utf-8")
    body = src[src.index("sub_terrains={"):]
    names = re.findall(r'"([a-z_0-9]+)":\s*\w+\(', body)
    props = re.findall(r"proportion=([0-9.]+)", body)
    if len(names) != len(props) or not names:
        return list(DEFAULT_BASE)
    return [(name, float(p)) for name, p in zip(names, props)]


def class_post_init(class_name: str) -> ast.FunctionDef | None:
    """取 `CMoE_env_cfg.py` 里指定类的 `__post_init__` 节点（类作用域解析的入口）。"""
    tree = ast.parse(CMOE_CFG.read_text(encoding="utf-8-sig"))
    cls = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name), None)
    if cls is None:
        return None
    return next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__post_init__"), None)


def scene_overrides(class_name: str = TRAIN_CLASS) -> dict[str, object]:
    """只解析指定类的 `__post_init__`，返回 ``{props, num_cols, num_rows, difficulty_range,
    sub_terrain_calls, cleared}``。

    2026-10-04：从"整文件 `ast.walk`"改成"类作用域"，因为同一文件里的 mix-test 类也会写
    `sub_terrains["mix"] = ...(proportion=1.0)`；不隔离就会污染训练侧的解析结果（且是静默的）。
    2026-10-04（第三批）：额外解析 `terrain_generator.difficulty_range`（mix-test 用它精确固定难度）。
    2026-10-04（第四批）：额外记录每个 `sub_terrains[name] = <Cfg>(...)` 调用的**全部 kwargs**
    （`sub_terrain_calls`），用于核对 mix-test 的 `pattern_spacing_scale` / `fill_stretched_gaps`。
    """
    info: dict[str, object] = {
        "props": {}, "num_cols": None, "num_rows": None, "difficulty_range": None,
        "sub_terrain_calls": {}, "cleared": False,
    }
    fn = class_post_init(class_name)
    if fn is None:
        return info
    ns = module_constants()
    props: dict[str, float] = info["props"]  # type: ignore[assignment]
    calls: dict[str, dict[str, object]] = info["sub_terrain_calls"]  # type: ignore[assignment]
    for node in ast.walk(fn):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            # `self.scene.terrain.terrain_generator.sub_terrains.clear()` ⇒ 清空基类地形
            if ast.unparse(node.value.func).endswith("sub_terrains.clear"):
                info["cleared"] = True
            continue
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = ast.unparse(node.targets[0])
        m = _SUB_PROP.search(target)
        if m:
            props[m.group(1)] = float(literal(node.value, ns))  # type: ignore[arg-type]
            continue
        m = _SUB_TERRAIN.search(target)
        if m and isinstance(node.value, ast.Call):
            kwargs = {kw.arg: literal(kw.value, ns) for kw in node.value.keywords if kw.arg is not None}
            calls[m.group(1)] = kwargs
            if "proportion" in kwargs:
                props[m.group(1)] = float(kwargs["proportion"])  # type: ignore[arg-type]
            continue
        if _NUM_COLS.search(target):
            info["num_cols"] = int(literal(node.value, ns))  # type: ignore[arg-type]
            continue
        if _NUM_ROWS.search(target):
            info["num_rows"] = int(literal(node.value, ns))  # type: ignore[arg-type]
            continue
        if _DIFFICULTY_RANGE.search(target):
            value = literal(node.value, ns)
            if isinstance(value, (tuple, list)) and len(value) == 2:
                info["difficulty_range"] = (float(value[0]), float(value[1]))
            continue
    return info


def terrain_size(class_name: str = TRAIN_CLASS,
                 default: tuple[float, float] = TRAIN_TILE_SIZE_EXPECTED) -> tuple[float, float]:
    """**指定类**的 `terrain_generator.size`（只在该类的 `__post_init__` 里找）。

    2026-10-04（第四批）：mix-test 的"图案是否占满整条道"要用它核算 ⇒ 从源码里真读，不用猜。
    2026-10-04（第五批）：`size` 是**共享字段**（父类 `Imgo2CMoERoughEnvCfg` 设 `(8, 4)`，评测类
    显式覆盖成 `(20, 4)`）⇒ 原来的"全文件第一个匹配"会**先撞到训练类**（类定义在前）⇒ 改成
    **按类作用域**解析。这样 `terrain_size(MIX_TEST_CLASS) = (20, 4)`、
    `terrain_size(TRAIN_CLASS) = (8, 4)`，也就能断言"训练侧未被改"。
    """
    fn = class_post_init(class_name)
    if fn is None:
        return default
    ns = module_constants()
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        if not _TERRAIN_SIZE.search(ast.unparse(node.targets[0])):
            continue
        value = literal(node.value, ns)
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return (float(value[0]), float(value[1]))
    return default


def episode_length_s(class_name: str = TRAIN_CLASS) -> float | None:
    """**指定类**的 `episode_length_s` 赋值（只在该类的 `__post_init__` 里找；没写返回 None）。

    2026-10-04（第五批）：道 20 m ÷ 1.0 m/s = 20 s ⇒ 评测场景必须把继承来的 20 s 提到 ≥ 25 s。
    返回 `float`（取值走模块常量求值），没写则 `None`（＝沿用继承来的值）。
    """
    fn = class_post_init(class_name)
    if fn is None:
        return None
    ns = module_constants()
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        if ast.unparse(node.targets[0]) != "self.episode_length_s":
            continue
        value = literal(node.value, ns)
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
    return None


def _top_level_constants(path: Path) -> dict[str, object]:
    """某个源文件顶层 `NAME = <可求值字面量>` 与 `NAME: type = <字面量>` 的常量表。

    两种写法都要收：`CMoE_env_cfg.py` 的 `MIX_TEST_*` 用前者，`cmoe_terrains.py` 的
    `COMPOSITE_*`（第七批新增）用后者（带注解）。
    """
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    ns: dict[str, object] = {}
    for node in tree.body:
        target = value = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target, value = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            target, value = node.target.id, node.value
        if target is None:
            continue
        try:
            ns[target] = eval(  # noqa: S307 - 只求值仓库自己的常量表达式
                compile(ast.Expression(value), str(path), "eval"), {}, dict(ns)
            )
        except Exception:
            continue
    return ns


def cmoe_module_constants() -> dict[str, object]:
    """`cmoe_terrains.py` 顶层的常量表（2026-10-05 起复合道也用它）。"""
    return _top_level_constants(CMOE_TERRAINS)


def cfg_field_defaults(class_name: str) -> dict[str, object]:
    """从 `cmoe_terrains.py` 读某个 `@configclass` 的字段默认值（AST；模块级常量就地求值）。

    **只用标准库** ⇒ 没有 Isaac Lab / trimesh 的机器上也能复核地形几何。
    """
    tree = ast.parse(CMOE_TERRAINS.read_text(encoding="utf-8-sig"))
    ns = cmoe_module_constants()
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name), None)
    if cls is None:
        return {}
    out: dict[str, object] = {}
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            try:
                out[node.target.id] = eval(  # noqa: S307
                    compile(ast.Expression(node.value), str(CMOE_TERRAINS), "eval"), {}, dict(ns)
                )
            except Exception:
                continue
    return out


def mix_terrain_defaults() -> dict[str, float]:
    """`CMoETrackMixTerrainCfg` 的字段默认值（**训练侧**参考；评测场景已改用复合道）。"""
    return cfg_field_defaults("CMoETrackMixTerrainCfg")  # type: ignore[return-value]


def composite_terrain_defaults() -> dict[str, object]:
    """`CMoETrackCompositeTerrainCfg` 的字段默认值（复合道核算的兜底值）。"""
    return cfg_field_defaults("CMoETrackCompositeTerrainCfg")


def composite_units(difficulty: float, kwargs: dict[str, object]) -> list[dict[str, object]]:
    """**只用标准库**重算复合道每个障碍单元的宽度/尺寸（与 `cmoe_terrains` 同一套难度律）。

    `kwargs` 是 `sub_terrains['composite'] = CMoETrackCompositeTerrainCfg(...)` 的实参；缺失的字段用
    类默认值补齐。障碍顺序取自 `cmoe_terrains.COMPOSITE_OBSTACLE_SEQUENCE`（模块级常量）。
    """
    defaults = composite_terrain_defaults()

    def field(name: str):
        value = kwargs.get(name, defaults.get(name))
        if value is None:
            raise KeyError(f"CMoETrackCompositeTerrainCfg 缺少字段 {name}")
        return value

    sequence = tuple(cmoe_module_constants().get("COMPOSITE_OBSTACLE_SEQUENCE",
                                                 COMPOSITE_SEQUENCE_EXPECTED))
    gap_lo, gap_hi = field("gap_width_range")           # type: ignore[misc]
    gap_width = float(gap_lo) + difficulty * (float(gap_hi) - float(gap_lo))
    stair_lo, stair_hi = field("stairs_step_height_range")  # type: ignore[misc]
    step_height = float(stair_lo) + difficulty * (float(stair_hi) - float(stair_lo))
    step_depth = float(field("stairs_step_depth"))
    num_steps = int(field("stairs_num_steps"))
    box_lo, box_hi = field("box_height_range")          # type: ignore[misc]
    box_height = float(box_lo) + difficulty * (float(box_hi) - float(box_lo))
    len_lo, len_hi = field("box_length_range")          # type: ignore[misc]
    box_length = float(len_lo) + difficulty * (float(len_hi) - float(len_lo))
    box_count = int(field("box_count"))
    box_spacing = float(field("box_spacing"))
    hurdle_lo, hurdle_hi = field("hurdle_len_range")    # type: ignore[misc]
    hurdle_length = float(hurdle_lo) + difficulty * (float(hurdle_hi) - float(hurdle_lo))
    hurdle_count = int(field("hurdle_count"))
    spacing_lo, spacing_hi = field("hurdle_spacing_range")  # type: ignore[misc]
    hurdle_gap = float(spacing_lo) + float(field("hurdle_spacing_fraction")) * (
        float(spacing_hi) - float(spacing_lo))
    height_min = float(field("hurdle_height_min_slope")) * difficulty
    height_max = float(field("hurdle_height_max_base")) + float(field("hurdle_height_max_slope")) * difficulty
    hurdle_height = height_min + float(field("hurdle_height_fraction")) * (height_max - height_min)

    units: list[dict[str, object]] = []
    for kind in sequence:
        if kind == "gap":
            units.append({"kind": "gap", "width": gap_width, "params": {"gap_width": gap_width}})
        elif kind == "stairs":
            # 上 num_steps 级 + 下 num_steps 级**首尾相接**（中间没有平地）⇒ 总长 2·n·step_depth，
            # 峰高 n·step_height（级高序列 (1..n)·h、(n−1..0)·h）。
            treads = [round((index + 1) * step_height, 9) for index in range(num_steps)]
            treads += [round((num_steps - index - 1) * step_height, 9) for index in range(num_steps)]
            units.append({
                "kind": "stairs", "width": 2.0 * num_steps * step_depth,
                "params": {"num_steps": num_steps, "step_count_total": 2 * num_steps,
                           "step_height": step_height,
                           "step_depth": step_depth, "peak_height": num_steps * step_height,
                           "tread_tops": tuple(treads)},
            })
        elif kind == "boxes":
            units.append({
                "kind": "boxes", "width": box_count * box_length + (box_count - 1) * box_spacing,
                "params": {"count": box_count, "block_height": box_height,
                           "block_length": box_length, "block_spacing": box_spacing},
            })
        elif kind == "hurdle":
            units.append({
                "kind": "hurdle", "width": hurdle_count * hurdle_length + (hurdle_count - 1) * hurdle_gap,
                "params": {"count": hurdle_count, "bar_length": hurdle_length, "bar_height": hurdle_height,
                           "bar_spacing": hurdle_gap, "height_range": (height_min, height_max)},
            })
        else:  # pragma: no cover
            raise ValueError(f"未知障碍种类 {kind!r}")
    return units


def composite_layout(difficulty: float, kwargs: dict[str, object],
                     size: tuple[float, float]) -> dict[str, object]:
    """**只用标准库**重算复合道布局（与 `cmoe_terrains.composite_track_layout` 同一套算式）。

    返回 `{sequence, units, obstacles, flats, adjacent_gaps, spacing, first_obstacle_x, spawn_x,
    spawn_clearance, total_width, last_obstacle_end, end_x, tail_length}`。真值由
    `tests/test_check_terrain_columns.py` 与"桩 isaaclab ＋ 真 trimesh"的实跑交叉核对。
    """
    defaults = composite_terrain_defaults()

    def field(name: str, fallback):
        return kwargs.get(name, defaults.get(name, fallback))

    units = composite_units(difficulty, kwargs)
    spawn_x = float(field("spawn_x", COMPOSITE_SPAWN_X_EXPECTED))
    clearance = float(field("spawn_clearance", COMPOSITE_SPAWN_CLEARANCE_MIN))
    first_x = spawn_x + clearance
    total_width = sum(float(unit["width"]) for unit in units)
    count = len(units)
    spacing_field = float(field("obstacle_spacing", -1.0))
    if spacing_field >= 0.0:
        spacing = spacing_field
        auto = False
    else:
        spacing = (float(size[0]) - first_x - total_width) / count
        auto = True
    obstacles: list[dict[str, object]] = []
    flats: list[dict[str, object]] = [
        {"kind": "lead_in", "x0": 0.0, "x1": first_x, "length": first_x}
    ]
    cursor = first_x
    for index, unit in enumerate(units):
        obstacles.append({
            "index": index, "kind": unit["kind"], "x0": cursor, "x1": cursor + float(unit["width"]),
            "width": float(unit["width"]), "params": unit["params"],
        })
        cursor += float(unit["width"])
        length = spacing if index < count - 1 else float(size[0]) - cursor
        flats.append({
            "kind": "tail" if index == count - 1 else "between", "after_index": index,
            "x0": cursor, "x1": cursor + length, "length": length,
        })
        cursor += length
    adjacent_gaps = [
        {"pair": (obstacles[index]["kind"], obstacles[index + 1]["kind"]),
         "x0": flats[index + 1]["x0"], "x1": flats[index + 1]["x1"],
         "length": flats[index + 1]["length"]}
        for index in range(count - 1)
    ]
    return {
        "sequence": tuple(unit["kind"] for unit in units),
        "units": units,
        "obstacles": obstacles,
        "flats": flats,
        "adjacent_gaps": adjacent_gaps,
        "spacing": spacing,
        "spacing_is_auto": auto,
        "first_obstacle_x": first_x,
        "spawn_x": spawn_x,
        "spawn_clearance": clearance,
        "total_width": total_width,
        "last_obstacle_end": obstacles[-1]["x1"],
        "end_x": flats[-1]["x1"],
        "tail_length": flats[-1]["length"],
    }


def mix_pattern_segments(difficulty: float) -> list[tuple[float, float, float]]:
    """**AST 解析** `track_mix_terrain` 里的 `segments` 表（`72.0 - gap_shrink` 按该难度就地求值）。

    返回非空段 `(start_units, end_units, height_units)`（顺序＝函数里的顺序）—— 分组与几何的第二来源。
    """
    defaults = mix_terrain_defaults()
    tree = ast.parse(CMOE_TERRAINS.read_text(encoding="utf-8-sig"))
    function = next((n for n in ast.walk(tree)
                     if isinstance(n, ast.FunctionDef) and n.name == "track_mix_terrain"), None)
    if function is None:
        return []
    table = next((node.value for node in ast.walk(function)
                  if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "segments"), None)
    if not isinstance(table, ast.Tuple):
        return []
    ns = {"gap_shrink": round(float(defaults.get("gap_shrink_units", 10.0)) * (1.0 - difficulty))}
    out: list[tuple[float, float, float]] = []
    for element in table.elts:
        try:
            start, end, height = (
                float(v) for v in eval(  # noqa: S307 - 只求值仓库自己的常量表达式
                    compile(ast.Expression(element), str(CMOE_TERRAINS), "eval"),
                    {"__builtins__": {}}, dict(ns),
                )
            )
        except Exception:
            continue
        if end > start:
            out.append((start, end, height))
    return out


def mix_contiguous_groups(segments) -> list[list[tuple[float, float, float]]]:
    """第六批的分组规则：**原始 units 上首尾相接**（`next.start == prev.end`）的连续段归为一组。"""
    groups: list[list[tuple[float, float, float]]] = []
    for segment in segments:
        if groups and abs(segment[0] - groups[-1][-1][1]) <= 1.0e-9:
            groups[-1].append(segment)
        else:
            groups.append([segment])
    return groups


def mix_group_layout(difficulty: float, scale: float) -> dict[str, object]:
    """第六批"组内连续、组间拉大、平地铺在坑的上游"的**独立几何核算**（只用标准库）。

    与 `track_mix_terrain` 的第六批段落逐条对应：`extra = 160·x_unit·(scale−1)` 均匀分给
    "两处坑所在空档 ＋ 尾段"，每处坑宽 `= 原始空档 units · x_unit` 且**紧贴下游组起点**，
    补出的 `height=0` 平地铺在**坑的上游**。返回供打印/判定的几何量。
    """
    defaults = mix_terrain_defaults()
    offset = float(defaults.get("pattern_start_x", 0.30))
    x_unit = float(defaults.get("x_unit", 0.02))
    z_unit = float(defaults.get("z_unit", 0.002))
    height_scale = float(defaults.get("height_scale", 1.1))
    diff = height_scale * difficulty
    segments = mix_pattern_segments(difficulty)
    groups = mix_contiguous_groups(segments)
    ends = [group[-1][1] for group in groups]
    gaps = [(index, group[0][0] - ends[index - 1])
            for index, group in enumerate(groups)
            if index and group[0][0] - ends[index - 1] > 1.0e-9]
    gap_indexes = {index for index, _width in gaps}
    extra_each = (scale - 1.0) * MIX_PATTERN_END_UNITS * x_unit / (len(gaps) + 1)
    shifts: dict[tuple[float, float], float] = {}
    shift = 0.0
    for index, group in enumerate(groups):
        if index in gap_indexes:
            shift += extra_each
        for start, end, _height in group:
            shifts[(start, end)] = shift
    blocks: list[tuple[float, float, float, float, float]] = []   # (start_units, x0, x1, top, height_units)
    for start, end, height in segments:
        x0 = offset + start * x_unit + shifts[(start, end)]
        blocks.append((start, x0, x0 + (end - start) * x_unit, height * z_unit * diff, height))
    pits: list[tuple[float, float]] = []
    fills: list[tuple[float, float]] = []
    for index, width_units in gaps:
        previous_start, previous_end, _previous_height = groups[index - 1][-1]
        upstream_x = offset + previous_end * x_unit + shifts[(previous_start, previous_end)]
        first_start, first_end, _first_height = groups[index][0]
        next_x0 = offset + first_start * x_unit + shifts[(first_start, first_end)]
        pits.append((next_x0 - width_units * x_unit, next_x0))
        fills.append((upstream_x, next_x0 - width_units * x_unit))
    last_start, last_end, _last_height = groups[-1][-1]
    last_x0 = offset + last_end * x_unit + shifts[(last_start, last_end)]
    fills.append((last_x0, offset + MIX_PATTERN_END_UNITS * x_unit * scale))
    # 楼梯 = "包含索引 30 的那一组"（＝第 1 组）里所有**抬高**段（首尾相接，数据驱动、不写死索引）
    staircase_group = next(
        group for group in groups if group[0][0] <= 30.0 <= group[-1][1]
    )
    stair_starts = {segment[0] for segment in staircase_group if segment[2] > 0.0}
    stairs = [(x0, x1, top) for start, x0, x1, top, _height in blocks if start in stair_starts]
    return {
        "groups": groups,
        "spans": [(group[0][0], group[-1][1]) for group in groups],
        "blocks": blocks,
        "stairs": stairs,
        "pits": pits,
        "fills": fills,
        "extra_each": extra_each,
        "pit_widths": [(b - a) for a, b in pits],
    }


def mix_hard_combinations(difficulty: float) -> list[str]:
    """图案里其它"难组合"的**只报告**清单（本轮不改几何，只给上层汇报用）。

    由 `segments` 表 ＋ cfg 默认值算出，不含任何实现细节：
    * ① `height == 0` 的窄段被两块抬高段夹住（可能卡脚）；
    * ② 最高的那块（`170` 索引）前后都是更矮的抬高段（上-下尖峰）；
    * ③ 最后一块抬高段之后直接落到 `height = 0`（落差）。
    """
    defaults = mix_terrain_defaults()
    x_unit = float(defaults.get("x_unit", 0.02))
    z_unit = float(defaults.get("z_unit", 0.002))
    diff = float(defaults.get("height_scale", 1.1)) * difficulty
    segments = mix_pattern_segments(difficulty)
    tops = [height * z_unit * diff for _start, _end, height in segments]
    widths = [(end - start) * x_unit for start, end, _height in segments]
    lines: list[str] = []
    for index in range(1, len(segments) - 1):
        if tops[index] == 0.0 and tops[index - 1] > 0.0 and tops[index + 1] > 0.0:
            lines.append(
                f"① {widths[index] * 100:.0f} cm 窄凹口（{widths[index]:.2f} m，位于 "
                f"{segments[index][0]:g}:{segments[index][1]:g}）夹在 {tops[index - 1]:.4f} m 与 "
                f"{tops[index + 1]:.4f} m 两块之间 ⇒ 可能卡脚（**只报告，本轮不改**）"
            )
    index = max(range(len(tops)), key=lambda i: tops[i])
    lines.append(
        f"② 最高块（{segments[index][0]:g}:{segments[index][1]:g}，顶面 {tops[index]:.4f} m）只有 "
        f"{widths[index]:.2f} m 长，前接 {tops[index - 1]:.4f} m、后接 {tops[index + 1]:.4f} m ⇒ "
        f"上-下尖峰（**只报告，本轮不改**）"
    )
    last = len(tops) - 1
    lines.append(
        f"③ 末尾抬高段（{segments[last][0]:g}:{segments[last][1]:g}）顶面 {tops[last]:.4f} m，之后直接"
        f"落到 0 ⇒ 落差 {tops[last]:.4f} m（尾段平地，**只报告，本轮不改**）"
    )
    return lines


def cmoe_overrides() -> tuple[dict[str, float], dict[str, float]]:
    """返回 (任务侧 proportion 覆盖, num_cols 覆盖)。保持旧签名（＝训练/play 那条链）。"""
    info = scene_overrides(TRAIN_CLASS)
    cols: dict[str, float] = {}
    if info["num_cols"] is not None:
        cols["num_cols"] = info["num_cols"]  # type: ignore[assignment]
    return info["props"], cols  # type: ignore[return-value]


def merged_scene(class_name: str = TRAIN_CLASS) -> tuple[list[tuple[str, float]], dict[str, object]]:
    """基类地形（Isaac Lab rough）+ 任务侧覆盖；若该类 `sub_terrains.clear()` 过 ⇒ 只用覆盖项。"""
    info = scene_overrides(class_name)
    props: dict[str, float] = info["props"]  # type: ignore[assignment]
    if info["cleared"]:
        return [(name, value) for name, value in props.items()], info
    merged: list[tuple[str, float]] = []
    base = base_sub_terrains()
    for name, default in base:
        merged.append((name, props.get(name, default)))
    for name, value in props.items():          # 新增键（如 gap/flat）按赋值顺序追加
        if name not in [n for n, _ in merged]:
            merged.append((name, value))
    return merged, info


def forward_only_names(class_name: str = TRAIN_CLASS) -> tuple[str, ...] | None:
    """读指定类里 `self.commands.base_velocity.forward_only_terrain_names`（没写则 None）。"""
    fn = class_post_init(class_name)
    if fn is None:
        return None
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        if ast.unparse(node.targets[0]) != "self.commands.base_velocity.forward_only_terrain_names":
            continue
        return tuple(ast.literal_eval(node.value))
    return None


def allocate(sub_terrains: list[tuple[str, float]], num_cols: int) -> list[str]:
    """逐字复现 terrain_generator.py:240 的累积和切列规则。"""
    total = sum(p for _, p in sub_terrains)
    norm = [p / total for _, p in sub_terrains]
    cols: list[str] = []
    for index in range(num_cols):
        threshold = index / num_cols + 0.001
        acc = 0.0
        chosen = sub_terrains[-1][0]
        for (name, _), p in zip(sub_terrains, norm):
            acc += p
            if threshold < acc:
                chosen = name
                break
        cols.append(chosen)
    return cols


def report(sub_terrains: list[tuple[str, float]], num_cols: int, label: str) -> dict[str, int]:
    # 2026-09-29：`num_cols == 类数` ⇒ 按**等比例**解释（每类恰好 1 列）。
    # 回放配置（`Imgo2CMoERoughPlayEnvCfg`）就是这么设的：`num_cols = 11` + `sub.proportion = 1.0`
    # （循环赋值，静态读不到）⇒ 若在这里仍按训练比例算，会误报 flat/slope 为 0 列。
    if num_cols == len(sub_terrains):
        sub_terrains = [(name, 1.0) for name, _ in sub_terrains]
        label = f"{label}｜num_cols==类数 ⇒ 等比例（每类 1 列）"
    cols = allocate(sub_terrains, num_cols)
    counts = {name: cols.count(name) for name, _ in sub_terrains}
    print(f"\n=== {label}（num_cols={num_cols}，共 {len(cols)} 列）===")
    for name, _ in sub_terrains:
        print(f"  {name:22s} {counts[name]:>3d} 列")
    zero = [name for name, c in counts.items() if c == 0]
    if zero:
        print(f"  ⚠️ 0 列（引用它的掩码会静默失效）：{', '.join(zero)}")
    return counts


def check_mask_columns(counts: dict[str, int], num_cols: int) -> int:
    """原判据（**不放宽**）：掩码引用的每个地形名都必须存在且 ≥1 列。"""
    problems = 0
    for key, names in MASKED_NAMES.items():
        for name in names:
            if name not in counts:
                print(f"\n❌ 掩码 {key} 引用的地形名 '{name}' 不在 sub_terrains 里 ⇒ 静默失效")
                problems += 1
            elif counts[name] == 0:
                print(f"\n❌ 掩码 {key} 引用的 '{name}' 在 num_cols={num_cols} 时只有 0 列 ⇒ 静默失效")
                problems += 1
    return problems


def mix_test_report() -> tuple[int, list[str]]:
    """`Imgo2-basemove-rough-cmoe-mix-test`（只 mix）的**单独**判定。

    与原判据的区别只有一处、且是**新增的标注**而不是放宽：本场景**只有 mix** ⇒ 其它地形名不在
    `sub_terrains` 里、引用它们的掩码项在本场景**恒为 0**（`is_env_assigned_to_terrain` 对未登记的
    名字返回全 False），这里把每一处**逐条打印成"预期"**；同时仍然强制：`sub_terrains.clear()` 过、
    只有 mix、mix 占满全部道、`num_cols=20`（＝`MIX_TEST_LANES`，20 条**并列**的 mix 道）、
    `num_rows=1`（**唯一一行难度**，2026-10-04 第三批：19 行原本永远用不到）、
    `difficulty_range=(0.70, 0.70)`（精确固定该行难度）。
    原任务（`--task cmoe-rough`）走 :func:`check_mask_columns`，判据一字未改。

    2026-10-04（第二批）：`num_cols` 由 **1 → 20**（用户："地形不要按照列排，放在行里面"）——20 条
    mix 道沿世界 Y 并排，难度沿世界 X；因此这里期望的列数也改成 20，且要求"每道恰好 1 列"。
    2026-10-04（第三批，用户："我不需要还保持那么多行，我需要他们并列"）：行数 **20 → 1**、世界
    X 由 160 m 缩到 8 m；难度不再靠"钉第 14 行"，改为本函数新检查的
    `difficulty_range = (0.70, 0.70)`。
    """
    info = scene_overrides(MIX_TEST_CLASS)
    props: dict[str, float] = info["props"]  # type: ignore[assignment]
    num_cols = info["num_cols"] if info["num_cols"] is not None else MIX_TEST_LANES_EXPECTED
    # 单块瓦片尺寸：**按类作用域**解析（评测 cfg 第五批覆盖成 (20,4)；训练/play 仍是 (8,4)）。
    size = terrain_size(MIX_TEST_CLASS)
    train_size = terrain_size(TRAIN_CLASS)
    lines: list[str] = []
    problems = 0

    def say(text: str = ""):
        lines.append(text)
        print(text)

    say(f"\n=== [{MIX_TEST_TASK}] 受控测试场景（只**复合道**；cfg 类 {MIX_TEST_CLASS}）===")
    if not info["cleared"]:
        say("  ❌ 该任务必须先 `sub_terrains.clear()` 再只放复合道（否则会继承基类的 11 类地形）")
        problems += 1
    if list(props.keys()) != [MIX_TEST_SUB_TERRAIN_KEY]:
        say(f"  ❌ 该任务的 sub_terrains 必须**有且只有** `{MIX_TEST_SUB_TERRAIN_KEY}`"
            f"（复合道），实测 {list(props.keys())}")
        problems += 1
    elif props[MIX_TEST_SUB_TERRAIN_KEY] != 1.0:
        say(f"  ❌ 复合道的 proportion 应为 1.0，实测 {props[MIX_TEST_SUB_TERRAIN_KEY]}")
        problems += 1
    if info["num_cols"] != MIX_TEST_LANES_EXPECTED:
        say(f"  ❌ num_cols 应为 {MIX_TEST_LANES_EXPECTED}（＝`MIX_TEST_LANES`：20 条并列的道，"
            f"道数＝可同时评估的环境数上限），实测 {info['num_cols']}")
        problems += 1
    if info["num_rows"] != MIX_TEST_LEVELS_EXPECTED:
        say(f"  ❌ num_rows 应为 {MIX_TEST_LEVELS_EXPECTED}（＝`MIX_TEST_LEVELS`：只留**唯一一行**难度，"
            f"沿世界 +X ⇒ 世界 X 只有 {size[0]:g} m），实测 {info['num_rows']}")
        problems += 1
    difficulty_range = info["difficulty_range"]
    if difficulty_range != (MIX_TEST_DIFFICULTY_EXPECTED, MIX_TEST_DIFFICULTY_EXPECTED):
        say(f"  ❌ difficulty_range 应为 ({MIX_TEST_DIFFICULTY_EXPECTED}, "
            f"{MIX_TEST_DIFFICULTY_EXPECTED})（上下界同值 ⇒ 课程公式里 `upper−lower = 0`，"
            f"行内 U(0,1) 抖动被消掉 ⇒ 每块瓦片精确 d = "
            f"{MIX_TEST_DIFFICULTY_EXPECTED}），实测 {difficulty_range}")
        problems += 1

    names = list(props.keys())
    if not names:
        # 防御：`sub_terrains` 解析为空（cfg 被改坏 / 无赋值）时不要崩在 `report()` 里，
        # 而是明确报错（**这是新增的健壮性，不是放宽既有判据**）。
        say("  ❌ 解析不到任何 `sub_terrains` 赋值 ⇒ 无法核算列数（cfg 被改坏？）")
        problems += 1
        counts = {}
    else:
        counts = report([(n, props[n]) for n in names], num_cols,
                        f"{MIX_TEST_TASK}｜只 {MIX_TEST_SUB_TERRAIN_KEY}（20 道并列）")
    for name in names:
        if counts[name] < 1:
            say(f"  ❌ 唯一地形 '{name}' 在 num_cols={num_cols} 下 0 列 ⇒ 命令与掩码都会静默失效")
            problems += 1
        elif counts[name] != MIX_TEST_LANES_EXPECTED:
            say(f"  ❌ 唯一地形 '{name}' 应占满全部 {MIX_TEST_LANES_EXPECTED} 道（每道 1 列），"
                f"实测 {counts[name]} 列")
            problems += 1
    if not problems:
        say(f"  ✅ 唯一地形 {MIX_TEST_SUB_TERRAIN_KEY} 有 "
            f"{counts.get(MIX_TEST_SUB_TERRAIN_KEY, 0)} 列（num_cols={num_cols}、"
            f"num_rows={info['num_rows']}）"
            f"⇒ 网格 **{MIX_TEST_LANES_EXPECTED} 道 × {MIX_TEST_LEVELS_EXPECTED} 难度行**"
            f"（世界 {size[0]:g} m(X) × {size[1] * MIX_TEST_LANES_EXPECTED:g} m(Y)），难度由 difficulty_range="
            f"({difficulty_range[0]}, {difficulty_range[1]}) 精确固定 ⇒ `--num_envs ≤ "
            f"{MIX_TEST_LANES_EXPECTED}` 时环境 i → 第 i 道")

    # ------------------------------------- 2026-10-04（第五批）：瓦片放大到 20 m ＋ 训练侧不得被改
    say(f"\n  ── 第五批检查：单块瓦片 `terrain_generator.size`（评测 {size[0]:g}×{size[1]:g} m；"
        f"训练 {train_size[0]:g}×{train_size[1]:g} m）──")
    if (round(size[0], 9), round(size[1], 9)) != MIX_TEST_TILE_SIZE_EXPECTED:
        say(f"  ❌ 评测 cfg 的 `terrain_generator.size` 应为 {MIX_TEST_TILE_SIZE_EXPECTED}"
            f"（第五批：单块瓦片 X 8 m → 20 m），实测 {size}")
        problems += 1
    else:
        say(f"  ✅ 评测 cfg 的 `terrain_generator.size` = {size[0]:g}×{size[1]:g} m"
            f"（第五批：X 由 8 m 放大到 20 m；Y 仍 4 m）")
    if (round(train_size[0], 9), round(train_size[1], 9)) != TRAIN_TILE_SIZE_EXPECTED:
        say(f"  ❌ **训练/play 侧**的 `terrain_generator.size` 必须仍是 {TRAIN_TILE_SIZE_EXPECTED}"
            f"（`size` 是共享字段，只有评测 cfg 能改），实测 {train_size}")
        problems += 1
    else:
        say(f"  ✅ **训练/play 侧**的 `terrain_generator.size` 仍是 {train_size[0]:g}×{train_size[1]:g} m"
            f"（共享字段未被评测场景带偏）")

    # ================== 2026-10-05（第七批）：**复合道**（障碍清单 / 逐对相邻间隔 / 占满 / 出生点）
    # 用户原话：「是有些障碍分到一起了，我说的是障碍本身。gap 和突台也在一起。我希望是和单独的那个
    # 几个场景类似的障碍放在这个整条的（道）上」＋「每个地形要间隔开，而不是混合到一起放在一个位置」
    # ⇒ 评测场景的地形已换成 `track_composite_terrain`；本段是它的**离线判据**（标准库算术复算，
    # 真 trimesh 侧的交叉核对在 `tests/test_check_terrain_columns.py` 与
    # `tests/test_cmoe_mix_test_scene.py`）。**原任务 `cmoe-rough` 的判据一字未放宽。**
    composite_kwargs = (info.get("sub_terrain_calls") or {}).get(MIX_TEST_SUB_TERRAIN_KEY)  # type: ignore[union-attr]
    say(f"\n  ── 第七批检查：复合道（`{MIX_TEST_SUB_TERRAIN_KEY}`，瓦片 size[0] = {size[0]:g} m）──")
    layout: dict[str, object] | None = None
    if not isinstance(composite_kwargs, dict):
        say(f"  ❌ 解析不到 `sub_terrains['{MIX_TEST_SUB_TERRAIN_KEY}'] = <Cfg>(...)`"
            f"（实测 sub_terrains = {list(props)}）")
        problems += 1
    else:
        if float(composite_kwargs.get("proportion", -1.0)) != 1.0:
            say(f"  ❌ 复合道 `proportion` 应为 1.0，实测 {composite_kwargs.get('proportion')!r}")
            problems += 1
        try:
            layout = composite_layout(MIX_TEST_DIFFICULTY_EXPECTED, composite_kwargs, size)
        except Exception as error:  # pragma: no cover - 正常仓库不会走到
            say(f"  ❌ 复核复合道布局失败（{error}）")
            problems += 1
    if layout is not None:
        sequence = tuple(layout["sequence"])  # type: ignore[arg-type]
        if sequence != COMPOSITE_SEQUENCE_EXPECTED:
            say(f"  ❌ 障碍顺序应为 {COMPOSITE_SEQUENCE_EXPECTED}，实测 {sequence}")
            problems += 1
        else:
            say("  ✅ 障碍按类型分组顺序摆放（不混在同一小段里）："
                + " → ".join(COMPOSITE_KIND_CN[k] for k in sequence))
        widths = [round(float(o["width"]), 6) for o in layout["obstacles"]]  # type: ignore[union-attr]
        if len(widths) != len(COMPOSITE_WIDTHS_EXPECTED) or any(
            abs(got - want) > 1.0e-6 for got, want in zip(widths, COMPOSITE_WIDTHS_EXPECTED)
        ):
            say(f"  ❌ 各障碍宽度应为 {COMPOSITE_WIDTHS_EXPECTED}（＝各自独立地形的难度律，d = "
                f"{MIX_TEST_DIFFICULTY_EXPECTED}），实测 {widths}")
            problems += 1
        total_width = float(layout["total_width"])
        if abs(total_width - COMPOSITE_TOTAL_WIDTH_EXPECTED) > 1.0e-6:
            say(f"  ❌ 障碍总宽应为 {COMPOSITE_TOTAL_WIDTH_EXPECTED} m，实测 {total_width:.4f} m")
            problems += 1

        # ---------------- 障碍清单与 X 区间表（障碍 / 平地间隔 / 种类 / 尺寸）
        say(f"  ── 障碍清单与 X 区间表（d = {MIX_TEST_DIFFICULTY_EXPECTED}，瓦片 {size[0]:g} m）──")
        flat_list = list(layout["flats"])  # type: ignore[arg-type]
        say(f"    [0.0000, {float(flat_list[0]['x1']):.4f}]  平地（出生平地，长 "
            f"{float(flat_list[0]['length']):.4f} m）")
        for index, obstacle in enumerate(layout["obstacles"]):  # type: ignore[union-attr]
            kind = str(obstacle["kind"])
            detail = _composite_detail(kind, obstacle["params"])  # type: ignore[arg-type]
            say(f"    [{float(obstacle['x0']):.4f}, {float(obstacle['x1']):.4f}]  "
                f"{COMPOSITE_KIND_CN[kind]:10s} 宽 {float(obstacle['width']):.4f} m（{detail}）")
            flat = flat_list[index + 1]
            tag = "尾平地" if flat["kind"] == "tail" else "平地间隔"
            say(f"    [{float(flat['x0']):.4f}, {float(flat['x1']):.4f}]  {tag}，长 "
                f"{float(flat['length']):.4f} m")
        say("    ⇒ 每个障碍都用**它自己那类独立地形**的难度律（坑宽＝沟宽律、楼梯级高＝步高律、"
            "箱子高/长＝步高/步长律、栏高＝栏高律）⇒ 尺寸逐项与独立地形一致"
            "（真 trimesh 对照见 tests/test_cmoe_mix_test_scene.py）。")

        # ---------------- **逐对相邻间隔**（硬约束：任意两个相邻障碍之间都有 ≥ obstacle_spacing 的平地）
        min_spacing = float(composite_kwargs.get("min_obstacle_spacing", COMPOSITE_MIN_SPACING_EXPECTED))
        say("  ── 逐对相邻障碍间隔表（硬约束：**所有**相邻对都必须 ≥ 下界）──")
        say("    障碍对                              间隔(m)   下界(m)   是否 ≥ 下界")
        worst = float("inf")
        for gap in layout["adjacent_gaps"]:  # type: ignore[union-attr]
            left, right = gap["pair"]
            length = float(gap["length"])
            worst = min(worst, length)
            mark = "✅" if length >= min_spacing - 1.0e-9 else "❌"
            say(f"    {COMPOSITE_KIND_CN[str(left)]} ↔ {COMPOSITE_KIND_CN[str(right)]:<14s} "
                f"{length:8.4f}  {min_spacing:8.4f}   {mark}")
        if worst + 1.0e-9 < min_spacing:
            say(f"  ❌ 最小相邻间隔 {worst:.4f} m < 下界 {min_spacing:.4f} m ⇒ 有障碍被挤在一起")
            problems += 1
        else:
            say(f"  ✅ 最小相邻间隔 {worst:.4f} m ≥ 下界 {min_spacing:.4f} m"
                "（**每一对相邻障碍之间都有平地** ⇒ 各自独立、不混在一起）")
        spacing = float(layout["spacing"])
        if abs(spacing - COMPOSITE_SPACING_EXPECTED) > 1.0e-6:
            say(f"  ❌ 均匀间隔应为反算值 {COMPOSITE_SPACING_EXPECTED} m，实测 {spacing:.4f} m")
            problems += 1
        if abs(min_spacing - COMPOSITE_MIN_SPACING_EXPECTED) > 1.0e-9:
            say(f"  ❌ `min_obstacle_spacing` 应为 {COMPOSITE_MIN_SPACING_EXPECTED} m，实测 {min_spacing}")
            problems += 1
        first_x = float(layout["first_obstacle_x"])
        reverse = (size[0] - first_x - total_width) / len(layout["obstacles"])  # type: ignore[arg-type]
        say(f"  ✅ 均匀间隔 ＝ 反算值：({size[0]:g} − {first_x:.4f} − {total_width:.4f}) / "
            f"{len(layout['obstacles'])} = **{reverse:.4f} m**"  # type: ignore[arg-type]
            f"（{'自动反算' if layout['spacing_is_auto'] else '显式指定'}；下界 {min_spacing:.4f} m，"
            "建议区间 (1.50, 2.50)）")

        # ---------------- 楼梯：**上下紧贴**（级间没有平地）＋ 楼梯与相邻障碍之间有平地
        stairs = [o for o in layout["obstacles"] if o["kind"] == "stairs"]  # type: ignore[union-attr]
        if len(stairs) != 1:
            say(f"  ❌ 复合道应恰有 1 段楼梯，实测 {len(stairs)}")
            problems += 1
        else:
            params = stairs[0]["params"]  # type: ignore[assignment]
            level_count = int(params["step_count_total"])  # type: ignore[index]
            tops = list(params["tread_tops"])  # type: ignore[index]
            step_depth = float(params["step_depth"])  # type: ignore[index]
            stair_x0 = float(stairs[0]["x0"])
            width = float(stairs[0]["width"])
            if level_count != 2 * COMPOSITE_STAIR_LEVELS_EXPECTED:
                say(f"  ❌ 楼梯级数应为上 {COMPOSITE_STAIR_LEVELS_EXPECTED} + 下 "
                    f"{COMPOSITE_STAIR_LEVELS_EXPECTED} = {2 * COMPOSITE_STAIR_LEVELS_EXPECTED} 级，"
                    f"实测 {level_count} 级")
                problems += 1
            elif abs(tops[COMPOSITE_STAIR_LEVELS_EXPECTED - 1]
                     - COMPOSITE_STAIR_PEAK_EXPECTED) > 1.0e-9:
                say(f"  ❌ 楼梯峰高应为 {COMPOSITE_STAIR_PEAK_EXPECTED} m"
                    f"（= {COMPOSITE_STAIR_LEVELS_EXPECTED} × "
                    f"{COMPOSITE_STAIR_STEP_HEIGHT_EXPECTED}），实测 "
                    f"{tops[COMPOSITE_STAIR_LEVELS_EXPECTED - 1]}")
                problems += 1
            elif abs(width - 2 * COMPOSITE_STAIR_LEVELS_EXPECTED * step_depth) > 1.0e-9:
                say(f"  ❌ 楼梯段总长应为 2 × {COMPOSITE_STAIR_LEVELS_EXPECTED} × {step_depth} = "
                    f"{2 * COMPOSITE_STAIR_LEVELS_EXPECTED * step_depth} m，实测 {width:.4f} m")
                problems += 1
            else:
                say(f"  ✅ 楼梯上下紧贴：上 {COMPOSITE_STAIR_LEVELS_EXPECTED} 级 + 下 "
                    f"{COMPOSITE_STAIR_LEVELS_EXPECTED} 级**首尾相接、级间无平地**（金字塔形），"
                    f"级高 {COMPOSITE_STAIR_STEP_HEIGHT_EXPECTED} m、步深 {step_depth} m、段长 "
                    f"{width:.2f} m、峰高 {COMPOSITE_STAIR_PEAK_EXPECTED:.2f} m")
                say("      " + " / ".join(
                    f"[{stair_x0 + i * step_depth:.4f},{stair_x0 + (i + 1) * step_depth:.4f}] {t:.4f}"
                    for i, t in enumerate(tops)))
            stair_gaps = [g for g in layout["adjacent_gaps"] if "stairs" in g["pair"]]  # type: ignore[union-attr]
            if stair_gaps:
                lengths = [float(g["length"]) for g in stair_gaps]
                if min(lengths) + 1.0e-9 < min_spacing:
                    say(f"  ❌ 楼梯与相邻障碍之间的平地 {min(lengths):.4f} m < 下界 {min_spacing:.4f} m")
                    problems += 1
                else:
                    say(f"  ✅ 楼梯与相邻障碍间隔 = {min(lengths):.4f} m ≥ 下界 {min_spacing:.4f} m"
                        "（「紧贴」只指上/下楼梯之间）")

        # ---------------- 占满整条道 ＋ 出生点
        last_end = float(layout["last_obstacle_end"])
        tail = float(layout["tail_length"])
        end_x = float(layout["end_x"])
        say(f"  ── 占满核算：最后一个障碍末端 {last_end:.4f} m ＋ 尾段平地 {tail:.4f} m = "
            f"{end_x:.4f} m（瓦片 {size[0]:g} m）──")
        if abs(end_x - size[0]) > 1.0e-6:
            say(f"  ❌ 复合道末端 {end_x:.4f} m ≠ 瓦片长 {size[0]:g} m（未占满整条道）")
            problems += 1
        elif abs(last_end - COMPOSITE_LAST_OBSTACLE_END_EXPECTED) > 1.0e-6:
            say(f"  ❌ 最后一个障碍末端应为 {COMPOSITE_LAST_OBSTACLE_END_EXPECTED} m，实测 "
                f"{last_end:.4f} m")
            problems += 1
        else:
            say(f"  ✅ 占满整条道：0 → {end_x:.4f} m（最后一段间隔＝尾段 {tail:.4f} m ≥ 下界 "
                f"{min_spacing:.4f} m）")
        spawn_x = float(layout["spawn_x"])
        clearance = float(layout["spawn_clearance"])
        if clearance + 1.0e-9 < COMPOSITE_SPAWN_CLEARANCE_MIN:
            say(f"  ❌ `spawn_clearance` = {clearance:.4f} m < 下界 {COMPOSITE_SPAWN_CLEARANCE_MIN} m")
            problems += 1
        elif abs(spawn_x - COMPOSITE_SPAWN_X_EXPECTED) > 1.0e-9:
            say(f"  ❌ `spawn_x` 应为 {COMPOSITE_SPAWN_X_EXPECTED}，实测 {spawn_x}")
            problems += 1
        elif abs(first_x - COMPOSITE_FIRST_OBSTACLE_X_EXPECTED) > 1.0e-9:
            say(f"  ❌ 第一个障碍左沿应为 {COMPOSITE_FIRST_OBSTACLE_X_EXPECTED} m（= "
                f"{COMPOSITE_SPAWN_X_EXPECTED} + {COMPOSITE_SPAWN_CLEARANCE_MIN}），实测 "
                f"{first_x:.4f} m")
            problems += 1
        else:
            say(f"  ✅ 出生点：`spawn_x` = {spawn_x} 落在 [0, {first_x:.4f}] 的实心平地上（顶面 0），"
                f"到第一个障碍 **{first_x - spawn_x:.4f} m ≥ {COMPOSITE_SPAWN_CLEARANCE_MIN} m**")

    # ---------------- 训练侧 `mix` 未受影响（**本轮不碰训练**；评测场景已不再引用 mix 图案）
    mix_defaults = mix_terrain_defaults()
    say("\n  ── 训练侧 `mix` 未受影响（本轮只换评测场景的地形类型）──")
    if abs(float(mix_defaults.get("pattern_spacing_scale", -1.0)) - 1.0) > 1.0e-9 or \
            mix_defaults.get("fill_stretched_gaps") is not False:
        say("  ❌ `CMoETrackMixTerrainCfg` 的默认值被改了（应为 `pattern_spacing_scale=1.0`、"
            f"`fill_stretched_gaps=False`），实测 {mix_defaults.get('pattern_spacing_scale')!r} / "
            f"{mix_defaults.get('fill_stretched_gaps')!r}")
        problems += 1
    else:
        say("  ✅ `CMoETrackMixTerrainCfg` 默认仍是 `pattern_spacing_scale = 1.0`、"
            "`fill_stretched_gaps = False` ⇒ 训练几何逐位不变")
    segments = mix_pattern_segments(MIX_TEST_DIFFICULTY_EXPECTED)
    groups = mix_contiguous_groups(segments)
    spans = [(group[0][0], group[-1][1]) for group in groups]
    raised = [segment for group in groups
              if group[0][0] <= 30.0 <= group[-1][1] for segment in group if segment[2] > 0.0]
    if len(segments) != 12 or spans != list(MIX_GROUP_SPANS_EXPECTED) or \
            len(raised) != MIX_STAIR_LEVELS_EXPECTED:
        say(f"  ❌ 训练侧 `mix` 图案被改了（段数/分组/楼梯级数与既有结论不一致）："
            f"{len(segments)} 段、{spans}、楼梯 {len(raised)} 级")
        problems += 1
    else:
        say(f"  ✅ 训练侧 `mix` 图案 12 段、按首尾相接仍切成 {len(spans)} 组、4 级楼梯仍连续"
            f"（拉大 {MIX_WIDE_SCALE_EXPECTED:g} 倍后的几何仍由 "
            "`tests/test_cmoe_mix_test_scene.py` 用真 trimesh 锁定）")
    if MIX_TEST_SUB_TERRAIN_KEY in (info.get("sub_terrain_calls") or {}):  # type: ignore[union-attr]
        say(f"  ✅ 评测场景的 `sub_terrains` 只有 `{MIX_TEST_SUB_TERRAIN_KEY}`"
            "（**不再引用** `CMoETrackMixTerrainCfg`）")

    # ---------------------------- 2026-10-04（第五批）：单局时长必须够走完 20 m（1.0 m/s ⇒ 20 s）
    episode_length = episode_length_s(MIX_TEST_CLASS)
    train_episode_length = episode_length_s(TRAIN_CLASS)
    needed = size[0] / MIX_TEST_FORWARD_SPEED
    say(f"\n  ── 第五批检查：单局时长（道 {size[0]:g} m ÷ 恒定 {MIX_TEST_FORWARD_SPEED:g} m/s = "
        f"{needed:.1f} s）──")
    if episode_length is None:
        say(f"  ❌ 评测 cfg 未覆盖 `episode_length_s`（继承来的默认是 20 s，正好等于走完 20 m 的时间、"
            f"没有余量）⇒ 必须显式设为 ≥ {MIX_TEST_EPISODE_LENGTH_MIN_S:g} s")
        problems += 1
    elif episode_length < MIX_TEST_EPISODE_LENGTH_MIN_S - 1.0e-9:
        say(f"  ❌ 评测 cfg 的 `episode_length_s` = {episode_length:g} s < 下界 "
            f"{MIX_TEST_EPISODE_LENGTH_MIN_S:g} s（走完 {size[0]:g} m 要 {needed:.1f} s，余量不足）")
        problems += 1
    else:
        say(f"  ✅ 评测 cfg 的 `episode_length_s` = {episode_length:g} s ≥ 下界 "
            f"{MIX_TEST_EPISODE_LENGTH_MIN_S:g} s（走完 {size[0]:g} m 要 {needed:.1f} s，余量 "
            f"{episode_length - needed:.1f} s）")
    if train_episode_length is None:
        say("  ℹ️ 训练/play 链**未**覆盖 `episode_length_s`（沿用父类 20 s）—— 本批只改评测 cfg，"
            "训练时长一字未动")
    else:
        say(f"  ℹ️ 训练/play 链的 `episode_length_s` = {train_episode_length:g} s（与评测侧各自独立）")

    absent = sorted({n for refs in MASKED_NAMES.values() for n in refs} - set(names))
    say(f"  ⚠️ 本场景**只有 `{MIX_TEST_SUB_TERRAIN_KEY}`（复合道）**；掩码引用但本场景**不存在**的"
        "地形名（原任务判据**不放宽**，"
        "这些项在本场景恒为 0 属**预期**）：")
    if not absent:
        say("    （无）")
    for key, refs in MASKED_NAMES.items():
        for name in refs:
            if name in absent:
                say(f"    - {key} → {name}：本场景无该地形 ⇒ 该项本场景恒为 0（预期，不是配置错误）")

    inherited = False
    fwd = forward_only_names(MIX_TEST_CLASS)
    if fwd is None:
        inherited = True
        fwd = forward_only_names(TRAIN_CLASS)
    if fwd is None:
        say("  ❌ 继承链里找不到 `forward_only_terrain_names`（掩码/命令会静默退化）")
        problems += 1
    else:
        tag = "（继承父类）" if inherited else ""
        extra = [n for n in fwd if n not in names]
        note = ""
        if extra:
            note = (f"；名单里另含 {len(extra)} 个本场景不存在的地形名（{', '.join(extra)}）—— 命令项"
                    " `MixTestVelocityCommand` 根本不读这张表（恒给前进 1.0 m/s），且"
                    " `is_env_assigned_to_terrain` 对未登记的名字返回全 False ⇒ 在这里是**惰性**的，"
                    "属预期")
        say(f"  ✅ `forward_only_terrain_names`{tag} 存在（训练侧 11 类）{note}")

    say("\n结论：" + (f"test 任务只有复合道 `{MIX_TEST_SUB_TERRAIN_KEY}` 一种地形；"
                      "掩码引用的其它地形名在本场景恒为 0，属**预期** ✅"
                      if problems == 0 else f"test 任务有 {problems} 处问题 ❌"))
    return problems, lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cols", nargs="*", type=int, default=None,
                        help="要算的 num_cols（默认＝CMoE_env_cfg 里配置的值，2026-09-28 起为 40）")
    parser.add_argument("--task", default="cmoe-rough",
                        help="要检查的任务：cmoe-rough（默认，训练/play 的 11 类按比例）或 "
                             f"mix-test（＝{MIX_TEST_TASK}，只有 mix）")
    args = parser.parse_args(argv)

    if args.task not in TASK_ALIASES:
        parser.error(f"未知任务 {args.task!r}；可选：{', '.join(sorted(TASK_ALIASES))}")
    task = TASK_ALIASES[args.task]

    if task == MIX_TEST_TASK:
        if args.cols:
            print(f"提示：`--task {args.task}` 的列数由 cfg 决定（现为 {MIX_TEST_LANES_EXPECTED} 条 mix 道），"
                  "`--cols` 被忽略")
        return 1 if mix_test_report()[0] else 0

    merged, info = merged_scene(TRAIN_CLASS)
    col_over = {k: v for k, v in {"num_cols": info["num_cols"]}.items() if v is not None}

    print("CMoE rough 的 sub_terrains 比例（基类 + 任务侧覆盖）：")
    props = info["props"]
    for name, value in merged:
        mark = "  ←覆盖" if name in props else ""  # type: ignore[operator]
        print(f"  {name:22s} {value:.4f}{mark}")

    default_cols = [int(col_over["num_cols"])] if col_over.get("num_cols") else [40]
    cols_to_check = args.cols if args.cols else default_cols
    counts_by_cols = {}
    for num_cols in cols_to_check:
        label = "训练／play 当前值" if num_cols in default_cols else f"num_cols={num_cols}（手动指定）"
        counts_by_cols[num_cols] = report(merged, num_cols, label)

    problems = 0
    for num_cols, counts in counts_by_cols.items():
        problems += check_mask_columns(counts, num_cols)
    if col_over:
        print(f"\n提示：配置里显式覆盖过 num_cols = {col_over['num_cols']}（训练与 play 都是）")

    # 2026-09-24 用户决定「所有场景都只给超前的速度」⇒ `forward_only_terrain_names` 必须覆盖
    # **全部** sub_terrains（漏一项，那一列就会静默退回全向命令）。
    names = [n for n, _ in merged]
    fwd = forward_only_names(TRAIN_CLASS)
    if fwd is None:
        print("\n（配置未设置 forward_only_terrain_names，跳过覆盖率校验）")
    else:
        missing = [n for n in names if n not in fwd]
        unknown = [n for n in fwd if n not in names]
        if missing or unknown:
            print(f"\n❌ forward_only_terrain_names 未覆盖全部地形：缺 {missing}／多余 {unknown}")
            problems += 1
        else:
            print(f"\n✅ forward_only_terrain_names 覆盖全部 {len(names)} 类地形（全场景前向命令）")

    print("\n结论：" + ("全部掩码引用的地形名都有 ≥1 列 ✅" if problems == 0 else f"有 {problems} 处问题 ❌"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
