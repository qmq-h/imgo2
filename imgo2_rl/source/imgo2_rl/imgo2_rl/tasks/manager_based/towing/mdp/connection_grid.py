"""60 columns: 20 flat + 10 five-degree + 10 ten-degree + 10 speed-bump + 10 wave.

Flat columns keep 8 compliant / 8 rigid / 4 inextensible. Each of the four 10-column
terrain blocks has 4 / 4 / 2 columns and is structurally identical to the first slope
block. Not a full factorial design at each length.

2026-10-09：地形改成**一条连续剖面**（平地 → 上坡 → 坡顶 → 下坡 → 平地，见
`slope_geometry.profile_*`），上坡与下坡在**同一条 lane 内成对出现**，所以
`slope_degrees` 只保留**量级**（0 / 5 / 10），不再按行列交替正负 —— 方向平衡是构造上成立
的，不需要再用交替来避免「某个弹性档只遇到上坡」。

2026-10-16（用户要求「直接把这个拼接到之前的设置中」→ 最终定为 **10 列、照 5° 坡块同构**）：
减速带不再单独换一整套地形，而是**追加一个 10 列的块拼进同一张网格**（列 40–49）。这 10 列的
地形**完全相同**（`speedbump_geometry`：一条底 300 mm、顶 100 mm、高 50 mm 的等腰梯形凸条，
居中放在剖面区 x = 5.625 m），列与列之间只差连接类型。⇒ `COLUMNS` 40 → **50**。

2026-10-16（用户要求「最后添加一个波浪的起伏地形，和上下坡占据同样的位置，最后用一个平地收尾」）：
再追加**一个 10 列的波浪块**（列 50–59），同样照 5° 坡块同构；地形是 `wave_geometry` 的
**3 周期等幅正弦**（周期 2.25 m、振幅 10 cm），占**与平坡相同的剖面窗口** x = 2.25–9.0，
两端用半个周期的 `cos²` 包络平滑归零、9.0 之后回到平地收尾（10 m 的 STOP 点仍在平地上）。
⇒ `COLUMNS` 50 → **60**、`GRID_SIZE` 1000 → **1200**；前 50 列的列号/语义一字未动。

`terrain_family(column)` 给出该列的**地形族**（`smooth` = 平地/平坡列、`speedbump` = 减速带块、
`wave` = 波浪块），`slope_degrees` 对该列报出**真实的剖面最大纵向坡度**（减速带列 = 梯形侧坡
`SPEEDBUMP_RAMP_DEG` = 26.565°、波浪列 = 正弦最大侧坡 `WAVE_MAX_SLOPE_DEG`，都不是 0/5/10
那套量级档；见 `speedbump_geometry` / `wave_geometry`）。
"""

from __future__ import annotations

import math

MODEL_INDEX = {"compliant": 0, "inextensible": 1, "rigid": 2}
FLAT_COLUMNS = 20
FIVE_DEGREE_COLUMNS = 10
TEN_DEGREE_COLUMNS = 10
SPEEDBUMP_COLUMNS = 10
WAVE_COLUMNS = 10
FIVE_DEGREE_START_COLUMN = FLAT_COLUMNS                                     # 20
TEN_DEGREE_START_COLUMN = FIVE_DEGREE_START_COLUMN + FIVE_DEGREE_COLUMNS    # 30
SPEEDBUMP_START_COLUMN = TEN_DEGREE_START_COLUMN + TEN_DEGREE_COLUMNS       # 40
WAVE_START_COLUMN = SPEEDBUMP_START_COLUMN + SPEEDBUMP_COLUMNS              # 50
COLUMNS = WAVE_START_COLUMN + WAVE_COLUMNS                                  # 60
ELASTIC_COLUMNS = 24
RIGID_COLUMNS = 24
INEXTENSIBLE_COLUMNS = 12
ROWS = 20
# 2026-10-09 用户决定：**绳类与杆类用各自的长度范围**（同一行索引在两类里映射到各自的区间）。
#   绳（compliant / inextensible）0.5–1.5 m：真实拖绳偏长，且出生比 0.8 让短绳也有实体间隙；
#   杆（rigid）0.5–1.0 m：拖杆本来就短，而且杆出生用**全长**（无松弛）⇒ 太长会把小车顶出车道
#   （车道后向边界 = −BACK_M + BOUNDARY_MARGIN_M = −1.65 m；r=0.8 的绳在 L=1.5 时小车在
#    −1.598 m，已经接近上限，所以绳的上界就卡在 1.5）。
ROPE_LENGTH_MIN_M = 0.5
ROPE_LENGTH_MAX_M = 1.5
RIGID_LENGTH_MIN_M = 0.5
RIGID_LENGTH_MAX_M = 1.0
GRID_SIZE = COLUMNS * ROWS

# ---- 地形族与减速带（等腰梯形）尺寸 ------------------------------------------------------
#: 平滑段（平地 + 平坡）、减速带块、波浪块三种地形族。
SMOOTH_TERRAIN = "smooth"
SPEEDBUMP_TERRAIN = "speedbump"
WAVE_TERRAIN = "wave"

#: 减速带梯形尺寸（m，用户 2026-10-16 口径：底 300 / 顶 100 / 高 50 mm）。
#: **放在这个最底层模块里**是有意的：`slope_degrees` 必须报出该列的真实最大纵向坡度 = 梯形
#: 侧坡，而 `speedbump_geometry` 反过来要 import 本模块（不能反向依赖），所以尺寸只能在这层
#: 定义、由几何模块取用（`test_towing_speedbump_geometry` 交叉核对两处一致）。
SPEEDBUMP_BOTTOM_M = 0.30
SPEEDBUMP_TOP_M = 0.10
SPEEDBUMP_HEIGHT_M = 0.05
SPEEDBUMP_RAMP_RUN_M = (SPEEDBUMP_BOTTOM_M - SPEEDBUMP_TOP_M) / 2.0        # 0.10
SPEEDBUMP_RAMP_DEG = math.degrees(math.atan(SPEEDBUMP_HEIGHT_M / SPEEDBUMP_RAMP_RUN_M))
if min(SPEEDBUMP_BOTTOM_M, SPEEDBUMP_TOP_M, SPEEDBUMP_HEIGHT_M) <= 0.0 \
        or SPEEDBUMP_TOP_M >= SPEEDBUMP_BOTTOM_M:
    raise ValueError("减速带梯形必须满足 0 < 顶 < 底 且高 > 0")

#: 波浪（等幅正弦）参数（m；用户 2026-10-16 口径「振幅不要太大，先定义大概 10 cm」）。
#: 3 个周期铺满与平坡相同的剖面窗口（2.25–9.0 m ⇒ 周期 2.25 m）；振幅 A = 10 cm 时
#: 最大纵向坡度 = atan(2πA/λ) = **15.61°**（正弦在等幅段的最大斜率）。
#: 同样放在最底层模块里，理由与减速带相同：`slope_degrees` 要报出这个真实坡度。
WAVE_PERIOD_COUNT = 3
WAVE_PERIOD_M = 2.25
WAVE_PEAK_M = 0.10
WAVE_MAX_SLOPE_DEG = math.degrees(math.atan(
    2.0 * math.pi * WAVE_PEAK_M / WAVE_PERIOD_M))
if min(WAVE_PERIOD_COUNT, WAVE_PERIOD_M, WAVE_PEAK_M) <= 0.0:
    raise ValueError("波浪参数必须为正")


# k [N/m], c [N s/m], unchanged from the flat towing baseline.
ELASTIC_KC = ((1000.0, 50.0), (4000.0, 100.0), (20000.0, 220.0), (100000.0, 460.0))
# 绳的出生挂点距 = SLACK_RATIO × L（<1 ⇒ 仍是松弛、第一拍无约束力）。
# 2026-10-09 由 0.5 提到 0.8；新几何（Δz=0.22、base_offset=0.0025，与 upper_mdp.spawn_clearance 同源）下
# 出生间隙 = √((ratio·L)²−0.22²) + 0.0025：0.5 在 L=0.6 时只有 206 mm、0.8 给出 429 mm；
# 代价是松弛量从 50% 降到 20%（取绳空走变短、加载开始得更早）。
SLACK_RATIO = 0.8

# 兼容旧名：历史代码/文档把 `LENGTH_MIN_M` 当"绳的长度下界"用。
LENGTH_MIN_M = ROPE_LENGTH_MIN_M
LENGTH_MAX_M = ROPE_LENGTH_MAX_M

LENGTH_RANGES = {
    "compliant": (ROPE_LENGTH_MIN_M, ROPE_LENGTH_MAX_M),
    "inextensible": (ROPE_LENGTH_MIN_M, ROPE_LENGTH_MAX_M),
    "rigid": (RIGID_LENGTH_MIN_M, RIGID_LENGTH_MAX_M),
}


def row_length(row: int, model_name: str = "compliant") -> float:
    """该行在给定连接类型下的长度（等距 20 档，含两端）。

    `model_name` 决定用绳还是杆的区间（见 `LENGTH_RANGES`）；默认绳区间，保持旧调用可用。
    """
    if not 0 <= row < ROWS:
        raise ValueError(f"row must be in [0, {ROWS}), got {row}")
    if model_name not in LENGTH_RANGES:
        raise ValueError(f"unknown connection model {model_name!r}")
    low, high = LENGTH_RANGES[model_name]
    return low + (high - low) * row / (ROWS - 1)


def column_spec(column: int):
    """Return (connection model, stiffness, damping); elasticity stays column-only.

    三个非平地块（5° / 10° / 减速带）都是 10 列同构：slot 0–3 弹性绳（4 档 k/c 各 1 列）、
    slot 4–7 刚体杆、slot 8–9 低弹性绳；平地块是同样的顺序但每档 2 列、且只有 8/8/4。
    """
    if not 0 <= column < COLUMNS:
        raise ValueError(f"column must be in [0, {COLUMNS}), got {column}")
    if column < FLAT_COLUMNS:
        slot, elastic, rigid, per_level = column, 8, 8, 2
    else:
        slot, elastic, rigid, per_level = (column - FLAT_COLUMNS) % 10, 4, 4, 1
    if slot < elastic:
        return "compliant", *ELASTIC_KC[slot // per_level]
    if slot < elastic + rigid:
        return "rigid", None, None
    return "inextensible", None, None


def terrain_family(column: int) -> str:
    """该列的地形族：`smooth`（平地/平坡，列 0–39）、`speedbump`（40–49）、`wave`（50–59）。"""
    if not 0 <= column < COLUMNS:
        raise ValueError(f"column must be in [0, {COLUMNS}), got {column}")
    if column < SPEEDBUMP_START_COLUMN:
        return SMOOTH_TERRAIN
    if column < WAVE_START_COLUMN:
        return SPEEDBUMP_TERRAIN
    return WAVE_TERRAIN


def is_speedbump(column: int) -> bool:
    """该列是否属于减速带块（供测试台/统计工具按列分派高度场）。"""
    return terrain_family(column) == SPEEDBUMP_TERRAIN


def is_wave(column: int) -> bool:
    """该列是否属于波浪块（同上，供按列分派）。"""
    return terrain_family(column) == WAVE_TERRAIN


def slope_degrees(column: int, row: int = 0) -> float:
    """这条 lane 剖面的**最大纵向坡度**（deg）。

    - 平地列 = 0、5° 坡块 = 5、10° 坡块 = 10（每列剖面都是「平地 2.25 m → 上坡 3 m →
      坡顶 0.75 m → 下坡 3 m → 平地 2.25 m」，上/下坡成对出现，所以只给量级、没有方向）；
    - **减速带块 = 梯形侧坡 26.565°**（`SPEEDBUMP_RAMP_DEG`）：那 10 列的地形完全相同，
      这个值是该剖面的真实最大坡度，也是 timeout 弧长与相对俯仰要用的量；
    - **波浪块 = 正弦最大侧坡**（`WAVE_MAX_SLOPE_DEG`，9 个周期参数下约 15.61°）：同样地，
      那 10 列地形相同，报的是剖面的真实最大坡度。

    `row` 只为兼容旧签名保留（不影响取值）。
    """
    if not 0 <= column < COLUMNS or not 0 <= row < ROWS:
        raise ValueError(f"invalid cell row={row}, column={column}")
    if column < FLAT_COLUMNS:
        return 0.0
    if column < TEN_DEGREE_START_COLUMN:
        return 5.0
    if column < SPEEDBUMP_START_COLUMN:
        return 10.0
    if column < WAVE_START_COLUMN:
        return SPEEDBUMP_RAMP_DEG
    return WAVE_MAX_SLOPE_DEG


def initial_attachment_distance(model_name: str, length: float) -> float:
    if model_name == "rigid":
        return length
    if model_name in ("compliant", "inextensible"):
        return length * SLACK_RATIO
    raise ValueError(f"unknown connection model {model_name!r}")


def env_spec(env_index: int) -> dict:
    """Row-major logical cell, repeated for multiples of `GRID_SIZE` envs."""
    if env_index < 0:
        raise ValueError(f"env_index must be nonnegative, got {env_index}")
    index = env_index % GRID_SIZE
    column, row = index % COLUMNS, index // COLUMNS
    model_name, stiffness, damping = column_spec(column)
    # 长度按**该列的类型**取（绳/杆各自区间），不是全局一个区间
    length = row_length(row, model_name)
    return {
        "env_index": env_index, "grid_index": index, "column": column, "row": row,
        "model_name": model_name, "model_index": MODEL_INDEX[model_name],
        "length": length, "stiffness": stiffness, "damping": damping,
        "initial_distance": initial_attachment_distance(model_name, length),
        "slope_degrees": slope_degrees(column, row),
        "terrain_family": terrain_family(column),
    }


def is_full_grid(num_envs: int) -> bool:
    return num_envs > 0 and num_envs % GRID_SIZE == 0
