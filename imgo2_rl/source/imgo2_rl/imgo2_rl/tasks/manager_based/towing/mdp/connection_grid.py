"""20 flat + 10 five-degree + 10 ten-degree columns, 20 length rows.

Flat columns keep 8 compliant / 8 rigid / 4 inextensible. Each slope
magnitude has 4 / 4 / 2 columns. Not a full factorial design at each length.

2026-10-09：地形改成**一条连续剖面**（平地 → 上坡 → 坡顶 → 下坡 → 平地，见
`slope_geometry.profile_*`），上坡与下坡在**同一条 lane 内成对出现**，所以
`slope_degrees` 只保留**量级**（0 / 5 / 10），不再按行列交替正负 —— 方向平衡是构造上成立
的，不需要再用交替来避免「某个弹性档只遇到上坡」。
"""

from __future__ import annotations

MODEL_INDEX = {"compliant": 0, "inextensible": 1, "rigid": 2}
FLAT_COLUMNS = 20
FIVE_DEGREE_COLUMNS = 10
TEN_DEGREE_COLUMNS = 10
COLUMNS = FLAT_COLUMNS + FIVE_DEGREE_COLUMNS + TEN_DEGREE_COLUMNS
ELASTIC_COLUMNS = 16
RIGID_COLUMNS = 16
INEXTENSIBLE_COLUMNS = 8
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

# k [N/m], c [N s/m], unchanged from the flat towing baseline.
ELASTIC_KC = ((1000.0, 50.0), (4000.0, 100.0), (20000.0, 220.0), (100000.0, 460.0))
# 绳的出生挂点距 = SLACK_RATIO × L（<1 ⇒ 仍是松弛、第一拍无约束力）。
# 2026-10-09 由 0.5 提到 0.8：出生间隙 = √((ratio·L)²−0.17²) − 0.238，0.5 在 L=0.6 时只有 9 mm，
# 0.8 在同一长度下给出 211 mm；代价是松弛量从 50% 降到 20%（取绳空走变短、加载开始得更早）。
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
    """Return (connection model, stiffness, damping); elasticity stays column-only."""
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


def slope_degrees(column: int, row: int = 0) -> float:
    """这条 lane 的**坡度量级**（deg）：0 = 纯平地 lane，5 / 10 = 剖面上的上/下坡档。

    每条 lane 的剖面都是「平地 2.25 m → 上坡 3 m → 坡顶 0.75 m → 下坡 3 m → 平地 2.25 m」，
    上坡与下坡成对出现在同一条 lane 里，所以这里只给量级、没有方向。`row` 只为兼容旧签名
    保留（不再影响取值）。
    """
    if not 0 <= column < COLUMNS or not 0 <= row < ROWS:
        raise ValueError(f"invalid cell row={row}, column={column}")
    if column < FLAT_COLUMNS:
        return 0.0
    return 5.0 if column < FLAT_COLUMNS + FIVE_DEGREE_COLUMNS else 10.0


def initial_attachment_distance(model_name: str, length: float) -> float:
    if model_name == "rigid":
        return length
    if model_name in ("compliant", "inextensible"):
        return length * SLACK_RATIO
    raise ValueError(f"unknown connection model {model_name!r}")


def env_spec(env_index: int) -> dict:
    """Row-major logical cell, repeated for multiples of 800 envs."""
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
    }


def is_full_grid(num_envs: int) -> bool:
    return num_envs > 0 and num_envs % GRID_SIZE == 0
