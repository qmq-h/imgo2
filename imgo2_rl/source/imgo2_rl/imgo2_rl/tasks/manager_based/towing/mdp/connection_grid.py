"""拖曳场景的「列 × 行」确定性网格：列 = 连接类型／弹性档，行 = 连接长度。

用户 2026-10-08 指定：

* **20 列** = 弹性绳 8 / 刚体 8 / 普通绳（inextensible）4，即 4:4:2 比例放大；
* **20 行** = 长度从 0.4 m 到 0.8 m 均分（对绳是绳长 L0、对刚体是杆长 L）；
* **弹性绳只在列上做弹性区分**：4 档 k/c，每档 2 列；刚体与普通绳的列是同参数重复列；
* 训练环境按 **env index** 确定性映射（不再逐 env 随机类型与长度），其余域随机化保持。

网格按**行优先**铺：`列 = i % 20`、`行 = i // 20`。Isaac Sim 的 `GridCloner`（`InteractiveScene`
用它摆 env）对 `num_envs=400` 取 `num_rows=num_cols=20`，位置是
`x = row_offset − (i // num_cols)·spacing`、`y = (i % num_cols)·spacing − col_offset`
（`isaacsim/core/cloner/impl/grid_cloner.py`）。因此**同一列（同一类型/弹性档）落在同一条 x 线上、
同一行（同一长度）落在同一条 y 线上**，两类参数在可视化里各自成条带，便于逐类逐长度读表。

**初始摆位**：三类连接在 spawn 时都让三维挂点距等于各自的目标值——
绳 = `SLACK_RATIO × L0`（保持现有 L0=0.8 时 0.4 m 的松弛比例；L0=0.4 时 0.2 m，仍大于
两挂点高差 0.17 m），刚体 = `L`（杆必须正好是 L）。`upper_mdp` 用
`rope_model.attachment_horizontal_gap` 把它换算成水平间距后摆放小车。

本模块**纯算术**（无 torch / Isaac Lab 导入），可离线测；`upper_mdp` 只在 `__init__` 里
调用一次，把逐 env 的类型/长度/k/c 变成设备上的张量。
"""

from __future__ import annotations

# 与 `rope_model.CONNECTION_MODELS = ("compliant", "inextensible", "rigid")` 的下标一致；
# `test_towing_connection_grid.py` 断言两者不漂移。
MODEL_INDEX = {"compliant": 0, "inextensible": 1, "rigid": 2}

# 列布局（用户给定顺序：弹性 → 刚体 → 普通绳）
ELASTIC_COLUMNS = 8
RIGID_COLUMNS = 8
INEXTENSIBLE_COLUMNS = 4
COLUMNS = ELASTIC_COLUMNS + RIGID_COLUMNS + INEXTENSIBLE_COLUMNS          # 20

# 行 = 长度均分
ROWS = 20
LENGTH_MIN_M = 0.4
LENGTH_MAX_M = 0.8

GRID_SIZE = COLUMNS * ROWS                                               # 400

# 弹性绳 4 档 (k [N/m], c [N·s/m])。按 ζ ≈ 0.3 与名义折合质量 μ ≈ 5.9 kg 配对：
# c = 2ζ·sqrt(k·μ)。对应「弹力绳 / 尼龙拖车绳（现有默认 4000/100）/ 较硬涤纶 / 接近工装用绳」。
# 最硬档在最坏质量（5 kg 小车，μ≈3.8）下显式弹簧步长上限约 8.5 ms，仍高于 5 ms 一个余量；
# 再硬就应改用约束模型或子步，而不是继续堆 k（见 docs/towing_simulation_validation.md §3）。
ELASTIC_KC = ((1000.0, 50.0), (4000.0, 100.0), (20000.0, 220.0), (100000.0, 460.0))

# 绳环境初始挂点距 = SLACK_RATIO × L0（刚体为 1.0 × L）
SLACK_RATIO = 0.5

# 弹性列每档占几列（8 列 / 4 档）
_COLUMNS_PER_KC = ELASTIC_COLUMNS // len(ELASTIC_KC)


def row_length(row: int) -> float:
    """第 `row` 行的连接长度（m）：0.4 → 0.8 均分 20 档，含两端。"""
    if not 0 <= row < ROWS:
        raise ValueError(f"row 必须在 [0, {ROWS})，收到 {row}")
    return LENGTH_MIN_M + (LENGTH_MAX_M - LENGTH_MIN_M) * row / (ROWS - 1)


def column_spec(column: int):
    """第 `column` 列 → `(model_name, stiffness, damping)`；非弹性模型 k/c 为 `None`。"""
    if not 0 <= column < COLUMNS:
        raise ValueError(f"column 必须在 [0, {COLUMNS})，收到 {column}")
    if column < ELASTIC_COLUMNS:
        stiffness, damping = ELASTIC_KC[column // _COLUMNS_PER_KC]
        return "compliant", stiffness, damping
    if column < ELASTIC_COLUMNS + RIGID_COLUMNS:
        return "rigid", None, None
    return "inextensible", None, None


def initial_attachment_distance(model_name: str, length: float) -> float:
    """spawn 时的目标三维挂点距（m）：绳留松弛（`SLACK_RATIO × L0`），刚体等于杆长。"""
    if model_name == "rigid":
        return length
    if model_name in ("compliant", "inextensible"):
        return length * SLACK_RATIO
    raise ValueError(f"未知的连接模型 {model_name!r}")


def env_spec(env_index: int) -> dict:
    """env index → 该环境的列/行/类型/长度/弹性档与初始挂点距。

    行优先且对 `GRID_SIZE` 取模：`num_envs` 是 400 的整数倍时每个环境恰好覆盖一次网格
    （`is_full_grid`）；更小的 `num_envs`（例如 4 环境的冒烟）只覆盖网格前缀，可用于跑通，
    但不是「平衡网格」。训练默认 400，即完整覆盖。
    """
    if env_index < 0:
        raise ValueError(f"env_index 必须 ≥ 0，收到 {env_index}")
    index = env_index % GRID_SIZE
    column = index % COLUMNS
    row = index // COLUMNS
    model_name, stiffness, damping = column_spec(column)
    length = row_length(row)
    return {
        "env_index": env_index,
        "grid_index": index,
        "column": column,
        "row": row,
        "model_name": model_name,
        "model_index": MODEL_INDEX[model_name],
        "length": length,
        "stiffness": stiffness,
        "damping": damping,
        "initial_distance": initial_attachment_distance(model_name, length),
    }


def is_full_grid(num_envs: int) -> bool:
    """`num_envs` 是否恰好由完整的 400 环境网格（或其整数倍）组成。"""
    return num_envs > 0 and num_envs % GRID_SIZE == 0
