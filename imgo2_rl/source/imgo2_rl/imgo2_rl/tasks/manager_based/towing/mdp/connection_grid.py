"""20 flat + 10 five-degree + 10 ten-degree columns, 20 length rows.

Flat columns keep 8 compliant / 8 rigid / 4 inextensible. Each slope
magnitude has 4 / 4 / 2 columns. Direction alternates by row and column,
so each row has five up/down cells per magnitude and each stiffness column
sees both directions across rows. Not a full factorial design at each length.
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
LENGTH_MIN_M = 0.6
LENGTH_MAX_M = 1.2
GRID_SIZE = COLUMNS * ROWS

# k [N/m], c [N s/m], unchanged from the flat towing baseline.
ELASTIC_KC = ((1000.0, 50.0), (4000.0, 100.0), (20000.0, 220.0), (100000.0, 460.0))
SLACK_RATIO = 0.5


def row_length(row: int) -> float:
    if not 0 <= row < ROWS:
        raise ValueError(f"row must be in [0, {ROWS}), got {row}")
    return LENGTH_MIN_M + (LENGTH_MAX_M - LENGTH_MIN_M) * row / (ROWS - 1)


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
    """Signed grade along +X. Alternate directions without fixing k to up/down."""
    if not 0 <= column < COLUMNS or not 0 <= row < ROWS:
        raise ValueError(f"invalid cell row={row}, column={column}")
    if column < FLAT_COLUMNS:
        return 0.0
    magnitude = 5.0 if column < FLAT_COLUMNS + FIVE_DEGREE_COLUMNS else 10.0
    return magnitude if (column - FLAT_COLUMNS + row) % 2 == 0 else -magnitude


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
    length = row_length(row)
    return {
        "env_index": env_index, "grid_index": index, "column": column, "row": row,
        "model_name": model_name, "model_index": MODEL_INDEX[model_name],
        "length": length, "stiffness": stiffness, "damping": damping,
        "initial_distance": initial_attachment_distance(model_name, length),
        "slope_degrees": slope_degrees(column, row),
    }


def is_full_grid(num_envs: int) -> bool:
    return num_envs > 0 and num_envs % GRID_SIZE == 0
