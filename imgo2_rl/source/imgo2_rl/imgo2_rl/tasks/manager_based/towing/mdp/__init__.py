"""Towing physics and manager terms: wheel resistance, rope models, and task signals.

Implementations are added with their experiments; importing this package
must not launch a simulator or modify a stage. All modules re-exported below
are pure arithmetic (no torch / Isaac Lab import), so that constraint holds and
the physics stays verifiable on a machine without a simulator.

有三种可切换的拖曳连接模型（`rope_model = "compliant" | "inextensible" | "rigid"`）：
`rope.py` 是共用算术与 compliant 公式，`rope_model.py` 是统一接口与三套实现
（前两者是绳、只有拉力，`rigid` 是双边球铰连杆、可推可拉）。
`connection_grid.py` 是训练场景的「列 × 行」确定性网格（列 = 类型/弹性档，行 = 长度）。
"""

from .connection_grid import (
    COLUMNS,
    ELASTIC_KC,
    GRID_SIZE,
    LENGTH_MAX_M,
    LENGTH_MIN_M,
    MODEL_INDEX,
    ROWS,
    SLACK_RATIO,
    column_spec,
    env_spec,
    initial_attachment_distance,
    is_full_grid,
    row_length,
)
from .resistance import viscous_resistance
from .rope import (
    RopeState,
    config_value,
    cross,
    offset_torque,
    point_velocity,
    rope_extension,
    rope_state,
    rope_tension,
)
from .rope_model import (
    CONNECTION_MODELS,
    RIGID_MODEL,
    ROPE_MODELS,
    SLACK,
    TAUT,
    BodyProperties,
    CompliantRope,
    InextensibleRope,
    MultiRopeModel,
    RigidLink,
    RopeModel,
    RopeSample,
    SplitRopeModel,
    attachment_horizontal_gap,
    effective_inverse_mass,
    make_rope_model,
)

__all__ = [
    "COLUMNS",
    "CONNECTION_MODELS",
    "ELASTIC_KC",
    "GRID_SIZE",
    "LENGTH_MAX_M",
    "LENGTH_MIN_M",
    "MODEL_INDEX",
    "RIGID_MODEL",
    "ROPE_MODELS",
    "ROWS",
    "SLACK",
    "SLACK_RATIO",
    "TAUT",
    "BodyProperties",
    "CompliantRope",
    "InextensibleRope",
    "MultiRopeModel",
    "RigidLink",
    "RopeModel",
    "RopeSample",
    "RopeState",
    "SplitRopeModel",
    "attachment_horizontal_gap",
    "column_spec",
    "config_value",
    "cross",
    "effective_inverse_mass",
    "env_spec",
    "initial_attachment_distance",
    "is_full_grid",
    "make_rope_model",
    "offset_torque",
    "point_velocity",
    "rope_extension",
    "rope_state",
    "rope_tension",
    "row_length",
    "viscous_resistance",
]
