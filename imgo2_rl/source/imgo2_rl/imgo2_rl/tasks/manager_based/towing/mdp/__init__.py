"""Towing physics and manager terms: wheel resistance, rope models, and task signals.

Implementations are added with their experiments; importing this package
must not launch a simulator or modify a stage. All modules re-exported below
are pure arithmetic (no torch / Isaac Lab import), so that constraint holds and
the physics stays verifiable on a machine without a simulator.

绳索有两套可切换模型（`rope_model = "compliant" | "inextensible"`）：
`rope.py` 是共用算术与 compliant 公式，`rope_model.py` 是统一接口与两套实现。
"""

from .resistance import viscous_resistance
from .rope import (
    RopeState,
    cross,
    offset_torque,
    point_velocity,
    rope_extension,
    rope_state,
    rope_tension,
)
from .rope_model import (
    ROPE_MODELS,
    SLACK,
    TAUT,
    BodyProperties,
    CompliantRope,
    InextensibleRope,
    RopeModel,
    RopeSample,
    effective_inverse_mass,
    make_rope_model,
)

__all__ = [
    "ROPE_MODELS",
    "SLACK",
    "TAUT",
    "BodyProperties",
    "CompliantRope",
    "InextensibleRope",
    "RopeModel",
    "RopeSample",
    "RopeState",
    "cross",
    "effective_inverse_mass",
    "make_rope_model",
    "offset_torque",
    "point_velocity",
    "rope_extension",
    "rope_state",
    "rope_tension",
    "viscous_resistance",
]
