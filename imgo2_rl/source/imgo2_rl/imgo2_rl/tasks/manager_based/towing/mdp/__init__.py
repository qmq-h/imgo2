"""Towing physics and manager terms: wheel resistance, rope, and task signals.

Implementations are added with their experiments; importing this package
must not launch a simulator or modify a stage. Both modules re-exported below
are pure arithmetic (no torch / Isaac Lab import), so that constraint holds and
the physics stays verifiable on a machine without a simulator.
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

__all__ = [
    "RopeState",
    "cross",
    "offset_torque",
    "point_velocity",
    "rope_extension",
    "rope_state",
    "rope_tension",
    "viscous_resistance",
]
