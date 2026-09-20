"""Wheel bearing resistance; no propulsion or target-speed controller."""

import math


def viscous_resistance(angular_velocity, damping: float):
    """Return -b*omega for a scalar, numpy array, or torch tensor (SI units)."""
    if not math.isfinite(damping) or damping < 0:
        raise ValueError("Wheel damping must be finite and nonnegative")
    return -damping * angular_velocity
