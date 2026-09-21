"""Simulator-independent contracts for the upper towing policy."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class UpperActionSpec:
    control_dt: float = 0.05
    acceleration_min: tuple[float, float, float] = (-1.0, -0.5, -1.0)
    acceleration_max: tuple[float, float, float] = (0.5, 0.5, 1.0)
    reference_min: tuple[float, float, float] = (0.0, -0.3, -1.0)
    reference_max: tuple[float, float, float] = (1.0, 0.3, 1.0)

    def validate(self):
        values = (self.control_dt, *self.acceleration_min, *self.acceleration_max,
                  *self.reference_min, *self.reference_max)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("action spec 必须全部为有限数")
        if self.control_dt <= 0 or any(lo >= hi for lo, hi in
                                      zip(self.acceleration_min, self.acceleration_max)):
            raise ValueError("control_dt 必须为正，且每维 acceleration_min < acceleration_max")
        if any(lo >= hi for lo, hi in zip(self.reference_min, self.reference_max)):
            raise ValueError("每维 reference_min 必须小于 reference_max")
        if self.reference_min[0] < 0:
            raise ValueError("第一版纵向参考速度不允许倒车")


@dataclass(frozen=True)
class UpperObservationSpec:
    """The exact v0.1 actor contract requested in paper_plan_rl.md."""

    terms: tuple[tuple[str, int], ...] = (
        ("cmd_vel", 3),       # [vx, vy, yaw_rate]
        ("last_action", 3),      # [ax, ay, yaw_acceleration]
        ("base_ang_vel", 3),     # body-frame IMU gyroscope
        ("projected_gravity", 3),
        ("last_loco_action", 12),
        ("joint_pos", 12),
        ("joint_vel", 12),
    )
    history_steps: int = 2

    @property
    def frame_dim(self):
        return sum(dim for _, dim in self.terms)

    @property
    def actor_dim(self):
        return self.frame_dim * self.history_steps


@dataclass(frozen=True)
class DecoderSpec:
    """First-stage privileged target used only by the training-time decoder."""

    terms: tuple[tuple[str, int], ...] = (
        ("cart_mass", 1),
    )

    @property
    def dim(self):
        return sum(dim for _, dim in self.terms)

    # Physical ranges used to normalize regression targets.  Values outside are clipped and logged;
    # these are training-domain contracts, not claims about the robot's absolute capability.
    ranges: tuple[tuple[float, float], ...] = (
        (5.0, 15.0),       # kg; matches the first training curriculum
    )


def normalize_decoder_targets(values, spec: DecoderSpec = DecoderSpec()):
    """Normalize a flat target sequence to [-1, 1] without importing torch."""
    if len(values) != spec.dim:
        raise ValueError(f"decoder target 维数 {len(values)} != {spec.dim}")
    normalized = []
    for value, (lower, upper) in zip(values, spec.ranges):
        clipped = min(upper, max(lower, float(value)))
        normalized.append(2.0 * (clipped - lower) / (upper - lower) - 1.0)
    return tuple(normalized)


def normalized_acceleration(action, spec: UpperActionSpec = UpperActionSpec()):
    """Map three normalized actions to [ax, ay, yaw_acceleration]."""
    spec.validate()
    if len(action) != 3:
        raise ValueError("upper action 必须是 3 维 [ax, ay, a_yaw]")
    result = []
    for value, lower, upper in zip(action, spec.acceleration_min, spec.acceleration_max):
        u = min(1.0, max(-1.0, float(value)))
        result.append(lower + 0.5 * (u + 1.0) * (upper - lower))
    return tuple(result)


def integrate_reference_speed(reference, action, spec: UpperActionSpec = UpperActionSpec()):
    if len(reference) != 3:
        raise ValueError("reference 必须是 3 维 [vx, vy, yaw_rate]")
    acceleration = normalized_acceleration(action, spec)
    return tuple(min(hi, max(lo, value + accel * spec.control_dt))
                 for value, accel, lo, hi in
                 zip(reference, acceleration, spec.reference_min, spec.reference_max))


def scheduled_command(elapsed_s, tow_speed, *, tow_start_s=1.0, stop_time_s=5.0):
    """Return the v0.1 longitudinal command for settle, tow, and zero-command phases."""
    if not 0.0 <= tow_start_s < stop_time_s:
        raise ValueError("必须满足 0 <= tow_start_s < stop_time_s")
    if not math.isfinite(elapsed_s) or not math.isfinite(tow_speed) or tow_speed < 0.0:
        raise ValueError("elapsed_s 和 tow_speed 必须为有限非负数")
    return float(tow_speed) if tow_start_s <= elapsed_s < stop_time_s else 0.0


def classify_training_case(*, valid: bool, tracking_ratio: float, tracking_min: float = 0.8) -> str:
    """Training-domain gate shared with the boundary-scan semantics."""
    return "feasible" if valid and tracking_ratio >= tracking_min else "infeasible"
