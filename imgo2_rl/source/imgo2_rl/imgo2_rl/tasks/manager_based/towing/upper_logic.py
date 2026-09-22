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
        if any(lo > 0.0 or hi < 0.0 for lo, hi in
               zip(self.acceleration_min, self.acceleration_max)):
            raise ValueError("每维 acceleration 范围必须包含零")


@dataclass(frozen=True)
class UpperObservationSpec:
    """Raw recurrent input plus the deployable decoder estimate."""

    terms: tuple[tuple[str, int], ...] = (
        ("cmd_vel", 3),       # [vx, vy, yaw_rate]
        ("reference_command", 3),
        ("last_action", 3),      # [ax, ay, yaw_acceleration]
        ("base_ang_vel", 3),     # body-frame IMU gyroscope
        ("projected_gravity", 3),
        ("last_loco_action", 12),
        ("joint_pos", 12),
        ("joint_vel", 12),
    )
    decoder_dim: int = 5

    @property
    def frame_dim(self):
        return sum(dim for _, dim in self.terms)

    @property
    def actor_dim(self):
        return self.frame_dim + self.decoder_dim


@dataclass(frozen=True)
class DecoderSpec:
    """GRU decoder targets used only during training."""

    terms: tuple[tuple[str, int], ...] = (
        ("robot_velocity_xy", 2),
        ("cart_mass", 1),
        ("towing_force_xy", 2),
    )

    @property
    def dim(self):
        return sum(dim for _, dim in self.terms)

    mass_range: tuple[float, float] = (5.0, 15.0)
    velocity_scale: tuple[float, float] = (1.0, 0.5)
    force_scale: float = 10.0


def normalize_decoder_targets(values, spec: DecoderSpec = DecoderSpec()):
    """Normalize [vx, vy, mass, Fx, Fy] without importing torch."""
    if len(values) != spec.dim:
        raise ValueError(f"decoder target 维数 {len(values)} != {spec.dim}")
    lower, upper = spec.mass_range
    velocity = tuple(
        min(1.0, max(-1.0, float(value) / scale))
        for value, scale in zip(values[:2], spec.velocity_scale))
    mass = min(upper, max(lower, float(values[2])))
    force = tuple(
        float(value) / (abs(float(value)) + spec.force_scale)
        for value in values[3:])
    return (*velocity, 2.0 * (mass - lower) / (upper - lower) - 1.0, *force)


def denormalize_force(value, spec: DecoderSpec = DecoderSpec()):
    normalized = min(1.0 - 1.0e-6, max(-1.0 + 1.0e-6, float(value)))
    return spec.force_scale * normalized / (1.0 - abs(normalized))


def normalized_acceleration(action, spec: UpperActionSpec = UpperActionSpec()):
    """Map three normalized actions to [ax, ay, yaw_acceleration]."""
    spec.validate()
    if len(action) != 3:
        raise ValueError("upper action 必须是 3 维 [ax, ay, a_yaw]")
    result = []
    for value, lower, upper in zip(action, spec.acceleration_min, spec.acceleration_max):
        u = min(1.0, max(-1.0, float(value)))
        result.append((-u * lower) if u < 0.0 else (u * upper))
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
