"""Optional joint-side resistance on top of the Isaac Lab DC motor."""

import math

import torch
from isaaclab.actuators import DCMotor, DCMotorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class ResistanceDCMotor(DCMotor):
    """Apply a passive torque after the unchanged motor torque-speed limit."""

    def __init__(self, cfg, *args, **kwargs):
        if any(getattr(cfg, name, 0.0) != 0.0 for name in
               ("friction", "dynamic_friction", "viscous_friction")):
            raise ValueError("ResistanceDCMotor requires zero native joint friction")
        super().__init__(cfg, *args, **kwargs)
        self.fs = self._resistance_parameter(cfg.fs_nm, 0.0, "fs_nm")
        self.b = self._resistance_parameter(cfg.b_nms, 0.0, "b_nms")
        self.va = self._resistance_parameter(cfg.va_rad_s, 0.02, "va_rad_s")
        self.motor_effort = torch.zeros_like(self.computed_effort)
        self.resistance_effort = torch.zeros_like(self.computed_effort)

    def _resistance_parameter(self, values, default, name):
        unknown = values.keys() - set(self.joint_names)
        if unknown:
            raise ValueError(f"{name}: unknown joints {sorted(unknown)}")
        resolved = [values.get(joint, default) for joint in self.joint_names]
        if any(not math.isfinite(v) or v < 0 or (name == "va_rad_s" and v == 0) for v in resolved):
            raise ValueError(f"{name}: invalid resistance parameter")
        return self.computed_effort.new_tensor(resolved).expand_as(self.computed_effort).clone()

    def compute(self, control_action, joint_pos, joint_vel) -> ArticulationActions:
        action = super().compute(control_action, joint_pos, joint_vel)
        self.motor_effort = self.applied_effort
        self.resistance_effort = self.fs * torch.tanh(joint_vel / self.va) + self.b * joint_vel
        self.applied_effort = self.motor_effort - self.resistance_effort
        action.joint_efforts = self.applied_effort
        return action


@configclass
class ResistanceDCMotorCfg(DCMotorCfg):
    """Exact joint-name overrides in Nm, Nm·s/rad, and rad/s, respectively."""

    class_type: type = ResistanceDCMotor
    friction: float = 0.0
    dynamic_friction: float = 0.0
    viscous_friction: float = 0.0
    fs_nm: dict[str, float] = {}
    b_nms: dict[str, float] = {}
    va_rad_s: dict[str, float] = {}
