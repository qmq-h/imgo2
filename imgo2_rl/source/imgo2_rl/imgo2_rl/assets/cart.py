"""Passive cart import and effort passthrough. Import after AppLauncher."""

import os
from pathlib import Path

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg

from .cart_model import read_cart_model

_REPO_ROOT = Path(__file__).resolve().parents[5]
_DEFAULT_URDF_PATH = _REPO_ROOT / "imgo2_description" / "cart" / "cart.urdf"


def resolve_cart_path():
    override = os.environ.get("IMGO2_CART_URDF_PATH")
    return (Path(override).expanduser() if override else _DEFAULT_URDF_PATH).resolve()


class PassiveCartArticulation(Articulation):
    """Submit efforts without an actuator model or position/velocity servo.

    Isaac Lab fills its simulation effort buffer inside the actuator loop.
    An empty actuator map can otherwise leave that buffer at zero. Submit the
    public effort target after the normal write; both writes precede sim.step.
    This adapter is specific to the PhysX tensor backend.
    """

    def write_data_to_sim(self):
        if self.cfg.actuators:
            raise ValueError("PassiveCartArticulation requires actuators={}")
        super().write_data_to_sim()
        indices = torch.arange(self.num_instances, dtype=torch.long, device=self.device)
        self.root_physx_view.set_dof_actuation_forces(self.data.joint_effort_target, indices)


def make_cart_cfg(usd_dir: Path, *, drop_height: float = 0.03):
    path = resolve_cart_path()
    model = read_cart_model(path)
    cfg = ArticulationCfg(
        class_type=PassiveCartArticulation,
        prim_path="{ENV_REGEX_NS}/Cart",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(path), usd_dir=str(usd_dir), usd_file_name="cart.usd",
            force_usd_conversion=True, make_instanceable=False,
            fix_base=False, merge_fixed_joints=True, replace_cylinders_with_capsules=False,
            activate_contact_sensors=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                target_type="none", gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, linear_damping=0, angular_damping=0,
                max_depenetration_velocity=1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=8,
                solver_velocity_iteration_count=4),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, model["resting_height_m"] + drop_height),
            joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
        actuators={},
    )
    return cfg, model
