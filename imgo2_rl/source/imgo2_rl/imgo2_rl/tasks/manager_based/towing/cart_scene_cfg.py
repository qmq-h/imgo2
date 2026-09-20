"""P1/P2 independent cart scene (no robot and no RL policy)."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass


@configclass
class CartSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.8, restitution=0.0,
                friction_combine_mode="average", restitution_combine_mode="min")))
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000))
    cart: ArticulationCfg | None = None
    wheel_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Cart/wheel_.*", update_period=0.0, history_length=1)
