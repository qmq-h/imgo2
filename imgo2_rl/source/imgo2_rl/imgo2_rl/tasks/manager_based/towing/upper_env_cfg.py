"""Class-based configuration skeleton for the hierarchical towing RL environment.

This module intentionally defines the complete configuration contract but is not registered yet:
the ManagerBased per-physics-step rope/resistance adapter must first reproduce ``tow_drag.py``.
"""

from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from imgo2_rl.assets.cart import make_cart_cfg
from imgo2_rl.assets.imgo2 import IMGO2_CFG
import imgo2_rl.tasks.manager_based.towing.upper_mdp as mdp


_USD_CACHE = Path(__file__).resolve().parents[6] / "logs" / "usd" / "upper_towing"


@configclass
class UpperTowingSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.8, restitution=0.0,
                friction_combine_mode="average", restitution_combine_mode="min")))
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000))
    robot: ArticulationCfg = IMGO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cart: ArticulationCfg = make_cart_cfg(_USD_CACHE)[0]
    cart.init_state.pos = (-0.7564, 0.0, 0.18)
    # Each sensor body is filtered against all robot bodies. Separate sensors are required by
    # Isaac Lab's one-to-many filtering contract; wheel/ground forces never enter this matrix.
    cart_deck_robot_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Cart/base_link", update_period=0.0, history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/.*"])
    cart_wheel_fl_robot_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Cart/wheel_fl", update_period=0.0, history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/.*"])
    cart_wheel_fr_robot_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Cart/wheel_fr", update_period=0.0, history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/.*"])
    cart_wheel_rl_robot_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Cart/wheel_rl", update_period=0.0, history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/.*"])
    cart_wheel_rr_robot_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Cart/wheel_rr", update_period=0.0, history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/.*"])


@configclass
class UpperActionsCfg:
    high_level_velocity = mdp.HierarchicalVelocityActionCfg(
        asset_name="robot", cart_asset_name="cart", policy_name="amp",
        acceleration_min=(-1.0, -0.5, -1.0), acceleration_max=(0.5, 0.5, 1.0),
        reference_min=(0.0, -0.3, -1.0), reference_max=(1.0, 0.3, 1.0),
        upper_control_dt=0.05, low_level_control_dt=0.02,
    )


@configclass
class UpperObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        frame = ObsTerm(func=mdp.policy_frame)

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = True

    @configclass
    class CriticCfg(ObsGroup):
        policy_frame = ObsTerm(func=mdp.policy_frame)
        robot_velocity = ObsTerm(func=mdp.robot_velocity)
        cart_velocity = ObsTerm(func=mdp.cart_velocity)
        rope_state = ObsTerm(func=mdp.rope_privileged_state)
        towing_force = ObsTerm(func=mdp.towing_force)
        cart_parameters = ObsTerm(func=mdp.cart_privileged_parameters)

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    @configclass
    class DecoderCfg(ObsGroup):
        targets = ObsTerm(func=mdp.decoder_targets)
        mass_weight = ObsTerm(
            func=mdp.decoder_mass_weight,
            params={"minimum_force": 1.0, "force_scale": 10.0},
        )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    decoder: DecoderCfg = DecoderCfg()


@configclass
class UpperRewardsCfg:
    # Minimal v0 reward. Keep additional diagnostics out of the return until an observed failure
    # justifies adding a term.
    tracking_velocity = RewTerm(func=mdp.velocity_tracking_exp, weight=1.0,
                                params={"linear_std": 0.5, "yaw_std": 1.0})
    collision = RewTerm(func=mdp.cart_collision_cost, weight=-50.0)
    fall = RewTerm(func=mdp.robot_fall_cost, weight=-50.0,
                   params={"minimum_height": 0.18})
    clearance = RewTerm(func=mdp.clearance_barrier, weight=-1.0,
                        params={"warning_distance": 0.20, "scale": 0.05})
    stop_towing_force = RewTerm(func=mdp.post_stop_towing_force, weight=-1.0,
                                params={"force_scale": 10.0})
    extra_distance = RewTerm(func=mdp.post_stop_distance, weight=-0.1)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)


@configclass
class UpperTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    robot_fall = DoneTerm(func=mdp.robot_fall, params={"minimum_height": 0.18})
    cart_collision = DoneTerm(func=mdp.cart_collision)


@configclass
class UpperEventsCfg:
    reset_work_condition = EventTerm(
        func=mdp.reset_towing_episode,
        mode="reset",
        params={
            "speed_range": (0.2, 1.0),
            "stop_time_range": (4.0, 6.0),
            "mass_range": (5.0, 15.0),
            "friction_range": (0.4, 1.2),
            "wheel_damping_range": (0.008, 0.032),
            "robot_x_range": (-0.03, 0.03),
            "robot_y_range": (-0.02, 0.02),
            "robot_yaw_range": (-0.03, 0.03),
            # About 32 of 256 environments provide a zero-load anchor each reset batch.
            "no_cart_fraction": 0.125,
            "no_cart_lateral_offset": 2.0,
        },
    )


@configclass
class UpperTowingEnvCfg(ManagerBasedRLEnvCfg):
    scene: UpperTowingSceneCfg = UpperTowingSceneCfg(num_envs=256, env_spacing=6.0)
    observations: UpperObservationsCfg = UpperObservationsCfg()
    actions: UpperActionsCfg = UpperActionsCfg()
    rewards: UpperRewardsCfg = UpperRewardsCfg()
    terminations: UpperTerminationsCfg = UpperTerminationsCfg()
    events: UpperEventsCfg = UpperEventsCfg()
    commands = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 10              # upper policy: 0.005 * 10 = 0.05 s = 20 Hz
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.viewer.eye = (4.0, 4.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.2)
