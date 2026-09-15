import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import imgo2_rl.tasks.manager_based.locomotion.velocity.mdp as mdp
from imgo2_rl.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

##
# Pre-defined configs
##
from imgo2_rl.assets.imgo2 import IMGO2_CFG  # isort: skip

from .env import rewards


class Imgo2HandstandRoughRewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""

    handstand_hip_height_exp = RewTerm(
        func=rewards.link_height_exp,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "target_height": 0.0, "std": math.sqrt(0.25)},
    )


    handstand_feet_height_exp = RewTerm(
        func=rewards.link_height_exp,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "target_height": 0.0, "std": math.sqrt(0.25)},
    )

    handstand_feet_on_air = RewTerm(
        func=rewards.handstand_feet_on_air,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
        },
    )

    handstand_feet_air_time = RewTerm(
        func=rewards.handstand_feet_air_time,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "threshold": 5.0,
        },
    )

    handstand_stand_feet_air_time = RewTerm(
        func=rewards.handstand_stand_feet_air_time,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "threshold": 5.0,
            "handstand_type": "back",
        },
    )

    handstand_stand_feet_height_body = RewTerm(
        func=rewards.handstand_stand_feet_height_body,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "target_height": -0.4,
            "tanh_mult": 2.0,
            "handstand_type": "back",
        },
    )


    handstand_orientation_l2 = RewTerm(
        func=rewards.handstand_orientation_l2,
        weight=0.0,
        params={
            "target_gravity": [],
        },
    )


@configclass
class Imgo2HandstandRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: Imgo2HandstandRoughRewardsCfg = Imgo2HandstandRoughRewardsCfg()

    base_link_name = "base"
    foot_link_name = ".*_FOOT"
    # fmt: off
    joint_names = [
        "FL_hip_joint", "FL_thigh_joint", "FL_shank_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_shank_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_shank_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_shank_joint",
    ]
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = IMGO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------Observations------------------------------
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------Events------------------------------
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (-3.14, 3.14),
                "pitch": (-3.14, 3.14),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_rigid_body_com.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque = None

        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = 0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = 0
        self.rewards.ang_vel_xy_l2.weight = 0
        self.rewards.flat_orientation_l2.weight = 0
        self.rewards.base_height_l2.weight = 0
        self.rewards.base_height_l2.params["target_height"] = 0.72
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -1e-5
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_acc_l2.weight = -1e-7
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.2, [".*_hip_joint"])
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_power.weight = -1e-5
        self.rewards.stand_still.weight = 0

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.01

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = 0
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_yaw_frame_exp
        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_world_exp


        # Others   useless
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]

        # HandStand
        handstand_type = "back"  # which leg on air, can be "front", "back", "left", "right"
        # default
        air_foot_name = "F.*_FOOT"
        stand_foot_name = "R.*_FOOT"
        air_hip_name = "F.*_HIP"
        stand_hip_name = "R.*_HIP"
        if handstand_type == "front":
            self.rewards.handstand_orientation_l2.weight = -1.0
            self.rewards.handstand_orientation_l2.params["target_gravity"] = [-1.0, 0.0, 0.0]
            self.rewards.handstand_feet_height_exp.params["target_height"] = 0.7
            self.rewards.handstand_hip_height_exp.params["target_height"] = 0.94
        elif handstand_type == "back":
            air_foot_name = "R.*_FOOT"
            stand_foot_name = "F.*_FOOT"
            air_hip_name = "R.*_HIP"
            stand_hip_name = "F.*_HIP"
            self.rewards.handstand_orientation_l2.weight = -1.0
            self.rewards.handstand_orientation_l2.params["target_gravity"] = [1.0, 0.0, 0.0]
            self.rewards.handstand_feet_height_exp.params["target_height"] = 0.7
            self.rewards.handstand_hip_height_exp.params["target_height"] = 0.94

        # handstand
        self.rewards.handstand_feet_height_exp.weight = 10
        self.rewards.handstand_feet_height_exp.params["asset_cfg"].body_names = [air_foot_name]
        self.rewards.handstand_hip_height_exp.weight = 8
        self.rewards.handstand_hip_height_exp.params["asset_cfg"].body_names = [air_hip_name]
        self.rewards.handstand_feet_on_air.weight = 5.0
        self.rewards.handstand_feet_on_air.params["sensor_cfg"].body_names = [air_foot_name]
        # TODO: ? 
        # self.rewards.handstand_feet_air_time.weight = 5.0
        self.rewards.handstand_feet_air_time.weight = 0.0
        self.rewards.handstand_feet_air_time.params["sensor_cfg"].body_names = [air_foot_name]


        # 对站立腿的摆动奖励，只在基本站立时生效
        self.rewards.handstand_stand_feet_air_time.weight = 1.0
        self.rewards.handstand_stand_feet_air_time.params["command_name"] = "base_velocity"
        self.rewards.handstand_stand_feet_air_time.params["sensor_cfg"].body_names = [stand_foot_name]
        self.rewards.handstand_stand_feet_air_time.params["threshold"] = 0.5
        self.rewards.handstand_stand_feet_air_time.params["handstand_type"] = handstand_type
        self.rewards.handstand_stand_feet_height_body.weight = -2.5
        self.rewards.handstand_stand_feet_height_body.params["target_height"] = -0.4
        self.rewards.handstand_stand_feet_height_body.params["asset_cfg"].body_names = [stand_foot_name]
        self.rewards.handstand_stand_feet_height_body.params["handstand_type"] = handstand_type

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "Imgo2HandstandRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name, ".*_hip"]
        self.terminations.illegal_contact = None

        # ------------------------------Curriculums------------------------------
        # self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (0.2, 1.0)
        # self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (0.2, 1.0)
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        # ------------------------------Commands------------------------------
        # self.commands.base_velocity.ranges.lin_vel_x = (-2.0, 2.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (-2.0, 2.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)

