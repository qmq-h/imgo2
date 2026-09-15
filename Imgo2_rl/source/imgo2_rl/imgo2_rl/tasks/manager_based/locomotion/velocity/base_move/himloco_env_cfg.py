import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from imgo2_rl.assets.imgo2 import IMGO2_CFG  # isort: skip
import imgo2_rl.tasks.manager_based.locomotion.velocity.mdp as mdp
from imgo2_rl.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    CommandsCfg,
    CurriculumCfg,
    EventCfg,
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
    TerminationsCfg,
)


@configclass
class HIMObservationCfg:
    """HimLoco observation layout. Keep policy and critic term order fixed."""

    @configclass
    class PolicyCfg(ObsGroup):
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self):
            self.enable_corruption = True

    @configclass
    class CriticCfg(ObsGroup):
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0, clip=(-100, 100), noise=Unoise(n_min=-0.1, n_max=0.1))
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class Imgo2HimlocoRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    observations: HIMObservationCfg = HIMObservationCfg()

    base_link_name = "base"
    foot_link_name = ".*_FOOT"
    joint_names = [
        "FL_hip_joint", "FL_thigh_joint", "FL_shank_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_shank_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_shank_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_shank_joint",
    ]

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------Scene------------------------------
        self.scene.robot = IMGO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # ------------------------------Obs------------------------------
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos_rel.scale = 1.0
        self.observations.policy.joint_pos_rel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel_rel.scale = 0.05
        self.observations.policy.joint_vel_rel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.base_lin_vel.scale = 2.0

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.joint_pos.clip = {".*": (-3.0, 3.0)}
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------Events------------------------------
        self.events.randomize_reset_base.params = {
            # "pose_range": {
            #     "x": (-0.5, 0.5),
            #     "y": (-0.5, 0.5),
            #     "z": (0.0, 0.2),
            #     "roll": (-3.14, 3.14),
            #     "pitch": (-3.14, 3.14),
            #     "yaw": (-3.14, 3.14),
            # },
            # "velocity_range": {
            #     "x": (-0.5, 0.5),
            #     "y": (-0.5, 0.5),
            #     "z": (-0.5, 0.5),
            #     "roll": (-0.5, 0.5),
            #     "pitch": (-0.5, 0.5),
            #     "yaw": (-0.5, 0.5),
            # },
            "pose_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (0.0, 0.0),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.0, 0.0),
            },
        }
        # 添加外部力/力矩，抗干扰的东西
        # 添加推动机器人的随机
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None

        # ------------------------------Rewards------------------------------
        # 前后动作差异
        self.rewards.action_rate_l2.weight = -0.01
        # self.rewards.smoothness_2.weight = -0.0075
        # 基座高度和期望值对比
        self.rewards.base_height_l2.weight = -10.0 
        self.rewards.base_height_l2.params["target_height"] = 0.30
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        # 足端滞空时间
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        ## 足端运动均匀，惩罚运动中足端滞空时间的方差
        self.rewards.feet_air_time_variance.weight = -8.0
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        ## 足端滑动惩罚
        self.rewards.feet_slide.weight = -0.05
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        ## 正确站定时候的检测
        self.rewards.stand_still.weight = -0.1
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.stand_still.params["command_threshold"] = 0.05
        ## 足端和基座的距离
        self.rewards.feet_height_body.weight = -5.0
        self.rewards.feet_height_body.params["target_height"] = -0.20
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        ## 足端在踏步时候的期望高度
        self.rewards.feet_height.weight = -0.0
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.params["target_height"] = 0.08
        ## 
        self.rewards.joint_mirror.weight = -1.0
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|shank).*", "RL_(hip|thigh|shank).*"],
            ["FL_(hip|thigh|shank).*", "RR_(hip|thigh|shank).*"],
        ]

        ## 接触地面时候的触地力
        self.rewards.contact_forces.weight = -2e-2
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # 不希望基座上下移动
        self.rewards.lin_vel_z_l2.weight = -2.0
        # 横滚和俯仰的限制
        self.rewards.ang_vel_xy_l2.weight = -0.05

        # 跟踪奖励
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.6

        # 计划外的碰撞
        self.rewards.undesired_contacts.weight = -0.5
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]

        # 关节扭矩、加速度限制
        self.rewards.joint_torques_l2.weight = -2.5e-6
        self.rewards.joint_acc_l2.weight = -5e-9
        # 髋关节的限制，尽量为
        self.rewards.joint_deviation_l1.weight = 0.0
        self.rewards.joint_deviation_l1.params["asset_cfg"].joint_names = [".*hip.*"]
        self.rewards.joint_pos_penalty.weight = -0.1
        
        # 功率限制
        self.rewards.joint_power.weight = -2e-5
        # 基座水平的惩罚
        self.rewards.flat_orientation_l2.weight = -5.0

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact = None

        # self.events.randomize_reset_base.params = {
        #     "pose_range": {
        #         "x": (-0.5, 0.5),
        #         "y": (-0.5, 0.5),
        #         "z": (0.0, 0.2),
        #         "roll": (-3.14, 3.14),
        #         "pitch": (-3.14, 3.14),
        #         "yaw": (-3.14, 3.14),
        #     },
        #     "velocity_range": {
        #         "x": (-0.5, 0.5),
        #         "y": (-0.5, 0.5),
        #         "z": (-0.5, 0.5),
        #         "roll": (-0.5, 0.5),
        #         "pitch": (-0.5, 0.5),
        #         "yaw": (-0.5, 0.5),
        #     },
        # }
        # self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        # self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            # f"^(?!.*{self.base_link_name}).*"
        # ]
        # self.events.randomize_rigid_body_com.params["asset_cfg"].body_names = [self.base_link_name]
        # self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        # self.events.randomize_apply_external_force_torque.params["force_range"] = (-30.0, 30.0)
        # self.events.randomize_apply_external_force_torque.params["torque_range"] = (-10.0, 10.0)
        # self.events.randomize_apply_external_force_torque = None

        # self.rewards.is_terminated.weight = 0.0
        # self.rewards.energy.weight = -2e-5
        # self.rewards.lin_vel_z_l2.weight = -2.0
        # self.rewards.ang_vel_xy_l2.weight = -0.05
        # self.rewards.flat_orientation_l2.weight = -2.0
        # self.rewards.base_height_l2.weight = -5.0
        # self.rewards.base_height_l2.params["target_height"] = 0.38
        # self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # self.rewards.joint_acc_l2.weight = -1e-9
        # self.rewards.joint_torques_l2.weight = 0.0
        # self.rewards.joint_vel_l2.weight = 0.0
        # self.rewards.joint_pos_limits.weight = 0.0
        # self.rewards.joint_pos.weight = 0.0
        # self.rewards.joint_vel_limits.weight = 0.0
        # self.rewards.applied_torque_limits.weight = 0.0

        # self.rewards.action_rate_l2.weight = -0.01
        # self.rewards.smoothness.weight = -0.001

        # self.rewards.other_undesired_contacts.weight = -1.0
        # self.rewards.other_undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        # self.rewards.head_undesired_contacts.weight = -1.0
        # self.rewards.head_undesired_contacts.params["sensor_cfg"].body_names = [self.base_link_name]

        # self.rewards.track_lin_vel_xy_exp.weight = 1.0
        # self.rewards.track_ang_vel_z_exp.weight = 0.5

        # self.rewards.feet_height_body.weight = -1.0
        # self.rewards.feet_height_body.params["target_height"] = -0.25
        # self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]

        if self.__class__.__name__ == "Imgo2HimlocoRoughEnvCfg":
            self.disable_zero_weight_rewards()

        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)

@configclass
class Imgo2HimlocoRoughPlayEnvCfg(Imgo2HimlocoRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.terrain.terrain_generator.num_cols = 10
        self.scene.terrain.max_init_terrain_level = 10
        self.scene.terrain.terrain_generator.curriculum = True
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(1.0, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        )
        self.commands.base_velocity.low_vel_env_lin_x_ranges = (1.0, 1.0)

        if self.__class__.__name__ == "Imgo2HimlocoRoughPlayEnvCfg":
            self.disable_zero_weight_rewards()

        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_rigid_body_com = None
        self.events.randomize_push_robot = None
