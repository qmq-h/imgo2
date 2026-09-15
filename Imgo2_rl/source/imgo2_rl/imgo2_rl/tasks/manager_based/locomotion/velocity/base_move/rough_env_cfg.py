# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from imgo2_rl.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from imgo2_rl.assets.imgo2 import IMGO2_CFG  # isort: skip

# TODO: 站立高度有问题，平面运动速度跟随还需要增强，抬腿高度也还需要增加，稳定性不赖

@configclass
class Imgo2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
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

        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_height_range = (0.025, 0.08)
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.025, 0.08)

        # ------------------------------Observations------------------------------
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

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "Imgo2RoughEnvCfg":
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
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.8, 0.8)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
