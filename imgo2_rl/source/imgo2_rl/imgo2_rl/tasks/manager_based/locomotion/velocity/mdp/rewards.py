# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from rl_lab.envs import HimlocoManagerBasedRLEnv


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_world_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """**世界系** xy 线速度跟踪（对照 parkour 的 `_reward_tracking_world_vel`）。

    2026-09-24 加入（用户："该参考 parkour 用全局的速度来约束了"）。与机体系版本
    `track_lin_vel_xy_exp` 的区别是本项比较**世界系**速度 `root_lin_vel_w`，目标速度也换算到世界系。

    为什么需要它（回放实测的缺陷）：机体系版本 + `heading_command` 下，yaw 指令是 heading 控制器
    **按当前误差实时生成**的（转身本身几乎不被罚），而奖励只看机体系速度 ⇒ 机器人可以"一边转身
    一边在机体系里前进"来拿满分，于是**横移绕开障碍**仍能保持很高的跟踪奖励（实测线速度核 0.88、
    偏航核仅 0.46）。改成世界系后，目标方向由**命令的目标朝向**（`heading_target`，世界系）决定、
    **不随机器人自身转动**，横移/绕行立刻体现为巨大的世界系速度误差。

    目标速度的构造：命令是机体系 `(vx, vy)`，用命令的**目标朝向** `heading_target` 旋转到世界系
    ⇒ `target_w = R_z(heading_target) · (vx, vy)`。这样在障碍列（`heading_target = 0`）目标就是
    纯 `+x`；在普通列则跟随采样到的目标朝向，仍然要求"朝那个方向走"。
    要求命令项是 heading 命令（本项目 CMoE 的 `heading_command=True`、`rel_heading_envs=1.0`）；
    若取不到 `heading_target`，退化为用**当前**朝向旋转（等价机体系）。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd_b = env.command_manager.get_command(command_name)
    term = env.command_manager.get_term(command_name)
    heading = getattr(term, "heading_target", None)
    if heading is None:
        heading = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)[2]
    cos_h, sin_h = torch.cos(heading), torch.sin(heading)
    target_w = torch.stack(
        (cos_h * cmd_b[:, 0] - sin_h * cmd_b[:, 1], sin_h * cmd_b[:, 0] + cos_h * cmd_b[:, 1]), dim=1
    )
    lin_vel_error = torch.sum(torch.square(target_w - asset.data.root_lin_vel_w[:, :2]), dim=1)
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def _cached_terrain_mask(env: ManagerBasedRLEnv, terrain_names: tuple[str, ...]) -> torch.Tensor:
    """按地形列取掩码并**缓存到 env 上**（供逐调用计算掩码的奖励项用）。

    地形列（`terrain.terrain_types`）逐环境固定、不随课程变化 ⇒ 只需算一次。
    """
    key = "_cmoe_terrain_mask__" + "__".join(terrain_names)
    mask = getattr(env, key, None)
    if mask is None:
        mask = _terrain_type_mask(env, terrain_names)
        setattr(env, key, mask)
    return mask


def lin_pos_y(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    terrain_names: tuple[str, ...] = (),
) -> torch.Tensor:
    """偏离**本 tile 中心线**的横向距离 `|y − env_origin_y|`（对照 parkour 的 `_reward_lin_pos_y`）。

    `env_origins` 就是该环境的出生点，其 y 恰好落在 tile 的 y 中心（sub-terrain 的 `origin` 是
    `(spawn_x, 0.5*size[1], ·)`，而 tile 也是以 `0.5*size[1]` 为中心摆放的）⇒ 这个差就是"离赛道中心线多远"。

    `terrain_names` 非空时只在这些地形列上生效（本项目＝`forward_only_terrain_names` 那 4 类）：
    普通列允许按命令做横向机动（`vy` 有 ±0.3 的指令），不该被罚。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    lateral = torch.abs(asset.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1])
    if terrain_names:
        lateral = lateral * _cached_terrain_mask(env, tuple(terrain_names)).float()
    return lateral


def yaw_abs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    terrain_names: tuple[str, ...] = (),
) -> torch.Tensor:
    """偏航角绝对值 `|yaw|`（对照 parkour 的 `_reward_yaw_abs`，目标朝向 +x）。

    与 `lin_pos_y` 一样可用 `terrain_names` 限制到指定地形列。障碍列上命令的 `heading_target = 0`
    ⇒ 该项就是"别转身"。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)[2]
    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi  # 归一化到 (-pi, pi]
    reward = torch.abs(yaw)
    if terrain_names:
        reward = reward * _cached_terrain_mask(env, tuple(terrain_names)).float()
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize absolute joint power."""
    return joint_power(env, asset_cfg)


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    # Penalize motion when command is nearly zero.
    reward = mdp.joint_deviation_l1(env, asset_cfg)
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def him_joint_position_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """HimLoco joint-position penalty kept separate from the generic command-threshold variant."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.1, body_vel > velocity_threshold), reward, stand_still_scale * reward)


def wheel_vel_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    running_reward = torch.sum(in_air * joint_vel, dim=1)
    standing_reward = torch.sum(joint_vel, dim=1)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        standing_reward,
    )
    return reward


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def _terrain_type_mask(env: ManagerBasedRLEnv, terrain_names: tuple[str, ...]) -> torch.Tensor:
    """(N,) bool：环境所属**地形列**是否落在给定名单里。

    地形列（`terrain.terrain_types`）逐环境固定、不随课程变化，所以调用方应在 `__init__` 里算一次并缓存。
    `is_env_assigned_to_terrain` 对未登记的地形名返回全 False，因此写错名字只会静默失效、不会报错。
    """
    from .utils import is_env_assigned_to_terrain  # 延迟导入，避免包内循环依赖

    mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for terrain_name in terrain_names:
        mask = mask | is_env_assigned_to_terrain(env, terrain_name)
    return mask


def _ray_count(span: float, resolution: float) -> int:
    """某个轴上的射线数，镜像 `patterns.grid_pattern` 的 `arange(start, end + 1e-9, step)`。"""
    return math.floor(span / resolution + 1.0e-9) + 1


class TrotWithoutGapReward(GaitReward):
    """`GaitReward`（trot 相位塑形）的**按地形豁免**版本。

    掩码与另外四项步态 shaping **完全同一套**：`free_terrain_names`（默认 `("boxes", "gap")`）那一类
    地形整列放开相位（那里通常是 bound/跃起），其余列保留 `GaitReward` 的 **6 核乘积**
    （2 个"同步"核 × 4 个"反相"核）。

    为什么需要它（2026-09-24 用户："那就开 feet gait，同样加掩码"）：这是全配方里**唯一**能区分
    trot 与另外两种对称步态的项 ——
    * `joint_mirror` 只做"对角对内相等" ⇒ trot ✓、bound ✗、**pronk ✓**；
    * `feet_air_time_variance` 罚"四足时长方差" ⇒ bound ✗、**pronk ✓**；
    * 只有 `GaitReward` 的 4 个 async 核显式要求**对角对之间反相** ⇒ 排除 bound/pace/**pronk**。
    代价：6 核相乘是个很窄的脊，随机策略早期诸核≈0 ⇒ 乘积≈0、梯度≈0，学得慢
    ⇒ 所以它与 `joint_mirror`（稠密二次先验）**并用**才是互补的组合。

    历史：本类曾用过"地形类型 + 前方空洞（`height_scanner` 最前几列任一射线落空）+ 足下整束落空"
    的复合掩码；2026-09-24 晚按用户要求**统一成地形类型掩码**（与 `MaskedJointMirror` /
    `MaskedFeetHeightBody` / `MaskedFeetAirTime` / `MaskedFeetAirTimeVariance` 同一套
    `_terrain_type_mask`）。若要恢复"前方有空洞就放开相位"，实现见提交 `a30b25d`。
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.free_terrain_names: tuple[str, ...] = tuple(cfg.params.get("free_terrain_names", ("boxes", "gap")))
        self._free_mask = _terrain_type_mask(env, self.free_terrain_names)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        free_terrain_names: tuple[str, ...] = ("boxes", "gap"),
    ) -> torch.Tensor:
        del free_terrain_names  # 已在 __init__ 缓存为静态掩码
        trot = super().__call__(
            env,
            std,
            command_name,
            max_err,
            velocity_threshold,
            command_threshold,
            synced_feet_pair_names,
            asset_cfg,
            sensor_cfg,
        )
        return trot * (~self._free_mask).float()


def _pairwise_joint_mirror(
    env: ManagerBasedRLEnv,
    asset: Articulation,
    mirror_joints: list[list[str]],
    cache_owner,
    cache_attr: str,
) -> torch.Tensor:
    """`joint_mirror` 的逐对子实现，缓存挂在**调用方实例**上而不是 env 上。

    为什么要自己写：上游 `joint_mirror`（本文件下方）把解析结果缓存在 `env.joint_mirror_joints_cache`，
    而且**只在第一次调用时解析**（之后无论传什么 `mirror_joints` 都直接用缓存）⇒ 想在同一环境里
    按地形切换两套对子（trot 对角对 / bound 左右对）时，第二次调用会被**静默忽略**、两套都按第一套算。
    算式、归一化与直立门控与上游逐字一致。
    """
    cache = getattr(cache_owner, cache_attr, None)
    if not cache:
        cache = [[asset.find_joints(joint_name) for joint_name in pair] for pair in mirror_joints]
        setattr(cache_owner, cache_attr, cache)
    if len(cache) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    reward = torch.zeros(env.num_envs, device=env.device)
    for joint_pair in cache:
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(cache)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


class MaskedJointMirror(ManagerTermBase):
    """`joint_mirror` 的**分地形换对子／豁免**版本（2026-09-24）。

    三种地形行为：

    | 地形 | 对子 | 语义 |
    |---|---|---|
    | `bound_terrain_names`（默认 `("gap",)`） | `bound_mirror_joints`＝**左右对**（FL↔FR、RL↔RR） | **bound**：前腿一对同相、后腿一对同相（跃沟） |
    | `free_terrain_names`（默认 `("boxes",)`） | 不计 | 完全自由 |
    | 其余（trot 地形，13/20 列） | `mirror_joints`＝**对角对**（FR↔RL、FL↔RR） | **trot**：对角腿同相 |

    为什么沟壑要换成 bound 而不是单纯豁免：过沟需要的是**前腿一起／后腿一起**的 bound 式跃起，
    只把 mirror 关掉等于不给任何结构先验；换上左右对等于**把罚项变成"bound 的形状先验"**，
    与 parkour 的做法同源——他们的 `_reward_sync_all_legs_cond`（注释即 "force same actuation on
    both front/rear legs when jump"）比较的正是**右侧两腿 vs 左侧两腿**，且**只在 engage `jump`
    障碍时**生效；区别只是他们的 URDF 左右轴约定相反所以要翻转肩关节符号，本 URDF 四条腿轴完全相同，
    因此同相对子直接取**同号**即可。

    ⚠️ 边界：本项只做**对内相等**，不做**对间反相** ⇒ 它把沟壑上"对角同相（trot）"这一**错误**先验
    换成"前对同相 + 后对同相"，与 bound **一致**，但**同样与 pronk（四足全同相）一致**；
    真正的 front/rear 反相要靠 `feet_gait` 那类相位项或课程自行涌现。
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.free_terrain_names: tuple[str, ...] = tuple(cfg.params.get("free_terrain_names", ("boxes",)))
        self.bound_terrain_names: tuple[str, ...] = tuple(cfg.params.get("bound_terrain_names", ("gap",)))
        self._free_mask = _terrain_type_mask(env, self.free_terrain_names)
        # bound 与 free 同时命中时以 free 优先（free 更强：完全不计）
        self._bound_mask = _terrain_type_mask(env, self.bound_terrain_names) & (~self._free_mask)
        self._trot_pairs = None
        self._bound_pairs = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        mirror_joints: list[list[str]],
        free_terrain_names: tuple[str, ...] = ("boxes",),
        bound_terrain_names: tuple[str, ...] = ("gap",),
        bound_mirror_joints: list[list[str]] = (),
    ) -> torch.Tensor:
        del free_terrain_names, bound_terrain_names  # 已在 __init__ 缓存为静态掩码
        asset: Articulation = env.scene[asset_cfg.name]
        active = ~(self._free_mask | self._bound_mask)
        reward = _pairwise_joint_mirror(env, asset, mirror_joints, self, "_trot_pairs") * active.float()
        if len(bound_mirror_joints) > 0:
            reward = reward + _pairwise_joint_mirror(
                env, asset, bound_mirror_joints, self, "_bound_pairs"
            ) * self._bound_mask.float()
        return reward


class MaskedFeetAirTimeVariance(ManagerTermBase):
    """`feet_air_time_variance` 的**按地形豁免**版本（2026-09-24 晚，用户问"没看到平地 trot"）。

    语义：在 `free_terrain_names`（默认 `("boxes", "gap")`）上不生效，其余地形保留
    `Σ_var(clip(滞空)) + var(clip(触地))`（四足之间的**时序均匀度**惩罚，带直立门控）。

    为什么在这个时点把它加回来：它是那次「trot 还行」的 PPO 配方里**量级最大的步态项（−8.0）**，
    也是我们从 `6220e43` 清零后**唯一一直没有恢复**的一项（docs §29.8/§29.16）。三项 shaping 里
    `joint_mirror` 只做"对角对内相等"、`feet_air_time` 反而偏袒"同时腾空多"的腾跃式，
    **没有任何一项在管"四足之间的时序是否均匀"** —— 而 bound（前对与后对错开）恰恰表现为
    前/后足的滞空与触地时长不一致。

    ⚠️ 边界：本项罚的是"四足时长不一致" ⇒ 排除 **bound/pace**；但 **pronk（四足完全同步）满足它**。
    要排除 pronk 必须用相位项 `feet_gait`（`TrotWithoutGapReward`）。
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.free_terrain_names: tuple[str, ...] = tuple(cfg.params.get("free_terrain_names", ("boxes", "gap")))
        self._free_mask = _terrain_type_mask(env, self.free_terrain_names)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        free_terrain_names: tuple[str, ...] = ("boxes", "gap"),
    ) -> torch.Tensor:
        del free_terrain_names  # 已在 __init__ 缓存为静态掩码
        reward = feet_air_time_variance_penalty(env, sensor_cfg)
        return reward * (~self._free_mask).float()


class MaskedLinVelZ(ManagerTermBase):
    """`lin_vel_z_l2`（机体竖直速度平方）的**按地形豁免**版本（2026-09-24 晚）。

    背景：用户回放 1000 轮反馈"**都还是蹦蹦跳跳的走的**"。查下来这条是奖励结构造成的：
    * `lin_vel_z_l2` 在 A 配方里被**清零**（当时理由："会与过沟所需的爆发式跃起对抗"）；
    * `feet_air_time +1.0 @0.5` 的展开是 `4(1−d) − 2·N落地/T` ⇒ **滞空越久给得越多**
      ⇒ 主动奖励"腾空/弹跳"；测下来占空比一降，收益可达 +0.5/s 量级。
    ⇒ 于是"蹦蹦跳跳"正是这套奖励付钱买来的。修法与其它四项一致：**在 trot 列上恢复竖直速度罚，
      豁免 `boxes`/`gap`（那里需要爆发式跃起）**——既不与跃起对抗，又把弹跳压下去。
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.free_terrain_names: tuple[str, ...] = tuple(cfg.params.get("free_terrain_names", ("boxes", "gap")))
        self._free_mask = _terrain_type_mask(env, self.free_terrain_names)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        free_terrain_names: tuple[str, ...] = ("boxes", "gap"),
    ) -> torch.Tensor:
        del free_terrain_names  # 已在 __init__ 缓存为静态掩码
        return lin_vel_z_l2(env, asset_cfg) * (~self._free_mask).float()


class MaskedFeetHeightBody(ManagerTermBase):
    """`feet_height_body` 的**按地形豁免**版本（2026-09-24）。

    用户指出：parkour 是**在已训好的行走策略上**再 fine-tune 技能，所以可以完全不要 feet 相关奖励；
    我们是从零训，**需要**足端塑形。这里把"摆动足在机体系的高度误差"（`target_height` 默认 −0.20 m，
    即 base 下方 0.20 m ≈ 离地约 0.10 m）作为**trot 地形上**的塑形项恢复：
    障碍块/沟槽（`free_terrain_names`）豁免——那里需要抬得更高或干脆跃起，不该被"固定抬到 0.10 m"约束。

    参考基线（真机录制）：足端 z 峰峰值中位 **0.090 m** ⇒ 目标抬升 0.10 m 与录制步态同量级；
    当前台阶 5–15 cm 高于它，所以在台阶上该约束由课程与自由相位来处理（台阶不在豁免名单里，
    但 `feet_height_body` 是**惩罚偏差**而非硬约束，抬更高不会额外受罚……只有偏差会）。
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.free_terrain_names: tuple[str, ...] = tuple(cfg.params.get("free_terrain_names", ("boxes", "gap")))
        self._free_mask = _terrain_type_mask(env, self.free_terrain_names)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg,
        target_height: float,
        tanh_mult: float,
        free_terrain_names: tuple[str, ...] = ("boxes", "gap"),
    ) -> torch.Tensor:
        del free_terrain_names
        reward = feet_height_body(env, command_name, asset_cfg, target_height, tanh_mult)
        return reward * (~self._free_mask).float()


class MaskedFeetAirTime(ManagerTermBase):
    """`feet_air_time` 的**按地形豁免**版本（2026-09-24，用户要求"然后添加地形掩码"）。

    语义：在 `free_terrain_names`（默认 `("boxes", "gap")`）上不生效，其余地形保留 PPO rough 原样的
    滞空时间奖励（`threshold=0.5`）。

    **关于 threshold（2026-09-24 更正此前记录）**：展开成
    `Σ_feet(last_air_time − c) = Σ_feet last_air_time − c × N_落地`，所以 `c` 的作用是
    **每落地一次扣一个常数**——`c` 越大，"减少落地次数（步幅更长／步频更低）"的压力越大，
    同时整体回报被压得越低。它对 `last_air_time` 的**梯度方向（+1）与 c 无关**，
    因此 0.25 与 0.5 的**方向一致**，差别在强度（0.5 的"别踩碎步"压力是 0.25 的 2 倍）
    与日志里的偏移量（0.5 时该分项通常为负）。详见 docs §29.7、§29.9。
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.free_terrain_names: tuple[str, ...] = tuple(cfg.params.get("free_terrain_names", ("boxes", "gap")))
        self._free_mask = _terrain_type_mask(env, self.free_terrain_names)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        sensor_cfg: SceneEntityCfg,
        threshold: float,
        free_terrain_names: tuple[str, ...] = ("boxes", "gap"),
    ) -> torch.Tensor:
        del free_terrain_names  # 已在 __init__ 缓存为静态掩码
        reward = feet_air_time(env, command_name, sensor_cfg, threshold)
        return reward * (~self._free_mask).float()


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "action_mirror_joints_cache") or env.action_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.action_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.action_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(
                torch.abs(env.action_manager.action[:, joint_pair[0][0]])
                - torch.abs(env.action_manager.action[:, joint_pair[1][0]])
            ),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_sync(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_groups: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Cache joint indices if not already done
    if not hasattr(env, "action_sync_joint_cache") or env.action_sync_joint_cache is None:
        env.action_sync_joint_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_group] for joint_group in joint_groups
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over each joint group
    for joint_group in env.action_sync_joint_cache:
        if len(joint_group) < 2:
            continue  # need at least 2 joints to compare

        # Get absolute actions for all joints in this group
        actions = torch.stack(
            [torch.abs(env.action_manager.action[:, joint[0]]) for joint in joint_group], dim=1
        )  # shape: (num_envs, num_joints_in_group)

        # Calculate mean action for each environment
        mean_actions = torch.mean(actions, dim=1, keepdim=True)

        # Calculate variance from mean for each joint
        variance = torch.mean(torch.square(actions - mean_actions), dim=1)

        # Add to reward (we want to minimize this variance)
        reward += variance.squeeze()
    reward *= 1 / len(joint_groups) if len(joint_groups) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact(
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact_without_cmd(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.sum(contact, dim=-1).float()
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_edge(
    env: ManagerBasedRLEnv,
    edge_sensor_names: tuple[str, ...],
    contact_sensor_cfg: SceneEntityCfg,
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize a contacting foot when its local ray grid straddles solid ground and a void.

    A gap ray has an infinite hit position in Isaac Lab.  Using one small ray grid per foot makes
    this term independent of the actor's base-mounted terrain scan and avoids penalizing ordinary
    stair height discontinuities, where every ray still hits a surface.
    """
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids, :]
    contacts = torch.linalg.norm(contact_forces, dim=-1) > contact_force_threshold
    if contacts.shape[1] != len(edge_sensor_names):
        raise ValueError(
            "feet_edge requires one edge ray caster per configured contact body: "
            f"got {len(edge_sensor_names)} scanners and {contacts.shape[1]} bodies"
        )

    edge_flags = []
    for sensor_name in edge_sensor_names:
        edge_sensor: RayCaster = env.scene.sensors[sensor_name]
        valid_hits = torch.isfinite(edge_sensor.data.ray_hits_w[..., 2])
        edge_flags.append(valid_hits.any(dim=1) & ~valid_hits.all(dim=1))
    feet_on_edge = torch.stack(edge_flags, dim=1)
    return torch.sum((feet_on_edge & contacts).float(), dim=1)


def feet_distance_y_exp(
    env: ManagerBasedRLEnv, stance_width: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)
    n_feet = len(asset_cfg.body_ids)
    footsteps_in_body_frame = torch.zeros(env.num_envs, n_feet, 3, device=env.device)
    for i in range(n_feet):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )
    side_sign = torch.tensor(
        [1.0 if i % 2 == 0 else -1.0 for i in range(n_feet)],
        device=env.device,
    )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    desired_ys = stance_width_tensor / 2 * side_sign.unsqueeze(0)
    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / (std**2))
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_xy_exp(
    env: ManagerBasedRLEnv,
    stance_width: float,
    stance_length: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the current footstep positions relative to the root
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)

    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )

    # Desired x and y positions for each foot
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    stance_length_tensor = stance_length * torch.ones([env.num_envs, 1], device=env.device)

    desired_xs = torch.cat(
        [stance_length_tensor / 2, stance_length_tensor / 2, -stance_length_tensor / 2, -stance_length_tensor / 2],
        dim=1,
    )
    desired_ys = torch.cat(
        [stance_width_tensor / 2, -stance_width_tensor / 2, stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )

    # Compute differences in x and y
    stance_diff_x = torch.square(desired_xs - footsteps_in_body_frame[:, :, 0])
    stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])

    # Combine x and y differences and compute the exponential penalty
    stance_diff = stance_diff_x + stance_diff_y
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    # feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # reward = torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


# def smoothness_1(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     return torch.sum(diff, dim=1)


# def smoothness_2(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - 2 * env.action_manager.prev_action + env.action_manager.prev_prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     diff = diff * (env.action_manager.prev_prev_action[:, :] != 0)  # ignore second step
#     return torch.sum(diff, dim=1)


def smoothness(env: HimlocoManagerBasedRLEnv) -> torch.Tensor:
    """Penalize second-order action changes used by HimLoco."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action * 2 + env.pre_pre_action), dim=1)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data.
        # 2026-09-24 修复：原实现是**整批**判定 —— 4096 个环境里只要有任意一个的射线落空
        # （例如它正悬在沟壑上方），**所有环境**都会退回 `adjusted = root_z`（误差恒 0），
        # 于是 −10 的高度惩罚被整场关掉。后果两条：① 奖励依赖其它环境的状态（污染信用分配）；
        # ② 奖励随 num_envs 变化（256 环境与 4096 环境实际不是同一个任务）。
        # 实测：旧 run（停在沟前）75/16717 个回合为 0；真跨沟的新 run 349/400 个回合恰好为 0。
        # 现改为**逐环境**判定：只用该环境自己的有效射线求局部地面高度，全部落空才退回 root_z
        # （与同文件 `him_base_height` 的 masked-nanmean 写法保持一致）。
        ray_hits = sensor.data.ray_hits_w[..., 2]  # (N, R)
        valid = ~torch.isnan(ray_hits) & ~torch.isinf(ray_hits) & (torch.abs(ray_hits) < 1e6)
        valid_count = valid.sum(dim=1).clamp_min(1)
        mean_hits = torch.where(valid, ray_hits, torch.zeros_like(ray_hits)).sum(dim=1) / valid_count
        adjusted_target_height = torch.where(
            valid.any(dim=1),
            target_height + mean_hits,
            asset.data.root_link_pos_w[:, 2],
        )
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def him_base_height(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """HimLoco base-height penalty with invalid ray filtering."""
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ray_hits = sensor.data.ray_hits_w[..., 2]
        valid_mask = ~torch.isnan(ray_hits) & ~torch.isinf(ray_hits) & (torch.abs(ray_hits) < 1e6)
        ray_hits_masked = torch.where(valid_mask, ray_hits, torch.tensor(float("nan"), device=ray_hits.device))
        adjusted_heights = torch.nanmean(ray_hits_masked, dim=1)
        all_invalid_mask = torch.isnan(adjusted_heights)
        if all_invalid_mask.any():
            adjusted_heights[all_invalid_mask] = asset.data.root_link_pos_w[all_invalid_mask, 2] - target_height
        adjusted_target_height = target_height + adjusted_heights
    else:
        adjusted_target_height = target_height
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def him_feet_height_body(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    command_name: str | None = None,
) -> torch.Tensor:
    """HimLoco body-frame foot-height penalty without tanh velocity scaling."""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[:, :].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_lateral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(env.num_envs, -1)
    reward = torch.sum(foot_z_target_error * foot_lateral_vel, dim=1)
    if command_name is not None:
        reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    return reward


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    reward = torch.sum(is_contact, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

