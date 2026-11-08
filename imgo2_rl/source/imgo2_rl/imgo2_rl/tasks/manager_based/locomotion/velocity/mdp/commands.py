# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import isaaclab.utils.math as math_utils
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

import imgo2_rl.tasks.manager_based.locomotion.velocity.mdp as mdp

from .utils import is_robot_on_terrain

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold.

    This command generator automatically detects "pits" terrain and applies restrictions:
    - For pit terrains: only allow forward movement (no lateral or rotational movement)
    """

    cfg: mdp.UniformThresholdVelocityCommandCfg  # type: ignore
    """The configuration of the command generator."""

    def __init__(
        self, cfg: mdp.UniformThresholdVelocityCommandCfg, env: ManagerBasedEnv
    ):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        super().__init__(cfg, env)
        # Track which robots were on pit terrain in the previous step
        self.was_on_pit = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample velocity commands with threshold."""
        super()._resample_command(env_ids)
        # set small commands to zero
        self.vel_command_b[env_ids, :2] *= (
            torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2
        ).unsqueeze(1)

    def _update_command(self):
        """Update commands and apply terrain-aware restrictions in real-time.

        This function:
        1. Calls parent's update to handle heading and standing envs
        2. Checks which robots are currently on pit terrain
        3. For robots leaving pits: resamples their commands
        4. For robots on pits: restricts to forward-only movement and sets heading to 0
        """
        # First, call parent's update command
        super()._update_command()

        # Check which robots are currently on pit terrain (real-time check every step)
        on_pits = is_robot_on_terrain(self._env, "pits")

        # Find robots that just left pit terrain (need to resample)
        left_pit_mask = self.was_on_pit & ~on_pits
        if left_pit_mask.any():
            left_pit_env_ids = torch.where(left_pit_mask)[0]
            # Resample commands for robots that left pits
            self._resample_command(left_pit_env_ids)

        # For robots currently on pits: restrict to forward-only movement with min/max speed
        if on_pits.any():
            pit_env_ids = torch.where(on_pits)[0]
            # Force forward-only movement with min and max speed limits
            self.vel_command_b[pit_env_ids, 0] = torch.clamp(
                torch.abs(self.vel_command_b[pit_env_ids, 0]), min=0.3, max=0.6
            )
            self.vel_command_b[pit_env_ids, 1] = 0.0  # no lateral movement
            self.vel_command_b[pit_env_ids, 2] = 0.0  # no yaw rotation
            # Set heading to 0 for pit robots
            if self.cfg.heading_command:
                self.heading_target[pit_env_ids] = 0.0

        # Update tracking state
        self.was_on_pit = on_pits


@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand


class UniformLevelVelocityCommand(mdp.UniformVelocityCommand):
    """Velocity command with separate low/high linear-x sampling ranges."""

    cfg: mdp.UniformLevelVelocityCommandCfg  # type: ignore

    def __str__(self) -> str:
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        return msg

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.low_vel_env_lin_x_ranges)
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = (
                r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
            )

        high_vel_env_ids = env_ids <= (self.num_envs * self.cfg.rel_high_vel_envs)
        high_vel_env_ids = env_ids[high_vel_env_ids.nonzero(as_tuple=True)]
        r_high = torch.empty(len(high_vel_env_ids), device=self.device)
        self.vel_command_b[high_vel_env_ids, 0] = r_high.uniform_(
            *self.cfg.ranges.lin_vel_x
        )

        low_vel_x_min = self.cfg.low_vel_env_lin_x_ranges[0]
        low_vel_x_max = self.cfg.low_vel_env_lin_x_ranges[1]
        in_low_vel_range = (
            self.vel_command_b[high_vel_env_ids, 0:1] >= low_vel_x_min
        ) & (self.vel_command_b[high_vel_env_ids, 0:1] <= low_vel_x_max)
        self.vel_command_b[high_vel_env_ids, 1:2] *= in_low_vel_range
        self.vel_command_b[env_ids, :2] *= (
            torch.norm(self.vel_command_b[env_ids, :2], dim=1)
            > self.cfg.min_command_norm
        ).unsqueeze(1)

    def _update_command(self):
        if self.cfg.heading_command:
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            heading_error = math_utils.wrap_to_pi(
                self.heading_target[env_ids] - self.robot.data.heading_w[env_ids]
            )
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )


@configclass
class UniformLevelVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for :class:`UniformLevelVelocityCommand`."""

    class_type: type = UniformLevelVelocityCommand
    curriculums_limit_ranges: tuple[float, float] = MISSING
    low_vel_env_lin_x_ranges: tuple[float, float] = MISSING
    rel_high_vel_envs: float = MISSING
    min_command_norm: float = MISSING


class DiscreteCommandController(CommandTerm):
    """
    Command generator that assigns discrete commands to environments.

    Commands are stored as a list of predefined integers.
    The controller maps these commands by their indices (e.g., index 0 -> 10, index 1 -> 20).
    """

    cfg: DiscreteCommandControllerCfg
    """Configuration for the command controller."""

    def __init__(self, cfg: DiscreteCommandControllerCfg, env: ManagerBasedEnv):
        """
        Initialize the command controller.

        Args:
            cfg: The configuration of the command controller.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Validate that available_commands is non-empty
        if not self.cfg.available_commands:
            raise ValueError("The available_commands list cannot be empty.")

        # Ensure all elements are integers
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements in available_commands must be integers.")

        # Store the available commands
        self.available_commands = self.cfg.available_commands

        # Create buffers to store the command
        # -- command buffer: stores discrete action indices for each environment
        self.command_buffer = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )

        # -- current_commands: stores a snapshot of the current commands (as integers)
        self.current_commands = [
            self.available_commands[0]
        ] * self.num_envs  # Default to the first command

    def __str__(self) -> str:
        """Return a string representation of the command controller."""
        return (
            "DiscreteCommandController:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tAvailable commands: {self.available_commands}\n"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Return the current command buffer. Shape is (num_envs, 1)."""
        return self.command_buffer

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics for the command controller."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands for the given environments."""
        sampled_indices = torch.randint(
            len(self.available_commands),
            (len(env_ids),),
            dtype=torch.int32,
            device=self.device,
        )
        sampled_commands = torch.tensor(
            [self.available_commands[idx.item()] for idx in sampled_indices],
            dtype=torch.int32,
            device=self.device,
        )
        self.command_buffer[env_ids] = sampled_commands

    def _update_command(self):
        """Update and store the current commands."""
        self.current_commands = self.command_buffer.tolist()


@configclass
class DiscreteCommandControllerCfg(CommandTermCfg):
    """Configuration for the discrete command controller."""

    class_type: type = DiscreteCommandController

    available_commands: list[int] = []
    """
    List of available discrete commands, where each element is an integer.
    Example: [10, 20, 30, 40, 50]
    """


# -----------------------------------------------------------------------------
# 可选的 PPO 分组速度采样（2026-11-05）
# 仅追加实现：上面的原有命令类保持不变；只有显式使用新 Cfg 的任务才会启用。
# 每次重采样独立抽组，默认概率为：10% 静止、40% 低速纯方向、50% 宽范围。
# “固定比例”指长期采样概率，不保证每次少量环境重置时人数刚好符合比例。
# -----------------------------------------------------------------------------
class GroupedVelocityCommand(UniformThresholdVelocityCommand):
    """Protect low-speed sampling while the mixed-command ranges expand.

    Sampled groups: 0 = stand, 1 = low-speed single-axis, 2 = wide mixed.
    Low-speed directions: 0/1 = +/-x, 2/3 = +/-y, 4/5 = +/-yaw.
    The inherited pit-terrain override still takes precedence over sampled groups.
    """

    cfg: GroupedVelocityCommandCfg

    def __init__(self, cfg: GroupedVelocityCommandCfg, env: ManagerBasedEnv):
        import math

        # 父类析构函数也可能处理初始化失败的对象。
        self._debug_vis_handle = None

        # 分组逻辑统一负责静止；禁止父类再次抽站立比例或用 heading 改写纯转向。
        if cfg.heading_command:
            raise ValueError("GroupedVelocityCommand requires heading_command=False")
        if cfg.rel_standing_envs != 0.0:
            raise ValueError(
                "Set rel_standing_envs=0.0; use standing_group_probability for grouped sampling"
            )
        probabilities = (
            cfg.standing_group_probability,
            cfg.low_speed_group_probability,
        )
        if any(not math.isfinite(p) or not 0.0 <= p <= 1.0 for p in probabilities):
            raise ValueError(
                "Grouped sampling probabilities must be finite and in [0, 1]"
            )
        if sum(probabilities) > 1.0:
            raise ValueError(
                "Standing and low-speed group probabilities must sum to at most 1"
            )

        low_ranges = (cfg.low_lin_vel_x, cfg.low_lin_vel_y, cfg.low_ang_vel_z)
        for name, bounds in zip(
            ("low_lin_vel_x", "low_lin_vel_y", "low_ang_vel_z"), low_ranges
        ):
            if (
                len(bounds) != 2
                or any(not math.isfinite(v) for v in bounds)
                or not 0.0 < bounds[0] <= bounds[1]
            ):
                raise ValueError(
                    f"{name} must contain finite positive magnitudes (min, max)"
                )
        # 原类的平移死区是范数 <= 0.2；低速纯方向不能落入该区间。
        if cfg.low_lin_vel_x[0] <= 0.2 or cfg.low_lin_vel_y[0] <= 0.2:
            raise ValueError(
                "Low-speed linear magnitudes must be > 0.2 m/s (existing deadzone)"
            )

        super().__init__(cfg, env)
        self._low_min = torch.tensor(
            [bounds[0] for bounds in low_ranges], device=self.device
        )
        self._low_max = torch.tensor(
            [bounds[1] for bounds in low_ranges], device=self.device
        )
        # 诊断缓冲：表示最近抽到的组/方向；并非地形后处理后的实际指令分类。
        # 不自动添加 TensorBoard 指标，避免把一次采样快照误认为时间平均比例。
        self.sampled_group = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.sampled_low_direction = torch.full_like(self.sampled_group, -1)

    def _resample_command(self, env_ids: Sequence[int]):
        # 父类先产生宽范围混合指令；该部分继续使用 cfg.ranges，可随原课程扩速。
        super()._resample_command(env_ids)
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        group_draw = torch.rand(env_ids.numel(), device=self.device)
        stand = group_draw < self.cfg.standing_group_probability
        low = (group_draw >= self.cfg.standing_group_probability) & (
            group_draw
            < self.cfg.standing_group_probability + self.cfg.low_speed_group_probability
        )
        self.sampled_group[env_ids] = 2
        self.sampled_group[env_ids[stand]] = 0
        self.sampled_group[env_ids[low]] = 1
        self.sampled_low_direction[env_ids] = -1

        # 必须同步父类 mask，否则后续 _update_command 会把其他组再次清零。
        self.is_standing_env[env_ids] = stand
        self.is_heading_env[env_ids] = False
        self.vel_command_b[env_ids[stand]] = 0.0

        # 低速组：六方向均匀抽取，只激活一轴；范围独立于 cfg.ranges。
        # 全部使用设备上的张量，不逐环境循环、不在每个物理步重新抽样。
        low_ids = env_ids[low]
        directions = torch.randint(0, 6, (low_ids.numel(),), device=self.device)
        axes = directions // 2
        signs = torch.where(directions % 2 == 0, 1.0, -1.0)
        magnitudes = self._low_min[axes] + torch.rand(
            low_ids.numel(), device=self.device
        ) * (self._low_max[axes] - self._low_min[axes])
        self.vel_command_b[low_ids] = 0.0
        self.vel_command_b[low_ids, axes] = signs * magnitudes
        self.sampled_low_direction[low_ids] = directions

        # 宽范围组不再覆写。父类的 _update_command / pit 限制继续有效。
        # 注意：抽到的组不等于执行后的实际类别，pit 上会被强制为前进。


@configclass
class GroupedVelocityCommandCfg(UniformThresholdVelocityCommandCfg):
    """Optional grouped sampler; replace only the desired task's command cfg."""

    class_type: type = GroupedVelocityCommand
    heading_command: bool = False
    rel_heading_envs: float = 0.0
    # 静止只由下面的 standing_group_probability 管理，不能叠加父类比例。
    rel_standing_envs: float = 0.0
    standing_group_probability: float = 0.10
    low_speed_group_probability: float = 0.40
    # 均为正的“幅值”范围，正负方向由六方向采样决定；其余概率使用 ranges。
    low_lin_vel_x: tuple[float, float] = (0.25, 0.60)
    low_lin_vel_y: tuple[float, float] = (0.25, 0.40)
    low_ang_vel_z: tuple[float, float] = (0.20, 0.60)
