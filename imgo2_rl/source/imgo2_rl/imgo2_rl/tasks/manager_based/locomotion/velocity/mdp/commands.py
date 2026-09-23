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

from .utils import is_env_assigned_to_terrain

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold.

    Selected terrain types can be restricted to forward-only commands.  This keeps
    obstacle-crossing samples intentional while preserving omnidirectional commands
    on ordinary rough terrain.
    """

    cfg: mdp.UniformThresholdVelocityCommandCfg  # type: ignore
    """The configuration of the command generator."""

    def __init__(self, cfg: mdp.UniformThresholdVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        super().__init__(cfg, env)

    def _forward_only_terrain_mask(self) -> torch.Tensor:
        """Return environments assigned to terrains that use the hard-terrain command ranges."""
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for terrain_name in self.cfg.forward_only_terrain_names:
            mask |= is_env_assigned_to_terrain(self._env, terrain_name)
        return mask

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample velocity commands with threshold."""
        super()._resample_command(env_ids)
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        # set small commands to zero
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        hard_env_ids = env_ids[self._forward_only_terrain_mask()[env_ids]]
        if len(hard_env_ids) > 0:
            forward_speed = torch.empty(len(hard_env_ids), device=self.device)
            self.vel_command_b[hard_env_ids, 0] = forward_speed.uniform_(*self.cfg.forward_speed_range)
            self.vel_command_b[hard_env_ids, 1] = 0.0
            if self.cfg.heading_command:
                self.heading_target[hard_env_ids] = self.cfg.forward_heading_target
                self.is_heading_env[hard_env_ids] = True
            else:
                self.vel_command_b[hard_env_ids, 2] = 0.0

    def _update_command(self):
        """Update commands and apply terrain-aware restrictions in real-time.

        Hard-terrain environments keep a fixed world-frame heading while Isaac Lab's
        built-in heading controller generates the corrective yaw-rate command.
        """
        hard_env_ids = torch.where(self._forward_only_terrain_mask())[0]
        if self.cfg.heading_command and len(hard_env_ids) > 0:
            self.heading_target[hard_env_ids] = self.cfg.forward_heading_target
            self.is_heading_env[hard_env_ids] = True

        super()._update_command()

        if len(hard_env_ids) > 0:
            self.vel_command_b[hard_env_ids, 1] = 0.0
            if not self.cfg.heading_command:
                self.vel_command_b[hard_env_ids, 2] = 0.0


@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand
    forward_only_terrain_names: tuple[str, ...] = ("pits",)
    forward_speed_range: tuple[float, float] = (0.3, 0.6)
    forward_heading_target: float = 0.0


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
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs

        high_vel_env_ids = env_ids <= (self.num_envs * self.cfg.rel_high_vel_envs)
        high_vel_env_ids = env_ids[high_vel_env_ids.nonzero(as_tuple=True)]
        r_high = torch.empty(len(high_vel_env_ids), device=self.device)
        self.vel_command_b[high_vel_env_ids, 0] = r_high.uniform_(*self.cfg.ranges.lin_vel_x)

        low_vel_x_min = self.cfg.low_vel_env_lin_x_ranges[0]
        low_vel_x_max = self.cfg.low_vel_env_lin_x_ranges[1]
        in_low_vel_range = (
            (self.vel_command_b[high_vel_env_ids, 0:1] >= low_vel_x_min)
            & (self.vel_command_b[high_vel_env_ids, 0:1] <= low_vel_x_max)
        )
        self.vel_command_b[high_vel_env_ids, 1:2] *= in_low_vel_range
        self.vel_command_b[env_ids, :2] *= (
            torch.norm(self.vel_command_b[env_ids, :2], dim=1) > self.cfg.min_command_norm
        ).unsqueeze(1)

    def _update_command(self):
        if self.cfg.heading_command:
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
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
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- current_commands: stores a snapshot of the current commands (as integers)
        self.current_commands = [self.available_commands[0]] * self.num_envs  # Default to the first command

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
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device
        )
        sampled_commands = torch.tensor(
            [self.available_commands[idx.item()] for idx in sampled_indices], dtype=torch.int32, device=self.device
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

