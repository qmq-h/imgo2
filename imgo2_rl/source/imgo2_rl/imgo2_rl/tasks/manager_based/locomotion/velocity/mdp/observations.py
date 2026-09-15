# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.utils import math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor


def base_external_force(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """External force applied on the selected body."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset._external_force_b[:, asset_cfg.body_ids, :].squeeze(1).clone()


def height_scan_clip(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    clip: tuple[float, float] = (-1.0, 1.0),
    offset: float = 0.5,
) -> torch.Tensor:
    """Height scan clipped after subtracting the given offset."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    height = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    return torch.clip(height, clip[0], clip[1])


def _apply_flat_mapping(data: torch.Tensor, mapping: list[int] | tuple[int, ...] | None) -> torch.Tensor:
    if mapping is None:
        return data
    mapping_tensor = torch.as_tensor(mapping, device=data.device, dtype=torch.long)
    return data[:, mapping_tensor]


def amp_joint_pos(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    mapping: list[int] | tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Absolute joint positions for AMP discriminator observations."""
    asset: Articulation = env.scene[asset_cfg.name]
    return _apply_flat_mapping(asset.data.joint_pos[:, asset_cfg.joint_ids], mapping)


def amp_joint_vel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    mapping: list[int] | tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Joint velocities for AMP discriminator observations."""
    asset: Articulation = env.scene[asset_cfg.name]
    return _apply_flat_mapping(asset.data.joint_vel[:, asset_cfg.joint_ids], mapping)


def amp_foot_pos_base(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    mapping: list[int] | tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Foot positions in the base frame for AMP discriminator observations."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    foot_pos_rel_w = foot_pos_w - asset.data.root_pos_w[:, :3].unsqueeze(1)
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, foot_pos_rel_w.shape[1], -1)
    foot_pos_b = math_utils.quat_apply_inverse(
        root_quat_w,
        foot_pos_rel_w,
    )
    foot_pos_b = foot_pos_b.reshape(env.num_envs, -1)
    return _apply_flat_mapping(foot_pos_b, mapping)


def amp_root_z(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root height for AMP discriminator observations."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2:3]
