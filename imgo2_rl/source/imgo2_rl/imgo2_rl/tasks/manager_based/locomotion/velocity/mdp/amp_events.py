from __future__ import annotations

from typing import Literal, Sequence, TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from rl_lab.datasets.motion_loader import AMPLoader

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_amp_reference_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    motion_files: Sequence[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_state_initialization_prob: float = 1.0,
    joint_mapping: Sequence[int] | None = None,
    root_height_offset: float = 0.0,
    velocity_frame: Literal["base", "world"] = "base",
    cache_key: str = "amp_reference_loader",
):
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    if len(env_ids) == 0:
        return

    asset: Articulation = env.scene[asset_cfg.name]
    env_ids = env_ids.to(device=asset.device, dtype=torch.long)

    if reference_state_initialization_prob <= 0.0:
        return
    if reference_state_initialization_prob < 1.0:
        mask = torch.rand(len(env_ids), device=asset.device) < reference_state_initialization_prob
        env_ids = env_ids[mask]
        if len(env_ids) == 0:
            return

    amp_loader = _get_amp_loader(env, motion_files, cache_key)
    frames = amp_loader.get_full_frame_batch(len(env_ids))

    joint_pos = AMPLoader.get_joint_pose_batch(frames).to(asset.device)
    joint_vel = AMPLoader.get_joint_vel_batch(frames).to(asset.device)
    if joint_mapping is not None:
        joint_order_tensor = _invert_mapping(joint_mapping, device=asset.device)
        joint_pos = joint_pos[:, joint_order_tensor]
        joint_vel = joint_vel[:, joint_order_tensor]

    root_pos = AMPLoader.get_root_pos_batch(frames).to(asset.device)
    # root_quat = AMPLoader.get_root_rot_batch(frames).to(asset.device)
    root_quat_xyzw = AMPLoader.get_root_rot_batch(frames).to(asset.device)
    root_quat = math_utils.convert_quat(root_quat_xyzw, to="wxyz")
    root_lin_vel = AMPLoader.get_linear_vel_batch(frames).to(asset.device)
    root_ang_vel = AMPLoader.get_angular_vel_batch(frames).to(asset.device)

    root_pos[:, :2] += env.scene.env_origins[env_ids, :2]
    root_pos[:, 2] += root_height_offset

    if velocity_frame == "base":
        root_lin_vel = math_utils.quat_apply(root_quat, root_lin_vel)
        root_ang_vel = math_utils.quat_apply(root_quat, root_ang_vel)
    elif velocity_frame != "world":
        raise ValueError(f"Unsupported velocity_frame: {velocity_frame!r}. Expected 'base' or 'world'.")

    asset.write_root_pose_to_sim(torch.cat((root_pos, root_quat), dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.cat((root_lin_vel, root_ang_vel), dim=-1), env_ids=env_ids)
    asset.write_joint_state_to_sim(
        joint_pos,
        joint_vel,
        joint_ids=asset_cfg.joint_ids,
        env_ids=env_ids,
    )


def _invert_mapping(mapping: Sequence[int], device: torch.device) -> torch.Tensor:
    mapping_tensor = torch.as_tensor(mapping, device=device, dtype=torch.long)
    if mapping_tensor.ndim != 1:
        raise ValueError(f"joint_mapping must be a 1-D sequence, got shape {tuple(mapping_tensor.shape)}.")
    if torch.unique(mapping_tensor).numel() != mapping_tensor.numel():
        raise ValueError(f"joint_mapping must be a permutation, got {mapping}.")
    inverse_mapping = torch.empty_like(mapping_tensor)
    inverse_mapping[mapping_tensor] = torch.arange(mapping_tensor.numel(), device=device)
    return inverse_mapping


def _get_amp_loader(env: ManagerBasedEnv, motion_files: Sequence[str], cache_key: str) -> AMPLoader:
    """Create or reuse the AMP loader stored on the environment."""
    cached_loader = getattr(env, cache_key, None)
    cached_files = getattr(env, f"_{cache_key}_motion_files", None)
    motion_files_tuple = tuple(motion_files)
    if cached_loader is not None and cached_files == motion_files_tuple:
        return cached_loader

    amp_loader = AMPLoader(
        motion_files=list(motion_files_tuple),
        device=env.device,
        time_between_frames=env.step_dt,
    )
    setattr(env, cache_key, amp_loader)
    setattr(env, f"_{cache_key}_motion_files", motion_files_tuple)
    return amp_loader
