# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _episode_tracking_average(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str
) -> torch.Tensor | None:
    """本回合的**平均速度跟踪核**（0–1）。取不到就返回 None（调用方退回纯距离判据）。

    数据来源是奖励管理器为每个分项累计的"回合和"：`RewardManager.compute` 每步做
    `_episode_sums[name] += term·weight·dt`，于是

        本回合 term 的时间平均 = _episode_sums[name] / weight / (回合步数 · step_dt)

    **时机是对的**：`ManagerBasedRLEnv._reset_idx` 里 `curriculum_manager.compute()` 在最前面
    （`manager_based_rl_env.py:358`），而 `reward_manager.reset()`（读走并清零 `_episode_sums`）在后面
    （同函数 ~377），`episode_length_buf` 更是最后才清零（~396）⇒ 此刻拿到的正是**刚结束那一回合**的值。
    """
    manager = getattr(env, "reward_manager", None)
    sums = getattr(manager, "_episode_sums", None) if manager is not None else None
    if not sums or term_name not in sums:
        return None
    try:  # 权重从 term cfg 取，避免和配置里写死的数字脱钩
        weight = float(manager.get_term_cfg(term_name).weight)
    except Exception:
        return None
    if weight == 0.0:
        return None
    episode_steps = env.episode_length_buf[env_ids].clamp_min(1).float()
    return sums[term_name][env_ids] / weight / (episode_steps * env.step_dt)


def terrain_levels_vel_logged(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tracking_term_name: str = "track_world_vel_xy_exp",
    tracking_move_up: float = 0.80,
    tracking_move_down: float = 0.35,
) -> dict:
    """带诊断、且**把速度跟踪效果计入晋级判据**的地形课程（2026-09-24 用户要求）。

    与上游 `terrain_levels_vel` 的两处**有意偏离**：

    1. **晋级看"沿 +x 的前向进度"**（`root_x − origin_x`），不再用含横向分量的欧氏距离
       —— 上游的欧氏距离把"横移绕开障碍"也算作通过（docs §29.15/§29.16 的实测漏洞）。
    2. **晋级还要速度跟踪达标**：本回合的 `track_world_vel_xy_exp` 时间平均必须 > `tracking_move_up`；
       低于 `tracking_move_down` 则**降级**。理由（用户）："地形等级提升还是需要考虑速度跟踪效果"
       —— 只看走了多远，会奖励"慢慢蹭过去/绕过去"，而不管有没有按指令跟速。

    判据形式：

    * 晋级：`progress > size[0]/2` **且** `track_avg > tracking_move_up`
    * 降级：`progress < ‖cmd_xy‖·T·0.5` **或** `track_avg < tracking_move_down`（晋级优先）
    * 取不到跟踪分项（名字变了/被移除）时自动退回纯距离判据，并在日志里给出 `tracking_is_used=0`。

    等级增减与"到顶随机重开"仍由 `terrain.update_env_origins` 负责（`terrain_importer.py:314-321`）。

    `CurriculumManager.reset` 会把 dict 的每一项展开成 `Curriculum/<term>/<key>`，
    再经 CMoE runner 前缀成 `Episode/Curriculum/terrain_levels/*`。
    """
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    # ① 前向进度（沿 +x；不含横向 ⇒ 绕开不再算通过）
    progress = asset.data.root_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    command_xy = torch.norm(command[env_ids, :2], dim=1)

    # ② 本回合的平均速度跟踪核
    track_avg = _episode_tracking_average(env, env_ids, tracking_term_name)

    move_up = progress > terrain.cfg.terrain_generator.size[0] / 2
    move_down = progress < command_xy * env.max_episode_length_s * 0.5
    if track_avg is not None:
        move_up = move_up & (track_avg > tracking_move_up)
        move_down = move_down | (track_avg < tracking_move_down)
    move_down *= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)

    levels = terrain.terrain_levels.float()
    out = {
        "level_mean": torch.mean(levels),
        "level_min": torch.min(levels),
        "level_max": torch.max(levels),
        "move_up_frac": move_up.float().mean(),
        "move_down_frac": move_down.float().mean(),
        "frozen_frac": (~move_up & ~move_down).float().mean(),
        "progress_mean": torch.mean(progress),
        "distance_mean": torch.mean(distance),
        "command_norm_mean": torch.mean(command_xy),
        "tracking_is_used": torch.tensor(1.0 if track_avg is not None else 0.0),
    }
    if track_avg is not None:
        out["tracking_mean"] = torch.mean(track_avg)
        out["tracking_min"] = torch.min(track_avg)
        out["tracking_pass_frac"] = (track_avg > tracking_move_up).float().mean()
        out["tracking_fail_frac"] = (track_avg < tracking_move_down).float().mean()
    # 按地形列分组的等级均值（地形列逐环境固定，故可稳定对比「障碍列 vs 粗糙列」）
    from .utils import is_env_assigned_to_terrain  # 延迟导入，避免包内循环依赖

    for name in terrain.cfg.terrain_generator.sub_terrains.keys():
        mask = is_env_assigned_to_terrain(env, name)
        if mask.any():
            out[f"level_{name}"] = torch.mean(levels[mask])
    return out


def command_levels_lin_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_lin_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            # Update ranges
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)


def command_levels_ang_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_ang_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original angular velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device)
        env._initial_ang_vel_z = env._original_ang_vel_z * range_multiplier[0]
        env._final_ang_vel_z = env._original_ang_vel_z * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.ang_vel_z = env._initial_ang_vel_z.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_ang_vel_z = torch.clamp(new_ang_vel_z, min=env._final_ang_vel_z[0], max=env._final_ang_vel_z[1])

            # Update ranges
            base_velocity_ranges.ang_vel_z = new_ang_vel_z.tolist()

    return torch.tensor(base_velocity_ranges.ang_vel_z[1], device=env.device)


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy_exp",
) -> torch.Tensor:
    """Curriculum used by the level-sampled HimLoco velocity command."""
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.curriculums_limit_ranges

    low_vel_env_ids = env_ids >= (env.num_envs * command_term.cfg.rel_high_vel_envs)
    high_vel_env_ids = env_ids < (env.num_envs * command_term.cfg.rel_high_vel_envs)
    low_vel_env_ids = env_ids[low_vel_env_ids.nonzero(as_tuple=True)]
    high_vel_env_ids = env_ids[high_vel_env_ids.nonzero(as_tuple=True)]

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)

    if env.common_step_counter % env.max_episode_length == 0:
        reward_low = (
            torch.mean(env.reward_manager._episode_sums[reward_term_name][low_vel_env_ids])
            / env.max_episode_length_s
            if len(low_vel_env_ids) > 0
            else 0.0
        )
        reward_high = (
            torch.mean(env.reward_manager._episode_sums[reward_term_name][high_vel_env_ids])
            / env.max_episode_length_s
            if len(high_vel_env_ids) > 0
            else 0.0
        )

        if reward_low > reward_term.weight * 0.8 and reward_high > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.2, 0.2], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges[0],
                limit_ranges[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)

