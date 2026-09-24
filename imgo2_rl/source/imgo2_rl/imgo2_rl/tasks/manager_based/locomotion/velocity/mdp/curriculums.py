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


def _cached_terrain_mask(env: ManagerBasedRLEnv, terrain_names: tuple[str, ...]) -> torch.Tensor:
    """(N,) bool：环境所属地形列是否落在名单里；按名单缓存到 env 上（地形列逐环境固定）。"""
    key = "_cmoe_curriculum_mask__" + "__".join(terrain_names)
    mask = getattr(env, key, None)
    if mask is None:
        from .utils import is_env_assigned_to_terrain  # 延迟导入，避免包内循环依赖

        mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for name in terrain_names:
            mask = mask | is_env_assigned_to_terrain(env, name)
        setattr(env, key, mask)
    return mask


def _episode_term_average(
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


# 兼容旧名（原先只有跟踪那一项用它）
_episode_tracking_average = _episode_term_average


def terrain_levels_vel_logged(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tracking_term_name: str = "track_world_vel_xy_exp",
    tracking_move_up: float = 0.80,
    tracking_move_down: float = 0.35,
    relaxed_terrain_names: tuple[str, ...] = ("pyramid_stairs", "pyramid_stairs_inv", "boxes"),
    tracking_move_up_relaxed: float = 0.50,
    gait_metric_terms: tuple[tuple[str, str], ...] = (),
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
    up_threshold = None
    if track_avg is not None:
        # 2026-09-24 晚（实测反馈）：单一阈值 0.80 会让"台阶/独立块"这三类**只能慢慢过**的地形
        # 长期卡在 frozen 带（进度够、跟踪 0.35–0.80）⇒ 等级永远不变（反楼梯卡在 0.12、
        # boxes 卡在 0.10，而 flat/slope 已 4–6 级）。故对这三类放宽到 `tracking_move_up_relaxed`。
        # ⚠️ **0.50 在这三类地形上等价于"取消跟踪门控"**：这个代理（**整回合**平均跟踪核）又高又窄
        # —— 实测 `tracking_mean≈0.78`、`tracking_min≈0.69`（全在 0.69 以上），因为每块 tile 有
        # 70–80% 是平地、障碍只占约 20%（反楼梯 tile：平台 2 m ＋ 台阶 1.8 m ＋ 平地 4.2 m）。
        # 所以 0.50 意味着"这三类退回**只看前进进度**"（走够 `size[0]/2`＝4 m 就晋级），而 0.80 会
        # 把它们永久冻死（跟踪 0.69–0.80 落在 frozen 带）。用户 2026-09-24 决定：**台阶与 boxes 用 0.50，
        # 其余仍 0.80**。依据：旧判据（只看进度）在这三类上曾把 A 跑推到 level 6（反楼梯 6.06／boxes 5.99），
        # 而"能不能过障碍"本来就该由**进度**管；跟踪门控更适合**容易地形**（那里"糊弄过去"才是失败模式）。
        # 若想保留一点牙齿，把 0.50 提到 0.60–0.65（仍低于实测 min 0.69 ⇒ 依旧近乎无门控，但语义清楚）。
        up_threshold = torch.full_like(track_avg, float(tracking_move_up))
        if len(relaxed_terrain_names) > 0:
            relaxed = _cached_terrain_mask(env, tuple(relaxed_terrain_names))[env_ids]
            up_threshold[relaxed] = float(tracking_move_up_relaxed)
        move_up = move_up & (track_avg > up_threshold)
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
        out["tracking_pass_frac"] = (track_avg > up_threshold).float().mean()
        out["tracking_fail_frac"] = (track_avg < tracking_move_down).float().mean()
        out["tracking_up_threshold_mean"] = torch.mean(up_threshold)
    # 按地形列分组的等级均值（地形列逐环境固定，故可稳定对比「障碍列 vs 粗糙列」）
    from .utils import is_env_assigned_to_terrain  # 延迟导入，避免包内循环依赖

    # 步态度量项（每个只用来**测量**、几乎不产生奖励：权重被设成 1e-6 ⇒ ≤1e-6/s，
    # 相对整回合 ~4/s 可忽略）。这里按列取"本回合平均"⇒ 逐列步态画像。
    # 三个成对方式给出一个**分类器**：diagonal＝trot、left-right＝bound、same-side＝pace，
    # 谁高就是谁；三者都接近下界（`0.893^6 ≈ 0.508`）则既不是 trot/bound/pace（pronk／乱走）。
    metric_avg: dict[str, torch.Tensor] = {}
    for label, term_name in gait_metric_terms:
        value = _episode_term_average(env, env_ids, term_name)
        if value is not None:
            metric_avg[label] = value
            out[f"gait_{label}_mean"] = torch.mean(value)

    for name in terrain.cfg.terrain_generator.sub_terrains.keys():
        mask = is_env_assigned_to_terrain(env, name)
        if mask.any():
            # 等级是**全体环境的状态**（每个环境固定占一列）⇒ 只要该列有环境就一定有值。
            out[f"level_{name}"] = torch.mean(levels[mask])
            # ⚠️ `env_ids` 只是**这一步刚结束的那些环境**，不是整轮：4096 环境 × 24 步 ÷ 平均回合长度
            # ≈ 150 个/轮，再摊到 24 次 `_reset_idx` ⇒ **每次只剩约 6 个**，20 列里常见某列一个都没有。
            # 空子集取 `mean` 得到 **NaN**，而 runner 会把该轮每次 `_reset_idx` 的这个标量一起求平均
            # ⇒ 只要有一次 NaN，**整轮的该 tag 就是 NaN**。2026-09-24 run F（21:14）实测正是如此：
            # 8 列里 7 列的 `tracking_*`／`gait_*_*` 全 NaN，只有样本多的 `gap` 偶尔有值
            # （`level_*` 反而正常，因为它按全体环境统计）。
            # 修法：**该列本步没有"本回合"样本就不写这个键**（键缺失会被 `CMoEOnPolicyRunner.log`
            # 跳过——那里已改成按键集合的并集遍历）⇒ 该轮的值＝该轮里"确实有该列样本的那几次"的均值，
            # 既不是 NaN，也不是拿别的列/历史值顶替。
            inner = mask[env_ids]
            has_sample = bool(inner.any())
            if track_avg is not None and has_sample:
                # 逐列跟踪均值：用来判断"某一列卡住"到底是跟踪不达标还是真的过不去
                out[f"tracking_{name}"] = torch.mean(track_avg[inner])
            if has_sample:
                for label, value in metric_avg.items():
                    out[f"gait_{label}_{name}"] = torch.mean(value[inner])
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

