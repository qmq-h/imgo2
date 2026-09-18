# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Imgo2-basemove-rough-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:Imgo2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Imgo2RoughPPORunnerCfg",
    },
)

gym.register(
    id="Imgo2-basemove-flat-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:Imgo2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Imgo2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Imgo2-basemove-rough-himloco",
    entry_point="rl_lab.envs:HimlocoManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.himloco_env_cfg:Imgo2HimlocoRoughEnvCfg",
        "himloco_rsl_rl_cfg": f"{agents.__name__}.himloco_rsl_rl_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Imgo2-basemove-rough-himloco-play",
    entry_point="rl_lab.envs:HimlocoManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.himloco_env_cfg:Imgo2HimlocoRoughPlayEnvCfg",
        "himloco_rsl_rl_cfg": f"{agents.__name__}.himloco_rsl_rl_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Imgo2-basemove-flat-amp",
    entry_point="rl_lab.envs:AmpManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_env_cfg:Imgo2AmpMoveEnvCfg",
        "amp_rsl_rl_cfg": f"{agents.__name__}.amp_rsl_rl_cfg:AMPRunnerCfg",
    },
)

gym.register(
    id="Imgo2-basemove-flat-amp-play",
    entry_point="rl_lab.envs:AmpManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_env_cfg:Imgo2AmpMovePlayEnvCfg",
        "amp_rsl_rl_cfg": f"{agents.__name__}.amp_rsl_rl_cfg:AMPRunnerCfg",
    },
)

# amp_go2 配方（2026-09-18，用户要求"尽可能参考 amp_go2"）：任务侧保留 legged_gym 整套步态奖励
# （feet_air_time / collision / action_rate / dof_acc / torques），AMP 只做轻量风格先验
# （coef 0.2 / lerp 0.8）。与上面的 Imgo2-basemove-flat-amp 并存，便于对照与回滚。
# 依据：docs/amp_gait_adjust_plan_2026-09-18.md §7/§8。
gym.register(
    id="Imgo2-basemove-flat-amp-go2",
    entry_point="rl_lab.envs:AmpManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_env_cfg:Imgo2AmpGo2StyleEnvCfg",
        "amp_rsl_rl_cfg": f"{agents.__name__}.amp_rsl_rl_cfg:AMPGo2RunnerCfg",
    },
)

gym.register(
    id="Imgo2-basemove-flat-amp-go2-play",
    entry_point="rl_lab.envs:AmpManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_env_cfg:Imgo2AmpGo2StylePlayEnvCfg",
        "amp_rsl_rl_cfg": f"{agents.__name__}.amp_rsl_rl_cfg:AMPGo2RunnerCfg",
    },
)

# rl_amp（fan-ziqi）配方的平地版：只留线/角速度跟踪奖励（每步 1.0 / 0.3333），
# AMP 侧 coef 2.0 / lerp 0.3（风格约占 2/3）。与下面的 rough 版是**同一配方**，成对使用。
gym.register(
    id="Imgo2-basemove-flat-amp-rlamp",
    entry_point="rl_lab.envs:AmpManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_env_cfg:Imgo2AmpRLAmpEnvCfg",
        "amp_rsl_rl_cfg": f"{agents.__name__}.amp_rsl_rl_cfg:AMPRLAmpRunnerCfg",
    },
)

gym.register(
    id="Imgo2-basemove-flat-amp-rlamp-play",
    entry_point="rl_lab.envs:AmpManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_env_cfg:Imgo2AmpRLAmpPlayEnvCfg",
        "amp_rsl_rl_cfg": f"{agents.__name__}.amp_rsl_rl_cfg:AMPRLAmpRunnerCfg",
    },
)

# 粗糙地形版 AMP（2026-09-18）：**配方忠于 rl_amp**（只有线/角速度跟踪，其余全 0），
# 只加地形 + 地形相对根高（AMP 观测）+ 关掉无地形补偿的参考状态初始化；actor 保持 45 维盲走。
# 依据与"刻意不做"的清单见 amp_env_cfg.Imgo2AmpRoughEnvCfg 的 docstring。
gym.register(
    id="Imgo2-basemove-rough-amp-rlamp",
    entry_point="rl_lab.envs:AmpManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_env_cfg:Imgo2AmpRoughEnvCfg",
        "amp_rsl_rl_cfg": f"{agents.__name__}.amp_rsl_rl_cfg:AMPRLAmpRunnerCfg",
    },
)

gym.register(
    id="Imgo2-basemove-rough-amp-rlamp-play",
    entry_point="rl_lab.envs:AmpManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_env_cfg:Imgo2AmpRoughPlayEnvCfg",
        "amp_rsl_rl_cfg": f"{agents.__name__}.amp_rsl_rl_cfg:AMPRLAmpRunnerCfg",
    },
)
