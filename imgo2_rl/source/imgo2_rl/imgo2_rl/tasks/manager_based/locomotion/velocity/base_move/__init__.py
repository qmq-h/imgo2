import gymnasium as gym

from . import agents


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
    id="Imgo2-basemove-flat-amp-height",
    entry_point="rl_lab.envs:AmpManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_env_cfg:Imgo2AmpMoveEnvCfg",
        "amp_rsl_rl_cfg": f"{agents.__name__}.amp_rsl_rl_cfg:AMPHeightRunnerCfg",
    },
)

gym.register(
    id="Imgo2-basemove-flat-amp-height-play",
    entry_point="rl_lab.envs:AmpManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_env_cfg:Imgo2AmpMovePlayEnvCfg",
        "amp_rsl_rl_cfg": f"{agents.__name__}.amp_rsl_rl_cfg:AMPHeightRunnerCfg",
    },
)

gym.register(
    id="Imgo2-basemove-flat-amp-fanziqi",
    entry_point="rl_lab.envs:AmpManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_env_cfg:Imgo2AmpRLAmpEnvCfg",
        "amp_rsl_rl_cfg": f"{agents.__name__}.amp_rsl_rl_cfg:FanziqiAMPRunnerCfg",
    },
)

gym.register(
    id="Imgo2-basemove-flat-amp-fanziqi-play",
    entry_point="rl_lab.envs:AmpManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_env_cfg:Imgo2AmpRLAmpPlayEnvCfg",
        "amp_rsl_rl_cfg": f"{agents.__name__}.amp_rsl_rl_cfg:FanziqiAMPRunnerCfg",
    },
)
