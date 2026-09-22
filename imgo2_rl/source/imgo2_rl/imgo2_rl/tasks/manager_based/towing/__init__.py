"""Cart and towing tasks, alongside the locomotion task family.

P1/P2 validate the cart independently of a learning policy. Environment
configurations belong here, physical terms in ``mdp``, and upper-policy
training configurations in ``agents``.

上层拖曳任务的 agent 配置发布在 ``rl_lab_cfg_entry_point`` 下（它由仓库自有
``TowingOnPolicyRunner`` 训练，不走外部 RSL-RL runner）。Isaac Lab 2.2.1 的
``hydra_task_config`` 通过「注册键去掉 ``_cfg_entry_point`` 后缀」推导 agent 名，
因此 ``scripts/rl_lab/towing/train.py`` 必须传 ``--agent=rl_lab``（这也是它的默认值）。

注册状态：2026-09-22 由用户决定加入注册块，以便在训练机执行拖曳上层 RL 训练。
解锁前被移除的守门断言 ``test_environment_is_not_registered_before_physics_adapter_is_complete``
要求「运行验收前不注册」，该验收项**尚未完成**，因此本注册属"已接线，待验证"：
运行级验收清单见 ``docs/towing_training_prep_2026-09-22.md``。
"""

import gymnasium as gym

from . import agents


gym.register(
    id="Imgo2-towing-upper-rl-lab",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.upper_env_cfg:UpperTowingEnvCfg",
        "rl_lab_cfg_entry_point": f"{agents.__name__}.upper_ppo_cfg:UpperTowingPPORunnerCfg",
    },
)
