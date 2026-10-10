"""rl_lab configuration for the upper towing residual policy.

任务已注册为 ``Imgo2-towing-upper-rl-lab``（见同目录 ``__init__.py``）。

`decoder.frame_dim` 必须等于环境的 policy 帧维数（``upper_logic.UpperObservationSpec``
当前为 **58**，契约 v3）：runner 启动时会断言 ``decoder.frame_dim == env.num_obs``，
写错会直接报错。
"""

from isaaclab.utils import configclass
from rl_lab.config import (
    PPOAlgorithmCfg,
    TowingActorCriticCfg,
    TowingDecoderCfg,
    TowingOnPolicyRunnerCfg,
)


@configclass
class UpperTowingPPORunnerCfg(TowingOnPolicyRunnerCfg):
    num_steps_per_env = 48
    max_iterations = 3000
    save_interval = 100
    experiment_name = "towing_upper"
    # 上层动作现在是 **13 维**（2026-10-10 双头）：1 维 vx 偏移 + 12 维关节位置残差。
    # 本值只对**动作整体**做归一化裁剪，两头共用同一上限：
    #   u_cmd   = clip(u_0, ±1) ⇒ offset = u_0 × 0.5 m/s，再限在 [−0.2, +0.6]；
    #   u_joint = clip(u_1:, ±1) ⇒ deltapos = u_joint × action_scale
    #            （髋 ±0.125 rad、大腿/小腿 ±0.25 rad）。
    # 冻结策略自身的动作裁剪是 ±3.0（`LowLevelPolicyCfg.clip_actions_upper`），即残差
    # 最多只用到底层权限的 1/3。要放大上层权限就调大本值（同时复核 action_magnitude 权重）。
    clip_actions = 1.0
    policy = TowingActorCriticCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        rnn_type="gru",
        rnn_hidden_size=256,
        rnn_num_layers=1,
    )
    # critic 特权组 = policy_frame(58) + robot_velocity(2) + cart_velocity(2) + rope_state(4)
    #              + towing_force(3) + cart_privileged_parameters(4) = **73**
    # （`upper_policy_runtime.DEFAULT_CRITIC_OBS_DIM` 与它有离线交叉校验）。
    decoder = TowingDecoderCfg(
        frame_dim=58,
        feature_dim=128,
        hidden_dim=128,
        num_layers=1,
        latent_dim=16,
        kld_weight=0.005,
        force_scale=10.0,
        learning_rate=1.0e-3,
        max_grad_norm=1.0,
        velocity_coef=5.0,
        force_coef=10.0,
        mass_coef=1.0,
    )
    critic_empirical_normalization = True
    algorithm = PPOAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
