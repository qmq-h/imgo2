from isaaclab.utils import configclass

from imgo2_rl.assets.imgo2 import AMP_MOTION_FILES
from rl_lab.config import AMPActorCriticCfg, AMPAlgorithmCfg, AMPOnPolicyRunnerCfg


@configclass
class AMPRunnerCfg(AMPOnPolicyRunnerCfg):
    num_steps_per_env = 24
    # 40000 轮（用户 2026-09-17 决定）。依据：10000 轮那次到最后一轮仍在单调改善
    # （线速度误差 1.31→0.507、触地终止 25%→1.5%、任务奖励/步 1.88→4.51），未收敛；
    # 斜率在变小，故加长训练而不是继续调配比。约 1.4 s/轮 → 40000 轮约 15–16 小时。
    max_iterations = 40000
    # 每 500 轮存一次（runner 在 it % save_interval == 0 时写 model_<it>.pt）：
    # 40000 轮、间隔 500 = 81 个 12 MB 的 checkpoint（约 1 GB）；若嫌多可改 1000。
    save_interval = 500
    experiment_name = "base_move_amp"
    include_history_steps = None
    amp_motion_files = AMP_MOTION_FILES
    amp_num_preload_transitions = 2000000
    amp_reward_coef = 2.0
    amp_discr_hidden_dims = [1024, 512]
    # Task terms use per-step weights in Imgo2AmpMoveEnvCfg.
    # Initial tuning baseline; validate height and style with fresh training.
    amp_task_reward_lerp = 0.3
    min_normalized_std = [0.05, 0.02, 0.05] * 4

    policy = AMPActorCriticCfg(
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        fixed_std=False,
    )
    algorithm = AMPAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        amp_replay_buffer_size=1000000,
        # 不对动作噪声 std 施加下限 clamp（AMP-06）：标准 PPO 没有这个约束，策略变化由
        # KL 自适应学习率控制（上限 1e-2 / 下限 1e-5）。该 clamp 一向只有下限、没有上限，
        # 既挡不住 std 暴涨（Run2 涨到 12），又会与熵项（+0.01）对抗。
        # 关闭后 `min_normalized_std` 不再生效。
        clamp_noise_std=False,
    )
