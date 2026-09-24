"""Five-expert CMoE configuration for the Imgo2 PPO rough task."""

from isaaclab.utils import configclass

from rl_lab.config import CMoEActorCriticCfg, CMoEPPOAlgorithmCfg, CMoEOnPolicyRunnerCfg


@configclass
class Imgo2CMoERoughRunnerCfg(CMoEOnPolicyRunnerCfg):
    num_steps_per_env = 24
    # 2026-09-24（用户要求"训练拉长一点"）：2000 → **60000**。2000 是基类遗留的占位值，
    # 实际每次都靠 CLI `--max_iterations` 覆盖（上一轮用 40000）。定到 60000 让**配方自己**记住
    # 训练长度，命令行不必再带这个 flag。按上一轮 4096 环境实测 3.60–3.83 s/轮 ⇒ 60000 轮
    # ≈ **60–64 小时**（2.5 天）。仍可用 `--max_iterations=N` 临时改；
    # `save_interval=500` ⇒ 意外关机最多丢 ~30 分钟，`--resume` 可接着跑
    # （`--resume` 时 `--max_iterations` 表示**追加**轮数，`tot_iter = 当前 + 追加`）。
    max_iterations = 60000
    save_interval = 500
    experiment_name = "base_move_cmoe_rough"
    empirical_normalization = False
    history_steps = 10

    policy = CMoEActorCriticCfg(
        num_experts=5,
        state_latent_dim=16,
        terrain_latent_dim=16,
        explicit_dim=3,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        gate_hidden_dim=128,
        num_prototypes=32,
        projection_dim=16,
        temperature=0.2,
        activation="elu",
        init_noise_std=1.0,
        # critic = [base_lin_vel(3), policy proprioception(45), height_scan(**77**)]
        # 2026-09-24：地形扫描由 187 改为原版 77（`scan.size=(1.0,0.6)`）⇒ critic 235 → **125**，
        # actor 637 → **527**，expert/gate 267 → **157**。这里只标偏移量（3 / 0），维数全由配置推导。
        critic_explicit_start=0,
        critic_proprio_start=3,
    )
    algorithm = CMoEPPOAlgorithmCfg(
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
        contrastive_loss_coef=1.0,
    )
