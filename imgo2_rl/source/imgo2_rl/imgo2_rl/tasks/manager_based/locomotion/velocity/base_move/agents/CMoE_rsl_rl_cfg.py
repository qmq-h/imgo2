"""Five-expert CMoE configuration for the Imgo2 PPO rough task."""

from isaaclab.utils import configclass

from rl_lab.config import CMoEActorCriticCfg, CMoEPPOAlgorithmCfg, CMoEOnPolicyRunnerCfg


@configclass
class Imgo2CMoERoughRunnerCfg(CMoEOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 100
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
        # critic = [base_lin_vel(3), policy proprioception(45), height_scan(187)]
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
