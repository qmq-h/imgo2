"""rl_lab configuration for the upper command-shaping policy (not registered yet)."""

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
    decoder = TowingDecoderCfg(
        frame_dim=51,
        feature_dim=128,
        hidden_dim=128,
        num_layers=1,
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
