from dataclasses import MISSING

from isaaclab.utils import configclass

from .base_runner_cfg import RLLabBaseRunnerCfg


@configclass
class PPOActorCriticCfg:
    """Configuration of the plain PPO actor-critic."""

    actor_hidden_dims: list[int] = [256, 256, 256]
    critic_hidden_dims: list[int] = [256, 256, 256]
    activation: str = "elu"
    init_noise_std: float = 1.0
    fixed_std: bool = False


@configclass
class PPOAlgorithmCfg:
    """Configuration of the plain PPO algorithm."""

    num_learning_epochs: int = 1
    num_mini_batches: int = 1
    clip_param: float = 0.2
    gamma: float = 0.998
    lam: float = 0.95
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.0
    learning_rate: float = 1.0e-3
    max_grad_norm: float = 1.0
    use_clipped_value_loss: bool = True
    schedule: str = "fixed"
    desired_kl: float = 0.01


@configclass
class PPOOnPolicyRunnerCfg(RLLabBaseRunnerCfg):
    """Configuration of the rl_lab PPO on-policy runner."""

    class_name: str = "OnPolicyRunner"
    policy_class_name: str = "ActorCritic"
    algorithm_class_name: str = "PPO"
    policy: PPOActorCriticCfg = MISSING
    algorithm: PPOAlgorithmCfg = MISSING
