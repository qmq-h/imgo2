from dataclasses import MISSING

from isaaclab.utils import configclass

from .base_runner_cfg import RLLabBaseRunnerCfg


@configclass
class HIMPPOActorCriticCfg:
    """Configuration of the HimLoco actor-critic."""

    actor_hidden_dims: list[int] = [512, 256, 128]
    critic_hidden_dims: list[int] = [512, 256, 128]
    activation: str = "elu"
    init_noise_std: float = 1.0
    normalize_obs: bool = False


@configclass
class HIMPPPOAlgorithmCfg:
    """Configuration of the HimLoco PPO algorithm."""

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
class HIMOnPolicyRunnerCfg(RLLabBaseRunnerCfg):
    """Configuration of the HimLoco on-policy runner."""

    class_name: str = "HIMOnPolicyRunner"
    policy_class_name: str = "HIMActorCritic"
    algorithm_class_name: str = "HIMPPO"
    policy: HIMPPOActorCriticCfg = MISSING
    algorithm: HIMPPPOAlgorithmCfg = MISSING
    history_length: int = 0
    privileged_history_length: int = 0
