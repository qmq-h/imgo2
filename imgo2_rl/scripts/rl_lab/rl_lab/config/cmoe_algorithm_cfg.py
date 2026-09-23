"""Configuration schema for the repository-local CMoE training stack."""

from dataclasses import MISSING

from isaaclab.utils import configclass

from .base_runner_cfg import RLLabBaseRunnerCfg


@configclass
class CMoEActorCriticCfg:
    num_experts: int = 5
    state_latent_dim: int = 16
    terrain_latent_dim: int = 16
    explicit_dim: int = 3
    actor_hidden_dims: list[int] = [512, 256, 128]
    critic_hidden_dims: list[int] = [512, 256, 128]
    gate_hidden_dim: int = 128
    num_prototypes: int = 32
    projection_dim: int = 16
    temperature: float = 0.2
    activation: str = "elu"
    init_noise_std: float = 1.0
    critic_explicit_start: int | None = None
    critic_proprio_start: int = 0


@configclass
class CMoEPPOAlgorithmCfg:
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
    contrastive_loss_coef: float = 1.0


@configclass
class CMoEOnPolicyRunnerCfg(RLLabBaseRunnerCfg):
    class_name: str = "CMoEOnPolicyRunner"
    policy_class_name: str = "CMoEActorCritic"
    algorithm_class_name: str = "CMoEPPO"
    policy: CMoEActorCriticCfg = MISSING
    algorithm: CMoEPPOAlgorithmCfg = MISSING
    history_steps: int = 10
