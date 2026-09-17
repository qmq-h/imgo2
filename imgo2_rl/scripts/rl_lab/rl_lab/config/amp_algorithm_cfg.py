from dataclasses import MISSING

from isaaclab.utils import configclass

from .base_runner_cfg import RLLabBaseRunnerCfg

@configclass
class AMPActorCriticCfg:
    """Configuration of the AMP actor-critic."""

    actor_hidden_dims: list[int] = [256, 256, 256]
    critic_hidden_dims: list[int] = [256, 256, 256]
    activation: str = "elu"
    init_noise_std: float = 1.0
    fixed_std: bool = False


@configclass
class AMPAlgorithmCfg:
    """Configuration of the AMP PPO algorithm."""

    num_learning_epochs: int = 5
    num_mini_batches: int = 4
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
    amp_replay_buffer_size: int = 1000000
    # 见 RLLabBaseRunnerCfg.clamp_noise_std。设为 False 时 min_normalized_std 不再生效，
    # 策略变化完全由 KL 自适应学习率约束（标准 PPO 行为）。
    clamp_noise_std: bool = True


@configclass
class AMPOnPolicyRunnerCfg(RLLabBaseRunnerCfg):
    """Configuration of the AMP on-policy runner."""

    class_name: str = "AMPOnPolicyRunner"
    policy_class_name: str = "ActorCritic"
    algorithm_class_name: str = "AMPPPO"
    policy: AMPActorCriticCfg = MISSING
    algorithm: AMPAlgorithmCfg = MISSING
    include_history_steps: int | None = None
    amp_motion_files: list[str] = MISSING
    amp_num_preload_transitions: int = 2000000
    amp_reward_coef: float = 2.0
    amp_discr_hidden_dims: list[int] = [1024, 512]
    amp_task_reward_lerp: float = 0.1
    min_normalized_std: list[float] = [0.05, 0.02, 0.05] * 4
