from dataclasses import MISSING

from isaaclab.utils import configclass

from .base_runner_cfg import RLLabBaseRunnerCfg
from .ppo_algorithm_cfg import PPOAlgorithmCfg


@configclass
class TowingActorCriticCfg:
    actor_hidden_dims: list[int] = [256, 128, 64]
    critic_hidden_dims: list[int] = [256, 128, 64]
    activation: str = "elu"
    init_noise_std: float = 0.5
    fixed_std: bool = False
    rnn_type: str = "gru"
    rnn_hidden_size: int = 256
    rnn_num_layers: int = 1


@configclass
class TowingDecoderCfg:
    """CE Net：GRU → 显式 5 维（vel 2 + force 3）+ latent 16 → 重建下一帧（契约 **v5**）。

    质量不显式预测（2026-10-16 准则：decoder 只建模与四足机器人直接相关的量，质量交给
    latent）；因此这里也没有 `mass_coef`。2026-10-17 起加 `reconstruction_coef`：
    显式 5 维与 latent 16 拼接（21 维）重建下一时刻的 58 维 policy 帧。
    """

    # 必须等于环境的 policy 帧维数（`upper_logic.UpperObservationSpec.frame_dim`，契约 v3
    # 起为 58：`last_action` 12 → 13）；runner 会断言它与 `env.num_obs` 一致。
    frame_dim: int = 58
    feature_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 1
    latent_dim: int = 16
    kld_weight: float = 0.005
    reconstruction_coef: float = 1.0
    force_scale: float = 10.0
    learning_rate: float = 1.0e-3
    max_grad_norm: float = 1.0
    velocity_coef: float = 5.0
    force_coef: float = 10.0


@configclass
class TowingOnPolicyRunnerCfg(RLLabBaseRunnerCfg):
    class_name: str = "TowingOnPolicyRunner"
    policy_class_name: str = "ActorCriticRecurrent"
    algorithm_class_name: str = "PPO"
    policy: TowingActorCriticCfg = MISSING
    algorithm: PPOAlgorithmCfg = MISSING
    decoder: TowingDecoderCfg = MISSING
    clip_actions: float | None = 1.0
    critic_empirical_normalization: bool = True
    critic_normalization_clip: float = 10.0
