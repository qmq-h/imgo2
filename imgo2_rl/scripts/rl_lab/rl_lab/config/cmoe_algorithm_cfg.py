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
    # 控制台输出过滤（2026-09-25 用户要求「log 中一堆 gait 不打印」）：逐列指标**照常写 TensorBoard**，
    # 只是不刷控制台。可选 "gait"（默认，静音 `gait_*_<地形>` 的 7 组 × 8 列，保留 `gait_*_mean`）、
    # "all"（`Curriculum/terrain_levels/` 下所有逐列指标都静音，另含 `level_<地形>`/`tracking_<地形>`）、
    # "none"（旧行为，全打印）。
    console_skip_columns: str = "gait"

    # --- 「从已有步态策略起步」（2026-09-25 用户决定）-------------------------------------
    # 把一份 **45 维先验**（AMP 训练 checkpoint，或 `cmoe/play.py` 导出的 TorchScript）
    # 零填充装进专家（第一层 (512,45)→(512,157)，新增 112 列权重为 0），再开始训练：
    # 步态由先验提供，训练负责地形与跟速。配套使用 `Imgo2-basemove-rough-cmoe-gaitfree`
    # 任务（该项把手工业步态 shaping 归零）。契约与依据见
    # docs/cmoe_trot_warmstart_2026-09-25.md；离线核对见 scripts/tools/check_cmoe_expert_init.py。
    init_experts_from: str | None = None
    init_experts_mode: str = "all"  # "all"＝每个专家都装（则初始混合恒等于先验、与门控无关）；"first"＝只装第 0 个
    # ⚠️ 先验 critic 是在**另一套奖励尺度**下训的 ⇒ 搬过来会让早期 advantage 尺度错配，
    # 可能迅速改写刚装好的 actor。2026-09-25 首次实跑（_init）前 100 轮就明显差于随机初始化对照，
    # 首要嫌疑即此 ⇒ 想只搬 actor 时设 False（见 pretrained_prior.install_prior 的说明）。
    init_experts_critic: bool = True
    # 是否连先验的**噪声 std**（AMP 学出来的 (12,)，均值 0.334）一起装。默认装：它是"这份行为"
    # 的一部分（CMoE 默认 init_noise_std=1.0 的噪声会盖过先验的动作均值，第 0 步就不是先验步态）；
    # 它是可学习参数，训练中自己会调（实测 0.334→0.353）。
    init_experts_std: bool = True
    # 给专家的**新增 112 列**加 N(0, sigma) 扰动，打破"5 个专家完全相同"的对称性。
    # 实测 sigma=0.01：混合 vs 先验 RMSE 0.028（仍≈先验），地形列权重份额 0.0000→0.0897。
    init_experts_jitter: float = 0.0
    log_prior_rmse: bool = True  # 有先验时记录 Policy/prior_action_rmse＝混合策略与先验的动作 RMSE（漂移度量）

    # ---------------- v5（2026-09-28 用户决定）：只用一个专家装先验 + 只在 flat 上锚定 ----------------
    # 原话："5 个专家只有一个用 amp 预训练结果初始化"、"只在 flat 上锚定"。
    # 依据：AMP 是**平地 + 地形盲**（45 维本体，连 height_scan 都没有）⇒ 把锚放在斜坡/台阶上等于
    # 强迫策略忽略地形（实测先验在极端地形上 OOD，RMSE 放大到 15+）；放在 flat 上则正好是
    # "平地始终 trot"。门控初始偏向专家 0（它=先验）⇒ 第 0 步行为仍是先验，同时 5 个专家彼此不同。
    init_gate_bias: int | None = None        # 非 None ⇒ 门控初始偏置到第 k 个专家（配合 mode="first"）
    init_gate_bias_margin: float = 4.0       # 偏置 margin（softmax 初始 ≈ one-hot(该专家)）
    init_gate_bias_shrink: float = 0.0       # >0 ⇒ 同时把门控末层权重缩小该倍数（让偏置初始更占主导）
    anchor_terrain_names: tuple = ("flat",)  # 只在哪些地形上锚（默认只 flat）
    anchor_expert: int = 0                   # 锚哪个专家（默认专家 0 ＝装先验的那个）
    anchor_coef: float = 0.0                 # 初始锚定权重（0 ⇒ 关闭本功能）；典型 0.2~0.3
    anchor_coef_final: float = 0.0           # 衰减到的最终权重（典型 0.05~0.1）
    anchor_decay_iters: int = 1000           # 线性衰减到 final 所需轮数
