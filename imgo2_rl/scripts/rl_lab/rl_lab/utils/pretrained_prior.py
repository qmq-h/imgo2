# SPDX-License-Identifier: BSD-3-Clause
"""把 45 维运动先验移植进 CMoE 专家的工具函数（离线核对与训练期初始化共用）。

## 为什么需要它

CMoE 的 expert 输入是 **157 维**：45 当前本体感知 + 3 `explicit` + 16 `state_latent`
+ 77 地形扫描 + 16 `terrain_latent`（见 `modules/cmoe_actor_critic.py::actor_input_dim`），
而现成的 PPO / AMP 策略只吃 **45 维**本体感知。两者的前 45 维在顺序、尺度、动作定义上
逐项一致（训练侧 `tasks/.../base_move/rough_env_cfg.py` 与部署侧
`imgo2_deploy/policy/imgo2/{amp,ppo}/config.yaml`，核对脚本
`scripts/tools/check_cmoe_expert_init.py`），所以**不需要改观测**：把先验策略的第一层
从 `(512, 45)` 零填充成 `(512, 157)` 即可 —— 新增的 112 列权重为 0。

「零填充」的准确含义是**通路接进来了但初始权重为零**，不是「没有这个输入」：

* 结构上 77 维地形照常喂进 expert，只是初始化时它对输出没有贡献（先验本来是平地策略，
  它不可能"用过"地形）；
* 梯度不会被零权重挡住（``dL/dw = dL/dy * x``，地形输入非零就有梯度），地形通路从零学；
* 因此在初始化那一刻，混合策略（5 个专家都由先验初始化时）**恒等于先验**，与门控权重
  无关 —— 这正是"第 0 步就是先验步态"的依据。

若要让地形通路一开始就有容量，做法是把这 112 维放进**独立残差 adapter**（末层零初始化）：
``expert_out = trunk(45) + adapter(112)``，数学上与零填充等价，但不扰动预训练主干。

## 边界

* 本模块只依赖 ``torch``，不触发 Isaac Lab，可在没有 GPU 的机器上跑。
* 只做**权重移植**，不改任何观测/动作配置；契约不一致时断言失败而不是硬塞。
* ``torch.load(..., weights_only=False)`` 只用于本仓库自己的训练 checkpoint；
  不要指向不可信来源的文件。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

# nn.Sequential 里 Linear 的下标：Linear, ELU, Linear, ELU, Linear, ELU, Linear
LINEAR_INDICES = (0, 2, 4, 6)


@dataclass
class PriorWeights:
    """一份可移植的先验策略权重。

    ``actor`` / ``critic`` 用 ``nn.Sequential``（MLP）的键名（``"0.weight"`` 等）索引，
    与 `CMoEExpertActorCritic` 里 ``actor`` / ``critic`` 的键名一致，故可直接
    ``load_state_dict(zero_pad_first_layer(...))``。
    """

    source: str
    path: str
    actor: dict[str, torch.Tensor]
    actor_obs_dim: int
    action_dim: int
    critic: dict[str, torch.Tensor] | None = None
    critic_obs_dim: int | None = None
    std: torch.Tensor | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def has_critic(self) -> bool:
        return self.critic is not None

    def summary(self) -> str:
        parts = [
            f"source={self.source}",
            f"actor {self.actor_obs_dim}->{self.action_dim}",
        ]
        if self.critic is not None:
            parts.append(f"critic {self.critic_obs_dim}->1")
        parts.append(f"std={'yes' if self.std is not None else 'no'}")
        return ", ".join(parts)


def _check_mlp_keys(weights: Mapping[str, torch.Tensor], label: str) -> None:
    expected = {f"{i}.{name}" for i in LINEAR_INDICES for name in ("weight", "bias")}
    missing = expected.difference(weights)
    if missing:
        raise ValueError(f"{label}: MLP 权重缺键 {sorted(missing)}；实际键 {sorted(weights)}")
    extra = set(weights).difference(expected)
    if extra:
        raise ValueError(f"{label}: MLP 权重有多余键 {sorted(extra)}（层数不是 3 隐层？）")


def _mlp_dims(weights: Mapping[str, torch.Tensor]) -> tuple[int, int]:
    """(输入维, 输出维)。"""
    return int(weights["0.weight"].shape[1]), int(weights["6.weight"].shape[0])


def load_prior(path: str | Path, source: str = "auto") -> PriorWeights:
    """从 AMP 训练 checkpoint 或部署 TorchScript 导出件读取先验权重。

    Args:
        path: ``model_XXXX.pt``（AMP 训练 checkpoint，含 actor/critic/std）或
            ``policy.pt``（`play.py` 导出的 TorchScript，只有 actor）。
        source: ``"amp"`` / ``"ppo"`` / ``"auto"``。``auto`` 按内容判别：
            能被 ``torch.load`` 读出 ``model_state_dict`` 的当 checkpoint，
            能被 ``torch.jit.load`` 读出 ``mlp`` 的当 TorchScript。
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"先验权重不存在: {path}")
    if source not in {"auto", "amp", "ppo"}:
        raise ValueError(f"未知 source: {source}")

    resolved = source
    if source == "auto":
        resolved = "amp" if _looks_like_checkpoint(path) else "ppo"
    return _load_checkpoint(path) if resolved == "amp" else _load_torchscript(path)


def _looks_like_checkpoint(path: Path) -> bool:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 - 判别失败就按 TorchScript 试
        return False
    return isinstance(obj, dict) and "model_state_dict" in obj


def _load_checkpoint(path: Path) -> PriorWeights:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    actor = {k[len("actor.") :]: v for k, v in state.items() if k.startswith("actor.")}
    critic = {k[len("critic.") :]: v for k, v in state.items() if k.startswith("critic.")}
    _check_mlp_keys(actor, "actor")
    if critic:
        _check_mlp_keys(critic, "critic")
    actor_in, action_dim = _mlp_dims(actor)
    prior = PriorWeights(
        source="amp-checkpoint",
        path=str(path),
        actor=dict(actor),
        actor_obs_dim=actor_in,
        action_dim=action_dim,
        critic=dict(critic) if critic else None,
        critic_obs_dim=_mlp_dims(critic)[0] if critic else None,
        std=state.get("std"),
    )
    if prior.critic is not None and int(prior.critic["6.weight"].shape[0]) != 1:
        raise ValueError("critic 输出不是 1 维，键名对应关系可能搞错了")
    iteration = checkpoint.get("iter")
    if iteration is not None:
        prior.notes.append(f"iter={iteration}")
    return prior


def _load_torchscript(path: Path) -> PriorWeights:
    module = torch.jit.load(str(path), map_location="cpu")
    state = {k: v for k, v in module.named_parameters()}
    actor = {k[len("mlp.") :]: v for k, v in state.items() if k.startswith("mlp.")}
    _check_mlp_keys(actor, "mlp")
    actor_in, action_dim = _mlp_dims(actor)
    prior = PriorWeights(
        source="torchscript",
        path=str(path),
        actor=dict(actor),
        actor_obs_dim=actor_in,
        action_dim=action_dim,
    )
    normalizer = getattr(module, "obs_normalizer", None)
    name = getattr(normalizer, "original_name", type(normalizer).__name__) if normalizer is not None else "none"
    if name != "Identity":
        prior.notes.append(
            f"⚠️ obs_normalizer={name}（不是 Identity）⇒ 该先验期望的是**归一化后**的观测，"
            "零填充移植会喂错尺度，必须先还原归一化参数"
        )
    else:
        prior.notes.append("obs_normalizer=Identity")
    return prior


def zero_pad_first_layer(weights: Mapping[str, torch.Tensor], input_dim: int) -> dict[str, torch.Tensor]:
    """把 MLP 的第一层从 ``(hidden, prior_in)`` 零填充到 ``(hidden, input_dim)``。

    先验输入维必须 <= ``input_dim``：前 ``prior_in`` 列拷权重，其余列置 0；bias 与其余层原样。
    """
    prior_in, _ = _mlp_dims(weights)
    if input_dim < prior_in:
        raise ValueError(f"目标输入维 {input_dim} < 先验输入维 {prior_in}，不能零填充（只能截断，禁止）")
    padded: dict[str, torch.Tensor] = {}
    for index in LINEAR_INDICES:
        weight = weights[f"{index}.weight"]
        bias = weights[f"{index}.bias"]
        if index == 0 and input_dim != prior_in:
            new_weight = torch.zeros(weight.shape[0], input_dim, dtype=weight.dtype)
            new_weight[:, :prior_in] = weight
            padded["0.weight"] = new_weight
        else:
            padded[f"{index}.weight"] = weight.clone()
        padded[f"{index}.bias"] = bias.clone()
    return padded


def load_into_sequential(module: nn.Sequential, weights: Mapping[str, torch.Tensor], input_dim: int) -> dict[str, Any]:
    """把先验权重装进一个 MLP（``nn.Sequential``），返回移植报告。"""
    first = module[0] if len(module) else None
    if not isinstance(first, nn.Linear):
        raise TypeError(f"目标模块的第一层不是 nn.Linear：{type(first).__name__}")
    if first.in_features != input_dim:
        raise ValueError(f"目标模块输入维 {first.in_features} != 期望 {input_dim}")
    prior_in, _ = _mlp_dims(weights)
    module.load_state_dict(zero_pad_first_layer(weights, input_dim), strict=True)
    return {
        "input_dim": input_dim,
        "prior_input_dim": prior_in,
        "zeroed_columns": input_dim - prior_in,
        "first_layer": (first.out_features, input_dim),
    }


def install_prior(actor_critic: nn.Module, prior: PriorWeights, mode: str = "all",
                  include_critic: bool = True, include_std: bool = True) -> dict[str, Any]:
    """把先验装进 `CMoEActorCritic` 的专家（并可带 critic / std）。

    Args:
        actor_critic: ``CMoEActorCritic`` 实例，或其 ``experts`` 列表里单个
            ``CMoEExpertActorCritic``。
        mode: ``"all"``=每个专家都装；``"first"``=只装第 0 个专家。
        include_critic: 是否连 **critic** 一起装。⚠️ 见下。
        include_std: 是否把先验的**噪声 `std`**（AMP 学出来的 `(12,)`，均值 0.334）一起装。
            默认装：它属于"这份行为"的一部分（`std=1.0` 的默认噪声会**盖过**先验的动作均值，
            第 0 步就不再是先验的步态）；它是可学习参数，训练中会自己调整（实测 0.334→0.353）。

    ⚠️ **critic 该不该装是个真问题**：先验的 critic 是在**另一个任务/另一套奖励尺度**下训的
    （AMP 平地：critic 48 维；CMoE：critic 125 维 = 前 48 维 + 77 维地形）。把值函数搬过来会让
    早期的 advantage 尺度错配，可能**把刚装好的 actor 迅速改写**——2026-09-25 的
    `cmoe_gaitfree_amp24500_init` 前 100 轮 `illegal_contact` 0.72、`mean_reward` −3，
    明显差于同期"随机初始化"对照（0.45 / +16），首要嫌疑就是这个。
    ``include_critic=False`` 只装 actor（+std），留给 `RLRunnerCfg.init_experts_critic` 控制。
    """
    if mode not in {"all", "first"}:
        raise ValueError(f"未知 mode: {mode}")

    if hasattr(actor_critic, "experts"):
        experts = list(actor_critic.experts)
        target_count = len(experts) if mode == "all" else 1
        selected = experts[:target_count]
        report: dict[str, Any] = {
            "installed_experts": target_count,
            "total_experts": len(experts),
            "critic_installed": bool(include_critic and prior.has_critic),
            "experts": [],
        }
    else:
        selected = [actor_critic]
        report = {"installed_experts": 1, "total_experts": 1,
                  "critic_installed": bool(include_critic and prior.has_critic), "experts": []}

    if hasattr(actor_critic, "actor_input_dim"):
        actor_input_dim = int(actor_critic.actor_input_dim)
    else:
        actor_input_dim = int(selected[0].actor[0].in_features)

    for expert in selected:
        entry = {"actor": load_into_sequential(expert.actor, prior.actor, actor_input_dim)}
        if include_critic and prior.critic is not None:
            critic_dim = int(expert.critic[0].in_features)
            entry["critic"] = load_into_sequential(expert.critic, prior.critic, critic_dim)
        report["experts"].append(entry)

    if include_std and prior.std is not None and hasattr(actor_critic, "std"):
        std_param = actor_critic.std
        if tuple(std_param.shape) != tuple(prior.std.shape):
            raise ValueError(f"std 形状不匹配：模型 {tuple(std_param.shape)} vs 先验 {tuple(prior.std.shape)}")
        with torch.no_grad():
            std_param.copy_(prior.std.to(std_param.device, std_param.dtype))
        report["std"] = {
            "shape": tuple(std_param.shape),
            "mean": float(std_param.mean().item()),
        }
    return report


def prior_actions(policy: nn.Module, observations: torch.Tensor, one_step_obs: int, clip: float | None = 3.0) -> torch.Tensor:
    """用 45 维先验对 CMoE 的 527 维观测出动作（step-0 教师回放的核心逻辑）。

    CMoE 的 actor 观测是 ``[history_steps × one_step_obs | 地形扫描]``，**当前帧在最前面**
    （`wrapper/cmoe_vec_env_wrapper.py`：``obs_history_buf[:, 0] = current_policy``）。先验策略
    只吃当前帧本体感知，所以取前 ``one_step_obs`` 列 —— 与训练侧专家看到的那一份完全一致，
    也**不需要改任何观测**（契约见 `scripts/tools/check_cmoe_expert_init.py`）。

    Args:
        policy: 45 维先验的 MLP（``teacher_actor(prior)`` 或任何接受 45 维输入的模块）。
        observations: CMoE 观测，最后一维必须是 45 的整数倍加地形维数。
        one_step_obs: 单帧本体感知维数（本项目为 45）。
        clip: 动作裁剪；部署侧与训练侧动作项都裁 ±3，``None`` 或 ``<=0`` 表示不裁。
    """
    if observations.shape[-1] < one_step_obs:
        raise ValueError(
            f"CMoE 观测最后一维 {observations.shape[-1]} 小于单帧本体感知维数 {one_step_obs}，无法取当前帧"
        )
    actions = policy(observations[:, :one_step_obs])
    if clip is not None and clip > 0:
        actions = actions.clamp(-float(clip), float(clip))
    return actions


def init_gate_bias(
    actor_critic: nn.Module,
    expert_index: int = 0,
    margin: float = 4.0,
    shrink_last_layer: float = 0.0,
) -> dict[str, Any]:
    """把门控网络的初始输出**偏向第 `expert_index` 个专家**（2026-09-28，v5 方案）。

    **为什么需要它**：`install_prior(..., mode="first")` 只把先验装进专家 0，其余 4 个是随机/抖动策略；
    门控又是个**新网络**（随机初始化），它的初始 softmax 与"专家 0"毫无关系 ⇒ 5 个专家**等权混合**，
    step-0 行为离先验很远（离线实测动作 RMSE **2.33**）。把门控末层 bias 设成
    `bias[k] = margin`、其余 0（可选再把末层权重缩小 `shrink_last_layer` 倍）⇒ 初始
    `softmax ≈ one-hot(k)` ⇒ **第 0 步的混合动作 ≈ 专家 0 ＝ 先验的 trot**，而 5 个专家仍然彼此不同
    （这正是 MoE 需要的"多样性 + 一个可靠参考"）。

    只改 `gating_network` 的**最后一层 Linear**（`bias`，以及可选的 `weight`），不动任何专家参数。
    返回报告 dict（启动打印 + 测试断言用）。
    """
    gating = getattr(actor_critic, "gating_network", None)
    if gating is None:
        raise ValueError("actor_critic 没有 gating_network，无法设置门控偏置")
    last = None
    for module in gating.modules():
        if isinstance(module, nn.Linear):
            last = module
    if last is None:
        raise ValueError("gating_network 里找不到 nn.Linear")
    num_experts = int(last.out_features)
    index = int(expert_index)
    if not 0 <= index < num_experts:
        raise ValueError(f"expert_index={index} 超出 [0, {num_experts})")
    with torch.no_grad():
        bias = torch.zeros(num_experts, dtype=last.bias.dtype, device=last.bias.device)
        bias[index] = float(margin)
        last.bias.copy_(bias)
        if shrink_last_layer and shrink_last_layer != 1.0:
            last.weight.mul_(float(shrink_last_layer))
    return {
        "expert": index,
        "margin": float(margin),
        "shrink_last_layer": float(shrink_last_layer),
        "num_experts": num_experts,
    }


def jitter_new_columns(actor_critic: nn.Module, one_step_obs: int, sigma: float,
                       keep_first_pure: bool = True) -> dict[str, Any]:
    """给专家的**新增列**（地形/估计器那 112 列）加小随机扰动，打破"5 个专家完全相同"的对称性。

    为什么要它（2026-09-25 实测）：`mode="all"` 让 5 个专家完全一样 ⇒ 初始混合**严格等于先验**
    （RMSE 0.000，与门控无关），但 5 个专家没有任何分工、门控也没有理由分化。
    本函数只动 `actor[0].weight[:, one_step_obs:]`（前 45 列＝先验本体，一字不动），实测
    `sigma=0.01` 时：混合 vs 先验 RMSE **0.028**（仍≈先验），而地形列权重份额从 0.0000 → **0.0897**
    ⇒ 既保住初始步态、又让专家一开始就不同。`sigma=0` 表示不动。
    """
    if sigma <= 0:
        return {"jittered_experts": 0, "sigma": sigma}
    experts = list(getattr(actor_critic, "experts", [actor_critic]))
    start = 1 if keep_first_pure else 0
    with torch.no_grad():
        for expert in experts[start:]:
            weight = expert.actor[0].weight
            weight[:, one_step_obs:].normal_(0.0, float(sigma))
    return {"jittered_experts": max(len(experts) - start, 0), "sigma": float(sigma)}


def normalize_prior_path(value: object) -> str | None:
    """把配置/CLI 里的先验路径规范化：``None``／空串／纯空白 ⇒ ``None``（视为没给）。

    为什么需要：2026-09-25 的 `cmoe_gaitfree_amp24500` 实跑里 `--init_experts_from` 传成了空串，
    配置里落成 `init_experts_from: ''` 而 `if init_from:` 判假 ⇒ **静默地没装先验**，
    训练照跑、日志里只是少一个 `Policy/prior_action_rmse`，不容易发现。
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@torch.no_grad()
def prior_action_rmse(actor_critic: nn.Module, policy: nn.Module, observations: torch.Tensor,
                      one_step_obs: int) -> torch.Tensor:
    """混合策略（`CMoEActorCritic.act_inference`）与冻结先验在**动作均值**上的 RMSE。

    这是"先验漂没漂"的**唯一量化读数**：全专家用同一先验初始化时它应当是 0（且与门控无关，
    见 `install_prior` 的实测）；训练开始后它单调上升就说明先验正在被改写。**不做裁剪**——
    裁到 ±3 会把大偏差抹平，看不出漂移。

    Args:
        actor_critic: `CMoEActorCritic`（吃 527 维观测）。
        policy: 45 维先验的 MLP（`teacher_actor(prior)`）。
        observations: CMoE 观测，最后一维 = `history_steps × one_step_obs + 地形维数`。
        one_step_obs: 单帧本体感知维数（本项目 45）。
    """
    teacher_actions = policy(observations[:, :one_step_obs])
    actor_actions = actor_critic.act_inference(observations)
    return (actor_actions - teacher_actions).float().pow(2).mean().sqrt()


def teacher_actor(prior: PriorWeights) -> nn.Sequential:
    """按先验权重搭一个与原策略同形的 MLP，用作等价性对照（不改动专家）。"""
    hidden = [int(prior.actor[f"{i}.weight"].shape[0]) for i in LINEAR_INDICES[:-1]]
    input_dim = int(prior.actor["0.weight"].shape[1])
    layers: list[nn.Module] = []
    width = input_dim
    for size in hidden:
        layers.extend((nn.Linear(width, size), nn.ELU()))
        width = size
    layers.append(nn.Linear(width, int(prior.actor["6.weight"].shape[0])))
    model = nn.Sequential(*layers)
    model.load_state_dict({k: v.clone() for k, v in prior.actor.items()}, strict=True)
    model.eval()
    return model


def teacher_critic(prior: PriorWeights) -> nn.Sequential | None:
    """同 ``teacher_actor``，用于 critic 的等价性对照。"""
    if prior.critic is None:
        return None
    hidden = [int(prior.critic[f"{i}.weight"].shape[0]) for i in LINEAR_INDICES[:-1]]
    input_dim = int(prior.critic["0.weight"].shape[1])
    layers: list[nn.Module] = []
    width = input_dim
    for size in hidden:
        layers.extend((nn.Linear(width, size), nn.ELU()))
        width = size
    layers.append(nn.Linear(width, 1))
    model = nn.Sequential(*layers)
    model.load_state_dict({k: v.clone() for k, v in prior.critic.items()}, strict=True)
    model.eval()
    return model
