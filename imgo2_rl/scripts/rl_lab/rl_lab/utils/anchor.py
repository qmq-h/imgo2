"""先验锚定的**损失算式**（纯 torch，2026-09-28 v5 的输出层锚）。

把"锚"的数学从 `cmoe_ppo.py` 里抽出来，只为一件事：**能在没有 GPU / Isaac 的机器上单测**
（`tests/test_cmoe_v5_anchoring.py`）。语义必须与算法里逐行一致：

```
L = Σ_i m_i · mean_d (pred_i,d − teacher_i,d)²  /  Σ_i m_i        （对每个 pred 取平均）
```

* `m` 是**逐样本 0/1 掩码**（`[N]` 或 `[N,1]`）：runner 按地形算好写进 storage，不在目标地形上就是 0
  ⇒ 这些样本**完全不参与**（不是"参与但权重小"）；
* ⚠️ **公共系数必须乘在损失外面，不能乘进掩码**（2026-09-28 复查抓到的真 bug）：
  `Σ(c·m·MSE)/Σ(c·m) ≡ Σ(m·MSE)/Σ(m)` —— 归一化会把公共标量**约掉**，于是 `anchor_coef`
  从 0.2 改成 0.3 也不会有任何变化（实测两者都得到 0.972105，差 0.00e+00）。
  本函数因此**只接受 0/1**，非 0/1 直接报错；强度由调用方 `λ · weighted_mse(...)` 提供。
* `Σ w == 0`（没有教师或全不在目标地形上）⇒ 返回 0（调用方据此判定"未生效"、不写日志）；
* `predictions` 是一个**列表**：`[专家 k 的输出]`（只锚某个专家）或 `[混合均值]`（锚最终输出）
  或两者都给（`anchor_target="both"`）⇒ 各算一份再取平均。
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch


def weighted_mse(
    predictions: Sequence[torch.Tensor] | torch.Tensor,
    teacher_actions: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """逐样本加权的动作 MSE（对 `predictions` 里每一项各算一次再平均）。

    Args:
        predictions: 单个 `[N, D]`，或一组 `[N, D]`（`anchor_target="both"` 时两项）。
        teacher_actions: `[N, D]`，**已 detach** 的教师均值动作。
        mask: `[N]` 或 `[N, 1]` 的**0/1 掩码**（来自 storage 的 `anchor_weight`）。
            非 0/1 一律报错——公共系数乘进来会被归一化约掉（见模块 docstring 的 bug 说明）。
    """
    if isinstance(predictions, torch.Tensor):
        predictions = [predictions]
    predictions = list(predictions)
    if not predictions:
        raise ValueError("predictions 不能为空")
    weight = mask.reshape(-1)
    if weight.shape[0] != teacher_actions.shape[0]:
        raise ValueError(
            f"掩码长度 {weight.shape[0]} 与教师动作批大小 {teacher_actions.shape[0]} 不一致"
        )
    if not bool(torch.all((weight == 0) | (weight == 1))):
        raise ValueError(
            "anchor 掩码必须是 0/1：归一化的 Σ(w·MSE)/Σw 会把公共系数约掉"
            "（anchor_coef=0.2 与 0.3 得到同一损失）⇒ 强度必须用 λ·weighted_mse(...) 乘在外面"
        )
    weight_sum = weight.sum()
    if float(weight_sum) <= 0.0:
        return torch.zeros((), device=teacher_actions.device, dtype=teacher_actions.dtype)
    total = torch.zeros((), device=teacher_actions.device, dtype=teacher_actions.dtype)
    for prediction in predictions:
        if prediction.shape != teacher_actions.shape:
            raise ValueError(
                f"预测形状 {tuple(prediction.shape)} 与教师 {tuple(teacher_actions.shape)} 不一致"
            )
        per_sample = torch.square(prediction - teacher_actions).mean(dim=-1)
        total = total + (weight * per_sample).sum() / weight_sum.clamp_min(1e-6)
    return total / len(predictions)


def resolve_targets(anchor_target: str) -> tuple[bool, bool]:
    """`anchor_target` → `(锚专家, 锚混合输出)`。非法取值显式报错（防静默退化）。"""
    mapping = {
        "expert": (True, False),
        "mixture": (False, True),
        "both": (True, True),
    }
    if anchor_target not in mapping:
        raise ValueError(
            f"anchor_target={anchor_target!r} 非法，只能是 {sorted(mapping)}"
        )
    return mapping[anchor_target]


def mix_experts(expert_means: Iterable[torch.Tensor], gate_weights: torch.Tensor) -> torch.Tensor:
    """按门控权重把各专家均值合成最终动作（与 `CMoEActorCritic.act_inference` 同一算式）。

    `expert_means` 是 `K` 个 `[N, D]`；`gate_weights` 是 `[N, K]`（已 softmax）。
    """
    stacked = torch.stack(list(expert_means), dim=1)          # [N, K, D]
    return (stacked * gate_weights.unsqueeze(-1)).sum(dim=1)  # [N, D]
