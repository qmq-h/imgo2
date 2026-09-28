# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""抬脚高度奖励的**纯数学**（只依赖 `torch`，可离线单测）。

2026-09-28（用户定稿）：`-gaitfree` 链路里 `feet_height_body` 与 `feet_height` 都被归零
⇒ 训练对"脚离地多高"完全没有压力（实测平地滞空从先验的 0.084 s 掉到 0.015 s，在蹭地拖行）。
本模块实现新项 `feet_swing_clearance` 的算式：

```text
z_ref   = mean(世界 z of 接触中的足)          # 支撑面参考；只用 contact 传感器判定
h_i     = z_foot_i_world − z_ref              # 第 i 只脚的"相对支撑面离地高度"
swing_i = 1{第 i 只脚 不 在接触}
r       = k · Σ_i swing_i · clamp(1 − |h_i − h*| / band, 0, 1)
```

其中 `h*`（`target`）与 `band` 按地形给不同值（见 `CMoE_env_cfg.py` 的
`SWING_CLEARANCE_TERRAIN_GROUPS`）。`k`、命令门、直立门与地形掩码**不在本模块**，
它们在 `mdp/rewards.py::FeetSwingClearance` 里（本模块只做"给定参考面与参数，算出带形分数"）。

为什么单独拆一个文件：`mdp/rewards.py` 顶层 `import isaaclab`（本机缺 `omni.log`）**整模块
import 不了**，而这三条数学恰好是最容易写错、也最值得钉死的部分（参考面/飞行相/band=0）。

数值细节（写进 docs 的"未验证项"）：`(z−d)−(z_ref−d)` 与 `z−z_ref` 在浮点下**不恒等**
（`d=0.05` 不是二进制精确数，一般会差 1 ulp，实测 ~1e-8）⇒ "压身体不变性"在数学上成立、
在浮点上到 1e-6 级；测试里用一组差值精确可表示的数值钉"逐元素完全不变"，另用随机值钉 1e-6 一致。
"""

from __future__ import annotations

import torch

__all__ = ["stance_reference", "swing_clearance_reward"]


def _as_per_env(value, reference: torch.Tensor) -> torch.Tensor:
    """把标量或 `[N]` 参数统一成 `[N]`（与 `reference` 同设备/同 dtype）。"""
    tensor = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if tensor.dim() == 0:
        return tensor.reshape(1).expand(reference.shape[0])
    if tensor.shape != reference.shape:
        raise ValueError(
            f"逐环境参数形状应为 {tuple(reference.shape)} 或标量，得到 {tuple(tensor.shape)}"
        )
    return tensor


def _band_fraction(error: torch.Tensor, band: torch.Tensor) -> torch.Tensor:
    """`clamp(1 − error/band, 0, 1)`，并对 `band = 0` 单独定义。

    `band = 0` 时 `1 − 0/0` 未定义 ⇒ 显式约定为**精确相等指示**：`error == 0 ⇒ 1`，否则 `0`
    （即"只认正好落在 `h*` 上"）。这样 `band=0` 不会产生 NaN，也不会静默变成满分。
    """
    positive = band > 0
    safe_band = torch.where(positive, band, torch.ones_like(band))
    shaped = torch.clamp(1.0 - error / safe_band, min=0.0, max=1.0)
    exact = (error <= 0).to(dtype=shaped.dtype)
    return torch.where(positive, shaped, exact)


def swing_clearance_reward(
    foot_z_world: torch.Tensor,
    contact_mask: torch.Tensor,
    *,
    target,
    band,
    reference: torch.Tensor,
    foot_lin_vel_xy: torch.Tensor | None = None,
    tanh_mult: float | None = None,
) -> torch.Tensor:
    """逐环境抬脚带形分数 `Σ_i swing_i · clamp(1 − |h_i − h*|/band, 0, 1)`（**不含 `k`**）。

    Args:
        foot_z_world: `[N, F]` 足端**世界系** z（用世界系是为了不受基座高度/俯仰影响）。
        contact_mask: `[N, F]` bool，接触判定（由调用方用 contact 传感器给出）。
        target: `h*`，标量或 `[N]`（按地形逐环境不同）。
        band: 容差半宽，标量或 `[N]`；`|h − h*| ≥ band ⇒ 该项为 0`。
        reference: `[N]` 支撑面参考 `z_ref`，**无效用 NaN/Inf 标记**（例如长时间无接触、
            或基座下方整束射线落空 ⇒ 调用方给 NaN）。非有限参考的样本**输出 0**。
        foot_lin_vel_xy: `[N, F, 2]`，仅当 `tanh_mult` 非空时使用。
        tanh_mult: 可选的"速度门"（与 `feet_height` 同形：`tanh(tanh_mult·‖v_xy‖)`）。
            默认 `None` ⇒ **不加门**，即上文那个严格算式；CMoE 配置里也是 `None`。

    Returns:
        `[N]` float；全接触（无摆动足）、无参考、band=0 边界都有定义（见上）。
    """
    foot_z = torch.as_tensor(foot_z_world)
    if foot_z.dim() != 2:
        raise ValueError(f"foot_z_world 应为 [N, F]，得到 {tuple(foot_z.shape)}")
    contacts = torch.as_tensor(contact_mask).to(dtype=torch.bool)
    if contacts.shape != foot_z.shape:
        raise ValueError(f"contact_mask 形状 {tuple(contacts.shape)} 与 foot_z_world {tuple(foot_z.shape)} 不一致")
    reference = torch.as_tensor(reference, device=foot_z.device, dtype=foot_z.dtype)
    if reference.dim() != 1 or reference.shape[0] != foot_z.shape[0]:
        raise ValueError(f"reference 应为 [N]，得到 {tuple(reference.shape)}（N={foot_z.shape[0]}）")

    target_t = _as_per_env(target, reference)
    band_t = _as_per_env(band, reference)

    # h_i = z_foot − z_ref ⇒ 整体抬/压同一个量（基座与足一起动）时 h_i 不变
    height = foot_z - reference.unsqueeze(1)
    error = (height - target_t.unsqueeze(1)).abs()
    shaped = _band_fraction(error, band_t.unsqueeze(1))

    if tanh_mult is not None:
        if foot_lin_vel_xy is None:
            raise ValueError("tanh_mult 非空时必须同时给 foot_lin_vel_xy")
        speed = torch.as_tensor(foot_lin_vel_xy, device=foot_z.device, dtype=foot_z.dtype)
        if speed.shape[:2] != foot_z.shape or speed.shape[-1] != 2:
            raise ValueError(f"foot_lin_vel_xy 应为 [N, F, 2]，得到 {tuple(speed.shape)}")
        shaped = shaped * torch.tanh(float(tanh_mult) * torch.linalg.norm(speed, dim=-1))

    swing = (~contacts).to(dtype=shaped.dtype)
    reward = (shaped * swing).sum(dim=1)
    # 无效参考（无接触超过保持步数 / 基座悬空）与数值异常一律输出 0 —— 不给"错误参考"付钱
    valid = torch.isfinite(reference) & torch.isfinite(reward)
    return torch.where(valid, reward, torch.zeros_like(reward))


def stance_reference(
    foot_z_world: torch.Tensor,
    contact_mask: torch.Tensor,
    fallback: torch.Tensor,
    max_hold_steps: int,
    hold_steps: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """逐环境支撑面参考 `z_ref`（纯函数；"已沿用了几步"由调用方传入）。

    | 情形 | `reference` | `valid` |
    |---|---|---|
    | 有 ≥1 只脚接触 | 接触足世界 z 的均值 | `True` |
    | 无接触，但 `fallback` 有限**且** `hold_steps < max_hold_steps` | `fallback`（沿用最近一次有效值） | `True` |
    | 其余（无历史 / 沿用超时） | `NaN` | `False` |

    Args:
        foot_z_world: `[N, F]` 足端世界系 z。
        contact_mask: `[N, F]` bool 接触。
        fallback: `[N]` 上一次有效参考；没有历史时用 NaN/Inf。
        max_hold_steps: 无接触时最多沿用多少步（`0` ⇒ 只要没有脚接触就无效）。
        hold_steps: `[N]` long，本轮**之前**已连续沿用的步数（`None` ⇒ 全 0）。

    Returns:
        `(reference, valid)`：`reference` 是 `[N]`（无效处为 `NaN`），`valid` 是 `[N]` bool。
        任务书允许用 mask 作无效标记；这里返回 mask 而不是 `None`，因为调用方要把它
        直接当布尔掩码用（`swing_clearance_reward` 也接受 NaN 参考并输出 0）。

    ⚠️ **与任务书"签名建议"的偏离（有意，已写进 docs）**：`max_hold_steps` 必须配合
    "已经沿用了几步"才有意义，而纯函数不能保存状态 ⇒ 多一个**入参** `hold_steps`、
    多一个**出参** `valid`。调用方 `FeetSwingClearance` 每步维护 `hold_steps`：
    有接触 ⇒ 归零；无接触 ⇒ `+1`（于是"最近一次接触之后最多沿用 `max_hold_steps` 步"）。
    """
    foot_z = torch.as_tensor(foot_z_world)
    if foot_z.dim() != 2:
        raise ValueError(f"foot_z_world 应为 [N, F]，得到 {tuple(foot_z.shape)}")
    contacts = torch.as_tensor(contact_mask).to(dtype=torch.bool)
    if contacts.shape != foot_z.shape:
        raise ValueError(f"contact_mask 形状 {tuple(contacts.shape)} 与 foot_z_world {tuple(foot_z.shape)} 不一致")
    fallback_t = torch.as_tensor(fallback, device=foot_z.device, dtype=foot_z.dtype)
    if fallback_t.dim() != 1 or fallback_t.shape[0] != foot_z.shape[0]:
        raise ValueError(f"fallback 应为 [N]，得到 {tuple(fallback_t.shape)}（N={foot_z.shape[0]}）")
    if hold_steps is None:
        hold = torch.zeros(foot_z.shape[0], dtype=torch.long, device=foot_z.device)
    else:
        hold = torch.as_tensor(hold_steps, device=foot_z.device).to(dtype=torch.long).reshape(-1)
        if hold.shape[0] != foot_z.shape[0]:
            raise ValueError(f"hold_steps 应为 [N]，得到 {tuple(hold.shape)}")

    any_contact = contacts.any(dim=1)
    count = contacts.sum(dim=1).clamp_min(1)
    mean_contact = torch.where(contacts, foot_z, torch.zeros_like(foot_z)).sum(dim=1) / count
    reusable = torch.isfinite(fallback_t) & (hold < int(max_hold_steps))
    reference = torch.where(any_contact, mean_contact, torch.where(reusable, fallback_t, torch.full_like(foot_z[:, 0], float("nan"))))
    return reference, (any_contact | reusable)
