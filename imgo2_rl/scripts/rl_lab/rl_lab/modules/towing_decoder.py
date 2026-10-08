"""Recurrent dynamics decoder used by the upper towing policy."""

import torch
import torch.nn as nn
import torch.nn.functional as F


# 输出布局（2026-10-08 用户决定）：[vx, vy, m, Fx, Fy, Fz]
# - 机器人速度保持 **2 维**（机体系水平）：`root_lin_vel_b[:, 2]` 在 trot 步态下按步频
#   大幅振荡，不是负载量，而上层也没有垂直控制通道；
# - 牵引力改为 **3 维**：两挂点高差 0.17 m，绷紧时方向向量 z 分量 = 0.17/L0，在
#   L0 = 0.4…0.8 m 上就是张力的 21–43%（\|Fz\|/T），且 Fz 在 0.16 m 后置挂点上产生俯仰
#   力矩。丢掉它会让「绳子往下拽」这件事对 actor 完全不可见。
# 布局常量集中在这里，head 宽度、loss 切片与 `force_newtons` 全部由它们推导，
# 避免再出现「改了 head 忘了切片」的静默错位。
VELOCITY_DIM = 2
MASS_DIM = 1
FORCE_DIM = 3
OUTPUT_DIM = VELOCITY_DIM + MASS_DIM + FORCE_DIM
MASS_INDEX = VELOCITY_DIM
FORCE_SLICE = slice(VELOCITY_DIM + MASS_DIM, OUTPUT_DIM)


class TowingDynamicsDecoder(nn.Module):
    """Estimate robot velocity, load mass and body-frame towing force."""

    def __init__(self, frame_dim=51, feature_dim=128, hidden_dim=128, num_layers=1, force_scale=10.0):
        super().__init__()
        self.frame_dim = frame_dim
        self.force_scale = force_scale
        self.encoder = nn.Sequential(nn.Linear(frame_dim, feature_dim), nn.ELU())
        self.gru = nn.GRU(feature_dim, hidden_dim, num_layers=num_layers)
        self.velocity_head = nn.Linear(hidden_dim, VELOCITY_DIM)
        self.mass_head = nn.Linear(hidden_dim, MASS_DIM)
        self.force_head = nn.Linear(hidden_dim, FORCE_DIM)

    @property
    def output_dim(self):
        """输出维数（= ``UpperObservationSpec.decoder_dim``）。"""
        return OUTPUT_DIM

    def forward(self, frames, hidden_state=None):
        single_step = frames.ndim == 2
        if single_step:
            frames = frames.unsqueeze(0)
        if frames.ndim != 3 or frames.shape[-1] != self.frame_dim:
            raise ValueError(
                f"decoder frames must have shape [T, B, {self.frame_dim}] or [B, {self.frame_dim}], "
                f"got {tuple(frames.shape)}")
        features = self.encoder(frames)
        features, hidden_state = self.gru(features, hidden_state)
        # 2026-09-23 去掉三个 head 的 tanh：直接回归**物理量**（m/s、kg、N）。
        # 理由：target 归一化到 [-1,1] 会把 loss 的物理权重按 s² 压低
        # （力 s=10 ⇒ 0.01、质量 s=5 ⇒ 0.04），使这两项几乎训不动；tanh 还会在饱和区丢梯度。
        # 稳定性由 `loss()` 的 smooth_l1(β=1) 提供：大误差处梯度线性、不爆炸。
        prediction = torch.cat((
            self.velocity_head(features),
            self.mass_head(features),
            self.force_head(features),
        ), dim=-1)
        return (prediction.squeeze(0) if single_step else prediction), hidden_state

    def force_newtons(self, prediction):
        """取 force head 的输出。去掉 tanh 与归一化后它本身就是牛顿，故为恒等。

        保留此入口是为了让调用方语义不变（部署端与 play 都走这里）。
        """
        return prediction[..., FORCE_SLICE]

    @staticmethod
    def loss(
        prediction,
        targets,
        mass_supervision_weight,
        *,
        velocity_coef=1.0,
        force_coef=1.0,
        mass_coef=1.0,
    ):
        """Supervise velocity/force always and mass after the first towing interaction."""
        if prediction.shape != targets.shape or prediction.shape[-1] != OUTPUT_DIM:
            raise ValueError(
                f"decoder shape mismatch: prediction={prediction.shape}, target={targets.shape}")
        weight = mass_supervision_weight.to(
            device=prediction.device, dtype=prediction.dtype).reshape(prediction.shape[:-1])
        velocity_loss = F.smooth_l1_loss(
            prediction[..., :VELOCITY_DIM], targets[..., :VELOCITY_DIM])
        force_loss = F.smooth_l1_loss(
            prediction[..., FORCE_SLICE], targets[..., FORCE_SLICE])
        mass_error = F.smooth_l1_loss(
            prediction[..., MASS_INDEX], targets[..., MASS_INDEX], reduction="none")
        mass_loss = (mass_error * weight).sum() / weight.sum().clamp_min(1.0)
        # 三项加权后的实际贡献单独返回，供日志核对「权重是否真的配平了」。
        # 去掉归一化后三项尺度不同（m/s、kg、N），1:1:1 并不等权。
        parts = (velocity_coef * velocity_loss,
                 force_coef * force_loss,
                 mass_coef * mass_loss)
        return parts[0] + parts[1] + parts[2], parts


def augment_actor_observation(frames, prediction):
    """Append detached estimates to form the actor input (frame_dim + decoder_dim).

    ``frames`` 是 policy 帧（维数由 ``UpperObservationSpec.frame_dim`` 定义，
    runner 已断言 decoder 的 ``frame_dim`` 与 ``env.num_obs`` 一致），``prediction``
    是 decoder 输出；这里只做「按最后一维拼接 + detach」这一件事。
    """
    if frames.shape[:-1] != prediction.shape[:-1] or prediction.shape[-1] != OUTPUT_DIM:
        raise ValueError(
            f"actor augmentation expects matching batch dims and {OUTPUT_DIM}-D estimates, "
            f"got frames {tuple(frames.shape)} and prediction {tuple(prediction.shape)}")
    return torch.cat((frames, prediction.detach()), dim=-1)


def mass_supervision_weight(
    towing_force_newtons,
    *,
    minimum_force=1.0,
    force_scale=10.0,
):
    """Map GT towing-force magnitude to a continuous mass-loss weight.

    2026-10-08 起入参是**三维**力，范数即真实张力（原先 2 维范数 = T·cosθ，偏小 2–10%）。
    """
    if minimum_force < 0.0 or force_scale <= 0.0 or towing_force_newtons.shape[-1] != FORCE_DIM:
        raise ValueError(
            f"minimum_force must be nonnegative, force_scale positive, and force shape [...,{FORCE_DIM}]")
    effective_force = (
        torch.linalg.vector_norm(towing_force_newtons, dim=-1) - minimum_force).clamp_min(0.0)
    return effective_force / (effective_force + force_scale)


def reset_gru_hidden(hidden_state, dones):
    """Clear decoder state for the environments that ended."""
    if hidden_state is not None:
        hidden_state[:, dones, :] = 0.0
    return hidden_state


class DynamicsDecoderTrainer:
    """Update the decoder on episode-consistent sequences between PPO updates."""

    def __init__(
        self,
        decoder,
        *,
        learning_rate=1.0e-3,
        max_grad_norm=1.0,
        velocity_coef=1.0,
        force_coef=1.0,
        mass_coef=1.0,
    ):
        self.decoder = decoder
        self.max_grad_norm = max_grad_norm
        self.velocity_coef = velocity_coef
        self.force_coef = force_coef
        self.mass_coef = mass_coef
        self.optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

    def update(self, frames, targets, mass_supervision_weight, dones=None, hidden_state=None):
        self.decoder.train()
        if dones is None:
            prediction, _ = self.decoder(frames.detach(), hidden_state)
        else:
            if dones.shape != frames.shape[:2]:
                raise ValueError(f"decoder dones must have shape {frames.shape[:2]}, got {dones.shape}")
            predictions = []
            state = hidden_state
            for step in range(frames.shape[0]):
                if step > 0 and state is not None:
                    keep = (~dones[step - 1].bool()).to(frames.dtype).view(1, -1, 1)
                    state = state * keep
                step_prediction, state = self.decoder(frames[step].detach(), state)
                predictions.append(step_prediction)
            prediction = torch.stack(predictions)
        loss, parts = self.decoder.loss(
            prediction,
            targets.detach(),
            mass_supervision_weight.detach(),
            velocity_coef=self.velocity_coef,
            force_coef=self.force_coef,
            mass_coef=self.mass_coef,
        )
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.decoder.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.decoder.eval()
        # 记下最近一次的三项分量，供 runner 写日志（判断权重是否配平）
        self.last_parts = tuple(float(x.detach()) for x in parts)
        return loss.detach()
