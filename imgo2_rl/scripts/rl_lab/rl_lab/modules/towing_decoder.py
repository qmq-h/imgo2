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
    """Supervised variational bottleneck: history -> z -> explicit six-vector.

    Latent coordinates have no prescribed physical meanings. Unlike CMoE's
    next-proprioception reconstruction, this task reconstructs the privileged
    [vx, vy, mass, Fx, Fy, Fz] target through the latent bottleneck.
    """

    # 默认帧维 = 当前契约 v3 的 policy 帧（58）。调用方（runner / 运行时）一律显式传
    # 配置值，这里只是防止"不传参构造"时悄悄退回旧 57 维契约。
    def __init__(self, frame_dim=58, feature_dim=128, hidden_dim=128,
                 num_layers=1, force_scale=10.0, latent_dim=16):
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        self.frame_dim = frame_dim
        self.force_scale = force_scale
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(frame_dim, feature_dim), nn.ELU())
        self.gru = nn.GRU(feature_dim, hidden_dim, num_layers=num_layers)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        self.latent_decoder = nn.Sequential(nn.Linear(latent_dim, feature_dim), nn.ELU())
        self.velocity_head = nn.Linear(feature_dim, VELOCITY_DIM)
        self.mass_head = nn.Linear(feature_dim, MASS_DIM)
        self.force_head = nn.Linear(feature_dim, FORCE_DIM)

    @property
    def output_dim(self):
        return OUTPUT_DIM

    @property
    def actor_feature_dim(self):
        return self.output_dim + self.latent_dim

    @staticmethod
    def reparameterize(mu, log_var):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)

    @staticmethod
    def kl_loss(mu, log_var):
        return (-0.5 * (1 + log_var - mu.square() - log_var.exp()).sum(dim=-1)).mean()

    def encode(self, frames, hidden_state=None):
        single_step = frames.ndim == 2
        if single_step:
            frames = frames.unsqueeze(0)
        if frames.ndim != 3 or frames.shape[-1] != self.frame_dim:
            raise ValueError(f"decoder expects [T,B,{self.frame_dim}] or [B,{self.frame_dim}]")
        features, hidden_state = self.gru(self.encoder(frames), hidden_state)
        mu = self.fc_mu(features)
        # Bound exp(log_var) to keep sampling and KL finite at startup.
        log_var = self.fc_log_var(features).clamp(-10.0, 4.0)
        if single_step:
            mu, log_var = mu.squeeze(0), log_var.squeeze(0)
        return mu, log_var, hidden_state

    def decode(self, latent):
        features = self.latent_decoder(latent)
        return torch.cat((self.velocity_head(features), self.mass_head(features),
                          self.force_head(features)), dim=-1)

    def forward_with_latent(self, frames, hidden_state=None, *, sample=None):
        mu, log_var, state = self.encode(frames, hidden_state)
        if sample is None:
            sample = self.training
        latent = self.reparameterize(mu, log_var) if sample else mu
        return self.decode(latent), latent, state

    def forward(self, frames, hidden_state=None):
        prediction, _, state = self.forward_with_latent(frames, hidden_state)
        return prediction, state

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


def augment_actor_observation(frames, prediction, latent):
    """Actor receives detached explicit estimates and latent; PPO cannot train VAE."""
    if (frames.shape[:-1] != prediction.shape[:-1]
            or frames.shape[:-1] != latent.shape[:-1]
            or prediction.shape[-1] != OUTPUT_DIM):
        raise ValueError("actor augmentation batch or explicit dimension mismatch")
    return torch.cat((frames, prediction.detach(), latent.detach()), dim=-1)


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
        kld_weight=0.005,
    ):
        self.decoder = decoder
        self.max_grad_norm = max_grad_norm
        self.velocity_coef = velocity_coef
        self.force_coef = force_coef
        self.mass_coef = mass_coef
        if kld_weight < 0:
            raise ValueError("kld_weight must be nonnegative")
        self.kld_weight = kld_weight
        self.optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

    def update(self, frames, targets, mass_supervision_weight, dones=None, hidden_state=None):
        self.decoder.train()
        state = hidden_state.detach().clone() if hidden_state is not None else None
        if dones is None:
            mu, log_var, _ = self.decoder.encode(frames.detach(), state)
        else:
            if dones.shape != frames.shape[:2]:
                raise ValueError(f"decoder dones must have shape {frames.shape[:2]}, got {dones.shape}")
            mus, log_vars = [], []
            for step in range(frames.shape[0]):
                if step > 0 and state is not None:
                    keep = (~dones[step - 1].bool()).to(frames.dtype).view(1, -1, 1)
                    state = state * keep
                step_mu, step_log_var, state = self.decoder.encode(frames[step].detach(), state)
                mus.append(step_mu)
                log_vars.append(step_log_var)
            mu, log_var = torch.stack(mus), torch.stack(log_vars)
        latent = self.decoder.reparameterize(mu, log_var)
        prediction = self.decoder.decode(latent)
        loss, parts = self.decoder.loss(
            prediction,
            targets.detach(),
            mass_supervision_weight.detach(),
            velocity_coef=self.velocity_coef,
            force_coef=self.force_coef,
            mass_coef=self.mass_coef,
        )
        kl = self.decoder.kl_loss(mu, log_var)
        loss = loss + self.kld_weight * kl
        self.last_kl = float(kl.detach())
        self.last_weighted_kl = float((self.kld_weight * kl).detach())
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.decoder.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.decoder.eval()
        # 记下最近一次的三项分量，供 runner 写日志（判断权重是否配平）
        self.last_parts = tuple(float(x.detach()) for x in parts)
        return loss.detach()
