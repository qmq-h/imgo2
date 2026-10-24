"""CE Net: recurrent context encoder for the upper towing policy.

2026-10-17 结构（用户要求「GRU 并列输出显式动力学估计与变分隐向量，两者共同重建下一时刻
的 58 维观测」；2026-10-16 的「去质量」保留 ⇒ 显式 **5** 维）：

    o_t (58) ─ encoder(58→128)+GRU(128) ─┬─ velocity_head(128→2) ┐
                                          ├─ force_head(128→3)    ┴ 显式 5 维（有监督）
                                          ├─ fc_mu / fc_log_var → z(16)  （KL 只在这里）
                                          └─ cat(显式5, z16) = 21 → MLP → ô_{t+1} (58)
    actor 输入 = 58 + 5 + 16 = **79**（后两者 detach，PPO 不训练 CE Net）

显式量仍然**不经过变分瓶颈**（直接读 GRU hidden），所以质量不再由 decoder 预测、只留给
latent 与重建分支（2026-10-16 用户准则「decoder 只建模与四足机器人直接相关的量」）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# 显式输出布局（2026-10-16 起 **5 维**）：[vx, vy, Fx, Fy, Fz]
# - **质量不显式预测**（2026-10-16 用户准则）：机器人自身速度（2 维）与作用在机器人上的
#   牵引力（3 维）才是"与四足机器人直接相关的量"；质量是负载属性，交给 latent 与重建分支。
# - 机器人速度保持 **2 维**（机体系水平）：`root_lin_vel_b[:, 2]` 在 trot 步态下按步频
#   大幅振荡，不是负载量，而上层也没有垂直控制通道；
# - 牵引力是 **3 维**：两挂点高差 0.17 m，绷紧时方向向量 z 分量 = 0.17/L0，在
#   L0 = 0.4…0.8 m 上就是张力的 21–43%（\|Fz\|/T），且 Fz 在 0.16 m 后置挂点上产生俯仰
#   力矩。丢掉它会让「绳子往下拽」这件事对 actor 完全不可见。
# 布局常量集中在这里，head 宽度、loss 切片与 `force_newtons` 全部由它们推导，
# 避免再出现「改了 head 忘了切片」的静默错位。
VELOCITY_DIM = 2
FORCE_DIM = 3
OUTPUT_DIM = VELOCITY_DIM + FORCE_DIM
FORCE_SLICE = slice(VELOCITY_DIM, OUTPUT_DIM)


class TowingDynamicsDecoder(nn.Module):
    """CE Net: history -> explicit state + variational latent -> next frame.

    显式 5 维有特权监督；16 维 latent 只受高斯先验（KL）与下一帧重建约束。
    两路拼接（21 维）共同重建 ``o[t+1]``；推理只前向编码分支，不调用重建头。
    """

    # 默认帧维 = 当前 v6 契约的 policy 帧（57）。调用方（runner / 运行时）一律显式传配置值，
    # 这里只是防止"不传参构造"时悄悄退回旧 57 维契约。
    def __init__(self, frame_dim=57, feature_dim=128, hidden_dim=128,
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
        self.velocity_head = nn.Linear(hidden_dim, VELOCITY_DIM)
        self.force_head = nn.Linear(hidden_dim, FORCE_DIM)
        self.latent_decoder = nn.Sequential(
            nn.Linear(OUTPUT_DIM + latent_dim, feature_dim), nn.ELU(),
            nn.Linear(feature_dim, frame_dim))

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
        prediction = torch.cat((self.velocity_head(features),
                                self.force_head(features)), dim=-1)
        mu = self.fc_mu(features)
        # Bound exp(log_var) to keep sampling and KL finite at startup.
        log_var = self.fc_log_var(features).clamp(-10.0, 4.0)
        if single_step:
            prediction = prediction.squeeze(0)
            mu, log_var = mu.squeeze(0), log_var.squeeze(0)
        return prediction, mu, log_var, hidden_state

    def decode(self, prediction, latent):
        """从拼接的「显式 5 + latent 16」上下文码重建下一帧（58 维）。"""
        return self.latent_decoder(torch.cat((prediction, latent), dim=-1))

    def forward_with_latent(self, frames, hidden_state=None, *, sample=None):
        prediction, mu, log_var, state = self.encode(frames, hidden_state)
        if sample is None:
            sample = self.training
        latent = self.reparameterize(mu, log_var) if sample else mu
        return prediction, latent, state

    def forward(self, frames, hidden_state=None):
        prediction, _, state = self.forward_with_latent(frames, hidden_state)
        return prediction, state

    def force_newtons(self, prediction):
        """取 force head 的输出。去掉 tanh 与归一化后它本身就是牛顿，故为恒等。

        保留此入口是为了让调用方语义不变（部署端与 play 都走这里）。
        """
        return prediction[..., FORCE_SLICE]

    @staticmethod
    def loss(prediction, targets, *, velocity_coef=1.0, force_coef=1.0):
        """Supervise the explicit five-vector（2026-10-16 起没有质量项）。

        两项加权后的实际贡献单独返回，供日志核对「权重是否真的配平了」。
        去掉 target 归一化后两项尺度不同（m/s、N），1:1 并不等权。
        """
        if prediction.shape != targets.shape or prediction.shape[-1] != OUTPUT_DIM:
            raise ValueError(
                f"decoder shape mismatch: prediction={prediction.shape}, target={targets.shape}")
        velocity_loss = F.smooth_l1_loss(
            prediction[..., :VELOCITY_DIM], targets[..., :VELOCITY_DIM])
        force_loss = F.smooth_l1_loss(
            prediction[..., FORCE_SLICE], targets[..., FORCE_SLICE])
        parts = (velocity_coef * velocity_loss, force_coef * force_loss)
        return parts[0] + parts[1], parts


def augment_actor_observation(frames, prediction, latent):
    """Actor receives detached explicit estimates and latent; PPO cannot train CE Net."""
    if (frames.shape[:-1] != prediction.shape[:-1]
            or frames.shape[:-1] != latent.shape[:-1]
            or prediction.shape[-1] != OUTPUT_DIM):
        raise ValueError("actor augmentation batch or explicit dimension mismatch")
    return torch.cat((frames, prediction.detach(), latent.detach()), dim=-1)


def reset_gru_hidden(hidden_state, dones):
    """Clear decoder state for the environments that ended."""
    if hidden_state is not None:
        hidden_state[:, dones, :] = 0.0
    return hidden_state


class DynamicsDecoderTrainer:
    """Update the CE Net on episode-consistent sequences between PPO updates."""

    def __init__(
        self,
        decoder,
        *,
        learning_rate=1.0e-3,
        max_grad_norm=1.0,
        velocity_coef=1.0,
        force_coef=1.0,
        kld_weight=0.005,
        reconstruction_coef=1.0,
        reconstruction_mask=None,
    ):
        self.decoder = decoder
        self.max_grad_norm = max_grad_norm
        self.velocity_coef = velocity_coef
        self.force_coef = force_coef
        if kld_weight < 0:
            raise ValueError("kld_weight must be nonnegative")
        self.kld_weight = kld_weight
        if reconstruction_coef < 0:
            raise ValueError("reconstruction_coef must be nonnegative")
        self.reconstruction_coef = reconstruction_coef
        #: 逐维重建掩码（`True` = 参与 MSE）。`None` = 全维（旧行为）。
        #: 2026-10-17 起由 `UpperObservationSpec.reconstruction_mask()` 给出，
        #: 排除 `loco_command`(3) + `last_action`(13) —— 它们在 `o[t+1]` 里由策略下一拍
        #: 自己的动作决定，decoder 看不到（见 `docs/towing/towing_ce_net_2026-10-17.md` §5①）。
        if reconstruction_mask is None:
            self.reconstruction_mask = None
        else:
            mask = torch.as_tensor(tuple(bool(flag) for flag in reconstruction_mask),
                                   dtype=torch.bool).flatten()
            if mask.numel() != int(getattr(decoder, "frame_dim", mask.numel())):
                raise ValueError(
                    f"reconstruction_mask 有 {mask.numel()} 维，但 decoder.frame_dim="
                    f"{getattr(decoder, 'frame_dim', None)}")
            if not bool(mask.any()):
                raise ValueError("reconstruction_mask 全为 False ⇒ 重建损失恒为 0，无意义")
            self.reconstruction_mask = mask
        self.optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

    def update(self, frames, targets, *, dones=None, hidden_state=None, next_frames):
        """`dones` / `hidden_state` / `next_frames` 全是**关键字专用**（2026-10-16 起）：
        去掉质量权重后若保留位置参数，旧调用 `update(frames, targets, weight)` 会被静默
        当成 `dones` 接受，把「监督权重」当回合终止标志（非常难查的错位）。"""
        if frames.ndim != 3 or next_frames.shape != frames.shape:
            raise ValueError("frames and next_frames must have matching [T,B,D] shapes")
        self.decoder.train()
        state = hidden_state.detach().clone() if hidden_state is not None else None
        if dones is None:
            prediction, mu, log_var, _ = self.decoder.encode(frames.detach(), state)
        else:
            if dones.shape != frames.shape[:2]:
                raise ValueError(f"decoder dones must have shape {frames.shape[:2]}, got {dones.shape}")
            predictions, mus, log_vars = [], [], []
            for step in range(frames.shape[0]):
                if step > 0 and state is not None:
                    keep = (~dones[step - 1].bool()).to(frames.dtype).view(1, -1, 1)
                    state = state * keep
                step_prediction, step_mu, step_log_var, state = self.decoder.encode(
                    frames[step].detach(), state)
                predictions.append(step_prediction)
                mus.append(step_mu)
                log_vars.append(step_log_var)
            mu, log_var = torch.stack(mus), torch.stack(log_vars)
            prediction = torch.stack(predictions)
        latent = self.decoder.reparameterize(mu, log_var)
        estimate_loss, parts = self.decoder.loss(
            prediction,
            targets.detach(),
            velocity_coef=self.velocity_coef,
            force_coef=self.force_coef,
        )
        kl = self.decoder.kl_loss(mu, log_var)
        reconstruction = self.decoder.decode(prediction, latent)
        # Auto-reset observations belong to the NEXT episode. Exclude both
        # terminations and timeouts from reconstruction; retain current-state
        # supervision and KL, and reset GRU before encoding the next step.
        valid = (torch.ones(frames.shape[:2], dtype=torch.bool, device=frames.device)
                 if dones is None else ~dones.bool())
        if valid.any():
            prediction_next = reconstruction[valid]
            target_next = next_frames.detach()[valid]
            if self.reconstruction_mask is None:
                reconstruction_loss = F.mse_loss(prediction_next, target_next)
            else:
                mask = self.reconstruction_mask.to(prediction_next.device)
                if mask.numel() != prediction_next.shape[-1]:
                    raise ValueError(
                        f"reconstruction_mask 有 {mask.numel()} 维，但重建输出是 "
                        f"{prediction_next.shape[-1]} 维")
                reconstruction_loss = F.mse_loss(prediction_next[:, mask],
                                                 target_next[:, mask])
        else:
            reconstruction_loss = reconstruction.sum() * 0.0
        loss = (estimate_loss + self.reconstruction_coef * reconstruction_loss
                + self.kld_weight * kl)
        self.last_reconstruction = float(reconstruction_loss.detach())
        self.last_weighted_reconstruction = float(
            (self.reconstruction_coef * reconstruction_loss).detach())
        self.last_kl = float(kl.detach())
        self.last_weighted_kl = float((self.kld_weight * kl).detach())
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.decoder.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.decoder.eval()
        # 记下最近一次的两项分量，供 runner 写日志（判断权重是否配平）
        self.last_parts = tuple(float(x.detach()) for x in parts)
        return loss.detach()
