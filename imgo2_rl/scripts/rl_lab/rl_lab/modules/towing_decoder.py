"""Recurrent dynamics decoder used by the upper towing policy."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TowingDynamicsDecoder(nn.Module):
    """Estimate robot velocity, load mass and body-frame towing force."""

    def __init__(self, frame_dim=51, feature_dim=128, hidden_dim=128, num_layers=1, force_scale=10.0):
        super().__init__()
        self.frame_dim = frame_dim
        self.force_scale = force_scale
        self.encoder = nn.Sequential(nn.Linear(frame_dim, feature_dim), nn.ELU())
        self.gru = nn.GRU(feature_dim, hidden_dim, num_layers=num_layers)
        self.velocity_head = nn.Linear(hidden_dim, 2)
        self.mass_head = nn.Linear(hidden_dim, 1)
        self.force_head = nn.Linear(hidden_dim, 2)

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
        prediction = torch.cat((
            torch.tanh(self.velocity_head(features)),
            torch.tanh(self.mass_head(features)),
            torch.tanh(self.force_head(features)),
        ), dim=-1)
        return (prediction.squeeze(0) if single_step else prediction), hidden_state

    def force_newtons(self, prediction):
        normalized = prediction[..., 3:5].clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
        return self.force_scale * normalized / (1.0 - normalized.abs())

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
        if prediction.shape != targets.shape or prediction.shape[-1] != 5:
            raise ValueError(
                f"decoder shape mismatch: prediction={prediction.shape}, target={targets.shape}")
        weight = mass_supervision_weight.to(
            device=prediction.device, dtype=prediction.dtype).reshape(prediction.shape[:-1])
        velocity_loss = F.smooth_l1_loss(prediction[..., :2], targets[..., :2])
        force_loss = F.smooth_l1_loss(prediction[..., 3:5], targets[..., 3:5])
        mass_error = F.smooth_l1_loss(
            prediction[..., 2], targets[..., 2], reduction="none")
        mass_loss = (mass_error * weight).sum() / weight.sum().clamp_min(1.0)
        return velocity_coef * velocity_loss + force_coef * force_loss + mass_coef * mass_loss


def augment_actor_observation(frames, prediction):
    """Append detached estimates to form the 56-D actor input."""
    if frames.shape[:-1] != prediction.shape[:-1] or frames.shape[-1] != 51 or prediction.shape[-1] != 5:
        raise ValueError(
            f"actor augmentation expects [...,51] and [...,5], got {frames.shape} and {prediction.shape}")
    return torch.cat((frames, prediction.detach()), dim=-1)


def mass_supervision_weight(
    towing_force_newtons,
    *,
    minimum_force=1.0,
    force_scale=10.0,
):
    """Map GT towing-force magnitude to a continuous mass-loss weight."""
    if minimum_force < 0.0 or force_scale <= 0.0 or towing_force_newtons.shape[-1] != 2:
        raise ValueError(
            "minimum_force must be nonnegative, force_scale positive, and force shape [...,2]")
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

    def __init__(self, decoder, *, learning_rate=1.0e-3, max_grad_norm=1.0):
        self.decoder = decoder
        self.max_grad_norm = max_grad_norm
        self.optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

    def update(self, frames, targets, mass_supervision_weight, hidden_state=None):
        self.decoder.train()
        prediction, _ = self.decoder(frames.detach(), hidden_state)
        loss = self.decoder.loss(
            prediction, targets.detach(), mass_supervision_weight.detach())
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.decoder.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.decoder.eval()
        return loss.detach()
