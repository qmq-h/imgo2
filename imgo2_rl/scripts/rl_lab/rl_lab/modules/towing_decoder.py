"""Training-only cart-mass decoder and identification reward.

The decoder is deliberately separate from the actor. During rollout its parameters are frozen
and its detached prediction score may be added to the PPO reward. Decoder optimization happens
between rollouts from privileged simulation labels. No decoder output is a policy observation or
part of the deployable policy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TowingMassDecoder(nn.Module):
    """Predict normalized cart mass in ``[-1, 1]`` from two 48-D actor frames."""

    def __init__(self, history_dim=96, hidden_dims=(256, 128, 64)):
        super().__init__()
        layers = []
        input_dim = history_dim
        for output_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, output_dim), nn.ELU()))
            input_dim = output_dim
        layers.extend((nn.Linear(input_dim, 1), nn.Tanh()))
        self.network = nn.Sequential(*layers)

    def forward(self, observation_history):
        if observation_history.shape[-1] != self.network[0].in_features:
            raise ValueError(
                f"history dim {observation_history.shape[-1]} != "
                f"{self.network[0].in_features}")
        return self.network(observation_history)

    @staticmethod
    def loss(prediction, normalized_mass):
        """Supervised decoder loss; both tensors have shape ``[..., 1]``."""
        if prediction.shape != normalized_mass.shape or prediction.shape[-1] != 1:
            raise ValueError(
                f"mass decoder shape mismatch: prediction={prediction.shape}, "
                f"target={normalized_mass.shape}")
        return F.smooth_l1_loss(prediction, normalized_mass)


@torch.no_grad()
def mass_identification_reward(
    decoder,
    observation_history,
    normalized_mass,
    active_mask,
    *,
    temperature=0.25,
):
    """Return a detached dense score for active towing steps.

    ``active_mask`` should be true only while a non-zero tow command is active and the rope is
    taut. Weighting is intentionally left to PPO configuration so this helper stays unitless.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    prediction = decoder(observation_history)
    if prediction.shape != normalized_mass.shape or prediction.shape[-1] != 1:
        raise ValueError(
            f"mass reward shape mismatch: prediction={prediction.shape}, "
            f"target={normalized_mass.shape}")
    error = F.smooth_l1_loss(prediction, normalized_mass, reduction="none").squeeze(-1)
    mask = active_mask.to(device=error.device, dtype=error.dtype).reshape(error.shape)
    return torch.exp(-error / temperature) * mask


class MassDecoderTrainer:
    """Small optimizer wrapper used only between PPO rollouts."""

    def __init__(self, decoder, *, learning_rate=1.0e-3, max_grad_norm=1.0):
        self.decoder = decoder
        self.max_grad_norm = max_grad_norm
        self.optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

    def update(self, observation_history, normalized_mass):
        self.decoder.train()
        prediction = self.decoder(observation_history.detach())
        loss = self.decoder.loss(prediction, normalized_mass.detach())
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.decoder.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.decoder.eval()
        return loss.detach()
