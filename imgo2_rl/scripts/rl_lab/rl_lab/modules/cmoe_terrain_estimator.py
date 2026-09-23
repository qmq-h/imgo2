# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The CMoE Authors (Fudan University). All rights reserved.

"""CMoE terrain auto-encoder adapted from the original implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .cmoe_state_estimator import get_activation


class CMoETerrainEstimator(nn.Module):
    def __init__(
        self,
        terrain_obs_dim: int,
        terrain_obs_start: int,
        enc_hidden_dims: list[int] = [128, 64, 32],
        dec_hidden_dims: list[int] = [32, 64, 128],
        latent_dim: int = 16,
        activation: str = "elu",
        learning_rate: float = 1.0e-3,
        max_grad_norm: float = 10.0,
        **kwargs,
    ):
        super().__init__()
        self.use_latent_loss = kwargs.pop("use_latent_loss", True)
        if kwargs:
            print(f"CMoETerrainEstimator ignored arguments: {list(kwargs)}")
        self.terrain_obs_dim = terrain_obs_dim
        self.terrain_obs_start = terrain_obs_start
        self.latent_dim = latent_dim
        self.max_grad_norm = max_grad_norm
        activation_type = type(get_activation(activation))

        encoder_layers: list[nn.Module] = []
        input_dim = terrain_obs_dim
        for hidden_dim in enc_hidden_dims:
            encoder_layers.extend((nn.Linear(input_dim, hidden_dim), activation_type()))
            input_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(input_dim, latent_dim)

        decoder_layers: list[nn.Module] = []
        input_dim = latent_dim
        for hidden_dim in dec_hidden_dims:
            decoder_layers.extend((nn.Linear(input_dim, hidden_dim), activation_type()))
            input_dim = hidden_dim
        decoder_layers.append(nn.Linear(input_dim, terrain_obs_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def terrain_observations(self, observations: torch.Tensor) -> torch.Tensor:
        end = self.terrain_obs_start + self.terrain_obs_dim
        if observations.shape[-1] < end:
            raise ValueError(f"Expected at least {end} actor observations, got {observations.shape[-1]}.")
        return observations[:, self.terrain_obs_start : end]

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        return self.fc_mu(self.encoder(self.terrain_observations(observations).detach()))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.encode(observations).detach()

    def update(
        self,
        observations: torch.Tensor,
        next_critic_observations: torch.Tensor | None = None,
        lr: float | None = None,
    ) -> tuple[float, float, float, float]:
        del next_critic_observations
        if lr is not None:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        target = self.terrain_observations(observations).detach()
        reconstruction = self.decoder(self.encode(observations))
        reconstruction_loss = F.mse_loss(reconstruction, target)
        loss = self.use_latent_loss * reconstruction_loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()
        value = reconstruction_loss.item()
        return 0.0, value, value, 0.0
