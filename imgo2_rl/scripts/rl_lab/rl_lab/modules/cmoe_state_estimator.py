# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The CMoE Authors (Fudan University). All rights reserved.

"""CMoE proprioceptive state estimator adapted from the original implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CMoEStateEstimator(nn.Module):
    def __init__(
        self,
        temporal_steps: int,
        num_one_step_obs: int,
        prop_enc_hidden_dims: list[int] = [128, 64, 32],
        dec_hidden_dims: list[int] = [32, 64, 128],
        latent_dim: int = 16,
        explicit_dim: int = 3,
        activation: str = "elu",
        learning_rate: float = 1.0e-3,
        max_grad_norm: float = 10.0,
        kld_weight: float = 0.005,
        critic_explicit_start: int | None = None,
        critic_proprio_start: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.use_estimation_loss = kwargs.pop("use_estimation_loss", True)
        self.use_latent_loss = kwargs.pop("use_latent_loss", True)
        if kwargs:
            print(f"CMoEStateEstimator ignored arguments: {list(kwargs)}")

        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.num_prop_obs = temporal_steps * num_one_step_obs
        self.latent_dim = latent_dim
        self.explicit_dim = explicit_dim
        self.critic_explicit_start = num_one_step_obs if critic_explicit_start is None else critic_explicit_start
        self.critic_proprio_start = critic_proprio_start
        self.max_grad_norm = max_grad_norm
        self.kld_weight = kld_weight
        activation_type = type(get_activation(activation))

        encoder_layers: list[nn.Module] = []
        input_dim = self.num_prop_obs
        for hidden_dim in prop_enc_hidden_dims:
            encoder_layers.extend((nn.Linear(input_dim, hidden_dim), activation_type()))
            input_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_var = nn.Linear(input_dim, latent_dim)
        self.fc_explicit = nn.Linear(input_dim, explicit_dim)

        decoder_layers: list[nn.Module] = []
        input_dim = latent_dim + explicit_dim
        for hidden_dim in dec_hidden_dims:
            decoder_layers.extend((nn.Linear(input_dim, hidden_dim), activation_type()))
            input_dim = hidden_dim
        decoder_layers.append(nn.Linear(input_dim, num_one_step_obs))
        self.decoder = nn.Sequential(*decoder_layers)

        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def _proprio_history(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-1] < self.num_prop_obs:
            raise ValueError(
                f"Expected at least {self.num_prop_obs} actor observations, got {observations.shape[-1]}."
            )
        return observations[:, : self.num_prop_obs]

    def encode(self, observations: torch.Tensor):
        features = self.encoder(self._proprio_history(observations).detach())
        mu = self.fc_mu(features)
        log_var = self.fc_var(features)
        explicit = self.fc_explicit(features)
        latent = self.reparameterize(mu, log_var) if self.training else mu
        return explicit, latent, mu, log_var

    def forward(self, observations: torch.Tensor):
        explicit, latent, _, _ = self.encode(observations)
        return explicit.detach(), latent.detach()

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        return mu + torch.rand_like(std) * std

    def update(
        self,
        observations: torch.Tensor,
        critic_observations: torch.Tensor,
        next_critic_observations: torch.Tensor,
        lr: float | None = None,
    ) -> tuple[float, float, float, float]:
        if lr is not None:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

        explicit_target = critic_observations[
            :, self.critic_explicit_start : self.critic_explicit_start + self.explicit_dim
        ].detach()
        next_proprio_target = next_critic_observations[
            :, self.critic_proprio_start : self.critic_proprio_start + self.num_one_step_obs
        ].detach()

        explicit, latent, mu, log_var = self.encode(observations)
        predicted_next_proprio = self.decoder(torch.cat((latent, explicit), dim=-1))
        reconstruction_loss = F.mse_loss(predicted_next_proprio, next_proprio_target)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu.square() - log_var.exp(), dim=-1))
        latent_loss = reconstruction_loss + self.kld_weight * kld_loss
        estimation_loss = F.mse_loss(explicit, explicit_target)
        loss = self.use_estimation_loss * estimation_loss + self.use_latent_loss * latent_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return estimation_loss.item(), latent_loss.item(), reconstruction_loss.item(), kld_loss.item()


def get_activation(name: str) -> nn.Module:
    activations = {
        "elu": nn.ELU,
        "selu": nn.SELU,
        "relu": nn.ReLU,
        "crelu": nn.ReLU,
        "silu": nn.SiLU,
        "lrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[name]()
