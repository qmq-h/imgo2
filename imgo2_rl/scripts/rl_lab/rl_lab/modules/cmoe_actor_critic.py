# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The CMoE Authors (Fudan University). All rights reserved.

"""Conditional Mixture-of-Experts actor critic from the original CMoE project."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .actor_critic import get_activation
from .cmoe_expert_actor_critic import CMoEExpertActorCritic
from .cmoe_state_estimator import CMoEStateEstimator
from .cmoe_terrain_estimator import CMoETerrainEstimator


class CMoEActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_one_step_obs: int,
        num_actions: int,
        history_steps: int = 10,
        terrain_obs_dim: int = 77,
        num_experts: int = 5,
        state_latent_dim: int = 16,
        terrain_latent_dim: int = 16,
        explicit_dim: int = 3,
        actor_hidden_dims: list[int] = [512, 256, 128],
        critic_hidden_dims: list[int] = [512, 256, 128],
        gate_hidden_dim: int = 128,
        num_prototypes: int = 32,
        projection_dim: int = 16,
        temperature: float = 0.2,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        critic_explicit_start: int | None = None,
        critic_proprio_start: int = 0,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            print(f"CMoEActorCritic ignored arguments: {list(kwargs)}")
        self.device_name = device
        self.num_actor_obs = num_actor_obs
        self.num_one_step_obs = num_one_step_obs
        self.num_actions = num_actions
        self.history_steps = history_steps
        self.history_obs_dim = history_steps * num_one_step_obs
        self.terrain_obs_dim = terrain_obs_dim
        self.terrain_obs_start = self.history_obs_dim
        self.num_experts = num_experts
        expected_actor_obs = self.history_obs_dim + terrain_obs_dim
        if num_actor_obs != expected_actor_obs:
            raise ValueError(
                f"CMoE actor observation mismatch: expected {expected_actor_obs} "
                f"({history_steps}x{num_one_step_obs}+{terrain_obs_dim}), got {num_actor_obs}."
            )

        self.state_estimator = CMoEStateEstimator(
            temporal_steps=history_steps,
            num_one_step_obs=num_one_step_obs,
            latent_dim=state_latent_dim,
            explicit_dim=explicit_dim,
            activation=activation,
            critic_explicit_start=critic_explicit_start,
            critic_proprio_start=critic_proprio_start,
        )
        self.terrain_estimator = CMoETerrainEstimator(
            terrain_obs_dim=terrain_obs_dim,
            terrain_obs_start=self.terrain_obs_start,
            latent_dim=terrain_latent_dim,
            activation=activation,
        )

        self.actor_input_dim = (
            num_one_step_obs + explicit_dim + state_latent_dim + terrain_obs_dim + terrain_latent_dim
        )
        self.experts = nn.ModuleList(
            CMoEExpertActorCritic(
                actor_input_dim=self.actor_input_dim,
                num_critic_obs=num_critic_obs,
                num_actions=num_actions,
                actor_hidden_dims=actor_hidden_dims,
                critic_hidden_dims=critic_hidden_dims,
                activation=activation,
            )
            for _ in range(num_experts)
        )
        self.gating_network = nn.Sequential(
            nn.Linear(self.actor_input_dim, gate_hidden_dim),
            get_activation(activation),
            nn.Linear(gate_hidden_dim, num_experts),
            nn.Softmax(dim=-1),
        )
        self.gate_projector = nn.Sequential(
            nn.Linear(num_experts, 128),
            get_activation(activation),
            nn.Linear(128, 64),
            get_activation(activation),
            nn.Linear(64, projection_dim),
        )
        self.terrain_projector = nn.Sequential(
            nn.Linear(terrain_obs_dim + terrain_latent_dim, 128),
            get_activation(activation),
            nn.Linear(128, 64),
            get_activation(activation),
            nn.Linear(64, projection_dim),
        )
        self.prototypes = nn.Embedding(num_prototypes, projection_dim)
        self.temperature = temperature
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution: Normal | None = None
        self.gate_weights: torch.Tensor | None = None
        Normal.set_default_validate_args = False

    def _terrain(self, observations: torch.Tensor) -> torch.Tensor:
        return observations[:, self.terrain_obs_start : self.terrain_obs_start + self.terrain_obs_dim]

    def build_actor_input(self, observations: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            explicit, state_latent = self.state_estimator(observations)
            terrain_latent = self.terrain_estimator(observations)
        return torch.cat(
            (observations[:, : self.num_one_step_obs], explicit, state_latent, self._terrain(observations), terrain_latent),
            dim=-1,
        )

    def compute_gate_weights(self, observations: torch.Tensor) -> torch.Tensor:
        actor_input = self.build_actor_input(observations)
        self.gate_weights = self.gating_network(actor_input)
        return self.gate_weights

    def update_distribution(self, observations: torch.Tensor) -> None:
        actor_input = self.build_actor_input(observations)
        self.gate_weights = self.gating_network(actor_input)
        expert_means = torch.stack([expert.act(actor_input) for expert in self.experts], dim=1)
        weighted_mean = (expert_means * self.gate_weights.unsqueeze(-1)).sum(dim=1)
        self.distribution = Normal(weighted_mean, weighted_mean * 0.0 + self.std)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations: torch.Tensor, observations_extra=None) -> torch.Tensor:
        del observations_extra
        actor_input = self.build_actor_input(observations)
        self.gate_weights = self.gating_network(actor_input)
        expert_means = torch.stack([expert.act_inference(actor_input) for expert in self.experts], dim=1)
        return (expert_means * self.gate_weights.unsqueeze(-1)).sum(dim=1)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        if self.gate_weights is None:
            raise RuntimeError("CMoE gate weights must be computed before evaluating the critics.")
        expert_values = torch.stack([expert.evaluate(critic_observations) for expert in self.experts], dim=1)
        return (expert_values * self.gate_weights.detach().unsqueeze(-1)).sum(dim=1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def update_estimators(self, obs_batch, critic_obs_batch, next_critic_obs_batch, lr):
        state_losses = self.state_estimator.update(obs_batch, critic_obs_batch, next_critic_obs_batch, lr)
        terrain_losses = self.terrain_estimator.update(obs_batch, next_critic_obs_batch, lr)
        return (*state_losses, *terrain_losses)

    def compute_contrastive_loss(self, observations: torch.Tensor) -> torch.Tensor:
        actor_input = self.build_actor_input(observations)
        gate_input = self.gating_network(actor_input.detach())
        with torch.no_grad():
            terrain_latent = self.terrain_estimator(observations)
        terrain_input = torch.cat((self._terrain(observations), terrain_latent), dim=-1).detach()
        gate_z = F.normalize(self.gate_projector(gate_input), dim=-1)
        terrain_z = F.normalize(self.terrain_projector(terrain_input), dim=-1)
        with torch.no_grad():
            self.prototypes.weight.copy_(F.normalize(self.prototypes.weight.data, dim=-1))
        gate_scores = gate_z @ self.prototypes.weight.T
        terrain_scores = terrain_z @ self.prototypes.weight.T
        with torch.no_grad():
            gate_assignments = sinkhorn(gate_scores)
            terrain_assignments = sinkhorn(terrain_scores)
        gate_log_probs = F.log_softmax(gate_scores / self.temperature, dim=-1)
        terrain_log_probs = F.log_softmax(terrain_scores / self.temperature, dim=-1)
        return -0.5 * (
            gate_assignments * terrain_log_probs + terrain_assignments * gate_log_probs
        ).mean()

    def reset(self, dones=None):
        del dones

    def forward(self):
        raise NotImplementedError


@torch.no_grad()
def sinkhorn(output: torch.Tensor, eps: float = 0.05, iterations: int = 3) -> torch.Tensor:
    assignments = torch.exp(output / eps).T
    prototypes, batch = assignments.shape
    assignments /= assignments.sum().clamp_min(1.0e-12)
    for _ in range(iterations):
        assignments /= assignments.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        assignments /= prototypes
        assignments /= assignments.sum(dim=0, keepdim=True).clamp_min(1.0e-12)
        assignments /= batch
    return (assignments * batch).T
