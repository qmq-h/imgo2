# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The CMoE Authors (Fudan University).

"""One actor/critic expert used by CMoE."""

from __future__ import annotations

import torch
import torch.nn as nn

from .actor_critic import get_activation


def _build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(input_dim, hidden_dim), get_activation(activation)))
        input_dim = hidden_dim
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class CMoEExpertActorCritic(nn.Module):
    """Independent actor and critic pair; action noise is shared by the CMoE policy."""

    def __init__(
        self,
        actor_input_dim: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] = [512, 256, 128],
        critic_hidden_dims: list[int] = [512, 256, 128],
        activation: str = "elu",
    ):
        super().__init__()
        self.actor = _build_mlp(actor_input_dim, actor_hidden_dims, num_actions, activation)
        self.critic = _build_mlp(num_critic_obs, critic_hidden_dims, 1, activation)

    def act(self, actor_input: torch.Tensor) -> torch.Tensor:
        return self.actor(actor_input)

    def act_inference(self, actor_input: torch.Tensor) -> torch.Tensor:
        return self.actor(actor_input)

    def evaluate(self, critic_observations: torch.Tensor) -> torch.Tensor:
        return self.critic(critic_observations)
