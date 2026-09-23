"""Isaac Lab vector-environment adapter for CMoE."""

from __future__ import annotations

import torch

from ..envs.cmoe_manager_based_rl_env import CMoEManagerBasedRLEnv
from ..envs.vec_env import VecEnv


class CMoEVecEnvWrapper(VecEnv):
    """Build ``[proprioceptive history, current terrain scan]`` for CMoE.

    The wrapped Isaac Lab environment must expose ``policy``, ``terrain`` and
    ``critic`` observation groups.  Only proprioception is history-stacked.
    """

    def __init__(self, env: CMoEManagerBasedRLEnv, history_steps: int = 10):
        if not isinstance(env.unwrapped, CMoEManagerBasedRLEnv):
            raise ValueError(f"Expected CMoEManagerBasedRLEnv, got {type(env.unwrapped)}")
        group_dims = env.unwrapped.observation_manager.group_obs_dim
        missing = {"policy", "terrain", "critic"}.difference(group_dims)
        if missing:
            raise ValueError(f"CMoE requires policy, terrain and critic observation groups; missing {sorted(missing)}")
        if history_steps < 1:
            raise ValueError("history_steps must include the current frame and be at least one")

        self.env = env
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length
        self.num_actions = self.unwrapped.action_manager.total_action_dim
        self.num_one_step_obs = group_dims["policy"][0]
        self.num_terrain_obs = group_dims["terrain"][0]
        self.num_privileged_obs = group_dims["critic"][0]
        self.history_steps = history_steps
        self.num_obs = history_steps * self.num_one_step_obs + self.num_terrain_obs

        self.obs_history_buf = torch.zeros(
            self.num_envs,
            history_steps,
            self.num_one_step_obs,
            device=self.device,
        )
        self.terrain_obs_buf = torch.zeros(self.num_envs, self.num_terrain_obs, device=self.device)
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device)
        self._termination_ids = torch.empty(0, dtype=torch.long, device=self.device)
        self._termination_privileged_obs = torch.empty(
            0, self.num_privileged_obs, device=self.device
        )
        self.reset()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def cfg(self):
        return self.env.cfg

    @property
    def episode_length_buf(self):
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.unwrapped.episode_length_buf = value

    def seed(self, seed: int = -1) -> int:
        return self.env.seed(seed)

    def _fill_history(self, policy_obs: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.obs_history_buf.copy_(policy_obs.unsqueeze(1).expand(-1, self.history_steps, -1))
        elif env_ids.numel() > 0:
            repeated = policy_obs[env_ids].unsqueeze(1).expand(-1, self.history_steps, -1)
            self.obs_history_buf[env_ids] = repeated

    def _combined_observations(self) -> torch.Tensor:
        return torch.cat((self.obs_history_buf.flatten(1), self.terrain_obs_buf), dim=-1)

    def reset(self):
        obs_dict, extras = self.env.reset()
        self._fill_history(obs_dict["policy"])
        self.terrain_obs_buf.copy_(obs_dict["terrain"])
        self.privileged_obs_buf.copy_(obs_dict["critic"])
        return self._combined_observations(), {"observations": obs_dict, **extras}

    def get_observations(self) -> torch.Tensor:
        return self._combined_observations()

    def get_privileged_observations(self) -> torch.Tensor:
        return self.privileged_obs_buf

    def step(self, actions: torch.Tensor):
        obs_dict, obs_before_reset, rewards, terminated, truncated, infos = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        if not self.unwrapped.cfg.is_finite_horizon:
            infos["time_outs"] = truncated

        current_policy = obs_dict["policy"]
        self.obs_history_buf[:, 1:] = self.obs_history_buf[:, :-1].clone()
        self.obs_history_buf[:, 0] = current_policy

        self._termination_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
        self._fill_history(current_policy, self._termination_ids)
        self.terrain_obs_buf.copy_(obs_dict["terrain"])
        self.privileged_obs_buf.copy_(obs_dict["critic"])
        self._termination_privileged_obs = obs_before_reset["critic"][self._termination_ids].clone()

        observations = self._combined_observations()
        if not torch.isfinite(observations).all():
            raise ValueError("NaN/Inf detected in CMoE actor observations")
        if not torch.isfinite(rewards).all():
            raise ValueError("NaN/Inf detected in rewards")
        return (
            observations,
            self.privileged_obs_buf,
            rewards,
            dones,
            infos,
            self._termination_ids,
            self._termination_privileged_obs,
        )

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)
