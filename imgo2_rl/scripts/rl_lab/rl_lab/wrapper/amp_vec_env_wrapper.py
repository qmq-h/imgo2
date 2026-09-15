from __future__ import annotations

import torch

from rl_lab.envs import AmpManagerBasedRLEnv
from rl_lab.envs import VecEnv


class AmpVecEnvWrapper(VecEnv):
    """Wraps an AmpManagerBasedRLEnv into the AMP RSL-RL VecEnv interface."""

    def __init__(
        self,
        env: AmpManagerBasedRLEnv,
        include_history_steps: int | None = None,
        clip_actions: float | None = None,
    ):
        if not isinstance(env.unwrapped, AmpManagerBasedRLEnv):
            raise ValueError(
                "The environment must be inherited from AmpManagerBasedRLEnv. "
                f"Environment type: {type(env)}"
            )
        if not hasattr(env.unwrapped, "observation_manager"):
            raise ValueError("The environment must have an observation_manager.")
        if not hasattr(env.unwrapped, "action_manager"):
            raise ValueError("The environment must have an action_manager.")

        self.env = env
        self.clip_actions = clip_actions
        self.include_history_steps = include_history_steps

        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length
        self.num_actions = self.unwrapped.action_manager.total_action_dim
        self.num_obs = self.unwrapped.observation_manager.group_obs_dim["policy"][0]
        if "amp" not in self.unwrapped.observation_manager.group_obs_dim:
            raise ValueError("AMP environments must define an 'amp' observation group.")
        self.num_amp_obs = self.unwrapped.observation_manager.group_obs_dim["amp"][0]
        self.num_privileged_obs = (
            self.unwrapped.observation_manager.group_obs_dim["critic"][0]
            if "critic" in self.unwrapped.observation_manager.group_obs_dim
            else None
        )

        if self.include_history_steps is not None:
            self.obs_history_buf = torch.zeros(
                self.num_envs, self.num_obs * self.include_history_steps, device=self.device
            )
        else:
            self.obs_history_buf = None
        self.amp_obs_buf = torch.zeros(self.num_envs, self.num_amp_obs, device=self.device)

        self.reset()

    def __str__(self):
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        return str(self)

    @property
    def cfg(self) -> object:
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        return self.env.render_mode

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        self.unwrapped.episode_length_buf = value

    @property
    def dof_pos_limits(self) -> torch.Tensor:
        return self.unwrapped.dof_pos_limits

    @property
    def dt(self) -> float:
        return self.unwrapped.dt

    def seed(self, seed: int = -1) -> int:
        return self.unwrapped.seed(seed)

    def reset(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        obs_dict, _ = self.env.reset()
        policy_obs = obs_dict["policy"]
        self.amp_obs_buf = obs_dict["amp"]
        if self.include_history_steps is not None:
            self.obs_history_buf = policy_obs.repeat(1, self.include_history_steps)
            policy_obs = self.obs_history_buf
        privileged_obs = obs_dict["critic"] if "critic" in obs_dict else None
        return policy_obs, privileged_obs

    def get_observations(self) -> torch.Tensor:
        if self.include_history_steps is not None:
            return self.obs_history_buf
        return self.unwrapped.obs_buf["policy"]

    def get_privileged_observations(self) -> torch.Tensor | None:
        if "critic" in self.unwrapped.obs_buf:
            return self.unwrapped.obs_buf["critic"]
        return None

    def get_amp_observations(self) -> torch.Tensor:
        return self.amp_obs_buf

    def step(
        self, actions: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict,
        torch.Tensor,
        torch.Tensor,
    ]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        obs_dict, rewards, terminated, truncated, extras, reset_env_ids, terminal_amp_states = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)

        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        policy_obs = obs_dict["policy"]
        if self.include_history_steps is not None:
            self.obs_history_buf = torch.cat(
                (policy_obs[:, : self.num_obs], self.obs_history_buf[:, : -self.num_obs]), dim=-1
            )
            policy_obs = self.obs_history_buf

        privileged_obs = obs_dict["critic"] if "critic" in obs_dict else None
        amp_obs = obs_dict["amp"]
        self.amp_obs_buf = amp_obs
        return policy_obs, privileged_obs, amp_obs, rewards, dones, extras, reset_env_ids, terminal_amp_states

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        return getattr(self.unwrapped, name)
