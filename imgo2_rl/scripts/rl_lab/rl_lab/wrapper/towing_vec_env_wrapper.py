from __future__ import annotations

import torch

from rl_lab.envs import VecEnv


def _flat_dim(group_dim):
    if isinstance(group_dim, int):
        return group_dim
    result = 1
    for value in group_dim:
        result *= value
    return result


class TowingVecEnvWrapper(VecEnv):
    """Expose policy, critic and decoder groups without depending on RSL-RL wrappers."""

    def __init__(self, env, clip_actions=None):
        if not hasattr(env.unwrapped, "observation_manager") or not hasattr(env.unwrapped, "action_manager"):
            raise ValueError("Towing environment must provide observation_manager and action_manager")
        groups = env.unwrapped.observation_manager.group_obs_dim
        missing = {"policy", "critic", "decoder"} - set(groups)
        if missing:
            raise ValueError(f"Towing environment is missing observation groups: {sorted(missing)}")
        self.env = env
        self.clip_actions = clip_actions
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device
        self.max_episode_length = env.unwrapped.max_episode_length
        self.num_actions = env.unwrapped.action_manager.total_action_dim
        self.num_obs = _flat_dim(groups["policy"])
        self.num_privileged_obs = _flat_dim(groups["critic"])
        self.num_decoder_obs = _flat_dim(groups["decoder"])
        if self.num_obs != 51 or self.num_decoder_obs != 6:
            raise ValueError(
                f"Towing contract requires policy=51 and decoder=6, got {self.num_obs} and {self.num_decoder_obs}")
        self._obs_dict = None
        self.reset()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def episode_length_buf(self):
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.unwrapped.episode_length_buf = value

    def reset(self):
        self._obs_dict, _ = self.env.reset()
        return self.get_observations(), self.get_privileged_observations()

    def get_observations(self):
        return self._obs_dict["policy"]

    def get_privileged_observations(self):
        return self._obs_dict["critic"]

    def get_decoder_supervision(self):
        decoder = self._obs_dict["decoder"]
        return decoder[:, :5], decoder[:, 5]

    def step(self, actions):
        if self.clip_actions is not None:
            actions = actions.clamp(-self.clip_actions, self.clip_actions)
        self._obs_dict, rewards, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        return (
            self.get_observations(), self.get_privileged_observations(),
            rewards, dones, extras,
        )

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        return getattr(self.unwrapped, name)

