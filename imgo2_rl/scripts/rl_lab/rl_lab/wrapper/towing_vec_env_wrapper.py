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
        # 契约字面量（与 `upper_logic` 的 UpperObservationSpec / DecoderSpec 对应）：
        # policy 帧 57 = loco_command 3 + last_action 12 + ang_vel 3 + gravity 3 +
        #                 last_loco_action 12 + joint_pos 12 + joint_vel 12；
        # decoder 组 7 = targets 6（vel 2 + mass 1 + force 3）+ mass_weight 1。
        # 两处字面量都由 `test_towing_rl_lab_dimension_contract_matches_upper_logic` 交叉校验，
        # 避免 rl_lab 侧静默漂移（本模块不导入 isaac 侧包，以保持离线可导入）。
        if self.num_obs != 57 or self.num_decoder_obs != 7:
            raise ValueError(
                f"Towing contract requires policy=57 and decoder=7, got "
                f"{self.num_obs} and {self.num_decoder_obs}")
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
        # targets（6 = vel 2 + mass 1 + force 3）与质量监督权重（1）在同一组里拼接。
        decoder = self._obs_dict["decoder"]
        return decoder[:, :6], decoder[:, 6]

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

