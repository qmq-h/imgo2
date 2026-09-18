# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
import pickle
from collections import deque
import statistics

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch

from rl_lab.algorithms import AMPPPO, PPO
from rl_lab.modules import ActorCritic, ActorCriticRecurrent
from rl_lab.envs import VecEnv
from rl_lab.algorithms.amp_discriminator import AMPDiscriminator
from rl_lab.datasets.motion_loader import AMPLoader
from rl_lab.utils.utils import Normalizer


def _debug_print_observation_nans(group_name, obs, names, sizes):
    print(f"[AMP DEBUG] {group_name}: shape={tuple(obs.shape)} finite={torch.isfinite(obs).all()} nan={torch.isnan(obs).sum()}")
    start = 0
    for name, size in zip(names, sizes):
        term = obs[:, start : start + size]
        bad_env_ids = torch.nonzero(~torch.isfinite(term).all(dim=1), as_tuple=False).squeeze(-1)
        print(
            f"[AMP DEBUG] {group_name}.{name}: "
            f"slice=({start}:{start + size}) finite={torch.isfinite(term).all()} nan={torch.isnan(term).sum()}"
        )
        if len(bad_env_ids) > 0:
            print(f"[AMP DEBUG] {group_name}.{name} bad env ids: {bad_env_ids.detach().cpu().tolist()}")
            print(f"[AMP DEBUG] {group_name}.{name} bad values:")
            print(term[bad_env_ids].detach().cpu())
        start += size

class AMPOnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg = train_cfg.get("runner", train_cfg)
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        # 四足足端在接触传感器里的下标，第一次记录指标时解析（见 _feet_air_metrics）
        self._foot_ids = None
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        if self.env.include_history_steps is not None:
            num_actor_obs = self.env.num_obs * self.env.include_history_steps
        else:
            num_actor_obs = self.env.num_obs
        # actor_critic
        actor_critic: ActorCritic = actor_critic_class( 
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=self.env.num_actions,
            **self.policy_cfg).to(self.device)
        # amp loader
        amp_data = AMPLoader(
            device, time_between_frames=self.env.dt, preload_transitions=True,
            num_preload_transitions=self.cfg['amp_num_preload_transitions'],
            motion_files=self.cfg["amp_motion_files"])
        # 归一化
        amp_normalizer = Normalizer(amp_data.observation_dim)
        # 鉴别器
        discriminator = AMPDiscriminator(
            amp_data.observation_dim * 2, # state + next_state
            self.cfg['amp_reward_coef'],
            self.cfg['amp_discr_hidden_dims'], device,
            self.cfg['amp_task_reward_lerp']).to(self.device)

        # self.discr: AMPDiscriminator = AMPDiscriminator()
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        min_std = (
            torch.tensor(self.cfg["min_normalized_std"], device=self.device) *
            (torch.abs(self.env.dof_pos_limits[:, 1] - self.env.dof_pos_limits[:, 0])))
        self.alg: PPO = alg_class(actor_critic, discriminator, amp_data, amp_normalizer, device=self.device, min_std=min_std, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [num_actor_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        amp_obs = self.env.get_amp_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs, amp_obs = obs.to(self.device), critic_obs.to(self.device), amp_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)
        self.alg.discriminator.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        debug_printed_initial_obs = False
        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            # 必须每轮同步，否则 save() 写进 checkpoint 的 'iter' 恒为该次运行起点（通常 0），
            # resume 时会从错误位置继续（AMP/PPO 曾缺这一行，HIM runner 有）。
            self.current_learning_iteration = it
            start = time.time()
            amp_reward_sum = 0.0
            amp_disc_pred_sum = 0.0
            amp_reward_count = 0
            task_reward_sum = 0.0
            amp_height_sum = 0.0
            amp_low_height_fraction_sum = 0.0
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, amp_obs)
                    obs, privileged_obs, next_amp_obs, rewards, dones, infos, reset_env_ids, terminal_amp_states = (
                        self.env.step(actions)
                    )

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, next_amp_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), next_amp_obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                    # Account for terminal states. ifs reset，set teriminal_amp_states
                    next_amp_obs_with_term = torch.clone(next_amp_obs)
                    next_amp_obs_with_term[reset_env_ids] = terminal_amp_states

                    amp_rewards, amp_disc_pred = self.alg.discriminator.predict_amp_reward(
                        amp_obs, next_amp_obs_with_term, rewards, normalizer=self.alg.amp_normalizer)
                    amp_reward_sum += amp_rewards.mean().item()
                    amp_disc_pred_sum += amp_disc_pred.mean().item()
                    amp_reward_count += 1
                    task_reward_sum += rewards.mean().item()
                    # The AMP contract stores root height in its last column.
                    # Use pre-reset terminal states so falls are not hidden by reset.
                    amp_heights = next_amp_obs_with_term[:, -1]
                    amp_height_sum += amp_heights.mean().item()
                    amp_low_height_fraction_sum += (amp_heights < 0.20).float().mean().item()
                    rewards = amp_rewards
                    amp_obs = torch.clone(next_amp_obs)
                    self.alg.process_env_step(rewards, dones, infos, next_amp_obs_with_term)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'log' in infos and infos['log']:
                            ep_infos.append(infos['log'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            mean_value_loss, mean_surrogate_loss, mean_amp_loss, mean_grad_pen_loss, mean_policy_pred, mean_expert_pred = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))
        mean_amp_reward_step = locs['amp_reward_sum'] / max(locs['amp_reward_count'], 1)
        mean_amp_disc_pred_step = locs['amp_disc_pred_sum'] / max(locs['amp_reward_count'], 1)

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP', locs['mean_amp_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP_grad', locs['mean_grad_pen_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)
        self.writer.add_scalar('Train/mean_amp_reward_step', mean_amp_reward_step, locs['it'])
        self.writer.add_scalar('Train/mean_amp_disc_pred_step', mean_amp_disc_pred_step, locs['it'])
        count = max(locs['amp_reward_count'], 1)
        task_reward_mean = locs['task_reward_sum'] / count
        self.writer.add_scalar('Train/mean_task_reward_step', task_reward_mean, locs['it'])
        self.writer.add_scalar(
            'Train/weighted_task_reward_step',
            self.alg.discriminator.task_reward_lerp * task_reward_mean, locs['it'])
        self.writer.add_scalar('AMP/mean_root_height_m', locs['amp_height_sum'] / count, locs['it'])
        self.writer.add_scalar(
            'AMP/fraction_root_height_below_0_20m',
            locs['amp_low_height_fraction_sum'] / count, locs['it'])
        # 「滞空」物理量：训练日志里原本只有 Episode_Reward/feet_air_time（奖励值），
        # 看不到物理量，于是每次想确认步频有没有动都得占用 GPU 跑一次回放。
        # 这里直接写两条标量（与 eval_gait.py 的 air_time_mean_s / air_fraction 同口径）：
        #   mean_last_air_time_s     —— 四足「最近一次滞空时长」的均值（参考 0.20–0.24 s；实测 0.067–0.085 s）
        #   mean_air_time_fraction   —— 当前腾空足比例（≈ 1 − 占空比；实测 0.40–0.50）
        mean_air_time, air_fraction = self._feet_air_metrics()
        if mean_air_time is not None:
            self.writer.add_scalar('AMP/mean_last_air_time_s', mean_air_time, locs['it'])
            self.writer.add_scalar('AMP/mean_air_time_fraction', air_fraction, locs['it'])
        # 判别器对「参考动作」与「策略动作」各自的原始打分（训练目标分别是 +1 / -1）。
        # 这两条曲线是判断判别器是否还有区分能力的关键：若 expert 也显著为负，说明
        # 判别器把专家数据判成假（AMP-07 的「坐标系域差」嫌疑），此时风格奖励失效、
        # 任务奖励独自驱动训练；若二者分开，则是正常判别。此前只在控制台打印，未入库。
        if locs.get('mean_expert_pred') is not None:
            self.writer.add_scalar('AMP/disc_expert_pred', locs['mean_expert_pred'], locs['it'])
        if locs.get('mean_policy_pred') is not None:
            self.writer.add_scalar('AMP/disc_policy_pred', locs['mean_policy_pred'], locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if mean_air_time is None:
            air_line = f"""{'Mean last air time (s):':>{pad}} n/a\n"""
        else:
            # 参考 0.20–0.24 s；我们 24500 轮时实测 0.067–0.085 s（步频 3.1 倍的直接体现）
            air_line = (f"""{'Mean last air time (s):':>{pad}} {mean_air_time:.4f}"""
                        f"""  [target ~0.20, was 0.067-0.085]\n"""
                        f"""{'Mean air-borne feet frac:':>{pad}} {air_fraction:.3f}\n""")
        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'AMP loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
                          f"""{'AMP grad pen loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
                          f"""{'AMP mean policy pred:':>{pad}} {locs['mean_policy_pred']:.4f}\n"""
                          f"""{'AMP mean expert pred:':>{pad}} {locs['mean_expert_pred']:.4f}\n"""
                          f"""{'Mean AMP reward/step:':>{pad}} {mean_amp_reward_step:.4f}\n"""
                          f"""{'Mean AMP disc pred/step:':>{pad}} {mean_amp_disc_pred_step:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{air_line}"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean AMP reward/step:':>{pad}} {mean_amp_reward_step:.4f}\n"""
                          f"""{'Mean AMP disc pred/step:':>{pad}} {mean_amp_disc_pred_step:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{air_line}""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def _feet_air_metrics(self):
        """四足滞空物理量 → (最近一次滞空时长均值 s, 当前腾空足比例)；拿不到传感器时返回 (None, None)。

        口径与 `scripts/tools/eval_gait.py` 的 `air_time_mean_s` / `air_fraction` 一致，
        这样训练时在 TensorBoard 上就能盯步频，不必每次回放。
        """
        try:
            sensor = self.env.unwrapped.scene.sensors.get("contact_forces")
        except AttributeError:
            return None, None
        if sensor is None or not hasattr(sensor.data, "last_air_time"):
            return None, None
        if self._foot_ids is None:
            self._foot_ids = sensor.find_bodies(".*_FOOT", preserve_order=True)[0]
        if len(self._foot_ids) == 0:
            return None, None
        last_air = sensor.data.last_air_time[:, self._foot_ids]
        current_air = sensor.data.current_air_time[:, self._foot_ids]
        return float(last_air.mean().item()), float((current_air > 0).float().mean().item())

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'discriminator_state_dict': self.alg.discriminator.state_dict(),
            'amp_normalizer': _pack_normalizer(self.alg.amp_normalizer),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)


    def load(self, path, load_optimizer=True):
        # map_location=self.device：① 在 GPU 机上等价于原来的行为（同设备重映射是恒等的），
        # ② 让「在另一块 GPU 上存的 checkpoint」和「CPU 冒烟测试（--device=cpu）」都能载入 ——
        #    否则 torch.load 会按存档里的设备号恢复 storage，CPU-only 机器直接报
        #    "Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False"。
        try:
            loaded_dict = torch.load(path, weights_only=True, map_location=self.device)
        except pickle.UnpicklingError:
            loaded_dict = torch.load(path, weights_only=False, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        self.alg.discriminator.load_state_dict(loaded_dict['discriminator_state_dict'])
        self.alg.amp_normalizer = _unpack_normalizer(loaded_dict['amp_normalizer'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference


def _pack_normalizer(normalizer):
    if normalizer is None:
        return None
    return {
        "mean": normalizer.mean,
        "var": normalizer.var,
        "count": normalizer.count,
        "epsilon": normalizer.epsilon,
        "clip_obs": normalizer.clip_obs,
    }


def _unpack_normalizer(normalizer_data):
    if normalizer_data is None:
        return None
    if isinstance(normalizer_data, Normalizer):
        return normalizer_data

    normalizer = Normalizer(normalizer_data["mean"].shape)
    normalizer.mean = np.asarray(normalizer_data["mean"], dtype=np.float64)
    normalizer.var = np.asarray(normalizer_data["var"], dtype=np.float64)
    normalizer.count = normalizer_data["count"]
    normalizer.epsilon = normalizer_data.get("epsilon", normalizer.epsilon)
    normalizer.clip_obs = normalizer_data.get("clip_obs", normalizer.clip_obs)
    return normalizer
