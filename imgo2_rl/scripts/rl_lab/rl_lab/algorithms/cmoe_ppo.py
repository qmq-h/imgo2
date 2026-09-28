# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The CMoE Authors (Fudan University).
# Adapted from rsl_rl (BSD-3-Clause, Copyright (c) 2021 ETH Zurich, Nikita Rudin
# and NVIDIA CORPORATION & AFFILIATES). See rsl_rl/LICENSE.

import torch
import torch.nn as nn
import torch.optim as optim

from ..modules.cmoe_actor_critic import CMoEActorCritic
from ..storage.cmoe_rollout_storage import CMoERolloutStorage
from ..utils.anchor import mix_experts, resolve_targets, weighted_mse

class CMoEPPO:
    actor_critic: CMoEActorCritic
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 contrastive_loss_coef=1.0,
                 device='cpu',
                 teacher_policy=None,
                 teacher_obs_dim: int = 45,
                 anchor_expert: int = 0,
                 anchor_target: str = "expert",
                 anchor_coef: float = 0.0,
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = CMoERolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.contrastive_loss_coef = contrastive_loss_coef
        # 先验锚定（v5，2026-09-28）：把**某一个专家**的输出拉向一份固定的教师策略。
        # 强度不在这里，而在 storage 的逐样本 `anchor_weight`（由 runner 按地形掩码 + 衰减曲线算好）⇒
        # 算法侧保持通用：只负责"对哪些样本、用哪个专家、锚到谁"。
        self.teacher_policy = teacher_policy
        self.teacher_obs_dim = int(teacher_obs_dim)
        self.anchor_expert = int(anchor_expert)
        # 锚"谁"的输出（2026-09-28 用户拍定）：`expert`＝只锚某个专家（v5 初版）；
        # `mixture`＝锚**最终混合输出**（"平地上整体必须像 AMP"）；`both`＝两者都锚。
        self.anchor_target = str(anchor_target)
        resolve_targets(self.anchor_target)      # 非法取值在构造时就报错
        # ⚠️ λ **必须**乘在损失外面：`weighted_mse` 是 Σ(m·MSE)/Σm 的归一化均值，
        # 把公共系数乘进掩码会被约掉（2026-09-28 复查抓到的真 bug：0.2 与 0.3 得到同一损失，
        # 实测 0.972105 vs 0.972105）。runner 每轮把当前 λ 写进这里。
        self.anchor_coef = float(anchor_coef)
        self.last_anchor_loss = None
        self.last_anchor_mse = None

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = CMoERolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.eval()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos, next_critic_obs, anchor_weight=None):
        self.transition.next_critic_observations = next_critic_obs.clone()
        self.transition.anchor_weight = anchor_weight
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_obs, last_critic_obs):
        self.actor_critic.compute_gate_weights(last_obs)
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_estimation_loss = 0
        mean_latent_loss = 0
        mean_recons_loss = 0
        mean_kld_loss = 0
        mean_estimation_loss2 = 0
        mean_latent_loss2 = 0
        mean_recons_loss2 = 0
        mean_kld_loss2 = 0
        mean_contrastive_loss = 0
        
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for obs_batch, critic_obs_batch, actions_batch, next_critic_obs_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, anchor_weight_batch in generator:
                
                self.actor_critic.act(obs_batch)
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch)
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                # Estimator update
                estimation_loss, latent_loss, recons_loss, kld_loss, estimation_loss2, latent_loss2, recons_loss2, kld_loss2 = self.actor_critic.update_estimators(obs_batch, critic_obs_batch, next_critic_obs_batch, lr=self.learning_rate)
                
                contrastive_loss = self.actor_critic.compute_contrastive_loss(obs_batch)

                # 先验锚定（v5）：把被锚对象的动作拉向教师。权重来自 storage（runner 按
                # `地形掩码 × 衰减曲线` 逐样本算好；不在目标地形上就是 0）⇒ 未启用时恒为 0。
                # `anchor_target`：`expert`＝只锚某个专家；`mixture`＝锚**最终混合输出**（平地上
                # 整体必须像 AMP，用户 2026-09-28 拍定）；`both`＝两者都锚。算式在 `utils/anchor.py`。
                anchor_loss = torch.zeros((), device=self.device)
                weight_sum = anchor_weight_batch.sum()
                if (self.teacher_policy is not None and self.anchor_coef > 0.0
                        and float(weight_sum) > 0.0):
                    with torch.inference_mode():
                        teacher_actions = self.teacher_policy(obs_batch[:, : self.teacher_obs_dim])
                    # ⚠️ `detach()`：锚不应当穿过 `build_actor_input`，否则状态/地形**估计器**也会被
                    # 这个损失拉走（它们有自己的损失）⇒ 互相打架。混合输出因此在这里重算一遍（不用
                    # `mu_batch`，它的图里带着估计器；多一次小前向可忽略）。
                    actor_input = self.actor_critic.build_actor_input(obs_batch).detach()
                    anchor_expert_flag, anchor_mixture_flag = resolve_targets(self.anchor_target)
                    predictions = []
                    if anchor_expert_flag:
                        predictions.append(
                            self.actor_critic.experts[self.anchor_expert].act_inference(actor_input)
                        )
                    if anchor_mixture_flag:
                        gate_weights = self.actor_critic.gating_network(actor_input)
                        expert_means = [
                            expert.act_inference(actor_input) for expert in self.actor_critic.experts
                        ]
                        predictions.append(mix_experts(expert_means, gate_weights))
                    # 归一化加权均值（掩码为 0/1 时＝目标地形样本上的 MSE）；**λ 乘在外面**
                    anchor_mse = weighted_mse(predictions, teacher_actions, anchor_weight_batch)
                    anchor_loss = self.anchor_coef * anchor_mse
                    # 只在**真的生效**时记录 ⇒ 未启用（没有教师/λ=0/掩码全 0）不会写出恒 0 的日志项
                    self.last_anchor_mse = float(anchor_mse.detach())
                    self.last_anchor_loss = float(anchor_loss.detach())

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = (
                    surrogate_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_batch.mean()
                    + self.contrastive_loss_coef * contrastive_loss
                    + anchor_loss
                )

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_estimation_loss += estimation_loss
                mean_latent_loss += latent_loss
                mean_recons_loss += recons_loss
                mean_kld_loss += kld_loss                                                                                       
                mean_estimation_loss2 += estimation_loss2 
                mean_latent_loss2 += latent_loss2
                mean_recons_loss2 += recons_loss2
                mean_kld_loss2 += kld_loss2

                mean_contrastive_loss += contrastive_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_estimation_loss /= num_updates
        mean_latent_loss /= num_updates
        mean_recons_loss /= num_updates
        mean_kld_loss /= num_updates
        mean_estimation_loss2 /= num_updates
        mean_latent_loss2 /= num_updates
        mean_recons_loss2 /= num_updates
        mean_kld_loss2 /= num_updates

        mean_contrastive_loss /= num_updates

        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_latent_loss, mean_recons_loss, mean_kld_loss,  mean_estimation_loss2, mean_latent_loss2, mean_recons_loss2, mean_kld_loss2, mean_contrastive_loss
