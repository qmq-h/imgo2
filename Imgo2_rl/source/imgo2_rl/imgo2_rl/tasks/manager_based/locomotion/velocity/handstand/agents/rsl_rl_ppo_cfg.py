# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class Imgo2HandstandRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16 # 每个env中采样多少步
    max_iterations = 10000  # 训练最大周期；一般10000可以收敛，超过了就说明有问题
    save_interval = 100 # 多少个周期保存节点
    experiment_name = "handstand_rough"
    empirical_normalization = False
    clip_actions = 100
    # 策略框架
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu", 
        # elu: / relu / leaky_relu / tanh / sigmoid / selu / gelu / silu...
    )
    algorithm = RslRlPpoAlgorithmCfg(
        # L = L_actor + c1 * L_critic + c2 * entropy
        value_loss_coef=1.0, # c1
        use_clipped_value_loss=True,
        clip_param=0.2, # actor 更新范围限制 20%
        entropy_coef=0.01, # 探索系数
        num_learning_epochs=5, # 每个rollout训练多少个epochs
        num_mini_batches=4, # 每个epoch把样本分成多少batches
        learning_rate=1.0e-3, # 学习率
        schedule="adaptive", # 学习率调度策略 adaptive是根据KL自动降低学习率；fixed；linear 随着训练线性下降
        gamma=0.99, # 奖励折扣因子
        lam=0.95, # GAE参数lambda
        desired_kl=0.01, # 用于adaptive
        max_grad_norm=1.0, # 梯度裁剪，如果梯度大于max，取max
    )



