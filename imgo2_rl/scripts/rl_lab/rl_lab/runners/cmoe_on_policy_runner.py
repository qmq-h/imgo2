# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The CMoE Authors (Fudan University).
# Adapted from rsl_rl (BSD-3-Clause, Copyright (c) 2021 ETH Zurich, Nikita Rudin
# and NVIDIA CORPORATION & AFFILIATES). See rsl_rl/LICENSE.

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from ..algorithms.cmoe_ppo import CMoEPPO
from ..envs.vec_env import VecEnv
from ..modules.cmoe_actor_critic import CMoEActorCritic

# 控制台输出过滤：`Curriculum/terrain_levels/` 下的逐列指标有 7 组 × 8 列（gait_*）＋
# 逐列 level_*/tracking_* 等，一轮能刷几十行。**TensorBoard 里一个都不少**，只是不打印。
CONSOLE_MODES = ("none", "gait", "all")


def _is_aggregate_leaf(leaf: str) -> bool:
    """是否是"汇总量"（要打印）：`*_mean`／`*_min`／`*_max` 以及全局的几个 frac/used。"""
    if leaf.endswith(("_mean", "_min", "_max")):
        return True
    return leaf in {"move_up_frac", "move_down_frac", "frozen_frac", "tracking_pass_frac",
                    "tracking_fail_frac", "tracking_is_used"}


def _console_visible(key: str, mode: str = "gait") -> bool:
    """该 `ep_info` 键是否打印到控制台（**不影响写 TensorBoard**）。

    * ``"none"``：全打印（旧行为）；
    * ``"gait"``（默认）：不打印逐列 `gait_*_<地形>`（trot/bound/pace/bounce/height/airtime/mismatch × 8 列），
      但保留 `gait_*_mean` 六个汇总量；
    * ``"all"``：`Curriculum/terrain_levels/` 下**所有逐列**指标都不打印（另含 `level_<地形>`、
      `tracking_<地形>`、`tracking_pass_frac_<地形>` 等），只留汇总量。
    """
    if mode == "none":
        return True
    leaf = key.rsplit("/", 1)[-1]
    if mode == "gait":
        return not (leaf.startswith("gait_") and not leaf.endswith("_mean"))
    if mode == "all":
        if not key.startswith("Curriculum/terrain_levels/"):
            return True
        return _is_aggregate_leaf(leaf)
    raise ValueError(f"未知 console 模式 {mode!r}，可选 {CONSOLE_MODES}")


class CMoEOnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg = train_cfg
        console_mode = train_cfg.get("console_skip_columns", "gait")
        if console_mode not in CONSOLE_MODES:
            raise ValueError(f"console_skip_columns={console_mode!r} 未知，可选 {CONSOLE_MODES}")
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        self.num_actor_obs = self.env.num_obs
        self.num_critic_obs = num_critic_obs
        actor_critic = CMoEActorCritic(self.env.num_obs,
                                                        num_critic_obs,
                                                        self.env.num_one_step_obs,
                                                        self.env.num_actions,
                                                        history_steps=self.env.history_steps,
                                                        terrain_obs_dim=self.env.num_terrain_obs,
                                                        device = self.device,
                                                        **self.policy_cfg
                                                        ).to(self.device)
        self.alg = CMoEPPO(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = train_cfg["num_steps_per_env"]
        self.save_interval = train_cfg["save_interval"]

        # 「从已有步态策略起步」（2026-09-25）：把 45 维先验零填充装进专家，再开始训练。
        # 依据与契约见 docs/cmoe_trot_warmstart_2026-09-25.md；离线核对见
        # scripts/tools/check_cmoe_expert_init.py（前 45 维顺序/尺度、动作定义逐项一致）。
        self.prior_teacher = None
        self._prior_action_rmse = None
        self._prior_ref_obs = None
        self._prior_actor_snapshot = None
        from ..utils.pretrained_prior import normalize_prior_path

        raw_init_from = self.cfg.get("init_experts_from")
        init_from = normalize_prior_path(raw_init_from)
        if raw_init_from and init_from is None:
            print(
                f"[WARN] init_experts_from={raw_init_from!r} 是空串/空白 ⇒ 视为没给先验。"
                "若本意是「从先验起步」，请给真实路径（空串会静默不装）"
            )
        if init_from:
            from ..utils.pretrained_prior import (
                install_prior,
                jitter_new_columns,
                init_gate_bias,
                load_prior,
                prior_action_rmse,
                teacher_actor,
            )

            self._prior_action_rmse = prior_action_rmse

            prior = load_prior(init_from)
            if prior.actor_obs_dim != self.env.num_one_step_obs:
                raise ValueError(
                    f"先验输入维 {prior.actor_obs_dim} != 当前帧本体感知 {self.env.num_one_step_obs}；"
                    "先跑 scripts/tools/check_cmoe_expert_init.py 核对契约"
                )
            mode = self.cfg.get("init_experts_mode", "all")
            include_critic = bool(self.cfg.get("init_experts_critic", True))
            report = install_prior(
                self.alg.actor_critic, prior, mode=mode, include_critic=include_critic,
                include_std=bool(self.cfg.get("init_experts_std", True)),
            )
            jitter = float(self.cfg.get("init_experts_jitter", 0.0) or 0.0)
            if jitter > 0:
                jitter_report = jitter_new_columns(self.alg.actor_critic, self.env.num_one_step_obs, jitter)
                print(f"[INFO]   已给 {jitter_report['jittered_experts']} 个专家（第 1 个保持纯净）的"
                      f"新增 112 列加 N(0, {jitter}) 扰动以打破对称")
            gate_bias = self.cfg.get("init_gate_bias")
            if gate_bias is not None:
                bias_report = init_gate_bias(
                    self.alg.actor_critic,
                    expert_index=int(gate_bias),
                    margin=float(self.cfg.get("init_gate_bias_margin", 4.0)),
                    shrink_last_layer=float(self.cfg.get("init_gate_bias_shrink", 0.0) or 0.0),
                )
                print(f"[INFO]   门控初始偏置到专家 {bias_report['expert']}"
                      f"（margin={bias_report['margin']}, shrink={bias_report['shrink_last_layer']}）"
                      " ⇒ 第 0 步的混合动作 ≈ 该专家（配合 --init_experts_mode=first 即 ≈ 先验）")
            self.prior_teacher = teacher_actor(prior).to(self.device).eval()
            # v5 锚定：教师交给算法；**强度**由 runner 每个控制步按地形掩码 × 衰减算好写进 storage。
            self.alg.teacher_policy = self.prior_teacher
            self.alg.teacher_obs_dim = int(self.env.num_one_step_obs)
            self.alg.anchor_expert = int(self.cfg.get("anchor_expert", 0))
            self.alg.anchor_target = str(self.cfg.get("anchor_target", "expert"))
            from ..utils.anchor import resolve_targets as _resolve_anchor_targets
            _resolve_anchor_targets(self.alg.anchor_target)   # 非法取值启动即报错
            self._anchor_columns = self._terrain_columns_for_anchor()
            self._anchor_flat_share = None
            self._anchor_coef_now_value = 0.0
            if float(self.cfg.get("anchor_coef", 0.0) or 0.0) > 0.0:
                print(f"[INFO]   先验锚定：专家 {self.alg.anchor_expert}；地形 "
                      f"{tuple(self.cfg.get('anchor_terrain_names', ('flat',)))}"
                      f"（{len(self._anchor_columns)} 列）；权重 "
                      f"{self.cfg.get('anchor_coef')} → {self.cfg.get('anchor_coef_final')}"
                      f"（{self.cfg.get('anchor_decay_iters')} 轮线性衰减）；锚的对象："
                      f"{self.alg.anchor_target}；λ 乘在损失外面，掩码只决定\"哪些样本\"")
            # 参数空间漂移的基准（与观测分布无关，见 log() 里的说明）
            self._prior_actor_snapshot = {
                name: parameter.detach().clone()
                for name, parameter in self.alg.actor_critic.experts.named_parameters()
            }
            first = report["experts"][0]["actor"]
            print(
                f"[INFO] 先验初始化：{prior.summary()}；已装进 {report['installed_experts']}/"
                f"{report['total_experts']} 个专家（mode={mode}），actor 第一层 {first['first_layer']}"
                f"（置零列 {first['zeroed_columns']}）；critic "
                f"{'随先验装入' if report['critic_installed'] else '**保持随机初始化**'}；噪声 std "
                f"{'随先验装入' if 'std' in report else '**沿用 init_noise_std**'}"
            )
            if "std" in report:
                print(f"[INFO]   噪声 std 随先验装入：均值 {report['std']['mean']:.4f}")
            else:
                print("[WARN]   先验不含噪声 std（TorchScript 导出件没有），沿用 init_noise_std")

        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0


    def _terrain_columns_for_anchor(self) -> list[int]:
        """启动时算一次：`anchor_terrain_names` 占用的**列索引**（纯 torch/numpy，规则同 Isaac 的按比例分配）。"""
        from ..utils.terrain_masks import terrain_columns

        try:
            generator = self.env.unwrapped.cfg.scene.terrain.terrain_generator
            keys = list(generator.sub_terrains.keys())
            proportions = [generator.sub_terrains[key].proportion for key in keys]
            return terrain_columns(keys, proportions, generator.num_cols,
                                   self.cfg.get("anchor_terrain_names", ("flat",)))
        except Exception as exc:  # noqa: BLE001 - 取不到就不锚（宁可不生效也不让训练崩）
            print(f"[WARN] 取不到锚定地形列（{exc}）⇒ 本次不锚定")
            return []

    def _anchor_coef_now(self, iteration: int) -> float:
        """当前的锚定强度 λ(t)（线性衰减）。0 ⇒ 关闭。

        ⚠️ λ **不能**乘进掩码：`weighted_mse` 是归一化均值 `Σ(m·MSE)/Σm`，公共系数会被约掉
        （2026-09-28 复查抓到的真 bug：0.2 与 0.3 得到同一损失 0.972105）。所以这里只算标量，
        由算法乘在损失外面。
        """
        coef0 = float(self.cfg.get("anchor_coef", 0.0) or 0.0)
        if coef0 <= 0.0 or self.prior_teacher is None or not getattr(self, "_anchor_columns", None):
            return 0.0
        coef1 = float(self.cfg.get("anchor_coef_final", 0.0) or 0.0)
        decay = max(1, int(self.cfg.get("anchor_decay_iters", 1000) or 1000))
        frac = min(1.0, max(0.0, float(iteration) / float(decay)))
        return coef0 + (coef1 - coef0) * frac

    def _anchor_mask_now(self):
        """当前控制步的**0/1 地形掩码**（[N] float32）：哪些环境属于被锚定的地形。

        未启用、没有教师、或没有可用列时返回 `None` ⇒ storage 写 0 ⇒ 锚损失恒 0（与旧行为一致）。
        """
        coef0 = float(self.cfg.get("anchor_coef", 0.0) or 0.0)
        if coef0 <= 0.0 or self.prior_teacher is None or not getattr(self, "_anchor_columns", None):
            return None
        from ..utils.terrain_masks import anchor_weights

        terrain = self.env.unwrapped.scene.terrain
        return anchor_weights(terrain.terrain_types, columns=self._anchor_columns, scale=1.0)

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)

        if self.prior_teacher is not None and self.writer is not None:
            # 初始化自检：全专家同一先验时这里应当≈0（与门控无关）。
            # ⚠️ 参考观测**只抓这一次**：`prior_action_rmse` 是动作空间里的**绝对** RMSE，会随观测幅度
            # 线性放大（实测同一权重：|obs|~0.5 ⇒ 0.66，|obs|~10 ⇒ 15.5）。若每轮都在"当轮观测"上算，
            # 读到的其实是"机器人摔得有多凶"，不是"先验漂了多少"（2026-09-25 实测过这个坑）。
            self._prior_ref_obs = obs[: min(1024, obs.shape[0])].clone()
            initial_rmse = float(
                self._prior_action_rmse(
                    self.alg.actor_critic, self.prior_teacher, self._prior_ref_obs, self.env.num_one_step_obs
                ).item()
            )
            print(f"[INFO] 先验初始化自检：固定参考批次（1024 条）上 混合策略 vs 先验 动作 RMSE = "
                  f"{initial_rmse:.3e}（全专家初始化时应≈0；后续同批次读数上升才算先验被改写）")
            self.writer.add_scalar("Policy/prior_action_rmse", initial_rmse, 0)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            # λ(t)：锚定强度；算法把它乘在锚损失**外面**（乘进掩码会被归一化约掉）
            self.alg.anchor_coef = self._anchor_coef_now(it)
            self._anchor_coef_now_value = self.alg.anchor_coef
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos, termination_ids, termination_privileged_obs = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    termination_ids = termination_ids.to(self.device)
                    termination_privileged_obs = termination_privileged_obs.to(self.device)
                    next_critic_obs = critic_obs.clone().detach()
                    next_critic_obs[termination_ids] = termination_privileged_obs.clone().detach()

                    anchor_mask = self._anchor_mask_now()
                    self.alg.process_env_step(rewards, dones, infos, next_critic_obs,
                                              anchor_weight=anchor_mask)
                    if anchor_mask is not None:
                        self._anchor_flat_share = float(anchor_mask.mean())
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
                self.alg.compute_returns(obs, critic_obs)
                
            mean_value_loss, mean_surrogate_loss, estimation_loss, latent_loss, recons_loss, kld_loss, estimation_loss2, latent_loss2, recons_loss2, kld_loss2, contrastive_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals(), estimation_loss, latent_loss, recons_loss, kld_loss, estimation_loss2, latent_loss2, recons_loss2, kld_loss2, contrastive_loss)
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)), iteration=it)
            ep_infos.clear()

        self.current_learning_iteration = tot_iter
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, estimation_loss, latent_loss, recons_loss, kld_loss, estimation_loss2, latent_loss2, recons_loss2, kld_loss2, contrastive_loss, width=80, pad=35):
        console_mode = self.cfg.get("console_skip_columns", "gait")
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            # ⚠️ 键集合**可能逐条不同**，不能拿 `ep_infos[0]` 的键去索引所有条目（会 KeyError）：
            # 逐列指标 `Curriculum/terrain_levels/tracking_<地形>` / `gait_<标签>_<地形>` 只在
            # "这一步确实有该列的回合力样本"时才写（`env_ids` 只是这一步刚结束的环境，见
            # `mdp/curriculums.py`）。所以按键的**并集**遍历，并在每条里跳过缺这个键的条目 ⇒
            # 该轮的值＝该轮里有该列样本的那几次的均值（不会出现 NaN、也不会漏掉整轮的键）。
            keys = dict.fromkeys(key for ep_info in locs['ep_infos'] for key in ep_info)
            for key in keys:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    if key not in ep_info:
                        continue
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                # 控制台过滤（2026-09-25 用户要求「log 中一堆 gait 不打印」）：逐列指标仍写
                # TensorBoard，只是不刷控制台；汇总量（`*_mean`）照旧打印。模式见 `console_skip_columns`。
                if _console_visible(key, console_mode):
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/Estimation Loss', estimation_loss, locs['it'])
        self.writer.add_scalar('Loss/Latent Loss', latent_loss, locs['it'])
        self.writer.add_scalar('Loss/Recons Loss', recons_loss, locs['it'])
        self.writer.add_scalar('Loss/Kld Loss', kld_loss, locs['it'])
        self.writer.add_scalar('Loss/Estimation Loss2', estimation_loss2, locs['it'])
        self.writer.add_scalar('Loss/Latent Loss2', latent_loss2, locs['it'])
        self.writer.add_scalar('Loss/Recons Loss2', recons_loss2, locs['it'])
        self.writer.add_scalar('Loss/Kld Loss2', kld_loss2, locs['it'])
        self.writer.add_scalar('Loss/contrastive', contrastive_loss, locs['it'])
        if getattr(self.alg, 'last_anchor_loss', None) is not None:
            self.writer.add_scalar('Loss/anchor_prior', self.alg.last_anchor_loss, locs['it'])
        if getattr(self.alg, 'last_anchor_mse', None) is not None:
            # MSE 本身单独记一条（**可验收**：这是"平地混合 vs AMP"的动作 MSE，与 λ 无关）
            self.writer.add_scalar('Policy/anchor_mse_flat', self.alg.last_anchor_mse, locs['it'])
        if getattr(self, '_anchor_flat_share', None) is not None:
            self.writer.add_scalar('Policy/anchor_flat_share', self._anchor_flat_share, locs['it'])
        if getattr(self, '_anchor_coef_now_value', 0.0):
            self.writer.add_scalar('Policy/anchor_coef', self._anchor_coef_now_value, locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        if self.prior_teacher is not None and self.cfg.get("log_prior_rmse", True):
            # 漂移度量三条（都不受"摔倒导致观测跑偏"影响）：
            # ① prior_action_rmse：在**固定的 1024 条参考观测**上算的绝对动作 RMSE（见 learn() 里的说明）；
            # ② prior_weight_rel_drift：专家 actor 参数相对"刚装好那一刻"的**相对 L2 漂移**；
            # ③ prior_new_col_mass：专家第一层权重里**落在地形/估计器那 112 列上的份额**（装好时＝0，
            #    上升＝地形通路在被打开）。
            reference = self._prior_ref_obs if self._prior_ref_obs is not None else locs['obs']
            prior_rmse = self._prior_action_rmse(
                self.alg.actor_critic, self.prior_teacher, reference, self.env.num_one_step_obs
            )
            self.writer.add_scalar('Policy/prior_action_rmse', prior_rmse.item(), locs['it'])
            if self._prior_actor_snapshot is not None:
                with torch.no_grad():
                    current = dict(self.alg.actor_critic.experts.named_parameters())
                    numerator = torch.zeros((), device=self.device)
                    denominator = torch.zeros((), device=self.device)
                    for name, snapshot in self._prior_actor_snapshot.items():
                        now = current[name].detach()
                        numerator = numerator + (now - snapshot.to(now.device)).float().pow(2).sum()
                        denominator = denominator + snapshot.float().pow(2).sum()
                    relative = float((numerator / denominator).sqrt().item()) if float(denominator) > 0 else 0.0
                    new_mass = torch.zeros((), device=self.device)
                    total_mass = torch.zeros((), device=self.device)
                    for expert in self.alg.actor_critic.experts:
                        weight = expert.actor[0].weight.detach()
                        new_mass = new_mass + weight[:, self.env.num_one_step_obs:].abs().sum()
                        total_mass = total_mass + weight.abs().sum()
                    mass_ratio = float((new_mass / total_mass).item()) if float(total_mass) > 0 else 0.0
                self.writer.add_scalar('Policy/prior_weight_rel_drift', relative, locs['it'])
                self.writer.add_scalar('Policy/prior_new_col_mass', mass_ratio, locs['it'])
        if self.alg.actor_critic.gate_weights is not None:
            gate_weights = self.alg.actor_critic.gate_weights.detach()
            gate_entropy = -(gate_weights * gate_weights.clamp_min(1.0e-8).log()).sum(dim=-1).mean()
            self.writer.add_scalar('Policy/gate_entropy', gate_entropy.item(), locs['it'])
            for expert_id, weight in enumerate(gate_weights.mean(dim=0)):
                self.writer.add_scalar(f'Policy/expert_{expert_id}_mean_weight', weight.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None, iteration=None):
        iteration = self.current_learning_iteration if iteration is None else iteration
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'state_estimator_optimizer_state_dict': self.alg.actor_critic.state_estimator.optimizer.state_dict(),
            'terrain_estimator_optimizer_state_dict': self.alg.actor_critic.terrain_estimator.optimizer.state_dict(),
            'iter': iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            if 'state_estimator_optimizer_state_dict' in loaded_dict:
                self.alg.actor_critic.state_estimator.optimizer.load_state_dict(
                    loaded_dict['state_estimator_optimizer_state_dict']
                )
            if 'terrain_estimator_optimizer_state_dict' in loaded_dict:
                self.alg.actor_critic.terrain_estimator.optimizer.load_state_dict(
                    loaded_dict['terrain_estimator_optimizer_state_dict']
                )
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
