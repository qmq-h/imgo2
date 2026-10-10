import os
import statistics
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from rl_lab.algorithms import PPO
from rl_lab.modules import (
    ActorCriticRecurrent,
    DynamicsDecoderTrainer,
    EmpiricalNormalizer,
    TowingDynamicsDecoder,
    augment_actor_observation,
    reset_gru_hidden,
)


#: 上层拖曳 checkpoint 契约版本。**v3**（2026-10-10 双头迁移）= policy 帧 58
#: （`last_action` 12 → 13）/ actor 80 / critic 73 / 动作 13（1 维 vx 偏移 + 12 维关节残差）。
#: 旧 v2（frame 57 / actor 79 / 动作 12）checkpoint 一律拒绝，必须新建 run。
#: 与 `scripts/towing/upper_policy_runtime.CHECKPOINT_CONTRACT_VERSION` 同值（离线测试交叉核对）。
TOWING_CONTRACT_VERSION = 3


class TowingOnPolicyRunner:
    """Repository-owned recurrent PPO plus supervised towing dynamics decoder."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg.get("runner", train_cfg)
        self.policy_cfg = train_cfg["policy"]
        self.alg_cfg = train_cfg["algorithm"]
        self.decoder_cfg = train_cfg["decoder"]
        self.env = env
        self.device = device
        self.log_dir = log_dir
        # 足端滞空指标的 body id 缓存（首次打印时解析，与 AMP runner 同一口径）
        self._foot_ids = None

        if self.cfg["policy_class_name"] != "ActorCriticRecurrent":
            raise ValueError("Towing runner requires ActorCriticRecurrent")
        if self.cfg["algorithm_class_name"] != "PPO":
            raise ValueError("Towing runner requires PPO")

        decoder_kwargs = {
            key: self.decoder_cfg[key]
            for key in ("frame_dim", "feature_dim", "hidden_dim", "num_layers", "force_scale", "latent_dim")
        }
        if decoder_kwargs["frame_dim"] != env.num_obs:
            raise ValueError(
                f"decoder frame_dim={decoder_kwargs['frame_dim']} does not match policy obs={env.num_obs}")
        self.decoder = TowingDynamicsDecoder(**decoder_kwargs).to(device)
        self.decoder.eval()
        self.decoder_trainer = DynamicsDecoderTrainer(
            self.decoder,
            learning_rate=self.decoder_cfg["learning_rate"],
            max_grad_norm=self.decoder_cfg["max_grad_norm"],
            velocity_coef=self.decoder_cfg["velocity_coef"],
            force_coef=self.decoder_cfg["force_coef"],
            mass_coef=self.decoder_cfg["mass_coef"],
            kld_weight=self.decoder_cfg["kld_weight"],
        )

        actor_obs_dim = env.num_obs + self.decoder.actor_feature_dim
        actor_critic = ActorCriticRecurrent(
            num_actor_obs=actor_obs_dim,
            num_critic_obs=env.num_privileged_obs,
            num_actions=env.num_actions,
            **self.policy_cfg,
        ).to(device)
        # 上层动作是**叠加在冻结策略关节目标上的残差**，必须从 0 起步：`ActorCritic` 的
        # 末层默认随机初始化，实测（2026-10-08，torch 默认 init）首拍 |a|max ≈ 0.149
        # （归一化），按 `action_scale` 折算约 ±0.04 rad 的**系统性关节偏置**——量值不大，
        # 但它是"每回合开始时冻结步态被固定偏移"的来源，且与残差式 RL 的常规做法相反。
        # 零初始化后首拍残差精确为 0（有测试守着），即起点严格等于冻结策略自身的步态。
        # 探索噪声仍由 `std` 提供，所以零初始化只去掉"起点偏置"，不减探索。
        # **只对 towing runner 生效**：`ActorCriticRecurrent` 由 ppo／amp／himloco 共用，
        # 不能改模块的默认初始化。
        actor_last_linear = [layer for layer in actor_critic.actor
                             if isinstance(layer, torch.nn.Linear)][-1]
        torch.nn.init.zeros_(actor_last_linear.weight)
        torch.nn.init.zeros_(actor_last_linear.bias)
        self.alg = PPO(actor_critic, device=device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.alg.init_storage(
            env.num_envs,
            self.num_steps_per_env,
            [actor_obs_dim],
            [env.num_privileged_obs],
            [env.num_actions],
        )

        self.critic_normalizer = EmpiricalNormalizer(
            env.num_privileged_obs,
            clip=self.cfg["critic_normalization_clip"],
        ).to(device)
        self.normalize_critic = bool(self.cfg["critic_empirical_normalization"])
        self.decoder_hidden = None
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self.current_learning_iteration = 0
        self.env.reset()

    def _critic_obs(self, critic_obs, *, update):
        critic_obs = critic_obs.to(self.device)
        if self.normalize_critic:
            return self.critic_normalizer(critic_obs, update=update)
        return critic_obs

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        # 启动行：Isaac Sim 起来 + 第一轮采集完要等一会儿（256 环境约 2.5 分钟），
        # 中间完全没有输出会让人误判成卡住。这条先说明「已开始、写到哪、跑多少轮」。
        print(
            f"[towing] 开始训练 | envs={self.env.num_envs} | "
            f"iterations={num_learning_iterations} | steps/env={self.num_steps_per_env} | "
            f"save_interval={self.save_interval} | log_dir={self.log_dir or '(不落盘)'}\n"
            f"[towing] 第一条进度需等环境首次 rollout 完成，请勿在此期间判断为卡死。",
            flush=True,
        )
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length))

        raw_obs = self.env.get_observations().to(self.device)
        critic_obs = self.env.get_privileged_observations().to(self.device)
        decoder_targets, mass_weights = self.env.get_decoder_supervision()
        self.alg.actor_critic.train()

        reward_buffer = deque(maxlen=100)
        length_buffer = deque(maxlen=100)
        current_rewards = torch.zeros(self.env.num_envs, device=self.device)
        current_lengths = torch.zeros(self.env.num_envs, device=self.device)
        start_iter = self.current_learning_iteration
        total_iter = start_iter + num_learning_iterations

        for iteration in range(start_iter, total_iter):
            self.current_learning_iteration = iteration
            start = time.time()
            frame_rollout, target_rollout, weight_rollout, done_rollout = [], [], [], []
            episode_infos = []

            decoder_initial_hidden = (self.decoder_hidden.detach().clone()
                                      if self.decoder_hidden is not None else None)
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    estimate, latent, self.decoder_hidden = self.decoder.forward_with_latent(
                        raw_obs, self.decoder_hidden, sample=False)
                    actor_obs = augment_actor_observation(raw_obs, estimate, latent)
                    normalized_critic_obs = self._critic_obs(critic_obs, update=True)
                    actions = self.alg.act(actor_obs, normalized_critic_obs)

                    frame_rollout.append(raw_obs)
                    target_rollout.append(decoder_targets.to(self.device))
                    weight_rollout.append(mass_weights.to(self.device))
                    raw_obs, critic_obs, rewards, dones, infos = self.env.step(actions)
                    raw_obs = raw_obs.to(self.device)
                    critic_obs = critic_obs.to(self.device)
                    rewards, dones = rewards.to(self.device), dones.to(self.device)
                    done_rollout.append(dones.bool())
                    decoder_targets, mass_weights = self.env.get_decoder_supervision()

                    self.alg.process_env_step(rewards, dones, infos)
                    self.decoder_hidden = reset_gru_hidden(self.decoder_hidden, dones.bool())
                    # Isaac Lab 的 ``ManagerBasedRLEnv`` 把 RewardManager／TerminationManager
                    # 的回合统计写在 ``extras["log"]`` 里（`Episode_Reward/*`、
                    # `Episode_Termination/*`），**不是** ``extras["episode"]``。只读后者会让
                    # 这些量一个都进不了 TensorBoard（2026-09-22 4 环境冒烟即如此：日志里只有
                    # 8 个标量，没有任何 reward 分项与终止原因）。此处按官方 rsl_rl runner
                    # 的同样顺序兼容两个键。
                    if "episode" in infos:
                        episode_infos.append(infos["episode"])
                    elif "log" in infos:
                        episode_infos.append(infos["log"])
                    current_rewards += rewards
                    current_lengths += 1
                    ended = (dones > 0).nonzero(as_tuple=False).flatten()
                    reward_buffer.extend(current_rewards[ended].cpu().tolist())
                    length_buffer.extend(current_lengths[ended].cpu().tolist())
                    current_rewards[ended] = 0
                    current_lengths[ended] = 0

                self.alg.compute_returns(self._critic_obs(critic_obs, update=False))
            collection_time = time.time() - start

            learn_start = time.time()
            value_loss, surrogate_loss = self.alg.update()
            decoder_loss = self.decoder_trainer.update(
                torch.stack(frame_rollout),
                torch.stack(target_rollout),
                torch.stack(weight_rollout),
                dones=torch.stack(done_rollout),
                hidden_state=decoder_initial_hidden,
            )
            learn_time = time.time() - learn_start

            # **无条件**调用：`_log` 内部自己判断 writer 是否可用。此前整块藏在
            # `if self.writer is not None` 里，一旦 writer 没建起来就一行进度都不出，
            # 看起来像卡死（用户 2026-09-22 据此误判为「运行很慢」）。
            self._log(
                iteration, total_iter, collection_time, learn_time,
                value_loss, surrogate_loss, decoder_loss,
                reward_buffer, length_buffer, episode_infos, start_iter,
            )
            if self.log_dir is not None and iteration % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{iteration}.pt"))

        self.current_learning_iteration = total_iter
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{total_iter}.pt"))

    @staticmethod
    def _aggregate_episode_infos(episode_infos):
        """把逐步收集的 episode 统计按键聚合成一份（标量取均值）。

        ``episode_infos`` 是 rollout 内**每一步**只要有环境 reset 就追加一条
        ``infos["log"]``，48 步 × 多环境会累积几十份字典。逐份打印会把同一批指标刷屏
        （2026-09-22 实测：每次 _log 打印上百行同名列，用户报告「每个环境都输出了一条」）。
        聚合成一份后，每次迭代每个指标只输出一行。
        """
        values = {}
        for info in episode_infos:
            for key, value in info.items():
                value = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
                values.setdefault(key, []).append(value.float().mean().item())
        return {key: statistics.mean(vals) for key, vals in values.items()}

    def _write_scalars(self, iteration, collection_time, learn_time, value_loss,
                       surrogate_loss, decoder_loss, mean_std, fps, reward_buffer,
                       length_buffer, episode_stats):
        # 只在 TensorBoard writer 可用时写；**控制台输出不走这里**，见 `_log`。
        self.writer.add_scalar("Loss/value_function", value_loss, iteration)
        self.writer.add_scalar("Loss/surrogate", surrogate_loss, iteration)
        self.writer.add_scalar("Loss/decoder", decoder_loss.item(), iteration)
        # 三项分量单独记录：去掉 target 归一化后三项尺度不同（m/s、kg、N），
        # 1:1:1 的权重并不等权。用这三条判断权重是否真的配平。
        _trainer = getattr(self, "decoder_trainer", None)
        for _name, _val in zip(("velocity", "force", "mass"),
                               getattr(_trainer, "last_parts", ())):
            self.writer.add_scalar(f"Loss/decoder_{_name}", _val, iteration)
        self.writer.add_scalar("Loss/decoder_kl", getattr(_trainer, "last_kl", 0.0), iteration)
        self.writer.add_scalar("Loss/decoder_weighted_kl", getattr(_trainer, "last_weighted_kl", 0.0), iteration)
        self.writer.add_scalar("Policy/mean_noise_std", mean_std, iteration)
        self.writer.add_scalar("Perf/total_fps", fps, iteration)
        self.writer.add_scalar("Perf/collection_time", collection_time, iteration)
        self.writer.add_scalar("Perf/learning_time", learn_time, iteration)
        if reward_buffer:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(reward_buffer), iteration)
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(length_buffer), iteration)
        for key, value in episode_stats.items():
            self.writer.add_scalar(f"Episode/{key}", value, iteration)

    def _format_progress(self, iteration, total_iter, collection_time, learn_time,
                         value_loss, surrogate_loss, decoder_loss, mean_std, fps,
                         reward_buffer, length_buffer, episode_stats, start_iter):
        """控制台排版对齐 `rl_lab/runners/ppo_on_policy_runner.py::log()`。

        ETA 分母用 `iteration - start_iter + 1`：resume 时 tot_time 只累计本次运行的时间。
        """
        if getattr(self, "compact_log", False):
            head = (f"[it {iteration:>5}/{total_iter}] "
                    f"rew {statistics.mean(reward_buffer) if reward_buffer else float('nan'):>7.2f}  "
                    f"len {statistics.mean(length_buffer) if reward_buffer else float('nan'):>6.1f}  "
                    f"val {value_loss:>8.3f}  pol {surrogate_loss:>7.4f}  "
                    f"dec {decoder_loss.item():>7.3f}  std {mean_std:>5.2f}  "
                    f"coll {collection_time:>5.2f}s  lrn {learn_time:>5.2f}s  "
                    f"{fps:>6.0f} sps")
            # 分项只挑最需要观察的几项，避免又变长（完整分项在 TensorBoard 的 Episode_* 里）。
            # 2026-10-08：`reference_tracking` 已随残差方案删除（没有 reference_command 了）。
            # 2026-10-09：`yaw_heading` 已关闭（横向/朝向改由 PD 外环负责），从 picks 去掉；
            # 换成这天新加的四项——否则终端上看不到它们，只能去 TensorBoard 翻。
            picks = ("tracking_velocity", "low_level_pos_error", "towing_force_y",
                     "feet_slide", "stop_towing_force", "extra_distance",
                     "obs_stop_reached", "collision", "fall", "action_rate")
            detail = "  ".join(
                f"{k.replace('Episode_Reward/', '')[:9]}={episode_stats[k]:+.3f}"
                for k in (f"Episode_Reward/{p}" for p in picks) if k in episode_stats)
            eta = self.tot_time / (iteration - start_iter + 1) * (total_iter - iteration) / 3600
            return f"{head}  ETA {eta:5.2f}h" + (f"\n        {detail}" if detail else "")

        # 排版与 `rl_lab/runners/amp_on_policy_runner.py::log(width=80, pad=35)` 对齐
        # （2026-10-09 用户要求「AMP/PPO 训练时的终端形式」）：同一个 `#` 边框 + 居中标题 +
        # 右对齐标签，分项按 `Mean episode <key>: <value>` 打印。
        width, pad = 80, 35
        trainer = getattr(self, "decoder_trainer", None)
        air_time, air_fraction = self._feet_air_metrics()
        decoder_rows = [
            f"{'Mean decoder loss:':>{pad}} {decoder_loss.item():.4f}",
        ]
        for name, value in zip(("velocity", "force", "mass"),
                               getattr(trainer, "last_parts", ())):
            decoder_rows.append(f"{f'Mean decoder {name} loss:':>{pad}} {value:.4f}")
        decoder_rows.append(
            f"{'Decoder weighted KL:':>{pad}} {getattr(trainer, 'last_weighted_kl', 0.0):.4f}")
        air_rows = ([f"{'Mean last air time (s):':>{pad}} {air_time:.4f}",
                     f"{'Mean air-borne feet frac:':>{pad}} {air_fraction:.3f}"]
                    if air_time is not None else
                    [f"{'Mean last air time (s):':>{pad}} n/a"])
        lines = [
            "#" * width,
            f" Learning iteration {iteration}/{total_iter} ".center(width, " "),
            "",
            f"{'Computation:':>{pad}} {fps:.0f} steps/s "
            f"(collection: {collection_time:.3f}s, learning {learn_time:.3f}s)",
            f"{'Value function loss:':>{pad}} {value_loss:.4f}",
            f"{'Surrogate loss:':>{pad}} {surrogate_loss:.4f}",
            *decoder_rows,
            f"{'Mean action noise std:':>{pad}} {mean_std:.2f}",
            *air_rows,
        ]
        if reward_buffer:
            lines.append(f"{'Mean reward:':>{pad}} {statistics.mean(reward_buffer):.2f}")
            lines.append(f"{'Mean episode length:':>{pad}} {statistics.mean(length_buffer):.2f}")
        else:
            lines.append(f"{'Mean reward:':>{pad}} (本批尚无回合结束)")
        # 每个分项一行；`episode_stats` 已聚合，不会重复打印同名列。
        for key, value in episode_stats.items():
            lines.append(f"{('Mean episode ' + key + ':'):>{pad}} {value:.4f}")

        avg_iter_s = self.tot_time / (iteration - start_iter + 1)
        eta_s = avg_iter_s * (total_iter - iteration)
        lines += [
            "-" * width,
            f"{'Total timesteps:':>{pad}} {self.tot_timesteps}",
            f"{'Iteration time:':>{pad}} {collection_time + learn_time:.2f}s",
            f"{'Avg iteration:':>{pad}} {avg_iter_s:.2f}s",
            f"{'Total time:':>{pad}} {self.tot_time:.2f}s ({self.tot_time / 3600:.2f}h)",
            f"{'ETA:':>{pad}} {eta_s:.1f}s ({eta_s / 3600:.2f}h)",
        ]
        return "\n".join(lines)

    def _feet_air_metrics(self):
        """四足滞空物理量 → (最近一次滞空时长均值 s, 当前腾空足比例)；拿不到时返回 (None, None)。

        与 `rl_lab/runners/amp_on_policy_runner.py::_feet_air_metrics` 同一口径（同一套
        `track_air_time` 语义），只是传感器名换成本任务的 `foot_contacts`（`.*_FOOT`）。
        """
        try:
            sensor = self.env.unwrapped.scene.sensors.get("foot_contacts")
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

    def _log(
        self, iteration, total_iter, collection_time, learn_time,
        value_loss, surrogate_loss, decoder_loss,
        reward_buffer, length_buffer, episode_infos, start_iter,
    ):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += collection_time + learn_time
        fps = self.tot_timesteps / self.tot_time if self.tot_time > 0 else 0.0
        mean_std = self.alg.actor_critic.std.mean().item()
        episode_stats = self._aggregate_episode_infos(episode_infos)
        # TensorBoard 写入是可选路径，控制台输出是必须路径，两者不共用条件。
        if self.writer is not None:
            self._write_scalars(iteration, collection_time, learn_time, value_loss,
                                surrogate_loss, decoder_loss, mean_std, fps,
                                reward_buffer, length_buffer, episode_stats)
        print(self._format_progress(iteration, total_iter, collection_time, learn_time,
                                    value_loss, surrogate_loss, decoder_loss, mean_std,
                                    fps, reward_buffer, length_buffer, episode_stats,
                                    start_iter), flush=True)

    def save(self, path, infos=None):
        # 契约 **v3**（2026-10-10 双头迁移）：frame_dim 58 / 动作 13。旧 v2 的 57/79
        # checkpoint 会被 `load()` 明确拒绝（必须新建 run，不续训）。
        torch.save({
            "towing_contract": {"version": TOWING_CONTRACT_VERSION,
                                "frame_dim": self.decoder.frame_dim,
                                "explicit_dim": self.decoder.output_dim,
                                "latent_dim": self.decoder.latent_dim},
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "decoder_state_dict": self.decoder.state_dict(),
            "decoder_optimizer_state_dict": self.decoder_trainer.optimizer.state_dict(),
            "critic_normalizer_state_dict": self.critic_normalizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }, path)

    def load(self, path, load_optimizer=True):
        checkpoint = torch.load(path, map_location=self.device)
        expected = {"version": TOWING_CONTRACT_VERSION, "frame_dim": self.decoder.frame_dim,
                    "explicit_dim": self.decoder.output_dim, "latent_dim": self.decoder.latent_dim}
        if checkpoint.get("towing_contract") != expected:
            raise ValueError(
                f"Checkpoint predates/mismatches the towing contract "
                f"(got {checkpoint.get('towing_contract')!r}, expected {expected!r}); "
                f"旧 v2 的 57 维帧 / 79 维 actor / 12 维动作 checkpoint 已随双头迁移失效，"
                f"start a new run.")
        self.alg.actor_critic.load_state_dict(checkpoint["model_state_dict"])
        self.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        self.critic_normalizer.load_state_dict(checkpoint["critic_normalizer_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.decoder_trainer.optimizer.load_state_dict(checkpoint["decoder_optimizer_state_dict"])
        self.current_learning_iteration = checkpoint["iter"]
        self.decoder_hidden = None
        return checkpoint.get("infos")

    def get_inference_policy(self, device=None):
        if device is not None:
            self.alg.actor_critic.to(device)
            self.decoder.to(device)
        self.alg.actor_critic.eval()
        self.decoder.eval()
        # 暴露最近一次的 6 维估计和 16 维 latent，供 play 端与真值对照（decoder 是部署件，必须能核对精度）。
        self.last_estimate = None
        self.last_latent = None

        def policy(raw_obs):
            estimate, latent, self.decoder_hidden = self.decoder.forward_with_latent(
                        raw_obs, self.decoder_hidden, sample=False)
            self.last_estimate = estimate
            self.last_latent = latent
            return self.alg.actor_critic.act_inference(
                augment_actor_observation(raw_obs, estimate, latent))

        return policy
