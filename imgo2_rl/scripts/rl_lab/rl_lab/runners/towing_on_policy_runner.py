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

        if self.cfg["policy_class_name"] != "ActorCriticRecurrent":
            raise ValueError("Towing runner requires ActorCriticRecurrent")
        if self.cfg["algorithm_class_name"] != "PPO":
            raise ValueError("Towing runner requires PPO")

        decoder_kwargs = {
            key: self.decoder_cfg[key]
            for key in ("frame_dim", "feature_dim", "hidden_dim", "num_layers", "force_scale")
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
        )

        actor_obs_dim = env.num_obs + 5
        actor_critic = ActorCriticRecurrent(
            num_actor_obs=actor_obs_dim,
            num_critic_obs=env.num_privileged_obs,
            num_actions=env.num_actions,
            **self.policy_cfg,
        ).to(device)
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

            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    estimate, self.decoder_hidden = self.decoder(raw_obs, self.decoder_hidden)
                    actor_obs = augment_actor_observation(raw_obs, estimate)
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
            # 分项只挑最需要观察的几项，避免又变长（完整分项在 TensorBoard 的 Episode_* 里）
            picks = ("reference_tracking", "action_magnitude", "yaw_heading",
                     "tracking_velocity", "min_clearance")
            detail = "  ".join(
                f"{k.replace('Episode_Reward/', '')[:9]}={episode_stats[k]:+.3f}"
                for k in (f"Episode_Reward/{p}" for p in picks) if k in episode_stats)
            eta = self.tot_time / (iteration - start_iter + 1) * (total_iter - iteration) / 3600
            return f"{head}  ETA {eta:5.2f}h" + (f"\n        {detail}" if detail else "")

        width, pad = 80, 26
        lines = [
            "#" * width,
            f" Learning iteration {iteration}/{total_iter} ".center(width, " "),
            "",
            f"{'Computation:':>{pad}} {fps:.0f} steps/s "
            f"(collection: {collection_time:.3f}s, learning: {learn_time:.3f}s)",
            f"{'Value function loss:':>{pad}} {value_loss:.4f}",
            f"{'Surrogate loss:':>{pad}} {surrogate_loss:.4f}",
            f"{'Mean decoder loss:':>{pad}} {decoder_loss.item():.4f}",
            f"{'Mean action noise std:':>{pad}} {mean_std:.2f}",
        ]
        if reward_buffer:
            lines.append(f"{'Mean reward:':>{pad}} {statistics.mean(reward_buffer):.2f}")
            lines.append(f"{'Mean episode length:':>{pad}} {statistics.mean(length_buffer):.2f}")
        else:
            lines.append(f"{'Mean reward:':>{pad}} (本批尚无回合结束)")
        # 每个指标一行；episode_stats 已聚合，不会重复打印同名列。
        for key, value in episode_stats.items():
            lines.append(f"{key + ':':>{pad}} {value:.4f}")

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
        torch.save({
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
        # 暴露最近一次的 5 维估计，供 play 端与真值对照（decoder 是部署件，必须能核对精度）。
        self.last_estimate = None

        def policy(raw_obs):
            estimate, self.decoder_hidden = self.decoder(raw_obs, self.decoder_hidden)
            self.last_estimate = estimate
            return self.alg.actor_critic.act_inference(
                augment_actor_observation(raw_obs, estimate))

        return policy
