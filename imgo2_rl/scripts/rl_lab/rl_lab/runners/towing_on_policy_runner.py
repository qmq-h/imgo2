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
                    if "episode" in infos:
                        episode_infos.append(infos["episode"])
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

            if self.writer is not None:
                self._log(
                    iteration, total_iter, collection_time, learn_time,
                    value_loss, surrogate_loss, decoder_loss,
                    reward_buffer, length_buffer, episode_infos,
                )
            if self.log_dir is not None and iteration % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{iteration}.pt"))

        self.current_learning_iteration = total_iter
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{total_iter}.pt"))

    def _log(
        self, iteration, total_iter, collection_time, learn_time,
        value_loss, surrogate_loss, decoder_loss,
        reward_buffer, length_buffer, episode_infos,
    ):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += collection_time + learn_time
        self.writer.add_scalar("Loss/value_function", value_loss, iteration)
        self.writer.add_scalar("Loss/surrogate", surrogate_loss, iteration)
        self.writer.add_scalar("Loss/decoder", decoder_loss.item(), iteration)
        self.writer.add_scalar("Policy/mean_noise_std", self.alg.actor_critic.std.mean().item(), iteration)
        self.writer.add_scalar("Perf/collection_time", collection_time, iteration)
        self.writer.add_scalar("Perf/learning_time", learn_time, iteration)
        if reward_buffer:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(reward_buffer), iteration)
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(length_buffer), iteration)
        for info in episode_infos:
            for key, value in info.items():
                value = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
                self.writer.add_scalar(f"Episode/{key}", value.float().mean().item(), iteration)
        print(
            f"iteration {iteration}/{total_iter} | reward "
            f"{statistics.mean(reward_buffer) if reward_buffer else 0.0:.3f} | "
            f"value {value_loss:.4f} | policy {surrogate_loss:.4f} | "
            f"decoder {decoder_loss.item():.4f}"
        )

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

        def policy(raw_obs):
            estimate, self.decoder_hidden = self.decoder(raw_obs, self.decoder_hidden)
            return self.alg.actor_critic.act_inference(
                augment_actor_observation(raw_obs, estimate))

        return policy
