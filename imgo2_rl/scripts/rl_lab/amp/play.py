# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an AMP agent trained with rl_lab."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play a checkpoint with AMP RSL-RL agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="amp_rsl_rl_cfg", help="Name of the AMP agent config entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
cli_args.add_amp_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import imgo2_rl.tasks  # noqa: F401
from rl_lab.config import AMPOnPolicyRunnerCfg
from rl_lab.runners import AMPOnPolicyRunner
from rl_lab.wrapper import AmpVecEnvWrapper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: AMPOnPolicyRunnerCfg):
    """Play with AMP RSL-RL agent."""
    agent_cfg = cli_args.update_amp_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "amp_rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = AmpVecEnvWrapper(env, include_history_steps=agent_cfg.include_history_steps)

    print("[INFO] Environment wrapped successfully")
    print(f"[INFO] num_envs: {env.num_envs}")
    print(f"[INFO] num_obs: {env.num_obs}")
    if env.include_history_steps is not None:
        print(f"[INFO] include_history_steps: {env.include_history_steps}")
    if env.num_privileged_obs is not None:
        print(f"[INFO] num_privileged_obs: {env.num_privileged_obs}")
    print(f"[INFO] num_actions: {env.num_actions}")

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = AMPOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    print(f"[INFO] Exporting AMP actor policy to: {export_model_dir}")
    export_policy_as_jit(runner.alg.actor_critic, normalizer=None, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(runner.alg.actor_critic, normalizer=None, path=export_model_dir, filename="policy.onnx")

    dt = env.dt
    obs = env.get_observations()
    timestep = 0

    print("[INFO] Starting playback...")
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, privileged_obs, amp_obs, rewards, dones, infos, reset_env_ids, terminal_amp_states = env.step(actions)

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
