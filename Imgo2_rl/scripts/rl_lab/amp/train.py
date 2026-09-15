# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train an AMP agent with rl_lab."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train an RL agent with AMP RSL-RL.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="amp_rsl_rl_cfg", help="Name of the AMP agent config entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL policy training iterations.")
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs.")
cli_args.add_amp_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import inspect
import os
import shutil
from datetime import datetime

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import imgo2_rl.tasks  # noqa: F401
from rl_lab.config import AMPOnPolicyRunnerCfg
from rl_lab.runners import AMPOnPolicyRunner
from rl_lab.utils import export_deploy_cfg
from rl_lab.wrapper import AmpVecEnvWrapper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: AMPOnPolicyRunnerCfg):
    """Train with AMP RSL-RL agent."""
    agent_cfg = cli_args.update_amp_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    log_root_path = os.path.abspath(os.path.join("logs", "amp_rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    env_cfg.log_dir = log_dir

    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = AmpVecEnvWrapper(env, include_history_steps=agent_cfg.include_history_steps)

    print("[INFO] Environment wrapped successfully")
    print(f"[INFO] num_envs: {env.num_envs}")
    print(f"[INFO] num_obs: {env.num_obs}")
    if env.include_history_steps is not None:
        print(f"[INFO] include_history_steps: {env.include_history_steps}")
    if env.num_privileged_obs is not None:
        print(f"[INFO] num_privileged_obs: {env.num_privileged_obs}")
    print(f"[INFO] num_actions: {env.num_actions}")

    runner = AMPOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    export_deploy_cfg(
        env.unwrapped,
        log_dir,
        history_length=(agent_cfg.include_history_steps or 1) - 1,
        use_encoder=False,
    )

    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
