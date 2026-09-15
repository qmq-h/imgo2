from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl_lab.config import AMPOnPolicyRunnerCfg


def add_amp_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add AMP RSL-RL arguments to the parser."""
    arg_group = parser.add_argument_group("amp_rsl_rl", description="Arguments for AMP RSL-RL agent.")
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    arg_group.add_argument("--resume", action="store_true", default=False, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from or play.")
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )


def parse_amp_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> AMPOnPolicyRunnerCfg:
    """Load and update an AMP RSL-RL config from the Isaac Lab registry."""
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    agent_cfg: AMPOnPolicyRunnerCfg = load_cfg_from_registry(task_name, "amp_rsl_rl_cfg")
    return update_amp_rsl_rl_cfg(agent_cfg, args_cli)


def update_amp_rsl_rl_cfg(agent_cfg: AMPOnPolicyRunnerCfg, args_cli: argparse.Namespace):
    """Update AMP RSL-RL config from non-Hydra CLI arguments."""
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if hasattr(args_cli, "resume") and args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if hasattr(args_cli, "load_run") and args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if hasattr(args_cli, "checkpoint") and args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if hasattr(args_cli, "run_name") and args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if hasattr(args_cli, "experiment_name") and args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if hasattr(args_cli, "logger") and args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    if agent_cfg.logger in {"wandb", "neptune"} and getattr(args_cli, "log_project_name", None):
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    return agent_cfg
