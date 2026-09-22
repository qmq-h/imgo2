from __future__ import annotations

import argparse
import random


def add_towing_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("towing_rl_lab")
    group.add_argument("--experiment_name", type=str, default=None)
    group.add_argument("--run_name", type=str, default=None)
    group.add_argument("--resume", action="store_true", default=False)
    group.add_argument("--load_run", type=str, default=None)
    group.add_argument("--checkpoint", type=str, default=None)


def update_towing_cfg(agent_cfg, args_cli):
    if args_cli.seed is not None:
        agent_cfg.seed = random.randint(0, 10000) if args_cli.seed == -1 else args_cli.seed
    if args_cli.resume:
        agent_cfg.resume = True
    for cli_name, cfg_name in (
        ("load_run", "load_run"),
        ("checkpoint", "load_checkpoint"),
        ("run_name", "run_name"),
        ("experiment_name", "experiment_name"),
    ):
        value = getattr(args_cli, cli_name, None)
        if value is not None:
            setattr(agent_cfg, cfg_name, value)
    return agent_cfg

