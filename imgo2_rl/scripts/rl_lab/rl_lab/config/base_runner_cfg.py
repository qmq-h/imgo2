from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass


@configclass
class RLLabBaseRunnerCfg:
    """Base configuration shared by rl_lab runners."""

    seed: int = 1
    device: str = "cuda:0"
    num_steps_per_env: int = MISSING
    max_iterations: int = MISSING
    empirical_normalization: bool | None = None
    save_interval: int = MISSING
    experiment_name: str = MISSING
    run_name: str = ""
    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    neptune_project: str = "isaaclab"
    wandb_project: str = "isaaclab"
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
