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
    # 是否对策略的动作噪声 std 施加下限 clamp（min_normalized_std × 关节范围）。
    # 默认 True 保持既有行为；设为 False 则完全交给 PPO 自身的机制
    # （KL 自适应学习率，上限 1e-2 / 下限 1e-5）约束策略变化——这也是标准 PPO 的做法。
    # 注意该 clamp 一向只有下限、没有上限，因此它既不能防止 std 暴涨，又会干扰熵项。
    clamp_noise_std: bool = True
