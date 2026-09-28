# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the repository-local CMoE implementation."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with CMoE.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="cmoe_rsl_rl_cfg", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument(
    "--init_experts_from",
    type=str,
    default=None,
    help="把一份 45 维先验（AMP 训练 checkpoint，或 cmoe/play.py 导出的 TorchScript）零填充装进 CMoE 专家"
         "再开始训练 —— 「步态交给先验」。配合 --task=Imgo2-basemove-rough-cmoe-gaitfree 使用"
         "（那项把手工业步态 shaping 归零）。详见 docs/cmoe_trot_warmstart_2026-09-25.md。",
)
parser.add_argument(
    "--init_experts_mode", type=str, default=None, choices=("all", "first"),
    help="先验装进全部专家（默认 all；初始混合恒等于先验）还是只装第 0 个。",
)
parser.add_argument(
    "--init_experts_std", type=str, default=None, choices=("true", "false"),
    help="是否连先验的噪声 std 一起装（默认 true）。CMoE 默认 1.0 的噪声会盖过先验动作均值。",
)
parser.add_argument(
    "--init_experts_jitter", type=float, default=None,
    help="给专家新增的 112 列（地形/估计器）加 N(0, sigma) 扰动以打破 5 专家对称；默认 0。"
         "实测 sigma=0.01 ⇒ 混合仍≈先验（RMSE 0.028）但专家已分化。",
)
parser.add_argument(
    "--init_experts_critic", type=str, default=None, choices=("true", "false"),
    help="是否连先验的 **critic** 一起装（默认 true）。先验 critic 来自另一套奖励尺度，"
         "搬过来可能让早期 advantage 错配、迅速改写 actor ⇒ 只想要步态时给 false。",
)
parser.add_argument(
    "--init_gate_bias", type=int, default=None, metavar="K",
    help="**v5**：把门控初始偏置到第 K 个专家（配合 --init_experts_mode=first ⇒ 第 0 步的混合动作"
         "≈ 先验，而 5 个专家彼此不同）。不设则用配置默认（None＝不偏置）。",
)
parser.add_argument(
    "--init_gate_bias_margin", type=float, default=None,
    help="门控偏置的 margin（默认 4.0）；越大初始 softmax 越接近 one-hot。",
)
parser.add_argument(
    "--anchor_coef", type=float, default=None,
    help="**v5**：先验锚定的初始权重（0＝关闭）。只对 `--anchor_expert` 一个专家的输出做加权 MSE，"
         "权重＝地形掩码 × 线性衰减。典型 0.2~0.3。",
)
parser.add_argument(
    "--anchor_coef_final", type=float, default=None, help="锚定权重衰减到的终值（典型 0.05~0.1）。",
)
parser.add_argument(
    "--anchor_decay_iters", type=int, default=None, help="锚定权重线性衰减到终值所需轮数（默认 1000）。",
)
parser.add_argument(
    "--anchor_terrain_names", type=str, default=None,
    help="逗号分隔：**只在哪些地形上锚定**（默认 flat）。依据：AMP 先验是平地+地形盲策略，"
         "锚在斜坡/台阶上等于强迫策略忽略地形。",
)
parser.add_argument(
    "--anchor_expert", type=int, default=None, help="锚哪个专家（默认 0＝装先验的那个）。",
)
parser.add_argument(
    "--anchor_target", type=str, default=None, choices=("expert", "mixture", "both"),
    help="锚**谁**的输出：expert＝只锚某个专家；mixture＝锚**最终混合输出**（推荐："
         "「平地上整体必须像 AMP」——只锚专家时门控会绕过它，实测平地滞空从 0.084 掉到 0.018）；"
         "both＝两者都锚（默认 expert）。",
)
# append CMoE CLI arguments
cli_args.add_cmoe_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import shutil
import inspect
import torch
from datetime import datetime

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

import imgo2_rl.tasks  # noqa: F401
from rl_lab.runners import CMoEOnPolicyRunner
from rl_lab.wrapper import CMoEVecEnvWrapper
from rl_lab.config import CMoEOnPolicyRunnerCfg
from rl_lab.utils import export_deploy_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: CMoEOnPolicyRunnerCfg):
    """Train a CMoE agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_cmoe_cfg(agent_cfg, args_cli)
    # 「从已有步态策略起步」：先验路径只走 CLI（logs/ 被 .gitignore 忽略，不能写进配置默认值）。
    # ⚠️ 空串必须报错：`init_experts_from: ''` 在 runner 里判假 ⇒ 会**静默地不装先验**
    # （2026-09-25 `cmoe_gaitfree_amp24500` 实跑就是这么发生的，训练照跑、没人发现）。
    if args_cli.init_experts_from is not None:
        from rl_lab.utils.pretrained_prior import normalize_prior_path

        normalized = normalize_prior_path(args_cli.init_experts_from)
        if normalized is None:
            parser.error("--init_experts_from 传了空串/空白：要么给真实的先验路径，要么不要给这个参数")
        agent_cfg.init_experts_from = normalized
    if args_cli.init_experts_mode is not None:
        agent_cfg.init_experts_mode = args_cli.init_experts_mode
    if args_cli.init_experts_critic is not None:
        agent_cfg.init_experts_critic = args_cli.init_experts_critic == "true"
    if args_cli.init_experts_std is not None:
        agent_cfg.init_experts_std = args_cli.init_experts_std == "true"
    if args_cli.init_experts_jitter is not None:
        agent_cfg.init_experts_jitter = args_cli.init_experts_jitter
    # ---------------- v5：门控偏置 + 先验锚定（只在指定地形上） ----------------
    if args_cli.init_gate_bias is not None:
        agent_cfg.init_gate_bias = args_cli.init_gate_bias
    if args_cli.init_gate_bias_margin is not None:
        agent_cfg.init_gate_bias_margin = args_cli.init_gate_bias_margin
    if args_cli.anchor_coef is not None:
        agent_cfg.anchor_coef = args_cli.anchor_coef
    if args_cli.anchor_coef_final is not None:
        agent_cfg.anchor_coef_final = args_cli.anchor_coef_final
    if args_cli.anchor_decay_iters is not None:
        agent_cfg.anchor_decay_iters = args_cli.anchor_decay_iters
    if args_cli.anchor_expert is not None:
        agent_cfg.anchor_expert = args_cli.anchor_expert
    if args_cli.anchor_target is not None:
        agent_cfg.anchor_target = args_cli.anchor_target
    if args_cli.anchor_terrain_names is not None:
        agent_cfg.anchor_terrain_names = tuple(
            name.strip() for name in args_cli.anchor_terrain_names.split(",") if name.strip()
        )
    if agent_cfg.anchor_coef and not agent_cfg.init_experts_from:
        print("[WARN] 给了 --anchor_coef 但没有 --init_experts_from ⇒ 没有教师可锚，锚损失恒为 0")
    if agent_cfg.init_gate_bias is not None and agent_cfg.init_experts_mode != "first":
        print(f"[WARN] --init_gate_bias={agent_cfg.init_gate_bias} 通常配合 --init_experts_mode=first"
              f"（当前 mode={agent_cfg.init_experts_mode}）：mode=all 时 5 个专家都是先验，"
              "偏置只是让门控初始偏向某一个，效果等价但意义不大")
    if "gaitfree" in (args_cli.task or "") and not agent_cfg.init_experts_from:
        print(
            "[WARN] 任务名带 gaitfree（手工步态项已归零）但**没有**给 --init_experts_from ⇒ "
            "步态目前没有任何约束，策略会自行演化（若这是有意的对照实验，忽略本行）"
        )
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "cmoe", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the log directory for the environment
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    env = CMoEVecEnvWrapper(env, history_steps=agent_cfg.history_steps)

    print(f"[INFO] Environment wrapped successfully")
    print(f"[INFO] num_envs: {env.num_envs}")
    print(f"[INFO] num_one_step_obs: {env.num_one_step_obs}")
    print(f"[INFO] history_steps: {env.history_steps}")
    print(f"[INFO] num_terrain_obs: {env.num_terrain_obs}")
    print(f"[INFO] num_obs (total): {env.num_obs}")
    print(f"[INFO] num_privileged_obs: {env.num_privileged_obs}")
    print(f"[INFO] num_actions: {env.num_actions}")

    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # we use the same logic as Isaac Lab to find the checkpoint
        from isaaclab_tasks.utils import get_checkpoint_path
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    runner = CMoEOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    # load the checkpoint if resuming
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    
    # Preserve the environment-side deployment metadata next to the checkpoint.
    # The complete CMoE network itself is exported by cmoe/play.py.
    export_deploy_cfg(
        env.unwrapped, 
        log_dir,
        history_length=agent_cfg.history_steps - 1,
        use_encoder=True,
    )
     
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )
    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
