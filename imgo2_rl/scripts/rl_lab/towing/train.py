"""Train the upper towing policy with the repository-owned rl_lab stack."""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train the recurrent upper towing policy.")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, required=True)
# Isaac Lab 的 hydra_task_config(task, agent) 把 --agent 的值**当作完整的注册键**传给
# load_cfg_from_registry（官方入口的默认值就是 "rsl_rl_cfg_entry_point"，不做后缀推导）。
# 本任务的 agent 配置注册在 "rl_lab_cfg_entry_point" 下，所以这里必须传完整键名；
# 传去掉后缀的 "rl_lab" 会报 "Could not find configuration ... 'rl_lab'"（2026-09-22 实遇）。
parser.add_argument("--agent", type=str, default="rl_lab_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
# 覆盖 agent 配置里的 save_interval（默认 100）。长跑按时长/关机时间安排时，
# 间隔过大意味着最后不足一个间隔的进度全部丢失。
parser.add_argument("--save_interval", type=int, default=None)
# 控制台日志紧凑模式：每轮一行摘要（分项仍完整写进 TensorBoard）。
# 默认的详细模式每轮约 25 行，2000 轮会产出约 5 万行。
parser.add_argument("--compact-log", action="store_true", default=False,
                    help="Print one summary line per iteration instead of the full block.")
# 冻结低层：默认取 env cfg 的 amp（行为与之前逐位一致）。要用 PPO@2500 时传
# `--policy ppo --policy_checkpoint <play.py 导出的 policy.pt>`；只传 policy 会退回
# 部署目录里的 ppo 参考策略，不是 2500 那一份。
parser.add_argument("--policy", choices=("amp", "ppo"), default=None,
                    help="冻结低层策略名（默认取 env cfg 的 amp）；ppo 需配合 --policy_checkpoint。")
parser.add_argument("--policy_checkpoint", type=str, default=None,
                    help="覆盖冻结低层权重（play.py 导出的 TorchScript policy.pt）；契约仍按 --policy。")
# 质量域覆盖：默认用 env cfg 的 5–30 kg；做重车专项时用 `--mass-range 25 30`
# （或 `30 30` 固定 30 kg）。范围必须落在训练工作域 [5, 30] kg 内。
parser.add_argument("--mass-range", type=float, nargs=2, default=None,
                    metavar=("MIN", "MAX"),
                    help="覆盖 reset_work_condition.mass_range（kg，默认 5–30）；"
                         "例如 --mass-range 25 30 做重车专项。")
cli_args.add_towing_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.policy_checkpoint is not None and args_cli.policy is None:
    parser.error("--policy_checkpoint 需要同时指定 --policy amp|ppo（否则契约会按默认 amp 取）")
if args_cli.mass_range is not None:
    mass_low, mass_high = args_cli.mass_range
    if not (5.0 <= mass_low <= mass_high <= 30.0):
        parser.error(f"--mass-range 必须满足 5 <= MIN <= MAX <= 30 kg，"
                     f"收到 {mass_low!r} {mass_high!r}")
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import imgo2_rl.tasks  # noqa: F401
from imgo2_rl.tasks.manager_based.towing.utils.policy_cfg import get_policy, with_model_path
from rl_lab.config import TowingOnPolicyRunnerCfg
from rl_lab.runners import TowingOnPolicyRunner
from rl_lab.wrapper import TowingVecEnvWrapper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: TowingOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_towing_cfg(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.save_interval is not None:
        agent_cfg.save_interval = args_cli.save_interval
    if args_cli.mass_range is not None:
        env_cfg.events.reset_work_condition.params["mass_range"] = tuple(args_cli.mass_range)
        print(f"[towing] mass_range override: {tuple(args_cli.mass_range)} kg", flush=True)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device

    # 冻结低层选择：默认 amp（不传参数时与旧命令逐位一致）；PPO@2500 用
    # `--policy ppo --policy_checkpoint <导出的 policy.pt>`。这里先校验权重可读，
    # 免得跑完 AppLauncher/建场景才在 action term 里报错。
    action_cfg = env_cfg.actions.high_level_velocity
    if args_cli.policy is not None:
        action_cfg.policy_name = args_cli.policy
    if args_cli.policy_checkpoint is not None:
        action_cfg.policy_checkpoint = str(Path(args_cli.policy_checkpoint).expanduser().resolve())
    frozen = get_policy(action_cfg.policy_name)
    if action_cfg.policy_checkpoint:
        frozen = with_model_path(frozen, action_cfg.policy_checkpoint)
    elif action_cfg.policy_name == "ppo":
        print("[towing] WARNING: --policy ppo 未给 --policy_checkpoint，使用部署目录里的参考 PPO "
              "权重（sha256 53a57f90…），不是 2500 导出件", flush=True)
    print(f"[towing] frozen low-level: {frozen.name} -> {frozen.model_path}", flush=True)

    log_root = os.path.abspath(os.path.join("logs", "towing_rl_lab", agent_cfg.experiment_name))
    resume_path = None
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if resume_path is not None:
        # 恢复到哪个 run，就把后续 checkpoint 写回同一个目录，使 `iter` 能连续累加。
        # 否则每次恢复都新建时间戳目录，checkpoint 会散落在多个 run 里，而
        # ``get_checkpoint_path`` 只在一个目录内搜索，无法接续。
        # 与 Isaac Lab 官方 `scripts/reinforcement_learning/rsl_rl/train.py` 的做法一致。
        log_dir = os.path.dirname(resume_path)
    else:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            run_name += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root, run_name)
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)

    env = TowingVecEnvWrapper(
        gym.make(args_cli.task, cfg=env_cfg),
        clip_actions=agent_cfg.clip_actions,
    )
    runner = TowingOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.compact_log = bool(args_cli.compact_log)
    if resume_path is not None:
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    # 注意：`learn()` 把 max_iterations 当作**本次还要跑多少轮**，不是「总轮数目标」。
    # 因此恢复时它从 checkpoint 的 `iter` 起再跑 max_iterations 轮（总轮数会超出）。
    # 想恢复到某个总轮数，需自己传 `--max_iterations=<目标总数 - checkpoint.iter>`。
    # 该语义与 Isaac Lab 官方入口（max_iterations 为总目标）不同，属既有行为，本次未改。
    runner.learn(agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
