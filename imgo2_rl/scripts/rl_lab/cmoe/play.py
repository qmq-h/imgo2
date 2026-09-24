# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Play, record and export a checkpoint trained with the local CMoE stack."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Play a checkpoint trained with CMoE.")
parser.add_argument("--video", action="store_true", default=False, help="Record a playback video.")
parser.add_argument("--video_length", type=int, default=200, help="Recorded video length in steps.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="cmoe_rsl_rl_cfg")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument(
    "--scan187",
    action="store_true",
    default=False,
    help="Replay a checkpoint trained with the legacy 187-D height scan (17x11 @ 1.6x1.0 m).",
)
parser.add_argument(
    "--terrain_level",
    type=int,
    default=None,
    help="把回放环境的**课程等级**钉死在 N（0–9）并冻结课程升降级，用于检查指定难度下的地形。"
         "默认 None＝沿用 play 配置（`max_init_terrain_level=5`，即随机落在 0–5 级）。",
)
cli_args.add_cmoe_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import time

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import imgo2_rl.tasks  # noqa: F401
from rl_lab.config import CMoEOnPolicyRunnerCfg
from rl_lab.runners import CMoEOnPolicyRunner
from rl_lab.utils import export_cmoe_policy_as_jit, export_cmoe_policy_as_onnx
from rl_lab.wrapper import CMoEVecEnvWrapper


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: CMoEOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_cmoe_cfg(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.scan187:
        # 2026-09-24 之前用 187 维高度扫描训出的 checkpoint，必须用旧几何才能 load_state_dict
        # （契约 637/235 vs 现在的 527/125）。这里在 env cfg 构造之后覆盖，所以不会触发
        # CMoE_env_cfg 里「必须是 77 条射线」的断言；只影响回放，不影响训练。
        env_cfg.scene.height_scanner.pattern_cfg.size = (1.6, 1.0)
        env_cfg.scene.height_scanner.offset.pos = (0.0, 0.0, 20.0)
        print("[INFO] --scan187: 使用旧几何 17x11=187 维高度扫描（仅用于回放旧 checkpoint）")
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "cmoe", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path,
            agent_cfg.load_run,
            agent_cfg.load_checkpoint,
        )
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir
    print(f"[INFO] Loading CMoE checkpoint: {resume_path}")

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    if args_cli.terrain_level is not None:
        # 2026-09-24：play 默认只从 0–5 级起步（`max_init_terrain_level=5`），单环境又几乎不会晋级，
        # 于是**永远看不到配置区间的上端**（台阶 15 cm / boxes 30 cm 要 level≈9）。这个开关把等级钉死。
        # ⚠️ 本机无 GPU、此路径**未经运行验证**；因此全程 fail-soft：失败只打印告警，回放照常进行。
        try:
            terrain = env.unwrapped.scene.terrain
            origins = getattr(terrain, "terrain_origins", None)
            if origins is None or getattr(terrain, "terrain_levels", None) is None:
                print("[WARN] --terrain_level: 该任务不是课程地形（terrain_type != generator），忽略")
            else:
                level = int(args_cli.terrain_level)
                terrain.terrain_levels[:] = level
                terrain.env_origins[:] = origins[terrain.terrain_levels, terrain.terrain_types]
                # 冻结课程：否则回合结束仍会按判据升降级（一局之内就飘走）
                try:
                    manager = env.unwrapped.curriculum_manager
                    for index, name in enumerate(manager.active_terms):
                        if name == "terrain_levels":
                            manager._term_cfgs[index].func = lambda *args, **kwargs: {}
                    print("[INFO] --terrain_level: 已冻结课程（terrain_levels → no-op）")
                except Exception as exc:  # noqa: BLE001 - 调试开关，失败不影响回放
                    print(f"[WARN] --terrain_level: 课程未冻结（{exc}），等级可能在一局内变化")
                env.unwrapped.reset()  # 让机器人按新的 env_origins 重新出生
                print(f"[INFO] --terrain_level={level}: 已把全部环境的地形等级钉死在 {level}")
        except Exception as exc:  # noqa: BLE001 - 调试开关，失败不影响回放
            print(f"[WARN] --terrain_level 失败（忽略，继续按默认等级回放）：{exc}")
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = CMoEVecEnvWrapper(env, history_steps=agent_cfg.history_steps)
    print(
        "[INFO] CMoE observations: "
        f"policy={env.num_one_step_obs}, history_steps={env.history_steps}, "
        f"terrain={env.num_terrain_obs}, actor_total={env.num_obs}, "
        f"critic={env.num_privileged_obs}, actions={env.num_actions}"
    )

    runner = CMoEOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)

    export_dir = os.path.join(log_dir, "exported")
    export_cmoe_policy_as_jit(runner.alg.actor_critic, export_dir, "policy.pt")
    export_cmoe_policy_as_onnx(runner.alg.actor_critic, export_dir, "policy.onnx")

    observations = env.get_observations()
    dt = env.unwrapped.step_dt
    timestep = 0
    print("[INFO] Starting CMoE playback...")
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(observations)
            observations, _, _, _, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            if timestep >= args_cli.video_length:
                break
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
