# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""播放拖曳上层策略的 checkpoint（仓库自有 rl_lab 实现）。

与 `amp/play.py`、`ppo/play.py` 同一套约定：`--checkpoint` 缺省时按
`logs/towing_rl_lab/<experiment_name>/` 下最新的 run 取。

**重要**：这里用 `act_inference`（确定性，取分布的均值），而训练期采样的是带
`mean_noise_std` 的随机动作。所以本脚本的表现**不等于**训练日志里的奖励——若两者差异很大，
说明训练期的噪声在主导行为，应连同 `Policy/mean_noise_std` 一起判断。

用法（有显示的机器可去掉 `--headless`）：
    bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/rl_lab/towing/play.py \
        --task=Imgo2-towing-upper-rl-lab --num_envs=1 \
        --checkpoint=logs/towing_rl_lab/towing_upper/<run>/model_1000.pt
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play a towing checkpoint with the rl_lab runner.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Imgo2-towing-upper-rl-lab", help="Name of the task.")
# 与 train.py 一致：Isaac Lab 把 --agent 的值当**完整注册键**查表。
parser.add_argument("--agent", type=str, default="rl_lab_cfg_entry_point",
                    help="Agent configuration entry point key in the gym registry.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# 打印每个回合的摘要（终止原因、实际牵引力、间隙），便于在终端核对行为而不是只看画面。
# 注意不能用 --verbose：AppLauncher 已占用它（SimulationApp 从 sys.argv 读取）。
parser.add_argument("--episode-log", action="store_true", default=False,
                    help="Print a summary line for every finished episode.")
# 每 N 步打印一行指令对照。0 = 关闭。
parser.add_argument("--cmd-interval", type=int, default=0,
                    help="Print a command comparison row every N steps for env 0 (0 disables).")
# 回放结束时打印各环境的速度跟踪误差汇总。
parser.add_argument("--cmd-summary", action="store_true", default=False,
                    help="Print per-environment speed-tracking summary when the run ends.")
# 每 N 步打印一行 decoder 估计 vs 真值（0 = 关闭）。decoder 是部署件，必须能核对精度。
parser.add_argument("--decoder-interval", type=int, default=0,
                    help="Print decoder estimate vs ground truth every N steps for env 0.")
parser.add_argument("--decoder-summary", action="store_true", default=False,
                    help="Print decoder prediction-error summary (physical units) at the end.")
# 把汇总写成结构化文件，便于事后核对（默认写在 checkpoint 同级的 play_summary.txt）。
parser.add_argument("--summary-out", type=str, default=None,
                    help="Path of the structured summary file (default: <run>/play_summary.txt).")
parser.add_argument("--no-summary-file", action="store_true", default=False,
                    help="Do not write the structured summary file.")
# play 本身是无限循环；给步数上限便于自动结束（0 = 不限）。
parser.add_argument("--max-steps", type=int, default=0,
                    help="Stop after this many control steps (0 = run until closed).")
# 周期性落盘摘要，避免必须等进程结束才有文件可读。
parser.add_argument("--summary-every-s", type=float, default=30.0,
                    help="Rewrite the summary file every N seconds (0 disables).")
cli_args.add_towing_args(parser)
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
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import imgo2_rl.tasks  # noqa: F401
from rl_lab.config import TowingOnPolicyRunnerCfg
from rl_lab.modules import reset_gru_hidden
from rl_lab.modules.towing_decoder import mass_supervision_weight
from rl_lab.runners import TowingOnPolicyRunner
from rl_lab.wrapper import TowingVecEnvWrapper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: TowingOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_towing_cfg(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "towing_rl_lab", agent_cfg.experiment_name))
    print(f"[INFO] 从目录加载实验：{log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    print(f"[INFO] checkpoint：{resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] 录制回放视频。")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = TowingVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] num_envs={env.num_envs}  num_obs={env.num_obs}  "
          f"num_actions={env.num_actions}  max_episode_length={env.max_episode_length}")

    runner = TowingOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    print(f"[INFO] 已加载 checkpoint（iter={runner.current_learning_iteration}）")

    # 确定性推理：act_inference 取分布均值，不使用训练期的动作噪声。
    policy = runner.get_inference_policy(device=env.device)
    action_term = env.unwrapped.action_manager.get_term("high_level_velocity")

    dt = env.unwrapped.step_dt
    obs = env.get_observations().to(env.device)
    episode_steps = torch.zeros(env.num_envs, device=env.device)
    episode_return = torch.zeros(env.num_envs, device=env.device)
    episode_count = 0
    timestep = 0
    last_summary_t = time.time()
    # 指令对照统计：只统计「命令侧期望速度 > 0」的牵引阶段，避免把 settle/STOP 的零指令混进来。
    track_steps = torch.zeros(env.num_envs, device=env.device)
    err_user = torch.zeros(env.num_envs, device=env.device)      # |实际速度 − 命令期望|
    err_ref = torch.zeros(env.num_envs, device=env.device)       # |实际速度 − 上层积分指令|
    err_cmd = torch.zeros(env.num_envs, device=env.device)       # |上层积分指令 − 命令期望|
    # decoder 精度统计（decoder 直接输出物理量，故误差单位即 m/s、N、kg）
    d_vel = torch.zeros(env.num_envs, device=env.device)
    d_force = torch.zeros(env.num_envs, device=env.device)
    d_mass = torch.zeros(env.num_envs, device=env.device)   # 加权累计（权重 = 质量监督权重）
    d_mass_w = torch.zeros(env.num_envs, device=env.device)
    d_mass_n = torch.zeros(env.num_envs, device=env.device)
    d_n = torch.zeros(env.num_envs, device=env.device)
    if args_cli.decoder_interval > 0:
        print("[dec] 列含义：vxT/vyT=真值速度  vxP/vyP=decoder估计  mT/mP=真值/估计质量(kg)  "
              "|F|T、|F|P=真值/估计牵引力(N)")
        print(f"[dec] {'step':>6} {'vxT':>7} {'vxP':>7} {'vyT':>7} {'vyP':>7} "
              f"{'mT':>6} {'mP':>6} {'|F|T':>7} {'|F|P':>7}")
    if args_cli.cmd_interval > 0:
        print("[cmd] 列含义：t=时刻s  user=命令期望速度  ref=上层积分速度指令  "
              "achieved=实际体速  accel=上层输出的加速度(裁剪后)")
        print(f"[cmd] {'step':>6} {'t(s)':>7} {'user':>8} {'ref':>8} {'achv':>8} "
              f"{'err_cmd':>8} {'err_track':>9} {'accel':>8}")

    def _emit_summary(verbose=True):
        """构造/打印/落盘摘要；异常或 Ctrl+C 时由 finally 调用，也能留下部分数据。

        verbose=False 时只落盘不打印，供周期调用避免刷屏。"""
        # ---------------- 汇总：打印 + 落盘（供事后核对，终端看不到时就靠这个文件）----------------
        import json as _json
        summary = {
            "checkpoint": str(resume_path),
            "num_envs": int(env.num_envs),
            "steps": int(d_n[0].item()),
            "decoder_vel_mae_mps": float((d_vel / d_n.clamp_min(1.0)).mean().item()),
            "decoder_force_mae_N": float((d_force / d_n.clamp_min(1.0)).mean().item()),
            "decoder_mass_mae_kg_weighted": float((d_mass / d_mass_w.clamp_min(1e-6)).mean().item()),
            "decoder_mass_supervised_frac": float((d_mass_n / d_n.clamp_min(1.0)).mean().item()),
            "towing_force_gt_mean_N": float(action_term.towing_force_b.norm(dim=1).mean().item()),
        }
        track = track_steps.clamp_min(1.0)
        summary["track_steps_mean"] = float(track_steps.mean().item())
        summary["err_cmd_mean"] = float((err_cmd / track).mean().item())
        summary["err_track_mean"] = float((err_user / track).mean().item())
        summary["err_low_mean"] = float((err_ref / track).mean().item())

        print("\n[summary] " + _json.dumps(summary, ensure_ascii=False, indent=2))
        if not args_cli.no_summary_file:
            out = args_cli.summary_out or os.path.join(log_dir, "play_summary.json")
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            with open(out, "w", encoding="utf-8") as fh:
                _json.dump(summary, fh, ensure_ascii=False, indent=2)
            print(f"[summary] 已写入 {out}")

    if args_cli.cmd_summary:
        print("\n[cmd] 各环境速度跟踪汇总（仅统计命令侧期望速度 > 0 的步）")
        print(f"[cmd] {'env':>4} {'steps':>7} {'err_cmd':>9} {'err_track':>10} "
              f"{'err_low':>9} {'track/err_cmd':>14}")
        for env_id in range(env.num_envs):
            n = track_steps[env_id].item()
            if n <= 0:
                print(f"[cmd] {env_id:>4}        0        （无牵引阶段采样）")
                continue
            ec = err_cmd[env_id].item() / n
            et = err_user[env_id].item() / n
            el = err_ref[env_id].item() / n
            ratio = (et / ec) if ec > 1.0e-6 else float("nan")
            print(f"[cmd] {env_id:>4} {int(n):>7} {ec:>9.4f} {et:>10.4f} {el:>9.4f} "
                  f"{ratio:>14.2f}")
        print("[cmd] err_cmd  = |上层积分指令 − 命令期望|  （上层有没有把指令积到位）")
        print("[cmd] err_track= |实际速度 − 命令期望|      （最终跟速误差）")
        print("[cmd] err_low  = |实际速度 − 上层积分指令|  （底层执行误差）")
        print("[cmd] track/err_cmd ≫ 1 说明上层指令基本到位、误差主要来自底层执行；")
        print("[cmd] 该比值 ≈ 1 说明上层积分指令本身就没跟上命令期望。")

    print(f"[INFO] 开始回放（确定性策略，step_dt={dt} s）。Ctrl+C 退出。", flush=True)
    try:
      while simulation_app.is_running():
          start_time = time.time()
          with torch.inference_mode():
              actions = policy(obs)
              obs, _privileged, rewards, dones, extras = env.step(actions)
              obs = obs.to(env.device)

              episode_steps += 1
              episode_return += rewards

              # ---- 指令对照：user_command（命令侧期望）/ reference_command（上层积分）/ 实际速度 ----
              user = action_term.user_command
              ref = action_term.reference_command
              accel = action_term.processed_actions
              achieved = action_term._asset.data.root_lin_vel_b[:, :2]
              active = torch.linalg.vector_norm(user[:, :2], dim=1) > 1.0e-4
              if active.any():
                  track_steps += active.float()
                  err_user += torch.where(active, (achieved - user[:, :2]).norm(dim=1), 0.0)
                  err_ref += torch.where(active, (achieved - ref[:, :2]).norm(dim=1), 0.0)
                  err_cmd += torch.where(active, (ref[:, :2] - user[:, :2]).norm(dim=1), 0.0)
              if args_cli.cmd_interval > 0 and timestep % args_cli.cmd_interval == 0:
                  print(f"[cmd] {timestep:>6} {timestep * dt:>7.2f} "
                        f"{user[0, 0].item():>8.3f} {ref[0, 0].item():>8.3f} "
                        f"{achieved[0, 0].item():>8.3f} "
                        f"{(ref[0, :2] - user[0, :2]).norm().item():>8.3f} "
                        f"{(achieved[0, :2] - user[0, :2]).norm().item():>9.3f} "
                        f"{accel[0, 0].item():>8.3f}")
              # ---- decoder 估计 vs 真值 ----
              est = runner.last_estimate
              if est is not None:
                  gt_vel = action_term._asset.data.root_lin_vel_b[:, :2]
                  gt_f = action_term.towing_force_b
                  gt_m = action_term.cart_mass[:, 0]
                  # decoder 直接回归物理量（m/s、kg、N），误差也在物理量上算
                  d_vel += (est[:, :2] - gt_vel).abs().mean(dim=1)
                  d_force += (est[:, 3:5] - gt_f).abs().mean(dim=1)
                  mw = mass_supervision_weight(gt_f, minimum_force=1.0, force_scale=10.0)
                  d_mass += (est[:, 2] - gt_m).abs() * mw
                  d_n += 1.0
                  d_mass_n += (mw > 0).float()
                  d_mass_w += mw
                  if args_cli.decoder_interval > 0 and timestep % args_cli.decoder_interval == 0:
                      print(f"[dec] {timestep:>6} {gt_vel[0,0].item():>7.3f} {est[0,0].item():>7.3f} "
                            f"{gt_vel[0,1].item():>7.3f} {est[0,1].item():>7.3f} "
                            f"{gt_m[0].item():>6.2f} {est[0,2].item():>6.2f} "
                            f"{gt_f[0].norm().item():>7.3f} "
                            f"{est[0,3:5].norm().item():>7.3f}")

              finished = (dones > 0).nonzero(as_tuple=False).flatten()
              if len(finished) > 0:
                  episode_count += len(finished)
                  if args_cli.episode_log:
                      for env_id in finished.tolist():
                          print(f"[episode {episode_count}] env {env_id}: "
                                f"长度 {int(episode_steps[env_id])}  回报 {episode_return[env_id].item():.2f}  "
                                f"牵引力 {action_term.towing_force_b[env_id].norm().item():.3f} N  "
                                f"间隙 {action_term.rope_state[env_id, 0].item():.3f} m  "
                                f"参考速度 {action_term.reference_command[env_id].tolist()}")
                  episode_steps[finished] = 0
                  episode_return[finished] = 0
                  # 回合边界必须清掉三套 GRU 里对应环境的 hidden，否则下个回合会带着
                  # 上一回合的动力学历史（训练侧同一处逻辑，见 towing_on_policy_runner）。
                  runner.decoder_hidden = reset_gru_hidden(runner.decoder_hidden, dones.bool())

          timestep += 1
          if args_cli.summary_every_s > 0 and (time.time() - last_summary_t) >= args_cli.summary_every_s:
              last_summary_t = time.time()
              try:
                  _emit_summary(verbose=False)
              except Exception as exc:  # noqa: BLE001
                  print(f"[summary] 周期落盘失败：{type(exc).__name__}: {exc}", flush=True)
          if args_cli.max_steps and timestep >= args_cli.max_steps:
              print(f"[INFO] 已达 --max-steps={args_cli.max_steps}，退出。", flush=True)
              break
          if args_cli.video and timestep >= args_cli.video_length:
              print(f"[INFO] 视频录满 {args_cli.video_length} 步，退出。")
              break

          sleep_time = dt - (time.time() - start_time)
          if args_cli.real_time and sleep_time > 0:
              time.sleep(sleep_time)

      if args_cli.decoder_summary:
          n = d_n.clamp_min(1.0)
          vel_n = (d_vel / n).mean().item()
          force_n = (d_force / n).mean().item()
          mass_w = (d_mass / d_mass_w.clamp_min(1e-6)).mean().item()
          frac = (d_mass_n / n).mean().item()
          print("\n[dec] decoder 预测误差（确定性推理，全部环境平均）")
          print(f"[dec]   {env.num_envs} 个环境，共 {int(d_n[0].item())} 步")
          print(f"[dec]   速度   MAE : {vel_n:.5f} m/s")
          print(f"[dec]   牵引力 MAE : {force_n:.5f} N")
          print(f"[dec]   质量   MAE : {mass_w:.5f} kg   （仅统计被监督的步，"
                f"仅统计被监督的步）")
          print(f"[dec]   质量被监督的步占比 : {frac*100:.1f}%   （判据：|F_gt| > 1 N）")

    finally:
        # 无论正常结束、抛异常还是 Ctrl+C，都尽力留下已统计的部分数据。
        # 摘要文件是 agent 唯一能读到的通道（终端输出 agent 看不到）。
        try:
            _emit_summary()
        except Exception as exc:  # noqa: BLE001
            print(f"[summary] 生成摘要失败：{type(exc).__name__}: {exc}", flush=True)
        env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
