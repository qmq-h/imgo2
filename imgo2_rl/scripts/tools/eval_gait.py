"""AMP/PPO 策略的步态评估：回放 checkpoint 并统计足端接触与关节层面的步态指标。

训练日志只有情节级标量（高度、速度误差、终止率），**没有**足端接触时序或关节轨迹，
因此占空比、步频、相位关系这类指标必须回放采样。本脚本就是做这件事。

用法（在装有 Isaac Lab 的机器上，需要 GPU）：

    cd <工作区根>/imgo2_rl
    python scripts/tools/eval_gait.py \
        --task=Imgo2-basemove-flat-amp-play \
        --checkpoint=logs/amp_rsl_rl/base_move_amp/<run>/model_5000.pt \
        --num_envs=8 --steps=1000 --out docs/gait_eval_amp_5000.json

输出：终端摘要 + JSON（逐足指标 + 原始接触/关节时序可选）。
只用 Isaac Lab 与 numpy，不引入新依赖。

指标口径（每只足、每个环境分别统计后取均值）：
  * duty factor   — 支撑相时间占比 = 1 − 空中时间占比
  * stride freq   — 单位时间落地次数（由 first_contact 事件计数）
  * air/stance    — 平均单次空中/支撑时长（s）
  * swing amp     — 摆动期该腿 shank 关节的角度变化幅度（rad）
  * phase offset  — 相对前左腿（FL）的落地相位（0~1，用于判断 trot/pace/bound）
  * base height   — 基座高度的均值/标准差（与训练日志对照）
  * vel error     — 与指令的线/角速度误差（与训练日志对照，验证回放一致性）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Gait evaluation by replaying a checkpoint.")
parser.add_argument("--task", type=str, required=True, help="Task name, e.g. Imgo2-basemove-flat-amp-play.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt checkpoint.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel envs to sample.")
parser.add_argument("--steps", type=int, default=1000, help="Policy steps to record per env.")
parser.add_argument("--agent", type=str, default=None, help="Agent config entry point (defaults per task).")
parser.add_argument("--out", type=str, default=None, help="Write metrics JSON to this path.")
parser.add_argument("--dump-timeseries", action="store_true", help="Also store raw per-step arrays in the JSON.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import imgo2_rl.tasks  # noqa: F401, E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402


def _resolve_agent_cfg(task: str, agent: str | None):
    """Pick the agent cfg key registered for this task (amp / himloco / rsl_rl)."""
    from gymnasium.envs.registration import registry

    spec = registry[task].kwargs
    for key in ("amp_rsl_rl_cfg", "himloco_rsl_rl_cfg", "rsl_rl_cfg_entry_point"):
        if key in spec:
            entry = spec[key]
            module, _, name = entry.rpartition(":")
            return load_cfg_from_registry(task, key), (agent or name)
    raise RuntimeError(f"任务 {task} 没有注册任何 agent 配置: {list(spec)}")


def main() -> dict:
    env_cfg, _ = __import__("isaaclab_tasks.utils.parse_cfg", fromlist=["parse_env_cfg"]).parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    # play 配置里 num_envs 被写死为 1，这里按命令行覆盖，便于多样本统计
    env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg, agent_name = _resolve_agent_cfg(args_cli.task, args_cli.agent)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = env.unwrapped

    # ---- 载入 checkpoint ----
    ckpt_path = Path(args_cli.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = Path.cwd() / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")

    # 只需要策略网络做推理，因此直接构造 ActorCritic 并载入 state_dict，
    # 不经过 runner（避免依赖判别器/AMP 数据加载）。
    policy = _build_policy(env, agent_cfg)
    state = torch.load(ckpt_path, map_location=args_cli.device, weights_only=False)
    policy.load_state_dict(state["model_state_dict"])
    policy.eval()
    print(f"[eval] 载入 {ckpt_path.name}（iter={state.get('iter', '?')}） actor 输入 {policy.num_actor_obs if hasattr(policy,'num_actor_obs') else '?'}")

    # ---- 传感器与索引 ----
    contact_sensor = env.scene.sensors.get("contact_forces")
    if contact_sensor is None:
        raise RuntimeError("场景里没有 contact_forces 传感器；确认 IMGO2_CFG.spawn.activate_contact_sensors=True")
    foot_names = [n for n in contact_sensor.body_names if "FOOT" in n.upper()]
    foot_ids = [contact_sensor.body_names.index(n) for n in foot_names]
    robot = env.scene["robot"]
    shank_ids, _ = robot.find_joints([".*_shank_joint"])

    # ---- 回放采样 ----
    obs, _ = env.reset()
    rec = {k: [] for k in ("t", "contact", "base_z", "vel_err_xy", "vel_err_yaw", "joint_pos")}
    cmd = None
    dt = env.step_dt
    with torch.inference_mode():
        for _ in range(args_cli.steps):
            actions = policy.act_inference(obs) if hasattr(policy, "act_inference") else policy(obs)[0]
            obs, _, _, _, _ = env.step(actions)
            contact = contact_sensor.data.net_forces_w[:, foot_ids, :].norm(dim=-1) > 1.0
            rec["t"].append(float(env.episode_length_buf.float().mean().item()) * dt)
            rec["contact"].append(contact.cpu().numpy().astype(np.int8))
            rec["base_z"].append(robot.data.root_pos_w[:, 2].cpu().numpy())
            cmd = env.command_manager.get_command("base_velocity")
            lin_err = torch.norm(cmd[:, :2] - robot.data.root_lin_vel_b[:, :2], dim=1)
            ang_err = torch.abs(cmd[:, 2] - robot.data.root_ang_vel_b[:, 2])
            rec["vel_err_xy"].append(lin_err.cpu().numpy())
            rec["vel_err_yaw"].append(ang_err.cpu().numpy())
            rec["joint_pos"].append(robot.data.joint_pos[:, shank_ids].cpu().numpy())

    # ---- 统计 ----
    contact = np.stack(rec["contact"], axis=1)          # [E, T, F]
    t = np.asarray(rec["t"])                            # [T]
    E, T, F = contact.shape
    metrics: dict = {"checkpoint": str(ckpt_path), "iter": int(state.get("iter", -1)),
                     "num_envs": E, "steps": T, "step_dt": dt, "feet": foot_names,
                     "base_height_mean": float(np.nanmean(rec["base_z"])),
                     "base_height_std": float(np.nanstd(rec["base_z"])),
                     "vel_err_xy_mean": float(np.nanmean(rec["vel_err_xy"])),
                     "vel_err_yaw_mean": float(np.nanmean(rec["vel_err_yaw"]))}

    per_foot = {}
    for f, name in enumerate(foot_names):
        c = contact[:, :, f]                            # [E, T]
        # 落地事件（0→1）计数 → 步频
        first = np.zeros_like(c[:, 1:])
        first[c[:, 1:] > c[:, :-1]] = 1
        n_steps = first.sum(axis=1)                     # [E]
        duration = T * dt
        # 单次空中/支撑时长
        air_runs, stance_runs = [], []
        for e in range(E):
            seq = c[e]
            # 简单的游程统计
            idx = np.flatnonzero(np.diff(seq))
            bounds = np.concatenate(([0], idx + 1, [T]))
            for a, b in zip(bounds[:-1], bounds[1:]):
                (stance_runs if seq[a] else air_runs).append((b - a) * dt)
        per_foot[name] = {
            "air_fraction": float(1.0 - c.mean()),
            "duty_factor": float(c.mean()),
            "stride_freq_hz": float(n_steps.mean() / duration),
            "air_time_mean_s": float(np.mean(air_runs)) if air_runs else 0.0,
            "stance_time_mean_s": float(np.mean(stance_runs)) if stance_runs else 0.0,
        }
    # 相位：每只足相对 FL 的落地时刻差 / 周期
    fl = contact[:, :, 0]
    ref_first = np.flatnonzero(np.diff(fl[0]) > 0)
    if len(ref_first) >= 2:
        period = float(np.mean(np.diff(ref_first))) * dt
        for f, name in enumerate(foot_names):
            ff = np.flatnonzero(np.diff(contact[0, :, f]) > 0)
            if len(ff) and len(ref_first):
                off = (ff[0] - ref_first[0]) * dt
                per_foot[name]["phase_offset_vs_FL"] = float((off % period) / period) if period > 0 else None
    metrics["per_foot"] = per_foot

    # shank 摆动幅度（每腿）
    jp = np.stack(rec["joint_pos"], axis=1)             # [E, T, legs]
    metrics["shank_peak_to_peak_rad"] = {
        f"leg{i}": float(np.nanmean(jp[:, :, i].max(axis=1) - jp[:, :, i].min(axis=1)))
        for i in range(jp.shape[2])
    }
    metrics["episode_length_mean_steps"] = float(np.mean(rec["t"]) / dt)

    if args_cli.dump_timeseries:
        metrics["timeseries"] = {"t": t.tolist(), "contact": contact.tolist(),
                                 "base_z": np.stack(rec["base_z"]).tolist()}

    if args_cli.out:
        out = Path(args_cli.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[eval] 已写出 {out}")

    print("\n===== 步态评估摘要 =====")
    print(f"  基座高度 {metrics['base_height_mean']:.4f} ± {metrics['base_height_std']:.4f} m")
    print(f"  速度误差 线 {metrics['vel_err_xy_mean']:.4f} m/s / 角 {metrics['vel_err_yaw_mean']:.4f} rad/s")
    print(f"  {'足':10}{'占空比':>9}{'步频Hz':>9}{'空中s':>9}{'支撑s':>9}{'相位':>8}")
    for name, m in per_foot.items():
        ph = m.get("phase_offset_vs_FL")
        print(f"  {name:10}{m['duty_factor']:9.3f}{m['stride_freq_hz']:9.3f}"
              f"{m['air_time_mean_s']:9.4f}{m['stance_time_mean_s']:9.4f}"
              f"{(f'{ph:.2f}' if ph is not None else '-'):>8}")
    print("  shank 峰峰值(rad): " + ", ".join(f"{k}={v:.3f}" for k, v in metrics["shank_peak_to_peak_rad"].items()))

    env.close()
    return metrics


def _build_policy(env, agent_cfg):
    """构造与训练一致的 actor-critic（只需策略网络做推理）。"""
    from rl_lab.modules import ActorCritic

    policy_cfg = dict(agent_cfg.policy.to_dict()) if hasattr(agent_cfg.policy, "to_dict") else dict(agent_cfg.policy)
    num_actor_obs = env.observation_space["policy"].shape[-1] if isinstance(env.observation_space, gym.spaces.Dict) \
        else env.observation_space.shape[-1]
    num_critic_obs = env.observation_space["critic"].shape[-1] if isinstance(env.observation_space, gym.spaces.Dict) \
        else num_actor_obs
    policy = ActorCritic(num_actor_obs=num_actor_obs, num_critic_obs=num_critic_obs,
                         num_actions=env.action_space.shape[-1], **policy_cfg).to(args_cli.device)
    policy.num_actor_obs = num_actor_obs
    return policy


if __name__ == "__main__":
    main()
    simulation_app.close()
