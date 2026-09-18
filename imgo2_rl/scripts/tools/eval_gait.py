"""AMP/PPO 策略的步态评估：回放 checkpoint 并统计足端接触与关节层面的步态指标。

训练日志只有情节级标量（高度、速度误差、终止率），**没有**足端接触时序或关节轨迹，
因此占空比、步频、相位关系这类指标必须回放采样。本脚本就是做这件事。

用法（在装有 Isaac Lab 的机器上，需要 GPU）：

    cd <工作区根>/imgo2_rl
    python scripts/tools/eval_gait.py \
        --task=Imgo2-basemove-flat-amp-play \
        --checkpoint=logs/amp_rsl_rl/base_move_amp/<run>/model_24500.pt \
        --num_envs=8 --steps=1000 --warmup=50 --out ../docs/gait_eval_amp_24500.json

输出：终端摘要 + JSON（逐足指标 + 原始接触/关节时序可选）。
**统计口径全部在 `gait_metrics.py`**（纯 numpy，可离线回归，见 `tests/test_gait_metrics.py`），
本脚本只负责采样与打印。

指标口径（每只足、每个环境分别统计后再聚合；聚合顺序也必须自洽）：
  * duty factor   — 支撑相时间占比 = 1 − 空中时间占比（时间加权）
  * stride freq   — 单位时间落地次数（0→1 上升沿计数），另有落地间隔均值 `step_period_s`
  * air/stance    — 平均单次空中/支撑时长（s）
  * phase offset  — 相对前左腿（FL）落地栅格的相位（0~1 周期）+ 圆周集中度 R；
                    用**全部**落地事件、**全部**环境做圆周平均，避免单次抖动带偏
  * stride/lift   — 足端在 base 系的 x 行程与 z 抬脚高度（峰峰），与参考基线同口径
  * base height / vel error — 与训练日志对照，验证回放一致性
  * warmup        — 前 N 步（默认 50，即 1 s）只用于落地，不计入统计；
                    出生在 0.35 m、稳态 0.31 m，开头那段下落会污染峰峰类指标
  * 重置计数      — 回放期间若环境被终止重置，逐足峰峰会被打断污染，摘要里显式报出
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
parser.add_argument("--warmup", type=int, default=50,
                    help="Steps recorded but excluded from the statistics (startup transient, default 50 = 1 s).")
parser.add_argument("--command_vx", type=float, default=None,
                    help="Override the play cfg's fixed forward command (m/s). The play cfg hard-codes "
                         "1.0; use this to sweep 0.3/0.6/0.9 and compare against the reference baseline "
                         "inside the range the recorded motions actually cover.")
parser.add_argument("--command_yaw", type=float, default=None,
                    help="Override the play cfg's fixed yaw-rate command (rad/s).")
parser.add_argument("--agent", type=str, default=None, help="Agent config entry point (defaults per task).")
parser.add_argument("--out", type=str, default=None, help="Write metrics JSON to this path.")
parser.add_argument("--dump-timeseries", action="store_true", help="Also store raw per-step arrays in the JSON.")
parser.add_argument("--dump-npz", type=str, default=None,
                    help="Also save ALL raw arrays (含 43 维 AMP 观测) to this .npz — 有了它，"
                         "判别器/步态分析可以完全离线做（配合 scripts/tools/probe_amp_discriminator.py），"
                         "不必再占用 GPU 回放。建议写到被忽略的 logs/ 下。")
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
from isaaclab.utils import math as math_utils  # noqa: E402
# 指标口径全在纯 numpy 模块里，便于在没有 GPU 的机器上回归（tests/test_gait_metrics.py）。
# sys.path[0] 就是本文件所在目录，直接 import 即可。
from gait_metrics import build_report  # noqa: E402


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
    # parse_env_cfg 直接返回 cfg（不是元组）；早期写成 `env_cfg, _ = ...` 会报
    # "cannot unpack non-iterable ... object"。
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    # play 配置里 num_envs 被写死为 1，这里按命令行覆盖，便于多样本统计
    env_cfg.scene.num_envs = args_cli.num_envs
    # 与 scripts/rl_lab/amp/play.py:73-74 保持一致：agent_cfg 的 device 必须跟着 --device 走，
    # 否则 `--device=cpu` 时环境在 CPU、策略被 .to(cuda:0)，会在 runner 里报
    # "No CUDA GPUs are available"（这也是本脚本能被用来做 CPU 冒烟测试的前提）。
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    agent_cfg, agent_name = _resolve_agent_cfg(args_cli.task, args_cli.agent)
    agent_cfg.device = env_cfg.sim.device

    # 指令覆盖：play 配置把指令钉死成 1.0 m/s / 0 / 0，但录制参考只覆盖到 ~0.84 m/s，
    # 因此需要在数据覆盖内扫速才能分清「风格权重不足」与「指令超出数据覆盖」。
    if args_cli.command_vx is not None:
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.command_vx, args_cli.command_vx)
    if args_cli.command_yaw is not None:
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (args_cli.command_yaw, args_cli.command_yaw)
    cmd_lin_vel_x = tuple(env_cfg.commands.base_velocity.ranges.lin_vel_x)
    cmd_ang_vel_z = tuple(env_cfg.commands.base_velocity.ranges.ang_vel_z)
    print(f"[eval] 指令 lin_vel_x={cmd_lin_vel_x} ang_vel_z={cmd_ang_vel_z}")
    # 必须包 AmpVecEnvWrapper（与 train.py:96 / play.py:98 一致）：
    # num_privileged_obs / num_obs / obs_history_buf 都定义在 wrapper 上，
    # 用 .unwrapped 会得到 AmpManagerBasedRLEnv，runner 构造时即 AttributeError。
    from rl_lab.wrapper import AmpVecEnvWrapper

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = AmpVecEnvWrapper(env, include_history_steps=agent_cfg.include_history_steps)
    print("[eval] 环境已包装")

    # ---- 载入 checkpoint ----
    ckpt_path = Path(args_cli.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = Path.cwd() / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")

    # 走与 scripts/rl_lab/amp/play.py 完全相同的加载路径（AMPOnPolicyRunner.load），
    # 避免自拼 ActorCritic 时把维度/字段弄错。runner.load 也会恢复 iter。
    from rl_lab.runners import AMPOnPolicyRunner

    runner = AMPOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    state = torch.load(ckpt_path, map_location=args_cli.device, weights_only=False)
    runner.load(ckpt_path)
    print(f"[eval] 载入 {ckpt_path.name}（iter={state.get('iter', '?')}）")
    policy = runner.get_inference_policy(device=env.device)

    # ---- 传感器与索引 ----
    base_env = env.unwrapped          # AmpManagerBasedRLEnv：scene / command_manager 在这里
    contact_sensor = base_env.scene.sensors.get("contact_forces")
    if contact_sensor is None:
        raise RuntimeError("场景里没有 contact_forces 传感器；确认 IMGO2_CFG.spawn.activate_contact_sensors=True")
    # 足端索引：直接从 **机器人本体** 取，最可靠。
    # 接触传感器的 body_names 由 prim_path=".../Robot/.*" 展开而来，顺序取决于 USD 树，
    # 不保证等于 FL,FR,RL,RR；而 contact_forces.data.net_forces_w 的最后一维又必须按
    # 传感器自己的顺序索引 —— 因此两者分别解析：
    #   * 步幅/抬脚用 robot.body_names（与 robot.data.body_pos_w 同序）
    #   * 接触用 contact_sensor.find_bodies（与 net_forces_w 同序）
    FOOT_ORDER = ["FL_FOOT", "FR_FOOT", "RL_FOOT", "RR_FOOT"]
    robot = base_env.scene["robot"]
    robot_foot_ids = robot.find_bodies(FOOT_ORDER, preserve_order=True)[0]
    foot_names = [robot.body_names[i] for i in robot_foot_ids]
    sensor_foot_ids = contact_sensor.find_bodies(FOOT_ORDER, preserve_order=True)[0]
    sensor_feet = [contact_sensor.body_names[i] for i in sensor_foot_ids]
    print(f"[eval] 机器人足端顺序 : {foot_names}")
    print(f"[eval] 接触传感器顺序 : {sensor_feet}")
    if foot_names != FOOT_ORDER or sensor_feet != FOOT_ORDER:
        print(f"[eval] 注意: 解析顺序与声明 {FOOT_ORDER} 不同，已按解析结果索引各自的数据")
    joint_ids, _ = robot.find_joints(list(base_env.cfg.joint_names), preserve_order=True)

    # ---- 回放采样 ----
    obs, _ = env.reset()
    rec = {k: [] for k in ("t", "contact", "base_z", "vel_err_xy", "vel_err_yaw",
                           "joint_pos", "quat", "foot_pos_b", "ep_len", "done",
                           "amp_obs", "cmd")}
    dt = base_env.step_dt
    with torch.inference_mode():
        for _ in range(args_cli.steps):
            # policy 是 get_inference_policy 返回的 callable（= actor 的动作均值，无采样噪声）
            actions = policy(obs)
            obs, _priv, _amp, _rew, _dones, _infos, _reset_ids, _term = env.step(actions)
            # 策略侧 43 维 AMP 观测（判别器的输入域）：落盘后即可离线复现「风格项在看什么」
            rec["amp_obs"].append(_amp.cpu().numpy())
            contact = contact_sensor.data.net_forces_w[:, sensor_foot_ids, :].norm(dim=-1) > 1.0
            rec["t"].append(float(env.episode_length_buf.float().mean().item()) * dt)
            rec["ep_len"].append(env.episode_length_buf.cpu().numpy())
            rec["done"].append(_dones.cpu().numpy().astype(np.int8))
            rec["contact"].append(contact.cpu().numpy().astype(np.int8))
            rec["base_z"].append(robot.data.root_pos_w[:, 2].cpu().numpy())
            rec["quat"].append(robot.data.root_quat_w.cpu().numpy())   # wxyz
            # 足端在 base 系的位置：与 amp_foot_pos_base 同一口径（世界位置减 root 再逆旋）
            fp_w = robot.data.body_pos_w[:, robot_foot_ids, :]
            fp_rel = fp_w - robot.data.root_pos_w[:, :3].unsqueeze(1)
            rq = robot.data.root_quat_w.unsqueeze(1).expand(-1, fp_rel.shape[1], -1)
            rec["foot_pos_b"].append(
                math_utils.quat_apply_inverse(rq, fp_rel).cpu().numpy())   # [E, 4, 3]
            cmd = base_env.command_manager.get_command("base_velocity")
            rec["cmd"].append(cmd.cpu().numpy())
            lin_err = torch.norm(cmd[:, :2] - robot.data.root_lin_vel_b[:, :2], dim=1)
            ang_err = torch.abs(cmd[:, 2] - robot.data.root_ang_vel_b[:, 2])
            rec["vel_err_xy"].append(lin_err.cpu().numpy())
            rec["vel_err_yaw"].append(ang_err.cpu().numpy())
            rec["joint_pos"].append(robot.data.joint_pos[:, joint_ids].cpu().numpy())

    # ---- 统计 ----
    # 丢弃前 warmup 步（出生在 0.35 m、稳态 ~0.31 m，开头下落会污染峰峰类指标），
    # 再由 gait_metrics.build_report 统一汇总（口径与聚合都在那个纯 numpy 模块里，
    # 配套 tests/test_gait_metrics.py，可在没有 GPU 的机器上回归）。
    metrics = build_report(
        # **时间轴一律在 0**（build_report 会严格校验，写反会直接报错）：
        # 曾经把 contact/joint_pos 用 axis=1 堆叠成 [E,T,...]，于是关节峰峰值被算成
        # 「同一时刻跨环境的散布」≈0，12 个关节全打印 0.000。
        contact=np.stack(rec["contact"]),              # [T, E, F]
        base_z=np.stack(rec["base_z"]),                # [T, E]
        vel_err_xy=np.stack(rec["vel_err_xy"]),
        vel_err_yaw=np.stack(rec["vel_err_yaw"]),
        quat=np.stack(rec["quat"]),                    # [T, E, 4] wxyz
        foot_pos_b=np.stack(rec["foot_pos_b"]),        # [T, E, 4, 3]
        joint_pos=np.stack(rec["joint_pos"]),          # [T, E, 12]
        ep_len=np.stack(rec["ep_len"]),                # [T, E]
        done=np.stack(rec["done"]),                    # [T, E]
        dt=dt, foot_names=foot_names, checkpoint=str(ckpt_path),
        iteration=int(state.get("iter", -1)), warmup=int(args_cli.warmup),
        t_series=np.asarray(rec["t"]), dump_timeseries=bool(args_cli.dump_timeseries),
        meta={"cmd_lin_vel_x": cmd_lin_vel_x, "cmd_ang_vel_z": cmd_ang_vel_z},
    )
    per_foot = metrics["per_foot"]
    E, T = metrics["num_envs"], metrics["steps"]
    warmup = metrics["warmup_steps"]

    if args_cli.out:
        out = Path(args_cli.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[eval] 已写出 {out}")

    if args_cli.dump_npz:
        npz = Path(args_cli.dump_npz)
        npz.parent.mkdir(parents=True, exist_ok=True)
        # 全部原始数组（时间轴在 0），供离线做判别器/步态分析：
        #   amp_obs 是策略侧 43 维 AMP 观测 → 判别器探针的输入
        #   joint_pos / foot_pos_b / quat / contact 用于重算任何指标，不必再跑 GPU
        np.savez_compressed(
            npz,
            **{k: np.stack(v) for k, v in rec.items()},
            cmd_lin_vel_x=np.asarray(cmd_lin_vel_x), cmd_ang_vel_z=np.asarray(cmd_ang_vel_z),
            iter=np.asarray(int(state.get("iter", -1))),
            checkpoint=np.asarray(str(ckpt_path)),
        )
        print(f"[eval] 已写出原始数组 {npz}（含 amp_obs，可用于离线判别器分析）")

    resets = metrics["resets_per_env"]
    print("\n===== 步态评估摘要 =====")
    print(f"  指令 vx={metrics.get('cmd_lin_vel_x')} yaw_rate={metrics.get('cmd_ang_vel_z')}"
          f" | 回放 {E} 环境 × {T} 步（丢弃前 {warmup} 步）")
    if metrics["resets_total"]:
        msg = (f"  重置 {metrics['resets_total']} 次（{metrics['envs_with_reset']}/{E} 个环境）："
               f"末步 {metrics['resets_at_last_step']} 次 / 窗口内 {metrics['resets_before_last_step']} 次")
        if metrics["resets_before_last_step"] == 0 and metrics["resets_at_last_step"] >= E:
            msg += " ⇒ 全是片长到期的情节超时（每个环境各一次），不污染窗口统计"
        else:
            msg += f" ⇒ **窗口内发生过真重置**，逐足峰峰/步频会被截断，逐环境明细 {resets} 需人工判读"
        print(msg)
    print(f"  基座高度 {metrics['base_height_mean']:.4f} ± {metrics['base_height_std']:.4f} m"
          f" | 末步情节缓冲 {metrics['episode_length_final_steps_mean']:.1f} 步"
          f"（{metrics['steps_recorded']} 步片长，末步超时后归 0）")
    print(f"  速度误差 线 {metrics['vel_err_xy_mean']:.4f} m/s / 角 {metrics['vel_err_yaw_mean']:.4f} rad/s")
    print(f"  机身姿态 pitch 均值 {metrics['body_pitch_deg_mean']:+.2f}° RMS {metrics['body_pitch_deg_rms']:.2f}° "
          f"峰峰 {metrics['body_pitch_deg_ptp']:.2f}° | roll 均值 {metrics['body_roll_deg_mean']:+.2f}° RMS {metrics['body_roll_deg_rms']:.2f}°")
    yaw_line = f"           yaw 峰峰 {metrics['body_yaw_drift_deg']:.2f}°"
    if "body_yaw_drift_rate_deg_s" in metrics:
        yaw_line += f"（累积漂移率 {metrics['body_yaw_drift_rate_deg_s']:+.3f} °/s）"
    print(yaw_line + f" | 判定为恒定倾斜: {metrics['body_pitch_is_constant_lean']}")
    print(f"  步态周期 {metrics['gait_period_s']:.4f} s（参考 0.600 s / 1.67 Hz）"
          f" | 参考值取自 docs/gait_reference_baseline.json")
    print(f"  {'足':10}{'占空比':>8}{'步频Hz':>8}{'步幅x_m':>9}{'抬脚z_m':>9}{'相位°':>8}{'集中度':>8}")
    for name, m in per_foot.items():
        ph = m.get("phase_deg_vs_FL")
        conc = m.get("phase_concentration")
        print(f"  {name:10}{m['duty_factor']:8.3f}{m['stride_freq_hz']:8.3f}"
              f"{m['foot_stride_x_m']:9.3f}{m['foot_lift_z_m']:9.3f}"
              f"{(f'{ph:+.1f}' if ph is not None else '-'):>8}"
              f"{(f'{conc:.2f}' if conc is not None else '-'):>8}")
    print("  参考基线: 步频 1.67 Hz | 步幅x 0.215(0.6m/s)/0.268(0.9)/0.366(1.2) | 抬脚z 0.083/0.112/0.121")
    print("  trot 判据: FL 与 FR 相位应相差 180°；集中度越接近 1 落地越规整")
    print("  关节峰峰值(rad，每腿 hip/thigh/shank 按序): "
          + ", ".join(f"{k}={v:.3f}" for k, v in metrics["joint_peak_to_peak_rad"].items()))

    env.close()
    return metrics


if __name__ == "__main__":
    main()
    simulation_app.close()
