"""P4: frozen locomotion policy tows the passive cart through a virtual rope.

计划 P4「接入你现有 locomotion」：**不改 locomotion 的任何参数**，把训练侧配置
（`assets/imgo2.py` 的 `IMGO2_CFG`）当冻结策略使用，按速度指令驱动机器人，经由 P3 的
单侧绳力拖曳 P2 标定过的小车。

Run with an Isaac Lab Python environment；`--help` 只要标准 Python。

每物理步的顺序与 P1/P2 一致：先按当前状态算力（绳力 + 轮阻）再 `write_data_to_sim()`
→ `sim.step()` → `scene.update(dt)`，因此是显式欧拉，与 P2 的力矩口径相同。

绳的施加方式：`set_external_force_and_torque(positions=...)` 的作用点是**连杆坐标系**，
所以直接把挂点常量传进去，力臂由 PhysX 自己算（不必手写 offset×F，也不用管 CoM 约定）。
力本身要从世界系转到连杆系，用 `quat_apply_inverse`。
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid

RL_ROOT = Path(__file__).resolve().parents[2]
REPO = RL_ROOT.parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="amp", help="冻结底层策略名（见 policy_cfg.POLICIES）")
    parser.add_argument("--velocity", type=float, default=0.5, help="用户速度指令 v_user，m/s")
    parser.add_argument("--duration", type=float, default=5.0, help="指令阶段时长，s（计划 P4 要求 5 s）")
    parser.add_argument("--settle-time", type=float, default=1.0,
                        help="复位后站定时长，s（此阶段指令为 0，再交给速度指令）")
    parser.add_argument("--rope-length", type=float, default=1.0, help="绳长 L0，m")
    parser.add_argument("--stiffness", type=float, default=4000.0, help="绳刚度 k，N/m")
    parser.add_argument("--damping", type=float, default=100.0, help="绳阻尼 c，N·s/m")
    parser.add_argument("--slack", type=float, default=0.05,
                        help="初始松弛量，m（初始挂点间距 = L0 + slack）")
    parser.add_argument("--wheel-damping", type=float, default=0.016,
                        help="每个车轮的轴承阻力 b，N·m·s/rad（P2 的候选值之一）")
    parser.add_argument("--spawn-height", type=float, default=None,
                        help="机器人初始高度，m；默认用 towing_env_cfg.ROBOT_SPAWN_HEIGHT_M")
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--output-dir", type=Path, help="新建的实验目录；绝不覆盖已存在的目录")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    # 严格为正的量
    for key in ("duration", "settle_time", "rope_length", "stiffness", "dt"):
        value = getattr(args, key)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{key.replace('_', '-')} must be finite and positive")
    # 允许为 0 的量：无阻尼绳（c=0）与初始就绷直（slack=0）都是合法配置
    for key in ("damping", "slack"):
        value = getattr(args, key)
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{key.replace('_', '-')} must be finite and nonnegative")
    if args.settle_time <= 0:
        parser.error("--settle-time must be positive (reset contract needs a settle phase)")
    if not math.isfinite(args.velocity) or not 0.0 <= args.velocity <= 2.0:
        parser.error("--velocity must be in [0, 2] m/s")
    if args.velocity == 0.0:
        parser.error("--velocity must be > 0 for a tow run (use the settle phase to stand)")
    if not math.isfinite(args.wheel_damping) or not 0.0 <= args.wheel_damping <= 0.1:
        parser.error("--wheel-damping must be in [0, 0.1] N m s/rad")
    if args.dt > 0.01:
        parser.error("--dt must be <= 0.01 s")
    if args.spawn_height is not None and (not math.isfinite(args.spawn_height) or args.spawn_height <= 0):
        parser.error("--spawn-height must be finite and positive")
    return args


def git_info():
    def run(*args):
        result = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": run("rev-parse", "HEAD"), "working_tree": run("status", "--porcelain")}


def initial_cart_x(rope_length: float, slack: float) -> float:
    """让初始挂点间距 = L0 + slack。

    机器人挂在 base 的 x = -0.16（`ROBOT_ATTACHMENT_OFFSET_M`），小车挂在车体 +0.25
    （`cart.urdf`），机器人在原点 ⇒ 间距 = -cart_x - 0.41。
    """
    return -(rope_length + slack) - 0.41


def main(args):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output_dir or RL_ROOT / "logs/towing/tow_drag" / f"{stamp}_{uuid.uuid4().hex[:8]}").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    def json_file(path, data):
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n",
                        encoding="utf-8")

    manifest = {"state": "starting", "arguments": vars(args), "git": git_info(),
                "python": platform.python_version()}
    json_file(output / "experiment.json", manifest)
    print(f"[INFO] Output: {output}", flush=True)

    application = None
    recorder = None
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(headless=args.headless, device=args.device)
        application = launcher.app

        import torch
        import isaaclab.sim as sim_utils
        import isaaclab.utils.math as math_utils
        from isaaclab.scene import InteractiveScene
        from imgo2_rl.assets.cart import make_cart_cfg, resolve_cart_path
        from imgo2_rl.tasks.manager_based.towing.towing_env_cfg import (
            ROBOT_ATTACHMENT_OFFSET_M, ROBOT_SPAWN_HEIGHT_M, TowSceneCfg)
        from imgo2_rl.tasks.manager_based.towing.mdp.resistance import viscous_resistance
        from imgo2_rl.tasks.manager_based.towing.mdp.rope import point_velocity, rope_state
        from imgo2_rl.tasks.manager_based.towing.utils.low_level_policy import (
            FrozenLowLevelPolicy, parts_from_robot_state)
        from imgo2_rl.tasks.manager_based.towing.utils.policy_cfg import get_policy
        from imgo2_rl.tasks.manager_based.towing.utils.recording import TowRecorder

        policy_cfg = get_policy(args.policy)
        spawn_height = args.spawn_height if args.spawn_height is not None else ROBOT_SPAWN_HEIGHT_M

        cart_cfg, model = make_cart_cfg(output / "usd")
        cart_cfg.init_state.pos = (initial_cart_x(args.rope_length, args.slack), 0.0,
                                   model["resting_height_m"])
        cart_attachment = tuple(model["attachment_position_m"])
        robot_attachment = tuple(ROBOT_ATTACHMENT_OFFSET_M)

        sim_cfg = sim_utils.SimulationCfg(
            dt=args.dt, device=args.device,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.8, restitution=0.0,
                friction_combine_mode="average", restitution_combine_mode="min"))
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view((2.5, 2.5, 1.8), (-0.7, 0.0, 0.2))
        scene_cfg = TowSceneCfg(num_envs=1, env_spacing=6.0, cart=cart_cfg)
        # 机器人初始位姿必须在 **构造 InteractiveScene 之前** 写进配置：场景在构造时就把
        # 资产按 init_state 摆好，之后再改 cfg 不会生效（`--spawn-height` 会被静默忽略）。
        scene_cfg.robot.init_state.pos = (0.0, 0.0, spawn_height)
        scene_cfg.robot.init_state.joint_pos = dict(zip(policy_cfg.joint_names,
                                                        policy_cfg.default_dof_pos))
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        robot, cart, contacts = scene["robot"], scene["cart"], scene["wheel_contacts"]
        dt = sim.get_physics_dt()
        if not math.isclose(dt, args.dt, rel_tol=1e-6):
            raise RuntimeError("Simulator dt differs from requested dt")

        # ---------------------------------------------------------------- 契约核对
        if robot.num_joints != policy_cfg.num_joints:
            raise RuntimeError(f"机器人有 {robot.num_joints} 个关节，契约要求 {policy_cfg.num_joints}")
        if list(robot.joint_names) != list(policy_cfg.joint_names):
            raise RuntimeError(
                "机器人关节顺序与策略契约不一致：\n"
                f"  模型: {list(robot.joint_names)}\n  契约: {list(policy_cfg.joint_names)}")
        base_ids, base_names = robot.find_bodies(["base"])
        if len(base_ids) != 1:
            raise RuntimeError(f"机器人应当只有一个 'base' 刚体，实际 {base_names}")
        base_id = base_ids[0]
        cart_base_ids, cart_base_names = cart.find_bodies(["base_link"])
        if len(cart_base_ids) != 1:
            raise RuntimeError(f"小车应当只有一个 'base_link' 刚体，实际 {cart_base_names}")
        cart_base_id = cart_base_ids[0]
        cart_joint_ids, cart_joint_names = cart.find_joints(list(model["joint_names"]), preserve_order=True)
        contact_ids, contact_names = contacts.find_bodies(list(model["wheel_names"]), preserve_order=True)
        if len(cart_joint_ids) != 4 or cart.num_joints != 4 or len(contact_ids) != 4:
            raise RuntimeError("导入的小车必须有四个关节与四个同名轮接触")

        decimation = max(1, int(round(policy_cfg.control_dt / dt)))
        if not math.isclose(decimation * dt, policy_cfg.control_dt, rel_tol=1e-6):
            raise RuntimeError("policy_cfg.control_dt 必须是物理 dt 的整数倍")
        policy = FrozenLowLevelPolicy(policy_cfg, device=args.device)
        settle_steps = policy.reset()          # reset 契约：last_action 归零 + 站定步数
        settle_steps = max(settle_steps, int(round(args.settle_time / dt)))
        command_steps = int(round(args.duration / dt))

        gravity_world = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=args.device)
        robot_attach = torch.tensor(robot_attachment, dtype=torch.float32, device=args.device).view(1, 1, 3)
        cart_attach = torch.tensor(cart_attachment, dtype=torch.float32, device=args.device).view(1, 1, 3)
        zero_torque = torch.zeros(1, 1, 3, dtype=torch.float32, device=args.device)

        def link_frame_force(asset, body_id, force_world):
            local = math_utils.quat_apply_inverse(asset.data.body_quat_w[:, body_id], force_world)
            return local.unsqueeze(1)

        def apply_rope_and_resistance(command):
            """按当前状态算绳力与轮阻并写入缓冲（显式欧拉，同 P1/P2）。"""
            robot_p = tuple(float(v) for v in robot.data.body_pos_w[0, base_id])
            cart_p = tuple(float(v) for v in cart.data.body_pos_w[0, cart_base_id])
            robot_offset_w = math_utils.quat_apply(robot.data.body_quat_w[:, base_id], robot_attach.view(1, 3))
            cart_offset_w = math_utils.quat_apply(cart.data.body_quat_w[:, cart_base_id], cart_attach.view(1, 3))
            robot_v = point_velocity(tuple(float(v) for v in robot.data.body_lin_vel_w[0, base_id]),
                                     tuple(float(v) for v in robot.data.body_ang_vel_w[0, base_id]),
                                     tuple(float(v) for v in robot_offset_w[0]))
            cart_v = point_velocity(tuple(float(v) for v in cart.data.body_lin_vel_w[0, cart_base_id]),
                                    tuple(float(v) for v in cart.data.body_ang_vel_w[0, cart_base_id]),
                                    tuple(float(v) for v in cart_offset_w[0]))
            state = rope_state(robot_p, cart_p, robot_v, cart_v,
                               rest_length=args.rope_length, stiffness=args.stiffness,
                               damping=args.damping)
            force_robot = torch.tensor(state.force_on_robot, dtype=torch.float32, device=args.device).view(1, 3)
            force_cart = torch.tensor(state.force_on_cart, dtype=torch.float32, device=args.device).view(1, 3)
            robot.set_external_force_and_torque(link_frame_force(robot, base_id, force_robot),
                                                zero_torque, positions=robot_attach, body_ids=base_ids)
            cart.set_external_force_and_torque(link_frame_force(cart, cart_base_id, force_cart),
                                               zero_torque, positions=cart_attach,
                                               body_ids=cart_base_ids)
            effort = torch.zeros_like(cart.data.joint_pos)
            effort[:, cart_joint_ids] = viscous_resistance(
                cart.data.joint_vel[:, cart_joint_ids], args.wheel_damping)
            cart.set_joint_effort_target(effort)
            return state

        def policy_step(command):
            parts = parts_from_robot_state(
                base_ang_vel=robot.data.root_ang_vel_b,
                projected_gravity=math_utils.quat_apply_inverse(robot.data.root_quat_w,
                                                                gravity_world.view(1, 3)),
                velocity_command=torch.tensor([command, 0.0, 0.0], dtype=torch.float32,
                                              device=args.device),
                joint_pos=robot.data.joint_pos,
                joint_vel=robot.data.joint_vel)
            out = policy.step(parts)
            robot.set_joint_position_target(out.joint_targets)
            return out

        config = {
            "policy": args.policy, "model_sha256": hashlib.sha256(policy_cfg.model_path.read_bytes()).hexdigest(),
            "policy_contract": {"num_observations": policy_cfg.num_observations,
                                "observation_terms": list(policy_cfg.observation_terms),
                                "observation_scales": list(policy_cfg.observation_scales),
                                "action_scale": list(policy_cfg.action_scale),
                                "default_dof_pos": list(policy_cfg.default_dof_pos),
                                "control_dt": policy_cfg.control_dt},
            "cart_model_sha256": hashlib.sha256(resolve_cart_path().read_bytes()).hexdigest(),
            "cart_model": model, "robot_attachment_m": list(robot_attachment),
            "cart_attachment_m": list(cart_attachment), "spawn_height_m": spawn_height,
            "rope": {"rest_length_m": args.rope_length, "stiffness_n_per_m": args.stiffness,
                     "damping_ns_per_m": args.damping, "initial_slack_m": args.slack},
            "wheel_damping_nms_per_rad": args.wheel_damping, "user_command_mps": args.velocity,
            "settle_time_s": settle_steps * dt, "duration_s": command_steps * dt,
            "dt_s": dt, "decimation": decimation, "device": args.device,
            "torque_convention": "rope force and wheel torque are computed from the state at the start of each step",
            "git": manifest["git"],
        }

        recorder = TowRecorder(output, config)
        summary = {"state": "running", "settle_samples": 0, "command_samples": 0,
                   "steady_window_s": None}
        started = False
        for step in range(settle_steps + command_steps):
            in_command = step >= settle_steps
            command = args.velocity if in_command else 0.0
            if step % decimation == 0:
                policy_step(command)
            state = apply_rope_and_resistance(command)
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)
            quat = robot.data.root_quat_w[0]
            # roll/pitch 用与 P2 相同的公式（避免引入额外依赖）
            qw, qx, qy, qz = (float(v) for v in quat)
            pitch = math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx))))
            row = {
                "time_s": (step + 1) * dt,
                "user_cmd_mps": command,
                "ref_cmd_mps": command,          # P4 还没有 command shaping
                "robot_vx_mps": float(robot.data.root_lin_vel_w[0, 0]),
                "load_vx_mps": float(cart.data.root_lin_vel_w[0, 0]),
                "rope_tension_n": float(state.tension),
                "rope_distance_m": float(state.distance),
                "robot_x_m": float(robot.data.root_pos_w[0, 0]),
                "load_x_m": float(cart.data.root_pos_w[0, 0]),
                "body_pitch_rad": pitch,
                "body_pitch_rate_radps": float(robot.data.root_ang_vel_b[0, 1]),
            }
            recorder.append(row)
            if in_command:
                summary["command_samples"] += 1
                started = True
            else:
                summary["settle_samples"] += 1

        recorder.close()
        recorder = None
        if not started:
            raise RuntimeError("指令阶段一个样本都没有")

        # ------------------------------------------------------------ 汇总与判定
        import csv
        rows = list(csv.DictReader((output / "tow.csv").open(encoding="utf-8", newline="")))
        command_rows = [r for r in rows if float(r["user_cmd_mps"]) > 0.0]
        window = command_rows[len(command_rows) // 2:]      # 后半段当稳态窗
        def mean(key):
            return sum(float(r[key]) for r in window) / len(window)
        def spread(key):
            values = [float(r[key]) for r in window]
            mu = sum(values) / len(values)
            return (sum((v - mu) ** 2 for v in values) / len(values)) ** 0.5
        tension = [float(r["rope_tension_n"]) for r in command_rows]
        summary.update({
            "state": "completed",
            "steady_window_s": [window[0]["time_s"], window[-1]["time_s"]],
            "steady_robot_vx_mps": mean("robot_vx_mps"),
            "steady_load_vx_mps": mean("load_vx_mps"),
            "steady_speed_gap_mps": mean("robot_vx_mps") - mean("load_vx_mps"),
            "steady_tension_n": mean("rope_tension_n"),
            "steady_tension_std_n": spread("rope_tension_n"),
            "steady_distance_m": mean("rope_distance_m"),
            "mean_pitch_rad": mean("body_pitch_rad"),
            "max_abs_pitch_rad": max(abs(float(r["body_pitch_rad"])) for r in rows),
            "tension_peak_n": max(tension),
            "tension_slack_fraction": sum(1 for t in tension if t == 0.0) / len(tension),
            "final_load_x_m": float(rows[-1]["load_x_m"]),
            "final_robot_x_m": float(rows[-1]["robot_x_m"]),
        })
        # 「稳定拖曳」判据（计划 P4）：两体速度接近、张力进入相对稳定区间。
        failures = []
        if abs(summary["steady_speed_gap_mps"]) > 0.1:
            failures.append("robot_and_load_speeds_differ")
        if summary["steady_tension_std_n"] > 0.25 * max(summary["steady_tension_n"], 1e-6):
            failures.append("tension_not_steady")
        if summary["max_abs_pitch_rad"] > 0.6:
            failures.append("body_pitch_excessive")
        summary["failures"] = failures
        summary["valid"] = not failures
        manifest.update(state="completed", valid=summary["valid"], summary=summary)
        json_file(output / "experiment.json", manifest)
        json_file(output / "summary.json", summary)
        print(f"[SUMMARY] v_user={args.velocity:g} robot_vx={summary['steady_robot_vx_mps']:.4f} "
              f"load_vx={summary['steady_load_vx_mps']:.4f} T={summary['steady_tension_n']:.3f}±"
              f"{summary['steady_tension_std_n']:.3f} N d={summary['steady_distance_m']:.4f} m "
              f"valid={summary['valid']} failures={failures}", flush=True)
        if application is not None:
            application.close()
        return 0 if summary["valid"] else 2
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        json_file(output / "experiment.json", manifest)
        if recorder is not None:
            recorder.close()
        print(f"[FAILED] {type(exc).__name__}: {exc}", flush=True)
        # 不调用 application.close()：见 CART-02，Kit 关停会直接终止进程、把异常与退出码
        # 一起吞掉（实测三次尝试退出码都是 0）。这里显式带码退出，让调用方能可靠判定失败。
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(2)


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
