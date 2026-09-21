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
from dataclasses import dataclass
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
    parser.add_argument("--slack", type=float, default=0.40,
                        help="初始松弛量，m：两挂点初始三维距离 = L0 - slack，机器人要**先走约 slack**"
                             "绳才张紧发力。同时 slack 越大两者初始越近。"
                             "默认 0.40（间距 0.59 m，0.5 m/s 下走 0.8 s 才发力）。"
                             "下限由出生窜动决定：机器人出生时会向前窜一点，松弛不足会让绳在"
                             "**站定阶段**就被拉直（实测 0.05 时出现 73-86 N 猛拽，把小车甩出 0.24 m）")
    parser.add_argument("--stop-at", type=float, default=None,
                        help="阶跃停止：指令阶段走到第 T 秒时把指令归零并保持（计划 P6）。"
                             "缺省则整段保持指令")
    parser.add_argument("--wheel-damping", type=float, default=0.016,
                        help="每个车轮的轴承阻力 b，N·m·s/rad（P2 的候选值之一）")
    parser.add_argument("--spawn-height", type=float, default=None,
                        help="机器人初始高度，m；默认用 towing_env_cfg.ROBOT_SPAWN_HEIGHT_M")
    parser.add_argument("--cart-drop", type=float, default=0.03,
                        help="小车生成时离地高度，m（P1/P2 用 0.03 落定；精确贴地会产生接触自漂）")
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
    if args.stop_at is not None:
        if not math.isfinite(args.stop_at) or args.stop_at <= 0:
            parser.error("--stop-at must be finite and positive")
        if args.stop_at >= args.duration:
            parser.error("--stop-at must be smaller than --duration "
                         "(剩余时间作为指令归零后的滑行段)")
    if args.spawn_height is not None and (not math.isfinite(args.spawn_height) or args.spawn_height <= 0):
        parser.error("--spawn-height must be finite and positive")
    if not math.isfinite(args.cart_drop) or not 0.0 <= args.cart_drop <= 0.1:
        parser.error("--cart-drop must be in [0, 0.1] m")
    return args


def git_info():
    def run(*args):
        result = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": run("rev-parse", "HEAD"), "working_tree": run("status", "--porcelain")}


def cart_x_for_attachment_gap(rope_length: float, slack: float, *, robot_x: float, robot_z: float,
                              cart_height: float, robot_offset, cart_offset) -> float:
    """解出小车 x，使两个挂点的**三维**距离 = L0 − slack（即绳是**松的**）。

    `--slack` 的语义是「初始松弛量」：间距必须**小于** L0，于是张力为 0，机器人先起步、
    绳自然由松到紧。第一版写成了 `L0 + slack`（预张紧）——名字说松、实现是紧，
    5 cm 就对应 200 N 预载，实测直接把机器人拽倒。

    两个挂点的高度不同（机器人 base 在 `robot_z`，小车 base_link 在 `cart_height`），
    所以横向距离要按 `sqrt(target² − dz²)` 解，不能直接拿目标距离当 x 间距
    （第一版漏了这一点，又把刚体原点当成挂点，初始距离报成 1.478 m）。
    """
    target = rope_length - slack
    dz = (robot_z + robot_offset[2]) - (cart_height + cart_offset[2])
    if target * target < dz * dz:
        raise ValueError(
            f"绳长 {rope_length} m 减去松弛 {slack} m 后只有 {target:.3f} m，"
            f"小于两挂点的高差 {abs(dz):.3f} m，几何上不可能")
    horizontal = math.sqrt(target * target - dz * dz)
    # 机器人在前（x 大），小车的挂点在车体 +x ⇒ 机器人挂点 x − 小车挂点 x = horizontal
    return robot_x + robot_offset[0] - cart_offset[0] - horizontal


@dataclass(frozen=True)
class PhaseSchedule:
    """`station` / `tow` / `coast` 三段的步数划分（纯算术，可离线测）。"""

    station_steps: int
    tow_steps: int
    coast_steps: int
    dt: float

    @property
    def total_steps(self) -> int:
        return self.station_steps + self.tow_steps + self.coast_steps

    @property
    def tow_phase_s(self) -> float:
        return self.tow_steps * self.dt

    @property
    def coast_phase_s(self) -> float:
        return self.coast_steps * self.dt

    def phase_of(self, step: int) -> str:
        if step < self.station_steps:
            return "station"
        return "tow" if step - self.station_steps < self.tow_steps else "coast"


def make_schedule(*, settle_steps: int, duration: float, stop_at: float | None,
                  dt: float) -> PhaseSchedule:
    """由时长与 `--stop-at` 算出阶段划分。

    抽成纯函数是为了能离线测：这段算术原本内联在 `main()` 里，结果 `stop_steps` 在
    config 字典里被提前引用、赋值却在后面，实跑直接 `UnboundLocalError`——只有跑仿真
    才会暴露。现在算术在离线测试覆盖之下。
    """
    if settle_steps < 0:
        raise ValueError("station 步数不能为负")
    command_steps = int(round(duration / dt))
    if command_steps <= 0:
        raise ValueError("指令阶段步数为 0")
    if stop_at is None:
        tow_steps, coast_steps = command_steps, 0
    else:
        tow_steps = int(round(stop_at / dt))
        coast_steps = command_steps - tow_steps
    if tow_steps <= 0 or coast_steps < 0:
        raise ValueError(f"非法阶段划分：tow={tow_steps} coast={coast_steps}")
    return PhaseSchedule(station_steps=settle_steps, tow_steps=tow_steps,
                         coast_steps=coast_steps, dt=dt)


def initial_cart_x(rope_length: float, slack: float, *, spawn_height: float,
                   cart_height: float, robot_offset, cart_offset) -> float:
    """机器人在原点出生时的小车初始 x（`cart_x_for_attachment_gap` 的特例）。"""
    return cart_x_for_attachment_gap(rope_length, slack, robot_x=0.0, robot_z=spawn_height,
                                     cart_height=cart_height, robot_offset=robot_offset,
                                     cart_offset=cart_offset)


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

        import csv
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

        cart_cfg, model = make_cart_cfg(output / "usd", drop_height=args.cart_drop)
        cart_attachment = tuple(model["attachment_position_m"])
        robot_attachment = tuple(ROBOT_ATTACHMENT_OFFSET_M)
        cart_cfg.init_state.pos = (
            initial_cart_x(args.rope_length, args.slack, spawn_height=spawn_height,
                           cart_height=model["resting_height_m"],
                           robot_offset=robot_attachment, cart_offset=cart_attachment),
            0.0, model["resting_height_m"] + args.cart_drop)

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
        # 关节集合必须一致，但**顺序必然不同**：PhysX 的 DOF 顺序按运动学树广度优先
        # （全部 hip → 全部 thigh → 全部 shank），而策略的观测/动作按逐腿顺序
        # （AMP 任务显式指定）。所以这里比对集合、再建立置换，而不是直接比顺序。
        asset_perm = policy_cfg.asset_permutation(robot.joint_names)
        policy_to_asset = torch.tensor(asset_perm, dtype=torch.long, device=args.device)
        asset_to_policy = torch.empty_like(policy_to_asset)
        asset_to_policy[policy_to_asset] = torch.arange(policy_cfg.num_joints,
                                                        dtype=torch.long, device=args.device)
        # 用「默认姿态」验证这个置换真的对：资产默认角按置换重排后必须等于契约的默认角
        # （URDF 用正则把 hip/thigh/shank 统一设成 0/0.87/-1.82，所以这是一次强校验）
        reordered_default = [float(v) for v in robot.data.default_joint_pos[0, policy_to_asset]]
        if any(abs(a - b) > 1e-6 for a, b in zip(reordered_default, policy_cfg.default_dof_pos)):
            raise RuntimeError(
                "关节置换核对失败：按置换重排后的模型默认关节角与契约 default_dof_pos 不一致\n"
                f"  重排后: {[round(v, 4) for v in reordered_default]}\n"
                f"  契约  : {[round(v, 4) for v in policy_cfg.default_dof_pos]}")
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
        # 阶段划分必须先于 config 字典算好：上一版把 stop_steps 的赋值放在 config 之后，
        # 字典里提前引用 ⇒ 实跑 UnboundLocalError（纯算术已抽成 make_schedule，离线可测）
        schedule = make_schedule(settle_steps=settle_steps, duration=args.duration,
                                 stop_at=args.stop_at, dt=dt)
        command_steps = schedule.tow_steps + schedule.coast_steps

        gravity_world = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=args.device)
        robot_attach = torch.tensor(robot_attachment, dtype=torch.float32, device=args.device).view(1, 1, 3)
        cart_attach = torch.tensor(cart_attachment, dtype=torch.float32, device=args.device).view(1, 1, 3)
        zero_torque = torch.zeros(1, 1, 3, dtype=torch.float32, device=args.device)

        def link_frame_force(asset, body_id, force_world):
            local = math_utils.quat_apply_inverse(asset.data.body_quat_w[:, body_id], force_world)
            return local.unsqueeze(1)

        def apply_rope_and_resistance(command):
            """按当前状态算绳力与轮阻并写入缓冲（显式欧拉，同 P1/P2）。

            **挂点世界坐标 = 刚体原点 + 旋转后的挂点偏移**。第一版只加了偏移去算
            挂点速度、位置却直接用了刚体原点，于是绳长里混进了机器人与小车的高度差
            （0.35 vs 0.15 m），初始张力被抬到 ~1955 N 直接把机器人拽倒。
            """
            robot_offset_w = math_utils.quat_apply(robot.data.body_quat_w[:, base_id], robot_attach.view(1, 3))
            cart_offset_w = math_utils.quat_apply(cart.data.body_quat_w[:, cart_base_id], cart_attach.view(1, 3))
            robot_p = tuple(float(v) for v in (robot.data.body_pos_w[0, base_id] + robot_offset_w[0]))
            cart_p = tuple(float(v) for v in (cart.data.body_pos_w[0, cart_base_id] + cart_offset_w[0]))
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
                # 关节量必须重排成策略顺序后再进观测（PhysX 顺序 ≠ 策略顺序）
                joint_pos=robot.data.joint_pos[:, policy_to_asset],
                joint_vel=robot.data.joint_vel[:, policy_to_asset])
            out = policy.step(parts)
            # 动作是策略顺序的关节目标，要换回资产顺序才能下发给 Isaac Lab
            robot.set_joint_position_target(out.joint_targets[:, asset_to_policy])
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
            "isaac_lab_joint_order": list(robot.joint_names),
            "policy_joint_order": list(policy_cfg.joint_names),
            "policy_to_asset_permutation": list(asset_perm),
            "rope": {"rest_length_m": args.rope_length, "stiffness_n_per_m": args.stiffness,
                     "damping_ns_per_m": args.damping, "initial_slack_m": args.slack},
            "wheel_damping_nms_per_rad": args.wheel_damping, "user_command_mps": args.velocity,
            "settle_time_s": settle_steps * dt, "duration_s": command_steps * dt,
            "stop_at_s": args.stop_at, "tow_phase_s": schedule.tow_phase_s,
            "coast_phase_s": schedule.coast_phase_s,
            "phase_steps": {"station": schedule.station_steps, "tow": schedule.tow_steps,
                            "coast": schedule.coast_steps},
            "dt_s": dt, "decimation": decimation, "device": args.device,
            "torque_convention": "rope force and wheel torque are computed from the state at the start of each step",
            "git": manifest["git"],
        }

        recorder = TowRecorder(output, config)
        # 阶段：station（站定，指令 0）→ tow（指令 = v_user）→ coast（阶跃归零后滑行）
        # 初始条件不在这里「摆正」，而是由设计保证 + 事后判据检查：
        # 出生高度 0.35 m 时站定段机器人只窜 ~0.1 m，而 slack 0.40 m 让它拉不直绳，
        # 于是小车全程静止在设计位置（实测位移 1e-5 m、vx 1e-5 m/s、张力峰值 0 N）。
        # 曾在这里显式重摆小车并清零速度，实跑证明**有害**：它是按机器人当前位置摆的，
        # 机器人窜了 0.097 m 就把小车往前挪 0.081 m，还附带注入 −0.039 m/s 的速度——
        # 把本来已经正确的初始条件弄坏，并掩盖真正的问题。改为由
        # `rope_taut_during_settle` / `robot_lurches_during_settle` / `settle_load_drift_m`
        # 三条判据守住这条性质。
        for step in range(schedule.total_steps):
            phase = schedule.phase_of(step)
            command = args.velocity if phase == "tow" else 0.0
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
                "phase": phase,
                "time_s": (step + 1) * dt,
                "user_cmd_mps": command,
                "ref_cmd_mps": command,          # P4 还没有 command shaping
                "robot_vx_mps": float(robot.data.root_lin_vel_w[0, 0]),
                "load_vx_mps": float(cart.data.root_lin_vel_w[0, 0]),
                "rope_tension_n": float(state.tension),
                "rope_distance_m": float(state.distance),
                "robot_x_m": float(robot.data.root_pos_w[0, 0]),
                "load_x_m": float(cart.data.root_pos_w[0, 0]),
                "robot_z_m": float(robot.data.root_pos_w[0, 2]),
                "load_z_m": float(cart.data.root_pos_w[0, 2]),
                "body_pitch_rad": pitch,
                "body_pitch_rate_radps": float(robot.data.root_ang_vel_b[0, 1]),
            }
            recorder.append(row)

        recorder.close()
        recorder = None

        # ------------------------------------------------------------ 汇总与判定
        # 判读逻辑放在标准库工具里（与 P1/P2 的 summarize_cart_coast.py 同一模式），
        # 于是新判据可以用**真实轨迹**离线复算：scripts/tools/summarize_tow.py <run_dir>
        sys.path.insert(0, str(RL_ROOT / "scripts/tools"))
        from summarize_tow import summarize_tow
        with (output / "tow.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        summary = summarize_tow(rows, user_command=args.velocity)
        manifest.update(state="completed", valid=summary["valid"], summary=summary)
        json_file(output / "experiment.json", manifest)
        json_file(output / "summary.json", summary)
        print(f"[SUMMARY] v_user={args.velocity:g} robot_vx={summary['steady_robot_vx_mps']:.4f} "
              f"load_vx={summary['steady_load_vx_mps']:.4f} T={summary['steady_tension_n']:.3f}±"
              f"{summary['steady_tension_std_n']:.3f} N (漂移 {summary['tension_drift_ratio']:.1%}) "
              f"d={summary['steady_distance_m']:.4f} m "
              f"发力@+{summary['takeup_time_s']:.2f}s/走了{summary['takeup_robot_travel_m']:.3f}m "
              f"站定小车漂移={summary['settle_load_drift_m']:+.3f} m "
              f"valid={summary['valid']} failures={summary['failures']}", flush=True)
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
