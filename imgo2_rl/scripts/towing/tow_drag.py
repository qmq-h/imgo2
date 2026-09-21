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
    parser.add_argument("--rope-length", type=float, default=0.8,
                        help="绳长 L0，m。默认 0.8：0.5 m/s 下停车后小车滑行约 0.52 m，"
                             "留 0.28 m 余量不会追到机器人（L0=1.0 时间距更大但看不出追尾趋势）。"
                             "注意 L0 必须与速度和轮阻一起定：v=1.0 m/s、b=0.016 时滑行 1.06 m，"
                             "L0=1.0 就已经会撞（启动时会打印预测）")
    parser.add_argument("--rope-model", nargs="+", choices=("compliant", "inextensible"),
                        default=["compliant"],
                        help="绳索模型：compliant = 单边弹簧-阻尼（P3/P4 已有，靠伸长储能）；"
                             "inextensible = 单边距离约束（不可伸长，绷直靠约束冲量，"
                             "**不是**把 k 调大）。默认 compliant，现有实验不受影响。")
    parser.add_argument("--stiffness", type=float, default=4000.0,
                        help="绳刚度 k，N/m（仅 compliant 用）")
    parser.add_argument("--damping", type=float, default=100.0,
                        help="绳阻尼 c，N·s/m（仅 compliant 用）")
    parser.add_argument("--position-gain", type=float, default=0.2,
                        help="仅 inextensible：约束违反量的回拉增益 β（0~1，0.2 常见）")
    parser.add_argument("--max-correction-rate", type=float, default=0.2,
                        help="仅 inextensible：回拉相对速度上限，m/s（夹住深穿透时的猛拉；"
                             "每步力 ≈ J/dt，所以这个上限直接决定最大回拉力）")
    parser.add_argument("--slack", type=float, default=0.40,
                        help="初始松弛量，m：两挂点初始三维距离 = L0 - slack，机器人要**先走约 slack**"
                             "绳才张紧发力。同时 slack 越大两者初始越近。"
                             "默认 0.40（间距 0.59 m，0.5 m/s 下走 0.8 s 才发力）。"
                             "下限由出生窜动决定：机器人出生时会向前窜一点，松弛不足会让绳在"
                             "**站定阶段**就被拉直（实测 0.05 时出现 73-86 N 猛拽，把小车甩出 0.24 m）")
    parser.add_argument("--stop-at", type=float, default=None,
                        help="阶跃停止：指令阶段走到第 T 秒时把指令归零并保持（计划 P6）。"
                             "缺省则整段保持指令")
    parser.add_argument("--wheel-damping", type=float, nargs="+", default=[0.016],
                        help="每个车轮的轴承阻力 b，N·m·s/rad（P2 的候选值之一）。"
                             "**可一次给多个**，与 `--cart-mass` 做笛卡尔积，在同一进程内"
                             "逐 case 扫描（省掉每个组合重启一次 Isaac Sim）")
    parser.add_argument("--spawn-height", type=float, default=None,
                        help="机器人初始高度，m；默认用 towing_env_cfg.ROBOT_SPAWN_HEIGHT_M")
    parser.add_argument("--cart-mass", type=float, nargs="+", default=[None],
                        help="小车的目标总质量，kg（缺省用 URDF 的名义 10 kg）。"
                             "**质量与惯量按同一比例一起缩放**（AGENTS.md：只改质量不改惯量"
                             "会造成模型不自洽）⇒ 几何与质心位置不变、轮半径不变。"
                             "注意 m_eff 随质量线性增长，滑行距离也随之增长：默认 L0=0.8、"
                             "b=0.016 时 15 kg 只剩 +0.02 m 余量、20 kg 起会追到机器人")
    parser.add_argument("--ground-friction", type=float, default=0.8,
                        help="地面静/动摩擦系数（两者取同值），默认 0.8 与 P1/P2 一致。"
                             "注意：**轮子滚动时摩擦不耗散能量**，停车距离由轮轴阻力 b 决定，"
                             "所以改摩擦主要影响起步/打滑等瞬态，而不是滑行距离")
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
    for damping in args.wheel_damping:
        if not math.isfinite(damping) or not 0.0 <= damping <= 0.1:
            parser.error("--wheel-damping values must be in [0, 0.1] N m s/rad")
    if args.dt > 0.01:
        parser.error("--dt must be <= 0.01 s")
    if args.stop_at is not None:
        if not math.isfinite(args.stop_at) or args.stop_at <= 0:
            parser.error("--stop-at must be finite and positive")
        if args.stop_at >= args.duration:
            # 这条报错必须直接给出改法：交付命令里漏掉 --duration 时（默认 5），
            # 用户只会看到「必须小于」，不知道要补 --duration（2026-09-21 实际踩到）。
            parser.error(
                f"--stop-at {args.stop_at:g} 必须小于 --duration {args.duration:g}"
                f"（滑行段 = duration − stop_at，现在算出来是 ≤ 0）。"
                f"要 {args.stop_at:g} s 拖曳 + 5 s 滑行就写 --duration {args.stop_at + 5:g} "
                f"--stop-at {args.stop_at:g}；不做滑行段的话去掉 --stop-at")
    if args.spawn_height is not None and (not math.isfinite(args.spawn_height) or args.spawn_height <= 0):
        parser.error("--spawn-height must be finite and positive")
    if not math.isfinite(args.cart_drop) or not 0.0 <= args.cart_drop <= 0.1:
        parser.error("--cart-drop must be in [0, 0.1] m")
    for mass in args.cart_mass:
        if mass is not None and (not math.isfinite(mass) or not 2.0 <= mass <= 50.0):
            parser.error("--cart-mass values must be in [2, 50] kg")
    if not math.isfinite(args.ground_friction) or not 0.0 < args.ground_friction <= 2.0:
        parser.error("--ground-friction must be in (0, 2]")
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


@dataclass(frozen=True)
class SweepCase:
    """一个 case 的标识：绳索模型 + 质量 + 轮阻。

    用**具名结构**而不是裸元组：上一版是 `(mass, damping)`，加绳索模型这一维时漏改了
    `case_config` 里的 `for i, (m, b) in enumerate(cases)`，实跑第一例就
    `ValueError: too many values to unpack (expected 2)`。字段有名字之后，
    加字段不会再悄悄弄坏按位置解包的消费者。
    （NamedTuple 也是 tuple，所以与裸元组的相等比较仍然成立。）
    """

    cart_mass: float | None
    wheel_damping: float
    rope_model: str = "compliant"


@dataclass(frozen=True)
class CoastPrediction:
    """启动预判：滑行距离与最小挂点间距（都由 P2 验收过的解析式给出）。"""

    coast_m: float
    min_gap_m: float


def sweep_cases(cart_masses, wheel_dampings, rope_models=("compliant",)):
    """把 `--cart-mass` × `--wheel-damping` × `--rope-model` 做笛卡尔积，得到顺序固定的 case 列表。

    一个进程内顺序跑完所有 case（省掉每个组合重启一次 Isaac Sim）；`None` 表示用 URDF 名义质量。
    绳索模型也是一维：`--rope-model compliant inextensible` 就能在**同一条件**下把两套模型
    各跑一遍（同样的初始条件、同样的随机性），便于逐项对比（延长量/冲量/稳态张力/扰动）。
    """
    if not cart_masses or not wheel_dampings or not rope_models:
        raise ValueError("--cart-mass、--wheel-damping、--rope-model 都不能为空")
    return [SweepCase(mass, damping, name) for name in rope_models for mass in cart_masses
            for damping in wheel_dampings]


def sweep_records(cases):
    """`config.json` 里 `sweep.cases` 的记录（纯函数，可离线测）。

    抽出来是因为上一版把它内联在 `main()` 里、按位置解包 `cases`，加了一维就崩——
    而 `main()` 只有跑仿真才会执行，离线测试抓不到（与 §5.8 的 `UnboundLocalError` 同一类）。
    """
    return [{"index": index, "cart_mass_kg": case.cart_mass,
             "wheel_damping": case.wheel_damping, "rope_model": case.rope_model}
            for index, case in enumerate(cases)]


def case_label(case_index: int, mass_target, damping: float, nominal_total_kg: float,
               rope_model: str = "compliant") -> str:
    """case 目录名：绳索模型 + 质量（kg，None 表示名义值）+ 阻尼。

    模型名放进目录名是必要的：两套模型可以跑同一个 case 编号序列，不写进名字会互相覆盖。
    """
    mass = nominal_total_kg if mass_target is None else mass_target
    return f"case_{case_index:02d}_{rope_model}_m{mass:g}_b{damping:g}"


def mass_scale_factor(target_total_kg: float | None, nominal_total_kg: float) -> float:
    """目标总质量相对名义值的缩放比例（质量与惯量按同一比例一起缩放）。

    只缩质量不缩惯量会让模型不自洽（AGENTS.md 的模型约定），所以这里只给出**一个**比例，
    由调用方同时乘到 masses 与 inertias 上；质心位置与轮半径都不变。
    """
    if not math.isfinite(nominal_total_kg) or nominal_total_kg <= 0:
        raise ValueError("名义总质量必须为有限正数")
    if target_total_kg is None:
        return 1.0
    if not math.isfinite(target_total_kg) or target_total_kg <= 0:
        raise ValueError("目标质量必须为有限正数")
    return target_total_kg / nominal_total_kg


def predicted_coast_distance(initial_speed: float, wheel_damping: float, *,
                             cart_mass_kg: float, wheel_inertia_kgm2: float,
                             wheel_radius_m: float, stop_speed: float = 0.02) -> float:
    """预测停车后小车会滑多远：`D = v0·τ·(1 − v_stop/v0)`，`τ = m_eff·r²/(4b)`。

    绳一旦松弛，小车就只受轮轴阻力，所以这条解析式直接适用。P2 验收的 8 个工况实测与它
    吻合 **0.27%–0.96%**，因此在跑之前就能判断「停车后小车会不会追到机器人」。

    `m_eff = m_cart + 4·I_wheel/r²`（四个轮的转动惯量折算成平动质量）。
    """
    for name, value in (("initial_speed", initial_speed), ("wheel_damping", wheel_damping),
                        ("cart_mass_kg", cart_mass_kg), ("wheel_inertia_kgm2", wheel_inertia_kgm2),
                        ("wheel_radius_m", wheel_radius_m), ("stop_speed", stop_speed)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} 必须为有限正数")
    m_eff = cart_mass_kg + 4.0 * wheel_inertia_kgm2 / (wheel_radius_m ** 2)
    tau = m_eff * wheel_radius_m ** 2 / (4.0 * wheel_damping)
    if initial_speed <= stop_speed:
        return 0.0
    return initial_speed * tau * (1.0 - stop_speed / initial_speed)


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
        from imgo2_rl.tasks.manager_based.towing.mdp.rope import point_velocity
        from imgo2_rl.tasks.manager_based.towing.mdp.rope_model import (
            BodyProperties, make_rope_model, world_inverse_inertia)
        from imgo2_rl.tasks.manager_based.towing.utils.low_level_policy import (
            FrozenLowLevelPolicy, parts_from_robot_state)
        from imgo2_rl.tasks.manager_based.towing.utils.policy_cfg import get_policy
        from imgo2_rl.tasks.manager_based.towing.utils.recording import (
                TowRecorder, joint_position_fields)

        policy_cfg = get_policy(args.policy)
        spawn_height = args.spawn_height if args.spawn_height is not None else ROBOT_SPAWN_HEIGHT_M

        cart_cfg, model = make_cart_cfg(output / "usd", drop_height=args.cart_drop)
        cart_attachment = tuple(model["attachment_position_m"])
        robot_attachment = tuple(ROBOT_ATTACHMENT_OFFSET_M)
        # 逐 case 的笛卡尔积：--cart-mass × --wheel-damping，在同一进程内顺序执行
        cases = sweep_cases(args.cart_mass, args.wheel_damping, args.rope_model)
        # 计划期（plan-time）的标量都在这里一次算好，避免后面「先用后赋值」：
        # 之前 stop_steps 与 scale 各踩过一次（main() 只有跑仿真才执行，离线测试抓不到）。
        scales = [mass_scale_factor(case.cart_mass, model["total_mass_kg"])
                  for case in cases]
        cart_cfg.init_state.pos = (
            initial_cart_x(args.rope_length, args.slack, spawn_height=spawn_height,
                           cart_height=model["resting_height_m"],
                           robot_offset=robot_attachment, cart_offset=cart_attachment),
            0.0, model["resting_height_m"] + args.cart_drop)

        # 跑之前的预判：停车后小车会滑多远、会不会追到机器人（用 P2 验收过的解析式）。
        # 参数组合（L0 / 速度 / 轮阻 / 质量）必须一起看：v=1.0、b=0.016 时滑行 1.06 m，
        # L0=1.0 就已经会撞；质量越大滑得越远（m_eff 同比例）。只警告不拦——P6 本就想观察追尾。
        predictions = []
        for case, case_scale in zip(cases, scales):
            coast = predicted_coast_distance(
                args.velocity, case.wheel_damping,
                cart_mass_kg=model["total_mass_kg"] * case_scale,
                wheel_inertia_kgm2=model["inertias_kgm2"]["wheel_fl"][1] * case_scale,
                wheel_radius_m=model["wheel_radius_m"])
            predictions.append(CoastPrediction(coast, args.rope_length - coast))
            print(f"[PLAN] {case.rope_model} "
                  f"m={case.cart_mass or model['total_mass_kg']:.0f} kg b={case.wheel_damping:g} "
                  f"L0={args.rope_length:g} v={args.velocity:g} ⇒ 预测滑行 {coast:.3f} m，"
                  f"最小间距 {args.rope_length - coast:+.3f} m"
                  + ("   ⚠ 会追到机器人" if args.rope_length - coast <= 0.05 else ""), flush=True)

        sim_cfg = sim_utils.SimulationCfg(
            dt=args.dt, device=args.device,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=args.ground_friction, dynamic_friction=args.ground_friction,
                restitution=0.0, friction_combine_mode="average", restitution_combine_mode="min"))
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view((2.5, 2.5, 1.8), (-0.7, 0.0, 0.2))
        scene_cfg = TowSceneCfg(num_envs=1, env_spacing=6.0, cart=cart_cfg)
        # 地面材质也要在构造场景之前改（与 init_state 同理：构造时才会读配置建 prim）
        ground = scene_cfg.ground.spawn.physics_material
        ground.static_friction = args.ground_friction
        ground.dynamic_friction = args.ground_friction
        # 机器人初始位姿必须在 **构造 InteractiveScene 之前** 写进配置：场景在构造时就把
        # 资产按 init_state 摆好，之后再改 cfg 不会生效（`--spawn-height` 会被静默忽略）。
        scene_cfg.robot.init_state.pos = (0.0, 0.0, spawn_height)
        scene_cfg.robot.init_state.joint_pos = dict(zip(policy_cfg.joint_names,
                                                        policy_cfg.default_dof_pos))
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        robot, cart, contacts = scene["robot"], scene["cart"], scene["wheel_contacts"]
        deck_contacts = scene["deck_contacts"]
        cart_env_idx = torch.arange(cart.num_instances, dtype=torch.int, device="cpu")
        # 名义质量/惯量作为缩放基准（每 case 都从它算，保证可重复；不能读回上一次的结果）
        nominal_masses = cart.root_physx_view.get_masses().clone()
        nominal_inertias = cart.root_physx_view.get_inertias().clone()
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
        cart_wheel_ids, cart_wheel_names = cart.find_bodies(list(model["wheel_names"]), preserve_order=True)
        contact_ids, contact_names = contacts.find_bodies(list(model["wheel_names"]), preserve_order=True)
        deck_ids, deck_names = deck_contacts.find_bodies(["base_link"], preserve_order=True)
        if len(cart_joint_ids) != 4 or cart.num_joints != 4 or len(contact_ids) != 4:
            raise RuntimeError("导入的小车必须有四个关节与四个同名轮接触")
        if len(deck_ids) != 1:
            raise RuntimeError(f"车斗接触传感器应当只覆盖车斗一个刚体，实际 {deck_names}")

        decimation = max(1, int(round(policy_cfg.control_dt / dt)))
        if not math.isclose(decimation * dt, policy_cfg.control_dt, rel_tol=1e-6):
            raise RuntimeError("policy_cfg.control_dt 必须是物理 dt 的整数倍")
        policy = FrozenLowLevelPolicy(policy_cfg, device=args.device)

        def build_rope_model(name):
            """按名字建模型；`--rope-model` 可以给多个，于是同一进程里逐 case 换模型。"""
            return make_rope_model(name, rest_length=args.rope_length,
                                   stiffness=args.stiffness, damping=args.damping,
                                   position_gain=args.position_gain,
                                   max_correction_rate=args.max_correction_rate)
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
        robot_zero_torque = torch.zeros(robot.num_instances, robot.num_bodies, 3,
                                        dtype=torch.float32, device=args.device)
        cart_zero_torque = torch.zeros(cart.num_instances, cart.num_bodies, 3,
                                       dtype=torch.float32, device=args.device)

        def link_frame_force(asset, body_id, force_world):
            local = math_utils.quat_apply_inverse(asset.data.body_quat_w[:, body_id], force_world)
            return local.unsqueeze(1)

        def _quat_to_matrix(quat):                 # Isaac Lab 四元数是 (w, x, y, z)
            w, x, y, z = quat
            return ((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
                    (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
                    (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))

        def inverse_inertia_world(asset, body_id):
            """某刚体在**世界系**的逆惯量（3×3，元素为标量）。

            PhysX 给的是机体主轴系下的对角惯量（展平 9 个数，对角在 0/4/8），
            按刚体姿态旋转：`I_w⁻¹ = R diag(1/I) Rᵀ`（`world_inverse_inertia` 做这件事）。
            inextensible 模型只用它算挂点的转动项 `(r×e)ᵀ I⁻¹ (r×e)`；名义几何下力臂与
            绳方向平行、该项为零。逐 env 分配时同一套 API 直接吃 `(N,)` 张量即可。
            """
            flat = [float(v) for v in asset.root_physx_view.get_inertias()[0, body_id]]
            rotation = _quat_to_matrix([float(v) for v in asset.data.body_quat_w[0, body_id]])
            return world_inverse_inertia((flat[0], flat[4], flat[8]), rotation)

        # 机器人受绳冲量时四足撑地、整车一起抵抗 ⇒ 等效质量取**全部 link 之和**
        # （实测 12.6996 kg）。这是近似：严格的整机等效质量还取决于腿的接触状态，
        # 与真实值的偏差由「约束执行误差」在现场实测（见文档）。
        robot_mass_kg = float(robot.root_physx_view.get_masses().sum())

        def cart_mass_effective_kg():
            """小车纵向等效质量 = 整车质量 + 四轮滚动惯量折算 `4I/r²`。

            冲量要先把轮子带转，所以等效质量比总质量大（10 kg 车 → 10.8 kg），
            与滑行段解析式用的 `m_eff` 是同一口径。轮子绕 y 轴（index 4）转动。
            """
            masses = cart.root_physx_view.get_masses()
            inertias = cart.root_physx_view.get_inertias()
            spun = sum(float(inertias[0, body_id][4]) for body_id in cart_wheel_ids)
            return float(masses.sum()) + 4.0 * spun / model["wheel_radius_m"] ** 2

        def apply_rope_and_resistance(command, wheel_damping):
            """按当前状态算绳力与轮阻并写入缓冲（显式欧拉，同 P1/P2）。

            **挂点世界坐标 = 刚体原点 + 旋转后的挂点偏移**。第一版只加了偏移去算
            挂点速度、位置却直接用了刚体原点，于是绳长里混进了机器人与小车的高度差
            （0.35 vs 0.15 m），初始张力被抬到 ~1955 N 直接把机器人拽倒。

            绳模型可切换（`--rope-model`）：两套模型都是「读状态 → 给出挂点力」，
            所以这里只负责把状态凑齐、把力写进缓冲，物理差异全在模型内部。
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
            # `inextensible` 需要两侧刚体属性来算等效逆质量（转动项 = 挂点离质心的力臂效应）。
            # 质量取**整机**：机器人受冲量时四足撑地、整车一起抵抗，所以是全部 link 之和
            # （12.6996 kg），不是 base 单链的 5.5339 kg；小车再加上四轮滚动惯量折算项
            # `4I/r²`（冲量要先把轮子带转），与滑行段用的 m_eff 同一口径。
            robot_props = BodyProperties(
                mass=robot_mass_kg, inverse_inertia_world=inverse_inertia_world(robot, base_id),
                offset=tuple(float(v) for v in robot_offset_w[0]))
            cart_props = BodyProperties(
                mass=cart_mass_effective_kg(),
                inverse_inertia_world=inverse_inertia_world(cart, cart_base_id),
                offset=tuple(float(v) for v in cart_offset_w[0]))
            state = rope_model.update(robot_point=robot_p, cart_point=cart_p,
                                      robot_velocity=robot_v, cart_velocity=cart_v, dt=dt,
                                      robot=robot_props, cart=cart_props)
            force_robot = torch.tensor(state.force_on_robot, dtype=torch.float32, device=args.device).view(1, 3)
            force_cart = torch.tensor(state.force_on_cart, dtype=torch.float32, device=args.device).view(1, 3)
            robot.set_external_force_and_torque(link_frame_force(robot, base_id, force_robot),
                                                robot_zero_torque[:, :1], positions=robot_attach,
                                                body_ids=base_ids)
            cart.set_external_force_and_torque(link_frame_force(cart, cart_base_id, force_cart),
                                               cart_zero_torque[:, :1], positions=cart_attach,
                                               body_ids=cart_base_ids)
            effort = torch.zeros_like(cart.data.joint_pos)
            effort[:, cart_joint_ids] = viscous_resistance(
                cart.data.joint_vel[:, cart_joint_ids], wheel_damping)
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

        def case_config(case, mass_scale, coast, gap):
            return {
                "policy": args.policy,
                "model_sha256": hashlib.sha256(policy_cfg.model_path.read_bytes()).hexdigest(),
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
                "rope": {"model": case.rope_model,
                         "rest_length_m": args.rope_length, "stiffness_n_per_m": args.stiffness,
                         "damping_ns_per_m": args.damping, "initial_slack_m": args.slack,
                         # 仅 inextensible 用；写进产物便于复算与对比
                         "position_gain": args.position_gain,
                         "max_correction_rate_mps": args.max_correction_rate,
                         "model_note": "compliant = unilateral spring-damper (stores elastic energy); "
                                       "inextensible = unilateral distance constraint d<=L0, T>=0, "
                                       "T(L0-d)=0, engagement handled by constraint impulse "
                                       "(NOT a large k)"},
                "wheel_damping_nms_per_rad": case.wheel_damping, "user_command_mps": args.velocity,
                "cart_mass_target_kg": case.cart_mass, "cart_mass_scale": mass_scale,
                "cart_mass_actual_kg": model["total_mass_kg"] * mass_scale,
                "ground_friction": args.ground_friction,
                "predicted_coast_distance_m": coast, "predicted_min_gap_m": gap,
                "prediction_note": "由 P2 验收过的黏性衰减解析式 D = v0*tau*(1-v_stop/v0)，"
                                   "tau = m_eff*r^2/(4b)；P2 实测吻合 0.27%-0.96%",
                "settle_time_s": settle_steps * dt, "duration_s": command_steps * dt,
                "stop_at_s": args.stop_at, "tow_phase_s": schedule.tow_phase_s,
                "coast_phase_s": schedule.coast_phase_s,
                "phase_steps": {"station": schedule.station_steps, "tow": schedule.tow_steps,
                                "coast": schedule.coast_steps},
                "dt_s": dt, "decimation": decimation, "device": args.device,
                # 机器人总质量：弹性诊断要算绳的折合质量 μ = ((1/m_eff)+(1/m_robot))⁻¹。
                # 从仿真读（全部 link 之和，实测 **12.6996 kg**），**不是** base 单链的 5.5339 kg
                # ——用错会让 ω、ζ、步长上限全偏（我手算时踩过，见 docs §5.18）。
                "robot_mass_kg": float(robot.root_physx_view.get_masses().sum()),
                # 记录里 `robot_jp_00..11` 的关节顺序（策略/契约顺序，逐腿 FL/FR/RL/RR）：
                # 离线算间隙的 FK 需要它，写进产物免得靠"约定"记忆。
                "policy_joint_names": list(policy_cfg.joint_names),
                "contact_witness": "cart_deck_fx_n 为车斗 base_link 接触合力 x 分量（车斗不着地，"
                                   "非零即机器人压上来）；cart_wheel_fx_n 为四轮同类分量之和",
                "torque_convention": "rope force and wheel torque are computed from the state at the start of each step",
                "sweep": {"cases": sweep_records(cases)},
                "git": manifest["git"],
            }

        def reset_case(mass_target, case_scale):
            """把本 case 的初始状态写死，然后清缓冲。

            必需：`Articulation.reset()` **只清执行器与外力缓冲，不写位姿**（见 IsaacLab
            articulation.py:172），所以必须在每个 case 开始前显式重写两个刚体的位姿/速度/
            关节状态。与 P1/P2「每个 case 前重写小车状态」是同一约定。
            """
            cart.root_physx_view.set_masses(nominal_masses * case_scale, cart_env_idx)
            cart.root_physx_view.set_inertias(nominal_inertias * case_scale, cart_env_idx)
            for asset in (robot, cart):
                root = asset.data.default_root_state.clone()
                root[:, :3] += scene.env_origins        # 初始位姿来自配置（robot 的 z、cart 的 x/z）
                root[:, 7:] = 0.0                       # 线速度/角速度清零
                asset.write_root_pose_to_sim(root[:, :7])
                asset.write_root_velocity_to_sim(root[:, 7:])
                asset.write_joint_state_to_sim(asset.data.default_joint_pos.clone(),
                                               torch.zeros_like(asset.data.default_joint_vel))
            cart.set_joint_effort_target(torch.zeros_like(cart.data.joint_pos))
            robot.set_external_force_and_torque(robot_zero_torque, robot_zero_torque)
            cart.set_external_force_and_torque(cart_zero_torque, cart_zero_torque)
            policy.reset()                              # reset 契约：last_action 归零
            scene.reset()                               # 清执行器/外力缓冲
            scene.update(dt)

        results = []
        case_dirs = []
        for case_index, (case, case_scale, prediction) in enumerate(
                zip(cases, scales, predictions)):
            case_dir = output / case_label(case_index, case.cart_mass, case.wheel_damping,
                                           model["total_mass_kg"], case.rope_model)
            label = case_dir.name.split("_", 2)[2]
            case_dir.mkdir(parents=True, exist_ok=False)
            case_dirs.append(case_dir.name)
            rope_model = build_rope_model(case.rope_model)
            print(f"[CASE {case_index}] 绳索模型 = {case.rope_model}"
                  + ("" if case.rope_model == "compliant"
                     else f"（position_gain={args.position_gain:g}、"
                          f"max_correction_rate={args.max_correction_rate:g} m/s）"),
                  flush=True)
            reset_case(case.cart_mass, case_scale)
            actual_mass = float(cart.root_physx_view.get_masses().sum())
            print(f"[CASE {case_index}] {label}: 小车 {actual_mass:.3f} kg "
                  f"b={case.wheel_damping:g} ⇒ {case_dir.name}",
                  flush=True)
            recorder = TowRecorder(case_dir,
                                   case_config(case, case_scale,
                                               prediction.coast_m, prediction.min_gap_m))
            for step in range(schedule.total_steps):
                phase = schedule.phase_of(step)
                command = args.velocity if phase == "tow" else 0.0
                if step % decimation == 0:
                    policy_step(command)
                state = apply_rope_and_resistance(command, case.wheel_damping)
                scene.write_data_to_sim()
                sim.step()
                scene.update(dt)
                quat = robot.data.root_quat_w[0]
                qw, qx, qy, qz = (float(v) for v in quat)     # 与 P2 相同的 roll/pitch 公式
                pitch = math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx))))
                wheel_omega = {
                    f"wheel_{leg}_omega_radps": float(cart.data.joint_vel[0, jid])
                    for leg, jid in zip(("fl", "fr", "rl", "rr"), cart_joint_ids)
                }
                # 关节角按策略顺序存（policy_to_asset 是「策略下标 → 资产下标」）：
                # 离线工具用同一份 policy_joint_names 做 FK，才能算出机器人后腿伸到哪里，
                # 进而得到车头与机器人之间的真实间隙（挂点间距不是间隙）。
                joint_pos_policy = robot.data.joint_pos[0, policy_to_asset]
                quat_robot = robot.data.root_quat_w[0]
                quat_load = cart.data.root_quat_w[0]
                recorder.append({
                    **wheel_omega,
                    # Isaac Lab 四元数是 (w, x, y, z)，落盘按 x/y/z/w 命名，避免歧义
                    **joint_position_fields(joint_pos_policy),
                    "robot_quat_x": float(quat_robot[1]),
                    "robot_quat_y": float(quat_robot[2]),
                    "robot_quat_z": float(quat_robot[3]),
                    "robot_quat_w": float(quat_robot[0]),
                    "load_quat_x": float(quat_load[1]),
                    "load_quat_y": float(quat_load[2]),
                    "load_quat_z": float(quat_load[3]),
                    "load_quat_w": float(quat_load[0]),
                    # 接触合力的 x 分量（世界系）：车斗永不着地 ⇒ 非零即机器人压上来；
                    # 轮子那一列正常滚动时只有克服轮阻所需的 ~1.7 N，撞击时跳到几十 N。
                    "cart_deck_fx_n": float(deck_contacts.data.net_forces_w[0, 0, 0]),
                    "cart_wheel_fx_n": float(contacts.data.net_forces_w[0, :, 0].sum()),
                    "phase": phase,
                    "time_s": (step + 1) * dt,
                    "user_cmd_mps": command,
                    "ref_cmd_mps": command,          # P4 还没有 command shaping
                    "robot_vx_mps": float(robot.data.root_lin_vel_w[0, 0]),
                    "load_vx_mps": float(cart.data.root_lin_vel_w[0, 0]),
                    "rope_tension_n": float(state.rope_tension),
                    # 统一日志接口：两套模型字段一致（见 mdp/rope_model.py 的 RopeSample）
                    "rope_extension_m": float(state.rope_extension),
                    "rope_length_rate_mps": float(state.rope_length_rate),
                    "rope_taut": float(state.is_taut),
                    "rope_impulse_ns": float(state.rope_impulse),
                    "rope_distance_m": float(state.distance),
                    "robot_x_m": float(robot.data.root_pos_w[0, 0]),
                    "load_x_m": float(cart.data.root_pos_w[0, 0]),
                    "robot_z_m": float(robot.data.root_pos_w[0, 2]),
                    "load_z_m": float(cart.data.root_pos_w[0, 2]),
                    "body_pitch_rad": pitch,
                    "body_pitch_rate_radps": float(robot.data.root_ang_vel_b[0, 1]),
                })
            recorder.close()
            recorder = None

            # 判读逻辑在标准库工具里（与 P1/P2 的 summarize_cart_coast.py 同一模式），
            # 于是判据可以用真实轨迹离线复算：scripts/tools/summarize_tow.py <case 目录>
            sys.path.insert(0, str(RL_ROOT / "scripts/tools"))
            from summarize_tow import summarize_tow
            with (case_dir / "tow.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            summary = summarize_tow(rows, user_command=args.velocity,
                                    joint_names=policy_cfg.joint_names)
            summary["case"] = case_dir.name
            summary["cart_mass_kg"] = model["total_mass_kg"] * case_scale
            summary["wheel_damping"] = damping
            json_file(case_dir / "summary.json", summary)
            clearance = ("n/a" if summary["final_clearance_m"] is None
                         else f"{summary['final_clearance_m']:.3f} m")
            print(f"[SUMMARY {case_index}] {label} 跟速={summary['steady_tracking_ratio']:.3f} "
                  f"T={summary['steady_tension_n']:.3f} N "
                  f"最小间距={summary['min_gap_after_tow_start_m']:.3f} m "
                  f"最终间距={summary['final_gap_m']:.3f} m "
                  f"最终车头间隙={clearance} "
                  f"追到机器人={'是' if summary['reached_robot'] else '否'}"
                  f"({summary['reached_robot_source']}) "
                  f"valid={summary['valid']} failures={summary['failures']}", flush=True)
            results.append(summary)

        # 扫描汇总
        all_valid = all(r["valid"] for r in results)
        sweep = {"cases": results, "case_dirs": case_dirs, "all_valid": all_valid,
                 "arguments": {"velocity": args.velocity, "rope_length": args.rope_length,
                               "slack": args.slack, "ground_friction": args.ground_friction,
                               "cart_masses": args.cart_mass, "wheel_dampings": args.wheel_damping,
                               "rope_models": args.rope_model,
                               "stop_at": args.stop_at, "duration": args.duration}}
        json_file(output / "sweep.json", sweep)
        manifest.update(state="completed", valid=all_valid, cases=case_dirs)
        json_file(output / "experiment.json", manifest)
        print(f"[SWEEP] {len(results)} 个 case，all_valid={all_valid} ⇒ {output}", flush=True)
        # 结束方式与失败路径一致：**不依赖 `application.close()` 让进程退出**。
        # 实测（2026-09-21，friction 扫描）跑完后 summary.json 与 sweep.json 都已写好、
        # state=completed，但进程不返回 —— shell 的 for 循环因此进不到下一个值。
        # 产物都已落盘（CSV 逐行 flush、JSON 写完即关），所以直接显式带码退出；
        # 与 CART-02 同一类问题：Kit 的关停路径不可靠，退出码必须自己给。
        print(f"[DONE] all_valid={all_valid} ⇒ 退出码 {0 if all_valid else 2}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if all_valid else 2)
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
