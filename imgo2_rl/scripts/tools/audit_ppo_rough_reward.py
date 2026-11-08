#!/usr/bin/env python3
"""在已有 MuJoCo 逐帧 CSV 上重算 PPO rough 的奖励分项（离线核对工具）。

用途
----
`docs/xiaoji_cmd/PPO奖励与TensorBoard完整说明_2026-10-27.md` 解释的是训练时的
`Episode_Reward/*`。训练日志只有回合平均，缺的是「同一套权重下，旧步态和新步态
各自能拿多少分、差在哪些项」。本工具读取 `mujoco_policy_benchmark` 产出的 CSV，
按 `velocity_env_cfg.py` + `base_move/rough_env_cfg.py` 的公式逐项重算，
输出「每秒贡献」，因此可以：

1. 在相同实际速度下比较两代策略的奖励结构（谁被奖励、谁被惩罚）；
2. 只改一两个 weight 做离线敏感性检查（例如 joint_pos_penalty=0、
   feet_air_time=1.8、feet_height 打开），先判断权重是否值得上机短训。

限制（必须和结论一起读）
------------------------
- CSV 是 MuJoCo 平地 + 生产 PD，不是 Isaac 的 rough 地形；`base_height_l2` 用
  平地目标 0.30m，真实训练的射线读取未在 CSV 中。
- `contact_n` 是法向力，Isaac 用三维净力；`undesired_contacts`（非足端碰撞）
  在 CSV 里没有，按 0 计。`feet_slide` 的接触判定用 >1N 近似。
- 足端相对机身速度由 200Hz 位置中心差分 + omega×r 还原；动作由
  `qdes` 反推（q0/scale 来自配置），与网络原始动作在裁剪时不同。
- 结论是「这套奖励在这个轨迹上的分数」，不是「策略一定按奖励最优化」；
  优化器/探索/地形分布仍可能让它收敛到别的解。

用法
----
    /usr/bin/python3 imgo2_rl/scripts/tools/audit_ppo_rough_reward.py \
      --run imgo2_rl/logs/mujoco_policy_1_low_speed_2026-10-27_45a8 \
      --cases policy_1_forward03,policy_forward02

    # 改权重做离线消融（未改任何训练配置）
    ... --set feet_air_time=1.8,joint_pos_penalty=0 --json out.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import OrderedDict
from pathlib import Path

DT = 0.02
PHYS_DT = 0.005
LEGS = ("FL", "FR", "RL", "RR")
JOINTS = tuple(f"{leg}_{j}_joint" for leg in LEGS for j in ("hip", "thigh", "shank"))
Q0 = [0.0, 0.87, -1.82] * 4
SCALE = [0.125, 0.25, 0.25] * 4
Q0_BY_NAME = dict(zip(JOINTS, Q0))
SCALE_BY_NAME = dict(zip(JOINTS, SCALE))

# 当前配置的权重（base_move/rough_env_cfg.py）；-0.0 的 feet_height 按关闭处理。
DEFAULT_WEIGHTS = OrderedDict(
    [
        ("track_lin_vel_xy_exp", 1.5),
        ("track_ang_vel_z_exp", 0.6),
        ("base_height_l2", -10.0),
        ("flat_orientation_l2", -5.0),
        ("lin_vel_z_l2", -2.0),
        ("ang_vel_xy_l2", -0.05),
        ("joint_torques_l2", -2.5e-6),
        ("joint_acc_l2", -5e-9),
        ("joint_power", -2e-5),
        ("action_rate_l2", -0.01),
        ("stand_still", -0.1),
        ("joint_pos_penalty", -0.5),
        ("joint_mirror", -1.0),
        ("contact_forces", -0.02),
        ("feet_air_time", 1.2),
        ("feet_air_time_deficit", 0.0),
        ("feet_air_time_variance", -8.0),
        ("feet_slide", -0.05),
        ("feet_height_body", -6.0),
        ("feet_height", 0.0),
    ]
)

MIRROR_PAIRS = (
    ("FR_hip_joint", "RL_hip_joint"),
    ("FR_thigh_joint", "RL_thigh_joint"),
    ("FR_shank_joint", "RL_shank_joint"),
    ("FL_hip_joint", "RR_hip_joint"),
    ("FL_thigh_joint", "RR_thigh_joint"),
    ("FL_shank_joint", "RR_shank_joint"),
)

REQUIRED_COLUMNS = (
    ["t_s", "command_vx", "command_vy", "command_wz", "z_m",
     "vx_body", "vy_body", "vz_body", "roll_rad", "pitch_rad", "yaw_rad",
     "gyro_x_rads", "gyro_y_rads", "gyro_z_rads"]
    + [f"{leg}_contact_n" for leg in LEGS]
    + [f"{leg}_{axis}_body" for leg in LEGS for axis in ("x", "y", "z")]
    + [f"dq_{j}" for j in JOINTS]
    + [f"tau_{j}" for j in JOINTS]
    + [f"qdes_{j}" for j in JOINTS]
    + list(JOINTS)
)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def rotation(roll: float, pitch: float, yaw: float) -> list[list[float]]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def sample_variance(values: list[float]) -> float:
    """torch.var 默认 unbiased：除以 n-1。"""
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / (len(values) - 1)


def load_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"{path}: 缺少列 {missing}")
        return [{k: float(v) for k, v in row.items() if v not in (None, "")} for row in reader]


def body_xy_velocity(rows: list[dict[str, float]], i: int, leg: str) -> tuple[float, float, float]:
    """机身系足端相对速度 v_rel = dp_b/dt + omega_b x p_b（文档 3.19/3.20）。"""
    lo, hi = max(0, i - 1), min(len(rows) - 1, i + 1)
    span = (hi - lo) * PHYS_DT
    p = [rows[i][f"{leg}_{a}_body"] for a in ("x", "y", "z")]
    dp = [(rows[hi][f"{leg}_{a}_body"] - rows[lo][f"{leg}_{a}_body"]) / span for a in ("x", "y", "z")]
    w = (rows[i]["gyro_x_rads"], rows[i]["gyro_y_rads"], rows[i]["gyro_z_rads"])
    vx = dp[0] + (w[1] * p[2] - w[2] * p[1])
    vy = dp[1] + (w[2] * p[0] - w[0] * p[2])
    vz = dp[2] + (w[0] * p[1] - w[1] * p[0])
    return vx, vy, vz


def evaluate(
    path: Path,
    weights: dict[str, float],
    command=None,
    feet_height_target: float = 0.10,
    air_deficit_threshold: float = 0.20,
    fhb_swing_only: bool = False,
    slide_world: bool = False,
) -> dict:
    rows = load_rows(path)
    n = len(rows)
    # 50Hz 策略步：每 4 条 200Hz 记录取一条；统计从 4s 开始（与既有报告一致）。
    indices = [i for i in range(0, n, 4) if rows[i]["t_s"] >= 4.0]
    if not indices:
        raise SystemExit(f"{path}: 没有 t>=4s 的样本")

    # 200Hz 关节加速度
    ddq = {j: [0.0] * n for j in JOINTS}
    for j in JOINTS:
        col = f"dq_{j}"
        for i in range(1, n):
            ddq[j][i] = (rows[i][col] - rows[i - 1][col]) / PHYS_DT

    sums = {k: 0.0 for k in weights}
    last_air = [0.0] * 4
    last_contact = [0.0] * 4
    cur_air = [0.0] * 4
    cur_contact = [0.0] * 4
    in_contact = [False] * 4
    prev_action = None
    heights = []
    speeds = []

    for i in indices:
        r = rows[i]
        cmd = (r["command_vx"], r["command_vy"], r["command_wz"]) if command is None else command
        cnorm = math.sqrt(sum(x * x for x in cmd))
        R = rotation(r["roll_rad"], r["pitch_rad"], r["yaw_rad"])
        gx, gy, gz = -R[2][0], -R[2][1], -R[2][2]
        U = clamp(-gz, 0.0, 0.7) / 0.7

        vx, vy, vz = r["vx_body"], r["vy_body"], r["vz_body"]
        wx, wy, wz = r["gyro_x_rads"], r["gyro_y_rads"], r["gyro_z_rads"]
        q = {j: r[j] for j in JOINTS}
        dq = {j: r[f"dq_{j}"] for j in JOINTS}
        tau = {j: r[f"tau_{j}"] for j in JOINTS}
        qdes = {j: r[f"qdes_{j}"] for j in JOINTS}
        contact_force = {leg: r[f"{leg}_contact_n"] for leg in LEGS}
        pb = {leg: [r[f"{leg}_{a}_body"] for a in ("x", "y", "z")] for leg in LEGS}

        if "track_lin_vel_xy_exp" in weights:
            sums["track_lin_vel_xy_exp"] += (
                weights["track_lin_vel_xy_exp"]
                * math.exp(-((cmd[0] - vx) ** 2 + (cmd[1] - vy) ** 2) / 0.25)
                * U
            )
        if "track_ang_vel_z_exp" in weights:
            sums["track_ang_vel_z_exp"] += (
                weights["track_ang_vel_z_exp"] * math.exp(-((cmd[2] - wz) ** 2) / 0.25) * U
            )
        if "base_height_l2" in weights:
            sums["base_height_l2"] += weights["base_height_l2"] * (r["z_m"] - 0.30) ** 2 * U
        if "flat_orientation_l2" in weights:
            sums["flat_orientation_l2"] += weights["flat_orientation_l2"] * (gx * gx + gy * gy) * U
        if "lin_vel_z_l2" in weights:
            sums["lin_vel_z_l2"] += weights["lin_vel_z_l2"] * vz * vz * U
        if "ang_vel_xy_l2" in weights:
            sums["ang_vel_xy_l2"] += weights["ang_vel_xy_l2"] * (wx * wx + wy * wy) * U
        if "joint_torques_l2" in weights:
            sums["joint_torques_l2"] += weights["joint_torques_l2"] * sum(t * t for t in tau.values())
        if "joint_acc_l2" in weights:
            sums["joint_acc_l2"] += weights["joint_acc_l2"] * sum(ddq[j][i] ** 2 for j in JOINTS)
        if "joint_power" in weights:
            sums["joint_power"] += weights["joint_power"] * sum(abs(dq[j] * tau[j]) for j in JOINTS)

        action = [(qdes[j] - Q0_BY_NAME[j]) / SCALE_BY_NAME[j] for j in JOINTS]
        if "action_rate_l2" in weights and prev_action is not None:
            sums["action_rate_l2"] += weights["action_rate_l2"] * sum(
                (a - b) ** 2 for a, b in zip(action, prev_action)
            )
        prev_action = action

        dev = [q[j] - Q0_BY_NAME[j] for j in JOINTS]
        if "stand_still" in weights:
            sums["stand_still"] += (
                weights["stand_still"] * sum(abs(d) for d in dev) * (1.0 if cnorm < 0.05 else 0.0) * U
            )
        if "joint_pos_penalty" in weights:
            scale = 1.0 if (cnorm > 0.1 or math.hypot(vx, vy) > 0.5) else 5.0
            sums["joint_pos_penalty"] += (
                weights["joint_pos_penalty"] * scale * math.sqrt(sum(d * d for d in dev)) * U
            )
        if "joint_mirror" in weights:
            diff = sum((q[a] - q[b]) ** 2 for a, b in MIRROR_PAIRS) / len(MIRROR_PAIRS)
            sums["joint_mirror"] += weights["joint_mirror"] * diff * U

        if "contact_forces" in weights:
            cf = 0.0
            for leg in LEGS:
                window = max(rows[k][f"{leg}_contact_n"] for k in range(max(0, i - 2), i + 1))
                cf += max(window - 100.0, 0.0)
            sums["contact_forces"] += weights["contact_forces"] * cf

        first_contact = [False] * 4
        for s, leg in enumerate(LEGS):
            is_contact = contact_force[leg] > 1.0
            if is_contact and not in_contact[s]:
                first_contact[s] = True
                last_air[s] = cur_air[s] + DT
                cur_air[s] = 0.0
                cur_contact[s] = 0.0
            elif is_contact:
                cur_contact[s] += DT
            else:
                if in_contact[s]:
                    last_contact[s] = cur_contact[s] + DT
                    cur_contact[s] = 0.0
                cur_air[s] += DT
            in_contact[s] = is_contact

        if "feet_air_time" in weights:
            air_score = sum(
                (last_air[s] - 0.5) * (1.0 if first_contact[s] else 0.0) for s in range(4)
            )
            sums["feet_air_time"] += weights["feet_air_time"] * air_score * (1.0 if cnorm > 0.1 else 0.0) * U
        if "feet_air_time_deficit" in weights:
            deficit = sum(
                max(air_deficit_threshold - last_air[s], 0.0) * (1.0 if first_contact[s] else 0.0)
                for s in range(4)
            )
            sums["feet_air_time_deficit"] += (
                weights["feet_air_time_deficit"] * deficit * (1.0 if cnorm > 0.1 else 0.0) * U
            )
        if "feet_air_time_variance" in weights:
            variance = sample_variance([min(x, 0.5) for x in last_air]) + sample_variance(
                [min(x, 0.5) for x in last_contact]
            )
            sums["feet_air_time_variance"] += weights["feet_air_time_variance"] * variance * U

        vrel = {leg: body_xy_velocity(rows, i, leg) for leg in LEGS}
        if "feet_slide" in weights:
            slide = 0.0
            for leg in LEGS:
                if contact_force[leg] <= 1.0:
                    continue
                if slide_world:
                    v_body = [vrel[leg][0] + vx, vrel[leg][1] + vy, vrel[leg][2] + vz]
                    v_w = [sum(R[k][m] * v_body[m] for m in range(3)) for k in range(3)]
                    slide += math.hypot(v_w[0], v_w[1])
                else:
                    slide += math.hypot(vrel[leg][0], vrel[leg][1])
            sums["feet_slide"] += weights["feet_slide"] * slide * U
        if "feet_height_body" in weights:
            fhb = 0.0
            for leg in LEGS:
                if fhb_swing_only and contact_force[leg] > 1.0:
                    continue
                fhb += (pb[leg][2] + 0.20) ** 2 * math.tanh(
                    2.0 * math.hypot(vrel[leg][0], vrel[leg][1])
                )
            sums["feet_height_body"] += (
                weights["feet_height_body"] * fhb * (1.0 if cnorm > 0.1 else 0.0) * U
            )
        if "feet_height" in weights and weights["feet_height"] != 0.0:
            fh = 0.0
            for leg in LEGS:
                p = pb[leg]
                p_w = [sum(R[k][m] * p[m] for m in range(3)) for k in range(3)]
                v_body = [vrel[leg][0] + vx, vrel[leg][1] + vy, vrel[leg][2] + vz]
                v_w = [sum(R[k][m] * v_body[m] for m in range(3)) for k in range(3)]
                fh += (r["z_m"] + p_w[2] - feet_height_target) ** 2 * math.tanh(
                    2.0 * math.hypot(v_w[0], v_w[1])
                )
            sums["feet_height"] += weights["feet_height"] * fh * (1.0 if cnorm > 0.1 else 0.0) * U

        heights.append(r["z_m"])
        speeds.append(math.hypot(vx, vy))

    n_samples = len(indices)
    per_second = {k: v / n_samples for k, v in sums.items()}
    per_second["_total"] = sum(per_second.values())
    per_second["_samples"] = n_samples
    per_second["_base_height_mean_m"] = sum(heights) / len(heights)
    per_second["_speed_mean_mps"] = sum(speeds) / len(speeds)
    return per_second


def parse_weights(spec: str | None) -> dict[str, float]:
    weights = OrderedDict(DEFAULT_WEIGHTS)
    if not spec:
        return weights
    for item in spec.split(","):
        if not item.strip():
            continue
        name, value = item.split("=", 1)
        name = name.strip()
        if name not in weights:
            raise SystemExit(f"未知奖励项 {name}；可选：{', '.join(weights)}")
        weights[name] = float(value)
    # 0 权重项在训练里会被 disable_zero_weight_rewards 关掉，这里同样跳过。
    return OrderedDict((k, v) for k, v in weights.items() if v != 0.0)


def fmt(value: float) -> str:
    return f"{value:+.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, type=Path, help="含 CSV 的目录")
    parser.add_argument("--cases", required=True, help="逗号分隔的 CSV 名（不含 .csv）")
    parser.add_argument("--set", dest="weights", default=None, help="覆盖权重，如 feet_air_time=1.8,joint_pos_penalty=0")
    parser.add_argument("--command", default=None, help="用同一指令覆盖 CSV 中的 command_vx,vy,wz（逗号分隔）")
    parser.add_argument("--feet-height-target", type=float, default=0.10, help="feet_height 的世界目标高度，默认 0.10m")
    parser.add_argument("--air-deficit-threshold", type=float, default=0.20, help="feet_air_time_deficit 的滞空阈值，默认 0.20s")
    parser.add_argument("--feet-height-body-swing-only", action="store_true", help="feet_height_body 只统计未触地脚（语义修复候选）")
    parser.add_argument("--feet-slide-world", action="store_true", help="feet_slide 用世界系足速（语义修复候选）")
    parser.add_argument("--json", type=Path, default=None, help="把逐项结果写成 JSON")
    args = parser.parse_args()

    weights = parse_weights(args.weights)
    command = tuple(float(x) for x in args.command.split(",")) if args.command else None
    results = OrderedDict()
    for case in args.cases.split(","):
        case = case.strip()
        if not case:
            continue
        path = args.run / f"{case}.csv"
        if not path.exists():
            raise SystemExit(f"找不到 {path}")
        results[case] = evaluate(
            path,
            weights,
            command=command,
            feet_height_target=args.feet_height_target,
            air_deficit_threshold=args.air_deficit_threshold,
            fhb_swing_only=args.feet_height_body_swing_only,
            slide_world=args.feet_slide_world,
        )

    names = list(weights) + ["_total", "_base_height_mean_m", "_speed_mean_mps"]
    width = max(len(n) for n in names) + 2
    print("权重：" + ", ".join(f"{k}={v:g}" for k, v in weights.items()))
    print("单位：每秒（= Episode_Reward 的口径）；正=奖励，负=惩罚。")
    header = "term".ljust(width) + "".join(c.rjust(14) for c in results)
    print(header)
    print("-" * len(header))
    for name in names:
        label = name if not name.startswith("_") else name
        row = label.ljust(width)
        for case in results:
            value = results[case][name]
            row += (fmt(value) if name.startswith("_") and name not in ("_total",) else fmt(value)).rjust(14)
        print(row)
    print("\n注：_base_height_mean_m/_speed_mean_mps 分别是机身高度与平面速度均值，不是奖励。")
    print("未计入：undesired_contacts（CSV 无躯干/腿部接触力）；base_height_l2 用平地目标 0.30m。")

    if args.json:
        payload = {
            "run": str(args.run),
            "weights": weights,
            "command_override": command,
            "results": results,
            "limits": [
                "MuJoCo 平地轨迹，不是 Isaac rough；base_height_l2 用 0.30m 目标。",
                "contact_n 为法向力，contact_forces 以它近似 3D 净力；feet_slide 接触判定 >1N。",
                "undesired_contacts 未计入。",
                "qdes 反推动作；未考虑网络原始动作被裁剪的情形。",
            ],
        }
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        print(f"JSON 写入 {args.json}")


if __name__ == "__main__":
    main()
