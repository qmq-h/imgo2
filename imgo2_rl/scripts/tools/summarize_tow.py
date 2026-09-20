"""Recompute a tow run's summary and verdict using only the Python standard library.

与 P1/P2 的 `summarize_cart_coast.py` 同一模式：入口只负责采样，离线重算负责判读，
于是判据可以在没有仿真器的机器上用**真实轨迹**复算（新判据就是拿实跑数据定的）。

判据对应计划 P4：「`v_R ≈ v_L` 以及稳定拖曳阶段 `T(t)` 是否进入相对稳定区间」。

- `v_R ≈ v_L`：稳态窗两体平均 vx 之差 ≤ 0.1 m/s；
- 机器人真的在按指令走：`steady_tracking_ratio ≥ 0.5`（只看「两体同速」有洞：两者都静止时
  它也成立，实测踩过——机器人被拽倒后 `speed_gap` 只有 0.028）；
- 绳真的在拉：指令阶段必须出现过张力，且收掉初始松弛（`L0 - slack`）之后不再频繁回到松弛；
- `T(t)` 稳定：用**稳态窗前后的均值漂移**判定，而不是方差。步态会给张力带来 ~7.5 Hz 的纹波
  （实测主要由阻尼项 `c·ḋ` 贡献，弹簧项只波动 1.06 N），那是正常现象；计划要的是
  「相对稳定区间」，看的是水平是否稳定。纹波大小仍作为 `tension_ripple_ratio` 报告出来。
"""

import csv
import json
import math
from pathlib import Path

# 指令阶段前 20% 视为「收初始松弛 + 起步」过渡：那一段的 T = 0 是设计预期
TAKEUP_FRACTION = 0.2
SPEED_GAP_LIMIT_MPS = 0.1
TRACKING_RATIO_MIN = 0.5
TENSION_DRIFT_LIMIT = 0.25
TAIL_SLACK_LIMIT = 0.05
PITCH_LIMIT_RAD = 0.6


def _mean(values):
    return sum(values) / len(values)


def _pstdev(values):
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def _spread(rows, key):
    return _pstdev([float(r[key]) for r in rows])


def summarize_tow(rows, *, user_command, takeup_fraction=TAKEUP_FRACTION):
    """由逐物理步记录算出汇总与判据；`rows` 为 `tow.csv` 的字典列表。"""
    if not rows:
        raise ValueError("拖动记录为空")
    if not math.isfinite(user_command) or user_command <= 0:
        raise ValueError("用户指令必须为有限正数")
    command_rows = [r for r in rows if float(r["user_cmd_mps"]) > 0.0]
    settle_rows = [r for r in rows if float(r["user_cmd_mps"]) == 0.0]
    if not command_rows:
        raise ValueError("指令阶段一个样本都没有")

    window = command_rows[len(command_rows) // 2:]           # 后半段当稳态窗
    takeup = int(len(command_rows) * takeup_fraction)
    tension = [float(r["rope_tension_n"]) for r in command_rows]
    taut_rows = [r for r in command_rows if float(r["rope_tension_n"]) > 0.0]
    tail = command_rows[takeup:]
    half = len(window) // 2
    mean_first = _mean([float(r["rope_tension_n"]) for r in window[:half]])
    mean_second = _mean([float(r["rope_tension_n"]) for r in window[half:]])
    steady_tension = _mean([float(r["rope_tension_n"]) for r in window])

    summary = {
        "state": "completed",
        "settle_samples": len(settle_rows),
        "command_samples": len(command_rows),
        "steady_window_s": [window[0]["time_s"], window[-1]["time_s"]],
        "steady_robot_vx_mps": _mean([float(r["robot_vx_mps"]) for r in window]),
        "steady_load_vx_mps": _mean([float(r["load_vx_mps"]) for r in window]),
        "steady_tension_n": steady_tension,
        "steady_tension_std_n": _spread(window, "rope_tension_n"),
        "steady_distance_m": _mean([float(r["rope_distance_m"]) for r in window]),
        "mean_pitch_rad": _mean([float(r["body_pitch_rad"]) for r in window]),
        "max_abs_pitch_rad": max(abs(float(r["body_pitch_rad"])) for r in rows),
        "tension_peak_n": max(tension),
        "tension_slack_fraction": sum(1 for t in tension if t == 0.0) / len(tension),
        "time_to_taut_s": float(taut_rows[0]["time_s"]) if taut_rows else None,
        "tension_slack_fraction_after_takeup": (
            sum(1 for r in tail if float(r["rope_tension_n"]) == 0.0) / len(tail) if tail else None),
        "final_load_x_m": float(rows[-1]["load_x_m"]),
        "final_robot_x_m": float(rows[-1]["robot_x_m"]),
        "steady_robot_z_m": _mean([float(r["robot_z_m"]) for r in window]),
        "min_robot_z_m": min(float(r["robot_z_m"]) for r in rows),
        "settle_load_drift_m": (float(settle_rows[-1]["load_x_m"]) - float(settle_rows[0]["load_x_m"])
                                if settle_rows else None),
        "settle_max_abs_load_vx_mps": (max(abs(float(r["load_vx_mps"])) for r in settle_rows)
                                       if settle_rows else None),
    }
    summary["steady_speed_gap_mps"] = summary["steady_robot_vx_mps"] - summary["steady_load_vx_mps"]
    summary["steady_tracking_ratio"] = summary["steady_robot_vx_mps"] / user_command
    summary["tension_drift_ratio"] = (abs(mean_second - mean_first) / steady_tension
                                     if steady_tension > 0 else None)
    summary["tension_ripple_ratio"] = (summary["steady_tension_std_n"] / steady_tension
                                      if steady_tension > 0 else None)

    failures = []
    if abs(summary["steady_speed_gap_mps"]) > SPEED_GAP_LIMIT_MPS:
        failures.append("robot_and_load_speeds_differ")
    if summary["steady_tracking_ratio"] < TRACKING_RATIO_MIN:
        failures.append("robot_not_tracking_command")
    if summary["time_to_taut_s"] is None:
        failures.append("rope_never_taut_during_tow")
    elif summary["tension_slack_fraction_after_takeup"] > TAIL_SLACK_LIMIT:
        failures.append("rope_not_continuously_taut")
    if steady_tension <= 0.0:
        failures.append("no_steady_tension")
    elif summary["tension_drift_ratio"] > TENSION_DRIFT_LIMIT:
        failures.append("tension_not_steady")
    if summary["max_abs_pitch_rad"] > PITCH_LIMIT_RAD:
        failures.append("body_pitch_excessive")
    summary["failures"] = failures
    summary["valid"] = not failures
    return summary


def summarize_run(directory):
    """读一个运行目录里的 `tow.csv` 与 `config.json` 重算汇总。"""
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    with (directory / "tow.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    summary = summarize_tow(rows, user_command=float(config["user_command_mps"]))
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = summarize_run(args.run_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["valid"] else 1)
