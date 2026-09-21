"""Recompute a tow run's summary and verdict using only the Python standard library.

与 P1/P2 的 `summarize_cart_coast.py` 同一模式：入口只负责采样，离线重算负责判读，
于是判据可以在没有仿真器的机器上用**真实轨迹**复算（新判据就是拿实跑数据定的）。

三个阶段（`phase` 列；旧记录没有该列时按指令自动推断，见 `_phases`）：

- `station`：站定，指令 0，绳应当**全程松弛**；
- `tow`：指令 = `v_user`，判「稳定拖曳」；
- `coast`：阶跃把指令归零后的滑行段（计划 P6）：机器人停下、小车继续滑行，
  绳会松弛，甚至可能被追尾后重新绷紧。

判据对应计划 P4/P6：

- `v_R ≈ v_L`：`tow` 阶段稳态窗两体平均 vx 之差 ≤ 0.1 m/s；
- 机器人真的在按指令走：`steady_tracking_ratio ≥ 0.5`（只看「两体同速」有洞：两者都静止时
  它也成立，实测踩过——机器人被拽倒后 `speed_gap` 只有 0.028）；
- 绳真的在拉：`tow` 阶段必须出现过张力，且收掉初始松弛后不再频繁回到松弛；
- `T(t)` 稳定：用 `tow` 阶段稳态窗**前后半的均值漂移**判定，而不是方差。步态会给张力带来
  ~7.5 Hz 的纹波（实测主要由阻尼项 `c·ḋ` 贡献，弹簧项只波动 1.06 N），那是正常现象；
  计划要的是「相对稳定区间」，看的是水平是否稳定。纹波大小仍作为 `tension_ripple_ratio` 报告。
- `station` 阶段绳不得张紧：实测踩过——机器人按 0.35 m 出生（自然站高 0.284 m）会向前窜
  0.65 m/s，把绳多拉长约 0.06 m；初始松弛只有 0.05 m 时绳在站定阶段就被拉直，
  出现 73–86 N 猛拽，把小车甩出 0.24 m 自由滑行（看起来就像「小车自己会动」）。
- `coast` 阶段：报告小车滑行距离、最小间距（追尾风险）、张力归零耗时、是否重新绷紧。
"""

import csv
import json
import math
from pathlib import Path

TAKEUP_FRACTION = 0.2          # tow 阶段前 20% 视为「收初始松弛 + 起步」过渡
SPEED_GAP_LIMIT_MPS = 0.1
TRACKING_RATIO_MIN = 0.5
TENSION_DRIFT_LIMIT = 0.25
TAIL_SLACK_LIMIT = 0.05
PITCH_LIMIT_RAD = 0.6
SETTLE_TENSION_LIMIT_N = 1.0   # station 阶段张力超过此值 ⇒ 初始松弛量不够
STOP_ROBOT_VX_FRACTION = 0.2   # coast 段末机器人 vx 应降到指令的 20% 以下
SETTLE_ROBOT_TRAVEL_LIMIT_M = 0.15   # station 段机器人位移超过此值 ⇒ 启动窜动过大
SETTLE_LOAD_DRIFT_LIMIT_M = 0.02     # station 段小车位移超过此值 ⇒ 拖曳前它没静止


def _mean(values):
    return sum(values) / len(values)


def _pstdev(values):
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def _col(rows, key):
    return [float(r[key]) for r in rows]


def _phases(rows):
    """按 `phase` 列分段；旧记录没有该列时按指令推断。"""
    if "phase" in rows[0]:
        phases = {name: [r for r in rows if r["phase"] == name]
                  for name in ("station", "tow", "coast")}
        if not phases["tow"]:
            raise ValueError("指令阶段一个样本都没有")
        return phases
    # 旧记录：开头指令 0 = 站定；指令 > 0 = 拖曳；拖曳之后又是 0 = 滑行
    commanded = [i for i, r in enumerate(rows) if float(r["user_cmd_mps"]) > 0.0]
    if not commanded:
        raise ValueError("指令阶段一个样本都没有")
    return {"station": rows[:commanded[0]], "tow": rows[commanded[0]:commanded[-1] + 1],
            "coast": rows[commanded[-1] + 1:]}


def summarize_tow(rows, *, user_command, takeup_fraction=TAKEUP_FRACTION):
    """由逐物理步记录算出汇总与判据；`rows` 为 `tow.csv` 的字典列表。"""
    if not rows:
        raise ValueError("拖动记录为空")
    if not math.isfinite(user_command) or user_command <= 0:
        raise ValueError("用户指令必须为有限正数")
    phases = _phases(rows)
    station_rows, tow_rows, coast_rows = phases["station"], phases["tow"], phases["coast"]

    window = tow_rows[len(tow_rows) // 2:]                   # tow 阶段后半段当稳态窗
    takeup = int(len(tow_rows) * takeup_fraction)
    tail = tow_rows[takeup:]
    tension = _col(tow_rows, "rope_tension_n")
    taut_rows = [r for r in tow_rows if float(r["rope_tension_n"]) > 0.0]
    half = len(window) // 2
    mean_first = _mean(_col(window[:half], "rope_tension_n"))
    mean_second = _mean(_col(window[half:], "rope_tension_n"))
    steady_tension = _mean(_col(window, "rope_tension_n"))

    summary = {
        "state": "completed",
        "samples": {"station": len(station_rows), "tow": len(tow_rows), "coast": len(coast_rows)},
        "steady_window_s": [window[0]["time_s"], window[-1]["time_s"]],
        "steady_robot_vx_mps": _mean(_col(window, "robot_vx_mps")),
        "steady_load_vx_mps": _mean(_col(window, "load_vx_mps")),
        "steady_tension_n": steady_tension,
        "steady_tension_std_n": _pstdev(_col(window, "rope_tension_n")),
        "steady_distance_m": _mean(_col(window, "rope_distance_m")),
        "mean_pitch_rad": _mean(_col(window, "body_pitch_rad")),
        "max_abs_pitch_rad": max(abs(float(r["body_pitch_rad"])) for r in rows),
        "tension_peak_n": max(tension),
        "tension_slack_fraction": sum(1 for t in tension if t == 0.0) / len(tension),
        "time_to_taut_s": float(taut_rows[0]["time_s"]) if taut_rows else None,
        "tension_slack_fraction_after_takeup": (
            sum(1 for r in tail if float(r["rope_tension_n"]) == 0.0) / len(tail) if tail else None),
        "final_load_x_m": float(rows[-1]["load_x_m"]),
        "final_robot_x_m": float(rows[-1]["robot_x_m"]),
        "steady_robot_z_m": _mean(_col(window, "robot_z_m")),
        "min_robot_z_m": min(_col(rows, "robot_z_m")),
        # station 阶段（指令 0、绳本应松弛）：出生窜动与「绳是否被拉直」都要报出来
        "settle_max_tension_n": (max(_col(station_rows, "rope_tension_n")) if station_rows else None),
        "settle_max_abs_robot_vx_mps": (max(abs(v) for v in _col(station_rows, "robot_vx_mps"))
                                        if station_rows else None),
        "settle_load_drift_m": (float(station_rows[-1]["load_x_m"]) - float(station_rows[0]["load_x_m"])
                                if station_rows else None),
        # station 段机器人自己走了多少：出生窜动超过松弛量就会把绳拉直（实测 0.46 m 时
        # 直接把机器人拽翻），所以这是选出生高度/松弛量的关键读数
        "settle_robot_travel_m": (float(station_rows[-1]["robot_x_m"]) - float(station_rows[0]["robot_x_m"])
                                  if station_rows else None),
        "settle_max_abs_load_vx_mps": (max(abs(v) for v in _col(station_rows, "load_vx_mps"))
                                       if station_rows else None),
    }
    summary["steady_speed_gap_mps"] = summary["steady_robot_vx_mps"] - summary["steady_load_vx_mps"]
    summary["steady_tracking_ratio"] = summary["steady_robot_vx_mps"] / user_command
    summary["tension_drift_ratio"] = (abs(mean_second - mean_first) / steady_tension
                                     if steady_tension > 0 else None)
    summary["tension_ripple_ratio"] = (summary["steady_tension_std_n"] / steady_tension
                                      if steady_tension > 0 else None)

    # 收松弛的耗时与机器人走过多远才发力：设计上应当 ≈ `--slack`（可直接核对）
    if taut_rows:
        summary["takeup_time_s"] = float(taut_rows[0]["time_s"]) - float(tow_rows[0]["time_s"])
        summary["takeup_robot_travel_m"] = (float(taut_rows[0]["robot_x_m"])
                                            - float(tow_rows[0]["robot_x_m"]))
    else:
        summary["takeup_time_s"] = None
        summary["takeup_robot_travel_m"] = None

    if coast_rows:
        coast_tension = _col(coast_rows, "rope_tension_n")
        gaps = _col(coast_rows, "rope_distance_m")
        slack_at = next((i for i, t in enumerate(coast_tension) if t == 0.0), None)
        re_taut = ([t for t in coast_tension[slack_at + 1:] if t > 0.0]
                   if slack_at is not None else [])
        summary.update({
            "coast_duration_s": float(coast_rows[-1]["time_s"]) - float(coast_rows[0]["time_s"]),
            "coast_load_travel_m": float(coast_rows[-1]["load_x_m"]) - float(coast_rows[0]["load_x_m"]),
            "coast_robot_travel_m": float(coast_rows[-1]["robot_x_m"]) - float(coast_rows[0]["robot_x_m"]),
            "coast_gap_at_stop_m": gaps[0],
            "coast_min_gap_m": min(gaps),
            "coast_peak_tension_n": max(coast_tension),
            "coast_time_to_slack_s": (float(coast_rows[slack_at]["time_s"])
                                      - float(coast_rows[0]["time_s"])) if slack_at is not None else None,
            "coast_retension_peak_n": max(re_taut) if re_taut else 0.0,
            "coast_final_robot_vx_mps": float(coast_rows[-1]["robot_vx_mps"]),
            "coast_final_load_vx_mps": float(coast_rows[-1]["load_vx_mps"]),
        })

    failures = []
    # ---- station：绳不得被拉直（初始松弛量要够），且机器人不应自己窜出去
    if summary["settle_max_tension_n"] is not None and \
            summary["settle_max_tension_n"] > SETTLE_TENSION_LIMIT_N:
        failures.append("rope_taut_during_settle")
    if summary["settle_robot_travel_m"] is not None and \
            abs(summary["settle_robot_travel_m"]) > SETTLE_ROBOT_TRAVEL_LIMIT_M:
        failures.append("robot_lurches_during_settle")
    # 拖曳开始时小车必须已经静止在设计位置（不再用代码把它「摆正」，改为事后检查）
    if summary["settle_load_drift_m"] is not None and \
            abs(summary["settle_load_drift_m"]) > SETTLE_LOAD_DRIFT_LIMIT_M:
        failures.append("load_moved_during_settle")
    # ---- tow
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
    # ---- coast：机器人要真的停下；小车不得到顶到机器人（追尾）
    if coast_rows:
        if summary["coast_final_robot_vx_mps"] > STOP_ROBOT_VX_FRACTION * user_command:
            failures.append("robot_did_not_stop")
        if summary["coast_min_gap_m"] <= 0.0:
            failures.append("load_reached_robot")
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
