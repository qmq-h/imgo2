#!/usr/bin/env python3
"""按「地形段 × 连接类型」出拖曳测试的分段报表（纯标准库，不需要 torch / Isaac Sim）。

回答的问题（2026-10-17 用户口径）：

1. **平地**的速度跟踪；
2. **障碍段**（坡/减速带/波浪）里**只统计障碍段落**的速度跟踪，**上坡与下坡分开**；
3. **下坡**：机器人是否被小车影响（绳是否绷紧/松弛、是否超速），**步态是否明显变化**
   （步态周期与周期性强度，与平地对比）；
4. **启动**：关节是否超限、冲击是否过大；
5. **停止**：小车是否撞到机器人、机器人受到的冲击；
6. 以上**全部区分「绳」与「刚体」**（compliant + inextensible 归「绳」；rigid 归「刚体」）。

数据来源：`play_towing_test.py` 的 run 目录（必须 `--write-csv all`，否则没有逐行轨迹）：

    <run>/experiment.json            参数 + 场景 + 判定阈
    <run>/summaries/<case>.json      逐 case 全量指标（startup / speed / stop / lane / stability）
    <run>/envNNNN_cCCR.._<model>_.../tow.csv   逐 5 物理步（25 ms）一行

用法：

    python3 imgo2_rl/scripts/tools/towing_segment_report.py --latest
    python3 imgo2_rl/scripts/tools/towing_segment_report.py <run> [<run2> ...] \
        [--profile hill|rolling] [--out DIR] [--csv /tmp/seg.csv] [--min-cases 3]

多给几个 run 时按 `--velocities` 的**速度档**分组（每个速度档一张表），因为它们必须用不同
的 `--tow-duration` 才能都走到障碍段（0.5 m/s 要 ~20 s；按里程停车后不需要分速度档）。

约定（与训练侧/测试台同源，不复制魔数）：

- 分段边界来自 `play_test_stats.segment_of`（它反过来读 `mdp/` 的几何常量）；
- 地形族来自 `connection_grid.terrain_family()`（`smooth` / `speedbump` / `wave`）；
- 「绳」= compliant + inextensible（两者的差别只在弹性，不是刚体），「刚体」= rigid。

判读口径（写进输出，避免误读）：

- 分段的速度误差用**体系 x 速度**（`robot_vx_b_mps`，与冻结策略的观测同口径），
  平地上它≈世界系；误差 = `robot_vx_b − user_cmd`（带符号，正 = 超速）；
- 每个 case **先按段内平均**，再跨 case 取中位数与分位数（不把同一 case 的 200 行当 200 个
  独立样本，避免伪重复）；
- 步态周期：对 FL 大腿角（`robot_jp_01`）在段内做自相关，取 0.20–0.90 s 内第一个峰值；
  周期性强度 = 该峰值的归一化自相关（0–1，越低越不像稳定步态）；行数 < 40 的段不估周期；
- 启动冲击：`tow` 段起拖后 0.2 s 内 `rope_tension_n` / `rope_impulse_ns` 的最大值，
  以及 `t = 0.2 s` 时的机器人 |vx − cmd|；
- 停车冲击：`stop` 后 0.2 s 内机器人 |vx| 的最大值（=`stop` 那一刻它还在动多少）、
  车斗接触峰值力 `deck_contact_peak_n`、最小几何间隙与是否接触。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

RL_ROOT = Path(__file__).resolve().parents[2]          # <repo>/imgo2_rl
DEFAULT_ROOT = RL_ROOT / "logs" / "towing" / "play_test"
TOOLS_DIR = Path(__file__).resolve().parent
for _path in (str(TOOLS_DIR), str(RL_ROOT / "source/imgo2_rl")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import play_test_stats as stats                                 # noqa: E402

# `connection_grid` 在 towing/mdp 下；`play_test_stats._mdp_path()` 把它加进 sys.path，
# 这样两边读的是**同一份**几何常量（不在本工具里另抄一套）。
stats._mdp_path()
import connection_grid                                          # noqa: E402

#: 连接类型的归类：用户要的是「绳子 vs 刚体」两大类。
CONNECTION_CLASS = {"compliant": "绳", "inextensible": "绳", "rigid": "刚体"}
CONNECTION_ORDER = ("绳", "刚体")

#: 地形族标签（显示顺序）。`smooth` 按坡度量级再分。
FAMILY_ORDER = ("平地", "5°坡", "10°坡", "减速带", "波浪")
OBSTACLE_FAMILIES = ("5°坡", "10°坡", "减速带", "波浪")

#: 段名（`play_test_stats.segment_of` 的返回值里去掉「平地」前缀的四种）。
OBSTACLE_SEGMENTS = ("上坡", "坡顶", "带顶", "下坡")

#: 自相关的滞后范围（秒）：步态周期 0.20–0.90 s。
GAIT_MIN_S, GAIT_MAX_S = 0.20, 0.90
#: 估步态至少需要的样本数（40 行 = 1 s @ 40 Hz）。
GAIT_MIN_SAMPLES = 40
#: 「真正停住」的速度阈值（m/s）：与测试台 `stop.settle_time_s` 的判据一致。
REST_SPEED_MPS = 0.05


# ------------------------------------------------------------------ 基础工具
def latest_run(root: Path = DEFAULT_ROOT) -> Path:
    return stats.latest_run(root)


def family_of(column: int) -> str:
    """列号 → 地形族标签（平地/5°坡/10°坡/减速带/波浪）。"""
    family = connection_grid.terrain_family(column)
    if family == connection_grid.SMOOTH_TERRAIN:
        grade = connection_grid.slope_degrees(column, 0)
        return "平地" if grade == 0.0 else f"{grade:g}°坡"
    return "减速带" if family == connection_grid.SPEEDBUMP_TERRAIN else "波浪"


def connection_class(model: str) -> str:
    try:
        return CONNECTION_CLASS[model]
    except KeyError:                                  # 未知连接类型不静默归类
        raise SystemExit(f"未知连接类型 {model!r}，无法归到 绳/刚体（已知：{sorted(CONNECTION_CLASS)}）")


def quantile(values, fraction: float):
    """线性插值分位数（与 `statistics.quantiles` 的 inclusive 口径一致，但不要求 n ≥ 2）。"""
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * min(max(fraction, 0.0), 1.0)
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def spread(values):
    """(样本数, 中位数, p10, p90)；空集返回 (0, None, None, None)。"""
    if not values:
        return 0, None, None, None
    return (len(values), statistics.median(values), quantile(values, 0.10), quantile(values, 0.90))


def fmt(value, digits=3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float) and not math.isfinite(value):
        return "—"
    return f"{value:.{digits}f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)


def cell_spread(values, digits=3) -> str:
    """`中位数 [p10, p90]` 的紧凑写法。"""
    count, median, low, high = spread(values)
    if count == 0:
        return "—"
    return f"{fmt(median, digits)} [{fmt(low, digits)}, {fmt(high, digits)}]"


# ------------------------------------------------------------------ 逐 case 读取
def load_run(run: Path):
    """读一个 run：返回 (experiment, case 列表, 逐 case 的 CSV 目录)。"""
    if not (run / "experiment.json").is_file():
        raise SystemExit(f"{run} 不是测试台 run（缺 experiment.json）")
    experiment = json.loads((run / "experiment.json").read_text(encoding="utf-8"))
    summaries = {path.stem: json.loads(path.read_text(encoding="utf-8"))
                 for path in sorted((run / "summaries").glob("*.json"))}
    cases = []
    for directory in sorted(run.glob("env*")):
        if not (directory / "tow.csv").is_file():
            continue
        matched = [summary for name, summary in summaries.items()
                   if name == directory.name]
        summary = matched[0] if matched else None
        cases.append({"dir": directory, "summary": summary})
    if not cases:
        raise SystemExit(
            f"{run} 下没有 env*/tow.csv —— 本工具需要逐行轨迹，跑测试台时必须 "
            f"`--write-csv all`（`failed` 只有失败 case 才落盘，`none` 一个都不写）")
    return experiment, cases


def case_info(entry, experiment):
    """case 的地形/连接/速度/质量/长度（优先用 summary，回退到文件名）。"""
    summary = entry["summary"]
    if summary is not None and "case" in summary:
        info = dict(summary["case"])
    else:                                            # 只有 CSV 没有 summary（例如手工裁剪过）
        name = entry["dir"].name
        # 目录名：env0000_c00r00_g5_v1_compliant_m5kg[_nocart]
        matched = re.search(r"_c(\d+)r(\d+)_g[^_]*_v([0-9.]+)_([A-Za-z]+)_m([0-9.]+)kg", name)
        if not matched:
            raise ValueError(f"目录名不符合测试台命名：{name}")
        column, row = int(matched.group(1)), int(matched.group(2))
        info = {"column": column, "row": row, "connection": matched.group(4),
                "grade_deg": connection_grid.slope_degrees(column, row),
                "velocity_mps": float(matched.group(3)),
                "cart_mass_kg": float(matched.group(5)),
                "connection_length_m": None, "env_index": None, "cell_index": None,
                "cart_present": "_nocart" not in name}
    info["family"] = family_of(int(info["column"]))
    info["connection_class"] = connection_class(info["connection"])
    return info


# ------------------------------------------------------------------ 步态周期
def gait_period(samples, sample_dt: float):
    """自相关估步态周期：返回 (周期[s] 或 None, 周期性强度 0–1)。

    去均值后做归一化自相关，在 `GAIT_MIN_S…GAIT_MAX_S` 的滞后里取**第一个**局部峰值
    （第一个峰 = 步频，不是它的谐波）；峰值本身当周期性强度。
    """
    if len(samples) < GAIT_MIN_SAMPLES:
        return None, None
    mean = sum(samples) / len(samples)
    centered = [value - mean for value in samples]
    variance = sum(value * value for value in centered)
    if variance <= 1e-12:
        return None, None
    low = max(1, int(round(GAIT_MIN_S / sample_dt)))
    high = min(len(centered) - 2, int(round(GAIT_MAX_S / sample_dt)))
    if high <= low:
        return None, None
    correlations = {}
    for lag in range(low, high + 1):
        total = sum(centered[index] * centered[index + lag]
                    for index in range(len(centered) - lag))
        correlations[lag] = total / variance
    best_lag, best_value = None, None
    for lag in range(low + 1, high):
        value = correlations[lag]
        if value > 0 and value >= correlations[lag - 1] and value >= correlations[lag + 1]:
            best_lag, best_value = lag, value
            break
    if best_lag is None:
        return None, None
    return best_lag * sample_dt, max(0.0, min(1.0, best_value))


# ------------------------------------------------------------------ 主聚合
def collect(run_dirs, *, profile="hill", limit=None):
    """读所有 run，返回逐 case 的分段与事件记录。

    返回结构：

        per_case:  [ {info, speed, segments: {段名: {…行级指标…}}, events: {…启动/停车…}} ]
        skipped:   [ (run, case 目录名, 原因) ]
    """
    per_case, skipped = [], []
    for run in run_dirs:
        experiment, cases = load_run(run)
        profile = experiment.get("arguments", {}).get("terrain_profile", profile)
        sample_dt = float(experiment.get("arguments", {}).get("dt", 0.02)) * \
            float(experiment.get("arguments", {}).get("record_every", 5))
        if limit and len(per_case) >= limit:
            break
        for entry in cases:
            if limit and len(per_case) >= limit:
                break
            info = case_info(entry, experiment)
            try:
                record = read_case(entry, info, profile, sample_dt)
            except (KeyError, ValueError) as error:
                skipped.append((run, entry["dir"].name, f"{type(error).__name__}: {error}"))
                continue
            per_case.append(record)
    return per_case, skipped


def read_case(entry, info, profile, sample_dt):
    """读一个 case：分段行级指标 + 启动/停车事件。"""
    rows = []
    with (entry["dir"] / "tow.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows.append(row)
    if not rows:
        raise ValueError("tow.csv 为空")

    segments = defaultdict(list)
    for row in rows:
        progress = float(row["robot_progress_m"])
        segments[stats.segment_of(progress, stats.profile_for_column(
            profile, info["column"]))].append(row)

    summary = entry["summary"] or {}
    metrics = summary.get("metrics", {})
    segment_metrics = {}
    for name, group in segments.items():
        segment_metrics[name] = segment_row_metrics(group, sample_dt)
    return {
        "case": info,
        "speed": info.get("velocity_mps"),
        "segments": segment_metrics,
        "events": event_metrics(rows, metrics, summary),
        "meta": {"run": str(entry["dir"].parent.name), "dir": entry["dir"].name,
                 "summary": bool(entry["summary"]), "rows": len(rows)},
    }


def segment_row_metrics(rows, sample_dt):
    """一段内行级指标（只统计拖曳态的行做速度跟踪；滑行态只用于停车段）。"""
    tow = [row for row in rows if row["phase"] == "tow"]
    errors = [float(row["robot_vx_b_mps"]) - float(row["user_cmd_mps"]) for row in tow]
    speeds = [float(row["robot_vx_b_mps"]) for row in tow]
    commands = [float(row["user_cmd_mps"]) for row in tow]
    joints = [float(row["joint_err_rms_rad"]) for row in tow]
    tensions = [float(row["rope_tension_n"]) for row in tow]
    taut = [float(row["rope_taut"]) for row in tow]
    thigh = [float(row["robot_jp_01"]) for row in tow]
    period, strength = gait_period(thigh, sample_dt)
    speed_mean = statistics.fmean(speeds) if speeds else None
    command_mean = statistics.fmean(commands) if commands else None
    return {
        "samples": len(tow),
        "abs_err": statistics.fmean(abs(value) for value in errors) if errors else None,
        "signed_err": statistics.fmean(errors) if errors else None,
        "ratio": (speed_mean / command_mean) if (speed_mean and command_mean) else None,
        "vx_std": statistics.pstdev(speeds) if len(speeds) > 1 else None,
        "joint_err": statistics.fmean(joints) if joints else None,
        "tension_mean": statistics.fmean(tensions) if tensions else None,
        "tension_p90": quantile(tensions, 0.90),
        "slack_frac": (sum(1 for value in taut if value <= 0.5) / len(taut)) if taut else None,
        "gait_period": period,
        "gait_strength": strength,
    }


def event_metrics(rows, metrics, summary):
    """启动（关节超限 / 冲击）与停车（接触 / 冲击）的逐 case 指标。"""
    dt = float(rows[1]["time_s"]) - float(rows[0]["time_s"]) if len(rows) > 1 else 0.025
    tow = [row for row in rows if row["phase"] == "tow"]
    coast = [row for row in rows if row["phase"] == "coast"]
    startup = metrics.get("startup", {})
    stop = metrics.get("stop", {})

    def first_window(window_rows, width=0.2):
        if not window_rows:
            return []
        base = float(window_rows[0]["time_s"])
        return [row for row in window_rows if float(row["time_s"]) <= base + width]

    window = first_window(tow)
    takeup_tension = max((float(row["rope_tension_n"]) for row in window), default=None)
    takeup_impulse = max((float(row["rope_impulse_ns"]) for row in window), default=None)
    takeup_slip = max((abs(float(row["robot_vx_b_mps"]) - float(row["user_cmd_mps"]))
                       for row in window), default=None)
    window = first_window(coast)
    robot_move = max((abs(float(row["robot_vx_mps"])) for row in window), default=None)
    deck_force = max((abs(float(row["cart_deck_fx_n"])) for row in window), default=None)
    # 「cmd=0 之后真的停住」：看 coast 段**最后一行**的机器人速度（|vx| ≤ REST_SPEED_MPS）
    final_vx = abs(float(coast[-1]["robot_vx_mps"])) if coast else None
    return {
        "startup_joint_max": startup.get("joint_max_rad"),
        "startup_joint_rms": startup.get("joint_rms_rad"),
        "startup_over_rms_limit": (startup.get("joint_rms_rad") is not None
                                   and startup["joint_rms_rad"] > 0.1),
        "startup_over_max_limit": (startup.get("joint_max_rad") is not None
                                   and startup["joint_max_rad"] > 0.3),
        "startup_torque_sat": startup.get("torque_saturated_frac"),
        "startup_time90": startup.get("time_to_90pct_s"),
        "startup_overshoot": startup.get("overshoot_ratio"),
        "takeup_tension": takeup_tension,
        "takeup_impulse": takeup_impulse,
        "takeup_slip": takeup_slip,
        "stop_contact": stop.get("contact"),
        "stop_min_clearance": stop.get("min_clearance_coast_m"),
        "stop_clearance_at_stop": stop.get("clearance_at_stop_m"),
        "stop_deck_peak": stop.get("deck_contact_peak_n"),
        "time_to_contact": stop.get("time_to_contact_after_stop_s"),
        "cart_coast_distance": stop.get("cart_coast_distance_m"),
        "stop_robot_move": robot_move,
        "stop_deck_force_window": deck_force,
        "stop_joint_max": stop.get("joint_max_rad"),
        "stop_joint_rms": stop.get("joint_rms_rad"),
        "stop_settle_s": stop.get("settle_time_s"),
        "stop_final_vx": final_vx,
        "stop_rest": (final_vx is not None and final_vx <= REST_SPEED_MPS),
        "stop_coast_samples": len(coast),
        "verdict": (summary.get("verdict") or metrics.get("verdict") or {}).get("code"),
        "dt": dt,
    }


# ------------------------------------------------------------------ 分组
def group_cases(per_case, key_fn):
    groups = defaultdict(list)
    for record in per_case:
        groups[key_fn(record)].append(record)
    return groups


def segment_values(records, segment, field):
    values = []
    for record in records:
        value = record["segments"].get(segment, {}).get(field)
        if value is not None:
            values.append(value)
    return values


def speed_matrix(per_case, *, field, families, segments, connection, speeds):
    """段 × 速度矩阵：rows = (族, 段)，cols = 速度档，格 = 跨 case 中位数 [p10,p90]。"""
    table = {}
    for family in families:
        for segment in segments:
            cells = []
            for speed in speeds:
                values = []
                for record in per_case:
                    if (record["case"]["family"] == family
                            and record["case"]["connection_class"] == connection
                            and record["speed"] == speed):
                        value = record["segments"].get(segment, {}).get(field)
                        if value is not None:
                            values.append(value)
                cells.append(cell_spread(values) if values else "—")
            if any(cell != "—" for cell in cells):
                table[(family, segment)] = cells
    return table


def event_values(records, field):
    values = []
    for record in records:
        value = record["events"].get(field)
        if value is not None:
            values.append(value)
    return values


# ------------------------------------------------------------------ 报表
def build_report(per_case, *, profile, skipped=()):
    """返回 markdown 行列表（纯字符串，便于离线测试逐行断言）。"""
    lines = ["# 拖曳分段报表（绳 vs 刚体）", ""]
    speeds = sorted({record["speed"] for record in per_case if record["speed"] is not None})
    families = [name for name in FAMILY_ORDER
                if any(record["case"]["family"] == name for record in per_case)]
    lines += [f"- 地形剖面：`{profile}`；case 数：{len(per_case)}；"
              f"地形族：{'、'.join(families)}；速度档：{', '.join(f'{s:g}' for s in speeds) or '—'}",
              f"- 连接归类：绳 = compliant + inextensible；刚体 = rigid",
              "- **2026-10-19 收束口径**：本报表是**诊断**，不是 pass/fail；验收看 "
              "`play_towing_test.py` 的任务口径（小车速度稳定性 / 启停冲击 / 重车不翻车）。"
              "停车接触、几何间隙、机器人停车后是否往前走**都不是失败判据**；"
              "停车撞击只看冲击量（关节/力矩），本报表的接触列仅作诊断。",
              "- 每格 = 跨 case 的中位数 [p10, p90]（先按 case 在段内取平均，再跨 case 取分位）"]
    if skipped:
        lines.append(f"- ⚠ 跳过 {len(skipped)} 个 case（缺列/空文件），原因见 JSON `skipped`")
    moving = [record for record in per_case
              if record["events"].get("stop_rest") is False]
    if moving:
        lines.append(f"- ⚠ **{len(moving)}/{len(per_case)} 个 case 在 coast 结束时仍在动**"
                     f"（末速 > {REST_SPEED_MPS:g} m/s）⇒ 停车段没走完，把 `--coast-duration` 加大"
                     f"（或看第 5 节「coast 末已停住」列）")
    elif any(record["events"].get("stop_rest") for record in per_case):
        lines.append(f"- ✅ 停车段走完：全部 case 在 coast 结束时 |vx| ≤ {REST_SPEED_MPS:g} m/s")
    lines.append("")

    weights = [(name, cls) for name in families for cls in CONNECTION_ORDER]

    def table(title, header, rows, description=None):
        lines.append(f"## {title}")
        if description:
            lines.append(description)
        lines.append("")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join("---" for _ in header) + "|")
        for label, cells in rows:
            lines.append("| " + " | ".join([label] + cells) + " |")
        lines.append("")

    # ---- 1) 平地速度跟踪
    def flat_payloads(records):
        """每个 case 把所有「平地*」段合并成一个平均（先按 case 平均，再跨 case 取分位）。"""
        out = []
        for record in records:
            payloads = [payload for name, payload in record["segments"].items()
                        if name.startswith("平地") and payload.get("samples")]
            if not payloads:
                continue
            out.append({
                "abs_err": statistics.fmean(p["abs_err"] for p in payloads),
                "signed_err": statistics.fmean(p["signed_err"] for p in payloads),
                "ratio": (statistics.fmean(p["ratio"] for p in payloads
                                           if p["ratio"] is not None)
                          if any(p["ratio"] is not None for p in payloads) else None),
                "joint_err": statistics.fmean(p["joint_err"] for p in payloads
                                              if p["joint_err"] is not None),
            })
        return out

    rows = []
    for cls in CONNECTION_ORDER:
        records = [record for record in per_case if record["case"]["connection_class"] == cls]
        payloads = flat_payloads(records)
        if not payloads:
            rows.append((cls, ["—", "—", "—", "—"]))
            continue
        rows.append((cls, [
            f"{len(payloads)}",
            f"{cell_spread([p['abs_err'] for p in payloads])} m/s",
            f"{cell_spread([p['ratio'] for p in payloads if p['ratio'] is not None], 3)}",
            f"{cell_spread([p['signed_err'] for p in payloads], 3)} m/s",
            f"{cell_spread([p['joint_err'] for p in payloads if p['joint_err'] is not None])} rad",
        ]))
    table("1. 平地：速度跟踪", ["连接", "case 数", "|vx−cmd| 中位数 [p10,p90]", "跟速比 vx/cmd",
                          "带符号误差（正 = 超速）", "关节跟踪误差"], rows,
          "每个 case 先把它所有「平地*」段（出生/带前/带后/出口）平均，再跨 case 取分位；"
          "误差用体系 x 速度（与冻结策略观测同口径）。")

    # ---- 1b) 平地 × 速度档
    speeds = sorted({record["speed"] for record in per_case if record["speed"] is not None})
    rows = []
    for speed in speeds:
        for cls in CONNECTION_ORDER:
            records = [record for record in per_case
                       if record["case"]["connection_class"] == cls and record["speed"] == speed]
            payloads = flat_payloads(records)
            if not payloads:
                continue
            rows.append((f"{speed:g} m/s · {cls}", [
                f"{cell_spread([p['abs_err'] for p in payloads])} m/s",
                f"{cell_spread([p['ratio'] for p in payloads if p['ratio'] is not None], 3)}",
                f"{cell_spread([p['signed_err'] for p in payloads], 3)} m/s",
            ]))
    table("1b. 平地 × 速度档", ["速度 · 连接", "|vx−cmd|", "跟速比", "带符号误差"], rows)

    # ---- 2) 障碍段：上/下坡分开
    rows = []
    for family in OBSTACLE_FAMILIES:
        if family not in families:
            continue
        for segment in OBSTACLE_SEGMENTS:
            cells = []
            for cls in CONNECTION_ORDER:
                records = [record for record in per_case
                           if record["case"]["family"] == family
                           and record["case"]["connection_class"] == cls]
                values = segment_values(records, segment, "abs_err")
                ratios = segment_values(records, segment, "ratio")
                cells.append(f"{cell_spread(values)} m/s / {cell_spread(ratios, 3)}"
                             if values else "—")
            if all(cell == "—" for cell in cells):
                continue                                  # 该族没有这个分段（如波浪没有坡顶）
            rows.append((f"{family} · {segment}", cells))
    table("2. 障碍段：速度跟踪（上/下坡分开）", ["地形 · 段", "绳：|vx−cmd| / 跟速比",
                                   "刚体：|vx−cmd| / 跟速比"], rows,
          "只统计障碍段内的行（平地段的「带前/带后」不计入本表）；坡顶/带顶单列。")

    # ---- 3) 下坡：小车影响 + 步态变化
    rows = []
    for family in families:
        for cls in CONNECTION_ORDER:
            records = [record for record in per_case
                       if record["case"]["family"] == family
                       and record["case"]["connection_class"] == cls]
            down = [record["segments"].get("下坡", {}) for record in records]
            down = [payload for payload in down if payload.get("samples")]
            if not down:
                continue
            tension = [payload["tension_mean"] for payload in down
                       if payload.get("tension_mean") is not None]
            tension90 = [payload["tension_p90"] for payload in down
                         if payload.get("tension_p90") is not None]
            slack = [payload["slack_frac"] for payload in down
                     if payload.get("slack_frac") is not None]
            signed = [payload["signed_err"] for payload in down
                      if payload.get("signed_err") is not None]
            periods = [payload["gait_period"] for payload in down
                       if payload.get("gait_period")]
            strengths = [payload["gait_strength"] for payload in down
                         if payload.get("gait_strength") is not None]
            # 与同 case 出生平地的步态对比
            deltas = []
            for record in records:
                down_payload = record["segments"].get("下坡", {})
                flat_payload = record["segments"].get("平地(出生)", {})
                if down_payload.get("gait_period") and flat_payload.get("gait_period"):
                    deltas.append(abs(down_payload["gait_period"] - flat_payload["gait_period"])
                                  / flat_payload["gait_period"])
            rows.append((f"{family} · {cls}",
                         [f"{cell_spread(tension, 2)} N", f"{cell_spread(tension90, 2)} N",
                          f"{cell_spread(slack, 2)}",
                          f"{cell_spread(signed, 3)} m/s",
                          f"{cell_spread(periods, 3)} s",
                          f"{cell_spread(strengths, 2)}",
                          f"{cell_spread(deltas, 3)}"]))
    table("3. 下坡：小车影响与步态", ["地形 · 连接", "绳张力均值", "张力 p90", "松弛占比",
                              "速度误差（正 = 超速）", "步态周期", "周期强度",
                              "周期相对平地变化"], rows,
          "松弛占比 = `rope_taut ≤ 0.5` 的行比例；步态周期由 FL 大腿角自相关估（0.20–0.90 s）；"
          "周期强度越低越不像稳定步态；最后列是 |T_下坡 − T_平地| / T_平地。")

    # ---- 2b) 障碍段 × 速度（与旧 stats.txt 的「段 × 速度」矩阵同形）
    obstacle_segments = [name for name in OBSTACLE_SEGMENTS
                         if any(name in record["segments"] for record in per_case)]
    for cls in CONNECTION_ORDER:
        table_map = speed_matrix(per_case, field="abs_err", families=OBSTACLE_FAMILIES,
                                 segments=obstacle_segments, connection=cls, speeds=speeds)
        if not table_map:
            continue
        rows = [(f"{family} · {segment}", cells)
                for (family, segment), cells in table_map.items()]
        table(f"2b. 障碍段 × 速度（{cls}）：|vx−cmd|", ["地形 · 段"] + [f"{speed:g} m/s" for speed in speeds],
              rows, "与旧 `stats.txt` 的「段 × 速度」矩阵同形；空格 = 该档没有样本（例如远端段只有快档到得了）。")

    # ---- 3b) 下坡 × 速度（带符号速度误差，正 = 被推得超速）
    for cls in CONNECTION_ORDER:
        table_map = speed_matrix(per_case, field="signed_err", families=OBSTACLE_FAMILIES,
                                 segments=("下坡",), connection=cls, speeds=speeds)
        if not table_map:
            continue
        rows = [(f"{family} · 下坡", cells) for (family, segment), cells in table_map.items()]
        table(f"3b. 下坡 × 速度（{cls}）：带符号速度误差", ["地形"] + [f"{speed:g} m/s" for speed in speeds],
              rows, "正值 = 机器人比指令快（被小车推着走），负值 = 被拖慢或跟不上。")

    # ---- 4) 启动
    rows = []
    for cls in CONNECTION_ORDER:
        records = [record for record in per_case if record["case"]["connection_class"] == cls]
        if not records:
            continue
        over_rms = sum(1 for record in records if record["events"].get("startup_over_rms_limit"))
        over_max = sum(1 for record in records if record["events"].get("startup_over_max_limit"))
        rows.append((cls, [
            f"{over_rms}/{len(records)}",
            f"{over_max}/{len(records)}",
            cell_spread(event_values(records, "startup_joint_max"), 3) + " rad",
            cell_spread(event_values(records, "startup_joint_rms"), 3) + " rad",
            cell_spread(event_values(records, "startup_torque_sat"), 2),
            cell_spread(event_values(records, "takeup_tension"), 2) + " N",
            cell_spread(event_values(records, "takeup_impulse"), 3) + " N·s",
            cell_spread(event_values(records, "takeup_slip"), 3) + " m/s",
            cell_spread(event_values(records, "startup_time90"), 3) + " s",
        ]))
    table("4. 启动：关节超限与冲击", ["连接", "关节 RMS > 0.1 rad", "关节峰值 > 0.3 rad",
                               "关节峰值", "关节 RMS", "力矩饱和占比", "起拖张力峰值",
                               "起拖冲量峰值", "起拖 0.2 s 内速度偏差", "到 90% 指令用时"], rows,
          "关节超限两列是**超阈 case 数/总数**（阈值沿用测试台的 0.1 / 0.3 rad，即 JNT 口径）；"
          "张力/冲量/速度偏差取起拖后 0.2 s 窗口内的峰值。")

    # ---- 5) 停车
    rows = []
    for cls in CONNECTION_ORDER:
        records = [record for record in per_case if record["case"]["connection_class"] == cls]
        if not records:
            continue
        contacts = sum(1 for record in records if record["events"].get("stop_contact"))
        rested = sum(1 for record in records if record["events"].get("stop_rest"))
        rows.append((cls, [
            f"{contacts}/{len(records)}",
            f"{rested}/{len(records)}",
            cell_spread(event_values(records, "stop_final_vx"), 3) + " m/s",
            cell_spread(event_values(records, "stop_min_clearance"), 3) + " m",
            cell_spread(event_values(records, "time_to_contact"), 2) + " s",
            cell_spread(event_values(records, "stop_deck_peak"), 2) + " N",
            cell_spread(event_values(records, "stop_deck_force_window"), 2) + " N",
            cell_spread(event_values(records, "stop_robot_move"), 3) + " m/s",
            cell_spread(event_values(records, "stop_joint_max"), 3) + " rad",
            cell_spread(event_values(records, "cart_coast_distance"), 3) + " m",
            cell_spread(event_values(records, "stop_settle_s"), 2) + " s",
        ]))
    table("5. 停车：是否撞到机器人与冲击", ["连接", "发生接触", "coast 末已停住",
                                "coast 末机器人速度", "滑行段最小间隙",
                                "到接触用时", "车斗接触峰值力", "停车 0.2 s 内车斗力",
                                "机器人自身仍在动的速度", "停车关节峰值",
                                "小车滑行距离", "机器人停稳用时"], rows,
          "接触判定沿用测试台三路见证（接触力 / 几何间隙 / 负载单步速度跃变）；"
          "「机器人自身仍在动的速度」= STOP 后 0.2 s 内 |robot_vx| 最大值（它带着小车冲过来多少）。")

    # ---- 6) 判定码（补充）
    rows = []
    for cls in CONNECTION_ORDER:
        records = [record for record in per_case if record["case"]["connection_class"] == cls]
        codes = defaultdict(int)
        for record in records:
            codes[record["events"].get("verdict") or "INV"] += 1
        total = sum(codes.values()) or 1
        rows.append((cls, [f"{code} {count}（{count / total:.0%}）"
                           for code, count in sorted(codes.items(), key=lambda item: -item[1])]))
    table("6. 判定码分布", ["连接", "判定码"], [(label, [", ".join(cells)])
                                       for label, cells in rows],
          "测试台的判定码（OK / LOW / LAT / SPD / COL / FALL / JNT / INV），仅供参考。")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", type=Path, help="run 目录（缺省用 --latest）")
    parser.add_argument("--latest", action="store_true", help="用最新的 run")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="run 根目录")
    parser.add_argument("--profile", choices=("hill", "rolling"), default=None,
                        help="平滑段剖面（默认取 experiment.json 里的 terrain_profile）")
    parser.add_argument("--limit", type=int, default=None, help="只读前 N 个 case（调试用）")
    parser.add_argument("--out", type=Path, default=None, help="markdown 输出路径（默认写到 run 目录）")
    parser.add_argument("--json", type=Path, default=None, help="JSON 输出路径（默认同 --out）")
    args = parser.parse_args(argv)

    runs = list(args.runs)
    if not runs:
        runs = [latest_run(args.root)]
    for run in runs:
        if not run.is_dir():
            raise SystemExit(f"找不到 run 目录：{run}")
    profile = args.profile or json.loads(
        (runs[0] / "experiment.json").read_text(encoding="utf-8")).get(
        "arguments", {}).get("terrain_profile", "hill")
    per_case, skipped = collect(runs, profile=profile, limit=args.limit)
    if not per_case:
        raise SystemExit("没有可统计的 case")
    lines = build_report(per_case, profile=profile, skipped=skipped)
    text = "\n".join(lines)
    print(text)
    out = args.out or (runs[0] / "segment_report.md")
    out.write_text(text + "\n", encoding="utf-8")
    payload = {"runs": [str(run) for run in runs], "profile": profile,
               "cases": [{key: record[key] for key in ("case", "speed", "segments", "events")}
                         for record in per_case],
               "skipped": [[str(run), name, why] for run, name, why in skipped]}
    json_path = args.json or out.with_suffix(".json")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[out] {out}\n[out] {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
