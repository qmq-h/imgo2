#!/usr/bin/env python3
"""必要性测试台 run 的离线统计（纯标准库，不需要 torch / Isaac Sim）。

`play_towing_test.py` 的 `report.md` 只给「按坡度量级」的判定矩阵与分组统计；这份工具把
`summaries/<case>.json`（逐 env 全量指标）重新切成交叉表，用来回答「失败集中在哪一档」：

    判定码分布 / 失败原因计数 / 按 坡度·连接·质量·速度·行 切分的判定码与指标中位数 /
    关键指标分位数 / 间隙不可用计数 / 逐 case 明细（可选）

**可以读正在跑的 run**（`summaries/` 是逐 case 落盘的）：会显示「已判读 n/总数」。

用法：

    python3 imgo2_rl/scripts/tools/play_test_stats.py --latest
    python3 imgo2_rl/scripts/tools/play_test_stats.py imgo2_rl/logs/towing/play_test/<run>
    python3 imgo2_rl/scripts/tools/play_test_stats.py --latest --csv /tmp/play_stats.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

RL_ROOT = Path(__file__).resolve().parents[2]          # <repo>/imgo2_rl
DEFAULT_ROOT = RL_ROOT / "logs" / "towing" / "play_test"

#: 判定码按严重度排序（与 `play_towing_test.format_verdict_matrix` 的语义一致，仅用于展示顺序）
CODE_ORDER = ("FALL", "COL", "LOW", "JNT", "SPD", "LAT", "INV", "OK")


def latest_run(root: Path) -> Path:
    runs = sorted((p for p in root.glob("*") if (p / "experiment.json").is_file()),
                  key=lambda p: p.stat().st_mtime)
    if not runs:
        raise SystemExit(f"在 {root} 下找不到任何 run（缺 experiment.json）")
    return runs[-1]


def load_cases(run: Path) -> tuple[list[dict], dict]:
    experiment = json.loads((run / "experiment.json").read_text(encoding="utf-8"))
    cases = []
    for path in sorted(glob.glob(str(run / "summaries" / "*.json"))):
        try:
            cases.append(json.loads(Path(path).read_text(encoding="utf-8")))
        except json.JSONDecodeError:      # 正在写的半个文件
            continue
    return cases, experiment


def metric_of(case: dict) -> dict:
    """把嵌套的 summary 摊平成统计需要的几个量（缺失一律 None，不伪造 0）。"""
    m = case["metrics"]
    tw = m.get("summarize_tow", {})
    return {
        "code": m["verdict"]["code"],
        "reasons": list(m["verdict"]["reasons"]),
        "fell": bool(m["stability"]["fell"]),
        "fall_phase": m["stability"].get("fell_phase"),
        "min_base_z": m["stability"].get("min_robot_surface_height_m"),
        "startup_rms": m["startup"].get("joint_rms_rad"),
        "startup_max": m["startup"].get("joint_max_rad"),
        "speed_mae": m["speed"].get("mae_mps"),
        "speed_ratio": m["speed"].get("steady_ratio_mean"),
        "stop_min_clearance": m["stop"].get("min_clearance_coast_m"),
        "stop_contact": m["stop"].get("contact"),
        "coast_distance": m["stop"].get("cart_coast_distance_m"),
        "lateral_max": m["lane"].get("y_max_abs_m"),
        "heading_max_deg": (m["lane"].get("heading_max_abs_rad") or 0.0) * 57.29577951308232,
        "clearance_unavailable": tw.get("clearance_unavailable_samples"),
        "settle_tension": tw.get("settle_max_tension_n"),
    }


# ---------------------------------------------------------------- 剖面分段统计

#: lane 局部 x 的剖面分段（边界与 `mdp/slope_geometry.py` 同源，不另写一份魔数）
SEGMENT_BOUNDS = ((0.0, 2.25, "平地(出生)"), (2.25, 5.25, "上坡"), (5.25, 6.0, "坡顶"),
                  (6.0, 9.0, "下坡"), (9.0, float("inf"), "平地(出口)"))
SEGMENT_ORDER = ("平地(出生)", "上坡", "坡顶", "下坡", "平地(出口)")


def segment_of(progress_m: float) -> str:
    """`robot_progress_m`（相对 lane 原点的水平行程）→ 剖面分段名。"""
    if progress_m < 0.0:
        return "平地(出生)"
    for low, high, name in SEGMENT_BOUNDS:
        if low <= progress_m < high:
            return name
    return "平地(出口)"


def segment_stats(case_dirs, summaries, *, limit=None, stride=1):
    """按剖面分段统计 `q − q*`、力矩与跟速（读 `env*/tow.csv` 的拖曳段）。

    为什么需要它：`startup_joint_error` 只看拖曳开始后的第一个 1 s —— 那 1 s 里机器人几乎
    都在**出生平地**上（0.5 m/s 走 0.5 m、1.5 m/s 走 1.5 m，而出生平地段有 2.25 m），
    坡上发生的事完全没进指标。分段统计按 `robot_progress_m` 把每行归到
    平地(出生)/上坡/坡顶/下坡/平地(出口)，于是「误差随剖面怎么变」可以直接看出来，
    拖曳与无负载两轮用同一套口径直接对比。
    """
    by_slug = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in summaries}
    entries = []
    for case_dir in case_dirs:
        if case_dir.name not in by_slug:
            continue
        matched = re.match(r"env\d+_c(\d+)r(\d+)_", case_dir.name)
        column, row = (int(matched.group(1)), int(matched.group(2))) if matched else (0, 0)
        entries.append((column, row, case_dir))
    # 抽样必须**按 (列, 行) 网格**铺开：按目录序号等距会只落到少数几列上（实测 40 个样本
    # 只命中 col 0 与 col 20 ⇒ 10° 档一个都没有）。列方向默认取 10 档（含 0/5/10°），
    # 行方向按 limit 铺开。
    picked = [case_dir for _, _, case_dir in entries]
    if limit and len(entries) > limit:
        columns = sorted({column for column, _, _ in entries})
        rows = sorted({row for _, row, _ in entries})
        n_columns = min(len(columns), 10)
        n_rows = max(1, limit // n_columns)
        column_stride = max(1, len(columns) // n_columns)
        row_stride = max(1, len(rows) // n_rows)
        picked = [case_dir for column, row, case_dir in entries
                  if column % column_stride == 0 and row % row_stride == 0]
    groups = {}
    speed_matrix = {}
    for case_dir in picked:
        summary = by_slug[case_dir.name]
        grade = f"{summary['case']['grade_deg']:g}°"
        # 是否拖车：新 run 写在 case 字典里；旧 run 没有该字段时用文件名后缀兜底。
        cart_present = bool(summary["case"].get("cart_present", "_nocart" not in case_dir.name))
        with (case_dir / "tow.csv").open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row["phase"] != "tow":
                    continue
                segment = segment_of(float(row["robot_progress_m"]))
                speed = f"{float(row['user_cmd_mps']):g}"
                speed_matrix.setdefault(cart_present, {}).setdefault(segment, {}).setdefault(
                    speed, {"sumsq": 0.0, "n": 0})
                speed_matrix[cart_present][segment][speed]["sumsq"] += sum(
                    (float(row[f"robot_jp_{j:02d}"]) - float(row[f"robot_jt_{j:02d}"])) ** 2
                    for j in range(12)) / 12.0
                speed_matrix[cart_present][segment][speed]["n"] += 1
                key = (cart_present, grade, segment)
                bucket = groups.setdefault(key, {
                    "sum_e": [0.0] * 12, "sum_e2": [0.0] * 12, "sum_abs_tau": [0.0] * 12,
                    "sum_vx_err": 0.0, "sum_z": 0.0, "n": 0, "cases": set()})
                for joint in range(12):
                    error = (float(row[f"robot_jp_{joint:02d}"])
                             - float(row[f"robot_jt_{joint:02d}"]))
                    bucket["sum_e"][joint] += error
                    bucket["sum_e2"][joint] += error * error
                    bucket["sum_abs_tau"][joint] += abs(float(row[f"robot_tau_{joint:02d}"]))
                bucket["sum_vx_err"] += abs(float(row["robot_vx_b_mps"])
                                            - float(row["user_cmd_mps"]))
                bucket["sum_z"] += float(row["robot_z_m"])
                bucket["n"] += 1
                bucket["cases"].add(case_dir.name)
    return groups, picked, speed_matrix


def _segment_block(title, groups, presence, picked) -> dict:
    """打印某一边（拖曳 / 无负载）的分段表，并返回 {(坡度, 段): 指标字典} 供差值表使用。"""
    subset = {key: value for key, value in groups.items() if key[0] is presence}
    print(f"\n### {title}")
    if not subset:
        print("（这一边没有样本：无负载需要 `--no-cart-fraction > 0`；拖曳需要 < 1）")
        return {}
    print(f"{'坡度':>5s} {'剖面段':>11s} {'case':>5s} {'行':>6s} {'池化RMS':>8s} "
          f"{'静差|均值|':>10s} {'动态std':>8s} {'平均|τ|':>8s} {'|vx−cmd|':>9s} {'base z':>7s}")
    out = {}
    for grade in sorted({key[1] for key in subset}, key=lambda g: float(g.rstrip("°"))):
        for segment in SEGMENT_ORDER:
            bucket = subset.get((presence, grade, segment))
            if not bucket or bucket["n"] == 0:
                continue
            n = bucket["n"]
            rms = [math.sqrt(total / n) for total in bucket["sum_e2"]]
            means = [total / n for total in bucket["sum_e"]]
            stds = [math.sqrt(max(0.0, e2 / n - m * m))
                    for e2, m in zip(bucket["sum_e2"], means)]
            pooled = math.sqrt(sum(v * v for v in rms) / 12.0)
            static = sum(abs(v) for v in means) / 12.0
            dynamic = sum(stds) / 12.0
            vx_err = bucket["sum_vx_err"] / n
            out[(grade, segment)] = {"pooled": pooled, "static": static, "dynamic": dynamic,
                                     "vx_err": vx_err, "cases": len(bucket["cases"]), "rows": n}
            print(f"{grade:>5s} {segment:>11s} {len(bucket['cases']):5d} {n:6d} {pooled:8.3f} "
                  f"{static:10.3f} {dynamic:8.3f} "
                  f"{sum(bucket['sum_abs_tau']) / (n * 12):8.2f} {vx_err:9.3f} "
                  f"{bucket['sum_z'] / n:7.3f}")
    return out


def print_segments(groups, picked) -> None:
    print(f"\n## 按剖面分段（拖曳段；样本 {len(picked)} 个 case）")
    print("  段 = `robot_progress_m` 落点；q* = 当拍下发的关节目标；误差 = q − q*（12 关节）；"
          "静差 = 12 关节的 |mean(e)| 平均，动态 = 其 std 平均")
    print("  注：0° lane 的「上坡/坡顶/下坡」只是**剖面位置**（坡度本身是 0，base z 可见高度不变）；"
          "5°/10° lane 才是真坡。无负载（`_nocart`）与拖曳在同一轮 run 里同时出现时会给差值表。")
    towed = _segment_block("拖曳（有负载，cart present）", groups, True, picked)
    free = _segment_block("无负载（`_nocart` env）", groups, False, picked)
    shared = sorted(set(towed) & set(free), key=lambda key: (float(key[0].rstrip("°")), SEGMENT_ORDER.index(key[1])))
    if shared:
        print("\n### 差值（拖曳 − 无负载；正 = 拖曳更差）")
        print(f"{'坡度':>5s} {'剖面段':>11s} {'Δ池化RMS':>10s} {'Δ静差':>8s} {'Δ动态':>8s} {'Δ|vx−cmd|':>11s}")
        for key in shared:
            a, b = towed[key], free[key]
            print(f"{key[0]:>5s} {key[1]:>11s} {a['pooled'] - b['pooled']:+10.3f} "
                  f"{a['static'] - b['static']:+8.3f} {a['dynamic'] - b['dynamic']:+8.3f} "
                  f"{a['vx_err'] - b['vx_err']:+11.3f}")
    else:
        print("\n（没有可对比的 (坡度, 段)：需要同一轮 run 里既有拖曳也有无负载 env —— "
              "例如 `--no-cart-fraction 0.125`）")


def print_segment_speed_matrix(matrix) -> None:
    """段 × 速度 的池化 RMS（拖曳/无负载各一块）—— 远端段只有快档到得了，用它暴露这个混杂。"""
    for presence, label in ((True, "拖曳（有负载）"), (False, "无负载")):
        per_segment = matrix.get(presence)
        if not per_segment:
            continue
        speeds = sorted({speed for per_speed in per_segment.values() for speed in per_speed},
                        key=float)
        print(f"\n### 段 × 速度：{label}（池化 RMS，rad；`-` = 该组合没有样本）")
        print(f"{'剖面段':>11s} " + " ".join(f"{speed:>9s}" for speed in speeds))
        for segment in SEGMENT_ORDER:
            cells = []
            for speed in speeds:
                bucket = per_segment.get(segment, {}).get(speed)
                cells.append("        -" if not bucket else
                             f"{math.sqrt(bucket['sumsq'] / bucket['n']):9.3f}")
            print(f"{segment:>11s} " + " ".join(cells))


def med(values):
    clean = [v for v in values if isinstance(v, (int, float))]
    return statistics.median(clean) if clean else float("nan")


def fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, float) and value != value:
        return "n/a"
    return f"{value:.{digits}f}"


def code_summary(codes: Counter, n: int) -> str:
    parts = []
    for code in CODE_ORDER:
        if codes.get(code):
            parts.append(f"{code} {codes[code]}（{100.0 * codes[code] / max(n, 1):.0f}%）")
    return "，".join(parts) if parts else "（无）"


def quantiles(values) -> str:
    clean = sorted(v for v in values if isinstance(v, (int, float)))
    if not clean:
        return "n/a"

    def q(fraction):
        index = min(len(clean) - 1, max(0, int(round(fraction * (len(clean) - 1)))))
        return clean[index]

    return (f"min {clean[0]:.3f} / p25 {q(0.25):.3f} / 中位 {q(0.5):.3f} / "
            f"p75 {q(0.75):.3f} / max {clean[-1]:.3f}")


def group_table(title: str, groups: dict, expected: int) -> None:
    print(f"\n## {title}")
    print(f"{'分组':>10s} {'n':>4s} {'判定码（数/占比）':<46s} "
          f"{'起步RMS':>8s} {'跟速MAE':>8s} {'停车间隙':>8s} {'滑移m':>7s} {'|y|max':>7s}")
    for key in sorted(groups, key=lambda k: (isinstance(k, str), k)):
        rows = groups[key]
        codes = Counter(r["code"] for r in rows)
        print(f"{str(key):>10s} {len(rows):4d} {code_summary(codes, len(rows)):<46s} "
              f"{fmt(med([r['startup_rms'] for r in rows])):>8s} "
              f"{fmt(med([r['speed_mae'] for r in rows])):>8s} "
              f"{fmt(med([r['stop_min_clearance'] for r in rows])):>8s} "
              f"{fmt(med([r['coast_distance'] for r in rows])):>7s} "
              f"{fmt(med([r['lateral_max'] for r in rows])):>7s}")
    if len(groups) and expected and sum(len(v) for v in groups.values()) != expected:
        print(f"（注意：只读到 {sum(len(v) for v in groups.values())}/{expected} 个 case —— run 可能还没跑完）")


def cross_table(title: str, cases: list[dict], row_key, col_key) -> None:
    cells = defaultdict(Counter)
    col_keys = sorted({col_key(c) for c in cases}, key=str)
    for case in cases:
        cells[row_key(case)][col_key(case)] += 1
    print(f"\n## {title}（格 = case 数）")
    print(f"{'':>12s} " + " ".join(f"{str(k):>10s}" for k in col_keys))
    for key in sorted(cells, key=str):
        print(f"{str(key):>12s} " + " ".join(f"{cells[key].get(k, 0):>10d}" for k in col_keys))


# ---------------------------------------------------------------- 起步/停车窗口 与 越界率

def print_windows(cases) -> None:
    """起步/停车窗口（各 1 s）按「是否拖车」和坡度/速度切分 —— 回答「负载有没有带来关节冲击」。

    只读 `summaries/*.json`（不用逐 case 轨迹，秒级）。入参是原始 case summary 列表。
    注意这里**没有关节过冲**这个量：`joint_max_rad` 是窗口内单点最大 |q − q*|（最大偏差，不是过冲）；
    唯一叫 overshoot 的是 `startup.overshoot_ratio` = 窗口内 max(体系 vx)/指令 − 1，**是速度**。
    """
    print("\n## 起步/停车窗口（各 1 s，判 JNT 的两个窗口）")
    print("  起步 = 拖曳相位前 1 s；停车 = 滑行相位前 1 s；峰值 = 单点最大 |q − q*|（≠过冲）；"
          "速度过冲 = max(体系 vx)/指令 − 1（负 = 那 1 s 内没到指令）")

    def median(values):
        clean = [v for v in values if isinstance(v, (int, float))]
        return statistics.median(clean) if clean else float("nan")

    def gather(subset, section, key):
        return [c["metrics"][section].get(key) for c in subset]

    def row(label, subset):
        if not subset:
            return
        print(f"{label:>16s} {len(subset):4d} "
              f"{median(gather(subset, 'startup', 'joint_rms_rad')):8.3f} "
              f"{median(gather(subset, 'startup', 'joint_max_rad')):8.3f} "
              f"{median(gather(subset, 'stop', 'joint_rms_rad')):8.3f} "
              f"{median(gather(subset, 'stop', 'joint_max_rad')):8.3f} "
              f"{median(gather(subset, 'startup', 'torque_saturated_frac')):9.4f} "
              f"{median(gather(subset, 'startup', 'overshoot_ratio')):+9.3f} "
              f"{median(gather(subset, 'startup', 'time_to_90pct_s')):8.2f} "
              f"{median(gather(subset, 'startup', 'body_vx_rms_err_mps')):8.3f}")

    print(f"\n{'分组':>16s} {'n':>4s} {'起步RMS':>8s} {'起步峰值':>8s} {'停车RMS':>8s} "
          f"{'停车峰值':>8s} {'力矩饱和':>9s} {'速度过冲':>9s} {'90%时间':>8s} {'vxRMS误差':>8s}")
    for presence, label in ((True, "拖曳"), (False, "无负载")):
        subset = [c for c in cases if bool(c["case"].get("cart_present", True)) is presence]
        row(f"— {label} 全部", subset)
        for grade in sorted({c["case"]["grade_deg"] for c in subset}):
            row(f"{label} {grade:g}°", [c for c in subset if c["case"]["grade_deg"] == grade])
        for speed in sorted({c["case"]["velocity_mps"] for c in subset}):
            row(f"{label} {speed:g} m/s", [c for c in subset if c["case"]["velocity_mps"] == speed])


def print_crossings(case_dirs, summaries, *, limit=70, phase="tow"):
    """关节是否**越过设定点**：窗口内 `e = q − q*` 与窗口均值反号的样本占比 + 越界量 p95。

    ⚠️ 这不是经典过冲：`q*` 每 20 ms 被策略改写一次，来回穿越是常态；这套 PD 是 kp=25/kd=0.5，
    阻尼比很大、不具备欠阻尼振荡条件。要经典阶跃过冲得先用「单拍阶跃」或「瞬态 Mp」定义。
    需要 run 保留了 `env*/tow.csv`（`--write-csv all`）。
    """
    by_slug = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in summaries}
    grouped = {True: [], False: []}
    for case_dir in case_dirs:
        summary = by_slug.get(case_dir.name)
        if summary is None:
            continue
        present = bool(summary["case"].get("cart_present", "_nocart" not in case_dir.name))
        if len(grouped[present]) >= limit:
            continue
        csv_path = case_dir / "tow.csv"
        if not csv_path.is_file():
            continue
        with csv_path.open(encoding="utf-8") as stream:
            rows = [row for row in csv.DictReader(stream) if row["phase"] == phase]
        window = rows[:40]                     # 前 1 s（--record-every 5 ⇒ 40 行）
        if not window:
            continue
        fractions, overshoots = [], []
        for joint in range(12):
            errors = [float(r[f"robot_jp_{joint:02d}"]) - float(r[f"robot_jt_{joint:02d}"])
                      for r in window]
            mean = sum(errors) / len(errors)
            opposite = sorted((v for v in errors if v * mean < 0.0), key=abs)
            fractions.append(len(opposite) / len(errors))
            overshoots.append(abs(opposite[int(0.95 * (len(opposite) - 1))]) if opposite else 0.0)
        grouped[present].append((sum(fractions) / 12, sum(overshoots) / 12))
    print(f"\n## 越过设定点（{phase} 相位前 1 s；反号样本占比与越界量 p95）")
    for presence, label in ((True, "拖曳"), (False, "无负载")):
        got = grouped[presence]
        if not got:
            continue
        print(f"- {label}：n={len(got):3d} 越过占比中位 **{statistics.median([g[0] for g in got]) * 100:.1f}%**、"
              f"越界量 p95 中位 {statistics.median([g[1] for g in got]):.3f} rad")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="必要性测试台 run 的离线统计（标准库，无 Isaac）")
    ap.add_argument("run", nargs="?", type=Path, help="run 目录（缺省配合 --latest）")
    ap.add_argument("--latest", action="store_true", help="用 logs/towing/play_test 下最新的 run")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="run 根目录")
    ap.add_argument("--csv", type=Path, default=None, help="把逐 case 关键指标另存为 CSV")
    ap.add_argument("--list-codes", action="store_true", help="额外打印每条失败原因的组合")
    ap.add_argument("--windows", action="store_true",
                    help="起步/停车窗口按 是否拖车·坡度·速度 切分（设 JNT 的两个窗口；只读 summaries）")
    ap.add_argument("--crossings", type=int, default=0, metavar="N",
                    help="额外统计「越过设定点」占比/越界量（每侧最多 N 个 case，需 env*/tow.csv）")
    ap.add_argument("--crossings-phase", choices=("tow", "coast", "station"), default="tow",
                    help="--crossings 用哪个相位的前 1 s（默认 tow）")
    ap.add_argument("--segments", action="store_true",
                    help="额外按剖面分段统计 q−q*（读 env*/tow.csv，需要 run 里保留了逐 case CSV）")
    ap.add_argument("--segments-limit", type=int, default=40,
                    help="分段统计最多读多少个 case（0 = 全部；默认 40，抽稀后覆盖整张网格）")
    args = ap.parse_args(argv)

    run = args.run or (latest_run(args.root) if args.latest or args.run is None else None)
    if run is None:
        ap.error("需要 run 目录或 --latest")
    run = run.resolve()
    cases, experiment = load_cases(run)
    if not cases:
        raise SystemExit(f"{run} 下没有可读的 summaries/*.json")

    expected = int((experiment.get("arguments") or {}).get("num_envs")
                   or (experiment.get("scene") or {}).get("num_envs") or 0)
    records = [{"case": c["case"], **metric_of(c)} for c in cases]
    n = len(records)
    codes = Counter(r["code"] for r in records)
    reasons = Counter(reason for r in records for reason in r["reasons"])
    falls = sum(1 for r in records if r["fell"])

    print(f"# {run}")
    print(f"- 状态：{experiment.get('state')}；已判读 **{n}/{expected or '?'}** 个 case"
          f"{'（run 还没跑完）' if expected and n < expected else ''}")
    print(f"- 判定码：{code_summary(codes, n)}")
    print(f"- 跌倒（base 高 < 0.15 m 或 |pitch| > 0.8 rad 占比超限）：**{falls}** 个"
          + (f"；相位分布 {dict(Counter(r['fall_phase'] for r in records if r['fell']))}" if falls else ""))
    print(f"- 失败原因（可叠加）：" + "，".join(f"{k} {v}" for k, v in reasons.most_common()))
    unavailable = sum(1 for r in records if r["clearance_unavailable"])
    print(f"- 几何间隙不可用样本非零的 case：{unavailable} 个（>0 说明末端/坏姿态被采到，见记录 §二跑①）")

    print("\n## 关键指标分位数（全体）")
    for key, label in (("startup_rms", "起步关节 RMS (rad)"), ("startup_max", "起步单关节峰值 (rad)"),
                       ("speed_mae", "跟速 MAE (m/s)"), ("stop_min_clearance", "停车最小几何间隙 (m)"),
                       ("coast_distance", "小车滑移距离 (m)"), ("lateral_max", "横向 |y|max (m)"),
                       ("heading_max_deg", "朝向偏差 max (deg)"), ("min_base_z", "最低 base 高 (m)")):
        print(f"- {label:26s} {quantiles([r[key] for r in records])}")

    def cart_label(case):
        return "拖曳（有负载）" if case["case"].get("cart_present", True) else "无负载（_nocart）"

    dims = (
        ("按是否拖车（无负载对照）", cart_label),
        ("按坡度量级（lane 剖面档）", lambda c: f"{c['case']['grade_deg']:g}°"),
        ("按连接类型", lambda c: c["case"]["connection"]),
        ("按质量档 (kg)", lambda c: f"{c['case']['cart_mass_kg']:g}"),
        ("按速度档 (m/s)", lambda c: f"{c['case']['velocity_mps']:g}"),
        ("按行（row，连接长度/出生间隙）", lambda c: f"row{c['case']['row']:02d}"),
    )
    for title, key_fn in dims:
        groups = defaultdict(list)
        for record in records:
            groups[key_fn(record)].append(record)
        group_table(title, groups, expected)

    cross_table("是否拖车 × 判定码", cases, cart_label,
                lambda c: c["metrics"]["verdict"]["code"])
    cross_table("连接 × 判定码", cases, lambda c: c["case"]["connection"],
                lambda c: c["metrics"]["verdict"]["code"])
    cross_table("坡度量级 × 判定码", cases, lambda c: f"{c['case']['grade_deg']:g}°",
                lambda c: c["metrics"]["verdict"]["code"])
    cross_table("质量 × 判定码", cases, lambda c: f"{c['case']['cart_mass_kg']:g}kg",
                lambda c: c["metrics"]["verdict"]["code"])
    cross_table("速度 × 判定码", cases, lambda c: f"{c['case']['velocity_mps']:g}m/s",
                lambda c: c["metrics"]["verdict"]["code"])
    cross_table("COL（停车追尾）集中在哪：连接 × 是否 COL", cases, lambda c: c["case"]["connection"],
                lambda c: "COL" if c["metrics"]["verdict"]["code"] == "COL" else "其它")

    if args.windows:
        print_windows(cases)

    if args.crossings:
        case_dirs = sorted(path for path in run.glob("env*") if path.is_dir())
        if not case_dirs:
            print("\n（--crossings 需要 env*/tow.csv：用 `--write-csv all` 重跑，或挑一个有轨迹的 run）")
        else:
            print_crossings(case_dirs, sorted((run / "summaries").glob("*.json")),
                            limit=args.crossings, phase=args.crossings_phase)

    if args.segments:
        case_dirs = sorted(path for path in run.glob("env*") if path.is_dir())
        if not case_dirs:
            print("\n（本 run 里没有 env*/tow.csv —— 默认 `--write-csv failed` 只给非 OK 的 case 留轨迹；"
                  "想要全网格的分段统计请用 `--write-csv all` 重跑）")
        else:
            groups, picked, speed_matrix = segment_stats(
                case_dirs, sorted((run / "summaries").glob("*.json")),
                limit=(args.segments_limit or None))
            print_segments(groups, picked)
            print_segment_speed_matrix(speed_matrix)

    if args.list_codes:
        combos = Counter(tuple(r["reasons"]) for r in records if r["code"] != "OK")
        print("\n## 失败原因组合")
        for combo, count in combos.most_common():
            print(f"- {count:4d}  {'+'.join(combo)}")

    if args.csv:
        keys = ["env_index", "column", "row", "connection", "connection_length_m", "grade_deg",
                "velocity_mps", "cart_mass_kg", "code", "fell", "fall_phase", "min_base_z",
                "startup_rms", "startup_max", "speed_mae", "speed_ratio", "stop_min_clearance",
                "stop_contact", "coast_distance", "lateral_max", "heading_max_deg",
                "clearance_unavailable", "settle_tension"]
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            for record in records:
                row = dict(record["case"])
                row.update({k: v for k, v in record.items() if k != "case"})
                row["reasons"] = "+".join(record["reasons"])
                writer.writerow({k: row.get(k) for k in keys})
        print(f"\n[csv] 逐 case 关键指标已写入 {args.csv}")

    report = run / "report.md"
    if report.is_file():
        text = report.read_text(encoding="utf-8")
        section = text.split("## 结论（任务是否有必要）")
        if len(section) > 1:
            print("\n## report.md 的结论（原文）")
            print(section[1].split("## ")[0].strip()[:1200])
    else:
        print("\n（report.md 还没生成 —— run 未跑完，或判读阶段被中断）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
