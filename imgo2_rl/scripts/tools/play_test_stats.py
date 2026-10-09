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
import os
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="必要性测试台 run 的离线统计（标准库，无 Isaac）")
    ap.add_argument("run", nargs="?", type=Path, help="run 目录（缺省配合 --latest）")
    ap.add_argument("--latest", action="store_true", help="用 logs/towing/play_test 下最新的 run")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="run 根目录")
    ap.add_argument("--csv", type=Path, default=None, help="把逐 case 关键指标另存为 CSV")
    ap.add_argument("--list-codes", action="store_true", help="额外打印每条失败原因的组合")
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

    dims = (
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
