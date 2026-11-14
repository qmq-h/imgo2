#!/usr/bin/env python3
"""Plot robot/load speed curves from a mass-step run's per-case `tow.csv`.

Requires play_towing_test.py to be run with `--write-csv all` (not `none`):
each case directory then contains `tow.csv`.  The script aggregates median and
10/90 percentiles over cases at each recorded time and marks the mass change.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def percentile(values, fraction):
    if not values:
        return float("nan")
    values = sorted(values)
    index = min(len(values) - 1, max(0, int(round(fraction * (len(values) - 1)))))
    return values[index]


def load_cases(run_dir: Path):
    cases = []
    for csv_path in sorted(run_dir.glob("*/tow.csv")):
        with csv_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if rows:
            cases.append(rows)
    if not cases:
        raise SystemExit(f"{run_dir} 下没有 */tow.csv；请用 --write-csv all 重跑")
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or (args.run_dir / "mass_step_speed.png")
    cases = load_cases(args.run_dir)
    by_t = defaultdict(lambda: {"robot": [], "load": [], "cmd": [], "mass": []})
    for rows in cases:
        for row in rows:
            t = round(float(row["time_s"]), 3)
            by_t[t]["robot"].append(float(row["robot_vx_b_mps"]))
            by_t[t]["load"].append(float(row["load_vx_mps"]))
            by_t[t]["cmd"].append(float(row["loco_cmd_vx_mps"]))
            by_t[t]["mass"].append(float(row["cart_mass_kg"]))
    times = sorted(by_t)
    robot = [statistics.median(by_t[t]["robot"]) for t in times]
    load = [statistics.median(by_t[t]["load"]) for t in times]
    cmd = [statistics.median(by_t[t]["cmd"]) for t in times]
    mass = [statistics.median(by_t[t]["mass"]) for t in times]
    r10 = [percentile(by_t[t]["robot"], 0.1) for t in times]
    r90 = [percentile(by_t[t]["robot"], 0.9) for t in times]
    # mass step: first time mass increases by > 5 kg
    step_t = None
    initial = mass[0] if mass else 0.0
    for t, m in zip(times, mass):
        if m > initial + 5.0:
            step_t = t
            break
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.fill_between(times, r10, r90, color="#1f77b4", alpha=0.18, label="robot vx 10-90%")
    ax.plot(times, robot, color="#1f77b4", linewidth=1.8, label="robot vx median")
    ax.plot(times, load, color="#ff7f0e", linewidth=1.4, label="load vx median")
    ax.plot(times, cmd, color="#2ca02c", linestyle="--", linewidth=1.2, label="loco cmd vx")
    if step_t is not None:
        ax.axvline(step_t, color="#d62728", linestyle=":", linewidth=1.5,
                   label=f"mass step @ {step_t:g}s")
        ax2.axvline(step_t, color="#d62728", linestyle=":", linewidth=1.5)
    ax.set_ylabel("vx [m/s]")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"{args.run_dir.name} | {len(cases)} cases")
    ax2.plot(times, mass, color="#8c564b", linewidth=1.6)
    ax2.set_ylabel("cart mass [kg]")
    ax2.set_xlabel("time [s]")
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    print(f"wrote {output}")
    if step_t is not None:
        print(f"mass step detected at t={step_t:g}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
