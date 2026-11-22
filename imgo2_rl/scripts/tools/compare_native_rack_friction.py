#!/usr/bin/env python3
"""Compare isolated shank response with explicit and solver joint friction."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics as st

from audit_joint_sine_log import phase_difference
from evaluate_resistance_actuator import reversals

STEPS = ("0.005", "0.0025", "0.0005")
VARIANTS = ("baseline", "previous_candidate", "native_060", "native_080", "native_100", "native_120")


def read(path):
    return json.loads(path.read_text())


def analyze(root):
    runs = {}
    for dt in STEPS:
        folder = root / f"rack_{dt}"
        meta = read(folder / "metadata.json")
        if not meta["completed"] or not meta["production_files_unchanged"]:
            raise ValueError(f"Incomplete trial: {folder}")
        plan = read(folder / "plan_used.json")
        windows = {case["id"]:case["window_s"] for case in plan["sources"]}
        jobs = read(folder / "scores.json")["jobs"]
        with (folder/"traces.csv").open() as f:
            traces = list(csv.DictReader(f))
        for j in jobs:
            lo, hi = windows[j["source_id"]]
            tag = f"dq_{j['env']:03}"
            vel = [float(r[tag]) for r in traces if lo <= float(r["test_s"]) < hi]
            j["velocity_rms_rad_s"] = math.sqrt(st.fmean(x*x for x in vel))
            j["velocity_reversals"] = reversals(vel)
            j["phase_reliable"] = j["q_fit"]["amplitude"] > 1e-5 and (j["q_fit"]["r_squared"] or 0) >= .5
        runs[dt] = {(j["candidate_id"], j["source_id"]):j for j in jobs}
    result = []
    for variant in VARIANTS:
        c = [j for (tag,_),j in runs[STEPS[0]].items() if tag==variant]
        pairs = []
        for j in c:
            fine = runs[STEPS[-1]][variant,j["source_id"]]
            amp = 100*abs(j["gain"]-fine["gain"])
            phase = abs(phase_difference(j["phase_deg"], fine["phase_deg"])) if j["phase_reliable"] and fine["phase_reliable"] else None
            extra = j["velocity_reversals"]-fine["velocity_reversals"]
            stuck = max(j["q_fit"]["amplitude"], fine["q_fit"]["amplitude"]) <= 1e-5
            pairs.append(dict(source_id=j["source_id"], amplitude_pp=amp, phase_deg=phase,
                both_stuck=stuck, extra_reversals=extra,
                passed=amp<=2 and (stuck or phase is not None and phase<=3) and extra<=2))
        result.append(dict(variant=variant, step_checks=pairs,
            step_passed_all=all(p["passed"] for p in pairs),
            mean_small_dynamic_rmse=st.fmean(j["ac_rmse_rad"] for j in c if j["amplitude_rad"]==.01),
            mean_large_dynamic_rmse=st.fmean(j["ac_rmse_rad"] for j in c if j["amplitude_rad"]==.05)))
    return dict(runs={k:list(v.values()) for k,v in runs.items()}, variants=result,
                step_comparison_ms=[5,.5], physics_steps_ms=[5,2.5,.5],
                step_limits=dict(amplitude_pp=2, phase_deg=3, extra_reversals=2),
                interpretation="Both-stuck pairs can be numerically repeatable. They still fail waveform matching if the real joint moves.")


def plot(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ("DCMotor", "tanh resistance", "PhysX ×.6", "PhysX ×.8", "PhysX ×1.0", "PhysX ×1.2")
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), layout="constrained")
    for c, case in enumerate((106,107,2,3)):
        for step, marker in zip(STEPS,("o","s","^")):
            rows = {j["candidate_id"]:j for j in result["runs"][step] if j["source_id"]==case}
            ref=rows["baseline"]
            axes[0,c].plot(range(len(VARIANTS)),[100*rows[v]["gain"] for v in VARIANTS], marker, label=f"{1000*float(step):g}ms")
            axes[1,c].plot(range(len(VARIANTS)),
                [rows[v]["phase_deg"] if rows[v]["phase_reliable"] else math.nan for v in VARIANTS], marker)
        axes[0,c].axhline(100*ref["real_gain"], color="black", ls="--", label="Measured")
        axes[1,c].axhline(ref["real_phase_deg"], color="black", ls="--")
        axes[0,c].set(title=f"{ref['frequency_hz']:g}Hz / ±{ref['amplitude_rad']:g} rad")
        for ax in axes[:,c]:
            ax.set_xticks(range(len(VARIANTS)), labels, rotation=30, ha="right")
            ax.grid(alpha=.2)
    axes[0,0].legend()
    axes[0,0].set_ylabel("Amplitude / target (%)")
    axes[1,0].set_ylabel("Phase relative to target (deg)")
    fig.suptitle("Left-front shank: same recorded targets; phase omitted for negligible or poorly fitted motion")
    fig.savefig(path,dpi=145)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    result = analyze(args.batch)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    if args.plot:
        plot(result, args.plot)
