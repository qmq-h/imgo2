#!/usr/bin/env python3
"""Summarize the measured 2026-10-14 AMP engine/material ablations (stdlib).

Raw measurements are under logs/engine_audit_2026-10-14; this does not launch
simulators or change a policy. Failed reset-heavy trials are explicitly invalid.
"""
import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import summarize_standing_response as standing

REPO = Path(__file__).resolve().parents[3]


def autocorrelation():
    source = REPO / "imgo2_deploy/scripts/eval_gazebo_policy.py"
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "autocorr_peak")
    namespace = {"math": math}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[node.name]


def finite(value):
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, dict): return {k: finite(v) for k, v in value.items()}
    if isinstance(value, list): return [finite(v) for v in value]
    return value


def summarize(root):
    period = autocorrelation()
    stages = []
    stems = ("isaac_diagnostics", "isaac_restitution0", "isaac_clean", "isaac_friction06", "isaac_pd1000")
    for stem in stems:
        summary = json.loads((root / (stem + "_summary.json")).read_text())
        raw = json.loads((root / (stem + "_raw.json")).read_text())
        stage = {"stage": stem, "policy_sha256": summary["policy_sha256"],
                 "physics": summary["physics"], "control_dt": summary["control_dt"], "cases": []}
        # The first diagnostic run used these old labels for PhysX passive
        # joint drives. Explicit DCMotor gains are separate (25 / 0.5).
        for gain in ("stiffness", "damping"):
            old = "actual_joint_" + gain
            if old in stage["physics"]:
                stage["physics"]["physx_joint_" + gain] = stage["physics"].pop(old)
        stage["physics"].setdefault("actor_observation_noise_enabled", stem in ("isaac_diagnostics", "isaac_restitution0"))
        for case in summary["cases"]:
            row = raw[case["case"]]
            metrics = standing.metrics(row["times"], row["joint_targets"], row["joint_positions"])
            metrics["actual_joint_velocity_rms_rad_s"] = math.sqrt(statistics.mean(
                v*v for r in row["joint_velocities"] for v in r))
            cycles = {}
            for leg in ("FL", "FR", "RL", "RR"):
                index = row["joint_names"].index(leg + "_thigh_joint")
                q = [r[index] for r in row["joint_positions"]]
                avg = statistics.mean(q)
                interval, score, hz = period([v-avg for v in q], summary["control_dt"])
                cycles[leg] = {"period_s": interval, "score": score, "hz": hz}
            vertical = [sum(v[2] for v in r) for r in row["contact_forces_w"]]
            stage["cases"].append({**case, "steady_window_valid": case["episode_resets"] == 0,
                "joint_metrics": metrics, "thigh_autocorrelation": cycles,
                "mean_net_contact_fz_N": statistics.mean(vertical)})
        stages.append(stage)
    gazebo = json.loads((root / "gazebo_diagnostics_summary.json").read_text())
    feet = json.loads((root / "gazebo_diagnostics_feet.json").read_text())
    for case in gazebo:
        name = case["case"]
        folder = root / "gazebo_diagnostics" / name
        transient = json.loads((folder / "transient.json").read_text())
        with (folder / "policy_trace.csv").open() as f:
            trace = [r for r in csv.DictReader(f) if float(r["wall_s"]) >= transient["policy_entry_wall"] + 4]
        times = [float(r["wall_s"]) for r in trace]
        targets = [[float(r["target_"+str(i)]) for i in range(12)] for r in trace]
        q = [[float(r["obs_"+str(9+i)]) + [0, .87, -1.82][i % 3] for i in range(12)] for r in trace]
        case["joint_metrics"] = standing.metrics(times, targets, q)
        case["joint_metrics"]["actor_dq_rms_rad_s"] = math.sqrt(statistics.mean(
            (float(r["obs_"+str(21+i)]) / .05)**2 for r in trace for i in range(12)))
        stable = json.loads((folder / "stable.json").read_text())
        case["joint_metrics"]["true_joint_velocity_rms_rad_s"] = math.sqrt(statistics.mean(
            pair[1]**2 for _, values in stable["joint_states"] for pair in values.values()))
        case["per_foot_geometry"] = next(v["per_foot"] for v in feet if Path(v["raw_file"]).parent.name == name)
    files = [root / (s + suffix) for s in stems for suffix in ("_summary.json", "_raw.json")]
    files += [root / name for name in ("gazebo_diagnostics_summary.json", "all_contracts.json", "model_physics.json", "contact_force_summary.json")]
    return finite({"date": "2026-10-14", "raw_archive": str(root.resolve()),
        "native_stages": stages, "gazebo": gazebo,
        "gazebo_contacts": json.loads((root / "contact_force_summary.json").read_text()),
        "conversion_contracts": json.loads((root / "all_contracts.json").read_text()),
        "model_physics": json.loads((root / "model_physics.json").read_text()),
        "measurement_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "limits": "One policy, seed42, one environment per native command and fresh Gazebo world per command. "
            "Noiseless/material ablations are evaluations, not retraining. Reset-heavy 1ms trial is invalid for gait comparison. "
            "Target jumps are not actual joint displacements; force and velocity sampling differ. Weak autocorrelation is not a reliable gait period. "
            "Gazebo contact CSVs used here contain robot-side link-frame wrenches, rotated using FK; static-ground wrench attempt was discarded."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=REPO / "imgo2_rl/logs/engine_audit_2026-10-14")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    output = summarize(args.archive)
    args.json.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print("Saved engine comparison:", args.json)


if __name__ == "__main__":
    main()
