#!/usr/bin/env python3
"""Standard-library swing-height and unit FHB-cost analysis on benchmark CSVs.

Updates only diagnostic JSON. Unit FHB scores use reconstructed foot velocity and
1 N normal-force contact approximations, not Isaac's original sensor samples.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics as st

from analyze_mujoco_policies import contact_flags, runs
from audit_ppo_rough_reward import evaluate

LEGS = ("FL", "FR", "RL", "RR")


def swing_profile(rows):
    dt = st.median(rows[i]["t_s"] - rows[i - 1]["t_s"] for i in range(1, len(rows)))
    profiles = []
    for leg in LEGS:
        flags = contact_flags([r[leg + "_contact_n"] for r in rows], dt)
        for start, end, contact in runs(flags)[1:-1]:
            if contact or (end - start) * dt < .03 - 1e-9:
                continue
            values = [r[leg + "_clearance_m"] for r in rows[start:end]]
            profile = []
            for k in range(51):
                q = (len(values) - 1) * k / 50
                lo, hi = int(q), min(int(q) + 1, len(values) - 1)
                profile.append(values[lo] if hi == lo else
                               values[lo] * (hi - q) + values[hi] * (q - lo))
            profiles.append(profile)
    return {
        "air_run_count": len(profiles), "phase": [k / 50 for k in range(51)],
        "mean_clearance_m": [st.mean(p[k] for p in profiles) for k in range(51)] if profiles else [],
        "std_clearance_m": [st.pstdev(p[k] for p in profiles) for k in range(51)] if profiles else [],
        "scope": "Four feet pooled, complete air runs >=30ms; dispersion includes leg differences. Phase uses future landing offline and is not an online policy variable.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", default="forward03,forward05,backward03,turn_left03,turn_right03")
    args = parser.parse_args()
    main_metrics = json.loads((args.run / "metrics.json").read_text())
    report = json.loads(args.output.read_text()) if args.output.exists() else {"main": main_metrics}
    report.update(swing_profiles={}, fhb_unit_cost=[], power_proxy={})
    policies = sorted({r["policy"] for r in main_metrics["results"]})
    for policy in policies:
        for case in args.cases.split(","):
            key = Path(policy).stem + "_" + case
            path = args.run / (key + ".csv")
            with path.open() as fh:
                rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]
            rows = [r for r in rows if r["t_s"] >= 4.0]
            if len(rows) < 100:
                raise ValueError(f"Insufficient samples: {path}")
            report["swing_profiles"][key] = swing_profile(rows)
            full = -evaluate(path, {"feet_height_body": -1.})["feet_height_body"]
            swing = -evaluate(path, {"feet_height_body": -1.}, fhb_swing_only=True)["feet_height_body"]
            report["fhb_unit_cost"].append({
                "policy": policy, "case": case, "all_feet_cost_per_s": full,
                "swing_only_cost_per_s": swing, "stance_share": (full - swing) / full if full else None,
            })
            joints = [f"{leg}_{joint}_joint" for leg in LEGS for joint in ("hip", "thigh", "shank")]
            power = st.mean(sum(abs(r["tau_" + j] * r["dq_" + j]) for j in joints) for r in rows)
            report["power_proxy"][key] = {
                "mean_sum_abs_joint_mechanical_power_W": power, "not_electrical_power": True,
            }
    report.setdefault("diagnostic_tool_sha256", {})[str(Path(__file__))] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Analyzed {len(report['swing_profiles'])} policy/cases; output={args.output}")


if __name__ == "__main__":
    main()
