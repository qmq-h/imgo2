#!/usr/bin/env python3
"""Rotate Gazebo FOOT contact forces from the lumped SHANK frame into world.

Requires the flat-world contact CSV from gazebo_contact_trace and the same
run's stable.json. Only FOOT-vs-ground-plane forces are reconstructed.
"""
import argparse
import bisect
import csv
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

import summarize_foot_clearance as sole

REPO = Path(__file__).resolve().parents[3]


def summarize(folder, model_path):
    model = ET.parse(model_path).getroot()
    chains = {leg: sole.fk.read_chain(model, leg + "_SHANK") for leg in sole.LEGS}
    stable = json.loads((folder / "stable.json").read_text())
    joints, odom = stable["joint_states"], stable["odom"]
    jts, ots = [r[0] for r in joints], [r[0] for r in odom]
    records = []
    with (folder / "contacts.csv").open() as source:
        for row in csv.DictReader(source):
            t = float(row["sim_s"])
            if not max(jts[0], ots[0]) <= t <= min(jts[-1], ots[-1]):
                continue
            ji = min(bisect.bisect_left(jts, t), len(joints) - 1)
            oi = min(bisect.bisect_left(ots, t), len(odom) - 1)
            angles, pose = joints[ji][1], odom[oi]
            base = sole.rpy_rotation([pose[4], pose[5], pose[3]])
            forces = {}
            for leg in sole.LEGS:
                q = [angles[leg + "_" + joint + "_joint"][0] for joint in ("hip", "thigh", "shank")]
                _, rot = sole.pose(chains[leg], q)
                forces[leg] = sole.fk.rotate(sole.fk.matmul(base, rot),
                    [float(row[leg + "_" + axis]) for axis in ("fx", "fy", "fz")])
            records.append({"t": t, "forces": forces,
                "sum": [sum(v[i] for v in forces.values()) for i in range(3)],
                "other_points": int(row["other_points"]), "self_pairs": int(row["self_pairs"]),
                "depth": float(row["max_depth_m"])})
    if len(records) < 2:
        raise ValueError("No stable contact samples: " + str(folder))
    result = {"case": folder.name, "samples": len(records),
        "mean_foot_force_world_N": [statistics.mean(r["sum"][i] for r in records) for i in range(3)],
        "p95_foot_force_z_N": sole.quantile([r["sum"][2] for r in records], .95),
        "max_penetration_m": max(r["depth"] for r in records),
        "nonfoot_contact_sample_fraction": statistics.mean(r["other_points"] > 0 for r in records),
        "self_contact_sample_fraction": statistics.mean(r["self_pairs"] > 0 for r in records), "per_foot": {}}
    for leg in sole.LEGS:
        contacts = [math.sqrt(sum(v*v for v in r["forces"][leg])) > 1 for r in records]
        start = None
        events, airs = [], []
        for r, contact in zip(records, contacts):
            if not contact and start is None:
                start = r["t"]
            elif contact and start is not None:
                air = r["t"] - start
                if air >= .0199:
                    events.append(r["t"])
                    airs.append(air)
                start = None
        periods = [b-a for a, b in zip(events, events[1:])]
        result["per_foot"][leg] = {"mean_fz_N": statistics.mean(r["forces"][leg][2] for r in records),
            "contact_fraction_force_over_1N": statistics.mean(contacts),
            "qualified_landings": len(events), "mean_air_time_s": statistics.mean(airs) if airs else None,
            "landing_median_period_s": statistics.median(periods) if periods else None}
    result["limits"] = ("ODE wrenches are in link frames. FK uses nearest received base/joint samples; "
        "FOOT collisions must be lumped into SHANK, as verified in this run's SDF. "
        "Approximately 100Hz force samples; 1N contact and 20ms air gates. Summed foot force is not full "
        "robot net force when nonfoot contacts exist; peaks depend on sampling.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--urdf", type=Path, default=REPO / "imgo2_description/urdf/imgo2_real.urdf")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = [summarize(p, args.urdf) for p in args.folders]
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
