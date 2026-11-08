#!/usr/bin/env python3
"""Analyze force-based contact pairing in saved deployment MuJoCo CSVs.

The core uses the standard library. --plot additionally needs matplotlib.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st

from analyze_mujoco_policies import contact_flags, runs


LEGS = ("FL", "FR", "RL", "RR")


def load_rows(path, skip):
    with path.open() as handle:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(handle)
                if float(row["t_s"]) >= skip]


def correlation(a, b):
    ma, mb = st.mean(a), st.mean(b)
    variance = ma * (1 - ma) * mb * (1 - mb)
    if variance <= 1e-12:
        return None
    return (st.mean(x * y for x, y in zip(a, b)) - ma * mb) / math.sqrt(variance)


def air_overlap(a, b):
    union = sum(not x or not y for x, y in zip(a, b))
    return sum(not x and not y for x, y in zip(a, b)) / union if union else None


def load_balance(rows, flags, dt):
    """Normal-force loading, not just contact timing; no symmetry target assumed."""
    feet = {}
    for leg in LEGS:
        forces = [r[leg + "_contact_n"] for r in rows]
        segments = runs(flags[leg])
        # Exclude window-truncated stances; compare typical whole-stance impacts.
        stances = [(a, b) for a, b, contact in segments[1:-1] if contact]
        peaks = [max(forces[a:b]) for a, b in stances]
        impulses = [sum(forces[a:b]) * dt for a, b in stances]
        feet[leg] = {
            "normal_force_time_mean_n": st.mean(forces),
            "stance_fraction": st.mean(flags[leg]),
            "stance_duration_median_s": st.median((b-a)*dt for a, b in stances) if stances else None,
            "stance_peak_median_n": st.median(peaks) if peaks else None,
            "stance_impulse_median_ns": st.median(impulses) if impulses else None,
            "complete_stances": len(stances),
        }
    groups = {}
    for name, left, right in (
        ("left_vs_right", ("FL", "RL"), ("FR", "RR")),
        ("diagonal_FL_RR_vs_FR_RL", ("FL", "RR"), ("FR", "RL")),
        ("front_vs_rear", ("FL", "FR"), ("RL", "RR")),
    ):
        a = sum(feet[leg]["normal_force_time_mean_n"] for leg in left)
        b = sum(feet[leg]["normal_force_time_mean_n"] for leg in right)
        groups[name] = {"first_mean_n": a, "second_mean_n": b,
                        "first_share": a/(a+b) if a+b else None,
                        "signed_difference_over_sum": (a-b)/(a+b) if a+b else None}
    return {"per_foot": feet, "groups": groups,
            "definition": "Time-mean positive contact normal force includes swing zeros; peaks/impulses use complete hysteresis-filtered stances. On this flat floor normal is vertical. Front/rear equality is not assumed."}


def analyze(rows):
    dt = st.median(rows[i]["t_s"] - rows[i - 1]["t_s"] for i in range(1, len(rows)))
    flags = {leg: contact_flags([r[leg + "_contact_n"] for r in rows], dt) for leg in LEGS}
    patterns = list(zip(*(flags[leg] for leg in LEGS)))
    pairs = lambda pairs: [correlation(flags[a], flags[b]) for a, b in pairs]
    return {
        "samples": len(rows), "sample_dt": dt,
        "load_balance": load_balance(rows, flags, dt),
        "diagonal_contact_correlation": pairs((("FL", "RR"), ("FR", "RL"))),
        "same_side_contact_correlation": pairs((("FL", "RL"), ("FR", "RR"))),
        "front_and_rear_contact_correlation": pairs((("FL", "FR"), ("RL", "RR"))),
        "diagonal_swing_jaccard": [air_overlap(flags[a], flags[b])
                                   for a, b in (("FL", "RR"), ("FR", "RL"))],
        "all_four_contact_fraction": st.mean(all(p) for p in patterns),
        "three_contact_fraction": st.mean(sum(p) == 3 for p in patterns),
        "diagonal_two_contact_fraction": st.mean(
            p in ((True, False, False, True), (False, True, True, False)) for p in patterns),
        "lateral_two_contact_fraction": st.mean(
            p in ((True, False, True, False), (False, True, False, True)) for p in patterns),
        "front_rear_two_contact_fraction": st.mean(
            p in ((True, True, False, False), (False, False, True, True)) for p in patterns),
        "per_foot_clearance_p95_m": {
            leg: sorted(r[leg + "_clearance_m"] for r in rows)[round(.95 * (len(rows) - 1))]
            for leg in LEGS},
        "all_feet_mean_slip_mps": st.mean(r[leg + "_slip_mps"] for r in rows for leg in LEGS),
    }


def plot(run, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    policies = list(dict.fromkeys(case["policy"] for case in run["cases"]))
    lookup = {(case["policy"], case["case"]): case for case in run["cases"]}
    fig, axes = plt.subplots(len(policies), 4, figsize=(16, 3 * len(policies)), squeeze=False)
    colors = ("#267cb5", "#e58c28", "#3c9b58", "#bf5d8a")
    for row, policy in enumerate(policies):
        for offset, case in enumerate(("forward03", "backward03")):
            rows = load_rows(Path(lookup[policy, case]["csv"]), 4)
            dt = st.median(rows[i]["t_s"] - rows[i - 1]["t_s"] for i in range(1, len(rows)))
            flags = {leg: contact_flags([r[leg + "_contact_n"] for r in rows], dt) for leg in LEGS}
            mask = [i for i, r in enumerate(rows) if 8 <= r["t_s"] <= 10]
            t = [rows[i]["t_s"] for i in mask]
            contacts, heights = axes[row, 2 * offset:2 * offset + 2]
            for index, leg in enumerate(LEGS):
                contacts.fill_between(t, index - .32, index + .32,
                                      where=[flags[leg][i] for i in mask],
                                      step="mid", color=colors[index], linewidth=0)
                heights.plot(t, [100 * rows[i][leg + "_clearance_m"] for i in mask],
                             color=colors[index], label=leg, linewidth=1.2)
            contacts.set_yticks(range(4), LEGS)
            contacts.set_ylim(3.6, -.6)
            contacts.set_ylabel(policy, fontsize=9)
            heights.set_ylim(-.5, 11)
            heights.set_ylabel("Foot-bottom clearance (cm)")
            heights.axhline(0, color="grey", linewidth=.6)
            direction = "Forward +0.3 m/s" if offset == 0 else "Backward -0.3 m/s"
            if row == 0:
                contacts.set_title(direction + " | contact bands")
                heights.set_title(direction + " | clearance")
            for ax in (contacts, heights):
                ax.set_xlim(8, 10)
                ax.set_xlabel("Simulation time after command (s)")
                ax.grid(axis="x", alpha=.15)
    axes[0, 1].legend(ncol=4, fontsize=8, loc="upper right")
    fig.suptitle("Identical MuJoCo conditions | colored band = contact, blank = air\n"
                 "Same model, PD gains and feedback timing; each case reset independently",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    for extension in ("png", "pdf"):
        fig.savefig(str(output) + "." + extension, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Benchmark run.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip", type=float, default=4.)
    parser.add_argument("--plot", type=Path, help="Optional output prefix for PNG/PDF")
    args = parser.parse_args()
    run = json.loads(args.run.read_text())
    result = {"scope": "Force contacts: 5N/2N hysteresis, 10ms persistence, <20ms chatter removal; "
                       "contact correlation and support fractions alone do not define a gait",
              "skip_seconds": args.skip, "results": [],
              "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for case in run["cases"]:
        path = Path(case["csv"])
        record = analyze(load_rows(path, args.skip))
        record.update(policy=case["policy"], case=case["case"], csv=path.name)
        result["results"].append(record)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.plot:
        plot(run, args.plot)
    print(f"Analyzed {len(result['results'])} cases -> {args.output}")


if __name__ == "__main__":
    main()
