#!/usr/bin/env python3
"""Check the discrete work of the held resistance torque in a saved rack run."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics as st

from analyze_leg_rack_batch import build

CASE = 107


def load_trial(batch, step, variant):
    folder = batch / f"rack_{step}"
    plan = json.loads((folder / "plan_used.json").read_text())
    source = next(s for s in plan["sources"] if s["id"] == CASE)
    scores = json.loads((folder / "scores.json").read_text())["jobs"]
    score = next(r for r in scores if (r["source_id"], r["candidate_id"]) == (CASE, variant))
    suffix = f"{score['env']:03}"
    with (folder / "traces.csv").open() as stream:
        trace = [[float(row[name]) for name in ("test_s", f"q_{suffix}", f"dq_{suffix}",
                                               f"pd_{suffix}", f"tau_{suffix}")]
                 for row in csv.DictReader(stream)]
    return source, score, trace


def audit(batch):
    source, score, coarse = load_trial(batch, "0.005", "previous_candidate")
    lo, hi = source["window_s"]
    steps = []
    for pre, after in zip(coarse, coarse[1:]):
        t = pre[0]
        if not lo - 1e-10 <= t < hi - 1e-10:
            continue
        if not math.isclose(after[0] - t, .005, abs_tol=1e-10):
            raise ValueError("Expected adjacent physics steps at the saved 5ms recording interval")
        # The recorded PD effort is inside the DC motor envelope in this case.
        max_v = abs(pre[2])
        effort_bound = 23.7 * max(0, 1 - max_v / 30.1)
        if abs(after[3]) >= effort_bound:
            raise ValueError("Cannot reconstruct resistance as net minus PD with motor clipping")
        applied_resistance = after[4] - after[3]
        expected = -.275 * math.tanh(pre[2] / .010)
        if abs(applied_resistance - expected) > 2e-6:
            raise ValueError("Reconstructed resistance disagrees with the shared actuator formula")
        steps.append((t, pre[2], after[2], applied_resistance,
                      applied_resistance * (after[1] - pre[1])))
    if len(steps) != round((hi-lo)/.005):
        raise ValueError("Incomplete scoring window")
    result = dict(source=source["source"], source_sha256=source["sha256"], window_s=[lo,hi],
                  physics_dt_s=.005, steps=len(steps),
                  positive_resistance_work_steps=sum(s[4]>1e-12 for s in steps),
                  negative_resistance_work_steps=sum(s[4]<-1e-12 for s in steps),
                  sign_reversals=sum(s[1]*s[2]<0 for s in steps),
                  summed_resistance_work_j=sum(s[4] for s in steps),
                  simulated_dq_rms_rad_s=math.sqrt(st.fmean(s[2]**2 for s in steps)),
                  max_abs_resistance_nm=max(abs(s[3]) for s in steps),
                  initial_power_max_w=max(s[1]*s[3] for s in steps),
                  max_reconstruction_error_nm=max(abs(s[3]+.275*math.tanh(s[1]/.010)) for s in steps),
                  score={k:score[k] for k in ("gain","phase_deg","real_gain","real_phase_deg")})
    return result, steps


def plot(batch, output, steps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source, _, _ = load_trial(batch, "0.005", "previous_candidate")
    _, data = build(Path(source["source"]), CASE, None, None)
    real = data["real"]
    fig, (wave, vel) = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    for times, values, title, color, ls in (
        (real["ref_t"], real["ref"], "Recorded target", "#5d5d5d", "--"),
        (real["t"], real["q"], "Measured joint", "#222222", "-"),
    ):
        mean = st.fmean(values)
        points = [(t,q-mean) for t,q in zip(times,values) if 2<=t<3.5]
        x,y = zip(*points)
        wave.plot(x,y,label=title,color=color,ls=ls)
    for step, label, color in (("0.005","Candidate at 5 ms","#d77736"),
                               ("0.00025","Candidate at 0.25 ms","#287b93")):
        _, score, trace = load_trial(batch, step, "previous_candidate")
        points = [(r[0],r[1]-score["q_fit"]["mean"]) for r in trace if 2<=r[0]<3.5]
        x,y = zip(*points)
        wave.plot(x,y,label=label,color=color,lw=1.2)
    wave.set(title="Left-front shank, same 2 Hz / ±0.01 rad target",ylabel="q minus own mean (rad)",xlabel="Time (s)")
    wave.legend(ncol=2)
    cut = [s for s in steps if 2-1e-10<=s[0]<=2.07+1e-10]
    t = [s[0] for s in cut]
    vel.plot(t,[s[1] for s in cut],"o-",lw=1,color="#287b93",label="Joint speed at step start")
    force = vel.twinx()
    force.step(t,[s[3] for s in cut],where="post",lw=1.5,color="#d77736",label="Held resistance torque")
    vel.axhline(0,color="gray",lw=.7)
    vel.set(title="5 ms zoom: torque is held while velocity changes sign",xlabel="Time (s)",ylabel="Speed (rad/s)")
    force.set_ylabel("Applied resistance torque (Nm)")
    vel.legend(loc="upper left")
    force.legend(loc="upper right")
    for axis in (vel,wave):
        axis.grid(alpha=.2)
    fig.savefig(output,dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    summary, steps = audit(args.batch)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    if args.plot:
        plot(args.batch, args.plot, steps)
    print(json.dumps(summary, indent=2))
