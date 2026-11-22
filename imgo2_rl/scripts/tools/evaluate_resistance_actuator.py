#!/usr/bin/env python3
"""Score Isaac actuator regression, step sensitivity, and walking A/B trials."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics as st

from analyze_new_motion_batch import interp, load, rms, spectrum
from audit_joint_sine_log import phase_difference
from replay_walking_isaac import DOCS, JOINTS, ROOT, axes, fingerprint

STEPS = (".005", ".0025", ".001", ".0005", ".00025")
REFERENCE = ROOT / "imgo2_deploy/logs/new_motion_20261121_validation/isaac/scores.json"


def read(path):
    return json.loads(path.read_text())


def save(path, values):
    path.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")


def score_index(rows):
    return {(r["source_id"], r["candidate_id"]): r for r in rows}


def reversals(values, threshold=.003):
    """Count changes between opposite signs outside a hysteresis band in rad/s."""
    sign = count = 0
    for value in values:
        if abs(value) < threshold:
            continue
        next_sign = 1 if value > 0 else -1
        count += int(sign != 0 and sign != next_sign)
        sign = next_sign
    return count


def score_rack(batch):
    series = {}
    for step in STEPS:
        directory = batch / ("rack_0" + step)
        metadata = read(directory / "metadata.json")
        if metadata["zero_resistance_effort_error_nm"] != 0 or metadata["min_resistance_dissipation_w"] < 0:
            raise ValueError(f"Actuator contract failed at {step}")
        plan = read(directory / "plan_used.json")
        windows = {s["id"]:s["window_s"] for s in plan["sources"]}
        rows = load(directory / "traces.csv")
        scores = read(directory / "scores.json")["jobs"]
        series[step] = score_index(scores)
        for score in scores:
            lo, hi = windows[score["source_id"]]
            column = f"dq_{score['env']:03}"
            score["velocity_reversals"] = reversals(r[column] for r in rows if lo <= r["test_s"] < hi)
    new = series[".0005"]
    reference = score_index(read(REFERENCE)["jobs"])
    regression = []
    for key, old in reference.items():
        row = new[key]
        regression.append(dict(source_id=key[0], candidate_id=key[1],
            amplitude_difference_pp=100*abs(row["gain"]-old["gain"]),
            phase_difference_deg=abs(phase_difference(row["phase_deg"], old["phase_deg"])),
            old_ac_rmse_rad=old["ac_rmse_rad"], new_ac_rmse_rad=row["ac_rmse_rad"]))
    max_gain = max(r["amplitude_difference_pp"] for r in regression)
    max_phase = max(r["phase_difference_deg"] for r in regression)
    comparisons = []
    fine = series[".00025"]
    selected = None
    for step in STEPS[:-1]:
        details = []
        for key, row in series[step].items():
            ref = fine[key]
            gain = 100*abs(row["gain"]-ref["gain"])
            phase = abs(phase_difference(row["phase_deg"],ref["phase_deg"]))
            extra = row["velocity_reversals"]-ref["velocity_reversals"]
            details.append(dict(**row, amplitude_difference_pp=gain, phase_difference_deg=phase,
                                extra_reversals=extra, passed=gain<=2 and phase<=3 and extra<=2))
        passed = all(row["passed"] for row in details)
        comparisons.append(dict(dt_s=float(step), passed=passed, jobs=details,
            failed_jobs=sum(not row["passed"] for row in details),
            max_amplitude_difference_pp=max(row["amplitude_difference_pp"] for row in details),
            max_phase_difference_deg=max(row["phase_difference_deg"] for row in details),
            max_extra_reversals=max(row["extra_reversals"] for row in details)))
        if passed and selected is None:
            selected = float(step)
    result = dict(regression_passed=max_gain<.2 and max_phase<.2,
        max_regression_amplitude_pp=max_gain, max_regression_phase_deg=max_phase,
        new_mean_dynamic_rmse_rad={name:st.fmean(r["ac_rmse_rad"] for (sid,n),r in new.items() if sid<100 and n==name)
                                   for name in ("baseline","previous_candidate")},
        regression=regression, steps=comparisons, fine_reference=list(fine.values()),
        reversal_check="200 Hz, +/-0.003 rad/s hysteresis, at most two extra reversals vs reference",
        step_gate_passed=selected is not None,
        walk_dt_s=selected if selected is not None else .00025)
    save(batch/"rack_result.json", result)
    return result


def rpy(row):
    values = [row["quat_"+k] for k in "wxyz"]
    w,x,y,z = [v/math.sqrt(sum(a*a for a in values)) for v in values]
    return (math.degrees(math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))),
            math.degrees(math.asin(max(-1,min(1,2*(w*y-z*x))))))


def grouped_errors(error):
    return dict(all=rms(error), front_left=rms(error[:3]), other_nine=rms(error[3:]),
                **{kind: rms(error[k::3]) for k,kind in enumerate(("hip","thigh","shank"))})


def score_walk(folder):
    metadata = read(folder/"metadata.json")
    measured = {}
    records = []
    replay_targets = {}
    for case in metadata["cases"]:
        rows = load(folder/case["filename"])
        t = [r["test_s"] for r in rows]
        roll,pitch = zip(*(rpy(r) for r in rows))
        fall = next((r["test_s"] for r, a, b in zip(rows,roll,pitch)
                     if r["base_z"]<.12 or abs(a)>45 or abs(b)>45),None)
        keep = [i for i,x in enumerate(t) if case["mode"]=="replay" or 3<=x<8]
        motion = spectrum([rows[i]["gyro_x"] for i in keep])
        track = [rms(rows[i][target]-rows[i][q] for i in keep) for target,q in zip(axes("q_des"),axes("q"))]
        entry = dict(case, fall_s=fall, mean_base_vx_m_s=st.fmean(rows[i]["base_vx"] for i in keep),
                     roll_std_deg=st.pstdev(roll[i] for i in keep),
                     gyro_x_rms_rad_s=rms(rows[i]["gyro_x"] for i in keep),
                     roll_spectrum=motion, sim_track_q_rmse_rad=grouped_errors(track))
        if case["mode"] == "replay":
            key = case["case_id"]
            targets = [tuple(r[k] for k in axes("q_des")) for r in rows]
            if key in replay_targets and targets != replay_targets[key]:
                raise ValueError(f"Replay A/B target mismatch in case {key}")
            replay_targets[key] = targets
            if key not in measured:
                path = ROOT/case["source"]
                if fingerprint(path) != case["sha256"]:
                    raise ValueError(f"Ground source changed: {path}")
                start = float(case["window"]["start_s"])
                actual = [r for r in load(path) if start-.06<=r["t_mono_ms"]/1000<=start+5.06]
                stamp = [r["t_mono_ms"]/1000-start for r in actual]
                measured[key] = {
                    field: [interp(stamp, [r[field] for r in actual], x) for x in t]
                    for field in axes("q")
                }
            actual = measured[key]
            errors = [rms(r[k]-a for r,a in zip(rows,actual[k])) for k in axes("q")]
            # Real targets were sent with a ZOH; interpolate only real feedback for q comparison.
            entry.update(q_sim_vs_real_rmse_rad=grouped_errors(errors),
                         per_joint_q_sim_vs_real_rmse_rad=dict(zip(JOINTS,errors)))
            prefix = [rms(rows[i][k]-actual[k][i] for i in range(100)) for k in axes("q")]
            entry["first_half_second_q_sim_vs_real_rmse_rad"] = grouped_errors(prefix)
        records.append(entry)
    by_id = {(r["case_id"],r["variant"]):r for r in records}
    replay = []
    for key in range(4):
        a,b = [by_id[key,v] for v in ("baseline","candidate")]
        replay.append(dict(case_id=key, direction=a["direction"], session=a["window"]["session"],
            baseline=a["q_sim_vs_real_rmse_rad"], candidate=b["q_sim_vs_real_rmse_rad"],
            baseline_first_half_second=a["first_half_second_q_sim_vs_real_rmse_rad"],
            candidate_first_half_second=b["first_half_second_q_sim_vs_real_rmse_rad"],
            baseline_fall_s=a["fall_s"], candidate_fall_s=b["fall_s"]))
    count_improved = sum(r["candidate"]["front_left"]<r["baseline"]["front_left"] for r in replay)
    aggregate_nine = {v:rms(r[v]["other_nine"] for r in replay) for v in ("baseline","candidate")}
    result = dict(cases=records,replay=replay, front_left_improved_windows=count_improved,
        other_nine_rad=aggregate_nine, gate_passed=count_improved>=3 and aggregate_nine["candidate"]<=aggregate_nine["baseline"],
        all_replays_upright=all(r["baseline_fall_s"] is None and r["candidate_fall_s"] is None for r in replay),
        replay_error_definition="sim q vs real q, linearly interpolated feedback on original time axis, no phase fit",
        fall_definition="logged base height below .12m or abs(roll/pitch) above 45deg")
    save(folder/"scores.json",result)
    return result


def plot(batch, out, rack, walk):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True,exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for a, title, field in zip(ax.flat, ("Gain", "Phase", "Additional reversals", "Failed trials"),
        ("max_amplitude_difference_pp","max_phase_difference_deg","max_extra_reversals","failed_jobs")):
        steps = rack["steps"]
        a.bar([str(1000*r["dt_s"]) for r in steps],[r[field] for r in steps],color="#287b93")
        threshold = {"Gain":2,"Phase":3,"Additional reversals":2}.get(title)
        if threshold is not None:
            a.axhline(threshold,color="#b34d37",ls="--",label="Limit")
            a.legend()
        a.set(xlabel="Physics step (ms)",ylabel={"Gain":"Percentage points","Phase":"Degrees"}.get(title,"Count"),title=title)
        a.grid(axis="y",alpha=.2)
    fig.suptitle("Rack step sensitivity against 0.25 ms | worst trial in each step")
    fig.savefig(out/"resistance_actuator_2026-11-21_steps.png",dpi=160)
    plt.close(fig)
    fig, aa = plt.subplots(3, 2, figsize=(13, 10), layout="constrained")
    reference_cases = score_index(rack["fine_reference"])
    for ridx,joint in enumerate(("FL_hip","FL_thigh","FL_shank")):
        cases = sorted(key[0] for key,row in reference_cases.items() if key[1]=="baseline" and row["joint"]==joint)
        ref = [reference_cases[sid,"baseline"] for sid in cases]
        for ax, field, real_field, scale, label in (
            (aa[ridx,0],"gain","real_gain",100,"Amplitude / target (%)"),
            (aa[ridx,1],"phase_deg","real_phase_deg",1,"Phase (degrees)"),
        ):
            ax.plot([r[real_field]*scale for r in ref],"o-",label="real",color="black")
            for variant,col in (("baseline","#637a91"),("previous_candidate","#e07d38")):
                ax.plot([reference_cases[sid,variant][field]*scale for sid in cases],"o-",label=variant.replace("previous_",""),color=col)
            labels = [f"{r['frequency_hz']:g}Hz / {r['amplitude_rad']:g}\n#{r['source_id']}" for r in ref]
            ax.set(xticks=range(len(cases)),xticklabels=labels,ylabel=label,title=joint)
            ax.grid(alpha=.2)
            ax.legend()
    fig.suptitle("Rack response at 0.25 ms | input amplitude in rad")
    fig.savefig(out/"resistance_actuator_2026-11-21_rack.png",dpi=160)
    plt.close(fig)
    fig, axes_fig = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
    for a,group in zip(axes_fig.flat, ("front_left","other_nine","all","hip","thigh","shank")):
        for v, dx, color in (("baseline",-.18,"#637a91"),("candidate",.18,"#e07d38")):
            a.bar([r["case_id"]+dx for r in walk["replay"]], [r[v][group] for r in walk["replay"]],
                  width=.36,label=v,color=color)
        a.set(xticks=range(4),xticklabels=["G1 forward","G1 reverse","G2 forward","G2 reverse"],
              title=group.replace("_"," ").title(),ylabel="Sim vs real q RMSE (rad)")
        a.legend()
        a.grid(axis="y",alpha=.2)
    fig.suptitle("Ground replay | identical targets and initialisation")
    fig.savefig(out/"resistance_actuator_2026-11-21_replay.png",dpi=160)
    plt.close(fig)
    fig, aa = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for j,direction in enumerate(("forward","reverse")):
        for spec_idx, row in enumerate(r for r in walk["cases"] if
                                       r["mode"]=="replay" and r["variant"]=="baseline" and r["direction"]==direction):
            start = float(row["window"]["start_s"])
            trace = [a for a in load(ROOT/row["source"]) if start-.06<=a["t_mono_ms"]/1000<=start+5.06]
            times = [a["t_mono_ms"]/1000-start for a in trace]
            gyro = [a["gyro_x"] for a in trace]
            spec = spectrum([interp(times,gyro,i*.005) for i in range(1000)])
            aa[1,j].semilogy(spec["f_hz"],spec["psd"],lw=.9,alpha=.7,ls="--",
                            color=("#535353","#929292")[spec_idx],label=f"ground log {spec_idx+1}")
        for row in walk["cases"]:
            if row["mode"]!="closedloop" or row["direction"]!=direction:
                continue
            trace = [r for r in load(batch/"walk"/row["filename"]) if 3<=r["test_s"]<8]
            fall = row["fall_s"]
            label = row["variant"] if fall is None else f"{row['variant']} (fell at {fall:.2f}s)"
            color = {"baseline":"#287b93","candidate":"#e07d38"}[row["variant"]]
            aa[0,j].plot([r["test_s"]-3 for r in trace],[r["gyro_x"] for r in trace],lw=.8,label=label,color=color)
            spec = row["roll_spectrum"]
            aa[1,j].semilogy(spec["f_hz"],spec["psd"],lw=1.1,label=label,color=color)
        aa[0,j].set(title=direction,xlabel="Window time (s)",ylabel="Gyro x (rad/s)")
        aa[1,j].set(xlim=(.5,20),xlabel="Frequency (Hz)",ylabel="PSD ((rad/s)²/Hz)")
        aa[1,j].axvspan(4,10,color="gray",alpha=.12)
        for a in aa[:,j]:
            a.legend()
            a.grid(alpha=.2)
    fig.suptitle("Own-feedback Isaac walking at ±0.5 m/s | 5 s windows")
    fig.savefig(out/"resistance_actuator_2026-11-21_closedloop.png",dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch",type=Path)
    parser.add_argument("--plots",type=Path)
    args = parser.parse_args()
    rack = score_rack(args.batch)
    print(f"regression={rack['regression_passed']} step_gate={rack['step_gate_passed']} walk_dt={rack['walk_dt_s']}")
    if (args.batch/"walk"/"metadata.json").exists():
        walk = score_walk(args.batch/"walk")
        print(f"ground_replay_gate={walk['gate_passed']} front_left_improved={walk['front_left_improved_windows']}/4")
        if args.plots:
            plot(args.batch,args.plots,rack,walk)
