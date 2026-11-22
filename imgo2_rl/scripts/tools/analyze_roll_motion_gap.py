#!/usr/bin/env python3
"""Compare roll-band excess and probe the frozen actor on recorded snapshots."""

import argparse
import json
import math
from pathlib import Path
import statistics as st

from analyze_new_motion_batch import interp, rms, spectrum
from compare_fourway_motion import (DOCS, LOGS, ROOT, axis_keys, infer,
                                   real, rows, sha)

HIP = range(0, 12, 3)
OTHER = [j for j in range(12) if j not in HIP]
GROUPS = {
    "roll gyro": [0],
    "roll gravity": [4],
    "other IMU": [1, 2, 3, 5],
    "hip q": [9+i for i in HIP],
    "hip dq": [21+i for i in HIP],
    "thigh/shank q,dq": [base+i for base in (9,21) for i in OTHER],
    "previous hip action": [33+i for i in HIP],
    "previous other action": [33+i for i in OTHER],
    "roll IMU": [0,4],
    "hip q,dq": [base+i for base in (9,21) for i in HIP],
    "roll IMU + hip q,dq": [0,4]+[base+i for base in (9,21) for i in HIP],
}


def load_aligned(path):
    rr = rows(path)
    return {k:[float(r[k]) for r in rr] for k in rr[0]}


def band_projection(values, sample_t, fs):
    """Rectangular 5s Fourier projection, evaluated on original timestamps."""
    n = round(5*fs)
    grid = [interp(sample_t,values,j/fs) for j in range(n)]
    wave = [0.]*len(sample_t)
    for k in range(20,51):
        omega = 2*math.pi*k/5
        a = (2/n)*sum(v*math.cos(omega*j/fs) for j,v in enumerate(grid))
        b = (2/n)*sum(v*math.sin(omega*j/fs) for j,v in enumerate(grid))
        for i,t in enumerate(sample_t):
            wave[i] += a*math.cos(omega*t)+b*math.sin(omega*t)
    return wave


def channel(values,t,fs):
    data = [interp(t,values,i/fs) for i in range(round(5*fs))]
    s = spectrum(data,fs)
    near_roll_peak = math.sqrt(s["df_hz"]*sum(p for f,p in zip(s["f_hz"],s["psd"]) if 6<=f<=8))
    return dict(mean=st.fmean(values), std=st.pstdev(values), rms=rms(values),
                p95_abs=sorted(map(abs,values))[int(.95*(len(values)-1))],
                band_4_10=s["band_4_10_rms"], band_6_8=near_roll_peak,
                peak_hz=s["dominant_05_20_hz"])


def hip_target_band(actions,t):
    return rms(.125*channel([r[k] for r in actions],t,50)["band_4_10"] for k in HIP)


def clip(a):
    return [[max(-3,min(3,v)) for v in row] for row in a]


def actor_probe(actor,policy):
    t = actor["t_s"]
    inputs = list(map(list,zip(*(actor[k] for k in axis_keys("obs",45)))))
    recorded = list(zip(*(actor[k] for k in axis_keys("act",12))))
    baseline = clip(infer(policy,inputs))
    err = max(abs(x-y) for a,b in zip(baseline,recorded) for x,y in zip(a,b))
    if err>=2e-5:
        raise ValueError(f"Recorded pair mismatch: {err:g}")
    ref = hip_target_band(baseline,t)
    projection = {k:band_projection(actor[f"obs_{k:02}"],t,50)
                  for k in sorted({k for v in GROUPS.values() for k in v})}
    answer = {}
    for name,indices in GROUPS.items():
        changed = [x.copy() for x in inputs]
        for k in indices:
            for j,row in enumerate(changed):
                row[k] = max(-100,min(100,row[k]-projection[k][j]))
        altered = clip(infer(policy,changed))
        band = hip_target_band(altered,t)
        delta = [[x-y for x,y in zip(a,b)] for a,b in zip(baseline,altered)]
        answer[name] = dict(hip_target_band_after_rad=band,
                            change_pct=100*(band/ref-1),
                            removed_direct_effect_band_rad=hip_target_band(delta,t))
    return dict(verified_pair_max_error=err, baseline_hip_target_band_rad=ref, groups=answer)


def corr(a,b):
    u,v=st.fmean(a),st.fmean(b)
    return st.fmean((x-u)*(y-v) for x,y in zip(a,b))/(st.pstdev(a)*st.pstdev(b))


def kinematic_consistency(robot):
    # Device measurements are not independent; consistency only supports a kinematic interpretation.
    t=robot["t_s"]
    tt=[i*.005 for i in range(1000)]
    roll=[math.radians(interp(t,robot["roll_deg"],x)) for x in tt]
    pitch=[math.radians(interp(t,robot["pitch_deg"],x)) for x in tt]
    gyro=[[interp(t,robot[f"gyro_{k}"],x) for x in tt] for k in "xyz"]
    euler_rate=[p+math.tan(th)*(math.sin(phi)*q+math.cos(phi)*r)
                for p,q,r,phi,th in zip(*gyro,roll,pitch)]
    roll_b=band_projection(roll,tt,200)
    gyro_b=band_projection(euler_rate,tt,200)[1:-1]
    angle_rate=[(roll_b[i+1]-roll_b[i-1])/.01 for i in range(1,999)]
    hip_corr=[]
    for j in HIP:
        q=band_projection([interp(t,robot[f"q_{j:02}"],x) for x in tt],tt,200)
        dq=band_projection([interp(t,robot[f"dq_{j:02}"],x) for x in tt],tt,200)[1:-1]
        hip_corr.append(corr([(q[i+1]-q[i-1])/.01 for i in range(1,999)],dq))
    return dict(roll_rate_from_quaternion_vs_gyro_4_10_r=corr(angle_rate,gyro_b),
                hip_q_derivative_vs_logged_dq_4_10_r=hip_corr)


def direct_hip_reaction(robot):
    # Each hip axis is +X in base coordinates. This omits ground reaction.
    time_s = robot["t_s"]
    torque = [-sum(v) for v in zip(*(robot[f"torque_{i:02}"] for i in HIP))]
    roll_rate = robot["gyro_x"]
    rate_band = band_projection(roll_rate, time_s, 200)
    torque_band = band_projection(torque, time_s, 200)
    return {
        "mean_direct_roll_power_w": st.fmean(a*b for a,b in zip(torque,roll_rate)),
        "band_4_10_mean_direct_roll_power_w": st.fmean(a*b for a,b in zip(torque_band,rate_band)),
        "mean_direct_roll_power_w_by_torque_shift_ms": {
            str(ms):st.fmean(interp(time_s,torque,t+ms/1000)*v
                             for t,v in zip(time_s,roll_rate) if .01<=t<4.99)
            for ms in (-10,-5,0,5,10)
        },
    }


def all_metrics(actor,robot,policy,meta):
    t=robot["t_s"]
    a = {k:channel(v,actor["t_s"],50) for k,v in actor.items() if k!="t_s"}
    p = {k:channel(v,t,200) for k,v in robot.items() if k!="t_s"}
    for j in range(12):
        p[f"P_{j:02}"] = channel([25*x for x in robot[f"error_{j:02}"]],t,200)
        p[f"D_{j:02}"] = channel([-.8*x for x in robot[f"dq_{j:02}"]],t,200)
    grouped = {name:{metric:rms(p[f"{stem}_{i:02}"][metric] for i in ids)
                     for metric in ("std","rms","band_4_10","band_6_8")}
               for stem in ("q","target","dq","error","torque","pd","P","D")
               for name,ids in [(f"hip {stem}",HIP),(f"thigh {stem}",range(1,12,3)),
                                (f"shank {stem}",range(2,12,3))]}
    return dict(metadata=meta, policy=a, physical=p, joint_groups=grouped,
                consistency=kinematic_consistency(robot), actor_probe=actor_probe(actor,policy),
                direct_hip_reaction=direct_hip_reaction(robot),
                policy_action_at_clip_rows=sum(any(abs(actor[f"act_{i:02}"][k])>=2.99999 for i in range(12))
                                               for k in range(len(actor["t_s"]))),
                reported_torque_max_abs_nm=max(max(map(abs,robot[f"torque_{i:02}"])) for i in range(12)))


def plot(result,path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data=result["cases"]
    g=result["ground_to_two_simulators"]
    names=["gyro_x","gyro_y","gyro_z","roll_deg","pitch_deg","hip q","thigh q","shank q",
           "hip dq","thigh dq","shank dq","hip target","thigh target","shank target"]
    fig, (left,right) = plt.subplots(1,2,figsize=(15,7.3),layout="constrained")
    for offset,ground,color in ((-.18,"Ground 1","#ba4e2a"),(.18,"Ground 2","#d89636")):
        v=[]
        for name in names:
            x=g[ground][name]
            v.append(rms(x.values()))
        left.barh([i+offset for i in range(len(names))], v,height=.34,color=color,label=ground)
    left.axvline(1,color="#333",ls="--",lw=1)
    left.set(yticks=range(len(names)),yticklabels=names,
             xlabel="Ground-to-simulation band RMS ratio (RMS across engines)")
    left.set_title("Where the 4–10Hz motion differs")
    left.invert_yaxis()
    left.legend()
    keys=list(GROUPS)
    for offset,ground,color in ((-.18,"Ground 1","#ba4e2a"),(.18,"Ground 2","#d89636")):
        probe=data[ground]["actor_probe"]["groups"]
        right.barh([i+offset for i in range(len(keys))],[probe[k]["change_pct"] for k in keys],
                   height=.34,color=color,label=ground)
    right.axvline(0,color="#333",lw=1)
    right.set(yticks=range(len(keys)),yticklabels=keys,xlabel="Change in hip target band RMS (%)")
    right.set_title("Offline actor probe: remove one input group's band")
    right.invert_yaxis()
    for ax in (left,right):
        ax.grid(axis="x",alpha=.2)
    fig.suptitle("Forward command +0.5 m/s | independent runs")
    fig.savefig(path,dpi=150)
    plt.close(fig)


def plot_joint_band(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = result["cases"]
    names = ["Ground 1", "Ground 2", "MuJoCo", "Isaac Sim"]
    legs = ("FL", "FR", "RL", "RR")
    kinds = ("Hip", "Thigh", "Shank")
    colors = ("#bc5025", "#e3a338", "#23778a", "#4e5791")
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), layout="constrained")
    for i, kind in enumerate(kinds):
        for axis, prefix, unit in zip(axes[i], ("q", "dq"), ("rad", "rad/s")):
            for j, (name, color) in enumerate(zip(names, colors)):
                stats = cases[name]["physical"]
                values = [stats[f"{prefix}_{leg*3+i:02}"]["band_6_8"] for leg in range(4)]
                positions = [leg+(j-1.5)*.18 for leg in range(4)]
                axis.bar(positions, values, width=.16, color=color, label=name)
            axis.set(title=f"{kind} {'angle' if prefix == 'q' else 'speed'}",
                     ylabel=f"RMS ({unit})", xticks=range(4), xticklabels=legs)
            axis.set_ylim(bottom=0)
            axis.grid(axis="y", alpha=.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=9)
    fig.suptitle("Relative joint motion near the real robot's roll peak · 6–8 Hz")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(args):
    import torch
    torch.set_num_threads(1)
    identity=json.loads((DOCS/"new_motion_2026-11-21_policy_all_rows.json").read_text())
    weights=ROOT/identity["policy"]
    if sha(weights)!=identity["sha256"]:
        raise ValueError("Policy identity changed")
    policy=torch.jit.load(str(weights),map_location="cpu").eval()
    original=json.loads((DOCS/"fourway_motion_2026-11-21.json").read_text())
    windows=rows(DOCS/"new_motion_2026-11-21_walk.csv")
    cases={}
    for token,label in (("_ground","Ground"),("_putup","Hand-supported")):
        chosen=[w for w in windows if w["direction"]=="forward" and w["session"].endswith(token)]
        for i,w in enumerate(chosen,1):
            root=LOGS/"new_motion_20261121_source"/w["session"]
            actor, robot, meta = real(root, w, policy)
            cases[f"{label} {i}"]=all_metrics(actor,robot,policy,meta)
    for name,slug in (("MuJoCo","mujoco"),("Isaac Sim","isaac_sim")):
        paths=[args.run/f"{slug}_aligned_{suffix}.csv" for suffix in ("policy","robot")]
        cases[name]=all_metrics(*(load_aligned(p) for p in paths),policy,
                                meta=dict(original["metadata"][name],aligned_sha256={p.name:sha(p) for p in paths}))
    combined={k:dict(v["physical"],**v["joint_groups"]) for k,v in cases.items()}
    ratio = {g:{key:{sim:data["band_4_10"]/combined[sim][key]["band_4_10"]
                     for sim in ("MuJoCo","Isaac Sim")} for key,data in combined[g].items()
                if all(key in combined[sim] and combined[sim][key]["band_4_10"]>1e-8
                       for sim in ("MuJoCo","Isaac Sim"))} for g in ("Ground 1","Ground 2")}
    result=dict(policy_sha256=identity["sha256"], cases=cases, ground_to_two_simulators=ratio,
                method="Independent command-selected forward windows; 4–10Hz and 6–8Hz Hann PSD; actor probe uses the 4–10Hz rectangular Fourier band at native timestamps and keeps all other inputs and previous actions recorded.",
                scope="Offline snapshots. Counterfactual history and feedback are not evolved, so output response does not identify the physical initiating fault or certify a control intervention.")
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    if args.plot:
        plot(result,args.plot)
    if args.joint_plot:
        plot_joint_band(result, args.joint_plot)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--plot",type=Path)
    p.add_argument("--joint-plot",type=Path,help="Plot all 12 axes near the measured roll peak")
    run(p.parse_args())
