#!/usr/bin/env python3
"""Compare two recorded support conditions and two own-feedback simulations."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st

from analyze_new_motion_batch import interp, rms, spectrum

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs/deployment"
LOGS = ROOT / "imgo2_deploy/logs"
JOINTS = [f"{l}_{k}" for l in ("FL", "FR", "RL", "RR") for k in ("hip", "thigh", "shank")]
COLORS = {"Ground": "#d65b2a", "Hand-supported": "#21a07d", "MuJoCo": "#3479aa", "Isaac Sim": "#8055a5"}


def rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vec(row, name, n=12):
    return [float(row[f"{name}_{i:02}"]) for i in range(n)]


def quaternion(row):
    q = [float(row["quat_" + k]) for k in "wxyz"]
    norm = math.sqrt(sum(v*v for v in q))
    return [v/norm for v in q]


def rpy(q):
    w,x,y,z = q
    return (math.degrees(math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))),
            math.degrees(math.asin(max(-1., min(1., 2*(w*y-z*x))))),
            math.degrees(math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))))


def gravity(q):
    w,x,y,z = q
    return [2*(w*y-x*z), -2*(w*x+y*z), -(1-2*(x*x+y*y))]


def columns(samples, names):
    if not samples or not all(len(s) == len(names)+1 and all(math.isfinite(x) for x in s) for s in samples):
        raise ValueError("Empty, malformed or nonfinite state trace")
    times = [s[0] for s in samples]
    if any(y <= x for x,y in zip(times,times[1:])):
        raise ValueError("Nonmonotonic state trace")
    return {k:list(v) for k,v in zip(("t_s",*names),zip(*samples))}


def axis_keys(name, n):
    return [f"{name}_{i:02}" for i in range(n)]


ACTOR = axis_keys("obs",45) + axis_keys("act",12) + axis_keys("raw",12)
BODY = ["roll_deg","pitch_deg","yaw_relative_deg","gyro_x","gyro_y","gyro_z"]
PHYSICAL = BODY + sum((axis_keys(k,12) for k in ("q","dq","target","error","torque","pd")),[])
EXTRA = ["base_z","base_vx","base_vy","base_vz"]


def physical(t, angles, gyro, q, dq, target, torque):
    error = [a-b for a,b in zip(target,q)]
    pd = [25*e-.8*v for e,v in zip(error,dq)]
    return [t,*angles,*gyro,*q,*dq,*target,*error,*torque,*pd]


def relative_yaw(data):
    y = data["yaw_relative_deg"]
    initial = y[0]
    last = 0
    for i,a in enumerate(y):
        last += (a-initial-last+180)%360-180
        y[i] = last


def infer(policy, inputs):
    import torch
    with torch.inference_mode():
        return policy(torch.tensor(inputs,dtype=torch.float32)).tolist()


def real(root, window, policy):
    lo, hi = map(float,(window["start_s"],window["end_s"]))
    files = [root/n for n in ("robot.csv","policy.csv","meta.json")]
    pr = [r for r in rows(files[1]) if lo<=float(r["input_captured_ms"])/1000<hi]
    rr = [r for r in rows(files[0]) if lo<=float(r["t_mono_ms"])/1000<hi]
    obs, act = [vec(r,"obs",45) for r in pr],[vec(r,"act") for r in pr]
    raw = infer(policy,obs)
    recheck = max(abs(max(-3.,min(3.,a))-b) for x,y in zip(raw,act) for a,b in zip(x,y))
    if recheck >= 2e-5:
        raise ValueError("Real inputs do not reproduce the recorded policy output")
    for row in rr:
        if (int(row["mode"])!=3 or int(row["valid_mask"])!=4095 or int(row["err_mask"])!=0
            or any(x!=25. for x in vec(row,"kp")) or any(x!=.8 for x in vec(row,"kd"))
            or any(vec(row,"dq_des")) or any(vec(row,"tau_ff"))):
            raise ValueError(f"Invalid real sample or unmatched gains in {root.name}")
    actor = columns([(float(r["input_captured_ms"])/1000-lo,*x,*a,*u)
                      for r,x,a,u in zip(pr,obs,act,raw)], ACTOR)
    phy = columns([physical(float(r["t_mono_ms"])/1000-lo,rpy(quaternion(r)),
                            [float(r["gyro_"+k]) for k in "xyz"],vec(r,"q"),vec(r,"dq"),
                            vec(r,"q_des"),vec(r,"tau")) for r in rr], PHYSICAL)
    relative_yaw(phy)
    return actor, phy, {"window":window, "pair_max_error":recheck, "files":{str(f.relative_to(ROOT)):sha(f) for f in files}}


def mujoco(run, policy):
    files = [run/n for n in ("mujoco_robot.csv","mujoco_policy.csv","mujoco_metadata.json","base.yaml","ppo.yaml")]
    m = json.loads(files[2].read_text())
    if m["failure"] or m["motor_delay_ms"] or m["imu_delay_ms"] or abs(m["policy_dt"]-.02)>1e-9:
        raise ValueError("Unexpected MuJoCo configuration or incomplete trial")
    rr = [r for r in rows(files[0]) if 1<=float(r["t_s"])<6]
    pr = [r for r in rows(files[1]) if 1<=float(r["t_s"])<6]
    actor = columns([(float(r["t_s"])-1,*vec(r,"obs",45),*vec(r,"act"),*vec(r,"raw")) for r in pr],ACTOR)
    predicted = infer(policy,[vec(r,"obs",45) for r in pr])
    err = max(abs(a-b) for r,n in zip(pr,predicted) for a,b in zip(vec(r,"raw"),n))
    if err>=2e-5:
        raise ValueError("MuJoCo raw actor trace failed identity check")
    data = []
    for r in rr:
        def joint(prefix):
            return [float(r[prefix+j+"_joint"]) for j in JOINTS]
        data.append(physical(float(r["t_s"])-1,
                             [math.degrees(float(r[k+"_rad"])) for k in ("roll","pitch","yaw")],
                             [float(r[f"gyro_{k}_rads"]) for k in "xyz"],
                             joint(""),joint("dq_"),joint("qdes_"),joint("tau_"))
                    +[float(r[k]) for k in ("z_m","vx_body","vy_body","vz_body")])
    phy = columns(data,PHYSICAL+EXTRA)
    relative_yaw(phy)
    sources = files + [ROOT/"imgo2_description/mjcf"/name for name in ("scene.xml","imgo2.xml")]
    sources.append(ROOT/"imgo2_deploy/ros2_ws/scripts/mujoco_policy_benchmark.cpp")
    return actor, phy, dict(m, pair_max_error=err, window_s=[1,6],
                            files={str(f.relative_to(ROOT)):sha(f) for f in sources})


def isaac(policy, policy_sha, cfg):
    base = LOGS / "resistance_actuator_20261121/walk"
    info_file, trace = base/"metadata.json",base/"baseline_04.csv"
    info = json.loads(info_file.read_text())
    case = next(c for c in info["cases"] if c["filename"]==trace.name)
    if info["policy_sha256"]!=policy_sha or info["pd"]!=[25,.8] or case["mode"]!="closedloop" or case["vx_cmd_m_s"]!=.5:
        raise ValueError("Unexpected Isaac input case")
    nominal, scales = cfg["default_pose"],cfg["action_scale"]
    prev = [0.]*12
    commands=[.5,0.,0.]
    samples, inputs, observed = [], [], []
    for i,r in enumerate(rows(trace)):
        t=float(r["test_s"])
        gyro=[float(r["gyro_"+k]) for k in "xyz"]
        quat=quaternion(r)
        q,dq,target=vec(r,"q"),vec(r,"dq"),vec(r,"q_des")
        if t>=2 and i%4==0:
            x=[v*.25 for v in gyro]+gravity(quat)+commands+[a-b for a,b in zip(q,nominal)]+[v*.05 for v in dq]+prev
            act=[max(-3.,min(3.,(v-d)/s)) for v,d,s in zip(target,nominal,scales)]
            inputs.append([max(-100.,min(100.,v)) for v in x])
            observed.append((t,act))
            prev=act
        if 3<=t<8:
            samples.append(physical(t-3,rpy(quat),gyro,q,dq,target,vec(r,"tau_joint"))
                           +[float(r[k]) for k in EXTRA])
    raw = infer(policy,inputs)
    err = max(abs(max(-3.,min(3.,a))-b) for rr,(_,aa) in zip(raw,observed) for a,b in zip(rr,aa))
    if err>=2e-5:
        raise ValueError(f"Reconstructed Isaac inputs do not reproduce simulated targets: {err:g}")
    pairs = [(t-3,*obs,*act,*y) for (t,act),obs,y in zip(observed,inputs,raw) if 3<=t<8]
    phy = columns(samples,PHYSICAL+EXTRA)
    relative_yaw(phy)
    return columns(pairs,ACTOR),phy,{"window_s":[3,8],"physics_dt_s":info["physics_dt_s"],
                                   "pair_max_error":err,
                                   "actor_origin":"reconstructed at the original actor ticks; verified against target",
                                   "files":{str(f.relative_to(ROOT)):sha(f) for f in (info_file,trace)}}


def summarize(actor,phy):
    t = phy["t_s"]
    spec = spectrum([interp(t,phy["gyro_x"],k*.005) for k in range(1000)])
    actspec = {}
    at=actor["t_s"]
    for k in ACTOR:
        vals = actor[k]
        series = [interp(at,vals,i*.02) for i in range(250)]
        channel = spectrum(series,50.)
        actspec[k] = {"min":min(vals),"max":max(vals),"std":st.pstdev(vals),"band_4_10_rms":channel["band_4_10_rms"]}
    return {"actor_rows":len(at),"physical_rows":len(t),"gyro_x_rms_rad_s":rms(phy["gyro_x"]),
            "roll_std_deg":st.pstdev(phy["roll_deg"]),"roll_spectrum":spec,
            "base_vx_mean_m_s":st.fmean(phy["base_vx"]) if "base_vx" in phy else None,
            "actor_channels":actspec,
            "joint_groups":{stem:{j:rms(rms(phy[f"{stem}_{i:02}"]) for i in range(k,12,3))
                                   for k,j in enumerate(("hip","thigh","shank"))} for stem in ("error","dq","torque","pd")}}


def plot(data, metrics, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sections = [
        ("PPO inputs 00–08", list(zip(axis_keys("obs",9),["Gyro x ×.25","Gyro y ×.25","Gyro z ×.25",
                              "Gravity x","Gravity y","Gravity z","Command vx","Command vy","Command wz"]))),
        ("Measured / simulated body",[(k,k.replace("_"," ")) for k in BODY+EXTRA]+[("psd","Roll gyro PSD")]),
    ]
    for j,(start,label) in enumerate(((9,"q relative"),(21,"dq ×.05"),(33,"Previous action"))):
        sections.append((f"PPO inputs {start:02}–{start+11:02} · {label}",
                         [(f"obs_{start+i:02}",k) for i,k in enumerate(JOINTS)]))
    for key,label in (("act","Clipped action"),("raw","Raw actor output (real/Isaac recomputed)")):
        sections.append((label,list(zip(axis_keys(key,12),JOINTS))))
    for key,label in (("q","Joint angle (rad)"),("dq","Joint speed (rad/s)"),("target","Joint target (rad)"),
                      ("error","Target minus measured q (rad)"),("torque","Torque feedback / applied torque (Nm)"),
                      ("pd","Nominal PD from recorded q and dq (Nm)")):
        sections.append((label,list(zip(axis_keys(key,12),JOINTS))))

    ncols = 6
    nrows = sum(math.ceil(len(items)/ncols) for _,items in sections)
    fig = plt.figure(figsize=(23, nrows*1.58+2.2))
    outer = fig.add_gridspec(len(sections),1,height_ratios=[math.ceil(len(items)/ncols)+.30 for _,items in sections],
                            left=.052,right=.986,bottom=.012,top=.965,hspace=.44)
    handles = {}
    for s,(title,items) in enumerate(sections):
        grid=outer[s].subgridspec(math.ceil(len(items)/ncols),ncols,hspace=.48,wspace=.30)
        for i,(key,label) in enumerate(items):
            ax = fig.add_subplot(grid[i//ncols,i%ncols])
            for name,(a,p,_) in data.items():
                src = a if key in ACTOR else p
                if key=="psd":
                    spec=metrics[name]["roll_spectrum"]
                    ax.semilogy(spec["f_hz"],spec["psd"],lw=1.,color=COLORS[name])
                elif key in src:
                    lines=ax.plot(src["t_s"],src[key],color=COLORS[name],alpha=.88,lw=.85,label=name)
                    handles[name]=lines[0]
            if key=="psd":
                ax.set(xlim=(.5,20),xticks=[5,10,15,20],xlabel="Hz")
                ax.axvspan(4,10,color="#ccc",alpha=.16)
            else:
                ax.set(xlim=(0,5),xticks=[0,2.5,5])
            if key in EXTRA:
                ax.text(.99,.05,"sim only",ha="right",transform=ax.transAxes,fontsize=7,color="#666")
            ax.grid(alpha=.2)
            ax.tick_params(labelsize=8)
            ax.set_title(label,fontsize=9,pad=1.5)
            if i==0:
                ax.text(0,1.25,title,transform=ax.transAxes,fontsize=12,fontweight="bold")
    fig.suptitle("Four-way walking comparison | +0.5 m/s target, PD 25 / 0.8",x=.052,y=.995,ha="left",fontsize=21,fontweight="bold")
    fig.legend([handles[k] for k in COLORS],list(COLORS),loc="upper left",bbox_to_anchor=(.048,.987),ncol=4,frameon=False,fontsize=13)
    fig.text(.052,.974,"Five-second windows. Time axes share duration, not gait phase. Actual body translation is available only in simulation.",fontsize=11)
    fig.savefig(output,dpi=155,facecolor="white")
    plt.close(fig)


def save_csv(path,data):
    with path.open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(data)
        writer.writerows(zip(*data.values()))


def run(args):
    import torch
    torch.set_num_threads(1)
    identity = json.loads((DOCS/"new_motion_2026-11-21_policy_all_rows.json").read_text())
    wp=ROOT/identity["policy"]
    if sha(wp)!=identity["sha256"]:
        raise ValueError("Policy hash mismatch")
    policy=torch.jit.load(str(wp),map_location="cpu").eval()
    windows=rows(DOCS/"new_motion_2026-11-21_walk.csv")
    choose = lambda tag: next(w for w in windows if w["direction"]=="forward" and w["session"].endswith(tag))
    unlocked, supported = choose("_ground"),choose("_putup")
    folders=[LOGS/"new_motion_20261121_source"/w["session"] for w in (unlocked,supported)]
    cfg = json.loads((folders[0]/"meta.json").read_text())["config"]
    data={"Ground":real(folders[0],unlocked,policy),"Hand-supported":real(folders[1],supported,policy),
          "MuJoCo":mujoco(args.run,policy),"Isaac Sim":isaac(policy,identity["sha256"],cfg)}
    metrics={k:summarize(v[0],v[1]) for k,v in data.items()}
    for name,(a,p,m) in data.items():
        slug=name.lower().replace(" ","_").replace("-","_")
        save_csv(args.run/f"{slug}_aligned_policy.csv",a)
        save_csv(args.run/f"{slug}_aligned_robot.csv",p)
    result={"policy_sha256":identity["sha256"],"metadata":{k:v[2] for k,v in data.items()},"metrics":metrics,
            "selection":"first forward whole window per support label from the preselected command-based table",
            "time_alignment":"relative to independent 5 second windows, no phase alignment",
            "plot_channel_count":len(ACTOR)+len(PHYSICAL)+len(EXTRA)+1}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    plot(data,metrics,args.plot)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--plot",type=Path,required=True)
    a=p.parse_args()
    a.run=a.run.resolve()
    run(a)
