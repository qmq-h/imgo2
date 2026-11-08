"""Reproduce this review figure; requires matplotlib/numpy, no hardware access."""
import csv, json, math, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"imgo2_rl/scripts/tools"))
from audit_imu_hip_log import mul,body_rotation_rate
from compare_logged_imu_rates import startup_reference
D=ROOT/"docs/deployment"
real=ROOT/"imgo2_deploy/logs/2026-11-08_21-32-40_7802/robot.csv"
rows=list(csv.DictReader(real.open()))
t=np.array([float(r["t_mono_ms"])*.001 for r in rows])
B=startup_reference(real)["quaternion_wxyz"]; inv=[B[0],*[-v for v in B[1:]]]
stats=json.loads((D/"sim_real_chain_2026-11-08.json").read_text())
fig,axs=plt.subplots(2,2,figsize=(13,8),layout="constrained")
ax=axs[0,0]
for k,c in [("vx","C0"),("vy","C1"),("wz","C2")]:
 ax.plot(t,[float(r["cmd_"+k]) for r in rows],label=k,color=c,lw=1)
for lo,hi in [(46,52),(66,70),(74,78)]:ax.axvspan(lo,hi,color="grey",alpha=.15)
ax.set(xlabel="Time since logger start (s)",ylabel="Command (m/s; yaw rad/s)",title="Real commands; shaded = pure-direction comparison")
ax.legend(ncol=3);ax.grid(alpha=.2)
ax=axs[0,1]
labels=["Forward","Backward 1","Backward 2"]
x=np.arange(3)
a=[stats["real"][k]["gyro_xy_rms"] for k in ["forward_pure","backward_pure_1","backward_pure_2"]]
b=[stats["sim"][k]["gyro_xy_rms"] for k in ["sim_kd08_forward","sim_kd08_backward","sim_kd08_backward"]]
ax.bar(x-.18,a,.36,label="Real P25 D0.8");ax.bar(x+.18,b,.36,label="Gazebo P25 D0.8")
for i,v in enumerate(a):ax.text(i-.18,v+.02,f"{v:.3f}",ha="center",fontsize=9)
for i,v in enumerate(b):ax.text(i+.18,v+.02,f"{v:.3f}",ha="center",fontsize=9)
ax.set(xticks=x,xticklabels=labels,ylabel="sqrt(mean(gx^2 + gy^2)) (rad/s)",title="Body angular motion: same nominal PD")
ax.legend();ax.grid(axis="y",alpha=.2)
bins={}
for ti,r in zip(t,rows):
 if 46<=ti<48:
  bins.setdefault(math.floor(ti*100),[]).append((ti,np.array([float(r["quat_"+k]) for k in "wxyz"]),[float(r["gyro_"+k]) for k in "xyz"]))
s={}
for k,rr in bins.items():
 q=np.mean([v[1]*(1 if np.dot(v[1],rr[0][1])>=0 else -1) for v in rr],axis=0);q/=np.linalg.norm(q)
 s[k]=(np.mean([v[0] for v in rr]),q,np.mean([v[2] for v in rr],axis=0))
times=[];bad=[];fixed=[];gyro=[]
for k in sorted(s):
 if not all(k+j in s for j in range(-2,3)):continue
 ta,qa,_=s[k-2];tb,qb,_=s[k+2]
 times.append(s[k][0]);bad.append(body_rotation_rate(qa,qb,tb-ta))
 fixed.append(body_rotation_rate(mul(mul(inv,qa),B),mul(mul(inv,qb),B),tb-ta))
 gyro.append(np.mean([s[k+j][2] for j in range(-2,3)],axis=0))
for i in range(2):
 ax=axs[1,i]
 for values,label,color in [(gyro,"Gyro body axes","black"),(bad,"Quaternion: logged right correction","C3"),(fixed,"Quaternion: offline left-reference counterfactual","C0")]:
  ax.plot(times,np.array(values)[:,i],label=label,color=color,lw=1.25)
 ax.set(xlim=(46.1,47.3),xlabel="Logger time (s)",ylabel="40 ms mean angular rate (rad/s)",title=f"Forward: body {'XY'[i]} rate consistency")
 ax.grid(alpha=.2)
 if i==0:ax.legend(fontsize=8)
fig.suptitle("2026-11-08 real / Gazebo comparison — offline reference correction is not a recovered hardware run",fontsize=12)
fig.savefig(D/"sim_real_chain_2026-11-08.png",dpi=170)
