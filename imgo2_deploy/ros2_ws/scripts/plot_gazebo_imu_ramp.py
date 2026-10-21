#!/usr/bin/env python3
"""Plot a completed run_gazebo_ppo_ramp recording; needs numpy and matplotlib."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def table(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def column(rows, name):
    return np.array([float(r[name]) for r in rows])


def smooth(values, samples):
    n = max(1, samples)
    left = n // 2
    return np.convolve(np.pad(values, (left, n-1-left), mode="edge"), np.ones(n)/n, mode="valid")


def stats(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values)),
            "min": float(np.min(values)), "max": float(np.max(values)),
            "p95_abs": float(np.percentile(np.abs(values), 95))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--output-prefix", type=Path, required=True)
    a = ap.parse_args()
    metadata = json.loads((a.run / "run_metadata.json").read_text())
    duration = 4*metadata["ramp_seconds"]+2*metadata["hold_seconds"]
    imu = [r for r in table(a.run/"imu.csv") if 0 <= float(r["t_s"]) <= duration]
    odom = table(a.run/"odom.csv")
    commands = [r for r in table(a.run/"commands.csv") if 0 <= float(r["t_s"]) <= duration]
    t, ot = column(imu,"t_s"), column(odom,"t_s")
    if len(t)<100 or not np.all(np.diff(t)>0) or not np.all(np.diff(ot)>0):
        raise ValueError("Need sufficient strictly increasing IMU/odom timestamps")
    roll, pitch = np.degrees(column(imu,"roll_rad")), np.degrees(column(imu,"pitch_rad"))
    yaw = np.degrees(np.unwrap(column(imu,"yaw_rad")))
    yaw -= yaw[0]
    dt = float(np.median(np.diff(t)))
    # Positions are world-frame truth. Differentiate there, then rotate into body.
    dx = np.gradient(column(odom,"x_m"), ot)
    dy = np.gradient(column(odom,"y_m"), ot)
    oy = column(odom,"yaw_rad")
    vx_raw = np.interp(t, ot, np.cos(oy)*dx + np.sin(oy)*dy)
    vx = smooth(vx_raw, round(.2/dt))
    ct, cv = column(commands,"t_s"), column(commands,"vx_mps")
    # Use actually published commands (zero-order hold), not ideal future values.
    idx = np.clip(np.searchsorted(ct,t,side="right")-1,0,len(ct)-1)
    command = cv[idx]
    gyro = np.stack([column(imu,f"gyro_{axis}_rads") for axis in "xyz"],axis=1)
    accel = np.stack([column(imu,f"accel_{axis}_mps2") for axis in "xyz"],axis=1)
    z = np.interp(t, ot, column(odom,"z_m"))
    phases = []
    for label in dict.fromkeys(r["stage"] for r in imu):
        mask = np.array([r["stage"]==label for r in imu])
        phases.append({"phase":label,"samples":int(mask.sum()),"t_range_s":[float(t[mask][0]),float(t[mask][-1])],
            "command_range_mps":[float(command[mask].min()),float(command[mask].max())],
            "actual_vx_mps":stats(vx[mask]),"vx_rmse_mps":float(np.sqrt(np.mean((vx[mask]-command[mask])**2))),
            "roll_deg":stats(roll[mask]),"pitch_deg":stats(pitch[mask]),
            "yaw_change_deg":float(yaw[mask][-1]-yaw[mask][0]),
            "gyro_rms_rads":np.sqrt(np.mean(gyro[mask]**2,axis=0)).tolist(),
            "height_m":stats(z[mask])})
    summary={"metadata":metadata,"imu_policy_samples":len(imu),"imu_median_rate_hz":1/dt,
             "imu_max_gap_s":float(np.diff(t).max()),"duration_s":float(t[-1]-t[0]),
             "yaw_total_change_deg":float(yaw[-1]),"height_min_m":float(z.min()),
             "vx_rmse_mps":float(np.sqrt(np.mean((vx-command)**2))),
             "roll_deg":stats(roll),"pitch_deg":stats(pitch),"phases":phases,
             "processing":{"attitude":"raw quaternion converted to RPY; yaw unwrapped relative to first policy sample",
             "actual_speed":"world pose central difference rotated to body; centered 0.2s moving average for plot/metrics",
             "gyro_accel":"raw IMU, unfiltered; acceleration includes specific force, not projected gravity",
             "limits":"one simulation, startup in first stand window; not hardware validation"}}
    prefix=a.output_prefix
    prefix.parent.mkdir(parents=True,exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    # Matplotlib's TTC family name can differ from fontconfig's SC alias.
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    family = "DejaVu Sans"
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        family = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"font.family":family,"axes.unicode_minus":False,"font.size":10})
    fig,axes=plt.subplots(6,1,figsize=(14,15),sharex=True,layout="constrained")
    axes[0].plot(t,command,color="black",ls="--",lw=1.5,label="目标前后速度")
    axes[0].plot(t,vx,color="#1875bb",lw=1,label="实际机身前后速度（0.2 s均值）")
    axes[0].set_ylabel("速度（m/s）")
    for ax,values,label,color in zip(axes[1:4],[roll,pitch,yaw],["Roll 横滚（°）","Pitch 俯仰（°）","Yaw 相对偏航（°）"],["#1875bb","#e17923","#299058"]):
        ax.plot(t,values,color=color,alpha=.4,lw=.6,label="原始姿态")
        ax.plot(t,smooth(values,round(.1/dt)),color=color,lw=1.1,label="0.1 s均值")
        ax.set_ylabel(label)
    for i,label in enumerate(["X","Y","Z"]):
        axes[4].plot(t,gyro[:,i],lw=.6,alpha=.8,label=label)
        axes[5].plot(t,accel[:,i],lw=.6,alpha=.8,label=label)
    axes[4].set_ylabel("角速度（rad/s）")
    axes[5].set_ylabel("加速度（m/s²）")
    axes[5].set_xlabel("进入PPO后的仿真时间（s）")
    hold,ramp=metadata["hold_seconds"],metadata["ramp_seconds"]
    for ax in axes:
        ax.set_xlim(0,duration)
        ax.grid(alpha=.22)
        ax.axhline(0,color="grey",lw=.5)
        for edge in [hold+i*ramp for i in range(5)]:
            ax.axvline(edge,color="grey",ls=":",lw=.8)
        ax.axvspan(0,hold,color="grey",alpha=.08)
        ax.axvspan(hold+4*ramp,duration,color="grey",alpha=.08)
        ax.legend(loc="upper right",ncol=3,fontsize=8)
    for i,label in enumerate(["0→+0.5","+0.5→0","0→−0.5","−0.5→0"]):
        axes[0].text(hold+(i+.5)*ramp,.98,label,ha="center",va="top",transform=axes[0].get_xaxis_transform(),fontsize=9)
    fig.suptitle("PPO · Gazebo无界面前进/后退变速与IMU测量\n"
                 f"每段{ramp:g} s，vy=0、yaw指令=0；采样约{1/dt:.0f} Hz；策略SHA {next(value for name, value in metadata['files'].items() if name.endswith('/policy/imgo2/ppo/policy.pt'))[:12]}",fontsize=15)
    fig.savefig(prefix.with_suffix(".png"),dpi=160)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)
    print(json.dumps({"samples":len(imu),"rate_hz":1/dt,"vx_rmse":summary["vx_rmse_mps"],
                      "yaw_delta_deg":summary["yaw_total_change_deg"],"min_height":summary["height_min_m"]}))


if __name__=="__main__":
    main()
