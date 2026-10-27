#!/usr/bin/env python3
"""Standard-library analysis of recorded PPO maneuver IMU/body data; no simulator needed."""
import argparse
import bisect
import csv
import gzip
import json
import math
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "imgo2_deploy/ros2_ws/scripts"))
from ppo_motion_profile import schedule


def read_csv(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return list(csv.DictReader(f))


def summarize(rows, start, end):
    data = [r for r in rows if start <= r["t_s"] < end]
    if len(data) < 2:
        raise ValueError(f"Too few samples in {start}..{end}")
    def rms(vals):
        return math.sqrt(statistics.mean(v*v for v in vals))
    result = dict(start_s=start, end_s=end, samples=len(data))
    for name in ("roll", "pitch"):
        v = [math.degrees(r[name+"_rad"]) for r in data]
        result[name+"_std_deg"] = statistics.pstdev(v)
        result[name+"_peak_to_peak_deg"] = max(v)-min(v)
        result[name+"_max_abs_deg"] = max(map(abs, v))
    result["roll_pitch_gyro_rms_rads"] = rms([math.hypot(r["gyro_x_rads"], r["gyro_y_rads"]) for r in data])
    # A time-domain rapid-motion measure, NOT a sharp frequency band filter.
    times = [r["t_s"] for r in data]
    hp = []
    for i, r in enumerate(data):
        left = bisect.bisect_left(times, times[i]-.1)
        right = bisect.bisect_right(times, times[i]+.1)
        hp.append(sum((r[k]-statistics.mean(s[k] for s in data[left:right]))**2
                      for k in ("gyro_x_rads", "gyro_y_rads")))
    result["gyro_xy_minus_0p2s_mean_rms_rads"] = math.sqrt(statistics.mean(hp))
    result["mean_yaw_rate_rads"] = statistics.mean(r["gyro_z_rads"] for r in data)
    result["yaw_rate_error_rms_rads"] = rms([r["gyro_z_rads"]-r["command_wz"] for r in data])
    if all("vx_body" in r for r in data):
        result["mean_vx_body_mps"] = statistics.mean(r["vx_body"] for r in data)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gazebo",type=Path,required=True)
    p.add_argument("--isaac",type=Path,required=True)
    p.add_argument("--gazebo-odom",type=Path,help="Optional odometry CSV for centered 0.2s body-forward speed estimates")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--low-speed",type=float,default=.1)
    a=p.parse_args()
    datasets={}
    for engine,path in (("gazebo",a.gazebo),("isaac",a.isaac)):
        rows=read_csv(path)
        data=[]
        for row in rows:
            keys=("t_s","roll_rad","pitch_rad","yaw_rad","gyro_x_rads","gyro_y_rads","gyro_z_rads")
            r={k:float(row[k]) for k in keys}
            r["command_vx"]=float(row.get("command_vx",row.get("scheduled_vx_mps",0)))
            r["command_wz"]=float(row.get("command_wz",row.get("scheduled_wz_rads",0)))
            if "vx_body" in row:r["vx_body"]=float(row["vx_body"])
            data.append(r)
        datasets[engine]=data
    if a.gazebo_odom:
        odom=read_csv(a.gazebo_odom)
        ts=[float(r["t_s"]) for r in odom]
        for r in datasets["gazebo"]:
            lo=max(0,min(len(ts)-1,bisect.bisect_left(ts,r["t_s"]-.1)))
            hi=max(0,min(len(ts)-1,bisect.bisect_left(ts,r["t_s"]+.1)))
            if ts[hi]>ts[lo]:
                dx=float(odom[hi]["x_m"])-float(odom[lo]["x_m"])
                dy=float(odom[hi]["y_m"])-float(odom[lo]["y_m"])
                r["vx_body"]=(dx*math.cos(r["yaw_rad"])+dy*math.sin(r["yaw_rad"]))/(ts[hi]-ts[lo])
    stages={r["stage"]:r for r in schedule(a.low_speed)}
    windows={}
    for label in ("initial_stand","low_baseline","high_before_fast","after_fast_decel",
                  "after_slow_decel","stand_before_turn","left_turn","right_turn","arc_turn","final_stand"):
        s=stages[label]
        windows[label+"_steady"]=(s["start_s"]+2,s["end_s"])
    for label in ("after_fast_decel","after_slow_decel","left_accel","right_accel","arc_accel"):
        s=stages[label]
        windows[label+"_first2s"]=(s["start_s"],s["start_s"]+2)
    result={"schedule":schedule(a.low_speed),"metrics":{engine:{name:summarize(rows,*w) for name,w in windows.items()}
            for engine,rows in datasets.items()},
            "definitions":{"roll_pitch_gyro_rms_rads":"sqrt(mean(gyro_x^2+gyro_y^2)); commanded yaw excluded",
            "gyro_xy_minus_0p2s_mean_rms_rads":"subtract local centered 0.2s mean per axis, then vector RMS; not a sharp spectral filter",
            "steady":"exclude first 2s of each labeled segment; a descriptive window, not proof of convergence"}}
    result["definitions"]["gazebo_vx_body"]="if odometry supplied: world displacement over centered 0.2s rotated by IMU yaw, an estimate; Isaac uses native root_lin_vel_b"
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result["metrics"],indent=2))


if __name__=="__main__":main()
