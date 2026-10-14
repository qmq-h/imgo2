#!/usr/bin/env python3
"""Compare recorded base-frame velocity with root-pose differences (stdlib only)."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from audit_amp_dataset import quaternion_rotation

REPO = Path(__file__).resolve().parents[3]


def analyze(path):
    data = json.loads(path.read_text())
    frames, dt = data["Frames"], float(data["FrameDuration"])
    world_error, body_error, body_vel = [], [], []
    for a, b in zip(frames, frames[1:]):
        velocity = [(b[i]-a[i])/dt for i in range(3)]
        rotation = quaternion_rotation(a[3:7])
        body = [sum(rotation[j][i]*velocity[j] for j in range(3)) for i in range(3)]
        stored = [(a[31+i]+b[31+i])/2 for i in range(3)]
        world_error.append(sum((velocity[i]-stored[i])**2 for i in range(3)))
        body_error.append(sum((body[i]-stored[i])**2 for i in range(3)))
        body_vel.append(body)
    return {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "frames": len(frames), "dt": dt, "weight": data["MotionWeight"],
            "mean_pose_derived_body_velocity": [statistics.mean(v[i] for v in body_vel) for i in range(3)],
            "mean_stored_linear_velocity": [statistics.mean(f[31+i] for f in frames) for i in range(3)],
            "mean_stored_angular_velocity": [statistics.mean(f[34+i] for f in frames) for i in range(3)],
            "linear_velocity_body_rmse": math.sqrt(statistics.mean(body_error)),
            "linear_velocity_world_rmse": math.sqrt(statistics.mean(world_error))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-dir", type=Path, default=REPO/"imgo2_rl/datasets/imgo2_motion_new")
    parser.add_argument("--json", type=Path)
    args=parser.parse_args()
    result={"clips": [analyze(p) for p in sorted(args.motion_dir.glob("*.txt"))],
            "limits": "Root pose differences use nominal FrameDuration; this cannot verify original ROS stamps or all recording labels."}
    if args.json: args.json.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
