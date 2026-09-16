"""Offline AMP height/kinematics audit; Python standard library only.

Run from imgo2_rl: python scripts/tools/audit_amp_dataset.py
This checks stored data against URDF link origins, not simulated contact points.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET


LEGS = ("FL", "FR", "RL", "RR")
IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def rotate(r, v):
    return [sum(r[i][j] * v[j] for j in range(3)) for i in range(3)]


def axis_rotation(axis, angle):
    norm = math.sqrt(sum(x * x for x in axis))
    x, y, z = [a / norm for a in axis]
    c, s, t = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return [[t*x*x+c, t*x*y-s*z, t*x*z+s*y],
            [t*x*y+s*z, t*y*y+c, t*y*z-s*x],
            [t*x*z-s*y, t*y*z+s*x, t*z*z+c]]


def quaternion_rotation(q):
    norm = math.sqrt(sum(x*x for x in q))
    x, y, z, w = [x / norm for x in q]  # Dataset uses xyzw.
    return [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]


def read_chain(urdf, tip):
    by_child = {j.find("child").get("link"): j for j in urdf.findall("joint")}
    chain = []
    while tip != "base":
        joint = by_child[tip]
        origin = joint.find("origin")
        xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()] if origin is not None else [0, 0, 0]
        rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()] if origin is not None else [0, 0, 0]
        r = matmul(axis_rotation([0, 0, 1], rpy[2]),
                   matmul(axis_rotation([0, 1, 0], rpy[1]), axis_rotation([1, 0, 0], rpy[0])))
        axis = joint.find("axis")
        axis = [float(v) for v in axis.get("xyz").split()] if axis is not None else [1, 0, 0]
        kind = joint.get("type")
        if kind not in ("revolute", "continuous", "fixed"):
            raise ValueError(f"Unsupported joint type: {kind}")
        chain.append((kind, xyz, r, axis))
        tip = joint.find("parent").get("link")
    return chain[::-1]


def forward_kinematics(chain, angles):
    p, r, index = [0.0, 0.0, 0.0], IDENTITY, 0
    for kind, xyz, origin_r, axis in chain:
        p = [a+b for a, b in zip(p, rotate(r, xyz))]
        r = matmul(r, origin_r)
        if kind != "fixed":
            r = matmul(r, axis_rotation(axis, angles[index]))
            index += 1
    if index != len(angles):
        raise ValueError("Joint count mismatch")
    return p


def summary(values):
    return {"min": min(values), "mean": statistics.mean(values), "max": max(values)}


def audit(motion_dir, urdf_path):
    urdf = ET.parse(urdf_path).getroot()
    chains = [read_chain(urdf, f"{leg}_FOOT") for leg in LEGS]
    rows = []
    for path in sorted(motion_dir.glob("*.txt")):
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        frames = data["Frames"]
        if not frames or any(len(f) != 61 or not all(math.isfinite(x) for x in f) for f in frames):
            raise ValueError(f"Invalid frames in {path}")
        errors = {"FL_FR_RL_RR": [], "FL_RL_FR_RR": []}
        mean_foot = [[statistics.mean(f[19+3*i+k] for f in frames) for k in range(3)] for i in range(4)]
        ref_world_z, fk_world_z, quat_norm_error = [], [], []
        for frame in frames:
            quat_norm_error.append(abs(math.sqrt(sum(v*v for v in frame[3:7])) - 1))
            root_r = quaternion_rotation(frame[3:7])
            for label, order in (("FL_FR_RL_RR", (0, 1, 2, 3)), ("FL_RL_FR_RR", (0, 2, 1, 3))):
                for leg_index, block in enumerate(order):
                    fk = forward_kinematics(chains[leg_index], frame[7+3*block:10+3*block])
                    ref = frame[19+3*block:22+3*block]
                    errors[label].extend(a-b for a, b in zip(fk, ref))
                    if label == "FL_FR_RL_RR":
                        ref_world_z.append(frame[2] + rotate(root_r, ref)[2])
                        fk_world_z.append(frame[2] + rotate(root_r, fk)[2])
        rows.append({
            "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "frames": len(frames), "frame_duration": data["FrameDuration"],
            "root_height_m": summary([f[2] for f in frames]),
            "max_quaternion_norm_error": max(quat_norm_error),
            "mean_foot_positions_in_stored_order_m": mean_foot,
            "fk_coordinate_rmse_m_by_order": {key: math.sqrt(statistics.mean(x*x for x in val)) for key, val in errors.items()},
            "reference_foot_origin_world_z_m": summary(ref_world_z),
            "urdf_foot_origin_world_z_m_FL_FR_RL_RR": summary(fk_world_z),
        })
    if not rows:
        raise ValueError(f"No motion .txt files in {motion_dir}")
    return {
        "motion_dir": str(motion_dir), "urdf": str(urdf_path),
        "urdf_sha256": hashlib.sha256(urdf_path.read_bytes()).hexdigest(),
        "file_count": len(rows), "frame_count": sum(row["frames"] for row in rows),
        "note": "FK compares foot link origins, not collision surfaces. Order candidates apply to joint and foot blocks together. This does not prove which mapping the running training uses.",
        "motions": rows,
    }


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-dir", type=Path, default=root / "datasets/imgo2_motion")
    parser.add_argument("--urdf", type=Path, default=root.parent / "imgo2_description/urdf/imgo2.urdf")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    result = audit(args.motion_dir, args.urdf)
    for row in result["motions"]:
        err = row["fk_coordinate_rmse_m_by_order"]
        print(f"{row['file']}: height={row['root_height_m']['mean']:.4f} m, "
              f"FK RMSE identity={err['FL_FR_RL_RR']:.4f} m, swapped={err['FL_RL_FR_RR']:.4f} m")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
