"""Independently validate a C++ pitch FSM trace with canonical URDF FK (stdlib).

Also exports a small animation dataset from those actual commands. No dynamics.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[3]
MODEL = REPO / "imgo2_description/urdf/imgo2_real.urdf"
LEGS = ("FL", "FR", "RL", "RR")


def eye():
    return [[float(i == j) for j in range(4)] for i in range(4)]


def mul(a, b):
    return [[sum(a[i][k]*b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def rotation(axis, a):
    c, s, t = math.cos(a), math.sin(a), 1-math.cos(a)
    x, y, z = axis
    r = eye()
    for i, row in enumerate(((t*x*x+c, t*x*y-s*z, t*x*z+s*y),
                             (t*x*y+s*z, t*y*y+c, t*y*z-s*x),
                             (t*x*z-s*y, t*y*z+s*x, t*z*z+c))):
        r[i][:3] = row
    return r


def vec(element, attr):
    return [float(x) for x in (element.get(attr, "0 0 0") if element is not None else "0 0 0").split()]


def origin(element):
    x, y, z = vec(element, "xyz")
    r, p, w = vec(element, "rpy")
    out = mul(mul(rotation((0, 0, 1), w), rotation((0, 1, 0), p)), rotation((1, 0, 0), r))
    for i, value in enumerate((x, y, z)):
        out[i][3] = value
    return out


def point(transform, p=(0, 0, 0)):
    return [sum(transform[i][j]*p[j] for j in range(3))+transform[i][3] for i in range(3)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = ET.parse(MODEL).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    contacts = {leg: next(c for c in root.findall(f"link[@name='{leg}_FOOT']/collision")
                          if c.find("geometry/sphere") is not None) for leg in LEGS}

    def skeleton(q, angle):
        body = rotation((0, 1, 0), -math.radians(angle))
        chains = []
        for i, leg in enumerate(LEGS):
            transform = body
            nodes = []
            for k, role in enumerate(("hip_joint", "thigh_joint", "shank_joint", "Ankle")):
                joint = joints[f"{leg}_{role}"]
                transform = mul(transform, origin(joint.find("origin")))
                if k < 3:
                    nodes.append(point(transform))
                    transform = mul(transform, rotation(vec(joint.find("axis"), "xyz"), q[i*3+k]))
            nodes.append(point(mul(transform, origin(contacts[leg].find("origin")))))
            chains.append(nodes)
        return chains

    text = (REPO/"imgo2_deploy/real/config/deployment.yaml").read_text()
    stand_text = re.search(r"stand_pose:\s*\[([^]]+)\]", text).group(1)
    stand = [float(x) for x in re.sub(r"#[^\n]*", "", stand_text).split(",") if x.strip()]
    anchors = [chain[-1] for chain in skeleton(stand, 0)]
    rows = list(csv.DictReader(args.trace.open()))
    assert rows, "empty trace"
    maximum = 0
    frames = []
    start = float(rows[0]["t_s"])
    min_angle, max_angle = math.inf, -math.inf
    for index, row in enumerate(rows):
        angle = float(row["pitch_deg"])
        min_angle, max_angle = min(min_angle, angle), max(max_angle, angle)
        assert -20-1e-7 <= angle <= 30+1e-7
        q = [float(row[f"q_{i}"]) for i in range(12)]
        chains = skeleton(q, angle)
        error = max(math.dist(chain[-1], anchor) for chain, anchor in zip(chains, anchors))
        maximum = max(maximum, error)
        assert error < 2e-7, (index, angle, error)
        for i, leg in enumerate(LEGS):
            for k, role in enumerate(("hip", "thigh", "shank")):
                limit = joints[f"{leg}_{role}_joint"].find("limit")
                assert float(limit.get("lower")) <= q[i*3+k] <= float(limit.get("upper"))
                assert abs(float(row[f"dq_{i*3+k}"])) <= float(limit.get("velocity"))
        if index % 10 == 0 or index == len(rows)-1:
            frames.append({"t": round(float(row["t_s"])-start, 4), "mode": int(row["mode"]),
                           "angle": round(angle, 4),
                           "legs": [[[round(x, 8) for x in p] for p in chain] for chain in chains]})
    assert abs(min_angle+20)<1e-6 and abs(max_angle-30)<1e-6
    result = {"model": str(MODEL.relative_to(REPO)),
              "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
              "trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(),
              "samples": len(rows), "max_foot_error_m": maximum,
              "min_deg": min_angle, "max_deg": max_angle,
              "pivot": [0, 0, 0], "anchors": anchors,
              "nominal_body_height_m": -sum(p[2] for p in anchors)/4+.022,
              "frames": frames}
    args.output.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":"))+"\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("frames", "anchors")}, indent=2))


if __name__ == "__main__":
    main()
