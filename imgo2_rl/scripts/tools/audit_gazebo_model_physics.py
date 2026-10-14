#!/usr/bin/env python3
"""Compare training/Gazebo URDF physics and validate inertias (stdlib only)."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[3]


def eigenvalues3(matrix):
    a = [row[:] for row in matrix]
    for _ in range(40):
        p, q = max(((0, 1), (0, 2), (1, 2)), key=lambda ij: abs(a[ij[0]][ij[1]]))
        if abs(a[p][q]) < 1e-15:
            break
        angle = 0.5 * math.atan2(2 * a[p][q], a[q][q] - a[p][p])
        c, s = math.cos(angle), math.sin(angle)
        app, aqq, apq = a[p][p], a[q][q], a[p][q]
        for k in range(3):
            if k not in (p, q):
                akp, akq = a[k][p], a[k][q]
                a[k][p] = a[p][k] = c * akp - s * akq
                a[k][q] = a[q][k] = s * akp + c * akq
        a[p][p] = c*c*app - 2*s*c*apq + s*s*aqq
        a[q][q] = s*s*app + 2*s*c*apq + c*c*aqq
        a[p][q] = a[q][p] = 0.0
    return sorted(a[i][i] for i in range(3))


def signature(element):
    if element is None:
        return None
    attrs = {}
    for key, value in element.attrib.items():
        if element.tag == "mesh" and key in ("filename", "file"):
            value = Path(value).name
        else:
            try:
                value = tuple(float(x) for x in value.split())
            except ValueError:
                pass
        attrs[key] = value
    return (element.tag, sorted(attrs.items()), [signature(x) for x in element])


def load(path):
    root = ET.parse(path).getroot()
    inertias, links, total = [], {}, 0.0
    for link in root.findall("link"):
        name = link.get("name")
        links[name] = {"inertial": signature(link.find("inertial")),
                       "collisions": [signature(x) for x in link.findall("collision")]}
        block = link.find("inertial")
        if block is None:
            continue
        mass = float(block.find("mass").get("value"))
        tensor = block.find("inertia")
        v = {key: float(tensor.get(key, "0")) for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")}
        eig = eigenvalues3([[v["ixx"], v["ixy"], v["ixz"]],
                            [v["ixy"], v["iyy"], v["iyz"]],
                            [v["ixz"], v["iyz"], v["izz"]]])
        finite = all(math.isfinite(x) for x in [mass, *v.values()])
        inertias.append({"link": name, "mass": mass, "principal_inertias": eig,
                         "positive_definite": finite and min(eig) > 0,
                         "triangle_inequality": finite and eig[2] <= eig[0] + eig[1] + 1e-12,
                         "positive_mass": finite and mass > 0})
        total += mass
    joints = {j.get("name"): signature(j) for j in root.findall("joint")}
    return root, links, joints, {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "mass_total": total, "inertias": inertias}


def audit(train_path, gazebo_path):
    _, train_links, train_joints, train = load(train_path)
    _, gazebo_links, gazebo_joints, gazebo = load(gazebo_path)
    common = sorted(set(train_links) & set(gazebo_links))
    differences = [{"link": name, "fields": [key for key in train_links[name]
                                             if train_links[name][key] != gazebo_links[name][key]]}
                   for name in common if train_links[name] != gazebo_links[name]]
    joint_differences = [name for name in train_joints if train_joints[name] != gazebo_joints.get(name)]
    missing_links = sorted(set(train_links) - set(gazebo_links))
    extra_links = sorted(set(gazebo_links) - set(train_links))
    invalid = [{"model": model["path"], **item} for model in (train, gazebo) for item in model["inertias"]
               if not all(item[key] for key in ("positive_definite", "triangle_inequality", "positive_mass"))]
    return {"training": train, "gazebo": gazebo, "common_link_count": len(common),
            "link_differences": differences, "joint_differences": joint_differences,
            "missing_links": missing_links, "extra_links": extra_links, "invalid_inertias": invalid,
            "passed": not (differences or joint_differences or missing_links or invalid or set(extra_links) - {"base_imu"}),
            "limits": "URDF consistency/physical validity only; does not verify server USD cache or identify real hardware inertia."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, default=REPO / "imgo2_description/urdf/imgo2_real.urdf")
    parser.add_argument("--gazebo", type=Path, default=REPO / "imgo2_description/urdf/imgo2_real.gazebo.urdf")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    out = audit(args.training.resolve(), args.gazebo.resolve())
    if args.json:
        args.json.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({key: value for key, value in out.items() if key not in ("training", "gazebo")}, indent=2))
    print("Mass totals:", out["training"]["mass_total"], out["gazebo"]["mass_total"])
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
