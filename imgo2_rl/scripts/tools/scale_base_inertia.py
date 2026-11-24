#!/usr/bin/env python3
"""Write a URDF variant with base mass concentrated toward a body-frame axis."""

import argparse
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from audit_gazebo_model_physics import eigenvalues3


def scale_tensor(inertial, axis, scale):
    element = inertial.find("inertia")
    mass = float(inertial.find("mass").attrib["value"])
    origin = inertial.find("origin")
    rpy = origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
    roll, pitch, yaw = map(float, rpy.split())
    terms = ["ixx", "iyy", "izz", "ixy", "ixz", "iyz"]
    ixx, iyy, izz, ixy, ixz, iyz = (float(element.attrib[key]) for key in terms)
    if not mass > 0 or not all(map(math.isfinite, (mass, roll, pitch, yaw, ixx, iyy, izz, ixy, ixz, iyz))):
        raise ValueError("base mass, inertia and rpy must be finite, with positive mass")

    # URDF inertia is about the COM, in the inertial frame. Convert it to
    # the second moment S = integral(r r^T dm) in that same frame.
    half_trace = (ixx + iyy + izz) / 2
    moment = [[half_trace - ixx, -ixy, -ixz],
              [-ixy, half_trace - iyy, -iyz],
              [-ixz, -iyz, half_trace - izz]]
    if not math.isfinite(half_trace) or half_trace <= 0 or min(eigenvalues3(moment)) < -1e-10 * half_trace:
        raise ValueError("base inertia does not describe a physical mass distribution")

    cr, cp, cy = map(math.cos, (roll, pitch, yaw))
    sr, sp, sy = map(math.sin, (roll, pitch, yaw))
    # A row of Rz(yaw) Ry(pitch) Rx(roll) is the body axis in inertial coordinates.
    body_axes = ((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
                 (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
                 (-sp, cp * sr, cp * cr))
    u = body_axes["xyz".index(axis)]
    transform = [[scale * (i == j) + (1 - scale) * u[i] * u[j] for j in range(3)] for i in range(3)]
    left = [[sum(a * b for a, b in zip(row, column)) for column in zip(*moment)] for row in transform]
    scaled = [[sum(a * b for a, b in zip(row, col)) for col in transform] for row in left]

    indices = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
    trace = sum(scaled[i][i] for i in range(3))
    return {key: f"{trace * (i == j) - scaled[i][j]:.12g}" for key, (i, j) in zip(terms, indices)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Baseline URDF to read")
    parser.add_argument("--axis", choices=("x", "y", "z"), required=True, help="Base axis through the center of mass")
    parser.add_argument(
        "--shrink-percent", type=float, required=True, help="Shrink both perpendicular dimensions by this percent"
    )
    parser.add_argument("--output", type=Path, required=True, help="New URDF path")
    args = parser.parse_args()
    if not 0 <= args.shrink_percent < 100:
        parser.error("--shrink-percent must be in [0, 100)")

    source, output = args.input.resolve(), args.output.resolve()
    try:
        tree = ET.parse(source, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
        root = tree.getroot()
        inertial = root.find("link[@name='base']/inertial")
        if inertial is None or inertial.find("inertia") is None or inertial.find("mass") is None:
            raise ValueError("expected an inertial block with mass and inertia on link 'base'")
        values = scale_tensor(inertial, args.axis, 1 - args.shrink_percent / 100)
        inertial.find("inertia").attrib.update(values)
        for mesh in root.iter("mesh"):
            filename = mesh.get("filename")
            if filename and "://" not in filename and not Path(filename).is_absolute():
                mesh.set("filename", os.path.relpath((source.parent / filename).resolve(), output.parent))

        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as file:
            tree.write(file, encoding="utf-8", xml_declaration=True)
    except (OSError, ValueError, KeyError, ET.ParseError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(f"Wrote {output}")
    print("Base inertia (kg·m²): " + " ".join(f"{key}={value}" for key, value in values.items()))


if __name__ == "__main__":
    main()
