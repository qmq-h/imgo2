"""Stdlib-only geometry reader and validator for the P1 cart URDF.

The model deliberately supports one box and four equal solid cylinder wheels.
Fail closed on unsupported edits rather than silently using stale geometry.
"""

import math
from pathlib import Path
import xml.etree.ElementTree as ET

WHEEL_NAMES = tuple(f"wheel_{leg}" for leg in ("fl", "fr", "rl", "rr"))
JOINT_NAMES = tuple(f"{name}_joint" for name in WHEEL_NAMES)


def _vector(element, attribute, default="0 0 0"):
    values = tuple(float(v) for v in (element.get(attribute, default) if element is not None else default).split())
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError(f"Invalid {attribute}: {values}")
    return values


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def read_cart_model(path: Path) -> dict:
    """Validate the supported geometry, topology and physical parameters."""
    root = ET.parse(path).getroot()
    _require(root.tag == "robot", "Expected a complete robot URDF")
    links = {e.get("name"): e for e in root.findall("link")}
    joints = {e.get("name"): e for e in root.findall("joint")}
    _require(len(root.findall("link")) == len(links) == 6, "Expected six uniquely named links")
    _require(set(links) == {"base_link", "rope_attachment", *WHEEL_NAMES}, "Unexpected links")
    _require(len(root.findall("joint")) == len(joints) == 5, "Expected five unique joints")
    _require(set(joints) == {*JOINT_NAMES, "rope_attachment_joint"}, "Unexpected joints")
    _require(not root.findall("transmission"), "Passive cart must not contain transmissions")
    masses, inertias, origins, effort_limits, velocity_limits = {}, {}, {}, {}, {}
    radius = width = None
    size = None
    for name in ("base_link", *WHEEL_NAMES):
        link = links[name]
        inertial = link.find("inertial")
        _require(inertial is not None, f"Missing inertia: {name}")
        mass = float(inertial.find("mass").get("value"))
        _require(math.isfinite(mass) and mass > 0, f"Invalid mass: {name}")
        _require(_vector(inertial.find("origin"), "xyz") == (0, 0, 0), f"Unsupported CoM: {name}")
        _require(_vector(inertial.find("origin"), "rpy") == (0, 0, 0), f"Unsupported inertia frame: {name}")
        tensor = inertial.find("inertia")
        diagonal = tuple(float(tensor.get(k)) for k in ("ixx", "iyy", "izz"))
        _require(all(float(tensor.get(k)) == 0 for k in ("ixy", "ixz", "iyz")), f"Non-diagonal inertia: {name}")
        for kind in ("collision", "visual"):
            elems = link.findall(kind)
            _require(len(elems) == 1, f"Expected one {kind}: {name}")
            e = elems[0]
            _require(_vector(e.find("origin"), "xyz") == (0, 0, 0), f"Offset geometry: {name}")
            geom = e.find("geometry")
            _require(geom is not None and len(geom) == 1, f"Invalid geometry: {name}")
            if name == "base_link":
                _require(geom[0].tag == "box", "Deck must be a box")
                dims = _vector(geom[0], "size")
                _require(all(v > 0 for v in dims), "Invalid box dimensions")
                _require(size is None or dims == size, "Visual and collision box differ")
                size = dims
                _require(_vector(e.find("origin"), "rpy") == (0, 0, 0), "Rotated box unsupported")
            else:
                _require(geom[0].tag == "cylinder", "Wheel must be a cylinder")
                r, w = (float(geom[0].get(k)) for k in ("radius", "length"))
                _require(all(math.isfinite(v) and v > 0 for v in (r, w)), "Invalid wheel dimensions")
                _require(radius is None or (r, w) == (radius, width), "Wheels/visuals must have equal dimensions")
                radius, width = r, w
                angles = _vector(e.find("origin"), "rpy")
                _require(all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(angles, (math.pi / 2, 0, 0))), "Cylinder must rotate onto y")
        if name == "base_link":
            x, y, z = size
            expected = (mass * (y*y + z*z) / 12, mass * (x*x + z*z) / 12, mass * (x*x + y*y) / 12)
        else:
            transverse = mass * (3 * radius**2 + width**2) / 12
            expected = (transverse, mass * radius**2 / 2, transverse)
            joint = joints[f"{name}_joint"]
            _require(joint.get("type") == "continuous", f"Wheel must rotate continuously: {name}")
            _require(joint.find("parent").get("link") == "base_link" and joint.find("child").get("link") == name, "Invalid wheel topology")
            _require(_vector(joint.find("axis"), "xyz") == (0, 1, 0), f"Wrong wheel axis: {name}")
            _require(_vector(joint.find("origin"), "rpy") == (0, 0, 0), "Joint frame rotated")
            origins[name] = _vector(joint.find("origin"), "xyz")
            dynamics = joint.find("dynamics")
            _require(dynamics is not None and all(float(dynamics.get(k, 0)) == 0 for k in ("damping", "friction")), "Hidden joint resistance")
            limit = joint.find("limit")
            _require(limit is not None and not any(k in limit.attrib for k in ("lower", "upper")), "Unexpected angular limits")
            _require(all(math.isfinite(float(limit.get(k))) and float(limit.get(k)) > 0 for k in ("effort", "velocity")), "Invalid joint limits")
            effort_limits[f"{name}_joint"] = float(limit.get("effort"))
            velocity_limits[f"{name}_joint"] = float(limit.get("velocity"))
        _require(all(math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-10) for a, b in zip(diagonal, expected)), f"Solid geometry inertia mismatch: {name}")
        masses[name], inertias[name] = mass, diagonal
    attachment = joints["rope_attachment_joint"]
    _require(attachment.get("type") == "fixed" and attachment.find("parent").get("link") == "base_link" and attachment.find("child").get("link") == "rope_attachment", "Invalid attachment frame")
    _require(len(links["rope_attachment"]) == 0, "Attachment must be a massless frame")
    _require(_vector(attachment.find("origin"), "rpy") == (0, 0, 0), "Rotated attachment unsupported")
    fl, fr, rl, rr = (origins[n] for n in WHEEL_NAMES)
    _require(fl[0] > 0 and fl[1] > 0 and fl[2] < 0, "Invalid front-left wheel placement")
    _require(fr == (fl[0], -fl[1], fl[2]) and rl == (-fl[0], fl[1], fl[2]) and rr == (-fl[0], -fl[1], fl[2]), "Wheel layout must be symmetric")
    _require(fl[1] - width / 2 > size[1] / 2, "Wheel intersects deck")
    resting_height = radius - fl[2]
    _require(resting_height - size[2] / 2 > 0, "Deck touches ground before wheels")
    return dict(total_mass_kg=sum(masses.values()), masses_kg=masses, inertias_kgm2=inertias,
                wheel_radius_m=radius, wheel_width_m=width, deck_size_m=size,
                wheel_origins_m=origins, resting_height_m=resting_height,
                attachment_position_m=_vector(attachment.find("origin"), "xyz"),
                effort_limits_nm=effort_limits, velocity_limits_radps=velocity_limits,
                wheel_names=WHEEL_NAMES, joint_names=JOINT_NAMES)
