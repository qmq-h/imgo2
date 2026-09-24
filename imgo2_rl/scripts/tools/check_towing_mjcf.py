"""Check the MuJoCo cart and towing scenes against the canonical cart URDF.

Standard library only; this is a static contract check, not a MuJoCo rollout.
"""

from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[3]
DESCRIPTION = ROOT / "imgo2_description"


def floats(value: str) -> tuple[float, ...]:
    return tuple(float(part) for part in value.split())


def close(actual: tuple[float, ...], expected: tuple[float, ...], label: str) -> None:
    if len(actual) != len(expected) or not all(
        math.isclose(a, b, rel_tol=0, abs_tol=1e-10) for a, b in zip(actual, expected)
    ):
        raise AssertionError(f"{label}: {actual} != {expected}")


def check() -> None:
    urdf = ET.parse(DESCRIPTION / "cart/cart.urdf").getroot()
    cart = ET.parse(DESCRIPTION / "mjcf/cart.xml").getroot()
    robot = ET.parse(DESCRIPTION / "mjcf/imgo2.xml").getroot()
    links = {link.get("name"): link for link in urdf.findall("link")}
    cart_bodies = {body.get("name"): body for body in cart.findall(".//body")}
    expected_bodies = {"base_link": "cart_base", **{
        f"wheel_{leg}": f"cart_wheel_{leg}" for leg in ("fl", "fr", "rl", "rr")
    }}
    total_mass = 0.0
    for urdf_name, mjcf_name in expected_bodies.items():
        source = links[urdf_name].find("inertial")
        target = cart_bodies[mjcf_name].find("inertial")
        mass = float(source.find("mass").get("value"))
        total_mass += mass
        close((float(target.get("mass")),), (mass,), mjcf_name + " mass")
        inertia = source.find("inertia")
        close(floats(target.get("diaginertia")), tuple(float(inertia.get(key))
              for key in ("ixx", "iyy", "izz")), mjcf_name + " inertia")
        if urdf_name != "base_link":
            joint = urdf.find(f"joint[@name='{urdf_name}_joint']")
            close(floats(cart_bodies[mjcf_name].get("pos")),
                  floats(joint.find("origin").get("xyz")), mjcf_name + " origin")
            hinge = cart_bodies[mjcf_name].find("joint")
            close(floats(hinge.get("axis")), floats(joint.find("axis").get("xyz")),
                  mjcf_name + " axis")
            close((float(hinge.get("damping")),), (0.016,), mjcf_name + " damping")
    close((total_mass,), (10.0,), "nominal cart mass")
    close(floats(robot.find(".//site[@name='tow_robot']").get("pos")),
          (-0.16, 0.0, 0.0), "robot tow point")
    close(floats(cart.find(".//site[@name='tow_cart']").get("pos")),
          floats(urdf.find("joint[@name='rope_attachment_joint']/origin").get("xyz")),
          "cart tow point")
    for scene_name in ("scene_tow_compliant", "scene_tow_inextensible"):
        scene = ET.parse(DESCRIPTION / "mjcf" / (scene_name + ".xml")).getroot()
        includes = {element.get("file") for element in scene.findall("include")}
        assert includes == {"imgo2.xml", "cart.xml"}, (scene_name, includes)
        rope = scene.find("tendon/spatial[@name='tow_rope']")
        assert rope is not None and rope.get("limited") == "true", scene_name
        close(floats(rope.get("range")), (0.0, 0.8), scene_name + " rope range")
        assert [site.get("site") for site in rope.findall("site")] == [
            "tow_robot", "tow_cart"
        ], scene_name
    # The new cart must collide with both the floor (1/2) and robot (2/1).
    for geom in cart.findall(".//geom"):
        contype, affinity = int(geom.get("contype")), int(geom.get("conaffinity"))
        assert contype & 2 and affinity & 1, geom.get("name")
    print("towing MJCF static contract: OK (5–15 kg scales mass and inertia together)")


if __name__ == "__main__":
    check()
