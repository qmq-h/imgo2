#!/usr/bin/env python3
"""Generate the deterministic Gazebo Classic terrain observation course (stdlib)."""
from pathlib import Path
import math
import random
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[3]
WORLD_DIR = ROOT / "imgo2_deploy/ros2_ws/src/imgo2_deploy/worlds"


def box(link, name, size, xyz, color, pitch=0):
    pose = " ".join(f"{v:.8f}" for v in (*xyz, 0, pitch, 0))
    for kind in ("collision", "visual"):
        item = ET.SubElement(link, kind, name=f"{name}_{kind}")
        ET.SubElement(item, "pose").text = pose
        geometry = ET.SubElement(item, "geometry")
        ET.SubElement(ET.SubElement(geometry, "box"), "size").text = " ".join(
            f"{v:.8f}" for v in size
        )
        if kind == "collision":
            ode = ET.SubElement(ET.SubElement(ET.SubElement(item, "surface"), "friction"), "ode")
            ET.SubElement(ode, "mu").text = "1.0"
            ET.SubElement(ode, "mu2").text = "1.0"
        else:
            material = ET.SubElement(item, "material")
            for field in ("ambient", "diffuse"):
                ET.SubElement(material, field).text = " ".join(map(str, (*color, 1)))


def model(world, name):
    item = ET.SubElement(world, "model", name=name)
    ET.SubElement(item, "static").text = "true"
    return ET.SubElement(item, "link", name="terrain")


def generate():
    # Preserve the existing world's physics/ground/lighting, replacing only obstacles.
    tree = ET.parse(WORLD_DIR / "stairs_8cm.world")
    world = tree.getroot().find("world")
    for item in list(world.findall("model")):
        world.remove(item)
    gui = ET.SubElement(world, "gui")
    camera = ET.SubElement(gui, "camera", name="terrain_overview")
    ET.SubElement(camera, "pose").text = "-6 -16 23 0 0.65 0.65"
    rng = random.Random(42)
    for level, (y, step, angle, block_h, rough_h, color) in enumerate([
        (0, .025, 5, .025, (.01, .02), (.30, .65, .40)),
        (5, .05, 10, .05, (.01, .03), (.85, .65, .20)),
        (10, .08, 15, .10, (.01, .06), (.75, .32, .25)),
    ], 1):
        prefix = f"level{level}"
        # Six ascending treads, a level platform, six descending treads.
        link = model(world, f"{prefix}_stairs_{round(step*1000)}mm")
        for i in range(6):
            height = (i + 1) * step
            box(link, f"up{i}", (.3, 2.4, height), (3 + (i+.5)*.3, y, height/2), color)
            box(link, f"down{i}", (.3, 2.4, height), (7.8-(i+.5)*.3, y, height/2), color)
        box(link, "platform", (1.2, 2.4, 6*step), (5.4, y, 3*step), color)
        # Tilted slabs: top surfaces meet ground at x=10 and 17, join a platform.
        link = model(world, f"{prefix}_ramp_{angle}deg")
        a = math.radians(angle)
        height = 3 * math.tan(a)
        for name, x, pitch in (("up", 11.5, -a), ("down", 15.5, a)):
            thickness = .20
            normal_x = math.sin(pitch)
            box(link, name, (3/math.cos(a), 2.4, thickness),
                (x-thickness/2*normal_x, y, height/2-thickness/2*math.cos(a)), color, pitch)
        box(link, "platform", (1, 2.4, height), (13.5, y, height/2), color)
        # Separated blocks leave ground-level footholds between obstacles.
        link = model(world, f"{prefix}_blocks_{round(block_h*1000)}mm")
        for i in range(6):
            for j in range(4):
                height = block_h * rng.uniform(.5, 1)
                box(link, f"block{i}_{j}", (.45, .45, height),
                    (20+(i+.5)*.6, y-1.2+(j+.5)*.6, height/2), color)
        # A tiled height field approximation, explicitly NOT Isaac's interpolated HF.
        link = model(world, f"{prefix}_rough_{round(rough_h[1]*1000)}mm")
        for i in range(12):
            for j in range(8):
                height = rng.uniform(*rough_h)
                box(link, f"tile{i}_{j}", (.3, .3, height),
                    (27+(i+.5)*.3, y-1.2+(j+.5)*.3, height/2), color)
    ET.indent(tree, space="  ")
    path = WORLD_DIR / "terrain_course.world"
    tree.write(path, encoding="utf-8", xml_declaration=True)
    print(path)


if __name__ == "__main__":
    generate()
