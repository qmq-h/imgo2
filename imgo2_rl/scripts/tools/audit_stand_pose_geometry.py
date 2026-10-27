#!/usr/bin/env python3
"""Stdlib-only nominal stance geometry, keeping FOOT origin and sphere center distinct.

This checks rigid URDF geometry, not dynamics, stability or physical calibration.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from audit_amp_dataset import IDENTITY, axis_rotation, forward_kinematics, matmul, read_chain, rotate

ROOT = Path(__file__).resolve().parents[3]


def audit(path, angles):
    urdf = ET.parse(path).getroot()
    feet = {}
    for leg in ('FL', 'FR', 'RL', 'RR'):
        link_name = leg + '_FOOT'
        chain = read_chain(urdf, link_name)
        origin = forward_kinematics(chain, angles)
        orientation = IDENTITY
        index = 0
        for kind, _, origin_rotation, axis in chain:
            orientation = matmul(orientation, origin_rotation)
            if kind != 'fixed':
                orientation = matmul(orientation, axis_rotation(axis, angles[index]))
                index += 1
        spheres = [c for c in urdf.find(f"link[@name='{link_name}']").findall('collision')
                   if c.find('geometry/sphere') is not None]
        if len(spheres) != 1:
            raise ValueError(f'{link_name}: expected exactly one collision sphere')
        collision = spheres[0]
        element = collision.find('origin')
        offset = [float(v) for v in element.get('xyz', '0 0 0').split()] if element is not None else [0.] * 3
        rotated_offset = rotate(orientation, offset)
        center = [a + b for a, b in zip(origin, rotated_offset)]
        feet[leg] = {
            'link_origin_body_m': origin,
            'sphere_collision_center_body_m': center,
            'sphere_radius_m': float(collision.find('geometry/sphere').get('radius')),
            'sphere_offset_in_link_m': offset,
            'sphere_center_z_minus_link_origin_z_m': rotated_offset[2],
        }
    height = max(v['sphere_radius_m'] - v['sphere_collision_center_body_m'][2] for v in feet.values())
    return {
        'scope': 'Rigid URDF FK only; no dynamics, stability or physical calibration validation.',
        'model_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'default_leg_q_rad': angles,
        'feet': feet,
        'base_height_when_lowest_foot_sphere_touches_flat_plane_m': height,
        'FL_foot_link_origin_world_height_when_sphere_touches_floor_m': height + feet['FL']['link_origin_body_m'][2],
        'front_sphere_width_m': feet['FL']['sphere_collision_center_body_m'][1] - feet['FR']['sphere_collision_center_body_m'][1],
        'rear_sphere_width_m': feet['RL']['sphere_collision_center_body_m'][1] - feet['RR']['sphere_collision_center_body_m'][1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urdf', type=Path, default=ROOT / 'imgo2_description/urdf/imgo2_real.urdf')
    parser.add_argument('--joint-angles', type=float, nargs=3, default=[0., .87, -1.82], metavar=('HIP', 'THIGH', 'SHANK'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    text = json.dumps(audit(args.urdf, args.joint_angles), indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
