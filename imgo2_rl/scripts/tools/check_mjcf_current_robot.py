#!/usr/bin/env python3
"""Check explicit MJCF body inertials against the current training URDF (stdlib).

Limited to the aligned body frames and zero inertial-frame rotations used here;
this is a readiness gate, not a MuJoCo runtime or collision-conversion test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[3]


def numbers(value):
    return [float(x) for x in value.split()]


def audit(urdf, mjcf):
    u, m = ET.parse(urdf).getroot(), ET.parse(mjcf).getroot()
    links = {x.get('name'): x for x in u.findall('link')}
    bodies = {x.get('name'): x for x in m.findall('.//body')}
    rows = []
    for name in sorted(set(links) & set(bodies)):
        a, b = links[name].find('inertial'), bodies[name].find('inertial')
        if a is None or b is None:
            rows.append({'body': name, 'fields': ['missing_explicit_inertial']})
            continue
        origin = a.find('origin')
        if origin is not None and any(abs(x) > 1e-12 for x in numbers(origin.get('rpy', '0 0 0'))):
            raise ValueError('Rotated URDF inertial frame unsupported: ' + name)
        if any(k in b.attrib for k in ('quat', 'euler', 'axisangle', 'xyaxes', 'zaxis')):
            raise ValueError('Rotated MJCF inertial frame unsupported: ' + name)
        am, bm = float(a.find('mass').get('value')), float(b.get('mass'))
        apos = numbers(origin.get('xyz', '0 0 0')) if origin is not None else [0, 0, 0]
        bpos = numbers(b.get('pos', '0 0 0'))
        at = a.find('inertia')
        av = [float(at.get(k, '0')) for k in ('ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz')]
        bv = numbers(b.get('fullinertia'))
        fields = []
        if abs(am-bm) > 1e-7: fields.append('mass')
        if max(abs(x-y) for x, y in zip(apos, bpos)) > 1e-7: fields.append('com')
        if max(abs(x-y) for x, y in zip(av, bv)) > 1e-7: fields.append('inertia')
        if fields:
            rows.append({'body': name, 'fields': fields, 'urdf_mass': am, 'mjcf_mass': bm,
                'urdf_com': apos, 'mjcf_com': bpos, 'urdf_inertia': av, 'mjcf_inertia': bv})
    missing = sorted(set(links) - set(bodies))
    return {'scope': 'Explicit XML values in aligned body frames; no runtime, collision or axis validation.',
        'urdf': str(urdf), 'mjcf': str(mjcf),
        'urdf_sha256': hashlib.sha256(urdf.read_bytes()).hexdigest(),
        'mjcf_sha256': hashlib.sha256(mjcf.read_bytes()).hexdigest(),
        'different_bodies': rows, 'missing_bodies': missing,
        'joint_damping_and_armature': sorted({(j.get('damping'), j.get('armature'))
                                            for j in m.findall('.//joint') if j.get('type') == 'hinge'}),
        'passed': not rows and not missing}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urdf', type=Path, default=REPO/'imgo2_description/urdf/imgo2_real.urdf')
    parser.add_argument('--mjcf', type=Path, default=REPO/'imgo2_description/mjcf/imgo2.xml')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    result = audit(args.urdf, args.mjcf)
    if args.json: args.json.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'passed': result['passed'], 'different_bodies':
                     [(r['body'], r['fields']) for r in result['different_bodies']]}, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
