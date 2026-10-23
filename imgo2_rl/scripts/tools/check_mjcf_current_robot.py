#!/usr/bin/env python3
"""Check explicit MJCF inertials and structure against the training URDF (stdlib).

Supports the explicit body frames, primitive collisions and zero inertial-frame
rotations used here; this is not a MuJoCo runtime/contact-equivalence test.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[3]


def numbers(value):
    return [float(x) for x in value.split()]


def matmul(a, b):
    return [[sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def rotation(value, intrinsic=False):
    x, y, z = numbers(value)
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    rx = [[1, 0, 0], [0, cx, -sx], [0, sx, cx]]
    ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]
    rz = [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]
    r = matmul(matmul(rx, ry), rz) if intrinsic else matmul(matmul(rz, ry), rx)
    return [v for row in r for v in row]


def near(a, b):
    return len(a) == len(b) and all(abs(x-y) <= 1e-7 for x, y in zip(a, b))


def structural_audit(u, m):
    """Explicit robot XML only: reject unsupported transform/geometry conventions."""
    compiler = m.find('compiler')
    if compiler is None or compiler.get('angle') != 'radian' or compiler.get('eulerseq', 'xyz') != 'xyz':
        raise ValueError('Structural audit requires radian angles and xyz Euler sequence')
    bodies = {b.get('name'): b for b in m.findall('.//body')}
    parents = {b.get('name'): parent.get('name') for parent in bodies.values() for b in parent.findall('body')}
    issues, joints_checked, collisions_checked = [], 0, 0
    for j in u.findall('joint'):
        n, child, parent = j.get('name'), j.find('child').get('link'), j.find('parent').get('link')
        b = bodies.get(child)
        if b is None:
            issues.append({'item': n, 'field': 'missing_child_body'})
            continue
        if parents.get(child) != parent:
            issues.append({'item': n, 'field': 'parent'})
        origin = j.find('origin')
        pos = numbers(origin.get('xyz', '0 0 0')) if origin is not None else [0, 0, 0]
        r = rotation(origin.get('rpy', '0 0 0')) if origin is not None else rotation('0 0 0')
        if any(k in b.attrib for k in ('quat', 'axisangle', 'xyaxes', 'zaxis')):
            raise ValueError('Unsupported body orientation: ' + child)
        if not near(pos, numbers(b.get('pos', '0 0 0'))) or not near(r, rotation(b.get('euler', '0 0 0'), True)):
            issues.append({'item': n, 'field': 'origin'})
        hinges = b.findall('joint')
        if j.get('type') == 'fixed':
            if hinges or b.find('freejoint') is not None:
                issues.append({'item': n, 'field': 'fixed_body_has_joint'})
            continue
        if j.get('type') != 'revolute':
            raise ValueError('Unsupported URDF joint type: ' + j.get('type'))
        joints_checked += 1
        h = next((h for h in hinges if h.get('name') == n), None)
        if h is None:
            issues.append({'item': n, 'field': 'missing_hinge'})
            continue
        lim = j.find('limit')
        for field, a, v in (
            ('axis', numbers(j.find('axis').get('xyz')), numbers(h.get('axis', '0 0 1'))),
            ('range', [float(lim.get('lower')), float(lim.get('upper'))], numbers(h.get('range', '0 0'))),
            ('hinge_pos', [0, 0, 0], numbers(h.get('pos', '0 0 0')))):
            if not near(a, v):
                issues.append({'item': n, 'field': field, 'urdf': a, 'mjcf': v})
        motors = [a for a in m.findall('./actuator/motor') if a.get('joint') == n]
        effort = float(lim.get('effort'))
        if len(motors) != 1 or not near(numbers(motors[0].get('ctrlrange', '0 0')), [-effort, effort]) or not near(numbers(motors[0].get('gear', '1')), [1]):
            issues.append({'item': n, 'field': 'motor_ctrlrange_or_gear'})
    for link in u.findall('link'):
        name = link.get('name')
        if name not in bodies:
            continue
        geoms = [g for g in bodies[name].findall('geom') if int(g.get('contype', '1')) or int(g.get('conaffinity', '1'))]
        expected = []
        for c in link.findall('collision'):
            o = c.find('origin')
            p = numbers(o.get('xyz', '0 0 0')) if o is not None else [0, 0, 0]
            r = rotation(o.get('rpy', '0 0 0')) if o is not None else rotation('0 0 0')
            shape = list(c.find('geometry'))[0]
            if shape.tag == 'box': size = [v/2 for v in numbers(shape.get('size'))]
            elif shape.tag == 'cylinder': size = [float(shape.get('radius')), float(shape.get('length'))/2]
            elif shape.tag == 'sphere': size = [float(shape.get('radius'))]
            else: raise ValueError('Unsupported collision geometry: ' + shape.tag)
            expected.append((shape.tag, p, r, size))
        remaining = list(geoms)
        for typ, pos, rot, size in expected:
            collisions_checked += 1
            match = None
            for g in remaining:
                if any(k in g.attrib for k in ('quat', 'axisangle', 'xyaxes', 'zaxis', 'fromto')):
                    raise ValueError('Unsupported geom transform: ' + name)
                if g.get('type') == typ and near(pos, numbers(g.get('pos', '0 0 0'))) and near(rot, rotation(g.get('euler', '0 0 0'), True)) and near(size, numbers(g.get('size', ''))):
                    match = g
                    break
            if match is None:
                issues.append({'item': name, 'field': 'collision_missing_or_different', 'type': typ, 'pos': pos})
            else:
                remaining.remove(match)
        if remaining:
            issues.append({'item': name, 'field': 'extra_collisions', 'count': len(remaining)})
    return {'issues': issues, 'active_joints_checked': joints_checked, 'joint_connections_checked': len(u.findall('joint')),
            'collisions_checked': collisions_checked,
            'scope': 'Body hierarchy, all source joint origins, active axes/ranges/anchors, motor gear/ctrlrange, explicit collision shapes/transforms/sizes. No contact dynamics equivalence.'}


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
    structure = structural_audit(u, m)
    return {'scope': 'Explicit XML inertials plus structural audit; not a runtime or contact-equivalence test.',
        'urdf': str(urdf), 'mjcf': str(mjcf),
        'urdf_sha256': hashlib.sha256(urdf.read_bytes()).hexdigest(),
        'mjcf_sha256': hashlib.sha256(mjcf.read_bytes()).hexdigest(),
        'different_bodies': rows, 'missing_bodies': missing,
        'structure': structure,
        'urdf_total_mass': sum(float(l.find('inertial/mass').get('value')) for l in links.values()),
        'mjcf_total_mass': sum(float(b.find('inertial').get('mass')) for b in bodies.values()),
        'joint_damping_and_armature': sorted({(j.get('damping'), j.get('armature'))
                                            for j in m.findall('.//joint') if j.get('type') == 'hinge'}),
        'passed': not rows and not missing and not structure['issues']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urdf', type=Path, default=REPO/'imgo2_description/urdf/imgo2_real.urdf')
    parser.add_argument('--mjcf', type=Path, default=REPO/'imgo2_description/mjcf/imgo2.xml')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    result = audit(args.urdf, args.mjcf)
    if args.json: args.json.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'passed': result['passed'], 'different_bodies':
                     [(r['body'], r['fields']) for r in result['different_bodies']],
                     'structure': result['structure']}, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
