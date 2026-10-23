#!/usr/bin/env python3
"""Synchronize this explicit Imgo2 MJCF with the Gazebo URDF (stdlib).

Not a general converter. Preserves joint/collision structure, sensor order and
MuJoCo contact solver settings. Preview by default; --write updates the MJCF.
"""
import argparse
import difflib
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = ROOT/'imgo2_description/urdf/imgo2_real.gazebo.urdf'
DEFAULT_TARGET = ROOT/'imgo2_description/mjcf/imgo2.xml'
DEFAULT_WORLD = ROOT/'imgo2_deploy/ros2_ws/src/imgo2_deploy/worlds/earth.world'


def synchronize(source, target, world):
    u = ET.parse(source).getroot()
    old = target.read_text()
    text = old
    bodies = {b.get('name') for b in ET.fromstring(old).findall('.//body')}
    for link in u.findall('link'):
        name = link.get('name')
        inertial = link.find('inertial')
        if inertial is None:
            raise ValueError('Missing inertial: ' + name)
        o = inertial.find('origin')
        if o is not None and any(float(v) != 0 for v in o.get('rpy', '0 0 0').split()):
            raise ValueError('Rotated inertial frame unsupported: ' + name)
        i = inertial.find('inertia')
        attrs = {'pos': o.get('xyz', '0 0 0') if o is not None else '0 0 0',
                 'mass': inertial.find('mass').get('value'),
                 'fullinertia': ' '.join(i.get(k, '0') for k in ('ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz'))}
        element = ET.tostring(ET.Element('inertial', attrs), encoding='unicode')
        if name not in bodies:
            if name != 'base_imu':
                raise ValueError('Missing MJCF body: ' + name)
            joint = next(j for j in u.findall('joint') if j.find('child').get('link') == name)
            if joint.get('type') != 'fixed' or joint.find('parent').get('link') != 'base':
                raise ValueError('Expected fixed base IMU')
            origin = joint.find('origin')
            if any(float(v) != 0 for key in ('xyz', 'rpy') for v in origin.get(key, '0 0 0').split()):
                raise ValueError('Nonzero IMU mount requires explicit sensor review')
            addition = ('          <!-- Gazebo fixed IMU inertial mass; sensor site remains on base. -->\n'
                        f'          <body name="base_imu" pos="{origin.get("xyz", "0 0 0")}">\n'
                        f'            {element}\n          </body>\n')
            anchor = '          <body name="FL_HIP"'
            if text.count(anchor) != 1:
                raise ValueError('Unexpected base hierarchy')
            text = text.replace(anchor, addition + anchor, 1)
        else:
            pattern = r'(<body\b[^>]*\bname="' + re.escape(name) + r'"[^>]*>\s*(?:(?!<body\b).)*?)(<inertial\b[^>]*/>)'
            text, count = re.subn(pattern, lambda m: m[1]+element, text, count=1, flags=re.S)
            if count != 1:
                raise ValueError('Cannot locate explicit inertial for ' + name)
    # Gazebo URDF has no rotor inertia. Passive damping uses explicit URDF dynamics or zero.
    for joint in u.findall('joint'):
        if joint.get('type') != 'revolute':
            continue
        dynamics = joint.find('dynamics')
        damping = dynamics.get('damping', '0') if dynamics is not None else '0'
        pattern = r'<joint\b[^>]*\bname="' + re.escape(joint.get('name')) + r'"[^>]*/>'
        def update_joint(match):
            e = ET.fromstring(match[0]); e.set('damping', damping); e.set('armature', '0')
            return ET.tostring(e, encoding='unicode')
        text, count = re.subn(pattern, update_joint, text, count=1)
        if count != 1:
            raise ValueError('Missing MJCF joint: ' + joint.get('name'))
    dt = ET.parse(world).getroot().find('.//physics/max_step_size')
    if dt is None or float(dt.text) <= 0:
        raise ValueError('Missing Gazebo physics step')
    text = re.sub(r'(<option\b[^>]*\btimestep=")[^"]*(")', lambda m: m[1]+dt.text.strip()+m[2], text, count=1)
    # Higher robot priority makes its friction win over the floor's larger coefficient.
    friction = {}
    for g in u.findall('gazebo'):
        a, b = g.find('mu1'), g.find('mu2')
        if a is not None and b is not None:
            if abs(float(a.text)-float(b.text)) > 1e-12:
                raise ValueError('Anisotropic friction needs explicit conversion')
            friction[g.get('reference')] = a.text.strip()
    stack, lines = [], []
    for line in text.splitlines(keepends=True):
        body = re.search(r'<body\b[^>]*name="([^"]+)"', line)
        if body:
            stack.append(body[1])
        if '<geom ' in line and stack and stack[-1] in friction:
            geom = ET.fromstring(line.strip())
            if int(geom.get('contype', '1')) or int(geom.get('conaffinity', '1')):
                coeffs = geom.get('friction', '1 .005 .0001').split()
                coeffs[0] = friction[stack[-1]]
                geom.set('friction', ' '.join(coeffs)); geom.set('priority', '1')
                line = line[:len(line)-len(line.lstrip())]+ET.tostring(geom, encoding='unicode')+'\n'
        lines.append(line)
        if '</body>' in line:
            stack.pop()
    text = ''.join(lines)
    ET.fromstring(text)
    return old, text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    p.add_argument('--target', type=Path, default=DEFAULT_TARGET)
    p.add_argument('--world', type=Path, default=DEFAULT_WORLD)
    p.add_argument('--write', action='store_true')
    a = p.parse_args()
    old, new = synchronize(a.source, a.target, a.world)
    print(''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile=str(a.target), tofile='Gazebo-aligned MJCF')), end='')
    if a.write:
        a.target.write_text(new)
    print('updated' if old != new and a.write else 'changes pending' if old != new else 'already synchronized')


if __name__ == '__main__':
    main()
