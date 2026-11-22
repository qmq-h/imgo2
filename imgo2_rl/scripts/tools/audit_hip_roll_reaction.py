#!/usr/bin/env python3
"""Estimate only direct hip actuator reaction about base X, not total body moment."""
import argparse
from bisect import bisect_right
import hashlib
import json
from pathlib import Path
import statistics as st
import xml.etree.ElementTree as ET
from audit_sim_real_chain import load
from audit_imu_hip_log import corr


def audit(path, urdf, lo, hi):
    root = ET.parse(urdf).getroot()
    for leg in ['FL', 'FR', 'RL', 'RR']:
        joint = root.find(f"joint[@name='{leg}_hip_joint']")
        assert joint.find('parent').get('link') == 'base'
        assert list(map(float, joint.find('axis').get('xyz').split())) == [1, 0, 0]
        assert list(map(float, joint.find('origin').get('rpy', '0 0 0').split())) == [0, 0, 0]
    rows = load(path/'robot.csv')
    times = [r['t_mono_ms']*.001 for r in rows]
    tau = [-sum(r[f'tau_{i:02}'] for i in [0, 3, 6, 9]) for r in rows]
    windows = []
    for a, b in [(lo, hi)]+[(x, min(x+2, hi)) for x in range(int(lo), int(hi), 2)]:
        ids = [i for i in range(len(rows)-1) if a <= times[i] < b]
        dt = [min(times[i+1], b)-times[i] for i in ids]
        omega = [rows[i]['gyro_x'] for i in ids]
        products = [tau[i]*w for i, w in zip(ids, omega)]
        shifts = {}
        for shift in [-10, -5, 0, 5, 10]:
            values = [tau[max(0, bisect_right(times, times[i]-shift*.001)-1)]*rows[i]['gyro_x'] for i in ids]
            shifts[str(shift)] = sum(v*d for v, d in zip(values, dt))/sum(dt)
        windows.append({'window_s': [a, b], 'samples': len(ids),
                        'reaction_vs_gyro_x_corr': corr([tau[i] for i in ids], omega),
                        'mean_reaction_component_power_w': sum(p*d for p, d in zip(products, dt))/sum(dt),
                        'negative_product_time_fraction': sum(d for p, d in zip(products, dt) if p < 0)/sum(dt),
                        'tau_snapshot_shift_ms_to_component_power_w': shifts})
    return {'source': str(path), 'urdf': str(urdf), 'robot_sha256': hashlib.sha256((path/'robot.csv').read_bytes()).hexdigest(),
            'formula': 'M_hip_direct_base_x = -(tau_FL_hip+tau_FR_hip+tau_RL_hip+tau_RR_hip); component P = M*gyro_x',
            'windows': windows,
            'limits': ['Assumes aligned gyro/base and signed output torque estimates equal transmitted joint torques.',
                       'Only direct actuator reaction torque; excludes hip joint constraint-force moments, passive effects, other contacts and external forces.',
                       'Not total base power, total robot energy, net damping identification, or proof of stability/causality.',
                       'Torque estimate and gyro are asynchronous host snapshots; shift scan is a sensitivity check, not a calibrated latency estimate.']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log', type=Path, required=True)
    p.add_argument('--urdf', type=Path, required=True)
    p.add_argument('--start', type=float, required=True)
    p.add_argument('--end', type=float, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.log, a.urdf, a.start, a.end)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    for r in result['windows']:
        print(json.dumps(r))
