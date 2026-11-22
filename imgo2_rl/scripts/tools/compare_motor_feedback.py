#!/usr/bin/env python3
"""Stdlib-only comparison of measured feedback distributions; no hardware access."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics as st
from audit_sim_real_chain import load, metrics, rms


def audit(directory, kind, lo, hi):
    path = directory/'robot.csv'
    key, scale = ('t_mono_ms', .001) if kind == 'real' else ('sim_s', 1)
    rows = [r for r in load(path) if lo <= r[key]*scale < hi]
    result = metrics(rows)
    result.update(directory=str(directory), window=[lo, hi], kind=kind,
                  source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    for i, row in enumerate(result['per_joint']):
        row['dq_std'] = st.pstdev(r[f'dq_{i:02}'] for r in rows)
        row['tau_rms'] = rms([r[f'tau_{i:02}'] for r in rows])
        row['tau_std'] = st.pstdev(r[f'tau_{i:02}'] for r in rows)
    result['gyro_axis_rms'] = [rms([r['gyro_'+a] for r in rows]) for a in 'xyz']
    result['groups'] = {}
    for name, ids in [('hip', [0, 3, 6, 9]), ('thigh', [1, 4, 7, 10]), ('shank', [2, 5, 8, 11])]:
        result['groups'][name] = {k: math.sqrt(st.mean(result['per_joint'][i][k]**2 for i in ids))
                                  for k in ['dq_rms', 'dq_std', 'q_std', 'tracking_rms_rad', 'tau_rms', 'tau_std']}
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--real', type=Path, required=True)
    p.add_argument('--sim', type=Path, required=True)
    p.add_argument('--real-window', required=True)
    p.add_argument('--sim-window', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = {'real': audit(a.real, 'real', *map(float, a.real_window.split(':'))),
              'sim': audit(a.sim, 'sim', *map(float, a.sim_window.split(':'))),
              'limits': ['Distinct closed-loop trajectories, no phase matching or real measured base speed.',
                         'Real estimated output torque and simulated effort have different measurement models.',
                         'dq RMS includes desired gait motion; large RMS alone does not prove feedback noise.',
                         'q std is oscillation amplitude around its mean, not physical zero calibration.',
                         'Instant q_des-q compares async host records; it is not a matched-input actuator identification.']}
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    for label in ['real', 'sim']:
        print(label, 'gyro', report[label]['gyro_axis_rms'], 'groups', report[label]['groups'])
