#!/usr/bin/env python3
"""Compare labelled real deployment windows offline; standard library only.

Does not infer foot contacts, measured base speed, or firmware latency from
host snapshots. No hardware access or policy/config mutations.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics as st
from collections import defaultdict

from audit_sim_real_chain import load, metrics, rms
from audit_go_imu_runtime import audit_log
from audit_imu_hip_log import spectrum
from compare_logged_imu_rates import check


def audit(directory, windows):
    robot, policy = load(directory/'robot.csv'), load(directory/'policy.csv')
    meta = json.loads((directory/'meta.json').read_text())
    cfg = meta['config']
    by_tick = {int(r['tick']): r for r in robot}
    errors = defaultdict(list)
    for p in policy:
        r = by_tick[int(p['tick'])]
        w, x, y, z = [r['quat_'+k] for k in 'wxyz']
        n = math.sqrt(w*w+x*x+y*y+z*z); w, x, y, z = [v/n for v in (w, x, y, z)]
        gravity = [2*(w*y-x*z), -2*(y*z+w*x), 2*(x*x+y*y)-1]
        for i in range(3):
            errors['gravity_reconstruction'].append(p[f'obs_{3+i:02}']-gravity[i])
            errors['gyro_reconstruction'].append(p[f'obs_{i:02}']-r['gyro_'+'xyz'[i]]*cfg['obs_scales']['ang_vel'])
        for j, h in enumerate(cfg['policy_to_hardware']):
            errors['q_reconstruction'].append(p[f'obs_{9+j:02}']-(r[f'q_{h:02}']-cfg['default_pose'][j])*cfg['obs_scales']['dof_pos'])
            errors['dq_reconstruction'].append(p[f'obs_{21+j:02}']-r[f'dq_{h:02}']*cfg['obs_scales']['dof_vel'])
    ppo_start = min(r['t_mono_ms'] for r in robot if r['mode'] == 3)
    for r in robot:
        if r['mode'] != 3 or r['t_mono_ms'] < ppo_start+5000:
            continue
        for j, h in enumerate(cfg['policy_to_hardware']):
            errors['consumed_action_to_target'].append(r[f'q_des_{h:02}']-(cfg['default_pose'][j]+cfg['action_scale'][j]*r[f'act_{j:02}']))
    results = {}
    for name, lo, hi in windows:
        rows = [r for r in robot if lo*1000 <= r['t_mono_ms'] < hi*1000 and r['mode'] == 3]
        pp = [r for r in policy if lo*1000 <= r['t_mono_ms'] < hi*1000]
        if len(rows) < 20 or len(pp) < 5:
            raise ValueError(f'Insufficient PPO samples in {name}')
        m = metrics(rows)
        m['window_s'] = [lo, hi]
        m['gyro_axis_rms'] = [rms([r['gyro_'+axis] for r in rows]) for axis in 'xyz']
        m['action_step_rms'] = rms([b[f'act_{j:02}']-a[f'act_{j:02}'] for a, b in zip(pp, pp[1:]) for j in range(12)])
        m['action_clip_element_fraction'] = st.mean(abs(r[f'act_{j:02}']) >= 2.99999 for r in pp for j in range(12))
        m['tau_ge23_5_element_fraction'] = st.mean(abs(r[f'tau_{j:02}']) >= 23.5 for r in rows for j in range(12))
        m['target_joint_std_rad'] = [st.pstdev(r[f'q_des_{j:02}'] for r in rows) for j in range(12)]
        m['imu_kinematics'] = check(directory/'robot.csv', 't_mono_ms', .001, lo, hi)
        if m['gyro_xy_rms'] > .1:
            bins = defaultdict(list)
            for r in rows:
                bins[math.floor(r['t_mono_ms']/10)].append(r)
            # Guard against missing bins; this descriptive spectrum assumes 100Hz.
            keys = sorted(bins)
            if len(keys) == keys[-1]-keys[0]+1:
                m['descriptive_spectra'] = {k: spectrum([st.mean(r[k] for r in bins[i]) for i in keys])
                                           for k in ['gyro_x', 'gyro_y', 'q_00', 'q_des_00']}
        results[name] = m
    return {'directory': str(directory), 'windows': results, 'runtime': audit_log(directory),
            'meta_config': cfg,
            'contracts': {k: {'rms': rms(v), 'max_abs': max(abs(x) for x in v)} for k, v in errors.items()},
            'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                              [directory/'robot.csv', directory/'policy.csv', directory/'meta.json']}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old', required=True, type=Path)
    p.add_argument('--new', required=True, type=Path)
    p.add_argument('--old-window', action='append', required=True, help='name:start:end in logger seconds')
    p.add_argument('--new-window', action='append', required=True)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    def windows(items):
        return [(s.split(':')[0], *map(float, s.split(':')[1:])) for s in items]
    old, new = audit(args.old, windows(args.old_window)), audit(args.new, windows(args.new_window))
    report = {'old': old, 'new': new,
              'config_differences': {k: {'old': old['meta_config'].get(k), 'new': v}
                                     for k, v in new['meta_config'].items() if v != old['meta_config'].get(k)},
              'limits': ['Windows have different command histories and may include lateral commands; see recorded ranges.',
                         'Quaternion/gyro agreement does not independently prove IMU-to-base mounting alignment or absolute tilt.',
                         'Logged obs is reconstructed, not necessarily the actual forward input.',
                         'Torque feedback is an estimate; >=23.5 does not prove physical saturation.',
                         'Spectral peaks are motion/control components, not measured gait frequency.',
                         'No calibrated base velocity, clearance, contact, or external body pose in real logs.']}
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print('CONFIG DIFF', report['config_differences'])
    for group in ['old', 'new']:
        for label, row in report[group]['windows'].items():
            print(group, label, 'gyro', round(row['gyro_xy_rms'], 4), 'dq', round(row['joint_dq_rms'], 4),
                  'target-error', round(row['tracking_rms_rad'], 4), 'action-step', round(row['action_step_rms'], 4),
                  'clip%', round(100*row['action_clip_element_fraction'], 3), 'tau-max', row['feedback_tau_abs_max'])


if __name__ == '__main__':
    main()
