#!/usr/bin/env python3
"""Stdlib-only logged-state and action-contract comparison; never drives hardware."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st


def load(path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    result = []
    for row in rows:
        r = {}
        for key, value in row.items():
            try:
                r[key] = float(value)
            except (ValueError, TypeError):
                r[key] = value
        result.append(r)
    return result


def rms(values):
    return math.sqrt(st.mean(x*x for x in values))


def angles(row):
    w, x, y, z = [row['quat_'+k] for k in 'wxyz']
    n = math.sqrt(w*w+x*x+y*y+z*z)
    w, x, y, z = [v/n for v in (w, x, y, z)]
    return (math.degrees(math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))),
            math.degrees(math.asin(max(-1, min(1, 2*(w*y-z*x))))))


def metrics(rows):
    rp = [angles(r) for r in rows]
    result = {'samples': len(rows), 'cmd_mean': [st.mean(r['cmd_'+k] for r in rows) for k in ['vx', 'vy', 'wz']],
              'cmd_min': [min(r['cmd_'+k] for r in rows) for k in ['vx', 'vy', 'wz']],
              'cmd_max': [max(r['cmd_'+k] for r in rows) for k in ['vx', 'vy', 'wz']],
              'gyro_xy_rms': math.sqrt(st.mean(r['gyro_x']**2+r['gyro_y']**2 for r in rows)),
              'gyro_xyz_rms': math.sqrt(st.mean(sum(r['gyro_'+k]**2 for k in 'xyz') for r in rows)),
              'roll_mean_deg': st.mean(r[0] for r in rp), 'roll_std_deg': st.pstdev(r[0] for r in rp),
              'pitch_mean_deg': st.mean(r[1] for r in rp), 'pitch_std_deg': st.pstdev(r[1] for r in rp),
              'joint_dq_rms': rms([r[f'dq_{i:02}'] for r in rows for i in range(12)]),
              'tracking_rms_rad': rms([r[f'q_des_{i:02}']-r[f'q_{i:02}'] for r in rows for i in range(12)]),
              'feedback_tau_rms': rms([r[f'tau_{i:02}'] for r in rows for i in range(12)]),
              'feedback_tau_abs_max': max(abs(r[f'tau_{i:02}']) for r in rows for i in range(12)),
              'kp_kd_first_joint': [st.mean(r[k+'_00'] for r in rows) for k in ['kp', 'kd']],
              'per_joint': []}
    for i in range(12):
        result['per_joint'].append({'joint': i, 'q_mean': st.mean(r[f'q_{i:02}'] for r in rows),
            'q_std': st.pstdev(r[f'q_{i:02}'] for r in rows),
            'dq_rms': rms([r[f'dq_{i:02}'] for r in rows]),
            'q_des_std': st.pstdev(r[f'q_des_{i:02}'] for r in rows),
            'tracking_rms_rad': rms([r[f'q_des_{i:02}']-r[f'q_{i:02}'] for r in rows]),
            'tau_abs_max': max(abs(r[f'tau_{i:02}']) for r in rows)})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--real', type=Path, required=True)
    p.add_argument('--sim', action='append', default=[], help='label=directory; stable window from collection.json')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    robot, policy = load(a.real/'robot.csv'), load(a.real/'policy.csv')
    meta = json.loads((a.real/'meta.json').read_text())
    cfg = meta['config']
    windows = [('stand_hold', 34.5, 37.2), ('ppo_zero_initial', 40, 43),
               ('forward_pure', 46, 52), ('backward_pure_1', 66, 70),
               ('backward_pure_2', 74, 78), ('ppo_zero_final', 84, 94)]
    real_metrics = {}
    for label, lo, hi in windows:
        rows = [r for r in robot if lo*1000 <= r['t_mono_ms'] < hi*1000]
        real_metrics[label] = dict(start_s=lo, end_s=hi, **metrics(rows))
    by_tick = {int(r['tick']): r for r in robot}
    errors = {key: [] for key in ['gyro', 'gravity', 'command', 'joint_pos', 'joint_vel', 'previous_action']}
    targets = []
    for row in robot:
        if row['mode'] == 3 and row['t_mono_ms'] > 39500:
            for j, h in enumerate(cfg['policy_to_hardware']):
                targets.append(row[f'q_des_{h:02}']-(cfg['default_pose'][j]+cfg['action_scale'][j]*row[f'act_{j:02}']))
    for index, row in enumerate(policy):
        state = by_tick[int(row['tick'])]
        for j, axis in enumerate('xyz'):
            errors['gyro'].append(row[f'obs_{j:02}']-state['gyro_'+axis]*cfg['obs_scales']['ang_vel'])
        w, x, y, z = [state['quat_'+k] for k in 'wxyz']
        norm = math.sqrt(w*w+x*x+y*y+z*z)
        w, x, y, z = [v/norm for v in (w, x, y, z)]
        gravity = [2*(w*y-x*z), -2*(y*z+w*x), 2*(x*x+y*y)-1]
        for j in range(3):
            errors['gravity'].append(row[f'obs_{3+j:02}']-gravity[j])
            errors['command'].append(row[f'obs_{6+j:02}']-row['cmd_'+['vx', 'vy', 'wz'][j]]*cfg['commands_scale'][j])
        for j, h in enumerate(cfg['policy_to_hardware']):
            errors['joint_pos'].append(row[f'obs_{9+j:02}']-(state[f'q_{h:02}']-cfg['default_pose'][j])*cfg['obs_scales']['dof_pos'])
            errors['joint_vel'].append(row[f'obs_{21+j:02}']-state[f'dq_{h:02}']*cfg['obs_scales']['dof_vel'])
            if index:
                errors['previous_action'].append(row[f'obs_{33+j:02}']-policy[index-1][f'act_{j:02}'])
    contracts = {key: {'rms': rms(v), 'max_abs': max(abs(x) for x in v)} for key, v in errors.items()}
    contracts['consumed_act_to_published_target_after_entry'] = {'rms': rms(targets), 'max_abs': max(abs(x) for x in targets)}
    sim = {}
    for item in a.sim:
        label, path = item.split('=', 1)
        directory = Path(path)
        c = json.loads((directory/'collection.json').read_text())
        lo, hi = c['ppo_pd_start_sim_s']+2, c['stand_request_sim_s']-.2
        rows = [r for r in load(directory/'robot.csv') if lo <= r['sim_s'] < hi]
        sim[label] = dict(start_sim_s=lo, end_sim_s=hi, **metrics(rows))
    report = {'real_directory': str(a.real), 'mapping': cfg['policy_to_hardware'], 'real': real_metrics,
              'sim': sim, 'contracts': contracts,
              'real_main_dt_ms': [robot[i]['t_mono_ms']-robot[i-1]['t_mono_ms'] for i in range(1, len(robot))],
              'max_temperature_c': max(r[f'temp_{j:02}'] for r in robot for j in range(12)),
              'error_mask_values': sorted({r['err_mask'] for r in robot}),
              'valid_mask_values': sorted({r['valid_mask'] for r in robot}),
              'safety_clamped_rows': sum(bool(r['clamped']) for r in robot if r['mode'] == 3),
              'source_sha256': {name: hashlib.sha256((a.real/name).read_bytes()).hexdigest() for name in ['robot.csv', 'policy.csv', 'meta.json', 'events.log']},
              'limits': ['Real actual base velocity/height/contact forces unavailable; commands are not actual velocity.',
                         'Reconstructed observations are checked against same-tick robot snapshot, not asserted to be actual inference input.',
                         'Gyro RMS measures body motion, not an independently calibrated vibration-energy measure.',
                         'Fixed-window sim and real are not matched by arbitrary phase fitting or real-world terrain/contact identification.']}
    # Compact timing distribution instead of writing all control intervals.
    dt = sorted(report.pop('real_main_dt_ms'))
    report['real_main_dt_ms'] = {k: dt[round(q*(len(dt)-1))] for k, q in [('median', .5), ('p95', .95), ('p99', .99), ('max', 1)]}
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print('CONTRACT', json.dumps(contracts))
    for label, row in {**report['real'], **report['sim']}.items():
        print(label, 'gyro', round(row['gyro_xy_rms'], 4), 'roll/pitch std', round(row['roll_std_deg'], 3), round(row['pitch_std_deg'], 3),
              'dq', round(row['joint_dq_rms'], 3), 'qerr', round(row['tracking_rms_rad'], 4))


if __name__ == '__main__':
    main()
