#!/usr/bin/env python3
"""Compare existing real crouch-to-stand commands with bounded Gazebo replay.

Stdlib analysis, optional matplotlib figures. Never opens a hardware device.
Ground-contact height is a geometric assumption, not a measured real base pose.
"""
import argparse
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import xml.etree.ElementTree as ET

from audit_amp_dataset import matmul, rotate, axis_rotation, quaternion_rotation
from audit_paired_obs_replay import attitude, load, rms

LEGS = ('FL', 'FR', 'RL', 'RR')
KINDS = ('hip', 'thigh', 'shank')
JOINTS = [f'{leg}_{kind}_joint' for leg in LEGS for kind in KINDS]


def vec(e, key, default='0 0 0'):
    return list(map(float, e.get(key, default).split())) if e is not None else [0., 0., 0.]


def origin(e):
    r, p, y = vec(e, 'rpy')
    return vec(e, 'xyz'), matmul(axis_rotation([0, 0, 1], y),
        matmul(axis_rotation([0, 1, 0], p), axis_rotation([1, 0, 0], r)))


def ground_estimate(model, row):
    root = ET.parse(model).getroot()
    rotation = quaternion_rotation([row['quat_'+s] for s in 'xyzw'])
    transforms = {'base': ([0., 0., 0.], rotation)}
    angles = {name: row[f'q_{i:02}'] for i, name in enumerate(JOINTS)}
    pending = list(root.findall('joint'))
    while pending:
        progressed = False
        for j in pending[:]:
            parent = j.find('parent').get('link')
            if parent not in transforms:
                continue
            p, r = transforms[parent]; xyz, r0 = origin(j.find('origin'))
            pp = [a+b for a, b in zip(p, rotate(r, xyz))]; rr = matmul(r, r0)
            if j.get('type') != 'fixed':
                rr = matmul(rr, axis_rotation(vec(j.find('axis'), 'xyz', '1 0 0'), angles[j.get('name')]))
            transforms[j.find('child').get('link')] = pp, rr
            pending.remove(j); progressed = True
        if not progressed:
            raise ValueError('unresolved model tree')
    bottoms = []
    for link in root.findall('link'):
        p, r = transforms[link.get('name')]
        for i, collision in enumerate(link.findall('collision')):
            xyz, rc = origin(collision.find('origin'))
            pc = [a+b for a, b in zip(p, rotate(r, xyz))]; rr = matmul(r, rc)
            g = collision.find('geometry'); shape = list(g)[0]
            if shape.tag == 'box':
                size = vec(shape, 'size'); extent = sum(abs(rr[2][k])*size[k]/2 for k in range(3))
            elif shape.tag == 'sphere':
                extent = float(shape.get('radius'))
            elif shape.tag == 'cylinder':
                zaxis = rr[2][2]
                extent = abs(zaxis)*float(shape.get('length'))/2 + float(shape.get('radius'))*math.sqrt(max(0, 1-zaxis*zaxis))
            else:
                raise ValueError('unsupported collision '+shape.tag)
            bottoms.append({'link': link.get('name'), 'collision': i, 'minimum_z_at_zero_base_m': pc[2]-extent})
    bottoms.sort(key=lambda r: r['minimum_z_at_zero_base_m'])
    return {'minimum_nonpenetrating_base_height_m': -bottoms[0]['minimum_z_at_zero_base_m'],
        'lowest_collisions': bottoms[:8], 'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
        'limitation': 'FK of model collision primitives with logged q/quaternion; unknown true geometry/absolute zeros/support, not real measured height'}


def at(rows, times, t, key):
    if t < times[0]-1e-9 or t > times[-1]+1e-9:
        raise ValueError('outside source coverage')
    j = min(len(times)-1, max(1, bisect.bisect_right(times, t)))
    dt = times[j]-times[j-1]
    if not 0 < dt < .05:
        raise ValueError('state gap or nonmonotonic time')
    f = (t-times[j-1])/dt
    return rows[j-1][key]*(1-f)+rows[j][key]*f


def derive(row, cmd, feedback_prefix):
    out = {}
    for i in range(12):
        key = f'{i:02}'; q = row['q_'+key]; dq = row['dq_'+key]
        target = cmd['q_des_'+key]; p = cmd['kp_'+key]*(target-q)
        d = cmd['kd_'+key]*(cmd['dq_des_'+key]-dq)
        nominal = p+d+cmd['tau_ff_'+key]
        for name, value in {'q': q, 'dq': dq, 'target': target, 'error': target-q,
                'P': p, 'D': d, 'nominal': nominal, 'feedback': row[feedback_prefix+key]}.items():
            out[name+'_'+key] = value
    attitude(row)
    out.update({k: row[k] for k in ('roll_deg', 'pitch_deg', 'gyro_x', 'gyro_y', 'gyro_z')})
    if 'z' in row:
        out['z'] = row['z']
    for i in range(12):
        for name in ('raw_q', 'raw_dq'):
            if f'{name}_{i:02}' in row:
                out[f'{name}_{i:02}'] = row[f'{name}_{i:02}']
    return out


def summarize(rows):
    joints = []
    for i in range(12):
        m = {}
        for term in ('q', 'dq', 'error', 'P', 'D', 'nominal', 'feedback'):
            v = [r[f'{term}_{i:02}'] for r in rows]
            m[term] = {'mean': st.mean(v), 'rms': rms(v), 'std': st.pstdev(v), 'peak_abs': max(map(abs, v))}
        joints.append(m)
        dt = rows[-1]['t']-rows[0]['t']
        m['q_net_slope_rad_s'] = (rows[-1][f'q_{i:02}']-rows[0][f'q_{i:02}'])/dt
    groups = {}
    for k, kind in enumerate(KINDS):
        ids = range(k, 12, 3)
        groups[kind] = {term+'_rms': math.sqrt(st.mean(joints[i][term]['rms']**2 for i in ids))
            for term in ('error', 'dq', 'P', 'D', 'nominal', 'feedback')}
        groups[kind]['mean_error_rad'] = [joints[i]['error']['mean'] for i in ids]
    return {'samples': len(rows), 'joints': joints, 'groups': groups,
        'roll_mean_deg': st.mean(r['roll_deg'] for r in rows),
        'roll_std_deg': st.pstdev(r['roll_deg'] for r in rows),
        'gyro_x_rms': rms([r['gyro_x'] for r in rows])}


def main(a):
    real = load(a.source/'robot.csv'); rt = [r['t_mono_ms']*.001 for r in real]
    first_stand = next(r for r in real if int(r['mode']) == 1)
    first_hold = next(r for r in real if int(r['mode']) == 2)
    onset = first_stand['t_mono_ms']*.001; end = first_hold['t_mono_ms']*.001
    init = real[max(0, bisect.bisect_right(rt, a.start)-1)]
    geometry = ground_estimate(a.model, init)
    prep = {'source': str(a.source), 'start_s': a.start, 'duration_s': a.duration,
        'real_support_condition': a.real_support,
        'stand_onset_s': onset, 'stand_hold_onset_s': end, 'initial_geometry': geometry,
        'gain_pairs_in_stand': sorted({(r['kp_00'], r['kd_00']) for r in real if onset <= r['t_mono_ms']*.001 < end}),
        'source_sha256': hashlib.sha256((a.source/'robot.csv').read_bytes()).hexdigest()}
    if not a.replay:
        a.output_prefix.with_suffix('.json').write_text(json.dumps(prep, indent=2)+'\n')
        print(json.dumps(prep, indent=2)); return
    cases = {}; metas = {}; differences = {}; checks = {}
    # Use original real record times; common same-target timeline, no phase fitting.
    sampled = [r for r in real if a.start+.01 <= r['t_mono_ms']*.001 < a.start+a.duration-.03]
    cases['real'] = [dict(t=r['t_mono_ms']*.001-onset, **derive(dict(r), r, 'tau_')) for r in sampled]
    for spec in a.replay:
        label, path = spec.split('=', 1); path = Path(path)
        if label == 'real' or label in cases:
            raise ValueError('duplicate case label')
        meta = json.loads((path/'metadata.json').read_text()); data = load(path/'replay.csv')
        times = [r['t_s']+meta['source_start_s'] for r in data]
        targets = load(path/'targets.csv'); sent = load(path/'published.csv')
        expected = [real[max(0, bisect.bisect_right(rt, a.start)-1)]] + [r for r in real if a.start < r['t_mono_ms']*.001 <= a.start+a.duration]
        fields = [('q', 'q_des'), ('dq', 'dq_des'), ('tau', 'tau_ff'), ('kp', 'kp'), ('kd', 'kd')]
        assert len(targets) == len(expected), 'target count'
        max_target_error = max(abs(t[f'{dst}_{i:02}']-r[f'{src}_{i:02}']) for t, r in zip(targets, expected) for dst, src in fields for i in range(12))
        max_time_error = max(abs(t['t_s']-(0 if j == 0 else expected[j]['t_mono_ms']*.001-a.start)) for j, t in enumerate(targets))
        assert max_target_error < 1e-12 and max_time_error < 1e-12
        assert len(sent) == len(targets) and [int(r['index']) for r in sent] == list(range(len(targets)))
        out = []
        keys = ['quat_'+s for s in 'wxyz']+['gyro_'+s for s in 'xyz']+['z']
        keys += [f'{k}_{i:02}' for k in ('q', 'dq', 'controller_q', 'controller_dq', 'controller_tau') for i in range(12)]
        for r in sampled:
            t = r['t_mono_ms']*.001
            state = {k: at(data, times, t, k) for k in keys}
            # Match the q/dq feedback used by the unmodified PD controller.
            # Raw physics GetVelocity may contain constraint-solver velocities
            # inconsistent with position differences even in static contact.
            for i in range(12):
                for term in ('q', 'dq'):
                    state[f'raw_{term}_{i:02}'] = state[f'{term}_{i:02}']
                    state[f'{term}_{i:02}'] = state[f'controller_{term}_{i:02}']
            out.append(dict(t=t-onset, **derive(state, r, 'controller_tau_')))
        cases[label] = out; metas[label] = meta
        checks[label] = {'targets': len(targets), 'published': len(sent), 'max_target_error': max_target_error,
            'max_time_error_s': max_time_error, 'max_publish_lag_s': max(r['t_s']-r['scheduled_s'] for r in sent),
            'completed': meta.get('completed', False), 'production_files_unchanged': meta['production_files_unchanged'],
            'sha256': {n: hashlib.sha256((path/n).read_bytes()).hexdigest() for n in ('metadata.json', 'replay.csv', 'targets.csv', 'published.csv')}}
        differences[label] = {}
    phases = {'rise': (0, end-onset), 'hold_early': (end-onset+.5, end-onset+2),
        'hold_late': (end-onset+3, a.start+a.duration-onset-.05)}
    result = {**prep, 'phases': {}, 'checks': checks, 'replay_metadata': metas,
        'limitations': ['Stand uses actual 60/2, not walking 25/0.8; no claim to identify walking gains.',
            'Real base height/support/contact/absolute joint zeros unknown; geometric height and sensitivity are assumptions.',
            'Nominal PD uses logged feedback, not motor-internal velocity or independently measured torque.',
            'Replay controller_tau is asynchronous effort feedback, not direct physical GetForce column.',
            'Main replay q/dq use controller feedback, matching actual PD. Raw physics q/dq retained separately; contact solver velocity is not interchangeable with controller position-difference velocity.',
            'Real gyro bandwidth and Gazebo exact angular velocity differ; no phase adjustment or latency inference.',
            'One slow standup does not identify unique inertia/friction/gains or high-frequency walking response.']}
    for name, (lo, hi) in phases.items():
        result['phases'][name] = {'relative_window_s': [lo, hi], 'cases': {}, 'paired_sim_minus_real': {}}
        for label, rows in cases.items():
            rr = [r for r in rows if lo <= r['t'] < hi]
            result['phases'][name]['cases'][label] = summarize(rr)
            if label == 'real': continue
            real_rows = [r for r in cases['real'] if lo <= r['t'] < hi]
            paired = {}
            for k, kind in enumerate(KINDS):
                paired[kind] = {term+'_difference_rms': rms([r[f'{term}_{i:02}']-s[f'{term}_{i:02}'] for r, s in zip(rr, real_rows) for i in range(k, 12, 3)])
                    for term in ('q', 'dq', 'nominal', 'feedback')}
            result['phases'][name]['paired_sim_minus_real'][label] = paired
    if len(metas) > 1:
        labels = list(metas)
        result['height_sensitivity'] = {}
        for label in labels[1:]:
            rr = [r for r in cases[labels[0]] if 0 <= r['t'] < end-onset]
            ss = [r for r in cases[label] if 0 <= r['t'] < end-onset]
            result['height_sensitivity'][label] = {kind: {
                'q_difference_rms_rad': rms([r[f'q_{i:02}']-s[f'q_{i:02}'] for r, s in zip(rr, ss) for i in range(k, 12, 3)]),
                'nominal_rms_change_fraction': result['phases']['rise']['cases'][label]['groups'][kind]['nominal_rms']/result['phases']['rise']['cases'][labels[0]]['groups'][kind]['nominal_rms']-1}
                for k, kind in enumerate(KINDS)}
    a.output_prefix.with_suffix('.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with Path(str(a.output_prefix)+'_traces.csv').open('w') as f:
        fields = ['case']+list(cases['real'][0])+['z']+[f'raw_{k}_{i:02}' for i in range(12) for k in ('q', 'dq')]
        w = csv.DictWriter(f, fields); w.writeheader()
        for label, rows in cases.items():
            for r in rows: w.writerow(dict(case=label, **r))
    for phase, m in result['phases'].items():
        print(phase, {label: v['groups'] for label, v in m['cases'].items()})
        print('paired', m['paired_sim_minus_real'])
    if a.plot: plot(cases, end-onset, a.output_prefix)


def plot(cases, stand_duration, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Exclude the assumed initial contact-settling transient from plot scaling;
    # retain it in the complete traces CSV and replay logs.
    cases = {label: [r for r in rows if r['t'] >= -.3] for label, rows in cases.items()}
    colors = ['#0077bb', '#228833', '#ee7733', '#aa3377']
    for leg in range(4):
        fig, axes = plt.subplots(3, 3, figsize=(16, 10), layout='constrained')
        for j in range(3):
            i = leg*3+j
            for col, (label, rows) in enumerate(cases.items()):
                for row, term in enumerate(('q', 'dq', 'nominal')):
                    axes[row,j].plot([r['t'] for r in rows], [r[f'{term}_{i:02}'] for r in rows], label=label, color=colors[col], lw=.85)
            real = cases['real']; t = [r['t'] for r in real]
            axes[0,j].plot(t, [r[f'target_{i:02}'] for r in real], 'k--', label='same target', lw=1)
            for col, (label, rows) in enumerate(cases.items()):
                axes[2,j].plot([r['t'] for r in rows], [r[f'feedback_{i:02}'] for r in rows], color=colors[col], ls=':', alpha=.55, lw=.7)
            axes[0,j].set_title(KINDS[j]); axes[0,j].set_ylabel('q / target (rad)')
            axes[1,j].set_ylabel('dq (rad/s)'); axes[2,j].set_ylabel('nominal PD / dotted feedback (Nm)')
        for ax in axes.flat:
            ax.axvline(0, color='gray', ls='--', lw=.7); ax.axvline(stand_duration, color='gray', ls='--', lw=.7)
            ax.grid(alpha=.2); ax.set_xlabel('Time from real stand command (s)')
        axes[0,0].legend(fontsize=8)
        fig.suptitle(LEGS[leg]+' recorded crouch-to-stand | actual Kp60 / Kd2\nSame commands and host cadence; real initial height/contact unknown; no fitted time shift')
        fig.savefig(str(prefix)+'_'+LEGS[leg]+'.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), layout='constrained')
    for col, (label, rows) in enumerate(cases.items()):
        for ax, key in zip(axes.flat, ('roll_deg', 'pitch_deg', 'gyro_x', 'gyro_y', 'z', 'gyro_z')):
            if key not in rows[0]: continue
            ax.plot([r['t'] for r in rows], [r[key] for r in rows], label=label, color=colors[col], lw=.85)
            ax.set_title(key); ax.grid(alpha=.2); ax.set_xlabel('Time from real stand command (s)')
            ax.axvline(0, color='gray', ls='--', lw=.7); ax.axvline(stand_duration, color='gray', ls='--', lw=.7)
    axes[0,0].legend(); fig.suptitle('Crouch-to-stand body response | real height unavailable')
    fig.savefig(str(prefix)+'_body.png', dpi=150); plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'model', 'output-prefix'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--start', type=float, default=25.); p.add_argument('--duration', type=float, default=12.)
    p.add_argument('--replay', action='append', help='label=directory; omit to estimate geometric initial height only')
    p.add_argument('--real-support', default='unknown', help='Documented support condition; does not change simulation physics')
    p.add_argument('--plot', action='store_true'); main(p.parse_args())
