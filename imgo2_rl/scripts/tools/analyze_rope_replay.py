#!/usr/bin/env python3
"""Latest real free/rope trial and original-target Gazebo replay comparison.

Stdlib analysis, optional Matplotlib. No force, hardware or policy modification.
Early/late labels are temporal; exact rope boundaries were not logged.
"""
import argparse
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st

from audit_paired_obs_replay import load, attitude, summarize, rms, spectrum, wave

GROUPS = {'hip': (0, 3, 6, 9), 'thigh': (1, 4, 7, 10), 'shank': (2, 5, 8, 11)}
WINDOWS = {'early_backward_core': (61.2, 63.7), 'late_backward_core': (98.5, 101.),
    'early_backward_extended': (65., 69.), 'late_backward_extended': (103., 107.),
    'early_forward_core': (38.7, 41.2), 'late_forward_core': (117.6, 120.1)}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def before(times, t):
    n = bisect.bisect_right(times, t)-1
    if n < 0: raise ValueError('missing past command')
    return n


def extra(rows, lo, hi, real, policy_ticks):
    rr = [r for r in rows if lo <= r['t_s'] < hi]
    rt = [r['t_s'] for r in real]; groups = {}
    for name, ids in GROUPS.items():
        nominal = []; pp = []; dd = []; feedback = []; qstd = []; target_std = []
        for i in ids:
            qstd.append(st.pstdev(r[f'q_{i:02}'] for r in rr))
            targets = []
            for r in rr:
                cmd = real[before(rt, r['t_s'])]; targets.append(cmd[f'q_des_{i:02}'])
                p = cmd[f'kp_{i:02}']*(cmd[f'q_des_{i:02}']-r[f'q_{i:02}'])
                d = cmd[f'kd_{i:02}']*(cmd[f'dq_des_{i:02}']-r[f'dq_{i:02}'])
                pp.append(p); dd.append(d); nominal.append(p+d+cmd[f'tau_ff_{i:02}'])
                feedback.append(r[f'tau_{i:02}'])
            target_std.append(st.pstdev(targets))
        updates = [r for r in real if lo <= r['t_s'] < hi and int(r['tick']) in policy_ticks]
        groups[name] = {'q_std_pooled_rad': rms(qstd), 'dq_rms_rad_s': rms([r[f'dq_{i:02}'] for r in rr for i in ids]),
            'target_std_pooled_rad': rms(target_std), 'P_rms_Nm': rms(pp), 'D_rms_Nm': rms(dd),
            'nominal_rms_Nm': rms(nominal), 'feedback_rms_Nm': rms(feedback),
            'target_step_rms_rad': rms([b[f'q_des_{i:02}']-a[f'q_des_{i:02}'] for a, b in zip(updates, updates[1:]) for i in ids])}
    return groups


def check_replay(path, real):
    meta = json.loads((path/'metadata.json').read_text()); targets = load(path/'targets.csv'); sent = load(path/'published.csv')
    lo = meta['source_start_s']; hi = lo+meta['duration_s']; times = [r['t_s'] for r in real]
    expected = [real[before(times, lo)]]+[r for r in real if lo < r['t_s'] <= hi]
    if len(expected) != len(targets): raise ValueError('target count mismatch')
    fields = [('q', 'q_des'), ('dq', 'dq_des'), ('tau', 'tau_ff'), ('kp', 'kp'), ('kd', 'kd')]
    err = max(abs(t[f'{dst}_{i:02}']-r[f'{src}_{i:02}']) for t, r in zip(targets, expected) for dst, src in fields for i in range(12))
    terr = max(abs(t['t_s']-(0 if j == 0 else expected[j]['t_s']-lo)) for j, t in enumerate(targets))
    ordered = [int(r['index']) for r in sent] == list(range(len(targets)))
    if err > 1e-12 or terr > 1e-12 or not ordered or not meta.get('completed'): raise ValueError('replay checks failed')
    rows = load(path/'replay.csv')
    for r in rows:
        r['t_s'] += lo
        for i in range(12):
            for key in ('q', 'dq', 'tau'):
                r[f'raw_{key}_{i:02}'] = r[f'{key}_{i:02}']
                r[f'{key}_{i:02}'] = r[f'controller_{key}_{i:02}']
        attitude(r)
    failure = {}
    for key, test in [('first_tilt_over_60_s', lambda r: r['tilt_deg'] > 60),
                      ('first_height_below_0_2_s', lambda r: r['z'] < .2)]:
        failure[key] = next((r['t_s'] for r in rows if test(r)), None)
    failure['max_tilt_deg'] = max(r['tilt_deg'] for r in rows)
    failure['minimum_height_m'] = min(r['z'] for r in rows)
    return rows, {'metadata': meta, 'targets': len(targets), 'published': len(sent), 'indices_in_order': ordered,
        'motion_status': failure,
        'max_target_error': err, 'max_target_time_error_s': terr,
        'max_publish_lag_s': max(r['t_s']-r['scheduled_s'] for r in sent),
        'sha256': {n: sha(path/n) for n in ('targets.csv', 'replay.csv', 'published.csv', 'metadata.json')}}


def main(a):
    real = load(a.source/'robot.csv'); policy = load(a.source/'policy.csv')
    for r in real+policy: r['t_s'] = r['t_mono_ms']*.001
    for r in real: attitude(r)
    pticks = {int(r['tick']) for r in policy}
    assert all(b['t_s'] > x['t_s'] for x, b in zip(real, real[1:])), 'nonmonotonic log'
    assert all(math.isfinite(v) for r in real+policy for v in r.values()), 'nonfinite data'
    result = {'source': str(a.source), 'source_rows': len(real), 'policy_rows': len(policy),
        'support_description': 'User: walked an out-and-back independently, then with rope; feet still bear weight; rope may slightly constrain roll. Exact boundaries unknown.',
        'simulation_conditions': 'Original submitted targets and gains, full gravity, no rope/support force, free body, real q/dq/quaternion snapshot; initial heights recorded separately in run metadata, initial base linear/angular velocity zero.',
        'windows': {}, 'comparisons': {}, 'replays': {}, 'sensitivity_runs': {}, 'timeline_5s': [],
        'sha256': {n: sha(a.source/n) for n in ('robot.csv', 'policy.csv', 'meta.json', 'events.log')},
        'limitations': ['Early/late segments are temporal comparisons; not exact recorded rope-on/off labels.',
            'Rope unloading and roll constraint are confounded; results cannot establish mass-only or actuator-only causation.',
            'Equal commands do not guarantee equal measured real body speed, terrain, contact phase, support or disturbances.',
            'Target replay is open loop; no next simulated state feeds the real policy; cannot establish closed-loop deployment stability.',
            'Publish order/timing checks do not independently verify every controller receipt or application timestamp.',
            'No rope force, application point or constraint was measured. Both replays run without rope; unknown real base state limits absolute trajectory matching.',
            'Main matched windows are 2.5s with .4Hz spectral resolution; peak bins do not prove resonance. Extended backward windows differ in lateral command.',
            'Nominal PD and estimated feedback are not independently calibrated mechanical load; replay dq uses controller feedback, not raw physics GetVelocity.',
            'A late fault after lie-down is separate from selected walking windows; no cause inferred from event text alone.']}
    pdata = {}; all_cases = {}
    for spec in a.replay:
        label, directory = spec.split('=', 1)
        pdata[label], result['replays'][label] = check_replay(Path(directory), real)
    for spec in a.sensitivity:
        label, directory = spec.split('=', 1); rows, checks = check_replay(Path(directory), real)
        checks['windows'] = {}
        for name, (lo, hi) in WINDOWS.items():
            if rows[0]['t_s'] <= lo < hi <= rows[-1]['t_s']:
                m = summarize(rows, lo, hi, real); m['normal_motion_valid'] = m.get('upright', True)
                checks['windows'][name] = m
        result['sensitivity_runs'][label] = checks
    for name, (lo, hi) in WINDOWS.items():
        cases = {'real': real}; label = 'early_backward' if name.startswith('early_backward') else 'late_backward' if name.startswith('late_backward') else name.removesuffix('_core')
        if label in pdata: cases['gazebo'] = pdata[label]
        result['windows'][name] = {}
        source = [r for r in real if lo <= r['t_s'] < hi]
        assert all(r['mode'] == 3 and r['valid_mask'] == 4095 and r['err_mask'] == 0 for r in source)
        assert all(abs(r[f'kp_{i:02}']-25) < 1e-6 and abs(r[f'kd_{i:02}']-.8) < 1e-6 for r in source for i in range(12))
        for case, rows in cases.items():
            if not rows[0]['t_s'] <= lo < hi <= rows[-1]['t_s']: raise ValueError('window coverage '+name)
            m = summarize(rows, lo, hi, real); m['normal_motion_valid'] = m.get('upright', True)
            if not m['normal_motion_valid']:
                m['invalid_motion_reason'] = 'Not upright: metrics are raw failed-motion diagnostics; wrapped Euler std/spectrum must not be compared as walking stability.'
            m['joints_by_kind'] = extra(rows, lo, hi, real, pticks)
            if case == 'real':
                m['feedback_age_max_ms'] = max(r[f'age_{i:02}'] for r in source for i in range(12))
                m['temperature_range_C'] = [min(r[f'temp_{i:02}'] for r in source for i in range(12)), max(r[f'temp_{i:02}'] for r in source for i in range(12))]
            result['windows'][name][case] = m
            all_cases[(name, case)] = rows
    for direction in ('backward_core', 'backward_extended', 'forward_core'):
        early = result['windows']['early_'+direction]; late = result['windows']['late_'+direction]
        result['comparisons'][direction] = {}
        for case in set(early) & set(late):
            if not early[case]['normal_motion_valid'] or not late[case]['normal_motion_valid']:
                result['comparisons'][direction][case] = {'normal_motion_comparison_valid': False,
                    'reason': 'Early and/or late replay lost upright walking. No improvement percentage is meaningful.'}
                continue
            result['comparisons'][direction][case] = {key: {'early': early[case][key], 'late': late[case][key],
                'late_over_early': late[case][key]/early[case][key], 'reduction_percent': (1-late[case][key]/early[case][key])*100}
                for key in ('gyro_x_rms_rad_s', 'roll_std_deg', 'roll_half_p95_p05_deg', 'hip_dq_rms_rad_s')}
    for lo in range(20, 165, 5):
        rr = [r for r in real if lo <= r['t_s'] < lo+5 and r['mode'] == 3]
        result['timeline_5s'].append({'lo': lo, 'hi': lo+5, 'roll_std_deg': st.pstdev(r['roll_deg'] for r in rr),
            'gyro_x_rms': rms([r['gyro_x'] for r in rr]), 'shank_feedback_rms': rms([r[f'tau_{i:02}'] for r in rr for i in GROUPS['shank']])})
    a.output_prefix.with_suffix('.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with a.output_prefix.with_suffix('.csv').open('w') as f:
        w = csv.writer(f); w.writerow(['window', 'case', 'lo_s', 'hi_s', 'gyro_x_rms_rad_s', 'roll_std_deg', 'roll_half_p95_p05_deg', 'gyro_peak_hz', 'gyro_4_10_rms_rad_s', 'hip_dq_rms_rad_s'])
        for name, cases in result['windows'].items():
            for case, m in cases.items():
                if m['normal_motion_valid']: w.writerow([name, case, *WINDOWS[name], m['gyro_x_rms_rad_s'], m['roll_std_deg'], m['roll_half_p95_p05_deg'], m['gyro_spectrum']['peak_1_20_hz'], m['gyro_spectrum']['band_4_10_rms'], m['hip_dq_rms_rad_s']])
                else: w.writerow([name, case, *WINDOWS[name]]+['FAILED_NOT_WALKING']*6)
    print(json.dumps(result['comparisons'], indent=2)); print('replay checks', {k: {x: v[x] for x in ('targets', 'published', 'max_target_error', 'max_publish_lag_s')} for k, v in result['replays'].items()})
    if a.plot: plot(a.output_prefix, real, result, all_cases)


def plot(prefix, real, result, cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def save(fig, suffix):
        fig.savefig(str(prefix)+'_'+suffix+'.png', dpi=160, bbox_inches='tight'); plt.close(fig)
    rr = [r for r in real if 20 <= r['t_s'] <= 168]
    fig, ax = plt.subplots(5, 1, figsize=(16, 13), sharex=True, layout='constrained')
    for axis, key in zip(ax[:2], ('roll_deg', 'gyro_x')): axis.plot([r['t_s'] for r in rr], [r[key] for r in rr], lw=.55); axis.set_ylabel(key)
    for key in ('cmd_vx', 'cmd_vy', 'cmd_wz'): ax[2].plot([r['t_s'] for r in rr], [r[key] for r in rr], label=key, lw=.8)
    ax[2].legend(ncol=3); ax[2].set_ylabel('Commands')
    bins = result['timeline_5s']
    for axis, key in zip(ax[3:], ('gyro_x_rms', 'shank_feedback_rms')):
        axis.plot([(r['lo']+r['hi'])/2 for r in bins], [r[key] for r in bins], '-o', ms=3); axis.set_ylabel(key)
    for axis in ax:
        for name in ('early_backward_core', 'late_backward_core', 'early_forward_core', 'late_forward_core'):
            lo, hi = WINDOWS[name]; axis.axvspan(lo, hi, alpha=.12, color='#ee7733' if name.startswith('late') else '#0077bb')
        axis.grid(alpha=.2)
    ax[-1].set_xlabel('Real log time (s)'); fig.suptitle('Latest real trial: motion and commands | exact rope-on/off boundary unknown\nBlue/Orange shading: early/late matched-command samples, not force measurements')
    save(fig, 'timeline')
    colors = {('early_backward_core','real'): '#0077bb', ('late_backward_core','real'): '#ee7733',
        ('early_backward_core','gazebo'): '#228833', ('late_backward_core','gazebo'): '#aa3377'}
    fig, ax = plt.subplots(3, 2, figsize=(15, 11), layout='constrained')
    for col, case in enumerate(('real', 'gazebo')):
        for name in ('early_backward_core', 'late_backward_core'):
            if (name, case) not in cases: continue
            if not result['windows'][name][case]['normal_motion_valid']:
                for i in range(3): ax[i,col].text(.03,.88,'Early replay fell before this window; no walking metrics',transform=ax[i,col].transAxes,fontsize=8)
                continue
            lo, hi = WINDOWS[name]; rows = [r for r in cases[name,case] if lo <= r['t_s'] < hi]
            for i, key in enumerate(('roll_deg', 'gyro_x', 'dq_00')):
                ax[i,col].plot([r['t_s']-lo for r in rows], [r[key] for r in rows], label=name.split('_')[0], color=colors[name,case], lw=.8)
                ax[i,col].set(ylabel=key, xlabel='Within-window time (s)', title=case); ax[i,col].grid(alpha=.2)
        if ax[0,col].lines: ax[0,col].legend()
    fig.suptitle('Matched backward command -0.5m/s | 2.5s windows, independent gait phase\nGazebo: original targets, full gravity, no rope, controller joint feedback')
    save(fig, 'response')
    fig, ax = plt.subplots(1, 2, figsize=(15, 5), layout='constrained')
    for pair, color in colors.items():
        if pair not in cases: continue
        name, case = pair; lo, hi = WINDOWS[name]
        if not result['windows'][name][case]['normal_motion_valid']: continue
        for axis, key in zip(ax, ('gyro_x', 'roll_deg')):
            _, bins = spectrum(wave(cases[pair], key, lo, hi)); axis.plot([f for f,v in bins], [v for f,v in bins], label=case+' '+name.split('_')[0], color=color)
            axis.set(xlabel='Hz', ylabel='Sinusoid amplitude', title=key, xlim=(1,15)); axis.grid(alpha=.2)
    ax[0].legend(fontsize=8); fig.suptitle('Spectrum | Hann + linear detrend; 0.4Hz resolution; peaks do not identify cause')
    save(fig, 'spectra')
    forward_colors = {('early_forward_core','real'): '#0077bb', ('late_forward_core','real'): '#ee7733',
        ('early_forward_core','gazebo'): '#228833', ('late_forward_core','gazebo'): '#aa3377'}
    fig, ax = plt.subplots(3, 2, figsize=(15, 11), layout='constrained')
    for col, case in enumerate(('real', 'gazebo')):
        for name in ('early_forward_core', 'late_forward_core'):
            if (name, case) not in cases: continue
            if not result['windows'][name][case]['normal_motion_valid']:
                for axis in ax[:,col]: axis.text(.03,.88,name+' failed; no walking metrics',transform=axis.transAxes,fontsize=8)
                continue
            lo, hi = WINDOWS[name]; rows = [r for r in cases[name,case] if lo <= r['t_s'] < hi]
            for i, key in enumerate(('roll_deg', 'gyro_x', 'dq_00')):
                ax[i,col].plot([r['t_s']-lo for r in rows], [r[key] for r in rows], label=name.split('_')[0], color=forward_colors[name,case], lw=.8)
                ax[i,col].set(ylabel=key, xlabel='Within-window time (s)', title=case); ax[i,col].grid(alpha=.2)
        if ax[0,col].lines: ax[0,col].legend()
    fig.suptitle('Matched forward command +0.5m/s | independent phase; full-gravity Gazebo has no rope')
    save(fig, 'forward_response')
    fig, ax = plt.subplots(1, 2, figsize=(15, 5), layout='constrained')
    for pair, color in forward_colors.items():
        if pair not in cases: continue
        name, case = pair; lo, hi = WINDOWS[name]
        if not result['windows'][name][case]['normal_motion_valid']: continue
        for axis, key in zip(ax, ('gyro_x', 'roll_deg')):
            _, bins = spectrum(wave(cases[pair], key, lo, hi)); axis.plot([f for f,v in bins], [v for f,v in bins], label=case+' '+name.split('_')[0], color=color)
            axis.set(xlabel='Hz', ylabel='Sinusoid amplitude', title=key, xlim=(1,15)); axis.grid(alpha=.2)
    if ax[0].lines: ax[0].legend(fontsize=8)
    fig.suptitle('Forward spectra | Hann + linear detrend; 0.4Hz resolution')
    save(fig, 'forward_spectra')
    fig, ax = plt.subplots(4, 2, figsize=(15, 11), layout='constrained')
    for row, i in enumerate(GROUPS['hip']):
        for name in ('early_backward_core', 'late_backward_core'):
            lo, hi = WINDOWS[name]; rr = [r for r in real if lo <= r['t_s'] < hi]
            color = colors[name,'real']; label = name.split('_')[0]
            ax[row,0].plot([r['t_s']-lo for r in rr], [r[f'q_des_{i:02}'] for r in rr], label=label+' target', color=color)
            ax[row,0].plot([r['t_s']-lo for r in rr], [r[f'q_{i:02}'] for r in rr], color=color, ls=':', lw=.8)
            ax[row,1].plot([r['t_s']-lo for r in rr], [r[f'tau_{i:02}'] for r in rr], label=label+' feedback', color=color, lw=.8)
        ax[row,0].set_ylabel(['FL','FR','RL','RR'][row]+' hip rad'); ax[row,1].set_ylabel('Estimated Nm')
        for axis in ax[row]: axis.grid(alpha=.2); axis.set_xlabel('Within-window time (s)')
    ax[0,0].legend(); ax[0,1].legend(); fig.suptitle('Real hip targets / dotted positions and estimated feedback torque | phases independent')
    save(fig, 'hips')
    fig, ax = plt.subplots(3, 1, figsize=(15, 10), sharex=True, layout='constrained')
    for name in ('early_backward_core', 'late_backward_core', 'early_forward_core', 'late_forward_core'):
        if (name, 'gazebo') not in cases: continue
        rows = cases[name, 'gazebo']; start = rows[0]['t_s']
        for axis, key in zip(ax, ('tilt_deg', 'z', 'gyro_x')):
            axis.plot([r['t_s']-start for r in rows], [r[key] for r in rows], label=name.removesuffix('_core'), lw=.75)
            axis.set_ylabel(key); axis.grid(alpha=.2)
    ax[0].axhline(60, color='gray', ls='--', lw=.7); ax[1].axhline(.2, color='gray', ls='--', lw=.7)
    if ax[0].lines: ax[0].legend(fontsize=8)
    ax[-1].set_xlabel('Time from each independent replay start (s)'); fig.suptitle('Full replay motion status | classify falls before comparing vibration amplitudes')
    save(fig, 'replay_status')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--output-prefix', type=Path, required=True)
    p.add_argument('--replay', action='append', default=[], help='early_backward=directory or late_backward=directory; forward labels also supported')
    p.add_argument('--sensitivity', action='append', default=[], help='label=directory, secondary initial-height runs retained separately')
    p.add_argument('--plot', action='store_true'); main(p.parse_args())
