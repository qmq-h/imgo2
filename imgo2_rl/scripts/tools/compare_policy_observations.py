#!/usr/bin/env python3
"""Compare actual normal/real policy inputs and reconstructed replay state.

Core analysis/CSV/JSON use only the standard library. Matplotlib is optional.
Normal and real motion windows are independent: no phase fitting or causal claim.
"""
import argparse
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st

from audit_paired_obs_replay import quantile, spectrum

LEGS = ('FL', 'FR', 'RL', 'RR')
JOINTS = ('hip', 'thigh', 'shank')
LABELS = ['gyro '+k for k in 'XYZ']+['gravity '+k for k in 'XYZ']+['command '+k for k in ('vx', 'vy', 'wz')]
LABELS += [f'{leg} {joint} q-relative' for leg in LEGS for joint in JOINTS]
LABELS += [f'{leg} {joint} dq' for leg in LEGS for joint in JOINTS]
LABELS += [f'{leg} {joint} previous action' for leg in LEGS for joint in JOINTS]
GROUPS = {'gyro_xyz': list(range(3)), 'gravity_xyz': list(range(3, 6)),
          'hip_q': [9, 12, 15, 18], 'thigh_q': [10, 13, 16, 19], 'shank_q': [11, 14, 17, 20],
          'hip_dq': [21, 24, 27, 30], 'thigh_dq': [22, 25, 28, 31], 'shank_dq': [23, 26, 29, 32],
          'hip_previous_action': [33, 36, 39, 42]}
COLORS = {'normal': '#228833', 'real': '#0077bb', 'replay_state': '#ee7733'}
NAMES = {'normal': 'Normal Gazebo: actual actor input', 'real': 'Real: actual actor input',
         'replay_state': 'Obs replay: reconstructed Gazebo state'}


def read(path, ignore=()):
    with path.open() as f:
        return [{k: float(v) for k, v in r.items() if k not in ignore} for r in csv.DictReader(f)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interpolate(rows, times, t, key):
    if not times[0] <= t <= times[-1]:
        raise ValueError(f'time outside coverage: {t}')
    j = min(len(times)-1, max(1, bisect.bisect_right(times, t)))
    dt = times[j]-times[j-1]
    if dt <= 0 or dt > .06:
        raise ValueError(f'nonmonotonic time or gap over 60ms: {dt}')
    f = (t-times[j-1])/dt
    return rows[j-1][key]+f*(rows[j][key]-rows[j-1][key])


def vectors(rows, lo, hi):
    times = [r['t'] for r in rows]
    n = round((hi-lo)*100)
    return [[interpolate(rows, times, lo+j*.01, f'o{i}') for j in range(n)] for i in range(45)]


def description(values, native):
    # Preserve amplitude on the original policy samples. Interpolation is only
    # for a common spectral grid and must not silently attenuate reported RMS.
    m = st.mean(native); deviation = st.pstdev(native)
    spec, bins = spectrum(values)
    if deviation < 1e-10:
        spec['peak_1_20_hz'] = None
    return {'samples':len(native),'mean': m, 'rms': math.sqrt(st.mean(v*v for v in native)), 'std': deviation,
            'peak_to_peak': max(native)-min(native),
            'half_p95_p05': (quantile(native, .95)-quantile(native, .05))/2,
            **spec}, bins


def selected(rows, lo, hi):
    return [r for r in rows if lo <= r['t'] < hi]


def aggregate(channel_stats, indices, scale=1):
    return {key: math.sqrt(st.mean(channel_stats[i][key]**2 for i in indices))/scale
            for key in ('rms', 'std', 'band_4_10_rms')}


def ratio(a, b):
    return a/b if b > 1e-9 else None


def main(a):
    a.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((a.real/'meta.json').read_text())['config']
    real_raw = read(a.real/'policy.csv')
    normal_raw = read(a.normal/'policy.csv', ('stage',))
    odom = read(a.normal/'odom.csv')
    rt = [r['host_s'] for r in odom]
    real, normal = [], []
    for r in real_raw:
        real.append({'t': r['input_captured_ms']*.001, **{f'o{i}': r[f'obs_{i:02}'] for i in range(45)}})
    for r in normal_raw:
        # The stored sim_s is a cached ~10Hz clock; host->odom interpolation gives
        # a monotonic estimate for plotting actual ~50Hz actor-input samples.
        t = interpolate(odom, rt, r['host_s'], 'sim_s')
        normal.append({'t': t, **{f'o{i}': r[f'obs_{i:02}'] for i in range(45)}})
    meta = json.loads((a.replay/'metadata.json').read_text())
    physical = read(a.replay/'replay.csv')
    pt = [r['t_s']+meta['source_start_s'] for r in physical]
    replay = []
    for r in real:
        t = r['t']
        if not pt[0] <= t <= pt[-1]:
            continue
        q = [interpolate(physical, pt, t, 'quat_'+k) for k in 'wxyz']
        norm = math.sqrt(sum(x*x for x in q)); w, x, y, z = [v/norm for v in q]
        obs = [interpolate(physical, pt, t, 'gyro_'+k)*cfg['obs_scales']['ang_vel'] for k in 'xyz']
        obs += [2*(w*y-x*z), -2*(w*x+y*z), -1+2*(x*x+y*y)]
        obs += [r[f'o{i}'] for i in range(6, 9)]
        obs += [(interpolate(physical, pt, t, f'q_{h:02}')-cfg['default_pose'][i])*cfg['obs_scales']['dof_pos'] for i, h in enumerate(cfg['policy_to_hardware'])]
        obs += [interpolate(physical, pt, t, f'dq_{h:02}')*cfg['obs_scales']['dof_vel'] for h in cfg['policy_to_hardware']]
        # There is no independent simulated actor recurrence in the replay.
        # Keep the actual driving context and label it as copied, not measured.
        obs += [r[f'o{i}'] for i in range(33, 45)]
        clip = cfg['obs_scales']['clip_obs']
        replay.append({'t': t, **{f'o{i}': max(-clip, min(clip, v)) for i, v in enumerate(obs)}})
    sources = {'normal': normal, 'real': real, 'replay_state': replay}
    windows = {'long_forward_mixed': {'normal': (12.9, 18.8), 'real': (86.5, 92.4), 'replay_state': (86.5, 92.4)},
               'pure_straight_short': {'normal': (12.9, 14.04), 'real': (86.14, 87.28), 'replay_state': (86.14, 87.28)}}
    report, spectra, traces = {}, {}, {}
    for name, case_windows in windows.items():
        stats = {}; spectra[name] = {}; traces[name] = {}
        for case, (lo, hi) in case_windows.items():
            vs = vectors(sources[case], lo, hi)
            actual=selected(sources[case],lo,hi)
            ds = [description(v,[r[f'o{i}'] for r in actual]) for i,v in enumerate(vs)]
            stats[case] = [v[0] for v in ds]; spectra[name][case] = [v[1] for v in ds]; traces[name][case] = vs
        groups = {}
        for label, indices in GROUPS.items():
            scale = cfg['obs_scales']['dof_vel'] if label.endswith('_dq') else (cfg['obs_scales']['ang_vel'] if label == 'gyro_xyz' else 1)
            groups[label] = {case: aggregate(stats[case], indices, scale) for case in sources}
            groups[label]['real_to_normal_std'] = ratio(groups[label]['real']['std'], groups[label]['normal']['std'])
        report[name] = {'windows_s': case_windows, 'channels': stats, 'groups_physical_units': groups,
                        'scaled_gyro_x_physical': {case: {k: stats[case][0][k]/cfg['obs_scales']['ang_vel'] for k in ('rms', 'std', 'band_4_10_rms')} for case in sources}}
    errors = [abs(b[f'obs_{i+33:02}']-r[f'act_{i:02}']) for r, b in zip(real_raw, real_raw[1:]) if r['gen'] == b['gen'] for i in range(12)]
    field_consistency = {'real_previous_action_vs_previous_logged_action_max_abs_error': max(errors),
                         'replay_context_indices_copied': list(range(6, 9))+list(range(33, 45))}
    # Verify the normal trace is in the same scale/order using its saved snapshot.
    normal_errors = []
    for r in normal_raw:
        expected = [r['gyro_'+k]*cfg['obs_scales']['ang_vel'] for k in 'xyz']
        w, x, y, z = [r['quat_'+k] for k in 'wxyz']
        expected += [2*(w*y-x*z), -2*(w*x+y*z), -1+2*(x*x+y*y)]
        expected += [r['cmd_'+k]*cfg['commands_scale'][i] for i, k in enumerate(('vx', 'vy', 'wz'))]
        expected += [(r[f'q_{i:02}']-cfg['default_pose'][i])*cfg['obs_scales']['dof_pos'] for i in range(12)]
        expected += [r[f'dq_{i:02}']*cfg['obs_scales']['dof_vel'] for i in range(12)]
        normal_errors += [abs(v-r[f'obs_{i:02}']) for i, v in enumerate(expected)]
    field_consistency['normal_first_33_obs_vs_saved_state_max_abs_error'] = max(normal_errors)
    velocity_consistency={}
    for case in ('normal','real'):
        lo,hi=windows['long_forward_mixed'][case];rr=selected(sources[case],lo,hi);joints=[]
        for i in range(12):
            qdiff=[];dqmean=[]
            for before,after in zip(rr,rr[1:]):
                dt=after['t']-before['t']
                qdiff.append((after[f'o{i+9}']-before[f'o{i+9}'])/(dt*cfg['obs_scales']['dof_pos']))
                dqmean.append((after[f'o{i+21}']+before[f'o{i+21}'])/(2*cfg['obs_scales']['dof_vel']))
            ma,mb=st.mean(qdiff),st.mean(dqmean)
            den=math.sqrt(sum((v-ma)**2 for v in qdiff)*sum((v-mb)**2 for v in dqmean))
            joints.append({'joint':f'{LEGS[i//3]} {JOINTS[i%3]}',
                'correlation':sum((u-ma)*(v-mb) for u,v in zip(qdiff,dqmean))/den,
                'qdiff_per_mean_dq_fit':sum(u*v for u,v in zip(qdiff,dqmean))/sum(v*v for v in dqmean),
                'rmse_rad_s':math.sqrt(st.mean((u-v)**2 for u,v in zip(qdiff,dqmean)))})
        velocity_consistency[case]=joints
    real_window = [r for r in real_raw if 86.5 <= r['input_captured_ms']*.001 < 92.4]
    timing = {}
    for label, v in {'input_to_generated_ms': [r['result_generated_ms']-r['input_captured_ms'] for r in real_window],
                     'input_to_log_consumption_ms': [r['t_mono_ms']-r['input_captured_ms'] for r in real_window],
                     'generated_to_log_consumption_ms': [r['t_mono_ms']-r['result_generated_ms'] for r in real_window],
                     'stored_result_age_ms': [r['result_age_ms'] for r in real_window]}.items():
        timing[label] = {'min': min(v), 'median': st.median(v), 'p95': quantile(v, .95), 'max': max(v)}
    # Cross-source pointwise errors are meaningful only as a descriptive replay
    # check on the common input-capture host axis, never on independent normal runs.
    forward = traces['long_forward_mixed']
    paired_difference = {}
    for i in range(33):
        diff = [b-c for b, c in zip(forward['replay_state'][i], forward['real'][i])]
        paired_difference[str(i)] = {'mean_replay_minus_real': st.mean(diff), 'rmse': math.sqrt(st.mean(d*d for d in diff))}
    files = [a.real/'policy.csv', a.real/'meta.json', a.normal/'policy.csv', a.normal/'odom.csv',
             a.normal/'policy_used.pt', a.replay/'replay.csv', a.replay/'metadata.json']
    result = {'obs_labels': LABELS, 'windows': report, 'checks': field_consistency, 'timing_context': timing,
              'q_dq_internal_consistency':velocity_consistency,
              'replay_minus_real_descriptive_errors': paired_difference,
              'method': {'real_time': 'actual input_captured_ms; not result consumption time',
                         'normal_time': 'policy trace host_s mapped to odom exact sim header; cached sim_s not differentiated',
                         'replay_time': 'physical state at same real input-capture host time relative to replay start',
                         'spectra': 'all three sampled at actual policy input cadence, then 100Hz linear interpolation; detrend+Hann 1-20Hz peaks; rectangular demeaned 4-10Hz RMS',
                         'sampling_limit': 'interpolation does not recover bandwidth above ~25Hz policy Nyquist; variable intervals and no antialias attribution',
                         'amplitudes': 'mean/std/RMS/peak-to-peak/percentiles from original samples; interpolation used only for spectra and descriptive replay differences',
                         'groups': 'pooled per-channel std/RMS/band RMS; gyro and dq scaled back to physical units'},
              'limits': ['Normal vs real: independent histories and gait phase; no pointwise RMSE or phase-delay estimate is reported.',
                         'Long real window has vx=.5 but nonzero lateral/yaw commands; pure straight window is only 1.14s with ~.877Hz frequency bins.',
                         'Replay state observations are reconstructed diagnostics, not new actual actor inputs; command and last-action context are copied.',
                         'Shared spectral peaks show coupling and response, not the initial cause, sensor noise, or signed feedback amplification.',
                         'Input capture/trace host times are not individual physical source sampling or motor application times.',
                         'Runtime deployed weight hash and updated capture-source archive remain unavailable; earlier whole-log action agreement supports candidate.'],
              'sha256': {str(p): digest(p) for p in files}}
    a.output_prefix.with_suffix('.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with a.output_prefix.with_suffix('.csv').open('w') as f:
        fields = ['window', 'case', 'index', 'label', 'samples', 'mean', 'rms', 'std', 'peak_to_peak', 'half_p95_p05', 'peak_1_20_hz', 'peak_sinusoid_amplitude', 'resolution_hz', 'band_4_10_rms']
        w = csv.DictWriter(f, fields); w.writeheader()
        for name, data in report.items():
            for case, stats in data['channels'].items():
                for i, s in enumerate(stats):
                    w.writerow({'window': name, 'case': case, 'index': i, 'label': LABELS[i], **s})
    with Path(str(a.output_prefix)+'_traces.csv').open('w') as f:
        w = csv.writer(f); w.writerow(['case', 'source_s', 'long_window_relative_s']+[f'obs_{i:02}' for i in range(45)])
        for case, rows in sources.items():
            lo, hi = windows['long_forward_mixed'][case]
            for r in selected(rows, lo, hi): w.writerow([case, r['t'], r['t']-lo]+[r[f'o{i}'] for i in range(45)])
    for label, m in report['long_forward_mixed']['groups_physical_units'].items(): print(label, json.dumps(m))
    print('checks', field_consistency); print('timing', timing)
    if a.plot:
        plot(a.output_prefix, sources, windows, report, spectra, traces, real_raw, normal_raw)


def plot(prefix, sources, windows, report, spectra, traces, real_raw, normal_raw):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10})
    name = 'long_forward_mixed'; case_windows = windows[name]
    def line(axis, i, seconds, spectral=False):
        for case, rows in sources.items():
            copied = case == 'replay_state' and (6 <= i < 9 or i >= 33)
            if copied: continue
            if spectral:
                bins = spectra[name][case][i]
                axis.plot([p[0] for p in bins], [p[1] for p in bins], color=COLORS[case], label=NAMES[case], lw=1)
            else:
                lo, hi = case_windows[case]; rr = selected(rows, lo, min(hi, lo+seconds))
                axis.plot([r['t']-lo for r in rr], [r[f'o{i}'] for r in rr], color=COLORS[case], label=NAMES[case], lw=.9)
            s = report[name]['channels'][case][i]
        axis.set_title(f'obs[{i:02}] {LABELS[i]}'); axis.grid(alpha=.23)
        axis.set_xlabel('Frequency (Hz)' if spectral else 'Within-window time (s); independent phase')
        axis.set_ylabel('Scaled obs peak amplitude' if spectral else 'Scaled actor observation')
        if spectral: axis.set_xlim(0, 20)
    def save(fig, suffix):
        fig.savefig(Path(str(prefix)+'_'+suffix+'.png'), dpi=160,bbox_inches='tight'); plt.close(fig)
    fig, ax = plt.subplots(3, 3, figsize=(17, 10), layout='constrained')
    for i, axis in enumerate(ax.flat): line(axis, i, 5.9)
    handles, labels = ax[0, 0].get_legend_handles_labels(); fig.legend(handles, labels, loc='lower center',bbox_to_anchor=(.5,-.055), ncol=3)
    fig.suptitle('0.5 m/s forward: actual normal/real policy inputs vs replay-state reconstruction\nReal 86.5-92.4s includes lateral/yaw commands; Normal 12.9-18.8s. Replay commands are copied context.')
    save(fig, 'body')
    for start, suffix, title in [(9, 'q', 'Relative joint positions'), (21, 'dq', 'Joint velocity observations'), (33, 'last_action', 'Previous action observations (replay context is copied)')]:
        fig, ax = plt.subplots(4, 3, figsize=(17, 12), layout='constrained')
        for i, axis in enumerate(ax.flat): line(axis, start+i, 2)
        handles, labels = ax[0, 0].get_legend_handles_labels(); fig.legend(handles, labels, loc='lower center',bbox_to_anchor=(.5,-.055), ncol=3)
        fig.suptitle(title+' | first 2s detail of independent 5.9s forward windows\nRows FL/FR/RL/RR; columns hip/thigh/shank. q scale=1, dq scale=.05; no phase fitting.')
        save(fig, suffix)
    indices = [0, 1, 2, 4, 9, 21, 33, 10, 22]
    fig, ax = plt.subplots(3, 3, figsize=(17, 10), layout='constrained')
    for axis, i in zip(ax.flat, indices): line(axis, i, 5.9, spectral=True)
    handles, labels = ax[0, 0].get_legend_handles_labels(); fig.legend(handles, labels, loc='lower center',bbox_to_anchor=(.5,-.055), ncol=3)
    fig.suptitle('Observation spectra: detrended Hann-window amplitudes, same 5.9s duration\nPeaks include gait/contact motion; frequency grid 0.1695Hz. Replay last action is copied and omitted.')
    save(fig, 'spectra')
    # Show every channel's amplitude difference without mixing unlike units.
    fig, ax = plt.subplots(1, 2, figsize=(15, 13), layout='constrained')
    inds = list(range(45)); base = report[name]['channels']['normal']; real = report[name]['channels']['real']; replay = report[name]['channels']['replay_state']
    for axis, key, title in [(ax[0], 'std', 'Std ratio to normal'), (ax[1], 'band_4_10_rms', '4-10Hz RMS ratio to normal')]:
        for values, color, shift, label in [(real, COLORS['real'], -.16, 'Real / normal'), (replay, COLORS['replay_state'], .16, 'Replay state / normal')]:
            ids = [i for i in inds if not 6 <= i < 9 and not (values is replay and i >= 33) and base[i][key] > 1e-9]
            axis.barh([i+shift for i in ids], [values[i][key]/base[i][key] for i in ids], height=.3, color=color, label=label)
        axis.set_yticks(inds, [f'{i:02} {LABELS[i]}' for i in inds]); axis.invert_yaxis(); axis.set_xscale('log'); axis.axvline(1, color='black', ls='--'); axis.grid(axis='x', alpha=.25); axis.set_title(title); axis.legend()
    fig.suptitle('All 45 observation channels | ratios are descriptive, not causal importance\nConstant command channels excluded; replay context not counted as independent measurement.')
    save(fig, 'amplitude_ratios')
    # Real start-up on its actual capture axis. Output is shown at generated time.
    rr = [r for r in real_raw if 84.8 <= r['input_captured_ms']*.001 <= 87.5]
    fig, ax = plt.subplots(3, 2, figsize=(15, 10), layout='constrained')
    t = [r['input_captured_ms']*.001 for r in rr]
    ax[0, 0].plot(t, [r['obs_06'] for r in rr], label='vx'); ax[0, 0].plot(t, [r['obs_07'] for r in rr], label='vy'); ax[0, 0].plot(t, [r['obs_08'] for r in rr], label='wz')
    for case in ('real', 'replay_state'):
        data = selected(sources[case], 84.8, 87.5)
        ax[0, 1].plot([r['t'] for r in data], [r['o0']/.25 for r in data], label=NAMES[case], color=COLORS[case])
    for j, leg in enumerate(LEGS):
        i = j*3
        ax[1, 0].plot(t, [r[f'obs_{9+i:02}'] for r in rr], label=leg)
        ax[1, 1].plot(t, [r[f'obs_{21+i:02}']/.05 for r in rr], label=leg)
        ax[2, 0].plot([r['result_generated_ms']*.001 for r in rr], [r[f'act_{i:02}']*.125 for r in rr], label=leg)
    ax[2, 1].plot(t, [r['result_generated_ms']-r['input_captured_ms'] for r in rr], label='capture -> result')
    ax[2, 1].plot(t, [r['t_mono_ms']-r['input_captured_ms'] for r in rr], label='capture -> log consumption')
    titles = [('Actual input commands', 'm/s or rad/s'), ('Roll rate', 'rad/s'), ('Actual hip q-relative inputs', 'rad'), ('Actual hip dq inputs', 'rad/s'), ('Hip target offsets from current action (at result generation)', 'rad'), ('Host timestamp intervals (not wire delay)', 'ms')]
    for axis, (title, unit) in zip(ax.flat, titles):
        axis.set(title=title, xlabel='Real source time (s)', ylabel=unit); axis.grid(alpha=.25); axis.legend(fontsize=8)
    fig.suptitle('Start-up detail | no time shifting or inferred motor application time\nCorrelated input/output changes cannot identify which physical disturbance started the loop.')
    save(fig, 'onset')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('real', 'normal', 'replay', 'output-prefix'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--plot', action='store_true')
    main(p.parse_args())
