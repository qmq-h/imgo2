#!/usr/bin/env python3
"""Stdlib statistics for paired real-obs -> actor -> Gazebo replay; optional plot.

Frequency and band calculations use 100 Hz interpolation over explicit windows.
No lag fitting, sensor correction, hardware access, or policy modification.
"""
import argparse
import bisect
import cmath
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st


def load(path):
    with path.open() as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rms(v):
    return math.sqrt(st.mean(x*x for x in v))


def quantile(values, p):
    v = sorted(values); j = (len(v)-1)*p; i = int(j)
    return v[i]+(v[min(i+1, len(v)-1)]-v[i])*(j-i)


def attitude(r):
    q = [r['quat_'+k] for k in 'wxyz']; norm = math.sqrt(sum(v*v for v in q))
    w, x, y, z = [v/norm for v in q]
    r['roll_deg'] = math.degrees(math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)))
    r['pitch_deg'] = math.degrees(math.asin(max(-1, min(1, 2*(w*y-z*x)))))
    r['yaw_rad'] = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    r['tilt_deg'] = math.degrees(math.acos(max(-1, min(1, 1-2*(x*x+y*y)))))
    if 'vx' in r:
        r['body_vx'] = (1-2*(y*y+z*z))*r['vx']+2*(x*y+w*z)*r['vy']+2*(x*z-w*y)*r['vz']
        r['body_vy'] = 2*(x*y-w*z)*r['vx']+(1-2*(x*x+z*z))*r['vy']+2*(y*z+w*x)*r['vz']


def wave(rows, key, lo, hi):
    times = [r['t_s'] for r in rows]
    if lo < times[0] or hi-.01 > times[-1]:
        raise ValueError('requested window not covered')
    out = []
    for j in range(round((hi-lo)*100)):
        t = lo+j*.01; i = min(len(rows)-1, max(1, bisect.bisect_right(times, t)))
        a, b = rows[i-1], rows[i]; dt = times[i]-times[i-1]
        if dt > .05:
            raise ValueError('data gap over 50ms in analysis window')
        f = (t-times[i-1])/dt
        out.append(a[key]+f*(b[key]-a[key]))
    return out


def spectrum(values):
    n = len(values); m = st.mean(values)
    hann = [.5-.5*math.cos(2*math.pi*j/(n-1)) for j in range(n)]
    # Linear detrending reduces slow-drift leakage in the descriptive spectrum.
    jm = (n-1)/2
    slope = sum((j-jm)*(v-m) for j, v in enumerate(values))/sum((j-jm)**2 for j in range(n))
    out = []
    for k in range(1, min(n//2, math.floor(20*n/100))+1):
        z = sum((v-m-slope*(j-jm))*hann[j]*cmath.exp(-2j*math.pi*k*j/n) for j, v in enumerate(values))
        out.append((k*100/n, 2*abs(z)/sum(hann)))
    peak = max((v for v in out if v[0] >= 1), key=lambda a: a[1])
    bins = range(math.ceil(4*n/100), math.floor(10*n/100)+1)
    energy = sum(abs(sum((v-m)*cmath.exp(-2j*math.pi*k*j/n) for j, v in enumerate(values)))**2 for k in bins)
    return {'peak_1_20_hz': peak[0], 'peak_sinusoid_amplitude': peak[1],
            'resolution_hz': 100/n, 'band_4_10_rms': math.sqrt(2*energy)/n}, out


def summarize(rows, lo, hi, commands=None):
    selected = [r for r in rows if lo <= r['t_s'] < hi]
    if not selected:
        raise ValueError('empty window')
    roll = [r['roll_deg'] for r in selected]; gyro = [r['gyro_x'] for r in selected]
    sg, _ = spectrum(wave(rows, 'gyro_x', lo, hi)); sr, _ = spectrum(wave(rows, 'roll_deg', lo, hi))
    result = {'window_s': [lo, hi], 'samples': len(selected), 'gyro_x_rms_rad_s': rms(gyro),
              'gyro_x_peak_to_peak_rad_s': max(gyro)-min(gyro),
              'roll_mean_deg': st.mean(roll), 'roll_std_deg': st.pstdev(roll),
              'roll_peak_to_peak_deg': max(roll)-min(roll),
              'roll_p95_minus_p05_deg': quantile(roll, .95)-quantile(roll, .05),
              'roll_half_p95_p05_deg': (quantile(roll, .95)-quantile(roll, .05))/2,
              'max_tilt_deg': max(r['tilt_deg'] for r in selected),
              'gyro_spectrum': sg, 'roll_spectrum': sr}
    if 'dq_00' in selected[0]:
        result['hip_dq_rms_rad_s'] = rms([r[f'dq_{i:02}'] for r in selected for i in (0, 3, 6, 9)])
    if 'z' in selected[0]:
        first, last = selected[0], selected[-1]; yaw = first['yaw_rad']
        dx, dy = last['x']-first['x'], last['y']-first['y']
        result.update(mean_body_vx_mps=st.mean(r['body_vx'] for r in selected),
                      mean_body_vy_mps=st.mean(r['body_vy'] for r in selected),
                      body_vx_std_mps=st.pstdev(r['body_vx'] for r in selected),
                      displacement_along_start_heading_m=dx*math.cos(yaw)+dy*math.sin(yaw),
                      displacement_lateral_start_heading_m=-dx*math.sin(yaw)+dy*math.cos(yaw),
                      yaw_change_deg=math.degrees(sum(math.atan2(math.sin(b['yaw_rad']-a['yaw_rad']), math.cos(b['yaw_rad']-a['yaw_rad'])) for a, b in zip(selected, selected[1:]))),
                      mean_height_m=st.mean(r['z'] for r in selected), min_height_m=min(r['z'] for r in selected))
        result['upright'] = result['max_tilt_deg'] < 60 and result['min_height_m'] > .20
    if commands:
        c = [r for r in commands if lo <= r['t_s'] < hi]
        result['commands'] = {k: {'min': min(r[k] for r in c), 'max': max(r[k] for r in c), 'mean': st.mean(r[k] for r in c)} for k in ('cmd_vx', 'cmd_vy', 'cmd_wz')}
    return result


def main(a):
    meta = json.loads((a.run/'metadata.json').read_text()); offset = meta['source_start_s']
    inf = json.loads((a.inference/'inference_summary.json').read_text())
    full_validation=a.inference/'full_log_pair_validation.json'
    if full_validation.exists():
        inf['full_log_pair_validation']=json.loads(full_validation.read_text())
    real = load(a.real/'robot.csv'); sim = load(a.run/'replay.csv'); cmd = load(a.real/'policy.csv')
    normal = load(a.normal/'imu.csv'); odom = load(a.normal/'odom.csv')
    for r in real+cmd:
        r['t_s'] = r['t_mono_ms']*.001
    for r in sim:
        r['t_s'] += offset
    for r in normal+odom:
        r['t_s'] = r['sim_s']
    for r in real+sim+normal+odom:
        if 'quat_w' in r: attitude(r)
    windows = [('quiet', 83., 85.), ('pure_straight_short', 86.14, 87.28), ('forward_with_mixed_commands', 86.5, 92.4)]
    cases = {label: {'real': summarize(real, lo, hi, cmd), 'replay': summarize(sim, lo, hi, cmd)} for label, lo, hi in windows}
    baseline = summarize(normal, 12.9, 18.8)
    baseline['motion'] = summarize(odom, 12.9, 18.8) if 'gyro_x' in odom[0] else {
        'mean_body_vx_mps': st.mean(r['body_vx'] for r in odom if 12.9 <= r['t_s'] < 18.8)}
    targets = load(a.run/'targets.csv'); published = load(a.run/'published.csv'); obs = load(a.inference/'obs_used.csv')
    # Latest consumed policy row must explain every recomputed target, without a lag fit.
    infer = load(a.inference/'inference.csv'); it = [r['source_s'] for r in infer]
    cfg = json.loads((a.real/'meta.json').read_text())['config']; mapping = cfg['policy_to_hardware']
    target_error = []
    for row in targets:
        sample = infer[max(0, bisect.bisect_right(it, offset+row['t_s']+1e-9)-1)]
        for i, h in enumerate(mapping):
            expected = cfg['default_pose'][i]+cfg['action_scale'][i]*sample[f'action_{i:02}']
            target_error.append(row[f'q_{h:02}']-max(cfg['limits']['q_min'][h], min(cfg['limits']['q_max'][h], expected)))
    checks = {'completed': meta.get('completed', False), 'gui_enabled': meta.get('gui_enabled', False),
              'production_files_unchanged': meta['production_files_unchanged'],
              'planned_targets': len(targets), 'published_targets': len(published),
              'indices_in_order': [int(r['index']) for r in published] == list(range(len(targets))),
              'max_publish_lag_s': max(r['t_s']-r['scheduled_s'] for r in published),
              'changed_obs_channels': [i for i in range(45) if any(r[f'original_{i:02}'] != r[f'used_{i:02}'] for r in obs)],
              'target_conversion_max_error_rad': max(map(abs, target_error))}
    result = {'source': str(a.real), 'run': str(a.run), 'inference_summary': inf, 'checks': checks,
              'windows': cases, 'normal_gazebo': baseline,
              'frequency_method': '100Hz linear interpolation; dominant 1-20Hz bin with Hann window and linear detrend; 4-10Hz RMS uses demeaned rectangular DFT/Parseval; no time shift',
              'amplitude_method': 'roll std and total peak-to-peak from normalized quaternion; robust half-amplitude=(p95-p05)/2; gyro RMS includes its mean',
              'limitations': ['Historical real observations drive the policy; simulated feedback does not drive the next input. This is not a new closed-loop controller.',
                             'Pure (0.5,0,0) command window is only 1.14s; frequency resolution is coarse and onset transients matter.',
                             'Longer forward window has nonzero lateral/yaw commands; do not label it pure straight-line motion.',
                             'Real base speed, height, contact state and individual source/application times are unavailable.',
                             'Spectral peaks include gait/contact motion and do not identify sensor noise or causality.',
                             'Original runtime weight hash not recorded; deterministic action agreement supports the candidate without proving file identity.'],
              'sha256': {str(p): sha(p) for p in [a.real/'policy.csv', a.real/'robot.csv', a.real/'meta.json', a.run/'replay.csv', a.run/'targets.csv', a.run/'published.csv', a.run/'metadata.json']}}
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for name, case in cases.items():
        for kind, m in case.items():
            print(name, kind, json.dumps({k: m[k] for k in ['gyro_x_rms_rad_s', 'roll_std_deg', 'roll_peak_to_peak_deg', 'roll_half_p95_p05_deg', 'gyro_spectrum']}, ensure_ascii=False))
    print('checks', checks)
    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 2, figsize=(14, 10), layout='constrained')
        for rows, label in [(real, 'Real robot'), (sim, 'Gazebo: paired obs -> actor')]:
            rr = [r for r in rows if offset <= r['t_s'] <= offset+meta['duration_s']]
            for axis, key in [(ax[0, 0], 'roll_deg'), (ax[0, 1], 'gyro_x')]:
                axis.plot([r['t_s'] for r in rr], [r[key] for r in rr], label=label, lw=.8)
            for axis, key in [(ax[1, 0], 'body_vx'), (ax[1, 1], 'body_vy')]:
                if key in rr[0]: axis.plot([r['t_s'] for r in rr], [r[key] for r in rr], label=label, lw=.8)
        cc = [r for r in cmd if offset <= r['t_s'] <= offset+meta['duration_s']]
        for axis, key in [(ax[1, 0], 'cmd_vx'), (ax[1, 1], 'cmd_vy')]:
            axis.plot([r['t_s'] for r in cc], [r[key] for r in cc], label=key, color='black', ls='--')
        for rows, label, lo in [(real, 'Real robot', 86.5), (sim, 'Paired-obs replay', 86.5), (normal, 'Normal Gazebo closed loop', 12.9)]:
            for axis, key in [(ax[2, 0], 'gyro_x'), (ax[2, 1], 'roll_deg')]:
                _, spec = spectrum(wave(rows, key, lo, lo+5.9))
                axis.plot([v[0] for v in spec], [v[1] for v in spec], label=label)
        titles = [('Body roll', 'deg'), ('Body roll rate', 'rad/s'), ('Forward speed: command vs simulation', 'm/s'), ('Lateral speed: command vs simulation', 'm/s'), ('Roll-rate spectrum (5.9s windows)', 'rad/s peak amplitude'), ('Roll-angle spectrum (5.9s windows)', 'deg peak amplitude')]
        for i, (axis, (title, unit)) in enumerate(zip(ax.flat, titles)):
            axis.set(title=title, ylabel=unit, xlabel='Source log time (s)' if i < 4 else 'Frequency (Hz)')
            axis.grid(alpha=.25); axis.legend(fontsize=8)
            if i < 4: axis.axvspan(86.14, 87.28, alpha=.12, color='green', label='Pure straight command')
            else: axis.set_xlim(0, 20)
        fig.suptitle('New paired observations, P25 / D0.8, original Gazebo controller\nNo filtering or phase compensation; longer forward segment includes lateral/yaw commands')
        fig.savefig(a.plot, dpi=160); plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('real', 'run', 'inference', 'normal', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--plot', type=Path)
    main(p.parse_args())
