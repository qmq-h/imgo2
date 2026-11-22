#!/usr/bin/env python3
"""Descriptive target/feedback/gyro coupling; stdlib only, no causal identification."""
import argparse
import cmath
import csv
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics as st
from audit_imu_hip_log import corr
from audit_sim_real_chain import angles


def dft(values, k, window=False):
    n = len(values); mean = st.mean(values)
    return sum((v-mean)*(.5-.5*math.cos(2*math.pi*i/(n-1)) if window else 1)
               * cmath.exp(-2j*math.pi*k*i/n) for i, v in enumerate(values))


def band(values):
    n = len(values)
    bins = list(range(math.ceil(4*n/100), math.floor(10*n/100)+1))
    power = [(k, abs(dft(values, k))**2) for k in bins]
    return {'band_4_10_hz_rms': math.sqrt(2*sum(v for _, v in power))/n,
            'peak_within_4_10_hz': max(power, key=lambda p: p[1])[0]*100/n}


def pair_at_frequency(x, y, k):
    # Five two-second segments in a six-second window, 50% overlap.
    n = 200
    pairs = [(dft(x[a:a+n], k, True), dft(y[a:a+n], k, True))
             for a in range(0, len(x)-n+1, n//2)]
    cross = sum(a.conjugate()*b for a, b in pairs)
    xx = sum(abs(a)**2 for a, _ in pairs); yy = sum(abs(b)**2 for _, b in pairs)
    return {'segments': len(pairs), 'frequency_hz': k*.5,
            'magnitude_squared_coherence': abs(cross)**2/(xx*yy) if xx*yy else None,
            'y_phase_minus_x_deg': math.degrees(cmath.phase(cross))}


def audit(directory, lo, hi):
    keys = ['gyro_x', 'gyro_y', 'gyro_z', 'logged_roll_deg', 'logged_pitch_deg']+[f'{kind}_{i:02}' for kind in ['q_des', 'q', 'dq', 'tau'] for i in range(12)]
    bins = defaultdict(list)
    path = directory/'robot.csv'
    with path.open() as f:
        for row in csv.DictReader(f):
            t = float(row['t_mono_ms'])*.001
            if lo <= t < hi:
                roll, pitch = angles({k: float(row[k]) for k in ['quat_'+a for a in 'wxyz']})
                row['logged_roll_deg'], row['logged_pitch_deg'] = roll, pitch
                bins[math.floor(t*100)].append([float(row[k]) for k in keys])
    ids = sorted(bins)
    if ids != list(range(ids[0], ids[-1]+1)):
        raise ValueError('Missing 100Hz bins; do not treat gaps as uniform samples')
    data = {key: [st.mean(r[j] for r in bins[i]) for i in ids] for j, key in enumerate(keys)}
    summary = {k: band(v) for k, v in data.items()}
    # Choose frequency using gyro energy alone, never by maximizing pair coupling.
    gyro_power = []
    for k in range(8, 21):
        power = sum(abs(dft(data['gyro_x'][a:a+200], k, True))**2
                    for a in range(0, len(ids)-199, 100))
        gyro_power.append((k, power))
    selected = max(gyro_power, key=lambda p: p[1])[0]
    pairs = {}
    for i in range(12):
        u, q = data[f'q_des_{i:02}'], data[f'q_{i:02}']
        pairs[str(i)] = {'gyro_x_to_target': pair_at_frequency(data['gyro_x'], u, selected),
                         'target_to_feedback': pair_at_frequency(u, q, selected)}
    hips = [0, 3, 6, 9]
    modes = {'common': [1, 1, 1, 1], 'left_right': [1, -1, 1, -1],
             'front_back': [1, 1, -1, -1], 'diagonal': [1, -1, -1, 1]}
    hip_focus = {}
    for kind in ['q_des', 'q', 'dq', 'tau']:
        common = [st.mean(data[f'{kind}_{i:02}'][j] for i in hips) for j in range(len(ids))]
        energies = {name: band([sum(s*data[f'{kind}_{i:02}'][j] for s, i in zip(sign, hips))/4
                               for j in range(len(ids))])['band_4_10_hz_rms']**2
                    for name, sign in modes.items()}
        hip_focus[kind] = {'band_mode_energy_fractions': {k: v/sum(energies.values()) for k, v in energies.items()},
                           'common_vs_logged_roll_corr': corr(common, data['logged_roll_deg']),
                           'common_vs_gyro_x_corr': corr(common, data['gyro_x']),
                           'gyro_x_to_common_at_selected_frequency': pair_at_frequency(data['gyro_x'], common, selected)}
    hip_focus['notes'] = 'Hadamard components divided by four. Energy fractions refer only to selected four-hip signals in 4-10Hz, not body energy or causal contribution. Logged roll uses each run quaternion frame; old frame is known incorrect.'
    return {'directory': str(directory), 'window_s': [lo, hi], 'bins': len(ids),
            'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'gyro_selected_welch_frequency_hz': selected*.5, 'signals': summary, 'pairs': pairs, 'hip_focus': hip_focus,
            'hip_target_band_rms_rad': math.sqrt(st.mean(summary[f'q_des_{i:02}']['band_4_10_hz_rms']**2 for i in [0, 3, 6, 9])),
            'hip_feedback_band_rms_rad': math.sqrt(st.mean(summary[f'q_{i:02}']['band_4_10_hz_rms']**2 for i in [0, 3, 6, 9]))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', action='append', required=True, help='label,directory,start_s,end_s')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = {'cases': {}, 'limits': [
        '100Hz host-time mean bins; physical sample clocks and true inference input are unavailable.',
        'Band RMS uses whole-window rectangular DFT; Welch uses 2s Hann segments with 50% overlap.',
        'Shared spectral content/coherence supports coupling, not feedback sign, causality, instability, or loop gain.',
        'Phase is wrapped/periodic and combines sensor, actuator, and sampling effects; not measured device delay.',
        'Shared gait/contact forcing can produce the same coherence; no contact labels or external intervention here.']}
    for item in a.case:
        name, directory, lo, hi = item.split(',')
        r = audit(Path(directory), float(lo), float(hi)); report['cases'][name] = r
        print(name, 'frequency', r['gyro_selected_welch_frequency_hz'],
              'hip target/feedback band RMS', r['hip_target_band_rms_rad'], r['hip_feedback_band_rms_rad'])
        for i in [0, 3, 6, 9]:
            print(i, r['pairs'][str(i)])
    a.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
