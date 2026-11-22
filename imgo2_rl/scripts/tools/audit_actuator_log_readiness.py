#!/usr/bin/env python3
"""Audit real actuator log suitability; stdlib only, no training or hardware access."""
import argparse
from bisect import bisect_right
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st


def quantile(xs, fraction):
    xs = sorted(xs)
    index = (len(xs)-1)*fraction
    lo = int(index)
    return xs[lo]+(xs[min(lo+1, len(xs)-1)]-xs[lo])*(index-lo)


def rms(xs):
    return math.sqrt(st.mean(x*x for x in xs))


def read(path):
    with path.open() as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = []
        for line, row in enumerate(reader, 2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f'{path}:{line}: malformed row')
            parsed = {k: float(v) for k, v in row.items()}
            if not all(math.isfinite(x) for x in parsed.values()):
                raise ValueError(f'{path}:{line}: nonfinite value')
            rows.append(parsed)
    return fields, rows


def window(rows, lo, hi):
    subset = [r for r in rows if lo*1000 <= r['t_mono_ms'] < hi*1000]
    if len(subset) < 3:
        raise ValueError('window too short')
    times = [r['t_mono_ms'] for r in subset]
    dt = [b-a for a, b in zip(times, times[1:])]
    roll = []
    for r in subset:
        w, x, y, z = [r['quat_'+k] for k in 'wxyz']
        norm = math.sqrt(w*w+x*x+y*y+z*z)
        w, x, y, z = [v/norm for v in (w, x, y, z)]
        roll.append(math.degrees(math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))))
    result = {'window_s': [lo, hi], 'rows': len(subset),
              'dt_ms': {'p50': quantile(dt, .5), 'p99': quantile(dt, .99), 'max': max(dt)},
              'roll_std_deg': st.pstdev(roll), 'gyro_x_rms_rad_s': rms([r['gyro_x'] for r in subset]),
              'clamped_rows': sum(r['clamped'] != 0 for r in subset),
              'invalid_rows': sum(int(r['valid_mask']) != 4095 for r in subset),
              'error_rows': sum(r['err_mask'] != 0 for r in subset),
              'command_mean': {k: st.mean(r['cmd_'+k] for r in subset) for k in ['vx','vy','wz']},
              'joints': []}
    for j in range(12):
        suffix = f'_{j:02}'
        def values(prefix):
            return [r[prefix+suffix] for r in subset]
        repeated = sum(all(a[k+suffix] == b[k+suffix] for k in ['q','dq','tau'])
                       for a, b in zip(subset, subset[1:]))
        errors = {}
        # Diagnostic host-target as-of scan, NOT measured transport/physical delay.
        # Equal comparison rows: omit first 20ms, hold old target; feedback remains current.
        eligible = [i for i, t in enumerate(times) if t-times[0] >= 20]
        for delay in [0, 5, 10, 15, 20]:
            residuals = []
            for i in eligible:
                r = subset[i]
                past = subset[bisect_right(times, times[i]-delay)-1]
                pred = (past['kp'+suffix]*(past['q_des'+suffix]-r['q'+suffix])
                        +past['kd'+suffix]*(past['dq_des'+suffix]-r['dq'+suffix])
                        +past['tau_ff'+suffix])
                residuals.append(pred-r['tau'+suffix])
            errors[str(delay)] = rms(residuals)
        result['joints'].append({'joint_hardware': j,
            'kp_values': sorted(set(values('kp'))), 'kd_values': sorted(set(values('kd'))),
            'dq_des_abs_max': max(map(abs, values('dq_des'))),
            'tau_ff_abs_max': max(map(abs, values('tau_ff'))),
            'age_ms_p50': quantile(values('age'), .5), 'age_ms_p99': quantile(values('age'), .99),
            'age_ms_max': max(values('age')), 'identical_q_dq_tau_adjacent_fraction': repeated/(len(subset)-1),
            'tau_rms_Nm_reported': rms(values('tau')), 'tau_abs_max_Nm_reported': max(map(abs, values('tau'))),
            'q_std_rad': st.pstdev(values('q')), 'dq_rms_rad_s': rms(values('dq')),
            'host_target_PD_residual_rms_Nm_by_shift_ms': errors})
    result['pooled_PD_residual_rms_by_shift_ms'] = {
        str(d): rms([j['host_target_PD_residual_rms_Nm_by_shift_ms'][str(d)] for j in result['joints']])
        for d in [0,5,10,15,20]}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log', type=Path, required=True)
    p.add_argument('--window', action='append', required=True, help='label:start_s:end_s')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    fields, rows = read(a.log/'robot.csv')
    if any(b['t_mono_ms'] <= c['t_mono_ms'] for c, b in zip(rows, rows[1:])):
        raise ValueError('timestamps not strictly increasing')
    report = {'source': str(a.log), 'files': {}, 'robot_rows': len(rows), 'columns': len(fields),
              'has_feedback_sequence': any('seq' in k.lower() for k in fields),
              'has_per_joint_send_timestamp': any('send_time' in k.lower() for k in fields),
              'windows': {},
              'limitations': ['Feedback torque is uncalibrated; no physical torque accuracy established.',
                  'Host targets and feedback snapshot are asynchronous, no exact motor application timestamps.',
                  'Repeated triplets do not prove duplicated frames; feedback may be quantized.',
                  'PD shift scan is descriptive and label-referenced, not an identified physical delay.',
                  'Fixed-gain walking data do not establish coverage of other gains, loads or voltages.',
                  'No network training or causal attribution of roll oscillation.']}
    for name in ['robot.csv', 'policy.csv', 'meta.json', 'events.log']:
        path = a.log/name
        if path.exists():
            report['files'][name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
    for spec in a.window:
        label, lo, hi = spec.split(':')
        report['windows'][label] = window(rows, float(lo), float(hi))
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    for label, w in report['windows'].items():
        print(label, 'rows', w['rows'], 'roll_std_deg', round(w['roll_std_deg'],3),
              'gyro_x_rms', round(w['gyro_x_rms_rad_s'],3),
              'PD_label_residual', w['pooled_PD_residual_rms_by_shift_ms'])


if __name__ == '__main__':
    main()
