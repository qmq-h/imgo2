#!/usr/bin/env python3
"""Compare quaternion-derived body rates with gyro using identical 40ms windows."""
import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics as st
from audit_imu_hip_log import body_rotation_rate, corr, rms, mul


def check(path, time_key, scale, lo, hi, reference=None):
    bins = defaultdict(list)
    with path.open() as f:
        for row in csv.DictReader(f):
            t = float(row[time_key])*scale
            if lo <= t < hi:
                bins[math.floor(t*100)].append((t, [float(row['quat_'+k]) for k in 'wxyz'],
                                              [float(row['gyro_'+k]) for k in 'xyz']))
    samples = {}
    for key, rows in bins.items():
        q = [st.mean(r[1][i]*(1 if sum(a*b for a, b in zip(r[1], rows[0][1])) >= 0 else -1) for r in rows) for i in range(4)]
        n = math.sqrt(sum(x*x for x in q))
        q = [x/n for x in q]
        if reference is not None:
            # Offline counterfactual: logged Qc=Qraw*B^-1. Recover the same
            # orientation with a WORLD-reference change B^-1*Qraw instead.
            inverse = [reference[0], *[-x for x in reference[1:]]]
            q = mul(mul(inverse, q), reference)
        samples[key] = (st.mean(r[0] for r in rows), q,
                        [st.mean(r[2][i] for r in rows) for i in range(3)])
    rates, gyros = {}, {}
    for k in samples:
        if all(k+j in samples for j in range(-2, 3)):
            rates[k] = body_rotation_rate(samples[k-2][1], samples[k+2][1], samples[k+2][0]-samples[k-2][0])
            gyros[k] = [st.mean(samples[k+j][2][i] for j in range(-2, 3)) for i in range(3)]
    axes = {}
    for i, axis in enumerate('xyz'):
        lag_scan = []
        for lag in range(-10, 11):
            keys = [k for k in rates if k+lag in gyros]
            x, y = [gyros[k+lag][i] for k in keys], [rates[k][i] for k in keys]
            lag_scan.append({'gyro_index_shift_ms': lag*10, 'corr': corr(x, y),
                             'quat_rate_over_gyro_fit': sum(a*b for a, b in zip(x, y))/sum(a*a for a in x),
                             'residual_rms': rms([a-b for a, b in zip(x, y)])})
        keys = list(rates)
        axes[axis] = {'zero_lag': lag_scan[10],
                      'best_absolute_corr': max(lag_scan, key=lambda x: abs(x['corr'] or 0)),
                      'gyro_smoothed_rms': rms([gyros[k][i] for k in keys]),
                      'quat_rate_rms': rms([rates[k][i] for k in keys])}
    keys = list(rates)
    cross = [[corr([rates[k][i] for k in keys], [gyros[k][j] for k in keys]) for j in range(3)] for i in range(3)]
    return {'source': str(path), 'interval': [lo, hi], 'axes': axes, 'rate_vs_gyro_cross_correlation': cross,
            'limits': '100Hz host/message-time bins; centered 40ms quaternion change/gyro mean. Index-shift fit is exploratory, not a measured device latency or axis calibration. Gyro and fused quaternion can differ under filtering/linear acceleration.'}


def startup_reference(path):
    """Estimate B independently of motion fitting from raw warmup samples.

    This log contains raw startup attitudes followed by a >30-degree jump to
    the zeroed attitude. Host samples can repeat sensor frames, so this is an
    estimate of the firmware's 50-frame mean, not its exact saved calibration.
    """
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            t = float(r['t_mono_ms'])*.001
            if t > 2:
                break
            q = [float(r['quat_'+k]) for k in 'wxyz']
            n = math.sqrt(sum(x*x for x in q)); q = [x/n for x in q]
            if rows and abs(sum(x*y for x, y in zip(q, rows[-1][1]))) < math.cos(math.radians(15)):
                break
            rows.append((t, q))
        else:
            raise ValueError('No startup reference discontinuity found')
    if len(rows) < 10 or t > 2:
        raise ValueError('Insufficient raw startup samples before discontinuity')
    q = [st.mean(r[1][i] for r in rows) for i in range(4)]
    n = math.sqrt(sum(x*x for x in q)); q = [x/n for x in q]
    w, x, y, z = q
    return {'quaternion_wxyz': q, 'raw_host_samples': len(rows),
            'raw_interval_s': [rows[0][0], rows[-1][0]], 'jump_time_s': t,
            'rpy_deg': [math.degrees(math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))),
                        math.degrees(math.asin(max(-1, min(1, 2*(w*y-z*x))))),
                        math.degrees(math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--real', type=Path, required=True)
    p.add_argument('--sim', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    collection = json.loads((a.sim/'collection.json').read_text())
    result = {'real_forward': check(a.real/'robot.csv', 't_mono_ms', .001, 46, 52),
              'real_backward': check(a.real/'robot.csv', 't_mono_ms', .001, 66, 70),
              'sim_forward': check(a.sim/'imu.csv', 'sim_s', 1, collection['ppo_pd_start_sim_s']+2, collection['stand_request_sim_s']-.2)}
    reference = startup_reference(a.real/'robot.csv')
    for name, lo, hi in [('real_forward_reference_counterfactual', 46, 52),
                         ('real_backward_reference_counterfactual', 66, 70)]:
        result[name] = check(a.real/'robot.csv', 't_mono_ms', .001, lo, hi,
                             reference=reference['quaternion_wxyz'])
    result['startup_reference'] = reference
    # A pure body-X rotation with an arbitrary initial world heading must still
    # produce body-X angular velocity. Right subtraction fails that invariant.
    b = reference['quaternion_wxyz']; inv = [b[0], *[-x for x in b[1:]]]
    q0, q1 = mul(b, [1, 0, 0, 0]), mul(b, [math.cos(.005), math.sin(.005), 0, 0])
    result['synthetic_body_x_test'] = {
        'expected_rate_rad_s': [1, 0, 0],
        'right_reference_rate': body_rotation_rate(mul(q0, inv), mul(q1, inv), .01),
        'left_reference_rate': body_rotation_rate(mul(inv, q0), mul(inv, q1), .01),
        'scope': 'Algebra check, not a hardware recovery or closed-loop simulation.'}
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    for name, row in result.items():
        print(name, json.dumps(row.get('axes', row)))


if __name__ == '__main__':
    main()
