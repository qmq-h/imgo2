#!/usr/bin/env python3
"""Velocity/target telemetry audit, stdlib only. No hardware or causal plant identification."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics as st
from audit_sim_real_chain import load, rms
from audit_imu_hip_log import corr

GROUPS = {'hip': [0, 3, 6, 9], 'thigh': [1, 4, 7, 10], 'shank': [2, 5, 8, 11]}


def agreement(a, b):
    return {'correlation': corr(a, b), 'qdot_over_dq_fit': sum(x*y for x, y in zip(a, b))/sum(y*y for y in b),
            'residual_rms_rad_s': rms([x-y for x, y in zip(a, b)])}


def targets(rows, prefix):
    events = []
    for row in rows:
        u = [row[f'{prefix}_{i:02}'] for i in range(12)]
        if not events or max(abs(a-b) for a, b in zip(u, events[-1])) > 1e-7:
            events.append(u)
    groups = {}
    for name, ids in GROUPS.items():
        groups[name] = {'per_new_target_delta_rms_rad': rms([b[i]-a[i] for a, b in zip(events, events[1:]) for i in ids]),
                        'target_std_rad': math.sqrt(st.mean(st.pvariance(r[f'{prefix}_{i:02}'] for r in rows) for i in ids))}
    return {'logged_rows': len(rows), 'distinct_target_events': len(events), 'groups': groups}


def audit(real_dir, sim_dir):
    real = [r for r in load(real_dir/'robot.csv') if 30000 <= r['t_mono_ms'] < 36000]
    sim = [r for r in load(sim_dir/'state.csv') if 12.9 <= r['sim_s'] < 20.7]
    commands = [r for r in load(sim_dir/'command.csv') if 12.9 <= r['sim_s'] < 20.7]
    odom = [r for r in load(sim_dir/'odom.csv') if sim[0]['host_s']-.05 <= r['host_s'] <= sim[-1]['host_s']+.05]
    mh, ms = st.mean(r['host_s'] for r in odom), st.mean(r['sim_s'] for r in odom)
    rate = sum((r['host_s']-mh)*(r['sim_s']-ms) for r in odom)/sum((r['host_s']-mh)**2 for r in odom)
    sim_agreement, real_agreement = [], []
    for j in range(12):
        # RobotState has no source stamp. Test known controller 1ms discretization,
        # then check its overall rate independently against stamped odometry.
        qdot = [(b[f'q_{j:02}']-a[f'q_{j:02}'])/.001 for a, b in zip(sim, sim[1:])]
        dq = [r[f'dq_{j:02}'] for r in sim[1:]]
        periods = [(b[f'q_{j:02}']-a[f'q_{j:02}'])/b[f'dq_{j:02}'] for a, b in zip(sim, sim[1:]) if abs(b[f'dq_{j:02}']) > .05]
        sim_agreement.append(dict(joint=j, **agreement(qdot, dq),
                                  median_deltaq_over_dq_s=st.median(periods),
                                  fraction_deltaq_over_dq_within_10us_of_1ms=st.mean(abs(t-.001) < 1e-5 for t in periods)))
        ids = list(range(2, len(real)-2))
        v = [(real[i+2][f'q_{j:02}']-real[i-2][f'q_{j:02}'])/((real[i+2]['t_mono_ms']-real[i-2]['t_mono_ms'])*.001) for i in ids]
        w = [st.mean(real[i+k][f'dq_{j:02}'] for k in range(-2, 3)) for i in ids]
        real_agreement.append(dict(joint=j, **agreement(v, w)))
    rt, sit = targets(real, 'q_des'), targets(commands, 'q')
    return {'real_directory': str(real_dir), 'sim_directory': str(sim_dir),
            'windows': {'real_host_s': [30, 36], 'sim_cached_clock_selection_s': [12.9, 20.7]},
            'sim_velocity': sim_agreement, 'real_velocity': real_agreement,
            'sim_independent_time_check': {'odom_source_time_vs_host_slope': rate,
                                           'state_rate_per_mapped_sim_second': (len(sim)-1)/(rate*(sim[-1]['host_s']-sim[0]['host_s'])),
                                           'state_messages': len(sim), 'distinct_cached_clock_values': len({r['sim_s'] for r in sim})},
            'real_targets': rt, 'sim_targets': sit,
            'real_to_sim_target_step_ratio': {k: rt['groups'][k]['per_new_target_delta_rms_rad']/sit['groups'][k]['per_new_target_delta_rms_rad'] for k in GROUPS},
            'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [real_dir/'robot.csv', sim_dir/'state.csv', sim_dir/'command.csv', sim_dir/'odom.csv']},
            'limits': ['Sim source timestamp per RobotState is unavailable. 1ms check uses configured update rate; global stamped odom rate is independent but cannot validate each individual source dt.',
                       'Real derivative uses centered ~20ms host intervals/gyro-independent joint snapshots; sim check uses one-step difference. Correlations are not comparable noise scores.',
                       'Different closed-loop target histories and real actual speed/contact labels unknown; this is not identical-input actuator identification.',
                       'Target events deduplicate all-12 target vectors, not each joint independently; logged host publications are not exact motor application timestamps.']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--real', type=Path, required=True)
    p.add_argument('--sim', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.real, a.sim)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['sim_independent_time_check']))
    print('target step ratios', result['real_to_sim_target_step_ratio'])
    for k in ['sim_velocity', 'real_velocity']:
        print(k, [(r['joint'], round(r['correlation'], 6), round(r['qdot_over_dq_fit'], 4)) for r in result[k]])
