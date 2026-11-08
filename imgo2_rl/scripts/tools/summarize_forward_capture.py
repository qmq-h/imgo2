#!/usr/bin/env python3
"""Normalize a bounded Gazebo capture; preserve raw files and explicit join ages."""
import argparse
import bisect
import csv
import json
import math
from pathlib import Path
import statistics as st


def load(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def save(path, rows):
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    out = args.directory
    collection = json.loads((out/'collection.json').read_text())
    start, end = collection['ppo_pd_start_sim_s'], collection['stand_request_sim_s']
    raw = load(out/'policy_raw.csv')
    sources = {key: load(out/(key+'.csv')) for key in ('state', 'command', 'imu', 'odom', 'velocity')}
    indexes = {key: [float(row['host_s']) for row in rows] for key, rows in sources.items()}
    raw_index = [float(row['host_s']) for row in raw]

    def previous(key, host):
        idx = bisect.bisect_right(indexes[key], host)-1
        return sources[key][idx] if idx >= 0 else None

    def stage(sim):
        if sim < start:
            return 'standing_to_ppo'
        return 'ppo_0p5' if sim < end else 'return_to_stand'

    mapping = {'raw': 'act_raw', 'target': 'q_des', 'obs': 'obs', 'state_q': 'q',
               'state_dq': 'dq', 'state_tau_est': 'tau', 'action_clipped': 'act',
               'output_dq': 'dq_des', 'diagnostic_tau': 'diagnostic_tau', 'kp': 'kp', 'kd': 'kd'}
    policy = []
    for row in raw:
        r = {'sim_s': row['sim_s'], 't_s': float(row['sim_s'])-start,
             'host_s': row['host_s'], 'wall_s': row['wall_s'], 'stage': stage(float(row['sim_s']))}
        for key, value in row.items():
            if key in r:
                continue
            prefix, _, index = key.rpartition('_')
            name = f'{mapping[prefix]}_{int(index):02}' if prefix in mapping and index.isdigit() else key
            r[name] = value
        policy.append(r)
    save(out/'policy.csv', policy)

    robot = []
    for command in sources['command']:
        host = float(command['host_s'])
        matched = {key: previous(key, host) for key in ('state', 'imu', 'odom', 'velocity')}
        r = {'host_s': host, 'sim_s': command['sim_s'], 't_s': float(command['sim_s'])-start,
             'requested_stage': matched['velocity']['stage'] if matched['velocity'] else '',
             'mode_inferred': 'ppo_pd' if all(abs(float(command[f'kp_{i:02}'])-25)<1e-3 for i in range(12)) else 'fixed_pd'}
        for key, row in matched.items():
            r[key+'_arrival_age_ms'] = (host-float(row['host_s']))*1000 if row else ''
            r[key+'_source_sim_s'] = row['sim_s'] if row else ''
        for key, source in [('cmd_vx', 'vx'), ('cmd_vy', 'vy'), ('cmd_wz', 'wz')]:
            r[key] = matched['velocity'][source] if matched['velocity'] else ''
        for prefix, source in [('q_des', 'q'), ('dq_des', 'dq'), ('tau_ff', 'tau'), ('kp', 'kp'), ('kd', 'kd')]:
            for i in range(12):
                r[f'{prefix}_{i:02}'] = command[f'{source}_{i:02}']
        for prefix in ('q', 'dq', 'tau'):
            for i in range(12):
                r[f'{prefix}_{i:02}'] = matched['state'][f'{prefix}_{i:02}'] if matched['state'] else ''
        for name in ('quat_w', 'quat_x', 'quat_y', 'quat_z', 'gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z'):
            r[name] = matched['imu'][name] if matched['imu'] else ''
        for name in ('x', 'y', 'z', 'yaw', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz'):
            r['odom_'+name] = matched['odom'][name] if matched['odom'] else ''
        idx = bisect.bisect_right(raw_index, host)-1
        r['latest_policy_arrival_age_ms'] = (host-raw_index[idx])*1000 if idx >= 0 else ''
        for i in range(12):
            r[f'latest_computed_act_{i:02}'] = raw[idx][f'action_clipped_{i}'] if idx >= 0 else ''
        robot.append(r)
    save(out/'robot.csv', robot)

    imu = [r for r in sources['imu'] if start+2 <= float(r['sim_s']) < end-.2]
    odom = sources['odom']
    velocities = []
    for a, b in zip(odom, odom[1:]):
        dt = float(b['sim_s'])-float(a['sim_s'])
        if dt > 0 and start+2 <= float(b['sim_s']) < end-.2:
            yaw = float(b['yaw'])
            velocities.append(((float(b['x'])-float(a['x']))*math.cos(yaw)+(float(b['y'])-float(a['y']))*math.sin(yaw))/dt)
    angles = [[], []]
    for row in imu:
        w, x, y, z = [float(row['quat_'+k]) for k in 'wxyz']
        angles[0].append(math.degrees(math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))))
        angles[1].append(math.degrees(math.asin(max(-1, min(1, 2*(w*y-z*x))))))
    heights = [float(r['z']) for r in odom if start+2 <= float(r['sim_s']) < end-.2]
    summary = {'policy_frames': len(raw), 'robot_rows': len(robot),
               'actual_ppo_pd_interval_sim_s': [start, end], 'actual_ppo_pd_duration_s': end-start,
               'policy_command_counts': {v: sum(r['cmd_vx'] == v for r in raw) for v in sorted({r['cmd_vx'] for r in raw})},
               'stable_interval_sim_s': [start+2, end-.2],
               'mean_body_vx_from_pose_mps': st.mean(velocities),
               'gyro_xy_rms_rads': math.sqrt(st.mean(float(r['gyro_x'])**2+float(r['gyro_y'])**2 for r in imu)),
               'roll_std_deg': st.pstdev(angles[0]), 'pitch_std_deg': st.pstdev(angles[1]),
               'height_mean_m': st.mean(heights), 'height_std_m': st.pstdev(heights),
               'final_kp': [float(sources['command'][-1][f'kp_{i:02}']) for i in range(12)],
               'final_q_des': [float(sources['command'][-1][f'q_{i:02}']) for i in range(12)],
               'time_limits': 'policy sim_s uses ROS /clock (coarse); host_s monotonic precise. IMU/odom sim_s use exact message header stamps. Unstamped motor topics use last received clock. robot.csv joins only previous host arrivals, with explicit ages; not simultaneous sensor sampling.',
               'policy_pairing': 'policy.csv obs/act_raw are exact same inference call. Real logger currently rebuilds next-input obs beside an older consumed clipped act; do not pair those real columns rowwise.',
               'missing_hardware_fields': 'No motor temperature/error codes/bus RTT in Gazebo; not fabricated.'}
    summary['observed_rates'] = {}
    for name, rows in [('policy_trace', raw), *sources.items()]:
        host_span = float(rows[-1]['host_s'])-float(rows[0]['host_s'])
        sim_span = float(rows[-1]['sim_s'])-float(rows[0]['sim_s'])
        summary['observed_rates'][name] = {
            'rows': len(rows), 'host_span_s': host_span, 'sim_span_s': sim_span,
            'hz_host_arrival': (len(rows)-1)/host_span,
            'hz_sim_observed': (len(rows)-1)/sim_span,
        }
    summary['rate_limits'] = 'ROS topic arrival rates do not distinguish publisher scheduling misses from transport drops; no header sequence in motor messages. Actual policy trace records each fresh inference.'
    assert abs(end-start-collection['hold_seconds']) < .15
    assert all(abs(x-60) < 1e-3 for x in summary['final_kp'])
    for row in raw:
        assert len(row) == 182 and None not in row
        assert all(math.isfinite(float(v)) for v in row.values())
        assert float(row['cmd_vy']) == float(row['cmd_wz']) == 0
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
