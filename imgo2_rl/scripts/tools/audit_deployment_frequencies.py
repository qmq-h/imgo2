#!/usr/bin/env python3
"""Offline frequency audit. Standard library only; never opens hardware.

CSV logging cadence is distinct from serial transactions and sensor updates.
Trace measurements cover only the saved ring window, not the full robot log.
"""
import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from audit_go_imu_runtime import read_csv, stats


ROOT = Path(__file__).resolve().parents[3]


def cadence(times):
    times = sorted(times)
    if len(times) < 2:
        return None
    intervals = [b - a for a, b in zip(times, times[1:])]
    span = times[-1] - times[0]
    return dict(samples=len(times), span_ms=span,
                mean_hz=(len(times)-1)*1000/span,
                interval_ms=stats(intervals),
                nonpositive_intervals=sum(x <= 0 for x in intervals))


def audit(directory):
    robot, bad_robot = read_csv(directory/'robot.csv')
    policy, bad_policy = read_csv(directory/'policy.csv')
    t = robot['t_mono_ms']
    ppo = [i for i, mode in enumerate(robot['mode']) if mode == 3]
    result = dict(real_log=str(directory.relative_to(ROOT)) if directory.is_relative_to(ROOT) else str(directory),
                  malformed_rows=dict(robot=bad_robot, policy=bad_policy),
                  robot=cadence(t), ppo_robot=cadence([t[i] for i in ppo]),
                  policy_consumed=cadence(policy['t_mono_ms']),
                  # Approximation: log time minus age measured just before logging.
                  # This is result completion, not inference start or duration.
                  policy_generated_approx=cadence([a-b for a,b in zip(policy['t_mono_ms'], policy['result_age_ms'])]),
                  policy_result_age_ms=stats(policy['result_age_ms']),
                  imu_reported_hz_after_2s=stats([v for at,v in zip(t,robot['imu_hz']) if at >= 2000]),
                  imu_reported_hz_min_after_2s=min(v for at,v in zip(t,robot['imu_hz']) if at >= 2000),
                  imu_age_ms=stats(robot['imu_age_ms']),
                  imu_valid_counts={str(int(v)): sum(x == v for x in robot['imu_valid']) for v in set(robot['imu_valid'])},
                  joint_age_ms=stats(v for j in range(12) for v in robot[f'age_{j:02}']),
                  joint_age_per_index_ms={str(j):stats(robot[f'age_{j:02}']) for j in range(12)},
                  bus_body_snapshot_us={str(b):stats(robot[f'b{b}_total_us']) for b in range(4)})
    with (directory/'trace.fault.csv').open() as stream:
        trace = list(csv.DictReader(stream))
    groups = defaultdict(list)
    for row in trace:
        groups[(int(row['bus']),int(row['id']))].append(row)
    trace_result = {}
    for (bus, motor), rows in sorted(groups.items()):
        success = [r for r in rows if r['ok'] == '1']
        trace_result[f'bus{bus}_id{motor}'] = dict(
            request_starts=cadence([float(r['t_ms']) for r in rows]),
            successful_completions=cadence([float(r['t_ms'])+float(r['waited_us'])/1000 for r in success]),
            transaction_us=stats([float(r['waited_us']) for r in success]),
            failures=len(rows)-len(success))
    result['fault_trace'] = dict(rows=len(trace),
        start_ms=min(float(r['t_ms']) for r in trace),
        end_ms=max(float(r['t_ms']) for r in trace), per_motor=trace_result)
    result['meta_timing'] = json.loads((directory/'meta.json').read_text())['config']['timing']
    result['sha256'] = {}
    for name in ['robot.csv','policy.csv','trace.fault.csv','meta.json','events.log']:
        digest = hashlib.sha256()
        with (directory/name).open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
        result['sha256'][name] = digest.hexdigest()
    source_paths = [
        'imgo2_deploy/real/config/real_config.h',
        'imgo2_deploy/real/config/deployment.yaml',
        'imgo2_deploy/real/rl_real_imgo2.cpp',
        'imgo2_deploy/real/sdk/inference_worker.cpp',
        'imgo2_deploy/real/Imu/src/imu_monitor.cpp',
        'imgo2_deploy/real/Motor/src/go_loop.cpp',
        'imgo2_deploy/real/Motor/src/go_bus.cpp',
        'imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py',
        'imgo2_deploy/ros2_ws/policy/imgo2/base.yaml',
        'imgo2_description/config/robot_control_ros2.yaml',
        'imgo2_description/xacro/gazebo.xacro',
    ]
    result['current_source_sha256'] = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths}
    gazebo = ROOT/'docs/deployment/ppo_gazebo_ramp_2026-10-16_imu.csv.gz'
    if gazebo.exists():
        with gzip.open(gazebo, 'rt') as stream:
            times = [float(r['sim_s'])*1000 for r in csv.DictReader(stream)]
        result['historical_gazebo_imu_sim_clock'] = cadence(times)
    result['limitations'] = [
        'IMU Hz counts parser sequence increments in ~1 second windows; not per-field or hardware timestamps.',
        'Robot rows are snapshots, not new motor/IMU frame counters. Ages are not update intervals.',
        'Fault trace covers only the final ~9.7 seconds; active-writer ring publication has a known concurrency concern.',
        'Policy rows count consumed distinct results; exact inference input and start/duration were not logged.',
        'No proof that current training config is the original config of the uploaded policy weight.',
        'Gazebo historical timestamps are simulation seconds; real logs use host monotonic time.',
    ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-log', type=Path, default=ROOT/'imgo2_deploy/real/build/log/2026-10-19_22-15-26_19277')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.real_log.resolve())
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(f"robot={result['robot']['mean_hz']:.3f} Hz; consumed policy={result['policy_consumed']['mean_hz']:.3f} Hz")
    print('IMU reported median:',result['imu_reported_hz_after_2s']['median'],'Hz')
    for name, row in result['fault_trace']['per_motor'].items():
        print(name, f"request={row['request_starts']['mean_hz']:.3f} Hz", f"feedback={row['successful_completions']['mean_hz']:.3f} Hz")


if __name__ == '__main__':
    main()
