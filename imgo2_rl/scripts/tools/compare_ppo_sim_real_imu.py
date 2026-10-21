#!/usr/bin/env python3
"""Compare existing Gazebo and robot logs offline, without hardware or ML libs.

Windows are explicitly selected examples, not labelled contact/lift trials.
100Hz bin averages reduce differing logging rates; they are not anti-alias filters.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

from audit_go_imu_runtime import read_csv

ROOT = Path(__file__).resolve().parents[3]


def rpy(w, x, y, z):
    norm = math.sqrt(w*w+x*x+y*y+z*z)
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError('invalid quaternion')
    w,x,y,z = (v/norm for v in (w,x,y,z))
    return [math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
            math.asin(max(-1., min(1., 2*(w*y-z*x)))),
            math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))]


def summarize(rows):
    def distribution(values):
        return dict(mean=statistics.mean(values), std=statistics.pstdev(values),
                    min=min(values), max=max(values))
    bins = {}
    for row in rows:
        bins.setdefault(math.floor(row['t']*100+1e-8), []).append(row['gyro'])
    binned = [[statistics.mean(g[a] for g in values) for a in range(3)]
              for values in bins.values()]
    return dict(samples=len(rows), command_ranges=[
        [min(row['cmd'][a] for row in rows),max(row['cmd'][a] for row in rows)] for a in range(3)],
        roll_deg=distribution([row['rpy'][0]*180/math.pi for row in rows]),
        pitch_deg=distribution([row['rpy'][1]*180/math.pi for row in rows]),
        gyro_rms_rad_s=[math.sqrt(statistics.mean(row['gyro'][a]**2 for row in rows)) for a in range(3)],
        gyro_100hz_bin_mean_rms_rad_s=[math.sqrt(statistics.mean(g[a]**2 for g in binned)) for a in range(3)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    path = ROOT/'imgo2_deploy/real/build/log/2026-10-19_22-15-26_19277/robot.csv'
    c, bad = read_csv(path)
    if bad:
        raise ValueError(f'{bad} malformed real rows')
    real = [dict(t=c['t_mono_ms'][i]/1000,
                 cmd=[c[k][i] for k in ['cmd_vx','cmd_vy','cmd_wz']],
                 rpy=rpy(*(c['quat_'+k][i] for k in 'wxyz')),
                 gyro=[c['gyro_'+k][i] for k in 'xyz'])
            for i in range(len(c['mode'])) if c['mode'][i] == 3]
    sim_path = ROOT/'docs/deployment/ppo_gazebo_ramp_2026-10-16_imu.csv.gz'
    with gzip.open(sim_path, 'rt') as stream:
        sim = [dict(t=float(row['t_s']),cmd=[float(row['scheduled_vx_mps']),0.,0.],
                    rpy=[float(row[k+'_rad']) for k in ['roll','pitch','yaw']],
                    gyro=[float(row['gyro_'+k+'_rads']) for k in 'xyz'])
               for row in csv.DictReader(stream) if 0 <= float(row['t_s']) < 50]
    windows = {
        'sim_static': (sim,2,4),
        'sim_forward_accel_025_035': (sim,10,12),
        'sim_forward_decel_035_025': (sim,18,20),
        'sim_backward_near_peak': (sim,34,36),
        'real_static': (real,52,54),
        'real_forward_03': (real,73,75),
        'real_backward_05': (real,165,168),
        'real_zero_command_active': (real,170,172),
        'real_zero_command_quiet': (real,176,178),
        'real_yaw_active': (real,244,246),
        'real_yaw_quiet': (real,248,250),
    }
    report = dict(windows={}, source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [path,sim_path]})
    for name, (rows, start, end) in windows.items():
        selected = [r for r in rows if start <= r['t'] < end]
        report['windows'][name] = dict(start_s=start,end_s=end,**summarize(selected))
        print(name, 'gyro RMS', [round(v,4) for v in report['windows'][name]['gyro_rms_rad_s']])
    report['limitations'] = [
        'Sim times are seconds after PPO in simulation clock; real times are seconds after logger start.',
        'Sim uses ramped vx, real uses stepped vx/vy/wz and unlabelled lifting. Not a controlled paired trial.',
        'Real robot.csv contains no accelerometer or actual base linear velocity.',
        'RPY uses normalized wxyz ZYX convention; absolute biases do not establish mount/axis mismatch.',
        '100Hz bins average existing host snapshots; cannot reconstruct original device samples or equalize unknown filters.',
    ]
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')


if __name__ == '__main__':
    main()
