#!/usr/bin/env python3
"""Start/stop only owned process groups; software GL, gui=false, isolated ROS domain59/master11369.
Uses the existing built deployment, not a checkpoint selected by the output path.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import argparse

parser = argparse.ArgumentParser(description="Short fresh-world Gazebo direction audit of the currently deployed AMP policy.")
parser.add_argument("--output-root", required=True, type=Path)
parser.add_argument("--summary-json", type=Path)
parser.add_argument("--cases", default="forward_04,backward_04,turn_left,turn_right")
args = parser.parse_args()
REPO = Path(__file__).resolve().parents[2]
LABEL = "direction_audit"
OUT = args.output_root.resolve()
OUT.mkdir(parents=True, exist_ok=True)
base = os.environ.copy()
for key in ('PYTHONHOME', 'PYTHONPATH', 'LD_LIBRARY_PATH'):
    base.pop(key, None)
raw_env = subprocess.check_output(['bash', '-c', 'source /opt/ros/humble/setup.bash\nsource imgo2_deploy/install/setup.bash\nenv -0'], cwd=REPO, env=base)
env = dict(item.decode().split('=', 1) for item in raw_env.split(b'\0') if b'=' in item)
env.update(ROS_DOMAIN_ID='59', ROS_LOCALHOST_ONLY='1', GAZEBO_MASTER_URI='http://127.0.0.1:11369', LIBGL_ALWAYS_SOFTWARE='1',
           IMGO2_GAZEBO_URDF=str(REPO / 'imgo2_description/urdf/imgo2_real.gazebo.urdf'))
cases = [('stand', 0.0, 0.0), ('forward', 0.5, 0.0), ('backward', -0.5, 0.0),
         ('turn_left', 0.0, 0.5), ('turn_right', 0.0, -0.5), ('backward_fast', -1.0, 0.0), ('turn_fast', 0.0, 1.0), ('backward_turn', -0.5, 0.5), ('forward_04', 0.4, 0.0), ('backward_04', -0.4, 0.0)]
names = args.cases.split(',')
if set(names)-{c[0] for c in cases}:
    parser.error('Unknown direction case')
cases = [c for c in cases if c[0] in names]
summaries = []
for name, vx, wz in cases:
    print('Starting', LABEL, name, flush=True)
    folder = OUT / name
    folder.mkdir(exist_ok=True)
    env['IMGO2_POLICY_TRACE'] = str(folder / 'policy_trace.csv')
    processes = []
    files = []
    try:
        for title, cmd in [('gazebo', ['ros2', 'launch', 'imgo2_deploy', 'gazebo.launch.py', 'gui:=false']),
                           ('rl_sim', [str(REPO / 'imgo2_deploy/install/lib/imgo2_deploy/rl_sim')])]:
            handle = (folder / (title + '.log')).open('w')
            files.append(handle)
            process = subprocess.Popen(cmd, cwd=REPO / 'imgo2_deploy', env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
            if title == 'gazebo':
                time.sleep(5)
        deadline = time.monotonic() + 65
        while time.monotonic() < deadline:
            text = (folder / 'rl_sim.log').read_text(errors='replace')
            if 'loop_keyboard' in text:
                break
            if any(p.poll() is not None for p in processes):
                raise RuntimeError('Gazebo/control exited during startup; see logs')
            time.sleep(1)
        else:
            raise RuntimeError('Controller startup timed out')
        cmd = ['/usr/bin/python3', str(REPO / 'imgo2_deploy/scripts/eval_gazebo_policy.py'),
               '--key', '3', '--vx', str(vx), '--wz', str(wz), '--duration', '26', '--skip', '4', '--sim-time',
               '--json', str(folder / 'summary.json'), '--raw-json', str(folder / 'stable.json'), '--transient-json', str(folder / 'transient.json')]
        result = subprocess.run(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=45)
        (folder / 'eval.log').write_text(result.stdout)
        if result.returncode:
            raise RuntimeError('Evaluation failed: ' + result.stdout[-1200:])
        summary = json.loads((folder / 'summary.json').read_text())
        summaries.append({'case': name, 'log_dir': str(folder), **summary})
        print(name, 'z=', round(summary['pose']['z_mean'], 4), 'vx=', round(summary['tracking']['speed_mean'], 4),
              'yaw=', round(summary['tracking']['yaw_rate_mean'], 4), 'ddq=', round(summary['jitter']['thigh']['mean_abs_ddq'], 2), flush=True)
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
        for process in reversed(processes):
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        for handle in files:
            handle.close()
    time.sleep(1)
    (args.summary_json or (OUT / 'summary.json')).write_text(json.dumps(summaries, indent=2) + '\n')
print('Finished and stopped all processes started by this audit.', flush=True)
