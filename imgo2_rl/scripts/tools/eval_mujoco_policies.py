#!/usr/bin/env python3
"""Standard-library orchestration of bounded headless deployment-MuJoCo policy benchmarks.

Requires the existing deployment C++ libraries, not Python MuJoCo or Isaac Sim.
Uses production MujocoMotorPd/MujocoFeedback and deterministic simulation-time ticks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
WS=ROOT/'imgo2_deploy/ros2_ws'
CASES={'stand':(0.,0.,0.),'forward01':(.1,0.,0.),'forward03':(.3,0.,0.),
       'forward05':(.5,0.,0.),'forward08':(.8,0.,0.),'backward03':(-.3,0.,0.),
       'backward05':(-.5,0.,0.),'left03':(0.,.3,0.),'right03':(0.,-.3,0.),
       'turn_left':(0.,0.,.8),'turn_right':(0.,0.,-.8),'arc':(.4,0.,.8)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--policies',nargs='+',default=['policy.pt','policy_now.pt'])
    p.add_argument('--cases',default=','.join(CASES))
    p.add_argument('--seconds',type=float,default=20.)
    p.add_argument('--binary',type=Path,help='Reuse a previously compiled benchmark executable')
    a=p.parse_args()
    if not 8<=a.seconds<=60:p.error('seconds must be between 8 and 60')
    cases=a.cases.split(',')
    if any(c not in CASES for c in cases):p.error('unknown case name')
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    runtime=WS/'library/inference_runtime/libtorch';mj=WS/'library/mujoco'
    source=WS/'scripts/mujoco_policy_benchmark.cpp'
    header=WS/'src/imgo2_deploy/include'
    scene=ROOT/'imgo2_description/mjcf/scene.xml'
    config=WS/'policy/imgo2/ppo/config.yaml';base=WS/'policy/imgo2/base.yaml'
    policies=[(WS/'policy/imgo2/ppo'/f).resolve() for f in a.policies]
    files=[source,header/'mujoco_motor_pd.hpp',header/'mujoco_feedback.hpp',scene,
           ROOT/'imgo2_description/mjcf/imgo2.xml',config,base,*policies]
    fingerprints={str(f.relative_to(ROOT)) if f.is_relative_to(ROOT) else str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    binary=a.binary.resolve() if a.binary else a.output/'benchmark'
    env=os.environ.copy()
    # The runtime search paths are embedded in the diagnostic binary.
    env.pop('LD_LIBRARY_PATH',None)
    cmd=['c++','-O1','-std=c++17','-D_GLIBCXX_USE_CXX11_ABI=1',str(source),
         '-I'+str(runtime/'include'),'-I'+str(runtime/'include/torch/csrc/api/include'),
         '-I'+str(mj/'include'),'-I'+str(header),'-L'+str(runtime/'lib'),'-L'+str(mj/'lib'),
         '-Wl,-rpath,'+str(runtime/'lib'),'-Wl,-rpath,'+str(mj/'lib'),
         '-ltorch','-ltorch_cpu','-lc10','-lmujoco','-lyaml-cpp','-pthread','-o',str(binary)]
    result={'scope':'Headless MuJoCo current model, production PD/feedback adapters, fixed simulation-time policy/control ticks; not GUI/FSM scheduling equivalence',
            'warmup':'2s fixed default stance followed by 3s zero-command PPO; independent reset for every case',
            'seconds_per_case':a.seconds,'source_sha256':fingerprints,'compile_command':cmd,'cases':[]}
    def save():
        (a.output/'run.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    if not a.binary:
        print('Compiling headless MuJoCo comparison...',flush=True)
        compiled=subprocess.run(cmd,env=env,text=True,capture_output=True,timeout=150)
        (a.output/'compile.log').write_text(compiled.stdout+compiled.stderr)
        compiled.check_returncode()
    for policy in policies:
        for case in cases:
            label=policy.stem+'_'+case
            start=time.monotonic()
            csv=a.output/(label+'.csv')
            run=subprocess.run([str(binary),str(scene),str(policy),str(base),str(config),str(csv),
                                *map(str,CASES[case]),str(a.seconds)],env=env,text=True,capture_output=True,timeout=120)
            (a.output/(label+'.log')).write_text(run.stdout+run.stderr)
            run.check_returncode()
            record=json.loads(run.stdout.strip().splitlines()[-1])
            record.update(policy=policy.name,case=case,command=CASES[case],csv=str(csv),wall_seconds=time.monotonic()-start)
            result['cases'].append(record);save()
            print(f'{label}: {record["rows"]} frames, failure={record["failure"] or "none"}, {record["wall_seconds"]:.1f}s wall',flush=True)
    after={str(f.relative_to(ROOT)) if f.is_relative_to(ROOT) else str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    result['source_and_policy_files_unchanged']=after==fingerprints
    save()
    if after!=fingerprints:raise RuntimeError('A source/configuration/policy changed during the comparison')
    print('Comparison finished; all production files unchanged.',flush=True)


if __name__=='__main__':main()
