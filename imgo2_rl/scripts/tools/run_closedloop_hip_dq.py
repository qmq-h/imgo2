#!/usr/bin/env python3
"""Four bounded own-feedback Gazebo cases, with isolated actor hip dq gain."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    scripts=ROOT/'imgo2_deploy/ros2_ws/scripts';saved=ROOT/'imgo2_deploy/logs/height23_compare_bundle_20261109_a'
    cases=[{'label':label,'gain':gain,'runner':str(out/(label+'_runner'))} for label,gain in [('gain100',1.),('gain075',.75),('gain050',.5),('gain025',.25)]]
    plan={'experiment':'own Gazebo feedback hip dq input scaling','saved_bundle':str(saved),'cases':cases,
          'hold_seconds':12.,'vx':.5,'kp':25.,'kd':.8,'window_after_ppo_s':[3.,9.],
          'maximum_attempts_per_case':2,'gui':True,
          'limitation':'normal Gazebo baseline is already stable; no claim this reproduces real loaded oscillation'}
    def save():(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    save()
    for case in cases:
        print('BUILD '+case['label'],flush=True)
        with (out/(case['label']+'_build.log')).open('w') as f:
            subprocess.run([sys.executable,str(scripts/'build_saved_policy_runner.py'),'--saved',str(saved),
                '--output',case['runner'],'--hip-dq-gain',str(case['gain'])],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        case['attempts']=[]
        for attempt in (1,2):
            run=out/(case['label']+f'_run{attempt}');case['attempts'].append(str(run));case['run']=str(run);save()
            print('GUI OWN FEEDBACK '+case['label']+f' attempt {attempt}',flush=True)
            with (out/(case['label']+f'_run{attempt}.log')).open('w') as f:
                r=subprocess.run([sys.executable,str(scripts/'record_ppo_forward_comparison.py'),'--output',str(run),
                    '--hold-seconds','12','--vx','.5','--expected-kp','25','--expected-kd','.8',
                    '--rl-sim',str(Path(case['runner'])/'rl_sim_saved'),'--gui'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            case['returncode']=r.returncode;save()
            if r.returncode:
                case['status']='failed_or_unstable';save();break
            collection=json.loads((run/'collection.json').read_text());t0=collection['ppo_pd_start_sim_s']
            with (run/'imu.csv').open() as f:tt=[float(row['sim_s'])-t0 for row in csv.DictReader(f)]
            gaps=[[x,y,y-x] for x,y in zip(tt,tt[1:]) if x<9 and y>=3 and y-x>.05]
            case['spectral_gaps_over_50ms']=gaps
            if not gaps:case['status']='complete';save();break
            case['status']='gap_rerun_required';save()
            print('RETAIN GAP RUN; retry same conditions '+case['label'],flush=True)
        print('CASE FINISHED '+case['label']+' '+case['status'],flush=True)

if __name__=='__main__':main()
