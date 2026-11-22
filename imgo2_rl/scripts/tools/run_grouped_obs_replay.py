#!/usr/bin/env python3
"""Bounded simulation-only ablations; never connects to physical robot."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--torch-python',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--video-encoder',type=Path)
    p.add_argument('--resume',action='store_true',help='Reuse completed cases; retain failed attempts with a suffix')
    p.add_argument('--hip-dq-scale-sweep',action='store_true',help='Four cases: original, hip dq gains .75/.5/.25')
    p.add_argument('--body-hip-q-sweep',action='store_true',help='Original, donor IMU only, donor IMU plus hip q gains .75/.5/.25')
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(exist_ok=a.resume,parents=True)
    if a.hip_dq_scale_sweep and a.body_hip_q_sweep:p.error('choose only one experiment')
    source=ROOT/'imgo2_deploy/logs/2026-11-17_21-10-22_7163'
    donor=ROOT/'imgo2_deploy/logs/height23_forward05_kd08_20261109_a'
    policy=ROOT/'imgo2_deploy/logs/nov17_policy_identity_20261117_a/policy_used.pt'
    scripts=ROOT/'imgo2_deploy/ros2_ws/scripts'
    cases=[]
    groups=('baseline','dq075','dq050','dq025') if a.hip_dq_scale_sweep else ('baseline','body','hip_state','hip_history','combined','all_joint_state')
    if a.body_hip_q_sweep:groups=('baseline','body','body_q075','body_q050','body_q025')
    gains={'baseline':1.,'dq075':.75,'dq050':.5,'dq025':.25}
    q_gains={'baseline':1.,'body':1.,'body_q075':.75,'body_q050':.5,'body_q025':.25}
    for group in groups:
        inf=a.output/(group+'_inference');run=a.output/(group+'_gazebo')
        case=dict(group=group,inference=str(inf),run=str(run))
        if a.hip_dq_scale_sweep:case['hip_dq_gain']=gains[group]
        if a.body_hip_q_sweep:case['hip_q_gain']=q_gains[group]
        cases.append(case)
    plan=dict(source=str(source),donor=str(donor),policy=str(policy),start=70.,duration=10.,
        donor_start=13.,analysis_window=[76.8,79.6],height=.3032156576428986,cases=cases,
        alignment='predeclared fixed offset, no phase fitting',gui=True)
    if a.hip_dq_scale_sweep:plan.update(experiment='hip_velocity_scaling',alignment='original real input phase and timing preserved; no donor input or additional filter')
    if a.body_hip_q_sweep:plan.update(experiment='body_donor_hip_position_scaling',
        alignment='fixed input-capture offset: real70s to donor13s; no phase fitting; hip q relative to default; dq/history real')
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    for case in cases:
        group=case['group'];print('PREPARE '+group,flush=True)
        run=Path(case['run']);inf=Path(case['inference'])
        if a.resume and (run/'metadata.json').exists() and json.loads((run/'metadata.json').read_text()).get('completed'):
            print('REUSE COMPLETED '+group,flush=True);continue
        if run.exists():
            n=1
            while run.with_name(run.name+f'_failed_{n}').exists():n+=1
            run.rename(run.with_name(run.name+f'_failed_{n}'))
        cmd=[str(a.torch_python),str(scripts/'prepare_obs_replay.py'),'--source',str(source),
             '--policy',str(policy),'--output',case['inference'],'--start','70','--duration','10',
             '--stats-window','main,76.8,79.6']
        if a.body_hip_q_sweep and group!='baseline':
            cmd+=['--obs-donor',str(donor),'--obs-group','body','--obs-donor-start','13','--hip-q-gain',str(case['hip_q_gain'])]
        elif a.hip_dq_scale_sweep and group!='baseline':cmd+=['--hip-dq-gain',str(case['hip_dq_gain'])]
        elif group!='baseline':cmd+=['--obs-donor',str(donor),'--obs-group',group,'--obs-donor-start','13']
        else:cmd+=['--verify-all-pairs']
        if not (a.resume and inf.exists()):
            with (a.output/(group+'_prepare.log')).open('w') as f:
                subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        print('GUI REPLAY '+group,flush=True)
        cmd=[sys.executable,str(scripts/'replay_gazebo_targets.py'),'--source',str(source),
            '--kind','real','--output',case['run'],'--start','70','--duration','10',
            '--height',str(plan['height']),'--targets-override',str(Path(case['inference'])/'targets.csv'),
            '--gui','--gui-warmup','8']
        if a.video_encoder:cmd+=['--video-encoder',str(a.video_encoder)]
        with (a.output/(group+'_run.log')).open('w') as f:
            subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        print('DONE '+group,flush=True)

if __name__=='__main__':main()
