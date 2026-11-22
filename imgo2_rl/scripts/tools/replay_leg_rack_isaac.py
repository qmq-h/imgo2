#!/usr/bin/env python3
"""Bounded Isaac Sim rack replay: original training asset, recorded targets.

Keeps complete legs and gravity. Tests matched recorded Kd=.8 and training
nominal Kd=.5 side by side, without DR, policy, hardware or production edits.
Requires the project Isaac Lab environment; numeric analysis is separate.
"""
import argparse
import bisect
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import time

from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-dir',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--physics-dt',type=float,default=.005)
parser.add_argument('--log-period',type=float,default=.005)
parser.add_argument('--only-hz',type=float)
parser.add_argument('--only-amplitude',type=float)
parser.add_argument('--matched-only',action='store_true')
parser.add_argument('--unitree-source',type=Path,help='Verified unmodified official unitree_actuators.py snapshot, diagnostic only')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
app_launcher=AppLauncher(args)
simulation_app=app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

ROOT=Path(__file__).resolve().parents[3]
JOINTS=[f'{l}_{k}' for l in ('FL','FR','BL','BR') for k in ('hip','thigh','shank')]
MODEL_JOINTS=[f'{l}_{k}_joint' for l in ('FL','FR','RL','RR') for k in ('hip','thigh','shank')]


def main():
    sources=sorted(args.source_dir.resolve().glob('*.log'))
    sources=[p for p in sources if (args.only_hz is None or float(re.search(r'_([0-9.]+)Hz_',p.name)[1])==args.only_hz)
             and (args.only_amplitude is None or float(re.search(r'Hz_([0-9.]+)\.log$',p.name)[1])==args.only_amplitude)]
    if not sources:raise ValueError('No source logs')
    asset_file=ROOT/'imgo2_rl/source/imgo2_rl/imgo2_rl/assets/imgo2.py'
    spec=importlib.util.spec_from_file_location('rack_asset',asset_file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    protected=[asset_file,Path(module.URDF_PATH),ROOT/'imgo2_description/urdf/imgo2_real.gazebo.urdf',ROOT/'imgo2_description/xacro/core.xacro']
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    records=[]
    for source in sources:
        with source.open() as f:raw=list(csv.DictReader(f))
        rows=[{k:float(v) for k,v in r.items() if k not in ('leg','joint')} for r in raw]
        first=rows[0]['send_mono_ms']
        records.append(dict(source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            rows=rows,times=[(r['send_mono_ms']-first)/1000 for r in rows]))
    variants=[('matched_kd08',.8)] if args.matched_only else [('matched_kd08',.8),('training_kd05',.5)]
    cases=[dict(record=i,variant=variant,kd=kd) for variant,kd in variants for i in range(len(records))]
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=args.physics_dt,device=args.device))
    origins=[]
    for i in range(len(cases)):
        xyz=[2.*(i%6),2.*(i//6),0.]
        sim_utils.create_prim(f'/World/envs/env_{i}', 'Xform', translation=xyz)
        origins.append(xyz)
    cfg=module.IMGO2_CFG.copy()
    cfg.prim_path='/World/envs/env_.*/Robot'
    cfg.spawn.fix_base=True
    cfg.spawn.usd_dir=str(out/'usd_cache')
    cfg.spawn.usd_file_name='rack_robot.usd'
    cfg.spawn.force_usd_conversion=False
    cfg.init_state.pos=(0.,0.,.6)
    # Original DCMotor envelope retained; per-instance D is set after initialization.
    cfg.actuators['legs'].damping=.8
    reference=None
    if args.unitree_source:
        reference=args.unitree_source.resolve()
        rspec=importlib.util.spec_from_file_location('rack_unitree_reference',reference)
        refmodule=importlib.util.module_from_spec(rspec);sys.modules[rspec.name]=refmodule;rspec.loader.exec_module(refmodule)
        cfg.actuators['legs']=refmodule.UnitreeActuatorCfg_Go2HV(
            joint_names_expr=['.*_joint'],stiffness=25.,damping=.8,friction=.01)
    robot=Articulation(cfg=cfg)
    sim.reset()
    if not robot.is_fixed_base or robot.num_instances!=len(cases):raise RuntimeError('Rack instance/fixation mismatch')
    indices=[robot.joint_names.index(j) for j in MODEL_JOINTS]
    q=robot.data.default_joint_pos.clone();v=torch.zeros_like(q)
    for env,c in enumerate(cases):
        r=records[c['record']]['rows'][0]
        q[env,indices]=torch.tensor([r['q_act_'+j] for j in JOINTS],device=sim.device)
    robot.write_joint_state_to_sim(q,v)
    robot.reset()
    actuator=robot.actuators['legs']
    actuator.stiffness[:]=25.
    for env,c in enumerate(cases):actuator.damping[env,:]=c['kd']
    warmup=3.;dt=sim.get_physics_dt()
    log_stride=round(args.log_period/dt)
    if log_stride<1 or abs(log_stride*dt-args.log_period)>1e-9:raise ValueError('Log period must be a multiple of physics dt')
    max_duration=max(r['times'][-1] for r in records)
    if max_duration>30 or len(cases)>40:raise ValueError('Bound exceeded')
    headers=['t_s','test_s','target_index','target_s','x','y','z']+[f'{k}_{i:02}' for k in ('q','dq','q_des','dq_des','tau_applied','tau_computed') for i in range(12)]
    handles=[];writers=[]
    for c in cases:
        f=(out/f"case_{c['record']:02d}_{c['variant']}.csv").open('w',newline='');handles.append(f)
        w=csv.writer(f);w.writerow(headers);writers.append(w)
    target=torch.empty_like(q);target_v=torch.zeros_like(q);zero=torch.zeros_like(q)
    schedule=[];last_indices=[-1]*len(records)
    started=time.monotonic()
    try:
        for step in range(math.ceil((warmup+max_duration+.02)/dt)):
            t=step*dt;test=t-warmup
            idx=[max(0,bisect.bisect_right(r['times'],max(0.,test))-1) for r in records]
            for rec,ind in enumerate(idx):
                if ind!=last_indices[rec] and test>=0:
                    schedule.append(dict(record=rec,t_s=test,index=ind,source_s=records[rec]['times'][ind]));last_indices[rec]=ind
            for env,c in enumerate(cases):
                r=records[c['record']]['rows'][idx[c['record']]]
                target[env,indices]=torch.tensor([r['q_set_'+j] for j in JOINTS],device=sim.device)
                target_v[env,indices]=torch.tensor([r['dq_set_'+j] for j in JOINTS],device=sim.device)
            robot.set_joint_position_target(target)
            robot.set_joint_velocity_target(target_v)
            robot.set_joint_effort_target(zero)
            robot.write_data_to_sim()
            sim.step()
            robot.update(dt)
            if step%500==0:print(f'RACK STEP {step}: sim {t:.3f}s wall {time.monotonic()-started:.1f}s',flush=True)
            if step%log_stride:continue
            pos=robot.data.joint_pos[:,indices].detach().cpu().tolist()
            vel=robot.data.joint_vel[:,indices].detach().cpu().tolist()
            effort=robot.data.applied_torque[:,indices].detach().cpu().tolist()
            computed=robot.data.computed_torque[:,indices].detach().cpu().tolist()
            des=target[:,indices].detach().cpu().tolist();dvel=target_v[:,indices].detach().cpu().tolist()
            root=robot.data.root_pos_w.detach().cpu().tolist()
            for env,c in enumerate(cases):
                rec=c['record']
                if test<=records[rec]['times'][-1]+.005:
                    writers[env].writerow([t+dt,test+dt,idx[rec],records[rec]['times'][idx[rec]],*root[env],*pos[env],*vel[env],*des[env],*dvel[env],*effort[env],*computed[env]])
        metadata=dict(completed=True,dt_s=dt,warmup_s=warmup,asset_file=str(asset_file),model=str(module.URDF_PATH),
            log_period_s=args.log_period,runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            joint_order=MODEL_JOINTS,isaac_joint_order=robot.joint_names,joint_indices=indices,
            fixed_base=robot.is_fixed_base,instances=robot.num_instances,cases=cases,
            sources=[{k:v for k,v in r.items() if k not in ('rows','times')} for r in records],
            actuator_class=type(actuator).__name__,device=sim.device,
            unitree_reference_source=str(reference) if reference else None,
            unitree_reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest() if reference else None,
            configured_joint_friction=cfg.actuators['legs'].friction,
            min_delay=getattr(cfg.actuators['legs'],'min_delay',None),max_delay=getattr(cfg.actuators['legs'],'max_delay',None),
            protected_hashes=before,production_files_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before.items()),
            wall_s=time.monotonic()-started,
            assumptions=['Level fixed base at 0.6 m; complete legs and gravity retained.',
                         '3 s synthetic constant first target settling; measured source send endpoint ZOH thereafter.',
                         ('Official Go2HV actuator preset with joint friction .01, zero configured delay, recorded Kp 25; diagnostic copy only.'
                          if reference else 'Original training DCMotor limits/friction/solver settings; matched Kd .8 and nominal Kd .5 separate.'),
                         'No policy or domain randomization; no additional dynamics beyond the selected actuator configuration.'])
        (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        (out/'applied_schedule.json').write_text(json.dumps(schedule,indent=2)+'\n')
        print('ALL ISAAC RACK CASES COMPLETE',flush=True)
    finally:
        for f in handles:f.close()


if __name__=='__main__':
    try:main()
    finally:simulation_app.close(wait_for_replicator=False,skip_cleanup=True)
