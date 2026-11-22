#!/usr/bin/env python3
"""Bounded, simulation-only parameter screening on full fixed-base legs.

Reads an explicit plan. DCMotor envelope/PD retained; candidate mechanical
resistance is Fs*tanh(dq/Va)+B*dq after motor torque clipping.
Existing zero-lag trial plans retain their input and scoring formats.
An optional physx_friction map uses the solver's joint friction instead.
"""
import argparse,csv,hashlib,json,math,sys,time
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--physics-dt',type=float,default=.0005);p.add_argument('--log-period',type=float,default=.005)
AppLauncher.add_app_launcher_args(p);args=p.parse_args()
ROOT=Path(__file__).resolve().parents[3];out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
plan=json.loads(args.plan.read_text());assert 0<len(plan['jobs'])<=512
app=AppLauncher(args).app
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
sys.path.insert(0, str(ROOT/'imgo2_rl/source/imgo2_rl'))
from imgo2_rl.assets import imgo2 as mod

JOINTS=[f'{l}_{k}' for l in ('FL','FR','BL','BR') for k in ('hip','thigh','shank')]
MODEL=[f'{l}_{k}_joint' for l in ('FL','FR','RL','RR') for k in ('hip','thigh','shank')]

def main():
    dt=args.physics_dt;stride=round(args.log_period/dt)
    assert dt>0 and stride>=1 and abs(stride*dt-args.log_period)<1e-10
    protected=[Path(mod.__file__),Path(mod.URDF_PATH),ROOT/'imgo2_description/urdf/imgo2_real.gazebo.urdf',ROOT/'imgo2_description/xacro/core.xacro',ROOT/'imgo2_deploy/ros2_ws/policy/imgo2/ppo/policy.pt']
    before={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in protected}
    records={}
    for item in plan['sources']:
        source=ROOT/item['source'];sha=hashlib.sha256(source.read_bytes()).hexdigest();assert sha==item['sha256']
        with source.open() as f:rows=[{k:float(v) for k,v in r.items() if k not in ('leg','joint')} for r in csv.DictReader(f)]
        for r in rows:
            assert all(r['kp_'+j]==25 and r['kd_'+j]==.8 and r['dq_set_'+j]==0 for j in JOINTS)
        first=rows[0]['send_mono_ms'];times=np.asarray([(r['send_mono_ms']-first)/1000 for r in rows])
        records[item['id']]=dict(item=item,rows=rows,times=times)
    jobs=plan['jobs'];used=sorted({j['source_id'] for j in jobs});recmap={i:k for k,i in enumerate(used)}
    maxdur=max(records[i]['times'][-1] for i in used);assert maxdur<30
    warm=3.;steps=math.ceil((warm+maxdur+.01)/dt)
    timestamps=np.arange(steps)*dt-warm
    # Build each source command schedule once, then gather it per candidate on GPU.
    commands=np.empty((steps,len(used),12),dtype=np.float32);schedules={}
    for k,i in enumerate(used):
        r=records[i];ii=np.maximum(0,np.searchsorted(r['times'],np.maximum(0,timestamps),side='right')-1)
        raw=np.asarray([[row['q_set_'+j] for j in JOINTS] for row in r['rows']],dtype=np.float32)
        commands[:,k,:]=raw[ii];schedules[i]=ii
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt,device=args.device))
    for i in range(len(jobs)):sim_utils.create_prim(f'/World/envs/env_{i}','Xform',translation=(2.*(i%24),2.*(i//24),0.))
    cfg=mod.make_imgo2_resistance_cfg();cfg.prim_path='/World/envs/env_.*/Robot';cfg.spawn.fix_base=True
    cfg.spawn.usd_dir=str(out/'usd_cache');cfg.spawn.usd_file_name='rack_robot.usd';cfg.init_state.pos=(0.,0.,.6)
    cfg.actuators['legs'].stiffness=25.
    cfg.actuators['legs'].damping=.8
    robot=Articulation(cfg);sim.reset();assert robot.is_fixed_base and robot.num_instances==len(jobs)
    indices=[robot.joint_names.index(j) for j in MODEL];inverse=[indices.index(i) for i in range(12)]
    ac=robot.actuators['legs'];assert list(ac.joint_indices)==list(range(12)) if isinstance(ac.joint_indices,list) else True
    ac.fs.zero_()
    ac.b.zero_()
    ac.va.fill_(.02)
    q=robot.data.default_joint_pos.clone();v=torch.zeros_like(q)
    native_friction = torch.zeros((*q.shape, 3), device=sim.device)
    envsel=[]
    for e,job in enumerate(jobs):
        record=records[job['source_id']];row=record['rows'][0]
        q[e,indices]=torch.tensor([row['q_act_'+j] for j in JOINTS],device=sim.device)
        selected=JOINTS.index(record['item']['joint']);envsel.append(indices[selected])
        native = job.get("physx_friction", {})
        if native and job["parameters"]:
            raise ValueError("Select either explicit resistance or PhysX friction for a trial")
        for j, par in native.items():
            static, dynamic, viscous = (par[key] for key in ("static_nm", "dynamic_nm", "viscous_nms"))
            if not (0 <= dynamic <= static <= 2 and 0 <= viscous <= 10):
                raise ValueError(f"Invalid native joint friction for {j}")
            k = indices[JOINTS.index(j)]
            native_friction[e, k] = native_friction.new_tensor((static, dynamic, viscous))
        for j,par in job['parameters'].items():
            k=indices[JOINTS.index(j)];fs=par.get('fs_nm',0.);b=par.get('b_nms',0.);va=par.get('va_rad_s',.02);lag=par.get('motor_lag_s',0.)
            if lag != 0:
                raise ValueError("The shared resistance model accepts only zero motor_lag_s")
            assert 0<=fs<=2 and 0<=b<=10 and 0<va<=1
            ac.fs[e,k]=fs;ac.b[e,k]=b;ac.va[e,k]=va
    robot.write_joint_state_to_sim(q,v);robot.reset()
    use_native = any(job.get("physx_friction") for job in jobs)
    if use_native:
        from isaaclab.utils.version import get_isaac_sim_version

        if get_isaac_sim_version().major < 5:
            raise RuntimeError("Native torque-valued joint friction requires Isaac Sim 5+")
        robot.write_joint_friction_coefficient_to_sim(*(native_friction[..., i] for i in range(3)))
        applied = robot.root_physx_view.get_dof_friction_properties()
        torch.testing.assert_close(applied, native_friction.cpu(), rtol=0, atol=1e-7)
    ct=torch.as_tensor(commands,device=sim.device);recids=torch.tensor([recmap[j['source_id']] for j in jobs],device=sim.device)
    inv=torch.tensor(inverse,device=sim.device);sel=torch.tensor(envsel,device=sim.device);envs=torch.arange(len(jobs),device=sim.device)
    target_v=torch.zeros_like(q);zero=torch.zeros_like(q)
    zero_resistance=(ac.fs==0)&(ac.b==0)
    buffers=[];torques=[];logtimes=[];root0=robot.data.root_pos_w.clone();maxdrift=0.;maxspeed=0.;maxeffort=0.;minpower=0.;zero_error=0.;started=time.monotonic()
    for step in range(steps):
        target=ct[step,recids,:][:,inv]
        robot.set_joint_position_target(target);robot.set_joint_velocity_target(target_v);robot.set_joint_effort_target(zero)
        robot.write_data_to_sim();sim.step();robot.update(dt)
        if step%2000==0:print(f'SCAN step {step}/{steps}, {len(jobs)} envs, wall {time.monotonic()-started:.1f}s',flush=True)
        if timestamps[step]<0 or step%stride:continue
        selected=torch.stack([robot.data.joint_pos[envs,sel],robot.data.joint_vel[envs,sel],robot.data.computed_torque[envs,sel],robot.data.applied_torque[envs,sel]],dim=-1)
        if not torch.isfinite(selected).all():raise RuntimeError('Nonfinite candidate state')
        buffers.append(selected.detach().cpu().numpy());logtimes.append(float(timestamps[step]+dt))
        torque_sample = torch.stack([ac.computed_effort[:, indices], ac.motor_effort[:, indices],
                                     ac.resistance_effort[:, indices], ac.applied_effort[:, indices]], dim=-1)
        torques.append(torque_sample.detach().cpu().numpy())
        maxdrift=max(maxdrift,float((robot.data.root_pos_w-root0).abs().max()))
        maxspeed=max(maxspeed,float(robot.data.joint_vel.abs().max()));maxeffort=max(maxeffort,float(robot.data.applied_torque.abs().max()))
        minpower=min(minpower,float((ac.resistance_effort*ac._joint_vel).min()))
        zero_error=max(zero_error,float(((ac.applied_effort-ac.motor_effort)*zero_resistance).abs().max()))
    data=np.asarray(buffers);assert data.shape==(len(logtimes),len(jobs),4)
    with (out/'traces.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['test_s']+[f'{k}_{e:03d}' for e in range(len(jobs)) for k in ('q','dq','pd','tau')])
        for t,row in zip(logtimes,data):w.writerow([t,*row.ravel().tolist()])
    assert zero_error == 0.0 and minpower >= 0.0
    with (out/'torques.npz').open('wb') as f:
        np.savez_compressed(f, command_eval_time_s=np.asarray(logtimes)-dt,
                            joint_names=MODEL, columns=['pd_nm','motor_nm','resistance_nm','joint_nm'],
                            values=np.asarray(torques))
    (out/'plan_used.json').write_text(json.dumps(plan,indent=2)+'\n')
    (out/'runner_used.py').write_text(Path(__file__).read_text())
    meta=dict(completed=True,dt_s=dt,log_period_s=args.log_period,warmup_s=warm,instances=len(jobs),wall_s=time.monotonic()-started,
        model='DCMotor torque envelope -> minus Fs*tanh(dq/Va)+B*dq',
        pd=[25.,.8],joint_friction_physx=native_friction.cpu().tolist() if use_native else 0.,
        native_friction_telemetry="Solver reaction excluded from actuator torque fields" if use_native else None,
        fixed_base=robot.is_fixed_base,joint_order=robot.joint_names,indices=indices,
        max_base_drift_m=maxdrift,max_joint_speed_rad_s=maxspeed,max_applied_effort_nm=maxeffort,min_resistance_dissipation_w=minpower,
        zero_resistance_effort_error_nm=zero_error,
        protected_hashes=before,production_files_unchanged=all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in before.items()),
        sources=[records[i]['item'] for i in used],device=sim.device,
        source_frames_not_separately_applied={i:len(records[i]['rows'])-len(set(schedules[i][timestamps>=0].tolist())) for i in used},
        motor_lag_is_not_communication_delay=True,physical_parameters_not_uniquely_identified=True)
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print('SCAN COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    finally:app.close(wait_for_replicator=False,skip_cleanup=True)
