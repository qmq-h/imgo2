#!/usr/bin/env python3
"""Logged-obs inference -> simulation-only target replay input.
Does not reconstruct observations, export weights, or touch real hardware.
"""
import argparse
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path


def read(path):
    with path.open() as f:return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
def rms(values):return math.sqrt(sum(x*x for x in values)/len(values))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def describe_group_replacement(report):
    """Keep provenance labels consistent with the actual replaced indices."""
    group=report.get('group_replacement')
    if not group:return
    indices=group['obs_indices']
    if any(33<=i<45 for i in indices):
        report['last_action_input']='selected donor previous actions replace logged channels; retained channels remain logged; NOT own recomputed previous output'
    if any(21<=i<33 for i in indices):
        report['dq_source']='selected captured normal-policy velocity channels; all unreplaced velocity channels remain logged real'
    if any(3<=i<6 for i in indices):report['filters']['pitch_preserved']=False

def pull_shank_positions(chosen,used,robot,cfg,fraction):
    """Retrospective input intervention toward ORIGINAL same-tick q_des.

    Same-tick q_des was generated from the original logged forward, not from
    the modified forward. It is hindsight information, not a causal controller.
    Only four scaled position channels change; dq/IMU/history remain logged.
    """
    indices=(2,5,8,11)
    if not 0<fraction<=1:raise ValueError('shank pull fraction must be in (0,1]')
    if len(chosen)!=len(used):raise ValueError('observation row count mismatch')
    ticks={}
    for r in robot:
        if 'tick' not in r:raise ValueError('shank same-tick pull requires robot ticks')
        tick=r['tick']
        if tick in ticks:raise ValueError('duplicate robot tick')
        ticks[tick]=r
    output=[];records=[];scale=cfg['obs_scales']['dof_pos']
    if not math.isfinite(scale) or scale<=0:raise ValueError('invalid position observation scale')
    for r,x0 in zip(chosen,used):
        if r.get('tick') not in ticks:raise ValueError('policy tick missing from robot log')
        target=ticks[r['tick']];x=x0.copy()
        row={'source_s':r['t_mono_ms']*.001,'input_captured_s':r['input_captured_ms']*.001,
             'tick':r['tick'],'target_record_s':target['t_mono_ms']*.001}
        for i in indices:
            h=cfg['policy_to_hardware'][i]
            q=x0[9+i]/scale+cfg['default_pose'][i]
            q_des=target[f'q_des_{h:02}'];q_used=q+fraction*(q_des-q)
            x[9+i]=(q_used-cfg['default_pose'][i])*scale
            if not math.isfinite(x[9+i]) or abs(x[9+i])>cfg['obs_scales']['clip_obs']:
                raise ValueError('shank position intervention exceeds observation bounds')
            row.update({f'original_q_{i:02}':q,f'target_q_{i:02}':q_des,
                        f'used_q_{i:02}':q_used,f'error_before_{i:02}':q_des-q,
                        f'error_after_{i:02}':q_des-q_used})
        output.append(x);records.append(row)
    return output,records

def replace_gravity(chosen,used,donor,real_onset,normal_onset,minimum):
    """Offline counterfactual: only projected gravity changes, not gyro/recurrence.
    Reference time follows predeclared command-onset alignment, never a lag fit.
    Before minimum donor time, hold its first quiet reference orientation.
    """
    imu=read(donor/'imu.csv');tt=[r['sim_s'] for r in imu];output=[];records=[]
    for r,x0 in zip(chosen,used):
        captured=r['input_captured_ms']*.001
        requested=normal_onset+captured-real_onset;t=max(minimum,requested)
        if not tt[0]<=t<=tt[-1]:raise ValueError('gravity donor does not cover requested time')
        j=min(len(tt)-1,max(1,bisect.bisect_right(tt,t)));a,b=imu[j-1],imu[j]
        if tt[j]-tt[j-1]>.02:raise ValueError('gravity donor IMU gap over 20ms')
        fraction=(t-tt[j-1])/(tt[j]-tt[j-1]);qa=[a['quat_'+k] for k in 'wxyz'];qb=[b['quat_'+k] for k in 'wxyz']
        if sum(u*v for u,v in zip(qa,qb))<0:qb=[-v for v in qb]
        q=[u+fraction*(v-u) for u,v in zip(qa,qb)];norm=math.sqrt(sum(v*v for v in q))
        if norm<1e-6:raise ValueError('invalid donor quaternion')
        w,qx,qy,qz=[v/norm for v in q]
        gravity=[2*(w*qy-qx*qz),-2*(w*qx+qy*qz),-1+2*(qx*qx+qy*qy)]
        x=x0.copy();x[3:6]=gravity;output.append(x)
        records.append([r['t_mono_ms']*.001,captured,requested,t,int(requested<minimum)]+gravity+
            [math.degrees(math.atan2(2*(w*qx+qy*qz),1-2*(qx*qx+qy*qy))),
             math.degrees(math.asin(max(-1,min(1,2*(w*qy-qz*qx))))),
             math.degrees(math.atan2(2*(w*qz+qx*qy),1-2*(qy*qy+qz*qz)))])
    return output,records

def replace_joint_state(chosen,used,donor,cfg,real_onset,normal_onset,minimum):
    """Replace q/dq together from the same normal-state samples; keep IMU real.
    RobotState lacks a source timestamp: map host receipt to odom source time,
    rather than differentiating its cached 10Hz sim_s column.
    """
    state=read(donor/'state.csv');odom=read(donor/'odom.csv');hosts=[r['host_s'] for r in odom]
    def sample(rows,times,t,key):
        if not times[0]<=t<=times[-1]:raise ValueError('joint donor time outside coverage')
        j=min(len(times)-1,max(1,bisect.bisect_right(times,t)));dt=times[j]-times[j-1]
        if not 0<dt<=.06:raise ValueError('joint donor nonmonotonic time/gap')
        f=(t-times[j-1])/dt;return rows[j-1][key]+f*(rows[j][key]-rows[j-1][key])
    state=[r for r in state if hosts[0]<=r['host_s']<=hosts[-1]]
    tt=[sample(odom,hosts,r['host_s'],'sim_s') for r in state]
    output=[];records=[]
    for r,x0 in zip(chosen,used):
        captured=r['input_captured_ms']*.001;requested=normal_onset+captured-real_onset;t=max(minimum,requested)
        q=[sample(state,tt,t,f'q_{h:02}') for h in cfg['policy_to_hardware']]
        dq=[sample(state,tt,t,f'dq_{h:02}') for h in cfg['policy_to_hardware']]
        x=x0.copy();x[9:21]=[(v-cfg['default_pose'][i])*cfg['obs_scales']['dof_pos'] for i,v in enumerate(q)]
        x[21:33]=[v*cfg['obs_scales']['dof_vel'] for v in dq];output.append(x)
        records.append([r['t_mono_ms']*.001,captured,requested,t,int(requested<minimum)]+q+dq)
    return output,records

def filter_observations(rows,dq_hz=0.,roll_hz=0.,gyro_hz=0.):
    """Causal diagnostic EMA on selected scaled dq, gravity roll and/or gyro.
    Warm up on all earlier logged policy rows, resetting across generation/gaps.
    """
    output=[];previous=None;filtered_dq=None;filtered_roll=None;filtered_gyro=None
    for r in rows:
        x=[r[f'obs_{i:02}'] for i in range(45)]
        dt=(r['t_mono_ms']-previous['t_mono_ms'])*.001 if previous else 0.
        reset=previous is None or dt<=0 or dt>.2 or r.get('gen')!=previous.get('gen')
        if gyro_hz:
            alpha=1-math.exp(-2*math.pi*gyro_hz*dt) if not reset else 1.
            filtered_gyro=[a+alpha*(b-a) for a,b in zip(filtered_gyro or x[:3],x[:3])]
            if reset:filtered_gyro=x[:3].copy()
            x[:3]=filtered_gyro
        if dq_hz:
            alpha=1-math.exp(-2*math.pi*dq_hz*dt) if not reset else 1.
            filtered_dq=[a+alpha*(b-a) for a,b in zip(filtered_dq or x[21:33],x[21:33])]
            if reset:filtered_dq=x[21:33].copy()
            x[21:33]=filtered_dq
        if roll_hz:
            gx,gy,gz=x[3:6];norm=math.sqrt(gx*gx+gy*gy+gz*gz)
            if norm<1e-6:raise ValueError('invalid projected gravity for roll filtering')
            roll=math.atan2(-gy,-gz);pitch=math.atan2(gx,math.hypot(gy,gz))
            alpha=1-math.exp(-2*math.pi*roll_hz*dt) if not reset else 1.
            if reset:filtered_roll=roll
            else:filtered_roll+=alpha*math.atan2(math.sin(roll-filtered_roll),math.cos(roll-filtered_roll))
            x[3:6]=[norm*math.sin(pitch),-norm*math.sin(filtered_roll)*math.cos(pitch),-norm*math.cos(filtered_roll)*math.cos(pitch)]
        output.append(x);previous=r
    return output

def main(a):
    import torch
    torch.set_num_threads(1)
    a.source=a.source.resolve();a.policy=a.policy.resolve();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    path=a.source/'policy.csv';robotpath=a.source/'robot.csv';metapath=a.source/'meta.json'
    rows=read(path);robot=read(robotpath);meta=json.loads(metapath.read_text());cfg=meta['config']
    times=[r['t_mono_ms']*.001 for r in rows]
    rt=[r['t_mono_ms']*.001 for r in robot];r0=max(0,bisect.bisect_right(rt,a.start)-1)
    outer=[(a.start,robot[r0])]+[(t,r) for t,r in zip(rt,robot) if a.start<t<=a.start+a.duration]
    use_ticks=all('tick' in r for r in rows+robot)
    policy_ticks=[r['tick'] for r in rows] if use_ticks else []
    if use_ticks:
        first=max(0,bisect.bisect_right(policy_ticks,outer[0][1]['tick'])-1)
        stop=bisect.bisect_right(policy_ticks,outer[-1][1]['tick'])
    else:
        first=max(0,bisect.bisect_right(times,a.start)-1);stop=bisect.bisect_right(times,a.start+a.duration)
    chosen=rows[first:stop]
    if not chosen or (not use_ticks and times[first]>a.start):raise ValueError('missing initial prior policy observation')
    if chosen[-1]['t_mono_ms']*.001<a.start+a.duration-.05:raise ValueError('missing final policy observation')
    filter_rows=rows
    if a.dq_from_position:
        filter_rows=[r.copy() for r in rows]
        for j in range(1,len(rows)):
            r,before=rows[j],rows[j-1];dt=(r['t_mono_ms']-before['t_mono_ms'])*.001
            if not 0<dt<=.2 or r.get('gen')!=before.get('gen'):continue
            for i in range(12):
                velocity=(r[f'obs_{i+9:02}']-before[f'obs_{i+9:02}'])/(dt*cfg['obs_scales']['dof_pos'])
                value=velocity*cfg['obs_scales']['dof_vel'];clip=cfg['obs_scales']['clip_obs']
                filter_rows[j][f'obs_{i+21:02}']=max(-clip,min(clip,value))
    filtered_all=filter_observations(filter_rows,a.smooth_dq_hz,a.smooth_roll_hz,a.smooth_gyro_hz)
    used=filtered_all[first:first+len(chosen)]
    hip_velocity_scaling=None
    if a.hip_dq_gain!=1.:
        indices=[21,24,27,30]
        for x in used:
            for i in indices:x[i]*=a.hip_dq_gain
        hip_velocity_scaling={'gain':a.hip_dq_gain,'obs_indices':indices,
            'formula':'used obs[i] = gain * original obs[i], i in [21,24,27,30]',
            'preserved':'all other 41 observed channels; original timestamps and action history',
            'physical_PD_unchanged':True,'extra_filter_delay':False}
    group_metadata=None
    if a.obs_donor:
        donor=a.obs_donor.resolve();reference=read(donor/'policy_raw.csv');odom=read(donor/'odom.csv')
        hosts=[r['host_s'] for r in odom]
        def source_time(host):
            if not hosts[0]<=host<=hosts[-1]:raise ValueError('donor policy outside odometry coverage')
            j=min(len(hosts)-1,max(1,bisect.bisect_right(hosts,host)))
            f=(host-hosts[j-1])/(hosts[j]-hosts[j-1])
            return odom[j-1]['sim_s']+f*(odom[j]['sim_s']-odom[j-1]['sim_s'])
        reference=[r for r in reference if hosts[0]<=r['host_s']<=hosts[-1]]
        tt=[source_time(r['host_s']) for r in reference]
        hips=(0,3,6,9)
        groups={'body':list(range(6)), 'hip_state':[k+i for k in (9,21) for i in hips],
                'hip_history':[33+i for i in hips], 'all_joint_state':list(range(9,33))}
        groups['combined']=sorted(set(groups['body']+groups['hip_state']+groups['hip_history']))
        indices=groups[a.obs_group];records=[]
        for r,x in zip(chosen,used):
            requested=a.obs_donor_start+r['input_captured_ms']*.001-a.start
            if not tt[0]<=requested<=tt[-1]:raise ValueError('group donor time outside coverage')
            j=bisect.bisect_right(tt,requested)-1
            if requested-tt[j]>.05:raise ValueError('group donor sample age exceeds 50ms')
            for i in indices:x[i]=reference[j][f'obs_{i}']
            records.append([r['t_mono_ms']*.001,requested,tt[j],requested-tt[j],j])
        with (a.output/'group_donor.csv').open('w') as f:
            w=csv.writer(f);w.writerow(['source_s','requested_s','donor_s','age_s','donor_row']);w.writerows(records)
        group_metadata={'group':a.obs_group,'obs_indices':indices,'source':str(donor),
            'sha256':{n:sha(donor/n) for n in ('policy_raw.csv','odom.csv')},
            'real_start_s':a.start,'donor_start_s':a.obs_donor_start,
            'method':'fixed time offset using input capture; donor policy host mapped to odometry source time; previous captured donor row, no interpolation or phase fit',
            'maximum_sample_age_s':max(r[3] for r in records)}
    hip_position_scaling=None
    if a.hip_q_gain!=1.:
        indices=[9,12,15,18]
        for x in used:
            for i in indices:x[i]*=a.hip_q_gain
        hip_position_scaling={'gain':a.hip_q_gain,'obs_indices':indices,
            'formula':'q_used = default_q + gain * (original_q - default_q); scaled relative obs multiplied by gain',
            'preserved':'all non-hip-q channels except separately declared donor channels; hip dq and last action unchanged',
            'physical_PD_unchanged':True,'extra_filter_delay':False}
    donor_metadata=None;joint_metadata=None;shank_metadata=None
    if a.shank_target_pull:
        used,records=pull_shank_positions(chosen,used,robot,cfg,a.shank_target_pull)
        with (a.output/'shank_pull.csv').open('w') as f:
            writer=csv.DictWriter(f,list(records[0]));writer.writeheader();writer.writerows(records)
        shank_metadata={'fraction':a.shank_target_pull,'policy_joint_indices':[2,5,8,11],
            'obs_indices':[11,14,17,20],'reference':'original robot.csv q_des at SAME policy result/control tick',
            'formula':'q_used = original_q + fraction * (original_same_tick_q_des - original_q)',
            'preserved':'other 41 obs channels including dq, gyro, gravity, command and previous action',
            'target_is_not_recomputed_output':True,'causal_controller':False}
    if a.gravity_donor:
        donor=a.gravity_donor.resolve()
        used,records=replace_gravity(chosen,used,donor,a.gravity_real_onset,a.gravity_normal_onset,a.gravity_donor_min)
        with (a.output/'gravity_donor.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['source_s','input_captured_s','donor_requested_s','donor_used_s','quiet_hold','gravity_x','gravity_y','gravity_z','donor_roll_deg','donor_pitch_deg','donor_yaw_deg']);writer.writerows(records)
        donor_metadata={'source':str(donor),'imu_sha256':sha(donor/'imu.csv'),
            'real_command_onset_s':a.gravity_real_onset,'normal_command_onset_s':a.gravity_normal_onset,
            'donor_minimum_s':a.gravity_donor_min,'quiet_hold_rows':sum(r[4] for r in records),
            'method':'normal onset + real input capture - real onset; normalized quaternion interpolation; only obs[3:6] replaced',
            'rpy':'reference RPY logged for inspection; yaw and Euler angles are not independent actor inputs'}
    if a.joint_donor:
        donor=a.joint_donor.resolve()
        used,records=replace_joint_state(chosen,used,donor,cfg,a.joint_real_onset,a.joint_normal_onset,a.joint_donor_min)
        with (a.output/'joint_donor.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['source_s','input_captured_s','donor_requested_s','donor_used_s','quiet_hold']+[f'{k}_{i:02}' for k in ('q','dq') for i in range(12)]);writer.writerows(records)
        joint_metadata={'source':str(donor),'sha256':{n:sha(donor/n) for n in ('state.csv','odom.csv')},
            'real_command_onset_s':a.joint_real_onset,'normal_command_onset_s':a.joint_normal_onset,
            'donor_minimum_s':a.joint_donor_min,'quiet_hold_rows':sum(r[4] for r in records),
            'method':'normal onset + real input capture - real onset; state host mapped to odom exact sim stamps; only obs[9:33] replaced',
            'preserved':'all gyro/gravity/commands/previous-action inputs from real log unchanged'}
    obs=torch.tensor(used,dtype=torch.float32)
    if not torch.isfinite(obs).all():raise ValueError('nonfinite observation')
    model=torch.jit.load(str(a.policy),map_location='cpu');model.eval()
    with torch.inference_mode():
        if a.self_last_action:
            previous=None;previous_row=None;new_inputs=[];outputs=[]
            for j,(r,x0) in enumerate(zip(rows[:first+len(chosen)],filtered_all)):
                x=x0.copy()
                reset=previous is None or r.get('gen')!=previous_row.get('gen') or r['t_mono_ms']-previous_row['t_mono_ms']>200
                if not reset:x[33:45]=previous
                y=model(torch.tensor([x],dtype=torch.float32))
                previous=torch.maximum(torch.tensor(cfg['lower']),torch.minimum(torch.tensor(cfg['upper']),y[0])).tolist()
                previous_row=r
                if j>=first:new_inputs.append(x);outputs.append(y)
            used=new_inputs;obs=torch.tensor(used,dtype=torch.float32);raw=torch.cat(outputs,dim=0)
        else:
            raw=model(obs)
        if raw.shape!=(len(chosen),12) or not torch.isfinite(raw).all():raise ValueError('invalid actor output')
        # Ensure batched deterministic replay agrees with individual calls.
        single_diff=max((model(obs[j:j+1])-raw[j:j+1]).abs().max().item() for j in (0,len(chosen)//2,len(chosen)-1))
        clipped=torch.maximum(torch.tensor(cfg['lower']),torch.minimum(torch.tensor(cfg['upper']),raw))
    with (a.output/'obs_used.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['source_s']+[f'{k}_{i:02}' for k in ('original','used') for i in range(45)])
        for r,x in zip(chosen,used):w.writerow([r['t_mono_ms']*.001]+[r[f'obs_{i:02}'] for i in range(45)]+x)
    raw=raw.tolist();actions=clipped.tolist();ct=[r['t_mono_ms']*.001 for r in chosen]
    with (a.output/'inference.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['source_s']+[f'{k}_{i:02}' for k in ('raw','action','logged_action','error') for i in range(12)])
        for r,t,u,v in zip(chosen,ct,raw,actions):
            logged=[r[f'act_{i:02}'] for i in range(12)]
            w.writerow([t]+u+v+logged+[x-y for x,y in zip(v,logged)])
    # Robot and policy CSV timestamps are taken at different points in one loop.
    # Same-tick policy output already produced that tick's submitted target;
    # timestamp-asof alone incorrectly delays it to the next 5ms row.
    chosen_ticks=[r['tick'] for r in chosen] if use_ticks else []
    mapping=cfg['policy_to_hardware'];limits=cfg['limits'];clip_count=0
    with (a.output/'targets.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['t_s']+[f'{k}_{i:02}' for k in ('q','dq','tau','kp','kd') for i in range(12)])
        for t,r in outer:
            n=max(0,bisect.bisect_right(chosen_ticks,r['tick'])-1) if use_ticks else max(0,bisect.bisect_right(ct,t)-1)
            if use_ticks:
                if chosen_ticks[n]>r['tick']:raise ValueError('future policy tick lookup')
            elif ct[n]>t+1e-9:raise ValueError('future observation lookup')
            v=actions[n];q=[0.]*12
            for i,h in enumerate(mapping):q[h]=cfg['default_pose'][i]+cfg['action_scale'][i]*v[i]
            limited=[max(lo,min(hi,x)) for x,lo,hi in zip(q,limits['q_min'],limits['q_max'])]
            clip_count+=sum(x!=y for x,y in zip(q,limited))
            w.writerow([t-a.start]+limited+[0.]*24+[r[f'{k}_{i:02}'] for k in ('kp','kd') for i in range(12)])
    def stats(lo,hi):
        idx=[j for j,t in enumerate(ct) if lo<=t<hi]
        if not idx:return {'samples':0}
        return {'samples':len(idx),'clipped_action_rmse':rms([actions[j][i]-chosen[j][f'act_{i:02}'] for j in idx for i in range(12)]),
                'per_joint_action_rmse':[rms([actions[j][i]-chosen[j][f'act_{i:02}'] for j in idx]) for i in range(12)],
                'hip_target_error_rms_rad':rms([(actions[j][i]-chosen[j][f'act_{i:02}'])*cfg['action_scale'][i] for j in idx for i in (0,3,6,9)])}
    paired='精确输入' in meta.get('orders',{}).get('policy_csv','')
    all_errors=[actions[j][i]-chosen[j][f'act_{i:02}'] for j in range(len(chosen)) for i in range(12)]
    windows={'quiet_22_27':stats(22,27),'onset_28_30':stats(28,30),'forward_30_35_9':stats(30,35.9)}
    for item in a.stats_window:
        label,lo,hi=item.split(',');windows[label]=stats(float(lo),float(hi))
    limits=['simulation response is observation replay, not a closed loop with Gazebo feedback',
            'host input capture and result timestamps are not individual sensor source/application timestamps',
            'policy hash is computed here; original runtime weight identity needs its own saved hash']
    if not paired:limits.append('legacy policy.csv obs reconstructed at result consumption, not actual forward input')
    report={'source':str(a.source),'policy_candidate':str(a.policy),'sha256':{str(p):sha(p) for p in (path,robotpath,metapath,a.policy)},
            'hip_position_scaling':hip_position_scaling,
            'hip_velocity_scaling':hip_velocity_scaling,
            'group_replacement':group_metadata,
            'gravity_replacement':donor_metadata,'joint_replacement':joint_metadata,
            'input_output_pairing':'same-forward exact input/output declared by source metadata' if paired else 'legacy reconstructed observation',
            'whole_window_action_rmse':rms(all_errors),'whole_window_action_max_abs_error':max(map(abs,all_errors)),
            'dq_source':('causal backward difference of logged obs positions at policy host times' if a.dq_from_position else 'logged motor velocity observation'),
            'filters':{'dq_cutoff_hz':a.smooth_dq_hz,'gravity_roll_cutoff_hz':a.smooth_roll_hz,'gyro_cutoff_hz':a.smooth_gyro_hz,'method':'causal EMA alpha=1-exp(-2*pi*fc*actual_dt), warmup before window; no phase compensation','gyro_filtered':bool(a.smooth_gyro_hz),'pitch_preserved':True},
            'window_s':[a.start,a.start+a.duration],'policy_samples':len(chosen),'outer_commands':len(outer),
            'batch_vs_single_max_abs_diff':single_diff,'safety_position_clips':clip_count,
            'timing':('same control tick policy-to-submitted-command pairing; original robot host cadence; row write time is not an extra delay; no phase compensation' if use_ticks else 'legacy timestamp asof fallback; missing control tick prevents precise policy/command pairing'),
            'last_action_input':('own recomputed clipped previous action; warmed up from earliest logged PPO observation' if a.self_last_action else 'recorded obs[33:45] unchanged; not replaced by recomputed actions'),
            'windows':windows,'limitations':limits}
    if hip_velocity_scaling:
        report['dq_source']='original four hip velocity observation channels multiplied by fixed gain; remaining eight dq channels unchanged'
        report['limitations']+=['position and previous-action histories remain original; scaled velocity is not a new self-consistent physical feedback trajectory',
            'only actor observation changes; actual Gazebo velocity and motor internal PD derivative feedback are unchanged']
    if hip_position_scaling:
        report['limitations']+=['only four relative hip q observations are scaled; original dq/history and separately declared donor IMU are not physically self-consistent',
            'hip target output and physical motor angle are not directly scaled; all twelve actions are re-inferred from modified observations']
    if group_metadata:
        report['limitations']+=['donor groups mixed with retained historical real channels are not a self-consistent feedback trajectory; no physical root cause or sensor fault follows from improvement or failure',
            'donor previous-action channels are exogenous history, not the previous newly computed output; fixed alignment has no gait phase matching']
    if donor_metadata:
        report['filters']['pitch_preserved']=False
        report['limitations']+=['gravity is from a different run while gyro/q/dq/previous actions remain real; mixed observations are not a physically self-consistent feedback trajectory',
            'single predeclared command-onset alignment; no gait phase optimization; an intervention is not a sensor-fault or original-cause proof']
    if joint_metadata:
        report['dq_source']='normal Gazebo paired q/dq reference state'
        report['limitations']+=['normal q/dq mixed with real IMU and action history is not a self-consistent physical feedback trajectory',
            'replacing observations does not change motor internal PD or identify load/actuator parameters',
            'single predeclared command-onset alignment; no gait phase optimization']
    if shank_metadata:
        report['shank_position_pull']=shank_metadata
        report['limitations']+=['same-tick original target is hindsight information from the original forward; this is a retrospective sensitivity experiment, not a deployable causal observation',
            'only observed shank q is pulled; dq and action history remain original, so the modified sequence is not physically self-consistent',
            'no change to Gazebo internal PD, actual real actuator strength or real hardware']
    if a.verify_all_pairs:
        original_inputs=torch.tensor([[r[f'obs_{i:02}'] for i in range(45)] for r in rows],dtype=torch.float32)
        logged=torch.tensor([[r[f'act_{i:02}'] for i in range(12)] for r in rows],dtype=torch.float32)
        with torch.inference_mode():
            original_outputs=model(original_inputs)
            original_outputs=torch.maximum(torch.tensor(cfg['lower']),torch.minimum(torch.tensor(cfg['upper']),original_outputs))
        error=original_outputs-logged
        report['full_log_pair_validation']={'policy_rows':len(rows),'action_values':error.numel(),
            'action_rmse':error.square().mean().sqrt().item(),'action_max_abs_error':error.abs().max().item(),
            'values_over_2e_5':int((error.abs()>2e-5).sum()),
            'method':'all original paired rows, float32 CPU actor then configured action clip; no lag fit'}
    describe_group_replacement(report)
    (a.output/'inference_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--policy',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--start',type=float,default=22);p.add_argument('--duration',type=float,default=14)
    p.add_argument('--smooth-dq-hz',type=float,default=0.);p.add_argument('--smooth-roll-hz',type=float,default=0.);p.add_argument('--smooth-gyro-hz',type=float,default=0.)
    p.add_argument('--self-last-action',action='store_true',help='Diagnostic actor recurrence: feed its own last clipped output, other logged observations retained')
    p.add_argument('--dq-from-position',action='store_true',help='Diagnostic replace dq with causal position finite difference at logged policy times')
    p.add_argument('--stats-window',action='append',default=[],help='Additional source-time statistics: label,start_s,end_s')
    p.add_argument('--hip-dq-gain',type=float,default=1.,help='Simulation diagnostic only: scale four hip dq obs about zero, default unchanged')
    p.add_argument('--hip-q-gain',type=float,default=1.,help='Simulation diagnostic only: scale four hip q observations relative to default pose; optionally with body donor')
    p.add_argument('--obs-donor',type=Path,help='Simulation-only grouped replacement from captured normal policy observations')
    p.add_argument('--obs-group',choices=('body','hip_state','hip_history','combined','all_joint_state'))
    p.add_argument('--obs-donor-start',type=float,default=13.,help='Fixed normal source time corresponding to real replay start; no fitted phase')
    p.add_argument('--verify-all-pairs',action='store_true',help='Also check actor output against every original paired row in the complete source log')
    p.add_argument('--shank-target-pull',type=float,default=0.,help='Simulation-only retrospective q input pull toward ORIGINAL same-tick q_des; 0 off, .5 halves four shank position errors')
    p.add_argument('--gravity-donor',type=Path,help='Normal Gazebo directory: replace ONLY gravity from its IMU quaternion; simulation diagnostic only')
    p.add_argument('--gravity-real-onset',type=float)
    p.add_argument('--gravity-normal-onset',type=float)
    p.add_argument('--gravity-donor-min',type=float,default=9.9,help='Hold reference orientation before this quiet donor time')
    p.add_argument('--joint-donor',type=Path,help='Normal Gazebo directory: replace only q/dq obs[9:33]; keep IMU and other inputs real')
    p.add_argument('--joint-real-onset',type=float)
    p.add_argument('--joint-normal-onset',type=float)
    p.add_argument('--joint-donor-min',type=float,default=9.9,help='Hold normal quiet q/dq reference before this donor time')
    a=p.parse_args()
    if not 0<=a.hip_q_gain<=1:p.error('hip q gain must be between 0 and 1')
    if a.hip_q_gain!=1.:
        if any((a.hip_dq_gain!=1.,a.gravity_donor,a.joint_donor,a.shank_target_pull,a.smooth_dq_hz,a.smooth_roll_hz,a.smooth_gyro_hz,a.self_last_action,a.dq_from_position)):
            p.error('hip q scaling permits only a separately declared body donor')
        if a.obs_donor and a.obs_group!='body':p.error('hip q scaling can combine only with body donor')
    if not 0<=a.hip_dq_gain<=1:p.error('hip dq gain must be between 0 and 1')
    if a.hip_dq_gain!=1. and any((a.obs_donor,a.gravity_donor,a.joint_donor,a.shank_target_pull,a.smooth_dq_hz,a.smooth_roll_hz,a.smooth_gyro_hz,a.self_last_action,a.dq_from_position)):
        p.error('hip velocity scaling cannot be combined with other interventions')
    if bool(a.obs_donor)!=bool(a.obs_group):p.error('group replacement requires donor and group')
    if a.obs_donor and any((a.gravity_donor,a.joint_donor,a.shank_target_pull,a.smooth_dq_hz,a.smooth_roll_hz,a.smooth_gyro_hz,a.self_last_action,a.dq_from_position)):
        p.error('group replacement cannot be combined with legacy interventions')
    if not 0<=a.shank_target_pull<=1:p.error('shank target pull must be in [0,1]')
    if a.shank_target_pull and any((a.gravity_donor,a.joint_donor,a.smooth_dq_hz,a.smooth_roll_hz,a.smooth_gyro_hz,a.self_last_action,a.dq_from_position)):
        p.error('shank-position-only pull cannot be combined with other observation interventions')
    if not all(0<=v<20 for v in (a.smooth_dq_hz,a.smooth_roll_hz,a.smooth_gyro_hz)):p.error('filter cutoff must be 0 (off) or below 20Hz')
    if a.gravity_donor:
        if a.gravity_real_onset is None or a.gravity_normal_onset is None:p.error('gravity replacement requires both predeclared onset times')
        if any((a.smooth_dq_hz,a.smooth_roll_hz,a.smooth_gyro_hz,a.self_last_action,a.dq_from_position)):p.error('gravity-only replacement cannot be combined with other observation interventions')
    if a.joint_donor:
        if a.joint_real_onset is None or a.joint_normal_onset is None:p.error('joint replacement requires both predeclared onset times')
        if any((a.gravity_donor,a.smooth_dq_hz,a.smooth_roll_hz,a.smooth_gyro_hz,a.self_last_action,a.dq_from_position)):p.error('joint-only replacement cannot be combined with other observation interventions')
    main(a)
