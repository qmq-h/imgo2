#!/usr/bin/env python3
"""Stdlib time-domain nominal PD/feedback comparison, not gain identification.

Never commands hardware. Target zero-order hold; no fitted lag compensation.
Optional figures require matplotlib.
"""
import argparse
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st
from audit_paired_obs_replay import load,quantile

LEGS=('FL','FR','RL','RR')
KINDS=('hip','thigh','shank')
GROUPS={name:[j*3+i for j in range(4)] for i,name in enumerate(KINDS)}


def before(times,t):
    j=bisect.bisect_right(times,t)-1
    if j<0:raise ValueError('no past sample')
    return j


def interp(rows,times,t,key):
    if not times[0]<=t<=times[-1]:raise ValueError('interpolation beyond coverage')
    j=min(len(times)-1,max(1,bisect.bisect_right(times,t)));dt=times[j]-times[j-1]
    if not 0<dt<.06:raise ValueError('nonmonotonic or missing samples')
    f=(t-times[j-1])/dt
    return rows[j-1][key]+f*(rows[j][key]-rows[j-1][key])


def construct(t,q,dq,target,desired_dq,ff,kp,kd,feedback):
    r={'t':t}
    for i in range(12):
        error=target[i]-q[i];p=kp[i]*error;d=kd[i]*(desired_dq[i]-dq[i]);total=p+d+ff[i]
        for k,v in {'q':q[i],'dq':dq[i],'target':target[i],'error':error,'P':p,'D':d,'ff':ff[i],
                    'nominal':total,'clipped':max(-23.7,min(23.7,total)),'feedback':feedback[i],
                    'kp':kp[i],'kd':kd[i],'residual':feedback[i]-total}.items():r[f'{k}_{i:02}']=v
    return r


def values(r,key):return [r[f'{key}_{i:02}'] for i in range(12)]


def rms(v):return math.sqrt(st.mean(x*x for x in v))


def metrics(rows):
    joints=[]
    for i in range(12):
        m={}
        for key in ('error','dq','P','D','ff','nominal','clipped','feedback','residual'):
            v=[r[f'{key}_{i:02}'] for r in rows]
            m[key]={'mean':st.mean(v),'rms':rms(v),'std':st.pstdev(v),'abs_p95':quantile([abs(x) for x in v],.95),'abs_max':max(map(abs,v)),
                    'peak_to_peak':max(v)-min(v)}
        p=[r[f'P_{i:02}'] for r in rows];d=[r[f'D_{i:02}'] for r in rows]
        m['mean_P_times_D']=st.mean(u*v for u,v in zip(p,d))
        m['nominal_D_power_W']=st.mean(r[f'D_{i:02}']*r[f'dq_{i:02}'] for r in rows)
        m['computed_over_23_7_fraction']=st.mean(abs(r[f'nominal_{i:02}'])>23.7 for r in rows)
        m['feedback_over_23_7_rows']=sum(abs(r[f'feedback_{i:02}'])>23.7 for r in rows)
        joints.append(m)
    groups={}
    for name,ids in GROUPS.items():
        groups[name]={key:{stat:math.sqrt(st.mean(joints[i][key][stat]**2 for i in ids)) for stat in ('rms','std')}
            for key in ('error','dq','P','D','nominal','clipped','feedback','residual')}
        groups[name]['nominal_peak_abs']=max(joints[i]['nominal']['abs_max'] for i in ids)
        groups[name]['feedback_peak_abs']=max(joints[i]['feedback']['abs_max'] for i in ids)
    return {'samples':len(rows),'joints':joints,'groups':groups}


def main(a):
    real=load(a.real/'robot.csv');normal_state=load(a.normal/'state.csv');normal_cmd=load(a.normal/'command.csv');odom=load(a.normal/'odom.csv')
    ht=[r['host_s'] for r in odom];normal_state=[r for r in normal_state if ht[0]<=r['host_s']<=ht[-1]]
    nt=[interp(odom,ht,r['host_s'],'sim_s') for r in normal_state];ct=[r['host_s'] for r in normal_cmd]
    rt=[r['t_mono_ms']*.001 for r in real]
    replay=load(a.replay/'replay.csv');meta=json.loads((a.replay/'metadata.json').read_text());pt=[r['t_s']+meta['source_start_s'] for r in replay]
    windows={'forward_mixed':(86.5,92.4,12.9),'pure_straight_short':(86.14,87.28,12.9)}
    datasets={};report={}
    for label,(lo,hi,nlo) in windows.items():
        cases={'real':[],'normal':[],'same_target_replay':[]}
        for j in range(round((hi-lo)*200)):
            t=lo+j/200;n=nlo+j/200
            r=real[before(rt,t)];rs=normal_state[before(nt,n)];cmd=normal_cmd[before(ct,rs['host_s'])]
            rr=construct(t,values(r,'q'),values(r,'dq'),values(r,'q_des'),values(r,'dq_des'),values(r,'tau_ff'),values(r,'kp'),values(r,'kd'),values(r,'tau'))
            nn=construct(n,values(rs,'q'),values(rs,'dq'),values(cmd,'q'),values(cmd,'dq'),values(cmd,'tau'),values(cmd,'kp'),values(cmd,'kd'),values(rs,'tau'))
            # Use exactly the same real command for both states. This removes
            # q_des as a variable in the descriptive difference of nominal PD.
            # Use the actual controller feedback for a consistent PD comparison.
            # Raw physics GetVelocity can retain constraint-solver velocities
            # even when q is steady, unlike the controller's position derivative.
            q=[interp(replay,pt,t,f'controller_q_{i:02}') for i in range(12)];dq=[interp(replay,pt,t,f'controller_dq_{i:02}') for i in range(12)]
            fb=[interp(replay,pt,t,f'controller_tau_{i:02}') for i in range(12)]
            ss=construct(t,q,dq,values(r,'q_des'),values(r,'dq_des'),values(r,'tau_ff'),values(r,'kp'),values(r,'kd'),fb)
            cases['real'].append(rr);cases['normal'].append(nn);cases['same_target_replay'].append(ss)
        for case,rows in cases.items():
            assert all(abs(r[f'kp_{i:02}']-25)<1e-4 and abs(r[f'kd_{i:02}']-.8)<1e-4 for r in rows for i in range(12)),case
        summaries={case:metrics(rows) for case,rows in cases.items()}
        paired=[]
        for i in range(12):
            delta=[r[f'nominal_{i:02}']-s[f'nominal_{i:02}'] for r,s in zip(cases['real'],cases['same_target_replay'])]
            dp=[r[f'P_{i:02}']-s[f'P_{i:02}'] for r,s in zip(cases['real'],cases['same_target_replay'])]
            dd=[r[f'D_{i:02}']-s[f'D_{i:02}'] for r,s in zip(cases['real'],cases['same_target_replay'])]
            paired.append({'nominal_difference_rmse':rms(delta),'P_difference_rms':rms(dp),'D_difference_rms':rms(dd),'nominal_difference_mean':st.mean(delta)})
        paired_groups={name:{k:math.sqrt(st.mean(paired[i][k]**2 for i in ids)) for k in ('nominal_difference_rmse','P_difference_rms','D_difference_rms')} for name,ids in GROUPS.items()}
        report[label]={'real_window_s':[lo,hi],'normal_window_s':[nlo,nlo+hi-lo],'cases':summaries,
            'same_target_descriptive_difference':paired,'same_target_descriptive_groups':paired_groups}
        datasets[label]=cases
    # Complement the host-command comparison with exact same-forward snapshot
    # nominal torques. These are prospective targets, not physically applied PD.
    cfg=json.loads((a.real/'meta.json').read_text())['config'];real_policy=load(a.real/'policy.csv')
    with (a.normal/'policy.csv').open() as f:normal_policy=[{k:float(v) for k,v in r.items() if k!='stage'} for r in csv.DictReader(f)]
    snapshots={}
    for case,rows,lo,hi in [('real',real_policy,86.5,92.4),('normal',normal_policy,12.9,18.8)]:
        out=[]
        for r in rows:
            t=r['input_captured_ms']*.001 if case=='real' else interp(odom,ht,r['host_s'],'sim_s')
            if not lo<=t<hi:continue
            q=[cfg['default_pose'][i]+r[f'obs_{9+i:02}']/cfg['obs_scales']['dof_pos'] for i in range(12)]
            dq=[r[f'obs_{21+i:02}']/cfg['obs_scales']['dof_vel'] for i in range(12)]
            target=[cfg['default_pose'][i]+cfg['action_scale'][i]*r[f'act_{i:02}'] for i in range(12)]
            out.append(construct(t,q,dq,target,[0.]*12,[0.]*12,[25.]*12,[.8]*12,[0.]*12))
        m=metrics(out)
        for group in m['groups'].values():group.pop('feedback');group.pop('residual');group.pop('feedback_peak_abs')
        for joint in m['joints']:joint.pop('feedback');joint.pop('residual');joint.pop('feedback_over_23_7_rows')
        snapshots[case]=m
    old=load(a.old_obs_replay/'targets.csv');old_original=load(a.old_obs_replay/'original_targets.csv');corrected=load(a.corrected_inference/'targets.csv')
    old_diff=[x[f'q_{i:02}']-y[f'q_{i:02}'] for x,y in zip(old,old_original) for i in range(12)]
    corrected_diff=[x[f'q_{i:02}']-y[f'q_{i:02}'] for x,y in zip(corrected,old_original) for i in range(12)]
    cadence_fix={'old_target_rmse_rad':rms(old_diff),'old_target_max_abs_rad':max(map(abs,old_diff)),
        'tick_corrected_target_rmse_rad':rms(corrected_diff),'tick_corrected_target_max_abs_rad':max(map(abs,corrected_diff)),
        'rows':len(corrected),'explanation':'same-tick policy row written after robot row; timestamp-asof delayed each new result by one submission row; exact tick pairing repairs diagnostic mapping, not physical latency'}
    exact_targets=load(a.replay/'targets.csv');sent=load(a.replay/'published.csv')
    checks={'raw_targets_equal_logged_source':exact_targets==old_original,
        'targets':len(exact_targets),'published':len(sent),
        'published_in_order':[int(r['index']) for r in sent]==list(range(len(exact_targets))),
        'max_publish_lag_s':max(r['t_s']-r['scheduled_s'] for r in sent),
        'production_files_unchanged':meta['production_files_unchanged'],'replay_completed':meta.get('completed',False),
        'tick_corrected_matches_logged_targets_to_1e_5_rad':max(map(abs,corrected_diff))<1e-5,
        'all_motion_gains_25_0_8':True}
    paths=[a.real/'robot.csv',a.real/'policy.csv',a.normal/'state.csv',a.normal/'command.csv',a.normal/'policy.csv',a.normal/'odom.csv',a.replay/'replay.csv',a.replay/'targets.csv',a.replay/'metadata.json']
    result={'formula':'P=kp*(q_des-q), D=kd*(dq_des-dq), nominal=P+D+tau_ff; clipped is hypothetical +/-23.7Nm comparison, not real PD saturation',
        'windows':report,'same_forward_snapshot_nominal':snapshots,'diagnostic_target_cadence_fix':cadence_fix,'checks':checks,
        'replay_metadata':meta,'limitations':['No fitted delay or gain regression. Larger torque is not evidence that a PD gain is too large.',
            'Normal vs real: independent trajectories/phases, and longer real window has lateral/yaw commands.',
            'Control-time stats: common 200Hz query grid; past logged feedback and zero-order-held command. Host receipt/write time is not motor source/application time.',
            'Same-target replay uses exact logged real targets and controller_q/controller_dq, matching the actual PD feedback definition. Paired PD differences remain descriptive; feedback is not synchronized or independently torque-calibrated.',
            'Real torque is protocol-estimated motor feedback; Gazebo torque is simulated effort; clipping computed torque does not implement real motor protection.',
            'Same-forward snapshots compute prospective torque from exact actor input/output; it is not necessarily the applied torque when the target reaches the motor.',
            'Do not use replay GetForce tau columns to infer physical torque; controller_tau is asynchronous simulated feedback.',
            'Earlier observation interventions used timestamp-asof target mapping; their inputs/actions remain valid, but exact submitted-cadence fidelity needs tick correction and rerun.'],
        'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    a.output_prefix.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with a.output_prefix.with_suffix('.csv').open('w') as f:
        keys=['window','case','joint','term','mean','rms','std','abs_p95','abs_max','peak_to_peak'];w=csv.DictWriter(f,keys);w.writeheader()
        for window,v in report.items():
            for case,m in v['cases'].items():
                for i,joint in enumerate(m['joints']):
                    for term in ('error','dq','P','D','nominal','clipped','feedback','residual'):
                        w.writerow({'window':window,'case':case,'joint':LEGS[i//3]+' '+KINDS[i%3],'term':term,**joint[term]})
    with Path(str(a.output_prefix)+'_traces.csv').open('w') as f:
        fields=['case','t','window_time_s']+list(datasets['forward_mixed']['real'][0])[1:];w=csv.DictWriter(f,fields);w.writeheader()
        for case,rows in datasets['forward_mixed'].items():
            for r in rows:w.writerow({'case':case,'window_time_s':r['t']-rows[0]['t'],**r})
    for case,m in report['forward_mixed']['cases'].items():print(case,json.dumps(m['groups']))
    print('paired',report['forward_mixed']['same_target_descriptive_groups']);print('cadence',cadence_fix)
    print('snapshots',{k:v['groups'] for k,v in snapshots.items()})
    if a.plot:plot(a,datasets)


def plot(a,datasets):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=datasets['forward_mixed'];colors={'normal':'#228833','real':'#0077bb','same_target_replay':'#ee7733'}
    def save(fig,name):fig.savefig(str(a.output_prefix)+'_'+name+'.png',dpi=160,bbox_inches='tight');plt.close(fig)
    for term,name,title in [('nominal','torque','Computed nominal PD torque'),('feedback','feedback','Logged feedback torque (different measurement models)')]:
        fig,ax=plt.subplots(4,3,figsize=(16,11),layout='constrained')
        for i,axis in enumerate(ax.flat):
            for case,rows in data.items():
                rr=rows[:400];axis.plot([r['t']-rows[0]['t'] for r in rr],[r[f'{term}_{i:02}'] for r in rr],label=case,color=colors[case],lw=.9)
            axis.set(title=LEGS[i//3]+' '+KINDS[i%3],xlabel='Window time (s)',ylabel='Nm');axis.grid(alpha=.25)
        ax[0,0].legend(fontsize=8);fig.suptitle(title+' | first 2s of 5.9s windows\nNormal phase is independent; real and exact-target replay share command schedule; no lag compensation')
        save(fig,name)
    fig,ax=plt.subplots(4,3,figsize=(16,11),layout='constrained')
    for leg in range(4):
        i=leg*3
        for col,term in enumerate(('P','D','nominal')):
            axis=ax[leg,col]
            for case,rows in data.items():
                rr=rows[:400];axis.plot([r['t']-rows[0]['t'] for r in rr],[r[f'{term}_{i:02}'] for r in rr],label=case,color=colors[case],lw=.9)
            axis.set(title=LEGS[leg]+' hip: '+term,ylabel='Nm',xlabel='Window time (s)');axis.grid(alpha=.25)
    ax[0,0].legend(fontsize=8);fig.suptitle('Hip proportional / damping / total components in time\nD = -0.8*dq; P and D RMS do not add linearly')
    save(fig,'hip_PD')
    for leg in range(4):
        fig,ax=plt.subplots(3,3,figsize=(16,9),layout='constrained')
        for joint in range(3):
            i=leg*3+joint
            for case in ('normal','real'):
                rows=data[case][:400];t=[r['t']-rows[0]['t'] for r in rows]
                ax[0,joint].plot(t,[r[f'q_{i:02}'] for r in rows],label=case+' q',color=colors[case])
                ax[0,joint].plot(t,[r[f'target_{i:02}'] for r in rows],label=case+' target',color=colors[case],ls='--',alpha=.6)
                ax[1,joint].plot(t,[r[f'dq_{i:02}'] for r in rows],label=case,color=colors[case])
                ax[2,joint].plot(t,[r[f'nominal_{i:02}'] for r in rows],label=case+' computed PD',color=colors[case])
                ax[2,joint].plot(t,[r[f'feedback_{i:02}'] for r in rows],label=case+' feedback',color=colors[case],ls=':',alpha=.65)
            ax[0,joint].set_title(KINDS[joint]);ax[0,joint].set_ylabel('q / target (rad)');ax[1,joint].set_ylabel('dq (rad/s)');ax[2,joint].set_ylabel('torque (Nm)')
        for axis in ax.flat:axis.grid(alpha=.25);axis.set_xlabel('Independent window time (s)')
        ax[0,0].legend(fontsize=7);ax[2,0].legend(fontsize=7);fig.suptitle(LEGS[leg]+' time-domain position, velocity and torque | P25/D0.8\nNo phase matching: normal 12.9s start, real 86.5s start. Feedback is not an independent torque calibration.')
        save(fig,LEGS[leg]+'_time')
    fig,ax=plt.subplots(4,3,figsize=(16,11),layout='constrained')
    for i,axis in enumerate(ax.flat):
        rr=data['real'][:400];ss=data['same_target_replay'][:400];t=[r['t']-rr[0]['t'] for r in rr]
        for term,style in [('nominal','-'),('P','--'),('D',':')]:axis.plot(t,[r[f'{term}_{i:02}']-s[f'{term}_{i:02}'] for r,s in zip(rr,ss)],label='real - replay '+term,ls=style,lw=.9)
        axis.set(title=LEGS[i//3]+' '+KINDS[i%3],ylabel='Nm',xlabel='Common source time relative to 86.5s');axis.grid(alpha=.25)
    ax[0,0].legend(fontsize=7);fig.suptitle('Same real target: nominal torque differences from q/dq response\nHost-state/source-time and actuator dynamics are not separated; no gain identification or phase shifting')
    save(fig,'same_target_difference')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('real','normal','replay','old-obs-replay','corrected-inference','output-prefix'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--plot',action='store_true');main(p.parse_args())
