#!/usr/bin/env python3
"""Stdlib numeric comparison of real wide sine logs and Gazebo target replay.

Uses the original controller's position-difference velocity for PD analysis;
also audits physical GetVelocity separately. Optional Matplotlib figures.
"""
import argparse
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st

from audit_joint_sine_log import fit, phase_difference, integrate, rms

JOINTS=[f'{l}_{k}' for l in ('FL','FR','BL','BR') for k in ('hip','thigh','shank')]


def load(path):
    with path.open() as f:return [{k:float(v) for k,v in r.items() if k not in ('leg','joint')} for r in csv.DictReader(f)]


def summarize(t,q,v,ref,tau,feedback,freq,reference_time=None):
    def harmonic(clock,y):
        if max(y)-min(y)<1e-12:
            return dict(mean=st.fmean(y),amplitude=0.,phase_deg=None,residual_rms=0.,r_squared=None)
        return fit(clock,y,freq)
    fq=harmonic(t,q);fr=harmonic(reference_time if reference_time is not None else t,ref)
    return dict(mean_q=st.fmean(q),mean_error=st.fmean(a-b for a,b in zip(ref,q)),
        error_rms=rms(a-b for a,b in zip(ref,q)),q_fit=fq,target_fit=fr,
        gain=fq['amplitude']/fr['amplitude'] if fr['amplitude']>1e-8 else None,
        phase_deg=phase_difference(fq['phase_deg'],fr['phase_deg']) if fr['amplitude']>1e-8 and fq['phase_deg'] is not None else None,
        mean_dq=st.fmean(v),dq_rms=rms(v),q_change=q[-1]-q[0],dq_integral=integrate(t,v),
        mean_pd=st.fmean(tau),pd_rms=rms(tau),mean_feedback_torque=st.fmean(feedback),
        feedback_torque_rms=rms(feedback),pd_fit=harmonic(t,tau),feedback_torque_fit=harmonic(t,feedback))


def compare(path,case,end,start=0.):
    real=load(path);meta=json.loads((case/'sine_metadata.json').read_text());warm=meta['warmup_s']
    sim=load(case/'run/replay.csv');targets=load(case/'run/targets.csv')
    t0=real[0]['send_mono_ms']/1000
    # Main comparison uses each feedback's receive endpoint relative to first send.
    # Source timestamps remain host endpoints, not motor device source timestamps.
    rt=[r['t_mono_ms']/1000-t0 for r in real]
    target_time=[r['t_s'] for r in targets]
    chosen_real=[r for r,t in zip(real,rt) if start<=t<end]
    chosen_sim=[r for r in sim if warm+start<=r['t_s']<warm+end and math.isfinite(r['controller_q_00'])]
    if len(chosen_real)<100 or len(chosen_sim)<100:raise ValueError('Insufficient valid samples')
    freq=1.
    axes=[];traces=[]
    for i,j in enumerate(JOINTS):
        tr=[r['recv_ms_'+j]/1000-t0 for r in chosen_real]
        reference_time=[r['send_mono_ms']/1000-t0 for r in chosen_real]
        qr=[r['q_act_'+j] for r in chosen_real];vr=[r['dq_act_'+j] for r in chosen_real]
        rr=[r['q_set_'+j] for r in chosen_real]
        pr=[r['kp_'+j]*(r['q_set_'+j]-r['q_act_'+j])+r['kd_'+j]*(r['dq_set_'+j]-r['dq_act_'+j]) for r in chosen_real]
        er=[r['tau_act_'+j] for r in chosen_real]
        ts=[r['t_s']-warm for r in chosen_sim]
        qs=[r[f'controller_q_{i:02}'] for r in chosen_sim];vs=[r[f'controller_dq_{i:02}'] for r in chosen_sim]
        commands=[targets[max(0,bisect.bisect_right(target_time,r['t_s'])-1)] for r in chosen_sim]
        rs=[r[f'q_{i:02}'] for r in commands]
        ps=[c[f'kp_{i:02}']*(c[f'q_{i:02}']-q)+c[f'kd_{i:02}']*(c[f'dq_{i:02}']-v)+c[f'tau_{i:02}'] for c,q,v in zip(commands,qs,vs)]
        es=[r[f'controller_tau_{i:02}'] for r in chosen_sim]
        axis=dict(joint=j,real=summarize(tr,qr,vr,rr,pr,er,freq,reference_time),
                  gazebo=summarize(ts,qs,vs,rs,ps,es,freq))
        axis['gazebo']['physical_dq_mean']=st.fmean(r[f'dq_{i:02}'] for r in chosen_sim)
        axis['gazebo']['physical_dq_integral']=integrate(ts,[r[f'dq_{i:02}'] for r in chosen_sim])
        axes.append(axis)
        traces.append(dict(joint=j,real=dict(t=tr,ref_t=reference_time,q=qr,dq=vr,ref=rr,pd=pr,effort=er),
                           gazebo=dict(t=ts,q=qs,dq=vs,ref=rs,pd=ps,effort=es)))
    pub=load(case/'run/published.csv');errors=[r['t_s']-r['scheduled_s'] for r in pub]
    # Verify the converted tail is exactly the source q/dq/Kp/Kd, FF remains zero.
    original=[r for r in real if r['send_mono_ms']/1000-t0<=meta['real_duration_s']+1e-8]
    tail=[r for r in targets if r['t_s']>=warm-1e-9]
    if len(original)!=len(tail):raise ValueError('Converted target count mismatch')
    maxdiff=0.
    for r,c in zip(original,tail):
        maxdiff=max(maxdiff,abs(c['t_s']-warm-(r['send_mono_ms']/1000-t0)))
        for i,j in enumerate(JOINTS):
            for target,raw in [('q','q_set'),('dq','dq_set'),('kp','kp'),('kd','kd')]:
                maxdiff=max(maxdiff,abs(c[f'{target}_{i:02}']-r[raw+'_'+j]))
            if c[f'tau_{i:02}']!=0:raise ValueError('PD total incorrectly used as FF')
    result=dict(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),case=str(case),
        condition=meta['condition'],window_s=[start,end],warmup_s=warm,axes=axes,
        real_samples=len(chosen_real),gazebo_samples=len(chosen_sim),
        target_conversion_max_absolute_difference=maxdiff,published_count=len(pub),target_count=len(targets),
        publication_max_lateness_s=max(errors),published_indices_ordered=[r['index'] for r in pub]==list(range(len(targets))),
        production_files_unchanged=meta['production_files_unchanged'],
        base_height_range_m=[min(r['z'] for r in chosen_sim),max(r['z'] for r in chosen_sim)],
        rack_base_max_position_drift_m=max(math.sqrt(sum((r[k]-chosen_sim[0][k])**2 for k in ('x','y','z'))) for r in chosen_sim),
        assumptions=meta['assumptions'])
    return result,traces


def plot(cases,traces,prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'real':'tab:blue','gazebo':'tab:orange'}
    fig,axs=plt.subplots(4,2,figsize=(15,12),constrained_layout=True)
    for col,(case,tr) in enumerate(zip(cases[:2],traces[:2])):
        j=tr[1]
        for key in ('real','gazebo'):
            r=j[key];a=axs[0,col];a.plot(r['t'],r['q'],label=key,color=colors[key])
            axs[1,col].plot(r['t'],[v-st.fmean(r['q']) for v in r['q']],label=key,color=colors[key])
            axs[2,col].plot(r['t'],r['dq'],label=key,color=colors[key],alpha=.85)
            axs[3,col].plot(r['t'],r['pd'],label=key+' nominal PD',color=colors[key])
        axs[0,col].plot(j['real']['ref_t'],j['real']['ref'],'k--',label='same q target')
        axs[1,col].plot(j['real']['ref_t'],[v-st.fmean(j['real']['ref']) for v in j['real']['ref']],'k--',label='target AC')
        axs[3,col].plot(j['real']['t'],j['real']['effort'],color='tab:green',alpha=.65,label='real estimated torque')
        for row,label in enumerate(['FL thigh angle (rad)','Angle minus mean (rad)','Joint speed (rad/s)','Torque (N m)']):
            ax=axs[row,col];ax.set(ylabel=label,xlabel='Time relative to first test send (s)');ax.grid(alpha=.25);ax.legend(fontsize=8)
        axs[0,col].set_title(case['condition']+' | Kp25 Kd0.8 | 1 Hz +/-0.01 rad')
    fig.suptitle('Same recorded commands: real vs Gazebo | synthetic 3 s settling excluded',fontsize=14)
    fig.savefig(str(prefix)+'.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(15,9),constrained_layout=True)
    for col,c in enumerate(cases[:2]):
        for offset,key in [(-.18,'real'),(.18,'gazebo')]:
            axs[0,col].bar([i+offset for i in range(12)],[a[key]['mean_error']*180/math.pi for a in c['axes']],width=.36,label=key)
            axs[1,col].bar([i+offset for i in range(12)],[a[key]['mean_pd'] for a in c['axes']],width=.36,label=key)
        for row,label in enumerate(['Mean target - q (deg)','Mean nominal PD torque (N m)']):
            ax=axs[row,col];ax.set_xticks(range(12),[j.replace('_','\n') for j in JOINTS],fontsize=8);ax.set_ylabel(label);ax.grid(axis='y',alpha=.25);ax.legend();ax.set_title(c['condition'])
    fig.suptitle('All 12 joints: static error and nominal PD load | rear BL/BR = RL/RR',fontsize=14)
    fig.savefig(str(prefix)+'_all_axes.png',dpi=150);plt.close(fig)
    # Full per-leg time domain traces, original real estimates and simulator feedback retained in JSON metrics.
    for leg in range(4):
        fig,axs=plt.subplots(3,3,figsize=(15,10),constrained_layout=True)
        for kind in range(3):
            for case_index in range(2):
                a=traces[case_index][leg*3+kind]
                for key in ('real','gazebo'):
                    r=a[key];style='-' if case_index==0 else '--';label=cases[case_index]['condition']+' '+key
                    for row,field in enumerate(('q','dq','pd')):axs[row,kind].plot(r['t'],r[field],style,color=colors[key],alpha=.8,label=label)
            axs[0,kind].set_title(JOINTS[leg*3+kind])
            for row,unit in enumerate(('q rad','dq rad/s','nominal PD N m')):
                axs[row,kind].set(xlabel='Test-relative s',ylabel=unit);axs[row,kind].grid(alpha=.25)
        axs[0,0].legend(fontsize=7);fig.suptitle('Real vs Gazebo, ground solid / rack dashed')
        fig.savefig(str(prefix)+'_'+JOINTS[leg*3].split('_')[0]+'.png',dpi=150);plt.close(fig)
    if len(cases)>2:
        rows=load(Path(cases[2]['case'])/'run/replay.csv')
        rows=[r for r in rows if math.isfinite(r['controller_q_01'])]
        t=[r['t_s']-3 for r in rows]
        fig,axs=plt.subplots(3,1,figsize=(12,9),constrained_layout=True)
        axs[0].plot(t,[r['controller_q_01'] for r in rows],label='Gazebo FL thigh')
        axs[0].axhline(.862052,color='black',ls='--',label='approx target center')
        axs[0].set_ylabel('Joint angle (rad)')
        axs[1].plot(t,[r['z'] for r in rows],label='Gazebo base height');axs[1].set_ylabel('Height (m)')
        axs[2].plot(t,[r['controller_tau_11'] for r in rows],label='Gazebo BR shank effort');axs[2].set_ylabel('N m')
        for ax in axs:
            ax.axvline(0,color='red',ls='--',label='60/2 to 25/0.8');ax.set_xlabel('Time relative to simulated PD switch (s)');ax.grid(alpha=.25);ax.legend(fontsize=8)
        fig.suptitle('Synthetic Gazebo switch baseline | real pre-switch data absent | transient, not a frequency-response fit')
        fig.savefig(str(prefix)+'_switch.png',dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ground-source',type=Path,required=True);p.add_argument('--rack-source',type=Path,required=True)
    p.add_argument('--ground',type=Path,required=True);p.add_argument('--rack',type=Path,required=True)
    p.add_argument('--switch',type=Path);p.add_argument('--output-prefix',type=Path,required=True);p.add_argument('--plot',action='store_true')
    a=p.parse_args();cases=[];traces=[]
    for source,case,end in [(a.ground_source,a.ground,4.),(a.rack_source,a.rack,6.)]+([(a.ground_source,a.switch,4.)] if a.switch else []):
        c,t=compare(source,case,end)
        steady,_=compare(source,case,end,start=1.)
        c['exclude_first_second']=dict(window_s=[1.,end],axes=steady['axes'])
        c['harmonic_fit_interpretation']='Descriptive only: PD-switch transient is not stationary' if c['condition']=='switch' else 'Local 1 Hz response under assumed support; not broadband identification'
        cases.append(c);traces.append(t)
    result=dict(cases=cases,limitations=['Same commands, not identical unknown real contacts/fixture pose or mechanical calibration.',
        'Real feedback estimates are uncalibrated; simulator nominal PD uses controller q/dq, not physical GetVelocity.',
        '1 Hz local response only, not a full frequency response or PPO roll diagnosis.',
        'Ground simulated 3 s settling and switch 60/2 baseline are synthetic, absent in real file.',
        'Feedback receive clocks are host endpoints and controller feedback is asynchronously received.'])
    a.output_prefix.parent.mkdir(parents=True,exist_ok=True)
    a.output_prefix.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    if a.plot:plot(cases,traces,a.output_prefix)
    for c in cases:
        x=c['axes'][1];print(json.dumps(dict(condition=c['condition'],real=x['real'],gazebo=x['gazebo']),indent=2))


if __name__=='__main__':main()
