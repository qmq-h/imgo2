#!/usr/bin/env python3
"""Audit complete-leg rack sine logs and compare Gazebo / Isaac responses.

Numerical core uses stdlib. No model fitting, controller edits or hardware.
Source filenames do not determine support: user confirms all tests on rack.
"""
import argparse
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics as st

from audit_joint_sine_log import fit,integrate,phase_difference,rms

JOINTS=[f'{l}_{k}' for l in ('FL','FR','BL','BR') for k in ('hip','thigh','shank')]
SIGNS=[1,1,1,1,-1,-1,-1,1,1,-1,-1,-1]


def load(path):
    with path.open() as f:return [{k:float(v) for k,v in r.items() if k not in ('leg','joint')} for r in csv.DictReader(f)]


def harmonic(t,y,f):
    if max(y)-min(y)<1e-12:return dict(mean=st.fmean(y),amplitude=0.,phase_deg=0.,residual_rms=0.,r_squared=None)
    return fit(t,y,f)


def metrics(d,f,reference_fit=None):
    fq=harmonic(d['t'],d['q'],f);fr=reference_fit if reference_fit is not None else harmonic(d['ref_t'],d['ref'],f)
    fv=harmonic(d['t'],d['dq'],f);fp=harmonic(d['t'],d['pd'],f)
    harmonics=[harmonic(d['t'],d['q'],k*f)['amplitude'] for k in range(2,6)]
    return dict(samples=len(d['t']),q_fit=fq,target_fit=fr,dq_fit=fv,pd_fit=fp,
        gain=fq['amplitude']/fr['amplitude'],phase_deg=phase_difference(fq['phase_deg'],fr['phase_deg']),
        mean_error=st.fmean(d['ref'])-st.fmean(d['q']),mean_dq=st.fmean(d['dq']),dq_rms=rms(d['dq']),
        q_change=d['q'][-1]-d['q'][0],dq_integral=integrate(d['t'],d['dq']),
        mean_pd=st.fmean(d['pd']),pd_rms=rms(d['pd']),pd_peak=max(map(abs,d['pd'])),
        mean_effort=st.fmean(d['effort']),effort_rms=rms(d['effort']),effort_peak=max(map(abs,d['effort'])),
        higher_harmonic_amplitudes=harmonics,
        harmonic_ratio_2_to_5=math.sqrt(sum(a*a for a in harmonics))/fq['amplitude'] if fq['amplitude']>1e-8 else None)


def build(source,case_id,gazebo_root,isaac_root,unitree_root=None):
    rows=load(source);t0=rows[0]['send_mono_ms']/1000
    times=[r['send_mono_ms']/1000-t0 for r in rows]
    if min(b-a for a,b in zip(times,times[1:]))<=0:raise ValueError('Nonmonotonic send time')
    if not all(math.isfinite(v) for r in rows for v in r.values()):raise ValueError('Nonfinite real log')
    moving=[j for j in JOINTS if max(r['q_set_'+j] for r in rows)-min(r['q_set_'+j] for r in rows)>1e-5]
    if len(moving)!=1:raise ValueError('Expected single excited joint')
    joint=moving[0];index=JOINTS.index(joint)
    m=re.search(r'_([0-9.]+)Hz_([0-9.]+)\.log$',source.name);f=float(m[1]);amplitude=float(m[2])
    center=(max(r['q_set_'+joint] for r in rows)+min(r['q_set_'+joint] for r in rows))/2
    cross=[]
    for k in range(1,len(rows)):
        a=rows[k-1]['q_set_'+joint]-center;b=rows[k]['q_set_'+joint]-center
        if a<=0<b:cross.append(times[k-1]+(times[k]-times[k-1])*(-a)/(b-a))
    measured_f=1/st.median(b-a for a,b in zip(cross,cross[1:]))
    if abs(measured_f/f-1)>.005:raise ValueError('Frequency does not match source label')
    cycles=math.floor((times[-1]-1.25)*f);start=1.;end=start+cycles/f
    if cycles<2:raise ValueError('Fewer than two full usable periods')
    chosen=[r for r,t in zip(rows,times) if start<=t<end]
    pd_errors=[];axes=[]
    for j,sign in zip(JOINTS,SIGNS):
        if any(r['kp_'+j]!=25 or r['kd_'+j]!=.8 or r['dq_set_'+j]!=0 for r in rows):raise ValueError('Unexpected recorded gains / velocity target')
        axis_rows=[r for k,r in enumerate(chosen) if k==0 or r['recv_ms_'+j]!=chosen[k-1]['recv_ms_'+j]]
        rx=[r['recv_ms_'+j]/1000-t0 for r in axis_rows];q=[r['q_act_'+j] for r in axis_rows];v=[r['dq_act_'+j] for r in axis_rows]
        if min(b-a for a,b in zip(rx,rx[1:]))<=0:raise ValueError('Nonmonotonic receive endpoint')
        iv=integrate(rx,v);delta=q[-1]-q[0]
        axes.append(dict(joint=j,duplicate_receive_snapshots=len(chosen)-len(axis_rows),q_change=delta,dq_integral=iv,mean_dq=st.fmean(v),
            mean_error=st.fmean(r['q_set_'+j]-r['q_act_'+j] for r in chosen),
            mean_effort=st.fmean(r['tau_act_'+j] for r in chosen),
            mismatch_rate=(iv-delta)/(rx[-1]-rx[0]),direction_neutral_mismatch_rate=sign*(iv-delta)/(rx[-1]-rx[0])))
        pd_errors.extend(abs(r['tau_set_'+j]-(25*(r['q_set_'+j]-r['q_act_'+j])-.8*r['dq_act_'+j])) for r in rows)
    streams={}
    # A repeated host snapshot is not a new motor sample.
    feedback_rows=[r for k,r in enumerate(chosen) if k==0 or r['recv_ms_'+joint]!=chosen[k-1]['recv_ms_'+joint]]
    selected_commands=chosen
    chosen=feedback_rows
    streams['real']=dict(t=[r['recv_ms_'+joint]/1000-t0 for r in chosen],
        ref_t=[r['send_mono_ms']/1000-t0 for r in chosen],q=[r['q_act_'+joint] for r in chosen],
        dq=[r['dq_act_'+joint] for r in chosen],ref=[r['q_set_'+joint] for r in chosen],
        pd=[25*(r['q_set_'+joint]-r['q_act_'+joint])-.8*r['dq_act_'+joint] for r in chosen],
        effort=[r['tau_act_'+joint] for r in chosen])
    base=dict(id=case_id,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        joint=joint,frequency_hz=f,measured_target_frequency_hz=measured_f,amplitude_rad=amplitude,
        reference_center_rad=center,duration_s=times[-1],rows=len(rows),window_s=[start,end],cycles=cycles,
        host_max_gap_ms=max((b-a)*1000 for a,b in zip(times,times[1:])),
        receive_age_ms_mean=st.fmean(r['t_mono_ms']-r['recv_ms_'+joint] for r in chosen),
        pd_formula_max_difference=max(pd_errors),all_axes=axes)
    if gazebo_root:
        folder=gazebo_root/f'case_{case_id:02d}'
        meta=json.loads((folder/'sine_metadata.json').read_text())
        if meta['source_sha256']!=base['source_sha256']:raise ValueError('Gazebo source identity mismatch')
        raw=load(folder/'run/replay.csv');targets=load(folder/'run/targets.csv');tt=[r['t_s'] for r in targets];warm=meta['warmup_s']
        raw=[r for r in raw if warm+start<=r['t_s']<warm+end and math.isfinite(r[f'controller_q_{index:02}'])]
        commands=[targets[max(0,bisect.bisect_right(tt,r['t_s'])-1)] for r in raw]
        q=[r[f'controller_q_{index:02}'] for r in raw];v=[r[f'controller_dq_{index:02}'] for r in raw]
        t=[r['t_s']-warm for r in raw];ref=[r[f'q_{index:02}'] for r in commands]
        streams['gazebo']=dict(t=t,ref_t=t,q=q,dq=v,ref=ref,
            pd=[25*(a-b)-.8*c for a,b,c in zip(ref,q,v)],effort=[r[f'controller_tau_{index:02}'] for r in raw])
        base['gazebo_production_files_unchanged']=meta['production_files_unchanged']
        base['gazebo_base_height_range']=[min(r['z'] for r in raw),max(r['z'] for r in raw)]
    if isaac_root:
        meta=json.loads((isaac_root/'metadata.json').read_text())
        if meta['sources'][case_id]['sha256']!=base['source_sha256']:raise ValueError('Isaac source identity mismatch')
        for variant,kd in [('matched_kd08',.8),('training_kd05',.5)]:
            raw=load(isaac_root/f'case_{case_id:02d}_{variant}.csv');raw=[r for r in raw if start<=r['test_s']<end]
            t=[r['test_s'] for r in raw];q=[r[f'q_{index:02}'] for r in raw];v=[r[f'dq_{index:02}'] for r in raw]
            ref=[r[f'q_des_{index:02}'] for r in raw]
            streams['isaac_'+variant]=dict(t=t,ref_t=[x-meta['dt_s'] for x in t],q=q,dq=v,ref=ref,
                pd=[r[f'tau_computed_{index:02}'] for r in raw],effort=[r[f'tau_applied_{index:02}'] for r in raw])
        base['isaac_production_files_unchanged']=meta['production_files_unchanged']
    if unitree_root:
        meta=json.loads((unitree_root/'metadata.json').read_text())
        if meta['sources'][case_id]['sha256']!=base['source_sha256']:raise ValueError('Go2HV source identity mismatch')
        raw=load(unitree_root/f'case_{case_id:02d}_matched_kd08.csv');raw=[r for r in raw if start<=r['test_s']<end]
        t=[r['test_s'] for r in raw]
        streams['isaac_go2hv_kd08']=dict(t=t,ref_t=[x-meta['dt_s'] for x in t],
            q=[r[f'q_{index:02}'] for r in raw],dq=[r[f'dq_{index:02}'] for r in raw],ref=[r[f'q_des_{index:02}'] for r in raw],
            pd=[r[f'tau_computed_{index:02}'] for r in raw],effort=[r[f'tau_applied_{index:02}'] for r in raw])
        base['unitree_production_files_unchanged']=meta['production_files_unchanged']
    reference_fit=harmonic([r['send_mono_ms']/1000-t0 for r in selected_commands],[r['q_set_'+joint] for r in selected_commands],f)
    base['streams']={key:metrics(d,f,reference_fit) for key,d in streams.items()}
    cycle_fits=[];real=streams['real']
    for k in range(cycles):
        lo=start+k/f;hi=lo+1/f;ii=[i for i,t in enumerate(real['t']) if lo<=t<hi]
        h=harmonic([real['t'][i] for i in ii],[real['q'][i] for i in ii],f)
        cycle_fits.append(dict(window_s=[lo,hi],mean_q=h['mean'],gain=h['amplitude']/reference_fit['amplitude'],r_squared=h['r_squared']))
    base['real_cycle_fits']=cycle_fits
    gains=[x['gain'] for x in cycle_fits]
    base['event_windows_s']=[x['window_s'] for x in cycle_fits if x['gain']>2*st.median(gains) or (x['r_squared'] is not None and x['r_squared']<.6)]
    base['steady_usable']=base['streams']['real']['q_fit']['r_squared']>.6 and max(gains)<2*max(min(gains),1e-8)
    base['quality_note']='One recording only; no independent repeats' if base['steady_usable'] else 'Nonstationary extra motion; whole-window fit is descriptive, excluded from steady response line'
    return base,streams


def plots(cases,traces,prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors={'real':'tab:blue','gazebo':'tab:orange','isaac_matched_kd08':'tab:green','isaac_training_kd05':'tab:red','isaac_go2hv_kd08':'magenta'}
    names={'real':'Real 25/.8','gazebo':'Gazebo 25/.8','isaac_matched_kd08':'Isaac DC 25/.8','isaac_training_kd05':'Isaac DC 25/.5','isaac_go2hv_kd08':'Isaac Go2HV 25/.8'}
    fig,axs=plt.subplots(3,2,figsize=(15,12),constrained_layout=True)
    for row,joint in enumerate(['FL_hip','FL_thigh','FL_shank']):
        group=[c for c in cases if c['joint']==joint]
        for key in group[0]['streams']:
            for amp in sorted(set(c['amplitude_rad'] for c in group)):
                cc=sorted([c for c in group if c['amplitude_rad']==amp and (key!='real' or c['steady_usable'])],key=lambda c:c['frequency_hz'])
                for col,metric,scale in [(0,'gain',100),(1,'phase_deg',1)]:
                    axs[row,col].plot([c['frequency_hz'] for c in cc],[c['streams'][key][metric]*scale for c in cc],
                        '-' if amp==.01 else '--',marker='o',color=colors[key],label=f'{names[key]} A={amp:g}')
        for col,ylabel in [(0,'Fundamental amplitude / target (%)'),(1,'Phase relative to target (deg)')]:
            axs[row,col].set(title=joint,xlabel='Excitation frequency (Hz)',ylabel=ylabel,xticks=[.5,1,2]);axs[row,col].grid(alpha=.25)
        axs[row,0].legend(fontsize=7,ncol=2)
        if any(not c['steady_usable'] for c in group):axs[row,0].text(.03,.04,'Real transient window(s) excluded',transform=axs[row,0].transAxes,fontsize=8,color='purple')
    fig.suptitle('Complete-leg rack tests: local frequency response | no parameter tuning',fontsize=14)
    fig.savefig(str(prefix)+'_frequency.png',dpi=150);plt.close(fig)
    for joint in ['FL_hip','FL_thigh','FL_shank']:
        group=sorted([c for c in cases if c['joint']==joint],key=lambda c:(c['amplitude_rad'],c['frequency_hz']))
        fig,axs=plt.subplots(len(group),3,figsize=(17,3*len(group)),constrained_layout=True)
        for row,c in enumerate(group):
            streams=traces[c['id']]
            for key,d in streams.items():
                axs[row,0].plot(d['t'],[q-st.fmean(d['q']) for q in d['q']],color=colors[key],label=names[key],alpha=.85)
                axs[row,1].plot(d['t'],d['dq'],color=colors[key],label=names[key],alpha=.7,lw=.8)
                axs[row,2].plot(d['t'],d['pd'],color=colors[key],label=names[key],alpha=.85,lw=.9)
            d=streams['real'];axs[row,0].plot(d['ref_t'],[v-st.fmean(d['ref']) for v in d['ref']],'k--',label='Target AC',lw=.9)
            axs[row,2].plot(d['t'],d['effort'],color='gray',alpha=.55,lw=.7,label='Real estimated effort')
            for col,label in [(0,'q minus own mean (rad)'),(1,'dq (rad/s)'),(2,'Nominal PD torque (N m)')]:
                ax=axs[row,col];ax.set(title=f"{joint} | {c['frequency_hz']:g}Hz A={c['amplitude_rad']:g} | {c['cycles']} periods",xlabel='Time from first send (s)',ylabel=label);ax.grid(alpha=.25)
            if row==0:
                axs[row,0].legend(fontsize=7);axs[row,2].legend(fontsize=7)
            if not c['steady_usable']:
                for ax in axs[row,:]:
                    for lo,hi in c['event_windows_s']:ax.axvspan(lo,hi,color='red',alpha=.09)
        fig.suptitle('Same recorded commands | means removed only for angle-amplitude display; no phase alignment',fontsize=13)
        fig.savefig(str(prefix)+'_'+joint+'.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(3,5,figsize=(20,11),constrained_layout=True)
    for row,joint in enumerate(['FL_hip','FL_thigh','FL_shank']):
        group=sorted([c for c in cases if c['joint']==joint],key=lambda c:(c['amplitude_rad'],c['frequency_hz']))
        for col,c in enumerate(group):
            ax=axs[row,col]
            for key,d in traces[c['id']].items():
                t=np.asarray(d['t']);q=np.asarray(d['q']);dt=.005;grid=np.arange(t[0],t[-1],dt)
                y=np.interp(grid,t,q);y-=y.mean();win=np.hanning(len(y));psd=np.abs(np.fft.rfft(y*win))**2*dt/(win@win);psd[1:-1]*=2
                f=np.fft.rfftfreq(len(y),dt);ax.semilogy(f,np.maximum(psd,1e-16),color=colors[key],label=names[key])
            ax.set(title=f"{joint} {c['frequency_hz']:g}Hz A{c['amplitude_rad']:g}"+(' [transient]' if not c['steady_usable'] else ''),xlim=(0,10),xlabel='Hz',ylabel='q PSD rad²/Hz');ax.grid(alpha=.25)
    axs[0,0].legend(fontsize=7);fig.suptitle('Position spectra: each continuous integer-period window, own mean removed, Hann window',fontsize=14)
    fig.savefig(str(prefix)+'_spectra.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(3,2,figsize=(15,11),constrained_layout=True)
    for row,joint in enumerate(['FL_hip','FL_thigh','FL_shank']):
        group=sorted([c for c in cases if c['joint']==joint],key=lambda c:(c['amplitude_rad'],c['frequency_hz']))
        for ki,key in enumerate(group[0]['streams']):
            for col,metric,scale in [(0,'mean_error',180/math.pi),(1,'mean_pd',1)]:
                count=len(group[0]['streams']);width=.8/count
                axs[row,col].bar([i+(ki-(count-1)/2)*width for i in range(5)],[c['streams'][key][metric]*scale for c in group],width=width,color=colors[key],label=names[key])
        for col,label in [(0,'Mean target - q (deg)'),(1,'Mean nominal PD (N m)')]:
            axs[row,col].set_xticks(range(5),[f"{c['frequency_hz']:g}Hz\nA{c['amplitude_rad']:g}" for c in group]);axs[row,col].set(title=joint,ylabel=label);axs[row,col].grid(axis='y',alpha=.25)
        axs[row,0].legend(fontsize=7)
    fig.suptitle('Mean/static error is separate from amplitude tracking',fontsize=14)
    fig.savefig(str(prefix)+'_static.png',dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,required=True);p.add_argument('--gazebo-root',type=Path);p.add_argument('--isaac-root',type=Path)
    p.add_argument('--unitree-root',type=Path,help='Official UnitreeActuator Go2HV diagnostic comparison')
    p.add_argument('--output-prefix',type=Path,required=True);p.add_argument('--plot',action='store_true')
    a=p.parse_args();cases=[];traces={}
    for i,source in enumerate(sorted(a.source_dir.glob('*.log'))):
        c,d=build(source,i,a.gazebo_root,a.isaac_root,a.unitree_root);cases.append(c);traces[i]=d
    result=dict(cases=cases,user_support='All rack, no ground; filename g1/g2 is not a support label',
        limitations=['Single recording per condition; no independent three-repeat uncertainty estimate.',
            'No pre-switch 60/2 rows; no ground sinking experiment in this archive.',
            'Assumed level rigidly fixed base, unmeasured actual fixture orientation/compliance.',
            'Real receive clocks are host endpoints; SDK fields are converted joint-side values.',
            'Gazebo uses controller position-difference dq; Isaac uses PhysX dq; real dq consistency unverified.',
            'Go2HV reference uses official preset on our legs; not identified GO-M8010 parameters.',
            'No fitted parameters, policy, DR or production actuator-model replacement.'])
    a.output_prefix.parent.mkdir(parents=True,exist_ok=True);a.output_prefix.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    with a.output_prefix.with_suffix('.csv').open('w',newline='') as f:
        fields=['id','joint','frequency_hz','amplitude_rad','reference_center_rad','cycles','steady_usable','stream','gain','phase_deg','mean_error','mean_dq','mean_pd','pd_peak','mean_effort','effort_peak','harmonic_ratio_2_to_5']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for c in cases:
            for key,m in c['streams'].items():w.writerow({**{k:c[k] for k in fields if k in c},'stream':key,**{k:m[k] for k in fields if k in m}})
    if a.plot:plots(cases,traces,a.output_prefix)
    for c in cases:
        print(c['id'],c['joint'],c['frequency_hz'],c['amplitude_rad'],'cycles',c['cycles'],{k:(round(m['gain'],3),round(m['phase_deg'],1)) for k,m in c['streams'].items()})


if __name__=='__main__':main()
