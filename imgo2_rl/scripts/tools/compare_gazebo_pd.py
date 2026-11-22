#!/usr/bin/env python3
"""Compare isolated Gazebo PD runs; stdlib calculations, optional Matplotlib.

Full-feedback runs and fixed-history target replay are reported separately.
No hardware access, weight export, phase fitting, or parameter identification.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics as st

from analyze_command_history import map_host
from audit_paired_obs_replay import attitude, load, rms, sha, spectrum, summarize, wave

GROUPS={'hip':(0,3,6,9),'thigh':(1,4,7,10),'shank':(2,5,8,11)}
DEG=180/math.pi


def array(text,name):
    match=re.search(r'(?m)^\s*'+name+r':\s*\[([^\]]+)\]',text)
    if not match:raise ValueError('missing configuration array '+name)
    return [float(x.strip()) for x in match.group(1).split(',')]


def motion(rows,lo,hi):
    rr=[r for r in rows if lo<=r['t_s']<hi]
    return {'samples':len(rr),'body_vx_mean_mps':st.mean(r['body_vx'] for r in rr),
        'body_vy_mean_mps':st.mean(r['body_vy'] for r in rr),
        'height_mean_m':st.mean(r['z'] for r in rr),'height_min_m':min(r['z'] for r in rr),
        'max_tilt_deg':max(r['tilt_deg'] for r in rr),
        'yaw_change_deg':DEG*sum(math.atan2(math.sin(b['yaw_rad']-a['yaw_rad']),math.cos(b['yaw_rad']-a['yaw_rad'])) for a,b in zip(rr,rr[1:]))}


def closed(path,kd,lo,hi,kp=25.):
    meta=json.loads((path/'metadata.json').read_text())
    collection=json.loads((path/'collection.json').read_text())
    assert meta['completed'] and meta['sources_unchanged'] and collection['complete']
    assert meta['motor_extra_delay_ms']==meta['imu_extra_delay_ms']==0
    assert meta['profile_source'] is None and meta['command_vx']==.5
    assert collection['expected_ppo_kd']==kd and collection.get('expected_ppo_kp',25.)==kp
    runner=Path(meta['rl_sim_binary']);tree=runner.parent/'policy/imgo2'
    cfg=tree/'ppo/config.yaml';text=cfg.read_text()
    assert array(text,'rl_kp')==[kp]*12 and array(text,'rl_kd')==[kd]*12
    t0=collection['ppo_pd_start_sim_s']
    imu=load(path/'imu.csv');odom=load(path/'odom.csv');policy=load(path/'policy_raw.csv')
    for r in imu+odom:r['t_s']=r['sim_s']-t0;attitude(r)
    map_host(policy,odom)
    for r in policy:r['t_s']-=t0
    pp=[r for r in policy if lo<=r['t_s']<hi]
    assert len(pp)>150
    assert all(r['cmd_vx']==.5 and r['cmd_vy']==r['cmd_wz']==0 for r in pp)
    assert all(abs(r[f'kp_{i}']-kp)<1e-6 and abs(r[f'kd_{i}']-kd)<1e-6 for r in pp for i in range(12))
    command=load(path/'command.csv');map_host(command,odom)
    cc=[r for r in command if lo<=r['t_s']-t0<hi]
    assert len(cc)>300
    assert all(abs(r[f'kp_{i:02}']-kp)<1e-6 and abs(r[f'kd_{i:02}']-kd)<1e-6 for r in cc for i in range(12))
    assert all(math.isfinite(v) for rows in (imu,odom,policy) for r in rows for v in r.values())
    stats=summarize(imu,lo,hi);stats['motion']=motion(odom,lo,hi)
    stats['normal_walking_valid']=stats['max_tilt_deg']<=20 and stats['motion']['height_min_m']>.2
    active=[r for r in imu if 0<=r['t_s']<collection['hold_seconds']]
    stats['whole_ppo_max_tilt_deg']=max(r['tilt_deg'] for r in active)
    stats['groups']={}
    for kind,ids in GROUPS.items():
        stats['groups'][kind]={
            'position_error_rms_deg':DEG*rms([r[f'target_{i}']-r[f'state_q_{i}'] for r in pp for i in ids]),
            'velocity_rms_rad_s':rms([r[f'state_dq_{i}'] for r in pp for i in ids]),
            'nominal_pd_rms_Nm':rms([kp*(r[f'target_{i}']-r[f'state_q_{i}'])-kd*r[f'state_dq_{i}'] for r in pp for i in ids]),
            'feedback_tau_rms_Nm':rms([r[f'state_tau_est_{i}'] for r in pp for i in ids]),
            'target_step_rms_rad':rms([b[f'target_{i}']-a[f'target_{i}'] for a,b in zip(pp,pp[1:]) for i in ids])}
    files=[path/n for n in ('metadata.json','collection.json','imu.csv','odom.csv','policy_raw.csv','command.csv')]
    files += [cfg,tree/'ppo/policy.pt',tree/'base.yaml',runner.parent/'build_metadata.json']
    return {'path':str(path),'kp':kp,'kd':kd,'window_s_after_ppo':[lo,hi],'metrics':stats,
        'checks':{'actual_actor_kp':kp,'actual_actor_kd':kd,'actual_actor_samples':len(pp),
                  'actual_actor_command':[.5,0,0],'actual_ros_pd_commands':len(cc),
                  'actual_ros_kp':kp,'actual_ros_kd':kd,'complete':True,'sources_unchanged':True},
        'policy_sha256':sha(tree/'ppo/policy.pt'),'base_sha256':sha(tree/'base.yaml'),
        'config_except_pd':re.sub(r'(?m)^(\s*rl_k[dp]:\s*)\[[^\]]+\]',r'\1[IGNORED]',text),
        'model_sha256':meta['files']['imgo2_description/urdf/imgo2_real.gazebo.urdf'],
        'sha256':{str(p):sha(p) for p in files}},(imu,odom,policy)


def fixed_replay(before,after,lo,hi,kp=25.,kd=1.,label='kd10'):
    bm=json.loads((before/'metadata.json').read_text());am=json.loads((after/'metadata.json').read_text())
    assert bm['completed'] and am['completed'] and bm['production_files_unchanged'] and am['production_files_unchanged']
    assert bm['protected_hashes']==am['protected_hashes'] and bm['plugin_sha256']==am['plugin_sha256']
    assert (before/'initial.csv').read_bytes()==(after/'initial.csv').read_bytes()
    b=load(before/'targets.csv');a=load(after/'targets.csv')
    assert len(b)==len(a)==1600
    for x,y in zip(b,a):
        for k in x:
            if k.startswith('kd_'):assert x[k]==.8 and y[k]==kd
            elif k.startswith('kp_'):assert x[k]==25. and y[k]==kp
            else:assert x[k]==y[k],k
    result={};data={}
    for case_label,path,meta in [('kd08',before,bm),(label,after,am)]:
        rows=load(path/'replay.csv')
        for r in rows:r['t_s']+=meta['source_start_s'];attitude(r)
        assert all(math.isfinite(v) for r in rows for v in r.values())
        sent=load(path/'published.csv')
        assert len(sent)==1600 and [int(r['index']) for r in sent]==list(range(1600))
        m=summarize(rows,lo,hi)
        m['normal_walking_valid']=m['upright'] and m['max_tilt_deg']<=20
        full=[r for r in rows if meta['source_start_s']<=r['t_s']<=meta['source_start_s']+meta['duration_s']]
        m['whole_replay_max_tilt_deg']=max(r['tilt_deg'] for r in full)
        m['whole_replay_min_height_m']=min(r['z'] for r in full)
        m['first_tilt_over_60_source_s']=next((r['t_s'] for r in full if r['tilt_deg']>60),None)
        m['whole_replay_fall_detected']=m['first_tilt_over_60_source_s'] is not None
        m['normal_walking_valid']=m['normal_walking_valid'] and not m['whole_replay_fall_detected']
        m['published_targets']=len(sent)
        m['max_publish_lag_s']=max(r['t_s']-r['scheduled_s'] for r in sent)
        m['kp']=25. if case_label=='kd08' else kp;m['kd']=.8 if case_label=='kd08' else kd
        result[case_label]=m;data[case_label]=rows
    return {'paths':{'kd08':str(before),label:str(after)},'window_source_s':[lo,hi],
        'only_pd_gains_changed':True,'same_targets_and_initial_state':True,'metrics':result,
        'sha256':{str(p/n):sha(p/n) for p in (before,after) for n in ('metadata.json','targets.csv','replay.csv','published.csv')}},data


def gap_run(path,lo,hi):
    """Retain a completed run's native statistics without bridging missing IMU."""
    meta=json.loads((path/'metadata.json').read_text())
    collection=json.loads((path/'collection.json').read_text())
    assert meta['completed'] and meta['sources_unchanged'] and collection['complete']
    t0=collection['ppo_pd_start_sim_s'];rows=load(path/'imu.csv')
    for r in rows:r['t_s']=r['sim_s']-t0;attitude(r)
    gaps=[[a['t_s'],b['t_s'],b['t_s']-a['t_s']] for a,b in zip(rows,rows[1:])
          if a['t_s']<hi and b['t_s']>=lo and b['t_s']-a['t_s']>.05]
    assert gaps,'This path is only for an explicitly reported data-gap run'
    rr=[r for r in rows if lo<=r['t_s']<hi]
    return {'path':str(path),'kp':collection.get('expected_ppo_kp',25.),'kd':collection['expected_ppo_kd'],'window_s_after_ppo':[lo,hi],
        'imu_gaps_over_50ms_s':gaps,'native_only_statistics':{'samples':len(rr),
            'gyro_x_rms_rad_s':rms([r['gyro_x'] for r in rr]),
            'roll_std_deg':st.pstdev(r['roll_deg'] for r in rr),'max_tilt_deg':max(r['tilt_deg'] for r in rr)},
        'frequency_analysis_performed':False,
        'reason':'Retained native samples and missing-data intervals; rerun because the predeclared spectral window has >50ms gaps. No interpolation across those gaps.',
        'sha256':{str(path/n):sha(path/n) for n in ('imu.csv','collection.json','metadata.json')}}


def main(a):
    cases={};data={}
    for spec in a.closed_loop:
        parts=spec.split(',');assert len(parts) in (3,4)
        label,path,kd=parts[:3];kp=float(parts[3]) if len(parts)==4 else 25.
        assert label not in cases
        cases[label],data[label]=closed(Path(path),float(kd),a.lo,a.hi,kp)
    assert len(cases)>=2
    first=next(iter(cases.values()))
    reference={key:first[key] for key in ('policy_sha256','base_sha256','config_except_pd','model_sha256')}
    for case in cases.values():
        for key,value in reference.items():assert case[key]==value,key
        del case['config_except_pd']
    result={'closed_loop':cases,'checks':{'identical_policy_base_model':True,'only_rl_pd_configuration_differs':True},
        'frequency_method':'100Hz interpolation, linear detrend/Hann; 1..20Hz peak and 4..10Hz DFT RMS; independent gait phases; no lag fitting',
        'limitations':['Nominal Gazebo was already much steadier than real hardware; this tests PD candidates, not real repair.',
            'Closed-loop runs use own IMU/q/dq/previous actions. Fixed-history replay is a separate forced-response experiment.',
            'Nominal PD from actor snapshots is not independent physical joint-load measurement.',
            'Two Kp25/Kd0.8 baselines provide a repeatability reference; individual candidates are single runs, not confidence intervals or guarantees of hardware stability.',
            'The gap-run candidate occurred before the restored baseline; the gap-free candidate retry occurred afterward. Plot overlay order is not chronological execution order.']}
    result['data_gap_runs']=[gap_run(p,a.lo,a.hi) for p in a.gap_run]
    replay_data=None
    replay_specs=list(a.replay_case)
    if a.replay10:replay_specs.insert(0,'kd10,'+str(a.replay10)+',25,1')
    for spec in replay_specs:
        label,path,kp,kd=spec.split(',');assert label!='kd08'
        rr,dd=fixed_replay(a.replay08,Path(path),38.7,41.2,float(kp),float(kd),label)
        if replay_data is None:
            result['fixed_history_replay']=rr;replay_data=dd
        else:
            dest=result['fixed_history_replay'];assert label not in dest['metrics']
            for key in ('paths','metrics','sha256'):dest[key].update(rr[key])
            replay_data.update(dd)
    if replay_data:
        rank={}
        for case in cases.values():rank.setdefault((case['kp'],case['kd']),len(rank))
        metrics=result['fixed_history_replay']['metrics']
        labels=sorted(metrics,key=lambda label:rank.get((metrics[label]['kp'],metrics[label]['kd']),len(rank)))
        result['fixed_history_replay']['metrics']={label:metrics[label] for label in labels}
        replay_data={label:replay_data[label] for label in labels}
    a.output_prefix.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    rows=[]
    for label,c in cases.items():
        m=c['metrics'];rows.append({'kind':'closed_loop','case':label,'kp':c['kp'],'kd':c['kd'],
            'gyro_x_rms_rad_s':m['gyro_x_rms_rad_s'],'roll_std_deg':m['roll_std_deg'],
            'roll_mean_deg':m['roll_mean_deg'],'roll_peak_to_peak_deg':m['roll_peak_to_peak_deg'],
            'gyro_band_4_10_rms':m['gyro_spectrum']['band_4_10_rms'],
            'gyro_peak_hz':m['gyro_spectrum']['peak_1_20_hz'],
            'body_vx_mps':m['motion']['body_vx_mean_mps'],'min_height_m':m['motion']['height_min_m'],
            'max_tilt_deg':m['max_tilt_deg'],'normal_walking_valid':m['normal_walking_valid']})
    if replay_data:
        for label,m in result['fixed_history_replay']['metrics'].items():
            rows.append({'kind':'fixed_history_replay','case':label,'kp':m['kp'],'kd':m['kd'],
                'gyro_x_rms_rad_s':m['gyro_x_rms_rad_s'],'roll_std_deg':m['roll_std_deg'],
                'roll_mean_deg':m['roll_mean_deg'],'roll_peak_to_peak_deg':m['roll_peak_to_peak_deg'],
                'gyro_band_4_10_rms':m['gyro_spectrum']['band_4_10_rms'],
                'gyro_peak_hz':m['gyro_spectrum']['peak_1_20_hz'],
                'body_vx_mps':m['mean_body_vx_mps'],'min_height_m':m['min_height_m'],
                'max_tilt_deg':m['max_tilt_deg'],'normal_walking_valid':m['normal_walking_valid']})
    with a.output_prefix.with_suffix('.csv').open('w') as f:
        writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(rows,indent=2))
    if a.plot:plot(a,cases,data,result.get('fixed_history_replay'),replay_data)


def plot(a,cases,data,replay,replay_data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def save(fig,suffix):
        fig.savefig(str(a.output_prefix)+'_'+suffix+'.png',dpi=150,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for label,(imu,odom,policy) in data.items():
        for axis,rows,key in [(axes[0,0],imu,'roll_deg'),(axes[0,1],imu,'gyro_x'),(axes[0,2],odom,'body_vx')]:
            rr=[r for r in rows if a.lo<=r['t_s']<a.hi]
            axis.plot([r['t_s']-a.lo for r in rr],[r[key] for r in rr],label=label,lw=.8)
        _,bins=spectrum(wave(imu,'gyro_x',a.lo,a.hi))
        axes[1,0].plot([x for x,_ in bins],[y for _,y in bins],label=label)
        rr=[r for r in policy if a.lo<=r['t_s']<a.hi]
        for axis,key in [(axes[1,1],'state_dq_11'),(axes[1,2],'state_tau_est_11')]:
            axis.plot([r['t_s']-a.lo for r in rr],[r[key] for r in rr],label=label,lw=.8)
    for axis,title,unit in zip(axes.flat,['Body roll','Roll rate','Forward speed','Roll-rate spectrum','RR shank dq','RR shank feedback torque'],['deg','rad/s','m/s','rad/s peak amplitude','rad/s','Nm']):
        axis.set(title=title,ylabel=unit,xlabel='Within-window time (s); independent phase');axis.grid(alpha=.2);axis.legend(fontsize=8)
    axes[1,0].set(xlabel='Hz',xlim=(0,20));axes[0,2].axhline(.5,color='black',ls='--',lw=.8)
    fig.suptitle(f'Saved 9299 | isolated PD comparison; baseline repeated | Gazebo own-feedback closed loop\nSame +0.5m/s; PPO +{a.lo}..{a.hi}s; independent phases')
    save(fig,'closedloop')
    if replay_data:
        fig,axes=plt.subplots(2,2,figsize=(13,7),layout='constrained')
        for label,rows in replay_data.items():
            if not replay['metrics'][label]['normal_walking_valid']:continue
            rr=[r for r in rows if 36<=r['t_s']<=44]
            for axis,key in [(axes[0,0],'roll_deg'),(axes[0,1],'gyro_x'),(axes[1,0],'body_vx')]:
                axis.plot([r['t_s'] for r in rr],[r[key] for r in rr],label=label,lw=.8)
            _,bins=spectrum(wave(rows,'gyro_x',38.7,41.2))
            axes[1,1].plot([x for x,_ in bins],[y for _,y in bins],label=label)
        for axis,title,unit in zip(axes.flat,['Body roll','Roll rate','Forward speed','Roll-rate spectrum'],['deg','rad/s','m/s','rad/s peak amplitude']):
            axis.set(title=title,ylabel=unit,xlabel='Source time (s)');axis.grid(alpha=.2);axis.legend(fontsize=8)
        axes[1,1].set(xlabel='Hz',xlim=(0,20))
        fig.suptitle('Original real obs -> saved actor -> identical target sequence | only PD gains changed\nFixed-history forced response; failed cases excluded here and plotted separately')
        save(fig,'fixedhistory')
        fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
        groups=[[(label,c['metrics'],c['kp'],c['kd']) for label,c in cases.items()],
                [(label,m,m['kp'],m['kd']) for label,m in replay['metrics'].items()]]
        for row,group in enumerate(groups):
            labels=[f'{kp:g}/{kd:g}' + (' INVALID' if not m['normal_walking_valid'] else '') + (' repeat' if 'A2' in label else '') for label,m,kp,kd in group]
            for col,key in enumerate(['gyro_x_rms_rad_s','roll_std_deg','vx']):
                values=[(m['motion']['body_vx_mean_mps'] if row==0 else m['mean_body_vx_mps']) if key=='vx' else m[key] for label,m,kp,kd in group]
                ax=axes[row,col];bars=ax.bar(labels,values,color=['#4878b8' if kp==25 and kd==.8 else '#d88b37' for label,m,kp,kd in group])
                for bar,(_,m,_,_) in zip(bars,group):
                    if not m['normal_walking_valid']:bar.set_color('#bbbbbb');bar.set_edgecolor('black');bar.set_hatch('//')
                ax.bar_label(bars,fmt='%.3f',fontsize=8);ax.tick_params(axis='x',rotation=30);ax.grid(axis='y',alpha=.2)
                ax.set(title=('Own-feedback closed loop' if row==0 else 'Fixed real-target history')+' | '+['Roll rate RMS','Roll standard deviation','Actual forward speed'][col],ylabel=['rad/s','deg','m/s'][col],xlabel='Kp / Kd')
                ax.set_ylim(top=max(values)*1.2)
        fig.suptitle('Saved 9299 | PD sweep | Same policy/model/command or same target history\nIndependent gait phases; single candidates, not hardware validation')
        save(fig,'overview')
        for label,rows in replay_data.items():
            m=replay['metrics'][label]
            if m['normal_walking_valid']:continue
            rr=[r for r in rows if 36<=r['t_s']<=44]
            fig,axes=plt.subplots(3,1,figsize=(11,8),layout='constrained',sharex=True)
            for ax,key,unit in zip(axes,['roll_deg','z','body_vx'],['roll (deg)','base height (m)','forward speed (m/s)']):
                ax.plot([r['t_s'] for r in rr],[r[key] for r in rr],lw=.9);ax.set_ylabel(unit);ax.grid(alpha=.2)
                ax.axvspan(38.7,41.2,color='orange',alpha=.12)
                if m['first_tilt_over_60_source_s'] is not None:ax.axvline(m['first_tilt_over_60_source_s'],color='red',ls='--',label='first tilt >60 deg');ax.legend()
            axes[-1].set_xlabel('Source time (s)')
            fig.suptitle(f"INVALID normal-walking comparison | {label} Kp{m['kp']:g}/Kd{m['kd']:g}\nShaded analysis window; failed/unstable trace retained, not a damping success")
            save(fig,'invalid_'+label)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--closed-loop',action='append',required=True,help='label,path,kd[,kp]; repeat for each isolated run')
    p.add_argument('--replay08',type=Path);p.add_argument('--replay10',type=Path)
    p.add_argument('--replay-case',action='append',default=[],help='label,path,kp,kd; compared with --replay08 baseline')
    p.add_argument('--gap-run',type=Path,action='append',default=[],help='Retain a completed run with >50ms core IMU gaps, reporting native statistics only')
    p.add_argument('--lo',type=float,default=3.);p.add_argument('--hi',type=float,default=9.)
    p.add_argument('--output-prefix',type=Path,required=True);p.add_argument('--plot',action='store_true')
    a=p.parse_args()
    if a.hi<=a.lo or bool(a.replay08)!=bool(a.replay10 or a.replay_case):p.error('invalid window or unpaired replay paths')
    main(a)
