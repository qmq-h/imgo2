#!/usr/bin/env python3
"""Stdlib audit of q/dq-only obs substitution with unchanged real IMU."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics as st
from audit_paired_obs_replay import load,attitude,summarize,spectrum,wave

HIPS=(0,3,6,9)
LEGS=('FL','FR','RL','RR')


def main(a):
    before=json.loads((a.baseline/'metadata.json').read_text());after=json.loads((a.swap/'metadata.json').read_text())
    inf=json.loads((a.inference/'inference_summary.json').read_text());cfg=json.loads((a.real/'meta.json').read_text())['config']
    data={}
    for label,path in [('baseline',a.baseline),('joint_swap',a.swap)]:
        rows=load(path/'replay.csv')
        for r in rows:r['t_s']+=after['source_start_s'];attitude(r)
        data[label]=rows
    real=load(a.real/'robot.csv')
    for r in real:r['t_s']=r['t_mono_ms']*.001;attitude(r)
    data['real']=real
    obs=load(a.inference/'obs_used.csv');donor=load(a.inference/'joint_donor.csv')
    assert len(obs)==len(donor)
    for r,d in zip(obs,donor):r['t_s']=d['input_captured_s']
    changed=[i for i in range(45) if any(r[f'original_{i:02}']!=r[f'used_{i:02}'] for r in obs)]
    q_error=max(abs(r[f'used_{i+9:02}']-(d[f'q_{i:02}']-cfg['default_pose'][i])*cfg['obs_scales']['dof_pos']) for r,d in zip(obs,donor) for i in range(12))
    dq_error=max(abs(r[f'used_{i+21:02}']-d[f'dq_{i:02}']*cfg['obs_scales']['dof_vel']) for r,d in zip(obs,donor) for i in range(12))
    windows={}
    for label,lo,hi in [('quiet',83.,85.),('pure_straight_short',86.14,87.28),('forward_mixed',86.5,92.4)]:
        physical={case:summarize(rows,lo,hi) for case,rows in data.items()};rr=[r for r in obs if lo<=r['t_s']<hi]
        inputs={}
        for name in ('original','used'):
            qm={i:st.mean(r[f'{name}_{9+i:02}']/cfg['obs_scales']['dof_pos'] for r in rr) for i in HIPS}
            inputs[name]={'hip_q_std_rad':math.sqrt(st.mean((r[f'{name}_{9+i:02}']/cfg['obs_scales']['dof_pos']-qm[i])**2 for r in rr for i in HIPS)),
                'hip_dq_rms_rad_s':math.sqrt(st.mean((r[f'{name}_{21+i:02}']/cfg['obs_scales']['dof_vel'])**2 for r in rr for i in HIPS)),
                'per_joint_q_relative_mean_rad':[st.mean(r[f'{name}_{9+i:02}']/cfg['obs_scales']['dof_pos'] for r in rr) for i in range(12)]}
        upright=physical['joint_swap'].get('upright',False)
        windows[label]={'physical':physical,'input':inputs,'upright_motion_comparison_valid':upright,
            'swap_change_pct_if_upright':{k:100*(physical['joint_swap'][k]/physical['baseline'][k]-1) for k in ('gyro_x_rms_rad_s','roll_std_deg','hip_dq_rms_rad_s')} if upright else None}
        if not upright:windows[label]['warning']='fell/lost upright posture; raw Euler std and gyro RMS are not a normal-walking comparison'
    sent=load(a.swap/'published.csv');targets=load(a.swap/'targets.csv')
    checks={'changed_obs_channels':changed,'only_q_dq_changed':changed==list(range(9,33)),
        'all_imu_exactly_unchanged':all(r[f'original_{i:02}']==r[f'used_{i:02}'] for r in obs for i in range(6)),
        'commands_previous_actions_exactly_unchanged':all(r[f'original_{i:02}']==r[f'used_{i:02}'] for r in obs for i in list(range(6,9))+list(range(33,45))),
        'donor_q_conversion_max_error':q_error,'donor_dq_conversion_max_error':dq_error,
        'initial_identical':before['initial']==after['initial'],'cadence_identical':[r['t_s'] for r in load(a.baseline/'targets.csv')]==[r['t_s'] for r in targets],
        'protected_hashes_match_baseline':before['protected_hashes']==after['protected_hashes'],'production_files_unchanged':after['production_files_unchanged'],
        'completed':after.get('completed',False),'planned_targets':len(targets),'published_targets':len(sent),
        'all_indices_in_order':[int(r['index']) for r in sent]==list(range(len(targets))),
        'max_publish_lag_s':max(r['t_s']-r['scheduled_s'] for r in sent)}
    trajectory=data['joint_swap']
    events={name:next((r['t_s'] for r in trajectory if condition(r)),None) for name,condition in [
        ('first_tilt_over_20_deg_s',lambda r:r['tilt_deg']>20),('first_tilt_over_60_deg_s',lambda r:r['tilt_deg']>60),
        ('first_height_below_0_20_m_s',lambda r:r['z']<.2)]}
    files=[a.inference/n for n in ('obs_used.csv','joint_donor.csv','inference.csv','targets.csv','inference_summary.json')]+[a.swap/n for n in ('replay.csv','published.csv','metadata.json')]
    result={'checks':checks,'windows':windows,'inference':inf,'posture_events':events,
        'limitations':['q/dq are taken together from the normal run, but their phase/history does not match retained real IMU/previous actions.',
            'Single command-onset alignment; no gait-phase optimization or actuator/load calibration.',
            'Observation replacement does not change motor internal PD; no hardware or loaded response identification was performed.',
            'All simulated feedback is recorded only; it does not replace later historical inputs.',
            'Long forward window includes lateral/yaw commands; pure straight window is short.',
            'If the robot falls, wrapped Euler and post-fall gyro RMS are not valid normal-walking jitter metrics.'],
        'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    a.output_prefix.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for label,v in windows.items():
        print(label,'input',v['input'],'upright',v['upright_motion_comparison_valid'])
        for case,m in v['physical'].items():print(case,{k:m[k] for k in ('gyro_x_rms_rad_s','roll_std_deg','roll_peak_to_peak_deg','hip_dq_rms_rad_s','max_tilt_deg')},m.get('min_height_m'),m.get('mean_body_vx_mps'),m.get('yaw_change_deg'),m['gyro_spectrum'])
        print('change',v['swap_change_pct_if_upright'])
    print('checks',checks)
    if a.plot:plot(a,data,obs,cfg)


def plot(a,data,obs,cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def save(fig,suffix):fig.savefig(str(a.output_prefix)+'_'+suffix+'.png',dpi=160,bbox_inches='tight');plt.close(fig)
    for start,scale,suffix,unit in [(9,cfg['obs_scales']['dof_pos'],'q_inputs','relative rad'),(21,cfg['obs_scales']['dof_vel'],'dq_inputs','rad/s')]:
        fig,ax=plt.subplots(4,3,figsize=(15,11),layout='constrained')
        rr=[r for r in obs if 85.5<=r['t_s']<=88.]
        for i,axis in enumerate(ax.flat):
            for name,label in [('original','Original real'),('used','Normal replacement')]:axis.plot([r['t_s'] for r in rr],[r[f'{name}_{start+i:02}']/scale for r in rr],label=label,lw=.9)
            axis.set(title=f'{LEGS[i//3]} '+('hip','thigh','shank')[i%3],ylabel=unit,xlabel='Input capture source time (s)');axis.grid(alpha=.25)
        ax[0,0].legend(fontsize=8);fig.suptitle('Only joint q/dq inputs replaced together; gyro/gravity/commands/previous actions unchanged\nNormal joint reference aligned by command onset; no gait-phase fitting')
        save(fig,suffix)
    colors={'baseline':'#0077bb','joint_swap':'#ee7733','real':'#777777'}
    fig,ax=plt.subplots(3,2,figsize=(14,10),layout='constrained')
    for label,rows in data.items():
        rr=[r for r in rows if 82<=r['t_s']<=93]
        for axis,key,title,unit in [(ax[0,0],'roll_deg','Physical roll','deg'),(ax[0,1],'pitch_deg','Physical pitch','deg'),(ax[1,0],'gyro_x','Physical roll rate','rad/s'),(ax[1,1],'body_vx','Forward speed','m/s'),(ax[2,0],'z','Base height','m'),(ax[2,1],'tilt_deg','Total body tilt','deg')]:
            if key in rr[0]:axis.plot([r['t_s'] for r in rr],[r[key] for r in rr],label=label,color=colors[label],lw=.8)
            axis.set(title=title,ylabel=unit,xlabel='Source time (s)');axis.grid(alpha=.25)
    ax[0,0].legend(fontsize=8);fig.suptitle('Original obs vs normal q/dq + retained real IMU\nSame initial state/model/P25/D0.8; changed observations are not a new self-consistent feedback loop')
    save(fig,'response')
    # Spectra are explicitly tagged if posture is lost.
    fig,ax=plt.subplots(1,2,figsize=(14,4),layout='constrained')
    for label,rows in data.items():
        rr=[r for r in rows if 86.5<=r['t_s']<92.4]
        fell=max(r['tilt_deg'] for r in rr)>60 or ('z' in rr[0] and min(r['z'] for r in rr)<.2)
        for axis,key,title in [(ax[0],'gyro_x','Roll-rate spectrum'),(ax[1],'roll_deg','Roll spectrum')]:
            _,bins=spectrum(wave(rows,key,86.5,92.4));axis.plot([x for x,_ in bins],[y for _,y in bins],label=label+(' (LOST UPRIGHT)' if fell else ''),color=colors[label]);axis.set(title=title,xlabel='Hz',xlim=(0,20));axis.grid(alpha=.25);axis.legend(fontsize=8)
    fig.suptitle('5.9s raw response spectra; lost upright posture invalidates a normal-walking comparison')
    save(fig,'spectra')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('real','baseline','swap','inference','output-prefix'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--plot',action='store_true');main(p.parse_args())
