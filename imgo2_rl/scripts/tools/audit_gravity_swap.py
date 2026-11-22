#!/usr/bin/env python3
"""Stdlib audit of gravity-only counterfactual obs; optional matplotlib plots."""
import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path
import statistics as st
from audit_paired_obs_replay import load,attitude,summarize,spectrum,wave


def interp(rows,times,t,key):
    j=min(len(rows)-1,max(1,bisect.bisect_right(times,t)))
    f=max(0,min(1,(t-times[j-1])/(times[j]-times[j-1])))
    return rows[j-1][key]+f*(rows[j][key]-rows[j-1][key])


def main(a):
    baseline_meta=json.loads((a.baseline/'metadata.json').read_text())
    swap_meta=json.loads((a.swap/'metadata.json').read_text())
    inf=json.loads((a.inference/'inference_summary.json').read_text())
    data={}
    for label,path in [('baseline',a.baseline),('gravity_swap',a.swap)]:
        rows=load(path/'replay.csv')
        for r in rows:
            r['t_s']+=swap_meta['source_start_s'];attitude(r)
        data[label]=rows
    real=load(a.real/'robot.csv')
    for r in real:r['t_s']=r['t_mono_ms']*.001;attitude(r)
    data['real']=real
    obs=load(a.inference/'obs_used.csv');donor=load(a.inference/'gravity_donor.csv')
    if len(obs)!=len(donor):raise ValueError('donor row count mismatch')
    rt=[r['t_s'] for r in real]
    for r,d in zip(obs,donor):
        r['t_s']=d['input_captured_s']
        for name in ('original','used'):
            gx,gy,gz=[r[f'{name}_{i:02}'] for i in (3,4,5)]
            r[name+'_roll_deg']=math.degrees(math.atan2(-gy,-gz))
            r[name+'_pitch_deg']=math.degrees(math.atan2(gx,math.hypot(gy,gz)))
        r['normal_yaw_deg']=d['donor_yaw_deg']
        r['real_yaw_deg']=math.degrees(interp(real,rt,r['t_s'],'yaw_rad'))
    windows={}
    for label,lo,hi in [('quiet',83.,85.),('pure_straight_short',86.14,87.28),('forward_mixed',86.5,92.4)]:
        m={k:summarize(rows,lo,hi) for k,rows in data.items()}
        s=[r for r in obs if lo<=r['t_s']<hi]
        input_stats={name:{'gravity_xyz_std':[st.pstdev(r[f'{name}_{i:02}'] for r in s) for i in (3,4,5)],
            'roll_std_deg':st.pstdev(r[name+'_roll_deg'] for r in s),'pitch_std_deg':st.pstdev(r[name+'_pitch_deg'] for r in s)} for name in ('original','used')}
        windows[label]={'physical':m,'inputs':input_stats,
            'swap_relative_change_pct':{k:100*(m['gravity_swap'][k]/m['baseline'][k]-1) for k in ('gyro_x_rms_rad_s','roll_std_deg','roll_peak_to_peak_deg','hip_dq_rms_rad_s')}}
    changed=[i for i in range(45) if any(r[f'original_{i:02}']!=r[f'used_{i:02}'] for r in obs)]
    published=load(a.swap/'published.csv');targets=load(a.swap/'targets.csv')
    checks={'changed_obs_channels':changed,'other_42_exactly_unchanged':changed==[3,4,5],
        'gravity_norm_max_error':max(abs(math.sqrt(sum(r[f'used_{i:02}']**2 for i in (3,4,5)))-1) for r in obs),
        'rpy_gravity_roll_max_error_deg':max(abs(r['used_roll_deg']-d['donor_roll_deg']) for r,d in zip(obs,donor)),
        'rpy_gravity_pitch_max_error_deg':max(abs(r['used_pitch_deg']-d['donor_pitch_deg']) for r,d in zip(obs,donor)),
        'initial_identical':baseline_meta['initial']==swap_meta['initial'],
        'target_cadence_identical': [r['t_s'] for r in load(a.baseline/'targets.csv')]==[r['t_s'] for r in targets],
        'targets':len(targets),'published':len(published),'all_indices_in_order':[int(r['index']) for r in published]==list(range(len(targets))),
        'max_publication_lag_s':max(r['t_s']-r['scheduled_s'] for r in published),
        'production_files_unchanged':swap_meta['production_files_unchanged'],'completed':swap_meta.get('completed',False),
        'protected_hashes_match_baseline':baseline_meta['protected_hashes']==swap_meta['protected_hashes']}
    paths=[a.inference/n for n in ('obs_used.csv','gravity_donor.csv','inference.csv','targets.csv','inference_summary.json')]+[a.swap/n for n in ('replay.csv','published.csv','metadata.json')]
    result={'checks':checks,'windows':windows,'inference':inf,
        'limitations':['Only gravity changes; inconsistent gyro/q/dq/action history is deliberately retained as requested.',
            'Simulation feedback does not become the next actor input; this is open-loop historical-observation intervention.',
            'Reference run is aligned by command onset without phase search; quiet prefix holds donor orientation at 9.9s.',
            'RPY inspected; yaw is not encoded in projected gravity and is not supplied as an extra actor input.',
            'Real runtime weight identity lacks original hash, but prior unchanged whole-log actor replay matched.',
            'Long forward segment contains lateral/yaw commands; pure straight segment is short and has coarse spectral resolution.',
            'Single trial cannot identify a sensor fault or rule out gravity contribution to a physically consistent closed loop.'],
        'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    a.output_prefix.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for label,v in windows.items():
        print(label,'input',v['inputs'])
        for case,m in v['physical'].items():print(case,{k:m[k] for k in ('gyro_x_rms_rad_s','roll_std_deg','roll_peak_to_peak_deg','hip_dq_rms_rad_s')},m['gyro_spectrum'],m.get('mean_body_vx_mps'))
        print('change %',v['swap_relative_change_pct'])
    print('checks',checks)
    if a.plot:plot(a,data,obs)


def plot(a,data,obs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def save(fig,suffix):
        fig.savefig(str(a.output_prefix)+'_'+suffix+'.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(3,2,figsize=(14,10),layout='constrained')
    t=[r['t_s'] for r in obs]
    for j in range(3):
        for name,label in [('original','Original real input'),('used','Normal Gazebo replacement')]:
            ax[j,0].plot(t,[r[f'{name}_{j+3:02}'] for r in obs],label=label,lw=.8)
        ax[j,0].set(title='Projected gravity '+('X','Y','Z')[j],ylabel='unit gravity vector')
    for j,angle in enumerate(('roll','pitch')):
        for name,label in [('original','Original real gravity-derived '+angle),('used','Normal reference '+angle)]:
            ax[j,1].plot(t,[r[name+'_'+angle+'_deg'] for r in obs],label=label,lw=.8)
        ax[j,1].set(title=angle.capitalize()+' represented by gravity',ylabel='deg')
    for key,label in [('real_yaw_deg','Real robot yaw change (nearby robot sample)'),('normal_yaw_deg','Normal reference yaw change')]:
        ax[2,1].plot(t,[r[key]-obs[0][key] for r in obs],label=label,lw=.8)
    ax[2,1].set(title='Yaw diagnostic only: NOT an actor input or replacement',ylabel='deg from first sample')
    for axis in ax.flat:axis.grid(alpha=.25);axis.legend(fontsize=8);axis.set_xlabel('Real input capture time (s)')
    fig.suptitle('Gravity-only counterfactual | all other 42 inputs unchanged\nNormal reference matched by command onset; no phase fitting; yaw is absent from gravity')
    save(fig,'inputs_rpy')
    fig,ax=plt.subplots(3,2,figsize=(14,10),layout='constrained')
    colors={'baseline':'#0077bb','gravity_swap':'#ee7733','real':'#777777'}
    for label,rows in data.items():
        rr=[r for r in rows if 82<=r['t_s']<=93]
        for axis,key,title,unit in [(ax[0,0],'roll_deg','Physical roll','deg'),(ax[0,1],'pitch_deg','Physical pitch','deg'),
                (ax[1,0],'yaw_rad','Physical yaw change','deg'),(ax[1,1],'gyro_x','Physical roll rate','rad/s'),
                (ax[2,0],'body_vx','Actual forward speed','m/s'),(ax[2,1],'z','Base height','m')]:
            if key not in rr[0]:continue
            values=[r[key] for r in rr]
            if key=='yaw_rad':
                accumulated=[0.]
                for before,after in zip(values,values[1:]):
                    accumulated.append(accumulated[-1]+math.degrees(math.atan2(math.sin(after-before),math.cos(after-before))))
                values=accumulated
            axis.plot([r['t_s'] for r in rr],values,label=label,color=colors[label],lw=.8);axis.set(title=title,ylabel=unit)
    for axis in ax.flat:axis.grid(alpha=.25);axis.legend(fontsize=8);axis.set_xlabel('Source time (s)')
    fig.suptitle('Gazebo response: original real obs vs only gravity replaced\nSame initial state, model, target cadence and P25/D0.8; historical feedback retained')
    save(fig,'response')
    fig,ax=plt.subplots(1,2,figsize=(14,4),layout='constrained')
    for label,rows in data.items():
        for axis,key,title,unit in [(ax[0],'gyro_x','Roll-rate spectrum','rad/s peak amplitude'),(ax[1],'roll_deg','Roll spectrum','deg peak amplitude')]:
            _,bins=spectrum(wave(rows,key,86.5,92.4));axis.plot([f for f,_ in bins],[v for _,v in bins],label=label,color=colors[label]);axis.set(title=title,ylabel=unit,xlabel='Hz',xlim=(0,20));axis.grid(alpha=.25);axis.legend()
    fig.suptitle('Forward 5.9s, detrended Hann spectrum; 0.1695Hz grid')
    save(fig,'spectra')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('real','baseline','swap','inference','output-prefix'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--plot',action='store_true');main(p.parse_args())
