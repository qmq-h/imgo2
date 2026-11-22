#!/usr/bin/env python3
"""Audit a retrospective shank-position-only input pull and Gazebo replay.

Core checks/statistics use the standard library. --plot needs Matplotlib.
Never opens hardware, changes policy weights, or fits a time shift.
"""
import argparse
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st

from audit_paired_obs_replay import attitude, load, rms, spectrum, summarize, wave
from compare_pd_torque_time import interp

SHANKS=(2,5,8,11)
LEGS=('FL','FR','RL','RR')
DEG=180/math.pi


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(a):
    bm=json.loads((a.baseline/'metadata.json').read_text())
    pm=json.loads((a.pull/'metadata.json').read_text())
    report=json.loads((a.inference/'inference_summary.json').read_text())
    baseline_report=json.loads((Path(bm['targets_override']['path']).parent/'inference_summary.json').read_text())
    assert report['sha256']==baseline_report['sha256']
    cfg=json.loads((a.real/'meta.json').read_text())['config']
    start=pm['source_start_s'];end=start+pm['duration_s']
    assert bm['source_start_s']==start and bm['duration_s']==pm['duration_s']
    assert bm['completed'] and pm['completed']
    assert bm['production_files_unchanged'] and pm['production_files_unchanged']
    assert bm['protected_hashes']==pm['protected_hashes']
    assert (a.baseline/'initial.csv').read_bytes()==(a.pull/'initial.csv').read_bytes()
    assert bm['plugin_sha256']==pm['plugin_sha256']
    with (a.real/'robot.csv').open() as f:
        real=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)
              if start-.2<=float(r['t_mono_ms'])*.001<=end+.2]
    with (a.real/'policy.csv').open() as f:
        policy=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)
                if start-.2<=float(r['t_mono_ms'])*.001<=end+.2]
    by_tick={r['tick']:r for r in real}
    by_time={r['t_mono_ms']*.001:r for r in policy}
    obs=load(a.inference/'obs_used.csv');refs=load(a.inference/'shank_pull.csv')
    infer=load(a.inference/'inference.csv')
    assert len(obs)==len(refs)==len(infer)
    assert all(math.isfinite(v) for rows in (obs,refs,infer) for r in rows for v in r.values())
    changed=[i for i in range(45) if any(r[f'original_{i:02}']!=r[f'used_{i:02}'] for r in obs)]
    assert changed==[11,14,17,20],changed
    fraction=report['shank_position_pull']['fraction']
    assert fraction==.5,'This audit compares the specified 50% intervention'
    target_error=0.;input_error=0.;contraction_error=0.
    for o,r in zip(obs,refs):
        p=by_time[o['source_s']];b=by_tick[p['tick']]
        assert r['tick']==p['tick'] and r['target_record_s']==b['t_mono_ms']*.001
        for i in range(45):assert o[f'original_{i:02}']==p[f'obs_{i:02}']
        for i in SHANKS:
            target=b[f'q_des_{cfg["policy_to_hardware"][i]:02}']
            target_error=max(target_error,abs(r[f'target_q_{i:02}']-target))
            q=o[f'original_{9+i:02}']/cfg['obs_scales']['dof_pos']+cfg['default_pose'][i]
            used=o[f'used_{9+i:02}']/cfg['obs_scales']['dof_pos']+cfg['default_pose'][i]
            input_error=max(input_error,abs(used-(q+fraction*(target-q))))
            contraction_error=max(contraction_error,abs((target-used)-(1-fraction)*(target-q)))
    assert max(target_error,input_error,contraction_error)<1e-12
    baseline_targets=load(a.baseline/'targets.csv');targets=load(a.pull/'targets.csv')
    assert [r['t_s'] for r in targets]==[r['t_s'] for r in baseline_targets]
    outer=[max((r for r in real if r['t_mono_ms']*.001<=start),key=lambda r:r['t_mono_ms'])]
    outer += [r for r in real if start<r['t_mono_ms']*.001<=end]
    assert len(outer)==len(targets)
    ticks=[by_time[o['source_s']]['tick'] for o in obs]
    conversion_error=0.
    for t,b,original in zip(targets,baseline_targets,outer):
        selected=max(0,bisect.bisect_right(ticks,original['tick'])-1)
        assert ticks[selected]<=original['tick']
        for i,h in enumerate(cfg['policy_to_hardware']):
            q=cfg['default_pose'][i]+cfg['action_scale'][i]*infer[selected][f'action_{i:02}']
            q=max(cfg['limits']['q_min'][h],min(cfg['limits']['q_max'][h],q))
            conversion_error=max(conversion_error,abs(t[f'q_{h:02}']-q))
        for k in ('dq','tau','kp','kd'):
            assert all(t[f'{k}_{i:02}']==b[f'{k}_{i:02}'] for i in range(12))
    assert conversion_error<1e-12
    publication={}
    data={'real':real}
    for label,path,meta in [('baseline',a.baseline,bm),('pull50',a.pull,pm)]:
        sent=load(path/'published.csv');planned=load(path/'targets.csv')
        assert len(sent)==len(planned)
        assert [int(r['index']) for r in sent]==list(range(len(sent)))
        publication[label]={'targets':len(planned),'published':len(sent),
                            'max_publish_lag_s':max(r['t_s']-r['scheduled_s'] for r in sent)}
        rows=load(path/'replay.csv')
        for r in rows:r['t_s']+=start
        assert all(math.isfinite(v) for r in rows for v in r.values())
        assert all(b['t_s']>a['t_s'] for a,b in zip(rows,rows[1:]))
        data[label]=rows
    for r in real+policy:r['t_s']=r['t_mono_ms']*.001
    for rows in data.values():
        for r in rows:attitude(r)
    windows={}
    for name,lo,hi in [('forward',a.lo,a.hi),('full_replay',start+.05,end-.05)]:
        metrics={label:summarize(rows,lo,hi,policy) for label,rows in data.items()}
        for label in ('baseline','pull50'):
            metrics[label]['normal_walking_comparison_valid']=(metrics[label]['upright'] and metrics[label]['max_tilt_deg']<=20)
            if not metrics[label]['normal_walking_comparison_valid']:
                metrics[label]['warning']='Exceeded 20deg tilt and/or lost upright posture; raw angle/spectrum metrics are descriptive, not a normal-walking jitter comparison.'
        inp={}
        selected=[r for r in refs if lo<=r['input_captured_s']<hi]
        for kind in ('before','after'):
            inp[kind+'_shank_error_rms_deg']=rms([r[f'error_{kind}_{i:02}']*DEG for r in selected for i in SHANKS])
        windows[name]={'physical':metrics,'observed_input':inp,
                       'change_pct_if_upright':{k:100*(metrics['pull50'][k]/metrics['baseline'][k]-1)
                       for k in ('gyro_x_rms_rad_s','roll_std_deg','roll_peak_to_peak_deg')}
                       if all(metrics[k]['normal_walking_comparison_valid'] for k in ('baseline','pull50')) else None}
    events={label:{name:next((r['t_s'] for r in rows if test(r)),None) for name,test in [
        ('tilt_over_20_deg',lambda r:r['tilt_deg']>20),('tilt_over_60_deg',lambda r:r['tilt_deg']>60),
        ('height_below_0_20_m',lambda r:r.get('z',1)<.2)]} for label,rows in data.items()}
    checks={'changed_obs_channels':changed,'other_41_channels_exactly_unchanged':True,
        'same_tick_original_target_max_error_rad':target_error,'input_formula_max_error_rad':input_error,
        'error_contraction_max_error_rad':contraction_error,'target_tick_conversion_max_error_rad':conversion_error,
        'identical_initial_state':True,'identical_submission_cadence':True,'identical_simulation_plugin':True,
        'unchanged_dq_des_tau_ff_kp_kd':True,'production_files_unchanged':True,'publication':publication}
    files=[a.real/n for n in ('meta.json','robot.csv','policy.csv')]
    files += [a.inference/n for n in ('obs_used.csv','shank_pull.csv','inference.csv','inference_summary.json')]
    files += [p/n for p in (a.baseline,a.pull) for n in ('replay.csv','metadata.json','targets.csv','published.csv')]
    result={'source':str(a.real),'baseline':str(a.baseline),'pull50':str(a.pull),
        'inference':report,'checks':checks,'windows':windows,'posture_events':events,
        'frequency_method':'100Hz interpolation, linear detrend + Hann; dominant 1..20Hz bin; 4..10Hz RMS via demeaned DFT; no lag fitting',
        'limitations':['Retrospective same-tick original target is hindsight information, not a causal replacement sensor.',
            'Only q inputs change. dq, IMU and previous actions remain real historical values, so this is not a self-consistent closed loop.',
            'Gazebo feedback is recorded but does not update subsequent actor inputs; PD remains 25/.8.',
            'Real base height/velocity/contact initial state and original runtime policy hash are not fully recorded.',
            'Failure/lost upright windows cannot be interpreted as normal walking jitter reduction.',
            'Normal-walking comparison flag also requires maximum tilt <=20deg; full-replay Euler roll wraps at inversion and its std/peak-to-peak are not vibration amplitudes.',
            'A single finite replay cannot establish root cause, actuator identification, or real hardware improvement.'],
        'sha256':{str(p):sha(p) for p in files}}
    a.output_prefix.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for name,v in windows.items():
        print(name,'input',v['observed_input'],'change',v['change_pct_if_upright'])
        for label,m in v['physical'].items():
            print(label,{k:m[k] for k in ('gyro_x_rms_rad_s','roll_std_deg','roll_peak_to_peak_deg','max_tilt_deg')},
                  m.get('mean_body_vx_mps'),m.get('min_height_m'),m['gyro_spectrum'])
    print('checks',checks)
    if a.plot:plot(a,data,obs,refs,targets,baseline_targets,cfg,start,end)


def plot(a,data,obs,refs,targets,baseline_targets,cfg,start,end):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def save(fig,suffix):
        fig.savefig(str(a.output_prefix)+'_'+suffix+'.png',dpi=150,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,7),layout='constrained')
    for axis,i,leg in zip(axes.flat,SHANKS,LEGS):
        for key,label,color in [('original_q','Original observed q','#0077bb'),('used_q','50% pull observed q','#ee7733'),('target_q','Original same-tick target','black')]:
            axis.plot([r['input_captured_s'] for r in refs],[r[f'{key}_{i:02}']*DEG for r in refs],label=label,color=color,lw=.9,ls='--' if key=='target_q' else '-')
        axis.set(title=leg+' shank input',xlabel='Input capture source time (s)',ylabel='deg');axis.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Only four shank position observations pulled 50% toward ORIGINAL same-tick targets\nOther 41 channels unchanged; targets are hindsight information, not new actor outputs')
    save(fig,'inputs')
    colors={'real':'#777777','baseline':'#0077bb','pull50':'#ee7733'}
    fig,axes=plt.subplots(3,2,figsize=(13,10),layout='constrained')
    terms=[('roll_deg','Roll','deg'),('gyro_x','Roll rate','rad/s'),('pitch_deg','Pitch','deg'),('body_vx','Forward speed','m/s'),('z','Base height','m'),('tilt_deg','Body tilt','deg')]
    for axis,(key,title,unit) in zip(axes.flat,terms):
        for label,rows in data.items():
            rr=[r for r in rows if start<=r['t_s']<=end]
            if key in rr[0]:
                values=[r[key] for r in rr]
                if key=='roll_deg':
                    continuous=[values[0]]
                    for u,v in zip(values,values[1:]):
                        continuous.append(continuous[-1]+math.degrees(math.atan2(math.sin(math.radians(v-u)),math.cos(math.radians(v-u)))))
                    values=continuous
                axis.plot([r['t_s'] for r in rr],values,label=label,color=colors[label],lw=.8)
        axis.axvspan(a.lo,a.hi,color='green',alpha=.07)
        first_unstable=next((r['t_s'] for r in data['pull50'] if r['tilt_deg']>20),None)
        first_inversion=next((r['t_s'] for r in data['pull50'] if r['tilt_deg']>60),None)
        if first_unstable is not None:axis.axvline(first_unstable,color='#cc6677',ls=':',lw=1)
        if first_inversion is not None:axis.axvspan(first_inversion,end,color='#cc6677',alpha=.10)
        if key=='roll_deg':title='Roll (continuous display; inverted after fall)'
        if key=='tilt_deg':
            axis.axhline(20,color='gray',ls=':',lw=.8);axis.axhline(60,color='gray',ls='--',lw=.8)
        axis.set(title=title,xlabel='Source time (s)',ylabel=unit);axis.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('Real obs -> same actor -> Gazebo | original vs shank-q 50% pull; P25/D0.8\nDotted line: first tilt >20deg; red band: tilt >60deg onward. Fallen motion is not walking jitter.')
    save(fig,'response')
    fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
    for axis,key,unit in zip(axes,('gyro_x','roll_deg'),('rad/s peak amplitude','deg peak amplitude')):
        for label,rows in data.items():
            _,bins=spectrum(wave(rows,key,a.lo,a.hi));m=summarize(rows,a.lo,a.hi)
            warning=' (UNSTABLE WINDOW)' if m.get('upright') is False or m['max_tilt_deg']>20 else ''
            axis.plot([x for x,_ in bins],[y for _,y in bins],label=label+warning,color=colors[label])
        axis.set(title=key,xlabel='Hz',ylabel=unit,xlim=(0,20));axis.grid(alpha=.2);axis.legend(fontsize=8)
    fig.suptitle(f'{a.lo}..{a.hi}s: detrend + Hann, 100Hz interpolation; frequency bin {1/(a.hi-a.lo):.3f}Hz')
    save(fig,'spectra')
    fig,axes=plt.subplots(4,3,figsize=(15,10),layout='constrained')
    for i,axis in enumerate(axes.flat):
        h=cfg['policy_to_hardware'][i]
        for rows,label,color in [(baseline_targets,'Original obs output','#0077bb'),(targets,'Shank-q pull output','#ee7733')]:
            axis.step([start+r['t_s'] for r in rows],[r[f'q_{h:02}']*DEG for r in rows],where='post',label=label,color=color,lw=.8)
        axis.set(title=LEGS[i//3]+' '+('hip','thigh','shank')[i%3],xlabel='Source time (s)',ylabel='Target deg');axis.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('All 12 recomputed joint targets: only 4 input q channels changed')
    save(fig,'targets')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('real','baseline','pull','inference','output-prefix'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--lo',type=float,default=38.7);p.add_argument('--hi',type=float,default=41.2)
    p.add_argument('--plot',action='store_true')
    a=p.parse_args()
    if a.hi<=a.lo:p.error('hi must be greater than lo')
    main(a)
