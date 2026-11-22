#!/usr/bin/env python3
"""Stdlib diagnostics for causal observation smoothing in Gazebo ONLY."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from audit_target_replay import load,metrics,roll,mean,rms


def summarize(run,inference):
    rows=load(run/'replay.csv');targets=load(run/'targets.csv');obs=load(inference/'obs_used.csv')
    publication=load(run/'published.csv');meta=json.loads((run/'metadata.json').read_text());inf=json.loads((inference/'inference_summary.json').read_text())
    windows=[]
    for lo,hi in ((4,5.8),(6,8),(8,13.9)):
        selected=[r for r in rows if lo<=r['t_s']<hi];m=metrics(rows,lo,hi,targets)
        bv=[];tilt=[]
        for r in selected:
            w,x,y,z=[r['quat_'+k] for k in ('w','x','y','z')]
            bv.append((1-2*(y*y+z*z))*r['vx']+2*(x*y+w*z)*r['vy']+2*(x*z-w*y)*r['vz'])
            tilt.append(math.degrees(math.acos(max(-1,min(1,1-2*(x*x+y*y))))))
        m['mean_body_vx_mps']=mean(bv);m['max_tilt_deg']=max(tilt)
        m['horizontal_displacement_m']=math.hypot(selected[-1]['x']-selected[0]['x'],selected[-1]['y']-selected[0]['y'])
        windows.append({'source_window_s':[lo+22,hi+22],'metrics':m})
    selected=[r for r in obs if 30<=r['source_s']<35.9]
    changed=[i for i in range(45) if any(abs(r[f'original_{i:02}']-r[f'used_{i:02}'])>1e-12 for r in obs)]
    checks={'planned':len(targets),'published':len(publication),'indices_in_order':all(int(r['index'])==i for i,r in enumerate(publication)),
            'max_publication_lag_s':max(r['t_s']-r['scheduled_s'] for r in publication),'changed_obs_channels':changed,
            'gravity_norm_max_delta':max(abs(math.sqrt(sum(r[f'original_{i:02}']**2 for i in (3,4,5)))-math.sqrt(sum(r[f'used_{i:02}']**2 for i in (3,4,5)))) for r in obs),
            'pitch_max_delta_rad':max(abs(math.atan2(r['original_03'],math.hypot(r['original_04'],r['original_05']))-math.atan2(r['used_03'],math.hypot(r['used_04'],r['used_05']))) for r in obs),
            'unchanged_gyro':all(r[f'original_{i:02}']==r[f'used_{i:02}'] for r in obs for i in range(3)),
            'unchanged_previous_actions':all(r[f'original_{i:02}']==r[f'used_{i:02}'] for r in obs for i in range(33,45)),
            'production_files_unchanged':meta['production_files_unchanged'],'completed':meta.get('completed',False)}
    input_summary={}
    for name,start,stop,scale in (('hip_dq',21,33,.05),):
        input_summary['dq_physical_all_joint_rms']={k:rms([r[f'{k}_{i:02}']/scale for r in selected for i in range(start,stop)]) for k in ('original','used')}
    input_summary['dq_physical_hip_rms']={k:rms([r[f'{k}_{i:02}']/.05 for r in selected for i in (21,24,27,30)]) for k in ('original','used')}
    input_summary['gyro_x_physical_rms']={k:rms([r[f'{k}_00']/.25 for r in selected]) for k in ('original','used')}
    def obs_roll(r,k):return math.degrees(math.atan2(-r[f'{k}_04'],-r[f'{k}_05']))
    input_summary['roll_deg_rms']={k:rms([obs_roll(r,k) for r in selected]) for k in ('original','used')}
    return {'run':str(run),'inference':str(inference),'inference_summary':inf,'checks':checks,'input_summary':input_summary,'windows':windows,
            'hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (run/'replay.csv',inference/'obs_used.csv',run/'targets.csv')}},rows,obs

def plot(data,path):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(3,2,figsize=(14,10),layout='constrained')
    for label,(summary,rows,obs) in data.items():
        s=[r for r in rows if 5.5<=r['t_s']<=14];t=[r['t_s']+22 for r in s]
        ax[0,0].plot(t,[r['gyro_x'] for r in s],label=label,lw=.8)
        ax[0,1].plot(t,[roll(r) for r in s],label=label,lw=.8)
        ax[1,0].plot(t,[r['dq_00'] for r in s],label=label,lw=.8)
        ax[1,1].plot(t,[r['z'] for r in s],label=label,lw=.8)
        o=[r for r in obs if 27.5<=r['source_s']<=36];ot=[r['source_s'] for r in o]
        ax[2,0].plot(ot,[r['used_21']/.05 for r in o],label=label,lw=.8)
        ax[2,1].plot(ot,[math.degrees(math.atan2(-r['used_04'],-r['used_05'])) for r in o],label=label,lw=.8)
    titles=[('Body roll rate','rad/s'),('Body roll','deg'),('Physical FL hip velocity','rad/s'),('Base height','m'),('Policy input FL hip velocity','rad/s, scale removed'),('Policy input gravity-derived roll','deg')]
    for axis,(title,unit) in zip(ax.flat,titles):axis.set(title=title,ylabel=unit,xlabel='Original log time (s), no shift');axis.grid(alpha=.25);axis.legend(fontsize=8)
    fig.suptitle('Causal 3Hz observation smoothing, P25 / D0.8\nRecorded gyro and last-action inputs unchanged; no phase compensation',fontsize=12)
    fig.savefig(path,dpi=160);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',action='append',required=True,help='label,run_directory,inference_directory');p.add_argument('--output',type=Path,required=True);p.add_argument('--plot',type=Path)
    a=p.parse_args();data={}
    for case in a.case:
        label,run,inf=case.split(',');data[label]=summarize(Path(run),Path(inf))
    result={'cases':{k:v[0] for k,v in data.items()},'limits':['reconstructed logged obs, not actual forward input','open-loop replay, recorded last actions retained','filter changes both frequency content and phase; cannot identify sampling fault','roll filtering preserves pitch; gyro is NOT filtered','only one cutoff/model/window; not hardware safety validation']}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if a.plot:plot(data,a.plot)
    for name,(s,_,_) in data.items():
        print(name,s['checks'])
        for w in s['windows']:print(w['source_window_s'],json.dumps(w['metrics']))
