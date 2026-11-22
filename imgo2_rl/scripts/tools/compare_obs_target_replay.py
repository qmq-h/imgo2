#!/usr/bin/env python3
"""Stdlib summary: real robot, logged-target replay, logged-obs actor replay."""
import argparse
import hashlib
import json
from pathlib import Path
from audit_target_replay import load,metrics,source,roll


def compare(a):
    old=load(a.target_run/'replay.csv');new=load(a.obs_run/'replay.csv')
    meta=json.loads((a.obs_run/'metadata.json').read_text())
    real,_=source(meta)
    tg=load(a.target_run/'targets.csv');ntg=load(a.obs_run/'targets.csv')
    published=load(a.obs_run/'published.csv')
    publication_check={'planned':len(ntg),'published':len(published),'indices_in_order':all(int(r['index'])==i for i,r in enumerate(published)),'max_schedule_lag_s':max(r['t_s']-r['scheduled_s'] for r in published)}
    rows=[]
    for lo,hi in ((4,5.8),(6,8),(8,13.9)):
        rows.append({'source_window_s':[lo+22,hi+22],
                     'real':metrics(real,lo,hi,tg),
                     'logged_target_replay':metrics(old,lo,hi,tg),
                     'logged_obs_actor_replay':metrics(new,lo,hi,ntg)})
    result={'time_correction_ms':0,'comparison_windows':rows,'publication_check':publication_check,
            'inference':json.loads((a.inference/'inference_summary.json').read_text()),
            'obs_replay_metadata':meta,'logged_target_metadata':json.loads((a.target_run/'metadata.json').read_text()),
            'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.obs_run/'replay.csv',a.target_run/'replay.csv',a.inference/'inference.csv')},
            'limits':['policy.csv contains reconstructed, not actual forward obs','old9299 candidate identity on real not hash-verified','open-loop obs playback; recorded last actions unchanged','no lag scan or time correction used']}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if a.plot:plot(a,real,old,new)
    for r in rows:
        print(r['source_window_s'])
        for k in ('real','logged_target_replay','logged_obs_actor_replay'):
            print(k,json.dumps(r[k]))


def plot(a,real,old,new):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    inf=load(a.inference/'inference.csv')
    fig,ax=plt.subplots(3,2,figsize=(14,10),layout='constrained')
    for k,label in [('logged_action_00','Logged action'),('action_00','Recomputed from logged obs')]:
        ax[0,0].plot([r['source_s'] for r in inf],[r[k] for r in inf],label=label,lw=.8)
    ax[0,0].set(title='FL hip actor output (after action clipping)',ylabel='action')
    for k,label in [('logged_action_02','Logged action'),('action_02','Recomputed from logged obs')]:
        ax[0,1].plot([r['source_s'] for r in inf],[r[k] for r in inf],label=label,lw=.8)
    ax[0,1].set(title='FL shank actor output (after action clipping)',ylabel='action')
    for rows,label in ((real,'Real robot'),(old,'Gazebo: logged targets'),(new,'Gazebo: actor(logged obs)')):
        rows=[r for r in rows if 0<=r['t_s']<=14]
        t=[r['t_s']+22 for r in rows]
        ax[1,0].plot(t,[r['gyro_x'] for r in rows],label=label,lw=.7)
        ax[1,1].plot(t,[roll(r) for r in rows],label=label,lw=.7)
        ax[2,0].plot(t,[r['dq_00'] for r in rows],label=label,lw=.7)
        ax[2,1].plot(t,[r['q_00'] for r in rows],label=label,lw=.7)
    for axis,title,unit in [(ax[1,0],'Body roll rate','rad/s'),(ax[1,1],'Body roll','degrees'),(ax[2,0],'FL hip velocity','rad/s'),(ax[2,1],'FL hip position','rad')]:axis.set(title=title,ylabel=unit)
    for axis in ax.flat:axis.set_xlim(27.5,36);axis.grid(alpha=.25);axis.legend(fontsize=8);axis.set_xlabel('Original log time (s), no shift')
    fig.suptitle('Exploratory logged-obs playback, old9299 candidate, P25 / D0.8\nRecorded obs are reconstructed at consumption; recorded last-action inputs retained',fontsize=12)
    fig.savefig(a.plot,dpi=160);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--target-run',type=Path,required=True);p.add_argument('--obs-run',type=Path,required=True);p.add_argument('--inference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--plot',type=Path)
    compare(p.parse_args())
