#!/usr/bin/env python3
"""Standard-library audit of diagnostic Gazebo target replay; optional plot."""
import argparse
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path

HIPS=(0,3,6,9)
def load(path):
    with path.open() as f:return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
def rms(v):return math.sqrt(sum(x*x for x in v)/len(v)) if v else None
def mean(v):return sum(v)/len(v) if v else None
def std(v):
    m=mean(v);return rms([x-m for x in v]) if v else None
def roll(r):
    w,x,y,z=[r['quat_'+k] for k in ('w','x','y','z')]
    return math.degrees(math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)))
def at(rows,times,t,key):
    n=min(len(times)-1,max(1,bisect.bisect_right(times,t)))
    a,b=rows[n-1],rows[n];f=max(0,min(1,(t-times[n-1])/(times[n]-times[n-1])))
    return a[key]*(1-f)+b[key]*f

def correlation(a,b):
    ma,mb=mean(a),mean(b)
    den=math.sqrt(sum((x-ma)**2 for x in a)*sum((x-mb)**2 for x in b))
    return sum((x-ma)*(y-mb) for x,y in zip(a,b))/den if den else None

def source(meta):
    root=Path(meta['source']);offset=meta['source_start_s']
    if meta['kind']=='real':
        r=load(root/'robot.csv')
        for a in r:a['t_s']=a['t_mono_ms']*.001-offset
        return r,r
    state=load(root/'state.csv');odom=load(root/'odom.csv');imu=load(root/'imu.csv')
    hosts=[r['host_s'] for r in odom]
    for a in state:a['t_s']=at(odom,hosts,a['host_s'],'sim_s')-offset
    for a in imu:a['t_s']=a['sim_s']-offset
    return state,imu

def metrics(rows,lo,hi,targets):
    r=[a for a in rows if lo<=a['t_s']<hi];tt=[x['t_s'] for x in targets]
    def trace(k,inds):return [a[f'{k}_{i:02}'] for a in r for i in inds]
    if not r:return {}
    hip_means={i:mean([a[f'q_{i:02}'] for a in r]) for i in HIPS}
    result={'samples':len(r),'hip_dq_rms':rms(trace('dq',HIPS)),
            'thigh_dq_rms':rms(trace('dq',(1,4,7,10))),
            'shank_dq_rms':rms(trace('dq',(2,5,8,11))),
            'hip_q_std_pooled':rms([a[f'q_{i:02}']-hip_means[i] for a in r for i in HIPS])}
    result['hip_tracking_rms']=rms([a[f'q_{i:02}']-targets[max(0,bisect.bisect_right(tt,a['t_s'])-1)][f'q_{i:02}'] for a in r for i in HIPS])
    if 'gyro_x' in r[0]:result.update(gyro_x_rms=rms([a['gyro_x'] for a in r]),gyro_y_rms=rms([a['gyro_y'] for a in r]),roll_std_deg=std([roll(a) for a in r]),roll_maxabs_deg=max(abs(roll(a)) for a in r))
    if 'z' in r[0]:result.update(mean_height_m=mean([a['z'] for a in r]),min_height_m=min(a['z'] for a in r))
    return result

def audit(directory):
    meta=json.loads((directory/'metadata.json').read_text());r=load(directory/'replay.csv');targets=load(directory/'targets.csv');published=load(directory/'published.csv');s,imu=source(meta)
    windows=[(0,.1),(.1,.5),(.5,1),(1,2),(2,4),(4,7.9)] if meta['kind']=='sim' else [(0,4),(4,5),(5,5.25),(5.25,6),(6,8),(8,13.9)]
    st=[a['t_s'] for a in s];it=[a['t_s'] for a in imu]
    table=[]
    for lo,hi in windows:
        rr=[a for a in r if lo<=a['t_s']<hi]
        a=metrics(r,lo,hi,targets);b=metrics(s,lo,hi,targets)
        ii=[a for a in imu if lo<=a['t_s']<hi]
        b.update(gyro_x_rms=rms([a['gyro_x'] for a in ii]),gyro_y_rms=rms([a['gyro_y'] for a in ii]),roll_std_deg=std([roll(a) for a in ii]))
        qerr=[a[f'q_{j:02}']-at(s,st,a['t_s'],f'q_{j:02}') for a in rr for j in range(12)]
        table.append({'window_s':[lo,hi],'replay':a,'source':b,'joint_q_rmse_rad':rms(qerr),
                      'gyro_x_rmse':rms([a['gyro_x']-at(imu,it,a['t_s'],'gyro_x') for a in rr])})
    lags=[a['t_s']-a['scheduled_s'] for a in published];ordered=sorted(lags)
    result={'run':str(directory),'metadata':meta,'windows':table,
            'target_publish_check':{'source_rows':len(targets),'published_rows':len(published),'all_indices_in_order':all(int(a['index'])==i for i,a in enumerate(published)),
                                    'schedule_lag_mean_s':mean(lags),'schedule_lag_p99_s':ordered[int(.99*(len(ordered)-1))],'schedule_lag_max_s':max(lags)},
            'sha256':{n:hashlib.sha256((directory/n).read_bytes()).hexdigest() for n in ('replay.csv','published.csv','targets.csv','metadata.json')},
            'limitations':['ROS publish timing is not controller receipt/wire time','real initial base height/velocity and contacts unknown','open-loop divergence does not identify original closed-loop cause','replay tau_NN GetForce at update-begin is not useful; controller_tau is asynchronous feedback']}
    if meta['kind']=='real':
        early=[a for a in r if 6.5<=a['t_s']<8]
        scans={}
        for key in ('gyro_x','q_00','dq_00'):
            x=[a[key] for a in early]
            values=[(correlation(x,[at(s,st,a['t_s']+ms*.001,key) for a in early]),ms) for ms in range(-30,31)]
            best,shift=max(values)
            scans[key]={'same_time_correlation':correlation(x,[at(s,st,a['t_s'],key) for a in early]),'best_correlation':best,'later_real_shift_ms':shift}
        result['early_alignment_scan']={'source_window_s':[28.5,30.0],'search_ms':[-30,30],'signals':scans,'meaning':'positive shift: later real samples best match current simulation; aggregate phase clue, NOT measured wire/USB delay'}
    return result,r,s,imu,targets

def plot_all(data,path):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(3,2,figsize=(14,10),layout='constrained')
    _,selfr,selfs,selfi,_=data[0];_,rr,rs,ri,tg=data[1]
    def draw(a,rows,key,label,offset=0):a.plot([r['t_s']+offset for r in rows],[r[key] for r in rows],label=label,lw=.8)
    draw(ax[0,0],selfr,'q_00','Replay');draw(ax[0,0],[r for r in selfs if 0<=r['t_s']<=8],'q_00','Original Gazebo')
    ax[0,0].set(title='Self replay: FL hip position',ylabel='rad')
    draw(ax[0,1],selfr,'gyro_x','Replay');draw(ax[0,1],[r for r in selfi if 0<=r['t_s']<=8],'gyro_x','Original Gazebo');ax[0,1].set(title='Self replay: body roll rate',ylabel='rad/s')
    real=[r for r in rs if 0<=r['t_s']<=14]
    for rows,label in ((rr,'Gazebo (real targets)'),(real,'Real robot')):
        ax[1,0].plot([r['t_s']+22 for r in rows],[roll(r) for r in rows],label=label,lw=.8)
        draw(ax[1,1],rows,'gyro_x',label,22)
        draw(ax[2,0],rows,'q_00',label,22)
        draw(ax[2,1],rows,'dq_00',label,22)
    draw(ax[2,0],tg,'q_00','Identical target',22)
    ax[1,0].set(title='Real-target replay: body roll',ylabel='degrees')
    ax[1,1].set(title='Real-target replay: body roll rate',ylabel='rad/s')
    ax[2,0].set(title='Real-target replay: FL hip position',ylabel='rad')
    ax[2,1].set(title='Real-target replay: FL hip velocity',ylabel='rad/s')
    for a in ax.flat:a.grid(alpha=.25);a.legend(fontsize=8);a.set_xlabel('Source time (s)')
    fig.suptitle('P25 / D0.8, original Gazebo controller, no policy inference\nReal base height/velocity and contact initial state were not measured',fontsize=12)
    fig.savefig(path,dpi=160);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--self-run',type=Path,required=True);p.add_argument('--real-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--plot',type=Path)
    a=p.parse_args();data=[audit(a.self_run),audit(a.real_run)]
    a.output.write_text(json.dumps({'self_replay':data[0][0],'real_target_replay':data[1][0]},indent=2,allow_nan=False)+'\n')
    if a.plot:plot_all(data,a.plot)
    for d,*_ in data:
        print(d['run'],d['target_publish_check'])
        for w in d['windows']:print(w['window_s'],'qRMSE',w['joint_q_rmse_rad'],'gyro replay/source',w['replay'].get('gyro_x_rms'),w['source'].get('gyro_x_rms'),'hipdq',w['replay'].get('hip_dq_rms'),w['source'].get('hip_dq_rms'))
