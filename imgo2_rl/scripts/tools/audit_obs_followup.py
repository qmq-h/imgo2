#!/usr/bin/env python3
"""Compare diagnostic obs interventions with normal Gazebo +0.5m/s motion.
Standard-library stats, optional matplotlib plot; independent steady windows.
"""
import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path
from audit_target_replay import load,at,roll,mean,rms,std
from audit_obs_smoothing import summarize
from audit_logged_oscillation_coupling import band


def waveform(rows,key,start,duration):
    times=[r['t_s'] for r in rows]
    return [at(rows,times,start+j*.01,key) for j in range(round(duration*100))]

def normal(root):
    imu=load(root/'imu.csv');odom=load(root/'odom.csv');state=load(root/'state.csv');hosts=[r['host_s'] for r in odom]
    for r in imu:r['t_s']=r['sim_s'];r['roll']=roll(r)
    for r in state:r['t_s']=at(odom,hosts,r['host_s'],'sim_s')
    for r in odom:r['t_s']=r['sim_s']
    lo,hi=12.9,18.8
    ii=[r for r in imu if lo<=r['t_s']<hi];ss=[r for r in state if lo<=r['t_s']<hi];oo=[r for r in odom if lo<=r['t_s']<hi]
    bv=[]
    for r in oo:
        w,x,y,z=[r['quat_'+k] for k in 'wxyz']
        bv.append((1-2*(y*y+z*z))*r['vx']+2*(x*y+w*z)*r['vy']+2*(x*z-w*y)*r['vz'])
    m={'gyro_x_rms':rms([r['gyro_x'] for r in ii]),'roll_std_deg':std([r['roll'] for r in ii]),
       'hip_dq_rms':rms([r[f'dq_{i:02}'] for r in ss for i in (0,3,6,9)]),'all_joint_dq_rms':rms([r[f'dq_{i:02}'] for r in ss for i in range(12)]),
       'mean_body_vx_mps':mean(bv),'mean_height_m':mean([r['z'] for r in oo]),'min_height_m':min(r['z'] for r in oo)}
    gy=waveform(imu,'gyro_x',lo,5.9)
    return {'source':str(root),'window_s':[lo,hi],'metrics':m,'gyro_band':band(gy),
            'hashes':{str(root/n):hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ('imu.csv','odom.csv','state.csv')}},gy


def main(a):
    n,ng=normal(a.normal);data={'Normal Gazebo':(n,ng)}
    for case in a.case:
        label,run,inf=case.split(',');s,rows,_=summarize(Path(run),Path(inf))
        gy=waveform(rows,'gyro_x',8,5.9);s['metrics']=s['windows'][-1]['metrics'];s['gyro_band']=band(gy)
        s['upright_motion']=s['metrics']['max_tilt_deg']<60 and s['metrics']['min_height_m']>.20
        if not s['upright_motion']:s['comparison_warning']='fell or lost upright posture; wrapped roll std and steady gyro RMS are not valid normal-walking comparisons'
        s['ratios_to_normal']={k:s['metrics'][k]/n['metrics'][k] for k in ('gyro_x_rms','roll_std_deg','hip_dq_rms')}
        data[label]=(s,gy)
    result={'comparison':'normal own-feedback PPO vs open-loop recorded-observation interventions; same model/controller/PD, different input trajectory',
            'sampled_band_hz':[4,10],'band_sampling_hz':100,'cases':{k:v[0] for k,v in data.items()},
            'limits':['no 17ms correction','causal filtering changes both magnitude and phase','reconstructed obs and real weight identity limitations retained','normal and replay windows are independent, not exact gait-phase aligned']}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for label,(s,_) in data.items():print(label,s['metrics'],s['gyro_band'])
    if a.plot:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        labels=list(data);ticklabels=[k if data[k][0].get('upright_motion',True) else k+' (FELL)' for k in labels];gyro=[data[k][0]['metrics']['gyro_x_rms'] if data[k][0].get('upright_motion',True) else float('nan') for k in labels];rv=[data[k][0]['metrics']['roll_std_deg'] if data[k][0].get('upright_motion',True) else float('nan') for k in labels];br=[data[k][0]['gyro_band']['band_4_10_hz_rms'] if data[k][0].get('upright_motion',True) else float('nan') for k in labels]
        fig,ax=plt.subplots(2,2,figsize=(14,9),layout='constrained')
        for axis,values,title,unit in [(ax[0,0],gyro,'Roll angular velocity RMS','rad/s'),(ax[0,1],rv,'Roll angle standard deviation','degrees'),(ax[1,0],br,'Roll-rate energy in 4-10Hz band','rad/s RMS')]:
            axis.bar(range(len(labels)),values);axis.set_xticks(range(len(labels)),ticklabels,rotation=30,ha='right');axis.set(title=title,ylabel=unit);axis.axhline(values[0],color='red',ls='--',lw=1,label='Normal Gazebo');axis.grid(axis='y',alpha=.25)
        for label,(s,values) in data.items():
            if label in ('Normal Gazebo','Raw obs','DQ 1Hz','Gyro 3Hz','Gyro + roll 3Hz'):
                ax[1,1].plot([j*.01 for j in range(len(values))],values,label=label,lw=.8)
        ax[1,1].set(title='Independent steady motion windows (not phase aligned)',xlabel='Window time (s)',ylabel='Roll rate (rad/s)');ax[1,1].legend(fontsize=8);ax[1,1].grid(alpha=.25)
        fig.suptitle('P25 / D0.8, same model, normal forward-motion reference\nSimulation diagnostics only; no phase correction',fontsize=12)
        fig.savefig(a.plot,dpi=160);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--normal',type=Path,required=True);p.add_argument('--case',action='append',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--plot',type=Path)
    main(p.parse_args())
