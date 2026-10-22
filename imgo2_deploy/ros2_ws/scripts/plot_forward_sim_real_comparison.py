#!/usr/bin/env python3
"""Offline sim/real +0.5 m/s command comparison, with no phase/lag fitting.
Requires numpy/matplotlib. Torque is reconstructed nominal PD, not measured demand.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[3]

def load(path):
    if not path.exists():path=path.with_suffix(path.suffix+'.gz')
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:
        r=csv.DictReader(f);names=r.fieldnames;rows=[[float(x[k]) for k in names] for x in r]
    a=np.asarray(rows);assert np.isfinite(a).all(),str(path)
    return {k:a[:,i] for i,k in enumerate(names)}

def quat_rpy(c):
    w,x,y,z=[c['quat_'+k] for k in 'wxyz'];n=np.sqrt(w*w+x*x+y*y+z*z);w,x,y,z=[v/n for v in (w,x,y,z)]
    return np.array([np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)),np.arcsin(np.clip(2*(w*y-z*x),-1,1)),np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))])*180/np.pi

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sim',type=Path,required=True);p.add_argument('--real',type=Path,required=True);p.add_argument('--output-prefix',type=Path,required=True);a=p.parse_args();a.output_prefix.parent.mkdir(parents=True,exist_ok=True)
    for f in font_manager.fontManager.ttflist:
        if f.name in ('Noto Sans CJK JP','WenQuanYi Micro Hei','Noto Sans CJK SC'):plt.rcParams['font.family']=f.name;break
    plt.rcParams['axes.unicode_minus']=False
    real_csv=a.real/'robot.csv' if a.real.is_dir() else a.real
    r=load(real_csv);rt=r['t_mono_ms']/1000
    ok=(r['mode']==3)&np.isclose(r['cmd_vx'],.5)&np.isclose(r['cmd_vy'],0)&np.isclose(r['cmd_wz'],0)
    edges=np.diff(np.r_[False,ok,False].astype(int));runs=list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))
    lo,hi=max(runs,key=lambda v:rt[v[1]-1]-rt[v[0]]);ri=np.arange(lo,hi);duration=rt[hi-1]-rt[lo];tr=rt[ri]-rt[lo]
    s=load(a.sim/'state.csv');cmd=load(a.sim/'command.csv');imu=load(a.sim/'imu.csv');odom=load(a.sim/'odom.csv');vel=load(a.sim/'velocity.csv')
    clock=load(a.sim/'clock.csv');origin=clock['sim_s'][0]
    # Unstamped motor topics use a host-to-simulation clock interpolation.
    # IMU/odom use their original message stamps; never the stale latest /clock value.
    for stream in (s,cmd):
        stream['t_s']=np.interp(stream['host_s'],clock['host_s'],clock['sim_s'])-origin-6
    start=14.0;si=np.flatnonzero((s['t_s']>=start)&(s['t_s']<=start+duration));ts=s['t_s'][si]-start
    ii=np.flatnonzero((imu['t_s']>=start)&(imu['t_s']<=start+duration));ti=imu['t_s'][ii]-start
    assert len(si)>100 and len(ii)>100 and duration>1
    vi=(vel['t_s']>=start)&(vel['t_s']<=start+duration);assert np.allclose(vel['vx'][vi],.5) and np.allclose(vel['vy'][vi],0) and np.allclose(vel['wz'][vi],0)
    ci=np.searchsorted(cmd['host_s'],s['host_s'][si],side='right')-1;assert (ci>=0).all()
    assert all(np.allclose(cmd[f'kp_{j:02}'][ci],25) and np.allclose(cmd[f'kd_{j:02}'][ci],.5) for j in range(12))
    assert all(np.allclose(r[f'kp_{j:02}'][ri],25) and np.allclose(r[f'kd_{j:02}'][ri],.5) for j in range(12))
    report={'real_log':str(a.real.resolve().relative_to(ROOT)) if a.real.resolve().is_relative_to(ROOT) else str(a.real.resolve()),'real_window_s':[float(rt[lo]),float(rt[hi-1])],'sim_window_ppo_s':[start,start+duration],'duration_s':float(duration),'sim_state_samples':len(si),'sim_imu_samples':len(ii),'real_samples':len(ri),'command':[.5,0,0],'gains':[25,.5],'alignment':'independent window starts; no gait phase or lag fitting','contact':'real lift/contact unlabelled','joints':{}}
    fig,axes=plt.subplots(12,3,figsize=(20,25),sharex=True)
    for j in range(12):
        key=f'{j:02}';name=['FL','FR','RL','RR'][j//3]+' '+['髋','大腿','小腿'][j%3]
        sq=s['q_'+key][si];sd=s['dq_'+key][si];rq=r['q_'+key][ri];rd=r['dq_'+key][ri]
        sqd=cmd['q_'+key][ci];rqd=r['q_des_'+key][ri]
        spd=cmd['kp_'+key][ci]*(sqd-sq)+cmd['kd_'+key][ci]*(cmd['dq_'+key][ci]-sd)+cmd['tau_'+key][ci]
        rpd=r['kp_'+key][ri]*(rqd-rq)+r['kd_'+key][ri]*(r['dq_des_'+key][ri]-rd)+r['tau_ff_'+key][ri]
        axes[j,0].plot(ts,sq,color='#167cb5',lw=.8,label='仿真反馈');axes[j,0].plot(tr,rq,color='#e07724',lw=.8,label='实机反馈')
        axes[j,0].plot(ts,sqd,color='#167cb5',ls=':',lw=.65,alpha=.65,label='仿真目标');axes[j,0].plot(tr,rqd,color='#e07724',ls=':',lw=.65,alpha=.65,label='实机目标')
        axes[j,1].plot(ts,sd,color='#167cb5',lw=.75,label='仿真反馈');axes[j,1].plot(tr,rd,color='#e07724',lw=.75,label='实机反馈')
        axes[j,2].plot(ts,s['tau_'+key][si],color='#167cb5',lw=.8,label='仿真effort反馈');axes[j,2].plot(tr,r['tau_'+key][ri],color='#e07724',lw=.8,label='实机估算力矩')
        axes[j,2].plot(ts,spd,color='#167cb5',ls='--',lw=.6,alpha=.7,label='仿真名义PD');axes[j,2].plot(tr,rpd,color='#e07724',ls='--',lw=.6,alpha=.7,label='实机名义PD')
        axes[j,0].set_ylabel(name+'\n位置(rad)');axes[j,1].set_ylabel('速度(rad/s)');axes[j,2].set_ylabel('力矩(N·m)')
        report['joints'][name]={}
        for k,sv,rv in [('q_std',sq,rq),('dq_rms',sd,rd),('tau_rms',s['tau_'+key][si],r['tau_'+key][ri]),('position_error_rms',sqd-sq,rqd-rq),('pd_feedback_residual_rms',spd-s['tau_'+key][si],rpd-r['tau_'+key][ri])]:
            calc=(lambda v:float(np.std(v))) if k=='q_std' else (lambda v:float(np.sqrt(np.mean(v*v))))
            report['joints'][name][k]={'sim':calc(sv),'real':calc(rv)}
    for k,title in enumerate(['关节位置：实线反馈 / 点线目标','关节速度：反馈','力矩：实线反馈 / 虚线名义PD重建']):axes[0,k].set_title(title)
    for ax in axes.flat:ax.grid(alpha=.22);ax.set_xlim(0,duration)
    for ax in axes[0]:ax.legend(fontsize=7,ncol=2,loc='upper right')
    for ax in axes[-1]:ax.set_xlabel('各自窗口起点后的时间(s)，未对齐步态相位')
    subtitle=f'vx指令=+0.5 m/s，vy=wz=0；Kp/Kd=25/0.5；实机新日志{rt[lo]:.3f}–{rt[hi-1]:.3f}s / 仿真PPO {start:.2f}–{start+duration:.2f}s'
    fig.suptitle('PPO仿真与实机：12关节位置、速度及力矩\n'+subtitle+'\n名义PD=Kp(q目标−q反馈)+Kd(dq目标−dq反馈)+前馈；未做10ms平移，不代表电机内部真实目标力矩',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.958));prefix=str(a.output_prefix);fig.savefig(prefix+'_joints.png',dpi=140);fig.savefig(prefix+'_joints.pdf');plt.close(fig)
    sr=quat_rpy(imu)[:,ii];rr=quat_rpy(r)[:,ri];sr[2]-=sr[2,0];rr[2]-=rr[2,0]
    fig,axes=plt.subplots(3,2,figsize=(16,10),sharex=True);report['imu']={}
    for k,label in enumerate(['Roll','Pitch','Yaw相对窗口起点']):
        axes[k,0].plot(ti,sr[k],color='#167cb5',lw=.8,label='Gazebo');axes[k,0].plot(tr,rr[k],color='#e07724',lw=.8,label='实机');axes[k,0].set_ylabel(label+'(deg)')
        sg=imu['gyro_'+'xyz'[k]][ii];rg=r['gyro_'+'xyz'[k]][ri]
        axes[k,1].plot(ti,sg,color='#167cb5',lw=.75,label='Gazebo');axes[k,1].plot(tr,rg,color='#e07724',lw=.75,label='实机');axes[k,1].set_ylabel('Gyro '+'XYZ'[k]+'(rad/s)')
        report['imu']['xyz'[k]]={'sim_gyro_rms':float(np.sqrt(np.mean(sg*sg))),'real_gyro_rms':float(np.sqrt(np.mean(rg*rg))),'sim_rpy_std_deg':float(np.std(sr[k])),'real_rpy_std_deg':float(np.std(rr[k]))}
    for ax in axes.flat:ax.grid(alpha=.25);ax.legend(loc='upper right');ax.set_xlim(0,duration)
    for ax in axes[-1]:ax.set_xlabel('各自窗口起点后的时间(s)，原始数据无平滑')
    fig.suptitle('PPO仿真与实机：IMU姿态及角速度\n'+subtitle+'\n实机没有加速度记录/实际机身速度；托举时刻未知；Roll/Pitch保留绝对值，Yaw各自置零',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(prefix+'_imu.png',dpi=160);fig.savefig(prefix+'_imu.pdf');plt.close(fig)
    oi=(odom['t_s']>=start)&(odom['t_s']<=start+duration)
    ot=odom['t_s'][oi];ox=odom['x'][oi];oy=odom['y'][oi];yaw=odom['yaw'][oi]
    vx=np.gradient(ox,ot)*np.cos(yaw)+np.gradient(oy,ot)*np.sin(yaw)
    report['sim_body_vx_mean_mps']=float(np.mean(vx));report['sim_body_vx_std_mps']=float(np.std(vx))
    report['input_sha256']={str(p.resolve().relative_to(ROOT)) if p.resolve().is_relative_to(ROOT) else str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [real_csv,a.sim/'metadata.json']}
    report['sim_time_basis']='IMU/odom header stamps; unstamped motor topics mapped from host arrivals by clock interpolation, not exact device sampling times'
    report['sim_command_match_host_age_ms']={'median':float(np.median((s['host_s'][si]-cmd['host_s'][ci])*1000)),'max':float(np.max((s['host_s'][si]-cmd['host_s'][ci])*1000))}
    report['real_context']='.2 to .5 command ramp in about .100s immediately before selected window; actual base speed and contact unknown; sim window is late steady command'
    Path(prefix+'.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))

if __name__=='__main__':main()
