#!/usr/bin/env python3
"""Compare full-feedback command-history simulation with real and target replay.

Statistics use stdlib. Matplotlib is imported only with --plot. No phase fitting.
"""
import argparse
import bisect
import csv
import json
from pathlib import Path
import statistics as st
from audit_paired_obs_replay import load, attitude, summarize, rms, wave, spectrum, sha

def map_host(rows, references):
    tt=[x['host_s'] for x in references]
    for r in rows:
        i=min(len(tt)-1,max(1,bisect.bisect_right(tt,r['host_s'])))
        a,b=references[i-1:i+1]
        r['t_s']=a['sim_s']+(r['host_s']-a['host_s'])/(b['host_s']-a['host_s'])*(b['sim_s']-a['sim_s'])

def main(a):
    real=load(a.real/'robot.csv')
    for r in real:r['t_s']=r['t_mono_ms']/1000;attitude(r)
    rr=load(a.target/'replay.csv')
    offset=json.loads((a.target/'metadata.json').read_text())['source_start_s']
    for r in rr:r['t_s']+=offset;attitude(r)
    result={'windows':{},'runs':{},'frequency_method':'100Hz interpolation; detrend+Hann; 1-20Hz peak, 0.4Hz resolution for 2.5s; no phase fit',
            'limitations':['Command-history test starts from normal simulated stand, not measured real full-body/contact state.',
                           'Captured command profile is held at observed input times; ROS scheduling and PPO-entry anchor are approximate.',
                           'Same-target replay is open loop; command-history and constant runs use their own simulated feedback.',
                           'No hardware repair or unique causal attribution from this experiment.']}
    with (a.real/'policy.csv').open() as f:real_policy=list(csv.DictReader(f))
    rp=[r for r in real_policy if 38.7<=float(r['input_captured_ms'])/1000<41.2]
    result['real_core_policy']={'samples':len(rp),
        'hip_target_step_rms_rad':rms([(float(b[f'act_{i:02}'])-float(x[f'act_{i:02}']))*.125 for x,b in zip(rp,rp[1:]) for i in (0,3,6,9)]),
        'hip_dq_input_rms_rad_s':rms([float(r[f'obs_{21+i:02}'])/.05 for r in rp for i in (0,3,6,9)])}
    cases={'real':(real,38.7,41.2),'real_targets_gazebo':(rr,38.7,41.2)}
    history=None
    for label,path in [('constant05',a.constant),('command_history',a.history)]:
        c=json.loads((path/'collection.json').read_text());m=json.loads((path/'metadata.json').read_text())
        imu=load(path/'imu.csv');odom=load(path/'odom.csv');policy=load(path/'policy_raw.csv')
        for r in imu+odom:r['t_s']=r['sim_s'];attitude(r)
        if label=='command_history':
            delta=c['profile_source_origin_s']-c['ppo_pd_start_sim_s']
            for r in imu+odom:r['t_s']+=delta
            lo,hi=38.7,41.2;history=imu
        else:
            delta=0;lo=c['ppo_pd_start_sim_s']+2;hi=lo+2.5
        cases[label]=(imu,lo,hi)
        # Check actual actor commands and gains using host->stamped-odom time.
        map_host(policy,odom)
        for r in policy:r['t_s']+=delta
        pp=[r for r in policy if lo<=r['t_s']<hi]
        if not pp:raise ValueError('No actor samples in core window')
        result['runs'][label]={'collection':c,'metadata':m,
            'core_actor_count':len(pp),'core_actor_commands':{k:[min(r[k] for r in pp),max(r[k] for r in pp)] for k in ('cmd_vx','cmd_vy','cmd_wz')},
            'core_kp_range':[min(r[f'kp_{i}'] for r in pp for i in range(12)),max(r[f'kp_{i}'] for r in pp for i in range(12))],
            'core_kd_range':[min(r[f'kd_{i}'] for r in pp for i in range(12)),max(r[f'kd_{i}'] for r in pp for i in range(12))],
            'hip_target_step_rms_rad':rms([b[f'target_{i}']-x[f'target_{i}'] for x,b in zip(pp,pp[1:]) for i in (0,3,6,9)]),
            'hip_dq_input_rms_rad_s':rms([r[f'obs_{21+i}']/.05 for r in pp for i in (0,3,6,9)]),
            'motion':{'body_vx_mean_mps':st.mean(r['body_vx'] for r in odom if lo<=r['t_s']<hi),'height_min_m':min(r['z'] for r in odom if lo<=r['t_s']<hi)},
            'sha256':{name:sha(path/name) for name in ('imu.csv','policy_raw.csv','collection.json','metadata.json')}}
        if label=='command_history':
            with (a.real/'policy.csv').open() as f:source=list(csv.DictReader(f))
            ts=[float(r['input_captured_ms'])/1000 for r in source]
            errors={k:[] for k in ('cmd_vx','cmd_vy','cmd_wz')}
            for r in policy:
                if not ts[0]<=r['t_s']<ts[0]+24:continue
                src=source[max(0,bisect.bisect_right(ts,r['t_s'])-1)]
                for k in errors:errors[k].append(r[k]-float(src[k]))
            result['runs'][label]['actor_command_vs_source_no_shift']={k:{'rms':rms(e),'max_abs':max(map(abs,e))} for k,e in errors.items()}
    for label,(rows,lo,hi) in cases.items():result['windows'][label]=summarize(rows,lo,hi)
    result['history_segments']={label:{'real':summarize(real,lo,hi),'gazebo':summarize(history,lo,hi)}
        for label,lo,hi in [('quiet_before_turn',21,22.5),('backward',28,31),('quiet_before_forward',37,38),('forward',38.7,41.2)]}
    a.output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    if a.plot:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(2,2,figsize=(13,7),constrained_layout=True)
        for label,(rows,lo,hi) in cases.items():
            subset=[r for r in rows if lo<=r['t_s']<hi]
            ax[0,0].plot([r['t_s']-lo for r in subset],[r['gyro_x'] for r in subset],label=label,alpha=.8,lw=1)
            ax[1,0].plot([r['t_s']-lo for r in subset],[r['roll_deg'] for r in subset],label=label,alpha=.8,lw=1)
            _,spec=spectrum(wave(rows,'gyro_x',lo,hi));ax[0,1].plot([x[0] for x in spec],[x[1] for x in spec],label=label)
        for label,rows in [('real',real),('command_history',history)]:
            sel=[r for r in rows if 20.1<=r['t_s']<44]
            ax[1,1].plot([r['t_s'] for r in sel],[r['gyro_x'] for r in sel],label=label,lw=.7)
        for axis,title in zip(ax.flat,['Forward window: roll rate','Forward window: roll-rate spectrum','Forward window: roll angle','Same velocity-command history; own feedback']):
            axis.set_title(title);axis.grid(alpha=.25);axis.legend(fontsize=8)
        ax[0,0].set(xlabel='Within-window time (s); independent phase',ylabel='rad/s')
        ax[1,0].set(xlabel='Within-window time (s); independent phase',ylabel='deg')
        ax[0,1].set(xlabel='Hz',ylabel='Hann sinusoid amplitude (rad/s)',xlim=(1,15))
        ax[1,1].set(xlabel='Real source time (s); entry-aligned',ylabel='rad/s')
        fig.savefig(a.output.with_suffix('.png'),dpi=160);plt.close(fig)
    print(json.dumps({k:{'gyro_rms':v['gyro_x_rms_rad_s'],'roll_std':v['roll_std_deg'],'peak_hz':v['gyro_spectrum']['peak_1_20_hz']} for k,v in result['windows'].items()},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('real','target','constant','history','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--plot',action='store_true');main(p.parse_args())
