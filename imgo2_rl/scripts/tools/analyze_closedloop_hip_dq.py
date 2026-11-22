#!/usr/bin/env python3
"""Audit fresh own-feedback Gazebo hip-dq ablations; optional scientific plots."""
import argparse
import csv
import json
import math
import statistics as st
import subprocess
from pathlib import Path

from audit_paired_obs_replay import load, rms, sha, spectrum, wave
from compare_gazebo_pd import closed, array
from analyze_command_history import map_host

HIPS=(0,3,6,9)

def main(a):
    plan=json.loads(a.plan.read_text());lo,hi=plan['window_after_ppo_s']
    result={'plan':plan,'cases':{},'limits':[
        'All current IMU, q, dq and last actions evolve in this own-feedback simulation; only four actor input channels are scaled.',
        'Scaling actor-observed hip dq does not prescribe or directly scale physical hip velocity or IMU.',
        'Normal Gazebo baseline is stable; this experiment does not recreate or identify the real loaded oscillation.',
        'One accepted 12s case per gain; 6s main window, no significance claims or gait phase fitting.',
        'Existing separate position/velocity queues are unchanged; their pending timing issue is not isolated here.']}
    data={};table=[];reference=None
    if a.verify_actor:
        import torch
        torch.set_num_threads(1)
    for case in plan['cases']:
        label=case['label'];path=Path(case['run']);gain=case['gain']
        if case['status']!='complete':
            result['cases'][label]={'status':case['status'],'path':str(path),'gain':gain}
            continue
        report,(imu,odom,policy)=closed(path,.8,lo,hi,25.)
        runner=Path(case['runner']);build=json.loads((runner/'build_metadata.json').read_text());hook=build['observation_intervention']
        assert hook['hip_dq_gain']==gain and hook['obs_indices']==[21,24,27,30]
        original=Path(hook['original_source']);copy=Path(hook['isolated_source'])
        assert sha(original)==hook['original_sha256'] and sha(copy)==hook['isolated_sha256']
        assert copy.read_text().replace(hook['injected_text'],'')==original.read_text()
        cfg=(runner/'policy/imgo2/ppo/config.yaml').read_text();defaults=array(cfg,'default_dof_pos');scales=array(cfg,'action_scale')
        identity=[report[k] for k in ('policy_sha256','base_sha256','config_except_pd','model_sha256')]
        if reference is None:reference=identity
        assert identity==reference
        pp=[r for r in policy if lo<=r['t_s']<hi];errors={k:[] for k in ('gyro','gravity','q','dq_scaled','last_action','nominal_tau')}
        report['checks']['actor_max_interval_s']=max(b['t_s']-x['t_s'] for x,b in zip(policy,policy[1:]) if x['t_s']<hi and b['t_s']>=lo)
        if a.verify_actor:
            model=torch.jit.load(str(runner/'policy/imgo2/ppo/policy.pt'),map_location='cpu').eval()
            with torch.inference_mode():
                predicted=model(torch.tensor([[r[f'obs_{i}'] for i in range(45)] for r in policy],dtype=torch.float32))
            error=(predicted-torch.tensor([[r[f'raw_{i}'] for i in range(12)] for r in policy],dtype=torch.float32)).abs().max().item()
            assert error<2e-5
            report['checks']['all_captured_actor_pairs_CPU_max_error']=error
        for r in pp:
            w,x,y,z=[r['quat_'+k] for k in 'wxyz']
            gravity=[2*(w*y-x*z),-2*(w*x+y*z),-1+2*(x*x+y*y)]
            errors['gyro'].extend(r[f'obs_{i}']-r['gyro_'+axis]*.25 for i,axis in enumerate('xyz'))
            errors['gravity'].extend(r[f'obs_{3+i}']-v for i,v in enumerate(gravity))
            for i in range(12):
                errors['q'].append(r[f'obs_{9+i}']-(r[f'state_q_{i}']-defaults[i]))
                errors['dq_scaled'].append(r[f'obs_{21+i}']-r[f'state_dq_{i}']*.05*(gain if i in HIPS else 1.))
                expected=max(-23.5,min(23.5,25*(r[f'target_{i}']-r[f'state_q_{i}'])-.8*r[f'state_dq_{i}']))
                errors['nominal_tau'].append(r[f'diagnostic_tau_{i}']-expected)
        for prev,r in zip(policy,policy[1:]):
            if lo<=r['t_s']<hi:errors['last_action'].extend(r[f'obs_{33+i}']-prev[f'action_clipped_{i}'] for i in range(12))
        max_error={k:max(map(abs,v)) for k,v in errors.items()}
        assert max(max_error.values())<2e-4,max_error
        physical=rms([st.pstdev([r[f'state_dq_{i}'] for r in pp]) for i in HIPS])
        observed=rms([st.pstdev([r[f'obs_{21+i}']/.05 for r in pp]) for i in HIPS])
        assert abs(observed-physical*gain)<1e-5
        metrics=report['metrics'];metrics.update(physical_hip_dq_std_rad_s=physical,
            observed_hip_dq_std_rad_s=observed,gravity_y_input_std=st.pstdev(r['obs_4'] for r in pp),
            pitch_std_deg=st.pstdev(r['pitch_deg'] for r in imu if lo<=r['t_s']<hi))
        report['checks']['same_call_feedback_input_max_errors']=max_error
        report['checks']['history_uses_own_previous_clipped_output']=True
        report['checks']['only_isolated_source_observation_return_changed']=True
        report.update(gain=gain,status='complete');result['cases'][label]=report
        for r in policy:r['rr_dq']=r['state_dq_9'];r['rr_observed_dq']=r['obs_30']/.05;r['gravity_y']=r['obs_4']
        state=load(path/'state.csv');hosts=[r['host_s'] for r in odom]
        state=[r for r in state if hosts[0]<=r['host_s']<=hosts[-1]];map_host(state,odom)
        t0=json.loads((path/'collection.json').read_text())['ppo_pd_start_sim_s']
        for r in state:r['t_s']-=t0;r['rr_dq']=r['dq_09']
        report['sha256'][str(path/'state.csv')]=sha(path/'state.csv')
        data[label]=(imu,odom,policy,state)
        table.append({'case':label,'gain':gain,'physical_hip_dq_std_rad_s':physical,'observed_hip_dq_std_rad_s':observed,
            'gyro_x_rms_rad_s':metrics['gyro_x_rms_rad_s'],'roll_std_deg':metrics['roll_std_deg'],
            'roll_peak_to_peak_deg':metrics['roll_peak_to_peak_deg'],'gravity_y_input_std':metrics['gravity_y_input_std'],
            'gyro_peak_hz':metrics['gyro_spectrum']['peak_1_20_hz'],'gyro_band_4_10_rms':metrics['gyro_spectrum']['band_4_10_rms'],
            'body_vx_mps':metrics['motion']['body_vx_mean_mps'],'whole_ppo_max_tilt_deg':metrics['whole_ppo_max_tilt_deg']})
    guard=json.loads(Path('docs/deployment/real_gazebo_logic_review_2026-11-18.json').read_text())['sha256']
    assert all(sha(Path(p))==value for p,value in guard.items());result['production_guard']={'files':len(guard),'unchanged':True}
    a.output_prefix.parent.mkdir(parents=True,exist_ok=True)
    a.output_prefix.with_suffix('.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    with a.output_prefix.with_suffix('.csv').open('w') as f:
        writer=csv.DictWriter(f,list(table[0]));writer.writeheader();writer.writerows(table)
    print(json.dumps(table,indent=2,ensure_ascii=False))
    if a.plot:plot(a.output_prefix,result,data,lo,hi)

def plot(prefix,result,data,lo,hi):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font=subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK SC'],text=True).strip()
    font_manager.fontManager.addfont(font);plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name();plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(len(data),4,figsize=(19,3*len(data)),sharex=True,squeeze=False)
    for row,(label,(imu,odom,policy,state)) in enumerate(data.items()):
        gain=result['cases'][label]['gain'];pp=[r for r in policy if lo<=r['t_s']<hi];ii=[r for r in imu if lo<=r['t_s']<hi]
        ss=[r for r in state if lo<=r['t_s']<hi]
        for col,(rows,key,title) in enumerate([(ss,'rr_dq','右后髋速度（rad/s）'),(ii,'gyro_x','实际gyro X（rad/s）'),(pp,'gravity_y','输入gravity Y'),(ii,'roll_deg','实际roll（°）')]):
            ax=axes[row,col];ax.plot([r['t_s']-lo for r in rows],[r[key] for r in rows],lw=.8)
            if col==0:ax.plot([r['t_s']-lo for r in pp],[r['rr_observed_dq'] for r in pp],ls='--',lw=.8,label='策略所见速度');ax.set_ylabel(f'输入保留{gain*100:.0f}%')
            if row==0:ax.set_title(title)
            if row==len(data)-1:ax.set_xlabel('窗口内时间（s）；各组独立步态相位')
            ax.grid(alpha=.25)
    for col in range(4):
        low=min(ax.get_ylim()[0] for ax in axes[:,col]);high=max(ax.get_ylim()[1] for ax in axes[:,col])
        for ax in axes[:,col]:ax.set_ylim(low,high)
    axes[0,0].legend(fontsize=8);fig.suptitle('Gazebo完整自身反馈：IMU/q/dq/上一动作随本轮运动更新；同56d6、25/0.8、指令0.5m/s')
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(str(prefix)+'_motion.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8));keys=['physical_hip_dq_std_rad_s','gyro_x_rms_rad_s','roll_std_deg','body_vx_mean_mps']
    titles=['实际四髋速度波动（rad/s）','机身gyro X RMS（rad/s）','roll标准差（°）','实际前进速度（m/s）']
    xx=[result['cases'][k]['gain']*100 for k in data]
    for ax,key,title in zip(axes.flat,keys,titles):
        yy=[result['cases'][k]['metrics']['motion'][key] if key=='body_vx_mean_mps' else result['cases'][k]['metrics'][key] for k in data]
        ax.plot(xx,yy,'o-');ax.set_xticks([25,50,75,100]);ax.set_xlabel('策略髋速度输入保留比例（%）');ax.set_title(title);ax.grid(alpha=.25);ax.margins(y=.3)
        for x,y in zip(xx,yy):ax.annotate(f'{y:.3f}',(x,y),xytext=(0,7),textcoords='offset points',ha='center')
    fig.suptitle('自身反馈闭环结果；正常Gazebo基线已稳定，不能当作实机受载振荡复现')
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(str(prefix)+'_sweep.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for label,(imu,odom,policy,state) in data.items():
        gain=result['cases'][label]['gain']
        for ax,rows,key in [(axes[0],imu,'gyro_x'),(axes[1],state,'rr_dq')]:
            _,bins=spectrum(wave(rows,key,lo,hi));ax.plot([x for x,y in bins],[y for x,y in bins],label=f'{gain*100:.0f}%')
    for ax,title in zip(axes,['机身gyro X','右后髋实际速度']):ax.set_title(title);ax.set_xlim(1,15);ax.set_xlabel('频率（Hz）');ax.set_ylabel('单频正弦幅值（rad/s）');ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(str(prefix)+'_spectra.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output-prefix',type=Path,required=True);p.add_argument('--plot',action='store_true');p.add_argument('--verify-actor',action='store_true',help='Optional PyTorch verification of every captured input/raw-output pair');main(p.parse_args())
