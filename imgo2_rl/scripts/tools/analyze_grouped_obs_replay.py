#!/usr/bin/env python3
"""Audit grouped historical-obs counterfactuals; stdlib with optional plots."""
import argparse
import bisect
import csv
import json
import math
import statistics as st
import subprocess
from pathlib import Path

from audit_paired_obs_replay import load, attitude, summarize, wave, spectrum, sha, rms
from analyze_command_history import map_host
from analyze_ppo_input_attribution import describe

LABELS={'baseline':'原实机obs重算','body':'替换角速度＋重力','hip_state':'替换四髋q/dq',
        'hip_history':'替换四髋上一动作','combined':'以上三组一起替换','all_joint_state':'替换全部关节q/dq',
        'dq075':'髋速度保留75%','dq050':'髋速度保留50%','dq025':'髋速度保留25%',
        'body_q075':'仿真IMU＋髋q偏差75%','body_q050':'仿真IMU＋髋q偏差50%','body_q025':'仿真IMU＋髋q偏差25%'}

def dump_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)

def main(a):
    plan=json.loads(a.plan.read_text());source=Path(plan['source']);donor=Path(plan['donor'])
    lo,hi=plan['analysis_window'];normal_lo=plan['donor_start']+lo-plan['start'];normal_hi=normal_lo+hi-lo
    real=load(source/'robot.csv');normal=load(donor/'imu.csv');odom=load(donor/'odom.csv')
    for r in real:r['t_s']=r['t_mono_ms']*.001;attitude(r)
    for r in normal+odom:r['t_s']=r['sim_s'];attitude(r)
    reference_policy=load(donor/'policy_raw.csv');map_host(reference_policy,odom)
    normal_obs=[{'t':r['t_s'],'x':[r[f'obs_{i}'] for i in range(45)],'act':[r[f'action_clipped_{i}'] for i in range(12)]} for r in reference_policy]
    result={'plan':plan,'policy_sha256':sha(Path(plan['policy'])),
        'real':summarize(real,lo,hi),'normal':summarize(normal,normal_lo,normal_hi),
        'normal_motion':{'mean_body_vx_mps':st.mean(r['body_vx'] for r in odom if normal_lo<=r['t_s']<normal_hi)},
        'normal_input':describe(normal_obs,normal_lo,normal_hi)[0],'cases':{},
        'limits':['Historical mixed obs drive actor, simulated feedback never enters the next input; no self-consistent new controller.',
                  'One fixed time alignment, no gait phase optimization; broad substitutions can disturb inter-channel phase relations.',
                  'Real support/rope timing, base velocity and initial full physical/contact state are unknown.',
                  'Gravity is normalized and Z has tiny variation; a large ratio alone is not a large physical tilt.',
                  'Main window 2.8s gives 0.357Hz frequency spacing, not an identified natural frequency or causal delay.'],
        'sha256':{str(source/n):sha(source/n) for n in ('policy.csv','robot.csv','meta.json')}}
    if plan.get('experiment')=='hip_velocity_scaling':
        result['limits'][1]='Only four hip dq channels are scaled; original relative phase/timestamps retained, but q/dq and action-history consistency is altered. No donor mixing or new filter phase lag.'
    guard=json.loads(Path('docs/deployment/real_gazebo_logic_review_2026-11-18.json').read_text())['sha256']
    assert all(sha(Path(p))==value for p,value in guard.items())
    result['previous_production_audit_guard']={'files':len(guard),'unchanged':True}
    plots={};table=[];trace=[];initial=None
    for case in plan['cases']:
        name=case['group'];inf=Path(case['inference']);run=Path(case['run'])
        info=json.loads((inf/'inference_summary.json').read_text());meta=json.loads((run/'metadata.json').read_text())
        assert meta['completed'] and meta['gui_enabled'] and meta['production_files_unchanged']
        obs=load(inf/'obs_used.csv');changed=[i for i in range(45) if any(r[f'original_{i:02}']!=r[f'used_{i:02}'] for r in obs)]
        intervention=info.get('hip_velocity_scaling') or info['group_replacement']
        expected=(intervention or {}).get('obs_indices',[])
        position=info.get('hip_position_scaling')
        if position:expected=sorted(set(expected+position['obs_indices']))
        assert changed==expected
        if position:
            gain=position['gain'];assert gain==case['hip_q_gain']
            assert all(r[f'used_{i:02}']==r[f'original_{i:02}']*gain for r in obs for i in position['obs_indices'])
        if info.get('hip_velocity_scaling'):
            gain=info['hip_velocity_scaling']['gain'];assert gain==case['hip_dq_gain']
            assert all(r[f'used_{i:02}']==r[f'original_{i:02}']*gain for r in obs for i in expected)
        targets=load(run/'targets.csv');published=load(run/'published.csv')
        assert len(targets)==len(published)==info['outer_commands']
        assert [int(r['index']) for r in published]==list(range(len(targets)))
        assert all(r[f'kp_{i:02}']==25 and r[f'kd_{i:02}']==.8 and r[f'dq_{i:02}']==0 and r[f'tau_{i:02}']==0 for r in targets for i in range(12))
        assert sha(run/'targets.csv')==sha(inf/'targets.csv')
        assert info['safety_position_clips']==0 and info['batch_vs_single_max_abs_diff']<2e-5
        if initial is None:initial=sha(run/'initial.csv')
        assert sha(run/'initial.csv')==initial
        rows=load(run/'replay.csv')
        for r in rows:r['t_s']+=plan['start'];attitude(r)
        metrics=summarize(rows,lo,hi);full=[r for r in rows if plan['start']<=r['t_s']<=plan['start']+plan['duration']]
        first_fall=next((r['t_s'] for r in full if r['tilt_deg']>60),None)
        metrics.update(full_max_tilt_deg=max(r['tilt_deg'] for r in full),full_min_height_m=min(r['z'] for r in full),first_tilt_over_60_s=first_fall)
        metrics['normal_motion_comparison_valid']=first_fall is None and metrics['full_max_tilt_deg']<20 and metrics['min_height_m']>.2
        for r in rows:
            target=targets[max(0,min(len(targets)-1,int(r['target_index'])))]
            r['common_hip_q_des_deg']=st.mean(target[f'q_{i:02}'] for i in (0,3,6,9))*180/math.pi
        selected=[r for r in rows if lo<=r['t_s']<hi]
        metrics['physical_hip_q_std_deg']=180/math.pi*rms([st.pstdev(r[f'q_{i:02}'] for r in selected) for i in (0,3,6,9)])
        inference=load(inf/'inference.csv');assert len(obs)==len(inference)
        input_rows=[{'t':r['source_s'],'x':[r[f'used_{i:02}'] for i in range(45)],'act':[y[f'action_{i:02}'] for i in range(12)]} for r,y in zip(obs,inference)]
        result['cases'][name]={'metrics':metrics,'input':describe(input_rows,lo,hi)[0],'intervention':intervention,
            'hip_position_scaling':position,
            'checks':{'changed_obs_indices':changed,'unchanged_other_channels':True,'same_initial_state':True,
                      'targets':len(targets),'published':len(published),'publish_lag_max_s':max(r['t_s']-r['scheduled_s'] for r in published),
                      'batch_vs_single_max_diff':info['batch_vs_single_max_abs_diff'],'production_files_unchanged':True},
            'sha256':{str(p):sha(p) for p in (run/'replay.csv',run/'targets.csv',run/'initial.csv',inf/'obs_used.csv')}}
        if name=='baseline':
            assert info['full_log_pair_validation']['values_over_2e_5']==0
            result['full_log_pair_validation']=info['full_log_pair_validation']
            rt=[r['t_s'] for r in real]
            target_error=[]
            for row in targets:
                original=real[max(0,bisect.bisect_right(rt,plan['start']+row['t_s']+1e-9)-1)]
                target_error.extend(row[f'q_{i:02}']-original[f'q_des_{i:02}'] for i in range(12))
            assert max(map(abs,target_error))<1e-5
            result['baseline_same_tick_target_max_error_rad']=max(map(abs,target_error))
            old=load(Path('imgo2_deploy/logs/nov17_obs_forward_gazebo_20261117_a/replay.csv'))
            for r in old:r['t_s']+=plan['start'];attitude(r)
            result['previous_replay']=summarize(old,lo,hi)
        plots[name]=rows
        table.append({'case':name,'label':LABELS[name],'gyro_rms_rad_s':metrics['gyro_x_rms_rad_s'],
            'roll_std_deg':metrics['roll_std_deg'],'roll_p95_p05_half_deg':metrics['roll_half_p95_p05_deg'],
            'roll_peak_to_peak_deg':metrics['roll_peak_to_peak_deg'],'gyro_peak_hz':metrics['gyro_spectrum']['peak_1_20_hz'],
            'gyro_4_10_rms_rad_s':metrics['gyro_spectrum']['band_4_10_rms'],
            'body_vx_mps':metrics['mean_body_vx_mps'],'first_tilt_over_60_s':first_fall,
            'full_max_tilt_deg':metrics['full_max_tilt_deg'],'min_height_m':metrics['full_min_height_m'],
            'valid_motion_comparison':metrics['normal_motion_comparison_valid']})
        waves={key:wave(rows,key,lo,hi) for key in ('gyro_x','roll_deg','common_hip_q_des_deg')}
        for k in range(len(waves['gyro_x'])):trace.append({'case':name,'source_s':lo+k*.01,**{key:val[k] for key,val in waves.items()}})
    comparison=[]
    for r,n in zip(result['cases']['baseline']['input']['channels'],result['normal_input']['channels']):
        scale=4 if r['index']<3 else 20 if 21<=r['index']<33 else 1
        comparison.append({'index':r['index'],'label':r['label'],'real_mean_scaled':r['mean'],'normal_mean_scaled':n['mean'],
            'real_std_physical':r['std']*scale,'normal_std_physical':n['std']*scale,
            'std_ratio':r['std']/n['std'] if n['std']>1e-9 else None,
            'real_band_4_10_rms_physical':r['band_4_10_rms']*scale,'normal_band_4_10_rms_physical':n['band_4_10_rms']*scale,
            'band_ratio':r['band_4_10_rms']/n['band_4_10_rms'] if n['band_4_10_rms']>1e-9 else None,
            'real_peak_hz':r['peak_1_20_hz'],'normal_peak_hz':n['peak_1_20_hz']})
    result['input_comparison']=comparison
    prefix=a.output_prefix;prefix.parent.mkdir(parents=True,exist_ok=True)
    prefix.with_suffix('.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    dump_csv(prefix.with_suffix('.csv'),table);dump_csv(Path(str(prefix)+'_inputs.csv'),comparison);dump_csv(Path(str(prefix)+'_traces.csv'),trace)
    print(json.dumps(table,ensure_ascii=False,indent=2))
    if a.plot:plot(prefix,result,plots,real,normal,lo,hi,normal_lo,normal_hi)

def plot(prefix,result,plots,real,normal,lo,hi,nlo,nhi):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font=subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK SC'],text=True).strip()
    font_manager.fontManager.addfont(font);plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
    plt.rcParams['axes.unicode_minus']=False
    scaling=result['plan'].get('experiment') in ('hip_velocity_scaling','body_donor_hip_position_scaling')
    position_scaling=result['plan'].get('experiment')=='body_donor_hip_position_scaling'
    limits=((-17,17),(-2.7,2.7),(-5.5,5.5))
    if scaling:
        bounds=[]
        for key in ('roll_deg','gyro_x','common_hip_q_des_deg'):
            maximum=max(abs(r[key]) for rows in plots.values() for r in rows if lo<=r['t_s']<hi)
            maximum=max(.1,maximum*1.1);bounds.append((-maximum,maximum))
        limits=bounds
    fig,axes=plt.subplots(len(plots),3,figsize=(17,3*len(plots)),sharex=True)
    for row,(name,rows) in enumerate(plots.items()):
        m=result['cases'][name]['metrics'];selected=[r for r in rows if lo<=r['t_s']<hi]
        for col,key in enumerate(('roll_deg','gyro_x','common_hip_q_des_deg')):
            ax=axes[row,col];ax.plot([r['t_s']-lo for r in selected],[r[key] for r in selected],lw=.9,label=LABELS[name]);ax.grid(alpha=.25)
            if key!='common_hip_q_des_deg':
                nn=[r for r in normal if nlo<=r['t_s']<nhi]
                ax.plot([r['t_s']-nlo for r in nn],[r[key] for r in nn],color='green',lw=.7,alpha=.7,label='正常仿真（独立相位）')
            if row==0:ax.set_title(('回放机身横滚角（°）','回放机身横滚角速度（rad/s）','新推理的四髋共同目标角（°）')[col])
            ax.set_ylim(limits[col])
            if col==0:ax.set_ylabel(LABELS[name]+('\n倾角曾>60°' if m['first_tilt_over_60_s'] else ''))
            if row==len(plots)-1:ax.set_xlabel('窗口内时间（s）；无相位拟合')
    kind='仿真IMU＋四髋位置偏差降幅' if position_scaling else '四髋速度定比例降幅' if scaling else '分组替换'
    fig.suptitle('源76.8–79.6s；'+kind+'历史obs开环回放；蓝：各组回放，绿：正常仿真（独立相位）',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(str(prefix)+'_motion.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(15,8),sharex=True)
    for name,rows in plots.items():
        rows=[r for r in rows if 70<=r['t_s']<=80]
        tt=[r['t_s']-70 for r in rows];axes[0].plot(tt,[r['roll_deg'] for r in rows],lw=.8,label=LABELS[name]);axes[1].plot(tt,[r['z'] for r in rows],lw=.8)
    axes[0].set_ylabel('横滚角（°）');axes[1].set_ylabel('base高度（m）');axes[1].set_xlabel('回放开始后（s）')
    axes[0].legend(ncol=3);axes[0].set_title('完整10秒：横滚角与base高度，灰色为主统计窗' if scaling else '完整10秒：检查倒地与运动失效，不能只比较较小的角速度')
    for ax in axes:ax.grid(alpha=.25);ax.axvspan(lo-70,hi-70,color='grey',alpha=.12)
    fig.tight_layout();fig.savefig(str(prefix)+'_full.png',dpi=150);plt.close(fig)
    if scaling and not position_scaling:
        fig,axes=plt.subplots(1,3,figsize=(14,4.5))
        gains=[c['hip_dq_gain']*100 for c in result['plan']['cases']]
        for ax,key,title,reference in zip(axes,('gyro_x_rms_rad_s','roll_std_deg','mean_body_vx_mps'),
                ('横滚角速度RMS（rad/s）','横滚角度标准差（°）','实际前进速度（m/s）'),
                (result['normal']['gyro_x_rms_rad_s'],result['normal']['roll_std_deg'],result['normal_motion']['mean_body_vx_mps'])):
            values=[result['cases'][c['group']]['metrics'][key] for c in result['plan']['cases']]
            ax.plot(gains,values,'o-',label='实机历史obs降幅回放');ax.axhline(reference,color='green',ls='--',label='正常自身反馈仿真参考')
            for x,y in zip(gains,values):ax.annotate(f'{y:.3f}',(x,y),xytext=(0,7),textcoords='offset points',ha='center')
            ax.set_xticks([25,50,75,100]);ax.set_xlabel('四髋速度观测保留比例（%）');ax.set_title(title);ax.grid(alpha=.25);ax.margins(y=.25)
        axes[0].legend(fontsize=8,loc='center right');fig.suptitle('只缩小4维速度输入，保留其他41维与PD；主窗76.8–79.6s')
        fig.tight_layout(rect=(0,0,1,.94));fig.savefig(str(prefix)+'_sweep.png',dpi=150);plt.close(fig)
    if position_scaling:
        fig,axes=plt.subplots(2,2,figsize=(12,8));names=list(plots);xx=list(range(len(names)))
        titles=['机身gyro X RMS（rad/s）','roll标准差（°）','实际前进速度（m/s）','策略输入与物理髋角波动（°）']
        for ax,key,title in zip(axes.flat,('gyro_x_rms_rad_s','roll_std_deg','mean_body_vx_mps','physical_hip_q_std_deg'),titles):
            yy=[result['cases'][n]['metrics'][key] for n in names];ax.plot(xx,yy,'o-',label='物理回放反馈')
            if key=='physical_hip_q_std_deg':
                yy_input=[result['cases'][n]['input']['groups_scaled_obs_units']['q_hip']['std']*180/math.pi for n in names]
                ax.plot(xx,yy_input,'s--',label='策略所见髋位置');ax.legend(fontsize=8)
            for x,y in zip(xx,yy):ax.annotate(f'{y:.3f}',(x,y),xytext=(0,7),textcoords='offset points',ha='center')
            ax.set_xticks(xx,['原实机obs','只换IMU','IMU＋q75%','IMU＋q50%','IMU＋q25%'],rotation=15)
            ax.set_title(title);ax.grid(alpha=.25);ax.margins(y=.3)
        fig.suptitle('固定正常仿真IMU＋实机历史输入；仅改四髋位置偏差，不是自身反馈闭环')
        fig.tight_layout(rect=(0,0,1,.95));fig.savefig(str(prefix)+'_sweep.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(15,4.5))
        for case in result['plan']['cases']:
            name=case['group'];observed=load(Path(case['inference'])/'obs_used.csv')
            observed=[r for r in observed if lo<=r['source_s']<hi];tt=[r['source_s']-lo for r in observed]
            if name in ('baseline','body'):
                for ax,key,scale in ((axes[0],'used_00',4.),(axes[1],'used_04',1.)):
                    ax.plot(tt,[r[key]*scale for r in observed],lw=.9,label='原实机输入' if name=='baseline' else '替换后的仿真输入')
            axes[2].plot(tt,[r['used_18']*180/math.pi for r in observed],lw=.9,label=LABELS[name])
        for ax,title in zip(axes,('策略gyro X输入（rad/s）','策略gravity Y输入','策略右后髋角度偏差输入（°）')):
            ax.set_title(title);ax.set_xlabel('源窗口内时间（s），固定偏移，无步态相位拟合');ax.grid(alpha=.25);ax.legend(fontsize=8)
        fig.suptitle('实际送入Actor的数值：IMU来自正常仿真历史，髋q降幅，其余35维保留实机历史')
        fig.tight_layout(rect=(0,0,1,.94));fig.savefig(str(prefix)+'_used_inputs.png',dpi=150);plt.close(fig)
    comp=result['input_comparison'];fig,axes=plt.subplots(1,2,figsize=(14,14),sharey=True)
    for ax,key,title in zip(axes,('std_ratio','band_ratio'),('全频波动标准差：实机／正常仿真','4–10Hz波动RMS：实机／正常仿真')):
        values=[min(12,r[key]) if r[key] is not None else 0 for r in comp]
        ax.barh(range(45),values,color=['#d55e00' if v>2 else '#0072b2' for v in values]);ax.axvline(1,color='black',ls='--');ax.set_title(title);ax.set_xlim(0,12);ax.grid(axis='x',alpha=.25)
    axes[0].set_yticks(range(45),[r['label'] for r in comp]);axes[0].invert_yaxis()
    fig.suptitle('45维输入均检查；指令恒定时倍率留空；gravity Z绝对值很小，不能仅看倍率',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(str(prefix)+'_inputs.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output-prefix',type=Path,required=True);p.add_argument('--plot',action='store_true');main(p.parse_args())
