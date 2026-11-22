#!/usr/bin/env python3
"""Summarize saved continuation benchmarks and paired own-feedback Gazebo runs.

Stdlib numerical analysis; optional matplotlib. Does not run hardware or training.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics as st
from audit_paired_obs_replay import rms, spectrum, wave
from compare_policy1_9299 import simulate_summary
from read_tfevents import read_scalars

ROOT = Path(__file__).resolve().parents[3]


def main(a):
    comparison = json.loads(a.mujoco.read_text())
    pilot = json.loads(a.pilot.read_text()) if a.pilot else None
    result = {'mujoco': comparison, 'pilot10000': pilot, 'forward_detail': {}, 'gazebo': {},
              'limitations': [a.run_note,
                              'No hardware, rough-terrain traversal or multiple-seed validation.',
                              'Multiple reward weights changed together; total reward has a changed objective and is not an independent quality score.',
                              'Actuator bounded response tests are not implemented or run by this analysis.']}
    streams = {}
    for case in comparison['metadata']['cases']:
        if case['case'] != 'forward05': continue
        path = Path(case['csv'])
        with path.open() as f: rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
        sel = [r for r in rows if 4 <= r['t_s'] < 20]
        for r in rows:
            r['gyro_x'] = r['gyro_x_rads']
            r['roll_deg'] = r['roll_rad']*180/3.141592653589793
        name = case['policy']; streams[name] = rows
        result['forward_detail'][name] = {
            'gyro_x_rms_rad_s': rms([r['gyro_x'] for r in sel]),
            'gyro_x_spectrum_4_10s': spectrum(wave(rows, 'gyro_x', 4, 10))[0],
            'shank_dq_rms_rad_s': rms([r[f'dq_{leg}_shank_joint'] for r in sel for leg in ('FL','FR','RL','RR')]),
            'shank_tau_rms_nm': rms([r[f'tau_{leg}_shank_joint'] for r in sel for leg in ('FL','FR','RL','RR')])}
    gaz_streams = {}
    gazebo_cases = [(a.candidate_label, a.gazebo_new), (a.baseline_label, a.gazebo_parent)]
    if a.gazebo_extra:
        gazebo_cases.append((a.extra_label, a.gazebo_extra))
    if len({name for name, _ in gazebo_cases}) != len(gazebo_cases):
        raise ValueError('Gazebo labels must be unique')
    for name, path in gazebo_cases:
        result['gazebo'][name], gaz_streams[name] = simulate_summary(
            path, start_offset=a.gazebo_start_offset, duration=a.gazebo_duration)
    files = [d['isolated_policy_tree_sha256'] for d in result['gazebo'].values()]
    assert len({d['ppo_config'] for d in files}) == len({d['base_config'] for d in files}) == 1
    a.output.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
    if a.plot:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        policies = a.policies.split(',') if a.policies else ['original9299.pt','parent9299.pt',a.candidate_policy,'policy1.pt']
        labels = a.labels.split(',') if a.labels else ['old9299 -.20',a.baseline_label,a.candidate_label,'policy_1']
        if len(policies) != len(labels): raise ValueError('Policy/label counts differ')
        lookup = {r['policy']: r for r in comparison['results'] if r['case']=='forward05'}
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
        for ax, key, factor, title in zip(axes.flat,
                ['vx_mean','stride_frequency_hz','swing_clearance_peak_mean_m','gyro_xy_rms_rads','roll_std_deg','height_mean_m'],
                [1,1,100,1,1,100],
                ['Actual forward speed (m/s)','Stride frequency (Hz)','Swing clearance (cm)','Gyro XY RMS (rad/s)','Roll standard deviation (deg)','Mean base height (cm)']):
            ax.bar(range(len(policies)), [lookup[p][key]*factor for p in policies], color=[f'C{i}' for i in range(len(policies))])
            ax.set(xticks=range(len(policies)), xticklabels=labels, title=title)
            ax.tick_params(axis='x',labelrotation=20); ax.grid(axis='y',alpha=.25)
        fig.suptitle('MuJoCo benchmarks: command +0.5m/s; PD25/0.8; t=4..20s; run provenance in JSON')
        fig.savefig(a.output.with_name(a.output.name+'_overview.png'), dpi=150); plt.close(fig)
        if a.plot_feet:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
            legs = ('FL', 'FR', 'RL', 'RR'); width = .8/len(policies)
            for index, (policy, label) in enumerate(zip(policies, labels)):
                positions = [i-.4+width*(index+.5) for i in range(4)]
                for ax, key, factor, title in [(axes[0], 'swing_clearance_peak_mean_m', 100, 'Per-foot swing clearance (cm)'),
                                               (axes[1], 'freq_hz', 1, 'Per-foot stride frequency (Hz)')]:
                    ax.bar(positions, [lookup[policy]['feet'][leg][key]*factor for leg in legs], width=width,
                           label=label, color=f'C{index}')
                    ax.set(xticks=range(4), xticklabels=legs, title=title); ax.grid(axis='y', alpha=.25)
            axes[0].legend(fontsize=8)
            fig.suptitle('MuJoCo: +0.5m/s, PD25/0.8; complete swings in t=4..20s')
            fig.savefig(a.output.with_name(a.output.name+'_feet.png'), dpi=150); plt.close(fig)
        fig, axes = plt.subplots(3, 2, figsize=(13, 9), constrained_layout=True)
        time_policies = a.time_policies.split(',') if a.time_policies else [a.time_baseline_policy,a.candidate_policy,'policy1.pt']
        for name in time_policies:
            rows = streams[name]; sel = [r for r in rows if 4<=r['t_s']<6.5]
            for ax, key, title, units in [(axes[0,0],'gyro_x','Roll rate','rad/s'),
                    (axes[1,0],'roll_deg','Roll angle','deg'),
                    (axes[2,0],'FR_clearance_m','FR collision-bottom clearance','m'),
                    (axes[1,1],'qdes_FR_shank_joint','FR shank target','rad'),
                    (axes[2,1],'tau_FR_shank_joint','FR applied shank torque','Nm')]:
                ax.plot([r['t_s']-4 for r in sel],[r[key] for r in sel],label=name,lw=1)
                ax.set(title=title,ylabel=units,xlabel='Within-window time (s); independent phase')
            _, sp = spectrum(wave(rows,'gyro_x',4,10))
            axes[0,1].plot([p[0] for p in sp],[p[1] for p in sp],label=name)
        axes[0,1].set(title='Roll-rate spectrum',xlabel='Hz',ylabel='Hann amplitude (rad/s)',xlim=(1,15))
        for ax in axes.flat: ax.grid(alpha=.25); ax.legend(fontsize=8)
        fig.savefig(a.output.with_name(a.output.name+'_time.png'), dpi=150); plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        for name, (_, imu, _, lo, hi) in gaz_streams.items():
            sel=[r for r in imu if lo<=r['t_s']<lo+2.5]
            for ax,key in [(axes[0,0],'gyro_x'),(axes[1,0],'roll_deg')]:
                ax.plot([r['t_s']-lo for r in sel],[r[key] for r in sel],label=name)
            _,sp=spectrum(wave(imu,'gyro_x',lo,hi))
            axes[0,1].plot([p[0] for p in sp],[p[1] for p in sp],label=name)
        axes[0,0].set(title='Gazebo roll rate',ylabel='rad/s',xlabel='Independent window (s)')
        axes[1,0].set(title='Gazebo roll',ylabel='deg',xlabel='Independent window (s)')
        axes[0,1].set(title='Gazebo roll-rate spectrum',ylabel='Hann amplitude (rad/s)',xlabel='Hz',xlim=(1,15))
        axes[1,1].axis('off')
        extra_note = f'{a.extra_label} newly run\n' if a.gazebo_extra else ''
        axes[1,1].text(.02,.95,f'PD25/0.8; command +0.5m/s\nOwn simulated feedback\n{a.candidate_label} newly run\n{a.baseline_label} reused\n{extra_note}Statistics use entry+{a.gazebo_start_offset:g}..{a.gazebo_start_offset+a.gazebo_duration:g}s\nNo hardware or training run',va='top')
        for ax in axes.flat:
            if ax.axison:ax.grid(alpha=.25);ax.legend(fontsize=8)
        fig.savefig(a.output.with_name(a.output.name+'_gazebo.png'),dpi=150);plt.close(fig)
        fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
        tags=['Train/mean_reward','Metrics/base_velocity/error_vel_xy','Metrics/base_velocity/error_vel_yaw','Policy/mean_noise_std','Curriculum/terrain_levels','Episode_Reward/feet_height_body']
        training_runs = [(a.baseline_label,a.parent_run),(a.candidate_label,a.new_run)]
        if a.extra_run:
            training_runs.append((a.extra_label,a.extra_run))
        for name,path in training_runs:
            ss=read_scalars(str(next(path.glob('events*'))));lo=min(i for i,v in ss[tags[0]]);hi=max(i for i,v in ss[tags[0]])
            for ax,tag in zip(axes.flat,tags):
                chunks=[[p for p in ss[tag] if k<=p[0]<k+100] for k in range(lo,hi+1,100)]
                ax.plot([st.mean(p[0] for p in c) for c in chunks if c],[st.mean(p[1] for p in c) for c in chunks if c],label=name)
                ax.axvline(a.branch_marker,color='gray',ls=':',lw=1);ax.set(title=tag,xlabel='Training iteration; 100-iteration means');ax.grid(alpha=.25);ax.legend(fontsize=8)
        fig.savefig(a.output.with_name(a.output.name+'_training.png'),dpi=150);plt.close(fig)
    print(json.dumps({k:{'gyro_x':v['imu']['gyro_x_rms_rad_s'],'roll_std':v['imu']['roll_std_deg'],'vx':v['body']['mean_body_vx_mps']} for k,v in result['gazebo'].items()},indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('mujoco','gazebo-new','gazebo-parent','parent-run','new-run','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--pilot',type=Path)
    p.add_argument('--candidate-policy',default='continued10500.pt')
    p.add_argument('--time-baseline-policy',default='parent9299.pt')
    p.add_argument('--time-policies',help='Comma-separated policies for time/spectrum traces')
    p.add_argument('--gazebo-extra',type=Path)
    p.add_argument('--extra-run',type=Path)
    p.add_argument('--extra-label',default='extra_policy')
    p.add_argument('--plot-feet',action='store_true')
    p.add_argument('--gazebo-start-offset',type=float,default=5.0)
    p.add_argument('--gazebo-duration',type=float,default=6.0)
    p.add_argument('--candidate-label',default='continued10500')
    p.add_argument('--baseline-label',default='parent9299')
    p.add_argument('--policies',help='Comma-separated policy file names in the input result JSON')
    p.add_argument('--labels',help='Comma-separated labels, matching policies')
    p.add_argument('--run-note',default='All MuJoCo36+3 cases newly run at25/.8; Gazebo10500 new, parent9299 reused from previous identified run; full4000-iteration training not complete.')
    p.add_argument('--branch-marker',type=int,default=9299)
    p.add_argument('--plot',action='store_true');main(p.parse_args())
