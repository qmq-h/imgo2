#!/usr/bin/env python3
"""Audit selected exact-obs replays; stdlib statistics and optional plots.

The plan supplies explicit source windows chosen before observing replay results.
Simulation is forced by historical actor outputs, not a new policy feedback loop.
"""
import argparse
import bisect
import csv
import json
import math
from pathlib import Path
import statistics as st

from audit_paired_obs_replay import attitude, load, quantile, rms, sha, spectrum, summarize, wave

GROUPS = {'hip': (0, 3, 6, 9), 'thigh': (1, 4, 7, 10), 'shank': (2, 5, 8, 11)}


def past(rows, times, t):
    return rows[max(0, bisect.bisect_right(times, t + 1e-8) - 1)]


def interp(rows, times, t, key):
    i = min(len(times) - 1, max(1, bisect.bisect_right(times, t)))
    a, b = rows[i - 1], rows[i]
    dt = times[i] - times[i - 1]
    assert 0 < dt <= .05, 'joint trace data gap'
    f = (t - times[i - 1]) / dt
    return a[key] + f * (b[key] - a[key])


def main(a):
    plan = json.loads(a.plan.read_text())
    source = Path(plan['source'])
    real = load(source / 'robot.csv')
    policy = load(source / 'policy.csv')
    cfg = json.loads((source / 'meta.json').read_text())['config']
    assert cfg['policy_to_hardware'] == list(range(12))
    assert sha(Path(plan['policy'])) == plan['candidate_origin']['sha256']
    for r in real:
        r['t_s'] = r['t_mono_ms'] * .001
        attitude(r)
    for r in policy:
        r['t_s'] = r['t_mono_ms'] * .001
    rt = [r['t_s'] for r in real]
    pt = [r['t_s'] for r in policy]
    assert all(b > x for x, b in zip(rt, rt[1:]))
    assert all(b > x for x, b in zip(pt, pt[1:]))
    assert all(math.isfinite(v) for rows in (real, policy) for r in rows for v in r.values())
    ppo = [r for r in real if r['mode'] == 3]
    by_tick = {r['tick']: r for r in real}
    assert len(by_tick) == len(real)
    def timing(values):
        return {'min':min(values),'median':st.median(values),
                'p95':quantile(values,.95),'max':max(values)}
    result = {'source': str(source), 'plan': plan,
              'policy_sha256': sha(Path(plan['policy'])), 'cases': {},
              'source_quality': {'robot_rows': len(real), 'policy_rows': len(policy),
                  'ppo_rows': len(ppo), 'ppo_clamped_rows': sum(r['clamped'] != 0 for r in ppo),
                  'ppo_invalid_motor_rows': sum(r['valid_mask'] != 4095 for r in ppo),
                  'ppo_motor_error_rows': sum(r['err_mask'] != 0 for r in ppo),
                  'policy_capture_to_result_ms_p95': quantile([r['result_generated_ms'] - r['input_captured_ms'] for r in policy], .95),
                  'policy_capture_gap_ms_max': max(b['input_captured_ms']-x['input_captured_ms'] for x,b in zip(policy,policy[1:])),
                  'capture_to_result_ms':timing([r['result_generated_ms']-r['input_captured_ms'] for r in policy]),
                  'capture_to_same_tick_robot_row_ms':timing([by_tick[r['tick']]['t_mono_ms']-r['input_captured_ms'] for r in policy]),
                  'negative_logged_result_age_rows':sum(r['result_age_ms']<0 for r in policy),
                  'timing_limit':'Host timestamp differences, not measured wire arrival or physical actuator delay; target replay already includes consumed-tick timing.'},
              'limitations': [
                  'Runtime weight hash was not saved; full-log numerical agreement identifies a matching candidate, not original file bytes.',
                  'Original exact observations and previous actions are retained. Gazebo feedback never replaces the next historical observation.',
                  'Real base translation/height/linear velocity and contact state are absent. Initial height is an FK nonpenetration assumption, base velocities zero.',
                  'User confirms rope/handling occurred, but intervention times are unknown; selected real windows cannot yet be labelled assisted or autonomous. Replay has no rope.',
                  'Same historical excitation does not identify the first cause of real feedback-loop oscillation.',
                  'Forward command includes small lateral motion; windows are not all pure straight walking.',
                  'Joint PD calculations use contemporaneous command/feedback snapshots, not next-feedback residuals or calibrated physical load.',
                  'Frequency resolution differs with window duration. No phase fitting or extra communication delay is introduced.'],
              'sha256': {str(source/n): sha(source/n) for n in ('robot.csv', 'policy.csv', 'meta.json', 'events.log')}}
    plot_data = {}
    trace_rows = []
    main_rows = []
    for c in plan['cases']:
        label, start, duration = c['label'], c['start'], c['duration']
        lo, hi = c['analysis_window']
        run, inference = Path(c['run']), Path(c['inference'])
        meta = json.loads((run/'metadata.json').read_text())
        inf = json.loads((inference/'inference_summary.json').read_text())
        assert meta['completed'] and meta['production_files_unchanged']
        assert meta['source_start_s'] == start and meta['duration_s'] == duration
        assert inf['whole_window_action_max_abs_error'] < 2e-5
        assert inf['last_action_input'].startswith('recorded')
        assert not any(inf[k] for k in ('gravity_replacement', 'joint_replacement'))
        assert all(inf['filters'][k] == 0 for k in ('dq_cutoff_hz','gravity_roll_cutoff_hz','gyro_cutoff_hz'))
        if 'full_log_pair_validation' in inf:
            assert inf['full_log_pair_validation']['values_over_2e_5'] == 0
            result['full_log_pair_validation'] = inf['full_log_pair_validation']
        obs = load(inference/'obs_used.csv')
        assert all(r[f'original_{i:02}'] == r[f'used_{i:02}'] for r in obs for i in range(45))
        targets, sent = load(run/'targets.csv'), load(run/'published.csv')
        assert len(targets) == len(sent) == inf['outer_commands']
        assert [int(r['index']) for r in sent] == list(range(len(targets)))
        target_error = []
        for row in targets:
            r = past(real, rt, start + row['t_s'])
            assert r['mode'] == 3
            for i in range(12):
                assert row[f'kp_{i:02}'] == r[f'kp_{i:02}'] == 25
                assert row[f'kd_{i:02}'] == r[f'kd_{i:02}'] == .8
                assert row[f'dq_{i:02}'] == r[f'dq_des_{i:02}'] == 0
                assert row[f'tau_{i:02}'] == r[f'tau_ff_{i:02}'] == 0
                target_error.append(row[f'q_{i:02}'] - r[f'q_des_{i:02}'])
        assert max(map(abs,target_error)) < 1e-5, 'tick-to-target mismatch'
        sim = load(run/'replay.csv')
        for r in sim:
            r['t_s'] += start
            attitude(r)
        assert all(math.isfinite(v) for r in sim for v in r.values())
        rr = summarize(real,lo,hi,policy)
        ss = summarize(sim,lo,hi,policy)
        full = [r for r in sim if start <= r['t_s'] <= start+duration]
        first_fall = next((r['t_s'] for r in full if r['tilt_deg'] > 60),None)
        rr['normal_window_valid'] = rr['max_tilt_deg'] < 20
        ss['normal_window_valid'] = ss['upright'] and ss['max_tilt_deg'] < 20
        ss['whole_replay_first_tilt_over_60_s'] = first_fall
        ss['whole_replay_max_tilt_deg'] = max(r['tilt_deg'] for r in full)
        ss['whole_replay_min_height_m'] = min(r['z'] for r in full)
        smt = [r['t_s'] for r in sim]
        target_times = [start+r['t_s'] for r in targets]
        traces = []
        for k in range(round((hi-lo)*100)):
            t = lo+k*.01
            r = past(real,rt,t)
            x = past(targets,target_times,t)
            out = {'case':label,'source_s':t}
            for i in range(12):
                for key in ('q','dq','tau'):
                    out[f'real_{key}_{i:02}'] = interp(real,rt,t,f'{key}_{i:02}')
                    out[f'replay_{key}_{i:02}'] = interp(sim,smt,t,f'controller_{key}_{i:02}')
                out[f'target_{i:02}'] = x[f'q_{i:02}']
                for kind in ('real','replay'):
                    out[f'{kind}_nominal_{i:02}'] = 25*(out[f'target_{i:02}']-out[f'{kind}_q_{i:02}'])-.8*out[f'{kind}_dq_{i:02}']
            traces.append(out)
        trace_rows.extend(traces)
        groups = {}
        for kind,ids in GROUPS.items():
            groups[kind] = {}
            for name in ('real','replay'):
                groups[kind][name] = {
                    'q_error_rms_deg':180/math.pi*rms([r[f'target_{i:02}']-r[f'{name}_q_{i:02}'] for r in traces for i in ids]),
                    'dq_rms_rad_s':rms([r[f'{name}_dq_{i:02}'] for r in traces for i in ids]),
                    'feedback_tau_rms_Nm':rms([r[f'{name}_tau_{i:02}'] for r in traces for i in ids]),
                    'nominal_pd_rms_Nm':rms([r[f'{name}_nominal_{i:02}'] for r in traces for i in ids])}
        result['cases'][label] = {'real':rr,'replay':ss,'groups':groups,
            'checks':{'unmodified_45_obs':True,'actual_PD':[25,.8],
                'same_tick_target_max_abs_error_rad':max(map(abs,target_error)),
                'published_targets':len(sent),'max_publish_lag_s':max(r['t_s']-r['scheduled_s'] for r in sent)},
            'sha256':{str(p/n):sha(p/n) for p,nn in [(run,('metadata.json','targets.csv','initial.csv','replay.csv','published.csv')),
                      (inference,('inference_summary.json','inference.csv','obs_used.csv'))] for n in nn}}
        plot_data[label] = (c,sim,traces)
        for name,m in [('real',rr),('replay',ss)]:
            main_rows.append({'case':label,'condition':name,'lo_s':lo,'hi_s':hi,
                'gyro_x_rms_rad_s':m['gyro_x_rms_rad_s'],'roll_std_deg':m['roll_std_deg'],
                'roll_peak_to_peak_deg':m['roll_peak_to_peak_deg'],
                'gyro_peak_hz':m['gyro_spectrum']['peak_1_20_hz'],
                'gyro_band_4_10_rms':m['gyro_spectrum']['band_4_10_rms'],
                'normal_window_valid':m['normal_window_valid']})
    if a.normal:
        meta = json.loads((a.normal/'metadata.json').read_text())
        collection = json.loads((a.normal/'collection.json').read_text())
        runner = Path(meta['rl_sim_binary'])
        assert sha(runner.parent/'policy/imgo2/ppo/policy.pt') == result['policy_sha256']
        assert meta['completed'] and collection['complete']
        t0 = collection['ppo_pd_start_sim_s']
        normal = load(a.normal/'imu.csv')
        for r in normal:r['t_s']=r['sim_s']-t0;attitude(r)
        result['matched_policy_normal_closedloop_reference'] = {'path':str(a.normal),
            'metrics':summarize(normal,5,11),
            'note':'Earlier same-policy nominal Gazebo +.5m/s closed loop, own feedback; independent 6s window and zero lateral/yaw command, not a matched physical-state intervention.',
            'sha256':{str(a.normal/n):sha(a.normal/n) for n in ('metadata.json','collection.json','imu.csv')}}
    a.output_prefix.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for suffix,rows in [('.csv',main_rows),('_joint_traces.csv',trace_rows)]:
        with Path(str(a.output_prefix)+suffix).open('w') as f:
            w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    if a.plot:plot(a,real,policy,plot_data)
    print(json.dumps(main_rows,indent=2))


def plot(a,real,policy,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def save(fig,name):
        fig.savefig(str(a.output_prefix)+'_'+name+'.png',dpi=150,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(15,8),layout='constrained',sharex=True)
    pp=[r for r in real if r['mode']==3]
    axes[0].plot([r['t_s'] for r in pp],[r['gyro_x'] for r in pp],lw=.5)
    axes[1].plot([r['t_s'] for r in pp],[r['roll_deg'] for r in pp],lw=.6)
    for key in ('cmd_vx','cmd_vy','cmd_wz'):axes[2].plot([r['t_s'] for r in policy],[r[key] for r in policy],label=key)
    for c,_,_ in data.values():
        for ax in axes:ax.axvspan(c['start'],c['start']+c['duration'],alpha=.12)
        axes[0].text(c['start'],axes[0].get_ylim()[1]*.8,c['label'])
    for ax,unit in zip(axes,['gyro X rad/s','roll deg','commands m/s, rad/s']):ax.set_ylabel(unit);ax.grid(alpha=.2)
    axes[2].legend();axes[2].set_xlabel('Source time (s)')
    fig.suptitle('Nov17 real PPO log | three selected command segments\nRope/handling occurred; intervention times unconfirmed')
    save(fig,'timeline')
    fig,axes=plt.subplots(3,len(data),figsize=(16,9),layout='constrained',squeeze=False)
    fs,fas=plt.subplots(len(data),2,figsize=(13,9),layout='constrained',squeeze=False)
    fj,jax=plt.subplots(3,len(data),figsize=(16,9),layout='constrained',squeeze=False)
    for col,(label,(c,sim,traces)) in enumerate(data.items()):
        lo,hi=c['analysis_window']
        for rows,name in [(real,'real'),(sim,'obs -> actor -> Gazebo')]:
            rr=[r for r in rows if c['start']<=r['t_s']<=c['start']+c['duration']]
            for row,key in [(0,'roll_deg'),(1,'gyro_x')]:axes[row,col].plot([r['t_s'] for r in rr],[r[key] for r in rr],label=name,lw=.8)
            for j,key in enumerate(['gyro_x','roll_deg']):
                _,bins=spectrum(wave(rows,key,lo,hi));fas[col,j].plot([v[0] for v in bins],[v[1] for v in bins],label=name)
        pp=[r for r in policy if c['start']<=r['t_s']<=c['start']+c['duration']]
        for key in ('cmd_vx','cmd_vy','cmd_wz'):axes[2,col].plot([r['t_s'] for r in pp],[r[key] for r in pp],label=key,lw=1)
        axes[0,col].set_title(label)
        for row,i in enumerate((9,10,11)):
            ax=jax[row,col]
            for key,name in [(f'target_{i:02}','target'),(f'real_q_{i:02}','real q'),(f'replay_q_{i:02}','Gazebo feedback q')]:
                ax.plot([r['source_s'] for r in traces],[math.degrees(r[key]) for r in traces],label=name,lw=.9)
            ax.set(title=label+' | RR '+('hip','thigh','shank')[row],ylabel='deg',xlabel='Source time (s)');ax.grid(alpha=.2);ax.legend(fontsize=8)
        for row,unit in enumerate(['roll deg','gyro X rad/s','commands m/s, rad/s']):
            ax=axes[row,col];ax.axvspan(lo,hi,color='orange',alpha=.12);ax.set_ylabel(unit);ax.grid(alpha=.2);ax.legend(fontsize=8)
            if row==2:ax.set_xlabel('Source time (s)')
        for j,unit in enumerate(['rad/s peak amplitude','deg peak amplitude']):
            ax=fas[col,j];ax.set(title=label+' | '+('gyro X','roll')[j]+f' {lo:g}..{hi:g}s',xlabel='Hz',ylabel=unit,xlim=(0,20));ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Matched policy 56d6 | unchanged real obs/history -> actor -> Gazebo | P25/D0.8\nReal rope/handling times unconfirmed; replay has no assistance. Shaded analysis windows.')
    fs.suptitle('Time-aligned descriptive spectra | no fitted lag; frequency grids differ by window duration')
    fj.suptitle('Same target history | RR hip/thigh/shank feedback | no phase fitting\nAligned time windows, not next-feedback residuals or independent load calibration')
    save(fig,'body');save(fs,'spectra');save(fj,'joints')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--normal',type=Path)
    p.add_argument('--output-prefix',type=Path,required=True)
    p.add_argument('--plot',action='store_true')
    main(p.parse_args())
