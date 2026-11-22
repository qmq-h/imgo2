#!/usr/bin/env python3
"""Same recorded autonomous targets: real, previous and repeat Gazebo runs.

Stdlib calculations; optional Matplotlib. No fitted time shift or gain changes.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from audit_paired_obs_replay import load, attitude, summarize, rms, spectrum, wave
from analyze_rope_replay import check_replay
from compare_pd_torque_time import before, interp, construct, values, metrics, GROUPS


def paired(a, b):
    return {kind: {term+'_rmse': rms([x[f'{term}_{i:02}']-y[f'{term}_{i:02}'] for x, y in zip(a, b) for i in ids])
        for term in ('q', 'dq', 'P', 'D', 'nominal', 'feedback')} for kind, ids in GROUPS.items()}


def waveform_agreement(real, sim):
    import statistics as st
    def correlation(x, y):
        mx, my = st.mean(x), st.mean(y)
        den = math.sqrt(sum((v-mx)**2 for v in x)*sum((v-my)**2 for v in y))
        return sum((a-mx)*(b-my) for a, b in zip(x, y))/den if den else None
    joints = []
    for i in range(12):
        row = {'joint': ('FL','FR','RL','RR')[i//3]+' '+('hip','thigh','shank')[i%3]}
        for term in ('q', 'dq', 'nominal', 'feedback'):
            x = [r[f'{term}_{i:02}'] for r in real]; y = [r[f'{term}_{i:02}'] for r in sim]
            error = [b-a for a, b in zip(x, y)]; bias = st.mean(error)
            row[term] = {'correlation': correlation(x, y), 'bias_sim_minus_real': bias,
                'rmse': rms(error), 'demeaned_rmse': rms([v-bias for v in error]),
                'real_std': st.pstdev(x), 'sim_std': st.pstdev(y)}
        joints.append(row)
    body = {term: {'correlation': correlation([r[term] for r in real], [r[term] for r in sim]),
        'mean_difference': st.mean(r[term] for r in sim)-st.mean(r[term] for r in real)}
        for term in ('roll_deg','pitch_deg','gyro_x','gyro_y')}
    return {'method': 'Same source timeline, no phase fit; per-joint Pearson correlation, mean sim-real bias and demeaned RMSE. Correlation does not measure equal amplitude or identify causality.',
        'joints': joints, 'body': body}


def main(a):
    real = load(a.source/'robot.csv')
    for r in real: r['t_s'] = r['t_mono_ms']*.001; attitude(r)
    old, oc = check_replay(a.previous, real); new, nc = check_replay(a.repeat, real)
    assert (a.previous/'targets.csv').read_bytes() == (a.repeat/'targets.csv').read_bytes()
    assert (a.previous/'initial.csv').read_bytes() == (a.repeat/'initial.csv').read_bytes()
    assert oc['metadata']['protected_hashes'] == nc['metadata']['protected_hashes']
    assert oc['metadata']['plugin_sha256'] == nc['metadata']['plugin_sha256']
    for checks in (oc, nc):
        assert checks['metadata']['production_files_unchanged']
    lo, hi = a.lo, a.hi; rt = [r['t_s'] for r in real]
    native = {'real': real, 'previous_gazebo': old, 'repeat_gazebo': new}; aligned = {}
    for label, rows in native.items():
        times = [r['t_s'] for r in rows]; out = []
        for j in range(round((hi-lo)*200)):
            t = lo+j/200; cmd = real[before(rt, t)]
            assert cmd['mode'] == 3 and cmd['valid_mask'] == 4095 and cmd['err_mask'] == 0
            if label == 'real': q, dq, fb = values(cmd, 'q'), values(cmd, 'dq'), values(cmd, 'tau')
            else:
                q = [interp(rows, times, t, f'q_{i:02}') for i in range(12)]
                dq = [interp(rows, times, t, f'dq_{i:02}') for i in range(12)]
                fb = [interp(rows, times, t, f'tau_{i:02}') for i in range(12)]
            r = construct(t, q, dq, values(cmd, 'q_des'), values(cmd, 'dq_des'), values(cmd, 'tau_ff'), values(cmd, 'kp'), values(cmd, 'kd'), fb)
            assert all(r[f'kp_{i:02}'] == 25 and r[f'kd_{i:02}'] == .8 for i in range(12))
            for key in ('roll_deg', 'pitch_deg', 'gyro_x', 'gyro_y', 'gyro_z'):
                r[key] = cmd[key] if label == 'real' else interp(rows, times, t, key)
            out.append(r)
        aligned[label] = out
    result = {'source': str(a.source), 'previous': str(a.previous), 'repeat': str(a.repeat), 'window_s': [lo, hi],
        'checks': {'targets_byte_identical': True, 'initial_byte_identical': True, 'protected_hashes_identical': True,
            'plugin_identical': True, 'previous': oc, 'repeat': nc},
        'cases': {label: {'body_native': summarize(rows, lo, hi, real), 'joint_common_200Hz': metrics(aligned[label])} for label, rows in native.items()},
        'same_target_repeat_minus_real': paired(aligned['repeat_gazebo'], aligned['real']),
        'same_target_previous_minus_real': paired(aligned['previous_gazebo'], aligned['real']),
        'repeat_minus_previous': paired(aligned['repeat_gazebo'], aligned['previous_gazebo']),
        'body_repeat_minus_previous_rms': {key: rms([x[key]-y[key] for x, y in zip(aligned['repeat_gazebo'], aligned['previous_gazebo'])]) for key in ('roll_deg', 'pitch_deg', 'gyro_x')},
        'waveform_agreement': waveform_agreement(aligned['real'], aligned['repeat_gazebo']),
        'limitations': ['Open-loop original target replay, not simulated-feedback policy inference.',
            'Same initial snapshot means identical between simulations; real base height/linear/angular velocity/contact solver state are not measured completely.',
            'No phase fit or artificial compensation. Real command/feedback host records and simulated controller feedback have source/application-time uncertainty.',
            'Nominal PD is a logged-state calculation; feedback torque is estimated, not independent calibrated joint mechanical load.',
            'Body metrics native sample cadence; joint comparison on common 200Hz grid. Replay joint q/dq come from actual controller feedback.',
            '2.5s core frequency resolution 0.4Hz; closeness of RMS does not mean pointwise matching or identify causal direction.'],
        'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.source/'robot.csv', a.previous/'targets.csv', a.repeat/'targets.csv')}}
    a.output_prefix.with_suffix('.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    if a.agreement_output:
        a.agreement_output.write_text(json.dumps({'window_s': [lo, hi], **result['waveform_agreement']}, indent=2, allow_nan=False)+'\n')
    with Path(str(a.output_prefix)+'_traces.csv').open('w') as f:
        fields = ['case']+list(aligned['real'][0]); w = csv.DictWriter(f, fields); w.writeheader()
        for label, rows in aligned.items():
            for r in rows: w.writerow(dict(case=label, **r))
    with a.output_prefix.with_suffix('.csv').open('w') as f:
        w = csv.writer(f); w.writerow(['case', 'joint', 'term', 'mean', 'rms', 'std', 'abs_p95', 'abs_max', 'peak_to_peak'])
        for label, case in result['cases'].items():
            for i, joint in enumerate(case['joint_common_200Hz']['joints']):
                for term in ('error', 'dq', 'P', 'D', 'nominal', 'feedback', 'residual'):
                    m = joint[term]; w.writerow([label, i, term]+[m[k] for k in ('mean','rms','std','abs_p95','abs_max','peak_to_peak')])
    for label, r in result['cases'].items():
        m = r['body_native']; print(label, {k: m[k] for k in ('gyro_x_rms_rad_s','roll_std_deg','roll_half_p95_p05_deg','gyro_spectrum')})
    print('repeat_minus_previous',result['repeat_minus_previous']); print('repeat_minus_real',result['same_target_repeat_minus_real'])
    if a.plot: plot(a.output_prefix, native, aligned, lo, hi, nc['metadata'])


def plot(prefix, native, aligned, lo, hi, meta):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'real': '#0077bb', 'previous_gazebo': '#228833', 'repeat_gazebo': '#ee7733'}
    def save(fig, suffix): fig.savefig(str(prefix)+'_'+suffix+'.png', dpi=160, bbox_inches='tight'); plt.close(fig)
    fig, ax = plt.subplots(2,2,figsize=(15,9),layout='constrained')
    for label, rows in native.items():
        rr = [r for r in rows if meta['source_start_s'] <= r['t_s'] <= meta['source_start_s']+meta['duration_s']]
        for axis, key in zip(ax.flat, ('roll_deg','gyro_x','pitch_deg','gyro_y')):
            axis.plot([r['t_s'] for r in rr],[r[key] for r in rr],label=label,color=colors[label],lw=.75)
            axis.axvspan(lo,hi,color='gray',alpha=.035);axis.set(xlabel='Source log time (s)',ylabel=key);axis.grid(alpha=.2)
    ax[0,0].legend(fontsize=8);fig.suptitle('Autonomous targets replayed again | identical targets and simulated initial snapshot\nOriginal P25/D0.8, full gravity, no rope, no fitted time shift; gray band is comparison window')
    save(fig,'body')
    for leg,name in enumerate(('FL','FR','RL','RR')):
        fig,ax=plt.subplots(3,3,figsize=(16,10),layout='constrained')
        for col,kind in enumerate(('hip','thigh','shank')):
            i=leg*3+col
            for label in ('real','repeat_gazebo'):
                rows=aligned[label];t=[r['t']-lo for r in rows]
                for row,term in enumerate(('q','dq','nominal')):
                    ax[row,col].plot(t,[r[f'{term}_{i:02}'] for r in rows],label=label,color=colors[label],lw=.9)
                ax[2,col].plot(t,[r[f'feedback_{i:02}'] for r in rows],label=label+' estimated feedback',color=colors[label],ls=':',alpha=.6,lw=.8)
            rows=aligned['real'];ax[0,col].plot([r['t']-lo for r in rows],[r[f'target_{i:02}'] for r in rows],label='same target',color='black',ls='--',lw=.8)
            ax[0,col].set_title(kind);ax[0,col].set_ylabel('q / target (rad)');ax[1,col].set_ylabel('dq (rad/s)');ax[2,col].set_ylabel('nominal PD / dotted feedback (Nm)')
        for axis in ax.flat:axis.grid(alpha=.2);axis.set_xlabel('Time from '+str(lo)+'s (s)')
        ax[0,0].legend(fontsize=7);ax[2,0].legend(fontsize=7)
        fig.suptitle(name+' same-target autonomous response | real vs repeat Gazebo\nReal command/feedback is asynchronous; replay uses controller q/dq; no phase matching')
        save(fig,name)
    fig,ax=plt.subplots(4,3,figsize=(16,11),layout='constrained')
    for leg in range(4):
        for col,term in enumerate(('P','D','nominal')):
            for label in ('real','repeat_gazebo'):
                rows=aligned[label];ax[leg,col].plot([r['t']-lo for r in rows],[r[f'{term}_{leg*3:02}'] for r in rows],label=label,color=colors[label],lw=.85)
            ax[leg,col].set(title=('FL','FR','RL','RR')[leg]+' hip '+term,xlabel='Within-window time (s)',ylabel='Nm');ax[leg,col].grid(alpha=.2)
    ax[0,0].legend(fontsize=8);fig.suptitle('Hip P / D / nominal torque | same real targets, P25/D0.8')
    save(fig,'hip_PD')
    fig,ax=plt.subplots(1,2,figsize=(15,5),layout='constrained')
    for label,rows in native.items():
        for axis,key in zip(ax,('gyro_x','roll_deg')):
            _,bins=spectrum(wave(rows,key,lo,hi));axis.plot([f for f,v in bins],[v for f,v in bins],label=label,color=colors[label]);axis.set(title=key,xlabel='Hz',ylabel='Sinusoid amplitude',xlim=(1,15));axis.grid(alpha=.2)
    ax[0].legend(fontsize=8);fig.suptitle('Same autonomous window | 100Hz interpolation, detrend + Hann; 0.4Hz resolution')
    save(fig,'spectra')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','previous','repeat','output-prefix'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--lo',type=float,default=38.7);p.add_argument('--hi',type=float,default=41.2)
    p.add_argument('--agreement-output',type=Path,help='Optional standalone waveform correlation/bias JSON')
    p.add_argument('--plot',action='store_true');main(p.parse_args())
