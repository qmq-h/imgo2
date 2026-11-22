#!/usr/bin/env python3
"""Offline roll/hip IO timing and direct actuator reaction component comparison.

Standard-library analysis, optional scientific plots. No inferred device delay,
total base torque, physical loop gain, or causal stability claim.
"""
import argparse
import bisect
import cmath
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import xml.etree.ElementTree as ET

from analyze_command_history import map_host
from audit_paired_obs_replay import spectrum, rms

ROOT = Path(__file__).resolve().parents[3]
HIPS = [0, 3, 6, 9]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open() as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def corr(x, y):
    mx, my = st.mean(x), st.mean(y)
    d = math.sqrt(sum((v-mx)**2 for v in x)*sum((v-my)**2 for v in y))
    return sum((a-mx)*(b-my) for a, b in zip(x, y))/d if d > 1e-16 else None


def roll(r):
    w, x, y, z = [r['quat_'+k] for k in 'wxyz']
    n = math.sqrt(w*w+x*x+y*y+z*z); w, x, y, z = [a/n for a in (w, x, y, z)]
    return math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))


class Samples:
    def __init__(self, rows, key='t'):
        self.rows = rows
        self.times = [r[key] for r in rows]
        assert all(b > a for a, b in zip(self.times, self.times[1:])), 'nonmonotonic time'

    def at(self, t, key, hold=False):
        tt = self.times
        assert tt[0] <= t <= tt[-1], 'outside sample coverage'
        i = min(len(tt)-1, max(1, bisect.bisect_right(tt, t)))
        assert tt[i]-tt[i-1] <= .05, 'do not interpolate gaps over50ms'
        if hold:
            return self.rows[i-1][key]
        f = (t-tt[i-1])/(tt[i]-tt[i-1])
        return self.rows[i-1][key]*(1-f)+self.rows[i][key]*f


def complex_amplitude(x, frequency, hz=100):
    n = len(x); mean = st.mean(x)
    win = [.5-.5*math.cos(2*math.pi*j/(n-1)) for j in range(n)]
    return sum((v-mean)*w*cmath.exp(-2j*math.pi*frequency*j/hz)
               for j, (v, w) in enumerate(zip(x, win)))


def phase(x, y, frequency):
    a, b = complex_amplitude(x, frequency), complex_amplitude(y, frequency)
    # Three overlapping 1.4s segments; this is descriptive short-window coupling.
    pairs = [(complex_amplitude(x[s:s+140], frequency), complex_amplitude(y[s:s+140], frequency))
             for s in (0, 70, 140)]
    cross = sum(a.conjugate()*b for a, b in pairs)
    xx, yy = sum(abs(a)**2 for a, _ in pairs), sum(abs(b)**2 for _, b in pairs)
    return {'y_phase_minus_x_deg': math.degrees(cmath.phase(a.conjugate()*b)),
            'short_segment_coherence': abs(cross)**2/(xx*yy) if xx*yy > 1e-20 else None,
            'amplitude_ratio': abs(b)/abs(a) if abs(a) > 1e-12 else None,
            'segments': 3,
            'limit': 'Wrapped harmonic phase; not causal ordering or measured latency;3 short segments only'}


def audit(name, streams, lo, hi):
    times = [lo+j*.01 for j in range(round((hi-lo)*100))]
    body, fb, cmd, actor = streams
    data = {'t_relative_s': [t-lo for t in times]}
    for key in ('omega', 'roll'):
        data[key] = [body.at(t, key) for t in times]
    for key in ['q', 'dq', 'M_feedback']+[f'{kind}_{i}' for kind in ['tau', 'q', 'dq'] for i in HIPS]:
        data[key] = [fb.at(t, key) for t in times]
    for key in ['target', 'kp', 'kd']+[f'target_{i}' for i in HIPS]:
        data[key] = [cmd.at(t, key, hold=True) for t in times]
    data['M_pd_submitted'] = [-4*(p*(u-q)-d*v) for p, d, u, q, v in
                            zip(data['kp'], data['kd'], data['target'], data['q'], data['dq'])]
    data['actor_target'] = [actor.at(t, 'target', hold=True) for t in times]
    data['M_actor_proposal'] = [actor.at(t, 'M_proposal', hold=True) for t in times]
    data['actor_omega'] = [actor.at(t, 'omega') for t in times]
    for key in ('M_feedback', 'M_pd_submitted'):
        data[key+'_times_omega'] = [a*b for a, b in zip(data[key], data['omega'])]
    spec, _ = spectrum(data['omega'])
    # Select 4-10Hz using gyro alone, never maximizing torque/target correlation.
    _, bins = spectrum(data['omega'])
    frequency = max((x for x in bins if 4 <= x[0] <= 10), key=lambda x: x[1])[0]
    pair_stats = {}
    for key in ('target', 'q', 'M_feedback', 'M_pd_submitted'):
        pair_stats['omega_to_'+key] = {'correlation': corr(data['omega'], data[key]),
                                      **phase(data['omega'], data[key], frequency)}
    pair_stats['target_to_q'] = {'correlation': corr(data['target'], data['q']),
                                **phase(data['target'], data['q'], frequency)}
    pair_stats['pd_to_feedback_M'] = {'correlation': corr(data['M_pd_submitted'], data['M_feedback']),
                                     **phase(data['M_pd_submitted'], data['M_feedback'], frequency)}
    pd_rows = [{'t': t, 'M': v} for t, v in zip(times, data['M_pd_submitted'])]
    pd_samples = Samples(pd_rows)
    fixed_shift = {}
    for ms in (0, 5, 10):
        chosen = [j for j, t in enumerate(times) if lo+.02 <= t < hi-.02]
        proposed = [pd_samples.at(times[j]-ms*.001, 'M') for j in chosen]
        actual = [data['M_feedback'][j] for j in chosen]
        fixed_shift[str(ms)] = {'correlation': corr(proposed, actual),
                               'rmse_nm': rms([a-b for a, b in zip(proposed, actual)])}
    pair_stats['fixed_pd_timestamp_shift_sensitivity_ms'] = fixed_shift
    moments = {}
    for key in ('M_feedback', 'M_pd_submitted'):
        p = data[key+'_times_omega']
        shifts = {}
        for ms in (-10, -5, 0, 5, 10):
            shifted = [fb.at(t-ms*.001, 'M_feedback') for t in times] if key == 'M_feedback' else None
            if shifted is not None:
                shifts[str(ms)] = st.mean(a*b for a, b in zip(shifted, data['omega']))
        moments[key] = {'mean_M_times_omega_component_w': st.mean(p),
                        'opposes_angular_velocity_fraction': st.mean(v < 0 for v in p),
                        'minus_mean_Momega_over_mean_omega2_Nms': -st.mean(p)/st.mean(v*v for v in data['omega']),
                        'timestamp_shift_sensitivity_ms': shifts,
                        'limit': 'Only direct motor reaction component; ratio is not identified net damping'}
    per_hip = {}
    for i in HIPS:
        m = [-x for x in data[f'tau_{i}']]
        per_hip[str(i)] = {'mean_direct_component_Momega_w': st.mean(a*b for a, b in zip(m, data['omega'])),
                           'corr_direct_M_vs_omega': corr(m, data['omega']),
                           'tracking_error_rms_deg': math.degrees(rms([a-b for a,b in zip(data[f'target_{i}'],data[f'q_{i}'])]))}
    native = []
    for r, nxt in zip(fb.rows, fb.rows[1:]):
        t = r['t']
        if lo <= t < hi:
            dt = min(nxt['t'], hi)-t
            native.append((r['M_feedback']*body.at(t, 'omega'), dt))
    native_check = {'rows': len(native), 'covered_duration_s': sum(dt for _, dt in native),
                    'time_weighted_Momega_w': sum(p*dt for p,dt in native)/sum(dt for _,dt in native),
                    'opposing_time_fraction': sum(dt for p,dt in native if p<0)/sum(dt for _,dt in native)}
    summary = {'window_s': [lo, hi], 'rows_100hz': len(times),
               'gyro_selected_4_10_hz': frequency,
               'omega_rms_rad_s': rms(data['omega']),
               'roll_mean_deg': math.degrees(st.mean(data['roll'])),
               'roll_std_deg': math.degrees(st.pstdev(data['roll'])),
               'common_target_std_deg': math.degrees(st.pstdev(data['target'])),
               'common_q_std_deg': math.degrees(st.pstdev(data['q'])),
               'common_tracking_error_rms_deg': math.degrees(rms([u-q for u, q in zip(data['target'], data['q'])])),
               'direct_reaction_estimate_rms_nm': rms(data['M_feedback']),
               'phase_and_correlations': pair_stats, 'moment_component': moments,
               'per_hip_direct_reaction': per_hip, 'native_feedback_time_weighted_check': native_check,
               'policy_snapshot_component': {
                   'mean_proposed_M_times_input_omega_w': st.mean(a*b for a,b in zip(data['M_actor_proposal'], data['actor_omega'])),
                   'limit': 'Nominal PD evaluated on the forward snapshot and proposed target; not transmitted/measured motor torque'}}
    return summary, data


def main(a):
    real = ROOT/'imgo2_deploy/logs/2026-11-17_21-10-22_7163'
    normal = ROOT/'imgo2_deploy/logs/height23_forward05_kd08_20261109_a'
    urdf = ROOT/'imgo2_description/urdf/imgo2_real.urdf'
    tree = ET.parse(urdf).getroot()
    for leg in ['FL', 'FR', 'RL', 'RR']:
        joint = tree.find(f"joint[@name='{leg}_hip_joint']")
        assert joint.find('parent').get('link') == 'base'
        assert joint.find('axis').get('xyz') == '1 0 0'
        assert list(map(float, joint.find('origin').get('rpy', '0 0 0').split())) == [0, 0, 0]
    # Exact same captured source identity as the preceding full-forward validation.
    identity_path = ROOT/'docs/deployment/ppo_input_attribution_2026-11-18.json'
    identity = json.loads(identity_path.read_text())
    assert identity['policy_sha256'].startswith('56d6d043')
    for p in [real/'policy.csv', normal/'policy_raw.csv']:
        assert identity['sha256'][str(p.relative_to(ROOT))] == sha(p)
    raw = []
    with (real/'robot.csv').open() as f:
        for row in csv.DictReader(f):
            t = float(row['t_mono_ms'])/1000
            if not 71.5 <= t <= 100.6:
                continue
            cols = ['quat_'+k for k in 'wxyz']+['gyro_x']
            cols += [f'{k}_{i:02}' for k in ('q_des', 'q', 'dq', 'tau', 'kp', 'kd') for i in HIPS]
            r = {k: float(row[k]) for k in cols}
            assert all(abs(r[f'kp_{i:02}']-25) < 1e-6 and abs(r[f'kd_{i:02}']-.8) < 1e-6 for i in HIPS)
            raw.append({'t': t, 'omega': r['gyro_x'], 'roll': roll(r),
                        'q': st.mean(r[f'q_{i:02}'] for i in HIPS),
                        'dq': st.mean(r[f'dq_{i:02}'] for i in HIPS),
                        'target': st.mean(r[f'q_des_{i:02}'] for i in HIPS),
                        'kp': 25., 'kd': .8,
                        'M_feedback': -sum(r[f'tau_{i:02}'] for i in HIPS),
                        **{f'{kind}_{i}': r[f'{source}_{i:02}'] for kind,source in
                           [('tau','tau'),('q','q'),('dq','dq'),('target','q_des')] for i in HIPS}})
    rp = read(real/'policy.csv')
    actor_real = []
    delay = []
    for r in rp:
        t = r['input_captured_ms']/1000
        if not 71.5 <= t <= 100.6:
            continue
        target = st.mean(.125*r[f'act_{i:02}'] for i in HIPS)
        q = st.mean(r[f'obs_{9+i:02}'] for i in HIPS)
        dq = st.mean(r[f'obs_{21+i:02}']/.05 for i in HIPS)
        actor_real.append({'t': t, 'target': target, 'omega': r['obs_00']/.25,
                           'M_proposal': -4*(25*(target-q)-.8*dq)})
        delay.append({'t': t, 'capture_to_consumption_ms': r['t_mono_ms']-r['input_captured_ms'],
                      'forward_ms': r['result_generated_ms']-r['input_captured_ms']})
    odom = read(normal/'odom.csv')
    ns, nc, np = [read(normal/n) for n in ('state.csv', 'command.csv', 'policy_raw.csv')]
    for rows in [ns, nc, np]: map_host(rows, odom)
    nimu = read(normal/'imu.csv')
    nbody = [{'t': r['sim_s'], 'omega': r['gyro_x'], 'roll': roll(r)} for r in nimu]
    nfb = [{'t': r['t_s'], 'q': st.mean(r[f'q_{i:02}'] for i in HIPS),
            'dq': st.mean(r[f'dq_{i:02}'] for i in HIPS),
            'M_feedback': -sum(r[f'tau_{i:02}'] for i in HIPS),
            **{f'{kind}_{i}': r[f'{kind}_{i:02}'] for kind in ['tau','q','dq'] for i in HIPS}} for r in ns]
    ncmd = [{'t': r['t_s'], 'target': st.mean(r[f'q_{i:02}'] for i in HIPS),
             'kp': st.mean(r[f'kp_{i:02}'] for i in HIPS),
             'kd': st.mean(r[f'kd_{i:02}'] for i in HIPS),
             **{f'target_{i}':r[f'q_{i:02}'] for i in HIPS}} for r in nc]
    nactor = [{'t': r['t_s'], 'target': st.mean(r[f'target_{i}'] for i in HIPS),
               'omega': r['obs_0']/.25,
               'M_proposal': -sum(25*(r[f'target_{i}']-r[f'state_q_{i}'])-.8*r[f'state_dq_{i}'] for i in HIPS)} for r in np]
    real_streams = [Samples(raw), Samples(raw), Samples(raw), Samples(actor_real)]
    normal_streams = [Samples(nbody), Samples(nfb), Samples(ncmd), Samples(nactor)]
    cases = {'real_A': (real_streams, 71.9, 74.7), 'real_B': (real_streams, 76.8, 79.6),
             'real_C': (real_streams, 97.4, 100.2), 'normal_A': (normal_streams, 16.5, 19.3),
             'normal_B': (normal_streams, 19.5, 22.3)}
    report = {'policy_sha256': identity['policy_sha256'], 'cases': {},
              'formula': 'common=mean4hip; direct_M_base_x=-sum4hip_output_torque; component=M*body_gyroX',
              'limits': ['Direct actuator reaction only, excluding joint constraint-force moments, ground-contact wrench and all other external/constraint moments.',
                         'Real estimated torque not independently calibrated; host snapshots and ROS mapped times are not synchronized physical source/application clocks.',
                         'Negative M*omega describes this component opposing angular velocity, not net robot energy dissipation or a stability guarantee.',
                         'Real rope/handling periods unknown; small lateral commands, actual real base speed unmeasured.',
                         'Phase at periodic frequency is not causal time order or measured device latency; timing shifts are sensitivity checks only.',
                         'Same56d6 same25/.8 but independent histories and gait phases; no cross-run pointwise errors.',
                         'No new simulation, hardware, training or production changes.']}
    traces = {}
    for name, (streams, lo, hi) in cases.items():
        report['cases'][name], traces[name] = audit(name, streams, lo, hi)
        if name.startswith('real'):
            d = [r for r in delay if lo <= r['t'] < hi]
            report['cases'][name]['actor_timing'] = {key: {'mean': st.mean(r[key] for r in d),
                'min': min(r[key] for r in d), 'max': max(r[key] for r in d)}
                for key in ['capture_to_consumption_ms', 'forward_ms']}
    files = [real/'robot.csv', real/'policy.csv', real/'meta.json', urdf, identity_path,
             normal/'state.csv', normal/'command.csv', normal/'policy_raw.csv', normal/'imu.csv',
             normal/'odom.csv', Path(__file__)]
    report['sha256'] = {str(p.relative_to(ROOT)): sha(p) for p in files}
    a.output.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    with a.output.with_suffix('.csv').open('w') as f:
        keys = list(traces['real_B'])
        w = csv.writer(f); w.writerow(['case']+keys)
        for name, data in traces.items():
            for i in range(len(data['omega'])): w.writerow([name]+[data[k][i] for k in keys])
    if a.plot: plot(a.output, report, traces)
    for name, case in report['cases'].items():
        m = case['moment_component']['M_feedback']
        print(name, 'gyro_hz',case['gyro_selected_4_10_hz'], 'component_Momega_w',
              m['mean_M_times_omega_component_w'], 'opposite_fraction',m['opposes_angular_velocity_fraction'])


def plot(output, report, traces):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import subprocess
    # TTC collections can be indexed by their first family rather than the
    # Fontconfig SC alias. Register the resolved file and use its actual name.
    font_file = subprocess.check_output(['fc-match', '-f', '%{file}', 'Noto Sans CJK SC'], text=True).strip()
    font_manager.fontManager.addfont(font_file)
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_file).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(4, 2, figsize=(15, 11), constrained_layout=True)
    for col, case in enumerate(['real_B', 'normal_A']):
        d = traces[case]; t = d['t_relative_s']
        axes[0, col].plot(t, [math.degrees(x) for x in d['roll']], color='#444444', lw=1)
        label = '实机前进B' if case == 'real_B' else '正常Gazebo A'
        axes[0, col].set(title=label+'：身体横滚角', ylabel='度')
        axes[1, col].plot(t, d['omega'], color='#0077bb', lw=1)
        axes[1, col].set(title='身体横滚角速度', ylabel='rad/s')
        axes[2, col].plot(t, [math.degrees(x) for x in d['target']], label='已提交的四髋平均目标', lw=1)
        axes[2, col].plot(t, [math.degrees(x) for x in d['q']], label='四髋平均实际角度', lw=1)
        axes[2, col].set(title='四髋共同角度（不是单腿摆幅）', ylabel='度')
        axes[2, col].legend(fontsize=8)
        axes[3, col].plot(t, d['M_feedback'], label='直接反作用：反馈力矩估算', lw=1)
        axes[3, col].plot(t, d['M_pd_submitted'], label='直接反作用：本拍PD公式参考', lw=1, alpha=.75)
        axes[3, col].set(title='仅髋电机对机身的直接反作用', ylabel='N·m', xlabel='窗口内时间（秒）；两次运行相位独立')
        axes[3, col].legend(fontsize=8)
    for ax in axes.flat: ax.grid(alpha=.2); ax.axhline(0, color='black', lw=.5)
    # Keep equal row scales across runs so relative amplitude is visually honest.
    for row in range(4):
        lo = min(ax.get_ylim()[0] for ax in axes[row]); hi = max(ax.get_ylim()[1] for ax in axes[row])
        for ax in axes[row]: ax.set_ylim(lo, hi)
    fig.suptitle('身体左右摇晃与髋部动作：同一策略56d6、PD25/0.8；每行左右使用同一纵轴范围')
    fig.savefig(str(output)+'_timing.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for col, case in enumerate(['real_B', 'normal_A']):
        d = traces[case]; t = d['t_relative_s']; keep = [i for i,v in enumerate(t) if .7 <= v <= 1.7]
        ax = axes[0, col]; other = ax.twinx()
        ax.plot([t[i] for i in keep], [d['omega'][i] for i in keep], color='#0077bb', label='身体横滚速度')
        other.plot([t[i] for i in keep], [d['M_feedback'][i] for i in keep], color='#ee7733', label='髋电机直接反力矩估算')
        omega_limit = max(abs(d['omega'][i]) for i in keep)*1.1
        moment_limit = max(abs(d['M_feedback'][i]) for i in keep)*1.1
        ax.set_ylim(-omega_limit, omega_limit); other.set_ylim(-moment_limit, moment_limit)
        ax.axhline(0, color='black', lw=.5)
        label = '实机前进B' if case == 'real_B' else '正常Gazebo A'
        ax.set(title=label+'：横滚速度与髋电机反力矩（放大1秒）', ylabel='横滚速度 rad/s')
        other.set_ylabel('Nm'); ax.legend(loc='upper left', fontsize=8); other.legend(loc='upper right', fontsize=8)
        ax = axes[1, col]; power = d['M_feedback_times_omega']
        ax.plot(t, power, color='#555555', lw=.8)
        ax.fill_between(t, power, 0, where=[v<0 for v in power], color='#228833', alpha=.5, label='这部分反作用在抵抗摇动')
        ax.fill_between(t, power, 0, where=[v>=0 for v in power], color='#cc3311', alpha=.5, label='这部分反作用与摇动同向')
        fraction = report['cases'][case]['moment_component']['M_feedback']['opposes_angular_velocity_fraction']
        ax.set(title=f'直接反力矩×横滚速度；反向作用占{fraction:.1%}记录点', ylabel='W（仅此分量）', xlabel='窗口内时间（秒）')
        ax.legend(fontsize=8); ax.grid(alpha=.2); ax.axhline(0, color='black', lw=.5)
    fig.suptitle('仅检查髋电机这一部分的作用方向：不含地面和约束力矩，不能据此判定整机净阻尼')
    fig.savefig(str(output)+'_direction.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(4,2,figsize=(14,10),constrained_layout=True)
    for col,case in enumerate(['real_B','normal_A']):
        d=traces[case]; label='实机前进B' if col==0 else '正常Gazebo A'
        for row,(leg,i) in enumerate(zip(['左前','右前','左后','右后'],HIPS)):
            ax=axes[row,col]
            ax.plot(d['t_relative_s'],[math.degrees(v) for v in d[f'target_{i}']],label='目标角度')
            ax.plot(d['t_relative_s'],[math.degrees(v) for v in d[f'q_{i}']],label='反馈角度')
            ax.set(title=label+' / '+leg+'髋关节',ylabel='度',xlabel='窗口内时间（秒）')
            ax.grid(alpha=.2)
            if row==0:ax.legend(fontsize=8)
    for row in range(4):
        lo=min(ax.get_ylim()[0] for ax in axes[row]);hi=max(ax.get_ylim()[1] for ax in axes[row])
        for ax in axes[row]:ax.set_ylim(lo,hi)
    fig.suptitle('四个髋关节分别查看：已提交目标与实际反馈（不是跨运行逐点误差）')
    fig.savefig(str(output)+'_four_hips.png',dpi=160);plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plot', action='store_true')
    main(p.parse_args())
