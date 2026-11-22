#!/usr/bin/env python3
"""Offline PPO input comparison; stdlib statistics, Torch, optional plots.

Attribution describes adjacent observed input transitions, not physical causality.
No policies are exported, no robot/simulator is controlled, no input log is edited.
"""
import argparse
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st

from audit_paired_obs_replay import spectrum, rms
from analyze_command_history import map_host

ROOT = Path(__file__).resolve().parents[3]
GROUPS = {'gyro': list(range(3)), 'gravity': list(range(3, 6)),
          'commands': list(range(6, 9)), 'q': list(range(9, 21)),
          'dq': list(range(21, 33)), 'last_action': list(range(33, 45))}
JOINTS = [f'{leg} {joint}' for leg in ('FL', 'FR', 'RL', 'RR')
          for joint in ('hip', 'thigh', 'shank')]
LABELS = ['gyro '+x for x in 'XYZ']+['gravity '+x for x in 'XYZ']
LABELS += ['command '+x for x in ('vx', 'vy', 'wz')]
LABELS += [j+' q_rel' for j in JOINTS]+[j+' dq' for j in JOINTS]
LABELS += [j+' last_action' for j in JOINTS]
SCALE = [.125, .25, .25]*4


def read(p):
    with p.open() as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def corr(x, y):
    a, b = st.mean(x), st.mean(y)
    den = math.sqrt(sum((v-a)**2 for v in x)*sum((v-b)**2 for v in y))
    return sum((u-a)*(v-b) for u, v in zip(x, y))/den if den > 1e-16 else None


def waves(rows, lo, hi, hz=100):
    tt = [r['t'] for r in rows]
    assert all(b > a for a, b in zip(tt, tt[1:]))
    assert tt[0] <= lo and hi-1/hz <= tt[-1]
    result = [[] for _ in range(45)]
    for k in range(round((hi-lo)*hz)):
        t = lo+k/hz
        i = min(len(tt)-1, max(1, bisect.bisect_right(tt, t)))
        assert 0 < tt[i]-tt[i-1] <= .05, 'gap over50ms: no interpolation allowed'
        f = (t-tt[i-1])/(tt[i]-tt[i-1])
        for j in range(45):
            result[j].append(rows[i-1]['x'][j]*(1-f)+rows[i]['x'][j]*f)
    return result


def describe(rows, lo, hi):
    selected = [r for r in rows if lo <= r['t'] < hi]
    assert len(selected) >= 100
    data = waves(rows, lo, hi)
    channels, spectra = [], []
    for j, wave in enumerate(data):
        native = [r['x'][j] for r in selected]
        spec, bins = spectrum(wave)
        if st.pstdev(native) < 1e-9:
            spec['peak_1_20_hz'] = None
        channels.append({'index': j, 'label': LABELS[j], 'mean': st.mean(native),
                         'std': st.pstdev(native), 'rms': rms(native),
                         'peak_to_peak': max(native)-min(native), **spec})
        spectra.append(bins)
    groups = {}
    dims = {**GROUPS, **{f'{kind}_{joint}': [start+i for i in range(joint_id, 12, 3)]
            for kind, start in [('q', 9), ('dq', 21), ('last_action', 33)]
            for joint_id, joint in enumerate(('hip', 'thigh', 'shank'))}}
    for label, indices in dims.items():
        groups[label] = {k: rms([channels[j][k] for j in indices])
                         for k in ('std', 'rms', 'band_4_10_rms')}
    consistency = []
    for j in range(12):
        qdot = [(b['x'][9+j]-a['x'][9+j])/(b['t']-a['t']) for a, b in zip(selected, selected[1:])]
        dq = [(b['x'][21+j]+a['x'][21+j])/.1 for a, b in zip(selected, selected[1:])]
        consistency.append({'joint': JOINTS[j], 'q_difference_vs_average_dq_correlation': corr(qdot, dq),
                            'rmse_rad_s': rms([u-v for u, v in zip(qdot, dq)])})
    roll = [math.atan2(-r['x'][4], -r['x'][5]) for r in selected]
    derivative = [math.atan2(math.sin(b-a), math.cos(b-a))/(rb['t']-ra['t'])
                  for a, b, ra, rb in zip(roll, roll[1:], selected, selected[1:])]
    omega = [(a['x'][0]+b['x'][0])/.5 for a, b in zip(selected, selected[1:])]
    mode_signs = {'common': [1, 1, 1, 1], 'left_right': [1, -1, 1, -1],
                  'front_back': [1, 1, -1, -1], 'diagonal': [1, -1, -1, 1]}
    hip_modes = {}
    native_modes, uniform_modes = {}, {}
    output_rows = [{'t': r['t'], 'x': r['act']+[0.]*33} for r in rows]
    output_wave = waves(output_rows, lo, hi)
    for label, signs in mode_signs.items():
        native_modes[label] = [sum(s*r['act'][j]*SCALE[j] for s, j in zip(signs, [0, 3, 6, 9]))/4 for r in selected]
        # Actor action output corresponds to the next row's last_action. For
        # spectra use actual output, interpolated with the same gap gate.
        uniform_modes[label] = [sum(s*output_wave[j][k]*SCALE[j] for s, j in zip(signs, [0, 3, 6, 9]))/4
                                for k in range(len(data[0]))]
        spec, _ = spectrum(uniform_modes[label])
        hip_modes[label] = {'std_rad': st.pstdev(native_modes[label]),
                            'step_rms_rad': rms([b-a for a, b in zip(native_modes[label], native_modes[label][1:])]), **spec}
    band_energy = sum(v['band_4_10_rms']**2 for v in hip_modes.values())
    for v in hip_modes.values(): v['four_mode_band_energy_share'] = v['band_4_10_rms']**2/band_energy
    return {'window_s': [lo, hi], 'samples': len(selected),
            'max_actor_interval_ms': max((b['t']-a['t'])*1000 for a, b in zip(selected, selected[1:])),
            'channels': channels, 'groups_scaled_obs_units': groups,
            'commands': {k: [min(r['x'][j] for r in selected), max(r['x'][j] for r in selected)]
                         for k, j in [('vx', 6), ('vy', 7), ('wz', 8)]},
            'roll_from_gravity_std_deg': math.degrees(st.pstdev(roll)),
            'roll_from_gravity_mean_deg': math.degrees(st.mean(roll)),
            'roll_difference_vs_average_gyro_x': {'correlation': corr(derivative, omega),
                                                'rmse_rad_s': rms([a-b for a, b in zip(derivative, omega)]),
                                                'limit': 'roll rate approximately body gyroX near level; not independent ground truth'},
            'q_dq_consistency': consistency, 'hip_output_modes': hip_modes,
            'window_previous_action_max_error': max(abs(b['x'][33+j]-r['act'][j])
                for r, b in zip(selected, selected[1:]) for j in range(12))}, data, spectra, selected


def attribution(model, rows, torch):
    x = torch.tensor([r['x'] for r in rows], dtype=torch.float32)
    scale = torch.tensor(SCALE)
    before, delta = x[:-1], x[1:]-x[:-1]
    with torch.no_grad():
        target = model(x).clamp(-3, 3)*scale
    dy = target[1:]-target[:-1]
    for steps in (64, 256):
        alpha = (torch.arange(steps, dtype=torch.float32)+.5)/steps
        path = (before[:, None, :]+alpha[None, :, None]*delta[:, None, :]).reshape(-1, 45)
        path.requires_grad_(True)
        y = model(path).clamp(-3, 3)*scale
        terms = []
        for j in range(12):
            grad = torch.autograd.grad(y[:, j].sum(), path, retain_graph=j < 11)[0]
            terms.append(grad.reshape(len(before), steps, 45).mean(1)*delta)
        c = torch.stack(terms, dim=1)  # samples x output joints x input dimensions
        error = (c.sum(2)-dy).square().mean().sqrt().item()
        relative = error/max(dy.square().mean().sqrt().item(), 1e-10)
        if relative < .005:
            break
    modes = {'hip_pooled': [0, 3, 6, 9], 'thigh_pooled': [1, 4, 7, 10],
             'shank_pooled': [2, 5, 8, 11], 'hip_common': [0, 3, 6, 9]}
    result = {'samples': len(before), 'integration_midpoints': steps,
              'reconstruction_rmse_rad': error, 'reconstruction_relative_rms': relative,
              'method': 'Midpoint integrated input Jacobian along each adjacent actual obs pair; clipped output times joint scale',
              'modes': {}}
    for label, indices in modes.items():
        total = dy[:, indices]
        component = {g: c[:, indices][:, :, dims].sum(2) for g, dims in GROUPS.items()}
        if label == 'hip_common':
            total = total.mean(1, keepdim=True)
            component = {g: v.mean(1, keepdim=True) for g, v in component.items()}
        denom = total.square().mean().item()
        result['modes'][label] = {'target_step_rms_rad': math.sqrt(denom), 'groups': {
            g: {'contribution_rms_rad': v.square().mean().sqrt().item(),
                'signed_covariance_share': (v*total).mean().item()/max(denom, 1e-20)}
            for g, v in component.items()}}
    # Conditional command standardization; not a simulated trajectory.
    with torch.no_grad():
        standard = x.clone(); standard[:, 6:9] = torch.tensor([.5, 0., 0.])
        ys = model(standard).clamp(-3, 3)*scale
    result['conditional_standardize_commands'] = {
        'all_joint_target_change_rms_rad': (ys-target).square().mean().sqrt().item(),
        'hip_target_step_rms_rad': (ys[1:, [0, 3, 6, 9]]-ys[:-1, [0, 3, 6, 9]]).square().mean().sqrt().item()}
    endpoint = x.clone().requires_grad_(True)
    y = model(endpoint).clamp(-3, 3)*scale
    jac = torch.stack([torch.autograd.grad(y[:, j].sum(), endpoint, retain_graph=k < 3)[0]
                       for k, j in enumerate([0, 3, 6, 9])], dim=1)
    result['local_hip_jacobian'] = {}
    for group, dims in GROUPS.items():
        conversion = .25 if group == 'gyro' else .05 if group == 'dq' else 1
        # Last_action remains in action units; do not mix units across groups.
        result['local_hip_jacobian'][group] = (jac[:, :, dims]*conversion).square().sum(2).mean().sqrt().item()
    result['local_hip_jacobian_units'] = 'q_target rad per gyro rad/s, gravity unit, command unit, q rad, dq rad/s, last_action unit respectively; compare same group across cases only'
    common_gyro_x = jac[:, :, 0].mean(1)*.25
    result['conditional_common_hip_gyro_x_derivative'] = {
        'mean_s': common_gyro_x.mean().item(), 'min_s': common_gyro_x.min().item(),
        'max_s': common_gyro_x.max().item(), 'positive_fraction': (common_gyro_x > 0).float().mean().item(),
        'limit': 'Partial derivative of mean hip q_target with respect to body gyroX holding other inputs fixed; not net body damping or physical loop gain'}
    return result


def main(a):
    import torch
    torch.set_num_threads(1)
    real_path = ROOT/'imgo2_deploy/logs/2026-11-17_21-10-22_7163'
    normal_path = ROOT/'imgo2_deploy/logs/height23_forward05_kd08_20261109_a'
    policy = ROOT/'imgo2_rl/logs/rsl_rl/base_move_rough/2026-11-06_21-49-38/exported/policy.pt'
    expected = '56d6d0432e54a53126375834d7ab34750f139d7057e0a583b71ea7068bc67123'
    assert digest(policy) == expected
    model = torch.jit.load(str(policy), map_location='cpu').eval()
    rr = read(real_path/'policy.csv'); pp = read(normal_path/'policy_raw.csv')
    odom = read(normal_path/'odom.csv'); map_host(pp, odom)
    real = [{'t': r['input_captured_ms']/1000, 'x': [r[f'obs_{i:02}'] for i in range(45)],
             'act': [r[f'act_{i:02}'] for i in range(12)]} for r in rr]
    normal = [{'t': r['t_s'], 'x': [r[f'obs_{i}'] for i in range(45)],
               'act': [r[f'action_clipped_{i}'] for i in range(12)]} for r in pp]
    validations = {}
    with torch.no_grad():
        for label, rows in [('real', real), ('normal', normal)]:
            x = torch.tensor([r['x'] for r in rows], dtype=torch.float32)
            diff = model(x).clamp(-3, 3)-torch.tensor([r['act'] for r in rows])
            assert diff.abs().max().item() < 2e-5
            last = [abs(b['x'][33+j]-r['act'][j]) for r, b in zip(rows, rows[1:]) for j in range(12)]
            validations[label] = {'rows': len(rows), 'clipped_action_max_abs_error': diff.abs().max().item(),
                'clipped_action_rmse': diff.square().mean().sqrt().item(),
                'last_action_vs_previous_logged_clipped_action_max_abs': max(last)}
    # Windows fixed by command/coverage, before looking at output attribution.
    windows = {'real_A': (real, 71.9, 74.7), 'real_B': (real, 76.8, 79.6),
               'real_C': (real, 97.4, 100.2), 'normal_A': (normal, 16.5, 19.3),
               'normal_B': (normal, 19.5, 22.3)}
    report = {'policy_sha256': expected, 'validation': validations, 'windows': {},
              'attribution': {}, 'limits': [
                  'Real support/rope handling intervals unknown; no autonomous-versus-assisted labels.',
                  'Independent normal/real gait phases: compare amplitude/spectra, never pointwise RMSE.',
                  'Real windows have small nonzero lateral command; actual physical speed not measured.',
                  'Attribution describes policy output changes along observed input transitions; correlated inputs, cancellation and model feedback prevent causal identification.',
                  'Integrated path may leave the physical observation manifold; endpoints are actual captures, integration is only a mathematical decomposition.',
                  'Covariance shares can be negative; contribution RMS values must not be added as percentages.',
                  'Approximately50Hz actor samples,100Hz interpolation does not restore bandwidth above actor Nyquist.',
                  'Original deployed file hash not saved; numerical agreement provides matching evidence.',
                  'No new simulation, physical loop interventions, hardware or training.'
              ]}
    traces, spectra = {}, {}
    for name, (rows, lo, hi) in windows.items():
        result, data, bins, selected = describe(rows, lo, hi)
        report['windows'][name] = result; traces[name] = data; spectra[name] = bins
        report['attribution'][name] = attribution(model, selected, torch)
        print(name, 'done', flush=True)
    report['ratios_vs_normal_A'] = {}
    for name in ['real_A', 'real_B', 'real_C']:
        report['ratios_vs_normal_A'][name] = [{
            'index': j, 'label': LABELS[j], 'std_ratio': r['std']/n['std'] if n['std'] > 1e-9 else None,
            'band_4_10_ratio': r['band_4_10_rms']/n['band_4_10_rms'] if n['band_4_10_rms'] > 1e-9 else None}
            for j, (r, n) in enumerate(zip(report['windows'][name]['channels'], report['windows']['normal_A']['channels']))]
    dt = .02; alpha = 1-math.exp(-2*math.pi*3*dt); frequency = 6.8
    theta = 2*math.pi*frequency*dt
    h = alpha/(1-(1-alpha)*complex(math.cos(-theta), math.sin(-theta)))
    report['historical_3hz_ema_at_6p8hz'] = {'nominal_dt_s': dt, 'gain': abs(h),
        'phase_deg': math.degrees(math.atan2(h.imag, h.real)),
        'limit': 'Analytical response of existing causal EMA at constant20ms; not measured physical delay, not a new filter replay'}
    files = [real_path/'policy.csv', real_path/'meta.json', normal_path/'policy_raw.csv',
             normal_path/'odom.csv', normal_path/'metadata.json', normal_path/'collection.json', policy, Path(__file__)]
    files += [ROOT/p for p in [
        'imgo2_deploy/ros2_ws/src/imgo2_deploy/src/rl_sim.cpp',
        'imgo2_deploy/ros2_ws/src/imgo2_deploy/library/core/rl_sdk/rl_sdk.cpp',
        'imgo2_deploy/ros2_ws/src/imgo2_deploy/library/core/rl_sdk/rl_sdk.hpp',
        'imgo2_deploy/ros2_ws/src/imgo2_deploy/fsm_robot/fsm_imgo2.hpp',
        'imgo2_deploy/ros2_ws/scripts/prepare_obs_replay.py',
        'imgo2_rl/logs/rsl_rl/base_move_rough/2026-11-06_21-49-38/params/agent.yaml',
        'imgo2_rl/logs/rsl_rl/base_move_rough/2026-11-06_21-49-38/params/env.yaml']]
    report['sha256'] = {str(p.relative_to(ROOT)): digest(p) for p in files}
    a.output.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    with a.output.with_suffix('.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['case', 'index', 'label', 'mean', 'std', 'rms',
            'peak_to_peak', 'peak_1_20_hz', 'peak_sinusoid_amplitude', 'resolution_hz', 'band_4_10_rms'])
        writer.writeheader()
        for name, case in report['windows'].items():
            for row in case['channels']: writer.writerow({'case': name, **row})
    if a.plot:
        plot(a.output, report, traces, spectra)
    print('Saved', a.output, flush=True)


def plot(output, report, traces, spectra):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'normal_A': '#228833', 'real_B': '#0077bb'}
    def lines(ax, index, physical_factor=1):
        for case, color in colors.items():
            v = traces[case][index]
            ax.plot([j*.01 for j in range(len(v))], [x*physical_factor for x in v],
                    color=color, label=case, lw=1)
        ax.set_title(f'obs[{index:02}] {LABELS[index]}', fontsize=10)
        ax.set_xlabel('Within-window seconds; independent gait phase', fontsize=8)
        ax.grid(alpha=.2)
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), constrained_layout=True)
    for j in range(3):
        lines(axes[j, 0], j, 4); axes[j, 0].set_ylabel('rad/s')
        lines(axes[j, 1], j+3); axes[j, 1].set_ylabel('unit gravity')
    axes[0, 0].legend(); fig.suptitle('Same56d6 policy: real actor input vs normal Gazebo actor input')
    fig.savefig(str(output)+'_body.png', dpi=160); plt.close(fig)
    for kind, start, factor, units in [('q', 9, 1, 'relative rad'), ('dq', 21, 20, 'rad/s'),
                                        ('last_action', 33, 1, 'clipped action')]:
        fig, axes = plt.subplots(4, 3, figsize=(14, 11), constrained_layout=True)
        for j, ax in enumerate(axes.flat): lines(ax, start+j, factor); ax.set_ylabel(units)
        axes[0, 0].legend(); fig.suptitle(f'{kind}: real_B vs normal_A, all12 joints')
        fig.savefig(str(output)+'_'+kind+'.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ax, index in [(axes[0, 0], 0), (axes[0, 1], 4)]:
        for case, color in colors.items():
            bins = spectra[case][index]
            ax.plot([x[0] for x in bins], [x[1]*(4 if index == 0 else 1) for x in bins],
                    label=case, color=color)
        ax.set(title=LABELS[index]+' spectrum', xlabel='Hz', ylabel='Hann sinusoid amplitude', xlim=(1, 20))
        ax.grid(alpha=.2); ax.legend()
    groups = list(GROUPS)
    ax = axes[1, 0]
    for i, case in enumerate(['normal_A', 'real_A', 'real_B', 'real_C']):
        values = report['attribution'][case]['modes']['hip_pooled']['groups']
        ax.bar([j+(i-1.5)*.18 for j in range(6)], [values[g]['contribution_rms_rad'] for g in groups],
               width=.18, label=case)
    ax.set_xticks(range(6), groups); ax.set(title='Hip target step decomposition: observed adjacent inputs', ylabel='Contribution RMS (rad)')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=.2)
    ax = axes[1, 1]
    for i, case in enumerate(['normal_A', 'real_A', 'real_B', 'real_C']):
        values = report['attribution'][case]['modes']['hip_pooled']['groups']
        ax.bar([j+(i-1.5)*.18 for j in range(6)], [values[g]['signed_covariance_share']*100 for g in groups],
               width=.18, label=case)
    ax.set_xticks(range(6), groups); ax.set(title='Signed association with actual hip target step', ylabel='Signed covariance share (%)')
    ax.axhline(0, color='black', lw=.6); ax.grid(axis='y', alpha=.2)
    fig.suptitle('Input/output coupling, not physical causal shares; cancellation is allowed')
    fig.savefig(str(output)+'_attribution.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', action='store_true')
    main(parser.parse_args())
