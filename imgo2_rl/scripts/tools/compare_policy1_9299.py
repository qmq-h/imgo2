#!/usr/bin/env python3
"""Compare existing TorchScript policies on identical captured inputs and own-feedback runs.

Inference requires PyTorch; numerical summaries use stdlib and existing audit helpers.
This does not export, install, or modify policies. Historical last_action is retained
in the common-input test, so its outputs are conditional responses, not new rollouts.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics as st
from audit_paired_obs_replay import load, sha, rms, wave, spectrum, attitude, summarize
from analyze_command_history import map_host

ROOT = Path(__file__).resolve().parents[3]
SCALE = [.125, .25, .25]*4
DEFAULT = [0., .87, -1.82]*4
GROUPS = {'hip': [0, 3, 6, 9], 'thigh': [1, 4, 7, 10], 'shank': [2, 5, 8, 11]}
INPUTS = {'gyro': list(range(3)), 'gravity': list(range(3, 6)),
          'q': list(range(9, 21)), 'dq': list(range(21, 33)),
          'last_action': list(range(33, 45))}


def joint_summary(rows, lo, hi):
    sel = [r for r in rows if lo <= r['t_s'] < hi]
    if len(sel) < 50:
        raise ValueError('Too few actor samples')
    out = {'samples': len(sel), 'dt_mean_s': st.mean(b['t_s']-a['t_s'] for a, b in zip(sel, sel[1:])), 'groups': {}}
    for group, indices in GROUPS.items():
        bands = [spectrum(wave(rows, f'target_{i}', lo, hi))[0] for i in indices]
        out['groups'][group] = {
            'target_std_rad': rms([st.pstdev(r[f'target_{i}'] for r in sel) for i in indices]),
            'target_step_rms_rad': rms([b[f'target_{i}']-a[f'target_{i}'] for a, b in zip(sel, sel[1:]) for i in indices]),
            'target_rate_rms_rad_s': rms([(b[f'target_{i}']-a[f'target_{i}'])/(b['t_s']-a['t_s']) for a, b in zip(sel, sel[1:]) for i in indices]),
            'target_band_4_10_rms_rad': rms([s['band_4_10_rms'] for s in bands]),
            'target_peak_hz_per_leg': [s['peak_1_20_hz'] for s in bands],
            'nominal_pd_rms_nm': rms([r[f'pd_{i}'] for r in sel for i in indices]),
            'raw_action_clip_fraction': st.mean(abs(r[f'raw_{i}']) > 3 for r in sel for i in indices)}
        if 'state_q_0' in sel[0]:
            out['groups'][group].update(
                position_std_rad=rms([st.pstdev(r[f'state_q_{i}'] for r in sel) for i in indices]),
                tracking_error_rms_rad=rms([r[f'target_{i}']-r[f'state_q_{i}'] for r in sel for i in indices]),
                dq_rms_rad_s=rms([r[f'state_dq_{i}'] for r in sel for i in indices]),
                feedback_tau_rms_nm=rms([r[f'state_tau_est_{i}'] for r in sel for i in indices]))
    return out


def simulate_summary(path, *, start_offset=5.0, duration=6.0):
    c = json.loads((path/'collection.json').read_text())
    m = json.loads((path/'metadata.json').read_text())
    if not c['complete'] or not m['sources_unchanged'] or not m['completed']:
        raise ValueError('Incomplete run or changed source')
    odom, imu, pp = [load(path/n) for n in ('odom.csv', 'imu.csv', 'policy_raw.csv')]
    for r in odom+imu:
        r['t_s'] = r['sim_s']; attitude(r)
    map_host(pp, odom)
    # Identical post-entry window; an 85ms sensor-recording gap occurs at +4s
    # in the current-policy run. Do not interpolate across it for the spectrum.
    if start_offset < 0 or duration <= 0 or start_offset + duration > c['hold_seconds']:
        raise ValueError('Analysis window is outside the recorded PPO hold')
    lo = c['ppo_pd_start_sim_s']+start_offset
    hi = lo+duration
    for r in pp:
        for i in range(12):
            r[f'pd_{i}'] = r[f'kp_{i}']*(r[f'target_{i}']-r[f'state_q_{i}'])-r[f'kd_{i}']*r[f'state_dq_{i}']
    core = [r for r in pp if lo <= r['t_s'] < hi]
    assert all(abs(r['cmd_vx']-.5)<1e-6 and r['cmd_vy']==0 and r['cmd_wz']==0 for r in core)
    assert all(abs(r[f'kp_{i}']-25)<1e-5 and abs(r[f'kd_{i}']-.8)<1e-5 for r in core for i in range(12))
    body = [r for r in odom if lo <= r['t_s'] < hi]
    result = {'path': str(path), 'collection': c, 'metadata': m, 'window_s': [lo, hi],
              'body': {'mean_body_vx_mps': st.mean(r['body_vx'] for r in body),
                       'body_vx_std_mps': st.pstdev(r['body_vx'] for r in body),
                       'mean_height_m': st.mean(r['z'] for r in body),
                       'min_height_m': min(r['z'] for r in body),
                       'max_tilt_deg': max(r['tilt_deg'] for r in body)},
              'imu': summarize(imu, lo, hi),
              'actor': joint_summary(pp, lo, hi),
              'hashes': {n: sha(path/n) for n in ('policy_raw.csv', 'imu.csv', 'odom.csv', 'metadata.json')}}
    runner_dir = Path(m['rl_sim_binary']).parent
    result['runner_build'] = json.loads((runner_dir/'build_metadata.json').read_text())
    result['isolated_policy_tree_sha256'] = {n: sha(runner_dir/'policy'/p) for n, p in
            [('weight', 'imgo2/ppo/policy.pt'), ('ppo_config', 'imgo2/ppo/config.yaml'), ('base_config', 'imgo2/base.yaml')]}
    return result, (pp, imu, odom, lo, hi)


def main(a):
    import torch
    torch.set_num_threads(2)
    sources = {name: ROOT/'imgo2_deploy/ros2_ws/policy/imgo2/ppo'/name for name in ('policy.pt', 'policy_1.pt')}
    if a.current_policy:
        sources['policy.pt'] = a.current_policy.resolve()
    if a.extra_policy:
        if a.extra_name in sources:
            raise ValueError('Extra policy name must be unique')
        sources[a.extra_name] = a.extra_policy.resolve()
    models = {name: torch.jit.load(str(path), map_location='cpu').eval() for name, path in sources.items()}
    rows = load(a.real/'policy.csv')
    x = torch.tensor([[r[f'obs_{i:02}'] for i in range(45)] for r in rows], dtype=torch.float32)
    scale = torch.tensor(SCALE)
    with torch.no_grad():
        ys = {name: model(x) for name, model in models.items()}
    if not all(y.shape == (len(rows), 12) and torch.isfinite(y).all() for y in ys.values()):
        raise ValueError('Invalid actor output')
    actual = torch.tensor([[r[f'act_{i:02}'] for i in range(12)] for r in rows])
    error = ys['policy.pt'].clamp(-3, 3)-actual
    result = {'files': {name: {'path': str(path), 'sha256': sha(path),
                'parameters': sum(v.numel() for v in models[name].state_dict().values()),
                'architecture': str(models[name]), 'forward_code': models[name].code} for name, path in sources.items()},
              'real_policy_sha256': sha(a.real/'policy.csv'), 'real_rows': len(rows),
              'current_log_validation': {'max_abs': error.abs().max().item(), 'rmse': error.square().mean().sqrt().item(), 'comparison': 'configured clipped action versus recorded act'},
              'shared_input': {}, 'sensitivity': {}, 'closed_loop': {},
              'method': {'common_input': 'Original real 45 inputs retained, including current-policy historical last_action. This is not a policy_1 trajectory.',
                         'nominal_pd': 'Frozen source q/dq: 25*(default+scale*clip(action,-3,3)-q)-0.8*dq. Conditional demand only; not measured policy_1 torque.',
                         'spectra': '100Hz linear interpolation, detrend+Hann peak1-20Hz; rectangular DFT RMS4-10Hz. Common-input windows2.5s (0.4Hz resolution); own-feedback windows6s (1/6Hz).',
                         'sensitivity': 'One input group at a time advances from previous real input to next; all others remain previous. RMS conditional hip target change; nonlinear terms need not sum.',
                         'jacobian': 'Mean square Frobenius norm over samples and four hip outputs. Per physical input gyro rad/s, gravity dimensionless, q rad, dq rad/s, previous joint target rad. Units cannot be compared across input types.'},
              'limitations': ['policy_1 original training checkpoint/config not established.',
                             'User confirms same real PD25/0.8 and speed commands, less shaking with policy_1, but no policy_1 real log exists for quantified comparison.',
                             'Ideal Gazebo success does not establish real motor performance or isolate a root cause.']}
    streams = {}
    with a.output.with_name(a.output.name+'_shared.csv').open('w') as f:
        fields = ['policy', 't_s']+[f'{k}_{i}' for k in ('target', 'raw', 'pd') for i in range(12)]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for name, y in ys.items():
            rr = []
            for r, act in zip(rows, y.tolist()):
                row = {'t_s': r['input_captured_ms']/1000}
                for i in range(12):
                    target = DEFAULT[i]+SCALE[i]*max(-3, min(3, act[i]))
                    row.update({f'raw_{i}': act[i], f'target_{i}': target,
                                f'pd_{i}': 25*(target-(DEFAULT[i]+r[f'obs_{9+i:02}']))-.8*r[f'obs_{21+i:02}']/.05})
                writer.writerow({'policy': name, **row}); rr.append(row)
            streams[name] = rr
            result['shared_input'][name] = {label: joint_summary(rr, lo, hi) for label, lo, hi in
                    [('early_forward', 38.7, 41.2), ('late_forward', 117.6, 120.1)]}
    indices = [i for i, r in enumerate(rows) if 38.7 <= r['input_captured_ms']/1000 < 41.2]
    for name, model in models.items():
        prev = x[[i-1 for i in indices]]; now = x[indices]
        with torch.no_grad():
            base = model(prev).clamp(-3, 3)*scale
            group_response = {}
            for group, dims in INPUTS.items():
                altered = prev.clone(); altered[:, dims] = now[:, dims]
                delta = (model(altered).clamp(-3, 3)*scale-base)[:, GROUPS['hip']]
                group_response[group] = delta.square().mean().sqrt().item()
        inputs = now.clone().requires_grad_()
        outputs = model(inputs).clamp(-3, 3)*scale
        jac = torch.stack([torch.autograd.grad(outputs[:, i].sum(), inputs, retain_graph=True)[0] for i in GROUPS['hip']], dim=1)
        gains = {}
        for group, dims in INPUTS.items():
            factor = .25 if group == 'gyro' else .05 if group == 'dq' else 1
            conversion = torch.tensor([1/SCALE[i-33] for i in dims]) if group == 'last_action' else factor
            gains[group] = ((jac[:, :, dims]*conversion).square().sum(dim=2).mean().sqrt().item())
        result['sensitivity'][name] = {'window_s': [38.7, 41.2], 'samples': len(indices),
                                     'same_group_step_hip_target_rms_rad': group_response, 'local_hip_jacobian_physical_input': gains}
    sims = {}
    if a.sim_current and a.sim_old:
        run_paths = [('policy.pt', a.sim_current), ('policy_1.pt', a.sim_old)]
        if a.sim_extra:
            if not a.extra_policy:
                raise ValueError('--sim-extra requires --extra-policy')
            run_paths.append((a.extra_name, a.sim_extra))
        for name, path in run_paths:
            result['closed_loop'][name], sims[name] = simulate_summary(path)
            if result['closed_loop'][name]['isolated_policy_tree_sha256']['weight'] != result['files'][name]['sha256']:
                raise ValueError('Simulation weight does not match compared policy')
    a.output.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
    if a.plot:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
        for name, rr in streams.items():
            selected = [r for r in rr if 38.7 <= r['t_s'] < 41.2]
            for ax, key, title in [(axes[0, 0], 'target_3', 'FR hip target (same real inputs)'),
                                   (axes[1, 0], 'target_5', 'FR shank target (same real inputs)'),
                                   (axes[2, 0], 'pd_5', 'FR shank conditional nominal PD')]:
                ax.plot([r['t_s']-38.7 for r in selected], [r[key] for r in selected], label=name)
                ax.set_title(title); ax.set_xlabel('Within-window time (s)')
            _, sp = spectrum(wave(rr, 'target_3', 38.7, 41.2))
            axes[0, 1].plot([p[0] for p in sp], [p[1] for p in sp], label=name)
        axes[0, 0].set_ylabel('rad'); axes[1, 0].set_ylabel('rad'); axes[2, 0].set_ylabel('Nm; frozen real q/dq')
        axes[0, 1].set(title='FR hip target spectrum: same real inputs', xlabel='Hz', ylabel='Hann amplitude (rad)', xlim=(1, 15))
        labels = list(INPUTS)
        width = .8/len(result['sensitivity'])
        for n, (name, d) in enumerate(result['sensitivity'].items()):
            axes[1, 1].bar([i+(n-(len(result['sensitivity'])-1)/2)*width for i in range(len(labels))], [d['same_group_step_hip_target_rms_rad'][k] for k in labels], width=width, label=name)
        axes[1, 1].set(xticks=range(len(labels)), xticklabels=labels, ylabel='Hip target conditional step RMS (rad)', title='Identical observed input-group changes')
        axes[2, 1].axis('off')
        axes[2, 1].text(.02, .95, 'Common-input test retains the current policy last_action.\nThese are conditional outputs, not policy_1 motion.\nPD demand uses frozen measured q/dq.\nNo root-cause claim from this test alone.', va='top', transform=axes[2, 1].transAxes)
        for ax in axes.flat:
            if ax.axison: ax.grid(alpha=.25); ax.legend(fontsize=8)
        fig.savefig(a.output.with_name(a.output.name+'_shared.png'), dpi=150); plt.close(fig)
        if sims:
            fig, axes = plt.subplots(3, 2, figsize=(13, 9), constrained_layout=True)
            for name, (pp, imu, odom, lo, hi) in sims.items():
                for ax, rr, key in [(axes[0, 0], imu, 'gyro_x'), (axes[1, 0], imu, 'roll_deg'), (axes[2, 0], pp, 'target_3')]:
                    ss = [r for r in rr if lo <= r['t_s'] < lo+2.5]
                    ax.plot([r['t_s']-lo for r in ss], [r[key] for r in ss], label=name)
                for ax, rr, key in [(axes[0, 1], imu, 'gyro_x'), (axes[1, 1], pp, 'target_3')]:
                    _, sp = spectrum(wave(rr, key, lo, hi)); ax.plot([p[0] for p in sp], [p[1] for p in sp], label=name)
                ss = [r for r in odom if lo <= r['t_s'] < hi]
                axes[2, 1].plot([r['t_s']-lo for r in ss], [r['body_vx'] for r in ss], label=name)
            for ax, title, ylabel in zip(axes.flat, ['Own-feedback roll rate', 'Own-feedback roll-rate spectrum', 'Own-feedback roll', 'Own-feedback FR hip target spectrum', 'Own-feedback FR hip target', 'Own-feedback body forward speed'], ['rad/s', 'Hann amplitude (rad/s)', 'deg', 'Hann amplitude (rad)', 'rad', 'm/s']):
                ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(alpha=.25); ax.legend(fontsize=8)
            for ax in (axes[0, 1], axes[1, 1]): ax.set(xlim=(1, 15), xlabel='Hz')
            for ax in (axes[0, 0], axes[1, 0], axes[2, 0], axes[2, 1]): ax.set_xlabel('Within-window time (s); independent phase')
            fig.savefig(a.output.with_name(a.output.name+'_closedloop.png'), dpi=150); plt.close(fig)
            fig, axes = plt.subplots(3, 2, figsize=(13, 9), constrained_layout=True)
            for n, (name, (pp, imu, odom, lo, hi)) in enumerate(sims.items()):
                ss = [r for r in pp if lo <= r['t_s'] < lo+2.5]
                tt = [r['t_s']-lo for r in ss]
                for row, (j, joint) in enumerate([(3, 'FR hip'), (4, 'FR thigh'), (5, 'FR shank')]):
                    color = f'C{n}'
                    axes[row, 0].plot(tt, [r[f'target_{j}'] for r in ss], color=color, label=name+' target')
                    axes[row, 0].plot(tt, [r[f'state_q_{j}'] for r in ss], '--', color=color, label=name+' measured')
                    axes[row, 1].plot(tt, [r[f'pd_{j}'] for r in ss], color=color, label=name+' nominal PD')
                    axes[row, 0].set(title=joint, ylabel='rad')
                    axes[row, 1].set(title=joint+' nominal PD', ylabel='Nm')
            for ax in axes.flat:
                ax.grid(alpha=.25); ax.legend(fontsize=8); ax.set_xlabel('Within-window time (s); independent phase')
            fig.savefig(a.output.with_name(a.output.name+'_joints.png'), dpi=150); plt.close(fig)
    print(json.dumps({'validation': result['current_log_validation'], 'shared': result['shared_input'], 'sensitivity': result['sensitivity'],
                      'closed_loop': {k: {'body': v['body'], 'actor': v['actor']} for k, v in result['closed_loop'].items()}}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--real', type=Path, default=ROOT/'imgo2_deploy/logs/2026-11-09_19-41-59_5639')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sim-current', type=Path)
    p.add_argument('--sim-old', type=Path)
    p.add_argument('--extra-policy', type=Path, help='Existing formally exported candidate; no export or installation')
    p.add_argument('--current-policy', type=Path, help='Preserved current-policy snapshot when production file is unavailable')
    p.add_argument('--extra-name', default='latest_trained')
    p.add_argument('--sim-extra', type=Path)
    p.add_argument('--plot', action='store_true')
    main(p.parse_args())
