#!/usr/bin/env python3
"""Plot an archived policy_1/current PPO low-speed comparison (optional matplotlib)."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Output filename prefix')
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    document = json.loads(args.metrics.read_text())
    results = {(r['policy'], r['case']): r for r in document['results']}
    policies = [('policy_1.pt', '#277da8', 'Reference policy_1.pt'),
                ('policy.pt', '#df6d30', 'Current policy.pt (4999)')]
    cases = ['backward03', 'backward02', 'backward01', 'stand',
             'forward01', 'forward02', 'forward03', 'forward05']
    speeds = [-.3, -.2, -.1, 0., .1, .2, .3, .5]
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 150})
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), layout='constrained')
    fig.suptitle('PPO low-speed comparison | same model, PD, feedback and commands', fontsize=15)
    plots = [(axes[0, 0], 'vx_mean', 1., 'Forward / reverse tracking', 'Actual body vx (m/s)'),
             (axes[0, 1], 'stride_frequency_hz', 1., 'Gait cadence', 'Same-foot cycles / second (Hz)'),
             (axes[0, 2], 'height_mean_m', 100., 'Body height', 'Base height (cm)'),
             (axes[1, 0], 'gyro_xy_rms_rads', 1., 'Roll / pitch angular motion', 'Gyro xy RMS (rad/s)')]
    for ax, key, factor, title, ylabel in plots:
        for policy, color, label in policies:
            ys = [results[policy, case][key] for case in cases]
            ax.plot(speeds, [float('nan') if y is None else y*factor for y in ys],
                    '-o', color=color, label=label, markersize=4)
        ax.set(title=title, xlabel='Command vx (m/s)', ylabel=ylabel)
        ax.grid(alpha=.22)
    axes[0, 0].plot(speeds, speeds, '--', color='#666666', alpha=.65, label='Command')
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].text(.03, .04, 'Missing points: no sustained stepping detected',
                    transform=axes[0, 1].transAxes, fontsize=8)
    selected = ['forward03', 'forward05', 'backward03', 'turn_left03', 'turn_right03']
    labels = ['Forward\n+0.3', 'Forward\n+0.5', 'Reverse\n-0.3', 'Turn left\n+0.3 rad/s', 'Turn right\n-0.3 rad/s']
    ax = axes[1, 1]
    for i, (policy, color, label) in enumerate(policies):
        ax.bar([j+(i-.5)*.36 for j in range(len(selected))],
               [results[policy, c]['swing_clearance_peak_mean_m']*100 for c in selected],
               width=.34, color=color, label=label)
    ax.set(title='Mean peak swing clearance', ylabel='Foot sphere bottom clearance (cm)',
           xticks=range(len(selected)), xticklabels=labels)
    ax.tick_params(axis='x', labelsize=8)
    ax.grid(axis='y', alpha=.22)
    ax = axes[1, 2]
    for policy, color, label in policies:
        ax.plot([-.3, .3], [results[policy, c]['wz_mean'] for c in ['turn_right03', 'turn_left03']],
                '-o', color=color, label=label)
    ax.plot([-.3, .3], [-.3, .3], '--', color='#666666', alpha=.65)
    ax.set(title='Low-rate yaw tracking', xlabel='Command yaw rate (rad/s)', ylabel='Actual yaw rate (rad/s)')
    ax.grid(alpha=.22)
    fig.supxlabel('24 independent 20 s cases; final 16 s statistics; policy 50 Hz; motor feedback +10 ms.\n'
                  'Fixed simulation-time benchmark; not a GUI/FSM handoff test.', fontsize=9)
    for suffix in ('.png', '.pdf'):
        fig.savefig(str(args.output)+suffix)
    plt.close(fig)


if __name__ == '__main__':
    main()
