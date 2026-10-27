#!/usr/bin/env python3
"""Plot archived MuJoCo comparison metrics; only this presentation step needs matplotlib."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Output prefix')
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    document = json.loads(args.metrics.read_text())
    values = {(r['policy'], r['case']): r for r in document['results']}
    colors = {'policy.pt': '#df6d30', 'policy_now.pt': '#277da8'}
    labels = {'policy.pt': 'Latest policy.pt', 'policy_now.pt': 'Original policy_now.pt'}
    policies = list(colors)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 150})
    cases = ['backward05', 'backward03', 'stand', 'forward01', 'forward03', 'forward05', 'forward08']
    speeds = [-.5, -.3, 0, .1, .3, .5, .8]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), layout='constrained')
    fig.suptitle('MuJoCo: latest vs original PPO | identical model, PD and feedback delay', fontsize=15)
    plots = [(axes[0, 0], 'vx_mean', 1, 'Forward / reverse tracking', 'Actual body vx (m/s)'),
             (axes[0, 1], 'height_mean_m', 100, 'Body height', 'Base origin above floor (cm)'),
             (axes[0, 2], 'stride_frequency_hz', 1, 'Gait cadence', 'Same-foot cycles / second (Hz)'),
             (axes[1, 0], 'gyro_xy_rms_rads', 1, 'Body roll / pitch motion', 'Angular velocity RMS (rad/s)')]
    for ax, key, factor, title, ylabel in plots:
        for policy in policies:
            ys = [values[policy, case][key] for case in cases]
            ax.plot(speeds, [float('nan') if y is None else y*factor for y in ys],
                    '-o', color=colors[policy], label=labels[policy], markersize=4)
        ax.set(title=title, xlabel='Command vx (m/s)', ylabel=ylabel)
        ax.grid(alpha=.22)
    axes[0, 0].plot(speeds, speeds, '--', color='#666666', alpha=.65, label='Command')
    axes[0, 0].legend(fontsize=8)
    axes[0, 2].text(.04, .05, 'No sustained stepping at 0 or +0.1', transform=axes[0, 2].transAxes, fontsize=8)
    axes[1, 0].text(.04, .91, 'Includes normal gait motion; excludes desired yaw', transform=axes[1, 0].transAxes, fontsize=8)
    ax = axes[1, 1]
    selected = ['forward05', 'backward03', 'turn_left', 'turn_right']
    names = ['Forward\n+0.5 m/s', 'Reverse\n-0.3 m/s', 'Left turn\n+0.8 rad/s', 'Right turn\n-0.8 rad/s']
    for i, policy in enumerate(policies):
        ax.bar([j+(i-.5)*.36 for j in range(4)],
               [values[policy, c]['swing_clearance_peak_mean_m']*100 for c in selected],
               width=.34, color=colors[policy], label=labels[policy])
    ax.set(title='Foot lift depends on direction', ylabel='Mean peak swing clearance (cm)', xticks=range(4), xticklabels=names)
    ax.grid(axis='y', alpha=.22)
    ax = axes[1, 2]
    for i, policy in enumerate(policies):
        r = values[policy, 'forward05']
        ax.bar([j+(i-.5)*.36 for j in range(2)],
               [r['front_hip_outward_mean_deg'], r['rear_hip_outward_mean_deg']],
               width=.34, color=colors[policy])
    ax.axhline(0, color='#666666', lw=.7)
    ax.set(title='Hip posture at +0.5 m/s', ylabel='Outward angle (degrees; negative = inward)',
           xticks=[0, 1], xticklabels=['Front pair', 'Rear pair'])
    ax.grid(axis='y', alpha=.22)
    fig.supxlabel('24 independent 20 s trials; statistics from final 16 s; MuJoCo 3.2.7, policy 50 Hz, motor feedback +10 ms', fontsize=9)
    for suffix in ('.png', '.pdf'):
        fig.savefig(str(args.output)+suffix)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5), layout='constrained')
    for ax, policy in zip(axes, policies):
        r = values[policy, 'forward05']
        ax.add_patch(Rectangle((-.067, -.1575), .134, .315, fc='#d9e0e4', ec='#667788', lw=1.5))
        for leg, position in r['feet_body_mean_m'].items():
            x, y, _ = position
            hip_x = .1575 if leg.startswith('F') else -.1575
            hip_y = .067 if leg.endswith('L') else -.067
            ax.plot([hip_y, y], [hip_x, x], color=colors[policy], lw=3)
            ax.scatter([y], [x], color=colors[policy], s=65, zorder=3)
            ax.annotate(leg, (y, x), xytext=(6, 6), textcoords='offset points')
        ax.annotate('Forward', xy=(0, .30), xytext=(0, .24), ha='center', arrowprops={'arrowstyle': '->'})
        ax.set(title=labels[policy], xlabel='Robot left / right (m)', ylabel='Robot forward / rearward (m)',
               xlim=(.30, -.30), ylim=(-.30, .34), aspect='equal')
        ax.grid(alpha=.2)
        ax.text(.03, .02, f"Front width {100*r['front_feet_width_mean_m']:.1f} cm\nRear width {100*r['rear_feet_width_mean_m']:.1f} cm\nBody height {100*r['height_mean_m']:.1f} cm", transform=ax.transAxes, fontsize=10)
    fig.suptitle('Mean foot placement at +0.5 m/s | top view in body frame', fontsize=14)
    fig.supxlabel('Lines connect hip anchors to mean foot collision centers; not a rendered mesh or instantaneous leg geometry.', fontsize=9)
    for suffix in ('.png', '.pdf'):
        fig.savefig(str(args.output)+'_posture'+suffix)
    plt.close(fig)


if __name__ == '__main__':
    main()
