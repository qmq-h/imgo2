#!/usr/bin/env python3
"""Compare two saved training runs; stdlib TFEvent parsing, optional matplotlib."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import statistics as st
from read_tfevents import read_scalars

TAGS = ['Train/mean_reward', 'Metrics/base_velocity/error_vel_xy',
        'Metrics/base_velocity/error_vel_yaw', 'Policy/mean_noise_std',
        'Curriculum/terrain_levels', 'Episode_Reward/ang_vel_xy_l2']


def main(a):
    reports = {}; streams = {}
    for name, run in [('current_height20', a.old), ('latest_height23', a.new)]:
        events = sorted(run.glob('events.out.tfevents*'))
        if len(events) != 1:
            raise ValueError('Expected exactly one event file; do not silently merge resumes')
        path = events[0]; series = read_scalars(str(path)); streams[name] = series
        windows = {}
        for lo, hi in [(5300, 5500), (5800, 6300), (8800, 9300)]:
            windows[f'{lo}-{hi-1}'] = {}
            for tag in TAGS:
                values = [v for step, v in series[tag] if lo <= step < hi]
                if not values:
                    raise ValueError(f'Missing window {tag}')
                windows[f'{lo}-{hi-1}'][tag] = {'count': len(values), 'mean': st.mean(values)}
        reports[name] = {'run': str(run), 'event_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'step_range': [min(s for s, _ in series[TAGS[0]]), max(s for s, _ in series[TAGS[0]])],
                         'windows': windows}
    reports['env_diff'] = list(difflib.unified_diff((a.old/'params/env.yaml').read_text().splitlines(),
                                (a.new/'params/env.yaml').read_text().splitlines(), fromfile='current', tofile='latest'))
    reports['agent_bytes_equal'] = (a.old/'params/agent.yaml').read_bytes() == (a.new/'params/agent.yaml').read_bytes()
    reports['limitations'] = ['Single seed saved-run comparison; server reward implementation not fully archived.',
                              'Mean reward across different objective targets is not a common physical evaluation score.',
                              'Training metrics average random terrain/commands; not the fixed0.5m/s flat-ground test.']
    a.output.with_suffix('.json').write_text(json.dumps(reports, indent=2)+'\n')
    if a.plot:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
        for name, series in streams.items():
            for tag, ax in zip(TAGS, axes.flat):
                chunks = [[(s, v) for s, v in series[tag] if lo <= s < lo+100] for lo in range(5300, 9300, 100)]
                ax.plot([st.mean(s for s, v in c) for c in chunks if c], [st.mean(v for s, v in c) for c in chunks if c], label=name)
                ax.set_title(tag); ax.set_xlabel('Training iteration; 100-iteration means'); ax.grid(alpha=.25); ax.legend(fontsize=8)
        fig.savefig(a.output.with_suffix('.png'), dpi=150); plt.close(fig)
    print(json.dumps({k: v['windows']['8800-9299'] for k, v in reports.items() if isinstance(v, dict) and 'windows' in v}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('old', 'new', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--plot', action='store_true')
    main(p.parse_args())
