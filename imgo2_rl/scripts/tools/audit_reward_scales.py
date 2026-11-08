#!/usr/bin/env python3
"""Read saved weights and same-window episode contributions without importing Isaac/YAML.

The recovered feature is normalized over maximum episode duration, including all
gates and episode weighting. It is not an unconditional physical-state average.
"""
import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

from read_tfevents import read_scalars


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.start > args.end:
        parser.error('start must not exceed end')
    config = args.run / 'params/env.yaml'
    text = config.read_text()
    rewards = text.split('\nrewards:\n', 1)[1].split('\nterminations:', 1)[0]
    weights = {}
    term = None
    for line in rewards.splitlines():
        match = re.fullmatch(r'  (\w+):\s*', line)
        if match:
            term = match[1]
        match = re.fullmatch(r'    weight: (\S+)', line)
        if match and term:
            weights[term] = float(match[1])
    series = {}
    events = sorted(args.run.glob('events.out.tfevents.*'))
    for event in events:
        for tag, values in read_scalars(str(event)).items():
            series.setdefault(tag, {}).update(values)
    rows = []
    for name, weight in weights.items():
        tag = 'Episode_Reward/' + name
        points = sorted((s, v) for s, v in series.get(tag, {}).items()
                        if args.start <= s <= args.end)
        if not points:
            continue
        mean = statistics.mean(v for _, v in points)
        rows.append(dict(name=name, weight=weight, count=len(points),
                         first_step=points[0][0], last_step=points[-1][0],
                         mean_contribution=mean,
                         at_end=series[tag].get(args.end),
                         effective_normalized_feature=mean / weight if weight else None))
    positive = sum(row['mean_contribution'] for row in rows if row['mean_contribution'] > 0)
    negative = -sum(row['mean_contribution'] for row in rows if row['mean_contribution'] < 0)
    result = dict(run=str(args.run.resolve()), window=[args.start, args.end],
                  config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                  events=[str(p.resolve()) for p in events],
                  interpretation='episode contribution = integral(weight*feature*dt)/max_episode_seconds; feature includes gates',
                  positive_sum=positive, negative_magnitude_sum=negative, rows=rows,
                  other_scalars={tag: statistics.mean(vals) for tag, items in series.items()
                                 if not tag.startswith('Episode_Reward/') and not tag.endswith('/time')
                                 and (vals := [v for s, v in items.items() if args.start <= s <= args.end])})
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
