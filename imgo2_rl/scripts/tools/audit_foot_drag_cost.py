#!/usr/bin/env python3
"""Score a proposed flat-ground moving-foot clearance cost on existing CSVs.

Diagnostic only: does not modify rewards or predict training outcomes. Requires
MuJoCo benchmark clearance (collision sphere bottom) and world XY foot speed.
"""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--height', type=float, default=.03)
    parser.add_argument('--speed-scale', type=float, default=.2)
    parser.add_argument('--weight', type=float, default=.1)
    parser.add_argument('--skip', type=float, default=4.)
    args = parser.parse_args()
    if args.height <= 0 or args.speed_scale <= 0 or args.weight < 0:
        parser.error('height/speed-scale must be positive and weight nonnegative')
    metadata = json.loads(args.run.read_text())
    results = []
    for case in metadata['cases']:
        path = Path(case['csv'])
        if not path.exists():
            path = args.run.parent / path.name
        values = []
        with path.open() as handle:
            for row in csv.DictReader(handle):
                if float(row['t_s']) < args.skip:
                    continue
                enabled = math.sqrt(sum(float(row[k])**2 for k in
                                       ('command_vx', 'command_vy', 'command_wz'))) > .1
                value = 0.
                for leg in ('FL', 'FR', 'RL', 'RR'):
                    deficit = min(1., max(0., (args.height - float(row[leg+'_clearance_m'])) / args.height))
                    value += math.tanh(float(row[leg+'_slip_mps']) / args.speed_scale) * deficit**2
                values.append(-args.weight * value if enabled else 0.)
        results.append(dict(policy=case['policy'], case=case['case'],
                            samples=len(values), mean_cost_rate=statistics.mean(values)))
    args.output.write_text(json.dumps(dict(
        scope='Counterfactual flat-ground cost on unchanged trajectories; not a training result',
        parameters=vars(args) | {'run': str(args.run), 'output': str(args.output)},
        formula='-weight*sum(tanh(world_foot_xy_speed/speed_scale)*clip((height-clearance)/height,0,1)^2); command gate only',
        results=results), indent=2) + '\n')
    for row in results:
        if row['case'] in ('backward03', 'left03', 'turn_left03'):
            print(row['policy'], row['case'], round(row['mean_cost_rate'], 5))


if __name__ == '__main__':
    main()
