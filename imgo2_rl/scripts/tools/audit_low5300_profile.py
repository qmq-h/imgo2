#!/usr/bin/env python3
"""Audit saved 5300/control/candidate configurations without importing Isaac or Torch."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from audit_ppo_domain_randomization import block, saved, yaml_value

ROOT = Path(__file__).resolve().parents[3]


def at(lines, *parts):
    current = lines
    for depth, part in enumerate(parts[:-1]):
        current = block(current, part, 2 * depth)
    return yaml_value(current, parts[-1], 2 * (len(parts) - 1))


def snapshot(run):
    path = run / 'params/env.yaml'
    lines = path.read_text().splitlines()
    rewards = block(lines, 'rewards', 0)
    weights = {}
    for line in rewards[1:]:
        if line.startswith('  ') and not line.startswith('   ') and ':' in line:
            name = line.strip().split(':')[0]
            term = block(rewards, name, 2)
            weights[name] = None if term[0].split(':', 1)[1].strip() == 'null' else yaml_value(term, 'weight', 4)
    paths = [
        'scene.terrain.terrain_type', 'scene.terrain.terrain_generator',
        'curriculum.terrain_levels', 'curriculum.command_levels_lin_vel', 'curriculum.command_levels_ang_vel',
        'commands.base_velocity.class_type', 'commands.base_velocity.heading_command',
        'commands.base_velocity.resampling_time_range', 'commands.base_velocity.rel_standing_envs',
        'commands.base_velocity.ranges.lin_vel_x', 'commands.base_velocity.ranges.lin_vel_y',
        'commands.base_velocity.ranges.ang_vel_z',
        'rewards.base_height_l2.params.target_height', 'rewards.feet_air_time.params.threshold',
        'rewards.feet_height_body.params.target_height', 'rewards.feet_height_body.params.tanh_mult',
        'rewards.contact_forces.params.threshold', 'terminations.illegal_contact',
        'observations.policy.base_ang_vel.scale', 'observations.policy.joint_pos.scale',
        'observations.policy.joint_vel.scale', 'decimation', 'sim.dt',
    ]
    return {'run': str(run), 'num_envs': at(lines, 'scene', 'num_envs'), 'weights': weights,
            'fields': {p: at(lines, *p.split('.')) for p in paths},
            'actuator_and_randomization': saved(path)}


def main(args):
    baseline = snapshot(args.original)
    control = snapshot(args.control)
    candidate = snapshot(args.candidate)
    assert control['weights'] == baseline['weights'], 'Control reward weights differ from original5300'
    assert candidate['weights'] == baseline['weights'], 'Candidate reward weights differ from original5300'
    assert control['fields'] == baseline['fields'], 'Control named fields differ from original5300'
    expected = dict(baseline['fields'])
    expected['rewards.feet_height_body.params.target_height'] = -0.25
    expected['rewards.feet_air_time.params.threshold'] = 0.20
    assert candidate['fields'] == expected, 'Candidate has unexpected named-field differences'
    assert control['actuator_and_randomization'] == candidate['actuator_and_randomization'] == baseline['actuator_and_randomization']
    provenance = {}
    for label, run in [('control', args.control), ('candidate', args.candidate)]:
        provenance[label] = json.loads((run / 'warm_start.json').read_text())
        assert provenance[label]['loaded_iteration'] == 5300
        assert provenance[label]['optimizer_restored'] is False
    files = [args.original / 'model_5300.pt',
             ROOT / 'imgo2_rl/scripts/rl_lab/ppo/train_low5300.sh',
             ROOT / 'imgo2_rl/scripts/rl_lab/ppo/train.py',
             ROOT / 'imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/rough_env_cfg.py',
             ROOT / 'imgo2_deploy/ros2_ws/policy/imgo2/ppo/policy.pt']
    files += [run / 'params' / name for run in (args.original, args.control, args.candidate)
              for name in ('env.yaml', 'agent.yaml')]
    result = {'scope': 'Named-field/weight/DR equivalence plus actual two-iteration startup; not gait or hardware acceptance',
              'original': baseline, 'control': control, 'candidate': candidate,
              'intended_changes': {p: {'original': baseline['fields'][p], 'candidate': expected[p]}
                                   for p in expected if expected[p] != baseline['fields'][p]},
              'warm_start': provenance,
              'sha256': {str(p.resolve().relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              'limits': ['Original remote reward implementation was not archived; named-config equality does not prove source equality.',
                         'Local warm start resets optimizer state across a different RSL-RL runtime.',
                         'Two iterations at8 environments validate startup, not training efficacy or recovered gait.']}
    ast.parse((ROOT / 'imgo2_rl/scripts/rl_lab/ppo/train.py').read_text(encoding='utf-8-sig'))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'named_configuration_checks': 'pass', 'intended_changes': result['intended_changes']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'control', 'candidate', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())
