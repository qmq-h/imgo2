#!/usr/bin/env python3
"""CPU checks of the actual swing-only feet_height_body function; no Isaac import.

Requires torch in the selected interpreter. No simulation or training is run.
"""
import ast
import json
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[3]
REWARDS = ROOT / 'imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/mdp/rewards.py'


def main():
    import torch
    source = REWARDS.read_text()
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'feet_height_body']
    assert len(nodes) == 1
    archived = source.split('# BEGIN LEGACY feet_height_body --', 1)[1].split('\n', 1)[1].split('# END LEGACY feet_height_body', 1)[0]
    restored = '\n'.join(line[2:] if line.startswith('# ') else '' for line in archived.splitlines())
    old_tree = ast.parse(restored)
    assert len(old_tree.body) == 1 and old_tree.body[0].name == 'feet_height_body'

    def inverse(q, v):
        xyz = -q[..., 1:]
        cross = 2 * torch.cross(xyz, v, dim=-1)
        return v + q[..., :1] * cross + torch.cross(xyz, cross, dim=-1)

    class Entity:
        def __init__(self, name, **kwargs):
            self.name = name
            self.__dict__.update(kwargs)

    scope = dict(torch=torch, SceneEntityCfg=Entity, math_utils=NS(quat_apply_inverse=inverse))
    code = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), nodes[0]], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), str(REWARDS), 'exec'), scope)
    fn = scope['feet_height_body']
    names = ['FL_FOOT', 'FR_FOOT', 'RL_FOOT', 'RR_FOOT']
    # Different index spaces and a reversed sensor order are intentional.
    data = NS(body_pos_w=torch.zeros(1, 5, 3), root_pos_w=torch.zeros(1, 3),
              root_quat_w=torch.tensor([[1., 0., 0., 0.]]),
              body_lin_vel_w=torch.zeros(1, 5, 3), projected_gravity_b=torch.tensor([[0., 0., -1.]]))
    data.body_pos_w[0, 1:, 2] = -.3
    data.body_lin_vel_w[0, 1:, 0] = 1.
    class CountedNames:
        def __init__(self, body_names, **kwargs):
            self._names = body_names
            self.name_reads = 0
            self.__dict__.update(kwargs)
        @property
        def body_names(self):
            self.name_reads += 1
            return self._names
    asset = CountedNames(['base'] + names, data=data)
    sensor = CountedNames(list(reversed(names)), cfg=NS(track_air_time=True),
                          data=NS(current_air_time=torch.zeros(1, 4)))
    command = torch.tensor([[.3, 0., 0.]])
    class Scene(dict):
        sensors = {'contact_forces': sensor}
    env = NS(scene=Scene(robot=asset), command_manager=NS(get_command=lambda _: command))
    ac = Entity('robot', body_ids=[1, 2, 3, 4])
    sc = Entity('contact_forces', body_ids=slice(None))
    def run():
        return fn(env, 'base_velocity', ac, -.2, 2., sc)
    results = []
    def check(name, expected):
        actual = run()
        torch.testing.assert_close(actual, torch.tensor([expected]), rtol=1e-5, atol=1e-7)
        results.append(name)
    check('all stance contributes zero', 0.)
    sensor.data.current_air_time[0, 3] = .1  # FL at sensor index 3
    data.body_pos_w[0, 1, 2] = -.4
    expected = .04 * torch.tanh(torch.tensor(2.)).item()
    check('only swing foot contributes with reversed sensor order', expected)
    data.body_pos_w[0, 2:, 2] = -1.
    check('stance foot height does not affect cost', expected)
    data.root_lin_vel_w = torch.tensor([[99., 0., 0.]])
    check('base linear velocity does not affect world speed weight', expected)
    command[:] = torch.tensor([[0., 0., .3]])
    check('pure yaw command remains enabled', expected)
    command.zero_()
    check('zero command contributes zero', 0.)
    command[0, 0] = .3
    data.body_lin_vel_w.zero_()
    check('zero world foot speed contributes zero', 0.)
    data.body_lin_vel_w[0, 1:, 0] = 1.
    data.body_pos_w[0, 1, 2] = -.2
    check('swing foot at target has zero error', 0.)
    before_reads = (asset.name_reads, sensor.name_reads)
    for _ in range(100):
        run()
    assert (asset.name_reads, sensor.name_reads) == before_reads == (1, 1)
    results.append('100 steady-state calls perform zero body-name reads')
    sc.body_ids = [3, 2, 1, 0]
    check('changed sensor selection/order recomputes correct mapping', 0.)
    assert sensor.name_reads == 2
    sensor.cfg.track_air_time = False
    try:
        run()
    except ValueError as e:
        assert 'track_air_time' in str(e)
        results.append('disabled air-time tracking raises error')
    else:
        raise AssertionError('missing air-time guard')
    sensor.cfg.track_air_time = True
    sc.body_ids = [0, 1, 2]
    try:
        run()
    except ValueError as e:
        assert 'FL_FOOT' in str(e)
        results.append('missing selected foot raises error')
    else:
        raise AssertionError('missing foot mapping guard')
    print(json.dumps({'checks_passed': len(results), 'checks': results,
                      'legacy_restores_valid_function': True,
                      'scope': 'Actual function with CPU torch and mock scene; not Isaac runtime or training'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
