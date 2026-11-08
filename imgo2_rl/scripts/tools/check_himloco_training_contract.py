#!/usr/bin/env python3
"""Reproduce current HIM training contract problems on CPU without Isaac.

Diagnostic only: reports detected defects rather than changing the implementation.
Requires torch in the selected interpreter.
"""
import ast
import json
from pathlib import Path
from types import SimpleNamespace as NS
ROOT=Path(__file__).resolve().parents[3]
LAB=ROOT/'imgo2_rl/scripts/rl_lab/rl_lab'


def main():
    import torch
    tree=ast.parse((LAB/'modules/him_estimator.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='HIMEstimator')
    update=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='update')
    expr=next(n.value for n in update.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='next_obs' for t in n.targets))
    critic=torch.arange(235.).reshape(1,-1)
    nextobs=eval(compile(ast.Expression(expr),'<actual next_obs expression>','eval'),{'self':NS(num_one_step_obs=45),'next_critic_obs':critic})
    assert nextobs.tolist()[0]==list(range(3,48))
    assert not torch.equal(nextobs,critic[:,:45])

    tree=ast.parse((LAB/'wrapper/himloco_vec_env_wrapper.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('step','reset')]
    mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
    scope={'torch':torch}
    exec(compile(ast.fix_missing_locations(mod),'<actual HIM wrapper methods>','exec'),scope)
    obs={'policy':torch.ones(2,45),'critic':torch.ones(2,235)}
    before={'critic':torch.full((2,235),9.)}
    env=NS(step=lambda actions:(obs,before,torch.ones(2),torch.tensor([True,False]),torch.zeros(2,dtype=torch.bool),{}),reset=lambda:(obs,{}))
    state=NS(env=env,unwrapped=NS(cfg=NS(is_finite_horizon=True)),device='cpu',num_one_step_obs=45,num_one_step_privileged_obs=235,history_length=5,privileged_history_length=0,obs_history_buf=torch.full((2,270),42.),privileged_obs_history_buf=torch.zeros(2,235),compute_termination_observations=lambda ids,values:values[ids])
    scope['step'](state,torch.zeros(2,12))
    assert torch.all(state.obs_history_buf[0,45:]==42)
    assert torch.all(state.obs_history_buf[1,45:]==42)
    history_before_reset=state.obs_history_buf.clone()
    reset_result=scope['reset'](state)
    assert reset_result[0].shape==(2,45)
    assert torch.equal(history_before_reset,state.obs_history_buf)
    print(json.dumps(dict(scope='CPU executes current estimator slicing and actual wrapper methods with a mock environment; no training or deployment modifications',detected=dict(next_target_is_critic_columns=[3,48],expected_policy_prefix_columns=[0,45],next_target_starts_with='angular velocity instead of commands',next_target_ends_with='privileged base velocity instead of last action tail',old_episode_history_survives_done=True,explicit_reset_returns_single_frame=True,explicit_reset_leaves_wrapper_history_unchanged=True),note='These reproduce current-source defects, not the unknown training provenance of deployed HIM weights.'),indent=2))


if __name__=='__main__':main()
