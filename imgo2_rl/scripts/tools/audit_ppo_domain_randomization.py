#!/usr/bin/env python3
"""Stdlib audit of PPO source and saved-run DR; no Isaac imports or simulation.

The saved YAML reader intentionally handles only the specific scalar/tuple
fields needed here; it never constructs Python objects from YAML tags.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[3]
PACKAGE=ROOT/'imgo2_rl/source/imgo2_rl/imgo2_rl'

def literal(node):
    if isinstance(node,ast.Call):
        return {'call':ast.unparse(node.func),**{k.arg:literal(k.value) for k in node.keywords}}
    if isinstance(node,ast.Dict):return {literal(k):literal(v) for k,v in zip(node.keys,node.values)}
    if isinstance(node,(ast.Tuple,ast.List)):return [literal(v) for v in node.elts]
    try:return ast.literal_eval(node)
    except (ValueError,TypeError):return ast.unparse(node)

def block(lines,key,indent):
    start=next(i for i,l in enumerate(lines) if re.match(r'^ {'+str(indent)+r'}'+re.escape(key)+r':(?:\s|$)',l))
    end=start+1
    while end<len(lines):
        l=lines[end]
        level=len(l)-len(l.lstrip())
        if l.strip() and level<=indent and not (level==indent and l.lstrip().startswith('- ')):break
        end+=1
    return lines[start:end]

def yaml_value(lines,key,indent):
    b=block(lines,key,indent);value=b[0].split(':',1)[1].strip()
    if value=='null':return None
    if value=='true':return True
    if value=='false':return False
    if value=='!!python/tuple':return [float(l.strip()[2:]) for l in b[1:] if l.strip().startswith('- ')]
    try:return ast.literal_eval(value)
    except (ValueError,SyntaxError):return value

def saved(path):
    lines=path.read_text().splitlines();legs=block(lines,'legs',6);events=block(lines,'events',0)
    result={'actuator':{k:yaml_value(legs,k,8) for k in ('class_type','stiffness','damping','effort_limit','saturation_effort','velocity_limit','friction')},'events':{}}
    fields={'randomize_rigid_body_material':('static_friction_range','dynamic_friction_range','restitution_range','num_buckets'),
            'randomize_rigid_body_mass_base':('mass_distribution_params','operation','recompute_inertia'),
            'randomize_rigid_body_mass_others':('mass_distribution_params','operation','recompute_inertia'),
            'randomize_actuator_gains':('stiffness_distribution_params','damping_distribution_params','operation','distribution'),
            'randomize_reset_joints':('position_range','velocity_range')}
    for name,keys in fields.items():
        b=block(events,name,2);p=block(b,'params',4)
        result['events'][name]={'mode':yaml_value(b,'mode',4),'params':{k:yaml_value(p,k,6) for k in keys}}
    b=block(events,'randomize_com_positions',2);p=block(block(b,'params',4),'com_range',6)
    result['events']['randomize_com_positions']={'mode':yaml_value(b,'mode',4),'params':{k:yaml_value(p,k,8) for k in ('x','y','z')}}
    for name in ('randomize_apply_external_force_torque','randomize_push_robot'):
        result['events'][name]=yaml_value(events,name,2)
    result['dt_s']=yaml_value(block(lines,'sim',0),'dt',2)
    result['decimation']=yaml_value(lines,'decimation',0)
    result['policy_period_s']=result['dt_s']*result['decimation']
    return result

def main(a):
    cfg=PACKAGE/'tasks/manager_based/locomotion/velocity/velocity_env_cfg.py'
    asset=PACKAGE/'assets/imgo2.py';rough=PACKAGE/'tasks/manager_based/locomotion/velocity/base_move/rough_env_cfg.py'
    flat=rough.with_name('flat_env_cfg.py')
    tree=ast.parse(cfg.read_text(encoding='utf-8-sig'));eventclass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EventCfg')
    events={n.targets[0].id:literal(n.value) for n in eventclass.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call)}
    ac=next(n for n in ast.walk(ast.parse(asset.read_text(encoding='utf-8-sig'))) if isinstance(n,ast.Call) and ast.unparse(n.func)=='DCMotorCfg')
    actuator=literal(ac)
    overrides={}
    for n in ast.walk(ast.parse(rough.read_text(encoding='utf-8-sig'))):
        if isinstance(n,ast.Assign):
            for target in n.targets:
                if ast.unparse(target).startswith('self.events.'):
                    overrides[ast.unparse(target)]=literal(n.value)
    assert overrides['self.events.randomize_apply_external_force_torque'] is None
    assert overrides['self.events.randomize_push_robot'] is None
    assert not any('randomize_actuator_gains' in k for k in overrides)
    assert 'self.events.' not in flat.read_text(encoding='utf-8-sig')
    g=events['randomize_actuator_gains'];gp=g['params']
    derived={'kp_range':[actuator['stiffness']*v for v in gp['stiffness_distribution_params']],
             'kd_range':[actuator['damping']*v for v in gp['damping_distribution_params']],
             'range_contains_real_25_0_8':True,
             'multiplier_mean':sum(gp['stiffness_distribution_params'])/2,
             'uniform_fraction_below_nominal':(1-gp['stiffness_distribution_params'][0])/(gp['stiffness_distribution_params'][1]-gp['stiffness_distribution_params'][0])}
    assert derived['kp_range'][0]<=25<=derived['kp_range'][1]
    assert derived['kd_range'][0]<=.8<=derived['kd_range'][1]
    snapshots={}
    paths=[cfg,asset,rough,flat]
    for run in a.saved_run:
        run=run.resolve()
        p=run/'params/env.yaml';s=saved(p);paths.append(p)
        assert s['actuator']['stiffness']==actuator['stiffness'] and s['actuator']['damping']==actuator['damping']
        for k,v in actuator.items():
            if k in s['actuator']:assert s['actuator'][k]==v,k
        for key in ('stiffness_distribution_params','damping_distribution_params','operation','distribution'):
            assert s['events']['randomize_actuator_gains']['params'][key]==gp[key]
        assert s['events']['randomize_actuator_gains']['mode']==g['mode']
        for name,event in s['events'].items():
            if event is None:continue
            assert event['mode']==events[name]['mode'],name
            for key,value in event['params'].items():
                source_value=events[name]['params']['com_range'][key] if name=='randomize_com_positions' else events[name]['params'][key]
                assert value==source_value,(name,key)
        assert s['events']['randomize_push_robot'] is None and s['events']['randomize_apply_external_force_torque'] is None
        snapshots[str(run)]=s
    result={'scope':'Current base_move PPO rough/flat and selected saved configurations; not AMP/HIM or runtime DR rollout validation',
            'source_events':events,'rough_event_overrides':overrides,'actuator':actuator,'derived':derived,'saved_runs':snapshots,
            'unmodeled_or_not_randomized':['effort_limit / saturation_effort / velocity_limit variability',
                'command/application latency distribution, bus phase differences','torque rise-time / actuator history state',
                'joint friction/armature variability, backlash, transmission compliance',
                'contact compliance stiffness/damping variability','mechanical zero-offset bias (reset_joints_by_scale is initial state, not calibration bias)'],
            'limits':['Parameter support does not establish a policy stability guarantee or a tracking-error response envelope.',
                'DCMotor includes a fixed velocity-dependent torque cap; it is not an infinite-torque position servo.',
                'Current installed IsaacLab implementation is local evidence, not a snapshot of the original remote training runtime.',
                'A Gazebo tracking RMS is not a sampled IsaacLab parameter; torque/RMS ratios do not identify motor-strength multipliers.'],
            'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    if a.isaaclab_source:
        pd=a.isaaclab_source/'actuators/actuator_pd.py';ev=a.isaaclab_source/'envs/mdp/events.py'
        art=a.isaaclab_source/'assets/articulation/articulation.py'
        pdtext=pd.read_text();evtext=ev.read_text()
        arttext=art.read_text()
        assert 'actuator.stiffness[env_ids] = stiffness' in evtext and 'actuator.damping[env_ids] = damping' in evtext
        assert 'self._data.default_joint_stiffness[:, actuator.joint_indices] = actuator.stiffness' in arttext
        assert 'class DCMotor(IdealPDActuator)' in pdtext
        assert 'torque_speed_top = self._saturation_effort * (1.0 - self._joint_vel / self.velocity_limit)' in pdtext
        result['local_runtime_source_review']={'source_root':str(a.isaaclab_source.resolve()),
            'explicit_actuator_gain_buffer_assignment_present':True,'dc_motor_speed_dependent_cap_present':True,
            'default_gain_copied_from_explicit_actuator':True,
            'mode':'Static source review of currently installed local version; no simulator started; not the original remote runtime snapshot',
            'sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in (pd,ev,art)}}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'actuator':actuator,'derived':derived,'saved_runs':list(snapshots)},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--saved-run',type=Path,action='append',default=[]);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--isaaclab-source',type=Path,help='Optional local isaaclab package source for static implementation review')
    main(p.parse_args())
