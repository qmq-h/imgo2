#!/usr/bin/env python3
"""Simulation-only adapter for wide 12-axis motor sine logs.

Keeps logged targets/gains and their send endpoint spacing. Adds an explicitly
synthetic 3 s settling baseline, and optionally a fixed world-to-base joint.
Never imports a motor driver or loads a policy.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import replay_gazebo_targets as replay

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'imgo2_rl/scripts/tools'))
from analyze_standup_replay import ground_estimate


def protected_hashes():
    paths = []
    for folder in ('imgo2_description', 'imgo2_deploy/real', 'imgo2_deploy/ros2_ws/policy'):
        paths += [p for p in (ROOT/folder).rglob('*')
                  if p.is_file() and not any(x.startswith('build') for x in p.parts)
                  and p.suffix in ('.cpp','.h','.yaml','.xacro','.urdf','.xml','.pt')]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=args.source.resolve()
    with source.open() as stream:
        rows=list(csv.DictReader(stream))
    joints=[k[6:] for k in rows[0] if k.startswith('q_set_')]
    expected=[f'{leg}_{kind}' for leg in ('FL','FR','BL','BR') for kind in ('hip','thigh','shank')]
    if joints!=expected:raise ValueError('Unexpected joint order')
    first=rows[0];send0=float(first['send_mono_ms'])
    chosen=[(float(r['send_mono_ms'])/1000-send0/1000,r) for r in rows]
    chosen=[(t,r) for t,r in chosen if t<=args.duration+1e-9]
    if chosen[-1][0]<args.duration-.02:raise ValueError('Insufficient source duration')
    if any(b[0]<=a[0] for a,b in zip(chosen,chosen[1:])):raise ValueError('Send endpoints not monotonic')
    q=[float(first['q_act_'+j]) for j in joints]
    init={'quat_w':1.,'quat_x':0.,'quat_y':0.,'quat_z':0.,**{f'q_{i:02}':v for i,v in enumerate(q)}}
    model=ROOT/'imgo2_description/urdf/imgo2_real.gazebo.urdf'
    geometry=ground_estimate(model,init)
    height=.6 if args.condition=='rack' else geometry['minimum_nonpenetrating_base_height_m']+.001
    def convert(t,r,baseline=False):
        d={'t_mono_ms':1000*t,**init}
        for i,j in enumerate(joints):
            d[f'dq_{i:02}']=0. # Initial physical speed unknown; do not inject logged dq bias.
            d[f'q_des_{i:02}']=float(r['q_set_'+j])
            d[f'dq_des_{i:02}']=float(r['dq_set_'+j])
            d[f'tau_ff_{i:02}']=0. # tau_set is computed total PD, not a feedforward command.
            d[f'kp_{i:02}']=60. if baseline and args.condition=='switch' else float(r['kp_'+j])
            d[f'kd_{i:02}']=2. if baseline and args.condition=='switch' else float(r['kd_'+j])
        return d
    warmup=3.
    converted=[convert(i*.005,first,True) for i in range(600)]
    converted += [convert(warmup+t,r) for t,r in chosen]
    adapter=out/'prepared_source';adapter.mkdir()
    with (adapter/'robot.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(converted[0]));w.writeheader();w.writerows(converted)
    meta=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        condition=args.condition,real_command_rows=len(chosen),warmup_s=warmup,
        original_send_start_ms=send0,real_duration_s=chosen[-1][0],
        total_duration_s=warmup+chosen[-1][0],joint_mapping=dict(zip(joints,replay.JOINTS)),
        initial_height_m=height,geometry=geometry,
        assumptions=['Level base; true real base orientation and support fixture pose absent.',
                     'Initial q from first actual sample; initial velocities set to zero.',
                     'Synthetic 3 s constant-target baseline; gains 60/2 only for switch condition, otherwise logged gains.',
                     'Rack uses fixed world-to-base joint at 0.6 m; original legs, gravity, contacts and controller unchanged.',
                     'Joint tau_set is total computed PD, never replayed as feedforward.',
                     'No lifting trajectory reproduced; source support must be supplied by user, not inferred from filename.'])
    before=protected_hashes()
    try:
        replay.run(SimpleNamespace(source=adapter,output=out/'run',kind='real',start=0.,
            duration=meta['total_duration_s'],height=height,targets_override=None,
            fixed_base=args.condition=='rack',gui=args.gui,gui_warmup=8.,hold_after=0.,video_encoder=None,
            plugin_library=getattr(args,'plugin_library',None),
            resistance_parameters=getattr(args,'resistance_parameters',None),
            resistance_library=getattr(args,'resistance_library',None)))
        meta['completed']=True
    finally:
        after=protected_hashes();meta['production_files_unchanged']=before==after
        meta['protected_hashes']=before
        (out/'sine_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--condition',choices=('ground','rack','switch'),required=True)
    p.add_argument('--duration',type=float,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gui',action='store_true')
    p.add_argument('--plugin-library',type=Path,help='Reuse a fingerprint-verified diagnostic plugin')
    p.add_argument('--resistance-parameters',type=Path,help='Diagnostic only: JSON mapping joint names to Fs/B/Va')
    p.add_argument('--resistance-library',type=Path,help='Diagnostic additive resistance plugin library')
    a=p.parse_args()
    if not .5<=a.duration<=15:p.error('source duration must be 0.5..15 s')
    run(a)
