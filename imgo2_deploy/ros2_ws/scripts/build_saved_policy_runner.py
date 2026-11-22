#!/usr/bin/env python3
"""Relink a diagnostic rl_sim with an isolated, saved policy tree; stdlib only.

Uses existing local compile_commands/objects. Never modifies installed binaries,
production policy/config, or normal build objects. This does not export a policy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import math
import re
import shlex
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[3]

def build(saved, output, kd=None, kp=None, hip_dq_gain=None):
    if hip_dq_gain is not None and (not math.isfinite(hip_dq_gain) or not 0<=hip_dq_gain<=1):
        raise ValueError('diagnostic hip dq gain must be in [0,1]')
    if kd is not None and (not math.isfinite(kd) or not .1<=kd<=3):
        raise ValueError('diagnostic kd must be in [.1,3]')
    if kp is not None and (not math.isfinite(kp) or not 10<=kp<=40):
        raise ValueError("diagnostic kp must be in [10,40]")
    output.mkdir(parents=True,exist_ok=False)
    tree=output/'policy'
    (tree/'imgo2/ppo').mkdir(parents=True)
    inputs={}
    for source,dest in [('policy_used.pt','imgo2/ppo/policy.pt'),
                        ('policy_config.yaml','imgo2/ppo/config.yaml'),
                        ('base_config.yaml','imgo2/base.yaml')]:
        path=saved/source
        shutil.copyfile(path,tree/dest)
        inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    overrides=[]
    config=tree/'imgo2/ppo/config.yaml'
    for name,value in [('rl_kd',kd),('rl_kp',kp)]:
        if value is None:continue
        text=config.read_text()
        matches=list(re.finditer(r'(?m)^(\s*'+name+r':\s*)\[([^\]]+)\]',text))
        if len(matches)!=1:raise ValueError('expected one '+name+' array in saved config')
        match=matches[0];old=[float(x.strip()) for x in match.group(2).split(',')]
        if len(old)!=12:raise ValueError('expected 12 saved gains')
        text=text[:match.start()]+match.group(1)+'['+', '.join([str(value)]*12)+']'+text[match.end():]
        config.write_text(text)
        overrides.append({'field':name,'saved':old,'diagnostic':[value]*12,
                          'scope':'isolated diagnostic policy tree only; all other config text preserved'})
    b=ROOT/'imgo2_deploy/ros2_ws/build/imgo2_deploy'
    commands=json.loads((b/'compile_commands.json').read_text())
    entry=next(x for x in commands if x['file'].endswith('/rl_sdk.cpp'))
    cmd=shlex.split(entry['command'])
    intervention=None
    if hip_dq_gain is not None:
        # Modify an isolated source copy only. Keep physical dof_vel untouched,
        # so actuator PD, torque diagnostics and feedback snapshots use reality.
        original=Path(entry['file']);text=original.read_text()
        anchor='    return clamped_obs;'
        if text.count(anchor)!=1:raise ValueError('unexpected observation return site')
        patch=('    // Isolated simulation diagnostic: actor-input gain only.\n'
               '    if (clamped_obs.size() != 45) throw std::runtime_error("diagnostic expects PPO45");\n'
               f'    for (int i : {{21, 24, 27, 30}}) clamped_obs[i] *= {hip_dq_gain:.9f}f;\n')
        copied=output/'rl_sdk_hip_dq_diagnostic.cpp';copied.write_text(text.replace(anchor,patch+anchor))
        cmd[cmd.index('-c')+1]=str(copied)
        cmd+=['-I',str(original.parent)]
        intervention={'hip_dq_gain':hip_dq_gain,'obs_indices':[21,24,27,30],
            'original_source':str(original),'original_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
            'isolated_source':str(copied),'isolated_sha256':hashlib.sha256(copied.read_bytes()).hexdigest(),
            'injected_text':patch,'physical_PD_feedback_modified':False,
            'scope':'45-vector returned by ComputeObservation only; fresh own-run feedback, no historical real obs'}
    for i,arg in enumerate(cmd):
        if arg.startswith('-DPOLICY_DIR='):cmd[i]=f'-DPOLICY_DIR="{tree}"'
    cmd[cmd.index('-o')+1]=str(output/'rl_sdk_saved.cpp.o')
    with (output/'build.log').open('w') as log:
        subprocess.run(cmd,cwd=entry['directory'],stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run(['/usr/bin/ar','qc',str(output/'librl_sdk_saved.a'),str(output/'rl_sdk_saved.cpp.o')],check=True)
        subprocess.run(['/usr/bin/ranlib',str(output/'librl_sdk_saved.a')],check=True)
        link=shlex.split((b/'CMakeFiles/rl_sim.dir/link.txt').read_text())
        link[link.index('-o')+1]=str(output/'rl_sim_saved')
        link[link.index('librl_sdk.a')]=str(output/'librl_sdk_saved.a')
        subprocess.run(link,cwd=b,stdout=log,stderr=subprocess.STDOUT,check=True)
    meta={'saved_inputs':inputs,'compile_command':cmd,'link_command':link,
          'binary_sha256':hashlib.sha256((output/'rl_sim_saved').read_bytes()).hexdigest(),
          'limitation':'current local deployment sources/other existing objects; saved model/config only; not a historical binary reproduction'}
    if intervention:meta['observation_intervention']=intervention
    if overrides:
        meta['gain_overrides']=overrides
        meta['diagnostic_config_sha256']=hashlib.sha256(config.read_bytes()).hexdigest()
    (output/'build_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(output/'rl_sim_saved')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--saved',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--kd',type=float,help='Override only all 12 rl_kd values in the isolated diagnostic config; does not edit production')
    p.add_argument('--kp',type=float,help='Override only all 12 rl_kp values in isolated diagnostic config')
    p.add_argument('--hip-dq-gain',type=float,help='Only isolated SDK source: scale four actor hip dq channels; never physical PD/state')
    a=p.parse_args();build(a.saved.resolve(),a.output.resolve(),a.kd,a.kp,a.hip_dq_gain)
