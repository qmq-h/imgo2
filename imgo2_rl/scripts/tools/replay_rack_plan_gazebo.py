#!/usr/bin/env python3
"""Run fixed-base diagnostic baseline/resistance trials for a frozen plan.

No hardware or policy interface. Each Gazebo process is owned/closed by the
existing replay adapter. Source hashes are checked before starting.
"""
import argparse,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'imgo2_deploy/ros2_ws/scripts'))
import run_joint_sine_gazebo as sine

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plugin-library',type=Path,required=True);p.add_argument('--resistance-library',type=Path,required=True)
    p.add_argument('--duration',type=float,default=9.)
    a=p.parse_args();assert .5<=a.duration<=15
    plan=json.loads(a.plan.read_text());a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'plan_used.json').write_text(json.dumps(plan,indent=2)+'\n')
    par=a.output/'resistance_parameters.json';par.write_text(json.dumps(plan['selected_parameters'],indent=2)+'\n')
    for source in plan['sources']:
        path=Path(source['source']);assert hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256']
        for variant in ('baseline','previous_candidate'):
            print('BEGIN',source['id'],variant,flush=True)
            sine.run(SimpleNamespace(source=path,condition='rack',duration=a.duration,output=a.output/f"case_{source['id']:02}_{variant}",gui=False,plugin_library=a.plugin_library,resistance_parameters=par if variant!='baseline' else None,resistance_library=a.resistance_library))
            print('COMPLETE',source['id'],variant,flush=True)
if __name__=='__main__':main()
