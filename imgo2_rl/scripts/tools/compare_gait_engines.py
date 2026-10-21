#!/usr/bin/env python3
"""Apply the same thigh autocorrelation metric to native Isaac and Gazebo traces."""
import argparse
import ast
import json
import math
from pathlib import Path
import statistics

REPO=Path(__file__).resolve().parents[3]
LEGS=("FL","FR","RL","RR")


def finite(value):
    if isinstance(value,float) and not math.isfinite(value): return None
    if isinstance(value,dict): return {k:finite(v) for k,v in value.items()}
    if isinstance(value,list): return [finite(v) for v in value]
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("native_raw","native_summary","gazebo_summary","gazebo_foot_summary","json"):
        parser.add_argument("--"+name.replace('_','-'),required=True,type=Path)
    args=parser.parse_args()
    source=REPO/"imgo2_deploy/ros2_ws/scripts/eval_gazebo_policy.py"
    function=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=="autocorr_peak")
    namespace={"math":math}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),"exec"),namespace)
    native=json.loads(args.native_raw.read_text())
    feet=json.loads(args.native_summary.read_text())
    gazebo=json.loads(args.gazebo_summary.read_text())
    clearance=json.loads(args.gazebo_foot_summary.read_text())
    results=[]
    for case,g in (("forward","forward_04"),("backward","backward_04"),("turn_left","turn_left"),("turn_right","turn_right")):
        n=native[case]; periods={}
        for leg in LEGS:
            index=n['joint_names'].index(leg+'_thigh_joint')
            signal=[row[index] for row in n['joint_positions']]
            average=statistics.mean(signal)
            period,score,hz=namespace['autocorr_peak']([v-average for v in signal],feet['control_dt'])
            periods[leg]={'period_s':period,'score':score,'hz':hz}
        a=next(x for x in gazebo if x['case']==g)
        c=next(x for x in clearance if '/'+g+'/' in x['raw_file'])
        b=next(x for x in feet['cases'] if x['case']==case)
        results.append({'case':case,'gazebo':{'mean_vx':a['tracking']['speed_mean'],'mean_wz':a['tracking']['yaw_rate_mean'],
            'thigh_period':a['periodicity'],'p95_mm':[c['per_foot'][leg]['p95_m']*1000 for leg in LEGS]},
            'isaac':{'mean_vx':b['mean_vx'],'mean_wz':b['mean_wz'],'thigh_period':periods,
            'p95_mm':[b['per_foot'][leg+'_FOOT']['p95_m']*1000 for leg in LEGS],
            'contact_fraction':[b['per_foot'][leg+'_FOOT']['contact_fraction'] for leg in LEGS]}})
    output={'cases':finite(results),'limits':'Same joint autocorrelation metric; weak/irregular peaks are not proof of contact gait frequency. One environment per command; engine contact/actuator parameters differ.'}
    args.json.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print('Compared four directions; written',args.json)


if __name__=='__main__': main()
