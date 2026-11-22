#!/usr/bin/env python3
"""Conditional stand-height FK estimate from real logs; not measured height.

Stdlib only. Assumes model geometry/zeros, logged orientation, horizontal ground
and contacting feet. Four individual foot solutions expose inconsistency.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics as st
from analyze_standup_replay import ground_estimate

ROOT=Path(__file__).resolve().parents[3]

def load(p):
    with p.open() as f:return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]

def estimate(model,mean):
    g=ground_estimate(model,mean);feet={}
    for leg in ('FL','FR','RL','RR'):
        collisions=[r for r in g['lowest_collisions'] if r['link']==leg+'_FOOT']
        if not collisions:raise ValueError('Standing foot not in lowest collision set')
        feet[leg]=-min(x['minimum_z_at_zero_base_m'] for x in collisions)
    return {'foot_conditioned_base_heights_m':feet,'all_feet_least_squares_translation_m':st.mean(feet.values()),
            'foot_solution_span_m':max(feet.values())-min(feet.values()),'geometry':g}

def main(a):
    raw=load(a.real_log/'robot.csv');rr=[r for r in raw if a.lo<=r['t_mono_ms']/1000<a.hi]
    assert rr and all(r['mode']==2 for r in rr)
    assert all(r[f'kp_{i:02}']==60 and r[f'kd_{i:02}']==2 for r in rr for i in range(12))
    expected=[0,.87,-1.82]*4
    assert max(abs(r[f'q_des_{i:02}']-expected[i]) for r in rr for i in range(12))<1e-6
    mean={k:st.mean(r[k] for r in rr) for k in rr[0]};e=estimate(a.model,mean)
    result={'real_log':str(a.real_log.resolve()),'window_s':[a.lo,a.hi],'samples':len(rr),
            'method':'FK of window-mean q/quaternion using model foot collision bottoms; each foot separately assumed on plane z=0',
            'real_estimate':e,'height_is_directly_logged':False,
            'limitations':['No base translation/height or foot contact/pressure/sole compression measured in real robot log.',
                'Each foot solution is conditional, not a confidence interval or actual known support height.',
                'Mean of four inferred heights is a geometric fit, not independent evidence that four feet share an exact plane.',
                'Absolute zero, geometry, orientation reference and contact condition can affect inference; no unique cause identified.',
                'This estimate uses model geometry; it cannot independently validate that same geometry.'],
            'sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.real_log/'robot.csv',a.model)}}
    if a.gazebo_run:
        c=json.loads((a.gazebo_run/'collection.json').read_text());end=c['hold_seconds']
        ss=[r for r in load(a.gazebo_run/'state.csv') if end-4<=r['t_s']<end]
        oo=[r for r in load(a.gazebo_run/'odom.csv') if end-4<=r['t_s']<end]
        simmean={k:st.mean(r[k] for r in ss) for k in ss[0]}
        simmean.update({f'quat_{k}':st.mean(r[f'quat_{k}'] for r in oo) for k in 'wxyz'})
        se=estimate(a.model,simmean);actual=st.mean(r['z'] for r in oo)
        result['gazebo_method_check']={'run':str(a.gazebo_run.resolve()),'actual_base_height_m':actual,**se,
            'fit_minus_actual_m':se['all_feet_least_squares_translation_m']-actual}
        assert abs(result['gazebo_method_check']['fit_minus_actual_m'])<.001
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'real':e, 'gazebo_method_check':result.get('gazebo_method_check')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--real-log',type=Path,required=True)
    p.add_argument('--lo',type=float,default=15.);p.add_argument('--hi',type=float,default=19.)
    p.add_argument('--model',type=Path,default=ROOT/'imgo2_description/urdf/imgo2_real.gazebo.urdf')
    p.add_argument('--gazebo-run',type=Path);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
