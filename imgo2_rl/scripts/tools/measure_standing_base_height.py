#!/usr/bin/env python3
"""Measure base origin/COM height from bounded Gazebo stand-only recordings.

Stdlib only. Uses ground-truth base pose and URDF inertial origin; no hardware,
policy, simulation startup, or assumption that the mass is a uniform box.
"""
import argparse
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[3]

def load(path):
    with path.open() as f:return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]

def summary(v):return {'mean_m':st.mean(v),'min_m':min(v),'max_m':max(v),'std_m':st.pstdev(v),'peak_to_peak_m':max(v)-min(v)}

def main(a):
    c=json.loads((a.run/'collection.json').read_text());m=json.loads((a.run/'metadata.json').read_text())
    assert c['complete'] and c['stand_only'] and c['ppo_pd_start_sim_s'] is None
    assert m['completed'] and m['sources_unchanged']
    end=c['clock_origin_sim_s']+6+c['hold_seconds'];lo=end-a.window_seconds
    root=ET.parse(a.urdf).getroot();base=root.find("link[@name='base']")
    com=[float(v) for v in base.find('inertial/origin').get('xyz').split()]
    mass=float(base.find('inertial/mass').get('value'))
    box=base.find('collision');assert box.find('origin').get('xyz')=='0 0 0'
    size=[float(v) for v in box.find('geometry/box').get('size').split()]
    odom=load(a.run/'odom.csv');rows=[]
    for r in odom:
        if not lo<=r['sim_s']<end:continue
        q=[r['quat_'+k] for k in 'wxyz'];n=math.sqrt(sum(v*v for v in q));w,x,y,z=[v/n for v in q]
        # Third row of the base-to-world rotation matrix.
        r31=2*(x*z-w*y);r32=2*(y*z+w*x);r33=1-2*(x*x+y*y)
        center_z=r['z'];com_z=center_z+r31*com[0]+r32*com[1]+r33*com[2]
        rows.append({'sim_s':r['sim_s'],'base_origin_height_m':center_z-a.ground_z,
                     'base_inertial_com_height_m':com_z-a.ground_z,
                     'main_box_bottom_center_height_m':center_z-r33*size[2]/2-a.ground_z,
                     'main_box_lowest_corner_height_m':center_z-(abs(r31)*size[0]+abs(r32)*size[1]+abs(r33)*size[2])/2-a.ground_z})
    assert len(rows)>=a.window_seconds*90
    heights={k:summary([r[k] for r in rows]) for k in rows[0] if k!='sim_s'}
    # Map command/state host receipts to exact odometry header time, as in
    # existing collectors; no gain or target inference from directory names.
    ht=[r['host_s'] for r in odom]
    def in_window(r):
        h=r['host_s']
        if not ht[0]<=h<=ht[-1]:return False
        j=min(len(ht)-1,max(1,bisect.bisect_right(ht,h)));x,y=odom[j-1:j+1]
        t=x['sim_s']+(h-ht[j-1])/(ht[j]-ht[j-1])*(y['sim_s']-x['sim_s'])
        return lo<=t<end
    cmd=[r for r in load(a.run/'command.csv') if in_window(r)]
    state=[r for r in load(a.run/'state.csv') if in_window(r)]
    assert cmd and state
    assert all(abs(r[f'kp_{i:02}']-60)<1e-5 and abs(r[f'kd_{i:02}']-2)<1e-5 for r in cmd for i in range(12))
    expected=[0,.87,-1.82]*4
    assert max(abs(r[f'q_{i:02}']-expected[i]) for r in cmd for i in range(12))<1e-6
    assert max(heights['base_origin_height_m']['peak_to_peak_m'],heights['base_inertial_com_height_m']['peak_to_peak_m'])<.001
    result={'run':str(a.run.resolve()),'measurement':'Gazebo stand-only, final steady window; original base inertial COM, not whole-robot COM',
            'window_sim_s':[lo,end],'ground_z_m':a.ground_z,'samples':len(rows),'base_mass_kg':mass,'base_local_com_xyz_m':com,
            'main_collision_box_size_m':size,'heights':heights,'standing_gains':{'kp':60,'kd':2},
            'joint_target_rad':expected,'joint_feedback_mean_rad':[st.mean(r[f'q_{i:02}'] for r in state) for i in range(12)],
            'state_samples':len(state),'command_samples':len(cmd),
            'limits':['Ground must be the horizontal earth-world plane at the specified elevation.',
                      'Base COM is the original base link inertial origin transformed by measured base pose, not a uniform-box mass assumption.',
                      'Whole robot COM includes moving legs and is not measured by this base-only result.',
                      'Gazebo fixed-joint lumping may include the small base_imu mass in its combined base link inertial; this result explicitly uses original URDF base COM.',
                      'This is default GetUp stand with60/2, not PPO walking25/0.8 or a guarantee of real body height.'],
            'sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.urdf,a.run/'odom.csv',a.run/'state.csv',a.run/'command.csv',a.run/'collection.json',a.run/'metadata.json')}}
    a.output_prefix.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with a.output_prefix.with_suffix('.csv').open('w') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'samples':len(rows),'window_sim_s':[lo,end],'heights':heights},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--urdf',type=Path,default=ROOT/'imgo2_description/urdf/imgo2_real.gazebo.urdf')
    p.add_argument('--ground-z',type=float,default=0.);p.add_argument('--window-seconds',type=float,default=4.)
    p.add_argument('--output-prefix',type=Path,required=True);main(p.parse_args())
