#!/usr/bin/env python3
"""Summarize stationary policy target changes from native or Gazebo audit traces."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def metrics(times, targets, positions):
    steps=[max(abs(b[i]-a[i]) for i in range(12)) for a,b in zip(targets,targets[1:])]
    deltas=[b[i]-a[i] for a,b in zip(targets,targets[1:]) for i in range(12)]
    velocities=[(b[i]-a[i])/(tb-ta) for ta,tb,a,b in zip(times,times[1:],positions,positions[1:]) for i in range(12) if tb>ta]
    return {"samples":len(times),"duration_s":times[-1]-times[0],
            "max_target_step_rad":max(steps),
            "rms_target_step_rad":math.sqrt(statistics.mean(v*v for v in deltas)),
            "target_ranges_rad":[max(row[i] for row in targets)-min(row[i] for row in targets) for i in range(12)],
            "observed_joint_ranges_rad":[max(row[i] for row in positions)-min(row[i] for row in positions) for i in range(12)],
            "sampled_joint_velocity_rms_rad_s":math.sqrt(statistics.mean(v*v for v in velocities))}


def analyze(path, defaults):
    if path.suffix=='.csv':
        with path.open() as source: rows=list(csv.DictReader(source))
        telemetry=json.loads((path.parent/'transient.json').read_text())
        entry=telemetry['policy_entry_wall']
        rows=[r for r in rows if float(r['wall_s'])>=entry+4]
        times=[float(r['wall_s']) for r in rows]
        targets=[[float(r[f'target_{i}']) for i in range(12)] for r in rows]
        positions=[[float(r[f'obs_{9+i}'])+defaults[i] for i in range(12)] for r in rows]
        source='gazebo_actual_actor_trace'
        # Old CSVs printed epoch seconds with only 10 significant digits.
        # Use the actual actor's dq observation for velocity, not coarse-stamp FD.
        observed_dq=[float(row[f'obs_{21+i}'])/.05 for row in rows for i in range(12)]
    else:
        data=json.loads(path.read_text())['stand']
        times,targets,positions=data['times'],data['joint_targets'],data['joint_positions']
        source='native_isaac_stand'
    summary=metrics(times,targets,positions)
    if source=='gazebo_actual_actor_trace':
        summary['sampled_joint_velocity_rms_rad_s']=math.sqrt(statistics.mean(v*v for v in observed_dq))
        summary['velocity_source']='Actual actor dq observation, scale .05; not timestamp finite difference.'
    return {"file":str(path),"source":source,**summary,
            "limits":"12 joint order FL/FR/RL/RR. Gazebo q reconstructed from current 45D actor q_rel (scale=1); native sampled after physics step. Max target changes are not actual joint displacements; velocity estimates depend on sample rate."}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('traces',nargs='+',type=Path)
    parser.add_argument('--default-angles',nargs=3,type=float,default=[0,.87,-1.82])
    parser.add_argument('--json',type=Path)
    args=parser.parse_args()
    result=[analyze(path,args.default_angles*4) for path in args.traces]
    if args.json: args.json.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
