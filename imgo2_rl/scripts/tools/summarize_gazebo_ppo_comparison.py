#!/usr/bin/env python3
"""Summarize Gazebo PPO telemetry and reconstruct foot collision clearance."""
import argparse
import json
import math
from pathlib import Path
import statistics as st
import summarize_foot_clearance as sole


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--policies',nargs='+',default=['policy','policy_1'])
    p.add_argument('--cases',default='stand,forward03,backward03,turn_left03')
    a=p.parse_args();out=[]
    for label in a.policies:
        for case in a.cases.split(','):
            name='gazebo_'+label+'_'+case
            path=a.directory/(name+'.json');raw=a.directory/(name+'_raw.json')
            if not path.exists() or not raw.exists():continue
            d=json.loads(path.read_text());r=json.loads(raw.read_text());foot=sole.analyze(raw,sole.REPO/'imgo2_description/urdf/imgo2_real.urdf')
            peaks=[x['height_bout_mean_peak_m'] for x in foot['per_foot'].values() if x['height_bout_mean_peak_m'] is not None]
            out.append(dict(policy=label+'.pt',case=case,vx=d['tracking']['speed_mean'],vy=d['tracking']['lateral_speed_mean'],wz=d['tracking']['yaw_rate_mean'],height_m=d['pose']['z_mean'],height_std_m=st.pstdev(x[2] for x in r['odom']),roll_std_deg=math.degrees(st.pstdev(x[4] for x in r['odom'])),pitch_std_deg=math.degrees(st.pstdev(x[5] for x in r['odom'])),
                            frequency_hz=d['periodicity']['freq_hz'] if case in ('forward03','backward03') else None,
                            geometric_swing_peak_m=st.mean(peaks) if peaks else None,
                            jitter=d['jitter'],geometric_feet=foot))
    a.output.write_text(json.dumps(dict(scope='Production Gazebo path; cadence is thigh autocorrelation, clearance bouts are geometric, not force-based. Do not directly equate with MuJoCo force-event metrics.',results=out),indent=2,allow_nan=False))
    print('Summarized',len(out),'Gazebo cases')


if __name__=='__main__':main()
