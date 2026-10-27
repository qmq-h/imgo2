#!/usr/bin/env python3
"""Standard-library metrics for deterministic-time MuJoCo policy comparison CSVs."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import statistics as st

LEGS=('FL','FR','RL','RR')
JOINTS=[f'{leg}_{j}_joint' for leg in LEGS for j in ('hip','thigh','shank')]


def contact_flags(force, dt):
    """5N on / 2N off hysteresis; require 10ms persistence, remove <20ms chatter."""
    flags=[];state=force[0]>=5;candidate=state;count=0
    hold=max(1,round(.01/dt))
    for f in force:
        desired=not(f<=2) if state else f>=5
        if desired==state:count=0;candidate=state
        else:
            if candidate!=desired:candidate=desired;count=0
            count+=1
            if count>=hold:
                state=desired;count=0
                for k in range(1,min(hold-1,len(flags))+1):flags[-k]=state
        flags.append(state)
    for _ in range(2):
        segments=runs(flags)
        for a,b,state in segments[1:-1]:
            if (b-a)*dt<.02-1e-9:
                for i in range(a,b):flags[i]=not state
    return flags


def runs(values):
    bounds=[0]+[i for i in range(1,len(values)) if values[i]!=values[i-1]]+[len(values)]
    return [(a,b,values[a]) for a,b in zip(bounds,bounds[1:]) if b>a]


def mean(values):return st.mean(values) if values else None
def median(values):return st.median(values) if values else None
def rms(values):return math.sqrt(st.mean(x*x for x in values)) if values else None


def metrics(path, skip):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt') as f:rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
    rows=[r for r in rows if r['t_s']>=skip]
    if len(rows)<100:return {'samples':len(rows),'insufficient_steady_data':True}
    dt=st.median(rows[i]['t_s']-rows[i-1]['t_s'] for i in range(1,len(rows)))
    def col(k):return [r[k] for r in rows]
    out={'samples':len(rows),'start_s':rows[0]['t_s'],'end_s':rows[-1]['t_s'],'sample_dt':dt}
    for label,key,cmd in [('vx','vx_body','command_vx'),('vy','vy_body','command_vy'),('wz','gyro_z_rads','command_wz')]:
        v=col(key);errors=[r[key]-r[cmd] for r in rows]
        out[label+'_mean']=mean(v);out[label+'_mae']=mean(list(map(abs,errors)));out[label+'_rmse']=rms(errors)
    heights=col('z_m');out.update(height_mean_m=mean(heights),height_std_m=st.pstdev(heights),height_min_m=min(heights))
    for key in ('roll','pitch'):
        values=[math.degrees(r[key+'_rad']) for r in rows]
        out[key+'_mean_deg']=mean(values);out[key+'_std_deg']=st.pstdev(values)
        out[key+'_ptp_deg']=max(values)-min(values)
    out['gyro_xy_rms_rads']=math.sqrt(st.mean(r['gyro_x_rads']**2+r['gyro_y_rads']**2 for r in rows))
    # Subtract a centered 0.2s mean as a rapid-motion proxy (not a sharp frequency band).
    half=max(1,round(.1/dt));hp=[]
    for key in ('gyro_x_rads','gyro_y_rads'):
        v=col(key);prefix=[0.]
        for x in v:prefix.append(prefix[-1]+x)
        for i,x in enumerate(v):
            lo=max(0,i-half);hi=min(len(v),i+half+1)
            hp.append((x-(prefix[hi]-prefix[lo])/(hi-lo))**2)
    out['gyro_xy_minus_0p2s_mean_rms_rads']=math.sqrt(sum(hp)/len(rows))
    out['joint_dq_rms_rads']=rms([r['dq_'+j] for r in rows for j in JOINTS])
    out['joint_acc_difference_rms_rads2']=rms([(rows[i]['dq_'+j]-rows[i-1]['dq_'+j])/dt for i in range(1,len(rows)) for j in JOINTS])
    out['action_saturated_fraction']=st.mean(r['action_saturated']/12 for r in rows)
    out['torque_saturated_fraction']=st.mean(abs(r['tau_'+j])>=.99*23.7 for r in rows for j in JOINTS)
    out['motor_age_mean_ms']=mean(col('motor_age_ms'));out['imu_age_mean_ms']=mean(col('imu_age_ms'))
    footstats={};all_periods=[];all_air=[];all_clearances=[]
    for leg in LEGS:
        force=col(leg+'_contact_n');flags=contact_flags(force,dt)
        segments=runs(flags);landings=[];air=[];clearance=[]
        for index,(a,b,contact) in enumerate(segments):
            if not contact and index>0 and index<len(segments)-1 and (b-a)*dt>=.03-1e-9:
                landings.append(rows[b]['t_s']);air.append((b-a)*dt)
                clearance.append(max(r[leg+'_clearance_m'] for r in rows[a:b]))
        periods=[b-a for a,b in zip(landings,landings[1:]) if .1-1e-9<=b-a<=1.5+1e-9]
        footstats[leg]={'landing_count':len(landings),'period_median_s':median(periods),
                        'freq_hz':1/st.median(periods) if periods else None,
                        'air_time_mean_s':mean(air),'swing_clearance_peak_mean_m':mean(clearance),
                        'swing_clearance_peak_median_m':median(clearance),
                        'stance_fraction':st.mean(flags),
                        'hip_outward_mean_deg':math.degrees(st.mean(r[leg+'_hip_joint']*(1 if leg.endswith('L') else -1) for r in rows)),
                        'foot_outward_from_hip_mean_m':st.mean((r[leg+'_y_body']-r[leg+'_hip_y_body'])*(1 if leg.endswith('L') else -1) for r in rows)}
        all_periods.extend(periods);all_air.extend(air);all_clearances.extend(clearance)
    out['feet']=footstats
    out['stride_period_median_s']=median(all_periods)
    out['stride_frequency_hz']=1/st.median(all_periods) if all_periods else None
    out['air_time_mean_s']=mean(all_air);out['swing_clearance_peak_mean_m']=mean(all_clearances)
    out['front_hip_outward_mean_deg']=st.mean(footstats[l]['hip_outward_mean_deg'] for l in LEGS[:2])
    out['rear_hip_outward_mean_deg']=st.mean(footstats[l]['hip_outward_mean_deg'] for l in LEGS[2:])
    out['front_feet_width_mean_m']=st.mean(r['FL_y_body']-r['FR_y_body'] for r in rows)
    out['rear_feet_width_mean_m']=st.mean(r['RL_y_body']-r['RR_y_body'] for r in rows)
    out['joint_mean_rad']={j:mean(col(j)) for j in JOINTS}
    out['feet_body_mean_m']={leg:[mean(col(leg+'_'+k+'_body')) for k in 'xyz'] for leg in LEGS}
    # The current PPO contract has q0_hip=0, scale=.125 and raw action clip +/-3.
    # These are action clipping fractions, not physical joint-limit violations.
    out['hip_action_clip_fraction']={leg:st.mean(abs(r['qdes_'+leg+'_hip_joint'])>=.375-1e-7 for r in rows) for leg in LEGS}
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--skip',type=float,default=4.)
    a=p.parse_args();run=json.loads(a.run.read_text())
    result={'scope':run['scope'],'metadata':run,'skip_seconds':a.skip,'results':[],
            'definitions':{'cadence':'Same-foot repeated landings after 5N/2N hysteresis, 10ms persistence, <20ms chatter removal; complete air runs >=30ms, periods .1..1.5s; pooled median over four feet. Not policy/control Hz.',
            'clearance':'Collision sphere bottom above flat floor, peak per complete air run; not body/link origin z.',
            'outward_hip':'Left q_hip / right -q_hip, positive=abducted outward; averaged front/rear. This is not a yaw joint angle.',
            'foot_width':'Mean left-right collision center separation in base frame.',
            'jitter':'Physical roll/pitch and angular velocity; desired yaw excluded. Acceleration finite-difference and rapid-motion proxies are not measured hardware vibration.'}}
    for case in run['cases']:
        path=Path(case['csv'])
        if not path.exists():path=a.run.parent/path.name
        m=metrics(path,a.skip);m.update(policy=case['policy'],case=case['case'],command=case['command'],failure=case['failure'])
        result['results'].append(m)
        print(case['policy'],case['case'],'z',m.get('height_mean_m'),'freq',m.get('stride_frequency_hz'),'vx',m.get('vx_mean'),'hip front/rear',m.get('front_hip_outward_mean_deg'),m.get('rear_hip_outward_mean_deg'),flush=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
