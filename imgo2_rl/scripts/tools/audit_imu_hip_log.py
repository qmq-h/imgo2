#!/usr/bin/env python3
"""Offline logged hip/IMU audit. Standard library only; no hardware access.

Quaternion/gyro agreement checks internal kinematics, not absolute base alignment.
Spectra use 100Hz host-time bins; peaks are descriptive, not contact labels.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics as st

from audit_go_imu_runtime import read_csv, stats
from compare_ppo_sim_real_imu import rpy

HIPS = [0, 3, 6, 9]


def rms(values):
    values = list(values)
    return math.sqrt(st.mean(v*v for v in values))


def corr(a, b):
    ma, mb = st.mean(a), st.mean(b)
    aa, bb = [x-ma for x in a], [x-mb for x in b]
    den = math.sqrt(sum(x*x for x in aa)*sum(x*x for x in bb))
    return sum(x*y for x,y in zip(aa,bb))/den if den > 1e-14 else None


def mul(a, b):
    w,x,y,z=a; v,i,j,k=b
    return (w*v-x*i-y*j-z*k,w*i+x*v+y*k-z*j,w*j-x*k+y*v+z*i,w*k+x*j-y*i+z*v)


def body_rotation_rate(a, b, dt):
    d = mul((a[0],-a[1],-a[2],-a[3]), b)
    if d[0] < 0: d = tuple(-x for x in d)
    v = math.sqrt(sum(x*x for x in d[1:]))
    gain = 2*math.atan2(v,d[0])/v/dt if v > 1e-12 else 2/dt
    return [gain*x for x in d[1:]]


def spectrum(values, rate=100):
    n=len(values)
    if n < 80: return None
    mean=st.mean(values)
    y=[(v-mean)*(.5-.5*math.cos(2*math.pi*i/(n-1))) for i,v in enumerate(values)]
    powers=[]
    for k in range(1,n//2):
        f=k*rate/n
        if not 2 <= f <= 40: continue
        real=sum(v*math.cos(2*math.pi*k*i/n) for i,v in enumerate(y))
        imag=sum(v*math.sin(2*math.pi*k*i/n) for i,v in enumerate(y))
        powers.append((real*real+imag*imag,f))
    return dict(peak_hz=max(powers)[1], resolution_hz=rate/n,
                scope='2-40Hz demeaned Hann DFT; host bins, not device spectrum')


def audit(path, windows):
    c,bad=read_csv(path/'robot.csv'); n=len(c['tick'])
    quats=[]; norms=[]; angles=[]
    for i in range(n):
        q=[c['quat_'+k][i] for k in 'wxyz']; norm=math.sqrt(sum(v*v for v in q));norms.append(norm)
        if not math.isfinite(norm) or norm<1e-12: raise ValueError('invalid quaternion')
        q=[v/norm for v in q];quats.append(q);angles.append(rpy(*q))
    t=[v/1000 for v in c['t_mono_ms']]
    bins=defaultdict(list)
    for i in range(n):
        if c['mode'][i]==3 and c['imu_valid'][i]: bins[int(math.floor(t[i]*100))].append(i)
    binned={}
    for key,ix in bins.items():
        q=[st.mean(quats[i][k]*(1 if sum(a*b for a,b in zip(quats[ix[0]],quats[i]))>=0 else -1) for i in ix) for k in range(4)]
        norm=math.sqrt(sum(v*v for v in q));q=[v/norm for v in q]
        binned[key]=dict(t=st.mean(t[i] for i in ix),q=q,
            g=[st.mean(c['gyro_'+k][i] for i in ix) for k in 'xyz'],
            hips=[st.mean(c[f'q_{j:02}'][i] for i in ix) for j in HIPS])
    rates={}
    # 40ms centered quaternion change versus mean gyro in that same window.
    for key in binned:
        if not all(key+j in binned for j in range(-2,3)):continue
        a,b=binned[key-2],binned[key+2]
        rates[key]=body_rotation_rate(a['q'],b['q'],b['t']-a['t'])
    agreements={}
    for axis in range(3):
        pairs=[]
        for key,v in rates.items():
            g=st.mean(binned[key+j]['g'][axis] for j in range(-2,3))
            pairs.append((g,v[axis]))
        x,y=map(list,zip(*pairs));energy=sum(v*v for v in x)
        agreements['xyz'[axis]]=dict(correlation=corr(x,y),quat_rate_over_gyro_scale=sum(a*b for a,b in pairs)/energy if energy else None,
            residual_rms_rad_s=rms(a-b for a,b in pairs),gyro_smoothed_rms_rad_s=rms(x),samples=len(x))
    def metrics(ix):
        return dict(samples=len(ix),roll_std_deg=st.pstdev(angles[i][0]*180/math.pi for i in ix),
            pitch_std_deg=st.pstdev(angles[i][1]*180/math.pi for i in ix),
            gyro_rms_rad_s=[rms(c['gyro_'+k][i] for i in ix) for k in 'xyz'],
            command_ranges=[[min(c[k][i] for i in ix),max(c[k][i] for i in ix)] for k in ['cmd_vx','cmd_vy','cmd_wz']],
            hips={str(j):dict(q_std_deg=st.pstdev(c[f'q_{j:02}'][i]*180/math.pi for i in ix),
                target_std_deg=st.pstdev(c[f'q_des_{j:02}'][i]*180/math.pi for i in ix),
                error_rms_deg=rms((c[f'q_des_{j:02}'][i]-c[f'q_{j:02}'][i])*180/math.pi for i in ix),
                dq_rms_rad_s=rms(c[f'dq_{j:02}'][i] for i in ix)) for j in HIPS})
    selected=[]
    for start,end in windows:
        ix=[i for i,at in enumerate(t) if start<=at<end]
        if not ix:continue
        keys=sorted(k for k,v in binned.items() if start<=v['t']<end)
        row=dict(start_s=start,end_s=end,**metrics(ix))
        if len(keys)>80 and keys[-1]-keys[0]+1==len(keys):
            row['gyro_x_spectrum']=spectrum([binned[k]['g'][0] for k in keys])
            row['hip_spectra']={str(j):spectrum([binned[k]['hips'][a] for k in keys]) for a,j in enumerate(HIPS)}
        selected.append(row)
    command_windows=[]
    for start in range(int(t[0]),int(t[-1])-1,2):
        ix=[i for i in range(n) if start<=t[i]<start+2 and c['mode'][i]==3]
        if len(ix)<300:continue
        commands={tuple(round(c[k][i],2) for k in ['cmd_vx','cmd_vy','cmd_wz']) for i in ix}
        if len(commands)==1:command_windows.append(dict(start_s=start,end_s=start+2,**metrics(ix)))
    digest=hashlib.sha256((path/'robot.csv').read_bytes()).hexdigest()
    return dict(directory=str(path),rows=n,malformed_rows=bad,duration_s=t[-1]-t[0],robot_sha256=digest,
        quaternion_norm=dict(min=min(norms),**stats(norms)),nonfinite_imu_values=sum(not math.isfinite(c[k][i]) for k in ['gyro_x','gyro_y','gyro_z','quat_w','quat_x','quat_y','quat_z'] for i in range(n)),
        gyro_ranges_rad_s={k:[min(c['gyro_'+k]),max(c['gyro_'+k])] for k in 'xyz'},
        imu_valid_counts={str(int(v)):sum(x==v for x in c['imu_valid']) for v in set(c['imu_valid'])},
        quaternion_gyro_40ms_consistency=agreements,selected_windows=selected,
        constant_command_2s_windows=command_windows,
        cautions=['Unlabelled contact/lifting; quiet window is not proof of normal ground walking.',
            'Internal quaternion/gyro consistency is not independent IMU truth, nor proof of correct base axes.',
            '40ms kinematic comparison smooths high-frequency content; device time and original frames unavailable.',
            'q_des is host submitted target, not a timestamped acknowledgement of actual per-motor application.',
            '2s windows/time-bin spectra are descriptive; no causal inference from phase or correlation alone.'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--log',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--window',action='append',default=[],help='start:end seconds from logger start')
    a=p.parse_args();windows=[tuple(map(float,s.split(':'))) for s in a.window]
    report=audit(a.log,windows);a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['rows','duration_s','quaternion_gyro_40ms_consistency']},indent=2))


if __name__=='__main__':main()
