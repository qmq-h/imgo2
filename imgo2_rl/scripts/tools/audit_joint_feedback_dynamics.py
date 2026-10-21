#!/usr/bin/env python3
"""Offline q/dq and target/torque consistency; standard library, no hardware.

Best target alignment is an aggregate telemetry/model diagnostic, NOT measured
command transport latency or an identified firmware PD gain.
"""
import argparse
from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path
import statistics as st

from audit_go_imu_runtime import read_csv, stats
from audit_imu_hip_log import corr, rms

LAGS = [0, 2.5, 5, 7.5, 10, 15, 20, 30, 40]


def fit_pd(x, z, y):
    mx,mz,my=map(st.mean,(x,z,y))
    xx=sum((v-mx)**2 for v in x);zz=sum((v-mz)**2 for v in z)
    xz=sum((a-mx)*(b-mz) for a,b in zip(x,z))
    xy=sum((a-mx)*(b-my) for a,b in zip(x,y))
    zy=sum((a-mz)*(b-my) for a,b in zip(z,y))
    det=xx*zz-xz*xz
    if det<=1e-12:return None
    kp=(xy*zz-zy*xz)/det;kd=(zy*xx-xy*xz)/det;bias=my-kp*mx-kd*mz
    residual=[c-kp*a-kd*b-bias for a,b,c in zip(x,z,y)]
    variance=st.pvariance(y)
    return dict(kp_fit=kp,kd_fit=kd,bias=bias,r2=1-st.mean(v*v for v in residual)/variance if variance else None,
                residual_rms=rms(residual))


def audit(directory, windows):
    c,bad=read_csv(directory/'robot.csv');t=list(c['t_mono_ms']);n=len(t)
    assert all(a<b for a,b in zip(t,t[1:])), 'Log time must be increasing'
    ids=[i for i in range(n) if c['mode'][i]==3]
    start=t[ids[0]]
    ids=[i for i in ids if t[i]>=start+1000]
    shifts={lag:[max(0,bisect_right(t,t[i]-lag)-1) for i in ids] for lag in LAGS}
    joints=[]
    for j in range(12):
        q,dq,u,tau=[c[f'{key}_{j:02}'] for key in ['q','dq','q_des','tau']]
        k=2;derivative_ids=[i for i in ids if k<=i<n-k and t[i+k]-t[i-k]<40]
        qdot=[(q[i+k]-q[i-k])/((t[i+k]-t[i-k])/1000) for i in derivative_ids]
        vel=[st.mean(dq[i+s] for s in range(-k,k+1)) for i in derivative_ids]
        fits=[]
        for lag in LAGS:
            source=shifts[lag];x=[u[s]-q[i] for i,s in zip(ids,source)];z=[-dq[i] for i in ids];y=[tau[i] for i in ids]
            fit=fit_pd(x,z,y)
            nominal=[c[f'kp_{j:02}'][i]*a+c[f'kd_{j:02}'][i]*(c[f'dq_des_{j:02}'][i]+b)+c[f'tau_ff_{j:02}'][i] for i,a,b in zip(ids,x,z)]
            fits.append(dict(target_alignment_ms=lag,free_pd_fit=fit,
                nominal_pd_rmse=rms(a-b for a,b in zip(y,nominal))))
        selected=[]
        for a,b in windows:
            wi=[i for i in ids if a*1000<=t[i]<b*1000]
            if not wi:continue
            row=dict(start_s=a,end_s=b,samples=len(wi),nominal_pd_rmse_by_alignment={})
            for lag in LAGS:
                errors=[]
                for i in wi:
                    s=max(0,bisect_right(t,t[i]-lag)-1)
                    pred=c[f'kp_{j:02}'][i]*(u[s]-q[i])+c[f'kd_{j:02}'][i]*(c[f'dq_des_{j:02}'][i]-dq[i])+c[f'tau_ff_{j:02}'][i]
                    errors.append(tau[i]-pred)
                row['nominal_pd_rmse_by_alignment'][str(lag)]=rms(errors)
            selected.append(row)
        joints.append(dict(joint=j,qdot_vs_reported_dq=dict(correlation=corr(qdot,vel),
            qdot_over_dq=sum(a*b for a,b in zip(qdot,vel))/sum(v*v for v in vel)),
            tau_abs=stats(abs(tau[i]) for i in ids),torque_ge_23_5_rows=sum(abs(tau[i])>=23.5 for i in ids),
            alignments=fits,best_free_fit=max((x for x in fits if x['free_pd_fit']),key=lambda x:x['free_pd_fit']['r2']),
            best_nominal_fit=min(fits,key=lambda x:x['nominal_pd_rmse']),windows=selected))
    return dict(directory=str(directory),rows=n,malformed_rows=bad,ppo_samples_excluding_first_second=len(ids),
        command_abs_max={k:max(abs(c[k][i]) for i in ids) for k in ['cmd_vx','cmd_vy','cmd_wz']},
        meta_config=json.loads((directory/'meta.json').read_text())['config'],joints=joints,
        sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [directory/'robot.csv',directory/'meta.json']},
        limitations=['Estimated torque is not an independent force sensor.',
            'Host q_des is not a timestamped wire/applied target; source input/action pairing also unverified.',
            'Best lag combines command holding, actuation, feedback/filters and logging; not a measured USB delay.',
            'Position differentiation and velocity averaging use host snapshots, not device timestamps.',
            'Free PD fit is not independent identification; do not change gains based on fitted coefficients.'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--log',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--window',action='append',default=[]);a=p.parse_args()
    r=audit(a.log,[tuple(map(float,v.split(':'))) for v in a.window])
    a.output.write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n')
    for row in r['joints']:
        best=row['best_free_fit'];print(row['joint'],'q/dq corr',round(row['qdot_vs_reported_dq']['correlation'],4),
            'alignment',best['target_alignment_ms'],'fit',round(best['free_pd_fit']['r2'],4))


if __name__=='__main__':main()
