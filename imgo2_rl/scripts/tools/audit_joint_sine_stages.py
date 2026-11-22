#!/usr/bin/env python3
"""Audit same sine command across user-described support changes.

Core analysis uses the standard library; plotting optionally uses matplotlib.
Windows are explicitly selected after inspection, not inferred contact labels.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from audit_joint_sine_log import fit, integrate, phase_difference, quantile, rms


def draw(path, t, d, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 2, figsize=(15, 11), constrained_layout=True)
    a,b,c,e,f,g = axs.flat
    colors = ['tab:blue','tab:green','tab:purple']
    for ax in axs.flat:
        ax.grid(alpha=.25)
    a.plot(t,d['q_set_rad'],label='q target',color='tab:orange',lw=.8)
    a.plot(t,d['q_actual_rad'],label='q feedback',color='tab:blue',lw=.8)
    for w,color in zip(report['windows'],colors):
        a.axvspan(w['start_s'],w['end_s'],alpha=.12,color=color)
    a.set(title='Full position log; A/B/C analysis windows shaded',xlabel='Log-relative time (s)',ylabel='rad')
    a.legend(fontsize=8)
    b.bar([w['label'] for w in report['windows']], [w['amplitude_ratio']*100 for w in report['windows']],color=colors)
    b.set(title='1 Hz fundamental amplitude ratio (feedback / target)',ylabel='percent',ylim=(0,105))
    for i,w in enumerate(report['windows']):
        b.text(i,w['amplitude_ratio']*100+2,f"{w['amplitude_ratio']*100:.1f}%",ha='center')
    c.plot(t,d['dq_actual_rad_s'],lw=.5,alpha=.5,label='recorded dq feedback')
    c.axhline(0,color='black',lw=.7)
    for w,color in zip(report['windows'],colors):
        c.plot([w['start_s'],w['end_s']],[w['dq_consistency']['mean_dq_rad_s']]*2,color=color,lw=2,label=f"{w['label']} mean dq")
    c.set(title='Velocity target = 0; feedback has negative mean',xlabel='Log-relative time (s)',ylabel='rad/s')
    c.legend(fontsize=8)
    e.plot(t,d['tau_set_nm'],label='total computed PD torque',lw=.7,color='tab:red')
    e.plot(t,d['tau_actual_nm'],label='motor estimated torque',lw=.6,alpha=.8,color='tab:green')
    e.set(title='Total PD and estimated feedback torque',xlabel='Log-relative time (s)',ylabel='N m')
    e.legend(fontsize=8)
    for w,color in zip(report['windows'],colors):
        idx=[i for i,v in enumerate(t) if w['start_s']<=v<min(w['start_s']+2,w['end_s'])]
        tt=[t[i]-w['start_s'] for i in idx]
        if w is report['windows'][0]:
            f.plot(tt,[d['q_set_rad'][i]-w['fits']['q_set_rad']['mean'] for i in idx],label='q target',color='tab:orange',lw=2)
        f.plot(tt,[d['q_actual_rad'][i]-w['fits']['q_actual_rad']['mean'] for i in idx],label=f"{w['label']} q feedback",color=color)
        ii=[i for i,v in enumerate(t) if w['start_s']<=v<w['end_s']]
        it=[t[i] for i in ii]
        integ=[0.]
        for k in range(1,len(ii)):
            integ.append(integ[-1]+(it[k]-it[k-1])*(d['dq_actual_rad_s'][ii[k]]+d['dq_actual_rad_s'][ii[k-1]])/2)
        xx=[v-it[0] for v in it]
        g.plot(xx,integ,color=color,label=f"{w['label']} integral(dq)")
        g.plot(xx,[d['q_actual_rad'][i]-d['q_actual_rad'][ii[0]] for i in ii],color=color,ls='--',lw=.8,label=f"{w['label']} q change")
    f.set(title='First 2 s of each window; means removed, NO phase shift',xlabel='Window-relative time (s)',ylabel='rad',ylim=(-.011,.011))
    f.legend(fontsize=8)
    g.set(title='Consistency: integral(dq) vs actual q change',xlabel='Window-relative time (s)',ylabel='rad')
    g.legend(fontsize=7,ncol=2)
    fig.suptitle('FL thigh: pure position sine across support changes | labels are provisional | no simulation run',fontsize=13)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=150)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--figure',type=Path)
    p.add_argument('--window',action='append',required=True,help='LABEL:START_SECONDS:END_SECONDS')
    args=p.parse_args()
    with args.input.open(newline='') as stream:
        reader=csv.DictReader(stream);columns=reader.fieldnames;rows=list(reader)
    keys=['t_mono_ms','q_set_rad','dq_set_rad_s','tau_set_nm','q_actual_rad','dq_actual_rad_s','tau_actual_nm','kp','kd']
    d={k:[float(r[k]) for r in rows] for k in keys}
    if len(rows)<10 or not all(math.isfinite(v) for xs in d.values() for v in xs):
        raise ValueError('Too few or nonfinite rows')
    t=[(v-d['t_mono_ms'][0])/1000 for v in d['t_mono_ms']]
    dt=[b-a for a,b in zip(t,t[1:])]
    if min(dt)<=0:raise ValueError('Nonmonotonic time')
    center=(max(d['q_set_rad'])+min(d['q_set_rad']))/2
    cross=[]
    for i in range(1,len(t)):
        a,b=d['q_set_rad'][i-1]-center,d['q_set_rad'][i]-center
        if a<=0<b:cross.append(t[i-1]+(t[i]-t[i-1])*(-a)/(b-a))
    if len(cross)<3:raise ValueError('No identifiable sine cycles')
    freq=1/statistics.median([b-a for a,b in zip(cross,cross[1:])])
    windows=[]
    for arg in args.window:
        label,lo,hi=arg.split(':');lo=float(lo);hi=float(hi)
        idx=[i for i,v in enumerate(t) if lo<=v<hi]
        if len(idx)<10:raise ValueError('Insufficient window data')
        ts=[t[i] for i in idx];sd={k:[xs[i] for i in idx] for k,xs in d.items()}
        if max(b-a for a,b in zip(ts,ts[1:]))>.05:raise ValueError('Window crosses a record gap')
        fits={k:fit(ts,sd[k],freq) for k in ['q_set_rad','q_actual_rad','dq_actual_rad_s','tau_set_nm','tau_actual_nm']}
        err=[x-y for x,y in zip(sd['q_set_rad'],sd['q_actual_rad'])]
        qdelta=sd['q_actual_rad'][-1]-sd['q_actual_rad'][0]
        iv=integrate(ts,sd['dq_actual_rad_s'])
        windows.append(dict(label=label,start_s=lo,end_s=hi,first_sample_s=ts[0],last_sample_s=ts[-1],samples=len(idx),
            fits=fits,amplitude_ratio=fits['q_actual_rad']['amplitude']/fits['q_set_rad']['amplitude'],
            phase_difference_deg=phase_difference(fits['q_actual_rad']['phase_deg'],fits['q_set_rad']['phase_deg']),
            error=dict(mean=statistics.fmean(err),rms=rms(err),std=statistics.pstdev(err)),
            q_peak_to_peak=max(sd['q_actual_rad'])-min(sd['q_actual_rad']),
            dq_consistency=dict(q_change_rad=qdelta,integral_dq_rad=iv,mismatch_rad=iv-qdelta,mean_dq_rad_s=statistics.fmean(sd['dq_actual_rad_s']))))
    formula=[kp*(q-actual)+kd*(v-vel) for kp,q,actual,kd,v,vel in zip(d['kp'],d['q_set_rad'],d['q_actual_rad'],d['kd'],d['dq_set_rad_s'],d['dq_actual_rad_s'])]
    residual=[x-y for x,y in zip(d['tau_set_nm'],formula)]
    report=dict(input=str(args.input),sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),bytes=args.input.stat().st_size,
        columns=columns,rows=len(rows),duration_s=t[-1],t_mono_start_ms=d['t_mono_ms'][0],
        identities=sorted({(r['leg'],r['joint']) for r in rows}),
        logging_dt_ms={str(q):quantile(dt,q)*1000 for q in [0,.01,.5,.99,1]},gaps_over_10ms=sum(v>.01 for v in dt),
        frequency_hz=freq,full_target_fit=fit(t,d['q_set_rad'],freq),
        unique_kp=sorted(set(d['kp'])),unique_kd=sorted(set(d['kd'])),unique_dq_set=sorted(set(d['dq_set_rad_s'])),
        adjacent_q_target_change_fraction=statistics.fmean(a!=b for a,b in zip(d['q_set_rad'],d['q_set_rad'][1:])),
        pd_formula=dict(max_abs=max(map(abs,residual)),rms=rms(residual)),
        total_pd_torque_range=[min(d['tau_set_nm']),max(d['tau_set_nm'])],
        estimated_torque_range=[min(d['tau_actual_nm']),max(d['tau_actual_nm'])],windows=windows,
        limitations=['Support sequence from user: ground, rack, ground then lying down; no event timestamps.',
          'A/B/C are post-hoc relatively steady windows, not independently verified contact states.',
          'One joint and host log time only; no independent force or device receive/apply time.',
          'tau_set is total computed PD torque, not feedforward; tau_actual is motor estimate.',
          'Phase is not communication delay; no simulation or hardware run by this audit.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if args.figure:draw(args.figure,t,d,report)
    print(json.dumps([{k:w[k] for k in ['label','start_s','end_s','amplitude_ratio','phase_difference_deg','error','dq_consistency']} for w in windows],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
