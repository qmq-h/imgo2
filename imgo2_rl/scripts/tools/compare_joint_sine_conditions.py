#!/usr/bin/env python3
"""Compare ground/rack single-joint sine logs; stdlib analysis, optional plots."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from audit_joint_sine_log import fit, integrate, phase_difference, quantile, rms


def analyze(path, label, lo, hi):
    with path.open(newline='') as stream:
        reader=csv.DictReader(stream);columns=reader.fieldnames;rows=list(reader)
    keys=['t_mono_ms','q_set_rad','dq_set_rad_s','tau_set_nm','q_actual_rad','dq_actual_rad_s','tau_actual_nm','kp','kd']
    d={k:[float(r[k]) for r in rows] for k in keys}
    if len(rows)<10 or not all(math.isfinite(v) for xs in d.values() for v in xs):raise ValueError('Invalid rows')
    t=[(v-d['t_mono_ms'][0])/1000 for v in d['t_mono_ms']]
    dt=[b-a for a,b in zip(t,t[1:])]
    if min(dt)<=0:raise ValueError('Nonmonotonic time')
    center=(max(d['q_set_rad'])+min(d['q_set_rad']))/2
    crosses=[]
    for i in range(1,len(t)):
        a,b=d['q_set_rad'][i-1]-center,d['q_set_rad'][i]-center
        if a<=0<b:crosses.append(t[i-1]+(t[i]-t[i-1])*(-a)/(b-a))
    if len(crosses)<3:raise ValueError('Insufficient sine cycles')
    frequency=1/statistics.median(b-a for a,b in zip(crosses,crosses[1:]))
    fits_keys=['q_set_rad','q_actual_rad','dq_actual_rad_s','tau_set_nm','tau_actual_nm']
    def window(start,end):
        idx=[i for i,v in enumerate(t) if start<=v<end]
        ts=[t[i] for i in idx];sd={k:[xs[i] for i in idx] for k,xs in d.items()}
        if len(ts)<10 or max(b-a for a,b in zip(ts,ts[1:]))>.05:raise ValueError('Insufficient or gapped window')
        fits={k:fit(ts,sd[k],frequency) for k in fits_keys}
        err=[a-b for a,b in zip(sd['q_set_rad'],sd['q_actual_rad'])]
        qdelta=sd['q_actual_rad'][-1]-sd['q_actual_rad'][0]
        integral=integrate(ts,sd['dq_actual_rad_s'])
        return dict(start_s=start,end_s=end,first_sample_s=ts[0],last_sample_s=ts[-1],samples=len(ts),fits=fits,
            amplitude_ratio=fits['q_actual_rad']['amplitude']/fits['q_set_rad']['amplitude'],
            q_phase_difference_deg=phase_difference(fits['q_actual_rad']['phase_deg'],fits['q_set_rad']['phase_deg']),
            error_mean=statistics.fmean(err),error_rms=rms(err),q_peak_to_peak=max(sd['q_actual_rad'])-min(sd['q_actual_rad']),
            mean_dq=statistics.fmean(sd['dq_actual_rad_s']),q_change=qdelta,dq_integral=integral,integral_mismatch=integral-qdelta,
            mean_p_torque=statistics.fmean(kp*e for kp,e in zip(sd['kp'],err)),
            mean_d_torque=statistics.fmean(kd*(v-w) for kd,v,w in zip(sd['kd'],sd['dq_set_rad_s'],sd['dq_actual_rad_s'])),
            mean_total_pd_torque=statistics.fmean(sd['tau_set_nm']),mean_estimated_torque=statistics.fmean(sd['tau_actual_nm']))
    pd=[kp*(q-a)+kd*(v-b) for kp,q,a,kd,v,b in zip(d['kp'],d['q_set_rad'],d['q_actual_rad'],d['kd'],d['dq_set_rad_s'],d['dq_actual_rad_s'])]
    dif=[a-b for a,b in zip(d['tau_set_nm'],pd)]
    vals=sorted(set(d['q_actual_rad']));levels=[b-a for a,b in zip(vals,vals[1:]) if b>a]
    report=dict(input=str(path),label=label,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,
        rows=len(rows),columns=columns,identities=sorted({(r['leg'],r['joint']) for r in rows}),duration_s=t[-1],
        dt_ms={str(q):quantile(dt,q)*1000 for q in [0,.01,.5,.99,1]},gaps_over_10ms=sum(x>.01 for x in dt),
        unique_kp=sorted(set(d['kp'])),unique_kd=sorted(set(d['kd'])),unique_dq_set=sorted(set(d['dq_set_rad_s'])),
        frequency_hz=frequency,full_target_fit=fit(t,d['q_set_rad'],frequency),
        q_range=[min(vals),max(vals)],q_distinct_logged_values=len(vals),min_positive_logged_q_step=min(levels) if levels else None,
        velocity_range=[min(d['dq_actual_rad_s']),max(d['dq_actual_rad_s'])],
        total_pd_torque_range=[min(d['tau_set_nm']),max(d['tau_set_nm'])],estimated_torque_range=[min(d['tau_actual_nm']),max(d['tau_actual_nm'])],
        pd_formula=dict(max_abs=max(map(abs,dif)),rms=rms(dif)),primary=window(lo,hi),
        sensitivity_windows=[window(x,hi) for x in [lo+1,lo+2] if hi-x>=2])
    return report,(t,d)


def plot(path,reports,records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(5,2,figsize=(15,14),constrained_layout=True)
    colors=['tab:blue','tab:green']
    qlo=min(min(d['q_actual_rad']+d['q_set_rad']) for t,d in records)-.005
    qhi=max(max(d['q_actual_rad']+d['q_set_rad']) for t,d in records)+.005
    vlim=max(max(map(abs,d['dq_actual_rad_s'])) for t,d in records)*1.1
    torque_hi=max(max(d['tau_set_nm']+d['tau_actual_nm']) for t,d in records)*1.06
    for col,(r,(t,d),color) in enumerate(zip(reports,records,colors)):
        w=r['primary'];idx=[i for i,v in enumerate(t) if w['start_s']<=v<w['end_s']]
        tt=[t[i] for i in idx]
        a,b,c,e,f=axs[:,col]
        a.plot(t,d['q_set_rad'],color='tab:orange',label='q target')
        a.plot(t,d['q_actual_rad'],color=color,label='q feedback')
        a.set(title=f"{r['label']} | absolute q",ylabel='rad',ylim=(qlo,qhi))
        a.legend(fontsize=8)
        b.plot(tt,[d['q_set_rad'][i]-w['fits']['q_set_rad']['mean'] for i in idx],color='tab:orange',label='target - mean')
        b.plot(tt,[d['q_actual_rad'][i]-w['fits']['q_actual_rad']['mean'] for i in idx],color=color,label='feedback - mean')
        b.set(title=f"Fundamental gain: {w['amplitude_ratio']*100:.2f}% | no phase alignment",ylabel='rad',ylim=(-.011,.011))
        b.legend(fontsize=8)
        c.plot(t,d['dq_actual_rad_s'],color=color,lw=.65,alpha=.7,label='dq feedback')
        c.axhline(0,color='black',lw=.7)
        c.axhline(w['mean_dq'],color='red',lw=1,label=f"mean {w['mean_dq']:.5f}")
        c.set(title='Velocity target = 0',ylabel='rad/s',ylim=(-vlim,vlim));c.legend(fontsize=8)
        e.plot(t,d['tau_set_nm'],label='total computed PD torque',color='tab:red',lw=.8)
        e.plot(t,d['tau_actual_nm'],label='motor estimated torque',color=color,lw=.8)
        e.set(title='Torque (estimated, not independently calibrated)',ylabel='N m',ylim=(-.05,torque_hi));e.legend(fontsize=8)
        integ=[0.]
        for k in range(1,len(idx)):
            integ.append(integ[-1]+(tt[k]-tt[k-1])*(d['dq_actual_rad_s'][idx[k]]+d['dq_actual_rad_s'][idx[k-1]])/2)
        f.plot(tt,[d['q_actual_rad'][i]-d['q_actual_rad'][idx[0]] for i in idx],color=color,label='q - initial q')
        f.plot(tt,integ,color='tab:purple',label='integral(dq)')
        f.set(title='Kinematic consistency on common 0-6 s window',ylabel='rad',ylim=(-.14,.025));f.legend(fontsize=8)
        for ax in axs[:,col]:
            ax.grid(alpha=.25);ax.set_xlabel('Time from each log start (s)')
    fig.suptitle('FL thigh | ground vs rack | independent sine phases; same row uses same y scale | no Gazebo run',fontsize=13)
    path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ground',type=Path,required=True);p.add_argument('--rack',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--figure',type=Path)
    p.add_argument('--start',type=float,default=0);p.add_argument('--end',type=float,default=6)
    args=p.parse_args();reports=[];records=[]
    for path,label in [(args.ground,'Ground (g1)'),(args.rack,'Rack (g2)')]:
        report,record=analyze(path,label,args.start,args.end);reports.append(report);records.append(record)
    output=dict(records=reports,provenance='User confirms g1 normal four-foot standing with noticeable softening, g2 rack; no re-zero or other joint target change between files.',
        limitations=['One trial per condition and one recorded joint only; fixture/foot forces and all joint states unmeasured.',
                     'Each log has its own sine phase; no cross-log pointwise RMSE or time shifts.',
                     'tau_set is total computed PD torque, not feedforward; feedback torque is an estimate.',
                     'No device receive/apply timing, valid/error, transition baseline or independent force reference.',
                     'A small ground response under contact constraints is not an isolated motor capability measurement.',
                     'No hardware, simulation or training run.'])
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    if args.figure:plot(args.figure,reports,records)
    for r in reports:
        print(json.dumps(dict(label=r['label'],rows=r['rows'],duration_s=r['duration_s'],primary=r['primary']),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
