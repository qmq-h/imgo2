#!/usr/bin/env python3
"""Audit wide 12-axis sine logs with per-axis receive times.

Numerical core is stdlib only; matplotlib is optional. CLI direction signs are
an external coordinate hypothesis, not measured motor-side telemetry.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from audit_joint_sine_log import fit, integrate, phase_difference, quantile, rms


def bounds(v):
    return dict(min=min(v),median=statistics.median(v),max=max(v))


def inspect(path, directions, selected, windows):
    with path.open(newline='') as stream:
        reader=csv.DictReader(stream);columns=reader.fieldnames;rows=list(reader)
    joints=[k[6:] for k in columns if k.startswith('q_set_')]
    if len(joints)!=12 or len(directions)!=12:raise ValueError('Expected 12 joints and 12 direction signs')
    keys=[k for k in columns if k not in ['leg','joint']]
    d={k:[float(row[k]) for row in rows] for k in keys}
    if not all(math.isfinite(v) for xs in d.values() for v in xs):raise ValueError('Nonfinite value')
    t=[(v-d['t_mono_ms'][0])/1000 for v in d['t_mono_ms']]
    dt=[b-a for a,b in zip(t,t[1:])]
    if min(dt)<=0:raise ValueError('Nonmonotonic host time')
    qref=d['q_set_'+selected];center=(min(qref)+max(qref))/2;cross=[]
    for i in range(1,len(t)):
        a,b=qref[i-1]-center,qref[i]-center
        if a<=0<b:cross.append(t[i-1]+(t[i]-t[i-1])*(-a)/(b-a))
    freq=1/statistics.median(b-a for a,b in zip(cross,cross[1:]))
    def one_window(lo,hi):
        idx=[i for i,v in enumerate(t) if lo<=v<hi]
        ts=[t[i] for i in idx]
        if len(idx)<10 or max(b-a for a,b in zip(ts,ts[1:]))>.05:raise ValueError('Invalid window')
        axes=[]
        for j,sign in zip(joints,directions):
            q=[d['q_act_'+j][i] for i in idx];v=[d['dq_act_'+j][i] for i in idx]
            rx=[d['recv_ms_'+j][i]/1000 for i in idx]
            keep=[k for k in range(len(rx)) if k==0 or rx[k]!=rx[k-1]]
            rr=[rx[k] for k in keep];vv=[v[k] for k in keep];qq=[q[k] for k in keep]
            if any(b<=a for a,b in zip(rr,rr[1:])):raise ValueError('Nonmonotonic axis receive time')
            iv=integrate(rr,vv);delta=qq[-1]-qq[0];span=rr[-1]-rr[0]
            e=[d['q_set_'+j][i]-d['q_act_'+j][i] for i in idx]
            axes.append(dict(joint=j,direction_hypothesis=sign,mean_q=statistics.fmean(q),q_range=[min(q),max(q)],
                mean_dq=statistics.fmean(v),q_change=delta,integral_receive_time=iv,
                integral_host_time=integrate(ts,v),mismatch=iv-delta,mismatch_per_second=(iv-delta)/span,
                direction_neutral_mismatch_per_second=sign*(iv-delta)/span,
                error_mean=statistics.fmean(e),error_rms=rms(e),
                mean_total_pd=statistics.fmean(d['tau_set_'+j][i] for i in idx),
                mean_estimated_torque=statistics.fmean(d['tau_act_'+j][i] for i in idx)))
        fq=fit(ts,[d['q_act_'+selected][i] for i in idx],freq);fr=fit(ts,[qref[i] for i in idx],freq)
        return dict(start_s=lo,end_s=hi,samples=len(idx),axes=axes,selected_q_fit=fq,target_fit=fr,
                    selected_gain=fq['amplitude']/fr['amplitude'],selected_phase_deg=phase_difference(fq['phase_deg'],fr['phase_deg']))
    axis=[]
    for i,(j,sign) in enumerate(zip(joints,directions)):
        rx=d['recv_ms_'+j];diff=[b-a for a,b in zip(rx,rx[1:])]
        error=[d['tau_set_'+j][k]-(d['kp_'+j][k]*(d['q_set_'+j][k]-d['q_act_'+j][k])+d['kd_'+j][k]*(d['dq_set_'+j][k]-d['dq_act_'+j][k])) for k in range(len(t))]
        axis.append(dict(joint=j,direction_hypothesis=sign,kp_values=sorted(set(d['kp_'+j])),kd_values=sorted(set(d['kd_'+j])),
            dq_set_values=sorted(set(d['dq_set_'+j])),q_set_range=[min(d['q_set_'+j]),max(d['q_set_'+j])],
            receive_repeats=sum(v==0 for v in diff),receive_backwards=sum(v<0 for v in diff),receive_gap_ms=bounds(diff),
            snapshot_age_ms=bounds([a-b for a,b in zip(d['t_mono_ms'],rx)]),
            sdk_act_q_max_difference=max(abs(a-b) for a,b in zip(d['sdk_q_'+j],d['q_act_'+j])),
            sdk_act_dq_max_difference=max(abs(a-b) for a,b in zip(d['sdk_dq_'+j],d['dq_act_'+j])),
            sdk_conv_q_max_difference=max(abs(a-b) for a,b in zip(d['sdk_q_'+j],d['conv_q_p'+str(i)])),
            sdk_conv_dq_max_difference=max(abs(a-b) for a,b in zip(d['sdk_dq_'+j],d['conv_dq_p'+str(i)])),
            pd_formula_max_difference=max(map(abs,error)),estimated_torque_range=[min(d['tau_act_'+j]),max(d['tau_act_'+j])]))
    ws=[dict(label=label,**one_window(lo,hi)) for label,lo,hi in windows]
    report=dict(input=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,
        rows=len(rows),columns=len(columns),joint_order=joints,directions_hypothesis=directions,selected=selected,
        duration_s=t[-1],host_start_ms=d['t_mono_ms'][0],logging_dt_ms=bounds([v*1000 for v in dt]),
        logging_gaps_over_10ms=sum(v>.01 for v in dt),frequency_hz=freq,target_fit=fit(t,qref,freq),
        global_recv_vs_latest_axis_max_difference=max(abs(d['recv_mono_ms'][k]-max(d['recv_ms_'+j][k] for j in joints)) for k in range(len(t))),
        global_receive_before_send_fraction=statistics.fmean(a<b for a,b in zip(d['recv_mono_ms'],d['send_mono_ms'])),
        send_minus_previous_global_recv_ms=bounds([a-b for a,b in zip(d['send_mono_ms'],d['recv_mono_ms'])]),
        axes=axis,windows=ws,full=one_window(0,t[-1]+1e-6))
    return report,(t,d)


def plots(overview,consistency,reports,records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,2,figsize=(15,10),constrained_layout=True)
    for col,(r,(t,d)) in enumerate(zip(reports,records)):
        j=r['selected'];a,b,c=axs[:,col]
        a.plot(t,d['q_set_'+j],color='tab:orange',label='FL thigh target')
        a.plot(t,d['q_act_'+j],color='tab:blue',label='FL thigh feedback')
        for w in r['windows']:a.axvspan(w['start_s'],w['end_s'],alpha=.1,color='green')
        a.set(title=f"{'g1 ground / transition' if col==0 else 'g2 rack'} | all 12 axes Kp25 Kd0.8",ylabel='rad',ylim=(.73,.88));a.legend(fontsize=8)
        b.plot(t,d['tau_act_'+j],label='FL thigh estimate')
        b.plot(t,d['tau_act_BR_shank'],label='BR shank estimate')
        b.set(title='Support changes visible across multiple joints',ylabel='N m',ylim=(-.6,10));b.legend(fontsize=8)
        q=d['q_act_'+j];rx=[v/1000 for v in d['recv_ms_'+j]];v=d['dq_act_'+j];iv=[0.]
        for k in range(1,len(t)):iv.append(iv[-1]+(rx[k]-rx[k-1])*(v[k]+v[k-1])/2)
        c.plot(t,[x-q[0] for x in q],label='q - initial q')
        c.plot(t,iv,label='integral(dq), own receive clock')
        c.set(title='Position / velocity consistency',ylabel='rad',ylim=(-.18,.15));c.legend(fontsize=8)
        for ax in axs[:,col]:ax.grid(alpha=.25);ax.set_xlabel('Time from each log start (s)')
    fig.suptitle('23:18 wide logs | separate initial phases | receive age is NOT communication latency',fontsize=13)
    overview.parent.mkdir(parents=True,exist_ok=True);fig.savefig(overview,dpi=150);plt.close(fig)
    r=reports[1];t,d=records[1];fig,axs=plt.subplots(4,3,figsize=(15,11),constrained_layout=True)
    for ax,j in zip(axs.flat,r['joint_order']):
        q=d['q_act_'+j];rx=[v/1000 for v in d['recv_ms_'+j]];v=d['dq_act_'+j];iv=[0.]
        for k in range(1,len(t)):iv.append(iv[-1]+(rx[k]-rx[k-1])*(v[k]+v[k-1])/2)
        ax.plot(t,[x-q[0] for x in q],label='q change',color='tab:blue')
        ax.plot(t,iv,label='integral dq (receive time)',color='tab:purple')
        ax.set(title=j,xlabel='Log-relative s',ylabel='rad',ylim=(-.16,.16));ax.grid(alpha=.25)
    axs[0,0].legend(fontsize=7)
    fig.suptitle('Rack g2: all 12 joint position changes vs recorded velocity integrals',fontsize=14)
    consistency.parent.mkdir(parents=True,exist_ok=True);fig.savefig(consistency,dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ground',type=Path,required=True);p.add_argument('--rack',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--overview',type=Path);p.add_argument('--consistency',type=Path)
    p.add_argument('--directions',required=True,help='12 comma-separated signs in logged joint order')
    args=p.parse_args();directions=[int(v) for v in args.directions.split(',')]
    if any(v not in [-1,1] for v in directions):raise ValueError('Direction must be +/-1')
    selected='FL_thigh'
    r1,d1=inspect(args.ground,directions,selected,[('g1_early',0,4),('g1_late',8,12)])
    r2,d2=inspect(args.rack,directions,selected,[('g2_rack',0,4),('g2_rack_long',0,6)])
    result=dict(records=[r1,r2],limitations=[
        'Recorded gains only: no Kp60/Kd2 rows or pre-test baseline.',
        'g1 early/late split follows joint/load changes; user says lifting during g1 is possible, exact event remains unknown.',
        'Direction-neutral values are coordinate checks using workspace signs, not independent rotor measurements.',
        'User confirms sdk fields are joint-side converted values; sdk/conv/act equality cannot localize an upstream discrepancy.',
        'Receive timestamps are host endpoints, not device samples or command-application acknowledgements.',
        'No production control edits, hardware run, simulation or training.'])
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    if args.overview and args.consistency:plots(args.overview,args.consistency,[r1,r2],[d1,d2])
    for r in [r1,r2]:
        print(json.dumps(dict(input=r['input'],rows=r['rows'],duration_s=r['duration_s'],windows=[dict(label=w['label'],gain=w['selected_gain'],phase=w['selected_phase_deg']) for w in r['windows']]),ensure_ascii=False))


if __name__=='__main__':main()
