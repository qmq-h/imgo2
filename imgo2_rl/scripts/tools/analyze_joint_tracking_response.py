#!/usr/bin/env python3
"""Same-target joint tracking, next-record and pre-overwrite response comparison.

Stdlib calculations, optional matplotlib. Uses existing runs; never commands
hardware/simulation and never fits a time shift. IMU is not analyzed.
"""
import argparse
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st

from compare_pd_torque_time import interp, before
from audit_paired_obs_replay import rms, quantile

LEGS=('FL','FR','RL','RR')
KINDS=('hip','thigh','shank')
GROUPS={k:tuple(range(j,12,3)) for j,k in enumerate(KINDS)}
DEG=180/math.pi

def read(path):
    with path.open() as f:return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def scalar_stats(v):
    return {'mean':st.mean(v),'rms':rms(v),'abs_p95':quantile(list(map(abs,v)),.95),'abs_max':max(map(abs,v))}

def load_replay(path,real):
    meta=json.loads((path/'metadata.json').read_text())
    assert meta['completed'] and meta['production_files_unchanged']
    targets=read(path/'targets.csv');sent=read(path/'published.csv')
    assert len(targets)==len(sent) and [int(r['index']) for r in sent]==list(range(len(targets)))
    start=meta['source_start_s'];end=start+meta['duration_s'];rt=[r['t_s'] for r in real]
    expected=[real[before(rt,start)]]+[r for r in real if start<r['t_s']<=end]
    assert len(targets)==len(expected)
    err=max(abs(t[f'{a}_{i:02}']-r[f'{b}_{i:02}']) for t,r in zip(targets,expected)
            for a,b in [('q','q_des'),('dq','dq_des'),('tau','tau_ff'),('kp','kp'),('kd','kd')] for i in range(12))
    terr=max(abs(t['t_s']-(0 if j==0 else r['t_s']-start)) for j,(t,r) in enumerate(zip(targets,expected)))
    assert err<1e-12 and terr<1e-12
    rows=read(path/'replay.csv')
    for r in rows:r['t_s']+=start
    assert all(b['t_s']>a['t_s'] for a,b in zip(rows,rows[1:]))
    return rows,{'metadata':meta,'targets':len(targets),'published':len(sent),'max_target_error':err,
                 'max_target_time_error_s':terr,'controller_feedback_used':True,
                 'sha256':{n:sha(path/n) for n in ('replay.csv','targets.csv','metadata.json','published.csv')}}

def sample(real,sim,tt,j):
    r=real[j];t=r['t_s'];out={'t_s':t,'tick':r['tick']}
    for i in range(12):
        for key in ('q_des','dq_des','tau_ff','kp','kd'):out[f'{key}_{i:02}']=r[f'{key}_{i:02}']
        for case in ('real','gazebo'):
            q=r[f'q_{i:02}'] if case=='real' else interp(sim,tt,t,f'controller_q_{i:02}')
            dq=r[f'dq_{i:02}'] if case=='real' else interp(sim,tt,t,f'controller_dq_{i:02}')
            tau=r[f'tau_{i:02}'] if case=='real' else interp(sim,tt,t,f'controller_tau_{i:02}')
            e=r[f'q_des_{i:02}']-q;de=r[f'dq_des_{i:02}']-dq
            for key,v in {'q':q,'dq':dq,'tau':tau,'error':e,'P':r[f'kp_{i:02}']*e,
                          'D':r[f'kd_{i:02}']*de,'nominal':r[f'kp_{i:02}']*e+r[f'kd_{i:02}']*de+r[f'tau_ff_{i:02}']}.items():
                out[f'{case}_{key}_{i:02}']=v
        out[f'real_feedback_age_ms_{i:02}']=r[f'age_{i:02}']
    return out

def case_stats(rows,ids,case):
    result={k:scalar_stats([r[f'{case}_{k}_{i:02}'] for r in rows for i in ids])
            for k in ('error','dq','P','D','nominal','tau')}
    result['error_rms_deg']=result['error']['rms']*DEG
    result['error_mean_deg_by_joint']={str(i):st.mean(r[f'{case}_error_{i:02}'] for r in rows)*DEG for i in ids}
    result['within_1deg_fraction']=st.mean(abs(r[f'{case}_error_{i:02}'])<=math.pi/180 for r in rows for i in ids)
    result['feedback_abs_over_23_7_count']=sum(abs(r[f'{case}_tau_{i:02}'])>23.7 for r in rows for i in ids)
    return result

def make_events(real,sim,tt,lo,hi):
    events=[];updates=[j for j in range(1,len(real)) if any(abs(real[j][f'q_des_{i:02}']-real[j-1][f'q_des_{i:02}'])>1e-7 for i in range(12))]
    for j,end in zip(updates,updates[1:]):
        t=real[j]['t_s']
        if not lo<=t<hi or real[end]['t_s']>hi or j+1>=end:continue
        if not .008<real[end]['t_s']-t<.04:continue
        current=sample(real,sim,tt,j);nxt=sample(real,sim,tt,j+1);last=sample(real,sim,tt,end-1)
        for i in range(12):
            if abs(real[j][f'q_des_{i:02}']-real[j-1][f'q_des_{i:02}'])<=1e-7:continue
            e={'joint':i,'t_s':t,'tick':real[j]['tick'],'command_hold_ms':(real[end]['t_s']-t)*1000,
               'next_record_ms':(real[j+1]['t_s']-t)*1000,'last_record_ms':(real[end-1]['t_s']-t)*1000,
               'target':current[f'q_des_{i:02}'],'previous_target':real[j-1][f'q_des_{i:02}'],
               'target_step':current[f'q_des_{i:02}']-real[j-1][f'q_des_{i:02}'],
               'dq_des':current[f'dq_des_{i:02}'],'tau_ff':current[f'tau_ff_{i:02}'],
               'kp':current[f'kp_{i:02}'],'kd':current[f'kd_{i:02}'],
               'real_age0_ms':real[j][f'age_{i:02}'],'real_age_next_ms':real[j+1][f'age_{i:02}'],
               'real_age_last_ms':real[end-1][f'age_{i:02}']}
            for c in ('real','gazebo'):
                for tag,s in [('initial',current),('next',nxt),('last',last)]:
                    for k in ('q','dq','tau','error','nominal'):
                        e[f'{c}_{tag}_{k}']=s[f'{c}_{k}_{i:02}']
                for tag in ('next','last'):
                    e[f'{c}_{tag}_abs_error_change_deg']=(abs(e[f'{c}_{tag}_error'])-abs(e[f'{c}_initial_error']))*DEG
                    e[f'{c}_{tag}_q_change']=e[f'{c}_{tag}_q']-e[f'{c}_initial_q']
            events.append(e)
    return events

def event_stats(events,ids):
    rr=[r for r in events if r['joint'] in ids]
    return {'joint_events':len(rr),
        'command_hold_ms':scalar_stats([r['command_hold_ms'] for r in rr]),
        'next_record_ms':scalar_stats([r['next_record_ms'] for r in rr]),
        'last_record_ms':scalar_stats([r['last_record_ms'] for r in rr]),
        'cases':{c:{tag:{'error_rms_deg':rms([r[f'{c}_{tag}_error'] for r in rr])*DEG,
                        'abs_error_change_mean_deg':st.mean(r[f'{c}_{tag}_abs_error_change_deg'] for r in rr),
                        'q_change_rms_deg':rms([r[f'{c}_{tag}_q_change'] for r in rr])*DEG,
                        'feedback_tau_rms_Nm':rms([r[f'{c}_{tag}_tau'] for r in rr]),
                        'initial_nominal_vs_feedback_rmse_Nm':rms([r[f'{c}_initial_nominal']-r[f'{c}_{tag}_tau'] for r in rr]),
                        'same_record_nominal_vs_feedback_rmse_Nm':rms([r[f'{c}_{tag}_nominal']-r[f'{c}_{tag}_tau'] for r in rr])}
                    for tag in ('next','last')} for c in ('real','gazebo')}}

def analyze(real,sim,lo,hi):
    tt=[r['t_s'] for r in sim];indices=[j for j,r in enumerate(real) if lo<=r['t_s']<hi]
    assert indices and tt[0]<lo<hi<tt[-1]
    assert all(real[j]['mode']==3 and real[j]['valid_mask']==4095 and real[j]['err_mask']==0 for j in indices)
    rows=[sample(real,sim,tt,j) for j in indices]
    assert all(r[f'kp_{i:02}']==25 and r[f'kd_{i:02}']==.8 for r in rows for i in range(12))
    events=make_events(real,sim,tt,lo,hi)
    result={'window_s':[lo,hi],'source_records':len(rows),'event_joint_records':len(events),
            'groups':{k:{c:case_stats(rows,ids,c) for c in ('real','gazebo')} for k,ids in GROUPS.items()},
            'joints':{LEGS[i//3]+'_'+KINDS[i%3]:{c:case_stats(rows,(i,),c) for c in ('real','gazebo')} for i in range(12)},
            'next_record_and_pre_overwrite':{k:event_stats(events,ids) for k,ids in GROUPS.items()},
            'command_ranges':{k:[min(r[f'{k}_{i:02}'] for r in rows for i in range(12)),max(r[f'{k}_{i:02}'] for r in rows for i in range(12))] for k in ('dq_des','tau_ff','kp','kd')},
            'real_feedback_age_ms':scalar_stats([r[f'real_feedback_age_ms_{i:02}'] for r in rows for i in range(12)])}
    result['freshness_matched_events']={'definition':'Both initial and selected later real feedback age <=2ms, age difference <=1ms; subset only, no time shifting or synchronization claim','groups':{}}
    for k,ids in GROUPS.items():
        result['freshness_matched_events']['groups'][k]={}
        for tag in ('next','last'):
            ev=[e for e in events if e['joint'] in ids and max(e['real_age0_ms'],e[f'real_age_{tag}_ms'])<=2 and abs(e['real_age0_ms']-e[f'real_age_{tag}_ms'])<=1]
            if ev:
                es=event_stats(ev,ids)
                result['freshness_matched_events']['groups'][k][tag]={'joint_events':len(ev),'cases':{c:es['cases'][c][tag] for c in ('real','gazebo')}}
    # High/low demand is defined by simulated nominal torque, avoiding a
    # mechanical identity between the real position error and its own P torque.
    result['sim_demand_strata']={}
    result['both_slow_shank_samples']={}
    for i in GROUPS['shank']:
        rr=[r for r in rows if abs(r[f'real_dq_{i:02}'])<1 and abs(r[f'gazebo_dq_{i:02}'])<1]
        result['both_slow_shank_samples'][LEGS[i//3]]={'samples':len(rr),
            'definition':'Both observed |dq|<1rad/s; not a stance/contact/static classification',
            **{c:case_stats(rr,(i,),c) for c in ('real','gazebo')}}
    for k,ids in GROUPS.items():
        pairs=[(r,i) for r in rows for i in ids];threshold=quantile([abs(r[f'gazebo_nominal_{i:02}']) for r,i in pairs],.5)
        result['sim_demand_strata'][k]={'median_sim_abs_nominal_Nm':threshold}
        for label,test in [('lower_half',lambda v:v<=threshold),('upper_half',lambda v:v>threshold)]:
            selected=[(r,i) for r,i in pairs if test(abs(r[f'gazebo_nominal_{i:02}']))]
            result['sim_demand_strata'][k][label]={'samples':len(selected),**{c:{
                'error_rms_deg':rms([r[f'{c}_error_{i:02}'] for r,i in selected])*DEG,
                'nominal_rms_Nm':rms([r[f'{c}_nominal_{i:02}'] for r,i in selected]),
                'feedback_rms_Nm':rms([r[f'{c}_tau_{i:02}'] for r,i in selected])} for c in ('real','gazebo')}}
    return result,rows,events

def write_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)

def plot(prefix,data,events):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors={'real':'#0077bb','gazebo':'#ee7733'}
    summary,rows,_=data['autonomous'];lo,hi=summary['window_s']
    def save(fig,suffix):fig.savefig(str(prefix)+'_'+suffix+'.png',dpi=150,bbox_inches='tight');plt.close(fig)
    # Per-leg charts explicitly expose all commanded fields and later states.
    for leg,name in enumerate(LEGS):
        fig,ax=plt.subplots(5,3,figsize=(16,14),layout='constrained');t=[r['t_s']-lo for r in rows]
        for col,kind in enumerate(KINDS):
            i=leg*3+col
            ax[0,col].step(t,[r[f'q_des_{i:02}']*DEG for r in rows],where='post',color='black',ls='--',lw=1,label='same q_des')
            ax[2,col].step(t,[r[f'dq_des_{i:02}'] for r in rows],where='post',color='black',ls='--',lw=1,label='same dq_des')
            ax[3,col].step(t,[r[f'tau_ff_{i:02}'] for r in rows],where='post',color='black',ls='--',lw=1,label='same tau_ff')
            for c in colors:
                for row,key,scale in [(0,'q',DEG),(1,'error',DEG),(2,'dq',1),(3,'nominal',1),(4,'tau',1)]:
                    ax[row,col].plot(t,[r[f'{c}_{key}_{i:02}']*scale for r in rows],color=colors[c],label=c,lw=.9)
            ax[0,col].set_title(kind)
            for row,label in enumerate(('Position / target (deg)','q_des - q (deg)','Velocity / target (rad/s)','Calculated PD + FF (Nm)','Feedback torque estimate (Nm)')):
                ax[row,col].set_ylabel(label);ax[row,col].grid(alpha=.2)
                ax[row,col].set_xlabel('Time from '+str(lo)+'s (s)')
        for row in (0,2,3,4):ax[row,0].legend(fontsize=8,ncol=3)
        fig.suptitle(name+' | identical real targets and P25/D0.8 | controller feedback, no fitted delay\n'+
                     'tau_ff is the commanded feedforward; PD is calculated, feedback torque is estimated')
        save(fig,name)
    fig,ax=plt.subplots(4,2,figsize=(13,10),layout='constrained');t=[r['t_s']-lo for r in rows]
    for col,i in enumerate((5,11)):
        ax[0,col].step(t,[r[f'q_des_{i:02}']*DEG for r in rows],where='post',color='black',ls='--',lw=1,label='same q_des')
        ax[1,col].step(t,[r[f'dq_des_{i:02}'] for r in rows],where='post',color='black',ls='--',lw=1,label='same dq_des = 0')
        ax[2,col].step(t,[r[f'tau_ff_{i:02}'] for r in rows],where='post',color='black',ls='--',lw=1,label='same tau_ff = 0')
        for c in colors:
            for row,key,scale in [(0,'q',DEG),(1,'dq',1),(2,'nominal',1),(3,'tau',1)]:
                ax[row,col].plot(t,[r[f'{c}_{key}_{i:02}']*scale for r in rows],color=colors[c],label=c,lw=1)
        ax[0,col].set_title(LEGS[i//3]+' shank')
        for row,label in enumerate(('Position / target (deg)','Velocity / target (rad/s)','Calculated PD + FF (Nm)','Feedback torque estimate (Nm)')):
            ax[row,col].set(xlabel='Time from '+str(lo)+'s (s)',ylabel=label);ax[row,col].grid(alpha=.2)
    for row in range(4):ax[row,0].legend(fontsize=8,ncol=3)
    fig.suptitle('FR / RR shank | same original targets and P25/D0.8 | real vs Gazebo\nNo time shift; torque feedback estimated, calculated PD is not a directly measured firmware command')
    save(fig,'shank_detail')
    fig,ax=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    x=np.arange(12)
    for col,(term,title) in enumerate([('error_rms_deg','Position tracking error RMS (deg)'),('nominal','Calculated PD torque RMS (Nm)'),('tau','Feedback torque RMS (Nm)')]):
        for c,off in [('real',-.18),('gazebo',.18)]:
            vals=[]
            for i in range(12):
                v=summary['joints'][LEGS[i//3]+'_'+KINDS[i%3]][c]
                vals.append(v[term] if term=='error_rms_deg' else v[term]['rms'])
            ax[0,col].bar(x+off,vals,.36,label=c,color=colors[c])
        ax[0,col].set(title=title,xticks=x,xticklabels=[LEGS[i//3]+' '+KINDS[i%3] for i in range(12)])
        ax[0,col].tick_params(axis='x',rotation=65);ax[0,col].grid(axis='y',alpha=.2)
    for col,k in enumerate(KINDS):
        for c,off in [('real',-.18),('gazebo',.18)]:
            vals=[summary['next_record_and_pre_overwrite'][k]['cases'][c][tag]['error_rms_deg'] for tag in ('next','last')]
            ax[1,col].bar(np.arange(2)+off,vals,.36,label=c,color=colors[c])
        ax[1,col].set(title=k+' | residual target error',xticks=[0,1],xticklabels=['Next record\n~5 ms','Before next target\n~15 ms'],ylabel='RMS (deg)');ax[1,col].grid(axis='y',alpha=.2)
    handles,labels=ax[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.035),ncol=2)
    fig.suptitle('Same autonomous targets | real vs Gazebo | source 38.7-41.2s\nNext record is not a settled response or a measured wire-to-state latency')
    save(fig,'summary')
    # Three representative command events: one per joint class, with a fixed
    # selection rule; avoid hand-picking the biggest mismatch.
    fig,ax=plt.subplots(5,3,figsize=(15,14),layout='constrained')
    chosen=[]
    for col,k in enumerate(KINDS):
        ev=[e for e in events if e['joint']==9+col and abs(e['target_step'])*DEG>=.2
            and max(e['real_age0_ms'],e['real_age_next_ms'])<=2 and abs(e['real_age0_ms']-e['real_age_next_ms'])<=1]
        ev=sorted(ev,key=lambda e:abs(e['target_step']));e=ev[len(ev)//2];chosen.append(e)
        tt=e['t_s'];i=e['joint'];sel=[r for r in rows if tt-.02<=r['t_s']<=tt+.06]
        t=[(r['t_s']-tt)*1000 for r in sel]
        ax[0,col].step(t,[r[f'q_des_{i:02}']*DEG for r in sel],where='post',color='black',ls='--',label='q_des')
        ax[1,col].step(t,[r[f'dq_des_{i:02}'] for r in sel],where='post',color='black',ls='--',label='dq_des')
        ax[2,col].step(t,[r[f'tau_ff_{i:02}'] for r in sel],where='post',color='black',ls='--',label='tau_ff')
        ax[4,col].plot(t,[r[f'real_feedback_age_ms_{i:02}'] for r in sel],'.-',color=colors['real'],lw=1,ms=3)
        for c in colors:
            for row,key,scale in [(0,'q',DEG),(1,'dq',1),(2,'nominal',1),(3,'tau',1)]:
                ax[row,col].plot(t,[r[f'{c}_{key}_{i:02}']*scale for r in sel],'.-',color=colors[c],label=c,lw=1,ms=3)
            for tag,marker in [('initial','o'),('next','s'),('last','D')]:
                x0=0 if tag=='initial' else e['next_record_ms'] if tag=='next' else e['last_record_ms']
                ax[0,col].scatter([x0],[e[f'{c}_{tag}_q']*DEG],marker=marker,color=colors[c],s=45,zorder=6)
        ax[0,col].set_title('RR '+k+' | command at '+f'{tt:.6f}s')
        for row,label in enumerate(('Position (deg)','Velocity (rad/s)','Calculated PD + FF (Nm)','Feedback torque estimate (Nm)','Real host receive age (ms)')):
            ax[row,col].axvline(0,color='gray',lw=.8)
            ax[row,col].axvline(e['command_hold_ms'],color='gray',ls=':',lw=.8)
            ax[row,col].set(xlabel='Time from selected command (ms)',ylabel=label);ax[row,col].grid(alpha=.2)
    ax[0,0].legend(fontsize=8);fig.suptitle('Representative RR command events | o: same-row state; square: next record; diamond: pre-overwrite\nMedian |step| >=0.2 deg among initial/next age <=2ms, difference <=1ms; later steps are NEW targets')
    save(fig,'next_feedback');write_csv(Path(str(prefix)+'_selected_events.csv'),chosen)
    if 'later_forward' in data:
        later=data['later_forward'][0];fig,ax=plt.subplots(1,3,figsize=(13,4),layout='constrained')
        for axis,(term,title) in zip(ax,[('error_rms_deg','Tracking error RMS (deg)'),('nominal','Calculated PD torque RMS (Nm)'),('tau','Feedback torque RMS (Nm)')]):
            for case,off in [('autonomous',-.18),('later_forward',.18)]:
                d=data[case][0];vals=[d['groups'][k]['real'][term] if term=='error_rms_deg' else d['groups'][k]['real'][term]['rms'] for k in KINDS]
                axis.bar(np.arange(3)+off,vals,.36,label=case)
            axis.set(title=title,xticks=range(3),xticklabels=KINDS);axis.grid(axis='y',alpha=.2)
        ax[0].legend();fig.suptitle('Real early vs later forward | targets differ; rope timing/force unknown; later is NOT a controlled load-only test')
        save(fig,'early_later')

def main(a):
    real=read(a.source/'robot.csv')
    for r in real:r['t_s']=r['t_mono_ms']/1000
    assert all(b['t_s']>r['t_s'] for r,b in zip(real,real[1:]))
    sim,checks=load_replay(a.replay,real)
    data={'autonomous':analyze(real,sim,a.lo,a.hi)}
    report={'source':str(a.source),'replay':str(a.replay),'checks':{'autonomous':checks},
        'formula':'error=q_des-q; P=kp*error; D=kd*(dq_des-dq); nominal=P+D+tau_ff',
        'units':{'q':'rad in CSV, degrees in figures','dq':'rad/s','torque':'Nm'},
        'sampling':'Each actual real robot log timestamp; identical target zero-order hold. Simulated controller feedback interpolated at same timeline. No fitted time shift.',
        'limitations':['Same-row real feedback is read earlier than submitting its new target. Next record is about 5ms later, not a fully settled target response.',
            'Per-motor feedback age is host receive age rounded to ms, not a sensor-source or target-application timestamp; it is not subtracted to invent synchronization.',
            'Pre-overwrite uses last available real log record before next target (typically about15ms), not a response after holding one target forever.',
            'The protocol commands q_des/dq_des/tau_ff/kp/kd. Nominal total PD is calculated, not an independently logged firmware total torque command.',
            'Real feedback torque is estimated and Gazebo effort is simulated; neither is independent torque calibration. Raw physics GetVelocity/GetForce are not used.',
            'Joint targets/gains/cadence are identical, but real full-body initial state and contact forces are not captured. Torque demand strata are descriptive, not measured foot-load strata.',
            'Finite-gain PD can have tracking error under load in both systems; larger error alone does not identify a torque ceiling or prove insufficient motor capability.',
            'Later real segment has changed targets and uncertain rope constraint; cannot use it to identify a pure unloading effect.'],
        'sha256':{n:sha(a.source/n) for n in ('robot.csv','meta.json')}}
    if a.late_replay:
        late,ck=load_replay(a.late_replay,real);report['checks']['later_forward']=ck
        data['later_forward']=analyze(real,late,117.6,120.1)
    for label,(summary,rows,events) in data.items():
        report[label]=summary;write_csv(Path(str(a.output_prefix)+'_'+label+'_traces.csv'),rows)
        write_csv(Path(str(a.output_prefix)+'_'+label+'_events.csv'),events)
    a.output_prefix.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with a.output_prefix.with_suffix('.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['window','joint','case','error_rms_deg','dq_rms_rad_s','P_rms_Nm','D_rms_Nm','nominal_rms_Nm','feedback_rms_Nm'])
        for label,(s,_,_) in data.items():
            for name,cases in s['joints'].items():
                for c,d in cases.items():w.writerow([label,name,c,d['error_rms_deg'],d['dq']['rms'],d['P']['rms'],d['D']['rms'],d['nominal']['rms'],d['tau']['rms']])
    if a.plot:plot(a.output_prefix,data,data['autonomous'][2])
    print(json.dumps({label:{k:{c:round(v['error_rms_deg'],3) for c,v in d.items()} for k,d in s['groups'].items()} for label,(s,_,_) in data.items()},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','replay','output-prefix'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--late-replay',type=Path);p.add_argument('--lo',type=float,default=38.7);p.add_argument('--hi',type=float,default=41.2)
    p.add_argument('--plot',action='store_true');main(p.parse_args())
