#!/usr/bin/env python3
"""Optional matplotlib figures for the November21 diagnostic batch."""
import argparse,json,statistics as st
from pathlib import Path
from analyze_new_motion_batch import load,prepare
from analyze_leg_rack_batch import build

def main(prefix,validation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    result=json.loads(Path(str(prefix)+'_walk.json').read_text());sessions=result['sessions'];allrows={}
    labels=['Ground 1','Ground 2','Hands 1','Hands 2'];colors=['tab:blue','tab:orange','tab:green','tab:red']
    for s in sessions:
        r=load(Path(s['source'])/'robot.csv');prepare(r);allrows[s['name']]=r
    fig,axs=plt.subplots(4,3,figsize=(16,11),constrained_layout=True)
    for row,s in enumerate(sessions):
        r=allrows[s['name']];t=[a['t_mono_ms']/1000 for a in r]
        for col,key in enumerate(['cmd_vx','gyro_x','roll_deg']):
            ax=axs[row,col];ax.plot(t,[a[key] for a in r],lw=.65,color=colors[row]);ax.set(title=f'{labels[row]} | {key}',xlabel='Log time (s)',ylabel=['Command (m/s)','Gyro x (rad/s)','Roll (deg)'][col]);ax.grid(alpha=.2)
            for d,w in s['windows'].items():ax.axvspan(*w['window_s'],color='black' if d=='forward' else 'purple',alpha=.12)
    fig.suptitle('Full records | shaded: command-selected 5s windows; hands partly restrain roll',fontsize=13);fig.savefig(str(prefix)+'_overview.png',dpi=145);plt.close(fig)
    for direction in ['forward','reverse']:
        fig,axs=plt.subplots(3,2,figsize=(14,10),constrained_layout=True)
        for col,indices in enumerate([[0,1],[2,3]]):
            for i in indices:
                s=sessions[i];m=s['windows'][direction];lo,hi=m['window_s'];r=[a for a in allrows[s['name']] if lo<=a['t_mono_ms']/1000<hi];t=[a['t_mono_ms']/1000-lo for a in r]
                for row,key in enumerate(['gyro_x','roll_deg']):axs[row,col].plot(t,[a[key] for a in r],label=labels[i],lw=.8,color=colors[i],alpha=.85)
                sp=m['spectra']['gyro_x'];axs[2,col].semilogy(sp['f_hz'],sp['psd'],color=colors[i],label=labels[i])
            for row in range(3):
                ax=axs[row,col];ax.grid(alpha=.2);ax.legend(fontsize=9)
                ax.set(title=f"{'Unaided' if col==0 else 'Hand-supported base'} | {direction}",xlabel='Within-window time (s); independent phase' if row<2 else 'Frequency (Hz)',ylabel=['Gyro x (rad/s)','Roll (deg)','Gyro PSD ((rad/s)^2/Hz)'][row])
                if row<2:ax.set_ylim(*([-3,3] if row==0 else [-6,6]))
                else:ax.set(xlim=(0,20),ylim=(1e-7,2));ax.axvspan(4,10,color='gray',alpha=.08)
        fig.suptitle(f'{direction}: same nominal |vx|=.5m/s, PD25/.8 | different walks, no phase alignment',fontsize=13);fig.savefig(str(prefix)+f'_{direction}.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(3,2,figsize=(14,10),constrained_layout=True)
    for row,group in enumerate(['hip','thigh','shank']):
        for col,key in enumerate(['error','tau']):
            ax=axs[row,col];x=np.arange(4)
            for k,d in enumerate(['forward','reverse']):
                vals=[s['windows'][d]['groups'][group][key] for s in sessions];ax.bar(x+(k-.5)*.36,vals,.36,label=d)
            ax.set(xticks=x,xticklabels=labels,title=group,ylabel='Pooled 4-joint tracking error RMS (rad)' if key=='error' else 'Pooled estimated torque RMS (Nm)');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('Walking feedback: error = target - measured q | torque is motor estimate; different closed-loop targets',fontsize=12);fig.savefig(str(prefix)+'_joints.png',dpi=150);plt.close(fig)
    plan=json.loads((validation/'plan.json').read_text());sim=load(validation/'isaac/traces.csv');jobs=plan['jobs']
    fig,axs=plt.subplots(6,3,figsize=(17,17),constrained_layout=True)
    for row,source in enumerate(plan['sources']):
        case,streams=build(Path(source['source']),source['id'],None,None);d=streams['real'];ii=[i for i,t in enumerate(d['t']) if 1<=t<2.5];mean=st.fmean(d['q'])
        axs[row,0].plot([d['ref_t'][i] for i in ii],[d['ref'][i]-st.fmean(d['ref']) for i in ii],'k--',lw=1,label='Target AC')
        for col,key in enumerate(['q','dq','pd']):axs[row,col].plot([d['t'][i] for i in ii],[d[key][i]-(mean if key=='q' else 0) for i in ii],color='tab:blue',label='Real',lw=1)
        axs[row,2].plot([d['t'][i] for i in ii],[d['effort'][i] for i in ii],color='gray',alpha=.6,lw=.8,label='Real estimated torque')
        for env,job in enumerate(jobs):
            if job['source_id']!=source['id']:continue
            ss=[r for r in sim if 1<=r['test_s']<2.5];full=[r for r in sim if case['window_s'][0]<=r['test_s']<case['window_s'][1]];mq=st.fmean(r[f'q_{env:03}'] for r in full)
            color='tab:orange' if job['candidate_id']=='baseline' else 'tab:green';label='Isaac original' if job['candidate_id']=='baseline' else 'Isaac prior candidate'
            for col,key in enumerate(['q','dq','pd']):axs[row,col].plot([r['test_s'] for r in ss],[r[f'{key}_{env:03}']-(mq if key=='q' else 0) for r in ss],color=color,label=label,lw=.9)
        for col in range(3):
            ax=axs[row,col];ax.set(title=f"Case{source['id']}: {source['joint']} | 2Hz +/- .05rad",xlabel='Time from first command (s)',ylabel=['q minus own steady mean (rad)','dq (rad/s)','Nominal PD torque (Nm)'][col]);ax.grid(alpha=.2)
            if row==0:ax.legend(fontsize=7)
    fig.suptitle('Independent new rack data | no refitting, no time shift | 1.5s detail from each valid steady window',fontsize=14);fig.savefig(str(prefix)+'_rack_time.png',dpi=140);plt.close(fig)
    scores=json.loads((validation/'isaac/scores.json').read_text())['jobs'];cases=json.loads(Path(str(prefix)+'_rack_real.json').read_text())['cases']
    fig,axs=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
    for row,joint in enumerate(['FL_hip','FL_thigh','FL_shank']):
        cc=[c for c in cases if c['joint']==joint]
        for col,key,scale in [(0,'gain',100),(1,'phase_deg',1)]:
            x=np.arange(2);ax=axs[row,col]
            for k,name in enumerate(['Real','Isaac original','Isaac prior candidate']):
                if k==0:vals=[c['streams']['real'][key]*scale for c in cc]
                else:
                    variant='baseline' if k==1 else 'previous_candidate';vals=[next(r for r in scores if r['source_id']==c['id'] and r['candidate_id']==variant)[key]*scale for c in cc]
                bars=ax.bar(x+(k-1)*.25,vals,.25,label=name,color=['tab:blue','tab:orange','tab:green'][k]);ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=2)
            ax.set(title=joint,xticks=x,xticklabels=['Repeat 1','Repeat 2'],ylabel='Fundamental amplitude / target (%)' if col==0 else 'Phase relative to target (deg)');ax.grid(axis='y',alpha=.2)
            if col==0:ax.set_ylim(0,120)
            else:ax.set_ylim(-50,0)
            if row==0:ax.legend(fontsize=8,loc='lower left' if col==0 else 'lower right')
    fig.suptitle('New 2Hz +/- .05rad rack repeats | frozen prior parameters, independent validation',fontsize=13);fig.savefig(str(prefix)+'_rack_response.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prefix',type=Path,required=True);p.add_argument('--validation',type=Path,required=True);a=p.parse_args();main(a.prefix,a.validation)
