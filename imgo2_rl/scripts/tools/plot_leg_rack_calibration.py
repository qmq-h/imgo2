#!/usr/bin/env python3
"""Plot already-completed rack parameter experiments; no simulation or fitting."""
import argparse,csv,json,math,statistics as st
from pathlib import Path
from collections import defaultdict
from analyze_leg_rack_batch import build
from analyze_leg_rack_calibration import interpolate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output-prefix',type=Path,required=True);args=p.parse_args();root=args.root;prefix=args.output_prefix
 scores=json.loads((root/'validation/scores.json').read_text());plan=json.loads((root/'validation/plan_used.json').read_text());sources={s['id']:s for s in plan['sources']};jobs=scores['jobs'];allcases={};trace={}
 rows=list(csv.DictReader((root/'validation/traces.csv').open()))
 for sid in sorted({j['source_id'] for j in jobs}):
  s=sources[sid];c,d=build(Path(s['source']),sid,root/'gazebo',None);allcases[sid]=c
  start,end=c['window_s'];tt=[float(r['test_s']) for r in rows if start<=float(r['test_s'])<end];chosen=[r for r in rows if start<=float(r['test_s'])<end]
  trace[sid]={'real':(d['real']['t'],d['real']['q']),'gazebo_selected':(d['gazebo']['t'],d['gazebo']['q'])}
  for job in jobs:
   if job['source_id']==sid and job['candidate_id'] in ['baseline','selected_combined']:
    trace[sid][job['candidate_id']]=(tt,[float(r[f"q_{job['env']:03d}"]) for r in chosen])
  old,_=build(Path(s['source']),sid,Path('imgo2_deploy/logs/leg_fl_rack_gazebo_20261120'),None)
  c['gazebo_baseline']=old['streams']['gazebo']
 # Same metrics as calibration score for Gazebo predictions.
 gzmetrics=[]
 for sid,c in allcases.items():
  tr=trace[sid];t,q=tr['gazebo_selected'];rt,rq=tr['real'];r=[interpolate(rt,rq,x) for x in t];mq=st.fmean(q);mr=st.fmean(r);amp=c['streams']['real']['target_fit']['amplitude']
  mse=st.fmean(((a-mq)-(b-mr))**2 for a,b in zip(q,r));md=mq-mr
  gzmetrics.append(dict(source_id=sid,joint=c['joint'],frequency_hz=c['frequency_hz'],amplitude_rad=c['amplitude_rad'],split='holdout' if c['frequency_hz']==2 else 'fit',score=mse/amp**2+.25*(md/amp)**2,ac_nrmse=math.sqrt(mse)/amp,mean_q_error_rad=md,**{k:c['streams']['gazebo'][k] for k in ['gain','phase_deg']}))
 (root/'gazebo_scores.json').write_text(json.dumps(gzmetrics,indent=2)+'\n')
 colors={'real':'tab:blue','baseline':'tab:orange','selected_combined':'tab:green','gazebo_selected':'magenta','gazebo_baseline':'gray'}
 labels={'real':'Real','baseline':'Isaac original','selected_combined':'Isaac candidate','gazebo_selected':'Gazebo candidate','gazebo_baseline':'Gazebo original'}
 fig,axs=plt.subplots(3,2,figsize=(15,11),constrained_layout=True)
 for row,joint in enumerate(['FL_hip','FL_thigh','FL_shank']):
  cases=[c for c in allcases.values() if c['joint']==joint]
  for amp in sorted({c['amplitude_rad'] for c in cases}):
   cc=sorted([c for c in cases if c['amplitude_rad']==amp],key=lambda c:c['frequency_hz'])
   for key in colors:
    mm=[]
    for c in cc:
     if key=='real':m=c['streams']['real']
     elif key=='gazebo_selected':m=c['streams']['gazebo']
     elif key=='gazebo_baseline':m=c['gazebo_baseline']
     else:m=next(j for j in jobs if j['source_id']==c['id'] and j['candidate_id']==key)
     mm.append(m)
    for col,metric,scale in [(0,'gain',100),(1,'phase_deg',1)]:
     axs[row,col].plot([c['frequency_hz'] for c in cc],[m[metric]*scale for m in mm],'-' if amp==.01 else '--',marker='o',color=colors[key],label=f'{labels[key]} A={amp:g}',alpha=.85)
  for col,label in [(0,'Fundamental amplitude / command (%)'),(1,'Phase relative to command (deg)')]:
   axs[row,col].set(title=joint,xlabel='Hz',ylabel=label,xticks=[.5,1,2]);axs[row,col].grid(alpha=.25);axs[row,col].axvspan(1.88,2.12,color='tab:purple',alpha=.07)
  axs[row,0].legend(fontsize=7,ncol=2)
 fig.suptitle('Screening candidate, not accepted: fit 0.5/1 Hz; 2 Hz held out (purple) | PD 25 / 0.8')
 fig.savefig(str(prefix)+'_response.png',dpi=150);plt.close(fig)
 fig,axs=plt.subplots(3,3,figsize=(17,10),constrained_layout=True)
 for row,joint in enumerate(['FL_hip','FL_thigh','FL_shank']):
  cc=[c for c in allcases.values() if c['joint']==joint]
  conditions=[(1,.01),(1,max(c['amplitude_rad'] for c in cc)),(2,.01)]
  for col,(f,amp) in enumerate(conditions):
   c=next(c for c in cc if c['frequency_hz']==f and c['amplitude_rad']==amp);ax=axs[row,col]
   for key in ['real','baseline','selected_combined','gazebo_selected']:
    t,q=trace[c['id']][key];mean=st.fmean(q);ax.plot(t,[v-mean for v in q],color=colors[key],label=labels[key],lw=1)
   ax.set(xlim=(1,min(c['window_s'][1],1+3/f)),title=f'{joint} {f:g}Hz A={amp:g}'+(' [HELD OUT]' if f==2 else ''),xlabel='s from first source send',ylabel='q minus own mean (rad)');ax.grid(alpha=.25)
   if row==0:ax.legend(fontsize=8)
 fig.suptitle('No phase alignment; own means removed only for waveform display. Static errors scored separately.')
 fig.savefig(str(prefix)+'_time.png',dpi=150);plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
 for ax,joint in zip(axs,['FL_hip','FL_thigh','FL_shank']):
  cc=sorted([c for c in allcases.values() if c['joint']==joint],key=lambda c:(c['frequency_hz'],c['amplitude_rad']))
  for ki,key in enumerate(['baseline','selected_combined','gazebo_selected']):
   vals=[]
   for c in cc:
    m=next(j for j in (gzmetrics if key=='gazebo_selected' else jobs) if j['source_id']==c['id'] and (key=='gazebo_selected' or j['candidate_id']==key));vals.append(m['ac_nrmse'])
   ax.bar([i+(ki-1)*.24 for i in range(len(cc))],vals,.24,color=colors[key],label=labels[key])
  ax.set_xticks(range(len(cc)),[f"{c['frequency_hz']:g}Hz\nA{c['amplitude_rad']:g}"+('\nheld out' if c['frequency_hz']==2 else '') for c in cc]);ax.set(title=joint,ylabel='AC waveform RMSE / target amplitude');ax.grid(axis='y',alpha=.25);ax.legend(fontsize=8)
 fig.suptitle('Prediction error improves, but small-signal phase/amplitude mismatches remain')
 fig.savefig(str(prefix)+'_errors.png',dpi=150);plt.close(fig)
 fine=json.loads((root/'step_check/scores.json').read_text())['jobs']
 fig,axs=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
 for ax,joint in zip(axs,['FL_hip','FL_thigh','FL_shank']):
  c=next(c for c in allcases.values() if c['joint']==joint and c['frequency_hz']==1 and c['amplitude_rad']==.01)
  coarse=next(j for j in jobs if j['source_id']==c['id'] and j['candidate_id']=='selected_combined')
  f=next(j for j in fine if j['source_id']==c['id'] and j['candidate_id']=='selected_combined')
  ax.bar(['Real','Candidate\n0.5ms','Candidate\n0.25ms'],[100*c['streams']['real']['gain'],100*coarse['gain'],100*f['gain']],color=['tab:blue','tab:green','tab:red'])
  ax.set(title=joint,ylabel='1Hz small-signal amplitude ratio (%)');ax.grid(axis='y',alpha=.25)
 fig.suptitle('Candidate timestep sensitivity: shank changes ~9 percentage points; no convergence acceptance')
 fig.savefig(str(prefix)+'_step.png',dpi=150);plt.close(fig)
 print('Saved figures and Gazebo scores')
if __name__=='__main__':main()
