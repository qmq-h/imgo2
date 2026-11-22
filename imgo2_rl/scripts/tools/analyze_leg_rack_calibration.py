#!/usr/bin/env python3
"""Standard-library scoring of bounded Isaac rack parameter trials.

No time/phase shift, no feedback correction. Fits 0.5/1Hz only. Scores whole
2Hz holdout trajectories separately; never uses holdout to select parameters.
"""
import argparse,array,bisect,csv,json,math,statistics as st
from pathlib import Path
from collections import defaultdict
from analyze_leg_rack_batch import build,harmonic
from audit_joint_sine_log import phase_difference

def interpolate(t,y,x):
 i=bisect.bisect_right(t,x)
 if i==0:return y[0]
 if i==len(t):return y[-1]
 w=(x-t[i-1])/(t[i]-t[i-1]);return y[i-1]+w*(y[i]-y[i-1])

def analyze(folder):
 plan=json.loads((folder/'plan_used.json').read_text());meta=json.loads((folder/'metadata.json').read_text());assert meta['completed'] and meta['production_files_unchanged']
 info={s['id']:s for s in plan['sources']};real={}
 for sid in {j['source_id'] for j in plan['jobs']}:
  c,tr=build(Path(info[sid]['source']),sid,None,None);real[sid]=(c,tr['real'])
 n=len(plan['jobs']);qs=[array.array('d') for _ in range(n)];ts=array.array('d');vpeak=[0.]*n
 with (folder/'traces.csv').open() as f:
  r=csv.reader(f);heads=next(r);assert len(heads)==1+4*n
  for row in r:
   ts.append(float(row[0]))
   for i in range(n):qs[i].append(float(row[1+4*i]));vpeak[i]=max(vpeak[i],abs(float(row[2+4*i])))
 results=[]
 for e,job in enumerate(plan['jobs']):
  c,d=real[job['source_id']];start,end=c['window_s'];indices=[k for k,t in enumerate(ts) if start<=t<end]
  t=[ts[k] for k in indices];q=[qs[e][k] for k in indices];r=[interpolate(d['t'],d['q'],x) for x in t]
  fit=harmonic(t,q,c['frequency_hz']);ref=c['streams']['real']['target_fit'];rm=c['streams']['real'];amp=ref['amplitude'];mq=st.fmean(q);mr=st.fmean(r)
  ac_mse=st.fmean(((a-mq)-(b-mr))**2 for a,b in zip(q,r));mean_delta=mq-mr
  score=ac_mse/amp**2+.25*(mean_delta/amp)**2
  results.append(dict(env=e,**job,joint=c['joint'],frequency_hz=c['frequency_hz'],amplitude_rad=c['amplitude_rad'],
    samples=len(t),q_fit=fit,gain=fit['amplitude']/amp,phase_deg=phase_difference(fit['phase_deg'],ref['phase_deg']),
    real_gain=rm['gain'],real_phase_deg=rm['phase_deg'],mean_q_error_rad=mean_delta,
    ac_rmse_rad=math.sqrt(ac_mse),ac_nrmse=math.sqrt(ac_mse)/amp,score=score,
    raw_q_rmse_rad=math.sqrt(ac_mse+mean_delta**2),max_logged_speed_rad_s=vpeak[e],steady_usable=c['steady_usable']))
 groups=defaultdict(list)
 for r in results:
  if r['split']=='fit':groups[(r['joint'],r['candidate_id'])].append(r)
 rankings=defaultdict(list)
 for (joint,candidate),rows in groups.items():
  rankings[joint].append(dict(candidate_id=candidate,fit_score=st.fmean(r['score'] for r in rows),n_records=len(rows),parameters=rows[0]['parameters'],
    mean_ac_nrmse=st.fmean(r['ac_nrmse'] for r in rows),max_abs_mean_error_rad=max(abs(r['mean_q_error_rad']) for r in rows)))
 for joint in rankings:rankings[joint].sort(key=lambda r:r['fit_score'])
 answer=dict(batch=str(folder),scoring='mean over fit records of (AC waveform RMSE/target amplitude)^2 + .25*(mean q mismatch/target amplitude)^2; no phase shift',
    rankings=dict(rankings),jobs=results,held_out='2Hz trajectories not used in rankings',production_files_unchanged=meta['production_files_unchanged'])
 (folder/'scores.json').write_text(json.dumps(answer,indent=2)+'\n')
 for joint,rank in rankings.items():print(joint,'TOP',[(x['candidate_id'],round(x['fit_score'],4)) for x in rank[:6]],flush=True)
 return answer
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('batch',type=Path);a=p.parse_args();analyze(a.batch)
