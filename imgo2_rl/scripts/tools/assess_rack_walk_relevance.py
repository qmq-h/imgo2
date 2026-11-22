#!/usr/bin/env python3
"""Compare rack excitation scale with recorded walking targets and row errors.

Stdlib only. Target half-spread, new-target step, and instantaneous tracking
error are deliberately separate quantities, none equals sine half-amplitude.
"""
import argparse,csv,hashlib,json,math,statistics as st
from pathlib import Path

def quantile(xs,p):
 s=sorted(xs);x=(len(s)-1)*p;k=int(x);return s[k] if k==len(s)-1 else s[k]+(x-k)*(s[k+1]-s[k])
def describe(x):
 return dict(n=len(x),mean=st.fmean(x),std=st.pstdev(x),rms=math.sqrt(st.fmean(v*v for v in x)),p05=quantile(x,.05),median=quantile(x,.5),p95=quantile(x,.95))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 windows=[(71.9,74.7),(76.8,79.6),(97.4,100.2)];groups=[[] for _ in windows]
 with (args.source/'robot.csv').open() as f:
  for raw in csv.DictReader(f):
   t=float(raw['t_mono_ms'])/1000
   for group,(lo,hi) in zip(groups,windows):
    if lo<=t<hi:
     r={k:float(raw[k]) for k in ['t_mono_ms','gen','mode','cmd_vx','cmd_vy','cmd_wz']+[f'{s}_{i:02}' for s in ['q_des','q','dq'] for i in range(12)]};group.append(r)
 assert all(len(g)>0 and all(r['mode']==3 and r['cmd_vx']==.5 for r in g) for g in groups)
 axes=[]
 for i in range(12):
  qdes=[r[f'q_des_{i:02}'] for g in groups for r in g];errors=[r[f'q_des_{i:02}']-r[f'q_{i:02}'] for g in groups for r in g]
  deltas=[]
  for group in groups:
   last=group[0]
   for r in group[1:]:
    if any(r[f'q_des_{k:02}']!=last[f'q_des_{k:02}'] for k in range(12)):
     deltas.append(r[f'q_des_{i:02}']-last[f'q_des_{i:02}']);last=r
  absdel=[abs(v) for v in deltas];abserr=[abs(v) for v in errors]
  axes.append(dict(joint=f"{['FL','FR','RL','RR'][i//3]}_{['hip','thigh','shank'][i%3]}",target=describe(qdes),target_central90_halfspread_rad=(quantile(qdes,.95)-quantile(qdes,.05))/2,
    new_target_abs_step=describe(absdel),row_abs_tracking_error=describe(abserr),fraction_abs_error_le={str(x):st.fmean(v<=x for v in abserr) for x in [.01,.02,.05,.1]},fraction_new_target_step_le={str(x):st.fmean(v<=x for v in absdel) for x in [.01,.02,.05,.1]}))
 result=dict(source=str(args.source),source_robot_sha256=hashlib.sha256((args.source/'robot.csv').read_bytes()).hexdigest(),windows_s=windows,rows=[len(g) for g in groups],axes=axes,
    limitations=['Walking is load-bearing, multi-joint, non-sinusoidal and contains unknown support assistance intervals.',
    'Half-spread, command increment and feedback error are different; small increments do not imply unloaded small-signal operation.',
    'robot rows contain current targets and earlier loop feedback, not device applied/sample synchronized states.',
    '0.5-2Hz rack tests do not establish dynamics at observed 4-10Hz walking components.'])
 args.output.write_text(json.dumps(result,indent=2)+'\n')
 for k in range(3):
  group=axes[k::3];print(['hip','thigh','shank'][k], 'target halfspread', [round(a['target_central90_halfspread_rad'],4) for a in group], 'median target step',[round(a['new_target_abs_step']['median'],4) for a in group], 'median abs e',[round(a['row_abs_tracking_error']['median'],4) for a in group], 'fraction |e|<=.01',[round(a['fraction_abs_error_le']['0.01'],3) for a in group])
if __name__=='__main__':main()
