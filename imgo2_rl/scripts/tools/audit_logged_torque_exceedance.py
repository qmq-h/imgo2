#!/usr/bin/env python3
"""Stdlib audit of logged estimated motor torque, not a torque-sensor calibration."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

NAMES=[f'{leg}_{joint}' for leg in ('FL','FR','RL','RR') for joint in ('hip','thigh','shank')]

def summarize(rows):
    if not rows:return {'rows':0}
    peak=max(((abs(r[f'tau_{i:02}']),r,i) for r in rows for i in range(12)),key=lambda x:x[0])
    def peak_info(p):
        _,r,i=p
        return {'abs_Nm':abs(r[f'tau_{i:02}']),'signed_Nm':r[f'tau_{i:02}'],'time_s':r['t_mono_ms']*.001,'joint_index':i,'joint':NAMES[i],
                'mode':int(r['mode']),'cmd':[r['cmd_'+k] for k in ('vx','vy','wz')],
                'valid_mask':int(r['valid_mask']),'err_mask':int(r['err_mask']),'age_ms':r[f'age_{i:02}'],'motor_temp':r[f'temp_{i:02}']}
    out={'rows':len(rows),'joint_row_samples':len(rows)*12,'start_s':rows[0]['t_mono_ms']*.001,'end_s':rows[-1]['t_mono_ms']*.001,'peak':peak_info(peak),'thresholds':{}}
    for threshold in (23.5,23.7):
        per_joint=[];any_count=0
        for r in rows:any_count+=any(abs(r[f'tau_{i:02}'])>threshold for i in range(12))
        for i in range(12):
            over=[r for r in rows if abs(r[f'tau_{i:02}'])>threshold]
            longest=[];current=[]
            for r in rows:
                if abs(r[f'tau_{i:02}'])>threshold:
                    if current and r['t_mono_ms']-current[-1]['t_mono_ms']>20:
                        if len(current)>len(longest):longest=current
                        current=[]
                    current.append(r)
                else:
                    if len(current)>len(longest):longest=current
                    current=[]
            if len(current)>len(longest):longest=current
            if over:per_joint.append({'joint':NAMES[i],'index':i,'count':len(over),'fraction_of_joint_rows':len(over)/len(rows),
                                     'peak_abs_Nm':max(abs(r[f'tau_{i:02}']) for r in over),
                                     'longest_consecutive_rows':len(longest),'longest_first_to_last_ms':longest[-1]['t_mono_ms']-longest[0]['t_mono_ms']})
        out['thresholds'][str(threshold)]={'any_joint_rows':any_count,'fraction_of_rows':any_count/len(rows),'joint_row_exceedances':sum(v['count'] for v in per_joint),'per_joint':per_joint}
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',action='append',required=True,help='label,logdir,forward_start_s,forward_end_s');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report={'threshold_comparison':'abs(tau_est)>threshold, strict greater; values are reported output-side torque estimates',
                           'limits':['host rows may repeat one feedback frame; not independent device sample counts','consecutive first-to-last span is not measured physical above-limit duration','reported motor torque not independently calibrated; no firmware limit conclusion','PPO mode=3 from current FSM enum'], 'cases':{}}
    for spec in a.case:
        label,directory,lo,hi=spec.split(',');path=Path(directory)/'robot.csv'
        with path.open() as f:rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
        result={'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'all':summarize(rows),'ppo':summarize([r for r in rows if int(r['mode'])==3]),
                'forward':summarize([r for r in rows if float(lo)<=r['t_mono_ms']*.001<float(hi)])}
        report['cases'][label]=result
        for name in ('all','ppo','forward'):
            s=result[name];print(label,name,s['peak'],s['thresholds']['23.7'])
    a.output.write_text(json.dumps(report,indent=2)+'\n')
