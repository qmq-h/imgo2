#!/usr/bin/env python3
"""Recompute observed command/state rates from archived sim/real comparison.
Standard library only; no ROS, simulation, serial ports or hardware execution.
"""
import argparse
import bisect
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[3]

def read(path):
    with gzip.open(path,'rt') as f:return list(csv.DictReader(f))

def interpolate(x,xs,ys):
    i=bisect.bisect_right(xs,x)
    if i==0:return ys[0]
    if i==len(xs):return ys[-1]
    f=(x-xs[i-1])/(xs[i]-xs[i-1])
    return ys[i-1]+f*(ys[i]-ys[i-1])

def summary(host,sim=None):
    assert len(host)>1 and all(b>a for a,b in zip(host,host[1:]))
    dt=sorted((b-a)*1000 for a,b in zip(host,host[1:]))
    out={'samples':len(host),'observed_host_hz':(len(host)-1)/(host[-1]-host[0]),
         'interval_ms':{'median':statistics.median(dt),'p95':dt[int((len(dt)-1)*.95)],'p99':dt[int((len(dt)-1)*.99)],'max':dt[-1]}}
    if sim is not None:out['observed_sim_hz']=(len(sim)-1)/(sim[-1]-sim[0])
    return out

def changes(rows,times,prefix):
    out=[];previous=None
    for r,t in zip(rows,times):
        q=[float(r[f'{prefix}_{i:02}']) for i in range(12)]
        if previous is not None and max(abs(a-b) for a,b in zip(q,previous))>1e-7:out.append(t)
        previous=q
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prefix',type=Path,default=ROOT/'docs/deployment/forward_sim_real_2026-10-22');p.add_argument('--output',type=Path);a=p.parse_args()
    def path(n):return Path(str(a.prefix)+n)
    clock=read(path('_sim_clock.csv.gz'));ch=[float(r['host_s']) for r in clock];cs=[float(r['sim_s']) for r in clock];origin=cs[0]
    result={'window_ppo_s':[14,16.259643],'interpretation':'ROS receive cadence, not directly instrumented function execution or device sampling clock','sim':{}}
    for name in ['command','state','imu']:
        rows=read(path('_sim_'+name+'.csv.gz'));host=[float(r['host_s']) for r in rows]
        sim=[float(r['t_s']) if name=='imu' else interpolate(t,ch,cs)-origin-6 for r,t in zip(rows,host)]
        ids=[i for i,t in enumerate(sim) if 14<=t<=16.259643]
        selected=[rows[i] for i in ids];ht=[host[i] for i in ids];st=[sim[i] for i in ids]
        result['sim'][name]=summary(ht,st)
        if name=='command':result['sim']['target_value_changes']=summary(changes(selected,ht,'q'))
    rows=read(path('_real_window.csv.gz'));host=[float(r['t_mono_ms'])/1000 for r in rows]
    result['real']={'main_log_rows':summary(host),'target_value_changes':summary(changes(rows,host,'q_des'))}
    source=ROOT/'imgo2_deploy/ros2_ws/src/imgo2_deploy/library/core/loop/loop.hpp'
    result['loop_source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    result['limits']=['target value changes are not direct inference event timestamps','ROS receive jitter can differ from publisher scheduling','real per-motor feedback frequency cannot be obtained by counting repeated main snapshots']
    text=json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    if a.output:a.output.write_text(text)
    print(text,end='')

if __name__=='__main__':main()
