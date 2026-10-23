#!/usr/bin/env python3
"""Offline d30af52 pipeline/log review. Standard library only, no hardware."""
import argparse
import csv
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
from audit_go_imu_runtime import read_csv, stats, rms

ROOT=Path(__file__).resolve().parents[3]
def hz(t):return (len(t)-1)*1000/(t[-1]-t[0]) if len(t)>1 else None

def audit(log):
    c,bad=read_csv(log/'robot.csv');t=c['t_mono_ms'];ppo=[i for i,m in enumerate(c['mode']) if m==3]
    policy,pbad=read_csv(log/'policy.csv')
    result={'log':str(log.relative_to(ROOT)),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'rows':len(t),'malformed_rows':[bad,pbad],'main_hz':hz(t),'policy_consumed_hz':hz(policy['t_mono_ms']),'ppo_rows':len(ppo),'ppo_span_s':(t[ppo[-1]]-t[ppo[0]])/1000,'buses':{},'windows':[]}
    for b in range(4):result['buses'][str(b)]={'body_us':stats(c[f'b{b}_total_us'][i] for i in ppo),'flush_us':stats(c[f'b{b}_flush_us'][i] for i in ppo),'write_us':stats(c[f'b{b}_write_us'][i] for i in ppo),'reply_us':stats(c[f'b{b}_reply_us'][i] for i in ppo)}
    result['near_timeout_snapshot_review']={}
    for bus in range(4):
        keys=[f'b{bus}_{k}_us' for k in ['total','flush','write','reply']]
        out={};last=None;events=[]
        for i in range(len(t)):
            key=tuple(c[k][i] for k in keys)
            if key!=last and c[keys[0]][i]>=10000:
                events.append({'observed_main_s':t[i]/1000,'total_us':c[keys[0]][i],'reply_us':c[keys[3]][i],'mode':int(c['mode'][i])})
            last=key
        for metric in ('total','reply'):
            v=c[f'b{bus}_{metric}_us']
            out[metric]={'max_us':max(v),'snapshot_rows_ge_ms':{str(th):sum(x>=th*1000 for x in v) for th in (5,10,15,18,19,20)}}
        out['adjacent_distinct_timing_tuples_ge_10ms']=events
        out['warning']='snapshot counts are not transaction counts; some bus rounds can be overwritten between main reads'
        result['near_timeout_snapshot_review'][str(bus)]=out
    result['trace_wait_thresholds']={}
    result['joint_host_age_ms']=stats(c[f'age_{j:02}'][i] for i in ppo for j in range(12))
    result['imu_host_age_ms']=stats(c['imu_age_ms'][i] for i in ppo)
    result['joint_valid_mask_values']=sorted(set(c['valid_mask'][i] for i in ppo));result['joint_error_mask_values']=sorted(set(c['err_mask'][i] for i in ppo))
    for start,end in [(62,66),(78.682762,80.942405),(100,104),(123,125),(214,218),(250,253)]:
        ids=[i for i in ppo if start*1000<=t[i]<=end*1000]
        if not ids:continue
        result['windows'].append({'start_s':start,'end_s':end,'samples':len(ids),'commands':sorted(set((c['cmd_vx'][i],c['cmd_vy'][i],c['cmd_wz'][i]) for i in ids)),'gyro_x_rms':rms(c['gyro_x'][i] for i in ids),'hip_dq_rms':rms(c[f'dq_{j:02}'][i] for i in ids for j in (0,3,6,9)),'joint_host_age_ms':stats(c[f'age_{j:02}'][i] for i in ids for j in range(12)),'buses_body_us':{str(b):stats(c[f'b{b}_total_us'][i] for i in ids) for b in range(4)}})
    with (log/'trace.fault.csv').open() as f:trace=list(csv.DictReader(f))
    groups=defaultdict(list)
    for r in trace:groups[(int(r['bus']),int(r['id']))].append(r)
    result['trace']={'coverage_s':[min(float(r['t_ms']) for r in trace)/1000,max(float(r['t_ms']) for r in trace)/1000],'per_id':{}}
    for (bus,id),rows in sorted(groups.items()):
        tt=[float(r['t_ms']) for r in rows];ok=[r for r in rows if r['ok']=='1']
        result['trace']['per_id'][f'{bus}:{id}']={'request_hz':hz(tt),'interval_ms':stats(b-a for a,b in zip(tt,tt[1:])),'waited_us':stats(float(r['waited_us']) for r in ok),'failures':[r for r in rows if r['ok']!='1']}
    result['trace_wait_thresholds']={str(th):[r for r in trace if int(r['waited_us'])>=th*1000] for th in (10,15,18,20)}
    result['trace']['send_spacing_ms']={}
    for bus in range(4):
        rows=sorted((r for r in trace if int(r['bus'])==bus),key=lambda r:float(r['t_ms']))
        gaps=[float(b['t_ms'])-float(a['t_ms']) for a,b in zip(rows,rows[1:]) if int(b['id'])==int(a['id'])+1 and int(a['id'])<2]
        result['trace']['send_spacing_ms'][str(bus)]=stats(gaps)
    result['last_long_rounds']=[]
    for bus in range(4):
        rows=sorted((r for r in trace if int(r['bus'])==bus),key=lambda r:float(r['t_ms']))
        long=[r for r in rows if int(r['id'])==0 and float(r['waited_us'])>15000]
        if not long:continue
        r=long[-1];start=float(r['t_ms'])
        after=[x for x in rows if int(x['id'])==0 and float(x['t_ms'])>start]
        result['last_long_rounds'].append({'bus':bus,'record':r,'approx_end_trace_ms':start+float(r['waited_us'])/1000,'next_id0_start_trace_ms':float(after[0]['t_ms']) if after else None,'warning':'start is before write but waited starts after write; sum omits write time; not exact end'})
    result['fault_events']=[l for l in (log/'events.log').read_text().splitlines() if 'Fault' in l]
    result['sha256']={str(log.relative_to(ROOT)/n):hashlib.sha256((log/n).read_bytes()).hexdigest() for n in ['robot.csv','policy.csv','trace.fault.csv','meta.json']}
    for n in ['Motor/src/go_bus.cpp','Motor/src/go_motor_array.cpp','Motor/inc/motor_transport.h','config/real_config.h']:
        p=ROOT/'imgo2_deploy/real'/n;result['sha256'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    result['limitations']=['new source is compatible with log signatures but runtime binary identity not recorded','successful pipeline trace hardcodes bytes=16 bad=0 stray=0; not proof of an error-free round','waited_us starts after per-ID write while trace start is before it; summing them is not exact receipt time','feedback_time is set when applying the collected round, not sensor sampling or earliest host receipt','no exact inference input or command echo sequence in these logs','trace covers only the final segment; it cannot certify communication during earlier shake windows']
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--log',type=Path,default=ROOT/'imgo2_deploy/real/build/log/2026-10-21_17-10-15_9557');p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit(a.log.resolve());a.output.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+'\n');print('main',r['main_hz'],'policy',r['policy_consumed_hz'],'joint age',r['joint_host_age_ms']);print('fault',r['fault_events'])
