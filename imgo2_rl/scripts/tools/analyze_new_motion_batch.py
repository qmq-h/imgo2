#!/usr/bin/env python3
"""Audit ground/assisted walking repeats using only the standard library.

Windows depend only on commands/time, never on measured oscillation. Outputs
full source hashes, selection rules, raw-unit metrics and Hann periodograms.
Plotting is an optional separate entry point with matplotlib/numpy.
"""
import argparse,bisect,cmath,csv,hashlib,json,math,statistics as st
from pathlib import Path

def load(p):
    with p.open() as f:return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
def rms(x):return math.sqrt(st.fmean(v*v for v in x))
def quant(x,p):
    a=sorted(x);v=(len(a)-1)*p;i=int(v);return a[i]+(v-i)*(a[min(i+1,len(a)-1)]-a[i])
def summary(x):return dict(mean=st.fmean(x),std=st.pstdev(x),rms=rms(x),min=min(x),max=max(x),p05=quant(x,.05),p95=quant(x,.95))
def fft(x):
    n=len(x)
    if n==1:return x
    a=fft(x[::2]);b=fft(x[1::2]);c=[cmath.exp(-2j*math.pi*k/n)*b[k] for k in range(n//2)]
    return [a[k]+c[k] for k in range(n//2)]+[a[k]-c[k] for k in range(n//2)]
def spectrum(x,fs=200.):
    n=len(x);mean=st.fmean(x);w=[.5-.5*math.cos(2*math.pi*i/(n-1)) for i in range(n)]
    y=[(v-mean)*a for v,a in zip(x,w)];nn=1<<(n-1).bit_length();z=fft(y+[0.]*(nn-n));scale=fs*sum(a*a for a in w)
    p=[abs(v)**2/scale*(1 if k in (0,nn//2) else 2) for k,v in enumerate(z[:nn//2+1])];f=[k*fs/nn for k in range(len(p))];df=fs/nn
    peak=max((i for i,v in enumerate(f) if .5<=v<=20),key=lambda i:p[i])
    expected=sum(v*v for v in y)/sum(a*a for a in w)
    return dict(f_hz=f,psd=p,dominant_05_20_hz=f[peak],band_4_10_rms=math.sqrt(sum(v*df for v,hz in zip(p,f) if 4<=hz<=10)),parseval_abs_error=abs(sum(p)*df-expected),n=n,df_hz=df)
def interp(t,y,x):
    i=bisect.bisect_right(t,x)
    if i==0:return y[0]
    if i==len(t):return y[-1]
    w=(x-t[i-1])/(t[i]-t[i-1]);return y[i-1]+w*(y[i]-y[i-1])
def spans(rows,sign):
    out=[];cur=[]
    for r in rows:
        t=r['input_captured_ms']/1000
        ok=abs(r['cmd_vx']-sign*.5)<.005 and abs(r['cmd_vy'])<=.06 and abs(r['cmd_wz'])<=.03
        if not ok or (cur and t-cur[-1]>.045):
            if len(cur)>1:out.append([cur[0],cur[-1]])
            cur=[]
        if ok:cur.append(t)
    if len(cur)>1:out.append([cur[0],cur[-1]])
    return sorted(out,key=lambda a:a[1]-a[0],reverse=True)
def prepare(rows):
    for r in rows:
        q=[r['quat_'+k] for k in ('w','x','y','z')];norm=math.sqrt(sum(v*v for v in q));w,x,y,z=[v/norm for v in q]
        r['quat_norm']=norm;r['roll_deg']=math.degrees(math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)));r['pitch_deg']=math.degrees(math.asin(max(-1,min(1,2*(w*y-z*x)))))
        for j in range(12):
            s=f'{j:02}';r['error_'+s]=r['q_des_'+s]-r['q_'+s];r['pd_'+s]=r['kp_'+s]*r['error_'+s]+r['kd_'+s]*(r['dq_des_'+s]-r['dq_'+s])+r['tau_ff_'+s]
def window_metrics(robot,policy,lo,hi,include_spectrum=True):
    rr=[r for r in robot if lo<=r['t_mono_ms']/1000<hi];pp=[r for r in policy if lo<=r['input_captured_ms']/1000<hi]
    assert len(rr)>=.95*(hi-lo)*200 and len(pp)>=.95*(hi-lo)*50
    names=['gyro_x','gyro_y','gyro_z','roll_deg','pitch_deg'];m={k:summary([r[k] for r in rr]) for k in names}
    m.update(window_s=[lo,hi],robot_samples=len(rr),policy_samples=len(pp),command={k:summary([r[k] for r in pp]) for k in ('cmd_vx','cmd_vy','cmd_wz')})
    m['quality']=dict(kp_range=[min(r[f'kp_{j:02}'] for r in rr for j in range(12)),max(r[f'kp_{j:02}'] for r in rr for j in range(12))],kd_range=[min(r[f'kd_{j:02}'] for r in rr for j in range(12)),max(r[f'kd_{j:02}'] for r in rr for j in range(12))],action_bound_rows=sum(any(abs(r[f'act_{j:02}'])>=2.999999 for j in range(12)) for r in pp),action_bound_axes=[sum(abs(r[f'act_{j:02}'])>=2.999999 for r in pp) for j in range(12)])
    m['joints']=[dict(index=j,**{k:summary([r[f'{k}_{j:02}'] for r in rr]) for k in ['error','q','q_des','dq','tau','pd']}) for j in range(12)]
    m['groups']={name:{key:math.sqrt(st.fmean(m['joints'][j][key]['rms']**2 for j in range(offset,12,3))) for key in ['error','dq','tau','pd']} for offset,name in enumerate(['hip','thigh','shank'])}
    if include_spectrum:
        t=[r['t_mono_ms']/1000 for r in robot];grid=[lo+i/200 for i in range(round((hi-lo)*200))]
        m['spectra']={}
        for key in ['gyro_x','roll_deg','dq_00','dq_01','dq_02']:
            y=[r[key] for r in robot]
            m['spectra'][key]=spectrum([interp(t,y,x) for x in grid])
    return m
def analyze(source,prefix):
    sessions=[]
    for folder in sorted(p for p in source.iterdir() if p.is_dir() and (p/'robot.csv').exists()):
        robot=load(folder/'robot.csv');policy=load(folder/'policy.csv');meta=json.loads((folder/'meta.json').read_text());prepare(robot)
        assert all(math.isfinite(v) for r in robot+policy for v in r.values())
        for rows,key in [(robot,'t_mono_ms'),(policy,'t_mono_ms'),(policy,'input_captured_ms')]:assert all(b[key]>a[key] for a,b in zip(rows,rows[1:]))
        ppo=[r for r in robot if r['mode']==3];assert ppo
        s=dict(name=folder.name,support='hand-supported base on both sides' if 'putup' in folder.name else 'unaided',source=str(folder),sha256={n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in ['robot.csv','policy.csv','meta.json','events.log']},robot_rows=len(robot),policy_rows=len(policy),robot_header_fields=len(robot[0])-27,policy_header_fields=len(policy[0]),metadata_counts={k:v for k,v in meta.items() if 'columns' in k},windows={},eligible_spans={},full_eligible_metrics={})
        s['quality']=dict(ppo_rows=len(ppo),clamped_rows=sum(r['clamped']!=0 for r in ppo),invalid_motor_rows=sum(r['valid_mask']!=4095 for r in ppo),error_rows=sum(r['err_mask']!=0 for r in ppo),invalid_imu_rows=sum(r['imu_valid']!=1 for r in ppo),kp_values=sorted({r[f'kp_{j:02}'] for r in ppo for j in range(12)}),kd_values=sorted({r[f'kd_{j:02}'] for r in ppo for j in range(12)}),max_abs_dq_des=max(abs(r[f'dq_des_{j:02}']) for r in ppo for j in range(12)),max_abs_tau_ff=max(abs(r[f'tau_ff_{j:02}']) for r in ppo for j in range(12)),robot_max_gap_ms=max(b['t_mono_ms']-a['t_mono_ms'] for a,b in zip(ppo,ppo[1:])),policy_input_max_gap_ms=max(b['input_captured_ms']-a['input_captured_ms'] for a,b in zip(policy,policy[1:])),quat_norm_range=[min(r['quat_norm'] for r in ppo),max(r['quat_norm'] for r in ppo)],motor_age_ms=summary([r[f'age_{j:02}'] for r in ppo for j in range(12)]),imu_age_ms=summary([r['imu_age_ms'] for r in ppo]),result_age_ms=summary([r['result_age_ms'] for r in policy]),obs_age_ms=summary([r['obs_age_ms'] for r in policy]),bus_total_us={str(j):summary([r[f'b{j}_total_us'] for r in ppo]) for j in range(4)},feedback_torque_over_23_7_rows=sum(any(abs(r[f'tau_{j:02}'])>23.7 for j in range(12)) for r in ppo))
        for direction,sign in [('forward',1),('reverse',-1)]:
            eligible=spans(policy,sign);assert eligible and eligible[0][1]-eligible[0][0]>=6.5
            s['eligible_spans'][direction]=eligible;lo=eligible[0][0]+1;s['windows'][direction]=window_metrics(robot,policy,lo,lo+5)
            s['full_eligible_metrics'][direction]=[window_metrics(robot,policy,a+1,b-.5,False) for a,b in eligible if b-a>=3.5]
        sessions.append(s);print(folder.name,{d:round(w['gyro_x']['rms'],4) for d,w in s['windows'].items()},flush=True)
    out=dict(source=str(source),selection='policy input time; abs(vx-sign*.5)<.005, abs(vy)<=.06, abs(wz)<=.03; contiguous gap<=45ms; longest span per sign; skip first1s then first5s; full spans sensitivity skips1s/end.5s',support_user_confirmation='putup: hands on both sides of base, partly restrains roll; no clear hand shaking felt; assistance force unknown',spectral_method='200Hz linear interpolation of robot feedback, remove mean, symmetric Hann, next-power-of-two FFT, one-sided PSD/(Hz), Parseval checked; peak .5-20Hz, band4-10Hz; five-second duration sets spectral resolution ~.2Hz',sessions=sessions)
    prefix.with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
    with prefix.with_suffix('.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['session','direction','start_s','end_s','gyro_x_rms_rad_s','roll_std_deg','gyro_x_peak_hz','gyro_x_band4_10_rms','hip_error_rms_rad','thigh_error_rms_rad','shank_error_rms_rad'])
        for s in sessions:
            for d,m in s['windows'].items():w.writerow([s['name'],d,*m['window_s'],m['gyro_x']['rms'],m['roll_deg']['std'],m['spectra']['gyro_x']['dominant_05_20_hz'],m['spectra']['gyro_x']['band_4_10_rms'],*[m['groups'][k]['error'] for k in ['hip','thigh','shank']]])
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();analyze(a.source,a.output)
