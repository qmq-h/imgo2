#!/usr/bin/env python3
"""Standard-library independent DFT/hash/window checks for the new batch."""
import argparse,cmath,hashlib,json,math,statistics as st
from pathlib import Path
from analyze_new_motion_batch import load,prepare,interp
from analyze_leg_rack_batch import build,harmonic
from audit_joint_sine_log import phase_difference

def validate(prefix,folder):
    result=json.loads(Path(str(prefix)+'_walk.json').read_text());source=Path(result['source']);manifest=json.loads((source/'archive_manifest.json').read_text())
    # Manifest layout comes from the extractor; every member must still match.
    assert hashlib.sha256((source/'source_original.7z').read_bytes()).hexdigest()==manifest['sha256']
    for entry in manifest['entries']:
        path=source/entry['path'];assert path.stat().st_size==entry['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
    max_psd_error=0.;max_parseval=0.;selected=0
    for session in result['sessions']:
        root=Path(session['source']);rows=load(root/'robot.csv');prepare(rows);t=[r['t_mono_ms']/1000 for r in rows]
        for name,h in session['sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h
        for m in session['windows'].values():
            assert m['quality']['kp_range']==[25.,25.] and m['quality']['kd_range']==[.8,.8]
            lo,hi=m['window_s'];assert abs(hi-lo-5)<1e-10;selected+=1
            for key,spec in m['spectra'].items():
                yy=[r[key] for r in rows];x=[interp(t,yy,lo+i/200) for i in range(1000)];mean=st.fmean(x)
                w=[.5-.5*math.cos(2*math.pi*i/999) for i in range(1000)];z=[(v-mean)*a for v,a in zip(x,w)];scale=200*sum(a*a for a in w)
                for k in [0,1,5,20,35,40,100,512]:
                    d=sum(v*cmath.exp(-2j*math.pi*k*i/1024) for i,v in enumerate(z));p=abs(d)**2/scale*(1 if k in (0,512) else 2)
                    max_psd_error=max(max_psd_error,abs(p-spec['psd'][k]))
                max_parseval=max(max_parseval,spec['parseval_abs_error'])
    assert max_psd_error<1e-10 and max_parseval<1e-10
    coarse=json.loads((folder/'isaac/scores.json').read_text());fine=json.loads((folder/'isaac_fine/scores.json').read_text())
    steps=[dict(source_id=a['source_id'],candidate=a['candidate_id'],gain_percentage_point_change=100*(b['gain']-a['gain']),phase_deg_change=b['phase_deg']-a['phase_deg']) for a,b in zip(coarse['jobs'],fine['jobs'])]
    protected={}
    for sub in ['isaac','isaac_fine']:
        meta=json.loads((folder/sub/'metadata.json').read_text());assert meta['completed'] and meta['production_files_unchanged']
        for path,h in meta['protected_hashes'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
        protected[sub]=dict(instances=meta['instances'],dt_s=meta['dt_s'],max_base_drift_m=meta['max_base_drift_m'])
    for file in (folder/'gazebo').glob('case_*/sine_metadata.json'):
        meta=json.loads(file.read_text());assert meta['completed'] and meta['production_files_unchanged']
        for path,h in meta['protected_hashes'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
    plan=json.loads((folder/'plan.json').read_text());gz=[]
    for item in plan['sources']:
        case,streams=build(Path(item['source']),item['id'],None,None);real=streams['real'];ids=[i for i,t in enumerate(real['t']) if 1<=t<8.5]
        ref=harmonic([real['ref_t'][i] for i in ids],[real['ref'][i] for i in ids],2);rf=harmonic([real['t'][i] for i in ids],[real['q'][i] for i in ids],2)
        real_gain=rf['amplitude']/ref['amplitude'];real_phase=phase_difference(rf['phase_deg'],ref['phase_deg']);index=['FL_hip','FL_thigh','FL_shank'].index(case['joint'])
        for variant in ['baseline','previous_candidate']:
            path=folder/'gazebo'/f"case_{item['id']:02}_{variant}";meta=json.loads((path/'sine_metadata.json').read_text());assert meta['source_sha256']==item['sha256']
            rr=load(path/'run/replay.csv');rr=[r for r in rr if 4<=r['t_s']<11.5 and math.isfinite(r[f'controller_q_{index:02}'])];tt=[r['t_s']-3 for r in rr];q=[r[f'controller_q_{index:02}'] for r in rr];qfit=harmonic(tt,q,2);r=[interp(real['t'],real['q'],t) for t in tt];mq=st.fmean(q);mr=st.fmean(r)
            gz.append(dict(source_id=item['id'],joint=case['joint'],variant=variant,window_s=[1,8.5],real_gain=real_gain,real_phase_deg=real_phase,gain=qfit['amplitude']/ref['amplitude'],phase_deg=phase_difference(qfit['phase_deg'],ref['phase_deg']),ac_rmse_rad=math.sqrt(st.fmean(((a-mq)-(b-mr))**2 for a,b in zip(q,r))),mean_q_error_rad=mq-mr))
    Path(str(prefix)+'_gazebo.json').write_text(json.dumps(gz,indent=2)+'\n')
    answer=dict(archive_sha256=manifest['sha256'],archive_members_verified=len(manifest['entries']),selected_windows=selected,spectra_checked=40,fft_vs_direct_dft_max_abs_psd_error=max_psd_error,parseval_max_abs_error=max_parseval,isaac=protected,step_differences=steps,source_walk_hashes_unchanged=True,production_hashes_unchanged=True,gazebo_completed=len(list((folder/'gazebo').glob('case_*/sine_metadata.json'))))
    assert answer['gazebo_completed']==12
    Path(str(prefix)+'_validation.json').write_text(json.dumps(answer,indent=2)+'\n');print(json.dumps(answer,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prefix',type=Path,required=True);p.add_argument('--validation',type=Path,required=True);a=p.parse_args();validate(a.prefix,a.validation)
