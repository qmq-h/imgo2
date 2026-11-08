#!/usr/bin/env python3
"""Summarize recorded Isaac/Gazebo policy tests; standard library, optional plot."""
import argparse
import ast
import json
import math
from pathlib import Path
import statistics as st
import summarize_foot_clearance as sole


def spread(values):
    return st.pstdev(values) if len(values)>1 else 0.


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--json',type=Path,required=True)
    p.add_argument('--plot',type=Path)
    a=p.parse_args();out=[]
    src=sole.REPO/'imgo2_deploy/ros2_ws/scripts/eval_gazebo_policy.py'
    fn=next(n for n in ast.parse(src.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='autocorr_peak')
    ns={'math':math};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(src),'exec'),ns)
    for label in ['himloco','ppo_policy1']:
        path=a.directory/('isaac_'+label+'.json')
        if path.exists():
            d=json.loads(path.read_text())
            for c in d['cases']:
                rows=c['rows']
                roll=[math.atan2(-r['root'][8],-r['root'][9]) for r in rows]
                pitch=[math.asin(max(-1.,min(1.,r['root'][7]))) for r in rows]
                reconstructed={'odom':[], 'joint_states':[]}
                for r,rr,pp in zip(rows,roll,pitch):
                    reconstructed['odom'].append([r['t'],0.,r['root'][6],0.,rr,pp,0.])
                    reconstructed['joint_states'].append([r['t'],{name:[q,0.] for name,q in zip(d['joint_names'],r['q'])}])
                rp=a.directory/('isaac_'+label+'_'+c['case']+'_geometric_raw.json')
                rp.write_text(json.dumps(reconstructed))
                feet=sole.analyze(rp,sole.REPO/'imgo2_description/urdf/imgo2_real.urdf')
                peaks=[v['height_bout_mean_peak_m'] for v in feet['per_foot'].values() if v['height_bout_mean_peak_m'] is not None]
                frequencies=[]
                if abs(c['mean_vx'])>.05 and c['case'] in ['forward03','backward03']:
                    for leg in ('FL','FR','RL','RR'):
                        idx=d['joint_names'].index(leg+'_thigh_joint');q=[r['q'][idx] for r in rows];mean=st.mean(q)
                        _,_,f=ns['autocorr_peak']([v-mean for v in q],d['dt'])
                        if math.isfinite(f):frequencies.append(f)
                out.append(dict(engine='Isaac',policy=label,case=c['case'],command=c['command'],
                    vx=c['mean_vx'],wz=c['mean_wz'],height_m=c['mean_z'],
                    height_std_m=spread([r['root'][6] for r in rows]),
                    roll_std_deg=math.degrees(spread(roll)),pitch_std_deg=math.degrees(spread(pitch)),
                    gyro_xy_rms=c['gyro_xy_rms'],resets=c['resets'],
                    geometric_swing_peak_m=st.mean(peaks) if peaks else None,
                    autocorr_freq_hz=st.mean(frequencies) if frequencies else None,
                    height_bouts=feet,
                    foot_origin_z_p95_m=st.mean(sole.quantile([r['foot_origin_z'][i] for r in rows],.95) for i in range(4)),
                    limit='Current source task, not original training. Foot origin p95 is not sole clearance.'))
        for case in ['stand','forward03','backward03','turn_left03']:
            path=a.directory/('gazebo_'+label+'_'+case+'.json')
            rawpath=path.with_name(path.stem+'_raw.json')
            if not path.exists() or not rawpath.exists():continue
            d=json.loads(path.read_text());raw=json.loads(rawpath.read_text());od=raw['odom']
            feet=sole.analyze(rawpath,sole.REPO/'imgo2_description/urdf/imgo2_real.urdf')
            pp=list(feet['per_foot'].values())
            valid_locomotion=abs(d['tracking']['speed_mean'])>.05 and case in ['forward03','backward03']
            out.append(dict(engine='Gazebo',policy=label,case=case,command=[d['vx_cmd'],d['vy_cmd'],d['wz_cmd']],
                vx=d['tracking']['speed_mean'],wz=d['tracking']['yaw_rate_mean'],height_m=d['pose']['z_mean'],
                height_std_m=spread([r[2] for r in od]),roll_std_deg=math.degrees(spread([r[4] for r in od])),
                pitch_std_deg=math.degrees(spread([r[5] for r in od])),
                geometric_sole_p95_m=st.mean(v['p95_m'] for v in pp),
                geometric_swing_peak_m=st.mean(v['height_bout_mean_peak_m'] for v in pp if v['height_bout_mean_peak_m'] is not None) if any(v['height_bout_mean_peak_m'] is not None for v in pp) else None,
                autocorr_freq_hz=d['periodicity']['freq_hz'] if valid_locomotion else None,
                height_bouts=feet,
                limit='Gazebo production backend; geometrical feet reconstruction, no measured contact signal. Near-static oscillation not counted as gait frequency.'))
    a.json.write_text(json.dumps(out,indent=2,allow_nan=False))
    print(json.dumps([{k:v for k,v in r.items() if k not in ('height_bouts','limit')} for r in out],indent=2))
    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,3,figsize=(13,7))
        colors={'himloco':'#d45b47','ppo_policy1':'#327fa8'}
        for row,engine in enumerate(['Isaac','Gazebo']):
            for col,(metric,title,scale) in enumerate([('vx','Forward/backward velocity (m/s)',1),('height_m','Body height (cm)',100),('roll_std_deg','Roll variability (deg)',1)]):
                ax=axes[row,col]
                for j,label in enumerate(['himloco','ppo_policy1']):
                    vals=[next((r[metric]*scale for r in out if r['engine']==engine and r['policy']==label and r['case']==case),float('nan')) for case in ['stand','forward03','backward03','turn_left03']]
                    ax.bar([i+(j-.5)*.36 for i in range(4)],vals,width=.36,label=label,color=colors[label])
                if col==0:
                    ax.scatter(range(4),[0,.3,-.3,0],marker='_',s=160,color='black',label='command',zorder=3)
                ax.set_xticks(range(4),['Stand','Fwd .3','Back -.3','Yaw .3'],rotation=15)
                ax.set_title(engine+': '+title);ax.grid(axis='y',alpha=.25)
        axes[0,0].legend(fontsize=8)
        fig.suptitle('Same exported policies; current Isaac task vs Gazebo deployment\nDifferent interfaces/physics: not a reconstruction of the unknown HIM training run',fontsize=11)
        fig.tight_layout(rect=(0,0,1,.91));fig.savefig(a.plot,dpi=160);plt.close(fig)


if __name__=='__main__':main()
