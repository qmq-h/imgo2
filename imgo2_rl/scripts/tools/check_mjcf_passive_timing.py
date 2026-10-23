#!/usr/bin/env python3
"""Reproduce Passive sampled-PD timing sensitivity without PPO/ROS/hardware.

Python stdlib orchestrates a bounded C++ diagnostic linked to local MuJoCo.
Armature and initial-pose variants modify only in-memory models.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]

CPP = '#include <mujoco/mujoco.h>\n#include <cstdio>\n#include <cmath>\n#include <algorithm>\nint main(int argc,char**argv){for(int mode=0;mode<5;mode++){char e[2048]={};auto*m=mj_loadXML(argv[1],nullptr,e,sizeof(e));if(!m){puts(e);return 1;}auto*d=mj_makeData(m);int period=mode==1?1:mode==2?2:10;bool crouch=mode==4;if(mode==3)for(int a=0;a<12;a++){int j=m->actuator_trnid[2*a];m->dof_armature[m->jnt_dofadr[j]]=.1;}if(crouch)for(int a=0;a<12;a++){int j=m->actuator_trnid[2*a];d->qpos[m->jnt_qposadr[j]]=a%3==0?0:a%3==1?1.36:-2.65;}mj_forward(m,d);double sum=0,power=0,z=0,z2=0;int n=0;for(int i=0;i<10000;i++){if(i%period==0)for(int a=0;a<12;a++){int j=m->actuator_trnid[2*a];d->ctrl[a]=std::clamp(-8*d->qvel[m->jnt_dofadr[j]],-23.7,23.7);}mj_step(m,d);if(i>=8000){for(int a=0;a<12;a++){int j=m->actuator_trnid[2*a];double v=d->qvel[m->jnt_dofadr[j]];sum+=v*v;power+=d->actuator_force[a]*v;}z+=d->qpos[2];z2+=d->qpos[2]*d->qpos[2];n++;}}printf("mode=%d PD_Hz=%g armature=%g initial=%s final1s_joint_dq_rms=%g power_W=%g z_std=%g\\n",mode,1/(period*m->opt.timestep),mode==3?.1:0,crouch?"crouch":"zero",sqrt(sum/(n*12)),power/n,sqrt(std::max(0.,z2/n-z*z/n/n)));mj_deleteData(d);mj_deleteModel(m);}}\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mujoco-dir', type=Path, default=ROOT/'imgo2_deploy/ros2_ws/library/mujoco')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    lib = a.mujoco_dir.resolve()
    with tempfile.TemporaryDirectory(prefix='imgo2_passive_timing_') as tmp:
        source, binary = Path(tmp)/'check.cpp', Path(tmp)/'check'
        source.write_text(CPP)
        subprocess.run(['c++','-std=c++17',str(source),'-I'+str(lib/'include'),'-L'+str(lib/'lib'),'-Wl,-rpath,'+str(lib/'lib'),'-lmujoco','-o',str(binary)],check=True)
        run = subprocess.run([str(binary),str(ROOT/'imgo2_description/mjcf/scene.xml')],check=True,text=True,capture_output=True,timeout=60)
    cases = []
    for line in run.stdout.splitlines():
        values = dict(part.split('=',1) for part in line.split())
        cases.append({k: v if k == 'initial' else float(v) for k,v in values.items()})
    result = {'scope': 'Five 5-second independent Passive PD experiments, last-second metrics. No production FSM or PPO.', 'cases': cases}
    if a.output:
        a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
