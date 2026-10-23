#!/usr/bin/env python3
"""Optional headless model smoke check using installed MuJoCo and a C++ compiler.

Python uses only stdlib. No policy, ROS, viewer or physical hardware is started.
The bounded PD holds are diagnostics, not the production FSM or a PPO rollout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]
CPP = r'''
#include <mujoco/mujoco.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
int main(int argc,char**argv){
 char error[2048]={};mjModel*m=mj_loadXML(argv[1],nullptr,error,sizeof(error));
 if(!m){std::fprintf(stderr,"%s\n",error);return 1;}
 if(mj_version()!=327 || mjVERSION_HEADER!=327 || m->nu!=12)return 2;
 auto*d=mj_makeData(m);double mass=0;for(int b=1;b<m->nbody;b++)mass+=m->body_mass[b];
 int orientation=mj_name2id(m,mjOBJ_SENSOR,"orientation"),gyro=mj_name2id(m,mjOBJ_SENSOR,"angular_velocity");
 if(orientation<0||gyro<0||m->sensor_adr[orientation]!=36||m->sensor_adr[gyro]!=40)return 3;
 double results[2][5]={};int warnings=0;
 for(int run=0;run<2;run++){
  mj_resetData(m,d);d->qpos[2]=.35;
  for(int a=0;a<12;a++){int j=m->actuator_trnid[2*a];d->qpos[m->jnt_qposadr[j]]=a%3==0?0:a%3==1?.87:-1.82;
   int v=m->jnt_dofadr[j];if(m->dof_damping[v]!=0||m->dof_armature[v]!=0)return 4;}
  mj_forward(m,d);int n=(int)std::lround(10/m->opt.timestep),period=(int)std::lround(.005/m->opt.timestep);
  double maxerr=0,minz=1e9,maxz=-1e9,footfriction=-1,contacts=0;
  for(int i=0;i<n;i++){
   if(i%period==0)for(int a=0;a<12;a++){int j=m->actuator_trnid[2*a];double target=a%3==0?0:a%3==1?.87:-1.82;
    double tau=(run==0?60:25)*(target-d->qpos[m->jnt_qposadr[j]])-(run==0?2:.5)*d->qvel[m->jnt_dofadr[j]];
    d->ctrl[a]=std::clamp(tau,-23.7,23.7);}
   mj_step(m,d);
   for(int q=0;q<m->nq;q++)if(!std::isfinite(d->qpos[q]))return 5;
   if(i>n*9/10){minz=std::min(minz,d->qpos[2]);maxz=std::max(maxz,d->qpos[2]);
    for(int a=0;a<12;a++){int j=m->actuator_trnid[2*a];double target=a%3==0?0:a%3==1?.87:-1.82;maxerr=std::max(maxerr,std::abs(target-d->qpos[m->jnt_qposadr[j]]));}
    for(int c=0;c<d->ncon;c++){auto&ct=d->contact[c];int b1=m->geom_bodyid[ct.geom1],b2=m->geom_bodyid[ct.geom2];const char*s1=mj_id2name(m,mjOBJ_BODY,b1),*s2=mj_id2name(m,mjOBJ_BODY,b2);
     bool foot=(s1&&std::string(s1).find("FOOT")!=std::string::npos)||(s2&&std::string(s2).find("FOOT")!=std::string::npos);
     if(foot&&(b1==0||b2==0)){footfriction=ct.friction[0];contacts++;if(std::abs(footfriction-.6)>1e-12)return 6;}}
   }
  }
  for(int w=0;w<mjNWARNING;w++)warnings+=d->warning[w].number;
  results[run][0]=minz;results[run][1]=maxz;results[run][2]=maxerr;results[run][3]=footfriction;results[run][4]=contacts;
  // Gravity-loaded PD has static position error; report it rather than treating
  // an arbitrary angular tracking tolerance as a model validity requirement.
  if(minz<.15||maxz>.6||contacts==0){std::fprintf(stderr,"hold %d: z[%g,%g] maxerr=%g footfriction=%g contacts=%g\n",run,minz,maxz,maxerr,footfriction,contacts);return 7;}
 }
 std::printf("{\"version\":\"%s\",\"nq\":%d,\"nv\":%d,\"nu\":%d,\"body_count_without_world\":%d,\"mass_kg\":%.12g,\"timestep\":%.12g,\"warnings\":%d,\"holds\":[",mj_versionString(),m->nq,m->nv,m->nu,m->nbody-1,mass,m->opt.timestep,warnings);
 for(int r=0;r<2;r++)std::printf("%s{\"kp\":%g,\"kd\":%g,\"seconds\":10,\"last_second_z_min\":%.12g,\"last_second_z_max\":%.12g,\"last_second_max_joint_error_rad\":%.12g,\"foot_floor_friction\":%.12g}",r?",":"",r?25.:60.,r?.5:2.,results[r][0],results[r][1],results[r][2],results[r][3]);
 std::printf("]}\n");mj_deleteData(d);mj_deleteModel(m);return warnings?8:0;
}
'''.replace('#include <stdexcept>', '#include <stdexcept>\n#include <string>')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mujoco-dir', type=Path, default=ROOT/'imgo2_deploy/ros2_ws/library/mujoco')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    lib = args.mujoco_dir.resolve()
    result = {'scope': 'Three scenes load and bounded 60/2,25/.5 PD holds; no production FSM/PPO/hardware.', 'scenes': {}}
    with tempfile.TemporaryDirectory(prefix='imgo2_mjcf_runtime_') as tmp:
        cpp, binary = Path(tmp)/'check.cpp', Path(tmp)/'check'
        cpp.write_text(CPP)
        subprocess.run(['c++','-std=c++17',str(cpp),'-I'+str(lib/'include'),'-L'+str(lib/'lib'),'-Wl,-rpath,'+str(lib/'lib'),'-lmujoco','-o',str(binary)], check=True)
        for name in ('scene','scene_tow_compliant','scene_tow_inextensible'):
            path = ROOT/'imgo2_description/mjcf'/f'{name}.xml'
            proc = subprocess.run([str(binary),str(path)], text=True,capture_output=True,timeout=60)
            if proc.returncode:
                raise RuntimeError(f'{name}: exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}')
            result['scenes'][name] = json.loads(proc.stdout)
    result['robot_sha256'] = hashlib.sha256((ROOT/'imgo2_description/mjcf/imgo2.xml').read_bytes()).hexdigest()
    if args.output:
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
