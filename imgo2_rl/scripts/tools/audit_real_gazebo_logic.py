#!/usr/bin/env python3
"""Read-only PPO contract audit; optional C++ math check, never hardware/ROS.

The math check compiles real SDK sources and unmodified extracted Gazebo
ComputeObservation/ComputeOutput bodies. Config-container and unused mimic
dependencies are stubs: this is not a ROS runtime, scheduler or plant test.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]
REAL = ROOT/'imgo2_deploy/real'
GAZ = ROOT/'imgo2_deploy/ros2_ws/src/imgo2_deploy'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def field(path, name):
    m = re.search(r'(?m)^\s{2}'+name+r':\s*(\[[\s\S]*?\]|[^\n#]+)',path.read_text())
    if not m:raise ValueError('missing '+name+' in '+str(path))
    return ast.literal_eval(m.group(1).strip())


def function(text, signature):
    start = text.index(signature)
    opening = text.index('{', start)
    level = 0
    for end in range(opening,len(text)):
        if text[end]=='{':level+=1
        elif text[end]=='}':
            level-=1
            if level==0:return text[start:end+1]
    raise ValueError('unbalanced function '+signature)


def numbers(values):
    return '{'+','.join(str(float(x))+'f' for x in values)+'}'


def numeric(config, source, samples):
    bodies='\n'.join(function(source,sig) for sig in (
        'std::vector<float> RL::ComputeObservation()', 'void RL::ComputeOutput('))
    preamble=r'''
#include "sdk/rl_sdk.h"
#include "vector_math.hpp"
#include <any>
#include <map>
#include <random>
#include <iostream>
#include <iomanip>
struct Params {
    std::map<std::string,std::any> data;
    template<class T>T Get(const std::string& k) const {return std::any_cast<T>(data.at(k));}
};
struct MotionLoader {
    std::vector<float> GetJointPos(){return {};}
    std::vector<float> GetJointVel(){return {};}
    std::vector<float> GetAnchorQuat(){return {};}
    std::vector<float> GetInitQuat(){return {};}
    static std::vector<float> ComputeTorsoQuat(const std::vector<float>& q,const std::vector<float>&){return q;}
};
struct Obs {std::vector<float> lin_vel,ang_vel,gravity_vec,commands,base_quat,dof_pos,dof_vel,actions;};
struct RL {
    Params params;Obs obs;std::vector<int> obs_dims;std::string ang_vel_axis="body";
    unsigned long long episode_length_buf=0;float motion_length=1;
    MotionLoader* motion_loader=nullptr;
    int InverseJointMapping(int i)const{return i;}
    std::vector<float> ComputeObservation();
    void ComputeOutput(const std::vector<float>&,std::vector<float>&,std::vector<float>&,std::vector<float>&);
};
'''
    init=['int main(){RL g;imgo2_sdk::PolicyConfig c;']
    arrays={'default_dof_pos':'default_pose','action_scale':'action_scale',
            'clip_actions_lower':'lower','clip_actions_upper':'upper',
            'rl_kp':'kp','rl_kd':'kd','torque_limits':'torque_limits'}
    for key,dest in arrays.items():
        init.append(f'c.{dest}={numbers(config[key])};g.params.data["{key}"]=c.{dest};')
    for key,dest in [('ang_vel_scale','ang_vel_scale'),('dof_pos_scale','dof_pos_scale'),
                     ('dof_vel_scale','dof_vel_scale'),('clip_obs','clip_obs')]:
        init.append(f'c.{dest}={float(config[key])}f;g.params.data["{key}"]=c.{dest};')
    init.append('c.commands_scale='+numbers(config['commands_scale'])+';')
    init.append('g.params.data["commands_scale"]=std::vector<float>(c.commands_scale.begin(),c.commands_scale.end());')
    init.append('g.params.data["observations"]=std::vector<std::string>{"ang_vel","gravity_vec","commands","dof_pos","dof_vel","actions"};')
    init.append('g.params.data["wheel_indices"]=std::vector<int>{};')
    init.append('g.params.data["lin_vel_scale"]=2.f;')
    loop=r'''
std::mt19937 rng(180011);std::uniform_real_distribution<float> u(-1,1);
double max_obs=0,max_action=0,max_target=0,max_dq=0;
for(int n=0;n<SAMPLES;++n){
    imgo2_sdk::RobotState s;s.valid=true;s.q.resize(12);s.dq.resize(12);
    double norm=0;for(float& x:s.quaternion){x=u(rng);norm+=double(x)*x;}
    for(float& x:s.quaternion)x=float(x/std::sqrt(norm));
    for(float& x:s.gyro)x=u(rng)*(n%2?600:3);
    imgo2_sdk::Velocity cmd{{u(rng),u(rng),u(rng)}};
    std::vector<float> last(12),raw(12);
    for(int i=0;i<12;++i){s.q[i]=c.default_pose[i]+u(rng);s.dq[i]=u(rng)*(n%2?2500:10);last[i]=3*u(rng);raw[i]=6*u(rng);}
    auto ro=imgo2_sdk::buildObservation(s,cmd,last,c);
    g.obs.ang_vel={s.gyro.begin(),s.gyro.end()};g.obs.base_quat={s.quaternion.begin(),s.quaternion.end()};
    g.obs.gravity_vec={0,0,-1};g.obs.commands={cmd.begin(),cmd.end()};
    g.obs.dof_pos=s.q;g.obs.dof_vel=s.dq;g.obs.actions=last;
    auto so=g.ComputeObservation();if(so.size()!=45)return 2;
    for(int i=0;i<45;++i)max_obs=std::max(max_obs,double(std::abs(ro[i]-so[i])));
    auto ra=imgo2_sdk::clipActions(raw,c);auto sa=clamp(raw,c.lower,c.upper);
    auto target=imgo2_sdk::convertAction(ra,c);std::vector<float> pos,vel,diagnostic_tau;
    g.ComputeOutput(sa,pos,vel,diagnostic_tau);
    for(int i=0;i<12;++i){
        max_action=std::max(max_action,double(std::abs(ra[i]-sa[i])));
        max_target=std::max(max_target,double(std::abs(target.q[i]-pos[i])));
        max_dq=std::max(max_dq,double(std::abs(target.dq[i]-vel[i])));
        if(target.tau[i]!=0 || target.kp[i]!=c.kp[i] || target.kd[i]!=c.kd[i])return 3;
    }
}
std::cout<<std::setprecision(12)<<"{\"samples\":"<<SAMPLES
    <<",\"max_observation_abs_error\":"<<max_obs
    <<",\"max_clipped_action_abs_error\":"<<max_action
    <<",\"max_q_target_abs_error_rad\":"<<max_target
    <<",\"max_dq_target_abs_error_rad_s\":"<<max_dq<<"}\n";
return max_obs>2e-6 || max_action>0 || max_target>1e-6 || max_dq>0;
}
'''.replace('SAMPLES',str(samples))
    with tempfile.TemporaryDirectory(prefix='imgo2_logic_audit_') as temp:
        temp=Path(temp);cpp=temp/'compare.cpp';exe=temp/'compare'
        cpp.write_text(preamble+bodies+'\n'+'\n'.join(init)+loop)
        cmd=['/usr/bin/g++','-std=c++17','-O2','-I'+str(REAL),'-I'+str(REAL/'Motor/inc'),'-I'+str(REAL/'config'),
             '-I'+str(REAL/'serialport/inc'),'-I'+str(GAZ/'library/core/vector_math'),str(cpp),str(REAL/'sdk/rl_sdk.cpp'),
             str(REAL/'function/my_myth.cpp'),'-o',str(exe)]
        compiled=subprocess.run(cmd,capture_output=True,text=True)
        if compiled.returncode:raise RuntimeError('numeric harness compilation failed:\n'+compiled.stderr)
        ran=subprocess.run([str(exe)],check=True,capture_output=True,text=True)
        result=json.loads(ran.stdout)
        result['method']='real SDK translation units + verbatim extracted Gazebo functions; stub config container/unused mimic dependency, actual vector_math.hpp; unit random quaternions and finite normal/extreme inputs'
        result['scope']='math functions only; no model, ROS, scheduler, motor firmware, physical plant or real hardware executed'
        return result


def main(a):
    realcfg=REAL/'policy/ppo/config.yaml';simcfg=ROOT/'imgo2_deploy/ros2_ws/policy/imgo2/ppo/config.yaml'
    keys=['num_observations','observations','observations_history','num_of_dofs','wheel_indices',
          'clip_obs','clip_actions_lower','clip_actions_upper','rl_kp','rl_kd','fixed_kp','fixed_kd',
          'action_scale','ang_vel_scale','dof_pos_scale','dof_vel_scale','commands_scale','default_dof_pos','torque_limits']
    real={k:field(realcfg,k) for k in keys};sim={k:field(simcfg,k) for k in keys}
    differences={k:{'real':real[k],'gazebo':sim[k]} for k in keys if real[k]!=sim[k]}
    assert real['observations']==['ang_vel','gravity_vec','commands','dof_pos','dof_vel','actions']
    assert real['num_observations']==45 and not real['wheel_indices'] and not real['observations_history']
    base=ROOT/'imgo2_deploy/ros2_ws/policy/imgo2/base.yaml';deploy=REAL/'config/deployment.yaml'
    assert field(deploy,'control_ms')==1000*field(base,'dt')==5
    assert field(deploy,'policy_ms')==1000*field(base,'dt')*field(base,'decimation')==20
    source=GAZ/'library/core/rl_sdk/rl_sdk.cpp';text=source.read_text()
    paths=[realcfg,simcfg,base,deploy,source,GAZ/'src/rl_sim.cpp',GAZ/'fsm_robot/fsm_imgo2.hpp',
        GAZ/'library/core/loop/loop.hpp',GAZ/'library/core/vector_math/vector_math.hpp',
        REAL/'sdk/rl_sdk.cpp',REAL/'sdk/policy_runner.cpp',REAL/'sdk/inference_worker.cpp',
        REAL/'function/my_myth.cpp',REAL/'rl_real_imgo2.cpp',REAL/'hardware/robot_adapter.cpp',
        REAL/'hardware/real_hardware.cpp',REAL/'fsm/states/ppo_state.cpp',REAL/'fsm/deployment_states.cpp',
        REAL/'Imu/src/yesense_parser.cpp',REAL/'Imu/src/imu_monitor.cpp',
        REAL/'input/controller_input.cpp',GAZ/'include/policy_action_clip.hpp',
        REAL/'Motor/src/go_protocol.cpp',REAL/'Motor/src/go_safety.cpp',REAL/'Motor/src/go_motor_array.cpp',
        ROOT/'imgo2_deploy/ros2_ws/src/robot_joint_controller/ros2/src/robot_joint_controller_group.cpp',
        ROOT/'imgo2_description/config/robot_control_ros2.yaml']
    weights=[REAL/'policy/ppo/policy.pt',ROOT/'imgo2_deploy/ros2_ws/policy/imgo2/ppo/policy.pt']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths+weights}
    result={'real_PPO_contract':real,'gazebo_PPO_contract':sim,
        'configuration_differences':differences,'configuration_equal_for_listed_keys':not differences,
        'nominal_main_policy_period_ms':[5,20],'sha256':hashes,
        'current_default_weight_files_identical':sha(weights[0])==sha(weights[1]),
        'PPO_sent_feedforward_zero_source_check':'fsm_command->motor_command.tau[i] = 0;' in text,
        'limitations':['This is an audit of local current sources, not proof of deployed binary/source identity.',
            'Identical equations do not imply identical sensor timing, firmware PD, saturation or contact dynamics.',
            'Previous explicit saved-policy simulations did not necessarily load either current default weight.']}
    if a.numeric:
        result['numeric']=numeric(real,text,a.samples)
        result['numeric']['configuration']='Both math implementations supplied the same real YAML parameters for this equivalence check; current default Kd difference is reported separately.'
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    result['audited_source_files_unchanged']=True
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('configuration_equal_for_listed_keys','current_default_weight_files_identical','audited_source_files_unchanged')},indent=2))
    if a.numeric:print(json.dumps(result['numeric'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--numeric',action='store_true')
    p.add_argument('--samples',type=int,default=2048)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not 1<=a.samples<=100000:p.error('samples out of bounds')
    main(a)
