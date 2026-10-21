#include "apps/runtime.h"
#include "apps/deployment_input.h"
#include "apps/deployment_paths.h"
#include "config/deployment_config.h"
#include "display/terminal_display.hpp"
#include "fsm/deployment_states.h"
#include "function/my_myth.h"
#include "sdk/inference_worker.h"
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

namespace {
using Clock=imgo2_sdk::Clock;
using Mode=go_fsm::RobotFSM::Mode;

// 注入给策略的 IMU 姿态，默认取真机实测值（go_monitor 面板读数）：
//   RPY(deg) = 1.076 / -1.613 / -76.094，Gyro(rad/s) = 0.001 / 0.001 / 0.001
// 策略只消费 quaternion 算出的 projected_gravity 和 gyro；yaw 对策略没有影响
// （重力在机体坐标系里的方向与绕竖直轴的旋转无关），保留它只是为了如实复现。
struct InjectedImu {
    double roll_deg=1.076,pitch_deg=-1.613,yaw_deg=-76.094;
    std::array<float,3> gyro{{0.001f,0.001f,0.001f}};
};
InjectedImu g_imu;
std::array<float,4> g_imu_quaternion{{1,0,0,0}};

// ZYX(横滚/俯仰/偏航) 欧拉角 -> 单位四元数 wxyz
std::array<float,4> quaternionFromRpy(double roll,double pitch,double yaw) {
    const double cr=std::cos(roll*.5),sr=std::sin(roll*.5);
    const double cp=std::cos(pitch*.5),sp=std::sin(pitch*.5);
    const double cy=std::cos(yaw*.5),sy=std::sin(yaw*.5);
    return {{float(cr*cp*cy+sr*sp*sy),float(sr*cp*cy-cr*sp*sy),
             float(cr*sp*cy+sr*cp*sy),float(cr*cp*sy-sr*sp*cy)}};
}

// This program replaces feedback only. It never constructs/starts RealHardware.
real_hardware::HardwareState feedback(const std::vector<float>& q,
                                     const std::vector<float>& dq,Clock::time_point now) {
    real_hardware::HardwareState raw;
    raw.started=true;raw.imu_valid=true;raw.joints.resize(12);
    const auto stamp=std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    for(size_t i=0;i<12;++i) {
        auto& joint=raw.joints[i];
        joint.valid=true;joint.age_ms=0;joint.receive_time_ns=stamp;
        joint.feedback.valid=true;joint.feedback.q=q[i];joint.feedback.dq=dq[i];
    }
    auto& imu=raw.imu.data;
    imu.quaternion.assign(g_imu_quaternion.begin(),g_imu_quaternion.end());
    imu.gyroscope.assign(g_imu.gyro.begin(),g_imu.gyro.end());
    imu.euler={float(g_imu.roll_deg*my_myth::degrees_to_radians),
               float(g_imu.pitch_deg*my_myth::degrees_to_radians),
               float(g_imu.yaw_deg*my_myth::degrees_to_radians)};
    imu.valid_fields=imu_sensor::Quaternion|imu_sensor::AngularVelocity|imu_sensor::EulerAngles;
    imu.receive_time_ns=stamp;raw.imu.received=true;raw.imu_age_ms=0;
    return raw;
}

const char* stateName(const go_fsm::RobotFSM& fsm) {
    if(fsm.mode()==Mode::Crouching) return "Passive / Crouching";
    if(fsm.mode()==Mode::Passive) return "Passive / Unloaded";
    return fsm.name();
}

std::string panel(const go_fsm::RobotFSM& fsm,const go::Targets& target,
                  uint64_t sequence,go::ClampResult limit,const std::string& message,
                  const std::vector<float>& actions) {
    std::ostringstream out;
    out<<"State: "<<stateName(fsm)<<" | OFFLINE\n";
    out<<"Target(rad)       Hip         Thigh         Shank\n";
    const char* legs[]={"FL","FR","BL","BR"};
    out<<std::fixed<<std::setprecision(6);
    for(size_t leg=0;leg<4;++leg) {
        out<<legs[leg]<<"       ";
        for(size_t joint=0;joint<3;++joint) out<<std::setw(13)<<target.q[leg*3+joint];
        out<<'\n';
    }
    out<<"Act(raw)          Hip         Thigh         Shank\n";
    out<<std::setprecision(3);
    for(size_t leg=0;leg<4;++leg) {
        out<<legs[leg]<<"       ";
        for(size_t joint=0;joint<3;++joint) {
            if(actions.size()==12) out<<std::setw(13)<<actions[leg*3+joint];
            else out<<std::setw(13)<<"-";
        }
        out<<'\n';
    }
    const auto gravity=my_myth::projectedGravity(g_imu_quaternion);
    out<<"IMU "<<g_imu.roll_deg<<"/"<<g_imu.pitch_deg<<"/"<<g_imu.yaw_deg<<"deg | g("
       <<gravity[0]<<","<<gravity[1]<<","<<gravity[2]<<") | gyro "
       <<g_imu.gyro[0]<<","<<g_imu.gyro[1]<<","<<g_imu.gyro[2]<<'\n';
    const auto& v=fsm.velocityCommand();
    out<<std::setprecision(2)<<"cmd: "<<v[0]<<" "<<v[1]<<" "<<v[2]
       <<" | seq="<<sequence<<" | PD="<<target.kp[0]<<"/"<<target.kd[0]<<'\n';
    out<<"Limit: "<<(limit==go::ClampResult::Clamped?"CLAMP":"OK")<<" | "<<message<<'\n';
    out<<"0/1/2/9 + Enter | WASDQE / Space | Ctrl+C exits\n";
    return out.str();
}
}

int main(int argc,char** argv) try {
    auto policy_root=go_app::executableDirectory()/"policy";
    auto config_file=go_app::executableDirectory()/"config/deployment.yaml";
    double duration_s=0;
    for(int i=1;i<argc;++i) {
        const std::string arg=argv[i];
        if(arg=="--help") {
            std::cout<<"fsm_check [--policy-dir directory] [--config file] [--duration seconds]\n"
                         "          [--imu-rpy roll,pitch,yaw(deg)] [--imu-gyro x,y,z(rad/s)]\n"
                         "Deployment FSM/keyboard/PPO with simulated feedback; no hardware opened.\n"
                         "0/1/2/9 + Enter: stand/PPO/AMP placeholder/crouch then unload.\n"
                         "WASDQE immediate; Space zero; Ctrl+C exits. Duration 0 runs indefinitely.\n"
                         "IMU defaults are the measured robot attitude (1.076/-1.613/-76.094 deg,\n"
                         "gyro 0.001 rad/s); pass --imu-rpy 0,0,0 for the level/ideal case.\n";
            return 0;
        }
        if(arg=="--policy-dir" && i+1<argc) policy_root=argv[++i];
        else if(arg=="--config" && i+1<argc) config_file=argv[++i];
        else if(arg=="--duration" && i+1<argc) {
            const std::string value=argv[++i];size_t used=0;duration_s=std::stod(value,&used);
            if(used!=value.size() || !std::isfinite(duration_s) || duration_s<0)
                throw std::invalid_argument("duration requires finite seconds >=0");
        } else if(arg=="--imu-rpy" && i+1<argc) {
            const std::string value=argv[++i];
            std::istringstream fields(value);std::string field;
            double* slots[3]={&g_imu.roll_deg,&g_imu.pitch_deg,&g_imu.yaw_deg};
            for(int k=0;k<3;++k) {
                if(!std::getline(fields,field,',')) throw std::invalid_argument("imu-rpy needs roll,pitch,yaw in degrees");
                size_t used=0;const double parsed=std::stod(field,&used);
                if(used!=field.size() || !std::isfinite(parsed)) throw std::invalid_argument("imu-rpy requires finite degrees");
                *slots[k]=parsed;
            }
            if(std::getline(fields,field,',')) throw std::invalid_argument("imu-rpy takes exactly 3 values");
        } else if(arg=="--imu-gyro" && i+1<argc) {
            const std::string value=argv[++i];
            std::istringstream fields(value);std::string field;
            for(int k=0;k<3;++k) {
                if(!std::getline(fields,field,',')) throw std::invalid_argument("imu-gyro needs x,y,z in rad/s");
                size_t used=0;const double parsed=std::stod(field,&used);
                if(used!=field.size() || !std::isfinite(parsed)) throw std::invalid_argument("imu-gyro requires finite rad/s");
                g_imu.gyro[size_t(k)]=float(parsed);
            }
            if(std::getline(fields,field,',')) throw std::invalid_argument("imu-gyro takes exactly 3 values");
        } else throw std::invalid_argument("unknown/missing argument: "+arg);
    }
    g_imu_quaternion=quaternionFromRpy(g_imu.roll_deg*my_myth::degrees_to_radians,
                                       g_imu.pitch_deg*my_myth::degrees_to_radians,
                                       g_imu.yaw_deg*my_myth::degrees_to_radians);
    {
        const auto injected=my_myth::projectedGravity(g_imu_quaternion);
        std::cout<<std::setprecision(4)<<"fsm_check: injected IMU rpy(deg)="<<g_imu.roll_deg<<"/"
                 <<g_imu.pitch_deg<<"/"<<g_imu.yaw_deg<<" gyro(rad/s)="<<g_imu.gyro[0]<<"/"
                 <<g_imu.gyro[1]<<"/"<<g_imu.gyro[2]<<" -> projected_gravity=("<<injected[0]<<", "
                 <<injected[1]<<", "<<injected[2]<<")\n";
    }
    auto deployment=imgo2_sdk::DeploymentConfig::load(config_file);
    // Synthetic index mapping is usable for this test only; never change the real YAML lock.
    deployment.mapping_confirmed=true;
    const auto policy=imgo2_sdk::PolicyConfig::load(policy_root/"ppo");
    const go::MotorConfig motors;
    imgo2_sdk::RobotAdapter adapter(deployment.policy_to_hardware);
    std::vector<float> simulated_q,simulated_dq(12,0);
    for(const auto& bus:motors.buses)
        simulated_q.insert(simulated_q.end(),bus.start_pose_rad.begin(),bus.start_pose_rad.end());
    imgo2_sdk::PolicyRunner runner(policy);runner.load(deployment.torch_threads);runner.warmup();
    const auto start=Clock::now();
    go_fsm::RobotFSM fsm(policy,deployment,motors,adapter,simulated_q,start);
    imgo2_sdk::InferenceWorker worker(
        [&](const auto& s,const auto& v,uint64_t generation){return runner.infer(s,v,generation);},
        [&]{runner.reset();},deployment.policy_ms,deployment.result_timeout_ms);
    go_app::StopSignals signals;
    go_app::TerminalInput terminal;
    go_app::DeploymentInput input;
    imu_sensor::TerminalDisplay display;
    worker.start();
    auto next=start,refresh=start;
    auto previous_mode=fsm.mode();
    std::string message="Simulated feedback; no devices opened";
    int exit_code=0;
    while(!signals.requested()) {
        const auto now=Clock::now();
        if(duration_s>0 && std::chrono::duration<double>(now-start).count()>=duration_s) break;
        // Trajectories ideally follow the previous command. PPO always sees the fixed stand pose,
        // zero joint velocity and the injected IMU attitude (see --imu-rpy/--imu-gyro); the
        // policy output is never fed back as joint feedback, so this is an open-loop fixed point.
        auto raw=feedback(fsm.policyActive()?deployment.stand_pose:simulated_q,
                          fsm.policyActive()?std::vector<float>(12,0):simulated_dq,now);
        for(int n=0;n<64;++n) {
            char c=0;const auto status=go_app::readDeploymentChar(c);
            if(status==go_app::InputStatus::Waiting) break;
            if(status==go_app::InputStatus::Closed) {go_app::stopFlag()=1;break;}
            const auto event=input.feed(c);
            if(event.kind==go_app::KeyEvent::Kind::State) {
                const auto reply=fsm.request(event.digit,raw,now);
                if(!reply.empty()) {message=reply;refresh=now;}
            } else if(event.kind==go_app::KeyEvent::Kind::Velocity) fsm.velocity(event.key);
            // Also use stand feedback on the very first PPO observation after switching.
            if(fsm.policyActive()) raw=feedback(deployment.stand_pose,std::vector<float>(12,0),now);
        }
        if(signals.requested()) break;
        const auto result=worker.latest();
        auto target=fsm.update(raw,result,Clock::now());
        const auto limit=go::Safety::clamp(target,motors.limits);
        if(limit==go::ClampResult::Rejected) throw std::runtime_error("Motor safety rejected target");
        if(fsm.mode()==Mode::Fault) {message=fsm.error();exit_code=2;}
        if(previous_mode!=fsm.mode()) {refresh=now;previous_mode=fsm.mode();}
        if(!fsm.policyActive()) {simulated_q=target.q;simulated_dq=target.dq;}
        // On PPO exit, the motion FSM starts from its fixed synthetic standing feedback.
        auto observation=adapter.toPolicy(raw);observation.captured_at=now;
        worker.submit(observation,fsm.velocityCommand(),fsm.generation(),fsm.policyActive());
        if(now>=refresh) {
            const bool fresh=fsm.policyActive() && result.generation==fsm.generation();
            const uint64_t sequence=fresh?result.sequence:0;
            display.Show(panel(fsm,target,sequence,limit,message,fresh?result.actions:std::vector<float>{}));
            refresh=now+std::chrono::milliseconds(100);
        }
        next+=std::chrono::milliseconds(deployment.control_ms);
        if(next<Clock::now()) next=Clock::now();
        std::this_thread::sleep_until(next);
    }
    worker.stop();return exit_code;
} catch(const std::exception& error) {
    std::cerr<<"FSM check error: "<<error.what()<<'\n';return 1;
}
