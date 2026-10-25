#include "deployment_states.h"
#include "states/ppo_state.h"
#include "states/hold_state.h"
#include "states/fault_state.h"
#include "function/my_myth.h"
#include <cmath>
#include <cstdio>
#include <string>
namespace go_fsm {
namespace {
std::vector<float> measured(const real_hardware::HardwareState& h) {
    if(h.joints.size()!=12) throw std::invalid_argument("hardware requires 12 joints");
    std::vector<float> q;for(const auto& j:h.joints) {
        if(!j.valid || !std::isfinite(j.feedback.q)) throw std::invalid_argument("invalid motion feedback");
        q.push_back(j.feedback.q);
    }
    return q;
}
}
RobotFSM::RobotFSM(const imgo2_sdk::PolicyConfig& policy,const imgo2_sdk::DeploymentConfig& settings,
                   const go::MotorConfig& motors,const imgo2_sdk::RobotAdapter& adapter,
                   const std::vector<float>& initial,Clock::time_point now)
    :motors_(motors),adapter_(adapter) {
    settings.validate();c_.settings=settings;c_.stand_duration_s=settings.stand_s;
    c_.fixed.q=initial;c_.fixed.dq.assign(12,0);c_.fixed.tau.assign(12,0);
    c_.fixed.kp=adapter_.positionsToHardware(policy.fixed_kp);c_.fixed.kd=adapter_.positionsToHardware(policy.fixed_kd);
    c_.command=c_.fixed;c_.goal=c_.fixed;c_.previous=c_.fixed;
    stand_goal_=settings.stand_pose; // Fixed pose is independently configured in hardware order.
    auto goal=c_.fixed;std::string error;
    goal.q=stand_goal_;
    if(!go::Safety::trajectoryValid(c_.fixed,goal,settings.stand_s,motors_.limits,error)) throw std::invalid_argument(error);
    auto crouch=goal;crouch.q=settings.crouch_pose;
    if(!go::Safety::trajectoryValid(goal,crouch,settings.crouch_s,motors_.limits,error)) throw std::invalid_argument(error);
    machine_.addState(std::unique_ptr<State>(new states::HoldState(c_,"InitialHold")));
    auto stand=std::unique_ptr<states::StandState>(new states::StandState(c_));stand_=stand.get();machine_.addState(std::move(stand));
    machine_.addState(std::unique_ptr<State>(new states::HoldState(c_,"StandHold")));
    machine_.addState(std::unique_ptr<State>(new states::PPOState(c_)));
    auto down=std::unique_ptr<states::PassiveState>(new states::PassiveState(c_));crouch_=down.get();machine_.addState(std::move(down));
    machine_.addState(std::unique_ptr<State>(new states::FaultState(c_)));
    machine_.start(now);
}
void RobotFSM::change(Mode mode,const char* name,Clock::time_point now) {
    c_.previous=c_.command;++c_.generation;
    velocity_={{0,0,0}};
    settling_=false;
    mode_=mode;
    machine_.switchTo(name,now);
}
bool RobotFSM::beginMotion(Mode mode,const char* name,const std::vector<float>& target,double duration,
                           const real_hardware::HardwareState& h,Clock::time_point now,std::string& error) {
    auto start=c_.command;start.q=measured(h);start.dq.assign(12,0);
    auto goal=c_.fixed;goal.q=target;
    if(!go::Safety::trajectoryValid(start,goal,duration,motors_.limits,error)) return false;
    c_.measured=start.q;
    if(mode==Mode::Standing) c_.stand_duration_s=duration;
    change(mode,name,now);return true;
}
bool RobotFSM::ready(const real_hardware::HardwareState& h,Clock::time_point now) {
    if(mode_!=Mode::StandHold || h.faulted || !h.imu_valid || h.joints.size()!=12) {settling_=false;return false;}
    for(size_t i=0;i<12;++i) if(!h.joints[i].valid || !std::isfinite(h.joints[i].feedback.q) || !std::isfinite(h.joints[i].feedback.dq) ||
       std::abs(h.joints[i].feedback.q-stand_goal_[i])>c_.settings.position_tolerance ||
       std::abs(h.joints[i].feedback.dq)>c_.settings.velocity_tolerance) {settling_=false;return false;}
    if(!settling_) {settled_=now;settling_=true;}
    return std::chrono::duration<double>(now-settled_).count()>=c_.settings.settle_s;
}
std::string RobotFSM::request(int digit,const real_hardware::HardwareState& h,Clock::time_point now) {
    // 不要静默：Fault 下按 1 以前什么都不打印，看起来像"按键没反应"。
    if(mode_==Mode::Fault)
        return "故障中（" + error_ + "）：1 不会进入 PPO；排除故障后重启程序，或按 0/9 走安全流程";
    if(digit==2) return "AMP未开放，保持当前状态";
    std::string error;
    if(digit==0 && (mode_==Mode::InitialHold || mode_==Mode::Passive || mode_==Mode::PPO)) {
        const bool returning=mode_==Mode::PPO;
        const double duration=returning?c_.settings.return_stand_s:c_.settings.stand_s;
        if(beginMotion(Mode::Standing,"Stand",stand_goal_,duration,h,now,error))
            return returning?"开始回到站立姿态":"开始站起";
        return "轨迹拒绝: "+error;
    }
    if(digit==9 && (mode_==Mode::StandHold || mode_==Mode::PPO)) {
        if(beginMotion(Mode::Crouching,"LieDown",c_.settings.crouch_pose,c_.settings.crouch_s,h,now,error)) return "开始趴下";
        return "轨迹拒绝: "+error;
    }
    if(digit==1 && mode_==Mode::Standing)
        return "尚未就绪：站立轨迹未完成，完成并稳定后请重新输入1+回车";
    if(digit==1 && mode_==Mode::StandHold) {
        if(!c_.settings.mapping_confirmed) return "PPO锁定：关节映射尚未确认（config/deployment.yaml 的 mapping_confirmed）";
        if(!ready(h,now)) {
            // 把"差在哪"打出来：最差关节的位置/速度误差，方便判断是不是没站稳/被吊着。
            double worst_q=0, worst_dq=0; int worst_i=-1;
            for(size_t i=0;i<12;++i) {
                const double dq_err=std::abs(h.joints[i].feedback.q-stand_goal_[i]);
                const double dv=std::abs(h.joints[i].feedback.dq);
                if(dq_err>worst_q) {worst_q=dq_err;worst_i=static_cast<int>(i);}
                worst_dq=std::max(worst_dq,dv);
            }
            char buf[160];
            std::snprintf(buf,sizeof buf,
                "尚未就绪：站立尚未稳定（最差关节 %d 位置误差 %.3f rad，最大速度 %.3f rad/s；容差 %.2f/%.2f）",
                worst_i,worst_q,worst_dq,c_.settings.position_tolerance,c_.settings.velocity_tolerance);
            return buf;
        }
        change(Mode::PPO,"PPO",now);return "进入PPO，等待有效推理结果";
    }
    if(digit==1) return std::string("当前状态 ") + name() + " 不能进 PPO（先按 0 站起，等进入 StandHold 后再按 1）";
    return std::string("当前状态 ") + name() + " 不接受指令 " + std::to_string(digit);
}
void RobotFSM::fault(const std::string& error,Clock::time_point now) {
    if(mode_==Mode::Fault) return;
    error_=error;change(Mode::Fault,"Fault",now);
}
const go::Targets& RobotFSM::update(const real_hardware::HardwareState& h,const PolicyResult& result,Clock::time_point now) {
    if(mode_!=Mode::Fault) {
        if(!h.started || h.faulted || !h.imu_valid || h.joints.size()!=12) {
            std::string why=h.error;
            if(why.empty()) {
                if(!h.started) why="hardware not started";
                else if(h.joints.size()!=12) why="joint count="+std::to_string(h.joints.size());
                else if(!h.imu_valid) why="IMU invalid/stale: age="+std::to_string(h.imu_age_ms)+
                    "ms received="+(h.imu.received?"1":"0")+
                    " fields="+real_hardware::imuFieldNames(h.imu.data.valid_fields)+
                    " error="+(h.imu.error.empty()?"(none)":h.imu.error)+
                    " (need quat+gyro in the same frame)";
                else why="hardware faulted";
            }
            fault(why,now);
        }
        else for(const auto& j:h.joints) if(!j.valid || !std::isfinite(j.feedback.q) || !std::isfinite(j.feedback.dq)) {fault("invalid motor feedback",now);break;}
        if(mode_!=Mode::Fault) {
            try {
                const auto imu_state=adapter_.toPolicy(h);
                if(!imu_state.valid) throw std::invalid_argument("nonfinite IMU feedback");
                my_myth::projectedGravity(imu_state.quaternion);
            }catch(const std::exception& e){fault(e.what(),now);}
        }
    }
    if(mode_==Mode::StandHold) ready(h,now);
    if(mode_==Mode::PPO) {
        if(result.generation==c_.generation && !result.error.empty()) fault("inference: "+result.error,now);
        else if(result.generation==c_.generation && result.sequence>0) {
            const auto age=std::chrono::duration_cast<std::chrono::milliseconds>(now-result.generated).count();
            const auto source_age=std::chrono::duration_cast<std::chrono::milliseconds>(now-result.observed).count();
            if(age<0 || age>=c_.settings.result_timeout_ms || source_age<0 || source_age>=c_.settings.result_timeout_ms) fault("policy result expired",now);
            else {
                try {
                    c_.result=result;c_.result.command=adapter_.toHardware(result.command);
                    if(!c_.has_result) {c_.first_result=now;c_.has_result=true;}
                }catch(const std::exception& e){fault(e.what(),now);}
            }
        } else {
            const auto since=c_.has_result?c_.result.generated:c_.entered;
            if(std::chrono::duration_cast<std::chrono::milliseconds>(now-since).count()>=c_.settings.result_timeout_ms)
                fault("policy result timeout",now);
        }
    }
    if(mode_!=Mode::Fault) {
        c_.measured.clear();c_.measured_velocity.clear();
        for(const auto& j:h.joints) {
            c_.measured.push_back(j.feedback.q);c_.measured_velocity.push_back(j.feedback.dq);
        }
    }
    try {machine_.update(now);}
    catch(const std::exception& e) {fault(e.what(),now);machine_.update(now);}
    if(mode_==Mode::Standing && stand_->finished()) change(Mode::StandHold,"StandHold",now);
    if(mode_==Mode::Crouching && crouch_->unloaded()) {
        // PassiveState keeps running with zero gains; no restart or re-zero of hardware.
        mode_=Mode::Passive;++c_.generation;velocity_={{0,0,0}};settling_=false;
    }
    return c_.command;
}
std::string RobotFSM::gamepad(const imgo2_input::GamepadCommand& command,
                              const real_hardware::HardwareState& h,Clock::time_point now) {
    if(!command.connected) {
        if(mode_==Mode::PPO) velocity_={{0,0,0}}; // 断线/未连接时不允许残留速度
        return {};
    }
    std::string message;
    // 一次只处理一个状态键；A/B/Y 同时按下时按 站起 > 趴下 > PPO 的优先级。
    if(command.stand) message=request(0,h,now);
    else if(command.lie_down) message=request(9,h,now);
    else if(command.enter_ppo) message=request(1,h,now);
    if(mode_==Mode::PPO) {
        const auto& limits=c_.settings.velocity_limits;
        const auto scale=[&](float value,float limit) {
            return std::isfinite(value)?my_myth::clamp(value*limit,-limit,limit):0.f;
        };
        velocity_={{scale(command.vx,limits[0]),scale(command.vy,limits[1]),
                    scale(command.wz,limits[2])}};
    }
    return message;
}

void RobotFSM::velocity(char key) {
    if(mode_!=Mode::PPO) return;
    if(key==' ') {velocity_={{0,0,0}};return;}
    int axis=-1;float step=0;
    switch(key) {
        case 'w':axis=0;step=.1f;break;case 's':axis=0;step=-.1f;break;
        case 'a':axis=1;step=.1f;break;case 'd':axis=1;step=-.1f;break;
        case 'q':axis=2;step=.1f;break;case 'e':axis=2;step=-.1f;break;
        default:return;
    }
    velocity_[axis]=my_myth::clamp(velocity_[axis]+step,-c_.settings.velocity_limits[axis],c_.settings.velocity_limits[axis]);
}
}
