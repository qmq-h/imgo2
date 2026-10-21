#pragma once
#include "deployment_context.h"
#include "states/stand_state.h"
#include "states/passive_state.h"
#include "hardware/robot_adapter.h"
#include "sdk/rl_sdk.h"
namespace go_fsm {
class RobotFSM {
public:
    enum class Mode {InitialHold,Standing,StandHold,PPO,Crouching,Passive,Fault};
    RobotFSM(const imgo2_sdk::PolicyConfig&,const imgo2_sdk::DeploymentConfig&,
             const go::MotorConfig&,const imgo2_sdk::RobotAdapter&,const std::vector<float>& initial,
             Clock::time_point now);
    std::string request(int digit,const real_hardware::HardwareState&,Clock::time_point now);
    const go::Targets& update(const real_hardware::HardwareState&,const PolicyResult&,Clock::time_point now);
    void fault(const std::string&,Clock::time_point now);
    void velocity(char key);
    Mode mode() const {return mode_;}
    const char* name() const {return machine_.name();}
    bool policyActive() const {return mode_==Mode::PPO;}
    uint64_t generation() const {return c_.generation;}
    const imgo2_sdk::Velocity& velocityCommand() const {return velocity_;}
    const std::string& error() const {return error_;}
private:
    void change(Mode,const char* name,Clock::time_point now);
    bool beginMotion(Mode,const char*,const std::vector<float>&,double,const real_hardware::HardwareState&,Clock::time_point,std::string&);
    bool ready(const real_hardware::HardwareState&,Clock::time_point now);
    DeploymentContext c_;
    FSM machine_;
    states::StandState* stand_=nullptr;
    states::PassiveState* crouch_=nullptr;
    go::MotorConfig motors_;
    imgo2_sdk::RobotAdapter adapter_;
    std::vector<float> stand_goal_;
    Mode mode_=Mode::InitialHold;
    imgo2_sdk::Velocity velocity_{{0,0,0}};
    Clock::time_point settled_{};
    bool settling_=false;
    std::string error_;
};
}
