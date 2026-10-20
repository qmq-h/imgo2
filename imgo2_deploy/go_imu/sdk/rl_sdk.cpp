#include "rl_sdk.h"
#include "function/my_myth.h"
#include <cmath>
#include <stdexcept>
namespace imgo2_sdk {
std::vector<float> buildObservation(const RobotState& s,const Velocity& velocity,const std::vector<float>& last,const PolicyConfig& c) {
    if(!s.valid || s.q.size()!=12 || s.dq.size()!=12 || last.size()!=12) throw std::invalid_argument("invalid policy state/action shape");
    std::vector<float> out;out.reserve(45);
    for(float x:s.gyro) out.push_back(x*c.ang_vel_scale);
    const auto gravity=my_myth::projectedGravity(s.quaternion);
    out.insert(out.end(),gravity.begin(),gravity.end());
    for(size_t i=0;i<3;++i) out.push_back(velocity[i]*c.commands_scale[i]);
    for(size_t i=0;i<12;++i) out.push_back((s.q[i]-c.default_pose[i])*c.dof_pos_scale);
    for(float x:s.dq) out.push_back(x*c.dof_vel_scale);
    out.insert(out.end(),last.begin(),last.end());
    for(float& x:out) {
        if(!std::isfinite(x)) throw std::invalid_argument("nonfinite observation");
        x=my_myth::clamp(x,-c.clip_obs,c.clip_obs);
    }
    return out;
}
std::vector<float> clipActions(const std::vector<float>& raw,const PolicyConfig& c) {
    if(raw.size()!=12) throw std::invalid_argument("policy output requires 12 actions");
    auto clipped=raw;
    for(size_t i=0;i<12;++i) {
        if(!std::isfinite(raw[i])) throw std::invalid_argument("nonfinite policy output");
        clipped[i]=my_myth::clamp(raw[i],c.lower[i],c.upper[i]);
    }
    return clipped;
}
go::Targets convertAction(const std::vector<float>& action,const PolicyConfig& c) {
    if(action.size()!=12) throw std::invalid_argument("invalid action shape");
    go::Targets command;command.kp=c.kp;command.kd=c.kd;command.dq.assign(12,0);command.tau.assign(12,0);
    command.q.resize(12);
    for(size_t i=0;i<12;++i) {
        if(!std::isfinite(action[i])) throw std::invalid_argument("nonfinite action");
        command.q[i]=c.default_pose[i]+c.action_scale[i]*action[i];
        if(!std::isfinite(command.q[i])) throw std::invalid_argument("nonfinite target");
    }
    return command;
}
}
