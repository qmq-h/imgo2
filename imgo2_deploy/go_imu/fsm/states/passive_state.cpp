#include "passive_state.h"
#include <cmath>
#include "function/my_myth.h"
namespace go_fsm::states {
PassiveState::PassiveState(DeploymentContext& context):c_(context) {}
void PassiveState::enter(Clock::time_point now) {
    unloaded_=false;settling_=false;
    c_.goal=c_.fixed;c_.goal.q=c_.settings.crouch_pose;
    c_.entered=now;start_=c_.measured;
    run(now,Event::None);
}
void PassiveState::run(Clock::time_point now,Event) {
    if(unloaded_) return;
    const double elapsed=std::chrono::duration<double>(now-c_.entered).count();
    const double progress=my_myth::quinticProgress(elapsed,c_.settings.crouch_s);
    const double rate=my_myth::quinticRate(elapsed,c_.settings.crouch_s);
    const float blend=float(my_myth::quinticProgress(elapsed,c_.settings.blend_s));
    c_.command=c_.fixed;
    for(size_t i=0;i<start_.size();++i) {
        const double delta=double(c_.goal.q[i])-start_[i];
        c_.command.q[i]=float(start_[i]+progress*delta);
        c_.command.dq[i]=float(rate*delta);
        c_.command.kp[i]=(1-blend)*c_.previous.kp[i]+blend*c_.fixed.kp[i];
        c_.command.kd[i]=(1-blend)*c_.previous.kd[i]+blend*c_.fixed.kd[i];
    }
    if(elapsed<c_.settings.crouch_s) return;
    bool arrived=c_.measured.size()==12 && c_.measured_velocity.size()==12;
    if(arrived) for(size_t i=0;i<12;++i) {
        if(!std::isfinite(c_.measured[i]) || !std::isfinite(c_.measured_velocity[i]) ||
           std::abs(c_.measured[i]-c_.goal.q[i])>c_.settings.crouch_position_tolerance ||
           std::abs(c_.measured_velocity[i])>c_.settings.crouch_velocity_tolerance) {arrived=false;break;}
    }
    if(!arrived) settling_=false;
    else {
        if(!settling_) {settled_=now;settling_=true;}
        if(std::chrono::duration<double>(now-settled_).count()>=c_.settings.crouch_settle_s) {
            // Keep a legal q reference, but remove every force-producing command.
            for(auto* values:{&c_.command.kp,&c_.command.kd,&c_.command.tau,&c_.command.dq}) values->assign(12,0);
            unloaded_=true;return;
        }
    }
    if(elapsed>=c_.settings.crouch_s+c_.settings.crouch_arrival_timeout_s)
        throw std::runtime_error("crouch arrival timeout: measured pose/velocity did not settle");
}
}
