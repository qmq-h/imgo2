#include "ppo_state.h"
#include "function/my_myth.h"
namespace go_fsm::states {
void PPOState::enter(Clock::time_point now) {
    c_.entered=now;c_.first_result={};c_.has_result=false;c_.command=c_.previous;
}
void PPOState::run(Clock::time_point now,Event) {
    if(!c_.has_result) return;
    const float s=float(my_myth::quinticProgress(std::chrono::duration<double>(now-c_.first_result).count(),c_.settings.blend_s));
    c_.command=c_.result.command;
    for(size_t i=0;i<12;++i) {
        c_.command.q[i]=(1-s)*c_.previous.q[i]+s*c_.result.command.q[i];
        c_.command.kp[i]=(1-s)*c_.previous.kp[i]+s*c_.result.command.kp[i];
        c_.command.kd[i]=(1-s)*c_.previous.kd[i]+s*c_.result.command.kd[i];
    }
}
}
