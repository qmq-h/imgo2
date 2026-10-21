#include "fault_state.h"
#include "function/my_myth.h"
namespace go_fsm::states {
void FaultState::enter(Clock::time_point now) {
    c_.entered=now;c_.command=c_.fixed;
    for(auto* v:{&c_.command.q,&c_.command.dq,&c_.command.kp,&c_.command.kd,&c_.command.tau}) v->assign(12,0);
}
}
