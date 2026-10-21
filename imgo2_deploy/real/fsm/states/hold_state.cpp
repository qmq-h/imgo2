#include "hold_state.h"
namespace go_fsm::states {
void HoldState::enter(Clock::time_point now) {
    c_.entered=now;c_.command=c_.fixed;c_.command.q=c_.goal.q;
}
}
