#pragma once
#include "../fsm.h"
namespace go_fsm::states {
// Reserved abstract interface. Not instantiated, registered or called by RobotFSM.
// Implement enter/run/command and a verified AMP contract before enabling it.
class AmpState : public State {
public:
    const char* name() const override {return "AMP";}
};
}
