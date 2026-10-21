#pragma once
#include "../deployment_context.h"
namespace go_fsm::states {
class FaultState final : public State {
public:
    explicit FaultState(DeploymentContext& c):c_(c){}
    const char* name() const override {return "Fault";}
    const go::Targets& command() const override {return c_.command;}
    void enter(Clock::time_point now) override;
    void run(Clock::time_point,Event) override {}
private:
    DeploymentContext& c_;
};
}
