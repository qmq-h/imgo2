#pragma once
#include "../deployment_context.h"
namespace go_fsm::states {
// Crouch first, verify measured arrival, then release PD and feedforward effort.
class PassiveState final : public State {
public:
    explicit PassiveState(DeploymentContext& context);
    const char* name() const override {return unloaded_?"Passive":"LieDown";}
    void enter(Clock::time_point now) override;
    void run(Clock::time_point now,Event event) override;
    const go::Targets& command() const override {return c_.command;}
    bool unloaded() const {return unloaded_;}
private:
    DeploymentContext& c_;
    std::vector<float> start_;
    bool unloaded_=false,settling_=false;
    Clock::time_point settled_{};
};
}
