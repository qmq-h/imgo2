#pragma once
#include "../deployment_context.h"
namespace go_fsm::states {
class HoldState final : public State {
public:
    HoldState(DeploymentContext& context,std::string name):c_(context),name_(std::move(name)){}
    const char* name() const override {return name_.c_str();}
    void enter(Clock::time_point now) override;
    void run(Clock::time_point,Event) override {}
    const go::Targets& command() const override {return c_.command;}
private:
    DeploymentContext& c_;
    std::string name_;
};
}
