#pragma once
#include "fsm.h"
#include "config/deployment_config.h"
#include "sdk/robot_types.h"
namespace go_fsm {
using imgo2_sdk::PolicyResult;
struct DeploymentContext {
    imgo2_sdk::DeploymentConfig settings;
    go::Targets command,fixed,goal,previous;
    std::vector<float> measured, measured_velocity;
    PolicyResult result;
    uint64_t generation=0;
    double stand_duration_s=8; // Selected and safety-checked before entering Stand.
    Clock::time_point entered{},first_result{};
    bool has_result=false;
};
}
