#pragma once
#include "policy_config.h"
namespace imgo2_sdk {
std::vector<float> buildObservation(const RobotState&,const Velocity&,const std::vector<float>& previous_actions,const PolicyConfig&);
std::vector<float> clipActions(const std::vector<float>&,const PolicyConfig&);
go::Targets convertAction(const std::vector<float>& clipped,const PolicyConfig&);
}
