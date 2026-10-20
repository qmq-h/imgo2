#pragma once
#include "real_hardware.h"
#include "sdk/robot_types.h"
namespace imgo2_sdk {
class RobotAdapter {
public:
    explicit RobotAdapter(std::vector<int> policy_to_hardware);
    RobotState toPolicy(const real_hardware::HardwareState&) const;
    go::Targets toHardware(const go::Targets&) const;
    std::vector<float> positionsToHardware(const std::vector<float>&) const;
private:
    std::vector<int> map_;
};
}
