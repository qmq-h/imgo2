#pragma once
#include "real_config.h"
#include "sdk/robot_types.h"
#include <filesystem>
namespace imgo2_sdk {
struct DeploymentConfig {
    DeploymentConfig();
    int control_ms=5,policy_ms=20,result_timeout_ms=100,torch_threads=1;
    double stand_s=8,return_stand_s=2,crouch_s=8,blend_s=.5,settle_s=.5;
    float position_tolerance=.1f,velocity_tolerance=.5f;
    Velocity velocity_limits{{.5f,.2f,.5f}};
    // Logical output-side radians, hardware order FL/FR/BL/BR, hip/thigh/shank.
    std::vector<float> stand_pose, crouch_pose;
    double crouch_settle_s=.2,crouch_arrival_timeout_s=2;
    float crouch_position_tolerance=.05f,crouch_velocity_tolerance=.5f;
    std::vector<int> policy_to_hardware;
    bool mapping_confirmed=false;
    void validate() const;
    static DeploymentConfig load(const std::filesystem::path& path);
};
}
