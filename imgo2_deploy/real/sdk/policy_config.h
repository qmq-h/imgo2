#pragma once
#include "robot_types.h"
#include <filesystem>
namespace imgo2_sdk {
struct PolicyConfig {
    std::filesystem::path model_path;
    std::vector<float> default_pose,action_scale,lower,upper,kp,kd,fixed_kp,fixed_kd,torque_limits;
    Velocity commands_scale{{1,1,1}};
    float ang_vel_scale=.25f,dof_pos_scale=1,dof_vel_scale=.05f,clip_obs=100;
    static PolicyConfig load(const std::filesystem::path& directory);
};
}
