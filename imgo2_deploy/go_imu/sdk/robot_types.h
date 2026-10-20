#pragma once
#include "go_safety.h"
#include <array>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>
namespace imgo2_sdk {
constexpr size_t joint_count=12;
using Clock=std::chrono::steady_clock;
using Velocity=std::array<float,3>;
struct RobotState {
    std::vector<float> q,dq;
    std::array<float,4> quaternion{{1,0,0,0}}; // wxyz; body-to-world, user's aligned IMU axes.
    std::array<float,3> gyro{{0,0,0}};
    std::array<uint64_t,12> joint_receive_ns{};
    uint64_t imu_receive_ns=0;
    Clock::time_point captured_at{};
    bool valid=false;
};
struct PolicyResult {
    go::Targets command; // Policy-order logical output-side units.
    std::vector<float> actions; // Clipped network action, never offset position.
    Clock::time_point generated{},observed{};
    uint64_t generation=0,sequence=0;
    std::string error;
};
}
