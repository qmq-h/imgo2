#pragma once
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include "../function/my_myth.h"

// 用户参数入口：启动姿态、关节范围与电机上限，均为输出侧单位。
#define HIP_START_OFFSET (3.49f * my_myth::degrees_to_radians_f)
#define THIGH_START_OFFSET (67.43f * my_myth::degrees_to_radians_f)
#define SHANK_START_OFFSET (-165.01f * my_myth::degrees_to_radians_f)

#define MOTOR_LIMIT_TAU_MAX 23.7f
#define MOTOR_LIMIT_KP_MAX 70.0f
#define MOTOR_LIMIT_KD_MAX 10.0f

#define HIP_LIMIT_POS_MIN -0.523f
#define HIP_LIMIT_POS_MAX 0.523f
#define HIP_LIMIT_VEL_MAX 30.1f

#define THIGH_LIMIT_POS_MIN -0.9f
#define THIGH_LIMIT_POS_MAX 2.87f
#define THIGH_LIMIT_VEL_MAX 30.1f

#define SHANK_LIMIT_POS_MIN -3.0f
#define SHANK_LIMIT_POS_MAX -0.733f
#define SHANK_LIMIT_VEL_MAX 30.1f

namespace go
{
    // 顺序为总线→motor_ids；方向作用于位置/速度/力矩，启动角用于归零。
    struct BusConfig
    {
        std::string device;
        std::vector<int> motor_ids;
        std::vector<float> motor_directions;
        std::vector<float> start_pose_rad;
    };
    // 输出侧逻辑单位：rad, rad/s, N·m, N·m/rad, N·m/(rad/s)。
    // tau_max仅限制前馈；不代表电机内部PD总力矩上限。
    struct JointLimits
    {
        float q_min = -std::numeric_limits<float>::infinity();
        float q_max = std::numeric_limits<float>::infinity();
        float dq_max = std::numeric_limits<float>::infinity();
        float tau_max = std::numeric_limits<float>::infinity();
        float kp_max = std::numeric_limits<float>::infinity();
        float kd_max = std::numeric_limits<float>::infinity();
    };
    class MotorConfig
    {
    public:
        MotorConfig(); // 只填默认数据，不操作硬件。
        // 接线、方向及启动姿态
        std::vector<BusConfig> buses;
        // 每关节限幅，与总线展开顺序一致。
        std::vector<JointLimits> limits;
        // 通信与安全节拍
        uint32_t baud = 4000000;
        size_t io_timeout_us = 20000;
        size_t command_timeout_ms = 100;
        std::chrono::microseconds period = std::chrono::milliseconds(5);
        std::chrono::milliseconds watchdog = std::chrono::milliseconds(80);
        // 一轮的收发方式：true = 先连发本总线所有命令、再按 ID 收全部应答（流水线）；
        // false = 逐台一问一答。间隔必须大于"命令43us+换向10us+应答40us"，150us 留了余量。
        bool pipeline = true;
        std::uint32_t pipeline_gap_us = 150;
    };
    using RobotConfig = MotorConfig; // 保留已有硬件调用方类型名。
    MotorConfig makeDefaultConfig();
}
namespace real_config
{
    class ImuConfig
    {
    public:
        ImuConfig();
        ImuConfig(std::string device, unsigned int baud, int read_timeout_ms, int stale_after_ms);
        void validate() const; // 不打开设备，非法参数抛invalid_argument。
        std::string device;
        unsigned int baud = 460800;
        int read_timeout_ms = 100;
        int stale_after_ms = 500;
        bool mount_confirmed = false;
        std::vector<float> mount_quaternion = {1, 0, 0, 0}; // wxyz元数据；本轮不应用旋转。
    };
    ImuConfig makeImuConfig();
    // 组合两种配置；电机和IMU参数不在这里重复定义。
    struct HardwareConfig
    {
        go::MotorConfig motors;
        ImuConfig imu;
        int imu_ready_timeout_ms = 2000;
        int motor_stale_after_ms = 100;
        bool joint_mapping_confirmed = false;
        std::vector<std::string> joint_names;
    };
    HardwareConfig makeHardwareConfig();
}
