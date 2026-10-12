#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#define HIP_START_OFFSET 3.49f*(3.14159265358979323846f/180.0f)
#define THIGH_START_OFFSET 67.43f*(3.14159265358979323846f/180.0f)
#define SHANK_START_OFFSET -165.01f*(3.14159265358979323846f/180.0f)

namespace go
{

/**
 * 一条物理总线的配置。
 * motor_ids 为该总线上注册的电机 ID；motor_directions 与 start_pose_rad 都与它等长。
 * motor_directions 留空 = 全部 +1；start_pose_rad 留空 = 这条总线不做启动自动归零。
 */
struct BusConfig
{
    std::string device;
    std::vector<int> motor_ids;
    /**
     * 每台电机的方向系数，只填 +1 或 -1（留空 = 全部 +1）。
     * -1 表示这台电机的逻辑坐标整体取反：位置/速度/前馈力矩在读和写上一起反，
     * 而 offset 不反。镜像安装的左右腿用 -1，就能共用同一份姿态与轨迹。
     */
    std::vector<float> motor_directions;
    /**
     * 每台电机在**启动姿态**下的角度（**输出侧 rad，带符号**），与 motor_ids 等长。
     * 例：启动姿态是"趴着"，而仿真里那个姿态是大腿 +67.43°、小腿 −165.01°，
     * 就填这些值的 rad 形式（可以直接写你 config 里的 HIP/THIGH/SHANK_START_OFFSET 宏）。
     *
     * 启动时用它自动设零点（等价于手工跑 go_probe 贴 offset，改成运行时做）：
     *     offset = 本次实测角度 − direction × start_pose_rad
     * 留空 = 这条总线不做自动归零。
     */
    std::vector<float> start_pose_rad;
};

/**
 * 单个关节的限幅。默认全部是“不限”，所以不填就等于没有这一层。
 * 单位都是输出侧：q rad、dq rad/s、tau N·m、kp N·m/rad、kd N·m/(rad/s)。
 * dq_max 同时用作 Safety::approach() 的斜坡速度上限。
 */
struct JointLimits
{
    float q_min = -std::numeric_limits<float>::infinity();
    float q_max = std::numeric_limits<float>::infinity();
    float dq_max = std::numeric_limits<float>::infinity();
    float tau_max = std::numeric_limits<float>::infinity();
    float kp_max = std::numeric_limits<float>::infinity();
    float kd_max = std::numeric_limits<float>::infinity();
};

/**
 * 整个机器人（本工程里就是 12 台电机）的配置。
 * limits 可以留空（= 全部不限），也可以与展开后的关节一一对应。
 */
struct RobotConfig
{
    std::vector<BusConfig> buses;
    std::vector<JointLimits> limits;
    uint32_t baud = 4000000;
    size_t io_timeout_us = 20000;
    size_t command_timeout_ms = 100;
    std::chrono::microseconds period = std::chrono::milliseconds(5);
    std::chrono::milliseconds watchdog = std::chrono::milliseconds(80);
};

/**
 * 本工程默认接线：4 条 CH343 总线 × 每路 3 台（ID 0/1/2），按启动姿态自动归零；限幅仍为空。
 * demo 与后续的 deploy 共用这一份，避免两边各写一遍接线；改接线只需要改这里。
 */
RobotConfig makeDefaultConfig();

} // namespace go

namespace real_config {
struct ImuConfig {
    std::string device;
    unsigned int baud;
    int read_timeout_ms;
    int stale_after_ms;
};
ImuConfig makeImuConfig();
} // namespace real_config

namespace real_config {
struct HardwareConfig {
    go::RobotConfig motors;
    ImuConfig imu;
    int imu_ready_timeout_ms=2000;
    int motor_stale_after_ms=100;
    bool joint_mapping_confirmed=false;
    std::vector<std::string> joint_names;
    std::vector<size_t> policy_to_hardware;
    bool imu_mount_confirmed=false;
    std::vector<float> imu_mount_quaternion={1,0,0,0}; // wxyz, reserved metadata; current API always outputs sensor-frame IMU.
};
HardwareConfig makeHardwareConfig();
}
