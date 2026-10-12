#include "real_config.h"

namespace go
{

RobotConfig makeDefaultConfig()
{
    RobotConfig config;

    // 四条 CH343 总线，每路 ID 0/1/2。
    // 注意：同一个设备路径不能出现在两条总线里（GoMotorArray::start() 会拒绝）。
    //
    // 接线、方向与启动姿态沿用已成功站立的 go 配置。
    // 每次初始化由实测编码器值和 start_pose_rad 计算逻辑零点，
    // 所以启动时机器人必须处于配置所描述的机械姿态。
    // 所有公开角度、速度、力矩、增益均为输出侧单位。
    config.buses.push_back(BusConfig{"/dev/ttyCH343USB1", {0, 1, 2},
                                     {+1.0f, +1.0f, +1.0f},
                                     {HIP_START_OFFSET, THIGH_START_OFFSET, SHANK_START_OFFSET}});
    config.buses.push_back(BusConfig{"/dev/ttyCH343USB2", {0, 1, 2},
                                     {+1.0f, -1.0f, -1.0f},
                                     {-HIP_START_OFFSET, THIGH_START_OFFSET, SHANK_START_OFFSET}});
    config.buses.push_back(BusConfig{"/dev/ttyCH343USB0", {0, 1, 2},
                                     {-1.0f, +1.0f, +1.0f},
                                     {HIP_START_OFFSET, THIGH_START_OFFSET, SHANK_START_OFFSET}});
    config.buses.push_back(BusConfig{"/dev/ttyCH343USB3", {0, 1, 2},
                                     {-1.0f, -1.0f, -1.0f},
                                     {-HIP_START_OFFSET, THIGH_START_OFFSET, SHANK_START_OFFSET}});
    // TODO(限幅)：标定完 offset 之后按关节填限幅。留空 = 全部不限（等于没有这一层）。
    //   下面这几项**不受 offset 影响，现在就能填**：
    //       dq_max  最大关节速度（rad/s，也用作 Safety::approach() 的斜坡速度上限）
    //       tau_max 前馈力矩上限（N·m）
    //       kp_max / kd_max 增益上限
    //   这两项要等 offset 标定好再填（启动姿态未确认时 q 没有可靠的物理含义）：
    //       q_min / q_max 位置上下限（rad）
    //   填法见 config/real_config.h 的 JointLimits，或 README 第 7.2.2 节。
    config.limits.clear();

    config.baud = 4000000;
    config.io_timeout_us = 20000;
    config.command_timeout_ms = 100;
    config.period = std::chrono::milliseconds(5);       // 每条总线 200 Hz
    config.watchdog = std::chrono::milliseconds(80);    // 必须短于 command_timeout_ms
    return config;
}

} // namespace go

namespace real_config {
ImuConfig makeImuConfig() {
    return {"/dev/ttyACM0", 460800, 100, 500};
}
} // namespace real_config

namespace real_config {
HardwareConfig makeHardwareConfig() {
    HardwareConfig config;
    config.motors=go::makeDefaultConfig();
    config.imu=makeImuConfig();
    return config;
}
}
