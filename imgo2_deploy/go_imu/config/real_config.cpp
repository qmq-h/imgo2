#include "real_config.h"

namespace go
{

RobotConfig makeDefaultConfig()
{
    RobotConfig config;

    // 四条 CH343 总线，每路 ID 0/1/2。
    // 注意：同一个设备路径不能出现在两条总线里（GoMotorArray::start() 会拒绝）。
    //
    // 设备路径是 /dev/robot/leg_*，由 /etc/udev/rules.d/99-robot-serial.rules
    // 按【盒子序列号 + 接口号】固定下来的软链接，映射不会随拔插/换 USB 口改变：
    //     leg_fl → ttyCH343USB1(接口02)      leg_fr → ttyCH343USB2(接口04)
    //     leg_bl → ttyCH343USB0(接口00)      leg_br → ttyCH343USB3(接口06)
    // IMU 是 /dev/robot/imu → ttyACM0(接口01)，见本文件底部的 makeImuConfig()。
    // 若这些路径打不开，先查 udev 规则有没有生效：
    //     udevadm test /sys/class/tty/ttyCH343USB0
    // 注意路径里 **必须带 robot/ 这一层**，少写就会 No such file or directory。
    //
    // 接线、方向与启动姿态沿用已成功站立的 go 配置。
    // 每次初始化由实测编码器值和 start_pose_rad 计算逻辑零点，
    // 所以启动时机器人必须处于配置所描述的机械姿态。
    // 所有公开角度、速度、力矩、增益均为输出侧单位。
    config.buses.push_back(BusConfig{"/dev/robot/leg_fl", {0, 1, 2},
                                     {+1.0f, +1.0f, +1.0f},
                                     {HIP_START_OFFSET, THIGH_START_OFFSET, SHANK_START_OFFSET}});
    config.buses.push_back(BusConfig{"/dev/robot/leg_fr", {0, 1, 2},
                                     {+1.0f, -1.0f, -1.0f},
                                     {-HIP_START_OFFSET, THIGH_START_OFFSET, SHANK_START_OFFSET}});
    config.buses.push_back(BusConfig{"/dev/robot/leg_bl", {0, 1, 2},
                                     {-1.0f, +1.0f, +1.0f},
                                     {HIP_START_OFFSET, THIGH_START_OFFSET, SHANK_START_OFFSET}});
    config.buses.push_back(BusConfig{"/dev/robot/leg_br", {0, 1, 2},
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
    // /dev/robot/imu 由 udev 按【设备序列号 + 接口号 01】固定，见文件开头注释。
    return {"/dev/robot/imu", 460800, 100, 500};
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
