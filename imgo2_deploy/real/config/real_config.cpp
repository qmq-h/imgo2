#include "real_config.h"
#include "function/my_myth.h"
#include <cmath>
#include <iterator>
#include <stdexcept>
#include <utility>
namespace go
{
    MotorConfig::MotorConfig()
    {
        // 输出侧启动姿态。必须在这一机械姿态初始化，不在站起后再次归零。
        const float hip = HIP_START_OFFSET;
        const float thigh = THIGH_START_OFFSET;
        const float shank = SHANK_START_OFFSET;
        // 已由用户确认的udev设备名；四总线各ID0/1/2。
        buses = {
            BusConfig{"/dev/robot/leg_fl", {0, 1, 2}, {+1, +1, +1}, {hip, thigh, shank}},
            BusConfig{"/dev/robot/leg_fr", {0, 1, 2}, {+1, -1, -1}, {-hip, thigh, shank}},
            BusConfig{"/dev/robot/leg_bl", {0, 1, 2}, {-1, +1, +1}, {hip, thigh, shank}},
            BusConfig{"/dev/robot/leg_br", {0, 1, 2}, {-1, -1, -1}, {-hip, thigh, shank}},
        };
        // q_min, q_max, dq_max, tau_max, kp_max, kd_max；不加offset或方向。
        const JointLimits leg[] = {
            {HIP_LIMIT_POS_MIN, HIP_LIMIT_POS_MAX, HIP_LIMIT_VEL_MAX, MOTOR_LIMIT_TAU_MAX, MOTOR_LIMIT_KP_MAX, MOTOR_LIMIT_KD_MAX}, // 髋
            {THIGH_LIMIT_POS_MIN, THIGH_LIMIT_POS_MAX, THIGH_LIMIT_VEL_MAX, MOTOR_LIMIT_TAU_MAX, MOTOR_LIMIT_KP_MAX, MOTOR_LIMIT_KD_MAX},    // 大腿
            {SHANK_LIMIT_POS_MIN, SHANK_LIMIT_POS_MAX, SHANK_LIMIT_VEL_MAX, MOTOR_LIMIT_TAU_MAX, MOTOR_LIMIT_KP_MAX, MOTOR_LIMIT_KD_MAX},  // 小腿
        };
        limits.reserve(buses.size() * 3);
        for (size_t i = 0; i < buses.size(); ++i)
            limits.insert(limits.end(), std::begin(leg), std::end(leg));
    }
    MotorConfig makeDefaultConfig() { return MotorConfig{}; }
}
namespace real_config
{
    ImuConfig::ImuConfig() : device("/dev/robot/imu") {}
    ImuConfig::ImuConfig(std::string path, unsigned int speed, int read_ms, int stale_ms)
        : device(std::move(path)), baud(speed), read_timeout_ms(read_ms), stale_after_ms(stale_ms) {}
    void ImuConfig::validate() const
    {
        if (device.empty() || !baud || read_timeout_ms <= 0 || stale_after_ms <= 0)
            throw std::invalid_argument("invalid IMU device/baud/timeouts");
        if (mount_quaternion.size() != 4)
            throw std::invalid_argument("IMU mount quaternion requires wxyz");
        double norm = 0;
        for (float v : mount_quaternion)
        {
            if (!std::isfinite(v))
                throw std::invalid_argument("nonfinite IMU mount quaternion");
            norm += double(v) * v;
        }
        if (norm < 1e-12)
            throw std::invalid_argument("zero IMU mount quaternion");
    }
    ImuConfig makeImuConfig() { return ImuConfig{}; }
    HardwareConfig makeHardwareConfig() { return HardwareConfig{}; }
}
