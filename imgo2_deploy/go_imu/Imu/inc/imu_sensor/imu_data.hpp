#ifndef IMU_SENSOR_IMU_DATA_HPP
#define IMU_SENSOR_IMU_DATA_HPP

#include <cstdint>
#include <vector>

namespace imu_sensor {

enum DataField : std::uint32_t {
    Acceleration = 1U << 0,
    AngularVelocity = 1U << 1,
    EulerAngles = 1U << 2,
    Quaternion = 1U << 3
};

// Same field order and container types as RobotState<T>::IMU in rl_sdk.hpp.
// Values are in sensor axes; mounting rotation is the application's responsibility.
struct ImuData {
    std::vector<float> quaternion = {1.f, 0.f, 0.f, 0.f}; // w,x,y,z, unit quaternion
    std::vector<float> gyroscope = {0.f, 0.f, 0.f};       // x,y,z, rad/s
    std::vector<float> accelerometer = {0.f, 0.f, 0.f};   // x,y,z, m/s^2
    std::vector<float> euler = {0.f, 0.f, 0.f};           // roll,pitch,yaw, rad
    std::uint32_t valid_fields = 0; // Only fields present in THIS frame.
    std::uint16_t tid = 0;
    std::uint64_t sequence = 0;
    std::uint64_t receive_time_ns = 0; // Host steady clock, not sensor measurement time.

    bool Has(std::uint32_t fields) const {
        return (valid_fields & fields) == fields;
    }

    // No dependency on rl_sdk.hpp. Returns false without changing target if
    // quaternion + gyro + acceleration were not received in the same frame.
    template <typename TargetImu>
    bool CopyTo(TargetImu& target) const {
        if (!Has(Quaternion | AngularVelocity | Acceleration)) return false;
        target.quaternion.assign(quaternion.begin(), quaternion.end());
        target.gyroscope.assign(gyroscope.begin(), gyroscope.end());
        target.accelerometer.assign(accelerometer.begin(), accelerometer.end());
        return true;
    }
};

} // namespace imu_sensor
#endif
