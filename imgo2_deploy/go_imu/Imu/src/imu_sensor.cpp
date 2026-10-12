#include "imu_sensor/imu_sensor.hpp"

#include <chrono>
#include <stdexcept>

namespace imu_sensor {

void ImuSensor::Open(const std::string& device, unsigned int baud) {
    parser_.Reset();
    serial_.Open(device, baud);
}

void ImuSensor::Close() noexcept {
    serial_.Close();
    parser_.Reset();
}

bool ImuSensor::Read(ImuData& output, int timeout_ms) {
    if (timeout_ms < 0) throw std::invalid_argument("timeout must be nonnegative");
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(timeout_ms);
    int remaining = timeout_ms;
    for (;;) {
        std::uint8_t bytes[1024];
        const std::size_t count = serial_.Read(bytes, sizeof(bytes), remaining);
        if (count == 0) return false;
        std::vector<ImuData> samples;
        parser_.Feed(bytes, count, samples);
        if (!samples.empty()) { output = samples.back(); return true; }
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) return false;
        remaining = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - now).count()) + 1;
    }
}

} // namespace imu_sensor
