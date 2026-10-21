#include "imu_sensor/imu_monitor.hpp"
#include <chrono>
#include <iomanip>
#include <sstream>
#include <utility>
namespace imu_sensor {
ImuMonitor::ImuMonitor(real_config::ImuConfig config) : config_(std::move(config)) {}
ImuMonitor::~ImuMonitor() { Stop(); }
void ImuMonitor::Start() {
    if (worker_.joinable()) return;
    config_.validate();
    stop_ = false;
    worker_ = std::thread(&ImuMonitor::Run, this);
}
void ImuMonitor::Stop() {
    stop_ = true;
    if (worker_.joinable()) worker_.join();
}
MonitorSnapshot ImuMonitor::Snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return snapshot_;
}
void ImuMonitor::Run() {
    using Clock = std::chrono::steady_clock;
    try {
        ImuSensor sensor;
        sensor.Open(config_.device, config_.baud);
        auto window_start = Clock::now();
        std::uint64_t previous_sequence = 0, latest_sequence = 0;
        while (!stop_) {
            ImuData data;
            const bool received = sensor.Read(data, config_.read_timeout_ms);
            const auto now = Clock::now();
            std::lock_guard<std::mutex> lock(mutex_);
            if (received) {
                snapshot_.data = data;
                snapshot_.received = true;
                latest_sequence = data.sequence;
            }
            const double elapsed = std::chrono::duration<double>(now - window_start).count();
            if (elapsed >= 1.0) {
                snapshot_.receive_hz = double(latest_sequence - previous_sequence) / elapsed;
                previous_sequence = latest_sequence;
                window_start = now;
            }
        }
    } catch (const std::exception& error) {
        std::lock_guard<std::mutex> lock(mutex_);
        snapshot_.error = error.what();
        snapshot_.receive_hz = 0;
    }
}
} // namespace imu_sensor
