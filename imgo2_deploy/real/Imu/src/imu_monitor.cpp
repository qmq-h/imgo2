#include "imu_sensor/imu_monitor.hpp"
#include <chrono>
#include <cmath>
#include <utility>
namespace imu_sensor {
namespace {
constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 2.0f * kPi;

// Keeps an angle in (-pi, pi]; used for the yaw output only.
float wrapPi(float angle) {
    while (angle > kPi) angle -= kTwoPi;
    while (angle <= -kPi) angle += kTwoPi;
    return angle;
}

} // namespace

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

// Zeroes the displayed Euler angles from the first kBiasSamples valid samples.
// The quaternion is intentionally NOT corrected: the policy consumes its raw
// gravity direction, and q * conj(q_bias) is a body-fixed rotation (it mixes
// roll/pitch into gravity), not a heading change. Gyro is also left raw.
void ImuMonitor::collectBias(const ImuData& sample) {
    if (!sample.Has(Quaternion) || sample.quaternion.size() != 4) return;
    if (sample.Has(EulerAngles) && sample.euler.size() == 3) {
        for (int i = 0; i < 3; ++i) bias_euler_sum_[i] += sample.euler[i];
        ++euler_samples_;
    }
    ++bias_samples_;
    if (bias_samples_ < kBiasSamples) return;
    bias_ready_ = true;
    if (euler_samples_ > 0) {
        for (int i = 0; i < 3; ++i)
            bias_euler_[i] = static_cast<float>(bias_euler_sum_[i] / euler_samples_);
        euler_bias_ready_ = true;
    }
}

void ImuMonitor::applyBias(ImuData& sample) const {
    // Display-only zeroing. Do NOT rotate sample.quaternion here: keeping it raw
    // preserves the body-frame gravity direction used by the policy, and keeps
    // it consistent with the raw gyro.
    if (euler_bias_ready_ && sample.Has(EulerAngles) && sample.euler.size() == 3) {
        for (int i = 0; i < 3; ++i) sample.euler[i] = wrapPi(sample.euler[i] - bias_euler_[i]);
    }
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
            if (received) {
                // Warm-up publishes raw samples so start-up/readiness checks are not
                // delayed; the attitude turns to ~0 as soon as kBiasSamples arrive.
                if (!bias_ready_) collectBias(data);
                applyBias(data);
            }
            std::lock_guard<std::mutex> lock(mutex_);
            if (received) {
                snapshot_.data = data;
                snapshot_.received = true;
                latest_sequence = data.sequence;
            }
            snapshot_.bias_samples = bias_samples_;
            snapshot_.bias_ready = bias_ready_;
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
