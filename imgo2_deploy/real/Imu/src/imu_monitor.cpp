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

// Hamilton product of two wxyz quaternions: a ⊗ b.
std::array<float, 4> quatMultiply(const std::array<float, 4>& a,
                                  const std::array<float, 4>& b) {
    return {{
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    }};
}

bool normalizeQuaternion(std::array<float, 4>& q) {
    const float norm2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if (!(norm2 > 1e-12f) || !std::isfinite(norm2)) return false;
    const float inv = 1.0f / std::sqrt(norm2);
    for (float& value : q) value *= inv;
    return true;
}

std::array<float, 4> quaternionOf(const ImuData& sample) {
    return {{sample.quaternion[0], sample.quaternion[1],
             sample.quaternion[2], sample.quaternion[3]}};
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

// Averages the first kBiasSamples valid attitudes. Sign-aligns each quaternion to
// the first one so the double cover (+q/-q) cannot cancel the mean, and keeps a
// separate Euler sum for the display path (the panel prints the sensor's Euler
// field, not an Euler recomputed from the quaternion, so the two fields need
// independent offsets to stay consistent with whatever the sensor reports).
void ImuMonitor::collectBias(const ImuData& sample) {
    if (!sample.Has(Quaternion) || sample.quaternion.size() != 4) return;
    std::array<float, 4> q = quaternionOf(sample);
    if (bias_samples_ == 0) bias_reference_ = q;
    const float dot = q[0] * bias_reference_[0] + q[1] * bias_reference_[1] +
                      q[2] * bias_reference_[2] + q[3] * bias_reference_[3];
    if (dot < 0.f)
        for (float& value : q) value = -value;
    for (int i = 0; i < 4; ++i) bias_quaternion_sum_[i] += q[i];
    if (sample.Has(EulerAngles) && sample.euler.size() == 3) {
        for (int i = 0; i < 3; ++i) bias_euler_sum_[i] += sample.euler[i];
        ++euler_samples_;
    }
    ++bias_samples_;
    if (bias_samples_ < kBiasSamples) return;

    std::array<float, 4> mean{{
        static_cast<float>(bias_quaternion_sum_[0] / bias_samples_),
        static_cast<float>(bias_quaternion_sum_[1] / bias_samples_),
        static_cast<float>(bias_quaternion_sum_[2] / bias_samples_),
        static_cast<float>(bias_quaternion_sum_[3] / bias_samples_),
    }};
    if (!normalizeQuaternion(mean)) {
        // Degenerate mean (should not happen): throw the window away and retry.
        bias_samples_ = 0;
        euler_samples_ = 0;
        bias_quaternion_sum_.fill(0.0);
        bias_euler_sum_.fill(0.0);
        return;
    }
    bias_quaternion_ = mean;
    bias_ready_ = true;
    if (euler_samples_ > 0) {
        for (int i = 0; i < 3; ++i)
            bias_euler_[i] = static_cast<float>(bias_euler_sum_[i] / euler_samples_);
        euler_bias_ready_ = true;
    }
}

void ImuMonitor::applyBias(ImuData& sample) const {
    if (bias_ready_ && sample.Has(Quaternion) && sample.quaternion.size() == 4) {
        const std::array<float, 4> q = quaternionOf(sample);
        const std::array<float, 4> inv{{
            bias_quaternion_[0], -bias_quaternion_[1],
            -bias_quaternion_[2], -bias_quaternion_[3]}};
        std::array<float, 4> corrected = quatMultiply(q, inv);
        if (normalizeQuaternion(corrected)) {
            for (int i = 0; i < 4; ++i) sample.quaternion[i] = corrected[i];
        }
    }
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
