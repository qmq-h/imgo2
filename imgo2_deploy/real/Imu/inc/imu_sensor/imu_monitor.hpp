#pragma once
#include "imu_sensor/imu_sensor.hpp"
#include "real_config.h"
#include <array>
#include <atomic>
#include <mutex>
#include <thread>
#include <string>
namespace imu_sensor {
struct MonitorSnapshot {
    ImuData data;
    bool received = false;
    std::string error;
    double receive_hz = 0;
    // Startup calibration: how many samples were averaged and whether the
    // display-Euler offset is active. The quaternion is always raw; consumers
    // that need a zeroed yaw must apply their own display-only offset.
    int bias_samples = 0;
    bool bias_ready = false;
};
// The worker owns the serial object. Snapshot consumers never wait for serial I/O.
//
// Startup attitude calibration (2026-10-23, corrected 2026-11-08): the first
// kBiasSamples valid samples are averaged and the *displayed Euler* angles are
// offset to zero. The quaternion is deliberately NOT rotated: the policy uses
// its raw gravity direction, and q * conj(q_bias) is a body-fixed rotation that
// mixes roll/pitch into gravity (yaw -64.6 deg + roll 10 deg would give
// [-0.157,-0.074,-0.985] instead of [0,-0.174,-0.985]). The gyro is NOT rotated
// or biased.
class ImuMonitor {
public:
    static constexpr int kBiasSamples = 50;
    explicit ImuMonitor(real_config::ImuConfig config);
    ~ImuMonitor();
    void Start();
    void Stop();
    MonitorSnapshot Snapshot() const;
private:
    void Run();
    void collectBias(const ImuData& sample);
    void applyBias(ImuData& sample) const;
    real_config::ImuConfig config_;
    mutable std::mutex mutex_;
    MonitorSnapshot snapshot_;
    std::atomic<bool> stop_{false};
    std::thread worker_;
    // Only touched by the worker thread; no lock needed.
    int bias_samples_ = 0;
    int euler_samples_ = 0;
    bool bias_ready_ = false;
    bool euler_bias_ready_ = false;
    std::array<double, 3> bias_euler_sum_{{0.0, 0.0, 0.0}};
    std::array<float, 3> bias_euler_{{0.f, 0.f, 0.f}};
};
} // namespace imu_sensor
