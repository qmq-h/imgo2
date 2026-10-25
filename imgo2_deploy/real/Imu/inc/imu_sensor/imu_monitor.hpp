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
    // Startup attitude calibration: how many samples were averaged and whether
    // the correction is active. Consumers that need "raw" data must read this.
    int bias_samples = 0;
    bool bias_ready = false;
};
// The worker owns the serial object. Snapshot consumers never wait for serial I/O.
//
// Startup attitude bias (2026-10-23): the first kBiasSamples valid samples are
// averaged and treated as the zero attitude, so RPY is ~0 right after power-on.
// The quaternion is corrected as q_corrected = q_measured * conj(q_bias) (body-fixed,
// i.e. the rotation since startup); the gyro is NOT rotated or biased, because the
// observed yaw offset is a sensor reference offset, not a physical mount rotation.
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
    std::array<float, 4> bias_reference_{{1.f, 0.f, 0.f, 0.f}};
    std::array<double, 4> bias_quaternion_sum_{{0.0, 0.0, 0.0, 0.0}};
    std::array<double, 3> bias_euler_sum_{{0.0, 0.0, 0.0}};
    std::array<float, 4> bias_quaternion_{{1.f, 0.f, 0.f, 0.f}};
    std::array<float, 3> bias_euler_{{0.f, 0.f, 0.f}};
};
} // namespace imu_sensor
