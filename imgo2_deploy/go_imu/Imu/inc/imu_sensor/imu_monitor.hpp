#pragma once
#include "imu_sensor/imu_sensor.hpp"
#include "real_config.h"
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
};
// The worker owns the serial object. Snapshot consumers never wait for serial I/O.
class ImuMonitor {
public:
    explicit ImuMonitor(real_config::ImuConfig config);
    ~ImuMonitor();
    void Start();
    void Stop();
    MonitorSnapshot Snapshot() const;
private:
    void Run();
    real_config::ImuConfig config_;
    mutable std::mutex mutex_;
    MonitorSnapshot snapshot_;
    std::atomic<bool> stop_{false};
    std::thread worker_;
};
} // namespace imu_sensor
