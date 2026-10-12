#pragma once
#include "go_motor_array.h"
#include "imu_sensor/imu_monitor.hpp"
#include "real_config.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
namespace real_hardware {
struct JointState {
    go::Feedback feedback; // Logical output-side units, hardware expansion order.
    uint64_t receive_time_ns=0;
    long long age_ms=-1;
    bool valid=false;
    std::string error;
};
struct HardwareState {
    std::vector<JointState> joints;
    imu_sensor::MonitorSnapshot imu;
    long long imu_age_ms=-1;
    bool imu_valid=false, started=false, faulted=false;
    bool joint_mapping_confirmed=false, imu_mount_confirmed=false;
    bool policy_ordered=false, imu_in_body_frame=false; // Reserved transforms are not applied.
    std::string error;
};
// start/stop belong to one upper thread and must not race other public calls.
// Once started, readState/sendCommand are mutex-protected; the supervisor uses safe snapshots.
// stop/start reinitializes encoder zero against the configured mechanical START POSE.
class RealHardware {
public:
    explicit RealHardware(real_config::HardwareConfig config=real_config::makeHardwareConfig());
    ~RealHardware();
    bool start(std::string& error);
    HardwareState readState() const;
    bool sendCommand(const go::Targets& command);
    void stop();
private:
    HardwareState snapshotLocked() const;
    bool checkLocked();
    void latchLocked(const std::string& reason);
    void supervise();
    real_config::HardwareConfig config_;
    mutable std::mutex mutex_;
    std::unique_ptr<go::GoMotorArray> motors_;
    std::unique_ptr<imu_sensor::ImuMonitor> imu_;
    bool started_=false, faulted_=false;
    std::string error_;
    std::atomic<bool> stopping_{true};
    std::thread supervisor_;
};
}
