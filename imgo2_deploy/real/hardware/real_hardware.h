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
    std::string error;
};
// 把 ImuData::valid_fields 位掩码翻成可读名字（"quat,gyro" / "none"），
// 供 fault 消息使用：一眼看出这一帧到底带了哪些字段。
std::string imuFieldNames(std::uint32_t fields);
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
    void requestStop(const std::string& reason); // Thread-safe latched unload; does not join resources.
    // 诊断用（不参与控制）：每条总线最近一轮收发的耗时(ms)与设备名，顺序一致。
    // 用来判断 io_timeout_us 是否偏紧：健康轮询应该远小于 io_timeout。
    std::vector<double> busCycleMs() const;
    std::vector<std::string> busDevices() const;
    // 诊断用：各总线定周期线程的累计最坏值（取各总线最坏）。
    //   bodyMaxUs     回调耗时（含 cycle() 的 I/O 等待）
    //   gapMaxUs      睡醒后比计划晚了多少 —— **只有这一项是"被调度器/别的线程拖了"**
    //   intervalMaxUs 相邻两轮开始时刻的最大间隔（真实控制周期最坏值）
    go::LoopStats loopStats() const;
    // 诊断用：每条总线最近一轮的三段耗时（flush/write/reply/total），顺序同 busDevices()。
    std::vector<go::BusTiming> busTiming() const;
    // 诊断用：逐事务埋点；未启动（没有电机）时返回 nullptr。
    const go::TransactionTrace* transactionTrace() const;
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
