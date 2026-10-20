#include "hardware/real_hardware.h"
#include <chrono>
#include <stdexcept>
namespace real_hardware
{
    namespace
    {
        uint64_t nowNs()
        {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::steady_clock::now().time_since_epoch())
                .count();
        }
    }
    // 把 valid_fields 位掩码翻成可读名字：报错时一眼能看出这一帧缺哪个字段。
    std::string imuFieldNames(std::uint32_t fields)
    {
        std::string out;
        if (fields & imu_sensor::Acceleration) out += "accel,";
        if (fields & imu_sensor::AngularVelocity) out += "gyro,";
        if (fields & imu_sensor::EulerAngles) out += "euler,";
        if (fields & imu_sensor::Quaternion) out += "quat,";
        if (out.empty()) return "none";
        out.pop_back();
        return out;
    }
    RealHardware::RealHardware(real_config::HardwareConfig config) : config_(std::move(config))
    {
        config_.imu.validate();
    if (config_.imu_ready_timeout_ms <= 0 || config_.motor_stale_after_ms <= 0 ||
            config_.imu.stale_after_ms <= 0 || config_.imu.read_timeout_ms <= 0)
            throw std::invalid_argument("hardware timeouts must be positive");
        for (const auto &bus : config_.motors.buses)
            if (bus.device == config_.imu.device)
                throw std::invalid_argument("IMU and motor device overlap");
    }
    RealHardware::~RealHardware() { stop(); }
    bool RealHardware::start(std::string &error)
    {
        error.clear();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (started_)
            {
                error = error_;
                return !faulted_;
            }
        }
        stop();
        try
        {
            motors_.reset(new go::GoMotorArray(config_.motors));
            imu_.reset(new imu_sensor::ImuMonitor(config_.imu));
            faulted_ = false;
            error_.clear();
            if (!motors_->start(error))
            {
                error_ = error;
                stop();
                return false;
            }
            imu_->Start();
            const auto deadline = std::chrono::steady_clock::now() +
                                  std::chrono::milliseconds(config_.imu_ready_timeout_ms);
            bool ready = false;
            while (std::chrono::steady_clock::now() < deadline)
            {
                const auto sample = imu_->Snapshot();
                if (!sample.error.empty())
                {
                    error = sample.error;
                    break;
                }
                if (sample.received && sample.data.Has(imu_sensor::Quaternion | imu_sensor::AngularVelocity) &&
                    nowNs() - sample.data.receive_time_ns < uint64_t(config_.imu.stale_after_ms) * 1000000)
                {
                    ready = true;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            if (!ready)
            {
                if (error.empty())
                    error = "IMU ready timeout (quaternion + gyro required)";
                error_ = error;
                stop();
                return false;
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                started_ = true;
                if (!checkLocked())
                {
                    error = error_;
                }
            }
            if (!error.empty())
            {
                stop();
                return false;
            }
            stopping_ = false;
            supervisor_ = std::thread(&RealHardware::supervise, this);
            return true;
        }
        catch (const std::exception &exception)
        {
            error = exception.what();
            error_ = error;
            stop();
            return false;
        }
    }
    HardwareState RealHardware::snapshotLocked() const
    {
        HardwareState state;
        state.started = started_;
        state.faulted = faulted_;
        state.error = error_;
        state.joint_mapping_confirmed = config_.joint_mapping_confirmed;
        state.imu_mount_confirmed = config_.imu.mount_confirmed;
        if (motors_)
        {
            const auto snapshot = motors_->state();
            const auto motor_now = nowNs(); // 取在快照之后：否则刚到的反馈可能比 motor_now 还新，age 会变负
            if (snapshot.faulted)
            {
                state.faulted = true;
                if (state.error.empty())
                    state.error = snapshot.error;
            }
            for (size_t i = 0; i < snapshot.feedback.size(); ++i)
            {
                JointState joint;
                joint.feedback = snapshot.feedback[i];
                joint.receive_time_ns = snapshot.receive_time_ns[i];
                if (joint.receive_time_ns)
                    joint.age_ms = (static_cast<long long>(motor_now) - static_cast<long long>(joint.receive_time_ns)) / 1000000;
                joint.valid = joint.feedback.valid && joint.age_ms >= 0 &&
                              joint.age_ms < config_.motor_stale_after_ms && joint.feedback.error == 0 &&
                              joint.feedback.mode == go::MotorMode::Foc && joint.feedback.temp <= 80;
                if (!joint.valid)
                    joint.error = "invalid/unhealthy/stale motor feedback: age=" + std::to_string(joint.age_ms) +
                                  "ms valid=" + (joint.feedback.valid ? "1" : "0") +
                                  " mode=" + std::to_string(static_cast<int>(joint.feedback.mode)) +
                                  " (need Foc=" + std::to_string(static_cast<int>(go::MotorMode::Foc)) + ")" +
                                  " err=" + std::to_string(int(joint.feedback.error)) +
                                  " temp=" + std::to_string(int(joint.feedback.temp)) +
                                  " limits: age<" + std::to_string(config_.motor_stale_after_ms) + "ms temp<=80";
                state.joints.push_back(joint);
            }
        }
        if (imu_)
        {
            state.imu = imu_->Snapshot();
            const auto imu_now = nowNs(); // 同上：取在快照之后
            if (state.imu.received)
                state.imu_age_ms = (static_cast<long long>(imu_now) - static_cast<long long>(state.imu.data.receive_time_ns)) / 1000000;
            state.imu_valid = state.imu.received && state.imu.error.empty() && state.imu_age_ms >= 0 &&
                              state.imu_age_ms < config_.imu.stale_after_ms &&
                              state.imu.data.Has(imu_sensor::Quaternion | imu_sensor::AngularVelocity);
        }
        return state;
    }
    HardwareState RealHardware::readState() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return snapshotLocked();
    }
    void RealHardware::latchLocked(const std::string &reason)
    {
        if (!faulted_)
            error_ = reason;
        faulted_ = true;
        if (motors_)
            motors_->requestUnload(error_);
    }
    bool RealHardware::checkLocked()
    {
        if (!started_ || faulted_)
            return false;
        const auto state = snapshotLocked();
        if (state.faulted)
        {
            latchLocked(state.error.empty() ? "motor fault" : state.error);
            return false;
        }
        for (size_t i = 0; i < state.joints.size(); ++i)
            if (!state.joints[i].valid)
            {
                latchLocked("joint " + std::to_string(i) + ": " + state.joints[i].error);
                return false;
            }
        if (!state.imu_valid)
        {
            latchLocked("IMU invalid/stale: age=" + std::to_string(state.imu_age_ms) +
                        "ms received=" + (state.imu.received ? "1" : "0") +
                        " fields=" + imuFieldNames(state.imu.data.valid_fields) +
                        " error=" + (state.imu.error.empty() ? "(none)" : state.imu.error) +
                        " | need quat+gyro in the same frame and age<" +
                        std::to_string(config_.imu.stale_after_ms) + "ms");
            return false;
        }
        return true;
    }
    bool RealHardware::sendCommand(const go::Targets &command)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!checkLocked())
            return false;
        if (!motors_->setTargets(command))
        {
            latchLocked(motors_->lastError());
            return false;
        }
        return true;
    }
    void RealHardware::requestStop(const std::string& reason)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        latchLocked(reason);
    }
    std::vector<double> RealHardware::busCycleMs() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<double> out;
        if (!motors_) return out;
        const auto devices = motors_->busDevices();
        for (size_t i = 0; i < devices.size(); ++i) out.push_back(motors_->busMs(i));
        return out;
    }
    std::vector<std::string> RealHardware::busDevices() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return motors_ ? motors_->busDevices() : std::vector<std::string>{};
    }
    go::LoopStats RealHardware::loopStats() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return motors_ ? motors_->loopStats() : go::LoopStats{};
    }
    std::vector<go::BusTiming> RealHardware::busTiming() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<go::BusTiming> out;
        if (!motors_) return out;
        const auto devices = motors_->busDevices();
        for (size_t i = 0; i < devices.size(); ++i) out.push_back(motors_->busTiming(i));
        return out;
    }
    const go::TransactionTrace* RealHardware::transactionTrace() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return motors_ ? &motors_->transactionTrace() : nullptr;
    }
    void RealHardware::supervise()
    {
        while (!stopping_)
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                checkLocked();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
    void RealHardware::stop()
    {
        stopping_ = true;
        if (supervisor_.joinable())
            supervisor_.join();
        std::lock_guard<std::mutex> lock(mutex_);
        if (motors_)
        {
            motors_->requestUnload("hardware stopped");
            motors_->stop();
        }
        if (imu_)
            imu_->Stop();
        started_ = false;
    }
}
