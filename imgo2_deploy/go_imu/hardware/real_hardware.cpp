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
    RealHardware::RealHardware(real_config::HardwareConfig config) : config_(std::move(config))
    {
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
        state.imu_mount_confirmed = config_.imu_mount_confirmed;
        const auto now = nowNs();
        if (motors_)
        {
            const auto snapshot = motors_->state();
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
                    joint.age_ms = (static_cast<long long>(now) - static_cast<long long>(joint.receive_time_ns)) / 1000000;
                joint.valid = joint.feedback.valid && joint.age_ms >= 0 &&
                              joint.age_ms < config_.motor_stale_after_ms && joint.feedback.error == 0 &&
                              joint.feedback.mode == go::MotorMode::Foc && joint.feedback.temp <= 80;
                if (!joint.valid)
                    joint.error = "invalid/unhealthy/stale motor feedback";
                state.joints.push_back(joint);
            }
        }
        if (imu_)
        {
            state.imu = imu_->Snapshot();
            if (state.imu.received)
                state.imu_age_ms = (static_cast<long long>(now) - static_cast<long long>(state.imu.data.receive_time_ns)) / 1000000;
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
            latchLocked("IMU invalid/stale: " + state.imu.error);
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
