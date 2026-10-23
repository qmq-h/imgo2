// MuJoCo-side motor servo. Publish targets at the host rate; evaluate PD using
// current generalized coordinates at every dynamics evaluation (not wall time).
#pragma once
#include <mujoco/mujoco.h>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

class MujocoMotorPd
{
public:
    MujocoMotorPd()
    {
        // MuJoCo's callback is process-global. Install before starting physics;
        // destroy only after every thread using MuJoCo has joined.
        if (owner_ || mjcb_control)
            throw std::runtime_error("MuJoCo control callback already owned");
        owner_ = this;
        mjcb_control = &Callback;
    }
    ~MujocoMotorPd()
    {
        if (mjcb_control == &Callback) mjcb_control = nullptr;
        owner_ = nullptr;
    }
    MujocoMotorPd(const MujocoMotorPd&) = delete;
    MujocoMotorPd& operator=(const MujocoMotorPd&) = delete;

    void Bind(const mjModel* model, const std::vector<int>& mapping)
    {
        std::vector<Channel> channels;
        std::vector<bool> used(model->nu, false);
        for (int actuator : mapping)
        {
            if (actuator < 0 || actuator >= model->nu || used[actuator] ||
                model->actuator_trntype[actuator] != mjTRN_JOINT)
                throw std::invalid_argument("Invalid MuJoCo motor mapping");
            used[actuator] = true;
            const int joint = model->actuator_trnid[2 * actuator];
            if (joint < 0 || (model->jnt_type[joint] != mjJNT_HINGE && model->jnt_type[joint] != mjJNT_SLIDE) ||
                model->actuator_gear[6 * actuator] != 1)
                throw std::invalid_argument("MuJoCo PD requires scalar joints and unit motor gear");
            channels.push_back({actuator, model->jnt_qposadr[joint], model->jnt_dofadr[joint], {}});
        }
        std::lock_guard<std::mutex> lock(mutex_);
        channels_ = std::move(channels);
        model_ = model;
    }

    template<class T>
    void Publish(const std::vector<T>& q, const std::vector<T>& dq,
                 const std::vector<T>& kp, const std::vector<T>& kd,
                 const std::vector<T>& tau)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto n = channels_.size();
        if (!n || q.size() != n || dq.size() != n || kp.size() != n || kd.size() != n || tau.size() != n)
            throw std::invalid_argument("Incomplete MuJoCo motor command");
        // Validate the entire command before publishing any channel.
        for (std::size_t i = 0; i < n; ++i)
            if (!std::isfinite(q[i]) || !std::isfinite(dq[i]) || !std::isfinite(kp[i]) ||
                !std::isfinite(kd[i]) || !std::isfinite(tau[i]) || kp[i] < 0 || kd[i] < 0)
                throw std::invalid_argument("Invalid MuJoCo motor command");
        for (std::size_t i = 0; i < n; ++i)
            channels_[i].target = {q[i], dq[i], kp[i], kd[i], tau[i]};
    }

private:
    struct Target { double q = 0, dq = 0, kp = 0, kd = 0, tau = 0; };
    struct Channel { int actuator, qpos, qvel; Target target; };
    std::mutex mutex_;
    const mjModel* model_ = nullptr;
    std::vector<Channel> channels_;
    inline static MujocoMotorPd* owner_ = nullptr;

    static void Callback(const mjModel* model, mjData* data)
    {
        if (owner_) owner_->Apply(model, data);
    }
    void Apply(const mjModel* model, mjData* data)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // Loading/compiling another model may itself call mj_forward. Never
        // apply a previous model's index mapping to that temporary model.
        if (model != model_) return;
        for (const auto& c : channels_)
        {
            const auto& t = c.target;
            double torque = t.tau + t.kp * (t.q - data->qpos[c.qpos]) +
                            t.kd * (t.dq - data->qvel[c.qvel]);
            if (model->actuator_ctrllimited[c.actuator])
                torque = std::clamp(torque, model->actuator_ctrlrange[2*c.actuator],
                                            model->actuator_ctrlrange[2*c.actuator+1]);
            data->ctrl[c.actuator] = torque;
        }
    }
};
