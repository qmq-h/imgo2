#include "go_safety.h"

#include <cmath>

namespace go
{

namespace
{

// 把 value 夹到 [low, high]；不是有限数就报告 Rejected。
// 返回 Unchanged / Clamped / Rejected。
ClampResult clampOne(float &value, float low, float high)
{
    if (!std::isfinite(value))
        return ClampResult::Rejected;
    const float before = value;
    if (value < low)
        value = low;
    if (value > high)
        value = high;
    return value != before ? ClampResult::Clamped : ClampResult::Unchanged;
}

// 把一次 clampOne 的结果并进总结果（Rejected 优先级最高）
void merge(ClampResult &total, ClampResult one)
{
    if (one == ClampResult::Rejected)
        total = ClampResult::Rejected;
    else if (one == ClampResult::Clamped && total != ClampResult::Rejected)
        total = ClampResult::Clamped;
}

} // namespace

JointLimits Safety::limitsAt(const std::vector<JointLimits> &limits, size_t index)
{
    if (index < limits.size())
        return limits[index];
    return JointLimits{};
}

ClampResult Safety::clamp(Targets &targets, const std::vector<JointLimits> &limits)
{
    const size_t count = targets.q.size();
    if (targets.kp.size() != count || targets.kd.size() != count || targets.dq.size() != count ||
        targets.tau.size() != count)
        return ClampResult::Rejected; // 各 vector 长度不一致，整份作废

    // 第一遍只检查：发现 NaN / Inf 就整份拒绝，避免“改了一半才发现是垃圾”
    for (size_t i = 0; i < count; ++i)
        if (!std::isfinite(targets.q[i]) || !std::isfinite(targets.dq[i]) ||
            !std::isfinite(targets.tau[i]) || !std::isfinite(targets.kp[i]) ||
            !std::isfinite(targets.kd[i]))
            return ClampResult::Rejected;

    // 第二遍夹紧
    ClampResult total = ClampResult::Unchanged;
    for (size_t i = 0; i < count; ++i)
    {
        const JointLimits limit = limitsAt(limits, i);
        merge(total, clampOne(targets.q[i], limit.q_min, limit.q_max));
        merge(total, clampOne(targets.dq[i], -limit.dq_max, limit.dq_max));
        merge(total, clampOne(targets.tau[i], -limit.tau_max, limit.tau_max));
        merge(total, clampOne(targets.kp[i], 0.0f, limit.kp_max));
        merge(total, clampOne(targets.kd[i], 0.0f, limit.kd_max));
    }
    return total;
}

bool Safety::approach(Targets &current, const Targets &desired,
                      const std::vector<JointLimits> &limits, std::chrono::microseconds dt)
{
    const size_t count = current.q.size();
    if (desired.q.size() != count || desired.kp.size() != count || desired.kd.size() != count ||
        desired.dq.size() != count || desired.tau.size() != count)
        return false;
    if (current.kp.size() != count || current.kd.size() != count || current.dq.size() != count ||
        current.tau.size() != count)
        return false;
    if (dt <= std::chrono::microseconds::zero())
        return false;

    const double dt_s = double(dt.count()) / 1e6;
    bool arrived = true;
    for (size_t i = 0; i < count; ++i)
    {
        const JointLimits limit = limitsAt(limits, i);
        const double step = double(limit.dq_max) * dt_s;
        const double diff = double(desired.q[i]) - double(current.q[i]);
        if (!std::isfinite(step) || std::fabs(diff) <= step)
            current.q[i] = desired.q[i]; // 不限速，或这一步就能到
        else
        {
            current.q[i] = float(double(current.q[i]) + (diff > 0.0 ? step : -step));
            arrived = false;
        }
        current.dq[i] = desired.dq[i];
        current.tau[i] = desired.tau[i];
        current.kp[i] = desired.kp[i];
        current.kd[i] = desired.kd[i];
    }
    return arrived;
}

bool Safety::limitsValid(const std::vector<JointLimits> &limits, size_t jointCount,
                         std::string &error)
{
    if (limits.empty())
        return true; // 不限幅
    if (limits.size() != jointCount)
    {
        error = "limits must be empty or have one entry per joint (" + std::to_string(jointCount) +
                "), got " + std::to_string(limits.size());
        return false;
    }
    for (size_t i = 0; i < limits.size(); ++i)
    {
        const JointLimits &limit = limits[i];
        // 用 !(a <= b) 而不是 (a > b)：这样 NaN 也会被判为非法
        if (!(limit.q_min <= limit.q_max))
        {
            error = "joint " + std::to_string(i) + ": q_min must not exceed q_max";
            return false;
        }
        if (!(limit.dq_max >= 0.0f) || !(limit.tau_max >= 0.0f) || !(limit.kp_max >= 0.0f) ||
            !(limit.kd_max >= 0.0f))
        {
            error = "joint " + std::to_string(i) + ": limits must not be negative";
            return false;
        }
    }
    return true;
}

} // namespace go
