#ifndef IMGO2_POLICY_TRANSITION_HPP
#define IMGO2_POLICY_TRANSITION_HPP

#include <algorithm>
#include <cstddef>
#include <stdexcept>

// Interpolate the command as well as the gains, so the first policy message is
// continuous with the last stand command. Used only by the Gazebo adapter.
template <typename Command>
Command BlendPolicyCommand(const Command& start, const Command& target, float progress, bool blend_gains = true)
{
    const float t = std::clamp(progress, 0.0f, 1.0f);
    const float alpha = t * t * (3.0f - 2.0f * t);
    Command out = target;
    auto blend = [alpha](auto& result, const auto& from, const auto& to) {
        if (from.size() != to.size())
            throw std::invalid_argument("Policy transition command dimensions differ");
        for (std::size_t i = 0; i < to.size(); ++i)
            result[i] = (1.0f - alpha) * from[i] + alpha * to[i];
    };
    blend(out.motor_command.q, start.motor_command.q, target.motor_command.q);
    blend(out.motor_command.dq, start.motor_command.dq, target.motor_command.dq);
    if (blend_gains)
    {
        blend(out.motor_command.kp, start.motor_command.kp, target.motor_command.kp);
        blend(out.motor_command.kd, start.motor_command.kd, target.motor_command.kd);
        blend(out.motor_command.tau, start.motor_command.tau, target.motor_command.tau);
    }
    return out;
}

#endif
