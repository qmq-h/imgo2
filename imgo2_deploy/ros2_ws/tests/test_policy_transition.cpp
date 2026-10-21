#include "policy_transition.hpp"
#include <cassert>
#include <cmath>
#include <vector>

struct Command
{
    struct { std::vector<float> q, dq, kp, kd, tau; } motor_command;
};

int main()
{
    Command start{{{0.87f}, {0}, {60}, {2}, {0}}};
    Command target{{{1.47f}, {0}, {25}, {0.5f}, {0}}};
    auto initial = BlendPolicyCommand(start, target, 0);
    auto final = BlendPolicyCommand(start, target, 1);
    assert(initial.motor_command.q == start.motor_command.q);
    assert(initial.motor_command.kp == start.motor_command.kp);
    assert(final.motor_command.q == target.motor_command.q);
    assert(final.motor_command.kd == target.motor_command.kd);
    auto half = BlendPolicyCommand(start, target, 0.5f);
    assert(std::abs(half.motor_command.q[0] - 1.17f) < 1e-6f);
    assert(std::abs(half.motor_command.kp[0] - 42.5f) < 1e-6f);
    auto target_gains = BlendPolicyCommand(start, target, 0.5f, false);
    assert(std::abs(target_gains.motor_command.q[0] - 1.17f) < 1e-6f);
    assert(target_gains.motor_command.kp == target.motor_command.kp);
    assert(target_gains.motor_command.kd == target.motor_command.kd);
    auto first = BlendPolicyCommand(start, target, 0.01f);
    assert(std::abs(first.motor_command.q[0] - start.motor_command.q[0]) < 0.001f);
    float previous = start.motor_command.q[0];
    for (int tick = 1; tick <= 100; ++tick)
    {
        auto command = BlendPolicyCommand(start, target, tick / 100.0f);
        assert(command.motor_command.q[0] >= previous);
        assert(command.motor_command.q[0] - previous < 0.01f);
        previous = command.motor_command.q[0];
    }
    target.motor_command.q.push_back(0);
    bool caught = false;
    try { BlendPolicyCommand(start, target, 0.1f); }
    catch (const std::invalid_argument&) { caught = true; }
    assert(caught);
}
