#pragma once
#include "../fsm.h"
#include "go_trajectory.h"
namespace go_fsm { struct DeploymentContext; }
namespace go_fsm { namespace states
{
    struct StandSettings
    {
        double duration_s = 8.0;
        // 默认 25（与 stand_pty 测试契约一致）；想试小增益请用命令行：go_stand --kp 5
        float kp = 25.0f, kd = 0.5f, tau = 0.0f;
        std::vector<float> joint_kp, joint_kd; // Empty keeps the standalone scalar interface.
    };
    // One implementation shared by go_stand and RobotFSM.
    class StandState final : public State
    {
    public:
        enum class Phase
        {
            Waiting,
            Rising,
            Holding
        };
        StandState(std::vector<float> start, std::vector<float> goal, StandSettings settings = {});
        explicit StandState(DeploymentContext& context);
        const char *name() const override { return "Stand"; }
        void enter(Clock::time_point now) override;
        void run(Clock::time_point now, Event event) override;
        const go::Targets &command() const override { return command_; }
        Phase phase() const { return phase_; }
        bool triggered() const { return phase_ != Phase::Waiting; }
        bool finished() const { return phase_ == Phase::Holding; }
        double elapsed(Clock::time_point now) const;
        double progress(Clock::time_point now) const;

    private:
        void configure(std::vector<float> start, std::vector<float> goal, StandSettings settings);
        DeploymentContext* context_ = nullptr; // Null only for standalone motor tool.
        std::vector<float> start_, goal_;
        StandSettings settings_;
        go::SmoothTrajectory trajectory_;
        go::Targets command_;
        Phase phase_ = Phase::Waiting;
        Clock::time_point triggered_at_{};
    };
}}
