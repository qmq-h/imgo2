#include "stand_state.h"
#include "../deployment_context.h"
#include "function/my_myth.h"
#include <cmath>
#include <algorithm>
namespace go_fsm { namespace states
{
    StandState::StandState(std::vector<float> start, std::vector<float> goal, StandSettings settings)
        : trajectory_(settings.duration_s)
    {
        configure(std::move(start),std::move(goal),std::move(settings));
    }
    StandState::StandState(DeploymentContext& context)
        : context_(&context),trajectory_(context.settings.stand_s) {}
    void StandState::configure(std::vector<float> start,std::vector<float> goal,StandSettings settings)
    {
        start_=std::move(start);goal_=std::move(goal);settings_=settings;
        trajectory_=go::SmoothTrajectory(settings.duration_s);
        if (start_.empty() || start_.size() != goal_.size() || !std::isfinite(settings.duration_s) ||
            !std::isfinite(settings.kp) || settings.kp < 0 || !std::isfinite(settings.kd) || settings.kd < 0 ||
            !std::isfinite(settings.tau))
            throw std::invalid_argument("invalid stand settings/shape");
        for (size_t i = 0; i < start_.size(); ++i)
            if (!std::isfinite(start_[i]) || !std::isfinite(goal_[i]))
                throw std::invalid_argument("nonfinite stand pose");
        command_.kp.assign(start_.size(), settings.kp);
        command_.kd.assign(start_.size(), settings.kd);
        if (!settings.joint_kp.empty()) command_.kp=settings.joint_kp;
        if (!settings.joint_kd.empty()) command_.kd=settings.joint_kd;
        if(command_.kp.size()!=start_.size() || command_.kd.size()!=start_.size())
            throw std::invalid_argument("stand gain vector length mismatch");
        for(float v:command_.kp) if(!std::isfinite(v) || v<0) throw std::invalid_argument("invalid stand kp");
        for(float v:command_.kd) if(!std::isfinite(v) || v<0) throw std::invalid_argument("invalid stand kd");
        command_.tau.assign(start_.size(), settings.tau);
        command_.dq.assign(start_.size(), 0.0f);
        command_.q = start_;
    }
    void StandState::enter(Clock::time_point now)
    {
        if(context_) {
            StandSettings settings;
            settings.duration_s=context_->stand_duration_s;
            settings.joint_kp=context_->fixed.kp;settings.joint_kd=context_->fixed.kd;
            configure(context_->measured,context_->settings.stand_pose,settings);
            context_->entered=now;context_->goal=context_->fixed;context_->goal.q=goal_;
        }
        phase_ = context_?Phase::Rising:Phase::Waiting;
        triggered_at_ = now;
        command_.q = start_;
        command_.dq.assign(start_.size(), 0.0f);
        if(context_) run(now,Event::None);
    }
    double StandState::elapsed(Clock::time_point now) const
    {
        return triggered() ? std::max(0.0, std::chrono::duration<double>(now - triggered_at_).count()) : 0.0;
    }
    double StandState::progress(Clock::time_point now) const { return trajectory_.progress(elapsed(now)); }
    void StandState::run(Clock::time_point now, Event event)
    {
        if (phase_ == Phase::Waiting && event == Event::Stand)
        {
            phase_ = Phase::Rising;
            triggered_at_ = now;
        }
        const double t = elapsed(now), s = trajectory_.progress(t), rate = trajectory_.rate(t);
        for (size_t i = 0; i < start_.size(); ++i)
        {
            const double delta = double(goal_[i]) - start_[i];
            command_.q[i] = float(start_[i] + s * delta);
            command_.dq[i] = float(rate * delta);
        }
        if(context_) {
            const float blend=float(my_myth::quinticProgress(t,context_->settings.blend_s));
            for(size_t i=0;i<start_.size();++i) {
                command_.kp[i]=(1-blend)*context_->previous.kp[i]+blend*context_->fixed.kp[i];
                command_.kd[i]=(1-blend)*context_->previous.kd[i]+blend*context_->fixed.kd[i];
            }
            context_->command=command_;
        }
        if (phase_ == Phase::Rising && trajectory_.finished(t))
            phase_ = Phase::Holding;
    }
}}
