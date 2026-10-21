#pragma once
#include <chrono>
#include <cmath>
#include <deque>
#include <stdexcept>

// Causal, zero-order-held delivery delay. Caller serializes push/read/configure.
// Times are host monotonic arrival times: this adds latency, not sensor physics.
template<class T>
class FeedbackDelay
{
public:
    using Clock = std::chrono::steady_clock;
    using Time = Clock::time_point;
    void configure(double milliseconds)
    {
        if (!std::isfinite(milliseconds) || milliseconds < 0 || milliseconds > 1000)
            throw std::invalid_argument("Gazebo feedback delay must be finite and in [0, 1000] ms");
        delay_ = std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double, std::milli>(milliseconds));
        samples_.clear();
    }
    void push(const T& value, Time arrival)
    {
        if (!samples_.empty() && arrival < samples_.back().time)
            samples_.clear();
        samples_.push_back({arrival, value});
        // Retain the latest already-due sample plus not-yet-due samples.
        trim(arrival - delay_);
    }
    bool read(Time now, T& value, Time* arrival = nullptr)
    {
        const auto cutoff = now - delay_;
        trim(cutoff);
        if (samples_.empty() || samples_.front().time > cutoff)
            return false; // Warmup: never leak a sample before its due time.
        value = samples_.front().value;
        if (arrival) *arrival = samples_.front().time;
        return true;
    }
    std::size_t size() const { return samples_.size(); }
private:
    struct Sample { Time time; T value; };
    void trim(Time cutoff)
    {
        while (samples_.size() > 1 && samples_[1].time <= cutoff)
            samples_.pop_front();
    }
    Clock::duration delay_{};
    std::deque<Sample> samples_;
};
