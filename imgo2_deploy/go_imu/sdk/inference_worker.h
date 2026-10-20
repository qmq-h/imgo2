#pragma once
#include "policy_runner.h"
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
namespace imgo2_sdk {
class InferenceWorker {
public:
    using Infer=std::function<PolicyResult(const RobotState&,const Velocity&,uint64_t)>;
    InferenceWorker(Infer infer,std::function<void()> reset,int period_ms,int source_timeout_ms=100);
    ~InferenceWorker();
    void start();
    void stop();
    void submit(const RobotState&,const Velocity&,uint64_t generation,bool active);
    PolicyResult latest() const;
private:
    void run();
    Infer infer_;
    std::function<void()> reset_;
    int period_ms_,source_timeout_ms_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    RobotState state_;
    Velocity velocity_{{0,0,0}};
    uint64_t generation_=0;
    bool active_=false,stopping_=false;
    PolicyResult result_;
    std::thread thread_;
};
}
