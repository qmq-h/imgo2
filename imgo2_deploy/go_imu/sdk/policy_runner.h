#pragma once
#include "rl_sdk.h"
#include <memory>
namespace imgo2_sdk {
class PolicyRunner {
public:
    explicit PolicyRunner(PolicyConfig config);
    ~PolicyRunner();
    PolicyRunner(const PolicyRunner&)=delete;
    PolicyRunner& operator=(const PolicyRunner&)=delete;
    void load(int threads=1);
    void warmup();
    void reset();
    PolicyResult infer(const RobotState&,const Velocity&,uint64_t generation);
    std::vector<float> forward(const std::vector<float>& observations); // Offline deterministic check.
    const PolicyConfig& config() const {return config_;}
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    PolicyConfig config_;
    std::vector<float> previous_;
    uint64_t sequence_=0;
};
}
