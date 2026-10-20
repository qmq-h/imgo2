#include "policy_runner.h"
#include <torch/script.h>
#include <ATen/Parallel.h>
#include <cmath>
#include <stdexcept>
namespace imgo2_sdk {
struct PolicyRunner::Impl {torch::jit::script::Module model;bool loaded=false;};
PolicyRunner::PolicyRunner(PolicyConfig c):impl_(new Impl),config_(std::move(c)),previous_(12,0) {}
PolicyRunner::~PolicyRunner()=default;
void PolicyRunner::load(int threads) {
    if(threads<1) throw std::invalid_argument("torch threads must be positive");
    at::set_num_threads(threads);
    impl_->model=torch::jit::load(config_.model_path.string(),torch::kCPU);
    impl_->model.eval();impl_->loaded=true;
}
std::vector<float> PolicyRunner::forward(const std::vector<float>& obs) {
    if(!impl_->loaded || obs.size()!=45) throw std::invalid_argument("model not loaded or observation is not 45D");
    for(float v:obs) if(!std::isfinite(v)) throw std::invalid_argument("nonfinite model input");
    torch::NoGradGuard guard;
    auto input=torch::from_blob(const_cast<float*>(obs.data()),{1,45},torch::kFloat32);
    auto output=impl_->model.forward({input});
    if(!output.isTensor()) throw std::runtime_error("policy output is not a tensor");
    auto tensor=output.toTensor().to(torch::kCPU).to(torch::kFloat32).contiguous();
    if(!((tensor.dim()==1 && tensor.size(0)==12) || (tensor.dim()==2 && tensor.size(0)==1 && tensor.size(1)==12)))
        throw std::runtime_error("policy output must be [12] or [1,12]");
    const float* data=tensor.data_ptr<float>();std::vector<float> actions(data,data+12);
    for(float v:actions) if(!std::isfinite(v)) throw std::runtime_error("nonfinite model output");
    return actions;
}
void PolicyRunner::warmup() {
    std::vector<float> nominal(45,0);nominal[5]=-1;
    for(int i=0;i<16;++i) forward(nominal);
    reset();
}
void PolicyRunner::reset() {previous_.assign(12,0);sequence_=0;}
PolicyResult PolicyRunner::infer(const RobotState& state,const Velocity& cmd,uint64_t generation) {
    PolicyResult result;
    result.actions=clipActions(forward(buildObservation(state,cmd,previous_,config_)),config_);
    result.command=convertAction(result.actions,config_);
    previous_=result.actions;result.generation=generation;result.sequence=++sequence_;result.generated=Clock::now();result.observed=state.captured_at;
    return result;
}
}
