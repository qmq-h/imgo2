#include "inference_worker.h"
#include <stdexcept>
namespace imgo2_sdk {
InferenceWorker::InferenceWorker(Infer infer,std::function<void()> reset,int period,int source_timeout)
    :infer_(std::move(infer)),reset_(std::move(reset)),period_ms_(period),source_timeout_ms_(source_timeout) {
    if(!infer_ || !reset_ || period<=0 || source_timeout<=0) throw std::invalid_argument("invalid inference worker");
}
InferenceWorker::~InferenceWorker(){stop();}
void InferenceWorker::start() {
    std::lock_guard<std::mutex> lock(mutex_);
    if(thread_.joinable()) return;
    stopping_=false;thread_=std::thread(&InferenceWorker::run,this);
}
void InferenceWorker::stop() {
    {std::lock_guard<std::mutex> lock(mutex_);stopping_=true;active_=false;}
    cv_.notify_all();if(thread_.joinable()) thread_.join();
}
void InferenceWorker::submit(const RobotState& state,const Velocity& velocity,uint64_t generation,bool active) {
    {std::lock_guard<std::mutex> lock(mutex_);
     if(generation!=generation_ || active!=active_) result_=PolicyResult{};
     state_=state;velocity_=velocity;generation_=generation;active_=active;}
    cv_.notify_all();
}
PolicyResult InferenceWorker::latest() const {std::lock_guard<std::mutex> lock(mutex_);return result_;}
void InferenceWorker::run() {
    uint64_t previous=0;bool need_reset=true;
    auto next=Clock::now();
    for(;;) {
        RobotState state;Velocity velocity;uint64_t generation;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock,[&]{return stopping_ || active_;});
            if(stopping_) return;
            state=state_;velocity=velocity_;generation=generation_;
        }
        PolicyResult out;
        try {
            if(need_reset || previous!=generation) {reset_();previous=generation;need_reset=false;next=Clock::now();}
            const auto age=std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now()-state.captured_at).count();
            if(age<0 || age>=source_timeout_ms_) throw std::runtime_error("inference input snapshot expired");
            out=infer_(state,velocity,generation);
            out.generation=generation;out.observed=state.captured_at;
        } catch(const std::exception& e) {out.error=e.what();out.generation=generation;out.generated=Clock::now();}
          catch(...) {out.error="unknown inference failure";out.generation=generation;out.generated=Clock::now();}
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if(stopping_) return;
            if(active_ && generation_==generation) result_=std::move(out);
            else need_reset=true; // Never publish an old state's in-flight result.
            next+=std::chrono::milliseconds(period_ms_);
            if(next<Clock::now()) next=Clock::now();
            cv_.wait_until(lock,next,[&]{return stopping_ || !active_ || generation_!=generation;});
            if(!active_) need_reset=true;
        }
    }
}
}
