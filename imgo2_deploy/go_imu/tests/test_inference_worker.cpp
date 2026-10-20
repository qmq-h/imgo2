#include "sdk/inference_worker.h"
#include <atomic>
#include <thread>
#include <iostream>
#define CHECK(x) do{if(!(x)){std::cerr<<#x<<'\n';return 1;}}while(0)
using namespace imgo2_sdk;
int main() {
    std::atomic<int> entered{0},reset_count{0};
    InferenceWorker worker([&](const RobotState&,const Velocity&,uint64_t g){
        entered++;std::this_thread::sleep_for(std::chrono::milliseconds(80));
        PolicyResult r;r.sequence=1;r.generated=Clock::now();r.generation=g;return r;
    },[&]{reset_count++;},20);
    RobotState state;state.captured_at=Clock::now();
    worker.start();worker.submit(state,{{0,0,0}},1,true);
    auto deadline=Clock::now()+std::chrono::seconds(1);while(entered==0 && Clock::now()<deadline) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    CHECK(entered>0);worker.submit({},{{0,0,0}},2,false);
    std::this_thread::sleep_for(std::chrono::milliseconds(120));CHECK(worker.latest().sequence==0);
    state.captured_at=Clock::now();worker.submit(state,{{0,0,0}},3,true);deadline=Clock::now()+std::chrono::seconds(1);
    while(worker.latest().sequence==0 && Clock::now()<deadline) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    CHECK(worker.latest().generation==3 && reset_count>=2);
    worker.stop();worker.stop();
    InferenceWorker failure([](const auto&,const auto&,uint64_t)->PolicyResult{throw std::runtime_error("test failure");},[]{},20);
    state.captured_at=Clock::now();failure.start();failure.submit(state,{{0,0,0}},4,true);deadline=Clock::now()+std::chrono::seconds(1);
    while(failure.latest().error.empty() && Clock::now()<deadline) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    CHECK(failure.latest().error=="test failure");failure.stop();
    std::cout<<"worker generation/reset/exception passed\n";
}
