#include "fsm/states/stand_state.h"
#include <cmath>
#include <iostream>
#define CHECK(x) do {if(!(x)) {std::cerr<<#x<<'\n';return 1;}} while(0)
int main() {
    using namespace go_fsm;
    using namespace go_fsm::states;
    StandInput input;
    CHECK(!input.feed('0')); CHECK(input.feed('\n'));
    CHECK(!input.feed('1')); CHECK(!input.feed('0')); CHECK(!input.feed('\n'));
    CHECK(!input.feed('0')); CHECK(!input.feed('\r')); CHECK(input.feed('\n'));
    const std::vector<float> from={0.06f,1.17f,-2.88f}, to={0,0.87f,-1.82f};
    auto owner=std::unique_ptr<StandState>(new StandState(from,to));auto* state=owner.get();
    FSM fsm(std::move(owner));const auto t=Clock::time_point{};fsm.start(t);
    const auto seconds=[](int n){return std::chrono::seconds(n);};
    CHECK(fsm.update(t+seconds(100)).q==from);CHECK(!state->triggered());
    CHECK(fsm.update(t+seconds(100),Event::Stand).q==from);
    auto middle=fsm.update(t+seconds(104),Event::Stand); // 再次按0不重置计时
    for(size_t i=0;i<3;++i) {
        CHECK(std::abs(middle.q[i]-(from[i]+to[i])/2)<1e-6);
        CHECK(std::abs(middle.dq[i]-1.875*(to[i]-from[i])/8)<1e-6);
    }
    CHECK(fsm.update(t+seconds(108)).q==to);CHECK(state->finished());
    const auto held=fsm.update(t+seconds(200),Event::Stand);
    CHECK(held.q==to);for(auto v:held.dq) CHECK(v==0);
    fsm.stop();fsm.stop();bool rejected=false;
    try{fsm.update(t);}catch(const std::logic_error&){rejected=true;}CHECK(rejected);
    // 默认整机轨迹全程通过配置限幅，且不超速度上限。
    const auto config=go::makeDefaultConfig();std::vector<float> goal;for(size_t i=0;i<config.buses.size();++i)goal.insert(goal.end(),{0.f,.87f,-1.82f});
    std::vector<float> start;for(const auto& b:config.buses) start.insert(start.end(),b.start_pose_rad.begin(),b.start_pose_rad.end());
    StandState real(start,goal);real.enter(t);real.run(t,Event::Stand);
    for(int ms=0;ms<=8500;ms+=10) {
        real.run(t+std::chrono::milliseconds(ms),Event::None);auto cmd=real.command();
        CHECK(go::Safety::clamp(cmd,config.limits)==go::ClampResult::Unchanged);
    }
    bool bad=false;try{StandSettings s;s.duration_s=0;StandState invalid(from,to,s);}catch(const std::invalid_argument&){bad=true;}CHECK(bad);
    std::cout<<"stand FSM passed\n";
}
