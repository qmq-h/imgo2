#pragma once
#include "go_safety.h"
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <unordered_map>
namespace go_fsm {
using Clock = std::chrono::steady_clock;
enum class Event { None, Stand };
// FSM只生成目标，不打开串口；硬件故障与退出仍由应用/硬件层处理。
class State {
public:
    virtual ~State() = default;
    virtual const char* name() const = 0;
    virtual void enter(Clock::time_point now) = 0;
    virtual void run(Clock::time_point now, Event event) = 0;
    virtual void exit() {}
    virtual const go::Targets& command() const = 0;
};
class FSM {
public:
    FSM()=default;
    explicit FSM(std::unique_ptr<State> state) {addState(std::move(state));}
    ~FSM() {stop();}
    void addState(std::unique_ptr<State> state) {
        if(running_ || !state) throw std::invalid_argument("cannot register null/running FSM state");
        const std::string name=state->name();
        if(states_.count(name)) throw std::invalid_argument("duplicate FSM state");
        if(!current_) current_=state.get();
        states_.emplace(name,std::move(state));
    }
    void start(Clock::time_point now) {
        if(!current_) throw std::logic_error("FSM has no state");
        if(!running_) {current_->enter(now);running_=true;}
    }
    bool switchTo(const std::string& name,Clock::time_point now) {
        const auto it=states_.find(name);
        if(!running_ || it==states_.end() || it->second.get()==current_) return false;
        current_->exit();current_=it->second.get();current_->enter(now);return true;
    }
    const char* name() const {return current_?current_->name():"Stopped";}
    const go::Targets& update(Clock::time_point now,Event event=Event::None) {
        if(!running_) throw std::logic_error("FSM is not running");
        current_->run(now,event);return current_->command();
    }
    void stop() {if(running_) {current_->exit();running_=false;}}
private:
    std::unordered_map<std::string,std::unique_ptr<State>> states_;
    State* current_=nullptr;
    bool running_=false;
};
// 完整的一行恰好为0才触发，单独字符0、10、abc0均不会触发。
class StandInput {
public:
    bool feed(char c) {
        if (c=='\n') {
            const bool stand=!overflow_ && (line_=="0" || line_=="0\r");
            line_.clear(); overflow_=false; return stand;
        }
        if (line_.size()<32) line_+=c; else overflow_=true;
        return false;
    }
private:
    std::string line_;
    bool overflow_=false;
};
}
