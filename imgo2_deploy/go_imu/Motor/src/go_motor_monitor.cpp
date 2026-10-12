#include "go_motor_monitor.h"
#include <algorithm>
#include <stdexcept>
namespace go {
GoMotorMonitor::GoMotorMonitor(std::vector<DiagnosticBus> buses, uint32_t baud,
                               size_t timeout, std::chrono::milliseconds period)
    : states_(std::move(buses)), baud_(baud), timeout_us_(timeout), period_(period) {
    if (!baud || !timeout || period.count()<=0) throw std::invalid_argument("invalid monitor timing");
}
GoMotorMonitor::~GoMotorMonitor() { stop(); }
void GoMotorMonitor::start() {
    if (!workers_.empty()) return;
    stopping_=false;
    try { for (size_t i=0;i<states_.size();++i) workers_.emplace_back(&GoMotorMonitor::run,this,i); }
    catch (...) { stop(); throw; }
}
void GoMotorMonitor::stop() {
    { std::lock_guard<std::mutex> lock(mutex_); stopping_=true; }
    cv_.notify_all();
    for (auto& worker:workers_) if (worker.joinable()) worker.join();
    workers_.clear();
}
std::vector<DiagnosticBus> GoMotorMonitor::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_); return states_;
}
void GoMotorMonitor::run(size_t index) {
    DiagnosticBus state;
    { std::lock_guard<std::mutex> lock(mutex_); state=states_[index]; }
    SerialPort port;
    try {
        state.opened=port.open(state.device,baud_,timeout_us_);
        if (!state.opened) {
            state.error=port.lastError();
            std::lock_guard<std::mutex> lock(mutex_); states_[index]=state; return;
        }
        auto next=SerialPort::Clock::now();
        for (;;) {
            { std::lock_guard<std::mutex> lock(mutex_); if (stopping_) break; }
            const auto begin=SerialPort::Clock::now();
            TransportStats stats{state.bytesIn,state.badFrames,state.strayFrames};
            state.error.clear();
            for (auto& motor:state.motors) {
                { std::lock_guard<std::mutex> lock(mutex_); if (stopping_) break; }
                Command command; command.reset(); command.id=static_cast<uint8_t>(motor.id);
                Feedback raw;
                const auto start=SerialPort::Clock::now();
                motor.request_ok=MotorTransport::exchange(port,command,
                    start+std::chrono::microseconds(timeout_us_),raw,stats,motor.error);
                motor.feedback.valid=motor.request_ok;
                motor.rttMs=std::chrono::duration<double,std::milli>(SerialPort::Clock::now()-start).count();
                if (motor.request_ok) {
                    motor.feedback=raw;
                    motor.receive_time_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(
                        SerialPort::Clock::now().time_since_epoch()).count();
                    if (!motor.ever) motor.offset=raw.q-motor.direction*motor.startPose;
                    motor.qL=motor.direction*(raw.q-motor.offset);
                    if (!motor.ever) {
                        motor.ever=true; motor.qFirst=motor.qMin=motor.qMax=motor.qL;
                    } else {
                        motor.qMin=std::min(motor.qMin,motor.qL);
                        motor.qMax=std::max(motor.qMax,motor.qL);
                    }
                } else state.error=motor.error;
            }
            state.bytesIn=stats.bytesIn; state.badFrames=stats.badFrames; state.strayFrames=stats.strayFrames;
            state.sweepMs=std::chrono::duration<double,std::milli>(SerialPort::Clock::now()-begin).count();
            ++state.sweeps;
            std::unique_lock<std::mutex> lock(mutex_);
            states_[index]=state;
            next+=period_;
            if (next<SerialPort::Clock::now()) next=SerialPort::Clock::now();
            if (cv_.wait_until(lock,next,[&]{return stopping_;})) break;
        }
    } catch (const std::exception& error) {
        state.error=error.what();
        std::lock_guard<std::mutex> lock(mutex_); states_[index]=state;
    }
}
}
