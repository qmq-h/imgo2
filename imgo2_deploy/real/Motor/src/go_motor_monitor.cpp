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

void GoMotorMonitor::setPipelined(bool on, std::chrono::microseconds spacing) {
    if (workers_.empty()) { // 只允许在 start() 之前切换，避免运行中改语义
        pipelined_ = on;
        if (spacing.count() > 0) spacing_ = spacing;
    }
}

void GoMotorMonitor::noteError(DiagnosticBus& state, const std::string& message) {
    state.errors.push_back("[第 " + std::to_string(state.sweeps + 1) + " 轮] " + message);
    ++state.errorSeq;
    if (state.errors.size() > 64) state.errors.erase(state.errors.begin());
}

void GoMotorMonitor::applyFeedback(DiagnosticMotor& motor, const Feedback& raw) {
    motor.feedback.valid = true;
    motor.feedback = raw;
    motor.receive_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        SerialPort::Clock::now().time_since_epoch()).count();
    ++motor.okCount;
    if (!motor.ever) motor.offset = raw.q - motor.direction * motor.startPose;
    motor.qL = motor.direction * (raw.q - motor.offset);
    if (!motor.ever) {
        motor.ever = true; motor.qFirst = motor.qMin = motor.qMax = motor.qL;
    } else {
        motor.qMin = std::min(motor.qMin, motor.qL);
        motor.qMax = std::max(motor.qMax, motor.qL);
    }
}

// 逐台一问一答（原有行为，保留给 --serial）
void GoMotorMonitor::roundSerial(SerialPort& port, DiagnosticBus& state, TransportStats& stats) {
    for (auto& motor : state.motors) {
        { std::lock_guard<std::mutex> lock(mutex_); if (stopping_) break; }
        Command command; command.reset(); command.id = static_cast<uint8_t>(motor.id);
        Feedback raw;
        const TransportStats before = stats;
        const auto start = SerialPort::Clock::now();
        motor.request_ok = MotorTransport::exchange(port, command,
            start + std::chrono::microseconds(timeout_us_), raw, stats, motor.error);
        motor.feedback.valid = motor.request_ok;
        motor.rttMs = std::chrono::duration<double, std::milli>(SerialPort::Clock::now() - start).count();
        if (motor.request_ok) {
            motor.rttMaxMs = std::max(motor.rttMaxMs, motor.rttMs);
            applyFeedback(motor, raw);
        } else {
            ++motor.timeoutCount; ++state.timeoutTotal;
            const auto waited = std::chrono::duration_cast<std::chrono::microseconds>(
                SerialPort::Clock::now() - start).count();
            motor.error += " [waited=" + std::to_string(waited) + "us budget=" +
                           std::to_string(timeout_us_) + "us bytesIn=" +
                           std::to_string(stats.bytesIn - before.bytesIn) + " badFrames=" +
                           std::to_string(stats.badFrames - before.badFrames) + " strayFrames=" +
                           std::to_string(stats.strayFrames - before.strayFrames) + "]";
            state.error = motor.error;
        }
    }
}

// 流水线一轮：连发 N 条（间隔 spacing_），再按 ID 收 N 条
void GoMotorMonitor::roundPipelined(SerialPort& port, DiagnosticBus& state, TransportStats& stats) {
    const auto budget = std::chrono::microseconds(timeout_us_);
    port.discardInput();            // 丢掉上一轮可能残留的字节，避免迟到帧被误认成本轮的
    const auto send_begin = SerialPort::Clock::now();

    // 1) 连发：每台一条全零(FOC)命令
    int sent = 0;
    const auto send_deadline = send_begin + budget;
    for (size_t i = 0; i < state.motors.size(); ++i) {
        { std::lock_guard<std::mutex> lock(mutex_); if (stopping_) return; }
        DiagnosticMotor& motor = state.motors[i];
        Command command; command.reset(); command.id = static_cast<uint8_t>(motor.id);
        std::string err;
        // 一轮只在开头 flush 一次（见 roundPipelined 的第一行），这里不能再 flush
        if (MotorTransport::send(port, command, send_deadline, err, nullptr, /*flush=*/false)) {
            ++sent;
        } else {
            motor.request_ok = false;
            motor.error = "send failed: " + err;
            noteError(state, "ID" + std::to_string(motor.id) + " 发送失败: " + err);
        }
        if (i + 1 < state.motors.size()) std::this_thread::sleep_for(spacing_);
    }
    const auto send_done = SerialPort::Clock::now();
    state.roundSent = sent;
    state.sendMs = std::chrono::duration<double, std::milli>(send_done - send_begin).count();

    // 2) 收：按 ID 归集，直到收齐或预算用尽
    std::array<uint8_t, 512> buffer{};
    size_t used = 0;
    std::vector<int> got;
    int duplicates = 0;
    const auto deadline = send_done + budget;
    for (;;) {
        // 先解析再判超时（理由同上：截止前最后一刻到达的帧也要被解析）
        while (findFrame(buffer.data(), used)) {
            Feedback raw;
            if (!unpackFeedback(buffer.data(), kReplyLen, raw)) {
                ++stats.badFrames;
                std::memmove(buffer.data(), buffer.data() + 1, used - 1);
                --used;
                continue;
            }
            const auto it = std::find_if(state.motors.begin(), state.motors.end(),
                [&](const DiagnosticMotor& m) { return m.id == raw.id; });
            if (it == state.motors.end()) {
                ++stats.strayFrames;
            } else if (std::find(got.begin(), got.end(), raw.id) != got.end()) {
                ++duplicates;
                noteError(state, "ID" + std::to_string(raw.id) + " 重复应答（多为上一轮的迟到帧）");
            } else {
                applyFeedback(*it, raw);
                got.push_back(raw.id);
            }
            std::memmove(buffer.data(), buffer.data() + kReplyLen, used - kReplyLen);
            used -= kReplyLen;
        }
        // 收齐就立刻返回，不要再等一次 readUntil（否则会白等到预算结束）
        if (static_cast<int>(got.size()) >= sent) break;
        if (SerialPort::Clock::now() >= deadline) break;
        const auto count = port.readUntil(buffer.data() + used, buffer.size() - used, deadline);
        if (!count) break;
        used += count; stats.bytesIn += count;
    }
    const auto receive_done = SerialPort::Clock::now();
    state.roundGot = static_cast<int>(got.size());
    state.roundDuplicate = duplicates;
    state.replyMs = std::chrono::duration<double, std::milli>(receive_done - send_done).count();
    for (DiagnosticMotor& motor : state.motors) motor.rttMs = state.replyMs;
    if (state.replyMs > 0) {
        for (DiagnosticMotor& motor : state.motors) motor.rttMaxMs = std::max(motor.rttMaxMs, state.replyMs);
    }

    // 3) 没收到的：记缺失 + 明确报错
    state.roundMissing = 0;
    for (DiagnosticMotor& motor : state.motors) {
        if (std::find(got.begin(), got.end(), motor.id) != got.end()) { motor.request_ok = true; continue; }
        motor.request_ok = false;
        ++motor.timeoutCount; ++state.timeoutTotal; ++state.roundMissing;
        const auto waited = std::chrono::duration_cast<std::chrono::microseconds>(
            receive_done - send_begin).count();
        motor.error = "missing reply [round waited=" + std::to_string(waited) + "us budget=" +
                      std::to_string(timeout_us_) + "us got=" + std::to_string(got.size()) + "/" +
                      std::to_string(sent) + "]";
        noteError(state, "ID" + std::to_string(motor.id) + " 无应答（本轮收 " +
                         std::to_string(got.size()) + "/" + std::to_string(sent) +
                         "，等待 " + std::to_string(waited) + "us）");
    }
    const bool round_bad = state.roundMissing > 0 || duplicates > 0 ||
                           state.roundSent < static_cast<int>(state.motors.size());
    if (round_bad) ++state.errorRounds;
    if (!state.errors.empty()) state.error = state.errors.back();
}
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
            if (pipelined_) roundPipelined(port, state, stats);
            else roundSerial(port, state, stats);
            state.bytesIn=stats.bytesIn; state.badFrames=stats.badFrames; state.strayFrames=stats.strayFrames;
            state.sweepMs=std::chrono::duration<double,std::milli>(SerialPort::Clock::now()-begin).count();
            state.sweepMaxMs=std::max(state.sweepMaxMs,state.sweepMs);
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
