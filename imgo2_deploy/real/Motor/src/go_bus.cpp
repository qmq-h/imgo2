#include "go_bus.h"
#include "motor_transport.h"
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>
#include <thread>

namespace go {
/**
 * 创建一条物理 RS485 总线的所有者，记录 device、baud，不打开设备、不启动线程。
 * io_timeout_us 为每台电机一次发送+接收的预算（微秒）；command_timeout_ms 为命令有效期
 * （毫秒，1～60000，非法时抛 invalid_argument）。串口参数在 open() 时检查。
 */
GoBus::GoBus(const std::string& device, uint32_t baud, size_t io_us, size_t command_ms)
    : device_(device), baud_(baud), io_timeout_us_(io_us), command_timeout_ms_(command_ms) {
    if (command_ms == 0 || command_ms > 60000)
        throw std::invalid_argument("command timeout must be 1..60000 ms");
}
/**
 * 析构时调用 close()，限时尝试逐台卸力后释放串口及电机对象。
 * 可能等待多次事务；销毁后 addMotor() 返回的所有引用失效。
 */
GoBus::~GoBus() { close(); }
/**
 * 在 open() 前注册本总线电机，id 为 0～14，offset_rad 为有限**输出侧**零点偏移（rad）。
 * 重复 ID/非法参数抛 invalid_argument，已打开时调用抛 logic_error。
 * 返回总线持有的稳定引用；每个电机分别调用 setPD()，不要删除该对象。不同总线可复用 ID。
 */
GoMotor& GoBus::addMotor(int id, float offset_rad, float direction) {
    if (isOpen()) throw std::logic_error("register motors before opening bus");
    if (id < 0 || id > 14 || !std::isfinite(offset_rad)) throw std::invalid_argument("invalid motor ID/offset");
    // 只允许 ±1：它表示"逻辑正方向与电机原始正方向相同/相反"，不是缩放系数
    if (direction != 1.0f && direction != -1.0f)
        throw std::invalid_argument("invalid direction (must be +1 or -1)");
    for (const auto& motor : motors_)
        if (motor->id_ == id) throw std::invalid_argument("duplicate motor ID on bus");
    motors_.emplace_back(new GoMotor(id, offset_rad, direction));
    return *motors_.back();
}
/**
 * 打开并配置串口；至少先注册一台电机。成功 true，失败 false 并查看 lastError()。
 * 新连接会清除全部旧命令、反馈和故障；这里只配置，不发送、不证明电机在线。
 * 已打开时重复调用直接返回 true，不重置状态；新打开后未提交目标时 cycle() 发送零指令。
 */
bool GoBus::open() {
    if (isOpen()) return true; // Do not reset live commands or faults on repeated open.
    error_.clear();
    if (motors_.empty()) { error_ = "register at least one motor"; return false; }
    if (!port_.open(device_, baud_, io_timeout_us_)) { error_ = port_.lastError(); return false; }
    stats_ = {};
    fault_ = false;
    for (auto& motor : motors_) motor->resetSession();
    return true;
}
/**
 * 内部检查反馈有效、模式为 FOC、错误码为 0、温度不超过 80℃。
 * 不检查反馈年龄或是否到达目标；恢复时由 clearFault() 另外检查新鲜度。
 */
bool GoBus::healthy(const Feedback& fb) {
    return fb.valid && fb.mode == MotorMode::Foc && fb.error == 0 && fb.temp <= 80;
}
/**
 * 内部保留总线首次故障原因 reason，并锁定本总线所有电机、清除活动命令。
 * 不执行串口操作；调用方随后使用 unloadOnce() 尝试卸力。
 */
void GoBus::latchFault(const std::string& reason) {
    if (!fault_) error_ = reason;
    fault_ = true;
    for (auto& motor : motors_) motor->latchFault(reason);
}
/**
 * 内部执行一台电机的一次事务：使反馈失效、准备命令、清旧输入、发送并等待目标回复。
 * zero=true 强制卸力；false 使用活动目标或待机零指令。发送与接收共用同一个截止时间。
 * 返回 true 只代表得到可解析的目标 ID 反馈，健康检查由 cycle() 负责。
 */
bool GoBus::exchange(GoMotor& motor, bool zero) {
    motor.fb_.valid = false;
    Command command;
    if (!motor.wireCommand(command, zero)) { motor.error_="invalid command encoding"; return false; }
    Feedback next;
    const auto began=GoMotor::Clock::now();
    const auto trace_start=trace_?trace_->nowNs():0;
    const TransportStats before=stats_;
    TransactionTiming timing;
    const bool ok=MotorTransport::exchange(port_, command, port_.deadline(), next, stats_, motor.error_, &timing);
    round_flush_us_ += timing.flush_us;
    round_write_us_ += timing.write_us;
    round_reply_us_ += timing.reply_us;
    round_first_us_ += timing.first_byte_us;
    if(trace_) {
        // 逐事务埋点：把"有没有发、等了多久、回来多少字节、坏了几帧"记下来，
        // 事后才能分清"没发（线程没跑）/ 发了没回（bytes=0）/ 回了但是垃圾（bad>0）"。
        auto clamp16=[](size_t v){return static_cast<std::uint16_t>(v>65535?65535:v);};
        TraceRecord r;
        r.start_ns=trace_start;
        r.bus=static_cast<std::uint16_t>(trace_bus_);
        r.id=static_cast<std::uint8_t>(motor.id_);
        r.ok=ok?1:0;
        r.waited_us=static_cast<std::uint32_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                        GoMotor::Clock::now()-began).count());
        r.bytes=clamp16(stats_.bytesIn-before.bytesIn);
        r.bad=clamp16(stats_.badFrames-before.badFrames);
        r.stray=clamp16(stats_.strayFrames-before.strayFrames);
        trace_->record(r);
    }
    if (!ok) {
        // 把"等了多久 / 预算多少 / 这条总线上这次事务收到了什么"一起带上：
        //   bytesIn=0             → 一个字节都没回来（电机没应答 / 链路断了）
        //   badFrames>0           → 字节回来了但 CRC 不过（多为电机换向噪声/EMI）
        //   strayFrames>0         → 收到的是别的 ID 的回复（接线串了/回显）
        const auto waited=std::chrono::duration_cast<std::chrono::microseconds>(GoMotor::Clock::now()-began).count();
        motor.error_ += " [id=" + std::to_string(motor.id_) +
                        " waited=" + std::to_string(waited) + "us" +
                        " budget=" + std::to_string(io_timeout_us_) + "us" +
                        " bytesIn=" + std::to_string(stats_.bytesIn - before.bytesIn) +
                        " badFrames=" + std::to_string(stats_.badFrames - before.badFrames) +
                        " strayFrames=" + std::to_string(stats_.strayFrames - before.strayFrames) + "]";
        return false;
    }
    const float q=motor.direction_*(next.q-motor.offset_);
    if (!std::isfinite(q)) { motor.error_="feedback-offset overflow"; return false; }
    next.q=q; next.dq*=motor.direction_; next.tau*=motor.direction_;
    motor.fb_=next;
    motor.feedback_time_=GoMotor::Clock::now();
    if (!motor.fault_) motor.error_.clear();
    return true;
}
/**
 * 内部按注册顺序对每台电机执行一次零指令请求/应答；一台失败仍继续下一台。
 * 每次等待有期限，不连续突发多台请求；不解除故障，也不保证全部电机已卸力。
 */
void GoBus::unloadOnce() {
    // Request/reply even during cleanup: never burst writes onto a half-duplex bus.
    for (auto& motor : motors_) exchange(*motor, true);
}
/**
 * 由调用方在单线程周期中调用：按注册顺序逐台发送一帧并等待对应反馈，再处理下一台。
 * 本函数不创建线程、不休眠调度；先分别更新各电机目标，再调用一次即可处理整条总线。
 * 全部事务成功、反馈健康且无锁定故障时返回 true；否则锁定本总线故障并逐台尝试卸力。
 * 故障期间仍可调用以采集零指令反馈，但仍返回 false，不自动恢复运动。
 * 活动命令仅由上层成功提交续期；过期检查依赖 cycle() 运行，线程停止时不会自动执行。
 */
bool GoBus::cycle() {
    round_flush_us_ = 0;
    round_write_us_ = 0;
    round_reply_us_ = 0;
    round_first_us_ = 0;
    if (!isOpen()) { error_ = "bus closed"; return false; }
    if (external_stop_ && external_stop_->load()) latchFault("whole robot stop requested");
    for (auto& motor : motors_) motor->fb_.valid = false;
    for (auto& motor : motors_) {
        if (motor->active_ && GoMotor::Clock::now()-motor->command_time_ >= std::chrono::milliseconds(command_timeout_ms_))
            motor->latchFault("command expired", true);
        if (motor->fault_) latchFault("motor " + std::to_string(motor->id_) + ": " + motor->error_);
    }
    if (fault_) { unloadOnce(); return false; } // Allows feedback recovery, never automatic motion.
    if (pipelined_) return cyclePipelined();
    for (auto& motor : motors_) {
        if (external_stop_ && external_stop_->load()) {
            latchFault("whole robot stop requested"); unloadOnce(); return false;
        }
        // Recheck immediately before sending: previous transactions may have consumed time.
        if (motor->active_ && GoMotor::Clock::now()-motor->command_time_ >= std::chrono::milliseconds(command_timeout_ms_)) {
            motor->latchFault("command expired", true);
            latchFault("motor " + std::to_string(motor->id_) + ": command expired");
            unloadOnce(); return false;
        }
        if (!exchange(*motor, false) || !healthy(motor->fb_)) {
            const std::string reason = motor->fb_.valid ? "unhealthy feedback (mode/error/temperature)" : motor->error_;
            latchFault("motor " + std::to_string(motor->id_) + ": " + reason);
            unloadOnce(); return false;
        }
    }
    error_.clear(); return true;
}
/**
 * 流水线一轮：先连发本总线全部命令（相邻间隔 gap_），再按 ID 收全部应答。
 * 判定与故障语义和逐台模式完全一致：任一台没回/反馈不健康 → 锁故障 + 逐台卸力。
 */
bool GoBus::cyclePipelined() {
    const auto round_begin = GoMotor::Clock::now();
    if (external_stop_ && external_stop_->load()) { latchFault("whole robot stop requested"); unloadOnce(); return false; }
    struct Sent { GoMotor* motor; GoMotor::Clock::time_point at; std::uint64_t trace_ns; };
    std::vector<Sent> sent;
    sent.reserve(motors_.size());
    for (size_t i = 0; i < motors_.size(); ++i) {
        GoMotor& motor = *motors_[i];
        if (external_stop_ && external_stop_->load()) { latchFault("whole robot stop requested"); unloadOnce(); return false; }
        if (motor.active_ && GoMotor::Clock::now() - motor.command_time_ >= std::chrono::milliseconds(command_timeout_ms_)) {
            motor.latchFault("command expired", true);
            latchFault("motor " + std::to_string(motor.id_) + ": command expired");
            unloadOnce(); return false;
        }
        Command command;
        if (!motor.wireCommand(command, false)) {
            motor.error_ = "invalid command encoding";
            latchFault("motor " + std::to_string(motor.id_) + ": " + motor.error_);
            unloadOnce(); return false;
        }
        const std::uint64_t trace_ns = trace_ ? trace_->nowNs() : 0;
        TransactionTiming timing;
        std::string err;
        // 一轮只在第一条 flush；后续不能 flush，否则会把前一条命令的应答清掉
        if (!MotorTransport::send(port_, command, port_.deadline(), err, &timing, i == 0)) {
            motor.error_ = err;
            latchFault("motor " + std::to_string(motor.id_) + ": " + err);
            unloadOnce(); return false;
        }
        round_flush_us_ += timing.flush_us;
        round_write_us_ += timing.write_us;
        sent.push_back({&motor, GoMotor::Clock::now(), trace_ns});
        if (i + 1 < motors_.size()) std::this_thread::sleep_for(gap_);
    }
    const auto send_done = GoMotor::Clock::now();
    const TransportStats before = stats_;

    // 收：按 ID 归集，直到全部到齐或预算用尽
    std::array<std::uint8_t, 512> buffer{};
    std::size_t used = 0;
    std::map<int, Feedback> got;
    std::map<int, GoMotor::Clock::time_point> reply_at;
    const auto deadline = send_done + std::chrono::microseconds(io_timeout_us_);
    for (;;) {
        // 先解析已到的字节，再判超时：否则"在截止前最后一刻到达"的数据会被计数却不解析
        while (findFrame(buffer.data(), used)) {
            Feedback raw;
            if (!unpackFeedback(buffer.data(), kReplyLen, raw)) {
                ++stats_.badFrames;
                std::memmove(buffer.data(), buffer.data() + 1, used - 1);
                --used;
                continue;
            }
            bool known = false;
            for (const auto& item : sent) if (item.motor->id_ == raw.id) { known = true; break; }
            if (!known || got.count(raw.id)) {
                ++stats_.strayFrames;
            } else {
                got.emplace(raw.id, raw);
                reply_at.emplace(raw.id, GoMotor::Clock::now());
            }
            std::memmove(buffer.data(), buffer.data() + kReplyLen, used - kReplyLen);
            used -= kReplyLen;
        }
        if (got.size() >= sent.size()) break;
        if (GoMotor::Clock::now() >= deadline) break;
        const auto count = port_.readUntil(buffer.data() + used, buffer.size() - used, deadline);
        if (!count) break;
        if (round_first_us_ == 0)
            round_first_us_ = static_cast<std::uint32_t>(std::chrono::duration_cast<
                std::chrono::microseconds>(GoMotor::Clock::now() - round_begin).count());
        used += count;
        stats_.bytesIn += count;
    }
    const auto collect_done = GoMotor::Clock::now();
    round_reply_us_ = static_cast<std::uint32_t>(std::chrono::duration_cast<
        std::chrono::microseconds>(collect_done - send_done).count());

    // 判定：按注册顺序，报告最低 ID 的故障（与逐台模式一致）
    for (const auto& item : sent) {
        GoMotor& motor = *item.motor;
        const auto it = got.find(motor.id_);
        if (it != got.end()) {
            if (trace_) traceRound(motor, item.trace_ns, true,
                static_cast<std::uint32_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                    reply_at[motor.id_] - item.at).count()), kReplyLen, 0, 0);
            if (!applyFeedback(motor, it->second)) {
                latchFault("motor " + std::to_string(motor.id_) + ": " + motor.error_);
                unloadOnce(); return false;
            }
            if (!healthy(motor.fb_)) {
                latchFault("motor " + std::to_string(motor.id_) + ": unhealthy feedback (mode/error/temperature)");
                unloadOnce(); return false;
            }
        } else {
            const auto waited = static_cast<long>(std::chrono::duration_cast<std::chrono::microseconds>(
                collect_done - item.at).count());
            const size_t bytes = stats_.bytesIn - before.bytesIn;
            const size_t bad = stats_.badFrames - before.badFrames;
            const size_t stray = stats_.strayFrames - before.strayFrames;
            auto clamp16 = [](size_t v) { return static_cast<std::uint16_t>(v > 65535 ? 65535 : v); };
            if (trace_) traceRound(motor, item.trace_ns, false,
                static_cast<std::uint32_t>(waited), clamp16(bytes), clamp16(bad), clamp16(stray));
            motor.fb_.valid = false;
            motor.error_ = std::string(bytes ? "feedback timeout" : "I/O timeout") +
                " [id=" + std::to_string(motor.id_) + " round waited=" + std::to_string(waited) +
                "us budget=" + std::to_string(io_timeout_us_) + "us bytesIn=" + std::to_string(bytes) +
                " badFrames=" + std::to_string(bad) + " strayFrames=" + std::to_string(stray) + "]";
            latchFault("motor " + std::to_string(motor.id_) + ": " + motor.error_);
            unloadOnce(); return false;
        }
    }
    error_.clear();
    return true;
}

/**
 * 把一帧已解析的反馈按方向/零点变换后写入电机（与逐台模式同一套换算）。
 */
bool GoBus::applyFeedback(GoMotor& motor, Feedback next) {
    const float q = motor.direction_ * (next.q - motor.offset_);
    if (!std::isfinite(q)) { motor.error_ = "feedback-offset overflow"; return false; }
    next.q = q;
    next.dq *= motor.direction_;
    next.tau *= motor.direction_;
    motor.fb_ = next;
    motor.feedback_time_ = GoMotor::Clock::now();
    if (!motor.fault_) motor.error_.clear();
    return true;
}

/**
 * 流水线模式下按电机补一条埋点（逐台模式在 exchange() 内记录）。
 */
void GoBus::traceRound(const GoMotor& motor, std::uint64_t start_ns, bool ok,
                       std::uint32_t waited_us, std::uint16_t bytes,
                       std::uint16_t bad, std::uint16_t stray) {
    if (!trace_) return;
    TraceRecord r;
    r.start_ns = start_ns;
    r.bus = static_cast<std::uint16_t>(trace_bus_);
    r.id = static_cast<std::uint8_t>(motor.id_);
    r.ok = ok ? 1 : 0;
    r.waited_us = waited_us;
    r.bytes = bytes;
    r.bad = bad;
    r.stray = stray;
    trace_->record(r);
}

/**
 * 显式恢复入口，不收发：要求串口打开且所有电机现有反馈健康、年龄小于命令有效期。
 * 可先在故障状态调用 cycle() 获取新反馈，再调用本函数；成功 true，否则 false 并保留故障。
 * 成功会清除故障和所有旧目标，即使原先无故障也会清目标；必须重新提交才能运动。
 */
bool GoBus::clearFault() {
    if (!isOpen()) { error_ = "bus closed"; return false; }
    const auto now = GoMotor::Clock::now();
    for (const auto& motor : motors_) {
        if (!healthy(motor->fb_) || now-motor->feedback_time_ >= std::chrono::milliseconds(command_timeout_ms_)) {
            // Preserve the original fault; failed recovery is not a new root cause.
            if (error_.empty()) error_ = "fresh healthy feedback required for all motors";
            return false;
        }
    }
    for (auto& motor : motors_) {
        motor->setZero(); motor->fault_ = motor->expired_ = false;
        motor->command_time_ = GoMotor::Clock::time_point{}; motor->error_.clear();
    }
    fault_ = false; error_.clear(); return true;
}
/**
 * 逐台尝试全零 FOC 请求/应答，共进行三轮，随后关闭串口并使全部反馈失效。
 * 每次事务受 io_timeout_us 限制，总等待预算约为 3×电机数×单次超时，另有系统开销。
 * 保留注册对象以便重新打开；重复关闭不再通信。发送零指令不等于已确认实机卸力。
 */
void GoBus::close() {
    if (isOpen()) {
        for (auto& motor : motors_) motor->setZero();
        for (int attempt = 0; attempt < 3; ++attempt) {
            for (auto& motor : motors_) {
                if (!exchange(*motor, true) && error_.empty())
                    error_ = "shutdown motor " + std::to_string(motor->id_) + ": " + motor->error_;
            }
        }
    }
    port_.close(); stats_ = {};
    for (auto& motor : motors_) { motor->setZero(); motor->fb_.valid = false; }
}
}
