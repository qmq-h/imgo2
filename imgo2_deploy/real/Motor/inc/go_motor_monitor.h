#pragma once
#include "go_protocol.h"
#include "motor_transport.h"
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
namespace go {
struct DiagnosticMotor {
    int id=-1;
    float direction=1, offset=0, startPose=0;
    bool fromConfig=false, ever=false, request_ok=false;
    Feedback feedback; // Last successful raw output-side measurement; valid = current request.
    double qFirst=0, qMin=0, qMax=0, qL=0, rttMs=0;
    uint64_t receive_time_ns=0;
    // 累计通信计数：用来判断"卸力（不使能）状态下会不会偶发超时"。
    uint64_t okCount=0, timeoutCount=0;
    double rttMaxMs=0;
    std::string error;
};
struct DiagnosticBus {
    std::string device, error;
    bool opened=false;
    std::vector<DiagnosticMotor> motors;
    double sweepMs=0, sweepMaxMs=0;
    long long sweeps=0;
    uint64_t timeoutTotal=0;
    size_t bytesIn=0, badFrames=0, strayFrames=0;
    // ---- 流水线轮（发 N 条 → 收 N 条）的诊断 ----
    double sendMs=0, replyMs=0;   // 本轮：发完所有命令耗时 / 发完到收齐（或超时）耗时
    int roundSent=0, roundGot=0, roundMissing=0, roundDuplicate=0;
    long long errorRounds=0;      // 有过错误的轮数
    std::vector<std::string> errors; // 最近若干条错误（带轮次），由上层 drain 打印
    uint64_t errorSeq=0;             // 错误累计条数（主机据此增量读取）
};
// Diagnostic-only: deliberately has no method accepting a motion command.
class GoMotorMonitor {
public:
    GoMotorMonitor(std::vector<DiagnosticBus> buses, uint32_t baud,
                   size_t timeout_us, std::chrono::milliseconds period);
    ~GoMotorMonitor();
    void start();
    std::vector<DiagnosticBus> snapshot() const;
    void stop();
    /**
     * 切换"一轮"的收发方式：
     *   false（默认）= 逐台一问一答；
     *   true = 先连发 N 条全零命令（间隔 spacing），再按 ID 收 N 条应答，
     *          收不齐/重复/坏帧都会记进 DiagnosticBus::errors，并累加缺失计数。
     */
    void setPipelined(bool on, std::chrono::microseconds spacing = std::chrono::microseconds(200));
private:
    void run(size_t index);
    void roundSerial(SerialPort& port, DiagnosticBus& state, TransportStats& stats);
    void roundPipelined(SerialPort& port, DiagnosticBus& state, TransportStats& stats);
    void applyFeedback(DiagnosticMotor& motor, const Feedback& raw);
    void noteError(DiagnosticBus& state, const std::string& message);
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<DiagnosticBus> states_;
    std::vector<std::thread> workers_;
    bool stopping_=false;
    uint32_t baud_;
    size_t timeout_us_;
    std::chrono::milliseconds period_;
    bool pipelined_ = false;
    std::chrono::microseconds spacing_{200};
};
}
