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
    std::string error;
};
struct DiagnosticBus {
    std::string device, error;
    bool opened=false;
    std::vector<DiagnosticMotor> motors;
    double sweepMs=0;
    long long sweeps=0;
    size_t bytesIn=0, badFrames=0, strayFrames=0;
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
private:
    void run(size_t index);
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<DiagnosticBus> states_;
    std::vector<std::thread> workers_;
    bool stopping_=false;
    uint32_t baud_;
    size_t timeout_us_;
    std::chrono::milliseconds period_;
};
}
