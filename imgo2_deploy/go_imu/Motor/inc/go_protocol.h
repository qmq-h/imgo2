#pragma once
#include <cstddef>
#include <cstdint>
namespace go {
constexpr size_t kCmdLen=17, kReplyLen=16;
constexpr uint8_t kCmdHead0=0xFE, kReplyHead0=0xFD, kHead1=0xEE;
constexpr float kGearRatio=6.33f;
enum class MotorMode : uint8_t { Lock=0, Foc=1, Calibrate=2 };
enum MotorError : uint8_t { kErrNone=0, kErrOverheat=1, kErrOverCurrent=2,
    kErrOverVoltage=3, kErrEncoder=4, kErrBusVoltageLow=5, kErrWindingHot=6 };
// All public motion quantities are OUTPUT-side: rad, rad/s, N.m,
// kp in N.m/rad and kd in N.m/(rad/s). No automatic mechanical zeroing.
struct Command {
    uint8_t id=0;
    MotorMode mode=MotorMode::Foc;
    float tau=0, dq=0, q=0, kp=0, kd=0;
    void reset() { tau=dq=q=kp=kd=0; mode=MotorMode::Foc; }
};
struct Feedback {
    bool valid=false;
    uint8_t id=0;
    MotorMode mode=MotorMode::Lock;
    float tau=0, dq=0, q=0;
    int8_t temp=0;
    uint8_t error=0;
    uint16_t force=0;
};
const char* motorErrorText(uint8_t error);
uint16_t crc16(const uint8_t* data, size_t len, uint16_t init=0);
// Invalid input returns false and clears out; finite quantities saturate.
bool packCommand(const Command& cmd, uint8_t out[kCmdLen]);
bool unpackFeedback(const uint8_t* in, size_t len, Feedback& out);
// Aligns a reply header, retaining a trailing partial header if necessary.
bool findFrame(uint8_t* buf, size_t& len, size_t frameLen=kReplyLen);
}
