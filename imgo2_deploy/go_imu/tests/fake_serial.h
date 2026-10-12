#pragma once

// ============================================================================
//  tests/fake_serial.h —— 测试用的假串口
//
//  用法：测试目标**不编译** serialport/src/serial_port.cpp，改为包含本文件；于是
//  GoBus / GoMotor / GoMotorArray 跑的都是真实代码，字节却只在本进程里转一圈：
//    - 确定、无时序抖动、不需要任何硬件
//    - 每次写出的 17 字节命令都记录在 fake_serial::sent() 里，可以逐帧检查
//    - 默认总是回一个“健康”的 16 字节反馈
//
//  （test_motor_fake.cpp 用的是另一套带故障注入的假串口，本文件是最小版本。）
// ============================================================================

#include "go_protocol.h"
#include "serial_port.h"

#include <algorithm>
#include <cstring>
#include <deque>
#include <vector>
#include <mutex>

namespace fake_serial
{

using Bytes = std::vector<uint8_t>;

// 收到但还没被读走的反馈
inline std::deque<Bytes> &incoming()
{
    static std::deque<Bytes> queue;
    return queue;
}

// 已经写出去的每一帧命令
inline std::vector<Bytes> &rawSent()
{
    static std::vector<Bytes> frames;
    return frames;
}

inline std::mutex& mutex() { static std::mutex value; return value; }
inline std::vector<Bytes> sent() {
    std::lock_guard<std::mutex> lock(mutex()); return rawSent();
}
inline void clearSent() {
    std::lock_guard<std::mutex> lock(mutex()); rawSent().clear();
}
inline void clear()
{
    std::lock_guard<std::mutex> lock(mutex());
    incoming().clear();
    rawSent().clear();
}

// 合法的 16 字节反馈：FOC 模式、错误码 0、温度 20、位置/速度/力矩 0
inline Bytes healthyReply(int id)
{
    Bytes b{0xfd, 0xee, uint8_t(0x10 | id), 0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0};
    const auto crc = go::crc16(b.data(), 14);
    b[14] = uint8_t(crc);
    b[15] = uint8_t(crc >> 8);
    return b;
}

// 零控制量：模式/ID 之外 12 个字节全为 0（也就是“卸力”那一帧）
inline bool isZeroCommand(const Bytes &frame)
{
    return frame.size() == 17 && (frame[2] & 15) <= 14 &&
           std::all_of(frame.begin() + 3, frame.begin() + 15, [](uint8_t v) { return v == 0; });
}

// 命令帧字段（协议是小端）
inline int32_t field32(const Bytes &frame, size_t offset)
{
    return int32_t(uint32_t(frame[offset]) | (uint32_t(frame[offset + 1]) << 8) |
                   (uint32_t(frame[offset + 2]) << 16) | (uint32_t(frame[offset + 3]) << 24));
}

inline int16_t field16(const Bytes &frame, size_t offset)
{
    return int16_t(uint16_t(frame[offset]) | (uint16_t(frame[offset + 1]) << 8));
}

inline double commandQ(const Bytes &frame)
{
    return double(field32(frame, 7)) / 32768.0 * 2.0 * 3.14159265358979323846 / 6.33;
}

inline double commandDq(const Bytes &frame)
{
    return double(field16(frame, 5)) / 256.0 * 2.0 * 3.14159265358979323846 / 6.33;
}

inline double commandTau(const Bytes &frame)
{
    return double(field16(frame, 3)) / 256.0 * 6.33;
}

inline double commandKp(const Bytes &frame)
{
    return double(field16(frame, 11)) / 1280.0 * 6.33 * 6.33;
}

inline double commandKd(const Bytes &frame)
{
    return double(field16(frame, 13)) / 1280.0 * 6.33 * 6.33;
}

} // namespace fake_serial

// ---- 用假实现替换 SerialPort（测试目标不编译 serialport/src/serial_port.cpp）----
namespace go
{

SerialPort::~SerialPort() { close(); }

bool SerialPort::open(const std::string &, uint32_t b, size_t t)
{
    fd_ = 1;
    baud_ = b;
    timeout_us_ = t;
    return true;
}

void SerialPort::close()
{
    fd_ = -1;
    baud_ = 0;
}

bool SerialPort::discardInput()
{
    std::lock_guard<std::mutex> lock(fake_serial::mutex());
    fake_serial::incoming().clear();
    return true;
}

size_t SerialPort::write(const uint8_t *p, size_t n) { return writeUntil(p, n, deadline()); }

size_t SerialPort::writeUntil(const uint8_t *p, size_t n, Deadline)
{
    std::lock_guard<std::mutex> lock(fake_serial::mutex());
    fake_serial::rawSent().emplace_back(p, p + n);
    fake_serial::incoming().push_back(fake_serial::healthyReply(p[2] & 15));
    return n;
}

size_t SerialPort::readUntil(uint8_t *p, size_t n, Deadline)
{
    std::lock_guard<std::mutex> lock(fake_serial::mutex());
    if (fake_serial::incoming().empty())
    {
        error_ = "injected timeout";
        return 0;
    }
    auto &b = fake_serial::incoming().front();
    const size_t take = std::min(n, b.size());
    std::memcpy(p, b.data(), take);
    b.erase(b.begin(), b.begin() + static_cast<std::ptrdiff_t>(take));
    if (b.empty())
        fake_serial::incoming().pop_front();
    return take;
}

} // namespace go
