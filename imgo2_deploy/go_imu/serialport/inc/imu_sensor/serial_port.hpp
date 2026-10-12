#pragma once
#include "serial_port.h"
#include <stdexcept>
namespace imu_sensor {
class SerialPort {
public:
    void Open(const std::string& device, unsigned int baud = 460800) {
        if (!port_.open(device, baud)) throw std::runtime_error(port_.lastError());
    }
    void Close() noexcept { port_.close(); }
    bool IsOpen() const { return port_.isOpen(); }
    std::size_t Read(std::uint8_t* data, std::size_t capacity, int timeout_ms) {
        if (timeout_ms < 0) throw std::invalid_argument("timeout must be nonnegative");
        auto result = port_.readSome(data, capacity, timeout_ms);
        if (result.status == go::SerialPort::ReadStatus::Error)
            throw std::runtime_error(port_.lastError());
        return result.count;
    }
private:
    go::SerialPort port_;
};
} // namespace imu_sensor
