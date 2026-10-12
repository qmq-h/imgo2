#ifndef IMU_SENSOR_DRIVER_HPP
#define IMU_SENSOR_DRIVER_HPP

#include "imu_sensor/serial_port.hpp"
#include "imu_sensor/yesense_parser.hpp"

namespace imu_sensor {

// Blocking/polling interface. No hidden worker thread; all methods require
// single-threaded access or external synchronization, including Open/Close.
class ImuSensor {
public:
    void Open(const std::string& device, unsigned int baud = 460800);
    void Close() noexcept;
    bool IsOpen() const { return serial_.IsOpen(); }
    // Returns the newest parsed sample in a received chunk. Older samples in
    // that chunk are intentionally dropped to reduce control-loop latency.
    // false = timeout (output unchanged), throws std::exception on I/O failure.
    // timeout applies to the whole call, including incomplete/corrupt frames.
    bool Read(ImuData& output, int timeout_ms = 100);

private:
    SerialPort serial_;
    YesenseParser parser_;
};

} // namespace imu_sensor
#endif
