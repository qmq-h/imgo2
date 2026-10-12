#ifndef IMU_SENSOR_YESENSE_PARSER_HPP
#define IMU_SENSOR_YESENSE_PARSER_HPP

#include "imu_sensor/imu_data.hpp"
#include <cstddef>
#include <vector>

namespace imu_sensor {

// Incremental binary stream parser, independent of Linux/serial hardware.
// Instances must be used from one thread or externally synchronized.
class YesenseParser {
public:
    // Appends valid samples to output, preserving packet order.
    // Handles fragmented packets, multiple packets, noise and checksum failures.
    void Feed(const std::uint8_t* bytes, std::size_t size,
              std::vector<ImuData>& output);
    void Reset();

private:
    bool Decode(std::size_t frame_size, ImuData& sample) const;
    std::vector<std::uint8_t> buffer_;
    std::uint64_t sequence_ = 0;
};

} // namespace imu_sensor
#endif
