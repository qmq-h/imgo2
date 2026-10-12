#include "imu_sensor/yesense_parser.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>

namespace imu_sensor {
namespace {
constexpr float kScale = 1e-6f;
constexpr float kDegreesToRadians = 0.017453292519943295f;

float ReadScaledI32(const std::uint8_t* p) {
    const std::uint32_t bits = static_cast<std::uint32_t>(p[0]) |
        (static_cast<std::uint32_t>(p[1]) << 8) |
        (static_cast<std::uint32_t>(p[2]) << 16) |
        (static_cast<std::uint32_t>(p[3]) << 24);
    // Explicit signed conversion, avoiding implementation-defined unsigned cast.
    const std::int64_t value = (bits & 0x80000000U)
        ? static_cast<std::int64_t>(bits) - 0x100000000LL
        : static_cast<std::int64_t>(bits);
    return static_cast<float>(value) * kScale;
}
}

void YesenseParser::Reset() {
    buffer_.clear();
    sequence_ = 0;
}

void YesenseParser::Feed(const std::uint8_t* bytes, std::size_t size,
                         std::vector<ImuData>& output) {
    if (size != 0 && bytes == nullptr) throw std::invalid_argument("null input");
    for (std::size_t i = 0; i < size; ++i) {
        buffer_.push_back(bytes[i]);
        for (;;) {
            if (buffer_.empty()) break;
            if (buffer_[0] != 0x59 || (buffer_.size() >= 2 && buffer_[1] != 0x53)) {
                buffer_.erase(buffer_.begin());
                continue;
            }
            if (buffer_.size() < 5) break;
            const std::size_t frame_size = 7U + buffer_[4];
            if (buffer_.size() < frame_size) break;
            std::uint8_t a = 0, b = 0;
            for (std::size_t j = 2; j < frame_size - 2; ++j) {
                a = static_cast<std::uint8_t>(a + buffer_[j]);
                b = static_cast<std::uint8_t>(b + a);
            }
            if (a != buffer_[frame_size - 2] || b != buffer_[frame_size - 1]) {
                buffer_.erase(buffer_.begin()); // Search again inside damaged frame.
                continue;
            }
            ImuData sample;
            if (Decode(frame_size, sample)) {
                sample.sequence = ++sequence_;
                sample.receive_time_ns = static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count());
                output.push_back(sample);
            }
            buffer_.erase(buffer_.begin(), buffer_.begin() + frame_size);
        }
    }
}

bool YesenseParser::Decode(std::size_t frame_size, ImuData& sample) const {
    sample.tid = static_cast<std::uint16_t>(buffer_[2] |
        (static_cast<std::uint16_t>(buffer_[3]) << 8));
    const std::size_t end = frame_size - 2;
    for (std::size_t pos = 5; pos < end;) {
        if (end - pos < 2) return false;
        const std::uint8_t id = buffer_[pos];
        const std::size_t length = buffer_[pos + 1];
        pos += 2;
        if (length > end - pos) return false;
        const std::uint8_t* data = &buffer_[pos];
        switch (id) {
        case 0x10: // acceleration, m/s^2
        case 0x20: // angular velocity, deg/s on wire
        case 0x40: { // Euler angles, pitch/roll/yaw in degrees on wire
            if (length != 12) return false;
            float v[3];
            for (int j = 0; j < 3; ++j) v[j] = ReadScaledI32(data + 4 * j);
            if (id == 0x10) {
                sample.accelerometer.assign(v, v + 3);
                sample.valid_fields |= Acceleration;
            } else if (id == 0x20) {
                for (int j = 0; j < 3; ++j) sample.gyroscope[j] = v[j] * kDegreesToRadians;
                sample.valid_fields |= AngularVelocity;
            } else {
                sample.euler = {v[1] * kDegreesToRadians, v[0] * kDegreesToRadians,
                                v[2] * kDegreesToRadians};
                sample.valid_fields |= EulerAngles;
            }
            break;
        }
        case 0x41: {
            if (length != 16) return false;
            float norm2 = 0.f;
            for (int j = 0; j < 4; ++j) {
                sample.quaternion[j] = ReadScaledI32(data + 4 * j);
                norm2 += sample.quaternion[j] * sample.quaternion[j];
            }
            if (norm2 < 1e-12f) return false;
            const float norm = std::sqrt(norm2);
            for (float& v : sample.quaternion) v /= norm;
            sample.valid_fields |= Quaternion;
            break;
        }
        default: // Skip entire unknown TLV, never scan its data as new IDs.
            break;
        }
        pos += length;
    }
    return sample.valid_fields != 0;
}

} // namespace imu_sensor
