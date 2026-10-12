#ifndef IMU_SENSOR_TEST_PACKETS_HPP
#define IMU_SENSOR_TEST_PACKETS_HPP
#include <cstdint>
#include <vector>
#include <stdexcept>

inline void Require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
inline void AddField(std::vector<std::uint8_t>& payload, std::uint8_t id,
                     const std::vector<std::int32_t>& values) {
    payload.push_back(id);
    payload.push_back(static_cast<std::uint8_t>(values.size() * 4));
    for (std::int32_t value : values) {
        const std::uint32_t bits = static_cast<std::uint32_t>(value);
        for (unsigned int i = 0; i < 4; ++i)
            payload.push_back(static_cast<std::uint8_t>(bits >> (8 * i)));
    }
}
inline std::vector<std::uint8_t> Packet(const std::vector<std::uint8_t>& payload,
                                       std::uint16_t tid = 0x1234) {
    Require(payload.size() <= 255, "payload too large");
    std::vector<std::uint8_t> frame{0x59, 0x53,
        static_cast<std::uint8_t>(tid), static_cast<std::uint8_t>(tid >> 8),
        static_cast<std::uint8_t>(payload.size())};
    frame.insert(frame.end(), payload.begin(), payload.end());
    std::uint8_t a = 0, b = 0;
    for (std::size_t i = 2; i < frame.size(); ++i) {
        a = static_cast<std::uint8_t>(a + frame[i]);
        b = static_cast<std::uint8_t>(b + a);
    }
    frame.push_back(a);
    frame.push_back(b);
    return frame;
}
inline std::vector<std::uint8_t> FullPacket() {
    std::vector<std::uint8_t> payload;
    AddField(payload, 0x10, {1000000, -2000000, 9810000});
    AddField(payload, 0x20, {180000000, -90000000, 0});
    AddField(payload, 0x40, {30000000, -15000000, 60000000});
    AddField(payload, 0x41, {1000000, 0, 0, 0});
    return Packet(payload);
}
#endif
