#include "imu_sensor/yesense_parser.hpp"
#include "test_packets.hpp"
#include <cmath>
#include <iostream>

struct Target {
    std::vector<double> quaternion, gyroscope, accelerometer;
};
int main() {
    try {
        const auto frame = FullPacket();
        imu_sensor::YesenseParser parser;
        std::vector<imu_sensor::ImuData> samples;
        // All possible two-chunk boundaries, including zero-sized chunks.
        for (std::size_t split = 0; split <= frame.size(); ++split) {
            parser.Reset(); samples.clear();
            parser.Feed(frame.data(), split, samples);
            if (split < frame.size()) Require(samples.empty(), "published incomplete frame");
            parser.Feed(frame.data() + split, frame.size() - split, samples);
            Require(samples.size() == 1, "fragmented frame lost");
        }
        const auto& d = samples[0];
        Require(d.tid == 0x1234 && d.sequence == 1 && d.receive_time_ns != 0, "metadata");
        Require(std::fabs(d.gyroscope[0] - 3.14159265f) < 1e-5f, "deg/s to rad/s");
        Require(std::fabs(d.gyroscope[1] + 1.57079633f) < 1e-5f, "signed decoding");
        Require(std::fabs(d.accelerometer[2] - 9.81f) < 1e-5f, "acceleration scale");
        Require(std::fabs(d.euler[0] + 0.26179939f) < 1e-5f, "Euler order/units");
        Require(d.quaternion[0] == 1.f && d.quaternion[3] == 0.f, "wxyz order");
        Target target;
        Require(d.CopyTo(target) && target.gyroscope.size() == 3, "SDK adapter");

        std::vector<std::uint8_t> quat_payload;
        AddField(quat_payload, 0x41, {0, 300000, 0, 400000});
        const auto quat_frame = Packet(quat_payload);
        std::vector<imu_sensor::ImuData> quaternion_samples;
        parser.Feed(quat_frame.data(), quat_frame.size(), quaternion_samples);
        Require(quaternion_samples.size() == 1 &&
                std::fabs(quaternion_samples[0].quaternion[1] - 0.6f) < 1e-5f &&
                std::fabs(quaternion_samples[0].quaternion[3] - 0.8f) < 1e-5f &&
                quaternion_samples[0].quaternion[0] == 0.f, "quaternion order/normalization");

        parser.Reset(); samples.clear();
        std::vector<std::uint8_t> stream{0, 0x59, 0x59};
        auto bad = frame; bad.back() ^= 0x80;
        stream.insert(stream.end(), bad.begin(), bad.end());
        stream.insert(stream.end(), frame.begin(), frame.end());
        stream.insert(stream.end(), frame.begin(), frame.end());
        parser.Feed(stream.data(), stream.size(), samples);
        Require(samples.size() == 2 && samples[1].sequence == 2, "noise/checksum/concatenation");

        std::vector<std::uint8_t> payload{0x99, 4, 0x20, 12, 0x59, 0x53};
        AddField(payload, 0x20, {0, 1000000, 0});
        const auto partial = Packet(payload);
        samples.clear(); parser.Feed(partial.data(), partial.size(), samples);
        Require(samples.size() == 1 && samples[0].valid_fields == imu_sensor::AngularVelocity,
                "unknown TLV skip or stale fields");
        const auto old = target.quaternion;
        Require(!samples[0].CopyTo(target) && target.quaternion == old, "partial adapter writes");

        // Valid checksum but invalid TLV bounds/known lengths/zero quaternion.
        for (const auto& invalid_payload : std::vector<std::vector<std::uint8_t>>{
                {0x20, 12, 0}, {0x20, 0}, {0x99},
                {0x41, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {}}) {
            const auto invalid = Packet(invalid_payload);
            samples.clear(); parser.Feed(invalid.data(), invalid.size(), samples);
            Require(samples.empty(), "malformed packet accepted");
            parser.Feed(frame.data(), frame.size(), samples);
            Require(samples.size() == 1, "malformed packet recovery");
        }
        // Maximum-length payload and byte-by-byte delivery.
        payload.assign(255, 0);
        payload[0] = 0x99; payload[1] = 239; // unknown 239-byte field
        std::vector<std::uint8_t> gyro;
        AddField(gyro, 0x20, {0, 0, -1000000});
        for (std::size_t i = 0; i < gyro.size(); ++i) payload[241 + i] = gyro[i];
        const auto max_frame = Packet(payload);
        samples.clear();
        for (std::uint8_t byte : max_frame) parser.Feed(&byte, 1, samples);
        Require(samples.size() == 1 && samples[0].Has(imu_sensor::AngularVelocity), "max frame");
        std::cout << "Parser tests passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
    return 0;
}
