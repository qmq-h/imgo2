#include "imu_sensor/imu_sensor.hpp"
#include "test_packets.hpp"
#include <chrono>
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <unistd.h>

int main() {
    int master = -1;
    try {
        master = posix_openpt(O_RDWR | O_NOCTTY | O_CLOEXEC);
        Require(master >= 0 && grantpt(master) == 0 && unlockpt(master) == 0, "create PTY");
        const char* slave = ptsname(master);
        Require(slave != nullptr, "PTY slave name");
        imu_sensor::ImuSensor sensor;
        sensor.Open(slave);
        imu_sensor::ImuData sample;
        sample.tid = 42;
        Require(!sensor.Read(sample, 10) && sample.tid == 42, "timeout changes output");
        const auto frame = FullPacket();
        Require(::write(master, frame.data(), 3) == 3, "write partial");
        Require(!sensor.Read(sample, 10), "partial frame published");
        Require(::write(master, frame.data() + 3, frame.size() - 3) ==
                static_cast<ssize_t>(frame.size() - 3), "write remainder");
        Require(sensor.Read(sample, 100) && sample.tid == 0x1234, "serial frame parse");
        // A chunk containing two frames must return the newest frame.
        auto later = frame;
        later[2] = 0x35;
        later[later.size() - 2] = static_cast<std::uint8_t>(later[later.size() - 2] + 1);
        later.back() = static_cast<std::uint8_t>(later.back() + later.size() - 4);
        auto pair = frame;
        pair.insert(pair.end(), later.begin(), later.end());
        Require(::write(master, pair.data(), pair.size()) == static_cast<ssize_t>(pair.size()),
                "write concatenated frames");
        Require(sensor.Read(sample, 100) && sample.tid == 0x1235, "latest frame selection");
        auto bad = frame; bad.back() ^= 0xff;
        Require(::write(master, bad.data(), bad.size()) == static_cast<ssize_t>(bad.size()), "bad write");
        const auto start = std::chrono::steady_clock::now();
        Require(!sensor.Read(sample, 20), "bad checksum published");
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        Require(elapsed < 500, "timeout not bounded");
        sensor.Close(); sensor.Close();
        sensor.Open(slave, 115200);
        ::close(master); master = -1;
        bool disconnected = false;
        try { sensor.Read(sample, 100); } catch (const std::exception&) { disconnected = true; }
        Require(disconnected, "disconnect not reported");
        std::cout << "Linux PTY serial tests passed\n";
    } catch (const std::exception& e) {
        if (master >= 0) ::close(master);
        std::cerr << e.what() << '\n'; return 1;
    }
    return 0;
}
