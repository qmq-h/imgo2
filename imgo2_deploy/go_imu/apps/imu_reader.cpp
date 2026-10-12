#include "imu_sensor/imu_monitor.hpp"
#include "display/terminal_display.hpp"
#include "display/imu_panel.h"
#include <csignal>
#include <iostream>
#include <thread>
namespace {
volatile std::sig_atomic_t running = 1;
void Stop(int) { running = 0; }
}
int RunImuReader(int argc, char** argv) {
    auto config = real_config::makeImuConfig();
    if (argc > 3) { std::cerr << "Usage: imu_reader [device [baud]]\n"; return 1; }
    try {
        if (argc >= 2) config.device = argv[1];
        if (argc == 3) {
            const std::string value(argv[2]);
            size_t end = 0;
            const auto baud = std::stoul(value, &end);
            if (value.empty() || value[0] == '-' || end != value.size() || baud > 921600 || !baud)
                throw std::invalid_argument("invalid baud rate");
            config.baud = static_cast<unsigned int>(baud);
        }
        std::signal(SIGINT, Stop);
        std::signal(SIGTERM, Stop);
        imu_sensor::TerminalDisplay display;
        imu_sensor::ImuMonitor monitor(config);
        monitor.Start();
        int result = 0;
        while (running) {
            const auto snapshot = monitor.Snapshot();
            display.Show("IMU monitor | " + std::to_string(config.baud) +
                " baud | Ctrl+C exits\n" + imu_sensor::MonitorPanel(snapshot, config));
            if (!snapshot.error.empty()) { result = 2; break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        monitor.Stop();
        display.Restore();
        if (result) std::cerr << "IMU error: " << monitor.Snapshot().error << '\n';
        return result;
    } catch (const std::exception& error) {
        std::cerr << "IMU error: " << error.what() << '\n'; return 2;
    }
}
