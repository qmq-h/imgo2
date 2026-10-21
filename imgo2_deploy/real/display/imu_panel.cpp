#include "function/my_myth.h"
#include "display/imu_panel.h"
#include <chrono>
#include <sstream>
#include <iomanip>
namespace imu_sensor {
std::string MonitorPanel(const MonitorSnapshot& snapshot,
                         const real_config::ImuConfig& config) {
    const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    const auto age_ms = snapshot.received ?
        (now_ns - static_cast<long long>(snapshot.data.receive_time_ns)) / 1000000 : 0;
    const bool fresh = snapshot.received && snapshot.error.empty() && age_ms < config.stale_after_ms;
    std::ostringstream out;
    out << std::fixed << std::setprecision(3);
    out << "IMU " << config.device << " | " << (!snapshot.error.empty() ? "ERROR" :
        !snapshot.received ? "WAITING" : fresh ? "RECEIVING" : "STALE")
        << " | age: ";
    if (snapshot.received) out << age_ms << " ms"; else out << "--";
    out << " | RX: " << std::setprecision(1) << snapshot.receive_hz << " Hz\n";
    out << std::setprecision(3);
    if (!snapshot.error.empty()) out << "IMU error: " << snapshot.error << '\n';
    out << "---------------------------------------------------\n"
           "                  Roll / X  Pitch / Y    Yaw / Z\n";
    const auto row = [&](const char* label, const std::vector<float>& values,
                         std::uint32_t field, float scale) {
        out << std::left << std::setw(18) << label << std::right;
        for (int i = 0; i < 3; ++i) {
            if (fresh && snapshot.data.Has(field)) out << std::setw(11) << values[i] * scale;
            else out << std::setw(11) << "--";
        }
        out << '\n';
    };
    row("RPY (deg)", snapshot.data.euler, EulerAngles, my_myth::radians_to_degrees_f);
    row("Gyro (rad/s)", snapshot.data.gyroscope, AngularVelocity, 1.f);
    row("Accel (m/s^2)", snapshot.data.accelerometer, Acceleration, 1.f);
    out << "---------------------------------------------------\n";
    return out.str();
}
} // namespace imu_sensor
