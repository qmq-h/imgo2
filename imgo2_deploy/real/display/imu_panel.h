#pragma once
#include "imu_sensor/imu_monitor.hpp"
namespace imu_sensor {
std::string MonitorPanel(const MonitorSnapshot&, const real_config::ImuConfig&);
}
