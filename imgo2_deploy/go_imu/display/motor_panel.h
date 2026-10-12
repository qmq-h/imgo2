#pragma once
#include "go_motor_monitor.h"
#include "display/terminal_display.hpp"
#include "display/imu_panel.h"
namespace go_display {
void RenderMonitor(const std::vector<go::DiagnosticBus>&,long long,
                   const imu_sensor::TerminalDisplay&,const imu_sensor::MonitorSnapshot&,
                   const real_config::ImuConfig&);
}
