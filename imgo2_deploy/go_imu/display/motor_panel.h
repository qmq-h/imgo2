#pragma once
#include "go_motor_monitor.h"
#include "display/terminal_display.hpp"
#include "display/imu_panel.h"
namespace go_display {
// timeout_us = 单次电机事务的等待预算，只用于显示（"单次预算 X ms"），不参与控制。
void RenderMonitor(const std::vector<go::DiagnosticBus>&,long long,
                   const imu_sensor::TerminalDisplay&,const imu_sensor::MonitorSnapshot&,
                   const real_config::ImuConfig&,size_t timeout_us);
}
