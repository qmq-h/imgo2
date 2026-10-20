#include "apps/runtime.h"
#include "real_config.h"
#include "go_motor_monitor.h"
#include "display/motor_panel.h"
#include <algorithm>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <unistd.h>
namespace {
using PortState=go::DiagnosticBus;
using MotorState=go::DiagnosticMotor;
constexpr size_t kTimeoutDefaultUs=40000;
constexpr long kPeriodDefaultMs=200;
void usage(const char *argv0)
{
    std::printf(
        "go_monitor —— 持续发送 FOC 全零命令（卸力），实时查看电机反馈\n"
        "\n"
        "用法:\n"
        "  %s [选项] [设备...]\n"
        "\n"
        "选项:\n"
        "  -t <毫秒>  每次请求的等待上限（默认 40）\n"
        "  -i <列表>  要轮询并显示的 ID，逗号分隔（默认 0,1,2）\n"
        "  -p <毫秒>  刷新周期（默认 200）\n"
        "  -h         显示本帮助\n"
        "\n"
        "默认设备与波特率: config/real_config.cpp；IMU 默认同时启用，无需传参\n"
        "\n"
        "输出: qOut = 输出端角度（协议层已 ÷6.33；不含 offset）\n"
        "      qL   = 逻辑角度 = direction × (qOut − offset)\n"
        "             —— 启动时 qL 对齐 start_pose_rad；控制目标用的也是 qL\n"
        "      TO   = 累计超时次数（单次等待预算 = -t 的值，默认 40 ms）\n"
        "             卸力状态下应当一直是 0；若增长，说明链路/噪声有问题，\n"
        "             面板上的 error 会带 waited/budget/bytesIn/badFrames/strayFrames\n"
        "\n"
        "用途: 核对接线、核对每台电机的**正方向**（搬动关节看 qL 往哪边变）、看 qL 范围。\n"
        "      方向据此填进 config/real_config.cpp 的 motor_directions（+1 原样 / -1 取反）。\n"
        "      offset 由每次启动姿态计算，qOut显示原始输出角。\n"
        "\n"
        "警告: 全零命令会让电机卸力、失去保持力矩，运行前请确保机械悬空或有人扶住。\n"
        "      “发送零指令”不等于“已确认实机卸力”。\n",
        argv0);
}

}
int main(int argc, char **argv) try
{
    go::MotorConfig config;
    std::vector<std::string> devices;
    std::vector<int> ids = {0, 1, 2};
    size_t timeout_us = kTimeoutDefaultUs;
    long period_ms = kPeriodDefaultMs;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help")
        {
            usage(argv[0]);
            return 0;
        }
        if (arg == "-t" && i + 1 < argc)
        {
            timeout_us = size_t(go_app::integer(argv[++i],1,10000,"-t")) * 1000;
            continue;
        }
        if (arg == "-i" && i + 1 < argc)
        {
            ids = go_app::ids(argv[++i]);
            continue;
        }
        if (arg == "-p" && i + 1 < argc)
        {
            period_ms = go_app::integer(argv[++i],10,10000,"-p");
            if (period_ms < 10 || period_ms > 10000)
            {
                std::fprintf(stderr, "-p 需要在 10~10000 毫秒之间\n");
                return 2;
            }
            continue;
        }
        if (!arg.empty() && arg[0] == '-')
        {
            std::fprintf(stderr, "无法识别的参数：%s\n\n", arg.c_str());
            usage(argv[0]);
            return 2;
        }
        devices.push_back(arg);
    }
    if (devices.empty()) {
        for (const auto& bus : config.buses) devices.push_back(bus.device);
        std::sort(devices.begin(), devices.end());
    }

    // 根据MotorConfig显示启动姿态与方向；底层诊断复用统一通信。

    std::vector<PortState> states;
    for (const std::string &device : devices)
    {
        PortState state;
        state.device = device;

        for (int id : ids)
        {
            MotorState motor;
            motor.id = id;
            for (const go::BusConfig &bus : config.buses)
            {
                if (bus.device != device)
                    continue;
                for (size_t k = 0; k < bus.motor_ids.size(); ++k)
                {
                    if (bus.motor_ids[k] != id)
                        continue;
                    // offset 与 direction 都可能留空（= 0 / +1），与 GoMotorArray 的处理一致。
                    // 两个都要读：只读一个会让 qL 悄悄算错（曾经漏读过 offset）。
                    motor.startPose =
                        k < bus.start_pose_rad.size() ? bus.start_pose_rad[k] : 0.0f;
                    motor.direction =
                        k < bus.motor_directions.size() ? bus.motor_directions[k] : 1.0f;
                    motor.fromConfig = true;
                }
            }
            state.motors.push_back(motor);
        }
        states.push_back(std::move(state));
    }

    // 按 config/real_config.cpp 里现有的总线顺序显示（例如 USB1/USB2/USB0/USB3 = FL/FR/BL/BR），
    // 这样屏幕顺序与配置、与关节编号顺序一致。配置里没有的口按给定顺序排到后面。
    {
        std::vector<size_t> order;
        std::vector<bool> used(states.size(), false);
        for (const go::BusConfig &bus : config.buses)
            for (size_t i = 0; i < states.size(); ++i)
                if (!used[i] && states[i].device == bus.device)
                {
                    order.push_back(i);
                    used[i] = true;
                }
        for (size_t i = 0; i < states.size(); ++i)
            if (!used[i])
                order.push_back(i);
        if (order.size() == states.size())
        {
            std::vector<PortState> sorted;
            sorted.reserve(states.size());
            for (size_t i : order)
                sorted.push_back(std::move(states[i]));
            states = std::move(sorted);
        }
    }

    std::printf("go_monitor: 监视 %zu 个口，ID", states.size());
    for (int id : ids)
        std::printf(" %d", id);
    std::printf("；等待上限 %zu us，刷新周期 %ld ms\n", timeout_us, period_ms);
    for (const PortState &state : states)
    {
        std::printf("go_monitor: %-22s 配置", state.device.c_str());
        for (const MotorState &motor : state.motors)
        {
            if (!motor.fromConfig)
            {
                std::printf(" | ID%d pose=+0.0000 dir=+1(默认)", motor.id);
                continue;
            }
            std::printf(" | ID%d pose=%+7.4f dir=%+d", motor.id, double(motor.startPose),
                        motor.direction < 0 ? -1 : 1);
        }
        std::printf("\n");
    }
    std::printf("go_monitor: 只发零指令（卸力）。机械请先架空或托住。\n");
    std::fflush(stdout);

    go_app::StopSignals signals;


    auto imu_config = real_config::makeImuConfig();
#ifdef GO_IMU_MONITOR_PTY_TEST
    // Test-only override; production always reads real_config.cpp.
    if (const char* path = std::getenv("GO_IMU_TEST_DEVICE")) imu_config.device = path;
#endif
    imu_sensor::ImuMonitor imu(imu_config);
    imu_sensor::TerminalDisplay display;
    go::GoMotorMonitor motors(states,config.baud,timeout_us,std::chrono::milliseconds(period_ms));
    motors.start();
    imu.Start();
    long long refresh = 0;
    auto next = std::chrono::steady_clock::now();
    while (!signals.requested())
    {
        states=motors.snapshot();
        go_display::RenderMonitor(states, ++refresh, display, imu.Snapshot(), imu_config, timeout_us);

        next += std::chrono::milliseconds(period_ms);
        if (next < std::chrono::steady_clock::now())
            next = std::chrono::steady_clock::now();
        // 分段小睡，而不是一次 sleep_until：sleep_until 不会被信号打断，
        // 那样 Ctrl+C 就得等满一整个刷新周期（-p 很大时很明显）。20 ms 一段，
        // 于是按 Ctrl+C 最多 20 ms 就退出。
        while (!signals.requested() && std::chrono::steady_clock::now() < next)
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    motors.stop();
    states=motors.snapshot();
    imu.Stop();
    display.Restore();

    // 退出汇总：本次运行见到的范围，可以直接用来填 config.limits 的 q_min/q_max
    std::printf("\ngo_monitor: 退出。本次见到的**逻辑角度**范围（可直接填 config.limits 的 q_min/q_max）：\n");
    for (const PortState &state : states)
    {
        std::printf("  %-22s", state.device.c_str());
        if (!state.opened)
        {
            std::printf(" | 打开失败: %s\n", state.error.c_str());
            continue;
        }
        for (const MotorState &motor : state.motors)
            if (motor.ever)
                std::printf(" | ID%d qL∈[%+6.2f,%+6.2f]", motor.id, motor.qMin, motor.qMax);
            else
                std::printf(" | ID%d 从未应答", motor.id);
        std::printf("\n");
        std::printf("  %-22s 轮询 %lld 次, 单轮 last/max=%.2f/%.2fms, 超时 %llu 次",
                    "", state.sweeps, state.sweepMs, state.sweepMaxMs,
                    (unsigned long long)state.timeoutTotal);
        for (const MotorState &motor : state.motors)
            std::printf(" | ID%d ok=%llu to=%llu rttMax=%.2fms", motor.id,
                        (unsigned long long)motor.okCount,
                        (unsigned long long)motor.timeoutCount, motor.rttMaxMs);
        if (state.timeoutTotal)
            std::printf("   <- 卸力状态下也超时了，原因见上方面板的 error 细节");
        std::printf("\n");
    }
    std::fflush(stdout);
    return 0;
}

catch(const std::exception& e) {std::fprintf(stderr,"monitor error: %s\n",e.what());return 2;}
