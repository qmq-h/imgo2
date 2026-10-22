/* ----------------------------------------------------------------------------
 * 用法
 *   ./go_monitor [参数] [设备...]
 *   不带设备 = 用 config/real_config.cpp 的 4 条总线（/dev/robot/leg_*），ID 0,1,2。
 *   只发全零 FOC 指令（卸力），不跑策略；机械必须吊住/托住。
 *
 * 参数（默认值）
 *   -t <ms>        单次请求 / 每轮收集的等待上限（默认 40）
 *   -i <列表>      轮询的 ID，逗号分隔（默认 0,1,2）
 *   -p <ms>        每轮轮询周期（流水线默认 5，逐台默认 200；最小 5）
 *   -r <ms>        面板刷新周期（默认 100，与 -p 解耦）
 *   -s <us>        流水线里相邻两条命令的间隔（默认 200；低于 ~150 有撞车风险）
 *   --pipelined    连发 N 条 → 按 ID 收 N 条（默认）
 *   --serial       逐台一问一答（旧行为）
 *   --plain        不打面板，逐行统计（适合重定向 / 管道）
 *   --quiet        只打每秒统计，不逐条打错误
 *   --error-lines  每秒最多打印几条错误（默认 5）
 *   --strict       第一次收发错误就退出（返回码 2）
 *   -h
 *
 * 面板读法：[发0.6/收1.7 缺0 重复0 错轮0]
 *   发 = 3 条命令写完的耗时（含 -s 间隔）；收 = 发完到全部收齐的耗时；
 *   缺 = 本轮缺应答台数；重复 = 重复应答；错轮 = 累计出错的轮数。
 * ---------------------------------------------------------------------------- */
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
        "  -p <毫秒>  每轮轮询周期（默认：流水线 5 / 逐台 200；最小 5）\n"
        "  -r <毫秒>  面板刷新周期（默认 100，与 -p 解耦）\n"
        "  -s <微秒>  流水线里相邻两条命令的间隔（默认 200）\n"
        "  --pipelined 连发 N 条 → 按 ID 收 N 条（默认）\n"
        "  --serial    逐台一问一答（旧行为）\n"
        "  --plain     不打面板：每秒一行统计，错误逐条立即打印\n"
        "  --strict    出现第一次收发错误就退出（返回码 2）\n"
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
    bool period_given = false;
    bool pipelined = true;          // 默认：发 N 条 → 收 N 条
    bool plain = false;             // 不打面板，逐条打印（错误立即出现）
    bool strict = false;            // 第一次收发错误就退出（返回 2）
    bool quiet = false;             // 只打 1Hz 统计，不逐条打错误
    long error_lines_per_s = 5;     // plain 模式每秒最多打印几条错误（防止刷屏）
    long refresh_ms = 100;          // 面板刷新周期（与轮询周期解耦）
    long spacing_us = 200;          // 流水线里相邻两条命令的间隔

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
            period_ms = go_app::integer(argv[++i],5,10000,"-p");
            period_given = true;
            continue;
        }
        if (arg == "-r" && i + 1 < argc) { refresh_ms = go_app::integer(argv[++i],10,10000,"-r"); continue; }
        if (arg == "-s" && i + 1 < argc) { spacing_us = go_app::integer(argv[++i],0,5000,"-s"); continue; }
        if (arg == "--pipelined") { pipelined = true; continue; }
        if (arg == "--serial") { pipelined = false; continue; }
        if (arg == "--plain") { plain = true; continue; }
        if (arg == "--strict") { strict = true; continue; }
        if (arg == "--quiet") { quiet = true; continue; }
        if (arg == "--error-lines" && i + 1 < argc) { error_lines_per_s = go_app::integer(argv[++i],0,1000,"--error-lines"); continue; }
        if (!arg.empty() && arg[0] == '-')
        {
            std::fprintf(stderr, "无法识别的参数：%s\n\n", arg.c_str());
            usage(argv[0]);
            return 2;
        }
        devices.push_back(arg);
    }
    if (!period_given && pipelined) period_ms = 5; // 流水线默认 5ms/轮（面板由 -r 单独控制）
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
    std::printf("；等待上限 %zu us，轮询周期 %ld ms，刷新 %ld ms\n", timeout_us, period_ms, refresh_ms);
    std::printf("go_monitor: 收发方式: %s\n",
                pipelined ? "流水线（连发 N 条 → 按 ID 收 N 条）" : "逐台一问一答");
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
    motors.setPipelined(pipelined, std::chrono::microseconds(spacing_us));
    motors.start();
    imu.Start();
    long long refresh = 0;
    std::vector<uint64_t> seen_errors(states.size(), 0);
    std::vector<std::string> logged_errors;
    long long error_count = 0;
    auto next_render = std::chrono::steady_clock::now();
    auto next_plain = next_render + std::chrono::seconds(1);
    auto next_error_window = std::chrono::steady_clock::now();
    long printed_errors = 0, suppressed_errors = 0;
    while (!signals.requested())
    {
        states=motors.snapshot();

        // 1) 增量取错误：plain 立即打印，TUI 只在退出时汇总（避免破坏面板）
        bool got_error = false;
        for (size_t i = 0; i < states.size(); ++i)
        {
            const PortState &st = states[i];
            if (st.errorSeq <= seen_errors[i]) continue;
            size_t n = static_cast<size_t>(st.errorSeq - seen_errors[i]);
            if (n > st.errors.size()) n = st.errors.size();
            for (size_t k = st.errors.size() - n; k < st.errors.size(); ++k)
            {
                ++error_count; got_error = true;
                const std::string line = st.device + " " + st.errors[k];
                logged_errors.push_back(line);
                if (logged_errors.size() > 200) logged_errors.erase(logged_errors.begin());
                if (plain && !quiet) {
                    const auto now_t = std::chrono::steady_clock::now();
                    if (now_t >= next_error_window) { next_error_window = now_t + std::chrono::seconds(1); printed_errors = 0; }
                    if (printed_errors < error_lines_per_s) { ++printed_errors; std::printf("[错误] %s\n", line.c_str()); std::fflush(stdout); }
                    else ++suppressed_errors;
                }
            }
            seen_errors[i] = st.errorSeq;
        }
        if (strict && got_error)
        {
            std::printf("go_monitor: --strict：检测到收发错误，退出。\n");
            motors.stop(); imu.Stop();
            if (!plain) display.Restore();
            return 2;
        }

        // 2) 显示：plain 每秒一行统计；否则按 -r 刷新面板
        const auto now = std::chrono::steady_clock::now();
        if (plain)
        {
            if (now >= next_plain)
            {
                for (const PortState &st : states)
                {
                    std::printf("%-22s 轮 %lld | 发 %.2fms 收 %.2fms | 收齐 %d/%d 缺 %d 重复 %d | 坏 %zu 串 %zu | 错轮 %lld\n",
                                st.device.c_str(), st.sweeps, st.sendMs, st.replyMs,
                                st.roundGot, st.roundSent, st.roundMissing, st.roundDuplicate,
                                st.badFrames, st.strayFrames, st.errorRounds);
                }
                const auto imu_snap = imu.Snapshot();
                long long imu_age_ms = -1;
                if (imu_snap.data.receive_time_ns != 0) {
                    const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count();
                    imu_age_ms = (now_ns - static_cast<long long>(imu_snap.data.receive_time_ns)) / 1000000;
                }
                std::printf("IMU age=%lldms hz=%.0f valid=%d | 错误累计 %lld 条%s\n",
                            imu_age_ms, imu_snap.receive_hz, imu_snap.received ? 1 : 0, error_count,
                            suppressed_errors ? "（本秒有错误被限流省略，见 --error-lines）" : "");
                suppressed_errors = 0;
                std::fflush(stdout);
                next_plain = now + std::chrono::seconds(1);
            }
        }
        else if (now >= next_render)
        {
            go_display::RenderMonitor(states, ++refresh, display, imu.Snapshot(), imu_config, timeout_us);
            next_render = now + std::chrono::milliseconds(refresh_ms);
        }
        // 分段小睡：既让错误尽快出现，也让 Ctrl+C 迅速响应
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    motors.stop();
    states=motors.snapshot();
    imu.Stop();
    if (!plain) display.Restore();

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
        if (pipelined)
            std::printf(" | 收齐 %d/%d 缺 %d 重复 %d | 发/收 %.2f/%.2fms | 错轮 %lld",
                        state.roundGot, state.roundSent, state.roundMissing, state.roundDuplicate,
                        state.sendMs, state.replyMs, state.errorRounds);
        if (state.timeoutTotal || state.errorRounds)
            std::printf("   <- 收发有问题，明细见下");
        std::printf("\n");
    }
    if (!logged_errors.empty())
    {
        std::printf("\ngo_monitor: 本次共 %lld 条收发错误，最近 %zu 条：\n",
                    error_count, logged_errors.size() < 10 ? logged_errors.size() : size_t(10));
        const size_t start = logged_errors.size() > 10 ? logged_errors.size() - 10 : 0;
        for (size_t i = start; i < logged_errors.size(); ++i)
            std::printf("  %s\n", logged_errors[i].c_str());
    }
    else std::printf("\ngo_monitor: 本次没有收发错误。\n");
    std::fflush(stdout);
    return 0;
}

catch(const std::exception& e) {std::fprintf(stderr,"monitor error: %s\n",e.what());return 2;}
