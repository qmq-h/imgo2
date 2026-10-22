/* ----------------------------------------------------------------------------
 * 用法
 *   ./go_stand [--kp N] [--kd N] [--duration 秒]
 *   --kp       位置增益 N·m/rad（默认 25.0，Safety 会夹到 kp_max=70）
 *   --kd       阻尼增益 N·m/(rad/s)（默认 0.5）
 *   --duration 站起轨迹时长（默认 8.0 秒）
 *   运行时按 0 + 回车开始站起；Ctrl+C 退出并卸力。
 *   首次真机建议小增益：./go_stand --kp 8 --kd 1 --duration 12
 * ---------------------------------------------------------------------------- */
#include "apps/runtime.h"
#include "go_motor_array.h"
#include "fsm/states/stand_state.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>
#ifdef GO_STAND_PTY_TEST
#include "pty_devices.h"
#endif

namespace
{
std::vector<float> buildStandGoal(const go::RobotConfig& config)
{
    if(config.buses.empty() || config.buses.size()>4)
        throw std::invalid_argument("stand requires 1..4 three-joint buses");
    std::vector<float> goal;
    for(const auto& bus:config.buses) {
        if(bus.motor_ids.size()!=3) throw std::invalid_argument("stand requires hip/thigh/shank on each bus");
        goal.insert(goal.end(),{0.f,.87f,-1.82f});
    }
    return goal;
}

go::Targets poseCommand(const std::vector<float>& q, const go_fsm::states::StandSettings& settings)
{
    go::Targets target;
    target.q = q;
    target.dq.assign(q.size(), 0);
    target.tau.assign(q.size(), settings.tau);
    target.kp.assign(q.size(), settings.kp);
    target.kd.assign(q.size(), settings.kd);
    return target;
}

std::vector<float> configuredStart(const go::MotorConfig& config)
{
    std::vector<float> q;
    for (const auto& bus : config.buses)
    {
        if (bus.start_pose_rad.size() != bus.motor_ids.size())
            throw std::invalid_argument("stand requires configured start pose for every joint");
        q.insert(q.end(), bus.start_pose_rad.begin(), bus.start_pose_rad.end());
    }
    return q;
}

double number(const std::string& text,const char* what)
{
    size_t used=0;double v=0;
    try { v=std::stod(text,&used); }
    catch(const std::exception&) { throw std::invalid_argument(std::string(what)+" requires a finite number"); }
    if(used!=text.size() || !std::isfinite(v)) throw std::invalid_argument(std::string(what)+" requires a finite number");
    return v;
}

void usage(const char* argv0)
{
    const go_fsm::states::StandSettings defaults;
    std::printf(
        "go_stand —— 从配置的启动姿态按五次曲线站到 (0, 0.87, -1.82)，全程 PD 保持。\n"
        "\n"
        "用法: %s [--kp N] [--kd N] [--duration 秒] [--help]\n"
        "  --kp       位置增益 N·m/rad（默认 %.1f；会被 Safety 夹到 config 的 kp_max=%.1f）\n"
        "  --kd       阻尼增益 N·m/(rad/s)（默认 %.1f）\n"
        "  --duration 站起轨迹时长，秒（默认 %.1f）\n"
        "运行时按 0 再回车开始站起；Ctrl+C 退出并卸力。\n"
        "首次真机测试建议先用小增益，例如: %s --kp 8 --kd 1 --duration 12\n",
        argv0, double(defaults.kp), double(MOTOR_LIMIT_KP_MAX), double(defaults.kd),
        defaults.duration_s, argv0);
}
}

int main(int argc, char** argv) try
{
    go::MotorConfig config;
    go_fsm::states::StandSettings settings; // 默认值在 stand_state.h 的 StandSettings 里。
    bool duration_from_cli = false;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--help") { usage(argv[0]); return 0; }
        if (arg != "--kp" && arg != "--kd" && arg != "--duration")
            throw std::invalid_argument("unknown/missing argument: " + arg);
        if (i + 1 >= argc) throw std::invalid_argument(arg + " requires a value");
        const std::string value = argv[++i];
        if (arg == "--kp")
        {
            const double v = number(value, "kp");
            if (v < 0) throw std::invalid_argument("kp must be >= 0");
            settings.kp = float(v);
        }
        else if (arg == "--kd")
        {
            const double v = number(value, "kd");
            if (v < 0) throw std::invalid_argument("kd must be >= 0");
            settings.kd = float(v);
        }
        else if (arg == "--duration")
        {
            const double v = number(value, "duration");
            if (v <= 0) throw std::invalid_argument("duration must be > 0");
            settings.duration_s = v; duration_from_cli = true;
        }
        else throw std::invalid_argument("unknown/missing argument: " + arg);
    }
#ifdef GO_STAND_PTY_TEST
    std::string test_error;
    if (!go_test::applyTestDevices(config, "GO_STAND_TEST_DEVICES", test_error))
    {
        std::fprintf(stderr, "%s\n", test_error.c_str());
        return 1;
    }
    if (std::getenv("GO_TEST_IDENTITY_FRAME"))
        go_test::useIdentityFrame(config);
    if (!duration_from_cli)
        if (const char* value = std::getenv("GO_STAND_DURATION_S"))
            settings.duration_s = std::atof(value);
#endif
    const auto goal = buildStandGoal(config);
    std::string error;
    // 配置非法/轨迹超速时在打开串口之前拒绝，不能靠裁剪改变站起动作。
    if (!go::Safety::trajectoryValid(poseCommand(configuredStart(config), settings),
                                    poseCommand(goal, settings), settings.duration_s,
                                    config.limits, error))
    {
        std::fprintf(stderr, "stand failed: %s\n", error.c_str());
        return 1;
    }

    go_app::StopSignals signals;
    go::GoMotorArray motors(config);
    if (!motors.start(error))
    {
        std::fprintf(stderr, "start failed: %s\n", error.c_str());
        return 1;
    }
    const auto start = motors.positions();
    if (!go::Safety::trajectoryValid(poseCommand(start, settings), poseCommand(goal, settings),
                                    settings.duration_s, config.limits, error))
    {
        std::fprintf(stderr, "stand failed: %s\n", error.c_str());
        return 1;
    }

    auto owned = std::unique_ptr<go_fsm::states::StandState>(new go_fsm::states::StandState(start, goal, settings));
    auto* stand = owned.get();
    go_fsm::FSM fsm(std::move(owned));
    fsm.start(go_fsm::Clock::now());
    go_fsm::StandInput input;
    std::printf("stand: kp=%.2f kd=%.2f 时长=%.1fs；保持启动位置，输入0再回车开始站起，Ctrl+C退出会卸力。\n",
                settings.kp, settings.kd, settings.duration_s);
    std::fflush(stdout);

    auto next = go_fsm::Clock::now();
    bool reported_done = false;
    int result = 0;
    while (!signals.requested())
    {
        char c = 0;
        const bool trigger = !stand->triggered() && go_app::readChar(c) && input.feed(c);
        const auto event = trigger ? go_fsm::Event::Stand : go_fsm::Event::None;
        const auto& target = fsm.update(go_fsm::Clock::now(), event);
        if (trigger)
        {
            std::printf("stand: 开始站起（%.1f秒）。\n", settings.duration_s);
            std::fflush(stdout);
        }
        if (!motors.setTargets(target) || motors.faulted())
        {
            std::fprintf(stderr, "control failed: %s\n", motors.lastError().c_str());
            result = 2;
            break;
        }
        if (stand->finished() && !reported_done)
        {
            reported_done = true;
            std::puts("stand: 已到达站立姿态，继续保持。");
            std::fflush(stdout);
        }
        next += std::chrono::milliseconds(1);
        const auto now = go_fsm::Clock::now();
        if (next < now)
            next = now;
        std::this_thread::sleep_until(next);
    }
    fsm.stop();
    motors.stop();
    return result;
}
catch (const std::exception& e)
{
    std::fprintf(stderr, "stand error: %s\n", e.what());
    return 1;
}
