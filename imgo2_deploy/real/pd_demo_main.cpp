/* ----------------------------------------------------------------------------
 * 用法
 *   ./go_pd_demo
 *   无参数：保持启动姿态（kp=25, kd=0.5），不发轨迹、不跑策略；Ctrl+C 退出并卸力。
 *   设备与 ID 来自 config/real_config.cpp 的 4 条总线。
 * ---------------------------------------------------------------------------- */
#include "apps/runtime.h"
#include "go_motor_array.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#ifdef GO_DEMO_PTY_TEST
#include "pty_devices.h"
#endif

// 保持当前位置；串口、反馈、看门狗和限幅由Motor统一负责。
int main() try
{
    go::MotorConfig config;
#ifdef GO_DEMO_PTY_TEST
    std::string test_error;
    if (!go_test::applyTestDevices(config, "GO_DEMO_TEST_DEVICES", test_error))
    {
        std::fprintf(stderr, "%s\n", test_error.c_str());
        return 1;
    }
    if (std::getenv("GO_TEST_IDENTITY_FRAME"))
        go_test::useIdentityFrame(config);
#endif
    go_app::StopSignals signals;
    go::GoMotorArray motors(config);
    std::string error;
    if (!motors.start(error))
    {
        std::fprintf(stderr, "start failed: %s\n", error.c_str());
        return 1;
    }

    go::Targets hold;
    const auto n = motors.motorCount();
    hold.q = motors.positions();
    hold.dq.assign(n, 0);
    hold.tau.assign(n, 0);
    hold.kp.assign(n, 25);
    hold.kd.assign(n, 0.5f);
    std::puts("demo: 保持启动位置；Ctrl+C退出会卸力。");
    std::fflush(stdout);

    auto next = std::chrono::steady_clock::now();
    int result = 0;
    while (!signals.requested())
    {
        if (!motors.setTargets(hold) || motors.faulted())
        {
            std::fprintf(stderr, "control failed: %s\n", motors.lastError().c_str());
            result = 2;
            break;
        }
        next += std::chrono::milliseconds(1);
        const auto now = std::chrono::steady_clock::now();
        if (next < now)
            next = now;
        std::this_thread::sleep_until(next);
    }
    motors.stop();
    return result;
}
catch (const std::exception& e)
{
    std::fprintf(stderr, "demo error: %s\n", e.what());
    return 1;
}
