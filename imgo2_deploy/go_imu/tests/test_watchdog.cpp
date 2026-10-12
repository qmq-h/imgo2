// ============================================================================
//  test_watchdog.cpp —— 上层失联看门狗
//
//  和 test_motor_fake.cpp 一样用“链接期替换 SerialPort”的方式拿到确定性的字节传输：
//  不碰硬件、不依赖真实串口时序，而且能逐帧检查总线到底发了什么。
//
//  覆盖：
//    [1] 持续提交目标时看门狗不误触发
//    [2] 上层停手后总线主动卸力，而且**不是**故障（faulted() 仍为 false）
//    [3] 上层恢复提交后自动回到 PD，不需要清故障
//    [4] 对照：关掉看门狗会落到库的“命令过期”路径，被锁成需要人工恢复的故障
//    [5] 配置校验：同一设备出现在两条总线、watchdog 与 command timeout 的关系
// ============================================================================
#include "go_motor_array.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

#define CHECK(x)                                                                               \
    do                                                                                         \
    {                                                                                          \
        if (!(x))                                                                              \
            throw std::runtime_error(std::string("CHECK failed: ") + #x);                      \
    } while (0)

#include "fake_serial.h"

static go::Targets targets3(float q)
{
    go::Targets t;
    for (int i = 0; i < 3; ++i)
    {
        t.kp.push_back(25.0f);
        t.kd.push_back(0.5f);
        t.q.push_back(q);
        t.dq.push_back(0.0f);
        t.tau.push_back(0.0f);
    }
    return t;
}

static void sleepMs(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

static std::unique_ptr<go::GoMotorArray> makeArray(std::chrono::milliseconds watchdog,
                                                   size_t command_timeout_ms)
{
    fake_serial::clear();
    go::RobotConfig config;
    config.buses = {{"fake0", {0, 1, 2}, {}}};
    config.command_timeout_ms = command_timeout_ms;
    config.period = std::chrono::milliseconds(2);
    config.watchdog = watchdog;
    return std::unique_ptr<go::GoMotorArray>(new go::GoMotorArray(config));
}

// ---- 用例 -----------------------------------------------------------------

static void testNoFalseTrigger()
{
    std::cout << "[1] 持续提交目标：看门狗不误触发\n";
    auto multi = makeArray(std::chrono::milliseconds(30), 500);
    std::string error;
    CHECK(multi->start(error));

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
    while (std::chrono::steady_clock::now() < deadline)
    {
        CHECK(multi->setTargets(targets3(0.5f)));
        sleepMs(5);
    }

    const go::State st = multi->state();
    CHECK(!st.faulted);
    CHECK(!st.stale);
    CHECK(st.targetAgeMs < 30);
    CHECK(st.cycleCount > 10);
    CHECK(!fake_serial::isZeroCommand(fake_serial::sent().back()));
    CHECK(std::fabs(fake_serial::commandQ(fake_serial::sent().back()) - 0.5) < 0.01);
    std::cout << "    跑了 " << st.cycleCount << " 轮，目标年龄 " << st.targetAgeMs
              << " ms，未触发\n";
    multi->stop();
}

static void testWatchdogUnloads()
{
    std::cout << "[2] 上层停手：主动卸力，且不是故障\n";
    auto multi = makeArray(std::chrono::milliseconds(30), 500);
    std::string error;
    CHECK(multi->start(error));

    // 先持续提交一小会儿，让 PD 稳稳挂着（否则单次提交后 30 ms 就会被看门狗收走）
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    while (std::chrono::steady_clock::now() < until)
    {
        CHECK(multi->setTargets(targets3(0.5f)));
        sleepMs(5);
    }
    CHECK(!multi->state().stale);
    CHECK(!fake_serial::isZeroCommand(fake_serial::sent().back()));

    const size_t mark = fake_serial::sent().size();
    sleepMs(150); // 期间一次都不提交

    const go::State st = multi->state();
    CHECK(st.stale);
    CHECK(!st.faulted);        // ← 关键：走的是看门狗，不是库的“命令过期”故障
    CHECK(st.error.empty());   // 既然不是故障，就没有故障原因
    CHECK(st.targetAgeMs >= 30);
    // 停手之后，非零目标会一直持续到看门狗到期（约 30 ms），此后必须全是零指令。
    size_t last_pd = mark;
    for (size_t i = mark; i < fake_serial::sent().size(); ++i)
        if (!fake_serial::isZeroCommand(fake_serial::sent()[i]))
            last_pd = i;
    CHECK(fake_serial::sent().size() > last_pd + 1);
    for (size_t i = last_pd + 1; i < fake_serial::sent().size(); ++i)
        CHECK(fake_serial::isZeroCommand(fake_serial::sent()[i]));
    CHECK(fake_serial::sent().size() - (last_pd + 1) >= 10); // 之后确实在持续发零指令（卸力），不是偶然一帧
    std::cout << "    停手 " << st.targetAgeMs << " ms 后：stale=1 faulted=0；最后 " 
              << (fake_serial::sent().size() - (last_pd + 1)) << " 帧全为零指令\n";
    multi->stop();
}

static void testRecoversAutomatically()
{
    std::cout << "[3] 上层恢复提交：自动回到 PD，不需要清故障\n";
    auto multi = makeArray(std::chrono::milliseconds(30), 500);
    std::string error;
    CHECK(multi->start(error));
    CHECK(multi->setTargets(targets3(0.5f)));
    sleepMs(150); // 先触发看门狗
    CHECK(multi->state().stale);

    // 恢复：持续提交（只提交一次的话，30 ms 后又会正确地被看门狗收走）
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    while (std::chrono::steady_clock::now() < until)
    {
        CHECK(multi->setTargets(targets3(0.7f)));
        sleepMs(5);
    }

    const go::State st = multi->state();
    CHECK(!st.stale);
    CHECK(!st.faulted);
    CHECK(!fake_serial::isZeroCommand(fake_serial::sent().back()));
    CHECK(std::fabs(fake_serial::commandQ(fake_serial::sent().back()) - 0.7) < 0.01);
    std::cout << "    恢复后最新帧的目标位置 = " << fake_serial::commandQ(fake_serial::sent().back()) << " rad（期望 0.7）\n";
    multi->stop();
}

static void testLibraryFaultContrast()
{
    std::cout << "[4] 对照：关掉看门狗 → 库的“命令过期”把总线锁成故障\n";
    auto multi = makeArray(std::chrono::milliseconds(0), 50); // watchdog=0（关闭），超时 50 ms
    std::string error;
    CHECK(multi->start(error));
    CHECK(multi->setTargets(targets3(0.5f)));
    sleepMs(250);

    const go::State st = multi->state();
    CHECK(st.faulted);  // 需要显式清故障才能恢复
    CHECK(!st.stale);
    CHECK(!st.error.empty());
    std::cout << "    faulted=1，原因：" << st.error << "\n";
    multi->stop();
}

static void testValidation()
{
    std::cout << "[5] 配置校验\n";

    // 同一个设备路径出现在两条总线：会变成“两条线程抢一个串口”，必须在 start() 就拦住
    go::RobotConfig dup_config;
    dup_config.buses = {{"fake0", {0, 1, 2}, {}}, {"fake0", {0, 1, 2}, {}}};
    go::GoMotorArray dup(dup_config);
    std::string error;
    CHECK(!dup.start(error));
    CHECK(error.find("more than one bus") != std::string::npos);
    std::cout << "    同一设备两条总线 → start 失败：" << error << "\n";

    // watchdog 必须短于 command timeout，否则会落到库的故障路径
    bool threw = false;
    try
    {
        go::RobotConfig bad_config;
        bad_config.buses = {{"fake0", {0}, {}}};
        bad_config.command_timeout_ms = 100;
        bad_config.watchdog = std::chrono::milliseconds(100); // >= 命令有效期，必须被拒绝
        go::GoMotorArray bad(bad_config);
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    CHECK(threw);

    // watchdog = 0 表示关闭，允许
    threw = false;
    try
    {
        go::RobotConfig off_config;
        off_config.buses = {{"fake0", {0}, {}}};
        off_config.command_timeout_ms = 100;
        off_config.watchdog = std::chrono::milliseconds(0); // 0 = 关闭看门狗，允许
        go::GoMotorArray off(off_config);
    }
    catch (const std::exception &)
    {
        threw = true;
    }
    CHECK(!threw);
    std::cout << "    watchdog=100(>=超时100) 被拒绝；watchdog=0（关闭）被接受\n";
}

int main()
{
    std::cout << "test_watchdog: 上层失联看门狗\n";
    try
    {
        testNoFalseTrigger();
        testWatchdogUnloads();
        testRecoversAutomatically();
        testLibraryFaultContrast();
        testValidation();
    }
    catch (const std::exception &e)
    {
        std::cerr << "test_watchdog: 失败：" << e.what() << "\n";
        return 1;
    }
    std::cout << "test_watchdog: 全部通过\n";
    return 0;
}
