// ============================================================================
//  test_loop.cpp —— 验证 go::Loop（定周期线程）
//
//  覆盖：节拍是否真的按周期走、stop() 能否立刻打断睡眠、
//        能否 start → stop → start、body 抛异常时是否兜住并自停、
//        非法参数是否拒绝。
// ============================================================================
#include "go_loop.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <thread>

namespace
{

int failures = 0;

#define CHECK(cond)                                                                            \
    do                                                                                         \
    {                                                                                          \
        if (!(cond))                                                                           \
        {                                                                                      \
            std::fprintf(stderr, "  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);             \
            ++failures;                                                                        \
        }                                                                                      \
    } while (0)

using Clock = std::chrono::steady_clock;
using Ms = std::chrono::milliseconds;

long long elapsedMs(Clock::time_point from)
{
    return std::chrono::duration_cast<Ms>(Clock::now() - from).count();
}

// 等到 pred() 为真，或超时；返回是否等到
template <typename Pred>
bool waitFor(Pred pred, Ms timeout)
{
    const auto deadline = Clock::now() + timeout;
    while (Clock::now() < deadline)
    {
        if (pred())
            return true;
        std::this_thread::sleep_for(Ms(1));
    }
    return pred();
}

void testPeriod()
{
    std::printf("[1] 节拍：2 ms × 200 轮\n");
    std::atomic<int> body_calls{0};
    go::Loop loop("tick", std::chrono::microseconds(2000), [&body_calls] { ++body_calls; });

    const auto start = Clock::now();
    CHECK(loop.start());
    CHECK(loop.running());
    CHECK(waitFor([&loop] { return loop.tickCount() >= 200; }, Ms(5000)));
    const long long ms = elapsedMs(start);
    loop.stop();
    loop.join();

    // 每轮至少等一个周期，所以 200 轮不可能快于 200×2ms=400ms
    CHECK(ms >= 350);
    // 上限只用来兜「睡过头」这种明显错误：断言的是行为，不是这台机器有多快，
    // 所以给足余量（编译后机器很忙时也不该红）
    CHECK(ms <= 5000);
    CHECK(loop.tickCount() >= 200);
    CHECK(body_calls.load() == static_cast<int>(loop.tickCount()));
    CHECK(!loop.running());
    CHECK(loop.lastError().empty());

    std::printf("    200 轮用了 %lld ms（期望约 400 ms），超期 %llu 轮，单轮 body %lld us\n",
                ms, static_cast<unsigned long long>(loop.lateCount()), loop.lastBodyMicros());
}

void testStopInterruptsSleep()
{
    std::printf("[2] stop() 立刻打断长周期睡眠（周期 2000 ms，跑 30 ms 后停）\n");
    // 周期故意取很长：如果 stop() 不能打断睡眠，这里就得等满一个周期（约 1970 ms）。
    go::Loop loop("slow", Ms(2000), [] {});
    CHECK(loop.start());
    std::this_thread::sleep_for(Ms(30));

    const auto from = Clock::now();
    loop.stop();
    loop.join();
    const long long stop_ms = elapsedMs(from);

    // 若睡眠不可打断，这里会是 ~1970 ms
    CHECK(stop_ms < 500);
    CHECK(loop.tickCount() <= 1);
    CHECK(!loop.running());
    std::printf("    stop+join 用了 %lld ms（不可打断的话约 1970 ms）\n", stop_ms);
}

void testRestart()
{
    std::printf("[3] start → stop → start\n");
    go::Loop loop("restart", Ms(1), [] {});

    CHECK(loop.start());
    CHECK(waitFor([&loop] { return loop.tickCount() >= 5; }, Ms(2000)));
    loop.stop();
    loop.join();
    CHECK(!loop.running());
    CHECK(loop.start()); // 停掉并 join 之后可以重新启动，计数清零重新算
    CHECK(waitFor([&loop] { return loop.tickCount() >= 5; }, Ms(2000)));
    // start() 会把计数清零，所以这里数的是第二轮
    CHECK(loop.tickCount() >= 5);
    loop.stop();
    loop.join();
    CHECK(!loop.running());
    std::printf("    第二轮跑了 %llu 轮\n",
                static_cast<unsigned long long>(loop.tickCount()));
}

void testBodyThrows()
{
    std::printf("[4] body 抛异常：要兜住、要自停、不能 terminate\n");
    int calls = 0;
    go::Loop loop("thrower", Ms(1), [&calls] {
        ++calls;
        throw std::runtime_error("body 故意抛的");
    });

    CHECK(loop.start());
    // 周期 1 ms，若没有自停，50 ms 内会被调用几十次
    std::this_thread::sleep_for(Ms(50));
    CHECK(!loop.running());
    CHECK(calls == 1);
    CHECK(!loop.lastError().empty());
    loop.join();
    std::printf("    body 只被调用 %d 次，已捕获 \"%s\"\n", calls, loop.lastError().c_str());
}

void testExitHook()
{
    std::printf("[6] 退出钩子：三种退出路径都恰好跑一次\n");

    // 正常 stop
    int exit_calls = 0;
    go::Loop loop("hook", Ms(1), [] {}, [&exit_calls] { ++exit_calls; });
    CHECK(loop.start());
    CHECK(waitFor([&loop] { return loop.tickCount() >= 3; }, Ms(2000)));
    CHECK(exit_calls == 0); // 还在跑，钩子不该被调用
    loop.stop();
    loop.join();
    CHECK(exit_calls == 1);

    // body 抛异常而自停
    int exit_calls2 = 0;
    go::Loop loop2("hook2", Ms(1), [] { throw std::runtime_error("x"); },
                   [&exit_calls2] { ++exit_calls2; });
    CHECK(loop2.start());
    std::this_thread::sleep_for(Ms(30));
    loop2.join();
    CHECK(exit_calls2 == 1);
    CHECK(!loop2.lastError().empty());

    // 钩子里抛异常不能把进程带走
    int exit_calls3 = 0;
    go::Loop loop3("hook3", Ms(1), [] {}, [&exit_calls3] {
        ++exit_calls3;
        throw std::runtime_error("钩子也抛");
    });
    CHECK(loop3.start());
    std::this_thread::sleep_for(Ms(20));
    loop3.stop();
    loop3.join();
    CHECK(exit_calls3 == 1);
    CHECK(loop3.lastError().find("onExit") != std::string::npos);
    std::printf("    正常停止/body 异常/钩子自身抛异常：各调用一次，进程存活\n");
}

void testStatistics()
{
    std::printf("[7] 统计量：慢回调记进 body_max，周期受限时 gap_max 很小\n");

    // (a) 回调本身很慢（睡 5ms），period 只有 1ms → body 与 interval 都该反映出来
    {
        go::Loop loop("slow", Ms(1), [] { std::this_thread::sleep_for(Ms(5)); });
        CHECK(loop.start());
        CHECK(waitFor([&loop] { return loop.tickCount() >= 6; }, Ms(3000)));
        loop.stop();
        loop.join();
        CHECK(loop.maxBodyMicros() >= 5000);
        CHECK(loop.maxIntervalMicros() >= 5000);
        CHECK(loop.minIntervalMicros() >= 5000);
        // 注意：这里**不能**断言 maxInterval >= maxBody。间隔统计从第 2 轮才开始
        //（第 1 轮没有"上一轮"），而 body 统计包含第 1 轮和最后一轮——最后一轮的 body
        // 没有被任何间隔覆盖，所以两者不可比。曾经写过这条断言，在机器忙时偶发失败。
        CHECK(loop.maxIntervalMicros() >= loop.minIntervalMicros());
        std::printf("    body_max=%lldus interval=%lld~%lldus gap_max=%lldus"
                    "（回调睡 5ms，period 1ms）\n",
                    loop.maxBodyMicros(), loop.minIntervalMicros(), loop.maxIntervalMicros(),
                    loop.maxGapMicros());
    }

    // (b) 回调很快、由 period 主导 → interval 应该贴着 period，gap（调度延迟）很小
    {
        go::Loop loop("periodic", Ms(20), [] {});
        CHECK(loop.start());
        CHECK(waitFor([&loop] { return loop.tickCount() >= 6; }, Ms(3000)));
        loop.stop();
        loop.join();
        CHECK(loop.maxBodyMicros() < 2000);           // 回调本身很快
        CHECK(loop.minIntervalMicros() >= 19000);     // 间隔就是 period
        // 上限只兜"明显坏掉"；机器忙时唤醒会晚一些，不该因此判失败
        CHECK(loop.maxIntervalMicros() <= 200000);
        CHECK(loop.maxGapMicros() < 200000);
        std::printf("    body_max=%lldus interval=%lld~%lldus gap_max=%lldus"
                    "（回调立刻返回，period 20ms）\n",
                    loop.maxBodyMicros(), loop.minIntervalMicros(), loop.maxIntervalMicros(),
                    loop.maxGapMicros());
    }
}

void testResetStats()
{
    std::printf("[8] resetStats()：把统计量清零（不动运行状态与 lastError）\n");
    go::Loop loop("stats", Ms(1), [] {});
    CHECK(loop.start());
    CHECK(waitFor([&loop] { return loop.tickCount() >= 5; }, Ms(2000)));
    CHECK(loop.maxIntervalMicros() > 0); // 已经采到样本
    loop.resetStats();
    CHECK(loop.tickCount() == 0);
    CHECK(loop.lastBodyMicros() == 0);
    CHECK(loop.maxBodyMicros() == 0);
    CHECK(loop.minIntervalMicros() == 0);
    CHECK(loop.maxIntervalMicros() == 0);
    CHECK(loop.maxGapMicros() == 0);
    CHECK(loop.running());            // 线程还在跑
    CHECK(loop.lastError().empty());  // 错误信息不被清掉（这里本来就没有）
    loop.stop();
    loop.join();
    std::printf("    清零后线程仍在运行，统计从 0 重新开始\n");
}

void testInvalidArguments()
{
    std::printf("[5] 非法参数\n");
    bool threw = false;
    try
    {
        go::Loop bad("bad", std::chrono::microseconds(0), [] {});
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try
    {
        go::Loop bad("bad", Ms(1), nullptr);
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    CHECK(threw);
    std::printf("    周期 0 与空回调都被拒绝\n");
}

} // namespace

int main()
{
    std::printf("test_loop: go::Loop 定周期线程\n");
    testPeriod();
    testStopInterruptsSleep();
    testRestart();
    testBodyThrows();
    testInvalidArguments();
    testExitHook();
    testStatistics();
    testResetStats();

    if (failures == 0)
    {
        std::printf("test_loop: 全部通过\n");
        return 0;
    }
    std::fprintf(stderr, "test_loop: %d 项失败\n", failures);
    return 1;
}
