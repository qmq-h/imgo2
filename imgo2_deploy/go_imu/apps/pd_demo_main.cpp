// ============================================================================
//  pd_demo_main.cpp —— 12 台电机（4 条总线 × 每路 3 台）的统一控制示例
//
//  线程与逐总线收发都在 go_motor_array.cpp / go_loop.cpp 里；**接线、offset、限幅**
//  在 config/real_config.h 的 makeDefaultConfig() 里（demo 与后续 deploy 共用一份）。
//  本文件只有配置、变量定义和主循环：用非阻塞的 setTargets() 把 12 台电机锁定在
//  启动位置（目标速度与前馈都为 0）。
//
//  两个节拍故意不同，用来演示上下层解耦：
//      上层   每 1 ms 确认一次目标（1 kHz）—— 只是“我还在，请保持”
//      每条总线 每 5 ms 跑一轮收发（200 Hz）—— 来自 config.period
//  上层再快也不会让总线变慢；上层偶尔慢一下也不会让总线停摆。
// ============================================================================
#include "go_log.h"
#include "go_motor_array.h"

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>
#ifdef GO_DEMO_PTY_TEST
#include "pty_devices.h" // 只在测试构建里需要（tests/ 只对测试目标加入 include 路径）
#endif

// 信号处理函数只写这一个变量。
static volatile std::sig_atomic_t stop = 0;
static void onSignal(int) { stop = 1; }

int RunPdDemo()
{
    // ======================= 配置：只改这一段 =======================
    // 接线、offset、限幅都在 go::makeDefaultConfig() 里（config/real_config.h）；
    // 改接线/改 offset 只需要改那一处，不会和 deploy 各写一遍。
    go::RobotConfig config = go::makeDefaultConfig();
#ifdef GO_DEMO_PTY_TEST
    std::string test_error;
    if (!go_test::applyTestDevices(config, "GO_DEMO_TEST_DEVICES", test_error))
    {
        std::fprintf(stderr, "%s\n", test_error.c_str());
        return 1;
    }
    if (std::getenv("GO_TEST_IDENTITY_FRAME"))
        go_test::useIdentityFrame(config); // 只有集成测试会设这个变量
#endif
    // ===============================================================

    const float kp = 25.0f; // 输出侧 N·m/rad
    const float kd = 0.5f;  // 输出侧 N·m/(rad/s)
    const float tau_ff = 0.0f;

    // 上层节拍（本文件自己的循环）：1 kHz，只负责“确认目标”。
    // 总线节拍来自 config.period（默认 5 ms = 200 Hz）。
    const std::chrono::milliseconds upper_period(1);

    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);

    go::GoMotorArray multi(config);
    std::string error;
    if (!multi.start(error))
    {
        std::fprintf(stderr, "start failed: %s\n", error.c_str());
        return 1;
    }

    // 锁定在**实测启动位置**：以各台当前位置为目标，目标速度与前馈为 0。
    // 用实测位置而不是写 0（= offset 定义的逻辑零点），是为了在 offset 还没标定、
    // 或者标定后电机被移动过时，第一帧也只是“保持现在这里”，不会突然跳回标定点。
    go::Targets hold;
    hold.kp.assign(multi.motorCount(), kp);
    hold.kd.assign(multi.motorCount(), kd);
    hold.q = multi.positions();
    hold.dq.assign(multi.motorCount(), 0.0f);
    hold.tau.assign(multi.motorCount(), tau_ff);

    // ======================= 日志：每次运行新建一个文件 =======================
    // 目前只记 12 台电机的角度反馈（q），每行是 t_s,q0..qN（6 位小数）。
    // 目录默认是相对当前目录的 log/，可用环境变量 GO_DEMO_LOG_DIR 覆盖（集成测试用）。
    // 打不开日志不影响控制，只提示一句继续跑。
    const std::chrono::milliseconds log_interval(20); // 50 Hz 一行
    const char *const log_dir = std::getenv("GO_DEMO_LOG_DIR");
    go::Logger logger;
    {
        std::string log_error;
        if (logger.open(go::Logger::makeRunPath(log_dir ? log_dir : "log"), log_error))
        {
            logger.writeHeader(multi.motorLabels(), true); // true = 多写一组 q_des* 列
            std::printf("log: %s（每 %lld ms 一行：t_s,q0..q%zu）\n", logger.path().c_str(),
                        static_cast<long long>(log_interval.count()),
                        multi.motorCount() == 0 ? size_t(0) : multi.motorCount() - 1);
        }
        else
            std::printf("log: 本次不记录（%s）\n", log_error.c_str());
    }

    const auto started_at = std::chrono::steady_clock::now();
    unsigned long long upper_ticks = 0;
    int result = 0;

    // 实机调试用：每 500 ms 打一块状态（每条总线一行 + 每台电机一行 + 空行）。
    // 关于“限幅还没设”的提醒写在 config/real_config.cpp 的 TODO 里，不再占终端。
    // 集成测试只跑约 100 ms，所以这块不会出现在测试输出里。
    const std::chrono::milliseconds status_interval(500);
    auto next_status = started_at + status_interval;
    auto next_log = started_at; // 第一轮就先记一行

    auto next = std::chrono::steady_clock::now();
    while (!stop)
    {
        // 非阻塞：把 12 个目标写进各总线的命令槽就返回，不在这里等总线跑完。
        // 各总线会在自己的下一个周期采纳它。
        if (!multi.setTargets(hold))
        {
            std::fprintf(stderr, "control failed: %s\n", multi.lastError().c_str());
            result = 2;
            break;
        }
        ++upper_ticks;

        // 顺手看一眼总线有没有出事。faulted() 只读标志、不拷反馈，所以每轮查也不贵。
        // 注意这里查的是“真故障”；上层失联是另一回事，由底层看门狗处理（state().stale）。
        if (multi.faulted())
        {
            std::fprintf(stderr, "control failed: %s\n", multi.lastError().c_str());
            result = 2;
            break;
        }

        const auto now = std::chrono::steady_clock::now();
        if (now >= next_log)
        {
            logger.writeFeedback(std::chrono::duration<double>(now - started_at).count(),
                                 multi.feedback(), hold.q);
            next_log = now + log_interval;
        }
        if (now >= next_status)
        {
            std::fputs(multi.statusText().c_str(), stdout);
            std::fflush(stdout);
            next_status = now + status_interval;
        }

        next += upper_period;
        if (next < now)
            next = now; // 本轮超期时立即开始下一轮，不补发积压指令
        std::this_thread::sleep_until(next);
    }

    // 必须在 stop() 之前读：stop() 会把各总线的 Loop 对象销毁掉，之后就查不到了
    const go::State final_state = multi.state();
    const go::LoopStats stats = multi.loopStats();
    const double elapsed_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started_at).count();
    multi.stop();

    // 给测试（和调试）看的一组数：上层确认了多少次、总线一共跑了多少轮，
    // 以及循环质量的三个统计量（单位都是微秒，见 go::LoopStats 的说明）：
    //   interval_min/max —— 相邻两轮开始时刻的间隔范围 = 真实控制周期的波动
    //   body_max         —— 单轮收发最长耗时（串口/USB 的锅）
    //   gap_max          —— 醒来比计划最晚多少（调度器的锅）
    std::printf("demo: upper ticks=%llu bus rounds=%llu elapsed=%.2fs upper_hz=%.0f bus_hz=%.0f "
                "interval_min_us=%lld interval_max_us=%lld body_max_us=%lld gap_max_us=%lld\n",
                upper_ticks, static_cast<unsigned long long>(final_state.cycleCount), elapsed_s,
                elapsed_s > 0.0 ? double(upper_ticks) / elapsed_s : 0.0,
                elapsed_s > 0.0 ? double(final_state.cycleCount) / elapsed_s : 0.0,
                stats.intervalMinUs, stats.intervalMaxUs, stats.bodyMaxUs, stats.gapMaxUs);
    std::fflush(stdout);
    return result;
}
