// ============================================================================
//  stand_main.cpp —— 让机器人从趴下平滑站起来的程序
//
//  和 pd_demo_main.cpp 的区别只有一处：目标不再是“保持当前位置”，而是一条
//  **8 秒的五次多项式轨迹**，从实测启动位置平滑走到“站立姿态”。
//
//  站立姿态（相对 offset 零点，每条腿 3 个关节，顺序 = go_config.cpp 里 motor_ids 的顺序）：
//      髋关节   hip   =   0.00 rad   （不动）
//      大腿关节 thigh =  +0.87 rad
//      小腿关节 calf  =  -1.82 rad
//
//  速度曲线用 go::SmoothTrajectory（五次多项式）：
//      progress(τ) = 10τ³ − 15τ⁴ + 6τ⁵      τ = t / T
//  它的**两端速度与加速度都恰好为 0**，所以起步和停住都不会“顿”一下；
//  峰值速度出现在 T/2，等于 1.875 × |Δq| / T（某个关节的峰值可据此估算）。
//
//  除了“目标怎么来”，其余部分（线程、看门狗、限幅、日志、状态输出）与 demo 完全一样：
//  接线 / offset / 限幅 都在 config/real_config.h 的 makeDefaultConfig() 里，两个程序共用一份。
// ============================================================================
#include "go_log.h"
#include "go_motor_array.h"
#include "go_trajectory.h"

#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/select.h>
#include <thread>
#include <unistd.h>
#include <vector>
#ifdef GO_STAND_PTY_TEST
#include "pty_devices.h"
#endif

// 信号处理函数只写这一个变量。
static volatile std::sig_atomic_t stop = 0;
static void onSignal(int) { stop = 1; }

namespace
{

// 站立姿态：每条腿 3 个关节，顺序固定为 髋 / 大腿 / 小腿（与配置里 motor_ids 的顺序一致）。
// 角度都是**相对 offset 零点**的；髋关节是 0 表示“相对当前零点不动”。
constexpr size_t kJointsPerLeg = 3;
constexpr size_t kMaxLegs = 4;
constexpr double kStandDurationS = 8.0;

// ---------------------------------------------------------------------------
//  每条腿的站立姿态（行 = 腿，顺序与 config.buses 一致；列 = 髋 / 大腿 / 小腿）
//
//  ⚠️ 左右腿要不要**反号**，取决于电机是怎么装的，不能只凭“机器人是镜像的”就断定：
//     · 左右两侧电机**同向**安装 → 同一个角度值在两侧代表相同的物理转动方向，
//       四条腿用同一组数（下面现在就是这样）；
//     · 左右两侧**镜像**安装 → 想让两条腿做出对称动作，符号必须相反，
//       例如右腿 {0, +0.87, -1.82}、左腿 {0, -0.87, +1.82}。
//    两种装法在四足上都很常见，而且**要按关节分别判断**（很可能髋要反、大腿小腿不用）。
//
//  怎么实测：用 ./build/go_monitor —— 它一直发零指令（卸力），同时显示每台电机的反馈。
//  机械架空后用手慢慢搬动关节，看界面上的 q 往哪边变，就知道这台电机的正方向朝哪边；
//  哪台与“站起来”的方向相反，就把下面对应那一行的相应数字取负。
//
//  另外注意：改这里的姿态时，`config.limits` 里的 q_min/q_max 也要跟着对得上，
//  否则 Safety 会把目标夹住——启动时会打警告。
// ---------------------------------------------------------------------------
constexpr float kStandPose[kMaxLegs][kJointsPerLeg] = {
    {0.0f, 0.87f, -1.82f}, // 腿 0 —— config.buses[0]（例如右前）
    {0.0f, 0.87f, -1.82f}, // 腿 1 —— config.buses[1]
    {0.0f, 0.87f, -1.82f}, // 腿 2 —— config.buses[2]
    {0.0f, 0.87f, -1.82f}, // 腿 3 —— config.buses[3]
};

// 每条腿的关节数不对、或者腿数超过姿态表就退出：与其发一份错的目标，不如什么都不做。
bool checkShape(const go::RobotConfig &config)
{
    if (config.buses.size() > kMaxLegs)
    {
        std::fprintf(stderr, "stand: 配置里有 %zu 条总线，但站立姿态表只有 %zu 行\n",
                     config.buses.size(), kMaxLegs);
        return false;
    }
    for (const go::BusConfig &bus : config.buses)
        if (bus.motor_ids.size() != kJointsPerLeg)
        {
            std::fprintf(stderr,
                         "stand: 每条总线必须正好 %zu 个关节（髋/大腿/小腿），%s 有 %zu 个\n",
                         kJointsPerLeg, bus.device.c_str(), bus.motor_ids.size());
            return false;
        }
    return true;
}

// 按 config 的展开顺序铺开站立姿态（每条腿可以不同，见 kStandPose 的说明）。
std::vector<float> buildGoal(const go::RobotConfig &config)
{
    std::vector<float> goal;
    for (size_t leg = 0; leg < config.buses.size(); ++leg)
        for (size_t i = 0; i < config.buses[leg].motor_ids.size(); ++i)
            goal.push_back(kStandPose[leg][i]);
    return goal;
}

// 开跑之前先自查：目标会不会被 Safety 夹住、峰值速度会不会超过 dq_max。
// 被夹过的目标**不会报错**，只会“悄悄少走一点”——对站立动作来说很危险，所以宁可先提示。
void warnAboutLimits(const go::RobotConfig &config, const std::vector<float> &start,
                     const std::vector<float> &goal, double duration_s)
{
    if (config.limits.empty())
        return; // 没设限幅就没什么好查的
    for (size_t i = 0; i < goal.size() && i < config.limits.size(); ++i)
    {
        const go::JointLimits &limit = config.limits[i];
        if (goal[i] < limit.q_min || goal[i] > limit.q_max)
            std::fprintf(stderr,
                         "stand: 警告——关节 %zu 的目标 %+.2f 超出限幅 [%+.2f, %+.2f]，会被夹住\n", i,
                         double(goal[i]), double(limit.q_min), double(limit.q_max));
        const double peak = go::SmoothTrajectory::peakRate() *
                            std::fabs(double(goal[i]) - double(start[i])) / duration_s;
        if (peak > double(limit.dq_max))
            std::fprintf(stderr, "stand: 警告——关节 %zu 峰值速度 %.2f rad/s 超过 dq_max %.2f rad/s\n",
                         i, peak, double(limit.dq_max));
    }
}

} // namespace
// 起跑门：非阻塞地看一眼 stdin，读到字符 '0' 就返回 true（终端里要按回车才会送到）。
// 用 select + read 而不是 std::cin，是为了不阻塞控制循环 —— 等待期间必须继续喂看门狗。
static bool startKeyPressed()
{
    fd_set set;
    FD_ZERO(&set);
    FD_SET(STDIN_FILENO, &set);
    timeval timeout{0, 0};
    if (::select(STDIN_FILENO + 1, &set, nullptr, nullptr, &timeout) <= 0)
        return false;
    char c = 0;
    return ::read(STDIN_FILENO, &c, 1) == 1 && c == '0';
}

int RunStand()
{
    // ======================= 配置：只改这一段 =======================
    // 接线、offset、限幅都在 go::makeDefaultConfig() 里（config/real_config.cpp）。
    go::RobotConfig config = go::makeDefaultConfig();
    double duration_s = kStandDurationS;

#ifdef GO_STAND_PTY_TEST
    std::string test_error;
    if (!go_test::applyTestDevices(config, "GO_STAND_TEST_DEVICES", test_error))
    {
        std::fprintf(stderr, "%s\n", test_error.c_str());
        return 1;
    }
    if (std::getenv("GO_TEST_IDENTITY_FRAME"))
        go_test::useIdentityFrame(config); // 只有集成测试会设这个变量
    if (const char *value = std::getenv("GO_STAND_DURATION_S"))
        duration_s = std::atof(value); // 集成测试用它把 8 秒缩短
#endif
    // ===============================================================

    const float kp = 25.0f; // 输出侧 N·m/rad
    const float kd = 0.5f;  // 输出侧 N·m/(rad/s)
    const float tau_ff = 0.0f;

    // 上层节拍：1 kHz，只负责“确认目标”。总线节拍来自 config.period。
    const std::chrono::milliseconds upper_period(1);

    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);

    if (!checkShape(config))
        return 1;

    go::SmoothTrajectory trajectory(duration_s);

    go::GoMotorArray multi(config);
    std::string error;
    if (!multi.start(error))
    {
        std::fprintf(stderr, "start failed: %s\n", error.c_str());
        return 1;
    }

    // 起点用**实测启动位置**，而不是直接写 0：offset 若已过期（标定后关节被移动过），
    // 写 0 会让第一帧就跳过去；用实测位置则第一帧一定“保持现在这里”。
    const std::vector<float> q_start = multi.positions();
    const std::vector<float> q_goal = buildGoal(config);
    const size_t n = q_goal.size();

    warnAboutLimits(config, q_start, q_goal, duration_s);

    std::printf("stand: %zu 个关节，%.1f 秒平滑站起（两端速度与加速度为 0）\n", n, duration_s);
    for (size_t leg = 0; leg < config.buses.size(); ++leg)
        std::printf("stand:   关节 %2zu..%2zu（%s）  髋 %+.2f  大腿 %+.2f  小腿 %+.2f  rad\n",
                    leg * kJointsPerLeg, leg * kJointsPerLeg + kJointsPerLeg - 1,
                    config.buses[leg].device.c_str(),
                    double(q_goal[leg * kJointsPerLeg + 0]), double(q_goal[leg * kJointsPerLeg + 1]),
                    double(q_goal[leg * kJointsPerLeg + 2]));
    std::printf("stand:   峰值速度 = 1.875 × |Δq| / T，本次最大的关节约 %.3f rad/s\n",
                go::SmoothTrajectory::peakRate() *
                    [&] {
                        double biggest = 0.0;
                        for (size_t i = 0; i < n; ++i)
                            biggest = std::max(biggest, std::fabs(double(q_goal[i]) - double(q_start[i])));
                        return biggest / duration_s;
                    }());

    // 目标会每周期重算，这里先按起点填好，循环里再改。
    go::Targets target;
    target.kp.assign(n, kp);
    target.kd.assign(n, kd);
    target.q = q_start;
    target.dq.assign(n, 0.0f);
    target.tau.assign(n, tau_ff);

    // ======================= 日志：每次运行新建一个文件 =======================
    // 站起过程最需要对照“指令 vs 实际”，所以这次日志同时记 q_des*（目标）和 q*（反馈）。
    const std::chrono::milliseconds log_interval(20); // 50 Hz 一行
    const char *const log_dir = std::getenv("GO_STAND_LOG_DIR");
    go::Logger logger;
    {
        std::string log_error;
        if (logger.open(go::Logger::makeRunPath(log_dir ? log_dir : "log"), log_error))
        {
            logger.writeHeader(multi.motorLabels(), true);
            std::printf("log: %s（每 %lld ms 一行：t_s,q_des0..q_des%zu,q0..q%zu）\n",
                        logger.path().c_str(), static_cast<long long>(log_interval.count()),
                        n == 0 ? size_t(0) : n - 1, n == 0 ? size_t(0) : n - 1);
        }
        else
            std::printf("log: 本次不记录（%s）\n", log_error.c_str());
    }

    // ---- 起跑门：等用户输入 0 ----
    // 等待期间仍然每个周期确认"保持启动位置"，否则看门狗会把电机卸力
    //（那是"松掉"，不是"保持在启动姿态"）。8 秒轨迹从收到 0 那一刻才开始算。
    std::printf("stand: 已就位，保持在启动位置（%zu 个关节）。\n", n);
    std::printf("stand: 确认机械已架空/固定、周围无人后，输入 0 再回车开始站起（Ctrl+C 退出）。\n");
    std::fflush(stdout);

    bool triggered = false;
    auto started_at = std::chrono::steady_clock::now(); // 收到 0 时重置
    unsigned long long upper_ticks = 0;
    bool reported_done = false;
    int result = 0;

    const std::chrono::milliseconds status_interval(500);
    auto next_status = started_at + status_interval;
    auto next_log = started_at; // 第一轮就先记一行

    auto next = std::chrono::steady_clock::now();
    while (!stop)
    {
        const auto now = std::chrono::steady_clock::now();
        if (!triggered && startKeyPressed())
        {
            triggered = true;
            started_at = now; // 时间基准移到触发这一刻
            std::printf("stand: 收到 0，开始站起（%.1f 秒平滑到站立姿态）\n", duration_s);
            std::fflush(stdout);
        }
        // 还没触发时 elapsed = 0 → progress = 0 → 目标就是 q_start（保持不动）
        const double elapsed_s =
            triggered ? std::chrono::duration<double>(now - started_at).count() : 0.0;

        // ---- 算这一周期的目标：位置按平滑曲线插值，速度前馈取曲线的导数 ----
        const double progress = trajectory.progress(elapsed_s);
        const double rate = trajectory.rate(elapsed_s);
        for (size_t i = 0; i < n; ++i)
        {
            const double delta = double(q_goal[i]) - double(q_start[i]);
            target.q[i] = float(double(q_start[i]) + progress * delta);
            target.dq[i] = float(rate * delta);
        }

        // 非阻塞：把目标写进各总线的命令槽就返回，不在这里等总线跑完
        if (!multi.setTargets(target))
        {
            std::fprintf(stderr, "control failed: %s\n", multi.lastError().c_str());
            result = 2;
            break;
        }
        ++upper_ticks;

        if (multi.faulted())
        {
            std::fprintf(stderr, "control failed: %s\n", multi.lastError().c_str());
            result = 2;
            break;
        }

        if (!reported_done && trajectory.finished(elapsed_s))
        {
            reported_done = true;
            std::printf("stand: 已到达站立姿态（%.1f s），继续保持；Ctrl+C 退出会卸力\n",
                        elapsed_s);
            std::fflush(stdout);
        }

        if (now >= next_log)
        {
            logger.writeFeedback(elapsed_s, multi.feedback(), target.q);
            next_log = now + log_interval;
        }
        if (now >= next_status)
        {
            if (triggered && !reported_done)
                std::printf("stand: t=%5.2f / %.1f s   进度 %3.0f%%\n", elapsed_s, duration_s,
                            trajectory.progress(elapsed_s) * 100.0);
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
    const double elapsed_s = triggered
        ? std::chrono::duration<double>(std::chrono::steady_clock::now() - started_at).count()
        : 0.0;
    multi.stop();

    // 给测试（和调试）看的一组数，格式与 demo 一致，另外带上轨迹进度。
    std::printf("stand: upper ticks=%llu bus rounds=%llu elapsed=%.2fs progress=%.3f upper_hz=%.0f "
                "bus_hz=%.0f interval_min_us=%lld interval_max_us=%lld body_max_us=%lld gap_max_us=%lld\n",
                upper_ticks, static_cast<unsigned long long>(final_state.cycleCount), elapsed_s,
                trajectory.progress(elapsed_s), elapsed_s > 0.0 ? double(upper_ticks) / elapsed_s : 0.0,
                elapsed_s > 0.0 ? double(final_state.cycleCount) / elapsed_s : 0.0,
                stats.intervalMinUs, stats.intervalMaxUs, stats.bodyMaxUs, stats.gapMaxUs);
    std::fflush(stdout);
    return result;
}
