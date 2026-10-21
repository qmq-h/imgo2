// ============================================================================
//  test_safety.cpp —— Safety 限幅层
//
//  两部分：
//    [A] 纯函数测试：clamp / approach / limitsValid。不需要串口，也不涉及线程。
//    [B] 端到端测试：真的把目标提交给 GoMotorArray，检查**发到线上的字节**已经被夹过。
//        用 tests/fake_serial.h 在链接期替换 SerialPort（见那个文件的说明）。
// ============================================================================
#include "go_safety.h"

#include "fake_serial.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#define CHECK(x)                                                                               \
    do                                                                                         \
    {                                                                                          \
        if (!(x))                                                                              \
            throw std::runtime_error(std::string("CHECK failed: ") + #x);                      \
    } while (0)

using Ms = std::chrono::milliseconds;

// ---- 小工具 ---------------------------------------------------------------

static go::Targets makeTargets(size_t count, float q, float kp = 25.0f, float kd = 0.5f,
                              float dq = 0.0f, float tau = 0.0f)
{
    go::Targets t;
    t.kp.assign(count, kp);
    t.kd.assign(count, kd);
    t.q.assign(count, q);
    t.dq.assign(count, dq);
    t.tau.assign(count, tau);
    return t;
}

// 注意：不能用 vector == 来判“目标没被改动”——NaN != NaN，含 NaN 的向量永远不相等。
// 所以这里用 NaN 感知的逐元素比较。
static bool sameValue(float a, float b)
{
    if (std::isnan(a) && std::isnan(b))
        return true;
    return a == b;
}

static bool sameVector(const std::vector<float> &a, const std::vector<float> &b)
{
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!sameValue(a[i], b[i]))
            return false;
    return true;
}

static bool sameTargets(const go::Targets &a, const go::Targets &b)
{
    return sameVector(a.q, b.q) && sameVector(a.dq, b.dq) && sameVector(a.tau, b.tau) &&
           sameVector(a.kp, b.kp) && sameVector(a.kd, b.kd);
}

static void sleepMs(int ms) { std::this_thread::sleep_for(Ms(ms)); }

// ---- [A] 纯函数 -----------------------------------------------------------

static void testClampByLimits()
{
    std::cout << "[A1] clamp：位置/速度/力矩/增益都被夹到边界\n";
    std::vector<go::JointLimits> limits(2);
    limits[0].q_min = -1.0f;
    limits[0].q_max = 1.0f;
    limits[0].dq_max = 2.0f;
    limits[0].tau_max = 3.0f;
    limits[0].kp_max = 30.0f;
    limits[0].kd_max = 1.0f;
    limits[1] = limits[0];

    go::Targets t = makeTargets(2, 5.0f, 100.0f, 9.0f, 50.0f, -80.0f);
    CHECK(go::Safety::clamp(t, limits) == go::ClampResult::Clamped);
    for (size_t i = 0; i < 2; ++i)
    {
        CHECK(t.q[i] == 1.0f);
        CHECK(t.dq[i] == 2.0f);
        CHECK(t.tau[i] == -3.0f);
        CHECK(t.kp[i] == 30.0f);
        CHECK(t.kd[i] == 1.0f);
    }
    std::cout << "    q=5→1, dq=50→2, tau=-80→-3, kp=100→30, kd=9→1\n";

    // 负位置夹到下限
    go::Targets low = makeTargets(2, -5.0f);
    CHECK(go::Safety::clamp(low, limits) == go::ClampResult::Clamped);
    CHECK(low.q[0] == -1.0f);

    // 负增益夹到 0
    go::Targets neg = makeTargets(2, 0.0f, -25.0f, -1.0f);
    CHECK(go::Safety::clamp(neg, limits) == go::ClampResult::Clamped);
    CHECK(neg.kp[0] == 0.0f && neg.kd[0] == 0.0f);
}

static void testClampPassThrough()
{
    std::cout << "[A2] clamp：正常值和「不限」都原样通过\n";
    // 限幅留空 = 全部不限
    go::Targets t = makeTargets(3, 100.0f, 1e6f, 1e6f, 1e6f, 1e6f);
    const go::Targets before = t;
    CHECK(go::Safety::clamp(t, {}) == go::ClampResult::Unchanged);
    CHECK(sameTargets(t, before));

    // 在范围内
    std::vector<go::JointLimits> limits(3);
    limits[0].q_min = -1.0f;
    limits[0].q_max = 1.0f;
    limits[1] = limits[0];
    limits[2] = limits[0];
    go::Targets ok = makeTargets(3, 0.5f);
    const go::Targets ok_before = ok;
    CHECK(go::Safety::clamp(ok, limits) == go::ClampResult::Unchanged);
    CHECK(sameTargets(ok, ok_before));
    std::cout << "    不限幅与范围内目标都未被修改\n";
}

static void testClampRejectsNonFinite()
{
    std::cout << "[A3] clamp：NaN / Inf 整份拒绝，而且不修改目标\n";
    std::vector<go::JointLimits> limits(1);
    limits[0].q_min = -1.0f;
    limits[0].q_max = 1.0f;

    const float nan = std::nanf("");
    const float inf = std::numeric_limits<float>::infinity();

    for (float bad : {nan, inf, -inf})
    {
        go::Targets t = makeTargets(1, bad);
        const go::Targets before = t;
        CHECK(go::Safety::clamp(t, limits) == go::ClampResult::Rejected);
        CHECK(sameTargets(t, before)); // 拒绝时必须一个字都不改
    }
    // NaN 出现在别的字段上同样要拒
    for (int field = 0; field < 4; ++field)
    {
        go::Targets t = makeTargets(1, 0.5f);
        if (field == 0)
            t.dq[0] = nan;
        else if (field == 1)
            t.tau[0] = nan;
        else if (field == 2)
            t.kp[0] = nan;
        else
            t.kd[0] = nan;
        CHECK(go::Safety::clamp(t, limits) == go::ClampResult::Rejected);
    }
    // 各 vector 长度不一致也拒
    go::Targets broken = makeTargets(2, 0.5f);
    broken.tau.pop_back();
    CHECK(go::Safety::clamp(broken, limits) == go::ClampResult::Rejected);
    std::cout << "    NaN/Inf（位置或别的字段）、长度不一致 → 全部 Rejected 且目标未变\n";
}

static void testApproach()
{
    std::cout << "[A4] approach：限速推进，不会一步跳过去\n";
    std::vector<go::JointLimits> limits(1);
    limits[0].q_min = -10.0f;
    limits[0].q_max = 10.0f;
    limits[0].dq_max = 0.5f; // 0.5 rad/s

    go::Targets current = makeTargets(1, 0.0f);
    const go::Targets desired = makeTargets(1, 1.0f, 25.0f, 0.5f);

    // dt = 10 ms → 每步最多 0.005 rad
    CHECK(!go::Safety::approach(current, desired, limits, Ms(10)));
    CHECK(std::fabs(current.q[0] - 0.005f) < 1e-6f);
    // 增益/速度/力矩立刻跟随（不需要斜坡）
    CHECK(current.kp[0] == 25.0f && current.kd[0] == 0.5f);

    // 再走 199 步应该正好到达（0.005 × 200 = 1.0）
    bool arrived = false;
    for (int i = 0; i < 250 && !arrived; ++i)
    {
        const float before = current.q[0];
        arrived = go::Safety::approach(current, desired, limits, Ms(10));
        CHECK(std::fabs(current.q[0] - before) <= 0.005f + 1e-6f); // 每步都不超速
        CHECK(current.q[0] <= 1.0f + 1e-6f);                       // 也不会冲过头
    }
    CHECK(arrived);
    CHECK(current.q[0] == 1.0f);
    std::cout << "    0→1 rad、限速 0.5 rad/s、dt=10ms：每步 0.005 rad，第 200 步到达\n";

    // 不限速时一步到位
    go::Targets jump = makeTargets(1, 0.0f);
    CHECK(go::Safety::approach(jump, desired, {}, Ms(10)));
    CHECK(jump.q[0] == 1.0f);

    // 非法输入
    go::Targets bad_dt = makeTargets(1, 0.0f);
    CHECK(!go::Safety::approach(bad_dt, desired, limits, Ms(0)));
    go::Targets size_mismatch = makeTargets(2, 0.0f);
    CHECK(!go::Safety::approach(size_mismatch, desired, limits, Ms(10)));
    std::cout << "    不限速一步到位；dt=0 与长度不一致被拒绝\n";
}

static void testLimitsValid()
{
    std::cout << "[A5] limitsValid：配置检查\n";
    std::string error;
    CHECK(go::Safety::limitsValid({}, 4, error)); // 空 = 不限

    std::vector<go::JointLimits> wrong_size(3);
    CHECK(!go::Safety::limitsValid(wrong_size, 4, error));
    CHECK(!error.empty());

    std::vector<go::JointLimits> inverted(1);
    inverted[0].q_min = 1.0f;
    inverted[0].q_max = -1.0f;
    CHECK(!go::Safety::limitsValid(inverted, 1, error));

    std::vector<go::JointLimits> negative(1);
    negative[0].dq_max = -1.0f;
    CHECK(!go::Safety::limitsValid(negative, 1, error));

    std::vector<go::JointLimits> nan_limit(1);
    nan_limit[0].q_min = std::nanf("");
    CHECK(!go::Safety::limitsValid(nan_limit, 1, error));

    std::vector<go::JointLimits> ok(1);
    ok[0].q_min = -1.0f;
    ok[0].q_max = 1.0f;
    CHECK(go::Safety::limitsValid(ok, 1, error));
    std::cout << "    空表通过；长度不符 / 上下限写反 / 负上限 / NaN 都被拒绝\n";
}

// ---- [B] 端到端 -----------------------------------------------------------

static go::RobotConfig makeConfig()
{
    go::RobotConfig config;
    config.limits.clear();
    config.buses = {{"fake0", {0, 1, 2}, {}}};
    config.period = Ms(2);
    config.command_timeout_ms = 500;
    config.watchdog = Ms(100);
    return config;
}

static void testWireIsClamped()
{
    std::cout << "[B1] 端到端：超限的目标发到线上已经被夹过\n";
    fake_serial::clear();

    go::RobotConfig config = makeConfig();
    config.limits.assign(3, go::JointLimits{});
    for (go::JointLimits &limit : config.limits)
    {
        limit.q_min = -1.0f;
        limit.q_max = 1.0f;
        limit.kp_max = 30.0f;
    }

    go::GoMotorArray multi(config);
    std::string error;
    CHECK(multi.start(error));
    fake_serial::clearSent(); // 丢掉启动探测那几帧

    // 故意给一个远超限幅的目标
    CHECK(multi.setTargets(makeTargets(3, 100.0f, 1e6f, 0.5f)));
    sleepMs(40);

    CHECK(!fake_serial::sent().empty());
    // 注意：clear() 之后、新目标被采纳之前，总线上发零指令是正常的（还没有活动目标）。
    // 所以只检查“非零的那些帧”，并且必须至少出现过一帧。
    bool saw_target = false;
    for (const auto &frame : fake_serial::sent())
    {
        if (fake_serial::isZeroCommand(frame))
            continue;
        saw_target = true;
        CHECK(std::fabs(fake_serial::commandQ(frame) - 1.0) < 0.01);  // 被夹到 q_max
        CHECK(std::fabs(fake_serial::commandKp(frame) - 30.0) < 0.5); // 被夹到 kp_max
    }
    CHECK(saw_target);
    CHECK(multi.state().clampedCount > 0); // 上层能看出自己顶到限幅了
    std::cout << "    请求 q=100/kp=1e6 → 线上是 q≈1.0/kp≈30，clampedCount="
              << multi.state().clampedCount << "\n";
    multi.stop();
}

static void testWireRejectsNaN()
{
    std::cout << "[B2] 端到端：NaN 目标被拒绝，线上继续发上一份好目标\n";
    fake_serial::clear();

    go::GoMotorArray multi(makeConfig());
    std::string error;
    CHECK(multi.start(error));
    fake_serial::clearSent();

    CHECK(multi.setTargets(makeTargets(3, 0.25f)));
    sleepMs(30);
    CHECK(!fake_serial::sent().empty());
    const size_t mark = fake_serial::sent().size();

    // 上层算出了 NaN：setTargets 必须失败，而且不能污染已经建立的目标
    go::Targets bad = makeTargets(3, std::nanf(""));
    CHECK(!multi.setTargets(bad));
    CHECK(!multi.lastError().empty());
    sleepMs(30);

    CHECK(fake_serial::sent().size() > mark);
    for (size_t i = mark; i < fake_serial::sent().size(); ++i)
        CHECK(std::fabs(fake_serial::commandQ(fake_serial::sent()[i]) - 0.25) < 0.01); // 还是上一份好目标
    CHECK(!multi.faulted()); // 上层参数错误不算总线故障
    std::cout << "    NaN 被拒（lastError=\"" << multi.lastError() << "\"），线上仍是 0.25 rad\n";
    multi.stop();
}

static void testArrayRejectsBadLimits()
{
    std::cout << "[B3] 端到端：限幅表长度不对时 start() 直接失败\n";
    fake_serial::clear();
    go::RobotConfig config = makeConfig(); // 3 台电机
    config.limits.assign(2, go::JointLimits{}); // 但只给了 2 组限幅

    go::GoMotorArray multi(config);
    std::string error;
    CHECK(!multi.start(error));
    CHECK(error.find("limits") != std::string::npos);
    std::cout << "    " << error << "\n";
}

// 默认限位必须覆盖完整硬件顺序，且实际裁剪目标、允许站立姿态。
static void testDefaultConfigLimits()
{
    const auto config = go::makeDefaultConfig();
    CHECK(config.buses.size() == 4);
    CHECK(config.limits.size() == 12);
    std::string error;
    CHECK(go::Safety::limitsValid(config.limits, 12, error));
    const float lower[] = {-0.523f, -0.9f, -3.0f};
    const float upper[] = {0.523f, 2.87f, -0.733f};
    auto targets = makeTargets(12, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f);
    CHECK(go::Safety::clamp(targets, config.limits) == go::ClampResult::Clamped);
    for (size_t i = 0; i < 12; ++i) {
        CHECK(targets.q[i] == upper[i % 3]);
        CHECK(targets.dq[i] == 30.1f && targets.tau[i] == 23.7f);
        CHECK(targets.kp[i] == 70.0f && targets.kd[i] == 10.0f);
    }
    targets = makeTargets(12, -100.0f);
    CHECK(go::Safety::clamp(targets, config.limits) == go::ClampResult::Clamped);
    for (size_t i = 0; i < 12; ++i) CHECK(targets.q[i] == lower[i % 3]);
    targets = makeTargets(12, 0.0f);
    const float stand[] = {0.0f, 0.87f, -1.82f};
    for (size_t i = 0; i < 12; ++i) targets.q[i] = stand[i % 3];
    CHECK(go::Safety::clamp(targets, config.limits) == go::ClampResult::Unchanged);
}

int main()
{
    std::cout << "test_safety: Safety 限幅层\n";
    try
    {
        testDefaultConfigLimits();
        testClampByLimits();
        testClampPassThrough();
        testClampRejectsNonFinite();
        testApproach();
        testLimitsValid();
        testWireIsClamped();
        testWireRejectsNaN();
        testArrayRejectsBadLimits();
    }
    catch (const std::exception &e)
    {
        std::cerr << "test_safety: 失败：" << e.what() << "\n";
        return 1;
    }
    std::cout << "test_safety: 全部通过\n";
    return 0;
}
