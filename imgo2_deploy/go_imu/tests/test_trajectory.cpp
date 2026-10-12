// ============================================================================
//  test_trajectory.cpp —— SmoothTrajectory（五次多项式平滑曲线）
//
//  纯数学，不需要串口也不需要线程。重点验证：
//    两端位置正确、**两端速度与加速度都为 0**、中间单调、峰值速度 = 1.875/T。
// ============================================================================
#include "go_trajectory.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

#define CHECK(x)                                                                               \
    do                                                                                         \
    {                                                                                          \
        if (!(x))                                                                              \
            throw std::runtime_error(std::string("CHECK failed: ") + #x);                      \
    } while (0)

namespace
{

constexpr double kTol = 1e-9;

void testEndpoints()
{
    std::printf("[1] 两端位置与边界\n");
    const double duration = 4.0;
    go::SmoothTrajectory traj(duration);

    CHECK(std::fabs(traj.progress(0.0) - 0.0) < kTol);
    CHECK(std::fabs(traj.progress(duration) - 1.0) < kTol);
    CHECK(std::fabs(traj.progress(duration * 0.5) - 0.5) < kTol); // 五次多项式是对称的
    // 超出范围要夹住，不能外推
    CHECK(traj.progress(-1.0) == 0.0);
    CHECK(traj.progress(duration * 3.0) == 1.0);
    CHECK(!traj.finished(duration - 0.01));
    CHECK(traj.finished(duration));
    CHECK(traj.duration() == duration);
    std::printf("    s(0)=0  s(T/2)=0.5  s(T)=1，超出范围被夹住\n");
}

void testZeroEndVelocities()
{
    std::printf("[2] 两端速度与加速度都为 0（选五次多项式的原因）\n");
    const double duration = 8.0;
    go::SmoothTrajectory traj(duration);

    CHECK(traj.rate(0.0) == 0.0);
    CHECK(traj.rate(duration) == 0.0);
    CHECK(traj.rate(-1.0) == 0.0);
    CHECK(traj.rate(duration + 1.0) == 0.0);

    // 用数值二阶差分检查两端加速度也接近 0（限速斜坡在这里会有明显跳变）
    const double h = 1e-4;
    const double accel_start = (traj.progress(2 * h) - 2 * traj.progress(h) + traj.progress(0.0)) / (h * h);
    const double accel_end = (traj.progress(duration) - 2 * traj.progress(duration - h) +
                              traj.progress(duration - 2 * h)) / (h * h);
    CHECK(std::fabs(accel_start) < 1e-4);
    CHECK(std::fabs(accel_end) < 1e-4);
    std::printf("    v(0)=v(T)=0，两端数值加速度 |a| < 1e-4\n");
}

void testPeakRate()
{
    std::printf("[3] 峰值速度 = 1.875 / T，出现在 T/2\n");
    const double duration = 2.5;
    go::SmoothTrajectory traj(duration);

    const double expect = go::SmoothTrajectory::peakRate() / duration;
    CHECK(std::fabs(traj.rate(duration * 0.5) - expect) < 1e-9);

    // 扫一遍，确认没有比 T/2 更大的
    double peak = 0.0;
    for (int i = 0; i <= 10000; ++i)
    {
        const double t = duration * double(i) / 10000.0;
        peak = std::max(peak, traj.rate(t));
    }
    CHECK(std::fabs(peak - expect) < 1e-6);
    std::printf("    T=%.1fs：峰值 %.4f 1/s（= 1.875/%.1f），扫描 10001 点无更大值\n", duration,
                peak, duration);
}

void testShape()
{
    std::printf("[4] 形状：单调不减、数值导数与 rate() 一致、平均速度 = 1/T\n");
    const double duration = 3.0;
    go::SmoothTrajectory traj(duration);

    const int steps = 3000;
    double previous = -1.0;
    for (int i = 0; i <= steps; ++i)
    {
        const double t = duration * double(i) / double(steps);
        const double s = traj.progress(t);
        CHECK(s >= previous - 1e-12); // 单调不减
        CHECK(s >= -kTol && s <= 1.0 + kTol);
        previous = s;
    }

    // 数值导数 vs 解析导数
    const double h = 1e-6;
    for (double t : {0.3, 1.0, 1.5, 2.7})
    {
        const double numeric = (traj.progress(t + h) - traj.progress(t - h)) / (2 * h);
        CHECK(std::fabs(numeric - traj.rate(t)) < 1e-6);
    }

    // 平均速度必须正好是 1/T（走完 1 个单位用 T 秒）
    double sum = 0.0;
    for (int i = 0; i < steps; ++i)
    {
        const double t = duration * (double(i) + 0.5) / double(steps);
        sum += traj.rate(t);
    }
    const double average = sum / double(steps);
    CHECK(std::fabs(average - 1.0 / duration) < 1e-6);
    std::printf("    单调不减；数值导数与解析导数一致；平均速度 %.6f = 1/T\n", average);
}

void testJointApplication()
{
    std::printf("[5] 套到一个关节上：8 秒从 0 转到 0.87 rad\n");
    const double duration = 8.0;
    const double delta = 0.87;
    go::SmoothTrajectory traj(duration);

    // 起步和收尾的位移应该极小（速度是 0），中间应该明显更快
    const double step = 0.02; // 20 ms 采样
    const double first = std::fabs(traj.progress(step) - traj.progress(0.0)) * delta;
    const double last = std::fabs(traj.progress(duration) - traj.progress(duration - step)) * delta;
    const double middle =
        std::fabs(traj.progress(duration * 0.5 + step) - traj.progress(duration * 0.5)) * delta;

    CHECK(first < middle / 50.0);
    CHECK(last < middle / 50.0);
    CHECK(std::fabs(traj.progress(duration) * delta - delta) < kTol);

    const double peak_dq = go::SmoothTrajectory::peakRate() * delta / duration;
    std::printf("    首个 20ms 走 %.6f rad，中间 20ms 走 %.6f rad，末个 20ms 走 %.6f rad\n", first,
                middle, last);
    std::printf("    峰值速度 = 1.875 × %.2f / %.0f = %.4f rad/s\n", delta, duration, peak_dq);
}

void testInvalidDuration()
{
    std::printf("[6] 非法时长\n");
    for (double bad : {0.0, -1.0})
    {
        bool threw = false;
        try
        {
            go::SmoothTrajectory traj(bad);
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        CHECK(threw);
    }
    std::printf("    时长 0 与负数都被拒绝\n");
}

} // namespace

int main()
{
    std::printf("test_trajectory: 五次多项式平滑曲线\n");
    try
    {
        testEndpoints();
        testZeroEndVelocities();
        testPeakRate();
        testShape();
        testJointApplication();
        testInvalidDuration();
    }
    catch (const std::exception &e)
    {
        std::fprintf(stderr, "test_trajectory: 失败：%s\n", e.what());
        return 1;
    }
    std::printf("test_trajectory: 全部通过\n");
    return 0;
}
