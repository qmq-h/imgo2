#pragma once
#include "function/my_myth.h"

namespace go
{

/**
 * 一条「从起点平滑走到终点」的位置曲线，**只描述进度**（0 → 1），不关心具体是哪个关节。
 *
 * 用五次多项式：
 *
 *     τ = clamp(t / T, 0, 1)
 *     progress(τ) = 10τ³ − 15τ⁴ + 6τ⁵
 *     rate(τ)     = (30τ² − 60τ³ + 30τ⁴) / T
 *
 * 它的关键性质是**两端的速度和加速度都恰好为 0**：起步和停止都不"顿"一下，
 * 也不会像"限速斜坡"那样在两端突然出现加速度跳变。峰值速度出现在 T/2 处，
 * 归一化值是 1.875，所以某个关节的峰值速度 = 1.875 × |Δq| / T。
 *
 * 典型用法（每个控制周期算一次）：
 *
 *     const double s  = traj.progress(elapsed);          // 0 → 1
 *     const double sd = traj.rate(elapsed);              // 1/s
 *     for (size_t i = 0; i < n; ++i) {
 *         q[i]  = q_start[i] + float(s  * (q_goal[i] - q_start[i]));
 *         dq[i] = float(sd * (q_goal[i] - q_start[i]));  // 速度前馈，让 PD 跟得更准
 *     }
 */
class SmoothTrajectory
{
public:
    /**
     * duration_s 必须大于 0，否则抛 std::invalid_argument。
     */
    explicit SmoothTrajectory(double duration_s);

    /**
     * 总时长（秒）。
     */
    double duration() const { return duration_; }

    /**
     * 归一化进度：t <= 0 返回 0，t >= T 返回 1，中间是五次多项式。
     */
    double progress(double t_s) const;

    /**
     * 归一化进度的导数（单位 1/s）：两端为 0，T/2 处达到峰值 1.875 / T。
     * 已经走完或还没开始时返回 0。
     */
    double rate(double t_s) const;

    /**
     * 是否已经到达终点（t >= T）。
     */
    bool finished(double t_s) const { return t_s >= duration_; }

    /**
     * 归一化速度的峰值（= 1.875），用来估算某个关节的最大速度：
     *     peak_dq = 1.875 * |q_goal - q_start| / T
     */
    static double peakRate() { return my_myth::quintic_peak_rate; }

private:
    double duration_;
};

} // namespace go
