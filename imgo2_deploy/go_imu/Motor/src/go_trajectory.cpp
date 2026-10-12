#include "go_trajectory.h"

#include <stdexcept>

namespace go
{

SmoothTrajectory::SmoothTrajectory(double duration_s) : duration_(duration_s)
{
    if (!(duration_s > 0.0))
        throw std::invalid_argument("SmoothTrajectory duration must be greater than zero");
}

double SmoothTrajectory::progress(double t_s) const
{
    if (t_s <= 0.0)
        return 0.0;
    if (t_s >= duration_)
        return 1.0;
    const double tau = t_s / duration_;
    const double tau2 = tau * tau;
    const double tau3 = tau2 * tau;
    // 10τ³ − 15τ⁴ + 6τ⁵
    return tau3 * (10.0 + tau * (-15.0 + 6.0 * tau));
}

double SmoothTrajectory::rate(double t_s) const
{
    if (t_s <= 0.0 || t_s >= duration_)
        return 0.0; // 两端速度恰好为 0（这是选五次多项式的原因）
    const double tau = t_s / duration_;
    const double tau2 = tau * tau;
    // d/dτ (10τ³ − 15τ⁴ + 6τ⁵) = 30τ² − 60τ³ + 30τ⁴ = 30τ²(1 − 2τ + τ²)，再除以 T 换成对 t 的导数
    return 30.0 * tau2 * (1.0 + tau * (-2.0 + tau)) / duration_;
}

} // namespace go
