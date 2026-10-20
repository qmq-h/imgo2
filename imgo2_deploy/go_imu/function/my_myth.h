#pragma once
#include <algorithm>
namespace my_myth {
constexpr double pi = 3.14159265358979323846;
constexpr double two_pi = 2.0 * pi;
constexpr double degrees_to_radians = pi / 180.0;
constexpr double radians_to_degrees = 180.0 / pi;
constexpr float pi_f = 3.14159265358979323846f;
constexpr float degrees_to_radians_f = pi_f / 180.0f;
constexpr float radians_to_degrees_f = 57.295779513f;
template <typename T> constexpr double degToRad(T value) {return value * degrees_to_radians;}
template <typename T> constexpr double radToDeg(T value) {return value * radians_to_degrees;}
// 纯数值操作；调用方须先校验有限性及 low<=high，安全策略不在数学模块。
template <typename T> constexpr T clamp(T value,T low,T high) {return std::max(low,std::min(value,high));}
constexpr double quintic_peak_rate=1.875;
double quinticProgress(double elapsed_s,double duration_s);
double quinticRate(double elapsed_s,double duration_s);
}

#include <array>
namespace my_myth {
// Unit body-to-world wxyz. Returns R(q)^T * [0,0,-1], no accelerometer/translation.
std::array<float,3> projectedGravity(const std::array<float,4>& quaternion);
}
