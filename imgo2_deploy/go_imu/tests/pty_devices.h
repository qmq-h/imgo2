#pragma once

// ============================================================================
//  tests/pty_devices.h —— 只给伪终端集成测试用的胶水
//
//  把「环境变量里给的设备路径」覆盖到配置上（ID 与 offset 沿用配置里的），
//  这样各个 *_main.cpp 里不需要出现任何设备路径或电机 ID。
//
//  正式构建**不会**包含这个头：调用点在 `#ifdef GO_..._PTY_TEST` 里，
//  而且 `tests/` 目录只在测试目标上被加入 include 路径（见 CMakeLists.txt）。
// ============================================================================

#include "real_config.h"

#include <cstdlib>
#include <string>
#include <vector>

namespace go_test
{

/**
 * env_name 是逗号分隔的设备路径（例如 "/dev/pts/3,/dev/pts/5,..."）。
 * 成功时把 config.buses[i].device 逐个替换掉，返回 true；
 * 变量不存在或个数与 config.buses 对不上时返回 false 并把原因写进 error。
 */
inline bool applyTestDevices(go::RobotConfig &config, const char *env_name, std::string &error)
{
    const char *const devices = std::getenv(env_name);
    if (!devices)
    {
        error = "missing test devices";
        return false;
    }

    std::vector<std::string> paths;
    const std::string list(devices);
    for (size_t pos = 0; pos <= list.size();)
    {
        const size_t comma = list.find(',', pos);
        paths.push_back(
            list.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos));
        if (comma == std::string::npos)
            break;
        pos = comma + 1;
    }

    if (paths.size() != config.buses.size())
    {
        error = "test device count mismatch: expected " + std::to_string(config.buses.size()) +
                ", got " + std::to_string(paths.size());
        return false;
    }
    for (size_t i = 0; i < paths.size(); ++i)
        config.buses[i].device = paths[i];
    return true;
}

/**
 * 把整机设成**标准坐标系**：每台电机 offset = 0、direction = +1。
 *
 * 为什么测试需要它：假电机在伪终端上回的是**固定的原始角度**，而驱动器算出的逻辑角度是
 *     qL = direction × (qOut − offset/6.33)
 * 用户每重新标定一次 offset、或每给一台电机填上 -1（镜像安装），测试里看到的数就变一次，
 * 断言绝对角度的用例会时好时坏（而且失败信息完全看不懂）。设成标准坐标系之后
 * "线上角度 == 逻辑角度 == 假电机回的角度"，测试就可以直接断言绝对角度。
 *
 * 只有设了 GO_TEST_IDENTITY_FRAME 的程序才会调用它。
 */
inline void useIdentityFrame(go::RobotConfig &config)
{
    for (go::BusConfig &bus : config.buses)
    {
        bus.motor_directions.assign(bus.motor_ids.size(), 1.0f);
        bus.start_pose_rad.assign(bus.motor_ids.size(), 0.0f);
    }
}

} // namespace go_test
