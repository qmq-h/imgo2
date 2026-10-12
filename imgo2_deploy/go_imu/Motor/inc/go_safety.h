#pragma once

#include "real_config.h"
#include "go_motor_array.h" // Targets

#include <chrono>
#include <string>
#include <vector>

namespace go
{

/**
 * 限幅结果。
 */
enum class ClampResult
{
    Unchanged, // 原样通过
    Clamped,   // 有值被夹到边界（上层应该知道）
    Rejected   // 出现 NaN / Inf，或者各个 vector 长度不一致：整份目标作废，调用方应保留上一次的好目标
};

/**
 * Safety 层：把上层算出来的目标夹到安全范围内。
 *
 * 这一层**只管数值安全，不管策略对不对**：
 *   - 位置夹到 [q_min, q_max]
 *   - 速度夹到 ±dq_max
 *   - 前馈力矩夹到 ±tau_max
 *   - 增益夹到 [0, kp_max] / [0, kd_max]
 *   - 出现 NaN / Inf **直接整份拒绝**，而不是夹成边界——把“算错了”夹成“朝边界猛冲”更危险
 *
 * `GoMotorArray::setTargets()` 内部会自动调用 `clamp()`，所以即使上层算错也发不出去。
 * 这里全是纯函数（不持有状态、不阻塞、不碰线程），便于单独测试。
 *
 * 看门狗和它是互补的两层：**看门狗管“上层还在不在说话”，Safety 管“上层说的话能不能照做”。**
 */
class Safety
{
public:
    /**
     * 夹紧一份多关节目标（原地修改）。
     * limits 留空表示全部不限；否则长度必须等于 targets 的长度。
     * 任一项不是有限数、或各 vector 长度不一致 → 返回 Rejected 且**不修改** targets。
     */
    static ClampResult clamp(Targets &targets, const std::vector<JointLimits> &limits);

    /**
     * 限速推进：把 current 朝 desired 推进一步，每个关节的位置最多移动 dq_max × dt，
     * 原地修改 current；速度/力矩/增益直接取 desired（它们不需要斜坡）。
     * 返回是否所有关节都已经到达 desired。
     *
     * 典型用法：阶段切换（例如“锁住” → “站立”）时逐周期调用，避免目标一下子跳很远
     * 导致电机以最大能力冲过去。
     */
    static bool approach(Targets &current, const Targets &desired,
                         const std::vector<JointLimits> &limits, std::chrono::microseconds dt);

    /**
     * 校验 limits：长度必须是 0（不限）或 jointCount，且 q_min <= q_max、各项上限非负。
     */
    static bool limitsValid(const std::vector<JointLimits> &limits, size_t jointCount,
                            std::string &error);

    /**
     * 取第 index 个关节的限幅；limits 里没有这一项时返回“不限”。
     */
    static JointLimits limitsAt(const std::vector<JointLimits> &limits, size_t index);
};

} // namespace go
