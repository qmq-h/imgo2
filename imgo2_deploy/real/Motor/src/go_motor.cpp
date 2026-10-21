#include "go_motor.h"
#include <cmath>

namespace go {
/**
 * 内部构造固定 ID 和转子侧 offset 的电机对象，不打开串口。
 * 调用方通过 bus.addMotor(id, offset_rad) 获取引用；参数校验和对象生命周期由总线负责。
 */
GoMotor::GoMotor(int id, float offset_rad, float direction)
    : id_(id), offset_(offset_rad), direction_(direction) {
    cmd_.id = static_cast<uint8_t>(id);
}
/**
 * 将本地命令恢复固定 ID、FOC 模式和全部零控制量，取消活动目标。
 * 这是卸力请求，不是编码器清零或保持当前位置；下一次 cycle() 才发送。
 * 不解除故障、不清除时间戳，也不保证实机已卸力。
 */
void GoMotor::setZero() {
    cmd_.reset();
    cmd_.id = static_cast<uint8_t>(id_);
    active_ = false;
}
/**
 * 内部会话重置：清空命令、反馈、时间戳及故障。仅在总线新连接成功后调用。
 * 因此 open() 前提交的目标不会保留，应在打开和反馈检查后重新提交。
 */
void GoMotor::resetSession() {
    setZero();
    fb_ = Feedback{};
    command_time_ = feedback_time_ = Clock::time_point{};
    fault_ = expired_ = false;
    error_.clear();
}
/**
 * 内部锁定故障并清除活动目标，保留首次故障原因；expired 为 true 时同时锁定过期标志。
 * 只改本地状态，不发送；由 GoBus 随后的卸力事务执行通信。
 */
void GoMotor::latchFault(const std::string& error, bool expired) {
    if (!fault_) error_ = error;
    fault_ = true;
    expired_ = expired_ || expired;
    setZero();
}
/**
 * 内部生成发送用命令副本 out：活动目标加一次输出侧 offset 并乘方向（减速比在 packCommand 里做），不修改保存的逻辑目标。
 * zero 为 true 或无活动目标时生成全零 FOC 控制量，不给卸力帧加 offset。
 * 成功 true，位置加偏移溢出 false；减速比和字节编码由 packCommand() 继续处理。
 */
bool GoMotor::wireCommand(Command& out, bool zero) const {
    out = cmd_;
    if (zero || !active_) {
        out.reset(); // Unload frames must remain all-zero even with a nonzero offset.
        return true;
    }
    // direction = -1 时整个逻辑坐标系取反：位置、速度、前馈力矩一起反。
    // offset 不参与取反——它定义的是原始坐标系里的零点，与逻辑方向无关。
    const float q = direction_ * out.q + offset_; // offset 是输出侧，与位置同单位
    if (!std::isfinite(q)) return false;
    out.q = q;
    out.dq = direction_ * out.dq;
    out.tau = direction_ * out.tau;
    return true;
}
/**
 * 内部提交入口：验证命令可编码及逻辑位置加转子 offset/6.33 不溢出，成功后保存命令并更新时间。
 * 已有故障直接拒绝，非法命令锁定故障；不操作串口。相同目标再次成功提交也会续期。
 */
// 把 target 对齐到当前位置：见头文件说明。
bool GoMotor::alignTo(float logical_target) {
    if (!fb_.valid) { error_ = "no fresh feedback to align against"; return false; }
    offset_ += direction_ * (fb_.q - logical_target);
    fb_.q = logical_target; // 按新零点，当前位置就是 target
    active_ = false;        // 旧目标是按旧零点算的，作废
    cmd_.reset();
    cmd_.id = static_cast<uint8_t>(id_);
    return true;
}
bool GoMotor::submit(const Command& cmd) {
    if (fault_) return false;
    uint8_t frame[kCmdLen];
    const float q = direction_ * cmd.q + offset_;
    if (!packCommand(cmd, frame) || !std::isfinite(q)) {
        latchFault("invalid command or position+offset overflow");
        return false;
    }
    cmd_ = cmd;
    active_ = true;
    command_time_ = Clock::now();
    error_.clear();
    return true;
}
/**
 * 提交该电机的输出侧 PD 目标：kp 单位 N·m/rad，kd 单位 N·m/(rad/s)，
 * q 为逻辑目标位置 rad，dq 为目标速度 rad/s，tau_ff 为输出端前馈力矩 N·m。
 * 前馈中不要再叠加同一组 PD 算出的力矩，否则会重复计算 PD。
 * 返回 true 仅表示目标已接受并刷新有效期；真正发送需要随后调用 bus.cycle()。
 * 非有限参数或逻辑位置加转子 offset/6.33 溢出会返回 false 并锁定故障；已有故障时拒绝提交。
 * 有限值超协议范围在编码时饱和，保存的 command() 仍是提交值；这不是总力矩限幅。
 */
bool GoMotor::setPD(float kp, float kd, float q, float dq, float tau) {
    Command c;
    c.id = static_cast<uint8_t>(id_);
    c.kp = kp; c.kd = kd; c.q = q; c.dq = dq; c.tau = tau;
    return submit(c);
}
/**
 * 提交速度阻尼命令，相当于 setPD(0, kd, 0, dq, 0)。
 * kd 为输出侧阻尼增益，dq 为目标速度（默认 0）；返回值及提交/续期规则同 setPD()。
 */
bool GoMotor::setDamping(float kd, float dq) { return setPD(0, kd, 0, dq, 0); }
/**
 * 提交纯前馈力矩 tau（输出端 N·m），相当于 setPD(0, 0, 0, 0, tau)。
 * 关闭位置/速度 PD 项；返回值及实际发送时机同 setPD()。
 */
bool GoMotor::setTorque(float tau) { return setPD(0, 0, 0, 0, tau); }
}
