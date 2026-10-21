#pragma once
#include "go_protocol.h"
#include <chrono>
#include <string>

namespace go {
class GoBus;

// Logical joint/output-side coordinates. Only GoBus owns transport and motor lifetime.
// 由 GoBus 管理的单电机逻辑状态，不持有串口、不启动线程。控制与反馈使用输出侧单位，offset 使用转子 rad。
class GoMotor {
public:
    using Clock = std::chrono::steady_clock;
    /**
     * 禁止复制/赋值：对象由单一所有者管理，避免重复持有设备或电机状态。
     */
    GoMotor(const GoMotor&) = delete;
    /**
     * 禁止复制/赋值：对象由单一所有者管理，避免重复持有设备或电机状态。
     */
    GoMotor& operator=(const GoMotor&) = delete;
    /**
     * 返回本总线内固定的电机 ID（0～14），用于区分电机；不同总线可以使用相同 ID。
     */
    int id() const { return id_; }
    /**
     * 返回固定的**输出侧**零点偏移（rad）——与其它所有公开量同一套单位（见 README 1.2）。
     * 含义：让「这个角度」成为逻辑零点。发送与接收都只用它做一次加减，不涉及减速比：
     *     发送线上的转子位置 = (direction × 逻辑位置 + offset()) × 6.33
     *     反馈得到的逻辑位置 = direction × (输出侧位置 − offset())
     * 减速比只在 packCommand()/unpackFeedback() 里出现一次。
     */
    float offset() const { return offset_; }
    /**
     * 返回固定的**方向系数**，只可能是 +1 或 -1。
     *   +1：逻辑正方向与电机原始正方向一致（默认）；
     *   -1：整个逻辑坐标系取反——**位置、速度、前馈力矩在读和写上一起取反**，
     *       而 offset 不参与取反（它是原始坐标系里的零点，与方向无关）。
     * 用途：左右腿镜像安装时，用 -1 让"同一个目标角度"在两侧代表同一个物理动作，
     * 这样同一份姿态/轨迹可以四条腿通用，状态反馈也都是逻辑坐标。
     */
    float direction() const { return direction_; }

    // Submission only: valid submissions refresh the watchdog; bus retransmission does not.
    /**
     * 提交该电机的输出侧 PD 目标：kp 单位 N·m/rad，kd 单位 N·m/(rad/s)，
     * q 为逻辑目标位置 rad，dq 为目标速度 rad/s，tau_ff 为输出端前馈力矩 N·m。
     * 前馈中不要再叠加同一组 PD 算出的力矩，否则会重复计算 PD。
     * 返回 true 仅表示目标已接受并刷新有效期；真正发送需要随后调用 bus.cycle()。
     * 非有限参数或逻辑位置加转子 offset/6.33 溢出会返回 false 并锁定故障；已有故障时拒绝提交。
     * 有限值超协议范围在编码时饱和，保存的 command() 仍是提交值；这不是总力矩限幅。
     */
    bool setPD(float kp, float kd, float q, float dq = 0, float tau_ff = 0);
    /**
     * 提交速度阻尼命令，相当于 setPD(0, kd, 0, dq, 0)。
     * kd 为输出侧阻尼增益，dq 为目标速度（默认 0）；返回值及提交/续期规则同 setPD()。
     */
    bool setDamping(float kd, float dq = 0);
    /**
     * 提交纯前馈力矩 tau（输出端 N·m），相当于 setPD(0, 0, 0, 0, tau)。
     * 关闭位置/速度 PD 项；返回值及实际发送时机同 setPD()。
     */
    bool setTorque(float tau);
    /**
     * 将本地命令恢复固定 ID、FOC 模式和全部零控制量，取消活动目标。
     * 这是卸力请求，不是编码器清零或保持当前位置；下一次 cycle() 才发送。
     * 不解除故障、不清除时间戳，也不保证实机已卸力。
     */
    void setZero(); // Unload, not encoder reset. Does not clear a latched fault.
    /**
     * 把某个逻辑角度对齐到"当前位置"：`offset += direction × (当前 qL − target)`。
     * target = 0 就是"把现在这里设成零点"；启动自动归零用 `target = start_pose_rad`。
     * 需要已收到过有效反馈；成功后当前反馈的 q 会被置成 target（按新零点是自洽的）。
     * 不操作串口；已有活动目标会作废（旧目标是按旧零点算的）。
     */
    bool alignTo(float logical_target);

    /**
     * 只读访问最近保存的逻辑输出侧命令（位置尚未加转子 offset/6.33）。
     * 它不是已发送或电机已执行的确认；后续 set 操作会改变引用所指内容。
     */
    const Command& command() const { return cmd_; }
    /**
     * 只读访问最近反馈；位置已减转子 offset/6.33，位置/速度/力矩均为输出侧量。
     * 使用前检查 valid、故障和时间戳；失效时旧数值可能仍保留。需要历史快照时自行复制。
     */
    const Feedback& feedback() const { return fb_; }
    /**
     * 返回 feedback().valid，仅表示反馈有效标志；不等于模式/温度/错误码全部正常，也不检查数据年龄。
     */
    bool lastOk() const { return fb_.valid; }
    /**
     * 返回该电机的锁定故障标志。故障期间拒绝运动命令，需要总线 clearFault() 成功后重新提交。
     */
    bool faulted() const { return fault_; }
    /**
     * 返回是否已经检测并锁定命令过期。检测在 bus.cycle() 中执行，不是后台自动计时结果。
     */
    bool commandExpired() const { return expired_; }
    /**
     * 返回本地是否保存活动目标；不表示已发送、未过期或运动已完成。setZero()/故障后为 false。
     */
    bool hasCommand() const { return active_; }
    /**
     * 返回最近有效目标 ID 反馈的单调时钟时间点；未收到时为默认时间点。
     * 反馈失效时可能保留旧时间，需结合 valid 使用；不能当作北京时间打印。
     */
    Clock::time_point lastFeedbackTime() const { return feedback_time_; }
    /**
     * 返回最近一次成功提交运动命令的单调时钟时间点。
     * 总线重复发送不会刷新；setZero() 保留原时间，初始化/会话重置/清故障时归零。
     */
    Clock::time_point lastCommandTime() const { return command_time_; }
    /**
     * 返回本电机的错误信息；提交或总线通信失败后查看，后续操作可能更新该内容。
     */
    const std::string& lastError() const { return error_; }
private:
    friend class GoBus;
    /**
     * 内部构造固定 ID 和转子侧 offset 的电机对象，不打开串口。
     * 调用方通过 bus.addMotor(id, offset_rad) 获取引用；参数校验和对象生命周期由总线负责。
     */
    GoMotor(int id, float offset_rad, float direction);
    /**
     * 内部提交入口：验证命令可编码及逻辑位置加转子 offset/6.33 不溢出，成功后保存命令并更新时间。
     * 已有故障直接拒绝，非法命令锁定故障；不操作串口。相同目标再次成功提交也会续期。
     */
    bool submit(const Command& cmd);
    /**
     * 内部生成发送用命令副本 out：活动目标的位置加一次转子 offset/6.33，不修改保存的逻辑目标。
     * zero 为 true 或无活动目标时生成全零 FOC 控制量，不给卸力帧加 offset。
     * 成功 true，位置加偏移溢出 false；减速比和字节编码由 packCommand() 继续处理。
     */
    bool wireCommand(Command& out, bool zero) const;
    /**
     * 内部锁定故障并清除活动目标，保留首次故障原因；expired 为 true 时同时锁定过期标志。
     * 只改本地状态，不发送；由 GoBus 随后的卸力事务执行通信。
     */
    void latchFault(const std::string& error, bool expired = false);
    /**
     * 内部会话重置：清空命令、反馈、时间戳及故障。仅在总线新连接成功后调用。
     * 因此 open() 前提交的目标不会保留，应在打开和反馈检查后重新提交。
     */
    void resetSession();
    const int id_;
    float offset_; // 输出侧；zero_at_start 时会在启动时被改写
    const float direction_; // +1 / -1
    Command cmd_;
    Feedback fb_;
    bool active_ = false, fault_ = false, expired_ = false;
    Clock::time_point command_time_{}, feedback_time_{};
    std::string error_;
};
}
