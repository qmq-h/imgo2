#pragma once
#include "go_motor.h"
#include "go_trace.h"
#include "motor_transport.h"
#include <atomic>
#include "serial_port.h"
#include <array>
#include <memory>
#include <vector>

namespace go {
// Single-threaded bus owner. Use one external worker per bus when scaling up.
// 一条物理总线对应一个 GoBus；内部不加锁。set 命令与 cycle 等操作应在同一线程串行执行。
// 多线程策略接入时需在外部同步命令快照，不能同时修改电机目标和运行 cycle。
class GoBus {
public:
    /**
     * 创建一条物理 RS485 总线的所有者，记录 device、baud，不打开设备、不启动线程。
     * io_timeout_us 为每台电机一次发送+接收的预算（微秒）；command_timeout_ms 为命令有效期
     * （毫秒，1～60000，非法时抛 invalid_argument）。串口参数在 open() 时检查。
     */
    GoBus(const std::string& device, uint32_t baud = 4000000,
          size_t io_timeout_us = 20000, size_t command_timeout_ms = 100);
    /**
     * 析构时调用 close()，限时尝试逐台卸力后释放串口及电机对象。
     * 可能等待多次事务；销毁后 addMotor() 返回的所有引用失效。
     */
    ~GoBus();
    /**
     * 禁止复制/赋值：对象由单一所有者管理，避免重复持有设备或电机状态。
     */
    GoBus(const GoBus&) = delete;
    /**
     * 禁止复制/赋值：对象由单一所有者管理，避免重复持有设备或电机状态。
     */
    GoBus& operator=(const GoBus&) = delete;
    void bindStopFlag(const std::atomic<bool>* flag) { external_stop_ = flag; }
    /**
     * 切换一轮的收发方式（必须在 open() 之前调用）：
     *   false（默认）= 逐台"发送→等应答→下一台"，半双工最保守；
     *   true          = 先连发本总线全部命令（相邻间隔 gap），再按 ID 收全部应答。
     * 间隔下限受"命令 43µs + 电机换向 ~10µs + 应答 40µs"约束，建议 >=100µs。
     * 任何一台没收到/反馈不健康，仍然和逐台模式一样锁故障并逐台卸力。
     */
    void setPipelined(bool on, std::chrono::microseconds gap = std::chrono::microseconds(150)) {
        if (isOpen()) return; // 只允许打开前切换，避免运行中改语义
        pipelined_ = on;
        if (gap.count() > 0) gap_ = gap;
    }
    TransportStats transportStats() const { return stats_; }
    /**
     * 诊断埋点（只记录，不改变控制与超时行为）：每次 exchange() 记一条 TraceRecord。
     * 必须在 open() / 本总线线程启动之前调用；trace 的生命周期要长于本对象。
     */
    void attachTrace(TransactionTrace* trace, std::size_t bus_index) { trace_ = trace; trace_bus_ = static_cast<unsigned>(bus_index); }
    /**
     * 最近一轮 cycle() 的分段耗时（µs，含该轮的卸载重试）。只在诊断/日志里用，不参与判决；
     * 每次 cycle() 开头清零。
     *   逐台模式：flush/write/reply 分别为各台事务的累计，first 为首字节等待累计；
     *   流水线模式：flush 只有一次，write 为各条写入合计，first 为"开始发送→第一个应答字节"，
     *               reply 为"最后一条写完→全部收齐"（即一个芯片攒包周期的量级）。
     */
    std::uint32_t roundFlushUs() const { return round_flush_us_; }
    std::uint32_t roundFirstUs() const { return round_first_us_; }
    std::uint32_t roundWriteUs() const { return round_write_us_; }
    std::uint32_t roundReplyUs() const { return round_reply_us_; }
    // Throws invalid_argument for invalid/duplicate ID or offset; logic_error if open.
    // Reference remains stable until the bus is destroyed. Different buses may reuse IDs.
    /**
     * 在 open() 前注册本总线电机，id 为 0～14，offset_rad 为有限的**输出侧**零点偏移（rad，与其它公开量同单位）。
     * direction 只接受 +1 或 -1：+1 表示逻辑正方向与电机原始正方向一致（默认），
     * -1 表示逻辑坐标整体取反（位置/速度/力矩的读写都反，offset 不反），用于镜像安装的腿。
     * 重复 ID/非法参数（含 direction 不是 ±1）抛 invalid_argument，已打开时调用抛 logic_error。
     * 返回总线持有的稳定引用；每个电机分别调用 setPD()，不要删除该对象。不同总线可复用 ID。
     */
    GoMotor& addMotor(int id, float offset_rad = 0, float direction = 1.0f);
    /**
     * 打开并配置串口；至少先注册一台电机。成功 true，失败 false 并查看 lastError()。
     * 新连接会清除全部旧命令、反馈和故障；这里只配置，不发送、不证明电机在线。
     * 已打开时重复调用直接返回 true，不重置状态；新打开后未提交目标时 cycle() 发送零指令。
     */
    bool open();
    /**
     * 逐台尝试全零 FOC 请求/应答，共进行三轮，随后关闭串口并使全部反馈失效。
     * 每次事务受 io_timeout_us 限制，总等待预算约为 3×电机数×单次超时，另有系统开销。
     * 保留注册对象以便重新打开；重复关闭不再通信。发送零指令不等于已确认实机卸力。
     */
    void close(); // Up to three zero-command transactions per registered motor.
    /**
     * 返回底层串口是否打开；不代表所有电机在线或反馈健康。
     */
    bool isOpen() const { return port_.isOpen(); }
    /**
     * 返回串口已配置并经驱动回读匹配的波特率，关闭时为 0；不是物理线速测量。
     */
    uint32_t baudrate() const { return port_.baudrate(); }
    /**
     * 返回已注册电机数量；关闭总线不会删除这些注册对象。
     */
    size_t motorCount() const { return motors_.size(); }
    /**
     * 由调用方在单线程周期中调用：按注册顺序逐台发送一帧并等待对应反馈，再处理下一台。
     * 本函数不创建线程、不休眠调度；先分别更新各电机目标，再调用一次即可处理整条总线。
     * 全部事务成功、反馈健康且无锁定故障时返回 true；否则锁定本总线故障并逐台尝试卸力。
     * 故障期间仍可调用以采集零指令反馈，但仍返回 false，不自动恢复运动。
     * 活动命令仅由上层成功提交续期；过期检查依赖 cycle() 运行，线程停止时不会自动执行。
     */
    bool cycle(); // One request/reply per registered motor, in registration order.
    /**
     * 返回整条总线是否锁定故障；任一电机异常会影响本总线全部电机，不自动传播到其他总线。
     */
    bool faulted() const { return fault_; }
    // Requires fresh (age < command timeout) healthy replies for ALL motors.
    // Clears commands too: fresh setPD submissions are required after recovery.
    /**
     * 显式恢复入口，不收发：要求串口打开且所有电机现有反馈健康、年龄小于命令有效期。
     * 可先在故障状态调用 cycle() 获取新反馈，再调用本函数；成功 true，否则 false 并保留故障。
     * 成功会清除故障和所有旧目标，即使原先无故障也会清目标；必须重新提交才能运动。
     */
    bool clearFault();
    /**
     * 返回总线最近错误/锁定故障原因。cycle()/open()/clearFault() 失败后查看；成功操作可能清空。
     */
    const std::string& lastError() const { return error_; }
private:
    /**
     * 内部执行一台电机的一次事务：使反馈失效、准备命令、清旧输入、发送并等待目标回复。
     * zero=true 强制卸力；false 使用活动目标或待机零指令。发送与接收共用同一个截止时间。
     * 返回 true 只代表得到可解析的目标 ID 反馈，健康检查由 cycle() 负责。
     */
    bool exchange(GoMotor& motor, bool zero);
    bool cyclePipelined();
    bool applyFeedback(GoMotor& motor, Feedback next);
    void traceRound(const GoMotor& motor, std::uint64_t start_ns, bool ok,
                    std::uint32_t waited_us, std::uint16_t bytes,
                    std::uint16_t bad, std::uint16_t stray);
    /**
     * 内部保留总线首次故障原因 reason，并锁定本总线所有电机、清除活动命令。
     * 不执行串口操作；调用方随后使用 unloadOnce() 尝试卸力。
     */
    void latchFault(const std::string& reason);
    /**
     * 内部按注册顺序对每台电机执行一次零指令请求/应答；一台失败仍继续下一台。
     * 每次等待有期限，不连续突发多台请求；不解除故障，也不保证全部电机已卸力。
     */
    void unloadOnce();
    /**
     * 内部检查反馈有效、模式为 FOC、错误码为 0、温度不超过 80℃。
     * 不检查反馈年龄或是否到达目标；恢复时由 clearFault() 另外检查新鲜度。
     */
    static bool healthy(const Feedback& fb);
    std::string device_, error_;
    uint32_t baud_;
    size_t io_timeout_us_, command_timeout_ms_;
    bool fault_ = false;
    SerialPort port_;
    std::vector<std::unique_ptr<GoMotor>> motors_;
    TransportStats stats_;
    const std::atomic<bool>* external_stop_ = nullptr;
    TransactionTrace* trace_ = nullptr; // 诊断埋点，可为空
    unsigned trace_bus_ = 0;
    std::uint32_t round_flush_us_ = 0, round_write_us_ = 0, round_reply_us_ = 0;
    std::uint32_t round_first_us_ = 0;
    bool pipelined_ = false;
    std::chrono::microseconds gap_{150};
};
}
