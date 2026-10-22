#pragma once

#include "go_bus.h"
#include "real_config.h"
#include "go_loop.h"

#include <chrono>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace go
{

/**
 * 多台电机的统一目标。所有 vector 的长度都必须等于 GoMotorArray::motorCount()，
 * 索引顺序 = 按配置里总线的顺序展开、每条总线内再按 motor_ids 的顺序。
 * 单位与 GoMotor::setPD 一致：kp N·m/rad、kd N·m/(rad/s)、q rad、dq rad/s、tau N·m，
 * 全部为输出侧量。
 */
struct Targets
{
    std::vector<float> kp, kd, q, dq, tau;

    size_t size() const { return q.size(); }
};

/**
 * 一份状态快照（按值返回）。
 */
struct State
{
    std::vector<uint64_t> receive_time_ns;
    std::vector<go::Feedback> feedback; // 按展开顺序
    std::vector<float> q;               // 输出侧位置，feedback[i].q 的快捷方式
    uint64_t cycleCount = 0;            // 每条总线都至少完成过这么多轮（取各总线最小值）
    uint64_t targetSeq = 0;             // 上层最后一次“确认”的序号
    long long targetAgeMs = 0;          // 那份目标距今多少毫秒（生成快照那一刻计算）
    bool stale = false;                 // 是否有总线因为目标过期而主动卸力（会自动恢复，不是故障）
    bool faulted = false;               // 真故障：**总线**打不开 / 反馈不健康等，需要上层处理
    uint64_t clampedCount = 0;          // 累计有多少次提交被 Safety 层夹过（说明上层顶到限幅了）
    std::string error;                  // 最近一次出问题的原因：可能是总线故障，也可能是上层调用出错
};

/**
 * 各总线循环质量的汇总（累计值，取各总线的最坏情况）。
 * 用来判断“节拍不稳”到底是我们自己慢、还是被调度器拖了：
 *   - bodyMaxUs 大、gapMaxUs 小 → 慢在一轮收发本身（串口/USB/电机回复），绑 CPU 没用
 *   - gapMaxUs 大               → 是唤醒被拖了，才值得考虑 CPU 绑定 / 实时优先级
 */
struct LoopStats
{
    long long intervalMinUs = 0; // 相邻两轮开始时刻的最短间隔；0 = 还没有样本
    long long intervalMaxUs = 0; // 最长间隔（= 真实控制周期的最坏情况）
    long long bodyMaxUs = 0;     // 单轮回调最长耗时
    long long gapMaxUs = 0;      // 每次醒来比计划最晚多少（调度延迟）
};

/**
 * 一条总线最近一轮的三段耗时（微秒）。只用于诊断，不参与任何判决。
 * 三段之和 + 少量代码开销 ≈ total_us（该轮 cycle() 的实际耗时）。
 */
struct BusTiming
{
    std::uint32_t total_us = 0; // 该轮 cycle() 总耗时
    std::uint32_t flush_us = 0; // 每次写之前的 tcflush 累计
    std::uint32_t write_us = 0; // 把命令写进驱动的累计
    std::uint32_t reply_us = 0; // 等应答 + 收字节 + 解析的累计
};

/**
 * 把若干条总线上的电机当成一组统一控制。
 *
 * 线程模型（B 方向）：每条总线一个定周期线程（go::Loop）自己跑，上层不驱动它。
 *   - 每个线程每轮：看上层有没有写新目标 → 有就 setPD → bus.cycle() → 发布反馈；
 *   - 上层用 setTargets() 非阻塞地把目标写进各总线的命令槽，用 state() 非阻塞地读反馈；
 *   - 因此上下层频率可以不同（例如上层 1 kHz、总线 200 Hz），互不拖累。
 *
 * 线程约定：start() / setTargets() / stop() 都只能在同一个上层线程里调用，
 * 上层不需要（也不允许）另外加锁。每条总线的 GoBus 只在它自己的 Loop 线程里构造、
 * 使用、关闭，跨线程只交换命令槽与反馈快照。
 */
class GoMotorArray
{
public:
    /**
     * config 为整机配置（总线、限幅、波特率、两个超时、总线节拍、看门狗时限）。
     * 看门狗的作用：如果上层超过 config.watchdog 没有再调用 setTargets()，
     * 总线线程会主动逐台 setZero() 卸力，并且**不会**让库把“命令过期”锁成需要
     * 人工恢复的故障；等上层恢复提交目标就自动重新接管。
     * 因此它必须**小于** command_timeout_ms，才能抢在库的命令过期判断之前动手；
     * 否则会落到库的故障路径上（需要显式 clearFault 才能恢复）。
     *
     * 参数非法时抛 std::invalid_argument：command_timeout_ms 不在 1..60000、
     * period <= 0、watchdog 为负、或 watchdog 不为 0 且 >= command_timeout_ms。
     * 构造只做准备，不打开串口、不启动线程。
     */
    explicit GoMotorArray(RobotConfig config);

    /**
     * 停止全部线程、逐台零指令卸力并关闭串口；可重复调用。
     */
    ~GoMotorArray();

    GoMotorArray(const GoMotorArray &) = delete;
    GoMotorArray &operator=(const GoMotorArray &) = delete;

    /**
     * 返回全部总线展开后的电机总数。
     */
    size_t motorCount() const { return motor_count_; }

    /**
     * 返回每条总线的目标节拍。
     */
    std::chrono::microseconds period() const { return period_; }

    /**
     * 返回上层失联看门狗的时限；0 表示关闭。
     */
    std::chrono::milliseconds watchdog() const { return watchdog_; }

    /**
     * 返回每个关节的限幅（可能为空 = 全部不限）。
     */
    const std::vector<JointLimits> &limits() const { return limits_; }

    /**
     * 为每条总线启动一个线程：线程内打开串口、注册电机、跑一次 FOC 全零探测，
     * 然后就进入各自的定周期循环。本函数阻塞到所有总线都完成启动探测为止。
     * 任一条失败则返回 false、把原因写入 error，并停掉已经起来的线程。
     */
    bool start(std::string &error);

    /**
     * 非阻塞：把全部电机的目标写进各总线的命令槽，并递增“上层又确认了一次”的序号；
     * 各总线在各自的下一个周期采纳这份目标。长度不符、未启动、或者
     * Safety 层判定目标不可用（出现 NaN / Inf）时返回 false。
     *
     * 写进去之前会先过一遍 Safety 限幅：位置、速度、前馈力矩、增益都会被夹到
     * config.limits 指定的范围内，所以即使上层算错也发不出去。
     * 被夹过多少次可以从 state().clampedCount 看出来。
     *
     * 调用它等于告诉底层“我还在，请按这份目标保持”。如果上层停止调用超过看门狗时限，
     * 底层会因为目标过期而主动卸力——这正是上层失联时的保护。
     */
    // Thread-safe latched stop. Each bus checks before its next normal transaction.
    // Only explicit stop/start can clear it. An in-flight transaction cannot be recalled.
    void requestUnload(const std::string& reason);
    bool setTargets(const Targets &targets);
    bool setTargets(const std::vector<float> &kp, const std::vector<float> &kd,
                    const std::vector<float> &q, const std::vector<float> &dq,
                    const std::vector<float> &tau);

    /**
     * 非阻塞：读一份状态快照。不保证是“你刚提交那一轮”的结果，
     * 可以用 cycleCount 判断底层又跑了多少轮。
     */
    // Thread-safe snapshot; lifecycle calls must not race external callers.
    State state() const;

    /**
     * 循环质量汇总（累计值，取各总线最坏情况）。见 LoopStats 的说明。
     */
    LoopStats loopStats() const;

    /**
     * 全部电机最近一次反馈，按展开顺序（等价于 state().feedback）。
     */
    std::vector<go::Feedback> feedback() const;

    /**
     * 全部电机的输出侧逻辑位置（rad）（等价于 state().q）。
     */
    std::vector<float> positions() const;

    /**
     * 返回第 bus_index 条总线最近一轮 cycle() 的耗时（毫秒）；越界返回 0。
     */
    double busMs(size_t bus_index) const;

    /**
     * 每条总线的设备路径，按总线顺序 —— 长度 = 总线数（不是电机数）。
     * 与 busMs() 同序，用来打印"哪条总线的一轮花了多久"。
     */
    std::vector<std::string> busDevices() const;

    /**
     * 最近一轮的三段耗时（µs，含该轮卸载重试）。只用于诊断/日志，不参与判决。
     * 三个分量的和 ≈ total_us 减去少量代码开销；用它定位"2 ms 到底花在哪一段"。
     */
    BusTiming busTiming(size_t bus_index) const;

    /**
     * 逐事务诊断埋点（只读）。每次 exchange() 一条，环形缓冲 16384 条。
     * 用来说清"某段没有新反馈的时间里，是没发、发了没回、还是回了但是坏帧"。
     */
    const TransactionTrace& transactionTrace() const { return trace_; }

    /**
     * 是否有任一条总线失败（真故障）。
     * 刻意**不**包含“上层调用出错”（目标长度不符、被 Safety 拒绝）：那是上层的编程错误，
     * 总线本身没坏，不该让上层误以为要收工。那些错误由 setTargets() 的返回值和
     * lastError() 报告。
     */
    bool faulted() const;

    /**
     * 最近一次出问题的原因；无错误时为空串。
     * 总线故障时是总线的错误；否则可能是最近一次 setTargets() 被拒的原因。
     * 想区分“要不要收工”，请用 faulted()。
     */
    std::string lastError() const;

    /**
     * 生成给终端看的状态文本（多行）：
     *   每条总线一行——设备、该轮 cycle() 耗时、完成轮数、STALE/FAILED 标记；
     *   每台电机一行——ID、位置、速度、估计力矩、温度、错误码；
     *   每条总线之后空一行。
     * 末尾自带换行，上层直接打印即可。仅供人看，不要用它做程序判断。
     */
    std::string statusText() const;

    /**
     * 停掉全部线程（各线程在自己的线程里完成零指令卸力并关闭串口）并回收。
     * 可重复调用；析构时也会调用。
     */
    void stop();

private:
    struct Worker;

    /**
     * 内部：校验配置（设备非空、至少一台、offset 长度匹配、ID 范围与同总线不重复）。
     */
    bool validate(std::string &error) const;

    /**
     * 内部：Loop 的每轮函数。第一轮做初始化（open + 探测），之后每轮跑一次收发。
     */
    void workerBody(size_t index);

    /**
     * 内部：Loop 的退出钩子，在 worker 自己的线程里关闭串口（含三轮零指令卸力）。
     */
    void closeWorker(size_t index);

    /**
     * 内部：worker 的初始化：构造 GoBus、注册电机、open、启动探测。
     * 成功返回 true；失败或抛异常都会转成该总线的错误状态（不会让线程 terminate）。
     */
    bool initWorker(Worker &worker, std::size_t bus_index);

    /**
     * 内部：跑一轮 = 有新区号就提交目标 → bus.cycle() → 发布反馈与本轮结果。
     */
    void roundOnce(Worker &worker);

    /**
     * 内部：把目标写进各总线的命令槽并递增序号，返回本轮序号（在锁内调用）。
     */
    uint64_t publishTargets(const Targets &targets);

    /**
     * 内部：校验 5 个 vector 的长度是否都等于 motorCount()。
     */
    bool targetsValid(const Targets &targets, std::string &error) const;

    std::vector<std::unique_ptr<Worker>> workers_;
    size_t motor_count_ = 0;
    std::vector<JointLimits> limits_;
    uint32_t baud_;
    size_t io_timeout_us_;
    size_t command_timeout_ms_;
    std::chrono::microseconds period_;
    std::chrono::milliseconds watchdog_;
    bool pipeline_ = true;
    std::uint32_t pipeline_gap_us_ = 150;
    std::chrono::milliseconds start_timeout_{5000}; // start() 等启动探测的上限
    TransactionTrace trace_; // 诊断埋点：各总线线程写入，停机/故障时读

    // ---- 以下由 mutex_ 保护，跨线程访问 ----
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    uint64_t target_seq_ = 0; // “上层又确认了一次”的计数
    uint64_t clamp_count_ = 0;                            // 被 Safety 夹过的次数
    std::chrono::steady_clock::time_point target_time_{}; // 上层最后一次确认的时刻

    // ---- 以下只由上层线程读写 ----
    std::atomic<bool> unload_requested_{false};
    std::string unload_reason_;
    bool started_ = false;
    bool stopped_ = false;
    std::string error_;
};

} // namespace go
