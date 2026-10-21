#pragma once
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

namespace go_app {

// 一个控制节拍（默认 5 ms）的样本，**硬件顺序**（= 总线/电机 ID 顺序）。
// 只在控制线程里填充并 push()；格式化与写盘都在写线程，控制线程不做字符串、不做 IO。
struct LogSample {
    double t_mono_ms = 0;       // 单调时钟，程序起点为 0
    std::int64_t t_wall_ms = 0; // Unix epoch 毫秒（和 dmesg -T 对齐用）
    std::uint64_t tick = 0;     // 控制节拍序号
    std::uint64_t generation = 0;
    std::uint8_t mode = 0;      // RobotFSM::Mode 的整数值
    std::uint8_t clamped = 0;   // Safety 是否改过本次目标
    std::uint16_t valid_mask = 0;
    std::uint16_t err_mask = 0;
    std::uint8_t imu_valid = 0;
    float cmd[3]{};                                                        // 速度指令（已按速度上限夹过）
    float kp[12]{}, kd[12]{}, q_des[12]{}, dq_des[12]{}, tau_ff[12]{};     // 实际下发（Safety 夹后）
    float q[12]{}, dq[12]{}, tau[12]{}, temp[12]{};                        // 电机反馈
    std::int16_t age_ms[12]{};                                             // 每关节各自的反馈年龄
    float quat[4]{}, gyro[3]{};                                            // IMU（PPO 只用这两项）
    std::int16_t imu_age_ms = -1;
    float imu_hz = 0;
    float act[12]{};                                                       // 最近一次策略输出（策略顺序）
    // 每条总线最近一轮的三段耗时（µs）：total ≈ flush + write + reply + 少量代码开销。
    // 用来回答"一个事务 ~2 ms 到底花在哪一段"，不需要再单独开 trace。
    std::uint32_t bus_total_us[4]{}, bus_flush_us[4]{}, bus_write_us[4]{}, bus_reply_us[4]{};
};

// 策略样本：只在 PPO 期间、每次消费到新结果时写一条（约 50 Hz）。
// 约定：obs 是供**下一次**推理使用的输入（其中 obs[33..44] 就是 act 的上一版）；
//       act 是本次刚消费到的网络输出；result_age_ms 说明它有多旧。
struct PolicySample {
    double t_mono_ms = 0;
    std::int64_t t_wall_ms = 0;
    std::uint64_t tick = 0, generation = 0;
    std::uint8_t mode = 0;
    float cmd[3]{};
    float obs[45]{};
    float act[12]{};
    float result_age_ms = 0, obs_age_ms = 0;
};

// 打开时一次写入 meta.json 的自描述内容（由调用方拼好片段，写线程不依赖工程类型）。
struct LogMeta {
    std::string exe, args, created;
    int every = 1, control_ms = 5, policy_ms = 20;
    std::string joints_json;  // [{index,label,device,motor_id,direction},...]
    std::string config_json;  // 映射/增益/标度/限幅/时序等（JSON 对象）
};

// 单生产者（控制线程）→ 单消费者（写线程）环形队列。
// 满时丢弃**最新**样本并计数，绝不阻塞控制线程；计数写到 events.log。
class RunLogger {
public:
    RunLogger() = default;
    ~RunLogger();
    RunLogger(const RunLogger&) = delete;
    RunLogger& operator=(const RunLogger&) = delete;

    // 建目录并打开三份文件，写 meta.json。失败返回 false 并给 error（是否继续由调用方决定）。
    bool open(const std::string& dir, const LogMeta& meta, int max_mb, std::string& error);
    void push(const LogSample&);       // 非阻塞
    void pushPolicy(const PolicySample&);
    void event(const std::string&);    // 低频：状态切换/按键/故障/丢行
    void close();
    bool enabled() const { return enabled_; }
    std::string directory() const { return dir_; }
    std::uint64_t dropped() const { return dropped_.load(std::memory_order_relaxed); }

private:
    void run();
    void rotateIfNeeded();
    void writeHeaders();
    void row(const LogSample&, std::string&);
    void prow(const PolicySample&, std::string&);

    static constexpr std::size_t kRobotRing = 4096;  // ~20 s @200 Hz
    static constexpr std::size_t kPolicyRing = 512;  // ~10 s @50 Hz
    std::vector<LogSample> robot_ring_;
    std::vector<PolicySample> policy_ring_;
    std::atomic<std::size_t> robot_head_{0}, robot_tail_{0};
    std::atomic<std::size_t> policy_head_{0}, policy_tail_{0};
    std::atomic<std::uint64_t> dropped_{0};
    std::atomic<bool> stop_{false};
    std::thread writer_;
    std::FILE* robot_file_ = nullptr;
    std::FILE* policy_file_ = nullptr;
    std::FILE* events_file_ = nullptr;
    std::string dir_, robot_path_, events_path_;
    int max_bytes_ = 0;
    long long robot_bytes_ = 0;
    int file_index_ = 1;
    bool enabled_ = false;
    LogMeta meta_;
};
} // namespace go_app
