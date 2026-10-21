#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

namespace go
{

/**
 * 定周期执行一段代码的线程。
 *
 * 语义：
 *   - 构造只记录名字、周期和回调，不创建线程；调用 start() 才真正起线程。
 *   - 每轮先执行 body，再睡到「上一轮的计划时刻 + period」。
 *     如果 body 本身比 period 还慢，就立刻开始下一轮（不补发积压的轮次），
 *     并把这一轮计入 lateCount()。
 *   - 睡眠用的是 condition_variable 的 wait_until，能被 stop() 立刻打断，
 *     不需要等满一个周期。
 *   - body 里抛出的异常会被 Loop 接住（否则线程里逃出去会 std::terminate）：
 *     记录到 lastError()，然后线程自己停下。此时 running() 变为 false。
 *
 * 线程约定：start() / stop() / join() 以及所有统计量的读取，都只能在
 * 「拥有这个 Loop 的那一个线程」里调用；body 在 Loop 自己的线程里执行。
 */
class Loop
{
public:
    using Clock = std::chrono::steady_clock;
    using Period = std::chrono::microseconds;

    /**
     * name 仅用于诊断输出；period 必须大于 0（否则抛 std::invalid_argument）。
     * onExit 可选：线程退出前会在 Loop 自己的线程里被调用一次（正常停止、
     * 或被 stop() 打断、或 body 抛异常而自停，三种情况都会调用），用来做
     * “必须在自己线程里完成”的收尾，例如关闭串口、卸力。onExit 里抛出的
     * 异常同样会被兜住并记入 lastError()。
     */
    Loop(std::string name, Period period, std::function<void()> body,
         std::function<void()> onExit = {});

    /**
     * 析构时自动 stop() + join()，不会让线程逃逸。
     */
    ~Loop();

    Loop(const Loop &) = delete;
    Loop &operator=(const Loop &) = delete;

    /**
     * 启动线程。返回时线程已经进入主循环（因此紧接着 running() 一定是 true）。
     * 已经在运行时返回 false 且不做任何事；若上一轮线程已结束但还没回收，
     * 会先 join 掉再重新启动，并把统计计数清零。成功返回 true。
     */
    bool start();

    /**
     * 请求线程退出：置退出标志并唤醒正在睡眠的线程。不阻塞、不回收线程。
     */
    void stop();

    /**
     * 等线程真正结束并回收。未启动或已回收时立即返回。
     */
    void join();

    /**
     * 线程是否正在跑（body 抛异常导致自停后也会变成 false）。
     */
    bool running() const { return running_.load(std::memory_order_acquire); }

    /**
     * 返回构造时给定的名字。
     */
    const std::string &name() const { return name_; }

    /**
     * 返回构造时给定的周期。
     */
    Period period() const { return period_; }

    /**
     * 已成功完成的轮数（body 正常返回才算一轮）。start() 时清零。
     */
    uint64_t tickCount() const { return ticks_.load(std::memory_order_relaxed); }

    /**
     * 超期轮数：body 耗时超过一个 period、以至于下一轮不得不立即开始的次数。
     */
    uint64_t lateCount() const { return late_.load(std::memory_order_relaxed); }

    /**
     * 上一轮 body 自身的耗时（微秒，不含睡眠）。start() 时清零。
     */
    long long lastBodyMicros() const { return last_body_us_.load(std::memory_order_relaxed); }

    /**
     * body 自身耗时的历史最大值（微秒）。它回答的是“**我们自己的代码**最慢一次花了多久”，
     * 在总线循环里这一项基本就是“一轮收发等串口等了多久”。
     */
    long long maxBodyMicros() const { return max_body_us_.load(std::memory_order_relaxed); }

    /**
     * 相邻两轮**开始时刻**间隔的最小值 / 最大值（微秒）——也就是真实控制周期的波动范围。
     * 0 表示还没有样本：**第 1 轮不参与间隔统计**（它没有"上一轮"），所以只跑了一轮的 Loop
     * 这里会是 0。注意 body 统计不受这个限制（第 1 轮和最后一轮都算），
     * 因此 maxInterval 与 maxBody 之间**没有**大小关系可比。
     */
    long long minIntervalMicros() const { return min_interval_us_.load(std::memory_order_relaxed); }
    long long maxIntervalMicros() const { return max_interval_us_.load(std::memory_order_relaxed); }

    /**
     * 每次从睡眠中醒来时“比计划晚”的历史最大值（微秒，负数按 0 计）。
     * 这一项回答的是“**调度器**把我拖了多久”：如果它一直很小，说明节拍完全由 body
     * 自己决定（例如一轮收发本来就比 period 慢），那绑 CPU / 实时优先级都帮不上忙。
     */
    long long maxGapMicros() const { return max_gap_us_.load(std::memory_order_relaxed); }

    /**
     * body 抛出异常时的说明；正常为空串。按值返回，可在任何线程调用。
     */
    std::string lastError() const;

    /**
     * 把上面那些统计量清零（不动运行状态，也不动 lastError）。
     *
     * 用途：如果某一轮 body 里含一次性的初始化（打开串口、启动探测…），
     * 那一轮的开销不该算进“最大耗时/周期波动”里，否则诊断会被它带偏。
     * 可以在任意线程调用；如果调用时 loop 正在跑，最多多统计到一轮（无害）。
     */
    void resetStats();

private:
    /**
     * 内部：线程主函数。先置 running_，然后循环执行 body 并按周期等待。
     */
    void threadMain();

    /**
     * 内部：记录异常信息并请求退出（加锁保护 error_）。
     */
    void fail(const std::string &message);

    std::string name_;
    Period period_;
    std::function<void()> body_;
    std::function<void()> on_exit_;

    std::thread thread_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool quit_ = false;    // 由 mutex_ 保护
    bool started_ = false; // 由 mutex_ 保护：线程是否已进入主循环（start() 的握手）
    std::string error_;    // 由 mutex_ 保护

    std::atomic<bool> running_{false};
    std::atomic<uint64_t> ticks_{0};
    std::atomic<uint64_t> late_{0};
    std::atomic<long long> last_body_us_{0};
    std::atomic<long long> max_body_us_{0};
    std::atomic<long long> min_interval_us_{0}; // 0 = 还没有样本
    std::atomic<long long> max_interval_us_{0};
    std::atomic<long long> max_gap_us_{0};
};

} // namespace go
