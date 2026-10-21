#include "go_loop.h"

#include <stdexcept>
#include <utility>

namespace go
{

Loop::Loop(std::string name, Period period, std::function<void()> body,
           std::function<void()> onExit)
    : name_(std::move(name)), period_(period), body_(std::move(body)),
      on_exit_(std::move(onExit))
{
    if (period_ <= Period::zero())
        throw std::invalid_argument("Loop period must be greater than zero");
    if (!body_)
        throw std::invalid_argument("Loop body must not be empty");
}

Loop::~Loop()
{
    stop();
    join();
}

bool Loop::start()
{
    if (running_.load(std::memory_order_acquire))
        return false; // 已经在跑

    join(); // 回收上一次的线程（如果有）

    {
        std::lock_guard<std::mutex> lock(mutex_);
        quit_ = false;
        started_ = false;
        error_.clear();
    }
    resetStats();

    thread_ = std::thread(&Loop::threadMain, this);

    // 握手：等线程真的进了主循环再返回，这样 start() 返回后 running() 一定是 true
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return started_ || quit_; });
    }
    return true;
}

void Loop::stop()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        quit_ = true;
    }
    cv_.notify_all();
}

void Loop::join()
{
    if (thread_.joinable())
        thread_.join();
}

void Loop::resetStats()
{
    ticks_.store(0, std::memory_order_relaxed);
    late_.store(0, std::memory_order_relaxed);
    last_body_us_.store(0, std::memory_order_relaxed);
    max_body_us_.store(0, std::memory_order_relaxed);
    min_interval_us_.store(0, std::memory_order_relaxed);
    max_interval_us_.store(0, std::memory_order_relaxed);
    max_gap_us_.store(0, std::memory_order_relaxed);
}

std::string Loop::lastError() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return error_;
}

void Loop::fail(const std::string &message)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (error_.empty())
        error_ = message;
    quit_ = true;
}

void Loop::threadMain()
{
    running_.store(true, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        started_ = true;
    }
    cv_.notify_all();

    // 三个统计量分别在三个位置测量：
    //   body 耗时   —— body 前后
    //   周期波动   —— 相邻两轮的 body 开始时刻之差
    //   调度延迟   —— 睡眠醒来时刻减计划时刻（只有这一项是“被别人拖了”）
    auto next = Clock::now();
    std::chrono::steady_clock::time_point prev_body_start{};
    bool first_round = true;

    for (;;)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (quit_)
                break;
        }

        const auto body_start = Clock::now();
        if (!first_round)
        {
            const auto interval_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                         body_start - prev_body_start)
                                         .count();
            const long long old_min = min_interval_us_.load(std::memory_order_relaxed);
            if (old_min == 0 || interval_us < old_min)
                min_interval_us_.store(interval_us, std::memory_order_relaxed);
            if (interval_us > max_interval_us_.load(std::memory_order_relaxed))
                max_interval_us_.store(interval_us, std::memory_order_relaxed);
        }
        first_round = false;
        prev_body_start = body_start;

        try
        {
            body_();
        }
        catch (const std::exception &e)
        {
            // 线程里放任异常逃出去会 std::terminate，所以在这里兜住并自停
            fail(std::string("body threw: ") + e.what());
            break;
        }
        catch (...)
        {
            fail("body threw an unknown exception");
            break;
        }
        const auto body_end = Clock::now();
        const auto body_us =
            std::chrono::duration_cast<std::chrono::microseconds>(body_end - body_start).count();
        last_body_us_.store(body_us, std::memory_order_relaxed);
        if (body_us > max_body_us_.load(std::memory_order_relaxed))
            max_body_us_.store(body_us, std::memory_order_relaxed);
        ticks_.fetch_add(1, std::memory_order_relaxed);

        next += period_;
        const auto now = Clock::now();
        if (next < now)
        {
            next = now; // 本轮超期：立即开始下一轮，不补发积压的轮次
            late_.fetch_add(1, std::memory_order_relaxed);
        }

        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait_until(lock, next, [this] { return quit_; });
        }
        // 醒来时比计划晚了多少：这一项才是“调度器/别的线程把我拖了多久”。
        // next 已经被上面按需重置成 now，所以“body 自己慢”不会算到这里来。
        const auto wake_us =
            std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - next).count();
        if (wake_us > max_gap_us_.load(std::memory_order_relaxed))
            max_gap_us_.store(wake_us, std::memory_order_relaxed);
    }

    // 退出钩子：三种退出路径（正常停、被 stop 打断、body 抛异常）都会走到这里。
    // 它跑在 Loop 自己的线程里，所以调用方可以在里面安全地释放“只属于本线程”的资源。
    if (on_exit_)
    {
        try
        {
            on_exit_();
        }
        catch (const std::exception &e)
        {
            fail(std::string("onExit threw: ") + e.what());
        }
        catch (...)
        {
            fail("onExit threw an unknown exception");
        }
    }

    running_.store(false, std::memory_order_release);
    cv_.notify_all(); // 线程已退出，唤醒可能还在等握手的人
}

} // namespace go
