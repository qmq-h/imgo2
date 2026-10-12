#include "go_motor_array.h"

#include <algorithm>
#include <cmath>

#include "go_safety.h"

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace go
{

// 一条总线的运行状态，分三类：
//
//   ① 启动后只读：config
//   ② 只属于本 worker 的 Loop 线程，不加锁：bus / motors / loop / applied / initialized
//      （bus 在本线程构造、使用、并在 Loop 的退出钩子里关闭，因此“一条总线只被一个线程碰”）
//   ③ 跨线程访问，一律在 GoMotorArray::mutex_ 保护下：
//      commands（上层写） / feedback、applied_seq、cycle_count、bus_ms、ready、failed、error（worker 写）
struct GoMotorArray::Worker
{
    // ① 只读
    BusConfig config;

    // ② 线程私有
    std::unique_ptr<go::GoBus> bus;
    std::vector<go::GoMotor *> motors;
    std::unique_ptr<Loop> loop;
    std::vector<go::Command> applied; // 本线程最近一次采纳的目标（预分配，避免每轮分配）
    bool initialized = false;
    bool pd_active = false;      // 本总线当前是否挂着 PD 目标（挂上了才需要看门狗管）
    bool stale_latched = false;  // 是否处于“上层失联已卸力”状态；要等上层重新提交才解除

    // ③ 加锁访问
    std::vector<go::Command> commands;
    std::vector<go::Feedback> feedback;
    std::vector<uint64_t> receive_time_ns;
    std::vector<float> zero_shifts; // 启动自动归零修正量（qL − start_pose）
    uint64_t applied_seq = 0;  // 已完成到哪个序号
    uint64_t cycle_count = 0;  // 完成过多少轮
    double bus_ms = 0.0;       // 最近一轮 cycle() 耗时
    bool ready = false;        // 启动探测是否成功
    bool failed = false;       // 最近一轮是否失败（真故障）
    bool stale = false;        // 是否因为目标过期而主动卸力（会自动恢复，不算故障）
    std::string error;
};

GoMotorArray::GoMotorArray(RobotConfig config)
    : limits_(std::move(config.limits)), baud_(config.baud),
      io_timeout_us_(config.io_timeout_us), command_timeout_ms_(config.command_timeout_ms),
      period_(config.period), watchdog_(config.watchdog)
{
    if (command_timeout_ms_ == 0 || command_timeout_ms_ > 60000)
        throw std::invalid_argument("command timeout must be 1..60000 ms");
    if (period_ <= std::chrono::microseconds::zero())
        throw std::invalid_argument("period must be greater than zero");
    if (watchdog_ < std::chrono::milliseconds::zero())
        throw std::invalid_argument("watchdog must not be negative");
    if (watchdog_ > std::chrono::milliseconds::zero() &&
        watchdog_ >= std::chrono::milliseconds(command_timeout_ms_))
        throw std::invalid_argument(
            "watchdog must be shorter than command timeout, otherwise the bus falls into the "
            "library's latched command-expiry fault instead of unloading and recovering by itself");

    for (BusConfig &bus : config.buses)
    {
        if (bus.motor_directions.empty())
            bus.motor_directions.assign(bus.motor_ids.size(), 1.0f);
        if (bus.start_pose_rad.empty())
            bus.start_pose_rad.assign(bus.motor_ids.size(), 0.0f);

        std::unique_ptr<Worker> worker(new Worker());
        worker->config = std::move(bus);

        const size_t count = worker->config.motor_ids.size();
        worker->commands.resize(count);
        worker->applied.resize(count);
        worker->feedback.assign(count, go::Feedback{});
        worker->receive_time_ns.assign(count, 0);
        worker->zero_shifts.assign(count, 0.0f);
        worker->motors.reserve(count);
        motor_count_ += count;

        workers_.push_back(std::move(worker));
    }
}

GoMotorArray::~GoMotorArray()
{
    stop();
}

bool GoMotorArray::validate(std::string &error) const
{
    if (workers_.empty())
    {
        error = "at least one bus is required";
        return false;
    }
    // 限幅表要么留空（全不限），要么和展开后的关节数一致
    if (!Safety::limitsValid(limits_, motor_count_, error))
        return false;
    // 同一个设备路径不能出现在两条总线里：那会创建两个 GoBus、两条线程去碰同一个串口，
    // 表现和“两个进程抢一个串口”完全一样（互相冲帧、互相偷反馈）。
    for (size_t i = 0; i < workers_.size(); ++i)
        for (size_t j = 0; j < i; ++j)
            if (workers_[i]->config.device == workers_[j]->config.device)
            {
                error = "device " + workers_[i]->config.device +
                        " is configured on more than one bus; a bus must be owned by exactly one "
                        "thread, otherwise both threads fight over the same 485 wire";
                return false;
            }
    for (const std::unique_ptr<Worker> &worker : workers_)
    {
        const BusConfig &config = worker->config;
        if (config.device.empty())
        {
            error = "empty device path";
            return false;
        }
        const std::string tag = "bus " + config.device;
        if (config.motor_ids.empty())
        {
            error = tag + ": at least one motor is required";
            return false;
        }
        if (config.motor_directions.size() != config.motor_ids.size())
        {
            error = tag + ": motor_directions must have the same length as motor_ids";
            return false;
        }
        for (size_t i = 0; i < config.motor_directions.size(); ++i)
            if (config.motor_directions[i] != 1.0f && config.motor_directions[i] != -1.0f)
            {
                error = tag + ": motor_directions[" + std::to_string(i) + "] must be +1 or -1";
                return false;
            }
        for (size_t i = 0; i < config.motor_ids.size(); ++i)
        {
            if (config.motor_ids[i] < 0 || config.motor_ids[i] > 14)
            {
                error = tag + ": motor ID " + std::to_string(config.motor_ids[i]) +
                        " out of range 0..14";
                return false;
            }
            for (size_t j = 0; j < i; ++j)
                if (config.motor_ids[j] == config.motor_ids[i])
                {
                    error = tag + ": duplicate motor ID " + std::to_string(config.motor_ids[i]);
                    return false;
                }
        }
    }
    return true;
}

bool GoMotorArray::start(std::string &error)
{
    if (started_)
    {
        error = "already started";
        error_ = error;
        return false;
    }
    if (!validate(error))
    {
        error_ = error;
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        unload_requested_=false;
        unload_reason_.clear();
        target_seq_ = 0;
        clamp_count_ = 0;
        target_time_ = std::chrono::steady_clock::now();
        for (const std::unique_ptr<Worker> &worker : workers_)
        {
            worker->ready = false;
            worker->failed = false;
            worker->stale = false;
            worker->error.clear();
            worker->applied_seq = 0;
            worker->cycle_count = 0;
            worker->bus_ms = 0.0;
            worker->initialized = false;
            worker->pd_active = false;
            worker->stale_latched = false;
        }
    }

    // 每条总线一个定周期线程：第一轮做 open + 探测，之后每轮跑一次收发；
    // 退出钩子在自己线程里关串口（卸力）。
    for (size_t i = 0; i < workers_.size(); ++i)
    {
        Worker &worker = *workers_[i];
        const std::string name = "bus" + std::to_string(i) + " " + worker.config.device;
        worker.loop.reset(new Loop(name, period_, [this, i] { workerBody(i); },
                                   [this, i] { closeWorker(i); }));
        worker.loop->start();
    }

    // 等所有总线完成“第一轮 = 启动探测”。用带超时的等待，避免某条总线的
    // 线程没能报到时 start() 永久挂住。
    error.clear();
    {
        std::unique_lock<std::mutex> lock(mutex_);
        const bool reported = cv_.wait_for(lock, start_timeout_, [this] {
            for (const std::unique_ptr<Worker> &worker : workers_)
                if (!worker->ready && !worker->failed)
                    return false;
            return true;
        });
        if (!reported)
            error = "startup probe did not report within " +
                    std::to_string(start_timeout_.count()) + " ms";
        else
            for (const std::unique_ptr<Worker> &worker : workers_)
                if (!worker->ready)
                {
                    error = worker->error.empty() ? "startup probe failed" : worker->error;
                    break;
                }
    }

    if (!error.empty())
    {
        error_ = error;
        stop();
        return false;
    }

    // 到这里初始化那一轮（打开串口 + 启动探测）已经全部做完。把各总线的循环质量
    // 统计清零，让 body_max/interval_max 只反映稳定运行期——否则它们会被那次
    // 一次性开销带偏，看上去像“抖动很大”。
    for (const std::unique_ptr<Worker> &worker : workers_)
        if (worker->loop)
            worker->loop->resetStats();

    started_ = true;
    stopped_ = false;
    error_.clear();
    return true;
}

void GoMotorArray::workerBody(size_t index)
{
    Worker &worker = *workers_[index];
    if (!worker.initialized)
    {
        worker.initialized = true;
        if (!initWorker(worker))
        {
            if (worker.loop)
                worker.loop->stop(); // 初始化失败：不再跑下一轮（退出钩子仍会执行）
            return;
        }
    }
    roundOnce(worker);
}

void GoMotorArray::closeWorker(size_t index)
{
    Worker &worker = *workers_[index];
    if (worker.bus)
        worker.bus->close(); // 三轮逐台零指令卸力，然后关串口
}

bool GoMotorArray::initWorker(Worker &worker)
{
    const std::string tag = "bus " + worker.config.device;
    bool ok = false;
    try
    {
        worker.bus.reset(
            new go::GoBus(worker.config.device, baud_, io_timeout_us_, command_timeout_ms_));
        worker.bus->bindStopFlag(&unload_requested_);
        for (size_t i = 0; i < worker.config.motor_ids.size(); ++i)
            worker.motors.push_back(&worker.bus->addMotor(worker.config.motor_ids[i],
                                                          0.0f, // 零点由启动自动归零设置
                                                          worker.config.motor_directions[i]));

        // 启动探测：逐台发 FOC 全零控制量，全部反馈健康才算就绪（电机上电进入 FOC 待机）
        if (!worker.bus->open() || !worker.bus->cycle())
        {
            std::lock_guard<std::mutex> lock(mutex_);
            worker.failed = true;
            worker.error = tag + ": " + worker.bus->lastError();
        }
        else
        {
            // 启动自动归零（**每次初始化都做**）：令每台的逻辑位置 = start_pose_rad。
            // 编码器参考每次上电都会变，所以每次上电都必须重设一次 offset ——
            // 等价于把"每次手工跑 go_probe 贴 offset"自动化。
            // start_pose_rad 就是 config 里 offset 行中"数字之后"的部分（± 宏）：
            // offset = 本次实测 − direction × start_pose_rad。
            if (!worker.config.start_pose_rad.empty()) // 留空 = 这条总线不做自动归零
            {
                for (size_t i = 0; i < worker.motors.size(); ++i)
                {
                    go::GoMotor *motor = worker.motors[i];
                    const float want = worker.config.start_pose_rad[i];
                    worker.zero_shifts[i] = motor->feedback().q - want; // 本次和期望差多少
                    if (!motor->alignTo(want))
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        worker.failed = true;
                        worker.error = tag + ": 自动归零失败（motor " +
                                       std::to_string(motor->id()) + "）: " + motor->lastError();
                        cv_.notify_all();
                        return false;
                    }
                }
            }
            ok = true;
            std::lock_guard<std::mutex> lock(mutex_);
            worker.ready = true;
            for (size_t i = 0; i < worker.motors.size(); ++i) {
                worker.feedback[i] = worker.motors[i]->feedback();
                worker.receive_time_ns[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    worker.motors[i]->lastFeedbackTime().time_since_epoch()).count();
            }
        }
    }
    catch (const std::exception &e)
    {
        // 线程里抛出去会 std::terminate，所以转成该总线的错误状态
        std::lock_guard<std::mutex> lock(mutex_);
        worker.failed = true;
        worker.error = tag + ": exception: " + e.what();
    }
    cv_.notify_all();
    return ok;
}

void GoMotorArray::roundOnce(Worker &worker)
{
    const std::string tag = "bus " + worker.config.device;

    // 1) 取本轮序号。只有序号变了才重新提交目标——这样库里的“命令有效期”才能正常计时；
    //    把同一份旧目标反复提交会刷新有效期，等于用旧数据骗过超时保护。
    uint64_t seq = 0;
    std::chrono::steady_clock::time_point target_time{};
    bool fresh = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        seq = target_seq_;
        target_time = target_time_;
        if (seq != worker.applied_seq)
        {
            worker.applied = worker.commands; // 等长预分配，不会新分配内存
            fresh = true;
        }
    }

    bool round_ok = true;
    std::string round_error;
    // stale 的取值见下面的 worker.stale_latched（要锁存到上层恢复为止）
    if (fresh)
    {
        for (size_t i = 0; i < worker.motors.size(); ++i)
        {
            const go::Command &cmd = worker.applied[i];
            if (!worker.motors[i]->setPD(cmd.kp, cmd.kd, cmd.q, cmd.dq, cmd.tau))
            {
                round_ok = false;
                round_error = tag + ": motor " + std::to_string(worker.config.motor_ids[i]) +
                              ": " + worker.motors[i]->lastError();
                break; // 仍然往下跑 cycle()，让库执行零指令与故障流程
            }
        }
        worker.pd_active = round_ok; // 挂上 PD 目标了，此后交给看门狗盯
        worker.stale_latched = false; // 上层又说话了 → 解除“失联”状态
    }
    else if (worker.pd_active && watchdog_ > std::chrono::milliseconds::zero())
    {
        // 上层失联看门狗：目标太旧就主动卸力。
        // 这里用 setZero() 而不是干等库判“命令过期”，是为了不被锁成需要人工恢复的故障：
        // setZero 会清掉活动目标，上层一恢复提交目标就自动重新接管。
        const auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - target_time);
        if (age >= watchdog_)
        {
            for (go::GoMotor *motor : worker.motors)
                motor->setZero();
            worker.pd_active = false;
            worker.stale_latched = true; // 锁存：直到上层重新提交目标才解除
            round_error = tag + ": no new target for " + std::to_string(age.count()) +
                          " ms (watchdog " + std::to_string(watchdog_.count()) +
                          " ms): unloaded";
        }
    }
    const bool stale = worker.stale_latched;

    // 2) 一轮收发（没有新目标时 cycle() 会继续发送上一次的目标）
    const auto cycle_start = go::SerialPort::Clock::now();
    const bool cycle_ok = worker.bus->cycle();
    const auto cycle_end = go::SerialPort::Clock::now();

    if (round_ok && !cycle_ok)
    {
        round_ok = false;
        round_error = tag + ": " + worker.bus->lastError();
    }

    // 3) 发布反馈与本轮结果
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t i = 0; i < worker.motors.size(); ++i) {
            worker.feedback[i] = worker.motors[i]->feedback();
            worker.receive_time_ns[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(
                worker.motors[i]->lastFeedbackTime().time_since_epoch()).count();
        }
        worker.bus_ms = std::chrono::duration<double, std::milli>(cycle_end - cycle_start).count();
        worker.applied_seq = seq; // 含这个序号的这一轮已经做完
        ++worker.cycle_count;
        worker.failed = !round_ok;
        worker.stale = stale;
        if (!round_ok)
            worker.error = round_error; // 真故障
        else if (stale)
            worker.error = round_error; // 只是“上层失联已卸力”的提示，faulted 仍为 false
        else
            worker.error.clear();
    }
    cv_.notify_all();
}

uint64_t GoMotorArray::publishTargets(const Targets &targets)
{
    std::lock_guard<std::mutex> lock(mutex_);
    size_t flat = 0;
    for (const std::unique_ptr<Worker> &worker : workers_)
    {
        for (size_t i = 0; i < worker->commands.size(); ++i, ++flat)
        {
            go::Command &cmd = worker->commands[i];
            cmd.reset(); // 全零 + FOC 模式，再逐项填入本轮目标
            cmd.id = static_cast<uint8_t>(worker->config.motor_ids[i]);
            cmd.kp = targets.kp[flat];
            cmd.kd = targets.kd[flat];
            cmd.q = targets.q[flat];
            cmd.dq = targets.dq[flat];
            cmd.tau = targets.tau[flat];
        }
    }
    ++target_seq_;
    target_time_ = std::chrono::steady_clock::now(); // 看门狗据此判断“上层多久没说话了”
    return target_seq_;
}

bool GoMotorArray::targetsValid(const Targets &targets, std::string &error) const
{
    const size_t n = motor_count_;
    if (targets.kp.size() != n || targets.kd.size() != n || targets.q.size() != n ||
        targets.dq.size() != n || targets.tau.size() != n)
    {
        error = "every target vector must have length " + std::to_string(n);
        return false;
    }
    return true;
}

bool GoMotorArray::setTargets(const Targets &targets)
{
    if (unload_requested_) { error_="whole robot stop latched"; return false; }
    if (!started_ || stopped_)
    {
        error_ = "controller is not running";
        return false;
    }
    std::string error;
    if (!targetsValid(targets, error))
    {
        error_ = error;
        return false;
    }

    // Safety 层：先限幅再发。出现 NaN / Inf 就整份拒绝——把它夹成边界等于把
    // “上层算错了”变成“朝边界猛冲”，更危险。
    Targets safe = targets;
    const ClampResult clamped = Safety::clamp(safe, limits_);
    if (clamped == ClampResult::Rejected)
    {
        error_ = "target rejected by Safety: every value must be finite";
        return false;
    }
    if (clamped == ClampResult::Clamped)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++clamp_count_; // 让上层能从 state().clampedCount 看出自己顶到限幅了
    }

    publishTargets(safe);
    cv_.notify_all();
    error_.clear();
    return true;
}

bool GoMotorArray::setTargets(const std::vector<float> &kp, const std::vector<float> &kd,
                              const std::vector<float> &q, const std::vector<float> &dq,
                              const std::vector<float> &tau)
{
    Targets targets;
    targets.kp = kp;
    targets.kd = kd;
    targets.q = q;
    targets.dq = dq;
    targets.tau = tau;
    return setTargets(targets);
}

State GoMotorArray::state() const
{
    State out;
    std::lock_guard<std::mutex> lock(mutex_);

    out.faulted = unload_requested_;
    if (unload_requested_) out.error = unload_reason_;
    out.feedback.reserve(motor_count_);
    out.q.reserve(motor_count_);
    out.targetSeq = target_seq_;
    out.clampedCount = clamp_count_;
    out.targetAgeMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - target_time_)
                          .count();
    bool first = true;
    for (const std::unique_ptr<Worker> &worker : workers_)
    {
        out.feedback.insert(out.feedback.end(), worker->feedback.begin(), worker->feedback.end());
        out.receive_time_ns.insert(out.receive_time_ns.end(), worker->receive_time_ns.begin(), worker->receive_time_ns.end());
        for (const go::Feedback &feedback : worker->feedback)
            out.q.push_back(feedback.q);

        if (first)
        {
            out.cycleCount = worker->cycle_count;
            first = false;
        }
        else if (worker->cycle_count < out.cycleCount)
            out.cycleCount = worker->cycle_count; // 取最小值 = “每一路都至少跑了这么多轮”

        if (worker->stale)
            out.stale = true;

        if (worker->failed)
        {
            out.faulted = true;
            if (out.error.empty())
                out.error = worker->error; // 只把“真故障”的原因带出去
        }
    }
    if (!out.faulted && !error_.empty())
        out.error = error_; // 上层调用错误（长度不符 / 被 Safety 拒绝）：报原因，但总线没坏
    return out;
}

LoopStats GoMotorArray::loopStats() const
{
    LoopStats out;
    // worker->loop 只由上层线程创建/销毁，而 Loop 的统计量都是原子量，所以这里不用加锁。
    for (const std::unique_ptr<Worker> &worker : workers_)
    {
        if (!worker->loop)
            continue;
        const long long interval_min = worker->loop->minIntervalMicros();
        if (interval_min > 0 && (out.intervalMinUs == 0 || interval_min < out.intervalMinUs))
            out.intervalMinUs = interval_min;
        if (worker->loop->maxIntervalMicros() > out.intervalMaxUs)
            out.intervalMaxUs = worker->loop->maxIntervalMicros();
        if (worker->loop->maxBodyMicros() > out.bodyMaxUs)
            out.bodyMaxUs = worker->loop->maxBodyMicros();
        if (worker->loop->maxGapMicros() > out.gapMaxUs)
            out.gapMaxUs = worker->loop->maxGapMicros();
    }
    return out;
}

std::vector<go::Feedback> GoMotorArray::feedback() const
{
    return state().feedback;
}

std::vector<float> GoMotorArray::zeroShifts() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<float> out;
    out.reserve(motor_count_);
    for (const std::unique_ptr<Worker> &worker : workers_)
        out.insert(out.end(), worker->zero_shifts.begin(), worker->zero_shifts.end());
    return out;
}

float GoMotorArray::maxZeroShift() const
{
    float worst = 0.0f;
    for (float shift : zeroShifts())
        worst = std::max(worst, std::fabs(shift));
    return worst;
}

std::vector<float> GoMotorArray::positions() const
{
    return state().q;
}

std::vector<std::string> GoMotorArray::motorDevices() const
{
    std::vector<std::string> out;
    out.reserve(motor_count_);
    for (const std::unique_ptr<Worker> &worker : workers_)
        for (size_t i = 0; i < worker->config.motor_ids.size(); ++i)
            out.push_back(worker->config.device);
    return out;
}

std::vector<std::string> GoMotorArray::motorLabels() const
{
    std::vector<std::string> out;
    out.reserve(motor_count_);
    for (const std::unique_ptr<Worker> &worker : workers_)
        for (int id : worker->config.motor_ids)
            out.push_back(worker->config.device + "/ID" + std::to_string(id));
    return out;
}

double GoMotorArray::busMs(size_t bus_index) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (bus_index >= workers_.size())
        return 0.0;
    return workers_[bus_index]->bus_ms;
}

bool GoMotorArray::faulted() const
{
    // 刻意不调用 state()：那条路径会拷一份 12 台电机的反馈，
    // 而这个函数会被上层每个周期调用，只该读标志。
    // 也刻意不看 error_：那个字段记的是“上层调用出错”（长度不符、目标被 Safety 拒绝），
    // 总线本身没坏，不该让上层误以为要收工。
    std::lock_guard<std::mutex> lock(mutex_);
    if (unload_requested_) return true;
    for (const std::unique_ptr<Worker> &worker : workers_)
        if (worker->failed)
            return true;
    return false;
}

std::string GoMotorArray::lastError() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (unload_requested_) return unload_reason_;
    for (const std::unique_ptr<Worker> &worker : workers_)
        if (worker->failed)
            return worker->error;
    return error_;
}

std::string GoMotorArray::statusText() const
{
    std::string out;
    char line[192];
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t b = 0; b < workers_.size(); ++b)
    {
        const Worker &worker = *workers_[b];

        std::string flags;
        if (worker.stale)
            flags += "  STALE";
        if (worker.failed)
            flags += "  FAILED";
        // bus = 最近一轮的耗时（瞬时）；max = 历史最慢一轮（串口的锅）；
        // gap = 醒来比计划最晚多少（调度器的锅）。后两个来自 Loop 的累计统计。
        const double max_body_ms =
            worker.loop ? double(worker.loop->maxBodyMicros()) / 1000.0 : 0.0;
        const double max_gap_ms =
            worker.loop ? double(worker.loop->maxGapMicros()) / 1000.0 : 0.0;
        std::snprintf(line, sizeof(line), "%s  bus=%.2fms  max=%.2fms  gap=%.2fms  n=%llu%s\n",
                      worker.config.device.c_str(), worker.bus_ms, max_body_ms, max_gap_ms,
                      static_cast<unsigned long long>(worker.cycle_count), flags.c_str());
        out += line;

        for (const go::Feedback &feedback : worker.feedback)
        {
            std::string err_text;
            if (feedback.error)
                err_text = std::string(" (") + go::motorErrorText(feedback.error) + ")";
            std::snprintf(line, sizeof(line), "  ID%-2u q=%+6.2f  dq=%+6.2f  tau=%+6.2f  T=%d  err=%u%s\n",
                          unsigned(feedback.id), double(feedback.q), double(feedback.dq),
                          double(feedback.tau), int(feedback.temp), unsigned(feedback.error),
                          err_text.c_str());
            out += line;
        }
        if (worker.failed && !worker.error.empty())
            out += "  ! " + worker.error + "\n";
        out += "\n"; // 每条总线结束空一行
    }
    return out;
}

void GoMotorArray::stop()
{
    if (stopped_)
        return;
    stopped_ = true;

    for (const std::unique_ptr<Worker> &worker : workers_)
        if (worker->loop)
            worker->loop->stop();
    cv_.notify_all(); // 放行可能还在等状态变化的人

    // 各线程在自己的退出钩子里关串口（卸力），这里只负责等它们结束
    for (const std::unique_ptr<Worker> &worker : workers_)
        if (worker->loop)
            worker->loop->join();

    for (const std::unique_ptr<Worker> &worker : workers_)
        worker->loop.reset();

    started_ = false;
}

} // namespace go

namespace go {
void GoMotorArray::requestUnload(const std::string& reason) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!unload_requested_) unload_reason_=reason;
    unload_requested_=true;
}
}
