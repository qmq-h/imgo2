#pragma once
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <vector>
namespace go {
/**
 * 一次"电机事务"（发送 + 等应答）的埋点记录。
 *
 * 目的：事后回答"这段没有新反馈的时间里，到底发生了什么"。判读方式：
 *   ok=1                    → 拿到该 ID 的合法帧
 *   ok=0, bytes=0           → 一个字节都没回来（链路/驱动没回应）
 *   ok=0, bytes>0, bad>0    → 收到字节但 CRC 不过（噪声/串扰/EMI）
 *   ok=0, bytes>0, bad=0    → 收到字节但没有该 ID 的合法帧（串线/回显/半双工冲突）
 * 若某段时间里**一条记录都没有**，说明总线线程本身没在跑（被冻结/被抢占/没被调度）。
 */
struct TraceRecord {
    std::int64_t start_ns = 0;   // 事务开始时刻（相对 trace 创建时刻的 steady_clock）
    std::uint32_t waited_us = 0; // 本次事务耗时（含发送与等待应答）
    std::uint16_t bus = 0;       // 总线序号（0..N，与 busDevices() 同序）
    std::uint8_t id = 0;         // 电机 ID
    std::uint8_t ok = 0;         // 1 = 成功拿到该 ID 的合法帧
    std::uint16_t bytes = 0;     // 本次事务收到的字节数
    std::uint16_t bad = 0;       // 本次事务里 CRC 不过的帧数
    std::uint16_t stray = 0;     // 本次事务里"别的 ID"的帧数
};

/**
 * 多线程写、停机时读的环形缓冲：每条总线由它自己的线程写，读侧只在故障/退出时快照。
 * 写入只用一次 fetch_add 占位，不在热路径上加锁；每条记录最多 24 字节，容量 16384 条约 384 KB。
 */
class TransactionTrace {
public:
    using Clock = std::chrono::steady_clock;
    static constexpr std::size_t kCapacity = 16384;

    TransactionTrace() : epoch_(Clock::now()) {}

    void record(const TraceRecord& r) {
        const auto slot = write_.fetch_add(1, std::memory_order_relaxed);
        ring_[slot % kCapacity] = r;
    }
    // 相对 trace 创建时刻的纳秒，供写入 start_ns
    std::int64_t nowNs() const {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - epoch_).count();
    }
    std::uint64_t written() const { return write_.load(std::memory_order_relaxed); }

    // 按写入顺序返回最近的 count 条（最多 kCapacity）。多线程写入时时刻可能轻微交错，
    // 调用方若需要严格时序，请按 start_ns 排序。
    std::vector<TraceRecord> snapshot(std::size_t count = kCapacity) const {
        const auto end = write_.load(std::memory_order_acquire);
        const auto begin = end > count ? end - count : 0;
        std::vector<TraceRecord> out;
        out.reserve(static_cast<std::size_t>(end - begin));
        for (auto i = begin; i < end; ++i) out.push_back(ring_[i % kCapacity]);
        return out;
    }

private:
    std::array<TraceRecord, kCapacity> ring_{};
    std::atomic<std::uint64_t> write_{0};
    Clock::time_point epoch_;
};
} // namespace go
