#pragma once
#include "go_protocol.h"
#include "serial_port.h"
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
namespace go {
struct TransportStats { size_t bytesIn=0, badFrames=0, strayFrames=0; };
// 一次"电机事务"的三段耗时（微秒）。只用于诊断，不参与任何判决：
//   flush_us = 清空接收缓冲（tcflush）耗时
//   write_us = 把 17 字节命令写进驱动（writeUntil）耗时
//   reply_us = 从写完到拿到该 ID 的合法应答（含等待、收字节、解析）
struct TransactionTiming {
    std::uint32_t flush_us = 0;      // 写之前清接收缓冲（tcflush）
    std::uint32_t write_us = 0;      // 把 17 字节写进驱动
    std::uint32_t first_byte_us = 0; // 写完命令 → 收到第一个字节（从 receive 起点算，误差 <5us）
    std::uint32_t reply_us = 0;      // 写完命令 → 收完并解析出该 ID 的合法帧（含首字节）
};
class MotorTransport {
public:
    static bool receive(SerialPort& port, int id, SerialPort::Deadline end,
                        Feedback& out, TransportStats& stats, std::string& error,
                        std::uint32_t* first_byte_us=nullptr) {
        std::array<uint8_t,512> buffer{};
        size_t used=0;
        const auto began=SerialPort::Clock::now();
        bool got_first=false;
        for (;;) {
            // 先解析已缓冲的字节，再判超时：截止前最后一刻到达的完整帧也必须被解析
            while (findFrame(buffer.data(),used)) {
                Feedback next;
                const bool valid=unpackFeedback(buffer.data(),kReplyLen,next);
                const size_t consumed=valid?kReplyLen:1;
                if (!valid) ++stats.badFrames;
                std::memmove(buffer.data(),buffer.data()+consumed,used-consumed);
                used-=consumed;
                if (valid) {
                    if (next.id==id) { out=next; error.clear(); return true; }
                    ++stats.strayFrames;
                }
            }
            if (SerialPort::Clock::now()>=end) break;
            const auto count=port.readUntil(buffer.data()+used,buffer.size()-used,end);
            if (!count) { error=port.lastError(); return false; }
            if (!got_first) {
                got_first=true;
                if (first_byte_us)
                    *first_byte_us=static_cast<std::uint32_t>(std::chrono::duration_cast<
                        std::chrono::microseconds>(SerialPort::Clock::now()-began).count());
            }
            used+=count; stats.bytesIn+=count;
        }
        error="feedback timeout"; return false;
    }
    // flush=true：写之前清接收缓冲（一问一答模式必须）。流水线（连发 N 条）时必须传 false，
    // 否则会把前一条命令已经收到的应答清掉。
    static bool send(SerialPort& port,const Command& command,SerialPort::Deadline end,
                     std::string& error,TransactionTiming* timing=nullptr,bool flush=true) {
        uint8_t frame[kCmdLen];
        if (!packCommand(command,frame)) { error="invalid command encoding"; return false; }
        const auto start=SerialPort::Clock::now();
        const bool flushed=flush ? port.discardInput() : true;
        const auto flushed_at=SerialPort::Clock::now();
        if (!flushed) { error=port.lastError(); return false; }
        const bool written=port.writeUntil(frame,kCmdLen,end)==kCmdLen;
        const auto written_at=SerialPort::Clock::now();
        if (timing) {
            timing->flush_us=static_cast<std::uint32_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(flushed_at-start).count());
            timing->write_us=static_cast<std::uint32_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(written_at-flushed_at).count());
        }
        if (!written) { error=port.lastError(); return false; }
        return true;
    }
    static bool exchange(SerialPort& port,const Command& command,SerialPort::Deadline end,
                         Feedback& feedback,TransportStats& stats,std::string& error,
                         TransactionTiming* timing=nullptr) {
        if (timing) *timing=TransactionTiming{};
        if (!send(port,command,end,error,timing)) return false;
        const auto sent_at=SerialPort::Clock::now();
        std::uint32_t first_us=0;
        const bool ok=receive(port,command.id,end,feedback,stats,error,&first_us);
        if (timing) {
            timing->first_byte_us=first_us;
            timing->reply_us=static_cast<std::uint32_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(SerialPort::Clock::now()-sent_at).count());
        }
        return ok;
    }
};
} // namespace go
