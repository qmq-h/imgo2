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
struct TransactionTiming { std::uint32_t flush_us=0, write_us=0, reply_us=0; };
class MotorTransport {
public:
    static bool receive(SerialPort& port, int id, SerialPort::Deadline end,
                        Feedback& out, TransportStats& stats, std::string& error) {
        std::array<uint8_t,512> buffer{};
        size_t used=0;
        while (SerialPort::Clock::now()<end) {
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
            const auto count=port.readUntil(buffer.data()+used,buffer.size()-used,end);
            if (!count) { error=port.lastError(); return false; }
            used+=count; stats.bytesIn+=count;
        }
        error="feedback timeout"; return false;
    }
    static bool send(SerialPort& port,const Command& command,SerialPort::Deadline end,
                     std::string& error,TransactionTiming* timing=nullptr) {
        uint8_t frame[kCmdLen];
        if (!packCommand(command,frame)) { error="invalid command encoding"; return false; }
        const auto start=SerialPort::Clock::now();
        const bool flushed=port.discardInput();
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
        const bool ok=receive(port,command.id,end,feedback,stats,error);
        if (timing)
            timing->reply_us=static_cast<std::uint32_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(SerialPort::Clock::now()-sent_at).count());
        return ok;
    }
};
} // namespace go
