#pragma once
#include "go_protocol.h"
#include "serial_port.h"
#include <array>
#include <cstring>
#include <string>
namespace go {
struct TransportStats { size_t bytesIn=0, badFrames=0, strayFrames=0; };
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
                     std::string& error) {
        uint8_t frame[kCmdLen];
        if (!packCommand(command,frame)) { error="invalid command encoding"; return false; }
        if (!port.discardInput() || port.writeUntil(frame,kCmdLen,end)!=kCmdLen) {
            error=port.lastError(); return false;
        }
        return true;
    }
    static bool exchange(SerialPort& port,const Command& command,SerialPort::Deadline end,
                         Feedback& feedback,TransportStats& stats,std::string& error) {
        return send(port,command,end,error) && receive(port,command.id,end,feedback,stats,error);
    }
};
} // namespace go
