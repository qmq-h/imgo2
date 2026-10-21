// Real GoBus/GoMotor/protocol with a deterministic byte transport, no hardware.
#include "go_bus.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <cstdint>
#include <thread>
#include <vector>
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)
using Bytes = std::vector<uint8_t>;
static std::deque<Bytes> incoming, scripted;
static std::vector<Bytes> sent;
static bool autoReply = true, failWrite = false;
static int opens = 0, closes = 0, silentId = -1, hotId = -1, modeId = -1, faultId = -1;
static Bytes reply(int id) {
    Bytes b{0xfd,0xee,uint8_t((id==modeId?0:0x10)|id),0,1,0,1,0,0x80,0,0,
            uint8_t(id==hotId?81:20),uint8_t(id==faultId?2:0),0,0,0};
    auto crc = go::crc16(b.data(),14); b[14]=uint8_t(crc); b[15]=uint8_t(crc>>8); return b;
}
namespace go {
SerialPort::~SerialPort() { close(); }
bool SerialPort::open(const std::string&,uint32_t b,size_t t) {
    ++opens; fd_=1; baud_=b; timeout_us_=t; return true;
}
void SerialPort::close() { if(fd_>=0) ++closes; fd_=-1; baud_=0; }
bool SerialPort::discardInput() { incoming.clear(); return true; }
size_t SerialPort::write(const uint8_t* p,size_t n) { return writeUntil(p,n,deadline()); }
size_t SerialPort::writeUntil(const uint8_t* p,size_t n,Deadline) {
    if(failWrite) { error_="injected write failure"; return 0; }
    sent.emplace_back(p,p+n);
    if(!scripted.empty()) {
        while(!scripted.empty()) { incoming.push_back(scripted.front()); scripted.pop_front(); }
    } else if(autoReply && (p[2]&15)!=silentId) incoming.push_back(reply(p[2]&15));
    return n;
}
size_t SerialPort::readUntil(uint8_t* p,size_t n,Deadline) {
    if(incoming.empty()) { error_="injected timeout/disconnect"; return 0; }
    auto& b=incoming.front(); size_t take=std::min(n,b.size()); std::memcpy(p,b.data(),take);
    b.erase(b.begin(),b.begin()+take); if(b.empty()) incoming.pop_front(); return take;
}
}
static bool zero(const Bytes& b) {
    return b[2]>=0x10 && b[2]<=0x1e && std::all_of(b.begin()+3,b.begin()+15,[](uint8_t v){return v==0;});
}
static void throwsRegister(go::GoBus& bus, int id, float offset=0) {
    bool threw=false; try { bus.addMotor(id,offset); } catch(const std::exception&) { threw=true; } CHECK(threw);
}
int main() {
    {
        go::GoBus bus("fake");
        CHECK(!bus.open());
        auto& a=bus.addMotor(0,0.8f); auto* stable=&a;
        auto& b=bus.addMotor(1,-0.4f); auto& c=bus.addMotor(2);
        CHECK(&a==stable && bus.motorCount()==3);
        throwsRegister(bus,0); throwsRegister(bus,15); throwsRegister(bus,-1);
        throwsRegister(bus,4,std::numeric_limits<float>::infinity());
        CHECK(bus.open() && opens==1 && sent.empty());
        CHECK(bus.open() && opens==1); throwsRegister(bus,3);
        CHECK(bus.cycle()); CHECK(sent.size()==3);
        for(int i=0;i<3;++i) CHECK(zero(sent[size_t(i)]) && (sent[size_t(i)][2]&15)==i);
        CHECK(std::abs(a.feedback().q-(6.283185307179586/6.33-0.8f))<1e-5);
        CHECK(std::abs(b.feedback().q-(6.283185307179586/6.33+0.4f))<1e-5);
        CHECK(a.feedback().dq==b.feedback().dq && a.feedback().tau==b.feedback().tau);
        CHECK(a.setPD(60,2,0.2f,0.1f,1.7f)); CHECK(b.setPD(25,0.5f,-0.2f));
        auto stamp=a.lastCommandTime(); auto n=sent.size();
        CHECK(bus.cycle()); CHECK(a.lastCommandTime()==stamp);
        go::Command expected=a.command();
        expected.q+=a.offset();
        uint8_t wire[17];
        CHECK(go::packCommand(expected,wire)); CHECK(std::memcmp(wire,sent[n].data(),17)==0);
        expected=b.command();
        expected.q+=b.offset();
        CHECK(go::packCommand(expected,wire));
        CHECK(std::memcmp(wire,sent[n+1].data(),17)==0); CHECK(zero(sent[n+2]));
        // 线上角度仍按转子编码：q_rotor = q_logic × 6.33 + rotor_offset。
        const auto& frame = sent[n];
        const uint32_t raw_q = uint32_t(frame[7]) | uint32_t(frame[8])<<8 |
                               uint32_t(frame[9])<<16 | uint32_t(frame[10])<<24;
        const float rotor_q = float(raw_q) * 6.2831853f / 32768.0f;
        CHECK(std::abs(rotor_q - ((0.2f+0.8f)*6.33f)) < 0.0003f);
        CHECK(a.command().q==0.2f); // Encoding must not mutate/accumulate the logical target.
        n=sent.size(); CHECK(bus.cycle()); CHECK(sent[n]==sent[n-3]);
        a.setZero(); b.setZero(); CHECK(bus.cycle()); CHECK(zero(sent[sent.size()-3]));
        // Fragmented noise + wrong ID + bad CRC + correct response.
        Bytes good=reply(0), bad=good, other=reply(2), stream{1,0xfd,0,0xfe,0xee}; bad[3]^=1;
        for(const auto& v:{other,bad,good}) stream.insert(stream.end(),v.begin(),v.end());
        for(auto byte:stream) scripted.push_back(Bytes{byte});
        CHECK(bus.cycle() && a.lastOk());
        // Failure of ID 1 ends normal commands. All subsequent transactions are zero-only.
        CHECK(a.setPD(1,1,0)); CHECK(b.setPD(1,1,0)); CHECK(c.setPD(1,1,0));
        silentId=1; n=sent.size(); CHECK(!bus.cycle() && bus.faulted());
        CHECK((sent[n][2]&15)==0 && (sent[n+1][2]&15)==1);
        for(size_t i=n+2;i<sent.size();++i) CHECK(zero(sent[i]));
        CHECK(!b.lastOk() && !bus.clearFault()); CHECK(!a.setTorque(1));
        silentId=-1; n=sent.size(); CHECK(!bus.cycle());
        for(size_t i=n;i<sent.size();++i) CHECK(zero(sent[i]));
        CHECK(bus.clearFault() && !a.hasCommand());
        CHECK(bus.cycle()); CHECK(zero(sent[sent.size()-3]));
        CHECK(a.setTorque(1)); CHECK(bus.cycle());
        // Invalid submissions latch instead of silently retaining a previous motion command.
        CHECK(!a.setPD(1,1,std::numeric_limits<float>::quiet_NaN())); CHECK(!bus.cycle());
        CHECK(bus.clearFault());
        for(int kind=0;kind<3;++kind) {
            if(kind==0) hotId=2; else if(kind==1) modeId=2; else faultId=2;
            CHECK(!bus.cycle() && bus.faulted()); CHECK(!bus.clearFault());
            hotId=modeId=faultId=-1; CHECK(!bus.cycle()); CHECK(bus.clearFault());
        }
        failWrite=true; CHECK(!bus.cycle() && !a.lastOk()); failWrite=false;
        CHECK(!bus.cycle()); CHECK(bus.clearFault());
        n=sent.size(); bus.close(); CHECK(closes==1 && sent.size()==n+9);
        for(size_t i=n;i<sent.size();++i) CHECK(zero(sent[i]));
        CHECK(!a.lastOk() && !b.lastOk() && bus.baudrate()==0);
        CHECK(bus.open() && !a.hasCommand() && !a.lastOk()); bus.close();
    }
    {
        go::GoBus bus("watchdog",4000000,20000,8);
        auto& m=bus.addMotor(0,1); CHECK(bus.open()); CHECK(bus.cycle());
        CHECK(m.setTorque(2)); CHECK(bus.cycle());
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
        auto n=sent.size(); CHECK(!bus.cycle() && m.commandExpired());
        for(size_t i=n;i<sent.size();++i) CHECK(zero(sent[i]));
        m.setZero(); CHECK(m.faulted()); CHECK(!m.setPD(1,1,0));
        std::this_thread::sleep_for(std::chrono::milliseconds(15)); CHECK(!bus.clearFault());
        CHECK(!bus.cycle()); CHECK(bus.clearFault()); CHECK(!m.commandExpired());
        CHECK(m.setTorque(1)); CHECK(bus.cycle());
        // Explicit valid renewal keeps command alive; receive/transmit never renews it.
        CHECK(m.setTorque(1)); CHECK(bus.cycle());
    }
    {
        go::GoBus bus1("one"),bus2("two"); bus1.addMotor(0); bus2.addMotor(0);
        CHECK(bus1.open() && bus2.open()); CHECK(bus1.cycle() && bus2.cycle());
    }
    {
        // 方向系数 direction：-1 时逻辑坐标**整体取反**——位置/速度/力矩在读和写上一起反，
        // 而 offset 不反（它是原始坐标系里的零点）。这是镜像安装的左右腿能共用一份姿态的前提。
        go::GoBus bus("direction");
        auto& plus =bus.addMotor(0, 0.8f, +1.0f);
        auto& minus=bus.addMotor(1, 0.8f, -1.0f);
        bool bad=false;
        try { bus.addMotor(2,0,0.0f); } catch(const std::exception&) { bad=true; }
        CHECK(bad);                                  // 0 不是合法方向
        bad=false;
        try { bus.addMotor(2,0,2.0f); } catch(const std::exception&) { bad=true; }
        CHECK(bad);                                  // 只接受 ±1，不是缩放系数
        CHECK(bus.open() && bus.cycle());
        CHECK(plus.direction()==1.0f && minus.direction()==-1.0f);
        // 同一个原始回复：+1 原样，-1 三个量全部反号
        CHECK(plus.feedback().q  == -minus.feedback().q);
        CHECK(plus.feedback().dq == -minus.feedback().dq);
        CHECK(plus.feedback().tau== -minus.feedback().tau);
        // 写：同一个逻辑目标，线上位置关于 offset 对称（证明 offset 没被取反），速度/力矩反号
        CHECK(plus.setPD(20,1,0.3f,0.2f,1.5f) && minus.setPD(20,1,0.3f,0.2f,1.5f));
        const size_t n=sent.size(); CHECK(bus.cycle());
        const auto rotorOf=[](const Bytes& f) {
            // 位置字段是**有符号** 32 位：direction=-1 时它是负数，按无符号读会变成天文数字
            const uint32_t bits=uint32_t(f[7])|uint32_t(f[8])<<8|uint32_t(f[9])<<16|uint32_t(f[10])<<24;
            return double(int32_t(bits))*6.2831853/32768.0;
        };
        const auto word16=[](const Bytes& f,size_t i) {
            return int16_t(uint16_t(f[i])|uint16_t(f[i+1])<<8);
        };
        const double p0=rotorOf(sent[n]), p1=rotorOf(sent[n+1]);
        CHECK(std::abs((p0+p1)/2.0-0.8*6.33)<1e-3);                    // 中点就是 offset
        CHECK(std::abs(std::abs(p0-0.8*6.33)-std::abs(p1-0.8*6.33))<1e-3);  // 偏离 offset 的幅度相同
        CHECK(word16(sent[n],5)==-word16(sent[n+1],5) && word16(sent[n],5)>0); // 目标速度反号
        CHECK(word16(sent[n],3)==-word16(sent[n+1],3) && word16(sent[n],3)>0); // 前馈力矩反号
        CHECK(plus.command().q==0.3f && minus.command().q==0.3f); // 逻辑目标不被改写
        bus.close();
    }
    CHECK(opens==closes);
    std::cout<<"Shared bus ownership, offsets, direction, stream parsing, watchdog and fault recovery: PASS\n";
}
