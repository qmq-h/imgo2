#include "go_bus.h"
#include <pty.h>
#include <unistd.h>
#include <poll.h>
#include <thread>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <csignal>
#define CHECK(x) do { if(!(x)) throw std::runtime_error(#x); } while(0)
static void signalHandler(int) {}
static std::vector<uint8_t> reply(int id) {
    std::vector<uint8_t> b{0xfd,0xee,uint8_t(0x10|id),0,1,0,1,0,0x80,0,0,25,0,0,0,0};
    auto c=go::crc16(b.data(),14); b[14]=uint8_t(c); b[15]=uint8_t(c>>8); return b;
}
int main() {
    int master,slave; char name[128]; CHECK(openpty(&master,&slave,name,nullptr,nullptr)==0);
    go::GoBus bus(name,4000000,50000); auto& m=bus.addMotor(2,0.4f); CHECK(bus.open()); ::close(slave);
    std::thread device([&] {
        uint8_t tx[17]; size_t n=0;
        while(n<17) { auto k=::read(master,tx+n,17-n); if(k<=0) return; n+=size_t(k); }
        auto b=reply(1), good=reply(2), corrupt=good; corrupt[3]^=1;
        b.insert(b.end(),corrupt.begin(),corrupt.end()); b.insert(b.end(),good.begin(),good.end());
        uint8_t noise[]={0,0xfd,0,0xfe,0xee}; ::write(master,noise,5);
        for(auto v:b) { ::write(master,&v,1); usleep(100); }
    });
    bool ok=bus.cycle(); device.join(); CHECK(ok && m.feedback().id==2);
    CHECK(m.feedback().q>0.59f && m.feedback().q<0.60f); // 2π/6.33 - 0.4（offset 现在是输出侧 rad）
    auto begin=go::SerialPort::Clock::now(); CHECK(!bus.cycle());
    CHECK(bus.faulted() && !m.lastOk());
    CHECK(go::SerialPort::Clock::now()-begin<std::chrono::milliseconds(300));
    // Drain the failed request and subsequent zero recovery attempt.
    uint8_t tmp[128]; pollfd pending{master,POLLIN,0};
    while(::poll(&pending,1,0)>0) { if(::read(master,tmp,sizeof(tmp))<=0) break; }
    bus.close();
    // close waits for each reply deadline, rather than bursting three commands.
    std::vector<uint8_t> shutdown;
    for(int i=0;i<10 && shutdown.size()<51;++i) {
        pollfd p{master,POLLIN,0}; if(::poll(&p,1,50)>0) { auto n=::read(master,tmp,sizeof(tmp)); if(n>0) shutdown.insert(shutdown.end(),tmp,tmp+n); }
    }
    CHECK(shutdown.size()==51);
    for(size_t i=0;i<51;i+=17) { CHECK(shutdown[i+2]==0x12); for(int j=3;j<15;++j) CHECK(shutdown[i+j]==0); }
    ::close(master);
    CHECK(openpty(&master,&slave,name,nullptr,nullptr)==0);
    go::SerialPort port; CHECK(port.open(name,4000000,20000)); ::close(slave);
    // Do not drain master: output buffer must fill and write must time out.
    std::vector<uint8_t> huge(4*1024*1024,0x55);
    begin=go::SerialPort::Clock::now(); CHECK(port.write(huge.data(),huge.size())<huge.size());
    CHECK(go::SerialPort::Clock::now()-begin<std::chrono::milliseconds(500));
    CHECK(!port.lastError().empty());
    std::signal(SIGALRM,signalHandler); ualarm(5000,5000);
    begin=go::SerialPort::Clock::now(); CHECK(port.read(tmp,sizeof(tmp))==0); ualarm(0,0);
    CHECK(go::SerialPort::Clock::now()-begin<std::chrono::milliseconds(200));
    ::close(master); CHECK(port.read(tmp,sizeof(tmp))==0 && !port.lastError().empty());
    port.close(); CHECK(port.baudrate()==0);
    std::cout<<"Linux PTY I/O, fragments, deadlines, EINTR, disconnect and shutdown: PASS\n";
}
