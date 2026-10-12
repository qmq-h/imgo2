#include "go_protocol.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#define CHECK(x) do { if(!(x)) throw std::runtime_error(#x); } while(0)
static void seal(uint8_t* p) { auto c=go::crc16(p,14); p[14]=uint8_t(c); p[15]=uint8_t(c>>8); }
static bool near(double a,double b,double e=1e-5) { return std::abs(a-b)<e; }
int main() {
    const uint8_t digits[]="123456789";
    CHECK(go::crc16(digits,9)==0x2189); // CRC-16/KERMIT known check value.
    go::Command c; uint8_t p[17];
    c.q=c.dq=c.tau=c.kp=c.kd=1;
    const uint8_t expected[]={0xfe,0xee,0x10,0x28,0,1,1,0xf4,0x80,0,0,0x1f,0,0x1f,0,0x6a,0x8d};
    CHECK(go::packCommand(c,p)); CHECK(std::memcmp(p,expected,17)==0);
    c.q=c.dq=c.tau=-1;
    CHECK(go::packCommand(c,p));
    CHECK(p[3]==0xd8 && p[4]==0xff && p[5]==0xff && p[6]==0xfe);
    CHECK(p[7]==0x0c && p[8]==0x7f && p[9]==0xff && p[10]==0xff);
    // Raw rotor: torque=1 Nm, speed=1 rev/s, position=1 rev; error=5, force=4095.
    uint8_t reply[]={0xfd,0xee,0x10,0,1,0,1,0,0x80,0,0,0xfb,0xfd,0x7f,0,0};
    seal(reply); go::Feedback f;
    CHECK(go::unpackFeedback(reply,16,f));
    CHECK(near(f.tau,6.33) && near(f.q,6.283185307179586/6.33) && near(f.dq,f.q));
    CHECK(f.temp==-5 && f.error==5 && f.force==4095);
    reply[0]=0xfe; seal(reply); CHECK(!go::unpackFeedback(reply,16,f) && !f.valid);
    reply[0]=0xfd; seal(reply); reply[14]^=1; CHECK(!go::unpackFeedback(reply,16,f));
    seal(reply); CHECK(!go::unpackFeedback(reply,15,f));
    reply[2]=0x1f; seal(reply); CHECK(!go::unpackFeedback(reply,16,f));
    // Negative raw feedback, including signed 32-bit minimum.
    reply[2]=0x10; reply[3]=0; reply[4]=0xff; reply[5]=0; reply[6]=0xff;
    reply[7]=reply[8]=reply[9]=0; reply[10]=0x80; seal(reply);
    CHECK(go::unpackFeedback(reply,16,f)); CHECK(f.q<0 && f.dq<0 && near(f.tau,-6.33));
    for(int field=0;field<5;++field) {
        for(float invalid:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()}) {
            c=go::Command{}; float* fields[]={&c.q,&c.dq,&c.tau,&c.kp,&c.kd}; *fields[field]=invalid;
            CHECK(!go::packCommand(c,p)); for(auto b:p) CHECK(b==0);
        }
    }
    c=go::Command{}; c.id=16; CHECK(!go::packCommand(c,p)); c.id=15; CHECK(go::packCommand(c,p));
    c.mode=go::MotorMode(3); CHECK(!go::packCommand(c,p));
    c=go::Command{}; c.q=c.dq=c.tau=c.kp=c.kd=std::numeric_limits<float>::max();
    CHECK(go::packCommand(c,p)); CHECK((p[4]&0x80)==0 && (p[6]&0x80)==0 && (p[10]&0x80)==0);
    CHECK(p[11]==0xfe && p[12]==0x7f); // 25.599 * 1280 -> 32766
    c.q=c.dq=c.tau=-std::numeric_limits<float>::max(); c.kp=c.kd=-1;
    CHECK(go::packCommand(c,p)); CHECK((p[4]&0x80) && (p[6]&0x80) && (p[10]&0x80)); CHECK(p[11]==0 && p[13]==0);
    // Output kp=60 must not be clamped at 25.599 before gear conversion.
    c=go::Command{}; c.kp=60; CHECK(go::packCommand(c,p)); CHECK((p[11]|int(p[12])<<8)==1916);
    // Measure decoded wire PD law against output-side PD, allowing quantization.
    c.q=0.8f; c.dq=-0.3f; c.tau=1.7f; c.kd=2;
    CHECK(go::packCommand(c,p));
    auto i16=[&](int i) { int v=p[i]|int(p[i+1])<<8; return v<32768?v:v-65536; };
    double r=go::kGearRatio, pi2=6.283185307179586;
    double qr=(p[7]|uint32_t(p[8])<<8|uint32_t(p[9])<<16|uint32_t(p[10])<<24)*pi2/32768;
    double rotor=i16(3)/256.0+(p[11]|int(p[12])<<8)/1280.0*(qr-0.6*r)+
        (p[13]|int(p[14])<<8)/1280.0*(i16(5)*pi2/256-0.1*r);
    CHECK(near(rotor*r,1.7+60*(0.8-0.6)+2*(-0.3-0.1),0.08));
    uint8_t buf[32]={9,8,0xfd}; size_t len=3;
    CHECK(!go::findFrame(buf,len) && len==1 && buf[0]==0xfd);
    std::memcpy(buf+1,reply+1,15); len=16; CHECK(go::findFrame(buf,len));
    std::cout<<"Protocol, units, limits and PD equivalence: PASS\n";
}
