#include "feedback_delay.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>

void check(bool ok) { if (!ok) throw std::runtime_error("feedback delay test failed"); }
int main()
{
    using Buffer=FeedbackDelay<int>;
    const Buffer::Time epoch{};
    auto at=[&](double ms){return epoch+std::chrono::duration_cast<Buffer::Clock::duration>(
        std::chrono::duration<double,std::milli>(ms));};
    Buffer motor,imu;int v=-1;Buffer::Time arrival;
    motor.configure(1.5);imu.configure(0);
    motor.push(10,at(0));imu.push(20,at(0));
    check(!motor.read(at(1.499),v));check(v==-1);
    check(imu.read(at(0),v)&&v==20);
    motor.push(11,at(1));
    check(motor.read(at(1.5),v,&arrival)&&v==10&&arrival==at(0));
    check(motor.read(at(2.499),v)&&v==10);
    check(motor.read(at(2.5),v)&&v==11);
    check(motor.read(at(20),v)&&v==11); // hold; never fabricate an interpolated sensor frame
    motor.configure(0);check(!motor.read(at(20),v));
    motor.push(12,at(21));motor.push(13,at(22));check(motor.read(at(22),v)&&v==13);
    motor.configure(1000);
    for(int i=0;i<10000;++i) motor.push(i,at(i));
    check(motor.size()<=1001);check(motor.read(at(9999),v)&&v==8999);
    motor.push(9,at(0));check(!motor.read(at(500),v)); // rollback clears future history
    for(double bad:{-1.,1000.1,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()})
    { bool rejected=false;try{motor.configure(bad);}catch(const std::invalid_argument&){rejected=true;}check(rejected); }
    struct Packet { int q, dq, torque; };
    FeedbackDelay<Packet> packets;packets.configure(1.5);
    Packet output{};Packet input{1,2,3};packets.push(input,at(0));input.q=99;
    packets.push({4,5,6},at(1));
    check(packets.read(at(1.5),output)&&output.q==1&&output.dq==2&&output.torque==3);
    check(packets.read(at(2.5),output)&&output.q==4&&output.dq==5&&output.torque==6);
    std::cout<<"feedback_delay: fractional timing, independent channels, zero, warmup, hold, bounds passed\n";
}
