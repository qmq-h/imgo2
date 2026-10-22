#include "function/my_myth.h"
#include "go_safety.h"
#include "go_motor_array.h"
#include <cmath>
#include <iostream>
#include <limits>
#define CHECK(x) do {if(!(x)){std::cerr<<#x<<'\n';return 1;}}while(0)
int main() {
    CHECK(std::abs(my_myth::degToRad(180)-my_myth::pi)<1e-12);
    CHECK(std::abs(my_myth::radToDeg(my_myth::pi)-180)<1e-12);
    CHECK(my_myth::clamp(3,0,2)==2 && my_myth::clamp(-1,0,2)==0);
    CHECK(my_myth::quinticProgress(4,8)==0.5);
    CHECK(my_myth::quinticRate(4,8)==1.875/8);
    CHECK(my_myth::quinticRate(0,8)==0 && my_myth::quinticRate(8,8)==0);
    go::MotorConfig motor;
    CHECK(motor.baud==4000000 && motor.io_timeout_us==20000 && motor.command_timeout_ms==100);
    // period=5 ms：流水线（pipeline=true）一轮约 2.5 ms，5 ms 节拍有余量；
    // 若关掉流水线（逐台一问一答，一轮 6~7 ms），这里要相应改成 7 ms 或更大。
    CHECK(motor.period==std::chrono::milliseconds(5) && motor.watchdog==std::chrono::milliseconds(80));
    CHECK(motor.pipeline && motor.pipeline_gap_us==150);
    const char* paths[]={"/dev/robot/leg_fl","/dev/robot/leg_fr","/dev/robot/leg_bl","/dev/robot/leg_br"};
    const float directions[][3]={{1,1,1},{1,-1,-1},{-1,1,1},{-1,-1,-1}};
    CHECK(motor.buses.size()==4 && motor.limits.size()==12);
    for(size_t b=0;b<4;++b) {
        CHECK(motor.buses[b].device==paths[b]);CHECK(motor.buses[b].motor_ids==std::vector<int>({0,1,2}));
        for(size_t j=0;j<3;++j) CHECK(motor.buses[b].motor_directions[j]==directions[b][j]);
        const float sign=(b%2)?-1:1;
        CHECK(motor.buses[b].start_pose_rad[0]==sign*3.49f*my_myth::degrees_to_radians_f);
        CHECK(motor.buses[b].start_pose_rad[1]==67.43f*my_myth::degrees_to_radians_f);
        CHECK(motor.buses[b].start_pose_rad[2]==-165.01f*my_myth::degrees_to_radians_f);
    }
    real_config::ImuConfig imu;imu.validate();
    CHECK(imu.device=="/dev/robot/imu" && imu.baud==460800 && imu.read_timeout_ms==100 && imu.stale_after_ms==500);
    bool rejected=false;imu.baud=0;try{imu.validate();}catch(const std::invalid_argument&){rejected=true;}CHECK(rejected);
    // 非法启动姿态在启动前被拒绝，不访问设备。
    motor.buses[0].start_pose_rad.pop_back();go::GoMotorArray bad(motor);std::string error;
    CHECK(!bad.start(error) && error.find("start_pose_rad")!=std::string::npos);
    // 调用者配置变化不改变已构造硬件对象快照；这里不启动设备。
    motor=go::MotorConfig{};go::GoMotorArray snapshot(motor);motor.buses.clear();CHECK(snapshot.motorCount()==12);
    std::cout<<"math and constructor configuration passed\n";
}
