#include "fsm/deployment_states.h"
#include <cmath>
#include <iostream>
#include <limits>
#define CHECK(x) do{if(!(x)){std::cerr<<#x<<" at "<<__LINE__<<'\n';return 1;}}while(0)
using namespace imgo2_sdk;
using Mode=go_fsm::RobotFSM::Mode;
real_hardware::HardwareState hardware(const std::vector<float>& q) {
    real_hardware::HardwareState h;h.started=true;h.imu_valid=true;h.joints.resize(12);
    for(size_t i=0;i<12;++i){h.joints[i].valid=true;h.joints[i].feedback.q=q[i];}
    return h;
}
int main(int argc,char** argv) {
    CHECK(argc==3);auto p=PolicyConfig::load(argv[1]);go::MotorConfig motor;DeploymentConfig d;
    std::vector<float> initial;for(const auto& b:motor.buses) initial.insert(initial.end(),b.start_pose_rad.begin(),b.start_pose_rad.end());
    auto loaded=DeploymentConfig::load(argv[2]);CHECK(loaded.stand_pose==d.stand_pose);
    // crouch_pose 故意比“上电姿态”d.crouch_pose（= real_config 的 START_OFFSET 宏）浅：
    // 卸力发生在完全趴下之前，避免编码器零点有偏差时把关节硬顶到机械限位。用字面量
    // 锁住这个决定——数值一漂移测试就红，改它必须是有意的。
    const std::vector<float> crouch_expected={0.0609119907f,1.16687547f,-2.77996769f,
        -0.0609119907f,1.16687547f,-2.77996769f,0.0609119907f,1.16687547f,-2.77996769f,
        -0.0609119907f,1.16687547f,-2.77996769f};
    CHECK(loaded.crouch_pose==crouch_expected);
    RobotAdapter adapter(d.policy_to_hardware);const auto t=Clock::time_point{};
    go_fsm::RobotFSM locked(p,d,motor,adapter,initial,t);auto h=hardware(initial);
    CHECK(locked.update(h,{},t).kp[0]==60);CHECK(locked.request(1,h,t).empty());
    CHECK(!locked.request(0,h,t).empty());CHECK(locked.mode()==Mode::Standing);
    CHECK(locked.request(9,h,t).empty());
    auto end=locked.update(h,{},t+std::chrono::seconds(8));CHECK(locked.mode()==Mode::StandHold && end.q==p.default_pose);
    CHECK(locked.request(1,hardware(p.default_pose),t+std::chrono::seconds(9)).find("PPO锁定")!=std::string::npos);
    CHECK(!locked.request(2,h,t).empty() && locked.mode()==Mode::StandHold);
    d.mapping_confirmed=true;d.settle_s=0;
    go_fsm::RobotFSM fsm(p,d,motor,adapter,initial,t);fsm.request(0,h,t);fsm.update(h,{},t+std::chrono::seconds(8));
    h=hardware(p.default_pose);const auto enter=t+std::chrono::seconds(9);fsm.request(1,h,enter);CHECK(fsm.mode()==Mode::PPO);
    fsm.velocity('w');CHECK(fsm.velocityCommand()[0]==.1f);for(int i=0;i<20;++i)fsm.velocity('w');CHECK(fsm.velocityCommand()[0]==.5f);
    fsm.velocity(' ');CHECK(fsm.velocityCommand()[0]==0);
    PolicyResult r;r.generation=fsm.generation();r.sequence=1;r.generated=enter;r.observed=enter;r.actions.assign(12,0);r.command=convertAction(r.actions,p);
    CHECK(fsm.update(h,r,enter).kp[0]==60); // no gain jump at handover
    r.generated=enter+std::chrono::milliseconds(500);r.sequence=2;r.observed=r.generated;
    CHECK(fsm.update(h,r,r.generated).kp[0]==25);
    fsm.request(9,h,r.generated);CHECK(fsm.mode()==Mode::Crouching && fsm.velocityCommand()[0]==0);
    CHECK(fsm.update(h,r,r.generated).kp[0]==25); // starts from policy gains
    auto end_time=r.generated+std::chrono::seconds(8);
    auto down=fsm.update(h,{},end_time);CHECK(fsm.mode()==Mode::Crouching && down.q==initial && down.kp[0]==60);
    // Trajectory time alone cannot release. Actual measured arrival must persist.
    h=hardware(initial);fsm.update(h,{},end_time+std::chrono::milliseconds(10));
    CHECK(fsm.mode()==Mode::Crouching);
    down=fsm.update(h,{},end_time+std::chrono::milliseconds(220));
    CHECK(fsm.mode()==Mode::Passive && down.q==initial && down.kp[0]==0 && down.kd[0]==0);
    fsm.update(h,{},end_time+std::chrono::seconds(1));CHECK(fsm.mode()==Mode::Passive);
    CHECK(fsm.request(1,h,end_time).empty());
    CHECK(!fsm.request(0,h,end_time+std::chrono::seconds(1)).empty());
    CHECK(fsm.mode()==Mode::Standing);
    CHECK(fsm.update(h,{},end_time+std::chrono::seconds(1)).kp[0]==0);
    h=hardware(p.default_pose);
    // Fresh FSM: stale generation cannot enter command stream, missing result faults.
    go_fsm::RobotFSM timeout(p,d,motor,adapter,initial,t);timeout.request(0,hardware(initial),t);timeout.update(h,{},t+std::chrono::seconds(8));timeout.request(1,h,enter);
    r.generation=timeout.generation()-1;r.generated=enter+std::chrono::milliseconds(50);
    timeout.update(h,r,r.generated);CHECK(timeout.mode()==Mode::PPO);
    timeout.update(h,r,enter+std::chrono::milliseconds(100));CHECK(timeout.mode()==Mode::Fault);
    CHECK(timeout.request(0,h,enter).empty());
    go_fsm::RobotFSM invalid_imu(p,d,motor,adapter,initial,t);
    auto broken=hardware(initial);broken.imu.data.quaternion[0]=0;
    invalid_imu.update(broken,{},t);CHECK(invalid_imu.mode()==Mode::Fault);
    auto zero=timeout.update(h,{},enter);for(float k:zero.kp)CHECK(k==0);
    // Independent fixed targets (hardware order), policy reference remains untouched.
    auto custom=d;custom.stand_pose[0]=.1f;custom.crouch_pose[1]=1.1f;
    go_fsm::RobotFSM poses(p,custom,motor,adapter,initial,t);
    poses.request(0,hardware(initial),t);
    CHECK(poses.update(h,{},t+std::chrono::seconds(8)).q==custom.stand_pose);
    poses.request(9,hardware(custom.stand_pose),t+std::chrono::seconds(9));
    CHECK(poses.update(h,{},t+std::chrono::seconds(17)).q==custom.crouch_pose);
    CHECK(poses.mode()==Mode::Crouching);
    poses.update(h,{},t+std::chrono::seconds(20));CHECK(poses.mode()==Mode::Fault);
    CHECK(poses.error().find("crouch arrival timeout")!=std::string::npos);
    go_fsm::RobotFSM arrival(p,custom,motor,adapter,initial,t);
    arrival.request(0,hardware(initial),t);arrival.update(h,{},t+std::chrono::seconds(8));
    arrival.request(9,hardware(custom.stand_pose),t+std::chrono::seconds(9));
    auto reached=hardware(custom.crouch_pose);reached.joints[0].feedback.dq=.6f;
    const auto finish=t+std::chrono::seconds(17);
    arrival.update(reached,{},finish);CHECK(arrival.mode()==Mode::Crouching);
    reached.joints[0].feedback.dq=0;arrival.update(reached,{},finish+std::chrono::milliseconds(300));
    reached.joints[0].feedback.q+=.06f;arrival.update(reached,{},finish+std::chrono::milliseconds(450));
    reached=hardware(custom.crouch_pose);arrival.update(reached,{},finish+std::chrono::milliseconds(500));
    arrival.update(reached,{},finish+std::chrono::milliseconds(690));CHECK(arrival.mode()==Mode::Crouching);
    auto released=arrival.update(reached,{},finish+std::chrono::milliseconds(710));CHECK(arrival.mode()==Mode::Passive);
    for(const auto* values:{&released.kp,&released.kd,&released.tau,&released.dq}) for(float value:*values) CHECK(value==0);
    auto invalid=custom;invalid.stand_pose[0]=9;
    bool rejected=false;try{go_fsm::RobotFSM bad(p,invalid,motor,adapter,initial,t);}catch(const std::exception&){rejected=true;}CHECK(rejected);
    // 站起/趴下时长在 yaml 里被调成 4 s（匹配实际动作节奏与总线耗时），PPO 返回仍是 2 s。
    auto repeat_settings=loaded;repeat_settings.mapping_confirmed=true;
    CHECK(repeat_settings.stand_s==4 && repeat_settings.return_stand_s==2);
    go_fsm::RobotFSM repeat(p,repeat_settings,motor,adapter,initial,t);
    repeat.request(0,hardware(initial),t);
    CHECK(repeat.request(1,hardware(initial),t+std::chrono::seconds(1)).find("尚未就绪")!=std::string::npos);
    repeat.update(h,{},t+std::chrono::seconds(2));CHECK(repeat.mode()==Mode::Standing);
    auto clock=t+std::chrono::seconds(8);
    repeat.update(h,{},clock);CHECK(repeat.mode()==Mode::StandHold);
    h=hardware(repeat_settings.stand_pose);
    repeat.update(h,{},clock+std::chrono::milliseconds(1));
    CHECK(repeat.request(1,h,clock+std::chrono::milliseconds(100)).find("尚未就绪")!=std::string::npos);
    clock+=std::chrono::milliseconds(600);repeat.update(h,{},clock);
    repeat.request(1,h,clock);CHECK(repeat.mode()==Mode::PPO);
    for(int cycle=0;cycle<2;++cycle) {
        PolicyResult fresh;fresh.generation=repeat.generation();fresh.sequence=1;
        fresh.actions.assign(12,0);fresh.command=convertAction(fresh.actions,p);
        fresh.command.q[2]+=.2f;fresh.generated=clock;fresh.observed=clock;
        repeat.update(h,fresh,clock); // Reset first_result on each entry.
        clock+=std::chrono::milliseconds(500);fresh.generated=clock;fresh.observed=clock;++fresh.sequence;
        auto active=repeat.update(h,fresh,clock);CHECK(active.kp[0]==25 && active.kd[0]==.5f);
        repeat.velocity('w');CHECK(repeat.velocityCommand()[0]>.0f);
        // Real motion must start from measured feedback, rather than the last policy target.
        auto actual=active.q;actual[2]+=.03f;h=hardware(actual);
        const auto generation=repeat.generation();
        CHECK(!repeat.request(0,h,clock).empty());CHECK(repeat.mode()==Mode::Standing);
        CHECK(repeat.generation()>generation && repeat.velocityCommand()[0]==0);
        auto start_command=repeat.update(h,fresh,clock);
        CHECK(start_command.q==actual && start_command.kp[0]==25 && start_command.kd[0]==.5f);
        auto middle=repeat.update(h,fresh,clock+std::chrono::milliseconds(250));
        CHECK(std::abs(middle.kp[0]-42.5f)<1e-5 && std::abs(middle.kd[0]-1.25f)<1e-5);
        CHECK(middle.q[2]<actual[2] && middle.q[2]>repeat_settings.stand_pose[2]);
        auto fixed=repeat.update(h,fresh,clock+std::chrono::milliseconds(500));
        CHECK(fixed.kp[0]==60 && fixed.kd[0]==2 && fixed.dq[2]<0);
        repeat.update(h,fresh,clock+std::chrono::milliseconds(1999));CHECK(repeat.mode()==Mode::Standing);
        CHECK(repeat.request(1,h,clock+std::chrono::milliseconds(1999)).find("尚未就绪")!=std::string::npos);
        clock+=std::chrono::seconds(2);
        auto returned=repeat.update(h,fresh,clock);
        CHECK(repeat.mode()==Mode::StandHold && returned.q==repeat_settings.stand_pose);
        h=hardware(returned.q);repeat.update(h,fresh,clock+std::chrono::milliseconds(1));
        CHECK(repeat.request(1,h,clock+std::chrono::milliseconds(100)).find("尚未就绪")!=std::string::npos);
        clock+=std::chrono::milliseconds(600);repeat.update(h,fresh,clock);
        CHECK(repeat.mode()==Mode::StandHold); // Early key did not queue a transition.
        repeat.request(1,h,clock);CHECK(repeat.mode()==Mode::PPO);
        CHECK(repeat.generation()!=fresh.generation);
        auto waiting=repeat.update(h,fresh,clock);
        CHECK(repeat.mode()==Mode::PPO && waiting.q==repeat_settings.stand_pose && waiting.kp[0]==60);
    }
    std::cout<<"robot FSM transitions/gains/generation/timeouts passed\n";
}
