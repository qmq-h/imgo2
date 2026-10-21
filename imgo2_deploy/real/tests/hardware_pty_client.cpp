#include "hardware/real_hardware.h"
#include "go_motor_monitor.h"
#include <chrono>
#include <cmath>
#include <limits>
#include <iostream>
#include <thread>
#include <stdexcept>
#define CHECK(x) do {if (!(x)) throw std::runtime_error(#x);} while(0)
using namespace std::chrono;
int main(int argc,char** argv) {
    try {
        CHECK(argc==5);
        const std::string mode=argv[1];
        auto config=real_config::makeHardwareConfig();
        config.motors.buses={{argv[2],{0,1,2},{1,1,1},{0.1f,0.2f,0.3f}},
                            {argv[3],{0,1,2},{1,1,1},{0.1f,0.2f,0.3f}}};
        // 本客户端使用两条总线、正角度人工关节；不复用真机12关节限位。
        config.motors.limits.assign(6, go::JointLimits{});
        config.motors.period=milliseconds(5);
        config.motors.io_timeout_us=15000;
        config.imu.device=argv[4]; config.imu.read_timeout_ms=20;
        config.imu.stale_after_ms=120; config.imu_ready_timeout_ms=180;
        config.motor_stale_after_ms=100;
        if (mode=="diagnostic") {
            std::vector<go::DiagnosticBus> buses;
            for (const auto& bus:config.motors.buses) {
                go::DiagnosticBus state;state.device=bus.device;
                for (int id:bus.motor_ids) {
                    go::DiagnosticMotor motor;motor.id=id;motor.startPose=.25f;
                    state.motors.push_back(motor);
                }
                buses.push_back(state);
            }
            go::GoMotorMonitor monitor(buses,config.motors.baud,15000,milliseconds(5));
            monitor.start(); std::this_thread::sleep_for(milliseconds(120));
            auto before=monitor.snapshot();
            CHECK(before[0].motors[0].request_ok);
            CHECK(before[0].motors[0].receive_time_ns>0);
            CHECK(std::abs(before[0].motors[0].qL-.25)<.001);
            CHECK(std::abs(before[0].motors[0].feedback.q-.2)<.001);
            std::cout<<"ARMED"<<std::endl;
            std::this_thread::sleep_for(milliseconds(200));
            auto after=monitor.snapshot();
            CHECK(!after[0].motors[0].request_ok);
            CHECK(after[0].motors[0].ever);
            CHECK(after[0].motors[0].receive_time_ns>=before[0].motors[0].receive_time_ns);
            CHECK(std::abs(after[0].motors[0].feedback.q-.2)<.001);
            CHECK(after[0].motors[1].request_ok && after[0].motors[2].request_ok);
            CHECK(after[1].sweeps>after[0].sweeps);
            CHECK(after[1].sweeps>before[1].sweeps+10);
            CHECK(after[0].badFrames>0 && after[0].strayFrames>0);
            monitor.stop();monitor.stop();
            std::cout<<"PASS"<<std::endl;return 0;
        }
        if (mode=="motor_stale") config.motors.period=milliseconds(200);
        real_hardware::RealHardware hardware(config);
        std::string error;
        if (mode=="missing_imu" || mode=="missing_fields" || mode=="missing_motor") {
            CHECK(!hardware.start(error));CHECK(!error.empty());
            CHECK(!hardware.readState().started);
            hardware.stop();hardware.stop();
            std::cout<<"PASS startup cleanup"<<std::endl;return 0;
        }
        CHECK(hardware.start(error));
        auto initial=hardware.readState();
        CHECK(initial.started && initial.imu_valid && initial.joints.size()==6);
        CHECK(!initial.joint_mapping_confirmed && !initial.imu_mount_confirmed);
        go::Targets target;
        for (const auto& joint:initial.joints) {
            CHECK(joint.valid && joint.receive_time_ns>0 && joint.age_ms>=0 && joint.age_ms<100);
            target.q.push_back(joint.feedback.q);target.kp.push_back(25);target.kd.push_back(.5);
            target.dq.push_back(0);target.tau.push_back(0);
        }
        CHECK(hardware.sendCommand(target));
        if (mode!="motor_stale") {
            const auto until=steady_clock::now()+milliseconds(130);
            while (steady_clock::now()<until) {
                CHECK(hardware.sendCommand(target));std::this_thread::sleep_for(milliseconds(5));
            }
        }
        std::cout<<"ARMED"<<std::endl;
        if (mode=="normal") {
            CHECK(hardware.sendCommand(target));hardware.stop();hardware.stop();
            std::cout<<"PASS"<<std::endl;return 0;
        }
        if (mode=="invalid_command") {
            auto invalid=target; invalid.q[0]=std::numeric_limits<float>::quiet_NaN();
            CHECK(!hardware.sendCommand(invalid));
        }
        // Do not call sendCommand: the supervisor must stop all buses autonomously.
        const auto deadline=steady_clock::now()+milliseconds(600);
        while (!hardware.readState().faulted && steady_clock::now()<deadline)
            std::this_thread::sleep_for(milliseconds(5));
        CHECK(hardware.readState().faulted);
        CHECK(!hardware.sendCommand(target));
        std::cout<<"FAULT"<<std::endl;
        std::this_thread::sleep_for(milliseconds(250));
        CHECK(!hardware.sendCommand(target));
        if (mode=="imu_timeout" || mode=="motor_failure") {
            hardware.stop();
            CHECK(hardware.start(error));CHECK(hardware.sendCommand(target));
        }
        hardware.stop();hardware.stop();
        std::cout<<"PASS"<<std::endl;
    } catch (const std::exception& error) {std::cerr<<error.what()<<std::endl;return 1;}
}
