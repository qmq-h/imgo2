/* ----------------------------------------------------------------------------
 * 用法
 *   ./rl_real_imgo2 [参数]
 *   真机部署：读可执行文件旁的 config/deployment.yaml 与 policy/ppo/config.yaml。
 *
 * 参数（默认值）
 *   --config <yaml>     部署配置（默认 <exe 目录>/config/deployment.yaml）
 *   --policy-dir <dir>  策略目录，下含 ppo/config.yaml（默认 policy）
 *   --controller-device <path>  覆盖 real_config 里的手柄设备名（仅 gamepad 模式）
 *   --check-policy      只校验策略/配置，不打开硬件
 *   --log               写日志：robot.csv(200Hz) / policy.csv(50Hz) / events.log / meta.json / trace
 *   --log-dir <dir>     日志根目录（默认 ./log）
 *   --log-every <N>     每 N 个控制拍写一行（默认 1=200Hz，2=100Hz）
 *   --log-max-mb <N>    robot.csv 超过 N MB 切分（默认 200，0=不切）
 *   --bus-stats         每秒打印每总线耗时/线程最坏值/IMU age（--log 会自动开）
 *   --trace <前缀>      逐事务时间线（--log 会自动开）
 *   --help
 *
 * 键盘：0 站起 / 1 PPO / 2 占位 / 9 趴下 / WASDQE 即时速度 / 空格清零 / Ctrl+C 退出
 * 安全：任何 Fault、看门狗、9、Ctrl+C 都是力矩归零（会塌），测试时必须吊住/托住。
 * ---------------------------------------------------------------------------- */
#include "apps/runtime.h"
#include "apps/deployment_paths.h"
#include "apps/deployment_input.h"
#include "apps/log_writer.h"
#include "config/deployment_config.h"
#include "fsm/deployment_states.h"
#include "input/controller_input.h"
#include "sdk/inference_worker.h"
#include "sdk/rl_sdk.h"
#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>
#ifdef GO_RL_PTY_TEST
#include "pty_devices.h"
#endif
namespace {
std::vector<float> startupPose(const go::MotorConfig& c) {
    std::vector<float> q;
    for(const auto& b:c.buses) q.insert(q.end(),b.start_pose_rad.begin(),b.start_pose_rad.end());
    return q;
}
std::vector<float> measured(const real_hardware::HardwareState& h) {
    std::vector<float> q;for(const auto& j:h.joints) q.push_back(j.feedback.q);return q;
}
// 逐事务时间线：把环形缓冲里的埋点排序后写成 CSV，并在终端打印一段判读摘要。
// 判读关键：看"最长无事务间隔"——它大 ≈ 总线线程根本没在跑（被冻结/抢占）；
// 它小但失败多 ≈ 事务一直在发，只是没人应/回了垃圾（链路问题）。
void dumpTrace(const std::string& path,const go::TransactionTrace* trace,const char* tag) {    if(!trace) {std::fprintf(stderr,"trace: 没有电机总线，跳过\n");return;}
    auto records=trace->snapshot();
    std::sort(records.begin(),records.end(),
              [](const go::TraceRecord&a,const go::TraceRecord&b){return a.start_ns<b.start_ns;});
    const std::string file=path+"."+tag+".csv";
    if(std::FILE* f=std::fopen(file.c_str(),"w")) {
        std::fprintf(f,"t_ms,bus,id,ok,waited_us,bytes,bad,stray\n");
        for(const auto& r:records)
            std::fprintf(f,"%.3f,%u,%u,%u,%u,%u,%u,%u\n",double(r.start_ns)/1e6,unsigned(r.bus),
                         unsigned(r.id),unsigned(r.ok),unsigned(r.waited_us),unsigned(r.bytes),
                         unsigned(r.bad),unsigned(r.stray));
        std::fclose(f);
    }
    std::printf("trace[%s]: %zu 条事务 -> %s\n",tag,records.size(),file.c_str());
    for(unsigned bus=0;bus<8;++bus) {
        size_t n=0,fail=0,bytes=0,bad=0,stray=0;std::int64_t prev=0,gap=0;std::uint32_t wait=0;
        for(const auto& r:records) {
            if(r.bus!=bus) continue;
            ++n; if(!r.ok) ++fail;
            bytes+=r.bytes; bad+=r.bad; stray+=r.stray;
            if(r.waited_us>wait) wait=r.waited_us;
            if(prev&&r.start_ns-prev>gap) gap=r.start_ns-prev;
            prev=r.start_ns;
        }
        if(!n) continue;
        std::printf("  bus%u 事务=%zu 失败=%zu 最长无事务间隔=%.1fms 最长单事务=%.1fms bytes=%zu bad=%zu stray=%zu\n",
                    bus,n,fail,double(gap)/1e6,double(wait)/1000.0,bytes,bad,stray);
    }
    std::printf("  判读: 最长无事务间隔≈100ms → 线程没被调度(冻结/抢占)；"
                "间隔很小但失败多 → 一直在发但没人应(bytes=0)或回了坏帧(bad>0)\n");
}
// ---- 日志 meta 片段（自描述，便于以后用 pandas 还原现场）----
std::string vecJson(const std::vector<float>& v) {
    std::string out="[";char buf[32];
    for(size_t i=0;i<v.size();++i) {
        std::snprintf(buf,sizeof buf,"%s%.5f",i?",":"",double(v[i]));out+=buf;
    }
    return out+"]";
}
std::string vecJson(const std::array<float,3>& v) {
    std::string out="[";char buf[32];
    for(size_t i=0;i<v.size();++i) {
        std::snprintf(buf,sizeof buf,"%s%.5f",i?",":"",double(v[i]));out+=buf;
    }
    return out+"]";
}
std::string intJson(const std::vector<int>& v) {
    std::string out="[";char buf[32];
    for(size_t i=0;i<v.size();++i) {
        std::snprintf(buf,sizeof buf,"%s%d",i?",":"",v[i]);out+=buf;
    }
    return out+"]";
}
std::string jointsJson(const go::MotorConfig& m) {
    std::string out="[";
    int index=0;
    for(const auto& bus:m.buses) {
        const auto slash=bus.device.find_last_of('/');
        const std::string label=slash==std::string::npos?bus.device:bus.device.substr(slash+1);
        for(size_t i=0;i<bus.motor_ids.size();++i,++index) {
            char buf[320];
            std::snprintf(buf,sizeof buf,
                "%s{\"index\":%d,\"label\":\"%s_%zu\",\"device\":\"%s\",\"motor_id\":%d,\"direction\":%.0f,\"start_pose_rad\":%.6f}",
                index?",":"",index,label.c_str(),i,bus.device.c_str(),bus.motor_ids[i],
                double(bus.motor_directions[i]),double(bus.start_pose_rad[i]));
            out+=buf;
        }
    }
    return out+"]";
}
std::string configJson(const imgo2_sdk::PolicyConfig& p,const imgo2_sdk::DeploymentConfig& d,
                       const go::MotorConfig& m,const real_config::HardwareConfig& hw) {
    std::string qmin="[",qmax="[",dqm="[",taum="[",kpm="[",kdm="[";char buf[512];
    for(size_t i=0;i<m.limits.size();++i) {
        const auto& l=m.limits[i];
        std::snprintf(buf,sizeof buf,"%s%.5f",i?",":"",double(l.q_min));qmin+=buf;
        std::snprintf(buf,sizeof buf,"%s%.5f",i?",":"",double(l.q_max));qmax+=buf;
        std::snprintf(buf,sizeof buf,"%s%.4f",i?",":"",double(l.dq_max));dqm+=buf;
        std::snprintf(buf,sizeof buf,"%s%.4f",i?",":"",double(l.tau_max));taum+=buf;
        std::snprintf(buf,sizeof buf,"%s%.2f",i?",":"",double(l.kp_max));kpm+=buf;
        std::snprintf(buf,sizeof buf,"%s%.2f",i?",":"",double(l.kd_max));kdm+=buf;
    }
    qmin+="]";qmax+="]";dqm+="]";taum+="]";kpm+="]";kdm+="]";
    std::string out="{";
    out+="\"policy_to_hardware\": "+intJson(d.policy_to_hardware);
    out+=",\"default_pose\": "+vecJson(p.default_pose);
    out+=",\"action_scale\": "+vecJson(p.action_scale);
    out+=",\"lower\": "+vecJson(p.lower)+",\"upper\": "+vecJson(p.upper);
    out+=",\"kp\": "+vecJson(p.kp)+",\"kd\": "+vecJson(p.kd);
    out+=",\"fixed_kp\": "+vecJson(p.fixed_kp)+",\"fixed_kd\": "+vecJson(p.fixed_kd);
    out+=",\"torque_limits\": "+vecJson(p.torque_limits);
    out+=",\"commands_scale\": "+vecJson(p.commands_scale);
    std::snprintf(buf,sizeof buf,",\"obs_scales\": {\"ang_vel\": %.5f, \"dof_pos\": %.5f, \"dof_vel\": %.5f, \"clip_obs\": %.1f}",
                  double(p.ang_vel_scale),double(p.dof_pos_scale),double(p.dof_vel_scale),double(p.clip_obs));
    out+=buf;
    out+=",\"velocity_limits\": "+vecJson(d.velocity_limits);
    out+=",\"input_source\": ";
    out+=d.input_source==imgo2_sdk::InputSource::Gamepad?"\"gamepad\"":"\"keyboard\"";
    out+=",\"controller_device\": \""+hw.controller.device+"\"";
    std::snprintf(buf,sizeof buf,",\"controller_deadzone\": %.3f",double(hw.controller.deadzone));
    out+=buf;
    std::snprintf(buf,sizeof buf,
        ",\"timing\": {\"control_ms\": %d, \"policy_ms\": %d, \"result_timeout_ms\": %d,"
        " \"io_timeout_us\": %zu, \"command_timeout_ms\": %zu, \"period_ms\": %.3f,"
        " \"watchdog_ms\": %lld, \"motor_stale_after_ms\": %d, \"imu_stale_after_ms\": %d,"
        " \"imu_ready_timeout_ms\": %d}",
        d.control_ms,d.policy_ms,d.result_timeout_ms,m.io_timeout_us,m.command_timeout_ms,
        double(m.period.count())/1000.0,(long long)m.watchdog.count(),
        hw.motor_stale_after_ms,hw.imu.stale_after_ms,hw.imu_ready_timeout_ms);
    out+=buf;
    out+=",\"limits\": {\"q_min\": "+qmin+", \"q_max\": "+qmax+", \"dq_max\": "+dqm+
         ", \"tau_max\": "+taum+", \"kp_max\": "+kpm+", \"kd_max\": "+kdm+"}";
    out+=",\"mapping_confirmed\": "+(d.mapping_confirmed?std::string("true"):std::string("false"));
    return out+"}";
}
std::string formatTime(const char* fmt) {
    const auto now=std::chrono::system_clock::now();
    const std::time_t t=std::chrono::system_clock::to_time_t(now);
    std::tm tm{};localtime_r(&t,&tm);
    char buf[64];std::strftime(buf,sizeof buf,fmt,&tm);
    return buf;
}
double msSince(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
}
std::int64_t wallMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// Exception cleanup requests unload BEFORE joining an in-flight inference or terminal restore.
struct StopOnExit {
    real_hardware::RealHardware& hardware;
    ~StopOnExit() {hardware.requestStop("deployment exiting");}
};
}
int main(int argc,char** argv) try {
    auto policy_root=go_app::executableDirectory()/"policy";
    auto deployment_file=go_app::executableDirectory()/"config/deployment.yaml";
    bool check_only=false;
    bool bus_stats=false;
    bool log_on=false;
    std::string trace_file;
    std::string controller_device;
    std::string log_dir="log";
    int log_every=1,log_max_mb=200;
    for(int i=1;i<argc;++i) {
        const std::string arg=argv[i];
        if(arg=="--help") {
            std::puts("rl_real_imgo2 [--policy-dir directory] [--config deployment.yaml] [--check-policy]\n"
                      "             [--controller-device path]\n"
                      "             [--bus-stats] [--trace prefix] [--log] [--log-dir dir]\n"
                      "             [--log-every N] [--log-max-mb N]\n"
                      "0/1/2/9 + Enter: stand/PPO/AMP placeholder/crouch. WASDQE immediate; Space zero; Ctrl+C exits.\n"
                      "gamepad (input_source: 2): left stick = vx/vy (up/left +), right stick = wz (right +), A/B/Y = stand/crouch/PPO.\n"
                      "--bus-stats: 每秒打印每条总线的轮询耗时(last/max)、线程最坏值(gap/body)与 IMU age。\n"
                      "--trace P: 逐事务时间线，故障时写 P.fault.csv、退出时写 P.exit.csv（含判读摘要）。\n"
                      "--log: 写 log/<时间>_<pid>/{robot.csv,policy.csv,events.log,meta.json,trace.*.csv}；\n"
                      "       robot.csv 每个控制节拍一行（200 Hz，硬件顺序），policy.csv 每次消费到策略结果一行（约 50 Hz）；\n"
                      "       控制线程只入队，格式化与写盘在独立线程，不阻塞控制。\n"
                      "       同时自动打开 --bus-stats（线程最坏值/IMU age）与 --trace（逐事务时间线，落到同一目录）。\n"
                      "--log-every N: 每 N 个节拍写一行（默认 1 = 200 Hz；N=2 -> 100 Hz）。\n"
                      "--log-max-mb N: robot.csv 超过 N MB 自动切分（默认 200；0 = 不切分）。");
            return 0;
        }
        if(arg=="--check-policy") check_only=true;
        else if(arg=="--bus-stats") bus_stats=true;
        else if(arg=="--log") log_on=true;
        else if(arg=="--log-dir" && i+1<argc) {log_dir=argv[++i];log_on=true;}
        else if(arg=="--log-every" && i+1<argc) log_every=go_app::integer(argv[++i],1,100,"--log-every");
        else if(arg=="--log-max-mb" && i+1<argc) log_max_mb=go_app::integer(argv[++i],0,100000,"--log-max-mb");
        else if(arg=="--trace" && i+1<argc) trace_file=argv[++i];
        else if(arg=="--policy-dir" && i+1<argc) policy_root=argv[++i];
        else if(arg=="--controller-device" && i+1<argc) controller_device=argv[++i];
        else if(arg=="--config" && i+1<argc) deployment_file=argv[++i];
        else throw std::invalid_argument("unknown/missing argument: "+arg);
    }
    auto deployment=imgo2_sdk::DeploymentConfig::load(deployment_file);
    const auto policy=imgo2_sdk::PolicyConfig::load(policy_root/"ppo");
    imgo2_sdk::PolicyRunner runner(policy);
    runner.load(deployment.torch_threads);runner.warmup();
    // Guard all fixed poses/gains BEFORE opening any device.
    const go::MotorConfig motors;
    imgo2_sdk::RobotAdapter adapter(deployment.policy_to_hardware);
    go_fsm::RobotFSM preflight(policy,deployment,motors,adapter,startupPose(motors),imgo2_sdk::Clock::now());
    if(check_only) {
        std::puts("PPO model/config verified: 45 observations -> 12 actions. No hardware opened.");return 0;
    }
    auto hardware_config=real_config::makeHardwareConfig();
    hardware_config.joint_mapping_confirmed=deployment.mapping_confirmed;
    if(!controller_device.empty()) hardware_config.controller.device=controller_device;
    hardware_config.controller.validate();
#ifdef GO_RL_PTY_TEST
    // Test executable only: never opens configured device names; production has no env override.
    std::string test_error;
    if(!go_test::applyTestDevices(hardware_config.motors,"GO_RL_TEST_DEVICES",test_error))
        throw std::invalid_argument(test_error);
    const char* imu=std::getenv("GO_RL_TEST_IMU");
    if(!imu) throw std::invalid_argument("missing test IMU");
    int inference_delay_ms=0;
    if(const char* delay=std::getenv("GO_RL_TEST_INFERENCE_DELAY_MS"))
        inference_delay_ms=static_cast<int>(go_app::integer(delay,0,1000,"test delay"));
    hardware_config.imu.device=imu;
#endif
    go_app::StopSignals signals;
    go_app::TerminalInput terminal;
    go_app::DeploymentInput input;
    real_hardware::RealHardware hardware(hardware_config);
    std::string error;
    if(!hardware.start(error)) {std::fprintf(stderr,"start failed: %s\n",error.c_str());return 1;}
    go_fsm::RobotFSM fsm(policy,deployment,hardware_config.motors,adapter,measured(hardware.readState()),imgo2_sdk::Clock::now());
    imgo2_sdk::InferenceWorker worker(
        [&](const auto& s,const auto& v,uint64_t g) {
#ifdef GO_RL_PTY_TEST
            if(inference_delay_ms) std::this_thread::sleep_for(std::chrono::milliseconds(inference_delay_ms));
#endif
            return runner.infer(s,v,g);
        },
        [&]{runner.reset();},deployment.policy_ms,deployment.result_timeout_ms);
    worker.start();
    StopOnExit cleanup{hardware}; // Constructed last; on any unwind unloads before worker destructor.
    const bool gamepad=deployment.input_source==imgo2_sdk::InputSource::Gamepad;
    imgo2_input::ControllerInput controller;
    if(gamepad) {
        std::string controller_error;
        if(!controller.open(hardware_config.controller.device,hardware_config.controller.deadzone,
                            controller_error))
            throw std::runtime_error("gamepad: "+controller_error);
        std::printf("手柄: %s（死区 %.2f）\n",controller.device().c_str(),
                    double(hardware_config.controller.deadzone));
    }
    // 日志：打开失败只提示，不影响控制。控制线程只入队，格式化/写盘在 RunLogger 自己的线程。
    go_app::RunLogger logger;
    if(log_on) {
        go_app::LogMeta meta;
        meta.created=formatTime("%Y-%m-%d %H:%M:%S");
        meta.exe=go_app::executableDirectory().string();
        { std::string a; for(int i=0;i<argc;++i) { if(i) a+=' '; a+=argv[i]; } meta.args=a; }
        meta.every=log_every;meta.control_ms=deployment.control_ms;meta.policy_ms=deployment.policy_ms;
        meta.joints_json=jointsJson(hardware_config.motors);
        meta.config_json=configJson(policy,deployment,hardware_config.motors,hardware_config);
        const std::string dir=log_dir+"/"+formatTime("%Y-%m-%d_%H-%M-%S")+"_"+std::to_string(getpid());
        std::string log_error;
        if(logger.open(dir,meta,log_max_mb,log_error)) {
            std::printf("log: %s（robot.csv 每 %d 个控制节拍一行；policy.csv 约 50 Hz）\n",
                        dir.c_str(),log_every);
            // 一个 --log 就给全诊断：线程/调度统计 + 逐事务时间线，都落到同一目录，
            // 这样排查"链路静默 / 坏帧 / 线程被抢占"时不必再记额外参数。
            bus_stats=true;
            if(trace_file.empty()) trace_file=dir+"/trace";
        } else std::fprintf(stderr,"log 未启用: %s（继续运行）\n",log_error.c_str());
    }
    const auto log_start=imgo2_sdk::Clock::now();
    if(logger.enabled()) {
        char buf[192];
        std::snprintf(buf,sizeof buf,"0.000\tstart mode=%s every=%d",fsm.name(),log_every);
        logger.event(buf);
    }
    std::uint64_t tick=0;
    std::vector<float> prev_actions(12,0.0f);
    imgo2_sdk::Clock::time_point last_policy_time{};
    std::puts("初始保持：0+回车站起，1+回车PPO，2占位，9+回车趴下；WASDQE即时，空格清零。");
    if(!deployment.mapping_confirmed) std::puts("PPO入口锁定：关节映射尚未确认。");
    std::fflush(stdout);
    auto next=imgo2_sdk::Clock::now();
    auto next_stats=next;
    std::vector<double> bus_stats_max;
    long long imu_age_max=-1;
    long long ctrl_late_max_us=0;
    std::string last_name=fsm.name();bool reported_fault=false;int result=0;
    imgo2_sdk::Clock::time_point next_controller_retry{};
    bool controller_down=false;
    while(!signals.requested()) {
        const auto now=imgo2_sdk::Clock::now();
        ++tick;
        if(bus_stats && now>next) // 主控环自己迟到了多少（系统级停顿会同时体现在这里）
            ctrl_late_max_us=std::max(ctrl_late_max_us,
                static_cast<long long>(std::chrono::duration_cast<std::chrono::microseconds>(now-next).count()));
        const auto raw=hardware.readState();
        if(gamepad) {
            imgo2_input::GamepadCommand command;
            const auto controller_status=controller.poll(command);
            if(controller_status!=imgo2_input::ControllerPoll::Ok) {
                if(!controller_down) {
                    std::fprintf(stderr,"手柄断开/不可用: %s（速度清零，等待重连）\n",
                                 controller.lastError().c_str());
                    controller_down=true;
                }
                command.connected=false;
                if(now>=next_controller_retry) {
                    std::string controller_error;
                    if(controller.open(hardware_config.controller.device,
                                       hardware_config.controller.deadzone,controller_error)) {
                        std::printf("手柄已连接: %s\n",controller.device().c_str());
                        controller_down=false;
                    }
                    next_controller_retry=now+std::chrono::seconds(1);
                }
            }
            const auto message=fsm.gamepad(command,raw,now);
            if(!message.empty()) {
                std::puts(message.c_str());std::fflush(stdout);
                if(logger.enabled()) {
                    char buf[512];
                    std::snprintf(buf,sizeof buf,"%.3f\t%s",msSince(log_start),message.c_str());
                    logger.event(buf);
                }
            }
        } else {
            char c=0;
            // Bound input work per cycle; key repeats cannot starve motor target refresh.
            for(int n=0;n<64;++n) {
                const auto input_status=go_app::readDeploymentChar(c);
                if(input_status==go_app::InputStatus::Waiting) break;
                if(input_status==go_app::InputStatus::Closed) {
                    hardware.requestStop("keyboard input closed");
                    go_app::stopFlag()=1;result=2;break;
                }
                const auto event=input.feed(c);
                if(event.kind==go_app::KeyEvent::Kind::State) {
                    const auto message=fsm.request(event.digit,raw,now);
                    if(!message.empty()) {std::puts(message.c_str());std::fflush(stdout);}
                } else if(event.kind==go_app::KeyEvent::Kind::Velocity) {
                    const auto previous=fsm.velocityCommand();fsm.velocity(event.key);
                    if(previous!=fsm.velocityCommand()) {
                        const auto& v=fsm.velocityCommand();
                        std::printf("cmd vx=%.2f vy=%.2f yaw=%.2f\n",v[0],v[1],v[2]);std::fflush(stdout);
                        if(logger.enabled()) {
                            char buf[160];
                            std::snprintf(buf,sizeof buf,"%.3f\tcmd vx=%.2f vy=%.2f yaw=%.2f",
                                msSince(log_start),v[0],v[1],v[2]);
                            logger.event(buf);
                        }
                    }
                }
            }
        }
        if(signals.requested()) break;
        const auto latest=worker.latest();
        const auto& target=fsm.update(raw,latest,imgo2_sdk::Clock::now());
        if(fsm.mode()==go_fsm::RobotFSM::Mode::Fault) {
            hardware.requestStop(fsm.error());
            if(!reported_fault) {
                std::fprintf(stderr,"Fault: %s\n",fsm.error().c_str());reported_fault=true;result=2;
                // 故障当下就把逐事务时间线落盘：之后 Ctrl+C 也不影响这份证据。
                if(!trace_file.empty()) dumpTrace(trace_file,hardware.transactionTrace(),"fault");
                if(logger.enabled()) {
                    char buf[512];
                    std::snprintf(buf,sizeof buf,"%.3f\tFault: %s",msSince(log_start),fsm.error().c_str());
                    logger.event(buf);
                }
            }
            worker.submit({},{{0,0,0}},fsm.generation(),false);
        } else {
            if(!hardware.sendCommand(target)) {
                fsm.fault(hardware.readState().error,now);hardware.requestStop(fsm.error());continue;
            }
            auto observation=adapter.toPolicy(raw);observation.captured_at=now;
            worker.submit(observation,fsm.velocityCommand(),fsm.generation(),fsm.policyActive());
            if(logger.enabled()) {
                if(tick%static_cast<std::uint64_t>(log_every)==0) {
                    go_app::LogSample s;
                    s.t_mono_ms=msSince(log_start);s.t_wall_ms=wallMs();
                    s.tick=tick;s.generation=fsm.generation();
                    s.mode=static_cast<std::uint8_t>(fsm.mode());
                    for(int i=0;i<3;++i) s.cmd[i]=fsm.velocityCommand()[i];
                    // 记录 Safety 夹过之后真正下发的值（q_des/dq_des/tau_ff/kp/kd）
                    go::Targets applied=target;
                    const auto clamp_result=go::Safety::clamp(applied,hardware_config.motors.limits);
                    s.clamped=clamp_result==go::ClampResult::Clamped?1:0;
                    for(int i=0;i<12;++i) {
                        s.kp[i]=applied.kp[i];s.kd[i]=applied.kd[i];
                        s.q_des[i]=applied.q[i];s.dq_des[i]=applied.dq[i];s.tau_ff[i]=applied.tau[i];
                    }
                    for(size_t i=0;i<raw.joints.size()&&i<12;++i) {
                        const auto& j=raw.joints[i];
                        s.q[i]=j.feedback.q;s.dq[i]=j.feedback.dq;s.tau[i]=j.feedback.tau;
                        s.temp[i]=static_cast<float>(j.feedback.temp);
                        s.age_ms[i]=static_cast<std::int16_t>(std::clamp<long long>(j.age_ms,-32768,32767));
                        if(j.valid) s.valid_mask|=static_cast<std::uint16_t>(1u<<i);
                        if(j.feedback.error) s.err_mask|=static_cast<std::uint16_t>(1u<<i);
                    }
                    const auto& imu=raw.imu.data;
                    for(int i=0;i<4&&i<int(imu.quaternion.size());++i) s.quat[i]=imu.quaternion[i];
                    for(int i=0;i<3&&i<int(imu.gyroscope.size());++i) s.gyro[i]=imu.gyroscope[i];
                    s.imu_age_ms=static_cast<std::int16_t>(std::clamp<long long>(raw.imu_age_ms,-32768,32767));
                    s.imu_valid=raw.imu_valid?1:0;s.imu_hz=static_cast<float>(raw.imu.receive_hz);
                    if(latest.actions.size()==12)
                        for(int i=0;i<12;++i) s.act[i]=latest.actions[i];
                    // 每条总线最近一轮的三段耗时：直接回答"一个事务 ~2 ms 花在哪"。
                    const auto bus_timing=hardware.busTiming();
                    for(int b=0;b<4&&b<static_cast<int>(bus_timing.size());++b) {
                        s.bus_total_us[b]=bus_timing[b].total_us;
                        s.bus_flush_us[b]=bus_timing[b].flush_us;
                        s.bus_write_us[b]=bus_timing[b].write_us;
                        s.bus_reply_us[b]=bus_timing[b].reply_us;
                    }
                    logger.push(s);
                }
                // 策略样本：只在"消费到新结果"的那一刻写（约 50 Hz）；generated 每次推理都会变
                if(fsm.policyActive()&&latest.generated!=last_policy_time) {
                    last_policy_time=latest.generated;
                    if(latest.actions.size()==12&&observation.valid) {
                        try {
                            const auto obs=imgo2_sdk::buildObservation(
                                observation,fsm.velocityCommand(),prev_actions,policy);
                            go_app::PolicySample ps;
                            ps.t_mono_ms=msSince(log_start);ps.t_wall_ms=wallMs();
                            ps.tick=tick;ps.generation=fsm.generation();
                            ps.mode=static_cast<std::uint8_t>(fsm.mode());
                            for(int i=0;i<3;++i) ps.cmd[i]=fsm.velocityCommand()[i];
                            for(int i=0;i<45;++i) ps.obs[i]=obs[i];
                            for(int i=0;i<12;++i) ps.act[i]=latest.actions[i];
                            ps.result_age_ms=static_cast<float>(
                                std::chrono::duration<double,std::milli>(now-latest.generated).count());
                            ps.obs_age_ms=static_cast<float>(
                                std::chrono::duration<double,std::milli>(now-observation.captured_at).count());
                            logger.pushPolicy(ps);
                            for(int i=0;i<12;++i) prev_actions[i]=latest.actions[i];
                        } catch(const std::exception&) {} // 日志绝不影响控制
                    }
                }
            }
        }
        if(last_name!=fsm.name()) {
            last_name=fsm.name();std::printf("状态: %s\n",last_name.c_str());std::fflush(stdout);
            if(logger.enabled()) {
                char buf[192];
                std::snprintf(buf,sizeof buf,"%.3f\tmode=%s gen=%llu tick=%llu",
                    msSince(log_start),last_name.c_str(),
                    static_cast<unsigned long long>(fsm.generation()),
                    static_cast<unsigned long long>(tick));
                logger.event(buf);
            }
        }
        if(bus_stats) {
            const auto cycle_ms=hardware.busCycleMs();
            const auto devices=hardware.busDevices();
            if(raw.imu.received)
                imu_age_max=std::max(imu_age_max,static_cast<long long>(raw.imu_age_ms));
            if(cycle_ms.empty()) bus_stats=false; // 没有电机就不刷了
            else {
                if(bus_stats_max.size()!=cycle_ms.size()) bus_stats_max.assign(cycle_ms.size(),0.0);
                for(size_t i=0;i<cycle_ms.size();++i)
                    if(cycle_ms[i]>bus_stats_max[i]) bus_stats_max[i]=cycle_ms[i];
                if(now>=next_stats) {
                    std::printf("bus轮询(1s窗口 last/max):");
                    for(size_t i=0;i<cycle_ms.size();++i)
                        std::printf(" %s=%.2f/%.2fms",i<devices.size()?devices[i].c_str():"?",
                                    cycle_ms[i],bus_stats_max[i]);
                    std::printf(" | 预算%.1fms 节拍%.1fms 有效期%zums"
                                " | IMU age=%lld/%lldms hz=%.0f valid=%d\n",
                                hardware_config.motors.io_timeout_us/1000.0,
                                hardware_config.motors.period.count()/1000.0,
                                hardware_config.motors.command_timeout_ms,
                                static_cast<long long>(raw.imu_age_ms),imu_age_max,
                                raw.imu.receive_hz,raw.imu_valid?1:0);
                    std::fflush(stdout);
                    // 第二行：线程/调度层面的累计最坏值——用来区分"I/O 慢"还是"线程被抢占"
                    const auto ls=hardware.loopStats();
                    std::printf("          线程最坏值(累计): gap=%.1fms body=%.1fms interval=%.1fms"
                                " | 主控环迟到=%.1fms\n",
                                ls.gapMaxUs/1000.0,ls.bodyMaxUs/1000.0,ls.intervalMaxUs/1000.0,
                                ctrl_late_max_us/1000.0);
                    std::fflush(stdout);
                    next_stats=now+std::chrono::seconds(1);
                    bus_stats_max.assign(cycle_ms.size(),0.0);
                    imu_age_max=-1;
                    ctrl_late_max_us=0;
                }
            }
        }
        next+=std::chrono::milliseconds(deployment.control_ms);
        if(next<imgo2_sdk::Clock::now()) next=imgo2_sdk::Clock::now();
        std::this_thread::sleep_until(next);
    }
    hardware.requestStop("shutdown");
    if(!trace_file.empty()) dumpTrace(trace_file,hardware.transactionTrace(),"exit");
    if(logger.enabled()) {
        logger.event("exit");
        logger.close();
        if(logger.dropped())
            std::fprintf(stderr,"log: 有 %llu 个样本因队列满被丢弃\n",
                         static_cast<unsigned long long>(logger.dropped()));
    }
    worker.stop();hardware.stop();return result;
} catch(const std::exception& e) {
    std::fprintf(stderr,"deployment error: %s\n",e.what());return 1;
}
