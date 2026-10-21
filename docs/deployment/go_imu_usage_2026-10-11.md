# go_imu 使用、结构与故障手册

> 2026-10-15结构更新：全部具体状态已归入fsm/states，go_stand与部署共用唯一StandState；原行为与本文历史验证记录保留，当前结构见 [纠正记录](go_imu_state_split_2026-10-15.md#同日结构纠正记录删除双standstate与继承套层)。

当前入口更新：2026-10-15，ppo_check/imu_reader/go_probe已删除，离线键盘测试改用 [fsm_check](fsm_check_2026-10-15.md)。历史硬件参数说明继续保留。五阶段重构与21项离线验证见 [整理记录](go_imu_cleanup_2026-10-14.md)。用户已确认udev与原monitor实测正常；新版本控制实机仍待确认。

## 1. 结构与职责

```text
go_imu/
├─ config/     MotorConfig / ImuConfig及默认构造参数
├─ function/   my_myth.h/.cpp：数学常量、换算、clamp、五次曲线
├─ Motor/      协议、总线、整组控制、诊断、限幅、周期循环
├─ serialport/ 唯一Linux串口实现，IMU旧接口适配共用它
├─ Imu/        Yesense解析、采集和线程安全快照
├─ hardware/   RealHardware整机控制与传感器监督
├─ fsm/        基类、单站立状态与完整行0事件
├─ display/    电机/IMU面板、备用终端和光标恢复
├─ apps/       runtime.h共享信号、输入和CLI辅助
├─ tests/      离线与伪终端测试
└─ 五个入口cpp：monitor/demo/stand、rl_real_imgo2/fsm_check（后两项可选RL）
```

Motor不依赖终端；go_imu::hardware可供其他工程链接。硬件目标没有ROS、策略或UnitreeSDK依赖，没有电机文件日志；可选RL入口需要LibTorch/yaml-cpp。以下为原硬件源码清单（已移除删除项）；新增SDK/完整状态树见部署与FSM测试指南：

```text
CMakeLists.txt
Imu/inc/imu_sensor/imu_data.hpp
Imu/inc/imu_sensor/imu_monitor.hpp
Imu/inc/imu_sensor/imu_sensor.hpp
Imu/inc/imu_sensor/yesense_parser.hpp
Imu/src/imu_monitor.cpp
Imu/src/imu_sensor.cpp
Imu/src/yesense_parser.cpp
Motor/inc/go_bus.h
Motor/inc/go_loop.h
Motor/inc/go_motor.h
Motor/inc/go_motor_array.h
Motor/inc/go_motor_monitor.h
Motor/inc/go_protocol.h
Motor/inc/go_safety.h
Motor/inc/go_trajectory.h
Motor/inc/motor_transport.h
Motor/src/go_bus.cpp
Motor/src/go_loop.cpp
Motor/src/go_motor.cpp
Motor/src/go_motor_array.cpp
Motor/src/go_motor_monitor.cpp
Motor/src/go_protocol.cpp
Motor/src/go_safety.cpp
Motor/src/go_trajectory.cpp
apps/runtime.h
config/real_config.cpp
config/real_config.h
display/imu_panel.cpp
display/imu_panel.h
display/motor_panel.cpp
display/motor_panel.h
display/terminal_display.hpp
fsm/fsm.h
fsm/states/stand_state.cpp
fsm/states/stand_state.h
function/my_myth.cpp
function/my_myth.h
go_monitor.cpp
fsm_check.cpp
rl_real_imgo2.cpp
hardware/real_hardware.cpp
hardware/real_hardware.h
pd_demo_main.cpp
serialport/inc/imu_sensor/serial_port.hpp
serialport/inc/serial_port.h
serialport/src/serial_port.cpp
stand_main.cpp
tests/fake_serial.h
tests/hardware_pty_client.cpp
tests/imu/parser_test.cpp
tests/imu/serial_test.cpp
tests/imu/test_packets.hpp
tests/pty_devices.h
tests/pty_helpers.py
tests/test_application_preflight.py
tests/test_combined_monitor.py
tests/test_demo.py
tests/test_hardware_pty.py
tests/test_library_import.py
tests/test_loop.cpp
tests/test_math_config.cpp
tests/test_monitor.py
tests/test_motor_fake.cpp
tests/test_protocol.cpp
tests/test_safety.cpp
tests/test_serial_pty.cpp
tests/test_shared_serial.cpp
tests/test_stand.py
tests/test_stand_fsm.cpp
tests/test_trajectory.cpp
tests/test_trajectory_safety.cpp
tests/test_watchdog.cpp
udev.md
```

## 2. 参数设置与运行逻辑

- 关节启动角、位置/速度限位、力矩/增益上限宏已恢复在config/real_config.h，MotorConfig构造函数引用这些宏。接线/方向及IMU设备在real_config.cpp；通信时间默认成员在real_config.h。
- MotorConfig默认已有四总线/12项限位，按FL/FR/BL/BR，每总线ID0/1/2对应髋/大腿/小腿。设备/dev/robot/leg_fl、leg_fr、leg_bl、leg_br；IMU/dev/robot/imu。不要改成旧的临时CH343/ACM枚举名称。
- 配置可在构造硬件对象前修改；硬件对象保存副本。RobotConfig为MotorConfig别名，makeDefaultConfig等工厂兼容原调用。人工拓扑必须同步配置限位长度。
- q为输出侧逻辑rad，dq rad/s，tau估计N·m，增益也是输出侧。协议处理6.33减速比。qOut是原始输出角，qL=direction×(qOut−offset)。限位限制qL，不额外加offset。
- 启动根据实测编码器与start_pose_rad归零；首次反馈对齐±3.49/67.43/-165.01度，不是一律0。机器人必须位于这个机械启动姿态，站起来后不得重启再次归零。
- 限位每腿髋[-.523,.523]、大腿[-.9,2.87]、小腿[-3,-.733]，速度30.1、前馈力矩23.7、kp70、kd10。机械边界裕量待实机；前馈限幅不单独约束PD总力矩。
- 电机4Mbps，每请求20ms控制预算，控制5ms节拍目标；monitor/probe预算默认40ms，monitor显示/扫描200ms。四总线并行，同总线逐台请求应答。
- IMU460800、100ms读取等待、500ms过期。四元数wxyz已归一化，gyro rad/s，RPY显示度，数据仍传感器坐标。安装旋转在ImuConfig.mount_quaternion预留，不应用；策略映射亦尚未实现。

控制：main构造配置→GoMotorArray.start健康检查/归零→stand FSM或demo保持目标→setTargets全体校验/裁剪→总线线程/共用MotorTransport→反馈快照。无效数值拒绝，通信故障锁定，停止尝试卸力。stand在打开前校验目标/增益/规划峰值速度，不合法直接报错，不悄悄裁剪轨迹。

监测：main启用GoMotorMonitor四线程＋ImuMonitor线程，单显示线程取快照；缺电机/IMU错误不阻塞其他电机扫描，失败保留旧值和时间但标明失败。IMU不执行电机事务式清输入。不同设备没有同步采样保证，RX按sequence真实解析增量计算，不是显示频率。

RealHardware：start电机一次归零→2秒内等待q+gyro→约5ms健康监督；电机100ms、IMU500ms过期会整机锁定，数据恢复不自动恢复，显式stop/start仍要求规定启动姿态。接口详情见整理记录。

## 3. 构建与运行

从仓库根构建（不自动启动设备）：

```bash
cmake -S imgo2_deploy/real -B /tmp/goimu_user -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/goimu_user -j4
ctest --test-dir /tmp/goimu_user --output-on-failure
```

以下命令由用户在硬件主机按实际机械条件执行，同一串口不可同时被多个程序占用：

```bash
# 机械已架空/托住，联合监测电机和IMU：
/tmp/goimu_user/go_monitor -t 40 -i 0,1,2 -p 200
# 退出诊断，回到规定启动姿态，选择其中一个控制工具：
/tmp/goimu_user/go_pd_demo
/tmp/goimu_user/go_stand
```

monitor的-i选择轮询ID，末尾可指定电机设备；IMU使用统一配置。原probe/独立IMU入口已删除，配置基准仍由启动姿态决定。

stand启动后主动保持，输入完整一行0再回车才开始；8秒五次轨迹到0/.87/-1.82，目标速度取导数，末端零速度后继续保持。时长/增益在fsm/states/stand_state.h的StandSettings，独立工具站姿在stand_main.cpp的buildStandGoal。重复0不重启。demo直接用25/.5保持启动位置。两者只打印必要提示/错误，无持续刷屏、文件日志或退出性能长汇总。

Ctrl+C或SIGTERM退出循环并尝试卸力、回收线程、关闭串口；SIGKILL不能执行清理，断连也不能保证机械卸力。诊断终端用备用屏幕固定刷新并恢复光标；空间不足提示扩大，重定向输出纯文本并持续追加，文件增长正常。

## 4. 状态与错误

| 显示／错误 | 条件与处理 |
|---|---|
| 打开失败、No such file、Permission denied | 硬件主机检查设备/权限；udev已由用户验证，不能据本会话看不到设备否定它 |
| 不是tty、波特率回读不一致 | 串口节点/驱动条件不符，拒绝打开 |
| I/O timeout、feedback timeout | 未按期限收到目标ID合法帧；诊断继续，控制锁故障 |
| disconnected/error、EOF、read/write/poll | 底层错误；不自动重连，修复后显式重启 |
| badFrames、strayFrames | 候选CRC/格式错误、或收到其他ID合法帧；用于诊断线路 |
| WAITING / RECEIVING / STALE / ERROR | 无合法帧／新鲜帧／超过500ms／读取异常；有帧不代表所有字段齐全 |
| RPY/gyro/accel为-- | 对应字段缺失、过期或采集错误；RPY不会从四元数补算 |
| IMU ready timeout | 2秒无q+gyro同帧，整机启动失败并清理 |
| IMU invalid/stale、joint invalid/unhealthy/stale | 整机接口锁定并撤销全部活动目标；数据恢复不自动解锁 |
| unhealthy feedback | 非FOC、设备错误码或温度>80℃；控制锁定，诊断仍显示 |
| no new target for… | 80ms上层失联软看门狗卸力，新目标可恢复；区别于真故障 |
| whole robot stop latched | 整机停止/故障已锁定，拒绝目标 |
| target vector length / nonfinite | 目标长度或NaN/Inf错误，整份拒绝；Motor不发布部分目标 |
| trajectory… | 时长非法、两端目标/增益越界或规划峰值超速，站起拒绝执行 |
| start_pose_rad… | 姿态长度不匹配或非有限，串口打开前拒绝归零 |
| Terminal too small | 窗口尺寸不足，扩大；采集仍运行 |

设备错误码：0正常、1过热、2过流、3过压、4编码器、5欠压、6绕组过热、7未知。收到合法应答不等于设备健康。

CLI严格整数/ID解析；monitor -t 1..10000ms、-p 10..10000ms，probe -t相同、-n 1..10、ID0..14，缺参数或未知选项拒绝。reader正常退出0、参数数量错误1、读取/参数异常2；monitor参数/异常2、即使部分设备错误也可正常退出0；probe参数1、完全无电机应答2、至少一台应答0。不能只看退出码判断所有硬件健康。

## 5. 实机待确认

当前21项离线回归通过，不代替重构后实机。待确认新版本保持/站起/退出、实际频率和延迟、机身IMU坐标、策略关节映射、机械裕量与电机PD总力矩。尚未接入PPO或ROS策略；先同进程一次初始化、站起和保持，后续再接策略，不能站立后重启归零。
