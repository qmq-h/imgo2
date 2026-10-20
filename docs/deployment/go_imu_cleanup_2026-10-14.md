# go_imu 五阶段整理与验证（2026-10-14）

> 2026-10-15入口更新：ppo_check、imu_reader、go_probe源码及构建目标已按用户要求删除。本文相关命令/测试为历史记录，当前使用 [fsm_check指南](fsm_check_2026-10-15.md) 和 [入口精简记录](go_imu_entry_cleanup_2026-10-15.md)。

## 现象与已修复

宏和单位常量散落，默认配置依赖工厂函数；应用流程藏在apps的RunXXX，顶层main只有转发；stand/demo重复初始化与输出、持续打印、写文件日志。按用户计划整理五阶段，不接策略、不启动真实硬件。

1. **数学工具**：新增function/my_myth.h/.cpp。constexpr π、角度系数与内联角度转换/clamp；min/max用标准库。五次曲线唯一实现放cpp，SmoothTrajectory保留薄封装；协议单位、IMU单位和显示复用常量，CRC留在协议。阶段回归protocol/trajectory/imu_parser/safety/stand_fsm 5/5通过。
2. **配置类**：MotorConfig默认构造提供四总线及12限位；ImuConfig默认构造提供IMU参数，另保留四参构造供原调用。构造只填数据；Motor在启动验证，ImuConfig.validate在启动检查。原RobotConfig是MotorConfig别名，工厂函数返回默认构造对象。阶段相关7项回归通过。安装元数据现在属于ImuConfig.mount_confirmed/mount_quaternion；HardwareState的imu_mount_confirmed名字保留。
3. **安全入口**：setTargets仍整份校验/有限目标裁剪后发布，NaN/Inf拒绝、裁剪计数保留。新增Motor::Safety::trajectoryValid供站起启动前检查两端位置、增益及五次规划峰值速度；不再打印warning后继续走被裁剪轨迹。归零姿态长度与有限性在串口打开前拒绝。所有限幅仍是输出侧逻辑值。
4. **真实main**：五个顶层入口直接包含配置、启动、运行循环、退出和异常处理。apps仅runtime.h：共享信号、非阻塞字符读取、严格整数/ID解析。去除RunXXX、app_entrypoints及合并应用库。串口读取仍调用MotorTransport、GoMotorArray/Monitor和ImuMonitor；诊断流程各自需要的扫描/显示辅助函数留在main上方。probe不再读cpp文本生成易误用的配置建议，保留有限扫描、原始角/统计和同时发送自检；修正自检某口重开失败时设备标签可能错位。
5. **输出/日志**：删除go_log头/源/目标及相关调用、日志目录环境变量、日志测试辅助；不删除用户已有日志文件。stand/demo只保留输入、开始、完成、保持提示及错误；不逐周期刷屏，不输出性能长汇总。monitor/probe/reader保留诊断显示及终端恢复。内部通信/循环统计API仍可查询。

## 配置与公共接口

唯一默认参数在config/real_config.h/.cpp，类声明在h、接线/姿态/一腿限位默认表和传感器设备在cpp，通信节拍默认成员在h。

```cpp
go::MotorConfig motors; // 已包含四总线与12项限位
motors.period = std::chrono::milliseconds(5);
// 可在启动之前修改motors.buses/limits等；运行对象保存快照。
go::GoMotorArray array(motors);

real_config::ImuConfig imu;
imu.device = "/dev/robot/imu";
imu.validate();

real_config::HardwareConfig hardware;
hardware.imu.mount_confirmed = false;
// hardware.imu.mount_quaternion 为wxyz；仍仅元数据，不应用安装旋转。
```

兼容：RobotConfig类型名、makeDefaultConfig、makeImuConfig、makeHardwareConfig、setTargets、ImuSensor读取接口、RealHardware整机接口及go_imu::hardware CMake别名保留。**RobotConfig默认构造现在带真机默认总线/限位**，人工拓扑调用方必须覆盖buses并提供同长度limits（或明确clear表示不限）。已同步全部本工程夹具。

不兼容/删除：go_log/Logger与GO_DEMO_LOG_DIR/GO_STAND_LOG_DIR；RunXXX/app_entrypoints和go_imu_apps目标；HardwareConfig的imu_mount_confirmed/imu_mount_quaternion移到imu.mount_confirmed/mount_quaternion。头文件使用方按上例迁移，不存在两套安装元数据。

安全数值保持：hip[-.523,.523]、thigh[-.9,2.87]、shank[-3,-.733]rad，dq30.1、前馈tau23.7、kp70、kd10。启动角保持原±3.49/67.43/-165.01度；方向和/dev/robot设备路径不变；电机4Mbps、20ms事务、5ms周期、80ms看门狗/100ms命令期限，IMU460800、100ms读取、500ms过期。站立8秒0/.87/-1.82、kp25/kd.5、速度前馈保留。

数学clamp只计算数值，不承担NaN策略；Motor安全函数校验后调用它。无每周期数学模块堆分配；曲线复用cpp，简单函数编译期/内联。原命令快照复制与线程锁仍保留，未声称达到硬实时或提高了实际频率。

## 验证依据

Linux C++14兼容构建（GCC）、/usr/bin/python3；临时Release目录/tmp/goimu_cleanup。全部库、五工具、专用伪终端入口构建成功；最终CTest **21/21通过**（最终回归16.63秒），git diff --check通过，tracked-ignore检查为空。

- 原18项回归保留通信、单位、方向、限幅、看门狗、故障锁定、接口清理、导入无显示依赖、联合面板、站起轨迹与速度、信号退出。
- math_config：度/弧度含整数输入、clamp、曲线，默认构造设备/方向/启动角/超时、非法姿态启动前拒绝、配置副本不受外部修改影响。
- trajectory_safety：默认站起合法、超速/越界/增益超限/NaN拒绝、共同CLI解析。
- application_preflight：CLI非法参数、零/NaN/过快轨迹拒绝；专用/dev/null人工设备在打开前拒绝，无真实串口。
- demo/stand伪终端测试已删除文件日志/终端性能统计断言，保留线上CRC、目标、增益、轨迹平滑性、故障与退出；在独立临时工作目录确认没有生成任何文件，并限制输出行数。循环质量统计由既有loop/硬件测试继续覆盖。

保护范围：实施开始保存policy、参考FSM、rl_real_imgo2头/源中现存13个文件的SHA256，实施后逐文件一致。**本轮开始时原go和imu_sensor目录已经不在当前工作区**，没有创建、删除或修改它们；不能声称对不存在目录做了内容指纹验证。历史保留记录仅适用于当时版本。git pull因.git/FETCH_HEAD只读失败，保留现有未提交改动；未提交或推送。

## 待确认

用户已确认udev和monitor实测正常，保留该硬件事实。本轮重构后的控制/退出、实际站起、频率、机械限位裕量及真实卸力仍待实机；tau_max仅限制前馈，不代表PD总力矩上限。IMU安装旋转和策略映射仍需适配，未接入PPO或改rl_real_imgo2。

## 用户要求恢复关节参数宏

已恢复real_config.h中原15个参数宏：3个启动角、3个电机力矩/增益上限、髋/大腿/小腿各位置上下限和速度上限。MotorConfig构造函数引用这些宏，参数数值和其余结构不变；数学π/换算仍在my_myth。关节参数宏是用户设置入口，以后未经同意不删除或替换。

验证：重新构建成功，math_config、safety、trajectory_safety、stand_fsm、library_import五项回归通过；未运行真实硬件。实际机械裕量与新版本实机控制仍待DEPLOY-09。
