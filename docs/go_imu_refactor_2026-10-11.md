# go_imu 通信与整机接口重构（2026-10-11）

## 现象与根因

go_monitor 原入口 532 行，直接管理串口、报文与显示，各总线由主线程串行扫描。
GoBus、probe、monitor 各自维护一份反馈组帧逻辑，容易出现行为漂移。
已有一次提交全部目标的 GoMotorArray，但控制模式要求全部健康且锁故障，不能直接用来替代容错诊断。

## 已修复

只改 go_imu 与根文档；原 go、imu_sensor 和 rl_real_imgo2.cpp/.hpp 保持不变。

### 通信与监测

- MotorTransport 是唯一的电机发送／反馈组帧组件（头文件实现，便于既有假串口测试替换）。发送与接收共用截止时间，保留 CRC、分包与目标 ID 校验，统计字节、坏帧、其他 ID。
- GoBus 在共用组件之后完成方向、零偏转换及健康检查；控制模式的限幅、看门狗和故障锁定保持。
- GoMotorMonitor 每条总线一个线程，只构造全零 FOC 命令，没有接受运动目标的方法。单电机无回复仍继续该总线其他电机；某总线等待不阻塞其他总线。
- snapshot 提供 qOut（feedback.q）、qL、方向、零偏、dq/tau、温度、设备错误、request_ok、最近成功接收时间、单次请求错误和线路统计。失败保留旧值与旧成功时间，同时 valid/request_ok=false。
- probe 使用同一组件有限扫描；同时发送自检仍先逐口发送、再逐口接收，没有改为持续控制。
- 诊断正常回复不等于电机健康，诊断保持显示设备错误；不会自动打开断开的串口。

### 入口、显示与构建

五个顶层 .cpp 各 5 行，只调用 apps 中的应用入口。apps 保存参数、轨迹、日志与生命周期；display 保存电机／IMU 面板和终端控制。
Motor/serialport/Imu/config 职责保留，新增 hardware/apps/display。
stand/demo 不启动 IMU，运动轨迹、增益、方向和输出侧单位不变。monitor 默认同时读取 IMU，200 ms 显示周期与原 CLI 保持。
宽／窄终端、备用屏幕、防滚屏、RPY 与角速度／加速度显示，以及退出汇总保留。

构建目标分为硬件 real_hardware、显示 go_imu_display、应用 go_imu_apps。
外部工程链接别名 go_imu::hardware；设置 GO_IMU_BUILD_APPS=OFF 可关闭全部显示／应用目标。
原 ImuSensor::Open/Read/Close 与 GoMotorArray::setTargets 保留。

## 整机接口契约

公开头文件：imgo2_deploy/go_imu/hardware/real_hardware.h。

```cpp
real_hardware::RealHardware hardware(real_config::makeHardwareConfig());
std::string error;
if (!hardware.start(error)) {
    // error 是具体初始化错误；失败已停止已启动资源。
    return;
}
auto state = hardware.readState();
// go::Targets cmd 的所有数组长度必须等于关节数量。
// q/dq/kp/kd/tau 全部使用电机输出侧单位、硬件展开顺序。
// hardware.sendCommand(cmd) 返回 false 后从 readState().error 读取原因。
hardware.stop();
```

- start/stop 必须由同一上层线程调用，且不与其他公开调用并发；启动完成后 readState/sendCommand 由互斥锁保护。
- start 保持电机健康探测与启动姿态归零；等待同帧有效 quaternion + gyro，默认最多 2 秒。无需加速度即可满足这里的 IMU 就绪契约。失败时关闭已启动资源。
- readState 返回每关节反馈、receive_time_ns、age_ms、valid/error，及 IMU 数据／年龄／有效性、started/faulted/error。时间是主机单调时钟，不宣称设备同步采样。
- 电机过期默认 100 ms，IMU 默认 500 ms；阈值和 IMU 就绪期限由 real_config::HardwareConfig 配置。
- 私有监督线程每约 5 ms 检查数据。故障、过期、缺必要 IMU 字段或非法目标会锁定整机，sendCommand 拒绝正常运动。
- GoMotorArray::requestUnload 发布锁定的全机停止标志；GoBus 每次正常电机事务前检查并切换到零命令，剩余总线不继续正常运动。已经发送或正在执行的事务无法撤回；本方案不宣称硬实时或机械卸力已确认。
- 数据恢复不会解除整机锁定；必须显式 stop/start。每次重新初始化都按规定启动姿态归零，不能机器人站立后直接重启并把任意姿态当启动姿态。
- 原 GoMotorArray 上层失联看门狗行为保留；单独的底层看门狗和整机数据故障锁定是不同机制。

外部 CMake 使用：

```cmake
set(GO_IMU_BUILD_APPS OFF CACHE BOOL "" FORCE)
add_subdirectory(imgo2_deploy/go_imu)
target_link_libraries(your_target PRIVATE go_imu::hardware)
```

## 坐标与策略接入限制

硬件关节顺序仍为配置总线／ID 展开顺序；IMU 仍为传感器坐标。
HardwareConfig 预留 joint_names、policy_to_hardware、imu_mount_quaternion 与确认标志；默认映射为空、安装旋转为单位四元数，确认标志 false。
这些参数是待校准元数据，本轮不应用映射／旋转；状态中的 policy_ordered、imu_in_body_frame 恒 false，避免把已提供元数据误当已完成转换。
将来接 rl_real_imgo2 时须显式完成训练关节映射、机身坐标转换和状态／动作快照对接；本轮没有修改真机入口，也没有进行策略推理。

## 验证

Linux /usr/bin/c++ GCC 11.4.0、/usr/bin/python3 3.10.12、CMake 3.22.1。
Release 构建目录 /tmp/imgo2_go_imu_refactor，全部程序及库构建成功；CTest 17/17 通过（约 16.08 秒）。
既有 14 项全部迁移通过，新增三项：

1. hardware_interface（10 个伪终端场景）：正常反馈／命令、IMU 不存在／缺字段、启动电机缺失、IMU 超时／断连、电机故障／反馈过期、NaN 拒绝和全总线零指令、故障锁定与显式重启、并行容错诊断及旧值／时间／坏帧／其他 ID 统计。
2. library_import：独立临时工程 add_subdirectory 并链接 go_imu::hardware；编译、运行未启动硬件的消费者，确认链接命令中无显示／应用库。
3. probe_pty：有限扫描、参数／帮助、两口同时发送自检、CRC 和零控制量。

原 safety 测试暴露假串口 sent 容器在采集线程写入时被测试线程 clear/遍历的竞态，引发堆错误；仅修复 go_imu 测试副本，用互斥锁与按值快照，不修改旧 go 测试。
测试驱动还修正 stdout 缓冲导致故障事件观测延迟的问题，确保检查的是实际故障后的零指令窗口。
继承的测试辅助代码存在 GCC 警告，不作为硬件已验证证据。
未运行真实电机、未连接真实 IMU、未训练、未导出策略。tracked-ignore 检查为空，git diff --check 通过。

原两目录及 RL 真机 cpp/hpp 共 889 文件实施前后逐文件 SHA-256 一致。其中两目录排序路径→SHA-256 字典的 json.dumps(sort_keys=True) 指纹：

| 目录 | 文件数 | SHA-256 |
|---|---:|---|
| go | 793 | d81d42e359d7f451f2540ed1570d25c3c124fcb2ad69dd3a6541f4286aabf547 |
| imu_sensor | 94 | 75bc5ee245026d3ad96ebfb6e887ca875cdf7a7dcc296c096e3c6e46cc0c45da |

包含原目录的 .git 与构建产物，只证明本轮未修改，不用于跨机器版本比较。
pull 因根 .git/FETCH_HEAD 只读失败；保留当前已有未提交改动，没有提交或推送。

## 待确认／待修复

- 真实通信周期、IMU 接收频率、USB 调度与显示：需在连接硬件的电脑构建并实测。
- 训练关节名到硬件 ID 的映射、IMU 安装旋转和零偏：需明确接线、安装方向与轴向实验；当前不能直接将整机状态宣称为策略输入。
- 电机限幅仍沿用空配置，启动归零依赖固定姿态：接策略前需确认安全范围与启动约束。
- 全总线停止的实际机械效果、姿态过渡和动态控制：需有支撑条件的实机验收；模拟报文不能证明机械卸力。
- 只有整机硬件接口已准备，rl_real_imgo2 适配及 sim2real 仍待实施。
