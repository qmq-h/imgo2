# go_imu 使用、结构与故障手册

核对日期：2026-10-11。根据当前源码整理，供下一次实机测试使用。**本轮只写文档，go_imu 文件夹保持不变**。之前的 17/17 项是离线／伪终端测试；用户尚未进行此次重构后的实机测试。

## 1. 先读这些约束

- 本工程是 Linux 单进程 C++ 工具，使用串口和线程；不需要 ROS 2，不加载策略。`rl_real_imgo2` 尚未接入新接口。
- monitor/probe 是主动请求反馈，发送 FOC 全零控制量，会撤销保持力矩；机器人必须先架空或可靠托住。它们不能和正在站立保持的程序同时占用电机串口。
- demo/stand 会主动控制。demo 启动后直接保持当前位置；stand 启动后也已开始保持，输入 `0` 再回车才执行站起轨迹。不要把“还没输入 0”当作没有控制电机。
- 每次控制初始化根据当前编码器值和配置启动姿态计算 offset。**机器人必须摆回配置的启动姿态后才能重新初始化；不能站起来后重启另一个程序，再用同一趴姿假设归零。**监测工具也按首次成功反馈计算本次逻辑角基准，启动姿态错误时 qL 数值会具有误导性。
- 当前 `config.limits` **为空**，没有配置每关节的位置／速度／力矩／增益安全边界。保留了限幅机制不等于已经设置限幅。
- 硬件顺序和 IMU 传感器坐标尚未转换为策略顺序／机身坐标。安装旋转、映射配置只是预留元数据，本轮不应用。
- 退出及故障处理会尝试发送零控制量，但串口断连、进程强杀、系统故障时无法保证机械卸力。Ctrl+C 是受支持退出方式；不要用强杀验证正常退出。

## 2. 文件结构与每个文件职责

以下包含全部源文件、配置和测试辅助文件；已有 build 产物不是源码，重新构建可以生成，不在清单中展开。目录内没有另外一个 README，根 README 是入口。

| 文件（相对 go_imu） | 职责 |
|---|---|
| `CMakeLists.txt` | 库、五个工具、测试的构建定义；Linux 才构建真实串口和应用。 |
| `Imu/inc/imu_sensor/imu_data.hpp` | 字段有效位、四元数、欧拉角、角速度、加速度、sequence 与主机接收时间。 |
| `Imu/inc/imu_sensor/imu_monitor.hpp` | 独立采集线程及互斥快照。 |
| `Imu/inc/imu_sensor/imu_sensor.hpp` | Open/Read/Close 采集接口。 |
| `Imu/inc/imu_sensor/yesense_parser.hpp` | Yesense 解析器接口与缓存。 |
| `Imu/src/imu_monitor.cpp` | 连续采集、每秒 sequence 增量计接收帧率、错误保留。 |
| `Imu/src/imu_sensor.cpp` | 持续读取、分包解析，一次读多帧时返回最新帧。 |
| `Imu/src/yesense_parser.cpp` | 帧同步、校验与 TLV 字段解析／单位转换。 |
| `Motor/inc/go_bus.h` | 同总线逐台事务、健康判定、故障锁定、零命令和关闭接口声明。 |
| `Motor/inc/go_log.h` | 每次运行新日志路径与 CSV 内容接口声明。 |
| `Motor/inc/go_loop.h` | 周期线程、调度及循环异常／统计接口声明。 |
| `Motor/inc/go_motor.h` | 单电机逻辑坐标、方向、offset、命令与反馈状态接口声明。 |
| `Motor/inc/go_motor_array.h` | 多总线线程、一次提交全部目标、启动归零、快照、看门狗接口声明。 |
| `Motor/inc/go_motor_monitor.h` | 多总线容错诊断，start/snapshot/stop，仅全零控制量接口声明。 |
| `Motor/inc/go_protocol.h` | 协议打包／解包、CRC、模式、减速比与输出侧单位换算接口声明。 |
| `Motor/inc/go_safety.h` | 目标有限值检查、限幅与限速斜坡接口声明。 |
| `Motor/inc/go_trajectory.h` | 五次平滑轨迹进度、速度／加速度接口声明。 |
| `Motor/inc/motor_transport.h` | 唯一请求应答核心：报文发送、分包、CRC、ID、统计。 |
| `Motor/src/go_bus.cpp` | 同总线逐台事务、健康判定、故障锁定、零命令和关闭实现。 |
| `Motor/src/go_log.cpp` | 每次运行新日志路径与 CSV 内容实现。 |
| `Motor/src/go_loop.cpp` | 周期线程、调度及循环异常／统计实现。 |
| `Motor/src/go_motor.cpp` | 单电机逻辑坐标、方向、offset、命令与反馈状态实现。 |
| `Motor/src/go_motor_array.cpp` | 多总线线程、一次提交全部目标、启动归零、快照、看门狗实现。 |
| `Motor/src/go_motor_monitor.cpp` | 多总线容错诊断，start/snapshot/stop，仅全零控制量实现。 |
| `Motor/src/go_protocol.cpp` | 协议打包／解包、CRC、模式、减速比与输出侧单位换算实现。 |
| `Motor/src/go_safety.cpp` | 目标有限值检查、限幅与限速斜坡实现。 |
| `Motor/src/go_trajectory.cpp` | 五次平滑轨迹进度、速度／加速度实现。 |
| `apps/app_entrypoints.h` | 五个应用函数声明。 |
| `apps/go_monitor.cpp` | 参数解析、诊断与 IMU 启停、刷新及退出范围汇总。 |
| `apps/go_probe.cpp` | 有限扫描、配置建议、同时发送自检和汇总。 |
| `apps/imu_reader.cpp` | 独立 IMU 调试、参数解析和面板。 |
| `apps/pd_demo_main.cpp` | 当前位置保持、日志与退出统计。 |
| `apps/stand_main.cpp` | 启动保持、输入 0、8 秒站起、日志与退出统计。 |
| `config/real_config.cpp` | 唯一默认接线／方向／启动姿态／节拍／IMU 设备定义。 |
| `config/real_config.h` | 配置数据结构、启动姿态常量、JointLimits、HardwareConfig。 |
| `display/imu_panel.cpp` | IMU 状态与 RPY/gyro/accel 行，缺失或过期显示 --。 |
| `display/imu_panel.h` | IMU 面板函数声明。 |
| `display/motor_panel.cpp` | 完整／紧凑电机面板，诊断统计与 IMU 合并显示。 |
| `display/motor_panel.h` | 联合面板函数声明。 |
| `display/terminal_display.hpp` | 备用屏幕、隐藏／恢复光标、尺寸检查、UTF-8 截断、重定向纯文本。 |
| `go_monitor.cpp` | 薄入口：调用 apps 中的 参数解析、诊断与 IMU 启停、刷新及退出范围汇总应用。 |
| `go_probe.cpp` | 薄入口：调用 apps 中的 有限扫描、配置建议、同时发送自检和汇总应用。 |
| `hardware/real_hardware.cpp` | 组合电机与 IMU，严格启动、5 ms 健康监督、整机锁定及资源退出。 |
| `hardware/real_hardware.h` | 整机状态、命令接口与线程调用约束。 |
| `imu_reader.cpp` | 薄入口：调用 apps 中的 独立 IMU 调试、参数解析和面板应用。 |
| `pd_demo_main.cpp` | 薄入口：调用 apps 中的 当前位置保持、日志与退出统计应用。 |
| `serialport/inc/imu_sensor/serial_port.hpp` | IMU 旧接口适配，调用同一串口底层；不是第二套驱动。 |
| `serialport/inc/serial_port.h` | 唯一串口底层接口、截止时间、读状态。 |
| `serialport/src/serial_port.cpp` | Linux 打开、波特率设置／回读、部分收发、超时和断连。 |
| `stand_main.cpp` | 薄入口：调用 apps 中的 启动保持、输入 0、8 秒站起、日志与退出统计应用。 |
| `tests/fake_serial.h` | 线程安全假串口、发送快照与模拟反馈。 |
| `tests/hardware_pty_client.cpp` | 供整机伪终端驱动调用的客户端。 |
| `tests/imu/parser_test.cpp` | IMU 帧及字段解析。 |
| `tests/imu/serial_test.cpp` | IMU 分包串口读取。 |
| `tests/imu/test_packets.hpp` | IMU 测试帧构造。 |
| `tests/pty_devices.h` | 测试专用设备替换，不对生产工具生效。 |
| `tests/pty_helpers.py` | Python 伪终端协议辅助。 |
| `tests/test_combined_monitor.py` | 电机＋IMU 面板、窄窗口、断连、退出。 |
| `tests/test_demo.py` | 保持控制、日志与退出。 |
| `tests/test_hardware_pty.py` | 整机正常／启动清理／过期／锁定／并行诊断。 |
| `tests/test_library_import.py` | 其他 CMake 工程导入硬件库且不链接显示。 |
| `tests/test_loop.cpp` | 周期循环。 |
| `tests/test_monitor.py` | 默认配置摘要核对；存在真实设备时跳过。 |
| `tests/test_motor_fake.cpp` | 单电机／总线假串口测试。 |
| `tests/test_probe_pty.py` | probe CLI、有限扫描与同时发送。 |
| `tests/test_protocol.cpp` | 协议及输出侧单位。 |
| `tests/test_safety.cpp` | 有限值／限幅／限速与线上命令。 |
| `tests/test_serial_pty.cpp` | 电机分包／CRC／ID 串口事务。 |
| `tests/test_shared_serial.cpp` | 共享串口状态。 |
| `tests/test_stand.py` | 轨迹、日志与退出。 |
| `tests/test_trajectory.cpp` | 轨迹曲线。 |
| `tests/test_watchdog.cpp` | 上层失联卸力与软恢复。 |

## 3. 默认配置与角度含义

| 硬件索引 | 设备 | 总线 ID | direction | 启动角（度，髋／大腿／小腿） |
|---|---|---|---|---|
| 0–2 | /dev/ttyCH343USB1 | 0,1,2 | +,+,+ | +3.49,+67.43,-165.01 |
| 3–5 | /dev/ttyCH343USB2 | 0,1,2 | +,-,- | -3.49,+67.43,-165.01 |
| 6–8 | /dev/ttyCH343USB0 | 0,1,2 | -,+,+ | +3.49,+67.43,-165.01 |
| 9–11 | /dev/ttyCH343USB3 | 0,1,2 | -,-,- | -3.49,+67.43,-165.01 |

此表不声明各总线对应 FL/FR/RL/RR 已经确认。以实测接线和训练关节顺序为准。

- 电机 4,000,000 baud；控制每台一次发送＋接收默认预算 20 ms；控制总线周期目标 5 ms（200 Hz）。monitor/probe 默认每请求预算 40 ms。
- monitor 默认扫描与显示周期 200 ms（约 5 Hz），IMU 线程持续读取；一轮超时较多时扫描周期会变长。四条总线互不等待。
- 上层 demo/stand 目标更新约 1 ms；它不是串口反馈频率。实际 bus_hz、耗时受串口、负载和缺帧影响。
- IMU `/dev/ttyACM0`、460800 baud；单次 Read 等待 100 ms，超过 500 ms 的数据视为过期。波特率不能推导设备输出帧率，面板 RX 使用解析 sequence 每秒增量／实际时间。
- 协议层已处理减速比 6.33。公开 q 为输出侧 rad，dq 为 rad/s，tau 为 N·m，kp 为 N·m/rad，kd 为 N·m/(rad/s)。不要在 RL 适配处再次除减速比。
- `qOut` 是输出侧原始编码器角；`qL = direction × (qOut − offset)`。启动 `offset = qOut首次 − direction × start_pose_rad`，所以启动后的 qL 应接近配置启动角，**不是一律 0**。现有 monitor 帮助文字“标定姿态应接近 0”不符合当前非零 start_pose；本轮冻结代码，以这个公式为准。
- 方向同时作用于位置、速度与力矩；RPY 展示来自设备欧拉角转度，四元数仍保存在采集数据中。投影重力 3 维尚不在此工具生成，未来策略适配须按已确认坐标由四元数旋转重力向量。

## 4. 明天的操作顺序与命令

命令从仓库根目录执行。构建放 /tmp，避免修改被冻结的工具文件夹：

```bash
cmake -S imgo2_deploy/go_imu -B /tmp/imgo2_go_imu_manual_build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/imgo2_go_imu_manual_build -j4
```

先确认设备路径和访问权限：

```bash
ls -l /dev/ttyACM* /dev/ttyCH343USB*
id
```

设备不存在时先排查接线／枚举；Permission denied 时确认当前用户有设备所属组权限。不要把所有无输出都当成算法错误。需要管理员调整权限时按实际设备组处理。

### 4.1 先单独测 IMU（不打开电机）

```bash
/tmp/imgo2_go_imu_manual_build/imu_reader
# 临时指定设备和波特率，仅此程序支持 IMU 覆盖参数：
/tmp/imgo2_go_imu_manual_build/imu_reader /dev/ttyACM0 460800
```

观察 RECEIVING、年龄、RX、RPY/gyro/accel；静止放置后看角速度和加速度是否合理；手动只绕一个已知轴转动，记录符号和轴对应。首次帧率窗口约 1 秒，刚启动 RX=0 不能直接判定没有数据。没有 EulerAngles 时 RPY 是 --，即使四元数有效。Ctrl+C 后检查光标恢复。

### 4.2 确保机械托住，再有限扫描电机

```bash
/tmp/imgo2_go_imu_manual_build/go_probe
# 只检查一个设备，40 ms，每 ID 两次：
/tmp/imgo2_go_imu_manual_build/go_probe -t 40 -n 2 -i 0,1,2 /dev/ttyCH343USB1
```

probe 扫描 ID 0..14；`-i` 只决定汇总显示哪些 ID，**不缩小扫描范围**。先逐口扫描，再执行已有同时发送自检；不是连续控制。检查各口应答 ID、波特率、坏帧、错误 ID、耗时。probe 显示的是 qOut，其 offset 建议不能覆盖已经确认的启动姿态配置。

### 4.3 联合监测

```bash
/tmp/imgo2_go_imu_manual_build/go_monitor
/tmp/imgo2_go_imu_manual_build/go_monitor -t 40 -i 0,1,2 -p 200 /dev/ttyCH343USB1
```

monitor 的 `-i` 决定真正轮询 ID。设备参数只影响电机口，IMU 仍默认 ACM0；它没有生产版 IMU 参数／环境覆盖入口。未经配置登记的设备／ID 按方向 +1、启动角 0 建立临时基准，不能当训练坐标。

终端宽时全信息，普通宽度用紧凑 qOut/qL；小于 53 列或高度不足时出现 Terminal too small。扩大窗口即可，采集仍在运行。重定向时连续追加纯文本，不含终端控制码，文件增长是正常行为：

```bash
/tmp/imgo2_go_imu_manual_build/go_monitor > /tmp/go_imu_monitor.txt
```

Ctrl+C 终止、回收线程并打印本次 qL 范围。**该范围只是观察范围，不能自动当安全机械限位**；本次启动姿态错误或遍历不完整都会影响数值。

### 4.4 再测试保持／站起（需要单独判断机械条件）

先退出 monitor/probe/imu 联合工具，再将机器人摆回规定启动姿态。

```bash
/tmp/imgo2_go_imu_manual_build/go_pd_demo
# 退出、恢复启动机械姿态后，再运行：
/tmp/imgo2_go_imu_manual_build/go_stand
```

demo 以实测逻辑位置保持，kp=25、kd=0.5、dq=0、前馈力矩=0。stand 同样增益，启动后先保持；输入 `0` 回车后四腿到 `{0,0.87,-1.82}` rad，8 秒五次平滑轨迹，两端速度和加速度为 0，完成后持续保持，Ctrl+C 尝试卸力。它们不依赖 IMU；IMU 有错也不会阻止这两个工具启动。

两个程序不提供生产版轨迹时长／设备 CLI；测试专用 GO_TEST_* 变量不改变生产程序。日志约每 20 ms 写一行，默认当前工作目录 `log/run_时间.log`，列 t_s、q_des、q；GO_DEMO_LOG_DIR／GO_STAND_LOG_DIR 可覆盖日志目录。日志打开失败会提示不记录，控制仍可能继续。退出 upper_hz 与 bus_hz 用于区分上层刷新和通信轮次。

离线回归可单独运行，存在真实串口时 monitor 配置摘要测试会跳过：

```bash
ctest --test-dir /tmp/imgo2_go_imu_manual_build --output-on-failure
```

## 5. 代码运行链路

### 5.1 控制工具

外层 main → apps demo/stand → makeDefaultConfig → GoMotorArray.start。
GoMotorArray 验证配置、每总线建线程 → GoBus 注册 ID、打开串口 → 对全部电机零命令健康探测 → 根据 start_pose_rad 归零 → 报告 ready。任一启动失败会停止已启动资源。

上层构造一次 12 关节 Targets（q/dq/tau/kp/kd）→ setTargets 检查长度／有限值、Safety 限幅 → 将快照分发到四个 worker → 同总线按电机 ID 调 GoBus → MotorTransport 打包、清除事务前旧输入、发送、分包／CRC／ID 匹配 → GoMotor 转成逻辑反馈 → GoMotorArray 发布线程安全状态。

控制事务单台共用发送＋接收截止时间。setTargets“一次提交 12 个”是上层接口，不是硬件支持一次广播获得 12 台反馈。不同总线并行；同总线逐台等待。

### 5.2 监测与 IMU

monitor → GoMotorMonitor.start 四个线程，各自逐 ID 全零 exchange → 更新 raw、qL、最近成功时间、失败标记、统计 → snapshot。
另一个 ImuMonitor 线程 → ImuSensor.Read → 共享串口 readSome → YesenseParser 分包缓存 → 最新帧与 sequence → Snapshot。**IMU 不执行电机事务式清输入**。
主显示线程取两个独立快照 → RenderMonitor → TerminalDisplay。设备采样时间没有同步，面板刷新不改变传感器输出频率。

监测缺某电机继续扫该总线其他电机；某串口打开失败不妨碍其他线程。断连后不重新打开端口，故障或无帧不会自动恢复连接。失败保留最后有效数值／时间，但 request_ok/valid=false；读数陈旧不能当本轮新反馈。

### 5.3 未来整机接口（五个工具没有改为使用它）

RealHardware.start → 严格电机初始化 → 启动 IMU → 最多 2 秒等待同帧 quaternion＋gyro → 启动约 5 ms 监督线程。失败会停止电机和 IMU资源。
readState 返回硬件展开顺序的逻辑关节状态、有效性、主机单调时间戳、数据年龄与错误；IMU 仍传感器坐标。
sendCommand 检查整机健康 → GoMotorArray.setTargets；故障／过期／非法目标后锁定整个接口，通知所有总线取消活动目标并发送零量。已经发送的事务不能撤回。

电机反馈默认 age≥100 ms 不可用；IMU age≥500 ms 或缺 q/gyro 不可用。数据恢复不解锁，须显式 stop/start；重新 start 仍需规定机械启动姿态。

GoMotorArray 自己的上层失联看门狗 80 ms 是软卸力，可在新目标到达后接管；GoBus 命令 100 ms 过期故障及 RealHardware 整机数据故障是锁定机制，两者不要混淆。仅使用 demo/stand 的电机接口，不具备 RealHardware 的 IMU 监督和整机故障联动保证。

硬件库链接 `go_imu::hardware` 不引入终端显示。调用示例及 CMake 导入见 [重构记录](go_imu_refactor_2026-10-11.md)。start/stop 由一个上层线程调用，不与其他公开方法并发；启动完成后 readState/sendCommand 使用互斥保护。stop 可重复调用。

## 6. 错误、触发条件与处理

报错可能带设备名／ID／errno。下表是当前代码的错误关键字，完整字符串可能含数值。仅有坏帧统计不等于已经识别具体电机设备故障。

| 输出／状态 | 触发条件 | 当前行为和排查 |
|---|---|---|
| 打开 …: No such file or directory | 设备路径不存在 | 查 ACM0/CH343 实际枚举；监测其他设备继续，控制启动失败 |
| Permission denied | 当前用户无串口权限 | 查设备组和用户组；不是协议错误 |
| 不是 tty 设备 | 参数指向普通文件等 | 使用实际串口节点 |
| TCGETS2/TCSETS2 失败，尝试降级 | 自定义波特率 ioctl 不支持 | 尝试标准波特率方案；最终是否成功看后续打开结果 |
| baud readback mismatch / standard baud readback mismatch | 请求与驱动回读波特率不符 | 拒绝打开，检查驱动支持；不能当已经运行在 4 Mbps |
| invalid baud/timeout (1..10000000 us) | 波特率 0 或串口预算非法 | 使用正波特率和有效超时 |
| I/O timeout / feedback timeout | 期限内没有目标 ID 的合法反馈 | 查供电、接线、ID、波特率、负载；monitor继续，控制故障锁定 |
| serial device disconnected/error / serial EOF / read / write / poll | 拔线、驱动或底层 I/O 错误 | 不自动重连；先退出修复连接，再按正确姿态重启 |
| badFrames 增长 | 候选帧 CRC／格式失败 | 查干扰、速率、线缆；非目标 ID 合法帧计 strayFrames |
| WAITING | IMU 尚未收到合法帧，无底层硬错误 | 可能设备无输出、速率／协议错误；没有总等待超时退出，Ctrl+C 可退出 |
| RECEIVING | IMU有新鲜帧且无错误 | 不代表所有字段都存在，单行 -- 仍可能出现 |
| STALE、字段 -- | 最近帧≥500 ms，或该字段本帧缺失 | 查设备输出配置／丢帧；字段缺失不会自动用其他字段推算 |
| ERROR / IMU error: … | IMU 打开或读取抛异常 | reader退出；monitor保留错误并继续电机诊断 |
| IMU ready timeout (quaternion + gyro required) | 2 秒内没有新鲜 q＋gyro 同帧 | RealHardware启动失败并清理；RPY可显示不代表具备 q字段 |
| IMU invalid/stale: … | 启动后IMU缺必要字段、过期或底层错误 | 整机锁定并请求所有总线卸力；无底层错误时冒号后可为空 |
| joint N: invalid/unhealthy/stale motor feedback | 整机电机 valid=false、过期、错误码、非FOC或温度>80℃ | 整机锁定；N是硬件展开索引 |
| unhealthy feedback (mode/error/temperature) | 电机应答有效但不满足健康条件 | GoBus锁定，诊断工具仍显示／扫描 |
| no new target for … | 上层超过80 ms未更新目标 | GoMotorArray软看门狗卸力；新目标可恢复，区别于真故障 |
| whole robot stop latched | 已 requestUnload／整机故障 | 拒绝目标，不随数据恢复解除 |
| controller is not running | 未成功启动就setTargets | 先检查启动结果 |
| every target vector must have length … | q/dq/tau/kp/kd不是全部12项（默认） | 底层拒绝该调用；经RealHardware调用会锁定整机 |
| target rejected by Safety: every value must be finite | NaN／Inf 目标 | 拒绝；RealHardware锁定；不要只检查q而漏其他数组 |
| startup probe did not report within … / startup probe failed | 启动线程未及时ready或探测失败 | 停止资源，错误可能被具体总线原因替代 |
| 自动归零失败 / no fresh feedback to align against | 没有有效反馈供计算零点 | 控制启动失败，不继续轨迹 |
| feedback-offset overflow / invalid command encoding | 坐标计算溢出或编码参数不合法 | 事务失败，检查数值及配置 |
| shutdown motor … | 关闭时零命令交互失败 | 不能以正常退出断言机械已卸力 |
| start failed: … / control failed: … | demo/stand启动或运行返回失败 | 查看后面的原始原因，程序尝试停止并退出 |
| log: 本次不记录（…） | 不能创建日志目录／文件 | 控制可能继续；检查工作目录权限和日志环境变量 |
| Terminal too small | 窗口宽／高不足 | 放大终端；不是采集失败 |

设备错误码由协议 flags 低 3 位得到：0 normal；1 overheat；2 overcurrent；3 overvoltage；4 encoder fault；5 bus undervoltage；6 winding overheat；7 unknown fault。控制要求错误0、FOC、温度≤80℃；诊断 request_ok=true只说明通信成功，不能说明设备健康。

配置API还会报：至少一条总线／电机、空路径、重复设备、重复或越界ID（0..14）、方向必须±1、方向长度不一致、limits长度必须等于关节数、q_min>q_max、负速度／力矩／增益上限、周期≤0、watchdog负值或不短于command_timeout等。RealHardware构造器另检查超时均为正和IMU／电机设备路径重复（`hardware timeouts must be positive`、`IMU and motor device overlap`）。调用者须捕获异常；这些不是设备反馈故障。

CLI：reader参数过多提示Usage返回1；错误波特率／IMU打开读取错误返回2，正常Ctrl+C为0。monitor未知参数、无效ID或-p不在10..10000返回2；其正常退出为0，即使部分设备失败也可能返回0，**退出码不能代替面板健康判断**。probe参数-t需1..10000、-n需1..10、-i需0..14且非空，非法时返回1；完全没有电机应答时返回2；只要有一个应答就可返回0，部分缺失仍需看报告。demo/stand失败返回1。

当前CLI用atoi/atol等宽松解析，非数字输入可能被当0；monitor -t并没有与probe同样的完整边界检查；probe未知选项可能被当设备路径。明天使用本文列出的纯数字参数，不依赖错误参数被严格拒绝。本轮只记录，未修改解析实现。

## 7. 明天需要留下的实机结果

1. 实际接线设备／ID、每关节正方向及总线对应腿；启动姿态照片或角度记录。
2. IMU静止／单轴旋转的RPY、gyro、accel与RX、age；确认安装轴和符号。
3. monitor宽／窄窗口、Ctrl+C光标恢复，缺电机时其他总线是否持续反馈。
4. probe坏帧／错误ID／收发耗时；demo/stand退出bus_hz、bus耗时、日志。
5. 按机械边界与保守裕量确认安全限幅，不能只复制一次监测范围。

尚待确认：实际通信频率、全零指令真实卸力效果、机身坐标、关节映射、每关节限幅、重构后实际站起和日志。上述条件完成前不把新接口称作“已经能直接运行策略”。

## 8. 本轮维护与核对

已完成：全文件职责、配置、程序逻辑、命令、状态／错误条件和测试准备说明；工具目录冻结。文档依据为源码阅读与路径／指纹核对，本轮没有再次运行编译、离线测试或任何硬件程序。上一轮17/17的证据见重构记录。
待修复／待确认：CLI严格数字解析、monitor帮助中非零启动姿态的解释，以及第7节实机条件。它们本轮保留原样，以免改变待测试版本。
