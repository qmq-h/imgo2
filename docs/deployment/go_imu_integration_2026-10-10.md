# go_imu 合并与联合监测（2026-10-10）

> 2026-10-15入口更新：ppo_check、imu_reader、go_probe源码及构建目标已按用户要求删除。本文相关命令/测试为历史记录，当前使用 [fsm_check指南](fsm_check_2026-10-15.md) 和 [入口精简记录](go_imu_entry_cleanup_2026-10-15.md)。

## 现象、根因与范围

原 go 与 imu_sensor 均可独立工作，但串口底层重复、程序入口分散，电机与 IMU 无统一监测入口。
按用户要求新增独立 imgo2_deploy/go_imu；原两目录全部保留且未修改。
本轮不接入策略推理、ROS 2，不启动真实电机。

## 已修复与实现

目录：Motor/inc 与 src 保存电机库；serialport/inc 与 src 保存唯一串口实现；
Imu/inc/imu_sensor 与 src 保存数据、解析、读取、采集线程与面板；
config/real_config.h 与 cpp 保存电机及 IMU 配置；所有程序入口在外层，测试在 tests。
未复制第三方驱动、编译产物和子目录 README；新工程不依赖旧两目录。

### 配置与程序兼容

- 电机接线、ID、方向、启动姿态、5 ms 总线节拍、80 ms 看门狗等与原 go 一致；makeDefaultConfig 接口保留。
- go_pd_demo 和 go_stand 与原入口相比仅修改头文件及说明路径，不改变控制流程。
- go_probe 保留扫描、参数及控制语义；默认设备与波特率改从统一配置取得，配置建议的源码读取路径改为 config/real_config.cpp。
- go_monitor 保留电机参数、零控制量轮询和退出汇总，增加独立 IMU 采集线程。
- IMU 默认 /dev/ttyACM0、460800、读取超时 100 ms、过期阈值 500 ms，全部在 real_config.cpp 设置。修改后需要重新编译。
- imu_reader 不传参数时使用统一配置，仍支持可选 device 和 baud 参数单独排查。

### 唯一串口实现

底层采用 go 的 SerialPort，保留 4 Mbps、BOTHER/标准波特率设置与回读、共同截止时间、部分写入、请求前清理输入等电机能力。
新增 readSome 返回 Success/Timeout/Error 与字节数，支持 timeout=0 立即检查。
IMU 原 SerialPort 头变为薄适配层，委托共同底层并将真实错误转为异常；没有第二份 termios 实现。
IMU 流式读取不执行请求前清理输入，保持解析器分包缓存、最新帧选择和原 ImuSensor::Open/Read/Close 接口。
每台设备独立串口对象，不共用文件描述符；采集线程独占 IMU 对象，主线程只读互斥保护的快照。

### 显示与故障

电机区下方显示 RPY（度）、角速度（rad/s）、加速度（m/s²），不显示四元数；库仍保留四元数供将来策略使用。
RPY 使用设备 Euler 字段，而非另行选取四元数的欧拉角分支。没有字段或超过 500 ms 未收到有效帧时显示 --。
状态行显示设备、WAITING/RECEIVING/STALE/ERROR、最新帧年龄、RX 接收帧率。
RX 使用 sequence 计数在约一秒窗口内的增量除以实际时间，包括一次读取中被跳过显示的有效旧帧；它是主机成功解析帧率，不是保证的设备内部采样率。
电机显示默认 200 ms；IMU 持续独立读取，显示频率不会限制 IMU 接收频率。
打开失败或断连只显示错误，电机轮询继续；不自动重连。Ctrl+C 回收线程并关闭串口。

交互终端使用备用屏幕、固定位置刷新、隐藏光标，正常退出及栈展开恢复；只有主线程打印。
宽终端保留原电机详情；宽度不足 200 列或高度不足 24 行使用紧凑的 qOut/qL 与错误码显示。
行内容按显示列宽裁剪（含中文），高度不足或宽度不足 53 列时显示扩大窗口提示，避免自动换行和持续滚屏。
重定向输出使用普通文本，不输出终端控制码。

## 构建与运行

从仓库根目录：

```bash
cmake -S imgo2_deploy/go_imu -B imgo2_deploy/go_imu/build -DCMAKE_BUILD_TYPE=Release
cmake --build imgo2_deploy/go_imu/build -j
ctest --test-dir imgo2_deploy/go_imu/build --output-on-failure
```

新工程五个入口：

```bash
./imgo2_deploy/go_imu/build/go_monitor
./imgo2_deploy/go_imu/build/imu_reader
./imgo2_deploy/go_imu/build/go_probe
./imgo2_deploy/go_imu/build/go_pd_demo
./imgo2_deploy/go_imu/build/go_stand
```

monitor/probe 发送零指令请求卸力，demo/stand 有主动控制；实机运行应分别选择程序，不能同时占用同一串口。
电机串口权限沿用原环境。监测实际接收帧率需至少等待一个统计窗口。

## 验证依据

环境：Linux，/usr/bin/c++ GCC 11.4.0，/usr/bin/python3 3.10.12，CMake 3.22.1。
临时构建目录 /tmp/imgo2_go_imu_build；Release 构建全部五个入口及测试目标成功。
CTest 最终 14/14 通过，约 9.25 秒：共同串口状态、串口 PTY、联合监测、demo/stand PTY、配置摘要、循环、协议、轨迹、模拟电机、看门狗、限幅、IMU 解析、IMU 串口。
联合监测一项包含三个场景：正常/分包/多帧约 100 Hz 接收统计与 RPY 数值、缺字段/过期/断连后电机继续零指令轮询；IMU 不存在；宽/普通/小终端刷新与 Ctrl+C 恢复。
只使用模拟设备；验证电机报文 CRC 和全零控制量，未触及真实电机。
新增串口测试同时验证 460800 与 4000000 驱动回读、立即读取、等待超时、断连、关闭后的错误状态。
继承的测试辅助代码仍有 GCC stringop-overread 及未使用 write 返回值警告，本轮未扩大范围处理。
git ls-files -i -c --exclude-standard 输出为空；没有新增忽略规则。

实施前后全部原目录文件的逐文件 SHA-256 字典一致（含其原有构建产物）：

| 原目录 | 文件数量 | 指纹 SHA-256 |
|---|---:|---|
| go | 793 | 2db83dadce48f8eb74e8ce4fd7e8fe53d070331f6f9687adfd78d8cb8c7bd4fd |
| imu_sensor | 94 | 75bc5ee245026d3ad96ebfb6e887ca875cdf7a7dcc296c096e3c6e46cc0c45da |

指纹算法为以仓库相对路径为键、文件 SHA-256 为值的排序字典，json.dumps(sort_keys=True) 后再 SHA-256；数量和指纹只证明本次工作期间未修改，不作为另一机器构建产物的版本契约。
基于本地 fc8b341 和已有未提交工作；git pull --ff-only 因 .git/FETCH_HEAD 只读失败，未提交或推送。

## 待确认／待修复及所缺条件

- 实机联合监测显示、IMU 实际接收帧率与串口通信：需在连接设备的电脑构建并运行新 go_monitor，查看至少数秒数据。
- IMU 安装坐标、静止零偏、更新频率：需真实静止/轴向旋转实验；本轮不做机身坐标转换。
- 电机配置限幅仍为空、启动归零依赖规定机械姿态，均沿用原代码；未来策略接入前需确认安全范围和初始化约束。
- 真机 RL 入口尚未链接 go_imu，本轮硬件整理不代表 sim2real 完成。
