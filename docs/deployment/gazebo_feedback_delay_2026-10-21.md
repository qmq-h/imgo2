# Gazebo ROS2可配置关节/IMU反馈延迟

2026-10-21。按用户要求实现第一版：关节反馈默认额外延迟1.5ms，IMU独立可配置、默认0ms，先便于单项对照。只改ROS2 Gazebo部署路径、配置和离线测试；没有修改real工程、关节宏、模型质量/增益、策略权重或运行硬件。

## 1. 参数与启动

在`imgo2_deploy/ros2_ws/policy/imgo2/base.yaml`的`imgo2:`下：

```yaml
  gazebo_motor_feedback_delay_ms: 1.5
  gazebo_imu_feedback_delay_ms: 0.0
```

单位ms，允许有限值0–1000，支持1.5等小数；0关闭该通道附加延迟。负值、NaN/Inf、超过1000启动时报错。没有这两个键的旧配置默认0，不改变旧配置语义。

本机已经重新编译`rl_sim`；安装目录的可执行文件为build软链，当前`ros2 run`会使用新版。以后只改以上YAML数值无需重新编译：**停止并重新启动rl_sim**读取新值，Gazebo可保留。不是进入PPO时才读这两个值，也不支持运行中自动重载。即使用`ros2 param set`改变参数服务器值，也不会重新配置缓冲，测试请通过重新启动或启动参数设置。

加载ROS与workspace环境后，正常运行：

```bash
ros2 run imgo2_deploy rl_sim
```

单次测试也可用ROS启动参数覆盖YAML，例如独立测IMU1.5ms：

```bash
ros2 run imgo2_deploy rl_sim --ros-args \
  -p gazebo_motor_feedback_delay_ms:=0.0 \
  -p gazebo_imu_feedback_delay_ms:=1.5
```

两者都0是关闭基准。启动日志打印实际采用值：`Gazebo additional feedback delay (host time): motor=... ms, IMU=... ms`。

## 2. 延迟准确含义

- 只对C++ `rl_sim::GetState()`向FSM/策略提供的反馈生效。关节q/dq/tau整帧一起延迟，保留原joint_mapping；IMU四元数/gyro同一消息一起延迟。
- 回调按**主机steady_clock到达时间**存历史，GetState在现在−配置延迟之前选择最新已到期消息。没有sleep、插值、外推或合成四元数。回调与取样用锁保护，控制给推理的状态副本也在同一锁下复制，避免此路径读半份状态。
- 这是原有采样/ROS/控制取样延迟之外的附加消息交付延迟；不代表从物理采样到策略的完整延迟。ROS消息header与设备内部采样时间未纳入本模型。
- 1.5ms不会取整成5ms，但离散传感器样本采用保持：200Hz IMU每5ms一帧，实际可见样本可能比配置值再旧0–5ms及调度时间。关节消息同理。0ms也不意味着实际数据年龄为0。
- 使用主机时间是为了与现有C++ LoopFunc线程时钟一致。Gazebo暂停期间已到达消息仍按真实时间到期；实时因子不接近1时，不能把1.5主机ms直接当1.5仿真ms。
- 不改变命令下发路径，也不延迟Gazebo关节控制器内部PD所用的物理状态。因此模拟的是外部FSM/策略的反馈链路，不是电机内部PD延迟，也未模拟逐电机批量发送偏移、长尾/随机抖动。
- 初始尚无已到期消息时保留初始化状态，不提前泄露新反馈。缓存随时间推进保留最新到期帧与尚未到期帧，不无限保留整段运行历史。

默认对FSM各状态都生效，包含起身和PPO；此版本不是只在运动状态启用。尚未实现随机延迟或每关节不同延迟。

## 3. 实现位置

- `include/feedback_delay.hpp`：独立、无ROS依赖的因果延迟缓冲，调用侧负责同步。
- `include/rl_sim.hpp`、`src/rl_sim.cpp`：ROS2回调历史、GetState取延迟反馈、控制→推理状态快照锁和配置读取。
- `policy/imgo2/base.yaml`：两项默认参数。
- `test/test_feedback_delay.cpp`、CMake：小数边界、独立通道、0关闭、warmup、保持、缓存窗口/回退、非法参数以及q/dq/tau包一致性。
- `scripts/run_gazebo_ppo_ramp.py`仅补充base.yaml输入哈希，使延迟实验配置可追溯；速度计划与控制流程未变。

ROS1、MuJoCo与真实硬件没有接入此功能。本轮不声称修复整个仿真程序的所有线程共享状态，仅同步与新增反馈读取相关的数据路径。

## 4. 已完成验证

1. 独立g++ C++17、Wall/Wextra/Werror构建并运行延迟测试通过。正式rl_sim重新构建成功，CMake反馈测试CTEST **1/1通过**；既有Gazebo Python离线测试 **7/7通过**。脚本py_compile与git diff --check通过。
2. 两组独立无界面Gazebo短测均完成起身+12仿真秒PPO：初末零指令各2秒、四段变速各2秒。只用于功能连通与年龄验证，不代表运动稳定性完整验收。当前策略/增益/模型保持用户现值，没有为测试改YAML。
3. 第一组YAML默认motor1.5/IMU0；第二组通过ROS启动参数覆盖motor0/IMU7.5，验证独立通道和覆盖路径。本轮进程均停止，输入哈希前后一致，没有连接真实硬件。

可选`IMGO2_FEEDBACK_DELAY_TRACE=/tmp/delay.csv`在GetState记录反馈到达年龄和ready标志，默认不创建该日志；测试启用，正式正常运行无每周期输出。记录的是历史消息回调到GetState的年龄，不是仿真物理采样年龄。该开关会产生诊断写盘开销，不用于精确调度性能基准。

| 配置motor/IMU | 实测motor年龄min/median/P99 | 实测IMU年龄min/median/P99 |
|---|---:|---:|
| 1.5/0ms | 1.501/1.975/2.589ms | .002/2.549/4.957ms |
| 0/7.5ms | .001/.515/1.088ms | 7.502/9.957/12.512ms |

所有ready样本均未提前释放到期时间之前的数据。IMU7.5ms的中位约10ms符合离散5ms样本保持。不能因此称“7.5ms没有生效”，也不能把0配置时的2.5ms重复当新增延迟。

[验证JSON](gazebo_feedback_delay_2026-10-21.json)保存输入/代码哈希与各组分位数；[motor1.5诊断CSV](gazebo_feedback_delay_2026-10-21_motor15.csv.gz)、[IMU7.5诊断CSV](gazebo_feedback_delay_2026-10-21_imu75.csv.gz)可直接核验阈值。原始IMU/odom和过程日志暂存在`/tmp/imgo2_feedback_*_run_20261021/`，不作为持久运动品质证据；本轮没有绘制/宣称新稳定性结果。

### 离线重跑

```bash
g++ -std=c++17 -Wall -Wextra -Werror \
  -I imgo2_deploy/ros2_ws/src/imgo2_deploy/include \
  imgo2_deploy/ros2_ws/src/imgo2_deploy/test/test_feedback_delay.cpp \
  -o /tmp/test_feedback_delay
/tmp/test_feedback_delay
```

常规colcon构建会编译测试；仅构建本目标可使用`cmake --build build/imgo2_deploy --target test_feedback_delay`，之后`ctest --test-dir build/imgo2_deploy -R feedback_delay --output-on-failure`（工作目录为ros2_ws）。项目开关`IMGO2_BUILD_FEEDBACK_DELAY_TEST=OFF`可关闭此测试目标。

## 5. 待确认

- 长时、固定速度/横移/yaw和多档延迟的振荡敏感性尚未系统扫描；本轮只证明延迟实现/配置/取样年龄可用。
- 真实批量收发源码与端到端采样/应用时标仍待补齐。配置1.5ms是用户希望的试验点，不代表已证明全部真实额外延迟恰为1.5ms。
- 单独电机延迟、单独IMU延迟、组合延迟需保持相同模型/增益/策略分别比较；不能同时改噪声或Kp再归因。本轮没有确认实机振荡根因或修复它。
