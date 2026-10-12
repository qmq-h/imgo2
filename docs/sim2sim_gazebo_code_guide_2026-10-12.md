# Gazebo sim2sim 源码阅读与运行逻辑

核对日期：2026-10-12。范围：当前 ROS2/Humble + Gazebo Classic 路径、键 3 AMP。按源码核对，本轮没有编译、运行仿真或更改控制逻辑。

## 1. sim2sim 在做什么

Isaac Lab 中训练好的 actor 经 AMP `play.py` 导出为 TorchScript `policy.pt`，部署端把 Gazebo 中的反馈拼成训练时同样的观测，运行网络，再把动作变为机器人关节目标。Gazebo 计算力矩作用下的运动与碰撞，产生下一次反馈，形成闭环。

训练 checkpoint 的优化器、critic、AMP 判别器、奖励与专家数据不参与当前 Gazebo 推理。部署选择 `policy.pt` 和对应 YAML。奖励与专家动作影响训练出的 actor；它们不在部署时逐步计算。

## 2. 文件分工

以下源码路径以仓库根为起点。

| 文件 | 职责 |
|---|---|
| `imgo2_deploy/src/imgo2_deploy/launch/gazebo.launch.py` | 启动世界/GUI，解析并生成 URDF 实体、robot_state_publisher、joint_state_broadcaster、参数节点和手柄节点 |
| `imgo2_description/urdf/imgo2_real.gazebo.urdf` | 当前 real 策略对应的生成模型；物理、碰撞、关节、IMU 与 ros2_control 插件。launch 默认仍为 imgo2.gazebo.urdf，real 需按现有运行说明选择 |
| `imgo2_deploy/src/imgo2_deploy/src/rl_sim.cpp` | ROS 入口与通信、启动关节控制器、读取反馈/发送命令、控制与推理循环、暂停/重置 |
| `imgo2_deploy/src/imgo2_deploy/include/rl_sim.hpp` | RL_Sim 类声明、ROS 消息缓存、线程与服务成员；`class RL_Sim : public RL` 表示共用算法工具由基类提供 |
| `imgo2_deploy/src/imgo2_deploy/library/core/rl_sdk/rl_sdk.cpp` | 观测拼接、读取配置/模型、动作缩放、键盘速度指令、姿态插值、把策略目标写入电机命令 |
| `imgo2_deploy/src/imgo2_deploy/fsm_robot/fsm_imgo2.hpp` | Imgo2 状态：Passive、GetUp、GetDown、PPO/HIM-Loco/AMP；每个状态定义 Enter/Run/Exit/CheckChange |
| `imgo2_deploy/src/imgo2_deploy/library/core/fsm/fsm.hpp` | 通用状态切换机制；不含机器人具体起身角度 |
| `imgo2_deploy/src/imgo2_deploy/library/core/inference_runtime/inference_runtime.cpp` | TorchScript/ONNX 加载及 forward；当前 policy.pt 经 libtorch 推理 |
| `imgo2_deploy/src/robot_joint_controller/ros2/src/robot_joint_controller_group.cpp` | Gazebo 侧关节控制器：接收目标，计算并限幅 PD 力矩，写 effort 接口，发布关节反馈 |
| `imgo2_deploy/policy/imgo2/base.yaml`、`amp/config.yaml` | 控制周期、关节顺序/映射、观察顺序与缩放、动作缩放、默认角度与 PD；amp 配置会覆盖同名基础参数 |

所以 Gazebo 的可执行入口为 rl_sim，而整体逻辑分布在这些模块。`rl_sim_mujoco.cpp` 是 MuJoCo 入口；`rl_real_imgo2.cpp` 是真机入口，它们也复用 RL 基类与 FSM。

## 3. 从启动到运行 AMP

1. `gazebo.launch.py` 启动 gzserver、可选 gzclient，加载世界并生成机器人。URDF 中的 gazebo_ros2_control 插件创建 controller_manager；IMU 插件发布 `/imu`。模型生成后启动 joint_state_broadcaster。
2. `rl_sim.cpp:811` 的 main 初始化 ROS2，构造 RL_Sim，然后 spin 接收消息回调。
3. `RL_Sim::RL_Sim()` 从 `/param_node` 读取 robot_name，读 base.yaml，创建 Imgo2 FSM（初始 Passive），启动 robot_joint_controller，建立发布/订阅与服务，发送 reset_world，再启动循环线程。
4. 按 `0`：FSM 进入 GetUp，以固定 PD 先弯腿再站起。此时主要是预设姿态插值。
5. 站起完成后按 `3`：进入 RLFSMStateAMPLocomotion，Enter 调用 `RL::InitRL("imgo2/amp")`，读取 amp/config.yaml、初始化观测/输出并加载 policy.pt。Run 开启推理并调用 RLControl；Exit 关闭推理标志。
6. 推理线程持续计算目标；控制线程持续发布命令；Gazebo 控制器和物理继续产生反馈。

Passive→GetUp→AMP 的具体按键条件在 fsm_imgo2.hpp；启动期翻倒、重置再掉落与不整齐趴姿详见 [初始化排查](gazebo_startup_posture_2026-10-12.md)，不在本篇重复展开。

## 4. 运行时的两条主循环

### 控制线程：配置周期 5 ms

`rl_sim.cpp:548` 的 RobotControl 大致按以下顺序执行：

```text
GetState                 从 IMU/关节消息缓存取得反馈，按 joint_mapping 排列
StateController          给 FSM 当前反馈，执行当前状态的 Run 和切换检查
                         同时处理速度指令按键
处理 R/Enter             世界重置或暂停/恢复
清理按键事件
SetCommand               按映射发布 12 个电机的 q/dq/kp/kd/tau
```

起身状态产生插值关节目标；AMP 状态的 RLControl 从队列取网络产生的目标。没有新目标时保留此前命令。因此不是每一拍都要跑网络。

### 推理线程：配置周期 20 ms

`rl_sim.cpp:697` 的 RunModel 只在 rl_init_done 且 simulation_running 时执行：

```text
机器人反馈 + 速度指令
→ Forward
  → RL::ComputeObservation（拼接/缩放/裁剪）
  → model->forward（当前 AMP 无历史堆叠）
  → 动作裁剪
→ RL::ComputeOutput（动作缩放 + 默认姿态）
→ 目标关节角/速度入队
→ 由控制线程的 RLControl 消费并发送
```

还有 50 ms 的键盘读取线程；ROS 的 spin 负责 IMU、关节与手柄等消息回调，不是上述两个控制线程。

### 时钟区别

| 环节 | 当前配置 | 时钟/说明 |
|---|---|---|
| Gazebo earth 物理 | 步长 0.0005 s，目标 2000 步/s | 仿真时间 |
| controller_manager/PD | update_rate=1000 | 仿真侧调度 |
| rl_sim 控制/命令 | dt=0.005 s，名义 200 Hz | LoopFunc 用 steady_clock 和线程等待，属于墙钟调度 |
| rl_sim 策略 | dt×decimation=0.02 s，名义 50 Hz | 墙钟调度 |
| IMU | 配置 100 Hz | 仿真传感器 |

这些是配置频率，不是本轮实测吞吐。Gazebo 实时倍率偏离 1 时，不能把墙钟 20 ms 直接当成仿真时间 20 ms；评测周期使用仿真时间，并留意调度负载。更详细的仿真器差异见 [已有对照](sim2sim_mujoco_vs_gazebo_2026-09-19.md)。

## 5. 策略输入、输出与真正的力矩

当前 AMP 输入 45 维，按配置顺序：

```text
机体角速度 3 + 机体坐标重力方向 3 + 速度指令 3
+ 相对默认姿态的关节角 12 + 关节速度 12 + 上次动作 12
= 45
```

角速度缩放 .25，关节速度缩放 .05，其余这里为 1；观测裁剪 ±100。当前 AMP 不读取基座真实线速度作为 actor 输入。关节反馈来自 `/robot_joint_controller/state`，IMU 来自 `/imu`；`/joint_states` 主要供 robot_state_publisher/查看，当前 rl_sim ROS2 路径没有订阅它。

网络输出 12 个动作，先裁剪 ±3，再逐关节换算：

```text
q_des[i] = default_dof_pos[i] + action_scale[i] * action[i]
```

hip 尺度 .125，thigh/shank .25；默认每腿 `(0, .87, -1.82)` rad。例如 thigh 动作 .4 → 目标 .87+.25×.4=.97 rad。q_des 是目标关节角，不是力矩，也不是预设步频。

RLControl 发送 q_des、dq_des=0、kp=25、kd=.5、前馈 tau=0。Gazebo 侧 RobotJointControllerGroup::UpdateFunc 再计算：

```text
tau = kp*(q_des-q) + kd*(dq_des-dq) + tau_ff
```

最终按 URDF 力矩限制裁剪后写入 effort 接口。当前该控制器的 dq 是位置差分估计。共用 ComputeOutput 虽然也算了 output_dof_tau，但当前 RLControl 没有把它作为前馈发出，所以这条链路不能解释成两次叠加 PD。

速度命令通常来自键盘/手柄；按 N（或手柄 X）打开 navigation_mode 后才选 `/cmd_vel`。单纯发布 `/cmd_vel` 而没有打开该模式，不能指望改变网络速度输入。

## 6. 怎样读代码与怎样整理

第一次只看 USE_ROS2 分支，阅读顺序：

1. `RobotControl()`：入口循环只有哪些步骤。
2. `fsm_imgo2.hpp`：按 0/3 分别进入什么状态。
3. `RunModel()` → `Forward()`：策略每一拍怎样产生动作。
4. `RL::ComputeObservation()` → `RL::ComputeOutput()` → `RLFSMState::RLControl()`：输入/输出怎样对应训练。
5. `RobotJointControllerGroup::UpdateFunc()`：目标如何变成力矩。
6. 最后读构造函数、StartJointController、消息回调与 launch。

源码难读的具体原因：同一文件兼容 ROS1/ROS2，构造函数含大段启动诊断，控制/推理/ROS 回调分属多个线程，关键函数继承自 RL，而机器人状态逻辑放在头文件；还保留其他策略、日志/绘图与未使用输出字段。

后续可以分别整理启动/ROS 接口、策略执行、Imgo2 状态与配置，先保留观察顺序、映射、频率和 PD 行为，再用同一策略/输入与固定速度回放对照。当前用户要求讲解，本轮不实施重构。

## 7. 已修复与待确认

- 已修复：根 README 的旧 AMP 表格从 48 维改为当前 45 维；纠正控制器 YAML 注释里“rl_sim 订阅 joint_states”的错误，运行参数未变；本篇补全跨文件调用关系。
- 待确认：DEPLOY-11 初始化时序、AMP-17 当前策略 Gazebo 效果，沿用已有记录。
- 待修复（DEPLOY-12）：全仓检索发现 output_dof_tau_queue 只有 push/声明，没有消费。当前 Gazebo AMP 不使用这份前馈力矩，但每次推理仍入队，持续运行会累积未消费数据；本轮不改代码，未测内存增长。后续需删除无用队列或明确消费用途，并验证目标/PD 没有改变及长时队列不再增长。
- 验证方式：逐项阅读当前函数、配置、构建定义和消息接口；没有新增运行测量，配置 Hz 不标作实测。
