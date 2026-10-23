# MuJoCo代码与PPO使用说明（2026-10-23）

> 10-23更新：SetCommand现只发布目标/增益，电机PD改为物理步回调；GetState可读取独立延迟/加噪反馈。延迟在base.yaml、IMU噪声在MJCF的sensor段修改，详见[最新实现说明](../deployment/mujoco_pd_feedback_2026-10-23.md)。下文旧的时序描述作为学习背景，当前以该说明为准。

## 先使用PPO

本程序是ros2_ws中的参考部署入口，不是real中的真机FSM。终端按键逐字符读取，状态切换也无需回车；Enter会暂停/继续仿真，不能照搬真机“数字+回车”。

```bash
cd /home/tommy0929/Documents/imgo2/imgo2_deploy/ros2_ws
./cmake_build_mujoco/bin/rl_sim_mujoco imgo2 scene
```

保持程序前台运行，点击启动它的终端（不要把输入交给MuJoCo图形窗口）。正常默认运行；先按0，等Getting up到100%、机器人站稳后再按1。参考GetUp从Passive通常先1秒预姿态、再2秒站起，若差值小会跳过部分插值；不是real中的8秒曲线，也不是严格墙钟3秒。过早按1不会排队，完成后重新按1。进入后终端应出现`RL Controller [ppo] x:0 y:0 yaw:0`（格式可能带颜色）。

先保持零指令观察，再逐次按W给+.1、+.2，Space清零。指令持续保持，松手不会归零；长按可能由键盘自动重复累加，当前参考接口没有自动限制到实机的速度上限。先一次一个按键，观察显示值。

| 终端键 | 当前含义 |
|---|---|
| 0 | GetUp站起/回站立；等待完成后才能选择策略 |
| 1 | PPO，自动加载policy/imgo2/ppo/config.yaml和其model_name指定权重 |
| 2 | HIM-Loco旧参考入口，非当前real的AMP占位，不建议本轮使用 |
| 3 | AMP旧参考入口，契约差异尚待修，暂不作为已验证策略 |
| W / S | 前向速度指令每次+.1 / -.1 |
| A / D | 横向速度指令每次+.1 / -.1 |
| Q / E | 转向角速度指令每次+.1 / -.1 |
| Space | x/y/yaw指令全部清零，仍处于PPO |
| 9 | GetDown约2秒插值回本次GetUp记录的start_state，完成进Passive；不是real的自定义趴姿后卸力 |
| P | 直接Passive，Kp0/Kd8，仍有速度阻尼，不是完全失能 |
| Enter | 暂停/继续仿真；不作状态命令的提交键 |
| R | 重置MuJoCo物理数据，不等于重启程序/FSM和清理策略全部历史 |

如果按1没有变化，先检查终端焦点、是否完成GetUp、是否误按Enter暂停。若出现InitRL() failed/Failed to load model，保留完整报错检查config路径与TorchScript格式；不要用训练checkpoint直接替代导出的policy.pt。手柄缺失提示本身不会退出，仍可键盘使用。GUI内独立暂停可能与程序simulation_running标志不一致，实验时优先只用终端Enter并确认实际在推进；严谨时序实验仍需修复/统一调度。

## 文件职责与推荐阅读顺序

以下均相对仓库根：

1. `imgo2_deploy/ros2_ws/src/imgo2_deploy/src/rl_sim_mujoco.cpp`：main、初始化、GetState、RobotControl、RunModel、SetCommand和Forward。先看构造函数，再看后三个循环函数。
2. `imgo2_deploy/ros2_ws/src/imgo2_deploy/fsm_robot/fsm_imgo2.hpp`：Passive/GetUp/PPO/GetDown等具体状态，Enter/Run/Exit/CheckChange。
3. `imgo2_deploy/ros2_ws/src/imgo2_deploy/library/core/rl_sdk/rl_sdk.cpp`：StateController、InitRL、ComputeObservation、ComputeOutput、RLControl、KeyboardInterface和插值。
4. `imgo2_deploy/ros2_ws/policy/imgo2/base.yaml`与`ppo/config.yaml`：默认控制周期/映射/固定姿态增益与PPO观测/动作/增益配置。
5. `imgo2_description/mjcf/scene.xml`和`imgo2.xml`：世界与机器人动力学。`library/thirdparty/mujoco_simulate/`是窗口/物理线程基础设施，初学先理解调用关系即可。

## 从启动到循环

main注册SIGINT并构造RL_Sim。构造函数读取命令行imgo2/scene→创建MuJoCo窗口与物理线程→加载模型并等待mjData→读base.yaml→创建参考FSM→初始化12关节状态/命令→启动控制、推理、键盘和手柄循环→主线程进入RenderLoop。启动到窗口存在并不意味着已经加载PPO；进入PPO状态才加载策略。

```text
键盘0 → GetUp插值 → 完成后键盘1 → PPO.Enter → InitRL
                                           ↓
                              读取ppo/config.yaml并加载policy.pt
                                           ↓
物理状态 → GetState → 观测构造 → 神经网络 → 动作裁剪/缩放/默认角
   ↑                                                  ↓
 mj_step ← ctrl力矩 ← SetCommand的PD ← FSM命令 ← q/dq队列
```

base.yaml先提供通用配置，PPO.Enter的InitRL再读ppo/config.yaml覆盖同名参数，并初始化观测/历史/输出/指令，最后ModelFactory加载权重。进入PPO时速度指令会被InitControl清零，不需自己输入模型路径；默认路径由编译时POLICY_DIR定位ros2_ws/policy。

## 几个循环分别做什么

| 循环 | 配置/名义频率 | 职责 |
|---|---|---|
| 物理线程 | MJCF timestep=.005，物理步长5ms | mj_step推进动力学并更新传感器；GUI可改变推进速度 |
| RobotControl | base.dt=.005，约200Hz | 锁sim互斥量，GetState→StateController运行FSM→处理暂停/重置→SetCommand |
| RunModel | dt*decimation=.02，约50Hz | PPO有效且仿真运行时，构造观测→Forward→ComputeOutput→向目标队列发布 |
| KeyboardInterface | .05秒，约20Hz轮询 | 非阻塞读取一个字符并交给控制循环 |
| 手柄轮询 | .01秒，约100Hz | 可选输入；没有手柄仍可运行 |

上述频率只是源码设置，并非本轮测出的实际频率。物理和控制分线程，墙钟/仿真时间没有严格逐步绑定，已有LoopFunc调度误差也未修。

## 策略收到什么，输出如何控制关节

PPO观测按顺序为：gyro*0.25（3）、四元数逆旋转世界重力[0,0,-1]得到机身重力方向（3）、速度指令（3）、q-default_q（12）、dq*.05（12）、上一轮裁剪后动作（12），合计45维；再裁剪观测幅度。四元数本身不直接占4维输入；加速度当前没有进入该PPO。

policy输出12个无量纲动作a，Forward裁剪到[-3,3]。ComputeOutput逐关节转换：

```text
q_des = default_q + action_scale * clamp(a, -3, 3)
default_q每腿 = [0, .87, -1.82] rad
action_scale每腿 = [.125, .25, .25]
dq_des = 0（当前没有轮关节）
```

RunModel把q/dq放入队列。PPO.Run调用RLControl，由控制循环取出目标并设置Kp25/Kd.5、tau_ff=0。最后SetCommand按当前MuJoCo反馈计算：

```text
tau = tau_ff + Kp*(q_des-q) + Kd*(dq_des-dq)
```

把tau写入mjData.ctrl，motor执行器在物理步施力。不是把策略输出直接当力矩，也不是把q_des直接覆盖到机器人姿态。ComputeOutput另算的output_dof_tau当前不是通过RLControl送入电机的前馈，不能误认PD算了两遍叠加。最终MJCF motor的ctrlrange为±23.7；YAML诊断估计力矩限制±23.5不是该路径最终力矩限制的唯一来源。

## 本次确认范围与待处理项

用户报告程序已能启动；现场只读确认cmake_build_mujoco/bin/rl_sim_mujoco存在，ldd正确找到libmujoco.so.3.2.7、libglfw.so.3和工程LibTorch，未见not found。本轮没有替用户运行GUI、按键、PPO步态或硬件，故“进入策略并运动”仍待实际操作确认；此前“依赖缺失”是安装前历史，当前已找到所需动态库。

模型8body差异、额外被动参数、PD/物理时序、并发快照/队列和线程退出问题仍在，详见[复审](mujoco_sim2sim_review_2026-10-23.md)。本轮新增文档而未改源码/参数。新cmake_build_mujoco目录当前显示为未跟踪，构建产物忽略规则待处理，勿将其作为源码提交；本轮不擅自改gitignore。
