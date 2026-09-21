# MuJoCo 与 Gazebo 的步态/抖动差异（2026-09-19）

关联问题：SIM-01（新增）。本文只记「用户反馈 + 参数层面对照」，**本轮没有跑任何仿真、没有改任何代码或模型**。
结论与状态同步进根 `README.md` §6.1、§7 问题表与 §8.3 维护记录。

## 1. 用户反馈

2026-09-19 用户：当前键 1 的 PPO（`policy/imgo2/ppo/`，即参考项目 `base_move/policy_flat.pt`，2026-06-30 12:45）
**「反而在 Gazebo 表现更好，在 MuJoCo 抖动更大」**。

注意这与 2026-09-17 那次方向相反的反馈不是同一对象：那次说的是 `imgo2_rough/2026-06-21_16-37-38`，
而且后来查明「Gazebo 完全不行」的一部分是用法问题（`/cmd_vel` 要 nav mode、槽位顺序等，见
[Gazebo 记录](gazebo_ros2_bringup_2026-09-17.md) 与 [JOINT-01](../README.md)）。

## 2. 三方配置对照（已逐条核对源码与模型，未跑仿真）

| | 训练（Isaac Lab） | MuJoCo（部署，键 1 走这条） | Gazebo（部署，键 1 也走这条） |
|---|---|---|---|
| 物理步长 | `0.005`（200 Hz）<br>`velocity_env_cfg.py:718` | `0.005`（200 Hz）<br>`imgo2.xml:9` | **`0.0005`（2000 Hz）**<br>`worlds/earth.world:6` |
| PD 计算频率 | 200 Hz（隐式，随物理步） | **200 Hz**（显式）<br>`rl_sim_mujoco.cpp:109` `loop_control = dt` | **1000 Hz**<br>`config/robot_control_ros2.yaml:23` `update_rate: 1000` |
| PD 的 D 项速度来源 | 真实 qvel | 真实 qvel（`jointvel` 传感器，`imgo2.xml` 传感器 12–23） | **位置一阶差分 / 1 ms**<br>`robot_joint_controller_group.cpp` `UpdateFunc`：`currentVel = (q - q_last)/period` |
| 关节阻尼 / armature | URDF 无 `<dynamics>` ⇒ 0 / 0；`friction=0.0`（`assets/imgo2.py:82-84`） | **`damping=1`、`armature=0.1`**<br>`imgo2.xml:34` 等 12 处 | URDF 无 `<dynamics>` ⇒ 0 / 0 |
| 接触 | 训练接触模型 | `condim=3`、**`solref="0.005 1"`**、`friction 1/0.01/0.01`（足端 `0.4/0.02/0.01`）<br>`imgo2.xml:28-54` | ODE 默认（URDF 无 `<material>`，用 ODE 默认 mu/kp/kd） |
| 关节力矩上限 | 23.7（`imgo2.py`） | `ctrlrange ±23.7`（`imgo2.xml:140-151`） | URDF `<limit effort="23.7">` |
| 策略频率 | 50 Hz（`decimation = 4`，`velocity_env_cfg.py:715`） | 50 Hz（`loop_rl = dt × decimation`，`rl_sim_mujoco.cpp:110`） | 50 Hz（同一份 `rl_sim.cpp` 代码） |

**Gazebo 那两个数（0.0005 s / 1000 Hz）的出处**：世界文件是我们自己写的**显式覆盖**——
上游 Gazebo 默认 `max_step_size` 是 `0.001`，而 `worlds/earth.world` 与 `worlds/stairs.world` 都写死
`<max_step_size>0.0005</max_step_size>` + `<real_time_update_rate>2000</real_time_update_rate>`
（参考项目的同名 world 也是这两个值，只有它的 `stair_1.world` 是 `0.001`）。launch 用
`gzserver -s libgazebo_ros_init.so -s libgazebo_ros_factory.so <share>/worlds/<wname>.world` 加载，
`wname` 默认 `earth`；`install/share/.../worlds/*.world` 是 symlink-install，指回 `src/`，所以二者一致。

运行期证据是我们自己的日志（不是推断）：

```
[gazebo_ros2_control]: Desired controller update period (0.001 s) is slower than the gazebo simulation period (0.0005 s).
```

（`imgo2_deploy/build/diag.log:131`、`e2e.log:133`、`codefix.log:133`）。这条 WARN 来自 Humble 的
`gazebo_ros2_control` 插件：它每个物理步都被调一次，但 `read()`/`update()`（PD 就在 `update()` 里算）
只在 `sim_period >= 1/update_rate` 时执行，而 `write()` 每个物理步都执行
（[插件源码 humble 分支](https://github.com/ros-controls/gazebo_ros2_control/blob/humble/gazebo_ros2_control/src/gazebo_ros2_control_plugin.cpp)
的 `GazeboRosControlPrivate::Update`；`control_period_ = 1/controller_manager.update_rate`，
我们的 yaml 是 1000）⇒ **PD 1000 Hz、力矩写入 2000 Hz、物理积分 2000 Hz**。

**关键的一处「抄了参数但没抄前提」**：`solref="0.005 1"` 的接触时间常数是从参考部署
（`~/RL/sim2sim/Imgo2_deploy/imgo2_description/mjcf/imgo2.xml:3,14`）抄来的，而**参考那份 MJCF 没有写 `timestep`**
（用 MuJoCo 默认 2 ms），所以它的「时间常数 / 步长」= 0.005/0.002 = **2.5**；
我们为了对齐训练把 timestep 改成 5 ms 之后，这个比值掉到 **1.0**（接触时间常数正好等于一个物理步）。
MuJoCo 的调参指南把「降低稳态接触抖动」的首选手段列为**减小 dt / 增加 substeps**
（见 [MuJoCo-Warp Contact Tuning](https://newton-physics.github.io/newton/stable/concepts/simulation_tuning_mujoco.html)
的 "Make Harder vs. Make Stable" 表），并没有「时间常数可以等于步长」的说法。

## 3. 候选机制（**都是假设，均未实测**）

按嫌疑排序：

1. **接触欠采样**（§2 末段）：MuJoCo 侧接触时间常数 = 1 个物理步、Gazebo 侧 ODE 步长 0.5 ms。
   若足端接触在高频上互相"抢"，表现就是腿在抖。
2. **PD 频率差 5 倍**：MuJoCo 200 Hz 显式 PD（与训练一致）对 Gazebo 1000 Hz。离散控制的等效阻尼
   随步长减小而增大，Gazebo 那条链路本身就相当于一个更强的低通；也就是说 **Gazebo 的"更稳"可能部分来自
   它离训练物理更远**。
3. **关节阻尼/armature 偏离训练**：MJCF 的 `damping=1 / armature=0.1` 是照参考部署抄的，
   而训练侧（URDF 无 `<dynamics>`、Isaac `friction=0.0`）是 0 / 0。这与 MODEL-01 的决定
   「物理参数一律以训练侧为准」其实不一致——只不过 MODEL-01 当时覆盖的是 link 质量/惯量/轴/限位，
   joint 级阻尼与 armature 不在其中。

## 4. 为什么现在还不能下结论

### 4.1 两边的数字不是同一个口径

| 来源 | 策略 | 指标 | 采样率 | 结果摘要 |
|---|---|---|---|---|
| Gazebo `imgo2_deploy/build/e_ref_flat_0.5.json` | **当前** `policy_flat.pt`（kp25/kd0.5） | `jitter.*.mean_abs_ddq` | `/joint_states` 10964 样本 / ~11 s ⇒ **~1 kHz** | thigh 45.3、shank 93.9、hip 23.8；z 0.288、0.428 m/s、周期强度 0.955、FL-FR 174.5° |
| MuJoCo [策略运行期排查](sim2sim_policy_runtime_2026-09-17.md) §3/§7.1 | **另一份**：`imgo2_flat/2026-06-21_23-24-09`（kp20/kd1.0） | `max\|dq\|`、`thigh_range` | 物理步 **200 Hz** | vx=0.5：z 0.128、0.43–0.51 m/s、`max\|dq\|` 11.9、thigh 极差 1.11 rad |

即：**策略不同、指标不同、采样率不同**。而且 `abs(Δdq)/dt` 这类指标对采样率极其敏感——
同一段物理运动以 1 kHz 采样得到的值天然高于 200 Hz，所以两边的数字**根本不能直接比大小**。

### 4.2 选型本身只在 Gazebo 做

PPO-01（2026-09-17）是在 **Gazebo** 上用四项指标从 5 个候选里挑的 `policy_flat.pt`。
因此「它 Gazebo 更好」有一部分是循环论证：它是被 Gazebo 指标选出来的。
换到 MuJoCo 用同样四项指标重排，名次很可能不同。

### 4.3 本机跑不了 MuJoCo 闭环

- `rl_sim_mujoco` 的 GUI 入口需要可用 OpenGL/显示（本机无 NVIDIA 驱动；没有 `Xvfb`/`xvfb-run`）。
- Python 侧：`python3` 有 numpy/yaml/rclpy 但**没有 mujoco**（五个 conda 环境也都没有）；
  `~/miniconda3/envs/isaaclab` 现在**没有 torch**（旧记录里的 torch 2.7.0 已不成立，与 AMP-03 的更正一致）；
  `pip download mujoco` 60 s 超时 ⇒ 网络不可用。
- 所以唯一可行路径是**仓库外临时 C++ harness**：复用真实 `RL`/FSM/`librl_sdk` + 部署自带
  libtorch 2.3.0（`imgo2_deploy/library/inference_runtime/libtorch`）+ libmujoco 3.2.7
  （`imgo2_deploy/library/mujoco` → 参考项目），只重写 `GetState`/`SetCommand`/`RunModelStep`/`Forward`。
  2026-09-17 用过同样做法，复现要点见[策略运行期排查](sim2sim_policy_runtime_2026-09-17.md) §5
  （必须把 `rl_sdk.cpp` 编进来并用 `-DPOLICY_DIR=<目录>`；临时策略目录要带 `base.yaml`）。

## 5. 要量化的话，需要做什么

1. 用上述 harness 跑**当前** `policy_flat.pt`（kp25/kd0.5）的 `vx=0` 与 `vx=0.5`，记录**逐拍 `q`/`dq`**（200 Hz，同时按策略周期与物理周期两种口径各算一次抖动），
   在 Gazebo 侧用同一段代码、同一采样率重算，得到**同口径**对比。
2. 单变量 A/B（每次只改一个）：
   - **timestep/solref**：`0.005 + solref 0.005`（现状）对 `0.002 + solref 0.005`（参考部署）对 `0.005 + solref 0.02`；
   - **PD 频率**：`loop_control = 0.005` 对 `0.001`（同时把物理步降到 0.001 或 0.0005，否则 `ctrl` 在步与步之间不变）；
   - **关节阻尼/armature**：`1 / 0.1`（现状）对 `0 / 0`（训练侧）。
3. 若确认是接触或 PD 频率，再决定改 MJCF（属模型改动，要同步 MOD-* 记录与 `check_model_sync.py` 之外的说明——
   该脚本只比对 URDF，不检查 MJCF）。

### 5.1 把 MuJoCo 的 PD 提到 1000 Hz：可行，但必须连物理步长一起降（2026-09-19 补充）

**结论：可以，改动只有两处，C++ 不用动。**

1. `imgo2_description/mjcf/imgo2.xml:9` 的 `<option timestep="0.005" …>` → `0.001`；
2. `imgo2_deploy/policy/imgo2/base.yaml`：`dt: 0.001`、`decimation: 20`（`dt × decimation` 必须仍是 `0.02`，
   即策略保持 50 Hz）。

不用改代码的原因：`loop_control` 周期就是 `dt`、`loop_rl` 是 `dt × decimation`
（`rl_sim_mujoco.cpp:109-110`）；`Interpolate()` 用 `ceil(duration / dt)` 算帧数，所以 GetUp 的 1 s / 2 s
墙钟时长不变（`rl_sdk.cpp:558`）；`dt` 在 `ComputeObservation()` 里只被 RoboMimic 的 phase 项用到
（`rl_sdk.cpp:162`，我们的任务不用）。

**但"只把 PD 调快"是无效的**：MuJoCo 的状态只在 `mj_step()` 之后变化，而写 `ctrl` 的
`RobotControl()`（每 `dt` 一次，持 `sim->mtx`）与物理线程（`main.cc:330-338`：睡 1 ms → 持同一把锁
→ 按实时同步步进）是串行化的。物理步仍是 5 ms 时，5 次 PD 重算读到的 `q/dq` 完全相同，算出的力矩也一样，
等于没提速。所以"1000 Hz PD"隐含"物理 ≤ 1 ms"。

**本机实测的模型/默认值**（临时 `mjprobe.c`，只链 libmujoco，未进仓库）：

```
mj_defaultOption : timestep=0.002 integrator=0(Euler) cone=0(pyramidal) solver=2(Newton) iterations=100 impratio=1
加载我们的 scene.xml: timestep=0.005 integrator=0(Euler) cone=1(elliptic) solver=2(Newton) iterations=100 impratio=100
                     dof damping[6..8]=1 1 1   armature=0.1 0.1 0.1   nsensordata=46
```

⇒ 我们现在是 **Euler 积分 + 外部显式 PD + `solref` 时间常数 = 1 个步长**，最不稳的三件事叠在一起；
顺带确认参考项目默认用的 2 ms 就是 MuJoCo 默认值。

**代价与坑**：

- 物理步 5→1 ms ⇒ 每秒 5 倍步数；模型很小、算力不是瓶颈，但控制线程要做 1000 次/秒并与物理线程抢
  `sim->mtx`，时序抖动会明显变大（现在是 200 次/秒）。
- `LoopFunc` 把每轮耗时按**整毫秒截断**（`library/core/loop/loop.hpp` 的 `duration_cast<milliseconds>`），
  所以 `dt=0.001` 实际跑不到 1000 Hz（大概率 700–900 Hz）。要真实 1 kHz 得先修这个循环
  （用 steady_clock 差值 + 绝对时间/自旋，而不是"跑完再睡剩余毫秒"）。
- GUI 版的物理线程是"睡 1 ms → 一步"的实时同步循环，1 kHz 基本贴着上限、容易掉帧；
  无头 harness 自己写步进循环则没有这个问题（也是量化时应采用的形态）。

**方向提醒**：训练侧（Isaac Lab）是 200 Hz PD / 5 ms 物理，Gazebo 是 1000 Hz PD / 0.5 ms 物理。
把 MuJoCo 提到 1000 Hz 是**向 Gazebo 靠、离训练更远**。若目标只是"更稳"而不是"更像 Gazebo"，
更对症的三个独立旋钮是：

- `<option integrator="implicitfast">`：让关节阻尼与 armature 隐式积分（不改频率，也不改 PD 语义）；
- 把 `<motor>` 换成内建 PD 执行器（如 `<position kp="25" kv="0.5">` / `<general>`），让 PD 在积分器内部
  隐式求解 —— 但这样 `SetCommand()` 要改成写目标角而不是力矩，且与训练侧的**显式** IdealPD 语义不同；
- 只治接触：`solref` 时间常数从 `0.005` 提到 `0.02`（4 × 步长）。

这三条与"1000 Hz PD"是互相独立的，A/B 必须一次只动一个。

## 6. 完整差异清单（Gazebo 路径 vs MuJoCo 路径）

§2 只列了物理与控制频率。下面是两条**部署路径**（`rl_sim.cpp` + ROS 2 控制器 vs `rl_sim_mujoco.cpp` +
MJCF）逐项对照，同样只读源码/模型，未跑仿真。

### 6.1 模型与几何：几何完全相同，属性不同

- **碰撞几何完全一致**：URDF（`imgo2.gazebo.urdf`）33 个 collision = box 9 / cylinder 20 / sphere 4；
  MJCF（`imgo2.xml`）33 个碰撞 geom，同样是 box 9 / cylinder 20 / sphere 4（另外 17 个是
  `contype="0" conaffinity="0"` 的纯视觉 mesh）。⇒ **几何不是差异来源**。
- **自碰撞**：Gazebo 12 条腿 link 全部 `<self_collide>1</self_collide>`（`gazebo.xacro:118-212`）；
  MJCF 全部 `contype="2" conaffinity="1"`，机器人内部两两不产生接触（自碰撞关）。
- **关节阻尼 / armature**：URDF 无 `<dynamics>` ⇒ 0 / 0；MJCF 每个关节 `damping="1" armature="0.1"`。
- **摩擦**：Gazebo 每 link `mu1/mu2`（足端 `0.6`、其它 `0.2`），没有 spin/rolling 分量；
  MJCF `friction`（足端 `0.4 0.02 0.01`、其它 `1 0.01 0.01`）。
- **接触刚度/阻尼**：Gazebo 部分 link `kp=1e6 / kd=1.0`，其余 ODE 默认，整体再由 world 的
  `erp 0.2 / cfm 0.0 / contact_surface_layer 0.001 / contact_max_correcting_vel 10.0` 决定；
  MJCF 统一 `condim="3" solref="0.005 1"`（时间常数 = 1 个物理步，见 §2）。
- **IMU 载体**：Gazebo 是独立 link `base_imu`（0.01 kg，原点与 `base` 重合，`disableFixedJointLumping=true`；
  `imu.xacro`）；MuJoCo 是 `base` 内原有的 `imu` site。

### 6.2 物理与积分

| | Gazebo | MuJoCo |
|---|---|---|
| 物理步长 | `0.0005`（2000 Hz） | `0.005`（200 Hz） |
| 求解器 | ODE `quick`、50 iters、sor 1.3 | MuJoCo 默认（Newton）+ `cone=elliptic`、`impratio=100` |
| 接触约束 | `erp 0.2`、`cfm 0.0`、`surface_layer 0.001`、`max_correcting_vel 10`、per-link `kp/kd` | `solref="0.005 1"`、`condim=3`、per-geom friction |
| 自碰撞 | **开**（12 link） | **关** |
| 关节阻尼 / armature | 0 / 0 | 1 / 0.1 |
| 足端摩擦 | `mu1=mu2=0.6` | `0.4 0.02 0.01` |

### 6.3 控制与执行链

| | Gazebo | MuJoCo |
|---|---|---|
| PD 计算位置/频率 | `RobotJointControllerGroup::update()` @ **1000 Hz** | `RL_Sim::SetCommand()` @ **200 Hz**（`loop_control`） |
| D 项速度来源 | **控制器自己算的 1 ms 位置差分**：`currentVel=(q−q_last)/period` | `jointvel` 传感器真值 |
| 下发通路 | 发布 `robot_msgs/RobotCommand` → `gazebo_ros2_control` effort 接口 | 直接写 `mj_data->ctrl[k]` |
| 力矩限幅 | 控制器 `EffortLimit` + URDF `effort="23.7"`（每 1 ms 一次） | `ctrlrange="±23.7"`（每 200 Hz 写一次） |
| 位置目标限位 | 控制器 `PositionLimit()` 用 URDF 关节限位 clamp `q_des` | **不 clamp `q_des`**，靠 MuJoCo 关节限位约束兜底 |
| 通信/时延 | DDS 发布订阅（命令 200 Hz、状态 1000 Hz）+ RealtimeBuffer | 进程内共享内存 + `sim->mtx` 互斥，无 DDS |
| 复位 / 暂停 | `/reset_world`、`/pause_physics` 服务 | `mj_resetData()` / `sim->run` |
| `ang_vel_axis` | ROS 2 = `body`（ROS 1 = `world`） | `body` |

### 6.4 观测与传感：同一条 45 维拼接，输入不同源

| 观测项 | Gazebo 来源 | MuJoCo 来源 |
|---|---|---|
| `ang_vel` | `/imu`（`libgazebo_ros_imu_sensor`，**100 Hz，带高斯噪声 stddev 2e-4 与零漂**） | `gyro` site 传感器，每个物理步（200 Hz，无噪声） |
| `gravity_vec` | 同一个 `/imu` 的四元数（无噪声配置，但 100 Hz 零阶保持） | `framequat` site 传感器（200 Hz） |
| `dof_pos` | ODE 关节位置（控制器 `state_interfaces_[2i]` → `~/state`） | `jointpos` 传感器 |
| `dof_vel` | **控制器 1 ms 位置差分**（`~/state` 的 `motor_state.dq`） | `jointvel` 传感器真值 |
| `tau_est` | `state_interfaces_[2i+1]` = effort | `jointactuatorfrc` 传感器 |
| `commands` / `actions` | 公共 `rl_sdk` 代码，两边相同 | 同 |

**另一个容易混淆的点**：Gazebo 的评测脚本 `eval_gazebo_policy.py` 读 `/odom` + `/joint_states`
（广播器给出的 ODE 速度），而**策略自己读的是 `robot_joint_controller/state`（控制器差分速度）**——
同一个 Gazebo 栈里存在两个速度信号，所以 §4.1 那些抖动数字既不是 MuJoCo 的口径，
也不是策略实际输入的口径。

### 6.5 小结：同一份策略为什么可能表现不同

按可疑度排序（全部**未做同口径实测**，方案见 §5）：

1. 接触时间常数 / 步长比：MuJoCo 1.0（欠采样）对 Gazebo 的 ODE 0.5 ms；
2. PD 频率 200 Hz 对 1000 Hz（且 Gazebo 的 D 项是位置差分）；
3. 关节阻尼 / armature 0/0 对 1/0.1；
4. 自碰撞开 / 关、足端摩擦 0.6 / 0.4、接触刚度模型（ODE kp·erp 对 MuJoCo solref·impratio）；
5. IMU 100 Hz 带噪声对 200 Hz 无噪声（策略的 `ang_vel`/`gravity_vec` 新鲜度差最多 10 ms）。

## 7. 状态

- **待量化**，已登记 README `SIM-01`。
- 本轮**未改**任何策略、配置、MJCF、URDF 或代码；用户选择「先只记录，不动代码」。
- 一个顺带发现（与 MODEL-01 的口径有关）：MJCF 的 joint `damping=1 / armature=0.1` 与接触参数来自
  参考部署，而训练侧是 0 / 0 —— 这两项**属于「以训练侧为准」这条决定尚未覆盖的角落**，值得单独决定。
