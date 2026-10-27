# PPO仿真与真机频率核对（2026-10-21）

> 主体是2026-10-21的源码/日志快照，历史IMU与MuJoCo参数不代表现在。2026-10-27的训练/部署周期、0.05s键盘轮询澄清见第5节；当前MuJoCo物理步长已为0.0005s，逐物理步PD与反馈组件更新见 [10月23日记录](mujoco_pd_feedback_2026-10-23.md)。

## 结论与范围

**不完全一致。PPO标称50Hz、上层主循环标称200Hz对齐；实际每电机外部命令与反馈约140–141Hz，IMU真机约200Hz、Gazebo约100Hz。** 另外，平均频率接近设定不代表周期稳定。当前日志没有测出电机固件内部PD环频率，也不能据此证明频率是落地颤抖的唯一根因。

本次只读控制源码、配置与已上传日志；新增离线统计工具及文档，没有修改部署源码、参数、关节宏、策略或驱动，没有启动真实设备/仿真。这里的“仿真”分别指当前Isaac PPO训练配置和ROS2 Gazebo部署，不能混为一条链路。旧权重`53a57f909b61…`的原始训练配置尚未找齐，当前训练源码与它原训练时一致仍是待确认条件。

## 1. 数据和复现

- 真机：`imgo2_deploy/real/build/log/2026-10-19_22-15-26_19277/`。`robot.csv`共67,597行，首末单调时间相差338.160518秒；`policy.csv`共14,485行，覆盖约289.710秒PPO阶段。events中48.445217秒进入PPO、338.187301秒Fault，前面包含InitialHold/Stand/StandHold。
- `trace.fault.csv`保存16,384次事务，trace自身时间328.472011–338.177138秒，覆盖约9.705秒。它不是完整338秒逐帧记录。`trace.exit.csv`位于Fault之后，不用于本次PPO通信频率结论。trace与robot日志起算点不同，不能直接按相同数字硬对齐。
- `2026-10-19_22-47-33_21507`是另一次短时InitialHold记录，不混入PPO统计。
- 独立历史Gazebo实录：`ppo_gazebo_ramp_2026-10-16_imu.csv.gz`，5,599帧、55.98仿真秒，验证该次运行的IMU为100Hz；不是本次新跑的仿真。
- 解释器：本机`python3`，仅标准库。算法：平均频率=`(样本数−1)/首末时间差`；周期分位数来自相邻时间差，不能用`1/周期中位数`代替平均频率。

在仓库根目录复现：

```bash
python3 imgo2_rl/scripts/tools/audit_deployment_frequencies.py \
  --real-log imgo2_deploy/real/build/log/2026-10-19_22-15-26_19277 \
  --output /tmp/deployment_frequency_audit.json
```

归档[机器可读结果](deployment_frequency_audit_2026-10-21.json)含原始五份日志及当前关键源码SHA256、各关节/总线明细。原始build日志是本地运行产物，跨机复现需要同时拷贝该日志目录，只有JSON不能重建逐行证据。

## 2. 各环节频率对照

| 环节 | 当前Isaac PPO训练配置 | ROS2 Gazebo部署配置 | 真机配置与此次实测 |
|---|---|---|---|
| 物理步进 | dt=0.005s，200Hz | world步长0.0005s，目标2000Hz | 物理机器人，不适用 |
| 策略 | decimation=4，50Hz | base.yaml dt=.005×4，50Hz线程 | policy_ms=20，50Hz；取得新结果平均49.995Hz |
| 上层控制/状态读取 | 策略每20ms取观测，动作在物理子步维持 | C++ RobotControl标称200Hz | control_ms=5；日志主循环199.893Hz，PPO段199.875Hz |
| 执行器PD计算 | 仿真执行器随物理子步200Hz | controller_manager标称1000Hz，控制器每次update计算PD | 运行在GO电机内部，实际频率未知；不能拿USB请求频率代替 |
| 每电机外部目标刷新 | 无串口；策略目标通常保持4个物理子步 | C++目标消息标称200Hz，策略目标50Hz改变 | 总线period=7ms，即标称142.857Hz；末尾trace请求140.37–141.06Hz |
| 关节有效反馈 | 仿真状态随物理步更新 | 控制器每次update尝试发布state，标称上限1000Hz；C++200Hz取最新快照 | 末尾trace每关节成功事务完成140.31–141.06Hz |
| IMU/机身姿态 | 直接读取仿真机身姿态/角速度；无独立串口IMU，policy以50Hz消费 | IMU插件100Hz；历史实录100.000Hz（仿真时间） | 串口解析sequence增量约200Hz，policy仍以约50Hz消费 |

Gazebo物理/控制器/IMU频率是每仿真秒口径；C++LoopFunc使用主机steady_clock，是真实时间口径。只有实时因子接近1时两者才能直接按Hz比较。controller state采用trylock发布，实际消息到达率可能更低，本次没有新测ROS关节消息频率。MuJoCo当前模型步长0.005s，C++控制/策略也使用同一base.yaml的200/50Hz，IMU读sensordata，没有单独100Hz IMU插件；本次没有运行MuJoCo，不宣称测得其实际频率。

### 源码依据

- [训练基础配置](../../imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py)：`decimation=4`、`sim.dt=.005`；`base_move/flat_env_cfg.py`继承rough再继承此配置，没有另改这两个值。不是所有AMP/拖曳任务都使用同一decimation，本报告只对当前基础PPO链路下结论。
- [仿真base.yaml](../../imgo2_deploy/ros2_ws/policy/imgo2/base.yaml)、[rl_sim.cpp](../../imgo2_deploy/ros2_ws/src/imgo2_deploy/src/rl_sim.cpp)：分别以dt、dt×decimation建立控制与策略线程。
- [ROS2控制器配置](../../imgo2_description/config/robot_control_ros2.yaml)为1000；`robot_joint_controller_group.cpp::UpdateFunc()`计算PD和尝试发布状态；`earth.world`/`stairs.world`步长0.0005。
- [Gazebo模块](../../imgo2_description/xacro/gazebo.xacro)中IMU sensor update_rate=100；生成URDF也声明100。
- [真机配置](../../imgo2_deploy/real/config/real_config.h)总线period=7ms；[部署YAML](../../imgo2_deploy/real/config/deployment.yaml)主循环5ms、策略20ms，上传meta记录一致。
- `rl_real_imgo2.cpp`主循环提交目标与观测；`sdk/inference_worker.cpp`独立20ms推理；`Motor/src/go_loop.cpp`按总线period调度，超期不补积压；`GoBus::cycle()`同一总线依次请求3台，四条总线线程并行。
- `Imu/src/imu_monitor.cpp`持续读取，以每约1秒的sequence增量计算receive_hz；read_timeout_ms=100是等待数据上限，**不是10Hz读取周期**；460800/4Mbps均是波特率，也不是反馈Hz。

## 3. 日志实测与抖动

### 主循环和策略（338秒主体、PPO段）

| 项目 | 周期中位数 | P95 | P99 | 最大 |
|---|---:|---:|---:|---:|
| 主循环/robot日志 | 5.000ms | 5.422ms | 7.640ms | 18.749ms |
| 主循环取得新PPO结果 | 20.000ms | 25.002ms | 29.248ms | 40.006ms |

PPO结果在被取得时年龄中位2.30ms、P99 5.76ms、最大13.58ms。这是结果生成到主循环取用的时间，不是网络推理耗时。用记录时间减result_age近似还原完成时间，平均49.995Hz、周期P99 27.408ms、最大42.221ms；减法有记录时刻/精度误差，也无法证明没有中间结果被覆盖，因此不把它作为精确推理起点或每次推理计数。

日志没有记录Fault后每一个主循环，最后一次触发故障的长等待也不完整包含于robot相邻行间隔，不能用18.749ms声称“整个程序最长只停了18.749ms”。`policy.csv`还存在重算obs而非保存实际推理输入的问题，obs_age全零不能用来证明观测无延迟；详见既有DEPLOY-DIAG-01。

### 电机事务和反馈（末尾约9.7秒trace）

下表用各总线ID0的连续请求开始/成功事务完成计时；所有12关节明细在JSON。

| 总线 | ID0请求平均Hz | ID0成功完成平均Hz | 反馈周期中位 | 反馈周期P99 | 反馈周期最大 |
|---|---:|---:|---:|---:|---:|
| 0 / FL | 140.50 | 140.40 | 7.000ms | 11.572ms | 29.861ms |
| 1 / FR | 141.06 | 141.06 | 6.999ms | 10.100ms | 23.276ms |
| 2 / BL | 140.44 | 140.44 | 7.000ms | 11.221ms | 25.300ms |
| 3 / BR | 140.73 | 140.73 | 7.000ms | 10.641ms | 25.391ms |

全部关节成功完成间隔最大32.312ms。成功完成时间用事务开始+waited_us近似，包含解析/记账开销，不是示波器接收末沿。trace有已知并发发布隐患，本次分bus/id排序后的间隔均正且与7ms配置吻合，作为已有记录的统计线索，仍需安全导出的新trace复核长尾；不把此窗口外推成338秒全程的精确反馈率。

完整338秒robot快照中，四条总线一轮3台的通信body中位分别5.020/5.029/5.020/5.013ms，P99约9.944–10.056ms。它们是**被200Hz快照重复采样的耗时值分布**，不是每轮唯一计数。通常body约5ms后睡到7ms计划点，故不是5ms一次完整刷新，更不是每台电机600Hz。超期轮会拖长更新间隔。

完整运行的关节反馈年龄池化中位5ms、P99 12ms、最大34ms；同腿髋/大腿/小腿中位约7/5/3ms。原因是先收ID0再ID1再ID2、整条总线结束后才发布快照，前三台并非同一时刻采样。**年龄不是反馈周期**，不能从5ms年龄推出200Hz。上层200Hz提交是最新目标覆盖，较慢总线只取最新目标，并非每次上层提交都会发成串口包；同样，50Hz策略目标可能在多个总线轮次重复发送。

### IMU（完整338秒）

排除启动前2秒后，日志receive_hz中位200.0、范围199.1–201.0Hz，P99 200.9Hz；这些是约1秒更新一次并重复写入快照的窗口频率，不是67,197个独立频率实验。imu_age中位2ms、P99 7ms、最大23ms。

这证实串口解析接收约200帧/秒，不是根据波特率猜出的1000Hz。解析器sequence计数所有成功解码帧，一次read解析多帧时公开接口仅返回最新一帧；原始sequence、设备时间、各字段独立序号没有写入CSV，故不能由本日志精确证明每个四元数/gyro字段都200Hz或量化USB逐帧抖动/IMU内部滤波延迟。

真机200Hz IMU约每策略周期4帧，Gazebo100Hz约2帧；两者PPO最终仍只每20ms读取最新姿态。频率更高本身不构成坐标错误，也无需为了数值相同先降到100Hz。关键是采样年龄、滤波与角速度/姿态时间一致性。

## 4. 已完成与待确认

已完成：重新核对当前配置、日志meta与线程调度；用Python3跑完338秒主日志、PPO结果、12关节末尾事务及历史Gazebo IMU统计；确认主循环/策略平均接近200/50Hz、总线约140–141Hz、IMU约200Hz；修正文档中“5ms轮询=200Hz每电机反馈”的理解。周期算法已用规则/乱序时间样例核验；全量结果检查确认CSV无畸形行、12关节齐全、分组排序后无非正间隔，py_compile与git diff --check通过。只完成诊断，没有修复实际通信延迟或运动颤抖。

待确认/处理：

1. 获取当前53a57f权重对应原训练env/agent配置，才能确认其真实训练步长、decimation及执行器设置；当前仓库50Hz是对照依据，不是该旧权重原训练凭证。
2. 分解约1.65ms事务中的线侧/桥接/驱动延迟，见[既有通信排查](go_imu_timing_vibration_review_2026-10-20.md)。先解决吞吐和长尾，再决定总线是否可以稳定达到5ms。当前一轮P99约10ms，直接7ms改5ms不能保证200Hz，不改超时/锁定绕过故障。
3. 记录安全发布的完整逐事务时间、实际推理输入与时间戳、IMU sequence/设备时标，才能精确还原338秒每次采样到执行的延迟。现有trace仅最后窗口且存在并发限制。
4. 真机电机固件内部PD环及限力矩语义须厂商/实测确认；与训练200Hz或Gazebo1000Hz PD执行机制不同，不能只提高串口下发频率就声称一致。
5. 在相同权重/命令、明确落地/提起标签下做可对照日志。已有长尾和较慢外部更新可能加重接触时颤抖，但不是证明训练或IMU任一项正确/错误的充分证据。

无需更改策略20ms、主循环5ms或IMU输出率来“凑频率”。本轮没有授权自动改这些参数；优先补齐执行链路测量和权重原配置。

## 5. 2026-10-27：0.02s训练步长与部署0.005/0.05s的区别

本轮只读当前源码/配置和8400对应run保存的env.yaml，没有重新测量运行频率，没有改周期或启动硬件/仿真。

| 环节 | 当前配置周期 | 对应标称频率 | 依据 |
|---|---:|---:|---|
| Isaac训练物理子步 | 0.005s | 200Hz | 保存env.yaml sim.dt=.005；训练基础配置一致 |
| Isaac策略/环境step/奖励结算 | **0.02s** | **50Hz** | decimation=4，step_dt=sim.dt*4 |
| Gazebo/MuJoCo部署上层RobotControl | **0.005s** | **200Hz** | base.yaml dt=.005；LoopFunc使用dt |
| Gazebo/MuJoCo部署RunModel | **0.02s** | **50Hz** | LoopFunc使用dt*decimation=.005*4 |
| Gazebo/MuJoCo部署键盘轮询 | **0.05s** | **20Hz** | rl_sim.cpp/rl_sim_mujoco.cpp的loop_keyboard |
| 当前独立真机主循环 | 5ms | 200Hz | real/config/deployment.yaml control_ms=5 |
| 当前独立真机策略线程 | 20ms | 50Hz | policy_ms=20；rl_real_imgo2传给InferenceWorker |

所以，当前配置中没有“电机上层控制每0.05秒一次”的对应设置；用户可能记的是0.005s，或看到的是键盘线程。0.05s是50ms、20Hz，0.005s是5ms、200Hz，二者差10倍。

训练每20ms生成一拍动作，期间仿真推进4个5ms物理子步；部署策略每20ms生成一拍目标，5ms上层控制环可在两次策略更新之间重复下发/读取。这两层可以有不同频率。上层RobotControl的200Hz不等于电机内部PD频率，也不保证总线或反馈真的达到200Hz；此处只说明当前配置，不能代替运行实测。当前MuJoCo物理.5ms、生产组件随物理步计算PD，另见顶部链接。

训练与部署策略周期应相互匹配；此处二者都标称20ms。若实际把部署策略改成50ms，则变成20Hz，执行时间尺度和反馈间隔发生变化，不能视为当前50Hz训练策略的等价部署。步态周期约.3–.5s又是另一种物理运动周期，不能拿它替代策略或电机刷新周期。

**已完成**：保存训练参数、仿真base.yaml、两个仿真线程构造、独立真机YAML与调用参数的交叉核对。**待确认**：当前生产程序在实际负载下的时序/总线/固件PD频率仍需带时间戳运行记录；没有因配置对齐就关闭原有时序与步态问题。
