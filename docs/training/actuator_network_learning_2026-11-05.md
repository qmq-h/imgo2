# Imgo2 执行器网络：学习与使用路线

核对日期：2026-11-05。范围：源码、已有诊断记录与 Isaac Lab 2.3.2 本地实现核对；未训练网络、未运行仿真或硬件，未更改控制或训练配置。用户报告 PPO 已能实机运动但仍抖动；本轮未获得新的同步实机数据，不能确认根因。

## 1. 先理解它解决什么

PPO 输出关节位置目标，执行器决定目标如何转成实际力矩。简单模型近似为 `tau = kp*(q_des-q) + kd*(dq_des-dq) + tau_ff`，还要考虑力矩/转速包络。真实系统有反馈与命令延迟、速度估计滤波、摩擦、减速器柔性和饱和等差异。

本工程现有执行器 MLP 是一个监督学习的正向模型：

`[e(t), e(t-1), e(t-2), dq(t), dq(t-1), dq(t-2)] -> tau_est(t)`，其中 `e=q_des-q`。

它不输出 PPO 动作，不是逆模型，也不是直接加在真机上的消抖控制器。常规使用链：真实数据辨识 → 固定执行器网络 → 在仿真中替换关节执行模型 → 训练/微调 PPO → 正式 play.py 导出 PPO → sim2sim 与实机验收。网络通常留在训练仿真中，真机仍使用其原有电机控制器。原 PPO 不会因为文件旁边多了执行器权重而自动改善。

## 2. 工程现状与阅读顺序

| 先后 | 入口 | 看什么 |
|---|---|---|
| 1 | `imgo2_rl/source/imgo2_rl/imgo2_rl/assets/imgo2.py` | 当前 legs 为 DCMotorCfg，P25/D0.5，effort/saturation 23.7Nm，velocity 30.1rad/s；不是神经执行器，实际增益可能受环境随机化影响 |
| 2 | `imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py` | sim.dt=.005，decimation=4，即物理200Hz、策略50Hz；动作缩放、默认姿态、随机化也需一起看 |
| 3 | `imgo2_deploy/ros2_ws/src/imgo2_deploy/scripts/actuator_net.py` | Config → load_data → process_data → train_actuator_network → main |
| 4 | `imgo2_deploy/real/rl_real_imgo2.cpp` | 独立实机入口的日志、Safety后实际下发目标与反馈；不同于ROS SDK2入口 |
| 5 | `imgo2_deploy/ros2_ws/src/imgo2_deploy/library/core/rl_sdk/rl_sdk.cpp` | 旧CSVInit/CSVLogger列格式，不证明独立实机日志可直接用于训练 |

现有训练脚本：共享单关节 6→32→32→1 MLP，Softsign，Adam，lr=8e-4，200 epochs，batch128，MSE损失并打印MAE。12个关节样本混在一起训练一张网络；这隐含所有关节动力学可用同一映射表达，需要数据验证。不是完整的采集、辨识、接入、验收流水线。全工程Python检索未发现 Imgo2 的 ActuatorNetMLPCfg/ActuatorNetLSTMCfg 接入。

## 3. 抖动先怎么查

已有 [运动振荡路线](../deployment/sim2real_vibration_literature_plan_2026-10-21.md)、[线程时序复审](../deployment/thread_obs_timing_review_2026-10-23.md) 已指出执行/反馈时序、旧观测、接触与IMU等候选；历史7–8Hz现象不是本轮新测量，也不证明当前机器仍是同一机制。

先固定同一PPO权重、指令、PD与地面条件，保存真实输入、策略目标、Safety后实际目标、q/dq/反馈力矩、IMU、kp/kd、目标与反馈时间戳及有效性。比较：目标本身是否已高频变化；目标平滑时实际q/dq是否振荡；是否集中在触地、饱和或反馈停顿；仿真是否存在同频现象。闭环中仅凭“哪个信号先抖”不能唯一归因。

建议先比较受控动态响应与简单PD/延迟模型，再决定网络是否有必要。静站正常只说明静态条件可工作，不能排除动态执行器差异。网络也无法自动修复总线丢帧、线程同步、错误映射或IMU时延。

## 4. 学习路线及每步产物

1. **PD与系统辨识基础**：能解释位置误差、阻尼、力矩饱和、幅频/相频响应，以及50Hz策略与200Hz物理步的差别。产物：画出 PPO→目标→电机→反馈闭环，列出各时标。
2. **读当前脚本并做离线小练习**：用人工生成的 `tau=25e-.5dq` 学习监督回归，理解输入拼接、MSE、训练/验证集和TorchScript。只能验证训练链路，不能验证真实执行器。
3. **建立可用数据集**：先复用已有真实日志；后续受控台架采集应从支撑、小幅平滑目标和既有限位开始，记录覆盖的速度/负载/方向，不直接无界扫频或重放行走动作。产物：有时间戳、单位、坐标、采样有效性、增益/固件/电压等元数据的数据集。
4. **先建立简单基线，再训练MLP**：比较名义PD、拟合PD、PD+延迟/摩擦与MLP。按整段独立采集划分训练/验证/测试；先分段再造历史窗口，不让相邻窗口或同一时刻其它关节泄漏到测试。产物：逐关节/方向/速度/负载误差、波形和相位对比。
5. **接入Isaac并验证**：对固定输入离线核对特征、输出与限幅，然后单关节/站立/运动闭环。冻结执行器网络，用独立PPO实验目录短训；是否续训由观测、算法checkpoint和优化状态兼容性决定，不能默认直接resume安全。产物：与原DCMotor相同条件的对照。
6. **PPO重新验收与部署**：按对应算法正式play.py导出，遵守Actor数值一致、部署LibTorch加载和观测契约；验收角速度/姿态波动、目标变化、滑移、跟速、力矩与饱和率。抖动下降同时要检查跟速和步态是否退化。

## 5. 数据与现有脚本怎么用

脚本要求每关节以下列（i从0连续编号；12关节即0..11）：

| 列名 | 单位/含义 |
|---|---|
| joint_pos_i | rad，反馈角度 |
| joint_pos_target_i | rad，与该反馈时刻匹配的实际下发位置目标 |
| joint_vel_i | rad/s，反馈速度 |
| tau_est_i | Nm，已核实语义的输出侧力矩反馈/估计，监督标签 |
| tau_cal_i | Nm，基线计算力矩；脚本强制读入，但仅用于画图 |

严禁拿命令力矩或同一PD公式计算值冒充真实力矩标签；否则只会学回原公式。电流估算力矩不等于独立测得的输出轴力矩：须确认减速比、单位、符号、滤波/延迟和标定。若kp/kd、dq_des或tau_ff变化，当前6维输入不足以唯一描述结果：固定采集条件或扩展模型并同步接入。关节异质性明显时考虑分组网络，不直接用Go1/Go2预训练模型代替Imgo2辨识。

独立实机 robot.csv/policy.csv 不能简单改列名后就使用：需对齐Safety后目标与真实新反馈、硬件→逻辑关节顺序，去重重复旧反馈，处理异步时间与断点；跨段历史必须断开。重采样不能制造新的独立反馈。此适配尚未实现。

以下为**准备好数据及依赖后**的入口示例，文件名为待生成占位，不是已有数据或模型。本轮未执行训练。脚本依赖torch/pandas/numpy/matplotlib，默认训练设备cuda:0且会plt.show()；CPU训练需先改Config.device，无头运行建议MPLBACKEND=Agg。输出父目录必须预先存在。

```bash
# 在仓库根；确认使用安装了上述依赖的Python环境
# 先将合规数据准备为 imgo2_deploy/ros2_ws/policy/imgo2/actuator_train.csv
python3 imgo2_deploy/ros2_ws/src/imgo2_deploy/scripts/actuator_net.py \
  --mode train --data imgo2/actuator_train.csv --output imgo2/actuator_mlp.pt
python3 imgo2_deploy/ros2_ws/src/imgo2_deploy/scripts/actuator_net.py \
  --mode play --data imgo2/actuator_test.csv --output imgo2/actuator_mlp.pt
```

这里play仅计算预测并画图，不运行机器人。相对data/output由脚本解释为ros2_ws/policy下的路径；不要用PPO的policy.pt作为输出。执行器权重与部署策略权重用途不同。

现有脚本限制（本轮未修改）：

- random_split按样本随机80/20拆分，重叠时间窗口导致验证偏乐观；无固定seed，不适合直接作为辨识验收。
- Config.dt=.02仅用于画图，不控制采集、重采样或特征时间跨度；process_data只按CSV相邻行取历史。
- 所有关节共用网络，无kp/kd、电压、温度等条件输入；无时间戳/掉帧/单位检查。
- 每epoch覆盖同一模型文件，保存最后一轮而非最佳验证模型；未保存切分、训练配置和数据哈希。
- 推理图只展示前1000点且训练后默认用同一数据画图；这不能替代独立测试。输出图固定12轴。

## 6. 接入Isaac最关键的时间契约

本机 `Documents/IsaacLab-2.3.2/source/isaaclab/isaaclab/actuators/actuator_net{,_cfg}.py` 已核对：ActuatorNetMLP按compute调用维护历史，input_idx=0是当前，正数是此前若干次调用；pos_vel将全部位置误差放前、速度放后。它输出力矩并执行DCMotor包络限幅，stiffness/damping不再直接作为PD使用，reset清历史。

若模型按等间隔5ms数据的相邻三帧训练，200Hz执行器调用对应 `input_idx=[0,1,2]`；若按20ms数据训练，200Hz调用则对应 `[0,4,8]`，但还需验证在中间5ms时刻使用是否合理。不能通过修改Config.dt改变已训练特征的时间语义。真实反馈不规则时更不能直接以行号当固定毫秒数。

后续接入候选配置需要：`network_file`由项目路径推导，`input_order='pos_vel'`，原始rad/rad/s/Nm直接训练时三种scale=1；input_idx依采集/调用周期选择。保留经核实的力矩/速度限制，并检查网络输出是否已含限幅、随机增益事件是否仍兼容；避免重复施加同一延迟。此为接入说明，不是已接好的可运行配置。

## 7. 推荐资料

- [Hwangbo等，Learning agile and dynamic motor skills for legged robots](https://arxiv.org/abs/1901.08652)：重点理解为何辨识执行器及为何需要历史，ANYmal具体参数不能直接移植。
- [Isaac Lab执行器API](https://isaac-sim.github.io/IsaacLab/v2.1.1/source/api/lab/isaaclab.actuators.html)：查ActuatorNetMLP、ActuatorNetLSTM、DCMotor；网页为2.1.1，工程接入以安装的2.3.2源码为准。
- [Tan等，Sim-to-Real: Learning Agile Locomotion For Quadruped Robots](https://www.roboticsproceedings.org/rss14/p10.pdf)：区分执行器建模与延迟建模。

## 8. 已修复与待修复/待确认

已修复：本轮仅补齐学习文档，明确现有脚本与当前PPO未接线；无生产代码修复，未证明抖动改善。

待修复/待确认（ACT-NET-01）：实际力矩反馈可信度与时间戳语义；独立实机日志适配；按采集段拆分/模型选择/可复现元数据；采样历史与Isaac时标；固定增益或条件输入方案；训练及仿真接线和闭环效果。完成需要真实同步数据、增益/固件信息及可用PyTorch/Isaac环境。

验证范围：/usr/bin/python3 AST检查确认MLP配置与特征顺序，脚本语法解析通过；本机该解释器无torch/pandas，未运行导入/训练/CLI。README与本记录链接检查通过。git pull --ff-only因.git/FETCH_HEAD只读失败；保留已有工作区修改，仅新增本记录及README入口，不覆盖既有未提交内容。

## 9. 补充：如何采集，是否必须训练执行器网络

同日用户追问后核对：Isaac Lab官方当前Unitree资产同时提供Go1执行器MLP与A1/Go2 DCMotor；Unitree官方unitree_rl_lab的UnitreeActuator继承DelayedPDActuator并实现摩擦/力矩速度曲线；legged_gym ANYmal支持执行器网络。由这些代表性工程只能得出两条路线并存，不能据此统计哪条占多数，更不能称神经执行器是sim2real必需项。

来源：[Isaac Lab Unitree配置](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_assets/isaaclab_assets/robots/unitree.py)、[Unitree官方执行器](https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree_actuators.py)、[ANYmal实现](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/anymal_c/anymal.py)。在线main会变化，非本地安装版本契约。

### 当前可执行的日志入口

独立实机rl_real_imgo2已有 `--log --log-every 1 --log-dir <目录>`。在硬件机原本可用的启动命令后追加这些参数即可；沿用该机原有policy-dir、config等参数，不另换启动目录/可执行文件。默认记录robot.csv约200Hz、policy.csv约50Hz、events.log、meta.json以及trace；本轮未运行硬件。每段单独运行目录，保持policy/config副本或哈希、地面/载荷、支撑条件、方向、温度、电压（若设备可读）、起止与抖动标记。

robot.csv已含 `t_mono_ms`、`q_des_00..11`、`dq_des_*`、`tau_ff_*`、`kp_*`、`kd_*`、`q_*`、`dq_*`、`tau_*`、`temp_*`、`age_*`、valid_mask/err_mask/clamped与IMU。注意：源码在sendCommand后从最新目标重算Safety夹值写日志，记录的是控制线程提交意图，不保证该目标在同一时刻已被总线发送/电机应用；反馈raw又取自该次发送之前的快照。因此同一行的目标与反馈不是天然同步训练样本。正式辨识仍应补每关节新反馈序号/精确接收时间及实际发送目标序号/时间，确认固件tau的估算和滤波语义。age只有毫秒精度，适合初步分析，不能替代精确时标。已有trace并发快照可靠性问题见线程复审，不能直接当高精度真值。

### 两阶段采集方案（建议，未执行）

A. 首先复用已能稳定运行的控制流程，记录无新增激励的静站、零指令PPO、已验证低速前进/后退/横移/转向，各自独立短段并包含开始/停止。首次可每类20–30秒、重复几次，属于初筛计划，不是训练所需固定总时长；出现异常即停止，不为凑数据延长明显抖动。不要故意提高指令追求更强振荡。用途是诊断、确定工作范围，以及识别普通行走数据的覆盖缺口。

B. 确需执行器辨识时，再开发独立受限轨迹采集模式。支撑下选择中间姿态，单关节平滑小幅正弦/多段轨迹，从低频开始，固定kp/kd、dq_des和tau_ff定义，记录目标速度与加速度上界。角度/频率由行程、惯量、力矩、机械碰撞和已有安全界限决定，不给未经硬件核验的统一数值。小幅多频轨迹满足 `v_peak=2*pi*f*A`、`a_peak=(2*pi*f)^2*A`；提高频率会快速提高加速度。之后逐步覆盖已验收的受载条件、正反方向与关节类型。空载数据不代表落地承重动力学；只用PPO轨迹又可能缺少独立激励。现有PPO启动参数没有通用正弦辨识模式，不应声称加--log便实现了该功能。

### 入训前处理与验收

保持原始数据不覆盖；按关节有效性、错误、温度、时序、目标序号清洗。故障/长停顿/切模式处断开窗口；正常饱和要标记并单独评估，不能无区别全删导致缺失关键区域。先确认轴向/减速比与单位，再对齐目标作用时刻和反馈时刻。原始日志200Hz写行不等于200Hz新反馈；不能把重复值当独立样本。模型所含执行延迟与外部仿真延迟须界定边界，避免双重注入。

按整次采集分组，例如60/20/20拆训练/验证/测试，比例只是例子；同一时刻的不同关节也不能跨集合。优先报告逐关节Nm误差、波形/相位、在不同速度/负载和饱和状态下的泛化；与PD及PD+简单时延/摩擦基线比较后才决定是否接入。没有可信力矩标签时可先用q/dq响应辨识简单动态与时延，不能直接训练现有tau监督脚本，也不要仅凭数值微分反推动力矩作为真值。

本次已修复：补齐采集字段、提交目标与实际发送目标的区别及路线对比文档；无代码修复。待确认：ACT-NET-01仍未完成，需新反馈/发送时标、力矩语义与真实数据；本轮只有源码/在线一手来源核对和文档检查，无采集/训练/仿真。

## 10. 没有电流—力矩标定时的路线

用户确认目前未标定电流与力矩关系。应区分“用户没有自行标定”和“厂商也没有提供可信输出侧力矩估计”：前者不自动否定厂商tau，但须查明协议、标定和误差；后者则不能直接把未知电流乘随意常数作为现有脚本的Nm标签。

没有可信力矩仍能做辨识。先采已知目标与q/dq轨迹，在相同指令、初态和负载下对比仿真；优先拟合延迟、有效增益、阻尼、摩擦等少量参数。更进一步可在仿真内用执行器模型输出力矩，滚动生成q/dq，以真实和仿真轨迹误差训练模型，而非以tau为监督；已有[轨迹驱动的无力矩传感执行器辨识研究](https://arxiv.org/abs/2604.10351)展示该方向。这需要新的辨识/优化流水线，现有actuator_net.py不支持。没有力矩标签且惯量/摩擦也未知时，参数会相互补偿，拟合运动不代表辨识出唯一真实力矩；需可信本体参数、受控负载、多段激励及留出数据检验。

采集仍沿用第9节：保留每关节实际目标/发送时标、q/dq/新反馈时标、kp/kd/dq_des/tau_ff、状态有效性及温度/电压/负载条件。电流能读则保留原始值及其语义；tau可信度未定时标为辅助通道，不作真值。先采稳定运行日志；专门辨识再以受支撑单关节受限平滑轨迹覆盖不同方向/速度与后续受载工况。不可直接学习q_des到q然后强设仿真关节位置，这会绕过关节力矩、接触与刚体动力学。

已修复：补充无力矩标定时的选择与监督边界说明，无生产代码修复。待确认：ACT-NET-01增加厂商tau依据或输出侧标定，若走无力矩路线则需新辨识实现和可信模型约束。本轮仅一手论文与文档核对，未采集/训练。
