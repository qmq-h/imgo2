# 新 motion_new 25000：切换抖动、后退与转向排查

日期：2026-10-13。关联：AMP-17、AMP-20/21/22、DEPLOY-14、MODEL-05。随后按用户前进0.4/后退/转向贴地反馈，补测足底高度与原Isaac接触，详见 [足端记录](../amp/amp_foot_lift_review_2026-10-13.md)。

后续AMP-20的Gazebo策略裁剪/last_action已修复并完成实际对照，见 [最新记录](gazebo_amp_clip_fix_comparison_2026-10-13.md)。本文保留首次排查/过渡实验的状态与数据。

## 1. 结论与限制

- 没有发现当前 real 训练/Gazebo URDF 的本体质量、质心、惯量或关节轴/限位错配，也没有非正定或不满足主惯量三角约束的惯量。不能据此保证这些CAD数值等于实机测量值。
- 后退不足在同一actor的原Isaac仿真里也存在：命令−0.5，Isaac约−0.317、Gazebo约−0.327 m/s。因此这部分不能仅归咎于Gazebo代码或惯量。后退关节节律也比前进弱。
- 切换时存在真实的目标角/PD突变；原实现首条策略命令最大关节跳变约0.10–0.62 rad。可选渐变显著减少第一条跳变，但未一致改善关节速度峰值，因此当前默认关闭，不能标记“抖动已修好”。
- 存在动作裁剪阶段、DC电机限幅、自碰撞及PD/反馈频率等训练/部署差异。其中动作裁剪语义不同是明确的接口问题，需受控修复验证；其余差异在本次没有被单独证明为抖动根因。
- 本轮没有修改惯量、关节轴、模型、训练奖励或权重；没有训练，没有重新导出/替换策略，没有真机运行。只读核对用户已安装的新策略，增加测量/复现工具及默认关闭的Gazebo切换实验功能。工作区launch另有变更，本轮未编辑它；测试始终显式指定real模型。

## 2. 实际使用的策略与接口

运行目录：`imgo2_rl/logs/amp_rsl_rl/base_move_amp_height/2026-10-13_00-35-21_motion_new_resume/`，checkpoint为model_25000.pt（iter=25000），不是10月11日旧滞空微调目录里的同名文件。

| 文件 | SHA256 |
|---|---|
| 新model_25000.pt | `93284a62dfff214f9f4e79109c524dca7e6b3fb2892ae8651b1004a06a524f94` |
| exported/policy.pt及部署policy.pt | `e40d81ec219c777a4aa88acffed935f12be526a2b8a52d7d56d38f3c9b7cc9d1` |

部署文件与用户通过官方AMP play.py导出的文件逐字节相同。128组seed1输入，TorchScript与checkpoint的ELU actor最大差=0；部署自己的libtorch 2.3.0+cpu加载并完成有限值45→12推理。

保存的deploy.yaml确认：actor45、critic48；FL/FR/RL/RR逐腿hip/thigh/shank；原生分组关节通过joint_ids映射后默认姿态为每腿0/.87/−1.82；动作缩放.125/.25/.25；策略PD25/.5；控制周期.02 s；观测角速度/重力/命令/相对关节角/关节速度/上次动作的顺序及缩放与C++对应。**裁剪数值相同不等于作用阶段相同，见第6节。**

保存agent.yaml：从原motion_new的model_10000恢复，再训15000轮；seed42；task lerp=.3、AMP coef=2，新21段数据。保存env.yaml中范围vx±1、vy±.4、yaw±1，feet_air_time=None。服务器资产路径指向real URDF，但本轮无法核验当时服务器转换出的USD或CAD原始测量。

[数值/接口及源码哈希](../amp/data/amp_motion_new_25000_contract_2026-10-13.json)。策略配置的旧4000来源注释已更新为本次25000，PD/缩放等数值未改。

## 3. 模型与惯量检查

独立标准库检查 `imgo2_rl/scripts/tools/audit_gazebo_model_physics.py` 对当前两份real URDF：

- 17个共同link的inertial（质量、完整6项张量、xyz和rpy）与碰撞几何一致；全部关节结构/轴/限位/原点一致。
- 正质量、惯量正定与主惯量三角约束均通过。非零或负的惯量交叉项本身不表示错误。
- 训练本体总质量12.95640984 kg；Gazebo多一个0.01 kg的base_imu，总质量12.96640984 kg。
- base质量5.53394020，主对角0.03866860/0.10411461/0.12554111；不是body对角符号错误。

[逐link结果](gazebo_model_physics_2026-10-13.json)。这是模型文件一致性/物理有效性检查，不是实机惯量辨识。

全仓 `check_model_sync.py` **仍失败**：14网格/指纹66c496556453与旧登记10文件基线不符，两份real URDF未登记，沿用MODEL-05。本轮独立检查补足real模型对比，没有假装全仓守门已通过，也没有修改网格或免除未登记模型。

## 4. 实际回放与方向表现

### Gazebo

ROS2 Humble/Gazebo Classic，独立ROS_DOMAIN_ID59、localhost、master11369、gui=false、软件GL；每个指令重建一套世界和rl_sim。real URDF；当前用户已部署25000策略；PD/缩放保持原值。26墙钟秒，前6秒起身，进入AMP后跳过4秒；指标用仿真时间，关节统一100 Hz。所有本轮启动的Gazebo/控制进程已结束。

基线8个指令单次短测试；过渡候选两种合计8次；默认关闭做1次短验证。不能把单次结果作为多种子置信区间或长期不摔保证。

### Isaac

同一导出actor，当前本机env_isaaclab，headless、6环境、900步（18仿真秒），跳过前200步；每环境一个固定身体速度指令，关闭startup/interval随机化，重转当前URDF，使用当前play任务的原生动作与DCMotor实现。均未发生episode reset。

这是原仿真器的确定性回放检查；接触、摩擦、传感器和初始状态不与Gazebo完全一致，所以不把两个数字的差全部归因于某单项参数。

| 指令 | Isaac实际vx / yaw | Gazebo实际vx / yaw | 说明 |
|---|---|---|---|
| 前进vx=.5 | .364 m/s / .008 rad/s | .508 m/s / −.003 rad/s | Gazebo此条件跟速好，Isaac也存在误差 |
| 后退vx=−.5 | −.317 / −.047 | −.327 / −.050 | 两边均明显低于目标，均有偏航 |
| 后退vx=−1 | −.835 / −.032 | −.777 / −.029 | 仍有后退欠跟踪 |
| 原地yaw=1 | .026 / .952 | .022 / .938 | 均能跟踪纯转向，仍有小平移 |
| 后退−.5加yaw=.5 | −.323 / .422 | −.331 / .436 | 组合指令两边都欠跟踪 |

Gazebo稳定站高约.285–.291 m；Isaac约.297–.307 m，存在约1–2 cm差异。Gazebo前进大腿节律周期约.268 s（3.74 Hz），强度.854；后退强度仅.381、FL/FR相位估计偏离交替关系；后退加转向强度.373且多频混合，不能把6.25 Hz候选直接当可靠足端接触步频。纯站立的小幅信号也不能按自相关峰解释为“站着仍在迈步”。

[Gazebo基线](gazebo_amp_motion_new_25000_baseline_2026-10-13.json)、[边界/组合指令](gazebo_amp_motion_new_25000_baseline_high_2026-10-13.json)、[Isaac方向回放](../amp/data/isaac_amp_motion_new_25000_directions_2026-10-13.json)、[关节频谱](../amp/data/amp_motion_new_25000_spectrum_2026-10-13.json)。这一阶段未测足端接触；随后追加Gazebo±0.4的足底几何检查与原Isaac8环境900步实际接触/滞空检查，证实后退/转向贴地，见 [补测](../amp/amp_foot_lift_review_2026-10-13.md)。未复现用户GUI里的全部速度操作。

## 5. 切换问题与实验修改

原代码GetUp使用fixed PD60/2，进入AMP后RLControl立即发送新q_des和PD25/.5。AMP中的percent_transition没有被用于平滑命令。单次目标角跳变最大约.102（站立）、.220（前进）、.130（后退）、.619 rad（yaw=1）；控制量确有不连续。

在 `RL_Sim::RobotControl` 增加只作用于Gazebo的可选交接，保存前一状态命令，以smoothstep渐变目标；每拍先恢复未混合目标，避免对已混合命令递归混合。共用FSM、硬件和MuJoCo入口未改。

对照结果（单位rad；速度峰为首秒最大abs(dq)，rad/s）：

| 场景 | 原版第一条q跳变 / dq峰 | .5s目标+增益渐变 | .5s只渐变目标 |
|---|---|---|---|
| 站立 | .102 / 5.52 | .0034 / 6.85 | .0023 / 10.42 |
| 前进.5 | .220 / 13.19 | .0062 / 18.76 | .0051 / 17.38 |
| yaw=1 | .619 / 8.61 | .0141 / 10.59 | .0171 / 5.51 |

渐变减少第一条q跳变，但速度峰不是全面改善。它还改变了执行目标与网络last_action之间的关系；这个反馈一致性问题需下一轮研究，不能据本表宣称根因已经完全修好。当前 `amp/config.yaml`：

```yaml
policy_transition_duration: 0.0
policy_transition_blend_gains: false
```

0表示关闭，参数缺省也关闭，当前默认继续原控制行为。候选.5仅作为可复现实验值。已完成编译、C++端点/连续性检查和实际回放；默认关闭验证首条q跳变约.101 rad，与原控制形态一致。整体切换抖动状态仍待DEPLOY-14。

[增益渐变稳定窗口](gazebo_amp_motion_new_25000_transition_2026-10-13.json)、[目标渐变稳定窗口](gazebo_amp_motion_new_25000_target_only_2026-10-13.json)；瞬态JSON索引： [原版](gazebo_amp_transition_baseline_2026-10-13.json)、[快/组合指令](gazebo_amp_transition_baseline_high_2026-10-13.json)、[增益渐变](gazebo_amp_transition_transition_2026-10-13.json)、[目标渐变](gazebo_amp_transition_target_only_2026-10-13.json)、[默认关闭](gazebo_amp_transition_default_disabled_2026-10-13.json)。

## 6. 明确差异与下一步

### 动作裁剪（AMP-20）

本机Isaac Lab的JointAction.process_actions：先 `raw*scale+offset`，再对关节目标裁剪到±3 rad；AMP wrapper的clip_actions=None。Gazebo Forward先裁剪网络原始动作到±3，再缩放/偏置，last_action也保存该裁剪结果。两者语义不同。

以shank默认−1.82、scale=.25、raw=−5举例：训练目标裁剪后−3 rad；Gazebo为−2.57 rad。值相同的clip配置没有证明接口等价，已修正旧文档“裁剪相同”的说法。基线稳定窗口从传输目标反推的边界占比约0–.67%，提示本次大部分动作不在边界，但不能排除少数关键帧作用。

完成所需：为已知训练配方明确裁剪阶段及last_action定义，兼容其它PPO/HIM策略，记录实际原始动作，修正后做单变量回放与数值契约测试。本轮未改裁剪实现。

### 执行器、接触与时钟

- 训练DCMotor按速度衰减可用驱动力矩；Gazebo当前只做±23.7固定限幅。
- Isaac物理/执行器200 Hz；Gazebo物理2000 Hz、PD1000 Hz，PD反馈速度由位置差分估计；策略/命令线程名义50/200 Hz按墙钟运行。
- 训练关闭自碰撞；Gazebo多个link启用self_collide；训练随机摩擦范围与Gazebo固定foot μ=.6也不是完全相同条件。

这些是已确认的配置/实现差异，但未做逐项消融，不标记它们是已证明的根因。不要以这份记录为理由随意修改关节轴或惯量。优先完善裁剪/执行器与反馈接口对照，并用原Isaac同指令复核策略。

### 训练侧（AMP-21）

后退欠跟踪与组合运动问题在原仿真中存在。保存的速度tracking std=.5，按本次后退均值误差.183 m/s估算，单项理想kernel仍约exp(−(.183/.5)^2)=.874；这是均值代入示例，不是实测平均分项奖励。该奖励并不强迫速度误差趋近零。需要固定方向/组合命令回放、不同种子与关节/足端时序，再单变量研究速度tracking核或权重、AMP配比和数据；本轮不把任何调权方案标成已验证收益。

运行时还应看rl_sim显示的x/y/yaw是否在训练范围内：键盘W/S/Q/E是累加值，目前没有自动限幅；超过vx±1或yaw±1属于未覆盖指令，不能由本次范围内短测保证表现。

## 7. 复现入口与完成状态

离线模型：

```bash
python3 imgo2_rl/scripts/tools/audit_gazebo_model_physics.py
```

已有Gazebo与rl_sim按正常顺序启动后，每个指令使用全新栈，例如后退转向：

```bash
python3 imgo2_deploy/scripts/eval_gazebo_policy.py --key 3 \
  --vx=-0.5 --wz=0.5 --duration=26 --skip=4 --sim-time \
  --json=/tmp/imgo2_direction.json --transient-json=/tmp/imgo2_transition.json
```

原Isaac方向检查（仓库根、实际Isaac环境）：

```bash
IMGO2_ISAACLAB_PYTHON=/home/tommy0929/miniconda3/envs/env_isaaclab/bin/python \
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/tools/eval_actor_directions.py \
  --headless --device=cuda:0 --steps=900 --warmup=200 --json=/tmp/imgo2_native_directions.json
```

原始ROS日志、全起身/切换/稳定telemetry在上述训练run的 `gazebo_audit_2026-10-13_{baseline,baseline_high,transition,target_only,default_disabled}/`（logs不入库）。900步原Isaac检查已实际完成；持久化脚本另做20步冒烟通过。命令与路径按本机核对，跨机需选实际解释器。

已修复/完成：策略来源注释、旧裁剪等价说明、评测器新增vy/wz与瞬态记录、后退error_pct分母、real模型有效性检查与方向/切换实测、可选过渡实现及编译。4项审计测试、3项周期测试、C++过渡测试与rl_sim构建通过；默认关闭做过运行验证。

待修复/确认：整体切换抖动（DEPLOY-14）、裁剪契约（AMP-20）、后退/组合策略品质（AMP-21）、抬脚/周期与slide奖励语义（AMP-22）、各物理差异的单变量影响、长期/GUI/Gazebo接触力验证和全仓模型登记指纹（MODEL-05）。原Isaac接触已在补测完成，不能外推到长期或真机。需要下一轮针对这些明确风险的对照；本轮不将代码实验等同于现象修复。
