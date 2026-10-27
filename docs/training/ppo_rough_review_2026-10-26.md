# PPO rough奖励、模型与复杂地形训练复审（2026-10-26）

## 1. 任务和模型

注册任务 `Imgo2-basemove-rough-ppo` → `base_move/rough_env_cfg.py:Imgo2RoughEnvCfg` → 公共 `velocity_env_cfg.py`；算法配置在 `base_move/agents/rsl_rl_ppo_cfg.py:Imgo2RoughPPORunnerCfg`。建议入口 `imgo2_rl/scripts/rl_lab/ppo/train.py`，其agent注册键为rsl_rl_cfg_entry_point。

机器人来自 `assets/imgo2.py:IMGO2_CFG`，默认模型唯一源为 `imgo2_description/urdf/imgo2_real.urdf`，可用IMGO2_URDF_PATH覆盖。本轮检查环境未设覆盖；路径自检通过，17个网格引用存在。不是imgo2_real_wrong，也不是Gazebo专用URDF。PPO不使用AMP motion、判别器或模仿奖励。

12个关节按FL/FR/RL/RR，每腿hip/thigh/shank。初始base高.35m，默认角0/.87/−1.82rad；DCMotorCfg标称Kp25/Kd.5、力矩23.7Nm、速度30.1rad/s；训练还随机化增益，不能理解为固定25/.5。动作先裁剪±3，再乘hip .125、其它 .25并加默认角。物理dt=.005s、decimation4、策略50Hz、每回合20s。

Actor：45→512→256→128→12，ELU，非循环网络；观测为机身角速度3、重力投影3、指令3、相对关节角12、相对关节速度12、上次动作12，开启观测噪声。actor的base_lin_vel和height_scan被显式关闭，部署不输入地形高度图。Critic保留线速度和1.6×1.0m、.1m间隔的扫描；按当前本地pattern公式推算187射线，输入约235维，仅用于训练。维数来自源码推算，本轮没有运行环境实例化。

默认4096环境、24控制步/rollout（.48s）、2000轮、100轮存盘、初始噪声std1、学习率1e−3自适应、5个epoch/4个minibatch。配置注释里的“多少轮一定收敛”不作为保证。默认新训练从初始化网络开始，不自动使用当前部署policy.pt。

## 2. 最终生效的19项奖励

用标准库AST合并RewardsCfg定义与rough覆盖，之后零权重被disable_zero_weight_rewards设为None；以下是源码最终非零项，非运行时manager实测。细节函数在 `velocity/mdp/rewards.py`，未重写的项来自Isaac Lab。

| 项 | 权重 | 实际意义 |
|---|---:|---|
| track_lin_vel_xy_exp | +1.5 | 指令xy与机身系实际xy速度的平方误差经exp核，std=.5 |
| track_ang_vel_z_exp | +.6 | yaw角速度跟踪，std=.5 |
| base_height_l2 | −10 | base距扫描地面的高度保持.30m；不是世界z固定.30m |
| flat_orientation_l2 | −5 | 重力投影xy平方，倾向机身水平 |
| lin_vel_z_l2 | −2 | 机身系竖直速度平方 |
| ang_vel_xy_l2 | −.05 | roll/pitch角速度平方，不直接惩罚yaw |
| action_rate_l2 | −.01 | 本次与上次网络动作差平方 |
| feet_air_time | +1 | 首次落地结算sum(last_air_time−.5)，指令范数>.1时有效；可以为负 |
| feet_air_time_variance | −8 | 四足上次air/contact时长各裁到.5s后，惩罚两组方差之和 |
| feet_slide | −.05 | 当前实现罚触地足相对机身的平面速度，语义偏离真实世界滑动，见下文 |
| feet_height_body | −5 | 机身系足端z接近−.20m，按相对水平速度tanh加权，运动指令门控；不是地形净空8cm |
| joint_mirror | −1 | FR/RL、FL/RR对角腿对应关节角差平方，鼓励对角腿角度同步 |
| stand_still | −.1 | 指令范数<.05时罚相对默认关节角L1偏差，不等于直接罚实际自转 |
| joint_pos_penalty | −.1 | 相对默认角的L2范数，低指令/低速度分支加重5倍；不是关节越限项 |
| undesired_contacts | −.5 | 非FOOT部位接触力超过1N时计数扣分 |
| contact_forces | −.02 | 各足接触合力超过100N的超出量 |
| joint_torques_l2 | −2.5e−6 | 关节力矩平方 |
| joint_acc_l2 | −5e−9 | 关节加速度平方 |
| joint_power | −2e−5 | sum(abs(关节速度×力矩)) |

2026-10-27后续核对：上表joint_mirror的“角度同步”只描述原始角相等，不代表物理镜像正确。对当前四髋+X轴模型，该项缺少左右外展符号，会允许前外后内组合；8400保存配置与checkpoint还确认后髋探索std达22.20/28.36。原函数反例、日志和代表输入推理详见 [后髋饱和专项](ppo_hip_saturation_reward_audit_2026-10-27.md)。该专项为当前解释依据，修复与短训尚未实施。

多项自定义奖励另乘直立度门控。RewardManager结算为 `r_step=.02*sum(weight_i*term_i)`；上表数值是raw weight，不是AMP代码中先除step_dt后的“每步系数”，不应直接比较两套配置的权重数字。

旧feet_air_time在PPO这里确实启用，与AMP任务之前关闭/改deficit是两件事。该函数没有所谓L2 airtime kernel，源码实际为落地事件线性差值；threshold=.5s是单足滞空门槛，不是完整步态周期。举例air=.2s，一只脚落地该项贡献约(.2−.5)×1×.02=−.006（直立且指令门控开启）。

feet_height权重为−0.0，最终关闭，因此没有“已启用的足底净空8cm奖励”。joint_pos_limits、joint_vel_limits、joint_vel_l2、joint_deviation_l1、applied_torque_limits、feet_gait、feet_stumble等也为零并关闭；动作裁剪及执行器物理限位与这些奖励是否启用是不同机制。

## 3. 地形和指令

rough使用Isaac Lab的生成器，不读取Gazebo的.world。当前解释器定位到IsaacLab-rsl5；其ROUGH_TERRAINS_CFG结合项目覆盖后的地形如下。

| 类型 | 采样比例 | 参数 |
|---|---:|---|
| pyramid_stairs / inverse | 各20% | 上/下金字塔台阶，每阶2.5–8cm、踏面30cm、平台3m |
| boxes | 20% | 网格块宽45cm、块高2.5–10cm |
| random_rough | 20% | 起伏1–6cm，离散增量1cm |
| hf_pyramid_slope / inverse | 各10% | 上/下坡，斜率范围0–.4（不是.4rad） |

每块8×8m，10难度行×20地形列，max_init_terrain_level=5。terrain_levels课程保留，会按行走距离升降难度；线/角速度课程被关闭。不是所有环境初始都在最简单级别，也没有复杂地形成功的自动保证。

指令vx±1m/s、vy±.8m/s、yaw上限±1.5rad/s，10s重采样，2%静止组；heading_command=true且rel_heading_envs=1，因此yaw实际由目标航向误差控制并裁到范围，不能简单认为yaw均匀采样。平移模长≤.2m/s会清零，若想训练.1m/s低速需另处理死区。

新增Gazebo八阶8cm楼梯与训练的高度/踏面范围接近，但训练为金字塔混合地形，其排列、数量和平台不等同直线八阶场景。训练rough不会自动读取新增Gazebo楼梯，也不会自动修改已有policy.pt。

## 4. 能否直接训练，以及待修复项

结构上具备混合粗糙地形训练配置；45维Actor不读取地形扫描、Critic有特权地形信息。不能由此推断PPO不能学台阶，也不能推断8cm必然爬过。当前配方并非已验证可直接长训的版本。

1. **脚滑语义需修正**：feet_slide先减root_lin_vel_w再变换，世界静止的支撑脚也会因机身前进被惩罚；应明确要罚世界接触滑动还是足端相对机身运动，再验证，不能按函数名当作已正确的防滑项。
2. **倒地终止关闭**：illegal_contact=None，只保留超时/越界；base贴地不立即结束，undesired_contacts扣分不是终止。应按可允许的爬楼腿部接触单独确定base碰撞/倾倒终止，不能把全部腿碰楼梯都直接终止。
3. **滞空与姿态约束需对照**：.5s滞空门槛、variance−8、对角镜像−1、水平−5和竖直速度−2可能与短步登阶/非对称支撑所需动作竞争；这属于需实测的配方风险，不凭raw weight大小断言一定主导。按固定8cm楼梯完成率、姿态、净空和速度误差单变量调整。
4. **高度异常门控为跨环境全局判断**：base_height_l2用ray_hits全批次any NaN/Inf判断；一个环境射线异常可能使整个批次高度项走回退、近乎失效。台阶边缘/扫描越界时需验证并改成按环境有效性处理。本轮只报告未修改。
5. **本机前置仍未完成**：env_isaaclab分发版本0.54.4/isaaclab_rl .5.2/RSL-RL5.0.1；当前从仓库根find_spec看到imgo2_rl为外层namespace、rl_lab指向Downloads旧工程；runner没声明obs_groups，本地基类为MISSING。需正确安装/定位当前项目并完成观测分组兼容，再做短训练。旧记录说明见 [训练前置](ppo_training_preflight_2026-10-14.md)。不将这些静态发现冒充本轮已复现的训练异常。

下一步应先明确碰撞终止、修正脚滑/高度异常项并处理本机启动前置，做小环境短训练，确认19项结算、有限rollout与PPO更新，再开展地形课程训练。对当前8cm目标优先验收2.5–8cm楼梯，而非一开始扩大到15cm；范围/配方调整均尚未实施。PPO不需要重新采集AMP专家动作。

## 5. 已完成与验证限制

已完成：实际注册/继承配置与函数逐项核查、AST合并19项非零奖励、资产路径检查通过、读取实际解释器元数据及包定位、读取其Isaac Lab粗糙地形/奖励管理器/扫描pattern实现。`check_asset_paths.py`使用系统/usr/bin/python3；包定位检查使用env_isaaclab/bin/python。机器可读 [JSON](ppo_rough_review_2026-10-26.json) 保存源码/模型哈希和结论。

待修复/确认集中在第4节；需要源码修复、运行配置实例化、短训练和登阶评测才能关闭。本轮只更新文档与离线核对结果，没有修改训练代码、奖励、模型、安装环境或部署配置，没有启动Isaac Sim、训练或硬件。源代码可读不等于运行验证通过。

## 6. 课程的定义、开关与同时启用

公共 `velocity_env_cfg.py:669` 的CurriculumCfg注册三项：terrain_levels、command_levels_lin_vel、command_levels_ang_vel。当前rough继承terrain_levels，且没有把它设None；公共环境post_init会据此把terrain_generator.curriculum设True。rough_env_cfg.py:182–183明确把两项速度课程设None，因此只有地形课程生效。flat显式关闭terrain_levels；不要把基类定义误认为所有任务均开启。

| 课程 | 当前rough状态 | 更新规则 |
|---|---|---|
| terrain_levels | 开 | 环境重置时按实际位移升级/降级，更新相应环境地形等级 |
| command_levels_lin_vel | 关 | 若启用，初始xy范围缩为range_multiplier[0]，跟速项回合平均达到该项最高值80%后，两端各扩.1m/s，限于最终范围 |
| command_levels_ang_vel | 关 | 类似，按yaw跟踪奖励扩角速度范围，两端各.1rad/s |

速度课程函数在 `velocity/mdp/curriculums.py`。默认range_multiplier=(.1,1)，只在common_step_counter满足整回合步数倍数时检查扩范围；函数由环境重置入口调用，不是按PPO学习轮数固定升级。速度课程修改的是共享命令配置，地形课程修改各重置环境的地形级别，不能把两者都理解为每机器人独立速度上限。

开启速度课程可在rough配置中删除/注释两条设None的赋值，使继承定义保留；然后可以配置它们的params。这里只说明方法，没有改代码。要注意，当前默认初始xy范围±.1/±.08的最大平移模长约.128m/s，小于命令死区.2m/s，第一阶段平移指令将被清零；yaw不受该平移门控影响。因此不应只注释None就默认完成了低速课程设计，须一起明确起始范围/门限并短训验证。

框架CurriculumManager.compute逐项调用所有非None项，可同时启用多个。现有地形+线速度+yaw三项可同时注册；它们各自修改地形级别、xy范围、yaw范围。仍有耦合：地形升级/降级判据使用速度指令，速度范围和yaw运动变化会改变地形表现。避免多个课程同时修改同一字段且相互覆盖，并验证更新门槛，不保证多开一定更好。

项目另有lin_vel_cmd_levels辅助函数，面向HimLoco低/高速分组命令，需要rel_high_vel_envs和curriculums_limit_ranges等对应命令字段；本次全仓查找未见其作为CurrTerm注册，不属于当前PPO已接通的第四项课程。不能直接把它与command_levels_lin_vel一起接上普通PPO命令。

观察实际启用状态：训练启动打印Curriculum Manager活动项；日志标量使用Curriculum/terrain_levels、Curriculum/command_levels_lin_vel和Curriculum/command_levels_ang_vel（后两项仅在启用后产生有效状态）。这些值分别为地形平均级别与当前范围正上限，不是通用的课程完成百分比。

已完成：源码注册、子类覆盖、框架调用顺序和日志键复核。待确认：多项启用后的门限/耦合与短训效果、默认10%范围与.2死区的协调，继续由PPO-ROUGH-01跟踪。本轮未开关任何课程、修改参数或启动训练。

## 7. 速度课程的80%最高奖励如何计算

最高值来自奖励函数理论上界，不是运行历史最大值。两项跟速函数均为exp(−平方误差/std²)乘直立度门，二者都≤1，所以函数理论最高为1。

RewardManager每步累加weight*term*dt，当前dt=.02s，完整回合20s/1000步。线速度weight1.5：每步最高.03，20秒累计最高30；yaw weight.6：每步最高.012，累计最高12。课程实际比较 mean(对应分项_episode_sums[env_ids])/20 > .8*weight。因此线速度归一化上限1.5、门槛严格大于1.2（累计均值>24）；yaw上限.6、门槛>.48（累计均值>9.6）。仅统计指定跟速分项，不使用总奖励，不减其它项。

门槛不是单帧瞬时分数，也不是历史最好成绩的80%；均值取本次回调env_ids。提前结束仍除最大回合20s，最高可得值会按存活时长下降。改变权重时，跟速分项累计值和门槛同时同比改变，单纯增权不会降低这个相对达标要求。该80%判据依赖当前奖励函数确实以1为上界，未来若改函数尺度要同步改课程规则。

已完成：核对两个跟速函数、RewardManager的dt/回合归一化和两项课程判断，进行上述代数计算。待确认：课程当前仍为None，本轮没有启用、训练或核对真实达标日志；与默认起始范围/死区协调的待办沿用PPO-ROUGH-01。

## 8. 当前楼梯形状、课程最高级和实际高度

本节针对当前准备重训的rough源码，不是旧部署policy原始env.yaml。另核对Downloads旧rough/server_smoke保存env.yaml中的楼梯range也为.025/.08，但该run不匹配部署Actor，不能代替其来源证明。

两种台阶各占20%：pyramid_stairs为向中央升高的方形同心台阶，pyramid_stairs_inv为中央下凹的反向台阶。每块8×8m，外缘border1m、踏面.3m、holes=false、platform_width配置3m。生成器origin在中央平台，机器人通常从高台往外下阶或从凹底往外上阶，不是Gazebo的“前方1.5m处开始单方向直线8阶”。

台阶高度h=.025+difficulty*(.08−.025)。当前10行难度，difficulty_range=(0,1)，第i行随机取difficulty=(i+uniform[0,1))/10：

| 行/级别 | 单阶高度区间（cm） |
|---|---|
| 0 | 2.50–3.05 |
| 1 | 3.05–3.60 |
| 5 | 5.25–5.80 |
| 8 | 6.90–7.45 |
| 9（最高） | 7.45–接近8.00 |

因此配置上限是每阶8cm，实际最高一行接近8cm，并非正好每块都8cm。max_init_terrain_level=5只限制初始随机等级0…5，约覆盖2.5…5.8cm；后续可以升到9。通过末级后terrain_importer将环境随机分回0…9，不会生成第10行或无限升到10/15cm。

**实际几何离散补充**：本机当前mesh_terrains.py以num_steps=int((8−2*1−3)//(2*.3)+1)得到6圈，再加中心台面。按实际源码构造的axis-aligned box坐标核对，中央有效平面宽约2.4m（不是把配置3m直接作为最终实测宽度），沿边到中央有约7次高度变化；理论difficulty=1、单阶8cm时中心相对外缘±.56m。先前表格“平台3m”是配置参数，这里补充实际离散结果，不能把它当成Gazebo八阶总高.64m。真实最高行difficulty<1，所以累计高度通常略小于.56m。

训练这些地形有机会学会小台阶上下，但这是训练样本范围，不是对策略爬越能力的保证。需先处理第4节配方和环境前置，短训验证后训练；以实际Gazebo八阶8cm场景验收。若要更高，需显式改两种stairs的step_height_range上限并验证，不是多训练几轮自动突破8cm。本轮没有改高度、课程或启动训练。

已验证：当前rough覆盖、本地实际terrain_generator.py难度采样、terrain_importer.py初始/最高级回收逻辑；标准库执行实际台阶函数AST并以记录器代替trimesh/数组构造核对box尺寸/位姿，不启动Isaac。结果见 [课程与几何JSON](ppo_rough_stair_levels_2026-10-26.json)。限制：只验证构造参数与解析几何，非重新生成完整训练场景或完成新的策略训练；登阶效果仍待PPO-ROUGH-01/GAZEBO-STAIRS-01。

## 9. 上下楼梯选择重新训练还是续训

两条路线都成立：直接用rough任务新训，或者从已有行走Actor初始化后在rough中微调。地形课程在同一次运行中逐步更新难度，不需要每一级重新训练一个policy；一个策略可以同时学习平地、上阶和下阶。需评估混合训练样本是否覆盖目标场景，而不是只凭任务名确认能力。

针对当前工程，建议先处理第4节前置，再单独新建 `Imgo2-basemove-rough-ppo` 训练运行，保留现部署策略作为对照。当前部署只有已核对的导出JIT，59个旧checkpoint未找到对应Actor，不能把这个policy.pt直接当作普通训练checkpoint交给resume。现有ppo/train.py通过get_checkpoint_path定位checkpoint并调用runner.load恢复训练状态，没有Actor单独迁移的分支。

此外，当前flat关闭critic.height_scan，而rough保留：按源码推算flat Critic为48维、rough约235维，Actor均45维。直接把flat完整checkpoint加载到rough可能因Critic形状不一致失败，不能仅换任务名加resume就保证可用。若明确选择迁移，应先验证Actor观测顺序、动作尺度、模型/PD和归一化兼容，单独加载兼容Actor并初始化新的Critic和优化器；这是待实现和短训验证的方案，不是现有入口已支持的行为。新建运行也可以用迁移权重初始化，不等于必须随机初始化。

已完成：重新核对flat/rough的观测覆盖和ppo/train.py恢复入口，澄清训练路线。待修复/确认：第4节训练前置、实际上下阶完成率；Actor迁移仅为可选方案，尚未实现。未改训练代码、参数、部署policy或启动训练。

## 10. 本轮正常行走目标与旧部署策略的关系

当前部署PPO已完成两引擎转向/减速实测和.1/.3m/s低速对照，数据结论统一写在 [专项记录](../deployment/ppo_maneuvers_2026-10-26.md)。转向摆动与急减速瞬态在两边存在、Gazebo较大；.1m/s几乎不走。未重建原run奖励，不能拿当前rough权重直接断言旧策略表现的根因。

本轮重新读取rough覆盖，AST核对本文件第2节19项非零权重不变；feet_height=0、stand_still为默认角偏差、feet_air_time=.5为单足滞空门槛等语义不变。后续先以站立、低速持续行走、左右转向、快速/缓慢加减速作为验收，明确.2m/s死区与低速覆盖，处理脚滑/高度/终止及训练前置，再做短训对照。没有擅自调权重、改课程或启动训练；待办由PPO-GAIT-01/PPO-ROUGH-01跟踪。
