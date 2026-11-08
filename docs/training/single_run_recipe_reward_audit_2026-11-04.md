# 少量完整训练、接近policy_1平地品质并适应rough的方案审查（2026-11-04）

## 用户目标与状态

用户明确：平地尽量接近policy_1，同时适应多种地形；减少人工改参数后多次续训，训练轮数可以充分。前面多轮短对照是排查策略，不作为接下来长期主路线。建议先定一套可复用配置，自动课程在一个run内完成，最多两服务器同配方不同seed验证；不能保证一次训练必成或所有指标同时超越基线。

本轮仅源码/日志/作者配置审查与CPU Actor初始化可行性检查。没有修改训练入口、奖励、配置、部署或启动训练。所有下面配方更改仍是候选，不能标成训练效果已验证。

## 1. 可行起点：policy_1的Actor初始化（新核实）

policy_1没有已确认原训练checkpoint，所以无法完整恢复其原Critic/优化器/std；但不等于只能当外部教师。JIT含完整mlp权重，结构45→512→256→128→12，ELU，obs_normalizer与deterministic_output均Identity，八个权重形状与当前5899 actor完全一致。CPU读取其权重到同结构nn.Sequential，seed42的1024组45维输入max_abs_diff=0，[证据](policy1_actor_initialization_2026-11-04.json)。未写新checkpoint、未进行手写部署导出，仅内存验证读取参数可行。

因此可新建run，用policy_1初始化Actor，Critic/优化器/探索分布重新建立，保持45维观测顺序/尺度、动作映射、默认角/模型/PD不变。训练起始确定性行为应验证与policy_1一致。初始化新Critic和探索后不能等同于完整resume，也不能保证优化后自动保留原步态。启动阶段应保护已有Actor、控制探索，必要时短暂Critic预热和逐步释放在同一入口自动执行；具体调度未实现。

若需要进一步防止平地退化，可预先设计仅平地直行/后退/横移的冻结policy_1辅助损失，避开教师本身欠跟踪的纯yaw和rough。这是可选新增代码，不能将其当成已有flag。若优先实现简单性，第一版可先只用Actor初始化+持续平地样本，避免同时引入教师权重调参。

## 2. 一次完整run的组织

- 固定一套reward与指令分组定义，地形保留约40%平地，其余rough分级，课程只升级rough，不让全部样本迁移到复杂地形；40%是候选并非最佳值。
- 起步覆盖前后/横移/直接yaw/静止和混合；后退/纯yaw有明确比例，基础范围常驻。扩速时只扩展一部分样本，而非覆盖全部低速样本。保留当前可接受的小速度死区，初始平移运动幅值避开.2清零。
- 地形3～5cm到8cm、速度支持集从基础到高速，由预设能力门槛自动扩展，始终保留基础样本；不是人工每次改文件再resume。现有command_levels函数只按tracking阈值升级，不足以保证步态/姿态质量门槛，需检查/扩展后再用，不直接宣称当前已支持完整自动验收。
- 一次少量冒烟核对程序、实际采样分布和Actor身份，随后完整训练；定期评估/选checkpoint不算一次新的续训练。预算按实际指标决定，不保证固定5000/10000最佳。
- 两服务器优先同一配方seed42/43完整跑，检验复用性，而非再次给两个服务器互相不同配方。源码/依赖/模型及配置一致；保存commit+未提交diff、模型/策略hash、env/agent，避免此前git目录为空而只能猜服务器reward版本。
- 完整run可从上述Actor初始化开始；若选择5899作为起点也可一次长run执行，但它已偏离基线，平地恢复负担较大，不把这个判断当已完成训练比较。相同配置的中断恢复是正常功能，不应禁止resume本身。

## 3. 奖励审查：目标冲突与强度问题分开

### FHB（优先改机制）

当前-4、body目标-.20，摆动掩码×高度平方误差×world xy速度tanh；只摆动生效已修，但仍无摆动相位目标，机身系高度可通过降低身体减小误差，且水平足速接近0会减弱信号。前进+.5已有8.36cm净空、后退只有2.19，不能继续统一抬高目标或加权。面向最终通用配方建议采用相对局部地面的摆动净空/平滑阶段目标或合理净空范围，保留支撑掩码，起落阶段允许接近地面，平地与rough允许不同需求。需要训练侧足下地面信息，不能把world z当rough净空；这些信息不必新增部署Actor输入。地形完成/速度跟踪需防止“不迈步也无惩罚”的投机，不能只靠足高项驱动起步。

### feet_air_time（重审目标，不是简单加重）

现1.8、threshold .5，落地时(last_air-.5)，不是周期误差。policy_1同条件实测+.3/+.5/-.3 air约.191/.174/.179s，侧移约.166/.167；.5不是需要达到的周期。threshold .5本身不是程序错误，公开Unitree也有它，但本项目强度和组合不同；不要把负reward必然解释为没收敛。最终低速慢步配方可用约.17～.20s的有限滞空目标/短缺惩罚作候选，而非无界鼓励更长滞空；仍不能单独控制周期/防止不落地取巧，也不应将低速时间固定强加高速。权重需按实际贡献定标，不盲目复制AMP每步缩放或改成巨大值。

### joint_mirror（与通用rough目标有结构性冲突）

现-1，要求对角thigh/shank实际角度接近，不是接触相位trot奖励。不同脚踏在不同高度、转向/横移时，合理关节角可以不同；全时同角度会限制适应。最终方案建议仅平地合适运动下弱化保留，rough关闭，或直接省去这一额外姿态约束；不因此强制加入feet_gait，真实trot可自然形成。两选项需最终实现时确定一条，不能暗示任一已证实最好。

### feet_air_time_variance（强度/适用范围问题）

现-8，惩罚各足last_air与last_contact时间的方差，无command gate且时长clip .5。平地规律步态可受益，但rough允许前后足支撑时长不同；建议最终配方弱化并限定平地/规律运动，不以四足时长完全一致为全地形目标。不是语法错误或已证实抖动唯一根因。

### 多个姿态项重叠（需简化）

joint_deviation_l1四髋-1、全关节joint_pos_penalty-.1、stand_still-.1，再加mirror；会共同约束偏离默认姿态。应保留站立约束与适度髋软约束，避免运动中多项都强烈拉回默认角。髋不应硬锁0；全关节偏差成本可能压制大步幅，但本轮没有证明它导致快步。不要把所有姿态惩罚都设0，再次放任极端动作。

### base_height与稳定项

base_height=-18/target.30方向合理，但rough使用射线均值作ground reference；楼梯跨高度时它不必等于支撑面，信号基准变化可与姿态目标竞争。valid-ray修复不代表任意地形高度定义都正确。flat_orientation=-5不应在坡/阶无限增强；身体需要一定pitch。lin_z=-4、ang_xy=-.05均为合理正则方向，现有试验不支持把所有系数同时加大能解决机身不稳。

动作变化、力矩、加速度、功率等项方向合理，不能仅比较weight大小；力矩/加速度的现系数较Unitree参考小，是否需要增强应按真实贡献、饱和率和动作波形定标。动作变化项已非零，不说当前完全没有平滑性约束。

## 4. 当前量级证据与公开配置边界

5899最后5500–5899日志分项：track_lin约+1.270、track_yaw+.495；hip L1-.227、airtime-.193、lin_z-.093、contact force-.094、action rate-.085、airvariance-.077、angxy-.071、jointpos-.062、mirror-.054、base height-.01265、jointacc-.01152、FHB-.00810、torque-.00097。它们是同一日志口径的加权贡献，可看量级，不是因果梯度/重要性排名；base高度与airtime符号不同、原量单位不同，不能反过来直接决定最佳系数。[完整分项](ppo_reward_contributions_5899_2026-11-04.json)。

[Unitree官方Go2配置](https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py)本次核对含feet_air_time .1/threshold .5、variance -1、lin_z -2、angxy -.05、action_rate -.1、torques -2e-4、joint_acc -2.5e-7，以及姿态/关节约束和跌倒终止。说明多类正则有公开先例，但不是本机器人直接照抄依据。参考机器人质量/惯量、函数细节、动作缩放、部署PD和版本可能不同；本仓需按相同物理定义/日志量级校准，不把比值视为调参指令。

## 5. 实现顺序与验证边界

将上述选择一次性定为版本化配方：Actor初始化、显式指令、平地常驻、自动难度、精简姿态项与合理足高/时间定义。先做单元/静态/运行冒烟，足高/时间在现有回放上估算量级；再启动完整run，不把大量人工短续训作为常规流程。不保证无需任何后续调整，也不声称所有候选权重已测。若希望最少新增机制，先不加入额外教师损失，由Actor初始化保留起点；平地样本与评测门槛持续保护目标。

已完成：源码/5899日志审查、作者配置核对、policy_1 Actor复制CPU输出差0验证。待实现/验证：训练入口初始化、Critic/探索安排、分组采样/平地地形、课程门槛、最终reward定义与固定系数、完整短冒烟/训练/多seed部署。当前reward未改，建议不等于修复。


## 三项奖励含义澄清与rough公开实例（2026-11-04补充）

1. last_air_time本身就是最近一次完整离地到触地的时长，通常近似摆动/摆腿时间；last不是上一训练轮。它由接触状态定义，不是关节是否正在运动的运动学分段，碰地拖动或四足同时腾空时与预期摆腿定义可能不同。当前只在first_contact落地事件结算(last_air_time-threshold)。因此“不是摆腿时间”的表述应改为“奖励不是时间本身，而是超过门槛的差值；它不是完整步态周期或追踪指定时间的平方误差”。θ=.5，air=.2/.5/.6对应原始单脚值-.3/0/+.1，尚未乘weight/dt。减门槛使短滞空落地也可能扣分，而不是每次抬脚落地都获正奖励；仍无唯一目标周期，拉长支撑/减少落地也影响收益。

2. joint_mirror本地实现是对配置中的对角thigh/shank关节角做同一时刻差平方，非足端接触相位奖励，也不等于一般对称数据增强/镜像策略一致性。公开[legged_gym基础reward](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py)及[ANYmal C rough继承配置](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/anymal_c/mixed_terrains/anymal_c_rough_config.py)、[HIMLoco A1 rough配置](https://github.com/InternRobotics/HIMLoco/blob/main/legged_gym/legged_gym/envs/a1/a1_config.py)及其base/机器人源码，本次核对未使用这种逐时刻对角关节同角度项。所以它不是rough训练的必需条件，但不能据此说所有开源项目都没有任何对称约束，也不能单凭源码证明本机步态问题由mirror造成。

3. feet_air_time_variance_penalty实际为跨所选四脚的Var(clip(last_air,max=.5))+Var(clip(last_contact,max=.5))，乘直立门、负weight和dt形成惩罚；不只是抬脚时长方差，还含支撑时长，比较各脚最近完成时段，不是单脚历次步态的时间序列方差。各脚时长接近时小，长短不一时大；时长都短/都长仍可能方差小，不能单独规定频率、接触相位或保证trot。rough可有合理时长不一致，因此适用范围/强度需与地形目标匹配，不断言本项必然有害。

补充来源边界：当前Unitree官方Go2 velocity_env_cfg的air_time_variance=-1确实存在，但当前公开页面启用的地形字典只有flat，其余rough类型被注释；不能单用该页证明这一版本已验证rough训练。数值不能直接作为本机-8必须改成-1的依据。此次仅解释/查证，无奖励代码/配置修改、无训练。


## .5阈值与宇树配置再确认（2026-11-04）

当前正常步态air<.5时，首次落地且移动/直立门开启，该项为负；非落地步通常0，零指令门关闭也为0，所以不说“每步一直扣”。负贡献不自动表示未收敛/配置错误，较长air可以获得较少负值，即相对改善。只凭负数不能决定必须改阈值。

再次核对[Unitree官方Go2配置](https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py)：feet_air_time threshold=.5、weight=.1；本5899为.5/1.8。权重数值为18倍，不等于跨机器人/实现/分布后的实际影响必然18倍。不能泛称所有宇树型号/原厂闭源策略如此。将阈值改为基准时长的建议是目标设计候选，不是修复一个已证明的.5 bug；当前长期负值本身不支持直接改。仅核对与记录，未改代码/训练。

## 既有对照后的保守续训配方（2026-11-04，未运行）

用户希望综合既有对照确定可复用参数。本轮重读feetheight_ab、followup和terrain_resume报告，检查当前rough源码及logs目录；未发现新增airtime1.0运行材料，因此不能报告其效果。本轮没有重新导出/仿真/训练，也未更改训练配置。已完成的是结果复审和候选整理，参数最佳性与因果仍待验证。

既有实测支持保留：FHB -4/target-.20；-1和target-.24均没有全面改善。ang_xy保持-.05：-.10部分方向更稳，但后退跟速/净空退步。5899的前进+.3约2.17Hz/27.26cm、后退-.3约2.50Hz/27.49cm/2.19cm净空，参考policy_1分别约2.02Hz/28.91cm与2.08Hz/28.93cm/3.72cm。rough总体5899较好但后退仍弱，不能宣布通用策略完成。

当前本机源码并非5899配方：base_height -16（5899 -18），lin_z -5（原-4），airtime1.6（原1.8），variance-5（原-8），yaw tracking2.0（原.6），vx±1.2/vy±.8（原±1/±.6），父类初始maxlevel5（原1）。服务器配置未知，不能自动视为相同。特别是yaw奖励2.0未经本轮对照，不能与airtime1.0一起改变再归因。

建议当前5800续训采用可追溯的保守候选：继承该run的保存配置，仅airtime1.8→1.0，threshold.5不变。关键固定：tracklin1.5、trackyaw.6、baseheight-18/.30、lin_z-4、ang_xy-.05、flat_orientation-5、FHB-4/-.20、airvariance-8、hipL1-1、jointpos-.1、stand-.1、mirror-1、actionrate-.01；其余力矩/加速度/功率/接触/滑动沿原run，PD/模型/动作缩放不改。速度±1/±.6/±1，地形课程开、速度课程关、初始maxlevel1。该选择用于有依据地减少改动，不代表mirror/variance等已经证明是rough最佳值。实际训练不自动读取旧env.yaml，需要服务器手动保持或覆盖并核对新生成的env.yaml。已有airtime1.0运行若还改了其它项，记录为联合修改，不追认为单变量试验。

先追加1000轮观察，保存中途checkpoint；同条件比较前后±.3/±.5、横移±.3、纯yaw±.3、减速到零和rough前后。看实际跟速、每足摆动/支撑/周期、净空、机身均值/波动及角速度，不用总reward或奖励符号作验收。降airtime不保证慢步，可能减少异常长腾空，也可能进一步快步；方向必须实测。

若仍然前进足够高而后退低，下一优先是显式静止/后退/横移/纯转向采样与平地常驻，而非再全局抬高足高目标。候选静止10%、前进20%、后退25%、横移15%、纯转向15%、混合15%，平地约40%其余rough，均未实现且非最优数值。只缩范围不能保证纯转向，heading=True可能改写yaw。通用rough配方下一机制工作仍为局部地面摆动净空/阶段目标、弱化或按地形限定mirror与时长方差；当前保持它们是为了控制本轮变量，并非否定此前结构性问题。待补服务器新run材料和实际评测后再决定是否采用候选，不同时叠加这些未验证更改。

## 新附件5800续训配置复核（2026-11-04）

以1f8fef9b-1c82-4d0e-8f48-733d2112a7c8附件为准，上一份b9eb附件不作本轮结论依据。已完成AST/compile（仅编译，不导入）、显式赋值提取，证据见[附件指纹与赋值](ppo_resume5800_attachment_2026-11-04.json)。未修改附件或项目训练配置，未构造Isaac环境、未启动训练/仿真，服务器继承源码与实际env.yaml待核对。

没有发现Python语法错误。新附件baseheight-18/.30、lin_z-4、tracklin1.5/yaw.6、air1.0/.5符合前述候选；但相对5800所属run还有FHB-4→-5、ang_xy-.05→-.10、vx±1→±.8、地形课程开启→关闭四项额外变化，不是只改airtime。若目标是airtime对照，恢复FHB-4与ang_xy-.05、vx±1，删去两个terrain关闭赋值并显式初始maxlevel1，其余沿旧run。若有意做新组合，语法上可以，但不能将结果归因airtime，-5与-.10的组合尚无验证。

两个terrain关闭赋值互相匹配，确实同时关闭课程项与生成器分级开关；这不等于切为平地或简单地形，既有boxes/rough/stairs仍在，关闭时生成器可以在完整难度范围抽样。继承当前本机父类时standing为2%、heading=True/heading env比例1，xy指令模长≤.2清零，没有明确方向分组或纯yaw配额；服务器父类不同则需重核对。actor无height_scan仍可训练无感知rough，但没有前视地形高度输入。

feet_height=-0等于0，对该精确类名调用disable_zero_weight_rewards后禁用，不会与FHB叠加。illegal_contact=None意味着未配置机身碰撞立即终止；按本机父类只有超时和越界终止，undesired_contacts只是惩罚，不能把长episode解释为始终正常站立。现与旧run一致，单变量实验不额外改；长期通用方案另评估跌倒终止。FHB是否摆动专用由服务器rewards.py实现决定，仅附件权重不能证明已使用本机新函数。启动命令末尾的Hydra覆盖优先于源码，特别检查不要残留之前airtime=.1覆盖而误认为实际是1.0。

已确认/已完成：语法和静态配置复核、额外变更识别；无运行修复项。待确认：服务器父类/reward版本、入口最终有效配置、短运行和真实步态效果。

## 当前指令分布关系（2026-11-04源码核对）

已确认本机父类与Isaac UniformVelocityCommand：每次重采样分别均匀抽vx/vy/yaw，另抽heading；2%概率standing全部置零，heading=True且rel_heading_envs=1使其余yaw由clip(.5*wrap(heading_target-heading),-1,1)逐步更新，最初均匀yaw被覆盖。重采样间隔10s，yaw反馈不是保持10s不变。当前自定义类对平移向量模长≤.2清零，因此前后/横移/旋转不是互斥类别，通常是混合命令。

对最新附件vx±.8/vy±.6，理想连续均匀重采样下平移清零概率pi*.2²/(1.6*1.2)=6.545%；排除standing后约6.414%全部样本为平移零且非standing（可获得纯转向，但yaw可能因已对准朝向接近零）。正vx与负vx各约45.793%、standing2%，其余约6.414%平移零；vx正负样本通常仍有横移/转向，不能称纯前后各45.8%。这些是采样时的理论比例，非运行统计，不含额外服务器覆盖或地形特例。改范围会改变死区面积占比。

本轮仅函数/配置核对和解析推导，未修改代码/运行。服务器父类、实际分方向持续时间及指令分布直方图待运行核验；若要固定静止/前/后/横移/纯转向配额，仍需显式分组采样实现，关闭heading也不能自动产生配额。

## 用户已开训版本432f复核（2026-11-04）

附件432f1f80-971a-4a9e-9a55-8b0657e6d36e为本次依据，AST/compile通过，未导入模块或构造Isaac环境，[指纹记录](ppo_training_attachment_432f_2026-11-04.json)。用户表示已开始训练；本轮未启动/停止服务器任务，未修改配置。相对5800所属run：air1.8→1.0、baseheight-18→-10、angxy-.05→-.10，其余关键显式项FHB-4/-.20、linz-4、track1.5/.6、variance-8、vx±1/vy±.6/yaw±1对齐。是联合修改，不能仅归因airtime。

地形课程两条关闭语句已注释，若服务器继承本机同版父类则课程开启；未显式max_init_level，继承本机默认5，而旧run为1。范围修改未实现前/后/纯yaw分组，附件也没有关闭heading或覆盖standing；若父类同版本仍heading=True、standing2%、10秒重采样。需服务器生成的env.yaml确认。

baseheight-10减弱高度偏离惩罚，非增加高度稳定性，可能允许更低机身，但不能断言必然降低或抖动；angxy-.10已知部分方向收益但后退退步，组合效果未知。既然已开训，不因静态系数单独要求立即停止，保存为独立候选并固定参数，评测中途/末checkpoint的实际高度、角速度、后退净空与跟速，不能用总reward改善验收。若目标是严格airtime单变量则应新run恢复base-18/ang-.05，不在线修改并误以为运行配置随文件变化。

已完成静态配置复审，无运行修复项；待确认服务器父类/rewards、有效env/命令覆盖、实际是否分组及新策略效果。

## 地形种类与课程配置入口（2026-11-04）

已核对本机velocity_env_cfg.py从isaaclab.terrains.config.rough导入ROUGH_TERRAINS_CFG，并交给MySceneCfg.terrain.terrain_generator。实际定义位于IsaacLab-rsl5/source/isaaclab/isaaclab/terrains/config/rough.py：pyramid_stairs、pyramid_stairs_inv、boxes、random_rough各proportion .2，hf_pyramid_slope及其inv各.1，没有独立平地类别；平台区域或低难度坡不等于显式平地样本配额。rough_env_cfg.py44–59行覆盖boxes高度.025–.1m、noise .01–.06m、正反楼梯阶高.025–.08m。proportion分配地形类型，课程改变等级/难度，不替换sub_terrains。建议任务专用修改在rough_env_cfg中覆盖，不直接改公共Isaac库；只训练一种时显式删去其他dict条目，不靠只改一个比例假定其他消失。生成器开启课程后的行/列布局决定实际离散占比，不将名义比例当精确运行步数比例。

本轮仅核对源码，无配置修改/训练。服务器Isaac版本和最终env.yaml中的地形字典/比例仍待核对，不能由本机库推定所有机器一致。

## 宇树Go2公开PD核对（2026-11-04）

已查阅官方当前源码：unitree_rl_lab的assets/robots/unitree.py中UNITREE_GO2_CFG为stiffness25/damping.5，friction.01是另一项；unitree_rl_gym的go2_config.py为20/.5。来源分别为https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py 与 https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/go2/go2_config.py 。因此不能泛称宇树PD统一25/.1。本机imgo2_rl/assets/imgo2.py显式训练PD为25/.5，URDF转换joint_drive的0/0不是执行器最终PD。本轮仅源码核对未修改参数或运行；服务器训练时实际PD、执行器随机化与部署等效性需对应run材料核对，不因公开值自动调PD。

## 奖励量级定量核算（2026-11-04）

范围：用户问高度-18是否压过其他项，按5800所属run的5899末400轮已存贡献与当前本机函数/Isaac RewardManager核算，不冒充最新服务器-10/air1/ang-.1运行结果。最后432f附件高度实际-10，本机rough源码又为-16，三者不得混用。

已确认reward manager每步r_i=weight*raw_i*dt；dt=.005*4=.02s，reset日志Episode_Reward为回合加权累积除max_episode_length_s=20，不是直接每步reward或统一按实际存活时长平均。相同口径各项可比较累计量级，早终止/门控/事件频率仍会影响，不能把它作为优化梯度大小排名。计算文件[JSON](ppo_reward_scale_calculation_2026-11-04.json)，标准库执行得到。

直立且高度射线参考有效时，height贡献率=-18*(h-h_target)^2：高度误差1/2/3/5/10cm，乘dt前为-.0018/-.0072/-.0162/-.045/-.18，每步再乘.02。对应每步-.000036/-.000144/-.000324/-.0009/-.0036。平移跟踪为1.5*exp(-||v_cmd-v||²/.25)，零误差贡献率1.5，每步.03；误差.3m/s时1.046514，每步.0209303。低3cm只扣约完美平移跟踪的1.08%，低5cm约3%，因此-18并无绝对压制证据；降低到-10进一步放松高度约束，不能据此改善高度保持，也不表示增大一定好。射线基准跨阶均值问题仍在。

5899窗口正项合计1.764486，负项绝对值合计1.094790，高度-.0126518占负项1.1556%、占正项.7170%。髋L1-.227047、air-.192673、contactforce-.094202、linz-.093029、actionrate-.085325、airvariance-.077280、angxy-.071004、jointpos-.062376、mirror-.054324、flat-.040627、slide-.036650、undesired-.021750、height-.012652、jointacc-.011520、FHB-.008096、power-.002691、stand-.002576、torque-.000968；正lin1.269903/yaw.494583。没有单项负项绝对值超过正项合计，但这不是所有目标都合理或不会相互冲突的证明。

髋L1例子：四髋各偏默认.1rad、weight-1，原始合计.4，贡献率-.4；不是与height相同的米平方量纲。足高例子：两摆动脚各误差.05m、速度权重.76、weight-4，贡献率约-.0152，与机身低3cm的-.0162同量级；不据此断言固定策略降低身体最优，其他项/动力学也改变。air例子：单足air.15、threshold.5、weight1，落地事件每步贡献-.007（门控开启），非落地步0，不能将它视为每步持续惩罚。力矩例子：12关节各5Nm，-2.5e-6*sum(tau²)=-.00075贡献率，权重很小并不说明没作用。

建议不因-18数字大而继续削弱高度项或同时加大多项；最新候选既然已训练，先评测实际高度均值/波动、各方向跟速、每足周期/净空/动作饱和，再决定。姿态偏离多项叠加和FHB机身参考存在机制问题，但量级不能证明具体因果；不要据此直接清零髋约束。新run需要生成env.yaml、同窗口分项和物理评测才能完成其量级/效果验收。已完成源码/历史贡献核算与数值示例，未改权重/未新训或复测；新run结果与根因待验证。

## 用户78bf平地低速→扩速→rough方案复核（2026-11-04）

附件78bf76da-1f31-41e1-9bcf-cf5aaea9788b，AST/compile通过，未导入/构造环境；[指纹与静态重加权](ppo_staged_recipe_attachment_2026-11-04.json)。本轮不修改项目训练源码/配置，不启动训练或仿真。实际为平地generator=None/terrain课程None，heading=False、vx±.5/vy±.4/yaw±.6，速度课程关闭；没有实现分组。相对5800：高度-40、linz-10、angxy-.1、yaw1.2、air1.6、hipL1-.5、jointpos0。

joint_pos_penalty确实是全关节偏离默认姿态的L2范数，运动中可能压制较大步幅，因v≈步长*频率，存在小步快频的优化途径；但不是直接步频项，既有日志贡献-.0622不说明无效，也无消融证据证明是主因。静止条件还乘5，取消将一并移除这层站姿约束。附件仍有hipL1-.5和stand_still-.1，不是全部姿态约束清空；jointpos=0可作为候选，但不能承诺慢步或稳定切换。

对原5700–5800同轨迹静态换weight：height-.01260→-.02800、linz-.09329→-.23323、angxy-.07121→-.14242、hip-.22819→-.11410、jointpos-.06217→0、yaw+.49395→+.98789、air-.19248→-.17109。不是新低速平地训练预测，初始随机策略和分布已改变，不反推最终动作。高度-40并非数值上压制全局，用户优先高度时可作为候选；linz-10对必要上下运动也扣分，不把它视为rough普适默认。

建议保留用户低速平地/headingFalse/高度-40/air1.6/FHB-4/-.20/yaw1.2/hip-.5/jointpos0候选，但第一版linz先-4、angxy先-.05，减少之前-.1方向性退步的重复风险；这是一套未验证的候选而非最优参数。全程先固定核心奖励，不把高速结束后全面改reward当必需步骤。静止比例可覆盖rel_standing_envs=.10（独立standing分支，不是完整方向分组），仍需纯前后/横移/yaw分组实现；heading关闭只使yaw直接均匀采样。xy模长≤.2清零仍存在，最大合成平移sqrt(.5²+.4²)=.6403m/s，不是所有样本低于.5。

阶段建议：4000轮平地低速是预算非保证，只有前后±.3/横移/纯yaw和站立稳定才扩速；第二阶段把范围扩到vx±1/vy±.6/yaw±1，保留低速样本而非仅高速，追加1000是初看不是必然充分；第三阶段恢复generator和terrain课程、max_init_level1，保留基础平地任务，核心reward尽量不变。若在rough_env文件直接编辑，删除三条plane/None覆盖后父类才重新提供生成器，不是在generator=None之后给它访问curriculum；不要只把terrain_type改回generator。平地常驻40%与分组是建议实现，当前附件未有该功能。阶段切换不是抖动和快步的自动解法，不能用凑够4000/1000轮或总reward作验收。

站立抖动可能涉及目标动作/PD/切换及训练分布；角速度tracking奖励不约束roll/pitch角速度，yaw欠跟踪不说明roll/pitch惩罚必然不足。高度误差与高度速度也应分开测。服务器若命令仍带airtime.weight=1.0/.1，优先覆盖附件1.6，必须核对新run保存env.yaml。

已完成静态审查/原日志同轨迹重加权，无修复训练项；待确认服务器父类和reward版本、最终配置、冒烟、独立配方真实效果、分组/平地混合实现以及多seed/部署评测。

## 新平地训练每轮0.9秒的初步解释（2026-11-04，服务器未核验）

用户报告新训练iteration .9s，旧rough约3s。最新78bf附件确实切为plane，父类本机num_envs4096、runner每env24步、dt.005*decimation4=.02s。若服务器与此一致，每轮样本4096*24=98304，.9s对应109226.7环境步/秒，3s对应32768；每env推进.48秒仿真时间，与墙钟每轮耗时是不同量。平地相较复杂网格地形可能减少接触/碰撞/射线开销，因此加速合理，不能仅凭耗时短判断少算或物理频率变快，也不能无服务器数据断言已确认原因。关闭一个jointpos reward不能单独解释三倍加速。

已完成源码与附件静态核对、理论吞吐计算；未改配置/运行训练，缺服务器Computation(collection/learning)、实际启动命令、生成env/agent与硬件信息，已请求用户提供。应核对num_envs/num_steps_per_env、dt/decimation一致及日志无异常后再确认，勿以GPU利用率单瞬时或reward小推定故障。

## 0.9秒与新配方效果后续核验（2026-11-05）

已收到23-37-54新run，4096×24及collection/learning约.75–.78/.09秒，与用户.9秒一致；不再仅是静态推测，没有少采样证据。新4000/5900正式导出与MuJoCo/Isaac实跑显示身高收益但后退/横移贴地，具体结论统一见[新平地低速复审](../deployment/plane_low_review_2026-11-05.md)，本页阶段配方仍为历史建议，不作为已验证最佳参数。
