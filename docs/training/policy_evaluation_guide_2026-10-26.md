# 四足行走策略的评估与训练日志判读（2026-10-26）

## 1. 评估顺序

针对用户当前“平地正常行走、低速稳定、转向和减速少晃”的目标，依次看任务完成、运动质量、训练趋势、优化器健康。总奖励用于同一配置下判断趋势；更改奖励权重、速度分布、地形难度、回合时长或统计方式后，总奖励不再是直接可比的质量分数。loss没有变成0不代表策略没学会，loss接近0也不保证运动符合目标。

当前部署PPO已测基准见 [双引擎转向/减速记录](../deployment/ppo_maneuvers_2026-10-26.md)：.3m/s可走，.1m/s几乎停住，转向和急减速有姿态摆动。具体数值统一引用该记录，不在本文重复另一份结果。

## 2. 固定测试集

比较旧/新checkpoint时固定机器人模型、PD、动作尺度、观测处理、控制步长、地面/摩擦、初始状态和指令序列，使用确定性Actor输出并关闭动作探索噪声。先用干净观测和固定材料排查基本能力，再在独立测试中加入观测噪声、延迟、摩擦/质量变化等扰动，不能把“关掉所有扰动仍失败”和“扰动鲁棒性不足”混成同一结果。

建议平地测试：零速20秒；前进和后退的.1/.3/.5/.8m/s；左右横移；原地左右转±.5/±1rad/s；行走转弯；.8降到.3以及降到0的快速/缓慢指令。每项分开统计，先重复5…10个初始状态作为工程筛选，报告均值、最差窗口、失败次数和样本数，后续扩大到未训练过的起点/摩擦/扰动。上述是建议测试设计，本轮未全部执行，不代表现有脚本已覆盖横移和所有倒退速度。

rough训练应另按地形类型与难度分组，固定测试地形/指令；不能仅看所有环境混合后的平均分。坡面/楼梯姿态应结合地形和期望姿态，不能套用平地“pitch始终接近0”的验收规则。楼梯单独统计整机上下阶完成率、用时、跌倒/卡住/绕行，不把碰上第一阶算完成。

## 3. 运动质量要看什么

| 目标 | 建议指标 | 解释 |
|---|---|---|
| 不倒地 | 独立的倾倒、base碰撞、低机身高度检测；成功回合比例 | 检测不能只依赖当前训练done配置 |
| 听指令 | 指令与实际机身系速度曲线、MAE/RMSE；前进/后退/转向分组 | 平均vx吻合仍可能有过冲、横漂或摆动 |
| 站立稳定 | xy位移、非指令yaw漂移、roll/pitch标准差及角速度 | 零指令时关节角接近默认值不保证机身没有移动 |
| 机身少晃 | roll/pitch标准差、峰峰值、sqrt(mean(wx²+wy²)) | 转向时排除有意的yaw；统一统计窗口，避免spawn/起身污染 |
| 制动平稳 | 最大过冲、停车距离、姿态峰值、恢复时间 | 动态过渡与恒速段分开，急/缓变化成对比较 |
| 足端正常 | 接触期世界系足端切向速度、摆动期离地净空、绊脚次数 | 净空要相对脚下地形及足端几何定义；不是相对base的z |
| 执行器可实现 | 力矩/关节速度/动作饱和比例、关节越限、功率 | 使用实际执行力矩与物理限额，不用网络原始输出替代 |

公式：MAE(vx)=mean(abs(vx−vx_cmd))；RMSE(vx)=sqrt(mean((vx−vx_cmd)²))。xy联合误差应明确是否采用二维范数，不能和单轴误差混用。角度用度还是弧度、角速度rad/s、速度m/s都要标清。步频本身不是越低越好；结合速度、步长、净空、滑动与稳定性判断，单足滞空时间不等于整周期。

初步验收线可以按用户需求自定，例如平地.3m/s直行MAE≤.05m/s；在相同左转测试中使俯仰峰峰值较现基准降低30%，同时保持跟踪；规定减速后恢复到低速基线的时间。它们是建议工程目标，非行业统一标准或已验证达标。恢复时间需要事先定义窗口和容差，例如滚动角速度RMS回到基线±20%并持续1秒，再统计首次恢复时刻，不能凭一帧判断。

## 4. 当前PPO训练日志的读法

- `Mean reward`：同配方下观察长期趋势和突然坍塌，不用一个瞬时点选策略，不与修改权重后的绝对值直接排名。
- `Mean episode length`：当前回合20s、控制dt=.02s，满回合为1000控制步；越接近上限说明较少提前终止，但**当前rough将illegal_contact设None**，倒地也可能跑满回合，必须结合独立倾倒/碰撞统计。
- `Episode_Reward/track_lin_vel_xy_exp`、`track_ang_vel_z_exp`：当前标准Isaac RewardManager先累计term*weight*dt，再除最大回合时长输出。因此完整20s回合理论日志上限分别1.5和.6；每步上限分别.03/.012、完整回合累计30/12。不要把这些不同口径混在一起。提前结束会降低该分母下的可达分数，AMP自定义尺度亦须另核实。
- `Metrics/base_velocity/error_vel_xy`、`error_vel_yaw`：当前继承的UniformVelocityCommand每步将误差除以“最大指令持续时间/step_dt”后累加，CommandTerm在环境reset时求均值并清零。当前最大指令时间10s、满回合20s，完整回合的该累计约为时间平均误差的2倍；提前终止、回调/聚合方式改变会影响口径。用它看相同条件下的趋势，真实m/s或rad/s验收以逐帧原始速度计算的MAE/RMSE为准，不能直接将日志0.4解释成MAE=.4m/s。
- `Value function loss`：Critic预测回报的拟合损失，受奖励尺度与状态分布变化影响；没有跨任务通用的好坏阈值。持续爆炸/非有限值或同时出现表现坍塌才值得优先排查。
- `Surrogate loss`：PPO每批更新的代理目标，随重新采样和优势变化而变化，不要求像固定数据集监督学习一样单调降到0；近0既可能更新小，也可能已被限制，结合实际性能/KL判断。
- `Mean action noise std`或entropy：训练探索程度，不是部署确定性动作的实际抖动幅度；不是越小越好，亦不能由std大直接断言部署一定抖。
- `Curriculum/terrain_levels`：训练难度分布的线索，不是成功率；课程升级后reward变平甚至回落，需在固定难度评估策略才能分清进步与退步。

当前PPO配置desired_kl=.01、adaptive学习率；若记录了KL/学习率可辅助观察更新幅度，但没有要求KL必须每轮精确等于.01。不能套用其它框架的CLI或early-stop实现到本工程。

## 5. 是否继续训练与选择checkpoint

定期取早/中/晚几个checkpoint跑相同测试，形成“训练步数—任务成功率—速度误差—姿态/制动指标”的表。比较实际采样步数/数据量，不跨不同环境数/rollout长度只比iteration编号。优先保留满足基本不倒地与跟踪要求、且转向和过渡质量更好的checkpoint，最后一个不一定最好。

若固定评估仍持续改善，可继续训练；若多个相隔一定数据量的checkpoint都维持相同失败模式或关键指标无改善，应检查样本覆盖、奖励语义、观测/动作和动力学配置。是否平台期须结合评测波动范围和重复试验，不存在“训练多少轮就必然收敛”的通用数值；提高权重后reward变大不能代替质量改善的证据。课程变化时另保留固定评测。

## 6. 已核对、限制与资料

已核对：当前rough的20s/.02s、illegal_contact关闭、10s指令重采样与命令指标继承；本机Isaac RewardManager的dt结算/日志归一化，CommandTerm累计/清零，以及PPO配置desired_kl与网络设置。以现有实测作教学例子，没有新跑训练、测试电机或修改奖励。

待完成：完整平地测试集、独立跌倒监测、足端接触/地形净空/执行器饱和的系统验收；目前双引擎记录不能冒充覆盖以上所有项目。继续由README的PPO-GAIT-01/PPO-ROUGH-01跟踪，需补记录通道、固定场景及新checkpoint才能比较新训练效果。

参考：[Isaac奖励管理器源码文档](https://isaac-sim.github.io/IsaacLab/v2.0.2/_modules/isaaclab/managers/reward_manager.html)、[OpenAI PPO说明](https://spinningup.openai.com/en/latest/algorithms/ppo.html)。本机版本行为以上述实际源码为准；外部资料用于解释原理。

## 7. 服务器TensorBoard找不到pkg_resources

用户2026-10-26提供服务器isaaclab环境的报错：TensorBoard在default.py导入pkg_resources失败。该模块属于旧版setuptools，setuptools自82起移除，不能用pip install pkg_resources作为对应发行包安装。没有读取服务器当前setuptools版本，具体是未安装、损坏还是版本过新仍待服务器确认。

先Ctrl+C结束用户正在运行的pip安装，再在原isaaclab终端、原logs目录运行：

```bash
python -m pip install --force-reinstall "setuptools<81"
python -c "import pkg_resources; print('pkg_resources OK')"
python -m tensorboard.main --logdir=. --port=6006
```

使用当前python调用pip与TensorBoard，以保持解释器一致。这里保留兼容旧TensorBoard的setuptools版本；pkg_resources弃用提示可能仍出现，须与实际ImportError区分。TensorBoard默认本机监听，远程查看使用服务器平台端口代理或SSH端口转发；本轮未配置网络访问。

已完成：根据用户堆栈与 [setuptools官方迁移说明](https://setuptools.pypa.io/en/stable/deprecated/pkg_resources.html) 核对依赖来源/移除版本并提供命令。待确认：服务器导入验证和TensorBoard启动结果；没有访问服务器或更改本机Python环境，不将命令建议标为已经修复，README以TB-PKG-01跟踪。

### 同日后续：setuptools安装成功，但packaging版本冲突

用户日志确认setuptools-80.10.2已经安装成功；同时pip报告当前packaging26.0不符合isaacsim-core5.1.0.0的packaging==23.0、isaaclab2.3.2.post1的packaging<24。两个已报告要求可由packaging23.0同时满足；无需再安装setuptools。当前日志不能证明packaging26.0由前一条命令引入，也不能证明环境中没有其它冲突。

在同一服务器环境、logs目录继续：

```bash
python -m pip install "packaging==23.0"
python -m pip check
python -m tensorboard.main --logdir=. --port=6006
```

pip check若仍报告冲突，须按具体包约束继续核对，不能把仅满足上述两个约束等同于整个环境已经健康。已确认的是用户提供的setuptools安装结果与两个约束的版本交集；packaging调整、pkg_resources导入、TensorBoard启动和Isaac训练仍待服务器验证。本机环境未修改。

### 服务器镜像与网页访问的澄清

用户口述Sim2.3.2/Lab5.1.0；按其实际pip报错，应为Isaac Sim5.1.0、Isaac Lab2.3.2.post1。TensorBoard读取事件日志并提供网页，可用于服务器环境；镜像这两个版本不构成禁止使用的理由。当前已经观察到的是Python依赖冲突，修正后仍需实际启动验证，不能据此承诺其它依赖均兼容。

按 [TensorBoard官方说明](https://github.com/tensorflow/tensorboard/blob/master/README.md)，2.0起默认localhost监听；服务器启动后通过平台TensorBoard入口、端口代理或SSH转发查看。本地电脑浏览器直接访问localhost:6006指向本地电脑，不自动指向服务器。平台具体访问方式与TensorBoard最终启动结果仍待用户提供，未擅自开放公网监听或修改服务器网络配置。

### 同日后续：wheel与Ray要求新版packaging

用户pip check确认服务器现为packaging23.0，但wheel0.46.3要求>=24.0、Ray2.54.0要求>=24.2；与Isaac Sim固定23.0不存在交集。因此维持Sim要求，调整这两个包版本。已核对 [wheel0.45.1项目元数据](https://raw.githubusercontent.com/pypa/wheel/0.45.1/pyproject.toml) 没有外部packaging>=24依赖，及 [Ray2.52.1基础依赖](https://raw.githubusercontent.com/ray-project/ray/ray-2.52.1/python/setup.py) 的packaging不设该最低版本。Ray2.53.0仍要求>=24.2，不能只降到2.53来解决。

```bash
python -m pip install "packaging==23.0" "wheel==0.45.1" "ray==2.52.1"
python -m pip check
```

以上不使用--no-deps，让解析器检查本次安装的直接约束。已完成：用户日志确认packaging版本和剩余两项冲突，核对候选版本官方源码约束。待确认：服务器安装结果、完整环境pip check以及TensorBoard启动；本机没有安装/降级包，不把依赖声明允许等同于已完成服务器运行验证。

## 8. 用户TensorBoard四张Loss图的解释

用户截图已归档为 [原图](ppo_loss_screenshot_2026-10-26.png)。显示多个run；主要粉色曲线到约5329日志step，下方灰色run的Smoothed数值属于另一个运行，不能直接当作粉线当前值。当前工程runner将iteration写为标量step；若服务器runner/横轴设置有改动需按实际日志核对。浅色原始曲线与深色平滑曲线应一起看，不能仅凭平滑后不尖锐排除异常。

- `Loss/entropy`：当前核对的高斯策略实现对动作维度的分布熵求和，再在优化批次求均值；12维总熵达到十几是可能的，不是关节误差多少rad。粉线早期降到约11、后半段升到15…16，表示高斯动作分布的总熵增大（标准差乘积增大），不等同于部署确定性动作抖动必然增大，也不要求熵一直下降。可能与熵奖励、数据/课程变化和策略收益权衡有关，单张图不能确定原因。需看Policy/mean_std或mean_noise_std、reward、误差及课程，且用确定性play比较。
- `Loss/learning_rate`：优化器学习率，决定参数更新步幅；图中整体下降且波动，符合自适应调节的表现。当前源码PPO配置为adaptive、desired_kl=.01，核对的本机实现为KL高于2倍目标时学习率÷1.5、低于半目标且正值时×1.5；服务器agent.yaml/版本未获取，不能仅从截图确认其使用同一实现，亦不能认为下降就是训练已收敛。
- `Loss/surrogate`：Actor的PPO裁剪代理损失，负号来自将最大化代理收益转成最小化损失。图中约−.012到−.011的变化本身不能判定退步；批次优势、状态分布和旧策略都在更新，不要求曲线单调接近0或一直变得更负。
- `Loss/value_function`：Critic预测回报的拟合损失；启用clipped value loss时取原始/裁剪预测误差的较大者再平均。图中约.026降到.020后回到.022附近，截图可见范围内没有数量级爆炸，但小幅回升的原因和策略质量待reward/课程/原始数据确认，没有“低于.02即合格”的通用门槛。

常见PPO总损失为L_surrogate+c_value*L_value−c_entropy*H；learning_rate控制优化步幅，不作为该式的加和项。图中名称前缀Loss不意味着每一项都要最小化：正熵H在总损失中以负系数鼓励探索。当前工程配置c_value=1、c_entropy=.01，截图对应服务器实际值仍需其agent.yaml验证。

已完成：读取截图并核对本机RSL-RL PPO损失、KL自适应、logger与高斯熵实现，以及工程runner记录step。截图标签value_function与本机RSL-RL5使用的value命名存在差异，保留版本限制，不据此认定服务器算法装错。参考 [PPO原理](https://spinningup.openai.com/en/latest/algorithms/ppo.html)。待确认：服务器对应run的agent.yaml、原始event数据、reward/速度误差/课程和确定性play，才能评估后半程探索增大是否影响性能。本轮未改权重、噪声或运行训练。

用户已展示TensorBoard页面，可确认有可用的可视化页面；不等同于此前指定服务器环境pip check已通过，剩余依赖验证仍待回传。

## 9. PPO损失如何从奖励计算

本节核对本机RSL-RL5的algorithms/ppo.py及项目rl_lab/algorithms/ppo.py；核心策略/价值裁剪与总损失计算一致。当前工程base_move配置为gamma=.99、lambda=.95、clip_param=.2、value_loss_coef=1、entropy_coef=.01、use_clipped_value_loss=True。截图对应服务器的实际覆盖值仍须该run的agent.yaml确认。

数据流为：采样观测/动作/奖励/旧策略概率与旧价值 → 估计回报和优势 → 用新网络重新计算同一批状态/动作的概率与价值 → 计算损失、求梯度并由Adam更新。当前环境单步奖励为dt乘各任务分项加权和；PPO损失是在采样批次上的优化目标。

连续非终止片段内，TD误差delta_t=r_t+gamma*V_old(s_next)−V_old(s_t)，GAE原始优势A_raw_t=delta_t+gamma*lambda*A_raw_next，目标回报R_hat_t=A_raw_t+V_old(s_t)。Actor使用标准化后的优势，目标回报先由原始优势构造，不使用标准化优势构造Critic回归目标。实现对真正终止截断后续项，并对时间超时做bootstrap；rollout末端用最后状态的价值预测衔接，不能把24步采样等同只考虑未来24步。

策略比率rho_t=exp(log_pi_new(a_t|s_t)−log_pi_old(a_t|s_t))，连续动作这里是同一动作的联合概率密度比。L_surrogate=−mean(min(rho*A,clip(rho,.8,1.2)*A))，与源码max(−rho*A,−clip(rho)*A)等价。优势为正时提高该动作的概率密度、为负时降低；裁剪限制额外改善激励，不是对全部参数更新施加严格20%硬上限。

基础价值回归是mean((V_new−R_hat)^2)。当前启用的裁剪版本：V_clip=V_old+clip(V_new−V_old,−.2,.2)，L_value=mean(max((V_new−R_hat)^2,(V_clip−R_hat)^2))。示意：目标回报6、预测5，未裁剪单样本平方误差为1；实际日志再按采样/优化批次平均，不能用一个样本误差代替整轮日志。

对角高斯策略的H=sum_j(log(sigma_j)+.5*log(2*pi*e))，对动作维求和、对批次求均值。记录的entropy是H本身；进入总损失的是−.01*H。总损失L=L_surrogate+1*L_value−.01*H；learning_rate控制Adam更新步幅，不直接作为加和项。若三个分项恰取−.011、.022、15.7，算术示例总损失为−.146；该例不是原始日志同一批数据的重建，总损失为负也不代表错误。

奖励权重改变会改变回报/优势及价值回归尺度，因此value loss绝对值不能跨配方直接判断训练质量。loss趋稳与固定评测性能趋稳需要分别核对，见第4…5节。理论参考 [PPO文档](https://spinningup.openai.com/en/latest/algorithms/ppo.html)。

已完成：源码计算顺序、GAE/优势归一化、概率比率、value clipping、熵和总损失公式及数值示例核对。待确认：服务器对应实现/覆盖配置和原始mini-batch数据；没有改奖励/算法、启动训练或宣称已解释截图中每一项波动的具体原因。

## 10. 2026-10-27本机PPO播放命令的checkpoint文件名

用户命令指定 `imgo2_rl/logs/2026-10-26_17-09-42/8400.pt`；本机实际查到的是 `imgo2_rl/logs/2026-10-26_17-09-42/model_8400.pt`。ppo/play.py在显式checkpoint分支调用retrieve_file_path，文件名必须对应实际文件。已核对任务注册存在 `Imgo2-basemove-rough-ppo`，不擅自添加未注册的play任务后缀。

以下本机解释器及checkpoint已确认存在，工作目录为imgo2_rl：

```bash
cd /home/tommy0929/Documents/imgo2/imgo2_rl
export IMGO2_ISAACLAB_PYTHON=/home/tommy0929/miniconda3/envs/env_isaaclab/bin/python
export PYTHONPATH="$PWD/source/imgo2_rl:$PWD/scripts/rl_lab${PYTHONPATH:+:$PYTHONPATH}"

bash scripts/run_isaaclab.sh scripts/rl_lab/ppo/play.py \
  --task=Imgo2-basemove-rough-ppo \
  --checkpoint="/home/tommy0929/Documents/imgo2/imgo2_rl/logs/2026-10-26_17-09-42/model_8400.pt" \
  --num_envs=1 --device=cuda:0 --headless
```

实际输入多行命令要用行末反斜线，且--task和路径不能人为插入换行；终端自动折行不影响命令。headless用于无界面，若本地想看窗口则去掉；它不会修复文件名错误。

已完成：只读查找本机文件、检查解释器可执行及播放参数/注册；未启动Isaac、加载checkpoint或执行导出。待确认：用户未贴本次实际错误堆栈，文件名是明确差异，但不能据此排除解释器/观测维度/runner兼容等其它启动问题；需用户使用修正命令重试并提供剩余错误。训练入口前置沿用PPO-ROUGH-01。

同日后续已实际运行并修复旧checkpoint/RSL-RL5加载格式不兼容，model_8400的正式JIT/ONNX导出及export_only关闭通过，Actor数值/部署LibTorch/输入契约通过；没有替换部署或评测8400步态。以上“未运行”保留为最初命令核对的历史，当前状态见 [修复与验证](../deployment/ppo_8400_export_2026-10-27.md)。
