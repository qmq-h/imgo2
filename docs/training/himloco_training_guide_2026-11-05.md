# HIMLoco 结构、训练入口与关节复审（2026-11-05）

## 范围、状态与证据

本轮审阅当前工作树，不修改训练/部署源码、配置或权重，不启动训练/仿真。工作树已有大量用户修改，本文依据当前文件而非声称复现历史版本。已修复：无新增生产修复；已完成结构说明、资源检查和已知缺陷复现。待修复/待确认：训练估计器target、历史初始化/回合清理、训练部署物理契约和旧策略来源，详见下文。此前接口诊断的完整根因和两种适配路径只在[接口复审](../deployment/himloco_training_contract_2026-11-03.md)展开；本文补训练使用说明。

解释器与验证：`/home/tommy0929/miniconda3/envs/env_isaaclab/bin/python imgo2_rl/scripts/tools/check_himloco_training_contract.py` 退出0，复现3:48错误切片、done保留旧历史、reset返回45维且缓存未重置。这是缺陷复现成功，不是修复通过。`python3 imgo2_rl/scripts/tools/check_asset_paths.py` 退出0，当前默认imgo2_real.urdf、17条网格引用与资源路径有效。未运行Isaac环境构造、短训、导出或新sim2sim。现有部署HIM SHA256为`d9945f85a39944aaf241a67f7d2b25a15326a0222d0e503aeb05a7f188430509`，与历史占位标记一致；缺原checkpoint/env.yaml/agent.yaml，不能确定旧权重训练接口。

## 结构与数据

这是本仓库的HIM实现：PPO学习运动控制，同时训练一个历史估计器。没有专家动作模仿损失，无需额外采集动作数据、AMP运动文件或预训练策略。质量/接触/地形变化通过仿真产生；训练时真实线速度与地形扫描是特权信息，不意味着部署需要速度真值或高度传感器。

代码链路（以下相对仓库根）：

| 层 | 文件 | 作用 |
|---|---|---|
| 注册 | `imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/__init__.py` | rough-himloco和rough-himloco-play任务 |
| 机器人 | `imgo2_rl/source/imgo2_rl/imgo2_rl/assets/imgo2.py` | 当前模型imgo2_real.urdf、默认角、执行器 |
| 公共环境 | `imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py` | 场景/地形/动作/随机化/奖励基础定义 |
| HIM环境 | 同级`base_move/himloco_env_cfg.py` | HIM观测顺序、关节列表、奖励及覆盖配置 |
| 超参数 | 同级`base_move/agents/himloco_rsl_rl_cfg.py` | rollout/PPO/网络/历史长度 |
| 启动 | `imgo2_rl/scripts/rl_lab/himloco/train.py` | 建环境、包装、保存配置、训练 |
| 环境接口 | `imgo2_rl/scripts/rl_lab/rl_lab/envs/him_manager_based_rl_env.py` | 额外保留重置前观测 |
| 历史包装 | `imgo2_rl/scripts/rl_lab/rl_lab/wrapper/himloco_vec_env_wrapper.py` | 帧堆叠、终止信息 |
| 学习核心 | `imgo2_rl/scripts/rl_lab/rl_lab/`下`runners/him_on_policy_runner.py`、`algorithms/him_ppo.py`、`storage/him_rollout_storage.py` | 采样、优势估计、PPO更新 |
| 网络 | 上述目录下`modules/him_actor_critic.py`、`modules/him_estimator.py` | Actor/Critic/历史估计器 |
| 回放导出 | `imgo2_rl/scripts/rl_lab/himloco/play.py` | 加载checkpoint，正式导出组合JIT及ONNX |

单帧45维 = commands3 + angular velocity3 + gravity3 + relative q12 + dq12 + last action12。历史5表示过去5帧加当前1帧，最新在前，共270维。50Hz下6帧首尾相距0.10秒。Estimator用270维估计3维速度和16维归一化latent；Actor输入当前45+3+16=64，输出12个关节位置偏移。Critic用额外线速度/地形信息训练价值函数，现有记录235维，实际以运行时观测管理器为准。估计器以仿真速度监督，并用下一帧target、32个prototype与Sinkhorn分配做表示学习；当前target取错字段，需先修。部署仅需组合Estimator+Actor。

## 当前默认设置（不是已验收配方）

- 并行4096环境；物理dt=.005秒、decimation=4、策略50Hz；回合20秒。
- 每环境采样100步后更新，即2秒仿真；4096环境合计409600条transition/iteration。默认2000轮不是2000个物理步，也不保证收敛或目标步态。
- Actor/Critic隐藏层512/256/128，ELU，初始探索std=1；history_length=5。
- PPO学习率.001、自适应KL、目标KL=.01；5 epochs、4 mini-batches、clip=.2、gamma=.99、lambda=.95、entropy=.01。
- 默认角各腿0/.87/-1.82 rad，名义Kp/Kd=25/.5，有增益等随机化；力矩23.7Nm。
- 动作目标是默认角+scale×动作，hip scale=.125，其余=.25；训练配置另在位置目标层clip±3rad。
- 粗糙地形生成器，HIM覆盖boxes高度.025–.10m、random rough噪声.01–.06m；不是平地专用任务。训练地形课程需以实际保存配置确认。
- vx/vy范围±1m/s，wz上限±1.5rad/s；继承heading_command=True，角速度实际受航向误差控制，并非简单始终独立均匀采样。指令10秒重采，2%站立；线/角速度课程在HIM配置中关闭。
- 跟踪奖励xy1.5/yaw.6、高度-10(target .30m)、airtime1(threshold .5秒)、时长方差-8、滑移-.05、机身系足高-5(target -.20m)、mirror-1(包含hip)、髋偏差L1关闭。还有扭矩/功率/水平姿态等项，不能把这里当完整奖励清单。
- `feet_height_body`现在调用共享的swing-only版本，-.20是机身坐标高度，不是20cm离地净空；当前重训不等于历史旧函数配方。mirror对角同角度包含hip，左右符号不适合盲目强约束，应单独评估；它不是关节轴被改错的证据。
- 推机器人和外力事件被HIM覆盖关闭；illegal_contact终止也关闭，满20秒不能证明没有跌倒。

## sim2sim关节核对

从C++读写路径核对，mapping[i]表示策略第i维对应物理槽位，反馈也按同一映射读回；ROS2控制器按模型顺序FL/FR/RL/RR启动，MuJoCo也是该物理腿顺序。每腿hip/thigh/shank。

| 动作索引 | 当前训练含义 | 当前HIM yaml映射到物理关节 |
|---|---|---|
| 0–2 | FL | FR |
| 3–5 | FR | FL |
| 6–8 | RL | RR |
| 9–11 | RR | RL |

所以当前训练导出策略若直接配现有HIM yaml，观测和动作均交换左右。该映射本身没有负号变换，不能用改URDF关节轴/限位来修。对于当前FL/FR/RL/RR新策略，仿真侧目标mapping应为0..11；这不等于断言来源未知的旧himloco.pt该立刻改恒等。

同时默认角、Kp/Kd、commands_scale、动作clip和力矩配置不一致，不能只修mapping就宣称sim2sim完成。完整差异、物理默认姿态及修法见[接口复审](../deployment/himloco_training_contract_2026-11-03.md)。该文还说明critic的q/dq内部顺序和dq缩放与actor不同，不能只把estimator切片改成:45就认为target一致。

建议新训练保留当前训练的FL/FR/RL/RR与模型轴线，只对HIM独立部署yaml按新训练契约对齐；PPO/AMP及公共资产不动。另一条路是HIM训练独立适配现有部署yaml，见接口复审，两条路线不能混用。本轮仅提出建议，未实施。

## 准备与命令

先修target/critic语义与历史初始化、done/reset；对齐HIM训练部署契约后做短训。以下从仓库根进入imgo2_rl，`python`必须来自已能启动Isaac Lab的环境，不能随意使用系统Python。安装与脚本入口均已在仓库确认；命令是使用说明，本轮未执行安装/训练。

```bash
cd imgo2_rl
python -m pip install -e source/imgo2_rl
python -m pip install -e scripts/rl_lab
python scripts/tools/list_envs.py

# 修复后先做短训，只验证训练管线，不验收步态
python scripts/rl_lab/himloco/train.py --task=Imgo2-basemove-rough-himloco --num_envs=64 --max_iterations=10 --headless

# 短训通过后新建正式run；环境数可按显存调整
python scripts/rl_lab/himloco/train.py --task=Imgo2-basemove-rough-himloco --num_envs=4096 --max_iterations=2000 --headless
```

日志在运行目录下`logs/himloco_rsl_rl/base_move_himloco/<run>/`，保存params/env.yaml、agent.yaml和model_*.pt。应保存完整run及代码版本；只有JIT不能常规恢复训练。修改接口后新建run，不从来源不明旧策略续训。

回放/正式导出示例（把占位路径换成实际存在checkpoint）：

```bash
python scripts/rl_lab/himloco/play.py --task=Imgo2-basemove-rough-himloco-play --num_envs=1 --headless --checkpoint=/absolute/path/to/model_N.pt
```

play会导出checkpoint同级exported/policy.pt（Estimator+Actor）、encoder.onnx与policy.onnx。play环境默认固定前进1m/s且开粗糙地形，这不是站立/前后左右/转向完整验收；headless回放持续运行，完成导出或观察后可结束进程。当前部署yaml读himloco.pt而非policy.pt，接新策略时必须同步HIM独立yaml的model_name和文件名，避免仍加载旧权重。不能只拷文件就认为契约对齐。

后续验收：先对同一确定性状态逐项比较270维输入与12维物理目标角；导出组合网络与原checkpoint推理对比、部署LibTorch加载；然后Isaac与MuJoCo/Gazebo固定命令对照，测速度、姿态高度、关节饱和与接触。缺这些结果，状态仍是待修/待验证。旧失败归因还缺其原checkpoint、params/env.yaml、params/agent.yaml及对应导出文件，或至少确认用户指的是当前d9945f85权重。

## 同日补充：调整文件与配置修复边界

用户追问“是否只剩关节顺序，PD在sim2sim config改即可”。澄清：不是只有顺序。训练腿顺序由himloco_env_cfg.py的joint_names及policy q/dq、action引用控制；奖励权重/目标值在同文件Rewards块调整，数学实现来自共享mdp/rewards.py。sim2sim独立himloco/config.yaml可对齐joint_mapping、rl_kp/rl_kd、default_dof_pos、commands_scale、action_scale及限幅，但估计器target和历史缓存必须修Python实现；critic语义也需训练侧对齐。奖励参数不进入部署yaml，修改奖励仅影响后续训练。

对于按当前训练契约新训练的策略，可将HIM仿真yaml对齐为恒等mapping、名义PD25/.5、各腿默认角0/.87/-1.82、指令缩放[1,1,1]；训练目标角clip与部署原始action clip不在同一处理阶段，不能直接把两处数值写成相同就认为等价。该建议不适用于来源未知旧权重的直接修复；未实施上述配置或代码变更。已修复：无新增生产修复；待修复与验证仍为本记录所列target、历史、完整契约及闭环。

## 同日补充：target逐段解释、完整问题边界与课程

### target布局及修复边界

以下切片均为Python左闭右开，单帧45维：

| 原始列 | Policy | Critic前缀 |
|---|---|---|
| 0:3 | 速度指令 | 速度指令 |
| 3:6 | 角速度×.25 | 角速度×.25 |
| 6:9 | 重力投影 | 重力投影 |
| 9:21 | 相对关节角，显式FL/FR/RL/RR逐腿 | 相对关节角，.*按资产内部顺序 |
| 21:33 | 关节速度×.05，显式逐腿 | 关节速度×1，.*内部顺序 |
| 33:45 | 上一次动作 | 上一次动作 |
| 45:48 | 无 | 机身线速度×2 |
| 48: | 无 | 高度扫描 |

因此当前estimator的critic[:,3:48]实得“角速度、重力、关节角、关节速度、完整12维上一动作、机身线速度”，共45维。不是丢3维动作，而是丢3维指令、追加3维线速度。若目标约定是下一帧policy，字段不匹配；如果刻意选择另一种特权target，表示学习在数学上仍可执行，因此不能单凭切片判定必然无法训练或已经证明旧策略失败根因。当前代码没有显式说明该特权target契约，需明确设计并验证。

Critic作为价值网络输入，本来不必与policy采用同顺序或缩放；问题只在于把它的切片当成policy等价向量。将切片改成:45以后，字段大类才相同，q/dq顺序及dq的20倍尺度差依然存在。要按下一帧policy约定修：优先显式传入next_policy_obs（终止样本用重置前policy），沿env/wrapper/runner/storage/algorithm送入estimator；或统一critic前45维语义后再取:45，并明确独立噪声是否保留。不能只改him_estimator.py一行就宣布完成。

第82行的critic[:,45:48]当前确实取到base_lin_vel，没有同样的字段偏移；它监督的是缩放后且带噪声的2v，不是物理原值v。这可以是自洽训练约定，部署Actor直接接收同一估计器输出，不应额外随意乘除2。是否改成无噪声v标签应另行决定，修改后要重训。

### 本轮检查范围内的问题与选择

1. 下一帧target契约：已确认与下一帧policy不等价，需明确目标设计并修接口或记录其特权target设计；不是已证实旧权重失败的唯一根因。
2. 历史边界实现缺陷：wrapper初始化调用底层reset却保留全零policy/critic缓存；runner从缓存取首拍，因此有效首帧丢失。done后新回合观测前拼上旧回合历史；显式reset返回45维且不维护270维缓存。应初始化当前帧+过去零帧，done只清对应环境的旧历史，显式reset返回同一堆叠接口，终止前target单独保留。
3. 训练与sim2sim接口不一致：mapping、默认角、PD、commands_scale、clip阶段/力矩参数，详见前文。这影响部署对齐，不能与训练Python缺陷混称“腿顺序错”。
4. 本次新增发现的日志实现问题：him_ppo.py:update计算mean_estimation_loss/mean_swap_loss，却在返回行192给出最后一个minibatch的estimation_loss/swap_loss。因此对应TensorBoard曲线不是一轮均值。已用标准库AST核对实际返回变量，未修改；这影响读图，不是已证明的运动失败原因。
5. 配方选择/待验证：illegal_contact=None保留坏姿态回合；mirror包含hip同角度约束而hip L1为0；足高权重-5与高度-10组合、足高共享函数已更新；指令范围较宽且无速度课程。这些不等于实现错误，也不保证必须改，需固定条件消融或短训。随机推扰和外力关闭是当前选择，不是遗漏课程实现的证据。

已修复：无新增生产修复；以上未改实现、未跑新训练/闭环。HIM-TRAIN-01仍待修，日志返回问题新增待修HIM-LOG-01。只靠部署yaml不能解决1/2/4。

### 当前是否只有地形课程

是，限定当前默认训练任务Imgo2-basemove-rough-himloco（外部Hydra覆盖或历史run以保存env.yaml为准）。公共CurriculumCfg定义terrain_levels、command_levels_lin_vel、command_levels_ang_vel，但HIM __post_init__把后两项设为None，保留terrain_levels。公共环境据此打开terrain_generator.curriculum。

地形课程按走出出生点的距离升降级：超过地块长度一半升级；达不到“当前平移指令模长×最大回合时长×.5”降级，升级优先。初始化max_init_terrain_level=5，非所有环境都从最简单地形开始，也没有先纯平地再自动切rough的独立阶段。当前解释器importlib.find_spec确认依赖来自本机IsaacLab-rsl5；其默认rough生成器10行，初始上限5不等于最终最多5级。这只是基于源码的配置/机制核对，没有运行课程变化。

线速度范围从开始就配置为vx/vy±1，wz范围±1.5；heading_command=True且rel_heading_envs=1，因此实际wz由航向误差控制并受范围约束；小平移指令有阈值清零，pits地形若存在还有前进限定。不能把固定配置范围理解成每拍所有速度独立均匀采样。

质量、摩擦、质心、增益随机化和观测噪声仍在；它们不是逐步增加难度的课程。当前没有已配置的奖励权重课程、随机化幅度课程或噪声幅度课程。速度课程关闭本身不是bug；是否开启，要先修确定接口/历史问题，再依据实际训练指标决定。
