# Imgo2 项目说明与维护记录

> **最后核对：2026-11-24。** Git 同步操作手册和当前研究摘要，实验明细与图表留在本机。9–10 月阶段记录见[项目历史总结](docs/maintenance/project_history_2026-09_to_2026-10.md)，实机约 7 Hz 横滚见[当前排查](docs/deployment/ppo_vibration_status_2026-11-22.md)。

> `docs/xiaoji_cmd/中期/` 整体留存本机。本文指向该目录的链接需要在本机打开，跨机使用时单独复制该目录。

| 当前工作 | 结论和入口 |
|---|---|
| 答辩素材 | [仿真素材总览](docs/xiaoji_cmd/中期/仿真/仿真素材总览.html)集中预览六张图和十六段视频。视频留在本机，跨机使用时整份复制仿真目录。原始数据见[素材采集记录](docs/research/defense_media_2026-11-22.md)和 [PPO 8000 轮数据](docs/research/defense_ppo8000_data_2026-11-23.md)。 |
| 中期答辩 | 采用 12 页、约 12 分 25 秒的汇报安排，含一页运动算法与控制逻辑，重点为仿真筛选、实物验证和电机控制改进。 [逐页安排](docs/xiaoji_cmd/中期/中期答辩分页.md) · [PPO 训练原理](docs/xiaoji_cmd/中期/仿真/PPO训练原理.md) · [仿真素材](docs/xiaoji_cmd/中期/仿真/仿真图表说明.md) · [实物实验](docs/xiaoji_cmd/中期/实物实验/实物实验说明.md) |
| 新模型 | 新髋、大腿和碰撞位置已同步到对应仿真资产，并完成新几何平地与粗糙地形回放。服务器 URDF 内容指纹和实机效果待核对。 [模型同步](docs/deployment/model_sync_2026-11-22.md) |
| 实机横滚 | 11/23 同一 8000 策略使用新 PD 后，实机约 −0.5 m/s 后退的横滚角速度 RMS 降低约 80%，跨工况复测待完成。 [三组实机与答辩图](docs/xiaoji_cmd/中期/实物实验/实物实验说明.md) |
| PPO 配方 | 11/22 Rough 从零完成 10000 轮。平地前进优先保留 5000，粗糙地形跟速优先保留 6000。 [配置与回放摘要](docs/training/pasted_rough_config_review_2026-11-22.md#当天回传的训练结果) |
| 操作命令 | [训练与回放](#4-训练与回放流程) · [仿真和实机部署](#6-部署流程) · [服务器命令](docs/xiaoji_cmd/) · [未关闭问题](#7-待办与已知问题) |

> 实机部署目录已恢复到提交 `81fa54a`。本文的俯仰和位控说明包含历史开发记录，运行时以该分支实际包含的入口为准。

## 快速理解与运行入口

**先读 [整体架构、主函数与移植指南](docs/maintenance/project_architecture_2026-10-12.md)**：目录分层、每个主入口做什么、反馈到策略到电机的数据流、训练／仿真／硬件运行命令，以及真机移植需要替换的接口。

```text
imgo2_description  模型唯一源 ──→ imgo2_rl 训练／回放／正式导出
imgo2_dataset     参考动作 ────→ AMP训练             │
                                                   ↓ policy.pt
imgo2_deploy/ros2_ws/policy + C++ FSM／推理 ──→ Gazebo / MuJoCo
                                   └─→ ROS包rl_real_imgo2（保留宇树参考后端）
imgo2_deploy/real/policy → SDK/FSM → 独立rl_real_imgo2 → 串口电机／IMU
```

| 要做什么 | 主入口／运行方式 | 当前边界 |
|---|---|---|
| 训练、回放、导出 | `imgo2_rl/scripts/rl_lab/<算法>/train.py`、`play.py`，用 `scripts/run_isaaclab.sh` 启动 | 需要匹配的 Isaac Lab 环境、任务和 checkpoint |
| Gazebo部署 | `src/rl_sim.cpp` → `ros2 run imgo2_deploy rl_sim` | 先启动 Gazebo 和控制器；不是直接读取串口 |
| MuJoCo部署 | `src/rl_sim_mujoco.cpp` → `rl_sim_mujoco imgo2 scene` | 独立CMake构建，需要模拟器依赖 |
| 电机／IMU调试 | `go_imu` 五个main → FSM/硬件库；`apps`只提供公共辅助 | 用户确认udev/monitor；重构后控制实机待验证 |
| 独立真机PPO | `real/rl_real_imgo2.cpp`，可选CMake RL目标；详见 [部署指南](docs/deployment/go_imu_sim2real_2026-10-15.md) | 30/30离线通过；映射默认锁定，AMP占位；实机未验收 |
| 自己的真机策略 | `src/rl_real_imgo2.cpp` 待替换硬件后端 | 不能直接运行到Imgo2；需接 `go_imu::hardware` |

上表的 `src` 指 `imgo2_deploy/ros2_ws/src/imgo2_deploy/src`。外层 `imgo2_deploy` 是部署总目录：`real`是独立真机工程，`ros2_ws`是仿真工作区，其中`src/imgo2_deploy`是ROS包。具体命令及工作目录见架构指南，硬件步骤见 [go_imu 手册](docs/deployment/go_imu_usage_2026-10-11.md)。

## 1. 项目总览

Imgo2 是面向四足机器人的强化学习运动控制工作区，包含机器人模型、Isaac Lab 训练环境、参考动作数据和 C++ 部署框架。

当前代码链路为：

```text
URDF 与网格 ──→ Isaac Lab 环境 ──→ PPO / HIM-Loco / AMP 训练
参考动作数据 ──────────────────→ AMP 数据加载与参考状态初始化
训练 checkpoint ──→ 策略回放与导出 ──→ 部署配置对齐
                                      └─→ MuJoCo / Gazebo / 真机入口
```

当前研究重点是 **Imgo2 本体运动训练与实物验证**：在 Isaac Lab 中训练并优化速度跟踪、足端动作、步态节律、机身稳定性和地形适应，正式导出后开展 MuJoCo／Gazebo 及实物对照，并利用实机反馈改进模型与训练条件。中期研究过程见[总结](docs/research/midterm_research_summary_2026-11-22.md)。拖曳为单独的扩展分支，其历史方案见[上层拖曳 RL 计划](docs/research/paper_plan_rl.md)。

最后一段部署链路仍需验证。`himloco` 仍是 Go2 参考占位策略；AMP 已于 2026-09-17 换成
Imgo2 自己的 checkpoint，**2026-09-18 已确认**：45 维 actor 正式导出并接入部署，Gazebo 侧
0.2–1.5 m/s 跟速 93.4–99.8%、站立漂移 0.001 m/s、FL-FR 相位 174°–188°（trot）；
训练侧 Isaac Lab 足端回放独立复现了「干净步态」并给出剩余差距的**根因是步幅偏小**
（每步只有参考的 0.61 倍，于是用 3.1 倍步频凑速度），见
[足端步态回放记录](docs/amp/gait_eval_amp_24500_2026-09-18.md)。
因此**已可以认为 AMP 完成了「训练→导出→sim2sim 行走」的闭环**；仍未闭环的是真机，
以及「步态形态对齐录制参考」这一档目标（AMP-06）。

### 目录职责

| 路径 | 内容与用途 |
|---|---|
| ~~imgo2_model/~~ | **2026-09-17 已删除**（git 可追溯）。原先只有腿部件片段 URDF（无 `base`、无 `<robot>`，XML 不能独立解析）+ 两套模型网格，无任何消费者；其网格与 `imgo2_description/meshes` 内容相同。删除理由见维护记录与 MODEL-02 |
| [imgo2_description/](imgo2_description) | **模型唯一源（2026-09-17 起）**。`xacro/core.xacro` 是物理内核（17 link/16 joint，命名 FL/FR/RL/RR），`xacro/robot.xacro` 是组装入口，按开关拼上 `transmission.xacro`（仅 ros_control）、`gazebo.xacro`（插件 + per-link 接触参数）、`imu.xacro`（imu_link）。生成物入库、勿手改：`urdf/imgo2.urdf`（纯）、`urdf/imgo2.gazebo.urdf`（+三者）；`mjcf/{imgo2.xml,scene.xml}` 是 `rl_sim_mujoco` 读的 MuJoCo 模型（训练物理 + 参考项目求解器/接触块 + `framelinvel`）。`meshes/` 是全仓唯一一套网格（指纹 `8dc5b5995a11`） |
| [imgo2_rl/](imgo2_rl) | Isaac Lab 扩展、任务配置、训练脚本、自定义算法包。**模型与网格副本已于 2026-09-17 删除**，训练直接读 `imgo2_description/urdf/imgo2_real.urdf`（见 `assets/imgo2.py`） |
| [imgo2_dataset/](imgo2_dataset) | 参考动作数据；当前数据位于 `datasets/imgo2_motion/` |
| [imgo2_deploy/](imgo2_deploy) | C++ 推理、观察缓存、状态机、MuJoCo/Gazebo/真机入口及策略配置。**模型副本 `robot_description/` 已于 2026-09-17 删除**，MuJoCo 读 `imgo2_description/mjcf/`、Gazebo 读 `imgo2_description/urdf/imgo2.gazebo.urdf` |
| [docs/](docs) | 排查记录与离线结果，直接按下方现存主题目录归档；根目录不放散文件 |
| [docs/towing/data/](docs/towing/data) | **归档的测试数据集**（跑完的 run 里可入库的部分：报告 + 逐 case 指标 + 统计；原始逐物理步轨迹仍在 `imgo2_rl/logs/`，不入库）。当前一份：[800 环境必要性网格](docs/towing/data/2026-10-09_necessity_800/overview.md) |
| `paper_plan_imgo2.md`（本地，Git 忽略） | 用户研究草稿，不随仓库同步；已实现的拖曳物理与验证结论见 [拖曳仿真验证](docs/towing/towing_simulation_validation.md) |
| 研究探索计划（原research_exploration_plan.md缺失，待确认） | 本体感知、执行器随机化、动作跟踪与真机准备的研究规划；不要将缺失文档当成已完成方案 |
| `imgo2_deploy/real/` | 独立C++工具：`Motor`通信与安全、`serialport`串口、`Imu`采集、`hardware`整机接口、`fsm/states`部署状态、`function/my_myth`数学、`display`面板。硬件monitor/demo/stand及可选RL的rl_real_imgo2/fsm_check均直接写main；`apps`仅信号/输入/CLI辅助，`config`用MotorConfig/ImuConfig构造默认参数。无ROS依赖；RL入口含上传的异步诊断日志，独立stand/demo不写文件日志；RL目标依赖LibTorch/yaml-cpp；原go/imu_sensor本轮开始已不在当前工作区，未恢复或改动 |
| [imgo2_rl/rlfromgym2lab.md](imgo2_rl/rlfromgym2lab.md) | IsaacGym AMP 迁移到 IsaacLab 时 env/wrapper 对接层的说明（背景资料，非当前结论） |

研究规划中的日期、目标和预期结果属于历史计划，不能直接视为当前进展或实验结论。

### docs 分类

| 目录 | 内容 |
|---|---|
| [docs/amp/](docs/amp/) | AMP 对齐、步态、实验与离线数据 |
| [docs/cmoe/](docs/cmoe/) | CMoE 网络、训练、地形与部署专属记录 |
| [docs/towing/](docs/towing/) | 机器人拖车、连接模型、地形、估计器与测试数据 |
| [docs/deployment/](docs/deployment/) | 通用部署、Gazebo、MuJoCo 与 sim2sim |
| [docs/training/](docs/training/) | 训练环境、启动器与依赖兼容性 |
| [docs/research/](docs/research/) | 研究计划、论文与参考实现 |
| [docs/maintenance/](docs/maintenance/) | 架构、代码复审、模型离线核对、Git 同步与文档维护 |
| [docs/xiaoji_cmd/](docs/xiaoji_cmd/) | 用户维护的服务器训练、策略导出与ROS2运行命令，以及PPO奖励/TensorBoard操作说明 |

历史阶段记录可汇总，移除细项前先核对实际引用并保留可恢复的原文。除非用户另有要求，新增文档直接归入现存子目录，不在 docs 根目录放散文件、不新建目录。机器可读结果优先放主题内已有 `data/`，没有则放主题目录；不另建子目录 README。

## 2. 当前状态

| 模块 | 已核对与待验证事项 |
|---|---|
| 模型 | 统一模型根为 `imgo2_description/`。新几何已由本机 Isaac Lab 导入，MuJoCo 静态和加载通过。Gazebo 启动路径已核对，新几何闭环与服务器短训练待实跑。 [模型同步](docs/deployment/model_sync_2026-11-22.md) |
| PPO | 训练、导出、Gazebo/MuJoCo/Isaac 回放和实机走动已有可用记录。无托举同速止抖、绝对零位、受载执行器与接触贡献待测。 [实机横滚](docs/deployment/ppo_vibration_status_2026-11-22.md) |
| AMP | 参考动作和模型关节映射已有 FK 核对。45 维策略完成正式导出与仿真行走，跨引擎足端和实机仍按问题表跟进。 [双引擎对照](docs/deployment/amp_engine_comparison_2026-10-14.md) |
| HIM-Loco | 训练、回放和导出入口已接入。部署中 Go2 参考占位与自训策略的配置应核对，独立走动验收待完成。 [训练指南](docs/training/himloco_training_guide_2026-11-05.md) |
| 拖曳 | Isaac Sim 实验覆盖弹性绳、低弹性绳与被动刚体连杆，含 800 环境对照。MuJoCo 已有两种绳场景加载验证，策略跨引擎的带载和停车效果待验证。 [验证与实验入口](docs/towing/towing_simulation_validation.md) |
| 独立真机 | `imgo2_deploy/real` 有独立硬件、诊断和 PPO 入口。协议和 FSM 有离线回归，新的诊断与止抖改动需逐项实机验收。 [真机流程](docs/deployment/go_imu_control_flow_2026-10-15.md) |

## 3. 代码阅读导航

首次学习 Isaac Lab：先读 [仿真最小结构与运行链路](docs/training/isaaclab_basics_2026-09-24.md)，区分模型资源、资产配置、环境和算法。

理解 `imgo2_rl` 的目录、配置继承、采样更新循环和启动步骤：先读 [RL 工程阅读与运行指南](docs/training/rl_structure_guide_2026-10-10.md)。
常用 AMP 导出、Gazebo、从头训练、续训和 Isaac Sim 回放命令见用户维护的 [command.md](command.md)；使用前替换跨机器路径，命令核对范围见 [填写记录](docs/maintenance/command_guide_2026-10-11.md)。
服务器安装、从头训练与续训练统一见 [服务器训练.md](docs/xiaoji_cmd/服务器训练.md)：系统Python已装Sim5.1.0/Lab2.3.2与空白环境新建Conda两种路线，项目固定/root/Desktop/imgo2；AMP算法在工程内，不要求RSL-RL5，但注册/导出仍需要外部配套包装依赖。

PPO当前奖励公式、TensorBoard全部标量、权重与action_scale及训练趋势判断见 [PPO奖励与TensorBoard完整说明](docs/xiaoji_cmd/PPO奖励与TensorBoard完整说明_2026-10-27.md)。

AMP 与 PPO 的理论关系、判别器损失和混合奖励、两套 AMP 配方及源码入口见 [理论与实现对照](docs/amp/amp_vs_ppo_theory_2026-11-22.md)。

机器人强化学习的研究方向、经典与 2026 年论文、十周学习顺序见 [论文与学习路线](docs/research/robot_control_rl_reading_route_2026-11-24.md)。优先结合四足本体运动阅读，文献结果不代表本项目已复现。

底层运动任务位于 `imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/`；拖曳任务包 [towing/](imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/towing) 与 `locomotion/` 同级。小车直接维护 [cart.urdf](imgo2_description/cart/cart.urdf)，离线与仿真端的完成状态见 [拖曳仿真验证](docs/towing/towing_simulation_validation.md)。

拖曳的**三类连接模型**统一在 [mdp/rope_model.py](imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/mdp/rope_model.py)：`ROPE_MODELS = ("compliant", "inextensible")` 是两套**绳**（单边、只拉，故 `rope_extension` 只报正部、`rope_tension ≥ 0`），`RIGID_MODEL = "rigid"` 是**双边球铰连杆**（`d ≡ L0`，张力有符号，负值即推力／反驱），`CONNECTION_MODELS` 才是三类全集。训练场景的**列 × 行确定性网格**在 [mdp/connection_grid.py](imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/mdp/connection_grid.py)（列 = 类型/弹性档、行 = 长度；纯算术可离线测）：[upper_mdp.py](imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/upper_mdp.py) 按 env index 一次算出逐 env 的类型/长度/k/c 张量，三套模型都按这些张量构造（`MultiRopeModel` 按整数 id 选择），reset 只做域随机化并按**本 env 的连接长度**用 `attachment_horizontal_gap()` 摆放小车初始位置。测量台 [tow_drag.py](imgo2_rl/scripts/towing/tow_drag.py) 的 `--rope-model` 同样接受三者。
首次学习 Isaac Lab：先读 [仿真最小结构与运行链路](docs/training/isaaclab_basics_2026-09-24.md)，区分模型资源、资产配置、环境和算法。


| 需要理解或修改的内容 | 主要入口 |
|---|---|
| 机器人模型、初始关节角、执行器 | [assets/imgo2.py](imgo2_rl/source/imgo2_rl/imgo2_rl/assets/imgo2.py) |
| 公共场景、观测、动作、奖励、随机化、课程 | [velocity_env_cfg.py](imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py) |
| 基础运动任务注册 | [base_move/__init__.py](imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/__init__.py) |
| 平地、粗糙地形、HIM-Loco、AMP 配置 | [base_move/](imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move) 下对应 `*_env_cfg.py` |
| 算法超参数与实验名称 | 同一任务目录下的 `agents/` |
| 自定义奖励、观测、重置、速度指令 | [mdp/](imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/mdp/) |
| 自定义算法、网络、runner、包装器、数据加载器 | [scripts/rl_lab/rl_lab/](imgo2_rl/scripts/rl_lab/rl_lab/) |
| MPC AMP 采集（ROS1）与重采前检查 | [record_legged_control_amp.py](imgo2_rl/scripts/record_legged_control_amp.py)、[采集复核](docs/maintenance/project_history_2026-09_to_2026-10.md#amp)；脚本默认话题/frame/输出需按实际MPC工程确认 |
| AMP 动作数据离线核对（高度、FK、腿顺序） | [audit_amp_dataset.py](imgo2_rl/scripts/tools/audit_amp_dataset.py)、[check_amp_joint_order.py](imgo2_rl/scripts/tools/check_amp_joint_order.py) |
| 资源路径自检（新机器/训练服务器前置） | [check_asset_paths.py](imgo2_rl/scripts/tools/check_asset_paths.py) |
| 模型副本一致性（关节/惯量/碰撞/网格/FK） | [check_model_sync.py](imgo2_rl/scripts/tools/check_model_sync.py) |
| 步态评估（占空比/步频/相位/关节幅值，需回放） | [eval_gait.py](imgo2_rl/scripts/tools/eval_gait.py) |
| 步态指标口径（纯 numpy，可离线回归，`eval_gait.py` 只采样与打印） | [gait_metrics.py](imgo2_rl/scripts/tools/gait_metrics.py)、[test_gait_metrics.py](imgo2_rl/tests/test_gait_metrics.py)（18 项，无需 GPU） |
| 参考轨迹回放（在 Isaac Lab 里放参考动作） | [play_reference_motion.py](imgo2_rl/scripts/tools/play_reference_motion.py) |
| 拖曳必要性测试台的**离线统计**（判定码分布 / 按坡度·连接·质量·速度·行切分 / 指标分位数，纯标准库，可读正在跑的 run） | [play_test_stats.py](imgo2_rl/scripts/tools/play_test_stats.py) |
| 拖曳 latent **聚类/可分性检验**（阶段均值；标准化 → PCA → k-means（ARI/NMI/纯度）→ 轮廓 → 质心与 kNN 留一 → 标签置换检验 → 组内 latent 半径；`--rows`/`--filter` 做绳杆同长度等受控对照；核心指标纯标准库，`--plots` 才需 matplotlib） | [analyze_latent_clusters.py](imgo2_rl/scripts/tools/analyze_latent_clusters.py)、[analyze_latent.py](imgo2_rl/scripts/tools/analyze_latent.py)（估计误差 + η²/LOO）、[analyze_latent_tsne.py](imgo2_rl/scripts/tools/analyze_latent_tsne.py)（t-SNE 图）、结论与限制见 [聚类检验记录](docs/maintenance/project_history_2026-09_to_2026-10.md#towing)、离线测试 [test_towing_latent_clusters.py](imgo2_rl/tests/test_towing_latent_clusters.py)（10 项，无需 GPU） |
| 参考步态基线（从 21 份动作直接算出，判读用尺子） | [gait_reference_baseline.json](docs/amp/data/gait_reference_baseline.json) |
| 24500 轮 AMP 的足端回放结果与逐腿对照 | [gait_eval_amp_24500_2026-09-18.md](docs/amp/gait_eval_amp_24500_2026-09-18.md) |
| 训练曲线（读 TensorBoard event 文件，纯标准库，训练中也能读；`--tags`/`--steps`/`--match`；**注意 `*/time` 系列 tag 的 x 轴是墙钟秒、不是迭代轮号**，工具会单独点名） | [read_tfevents.py](imgo2_rl/scripts/tools/read_tfevents.py)、[test_read_tfevents.py](imgo2_rl/tests/test_read_tfevents.py)（6 项，无需 GPU） |
| 判别器探针（离线，读 checkpoint 的判别器做破坏性对照，无需 GPU） | [probe_amp_discriminator.py](imgo2_rl/scripts/tools/probe_amp_discriminator.py) |
| PPO rough 奖励离线复算（在 MuJoCo 逐帧 CSV 上按公式重算每秒分项，可改 weight/步高/滞空阈值做消融；纯标准库） | [audit_ppo_rough_reward.py](imgo2_rl/scripts/tools/audit_ppo_rough_reward.py)、结果见 [奖励与步态复核](docs/training/ppo_rough_gait_reward_audit_2026-10-28.md) |
| 策略 vs 专家的 AMP 观测块级归因（需 `eval_gait.py --dump-npz` 的产物） | [probe_amp_policy_vs_expert.py](imgo2_rl/scripts/tools/probe_amp_policy_vs_expert.py) |
| λ_gp 对判别器分辨能力的合成对照（离线训练，CPU） | [probe_gp_effect.py](imgo2_rl/scripts/tools/probe_gp_effect.py) |
| 步态调整方案（风格项为什么没梯度、按成本排序的改法与判据） | [amp_gait_adjust_plan_2026-09-18.md](docs/amp/amp_gait_adjust_plan_2026-09-18.md) |
| 拖曳质量／速度／阻力边界扫描 | [scan_towing_boundary.py](imgo2_rl/scripts/towing/scan_towing_boundary.py)、[扫描口径](docs/maintenance/project_history_2026-09_to_2026-10.md#towing) |
| 拖曳上层任务必要性（**直接跑训练场景**：800 环境 = 训练网格全集；逐 env 只扫速度×质量，五项指标 + 横向/朝向 PD 保持） | [play_towing_test.py](imgo2_rl/scripts/towing/play_towing_test.py)、[口径与限制](docs/towing/towing_necessity_test_2026-10-09.md)、离线测试 [test_towing_play_test.py](imgo2_rl/tests/test_towing_play_test.py)（103 项，无需 GPU） |
| 部署观察处理与策略推理 | [rl_sdk.cpp](imgo2_deploy/ros2_ws/src/imgo2_deploy/library/core/rl_sdk/rl_sdk.cpp) |
| 部署状态切换 | [fsm_imgo2.hpp](imgo2_deploy/ros2_ws/src/imgo2_deploy/fsm_robot/fsm_imgo2.hpp) |
| 构建选项与外部依赖条件 | [build.sh](imgo2_deploy/ros2_ws/build.sh)、[CMakeLists.txt](imgo2_deploy/ros2_ws/src/imgo2_deploy/CMakeLists.txt) |

配置有继承关系：理解最终行为时，要同时检查公共配置和任务类的 `__post_init__()`。例如粗糙地形 PPO 和 HIM-Loco 配置都将随机推扰、外力事件设为 `None`，不能仅凭公共类存在相应定义就写成“已启用”。

## 4. 训练与回放流程

### 4.1 环境与路径前提

> 2026-10-10 当前源码复核：`assets/imgo2.py` 默认读的是 `imgo2_description/urdf/imgo2_real.urdf`，本节后续 `imgo2.urdf` 表述为原基线历史；当前 real 模型一致性仍见 MODEL-05。启动器默认解释器在本机不存在，需通过 `IMGO2_ISAACLAB_PYTHON` 指向实际 Isaac Lab 环境。已修复说明缺口，待确认运行环境与短训练，见 [阅读指南](docs/training/rl_structure_guide_2026-10-10.md)。

以下 Python 命令均从 `imgo2_rl/` 执行，并使用已配置好 Isaac Lab/Isaac Sim 的 Python 环境。

- 2026-10-10 本机 `IsaacLab-rsl5` 安装核对：它是 Isaac Lab 本体，实际提交 `c22775241e28f465fe345fa1a482ad6d29d712b0` 声明 RSL-RL 5.0.1；版本文件 2.3.2 不等于确认处于同名标签。本机 Sim 5.1.0，服务器复现命令及待验证项见 [安装记录](docs/training/isaaclab_rsl5_install_2026-10-10.md)。
- 全新 Ubuntu 22.04/24.04 的 AMP Height 安装、前置检查与 10 轮试跑命令见 [完整安装流程](docs/training/ubuntu_amp_install_2026-10-10.md)。使用环境内 pip 安装的 Sim 5.1；新机器安装与运行结果待确认。
- AMP Height 时间链路核对：物理 200 Hz、策略与专家状态对 50 Hz、24 步 rollout=0.48 s、部署策略 50 Hz；专家插值帧索引偏差已修并 CPU 验证，修正后短训练待运行，见 [采样检查](docs/maintenance/project_history_2026-09_to_2026-10.md#amp)。
- AMP 风格与滞空奖励复核：判别器仅观察相隔 20 ms 的运动状态，未显式观察接触／滞空时长；0.3 s 为线性落地奖励门槛，过度增权的副作用待接触测量确认，见 [复核记录](docs/amp/amp_air_time_style_2026-10-10.md)。
- 当前motion_new 25000可在保持45维观测/动作及数据尺度兼容的前提下做奖励微调；新高度/防滑/周期奖励尚未实现，身体稳定项不能等同于抬脚惩罚，目标与恢复入口见 [慢步态续训复核](docs/maintenance/project_history_2026-09_to_2026-10.md#amp)。

- 当前训练机版本基线由用户确认：**Isaac Lab 2.2.1、RSL-RL 2.3.3**；Python、Isaac Sim、PyTorch、CUDA 和驱动版本仍需在训练机实际命令输出中补齐。
- `imgo2_rl` 和 `rl_lab` 安装脚本声明 Python `>=3.10`，不代表完整兼容矩阵。上层拖曳已像 AMP 一样使用仓库自有 `rl_lab` runner／PPO／recurrent model，不依赖外部 RSL-RL runner/config API；普通 PPO 入口仍按 2.2.1 接口维护。详见 [训练栈兼容性记录](docs/training/training_stack_compatibility_2026-09-22.md)。
- `assets/imgo2.py` 的 URDF 与动作数据路径已改为**由文件自身位置推导**（`Path(__file__)` 上溯 4 层得到 `imgo2_rl/` 项目根），不再写死机器绝对路径。无论仓库 clone 到哪里、从哪个目录启动都成立；前提是可编辑安装（`pip install -e`），非可编辑安装会把包拷进 site-packages，届时数据目录不在上溯路径上。
- 需要把数据或模型放在别处时，用环境变量覆盖而不用改代码：`IMGO2_AMP_MOTION_DIR` 覆盖动作数据目录，`IMGO2_URDF_PATH` 覆盖 URDF 路径。
- 做 base 惯量对照时，可用 [`scale_base_inertia.py`](imgo2_rl/scripts/tools/scale_base_inertia.py) 的 `--input 基准.urdf --axis x --shrink-percent 45 --output 候选.urdf`，将垂直于机身 X 轴的两个质量坐标尺度缩小 45%。工具生成新文件，保留模型质量、质心和几何，并处理相对网格路径。Isaac Lab 用 `IMGO2_URDF_PATH` 选用候选文件。每个收缩档位都从同一基准生成。
- 在新机器（尤其训练服务器）上训练前，先运行 `python scripts/tools/check_asset_paths.py`：它不需要 Isaac Lab，会打印实际解析到的路径、动作文件数（应为 21）以及是否还有残留的机器绝对路径。
- 另有 [assets/amp_motions.py](imgo2_rl/source/imgo2_rl/imgo2_rl/assets/amp_motions.py) 用于查找另一个参考项目的 mocap 数据，也已去掉机器绝对路径。当前 AMP 环境与 runner 配置导入的是 `assets.imgo2.AMP_MOTION_FILES`，不经过该模块；仅设置环境变量不会改变当前训练数据来源。

安装项目包：

```bash
cd imgo2_rl
python -m pip install -e source/imgo2_rl
python -m pip install -e scripts/rl_lab
python scripts/tools/list_envs.py
```

子项目旧 README 中的 `script/himloco_rsl_rl` 安装路径已过时；当前本地算法包在 `scripts/rl_lab`。

### 4.2 任务与训练入口

普通 base_move PPO 的当前设置、本机环境前置项与短训练命令见 [PPO 训练核对](docs/training/ppo_training_preflight_2026-10-14.md)；推荐使用 `scripts/rl_lab/ppo/train.py`。rough的最终奖励、模型、地形与复杂地形训练限制见 [2026-10-26复审](docs/maintenance/project_history_2026-09_to_2026-10.md#ppo-training)。 训练曲线判读、固定测试集与checkpoint选择见 [评估指南](docs/training/policy_evaluation_guide_2026-10-26.md)。

5300平地低速的低抬脚/稍快频候选已准备：从仓库根运行`bash imgo2_rl/scripts/rl_lab/ppo/train_low5300.sh --dry-run`查看，去掉`--dry-run`会新建run并追加1000轮（本轮未执行长训）。支持`--baseline`原参数对照；显式`--warm_start`恢复模型/轮次、重建优化器，原checkpoint不改。候选/对照8环境2轮启动已验证，效果待，见完整参数、解释器与验收（本机 `low5300_retrain_2026-11-10.md`）。

下表来自源码注册项，尚未通过运行时枚举验证。

| 任务 ID | 训练入口，相对 `imgo2_rl/` |
|---|---|
| `Imgo2-basemove-rough-ppo` | `scripts/rl_lab/ppo/train.py` |
| `Imgo2-basemove-flat-ppo` | `scripts/rl_lab/ppo/train.py` |
| `Imgo2-heightmove-rough-ppo` | `scripts/rsl_rl/train.py` |
| `Imgo2-handstand-rough-ppo` | `scripts/rsl_rl/train.py` |
| `Imgo2-basemove-rough-himloco` | `scripts/rl_lab/himloco/train.py` |
| `Imgo2-basemove-flat-amp-height` | `scripts/rl_lab/amp/train.py`（平坦地形，高度与轻量姿态奖励；当前新数据实验关闭滞空项；`AMPHeightRunnerCfg`，见 [复审](docs/maintenance/project_history_2026-09_to_2026-10.md#amp)） |
| `Imgo2-basemove-flat-amp-fanziqi` | `scripts/rl_lab/amp/train.py`（Fanziqi A1 AMP 配方：`dt=0.005 s`、`decimation=6`，只留线／角速度任务奖励，42 维 actor，无持续外力；`FanziqiAMPRunnerCfg`） |

Rough PPO 的四腿原生摩擦候选通过进程开关启用。进入 `imgo2_rl/` 并配置服务器的 Isaac Lab 解释器后，可先运行短训练：

```bash
IMGO2_NATIVE_JOINT_FRICTION=1 bash scripts/run_isaaclab.sh scripts/rl_lab/ppo/train.py \
  --task=Imgo2-basemove-rough-ppo --headless --device=cuda:0 \
  --num_envs=8 --max_iterations=2 --run_name=native_friction_smoke
```

启用需 Isaac Sim 5 或更高版本，本机验证使用 5.1。原训练命令默认保留原执行器设定，同模型回放需要使用相同开关。 11/23 已在本机 PhysX 回读确认，开关开启后四个小腿使用静摩擦 0.325、动摩擦 0.195、黏性摩擦 0.20。该候选的大幅台架跟踪改善，小幅与步长验收仍待完成。 [服务器训练](docs/xiaoji_cmd/服务器训练.md)

此外注册了 `Imgo2-basemove-rough-himloco-play`，两套 AMP 分别使用同名 `-play` 任务。

基础检查及训练命令示例：

> **先读这条（2026-09-20）**：所有 Isaac Lab 命令都走 `imgo2_rl/scripts/run_isaaclab.sh`，例如
> `bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/rl_lab/amp/train.py --task=... --headless`。
> 原因是 `libtorch_cuda_linalg.so: cannot open shared object file` 这个报错（用户跑 `eval_gait.py` 时实遇）：
> Isaac Lab 的 `SimulationContext` 把 Isaac Sim 的 prim/变换后端硬编码成 `backend="torch"`，
> 建场景时会 `torch.linalg.solve` 到 CUDA 张量，而这一步要 **`dlopen` 裸名**的
> `libtorch_cuda_linalg.so` —— glibc **不会**用 `DT_RPATH` 解析显式 `dlopen`（torch 的
> `libtorch_cuda.so` 恰好只有 RPATH），所以直接把 `torch/lib` 放进 `LD_LIBRARY_PATH` 才可靠。
> 启动器追加这些库目录，并按仓库位置补齐 `scripts/rl_lab` 和 `source/imgo2_rl` 的模块搜索路径。
> 使用启动器导出时，终端无需提前设置 `PYTHONPATH`。
> 先跑 `bash imgo2_rl/scripts/run_isaaclab.sh --check` 自检（会打印 torch 版本、那个 .so 是否存在、
> 能否 dlopen、CPU/CUDA 上 `linalg.solve` 是否可用）。机理与离线复现见
> [启动器记录](docs/training/isaaclab_launcher_libtorch_dlopen_2026-09-20.md)。

```bash
python scripts/tools/zero_agent.py --task=Imgo2-basemove-flat-ppo --num_envs=1
python scripts/rsl_rl/train.py --task=Imgo2-basemove-flat-ppo --headless
python scripts/rsl_rl/train.py --task=Imgo2-basemove-rough-ppo --headless
python scripts/rl_lab/himloco/train.py --task=Imgo2-basemove-rough-himloco --headless
python scripts/rl_lab/amp/train.py --task=Imgo2-basemove-flat-amp-height --headless
python scripts/rl_lab/amp/train.py --task=Imgo2-basemove-flat-amp-fanziqi --num_envs=256 --max_iterations=100 --seed=42 --headless
```

公共配置默认 `num_envs=4096`。初次检查可使用脚本的 `--num_envs` 参数缩小规模；具体训练规模依实际显存确定。

### 4.3 回放与策略导出

**导出部署用的 `policy.pt` 一律走对应算法目录下的 `play.py`（headless），不要手写导出脚本或手工拼
TorchScript**（用户 2026-09-18 决定）。`play.py` 自己调用
`export_policy_as_jit(actor_critic, normalizer=None, ...)`，产物落在 checkpoint 同级的 `exported/`
（`policy.pt` + `policy.onnx`）；两个前提：

- **`imgo2_rl` 与 `rl_lab` 都必须在「你实际用来运行的那个解释器」里可编辑安装**。两个都装是正解
  （第 4.1 节那两条）：只装/只加路径的 `imgo2_rl` 会先报
  `ModuleNotFoundError: No module named 'imgo2_rl'`，随后 AMP 的 `play.py` 还会在
  `rl_lab.runners → rl_lab.datasets.motion_loader → from pybullet_utils import transformations`
  处报 `No module named 'pybullet_utils'`（2026-09-18 两次实遇）——因为 **`pybullet` 是
  `scripts/rl_lab/setup.py` 里声明的 `install_requires`**，走 `PYTHONPATH` 绕过打包就不会装它。
  运行时用 `isaaclab.sh -p` 就用同一个 `isaaclab.sh -p -m pip install -e`：
  `isaaclab.sh -p -m pip install -e source/imgo2_rl` 与 `-e scripts/rl_lab`
  （Isaac Sim 自带的 python 与别的 conda python 不是同一个解释器，在 A 里装、用 B 跑就会报错）。
  自检一条命令就够了：
  `isaaclab.sh -p -c "import imgo2_rl, rl_lab, pybullet_utils; print('ok')"`。
  应急（只想跑这一次）可以 `PYTHONPATH=<工作区根>/imgo2_rl/source/imgo2_rl` 加上
  `pip install pybullet`，但这只补 `imgo2_rl`，`pybullet` 仍要装。注意这里是 **Isaac Lab 终端**，
  与第 6.2 节 ROS 2 终端"先清 `PYTHONPATH`"的要求相反，不要照搬。
  AMP 这条链的额外第三方依赖**只有 `pybullet`**：`tensordict`（`mdp/symmetry/anymal.py`）不在我们的
  import 链上，`onnxruntime`/`pandas`（himloco 的 compare 工具）、`yaml`（`export_deploy_cfg`）、
  `packaging`/`rsl_rl`（`ppo/play.py`）都只属其它入口。
- **配置的观测维数与该 checkpoint 一致**，否则 `runner.load()` 会在 `load_state_dict` 处尺寸不匹配报错。

将下面的占位路径替换为实际 checkpoint 的绝对路径：

```bash
python scripts/rsl_rl/play.py --task=Imgo2-basemove-flat-ppo --num_envs=1 --checkpoint="/absolute/path/to/model.pt"
python scripts/rl_lab/himloco/play.py --task=Imgo2-basemove-rough-himloco-play --num_envs=1 --checkpoint="/absolute/path/to/model.pt"
python scripts/rl_lab/amp/play.py --task=Imgo2-basemove-flat-amp-height-play --num_envs=1 --headless --checkpoint="/absolute/path/to/model.pt"
```

以上三条假定当前目录是 `imgo2_rl/`。

**拖曳上层训练**（任务已注册，`--agent` 必须是完整注册键）。先看训练场景（去掉 `--headless` 就是 GUI）：

```bash
cd /root/Desktop/Imgo2
# 看场景：45 环境 = 网格第 0 行（列 0–19 平地、20–29 ±5°、30–39 ±10°）+ 第 1 行前 5 格
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/rl_lab/towing/train.py \
    --task=Imgo2-towing-upper-rl-lab --agent=rl_lab_cfg_entry_point \
    --num_envs 45 --max_iterations 2 --compact-log --run_name scene-look

# 正式训练：默认 800 环境（40 列 × 20 行，长度 0.6→1.2 m），3000 轮 × 48 步/轮
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/rl_lab/towing/train.py \
    --task=Imgo2-towing-upper-rl-lab --agent=rl_lab_cfg_entry_point --headless --compact-log
```

日志落在 `logs/towing_rl_lab/towing_upper/<时间戳>[_run_name]/`（`params/env.yaml`、`params/agent.yaml` + TensorBoard），默认每 100 轮存一次 checkpoint（`--save_interval` 可改）。**不要 `--resume` 旧 run**：2026-10-09 起 checkpoint 契约是 version=2（57 维帧 → 6 维显式 + 16 维 latent、actor 79 维），旧的 56/63 维 checkpoint 会被明确拒绝加载。`train.py` 没有接 `RecordVideo`，`--video` 只开相机、不会录视频。

**拖曳上层回放**（`--agent` 必须是完整注册键）在仓库根用统一入口跑，并把回放诊断一起打开：

```bash
cd /root/Desktop/Imgo2        # 或在任意位置用 --checkpoint 的绝对路径
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/rl_lab/towing/play.py \
    --task=Imgo2-towing-upper-rl-lab --agent=rl_lab_cfg_entry_point \
    --num_envs=1 --headless \
    --checkpoint="logs/towing_rl_lab/towing_upper/<run>/model_2000.pt" \
    --episode-log --cmd-interval=50 --cmd-summary --decoder-summary --max-steps=2000
```

`--num_envs=1` 约有 12.5% 概率抽到「无小车」环境，要统计量就开 `--num_envs=32`；`--max-steps` 以控制步计（`step_dt=0.05 s`，2000 步 ≈ 10 个回合）。汇总默认写到 checkpoint 同级的 `play_summary.json`，其键的判读口径、各档指令（看画面／录视频）与一次该 run 的 TensorBoard 读数见 [拖曳回放指令与前置核对](docs/towing/towing_play_2026-09-25.md)。

**上层任务必要性基线**（冻结 AMP 策略 + 脚本指令，不加载上层 checkpoint）用 [play_towing_test.py](imgo2_rl/scripts/towing/play_towing_test.py)，先在无 GPU 的机器上核对网格：

```bash
python3 imgo2_rl/scripts/towing/play_towing_test.py --dry-run      # 只打印网格与代价，不启动仿真
# 完整训练网格（800 环境 = 40 列 × 20 行，一次仿真跑完）
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/towing/play_towing_test.py --headless
# 冒烟（网格前缀）：40 环境 ≈ 三种连接 + 三个坡度量级都在
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/towing/play_towing_test.py \
    --headless --num-envs 40 --tow-duration 2 --coast-duration 2 --write-csv none
```

场景 = **训练场景** `UpperTowingSceneCfg`（40 列 × 20 行 = 800 条 lane），逐 env 的连接类型/长度/坡度量级取 `connection_grid.env_spec(i)`，**只扫速度 × 质量**（确定性轮转 `slot = row + column`：15 个组合各 50–55 个 env，且「连接 × 质量」「坡度量级 × 质量」严格均衡）；地面摩擦固定 `0.8`、轮轴阻尼 `0.032`（训练名义值）；横向/朝向用 PD 压在 lane 中线并保持超前（**vx 只给指令、不参与 PD**）；旧的「`gravity` / `terrain` 两种坡度后端」与 `--connections/--slopes/--slope-backend` 等开关已删除实现（与真实坡面同解），`--slope-backend terrain` 则用远端 2026-10-09 的真实坡面 tile（与训练场景同源，每坡度需 ≤100 case）。要对照「Fixed Ramp」基线加 `--command-shaping ramp`；要在坡上跑无制动真实溜坡用 `--slope-settle free`。产物是 `report.md`（逐坡度判定矩阵 + 结论）、`report.csv`、`report.json` 与逐 case 摘要，口径、阈值与限制见 [记录](docs/towing/towing_necessity_test_2026-10-09.md)。

以上三条假定当前目录是 `imgo2_rl/`。**拖曳上层**（`--agent` 必须是完整注册键）在仓库根用统一入口跑，并把回放诊断一起打开：

```bash
cd /root/Desktop/Imgo2        # 或在任意位置用 --checkpoint 的绝对路径
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/rl_lab/towing/play.py \
    --task=Imgo2-towing-upper-rl-lab --agent=rl_lab_cfg_entry_point \
    --num_envs=1 --headless \
    --checkpoint="logs/towing_rl_lab/towing_upper/<run>/model_2000.pt" \
    --episode-log --cmd-interval=50 --cmd-summary --decoder-summary --max-steps=2000
```

`--num_envs=1` 约有 12.5% 概率抽到「无小车」环境，要统计量就开 `--num_envs=32`；`--max-steps` 以控制步计（`step_dt=0.05 s`，2000 步 ≈ 10 个回合）。汇总默认写到 checkpoint 同级的 `play_summary.json`，其键的判读口径、各档指令（看画面／录视频）与一次该 run 的 TensorBoard 读数见 [拖曳回放指令与前置核对](docs/towing/towing_play_2026-09-25.md)。

导出后按三条契约复核，再拷进 `imgo2_deploy/ros2_ws/policy/imgo2/<算法>/policy.pt`：① 导出件与 checkpoint 的
actor 在同一批确定性输入上 max|diff| = 0；② 能被部署自己的 libtorch 加载；③ 部署 interface 的观测维数
与顺序和训练侧一致（AMP 的 45 维见第 5.4 节与 [Gazebo 记录](docs/deployment/gazebo_ros2_bringup_2026-09-17.md) 第 9 节）。

HIM-Loco 当前训练日志路径按 `logs/himloco_rsl_rl/<experiment_name>/<run>/` 组织，配置中的 `experiment_name` 为 `base_move_himloco`。不要直接沿用旧 README 示例的个人服务器 checkpoint 路径。

HIM-Loco 导出行为：

- 训练入口通过 `export_deploy_cfg()` 将训练侧参数写到运行目录的 `params/deploy.yaml`。
- 回放入口在 checkpoint 同级的 `exported/` 中导出 `policy.pt`（合并 encoder 与 policy 的 TorchScript）、`encoder.onnx` 和 `policy.onnx`（分开的两个网络）。
- [compare_pt_onnx.py](imgo2_rl/scripts/rl_lab/himloco/compare_pt_onnx.py) 可用于比较训练策略和 ONNX 输出；当前脚本有六帧历史的硬编码假设。修改历史长度后，应先同步检查该脚本再运行。
- 训练导出的 `deploy.yaml` 与 C++ 使用的 `base.yaml` / `config.yaml` 字段结构不同，需要进行字段映射和数值验证，不能直接视为可替换配置。

## 5. 训练与部署接口

### 5.1 基础运动配置的关键参数

以下主要对应 `assets/imgo2.py`、公共环境和 `base_move` 中的 PPO/HIM-Loco/AMP 配置；其他任务可能覆盖这些值。

| 项目 | 源码中的值或约定 |
|---|---|
| 自由度 | 12，每条腿 hip、thigh、shank 三个关节 |
| 基础运动动作腿顺序 | `FL, FR, RL, RR`；每腿依次 hip、thigh、shank |
| 基座与足端名称 | `base`；`FL_FOOT, FR_FOOT, RL_FOOT, RR_FOOT` |
| 初始关节角 | hip `0.0`、thigh `0.87`、shank `-1.82` rad |
| 仿真步长与降采样 | `sim.dt=0.005 s`、`decimation=4`，即策略周期 `0.02 s` / `50 Hz` |
| 训练模型执行器配置 | `DCMotorCfg`，刚度 `25.0`、阻尼 `0.5`、力矩上限与饱和值 `23.7`、速度上限 `30.1`；实际训练可能受随机化影响 |
| 动作尺度 | hip `0.125`，其余关节 `0.25`；使用默认关节位置偏置 |
| AMP 指令范围与数据覆盖 | 45 维任务 `lin_vel_x (-1.0, 1.5)`、`lin_vel_y ±1.0`、`ang_vel_z ±1.57`（`amp_env_cfg.py`）；fanziqi/rlamp `lin_vel_x (-1.0, 2.0)`、`lin_vel_y ±0.3`。**录制动作实测覆盖只有** vx −1.03…**+0.84** m/s、vy ±0.51、yaw ±0.765 rad/s ⇒ 前向正向半轴 44%、yaw 51% 的指令区间**没有专家参考**；19 份移动动作节律恒为 **0.600 s / 1.667 Hz**，速度全靠步幅（0.117→0.382 m）变。加宽指令范围不增加专家参考，核算与取舍见 [AMP 速度包线](docs/amp/amp_speed_coverage_2026-10-10.md) |

应从运行时 action manager 确认最终关节排列、裁剪和缩放。不要用 URDF 中的文本顺序代替运行时关节顺序。

**PPO可选分组指令（2026-11-05）**：`mdp/commands.py`末尾新增`GroupedVelocityCommandCfg`，可独立设置静止/低速六方向概率与低速幅值，宽范围组仍使用课程ranges；旧类和默认任务未变。启用时在rough配置末尾替换命令Cfg，父类`rel_standing_envs=0`、heading关闭，完整示例与短环境验证见[接入](docs/training/ppo_grouped_command_sampling_plan_2026-11-05.md)。尚无新训练/步态验收。

### 5.2 HIM-Loco 观察约定

HIM-Loco actor 单帧按以下顺序构造，按配置计算为 45 维：

| 顺序 | 观察项 | 维度 | 显式缩放 |
|---|---|---|---|
| 1 | 速度指令 | 3 | 未显式设置，按原值 |
| 2 | 基座角速度 | 3 | `0.25` |
| 3 | 重力投影 | 3 | 未显式设置，按原值 |
| 4 | 相对默认位置的关节角 | 12 | `1.0` |
| 5 | 相对默认速度的关节速度 | 12 | `0.05` |
| 6 | 上一次动作 | 12 | 未显式设置，按原值 |

当前 `history_length=5` 表示五帧历史加当前帧，共六帧，输入为 `45 × 6 = 270` 维。包装器将最新帧放在前面。critic 另含基座线速度和地形高度扫描等信息，不应与 actor 输入混用。

普通粗糙地形 PPO 的 actor 也移除了基座线速度和高度扫描，但其观察项顺序与 HIM-Loco 不同。两种策略不能共用未经核对的输入拼接规则。

### 5.3 部署配置当前差异

来源：[base.yaml](imgo2_deploy/ros2_ws/policy/imgo2/base.yaml)、[himloco/config.yaml](imgo2_deploy/ros2_ws/policy/imgo2/himloco/config.yaml)。下面是静态文件值，运行时还需核对加载和覆盖结果。

| 项目 | 训练侧 | 部署配置现状 |
|---|---|---|
| 物理关节列表 | 动作使用 `FL, FR, RL, RR` | 当前 `base.yaml` 已为 `FL, FR, RL, RR`；HIMLoco 配置仍用左右交换 mapping（2026-11-02复核） |
| 默认关节角 | 所有腿为 `0, 0.87, -1.82` | 当前 base 已为 `0, 0.87, -1.82`；HIMLoco 策略配置仍含左右 hip 偏置及前后 thigh 差异 |
| 策略 PD | 模型名义值 `25.0 / 0.5` | 策略 `rl_kp/rl_kd` 为 `40 / 1`，固定姿态 PD 是另组参数 |
| 力矩限制 | 模型 `23.7` | 当前 base 为 `23.7`，HIMLoco 策略配置为 `33.5`；实际还受模型/控制器限幅 |
| 指令缩放 | HIM-Loco 未显式设置 | 策略配置为 `[2.0, 2.0, 0.25]` |
| 网络文件 | 回放导出 `policy.pt` 或两个 ONNX 网络 | 策略配置加载 `himloco.pt`，属参考占位策略 |

上表描述的是**仍为 Go2 参考占位的 himloco 策略**（DEPLOY-01 未关闭）。AMP 策略已于
2026-09-17 按训练侧 + 参考项目对齐，见下表；详细依据见 [sim2sim 记录](docs/maintenance/project_history_2026-09_to_2026-10.md#amp)。

| AMP 项目 | 训练侧 | [amp/config.yaml](imgo2_deploy/ros2_ws/policy/imgo2/amp/config.yaml) 现状 |
|---|---|---|
| 观察 | 当前 actor：`ang_vel/gravity/commands/joint_pos/joint_vel/actions` = 45（critic 另含线速度，为 48） | 同顺序、同项数（45）；不读取真实基座线速度作为 actor 输入 |
| 关节映射 | 动作与观察均为 `FL, FR, RL, RR` | `joint_mapping` 恒等，MuJoCo 场景关节声明顺序即策略顺序 |
| 默认关节角 | `0, 0.87, -1.82` ×4 | 同（`base.yaml` 与 amp 配置一致） |
| 策略 PD | `25.0 / 0.5` | `25.0 / 0.5`（与参考项目一致） |
| 固定姿态 PD | — | `60.0 / 2.0`（参考项目） |
| 力矩限制 | `23.7` | `23.7`（与 URDF/训练一致，参考项目为 23.5） |
| 动作缩放 | `0.125 / 0.25 / 0.25` | 同 |
| 动作裁剪（2026-10-13修正） | 缩放/偏置后的关节目标裁剪±3 rad，wrapper不裁剪raw action | Gazebo AMP启用gazebo_clip_joint_targets，先转换再clip，last_action保持raw；4182实际帧数值验证通过。controller额外URDF限位/其它后端待对齐，见 [修正对照](docs/deployment/gazebo_amp_clip_fix_comparison_2026-10-13.md) |
| 观察缩放 | `1.0 / 0.25 / 1.0 / 0.05 / 1.0` | 同 |
| 网络文件 | `imgo2_rl` 的 `play.py` 导出 | `policy.pt`（与 checkpoint actor 数值一致） |

Isaac Sim 回放指定固定速度的命令见 [AMP play 速度设置](docs/xiaoji_cmd/策略导出与ROS2仿真.md#isaac-sim-amp-回放时指定速度)；当前平移命令模长不超过 0.2 m/s 会被清零，低速测试需注意。

本机从AMP checkpoint正式导出、备份替换到ROS2 Gazebo启动的完整顺序见 [策略导出与ROS2仿真](docs/xiaoji_cmd/策略导出与ROS2仿真.md)。该链路是sim2sim，真机入口另行验证。

### 5.4 部署侧三个策略与按键

部署侧现在并列三个策略，每个是「一个按键 + 一个完整的 FSM 状态类」：

| 键 | 状态类 | 配置目录 | 观察 | 网络 |
|---|---|---|---|---|
| `1` | `RLFSMStatePPOLocomotion` | `policy/imgo2/ppo/` | 45 维（无 `base_lin_vel`、无 `height_scan`） | `policy.pt`（参考项目 `~/RL/sim2sim/Imgo2_deploy/policy/imgo2/base_move/policy_flat.pt`，sha256 `53a57f909b61d2a7…`；2026-09-17 按四项指标评过 5 个候选后选它） |
| `2` | `RLFSMStateRLLocomotion` | `policy/imgo2/himloco/` | 45 维 × 6 帧 = 270（`observations_history: [0..5]`） | `himloco.pt`（Go2 参考占位，见 DEPLOY-01） |
| `3` | `RLFSMStateAMPLocomotion` | `policy/imgo2/amp/` | 45维actor/12动作 | 最新2026-10-14 motion_new_model_new的model_25000正式导出，SHA9c6bf22b...；CPU128输入actor差0，当前动态效果待验证，见 [身份](docs/maintenance/rebase_latest_policy_identity_2026-10-14.json) |

**2026-09-25 新入库的拖曳占位（未接线、不可加载）**：`policy/imgo2/towing/policy.pt` 是
`logs/towing_rl_lab/towing_upper/2026-09-23_22-19-00/model_2000.pt` 的**原样拷贝**
（sha256 `a4d800a26c1d1a50…`，9.76 MB），里面是**训练态** checkpoint（actor／critic／decoder 三套 GRU
权重 + 两个优化器 + critic normalizer + `iter`），**不是** `play.py` 导出的 TorchScript，
`inference_runtime` 的 libtorch 加载器读不了它；该目录也**没有** `config.yaml`，上表三个键之外
**没有**对应 FSM 状态类。它只表示"权重已留档入库"（用户 2026-09-25 决定），
不代表部署链路可用——真正的导出件（57 维帧 + 6 维 estimate + 12 维关节残差的 TorchScript）
仍需给 `towing/play.py` 加导出路径后在训练机 headless 跑一次，见 TOW-05。

**2026-09-25 新入库的拖曳占位（未接线、不可加载）**：`policy/imgo2/towing/policy.pt` 是
`logs/towing_rl_lab/towing_upper/2026-09-23_22-19-00/model_2000.pt` 的**原样拷贝**
（sha256 `a4d800a26c1d1a50…`，9.76 MB），里面是**训练态** checkpoint（actor／critic／decoder 三套 GRU
权重 + 两个优化器 + critic normalizer + `iter`），**不是** `play.py` 导出的 TorchScript，
`inference_runtime` 的 libtorch 加载器读不了它；该目录也**没有** `config.yaml`，上表三个键之外
**没有**对应 FSM 状态类。它只表示"权重已留档入库"（用户 2026-09-25 决定），
不代表部署链路可用——真正的导出件（51 维帧 + decoder／actor 双 GRU hidden 的 TorchScript）
仍需给 `towing/play.py` 加导出路径后在训练机 headless 跑一次，见 TOW-05。

`ppo/config.yaml` 的数值取自参考项目 `.../base_move/config.yaml`（可追溯）：`rl_kp 25 / rl_kd 0.5`、
`default_dof_pos 0 / 0.87 / -1.82`、`action_scale 0.125/0.25`、`clip ±3`、观测 scale
`0.25 / 1.0 / 1.0 / 1.0 / 0.05 / 1.0`、`joint_mapping` 恒等。**换 checkpoint 必须连这些值一起换**：
仓库里三份 45 维 PPO 候选用的是两套参数（`0/0.8/-1.5` + `20/0.2` 或 `20/1.0`，与
`0/0.87/-1.82` + `25/0.5`），混用会让 `dof_pos` 相对量与 PD 都错位。

**2026-09-17 PPO 选型（Gazebo，四项指标，脚本 `imgo2_deploy/ros2_ws/scripts/eval_gazebo_policy.py`）**：
判据是**位姿 / 速度跟随 / 腿部抖动 / 周期性**——只看姿态会被"四脚朝天也很稳"骗过，只看关节看不出
是否真的在走。指标定义：位姿 = 基座 z 与 roll/pitch/yaw 漂移（`/odom` p3d 真值）；速度跟随 =
`dx/dt` 对命令 vx 的误差；抖动 = 相邻控制周期 `abs(Δdq)/dt` 的均值（分 hip/thigh/shank）与 `dq`
换向率；周期性 = 大腿角自相关首个峰的周期与强度（0~1）、FL↔FR 相位（trot≈180°）。

| 键 1 候选（vx=0.5） | 实测速度 | z_mean（范围） | roll/pitch | yaw 漂 | 抖动 thigh/shank | 周期强度 | FL-FR |
|---|---|---|---|---|---|---|---|
| **base_move/policy_flat.pt（现在）** | 0.428 m/s | **0.288**（0.280–0.295） | **2.5°/1.3°** | **−0.5°/s** | **45 / 94** | **0.95** | **+175°** |
| base_move/policy.pt（备选） | **0.522 m/s** | 0.251 | 5.0°/4.6° | +1.5°/s | 79 / 153 | 0.74 | +198° |
| base_move/policy1.pt | 0.662 m/s（超调） | 0.244 | 8.3°/7.6° | −1.5°/s | 153 / 91 | 0.68 | −148° |
| rough/16-37-38（换之前） | 0.540 m/s | 0.271（0.219–0.382） | 13.4°/10.9° | −2.7°/s | 241 / 507 | 0.29 | −140° |
| amp/policy.pt（7/3） | 0.057 m/s ✗ | 0.182 | 10.0°/6.8° | — | 53 / 71 | 0.84 | −164° |

`vx=0` 静止时：`policy.pt` 与 `policy_flat.pt` 抖动 **0.0**、yaw 漂 0.0（完全静止）；rough/16-37-38
抖动 47–77、yaw 漂 +4.8°/s。⇒ 采用 `policy_flat.pt`（四项里三项最好，仅速度低 14%）；
若更看重速度精度可用 `policy.pt`。五份训练导出的横向对比见
[策略运行期排查](docs/deployment/sim2sim_policy_runtime_2026-09-17.md) 第 7 节，Gazebo 侧的完整评测见
[Gazebo 记录](docs/deployment/gazebo_ros2_bringup_2026-09-17.md) 第 9 节。

复跑评测（Gazebo 无头，脚本自己注入起身/键1/速度）：

```bash
ros2 launch imgo2_deploy gazebo.launch.py gui:=false      # 终端 A
ros2 run imgo2_deploy rl_sim                              # 终端 B
python3 imgo2_deploy/ros2_ws/scripts/eval_gazebo_policy.py --vx 0.5 --duration 20   # 终端 C
```

交付一个可部署策略时，至少记录：checkpoint 来源、模型版本、关节映射、默认姿态、动作缩放与裁剪、PD 与力矩限幅、观察顺序与缩放、历史排列及重置方式、四元数约定、控制周期、网络输入输出，以及同一输入下的数值比较结果。

### 5.5 拖曳链路的批处理与 JIT 边界

上层拖曳训练同时用到三个网络，**都在环境维度整批前向，没有逐环境循环**：

| 网络 | 调用位置与频率 | 每次 batch |
|---|---|---|
| actor / critic（`ActorCriticRecurrent`） | `towing_on_policy_runner.py` rollout 循环，每控制步各一次 | `num_envs` |
| decoder（`TowingDynamicsDecoder`） | 同上，每控制步一次；6 维 estimate（vel 2 / mass 1 / force 3）detach 后拼成 63 维 actor 输入 | `num_envs` |
| 冻结低层 AMP（`FrozenLowLevelPolicy`） | `apply_actions` 里每 4 个物理步一次（`low_level_control_dt/physics_dt`） | `num_envs` |

三者各维护自己的 GRU hidden，形状 `(1, num_envs, 256)`（decoder 为 128）。4096 环境时每轮前向次数约：
上层 19.7 万次、冻结低层 49.2 万次。

**JIT 的边界**（2026-09-22 澄清）：

- **冻结低层已经是 TorchScript**：`policy.pt` 由对应算法的 `play.py` 经 `export_policy_as_jit`
  导出，`low_level_policy.py` 用 `torch.jit.load` 加载。它每轮被调数十万次，JIT 在这里有实际收益。
- **上层 actor／critic／decoder 在训练期不应 JIT**：`alg.act()` 要返回带梯度的分布、要维护
  recurrent hidden，PPO 更新还要反传；TorchScript 不适合带梯度训练。上层只在**导出部署**那一步
  才 JIT（约定见 AGENTS.md：一律走 `play.py`，不要手工拼 TorchScript）。
- 因此「JIT 更适合」不等于「训练也该 JIT」：JIT 消除的是 Python 逐 op 开销，批处理决定的是
  并行度，两者是不同问题，而后者在本链路已经做到。

## 6. 部署流程

下面为现有 Bash 构建脚本对应的命令，面向具备相关依赖的 Linux 环境；本次未在 Windows PowerShell 中执行。

### 6.0 部署包结构

`imgo2_deploy/` 源自 `rl_sar-main` 并裁剪为单一机器人（上游为 Apache-2.0，其 `LICENSE` 保留在该目录内）。内部按 ROS 工作区组织：

| 路径 | 内容 |
|---|---|
| `src/imgo2_deploy` | RL 部署包：sim2sim、MuJoCo 仿真、Imgo2 真机入口 |
| `src/robot_msgs` | 共享的电机/机器人状态消息 |
| `src/robot_joint_controller` | ROS 仿真用的 Gazebo 关节控制器；用 URDF 的关节限位 clamp 指令（ROS1 生效，ROS2 原先失效，见 DEPLOY-07） |
| `policy/imgo2` | 策略配置：`base.yaml`、`ppo/`（键 1，参考项目 `base_move/policy_flat.pt`）、Go2 参考占位的 `himloco/`（键 2）、`amp/`（键 3，`model_5000.pt` 的 45 维导出） |
| `../../imgo2_description/urdf/imgo2.urdf` | 部署侧 ROS/Gazebo 用 URDF（唯一模型源在 `imgo2_description/`，见 MODEL-02） |
| `../../imgo2_description/mjcf/` | MuJoCo场景，由rl_sim_mujoco经IMGO2_MODEL_DIR读取；10-23已按real Gazebo源同步惯性/质量等，运行边界见MODEL-MJCF-01 |

策略配置里关节名用 `*_shank_joint`（对应本仓库 URDF），而不是 Go2 的 `*_calf_joint`。

### 6.1 MuJoCo：训练仿真到另一仿真器

> 2026-10-14准备核查：当前MJCF的8个HIP/THIGH参数未同步最新real URDF，AMP裁剪/反馈亦有已知差异，本机缺MuJoCo依赖和可执行文件。下面是准备完成后的入口，当前不可据此宣称公平对照已通过；见 [核查](docs/maintenance/project_history_2026-09_to_2026-10.md#simulation-and-deployment)。

```bash
cd imgo2_deploy/ros2_ws

# 推理运行时与 MuJoCo 依赖；若本机已有安装，可把 library/ 软链到它（见下）
bash scripts/download_inference_runtime.sh
bash scripts/download_mujoco.sh

# 构建（等价于参考项目的 ./build.sh -mj）
bash build.sh -mj

# 运行；参数为 <robot_name> <scene_name>，场景取 $IMGO2_MODEL_DIR/mjcf/<scene>.xml
# （IMGO2_MODEL_DIR 由 CMakeLists 编译期设为 <repo>/imgo2_description）
./cmake_build/bin/rl_sim_mujoco imgo2 scene

# 小车拖曳：默认 10 kg，--cart-mass 可设 5–15 kg；仅这两个场景接受该参数
./cmake_build/bin/rl_sim_mujoco imgo2 scene_tow_compliant --cart-mass 5
./cmake_build/bin/rl_sim_mujoco imgo2 scene_tow_inextensible --cart-mass 15

# 小车拖曳：默认 10 kg，--cart-mass 可设 5–15 kg；仅这两个场景接受该参数
./cmake_build/bin/rl_sim_mujoco imgo2 scene_tow_compliant --cart-mass 5
./cmake_build/bin/rl_sim_mujoco imgo2 scene_tow_inextensible --cart-mass 15
```

两个拖曳场景都包含 `cart.xml`（与 `cart/cart.urdf` 同尺寸／惯量）、四轮轴承阻尼和 0.8 m 单边绳长约束；默认机器人在前、小车在后，绳初始松弛 0.4 m。运行 `python imgo2_rl/scripts/tools/check_towing_mjcf.py` 可离线核对模型参数。现有键 `3` 的 AMP 可用于底层牵引场景试跑；上层 51 维 recurrent 策略仍无已验证导出件，因此完整上层拖曳 sim2sim 尚未闭环。MuJoCo 原生约束与 Isaac Lab 的外力／冲量算法不能直接视为数值等价，见 [拖曳 sim2sim 记录](docs/towing/towing_sim2sim_2026-09-24.md)。

两个拖曳场景都包含 `cart.xml`（与 `cart/cart.urdf` 同尺寸／惯量）、四轮轴承阻尼和 0.8 m 单边绳长约束；默认机器人在前、小车在后，绳初始松弛 0.4 m。运行 `python imgo2_rl/scripts/tools/check_towing_mjcf.py` 可离线核对模型参数。现有键 `3` 的 AMP 可用于底层牵引场景试跑；上层 51 维 recurrent 策略仍无已验证导出件，因此完整上层拖曳 sim2sim 尚未闭环。MuJoCo 原生约束与 Isaac Lab 的外力／冲量算法不能直接视为数值等价，见 [拖曳 sim2sim 记录](docs/towing/towing_sim2sim_2026-09-24.md)。

复用既有依赖（本机做法，零拷贝，仓库内不留文件）：

```bash
cd imgo2_deploy/ros2_ws
mkdir -p library/inference_runtime
ln -sfn ~/RL/sim2sim/Imgo2_deploy/library/inference_runtime/libtorch    library/inference_runtime/libtorch
ln -sfn ~/RL/sim2sim/Imgo2_deploy/library/inference_runtime/onnxruntime library/inference_runtime/onnxruntime
ln -sfn ~/RL/sim2sim/Imgo2_deploy/library/mujoco                      library/mujoco
```

窗口内的按键：`0` 起身、**`1` PPO**、**`2` himloco**、**`3` AMP**、`9` 下蹲、`P` 回 Passive、
`W/S/A/D/Q/E` 步进速度指令、`Space` 清零指令、`R` 重置仿真、`Enter` 暂停/继续。
（手柄：`RB+DPadUp`=PPO、`RB+DPadRight`=himloco、`RB+DPadDown`=AMP。）三个策略各自是一个
`RLFSMState*Locomotion` 类 + 一条按键指令，新增策略照这个模式加即可。

构建依赖网络与本地编译工具链。2026-09-17 已在本机通过 `build.sh -mj`；GUI 入口需要可用的
OpenGL/显示，本机无可用 NVIDIA 驱动（`nvidia-smi` 失败）故未运行，见
[sim2sim 记录](docs/maintenance/project_history_2026-09_to_2026-10.md#amp) 第 4 节。无 GUI 时的替代做法是仓库外临时 harness
（复用真实 `RL`/FSM/`librl_sdk` 与 `policy/`，只重写 `GetState`/`SetCommand`/`RunModelStep`/`Forward`），
本轮三个策略的验证都用它完成，要点与数值见
[策略运行期排查](docs/deployment/sim2sim_policy_runtime_2026-09-17.md)。

**改 MJCF 传感器时必须挂在 site 上**：`<framequat>`/`<framelinvel>` 若写成 `objtype="body"`，
返回值会再乘该 body 的惯量主轴旋转（`iquat`），base 的 `fullinertia` 主轴不是
`ixx<iyy<izz` 排列，于是姿态整体偏转 90°，策略会把直立判成翻倒（MODEL-03）。

### 6.2 ROS / Gazebo

当前8cm八阶楼梯：启动时加 `wname:=stairs_8cm`（默认仍为earth），已安装并通过无界面加载；命令与验证范围见 [楼梯记录](docs/maintenance/project_history_2026-09_to_2026-10.md#ppo-training)。

源码分工、启动顺序、控制/推理循环、45维输入和关节PD的阅读入口见 [Gazebo sim2sim代码指南](docs/deployment/sim2sim_gazebo_code_guide_2026-10-12.md)。

PPO前后速度缓变的自动无界面采集与绘图入口、实际曲线和CSV归档见 [2026-10-16记录](docs/maintenance/project_history_2026-09_to_2026-10.md#simulation-and-deployment)。

2026-09-17 已把这条链路从上游的 ROS 1 `liblegged_hw_sim.so` 改为**本机可用的 ROS 2
`gazebo_ros2_control`（Humble + Gazebo 11）**，并在无头下单次调用内跑通到策略闭环
（详见 [Gazebo 链路打通记录](docs/deployment/gazebo_ros2_bringup_2026-09-17.md)）：

```bash
cd imgo2_deploy/ros2_ws

# 先用干净环境：若 shell 里 source 过 Isaac Lab / 有 conda，其 PYTHONPATH 会带进一份给
# Python 3.11 编的 numpy，把系统 Python 3.10 的 python `spawner` 搞崩（rl_sim 会报
# "Failed to start joint controller"，见记录第 6 节）。顺序：先清、再 source ROS。
unset PYTHONPATH PYTHONHOME            # 有 conda 时再加 conda deactivate
source /opt/ros/humble/setup.bash    # 顺序不能反：反了会连 ROS 自己的 python 路径一起删掉，
                                     # ros2 会报 PackageNotFoundError: ros2cli

bash build.sh                          # 或 colcon build --merge-install --symlink-install
source install/setup.bash

# 终端 A：Gazebo（wname:=stairs 换世界，gui:=false 无头）
ros2 launch imgo2_deploy gazebo.launch.py

# 终端 B：策略节点；键盘 0 起身 → 1 PPO / 2 himloco / 3 AMP，9 下蹲，P 回 Passive
ros2 run imgo2_deploy rl_sim
```

要点（都是实测踩出来的，改动理由见记录第 3 节）：

- 模型侧新增 `<ros2_control>`（12 关节 effort）与 `libgazebo_ros2_control.so` 插件，控制器配置在
  `imgo2_description/config/robot_control_ros2.yaml`；控制器本体是仓库内的
  `robot_joint_controller/RobotJointControllerGroup`（effort 接口），由 `rl_sim` 自己 spawn。
- `gazebo_ros2_control` 0.4.x 的 `<parameters>` **必须是真实文件路径**（给 `package://` 或相对路径都会
  在 Load 里抛异常）；launch 因此把 URDF 文本里的 `package://imgo2_description` 换成实际 share 路径。
- URDF 传给 controller_manager 时必须**压成单行、去掉 XML 声明**，否则 rcl 报
  `Couldn't parse parameter override rule`，controller_manager 起不来。
- Gazebo 版 URDF 的网格用 `package://imgo2_description/meshes/...`（`xacro/robot.xacro` 的
  `mesh_prefix` 属性按 `gazebo` 开关切换）；纯 URDF 仍是 `../meshes/`，不受影响。
- 当前ROS2 launch默认模型为 `imgo2_real.gazebo.urdf`；`IMGO2_GAZEBO_URDF` 可覆盖为其它已核对模型。本轮测试显式指定real，与当前训练策略一致。测量脚本加 `--sim-time` 使用仿真时间并均匀采样关节，速度误差按身体坐标统计，详见 [29000 轮实测](docs/maintenance/project_history_2026-09_to_2026-10.md#amp)。
- 不要用 `gazebo_ros` 自带 launch：本机实测它起的 gzserver 未加载 factory 插件（`/spawn_entity`
  不出现），我们的 launch 直接起 `gzserver -s libgazebo_ros_init.so -s libgazebo_ros_factory.so`。
- `imgo2_description` 现在是工作区里的标准包（软链进 `src/`，`package.ros1.xml`/`package.ros2.xml`
  由 `build.sh` 替换 `package.xml`），`imgo2_deploy` 的 `install(DIRECTORY ...)` 会把
  `config launch worlds policy` 装进 share。

**未验证**：GUI 画面（本机 `DISPLAY=:1` 建 GL 上下文失败，`gzclient` 大概率渲染不了，需要可用
显示或 `LIBGL_ALWAYS_SOFTWARE=1`）、步态观感、真机。ROS 1 分支仍是代码支持路径，本机没有 ROS 1，
未核验。

**怎么判定"它在走"**：必须同时看**基座高度与位移**（p3d 真值 `/odom`，已加回模型：`ros2 topic echo /odom --field pose.pose.position --once`）。只看姿态/关节会被"四脚朝天也很稳"骗过。实测两条链路一致：

| | MuJoCo | Gazebo |
|---|---|---|
| `vx=0` 基座 z / 位移 | 0.327 m / 0.023 m | 0.333 m / −0.001 m |
| `vx=0.5` 基座 z / 速度 / 大腿摆幅 | 0.26 m / **0.51 m/s** / 0.88 rad | 0.276 m / **0.502 m/s** / 0.89–1.20 rad |

**速度指令在 ROS 路径有三个来源**：手柄 `/joy`（`axes[1]`=vx、`axes[0]`=vy、`axes[3]`=yaw）；键盘 `W/S/A/D/Q/E`（要求 `rl_sim` 的 stdin 是终端，否则 `kbhit` 收不到）；先按 `N`（手柄 `X`）打开 navigation mode 再用 `/cmd_vel`——**默认 OFF，此时给 `/cmd_vel` 完全无效果**（`rl_sim.cpp:465`）。无手柄时可用
`ros2 topic pub -r 20 /joy sensor_msgs/Joy "{buttons: [0,0,0,0,0,1,0,0,0,0,0], axes: [0,0.5,0,0,0,0,0,1]}"`（键 1 + vx=0.5）。另注意键 `2`=himloco（Go2 占位，会把机器人掀翻）、键 `3`=AMP（只站不走）。

### 6.3 真机入口

```bash
bash build.sh --cmake
```

该模式用于硬件部署构建。`src/imgo2_deploy/library/thirdparty/robot_sdk/unitree/unitree_sdk2/` 中需要实际 SDK 源码；缺失时 CMake 跳过 `rl_real_imgo2`，所以整体构建结束不等于真机程序已生成。

后续需验证 Imgo2 实际电机通信接口与 SDK 的对应关系、关节方向和零位、状态机切换、数据记录，再形成真机实验记录。当前没有证据可将此阶段标记为完成。

### 6.4 独立电机与 IMU 工具（go_imu）

**本机手柄路径（2026-11-13 已核对）：** 默认 `/dev/imgo2_joystick` 别名不存在；当前 Flydigi 可用现有 `--controller-device` 覆盖，指定已枚举的 by-id `event-joystick` 路径，实际短时启动通过。命令及另一台电脑的路径核对要求见记录（本机 `gamepad_device_path_2026-11-13.md`）。使用手柄时不要带 `--keyboard`。

**本机常用 build（2026-11-13 已修复）：** `imgo2_deploy/real/build` 已按本机路径重新配置，可以进入后直接 `make -j2`；无设备运行 `./fsm_check --keyboard`。旧 build 完整保留为 `build_backup_2026_11_13`，不要当作当前程序运行；缓存失效原因与跨机首次配置见记录（本机 `real_build_cache_2026-11-13.md`）。

**当前 real 配置（2026-11-13）：** `input_source: 2` 为手柄 A/B/X/Y 站起/趴下/俯仰切换/PPO；首次及卸力后站起、趴下均 4 秒，PPO 回站 2 秒，俯仰回站独立 1 秒，原交接 0.5 秒；固定 PD 60/2、PPO PD 25/0.8，映射已设 `true` 且为恒等排列。主控/推理配置周期为 5/20 ms；实际加载可执行文件旁的 YAML/策略，支持命令行覆盖路径。以下带日期的旧记录保留历史背景，当前值以本段及[俯仰实现记录](docs/deployment/pitch_control_2026-11-13.md)为准；用户确认原三状态逻辑已测试，本轮新增俯仰仅做离线/伪硬件验收，未运行真实硬件。

**俯仰与无设备测试：** `fsm_check --keyboard` 可覆盖默认手柄输入，不需修改 YAML；0+回车站稳后，3+回车进入俯仰，D 仰身/A 俯身/空格保持，0或3+回车用1秒回站。手柄 X 进入/再次 X 或 A 回站，右摇杆水平只取方向，固定10°/s；俯仰和回站途中都禁止直接进 PPO。几何表由唯一模型 `imgo2_description/urdf/imgo2_real.urdf` 导出，改模型后运行 `python3 imgo2_rl/scripts/tools/export_pitch_geometry.py --check`，需要更新时去掉 `--check`；不手改生成表。完整构建命令、数值验证及动画（本机 `pitch_control_preview_2026-11-13.gif`）统一见[俯仰记录](docs/deployment/pitch_control_2026-11-13.md)。

**在另一台电脑迁移相似实现：**使用[fuyang模式.md](docs/deployment/fuyang模式.md)，先核对目标工程的关节/坐标/模型和接口，再按职责合入；本机工作树尚未提交推送，不能假设另一端 `git pull` 已取得功能。

**状态与姿态参数最新变更：**全部具体状态统一在`fsm/states/`：Stand/Passive/PPO/Hold/Fault各自直接继承State，AMP仅抽象接口且不注册。独立go_stand与部署入口共用唯一StandState；旧fsm/stand_state与MotionState套层已删除。`config/deployment.yaml`中的`stand_pose`、`crouch_pose`均可按硬件顺序独立填写12项rad；9＋回车趴下后确认实际到位再卸力，不再60/2保持。31/31离线回归通过，说明及待确认条件见 [记录](docs/maintenance/project_history_2026-09_to_2026-10.md#simulation-and-deployment)。

**离线键盘FSM测试：**运行 `./imgo2_deploy/real/build_layout/fsm_check`，0/1/2占位/9需回车，WASDQE即时；首行状态、四腿目标原地刷新。Stand/Passive模拟跟随轨迹，PPO固定站姿+零IMU反馈，复用正式状态和SDK，不开设备、不改真机映射锁。运行命令及零输入偏差见 [测试指南](docs/deployment/fsm_check_2026-10-15.md)。

**PPO重复切换：**0从PPO回Stand使用`return_stand_s: 2.0`，PD在`blend_s: 0.5`内同时恢复60/2；轨迹结束后反馈连续稳定`settle_s: 0.5`才接受重新输入1，约需至少2.5秒。提前1提示未就绪、不排队；当前首次/Passive后站起为4秒，Fault不能用0恢复。调用链和其他检查见 [历史复审](docs/maintenance/project_history_2026-09_to_2026-10.md#simulation-and-deployment)。

**实际操作先看 [当前控制流程与键盘手册](docs/deployment/go_imu_control_flow_2026-10-15.md)**：已逐项补齐Motor/IMU/Hardware/Deployment/Policy参数地图、默认值/覆盖来源、各状态初始化、硬件启动与每周期流程；同时列出工程内build命令、键盘、Fault和退出行为。

**当前入口：**旧ppo_check、imu_reader、go_probe已删除，离线测试统一用fsm_check；真实电机/IMU诊断用go_monitor。新程序首次CMake配置需`-DGO_IMU_BUILD_RL=ON`并指定LibTorch，同一build已缓存后直接build即可；不需要ROS或额外测试宏。建议先离线0→1→9→0，再预检和真实固定姿态验证，见 [编译与测试次序](docs/maintenance/project_history_2026-09_to_2026-10.md#simulation-and-deployment)。

**手柄输入核对（2026-10-23）：** 新增 `controller_demo`（只读 `/dev/input/event*`，不碰电机/策略）：打印设备能力与每个按键/轴的 evdev 名称、编号、原始值与范围，用于确认 Xbox/黑武士4 Pro 在 2.4G 下的真实映射。`--list` 列设备，`-d N/路径/名字` 指定；默认摇杆值在最后一行原地刷新、按键另起一行，`--events` 退回逐行；普通用户需 `sudo` 或加入 `input` 组。新增 target 后先 `cmake -S imgo2_deploy/real -B imgo2_deploy/real/build` 再 `cmake --build build --target controller_demo`。见 [手柄核对](docs/deployment/xbox_controller_demo_2026-10-23.md)。

**手柄控制（2026-10-23）：** `input_source: 1|2`（1=键盘，2=手柄） 在 `config/deployment.yaml` 选输入源；设备名/死区在 `config/real_config.h` 的 `ControllerConfig`（默认 `/dev/imgo2_joystick`、死区 0.08），两个入口都可用 `--controller-device` 临时覆盖。映射：左摇杆前后/左右 → vx/vy（上/左为正），右摇杆水平 → wz（右为正），整段行程映射到 `velocity_limits`；A/B/Y → 站起/趴下/PPO，只在按下沿触发，状态切换仍走 `RobotFSM::request()`。摇杆只在 PPO 生效；断线清零速度并在 `rl_real_imgo2` 里每秒重连。验证见 [手柄控制](docs/deployment/gamepad_control_2026-10-23.md)。

**新增独立真机部署入口（2026-10-15）：[结构、参数、运行和故障指南](docs/deployment/go_imu_sim2real_2026-10-15.md)。** `sdk/`负责45维PPO与异步推理，`fsm/`负责0/1/2占位/9，`hardware/robot_adapter`负责唯一关节排列。用户原关节宏、策略与参考入口未改；新入口固定增益60/2、策略25/0.5，初次/Passive站起与趴下8秒，PPO回站2秒。PPO映射默认未确认，实机验收前保持锁定。

当前 CMake 默认 `GO_IMU_BUILD_RL=ON`、`GO_IMU_BUILD_APPS=ON`，旧 build 缓存可覆盖默认值。启用RL需要兼容C++11 ABI的LibTorch和yaml-cpp；本机已有LibTorch路径的构建示例：

```bash
cmake -S imgo2_deploy/real -B imgo2_deploy/real/build_layout \
  -DCMAKE_BUILD_TYPE=Release -DGO_IMU_BUILD_RL=ON \
  -DCMAKE_PREFIX_PATH="$PWD/imgo2_deploy/ros2_ws/library/inference_runtime/libtorch"
cmake --build imgo2_deploy/real/build_layout -j2
./imgo2_deploy/real/build_layout/rl_real_imgo2 --check-policy  # 不打开硬件
```

实际硬件入口为同目录 `rl_real_imgo2`（本轮未连接真实硬件运行）；不要与ROS包中的同名参考入口混淆。工程复制到仓库外独立编译、安装和移动资源的验证通过；第三方共享库仍需单独提供，详细依赖、安装及外部链接方式见上述指南。

底层接口整理的历史依据见 [五阶段整理记录](docs/maintenance/project_history_2026-09_to_2026-10.md#simulation-and-deployment)，当前入口以本节及入口精简记录为准：MotorConfig/ImuConfig构造函数提供默认值，使用前可改字段；Motor执行目标限幅，站起轨迹越界/超速启动前拒绝。当前五个入口直接写main，控制工具无文件日志、无周期刷屏。关节参数宏已按用户要求恢复，构造函数引用；RunXXX已删除，旧RobotConfig/工厂和硬件接口保留。

`go_stand`现使用`fsm/`内State基类和Stand状态；启动保持→0+回车→8秒五次轨迹→站姿保持，时长/增益在StandSettings设置。命令、文件职责和验证见 [FSM指南](docs/maintenance/project_history_2026-09_to_2026-10.md#simulation-and-deployment)。

2026-10-14 已按用户常量启用12关节逻辑限幅，不额外加offset；原冻结版本已有此次授权更新，须重新编译。udev与monitor实测已由用户确认，新限幅站起仍待实机验证，见 [配置与验证](docs/maintenance/project_history_2026-09_to_2026-10.md#simulation-and-deployment)。

**明天测试入口：[完整使用与故障手册](docs/deployment/go_imu_usage_2026-10-11.md)**，包含全部源文件职责、运行链路、默认配置、操作顺序、错误条件与退出行为。udev和monitor实测用户已确认；本次授权更新包含限幅及站立FSM，需重新构建并确认新的实机站起，可在 `/tmp` 构建。

从仓库根目录构建：

```bash
cmake -S imgo2_deploy/real -B imgo2_deploy/real/build_layout -DCMAKE_BUILD_TYPE=Release
cmake --build imgo2_deploy/real/build_layout -j
ctest --test-dir imgo2_deploy/real/build_layout --output-on-failure
./imgo2_deploy/real/build_layout/go_monitor
```

`go_monitor` 默认同时监测电机与 `/dev/robot/imu`（460800），无需 IMU 参数；设备和初始化参数统一修改 `config/real_config.cpp` 后重新编译。电机区下方显示 RPY／角速度／加速度、数据年龄及接收帧率，并在终端原地刷新。宽终端显示完整电机信息，普通终端使用紧凑的 qOut/qL；空间不足时提示扩大窗口。IMU 故障不停止电机监测。

同一build目录保留`go_monitor`、`go_pd_demo`、`go_stand`，另有[独立位控采集`go_position_test`](docs/deployment/position_test_2026-11-18.md)（无需RL，默认仅预览，硬件模式会控制12轴，实机待验收）；启用RL后另有`rl_real_imgo2`和`fsm_check`。monitor发送零指令请求卸力，demo/stand会主动控制；同一串口不要同时运行多个硬件程序。全部离线测试通过不代表真实电机或 IMU 验证完成；运行条件和历史合并依据见 [合并记录](docs/maintenance/project_history_2026-09_to_2026-10.md#simulation-and-deployment)。

整机接口 `RealHardware` 提供 `start/readState/sendCommand/stop`；其他 C++ 工程可通过 `add_subdirectory()` 链接 `go_imu::hardware`，不引入终端显示。电机控制与诊断均每条总线一个线程，总线内部按 ID 请求应答；诊断缺电机时继续扫描，控制保持严格启动与故障锁定。整机接口加入 IMU 就绪检查和反馈过期卸力，数据恢复不会自动恢复运动。初始化按原启动姿态归零，不能站起后重启再次归零。2026-10-15新增SDK已提供策略顺序排列和投影重力，物理映射与四元数实测仍待确认；原硬件接口用法、17 项历史测试及限制见 [重构记录](docs/maintenance/project_history_2026-09_to_2026-10.md#simulation-and-deployment)。

## 7. 待办与已知问题

| 本次新增问题 | 优先级 | 状态与完成条件 |
|---|---|---|
| RESEARCH-LEARN-01 | P2 | **文献书单与学习路线已整理，复现方向待选**。见[机器人强化学习论文路线](docs/research/robot_control_rl_reading_route_2026-11-24.md)。需确定每周时间、GPU/训练预算及下一阶段方向；正式复现前核对目标论文的代码、数据与依赖。本轮未运行论文实现，不改变既有训练和实机验收状态。 |
| PPO-ROLL-TRAIN-01 | P1 | **本机代码已撤回 7.5 Hz 训练扰动，实机止抖仍待验证**。11/23 实机已按逐行输出确认原 8000。历史续训 9100 在同帧实机输入下的髋目标波动更大。待区分触地、关节响应与姿态反馈的起振作用。见[最新实机复核](docs/xiaoji_cmd/中期/实物实验/原8000实机振荡复核.md)。 |
| DEFENSE-EVIDENCE-01 | P2 | **正文分页与现有素材对应已完成**。正式封面信息、个人承担工作和与日志对应的实物照片／视频待选定。实验验收沿用 DEPLOY-13、ACT-NET-01、MODEL-05、PPO-STAIR18-01。见[逐页安排](docs/xiaoji_cmd/中期/中期答辩分页.md)。 |
| PARALLEL-LEG-01 | P2 | **髋部双电机五连杆加髋关节已确认，几何与实现待确认**。仍缺轴线/杆长/装配支路、惯量与电机参数及实际 Isaac Sim/Lab 版本；需单腿闭环/受载验证、四足 reset 与 GPU 多环境检查，再做短训，见[路线](docs/training/parallel_leg_isaaclab_route_2026-11-18.md)。本轮仅静态与官方资料核对，未改训练/模型或运行仿真。 |
| PPO-SIM-QUEUE-01 | P2 | **Gazebo PPO分离结果队列与未消费tau队列待修**：位置/速度分别push，消费者pos.try_pop && vel.try_pop可在交错时丢掉位置；诊断tau队列持续push未见pop。建议单个带序号原子结果和有界诊断消费，需编译/并发及运行验证，见源码可达路径及限制（本机 `ppo_input_attribution_2026-11-18.md`）。绕过这些队列的历史obs回放仍抖，未认定为当前根因；本轮未修生产代码。 |
| MODEL-TRANSMISSION-01 | P2 | **用户确认部署角度映射已修复、电机壳体固定大腿；四杆惯性等效待核对**。core 与两份 real URDF 的 17 个共同 link 惯性定义静态一致，但缺逐零件 CAD 质量分组/运动关系，不能确认活动推杆是否已正确计入聚合值，见[记录](docs/training/parallel_leg_isaaclab_route_2026-11-18.md#同日补充用户已修部署映射四杆质量应按运动关系归属)。需明确固定件与活动杆，推导或验证质量/质心/惯量等效及受载响应；现有质量/PD随机化不等于活动传动惯性完整覆盖，见本轮DR边界（本机 `ppo_domain_randomization_coverage_2026-11-09.md`）。未判参数错误或抖动根因，模型与控制未改，无新仿真/硬件。 |
| PPO-STAIR18-01 | P1 | **5 cm、8 cm 三阶演示已有双引擎结果，18 cm 路线仍待验证**。11/23 从固定工况补采各四段，详见[楼梯补采](docs/research/defense_stairs_2026-11-23.md)。11/22 用 6000 策略完成 MuJoCo／Gazebo 的 5 cm 三阶上下行素材采集；MuJoCo 8 cm 对照停在首阶附近，见[采集记录](docs/research/defense_media_2026-11-22.md)。11/22[附件配置复核](docs/training/pasted_rough_config_review_2026-11-22.md)仍只覆盖2.5–8cm台阶，18cm待任务与专项验收。目标为18cm高/30–40cm深连续两阶稳定上下楼及平地直行/转弯；建议6999新分支混合课程，现有2.5–8cm和45维盲走不构成能力证明。缺两阶几何/动作与力矩可达性检查、感知选择、任务实施/短训及独立上下楼和平地验收，详见[路线](docs/training/ppo_18cm_stairs_route_2026-11-13.md)。本项 18 cm 专项尚未实施任务、训练或运行验收。 |
| PITCH-PORT-01 | P2 | **迁移说明已完成，目标电脑集成待核对**。用户计划在另一台电脑修改相似代码；已补[原理与移植说明](docs/deployment/fuyang模式.md)，目标源码/模型/关节接口/手柄映射和依赖尚未提供，需当地集成并通过运动学与原功能回归，不能沿用本机测试数作为对方通过依据。 |
| PITCH-HW-01 | P1 | **俯仰已实现并经离线/伪硬件验证；实机待验证**。已接入 X/右摇杆、−20°～+30°、1秒回站和 PPO 门禁；固定足端 URDF FK 通过，详见[记录](docs/deployment/pitch_control_2026-11-13.md)。缺真实设备，需核对承重姿态/中心偏差、打滑/机械与力矩裕量、全范围及实际摄像头画面；现为运动学位置控制，未新增 IMU 姿态闭环，不把动画当作实机验收。 |
| REAL-DOC-01 | P2 | **当前配置说明已修并静态核对；源码旧注释/提示待处理**。部分源码注释仍写旧手柄映射/PD，Fault 提示暗示 0/9 可恢复但实现锁定；本轮遵照用户要求不改 real 文件。后续需获准维护这些文字并核对帮助/提示与实际状态行为，见复审（本机 `real_control_review_2026-11-13.md`）。 |
| ACT-NET-01 | P1 | 训练原 DCMotor 可作基线，带载动态未匹配。 | 大幅台架加阻力改善约 71%，小幅 5ms 数值验收及整机横滚复现未过。待分辨阻力及带载幅相，并独立验证。见[执行器验收](docs/deployment/ppo_vibration_status_2026-11-22.md#actuator-validation)、[原生摩擦](docs/deployment/ppo_vibration_status_2026-11-22.md#actuator-validation) 12.5ms 固定目标延迟待实测，策略节拍抖动已核对。 [时序复查](docs/deployment/actuator_timing_review_2026-11-22.md) |
| DOC-CLASS-01 | P2 | **分类与现存目标引用已修并验证；历史缺失产物待确认**。docs 根目录散文件已归类、数据字节一致、未新增目录或失效链接；剩余目标缺原始实验归档或替代关系，详见 [整理记录](docs/maintenance/project_history_2026-09_to_2026-10.md#model-and-workspace)。 |
| TOW-07 | P2 | **待修复**：残差回放统计误标 rad、`clip_actions` 放大权限的说明与固定 ±1 裁剪不符；修正后需验证日志单位和动作边界。详见 [同步核对记录](docs/maintenance/git_sync_2026-10-08.md#第二次同步c321c14--c49be10) |
| TOW-08 | P1 | **地形已被同日后续的连续剖面取代，待训练机验证**：跑道 15 m，每条 lane 现在是「平地 2.25 m → 上坡 3 m → 坡顶 0.75 m → 下坡 3 m → 平地 2.25 m」（坡度量级 0/5/10 由列决定，仍 800 条 lane；10 m 的 STOP 点落在出口平地上，**越过坡后才置零指令**），出生在平地段、跌倒判据用离局部剖面高度、timeout 用坡面弧长（28.23 s / 564 步）；需 Isaac Lab 构造 + 短回放覆盖平地/5°/10°、上坡与过顶、**STOP 触发**、停车段两条奖励与超时。见 [连续坡面剖面记录](docs/towing/towing_profile_terrain_2026-10-09.md)（旧恒定坡度设计见 [构建记录](docs/towing/towing_slope_goal_vae_2026-10-09.md)） |
| TOW-09 | P1 | **已修，待训练机验证**：16 维 variational latent 经 decoder 对齐显式 6 维，actor 79 维；保存 rollout 起始隐状态、KL 日志与 checkpoint 契约。需运行网络测试和短训练，确认有限梯度、done 重置、估计精度及 KL 是否导致 latent 塌缩。见 [构建记录](docs/towing/towing_slope_goal_vae_2026-10-09.md) |

优先级含义：P0 为影响训练启动或部署正确性的前置问题，P1 为后续验证与维护事项。

| ID | 优先级 | 状态 | 问题与依据 | 完成标准 |
|---|---|---|---|---|
| HIM-LOG-01 | P2 | 源码已确认，待修复 | HIMPPO.update已计算估计/交换loss均值，返回值仍用末minibatch，影响日志读法。见[HIM训练说明](docs/training/himloco_training_guide_2026-11-05.md#同日补充target逐段解释完整问题边界与课程) | 返回均值并验证runner记录一致；未证明是运动异常成因 |
| HIM-TRAIN-01 | P1 | 已复现，待修 | 11/3CPU实际执行估计器next_obs及wrapper方法：切片3:48混入特权速度、done后旧历史保留、reset只返45维且未清270维缓存；critic前缀还存在q/dq排序与缩放差异。 | 仅改HIM训练/配置，先选定与独立部署yaml对应的物理契约，再修target与历史边界并跑输入/输出一致性和短训；见 [复审](docs/deployment/himloco_training_contract_2026-11-03.md)。未证实旧部署权重由当前源码训练 |
| HIM-SIM-01 | P1 | 双引擎短测完成，原训练接口待确认 | 11/3沙箱外两批Isaac及8组Gazebo完成；现有HIM在当前Isaac和Gazebo均不正常跟速，policy_1能前后行走。已核对真实键2/六帧路径，未改权重，临时PPO配置恢复；见 [实测](docs/deployment/himloco_ppo_comparison_2026-11-03.md)。 | 用户确认仅有policy，缺原训练配置/checkpoint；不能完整复现他人Isaac正常结论。默认角/PD/缩放/关节映射与当前源码不同，需来源资料或接口重建后做严格对照；未证实某一差异是唯一成因 |
| TB-PKG-01 | P2 | 用户已展示TensorBoard页面，wheel/Ray与完整依赖检查待验证 | packaging23满足Sim，但wheel0.46.3/Ray2.54要求>=24；已核对并提供wheel0.45.1/Ray2.52.1联合安装命令，见 [指南](docs/training/policy_evaluation_guide_2026-10-26.md#同日后续wheel与ray要求新版packaging) | 原isaaclab环境pip check通过、pkg_resources导入及TensorBoard启动成功；本轮未访问或更改服务器环境 |
| PPO-ENTRY-01 | P1 | 仿真PPO入口目标/增益交接待实现与闭环验证 | 公共PPO FSM收到目标后直接RLControl赋值；Gazebo外层现有交接仅识别AMP，MuJoCo未发现另一个PPO入口交接；独立真机已有blend_s=.5s，不能混同。默认角六处一致，最新策略静态首拍后髋目标±21.49°，见 [方案与证据](docs/maintenance/project_history_2026-09_to_2026-10.md#ppo-training) | 仿真PPO等待首个有效结果后保持指令连续、目标/PD可配交接与旧队列隔离；重复站起/PPO实跑并记录每拍目标/反馈/增益。平滑不能代替训练姿态修复；本轮只有源码/静态推理核对，未改控制或运行切换 |
| PPO-FHB-01 | P1 | 实现与名称查询性能已修，Isaac计时/训练效果待验证 | 11/2按用户要求注释保留旧feet_height_body，新函数仅摆动脚参与、世界水平足速加权、按足端名称对齐；10个CPU案例和完整回退检查通过。11/2当时rough权重为0；11/10当前分支−4已在8环境2轮短训中启用，步态效果仍待。 | 可用Isaac环境中构造/短训与净空、机身高度及转向评估；不能将代码改动视为抬脚已改善，见 [记录](docs/training/feet_height_body_swing_2026-11-02.md) |
| PPO-GAIT-01 | P1 | 11/22 平地低抬脚已验证，步频与横滚尚未同时改善 | 11/23 最新批次 8000 轮在平地有横向偏移，Gazebo 粗糙工况在进入障碍前倾倒，待同条件复测启动及过障。[采集记录](docs/research/defense_ppo8000_data_2026-11-23.md#结果)。11/22 从零 5000/6000/7000/8000/9999 同条件回放，5000 平地前进跟速和步频较好，6000 粗糙地形跟速最佳。实机及 18cm 两阶待验收。[摘要](docs/training/pasted_rough_config_review_2026-11-22.md#当天回传的训练结果)。11/11两新末checkpoint已复审，112模型有限，两正式导出+18新MuJoCo/两12sGazebo完成；续训9299脚更高、roll波动有代价，从零6999直行roll更小但右前脚9.23cm/姿态偏置/跟速仍待，见最新对照（本机 `nov11_two_runs_review_2026-11-11.md`）。 11/10新13:31/9100实际为混合地形−.25/.3；新18MuJoCo含原5300同25/.8重跑，新12sGazebo完成。步频提高而+.5净空仍7cm，前后脚不一致，横滚未明显改善；上传未到计划末段，见最新复审（本机 `nov10_h25_air30_review_2026-11-10.md`）。 11/10已准备5300平地独立分支（目标−.25、air阈值.20），恢复原权重；新/对照各8环境2轮通过，1000轮与净空/步频/姿态验收未运行，见新参数与命令（本机 `low5300_retrain_2026-11-10.md`）。 11/10最新air1.2/9299新9MuJoCo+12sGazebo完成：+.5步频1.905Hz/净空6.98cm，较基准更快更低，roll角速度降但角度波动/俯仰/后退有代价；已完成4000轮，未部署或实机验收，见最新对照（本机 `nov10_air12_review_2026-11-10.md`）。 11/10新续训10500已正式导出并新39MuJoCo/12sGazebo；目标步频/净空变化小，前进角速度降但roll/后退/右转有代价，中期10000另有取舍，实机/rough仍待，见新复审（本机 `nov10_continuation_review_2026-11-10.md`）。 11/10已确认两9299均从5300平行续训，可改奖励新建分支/从头训；尚未实施或验收，见说明（本机 `ppo_gait_continuation_motor_review_2026-11-09.md`）。 11/9步频/足高续训方案已整理但未实施：2.2–2.5Hz/4–6cm为本机平地候选，非产品标准；DC峰值接近不验收全动态，保存/当前配置须分辨，见建议与来源（本机 `ppo_gait_continuation_motor_review_2026-11-09.md`）。 11/7新−.23组：32MuJoCo/32Isaac，前进.5角运动.208→.134、净空9.14→8.31cm，后退-.3同步与跟速退步，rough仅小幅提升，见[最新复审](docs/deployment/height23_review_2026-11-07.md)。11/6旧−.20组：新32MuJoCo/44Isaac，rough vx MAE .2235→.1024，但楼梯/boxes前进慢、平地前进身高/角运动有代价，见[最新复审](docs/deployment/terrain5300_review_2026-11-06.md)。此前直接对照policy_1：32MuJoCo/4Isaac，6999前进.5/.8/1m/s高度std增加至1.66/2.64/2.00倍，Isaac+.3角运动约+44%；静止/后退及部分方向有收益。见[直接对照](docs/deployment/policy1_vs_scratch6999_review_2026-11-06.md)，[中期/最终](docs/deployment/grouped_scratch_review_2026-11-06.md)保留历史。 | 待5100后退前后偏载/对角时序因果、必要新依赖纳入同步、rough配置/分组默认一致性、双起点rough续训及平地保留验证、垂直弹动/角运动因果与修法、身体高度/跟踪、多初态、动态/FSM、服务器源码、Gazebo/rough/硬件；无生产替换，当前仅完成有界启动短训。原低速5400/5500请求仍待补测。 |
| LOG-CLEAN-01 | P3 | 旧失败权重清理候选已核对，实际范围待决定 | 六组目录约1.67GiB、根checkpoint约1.56GiB；无生产静态引用/保存agent显式父路径命中，但历史docs/JSON有引用，见[清单](docs/maintenance/logs_cleanup_candidates_2026-11-06.md)。 | 尚未删除；先确定清权重或整目录、是否留负例，执行前复核最终文件/保留路径，执行后注明历史原始文件已清理。 |
| PPO-ROUGH-01 | P1 | 已见含两奖励修复的服务器训练产物，步态/统计待验收；倒地终止待处理 | 高度/脚滑相邻注释旧版与14 CPU案例见 [修改记录](docs/training/velocity_reward_fixes_2026-10-31.md)。11/2回传新run保存diff包含修复，事件已至1999，说明已有可运行训练产物；但新策略步态未达标，见 [最新复核](docs/training/ppo_low_speed_1999_review_2026-11-02.md)。 | 缺实际射线有效率与Isaac接触/净空时序，不能量化两修复单项效果；出生点高度回退/world xy滑移仅为首轮近水平近似。复杂地形高度/坡地切向、base接触终止、新净空项及方向/rough/台阶验收仍待处理，不把CPU或高总reward当训练/实机达标 |
| GAZEBO-TERRAIN-01 | P2 | 多难度观察场加载已验证；GUI/性能与策略通过待确认 | terrain_course三档12模型/408碰撞、SDF及无界面实际查询通过，本机已安装，见[记录](docs/deployment/gazebo_terrain_course_2026-11-06.md)。 | 需要GUI观察、实时率与机器人ROS/策略在楼梯/坡/块面上的通过和跌倒统计；不是Isaac表面逐点复刻，本轮未运行策略。 |
| GAZEBO-STAIRS-01 | P1 | 地形已验证；当前PPO双引擎登阶失败已复现 | Isaac平地正常，8cm楼梯Isaac/Gazebo均停在首阶前，max x1.317/1.338；策略来源哈希已核对，原训练任务ID待原run确认，见 [实测](docs/maintenance/project_history_2026-09_to_2026-10.md#ppo-training) | 完善rough前置/配方并训练或微调，以实际登阶完成率验收；本轮没有训练，不把共同失败归为仅Gazebo转换问题；GUI未测 |
| PPO-RAMP-01 | P1 | 已完成变速采集，航向与低速跟踪待查 | 当前PPO/real模型50秒扫描无倒地、Roll/Pitch std约0.43°/0.39°；yaw指令0但累计−25.2°，升速早段实际速度滞后，见 [记录与原始数据](docs/maintenance/project_history_2026-09_to_2026-10.md#simulation-and-deployment) | 重复相同条件并做固定低速/固定速度对照，结合实际控制反馈定位；本轮未改策略或接触参数，不代表真机验证 |
| DEPLOY-MJCF-PD-01 | P1 | 已修并验证；长时表现待确认 | 逐物理步PD、整命令发布、回调/线程退出已实现；正式组件Passive dq RMS约.051、3/3 CTest和两组GUI状态流程通过，见 [实现](docs/deployment/mujoco_pd_feedback_2026-10-23.md) | 长时间GUI步态/实际频率与热重载仍待专项验证；保持0/8、60/2、25/.5，不据此判定实机根因 |
| MODEL-MJCF-01 | P1 | Gazebo模型参数已同步并验证；跨引擎动力学待确认 | 8body惯性及固定IMU已同步，总重12.96640984kg；.5ms、被动0/0、滑动摩擦.2/.6，静态/10项回归/21姿态FK/三场景6次保持通过，见 [结果](docs/deployment/mjcf_gazebo_sync_2026-10-23.md) | PPO步态、生产FSM启动、PD/观测时序、自碰撞与接触求解/执行器包络仍待确认，不能把模型检查当sim2real完成 |
| DEPLOY-MJCF-01 | P1 | 10-23复审：接口/时序与依赖问题待修 | PPO静态接口和CPU形状检查通过；MuJoCo临时配置因缺库失败，AMP契约/并发快照/队列/退出和PD时序仍需处理，见 [复审](docs/deployment/mujoco_sim2sim_review_2026-10-23.md) | 当前已发现3.2.7/GLFW和可执行文件，用户报告启动；逐步PD/状态快照交接和退出已修，PPO仅短流程；仍需长时步态、AMP契约/旧目标队列及全面并发审查；cmake_build_mujoco产物忽略规则待处理 |
| AMP-CURRICULUM-01 | P1 | 分阶段低速方案已修订，尚未实施/训练 | 低速门限、纯轴抽样、静止yaw/平滑约束、静止专家仅约4.76%需一起设计，见 [修订](docs/maintenance/project_history_2026-09_to_2026-10.md#amp) | 实施独立profile和专家采样/静止项，短训练与固定指令验收后扩大范围，继续保留低速与零指令 |
| AMP-EVAL-01 | P1 | 统一诊断回放入口已实跑，GUI/生产play配置待确认 | 当前诊断入口支持恢复0、关闭噪声、机器人摩擦.6与实时播放；不改训练/原play默认。实际60步配置核对通过，见 [统一命令](docs/xiaoji_cmd/策略导出与ROS2仿真.md#统一-isaac-与-gazebo-的回放基线2026-10-14) | 用户GUI回放与不同策略回归；生产play是否沿用该基线、反馈/执行时序与Gazebo残差待确认。1ms物理步长试验倒地，保留训练步长 |
| DEPLOY-14 | P1 | 切换与持续目标变化已实测，整体抖动待修 | PD60/2→25/.5与首条目标跳变；新25000早期跳变.10–.62 rad，本轮当前旧24000站立.158 rad。渐变未一致降低速度峰，仍默认0；持续站立另见AMP-23。历史见 [复核](docs/maintenance/project_history_2026-09_to_2026-10.md#amp)，当前持续变化见 [对照](docs/deployment/amp_engine_comparison_2026-10-14.md) | 完整交接/初始化、接触/力矩/速度与GUI对照；日志精度已修已编译，10-14新精度实录已验证；不能将第一条目标平滑等同于持续抖动修好 |
| AMP-23 | P1 | 持续目标变化已双引擎验证，静止品质待改进 | 当前9c6默认Isaac/Gazebo目标变化RMS .073/.087 rad；接触/噪声对齐后Isaac .021、Gazebo仍.087，见 [对照](docs/deployment/amp_engine_comparison_2026-10-14.md) | 固定回放基线、静止/动作变化约束单变量训练，同步反馈时序与固定动作轨迹对照；本轮未改奖励/PD，不能将残余抖动全归因于奖励 |
| AMP-20 | P1 | Gazebo AMP策略裁剪/last_action已修并验证，其它接口差异待确认 | 先转换后clip目标、raw反馈，4182帧目标/last_action差0，actor回放差≤2.15e−6，构建/C++回归/四方向通过。controller仍额外裁剪URDF限位，MuJoCo/硬件未改，见 [修正对照](docs/deployment/gazebo_amp_clip_fix_comparison_2026-10-13.md) | 更广指令/瞬态的controller限位、执行器/反馈/时钟及其它后端对齐；不把局部修复视为全部sim2sim通过 |
| AMP-21 | P1 | 新25000修正后方向已双仿真短测，后退/组合品质待改进 | 命令−.4，Isaac/Gazebo实际−.196/−.218；修正前后速度/步态未明显改善。纯yaw±.5能跟速但抬脚很少；新数据前后符号/身体速度核对未见错误。见 [对照](docs/deployment/gazebo_amp_clip_fix_comparison_2026-10-13.md) | 多种子固定方向/组合回放、接触时序与跟速训练单变量对照；未验证的调权建议不标为收益 |
| AMP-22 | P1 | 当前新模型策略后退贴地/高频已双引擎实测，步态待修 | 对齐恢复/噪声/摩擦后Isaac前后周期.24/.26s，Gazebo约.28/.30s；后退足底p95分别约0–1.9mm和.6–9.1mm，见 [对照](docs/deployment/amp_engine_comparison_2026-10-14.md) | 固定评价基线后验证静止、足底高度、防滑与周期目标；先校正已有feet_slide语义，单变量训练/多种子/长回放。本轮不改奖励或新采专家数据 |
| AMP-19 | P1 | 续训说明已修并离线核对，服务器恢复待确认 | AMP需要resume门控与load_run/checkpoint定位；max_iterations为额外轮数，恢复产物写新run。见 [续训命令](docs/xiaoji_cmd/服务器训练.md#续训练从已有-checkpoint-接着训练) | 确认服务器原run、checkpoint内部iter与当前配置兼容；短恢复验证加载路径、轮号及新目录保存，本轮未启动训练 |
| DEPLOY-13 | P1 | 11/23 实机匹配后退已观察到 6–9 Hz 横滚下降，普适性待复测 | 同一 8000 在新 PD 下的实机改善见[三组对照](docs/xiaoji_cmd/中期/实物实验/实物实验说明.md)。 同策略/同主指令，实机髋和横滚波动显著高于两引擎。11/22 新几何平地闭环已测，末期 9999 摇摆主峰约 2 Hz。7000 新旧 base 惯量配对在同速前进只引起小幅横滚变化。MuJoCo 中 7.5Hz IMU 注入诱发横滚，撤除后衰减。优先补独立姿态参照，区分 IMU 测量残差与真实横滚，再核对时序和受载幅相；见本机[区分方案](docs/deployment/sim2real_diagnostic_plan_2026-11-22.md)。实际惯量、机械零位、接触、实际目标应用时标及无托止抖仍待测。见[排查摘要及证据](docs/deployment/ppo_vibration_status_2026-11-22.md) |
| MODEL-LAYOUT-01 | P1 | 模型自检存在迁移前的未登记资源/网格漂移，待确认 | 本轮check_model_sync报告网格14文件/66c496556453，登记为10文件/8dc5b5995a11；4份real/real_wrong URDF未登记。路径与ROS构建通过不代表模型契约通过，见 [迁移记录](docs/maintenance/project_history_2026-09_to_2026-10.md#simulation-and-deployment) | 用户确认新增模型/网格的用途与物理准绳，再更新模型登记及契约检查；本轮未改或删除模型，不能跳过检查标完成 |
| DEPLOY-LAYOUT-01 | P2 | 路径与原位重建/无界面启动已验证，策略运动/GUI待确认 | real/ros2_ws迁移修复后，2026-10-21旧缓存/软链已备份并原位colcon四包完成，source和Gazebo spawn/Passive控制器启动通过，见 [实际修复](docs/maintenance/project_history_2026-09_to_2026-10.md#simulation-and-deployment) | 新构建完整起身/策略运动、GUI按需验证；既有网格漂移及4份未登记URDF仍待模型契约确认，未启动硬件 |
| DEPLOY-IMU-01 | P2 | 200Hz IMU及gyro白噪声对照已实跑；源生成不一致与动态误差待确认 | 当前15/.5、base6kg，两组50秒PPO实测约200Hz，无倾倒停止；gyro.005对照结束零指令摆动更明显、两组偏航。正式文件未改，见 [结果](docs/maintenance/project_history_2026-09_to_2026-10.md#simulation-and-deployment) | Xacro仍100Hz、real生成URDF200Hz，需确认用户模型改动后同步源；未测姿态融合/延迟、横移/yaw和硬件，不据白噪声对照排除实机IMU |
| DEPLOY-IMU-02 | P2 | **旧右乘问题已修，源码/新日志一致；闭环与物理外参待验证** | Rebase后applyBias只改显示Euler、quat/gyro原样；新构建8/8相关回归通过，新实机动态相关性约.998。见源码复审（本机 `deployment_source_rebase_review_2026-11-08.md`） | 动态数据不能代替本体安装外参/融合精度标定；真实二进制/策略哈希及运动抖动根因仍待DEPLOY-DIAG-01/DEPLOY-13，不再使用整段quat右乘初始姿态归零 |
| DEPLOY-CTRL-01 | P2 | **手柄核对 demo 已实现；实机映射待确认** | `controller_demo` 只读 `/dev/input/event*`，打印设备能力与每键/轴的 evdev 名称、编号、原始值、范围；支持 `--list`/`-d`/`--state`，断线等待重连。编译、`-Wall -Wextra -Wpedantic`、FIFO 合成 evdev 端到端通过；本机无手柄/无 `/dev/input`，2.4G 实际映射与背键未验证。见 [手柄核对](docs/deployment/xbox_controller_demo_2026-10-23.md) | 在小电脑上逐键核对（尤其背键/C/Z、LT/RT、十字键），把 code 固化成表；随后接入 FSM（按键→状态、摇杆→速度），未接线前 `rl_real_imgo2` 仍是键盘 |
| DEPLOY-CTRL-02 | P2 | **手柄控制已接入，离线/PTY 通过；实机待确认** | 11/13本机真实手柄已枚举，缺默认别名的问题已通过参数覆盖验证；无参数启动的固定别名及逐键/视觉/实机仍待，见设备路径记录（本机 `gamepad_device_path_2026-11-13.md`）。 `input_source` 1/2 选键盘/手柄，`ControllerConfig` 配设备名/死区；左摇杆→vx/vy（上/左为正）、右摇杆水平→wz（右为正），整段行程映射 `velocity_limits`；A/B/Y→站起/趴下/PPO（按下沿，走 `RobotFSM::request`）。FSM 单测、`fsm_check` FIFO 端到端（`cmd: 0.80 0.40 1.00`）、`rl_real_imgo2_pty` 冒烟通过；全量 ctest 28/29（仅既有 `rl_relocation`）。见 [手柄控制](docs/deployment/gamepad_control_2026-10-23.md) | 实机逐键核对标准 Xbox 码值（尤其 A/B/Y 与背键）、摇杆手感/死区/最大速度、PPO 下稳定性、断线重连；按键码目前未做成配置项。11/13 补充：焦点独立已静态核对，比赛电脑视觉/真实手柄并行联测与映射/独占冲突、负载仍待，见记录（本机 `gamepad_gui_focus_2026-11-13.md`） |
| DEPLOY-POLICY-01 | P2 | **最新 1199 策略 Gazebo 站立/行走通过；实机待验证** | 当前 `policy.pt`（md5 `1199a85e…`，`rl_kp=28`）在无头 Gazebo：`vx=0` 站立 z=0.2952 m、roll/pitch≤0.016°；`vx=0.3` 实测 0.2941 m/s（误差 2.0%）、周期 0.597 s、FL-FR −177°、无摔倒；但 shank 抖动 73.23 rad/s²、hip 换向 21.5/s、roll 2.83°、yaw 漂移 +0.88°/s（比 b25b 的 45/14.1/1.01/−1.85 更激进）。见 [Gazebo 验证](docs/deployment/new_policy_gazebo_2026-11-06.md) | 实机吊住/托住，先 0 站立再 0.1–0.2 低速；重点记录 shank 抖动、7–9 Hz、yaw 漂移；确认训练配置与 `rl_kp=28`/`rl_kd=1.0`/default/action_scale 一致 |
| DEPLOY-FREQ-01 | P1 | 当前配置与338秒频率已离线核对；执行链路差异与长尾待处理 | 本次Gazebo外层配置200Hz、实际命令接收约176Hz；时序差异待修/确认 | 归档复算及LoopFunc整毫秒截断线索见 [10月22日复核](docs/maintenance/project_history_2026-09_to_2026-10.md#simulation-and-deployment)。已实现ROS2附加延迟配置并短测通过，见 [延迟接口](docs/maintenance/project_history_2026-09_to_2026-10.md#simulation-and-deployment)；长时单变量试验待做。需旧权重原训练配置、完整安全trace/实际推理输入和IMU时标，以及电机内部PD资料；先分解通信长尾，再评估200Hz执行，不能直接改7ms为5ms声称对齐；未修运动颤抖 |
| DEPLOY-TIME-01 | P1 | 新批量源码已上传并离线核验；常态改善，偶发长尾待定位 | d30af52默认5ms、gap150us，CRC/ID归集；PPO整轮中位约2.12ms/P99约5.1–5.3ms，末尾多路近同时20ms超时，见 [源码/日志复审](docs/maintenance/project_history_2026-09_to_2026-10.md#simulation-and-deployment) | FT4232H候选的有效timer与逐台/批量A/B延迟待实测；补运行二进制/配置指纹、原始帧及真实采样/交付时标；主机年龄和成功trace统计不能证明零丢帧或固定2ms延迟，不放宽故障锁定 |
| GIT-REAL-01 | P2 | 合并修复已提交且原real已跟进；本机与远端跟踪引用对齐 | 当前d30af52在f3c8757修复之上同步小电脑源码；无合并冲突、检查开始干净。历史依据见 [Git复核](docs/maintenance/project_history_2026-09_to_2026-10.md#model-and-workspace) | 本轮未执行Git提交/推送；后续按正常分支同步，新增诊断文档另行提交 |
| DEPLOY-ROPE-FAULT-01 | P2 | 新19:41日志趴下后左后髋反馈健康检查Fault，原因待确认 | 172.710673s、leg_bl motor0报mode/error/temperature健康失败；走动选段valid4095/err0，未与横滚混作同因，见独立事件（本机 `rope_comparison_2026-11-09.md`） | 核对原始反馈/trace中的模式、错误码及温度字段，再决定修复；本轮未运行硬件或改保护规则 |
| DEPLOY-DIAG-01 | P1 | **新版同行推理和诊断目标tick映射已验证；源码归档/源时标/trace仍待** | 11/13复审确认本地real仍为重建输入的旧采集实现，未改已测试逻辑，见当前基线（本机 `real_control_review_2026-11-13.md`）。 11/9已修诊断脚本CSV写入时刻asof导致目标晚一提交拍的问题，tick对应2201条目标最大差5.792e-6rad；真实同行obs/action不受影响，旧干预修正节拍待复测，见修复与验证（本机 `pd_torque_time_2026-11-09.md`）。 用户部署端已改采集；11/9 11:55新日志含精确obs/action和输入/结果时标，4934行重算最大差1.26e-5，无需移行。旧11/8日志的重建输入限制不再套用此组，见[验证与边界](docs/deployment/ppo_vibration_status_2026-11-22.md#replay-and-filtering)。当前工作区real源码仍为旧实现，部署端修订需同步；Gazebo RobotState无源戳及trace ring安全发布问题未在本轮修复，历史见[复审§4](docs/deployment/go_imu_timing_vibration_review_2026-10-20.md#4-诊断代码中发现的限制尚未修改) | 新版result_age_ms有−.83ms而显式时间差非负，取时口径待核对，不影响同行配对结论，见[时标限制](docs/deployment/ppo_vibration_status_2026-11-22.md#replay-and-filtering)。 归档部署端采集源码和运行权重hash，补各传感器源采样/电机发送接收与实际应用时标、安全trace导出及相应回归；input_captured/result_generated不是各物理源时间，不能据此认定通信或运动总延迟。并发风险未跑TSan，不声称它导致横滚 |
| DEPLOY-THREAD-01 | P1 | **已源码定位，待实测；本轮未改代码** | 总线5ms／控制5ms／推理20ms 三个时钟自由运行、互不同相：两个5ms周期的相位使「反馈→主环」与「主环→总线」两段反相关，之和二峰（≈2ms 或 ≈7ms，差5ms只由开机决定），异步推理再固定跨一拍交接≈4.8ms，故状态→力矩的代码侧开销≈9–14ms；迟到时 `next=now` 还会永久平移相位，见 [线程复审](docs/maintenance/project_history_2026-09_to_2026-10.md#simulation-and-deployment) §2/§3.1 | 用 `robot.csv` 的 `age_ms[12]`/`imu_age_ms` 分布判相位是否漂移；再做「推理改由控制拍触发」A/B，比7–8Hz峰值与 `tau` RMS。未跑日志分析/硬件；5ms 不是抖动唯一原因，不能宣称已修 |
| DEPLOY-THREAD-02 | P1 | **已源码定位，待修复** | `GoMotorArray::state()` 算出的 `cycleCount/targetSeq/targetAgeMs/stale/clampedCount` 在 `RealHardware::snapshotLocked()` 被丢弃、全仓无消费者；`motor_stale_after_ms=100`／`imu.stale_after_ms=500` 过宽 ⇒ 15–25ms 停顿对FSM不可见（策略拿冻结观测继续出动作、`last_action` 自行前进），见 [线程复审](docs/maintenance/project_history_2026-09_to_2026-10.md#simulation-and-deployment) §3.2/§3.3 | 透出新鲜度字段并让控制拍按「新轮」或超时判据处理；验证停顿期间不再用旧观测重复推理，且不破坏现有 Fault/看门狗语义 |
| DEPLOY-12 | P2 | 已源码定位，待修复 | 推理持续向output_dof_tau_queue入队，全仓未找到消费；当前Gazebo RLControl发送目标角/PD且tau_ff=0，这份队列未用于驱动，见 [代码指南](docs/deployment/sim2sim_gazebo_code_guide_2026-10-12.md#7-已修复与待确认) | 明确或移除无用队列，验证控制输出保持一致与长时未消费数据不再累积；本轮未修改/运行测量 |
| DEPLOY-11 | P1 | 启动姿态已源码归因，初始化改进待验证 | 世界先无主动姿态控制落地；rl_sim启动发送reset_world再次回到z=.5；初始Passive为kp=0/kd=8，无固定趴姿。已补说明，未修改初始化，见 [排查](docs/maintenance/project_history_2026-09_to_2026-10.md#simulation-and-deployment) | 核对启动/重置/按0起身的位姿、关节与FSM时序；验证受控初始化后不会翻倒，再按3评估策略 |
| AMP-10 | P1 | 24000 轮运行已离线复核，确定性步态与微调收益待确认 | `2026-09-24_00-57-38_xiaoji_real1` 总奖励与滞空接近平台，末窗口跟速仍改善约 1.8%；最终 checkpoint 可恢复，旧 run 无滞空奖励。见 [复核及命令](docs/maintenance/project_history_2026-09_to_2026-10.md#amp) | 固定速度回放测周期／误差，验证新增滞空项微调能否达到 0.5–0.6 s 周期；确认原训练模型与本机模型一致 |
| AMP-DEFICIT-01 | P1 | 缺口接线与离线回归通过，待运行 | 当前Height保留feet_air_time_deficit，阈值0.20s、每步系数−3，旧feet_air_time关闭；19项离线测试通过，见 [复审](docs/maintenance/rebase_amp_air_time_resolution_2026-10-14.md#复审当前文件)。旧事件版见[历史](docs/maintenance/project_history_2026-09_to_2026-10.md#amp) | 新数据阈值、足端接触与短训/步态验证；不能将滞空阈值当完整周期 |
| REBASE-01 | P1 | rebase 已完成，训练/部署验证待完成 | 当前提交 a4b2103，Git 无冲突索引和 rebase 状态目录；最新策略/模型/数据身份已离线核对，见 [复核](docs/maintenance/rebase_amp_air_time_resolution_2026-10-14.md#最后9f3228e及最新策略算法身份) | 当前七项奖励短训练和动态回放仍待验证；缺失研究文档及重复问题编号保留待整理 |
| DEPLOY-10 | P2 | 已修并离线验证 | monitor帮助按非零start_pose解释qL；公共CLI严格数字/范围解析，拒绝非法/未知参数；相关测试通过，见 [整理记录](docs/maintenance/project_history_2026-09_to_2026-10.md#simulation-and-deployment) | 重构后的实际终端行为继续随DEPLOY-09实机确认 |
| DEPLOY-09 | P1 | 重构及21项离线回归完成；控制实机待确认 | 数学工具、MotorConfig/ImuConfig、真实main、Motor安全预检、无日志控制；21/21通过，参考FSM/真机入口/策略字节不变。见 [整理记录](docs/maintenance/project_history_2026-09_to_2026-10.md#simulation-and-deployment) 和 [手册](docs/deployment/go_imu_usage_2026-10-11.md) | 用户已确认udev与原monitor实测；新限幅/FSM/本轮重构的保持/站起/退出、频率、安装坐标、映射、机械裕量仍待确认；未接策略 |
| IMU-01 | P1 | 显示改进已验证；实机待确认 | 固定刷新 RPY（度）、角速度、加速度；等待／超时可见。用户确认 IMU 串口为 ACM0 并提供有效数据，见 [记录](docs/deployment/imu_terminal_2026-10-10.md) | 在 IMU 所在电脑重新编译确认面板；安装坐标、静止零偏、更新频率仍待实测 |
| AMP-11 | P1 | 29000 轮 real Gazebo 行走已验证；高步频／偏航待优化 | 正式导出及两次无界面步频测量通过，周期 0.2875–0.29 s、步频 3.45–3.48 Hz，身体速度误差 1.79%，偏航 −5.78°/s。见 [实测与奖励建议](docs/maintenance/project_history_2026-09_to_2026-10.md#amp) | 提供新训练 env.yaml/agent.yaml 和日志确认奖励版本；滞空增权／偏航核宽度单变量微调，复测 0.3/0.5/0.6 m/s 与接触周期 |
| AMP-18 | P1 | 新数据到24999曲线已核对，收益递减；目标步态待改 | 末等长窗口reward+1.06%、xy误差下降1.82%、value loss下降2.31%，无NaN/Inf；air快照.057→.059 s，不能由loss收敛推断目标周期达成。见 [复核](docs/maintenance/project_history_2026-09_to_2026-10.md#amp) | 有目标约束的新微调与固定命令/接触指标；原配置继续收益需新窗口证明，不保证慢步态 |
| AMP-17 | P1 | 最新model_new 25000身份、CPU契约与双引擎短测已验证，品质待改进 | 当前9c6bf22b...与最新run导出相同，checkpoint actor128输入差0；10-14三场景Gazebo/Isaac实测与转换检查通过、PD实录25/.5，见 [对照](docs/deployment/amp_engine_comparison_2026-10-14.md) | 静止、后退抬脚与慢周期、多种子/GUI长回放、新奖励训练效果；不把旧策略历史结果或局部契约通过当作当前品质达标 |
| AMP-16 | P1 | 当前新模型/21段数据身份已核对，七项微调待运行 | model/data与原9f3228e及离线清单一致，范围vx±1.2、vy±1、yaw±1.2；旧feet_air_time关闭、deficit保留。最新25000保存配置原为六项，见 [核对](docs/maintenance/rebase_amp_air_time_resolution_2026-10-14.md#最后9f3228e及最新策略算法身份) | 新七项配方短训与实际步态；侧向专家仅±.4，低速/平滑方案尚未实施 |
| AMP-15 | P1 | 原始24000基准补测与MPC采集复核完成；同步/元数据待修 | 原始24000同条件vx=.5：周期.2825 s、跟速误差1.11%、yaw−.15°/s、z .291 m，整体优于25000；录制器时间同步风险待修。见 [采集与基准复核](docs/maintenance/project_history_2026-09_to_2026-10.md#amp) | 确认实际MPC工作区/模型与采集参数；修同步、接触元数据/退出保存，小批重采对照 |
| AMP-14 | P1 | 25000微调sim2sim两次对比通过；周期目标、低站高与超调待优化 | 0.5 m/s命令下周期0.27 s、body vx约0.586 m/s、z约0.263 m、yaw约−2.2°/s；周期测量边界误判已修复并CPU/复测验证。见 [记录](docs/deployment/gazebo_amp_25000_2026-10-11.md) | 真实接触计步与0.3/0.6速度复测；单变量处理高度/跟速/节律，未永久切换默认策略 |
| AMP-13 | P1 | 服务器配置与sim2sim收益已确认；接触对齐待确认 | 实际任务比例0.7、滞空weight100/threshold0.3；25000相对本次24000周期/偏航改善，负日志不能直接判无效。见 [实测](docs/deployment/gazebo_amp_25000_2026-10-11.md) | 补参考与策略真实接触时间、其他速度回放及奖励目标验证 |
| AMP-12 | P1 | 专家帧时间索引与采样分布已修，待短训练验证 | 物理／策略／数据时间链一致；旧 N/(N−1) 插值使动作微加速且末帧越界，已统一为 time/FrameDuration；4 项 CPU 回归与旧版负向检查通过，见 [采样检查](docs/maintenance/project_history_2026-09_to_2026-10.md#amp) | 服务器同步 loader 并重启，确认 expert 预加载、4 环境×10轮训练和后续周期；不把约0.4%插值误差认定为高步频主因 |
| AMP-09 | P1 | 旧事件版已被deficit方案取代；当前方案待运行 | 历史feet_air_time步长尺度修正记录保留，当前旧项关闭，启用状态以AMP-DEFICIT-01为准；最新10-14 model_new保存配置没有滞空项，见 [当前澄清](docs/maintenance/project_history_2026-09_to_2026-10.md#amp) | 当前缺口项与低速门控需短训/步态验证，不能沿用旧事件版配置或把源码修改当作策略已训练 |
| DOC-02 | P1 | 待整理 | 原 real 提交包含 imgo2_description/README.md，本轮为保留改动未删除；见 [记录](docs/maintenance/rebase_main_2026-10-07.md) | 核对引用，将有效说明合入根 README 或 docs 后移除子目录 README |
| CHECK-01 | P1 | **主体离线检查通过；依赖项待补跑** | 2026-09-22 使用 Python 3.14.6：资源路径、模型同步、AMP 数据／关节顺序、`compileall`、tracked-ignore 检查均通过；拖曳测试 194 项通过、12 项按可选环境跳过。全量测试收集 281 项，得到 256 通过、24 跳过、1 个导入错误；错误仅为 `test_gait_metrics.py` 找不到 `numpy`，尚无代码断言失败证据。详见 [记录](docs/maintenance/project_history_2026-09_to_2026-10.md#model-and-workspace) | 在带 NumPy 的解释器重跑 `test_gait_metrics.py`；在带 PyTorch／Isaac Lab 的训练环境补跑当前跳过项。未执行前不得把这些项记为通过 |
| TOW-01 | P1 | **仿真与记录链路已完成** | 小车、弹性绳和不可伸长绳已接入同一拖曳场景；支持单／多环境、质量与阻力等参数扫描。记录包含两刚体状态、绳状态、张力／冲量、轮速、接触、阶段、真实间隙与追尾事件。可视化已确认机器人拖车和停车后小车前滑。详见 [拖曳仿真验证](docs/towing/towing_simulation_validation.md) | 保持入口、记录格式和离线汇总工具可复现；本项不再以“继续看画面”作为验收方式 |
| TOW-02 | P1 | **边界扫描脚本已实现，待实跑** | `scan_towing_boundary.py` 默认扫描速度 0.2–1.0、质量 5–25 kg 和两档轮阻，并区分“稳态拉不动”与“可拖但 Direct Stop 追尾”。地面摩擦默认固定 0.8，可在边界附近追加 0.4/0.8/1.2 复核 | 在训练机先跑 compliant 主网格，根据 `boundary.json` 缩小摩擦复核范围；之后再确定 v0.1 训练域。当前未实现 breakaway/Coulomb 阻力，不能把地面摩擦当成它的替代 |
| TOW-03 | P0 | **2026-10-08 起动作/观测契约已变（见 TOW-06），下列历史运行证据的维数与当前代码不再对应，需按 TOW-06 重跑**；自有 recurrent PPO＋decoder runner 已接线；任务已注册；4 环境冒烟（2026-09-22）、4096 环境 × 2000 轮完整 run（2026-09-23/24）与确定性回放（2026-09-25）均跑通；跨环境隔离、STOP 排序与 estimator 精度未完成；**2026-10-08 的刚体连杆与 400 环境网格改动早于上述 run，需重跑** | decoder 使用 `51→128→GRU(128)` 预测机器人机体系 `vx/vy`、负载质量和牵引力 `Fx/Fy`，5 维 estimate detach 后组成 56 维 actor 输入。仓库 `rl_lab` runner 已管理 decoder／actor／critic 三套 GRU、critic-only normalizer、rollout estimate 固化、PPO 后 decoder 更新及联合 checkpoint；不依赖外部 RSL-RL runner/config API。STOP 后有界 GT 拉力惩罚已加入。**2026-09-22 训练机前置核查**：拖曳离线测试 210 项全通过、51 维帧与 56 维 actor 契约一致、冻结 AMP 策略与 `torch.jit.load` 路径齐备；修掉 `train.py` 的 `--agent` 默认值（原 `towing_rl_lab_cfg` 全仓无对应注册项，必抛 `ValueError`）；加入注册 `Imgo2-towing-upper-rl-lab`（`--agent=rl_lab_cfg_entry_point`），并把原守门断言换成锁定注册接线的契约测试（已用两组负向测试确认有效）。详见 [GRU 记录](docs/maintenance/project_history_2026-09_to_2026-10.md#towing) 与 [训练前置记录](docs/towing/towing_training_prep_2026-09-22.md) | 在 Isaac Lab 2.2.1 训练机完成 wrapper 构造、4／256 环境 recurrent rollout、checkpoint 恢复、estimator 精度、STOP reward 排序和无小车隔离验证，重点排除策略为卸载拉力而让小车逼近的捷径。**注意**：注册是用户为执行训练而显式翻过 `towing_upper_rl_review` 第 98 行「未完成运行验收前不注册」这道保护，验收通过前该项不得记为完成。**2026-09-22 冒烟实跑（4 环境 × 10 轮）通过**：环境构造成功、51 维帧／56 维 actor 契约在运行时成立（`memory_a` 输入 56 维、`memory_c` 65 维）、checkpoint 含三套 GRU＋decoder＋两优化器＋critic-only normalizer、`iter` 正确写入 0→10（AMP-08 修好后首次在真实 checkpoint 上确认）、decoder 可从 checkpoint strict 加载并推理。详见 [冒烟记录](docs/maintenance/project_history_2026-09_to_2026-10.md#towing)。**2026-09-24 完整 run（用户执行）**：`logs/towing_rl_lab/towing_upper/2026-09-23_22-19-00/`，4096 环境 × 2000 轮跑满（`seed=1`、`save_interval=100`），`model_2000.pt` 的 56 维 actor／65 维 critic／3 维动作与运行时断言一致。TensorBoard 已记录全部分项与诊断：`obs_cart_present=0.8745`≈7/8（12.5% 无小车设计成立）、终止占比 `time_out=0.9882`／`cart_collision=0.0118`／`robot_fall=0`、`tracking_velocity=0.9211`、`yaw_heading` 折合 RMS 1.4°（不再自转）、`min_clearance` 几乎不触发；`obs_towing_force≈obs_towing_force_active`（Σ‖F‖≈328 N·步/回合）说明有拉力时都 >1 N。**2026-09-25 确定性回放（用户执行，`--num_envs=1`、2000 步 = 10 回合跑满）**：`err_cmd=0.339`、`err_track=0.387`、`err_low=0.065` m/s ⇒ `err_track/err_cmd≈1.14`、`err_track/err_low≈5.9`，即误差**仍主要来自上层积分指令没到位**（`ref` 平均比 `user` 少 0.34 m/s），而下层把上层的 `ref` 执行到 0.065 m/s；相比旧 run（不同 checkpoint／旧 `play.py`）的 `err_cmd=0.498`／`err_low=0.097` 有改善但未消除。**该缺口不是"抖"**：`ref` 是 20 Hz 限速率积分器（每步 `x∈[-0.05,+0.025]`、`yaw∈±0.05`），高频抖动上限仅 0.025 m/s；训练日志里 `action_rate`／`action_magnitude` 的数字几乎全部由探索噪声解释（`2Σσ²=0.477` vs 实测 `E‖Δa‖²=0.429`，`std` 取自 checkpoint），因此"抖动仍明显、继续提权这两项"的判断已作废（详见 [记录](docs/towing/towing_play_2026-09-25.md) §7.2）。缺一次 `--cmd-interval=1` 的逐帧 trace 才能区分"起步滞后"与"稳态缺口"。decoder：速度 MAE **0.033 m/s**、力 MAE 0.356 N、质量 MAE 1.62 kg（仅 5.4% 的步受监督，质量项证据薄）。**仍未验证**：跨环境隔离、STOP reward 排序、单环境以外的统计（需 `--num_envs=32`）、checkpoint 恢复、scripted policy 复现；上层策略仍无 `exported/policy.pt`，且 `towing/play.py` 没有导出分支。详见 [回放记录](docs/towing/towing_play_2026-09-25.md) **2026-10-08 新增第三类连接与场景网格**：`MultiRopeModel` 按整数 id 混合三套模型（长度不同时连 `rope_extension` 一并混合）；训练场景改为 20 列 × 20 行 = 400 环境的确定性网格（列 = 弹性 8 / 刚体 8 / 普通绳 4，行 = 长度 0.4→0.8 m 均分 20 档；弹性绳 4 档 k/c × 2 列），类型与长度按 env index 一次映射、reset 不再随机，spawn 按**本 env 连接长度**摆放（绳 = 0.5·L0、刚体 = L）；`min_clearance` 阈值改为逐 env 用连接长度、ratio 0.40→0.25。已由离线/静态契约锁定（拖曳离线测试 266 项通过；本机全量 317 项通过；400 env 逐 env 张量复核 spawn 时最大张力 0）。**仍未验证**：网格与刚体 env 未在 Isaac Lab 实跑（默认环境数 256→400），列映射是否在 400 环境下按预期生效、机器人出生下落对刚性杆的初始推挤、停车反驱、以及 12.5% 无小车掩码在网格下的隔离都无运行证据。 |
| TOW-03 | P0 | **自有 recurrent PPO＋decoder runner 已接线；任务已注册；4 环境冒烟（2026-09-22）、4096 环境 × 2000 轮完整 run（2026-09-23/24）与确定性回放（2026-09-25）均跑通；跨环境隔离、STOP 排序与 estimator 精度未完成** | decoder 使用 `51→128→GRU(128)` 预测机器人机体系 `vx/vy`、负载质量和牵引力 `Fx/Fy`，5 维 estimate detach 后组成 56 维 actor 输入。仓库 `rl_lab` runner 已管理 decoder／actor／critic 三套 GRU、critic-only normalizer、rollout estimate 固化、PPO 后 decoder 更新及联合 checkpoint；不依赖外部 RSL-RL runner/config API。STOP 后有界 GT 拉力惩罚已加入。**2026-09-22 训练机前置核查**：拖曳离线测试 210 项全通过、51 维帧与 56 维 actor 契约一致、冻结 AMP 策略与 `torch.jit.load` 路径齐备；修掉 `train.py` 的 `--agent` 默认值（原 `towing_rl_lab_cfg` 全仓无对应注册项，必抛 `ValueError`）；加入注册 `Imgo2-towing-upper-rl-lab`（`--agent=rl_lab_cfg_entry_point`），并把原守门断言换成锁定注册接线的契约测试（已用两组负向测试确认有效）。详见 [GRU 记录](docs/maintenance/project_history_2026-09_to_2026-10.md#towing) 与 [训练前置记录](docs/towing/towing_training_prep_2026-09-22.md) | 在 Isaac Lab 2.2.1 训练机完成 wrapper 构造、4／256 环境 recurrent rollout、checkpoint 恢复、estimator 精度、STOP reward 排序和无小车隔离验证，重点排除策略为卸载拉力而让小车逼近的捷径。**注意**：注册是用户为执行训练而显式翻过 `towing_upper_rl_review` 第 98 行「未完成运行验收前不注册」这道保护，验收通过前该项不得记为完成。**2026-09-22 冒烟实跑（4 环境 × 10 轮）通过**：环境构造成功、51 维帧／56 维 actor 契约在运行时成立（`memory_a` 输入 56 维、`memory_c` 65 维）、checkpoint 含三套 GRU＋decoder＋两优化器＋critic-only normalizer、`iter` 正确写入 0→10（AMP-08 修好后首次在真实 checkpoint 上确认）、decoder 可从 checkpoint strict 加载并推理。详见 [冒烟记录](docs/maintenance/project_history_2026-09_to_2026-10.md#towing)。**2026-09-24 完整 run（用户执行）**：`logs/towing_rl_lab/towing_upper/2026-09-23_22-19-00/`，4096 环境 × 2000 轮跑满（`seed=1`、`save_interval=100`），`model_2000.pt` 的 56 维 actor／65 维 critic／3 维动作与运行时断言一致。TensorBoard 已记录全部分项与诊断：`obs_cart_present=0.8745`≈7/8（12.5% 无小车设计成立）、终止占比 `time_out=0.9882`／`cart_collision=0.0118`／`robot_fall=0`、`tracking_velocity=0.9211`、`yaw_heading` 折合 RMS 1.4°（不再自转）、`min_clearance` 几乎不触发；`obs_towing_force≈obs_towing_force_active`（Σ‖F‖≈328 N·步/回合）说明有拉力时都 >1 N。**2026-09-25 确定性回放（用户执行，`--num_envs=1`、2000 步 = 10 回合跑满）**：`err_cmd=0.339`、`err_track=0.387`、`err_low=0.065` m/s ⇒ `err_track/err_cmd≈1.14`、`err_track/err_low≈5.9`，即误差**仍主要来自上层积分指令没到位**（`ref` 平均比 `user` 少 0.34 m/s），而下层把上层的 `ref` 执行到 0.065 m/s；相比旧 run（不同 checkpoint／旧 `play.py`）的 `err_cmd=0.498`／`err_low=0.097` 有改善但未消除。**该缺口不是"抖"**：`ref` 是 20 Hz 限速率积分器（每步 `x∈[-0.05,+0.025]`、`yaw∈±0.05`），高频抖动上限仅 0.025 m/s；训练日志里 `action_rate`／`action_magnitude` 的数字几乎全部由探索噪声解释（`2Σσ²=0.477` vs 实测 `E‖Δa‖²=0.429`，`std` 取自 checkpoint），因此"抖动仍明显、继续提权这两项"的判断已作废（详见 [记录](docs/towing/towing_play_2026-09-25.md) §7.2）。缺一次 `--cmd-interval=1` 的逐帧 trace 才能区分"起步滞后"与"稳态缺口"。decoder：速度 MAE **0.033 m/s**、力 MAE 0.356 N、质量 MAE 1.62 kg（仅 5.4% 的步受监督，质量项证据薄）。**仍未验证**：跨环境隔离、STOP reward 排序、单环境以外的统计（需 `--num_envs=32`）、checkpoint 恢复、scripted policy 复现；上层策略仍无 `exported/policy.pt`，且 `towing/play.py` 没有导出分支。详见 [回放记录](docs/towing/towing_play_2026-09-25.md) |
| TOW-04 | P0 | **已解除：GPU 在本机可用** | 2026-09-22 DSH 会话进程内曾观察到 GPU 不可见（`NVIDIA_VISIBLE_DEVICES=void`、无 `/dev/nvidia0`／`nvidiactl`／`nvidia-uvm`、`torch.cuda.is_available()==False`、`nvidia-smi` 报 `Failed to initialize NVML: Insufficient Permissions`）；**该观察不适用于执行训练的 shell**。用户同日实测 `bash imgo2_rl/scripts/run_isaaclab.sh --check` 输出 `cuda available: True` 与 `CUDA solve: OK`（torch 2.7.0+cu128，`dlopen(linalg)` OK），GPU 正常。教训：`torch.cuda` 可见性按进程求值，agent 会话进程的观测不能代表用户终端 | 无需再处理；后续以执行方的 `--check` 输出为准。拖曳训练的「已运行」判定改为取决于实际训练是否启动并产出日志 |
| TOW-05 | P0 | **MuJoCo 场景已修，待动态验证；完整 sim2sim 未闭环；2026-10-08 起入库的占位权重已与新契约不兼容** | 新增 5–15 kg 小车、四轮阻尼、弹性／不可伸长两种单边绳场景，静态契约通过；尚未编译或运行 MuJoCo，且上层策略尚无可部署导出件，Gazebo 小车场景未接入。详见 [拖曳 sim2sim 记录](docs/towing/towing_sim2sim_2026-09-24.md)。**2026-09-25 入库占位权重（用户决定）**：`imgo2_deploy/ros2_ws/policy/imgo2/towing/policy.pt` 是 `logs/towing_rl_lab/towing_upper/2026-09-23_22-19-00/model_2000.pt` 的原样拷贝（sha256 `a4d800a26c1d1a50…`，9.76 MB，含 optimizer／decoder／normalizer 的**训练态** checkpoint），**不是** `play.py` 导出的 TorchScript，libtorch 加载器读不了；该目录**没有** `config.yaml`，FSM 也**没有**对应按键／状态类。`towing/play.py` 的导出分支目前为 0 处（ppo／amp／himloco 才有）。详见 [回放记录](docs/towing/towing_play_2026-09-25.md) §9 | MuJoCo 编译／加载与 5、10、15 kg 拖曳停车回放通过；训练侧 checkpoint 经 `play.py` 导出并接入 51 维双 GRU 观测／控制链；Gazebo 场景实现并用相同案例对照速度、停车间隙和碰撞**2026-10-08 范围说明**：训练侧新增的刚体球铰连杆（双边、可反驱）与 20×20 场景网格按用户要求**只在训练侧实现**，MuJoCo 仍是两个单边绳场景；sim2sim 的第三类需要另加 `scene_tow_rigid.xml` 与 `rl_sim_mujoco.cpp` 的场景名判断，本轮未做 |
| TOW-06 | P0 | **代码已改，离线契约通过；Isaac Lab 运行验证未做** | 2026-10-08 用户决定并已落地两项改动。**① decoder 牵引力 2 维 → 3 维**（机体系 x/y/z），机器人速度保持 2 维 ⇒ decoder 输出 5→**6** 维。依据：两挂点高差 Δz=0.17 m，绷紧时方向向量 z 分量 = 0.17/L0，在 L0 = 0.4–0.8 m 上就是张力的 **21–43%**，且 Fz 在 −0.16 m 挂点上产生俯仰力矩；`force_on_robot` 本来就是 `tension × direction` 的三维世界系力，`towing_force_b` 此前在 `[:, 0, :2]` 处被截断。副作用：所有 `vector_norm(towing_force_b)` 的消费点（`stop_towing_force`、`towing_force_norm`、质量监督权重）从此算的是**真实张力**（比原 2 维范数大 2–10%，含竖直分力）。**② 上层动作 3 维加速度 → 12 维关节位置残差**：送冻结 AMP 策略的速度指令改由脚本调度直接给出（settle 0 → tow 速度 → STOP 后 0，与测量台 `tow_drag.py` 的 `user_cmd` 同相位、其记录里 `user_cmd == ref_cmd`），残差按 `deltapos = action_scale ⊙ clip(u, ±1)` 叠加在冻结策略的关节位置目标上，尺度复用冻结策略契约的 `action_scale`（髋 0.125 / 大腿·小腿 0.25 rad；上层 `clip_actions=1.0`，故残差只用到底层 ±3.0 权限的 1/3）。观测随之变为 **57 维单帧**（命令项只留 `loco_command`，`last_action` 3→12）、actor 输入 **63 维**、critic **65→72 维**、decoder 组 6→7。**顺带修一处奖励错误**：`velocity_tracking_exp` 的线性项原比 `reference_command − user_command`（与函数名和配置里的奖励文档都不符，且与 `reference_tracking_l2` 重复），现比**实测**机体系线速度；`reference_tracking`（−5.0）随 `reference_command` 删除。Towing runner 对 actor 末层**零初始化**（残差必须从 0 起步：实测未初始化时首拍 |a|max ≈ 0.149 归一化，约 ±0.04 rad 的逐关节系统性偏置；只在 towing 生效，ppo/amp/himloco 共用模块未动）。详见 [记录](docs/towing/towing_deltapos_residual_2026-10-08.md) | 在训练机跑 4 环境冒烟 + ≥400 环境 rollout 与一段完整 run，逐条确认：wrapper 的 `num_obs=57`／`num_decoder_obs=7` 断言成立、`memory_a` 输入 63、`memory_c` 输入 72、decoder 可从 checkpoint strict 加载（6 维输出）、**同一速度指令下残差=0 时关节指令与改造前逐位一致**（改造后的指令是脚本阶跃、改造前是积分斜坡，故只对同一指令序列成立，不是整回合等效）、残差不出现在冻结策略的 45 维观测里、残差从 0 起步（首轮动作≈噪声而非偏置）。**旧证据全部作废**：`model_2000.pt`（56 维 actor／5 维 decoder）与旧 run 的 TensorBoard 读数不能与当前契约对照。**待实跑复核的权重**：`action_rate`（−0.1）与 `action_magnitude`（−0.05）原先按 3 维动作标定，12 维下 `Σu²` 上限由 3 变 12、同等逐维幅值量级约 ×4；`low_level_pos_error`（初值 −5.0，2026-10-09 冒烟后改为 −0.1）为 2026-10-09 新增项（见 TOW-12） |
| TOW-10 | P1 | **训练机已实跑（2026-10-10/11，100 环境 × 3 速度档 × 4 质量档；800 环境完整网格与同版本基线仍未跑完）**：`v3_400/600/2700_policy` 三轮同网格对照 = 通过 52/50/**58**（详见 [2700 轮复核](docs/towing/towing_play_test_2700_2026-10-11.md)）；`v3_250_base` 那轮**没跑完**（只有 `experiment.json` + `summaries/`，无 `report.json`）⇒ 「策略 vs 基线」尚未对照 | 2026-10-09 新增「上层任务必要性」基线测试台 [play_towing_test.py](imgo2_rl/scripts/towing/play_towing_test.py)：冻结 AMP 策略 + 脚本速度指令（station 0 → tow v → STOP 0）在速度 `0.5/1.0/1.5 m/s` × 连接三类 × 质量 `5–25 kg` × 坡度 `0/±5/±10°` 的 **225 case** 网格上测五项指标（起步关节响应误差、全程跟速误差、停车小车滑移距离、停车间距维持、停车关节响应），地面摩擦固定 `0.8`，逐坡度 45 环境并行、共 5 次仿真过程；产物 `report.md`（逐坡度判定矩阵 + 必要性结论）/`report.csv`/`report.json` + 逐 case 摘要。坡度两种后端：默认 `gravity`（现有平地 + 旋转重力，与真实坡面同解、度量坐标系最简单），`terrain` 接远端的真实坡面 mesh（tile 几何与坡面出生姿态与训练场景同源）；坡上站定段默认 `--slope-settle hold`（驻车制动仿真，起拖释放）。离线测试 `test_towing_play_test.py` **103 项通过**（`/usr/bin/python3` 无 torch 时 98 通过 + 5 跳过）、全量 **434 项通过**、`compileall`/`git diff --check`/tracked-ignore 通过；`--dry-run` 已在无 GPU 环境核对默认网格与命令行校验。详见 [记录](docs/towing/towing_necessity_test_2026-10-09.md) | 在训练机先跑 8 环境冒烟（`--slopes 0 --velocities 0.5 1.0 --connections compliant rigid --cart-masses 5 25 --write-csv none`）确认环境构造、逐 env 质量/连接/指令张量、`physics_sim_view.set_gravity()` 生效、记录列与 `TEST_FIELDS` 一致；再单独冒烟一次 `--slope-backend terrain --slopes 0 5`（验证自定义 `TerrainImporterCfg` + 生成器/importer、坡面出生首拍误差）；然后跑默认 225 case 完整网格，并跑一次 `--command-shaping ramp` 对照，最后按 `report.md` 的分组统计回答「上层任务是否必要」。**在此之前不得把本项记为完成**；关节响应阈值（RMS 0.10 rad / 单关节 0.30 rad）是未标定的工程占位，首轮数据出来前 `JNT` 不能当定论。**未覆盖**：弹性绳另外 3 档 k/c、轮阻档、breakaway/Coulomb 阻力、跨环境隔离、真机、`--slope-settle free` |
| TOW-11 | P0 | **已修并实跑验证（2026-10-09 19:19 run）：`fell` 全 False、判定矩阵从 `FALL×9` 变 `JNT/SPD`**。根因是测试台物理步进写的是无参 `sim.step()`，走的是 `SimulationContext.step(render=True)` 的 `self._app.update()` 分支 —— 物理由 app/帧时序驱动，**一帧推进约 10 个物理 tick**（实测首拍 base 掉 7 mm、小腿转 0.1 rad ⇒ 等效 dt ≈ 50 ms 而非 5 ms），「每 4 次调用 = 20 ms 控制周期」的前提失效、按 200 Hz 标定的 PD 发散 | 现象：必要性测试台实跑（`--num-envs 9 --headless --write-csv all`）主循环跑完，逐 case 判读处抛 `TypeError: '<' not supported between instances of 'float' and 'NoneType'`，`state=failed`、`summaries/` 与逐 case `tow.csv` 全丢；同配置 GUI 下「机器人胡乱挣扎」。**① 崩溃**：`tow_clearance.clearance()` 的「面对面」判据（横向/竖向都要在 `FACING_TOL_M = 0.05 m` 内）在机器人被拽翻/腾空时一对点都对不上，**合法返回 `(None, None)`**，而 `summarize_tow` 直接 `min(...)`。已修：逐相位计数不可用样本（`clearance_unavailable_samples`/`clearance_unavailable_phases`）、只对可用样本取最小值、判接触与 `final_clearance_m` 守 None（后者只退回 coast 段，不让 station 段冒充「末端间隙」），并把 `--write-csv all` 的原始轨迹**提前到指标计算之前**落盘。**② 复跑（`20261009T100609Z_3b5735d0`，`state=completed`）**：9/9 判 `FALL`，**8/9 倒在 station 段**（起拖之前），最低 base 高 0.074–0.079 m（正常站高 0.30），起步关节 RMS **0.59–0.64 rad**（阈值 0.10 的 6 倍），`settle_robot_travel_m = −1.099 m`、`settle_load_drift_m = −0.649 m`（站定段整对儿倒退 1 m）。**三条互斥情形里两条被实测排除**：绳/杆不是起因（前 6 个记录点 `rope_tension_n ≡ 0`，张力 t≈0.33 s 才起跳，而 base 25 ms 内就从 0.3425 掉到 0.1653）、出生间隙 8.8 mm 也不是起因（9 case 只有 2 个有车斗接触，7 个 `cart_deck_fx_n ≡ 0`，9 个全倒）、也不是掉进地里（`robot_z_m` 弹跳而非单调下掉，`invalid_samples=0`）。剩下的一档：`robot_tau_*` 从第二拍起**正负交替打到 ±23.7 N·m（= effort_limit）**、关节被压到极限（`robot_jp_02` 一度 = −3.0000）；而**首拍是正常的**（目标 `jt=(−0.010, 0.912, −1.803)` = 默认站姿 + 小量，与 `default + action_scale×action` 逐项吻合）⇒ 观测/置换在首拍对，第二拍起策略输出饱和。**③ 出生间隙仍是真实的设计余量问题**（不是本次起因）：compliant 档 row0 的「机器人后表面 ↔ 车头」几何间隙只有 **0.0088 m**（row1–3：0.0277/0.0463/0.0645；rigid 档 ≥0.3370），而 `upper_env_cfg.py` 写的「spawn 间隙 ≈ 0.250 m」是**挂点距**（机器人后腿伸到 base 后方 0.398 m，挂点仅 −0.16 m）。详见 [必要性测试台记录](docs/towing/towing_necessity_test_2026-10-09.md) 的「二跑」一节 | **已修（离线验证，待实跑复验）**：`sim.step(render=False)` + `if not args.headless and (step + 1) % render_interval == 0: sim.render()` （`render_interval` 取 `training_cfg.sim.render_interval` = decimation = 10），与 `ManagerBasedRLEnv:190` 的物理步进口径一致；新增静态守卫 `test_physics_step_must_not_render`（禁止无参 `sim.step()`）。验证：离线 `test_towing_play_test.py` 88 项 OK、全量 423 项通过 0 失败、`py_compile`/`git diff --check` 通过；**实跑验证（`20261009T111933Z_50b72de5`，3 环境 × `--record-every 1`）**：首拍 base 掉 7 mm → **0.2 mm**、首拍小腿 0.10 rad → **0.002 rad**（与预测的 0.1 mm / 0.01 rad 同量级）、`fell` 9/9 True → **3/3 False**、最低 base 高 0.074–0.079 → **0.256–0.259 m**、站定段位移 −1.099 → **+0.027 m**、站定段最大张力 130–1543 N → **0 N**、起步关节 RMS 0.59–0.64 → **0.164–0.173 rad**（仍高于 0.10 阈值）。**剩余**：这 3 个 case 全是出生间隙只有 8.8 mm 的 row0，JNT（关节 RMS）/SPD（跟速 MAE 18–26% > 20%）缺口与「起步就被小车约束」尚未分开 ⇒ 下一步跑 `--num-envs 20`（第 0 列 row 0–19，间隙 8.8 mm → 0.337 m）看是否单调改善；阈值本身仍是未标定的工程占位。已完成的离线核对：② 把名义站姿喂给把名义站姿喂给 `imgo2_deploy/ros2_ws/policy/imgo2/amp/policy.pt`（sha256 `cba59d44…`）得到的 12 维动作换算出的关节目标与 `tow.csv` **首拍** `robot_jt_*`（−0.0100/0.9115/−1.8027…）**逐位吻合** ⇒ 观测组装、缩放、关节置换、`action_scale→目标`、TorchScript 加载这条链路正确，「底层 loco 没接上」**排除**；同一套 URDF FK 还算出默认站姿足端最低点只比 base 低 **0.3042 m**（0.35 m 出生 ⇒ 离地 **4.6 cm**，`towing_env_cfg.py` 注释里的 8.2 cm 偏大），且记录 t=0.005 首拍足端已触地 ⇒ 倒地发生在**落地后的最初几十毫秒**；③ 与训练任务（同一 policy、同一 action term，15:30 那轮 `goal_reached = 0.52`）做同条件对照，找出测试台与 `ManagerBasedRLEnv` 之间剩下的差异（当前只剩执行器/物理闭环这一档）。①③ 需要 GPU（本会话无 `/dev/nvidia*`）。出生几何是否收紧（`SLACK_RATIO`／`LENGTH_MIN_M`／compliant 的「最小几何间隙」下限）另需用户拍板，且会改动训练网格与 `min_clearance` 的 ratio |
| TOW-12 | P1 | **已修（离线契约通过）；权重待实跑标定、Isaac Lab 未跑** | 2026-10-09 用户要求新增「底层输出的 pos 与真实 pos 的差距」惩罚，并明确**用底层输出、不用经过残差的 pos**。新增 `mdp.low_level_position_error_l2`：`Σ(q_actual − loco_joint_targets)²`（12 关节求和，rad²），其中 `loco_joint_targets` 是冻结策略自己的 `output.joint_targets`（= `default_dof_pos + action_scale·clip(action)`，绝对位置、策略关节顺序），实际下发的 `output.joint_targets + delta_joint_pos` **不参与**；action term 里新增的缓存写在叠加残差**之前**。与 `action_magnitude`／`action_rate` 的分工：那两项罚残差本身（幅值/变化率），本项罚残差的**后果**——位置误差大通常意味着 PD 饱和、接触顶住或腿被压住，是真实的可控性信号。注册 `low_level_pos_error = RewTerm(..., weight=-0.1)`（初值 −5.0 已按同日冒烟实测下调为 −0.1；**参考量定为 `reference="commanded"`**，即期望 = 底层输出 + 残差 = 下发给 PD 的目标 ⇒ 误差是纯 PD 跟踪误差；另一分支 `"frozen"`（底层自己输出、误差含 δ）保留供对照）。**验证**：全量 `imgo2_rl/tests` **433 项通过 0 失败**；新增契约测试断言函数体不含 `delta_joint_pos`／`_held_joint_targets`、缓存来自 `output.joint_targets` 且先于叠加残差。**未验证**：权重 −5.0 按「逐关节 RMS 0.05 rad ⇒ Σ≈0.03 ⇒ 每步约 −0.0075，约为跟踪项满额的 15%」估的初值，未标定；本机无 Isaac Lab，未构造环境、未实跑 | 训练机跑一段 run，看 `Episode_Reward/low_level_pos_error` 与 `tracking_velocity` 的量级比：**2026-10-09 冒烟已回答该项**：w=−5 时 −0.086/步 ≈ `tracking_velocity` 的 2 倍、主导回报 ⇒ 已收到 −0.1（见维护记录同日「冒烟评估后」一条）。**用户 2026-10-09 决定先保留 −0.1**（承认当前作用有限：每步 ≈ −0.0017 ≈ 跟踪项的 4%，且被噪声驱动的 δ 与瞬态主导），待正式 run 里 `Episode_Reward/low_level_pos_error` 与 `action_magnitude`/`action_rate` 的对比再定是否上调到 −0.5~−1.0、以及是否调小那两项以让 δ 真正承担补偿；同时确认它在 STOP/超时相位下不会主导回合和 |
| TOW-13 | P0 | **已修（离线契约通过）；Isaac Lab 未跑** | 2026-10-09 用户判定「改成目标制不对」，要求改回并**保留**两条停车段奖励：`post_stop_towing_force`（−1.0，停车后绳子还绷着就扣分＝到点卸力）与 `post_stop_distance`（−0.1，停车后还往前多走的距离＝说停就停），并要求回合仍是「初始静止稳定 → 走一段 → cmd vel=0」，且**置零指令必须发生在越过坡之后**。**改**：`episode_geometry` 的 `GOAL_DISTANCE_M`→`STOP_DISTANCE_M`（10 m）并新增 `POST_STOP_WINDOW_S=3.0`；`upper_mdp` 恢复 `stop_time_s`（初值 `+inf`）／`stop_origin_x`／`_was_stopped`，`process_actions` 改由**进度**触发（`progress >= stop_distance_m`）并在那一拍记录停车时刻与 x，`goal_reached`→`stop_reached`（不再是终止条件）；两条 post_stop 奖励按 `elapsed_s >= stop_time_s` 门控（未停车前精确为 0）；`upper_env_cfg` 去掉 `goal_reached` 终止项、恢复两条奖励、新增诊断量 `obs_stop_reached`（0/1 粘性标志，回答「走没走到 STOP 点」——回合一律超时退出，不能靠终止原因区分）、timeout 的 margin = 停车窗口（正常回合全部由 timeout 收尾，用户 2026-10-09 确认）、保留 `robot_fall` 倒地退出，并加断言 `stop_distance_m > FLAT_OUT_START_M`（坡面出口 9.0 m）保证「越过坡之后」。**用进度而非时间触发的理由**：最慢速度（0.4 m/s）走到坡出口就要约 23 s，固定时间阈值无法保证越过坡。**2026-10-10 工作域收紧为 0.5–1.5 m/s 后**：最慢速度到坡出口约 18 s、timeout 24.185 s（484 步），进度触发仍保证「越过坡才置零指令」（见 [工作域收紧记录](docs/maintenance/project_history_2026-09_to_2026-10.md#towing)）。**验证**：全量 `imgo2_rl/tests` **424 项通过 0 失败**（契约测试改成 STOP 制并新增相位/奖励/顺序守卫；几何测试把 `GOAL_DISTANCE_M` 改名并加「STOP 点在坡后」断言）。**未验证**：本机无 Isaac Lab，未构造环境、未实跑；STOP 触发点的运行时行为、两条奖励的量级与 `POST_STOP_WINDOW_S=3 s` 是否够用都待训练机 | 在训练机跑冒烟 + 一段 run，确认：进度触发按预期置零指令、`stop_time_s`/`stop_origin_x` 记录正确、两条 post_stop 奖励只在停车后非零、`obs_stop_reached` 与 `Episode_Termination/time_out` 合起来能读出「走没走到 STOP 点」、倒地/碰撞确实按失败退出、以及 800 环境里平地/5°/10° 两类坡道的 STOP 都发生在出口平地 |
| TOW-14 | P1 | **已按 PD 方案落地（离线契约通过）；Isaac Lab 未跑** | 用户 2026-10-09 要求「运动过程中 3 维拉力在 y 维度保持为 0」⇒ 新增 `towing_force_y = RewTerm(func=mdp.towing_force_y_ratio_sq, weight=-10.0)`，取 **比值** `(F_y/‖F‖)² = sin²(偏离角)`（与张力无关、有界 [0,1]；原始 `F_y²` 会被起步绷直的百牛级峰值放大成变相惩罚大张力）。量级：偏 10° → −0.015/步、20° → −0.059/步。**已知限制（同一条也适用于「上坡保持中线 + 朝向稳定」）**：actor 的 57 维帧里**没有** F_y、横向偏置 y、朝向误差；critic 有 3 维 `towing_force`（⇒ 有 F_y）但**也没有** y 与朝向。所以策略只能靠 GRU 历史间接学，**无法闭环修正**；而 `F_y≈0` 只说明绳在机体矢状面内（小车正后方），**不等于沿车道中线/朝向对齐**——整体偏航但小车跟着偏到正后方时该值仍 ≈0。两条可选路线：**(a)** 把 `(y_offset, heading_err)`（可选再加 F_y 比值）加进 actor 帧 ⇒ 契约 57→59/60，且真机需要定位/路径信息；**(b)** 用 `loco_command[:,1:3]`（当前恒 0、冻结策略本来就当 vy/ω 指令接受）接 PD 外环 ⇒ **不改任何维数**，测量台 `play_towing_test.py` 已有同款实现。**验证**：全量 `imgo2_rl/tests` **430 项通过 0 失败**（新增守卫：必须用比值、不得退化成原始 `F_y²`）。**未验证**：本机无 Isaac Lab，未实跑；量级与 (a)/(b) 的取舍待定 | 在训练机跑一段 run：读 `Episode_Reward/towing_force_y`，确认 F_y 惩罚量级合适；**用户决定（2026-10-09）：走 (b)** —— 保持中线与 heading 用 PD，不让策略学。已落地：`upper_mdp.HierarchicalVelocityAction._lane_keeping_vy/_wz` 在 `process_actions` 里写 `loco_command[:,1:3]`（冻结策略的 (vy, ω) 指令槽，此前恒 0）；公式与增益**与测量台 `play_towing_test.lane_keeping_command` 逐项一致**（`kp_y=1.0 kd_y=0.3 kp_yaw=1.5 kd_yaw=0.3`、`lane_yaw_target=0`=车道轴向），输出限幅在冻结 AMP 训练分布内（`lin_vel_y ±1.0`、`ang_vel_z ±1.57`）。**不改任何观测/动作维数、不新增奖励**，且 `loco_command` 本就是 actor 观测项 ⇒ PD 意图对策略可见。**并按用户要求关掉了对应的奖励**：`yaw_heading` 不再注册（PD 已逐拍闭环，再叠一个奖励等于同一目标约束两次、还会掩盖 PD 的效果），`tracking_velocity` 改为 `use_lateral_and_heading=False`（只用纵向 `vx`）⇒ 横向/朝向的跟踪误差**不再进回报**；helper `heading_deviation`／`yaw_heading_l2` 保留在 `upper_mdp` 以便回开 |
| TOW-15 | P1 | **已修（离线契约通过）；Isaac Lab 未跑** | 2026-10-09 排查「heading 不稳」时发现：Isaac Lab 的 `isaaclab.utils.math.euler_xyz_from_quat` 返回顺序是 **(roll, pitch, yaw)**，而 `upper_mdp.heading_deviation` 原先写的是 `yaw, _pitch, _roll = euler_xyz_from_quat(...)`——**实际取到的是 roll**。后果：`yaw_heading`（权重 −2.0，2026-09-23 为修「开始就在自转」而加）一直在罚**横滚**偏差，而朝向只被 `tracking_velocity` 的 yaw **角速度**项间接约束；角速度归零 ≠ 朝向不漂，所以「朝向保持」这条从未真正生效，2026-09-23 那次提交的结论（`f028802` 系列）需按此重新评价。**修**：新增单一入口 `upper_mdp.yaw_from_quat(quat)`（取第三个分量），`heading_deviation` 与新增的横向/朝向 PD 统一走它；**验证**：新增纯标准库数值守卫（用纯 yaw 四元数按 Isaac Lab 公式复算，断言 roll/pitch≈0、yaw 在第三个分量）＋静态守卫（禁止 `...[0]` 与 `yaw, _pitch, _roll = ...` 回潮）；全量 `imgo2_rl/tests` **433 项通过 0 失败**。**未验证**：本机无 Isaac Lab，未实跑；修好后 `yaw_heading` 的量级会变化（原先量的是 roll），旧 TensorBoard 读数不可对照 | 该项奖励已随 PD 方案**关闭**（见 TOW-14），权重重估不再需要；要确认的是 **PD 拿到的 yaw 是对的**（给机器人一个已知 yaw，看 `_lane_keeping_wz` 的符号与量级是否合理——若仍取到 roll，符号会随横滚变化而非随偏航） |
| TOW-EXP-01 | P1 | **仿真实验章证据不足；最新力估计统计待修** | 2026-10-11 审阅远程11394e8：任务必要性与初步收益已有证据，但关节响应、受控质量辨识、模块贡献和正常越坡停车尚未形成完整实验链；测试台力模长切片把质量算入力。详见 [充分性评估](docs/maintenance/project_history_2026-09_to_2026-10.md#towing) | 修正力误差生产并复算；补固定条件质量扫描、表征/双头/简单基线消融及可控重复，归档最新原始报告；不可将去掉短连接当作全域安全证据 |
| TOW-16 | P1 | **已加（离线契约通过）；Isaac Lab 未跑** | 用户 2026-10-09 认可加入「支撑脚不许打滑」。**依据**：诊断「负载下速度跟踪变差、而角度跟踪没变差」时列了三条机制——① 力由地面反力承担而非关节挠度（力学必然：10 N 对 5.5 kg 是1.85 m/s² 的不平衡，但挂点只偏后 0.16 m ⇒ 附加俯仰力矩约 0.3 N·m，关节只需 Δq≈Δτ/kp≈0.01–0.05 rad）；② 足端打滑（**此前未测**，场景里没有足端接触传感器）；③ 残差权限有限（δ 只有 ±0.125/±0.25 rad，而底层动作权限 ±3.0 ⇒ 想靠改步态找回速度也提不动）。**改**：场景新增足端接触传感器 `foot_contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*_FOOT", history_length=3)`；`upper_mdp.feet_slide` 按 **PPO rough 的正本**实现（Isaac Lab `isaaclab_tasks/.../locomotion/velocity/mdp/rewards.py`：`Σ_feet ‖v_foot,xy‖·1(历史最大合力>1 N)`），但**按传感器 `body_names` 映射到资产 body id**，不依赖正本「传感器与资产按下标同序」的隐含假设（配错会静默把 A 脚速度算到 B 脚上）；注册 `feet_slide = RewTerm(..., weight=-0.05)` 与 PPO 同权重。**验证**：全量 `imgo2_rl/tests` **434 项通过 0 失败**（新增守卫：传感器路径/历史长度、与正本同式的四个关键表达式、权重与 param、按 body_names 映射、缺传感器/体名不匹配必须报错）；`compileall` 通过。**未验证**：本机无 Isaac Lab，传感器能否解析出 4 个足端 prim、`net_forces_w_history` 的形状、以及着地判据阈值 1 N 是否合适都待训练机 | 训练机跑短冒烟 → 读 `Episode_Reward/feet_slide`：着地脚静止时应 ≈0；若它明显 >0 再决定是否加权/改成带死区的 `relu(‖v_t‖ − v_slip)²`。同时用它回答第②条是否成立（配合 `--no-cart-fraction` 的无负载对照） |
| TOW-17 | P1 | **已改（离线契约通过）；Isaac Lab 未跑** | 用户 2026-10-09 要求：小车质量上限提到 **30 kg**、**机器人出生固定**、机器人侧域随机化**参考 AMP vel 跟踪任务恢复**。**改**：`mass_range (5,15)→(5,30)`（契约字段 `DecoderSpec.mass_range` 与测量台 `TRAINING_MASS_RANGE_KG` 同步，测量台的默认质量档 5/10/15/20/25 当时**全部落在训练分布内**）；**2026-10-10 再收紧为 (5,20) kg / 速度 0.5–1.5 m/s**（用户确认工作域）：`mass_range`、`DecoderSpec.mass_range`、测量台 `TRAINING_MASS_RANGE_KG` 三处同步改，测量台的 **25 kg 现为域外**（保留作外推检查），超时随之 29.231 → 24.185 s（585 → 484 步），契约不变并有守卫，见 [工作域收紧记录](docs/maintenance/project_history_2026-09_to_2026-10.md#towing)；`robot_x/y/yaw_range` 三个抖动全部归零；新增 `UpperRobotDomainRandCfg` 四项，参数逐项照抄 AMP（`velocity_env_cfg.EventCfg` + `base_move/amp_env_cfg` 的收窄）：base **加**质量 (−1,3) kg、其余**乘** (0.7,1.3)、base 质心 ±0.05 m（都 `recompute_inertia=True`）、12 个策略关节执行器增益 **reset 模式**乘 (0.5,2.0)。**刻意不恢复两项（照抄会变空操作，已在 docstring 与守卫里写清原因）**：① `randomize_rigid_body_material`——本任务每回合都会把 robot/cart 的 material 摩擦整体改写，startup 的随机化第一回合就被覆盖；② `randomize_apply_external_force_torque`——取绳物理每个物理步都 `set_external_force_and_torque(..., body_ids=[base])`，reset 时写入的随机力 5 ms 后即被冲掉；要真恢复必须把这份力**加进取绳合力**（待用户定）。**验证**：全量 `imgo2_rl/tests` **436 项通过 0 失败**（新增 AMP 逐项对齐守卫 + 出生固定守卫）；`compileall`、`git diff --check` 通过。**未验证**：本机无 Isaac Lab，四个新 EventTerm 的运行时行为（`randomize_rigid_body_com`、reset 模式的增益缩放）与 30 kg 下的动力学都未实跑 | 训练机跑短冒烟：确认四项 DR 生效（质量/质心可在构造后读回、增益每回合变化）、30 kg 档不出现新的失稳；并决定 external force 那项是否加进取绳合力 |
| TOW-18 | P0 | **已改（离线契约通过）；Isaac Lab 未跑** | 用户 2026-10-09 三项决定：**①绳 0.5–1.5 m、②杆 0.5–1.0 m、③后车碰撞由终止改为「更小的持续惩罚」**。**依据**：TOW-11③ 实测 compliant 最短行（L=0.6、出生比 0.5）的「机器人后表面↔车头」真实几何间隙只有 **8.8 mm**，导致约 39% 的回合以碰撞终止；根因是松弛靠在**把小车拉近**实现，而后腿比挂点多伸 0.238 m，把松弛量吃掉了。**改**：①长度按类型解耦（`connection_grid.ROPE_LENGTH_{MIN,MAX}_M = 0.5/1.5`、`RIGID_LENGTH_{MIN,MAX}_M = 0.5/1.0`，新增 `row_length(row, model_name)` 与 `LENGTH_RANGES` 表，旧名 `LENGTH_MIN/MAX_M` 保留为绳区间）；②出生比 `SLACK_RATIO 0.5→0.8`（仍 <1 ⇒ 绳仍松弛、第一拍无约束力）；③`UpperTerminationsCfg` 删除 `cart_collision` 这个 DoneTerm，`collision` 权重 **−50→−5.0**（从一次性终止改为可持续惩罚，故小一个量级：每步 −0.25，顶住 1 s ≈ −5，与一个回合的正奖励同量级）。**静态几何结果**：绳最短行真实间隙 **124 mm**（原 9 mm）、杆最短行 **232 mm**，全网格最小 124 mm；绳 L=1.5 时小车在 −1.598 m，仍在车道后向边界 −1.65 m 内 ⇒ **BACK_M 不需要改**。**验证**：全量 `imgo2_rl/tests` **437 项通过 0 失败**（网格/测量台测试改为按类型区间；新增守卫：碰撞不得注册成 DoneTerm、权重字面量 −5.0、`cart_collision` 函数必须保留；`min_clearance` 的「spawn 不生效」按两类型最短长度各自复算）；`compileall`、`git diff --check` 通过。**未验证**：本机无 Isaac Lab，未实跑；**动态**足端摆动会消耗一部分静态间隙（最短绳行 124 mm 仍可能在步态里蹭到），且「碰撞不终止」之后的学习行为（会不会学成「顶住不脱离」）没有任何运行证据 | 训练机跑 64 环境 × 20 轮冒烟：看碰撞占比（原 ~39%）与 `Episode_Reward/collision` 的量级；若最短绳行仍频繁接触，把绳下限提到 0.6–0.7 或出生比提到 0.9 |
| TOW-19 | P1 | **已评估（有实跑数据）；碰撞模式待用户定** | **用户观察**：「终止条件改了之后，机器人似乎不避讳碰撞了」——数据支持这个判断。同一套 4096 环境、同 seed 下对照（旧配置=碰撞终止+权重−50；新配置=不终止+权重−5）：**每步碰撞率** 0.082%（危险率，因为一碰就结束）→ **0.386%**（占用率，接触可持续）/ 末轮 0.52%；**每回合接触步数** 旧 ≈30% 的回合至少命中一次 → 新 **2.25 步/回合**；**有效威慑力**（换算成奖励单位）旧 ≈ **9.9**（损失剩余步数×每步奖励 7.4 + 一次性 2.5）→ 新 ≈ **0.56**（2.25 步 × −5×0.05）⇒ **掉了约 18 倍**。**还没发生的捷径**：TOW-03 完成标准点名的「为卸载拉力而让小车逼近」目前**没有证据**——牵引力均值仍 7.90 N（绳绷着）、拉力方向偏离只有 2.7°、`robot_fall`/`terrain_exit` 全程 0、出生类惩罚 `clearance` −0.0093／`min_clearance` −0.0010（≈0）。**恢复避讳的三个做法（推荐度递减）**：①**首次接触一次性重罚**（保留不终止，补回"终止级"代价，一次性权重约 −200 ≈ 旧的 10 奖励单位）；②**连续接触 N 步才终止**（"碰一下没事、顶着不放就结束"）；③把权重提到 −20~−50（持续接触会累积成灾难性惩罚，有"卡住即废"风险）。**用户 2026-10-09 要求**：把"是否启用上层网络"做成**本质测试台**（`play_towing_test.py`）里的开关——该台子当前**故意不加载任何上层 checkpoint**（跑冻结策略 + 脚本指令的基线），所以「策略 vs 基线」的同网格同指标对照目前**做不到**；这是下一步要加的东西。**未验证**：上述碰撞对照是 TensorBoard 训练量（危险率 vs 占用率两种口径不可直接比），不是独立评估 | 先跑 `towing/play.py` 评估 `model_1000.pt`（实测 decoder MAE / 停车指标），再给测试台加上层网络开关做对照，然后决定碰撞模式 |
| TOW-20 | P1 | **已加开关（离线契约通过）；Isaac Lab 未跑** | 用户 2026-10-09/10 要求把「是否启用上层网络」做成必要性测试台（`play_towing_test.py`）里的开关——TOW-19 记录该台子当时**故意不加载任何上层 checkpoint**，「策略 vs 基线」的同网格同指标对照做不到。**改**：新增 [upper_policy_runtime.py](imgo2_rl/scripts/towing/upper_policy_runtime.py)（纯 torch + 标准库，惰性加载 rl_lab 的 actor/decoder，**不 import Isaac Lab**）：57 维帧（`cat(loco_command 3, last_action 12, base_ang_vel·0.25 3, projected_gravity 3, last_loco_action 12, joint_pos−default 12, joint_vel·0.05 12)`，与 `upper_mdp.policy_frame` 逐项一致）→ decoder（`.eval()`、`sample=False` ⇒ latent 取均值）→ 79 维 actor（`act_inference` 确定性；`--upper-stochastic` 可采样）→ `delta = clamp(action,±1)⊙action_scale`（action_scale 取冻结策略契约：hip 0.125 / thigh·shank 0.25）；checkpoint 用 `torch.load(..., map_location="cpu", weights_only=False)`，加载前**硬校验** `towing_contract = {version:2, frame_dim:57, explicit_dim:6, latent_dim:16}`（旧 56/63 维 checkpoint 直接抛 `TowingCheckpointContractError`），网络超参从注册的 `agents/upper_ppo_cfg.py::UpperTowingPPORunnerCfg` 建、`load_state_dict(strict=True)`，另做零帧前向自检。测试台加 `--upper-checkpoint PATH`（**默认 None = 基线，残差恒 0，行为与加开关前逐位一致**）与 `--upper-stochastic`（默认确定性均值）；开关状态进 `[plan]`、`experiment.json.upper_policy`（含 contract/iter/actor 79/critic 72/残差尺度）与 `report.json.arguments`。节拍按训练侧：上层每 `upper_control_dt = 0.05 s`（10 个物理步）推理一次、两次之间残差不变，冻结策略每 4 步刷新并重算 `held = 冻结目标 + delta`，**`held` 既下发又作为 `robot_jt_*`/JNT 参考**（与训练侧 `low_level_position_error_l2(reference="commanded")` 同口径）。**验证**：全量 `imgo2_rl/tests` **473 项通过 0 失败**（新增 `test_towing_upper_policy_runtime.py` 24 项：帧布局与 `UpperObservationSpec.terms` + `policy_frame` 源码 AST 逐项一致、actor 79 / critic 72（按 `upper_env_cfg.CriticCfg` AST 求和）、注册 cfg 与 `UpperNetworkSpec` 逐项一致、合成 checkpoint 往返的残差 clamp/`_last_action` 时序/全量与部分 `reset`、contract 不匹配（56/63/version/latent/非 dict）与缺键/缺文件/伪造契约被 strict 加载拒绝、模块顶部无 isaaclab 且只有 stdlib+torch；`test_towing_play_test.py` +12 项：默认关、plan 行显示开关与契约、缺文件/单独 `--upper-stochastic` 报错、`--compare-report` 的 CLI 校验、**开关关时不构造 runtime**、held 唯一且只在开关开的分支里、模块顶部无 torch、可追溯字段（plan/experiment.json/report.json 记开关+checkpoint+iter+towing_contract+deterministic/stochastic）、「策略 vs 基线」汇总（通过数/判定码分布/五项指标中位数/样本数/逐坡度通过数）、归档基线常量与 `docs/towing/data/2026-10-09_necessity_800_noload/report.json` 复算值逐项一致、对照节的 JNT/出处/不可硬比提示、**判据与阈值函数里没有任何开关分支**（同指标/同阈值/同判定码））；`python3 imgo2_rl/scripts/towing/play_towing_test.py --dry-run` 仍只用标准库可跑（给 checkpoint 时验证 `torch` 不在 `sys.modules`）；`py_compile`/`git diff --check`/tracked-ignore 通过。**未验证**：本机无 Isaac Lab（无 GPU 可见性），**仿真一次都没跑**——残差接到 PhysX、20 Hz 节拍、`held` 下发与 JNT 参考、关节顺序/资产对接、真实 `model_*.pt` 的 strict 加载全待训练机；「关时逐位一致」是**结构性**结论（基线分支未改）不是运行读数。详见 [记录](docs/towing/towing_upper_switch_2026-10-10.md) | 训练机先跑 8 环境短回合冒烟（确认 `experiment.json.upper_policy.loaded == true`、contract/iter、残差非 0 且被 clamp 进 `±action_scale`），再跑两轮 **800 环境**、除开关外参数完全相同（一轮不给 `--upper-checkpoint`、一轮给 `.../model_1000.pt`），第 2 轮加 `--compare-report <基线>/report.json`，用 `report.md`/`report.json` 的「策略 vs 基线」一节读三项同口径结果（本轮/同版本基线/归档参照：通过数、判定码分布、五项指标中位数，另可查 `report.csv` 逐 cell 对照）；**验收结论只读 `COL`/`SPD`/`LOW`**（`JNT` 在基线的有负载/无负载两边都≈100%、量的是 PD 静差且开关前后参考量不同），然后据此回答 TOW-19 的碰撞模式选择 |
| TOW-21 | P1 | **待用户确认设计；未实现** | 用户 2026-10-10 两点判断：**①「用残差很难学会 cmd=0 之后继续前进」**（源码事实：STOP 后 `loco_command[:,0]` 归零 ⇒ 冻结策略被命令站住，残差只能把站姿重建成步态）；**②「抗冲击必须关节调整，停车后继续走两步可能要指令侧调整」**。方案：**上层网络改成双头**（cmd vel 头 + 现有 12 维关节残差头），动作 12 → 13（建议只开 vx，因为横向/朝向已定由 PD 负责）。**接口影响可枚举**（约 10 文件 + 部署侧）：帧 57→58、decoder frame_dim 58、actor 79→80、**checkpoint 契约 v2→v3（旧 checkpoint 失效）**、`rl_sdk.cpp`/导出件、`play.py`、TOW-20 的 `upper_policy_runtime.py`、契约测试。**最易漏且会静默毁训练的点**：新指令值绝不能同时喂奖励（必须拆 `task_command`（给奖励、到点归零）与 `loco_command`（给冻结策略+进帧））。另需处理 `post_stop_distance`（它现在明确惩罚「停车后继续走」）。**建议同时给脚本 ramp 作底**（否则偏移初值 0 ⇒ 行为仍要从零学）。**预研**：先用测试台 `--command-shaping ramp` 验证指令侧方向，再决定是否做契约迁移。详见 [双头方案记录](docs/towing/towing_upper_two_head_2026-10-10.md) | 先跑 towing test 口径的「基线 vs `model_1000`」验收，再看 ramp 预研结果 |
| TOW-22 | P1 | **已实现（离线测试通过）；Isaac Lab 未跑** | 用户 2026-10-10 要求「起步与停止的冲击时刻的关节响应必须独立统计」——只报全程/牵引段 RMS 会把冲击平均掉。**改**：`play_towing_test.py` 新增纯函数 `impact_stats()`（四个时刻各取窗口独立统计 + 稳态分母 + 比值/增量）：`impact.startup`（tow 首行起 `[t0, t0+W]`）、`impact.takeup`（首个 `\|rope_tension_n\| > --takeup-force-threshold` 的样本 ± W，**真实绷直时刻**）、`impact.stop`（coast 首行 = **指令归零**起 `[t0, t0+W]`）、`impact.stop_contact`（coast 首个接触见证 ± W，**真实停车撞击**，与归零窗口区分）、`impact.steady`（牵引段去首尾各 `--steady-margin-s`）；每个窗口 `joint_rms_rad`/`joint_max_rad`/`worst_joint`/`per_joint_rms_rad`/`torque_saturated_frac` + `rms_over_steady`/`rms_delta_rad`。CLI 三个新参数默认 0.2 s / 1.0 s / 1.0 N；`report.md` 增「## 冲击窗口 vs 稳态（关节响应）」一节、`report.json` 增 `impact_statistics`、`report.csv` 增 `impact_<窗口>_*`。**口径冻结**：既有 `startup.*`/`stop.*`（`--transition-window` 1.0 s）、`DEFAULT_THRESHOLDS`、判定码、`classify_case` 全未改；新参数不进判据（AST 守卫 + 同轨迹换窗口 `verdict` 逐位相同）。**验证**：`pytest imgo2_rl/tests -q` ⇒ **501 passed**（473 + 28：对齐/边界/无绷直/无接触/单步/空记录/尖峰不被平均掉/报告渲染/CLI/`runpy --dry-run` 无 torch）。**未验证**：本机无 Isaac Lab，未实跑；1 N 阈值、25 ms 记录下窗口样本数、比值能否区分基线/策略均待训练机。详见 [双头方案记录](docs/maintenance/project_history_2026-09_to_2026-10.md#towing) | 训练机按 TOW-20 的两轮命令跑 800 环境（基线 + `--upper-checkpoint .../model_1000.pt --compare-report <基线>/report.json`），先读 `report.md`「## 冲击窗口 vs 稳态（关节响应）」的 `绷直 RMS/稳态`、`归零 RMS/稳态`、`停车撞击 RMS/稳态` 三行的两轮对照，再回到 `COL`/`SPD`/`LOW` 判定码占比 |
| TOW-23 | P1 | **已改（离线测试与归档复算通过）；Isaac Lab 未跑** | **用户 2026-10-10 决定：关节跟踪误差（JNT）不进判定统计量。** 依据：`startup_joint_error` / `stop_joint_error` 两条在 800 cell 上**基线 800/800、策略 800/800**（零区分度）；它们量的是底层 PD 静差 `τ/kp`，而 `joint_rms_limit_rad=0.10` 连最好的 case 都超 1.6 倍（实测基线 startup RMS P1=0.161 / P50=0.221、stop RMS P1=0.126 / P50=0.154）⇒ **不是上层任务的职责**。**改**：`classify_case(..., count_jnt=False)` 新增口径开关（**默认 False**），`startup_joint_error`/`stop_joint_error` 不再产生失败原因与判定码（`JNT` 从判定码分布消失）；新增 `--count-jnt`（默认关）打开后与 2026-10-10 之前的旧口径**逐位一致**（供与旧归档对齐/追溯）；`startup.*`/`stop.*` 全部观测字段与 `--joint-rms-limit-rad` / `--joint-max-limit-rad` **保留**（观测，不是判据）；`[plan]` 行打印当前口径，`report.md` 限制节写明决定/理由/实测分布并保留起步·停车关节 RMS 中位数行（观测），`report.json`/`experiment.json` 的 `thresholds` 增 `"count_jnt"` 标记；归档对照 `docs/towing/data/2026-10-09_necessity_800*` 显式标注**口径不可比**（其通过数 0/800 与判定码 `JNT` 395 等是旧口径）。**实测重判（离线，两轮同版本 `git.commit=df593f49d6469be0ec1bbd2637fa66bb5616cb3c`、同几何、800 cell、`--no-cart-fraction 0.125`，来源 `imgo2_rl/logs/towing/play_test/upper_switch_{baseline,policy}/report.json`）**：剔 JNT 后**基线 356/800、策略 436/800**（旧口径两轮都是 0/800）；逐坡度量级 0°/5°/10°：基线 213/400、81/200、62/200 → 策略 245/400、96/200、95/200；按连接：compliant 152→174（/320）、inextensible 69→88（/160）、rigid 135→174（/320）；有负载 256→336（/700）、无负载 100/100（两侧相同）；剩余失败原因：基线 `stop_collision` 295 / `speed_track_error` 234 / `stop_margin_low` 28 → 策略 `stop_collision` 283 / `speed_track_error` 66 / `stop_margin_low` 36 / `lane_deviation` 1。**验证**：`pytest imgo2_rl/tests -q` ⇒ **509 passed**（新增/改写 8 项：默认不入统计、`--count-jnt` 逐位复现、报告渲染不含 JNT 失败行但保留观测行、plan 口径标记、两轮 `report.json` 离线复算逐项一致、`classify_case(count_jnt=True)` 对 800+800 case 逐位复现归档 verdict）；`--dry-run`（含 `--count-jnt`）退出 0 且 `torch` 不在 `sys.modules`；`py_compile`/`git diff --check`/tracked-ignore 通过。**未验证**：本机无 Isaac Lab —— 重判是**离线把 `cases[].verdict.reasons` 里的 JNT 剔除后重算**，不是重跑仿真；训练机需按 TOW-20 两轮命令实跑复核新口径通过数，并另跑一轮开 `--count-jnt` 与旧归档对齐。**待用户决定（本轮只记录不实现）**：`stop_collision` 的 `load_velocity_jump` 通道可能也是**过紧的占位阈值**（基线 `impact.stop_contact.channels` 首个见证构成 load_velocity_jump 254 / deck_contact_force 70，`deck \|fx\|` 中位 0 N；用绳拖 5–25 kg 车斗在 20 Hz 急停时速度在某一拍变化是物理必然）⇒ 建议像冲击统计那样改成**相对量**（负载减速度/与无负载对照的增量）。**2026-10-11 实测复核（v3 2700 轮 / 100 环境）给了这个待决定项的读数**：40 个 `COL` 里 **27 例只由该通道触发、既无甲板力也无表面穿透**（这 27 例 `min_clearance_coast_m` = +0.023…+0.295 m，25/27 ≥ 0.1 m），阈值实测为 **0.075 m/s / 25 ms** 而「最大跃变/阈值」中位仅 **1.4×**，且 40 例中 **12 例的最大跃变是「加速」**（弹性绳 `rope_taut` 0→1、张力 0→31…171 N）⇒ **建议把 `stop_collision` 收到硬证据口径（甲板力 or 表面穿透），或至少 AND 上几何条件**；按该口径本轮为 **85/100**（58+27）。该重判可**离线**完成（42 个失败 case 的 `tow.csv` 与逐例 `min_clearance_coast_m` 都在盘上，不必重跑仿真）。细节见 [记录](docs/towing/towing_verdict_jnt_excluded_2026-10-10.md) 与 [2700 轮复核](docs/towing/towing_play_test_2700_2026-10-11.md) | 训练机跑两轮 800 环境（基线 + `--upper-checkpoint .../model_1000.pt --compare-report <基线>/report.json`），确认新口径下 `report.md` 结论/判定矩阵/`groups[*].reasons` 里**不出现** `JNT` 与两条 `*_joint_error`，且「策略 vs 基线」的起步/停车关节响应 RMS 中位数行仍在；再跑一轮加 `--count-jnt` 与旧归档（`docs/towing/data/2026-10-09_necessity_800*` 的口径）对齐，确认判定码与通过数逐位回到旧口径；最后就 `stop_collision` 的速度跃变通道是否改相对量给出决定 |
| TOW-24 | P1 | **已改（离线契约与文档复算通过）；Isaac Lab 未跑** | 用户 2026-10-10 批准两项奖励改动。**① 删除 `extra_distance`**（`mdp.post_stop_distance`，`weight=-0.1`）：与 `tracking_velocity` 在停车段高度冗余（指令归零后 `vx→0` 已隐含位移只剩不可避免的惯性滑行），且量级小 20 倍（每步 −0.0025 vs +0.050）⇒ 给「停车后继续走几步」凭空加反向梯度。**函数本体与 `post_stop_allowance_m` 形参保留**（docstring 明写「已不作为奖励项」），`UpperRewardsCfg` 里该注册删除；TOW-21 的 `post_stop_allowance_m` 随之**目前未接入奖励**（只作函数形参）；`--compact-log` 分项 picks 里 `extra_distance` → `min_clearance`。**② `min_clearance` 改「出生几何 − 绝对死区」阈值并提权**：`weight -2.0 → -5.0`（保持不变）、`params={"deadband_m": 0.02, "softness": 0.02}`（`ratio`/`spawn_margin` 均删除；2026-10-10 当天先落成 `spawn_margin(0.85)` 相对余量、同日再微调为绝对死区），函数内 `threshold = term.spawn_clearance − deadband_m`，出生间隙 = `sqrt(initial_attachment_distance(类型, L)² − Δz²) + base_offset`（绳 `spawn_ratio` = `connection_grid.SLACK_RATIO` = 0.8、杆 = 1.0；`Δz` = 0.17 m = 机器人/小车出生高度差；`base_offset` = 0.0025 m = `−robot_rear_surface_x − robot_attachment_x + cart_attachment_x − cart_front_surface_x`；全部从源码常量派生，缓存在 `HierarchicalVelocityAction.spawn_clearance`）。**为什么用绝对死区而不是相对余量**：①语义就是用户要的「不许比出生时更近」，`出生间隙 − 死区` 直接表达；②解析出生间隙与仿真实际间隙差几毫米（落地/穿透/沉降），2 cm 绝对值正好吸收，而相对余量对长绳行浪费几十厘米（L=1.5 时 0.15×1.1904 = 0.179 m）；③出生点在 lane **后向平段**（`slope_geometry` 剖面从 +2.25 m 才起坡）⇒ 出生/停车都在平地、无重力驱动 ⇒ **没有被动的间隙漂移**，小死区就够（旧顾虑「坡上会溜车所以要大余量」不成立）。**「高于阈值精确为 0」的既有实现原样保留**（带截断的 softplus：`relu(softplus(缺口/softness)×softness − softplus(0)×softness)`），本次只改阈值那一行。**逐行数值**：绳行（L 0.5–1.5 m，20 行）阈值 **0.3446–1.1704 m**（= 0.689–0.780·L；L=0.5 → 0.9451×出生间隙、L=1.5 → 0.9832×），牵引段绳绷直时间隙 > 出生间隙 不触发、绳一松/车斗逼近穿过阈值才开始出力（＝「不许比出生时更近」+「停车后继续走几步」的梯度）；杆行（L 0.5–1.0 m）阈值 **0.4527–0.9679 m**（= 0.905–0.968·L），连杆固定挂点距 ⇒ 在出生高差上间隙恒等于出生间隙 ⇒ **永不触发**（但余量只剩 `deadband_m`：触发需 Δz 再大 **+0.0475 m**、最短杆 Δz*=0.2175 m；旧相对口径是 +0.131 m / 0.301 m —— 这是本口径的代价）。**停车段天然生效**：实测停车瞬间间隙 ≈0.64·L（归档离线复算 `stop.clearance_at_stop_m/L` 中位 0.700、compliant 0.702）、新阈值 ≈0.945×出生间隙 ≈0.69·L ⇒ 停车段间隙天然落在阈值之下约 0.05·L（L=1.0 时缺口 ≈0.05 m ⇒ 每步 −5×0.05×0.05 ≈ −0.0125＝线性铰链值，截断 softplus 实际 ≈−0.0094），与 `tracking_velocity` 满额 +0.05/步 同量级 ⇒ **不需要**再单独做「用停车瞬间间隙当参考」的方案；**备选**（只在训练机实测停车段间隙反而高于阈值、本项不生效时才启用）：在 `process_actions` 记录 `gap_at_stop`，阈值改用 `gap_at_stop − deadband_m`。**权重依据**：每步代价 = `w × 缺口 × step_dt` ⇒ w=−5 时超阈值 0.2 m ≈ **−0.05/步** = `tracking_velocity` 满额（+0.05/步）⇒ 有动机但不压倒；旧的 −2 只有 −0.02/步（40%），停车段惯性/车重面前太弱。**口径变更**：`docs/towing/data/2026-10-09_*` 归档的 LOW 计数用的是测试台**绝对 0.10 m** 判据、且当时训练侧是 ratio 口径 ⇒ **不可比**；测试台 `gap_margin_limit_m`（`play_towing_test.py`，绝对 0.10 m）本轮**未改**（用户要求），与训练侧新阈值不同口径。**验证**：`pytest imgo2_rl/tests -q` ⇒ **551 passed**（上一基线 550 + 1 条新增；`min_clearance` 契约测试覆盖用户点名的六条：①`deadband_m > 0`；②spawn 处精确为 0 且余量恰为 `deadband_m`（把源码函数抽出在 stub term 上**实跑**）；③逐类型逐行阈值 = 出生间隙 − 0.02；④阈值随 L 单调；⑤杆行永不触发（恒等式 + Δz ±0.04 扫描 + Δz* 二分 0.2175–0.2606 m）；⑥阈值恒 > 0（最小 0.3446 m）；另有 AST 断言奖励表无 `extra_distance` 且把 `post_stop_distance` 函数体抽出实跑、源码守卫旧 `spawn_margin`/`ratio` 口径不得回潮，以及把 [奖励改动记录](docs/towing/towing_reward_retune_2026-10-10.md) 的逐行表与源码复算到 4 位小数）；`py_compile`/`git diff --check`/`git ls-files -i -c --exclude-standard` 通过。**未验证**：本机无 Isaac Lab，仿真一次都没跑 —— `spawn_clearance` 缓存与出生实测间隙是否逐 env 相等（2 cm 死区是否够吸收那几毫米解析-仿真差）、−5.0/「出生间隙 − 0.02」在牵引段与停车段的实际量级、**停车段是否真的自动生效**（证据来自 L 0.6–1.2 m 的旧归档离线复算，最短绳行 L≈0.5 在停车瞬间可能擦边不触发）、杆行 Δz 余量只剩 +0.0475 m（最短杆）、删掉 `extra_distance` 后停车段是否需要重标 `stop_towing_force` 都待训练机；8 份旧记录 + 2 份归档 README 已加「2026-10-10 变更」注、口径改为「出生间隙 − `deadband_m`」。详见 [奖励改动记录](docs/towing/towing_reward_retune_2026-10-10.md) | 训练机跑 64 环境 × 20 轮冒烟：**reset 后第一拍**读 `Episode_Reward/min_clearance`（必须精确为 0，否则缓存 `spawn_clearance` 与实测几何有偏差、或 2 cm 死区不够吸收解析-仿真差）；再跑一段 run，按相位看该分项量级（牵引段应 ≈0、停车段随车斗逼近上升）与 `stop_towing_force`／`tracking_velocity`／`obs_stop_reached` 的对比；**专门核对停车段**——按连接类型/行分组读停车瞬间与滑行段最小的 `Episode_Reward/min_clearance`，若在停车段整体为 0（尤其最短绳行 L≈0.5）就按记录 §2.4 的备选改成 `gap_at_stop − deadband_m`；确认杆行全程恒 0（**最短杆最先失守**，Δz 只需漂 4.75 cm）；最后决定删掉 `extra_distance` 后停车段奖励是否需要重标 |
| TOW-25 | P1 | **已加（离线契约通过）；Isaac Lab 未跑** | 用户 2026-10-10 要求：「加一个拉力变化率的惩罚。但是考虑绳子瞬间卸载，就给一个掩码，在绳子卸载的时候拉力允许突变」。**物理理由（本项存在的依据）**：绳是**单边约束**（只能拉、不能推）⇒ **不存在"用绳缓慢刹车"**；绳行的停车机制是**绳松弛 + 车斗自由滑行 + 机器人往前走两步避让**，绳的**瞬间卸载（松弛）是正常且期望的** ⇒ 那一下必须允许突变。刚性杆是**双边约束**，可以主动管理力 ⇒ 杆行的目标是**缓慢卸载、避免冲击力过大**（力尖峰要罚）。**语义**：罚"拉力幅值的突变"，但放行"卸载到松弛"那一下——绷紧侧（`rate>0` 快速加载 = 被猛拽/绷直冲击）**要罚**；卸载且已掉到 ≈0（绳松弛 / 杆卸载过零）**放行**；**仍是负载态却断崖式掉力**才罚（`down` 那支）。**公式**（`mag = ‖towing_force_b‖` N、`dt = env.step_dt = 0.05 s`）：`rate = (mag_t − mag_{t−1})/dt`；`up = relu(rate − rate_limit_n_per_s)/rate_limit_n_per_s`；`slack = mag_t < slack_eps_n`；`down = relu(−rate − release_limit_n_per_s)/release_limit_n_per_s × (1 − slack)`；`penalty = clamp(up + down, 0, clip_max)`（无量纲）。**默认参数**（全部是 `RewTerm.params`）：`rate_limit_n_per_s=200.0`、`release_limit_n_per_s=600.0`（卸载侧更宽松）、`slack_eps_n=1.0`（与既有 `obs_towing_force_active` 的 1 N 阈值**同口径**）、`clip_max=3.0`；**权重 −0.5**（初始保守、容易调）。**标定**：`up` 在 `rate = 2 × limit` 时 = 1.0 ⇒ 每步 `−0.5 × 1.0 × 0.05 = −0.025` ≈ `tracking_velocity` 满额（+0.05/步）的**一半**（一次明显猛拽 ≈ 半秒跟速收益）；调参只需改 `rate_limit_n_per_s`。**实现要点**：`mag_{t−1}` 缓存在 `HierarchicalVelocityAction.prev_force_mag`，**只在 `process_actions`（每个控制步一次，与 `_previous/_processed` 同一处）更新，不在 200 Hz 物理子步里**；`reset()` 清零 ⇒ **reset 后第一拍 rate 精确为 0**（没有上一拍样本时不拿 0 当上一拍去算，与仓库"spawn 精确为 0"的纪律一致）；新项不读 `loco_command`（既有 AST 防作弊守卫自动覆盖）。**20 Hz 采样限制（写进 docstring，不当缺陷）**：一次在一个控制步内完成的"猛拽→回弹"冲量在 20 Hz 有限差分上可能与"缓慢加载到同一水平"完全一样，本项对它没有梯度 ⇒ 另加诊断 `obs_force_rate_max = RewTerm(func=mdp.obs_force_rate_max, weight=1.0e-6)`：200 Hz 子步内 `max \|Δmag\| / dt_phys`，每控制步取最大、**读后清零**，用来量化"被 20 Hz 采样漏掉了多少"。**不动** `stop_towing_force`（−1.0）、`min_clearance`（−5.0 / `deadband_m` 0.02）与已删除的 `extra_distance`（有契约测试钉住）。**验证**：`pytest imgo2_rl/tests -q` ⇒ **550 passed**（526 基线 + 24 新增，`test_towing_force_rate.py`）；`py_compile` 改动文件、`git diff --check` 干净、`git ls-files -i -c --exclude-standard` = 0 条。**未验证**：本机无 Isaac Lab，**仿真一次都没跑**——`rate_limit`/`release_limit`/权重一个都没实跑标定；20 Hz vs 200 Hz 的盲区程度只有诊断量能给（无读数）；`force_rate_max` 在真实子步上的量级与 `process_actions`/reward 的调用时序只在训练机可验。详见 [拉力变化率记录](docs/towing/towing_force_rate_2026-10-10.md) | 训练机先跑 64 环境 × 20 轮冒烟：① **reset 后第一拍**读 `Episode_Reward/towing_force_rate`（必须精确为 0）；② 读 `Episode_Reward/towing_force_rate` 与 `Episode_Reward/obs_force_rate_max ÷ 1e-6` 的比值——若后者远大于 `rate_limit_n_per_s`（200 N/s），说明真实冲量主要发生在 5 ms 子步尺度，**要提控制频率/改物理侧限速，而不是继续加这一项的权重**；③ 按相位看量级（牵引起步的"绷直"应有明显负值，绳行的松弛卸载那一下应为 0、杆行的卸载若断崖则应被罚）；④ 与 `tracking_velocity`/`stop_towing_force` 对比确认不压倒跟踪项，再决定是否把 `rate_limit_n_per_s` 调小/调大 |
| TOW-26 | P2 | **已修（离线契约通过）；训练机验证未做** | 2026-10-10 工作域收紧时顺带发现：`play_towing_test.py` 仍把上层 checkpoint 的**期望契约**写成 **v2 / `frame_dim=57`**（docstring、`[plan]` 打印行、`experiment.json.upper_policy.towing_contract_expected`），而双头迁移后现行契约是 **v3 / frame_dim=58**（`upper_policy_runtime.CHECKPOINT_CONTRACT_VERSION = 3`）。**影响**：不影响运行——实际加载与硬校验都走 `UpperPolicyRuntime`（v3/58），那三处只是记录/打印**失真**。**已修**（2026-10-10 晚，同一轮把测试台接到 v3 双头契约）：① 三处记录改 v3（帧 58 / 动作 13 / explicit 6 / latent 16），plan 行另外打印**offset 有界**（`clamp(clamp(u_cmd,±1)·0.5, −0.2, +0.6)` m/s，再把和裁进 AMP 包络 −1.0…1.5）与**门控**（`elapsed_s ≥ tow_start_s`，STOP 之后仍生效）；② `experiment.json.upper_policy` 新增 `towing_contract_expected`（**从实际加载的 `upper.spec` 派生**，与离线镜像逐字段核对，不一致直接报错）、`frame_dim_expected`/`action_dim_expected`/`cmd_action_dim`/`joint_action_dim` 与 `offset` 子字典（scale/min/max/amp_vx_range/gate/gate_after_s），`baseline_comparison` 记 `policy_action_dim=13`（`upper_policy_enabled`/`upper_contract_version`）；③ 接线：帧 58（`last_action` 13 维）、动作 13 切分、**偏移在冻结策略推理之前**合成 `loco_command`（两层限幅 + 门控）、残差在底层输出之后叠加 ⇒ `held = 冻结目标 + delta` 既下发又作为 `JNT` 参考；上层 20 Hz tick 与冻结策略 4 步刷新**解耦**（原先嵌在刷新门里 ⇒ 实际只有 100 ms）；④ 关开关时基线分支**逐字未改**，并由离线测试逐字钉住。**验证**：`pytest imgo2_rl/tests -q` ⇒ **563 passed**（553 基线 + 10 项 `UpperV3WiringTests`：契约/偏移常量与 `upper_policy_runtime` 逐项一致、v2 契约被拒、门控阈值 = `episode_geometry.SETTLE_TIME_S`、偏移为 0 时与任务指令 `torch.equal`、门控与两层限幅、与运行时 `compose_loco_vx` 逐位相等、`upper_policy_record()` 字段、链路顺序 + 基线路径逐字钉住）；`--dry-run`（默认与 `--upper-checkpoint`）退出 0 且 `torch` 不在 `sys.modules`；`py_compile`/`git diff --check`/tracked-ignore 通过。**未验证**：本机无 Isaac Lab，**仿真一次都没跑**——偏移门控/两层限幅的实际读数、20 Hz 与 50 Hz 的解耦、真实 v3 checkpoint 的 strict 加载、两轮 800 环境的策略/基线读数都待训练机。详见 [测试台上层开关记录](docs/towing/towing_upper_switch_2026-10-10.md) 的「v3 接线」一节 | 训练机跑两轮 800 环境（除开关外参数完全一致）：第 1 轮基线、第 2 轮 `--upper-checkpoint .../model_*.pt` + `--compare-report <基线>/report.json`，先确认 `experiment.json.upper_policy.loaded == true` 且 `towing_contract` = v3/58/6/16、`offset.gate_after_s == 1.0`，再用 `report.md` 的「策略 vs 基线」一节读 `COL`/`SPD`/`LOW`（`JNT` 默认已移出判定） |
| CART-01 | P1 | **基础模型已验证** | 小车落地、轮接触、质量惯量、黏性轮阻和自由滑行已通过训练机实验；关键数据见 [拖曳仿真验证](docs/towing/towing_simulation_validation.md) | 后续只在模型或记录接口变化时回归，不重复展开早期调试历史 |
| TOW-05 | P0 | **MuJoCo 场景已修，待动态验证；完整 sim2sim 未闭环** | 新增 5–15 kg 小车、四轮阻尼、弹性／不可伸长两种单边绳场景，静态契约通过；尚未编译或运行 MuJoCo，且上层策略尚无可部署导出件，Gazebo 小车场景未接入。详见 [拖曳 sim2sim 记录](docs/towing/towing_sim2sim_2026-09-24.md)。**2026-09-25 入库占位权重（用户决定）**：`imgo2_deploy/ros2_ws/policy/imgo2/towing/policy.pt` 是 `logs/towing_rl_lab/towing_upper/2026-09-23_22-19-00/model_2000.pt` 的原样拷贝（sha256 `a4d800a26c1d1a50…`，9.76 MB，含 optimizer／decoder／normalizer 的**训练态** checkpoint），**不是** `play.py` 导出的 TorchScript，libtorch 加载器读不了；该目录**没有** `config.yaml`，FSM 也**没有**对应按键／状态类。`towing/play.py` 的导出分支目前为 0 处（ppo／amp／himloco 才有）。详见 [回放记录](docs/towing/towing_play_2026-09-25.md) §9 | MuJoCo 编译／加载与 5、10、15 kg 拖曳停车回放通过；训练侧 checkpoint 经 `play.py` 导出并接入 51 维双 GRU 观测／控制链；Gazebo 场景实现并用相同案例对照速度、停车间隙和碰撞 |
| CART-01 | P1 | **基础模型已验证** | 小车落地、轮接触、质量惯量、黏性轮阻和自由滑行已通过训练机实验；关键数据见 [拖曳仿真验证](docs/towing/towing_simulation_validation.md) | 后续只在模型或记录接口变化时回归，不重复展开早期调试历史 |
| DOC-01 | P1 | 已完成 | 训练 README 曾引用失效的 `script/himloco_rsl_rl` 安装路径和写死的个人服务器 checkpoint | 已改为 `scripts/rl_lab`，checkpoint 改为 `<run>` 占位并注明不可沿用；根 README 与子项目 README 表述一致 |
| PPO-01 | P1 | 本机指定环境短训通过；服务器/跨版本完整恢复待确认 | 11/10指定env_isaaclab与PYTHONPATH，5300候选/对照各8环境2轮实际通过；显式warm_start用官方RSL5迁移恢复模型/轮次且重建优化器，源模型未改，见修复与限制（本机 `low5300_retrain_2026-11-10.md`）。 | 原默认解释器路径、服务器安装/旧RSL运行、原优化器跨版本连续恢复及完整步态效果仍待；普通resume旧格式路径未自动迁移，不把warm_start当完整状态恢复。 |
| ENV-01 | P0 | 代码已修正，待服务器验证 | 原先 `assets/imgo2.py` 的两条路径写死为 `/root/gpufree-data/Imgo2_rl/...`（当时目录名还是 `Imgo2_rl`），且假定该目录就是项目根，合并成 monorepo 后必然失效且 glob 为空时无明确报错 | 已改为由 `Path(__file__)` 推导项目根，并支持 `IMGO2_AMP_MOTION_DIR`／`IMGO2_URDF_PATH` 覆盖；新增 `scripts/tools/check_asset_paths.py` 供新机器自检（本机通过：URDF 存在、动作文件 21 份、无残留机器路径）。服务器上仍需运行该脚本并记录实际加载路径 |
| MODEL-01 | P0 | 已按训练侧参数统一并验证；Gazebo/ROS 侧待复核 | **用户决定：四份模型的物理参数一律以训练侧为准。** 已按此统一：① 部署份 12 个腿部关节 axis 原全部取反（q_部署 = −q_训练）且限位镜像 → 改为训练份约定；② 部署份的足端 CoM/惯量与后腿 shank/thigh 惯量非对角项是五份模型中的**唯一异类**（`imgo2_model/` 片段、`imgo2_description` 两个 URDF、训练份四份一致）→ 改为训练侧；③ 足端碰撞圆柱长度训练侧与片段为 `0.01`、采集模型两个 URDF 与部署份为 `0.02`（2 对 3）→ 用户选训练侧 `0.01`，四份已统一；④ base 质量与惯量统一为 `5.53394020` + `0.03866860/0.10411461/0.12554111`。**现在四份完整 URDF 的 12 个关节轴/限位、17 个 link 的质量/质心/惯量/碰撞几何、base 规范值全部一致**（按逻辑关节名与逻辑 link 名比对；统一前容忍 `imgo2_description` 的有意腿序与 LF_/LH_ 命名，2026-09-17 统一后已同序，见 MODEL-02）。**为什么必须改部署份**：deploy 三条推理链路（`rl_real_imgo2.cpp` 走 Unitree SDK、`rl_sim.cpp` 走 ROS 控制器、`rl_sim_mujoco.cpp` 走 MJCF）均无任何符号取反，只有 `joint_mapping` 索引置换，无补偿；且 `robot_joint_controller` 会用 URDF 的关节限位 clamp 指令。注意两版原本行为不同：ROS1 自己定义了按引用修改的 `clamp(double&,…)`，确实生效；**ROS2 把 `std::clamp(…)` 当语句调用、丢弃返回值，等于完全没有限位**，且单关节版请求的参数名多写了一个下划线（`"robot_description_"`），URL 从未解析成功、`joints_urdf_` 为空指针。两处已在本轮修正，但尚未编译验证（见 DEPLOY-07）——旧限位 `thigh -2.87..0.9`、`shank 0.733..3.0` 会把 `default_dof_pos` 的 shank `-1.50` clamp 成 `+0.733`，即该链路连默认站姿都不成立，属功能性断裂。**验证**：四份 URDF 的 FK 全部重现录制数据（最差恒等 RMSE 0.00214 m，部署份修符号前为 0.11790 m）；`check_model_sync.py` 五项检查全通过；两次负向测试（翻回符号、单独改足端 CoM）均能报错，恢复后 sha256 不变。唯一残留差异是外观：部署份比训练份多 4 个 `<material>` 块，不影响物理 | ① 在 Gazebo/ROS 链路复核（真机与 MuJoCo 链路不读该 URDF，不受影响）。② 若要把 `imgo2_model/` 片段补成完整 URDF 并让 rl/deploy 共用同一文件，还需处理 ROS 包内路径（建议 CMake 构建时拷贝）与 `build.sh:57` 的存在性检查；在此之前「共用一份文件」尚未实现。③ 遗留项：`imgo2_description/xacro/common/leg.xacro` 与 `urdf/imgo2.urdf` 全工作区无人引用，可考虑删除（未动） |
| AMP-01 | P0 | 恒等映射由数据独立确认 | 用户确认未覆盖采集参数；脚本按默认 LF/RF/LH/RH 写入关节和足端，等价于训练端 `[FL,FR,RL,RR]`。本地补充：改用实际采集模型 `imgo2_description.urdf` 复算 FK 与训练 URDF 结果一致（恒等 RMSE 均值 0.00107 m，声明顺序配对 0.22491 m）；髋外展左右对称性和关节位置/速度块相关性检查也不依赖 URDF 支持同一结论 | 采集来源、源码顺序和数据本身均已确认；新训练的行为改善仍待服务器验证 |
| AMP-02 | P0 | 历史报错，未复现 | 历史草稿记录 `RuntimeError: normal expects all elements of std >= 0.0`，调用栈为 `amp_on_policy_runner.py:136` → `amp_ppo.py:120` → `actor_critic.py:129` 的 `distribution.sample()` | 记录复现命令、数据和首个异常值；修复后训练验证；不能仅凭该报错断定根因 |
| AMP-03 | P0 | **Torch 回归已于 2026-10-10 实跑通过**（此前记为「未执行过」是环境结论失效）；待做的只剩「在新训练里检查判别器与归一化统计」。**证据**：本工作区所在机器上 `python3` = `/opt/conda/envs/isaaclab/bin/python3`（Python 3.11.13、**torch 2.7.0+cu128**、numpy 1.26.0；注意 CUDA 在 harness 进程内仍不可见，同 TOW-04，`/root/IsaacLab` 有 Isaac Lab 源码树），`pytest imgo2_rl/tests -q` ⇒ **567 passed / 0 skipped**，其中 `tests/test_amp_alignment.py::AMPUpdateTests::test_raw_statistics_and_normalized_gradient_penalty` **ok**（不再是 skip）。⇒ 旧记录「本机五个 conda 环境与 `/usr/bin/python3` 都没有 torch」只对当时那台机器成立，**不要把「本机无 torch」当作跨机器结论**；同时说明本机跑不了的是 **Isaac Lab 环境构造**（`isaaclab.envs` 缺 `omni.log` / `omni.physics.tensors.impl.api`，无 Isaac Sim 运行时），不是 torch | 原均值方差更新使用归一化值，梯度惩罚使用未归一化值；已与 `amp_go2-main` 的处理方式对齐。**更正（2026-09-17 晚）：此条此前写成"Torch 回归已通过／6 项全过"是错的。** 本机（Linux `qmq-linux`）在 `~/miniconda3/envs/isaaclab` 下跑 `tests/test_amp_alignment.py` 的实际结果是 **5 通过 + 1 跳过**，跳过的正是本项 Torch 回归 `test_raw_statistics_and_normalized_gradient_penalty`（原因 `CPU regression needs the training environment's torch and numpy`）；全机五个 conda 环境与 `/usr/bin/python3` 都没有 torch（均报 `ModuleNotFoundError`），所以"该环境有 torch 2.7.0+cu128"的旧记录不成立，**AMP-03 的 Torch 那一半至今没有任何一次执行证据**（Windows 侧同样因缺 torch 跳过） | 需要一台真正装了 torch（有 Isaac Lab 训练环境）的机器跑 `python -m unittest discover -s tests -p test_amp_alignment.py`，确认 `test_raw_statistics_and_normalized_gradient_penalty` 不再 skip 且通过；之后再检查新训练中的判别器与归一化统计 |
| AMP-04 | P0 | 已加入待验证配置 | 原配置删除高度项和非法接触终止；现在保留 0.30 m 高度项、基座触地终止，并补偿任务奖励的时间步长缩放，任务混合系数改为 0.3 | 新训练高度稳定、贴地比例降低、速度跟踪可接受；具体权重仍需实验 |
| AMP-05 | P0 | **45 维 + 正式导出 + sim2sim 确认完成** | actor 移除 `base_lin_vel`（真机需状态估计）：`amp_env_cfg.py` 设 `observations.policy.base_lin_vel = None`，actor 45 维、critic 保留 48 维；部署侧 `amp/config.yaml` 为 45 维、观测项/顺序/缩放/动作/增益已与训练配置**逐项核对一致**。**24500 轮 checkpoint（`model_24500.pt`，sha256 `cc8ee22a…`；部署 `policy.pt` `cba59d44…`）已在 Gazebo 确认**：契约① 与 actor max diff 0.0、② 部署 libtorch 2.3.0 加载通过；按「每条命令一条新栈」测得跟速 **0.2 → 93.4%、0.3 → 97.9%、0.5 → 95.9%、0.75 → 98.7%、1.0 → 99.8%、1.2 → 99.8%、1.5 → 98.5%**，高度 0.277–0.300 m，`vx=0` 漂移 **0.001 m/s**、抖动 8.6/6.1/9.3（5000 轮时是 0.047 m/s、35.7/39.8/80.1）。**顺带实证了 AMP-08**：这份 checkpoint 的 `iter` 字段为 24500（不再是 0）。详见 [Gazebo 记录](docs/deployment/gazebo_ros2_bringup_2026-09-17.md) 9.7 节 | 剩余项：**步频仍是录制参考的约 2.5 倍**（周期 0.208–0.365 s 对 0.600 s / 1.67 Hz），且抖动绝对值随速度上升（shank 57→412）；若要"像真机录的那样走"，这是下一档目标。另：极低速 0.2 m/s 的周期强度只有 0.23（0.3 是 0.78）。**不得直接截掉旧网络的 3 个输入。** |
| AMP-06 | P0 | 配比已调好并训练验证；**步态在 sim2sim 侧已确认由「高频乱倒腿」变为「干净对角步」**；训练侧足端回放把剩余差距的根因定为**步幅偏小（0.61 倍）**而非相位乱 | ① **"一启动就飞"已修**：`PhysicsThread` 不套 keyframe；场景 base 高度改 `0 0 0.5` 后不再弹飞。② 抄入参考项目求解器/接触块后能站起。③ **MODEL-03**：`framequat` 挂 `imu` site 后站姿 0.2663→0.3015。④/⑤ 已排除「缺 `lin_vel` 观测」与「模型物理错误」（参考策略在我们模型上 `vx=0.5` 走 0.454 m/s）。**⑥ 2026-09-17 定位真因并调好配比**：三次训练对照——Run1 `1.0/0.3/-10` 只站不走（误差恒 1.72、1000 轮后指标全平）；Run2 `50/17/-1` 退化为贴地滑行（高度 0.172、贴地 0.89、std 失控 12.17）；本次 `4.0/2.0/-5` + 三项轻量姿态约束（`lin_vel_z_l2 -1`/`ang_vel_xy_l2 -0.05`/`joint_pos_limits -2`）**线速度误差 1.31→0.554 m/s**、高度 **0.305**、贴地 **0.08%**、触地终止 2.1%、超时终止 97.9%、std 稳定 0.394。**关键不是绝对权重，而是速度项与高度项的量级比**：Run1 惩罚是速度的 7.7 倍（不动）、Run2 速度是惩罚的 66 倍（趴滑）、本次 0.75:1 才同时拿住「走 + 站得住」 | **待修**：速度跟踪 kernel 仍仅 0.037（误差波动大，平均速度约为指令的 84%），24500 轮后 Gazebo 复测（每条命令一条新栈）周期强度升到 **0.71–0.83**、FL-FR 相位 174°–188°（trot），与键 1 参考策略（强度 0.96–0.97）同量级。**2026-09-18 训练侧足端回放（Isaac Lab，8 环境 × 950 步、丢弃前 50 步，`model_24500`）**：**唯一实质缺陷是步频 3.1 倍**（周期 **0.1926 s / 5.19 Hz** 对参考 0.600 s / 1.67 Hz；0.6 m/s 下同样是 0.198 s ⇒ **不是指令超范围**）。其余指标都对得上参考：**相位是干净对角步**（FL-FR **+174.2°**、FL-RR **+17.3°**、集中度 R **0.86–0.91**，与 Gazebo 侧 176.9° 相差 3° 以内）、**关节峰峰值落在参考范围内**（shank 0.516–0.618 对参考中位 0.635 / 范围 0.477–1.063；thigh 0.407–0.470 对中位 0.477）、抬脚 0.067–0.085 m（参考 0.083–0.121）、前后腿比 1.06（参考 0.87–1.06）、步幅 0.139–0.160 m（参考同档 0.295）、高度 0.3117 m、线速度误差 0.108 m/s。**逐条更正（v1→v3）**：v1 报的「高抬腿 1.6 倍」「前后腿不对称 1.33」「shank 幅度 1.35–2.25 倍」全部是**未 warmup 的出生瞬态污染**（加 50 步 warmup 后真值 0.066–0.083 m、1.06、0.52–0.62 rad）外加一处 `joint_pos` 堆叠轴 bug，已撤回；v2 报的「相位 156°、集中度 0.17」是相位估计器被非整数周期（0.1926 s ÷ 0.02 s = 9.63 步）抹平，换插值法后恢复。**为什么「加风格权重」不是解法（离线判别器分析）**：终局每步混合回报 1.97 = 任务 1.402（71%）+ 风格 0.540（29%），风格项按判别器标定可达 1.335；把策略的「关节角+足端+关节速度」三块换成专家能补回 **86%** 的风格差距（**关节速度块单独占 53%**，其 σ 是专家的 **1.7 倍**＝3 倍步频的印记），而姿态/足端的边缘分布几乎重合（σ 比 1.00）；**但判别器在策略点的局部梯度只有 8e-5**（1σ 全维扰动只移动 d 约 5e-4）⇒ 风格奖励是个没有方向的台阶，20000 轮里 d_policy 只动了 0.07。合成对照进一步指向 **λ_gp=10**：同预算下 λ 10→1→0 时「正常节律 vs 3 倍速节律」的风格奖励差为 **0.105 / 0.564 / 1.262** 每步。⇒ 优先级：**P1 降 λ_gp（10→1~2）**、**P2 加 `feet_air_time`（目标滞空 ≈0.2 s，当前 0.067–0.085）**（**2026-10-10 已落地**：新增 `feet_air_time_deficit`「只罚不奖」版，阈值 **0.20 s**、每步系数 **−3.0**、接线在 `keep_only_amp_task_rewards`；**并更正**：0.067–0.085 是**抬脚高度 m**、不是滞空时间；实测参考逐足腾空是 **0.19–0.28 s（中位 0.21 s）**，策略按周期 0.198 s 估 ≈0.09 s。见 [滞空缺口记录](docs/amp/amp_feet_air_time_2026-10-10.md)）、P3 兜底试 `lerp 0.3→0.5`、P4 恒定抬头 +2.1°/+2.9° 用 `flat_orientation_l2` 单变量消融；压指令范围已降级。完整证据、表格与判据见 [AMP 步态调整方案](docs/amp/amp_gait_adjust_plan_2026-09-18.md)。三次训练完整对照与结论见 [AMP 实验记录](docs/amp/amp_experiments_2026-09-17.md)，代价/步态详情见 [策略运行期排查](docs/deployment/sim2sim_policy_runtime_2026-09-17.md)；2026-11-22 完成[理论与当前源码对照](docs/amp/amp_vs_ppo_theory_2026-11-22.md)，本轮未重训，步态差距保持待验证，需同条件对照训练和独立回放。 |
| AMP-08 | P0 | **已修（测试锁定）；影响此前所有 AMP/PPO checkpoint 的续训** | `amp_on_policy_runner.py` 与 `ppo_on_policy_runner.py` 的 `learn()` **循环内从不更新** `self.current_learning_iteration`（只在循环结束后 `+= num_learning_iterations`），而 `save()` 写的是它 —— 于是**循环内保存的 checkpoint 里 `iter` 恒为该次运行的起点（通常 0）**，只有最后一次保存是正确的。实测证据：运行 `2026-09-17_17-19-04` 的 **20 个 checkpoint 全部 `iter=0`**。`load()` 又用 `loaded_dict['iter']` 作续训起点 ⇒ **`--resume` 会从第 0 轮重跑**，且第一次保存（`it=0`）覆盖 `model_0.pt`、此后每 `save_interval` 覆盖同名文件，等于把已训轮次全部作废。HIM runner（`him_on_policy_runner.py:199`）本来就有这一行，故不受影响 | **已修**：两个 runner 的循环首行加 `self.current_learning_iteration = it`；新增 `test_runners_keep_checkpoint_iter_in_sync` 断言三个 runner 都有该行且 `save()` 写 `'iter'`（8 项测试通过）。**未验证**：修复后尚未产生新的 checkpoint（需一次新训练或续训才能确认 `iter` 写入正确） |
| AMP-09 | P1 | **两套任务已分开，待训练验证** | 2026-09-22 用户把 AMP 入口收缩为两套：`flat-amp-height` 继续使用可部署的 45 维 actor；`flat-amp-fanziqi` 严格采用参考的 42 维 actor（同时移除基座线速度和角速度），关闭非参考持续外力，并恢复参考 runner 的 std 下限、500000 轮和 50 轮保存配置。旧 45 维 checkpoint 与新 42 维任务不兼容 | 在训练机分别做两套 256 环境／100 轮冒烟；Fanziqi 新训完成后必须单独导出并建立自己的部署观测契约，不能覆盖当前 45 维部署策略。详见 [AMP 配置收缩记录](docs/maintenance/project_history_2026-09_to_2026-10.md#amp) |
| MODEL-02 | P0 | 已完成并验证（①–⑤） | 用户决定把模型统一到 `imgo2_description`（D1 命名 FL/FR/RL/RR；D2 Gazebo/IMU/transmission 留在 description 作可选模块；D3 生成物入库；D4 删冗余副本、git 兜底；D5 Gazebo 插件暂不处理）。**① 命名/网格**：`meshes/` 改 FL 命名（指纹 `8dc5b5995a11`，10 文件）。**② 模块化**：`xacro/core.xacro`（物理内核）＋ `{transmission,gazebo,imu}.xacro` ＋ `robot.xacro` 组装入口（开关默认 false）；生成物 `urdf/imgo2.urdf`（纯）与 `urdf/imgo2.gazebo.urdf`。**③ MJCF**：`mjcf/{imgo2.xml,scene.xml}`（训练物理 + 参考求解器/接触块 + `framelinvel`）。**④ 消费者切换**：RL 的 `assets/imgo2.py` 经 `_REPO_ROOT` 指向 `imgo2_description/urdf/imgo2.urdf`；`check_asset_paths.py` 改为按声明解析多个 root 变量；`audit_amp_dataset.py`/`tests`/`inertia_urdf.py` 同步；deploy 新增编译期 `IMGO2_MODEL_DIR`、`rl_sim_mujoco.cpp` 读 `imgo2_description/mjcf/<scene>.xml`；`build.sh` 与两个 Gazebo launch 改指 description；`CMakeLists` 安装列表去掉 `robot_description`。**⑤ 清理**：删除 `imgo2_model/`、`imgo2_rl/source/imgo2_rl/data/`、`imgo2_deploy/robot_description/`（共 46 文件、约 57 MB 工作树）；AGENTS.md 模型章节改写为单一源规则 | **验证**：`check_model_sync.py` 全 PASS（2 份已登记 URDF、网格指纹、mesh 引用存在性、FK 恒等 RMSE 0.00214 m / 交换 0.22470 m）；`check_amp_joint_order.py` 三项全 PASS 且数值与统一前一致（0.00107/0.00214 m、0.22491、0.0191/0.1339 rad、r 0.9546/0.2317）；`check_asset_paths.py` PASS（URDF 17 mesh 引用全在、21 份动作）；单元测试 5 通过 + 1 跳过（跳过的是需要 torch 的 AMP-03 回归）；`bash build.sh -mj` 删目录后仍构建成功，二进制内 `IMGO2_MODEL_DIR` 指向 `<repo>/imgo2_description`；`git ls-files -i -c --exclude-standard` 为空。**未运行**：GUI、Isaac Lab 训练/回放、ROS/Gazebo、真机；Gazebo 插件 `liblegged_hw_sim.so` 仍缺（DEPLOY-05 未解）。详见 [sim2sim 记录](docs/maintenance/project_history_2026-09_to_2026-10.md#amp) 第 6.6 节 |
| MODEL-03 | P0 | 已修并在无头回放验证，GUI 待确认 | MuJoCo 的 `framequat`/`framelinvel` 原先写成 `objtype="body" objname="base"`，返回值会再乘上该 body 的**惯量主轴旋转** `iquat`（base 的 `fullinertia` 主轴非 `ixx<iyy<izz` 排列，实际偏移 ≈180° 绕 (1,0,1)/√2），于是部署侧姿态观测整体偏转 90°：`GetUp` 结束后机器人明明直立（`xquat=(1,0,0,0)`），`gravity_vec` 却报 `(-1.000, 0.004, -0.025)` 而非 `(0,0,-1)`，策略把"站直"判成"已翻倒"——PPO 接管瞬间就输出饱和动作并塌成深蹲（`z_final=0.154`），himloco 侧倾，只有 AMP 恰好鲁棒。判据：`QuatRotateInverse(·,(0,0,-1))` 的第三分量恒为 `-1-2q_z² ≤ -1`，不可能出现 `-0.025`，故读到的三元组并非 `(g_x,g_y,g_z)`；再对照 `xquat` 与 `sensordata[36..39]` 的时间线确认是固定偏移。参考项目用的是 site（`objtype="site" objname="base_site"`）。修法：三个传感器改挂 base 内原有的 `imu` site（site 无 `quat`、`pos` 默认原点，故 site 系 = link 系），**声明顺序不变**，`sensordata` 偏移与 `GetState` 的 `[3n..3n+3]`/`[3n+4..3n+6]` 均无需改动；同时注释掉文件末尾那条陈旧 `<keyframe>`（`base z=0.35`，C++ 不应用但 GUI 的 Key 下拉框会应用） | 已通过：修后三个键的 `gravity_vec` 均为 `(0,-0,-1)`；PPO `vx=0/0.5/1.0` 站姿 0.321、实测 0.509/0.975 m/s（FL_thigh 极差 1.1–1.4 rad，是真步态）、AMP 站姿 0.3015；`check_model_sync.py`／`check_asset_paths.py`／`check_amp_joint_order.py` 仍全 PASS，单元测试 5 通过 + 1 跳过（跳过的是需要 torch 的 AMP-03 回归）。细节见[策略运行期排查](docs/deployment/sim2sim_policy_runtime_2026-09-17.md)。**待办**：在 GUI 里人工确认一次按键 1/2/3 与 1→2→3 切换 |
| DEPLOY-08 | P1 | 已归因：harness 假象，真实代码无此问题；GUI 仍待确认 | 2026-09-17 加 PPO（键 1）后，用**多键连续切换**的临时 harness（0→1→2→3）在进入 himloco 后崩溃：`mat1 and mat2 shapes cannot be multiplied (1x45 and 270x128)`，即把 45 维（单帧）输入喂给了 himloco 的 270 维（6 帧历史）网络。**2026-09-17 晚查明是我那版 harness 的错**：它的 `Forward()` 被简化成 `model->forward({ComputeObservation()})`，漏掉了真实 `RL_Sim::Forward()` 里的历史分支（`history_obs_buf.insert` → `get_obs_vec(observations_history)`）。把 `Forward()` 逐行照搬真实实现后，单键与 `0→1→2→3` 连续切换**都不再崩溃**；同一轮还确认仓库里 `RL_Sim::Forward()` 与参考项目逐行相同，故真实部署代码没有这个缺陷 | 用**真实 GUI 依次按 1→2→3** 再确认一次（预期不崩；会看到 himloco 把机器人掀翻，那是 DEPLOY-01/DEPLOY-06 的占位策略问题，不是崩溃）。若仍崩，再查 `RL::InitRL` 在「无历史配置 ↔ 有历史配置」切换时 `history_obs_buf` 与 `params["observations_history"]` 的时序 |
| HIM-TRAIN-01 | P0 | 已复现，待修复 | HIM estimator下一帧target字段错位，critic关节/速度缩放与policy不一；历史初始化、done/reset缓存不完整。CPU复现通过，不代表修复通过。见[训练说明](docs/training/himloco_training_guide_2026-11-05.md)与其中接口复审链接 | 修target与历史边界，补同状态契约比较、Isaac构造/短训及正式导出验证；旧策略归因需原checkpoint及保存配置 |
| DEPLOY-01 | P0 | 待对齐 | Go2 占位策略与 Imgo2 训练配置存在默认姿态、PD、限幅、指令缩放等差异（指 himloco 占位；AMP 已于 2026-09-17 对齐，见 §5.3；PPO 也已接入，见 §5.4） | 替换为来源明确的 Imgo2 策略，完成训练端与部署端同输入输出比较 |
| DEPLOY-02 | P0 | 场景已对齐并验证站起/行走，待 GUI 验证 | `imgo2_description/mjcf/{imgo2.xml,scene.xml}`（经编译期 `IMGO2_MODEL_DIR` 读取）：关节轴/限位与训练侧逐项一致、MuJoCo C API 可加载、`build.sh -mj` 通过；2026-09-17 按参考项目补齐初始高度（`base pos 0 0 0.5`）与求解器/接触块（`cone=elliptic impratio=100`、关节 `damping=1 armature=0.1`、碰撞 `condim=3 solref="0.005 1"`+friction），不再弹飞；同日又修 MODEL-03（姿态/线速度传感器由 `body` 改挂 `imu` site）并注释掉陈旧 keyframe，修后 PPO 站姿 0.321、`vx=0.5/1.0` 实测 0.509/0.975 m/s，AMP 站 0.3015 | 在有可用显示的机器上跑 `rl_sim_mujoco imgo2 scene` 并保存窗口记录。参考项目那份 MJCF 现在网格名已能对上（FL 改名的副产物，实测可加载），但它的 base 质量仍是旧值 6.53394、用 mesh 碰撞体且没写 `timestep`（=2 ms），同一个参考策略在它上面反而站不起来（`z_final=0.0771`），**不要改用它** |
| DEPLOY-03 | P0 | 缺依赖，待适配 | SDK2 目录为空，真机目标被跳过 | 依赖到位、目标生成、通信接口验证通过 |
| DEPLOY-04 | P1 | 已解决 | 原 `library/thirdparty/joystick/` 为空且未跟踪，`CMakeLists.txt` 在 `USE_MUJOCO` 下要求 `joystick.cc`，`rl_sim_mujoco.hpp` 还 `#include "joystick.hh"`；原无脚本会下载它。2026-09-17 已补入 `joystick.cc`/`joystick.hh`（与参考项目同源） | 已通过：2026-09-17 `bash build.sh -mj` 配置与编译均成功 |
| AMP-07 | P0 | 已定位差异（待改配置重训验证；**2026-09-18 更正了一处单位误读**） | AMP 策略「站着不动」的对照排查（见 [AMP 对照报告](docs/maintenance/project_history_2026-09_to_2026-10.md#amp)，三方逐项对照见该文第 11 节）。**① `clamp(min=0)` 不是差异**：两份参考（`~/RL/isaac/AMP/AMP_for_hardware-main`、`amp_go2-main`）的风格奖励公式与我们逐字相同、都有地板，`d` 为裸 logit。**② 判别器 loss 也相同**（MSE±1、`0.5(expert+policy)`、grad_pen 只对 expert λ=10、共用 Adam lr 1e-3 + trunk/head 不同 weight_decay、`min_std` clamp）。**③ 原「任务奖励量级差 50 倍」作废**：两份参考的 `_prepare_reward_function()` 都会做 `reward_scales[key] *= self.dt`（a1 `legged_robot.py:705-710`、go2 `legged_robot_amp.py:691-696`），Isaac Lab 也乘 dt，所以 a1 配置里的 50/16.7 是**原始权重**、每步真值是 **1.5 / 0.5**，我们是 **1.0 / 0.3**——同量级（差 1.5 倍）。**唯一量级离群项是 `base_height_l2`：我们每步 −10，a1 没有这一项，go2 是 −0.02**；再加上终止只认 `base` 触地（a1 还把 thigh/calf 纳入终止），趴着滑在我们的可行域里没有任何惩罚。速度误差 1.65 m/s 时 `exp(-(1.65/0.5)^2)≈2e-5` 梯度≈0，于是只剩高度项的梯度把策略按在「精确站 0.30 m」的局部最优 → 风格分近乎常数（风格≈0.467/步）→ PPO 优势≈0 → 1000 轮后全指标变平。**⑤ 探索与参考逐字相同**（`init_noise_std 1.0`、`min_normalized_std [0.05,0.02,0.05]*4`、`entropy_coef 0.01`），AMP 观测组成也相同（均不含 commands、43 维）。另两条已离线排除/新增：**专家数据速度是体系、不存在坐标系域差**（差分+逆旋转复核）；**录制数据实际只到 0.83 m/s、0.75 rad/s，而指令给到 1.5 / 1.57**，风格参考与任务指令互相打架 | 修正后优先级（第 11.3 节）：**0** 先确认 Run1/Run2 的单位（`50/17/−1` 是原始权重还是每步值，两者结论相反）；**1** 低姿态终止（thigh/shank 纳入 `illegal_contact`，a1 做法）；**2** 高度项 −10→0，且必须与第 1 条配对做；**3** `tracking_sigma` 0.25→1.0（σ 0.5→1.0，对参考的主动偏离）；**4** 指令范围压到数据覆盖内（x ±0.9、yaw ±0.8）（**2026-10-10 已实测复核**：实际覆盖比这更窄 —— vx 前向 ≤+0.84、yaw ±0.765、vy ±0.51，见 [AMP 速度包线](docs/amp/amp_speed_coverage_2026-10-10.md) §2；且该文结论是**压范围不解决节律问题**，只用于消除「风格 vs 任务」冲突）；**5** `max_iterations` 提到 ≥30000（参考 checkpoint 是 36450 轮，我们 9496 轮基本跑满 10000 上限）；**6** 可选去掉/降权 `imgo2_stance.txt`；**7** 独立项：std 上限、奖励尺度归一化、判别器饱和、AMP-05 按 go2 的 45 维 actor 新训。改后重训并按「高度/速度误差/关节幅值是否随训练变化」复查（需训练环境，本机无 GPU） |
| PPO-01 | P1 | 已按四项指标选定（`base_move/policy_flat.pt`） | PPO 权重此前用过 `imgo2_flat/2026-06-21_23-24-09` 与 `imgo2_rough/2026-06-21_16-37-38` 两份训练导出，用户反馈「腿部抖动很厉害」并给出评估标准：**位姿 / 速度跟随 / 抖动 / 周期性**。新增 `imgo2_deploy/ros2_ws/scripts/eval_gazebo_policy.py`（自注入 /joy，从 /odom + /joint_states 统计四项）并对 5 个候选实测：参考项目 `base_move/policy_flat.pt`（6/30）vx=0.5 时 z 0.288、roll 2.5°、yaw 漂 −0.5°/s、抖动(大腿) 45 rad/s²、周期强度 0.95、FL-FR 175°、速度 0.428 m/s；`policy.pt`（6/30）速度更准 0.522 但抖动 79、roll 5.0°、强度 0.74；我们的 `rough/16-37-38` 抖动 241/507、roll 13.4°、强度 0.29、yaw 漂 −2.7°/s（vx=0 时还漂 +4.8°/s）；`policy1.pt` 超调 0.662；`amp/policy.pt` 不走（0.057 m/s、z 0.182）。⇒ 采用 `policy_flat.pt`（配置 25/0.5、默认 0/0.87/-1.82、clip ±3） | 已装并复测；后续若要速度精度可换 `policy.pt`（同配置）；新训练版本可用同一脚本按四项指标对比。详见 Gazebo 记录第 9 节 |
| JOINT-01 | P0 | ROS2 路径已修并验证；真机路径待处理 | `joint_mapping` 的语义是「策略第 i 个关节 ↔ 该路径数组第 joint_mapping[i] 号」，但三条路径的数组不同：MuJoCo 是 MJCF 声明顺序（FL,FR,RL,RR=策略顺序，恒等正确）；ROS2 是消息槽位，顺序 = 控制器 `joints` 参数 = `base.yaml` 的 `joint_names`（原为 Unitree SDK 的 FR,FL,RR,RL）；真机是 SDK 电机数组（固定 SDK 顺序）。于是恒等映射在 MuJoCo 对、在 ROS2 把策略 FL 接到物理 FR：Gazebo 里按 1 进 PPO 会翻成四脚朝天（实测 `/odom` z=0.073 m、roll=180°、dx=0），给速度指令后更明显。**修法**：① `base.yaml` 的 `joint_names`/`joint_controller_names` 改为模型顺序；② `rl_sim.cpp` 的 ROS2 分支新增 `OrderJointsByModelOrder()`，按 URDF 声明顺序重排传给控制器的名单（读不到则回退），MuJoCo 路径代码与语义未动。**验证**：Gazebo `vx=0` z 0.328–0.332 m、roll ±2.9°、dx≈0；`vx=0.5` 向前 8.20 m/12 s（0.56 m/s）、z 0.26–0.27、大腿摆幅 0.99–1.08 rad（修前四脚朝天）；IMU 侧 `/imu` 角速度与 `/odom` yaw 速率同号同量级，排除 IMU 约定问题 | 真机路径（`rl_real_imgo2.cpp` 索引 SDK 电机数组）要另给映射或改代码；偏航/走偏已确认是策略性质（MuJoCo 同策略 `vx=0.5` 也偏航 −62°/12 s），待换 checkpoint 或用 `axes[3]` 做航向闭环验证。详见记录第 8 节 |
| DEPLOY-05 | P0 | 已解决（无头跑通到策略闭环），无头 PPO 闭环与步频已复测；GUI 待 GLX 环境修复 | 原状：部署份 URDF 只有 ROS 1 式 `<transmission>`，`gazebo.xacro` 挂的是仓库内没有的 `liblegged_hw_sim.so`（`gazebo_ros_control`），ROS 2 下没有 `gazebo_ros_control`，于是 `rl_sim.cpp` 的 controller spawner 找不到 controller_manager。**2026-09-17 照参考项目改为 ROS 2 `gazebo_ros2_control`**：`gazebo.xacro` 加 IMU 传感器 + `<ros2_control>`（12 关节 effort 命令 + position/velocity/effort 状态）+ `libgazebo_ros2_control.so`；新增 `imgo2_description/config/robot_control_ros2.yaml`（`joint_state_broadcaster` + `robot_joint_controller/RobotJointControllerGroup`，后者是仓库内插件、由 `rl_sim` 自己 spawn）；`imgo2_description` 做成标准 ROS 2 包（ament + `package.ros1.xml`/`package.ros2.xml` + 软链进工作区 `src/`）；Gazebo 版网格改 `package://`（`robot.xacro` 的 `mesh_prefix`）；`build.sh` 的包扫描改 `find -L`；`check_model_sync.py` 认两种网格写法。**两个实测坑**：`<parameters>` 必须是真实文件路径（`package://`/相对路径都会让插件 Load 抛异常）；URDF 作为 `--param robot_description:=` 传给 controller_manager 时必须压成单行（否则 rcl 报 `Couldn't parse parameter override rule`，控制器起不来）；另外不能用 `gazebo_ros` 自带 launch（本机它起的 gzserver 未加载 factory 插件），改为直接起 `gzserver -s ...`。**验证**（无头、单次调用内）：spawn 成功且网格 0 报错；节点 `/gazebo_ros2_control`、`/imu_plugin` 出现；`ros2 control list_controllers` → `joint_state_broadcaster active`；`/joint_states` 与 `/imu` 发布；`rl_sim` 启动后用 `/joy` 注入 `A`→`RB+DPadUp`，打印 `RL Controller [ppo]`（进入键 1 的 PPO 闭环） | ① 在有可用显示的机器上确认 `gzclient` 画面（本机 `DISPLAY=:1` 建 GL 上下文失败）；步态**已用位姿客观验证**：vx=0.5 时 Gazebo 0.502 m/s、z 0.276 m、大腿摆幅 0.89–1.20 rad，与 MuJoCo（0.51 m/s、0.26 m、0.88 rad）一致，见记录第 7 节；② 长时间跑 himloco/AMP；③ 真机链路仍待验证。详见 [Gazebo 链路打通记录](docs/deployment/gazebo_ros2_bringup_2026-09-17.md) |
| DEPLOY-06 | P1 | AMP 路径已解决，himloco 路径仍在 | `rl_sim_mujoco.cpp` 用 `joint_mapping` 直接索引 MJCF，因此配置的映射必须与场景顺序一致。2026-09-17 采用自建场景（关节声明顺序 = 策略顺序 `FL,FR,RL,RR`）+ `base.yaml`/`amp/config.yaml` 恒等映射，AMP 路径自洽 | AMP 已在 16 s 回放中确认基座高度稳定、无 NaN；`himloco/config.yaml` 仍带硬件置换映射 `[3,4,5,0,1,2,9,10,11,6,7,8]`，在 MuJoCo 上会索引错腿，需单独处理；`imgo2_description/mjcf/` 那份现成 MJCF 未再复用 |
| DEPLOY-07 | P0 | 已修，待编译验证 | ROS2 两个控制器把 `std::clamp(…)` 当语句调用、丢弃返回值，限位实际失效；单关节版还把参数名写成 `"robot_description_"`，URDF 从未解析成功。已改为赋值形式并加空指针/越界保护、补 `<algorithm>`、修正参数名 | 在装有 ROS 2 的 Linux 上编译并跑通，确认限位生效且不再有空指针风险 |
| EXPORT-01 | P1 | 待验证 | 导出配置与 C++ 配置格式不同；比较脚本假设六帧历史 | 明确转换规则，记录实际网络维度、历史规则与误差指标 |
| EXPORT-02 | P2 | 启动器导入已修，CPU 导出通过 | 11/22 已用清空 `PYTHONPATH` 的终端正式导出 PPO 5000。见[导出命令](docs/xiaoji_cmd/策略导出与ROS2仿真.md#2026-10-27补充ppo-model_8400仅导出)。 | 在可用 GPU 上复测 CUDA 导出 |
| EXPORT-02 | P1 | **已修（PPO 端到端实跑验证；amp/himloco 待实跑）** | 2026-10-28 导出报 `FileNotFoundError: USD file not found at path at: '.../UIElements/arrow_x.usd'`：`commands.base_velocity.debug_vis=true` 让 `CommandManager.__init__` 在**建场景时**就创建速度指令箭头，其 marker 原型是远端 S3 资产，`omni.client.stat` 不可用时建场景中断（早于 `runner.load()`，所以三次失败只留下 `outputs/`、没有 `exported/`）。新增 [play_utils.py](imgo2_rl/scripts/rl_lab/play_utils.py) 的 `disable_command_debug_visualizers()`，ppo（`--headless`/`--export_only`）与 amp/himloco（`--headless`）在建场景前关闭指令 debug_vis；GUI 回放不变。见 [导出记录](docs/deployment/ppo_5499_export_2026-10-28.md) | 已做：helper 6/6 离线单测、`py_compile`、PPO 原命令实跑退出码 0（日志无 `USD file not found`、无 `/Visuals/Command` marker 警告，JIT 与 Actor `max diff 0`，`policy.onnx` 字节一致）。待做：amp/himloco 的同一行改动实跑；部署 LibTorch 加载探针与 5499 的 MuJoCo/Gazebo 步态评测 |
| DATA-01 | P1 | 当前RL新目录已核对，外层副本/服务器同步待确认 | 2026-10-13重采21段，RL新目录与外层imgo2_dataset旧副本19同名全不同、1.0/1.2文件不同；默认读RL目录，新1.2文件未跟踪。见 [核查](docs/amp/amp_new_collection_preflight_2026-10-13.md) | 明确外层旧副本用途与同步方式，服务器包含新增文件并清除旧motion覆盖变量；不以相同文件夹名认定相同数据 |
| EXP-01 | P1 | 待补充 | 已记录 PPO 验证与 AMP 贴地现象的用户反馈，尚缺对应日志、命令和模型路径 | 按第 8 节补充真实实验与产物路径 |
| MODEL-05 | P1 | real模型对比/短回放已验证，全仓登记与指纹仍待修复 | 11/22[新关节几何同步](docs/deployment/model_sync_2026-11-22.md)通过本地资产和 MuJoCo 结构校验，服务器短训练、Gazebo 闭环和全仓登记待确认。14网格/指纹66c496556453、两份real URDF未登记使全仓守门失败；2026-10-13独立real对比17本体link一致、惯量有效，并完成两仿真短回放。见 [本轮排查](docs/maintenance/project_history_2026-09_to_2026-10.md#amp)；历史见 [rebase记录](docs/maintenance/rebase_main_2026-10-07.md) | 确认新网格/模型族设计基线、生成来源及注册，复跑全仓守门；不能把本体一致性当成实机惯量辨识 |
| MODEL-06 | P1 | real新惯量已生成并离线验证，动态待确认 | 用户将左前/左后thigh ixz改负；新real两份解析/惯量一致性与21段FK通过，见 [生成](docs/maintenance/project_history_2026-09_to_2026-10.md#model-and-workspace)。此前短回放使用旧惯量 | 新惯量下Isaac/Gazebo回放、服务器USD缓存；确认旧imgo2生成物/MJCF版本关系，四份real/real_wrong未登记仍由MODEL-05处理 |
| MODEL-04 | P1 | 已修复并验证 | `sorted(Path)` 在 Windows 与 Linux 对大小写排序不同导致同一套网格指纹不同；改为按文件名字符串排序，恢复原预期值，未改网格 | 大小写混合排序与内容篡改回归通过；全模型检查通过，指纹 `8dc5b5995a11` |
| TOW-CLUS-01 | P2 | **已修，待复现/定机理** | 2026-10-12 用 `v3_800_policy_c008` 的 700 个有车样本做聚类检验：无监督 k-means 能把「绳 vs 杆」分开（station 纯度 0.920/ARI 0.703，绳 210 例压成一点、rigid 按质量排开），但 **compliant 与 inextensible 从未被分开**；tow 全量上连接被质量/速度淹没（k=3 ARI 0.001），固定坡度/质量/速度后才恢复（ARI 1.000，n=13）。质量档在站定阶段**只在 rigid 下可见**（绳组 latent 半径 0.020、η² p=0.36/0.67、decoder 质量输出误差中位 5.001 kg = 常数；rigid 半径 3.468、η² 0.737、p=0.002），tow/coast 三类都可见且 rigid 最强；**固定速度后质量可分性更高**（η² 0.358→0.565、kNN 0.940→0.975），混淆只在低端 5 vs 10 kg。详见 [聚类检验记录](docs/maintenance/project_history_2026-09_to_2026-10.md#towing) | 缺两样：① **第二轮带 latent 的报告**（换 checkpoint 或换 c=0.032 阻尼端）复现「绳/杆边界 + 两类绳不可分 + rigid 质量最强」这三条；② 若要定机理（力语义 vs 出生松弛几何），需要**出生间隙对齐但传力不同**的对照工况——当前网格把 `SLACK_RATIO=0.8` 与 `d≡L0` 绑在连接类型上，离线无法拆分。单轮结论不得当成「连接可辨识」的通用结论 |
| TOW-CLUS-02 | P2 | **行为层已测出量级；latent 层的可比性缺第二个阻尼取值** | 用户 2026-10-12 追问「看看对轮摩擦的情况呢」。**核实**：`--wheel-damping`/`--ground-friction` 是**整轮标量**且 report 只在 `arguments` 记全局值（case 无摩擦字段）⇒ 单轮内部没有摩擦标签；本机 20 份 report 里 **μ 恒 0.8**（训练随机 0.4–1.2 从未扫）、**阻尼只有 0.008/0.032**、**latent 只在 c=0.008 那轮**。**行为层（同策略 `model_2700`、同 800 网格、同 μ=0.8，逐 cell 配对 700 有车）**：硬证据 **3.6% → 23.0%**（800 格口径 3.1%→20.1%，与 `compare_play_runs.py` 一致），滑移中位 0.26→**0.55 m**、P95 1.57→**3.80 m**、最小间隙中位 0.543→0.372 m、甲板峰值力 P95 0→**255 N**；代价几乎全在绳侧（compliant +31.8pt、inextensible +31.4pt），**rigid 只 +1.1pt（3 格）**；速度 1.5 m/s **+41.7pt**、20 kg **+25.1pt**。⇒ 阻尼影响量级大于端到端上层策略的收益量级（同 c=0.008 下策略只救回 68 格，见 §8.9）。详见 [聚类检验记录 §3](docs/maintenance/project_history_2026-09_to_2026-10.md#towing) | 缺一次**带 `--dump-latent` 的 c=0.032 复跑**（命令已 `--dry-run` 通过、checkpoint 在盘上，需 GPU shell）⇒ 之后可池化两轮 latent 做 η²/kNN/ARI/置换检验，与质量·连接强度对比；**工具还需加多报告池化**（如 `--report LABEL=PATH`）。另：μ 要单独补 `--ground-friction 0.4` 与 `1.2` 两轮；阻尼是整轮标量，dose-response 需多档（0.008/0.016/0.024/0.032）。**当前不得声称「latent 能/不能区分轮摩擦」** |

问题表只放结论与状态，依据、根因和未修项的细节见 [代码复审记录](docs/maintenance/project_history_2026-09_to_2026-10.md#model-and-workspace) 与 [AMP 对齐与参考项目对照](docs/amp/amp_alignment_review.md)。**代码改了不等于修好**：只有经过编译、运行或测试验证的项才改状态，仅改代码未验证的写成「已修，待验证」并保留条目（见 AGENTS.md 的同名约定）。

关于 AMP-02：当前 `actor_critic.py` 直接学习 `std`，同时 `amp_ppo.py` 中存在有条件的最小标准差裁剪。仍需检查是否出现 NaN/Inf、裁剪条件是否生效以及数据/梯度是否异常；本次不将任何一个猜测登记为已确认根因。

历史草稿已收编：`imgo2_rl/todo.md`（AMP 腿顺序与上述报错的原始记录）并入本表 AMP-01／AMP-02；`imgo2_deploy/todo.md`（部署项目搭建需求）的内容已由第 6 节部署流程与目录职责覆盖。两个草稿文件已删除。状态以本表最近核验结果为准。

### AMP 对齐排查与下一轮训练

详细依据见 [AMP 对齐与参考项目对照](docs/amp/amp_alignment_review.md)，离线结果见 [动作数据核对 JSON](docs/amp/data/amp_data_audit.json)。

- `joint_mapping` 已改为恒等映射，仅适用于当前已核对的 `imgo2_motion` 数据。更换数据源后重新检查。
- 恒等映射不只依赖采集参数确认：对实际采集模型 `imgo2_description.urdf` 复算 FK 与训练 URDF 结果相同；准对称支撑的髋外展角符号、关节位置块与速度块的逐通道相关性也支持 `FL,FR,RL,RR`。详见 [AMP 对齐与参考项目对照](docs/amp/amp_alignment_review.md) 第 3.3 节。
- 用户指定的采集脚本为 `C:/Users/qmq/Desktop/RL/AMP/amp_go2-main/datasets/record_legged_control_amp.py`，并确认未传入 `--joint-names` 或 `--foot-frames`。关节和足端按默认 LF/RF/LH/RH 写入，等价于 FL/FR/RL/RR；采集模型的声明顺序 LF/LH/RF/RH 不等于文件顺序。
- AMP 高度项目标 `0.30 m`。任务每步奖励为速度项 `1.0 / 0.3` 加高度误差项 `-10 × (z-0.30)^2`；配置中使用除以 `step_dt` 的权重以抵消 RewardManager 的时间步长乘法。
- 新混合系数为 `amp_task_reward_lerp=0.3`，这是待训练验证的起始配置，不是已证明最优的参数。
- 新日志包括 `AMP/mean_root_height_m`、`AMP/fraction_root_height_below_0_20m`、`Train/mean_task_reward_step` 和 `Train/weighted_task_reward_step`。原 `Train/mean_amp_reward_step` 名称实际记录混合后的总奖励。
- 建议从新运行开始验证；旧 checkpoint 的判别器及归一化统计来自错误输入，直接续训无法作为修复的干净对照。

从 `imgo2_rl/` 执行离线检查与短训练示例（后者本次未执行）：

```bash
python scripts/tools/check_asset_paths.py
python scripts/tools/check_model_sync.py
python scripts/tools/audit_amp_dataset.py --output logs/amp_data_audit.json
python scripts/tools/check_amp_joint_order.py
python -m unittest discover -s tests -p test_amp_alignment.py -v
python scripts/rl_lab/amp/train.py --task=Imgo2-basemove-flat-amp-height --num_envs=256 --max_iterations=100 --seed=42 --headless
```

`check_asset_paths.py` 确认 URDF 与动作数据能按 `Path(__file__)` 推导的路径找到（应为 21 份动作文件），是本机 ENV-01 的自检项。`check_model_sync.py` 按逻辑关节名与逻辑 link 名（与腿序、LF_/LH_ 命名无关）比对四份 URDF 的关节轴/限位、每个 link 的质量/质心/惯量/碰撞几何、base 规范值、三份共用的网格集合，并让每份 URDF 都对录制数据做 FK——五项任一破坏都会报错（已用翻符号与改足端 CoM 两次负向测试确认）。

`check_amp_joint_order.py` 只用标准库，默认读取 `imgo2_description/urdf/imgo2.urdf`（2026-09-17 统一后，该文件与训练 URDF 是同一个机器人的同一份物理；此前的 LF 命名采集副本已删除）。本机可用解释器见第 8.3 节维护记录。它检查模型 FK（含跨腿交换对照）、髋外展左右对称性和关节位置/速度块相关性，三项全部通过才返回 0。**注意**：其中"声明顺序对照"那一项随统一而消失（两份模型现在同序），保留下来的 URDF-free 两项（左右对称、位置-速度相关性）继续承载 AMP-01 的独立证据。

短训练用于检查奖励项、终止原因和数值是否正常，不足以判断最终步态收敛。新增高度项的消融可使用 `env.rewards.base_height_l2.weight=0`，其余条件保持一致。

## 8. 维护规则与记录

### 8.1 如何维护本文

1. 修改项目结构、任务注册、运行命令、训练参数、数据路径或部署接口时，在同一次工作中更新对应章节。
2. 优先依据当前代码和真实运行产物；历史 README、todo 和研究计划仅作为线索。冲突时说明差异，不自动把旧问题标为已解决。
3. 新发现问题使用稳定 ID，记录依据、状态和完成标准。解决问题后保留简短结论及验证证据。
4. 实验只记录实际结果。未运行、失败、受阻和已通过要明确区分；阶段计划不充当实验完成记录。
5. 每次有效更新修改顶部日期，并在下表追加一条简要记录。只有措辞或排版变化时无需重复完整审计。
6. 本文由后续项目工作同步维护，不代表存在后台自动监控或定时更新任务。

### 8.2 实验记录模板

```text
实验 ID / 日期：
目标与关联问题 ID：
任务 ID / 算法 / 随机种子：
工作目录与完整命令：
代码版本或本地修改摘要：
Python / Isaac Lab / Isaac Sim / PyTorch / CUDA / GPU：
模型路径与哈希 / 数据版本：
环境数量 / 训练迭代 / 关键配置差异：
checkpoint / 日志 / 视频 / 导出目录：
测试条件、指标与数值：
状态（通过 / 失败 / 未完成）与结论：
下一步：
```

### 8.3 维护记录

| 日期 | 工作与验证 | 待确认 |
|---|---|---|
| 2026-11-24 | 将实机目录恢复到 `81fa54a`。已核对工作区及索引无实机目录差异，恢复前的 24 个文件改动已备份到本机，补丁可应用。 | 本轮未运行实机或上传。 |
| 2026-11-24 | 撤回 Rough PPO 的 7.5 Hz 横滚力矩注入，原生摩擦选项保留。源码与 `81fa54a` 的 Rough 配置逐字节一致，Python 语法和引用检查通过。 | 未启动新训练。服务器副本和已有实机止抖结论待独立验证。 |
| 2026-11-24 | 以 `81fa54a` 为基线核对暂存内容，为 `docs/xiaoji_cmd/中期/` 添加根目录忽略规则并撤销目录内 29 个新增文件的暂存。已查验该目录在 Git 索引中无条目，本地资料保留。 | 其余改动待用户决定提交范围，`real/` 仍有 24 个文件暂存。本轮未提交或推送。 |
| 2026-11-24 | 整理[机器人控制强化学习论文与十周路线](docs/research/robot_control_rl_reading_route_2026-11-24.md)，已澄清 PPO、状态表示和动作先验的关系，核对经典及 2026 年代表论文来源。仅文献与文档检查，无控制代码修复、训练或硬件运行。 | RESEARCH-LEARN-01 学习时间、预算和复现方向；Git 元数据只读，pull 未完成 |
| 2026-11-24 | 将 [改进前实物与双仿真器横滚记录](docs/xiaoji_cmd/中期/实物实验/实物实验说明.md#改进前的实物横滚与仿真差距)归入中期资料。核对旧 7000 的训练归档、图像与两个后退窗，补齐模型标定、重训、实物 PD 对照的证据范围。 | 仿真模型改进的实物收益待用同 PD 的配对策略重复验证；Git 元数据只读，远端未同步 |
| 2026-11-24 | 整理 [PPO 训练原理](docs/xiaoji_cmd/中期/仿真/PPO训练原理.md)及 GAE 手算，补充 Actor 45 维观测的字段和关节顺序，重绘 PPO、HIM-Loco、AMP 的简版框图。更新[答辩第 3 页与算法答疑](docs/xiaoji_cmd/中期/中期答辩分页.md)，核对训练配置、算例、图片版面、链接和分页计时。 | 封面与实物素材待选，既有实验和 HIM-TRAIN-01 验收状态沿用。Git 元数据只读，远端未同步 |
| 2026-11-23 | 从仿真及实物材料整理[中期答辩分页](docs/xiaoji_cmd/中期/中期答辩分页.md)，12 页、12 分 25 秒，保留一页算法简述。核对素材和台阶示例视频。 | 正式封面及同期实物素材待选定，DEFENSE-EVIDENCE-01 |
| 2026-11-23 | 补充 [AMP 答辩用 MPC 参考轨迹说明](docs/xiaoji_cmd/中期/仿真/仿真图表说明.md#amp)，按用户确认标注为 LeggedControl MPC 的仿真闭环采集。核对 ROS 采集入口、默认目录及 21 段数据与归档 SHA-256 一致。 | 原始 ROS 时间戳同步、多工况覆盖及 AMP 相对 PPO 的增益沿用采集复核和 AMP-06；无新训练，Git 元数据只读 |
| 2026-11-23 | 将 16 段正式成品移入[仿真素材总览](docs/xiaoji_cmd/中期/仿真/仿真素材总览.html)，合并离线图片／视频预览，整理两份说明，清理旧入口、重复 ZIP、未入选成品。比对归档哈希，16 段全解码，复制目录验证相对链接。 | 原始采集证据仍在日志目录，视频为本机素材。既有仿真及实物验收状态沿用问题表，Git 元数据只读 |
| 2026-11-23 | 将中期答辩仿真资料归档到[仿真图表说明](docs/xiaoji_cmd/中期/仿真/仿真图表说明.md)。六张 PNG 保留图题与必要标注，合并原说明并更新入口。离线检查六图尺寸／dpi、版面与引用。 | 既有仿真实验与实物验收状态沿用现有问题表。Git 元数据只读，远端未同步 |
| 2026-11-23 | 完成[两批 8000 轮策略同工况对比](docs/xiaoji_cmd/中期/仿真/仿真图表说明.md#新旧-8000-轮策略对比)。两批训练奖励配置一致，复核训练回报并完成 8 次 MuJoCo 同条件测试。新策略前后向跟速误差和 6–8 Hz 横滚下降，后退全频带横滚标准差上升。 | 后退全频横滚、跨地形重复率及同 PD 实物新旧策略对照待确认。Git 元数据只读，远端未同步 |
| 2026-11-23 | 将 [8000 轮策略答辩结果](docs/xiaoji_cmd/中期/仿真/仿真图表说明.md)精简为两张图。用原指标复核同批六个候选的跟速／6–8 Hz 横滚排序，用两引擎轨迹复核 8 cm 上下阶。两张 4K 图及链接已检查。 | 沿用 Gazebo 粗糙工况、平地横向偏移、楼梯重复性及实物验收。Git 元数据只读，远端未同步 |
| 2026-11-23 | 收集 [11/23 Rough PPO 8000 轮闭环数据与曲线](docs/research/defense_ppo8000_data_2026-11-23.md)。核对 checkpoint 与 JIT 输出，两引擎各五个工况，保留 Gazebo 粗糙工况早停数据。 | Gazebo 粗糙工况复测、平地横向偏移、多速度楼梯与实物验收。Git 元数据只读，远端未同步 |
| 2026-11-23 | 补齐[楼梯成品的 PPO 训练批次](docs/research/defense_stairs_2026-11-23.md#权重来源)，核对四份 JIT SHA-256、成品清单与 9299 导出记录。 | 无新增待修项，既有实物及跨工况验证状态不变 |
| 2026-11-23 | 完成三条本体运动训练路线的[源码结构图与讲解](docs/xiaoji_cmd/中期/仿真/仿真图表说明.md#算法结构图)，核对三图版面、PNG 尺寸／dpi、SVG 和文档链接。两批总览资源及 ZIP 完整性通过，楼梯八段视频 SHA-256 一致。 | HIM-TRAIN-01 与既有实物验收状态不变。是否清理原始素材待决定，远端未同步（Git 元数据只读） |
| 2026-11-23 | 补采 5 cm、8 cm 楼梯各四段，Gazebo／MuJoCo 各两段。八段均完成上下行并通过媒体解码、抽帧和 ZIP 校验。见[楼梯补采](docs/research/defense_stairs_2026-11-23.md)。 | 8 cm 部分候选有停顿，复杂工况通过率、实机及 18 cm 专项待验收 |
| 2026-11-22 | 完成本体运动答辩素材：两引擎八段 1080p 视频、13 张 4K 图片、五张 600 dpi 曲线及矢量版。原生仿真与渲染、视频全解码、尺寸／dpi／ZIP 校验通过，登记生产文件哈希未变。见[采集记录](docs/research/defense_media_2026-11-22.md)。 | 8 cm 对照停滞，18 cm 未测；多次地形可靠性、方向漂移与实物稳定性沿用 §7，最终 PPT 素材绑定待完成 |
| 2026-11-22 | 按用户确认将研究总结聚焦 Imgo2 本体运动，移除无关分支，精简为训练／实物／协同反馈及 12 页答辩提纲，并修正总览的研究定位。见[修订总结](docs/research/midterm_research_summary_2026-11-22.md)。链接与页码离线核对，无新运行。 | DEFENSE-EVIDENCE-01 素材与个人工作归属；既有实验验收状态不变 |
| 2026-11-22 | 已澄清 AMP 保留 PPO 更新、判别器奖励和两套配方差异，补充[理论与源码说明](docs/amp/amp_vs_ppo_theory_2026-11-22.md)。`/usr/bin/python3` 的 AMP 配方离线测试 6/6 通过；无算法代码修复或训练。 | AMP-06 未关闭；同条件性能与实机效果、实际 PPO 环境的外部 RSL-RL 版本待确认。pull 因 Git 元数据只读未完成 |
| 2026-11-22 | 完成仿真／实物双线研究总结和中期答辩 16 页提纲，核对关键数字、相对链接及两张候选图。见[总结与证据索引](docs/research/midterm_research_summary_2026-11-22.md)。本轮仅改文档，pull 因 Git 元数据只读失败。 | DEFENSE-EVIDENCE-01 素材与个人工作归属；既有研究验收状态不变 |
| 2026-11-22 | 已澄清目标抖动/IMU 敏感性不足以确定首因，复核近期记录、归档频响与部署源码，补齐本机[区分方案](docs/deployment/sim2real_diagnostic_plan_2026-11-22.md)；链接、数值与延迟算术离线核对。本轮无控制修复或新仿真/硬件运行；pull 被 Git 元数据只读阻止。 | DEPLOY-13 未修；独立姿态、实际时序与承重 6–8 Hz 响应待测，远端未同步 |
| 2026-11-23 | 补画四腿原生摩擦回放的四窗实机黑线与三组仿真，逐轴对照位置、速度及电机力矩。见本机执行器验收记录。 | 独立实物摩擦测量及跨工况验证 |
| 2026-11-23 | 完成 MuJoCo 横滚外力矩量级扫描，持续 10 N·m 的 7.5Hz 激励使后退机身快抖接近实机。见本机 IMU 扰动 A/B 记录。 | Isaac Lab 短时激励校准及训练对照 |
| 2026-11-23 | 清理小腿候选配置中被覆盖的静摩擦赋值，Isaac Sim 5.1 的 12 轴 PhysX 参数回读通过。实物相关交付集中到[中期实物实验](docs/xiaoji_cmd/中期/实物实验/实物实验说明.md)。 | 小幅及训练闭环效果待验证，服务器未核对 |
| 2026-11-23 | 分别拟合 PhysX 静、动及黏性摩擦，双步长台架与 24 个承重仿真完成。小腿大幅留出误差再降约 61%，更换答辩曲线。见[摩擦再标定](docs/xiaoji_cmd/中期/实物实验/实物实验说明.md#仿真摩擦再标定)。 | ACT-NET-01 未关闭，小幅和步长门槛仍待通过 |
| 2026-11-23 | 完成三包实机数据的策略身份、关节 PD 与同后退指令对照，量化 6–9 Hz 横滚角速度 RMS 降低约 88%，交付四张 16:9 答辩图，含 Isaac Sim 原生摩擦模型与实测台架曲线，讲稿单独保存。见[实机减振总结](docs/xiaoji_cmd/中期/实物实验/实物实验说明.md)。 | DEPLOY-13 部分改善，待重复与拆分 P/D 贡献 |
| 2026-11-24 | 核对 BP2 高速电机包络和磁饱和的原文方法，记录与本项目低速往复辨识的适用范围。见[文献笔记](docs/training/actuator_network_learning_2026-11-05.md#12-bp2-论文中的力矩转速包络)。 | ACT-NET-01，实际力矩包络、电流和母线数据待核对 |
| 2026-11-23 | 核对执行器网络参考脚本和实机数据，确认旧增益可做原型与独立留出验证。见[数据复核](docs/training/actuator_network_learning_2026-11-05.md#11-11-月-23-日参考脚本与可用数据复核)。 | ACT-NET-01 继续保留，逐轴时标、估算力矩和新 PD 待确认 |
| 2026-11-23 | 静态确认独立实机支持手柄 A 从 PPO 经 2 秒轨迹回站，补充停步 A/B 和归零训练验收。见[停步方案](docs/xiaoji_cmd/中期/实物实验/原8000实机振荡复核.md#停步处理与续训验收)。 | DEPLOY-13 未修，待核对小电脑版本、回站实机效果及走动起振 |
| 2026-11-23 | 复核原 8000 实机 4954 次策略输入输出，量化多次零指令后 7–8 Hz 持续振荡，并用本次实机 IMU 做 8 组仿真对照。见[复核](docs/xiaoji_cmd/中期/实物实验/原8000实机振荡复核.md)。 | 起振因果、外部辅助状态与实机止抖待确认 |
| 2026-11-23 | 核对服务器续训到 9100 的配置及权重，完成 128 次闭环和 24 次历史实机 IMU 回放。短力矩响应改善，历史输入鲁棒性退步。见[续训复核](docs/xiaoji_cmd/中期/实物实验/横滚扰动续训复核.md)。 | 本轮 8000 实机日志与起振链路待核对 |
| 2026-11-23 | 补测 6000、8000、9000 的 60 个 MuJoCo 闭环工况。按低速跟踪和 6–8 Hz 横滚指标选择 8000 续训，已核对权重和服务器奖励覆盖。见[续训起点](docs/training/native_friction_ppo_review_2026-11-23.md#续训起点选择)。 | 恢复冒烟、续训前后对照与实机验收 |
| 2026-11-23 | 在服务器内存实例化旧版和当前版并递归比较，两种扰动开关状态下均只有十项有效差异。 核对 33 份归档，11/22、11/23 两次长训均使用低速范围。见[配置全量对照](docs/training/native_friction_ppo_review_2026-11-23.md#旧配置和当前配置的完整比较)。 | 目标奖励与速度／地形组合待确认，短训待验收 |
| 2026-11-23 | 再查服务器确认第一份配置已注释，唯一有效配置可实例化 7.5 Hz 扰动；当前机身高度与滞空权重为 −32／1.5。见[服务器复核](docs/training/native_friction_ppo_review_2026-11-23.md#服务器当前配置复核)。 | 目标奖励 −30／2.0 与新短训归档待核对 |
| 2026-11-23 | 通过 SSH 核对服务器双开关及实际任务配置。找到同名类覆盖，并在 Isaac Sim 中复现和验证内存修法。长训归档确认使用第二份配置的奖励。完成新旧三对 checkpoint 的 120 次 MuJoCo 闭环，末轮正常行走改善，受扰恢复未见一致提升。见本机[归档与策略对照](docs/training/native_friction_ppo_review_2026-11-23.md)。 | PPO-ROLL-TRAIN-01，服务器配置待修、双开关短训及长训待复核，实机未测 |
| 2026-11-23 | 接入 Rough PPO 可选间歇横滚扰动。8 环境 18 轮短训确认 5ms 更新、独立时间表、归零和局部重置。见本机 IMU 扰动 A/B 记录。 | Isaac Lab 幅值标定、长训和实机效果 |
| 2026-11-23 | 核对 Rough PPO 已有角速度和投影重力噪声，行走机身外扰未启用。见本机 IMU 扰动 A/B 记录。 | 物理横滚扰动校准与训练对照 |
| 2026-11-23 | 在共享资产新增四腿 PhysX 原生摩擦训练开关。本机 Rough PPO 的八环境两轮通过，PhysX 读回核对全部关节。 | 服务器短训、长期训练和实机毫秒级振荡 |
| 2026-11-23 | 验证四腿 PhysX 原生摩擦。四窗角度误差均下降，额外小腿黏性使合并位置／速度／力矩误差下降约 9%／14%／9%。前进回放仍有倾倒，默认参数保持。见本机执行器验收记录。 | 四腿带载独立复测、前进初始化与闭环速度 |
| 2026-11-23 | 完成左前整腿 PhysX 原生摩擦的同目标承重与自身反馈对照。黏性小腿改善后退速度跟踪，四窗未过带载门槛。见本机执行器验收记录。 | 前进初态与接触匹配、整机受载验证 |
| 2026-11-23 | PhysX 原生静、动及黏性摩擦完成左前三轴台架筛选和步长复核。各轴有通过数值门槛的诊断参数，大幅跟踪较原 DCMotor 改善。见本机执行器验收记录。 | 联合三轴、受载跟踪和小幅幅相 |
| 2026-11-23 | 复算左前三轴历史台架，现有显式阻力在 5ms 下三轴均出现额外反号，小腿最明显。见本机执行器验收记录。 | 三轴替代摩擦的步长与实测幅相验收 |
| 2026-11-22 | 完成小腿 Va=.010/.050 双步长与补充摩擦筛选。显式模型 5ms 换向异常仍在，PhysX 原生摩擦稳定但小幅幅相待校准。见本机执行器验收记录。 | 稳定摩擦模型通过实测幅相及受载校验 |
| 2026-11-22 | 按用户定义复算 q_des→力矩反馈。走路时各轴相位不同，实机重建 PD→力矩比 MuJoCo 更迟。见[时序复查](docs/deployment/actuator_timing_review_2026-11-22.md)。 | 力矩实际建立与反馈时延拆分 |
| 2026-11-22 | 复算 5160 策略与主机目标时标，确认平均约 20ms、非固定逐拍间隔，未确认固定 12.5ms 电机延迟。仅更新[时序记录](docs/deployment/actuator_timing_review_2026-11-22.md)。 | 驱动采用时标及延迟候选验证 |
| 2026-11-22 | 核对 MuJoCo 实际足地接触并完成手托分解仿真。同一外扰下，减载主要降低小腿力矩，横滚阻尼主要降低摆幅。见本机 5160 回放记录。 | 实物足地参数、手托施力与实机起振源 |
| 2026-11-22 | 5160 长站立前后匹配速度指令，含手托的后段横滚快抖、髋速度与髋/小腿力矩多数下降。见本机 5160 回放记录。 | 两种手托的切换边界与减载／限横滚独立贡献 |
| 2026-11-22 | 核对 5160 的 6–8Hz 实际角度和速度。四髋实际速度与机身横滚强相干，实机髋速度波动约为自身反馈仿真的 5.1 倍。见本机 IMU 扰动 A/B 记录。 | 实机净横滚力矩与起振原因 |
| 2026-11-22 | 复核 5160 横滚和力矩互谱。实机四髋及四小腿在 7.5Hz 相干明显，仿真换入实机 IMU 后复现接近的同频力矩幅相。见本机 IMU 扰动 A/B 记录。 | 实机起振机制与作用于机身的净横滚力矩 |
| 2026-11-22 | 核对 5160 无托后退力矩。步态相位波形部分吻合，实机三类关节的总体与 6–8Hz 力矩波动均高于 MuJoCo 同指令闭环。见本机 IMU 扰动 A/B 记录。 | 独立力矩校验、触地冲量与受载归因 |
| 2026-11-22 | 用 5160 两段无托后退对齐对角承重相位。MuJoCo 横滚在触地后短暂抬升再回落，实机在估计承重前后都维持明显横滚。见本机 IMU 扰动 A/B 记录。 | 实机触地真值、单次输入与受载闭环衰减 |
| 2026-11-22 | 完成两方向的 6Hz gyro X 低通及 7.5Hz 陷波对照。陷波压下人为观测扰动，但物理横滚力矩撤除后恢复变慢；6Hz 低通使注入工况更晃。见本机 IMU 扰动 A/B 记录。 | 实机测量残差与物理激励分离 |
| 2026-11-22 | 完成 5160 与两组 MuJoCo IMU 幅相对比。7.5Hz 下横滚角速度与姿态接近微分关系，主要差异在实际振幅。见本机 IMU 扰动 A/B 记录。 | 独立姿态真值、传感器实际延迟和实机振荡源 |
| 2026-11-22 | 用 5160 原始实机 IMU 与速度指令驱动 7000 MuJoCo，三段前后走的髋、大腿、小腿目标 6–8Hz 幅值均更接近实机。连续回放因横滚到停止阈值提前结束。见本机 IMU 扰动 A/B 记录。 | 实机振荡首因、gyro 与姿态贡献 |
| 2026-11-22 | 重画 MuJoCo 的 1Hz IMU 低通时域与频谱。持续注入组的主峰从 7.5Hz 变为约 2Hz，整体摆幅接近；无注入时约 2Hz 摆动也增大。见本机 IMU 扰动 A/B 记录。 | gyro 与投影重力各自对慢摆的贡献、实机验证 |
| 2026-11-22 | 7000 的 MuJoCo 闭环完成四次 IMU 脉冲撤除及持续注入对照。撤除后 6–8Hz 横滚首秒回到基线量级，整体摆幅次秒恢复。详见本机 `docs/deployment/imu_disturbance_filter_ab_2026-11-22.md`。 | 实机激励来源及受载闭环是否维持振荡 |
| 2026-11-22 | 完成 7000 的 7.5Hz IMU 注入、1/3Hz 因果低通仿真及实机日志重算。注入可驱出相近量级的横滚；强滤波在干净仿真加大摆幅。原文见本机 `docs/deployment/imu_disturbance_filter_ab_2026-11-22.md`。 | 实机测量误差、实际承重扰动和闭环减振 |
| 2026-11-22 | 补齐 17:41 无托举后退同窗的四髋目标/反馈和 IMU 三组对照图。见本机 `docs/deployment/unassisted_command_replay_5160_2026-11-22.md`。 | 实机触地与目标应用时刻 |
| 2026-11-22 | 补画 17:41 无托举实机与仿真四个小腿的目标和反馈，前进及后退分图。见本机 `docs/deployment/unassisted_command_replay_5160_2026-11-22.md`。 | 实机触地标签待同步核对 |
| 2026-11-22 | 核对 17:41 实机日志的 7000 权重与无托举段。MuJoCo 完整速度指令和 Gazebo 41s 自身反馈回放完成，未复现实机约 7–8Hz 横滚。详见本机 `docs/deployment/unassisted_command_replay_5160_2026-11-22.md`。 | 托举子段、实机触地与目标应用时序 |
| 2026-11-22 | 复算 7000 平地前进对角落地力。两组冲击峰值中位数差约 17%，平均承重占比 50.7%/49.3%。详见本机 `docs/training/nov22_rough_training_review_2026-11-22.md`。 | 实机同拍落脚与足端反力 |
| 2026-11-22 | 复核实机行走与台架频响。行走 6–8Hz 的最大关节组相对差在四髋；台架小幅最大幅比差在左前髋 2Hz、最大相位差在左前大腿 2Hz。详见本机 `docs/deployment/walk_rack_frequency_review_2026-11-22.md`。 | 承重时的髋频响和接触/时标证据 |
| 2026-11-22 | 完成 7000 横滚脉冲和对角支撑相位配对试验。MuJoCo 里只增强四髋的横滚角速度反馈 10%，恢复偏差 RMS 中位数降约 6%，跟速相近。见[横滚排查](docs/deployment/ppo_vibration_status_2026-11-22.md#current-checks)。 | Gazebo 复测、实机接触相位和目标应用时序 |
| 2026-11-22 | 新增 base 惯量缩放工具。已核对三轴变换、网格引用、源文件保护，并在 Isaac Lab 加载生成的 X 轴 45% 候选完成四组短回放。 | 真实 base 质量分布与实机同条件横滚 |
| 2026-11-22 | 更正为沿 X 轴集中，完成 7000 策略的 20%/45% 对照。45% 组同速前进的横滚角速度 RMS 在 MuJoCo/Isaac Lab 分别增约 19%/31%，6–8Hz 略降。见本机 `docs/deployment/base_inertia_7000_2026-11-22.md`。 | 实际 base 惯量和实机止抖 |
| 2026-11-22 | 7000 策略的 base 惯量 A/B 完成。MuJoCo 与 Isaac Lab 同速前进的横滚角速度 RMS 约增 3%，6–8Hz 分量均略降。详见本机 `docs/deployment/base_inertia_7000_2026-11-22.md`。 | 实际惯量、实机同条件横滚与模型基线 |
| 2026-11-22 | 从 18 组平地回放提取对脚落地时序。+0.5m/s 时 9999 比 7000 的交替间隔长短差更大，完整周期波动接近。见[训练摘要](docs/training/pasted_rough_config_review_2026-11-22.md#当天回传的训练结果)。 | 实机足端接触时序、相位奖励效果 |
| 2026-11-22 | 补测 Rough 6000/7000/8000，3 份正式导出、24 组平地与 36 组粗糙地形闭环完成。见[训练摘要](docs/training/pasted_rough_config_review_2026-11-22.md#当天回传的训练结果)。 | 实机无托同速横滚与楼梯通过率 |
| 2026-11-22 | 修复 Isaac Lab 启动器找不到共享 `play_utils` 的问题。五项启动器测试、PPO 5000 CPU 正式导出与输出一致性检查通过。详细验证见本机 `docs/training/isaaclab_export_launcher_2026-11-22.md`。 | CUDA 导出 |
| 2026-11-22 | 复核新 Rough 训练记录，完成中末权重正式导出、32 组平地、2 组旧几何和 36 组粗糙地形闭环。见[训练与回放摘要](docs/training/pasted_rough_config_review_2026-11-22.md#当天回传的训练结果)。 | 服务器训练资产哈希、实机无托止抖、18 cm 两阶上/下 |
| 2026-11-22 | 收窄 Git 文档同步范围。保留服务器操作、俯仰和位控手册，以及模型、训练与振荡摘要。实验明细、逐样本数据和图片留在本机。已验证清单和本地引用。 | 实机俯仰几何表更新与校验 |
| 2026-11-22 | 旧记录汇总，近期横滚实验归并，入口去重，原文归档并核对引用。见[历史总结](docs/maintenance/project_history_2026-09_to_2026-10.md)及[当前排查](docs/deployment/ppo_vibration_status_2026-11-22.md) | 原有训练、受载与实机验收按 §7 跟进 |
| 2026-11-22 | 新几何同步及 Rough 附件核对，含实际启用地形、课程、奖励与电机。见[同步](docs/deployment/model_sync_2026-11-22.md)、[训练设置](docs/training/pasted_rough_config_review_2026-11-22.md) | 服务器短训练、新几何走动、18 cm 台阶 |
| 2026-11-19～21 | 无托/手托、两仿真器逐轴对比，台架、步长及可选阻力模型已做成对验证。见[横滚](docs/deployment/ppo_vibration_status_2026-11-22.md)、[电机](docs/deployment/ppo_vibration_status_2026-11-22.md#actuator-validation) | 带载跟踪、机械零位和实际横滚触发 |
| 2026-11-13～18 | 同目标、PD、观察分组和完整反馈对照，形成 18 cm 路线。见[历史试验读法](docs/deployment/ppo_vibration_status_2026-11-22.md#replay-and-filtering)、[训练路线](docs/training/ppo_18cm_stairs_route_2026-11-13.md) | 无托同速止抖、上下两阶验收 |
| 2026-11-02～11 | 低速、速度扩展与地形续训对照，完成轴角、同目标回放和配对姿态诊断。见训练分支对照（本机 `nov11_two_runs_review_2026-11-11.md`）、[实机状态](docs/deployment/ppo_vibration_status_2026-11-22.md#replay-and-filtering) | 独立实机复验 |
| 2026-09～10 | 模型及部署整合、AMP 配对与导出、PPO 试验、拖曳验证。见[历史总结](docs/maintenance/project_history_2026-09_to_2026-10.md) | 各实验边界见原文和 §7 |
