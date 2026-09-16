# Imgo2 项目说明与维护记录

> 最后核对：2026-09-17。本文是整个工作区的维护入口，覆盖项目总览、训练部署流程、已知问题和变更记录。
> 状态依据包括用户反馈、本地源码/离线检查，以及 2026-09-17 实际执行的 `build.sh -mj`、
> MuJoCo 物理回放、一次针对三个策略的运行期排查（[记录](docs/sim2sim_policy_runtime_2026-09-17.md)）
> 和 ROS 2/Gazebo 链路打通（[记录](docs/gazebo_ros2_bringup_2026-09-17.md)，无头验证到策略闭环）。
> 本轮**未**重新运行训练、Isaac Lab 回放或真机实验，也**未**在 GUI 里人工确认（本机显示不可用）。

## 1. 项目总览

Imgo2 是面向四足机器人的强化学习运动控制工作区，包含机器人模型、Isaac Lab 训练环境、参考动作数据和 C++ 部署框架。

当前代码链路为：

```text
URDF 与网格 ──→ Isaac Lab 环境 ──→ PPO / HIM-Loco / AMP 训练
参考动作数据 ──────────────────→ AMP 数据加载与参考状态初始化
训练 checkpoint ──→ 策略回放与导出 ──→ 部署配置对齐
                                      └─→ MuJoCo / Gazebo / 真机入口
```

最后一段部署链路仍需验证。`himloco` 仍是 Go2 参考占位策略；AMP 已于 2026-09-17 换成
Imgo2 自己的 checkpoint（配置与关节/物理口径已对齐、构建与物理回放已跑通），但策略只给出
静态下蹲、未形成步态（AMP-06），因此**仍不能认定 Imgo2 已完成训练到部署的闭环**。

### 目录职责

| 路径 | 内容与用途 |
|---|---|
| ~~imgo2_model/~~ | **2026-09-17 已删除**（git 可追溯）。原先只有腿部件片段 URDF（无 `base`、无 `<robot>`，XML 不能独立解析）+ 两套模型网格，无任何消费者；其网格与 `imgo2_description/meshes` 内容相同。删除理由见维护记录与 MODEL-02 |
| [imgo2_description/](imgo2_description/) | **模型唯一源（2026-09-17 起）**。`xacro/core.xacro` 是物理内核（17 link/16 joint，命名 FL/FR/RL/RR），`xacro/robot.xacro` 是组装入口，按开关拼上 `transmission.xacro`（仅 ros_control）、`gazebo.xacro`（插件 + per-link 接触参数）、`imu.xacro`（imu_link）。生成物入库、勿手改：`urdf/imgo2.urdf`（纯）、`urdf/imgo2.gazebo.urdf`（+三者）；`mjcf/{imgo2.xml,scene.xml}` 是 `rl_sim_mujoco` 读的 MuJoCo 模型（训练物理 + 参考项目求解器/接触块 + `framelinvel`）。`meshes/` 是全仓唯一一套网格（指纹 `8dc5b5995a11`） |
| [imgo2_rl/](imgo2_rl/) | Isaac Lab 扩展、任务配置、训练脚本、自定义算法包。**模型与网格副本已于 2026-09-17 删除**，训练直接读 `imgo2_description/urdf/imgo2.urdf`（见 `assets/imgo2.py`） |
| [imgo2_dataset/](imgo2_dataset/) | 参考动作数据；当前数据位于 `datasets/imgo2_motion/` |
| [imgo2_deploy/](imgo2_deploy/) | C++ 推理、观察缓存、状态机、MuJoCo/Gazebo/真机入口及策略配置。**模型副本 `robot_description/` 已于 2026-09-17 删除**，MuJoCo 读 `imgo2_description/mjcf/`、Gazebo 读 `imgo2_description/urdf/imgo2.gazebo.urdf` |
| [docs/](docs/) | 排查记录与离线结果：`amp_alignment_review.md`、`code_review_2026-09-15.md`、`amp_data_audit.json` |
| [paper_plan_imgo2.md](paper_plan_imgo2.md) | 小论文与毕设选题、实验和写作建议 |
| [research_exploration_plan.md](research_exploration_plan.md) | 本体感知、执行器随机化、动作跟踪与真机准备的研究规划 |
| [imgo2_rl/rlfromgym2lab.md](imgo2_rl/rlfromgym2lab.md) | IsaacGym AMP 迁移到 IsaacLab 时 env/wrapper 对接层的说明（背景资料，非当前结论） |

研究规划中的日期、目标和预期结果属于历史计划，不能直接视为当前进展或实验结论。

## 2. 当前状态

| 模块 | 已核对事实 | 尚待验证 |
|---|---|---|
| 模型 | 四份完整 Imgo2 URDF 的关节轴/限位、每个 link 的质量/质心/惯量/碰撞几何已于 2026-09-15 按训练侧统一（`check_model_sync.py` 五项全通过）；`imgo2_description` 腿序不同是有意设计 | Gazebo/ROS 链路需复核；`imgo2_model/` 的 URDF 是腿部件片段，统一模型副本仍需先补全 |
| PPO | 用户于 2026-09-15 确认已完成训练及 sim 验证；代码有平地、粗糙地形、变高度、倒立注册项 | 待补具体已验证任务 ID、checkpoint 和指标；不推定所有注册任务均已验证 |
| HIM-Loco | 环境、历史观察包装器、训练、回放、JIT/ONNX 导出及比较脚本已存在 | 导出数值一致性与 C++ 端适配 |
| AMP | 用户反馈已有训练步态，但策略贴地爬行；已完成本地数据/URDF 离线核对并修正映射、归一化，补充高度约束及奖励量级调整 | 新配置尚待重新训练验证；正常初始化不视为策略能维持高度 |
| 数据集 | 21 份 JSON 格式 `.txt`，共 5097 帧；每帧 61 个数，帧间隔均为 0.02 秒 | 动作语义、腿顺序和运动学一致性的回放验证 |
| 数据副本 | `imgo2_dataset/datasets/imgo2_motion/` 的 21 份文件与训练目录对应副本逐文件哈希一致 | 后续更新时防止两处副本漂移 |
| MuJoCo 部署 | `robot_description/imgo2_mjcf/scene.xml` 已就位，关节轴/限位/传感器布局与训练侧逐项一致；2026-09-17 `bash build.sh -mj` 构建通过；参数已按参考项目对齐（见 §5.3） | GUI 入口需在有可用显示的机器上运行；AMP 策略当前只给出静态下蹲、不跟踪速度指令（AMP-06） |
| AMP sim2sim | FSM 增量式新增 `RLFSMStateAMPLocomotion`（键 `2`）；`policy.pt` 与 checkpoint actor 128 组输入最大误差 0；16 s 回放无 NaN | 步态未成立；`lin_vel` 观测恒为 0（AMP-06）；GUI 未运行 |
| 真机部署 | 有 `rl_real_imgo2.cpp` 和状态机代码 | Unitree SDK2 目录当前为空，CMake 会跳过真机目标；硬件通信适配未验证 |

## 3. 代码阅读导航

训练任务主要位于 `imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/`。

| 需要理解或修改的内容 | 主要入口 |
|---|---|
| 机器人模型、初始关节角、执行器 | [assets/imgo2.py](imgo2_rl/source/imgo2_rl/imgo2_rl/assets/imgo2.py) |
| 公共场景、观测、动作、奖励、随机化、课程 | [velocity_env_cfg.py](imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py) |
| 基础运动任务注册 | [base_move/__init__.py](imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/__init__.py) |
| 平地、粗糙地形、HIM-Loco、AMP 配置 | [base_move/](imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/) 下对应 `*_env_cfg.py` |
| 算法超参数与实验名称 | 同一任务目录下的 `agents/` |
| 自定义奖励、观测、重置、速度指令 | [mdp/](imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/mdp/) |
| 自定义算法、网络、runner、包装器、数据加载器 | [scripts/rl_lab/rl_lab/](imgo2_rl/scripts/rl_lab/rl_lab/) |
| AMP 动作数据离线核对（高度、FK、腿顺序） | [audit_amp_dataset.py](imgo2_rl/scripts/tools/audit_amp_dataset.py)、[check_amp_joint_order.py](imgo2_rl/scripts/tools/check_amp_joint_order.py) |
| 资源路径自检（新机器/训练服务器前置） | [check_asset_paths.py](imgo2_rl/scripts/tools/check_asset_paths.py) |
| 模型副本一致性（关节/惯量/碰撞/网格/FK） | [check_model_sync.py](imgo2_rl/scripts/tools/check_model_sync.py) |
| 部署观察处理与策略推理 | [rl_sdk.cpp](imgo2_deploy/src/imgo2_deploy/library/core/rl_sdk/rl_sdk.cpp) |
| 部署状态切换 | [fsm_imgo2.hpp](imgo2_deploy/src/imgo2_deploy/fsm_robot/fsm_imgo2.hpp) |
| 构建选项与外部依赖条件 | [build.sh](imgo2_deploy/build.sh)、[CMakeLists.txt](imgo2_deploy/src/imgo2_deploy/CMakeLists.txt) |

配置有继承关系：理解最终行为时，要同时检查公共配置和任务类的 `__post_init__()`。例如粗糙地形 PPO 和 HIM-Loco 配置都将随机推扰、外力事件设为 `None`，不能仅凭公共类存在相应定义就写成“已启用”。

## 4. 训练与回放流程

### 4.1 环境与路径前提

以下 Python 命令均从 `imgo2_rl/` 执行，并使用已配置好 Isaac Lab/Isaac Sim 的 Python 环境。

- `imgo2_rl` 和 `rl_lab` 安装脚本声明 Python `>=3.10`；这不是完整的 Isaac Lab、Isaac Sim、CUDA 和 PyTorch 兼容性记录。
- 本次没有核验这些依赖的实际安装版本。首次跑通后，在第 8 节记录实际版本和启动命令。
- `assets/imgo2.py` 的 URDF 与动作数据路径已改为**由文件自身位置推导**（`Path(__file__)` 上溯 4 层得到 `imgo2_rl/` 项目根），不再写死机器绝对路径。无论仓库 clone 到哪里、从哪个目录启动都成立；前提是可编辑安装（`pip install -e`），非可编辑安装会把包拷进 site-packages，届时数据目录不在上溯路径上。
- 需要把数据或模型放在别处时，用环境变量覆盖而不用改代码：`IMGO2_AMP_MOTION_DIR` 覆盖动作数据目录，`IMGO2_URDF_PATH` 覆盖 URDF 路径。
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

下表来自源码注册项，尚未通过运行时枚举验证。

| 任务 ID | 训练入口，相对 `imgo2_rl/` |
|---|---|
| `Imgo2-basemove-rough-ppo` | `scripts/rsl_rl/train.py` |
| `Imgo2-basemove-flat-ppo` | `scripts/rsl_rl/train.py` |
| `Imgo2-heightmove-rough-ppo` | `scripts/rsl_rl/train.py` |
| `Imgo2-handstand-rough-ppo` | `scripts/rsl_rl/train.py` |
| `Imgo2-basemove-rough-himloco` | `scripts/rl_lab/himloco/train.py` |
| `Imgo2-basemove-flat-amp` | `scripts/rl_lab/amp/train.py` |

此外注册了 `Imgo2-basemove-rough-himloco-play` 和 `Imgo2-basemove-flat-amp-play`，分别使用对应回放配置。

基础检查及训练命令示例：

```bash
python scripts/tools/zero_agent.py --task=Imgo2-basemove-flat-ppo --num_envs=1
python scripts/rsl_rl/train.py --task=Imgo2-basemove-flat-ppo --headless
python scripts/rsl_rl/train.py --task=Imgo2-basemove-rough-ppo --headless
python scripts/rl_lab/himloco/train.py --task=Imgo2-basemove-rough-himloco --headless
python scripts/rl_lab/amp/train.py --task=Imgo2-basemove-flat-amp --headless
```

公共配置默认 `num_envs=4096`。初次检查可使用脚本的 `--num_envs` 参数缩小规模；具体训练规模依实际显存确定。

### 4.3 回放与策略导出

将下面的占位路径替换为实际 checkpoint 的绝对路径：

```bash
python scripts/rsl_rl/play.py --task=Imgo2-basemove-flat-ppo --num_envs=1 --checkpoint="/absolute/path/to/model.pt"
python scripts/rl_lab/himloco/play.py --task=Imgo2-basemove-rough-himloco-play --num_envs=1 --checkpoint="/absolute/path/to/model.pt"
python scripts/rl_lab/amp/play.py --task=Imgo2-basemove-flat-amp-play --num_envs=1 --checkpoint="/absolute/path/to/model.pt"
```

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

应从运行时 action manager 确认最终关节排列、裁剪和缩放。不要用 URDF 中的文本顺序代替运行时关节顺序。

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

来源：[base.yaml](imgo2_deploy/policy/imgo2/base.yaml)、[himloco/config.yaml](imgo2_deploy/policy/imgo2/himloco/config.yaml)。下面是静态文件值，运行时还需核对加载和覆盖结果。

| 项目 | 训练侧 | 部署配置现状 |
|---|---|---|
| 物理关节列表 | 动作使用 `FL, FR, RL, RR` | `base.yaml` 列表为 `FR, FL, RR, RL`；策略配置有相应 mapping |
| 默认关节角 | 所有腿为 `0, 0.87, -1.82` | base 为 `0, 0.8, -1.5`；策略配置又含左右 hip 偏置及前后 thigh 差异 |
| 策略 PD | 模型名义值 `25.0 / 0.5` | 策略 `rl_kp/rl_kd` 为 `40 / 1`，固定姿态 PD 是另组参数 |
| 力矩限制 | 模型 `23.7` | base 为 `23.5`，策略配置为 `33.5` |
| 指令缩放 | HIM-Loco 未显式设置 | 策略配置为 `[2.0, 2.0, 0.25]` |
| 网络文件 | 回放导出 `policy.pt` 或两个 ONNX 网络 | 策略配置加载 `himloco.pt`，属参考占位策略 |

上表描述的是**仍为 Go2 参考占位的 himloco 策略**（DEPLOY-01 未关闭）。AMP 策略已于
2026-09-17 按训练侧 + 参考项目对齐，见下表；详细依据见 [sim2sim 记录](docs/sim2sim_amp_2026-09-17.md)。

| AMP 项目 | 训练侧 | [amp/config.yaml](imgo2_deploy/policy/imgo2/amp/config.yaml) 现状 |
|---|---|---|
| 观察 | `base_lin_vel/ang_vel/gravity/commands/joint_pos/joint_vel/actions` = 48 | 同顺序、同项数（48） |
| 关节映射 | 动作与观察均为 `FL, FR, RL, RR` | `joint_mapping` 恒等，MuJoCo 场景关节声明顺序即策略顺序 |
| 默认关节角 | `0, 0.87, -1.82` ×4 | 同（`base.yaml` 与 amp 配置一致） |
| 策略 PD | `25.0 / 0.5` | `25.0 / 0.5`（与参考项目一致） |
| 固定姿态 PD | — | `60.0 / 2.0`（参考项目） |
| 力矩限制 | `23.7` | `23.7`（与 URDF/训练一致，参考项目为 23.5） |
| 动作缩放 / 裁剪 | `0.125 / 0.25 / 0.25`，取值 `±3` | 同 |
| 观察缩放 | `1.0 / 0.25 / 1.0 / 0.05 / 1.0` | 同 |
| 网络文件 | `imgo2_rl` 的 `play.py` 导出 | `policy.pt`（与 checkpoint actor 数值一致） |

### 5.4 部署侧三个策略与按键

部署侧现在并列三个策略，每个是「一个按键 + 一个完整的 FSM 状态类」：

| 键 | 状态类 | 配置目录 | 观察 | 网络 |
|---|---|---|---|---|
| `1` | `RLFSMStatePPOLocomotion` | `policy/imgo2/ppo/` | 45 维（无 `base_lin_vel`、无 `height_scan`） | `policy.pt`（参考项目 `~/RL/sim2sim/Imgo2_deploy/policy/imgo2/base_move/policy_flat.pt`，sha256 `53a57f909b61d2a7…`；2026-09-17 按四项指标评过 5 个候选后选它） |
| `2` | `RLFSMStateRLLocomotion` | `policy/imgo2/himloco/` | 45 维 × 6 帧 = 270（`observations_history: [0..5]`） | `himloco.pt`（Go2 参考占位，见 DEPLOY-01） |
| `3` | `RLFSMStateAMPLocomotion` | `policy/imgo2/amp/` | 48 维（含 `lin_vel`） | `policy.pt`（`model_9000.pt` 导出） |

`ppo/config.yaml` 的数值取自参考项目 `.../base_move/config.yaml`（可追溯）：`rl_kp 25 / rl_kd 0.5`、
`default_dof_pos 0 / 0.87 / -1.82`、`action_scale 0.125/0.25`、`clip ±3`、观测 scale
`0.25 / 1.0 / 1.0 / 1.0 / 0.05 / 1.0`、`joint_mapping` 恒等。**换 checkpoint 必须连这些值一起换**：
仓库里三份 45 维 PPO 候选用的是两套参数（`0/0.8/-1.5` + `20/0.2` 或 `20/1.0`，与
`0/0.87/-1.82` + `25/0.5`），混用会让 `dof_pos` 相对量与 PD 都错位。

**2026-09-17 PPO 选型（Gazebo，四项指标，脚本 `imgo2_deploy/scripts/eval_gazebo_policy.py`）**：
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
[策略运行期排查](docs/sim2sim_policy_runtime_2026-09-17.md) 第 7 节，Gazebo 侧的完整评测见
[Gazebo 记录](docs/gazebo_ros2_bringup_2026-09-17.md) 第 9 节。

复跑评测（Gazebo 无头，脚本自己注入起身/键1/速度）：

```bash
ros2 launch imgo2_deploy gazebo.launch.py gui:=false      # 终端 A
ros2 run imgo2_deploy rl_sim                              # 终端 B
python3 imgo2_deploy/scripts/eval_gazebo_policy.py --vx 0.5 --duration 20   # 终端 C
```

交付一个可部署策略时，至少记录：checkpoint 来源、模型版本、关节映射、默认姿态、动作缩放与裁剪、PD 与力矩限幅、观察顺序与缩放、历史排列及重置方式、四元数约定、控制周期、网络输入输出，以及同一输入下的数值比较结果。

## 6. 部署流程

下面为现有 Bash 构建脚本对应的命令，面向具备相关依赖的 Linux 环境；本次未在 Windows PowerShell 中执行。

### 6.0 部署包结构

`imgo2_deploy/` 源自 `rl_sar-main` 并裁剪为单一机器人（上游为 Apache-2.0，其 `LICENSE` 保留在该目录内）。内部按 ROS 工作区组织：

| 路径 | 内容 |
|---|---|
| `src/imgo2_deploy` | RL 部署包：sim2sim、MuJoCo 仿真、Imgo2 真机入口 |
| `src/robot_msgs` | 共享的电机/机器人状态消息 |
| `src/robot_joint_controller` | ROS 仿真用的 Gazebo 关节控制器；用 URDF 的关节限位 clamp 指令（ROS1 生效，ROS2 原先失效，见 DEPLOY-07） |
| `policy/imgo2` | 策略配置：`base.yaml`、`ppo/`（键 1，`imgo2_rough/2026-06-21_16-37-38` 的导出）、Go2 参考占位的 `himloco/`（键 2）、`amp/`（键 3，`model_9000.pt` 的导出） |
| `../../imgo2_description/urdf/imgo2.urdf` | 部署侧 ROS/Gazebo 用 URDF（唯一模型源在 `imgo2_description/`，见 MODEL-02） |
| `../../imgo2_description/mjcf/` | MuJoCo 场景；由 `rl_sim_mujoco` 经编译期 `IMGO2_MODEL_DIR` 读取，关节轴/限位与训练侧逐项一致 |

策略配置里关节名用 `*_shank_joint`（对应本仓库 URDF），而不是 Go2 的 `*_calf_joint`。

### 6.1 MuJoCo：训练仿真到另一仿真器

```bash
cd imgo2_deploy

# 推理运行时与 MuJoCo 依赖；若本机已有安装，可把 library/ 软链到它（见下）
bash scripts/download_inference_runtime.sh
bash scripts/download_mujoco.sh

# 构建（等价于参考项目的 ./build.sh -mj）
bash build.sh -mj

# 运行；参数为 <robot_name> <scene_name>，场景取 $IMGO2_MODEL_DIR/mjcf/<scene>.xml
# （IMGO2_MODEL_DIR 由 CMakeLists 编译期设为 <repo>/imgo2_description）
./cmake_build/bin/rl_sim_mujoco imgo2 scene
```

复用既有依赖（本机做法，零拷贝，仓库内不留文件）：

```bash
cd imgo2_deploy
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
[sim2sim 记录](docs/sim2sim_amp_2026-09-17.md) 第 4 节。无 GUI 时的替代做法是仓库外临时 harness
（复用真实 `RL`/FSM/`librl_sdk` 与 `policy/`，只重写 `GetState`/`SetCommand`/`RunModelStep`/`Forward`），
本轮三个策略的验证都用它完成，要点与数值见
[策略运行期排查](docs/sim2sim_policy_runtime_2026-09-17.md)。

**改 MJCF 传感器时必须挂在 site 上**：`<framequat>`/`<framelinvel>` 若写成 `objtype="body"`，
返回值会再乘该 body 的惯量主轴旋转（`iquat`），base 的 `fullinertia` 主轴不是
`ixx<iyy<izz` 排列，于是姿态整体偏转 90°，策略会把直立判成翻倒（MODEL-03）。

### 6.2 ROS / Gazebo

2026-09-17 已把这条链路从上游的 ROS 1 `liblegged_hw_sim.so` 改为**本机可用的 ROS 2
`gazebo_ros2_control`（Humble + Gazebo 11）**，并在无头下单次调用内跑通到策略闭环
（详见 [Gazebo 链路打通记录](docs/gazebo_ros2_bringup_2026-09-17.md)）：

```bash
cd imgo2_deploy

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

## 7. 待办与已知问题

优先级含义：P0 为影响训练启动或部署正确性的前置问题，P1 为后续验证与维护事项。

| ID | 优先级 | 状态 | 问题与依据 | 完成标准 |
|---|---|---|---|---|
| DOC-01 | P1 | 已完成 | 训练 README 曾引用失效的 `script/himloco_rsl_rl` 安装路径和写死的个人服务器 checkpoint | 已改为 `scripts/rl_lab`，checkpoint 改为 `<run>` 占位并注明不可沿用；根 README 与子项目 README 表述一致 |
| ENV-01 | P0 | 代码已修正，待服务器验证 | 原先 `assets/imgo2.py` 的两条路径写死为 `/root/gpufree-data/Imgo2_rl/...`（当时目录名还是 `Imgo2_rl`），且假定该目录就是项目根，合并成 monorepo 后必然失效且 glob 为空时无明确报错 | 已改为由 `Path(__file__)` 推导项目根，并支持 `IMGO2_AMP_MOTION_DIR`／`IMGO2_URDF_PATH` 覆盖；新增 `scripts/tools/check_asset_paths.py` 供新机器自检（本机通过：URDF 存在、动作文件 21 份、无残留机器路径）。服务器上仍需运行该脚本并记录实际加载路径 |
| MODEL-01 | P0 | 已按训练侧参数统一并验证；Gazebo/ROS 侧待复核 | **用户决定：四份模型的物理参数一律以训练侧为准。** 已按此统一：① 部署份 12 个腿部关节 axis 原全部取反（q_部署 = −q_训练）且限位镜像 → 改为训练份约定；② 部署份的足端 CoM/惯量与后腿 shank/thigh 惯量非对角项是五份模型中的**唯一异类**（`imgo2_model/` 片段、`imgo2_description` 两个 URDF、训练份四份一致）→ 改为训练侧；③ 足端碰撞圆柱长度训练侧与片段为 `0.01`、采集模型两个 URDF 与部署份为 `0.02`（2 对 3）→ 用户选训练侧 `0.01`，四份已统一；④ base 质量与惯量统一为 `5.53394020` + `0.03866860/0.10411461/0.12554111`。**现在四份完整 URDF 的 12 个关节轴/限位、17 个 link 的质量/质心/惯量/碰撞几何、base 规范值全部一致**（按逻辑关节名与逻辑 link 名比对；统一前容忍 `imgo2_description` 的有意腿序与 LF_/LH_ 命名，2026-09-17 统一后已同序，见 MODEL-02）。**为什么必须改部署份**：deploy 三条推理链路（`rl_real_imgo2.cpp` 走 Unitree SDK、`rl_sim.cpp` 走 ROS 控制器、`rl_sim_mujoco.cpp` 走 MJCF）均无任何符号取反，只有 `joint_mapping` 索引置换，无补偿；且 `robot_joint_controller` 会用 URDF 的关节限位 clamp 指令。注意两版原本行为不同：ROS1 自己定义了按引用修改的 `clamp(double&,…)`，确实生效；**ROS2 把 `std::clamp(…)` 当语句调用、丢弃返回值，等于完全没有限位**，且单关节版请求的参数名多写了一个下划线（`"robot_description_"`），URL 从未解析成功、`joints_urdf_` 为空指针。两处已在本轮修正，但尚未编译验证（见 DEPLOY-07）——旧限位 `thigh -2.87..0.9`、`shank 0.733..3.0` 会把 `default_dof_pos` 的 shank `-1.50` clamp 成 `+0.733`，即该链路连默认站姿都不成立，属功能性断裂。**验证**：四份 URDF 的 FK 全部重现录制数据（最差恒等 RMSE 0.00214 m，部署份修符号前为 0.11790 m）；`check_model_sync.py` 五项检查全通过；两次负向测试（翻回符号、单独改足端 CoM）均能报错，恢复后 sha256 不变。唯一残留差异是外观：部署份比训练份多 4 个 `<material>` 块，不影响物理 | ① 在 Gazebo/ROS 链路复核（真机与 MuJoCo 链路不读该 URDF，不受影响）。② 若要把 `imgo2_model/` 片段补成完整 URDF 并让 rl/deploy 共用同一文件，还需处理 ROS 包内路径（建议 CMake 构建时拷贝）与 `build.sh:57` 的存在性检查；在此之前「共用一份文件」尚未实现。③ 遗留项：`imgo2_description/xacro/common/leg.xacro` 与 `urdf/imgo2.urdf` 全工作区无人引用，可考虑删除（未动） |
| AMP-01 | P0 | 恒等映射由数据独立确认 | 用户确认未覆盖采集参数；脚本按默认 LF/RF/LH/RH 写入关节和足端，等价于训练端 `[FL,FR,RL,RR]`。本地补充：改用实际采集模型 `imgo2_description.urdf` 复算 FK 与训练 URDF 结果一致（恒等 RMSE 均值 0.00107 m，声明顺序配对 0.22491 m）；髋外展左右对称性和关节位置/速度块相关性检查也不依赖 URDF 支持同一结论 | 采集来源、源码顺序和数据本身均已确认；新训练的行为改善仍待服务器验证 |
| AMP-02 | P0 | 历史报错，未复现 | 历史草稿记录 `RuntimeError: normal expects all elements of std >= 0.0`，调用栈为 `amp_on_policy_runner.py:136` → `amp_ppo.py:120` → `actor_critic.py:129` 的 `distribution.sample()` | 记录复现命令、数据和首个异常值；修复后训练验证；不能仅凭该报错断定根因 |
| AMP-03 | P0 | Torch 回归仍未执行过（本机无 torch，测试被跳过）；待训练环境验证 | 原均值方差更新使用归一化值，梯度惩罚使用未归一化值；已与 `amp_go2-main` 的处理方式对齐。**更正（2026-09-17 晚）：此条此前写成"Torch 回归已通过／6 项全过"是错的。** 本机（Linux `qmq-linux`）在 `~/miniconda3/envs/isaaclab` 下跑 `tests/test_amp_alignment.py` 的实际结果是 **5 通过 + 1 跳过**，跳过的正是本项 Torch 回归 `test_raw_statistics_and_normalized_gradient_penalty`（原因 `CPU regression needs the training environment's torch and numpy`）；全机五个 conda 环境与 `/usr/bin/python3` 都没有 torch（均报 `ModuleNotFoundError`），所以"该环境有 torch 2.7.0+cu128"的旧记录不成立，**AMP-03 的 Torch 那一半至今没有任何一次执行证据**（Windows 侧同样因缺 torch 跳过） | 需要一台真正装了 torch（有 Isaac Lab 训练环境）的机器跑 `python -m unittest discover -s tests -p test_amp_alignment.py`，确认 `test_raw_statistics_and_normalized_gradient_penalty` 不再 skip 且通过；之后再检查新训练中的判别器与归一化统计 |
| AMP-04 | P0 | 已加入待验证配置 | 原配置删除高度项和非法接触终止；现在保留 0.30 m 高度项、基座触地终止，并补偿任务奖励的时间步长缩放，任务混合系数改为 0.3 | 新训练高度稳定、贴地比例降低、速度跟踪可接受；具体权重仍需实验 |
| AMP-05 | P1 | 待重新训练 | actor 应移除 `base_lin_vel`，但现有 `model_9000.pt` 第一层为 `[512,48]`；2026-09-17 已按用户决定在配置中保留 `lin_vel` 项并以 48 维跑通管线 | 新建 45 维训练，取得对应 checkpoint 后同步修改回放、导出、部署观测并验证；不得直接截掉旧网络的 3 个输入。详见 [sim2sim 记录](docs/sim2sim_amp_2026-09-17.md) |
| AMP-06 | P0 | 启动异常已修；"不跟踪速度"已定性为策略性质，不再是未知缺陷 | ① **"一启动就飞"已修**：`PhysicsThread` 不套 keyframe，用的是模型默认 qpos（关节全 0）；原场景 `base pos="0 0 0.35"` 在直腿时足端 z=−0.0758（穿地），改为参考项目的 `0 0 0.5`（足端 +0.0743）后不再弹飞。② 又把参考项目的求解器/接触块抄进场景（`cone=elliptic impratio=100`、关节 `damping=1 armature=0.1`、碰撞 `condim=3 solref="0.005 1"` 与分几何 friction），GUI 等价启动下机器人从"躺地 0.075 m"变为**能站起**（AMP 后最高 0.2907、结束 0.2621；参考项目场景对照 0.3009／0.2927）。③ **2026-09-17 又查明姿态传感器缺陷（MODEL-03）**：`framequat` 原挂 body，`gravity_vec` 整体偏 90°，策略把直立当翻倒。改挂 `imu` site 后 AMP 站姿由 0.2663 变 0.3015，**但"不跟踪速度"依旧**。④ 把 `sensordata[43..45]`（framelinvel，世界系）经 `QuatRotateInverse` 旋进体系接进 `obs.lin_vel` 做对照实验：读数正确（行走时 `(0.087, 0.000, 0.012)`），AMP 结果不变（`vx=0.5` 时 `dx` 0.0612 对 0.0759、`thigh_range` 0.002 对 0.024）——**所以 `lin_vel` 恒 0 不是原因，本轮未改任何 deploy 代码**。⑤ 另一对照：参考项目 `base_move/policy.pt` 在同一 harness 的**我们模型**上站姿 0.2887、`vx=0.5` 走 0.454 m/s，而它在**参考自己的 MJCF** 上反而站不起来（0.0771），说明我们"物理取训练侧"的模型本身没问题 | 现结论：现有 `model_9000.pt` 就是一份"只站不走"的策略——既不跟速度指令，也不因 `lin_vel` 真假而改变（`vx=0.5` 时 10.5 s 只走 0.076 m = 0.007 m/s，四腿极差 0.014 rad，等于姿态锁死）。要速度跟踪须换一份会走的 AMP/运动模仿 checkpoint（顺带解决 AMP-05 的 45/48 维问题）；①②③ 已完成，仅剩 GUI 未运行与 AMP-05 的维数问题。详见[策略运行期排查](docs/sim2sim_policy_runtime_2026-09-17.md) |
| MODEL-02 | P0 | 已完成并验证（①–⑤） | 用户决定把模型统一到 `imgo2_description`（D1 命名 FL/FR/RL/RR；D2 Gazebo/IMU/transmission 留在 description 作可选模块；D3 生成物入库；D4 删冗余副本、git 兜底；D5 Gazebo 插件暂不处理）。**① 命名/网格**：`meshes/` 改 FL 命名（指纹 `8dc5b5995a11`，10 文件）。**② 模块化**：`xacro/core.xacro`（物理内核）＋ `{transmission,gazebo,imu}.xacro` ＋ `robot.xacro` 组装入口（开关默认 false）；生成物 `urdf/imgo2.urdf`（纯）与 `urdf/imgo2.gazebo.urdf`。**③ MJCF**：`mjcf/{imgo2.xml,scene.xml}`（训练物理 + 参考求解器/接触块 + `framelinvel`）。**④ 消费者切换**：RL 的 `assets/imgo2.py` 经 `_REPO_ROOT` 指向 `imgo2_description/urdf/imgo2.urdf`；`check_asset_paths.py` 改为按声明解析多个 root 变量；`audit_amp_dataset.py`/`tests`/`inertia_urdf.py` 同步；deploy 新增编译期 `IMGO2_MODEL_DIR`、`rl_sim_mujoco.cpp` 读 `imgo2_description/mjcf/<scene>.xml`；`build.sh` 与两个 Gazebo launch 改指 description；`CMakeLists` 安装列表去掉 `robot_description`。**⑤ 清理**：删除 `imgo2_model/`、`imgo2_rl/source/imgo2_rl/data/`、`imgo2_deploy/robot_description/`（共 46 文件、约 57 MB 工作树）；AGENTS.md 模型章节改写为单一源规则 | **验证**：`check_model_sync.py` 全 PASS（2 份已登记 URDF、网格指纹、mesh 引用存在性、FK 恒等 RMSE 0.00214 m / 交换 0.22470 m）；`check_amp_joint_order.py` 三项全 PASS 且数值与统一前一致（0.00107/0.00214 m、0.22491、0.0191/0.1339 rad、r 0.9546/0.2317）；`check_asset_paths.py` PASS（URDF 17 mesh 引用全在、21 份动作）；单元测试 5 通过 + 1 跳过（跳过的是需要 torch 的 AMP-03 回归）；`bash build.sh -mj` 删目录后仍构建成功，二进制内 `IMGO2_MODEL_DIR` 指向 `<repo>/imgo2_description`；`git ls-files -i -c --exclude-standard` 为空。**未运行**：GUI、Isaac Lab 训练/回放、ROS/Gazebo、真机；Gazebo 插件 `liblegged_hw_sim.so` 仍缺（DEPLOY-05 未解）。详见 [sim2sim 记录](docs/sim2sim_amp_2026-09-17.md) 第 6.6 节 |
| MODEL-03 | P0 | 已修并在无头回放验证，GUI 待确认 | MuJoCo 的 `framequat`/`framelinvel` 原先写成 `objtype="body" objname="base"`，返回值会再乘上该 body 的**惯量主轴旋转** `iquat`（base 的 `fullinertia` 主轴非 `ixx<iyy<izz` 排列，实际偏移 ≈180° 绕 (1,0,1)/√2），于是部署侧姿态观测整体偏转 90°：`GetUp` 结束后机器人明明直立（`xquat=(1,0,0,0)`），`gravity_vec` 却报 `(-1.000, 0.004, -0.025)` 而非 `(0,0,-1)`，策略把"站直"判成"已翻倒"——PPO 接管瞬间就输出饱和动作并塌成深蹲（`z_final=0.154`），himloco 侧倾，只有 AMP 恰好鲁棒。判据：`QuatRotateInverse(·,(0,0,-1))` 的第三分量恒为 `-1-2q_z² ≤ -1`，不可能出现 `-0.025`，故读到的三元组并非 `(g_x,g_y,g_z)`；再对照 `xquat` 与 `sensordata[36..39]` 的时间线确认是固定偏移。参考项目用的是 site（`objtype="site" objname="base_site"`）。修法：三个传感器改挂 base 内原有的 `imu` site（site 无 `quat`、`pos` 默认原点，故 site 系 = link 系），**声明顺序不变**，`sensordata` 偏移与 `GetState` 的 `[3n..3n+3]`/`[3n+4..3n+6]` 均无需改动；同时注释掉文件末尾那条陈旧 `<keyframe>`（`base z=0.35`，C++ 不应用但 GUI 的 Key 下拉框会应用） | 已通过：修后三个键的 `gravity_vec` 均为 `(0,-0,-1)`；PPO `vx=0/0.5/1.0` 站姿 0.321、实测 0.509/0.975 m/s（FL_thigh 极差 1.1–1.4 rad，是真步态）、AMP 站姿 0.3015；`check_model_sync.py`／`check_asset_paths.py`／`check_amp_joint_order.py` 仍全 PASS，单元测试 5 通过 + 1 跳过（跳过的是需要 torch 的 AMP-03 回归）。细节见[策略运行期排查](docs/sim2sim_policy_runtime_2026-09-17.md)。**待办**：在 GUI 里人工确认一次按键 1/2/3 与 1→2→3 切换 |
| DEPLOY-08 | P1 | 已归因：harness 假象，真实代码无此问题；GUI 仍待确认 | 2026-09-17 加 PPO（键 1）后，用**多键连续切换**的临时 harness（0→1→2→3）在进入 himloco 后崩溃：`mat1 and mat2 shapes cannot be multiplied (1x45 and 270x128)`，即把 45 维（单帧）输入喂给了 himloco 的 270 维（6 帧历史）网络。**2026-09-17 晚查明是我那版 harness 的错**：它的 `Forward()` 被简化成 `model->forward({ComputeObservation()})`，漏掉了真实 `RL_Sim::Forward()` 里的历史分支（`history_obs_buf.insert` → `get_obs_vec(observations_history)`）。把 `Forward()` 逐行照搬真实实现后，单键与 `0→1→2→3` 连续切换**都不再崩溃**；同一轮还确认仓库里 `RL_Sim::Forward()` 与参考项目逐行相同，故真实部署代码没有这个缺陷 | 用**真实 GUI 依次按 1→2→3** 再确认一次（预期不崩；会看到 himloco 把机器人掀翻，那是 DEPLOY-01/DEPLOY-06 的占位策略问题，不是崩溃）。若仍崩，再查 `RL::InitRL` 在「无历史配置 ↔ 有历史配置」切换时 `history_obs_buf` 与 `params["observations_history"]` 的时序 |
| DEPLOY-01 | P0 | 待对齐 | Go2 占位策略与 Imgo2 训练配置存在默认姿态、PD、限幅、指令缩放等差异（指 himloco 占位；AMP 已于 2026-09-17 对齐，见 §5.3；PPO 也已接入，见 §5.4） | 替换为来源明确的 Imgo2 策略，完成训练端与部署端同输入输出比较 |
| DEPLOY-02 | P0 | 场景已对齐并验证站起/行走，待 GUI 验证 | `imgo2_description/mjcf/{imgo2.xml,scene.xml}`（经编译期 `IMGO2_MODEL_DIR` 读取）：关节轴/限位与训练侧逐项一致、MuJoCo C API 可加载、`build.sh -mj` 通过；2026-09-17 按参考项目补齐初始高度（`base pos 0 0 0.5`）与求解器/接触块（`cone=elliptic impratio=100`、关节 `damping=1 armature=0.1`、碰撞 `condim=3 solref="0.005 1"`+friction），不再弹飞；同日又修 MODEL-03（姿态/线速度传感器由 `body` 改挂 `imu` site）并注释掉陈旧 keyframe，修后 PPO 站姿 0.321、`vx=0.5/1.0` 实测 0.509/0.975 m/s，AMP 站 0.3015 | 在有可用显示的机器上跑 `rl_sim_mujoco imgo2 scene` 并保存窗口记录。参考项目那份 MJCF 现在网格名已能对上（FL 改名的副产物，实测可加载），但它的 base 质量仍是旧值 6.53394、用 mesh 碰撞体且没写 `timestep`（=2 ms），同一个参考策略在它上面反而站不起来（`z_final=0.0771`），**不要改用它** |
| DEPLOY-03 | P0 | 缺依赖，待适配 | SDK2 目录为空，真机目标被跳过 | 依赖到位、目标生成、通信接口验证通过 |
| DEPLOY-04 | P1 | 已解决 | 原 `library/thirdparty/joystick/` 为空且未跟踪，`CMakeLists.txt` 在 `USE_MUJOCO` 下要求 `joystick.cc`，`rl_sim_mujoco.hpp` 还 `#include "joystick.hh"`；原无脚本会下载它。2026-09-17 已补入 `joystick.cc`/`joystick.hh`（与参考项目同源） | 已通过：2026-09-17 `bash build.sh -mj` 配置与编译均成功 |
| PPO-01 | P1 | 已按四项指标选定（`base_move/policy_flat.pt`） | PPO 权重此前用过 `imgo2_flat/2026-06-21_23-24-09` 与 `imgo2_rough/2026-06-21_16-37-38` 两份训练导出，用户反馈「腿部抖动很厉害」并给出评估标准：**位姿 / 速度跟随 / 抖动 / 周期性**。新增 `imgo2_deploy/scripts/eval_gazebo_policy.py`（自注入 /joy，从 /odom + /joint_states 统计四项）并对 5 个候选实测：参考项目 `base_move/policy_flat.pt`（6/30）vx=0.5 时 z 0.288、roll 2.5°、yaw 漂 −0.5°/s、抖动(大腿) 45 rad/s²、周期强度 0.95、FL-FR 175°、速度 0.428 m/s；`policy.pt`（6/30）速度更准 0.522 但抖动 79、roll 5.0°、强度 0.74；我们的 `rough/16-37-38` 抖动 241/507、roll 13.4°、强度 0.29、yaw 漂 −2.7°/s（vx=0 时还漂 +4.8°/s）；`policy1.pt` 超调 0.662；`amp/policy.pt` 不走（0.057 m/s、z 0.182）。⇒ 采用 `policy_flat.pt`（配置 25/0.5、默认 0/0.87/-1.82、clip ±3） | 已装并复测；后续若要速度精度可换 `policy.pt`（同配置）；新训练版本可用同一脚本按四项指标对比。详见 Gazebo 记录第 9 节 |
| JOINT-01 | P0 | ROS2 路径已修并验证；真机路径待处理 | `joint_mapping` 的语义是「策略第 i 个关节 ↔ 该路径数组第 joint_mapping[i] 号」，但三条路径的数组不同：MuJoCo 是 MJCF 声明顺序（FL,FR,RL,RR=策略顺序，恒等正确）；ROS2 是消息槽位，顺序 = 控制器 `joints` 参数 = `base.yaml` 的 `joint_names`（原为 Unitree SDK 的 FR,FL,RR,RL）；真机是 SDK 电机数组（固定 SDK 顺序）。于是恒等映射在 MuJoCo 对、在 ROS2 把策略 FL 接到物理 FR：Gazebo 里按 1 进 PPO 会翻成四脚朝天（实测 `/odom` z=0.073 m、roll=180°、dx=0），给速度指令后更明显。**修法**：① `base.yaml` 的 `joint_names`/`joint_controller_names` 改为模型顺序；② `rl_sim.cpp` 的 ROS2 分支新增 `OrderJointsByModelOrder()`，按 URDF 声明顺序重排传给控制器的名单（读不到则回退），MuJoCo 路径代码与语义未动。**验证**：Gazebo `vx=0` z 0.328–0.332 m、roll ±2.9°、dx≈0；`vx=0.5` 向前 8.20 m/12 s（0.56 m/s）、z 0.26–0.27、大腿摆幅 0.99–1.08 rad（修前四脚朝天）；IMU 侧 `/imu` 角速度与 `/odom` yaw 速率同号同量级，排除 IMU 约定问题 | 真机路径（`rl_real_imgo2.cpp` 索引 SDK 电机数组）要另给映射或改代码；偏航/走偏已确认是策略性质（MuJoCo 同策略 `vx=0.5` 也偏航 −62°/12 s），待换 checkpoint 或用 `axes[3]` 做航向闭环验证。详见记录第 8 节 |
| DEPLOY-05 | P0 | 已解决（无头跑通到策略闭环），GUI 画面待确认 | 原状：部署份 URDF 只有 ROS 1 式 `<transmission>`，`gazebo.xacro` 挂的是仓库内没有的 `liblegged_hw_sim.so`（`gazebo_ros_control`），ROS 2 下没有 `gazebo_ros_control`，于是 `rl_sim.cpp` 的 controller spawner 找不到 controller_manager。**2026-09-17 照参考项目改为 ROS 2 `gazebo_ros2_control`**：`gazebo.xacro` 加 IMU 传感器 + `<ros2_control>`（12 关节 effort 命令 + position/velocity/effort 状态）+ `libgazebo_ros2_control.so`；新增 `imgo2_description/config/robot_control_ros2.yaml`（`joint_state_broadcaster` + `robot_joint_controller/RobotJointControllerGroup`，后者是仓库内插件、由 `rl_sim` 自己 spawn）；`imgo2_description` 做成标准 ROS 2 包（ament + `package.ros1.xml`/`package.ros2.xml` + 软链进工作区 `src/`）；Gazebo 版网格改 `package://`（`robot.xacro` 的 `mesh_prefix`）；`build.sh` 的包扫描改 `find -L`；`check_model_sync.py` 认两种网格写法。**两个实测坑**：`<parameters>` 必须是真实文件路径（`package://`/相对路径都会让插件 Load 抛异常）；URDF 作为 `--param robot_description:=` 传给 controller_manager 时必须压成单行（否则 rcl 报 `Couldn't parse parameter override rule`，控制器起不来）；另外不能用 `gazebo_ros` 自带 launch（本机它起的 gzserver 未加载 factory 插件），改为直接起 `gzserver -s ...`。**验证**（无头、单次调用内）：spawn 成功且网格 0 报错；节点 `/gazebo_ros2_control`、`/imu_plugin` 出现；`ros2 control list_controllers` → `joint_state_broadcaster active`；`/joint_states` 与 `/imu` 发布；`rl_sim` 启动后用 `/joy` 注入 `A`→`RB+DPadUp`，打印 `RL Controller [ppo]`（进入键 1 的 PPO 闭环） | ① 在有可用显示的机器上确认 `gzclient` 画面（本机 `DISPLAY=:1` 建 GL 上下文失败）；步态**已用位姿客观验证**：vx=0.5 时 Gazebo 0.502 m/s、z 0.276 m、大腿摆幅 0.89–1.20 rad，与 MuJoCo（0.51 m/s、0.26 m、0.88 rad）一致，见记录第 7 节；② 长时间跑 himloco/AMP；③ 真机链路仍待验证。详见 [Gazebo 链路打通记录](docs/gazebo_ros2_bringup_2026-09-17.md) |
| DEPLOY-06 | P1 | AMP 路径已解决，himloco 路径仍在 | `rl_sim_mujoco.cpp` 用 `joint_mapping` 直接索引 MJCF，因此配置的映射必须与场景顺序一致。2026-09-17 采用自建场景（关节声明顺序 = 策略顺序 `FL,FR,RL,RR`）+ `base.yaml`/`amp/config.yaml` 恒等映射，AMP 路径自洽 | AMP 已在 16 s 回放中确认基座高度稳定、无 NaN；`himloco/config.yaml` 仍带硬件置换映射 `[3,4,5,0,1,2,9,10,11,6,7,8]`，在 MuJoCo 上会索引错腿，需单独处理；`imgo2_description/mjcf/` 那份现成 MJCF 未再复用 |
| DEPLOY-07 | P0 | 已修，待编译验证 | ROS2 两个控制器把 `std::clamp(…)` 当语句调用、丢弃返回值，限位实际失效；单关节版还把参数名写成 `"robot_description_"`，URDF 从未解析成功。已改为赋值形式并加空指针/越界保护、补 `<algorithm>`、修正参数名 | 在装有 ROS 2 的 Linux 上编译并跑通，确认限位生效且不再有空指针风险 |
| EXPORT-01 | P1 | 待验证 | 导出配置与 C++ 配置格式不同；比较脚本假设六帧历史 | 明确转换规则，记录实际网络维度、历史规则与误差指标 |
| DATA-01 | P1 | 当前一致 | 两处动作数据副本哈希一致 | 每次更新后核对副本，记录数据来源和版本 |
| EXP-01 | P1 | 待补充 | 已记录 PPO 验证与 AMP 贴地现象的用户反馈，尚缺对应日志、命令和模型路径 | 按第 8 节补充真实实验与产物路径 |

问题表只放结论与状态，依据、根因和未修项的细节见 [代码复审记录](docs/code_review_2026-09-15.md) 与 [AMP 对齐与参考项目对照](docs/amp_alignment_review.md)。**代码改了不等于修好**：只有经过编译、运行或测试验证的项才改状态，仅改代码未验证的写成「已修，待验证」并保留条目（见 AGENTS.md 的同名约定）。

关于 AMP-02：当前 `actor_critic.py` 直接学习 `std`，同时 `amp_ppo.py` 中存在有条件的最小标准差裁剪。仍需检查是否出现 NaN/Inf、裁剪条件是否生效以及数据/梯度是否异常；本次不将任何一个猜测登记为已确认根因。

历史草稿已收编：`imgo2_rl/todo.md`（AMP 腿顺序与上述报错的原始记录）并入本表 AMP-01／AMP-02；`imgo2_deploy/todo.md`（部署项目搭建需求）的内容已由第 6 节部署流程与目录职责覆盖。两个草稿文件已删除。状态以本表最近核验结果为准。

### AMP 对齐排查与下一轮训练

详细依据见 [AMP 对齐与参考项目对照](docs/amp_alignment_review.md)，离线结果见 [动作数据核对 JSON](docs/amp_data_audit.json)。

- `joint_mapping` 已改为恒等映射，仅适用于当前已核对的 `imgo2_motion` 数据。更换数据源后重新检查。
- 恒等映射不只依赖采集参数确认：对实际采集模型 `imgo2_description.urdf` 复算 FK 与训练 URDF 结果相同；准对称支撑的髋外展角符号、关节位置块与速度块的逐通道相关性也支持 `FL,FR,RL,RR`。详见 [AMP 对齐与参考项目对照](docs/amp_alignment_review.md) 第 3.3 节。
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
python scripts/rl_lab/amp/train.py --task=Imgo2-basemove-flat-amp --num_envs=256 --max_iterations=100 --seed=42 --headless
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

| 日期 | 变更 | 验证与限制 |
|---|---|---|
| 2026-09-17 | 建立 PPO 四项评估标准并完成选型（PPO-01）：换用参考项目 base_move/policy_flat.pt | 用户反馈当前 PPO 腿部抖动厉害，并定下评估标准：位姿 / 速度跟随 / 腿部抖动 / 周期性。**新增** `imgo2_deploy/scripts/eval_gazebo_policy.py`（只用 rclpy+标准库：自己注入 A→键1→axes[1]=vx，从 /odom(p3d 真值)+/joint_states 统计 z/roll/pitch/yaw漂、dx/dt、`abs(Δdq)/dt` 均值与 dq 换向率、大腿自相关周期与强度、FL-FR 相位）。**5 候选实测**（vx=0.5）：`base_move/policy_flat.pt`(6/30 12:45) z 0.288/roll 2.5°/yaw −0.5°/s/抖动 45.3/强度 0.95/相位 175°/速度 0.428；`base_move/policy.pt`(6/30 16:52) 0.522 m/s 但抖动 79.3、roll 5.0°、强度 0.74；`policy1.pt`(6/26) 0.662 超调、抖动 153；`rough/16-37-38`(换之前) 抖动 240.9/507.4、roll 13.4°、强度 0.29、yaw −2.7°/s；`amp/policy.pt`(7/3) 只 0.057 m/s、z 0.182 不走。vx=0 时两份 ref 抖动 0.0/yaw 漂 0（完全静止），rough 抖动 47–77、yaw +4.8°/s。**执行**：`policy/imgo2/ppo/policy.pt` ← `policy_flat.pt`（sha256 `53a57f909b61d2a7…`），`config.yaml` 换成参考 base_move 的值（25/0.5、0/0.87/-1.82、clip ±3、恒等映射）并写明来源与备选。**验证**：脚本对每个候选都跑出 JSON（/odom 100 Hz、/joint_states ~50 Hz），数字见 Gazebo 记录第 9 节与 README §5.4。**未做**：新训练 checkpoint 的四项对比（用同一脚本即可）、真机 |
| 2026-09-17 | 查明并修复 Gazebo 翻车主因：关节顺序（JOINT-01） | 用户报告 Gazebo 里 PPO 翻成四脚朝天且"只有 ppo 能前进"。用 `/odom` 真值定位：修前 z=0.073 m、roll=180°、dx=0（翻车）。**根因**：`joint_mapping` 被套在三条不同顺序的数组上——MuJoCo 是 MJCF 声明顺序（FL,FR,RL,RR，恒等正确）、ROS2 是消息槽位顺序（= 控制器 `joints` 参数 = `base.yaml` 的 `joint_names`，原为 SDK 的 FR,FL,RR,RL）、真机是 SDK 电机数组；ROS2 的槽位顺序只由我们传入的名单决定，与 URDF 无关，于是恒等映射把策略 FL 接到物理 FR。**修法（只动 ROS 路径与数据，模型/MuJoCo 不动）**：`base.yaml` 的 `joint_names`/`joint_controller_names` 改模型顺序；`rl_sim.cpp` ROS2 分支新增 `OrderJointsByModelOrder()` 按 URDF 声明顺序重排并把结果打印出来，读不到则回退+告警。**验证**：Gazebo `vx=0` z 0.328–0.332、roll ±2.9°、dx≈0；`vx=0.5` 向前 8.20 m/12 s（0.56 m/s）、z 0.26–0.27、大腿摆幅 0.99–1.08 rad。**顺带**：① IMU 排除——`/imu` 角速度与 `/odom` yaw 速率同号同量级；② 偏航/走偏是策略性质——MuJoCo 同策略 `vx=0.5` 也偏航到 −61.9°/12 s（仍前进 6.18 m）。**未做**：真机路径映射、GUI 画面、其它 checkpoint 对比 |
| 2026-09-17 | 加回 Gazebo 基座真值并完成 MuJoCo↔Gazebo 位姿对比（回答"Gazebo 完全不行"） | 用户反馈同策略 MuJoCo 好、Gazebo 完全不行，并指出"四脚朝天也很稳"说明只看姿态会误判。**做了什么**：`gazebo.xacro` 加回 `libgazebo_ros_p3d.so`（`/odom`，100 Hz，body `base`，world 系）并重生成 URDF（`check_model_sync.py` 仍 PASS）；随后用 `/joint_states`+`/odom` 客观测量。**实测**（策略 `ppo`=rough/16-37-38）：`vx=0` MuJoCo z 0.327/dx 0.023 对 Gazebo z 0.333（0.326–0.339）/dx −0.001；`vx=0.5` MuJoCo z 0.26、0.51 m/s、大腿摆幅 0.88 rad 对 Gazebo z 0.276（0.244–0.332）、**0.502 m/s**、0.89–1.20 rad、roll ±6°；GetUp 段两者都正常（roll/pitch≈0.1°、关节跟到 `0/0.87/-1.82`、命令 kp=60/kd=2）。**⇒ Gazebo 侧并非不可用，此前"完全不行"是用法问题**：① `/cmd_vel` 默认不生效（`rl_sim.cpp:465` 要 `navigation_mode`，默认 OFF），没给速度指令时策略当然只站着；② ROS 路径速度来源是手柄 `axes`／键盘 `W-S-A-D-Q-E`（需 `rl_sim` 的 stdin 是终端）／按 `N` 开 nav mode 后的 `/cmd_vel`；③ 键 `2`=himloco（Go2 占位会把机器人掀翻）、键 `3`=AMP（只站不走）。**一处副产物**：ROS 路径的槽位顺序是 `base.yaml` 的 `joint_names`（FR,FL,RR,RL）而策略是 FL,FR,RL,RR，`joint_mapping` 恒等时左/右腿互换；MuJoCo 路径数组本身就是 FL 顺序、恒等才对——两条路径语义不同，`joint_mapping` 需要按路径给（实测镜像步态在 Gazebo 仍能走，故暂未改，登记为待决定项）。**未验证**：GUI 画面渲染、长时间稳定性、真机 |
| 2026-09-17 | 复现并记录 Gazebo 运行时的 Python 环境陷阱（Isaac Lab 的 `PYTHONPATH` 顶掉 numpy，导致 `Failed to start joint controller`） | 用户实跑 `ros2 run imgo2_deploy rl_sim` 在进入 Passive 之后抛 Python traceback，最后 `Failed to start joint controller`。**根因**：`rl_sim.cpp` 的 `StartJointController()` 用 `sh -c "ros2 run controller_manager spawner robot_joint_controller -p <临时yaml>"` 起控制器，子进程继承 shell 环境；该 shell 若 source 过 Isaac Lab（或挂着 conda），`PYTHONPATH` 里会有 `~/isaac/IsaacLab/_isaac_sim/extscache/omni.kit.pip_archive-*/pip_prebundle`——一份给 **Python 3.11** 编的 numpy 1.26.0，被系统 **Python 3.10** 的 `spawner` 优先命中，C 扩展对不上即崩，spawner 非 0 退出 → `rl_sim` 抛异常。**本机复现**：`PYTHONPATH="$PYTHONPATH:<pip_prebundle>"` 时 `/usr/bin/python3 -c "import numpy"` 与 `ros2 run controller_manager spawner --help` 都报 `should not try to import numpy from its source directory`，清掉即正常（系统 numpy 1.21.5 在 `/usr/lib/python3/dist-packages`）。**修法（使用侧，不改代码）**：先 `unset PYTHONPATH PYTHONHOME`（有 conda 则 `conda deactivate`）**再** `source /opt/ros/humble/setup.bash`（顺序不能反，否则会把 ROS 自己的路径也清掉）；launch 那个终端同样要清（它也要起 `joint_state_broadcaster`）。已写入 README §6.2 与 [Gazebo 记录](docs/gazebo_ros2_bringup_2026-09-17.md) 第 6 节；Isaac Lab 训练与 ROS 2 部署建议分用不同终端 |
| 2026-09-17 | Gazebo/ROS 2 链路打通（DEPLOY-05 关闭；用户要看新 PPO 的步态） | **改动**：照参考项目把 Gazebo 链路从仓库内没有的 ROS 1 `liblegged_hw_sim.so` 换成 ROS 2 `gazebo_ros2_control`（本机 Humble 自带 0.4.10）：`xacro/gazebo.xacro` 加 IMU 传感器 + `<ros2_control>`（12 关节 effort 命令 + position/velocity/effort 状态）+ `libgazebo_ros2_control.so`，删掉不可用的 `liblegged_hw_sim` 与 ROS 1 p3d 插件；新增 `imgo2_description/config/robot_control_ros2.yaml`（`joint_state_broadcaster` + 仓库内 `robot_joint_controller/RobotJointControllerGroup`，后者由 `rl_sim` 自己 spawn）；`imgo2_description` 改成标准包（`package.ros1.xml`/`package.ros2.xml` + 由 `build.sh` 替换的 `package.xml` 软链 + ament CMakeLists 装 `config launch meshes urdf xacro mjcf`），并软链进工作区 `src/imgo2_description`（不复制模型）；`core.xacro`/`robot.xacro` 引入 `mesh_prefix`（Gazebo 版用 `package://imgo2_description/meshes/`，纯 URDF 仍 `../meshes/`，重生成后纯 URDF 与原来逐字节相同）；`build.sh` 包扫描改 `find -L src`；`check_model_sync.py` 网格检查同时认两种写法；`launch/gazebo.launch.py` 重写（自己起 gzserver/gzclient、加 `wname`/`gui` 参数、把 URDF 里的 `package://imgo2_description` 换成实际 share 路径并压成单行写到临时文件、用 `-file` spawn、等 spawn 完再起 broadcaster）。**实测两个坑**：① `gazebo_ros2_control` 0.4.x 的 `<parameters>` 只认真实文件路径——给 `package://` 或相对路径都会在插件 `Load()` 抛异常（Gazebo 只打印 `Exception occured in the Load function`）；② URDF 以 `--param robot_description:=<xml>` 交给 controller_manager 时**带换行**会报 `Couldn't parse parameter override rule`，CM 拿不到 URDF、控制器起不来（现象是 spawner 一直 `Could not contact service /controller_manager/list_controllers`）；另外不能用 `gazebo_ros` 自带 launch：本机它起的 gzserver 没加载 `libgazebo_ros_factory.so`（`/spawn_entity` 永不出现），手动 `-s` 起却正常。**验证**：`bash build.sh` 4 个包全过（`rl_sim` 43.8 MB，二进制内 `POLICY_DIR` 指向本仓库 policy，`share/imgo2_description/config/robot_control_ros2.yaml` 就位）；无头单次调用内 `SpawnEntity: Successfully spawned entity`、网格 0 报错、节点 `/gazebo_ros2_control` 与 `/imu_plugin` 出现、`ros2 control list_controllers` → `joint_state_broadcaster ... active`、`/joint_states` 与 `/imu` 发布、`rl_sim` 经 `/joy` 注入 `A`→`RB+DPadUp` 后打印 `RL Controller [ppo]`；`check_model_sync.py`/`check_asset_paths.py`/`check_amp_joint_order.py` 仍全 PASS。**未验证**：GUI 画面与步态观感（本机 `DISPLAY=:1` 建 GL 上下文失败：`X_GLXCreateContext ... BadValue`，gzclient 大概率也渲染不了）、himloco/AMP 在本链路的长时间行为、真机。详见 [Gazebo 链路打通记录](docs/gazebo_ros2_bringup_2026-09-17.md) |
| 2026-09-17 | 键 1（PPO）换成 `imgo2_rough/2026-06-21_16-37-38`（能保持行走并维持高度）；查明「ppo 那份是 AMP 训练产物」不成立 | **用户反馈**：当前 ppo 那份其实是 AMP 训练出来的、应放到 amp 栏，另记得有一份能保持行走并维持高度的训练结果。**来源核对（新方法，只用标准库）**：torch 的 `.pt` 是 zip，成员形如 `<名字>/data.pkl` + `<名字>/data/<key>`，每个 tensor 的原始字节就是一个 storage 成员；按 storage 的 md5 求交集即可给导出验明正身。结果：换之前 `ppo/policy.pt` 的 8 个 storage **8/8 命中 `imgo2_flat/2026-06-21_23-24-09/model_1999.pt`**（该 run `agent.yaml` = `OnPolicyRunner` + `class_name: PPO`，env 里无任何 motion/AMP 项），`amp/policy.pt` 则 8/8 命中 `amp/model_9000.pt`；再用 `strings` 看 state-dict 键名可一眼分开两类：五个 run 只有 `actor/critic/normalizer`，只有 `model_9000.pt` 含 `discriminator`（11.9 MB 对 4.6–5.7 MB）。**⇒ 五个 `rsl_rl` run 全是普通 PPO，全机唯一的 AMP 训练产物是 `model_9000.pt`，它已经在 amp 栏（键 3）。** **五份导出横向实测**（各自 run 的 kp/kd，窗口 12.5 s）：flat(2000it, 20/1.0) 跟速最好但 **z 只有 0.13**；rough 19-58-19(900it, 25/0.5) z 0.30、0.47/0.74 m/s；rough 08-18-30(4400it, 25/1.0) z 0.26、0.54/0.80；**rough 16-37-38(4400it, 20/0.2) z 0.26、0.44/0.89 → 按用户选择用它**；rough 23-23-13(4999it, 20/1.0) z 0.17、0.22/0.58。**执行**：`ppo/policy.pt` ← `imgo2_rough/2026-06-21_16-37-38/exported/policy.pt`（sha256 `618cc4ea4737c343…`），`ppo/config.yaml` 的 `rl_kp/rl_kd` 改为该 run 的 `20/0.2`，其余（45 维观测、`0/0.8/-1.5`、`0.125/0.25`、`clip ±100`、观测缩放）逐项沿用该 run 的 `env.yaml`；amp 栏不动；flat 那份从此不再入库（训练日志 `~/RL/isaac/...` 里保留）。**换后复测**（无头 harness，16 s 仿真）：键 1 `vx=0` 站姿 0.3271、位移 0.023 m；`vx=0.5` z 0.259 + **0.51 m/s**；`vx=1.0` z 0.286 + **1.05 m/s**；FL_thigh 极差 0.88/1.96 rad；键 3 站姿 0.3015 不变；`1→3` 与 `1→2→3` 连续切换均不崩溃。**未运行**：GUI、真机；`kd=0.2` 是训练侧原始值（阻尼偏低），上真机前建议确认 |
| 2026-09-17 | 修复三个策略的运行期故障：MuJoCo 姿态传感器挂错 frame（新增 MODEL-03）；更正 AMP-06／DEPLOY-08 里的两条错误结论 | **现象**：用户反馈三个策略运行都有问题，先查 PPO。**根因（MODEL-03）**：`imgo2_description/mjcf/imgo2.xml` 的 `framequat`/`framelinvel` 写成 `objtype="body"`，返回值会再乘该 body 的惯量主轴旋转 `iquat`（base 的 `fullinertia` 主轴非 `ixx<iyy<izz` 排列，偏移 ≈180° 绕 (1,0,1)/√2），姿态观测整体偏 90°：`GetUp` 结束直立时 `xquat=(1,0,0,0)` 而 `gravity_vec=(-1.000, 0.004, -0.025)`（应为 `(0,0,-1)`），策略把站直判成翻倒；PPO 接管瞬间就输出 ±3.9 的饱和动作并塌成深蹲（`z_final=0.154`）。判据：`QuatRotateInverse(·,(0,0,-1))` 第三分量恒为 `-1-2q_z² ≤ -1`，不可能出现 `-0.025`；再用 `xquat` 与 `sensordata[36..39]` 的 0.25 s 时间线确认是固定偏移；参考项目同位置用的是 site（`base_site`）。**修法**：三个传感器改挂 base 内原有的 `imu` site（site 无 `quat`、`pos` 默认原点，site 系 = link 系），**声明顺序不变**，`sensordata` 偏移（0–11 关节位置／12–23 速度／24–35 力矩／36–39 姿态／40–42 陀螺／43–45 线速度，`nsensordata=46`）与 `rl_sim_mujoco.cpp` 的读取处都不动；顺手注释掉文件末尾 `base z=0.35` 的陈旧 `<keyframe>`（C++ 不应用，但 MuJoCo `simulate` GUI 的 Key 下拉框会应用）。**验证**（无头 harness：复用真实 `RL`/FSM/`librl_sdk`/libtorch/MuJoCo 与 `policy/`，只重写 `GetState`/`SetCommand`/`RunModelStep`/`Forward`，按键序列与 GUI 相同）：修后三个键 `gravity_vec` 均为 `(0,-0,-1)`；PPO 站姿 `z_final=0.3212`（修前 0.154）、`vx=0.5/1.0` 实测 0.509/0.975 m/s（跟踪误差 <2%）、FL_thigh 极差 1.1–1.4 rad（真步态非滑行）；AMP 站姿 0.3015（修前 0.2663）但 `vx=0.5` 时 10.5 s 只走 0.076 m；himloco（Go2 占位）侧倾仍不可用；`0→1→2→3` 连续切换不再崩溃。**更正两条旧记录**：① AMP-06 里"用同一 harness 跑参考 45 维 `policy.pt` 结果四位小数完全相同"无效——`POLICY_DIR` 是编译期 `-D` 烘进 `librl_sdk.a` 的，当时换目录并没真正换到策略；本轮改为把 `rl_sdk.cpp` 编进 harness 并用 `-DPOLICY_DIR=<临时目录>` 才换到。② DEPLOY-08 那条 `1x45 vs 270x128` 崩溃是我第一版 harness 的 `Forward()` 漏掉历史分支（`history_obs_buf.insert`/`get_obs_vec`）造成的假象，逐行照搬真实 `RL_Sim::Forward()` 后单键与连续切换都不崩，真实 deploy 代码无此缺陷。**对照实验**：参考 `base_move/policy.pt` 在我们模型上站 0.2887、`vx=0.5` 走 0.454 m/s，在参考自己的 MJCF 上反而站不起来（0.0771；其 base 仍是 6.53394、mesh 碰撞体、没写 `timestep`），支持"物理参数取训练侧"。**同步更新**：README §5.4／§6.0／§6.1／问题表，新增 [策略运行期排查](docs/sim2sim_policy_runtime_2026-09-17.md)；`check_model_sync.py`／`check_asset_paths.py`／`check_amp_joint_order.py` 仍全 PASS。**未运行**：GUI（本机无显示与 NVIDIA 驱动）、Isaac Lab 训练/回放、ROS/Gazebo、真机 |
| 2026-09-17 | 部署侧接入 PPO，按键改为 1=PPO / 2=himloco / 3=AMP | **做了什么**：`policy/imgo2/ppo/` 新增 `policy.pt`（拷自 `~/RL/isaac/Imgo2_rl/logs/rsl_rl/imgo2_flat/2026-06-21_23-24-09/exported/policy.pt`，sha256 `7cbb4def63c361db…`）与 `config.yaml`；`config.yaml` 的每个值都抄自该 run 的 `params/env.yaml`：45 维观测（`base_lin_vel`/`height_scan` 均为 null，顺序 ang_vel/gravity_vec/commands/dof_pos/dof_vel/actions）、`rl_kp 20`/`rl_kd 1.0`、`default_dof_pos 0/0.8/-1.5`、`action_scale 0.125/0.25`、`clip ±100`、观测 scale `0.25/1.0/1.0/1.0/0.05/1.0`、`joint_mapping` 恒等；`fsm_imgo2.hpp` 新增 `RLFSMStatePPOLocomotion`（config `ppo`）占用键 1（手柄 `RB+DPadUp`），himloco 从键 1 挪到键 2（`RB+DPadRight`），AMP 从键 2 挪到键 3（`RB+DPadDown`），Passive/GetUp/三个 locomotion 的 CheckChange 与工厂一并同步；`.gitignore` 的策略白名单由 `amp/policy.pt` 泛化为 `imgo2_deploy/policy/imgo2/**/policy.pt`。**验证**：`bash build.sh -mj` 通过；用仓库外临时 harness 单独按 1/2/3 各跑 9 s——三者都成功进入（`entered=1`，`calls=274`），PPO 读到 `rl_kp=20 default=0 0.8 -1.5`、himloco 的 6 帧历史 `co=45 → hist_obs=270` 正确、AMP 结束高度 0.2663，无 NaN 无崩溃。**未通过**：一次多键连续切换（0→1→2→3）出现 `1x45 vs 270x128` 崩溃，单键路径复现不了，登记为 DEPLOY-08（待 GUI 复现）。**未运行**：GUI、训练、ROS、真机 |
| 2026-09-17 | 推送打通：配置 SSH key、`origin` 改为 SSH、7 个本地提交推上 `main` | 用户配置 `~/.ssh/id_ed25519` 并加到 GitHub 后，`ssh -T git@github.com` 返回 `Hi qmq-h! You've successfully authenticated`；`origin` 由 HTTPS 改为 `git@github.com:qmq-h/imgo2.git`；`git push origin main` 成功把 `271edd2..72e3193`（7 个提交）推上去，校验本地与远程 HEAD 同为 `72e3193b18e7bf3484aac01a371fcbdd106ddca2`，`git log origin/main..HEAD` 为空。此前失败的原因是**凭据**而非网络：本机无 credential helper / `~/.git-credentials` / `gh` / token，harness 的 shell 没有 TTY 无法交互输入（`user.name`/`user.email` 只是提交署名，不参与认证；公共仓库匿名 `ls-remote` 能成不代表 `push` 能成）。事实已写入 AGENTS.md「多机与同步」。**未做**：训练服务器尚未拉取 |
| 2026-09-17 | 模型统一阶段②④⑤：切换全部消费者、删除三处镜像副本、改写 AGENTS 模型章节 | **消费者**：`assets/imgo2.py` 新增 `_REPO_ROOT`（`parents[5]`）并把 `_DEFAULT_URDF_PATH` 指向 `<repo>/imgo2_description/urdf/imgo2.urdf`；`check_asset_paths.py` 重写为「按声明解析任意 root 变量」（旧版只认 `_PROJECT_ROOT`，且会复制预期路径）；`audit_amp_dataset.py`/`tests/test_amp_alignment.py`(2 处)/`inertia_urdf.py` 同步；deploy 新增编译期 `IMGO2_MODEL_DIR="${PROJECT_ROOT_DIR}/../imgo2_description"`，`rl_sim_mujoco.cpp` 场景路径改为 `IMGO2_MODEL_DIR "/mjcf/" + scene`；`build.sh` 的 `setup_robot_descriptions` 与两个 Gazebo launch 改指 description（`imgo2.gazebo.urdf`）；`CMakeLists` 安装列表去掉 `robot_description`。**删除**（git 可追溯）：`imgo2_model/`（21 文件 27 MB）、`imgo2_rl/source/imgo2_rl/data/`（12 文件 15 MB）、`imgo2_deploy/robot_description/`（13 文件 15 MB）；删后全仓只剩 2 份 URDF、1 套网格、1 套 MJCF，工作树由约 126 MB 降到 69 MB。**check_model_sync.py 改造**：登记项改为 description 的 core+gazebo 两份，新增网格指纹断言（`8dc5b5995a11`/10 文件）与「每个 mesh 引用都存在」检查，去掉 `DESC_TO_TRAIN` 与镜像/fragment 提示。**AGENTS.md**：模型章节由「四份副本不要合并」改写为「唯一源 + 生成物勿手改 + 改完跑检查」，并更新目录命名清单。**验证**：`check_model_sync.py` 全 PASS；`check_amp_joint_order.py` 三项全 PASS（数值与统一前一致）；`check_asset_paths.py` PASS（新增负向测试：把 URDF 文件名改错 → FAIL，恢复 → PASS）；单元测试 5 通过 + 1 跳过（跳过的是需要 torch 的 AMP-03 回归）；`bash build.sh -mj` 删目录后仍构建成功且二进制内 `IMGO2_MODEL_DIR` 指向 `<repo>/imgo2_description`；`git ls-files -i -c --exclude-standard` 为空。**未运行**：GUI、Isaac Lab 训练/回放、ROS/Gazebo、真机 |
| 2026-09-17 | 阶段③回归：跑通单元测试与三项离线检查（**更正 2026-09-17 晚**：当时记的「AMP-03 的 Torch 回归首次真正执行并通过」不成立，本机根本没有 torch） | **环境**：本机 Linux `~/miniconda3/envs/isaaclab`（Python 3.11.14、CPU；**该环境没有 torch，全机五个 conda 环境与系统 python 都没有**）。`python -m unittest discover -s tests -p test_amp_alignment.py` → **5 项通过 + 1 项跳过**，跳过的是 `test_raw_statistics_and_normalized_gradient_penalty`，原因就是缺 torch/numpy（`skipped "CPU regression needs the training environment's torch and numpy"`），所以 AMP-03 的 Torch 回归**至今没有任何一次真实执行记录**（Windows 侧同样跳过）。`check_asset_paths.py` / `check_model_sync.py` / `check_amp_joint_order.py` 三者 exit=0。全仓 79 个 URDF/xacro/XML 只有 1 个解析失败，即已知的 `imgo2_model/` 腿部件片段（无 `<robot>` 开头）。**限制**：仍未在 Isaac Lab 里构造环境、未训练 |
| 2026-09-17 | 模型统一阶段③：MuJoCo 场景归位到 `imgo2_description/mjcf/` | **做了什么**：把 `imgo2_deploy/robot_description/imgo2_mjcf/scene.xml`（训练物理 + 参考项目求解器/接触块的那份）拆成 `imgo2_description/mjcf/imgo2.xml`（机器人本体：compiler/option/asset/worldbody(base)/actuator/sensor/keyframe）与 `scene.xml`（世界：statistic/visual/纹理/灯光/地面，`<include file="imgo2.xml"/>`），网格引用由 `../imgo2_urdf/meshes/` 改为 `../meshes/`（10 个引用全部命中）；追加参考项目有而我们缺的 `framelinvel`（adr 43，在 36-42 之后，不影响 C++ 现有偏移），为 AMP-06 的 `lin_vel` 留出通路。旧参考原版 `imgo2.xml`（base 惯量旧值 6.53394 + 简化碰撞体）被取代。**验证**：`mj_loadXML` 新旧一致（nq=19 nv=18 nu=12 nsensordata 43→46，默认 qpos base_z=0.5000 足端 +0.0743）；同一 harness 物理回放 16 s 结果**逐位相同**（`z_final=0.2621`、AMP 后最高 `0.2907`、`z_min=0.1132`）。**未做**：deploy 仍读自己的 `robot_description/imgo2_mjcf/scene.xml`，消费者切换与删镜像见 MODEL-02 ②④；未运行 GUI/训练/ROS/真机 |
| 2026-09-17 | 模型统一阶段①：`imgo2_description` 命名与网格统一、xacro 模块化、生成物入库 | **做了什么**：`meshes/` 改名 LF→FL（6 个文件，内容逐字节不变，四份网格指纹统一为 `8dc5b5995a11`）；新增 `xacro/core.xacro`（由训练 URDF 抽出 17 link/16 joint，含 `<robot>` 包裹以便 include）、`xacro/transmission.xacro`（修正旧宏 `params="name"` 却用 `${prefix}` 的错误）、`xacro/gazebo.xacro`（2 插件 + 17 per-link 接触块，link 名改 FL）、`xacro/imu.xacro`，并重写 `xacro/robot.xacro` 为组装入口（开关 `transmission`/`gazebo`/`imu`，默认全 false）；生成并入库 `urdf/imgo2.urdf`（17 link/16 joint/无附加）与 `urdf/imgo2.gazebo.urdf`（+12 transmission +21 gazebo +base_imu）；删除 `urdf/imgo2_description.urdf` 与 `xacro/common/`（5 个文件，均被取代或本就是死代码，`leg.xacro` 还引用仓库外包）；`empty_world.launch` 改为传三个开关、`CMakeLists.txt` 安装目录补 `xacro mjcf`；`check_model_sync.py` 登记两份生成物（Gazebo 份允许 `base_imu`）、网格集纳入 `imgo2_description/meshes`；`check_amp_joint_order.py` 默认路径与 FK 段随统一调整。**验证**：`check_model_sync.py` 五项全 PASS（4 份已登记 URDF，FK 恒等 RMSE 0.00214 m / 交换 0.22470 m）；`check_amp_joint_order.py` 三项全 PASS 且数值与统一前完全一致（0.00107/0.00214 m、0.22491、0.0191/0.1339 rad、r 0.9546/0.2317，新 hash `c4f14273959e94c1`）；生成物与训练 URDF **结构等价**（links/joints/轴/限位/origin/惯量/碰撞逐项相同）；`xacro` 重跑两次与入库文件逐字节一致；负向测试（把首个 hip 轴改成 `-1 0 0`）触发 2 条 FAIL（关节轴不同 + FK 退化到 0.02910 m），恢复后 SHA256 不变。**未做**：消费者切换、MJCF 归位、删除镜像副本（见 MODEL-02）；未运行训练/GUI/ROS/真机 |
| 2026-09-17 | 落库两次提交（`55f5cde`、`0ca1af3`）；记录本机（Linux）推送环境；启动模型统一 | ① `55f5cde` AMP sim2sim 对齐 + 场景修复 + 文档（11 文件）；② `0ca1af3` 模型统一 WIP：`imgo2_description/meshes` 改名为 FL/FR/RL/RR + `L_shank/R_shank`（内容逐字节不变），新增 `xacro/core.xacro`（由训练 URDF 抽出 17 link/16 joint）、`xacro/gazebo.xacro`（2 插件 + 17 per-link 接触块，已改 FL 命名）、`xacro/imu.xacro`；`git status` 干净。**推送阻塞**：本机环境变量指向的代理 `127.0.0.1:7897` 在监听但 TLS 握手失败，`7890`/`10808` 未监听；清空代理后直连可通（`curl --noproxy '*' https://github.com` = 200），但本机无 credential helper、`~/.git-credentials`、`gh`、token，`git push` 报 `could not read Username`，需用户提供凭据；已写入 AGENTS.md「多机与同步」。**中间态**：`imgo2_description/urdf/*.urdf` 与 `mjcf` 仍按旧命名/旧网格名，description 链路暂不可加载；训练链与部署链未动 |
| 2026-09-17 | 修「一启动就飞」；按参考项目对齐 MuJoCo 场景；完成模型副本对照 | **根因**：`mujoco_utils.hpp` 的 `PhysicsThread` 只做 `LoadModel → mj_makeData → mj_forward`，**不套 keyframe**，用的是模型默认 qpos（关节全 0 直腿）；我们原场景 `base pos="0 0 0.35"` 直腿时足端 z=−0.0758（穿地）→ 弹飞（实测 `z_max=1.3623`）。**撤销一次错误改法**：给关节加 `<joint ref>` 只改 `qpos0`，而 MuJoCo 关节运动学按 `qpos−ref` 计，等于把姿态抵消回直腿，仍穿地。**最终按参考项目改场景参数**：`base pos→0 0 0.5`（足端 +0.0743，与参考数值相同）、`<option cone="elliptic" impratio="100">`、关节 `damping="1" armature="0.1"`、碰撞 `condim="3" solref="0.005 1" friction="1 0.01 0.01"`、足端 `friction="0.4 0.02 0.01"`；`timestep` 保持 0.005（参考未设=2 ms，未跟抄）。**验证**（GUI 等价启动、不套 keyframe、16 s）：`z_max=0.5000` 不飞，AMP 后最高 0.2907、结束 0.2621，**能站起**；改前为躺地 0.0754。**模型对照**：参考 `imgo2_description.urdf` 与训练 URDF 的 17 个公共 link 质量/质心/惯量/碰撞体数全同、同名关节 axis/limit/父子 0 处不同，仅多 `imu_link`/`imu_joint` 且用 `FL/FR/RL/RR` 命名；参考与本仓库的 `mjcf/imgo2.xml`、`scene.xml` 逐字节相同，但该 MJCF 在本仓库**缺 6/10 网格**无法加载，且 base 惯量仍是旧值 6.53394。**未运行**：GUI、Isaac Lab 回放、ROS、真机；速度指令仍未跟踪（`dx=−0.23` 对参考 `+0.10`） |
| 2026-09-17 | AMP sim2sim 对齐：把 6 个部署运行逻辑文件还原到 `271edd2`；FSM 只增量式新增一个 `RLFSMStateAMPLocomotion` 类 + 按键 `2` + 工厂注册；按训练侧/参考项目改 `base.yaml`、重写 `amp/config.yaml`；删除 deploy 侧导出与校验脚手架；补 `.gitignore` 两条 | **已构建**：`bash build.sh -mj` 退出码 0（复用参考项目 `library/` 的 libtorch 2.3.0 + ONNX Runtime 1.22.0 + MuJoCo 3.2.7）。**已核对**：`policy.pt` 与 `model_9000.pt` 的 actor 在 128 组随机 48 维输入上最大误差 0；`scene.xml` 12 个关节的 axis/range 与训练 URDF 逐项相同；场景传感器偏移（0-11 关节位置、12-23 关节速度、24-35 关节力矩、36-39 四元数、40-42 陀螺）正好落在 `GetState` 读取处；打印观测确认 `commands` 正确、`lin_vel` 恒 0。**仿真**：仓库外临时 harness（复用已编译的 `librl_sdk.a` 与真实 FSM/策略，仅重写 `GetState/SetCommand/RunModelStep`）跑 16 s，`0`→起身→`2` 成功进入 AMP，无 NaN，基座高度 ~0.2455 m。**未通过**：策略只给出静态下蹲、`vx_mean≈0.0000` 不跟踪速度指令；用同一 harness 跑参考项目 45 维 `policy.pt` 结果四位小数完全相同，故非本轮映射问题，登记为 AMP-06。**未运行**：GUI 入口（本机无可用 NVIDIA 驱动，`glfw` 建窗失败）、Isaac Lab 回放、ROS/Gazebo、真机 |
| 2026-09-15 | 首次建立项目总览、代码导航、训练回放与部署流程、接口参数、问题表和维护模板；增加根目录维护约定 | 核对任务注册与关键脚本；解析 21 份动作数据，共 5097 帧、61 列、0.02 秒间隔；逐文件确认对应数据副本哈希一致；三份 URDF 哈希不同；确认 MuJoCo 场景与 SDK 缺失。未运行训练/回放/编译/真机。当前 shell 的 `python` 命令不可用，数据检查由 PowerShell 完成 |
| 2026-09-15 | 根据用户反馈确认 PPO 已训练及 sim 验证，记录 AMP 贴地爬行；对照两个本地 AMP 项目，修正数据映射和归一化处理，加入高度约束、基座触地终止、任务奖励量级调整及日志 | 21 份参考动作与训练 URDF 运动学对照通过；4 项离线测试通过，1 项 Torch 更新测试因当前解释器缺少 torch/numpy 跳过；6 个变更 Python 文件语法检查通过。找到可用 Python 3.14 解释器，但未进行新训练或仿真验证 |
| 2026-09-15 | 根据用户补充追踪实际采集模型和脚本，区分 URDF 声明顺序与按名称重排后的数据顺序，收窄映射结论的适用范围 | 仅阅读源码与模型结构；确认脚本默认 LF/RF/LH/RH，采集 URDF 声明 LF/LH/RF/RH；实际采集参数尚缺。本轮未改映射、未运行测试或仿真 |
| 2026-09-15 | 用户确认实际采集使用默认关节与足端参数，闭合数据顺序来源 | 关节位置/速度、足端位置/速度均按 LF/RF/LH/RH 写入；确认当前恒等映射适用。未追加本机验证 |
| 2026-09-15 | 对恒等映射补做不依赖采集参数回忆的独立验证，并核对默认姿态与数据的关节角数值 | 改用实际采集模型 `imgo2_description.urdf` 复算 FK（21 动作/5097 帧，恒等 RMSE 均值 0.00107 m、最大 0.00214 m；LF/LH/RF/RH 声明顺序配对 0.22491 m），与训练 URDF 数值相同；髋外展左右对称性残差 0.0191 rad 对 0.1339 rad；关节位置块与速度块同块配对平均 r 0.95、交换后 0.76；重跑 `tests/test_amp_alignment.py` 为 4 通过 1 跳过；确认 `python` 别名退出码 9009，可用解释器为 `C:\Users\qmq\AppData\Local\Python\pythoncore-3.14-64\python.exe`。未运行 Isaac Lab、未重训、未验证贴地爬行原因 |
| 2026-09-15 | 文档整理：删除过期与草稿文档，收编历史 todo，补齐文档入口并关闭 DOC-01 | 删除 4 份内容相同的 URDF 目录质量草稿 `msg.md`（哈希 `1F3118F101A6FE78`，无脚本引用）和已过期的 `imgo2_rl/tree.md`（其中的 `manager_based/velocity_env_cfg.py`、`config/imgo2_base_move/` 路径已不存在）；`imgo2_rl/todo.md`、`imgo2_deploy/todo.md` 要点并入本表 AMP-01／AMP-02 与部署子项目说明后删除；目录表补入 `docs/` 和 `rlfromgym2lab.md`，删除实际不存在的 `imgo2_rl.rar` 行。工作区 `.git` 是空目录，删除无版本控制可恢复 |
| 2026-09-15 | 补做文档链接与编码核对；把训练前置检查细化为「先同步代码」并列出清单；约定多机同步规则 | Python 复核 8 份文档共 33 个相对链接全部可解析，文件均为 UTF-8 无 BOM、LF 换行、无异常控制字节（此前 PowerShell 的行数与链接计数偏小，是读取方式的假象，不代表文件有问题）；确认 `imgo2_rl` 多数源码带同一批拷贝时间戳，不能据 mtime 判断新旧，本次修复按内容确认为 9 个文件而非 6 个；确认 harness 只在 Host 文件系统上读写、不做跨机同步，规则写入 AGENTS.md。未运行训练或仿真 |
| 2026-09-15 | 记录 9 个修复文件的 SHA256 供同步后核对；把跨机交流约定写进 AGENTS.md | 修复文件哈希见 [AMP 对齐与参考项目对照](docs/amp_alignment_review.md) 7.1 节（如 `amp_env_cfg.py` = `cbc890318c5f`、`amp_events.py` = `3b529139161a`）。约定：跨机只靠文件系统传递，`AGENTS.md` 传规则、README 与 `docs/` 传状态、维护记录表充当消息日志，同一时间只有一端写同一份文档。未做实际跨机同步，也未运行训练 |
| 2026-09-15 | 建立 git 仓库：合并三个嵌套仓库为单一 monorepo，加入 `.gitattributes` 与根 `.gitignore`，首次提交并推送到 GitHub | 合并前 `imgo2_model/`（4 提交）与 `imgo2_rl/`（2 提交）各有独立 `.git` 且无 remote，`imgo2_deploy/.git` 为空目录；历史已导出为 bundle 存于 `.git-backups/`（已 `git bundle verify` 并实际 clone 回验，4／2 提交均可恢复），该目录被忽略、不入库。原先直接 `git add` 会把这两个子树当作 gitlink，训练代码根本不会进仓库。合并时按 `imgo2_rl` 仓库的 `git diff` 修正了改动集：实为 12 个文件，此前文档只列 9 个；并发现原映射是 4 元 `amp_leg_mapping = [0, 2, 1, 3]`（按腿轴索引），不是文档原先写的 12 元列表。另修正 `imgo2_rl/.gitignore` 中 `tests/` 与 `/datasets/` 两条会漏掉修复文件和 AMP 数据的规则。未运行训练或仿真 |
| 2026-09-15 | 首次提交 `f6acff9` 已推送到 `https://github.com/qmq-h/imgo2_rl`（分支 `main`） | 远程 `refs/heads/main` 与本地 HEAD 同为 `f6acff9`，共 314 个文件，`main` 已跟踪 `origin/main`。推送需要临时覆盖三处本机配置：`~/.gitconfig` 的 `http.proxy`／`https.proxy` 指向 `127.0.0.1:10808`，而该端口没有进程在监听；系统 `gitconfig` 的 `http.sslBackend=schannel` 在受限 shell 下报 `SEC_E_NO_CREDENTIALS`。实际生效命令为 `git -c http.sslBackend=openssl -c http.proxy= -c https.proxy= push`。另：受限 shell 下 git 凭据助手无法创建命名管道（`Win32 error 5`），推送需放宽沙箱。未改动用户的全局 git 配置 |
| 2026-09-15 | 补上 checkpoint 忽略规则（提交 `653b661`） | 原根 `.gitignore` 只排除 `*.pth`／`*.ckpt`／`*.onnx`／`*.jit`，未排除 `*.pt`，与其自身注释「训练 checkpoint 体积大且可再生成，不入库」矛盾：`model_*.pt` 只要落在 `logs/` 之外就会被提交。现排除 `*.pt` 并用否定规则保留 `imgo2_deploy/policy/imgo2/himloco/himloco.pt`（不足 1 MB，部署配置引用）。以 `git check-ignore` 退出码验证：该占位策略退出 1（未被忽略），其余 `.pt` 退出 0（被忽略） |
| 2026-09-15 | 记录 monorepo 对 ENV-01 硬编码路径的影响 | 仓库结构变为「根 = 工作区、`imgo2_rl/` 为子目录」后，`assets/imgo2.py` 中假定 `imgo2_rl` 为项目根的两条绝对路径不再自然成立；已在问题表 ENV-01 写明失效条件与两种解法。本轮只更新文档，未改代码、未在服务器上验证 |
| 2026-09-15 | 资源路径改为由文件位置推导（去掉机器绝对路径），并补齐模型副本比较结论 | `assets/imgo2.py` 改为 `Path(__file__)` 上溯 4 层得 `imgo2_rl/` 项目根，支持 `IMGO2_AMP_MOTION_DIR`／`IMGO2_URDF_PATH` 覆盖；`assets/amp_motions.py` 一并去掉 3 条机器路径（含一条 Windows 个人目录）。新增 `scripts/tools/check_asset_paths.py`（纯标准库）：本机运行 PASS——URDF 存在、动作文件 21 份、无残留机器路径；用不存在的 `IMGO2_AMP_MOTION_DIR` 覆盖时正确返回 FAIL（计数 0≠21）。ENV-01 状态随之改为「代码已修正，待服务器验证」。另完成 MODEL-01 要求的逐项比较：部署份 12 个腿部关节 axis 与训练份全部相反、基座质量相差 2.0 kg、网格 10 个文件逐字节相同，故两份 URDF 属语义差异而非冗余副本，暂不合并。未运行 Isaac Lab |
| 2026-09-15 | 明确 `imgo2_description` 的腿部顺序差异是有意设计 | 该副本腿序 LF/LH/RF/RH 与训练侧不同，服务另一处实现；已在 README 目录表与 AGENTS.md 写明不要为「统一」而改写或合并它。本轮只更新文档 |
| 2026-09-15 | 修正推送命令：本机代理实际监听 7890，直连推送会被重置（提交 `d310d4a` 已推送） | 推送时依次排除三种失败：`Recv failure: Connection was reset`（直连，批量上传被重置）、`Failed to connect ... 127.0.0.1:10808`（全局配置的代理端口没监听）、`SEC_E_NO_CREDENTIALS`（schannel）。实测端口探测发现代理客户端监听 **7890**，且 `~/.gitconfig` 把 `https.proxy` 写成 `https://` 也是错的。可用命令：`git -c http.sslBackend=openssl -c http.proxy=http://127.0.0.1:7890 -c https.proxy=http://127.0.0.1:7890 push origin main`。已写入 AGENTS.md，未改动全局配置 |
| 2026-09-15 | 仓库改名为 `imgo2`，本地 remote 与文档同步更新 | 经 GitHub API 核对：仓库 `id` 仍为 `1371418917`（是改名而非新建），`full_name` 由 `qmq-h/imgo2_rl` 变为 `qmq-h/imgo2`，`default_branch` 仍为 `main`，`pushed_at` 保持 `12:53:21Z` 说明 7 个提交完整保留；旧地址 `imgo2_rl` 由 GitHub 重定向。本地已 `git remote set-url origin https://github.com/qmq-h/imgo2.git` 并用 `ls-remote` 验证仍指向 `6e9e2e0`。此前记录的「仓库名与内容不匹配」（博客仓库名 `imgo2_rl` 装的是整个工作区）随之消除 |
| 2026-09-15 | 更正模型副本描述：`imgo2_model/` 的 URDF 是腿部件片段；网格共三套而非一套 | 用户问及关节轴与基座质量的差异含义时复算发现：`imgo2_model/imgo2_urdf/urdf/imgo2.urdf`（523 行）只有 16 link／16 joint，全是四条腿的部件，**无 `base` link、无 `<robot>` 起始标签**（只有尾部 `</robot>`），XML 解析报 `junk after document element`，因此不能作为基座质量的对照基准；但它的腿部关节轴（`FL_hip_joint` = `1 0 0`）与训练份一致，说明部署份是唯一反向者。按内容指纹比对五个网格目录：`imgo2_model/imgo2_urdf`、`imgo2_rl/source/.../data/`、`imgo2_deploy/...` **三份共用同一套**（`bc421eb1cbef`），`imgo2_model/imgo2_mjcf`（`1ac2f43d08b3`）与 `imgo2_description`（`24b1aa343fb9`）各为独立命名体系。此前的「四份 mesh 全同」说法有误，已在 README 目录表、MODEL-01 与 AGENTS.md 一并更正 |
| 2026-09-15 | 查清 `imgo2_model/` 的实际内容、deploy 链路是否补偿符号、以及基座质量的真实分布；更正一处错误数字 | `imgo2_model/` 只有模型素材：腿部件片段 URDF（无 `base`、无 `<robot>`）+ 两套网格（`imgo2_urdf/meshes` 14.8 MB、`imgo2_mjcf/meshes` 11.6 MB），无任何 XML/launch/xacro/yaml/py。逐条核对 deploy 三条推理链路源码：`rl_real_imgo2.cpp`、`rl_sim.cpp`、`rl_sim_mujoco.cpp` 均无符号取反，只有 `joint_mapping` 索引置换；`robot_control.yaml` 只做控制器到关节名映射——即没有任何地方补偿 URDF 的反向，读该 URDF 的只有 Gazebo/ROS 链路。按 link 统计质量：差异只在 `base` 一个 link（其余 16 个完全一致），四份取值 `3.53394020`／`5.53394020`／`6.5339402`（`imgo2_model/` 片段无 base），每次差整 1.0 kg，总质量对应 10.69955／12.69955／13.69955 kg；采集模型 `imgo2_description.urdf` 与训练份同为 5.53394020。另更正 README 原先「占总重约 9.5 kg」的错误：实测总质量为 12.69955（训练）／10.69955（部署）。本轮只改文档，未改任何 URDF 或代码 |
| 2026-09-15 | 修正部署份 URDF 的关节符号、统一 base 惯量；新增模型一致性检查 | 部署份 12 个腿部关节 axis 原本全部取反且限位镜像，改为训练份约定（24 行：12 axis + 8 限位 + base 惯量块）。FK 验证：修正前对录制数据恒等 RMSE 0.057–0.118 m，修正后 0.0001–0.0016 m，与训练份一致。base 质量按确认真值 `5.53394020` 统一四份；`imgo2_description/urdf/imgo2.urdf` 原为 `6.53394020` 却带 3.5339 那份惯量（质量与惯量不自洽，比值 1.566 恰为 5.5339/3.5339），已改为规范块。核实 `imgo2_description` 两个 URDF 的符号与限位本来即与训练份一致，故录制数据无问题。新增 `scripts/tools/check_model_sync.py`（纯标准库）：按逻辑关节名比对四份 URDF 的轴/限位、base 惯量、三份共用网格，并对每份做 FK；本机全通过，并以「故意翻回符号」的负向测试确认可报错（12 关节 + FK 报 FAIL，恢复后 sha256 不变）。另发现 `imgo2_description/xacro/common/leg.xacro` 与 `urdf/imgo2.urdf` 全工作区无人引用、`imgo2_description/mjcf/scene.xml` 是 DEPLOY-02 的线索。未运行 Isaac Lab 或 Gazebo |
| 2026-09-15 | 查明 rl 与 deploy 各自读取哪一份 URDF，以及 ROS 侧会用 URDF 限位 clamp 指令 | RL 运行时唯一入口是 `assets/imgo2.py` 的 `IMGO2_CFG.spawn.asset_path`（默认 `imgo2_rl/source/imgo2_rl/data/imgo2_model/imgo2_urdf/urdf/imgo2.urdf`，可用 `IMGO2_URDF_PATH` 覆盖）。deploy 三条推理链路都不读 URDF：`rl_real_imgo2.cpp` 走 Unitree SDK、`rl_sim_mujoco.cpp` 读 `robot_description/<robot>_mjcf/<scene>.xml`（MJCF，当前缺失）、`rl_sim.cpp` 走 ROS 控制器；真正读 URDF 的是 `gazebo.launch`（`cat $(arg robot_urdf)` 塞进 `robot_description`）、`gazebo.launch.py`（package_share 路径）与 `build.sh`（仅存在性检查），指向 `imgo2_deploy/robot_description/imgo2_urdf/urdf/imgo2.urdf`。**关键补充**：`robot_joint_controller` 用 `urdf.initParamWithNodeHandle("robot_description")` 解析该 URDF，并用其限位 clamp 位置/速度/力矩；修正前旧限位会让 `default_dof_pos` 的 shank `-1.50` 被 clamp 成 `+0.733`。另注 `scripts/tools/inertia_urdf.py` 读的是 `imgo2_model/` 那个腿部件片段（工具脚本，非训练路径）。已写入 MODEL-01 |
| 2026-09-15 | 复核「两份 URDF 是否已等价」：仍有 22 处物理差异，部署份是唯一异类 | 按 link/字段逐个比对训练份与部署份，差异 22 处，全在足端与后腿：足端 CoM 与惯量（训练各向同性 `9.6e-06`，部署各向异性 `1.175e-6/1.2778e-5/1.189e-5`）、后腿 shank/thigh 惯量非对角项符号、足端碰撞圆柱长度（`0.01` 对 `0.02`），另有材质块与尾随空格。用五份模型溯源：`imgo2_model/` 片段、`imgo2_description` 两个 URDF、训练份在除圆柱长度外的各项上**四份一致**，部署份单独偏离——与符号问题同一模式，因此应改部署份。足端碰撞圆柱长度为 2 对 3（训练份+片段 `0.01`，采集模型两个 URDF+部署份 `0.02`），列为待用户定夺项。结论写入 MODEL-01。本轮只改文档，未改 URDF |
| 2026-09-15 | 按用户决定把四份模型的物理参数全部对齐到训练侧；扩展一致性检查覆盖逐 link 物理量 | 用户决定「都使用训练侧的参数」。据此修改：`imgo2_deploy` 的足端 CoM/惯量与后腿 shank/thigh 惯量非对角项（五份模型中的唯一异类）改为训练侧，共替换 8 个 inertial 块；四份 URDF 的足端碰撞圆柱由 2 对 3 统一为训练侧的 `0.01`；`imgo2_deploy` 的质量值尾随空格清理。结果：四份完整 URDF 按逻辑 link 名比对的物理指纹完全一致（`aae81e7301f5`），`imgo2_deploy` 与训练份现仅差 4 个不影响物理的 `<material>` 块。`check_model_sync.py` 增加第 2 项「逐 link 质量/质心/惯量/碰撞几何」检查并重新编号；本机五项全通过，并新增第 2 次负向测试（单独改足端 CoM）确认能报错、恢复后 sha256 不变。未运行 Isaac Lab 或 Gazebo |
| 2026-09-15 | 整合为单一项目：统一顶层命名、消除子项目痕迹、合并 .gitignore | 按用户决定做「元数据 + 命名整合」。① 顶层目录统一小写下划线：`Imgo2`→`imgo2_model`、`Imgo2_rl`→`imgo2_rl`、`Imgo2_deploy`→`imgo2_deploy`（`git mv`，case-only 的分两步走；264 个文件按重命名记录，历史保留）；② 删除两个子项目 README，把部署包结构与「关节名用 `*_shank_joint` 而非 Go2 的 `*_calf_joint`」并入根 README 新增的 6.0 节，训练侧内容根 README 本已覆盖；③ 删除两个嵌套 `.gitignore`，其中只对子树成立的规则加路径前缀后并入根 `.gitignore`（`imgo2_rl` 37 条、`imgo2_deploy` 18 条）；④ 删除 `imgo2_deploy/VERSION`（4.0.0，无脚本读取），**保留 `imgo2_deploy/LICENSE`**（上游 rl_sar 的 Apache-2.0 归属，已在 6.0 节注明）。**验证**：全仓库文本文件中旧目录名残留 0 处；原先被忽略的 33 个路径合并后仍全部被忽略；`git ls-files -i -c --exclude-standard` 为空；三项离线检查 + 5 项单元测试全通过；`docs/amp_data_audit.json` 已重新生成，去掉内嵌的旧绝对路径。未改名 `imgo2_rl/source/imgo2_rl/data/imgo2_model/`（RL 内部模型副本目录，仍为大写）。未运行 Isaac Lab 或 Gazebo |
| 2026-09-15 | 清掉最后两个大写目录，全仓库目录名统一为小写下划线 | 承接上一轮的顶层命名整合，把仓库内最后两个含大写字母的目录也改掉：`imgo2_model/Imgo2_urdf`→`imgo2_model/imgo2_urdf`，以及 RL 包内的 `imgo2_rl/source/imgo2_rl/data/Imgo2/Imgo2_urdf`→`.../data/imgo2_model/imgo2_urdf`（case-only 改名分两步走）。引用同步更新：`data/Imgo2/` 8 处、JSON 内 Windows 转义形式 1 处、`Imgo2_urdf` 20 处。**验证**：按 `git ls-files` 统计的 108 个目录中，含大写字母的由 7 个降为 **0 个**；`check_asset_paths` 打印的 URDF 路径已落在新目录且存在；三项离线检查 + 5 项单元测试全通过；`docs/amp_data_audit.json` 重新生成（21 文件 / 5097 帧不变）。残留的大写 `Imgo2` 仅为项目名、任务 ID、类名与 URDF 的 `<robot name="Imgo2">`，以及本机工作区目录 `Desktop\Imgo2`。未运行 Isaac Lab 或 Gazebo |
| 2026-09-15 | 代码复审（自查部分）：修掉两个检查脚本的盲区、一个指向无效模型的工具、一处 BOM 隐患，并纠正改名造成的史实误引 | ① `check_asset_paths.py` 原先把期望的相对路径硬编码在脚本里，因此即便 `imgo2.py` 声明的路径写错也照样通过；改为从源码读出 `_DEFAULT_URDF_PATH`/`_DEFAULT_MOTION_DIR` 的字面成分再解析，并新增校验 URDF 引用的 17 个网格是否都在；负向测试（故意改错 `imgo2.py` 的路径）确认现在会 FAIL。② `check_model_sync.py` 只覆盖写死的 4 份 URDF，新增第 6 项以 `git ls-files *.urdf` 枚举全仓、未登记即 FAIL；负向测试（加入第 6 份副本）确认会报错。③ `inertia_urdf.py` 原指向 `imgo2_model/imgo2_urdf/urdf/imgo2.urdf`（腿部件片段，XML 都不能独立解析），必然在 `buildModelFromUrdf` 失败；改为指向完整训练模型并加说明。④ `tests/test_amp_alignment.py` 两处 `read_text("utf-8")` 后 `ast.parse`，遇到带 BOM 的源文件会抛 U+FEFF——全仓 27 个上游文件带 BOM，改为 `utf-8-sig`。⑤ 上一轮改名把文档中引用的历史路径 `/root/gpufree-data/Imgo2_rl/...` 一并改成了 `imgo2_rl`，属改写史实，已恢复原样并注明当时目录名。**全仓核查**：113 个 Python 文件按字节全部可编译；25 个 XML/URDF/xacro/launch 仅 1 个解析失败（即那个已知片段）；代码与配置中旧目录名残留 0 处；README 的 8 个任务 ID 与 8 个注册项完全对应，12 个脚本路径与 `build.sh` 的 `--cmake`/`--mujoco` 均存在；57 处路径式引用中 8 处未解析，逐条核对均为上下文相对路径、省略写法或日志里刻意提到的已删文件。未运行 Isaac Lab、Gazebo 或 pinocchio |
| 2026-09-15 | 代码复审（独立复核部分）：发现并修复两个阻断训练的缺陷，以及 deploy 侧若干真实缺陷 | 由两个并行只读复核 + 本人复核交叉验证，逐条在源码中确认后才采纳。**阻断训练的两处（已修）**：① `amp_env_cfg.py` 用了 `mdp.reset_amp_reference_state`，但 `mdp/__init__.py` 的 `from .amp_events import *` 是注释掉的、全仓无人导入 amp_events → 导入该配置即 `AttributeError`，AMP 任务根本无法创建（含文档推荐的训练命令）；改为在 `amp_env_cfg.py` 内直接导入 `amp_events` 模块（不放进 `mdp/__init__`，避免把 torch 拖进所有任务的注册）。② `base_height_l2` 沿用 `RewardsCfg` 的 `body_names=""`，而 Isaac Lab 会把空串当正则解析并因无匹配抛出 "Not all regular expressions are matched!" → 环境构造即失败；已设为 `[self.base_link_name]`（与 rough/himloco 配置一致）。两处都补了可在无仿真环境运行的回归测试，并各自做了负向测试（回退后测试失败、恢复后 sha256 不变）。**deploy 侧（已修，未编译）**：ROS2 两版控制器的 `std::clamp(…)` 丢弃返回值导致限位完全失效，单关节版参数名多加下划线使 URDF 从未解析成功（`joints_urdf_` 空指针）——已修并加保护；`actuator_net.py` 的 `BASE_PATH` 少上溯两层、`build.sh` 未 `cd` 到脚本目录（从仓库根调用会失败）——已修。**记录未修**：DEPLOY-04/05/06（joystick 目录空、Gazebo 链路缺 transmission 与控制器插件、MuJoCo 路径的映射与 MJCF 顺序冲突）。另更正上一轮那句「ros 两版都会 clamp 限位」的过度陈述。未运行 Isaac Lab、Gazebo、ROS 或编译器 |
| 2026-09-15 | 建立复审记录文档并把约定写进 AGENTS.md | 新建 [代码复审记录](docs/code_review_2026-09-15.md)，收纳本轮全部内容：① 两个阻断训练缺陷的现象/根因/修法/守位与负向测试；② 工具与检查脚本的六项修复；③ deploy 侧四处「已修未编译」及其验证前提；④ 已确认未修项（DEPLOY-04/05/06 等）与需要决定的项（MJCF 复用、安装规则、外部依赖、himloco 配置差异、base.yaml 定位、amp_motions 死代码、extension.toml 悬空 readme、Jetson 分支、GetUp/GetDown 快照等），每项写明「缺什么才能完成」；⑤ 已核实无问题的部分；⑥ 剔除的 7 条复核误报；⑦ 本轮限制。AGENTS.md 新增约定：每次工作都要把已修与待修写入维护文档；单次细节放 docs/、README 只留结论与链接；**代码改了不等于修好**，未验证的保留「已修，待验证」。按此口径复核当前问题表：AMP-03、ENV-01、DEPLOY-07 均保持待验证状态，未关闭任何条目。未运行 Isaac Lab、Gazebo、ROS 或编译器 |
