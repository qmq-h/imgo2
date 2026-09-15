# Imgo2 项目说明与维护记录

> 最后核对：2026-09-15。本文是整个工作区的维护入口，覆盖项目总览、训练部署流程、已知问题和变更记录。
> 状态依据包括用户反馈和本地源码/离线检查。用户已确认 PPO 完成训练与仿真验证；本文维护过程中未重新运行训练、策略回放、部署编译或真机实验。

## 1. 项目总览

Imgo2 是面向四足机器人的强化学习运动控制工作区，包含机器人模型、Isaac Lab 训练环境、参考动作数据和 C++ 部署框架。

当前代码链路为：

```text
URDF 与网格 ──→ Isaac Lab 环境 ──→ PPO / HIM-Loco / AMP 训练
参考动作数据 ──────────────────→ AMP 数据加载与参考状态初始化
训练 checkpoint ──→ 策略回放与导出 ──→ 部署配置对齐
                                      └─→ MuJoCo / Gazebo / 真机入口
```

最后一段部署链路仍需验证。当前部署策略为 Go2 参考占位策略，不能据此认定 Imgo2 已完成训练到部署的闭环。

### 目录职责

| 路径 | 内容与用途 |
|---|---|
| [imgo2_model/](imgo2_model/) | 原始建模目录。`imgo2_model/imgo2_urdf/urdf/imgo2.urdf` 只是**腿部件片段**（16 link／16 joint，仅 HIP/THIGH/SHANK/FOOT，无 `base` link、无 `<robot>` 起始标签，XML 不能独立解析），不是可直接加载的模型；`imgo2_mjcf/` 目前只有一套 MuJoCo 命名的网格（torso/hip_/thigh_/shank_/foot），无 XML |
| [imgo2_description/](imgo2_description/) | 用户确认的实际动作采集模型；`xacro/robot.xacro` 直接 include `urdf/imgo2_description.urdf`，采用 LF/LH/RF/RH 命名。**其腿部顺序与训练侧不同是有意为之，服务于另一处实现**：不要为「统一」而改写它，也不要把它与其它副本合并。两个 URDF 的关节符号与限位已核实与训练份一致（FK 重现录制数据）；`mjcf/` 下有 `imgo2.xml` 与 `scene.xml`，是 DEPLOY-02 的线索 |
| [imgo2_rl/](imgo2_rl/) | Isaac Lab 扩展、任务配置、训练脚本、自定义算法包、训练用模型及数据副本 |
| [imgo2_dataset/](imgo2_dataset/) | 参考动作数据；当前数据位于 `datasets/imgo2_motion/` |
| [imgo2_deploy/](imgo2_deploy/) | C++ 推理、观察缓存、状态机、MuJoCo/Gazebo/真机入口及策略配置 |
| [docs/](docs/) | 排查记录与离线结果：`amp_alignment_review.md`、`amp_data_audit.json` |
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
| MuJoCo 部署 | 有 C++ 入口及构建选项 | `robot_description/imgo2_mjcf/` 当前仅有 `.gitkeep`，缺少场景 XML |
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

交付一个可部署策略时，至少记录：checkpoint 来源、模型版本、关节映射、默认姿态、动作缩放与裁剪、PD 与力矩限幅、观察顺序与缩放、历史排列及重置方式、四元数约定、控制周期、网络输入输出，以及同一输入下的数值比较结果。

## 6. 部署流程

下面为现有 Bash 构建脚本对应的命令，面向具备相关依赖的 Linux 环境；本次未在 Windows PowerShell 中执行。

### 6.0 部署包结构

`imgo2_deploy/` 源自 `rl_sar-main` 并裁剪为单一机器人（上游为 Apache-2.0，其 `LICENSE` 保留在该目录内）。内部按 ROS 工作区组织：

| 路径 | 内容 |
|---|---|
| `src/imgo2_deploy` | RL 部署包：sim2sim、MuJoCo 仿真、Imgo2 真机入口 |
| `src/robot_msgs` | 共享的电机/机器人状态消息 |
| `src/robot_joint_controller` | ROS 仿真用的 Gazebo 关节控制器；**会用 URDF 的关节限位 clamp 指令** |
| `policy/imgo2` | 策略配置；`base.yaml` 与 `himloco/` 下的 Go2 参考占位策略 |
| `robot_description/imgo2_urdf` | 部署用 URDF 与网格 |
| `robot_description/imgo2_mjcf` | MuJoCo 场景放置处（当前为空，见 DEPLOY-02） |

策略配置里关节名用 `*_shank_joint`（对应本仓库 URDF），而不是 Go2 的 `*_calf_joint`。

### 6.1 MuJoCo：训练仿真到另一仿真器

前置工作：在 `imgo2_deploy/robot_description/imgo2_mjcf/` 放入有效的 `scene.xml` 及其引用模型，完成第 5 节的策略参数对齐。

```bash
cd imgo2_deploy
bash build.sh --mujoco
./cmake_build/bin/rl_sim_mujoco imgo2 scene
```

构建脚本会调用推理运行时和 MuJoCo 的下载/检查脚本，因此构建依赖网络和本地编译工具链。当前缺少场景文件，以上是待补齐前提后的操作流程，不是已验证的运行记录。

### 6.2 ROS / Gazebo

先加载已安装 ROS 的环境，再从部署根目录执行。下面启动示例对应 ROS 2：

```bash
bash build.sh
source install/setup.bash
ros2 launch imgo2_deploy gazebo.launch.py
```

现有脚本分别处理 ROS 1 Noetic 和 ROS 2 Foxy/Humble 的包配置；这些是代码支持分支，实际兼容性尚未核验。Gazebo 和控制器依赖由相应 ROS 包配置决定。

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
| MODEL-01 | P0 | 已按训练侧参数统一并验证；Gazebo/ROS 侧待复核 | **用户决定：四份模型的物理参数一律以训练侧为准。** 已按此统一：① 部署份 12 个腿部关节 axis 原全部取反（q_部署 = −q_训练）且限位镜像 → 改为训练份约定；② 部署份的足端 CoM/惯量与后腿 shank/thigh 惯量非对角项是五份模型中的**唯一异类**（`imgo2_model/` 片段、`imgo2_description` 两个 URDF、训练份四份一致）→ 改为训练侧；③ 足端碰撞圆柱长度训练侧与片段为 `0.01`、采集模型两个 URDF 与部署份为 `0.02`（2 对 3）→ 用户选训练侧 `0.01`，四份已统一；④ base 质量与惯量统一为 `5.53394020` + `0.03866860/0.10411461/0.12554111`。**现在四份完整 URDF 的 12 个关节轴/限位、17 个 link 的质量/质心/惯量/碰撞几何、base 规范值全部一致**（按逻辑关节名与逻辑 link 名比对，容忍 `imgo2_description` 的有意腿序与 LF_/LH_ 命名）。**为什么必须改部署份**：deploy 三条推理链路（`rl_real_imgo2.cpp` 走 Unitree SDK、`rl_sim.cpp` 走 ROS 控制器、`rl_sim_mujoco.cpp` 走 MJCF）均无任何符号取反，只有 `joint_mapping` 索引置换，无补偿；且 `robot_joint_controller`（ros 与 ros2）用 `urdf.initParamWithNodeHandle("robot_description")` 解析该 URDF 并用其限位 clamp 位置/速度/力矩——旧限位 `thigh -2.87..0.9`、`shank 0.733..3.0` 会把 `default_dof_pos` 的 shank `-1.50` clamp 成 `+0.733`，即该链路连默认站姿都不成立，属功能性断裂。**验证**：四份 URDF 的 FK 全部重现录制数据（最差恒等 RMSE 0.00214 m，部署份修符号前为 0.11790 m）；`check_model_sync.py` 五项检查全通过；两次负向测试（翻回符号、单独改足端 CoM）均能报错，恢复后 sha256 不变。唯一残留差异是外观：部署份比训练份多 4 个 `<material>` 块，不影响物理 | ① 在 Gazebo/ROS 链路复核（真机与 MuJoCo 链路不读该 URDF，不受影响）。② 若要把 `imgo2_model/` 片段补成完整 URDF 并让 rl/deploy 共用同一文件，还需处理 ROS 包内路径（建议 CMake 构建时拷贝）与 `build.sh:57` 的存在性检查；在此之前「共用一份文件」尚未实现。③ 遗留项：`imgo2_description/xacro/common/leg.xacro` 与 `urdf/imgo2.urdf` 全工作区无人引用，可考虑删除（未动） |
| AMP-01 | P0 | 恒等映射由数据独立确认 | 用户确认未覆盖采集参数；脚本按默认 LF/RF/LH/RH 写入关节和足端，等价于训练端 `[FL,FR,RL,RR]`。本地补充：改用实际采集模型 `imgo2_description.urdf` 复算 FK 与训练 URDF 结果一致（恒等 RMSE 均值 0.00107 m，声明顺序配对 0.22491 m）；髋外展左右对称性和关节位置/速度块相关性检查也不依赖 URDF 支持同一结论 | 采集来源、源码顺序和数据本身均已确认；新训练的行为改善仍待服务器验证 |
| AMP-02 | P0 | 历史报错，未复现 | 历史草稿记录 `RuntimeError: normal expects all elements of std >= 0.0`，调用栈为 `amp_on_policy_runner.py:136` → `amp_ppo.py:120` → `actor_critic.py:129` 的 `distribution.sample()` | 记录复现命令、数据和首个异常值；修复后训练验证；不能仅凭该报错断定根因 |
| AMP-03 | P0 | 代码已修正，待 Torch 回归 | 原均值方差更新使用归一化值，梯度惩罚使用未归一化值；已与 `amp_go2-main` 的处理方式对齐 | CPU Torch 更新回归通过，并检查新训练中的判别器与归一化统计 |
| AMP-04 | P0 | 已加入待验证配置 | 原配置删除高度项和非法接触终止；现在保留 0.30 m 高度项、基座触地终止，并补偿任务奖励的时间步长缩放，任务混合系数改为 0.3 | 新训练高度稳定、贴地比例降低、速度跟踪可接受；具体权重仍需实验 |
| DEPLOY-01 | P0 | 待对齐 | Go2 占位策略与 Imgo2 训练配置存在默认姿态、PD、限幅、指令缩放等差异 | 替换为来源明确的 Imgo2 策略，完成训练端与部署端同输入输出比较 |
| DEPLOY-02 | P0 | 缺场景 | `imgo2_deploy/robot_description/imgo2_mjcf/` 仅有 `.gitkeep`，且 `rl_sim_mujoco.cpp` 读的正是 `<robot>_mjcf/<scene>.xml` | **线索**：`imgo2_description/mjcf/` 下已有 `imgo2.xml` 与 `scene.xml`（15.9 KB／0.8 KB），可评估能否直接复用或需按 deploy 网格路径改写；生成并成功加载场景，保存 sim2sim 测试结果 |
| DEPLOY-03 | P0 | 缺依赖，待适配 | SDK2 目录为空，真机目标被跳过 | 依赖到位、目标生成、通信接口验证通过 |
| EXPORT-01 | P1 | 待验证 | 导出配置与 C++ 配置格式不同；比较脚本假设六帧历史 | 明确转换规则，记录实际网络维度、历史规则与误差指标 |
| DATA-01 | P1 | 当前一致 | 两处动作数据副本哈希一致 | 每次更新后核对副本，记录数据来源和版本 |
| EXP-01 | P1 | 待补充 | 已记录 PPO 验证与 AMP 贴地现象的用户反馈，尚缺对应日志、命令和模型路径 | 按第 8 节补充真实实验与产物路径 |

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

`check_amp_joint_order.py` 只用标准库，默认读取同级的 `imgo2_description/urdf/imgo2_description.urdf`；本机可用解释器见第 8.3 节维护记录。它检查采集模型 FK、髋外展左右对称性和关节位置/速度块相关性，三项全部通过才返回 0。

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
