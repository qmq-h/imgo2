# Imgo2 项目说明与维护记录

> 最后核对：2026-09-22。已在 Windows／Python 3.14.6 完成本轮离线检查；上层拖曳采用 dynamics decoder＋recurrent PPO：51 维本体帧经 GRU 估计机器人速度、负载质量和牵引力，估计值 detach 后进入 actor，预测误差不进入 reward。自有 `rl_lab` recurrent runner 已接线，当前仍缺 Isaac Lab 训练机运行验证。详见 [离线检查记录](docs/offline_check_2026-09-22.md)、[拖曳 GRU 记录](docs/towing_gru_design_2026-09-22.md) 与 [AMP 配置收缩记录](docs/amp_config_cleanup_2026-09-22.md)。
>
> 2026-09-22 又在**训练机本机**（`/opt/conda/envs/isaaclab`：Isaac Lab `0.45.9`、RSL-RL `2.3.3`、torch `2.7.0+cu128`）做了一轮训练前置核查：拖曳离线测试 **210 项全通过**（本机有 torch，此前因缺 torch 跳过的项已实跑），51 维帧／56 维 actor 契约与冻结 AMP 策略依赖均自洽。该 GPU 在本机**可用**（用户实测 `run_isaaclab.sh --check` → `cuda available: True`、`CUDA solve: OK`；agent 会话进程曾观察到不可见，属该进程视角，见 TOW-04）。已为用户执行训练落地两项代码改动：加入任务注册 `Imgo2-towing-upper-rl-lab`（`--agent=rl_lab_cfg_entry_point`）、修掉 `train.py` 的 `--agent` 默认值；注册表解析与环境构造已由 4 环境 × 10 轮冒烟实跑验证通过。修法与执行清单见 [拖曳训练前置记录](docs/towing_training_prep_2026-09-22.md)。
>
> 当前阶段只说明“仿真链路和基本行为成立”，不等于两类绳的物理真实性、瞬态品质或控制效果已经完成评价。可复现数据结论见 §2 和问题表；详细实验与调试历史放在 `docs/`，不在 README 展开。

## 1. 项目总览

Imgo2 是面向四足机器人的强化学习运动控制工作区，包含机器人模型、Isaac Lab 训练环境、参考动作数据和 C++ 部署框架。

当前代码链路为：

```text
URDF 与网格 ──→ Isaac Lab 环境 ──→ PPO / HIM-Loco / AMP 训练
参考动作数据 ──────────────────→ AMP 数据加载与参考状态初始化
训练 checkpoint ──→ 策略回放与导出 ──→ 部署配置对齐
                                      └─→ MuJoCo / Gazebo / 真机入口
```

当前拖曳研究主线是：①完成上层 recurrent PPO 与 decoder 训练闭环；②导出读取 51 维单帧并维护 GRU hidden state 的 actor，完成 MuJoCo／Gazebo sim2sim 拖曳和停车验证。真机作为独立后续分支，目前只维护可部署 observation 与控制频率契约，不纳入这一阶段验收。详细路线见 [上层拖曳 RL 计划](docs/paper_plan_rl.md)。

最后一段部署链路仍需验证。`himloco` 仍是 Go2 参考占位策略；AMP 已于 2026-09-17 换成
Imgo2 自己的 checkpoint，**2026-09-18 已确认**：45 维 actor 正式导出并接入部署，Gazebo 侧
0.2–1.5 m/s 跟速 93.4–99.8%、站立漂移 0.001 m/s、FL-FR 相位 174°–188°（trot）；
训练侧 Isaac Lab 足端回放独立复现了「干净步态」并给出剩余差距的**根因是步幅偏小**
（每步只有参考的 0.61 倍，于是用 3.1 倍步频凑速度），见
[足端步态回放记录](docs/gait_eval_amp_24500_2026-09-18.md)。
因此**已可以认为 AMP 完成了「训练→导出→sim2sim 行走」的闭环**；仍未闭环的是真机，
以及「步态形态对齐录制参考」这一档目标（AMP-06）。

### 目录职责

| 路径 | 内容与用途 |
|---|---|
| ~~imgo2_model/~~ | **2026-09-17 已删除**（git 可追溯）。原先只有腿部件片段 URDF（无 `base`、无 `<robot>`，XML 不能独立解析）+ 两套模型网格，无任何消费者；其网格与 `imgo2_description/meshes` 内容相同。删除理由见维护记录与 MODEL-02 |
| [imgo2_description/](imgo2_description/) | **模型唯一源（2026-09-17 起）**。`xacro/core.xacro` 是物理内核（17 link/16 joint，命名 FL/FR/RL/RR），`xacro/robot.xacro` 是组装入口，按开关拼上 `transmission.xacro`（仅 ros_control）、`gazebo.xacro`（插件 + per-link 接触参数）、`imu.xacro`（imu_link）。生成物入库、勿手改：`urdf/imgo2.urdf`（纯）、`urdf/imgo2.gazebo.urdf`（+三者）；`mjcf/{imgo2.xml,scene.xml}` 是 `rl_sim_mujoco` 读的 MuJoCo 模型（训练物理 + 参考项目求解器/接触块 + `framelinvel`）。`meshes/` 是全仓唯一一套网格（指纹 `8dc5b5995a11`） |
| [imgo2_rl/](imgo2_rl/) | Isaac Lab 扩展、任务配置、训练脚本、自定义算法包。**模型与网格副本已于 2026-09-17 删除**，训练直接读 `imgo2_description/urdf/imgo2.urdf`（见 `assets/imgo2.py`） |
| [imgo2_dataset/](imgo2_dataset/) | 参考动作数据；当前数据位于 `datasets/imgo2_motion/` |
| [imgo2_deploy/](imgo2_deploy/) | C++ 推理、观察缓存、状态机、MuJoCo/Gazebo/真机入口及策略配置。**模型副本 `robot_description/` 已于 2026-09-17 删除**，MuJoCo 读 `imgo2_description/mjcf/`、Gazebo 读 `imgo2_description/urdf/imgo2.gazebo.urdf` |
| [docs/](docs/) | 排查记录与离线结果：`amp_alignment_review.md`、`code_review_2026-09-15.md`、`amp_data_audit.json` |
| `paper_plan_imgo2.md`（本地，Git 忽略） | 用户研究草稿，不随仓库同步；已实现的拖曳物理与验证结论见 [拖曳仿真验证](docs/towing_simulation_validation.md) |
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
| 小车拖曳仿真 | 小车模型、弹性绳与不可伸长绳均已接入 Isaac Sim；可记录机器人/小车状态、绳长/张力/冲量、接触、轮速、阶段、停车间隙与追尾事件。可视化已确认拖曳与停车后前滑的基本行为 | 两类绳的物理合理性、瞬态差异、参数敏感性及控制策略效果仍需依据记录数据统一评价；可视化观察本身不作为这些结论的证据 |
| MuJoCo 部署 | `robot_description/imgo2_mjcf/scene.xml` 已就位，关节轴/限位/传感器布局与训练侧逐项一致；2026-09-17 `bash build.sh -mj` 构建通过；参数已按参考项目对齐（见 §5.3） | GUI 入口需在有可用显示的机器上运行；AMP 策略当前只给出静态下蹲、不跟踪速度指令（AMP-06） |
| AMP sim2sim | FSM 增量式新增 `RLFSMStateAMPLocomotion`（键 `2`）；`policy.pt` 与 checkpoint actor 128 组输入最大误差 0；16 s 回放无 NaN | 步态未成立；`lin_vel` 观测恒为 0（AMP-06）；GUI 未运行 |
| 真机部署 | 有 `rl_real_imgo2.cpp` 和状态机代码 | Unitree SDK2 目录当前为空，CMake 会跳过真机目标；硬件通信适配未验证 |

## 3. 代码阅读导航

底层运动任务位于 `imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/`；拖曳任务包 [towing/](imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/) 与 `locomotion/` 同级。小车直接维护 [cart.urdf](imgo2_description/cart/cart.urdf)，离线与仿真端的完成状态见 [拖曳仿真验证](docs/towing_simulation_validation.md)。

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
| 步态评估（占空比/步频/相位/关节幅值，需回放） | [eval_gait.py](imgo2_rl/scripts/tools/eval_gait.py) |
| 步态指标口径（纯 numpy，可离线回归，`eval_gait.py` 只采样与打印） | [gait_metrics.py](imgo2_rl/scripts/tools/gait_metrics.py)、[test_gait_metrics.py](imgo2_rl/tests/test_gait_metrics.py)（18 项，无需 GPU） |
| 参考轨迹回放（在 Isaac Lab 里放参考动作） | [play_reference_motion.py](imgo2_rl/scripts/tools/play_reference_motion.py) |
| 参考步态基线（从 21 份动作直接算出，判读用尺子） | [gait_reference_baseline.json](docs/gait_reference_baseline.json) |
| 24500 轮 AMP 的足端回放结果与逐腿对照 | [gait_eval_amp_24500_2026-09-18.md](docs/gait_eval_amp_24500_2026-09-18.md) |
| 训练曲线（读 TensorBoard event 文件，纯标准库，训练中也能读；`--tags`/`--steps`/`--match`；**注意 `*/time` 系列 tag 的 x 轴是墙钟秒、不是迭代轮号**，工具会单独点名） | [read_tfevents.py](imgo2_rl/scripts/tools/read_tfevents.py)、[test_read_tfevents.py](imgo2_rl/tests/test_read_tfevents.py)（6 项，无需 GPU） |
| 判别器探针（离线，读 checkpoint 的判别器做破坏性对照，无需 GPU） | [probe_amp_discriminator.py](imgo2_rl/scripts/tools/probe_amp_discriminator.py) |
| 策略 vs 专家的 AMP 观测块级归因（需 `eval_gait.py --dump-npz` 的产物） | [probe_amp_policy_vs_expert.py](imgo2_rl/scripts/tools/probe_amp_policy_vs_expert.py) |
| λ_gp 对判别器分辨能力的合成对照（离线训练，CPU） | [probe_gp_effect.py](imgo2_rl/scripts/tools/probe_gp_effect.py) |
| 步态调整方案（风格项为什么没梯度、按成本排序的改法与判据） | [amp_gait_adjust_plan_2026-09-18.md](docs/amp_gait_adjust_plan_2026-09-18.md) |
| 拖曳质量／速度／阻力边界扫描 | [scan_towing_boundary.py](imgo2_rl/scripts/towing/scan_towing_boundary.py)、[扫描口径](docs/towing_boundary_scan_2026-09-21.md) |
| 部署观察处理与策略推理 | [rl_sdk.cpp](imgo2_deploy/src/imgo2_deploy/library/core/rl_sdk/rl_sdk.cpp) |
| 部署状态切换 | [fsm_imgo2.hpp](imgo2_deploy/src/imgo2_deploy/fsm_robot/fsm_imgo2.hpp) |
| 构建选项与外部依赖条件 | [build.sh](imgo2_deploy/build.sh)、[CMakeLists.txt](imgo2_deploy/src/imgo2_deploy/CMakeLists.txt) |

配置有继承关系：理解最终行为时，要同时检查公共配置和任务类的 `__post_init__()`。例如粗糙地形 PPO 和 HIM-Loco 配置都将随机推扰、外力事件设为 `None`，不能仅凭公共类存在相应定义就写成“已启用”。

## 4. 训练与回放流程

### 4.1 环境与路径前提

以下 Python 命令均从 `imgo2_rl/` 执行，并使用已配置好 Isaac Lab/Isaac Sim 的 Python 环境。

- 当前训练机版本基线由用户确认：**Isaac Lab 2.2.1、RSL-RL 2.3.3**；Python、Isaac Sim、PyTorch、CUDA 和驱动版本仍需在训练机实际命令输出中补齐。
- `imgo2_rl` 和 `rl_lab` 安装脚本声明 Python `>=3.10`，不代表完整兼容矩阵。上层拖曳已像 AMP 一样使用仓库自有 `rl_lab` runner／PPO／recurrent model，不依赖外部 RSL-RL runner/config API；普通 PPO 入口仍按 2.2.1 接口维护。详见 [训练栈兼容性记录](docs/training_stack_compatibility_2026-09-22.md)。
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
| `Imgo2-basemove-flat-amp-height` | `scripts/rl_lab/amp/train.py`（平坦地形，保留 Imgo2 高度与轻量姿态奖励；`AMPHeightRunnerCfg`） |
| `Imgo2-basemove-flat-amp-fanziqi` | `scripts/rl_lab/amp/train.py`（Fanziqi A1 AMP 配方：`dt=0.005 s`、`decimation=6`，只留线／角速度任务奖励，42 维 actor，无持续外力；`FanziqiAMPRunnerCfg`） |

此外注册了 `Imgo2-basemove-rough-himloco-play`，两套 AMP 分别使用同名 `-play` 任务。

基础检查及训练命令示例：

> **先读这条（2026-09-20）**：所有 Isaac Lab 命令都走 `imgo2_rl/scripts/run_isaaclab.sh`，例如
> `bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/rl_lab/amp/train.py --task=... --headless`。
> 原因是 `libtorch_cuda_linalg.so: cannot open shared object file` 这个报错（用户跑 `eval_gait.py` 时实遇）：
> Isaac Lab 的 `SimulationContext` 把 Isaac Sim 的 prim/变换后端硬编码成 `backend="torch"`，
> 建场景时会 `torch.linalg.solve` 到 CUDA 张量，而这一步要 **`dlopen` 裸名**的
> `libtorch_cuda_linalg.so` —— glibc **不会**用 `DT_RPATH` 解析显式 `dlopen`（torch 的
> `libtorch_cuda.so` 恰好只有 RPATH），所以直接把 `torch/lib` 放进 `LD_LIBRARY_PATH` 才可靠。
> 启动器只做这一件事（并追加 `nvidia/*/lib`），**不依赖当前 shell 碰巧设了什么**；
> 先跑 `bash imgo2_rl/scripts/run_isaaclab.sh --check` 自检（会打印 torch 版本、那个 .so 是否存在、
> 能否 dlopen、CPU/CUDA 上 `linalg.solve` 是否可用）。机理与离线复现见
> [启动器记录](docs/isaaclab_launcher_libtorch_dlopen_2026-09-20.md)。

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

导出后按三条契约复核，再拷进 `imgo2_deploy/policy/imgo2/<算法>/policy.pt`：① 导出件与 checkpoint 的
actor 在同一批确定性输入上 max|diff| = 0；② 能被部署自己的 libtorch 加载；③ 部署 interface 的观测维数
与顺序和训练侧一致（AMP 的 45 维见第 5.4 节与 [Gazebo 记录](docs/gazebo_ros2_bringup_2026-09-17.md) 第 9 节）。

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
| `3` | `RLFSMStateAMPLocomotion` | `policy/imgo2/amp/` | 45 维（无 `base_lin_vel`，与参考 `amp_go2` 的 actor 组成一致） | `policy.pt`（`model_24500.pt` 的导出，sha256 `cba59d44e387834e…`；与 checkpoint actor 数值完全一致。**24500 轮已确认：0.2/0.3/0.5 m/s 跟速 93.4%/97.9%/95.9%，站立漂移 0.001 m/s，周期强度 0.71–0.83、相位 ±180°**，见 [Gazebo 记录](docs/gazebo_ros2_bringup_2026-09-17.md) 9.7 节） |

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

### 5.5 拖曳链路的批处理与 JIT 边界

上层拖曳训练同时用到三个网络，**都在环境维度整批前向，没有逐环境循环**：

| 网络 | 调用位置与频率 | 每次 batch |
|---|---|---|
| actor / critic（`ActorCriticRecurrent`） | `towing_on_policy_runner.py` rollout 循环，每控制步各一次 | `num_envs` |
| decoder（`TowingDynamicsDecoder`） | 同上，每控制步一次；estimate detach 后拼成 56 维 actor 输入 | `num_envs` |
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
| CHECK-01 | P1 | **主体离线检查通过；依赖项待补跑** | 2026-09-22 使用 Python 3.14.6：资源路径、模型同步、AMP 数据／关节顺序、`compileall`、tracked-ignore 检查均通过；拖曳测试 194 项通过、12 项按可选环境跳过。全量测试收集 281 项，得到 256 通过、24 跳过、1 个导入错误；错误仅为 `test_gait_metrics.py` 找不到 `numpy`，尚无代码断言失败证据。详见 [记录](docs/offline_check_2026-09-22.md) | 在带 NumPy 的解释器重跑 `test_gait_metrics.py`；在带 PyTorch／Isaac Lab 的训练环境补跑当前跳过项。未执行前不得把这些项记为通过 |
| TOW-01 | P1 | **仿真与记录链路已完成** | 小车、弹性绳和不可伸长绳已接入同一拖曳场景；支持单／多环境、质量与阻力等参数扫描。记录包含两刚体状态、绳状态、张力／冲量、轮速、接触、阶段、真实间隙与追尾事件。可视化已确认机器人拖车和停车后小车前滑。详见 [拖曳仿真验证](docs/towing_simulation_validation.md) | 保持入口、记录格式和离线汇总工具可复现；本项不再以“继续看画面”作为验收方式 |
| TOW-02 | P1 | **边界扫描脚本已实现，待实跑** | `scan_towing_boundary.py` 默认扫描速度 0.2–1.0、质量 5–25 kg 和两档轮阻，并区分“稳态拉不动”与“可拖但 Direct Stop 追尾”。地面摩擦默认固定 0.8，可在边界附近追加 0.4/0.8/1.2 复核 | 在训练机先跑 compliant 主网格，根据 `boundary.json` 缩小摩擦复核范围；之后再确定 v0.1 训练域。当前未实现 breakaway/Coulomb 阻力，不能把地面摩擦当成它的替代 |
| TOW-03 | P0 | **自有 recurrent PPO＋decoder runner 已接线；任务已注册；4 环境冒烟已通过（2026-09-22）；完整验收未完成** | decoder 使用 `51→128→GRU(128)` 预测机器人机体系 `vx/vy`、负载质量和牵引力 `Fx/Fy`，5 维 estimate detach 后组成 56 维 actor 输入。仓库 `rl_lab` runner 已管理 decoder／actor／critic 三套 GRU、critic-only normalizer、rollout estimate 固化、PPO 后 decoder 更新及联合 checkpoint；不依赖外部 RSL-RL runner/config API。STOP 后有界 GT 拉力惩罚已加入。**2026-09-22 训练机前置核查**：拖曳离线测试 210 项全通过、51 维帧与 56 维 actor 契约一致、冻结 AMP 策略与 `torch.jit.load` 路径齐备；修掉 `train.py` 的 `--agent` 默认值（原 `towing_rl_lab_cfg` 全仓无对应注册项，必抛 `ValueError`）；加入注册 `Imgo2-towing-upper-rl-lab`（`--agent=rl_lab_cfg_entry_point`），并把原守门断言换成锁定注册接线的契约测试（已用两组负向测试确认有效）。详见 [GRU 记录](docs/towing_gru_design_2026-09-22.md) 与 [训练前置记录](docs/towing_training_prep_2026-09-22.md) | 在 Isaac Lab 2.2.1 训练机完成 wrapper 构造、4／256 环境 recurrent rollout、checkpoint 恢复、estimator 精度、STOP reward 排序和无小车隔离验证，重点排除策略为卸载拉力而让小车逼近的捷径。**注意**：注册是用户为执行训练而显式翻过 `towing_upper_rl_review` 第 98 行「未完成运行验收前不注册」这道保护，验收通过前该项不得记为完成。**2026-09-22 冒烟实跑（4 环境 × 10 轮）通过**：环境构造成功、51 维帧／56 维 actor 契约在运行时成立（`memory_a` 输入 56 维、`memory_c` 65 维）、checkpoint 含三套 GRU＋decoder＋两优化器＋critic-only normalizer、`iter` 正确写入 0→10（AMP-08 修好后首次在真实 checkpoint 上确认）、decoder 可从 checkpoint strict 加载并推理。详见 [冒烟记录](docs/towing_smoke_run_2026-09-22.md)。**仍未验证**：TensorBoard 未记录拖曳诊断量（`cart_present`、绳力、终止原因），故 12.5% 无小车比例、跨环境隔离、STOP reward 排序三条关键验收项无法评价；未跑 256 环境、未做 checkpoint 恢复、未做 estimator 精度与 scripted policy 复现 |
| TOW-04 | P0 | **已解除：GPU 在本机可用** | 2026-09-22 DSH 会话进程内曾观察到 GPU 不可见（`NVIDIA_VISIBLE_DEVICES=void`、无 `/dev/nvidia0`／`nvidiactl`／`nvidia-uvm`、`torch.cuda.is_available()==False`、`nvidia-smi` 报 `Failed to initialize NVML: Insufficient Permissions`）；**该观察不适用于执行训练的 shell**。用户同日实测 `bash imgo2_rl/scripts/run_isaaclab.sh --check` 输出 `cuda available: True` 与 `CUDA solve: OK`（torch 2.7.0+cu128，`dlopen(linalg)` OK），GPU 正常。教训：`torch.cuda` 可见性按进程求值，agent 会话进程的观测不能代表用户终端 | 无需再处理；后续以执行方的 `--check` 输出为准。拖曳训练的「已运行」判定改为取决于实际训练是否启动并产出日志 |
| CART-01 | P1 | **基础模型已验证** | 小车落地、轮接触、质量惯量、黏性轮阻和自由滑行已通过训练机实验；关键数据见 [拖曳仿真验证](docs/towing_simulation_validation.md) | 后续只在模型或记录接口变化时回归，不重复展开早期调试历史 |
| DOC-01 | P1 | 已完成 | 训练 README 曾引用失效的 `script/himloco_rsl_rl` 安装路径和写死的个人服务器 checkpoint | 已改为 `scripts/rl_lab`，checkpoint 改为 `<run>` 占位并注明不可沿用；根 README 与子项目 README 表述一致 |
| ENV-01 | P0 | 代码已修正，待服务器验证 | 原先 `assets/imgo2.py` 的两条路径写死为 `/root/gpufree-data/Imgo2_rl/...`（当时目录名还是 `Imgo2_rl`），且假定该目录就是项目根，合并成 monorepo 后必然失效且 glob 为空时无明确报错 | 已改为由 `Path(__file__)` 推导项目根，并支持 `IMGO2_AMP_MOTION_DIR`／`IMGO2_URDF_PATH` 覆盖；新增 `scripts/tools/check_asset_paths.py` 供新机器自检（本机通过：URDF 存在、动作文件 21 份、无残留机器路径）。服务器上仍需运行该脚本并记录实际加载路径 |
| MODEL-01 | P0 | 已按训练侧参数统一并验证；Gazebo/ROS 侧待复核 | **用户决定：四份模型的物理参数一律以训练侧为准。** 已按此统一：① 部署份 12 个腿部关节 axis 原全部取反（q_部署 = −q_训练）且限位镜像 → 改为训练份约定；② 部署份的足端 CoM/惯量与后腿 shank/thigh 惯量非对角项是五份模型中的**唯一异类**（`imgo2_model/` 片段、`imgo2_description` 两个 URDF、训练份四份一致）→ 改为训练侧；③ 足端碰撞圆柱长度训练侧与片段为 `0.01`、采集模型两个 URDF 与部署份为 `0.02`（2 对 3）→ 用户选训练侧 `0.01`，四份已统一；④ base 质量与惯量统一为 `5.53394020` + `0.03866860/0.10411461/0.12554111`。**现在四份完整 URDF 的 12 个关节轴/限位、17 个 link 的质量/质心/惯量/碰撞几何、base 规范值全部一致**（按逻辑关节名与逻辑 link 名比对；统一前容忍 `imgo2_description` 的有意腿序与 LF_/LH_ 命名，2026-09-17 统一后已同序，见 MODEL-02）。**为什么必须改部署份**：deploy 三条推理链路（`rl_real_imgo2.cpp` 走 Unitree SDK、`rl_sim.cpp` 走 ROS 控制器、`rl_sim_mujoco.cpp` 走 MJCF）均无任何符号取反，只有 `joint_mapping` 索引置换，无补偿；且 `robot_joint_controller` 会用 URDF 的关节限位 clamp 指令。注意两版原本行为不同：ROS1 自己定义了按引用修改的 `clamp(double&,…)`，确实生效；**ROS2 把 `std::clamp(…)` 当语句调用、丢弃返回值，等于完全没有限位**，且单关节版请求的参数名多写了一个下划线（`"robot_description_"`），URL 从未解析成功、`joints_urdf_` 为空指针。两处已在本轮修正，但尚未编译验证（见 DEPLOY-07）——旧限位 `thigh -2.87..0.9`、`shank 0.733..3.0` 会把 `default_dof_pos` 的 shank `-1.50` clamp 成 `+0.733`，即该链路连默认站姿都不成立，属功能性断裂。**验证**：四份 URDF 的 FK 全部重现录制数据（最差恒等 RMSE 0.00214 m，部署份修符号前为 0.11790 m）；`check_model_sync.py` 五项检查全通过；两次负向测试（翻回符号、单独改足端 CoM）均能报错，恢复后 sha256 不变。唯一残留差异是外观：部署份比训练份多 4 个 `<material>` 块，不影响物理 | ① 在 Gazebo/ROS 链路复核（真机与 MuJoCo 链路不读该 URDF，不受影响）。② 若要把 `imgo2_model/` 片段补成完整 URDF 并让 rl/deploy 共用同一文件，还需处理 ROS 包内路径（建议 CMake 构建时拷贝）与 `build.sh:57` 的存在性检查；在此之前「共用一份文件」尚未实现。③ 遗留项：`imgo2_description/xacro/common/leg.xacro` 与 `urdf/imgo2.urdf` 全工作区无人引用，可考虑删除（未动） |
| AMP-01 | P0 | 恒等映射由数据独立确认 | 用户确认未覆盖采集参数；脚本按默认 LF/RF/LH/RH 写入关节和足端，等价于训练端 `[FL,FR,RL,RR]`。本地补充：改用实际采集模型 `imgo2_description.urdf` 复算 FK 与训练 URDF 结果一致（恒等 RMSE 均值 0.00107 m，声明顺序配对 0.22491 m）；髋外展左右对称性和关节位置/速度块相关性检查也不依赖 URDF 支持同一结论 | 采集来源、源码顺序和数据本身均已确认；新训练的行为改善仍待服务器验证 |
| AMP-02 | P0 | 历史报错，未复现 | 历史草稿记录 `RuntimeError: normal expects all elements of std >= 0.0`，调用栈为 `amp_on_policy_runner.py:136` → `amp_ppo.py:120` → `actor_critic.py:129` 的 `distribution.sample()` | 记录复现命令、数据和首个异常值；修复后训练验证；不能仅凭该报错断定根因 |
| AMP-03 | P0 | Torch 回归仍未执行过（本机无 torch，测试被跳过）；待训练环境验证 | 原均值方差更新使用归一化值，梯度惩罚使用未归一化值；已与 `amp_go2-main` 的处理方式对齐。**更正（2026-09-17 晚）：此条此前写成"Torch 回归已通过／6 项全过"是错的。** 本机（Linux `qmq-linux`）在 `~/miniconda3/envs/isaaclab` 下跑 `tests/test_amp_alignment.py` 的实际结果是 **5 通过 + 1 跳过**，跳过的正是本项 Torch 回归 `test_raw_statistics_and_normalized_gradient_penalty`（原因 `CPU regression needs the training environment's torch and numpy`）；全机五个 conda 环境与 `/usr/bin/python3` 都没有 torch（均报 `ModuleNotFoundError`），所以"该环境有 torch 2.7.0+cu128"的旧记录不成立，**AMP-03 的 Torch 那一半至今没有任何一次执行证据**（Windows 侧同样因缺 torch 跳过） | 需要一台真正装了 torch（有 Isaac Lab 训练环境）的机器跑 `python -m unittest discover -s tests -p test_amp_alignment.py`，确认 `test_raw_statistics_and_normalized_gradient_penalty` 不再 skip 且通过；之后再检查新训练中的判别器与归一化统计 |
| AMP-04 | P0 | 已加入待验证配置 | 原配置删除高度项和非法接触终止；现在保留 0.30 m 高度项、基座触地终止，并补偿任务奖励的时间步长缩放，任务混合系数改为 0.3 | 新训练高度稳定、贴地比例降低、速度跟踪可接受；具体权重仍需实验 |
| AMP-05 | P0 | **45 维 + 正式导出 + sim2sim 确认完成** | actor 移除 `base_lin_vel`（真机需状态估计）：`amp_env_cfg.py` 设 `observations.policy.base_lin_vel = None`，actor 45 维、critic 保留 48 维；部署侧 `amp/config.yaml` 为 45 维、观测项/顺序/缩放/动作/增益已与训练配置**逐项核对一致**。**24500 轮 checkpoint（`model_24500.pt`，sha256 `cc8ee22a…`；部署 `policy.pt` `cba59d44…`）已在 Gazebo 确认**：契约① 与 actor max diff 0.0、② 部署 libtorch 2.3.0 加载通过；按「每条命令一条新栈」测得跟速 **0.2 → 93.4%、0.3 → 97.9%、0.5 → 95.9%、0.75 → 98.7%、1.0 → 99.8%、1.2 → 99.8%、1.5 → 98.5%**，高度 0.277–0.300 m，`vx=0` 漂移 **0.001 m/s**、抖动 8.6/6.1/9.3（5000 轮时是 0.047 m/s、35.7/39.8/80.1）。**顺带实证了 AMP-08**：这份 checkpoint 的 `iter` 字段为 24500（不再是 0）。详见 [Gazebo 记录](docs/gazebo_ros2_bringup_2026-09-17.md) 9.7 节 | 剩余项：**步频仍是录制参考的约 2.5 倍**（周期 0.208–0.365 s 对 0.600 s / 1.67 Hz），且抖动绝对值随速度上升（shank 57→412）；若要"像真机录的那样走"，这是下一档目标。另：极低速 0.2 m/s 的周期强度只有 0.23（0.3 是 0.78）。**不得直接截掉旧网络的 3 个输入。** |
| AMP-06 | P0 | 配比已调好并训练验证；**步态在 sim2sim 侧已确认由「高频乱倒腿」变为「干净对角步」**；训练侧足端回放把剩余差距的根因定为**步幅偏小（0.61 倍）**而非相位乱 | ① **"一启动就飞"已修**：`PhysicsThread` 不套 keyframe；场景 base 高度改 `0 0 0.5` 后不再弹飞。② 抄入参考项目求解器/接触块后能站起。③ **MODEL-03**：`framequat` 挂 `imu` site 后站姿 0.2663→0.3015。④/⑤ 已排除「缺 `lin_vel` 观测」与「模型物理错误」（参考策略在我们模型上 `vx=0.5` 走 0.454 m/s）。**⑥ 2026-09-17 定位真因并调好配比**：三次训练对照——Run1 `1.0/0.3/-10` 只站不走（误差恒 1.72、1000 轮后指标全平）；Run2 `50/17/-1` 退化为贴地滑行（高度 0.172、贴地 0.89、std 失控 12.17）；本次 `4.0/2.0/-5` + 三项轻量姿态约束（`lin_vel_z_l2 -1`/`ang_vel_xy_l2 -0.05`/`joint_pos_limits -2`）**线速度误差 1.31→0.554 m/s**、高度 **0.305**、贴地 **0.08%**、触地终止 2.1%、超时终止 97.9%、std 稳定 0.394。**关键不是绝对权重，而是速度项与高度项的量级比**：Run1 惩罚是速度的 7.7 倍（不动）、Run2 速度是惩罚的 66 倍（趴滑）、本次 0.75:1 才同时拿住「走 + 站得住」 | **待修**：速度跟踪 kernel 仍仅 0.037（误差波动大，平均速度约为指令的 84%），24500 轮后 Gazebo 复测（每条命令一条新栈）周期强度升到 **0.71–0.83**、FL-FR 相位 174°–188°（trot），与键 1 参考策略（强度 0.96–0.97）同量级。**2026-09-18 训练侧足端回放（Isaac Lab，8 环境 × 950 步、丢弃前 50 步，`model_24500`）**：**唯一实质缺陷是步频 3.1 倍**（周期 **0.1926 s / 5.19 Hz** 对参考 0.600 s / 1.67 Hz；0.6 m/s 下同样是 0.198 s ⇒ **不是指令超范围**）。其余指标都对得上参考：**相位是干净对角步**（FL-FR **+174.2°**、FL-RR **+17.3°**、集中度 R **0.86–0.91**，与 Gazebo 侧 176.9° 相差 3° 以内）、**关节峰峰值落在参考范围内**（shank 0.516–0.618 对参考中位 0.635 / 范围 0.477–1.063；thigh 0.407–0.470 对中位 0.477）、抬脚 0.067–0.085 m（参考 0.083–0.121）、前后腿比 1.06（参考 0.87–1.06）、步幅 0.139–0.160 m（参考同档 0.295）、高度 0.3117 m、线速度误差 0.108 m/s。**逐条更正（v1→v3）**：v1 报的「高抬腿 1.6 倍」「前后腿不对称 1.33」「shank 幅度 1.35–2.25 倍」全部是**未 warmup 的出生瞬态污染**（加 50 步 warmup 后真值 0.066–0.083 m、1.06、0.52–0.62 rad）外加一处 `joint_pos` 堆叠轴 bug，已撤回；v2 报的「相位 156°、集中度 0.17」是相位估计器被非整数周期（0.1926 s ÷ 0.02 s = 9.63 步）抹平，换插值法后恢复。**为什么「加风格权重」不是解法（离线判别器分析）**：终局每步混合回报 1.97 = 任务 1.402（71%）+ 风格 0.540（29%），风格项按判别器标定可达 1.335；把策略的「关节角+足端+关节速度」三块换成专家能补回 **86%** 的风格差距（**关节速度块单独占 53%**，其 σ 是专家的 **1.7 倍**＝3 倍步频的印记），而姿态/足端的边缘分布几乎重合（σ 比 1.00）；**但判别器在策略点的局部梯度只有 8e-5**（1σ 全维扰动只移动 d 约 5e-4）⇒ 风格奖励是个没有方向的台阶，20000 轮里 d_policy 只动了 0.07。合成对照进一步指向 **λ_gp=10**：同预算下 λ 10→1→0 时「正常节律 vs 3 倍速节律」的风格奖励差为 **0.105 / 0.564 / 1.262** 每步。⇒ 优先级：**P1 降 λ_gp（10→1~2）**、**P2 加 `feet_air_time`（目标滞空 ≈0.2 s，当前 0.067–0.085）**、P3 兜底试 `lerp 0.3→0.5`、P4 恒定抬头 +2.1°/+2.9° 用 `flat_orientation_l2` 单变量消融；压指令范围已降级。完整证据、表格与判据见 [AMP 步态调整方案](docs/amp_gait_adjust_plan_2026-09-18.md)。三次训练完整对照与结论见 [AMP 实验记录](docs/amp_experiments_2026-09-17.md)，代价/步态详情见 [策略运行期排查](docs/sim2sim_policy_runtime_2026-09-17.md) |
| AMP-08 | P0 | **已修（测试锁定）；影响此前所有 AMP/PPO checkpoint 的续训** | `amp_on_policy_runner.py` 与 `ppo_on_policy_runner.py` 的 `learn()` **循环内从不更新** `self.current_learning_iteration`（只在循环结束后 `+= num_learning_iterations`），而 `save()` 写的是它 —— 于是**循环内保存的 checkpoint 里 `iter` 恒为该次运行的起点（通常 0）**，只有最后一次保存是正确的。实测证据：运行 `2026-09-17_17-19-04` 的 **20 个 checkpoint 全部 `iter=0`**。`load()` 又用 `loaded_dict['iter']` 作续训起点 ⇒ **`--resume` 会从第 0 轮重跑**，且第一次保存（`it=0`）覆盖 `model_0.pt`、此后每 `save_interval` 覆盖同名文件，等于把已训轮次全部作废。HIM runner（`him_on_policy_runner.py:199`）本来就有这一行，故不受影响 | **已修**：两个 runner 的循环首行加 `self.current_learning_iteration = it`；新增 `test_runners_keep_checkpoint_iter_in_sync` 断言三个 runner 都有该行且 `save()` 写 `'iter'`（8 项测试通过）。**未验证**：修复后尚未产生新的 checkpoint（需一次新训练或续训才能确认 `iter` 写入正确） |
| AMP-09 | P1 | **两套任务已分开，待训练验证** | 2026-09-22 用户把 AMP 入口收缩为两套：`flat-amp-height` 继续使用可部署的 45 维 actor；`flat-amp-fanziqi` 严格采用参考的 42 维 actor（同时移除基座线速度和角速度），关闭非参考持续外力，并恢复参考 runner 的 std 下限、500000 轮和 50 轮保存配置。旧 45 维 checkpoint 与新 42 维任务不兼容 | 在训练机分别做两套 256 环境／100 轮冒烟；Fanziqi 新训完成后必须单独导出并建立自己的部署观测契约，不能覆盖当前 45 维部署策略。详见 [AMP 配置收缩记录](docs/amp_config_cleanup_2026-09-22.md) |
| MODEL-02 | P0 | 已完成并验证（①–⑤） | 用户决定把模型统一到 `imgo2_description`（D1 命名 FL/FR/RL/RR；D2 Gazebo/IMU/transmission 留在 description 作可选模块；D3 生成物入库；D4 删冗余副本、git 兜底；D5 Gazebo 插件暂不处理）。**① 命名/网格**：`meshes/` 改 FL 命名（指纹 `8dc5b5995a11`，10 文件）。**② 模块化**：`xacro/core.xacro`（物理内核）＋ `{transmission,gazebo,imu}.xacro` ＋ `robot.xacro` 组装入口（开关默认 false）；生成物 `urdf/imgo2.urdf`（纯）与 `urdf/imgo2.gazebo.urdf`。**③ MJCF**：`mjcf/{imgo2.xml,scene.xml}`（训练物理 + 参考求解器/接触块 + `framelinvel`）。**④ 消费者切换**：RL 的 `assets/imgo2.py` 经 `_REPO_ROOT` 指向 `imgo2_description/urdf/imgo2.urdf`；`check_asset_paths.py` 改为按声明解析多个 root 变量；`audit_amp_dataset.py`/`tests`/`inertia_urdf.py` 同步；deploy 新增编译期 `IMGO2_MODEL_DIR`、`rl_sim_mujoco.cpp` 读 `imgo2_description/mjcf/<scene>.xml`；`build.sh` 与两个 Gazebo launch 改指 description；`CMakeLists` 安装列表去掉 `robot_description`。**⑤ 清理**：删除 `imgo2_model/`、`imgo2_rl/source/imgo2_rl/data/`、`imgo2_deploy/robot_description/`（共 46 文件、约 57 MB 工作树）；AGENTS.md 模型章节改写为单一源规则 | **验证**：`check_model_sync.py` 全 PASS（2 份已登记 URDF、网格指纹、mesh 引用存在性、FK 恒等 RMSE 0.00214 m / 交换 0.22470 m）；`check_amp_joint_order.py` 三项全 PASS 且数值与统一前一致（0.00107/0.00214 m、0.22491、0.0191/0.1339 rad、r 0.9546/0.2317）；`check_asset_paths.py` PASS（URDF 17 mesh 引用全在、21 份动作）；单元测试 5 通过 + 1 跳过（跳过的是需要 torch 的 AMP-03 回归）；`bash build.sh -mj` 删目录后仍构建成功，二进制内 `IMGO2_MODEL_DIR` 指向 `<repo>/imgo2_description`；`git ls-files -i -c --exclude-standard` 为空。**未运行**：GUI、Isaac Lab 训练/回放、ROS/Gazebo、真机；Gazebo 插件 `liblegged_hw_sim.so` 仍缺（DEPLOY-05 未解）。详见 [sim2sim 记录](docs/sim2sim_amp_2026-09-17.md) 第 6.6 节 |
| MODEL-03 | P0 | 已修并在无头回放验证，GUI 待确认 | MuJoCo 的 `framequat`/`framelinvel` 原先写成 `objtype="body" objname="base"`，返回值会再乘上该 body 的**惯量主轴旋转** `iquat`（base 的 `fullinertia` 主轴非 `ixx<iyy<izz` 排列，实际偏移 ≈180° 绕 (1,0,1)/√2），于是部署侧姿态观测整体偏转 90°：`GetUp` 结束后机器人明明直立（`xquat=(1,0,0,0)`），`gravity_vec` 却报 `(-1.000, 0.004, -0.025)` 而非 `(0,0,-1)`，策略把"站直"判成"已翻倒"——PPO 接管瞬间就输出饱和动作并塌成深蹲（`z_final=0.154`），himloco 侧倾，只有 AMP 恰好鲁棒。判据：`QuatRotateInverse(·,(0,0,-1))` 的第三分量恒为 `-1-2q_z² ≤ -1`，不可能出现 `-0.025`，故读到的三元组并非 `(g_x,g_y,g_z)`；再对照 `xquat` 与 `sensordata[36..39]` 的时间线确认是固定偏移。参考项目用的是 site（`objtype="site" objname="base_site"`）。修法：三个传感器改挂 base 内原有的 `imu` site（site 无 `quat`、`pos` 默认原点，故 site 系 = link 系），**声明顺序不变**，`sensordata` 偏移与 `GetState` 的 `[3n..3n+3]`/`[3n+4..3n+6]` 均无需改动；同时注释掉文件末尾那条陈旧 `<keyframe>`（`base z=0.35`，C++ 不应用但 GUI 的 Key 下拉框会应用） | 已通过：修后三个键的 `gravity_vec` 均为 `(0,-0,-1)`；PPO `vx=0/0.5/1.0` 站姿 0.321、实测 0.509/0.975 m/s（FL_thigh 极差 1.1–1.4 rad，是真步态）、AMP 站姿 0.3015；`check_model_sync.py`／`check_asset_paths.py`／`check_amp_joint_order.py` 仍全 PASS，单元测试 5 通过 + 1 跳过（跳过的是需要 torch 的 AMP-03 回归）。细节见[策略运行期排查](docs/sim2sim_policy_runtime_2026-09-17.md)。**待办**：在 GUI 里人工确认一次按键 1/2/3 与 1→2→3 切换 |
| DEPLOY-08 | P1 | 已归因：harness 假象，真实代码无此问题；GUI 仍待确认 | 2026-09-17 加 PPO（键 1）后，用**多键连续切换**的临时 harness（0→1→2→3）在进入 himloco 后崩溃：`mat1 and mat2 shapes cannot be multiplied (1x45 and 270x128)`，即把 45 维（单帧）输入喂给了 himloco 的 270 维（6 帧历史）网络。**2026-09-17 晚查明是我那版 harness 的错**：它的 `Forward()` 被简化成 `model->forward({ComputeObservation()})`，漏掉了真实 `RL_Sim::Forward()` 里的历史分支（`history_obs_buf.insert` → `get_obs_vec(observations_history)`）。把 `Forward()` 逐行照搬真实实现后，单键与 `0→1→2→3` 连续切换**都不再崩溃**；同一轮还确认仓库里 `RL_Sim::Forward()` 与参考项目逐行相同，故真实部署代码没有这个缺陷 | 用**真实 GUI 依次按 1→2→3** 再确认一次（预期不崩；会看到 himloco 把机器人掀翻，那是 DEPLOY-01/DEPLOY-06 的占位策略问题，不是崩溃）。若仍崩，再查 `RL::InitRL` 在「无历史配置 ↔ 有历史配置」切换时 `history_obs_buf` 与 `params["observations_history"]` 的时序 |
| DEPLOY-01 | P0 | 待对齐 | Go2 占位策略与 Imgo2 训练配置存在默认姿态、PD、限幅、指令缩放等差异（指 himloco 占位；AMP 已于 2026-09-17 对齐，见 §5.3；PPO 也已接入，见 §5.4） | 替换为来源明确的 Imgo2 策略，完成训练端与部署端同输入输出比较 |
| DEPLOY-02 | P0 | 场景已对齐并验证站起/行走，待 GUI 验证 | `imgo2_description/mjcf/{imgo2.xml,scene.xml}`（经编译期 `IMGO2_MODEL_DIR` 读取）：关节轴/限位与训练侧逐项一致、MuJoCo C API 可加载、`build.sh -mj` 通过；2026-09-17 按参考项目补齐初始高度（`base pos 0 0 0.5`）与求解器/接触块（`cone=elliptic impratio=100`、关节 `damping=1 armature=0.1`、碰撞 `condim=3 solref="0.005 1"`+friction），不再弹飞；同日又修 MODEL-03（姿态/线速度传感器由 `body` 改挂 `imu` site）并注释掉陈旧 keyframe，修后 PPO 站姿 0.321、`vx=0.5/1.0` 实测 0.509/0.975 m/s，AMP 站 0.3015 | 在有可用显示的机器上跑 `rl_sim_mujoco imgo2 scene` 并保存窗口记录。参考项目那份 MJCF 现在网格名已能对上（FL 改名的副产物，实测可加载），但它的 base 质量仍是旧值 6.53394、用 mesh 碰撞体且没写 `timestep`（=2 ms），同一个参考策略在它上面反而站不起来（`z_final=0.0771`），**不要改用它** |
| DEPLOY-03 | P0 | 缺依赖，待适配 | SDK2 目录为空，真机目标被跳过 | 依赖到位、目标生成、通信接口验证通过 |
| DEPLOY-04 | P1 | 已解决 | 原 `library/thirdparty/joystick/` 为空且未跟踪，`CMakeLists.txt` 在 `USE_MUJOCO` 下要求 `joystick.cc`，`rl_sim_mujoco.hpp` 还 `#include "joystick.hh"`；原无脚本会下载它。2026-09-17 已补入 `joystick.cc`/`joystick.hh`（与参考项目同源） | 已通过：2026-09-17 `bash build.sh -mj` 配置与编译均成功 |
| AMP-07 | P0 | 已定位差异（待改配置重训验证；**2026-09-18 更正了一处单位误读**） | AMP 策略「站着不动」的对照排查（见 [AMP 对照报告](docs/amp_standstill_diagnosis_2026-09-17.md)，三方逐项对照见该文第 11 节）。**① `clamp(min=0)` 不是差异**：两份参考（`~/RL/isaac/AMP/AMP_for_hardware-main`、`amp_go2-main`）的风格奖励公式与我们逐字相同、都有地板，`d` 为裸 logit。**② 判别器 loss 也相同**（MSE±1、`0.5(expert+policy)`、grad_pen 只对 expert λ=10、共用 Adam lr 1e-3 + trunk/head 不同 weight_decay、`min_std` clamp）。**③ 原「任务奖励量级差 50 倍」作废**：两份参考的 `_prepare_reward_function()` 都会做 `reward_scales[key] *= self.dt`（a1 `legged_robot.py:705-710`、go2 `legged_robot_amp.py:691-696`），Isaac Lab 也乘 dt，所以 a1 配置里的 50/16.7 是**原始权重**、每步真值是 **1.5 / 0.5**，我们是 **1.0 / 0.3**——同量级（差 1.5 倍）。**唯一量级离群项是 `base_height_l2`：我们每步 −10，a1 没有这一项，go2 是 −0.02**；再加上终止只认 `base` 触地（a1 还把 thigh/calf 纳入终止），趴着滑在我们的可行域里没有任何惩罚。速度误差 1.65 m/s 时 `exp(-(1.65/0.5)^2)≈2e-5` 梯度≈0，于是只剩高度项的梯度把策略按在「精确站 0.30 m」的局部最优 → 风格分近乎常数（风格≈0.467/步）→ PPO 优势≈0 → 1000 轮后全指标变平。**⑤ 探索与参考逐字相同**（`init_noise_std 1.0`、`min_normalized_std [0.05,0.02,0.05]*4`、`entropy_coef 0.01`），AMP 观测组成也相同（均不含 commands、43 维）。另两条已离线排除/新增：**专家数据速度是体系、不存在坐标系域差**（差分+逆旋转复核）；**录制数据实际只到 0.83 m/s、0.75 rad/s，而指令给到 1.5 / 1.57**，风格参考与任务指令互相打架 | 修正后优先级（第 11.3 节）：**0** 先确认 Run1/Run2 的单位（`50/17/−1` 是原始权重还是每步值，两者结论相反）；**1** 低姿态终止（thigh/shank 纳入 `illegal_contact`，a1 做法）；**2** 高度项 −10→0，且必须与第 1 条配对做；**3** `tracking_sigma` 0.25→1.0（σ 0.5→1.0，对参考的主动偏离）；**4** 指令范围压到数据覆盖内（x ±0.9、yaw ±0.8）；**5** `max_iterations` 提到 ≥30000（参考 checkpoint 是 36450 轮，我们 9496 轮基本跑满 10000 上限）；**6** 可选去掉/降权 `imgo2_stance.txt`；**7** 独立项：std 上限、奖励尺度归一化、判别器饱和、AMP-05 按 go2 的 45 维 actor 新训。改后重训并按「高度/速度误差/关节幅值是否随训练变化」复查（需训练环境，本机无 GPU） |
| PPO-01 | P1 | 已按四项指标选定（`base_move/policy_flat.pt`） | PPO 权重此前用过 `imgo2_flat/2026-06-21_23-24-09` 与 `imgo2_rough/2026-06-21_16-37-38` 两份训练导出，用户反馈「腿部抖动很厉害」并给出评估标准：**位姿 / 速度跟随 / 抖动 / 周期性**。新增 `imgo2_deploy/scripts/eval_gazebo_policy.py`（自注入 /joy，从 /odom + /joint_states 统计四项）并对 5 个候选实测：参考项目 `base_move/policy_flat.pt`（6/30）vx=0.5 时 z 0.288、roll 2.5°、yaw 漂 −0.5°/s、抖动(大腿) 45 rad/s²、周期强度 0.95、FL-FR 175°、速度 0.428 m/s；`policy.pt`（6/30）速度更准 0.522 但抖动 79、roll 5.0°、强度 0.74；我们的 `rough/16-37-38` 抖动 241/507、roll 13.4°、强度 0.29、yaw 漂 −2.7°/s（vx=0 时还漂 +4.8°/s）；`policy1.pt` 超调 0.662；`amp/policy.pt` 不走（0.057 m/s、z 0.182）。⇒ 采用 `policy_flat.pt`（配置 25/0.5、默认 0/0.87/-1.82、clip ±3） | 已装并复测；后续若要速度精度可换 `policy.pt`（同配置）；新训练版本可用同一脚本按四项指标对比。详见 Gazebo 记录第 9 节 |
| JOINT-01 | P0 | ROS2 路径已修并验证；真机路径待处理 | `joint_mapping` 的语义是「策略第 i 个关节 ↔ 该路径数组第 joint_mapping[i] 号」，但三条路径的数组不同：MuJoCo 是 MJCF 声明顺序（FL,FR,RL,RR=策略顺序，恒等正确）；ROS2 是消息槽位，顺序 = 控制器 `joints` 参数 = `base.yaml` 的 `joint_names`（原为 Unitree SDK 的 FR,FL,RR,RL）；真机是 SDK 电机数组（固定 SDK 顺序）。于是恒等映射在 MuJoCo 对、在 ROS2 把策略 FL 接到物理 FR：Gazebo 里按 1 进 PPO 会翻成四脚朝天（实测 `/odom` z=0.073 m、roll=180°、dx=0），给速度指令后更明显。**修法**：① `base.yaml` 的 `joint_names`/`joint_controller_names` 改为模型顺序；② `rl_sim.cpp` 的 ROS2 分支新增 `OrderJointsByModelOrder()`，按 URDF 声明顺序重排传给控制器的名单（读不到则回退），MuJoCo 路径代码与语义未动。**验证**：Gazebo `vx=0` z 0.328–0.332 m、roll ±2.9°、dx≈0；`vx=0.5` 向前 8.20 m/12 s（0.56 m/s）、z 0.26–0.27、大腿摆幅 0.99–1.08 rad（修前四脚朝天）；IMU 侧 `/imu` 角速度与 `/odom` yaw 速率同号同量级，排除 IMU 约定问题 | 真机路径（`rl_real_imgo2.cpp` 索引 SDK 电机数组）要另给映射或改代码；偏航/走偏已确认是策略性质（MuJoCo 同策略 `vx=0.5` 也偏航 −62°/12 s），待换 checkpoint 或用 `axes[3]` 做航向闭环验证。详见记录第 8 节 |
| DEPLOY-05 | P0 | 已解决（无头跑通到策略闭环），GUI 画面待确认 | 原状：部署份 URDF 只有 ROS 1 式 `<transmission>`，`gazebo.xacro` 挂的是仓库内没有的 `liblegged_hw_sim.so`（`gazebo_ros_control`），ROS 2 下没有 `gazebo_ros_control`，于是 `rl_sim.cpp` 的 controller spawner 找不到 controller_manager。**2026-09-17 照参考项目改为 ROS 2 `gazebo_ros2_control`**：`gazebo.xacro` 加 IMU 传感器 + `<ros2_control>`（12 关节 effort 命令 + position/velocity/effort 状态）+ `libgazebo_ros2_control.so`；新增 `imgo2_description/config/robot_control_ros2.yaml`（`joint_state_broadcaster` + `robot_joint_controller/RobotJointControllerGroup`，后者是仓库内插件、由 `rl_sim` 自己 spawn）；`imgo2_description` 做成标准 ROS 2 包（ament + `package.ros1.xml`/`package.ros2.xml` + 软链进工作区 `src/`）；Gazebo 版网格改 `package://`（`robot.xacro` 的 `mesh_prefix`）；`build.sh` 的包扫描改 `find -L`；`check_model_sync.py` 认两种网格写法。**两个实测坑**：`<parameters>` 必须是真实文件路径（`package://`/相对路径都会让插件 Load 抛异常）；URDF 作为 `--param robot_description:=` 传给 controller_manager 时必须压成单行（否则 rcl 报 `Couldn't parse parameter override rule`，控制器起不来）；另外不能用 `gazebo_ros` 自带 launch（本机它起的 gzserver 未加载 factory 插件），改为直接起 `gzserver -s ...`。**验证**（无头、单次调用内）：spawn 成功且网格 0 报错；节点 `/gazebo_ros2_control`、`/imu_plugin` 出现；`ros2 control list_controllers` → `joint_state_broadcaster active`；`/joint_states` 与 `/imu` 发布；`rl_sim` 启动后用 `/joy` 注入 `A`→`RB+DPadUp`，打印 `RL Controller [ppo]`（进入键 1 的 PPO 闭环） | ① 在有可用显示的机器上确认 `gzclient` 画面（本机 `DISPLAY=:1` 建 GL 上下文失败）；步态**已用位姿客观验证**：vx=0.5 时 Gazebo 0.502 m/s、z 0.276 m、大腿摆幅 0.89–1.20 rad，与 MuJoCo（0.51 m/s、0.26 m、0.88 rad）一致，见记录第 7 节；② 长时间跑 himloco/AMP；③ 真机链路仍待验证。详见 [Gazebo 链路打通记录](docs/gazebo_ros2_bringup_2026-09-17.md) |
| DEPLOY-06 | P1 | AMP 路径已解决，himloco 路径仍在 | `rl_sim_mujoco.cpp` 用 `joint_mapping` 直接索引 MJCF，因此配置的映射必须与场景顺序一致。2026-09-17 采用自建场景（关节声明顺序 = 策略顺序 `FL,FR,RL,RR`）+ `base.yaml`/`amp/config.yaml` 恒等映射，AMP 路径自洽 | AMP 已在 16 s 回放中确认基座高度稳定、无 NaN；`himloco/config.yaml` 仍带硬件置换映射 `[3,4,5,0,1,2,9,10,11,6,7,8]`，在 MuJoCo 上会索引错腿，需单独处理；`imgo2_description/mjcf/` 那份现成 MJCF 未再复用 |
| DEPLOY-07 | P0 | 已修，待编译验证 | ROS2 两个控制器把 `std::clamp(…)` 当语句调用、丢弃返回值，限位实际失效；单关节版还把参数名写成 `"robot_description_"`，URDF 从未解析成功。已改为赋值形式并加空指针/越界保护、补 `<algorithm>`、修正参数名 | 在装有 ROS 2 的 Linux 上编译并跑通，确认限位生效且不再有空指针风险 |
| EXPORT-01 | P1 | 待验证 | 导出配置与 C++ 配置格式不同；比较脚本假设六帧历史 | 明确转换规则，记录实际网络维度、历史规则与误差指标 |
| DATA-01 | P1 | 当前一致 | 两处动作数据副本哈希一致 | 每次更新后核对副本，记录数据来源和版本 |
| EXP-01 | P1 | 待补充 | 已记录 PPO 验证与 AMP 贴地现象的用户反馈，尚缺对应日志、命令和模型路径 | 按第 8 节补充真实实验与产物路径 |
| MODEL-04 | P1 | 已修复并验证 | `sorted(Path)` 在 Windows 与 Linux 对大小写排序不同导致同一套网格指纹不同；改为按文件名字符串排序，恢复原预期值，未改网格 | 大小写混合排序与内容篡改回归通过；全模型检查通过，指纹 `8dc5b5995a11` |

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

| 日期 | 变更 | 验证与限制 |
|---|---|---|
| 2026-09-22 | 修复控制台输出被条件吞掉（用户误判为「慢」） | 用户澄清「慢」其实是条件语句导致没有 log 输出。**根因**：`learn()` 里 `if self.writer is not None: self._log(...)`，而 `_log` 内部既写 TensorBoard 又 `print`；`writer` 仅在 `log_dir is not None` 时创建，条件不成立时控制台**一行进度都不输出**（这是我上一轮重构引入的）。**修**：解耦——`_log` 无条件调用，内部只在 writer 可用时调 `_write_scalars`，`print` 一定执行并加 `flush=True`（重定向到文件时按块缓冲同样会掩盖输出）。另在 `learn()` 开头加启动行，说明环境数／总轮数／日志目录，并提示「第一条进度需等首次 rollout 完成」（256 环境约 2.5 分钟静默，容易被再次误读为卡死）。**验证**：桩对象实测 writer 为 None 时仍输出完整进度（修复前该情形零输出），writer 存在时 TB 写入次数不变；用户确认现在可直接运行。详见 [观测指标记录](docs/towing_observability_2026-09-22.md) |
| 2026-09-22 | 消除每物理步的 PhysX 回读（性能） | 用户报告启动/运行慢。用历次 tfevents 拆开：启动 256 env 约 135 s；每轮拟合出**约 4.6 s 固定开销**，两段独立拟合（4→256 env 与 256→4096 env）得到同一值，说明与算力无关。**来源**：`ActionManager.apply_action()` 每物理步调用（每 5 ms），而 `_apply_towing_physics` 每步都 `get_masses()`／`get_inertias()` 回读 PhysX（`_body_properties` 内还有一次），4096 env 下每轮各约 **196 万次**设备往返。**修**：新增 `_refresh_mass_cache()` 缓存机器人／小车质量与对角惯量、四轮惯量；`reset_towing_episode` 在 `set_masses`／`set_inertias` 后调 `_invalidate_mass_cache()`（质量每回合随机化，不失效会用错值）；`_body_properties` 增加 `inertia_diag`／`mass_total` 参数。注意 `inverse_inertia_world` 依赖姿态，**不能**缓存，只缓存常量部分。**验证**：新增 `test_per_step_physics_does_not_readback_physx`（AST 断言热路径无回读、缓存失效顺序在 set_masses 之后），两组负向测试确认会失败；拖曳测试 216 通过、契约 25 通过。**实测结果（用户反馈）**：**无可观测提升**——故「每轮 4.6 s 固定开销」的来源判断未获证实（回读确实冗余、已消除且比原实现更正确，但不是主因；「慢」实为下述输出被吞的误判）。**教训**：不应由拟合出的固定开销直接断定瓶颈。详见 [观测指标记录](docs/towing_observability_2026-09-22.md) |
| 2026-09-22 | 修碰撞传感器 filter 通配（碰撞判据静默失效）与 USD 缓存路径 | 训练时反复报 `Filter pattern '/World/envs/env_*/Robot/*' did not match the correct number of entries (expected 256, found 4864)`。**根因**：Isaac Lab `ContactSensorCfg` 要求每个 filter 项在每个环境里只解析出**一个** prim，而 5 个车体传感器都用 `filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/.*"]`，每环境展开 19 个 prim（17 link + 根 prim 等）⇒ 4864。**危害**：过滤后的接触上报不工作 ⇒ `force_matrix_w` 拿不到真实接触力 ⇒ `cart_collision` 恒假 ⇒ 碰撞 reward（−50）与碰撞终止全部失效，策略可随意撞车不受罚。**修**：新增 `_robot_link_names_from_urdf()` 从训练 URDF 读 17 个 link 名，`_robot_body_filters()` 逐个构造 filter（5 个传感器共用 `_ROBOT_BODY_FILTERS`）。**顺带修既有路径 bug**：`upper_env_cfg.py` 的 `_USD_CACHE` 原用 `parents[6]`，而本文件在 `tasks/manager_based/towing/` 下、仓库根应为 `parents[7]`（`assets/imgo2.py` 在浅两层，那里是 `parents[5]`），原写法使缓存落到 `<repo>/imgo2_rl/logs/usd/`；已引入 `_REPO_ROOT` 统一。**验证**：新增 `test_collision_filters_resolve_one_prim_per_environment`（AST 剥字符串后断言无 `Robot/.*`、5 个传感器都用显式列表、`_REPO_ROOT` 为 `parents[7]`）与 `test_collision_filter_matches_urdf_link_count`；负向测试确认改回通配即失败；同时修正了原本断言旧通配设计的既有测试。拖曳离线测试 216 通过。**未验证**：未实测 `force_matrix_w` 是否恢复、`found` 是否变为 256 × 17（机器人 17 body 为何匹配 19 prim 仍未解开）。详见 [观测指标记录](docs/towing_observability_2026-09-22.md) |
| 2026-09-22 | 修复 Episode 指标重复写入/重复打印 | 修好键名后用户报告控制台「反复重复」。tfevents 定量核对确认：`21-49-15` 那次每个 `Episode_*` 指标被写 **960 次**（= 20 轮 × 48），同一 step 重复 **12220 条**；其中 48 正好等于 `num_steps_per_env`。**根因**：`episode_infos` 是 rollout 内**每一步**有环境 reset 就追加一份 `infos["log"]`，48 步累积几十份同键字典，而旧实现对它逐份遍历写 TB、又逐份遍历拼控制台字符串。**修**：新增 `_aggregate_episode_infos()` 按键聚合（标量取均值），写 TB 与打印都只用聚合结果；`_log` 拆为 `_aggregate_episode_infos`／`_write_scalars`／`_format_progress` 三个函数，`episode_infos` 全程只遍历一次。**验证**：新增端到端测试 `test_aggregation_writes_each_episode_scalar_once`（真实 SummaryWriter + 48 份同键字典，断言每个 `Episode/*` 恰好写 1 条）与静态测试 `test_episode_infos_are_aggregated_before_logging`，两者均经负向测试确认在旧实现下会失败；拖曳离线测试 214 通过。**未验证**：修复后未实跑，四次历史 run 都在修复前。详见 [观测指标记录](docs/towing_observability_2026-09-22.md) |
| 2026-09-22 | 拖曳训练控制台补耗时/吞吐/ETA，并按 PPO runner 样式统一 | `_log` 此前只把 `Perf/*` 写进 TensorBoard，控制台只有 reward/value/policy/decoder 四个数，长跑时无法判断进度。现改为与 `rl_lab/runners/ppo_on_policy_runner.py::log()` 同一套排版：`Computation`（steps/s + 采集/学习耗时）、`Iteration time`、`Avg iteration`、`Total time`（含小时）、`ETA`（含小时），并逐条打印该轮的 `Episode_*` 统计。**期间实遇并修复**：`start_iter` 是 `learn()` 局部变量，`_log` 取不到，实跑报 `NameError: name 'start_iter' is not defined`（只在跑到 `_log` 时暴露）；已把 `start_iter` 作为形参传入，并加静态回归测试 `test_towing_runner_log_gets_start_iter`（断言形参存在且实参个数匹配，负向测试确认会失败）。**验证**：用桩对象实际执行 `_log` 三种情形（无回合结束／带 Episode 统计／resume）均输出正常；拖曳离线测试 212 通过、契约测试 20 通过。详见 [观测指标记录](docs/towing_observability_2026-09-22.md) |
| 2026-09-22 | 修复拖曳 runner 的回合统计键名，补三个观测量 | 冒烟的 TensorBoard 只有 8 个标量、无任何 reward 分项与终止原因。**根因**：Isaac Lab 把 manager 的回合统计写在 `extras["log"]`（`Episode_Reward/*`、`Episode_Termination/*`，见 `manager_based_rl_env.py:369`），而 `TowingOnPolicyRunner` 只读 `infos["episode"]`（该键在 Isaac Lab 不存在）；官方 rsl_rl runner 是两键都认的（`on_policy_runner.py:226-229`）。**改**：runner 改为同样的 `if "episode" ... elif "log" ...`。**附带查得**：`RewardManager.compute()` 对 `weight == 0.0` 的项直接 `continue`，零权重 term 连日志都不产生，故诊断项改用 `1.0e-6` 权重。**新增诊断量**：`cart_present_flag`（无小车比例，期望≈0.875）、`towing_force_norm`（绳力均值）、`towing_force_norm_active`（仅绳力>1 N 的均值，用于区分「全程无拉力」与「有拉力但被稀释」）。**验证**：`compileall` 通过、拖曳离线测试 211 通过、契约测试 19 通过。**未验证**：修复后未实跑，指标是否真的出现在 TensorBoard 只依据主干源码与官方 runner 对照推断。设计与读表顺序见 [观测指标记录](docs/towing_observability_2026-09-22.md) |
| 2026-09-22 | 修正拖曳训练 `--resume` 的 checkpoint 目录逻辑 | 原 `train.py` 每次都用新时间戳建 `log_dir`，恢复出的权重写进新目录，checkpoint 散落在多个 run，而 `get_checkpoint_path` 只在一个目录内搜索，无法接续。**改**：恢复时 `log_dir = os.path.dirname(resume_path)`，checkpoint 写回被恢复的 run 目录、`iter` 连续累加（与 Isaac Lab 官方 `train.py` 一致）。**同时更正先前判断**：上一轮记录里写的「`--resume` 当前不可用」不准确——它能恢复，只是 checkpoint 分散；且此前未逐行核对 `log_dir` 与 `get_checkpoint_path` 的关系就下了结论。**语义陷阱（未改）**：`learn()` 把 `max_iterations` 当「本次还要跑多少轮」而非总轮数目标，恢复时会从 checkpoint 的 `iter` 再跑满 max_iterations 轮，总轮数超出。**验证**：`compileall` 通过、拖曳契约测试 19 通过。**未验证**：修正后未实跑恢复流程 |
| 2026-09-22 | 拖曳上层 RL 首次冒烟通过（用户执行） | 4 环境 × 10 轮跑完，日志 `logs/towing_rl_lab/towing_upper/2026-09-22_21-33-23/`。**验证**：环境构造成功（`decimation=10`、`episode_length_s=10.0`、`device=cuda:0`）；运行时确认 51 维帧契约（actor GRU 输入 56 维、critic 65 维）；checkpoint 含 actor／critic／decoder 三套 GRU、`critic_normalizer_state_dict`、两个优化器；`iter` 由 0 正确推进到 10（AMP-08 修复后首次实证）；decoder 按 checkpoint 权重 strict 加载，单步／序列推理与 `augment_actor_observation` 输出维度均正确。**限制**：TensorBoard 只有 8 个标量，未记录 `cart_present`、绳力、终止原因等拖曳诊断量，因此 12.5% 无小车比例、跨环境隔离、STOP reward 排序三条验收项**无法从本次产物评价**；`mean_reward` 由 −5.5 降至 −29.5 是回合变长（110→160 步）叠加惩罚型 reward 的算术结果，不得读作策略变差。未跑 256 环境、未做 checkpoint 恢复、未做 estimator 精度与 scripted 复现。详见 [冒烟记录](docs/towing_smoke_run_2026-09-22.md) |
| 2026-09-22 | 冒烟实跑定位并修正 `--agent` 语义（用户执行） | 用户在本机（GPU 实测可用：`cuda available: True`／`CUDA solve: OK`）首次跑冒烟，报 `ValueError: Could not find configuration ... entry point: 'rl_lab'`。**根因**：`load_cfg_from_registry` 把 `--agent` 的值**原样当作注册 kwargs 的键**查表，不做后缀推导；我先前的判断「Isaac Lab 由键名去 `_cfg_entry_point` 推导 agent 名」是错的。**修**：`--agent` 默认值改为完整键 `rl_lab_cfg_entry_point`，与注册块的键名逐字一致（与 Isaac Lab 官方入口默认 `rsl_rl_cfg_entry_point` 同语义）。**该错误的证据价值**：错误栈来自 `load_cfg_from_registry`，说明注册块与任务 ID 已生效，第一次实跑已越过任务注册这一关。**验证**：新增 `test_train_agent_default_is_a_full_registry_key`（断言默认值以 `_cfg_entry_point` 结尾且不等于 `rl_lab`），经负向测试确认会失败（写回 `rl_lab` 时两条测试同时失败）；契约测试 19 通过、拖曳离线测试 210 通过。**未验证**：修正后尚未再跑，环境构造／rollout 仍未开始 |
| 2026-09-22 | 训练前置核查（训练机本机，含 `fbb2115`） | 在 `/opt/conda/envs/isaaclab`（Isaac Lab `0.45.9`、RSL-RL `2.3.3`、torch `2.7.0+cu128`）核对基线：`run_isaaclab.sh --check` 通过、`imgo2_rl`／`rl_lab` 可编辑安装齐备、`pybullet` 已在、拖曳离线测试 **210 通过／0 跳过**（本机有 torch，此前跳过项已实跑）、51 维帧＋5 维 estimate＝56 维 actor 契约与 `TowingVecEnvWrapper` 断言一致、冻结 AMP 策略 `config.yaml`／`policy.pt` 齐备。**修**：`train.py` 的 `--agent` 默认值 `towing_rl_lab_cfg`（全仓无对应注册项，必抛 `ValueError`）改为 `rl_lab_cfg_entry_point`（**该值后又经实跑修正**：第一次误改为去后缀的 `rl_lab`，冒烟实跑报 `Could not find configuration ... 'rl_lab'`，证明 `--agent` 必须传完整注册键）。**未运行**：DSH 会话进程内曾观察到 GPU 不可见（按进程求值，不可外推；用户同日实测 GPU 正常，见 TOW-04），故未在运行时解析注册表、未构造环境、未训练（任务注册见下一条记录，为同日后续决定）。另查得 `--resume` 当前不可用（每次新建时间戳目录，checkpoint 恢复会写进新 run），未改。详见 [训练前置记录](docs/towing_training_prep_2026-09-22.md) |
| 2026-09-22 | 注册拖曳上层 RL 任务（用户决定，本轮加入） | 在 `towing/__init__.py` 加入 `gym.register(id="Imgo2-towing-upper-rl-lab", entry_point="isaaclab.envs:ManagerBasedRLEnv", kwargs={env_cfg_entry_point → upper_env_cfg:UpperTowingEnvCfg, rl_lab_cfg_entry_point → agents.upper_ppo_cfg:UpperTowingPPORunnerCfg})`，并在 `train.py` 把 `--agent` 默认值改为 `rl_lab_cfg_entry_point`（**修正经过**：最初改为去后缀的 `rl_lab` 是错的——`load_cfg_from_registry` 把该值原样当注册键查、不做后缀推导，冒烟实跑即报 `Could not find configuration ... 'rl_lab'`；Isaac Lab 官方入口默认值同样是完整键 `rsl_rl_cfg_entry_point`）。**这是显式翻过一道既有保护**：原守门断言与 `docs/towing_upper_rl_review_2026-09-22.md` 第 98 行要求「未完成运行验收前不注册任务，避免把'能 import'误当成'能训练'」；用户为在训练机执行训练而决定注册，故运行验收仍未完成。守门断言已替换为 `test_registry_entry_points_resolve_and_match_the_train_agent_name`（解析注册块、确认两个 entry point 的模块与类真实存在、`agent_key=="rl_lab"`、`--agent` 默认值为完整键 `rl_lab_cfg_entry_point`）与 `test_train_agent_default_is_a_full_registry_key`（专门防 `--agent` 被写回去后缀名字）。**验证**：拖曳离线测试 210 通过（删 1 加 2，总数不变）、`compileall` 通过、tracked-ignore 为空；新增契约测试经两组负向测试确认有效（改错 `--agent` 默认值、改错 agent 注册键均失败，恢复后通过）。**未验证**：本机无 Isaac Sim 运行时，注册表未被真实解析、环境未构造、训练未启动 |
| 2026-09-22 | 上层拖曳训练闭环移入仓库 `rl_lab` | 按现有 AMP 模式新增 `TowingOnPolicyRunner`、`TowingVecEnvWrapper`、独立 critic normalizer 和 towing 配置；三套 GRU、detached rollout estimate、done 边界 decoder 更新及联合 checkpoint 已接线，上层配置不再导入外部 RSL-RL runner/config。另修复自有 recurrent memory 把整型 done 当索引、未按环境清 hidden state 的旧问题。训练机基线为 Isaac Lab 2.2.1／RSL-RL 2.3.3；拖曳离线测试 191 项通过／12 项按可选环境跳过，尚未运行 Isaac Lab。详见 [兼容性记录](docs/training_stack_compatibility_2026-09-22.md) |
| 2026-09-22 | 上层拖曳 reward 增加 STOP 后绳力惩罚 | 仅在随机 `stop_time_s` 后计算 `||F_tow||/(||F_tow||+10 N)`，权重 `−1.0`；初始零速站定及无小车环境屏蔽。完整拖曳离线测试 188 项通过／12 项按可选环境跳过；尚需 Isaac Lab scripted rollout 验证其与 clearance／collision 的回报排序，排除快速松绳后追尾的策略捷径 |
| 2026-09-22 | 拖曳采用 dynamics decoder＋recurrent PPO | decoder 改为 `51→128→GRU(128)`，显式预测机器人 `vx/vy`、负载质量和牵引力 `Fx/Fy`；estimate detach 后与原始帧组成 56 维 actor 输入。预测误差移出 reward，质量 loss 由 GT 牵引力大小连续加权。actor／critic 各用独立 GRU。离线拖曳测试 187 通过／12 跳过；runner 与 Isaac Lab 验证仍未完成。详见 [记录](docs/towing_gru_design_2026-09-22.md) |
| 2026-09-22 | AMP 配置收缩为两套 | `amp_rsl_rl_cfg.py` 只保留 `AMPHeightRunnerCfg` 与 `FanziqiAMPRunnerCfg`；只注册平地高度奖励和 Fanziqi 两套 train/play 任务，删除 go2／粗糙 AMP 环境实现与旧测试。Fanziqi 任务改为 42 维 actor、无持续外力、参考 std clamp／训练轮数／保存间隔。AMP 离线测试 14 通过／1 跳过，未运行 Isaac Lab。详见 [记录](docs/amp_config_cleanup_2026-09-22.md) |
| 2026-09-22 | 修复上层拖曳 RL 静态复审阻断项 | 安全 producer、51 维单帧 observation、`reference_command`、零中心 action、per-env reset 和绳模型异步采样已落地；后续已用仓库 `rl_lab` 取代当时基于较新外部接口写入的 PPO model／normalization 配置。复核还补齐车斗／四轮过滤接触，修正 `extra_distance` 初始站定误生效，把冻结 AMP 推理周期修正为 20 ms（50 Hz），并加入默认 12.5% 无小车零负载环境及完整 mask。未运行 Isaac Lab，任务保持未注册。详见 [修复记录](docs/towing_upper_rl_fix_2026-09-22.md) |
| 2026-09-22 | 复审上层拖曳 RL 环境完整链路 | 核对 scene→event→action→冻结 AMP→绳力／轮阻→obs/reward/termination→PPO/decoder，并与本机 Isaac Lab 源码的 ActionTerm、ObservationManager、RewardManager、reset 顺序和 RSL-RL 配置接口对照。确认架构方向合理但存在 5 类训练阻断项，TOW-03 升为 P0；upper-RL 离线契约测试 7 通过／2 因无 PyTorch 跳过，未运行 Isaac Lab。详见 [复审记录](docs/towing_upper_rl_review_2026-09-22.md) |
| 2026-09-22 | 拉取 `cf465a2` 后执行全仓离线检查 | `check_asset_paths.py`、`check_model_sync.py`、`audit_amp_dataset.py`、`check_amp_joint_order.py`、`compileall` 与 tracked-ignore 检查通过；拖曳测试 194 项通过／12 跳过。全量测试为 256 通过／24 跳过／1 导入错误，错误原因是本机 Python 3.14.6 缺 NumPy；PyTorch／Isaac Lab 项也未在本机执行。无代码修复，待补验证见 CHECK-01 与 [详细记录](docs/offline_check_2026-09-22.md) |
| 2026-09-21 | 将 decoder 收缩为独立质量识别奖励 | 取代同日“6 维 decoder／latent 进 actor”的早期设计：actor 直接读取 96 维两帧 history；训练期 decoder 只预测小车质量，prediction 不进 observation，rollout reward 强制 detach，decoder 只在批间监督更新。保留 VIME 信息增益为后续方向。`test_towing*.py`：194 通过、5 跳过；`py_compile`、asset path、model sync 与 tracked-ignore 检查通过。两个 decoder 数值测试因当前解释器无 PyTorch 跳过；尚未接自定义 runner 或运行 Isaac Lab |
| 2026-09-21 | 明确拖曳项目主路线 | 当前主线固定为“上层 RL 训练闭环 → 两帧 history actor 导出 → MuJoCo/Gazebo sim2sim”；真机单列为后续分支，不混入当前验收标准 |
| 2026-09-21 | 实现上层拖曳 event、速度 schedule 与物理 adapter | 每回合先置零站定、随机速度牵引、4–6 s 时再次置零；随机质量／惯量、摩擦、轮阻、小幅位姿和两类绳 1:1 分配。轮阻与绳力进入每 5 ms 更新，底层策略每 20 ms 推理并保持目标。真实间隙／碰撞 producer 仍缺；未运行 Isaac Lab |
| 2026-09-21 | 上层 observation 加入 IMU 并缩短 encoder 历史 | 单帧加入机体系角速度与重力投影后为 48 维；history encoder 改用 2 帧／96 维，计划输出 32 维 latent 与当前帧组成 80 维 actor 输入。decoder head 与标签仅训练期使用，encoder 随 actor 部署。尚未接自定义 PPO、未运行 Isaac Lab |
| 2026-09-21 | 上层 action 扩为三维并精简 reward | command/last_action/action 统一为 `[x,y,yaw]` 三维；v0 reward 保留持续速度跟踪、碰撞／跌倒、clearance、extra-distance 与 action-rate，其他量先只记录。碰撞改用独立接口，停车距离按停止沿缓存；未运行 Isaac Lab |
| 2026-09-21 | 增加上层策略的负载状态 decoder 设计与网络骨架 | 新增历史 encoder、6 维 decoder、归一化目标与分项 loss；真实间隙作为动态监督目标，质量／摩擦／轮阻作为可辨识性探针。尚未接 rollout/optimizer，未训练；摩擦是否可辨识必须用未见工况验证 |
| 2026-09-21 | 开始构建上层拖曳 RL 环境 | 新增纯逻辑契约、ManagerBased class 配置、分层 action term、obs/reward/termination 与 PPO 初始配置，并维护 `docs/paper_plan_rl.md`。当前 6 项离线契约测试通过；绳力／轮阻、双频控制和事件未完成，未注册任务、未运行 Isaac Lab |
| 2026-09-21 | 新增拖曳能力／停止边界扫描脚本 | 默认 5 个速度 × 5 个质量 × 2 个轮阻 = 50 case，支持可选地面摩擦和两类绳分别扫描；5 项离线测试与默认 dry-run 通过，未启动 Isaac Sim。见 [扫描记录](docs/towing_boundary_scan_2026-09-21.md) |
| 2026-09-21 | 收束拖曳文档 | 将小车 P1/P2、绳索 P3、拖曳 P4 和早期上层设计等 8 份过程文档合并为 [拖曳仿真验证](docs/towing_simulation_validation.md)、[RL 计划](docs/paper_plan_rl.md) 与 [边界扫描](docs/towing_boundary_scan_2026-09-21.md) 三个入口；保留最终数据、限制和待办，删除逐次调试过程。未重跑仿真 |
| 2026-09-17～18 | 打通模型、训练导出与仿真部署链路 | 模型收敛到 `imgo2_description/`，完成模型同步检查；AMP 45 维策略完成正式导出与 Gazebo 闭环验证。训练配方与步态差距仍以对应 `docs/` 记录为准 |
| 2026-09-15～17 | 完成仓库结构与核心接口审查 | 统一模型物理参数、资源路径、关节映射和训练／部署观察契约；遗留问题保留在 §7，不在维护记录重复展开 |
