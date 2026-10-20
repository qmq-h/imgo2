# imgo2_rl 工程结构与运行阅读指南

核对日期：2026-10-10；源码基线：fc8b341。本轮为源码阅读与离线资源核对，没有启动仿真、训练或后台任务。

## 1. 工程分层

- `source/imgo2_rl/`：可编辑安装的机器人任务包。其内第二层 `imgo2_rl/` 才是 Python 包；`assets/` 配置机器人，`tasks/` 注册任务并定义环境。
- `scripts/rsl_rl/`：外部 RSL-RL 的普通 PPO 训练与回放入口。
- `scripts/rl_lab/`：另一个可编辑安装包。`amp/`、`himloco/`、`ppo/`、`towing/` 是入口脚本；内层 `rl_lab/` 放自有算法实现。
- `scripts/rl_lab/rl_lab/{envs,wrapper,runners,algorithms,modules,storage,datasets}/`：依次负责环境扩展、接口转换、采样训练调度、算法更新、神经网络、轨迹存储、参考动作加载。
- `scripts/tools/` 与 `tests/`：检查、评估、回归工具。
- `datasets/imgo2_motion/`：当前 AMP 默认读取的 21 份动作，不要与仓库外层 `imgo2_dataset/` 混淆。
- 机器人模型位于仓库外层 `imgo2_description/`，部署位于 `imgo2_deploy/`。

## 2. 用平地 PPO 贯穿理解

`--task=Imgo2-basemove-flat-ppo` → `base_move/__init__.py` 的 Gym 注册 → `Imgo2FlatEnvCfg` 与 `Imgo2FlatPPORunnerCfg` → 创建 ManagerBasedRLEnv → RslRlVecEnvWrapper → OnPolicyRunner → learn。

环境继承：`LocomotionVelocityRoughEnvCfg` → `Imgo2RoughEnvCfg` → `Imgo2FlatEnvCfg`。必须顺序阅读各自 `__post_init__()`，最终子类会覆盖公共配置。

公共文件 `tasks/manager_based/locomotion/velocity/velocity_env_cfg.py` 定义 scene、commands、actions、observations、events、rewards、terminations、curriculum；`mdp/` 中是函数实现，并复用 Isaac Lab 的 MDP 函数。

当前平地 PPO actor 源码推导为 45 维：角速度 3、投影重力 3、速度指令 3、相对关节角 12、关节速度 12、上一动作 12。critic 保留线速度，共 48 维；运行时应以环境打印为准。粗糙任务 actor 也去掉高度扫描，而 critic 仍保留地形信息。

动作顺序 FL、FR、RL、RR，每腿 hip、thigh、shank。PPO 目标关节角 = 默认角 + scale × 截断后的动作；髋 scale 0.125、其他 0.25。默认角为 0、0.87、-1.82 rad。DCMotorCfg 的 PD 参数为 stiffness=25、damping=0.5，并有力矩／速度限制；策略输出的是关节位置目标的偏移量。

公共物理步长 0.005 s，decimation=4：控制步长 0.02 s，即 50 Hz；episode 默认 20 s，4096 环境并行。PPO 每环境采样 24 控制步后更新，4096 × 24 = 98304 条 transition；这是一轮 iteration，不是一回合。平地 PPO 最大轮数 5000，保存间隔 100，actor/critic 隐藏层 [512,256,128]。粗糙任务配置明确关闭随机推扰和外力、非法接触终止；不能概括成所有任务都在跌倒时立即重置。

## 3. AMP 与 HIM-Loco

AMP 仍用 PPO 更新 actor/critic，额外训练判别器比较参考动作与策略的相邻状态，从而提供风格奖励。AMP Height 当前 actor=45、critic=48；单帧 AMP 特征按源码为 43 维（关节角12、足端位置12、线速度3、角速度3、关节速度12、高度1），它与 actor 的输入不是同一个向量。当前 runner 的混合为 0.7 × 风格奖励 + 0.3 × 任务奖励。还包含参考状态初始化，不是直接回放动作文件。

AMP Height 与 Fanziqi 是不同配方；后者控制步长 0.03 s、actor 42 维，不能随意互换 checkpoint。

HIM-Loco 通过历史观测和 estimator 推断速度／隐变量，再输入 actor；实现见 `him_actor_critic.py`、`him_estimator.py`、`himloco_vec_env_wrapper.py`。拖曳任务是独立的上下层控制扩展，初学基础运动时可后读。

## 4. 运行顺序（在仓库根）

先选定已经安装 Isaac Lab / Isaac Sim / PyTorch 的解释器。下面路径是占位，必须换成实际存在的环境；项目的两次安装也必须使用它。

```bash
cd /home/tommy0929/Documents/imgo2
export IMGO2_ISAACLAB_PYTHON=/实际环境路径/bin/python
"$IMGO2_ISAACLAB_PYTHON" -m pip install -e imgo2_rl/source/imgo2_rl
"$IMGO2_ISAACLAB_PYTHON" -m pip install -e imgo2_rl/scripts/rl_lab
python3 imgo2_rl/scripts/tools/check_asset_paths.py
bash imgo2_rl/scripts/run_isaaclab.sh --check
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/tools/list_envs.py
```

先验证环境生成（此脚本持续运行，用 Ctrl+C 退出；零动作仍产生默认位置目标）：

```bash
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/tools/zero_agent.py --task=Imgo2-basemove-flat-ppo --num_envs=1
```

短训练只检查链路，不能据此判断学会走路：

```bash
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/rsl_rl/train.py --task=Imgo2-basemove-flat-ppo --num_envs=64 --max_iterations=10 --headless
```

AMP Height 对应入口：

```bash
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/rl_lab/amp/train.py --task=Imgo2-basemove-flat-amp-height --num_envs=64 --max_iterations=10 --headless
```

完整训练在前置检查及短训练通过后再按显存确定并行规模和轮数。统一启动器修正动态库搜索路径，不负责安装依赖。

在仓库根运行时，PPO 日志落入 `logs/rsl_rl/base_move_flat/<时间>/`，AMP Height 落入 `logs/amp_rsl_rl/base_move_amp_height/<时间>/`。包含 checkpoint、配置及 TensorBoard 日志；相对日志目录依据当前工作目录，故统一在根启动。

```bash
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/rsl_rl/play.py --task=Imgo2-basemove-flat-ppo --num_envs=1 --checkpoint="/实际绝对路径/model_10.pt"
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/rl_lab/amp/play.py --task=Imgo2-basemove-flat-amp-height-play --num_envs=1 --checkpoint="/实际绝对路径/model_10.pt"
```

无可用显示时加 `--headless`。对应 play.py 自动导出 checkpoint 同级 `exported/policy.pt` 与 `policy.onnx`。回放脚本持续运行，结束用 Ctrl+C。训练 checkpoint 与部署导出件不是同一种文件；部署之前仍需数值一致性、libtorch 加载、观测与动作契约验证。

## 5. 已修复与待确认

已修复（文档）：补齐结构、配置继承与运行链路的学习入口；README §4.1 补当前模型路径说明，避免将历史默认当作当前源码。本轮未改模型或算法。

已验证：`/usr/bin/python3` 执行 `check_asset_paths.py` 退出 0；实际默认 `imgo2_description/urdf/imgo2_real.urdf` 存在，17 处 mesh 引用均可解析，21 份动作存在，无残留机器绝对资源路径。

待确认：默认启动解释器 `/opt/conda/envs/isaaclab/bin/python` 在本机不存在；需要用户真实的 Isaac Lab 环境，才可完成依赖、CUDA、注册枚举、环境生成、短训练、回放和导出验证。模型一致性问题沿用 MODEL-05，本轮资源存在性通过不代表模型一致性通过。版本兼容依据实际环境和现有兼容性记录，不能混用不同版本安装方案。

同步限制：已尝试 `git pull --ff-only`，由于 `.git/FETCH_HEAD` 为只读而未成功；说明基于当前本地 fc8b341，未声称已同步远端。
