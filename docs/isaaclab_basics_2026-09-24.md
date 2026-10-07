# Isaac Lab 仿真入门：结构与运行链路

核对日期：2026-09-24。依据为当前工作区源码与 Isaac Lab 2.2 系列官方教程；仅静态阅读，未启动仿真、训练或执行安装。本轮开始前 `git pull --ff-only` 返回 Already up to date；保留所有已有未提交修改。

## 1. 最小仿真与训练工程

Isaac Sim 提供场景、物理和渲染；Isaac Lab 封装资产、执行器、传感器、并行场景和环境接口；PPO/AMP 等算法负责根据环境交互数据更新策略。只搭建机器人仿真，不需要训练算法、奖励、checkpoint 或动作数据集。

最小本地机器人示例可以只有一个入口脚本和它引用的模型资源（以下是说明用结构，未创建）：

```text
my_sim/
  run_sim.py       # 启动应用、建场景、加载机器人、控制与步进
  robot.urdf      # 或可直接加载的 USD 模型
  meshes/         # 当模型引用外部网格时需要
```

模型路径和 Python 导入有效即可；不要求将模型搬入 Isaac Lab 安装目录。URDF 的网格引用、物理参数和关节命名必须有效。URDF 加载可通过 UrdfFileCfg 转换成 USD 场景资产；Xacro 先生成 URDF。文件路径与 USD prim_path 是两回事，后者描述场景中的对象位置，例如 /World/Robot。

工程增长后可以拆分为 scripts/（入口）、assets/（模型加载/初始状态/执行器配置）、tasks/（环境配置）、tasks/.../mdp/（观测/奖励/重置等函数）、tasks/.../agents/（算法配置）。这些分类名主要是组织约定；使用 Gym 注册和训练脚本时，要满足脚本期待的配置入口键、导入路径及环境接口。独立脚本不需要 Gym 注册、setup.py 或 extension.toml；可安装包需要打包元数据，使用扩展模板则保留模板所需的扩展配置。

## 2. 本工程对应关系

| 职责 | 当前文件或目录（相对仓库根） |
|---|---|
| 模型资源 | imgo2_description/urdf/ 与 meshes/ |
| 机器人加载、初始关节角、电机参数 | imgo2_rl/source/imgo2_rl/imgo2_rl/assets/imgo2.py |
| 场景、观测、动作、奖励等公共配置 | imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py |
| 机器人专用配置与平地覆盖 | 上述 velocity/ 下 base_move/rough_env_cfg.py、flat_env_cfg.py |
| 任务名与环境/算法配置绑定 | 上述 base_move/__init__.py |
| 算法超参数 | 上述 base_move/agents/ |
| 自有算法实现 | imgo2_rl/scripts/rl_lab/rl_lab/ |
| 无学习算法的环境运行入口 | imgo2_rl/scripts/tools/zero_agent.py |
| 普通 PPO 训练入口 | imgo2_rl/scripts/rsl_rl/train.py（调用外部 rsl_rl） |

assets/imgo2.py 是 Python 配置，不是网格本体。agents/ 是任务对应的算法配置，不等于算法实现。目录名 ppo 也不决定运行算法；由入口脚本和注册配置共同决定。

当前未提交的 assets/imgo2.py 默认读取 imgo2_description/urdf/imgo2_real.urdf，IMGO2_URDF_PATH 可覆盖；文件存在。README 的历史 imgo2.urdf 描述不能代表这一工作区实际默认值。

## 3. 从启动到步进

独立仿真：AppLauncher 启动应用 → 导入依赖运行时的模块 → SimulationContext 设置物理 → 创建地面/机器人 → reset 初始化 → 设置关节目标 → write_data_to_sim → sim.step → 更新读取机器人状态 → 循环。

zero_agent.py：AppLauncher → import imgo2_rl.tasks 触发注册 → parse_env_cfg(task) → gym.make(task, cfg) → env.reset → env.step(零动作)。它不训练，但仍构建完整 RL 环境，会执行环境自身的观测、奖励和重置逻辑。

普通平地 PPO 配置继承链：LocomotionVelocityRoughEnvCfg → Imgo2RoughEnvCfg → Imgo2FlatEnvCfg；子类 __post_init__ 会覆盖父类。当前公共 dt=0.005 s、decimation=4，对应每 0.020 s 仿真时间处理一次新动作，内部推进 4 个物理步。其他任务可能覆盖这些参数。

当前普通 PPO 动作是默认关节角加缩放后的动作，再经过限幅与执行器处理；hip 缩放 0.125，其余关节 0.25（rough_env_cfg.py 覆盖公共配置的 0.5）。零动作意味着默认目标关节角，不代表零力矩，也不保证站稳。执行器根据目标和当前状态产生驱动力，物理引擎再求解接触与运动。

训练入口还会包装环境、创建 runner，重复策略推理、环境步进、轨迹收集和参数更新。play 则加载已有策略做推理，通常不更新网络参数。

## 4. 已修复与待确认

- 已修复：无运行代码修复；已补充最小结构、配置/算法职责和实际源码对应关系。
- 静态核对：上述文件存在；已阅读注册、继承、动作与步进配置。文档相对链接用 /usr/bin/python3 检查；不作为仿真验证。
- 待确认（README LAB-LEARN-01）：当前本地模型改动与已训练 checkpoint 的一致性，需模型同步检查及对应策略回放；未执行。
- 待确认（同条）：当前 scripts/rsl_rl/train.py 明确要求 rsl-rl-lib >=3.0.1，而 README 历史基线写 2.3.3。需确认实际解释器的版本与所用入口，再进行短运行；这不代表自有 AMP 入口存在同一问题。本轮不升级依赖或修改代码。

## 5. 官方阅读入口

- [空场景与 AppLauncher](https://isaac-sim.github.io/IsaacLab/v2.2.0/source/tutorials/00_sim/create_empty.html)
- [加载并操作关节机器人](https://isaac-sim.github.io/IsaacLab/v2.2.0/source/tutorials/01_assets/run_articulation.html)
- [Gym 环境注册](https://isaac-sim.github.io/IsaacLab/v2.2.0/source/tutorials/03_envs/register_rl_env_gym.html)
