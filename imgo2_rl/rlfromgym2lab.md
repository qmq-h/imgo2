# 从 IsaacGym AMP 迁移到 IsaacLab 的 env / wrapper 思考

本文以 `C:\Users\qmq\Desktop\RL\AMP\AMP_for_hardware-main` 中的 AMP 项目为参考，说明为什么把 IsaacGym 里的算法迁移到 IsaacLab 时，通常需要补 `env` 和 `wrapper` 这类对接层。

## 结论

在 IsaacLab 中，`env` 和 `wrapper` 不是算法本身的一部分，但它们通常是把 IsaacGym 风格算法接到 IsaacLab 上的必要胶水层。

原因是：IsaacGym 项目里的环境类本身就直接实现了算法需要的接口；而 IsaacLab 把仿真、任务、观察、奖励、终止、动作等拆成 manager-based 组件，标准环境输出格式也和老的 IsaacGym / rsl_rl 接口不同。因此迁移时需要一个环境子类和一个 wrapper，把 IsaacLab 的 manager-based 语义翻译成算法期望的 VecEnv 语义。

对 AMP 来说，关键接口包括：

```python
get_amp_observations()
step() -> obs, privileged_obs, rewards, dones, infos, reset_env_ids, terminal_amp_states
```

如果这些接口不存在，`AMPOnPolicyRunner` 无法正确获得判别器训练所需的 policy transition 和 terminal AMP state。

## IsaacGym 中为什么不需要额外 env / wrapper

参考 `AMP_for_hardware-main/legged_gym/envs/base/legged_robot.py`，IsaacGym 版本的 `LeggedRobot` 环境本身就把所有算法接口写在环境类里。

它的 `step()` 直接返回 AMP runner 需要的数据：

```python
return policy_obs, privileged_obs_buf, rew_buf, reset_buf, extras, reset_env_ids, terminal_amp_states
```

它也直接提供：

```python
get_amp_observations()
```

并且在 `post_physics_step()` 中，reset 前先保存：

```python
terminal_amp_states = self.get_amp_observations()[env_ids]
```

也就是说，在 IsaacGym 项目中：

```text
环境类 = 仿真 + 任务逻辑 + observation + reward + reset + AMP接口 + VecEnv接口
```

算法 runner 可以直接调用环境对象，不需要再经过适配层。

## IsaacLab 中为什么需要 env

IsaacLab 的 `ManagerBasedRLEnv` 是更模块化的结构。环境标准 `step()` 通常返回：

```python
obs_dict, rewards, terminated, truncated, extras
```

其中 observation 是按 group 管理的，例如：

```text
policy
critic
```

奖励、终止、命令、事件也分别由 manager 计算。这个设计更清晰、更可配置，但它不会天然返回 AMP 需要的：

```python
reset_env_ids
terminal_amp_states
```

所以需要一个 AMP 专用环境子类，例如：

```python
AmpManagerBasedRLEnv
```

它主要做三件事：

1. 在 reset 之前捕获 AMP observation。
2. 返回 `reset_env_ids` 和 `terminal_amp_states`。
3. 提供 `get_amp_observations()` 给 runner 调用。

这和 HimLoco 的情况类似。HimLoco 也需要 reset 前的 critic observation 或 privileged observation，所以也需要自定义 env 来捕获 pre-reset observation。

## IsaacLab 中为什么需要 wrapper

IsaacLab 环境返回的是 manager-based 的字典观察：

```python
obs_dict["policy"]
obs_dict["critic"]
```

而从 IsaacGym 移植来的 AMP runner 期待的是老式 VecEnv 接口：

```python
obs = env.get_observations()
privileged_obs = env.get_privileged_observations()
amp_obs = env.get_amp_observations()
obs, privileged_obs, rewards, dones, infos, reset_env_ids, terminal_amp_states = env.step(actions)
```

两边接口不一致，所以需要 wrapper 做翻译，例如：

```python
AmpVecEnvWrapper
```

它的职责不是改变算法，而是把 IsaacLab 环境包装成 AMP runner 认识的形状：

```text
IsaacLab obs_dict          -> policy_obs / privileged_obs
terminated | truncated    -> dones
reset前AMP状态             -> terminal_amp_states
ManagerBasedRLEnv属性      -> VecEnv属性
```

如果算法还需要历史观察，比如 IsaacGym 里 `include_history_steps`，wrapper 也适合在这里维护 history buffer。

## IsaacGym 和 IsaacLab 的核心区别

### 1. 环境职责不同

IsaacGym / legged_gym 中，环境类往往是大而全的：

```text
LeggedRobot
  - physics step
  - reset
  - compute_observations
  - compute_reward
  - get_amp_observations
  - history buffer
  - terminal AMP state
  - VecEnv接口
```

IsaacLab 中，这些被拆开：

```text
ManagerBasedRLEnv
  - action_manager
  - observation_manager
  - reward_manager
  - termination_manager
  - command_manager
  - event_manager
  - recorder_manager
```

所以迁移算法时，不能简单把 IsaacGym 的 runner 拿来就跑，必须告诉 runner：

```text
policy obs 在哪里？
critic obs 在哪里？
AMP obs 在哪里？
done 怎么算？
reset 前的 terminal state 怎么拿？
```

这些就是 env / wrapper 的工作。

### 2. observation 组织方式不同

IsaacGym 中 observation 常常是环境手写拼接：

```python
self.obs_buf = torch.cat(...)
self.privileged_obs_buf = torch.cat(...)
```

AMP observation 也是手写：

```python
torch.cat((joint_pos, foot_pos, base_lin_vel, base_ang_vel, joint_vel, z_pos), dim=-1)
```

IsaacLab 中 observation 通常由 `ObservationManager` 统一管理。更推荐的方式是定义一个 observation group：

```text
policy
critic
amp
```

如果定义了 `amp` group，`AmpManagerBasedRLEnv.get_amp_observations()` 可以直接读取：

```python
observation_manager.compute_group("amp")
```

如果没有定义 `amp` group，则只能在 env 中从 articulation data 手动拼接。

### 3. reset 时机和 terminal state 处理不同

AMP 判别器训练需要 transition：

```text
amp_obs_t -> amp_obs_t+1
```

如果环境在 done 后已经 reset，那么 `amp_obs_t+1` 会变成新 episode 的初始状态，这是错误的。

IsaacGym 版本在 reset 前保存：

```python
terminal_amp_states = self.get_amp_observations()[env_ids]
```

IsaacLab 的标准 `ManagerBasedRLEnv.step()` 不会自动为 AMP 保存这个状态，所以 AMP env 子类必须在 reset 前捕获它。

这就是 `AmpManagerBasedRLEnv` 存在的最重要原因。

### 4. 算法和环境的耦合方式不同

IsaacGym 项目里，算法 runner 和环境约定通常比较直接：

```text
runner 直接假设 env 有 get_amp_observations()
runner 直接假设 step 返回 terminal_amp_states
```

IsaacLab 更推荐让任务配置和 manager 定义 observation/reward/termination，再通过 wrapper 适配算法接口。

所以迁移时更合理的结构是：

```text
rl_lab/
  algorithms/      # AMP PPO 等算法本体
  runners/         # AMPOnPolicyRunner
  envs/            # AmpManagerBasedRLEnv，负责 IsaacLab step 语义
  wrapper/         # AmpVecEnvWrapper，负责 VecEnv 接口适配
```

## AMP 迁移到 IsaacLab 的推荐流程

### 第一步：迁移算法本体

包括：

```text
amp_ppo.py
amp_discriminator.py
motion_loader.py
motion_util.py
pose3d.py
amp_on_policy_runner.py
replay_buffer.py
rollout_storage.py
actor_critic.py
```

这一层应尽量不改算法逻辑，只改 import 路径。

### 第二步：实现 AmpManagerBasedRLEnv

它应负责：

```python
get_amp_observations()
step(...): return obs_dict, rewards, terminated, truncated, extras, reset_env_ids, terminal_amp_states
dof_pos_limits
dt
```

其中 `terminal_amp_states` 必须在 reset 前获取。

### 第三步：实现 AmpVecEnvWrapper

它应负责把 IsaacLab env 转成 AMP runner 需要的接口：

```python
get_observations()
get_privileged_observations()
get_amp_observations()
step()
```

并处理：

```text
obs_dict -> tensor
terminated/truncated -> dones
history obs 可选维护
time_outs 写入 extras
```

### 第四步：任务配置中增加 AMP observation

最好在 IsaacLab 的 observation cfg 中新增 group：

```text
amp
```

其内容应匹配 expert motion dataset 的 observation 维度和含义。参考 IsaacGym AMP 项目中的：

```python
joint_pos
foot_pos_in_base_frame
base_lin_vel
base_ang_vel
joint_vel
z_pos
```

如果 expert motion file 的 joint 顺序、足端顺序、坐标系和 IsaacLab robot 不一致，还必须显式做 reorder。

### 第五步：任务注册和 train/play 入口

Gym 注册中应使用：

```python
entry_point="rl_lab.envs:AmpManagerBasedRLEnv"
```

训练脚本中应使用：

```python
from rl_lab.runners import AMPOnPolicyRunner
from rl_lab.wrapper import AmpVecEnvWrapper
```

## 是否必须有 env 和 wrapper？

严格说，不是“语法上必须”，但从工程上几乎必须。

可以有三种做法：

### 做法 A：把所有 AMP 接口直接塞进任务环境

这最像 IsaacGym，但会让环境类变得很重，和 IsaacLab 的 manager-based 设计不太一致。

### 做法 B：env 子类 + wrapper

这是比较合适的迁移方式：

```text
AmpManagerBasedRLEnv 处理 reset 前状态和 AMP obs
AmpVecEnvWrapper 处理 runner 接口适配
```

优点是算法本体不用改太多，IsaacLab 任务结构也比较清楚。

### 做法 C：重写 AMP runner，让它原生支持 IsaacLab obs_dict

这是最干净但成本最高的做法。需要把 runner、storage、algorithm 对环境接口的假设全部改掉。

当前阶段更推荐做法 B。

## 当前 rl_lab 中新增 AMP 文件的意义

当前新增：

```text
rl_lab/envs/amp_manager_based_rl_env.py
rl_lab/wrapper/amp_vec_env_wrapper.py
```

它们不是 AMP 算法本体，而是 IsaacLab 对接层。

算法本体仍在：

```text
rl_lab/algorithms/amp_ppo.py
rl_lab/algorithms/amp_discriminator.py
rl_lab/runners/amp_on_policy_runner.py
rl_lab/datasets/motion_loader.py
```

env/wrapper 的作用是让这些从 IsaacGym 项目迁移来的代码能够在 IsaacLab 的 manager-based 环境里拿到同样语义的数据。

## 还需要补充和确认的内容

对照 `C:\Users\qmq\Desktop\rl_lab` 这种标准 IsaacLab 移植项目，当前 AMP 还不只是缺 `env` 和 `wrapper`。要真正训练/回放跑通，还需要补齐下面几类内容：

```text
scripts/rl_lab/amp/
  train.py
  play.py
  cli_args.py
```

这三个入口现在需要从 PPO/HimLoco 的 IsaacLab 脚本改造，而不是继续沿用 IsaacGym 的训练入口。训练入口中应创建 IsaacLab env，再包一层 `AmpVecEnvWrapper`，最后交给 `AMPOnPolicyRunner`。

```text
rl_lab/config/
  amp_algorithm_cfg.py
  rl_cfg.py 导出 AMP cfg

amp_rl_lab_cfg.py
```

标准 `rl_lab` 项目里每个算法都有自己的 runner/algorithm config，例如 `ppo_rl_lab_cfg.py`、`himloco_rl_lab_cfg.py`。AMP 也需要对应配置，把旧工程里的 `runner`、`policy`、`algorithm` 字典参数整理成 IsaacLab 可注册的 config class。尤其是这些字段必须有明确来源：

```text
amp_motion_files
amp_num_preload_transitions
amp_reward_coef
amp_discr_hidden_dims
amp_task_reward_lerp
min_normalized_std
```

```text
source/imgo2_rl/.../tasks/.../__init__.py
```

任务注册也要补 AMP 版本，入口应指向：

```python
entry_point="rl_lab.envs:AmpManagerBasedRLEnv"
```

并注册对应的 `amp_rl_lab_cfg.py`。否则 IsaacLab 的 `gym.make()` 仍然不知道这个任务应该使用 AMP env。

AMP observation 最好在 IsaacLab env cfg 中显式定义一个 `amp` observation group。当前 `AmpManagerBasedRLEnv` 已经提供 fallback，可以从 articulation data 手动拼接：

```text
joint_pos
foot_pos_in_base_frame
base_lin_vel
base_ang_vel
joint_vel
root_z
```

但更稳的做法是在 observation manager 中固定 `amp` group。这样可以避免后续机器人资产、body 名称、joint 顺序变化时，判别器输入悄悄变掉。

还必须重点确认 motion dataset 和机器人模型的顺序一致。旧 AMP 工程的 `motion_loader.py` 里有 `reorder_from_pybullet_to_isaac()`，这是针对原项目 A1 数据顺序写的。迁移到 Imgo2 时要确认：

```text
joint order
foot body order
root quaternion convention
linear/angular velocity frame
motion file 的动作维度
```

如果这些不一致，算法代码能运行，但 AMP discriminator 学到的是错位的数据。

旧 IsaacGym AMP 还有 reference-state initialization：

```text
reference_state_initialization
reference_state_initialization_prob
_reset_dofs_amp()
_reset_root_states_amp()
```

当前 IsaacLab AMP env 只补了 `terminal_amp_states` 和 `get_amp_observations()`，还没有实现从 motion frame 初始化 episode。旧项目中 `a1_amp_config.py` 默认开启这个能力，所以如果要复现原 AMP 训练效果，需要继续把这部分迁移成 IsaacLab reset/event 逻辑。

最后还要确认依赖和数据路径：

```text
pybullet / pybullet_utils
datasets/mocap_motions/*
checkpoint save/load/export
```

`motion_loader.py` 使用了 `pybullet_utils.transformations`，独立 `rl_lab` 包安装时应把对应依赖写进 `setup.py` 或环境说明。motion file 路径也建议改成绝对路径或基于配置文件的相对路径，避免从不同工作目录启动训练时找不到数据。
