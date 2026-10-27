# joint_pos_penalty复审与四髋角度软约束设计（2026-10-27）

> 后续同日复核：本文的-0.1是设计时/8400历史快照；用户当前工作树已改为-0.5，未更改全部12关节的选择。运动10°只是未验证的候选转折，横移/转向不应固定套用；后续建议先弱运动正则、再用正常运动角度分布定范围。公开RL/MPC对照及当前数值见[同日补充](ppo_hip_mpc_public_comparison_2026-10-27.md)。本文候选函数仍未接入生产配置。

## 1. 现有函数实际计算什么

源码在`velocity/mdp/rewards.py:110–132`；公共参数在`velocity_env_cfg.py:446–455`；当前rough仅覆盖weight=-0.1，没有将该项筛为髋，所以它仍选全部12关节。

```python
def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward
```

设误差e=q实际-q默认，当前直立时：

```text
P_all = sqrt(sum(e_j ** 2))       # L2范数，既非L1，也非平方和
m = 1    当 命令三维范数>0.1 或 实际平面速度>0.5m/s
m = 5    其他情况
r_step = -0.1 * m * P_all * upright_gate * 0.02
```

它鼓励整体姿态靠近默认，但**没有明确的每髋容许角度范围**，也不是关节越限项。仅把此项权重放大会连大腿/小腿正常摆动一起惩罚；其他关节偏差较大时，L2根号也使髋的独立影响与它们耦合。`stand_still_scale`不等于按精确静站标签放大：命令=0但实际漂移超过.5m/s时反而不放大；命令范数还把平移与yaw分量放在一起。

原函数另乘`clamp(-projected_gravity_z,0,.7)/.7`，直立时1、明显倾倒时减弱；不能把它当一直生效的硬角度安全约束。current joint_pos_limits是另一个未启用项，即使开启也通常是靠近模型软限位才罚，不会自动要求髋接近0。

## 2. 推荐独立项：范围内轻罚，超出范围加重

当前四髋默认角都是0，因此优先保持这个参考，基于**实际测得的髋角相对默认角**计算：

```text
e_i = q_hip_i - q_default_hip_i
b = 5°（零指令），10°（有运动指令）
s = 10°
P_hip = sum_i[ 0.05*(e_i/s)^2 + (max(|e_i|-b, 0)/s)^2 ]
r_step = -0.2 * P_hip * 0.02
```

- 第一项轻度回正，允许小幅平衡调整。
- 第二项只在超出容许范围后额外增长，持续20°偏转比5°偏转付出明显更大代价。
- 只选四髋，大腿/小腿正常摆动不会直接影响它。
- 按指令判定零指令组：平移模长<=.05m/s **且** yaw绝对值<=.05rad/s。实际机器人失控漂移时不自动放宽范围。阈值分开避免混合不同物理单位。
- 本候选不乘直立度门控；倾倒不能靠该门控把此髋项关掉。它仍不是倒地终止或硬安全限制。

5°/10°、scale10°、centering_ratio=.05和weight=-.2均是平地低速首轮对照建议，不是已经验证的生理解剖/最优限度；横移、转向和复杂地形可能需要更大范围。这里的“容许范围”是惩罚曲线转折点，不是禁止越过。需要大动作以避免跌倒时策略仍可权衡收益作出该动作。

## 3. 数值例子

假设只有两个后髋分别向内偏同样角度，前髋为0。下表是**乘weight后、未乘公共dt**的贡献（越接近0越好）：

| 两后髋各偏转 | 新项零指令 | 新项运动指令 |
|---|---:|---:|
| 0° | 0 | 0 |
| 5° | -.005 | -.005 |
| 10° | -.120 | -.020 |
| 20° | **-.980** | **-.480** |

每步实际贡献再乘.02，例如20°零指令时为-.0196。旧joint_pos_penalty在其他关节都为默认、两后髋各20°、直立时，静止分支约-.2468、正常运动分支约-.0494（未乘dt）；旧函数在命令恰为.1且实际速度不超过.5时仍归静止放大分支，不能误用“forward”这个测试标签认定走了正常运动分支。机器证据保留实际指令。

新项与之前建议的hip L1虽可使用相同weight数字，输出尺度不同，**不能直接当等价权重**。首轮二选一：用此新项时，不同时开启额外的hip joint_deviation_l1，原全关节joint_pos_penalty=-.1可保留作基线，监控与stand_still的叠加；不要把同一姿态偏差重复加很多强项。

## 4. 可接入的候选函数（仅设计，尚未写入生产rewards.py）

当前rewards.py已经import torch；该函数默认单位全部为rad。它只返回非负cost，权重与dt留给RewardManager：

```python
def hip_angle_soft_constraint(
    env,
    asset_cfg,
    command_name="base_velocity",
    stand_tolerance=0.08726646259971647,
    move_tolerance=0.17453292519943295,
    error_scale=0.17453292519943295,
    centering_ratio=0.05,
    linear_command_threshold=0.05,
    yaw_command_threshold=0.05,
):
    """Return a nonnegative cost; RewardManager supplies a negative weight and dt."""
    if error_scale <= 0 or stand_tolerance < 0 or move_tolerance < stand_tolerance:
        raise ValueError("Invalid angular scales")
    if centering_ratio < 0 or linear_command_threshold < 0 or yaw_command_threshold < 0:
        raise ValueError("Invalid penalty or command thresholds")
    robot = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    standing = (torch.linalg.vector_norm(command[:, :2], dim=1) <= linear_command_threshold) & (
        torch.abs(command[:, 2]) <= yaw_command_threshold
    )
    tolerance = torch.where(standing, stand_tolerance, move_tolerance)
    error = robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    excess = torch.relu(torch.abs(error) - tolerance[:, None])
    return torch.sum(
        centering_ratio * torch.square(error / error_scale) + torch.square(excess / error_scale),
        dim=1,
    )

```

若后续实施：

1. 在`mdp/rewards.py`加入此函数，保持旧joint_pos_penalty实现供其他任务使用。
2. 在公共RewardsCfg新增默认weight=0的RewTerm，func指向新函数，asset_cfg仅选`SceneEntityCfg("robot", joint_names=[".*_hip_joint"])`，command_name为base_velocity。
3. 在目标PPO低速配置中、disable_zero_weight_rewards之前开启该项weight=-.2；不要全局默认启用。上一方案的对角mirror先排除hip，避免同时施加不匹配的同号角度同步目标。
4. 记录四髋实际角分布、超范围比例与各方向跟速/站姿。若未来需要不同左右/前后范围，应明确逻辑外展符号和实际运动覆盖，再扩展本候选；不先改URDF轴/物理限位。

此函数是**实际髋角软约束**，无法直接防止Actor原始mu=-78或std=22。它们经过动作裁剪后可能形成相同的物理姿态；算法分布边界/原始均值正则仍需单独处理。该项也不保证PPO入口目标连续，部署交接仍是独立工作，见 [站姿接管方案](ppo_stand_entry_plan_2026-10-27.md)。

## 5. 本轮验证与限制

[机器可读证据](hip_angle_constraint_design_2026-10-27.json)保存旧函数原源码、候选函数完整文本、参数、21组角度/指令样例与每步贡献。使用本机env_isaaclab Python、CPU Torch；通过AST提取实际joint_pos_penalty并执行，候选函数同样独立执行，不启动IsaacSim。

已验证：默认角cost=0；两髋20°时零指令/运动cost分别4.9/2.4；零指令下实际漂移不会放宽候选范围；只动非髋关节不影响候选；旧函数速度>.5会退出静态放大分支。函数和配置的源码核对、文档链接/JSON有效性和git diff --check通过。

**已完成**：现有实现解释、一个有明确角度尺度的候选设计及孤立张量数值检查。**尚未实施/验证**：RewardsCfg注册、Isaac环境加载、真实动态/训练梯度行为、短训后的髋姿态与跟速、静站/横移/转向/粗糙地形的合适范围和weight。没有改动生产奖励、模型、策略、PD或部署控制，没有启动训练/硬件。因此不能把此设计称作后髋饱和已修复。
