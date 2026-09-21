# 上层拖曳强化学习计划

> 最后维护：2026-09-21。本文是上层 RL 环境的实现规格；当前完成的是类结构、接口契约和纯逻辑测试，尚未注册任务或启动训练。

## 1. 任务结构

冻结已有 locomotion policy，上层策略只做 command shaping：

```text
user cmd → upper action → reference velocity → frozen locomotion → joint targets
```

v0.1 只研究直线停止瞬态。上层频率 20 Hz，物理步长 0.005 s；底层策略保持自己的 0.02 s 控制周期。动作和观测由 class-based config 管理，不修改已有 locomotion 环境。

当前主线只有两个连续阶段：

```text
上层 RL 环境与训练闭环
        ↓
策略导出与 sim2sim 拖曳／停车验证
```

真机是独立后续分支。本阶段只保证 observation 和控制频率使用可部署接口，不把 SDK、电机映射、硬件安全或真机实验列入当前完成标准。

## 2. Actor observation

输入全部来自可部署的本体状态，并固定顺序：

| 顺序 | observation | 维数 | 说明 |
|---:|---|---:|---|
| 1 | `cmd_vel` | 3 | 完整用户指令 `[vx, vy, yaw_rate]` |
| 2 | `last_action` | 3 | 上一拍上层动作 `[a_x, a_y, a_yaw]` |
| 3 | `base_ang_vel` | 3 | IMU 机体系角速度，训练缩放为 `0.25` |
| 4 | `projected_gravity` | 3 | IMU 姿态对应的机体系重力方向 |
| 5 | `last_loco_action` | 12 | 冻结底层策略上一拍输出 |
| 6 | `joint_pos` | 12 | 策略关节顺序下的相对位置 |
| 7 | `joint_vel` | 12 | 策略关节顺序下的速度 |

单帧 48 维。actor 直接读取当前与上一拍共 2 帧，即展平后的 96 维；20 Hz 下两个采样点相隔 0.05 s。decoder 的预测和中间特征都不拼入 actor observation，因此部署侧只需要维护同样的两帧缓存。质量、摩擦、小车速度、绳状态、间隙等仿真真值不得进入 actor。

上层 action 与用户 command 都是三维。v0.1 的训练分布仍可先固定 `vy=yaw_rate=0`，但接口和网络不能退化为一维，否则后续转向时 observation/action 契约都要重做。

若 2 帧不足以识别负载，先对比 4 帧消融，再考虑增加历史长度；不能直接用仿真 root velocity 冒充可部署观测。关节目标与实际位置的跟踪残差可由 `last_loco_action` 和 `joint_pos` 构造，后续可显式化以减少 actor／decoder 的学习负担。

## 3. Action

`high_level_velocity` 是三维归一化动作，实际语义是参考加速度／角加速度，不是直接速度：

```text
u = [u_x, u_y, u_yaw] ∈ [-1, 1]³
a_ref = [a_x, a_y, a_yaw]
cmd_ref[t+1] = clip(cmd_ref[t] + a_ref * 0.05, limits)
```

初始范围：`a_x∈[-1,0.5] m/s²`、`a_y∈[-0.5,0.5] m/s²`、`a_yaw∈[-1,1] rad/s²`；参考指令限制为 `vx∈[0,1] m/s`、`vy∈[-0.3,0.3] m/s`、`yaw_rate∈[-1,1] rad/s`。纵向不允许倒车。`HierarchicalVelocityAction` 负责三维积分、限幅、构造底层 45 维 observation、执行冻结 AMP policy 并写入 12 个关节目标。

## 4. Critic 与质量 decoder

Critic 采用特权信息：机器人／小车真值速度、绳状态、质量、地面摩擦和轮阻。第一版 decoder 只预测小车质量，不同时预测摩擦、轮阻、间隙或绳状态，避免两帧短历史下不可辨识的目标把奖励来源混在一起。

`TowingMassDecoder` 使用 96 维本体短历史预测归一化质量 `m_norm∈[-1,1]`，监督损失使用 SmoothL1。它是与 actor 完全独立的训练期网络：prediction／hidden feature 都不进入 actor，decoder 梯度也不进入 actor。rollout 期间 decoder 固定并以 `no_grad` 推理；一批 rollout 结束后，才使用仿真质量标签更新 decoder。

牵引指令非零且绳 taut 时，PPO 可获得小权重识别奖励：

```text
e_t = SmoothL1(m_hat_norm, m_norm)
r_id = beta * exp(-e_t / temperature),  beta 初值 0.02, temperature 初值 0.25
```

其余阶段 `r_id=0`。该做法属于主动系统辨识的简化实现，接近 [Learning to Perform Physics Experiments via Deep Reinforcement Learning](https://arxiv.org/abs/1611.01843) 中“通过交互辨识隐藏物理量”的设置。后续可将绝对预测精度替换成 [VIME](https://proceedings.neurips.cc/paper/2016/hash/abd815286ba1007abfbb8415b83ae2cf-Abstract.html) 式动力学后验信息增益，但第一版不实现贝叶斯动力学模型。

**可辨识性限制**：即使 decoder 在训练分布上损失下降，也不代表动作真的包含质量信息。必须报告未见质量的 MAE／R²，并加入 constant-prior、shuffled-label 和“decoder 训练但奖励权重为零”的对照。当前网络、监督更新器和 detached reward 已实现，但尚未接入 PPO rollout，也未训练。

## 5. Reward 与 termination

v0 只启用一个主任务项、两个安全终止、一个碰撞前预警和两个轻量正则。速度跟踪全程生效；停止阶段的用户指令为零，因此无需单独的 `stop_velocity`。

| 名称 | 阶段 | 公式／触发 | 初始权重 | 目的与注意事项 |
|---|---|---|---:|---|
| `tracking_velocity` | 全程 | `exp(−(||v_xy−cmd_xy||/0.5)²−((ω_z−cmd_yaw)/1.0)²)` | +1.0 | 同一项同时负责正常跟速和零指令停车 |
| `collision` | 全程 | 机器人—小车发生接触时为 1 | −50.0 | 同时终止；不用“每步未碰撞加分”，避免靠拖长 episode 刷生存分 |
| `fall` | 全程 | base 高度 `<0.18 m` 时为 1 | −50.0 | 同时终止，防止策略用跌倒快速结束困难 episode |
| `clearance` | 全程 | `softplus((d_warn−d_clear)/0.05)`，`d_warn=0.20 m` | −1.0 | 真实间隙接近安全阈值时提供碰撞前梯度；只在训练 reward 使用 |
| `extra_distance` | 零指令后 | `(x_robot−x_at_stop)_+` | −0.1 | 轻量限制继续前进；停止沿由 action term 缓存，不依赖世界原点 |
| `action_rate` | 全程 | `||u_t−u_{t−1}||²` | −0.02 | 平滑三维上层动作；action 表示加速度，因此也是离散 jerk 代理，只保留这一项 |
| `mass_identification` | 牵引且 taut | `exp(−SmoothL1(m_hat_norm,m_norm)/0.25)` | +0.02 | decoder 输出只生成训练奖励，不进入 actor；由 runner 在 rollout 中追加，当前尚未接线 |

不进入总 reward、只记录／必要时做消融的量：closing speed、姿态、足端滑移、停止时间、绳冲量／峰值、质量预测误差。只有观测到明确失败模式后才增加对应项。两类绳的主比较使用冲量、间隙、碰撞和机器人稳定性，不用单步峰值张力作为主指标。

终止项：episode timeout、机器人跌倒、小车碰撞。绳松弛不是失败。`collision/fall` 是否在终止步被 reward manager 计入必须用一条 scripted 单步测试确认。

当前权重只是量纲归一化后的起点。注册训练前必须让 Direct、Fixed Ramp 和“持续向前”三条 scripted policy 跑同一批 case，确认奖励排序符合：安全且及时停车 > 安全但拖延 > 追尾／跌倒。

## 6. Event 与课程

v0 reset event 已按 episode 采样以下工况：

| 项目 | 范围／规则 |
|---|---|
| 用户纵向速度 | `0.2–1.0 m/s`；横向和 yaw 指令第一版固定为零 |
| 指令 schedule | `0–1 s` 置零站定，`1–t_stop` 牵引，`t_stop–10 s` 再次明确置零；`t_stop∈[4,6] s` |
| 小车质量 | `5–15 kg`；所有小车刚体质量和惯量从默认值同比例缩放，不跨 reset 累乘 |
| 接触摩擦 | `0.4–1.2`；同一环境的四足机器人和小车使用同一采样值 |
| 小车轮阻 | `0.008–0.032 N·m·s/rad`，episode 内固定 |
| 初始位置 | 小车固定；四足机器人 `x±0.03 m`、`y±0.02 m`、`yaw±0.03 rad` |
| 绳模型 | reset batch 内随机打乱后尽量严格 1:1；偶数环境数时完全相等 |

这些量是 critic／decoder 可用的训练期工况信息，不直接进入 actor。当前 event 已写入质量／惯量、接触材料和初始状态，并保存速度、停止时刻、轮阻与绳模型掩码；轮阻和绳模型掩码要等每物理步 physics adapter 接入后才真正影响动力学，因此尚不能标为 Isaac Lab 运行验证通过。

## 7. 代码结构与当前状态

| 文件 | 状态 |
|---|---|
| `towing/upper_logic.py` | 已完成：动作、观测、decoder 的纯逻辑契约 |
| `towing/upper_mdp.py` | 已建：分层 action、obs/reward/termination、每物理步绳力／轮阻和底层 50 Hz 保持；待 Isaac Lab 运行及真实间隙／碰撞接入 |
| `towing/upper_env_cfg.py` | 已建：scene/action/observation/reward/termination class 配置 |
| `towing/agents/upper_ppo_cfg.py` | 已建：PPO 初始配置 |
| `rl_lab/modules/towing_decoder.py` | 已建：独立质量 decoder、rollout detached reward 和批间监督更新器 |
| `tests/test_towing_upper_rl_contract.py` | 已完成：质量归一化、reward mask／detach 和 decoder 独立更新等离线契约测试 |
| Gym task registration | **未做**：物理 adapter 未与测量台对齐前禁止注册 |

## 8. 下一步与注册门槛

1. 在训练机运行环境构造冒烟，验证每物理步 compliant／inextensible 绳力、逐环境轮阻和底层 50 Hz 保持；当前代码已接但未运行。
2. 把测量台的真实车头间隙和三路碰撞见证接入 action term；当前 `clearance/collision` 缓冲尚无有效 producer，不能训练。
3. 自定义 PPO runner 保存 history／质量／active mask；rollout 内用冻结 decoder 追加识别奖励，rollout 后再更新 decoder，真实质量只进 decoder loss／critic。
4. 用 scripted action 在单环境复现 `tow_drag.py` 的跟速、稳态张力、停车滑行和间隙指标。
5. 完成短 rollout 后，才注册 `Imgo2-towing-upper-ppo`。
6. 注册后先跑单工况短训练，排查持续前进、故意碰撞／跌倒等 reward hacking，再扩展课程。

## 9. 总体实施路线

### 9.1 上层强化学习训练

1. 完成绳力、轮阻、碰撞见证和 50 Hz／20 Hz 双频 physics adapter。
2. 用 scripted policy 复现测量台，完成环境注册和短 rollout。
3. 接入独立质量 decoder、rollout target、detached 识别奖励和批间监督更新。
4. 先在单一安全工况训练，再扩展速度、质量、摩擦、轮阻和两类绳课程。
5. 对 Direct、Fixed Ramp、无 decoder、decoder 仅训练不加奖励、质量识别奖励和 Oracle 做统一评估。
6. 冻结最终 checkpoint，只导出读取 96 维两帧 history 的 actor；decoder、critic 和特权标签不导出。

训练阶段完成标准：未见工况上相对 Fixed Ramp 降低碰撞／危险接近，同时速度跟踪、停止距离和机器人稳定性没有通过“持续向前”退化。

### 9.2 Sim2sim 部署

1. 在部署侧实现与训练完全一致的 48 维单帧 observation、两帧缓存、32 维 encoder latent 和 3 维上层 action。
2. 保持上层 20 Hz、冻结底层策略 50 Hz；command 积分、缩放、限幅和初始化逐项对齐。
3. 将小车、轮阻和两类绳加入 MuJoCo／Gazebo 对照场景，不读取训练期质量、间隙或绳状态真值。
4. 先做同输入网络数值一致性，再跑站定、牵引、速度置零、低／高质量和两类绳工况。
5. 对比 Isaac Lab 与 sim2sim 的跟速、停车距离、最小间隙、碰撞、姿态和 action-rate；偏差必须按模型、控制或 observation 分类记录。

sim2sim 完成标准：导出网络与训练 actor 在确定性输入上数值一致，两个仿真器都能完成完整牵引—置零 episode，且关键安全指标的差异有记录和解释。

### 9.3 真机边界

真机部署另行规划。当前只保留可部署 observation、模型导出和频率契约；不在本计划中承诺真机质量边界、距离估计精度或安全停车能力。

## 10. 已知限制

- 本轮没有 Isaac Lab 运行验证；代码只能标为“已建，待运行验证”。
- 质量 decoder、监督更新和 detached reward 已定义，但 rollout／PPO runner 尚未接入，未训练；VIME 只保留为后续方向。
- ManagerBased 绳力／轮阻 adapter、双频控制、event 与 stop schedule 已接入代码，但未在 Isaac Lab 运行；真实间隙／碰撞 producer 仍缺，因此当前环境仍不能训练。
- 最大可拖质量仍待边界扫描实跑，不能从配置范围直接推断。
