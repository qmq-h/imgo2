# 上层拖曳强化学习计划

> **2026-10-10 变更**：`extra_distance`（`mdp.post_stop_distance`，−0.1）已从奖励表**删除**（函数本体与 `post_stop_allowance_m` 形参保留、目前未接入奖励）；`min_clearance` 的阈值由 `ratio(0.25) × 连接长度` 改为「**出生间隙 − 绝对死区 `deadband_m`(0.02 m)**」（2026-10-10 当天先落成 `spawn_margin(0.85)` 相对余量、同日再微调为绝对死区；语义＝「不许比出生时更近」）、权重由 −2.0 提到 **−5.0**。本文以下是变更**之前**的记录，现行口径见 [奖励改动记录](towing_reward_retune_2026-10-10.md) 与 README 问题表 TOW-24。

> 最后维护：2026-10-08。本文是上层 RL 环境的实现规格。**2026-10-08 架构改动**（用户决定）：上层动作由「3 维参考加速度」改为 **12 维关节位置残差**，送冻结策略的速度指令改由脚本调度给出；decoder 牵引力由 2 维改为 **3 维**（速度保持 2 维）。改动细节、验证与限制见 [残差动作与三维拉力记录](towing_deltapos_residual_2026-10-08.md)；历史复审结论仍见 [上层拖曳 RL 链路复审](towing_upper_rl_review_2026-09-22.md)。

## 1. 任务结构

冻结已有 locomotion policy，上层策略只输出叠加在它关节目标上的残差：

```text
user cmd（脚本调度） → frozen locomotion → joint targets
                                              + deltapos（上层 20 Hz）→ joint position command
```

速度指令不再由上层网络积分产生：它由 `scheduled_command` 逐拍写成 `loco_command`（settle 0 → tow 速度 → STOP 后 0），与测量台 `tow_drag.py` 的 `user_cmd` 同相位（其记录里 `user_cmd == ref_cmd`）。v0.1 只研究直线停止瞬态。上层频率 20 Hz，物理步长 0.005 s；底层策略保持自己的 0.02 s 控制周期。动作和观测由 class-based config 管理，不修改已有 locomotion 环境。

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
| 1 | `loco_command` | 3 | 脚本调度出来、**实际送给冻结底层策略**的速度指令 `[vx, vy, yaw_rate]` |
| 2 | `last_action` | 12 | 上一拍上层动作：策略关节顺序下的**归一化关节残差** |
| 3 | `base_ang_vel` | 3 | IMU 机体系角速度，训练缩放为 `0.25` |
| 4 | `projected_gravity` | 3 | IMU 姿态对应的机体系重力方向 |
| 5 | `last_loco_action` | 12 | 冻结底层策略上一拍输出 |
| 6 | `joint_pos` | 12 | 策略关节顺序下的相对位置 |
| 7 | `joint_vel` | 12 | 策略关节顺序下的速度 |

单帧原始 observation 固定为 **57 维**，不输入 base 线速度、小车速度／位置、robot-cart distance、绳长、绳刚度／阻尼、负载质量或牵引力。这些量要么难以部署获取，要么属于待估计的隐藏物理量。（2026-10-08 前的 51 维版本并列了 `cmd_vel` 与 `reference_command` 两项；残差方案下两者恒等，已合并为一项。）

Dynamics decoder 每步读取当前 57 维帧并维护自己的 GRU hidden state，输出机器人机体系线速度估计 `v_hat=[vx,vy]`（2 维，不含竖直）、质量估计 `m_hat` 和**三维**牵引力估计 `F_hat=[Fx,Fy,Fz]`（机体系；2026-10-08 由 2 维改为 3 维，理由见 §4）。这 6 维输出 detach 后与原始帧拼接，因此 PPO actor 实际输入为 **63 维**：

```text
57D frame -> Linear(128) -> GRU(128)
                         -> velocity head (2)
                         -> mass head (1)
                         -> towing-force head (3)

actor input = [57D frame, v_hat(2), m_hat(1), F_hat(3)]
```

PPO actor 自身仍使用独立的单层 256 维 GRU，critic 使用另一套 GRU。Decoder、actor、critic 不共享参数或 hidden state。

Actor 不启用经验归一化，保持上表约定的显式缩放和部署契约；仓库自有 `rl_lab` towing runner 只对 critic 特权输入维护 running mean/variance，以处理张力、质量、摩擦和轮阻等尺度差异。该处理不修改 reward，只通过价值拟合与 advantage 间接影响 Actor 更新；critic 及其 normalizer 不导出到部署策略。critic 特权输入当前为 **72 维**（57 帧 + 机器人速度 2 + 小车速度 2 + 绳状态 4 + 机体系拉力 3 + 质量/摩擦/轮阻/有无小车 4）。

GRU 的有效历史由 rollout 序列和 episode 边界决定，不再手工展平固定帧数。训练、回放和部署都必须在单个环境结束时分别清 decoder／actor／critic hidden state。

## 3. Action

`high_level_velocity` 是 **12 维**归一化动作，语义是**叠加在冻结策略关节位置目标上的残差**：

```text
u ∈ [-1, 1]¹²                         （上游 clip_actions = 1.0）
deltapos = action_scale ⊙ u            （逐关节：髋 0.125 rad，大腿/小腿 0.25 rad）
joint_cmd = default_dof_pos + action_scale · (loco_action + u)
```

残差尺度直接取冻结策略契约 `LowLevelPolicyCfg.action_scale`，与底层动作同量纲，因此幅值受底层动作范围界定：`clip_actions=1.0` 时残差最多用到底层权限（动作裁剪 ±3.0）的 **1/3**；要放大上层权限就调大 `clip_actions`，并同步复核 `action_magnitude` 权重。`HierarchicalVelocityAction` 负责写 `loco_command`、把归一化残差换算成 `delta_joint_pos`、构造底层 45 维 observation、执行冻结 AMP policy，并在**每次底层刷新（20 ms）**时重算 `joint_targets + delta_joint_pos`（残差 50 ms 变、底层输出 20 ms 变，只在 50 ms 处算一次会用错值）。

**残差绝不能进入冻结策略自己的观测**（尤其它的 `last_action`），否则 45 维冻结契约就被改写成另一个策略；这条由契约测试守着。动作必须从 0 起步，因此 towing runner 对 actor 末层做零初始化（只作用于 towing，ppo／amp／himloco 的共用模块未改）。

## 4. Critic 与负载 decoder

Critic 直接读取训练期真值：机器人／小车速度、质量、机体系牵引力、绳状态、地面摩擦和轮阻。Actor 只能读取 decoder 估计，不读取这些真值。

`TowingDynamicsDecoder` 的三类监督目标为 `v_GT`、`m_GT`、`F_GT`。速度和牵引力全程使用 Huber loss，包括松绳阶段的 `F_GT=0`。质量误差由当前 GT 牵引力大小连续加权：

```text
s_t = max(||F_GT_t|| - F_min, 0)
w_t = s_t / (s_t + F_scale)
L_D = lambda_v * Huber(v_hat, v_GT)
    + lambda_F * Huber(F_hat, F_GT)
    + lambda_m * w_t * Huber(m_hat, m_GT)
```

初始 `F_min=1 N`、`F_scale=10 N`。无有效拉力时 `w_t=0`，拉力越充分质量监督越强。权重只能由 detach 的 `F_GT` 计算，不能使用 `F_hat`，否则 decoder 可以通过压低自己的力预测逃避质量损失。该权重只作用于质量项，不屏蔽速度或力监督。若实验表明松绳后质量估计迅速遗忘，再把“episode 首次有效牵引后保持监督”的 persistent 版本作为消融，而不是初版默认。

当前 target **直接回归物理量**（m/s、kg、N），不做归一化也不 clamp：2026-09-23 去掉 target 归一化与 head 的 tanh，因为归一化尺度会按 s² 压低力与质量的 loss 权重（力 s=10 ⇒ 0.01、质量 s=5 ⇒ 0.04），且 `v/(1.0,0.5)` 的 clamp 会截断超速真值；稳定性改由 `smooth_l1(β=1)` 提供。环境的 training-only decoder group 另外直接输出由物理 `‖F_GT‖`（**三维**，含竖直分力）计算的 `w_t`，runner 不从 target 反推权重。

预测误差不进入 PPO reward。Decoder 只最小化 `L_D`，PPO 只最大化任务 return。Decoder 输出进入 actor 前 detach，因此第一版不存在 `L_PPO -> decoder` 梯度。rollout 必须保存采样当时的 **6 维**估计；decoder 只在 PPO 使用完该批数据后更新，不能更新 decoder 后重算旧 rollout 的 actor observation。

**可辨识性限制**：未知绳长、刚度、阻尼、轮阻和摩擦可能产生相近的机器人响应，因此质量与力只能在训练域和充分激励下做条件估计。必须分别报告未见质量／绳参数下的速度 RMSE、质量 MAE／R² 和牵引力 RMSE，并加入 constant-prior、shuffled-label、无 decoder 以及 Oracle 真值输入对照。

### 4.1 可行性评估与训练约束

整体方案可行。机器人速度估计与已有腿式机器人本体速度估计问题相近，57 维输入包含 IMU、关节状态、底层动作、脚本速度指令与上一拍关节残差，GRU 可以利用步态周期与跟踪残差恢复机体系 `vx/vy`。缺少足端接触标志会使打滑工况更难，因此必须单独报告未见摩擦、低附着和扰动下的速度误差。

牵引力可看作冻结 locomotion 在已知 reference command 下受到的外部扰动，但质量、轮阻、摩擦与未知绳参数之间存在等效性，质量不是任何时刻都可辨识。连续力权重避免绳尚未受力时强迫网络猜质量，并让强交互样本承担更高监督权重；它不能消除训练域外的不可辨识性。

Decoder 输出进入 actor 会造成一个新的非平稳来源：decoder 每次更新后，同一 57 维输入可能产生不同估计。第一版采用以下约束：

1. 先用 scripted／随机安全策略预训练 decoder（**尚未实现**，当前 runner 是每个 PPO 迭代后更新一次 decoder）；
2. rollout 内冻结 decoder，并保存采样时的 **6 维** estimate；
3. PPO 使用完该批数据后再更新 decoder，不用新 decoder 重算旧 observation；
4. actor 输入中的 estimate 全部 detach；
5. 评估时分别报告 estimator 精度和最终控制指标，不能用训练 loss 代替安全收益。

PPO actor 再使用 GRU 是合理的：decoder hidden 压缩动力学辨识信息，actor hidden 负责控制阶段、动作平滑和停止过程记忆。代价是训练和部署必须维护两套状态；若消融显示 actor GRU 无收益，可将 actor 降为 MLP，但 decoder GRU 保留。

STOP 目标在现有动作接口上可表达：到 `t_stop` 时脚本把 `loco_command` 置 0（冻结策略被要求站定），上层残差仍可在这段瞬态里改变姿态与蹬地方式，从而决定"前移必要距离后卸载"还是"立刻停死"。当前 tracking reward（**实测**速度 vs 指令）、`stop_towing_force` 和 `extra_distance` 倾向尽快卸载并停止，而 clearance／collision 项倾向避免追尾；正式训练前必须确认 scripted return 排序满足“安全前移必要距离 > 立即停死后追尾 > 持续前进”。若排序不成立，优先调整停车后距离项的死区和安全项，并复核拉力惩罚尺度，而不是加入 prediction reward。

## 5. Reward 与 termination

v0 启用一个主任务项、两个安全终止、一个碰撞前预警、一个 STOP 后拉力项和两个轻量正则。速度跟踪全程生效；停止阶段的用户指令为零，因此无需单独的 `stop_velocity`。

| 名称 | 阶段 | 公式／触发 | 初始权重 | 目的与注意事项 |
|---|---|---|---:|---|
| `tracking_velocity` | 全程 | `exp(−(||v_xy−cmd_xy||/0.5)²−((ω_z−cmd_yaw)/1.0)²)` | +1.0 | 同一项同时负责正常跟速和零指令停车 |
| `collision` | 全程 | 机器人—小车发生接触时为 1 | −50.0 | 同时终止；不用“每步未碰撞加分”，避免靠拖长 episode 刷生存分 |
| `fall` | 全程 | base 高度 `<0.18 m` 时为 1 | −50.0 | 同时终止，防止策略用跌倒快速结束困难 episode |
| `clearance` | 全程 | `softplus((d_warn−d_clear)/0.05)`，`d_warn=0.20 m` | −1.0 | 真实间隙接近安全阈值时提供碰撞前梯度；只在训练 reward 使用 |
| `stop_towing_force` | `t ≥ t_stop` | `||F_tow||/(||F_tow||+10 N)` | −1.0 | 收到本回合 STOP 后惩罚仍由绳子传递的拉力；有界归一化避免峰值张力支配回报，初始站定及无小车环境不生效 |
| `extra_distance` | `t ≥ t_stop` | `(x_robot−x_at_stop)_+` | −0.1 | 轻量限制继续前进；显式按停止时刻门控，初始站定阶段不生效 |
| `action_rate` | 全程 | `||u_t−u_{t−1}||²` | −0.02 | 平滑三维上层动作；action 表示加速度，因此也是离散 jerk 代理，只保留这一项 |

不进入总 reward、只记录／必要时做消融的量：closing speed、姿态、足端滑移、停止时间、绳冲量／峰值、质量预测误差。只有观测到明确失败模式后才增加对应项。三类连接（两套单边绳 + 双边刚体球铰连杆，见 [刚体连杆记录](towing_rigid_link_2026-10-08.md)）的主比较使用冲量、间隙、碰撞和机器人稳定性，不用单步峰值张力作为主指标。

终止项：episode timeout、机器人跌倒、小车碰撞。绳松弛不是失败。`collision/fall` 是否在终止步被 reward manager 计入必须用一条 scripted 单步测试确认。

当前权重只是量纲归一化后的起点。`stop_towing_force` 与 clearance 的配置权重均为 `−1.0`，在 20 Hz 上层控制周期下各自每步最大惩罚量级约为 `−0.05`；前者不能替代间隙和碰撞项，因为单纯快速卸载绳力也可能让小车逼近。注册训练前必须让 Direct、Fixed Ramp 和“持续向前”三条 scripted policy 跑同一批 case，确认奖励排序符合：安全且及时停车 > 安全但拖延 > 追尾／跌倒。

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
| 绳模型 | 每个环境 reset 时独立 Bernoulli 采样，兼容异步 singleton reset；长期期望比例 1:1 |
| 无小车基线 | 每次 reset 以 `12.5%` 概率采样，256 环境时期望约 32 个；小车实例横向停放 2 m，绳力、轮阻、碰撞／间隙项屏蔽；decoder 的力目标和质量监督权重均为零 |

这些量是 critic／decoder 可用的训练期工况信息，不直接进入 actor。当前 event 已写入质量／惯量、接触材料、初始状态和小车存在掩码，并保存速度、停止时刻、轮阻与绳模型掩码；physics adapter 已按环境应用或屏蔽绳力／轮阻，但尚未在 Isaac Lab 运行验证。

## 7. 代码结构与当前状态

| 文件 | 状态 |
|---|---|
| `towing/upper_logic.py` | 已完成：动作、观测、decoder 的纯逻辑契约 |
| `towing/upper_mdp.py` | 已修待运行：分层 action、每物理步绳力／轮阻、底层 50 Hz、车体表面间隙代理，以及车斗／四轮分别过滤机器人接触的碰撞判据均已接入 |
| `towing/upper_env_cfg.py` | 已建：scene/action/observation/reward/termination class 配置 |
| `towing/agents/upper_ppo_cfg.py` | 已建：使用仓库 `rl_lab.config.TowingOnPolicyRunnerCfg`，不再导入 `isaaclab_rl.rsl_rl` |
| `rl_lab/runners/towing_on_policy_runner.py` | 已建：三套 GRU 状态、detached estimate rollout、critic-only normalizer、PPO 后 decoder 更新及联合 checkpoint |
| `rl_lab/wrapper/towing_vec_env_wrapper.py` | 已建：适配 Isaac Lab 的 `policy/critic/decoder` observation groups 与五元 step 接口 |
| `scripts/rl_lab/towing/train.py` | 已建：自有 towing runner 训练入口。2026-09-22 修掉其 `--agent` 默认值（原值全仓无对应注册项），改为与 `rl_lab_cfg_entry_point` 推导一致的 `rl_lab`（**已修，待验证**） |
| `rl_lab/modules/towing_decoder.py` | 已建：`57→128→GRU(128)` dynamics decoder、三个 prediction heads（vel 2 / mass 1 / force 3）、连续力加权质量监督和梯度隔离；head 宽度与 loss 切片由模块内 `VELOCITY_DIM/MASS_DIM/FORCE_DIM/FORCE_SLICE` 常量推导 |
| `tests/test_towing_upper_rl_contract.py` | 已完成：六维 decoder target（vel 2 + mass 1 + force 3）、63 维 actor 拼接、57 维帧、12 维残差动作、force-weighted mass supervision 和梯度隔离契约；另守「残差不进冻结策略观测」「actor 末层零初始化」「rl_lab 侧维数字面量与 `upper_logic` 一致」 |
| Gym task registration | **已注册（2026-09-22 用户决定）**：`Imgo2-towing-upper-rl-lab`，`--agent=rl_lab_cfg_entry_point`。这显式翻过了原先「物理 adapter 未与测量台对齐前禁止注册」的保护，故运行验收项仍未完成。注册接线与验收清单见 [训练前置记录](towing_training_prep_2026-09-22.md) |

## 8. 下一步与注册门槛

1. 在训练机运行环境构造冒烟，验证每物理步 compliant／inextensible／rigid 连接力（含刚体连杆的压缩推力）、逐环境轮阻和底层 50 Hz 保持；两套绳的代码已接但未运行，刚体连杆（2026-10-08 新增）连静态检查之外都未实跑。
2. 在训练机核对车体表面间隙代理，以及车斗／四轮过滤机器人接触的判据，并与测量台 FK 间隙、车斗／车轮记录交叉验证。
3. 在训练机验证自有 recurrent runner：在线保存 6 维 decoder estimate 并拼成 63 维 actor observation；检查三套 GRU reset、GT-force mass weight、episode 边界切分、PPO 后 decoder 更新及 checkpoint 恢复，并确认**同一速度指令下残差=0 时关节指令与改造前逐位一致**（验证残差通路而非整回合等效：指令已由积分斜坡改为脚本阶跃）、残差从 0 起步。
4. 用 scripted action 在单环境复现 `tow_drag.py` 的跟速、稳态张力、停车滑行和间隙指标。
5. 任务已于 2026-09-22 注册为 `Imgo2-towing-upper-rl-lab`（`--agent=rl_lab_cfg_entry_point`），用于在训练机执行训练；第 1–4 项运行验收仍未完成，不得因任务可启动而视为通过。
6. 注册后先跑单工况短训练，排查持续前进、故意碰撞／跌倒等 reward hacking，再扩展课程。

## 9. 总体实施路线

### 9.1 上层强化学习训练

1. 完成绳力、轮阻、碰撞见证和 50 Hz／20 Hz 双频 physics adapter。
2. 用 scripted policy 复现测量台，完成环境注册和短 rollout。
3. 接入 dynamics decoder、force-weighted mass supervision、detached actor augmentation 和批间序列更新；decoder error 不进入 reward。
4. 先在单一安全工况训练，再扩展速度、质量、摩擦、轮阻课程；连接类型与长度已由 20×20 确定性网格（列 = 弹性绳 8 / 刚体 8 / 普通绳 4，行 = 长度 0.4→0.8 m）固定，不再作为课程维度。
5. 对 Direct、Fixed Ramp、无 decoder、decoder estimate 和 Oracle 真值输入做统一评估。
6. 冻结最终 checkpoint，联合导出 decoder 与 actor；部署保留 decoder／actor 两套 hidden state，不导出 critic 或训练期真值。

训练阶段完成标准：未见工况上相对 Fixed Ramp 降低碰撞／危险接近，同时速度跟踪、停止距离和机器人稳定性没有通过“持续向前”退化。

### 9.2 Sim2sim 部署

1. 在部署侧实现与训练完全一致的 57 维单帧 observation，顺序执行 decoder 和 actor，并在 reset 时清两套 hidden state；关节指令必须按 `冻结策略输出 + action_scale ⊙ clip(u)` 合成，且残差不得回灌冻结策略的观测。
2. 保持上层 20 Hz、冻结底层策略 50 Hz；脚本速度指令的相位、残差缩放／限幅和初始化逐项对齐。
3. 将小车、轮阻和连接模型加入 MuJoCo／Gazebo 对照场景，不读取训练期质量、间隙或连接状态真值。**当前只有两套单边绳场景**：刚体球铰连杆是 2026-10-08 的训练侧新增，sim2sim 第三场景（双边约束）尚未实现，见 TOW-05。
4. 先做同输入网络数值一致性，再跑站定、牵引、速度置零、低／高质量和各连接工况；刚体连杆需在 MuJoCo 场景补齐后才纳入。
5. 对比 Isaac Lab 与 sim2sim 的跟速、停车距离、最小间隙、碰撞、姿态和 action-rate；偏差必须按模型、控制或 observation 分类记录。

sim2sim 完成标准：导出网络与训练 actor 在确定性输入上数值一致，两个仿真器都能完成完整牵引—置零 episode，且关键安全指标的差异有记录和解释。

### 9.3 真机边界

真机部署另行规划。当前只保留可部署 observation、模型导出和频率契约；不在本计划中承诺真机质量边界、距离估计精度或安全停车能力。

## 10. 已知限制

- 本轮没有 Isaac Lab 运行验证；代码只能标为“已建，待运行验证”。
- GRU decoder、监督 loss、连续力加权质量监督、detached actor augmentation 和 `rl_lab` recurrent rollout／PPO runner 已接线，尚未训练机运行。
- ManagerBased 绳力／轮阻 adapter、双频控制、event、stop schedule、安全 producer 已接入代码，但尚未在 Isaac Lab 运行；间隙当前是 base 后表面到车斗前表面的有向代理，腿部几何由车斗和四轮对机器人过滤接触的终止信号兜底，仍需与离线 FK 指标交叉验证。
- 2026-09-22 已把 actor 改为 51 维单帧 GRU 输入，加入 `reference_command`，修复 per-env 底层 history／rope state reset 并改用逐环境 Bernoulli 绳模型采样；随后将 towing recurrent PPO、decoder 更新和 critic-only normalizer 全部移入仓库 `rl_lab`，不再依赖外部 RSL-RL runner/config API。任务仍未注册，训练机 rollout 尚未执行。
- 2026-10-08 动作改为 12 维关节位置残差、送冻结策略的速度指令改由脚本调度给出，decoder 牵引力改为 3 维（帧 51→57、decoder 5→6、actor 56→63、critic 65→72）；离线契约与守卫测试通过，但**未在 Isaac Lab 运行验证**，旧 run 的 checkpoint 与 TensorBoard 对照作废。详见 [记录](towing_deltapos_residual_2026-10-08.md)。
- 最大可拖质量仍待边界扫描实跑，不能从配置范围直接推断。
