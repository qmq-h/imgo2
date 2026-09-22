# 上层拖曳 RL 环境链路复审（2026-09-22）

## 结论

`cf465a2` 已经搭出一条有依据的上层拖曳 RL 骨架：上层输出参考加速度，冻结 AMP 底层策略生成关节目标，绳力和轮阻每个物理步写入，actor 只读可部署本体量，critic／decoder 才读特权量。这条分层路线适合继续做。

当前实现仍不是可注册、可训练的环境。复审发现 5 类阻断项，其中安全信号缺失和 observation 契约漂移会直接让训练目标失真；普通 PPO runner 与 decoder 也尚未形成闭环。仓库现在保持未注册是正确的。

## 复审范围与证据

检查链路：

```text
scene / reset event
  → upper action 与 reference command 积分
  → 冻结 AMP policy（50 Hz）
  → 关节目标 + 绳力/轮阻（200 Hz）
  → observation / reward / termination（20 Hz）
  → RSL-RL PPO / mass decoder
```

对照了仓库实现和本机 `C:\Users\qmq\Desktop\RL\IsaacLab` 源码中的 `ActionTerm` 调用频率、`ObservationManager` 历史拼接、`RewardManager` 的 `weight × dt`、reset 顺序以及 RSL-RL runner 配置接口。

离线命令：

```powershell
python -m unittest discover -s tests -p 'test_towing_upper_rl_contract.py' -v
```

结果为 7 通过、2 跳过；跳过项需要 PyTorch。通过项证明纯逻辑维数、端点映射、schedule、特权量隔离和“暂不注册”守卫符合当前文档，但没有构造 Isaac Lab 环境，也没有覆盖下面的运行级问题。

## 阻断项

### 1. 安全信号没有 producer，现有 reward 会被扭曲

`HierarchicalVelocityAction` 初始化 `rope_state[:, 0] = 0` 和 `cart_collision = false`，后续物理 adapter 只更新 tension、extension、taut，没有写真实间隙或碰撞。因此：

- `cart_collision` reward 和 termination 永远不触发；
- `clearance_barrier` 永远把 clearance 当 0；
- 配置中的 `cart_contacts` 没有 pairwise filter，单靠当前 net force 也不能可靠区分轮地接触、车斗接地和机器人撞车。

这不是“少一个诊断量”。`RewardManager` 会乘上 `step_dt=0.05`；当前常量 clearance 惩罚每步约为 `-softplus(4)×0.05 = -0.201`，完整 10 s episode 约为 `-40.2`。提前跌倒只收到一次经 dt 缩放的终止惩罚，反而可能比坚持到 episode 结束回报更高，形成明确的 reward hacking 通道。

需要把 `tow_drag.py` 已验证的真实车头间隙、车斗接触、轮部异常接触和速度跃变见证迁入环境，并写 scripted 单步测试确认终止步同时计入碰撞／跌倒惩罚。

### 2. 96 维 history 的实际顺序与部署规格不同

配置在 observation group 上设 `history_length=2`。Isaac Lab 的实现是：每个 term 先各自展开两帧，再按 term 顺序拼接。因此当前布局是：

```text
[cmd(t-1), cmd(t), action(t-1), action(t), ..., joint_vel(t-1), joint_vel(t)]
```

而计划和部署描述要求的是两个完整 48 维 frame：

```text
[frame(t-1), frame(t)]
```

维数同为 96，现有测试只检查维数和 term 名称，因此发现不了这个错位。若按文档在部署侧维护两帧完整缓存，网络输入会静默错位。应改为一个先拼成 48 维 frame 的 observation term，再对该 term 做 history，或明确采用 term-major 并同步训练、导出、部署和测试；建议保留更直观的 frame-major 契约。

### 3. 上层积分器状态对 actor 不可见

动作是参考加速度，真正送给底层策略的是积分后的 `reference_command`。Actor 目前只看到用户指令和上一拍动作，没有看到当前 `reference_command`。同一组本体状态和短历史可能对应不同的已积累参考速度，过程不是完整可观，策略只能从机器人滞后的响应猜内部积分状态。

`reference_command` 是部署时天然可得的控制器内部状态，应直接进入 actor observation；否则应把上层动作改成直接速度，或使用能承载完整历史的 recurrent policy。加入 3 维 reference 后要重新确定单帧和两帧维数，并同步 decoder 与部署接口。

动作映射还有一个次要但实际的偏置：纵向范围 `[-1.0, +0.5]` 采用线性映射后，归一化动作 0 对应 `-0.25 m/s²`，不是零加速度。维持恒速需要策略长期输出 `+1/3`。若要保留更强制动能力，适合用以 0 为中心的分段缩放。

### 4. PPO 与 decoder 没有训练闭环

当前任务没有 Gym registration；`TowingMassDecoder`、detached reward 和 optimizer wrapper 只有独立模块，没有 runner 调用。普通 `ppo/train.py` 不会：

- 从 rollout 保存 96 维 history、质量标签和 active mask；
- 在 rollout reward 中追加识别奖励；
- 在 rollout 之间更新 decoder；
- 保存／恢复 decoder 与 optimizer 状态。

此外，按当前本机 Isaac Lab/RSL-RL 源码，`RslRlBaseRunnerCfg.obs_groups` 是必填项；`UpperTowingPPORunnerCfg` 未声明 actor／critic 组映射。即便现在注册任务，当前配置也没有完整描述 `policy`、`critic`、`decoder` 三组该如何交给算法。

Decoder 训练时还要只选 active／有激励样本，或至少报告分阶段损失；当前 `MassDecoderTrainer.update()` 本身不接 mask，把站定和停车样本全部混入会强化 constant-prior。必须保留未见质量 MAE／R²、shuffled-label、constant-prior 和“decoder 训练但奖励为零”的对照。

### 5. 多环境 reset 存在串扰与分布漂移

Isaac Lab 会先执行 reset event，再调用 action term 的 `reset(env_ids)`。Action term 虽然只清指定 env 的张量，却调用无 env 参数的 `FrozenLowLevelPolicy.reset()`，从而把所有环境的 low-level `last_action` 一起清零。任一环境跌倒会污染其他仍在运行的环境 observation 和底层动作历史。

同一 reset 还没有清 `rope_state`，所以 reset 后、第一次物理步前的 critic observation 会带上上一回合的绳状态。

绳模型的“reset batch 内 1:1”在全量同步 reset 时成立，但异步 singleton reset 总会分到 compliant（`count // 2 = 0`），长期会让总体比例漂移。应使用逐 env Bernoulli／固定分层分配，或维护全局配额，而不是依赖每次 reset batch 的偶数性。

## 合理且应保留的设计

- 上层只做 command shaping，底层 locomotion 冻结，职责清楚，便于独立验证和导出。
- `ActionTerm.process_actions()` 每 20 Hz 处理上层动作，`apply_actions()` 每 5 ms 施加绳力／轮阻，并按 20 ms 节拍调用底层策略；放置位置符合 Isaac Lab 生命周期。
- 底层策略适配器显式核对模型输入输出、策略关节顺序、观测缩放、动作裁剪和关节目标，避免静默错腿。
- 质量和惯量从 immutable defaults 同比例缩放，不跨 reset 累乘；actor 不读取质量、摩擦、小车速度或绳状态，特权信息边界清楚。
- compliant／inextensible 共用统一接口并逐环境切换，适合做同批对照。
- 未完成运行验收前不注册任务，避免把“能 import”误当成“能训练”。

## 建议的闭环顺序

1. 实现真实间隙和三路碰撞 producer，补 reward/termination 单步测试，先消除错误学习目标。
2. 固定 frame-major observation 契约，加入 `reference_command`，对每个下标写确定性布局测试。
3. 把 low-level history、rope state 和绳模型采样改成真正 per-env reset。
4. 补 `obs_groups`；实现自定义 runner 的 decoder rollout、masked update、checkpoint 和日志。
5. 单环境用 Direct、Fixed Ramp、持续向前三条 scripted policy 复现 `tow_drag.py` 的跟速、张力、停车距离、间隙和碰撞排序。
6. 再做 4／256 环境短 rollout，检查 reset 串扰、两类绳比例、NaN、接触和 GPU/CPU view 写入。
7. 完成这些验证后注册任务，先单工况短训练，再逐步扩质量、摩擦、轮阻和绳模型课程。

## 已修复与待修复

本次是复审，没有修改训练代码。已修复的是维护状态：README 的 TOW-03 已从“骨架待运行”提升为 P0“链路复审未通过”，避免后续把当前实现误当成可训练环境。

待修复项就是上述 5 类阻断问题；缺少 Isaac Lab 可运行环境和 PyTorch，因此本次没有执行环境构造、scripted rollout 或 decoder 数值测试，也没有把任何运行项标为通过。

## 同日修复进展

后续“修一下”工作已处理第 1／2／3／5 类问题和第 4 类中的 `obs_groups`：安全信号已有 producer，actor 改为 51×2 的 frame-major 输入并加入积分器状态，动作映射改为零中心，异步 reset 与绳模型采样已修。追加复核还发现冻结 AMP 契约把 50 Hz 策略周期误写成 200 Hz 物理周期；现已统一为 `0.005 s × 4 = 0.02 s` 并加入物理／底层／上层三层周期校验。具体改动、验证和剩余限制见 [修复记录](towing_upper_rl_fix_2026-09-22.md)。

仍未完成的是 decoder runner 闭环和 Isaac Lab 运行验收，因此本复审最初的“不可注册”结论暂不撤销；安全 producer 的“恒零／恒 false”与 observation/reset 问题已不再是当前源码状态。

同日最终方案又改为 recurrent 架构：actor／critic／decoder 分别使用 GRU，actor 每步读取当前 51 维帧；decoder 同时预测质量和张力，并以连续张力替代本记录早期提出的 active mask。当前契约以 [GRU 改造记录](towing_gru_design_2026-09-22.md) 为准。
