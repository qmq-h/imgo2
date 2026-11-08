# HIMLoco 训练与现有部署接口复审（2026-11-03）

## 范围与结论

用户要求查清哪里不一致，尽量改训练侧，避免影响已经工作的PPO/AMP。本轮只读生产代码，新增离线复现工具与本记录，没有修改HIM训练、导出、部署配置/共享C++或权重，没有启动训练。

结论：可以不改共享部署代码，仅在HIM训练侧适配现有HIM配置。HIM/PPO/AMP本来各有独立yaml，不能把HIM独立yaml的变更等同于改动共享C++并破坏其他策略。上一轮“部署有问题”应准确理解为与当前训练接口不一致；缺原训练配置，不能认定现有HIM权重原先就按当前Imgo2源码训练。

另外确实复现了训练侧估计器输入切片、历史跨回合残留和reset接口的问题，不能只列PD等数值差异。这些问题属于当前源码，未证明部署旧权重也由这份源码产生。

## 1. 训练侧明确问题

### 1.1 估计器的下一帧目标字段错位

himloco_env_cfg.py的policy顺序是commands(3)、ang_vel(3)、gravity(3)、q(12)、dq(12)、last_action(12)，共45。critic顺序为同名六项共45，然后base_lin_vel(3)，height_scan(187)，运行日志证实235维。

him_estimator.py:83实际代码：

```python
next_obs = next_critic_obs.detach()[:, 3:self.num_one_step_obs+3]
```

因此num_one_step_obs=45时取3:48：删掉commands，最后加进base_lin_vel，不是下一帧policy观测。当前速度监督45:48确实对应base_lin_vel，不能为了修next_obs盲目一起把速度改成0:3。

同时critic的q/dq使用默认.*关节选择，未像policy赋值self.joint_names；当前Isaac内部关节顺序按髋/大腿/小腿分组，policy明确按腿交错，二者前45维的关节顺序不同。critic dq scale=1、policy=.05。因此只把3:48替换为:45还不能宣称下一帧语义与actor一致。

建议修法：显式传入下一帧policy观测作为target（含终止前观测），或统一critic前45维的字段顺序、关节顺序/速度缩放，再改成:45。仍保留critic的速度及地形特权尾部。base_lin_vel的scale=2是已有训练约定，不是部署直接读真实速度，是否改为1应单独决定，不能当作已证明的接口bug。

### 1.2 历史跨回合残留

HimlocoVecEnvWrapper.step先把新观测拼进270维历史，但计算dones后没有清理已结束环境的旧历史。底层环境已经重置，导致新回合起始输入混入上一回合的运动/跌倒帧。六帧正常排列“最新在前”本身正确，问题在回合边界。

reset()也只返回底层45维观测，没有初始化/清理wrapper的270维历史和critic缓存；get_observations随后仍可能返回旧缓存。建议reset和done采用明确一致的初始化规则：当前帧加零填充过去帧，与部署刚进入策略时的零历史匹配，并保留未结束环境历史；终止前critic/policy target另行留存，不能被清零操作覆盖。初始化也应按同样规则填入首帧，不让runner首次从全零状态读取本该有效的当前帧。

## 2. 如果完全不改部署，训练要适配什么

以当前真实控制器顺序FL/FR/RL/RR和HIM mapping [3,4,5,0,1,2,9,10,11,6,7,8]为准：

| 项目 | 当前训练 | 保持现有HIM部署不动时的训练目标 |
|---|---|---|
| 策略关节顺序 | FL,FR,RL,RR | FR,FL,RR,RL，各腿hip/thigh/shank |
| q/dq观测及action顺序 | 按上述训练列表 | actor、critic对应字段、动作和last_action统一新顺序，显式preserve_order=True |
| commands输入 | 原值 | 输入乘[2,2,.25]；物理指令/奖励目标仍用m/s与rad/s，不缩放实际command范围 |
| 默认q/观测参考q | 各腿0/.87/-1.82 | 按部署reference和mapping写成物理关节名配置，见下表 |
| 运动PD | 25/.5 | HIM专属机器人副本40/1；不可修改共享IMGO2_CFG |
| action_scale | hip .125，其余.25 | 已相同，维持 |
| clip | 缩放偏置后目标±3rad | 部署是原始action±100；训练需在缩放前采用同样裁剪，并使last_action也记录执行所用动作，不能直接将目标clip改成±100rad |
| 历史 | 6×45，最新在前 | 保持，修复初始化/回合重置 |

映射后的物理默认角（仅为精确适配现有yaml，不代表建议站姿）：

| 物理腿 | hip | thigh | shank |
|---|---:|---:|---:|
| FR | +.1 | .8 | -1.5 |
| FL | -.1 | .8 | -1.5 |
| RR | +.1 | 1.0 | -1.5 |
| RL | -.1 | 1.0 | -1.5 |

不应仅按FL/FR名字顺序把default_dof_pos原样抄入。训练可以在HIM cfg的self.scene.robot独立副本上配置参考角/PD，禁止改共享资产造成PPO/AMP一起变化。保持部署33.5Nm数字不动也不等于实际Gazebo允许33.5Nm；模型/控制器/力矩限制需实测核定，不能盲目把训练effort_limit从23.7改33.5。

要同时保留PPO那套0/.87/-1.82站姿和25/.5增益，就无法又“完全不改HIM独立yaml”又让训练/部署物理契约相同。可以仅更新HIM独立yaml而不动公共C++及PPO/AMP配置；用户当前偏好先研究训练修改，本轮未替用户选定或实施这项取舍。

## 3. 无需为了适配修改的部分

- 单帧观测大类顺序commands/angular/gravity/q/dq/action已与HIM部署一致；不应强改成PPO顺序。
- 45×6=270历史和最新帧在前一致。
- Actor使用45+3+16=64维、estimator输出速度与L2归一化latent、JIT组合推理结构，与当前训练act_inference逻辑一致。没有发现需修改共享Forward才能运行HIM的证据。
- PPO与AMP使用不同观测/配置是正常的；无需为了HIM统一重写公共部署接口。

## 4. 验证、限制与下一步

工具：imgo2_rl/scripts/tools/check_himloco_training_contract.py。解释器：/home/tommy0929/miniconda3/envs/env_isaaclab/bin/python。CPU实际执行源码中的next_obs表达式及wrapper.step/reset方法，模拟2环境，其中一个done，另一个继续。结果：当前切片确为3:48；done后过去225维仍保留旧回合值；显式reset返回45维且未清wrapper历史。机器结果见[JSON](himloco_training_contract_2026-11-03.json)。这是缺陷复现通过，不是修复通过。

已完成：接口逐项审查、明确训练缺陷及只改HIM的适配路径，PPO/AMP未动。

待修：上述生产实现均尚未修改；需要选定HIM目标物理契约后修改训练配置/估计器target/历史初始化，补训练-部署同状态270维输入及12维目标角一致性检查，再跑Isaac构造与短训。接口/归一化语义改变时应新建run重新训练，不拿仅有的JIT冒充可续训checkpoint。只有导出policy，没有完整checkpoint，当前无法按常规恢复PPO优化器和训练状态。旧policy不会因修改训练代码而自动修复。
