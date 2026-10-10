# 上层拖曳：12 维关节位置残差 + 三维牵引力（2026-10-08）

> **2026-10-10 变更**：`extra_distance`（`mdp.post_stop_distance`，−0.1）已从奖励表**删除**（函数本体与 `post_stop_allowance_m` 形参保留、目前未接入奖励），`--compact-log` 的分项 picks 里它也已被 `min_clearance` 取代；`min_clearance` 的阈值由 `ratio(0.25) × 连接长度` 改为 `spawn_margin(0.85) × 出生间隙`、权重由 −2.0 提到 **−5.0**。本文以下是变更**之前**的记录，现行口径见 [奖励改动记录](towing_reward_retune_2026-10-10.md) 与 README 问题表 TOW-24。

> 本轮由用户决定两项改动并已落地：**① decoder 牵引力 2 维 → 3 维（机器人速度保持 2 维）**；
> **② 上层动作由「3 维归一化加速度」改为「12 维关节位置残差」**，送冻结 AMP 策略的速度指令
> 改由脚本调度给出。现状与验收清单以 [README](../README.md) 问题表 TOW-06 为准。

## 1. 现象与动机

### 1.1 拉力为什么从 2 维变 3 维

改前 `towing_force_b` 是**机器人 base 机体系下的前两维**（x 前后、y 左右）：

```python
force_robot_w = torch.stack(tuple(state.force_on_robot), dim=-1) * present   # 世界系 3 维
force_robot_b = quat_apply_inverse(body_quat_w[base], force_robot_w)          # → base 系 3 维
self.towing_force_b[:] = force_robot_b[:, 0, :2]                              # ← 这里截断
```

而 `force_on_robot = tension × direction`，`direction` 本来就是三维单位向量（`mdp/rope_model.py`）。
所以 2 维是**接口截断**，不是物理简化。被丢掉的竖直分量不是小量：

- 两挂点有竖直高差 **Δz = 0.17 m**（机器人 spawn 0.35 − 小车 `base_link` 0.18；spawn 布局的
  勾股解 `attachment_horizontal_gap` 直接用它）；
- 绷紧时方向向量 z 分量 = 0.17 / L0，在网格的 L0 = 0.4…0.8 m 上就是张力的 **21–43%**；
- 挂点 `robot_attachment = (-0.16, 0, 0)`，`Fz` 在该后臂上产生**俯仰力矩**，这块扰动原先在
  观测与惩罚里都不可见；
- 所有 `vector_norm(towing_force_b)` 的消费点（`stop_towing_force`、`towing_force_norm`、
  质量监督权重）从此算的是**真实张力**（比原 2 维范数大 2–10%，因为原值是 T·cosθ）。

竖直速度则**保持不进 decoder**：`root_lin_vel_b[:, 2]` 在 trot 步态下按步频大幅振荡，
不是负载量，而上层也没有垂直控制通道。用户决定即为「拉力 3 维、vel 2 维」。

### 1.2 为什么动作改成 deltapos

改前：冻结策略的速度指令由上层网络积分产生（`reference_command`），链路是
`user cmd → upper action(3 维加速度) → reference velocity → frozen locomotion → joint targets`。

改后：速度指令由**脚本调度**直接给出，上层网络只输出叠加在冻结策略关节目标上的增量：

```text
user cmd(脚本) → frozen locomotion → joint targets
                                        + deltapos(上层 20 Hz) → joint position command
```

理由与收益：

1. **残差不进冻结策略观测**就不改写底层契约，底层仍是"冻结的"；
2. 上层不再拥有内部积分器状态，**没有 windup**：动作本身就是关节偏置，语义直白；
3. 脚本调度的相位与测量台 `tow_drag.py` **逐位一致**（`command = args.velocity if phase == "tow" else 0`，
   其记录里 `user_cmd == ref_cmd`，即"P4 还没有 command shaping"），因此这一步让训练环境更贴近
   复审要求的"物理 adapter 必须先复现 `tow_drag.py`"；
4. 顺带修掉一处奖励错误（见 §2 第 3 条）。

## 2. 改法（代码落点）

| 文件 | 改动 |
|---|---|
| `upper_logic.py` | `UpperActionSpec` 重写为 `deltapos = residual_scale ⊙ clip(u, ±1)`（`residual_scale` 必填、`control_dt` 保留）；删除 `normalized_acceleration` 与 `integrate_reference_speed`；`UpperObservationSpec` 改 57 维（`loco_command` 3 + `last_action` 12 + ang_vel 3 + gravity 3 + last_loco_action 12 + joint_pos 12 + joint_vel 12），`decoder_dim` 改为**由 `DecoderSpec` 派生的属性**；`DecoderSpec` 力改 3 维 ⇒ `dim=6`；`scheduled_command` 升为送冻结策略的指令本身 |
| `upper_mdp.py` | `action_dim` = 冻结策略关节数（12）；`_raw/_processed/_previous` 12 维；新增 `delta_joint_pos` 与 `loco_command`（删除 `reference_command`/`user_command`）；`process_actions` 只写脚本指令与 `delta_joint_pos`；`apply_actions` 在**每次 20 ms 底层刷新**时重算 `output.joint_targets + self.delta_joint_pos`，且 `velocity_command=self.loco_command`；`towing_force_b` 改 3 维（`[:, 0, :3]`）；`policy_frame` 57 维；删除 `reference_tracking_l2`；`velocity_tracking_exp` 线性项改比**实测**体速；`HierarchicalVelocityActionCfg` 去掉 `acceleration_*`/`reference_*` |
| `towing_decoder.py` | head 宽度、loss 切片、shape 校验全部由 `VELOCITY_DIM(2)/MASS_DIM(1)/FORCE_DIM(3)/OUTPUT_DIM(6)/MASS_INDEX/FORCE_SLICE` 推导；新增 `output_dim`；`mass_supervision_weight` 的 shape 校验 2→3 |
| `towing_on_policy_runner.py` | `actor_obs_dim = env.num_obs + self.decoder.output_dim`；**actor 末层零初始化**（只在此 runner，ppo／amp／himloco 共用模块未动）；`--compact-log` 分项去掉 `reference_tracking`、补 `stop_towing_force`/`extra_distance` |
| `towing_vec_env_wrapper.py` | 契约字面量 57 / 7；`get_decoder_supervision` 切 `[:, :6]` / `[:, 6]` |
| `agents/upper_ppo_cfg.py` | `decoder.frame_dim` 51→57；注明 `clip_actions` 现在就是残差上限（髋 ±0.125 rad、大腿/小腿 ±0.25 rad，为底层 ±3.0 权限的 1/3） |
| `upper_env_cfg.py` | actions 去掉四组旧范围参数；奖励删除 `reference_tracking`；`tracking_velocity`/`action_rate`/`action_magnitude` 注释按新语义更新 |
| `play.py` | 指令列改 `loco_command` + 12 维残差范数；decoder 力切片 `3:6`；摘要键改 `err_track_mean`/`residual_norm_mean`（删除 `err_cmd_mean`/`err_low_mean`） |

**第 3 条（顺带修的错误）**：`velocity_tracking_exp` 的线性项原写 `reference_command − user_command`，
也就是"上层自己积分出来的指令 vs 任务指令"，与本函数的名字、以及 `upper_env_cfg` 奖励文档里
"实际速度 vs 命令期望"的描述都不符，而且和 `reference_tracking_l2`（权重 −5.0）**重复**。
残差方案下 `reference_command` 不存在，该式会恒为 0（`exp(0)=1` 变成白送的正奖励），
因此改为实测机体系线速度 vs `loco_command`，口径与测量台 `summarize_tow` 的
`steady_tracking_ratio` 一致；`reference_tracking` 项删除。

**维度账**（务必按此核对）：

| 量 | 改前 | 改后 |
|---|---|---|
| policy 帧 | 51 | **57** |
| decoder 输出 | 5（vel 2 + mass 1 + force 2） | **6**（vel 2 + mass 1 + force 3） |
| decoder obs group | 6（targets 5 + weight 1） | **7** |
| actor 输入 | 56 | **63** |
| critic 输入 | 65（51+2+2+4+2+4） | **72**（57+2+2+4+3+4） |
| 上层动作 | 3 | **12** |

## 3. 验证

本机 `/opt/conda/envs/isaaclab`（Python 3.11.13、torch 2.7.0+cu128）；**本机没有 `omni.kit`**，
Isaac Lab 的管理模块无法 import，因此只能跑离线检查：

- `python -m unittest discover -s tests -p 'test_*.py'`：**323 项通过**（拖曳子集 251 项）；
- 新增/改写契约测试：残差映射（裁剪+逐关节尺度）、57/63 维契约、decoder head 宽度与索引常量、
  `frame_dim` 与 `upper_logic` 的交叉校验、残差加在 `joint_targets` 上、残差不进冻结观测、
  actor 末层零初始化；
- **用真实网络类跑了一次端到端前向**：`57 → GRU(128) → 6 维估计 → 63 → GRU(63,256) → MLP → 12`，
  critic `72 → GRU(72,256) → MLP → 1`；零初始化后首拍动作精确为 0，而未初始化时为 |a|max ≈ 0.149
  （归一化，约 ±0.04 rad），证明该断言非空；
- **四条新守卫各做过一次负向测试**（改坏 → 测试失败 → 恢复）：`frame_dim` 写回 51、
  去掉末层零初始化、把残差混进 `parts_from_robot_state`、把 decoder 的 `FORCE_DIM` 改回 2；
- `compileall`、`check_asset_paths.py`、`check_model_sync.py`、`check_towing_mjcf.py`、
  `git diff --check`、tracked-ignore（`git ls-files -i -c --exclude-standard` 为空）通过。

## 4. 限制与待办

- **未在 Isaac Lab 运行**：未构造环境、未训练、未回放。运行级结论全部待训练机验证，
  清单见 README TOW-06（wrapper 断言 57/7、`memory_a` 63、`memory_c` 72、
  **同一速度指令下残差=0 时关节指令与改造前逐位一致**、残差从 0 起步、decoder 6 维可从 checkpoint strict 加载）。注意「一致」的边界：改造后送给冻结策略的是**脚本阶跃**指令（`t=tow_start` 直接给 `tow_speed`），改造前是网络积分斜坡，所以这条只验证残差通路（求和位置、关节顺序、20/50 ms 时序），**不是整回合等效**。
- **旧证据作废**：`imgo2_deploy/policy/imgo2/towing/policy.pt`（`model_2000.pt` 的原样拷贝，
  56 维 actor／5 维 decoder）与旧 run 的 TensorBoard 读数都不能再与当前契约对照。
- **权重待实跑复核**：`action_rate`（−0.1）与 `action_magnitude`（−0.05）原先按 3 维动作标定；
  12 维下 `Σu²` 上限由 3 变 12，同等逐维幅值量级约 ×4。残差是否被这两项压得过小、或反过来
  顶掉冻结步态，需要实跑数据判断。
- **decoder 仍未预训练**：设计文档建议"先预训练 decoder 再交替"，当前 runner 仍是每个 PPO
  迭代后更新一次 decoder（既有的已知偏差，本轮未改）。
- **部署侧未跟上**：真机 `rl_real_imgo2.cpp`／MuJoCo 侧需要实现
  `关节指令 = 冻结策略输出 + action_scale ⊙ clip(u)`，且残差不得回灌冻结策略观测；
  上层策略本身仍无 `play.py` 导出件（TOW-05）。
- **残差幅值上限是设计选择，尚无实跑依据**：当前等于冻结策略的 `action_scale`（上层
  `clip_actions=1.0`）。若实跑发现上层权限不足（例如无法有效卸载绳力或维持牵引），
  应调大 `clip_actions` 并同步复核幅值／速率惩罚，而不是直接改 `action_scale`（那是
  冻结策略的契约）。
