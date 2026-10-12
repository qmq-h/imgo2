# 上层「双头」契约迁移 v2 → v3 落地记录（2026-10-10）

本次工作：按 [双头方案记录](towing_upper_two_head_2026-10-10.md) 与用户 2026-10-10 批准的口径，
把上层拖曳策略从**单头 12 维关节残差（契约 v2）**迁到**双头 13 维（1 维 vx 偏移 + 12 维关节
残差，契约 v3）**。

**本机没有 Isaac Lab**：以下所有结论都只由离线 `pytest` / AST / 纯 torch 验证得出；
凡是只有训练/仿真才能确认的，一律标「**待训练机验证**」。

---

## 1. 改了什么

### 1.1 动作与命令（训练侧唯一实现处 `upper_mdp.py`）

动作布局（唯一切分点 `upper_logic.CMD_ACTION_DIM = 1`）：

```
u        = cat(u_cmd(1), u_joint(12))          # 13 = 1 + 12
offset   = clamp(clamp(u_cmd, ±1) × cmd_offset_scale, offset_min, offset_max)
loco_vx  = clamp(task_vx + offset, AMP_VX_MIN, AMP_VX_MAX)   # 训练包络 −1.0 … 1.5
delta    = clamp(u_joint, ±1) ⊙ residual_scale                # hip 0.125 / thigh·shank 0.25
```

**两个命令量彻底解耦（防作弊红线）**：

| 量 | 内容 | 去向 |
|---|---|---|
| `task_command` (N,3) | 脚本调度 + 可选 STOP ramp，不含策略偏移 | **只给奖励**（`velocity_tracking_exp` 等） |
| `loco_command` (N,3) | `task_command` + 有界 vx 偏移（裁进 AMP 包络） | 冻结策略输入 + actor 帧的命令项 |

`velocity_tracking_exp` 的参考量由 `term.loco_command[:, i]` 改为 `term.task_command[:, i]`；
横向/朝向的 PD 也改为先写 `task_command`，再整体 `loco_command.copy_(task_command)` 后只在
vx 通道叠加偏移（这样奖励参考与送策略指令共享同一份 (vy, wz)）。

**偏移门控**：`offset_active = (elapsed_s >= tow_start_s)` —— 出生段（settle）不许推机器人。
注意**不是** `towing` 门控：STOP 之后 `task_command` 归零而偏移仍生效，这正是「停机后继续
走两步」的表达口。

### 1.2 新增 cfg 字段（只加字段、默认值不动语义）

| 字段 | 位置 | 默认 | 含义 |
|---|---|---|---|
| `cmd_offset_scale` | `HierarchicalVelocityActionCfg` | `0.5` | 偏移头尺度（m/s） |
| `offset_min` / `offset_max` | 同上 | `-0.2` / `+0.6` | **偏移量本身**的头权限 |
| `amp_vx_range` | 同上 | `(-1.0, 1.5)` | **合成后 vx** 的训练包络（见 §4） |
| `stop_command_ramp_s` | 同上 | `0.0`（关闭） | STOP 后脚本 vx 在多少秒内线性降到 0 |
| `post_stop_allowance_m` | `mdp.post_stop_distance` 的形参（原为 `extra_distance` 的 `RewTerm.params`） | `0.0`（关闭） | `relu(x − x_stop − allowance)` 的允走量。**2026-10-10 变更：`extra_distance` 已从奖励表删除 ⇒ 本参数目前未接入奖励**，只保留供复用（详见 [奖励改动记录](towing_reward_retune_2026-10-10.md) 与 README TOW-24） |

「停机之后」开关 `stop_command_ramp_s` 默认关闭 ⇒ 与迁移前的行为一致（见 §5 退化性）。用户明确要求本轮
**不得**改默认值。`post_stop_allowance_m` 在 2026-10-10 之后不再有任何奖励项消费它（同名形参仍在
`mdp.post_stop_distance` 上）。

### 1.3 契约 v3

| 项 | v2 | v3 |
|---|---|---|
| `last_action` | 12 | **13** |
| policy 帧 | 57 | **58** |
| decoder `frame_dim` | 57 | **58** |
| actor 输入 | 79 | **80** |
| critic 输入 | 72 | **73** |
| 动作 | 12 | **13**（1 + 12） |
| `towing_contract` | `{"version": 2, ...}` | `{"version": 3, "frame_dim": 58, "explicit_dim": 6, "latent_dim": 16}` |

critic 73 的求和（以源码为准重新复核，`upper_env_cfg.UpperObservationsCfg.CriticCfg`）：

```
policy_frame(58) + robot_velocity(2) + cart_velocity(2) + rope_state(4)
                 + towing_force(3) + cart_privileged_parameters(4) = 73
```

> 用户清单里写「现在 `CriticCfg` 求和为 72 + 3 + 4」——**与源码不符**：v2 时 `CriticCfg`
> 正好是 `57+2+2+4+3+4 = 72`，没有额外的 3 + 4 项。这里以源码为准。

### 1.4 改到的文件

| 文件 | 改动 |
|---|---|
| `imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/upper_logic.py` | `CMD_ACTION_DIM`；`UpperActionSpec` 双头字段/切分/偏移/包络/`scripted_vx`；`UpperObservationSpec.last_action` 13 |
| …/`upper_mdp.py` | `HierarchicalVelocityAction` 13 维动作、`task_command`、包络裁剪、STOP ramp、`post_stop_distance(allowance)`；cfg 新字段；模块/函数文档的链路图 |
| …/`upper_env_cfg.py` | 奖励改为读 `task_command`（注释）、`extra_distance` 带 allowance、`__post_init__` 断言包络覆盖脚本速度范围；文档 |
| …/`agents/upper_ppo_cfg.py` | `decoder.frame_dim=58`、critic 73 注释、13 维动作注释 |
| `imgo2_rl/scripts/rl_lab/rl_lab/config/towing_algorithm_cfg.py` | `frame_dim: int = 58` |
| `imgo2_rl/scripts/rl_lab/rl_lab/wrapper/towing_vec_env_wrapper.py` | 维数断言 `policy=58` |
| `imgo2_rl/scripts/rl_lab/rl_lab/runners/towing_on_policy_runner.py` | **`TOWING_CONTRACT_VERSION = 3`**，save/load 明确拒绝旧契约 |
| `imgo2_rl/scripts/rl_lab/rl_lab/modules/towing_decoder.py` | `frame_dim` 默认 57 → 58（不传参构造不再静默建错网络） |
| `imgo2_rl/scripts/rl_lab/towing/play.py` | 13 维动作 / 包络 / 偏移头的注释与打印口径 |
| `imgo2_rl/scripts/towing/upper_policy_runtime.py` | `FRAME_TERMS` last_action 13、帧 58、actor 80、critic 73、`CHECKPOINT_CONTRACT_VERSION=3`、13 维切分、`command_offset_vx` / `compose_loco_vx` |
| `imgo2_rl/tests/test_towing_upper_rl_contract.py`、`test_towing_upper_policy_runtime.py` | 契约断言全部更新 + 新增守卫（见 §5、§6） |

**未动**（另一并行任务 TOW-23 的文件）：`imgo2_rl/scripts/towing/play_towing_test.py`、
`imgo2_rl/tests/test_towing_play_test.py`、`README.md`。因此本轮落地时测试台仍是 v2 口径，其
13 维接线（尤其偏移头）**由用户随后补**；`upper_policy_runtime.act()` 返回 **12 维** `delta`
给调用方，避免在用户改测试台之前把它打断。
**2026-10-10 晚已接线**（README TOW-26）：测试台改 v3 口径（帧 58 / 动作 13）、用
`command_offset_vx`/`compose_loco_vx` 在冻结策略推理**之前**合成 `loco_command`（两层限幅 +
`elapsed_s >= tow_start_s` 门控，STOP 之后仍生效）、残差仍在底层输出之后叠加；
见 [测试台上层开关记录](towing_upper_switch_2026-10-10.md) 的「v3 接线」一节。

---

## 2. 三个设计决定的落地值

1. **维度：1 维（只 vx）** —— 落地 `CMD_ACTION_DIM = 1`。横向/朝向仍由 PD 外环负责，
   动作里没有 vy/ω 头。
2. **加性 + 有界** —— 落地为「两层限幅」：`offset` 先限在头权限 `[-0.2, +0.6]`，
   再把**和**裁进冻结策略训练包络 `[-1.0, +1.5]`（见 §4 的修正）。
   偏移在 `elapsed_s >= tow_start_s` 后生效；**测试台已按同一顺序接线**（2026-10-10 晚）。
3. **脚本 ramp 作底** —— 落地 `stop_command_ramp_s`（默认 **0.0 = 关闭**，= 迁移前行为）。
   打开后 STOP 之后 `ramp_s` 秒内 `task_command[:,0]` 从 `tow_speed` 线性降到 0，
   偏移头只做自适应修正而不是从零学步态；配套 `post_stop_allowance_m`
   （`post_stop_distance` 的允走量形参）解除 `extra_distance`（−0.1）与新偏移头的对打。
   **2026-10-10 变更**：`extra_distance` 已从奖励表**删除**（理由：与 `tracking_velocity`
   在停车段高度冗余、量级小 20 倍），所以这条「配套」只剩历史意义——`post_stop_allowance_m`
   目前未接入奖励；停车段唯一的奖励项是 `stop_towing_force`。见
   [奖励改动记录](towing_reward_retune_2026-10-10.md)。

---

## 3. 链路顺序与「两头不独立」（2026-10-10 补充，用户/协调方要求写清）

```
task_command ──(速度头: + 有界偏移, 裁进 AMP 包络)──→ loco_command
                                                        │
                                            （冻结 AMP 策略推理，偏移进它的观测）
                                                        ↓
                                               冻结关节目标
                                                        │
                                 (+ 12 维关节残差) ──────┘
                                                        ↓
                                           PD 位置目标（set_joint_position_target）
```

**上层同时影响底层的输入与输出**，所以两头**不再独立**：同一个关节目标变化既可以用偏移
（改变步态相位/速度）达成，也可以用残差（直接改关节角）达成 ⇒ 动作空间存在**别名/冗余**，
credit assignment 更难，两头也可能互相打架（例如偏移让机器人加速、残差同时反向顶住）。

缓解手段（写在 `upper_mdp` 模块文档里，尚未在训练中启用）：

- 把偏移当「步态级」旋钮：建议**低通 / 限速率**（秒级），残差保持「单步级」（20 Hz）；
- 必要时**只在特定相位放开偏移**（例如只在 STOP 窗口），牵引段锁 0；
- 日志里把偏移**按相位分开记**（牵引段 / 停车段各一条）：1.5 m/s 时正半轴被包络裁掉、
  梯度为 0，混在一起的平均值看不出这件事。

---

## 4. 与原始清单的偏离（以源码为准，必须指出）

原始清单写的是 `loco_command[:,0] = clamp(scripted_vx + offset, OFFSET_MIN=-0.2, OFFSET_MAX=+0.6)`。
**照字面实现会毁掉任务**：冻结 AMP 策略的训练指令范围是
`lin_vel_x ∈ (−1.0, 1.5)`（出处 `tasks/manager_based/locomotion/velocity/base_move/amp_env_cfg.py`
的 `commands.base_velocity.ranges.lin_vel_x`），而拖曳脚本速度是 0.5–1.5 m/s
（`mdp/episode_geometry.SPEED_RANGE`；**2026-10-10 当天由 0.4–1.5 收紧为 0.5–1.5**，
契约不受影响，见[工作域收紧记录](towing_work_domain_2026-10-10.md)）——**牵引段本来就顶在上界 1.5**。把"和"裁到 +0.6 会把
1.5 m/s 的牵引指令压成 0.6，而奖励参考量 `task_command` 仍是 1.5 ⇒ 策略被要求跟一个它根本
发不出的速度，退化性测试也不可能通过。

**落地口径**（协调方 2026-10-10 从源码核对后确认）：

1. `offset_min` / `offset_max` 限的是**偏移量本身**（头权限）；
2. **和**裁到 `amp_vx_range = (−1.0, 1.5)`，即冻结策略的训练包络（超出即 OOD、步态退化）；
3. `upper_env_cfg.__post_init__` **断言该包络覆盖整个脚本速度范围** —— 于是 `u_cmd = 0` 时
   裁剪是恒等的，退化性成立；若有人把 `amp_max` 调到 1.0，起环境时直接报错而不是静默分叉。

语义后果：牵引段速度高时正半轴偏移被包络钳掉（1.5 m/s 时 +0.5 无梯度 ⇒ 实际只能**减速**，
正好用于「绷直前降速、少灌冲击能量」）；STOP 后 `task_vx = 0`，偏移头在同一包络内可正可负
（可倒走制造松弛）。因此在 `amp_vx_range` 内，停车段的偏移权限实际是 `[-0.2, +0.5]`
（受 `cmd_offset_scale`/头权限限制），足以走路/倒走，但达不到 ±1.5 —— 若之后要把停车段的
偏移权限放大，只需调大 `cmd_offset_scale` 与 `offset_max`，包络仍是硬上界。

---

## 5. 退化性与防作弊：怎么验的

**退化性（偏移 = 0 / 开关默认关 ⇒ 与 v2 逐位一致）**

- 纯函数 ① `UpperActionSpec.loco_command_vx(u_cmd=0, scripted) == scripted`，覆盖
  `scripted ∈ {0.0, 0.4, 0.7, 1.5}`；`apply_offset=False` 时同样成立
  （`test_command_offset_is_bounded_additive_and_gated_off_at_spawn`）。
- 纯函数 ② `logic.scripted_vx(..., ramp_s=0.0)` 与旧脚本相位（settle 0 / tow speed / STOP 0）
  逐值相等；`ramp_s=2.0` 时 5.0 s→0.8、6.0 s→0.4、7.0 s→0
  （`test_scheduled_vx_matches_the_old_schedule_when_ramp_is_off`）。
- 纯函数 ③ `post_stop_distance(env, post_stop_allowance_m=0.0)` 的公式断言为
  `relu(x − x_stop − 0)`（默认参数写法 + 公式字符串守卫）。**2026-10-10 起该函数不再是奖励项**
  （`extra_distance` 删除），但公式与形参原样保留，测试也改成「AST 断言奖励表里没有它 +
  把函数体抽出来实跑」两条。
- 源码级 ④ 训练侧 `process_actions`：`task_command` 的脚本项与旧写法逐字相同
  （`torch.where(towing, self.tow_speed, torch.zeros_like(...))`）、ramp 代码块在
  `if self.cfg.stop_command_ramp_s > 0.0:` 内、偏移门控与 `_was_stopped` 无关、
  偏移只加在 vx 通道（`test_two_new_knobs_default_off_so_v3_reduces_to_v2_behaviour`）。
- 源码级 ⑤ `reset()` 同时清 `task_command` 与 `loco_command`
  （`test_reset_clears_the_task_command_too`）。

**防作弊（奖励只读 `task_command`）**

- **AST 守卫**：从 `UpperRewardsCfg` 解析出所有 `RewTerm(func=mdp.X)`，再在 `upper_mdp` 的 AST
  里禁止这些函数出现任何 `loco_command`（属性或名字）；同时断言
  `velocity_tracking_exp` 显式读 `task_command`
  （`test_reward_terms_never_read_loco_command`）。
- 源码级：`velocity_tracking_exp` 的 `forward_error / lateral_error / yaw_error` 三条都改成
  `term.task_command[:, i]`，且恢复旧口径的分支仍在
  （`test_tracking_reward_uses_measured_velocity_and_shaping_term_is_gone`）。
- 解耦本身：`loco_command` 由 `task_command` 拷贝 + `where(active, offset, 0)` 组成；
  `_processed` 必须先于偏移合成更新（读本拍 `u_cmd`）
  （`test_task_and_loco_command_are_decoupled_in_the_action_term`）。

**契约与切分**

- `13 == CMD_ACTION_DIM + JOINT_ACTION_DIM == 1 + 12`；`split_action` 逐维 clamp、维数不符硬报错；
  13 维整动作喂给 `delta_joint_pos` 必须报错（`test_action_split_is_one_command_plus_twelve_joints`）。
- 帧 58 / actor 80 / critic 73 / 动作 13 的**真实网络前向**（`test_towing_network_forward_contract_and_zero_init`）。
- decoder `frame_dim` 的**注册 cfg**、**rl_lab cfg**、**decoder 默认值**、**wrapper 字面量**、
  **运行时 `FRAME_TERMS`** 与 `UpperObservationSpec` 五方交叉校验。
- `amp_vx_range` 逐字等于 `amp_env_cfg` 的 `lin_vel_x`（正则从源码读出），且覆盖 `SPEED_RANGE`。

**checkpoint 契约**

- v3 `{"version": 3, "frame_dim": 58, "explicit_dim": 6, "latent_dim": 16}` 能加载；
- 旧 **v2（57/79）**、半迁移（version 3 但 frame 57）、更早的 51/56/63 维、非 dict、
  `None` 全部在 `load_state_dict` **之前**被 `TowingCheckpointContractError` 拒绝；
- 伪造 v3 契约 + 旧 79 维 actor 权重被 `strict=True` 抓住；
- 训练侧 runner 的 `TOWING_CONTRACT_VERSION` 与运行时常量同值，且 `save()` 写入的
  `towing_contract["version"]` 就是它（`load()` 用同一常量比对）。

**离线实测**（本机 `python3 -m pytest imgo2_rl/tests -q`）：**521 passed**（含 TOW-23 并行
任务已落地的测试）。另：
`python3 -m py_compile …` 通过；`git diff --check` 干净；
`git ls-files -i -c --exclude-standard` 为 **0 条**。

---

## 6. 未做 / 待训练机验证

1. **残差与偏移接 PhysX**：本机没有 Isaac Lab，`process_actions` 的张量路径、包络裁剪、
   ramp、偏移头对底层步态的影响**一次都没跑过**。需要在训练机起一次冒烟，确认
   `loco_command[:,0]` 在牵引段 == `tow_speed`、STOP 后 == `clamp(offset, …)`。
2. **strict 加载**：`runner.load()`（Isaac Lab 路径）与 `UpperPolicyRuntime`（纯 torch 路径）
   的拒绝逻辑都只有纯 torch/AST 测试覆盖；训练机应真实地用旧 v2 checkpoint 试一次，
   确认报错信息可读且**没有**进入 `load_state_dict`。
3. **`stop_command_ramp_s` / `post_stop_allowance_m` 未启用**：默认 0.0，本轮不训练。
   两者的分支只有纯函数与源码级测试，**没有**任何训练证据说明 ramp 能否改善停车段。
   **2026-10-10 起 `post_stop_allowance_m` 连奖励项都没有了**（`extra_distance` 删除），
   要恢复得先往 `UpperRewardsCfg` 加回一条以 `mdp.post_stop_distance` 为 func 的 `RewTerm`。
4. **偏移头的相位数**：`cmd_offset_scale` / `offset_min` / `offset_max` /
   `amp_vx_range` 的数值都是按源码包络推出来的，未经训练标定；1.5 m/s 段正半轴无梯度这一
   后果需要实跑日志（按相位分开记偏移）确认。
5. **动作别名/credit assignment**：§3 的缓解手段（低通、限速率、按相位放开）**尚未实现**，
   只是文档建议；需要实跑看两头是否互相打架（`action_magnitude` / `action_rate` 与
   `tracking_velocity` 的曲线）。
6. **部署侧（本机无法验证 C++/部署）**：

   - `imgo2_deploy` 的 `rl_sdk.cpp`：观测 **79 → 80**、动作 **12 → 13** 的切分
     （`u_cmd` 1 维先用于速度指令、`u_joint` 12 维再叠加到关节目标）；
   - **顺序要求（硬）**：偏移必须加在**底层策略推理之前**（它进的是底层观测），
     关节残差加在底层输出**之后** ⇒ 部署里是「两层串联、偏移在推理前生效」；
     检查项：把 `loco_command` 送进冻结策略的**那一处**必须已经含偏移；
   - `export_policy_as_jit` 产物契约（`towing_contract` v3、输入 58/80）；
   - `play.py` 导出的 `policy.pt` 需要按新维数重新导出（旧产物一律不可用）。

7. **测试台 13 维接线** —— **2026-10-10 晚已完成**（README TOW-26；当时这里的「待办」清单
   已逐条落地）：`play_towing_test.py` 改成 v3 口径（帧 58 / 动作 13 / `contract version 3`），
   用 `command_offset_vx` / `compose_loco_vx` 在冻结策略推理**之前**组 `loco_command`
   （两层限幅 + `elapsed_s >= tow_start_s` 门控，STOP 之后仍生效），`held = 冻结目标 + delta`
   （delta 仍 12 维）。详见 [测试台上层开关记录](towing_upper_switch_2026-10-10.md) 的
   「v3 接线」一节。**运行期仍未验证**（本机无 Isaac Lab）。

---

## 7. 下一步：怎么开 ramp 与 allowance 跑训练

> **2026-10-10 变更**：`extra_distance` 已从奖励表删除，§2/§6 里「开 `post_stop_allowance_m`
> 解除对打」的做法**已失效**（该参数目前没有消费方）。下面的步骤保留为历史方案；
> 现行停车段只由 `stop_towing_force`（−1.0）与 `min_clearance`（出生几何阈值，−5.0）塑形。
> 见 [奖励改动记录](towing_reward_retune_2026-10-10.md)。

前提：**先新建 run**（旧 v2 checkpoint 一律失效，不续训）。

1. **先跑 v3 基线（开关保持 0.0）**，确认新契约能起环境、奖励分项与 v2 基线可比
   （退化性：偏移=0 时 `loco_command == task_command == v2 的脚本指令`）。这一轮主要看
   `Episode_Reward/tracking_velocity`、`obs_stop_reached`、`stop_towing_force`、
   `min_clearance`（新口径）的量级。
2. 想验证「停机续走」，打开：

   - `actions.high_level_velocity.stop_command_ramp_s = 1.0 ~ 2.0`（脚本侧先给出可跟的参考量，
     `task_command` 与 `loco_command` 一起 ramp ⇒ 跟踪奖励在 ramp 段有非零目标）；
   - 若仍要用「允走量」这一路，需要先往 `UpperRewardsCfg` 加回一条以 `mdp.post_stop_distance`
     为 func 的 `RewTerm`（形参 `post_stop_allowance_m` 仍在），再设 0.5 ~ 1.0 m；
     否则**不需要**（2026-10-10 起停车后前进不再被`extra_distance` 惩罚）。
3. 观察量：把偏移按相位分开记（牵引段 / 停车段）——建议在 `play.py` 侧先手工看
   `action_term.processed_actions[:, 0]` 与 `task_command[:,0]`、`loco_command[:,0]` 三条曲线；
   若停车段偏移长期贴在 `+0.5` 且 `tracking_velocity` 掉，说明包络/尺度需要重标定。
4. 打开 ramp/allowance 后**不要**复用基线 run 的归一化统计（量纲/分布变了）。

---

## 8. 相关记录

- 设计依据与 9 项接口清单：[双头方案（未实现 → 已实现）](towing_upper_two_head_2026-10-10.md)；
- 冲击窗口独立统计（同一批工作的另一半）：[impact window](towing_impact_window_2026-10-10.md)；
- 测试台上层开关（TOW-20 加开关 / **TOW-26 已接到 v3**）：[upper switch](towing_upper_switch_2026-10-10.md)；
- 残差方案来源：[deltapos residual](towing_deltapos_residual_2026-10-08.md)。
