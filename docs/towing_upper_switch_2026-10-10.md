# 必要性测试台加「上层网络开关」（2026-10-10）

关联：[README](../README.md) 问题表 **TOW-20**、维护记录（2026-10-10 一行）；背景见
[必要性测试口径](towing_necessity_test_2026-10-09.md) 与 TOW-19。

## 需求与现状

TOW-19 里记录：`imgo2_rl/scripts/towing/play_towing_test.py` 是「冻结 AMP 策略 + 脚本速度指令」
的**必要性基线**测试台，当时**故意不加载任何上层 checkpoint**，所以「策略 vs 基线」的同网格、
同指标对照做不到。本轮加一个开关：**默认关（= 原基线）**，给了 checkpoint 就打开上层网络，
用同一轮 run、同一网格、同一指标做对照。

## 契约（逐条核对了源码，不是从文档抄的）

> ⚠ 本节是 2026-10-10 **白天**的 v2 口径（帧 57 / actor 79 / 动作 12、`version: 2`）。
> 当晚双头迁移落地后测试台已接到 **v3**（帧 58 / 动作 13 / `version: 3`）：**以本文后面的
> 「v3 接线」一节为准**，本节保留为当时的核对记录。

| 项 | 结论 | 出处 |
|---|---|---|
| 57 维帧 | `cat(loco_command(3), processed_actions(12), base_ang_vel_b(3)·0.25, projected_gravity(3), last_loco_action(12), joint_pos−default(12), joint_vel(12)·0.05)` | `upper_mdp.policy_frame`；顺序/维数与 `upper_logic.UpperObservationSpec.terms` 一致 |
| processed_actions | 上一拍上层动作 **clamp 到 ±1**（未缩放），首拍为零 | `upper_mdp.process_actions`（`_processed = actions.clamp(-1,1)`）、`upper_last_action` |
| projected_gravity | `quat_apply_inverse(root_quat_w, (0,0,-1))` | `upper_mdp.projected_gravity` |
| last_loco_action | 冻结策略**自己**的 12 维裁剪后动作（策略关节顺序） | `upper_mdp.last_locomotion_action`、`low_level_policy.step` 的 `out.action` |
| 残差 | `delta = clamp(action, ±1) ⊙ action_scale`；`action_scale` 取冻结策略契约（hip 0.125、thigh/shank 0.25） | `upper_mdp.process_actions` / `UpperActionSpec`、`policy_cfg.get_policy("amp").action_scale` |
| decoder | `TowingDynamicsDecoder(frame_dim=57, feature_dim=128, hidden_dim=128, num_layers=1, latent_dim=16, force_scale=10.0)`；`forward_with_latent(frame, hidden)` 必须 `.eval()`（`sample=None ⇒ self.training=False` ⇒ latent 取均值） | `towing_decoder.py`、`agents/upper_ppo_cfg.py` |
| actor | `ActorCriticRecurrent`（hidden 256/128/64、GRU 256×1、elu），输入 = `augment_actor_observation(frame, estimate, latent)` = **79**；确定性取均值 | `actor_critic_recurrent.py`、`towing_on_policy_runner.get_inference_policy` |
| critic 维数 | 72 = frame 57 + robot_vel 2 + cart_vel 2 + rope_state 4 + towing_force 3 + cart_params 4 | `upper_env_cfg.CriticCfg`（离线测试按 AST 逐项求和核对） |
| checkpoint | `torch.load(path, map_location="cpu", weights_only=False)`；键 `model_state_dict` / `decoder_state_dict` / `towing_contract` / `iter` | `towing_on_policy_runner.save/load` |
| 契约硬校验 | `{'version': 2, 'frame_dim': 57, 'explicit_dim': 6, 'latent_dim': 16}` 必须精确相等 | 同 `runner.load`；旧 56/63 维 checkpoint 在这里就被拒 |
| 上层节拍 | 训练侧 `upper_control_dt = 0.05 s`（20 Hz），冻结策略 50 Hz；两次上层更新之间残差保持不变 | `HierarchicalVelocityActionCfg.upper_control_dt`、`apply_actions` |
| JNT 参考 | 训练侧 `low_level_position_error_l2(reference="commanded")` = 冻结输出 + 残差 | `upper_mdp.low_level_position_error_l2` |

与时序有关的一处**实现取舍**：本测试台的冻结策略每 4 个物理步刷新（20 ms），上层每 10 个
物理步（50 ms）推理一次。`held = 冻结目标 + 残差` **只在冻结策略刷新那一拍重算并下发**
（与训练侧 `apply_actions` 完全一致：训练里 `_held_joint_targets` 也只在
`_physics_step % low_level_decimation == 0` 时重算，两次刷新之间即使 `delta` 刚变也不改
下发目标）。因此上层 tick 若落在两次冻结刷新之间（step 10/30/…），新残差会在下一次刷新
（step 12/32/…，即 10 ms 后）才生效——这不是简化，而是照抄训练侧的时序。

`reset()` 的取舍：`Memory.reset(dones=None)` 是 no-op（源码里 `if hidden_states is None or
dones is None: return`），所以**全量复位**只能把 `memory_a/memory_c.hidden_states` 置 `None`
（等价于 GRU 的默认零初值，也是 runner 在 `load()` 里的做法）；**部分 env 复位**走公开的
`actor.reset(dones)`（`env_ids` 语义保留）。

## 改动

- **新增** `imgo2_rl/scripts/towing/upper_policy_runtime.py`（纯 torch + 标准库，不 import
  Isaac Lab；rl_lab 的 actor/decoder 惰性加载，离线时按文件路径兜底）：
  `UpperPolicyRuntime(checkpoint, num_envs=, action_scale=, device=, deterministic=, agent_cfg=)`，
  对外 `reset(env_ids=None)` 与 `act(loco_command=, base_ang_vel=, projected_gravity=,
  last_loco_action=, joint_pos_rel=, joint_vel=) -> (delta, processed)`；持有 `_last_action`
  （首拍为零）与 decoder GRU 隐状态。加载前硬校验 `towing_contract`，随后
  `load_state_dict(strict=True)`，并做一次零帧前向自检（形状 + 有限性）。
- **改** `play_towing_test.py`：
  - CLI：`--upper-checkpoint PATH`（默认 `None` = 基线）与 `--upper-stochastic`（默认确定性
    均值）；checkpoint 不存在 / 单独给 `--upper-stochastic` 都在启动 Isaac Sim **之前**报错。
  - 惰性块（`AppLauncher` 之后）里按注册的 `UpperTowingPPORunnerCfg()` 建 spec、构造 runtime；
    `experiment.json` 的 `upper_policy` 段记 enabled/checkpoint/确定性与加载后的 contract/iter/
    维数/残差尺度（`report.json.arguments` 也已含两个 CLI 字段）。
  - `plan` 行 + 报告 + `make_row` 的 JNT 参考：开关关时与原来逐位一致（残差恒 0）；开时
    `joint_targets = loco_joint_targets + upper_delta` 既下发、又作为 `robot_jt_*`。
  - 顶部 docstring：把「本脚本不加载任何上层 checkpoint」改成「默认不加载，可用
    `--upper-checkpoint` 打开」，并加了一节「策略 vs 基线要对比哪些字段」。
  - **验收口径不变**：判定码、五项指标、阈值、`report.csv`/`report.json` 的字段与基线
    完全同一套（判据/指标函数里没有任何开关分支，离线测试按 AST 守着）。开关只改
    **残差怎么来**（0 → 网络输出）与 `JNT` 的参考量（冻结输出 → `held`，训练侧契约）。
  - `report.md` / `report.json` 新增「策略 vs 基线」一节：本轮 / **同版本基线**
    （`--compare-report <上一轮>/report.json`）/ **归档基线 2026-10-09**（仅参照）三列，
    同一 `case_metric_summary()` 口径：通过数、判定码分布、五项指标中位数（含样本数）、
    逐坡度量级通过数；归档那列写明 git `1bbb405`（工作树脏，复现用 `6fac89a`）与旧几何
    （绳 0.6–1.2 m、出生比 0.5）⇒ **不可逐格硬比**。
  - 可追溯字段：`experiment.json.upper_policy` = `enabled` / `checkpoint` / `mode`
    （deterministic|stochastic）/ `towing_contract`（version/frame_dim/explicit_dim/latent_dim）
    / `iter` / `actor_obs_dim` / `critic_obs_dim` / `action_dim` / `residual_scale`；
    `report.json` 带上 `upper_policy` 与 `baseline_comparison`；真实跑时另打印一行
    `[plan] 上层网络（checkpoint 已加载）：… iter=… towing_contract=… mode=…`。

### 为什么选「两次运行 + 报告对照」，而不是同轮切换

用户希望「同轮内可切换」；但本测试台**每个 env 只跑一个 episode、不 reset**，一次进程里
只有一次 rollout，同轮切换等于把两套完整的 800 env × 2200 步串起来跑，还要处理场景/GRU/
`policy.reset()` 的二次初始化——成本与风险都不低，而收益只是省一次启动。
因此本轮选**两次运行 + 报告里自动对照**（第 2 轮用 `--compare-report` 指向第 1 轮的
`report.json`），并在本文件与脚本 docstring 里写清这个选择。两轮参数除开关外必须完全一致。

## v3 接线（2026-10-10 晚：测试台接到**双头**契约）

> 本节覆盖上面「契约」表里的 v2 口径（帧 57 / actor 79 / 动作 12 / `version: 2`）：双头迁移
> （[实现记录](towing_upper_two_head_impl_2026-10-10.md)）之后**上层 checkpoint 契约是 v3**
> （帧 58 / explicit 6 / latent 16 / 动作 13），旧 v2 一律被拒；上表保留为当时的记录。
> README 问题表 **[TOW-26](../README.md)** 就是这次记录失真 + 接线的收口。

链路与 `upper_mdp.HierarchicalVelocityAction` 同序（偏移进底层**输入**、残差加底层**输出**）：

    task_command ──(速度头: +有界偏移, 裁进 AMP 包络 −1.0…1.5)──→ loco_command
                                                                      │
                                                          （冻结策略推理，50 Hz）
                                                                      ↓
                                                             冻结关节目标
                                                                      │
                                        (+ 12 维关节残差 → PD 位置目标) ┘
                                                                      ↓
                                                  set_joint_position_target

| 项 | 测试台实现 | 出处 |
|---|---|---|
| 帧 | **58** = `loco_command(3) + last_action(13) + base_ang_vel·0.25(3) + projected_gravity(3) + last_loco_action(12) + joint_pos−default(12) + joint_vel·0.05(12)`；`last_action` 由 `UpperPolicyRuntime` 自己持有（**上一拍** clamp 后的 13 维动作），`loco_command` 取上一拍合成的那条 | `upper_mdp.policy_frame`、`upper_policy_runtime.FRAME_TERMS`（离线测试逐项核对） |
| 动作 | **13** = 1 维 vx 偏移 + 12 维关节残差 | `upper_mdp.process_actions`、`upper_policy_runtime.ACTION_DIM` |
| 偏移（第 1 层） | `offset = clamp(clamp(u_cmd, ±1) × 0.5, −0.2, +0.6)` m/s —— 限的是**偏移量本身**，不是「和」 | `upper.command_offset_vx`（与 cfg 的 `cmd_offset_scale`/`offset_min`/`offset_max` 同值） |
| 合成（第 2 层） | `loco_vx = clamp(task_vx + offset, −1.0, +1.5)`（冻结 AMP 策略的训练包络），再拼回 `loco_command = (loco_vx, task_vy, task_wz)` | `upper.compose_loco_vx` / `play_towing_test.compose_loco_vx_offset` |
| 门控 | `elapsed_s = step · dt`，`elapsed_s >= tow_start_s`（训练侧 `tow_start_s = SETTLE_TIME_S = 1.0 s`）才叠加；**STOP 之后仍生效**（`task_vx = 0` ⇒ `loco_vx = offset`） | `upper_mdp.process_actions` 的 `offset_active` |
| 顺序 | 合成偏移 → 喂**冻结策略** ⇒ 冻结关节目标 → `held = 冻结目标 + delta`（12 维残差） | `upper_mdp.apply_actions`（`velocity_command=self.loco_command`） |
| JNT 参考 | `robot_jt_*` = `held`（真正下发的目标），同训练侧 `reference="commanded"` | `upper_mdp.low_level_position_error_l2` |
| 节拍 | 上层每 `upper_control_decimation`（= `upper_control_dt/dt` = 10 步 = 50 ms）一拍，**与冻结策略的 4 步刷新（20 ms）独立**；偏移与残差在两次上层 tick 之间保持不变 | `UpperTowingEnvCfg.decimation = 10`、`apply_actions` 的 `low_level_decimation = 4` |

**退化性（关开关）**：`upper = None` 时主循环只走基线分支——`command_tensor = lane_command(vx_command)`
（任务指令，含 PD 的 vy/wz）**原样**喂冻结策略、下发目标 = 冻结策略自己的输出，残差与偏移恒 0；
本轮新增的代码全部落在 `else`（开关开）分支。离线证据三条：① 基线分支被 AST 测试**逐字钉住**
（`test_switch_branch_order_offset_then_frozen_then_residual`）；② `compose_loco_vx_offset(task, 0)`
与任务指令 `torch.equal`（`test_compose_is_bit_identical_when_the_offset_is_zero`）；③ 关时
`experiment.json.upper_policy` 仍然记下 v3 期望契约（`towing_contract_expected` + 偏移边界）。
**运行读数（两轮 800 环境的逐 case 对照）只能由训练机给出。**

改动清单：

- `play_towing_test.py`：docstring / `[plan]` 行 / `experiment.json.upper_policy` 一律改 v3
  （帧 58、动作 13、偏移两层限幅 + 门控，TOW-26）；新增 `upper_policy_record()` 把期望契约与
  偏移边界做成**可离线核对的纯函数**；实跑时用 `expected_towing_contract(upper.spec)` 覆写期望值
  并与离线镜像**逐字段核对**（不一致直接 `RuntimeError`，防两处漂移）；主循环按上表接线
  （上层 tick 与冻结刷新解耦、偏移在策略推理前合成、残差在输出后叠加）。
- `upper_policy_runtime.py`：只改两处 docstring（「测试台 13 维接线由用户随后补」→ 已接线）。
- 新增 10 项离线测试（`UpperV3WiringTests`）：契约/偏移常量与运行时逐项一致、v2 被拒、
  门控阈值 = `SETTLE_TIME_S`、偏移为 0 时逐位相等、门控/两层限幅、与运行时
  `compose_loco_vx` **逐位相等**、`upper_policy_record()` 的字段、链路顺序 + 基线路径逐字钉住。
- **未验证项照旧**：本机无 Isaac Lab（无 GPU），**仿真一次都没跑**——偏移门控/两层限幅的实际
  读数、20 Hz 上层与 50 Hz 冻结策略的解耦、真实 v3 checkpoint 的 strict 加载、以及两轮
  800 环境的策略/基线读数都要训练机验证。

## 验收（用 towing test 口径，全网格 800）

```bash
# 第 1 轮：基线（开关关），全网格 800；--no-cart-fraction 0.125 与归档基线的负载配比一致
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/towing/play_towing_test.py \
    --headless --num-envs 800 --no-cart-fraction 0.125 \
    --output-dir imgo2_rl/logs/towing/play_test/upper_switch_baseline

# 第 2 轮：策略（开关开），同一套参数 + 第 1 轮的 report.json 做同版本对照列
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/towing/play_towing_test.py \
    --headless --num-envs 800 --no-cart-fraction 0.125 \
    --upper-checkpoint logs/towing_rl_lab/towing_upper/<run>/model_1000.pt \
    --compare-report imgo2_rl/logs/towing/play_test/upper_switch_baseline/report.json \
    --output-dir imgo2_rl/logs/towing/play_test/upper_switch_policy
```

判定仍用同一套阈值/判定码；`report.md` 的「策略 vs 基线」一节直接给出两轮（以及归档参照）
的通过数、判定码分布、五项指标中位数与逐坡度通过数。**验收结论只读 `COL`/`SPD`/`LOW`
（以及 `LAT`/`FALL`）**：归档基线里 `JNT` 在有负载（295/700）与无负载（100/100）两边都
接近 100%，量的是 `q − q*` 的 PD 静差（≈ τ/kp）、阈值未标定；而且开了开关后 `JNT` 的参考量
变成 `held`，与基线的冻结输出不是同一个量。报告里对这两点都有显式提示。

## 验证（本机能做的）

- `python3 -m pytest imgo2_rl/tests -q` ⇒ **473 passed**（基线 437 + 运行时 24 + 测试台 12）。
- 新增 `imgo2_rl/tests/test_towing_upper_policy_runtime.py`（24 项，torch 可用时全跑）：
  帧布局与 `UpperObservationSpec.terms` 及 `policy_frame` 源码 AST 逐项一致；actor 79 /
  critic 72（后者按 `upper_env_cfg.CriticCfg` AST 求和）；注册 cfg 的 policy/decoder kwargs 与
  `UpperNetworkSpec` 默认值逐项一致、`clip_actions == 1.0`；合成 checkpoint 往返
  （零初始化 actor ⇒ 首拍残差精确 0；饱和 ⇒ `delta == action_scale` 且 `|delta| ≤ scale`；
  `_last_action` 次拍等于上一拍 clamp 后的动作；帧切片逐段数值核对）；`reset()` 全量/部分；
  contract 不匹配（含 56/63、version、latent、非 dict）与缺键、缺文件、伪造契约被 strict 加载拒绝；
  模块顶部无 isaaclab、只有 stdlib+torch。
- `test_towing_play_test.py` 新增 12 项：默认关、`--dry-run` 的 plan 行显示开关与契约、
  缺文件/单独 `--upper-stochastic` 报错、`--compare-report` 的 CLI 校验与 plan 行、
  开关关时不构造 runtime（AST：构造点唯一且在 `if args.upper_checkpoint is not None:` 里）、
  held 唯一且只在开关开的分支里、模块顶部无 torch、可追溯字段（experiment/report/plan）、
  「策略 vs 基线」汇总（通过数/判定码/中位数/样本数/逐坡度）、归档常量与
  `docs/towingdata/2026-10-09_necessity_800_noload/report.json` 复算值逐项一致、
  对照节的 JNT/出处/不可硬比提示、**判据与阈值函数里没有任何开关分支**（口径不变）。
- `python3 imgo2_rl/scripts/towing/play_towing_test.py --dry-run` 仍可跑（无 torch / 无 GPU），
  plan 行输出「上层网络：**关闭**…」；加 `--upper-checkpoint <存在的文件>` 后输出「**开启**…」
  且 `torch` 不在 `sys.modules` 里（用 `runpy` 验证）。

**2026-10-10 晚（v3 接线，见上节）**：全量 `python3 -m pytest imgo2_rl/tests -q` ⇒
**564 passed**（553 基线 + 11 项：`UpperV3WiringTests` 10 项 + `--dry-run --upper-checkpoint` 1 项：契约/偏移常量与运行时逐项一致、
v2 契约被拒、门控阈值 = `SETTLE_TIME_S`、偏移 0 时逐位相等、门控与两层限幅、
与运行时 `compose_loco_vx` 逐位相等、`upper_policy_record()` 字段、链路顺序 + 基线路径逐字钉住）；
`--dry-run`（默认与 `--upper-checkpoint` 两条路径）都退出 0 且 `torch` 不在 `sys.modules`；
`py_compile` / `git diff --check` / `git ls-files -i -c --exclude-standard` 通过。**运行期仍然未验证。**

## 未验证（必须写清）

- 本机没有 Isaac Lab（无 GPU 可见性）：**仿真一次都没跑**。开关的运行时行为（残差接到 PhysX、
  上层 20 Hz 节拍、`held` 下发与 JNT 参考、与训练场景的资产/关节顺序对接）全部待训练机实跑；
- 「关时逐位一致」是**结构性**结论（基线分支未改、残差恒 0），不是运行读数；
- checkpoint 只用手工构造的合成件验证；**真实 `model_*.pt` 的 strict 加载未跑**（需要训练机上的
  真实 run）；
- 「策略 vs 基线」一节只在离线用**归档 report.json** 渲染过（`case_metric_summary` /
  `format_baseline_comparison` / `build_markdown_report` 全链路），两轮真实 run 与
  `--compare-report` 的运行时路径（含 9 MB report.json 的读取）未在训练机跑过。

## 训练机命令清单

验收用的两轮 800 命令见上面「验收（用 towing test 口径，全网格 800）」一节（基线一轮 +
`--compare-report` 的策略一轮，参数除开关外完全一致；本测试台是确定性的，没有随机种子）。
其余辅助命令：

```bash
# 冒烟（先跑这个，确认契约/加载/节拍）：8 环境、短回合、全写 CSV
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/towing/play_towing_test.py \
    --headless --num-envs 8 --tow-duration 2.0 --coast-duration 2.0 --write-csv all \
    --upper-checkpoint logs/towing_rl_lab/towing_upper/<run>/model_1000.pt

# 无 GPU 机器可先看开关状态（只校验 checkpoint 路径，不 import torch）
python3 imgo2_rl/scripts/towing/play_towing_test.py --dry-run \
    --upper-checkpoint logs/towing_rl_lab/towing_upper/<run>/model_1000.pt
```

注意：`--output-dir` 是可选参数（默认时间戳目录），两轮分开便于对比；`--write-csv failed`
只留失败 case 的逐记录步轨迹，做逐 case 归因时改 `all`（800 环境约 534 MB）。

## 开开关后要对比哪些字段

1. **先确认开关真的生效**：`experiment.json.upper_policy.loaded == true`、`contract`、
   `iter`、`residual_scale`（12 项）与 checkpoint 一致；`[plan]`/`[upper]` 两行打印；
   逐 case 的 `robot_jt_*` 与基线不再相同（残差非 0）。
2. **判定码**：`report.json.groups[*].reasons` 的 `COL` / `SPD` / `JNT` 占比（按坡度量级），
   以及 `report.csv` 逐 cell 的 `verdict` 对照表。
3. **跟速（主战场）**：`speed.mae_mps`、`rmse_mps`、`bias_mps`、`ratio_mean`、
   `p95_abs_err_mps`、`steady_mae_mps`。
4. **追尾**：`stop.cart_coast_distance_m`、`min_clearance_coast_m`、`final_clearance_m`、
   `contact`、`time_to_contact_after_stop_s`、`cart_coast_to_rest_m`/`_time_to_rest_s`。
5. **关节响应（口径已变，别当纯改善）**：`startup.joint_rms_rad`/`joint_max_rad`/
   `torque_saturated_frac`、`stop.joint_rms_rad`/`settle_time_s`——基线参考冻结输出、策略参考
   `held`；力矩饱和占比上升说明残差在顶底层动作。
6. **横向/朝向**：`lane.y_rms_m`、`y_max_abs_m`、`heading_rms_rad`、`vy_saturated_frac`、
   `wz_saturated_frac`（PD 与上层策略同轮起作用，限幅占比上升 = PD 在硬顶）。
7. **稳定性**：`stability.fell`、`min_robot_surface_height_m`、`max_abs_pitch_rel_rad`、
   `invalid_samples`。
8. **归因**：同一 case 的两轮 `tow.csv`（`--write-csv all`）看残差改了什么；
   `play_test_stats.py` 读 `report.json` 可做分组/交叉统计。

限制：`JNT` 的参考量在两种开关下不是同一个量，`startup.joint_rms_rad` 的差**不是**纯粹的
跟踪改善；阈值仍是未标定的占位值（见 TOW-10/TOW-11）。
