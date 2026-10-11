#!/usr/bin/env python3
"""拖曳「上层任务必要性」基线测试台（直接跑**训练场景**的 Isaac Sim play 仿真）。

## 这个脚本回答什么问题

上层拖曳 RL（脚本调度的速度指令 + 12 维关节位置残差）这个任务**有没有必要**？
必要性的下限来自它的对照基线：**冻结 AMP 底层策略 + 脚本速度指令**（settle 0 →
tow v → STOP 0），负载是仓库里那台被动小车，连接是三类（弹性绳 / 低弹性绳 / 刚体球铰
连杆）。如果这条基线在本脚本的网格上已经满足全部五项指标，那么上层任务在这些工况上
就没有必要性证据；如果它系统性地在某一相失败，失败模式就是上层任务要修的对象。

**默认不加载上层 checkpoint**：跑的是基线（关节位置残差恒 0），不是策略回放。
给 `--upper-checkpoint PATH` 就打开上层网络（见下节），用**同一轮 run、同一网格、同一指标**
做「策略 vs 基线」对照（TOW-19 里记录的待办）。

## 上层网络开关（`--upper-checkpoint`，默认关）

- **关（默认）**：残差恒 0、偏移恒 0（`loco_command` ≡ 任务指令），行为与加开关之前
  **逐位一致**——每 20 ms 把冻结策略的关节位置目标原样下发；`JNT` 观测的口径不变
  （参考量 = 冻结策略当拍输出），但**默认不进判定**（用户 2026-10-10 决定，见「判读与限制」；
  `--count-jnt` 可复现旧口径）。
- **开（给了 checkpoint 路径）**：从训练侧联合 checkpoint 恢复 actor + dynamics decoder
  （**v3 双头契约**：58 维 policy 帧 → 6 维显式估计 + 16 维 latent → 80 维 actor →
  **13 维动作** = 1 维 vx 偏移 + 12 维关节残差），并按**训练侧的链路顺序**接线
  （`upper_mdp.HierarchicalVelocityAction`）：

      task_command ──(速度头: +有界偏移, 裁进 AMP 包络 −1.0…1.5)──→ loco_command
                                                                        │
                                                            （冻结策略推理，50 Hz）
                                                                        ↓
                                                                冻结关节目标
                                                                        │
                                          (+ 12 维关节残差 → PD 位置目标) ┘
                                                                        ↓
                                                    set_joint_position_target

  1. 上层一拍（20 Hz）先组 **58 维帧**：`loco_command(3)` + `last_action(13)` +
     `base_ang_vel·0.25(3)` + `projected_gravity(3)` + `last_loco_action(12)` +
     `joint_pos−default(12)` + `joint_vel·0.05(12)`；`last_action` 取**上一拍** 13 维动作，
     `loco_command` 取**上一拍合成、这一拍仍在驱动底层**的那条指令（训练侧观测在 action
     之前算，看到的也是上一拍 `process_actions` 的结果）；
  2. 取动作**第 1 维 = vx 偏移**，两层限幅合成送冻结策略的指令：先限**偏移量本身**
     `[offset_min, offset_max] = [−0.2, +0.6] m/s`（`u_cmd` 先 clamp 到 ±1 再乘
     `cmd_offset_scale = 0.5`），再把**和**裁进冻结 AMP 策略的训练包络
     `amp_vx_range = (−1.0, 1.5)`；**门控 `elapsed_s >= tow_start_s`**（出生段不叠加偏移；
     **STOP 之后偏移仍然生效** ⇒ `task_vx = 0` 时 `loco_vx = offset`，这是「停机续走」的
     表达口）；
  3. 合成后的 `loco_command`（含 PD 的 vy/wz）喂**冻结策略** ⇒ 冻结关节目标；
  4. **再加 12 维残差**：`delta = clamp(action, ±1)[1:] ⊙ action_scale`，
     `held = 冻结目标 + delta`。**`held` 既下发、又作为 `JNT` 指标的参考量**，与训练侧
     `low_level_position_error_l2(reference="commanded")` 同口径。偏移加在**底层输入之前**、
     残差加在**底层输出之后** ⇒ 两头不独立（别名/冗余，见 `upper_mdp` 的说明）。
- 上层网络在训练侧是 **20 Hz**（`upper_control_dt = 0.05 s`），冻结策略是 50 Hz；
  本脚本沿用该节拍：每 10 个物理步（50 ms）推理一次上层（与冻结策略的 4 步刷新**独立**，
  与训练侧 `process_actions`（env step 驱动）/ `apply_actions`（物理步驱动）同一结构），
  两次之间偏移与残差都保持不变；冻结策略每 4 个物理步刷新一次并重算 `held`
  （同 `HierarchicalVelocityAction.apply_actions`）。`--upper-stochastic` 时用 actor 的分布
  采样（默认取均值，与 `towing/play.py` 的确定性口径一致）。
- **契约校验（加载前硬校验，不匹配就抛错）**：`towing_contract` 必须精确等于
  `{'version': 3, 'frame_dim': 58, 'explicit_dim': 6, 'latent_dim': 16}`；网络超参从注册的
  agent cfg（`agents/upper_ppo_cfg.py::UpperTowingPPORunnerCfg`）建，然后
  `load_state_dict(strict=True)`。**旧 v2（帧 57 / actor 79 / 动作 12）与更早的 51/56/63 维
  checkpoint 会被拒绝**（`TowingCheckpointContractError`）。
- **未验证**：本机没有 Isaac Lab（无 GPU），仿真一次都没跑过；开关的运行时行为、
  偏移/残差幅值、策略/基线对照读数都要在训练机实跑。见 README TOW-20 / **TOW-26**。

## 场景 = 训练场景（2026-10-09 起，用户确认）

场景直接用 `upper_env_cfg.UpperTowingSceneCfg`（`InteractiveScene`），**不再自建平地/
地形、也不再自己造扫描网格**：

- `--num-envs`（默认 800 = 训练网格全集 `connection_grid.GRID_SIZE` = 40 列 × 20 行，
  行距/列距由 `slope_geometry.ROW_SPACING_M` / `COLUMN_SPACING_M` 拼成一张连续地面）：
  环境 `i` 的**连接类型 / 连接长度 / 坡度量级**直接取 `connection_grid.env_spec(i)`
  （row-major：列 = i % 40、行 = i // 40），与训练时逐 env 的分配**逐位一致**。
- 地形 mesh、机器人与小车资产、五个「车体 vs 机器人」接触传感器、物理材质、重力
  （世界竖直）全部沿用训练配置；本脚本只覆盖**物理量标量**（地面摩擦、轮轴阻尼）与
  逐 env 的工作条件。
- 出生点统一在每条 lane 剖面的**平地段起点**（姿态竖直、单位四元数，lane 系 = 世界系）；
  两挂点三维距 = `env_spec(i)["initial_distance"]`（绳 = 0.8·L0、刚体 = L，2026-10-09 出生比由
  0.5 提到 0.8、且绳/杆长度区间解耦），由纯函数
  `spawn_offsets()` 逐 env 解出。
- **不再有 `--slope-backend` / `--slopes` / `--connections` / `--slope-settle`**：
  坡度量级与连接类型都是网格给定的，没有可扫的开关。

## 只扫「速度 × 质量」：确定性轮转

连接/长度/坡度由 cell 决定，所以工作条件只剩两项。`work_conditions()` 按**确定性轮转**
分配（`slot = row + column`：质量 `masses[slot % len]`、速度 `velocities[(slot // len(masses)) % len]`）。
**用 `row + column` 而不是 env 序号是关键**：列数 40 是质量档数的整数倍，用序号轮转会让质量只由
列决定，而列同时决定连接类型与坡度量级 ⇒ `(连接, 质量)` 严重混淆（inextensible 会完全没有
5 kg）。改后固定一列时 `slot` 随 20 行取 20 个连续整数，于是

- 800 环境下**每个 cell 恰好落到一种 (速度, 质量) 组合**（cell 与 env 一一对应）；
- 每个 (速度, 质量) 组合 50–55 个 env（`slot` 取值个数两头少中间多，不是严格 ±1）；
- **每个 (连接 × 质量) 与 (坡度量级 × 质量) 都严格均衡**（实测 compliant/rigid 各档 64 个、
  inextensible 各档 32 个；0° 各档 80 个、5°/10° 各档 40 个），每列都覆盖全部 15 个组合；
- 分配是**纯函数**，`--dry-run` 与实跑得到同一张表。

质量档默认 `5, 10, 15, 20, 25 kg`。训练侧 `reset_work_condition` 的 `mass_range` 于
**2026-10-09 由 (5, 15) 提到 (5, 30) kg**、**2026-10-10 收紧为 (5, 20) kg**（用户确认工作域＝
质量 5–20 kg、速度 0.5–1.5 m/s）⇒ 默认档里的 **25 kg 现在是域外**（20 kg 仍在域内），
正好当作外推检查档保留；要做更强的外推检查可显式传更大的 `--cart-masses`（>20 kg）。
`--velocities` 默认 `0.5/1.0/1.5` 与新下界 0.5 m/s 一致，无需改。

## 五项指标（每一项在 summary 里都有明确字段）

| 指标 | 字段 | 口径 |
|---|---|---|
| 起步关节响应误差 | `startup.joint_rms_rad` / `joint_max_rad` / `worst_joint` / `torque_saturated_frac` | 起拖后 `--transition-window`（默认 1.0 s）内，12 关节 `q − q*` 的 RMS 与最大绝对值；`q*` 是**当拍真正下发的关节位置目标**（基线 = 冻结策略输出；开了 `--upper-checkpoint` 则 = 冻结输出 + 上层残差 `held`，与训练侧 `reference="commanded"` 同口径），另报力矩饱和（`|τ| > 0.95·limit`）步占比 |
| 全程速度跟踪误差 | `speed.mae_mps` / `rmse_mps` / `bias_mps` / `ratio_mean` / `p95_abs_err_mps` | tow 段**全程**，**体系** x 速度（与冻结策略的观测同口径；坡上 ≠ 世界系速度）与指令之差；另有后半段稳态窗 `speed.steady_*` |
| 停止时小车滑移距离 | `stop.cart_coast_distance_m` / `cart_coast_to_rest_m` / `cart_coast_time_to_rest_s` / `cart_speed_at_stop_mps` | 从 STOP 那一刻到回合结束小车沿 x 的位移；以及速度降到 0.02 m/s 以下那一刻的距离与耗时 |
| 停止时机器人—小车距离维持 | `stop.clearance_at_stop_m` / `min_clearance_coast_m` / `final_clearance_m` / `time_to_contact_after_stop_s` / `contact` | **车头到机器人后腿的真实几何间隙**（全腿 FK，复用 `summarize_tow.py`，挂点距单独报）；接触由「车斗接触力 / 几何间隙 / 负载单步速度跃变」三路见证判定 |
| 停止时的关节响应 | `stop.joint_rms_rad` / `joint_max_rad` / `worst_joint` / `settle_time_s` / `body_vx_rms_mps` | STOP 后 `--transition-window` 内的同一套关节跟踪误差 + 机器人从指令归零到 `|vx| < 0.05 m/s` 的耗时 |

判据（阈值可用 CLI 覆盖）落成逐 case 码：`OK` / `LOW`（停车余量低）/ `LAT`（横向或朝向
保持超限）/ `SPD`（速度跟踪超限）/ `COL`（追尾接触）/ `FALL`（跌倒）/ `INV`（记录不可用）；
`JNT`（关节响应超限）**默认不计入判定**（用户 2026-10-10 决定），只有 `--count-jnt` 时才评估。

## 横向/朝向保持（`--lane-keeping pd`，默认开）

冻结策略的速度指令是 3 维 `(vx, vy, wz)`：`vx` **只由脚本调度给出、不参与任何反馈**；
`vy`/`wz` 由横向/朝向 PD 生成，把机器人压在 lane 中线（lane 系 `y = 0`）并保持超前
（`yaw = 0`，即 lane 的 +x 方向）。默认 `kp_y=1.0 kd_y=0.3 kp_yaw=1.5 kd_yaw=0.3`，
限幅 `vy ≤ 0.4 m/s`、`wz ≤ 0.8 rad/s`（都在 AMP 训练过的指令范围内）。`--lane-keeping off`
回到「只给 vx」的旧行为做对照。

## 度量坐标系（固定 lane 系）

地形是连续剖面（`slope_geometry.profile_*`），出生在平地段 ⇒ 切向 = +x、法向 = +z、
重力世界竖直。记录里的

- `robot/load_progress_m` = lane 系 x 行程；
- `robot/load_surface_height_m` = `z − profile_height(本 env 的坡度量级, x)`（**逐 env**
  的坡度量级参与换算，0° lane 上就是绝对 z）；
- `body_pitch_rel_rad` = 世界系俯仰 + **局部**坡度（`profile_slope_degrees`）。

## 运行

```
# 只看将执行的网格与逐 env 分配，不启动仿真（标准库即可）
python3 imgo2_rl/scripts/towing/play_towing_test.py --dry-run

# 默认：800 环境一次跑完（headless 建议）
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/towing/play_towing_test.py --headless

# 冒烟：40 环境（网格前缀，只覆盖前 40 个 cell）
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/towing/play_towing_test.py \
    --num-envs 40 --tow-duration 2.0 --coast-duration 2.0 --write-csv none

# 对照：不做横向反馈 / 斜坡整形 / 只留失败轨迹
    ... --lane-keeping off
    ... --command-shaping ramp --ramp-time-s 1.0
    ... --write-csv failed|all|none

# 策略 vs 基线：开关打开（同一网格、同一指标；`--dry-run` 也可先看开关状态）
python3 imgo2_rl/scripts/towing/play_towing_test.py --dry-run \
    --upper-checkpoint logs/towing_rl_lab/towing_upper/<run>/model_1000.pt
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/towing/play_towing_test.py --headless \
    --upper-checkpoint logs/towing_rl_lab/towing_upper/<run>/model_1000.pt
# 需要采样动作（而不是均值）时另加 --upper-stochastic
```

**对照的正确做法是跑两轮**（本测试台一次只跑一种模式：每个 env 只跑一个 episode，同一进程
里没有第二次 rollout 可用；所以不做「同轮切换」，改成**两次运行 + 报告里对照**）：

```
# 第 1 轮：基线（不给 --upper-checkpoint）
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/towing/play_towing_test.py \
    --headless --num-envs 800 --no-cart-fraction 0.125 \
    --output-dir imgo2_rl/logs/towing/play_test/upper_switch_baseline
# 第 2 轮：策略（同一套参数 + 那个 checkpoint + 把基线的 report.json 传进来做同版本对照）
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/towing/play_towing_test.py \
    --headless --num-envs 800 --no-cart-fraction 0.125 \
    --upper-checkpoint logs/towing_rl_lab/towing_upper/<run>/model_1000.pt \
    --compare-report imgo2_rl/logs/towing/play_test/upper_switch_baseline/report.json \
    --output-dir imgo2_rl/logs/towing/play_test/upper_switch_policy
```

两轮除开关（与 `--compare-report`）外参数必须完全一致（本测试台是确定性的，没有随机种子）。
`report.md` 与 `report.json` 的「策略 vs 基线」一节给出三项同口径数字：本轮、同版本基线
（`--compare-report`）、归档基线 2026-10-09（旧几何，只作参照、**不可逐格硬比**）。

## 产物

```
experiment.json       场景来源、逐 env 的 cell 与工作条件分配、阈值、git
report.md             人读：逐坡度量级判定矩阵、分组统计、结论、限制
report.csv            每 env 一行（扁平指标）
report.json           逐 case 摘要 + 分组统计 + 结论
summaries/<case>.json 单个 case 的全量指标（含 summarize_tow 全量输出）
<case>/tow.csv        逐记录步原始轨迹 + config.json（按 --write-csv 策略）
```

`--record-every` 默认 **5**（25 ms = 冻结策略周期）：800 环境 × 2200 步逐物理步记录是
176 万行（Python dict 约 2 GB 内存），不可行；接触的速度跃变阈值已按记录步长自动放大。

## 冲击窗口 vs 稳态（`--impact-window` / `--steady-margin-s` / `--takeup-force-threshold`）

**为什么**：只报全程/牵引段的关节 RMS 会把起步与停车的冲击**平均掉**，看不出安全性改善。
所以把四个**不同时刻**各取一个窗口独立统计，并给出与稳态的比值/增量（2026-10-10 用户要求）：

| 窗口 | 对齐到什么 | 取值范围 |
|---|---|---|
| `impact.startup`（`impact_startup_*`） | **起拖**（tow 段首行 = 速度指令阶跃那一刻） | `[t0, t0 + W]` |
| `impact.takeup`（`impact_takeup_*`） | **真实绷直/穿绳时刻**：首个 `\\|rope_tension_n\\| > 阈值` 的样本 | `[t − W, t + W]` |
| `impact.stop`（`impact_stop_*`） | **指令归零**（coast 段首行） | `[t0, t0 + W]` |
| `impact.stop_contact`（`impact_stop_contact_*`） | **真实停车撞击**：coast 段首个接触见证样本（车斗接触力或负载速度跃变） | `[t − W, t + W]` |
| `impact.steady`（`impact_steady_*`） | 牵引段**去掉首尾各 `--steady-margin-s`** | 分母 |

`W = --impact-window`（默认 0.2 s）。每个窗口报 `joint_rms_rad` / `joint_max_rad` /
`worst_joint` / 逐关节 `per_joint_rms_rad` / `torque_saturated_frac`，再加
`rms_over_steady`（比值，>1 = 比稳态猛）、`rms_delta_rad`（增量 [rad]）与逐关节比值。
`takeup` 另带 `time_s` / `tension_n` / `vx_mps`，`stop_contact` 另带 `time_s` / `channels` /
`deck_fx_n`。**归零窗口与撞击窗口是两件事**（实测可能差数百毫秒），不要互相替代。

**口径冻结（冲击部分）**：`--transition-window`（默认 1.0 s）的既有 `startup.*` / `stop.*` 字段
与 `DEFAULT_THRESHOLDS` 的**数值**不动，新增字段只进 `metrics["impact"]`；`impact_stats()` 里
没有阈值字典、也不做任何判定分支。（**例外**：`classify_case` 于 2026-10-10 增加了
`count_jnt` 开关，默认把 `JNT` 移出判定，见下节「判读与限制」；JNT 字段与阈值本身仍在。）
所以 `docs/towingdata/2026-10-09_necessity_800*/` 的归档对照**在 JNT 这一点上不可比**
（归档按旧口径算），报告里那列没有 `impact.*` 只能显示 `n/a`。
读法见表头（`report.md` 的「## 冲击窗口 vs 稳态（关节响应）」）。

`report.md` 增一节「## 冲击窗口 vs 稳态（关节响应）」（起拖 / 绷直 / 指令归零 / 停车撞击四行
+ 稳态一行 + `RMS/稳态`、`RMS−稳态` 两列），`report.json` 增顶层 `impact_statistics`（本轮 /
同版本基线的逐 case 中位数、样本数、最差关节众数），`report.csv` 增 `impact_<窗口>_*` 扁平列。
详见 README 问题表 **TOW-22** 与
[双头方案记录](../docs/towing_upper_two_head_2026-10-10.md)（本机无 Isaac Lab，只做了离线验证）。

## 判读与限制

- 结论只由「本网格 + 本阈值」给出，不能外推：未测训练侧的域随机化（质量档 **25 kg 超出**
  训练域 5–20 kg，20 kg 仍在域内；摩擦固定 0.8 而训练随机 0.4–1.2、轮轴阻尼固定 0.032
  而训练随机 0.008–0.032）、
  未测 **12.5% 无小车锚点**（本测试台所有 env 都拖车）、未测跨环境隔离、未验真机。
- 连接类型/长度/坡度量级**不能扫**：它们由网格决定，`--num-envs` 不是 800 的整数倍时
  只覆盖网格前缀（脚本会警告）。
- `summarize_tow` 的 `steady_tracking_ratio` 用的是**世界系** vx 除以指令，坡上口径不同，
  本脚本的跟速指标一律用**体系** vx；两者都写在产物里，不要混用。
- **JNT 已移出判定统计量（用户 2026-10-10 决定）**：`startup_joint_error` / `stop_joint_error`
  不再产生失败原因、`JNT` 不再出现在判定码分布里；两条观测字段与关节阈值照旧保留并打印
  （观测，不是判据）。原因：这两条在 800 cell 上基线 800/800、策略 800/800，零区分度，
  量的是底层 PD 静差 τ/kp（阈值 0.10 rad 连最好的 case 都超 1.6 倍），不是上层任务的职责。
  要复现旧口径（与归档对齐/追溯）加 `--count-jnt`，行为逐位一致。
- 速度指令是体系 x 速度（与冻结策略观测一致），只研究直线拖曳。

## 策略 vs 基线要对比哪些字段（`--upper-checkpoint` 对照用）

同一格式的产物逐项比两轮（基线 vs 策略），按优先级：

| 组 | 字段（`report.json.cases[].metrics`；`report.csv` 里加了 `startup_/speed_/stop_/lane_` 前缀） | 看什么 |
|---|---|---|
| 判定码 | `verdict.code` / `verdict.reasons`（分组计数在 `report.json.groups[*].reasons`） | 分组（坡度量级）里 `COL`（追尾）/`SPD`（跟速）的占比是否下降；`JNT` 默认已移出统计量（`--count-jnt` 才计入） |
| 跟速 | `speed.mae_mps`、`speed.rmse_mps`、`speed.bias_mps`、`speed.ratio_mean`、`speed.p95_abs_err_mps`、`speed.steady_mae_mps`、`speed.steady_ratio_mean` | 拖曳代价的主战场：策略应把 MAE/偏差压下来（尤其重车 + 陡坡 + 高速） |
| 追尾 | `stop.cart_coast_distance_m`、`stop.cart_coast_to_rest_m`、`stop.cart_coast_time_to_rest_s`、`stop.min_clearance_coast_m`、`stop.final_clearance_m`、`stop.contact`、`stop.time_to_contact_after_stop_s` | 停车段是否还撞上来；滑移衰减（`*_to_rest_*`）是否变好 |
| 关节响应（观测） | `startup.joint_rms_rad`、`startup.joint_max_rad`、`startup.worst_joint`、`startup.torque_saturated_frac`、`stop.joint_rms_rad`、`stop.joint_max_rad`、`stop.settle_time_s` | **默认不进判定**（`JNT` 已移出）；只作观测：口径随开关改变（基线参考冻结输出、策略参考 `held`），**不要把它当纯粹的"改善"**；要同时看 `torque_saturated_frac` 是否恶化（残差顶掉底层动作的信号） |
| 横向/朝向 | `lane.y_rms_m`、`lane.y_max_abs_m`、`lane.heading_rms_rad`、`lane.heading_max_abs_rad`、`lane.vy_saturated_frac`、`lane.wz_saturated_frac` | PD 外环与上层策略同在一轮里起作用；限幅占比上升说明 PD 在硬顶 |
| 稳定性 | `stability.fell`、`stability.min_robot_surface_height_m`、`stability.max_abs_pitch_rel_rad`、`stability.pitch_over_limit_frac`、`stability.invalid_samples` | 策略不能把机器人开倒，也不能让记录失效 |
| 策略自身 | `experiment.json.upper_policy`（checkpoint/契约/iter/确定性）与逐记录步 `robot_jp_*`（实测）− `robot_jt_*`（= 实际下发目标） | 先确认 checkpoint 与契约对上了；再用同一 case 的两轮 `tow.csv` 看残差到底改了什么 |
| 归因 | 两轮 `report.csv` 的 `verdict` 做成逐 cell 对照表（`play_test_stats.py` 可读 `report.json`） | 逐 cell 看「基线失败 → 策略通过/更差」的分布，而不是只看均值 |

**注意**：`JNT` 的参考量在两种开关下不是同一个东西（基线 = 冻结输出、策略 = 下发目标），
所以 `startup.joint_rms_rad` 的差**不是**纯粹的"跟踪改善"；阈值也仍是未标定的占位值。
**验收结论只读 `COL`/`SPD`/`LOW`（以及 `LAT`/`FALL`）**：`JNT` 已按用户决定移出统计量
（800/800 零区分度，量的是 PD 静差 τ/kp）；重判读数（同一批 800 cell，仅剔除 JNT）见
README TOW-23 与 `docs/towing_verdict_jnt_excluded_2026-10-10.md`。

`report.md` / `report.json` 里的「策略 vs 基线」一节把本轮、同版本基线（`--compare-report`）、
归档基线 2026-10-09 三项放在一起（同一个 `case_metric_summary`：通过数、判定码分布、
五项指标中位数、逐坡度量级通过数）；归档那列是**旧几何**（绳 0.6–1.2 m、出生比 0.5）
**且是旧口径**（JNT 计入判定），只作 sanity check，**不可逐格硬比**。
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

RL_ROOT = Path(__file__).resolve().parents[2]          # <repo>/imgo2_rl
REPO = RL_ROOT.parent                                  # <repo>
TOOLS_DIR = RL_ROOT / "scripts" / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from summarize_tow import summarize_tow  # noqa: E402  (纯标准库，离线工具)

# ---------------------------------------------------------------- 常量与默认值

GRAVITY_MPS2 = 9.81
#: 连接类型全集。与 `mdp/rope_model.py::CONNECTION_MODELS` 交叉核对（离线测试覆盖），
#: 这里不 import 那个模块：它 import torch，会让 `--dry-run`／`--help` 失去标准库可运行性。
CONNECTIONS = ("compliant", "inextensible", "rigid")
#: 工厂地面常规摩擦系数（混凝土/环氧地坪 0.6–0.9，本仓库历来固定 0.8）。
FACTORY_FLOOR_FRICTION = 0.8
#: 训练侧的轮轴黏性阻尼名义值（`HierarchicalVelocityActionCfg.initial_wheel_damping`）。
#: 这里只是 CLI 默认值；实跑时会与训练 cfg 交叉核对，不一致直接报错。
TRAINING_WHEEL_DAMPING = 0.032
#: 训练侧质量随机范围（`reset_work_condition.params["mass_range"]`）。
#: 2026-10-09 由 (5, 15) 提到 (5, 30)；**2026-10-10 收紧为 (5, 20)**（用户确认工作域＝
#: 质量 5–20 kg、速度 0.5–1.5 m/s）⇒ 默认质量档里的 25 kg **已超出训练分布**（外推检查），
#: 20 kg 仍在域内。`--dry-run` 的 plan 行与 `experiment.json.caveats` 会显式标出域外档位。
TRAINING_MASS_RANGE_KG = (5.0, 20.0)

# ---- 上层 checkpoint 契约 / 偏移头的离线镜像（v3 双头，TOW-21/TOW-26）----
#: `upper_policy_runtime` 顶部 import torch，而 `--dry-run`／`--help` 必须保持**纯标准库**可跑，
#: 所以期望契约与偏移边界在这里重述一份；离线测试把每个字段与 `upper_policy_runtime` 的常量、
#: `expected_towing_contract(UpperNetworkSpec())` 逐项交叉核对（两边漂移就报错），实跑时
#: `experiment.json` 的 `towing_contract_expected` 还会被运行时的 spec 覆写。
UPPER_CONTRACT_VERSION = 3
UPPER_FRAME_DIM = 58               # = 3 + 13 + 3 + 3 + 12 + 12 + 12
UPPER_EXPLICIT_DIM = 6
UPPER_LATENT_DIM = 16
UPPER_CMD_ACTION_DIM = 1           # 动作第 1 维 = vx 偏移头（`upper_mdp.CMD_ACTION_DIM`）
UPPER_JOINT_ACTION_DIM = 12        # 动作后 12 维 = 关节位置残差头
UPPER_ACTION_DIM = UPPER_CMD_ACTION_DIM + UPPER_JOINT_ACTION_DIM      # 13
#: 期望契约（写进 plan 行与 `experiment.json.upper_policy.towing_contract_expected`）。
UPPER_TOWING_CONTRACT_EXPECTED = {
    "version": UPPER_CONTRACT_VERSION, "frame_dim": UPPER_FRAME_DIM,
    "explicit_dim": UPPER_EXPLICIT_DIM, "latent_dim": UPPER_LATENT_DIM}
#: 速度头偏移的尺度/限幅与合成包络（`upper_mdp.HierarchicalVelocityActionCfg` 的镜像，
#: 与 `upper_policy_runtime.COMMAND_OFFSET_*` / `AMP_VX_*` 同值，由离线测试交叉核对）：
#: 第 1 层限的是**偏移量本身**（不是「和」），第 2 层把「和」裁进冻结 AMP 策略的训练包络。
UPPER_OFFSET_SCALE_MPS = 0.5
UPPER_OFFSET_RANGE_MPS = (-0.2, 0.6)
UPPER_AMP_VX_RANGE_MPS = (-1.0, 1.5)
#: 上层控制周期（s）：训练侧 `HierarchicalVelocityActionCfg.upper_control_dt = 0.05`（20 Hz）。
#: plan 行按它和 `--dt` 算出「每几个物理步推理一拍」；离线测试按 AST 交叉核对。
UPPER_CONTROL_DT_S = 0.05

DEFAULT_VELOCITIES = (0.5, 1.0, 1.5)
DEFAULT_CART_MASSES = (5.0, 10.0, 15.0, 20.0, 25.0)
#: 800 环境 × 2200 步逐物理步记录 ≈ 176 万行（约 2 GB），默认改成每 5 步（25 ms，等于
#: 冻结策略周期）记一行；接触的速度跃变阈值在 `contact_witness` 里按记录步长自动放大。
DEFAULT_RECORD_EVERY = 5
N_JOINTS = 12
PHASES = ("station", "tow", "coast")

#: 「策略 vs 基线」报告里统一用的五项指标中位数（与 docstring 的五项指标一一对应）。
FIVE_METRIC_FIELDS = (
    ("startup", "joint_rms_rad", "起步关节响应 RMS"),
    ("speed", "mae_mps", "全程跟速 MAE"),
    ("stop", "cart_coast_distance_m", "停车小车滑移"),
    ("stop", "min_clearance_coast_m", "停车最小几何间隙"),
    ("stop", "joint_rms_rad", "停车关节响应 RMS"),
)

#: 冲击窗口独立统计的默认参数（用户 2026-10-10 要求）。
#: **只进 `metrics["impact"]`，不进任何判据/阈值**：`classify_case` 不读它、`DEFAULT_THRESHOLDS`
#: 不含它（既有 `startup.*` / `stop.*` 字段与判定码语义逐位不变，归档对照不受影响）。
DEFAULT_IMPACT_WINDOW_S = 0.2      # 冲击瞬态窗口宽度 [s]
DEFAULT_STEADY_MARGIN_S = 1.0      # 稳态窗在牵引段首尾各去掉多少 [s]
DEFAULT_TAKEUP_FORCE_THRESHOLD_N = 1.0   # 判定「绷直/穿绳」的绳张力模长阈值 [N]

#: 冲击窗口统计的四个时刻 + 稳态（`metrics["impact"]` 的子键，报告表也按这个顺序）。
IMPACT_GROUPS = ("startup", "takeup", "stop", "stop_contact", "steady")

#: 冲击窗口统计里进「策略 vs 基线」中位数对照的字段（逐 case 中位数 + 样本数）。#: 与 `FIVE_METRIC_FIELDS` 同一套 `case_metric_summary` 汇总，读法：比值 ≈ 1 表示该时刻
#: 的关节响应与稳态同量级（没有冲击）；比值越大冲击越猛（>1 的倍数即「猛多少倍」）。
IMPACT_METRIC_FIELDS = (
    ("impact.startup.joint_rms_rad", "起拖窗口关节 RMS"),
    ("impact.takeup.joint_rms_rad", "绷直窗口关节 RMS"),
    ("impact.stop.joint_rms_rad", "归零窗口关节 RMS"),
    ("impact.stop_contact.joint_rms_rad", "停车撞击窗口关节 RMS"),
    ("impact.steady.joint_rms_rad", "稳态（牵引段去首尾）关节 RMS"),
    ("impact.startup.rms_over_steady", "起拖 RMS / 稳态"),
    ("impact.takeup.rms_over_steady", "绷直 RMS / 稳态"),
    ("impact.stop.rms_over_steady", "归零 RMS / 稳态"),
    ("impact.stop_contact.rms_over_steady", "停车撞击 RMS / 稳态"),
    ("impact.startup.rms_delta_rad", "起拖 RMS − 稳态 [rad]"),
    ("impact.stop.rms_delta_rad", "归零 RMS − 稳态 [rad]"),
    ("impact.startup.torque_saturated_frac", "起拖力矩饱和占比"),
    ("impact.takeup.torque_saturated_frac", "绷直力矩饱和占比"),
    ("impact.stop.torque_saturated_frac", "归零力矩饱和占比"),
    ("impact.stop_contact.torque_saturated_frac", "停车撞击力矩饱和占比"),
    ("impact.steady.torque_saturated_frac", "稳态力矩饱和占比"),
)

#: 归档基线（`docs/towingdata/2026-10-09_necessity_800_noload/`）的读数，作为
#: report.md「策略 vs 基线」一节里的**参照列**（不是对照列）。
#:
#: ⚠ **不能逐格硬比**：该 run 的 `git.commit = 1bbb405` 且**工作树是脏的**（上层侧 6 个
#: 文件当时未提交，随后成为 `6fac89a`；复现请用 `6fac89a`），而且用的是**旧几何**——
#: 绳 0.6–1.2 m（长度未按类型解耦）、出生比 0.5；本次代码是绳 0.5–1.5 / 杆 0.5–1.0、
#: 出生比 0.8。所以这一列只用于「同量级 sanity check」，有效对照必须在**同一版本里**
#: 跑两轮（基线一次 + 开开关一次，参数完全相同）。
#: 数值由 `case_metric_summary()` 从归档 report.json 复算（离线测试逐项核对）。
ARCHIVED_BASELINE_2026_10_09 = {
    "label": "归档基线 2026-10-09（旧几何，仅参照）",
    "report": "docs/towingdata/2026-10-09_necessity_800_noload/report.json",
    "git_commit": "1bbb4054e52c63be3bfaf5462f08a891a6ddae85",
    "reproduce_commit": "6fac89a",
    "working_tree_dirty": True,
    "grid_note": "旧几何：绳 0.6–1.2 m（未按类型解耦）、出生比 0.5；当前代码是绳 0.5–1.5 / "
                 "杆 0.5–1.0、出生比 0.8 ⇒ 不可逐格硬比，只能同版本内对照",
    "num_envs": 800,
    "no_cart_fraction": 0.125,
    "ok": 0,
    "codes": {"JNT": 395, "COL": 276, "SPD": 129},
    "medians": {
        "startup.joint_rms_rad": 0.215685,
        "speed.mae_mps": 0.160643,
        "stop.cart_coast_distance_m": 0.090157,
        "stop.min_clearance_coast_m": 0.489744,
        "stop.joint_rms_rad": 0.154061,
    },
    "median_samples": {
        "startup.joint_rms_rad": 800,
        "speed.mae_mps": 800,
        "stop.cart_coast_distance_m": 800,
        "stop.min_clearance_coast_m": 691,
        "stop.joint_rms_rad": 800,
    },
    "by_grade": {"grade0": (400, 0), "grade5": (200, 0), "grade10": (200, 0)},
    "jnt_note": "归档基线里 JNT 在有负载（295/700）与无负载（100/100）两边都接近 100%，"
                "量的是 q − q* 的 PD 静差（≈ τ/kp），阈值未标定",
    # 归档的通过数/判定码按**旧口径**（JNT 计入判定）算 ⇒ 与 2026-10-10 之后的新口径不可比。
    "count_jnt": True,
    "caliber_note": "归档的通过数（0/800）与判定码（JNT 395 / COL 276 / SPD 129）是按**旧口径**"
                    "（JNT 计入判定）算的；2026-10-10 起默认把 JNT 移出判定统计量 ⇒ 两边的"
                    "通过数与判定码分布**不可比**，要逐格对齐旧口径请用 `--count-jnt` 重跑",
}

#: 用户 2026-10-10 决定「JNT 移出判定统计量」后的**离线重判读数**（本变更的依据与验收基线）。
#:
#: 来源：同版本、同几何的两轮 800 cell ——
#: `imgo2_rl/logs/towing/play_test/upper_switch_baseline/report.json`（基线）与
#: `.../upper_switch_policy/report.json`（策略轮开 `--upper-checkpoint .../model_1000.pt`，
#: `upper_policy.enabled=true / iter=1000 / contract v2`）；两轮
#: `git.commit = df593f49d6469be0ec1bbd2637fa66bb5616cb3c`、800 cell、`--no-cart-fraction 0.125`。
#:
#: 复算方式 = 把 `cases[].verdict.reasons` 里的 `startup_joint_error`/`stop_joint_error` 剔除后
#: 重算通过数与分布（`imgo2_rl/tests/test_towing_play_test.py` 的
#: `JntExclusionRecountTests` 从上面两份 report.json 逐项核对，防手抄漂移）。
JNT_EXCLUDED_RECOUNT_2026_10_10 = {
    "source_report": "imgo2_rl/logs/towing/play_test/upper_switch_{baseline,policy}/report.json",
    "git_commit": "df593f49d6469be0ec1bbd2637fa66bb5616cb3c",
    "num_envs": 800,
    "no_cart_fraction": 0.125,
    #: 旧口径（JNT 计入）下两轮都是 0/800 —— 与重判值的差就是被移出统计量的那部分。
    "old_caliber_ok": {"baseline": 0, "policy": 0},
    "ok": {"baseline": 356, "policy": 436},
    "by_grade": {
        "baseline": {"grade0": (400, 213), "grade5": (200, 81), "grade10": (200, 62)},
        "policy": {"grade0": (400, 245), "grade5": (200, 96), "grade10": (200, 95)},
    },
    "by_connection": {
        "baseline": {"compliant": (320, 152), "inextensible": (160, 69), "rigid": (320, 135)},
        "policy": {"compliant": (320, 174), "inextensible": (160, 88), "rigid": (320, 174)},
    },
    "by_cart": {
        "baseline": {"cart": (700, 256), "nocart": (100, 100)},
        "policy": {"cart": (700, 336), "nocart": (100, 100)},
    },
    "reasons": {
        "baseline": {"stop_collision": 295, "speed_track_error": 234, "stop_margin_low": 28},
        "policy": {"stop_collision": 283, "speed_track_error": 66, "stop_margin_low": 36,
                   "lane_deviation": 1},
    },
    "note": "剔除 JNT 后剩下的失败全部与小车/停车/跟速有关（stop_collision / speed_track_error / "
            "stop_margin_low / lane_deviation）⇒ 这正是上层任务该修的对象；JNT 的 800/800 "
            "（≈ τ/kp 静差、阈值未标定）没有区分度。",
    #: 待用户决定的**下一条过紧判据**（本次不实现，仅记录；见 README TOW-23 的「待定」）。
    "open_question": {
        "topic": "stop_collision 的 `load_velocity_jump` 通道是否也是过紧的占位阈值",
        "evidence": "基线 `impact.stop_contact.channels`（首个接触见证）构成："
                    "load_velocity_jump 254、deck_contact_force 70；`deck |fx|` 中位 0 N；"
                    "策略轮同一口径为 load_velocity_jump 249 / deck_contact_force 62。"
                    "（按 `stop.contact_channels` 的并集统计则是基线 264/71、策略 254/77。）",
        "hypothesis": "用绳拖 5–25 kg 车斗在 20 Hz 急停时，速度在某一拍变化是物理必然 ⇒ "
                      "「负载是否发生速度跃变」可能是过紧的占位阈值。",
        "suggestion": "像冲击统计那样改成**相对量**（负载减速度 / 与无负载对照的增量），"
                      "列入待用户决定，本轮不改判据。",
    },
}

_JOINT_NAMES_FALLBACK = tuple(
    f"{leg}_{part}_joint" for leg in ("FL", "FR", "RL", "RR")
    for part in ("hip", "thigh", "shank"))

# 记录列：`summarize_tow` 需要的列（TOW_FIELDS）+ 本测试新增的列。
# `recording.py` / `mdp/connection_grid.py` / `mdp/slope_geometry.py` 都是纯标准库模块，
# 按文件路径加载，避免 import 重量级包 `__init__`（那会让 `--dry-run` 失去标准库可运行性）。
def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块 {name}：{path}")
    module = importlib.util.module_from_spec(spec)
    # 必须先注册进 sys.modules：`slope_geometry` 用 `from connection_grid import ...`
    # 的顶层回退（按路径加载时没有包上下文），且 dataclass 处理字符串注解也要用到它。
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MDP_DIR = RL_ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/mdp"
recording = _load_module(
    "imgo2_play_towing_test_recording",
    RL_ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/utils/recording.py")
# 训练网格（逐 env 的连接/长度/坡度量级）与坡面剖面几何：本测试台直接复用训练侧的定义，
# 不自己写第二套网格表，保证「同一 env 编号 ⇒ 同一 cell」与训练逐位一致。
connection_grid = _load_module("connection_grid", MDP_DIR / "connection_grid.py")
slope_geometry = _load_module("slope_geometry", MDP_DIR / "slope_geometry.py")
# 回合相位几何（纯标准库）：只需它的 `SETTLE_TIME_S` —— 训练侧
# `HierarchicalVelocityActionCfg.tow_start_s` 的默认值就是它，用来给偏移门控一个
# **离线可打印**的期望阈值（实跑时仍以 `action_cfg.tow_start_s` 为准）。
episode_geometry = _load_module("episode_geometry", MDP_DIR / "episode_geometry.py")

#: 偏移门控的时刻（s）：`elapsed_s >= UPPER_GATE_AFTER_S` 之后偏移才叠加到任务指令上。
#: 取自训练侧 `mdp/episode_geometry.SETTLE_TIME_S`（= `HierarchicalVelocityActionCfg` 的默认
#: `tow_start_s`，离线测试按 AST 交叉核对）；`--dry-run`／`experiment.json` 里打印的期望值。
UPPER_GATE_AFTER_S = float(episode_geometry.SETTLE_TIME_S)

#: 默认并行环境数 = 训练网格全集（40 列 × 20 行 = 800）。写成 800 的整数倍时每个 cell
#: 会重复出现，同一 cell 的不同 env 拿到不同工作条件（见 `work_conditions`）。
DEFAULT_NUM_ENVS = int(connection_grid.GRID_SIZE)
NUM_COLUMNS = int(connection_grid.COLUMNS)
NUM_ROWS = int(connection_grid.ROWS)

JOINT_TARGET_FIELDS = tuple(f"robot_jt_{index:02d}" for index in range(N_JOINTS))
JOINT_TORQUE_FIELDS = tuple(f"robot_tau_{index:02d}" for index in range(N_JOINTS))
WHEEL_LEGS = ("fl", "fr", "rl", "rr")
#: 本测试新增列。`connection` 是字符串列，其余都是数值列。
#: `*_progress_m` / `*_surface_height_m` / `body_pitch_rel_rad` 是**lane 坐标系**量：
#: 出生在剖面平地段 ⇒ lane 系 = 世界系（切向 +x、法向 +z、重力竖直），逐 env 的坡度量级
#: 只参与「离面高度」与「相对俯仰」的换算（`profile_height` / `profile_slope_degrees`）。
EXTRA_FIELDS = (
    "velocity_cmd_mps",     # 本 env 当前的速度指令（体系 x）
    "robot_vx_b_mps",       # 体系 x 速度 = 跟速误差用的口径
    "joint_err_rms_rad",    # 本拍 12 关节跟踪误差的 RMS（在线算，便于画曲线）
    "joint_err_max_rad",
    # 沿坡面切向的行程（起点 = 本 env 的坡面原点）：停车滑移、机器人位移都用它
    "robot_progress_m", "load_progress_m",
    # 离坡面的法向高度：跌倒判据用它（gravity 后端退化为绝对 z）
    "robot_surface_height_m", "load_surface_height_m",
    # 相对坡面参考姿态的俯仰（terrain 后端扣掉出生时的 −θ，gravity 后端就是世界系俯仰）
    "body_pitch_rel_rad",
    # 横向/朝向保持：送冻结策略的 (vy, wz) 指令、lane 系横向偏移与朝向误差、机体系实测
    "velocity_cmd_vy_mps", "velocity_cmd_wz_radps",
    "lane_offset_m", "lane_heading_rad", "robot_vy_b_mps", "robot_wz_b_radps",
    "load_offset_m",
    # 本 env 的 lane 坡度量级（0 / 5 / 10 deg，来自训练网格）+ 连接长度（训练网格逐行）
    "grade_deg", "connection_length_m",
    "gravity_x_mps2", "gravity_z_mps2",
    "connection", "cart_mass_kg",
)
EXTRA_NUMERIC_FIELDS = tuple(
    name for name in EXTRA_FIELDS if name not in ("connection",))
#: 记录里的非数值列（`make_row` 写字符串，数值完整性检查要跳过它们）。
STRING_FIELDS = ("phase", "connection")
TEST_FIELDS = ("phase", *recording.TOW_NUMERIC_FIELDS,
               *JOINT_TARGET_FIELDS, *JOINT_TORQUE_FIELDS, *EXTRA_FIELDS)

#: 逐 case 判定码。数值越大越严重（`classify_case` 取最严重的一条作为 `code`）。
VERDICT_CODES = ("OK", "LOW", "LAT", "JNT", "SPD", "COL", "FALL", "INV")
VERDICT_SEVERITY = {code: index for index, code in enumerate(VERDICT_CODES)}
_REASON_TO_CODE = {
    "invalid_record": "INV",
    "robot_fell": "FALL",
    "stop_collision": "COL",
    "stop_margin_low": "LOW",
    "lane_deviation": "LAT",
    "speed_track_error": "SPD",
    "startup_joint_error": "JNT",
    "stop_joint_error": "JNT",
}

#: `summarize_tow` 的 `valid/failures` 是**绳语义**判据（要求 tow 段出现过 `T > 0`）；
#: rigid 球铰连杆的张力有符号（压缩为负），因此 rigid case 可能报
#: `rope_never_taut_during_tow` 之类的失败 —— 那不代表工况失败。
#: 本脚本自己的判定（`verdict`）不依赖 `summarize_tow` 的 `valid`，只借用它的几何间隙。
SUMMARIZE_TOW_NOTE = (
    "summarize_tow.valid 是绳语义判据（要求 tow 段出现过 T>0）；rigid 连杆张力有符号，"
    "其失败项（如 rope_never_taut_during_tow）不代表工况失败。本脚本的 verdict 不依赖它。")

#: 横向/朝向保持（PD）与对应判据阈值：默认值取训练分布内的保守量
#: （AMP 的训练指令范围是 lin_vel_y ±1.0 m/s、ang_vel_z ±1.57 rad/s，见 velocity_env_cfg.py）。
DEFAULT_LANE_OPTIONS = {
    "lane_kp_y": 1.0,             # 横向位置 P 增益 [1/s]
    "lane_kd_y": 0.3,             # 横向速度 D 增益 [-]
    "lane_kp_yaw": 1.5,           # 朝向 P 增益 [1/s]
    "lane_kd_yaw": 0.3,           # 偏航角速度 D 增益 [-]
    "lane_vy_limit": 0.4,         # 送冻结策略的 vy 指令限幅 [m/s]（训练范围 ±1.0 之内）
    "lane_wz_limit": 0.8,         # 送冻结策略的 wz 指令限幅 [rad/s]（训练范围 ±1.57 之内）
}
DEFAULT_THRESHOLDS = {
    # 关节跟踪误差（rad）：起步与停车两段的 RMS 上限、单关节绝对值上限。
    # 初值是工程占位（AMP 稳态跟踪误差量级 0.05 rad，加载后到 0.2 rad 量级都会超限），
    # 没有标定过；跑完首轮后应按实测分布重新设定并写进文档。
    "joint_rms_limit_rad": 0.10,
    "joint_max_limit_rad": 0.30,
    # 跟速：|vx_b − cmd| 的全程 MAE 相对指令的比例上限。
    "speed_mae_ratio_limit": 0.20,
    # 停车余量：车头几何间隙低于此值算「余量不足」（未接触）。
    "gap_margin_limit_m": 0.10,
    # 跌倒：base 高度低于此值，或 |pitch| 超限的样本占比超过 pitch_fraction_limit。
    "fall_height_limit_m": 0.15,
    "pitch_limit_rad": 0.80,
    "pitch_fraction_limit": 0.20,
    # 横向/朝向保持（`--lane-keeping pd` 时生效）：超过阈值 ⇒ 判定码 `LAT`。阈值未标定。
    "lane_y_limit_m": 0.30,
    "lane_heading_limit_deg": 10.0,
}

#: 判定口径标记：**JNT（`startup_joint_error` / `stop_joint_error`）是否计入判定**。
#:
#: 用户 2026-10-10 决定：**默认不计入**（`False`）。依据：两条在 800 cell 上基线 800/800、
#: 策略 800/800，零区分度；它们量的是底层 PD 静差 `τ/kp`（`joint_rms_limit_rad=0.10` 连
#: 最好的 case 都超 1.6 倍：实测基线 startup RMS P1=0.161 / P50=0.221、stop RMS P1=0.126 /
#: P50=0.154），不是上层任务的职责。字段与阈值**保留**（观测），`--count-jnt` 打开后逐位
#: 复现旧口径（供与归档对齐/追溯）。详见 README TOW-23 与
#: `docs/towing_verdict_jnt_excluded_2026-10-10.md`。
DEFAULT_COUNT_JNT = False


def caliber_thresholds(args) -> dict:
    """阈值字典 + 口径标记（`report.json` / `experiment.json` 的 `thresholds` 用）。

    `count_jnt` 只是**口径标注**（不参与 `classify_case` 的阈值比较），但它必须跟阈值一起
    落盘：读归档的 `report.json` 时，先看这一位才知道某一轮的通过数/判定码是按哪套口径算的。
    """
    return {**{name: getattr(args, name) for name in DEFAULT_THRESHOLDS},
            "count_jnt": bool(args.count_jnt)}


# ---------------------------------------------------------------- 纯逻辑（离线可测）


@dataclass(frozen=True)
class EnvCase:
    """一个环境的固定设定：cell（来自训练网格） + 工作条件（速度 × 质量轮转）。

    与训练侧的分工完全一致：**连接类型 / 连接长度 / 坡度量级由 cell 决定**（`env_spec`），
    本测试台额外把「速度指令」和「小车质量」也变成逐 env 的量（训练时它们是随机化的），
    这样就得到「同一套网格、每个 env 一个固定 case」的确定性扫描。
    """

    env_index: int
    cell_index: int
    column: int
    row: int
    connection: str
    length_m: float
    grade_deg: float
    velocity_mps: float
    cart_mass_kg: float
    #: 本 env 是否拖车（`--no-cart-fraction`）：False = 小车横向停到 2 m 外、绳力/轮阻全部置 0，
    #: 用来测「不带负载」的参考统计量（判定时也跳过与小车有关的项）。
    cart_present: bool = True

    @property
    def slug(self) -> str:
        """文件名安全的短标识（`+`/`.` 合法，只有 `/` 要避开）。

        必须带 env 序号：工作条件是轮转分配的，同一个 (坡度量级, 速度, 连接, 质量) 组合
        会在多个 cell 上重复出现，只用那四项当文件名会互相覆盖。
        """
        suffix = "" if self.cart_present else "_nocart"
        return (f"env{self.env_index:04d}_c{self.column:02d}r{self.row:02d}"
                f"_g{self.grade_deg:g}_v{self.velocity_mps:g}"
                f"_{self.connection}_m{self.cart_mass_kg:g}kg{suffix}")

    def to_dict(self) -> dict:
        return {"env_index": self.env_index, "cell_index": self.cell_index,
                "column": self.column, "row": self.row, "connection": self.connection,
                "connection_length_m": self.length_m, "grade_deg": self.grade_deg,
                "velocity_mps": self.velocity_mps, "cart_mass_kg": self.cart_mass_kg,
                "cart_present": bool(self.cart_present)}


def work_conditions(num_envs: int, velocities, cart_masses,
                    columns: int = NUM_COLUMNS) -> list:
    """逐 env 的 `(速度 m/s, 质量 kg)` 确定性轮转分配。

    规则（`slot = row + column`，其中 `row = index // columns`、`column = index % columns`）：

    - 质量：`cart_masses[slot % len(cart_masses)]`
    - 速度：`velocities[(slot // len(cart_masses)) % len(velocities)]`

    **为什么用 `row + column` 而不是 `index`**：`COLUMNS = 40` 是质量档数的整数倍
    （40 % 5 == 0），若直接用 `index % len(masses)`，质量就只由 `column % 5` 决定 —— 而列同时
    决定连接类型与坡度量级，于是 `(连接, 质量)` 与 `(坡度量级, 质量)` 会严重混淆（实测：
    inextensible 完全没有 5 kg、5°/10° 的 inextensible 只落到重载，判定矩阵出现空格）。
    改成 `row + column` 后，固定一列时 `slot` 随 20 行取到 20 个连续整数 ⇒ **每个质量档在
    一列内出现 4 次、每个 (质量, 速度) 组合出现 1–2 次**，「连接 × 质量」「坡度量级 × 质量」
    都均衡了。

    性质（离线测试钉住）：**确定性**（与运行顺序、随机种子、`--num-envs` 无关；`--dry-run`
    打印的就是实跑用的表）；800 env / 15 组合时每组 50–55 个 env（`slot` 的取值个数在
    20×40 网格里两端少、中间多，所以不是严格的 ±1）。
    """
    if num_envs <= 0:
        raise ValueError(f"num_envs 必须是正整数，收到 {num_envs!r}")
    if columns <= 0:
        raise ValueError(f"columns 必须是正整数，收到 {columns!r}")
    velocities = [float(value) for value in velocities]
    cart_masses = [float(value) for value in cart_masses]
    if not velocities or not cart_masses:
        raise ValueError("速度档与质量档都不能为空")
    if any(not math.isfinite(v) or v <= 0.0 for v in velocities):
        raise ValueError(f"速度档必须是有限正数，收到 {velocities!r}")
    if any(not math.isfinite(m) or m <= 0.0 for m in cart_masses):
        raise ValueError(f"质量档必须是有限正数，收到 {cart_masses!r}")
    conditions = []
    for index in range(num_envs):
        slot = (index // columns) + (index % columns)
        conditions.append((velocities[(slot // len(cart_masses)) % len(velocities)],
                           cart_masses[slot % len(cart_masses)]))
    return conditions


#: 无小车 env 的小车横向停放距离（与训练侧 `reset_towing_episode` 的
#: `no_cart_lateral_offset` 同值；训练侧 12.5% 的锚点 env 就是这么处理的）。
NO_CART_LATERAL_OFFSET_M = 2.0


def cart_present_for(index: int, no_cart_fraction: float) -> bool:
    """`--no-cart-fraction` → 第 `index` 个 env 是否拖车（**确定性**，不引入随机种子）。

    `fraction = 0` ⇒ 全部拖车（默认，行为与以前完全一致）；`fraction = 1` ⇒ 全部不拖车
    （测「无负载」参考就是这个）；中间值按 `index % period == 0` 的等距抽样，
    `period = round(1/fraction)`（如 0.125 ⇒ 每 8 个 env 抽 1 个）。
    """
    if not 0.0 <= no_cart_fraction <= 1.0:
        raise ValueError(f"no_cart_fraction 必须在 [0, 1]，收到 {no_cart_fraction!r}")
    if no_cart_fraction <= 0.0:
        return True
    if no_cart_fraction >= 1.0:
        return False
    period = max(2, int(round(1.0 / no_cart_fraction)))
    return index % period != 0


def build_env_cases(num_envs: int, velocities, cart_masses, *, no_cart_fraction: float = 0.0) -> list:
    """逐 env 的 `EnvCase` 列表：cell 参数取训练网格，工作条件取 `work_conditions`。

    env → cell 的映射用训练侧同一函数（`connection_grid.env_spec`，row-major：列 = i % 40、
    行 = i // 40，`i % GRID_SIZE` 循环），所以本测试台的 env `i` 与训练时的 env `i`
    是**同一条 lane、同一种连接、同一段长度、同一个坡度量级**。
    """
    conditions = work_conditions(num_envs, velocities, cart_masses)
    cases = []
    for index in range(num_envs):
        spec = connection_grid.env_spec(index)
        velocity, mass = conditions[index]
        cases.append(EnvCase(
            env_index=index, cell_index=int(spec["grid_index"]), column=int(spec["column"]),
            row=int(spec["row"]), connection=str(spec["model_name"]),
            length_m=float(spec["length"]), grade_deg=float(spec["slope_degrees"]),
            velocity_mps=velocity, cart_mass_kg=mass,
            cart_present=cart_present_for(index, no_cart_fraction)))
    return cases


def env_case_summary(cases) -> dict:
    """`--dry-run`／`experiment.json` 用的分配摘要（纯逻辑，标准库可跑）。"""
    if not cases:
        raise ValueError("cases 不能为空")
    connections, grades, lengths, buckets = {}, {}, [], {}
    mass_by_connection = {}
    for case in cases:
        connections[case.connection] = connections.get(case.connection, 0) + 1
        grades[case.grade_deg] = grades.get(case.grade_deg, 0) + 1
        lengths.append(case.length_m)
        key = (case.velocity_mps, case.cart_mass_kg)
        buckets[key] = buckets.get(key, 0) + 1
        per_connection = mass_by_connection.setdefault(case.connection, {})
        per_connection[case.cart_mass_kg] = per_connection.get(case.cart_mass_kg, 0) + 1
    return {
        "envs": len(cases),
        "cells": len({case.cell_index for case in cases}),
        "cell_repeats": len(cases) / len({case.cell_index for case in cases}),
        "connections": dict(sorted(connections.items())),
        "grades_deg": {f"{grade:g}": count for grade, count in sorted(grades.items())},
        "length_m": {"min": min(lengths), "max": max(lengths)},
        "work_condition_buckets": {f"v{velocity:g}_m{mass:g}": count
                                   for (velocity, mass), count in sorted(buckets.items())},
        "bucket_env_counts": sorted(buckets.values()),
        # 「连接 × 质量」的 env 数：暴露「质量只由列决定」带来的不平衡
        # （COLUMNS=40 是质量档数的整数倍 ⇒ 列同时决定连接类型与坡度量级）
        "mass_counts_by_connection": {
            connection: {f"{mass:g}": count for mass, count in sorted(counts.items())}
            for connection, counts in sorted(mass_by_connection.items())},
    }


@dataclass(frozen=True)
class PhaseSchedule:
    """station / tow / coast 三段的步数划分（纯算术）。"""

    station_steps: int
    tow_steps: int
    coast_steps: int
    dt: float

    @property
    def total_steps(self) -> int:
        return self.station_steps + self.tow_steps + self.coast_steps

    def phase_of(self, step: int) -> str:
        if step < self.station_steps:
            return "station"
        return "tow" if step - self.station_steps < self.tow_steps else "coast"

    def step_in_phase(self, step: int) -> int:
        if step < self.station_steps:
            return step
        if step - self.station_steps < self.tow_steps:
            return step - self.station_steps
        return step - self.station_steps - self.tow_steps

    def to_dict(self) -> dict:
        return {"station_steps": self.station_steps, "tow_steps": self.tow_steps,
                "coast_steps": self.coast_steps, "dt_s": self.dt,
                "station_s": self.station_steps * self.dt,
                "tow_s": self.tow_steps * self.dt,
                "coast_s": self.coast_steps * self.dt,
                "total_s": self.total_steps * self.dt}


def wrap_to_pi(angle_rad: float) -> float:
    """把角度归一化到 (−π, π]（PD 的朝向误差必须跨 ±π 不跳变）。"""
    if not math.isfinite(angle_rad):
        raise ValueError(f"角度必须是有限值，收到 {angle_rad!r}")
    wrapped = math.fmod(angle_rad + math.pi, 2.0 * math.pi)
    if wrapped <= 0.0:
        wrapped += 2.0 * math.pi
    return wrapped - math.pi


def lane_keeping_command(*, kp_y: float, kd_y: float, kp_yaw: float, kd_yaw: float,
                         vy_limit: float, wz_limit: float, lane_y: float, lane_yaw: float,
                         body_vy: float, body_wz: float, yaw_target: float = 0.0) -> tuple:
    """横向/朝向 PD → 送给冻结策略的 `(vy, wz)` 指令；**前进速度不参与**（只给指令）。

    目标：机器人沿 lane 中线走（lane 系 `y = 0`），朝向超前（`yaw = yaw_target`，默认 0 = +x）。

    - 横向：误差向量 `(0, −y, 0)`（lane 系）投到**机体系横向轴**上 ⇒ `−y·cos(yaw)`（小角度下
      就是 `−y`）。P 项把机器人推回中线，D 项用**机体系**横向速度阻尼——冻结策略的
      `velocity_commands` 本来就是机体系，所以两项都要在同一坐标系里。
    - 朝向：`yaw_target − yaw` 归一化后用 P 项，D 项用机体系 yaw 角速度。
    - 两路输出都限幅在训练分布内（AMP 训练范围：`lin_vel_y ±1.0 m/s`、`ang_vel_z ±1.57 rad/s`）。

    纯算术、可离线测；张量版在主循环里按同一公式逐项实现。
    """
    for name, value in (("lane_y", lane_y), ("lane_yaw", lane_yaw), ("body_vy", body_vy),
                        ("body_wz", body_wz), ("yaw_target", yaw_target)):
        if not math.isfinite(value):
            raise ValueError(f"{name} 必须是有限值，收到 {value!r}")
    for name, value in (("vy_limit", vy_limit), ("wz_limit", wz_limit)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} 必须是有限正数，收到 {value!r}")
    lateral_error = -lane_y * math.cos(lane_yaw)
    heading_error = wrap_to_pi(yaw_target - lane_yaw)
    vy_command = kp_y * lateral_error - kd_y * body_vy
    wz_command = kp_yaw * heading_error - kd_yaw * body_wz
    return (max(-vy_limit, min(vy_limit, vy_command)),
            max(-wz_limit, min(wz_limit, wz_command)))


def scale_cart_mass_inertia(nominal_masses, nominal_inertias, mass_scales):
    """按**逐 env**的比例同时缩放小车的质量与惯量（AGENTS.md：只改质量不改惯量会不自洽）。

    形状契约（PhysX tensor API，实测）：
    - `get_masses()` → `(N, num_bodies)`（小车 `merge_fixed_joints=True`，无质量的挂点被并进
      base_link ⇒ **5** 个刚体：base_link + 四轮）；
    - `get_inertias()` → `(N, num_bodies, 9)`；
    - `mass_scales` → `(N,)`。

    因此缩放系数要分别扩成 `(N, 1)` 与 `(N, 1, 1)`。**踩过的坑**：把已经是 `(N, nb)` 的
    质量再 `.unsqueeze(0)` 会广播成 `(1, N, nb) × (N, 1)`，在 dim 2 上撞 `nb`(5) 与 `N`(45)，
    报 `The size of tensor a (5) must match the size of tensor b (45) at non-singleton
    dimension 2`（2026-10-09 默认 45 环境首跑实测）；N=1 时形状侥幸不报错，所以单环境冒烟
    抓不到它。
    """
    if nominal_masses.dim() != 2 or nominal_inertias.dim() != 3:
        raise ValueError(
            f"质量应为 (N, num_bodies)、惯量应为 (N, num_bodies, 9)，收到 "
            f"{tuple(nominal_masses.shape)} / {tuple(nominal_inertias.shape)}")
    if nominal_masses.shape[0] != nominal_inertias.shape[0] or \
            nominal_masses.shape[1] != nominal_inertias.shape[1]:
        raise ValueError("质量与惯量的前两维必须一致")
    scales = mass_scales.reshape(-1, 1)
    if scales.shape[0] != nominal_masses.shape[0]:
        raise ValueError(
            f"逐 env 缩放系数应为 {nominal_masses.shape[0]} 个，收到 {scales.shape[0]}")
    return nominal_masses * scales, nominal_inertias * scales.unsqueeze(-1)


def attachment_along(initial_distance: float, normal_difference: float) -> float:
    """两挂点三维距 = `initial_distance` 时，沿 lane 切向需要分开的距离。

    出生姿态竖直 ⇒ 挂点高差 `normal_difference` 与切向距离正交，于是
    `along = sqrt(d² − Δn²)`（勾股）。`d ≤ |Δn|` 时几何无解（挂不住），必须报错而不是
    开方成 NaN —— 那种失败会在 PhysX 里表现成「生成第一拍就有约束力」，很难倒查。
    """
    for name, value in (("initial_distance", initial_distance),
                        ("normal_difference", normal_difference)):
        if not math.isfinite(value):
            raise ValueError(f"{name} 必须是有限值，收到 {value!r}")
    if initial_distance <= 0.0:
        raise ValueError(f"initial_distance 必须是正数，收到 {initial_distance!r}")
    if initial_distance ** 2 <= normal_difference ** 2:
        raise ValueError(
            f"目标挂点距 {initial_distance:.3f} m ≤ 两挂点法向高差 {abs(normal_difference):.3f} m，"
            f"几何上无解（调大连接长度或减小挂点高差）")
    return math.sqrt(initial_distance ** 2 - normal_difference ** 2)


def spawn_offsets(initial_distances, *, robot_height: float, cart_height: float,
                  robot_offset, cart_offset) -> dict:
    """逐 env 的出生偏移（相对该 env 的 lane 原点）：机器人根与小车根。

    与训练侧 `upper_mdp.reset_towing_episode` 同一套几何，但**出生在剖面平地段** ⇒
    姿态竖直（单位四元数，lane 系 = 世界系），于是只有切向（+x）与法向（+z）两个分量：

    - 机器人根 = `(0, 0, robot_height)`（`robot_height` 取资产 default root 的 z）；
    - 小车根让两挂点三维距等于该 env 的 `initial_distance`（绳 = 0.8·L0、刚体 = L）：
      沿 +x 分开 `along`、沿 z 相差 `Δn = (robot 挂点高 − cart 挂点高)`；
    - 逐 env 的 `initial_distance` 不同（长度逐行不同、绳/刚体也不同），所以 `along` 是数组。

    返回的量都是**相对 lane 原点**的纯算术结果；内部对每个 env 重算挂点三维距做自检
    （目标是 spawn 第一拍不产生约束力）。
    """
    for name, value in (("robot_height", robot_height), ("cart_height", cart_height)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} 必须是有限正数，收到 {value!r}")
    robot_offset = tuple(float(value) for value in robot_offset)
    cart_offset = tuple(float(value) for value in cart_offset)
    if len(robot_offset) != 3 or len(cart_offset) != 3:
        raise ValueError("挂点偏移必须是三维向量")
    if any(not math.isfinite(value) for value in (*robot_offset, *cart_offset)):
        raise ValueError("挂点偏移必须是有限值")
    distances = [float(value) for value in initial_distances]
    if not distances:
        raise ValueError("initial_distances 不能为空")
    # Δn = 两挂点在 +z 上的高差（姿态竖直 ⇒ 偏移不经旋转，直接取 z 分量之差）
    normal_difference = (robot_height - cart_height) + (robot_offset[2] - cart_offset[2])
    robot_root = (0.0, 0.0, float(robot_height))
    robot_point = tuple(robot_root[i] + robot_offset[i] for i in range(3))
    alongs, cart_roots, checks = [], [], []
    for distance in distances:
        along = attachment_along(distance, normal_difference)
        cart_root = (robot_point[0] - cart_offset[0] - along,
                     robot_point[1] - cart_offset[1],
                     robot_point[2] - cart_offset[2] - normal_difference)
        cart_point = tuple(cart_root[i] + cart_offset[i] for i in range(3))
        checks.append(math.dist(robot_point, cart_point))
        if abs(checks[-1] - distance) > 1e-9:
            raise RuntimeError(
                f"出生几何自检失败：挂点距 {checks[-1]:.9f} ≠ 目标 {distance:.9f}")
        alongs.append(along)
        cart_roots.append(cart_root)
    return {
        "robot_root": robot_root,
        "cart_root": cart_roots,
        "along_m": alongs,
        "normal_difference_m": normal_difference,
        "attachment_distance_m": checks,
        "target_distance_m": distances,
    }


def make_schedule(*, settle_steps: int, tow_duration: float, coast_duration: float,
                  dt: float) -> PhaseSchedule:
    """由时长算出阶段划分；与测量台 `tow_drag.py` 的 station/tow/coast 语义一致。"""
    if settle_steps < 0:
        raise ValueError("station 步数不能为负")
    if not all(math.isfinite(v) and v > 0 for v in (tow_duration, coast_duration, dt)):
        raise ValueError("tow/coast 时长与 dt 必须是有限正数")
    tow_steps = int(round(tow_duration / dt))
    coast_steps = int(round(coast_duration / dt))
    if tow_steps <= 0 or coast_steps <= 0:
        raise ValueError("tow/coast 阶段的步数必须为正")
    if not math.isclose(tow_steps * dt, tow_duration, rel_tol=1e-6):
        raise ValueError(f"tow 时长 {tow_duration} 不是 dt={dt} 的整数倍")
    if not math.isclose(coast_steps * dt, coast_duration, rel_tol=1e-6):
        raise ValueError(f"coast 时长 {coast_duration} 不是 dt={dt} 的整数倍")
    return PhaseSchedule(station_steps=settle_steps, tow_steps=tow_steps,
                         coast_steps=coast_steps, dt=dt)


def shaped_command(*, phase: str, step_in_phase: int, velocity: float,
                   shaping: str = "direct", ramp_time_s: float = 1.0,
                   dt: float = 0.005) -> float:
    """脚本指令：station/coast 恒 0，tow 段按 shaping 给出速度指令。

    `direct` = 阶跃（本测试默认，与 `tow_drag.py` 的 `user_cmd` 同相位）；
    `ramp` = 在 `ramp_time_s` 内线性升到目标（「Fixed Ramp」对照基线）。
    """
    if phase != "tow":
        return 0.0
    if shaping == "direct":
        return float(velocity)
    if shaping != "ramp":
        raise ValueError(f"未知的指令整形 {shaping!r}；可选 direct / ramp")
    if ramp_time_s <= 0:
        return float(velocity)
    fraction = min(1.0, (step_in_phase + 1) * dt / ramp_time_s)
    return float(velocity) * fraction


def compose_loco_vx_offset(task_vx, offset_vx, *, active: bool,
                           amp_vx_min: float = UPPER_AMP_VX_RANGE_MPS[0],
                           amp_vx_max: float = UPPER_AMP_VX_RANGE_MPS[1]):
    """把**已限过幅的 vx 偏移**合成进任务指令：`clamp(task_vx + active·offset, min, max)`。

    链路与训练侧 `upper_mdp.HierarchicalVelocityAction.process_actions` **同序**（第 1 层限
    偏移量本身由 `upper.command_offset_vx` 完成，本函数做第 2 层包络裁剪 + 门控）：

    1. 门控 `active = elapsed_s >= tow_start_s`：出生段偏移不生效；**STOP 之后门控仍打开**
       （`task_vx = 0` ⇒ `loco_vx = offset`，这是「停机续走」的表达口）；
    2. 把「和」裁进冻结 AMP 策略的训练包络 `amp_vx_range = (−1.0, 1.5)`。

    **纯逻辑、不 import torch**（参数是 1 维张量或 float），所以「关开关时逐位一致」可以离线钉住：
    `offset = 0`（或 `active = False`）时结果与包络内的 `task_vx` **逐位相等**。
    门控放在裁剪之前 ⇒ 关掉时等价于「偏移先置 0 再走同一层包络裁剪」，与训练侧逐位一致
    （脚本速度 0.5–1.5 全在包络内 ⇒ 关掉时就是恒等裁剪）。
    """
    gated = offset_vx if active else offset_vx * 0.0
    return (task_vx + gated).clamp(amp_vx_min, amp_vx_max)


def phase_rows(rows, phase: str) -> list:
    """取某一阶段的记录（`phase` 列）。"""
    return [row for row in rows if row["phase"] == phase]


def _col(rows, name: str) -> list:
    return [float(row[name]) for row in rows]


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def _rms(values) -> float:
    values = list(values)
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else float("nan")


def _percentile(values, fraction: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    index = min(len(values) - 1, max(0, int(round(fraction * (len(values) - 1)))))
    return values[index]


def _nan_summary(reason: str) -> dict:
    return {"available": False, "note": reason}


def joint_track_stats(rows, *, joint_names=None, torque_limits=None,
                      saturation_frac: float = 0.95) -> dict:
    """一段记录里 12 关节的位置跟踪误差 `q − q*` 统计（`q*` = 当拍下发的关节目标）。

    这是「关节响应误差」的口径：上层任务的残差动作正是加在 `q*` 上，
    所以 `q − q*` 就是上层能修正的那部分偏差（包含负载把腿压塌的量）。
    """
    if not rows:
        return _nan_summary("该窗口没有记录")
    names = list(joint_names) if joint_names else list(_JOINT_NAMES_FALLBACK)
    if len(names) != N_JOINTS:
        raise ValueError(f"关节名应为 {N_JOINTS} 个，收到 {len(names)}")
    limits = list(torque_limits) if torque_limits is not None else [23.7] * N_JOINTS
    if len(limits) != N_JOINTS:
        raise ValueError(f"力矩上限应为 {N_JOINTS} 个，收到 {len(limits)}")

    sumsq = [0.0] * N_JOINTS
    peak = [0.0] * N_JOINTS
    saturated = 0
    samples = 0
    for row in rows:
        for joint in range(N_JOINTS):
            error = float(row[f"robot_jp_{joint:02d}"]) - float(row[f"robot_jt_{joint:02d}"])
            sumsq[joint] += error * error
            peak[joint] = max(peak[joint], abs(error))
            if abs(float(row[f"robot_tau_{joint:02d}"])) > saturation_frac * limits[joint]:
                saturated += 1
        samples += 1
    per_joint_rms = [math.sqrt(total / samples) for total in sumsq]
    worst = max(range(N_JOINTS), key=lambda index: per_joint_rms[index])
    return {
        "available": True,
        "samples": samples,
        "joint_rms_rad": math.sqrt(sum(sumsq) / (samples * N_JOINTS)),
        "joint_max_rad": max(peak),
        "worst_joint": names[worst],
        "worst_joint_rms_rad": per_joint_rms[worst],
        "per_joint_rms_rad": {name: value for name, value in zip(names, per_joint_rms)},
        "torque_saturated_frac": saturated / (samples * N_JOINTS),
    }


def speed_track_stats(rows, command_mps: float, *, steady_fraction: float = 0.5) -> dict:
    """跟速误差：体系 x 速度与指令之差（tow 段全程 + 后半段稳态窗）。"""
    if not rows:
        return _nan_summary("tow 段没有记录")
    if not math.isfinite(command_mps) or command_mps <= 0:
        raise ValueError("速度指令必须是有限正数")
    errors = [float(row["robot_vx_b_mps"]) - command_mps for row in rows]
    speeds = _col(rows, "robot_vx_b_mps")
    start = int(len(rows) * (1.0 - steady_fraction))
    steady = errors[start:] or errors
    return {
        "available": True,
        "samples": len(rows),
        "mae_mps": _mean(abs(value) for value in errors),
        "rmse_mps": _rms(errors),
        "bias_mps": _mean(errors),
        "p95_abs_err_mps": _percentile([abs(value) for value in errors], 0.95),
        "max_abs_err_mps": max(abs(value) for value in errors),
        "ratio_mean": _mean(speeds) / command_mps,
        "frac_within_10pct": sum(1 for value in errors if abs(value) <= 0.1 * command_mps) / len(errors),
        "steady_mae_mps": _mean(abs(value) for value in steady),
        "steady_ratio_mean": _mean(_col(rows[start:] or rows, "robot_vx_b_mps")) / command_mps,
    }


def lane_stats(rows, *, vy_limit: float, wz_limit: float, tow_rows=None) -> dict:
    """横向/朝向保持质量：偏移与朝向误差的 RMS/峰值 + 指令限幅占比。

    `rows` 传整个回合或只传牵引段（`tow_rows`）都可以：偏移/朝向按传进来的那段统计，
    限幅占比用 `vy_limit` / `wz_limit` 判「PD 是否已经顶到上限」——顶满说明这条工况的
    侧向扰动超出简单 PD 的权限，是任务必要性的一类证据。
    """
    if not rows:
        return _nan_summary("没有记录")
    if vy_limit <= 0.0 or wz_limit <= 0.0:
        raise ValueError("vy/wz 限幅必须是正数")
    offsets = _col(rows, "lane_offset_m")
    headings = [wrap_to_pi(value) if abs(value) <= math.pi else value
                for value in _col(rows, "lane_heading_rad")]
    vy_commands = _col(rows, "velocity_cmd_vy_mps")
    wz_commands = _col(rows, "velocity_cmd_wz_radps")
    tow_rows = rows if tow_rows is None else tow_rows
    saturated_vy = sum(1 for value in vy_commands if abs(value) >= 0.999 * vy_limit)
    saturated_wz = sum(1 for value in wz_commands if abs(value) >= 0.999 * wz_limit)
    return {
        "available": True,
        "samples": len(rows),
        "tow_samples": len(tow_rows),
        "y_rms_m": _rms(offsets),
        "y_max_abs_m": max(abs(value) for value in offsets),
        "heading_rms_rad": _rms(headings),
        "heading_max_abs_rad": max(abs(value) for value in headings),
        "vy_saturated_frac": saturated_vy / len(vy_commands),
        "wz_saturated_frac": saturated_wz / len(wz_commands),
        "vy_limit_mps": vy_limit,
        "wz_limit_radps": wz_limit,
        # 牵引段的横向/朝向（负载侧向力最大的那段）
        "tow_y_rms_m": _rms(_col(tow_rows, "lane_offset_m")),
        "tow_y_max_abs_m": (max(abs(value) for value in _col(tow_rows, "lane_offset_m"))
                            if tow_rows else None),
    }


def contact_witness(rows, *, record_dt: float, deck_limit_n: float = 1.0,
                    dv_limit_at_5ms: float = 0.015) -> dict:
    """追尾的三路见证（与 `summarize_tow.py` 同口径，但对记录步长做了标定）。

    `summarize_tow` 的 `CONTACT_LOAD_DV_LIMIT_MPS = 0.015` 是按 5 ms 记录定的；
    `--record-every > 1` 时单步速度跃变自然变大，所以这里把阈值按 `record_dt/5ms` 放大，
    避免采样变粗就误报接触。车斗接触力与几何间隙两路与采样无关。
    """
    if not rows:
        return {"available": False, "contact": None, "channels": []}
    channels = []
    peak_deck = max(abs(float(row["cart_deck_fx_n"])) for row in rows)
    if peak_deck > deck_limit_n:
        channels.append("deck_contact_force")
    limit = dv_limit_at_5ms * max(1.0, record_dt / 0.005)
    max_dv = max((abs(float(b["load_vx_mps"]) - float(a["load_vx_mps"]))
                  for a, b in zip(rows, rows[1:])), default=0.0)
    if max_dv > limit:
        channels.append("load_velocity_jump")
    return {"available": True, "contact": bool(channels), "channels": channels,
            "deck_contact_peak_n": peak_deck, "max_load_dv_mps": max_dv,
            "load_dv_limit_mps": limit}


def _rows_in_span(rows, start_s: float, end_s: float) -> list:
    """取**闭区间** `[start_s, end_s]` 内的记录行（按 `time_s` 过滤，与下标无关）。

    记录行的时间戳是 `(step + 1) · dt`（见 `make_row`），所以窗口的两端只按时间对齐：
    「指令归零那一刻」= coast 段第一行，`[t0, t0 + W]` 与 `[t0 − W, t0 + W]` 都从同一行起算。
    """
    return [row for row in rows
            if start_s <= float(row["time_s"]) <= end_s]


def _takeup_index(rows, threshold_n: float, *, phase: str = "tow"):
    """首个 ‖F‖ 越阈的样本下标（`None` = 整段没有绷直/穿绳）。

    绳张力是**有符号**的（rigid 球铰连杆压缩为负），所以判定用模长：`abs(rope_tension_n)`。
    `phase` 给 None 时在整个记录里找（`stop_contact` 用不到它，但保持接口通用）。
    """
    if not (math.isfinite(threshold_n) and threshold_n > 0.0):
        raise ValueError(f"绷直力阈值必须是有限正数，收到 {threshold_n!r}")
    for index, row in enumerate(rows):
        if phase is not None and row["phase"] != phase:
            continue
        if abs(float(row["rope_tension_n"])) > threshold_n:
            return index
    return None


def _contact_index(rows, *, phase: str = "coast", deck_limit_n: float = 1.0,
                   dv_limit_at_5ms: float = 0.015, record_dt: float = 0.005):
    """停车撞击的首个接触见证样本下标（`None` = 没有见证）。

    与 `contact_witness()` 同一套口径、同一套阈值：车斗接触力 `|cart_deck_fx_n| > deck_limit_n`
    或负载单步速度跃变（阈值按 `record_dt / 5 ms` 放大）。**只看 coast 段**——拖曳之前
    「生成/站定就贴上」是另一类问题（与 `stop_stats` 里的 witness 一致）。
    返回 `(index, channels)`：`channels` 是**该样本**触发的通道（可能两个都触发）。
    """
    coast = [row for row in rows if row["phase"] == phase] if phase else list(rows)
    if not coast:
        return None, []
    dv_limit = dv_limit_at_5ms * max(1.0, record_dt / 0.005)
    for index, row in enumerate(coast):
        channels = []
        if abs(float(row["cart_deck_fx_n"])) > deck_limit_n:
            channels.append("deck_contact_force")
        if index > 0 and abs(float(row["load_vx_mps"])
                             - float(coast[index - 1]["load_vx_mps"])) > dv_limit:
            channels.append("load_velocity_jump")
        if channels:
            return index, channels
    return None, []


def impact_stats(rows, *, record_dt: float, impact_window_s: float = 0.2,
                 steady_margin_s: float = 1.0, takeup_force_threshold_n: float = 1.0,
                 deck_limit_n: float = 1.0, joint_names=None, torque_limits=None) -> dict:
    """**冲击窗口 vs 稳态**的独立统计（用户 2026-10-10 要求；纯逻辑、离线可测）。

    为什么要独立统计：把全程/牵引段的关节响应 RMS 一平均，起步与停车的冲击会被运动中的
    小响应**抹平**，看不出安全性改善。这里把四个时刻各自取窗口单独报，并给出与稳态的
    **比值/增量**（不要让读者心算两个绝对量）：

    | 窗口 | 起点 | 对齐到什么 |
    |---|---|---|
    | `startup` | tow 段首行（**起拖**，指令阶跃那一刻） | `impact_startup_time_s` |
    | `takeup` | 首个 `|rope_tension_n| > takeup_force_threshold_n` 的样本 | 真实绷直/穿绳时刻 |
    | `stop` | coast 段首行（**指令归零**） | `impact_stop_time_s` |
    | `stop_contact` | coast 段首个接触见证样本 | 真实停车撞击时刻 |

    窗口宽度：`startup`/`stop` 是**向未来**取 `impact_window_s`（`[t0, t0 + W]`，与既有
    `--transition-window` 同方向、只是更短）；`takeup`/`stop_contact` 是**围绕时刻**
    取 `±impact_window_s`（`[t − W, t + W]`，因为冲量发生在时刻两侧）。

    稳态基线 `steady`：牵引段去掉**首尾各** `steady_margin_s` 后的关节跟踪误差
    （`[tow 首行 + margin, tow 末行 − margin]`），含逐关节 RMS。首尾都去掉是因为首端有
    起步瞬态、末端有绳张力卸载前的抬升。

    关节量口径与 `joint_track_stats` 完全一致：`q − q*`，`q*` = 当拍真正下发的关节位置目标。
    `joint_rms_rad` 是**全 12 关节合并**的 RMS（各窗口独立算，不再被别的窗口平均）。
    每个窗口都带 `available` / `samples` / `window_s`；取不到记录时是 `_nan_summary`，
    不会抛错（800 环境里出现空窗口要能全量落盘）。
    """
    if not (math.isfinite(impact_window_s) and impact_window_s > 0.0):
        raise ValueError(f"冲击窗口必须是有限正数，收到 {impact_window_s!r}")
    if not (math.isfinite(steady_margin_s) and steady_margin_s >= 0.0):
        raise ValueError(f"稳态边距必须是非负有限数，收到 {steady_margin_s!r}")
    names = list(joint_names) if joint_names else list(_JOINT_NAMES_FALLBACK)
    limits = list(torque_limits) if torque_limits is not None else [23.7] * N_JOINTS

    def joint_block(window_rows, *, note: str) -> dict:
        stats = joint_track_stats(window_rows, joint_names=names, torque_limits=limits)
        if not stats.get("available"):
            # 键集与「有数据」时**完全一致**（值为 None），下游（报告/CSV/JSON）不必分支，
            # 800 环境里出现空窗口也能全量落盘。
            block = {"available": False, "samples": 0,
                     "joint_rms_rad": None, "joint_max_rad": None, "worst_joint": None,
                     "per_joint_rms_rad": None, "torque_saturated_frac": None,
                     "note": note or stats.get("note")}
        else:
            block = {"available": True, "samples": stats["samples"],
                     "joint_rms_rad": stats["joint_rms_rad"],
                     "joint_max_rad": stats["joint_max_rad"],
                     "worst_joint": stats["worst_joint"],
                     "per_joint_rms_rad": stats["per_joint_rms_rad"],
                     "torque_saturated_frac": stats["torque_saturated_frac"]}
        block.update({"rms_over_steady": None, "rms_delta_rad": None,
                      "per_joint_rms_over_steady": None})
        return block

    def empty_block(note: str) -> dict:
        return joint_block([], note=note)

    def forward_window(rows_, time_s, label):
        window = _rows_in_span(rows_, time_s, time_s + impact_window_s)
        block = joint_block(window, note=f"{label} 窗口没有记录")
        block.update({"time_s": time_s, "window_s": impact_window_s, "aligned": "forward"})
        return block

    def centred_window(rows_, time_s, label):
        window = _rows_in_span(rows_, time_s - impact_window_s, time_s + impact_window_s)
        block = joint_block(window, note=f"{label} 窗口没有记录")
        block.update({"time_s": time_s, "window_s": 2.0 * impact_window_s,
                      "half_window_s": impact_window_s, "aligned": "centred"})
        return block

    tow = phase_rows(rows, "tow")
    coast = phase_rows(rows, "coast")

    # ---------------------------------------------------------------- 起拖（tow 首行）
    startup_time = float(tow[0]["time_s"]) if tow else None
    startup = (forward_window(tow, startup_time, "起拖")
               if tow else empty_block("tow 段没有记录"))

    # ---------------------------------------------------------------- 指令归零（coast 首行）
    stop_time = float(coast[0]["time_s"]) if coast else None
    stop = (forward_window(coast, stop_time, "指令归零")
            if coast else empty_block("coast 段没有记录"))

    # ---------------------------------------------------------------- 绷直/穿绳（首个越阈样本）
    takeup_index = _takeup_index(rows, takeup_force_threshold_n, phase="tow")
    if takeup_index is None:
        takeup = empty_block(f"tow 段没有 |rope_tension_n| > {takeup_force_threshold_n:g} N "
                             f"的样本（未绷直/未穿绳，或该连接类型没有张力）")
        takeup.update({"force_threshold_n": takeup_force_threshold_n, "vx_mps": None,
                       "tension_n": None, "time_since_tow_start_s": None})
    else:
        takeup_row = tow[takeup_index]
        takeup_time = float(takeup_row["time_s"])
        takeup = centred_window(rows, takeup_time, "绷直")
        takeup.update({
            "force_threshold_n": takeup_force_threshold_n,
            "tension_n": float(takeup_row["rope_tension_n"]),
            "vx_mps": float(takeup_row["robot_vx_b_mps"]),
            "time_since_tow_start_s": takeup_time - startup_time,
        })

    # ---------------------------------------------------------------- 停车撞击（首个接触见证）
    contact_index, contact_channels = _contact_index(coast, record_dt=record_dt,
                                                     deck_limit_n=deck_limit_n)
    if contact_index is None:
        stop_contact = empty_block("coast 段没有接触见证（没有追尾撞击）")
        stop_contact.update({"contact": False, "channels": [], "deck_fx_n": None,
                             "load_vx_mps": None, "time_since_stop_s": None})
    else:
        contact_row = coast[contact_index]
        contact_time = float(contact_row["time_s"])
        stop_contact = centred_window(coast, contact_time, "停车撞击")
        stop_contact.update({
            "contact": True,
            "channels": contact_channels,
            "deck_fx_n": float(contact_row["cart_deck_fx_n"]),
            "load_vx_mps": float(contact_row["load_vx_mps"]),
            "time_since_stop_s": contact_time - stop_time,
        })

    # ---------------------------------------------------------------- 稳态（牵引段去首尾）
    if tow:
        start_s = float(tow[0]["time_s"]) + steady_margin_s
        end_s = float(tow[-1]["time_s"]) - steady_margin_s
        steady_rows = _rows_in_span(tow, start_s, end_s) if start_s <= end_s else []
        steady = joint_block(steady_rows, note="稳态窗没有记录")
        steady.update({"window_s": end_s - start_s if end_s >= start_s else 0.0,
                       "margin_s": steady_margin_s,
                       "start_s": start_s, "end_s": end_s,
                       "span_s": (float(tow[-1]["time_s"]) - float(tow[0]["time_s"]))})
    else:
        steady = empty_block("牵引段没有记录")
        steady.update({"window_s": 0.0, "margin_s": steady_margin_s,
                       "start_s": None, "end_s": None, "span_s": 0.0})

    # ---------------------------------------------------------------- 冲击 vs 稳态
    steady_rms = steady.get("joint_rms_rad")
    steady_per_joint = steady.get("per_joint_rms_rad")

    def ratio_against_steady(block):
        value = block.get("joint_rms_rad")
        if value is None or steady_rms is None or not math.isfinite(steady_rms) \
                or steady_rms <= 0.0:
            return None, None
        return value / steady_rms, value - steady_rms

    def per_joint_ratio(block):
        values = block.get("per_joint_rms_rad")
        if not values or not steady_per_joint:
            return None
        out = {}
        for name in names:
            base = steady_per_joint.get(name)
            value = values.get(name)
            out[name] = (value / base if value is not None and base not in (None, 0.0)
                         and math.isfinite(float(base)) else None)
        return out

    def with_ratios(block):
        out = dict(block)
        ratio, delta = ratio_against_steady(block)
        out["rms_over_steady"] = ratio
        out["rms_delta_rad"] = delta
        out["per_joint_rms_over_steady"] = per_joint_ratio(block)
        return out

    return {
        "available": True,
        "window_s": impact_window_s,
        "steady_margin_s": steady_margin_s,
        "takeup_force_threshold_n": takeup_force_threshold_n,
        "startup": with_ratios(startup),
        "takeup": with_ratios(takeup),
        "stop": with_ratios(stop),
        "stop_contact": with_ratios(stop_contact),
        # 稳态不与自己比 ⇒ rms_over_steady / rms_delta_rad 保持 None（明确表示"分母"）
        "steady": steady,
        "joint_names": names,
        "note": "四个冲击窗口各自独立统计，并与稳态（牵引段去首尾）比比值/增量；"
                "既有 startup/stop（--transition-window）字段不动，两者并存。",
    }


def stop_stats(rows, *, record_dt: float, tow_summary: dict, command_mps: float,
               transition_window_s: float, gap_margin_limit_m: float,
               joint_names=None, torque_limits=None) -> dict:
    """停车段指标：小车滑移距离、间距维持、停车瞬态的关节响应。

    滑移距离用**沿坡面切向的行程** `load_progress_m`（`gravity` 后端退化为世界 x 位移，
    与测量台的 `coast_load_travel_m` 同口径；`terrain` 后端则是沿坡面的真实行程，
    不是它的水平投影）；间距维持优先用车头几何间隙（FK，来自 `summarize_tow`），
    没有时退回挂点距。
    """
    coast = phase_rows(rows, "coast")
    if not coast:
        return _nan_summary("coast 段没有记录")
    stop_speed = float(coast[0]["load_vx_mps"])
    start_x = float(coast[0]["load_progress_m"])
    total = float(coast[-1]["load_progress_m"]) - start_x
    to_rest = None
    time_to_rest = None
    for row in coast:
        if abs(float(row["load_vx_mps"])) < 0.02:
            to_rest = float(row["load_progress_m"]) - start_x
            time_to_rest = float(row["time_s"]) - float(coast[0]["time_s"])
            break
    gaps = _col(coast, "rope_distance_m")
    window_steps = max(1, int(round(transition_window_s / record_dt)))
    # 追尾见证只看 **coast 段**：拖曳之前的接触是「生成/站定就贴上」，另一类问题。
    witness = contact_witness(coast, record_dt=record_dt)

    def summarize_tow_key(name, default=None):
        value = tow_summary.get(name, default)
        return value

    min_clearance = summarize_tow_key("min_clearance_coast_m")
    time_to_contact = summarize_tow_key("time_to_contact_after_stop_s")
    # 停车瞬态：指令归零后多久机器人真正停下来（|vx| < 0.05 m/s 并保持到窗口结束）
    settle_time = None
    for index, row in enumerate(coast[:window_steps]):
        if all(abs(float(later["robot_vx_mps"])) < 0.05 for later in coast[index:window_steps]):
            settle_time = float(row["time_s"]) - float(coast[0]["time_s"])
            break
    stop_joint = joint_track_stats(coast[:window_steps], joint_names=joint_names,
                                   torque_limits=torque_limits)
    contact = bool(witness["contact"]) if witness["contact"] is not None else None
    if min_clearance is not None and float(min_clearance) <= 0.0:
        contact = True
    margin_low = (min_clearance is not None and 0.0 < float(min_clearance) <= gap_margin_limit_m)
    return {
        "available": True,
        "cart_speed_at_stop_mps": stop_speed,
        "cart_coast_distance_m": total,
        "cart_coast_to_rest_m": to_rest,
        "cart_coast_time_to_rest_s": time_to_rest,
        "robot_travel_after_stop_m": (float(coast[-1]["robot_progress_m"])
                                      - float(coast[0]["robot_progress_m"])),
        "gap_at_stop_m": gaps[0],
        "min_gap_after_stop_m": min(gaps),
        "final_gap_m": gaps[-1],
        "clearance_at_stop_m": summarize_tow_key("clearance_at_stop_m"),
        "min_clearance_coast_m": min_clearance,
        "final_clearance_m": summarize_tow_key("final_clearance_m"),
        "time_to_contact_after_stop_s": time_to_contact,
        "contact": contact,
        "contact_channels": witness.get("channels", []),
        "deck_contact_peak_n": witness.get("deck_contact_peak_n"),
        "gap_margin_low": margin_low,
        "final_robot_vx_mps": float(coast[-1]["robot_vx_mps"]),
        "settle_time_s": settle_time,
        "joint_rms_rad": stop_joint.get("joint_rms_rad"),
        "joint_max_rad": stop_joint.get("joint_max_rad"),
        "worst_joint": stop_joint.get("worst_joint"),
        "torque_saturated_frac": stop_joint.get("torque_saturated_frac"),
        "per_joint_rms_rad": stop_joint.get("per_joint_rms_rad"),
        "body_vx_rms_mps": _rms(_col(coast[:window_steps], "robot_vx_b_mps")),
    }


def stability_stats(rows, *, transition_window_s: float, record_dt: float,
                    fall_height_limit_m: float, pitch_limit_rad: float,
                    pitch_fraction_limit: float) -> dict:
    """稳定性与记录有效性：跌倒判定、最大俯仰、是否出现非有限值。

    跌倒判据用**坡面法向高度**（`robot_surface_height_m`）与**相对坡面的俯仰**
    （`body_pitch_rel_rad`）：`terrain` 后端上机器人被出生旋转到与坡面垂直，
    直接拿世界系 z 或世界系俯仰会把「正常的 10° 站姿」判成跌倒。
    """
    if not rows:
        return _nan_summary("没有记录")
    numeric_columns = [name for name in TEST_FIELDS if name not in STRING_FIELDS]
    invalid_samples = 0
    for row in rows:
        for name in numeric_columns:
            if not math.isfinite(float(row[name])):
                invalid_samples += 1
                break
    heights = _col(rows, "robot_surface_height_m")
    pitches = [abs(float(row["body_pitch_rel_rad"])) for row in rows]
    pitch_fraction = sum(1 for value in pitches if value > pitch_limit_rad) / len(pitches)
    fell = (min(heights) < fall_height_limit_m) or (pitch_fraction > pitch_fraction_limit)
    fallen_phase = None
    if fell:
        for row in rows:
            if float(row["robot_surface_height_m"]) < fall_height_limit_m or \
                    abs(float(row["body_pitch_rel_rad"])) > pitch_limit_rad:
                fallen_phase = row["phase"]
                break
    return {
        "available": True,
        "invalid_samples": invalid_samples,
        "invalid": invalid_samples > 0,
        "min_robot_surface_height_m": min(heights),
        "final_robot_surface_height_m": heights[-1],
        # 兼容/交叉核对：世界系绝对高度单列出来（坡上它会随行程线性变化，不是跌倒指标）
        "min_robot_z_m": min(_col(rows, "robot_z_m")),
        "final_robot_z_m": float(rows[-1]["robot_z_m"]),
        "max_abs_pitch_rel_rad": max(pitches),
        "final_abs_pitch_rel_rad": pitches[-1],
        "max_abs_pitch_world_rad": max(abs(float(row["body_pitch_rad"])) for row in rows),
        "pitch_over_limit_frac": pitch_fraction,
        "fell": bool(fell),
        "fell_phase": fallen_phase,
    }


def compute_case_metrics(rows, *, case: dict, command_mps: float | None = None,
                         schedule: PhaseSchedule, record_dt: float, tow_summary: dict,
                         thresholds: dict, joint_names=None, torque_limits=None,
                         transition_window_s: float = 1.0, lane_keeping: str = "off",
                         lane_vy_limit: float = 0.4, lane_wz_limit: float = 0.8,
                         cart_present: bool = True,
                         impact_window_s: float = DEFAULT_IMPACT_WINDOW_S,
                         steady_margin_s: float = DEFAULT_STEADY_MARGIN_S,
                         takeup_force_threshold_n: float = DEFAULT_TAKEUP_FORCE_THRESHOLD_N,
                         count_jnt: bool = False) -> dict:
    """由逐记录步轨迹算出五项指标 + 判定。与仿真无关，可离线用合成轨迹复核。

    `case` 是 `EnvCase.to_dict()`：带 env 序号、cell、连接、长度、坡度量级与工作条件。
    `command_mps` 缺省取 `case["velocity_mps"]`（避免两处各写一份速度指令）。
    """
    case = dict(case)
    if "velocity_mps" not in case:
        raise ValueError("case 字典必须带 velocity_mps")
    if command_mps is None:
        command_mps = float(case["velocity_mps"])
    station = phase_rows(rows, "station")
    tow = phase_rows(rows, "tow")
    coast = phase_rows(rows, "coast")
    window_steps = max(1, int(round(transition_window_s / record_dt)))
    startup_rows = tow[:window_steps]
    startup_joint = joint_track_stats(startup_rows, joint_names=joint_names,
                                      torque_limits=torque_limits)
    startup = {
        "window_s": transition_window_s,
        "samples": len(startup_rows),
        "joint_rms_rad": startup_joint.get("joint_rms_rad"),
        "joint_max_rad": startup_joint.get("joint_max_rad"),
        "worst_joint": startup_joint.get("worst_joint"),
        "per_joint_rms_rad": startup_joint.get("per_joint_rms_rad"),
        "torque_saturated_frac": startup_joint.get("torque_saturated_frac"),
        # 起步的**速度**响应：体系 vx 的 RMS 误差、90% 到位时间、过冲
        "body_vx_rms_err_mps": _rms([float(row["robot_vx_b_mps"]) - command_mps
                                     for row in startup_rows]) if startup_rows else None,
        "time_to_90pct_s": _time_to_fraction(tow, command_mps, fraction=0.9, hold_s=0.25),
        "overshoot_ratio": (
            max((float(row["robot_vx_b_mps"]) for row in startup_rows), default=float("nan"))
            / command_mps - 1.0 if startup_rows else None),
    }
    speed = speed_track_stats(tow, command_mps)
    stop = stop_stats(rows, record_dt=record_dt, tow_summary=tow_summary,
                      command_mps=command_mps, transition_window_s=transition_window_s,
                      gap_margin_limit_m=thresholds["gap_margin_limit_m"],
                      joint_names=joint_names, torque_limits=torque_limits)
    lane = lane_stats(rows, vy_limit=lane_vy_limit, wz_limit=lane_wz_limit, tow_rows=tow)
    lane.update({"mode": lane_keeping, "yaw_target_rad": 0.0})
    # 冲击窗口独立统计（新增字段，与既有 startup/stop 并存；不参与 classify_case）
    impact = impact_stats(rows, record_dt=record_dt, impact_window_s=impact_window_s,
                          steady_margin_s=steady_margin_s,
                          takeup_force_threshold_n=takeup_force_threshold_n,
                          joint_names=joint_names, torque_limits=torque_limits)
    stability = stability_stats(rows, transition_window_s=transition_window_s,
                                record_dt=record_dt,
                                fall_height_limit_m=thresholds["fall_height_limit_m"],
                                pitch_limit_rad=thresholds["pitch_limit_rad"],
                                pitch_fraction_limit=thresholds["pitch_fraction_limit"])
    metrics = {
        "case": case,
        "schedule": schedule.to_dict(),
        "samples": {"station": len(station), "tow": len(tow), "coast": len(coast)},
        "startup": startup,
        "speed": speed,
        "stop": stop,
        "lane": lane,
        "stability": stability,
        "impact": impact,
        "cart_present": bool(cart_present),
    }
    if not cart_present:
        # 不拖车的 env：小车被横向停在 NO_CART_LATERAL_OFFSET_M 之外、绳力为 0 ⇒
        # 「追尾接触」「停车余量」「几何间隙」全部没有意义（会算成"离停放的小车很远"）。
        # 这里显式中性化，`classify_case` 就不会再产出 `COL`/`LOW`（判据本身保持不变）。
        for key in ("contact", "clearance_at_stop_m", "min_clearance_coast_m",
                    "min_clearance_station_m", "final_clearance_m",
                    "time_to_contact_after_stop_s", "load_vx_at_contact_mps"):
            stop[key] = None
        stop["gap_margin_low"] = False
        stop["cart_present"] = False
    metrics["verdict"] = classify_case(metrics, thresholds, count_jnt=count_jnt)
    return metrics


def _time_to_fraction(rows, command_mps: float, *, fraction: float, hold_s: float) -> float:
    """tow 段开始后，|vx_b − cmd| ≤ (1−fraction)·cmd 并保持 `hold_s` 的首个时刻。"""
    if not rows:
        return None
    tolerance = (1.0 - fraction) * command_mps
    hold = max(1, int(round(hold_s / max(1e-9, float(rows[1]["time_s"]) - float(rows[0]["time_s"]))))) \
        if len(rows) > 1 else 1
    start_time = float(rows[0]["time_s"])
    for index in range(len(rows)):
        window = rows[index:index + hold]
        if len(window) < hold:
            break
        if all(abs(float(row["robot_vx_b_mps"]) - command_mps) <= tolerance for row in window):
            return float(rows[index]["time_s"]) - start_time
    return None


def classify_case(metrics: dict, thresholds: dict, *, count_jnt: bool = False) -> dict:
    """把五项指标折成判定码（`OK` 或最严重的一条）+ 全部原因。

    `count_jnt`（默认 **False**，用户 2026-10-10 决定）：是否把 `startup_joint_error` /
    `stop_joint_error` 计入判定。**默认不计入** ⇒ `JNT` 不再出现在 `verdict.code` /
    `verdict.reasons`；`startup.*` / `stop.*` 的关节观测值照旧计算并打印，阈值
    （`joint_rms_limit_rad` / `joint_max_limit_rad`）也保留，只是不再产生失败原因。
    `--count-jnt` 打开时逐位复现 2026-10-10 之前的旧口径（供与归档对齐/追溯）。
    """
    reasons = []
    stability = metrics.get("stability", {})
    stop = metrics.get("stop", {})
    speed = metrics.get("speed", {})
    startup = metrics.get("startup", {})
    if stability.get("invalid"):
        reasons.append("invalid_record")
    if stability.get("fell"):
        reasons.append("robot_fell")
    if stop.get("contact"):
        reasons.append("stop_collision")
    elif stop.get("gap_margin_low"):
        reasons.append("stop_margin_low")
    lane = metrics.get("lane", {})
    if lane.get("available") and lane.get("mode") != "off":
        if lane.get("y_max_abs_m") is not None and \
                lane["y_max_abs_m"] > thresholds["lane_y_limit_m"]:
            reasons.append("lane_deviation")
        elif lane.get("heading_max_abs_rad") is not None and \
                lane["heading_max_abs_rad"] > math.radians(thresholds["lane_heading_limit_deg"]):
            reasons.append("lane_deviation")
    if speed.get("available") and speed.get("mae_mps") is not None and \
            speed["mae_mps"] > thresholds["speed_mae_ratio_limit"] * metrics["case"]["velocity_mps"]:
        reasons.append("speed_track_error")
    # 关节跟踪误差（JNT）：**默认不计入判定**（用户 2026-10-10 决定，见 docstring）。
    # 打开 `count_jnt` 时逐位复现旧口径（先 RMS 阈值、再单关节阈值，顺序不变）。
    if count_jnt:
        for key, limit_key in (("startup", "joint_rms_limit_rad"),
                               ("stop", "joint_rms_limit_rad")):
            section = metrics.get(key, {})
            value = section.get("joint_rms_rad")
            if value is not None and value > thresholds[limit_key]:
                reasons.append(f"{key}_joint_error")
        for key in ("startup", "stop"):
            section = metrics.get(key, {})
            value = section.get("joint_max_rad")
            if value is not None and value > thresholds["joint_max_limit_rad"]:
                reason = f"{key}_joint_error"
                if reason not in reasons:
                    reasons.append(reason)
    codes = [_REASON_TO_CODE[reason] for reason in reasons]
    code = max(codes, key=lambda item: VERDICT_SEVERITY[item]) if codes else "OK"
    return {"code": code, "reasons": reasons, "severity": VERDICT_SEVERITY[code]}


# ---------------------------------------------------------------- 报告


def case_report_row(metrics: dict) -> dict:
    """逐 case 的扁平表行（report.csv 用）。"""
    case = metrics["case"]
    startup, speed, stop = metrics["startup"], metrics["speed"], metrics["stop"]
    stability = metrics["stability"]
    verdict = metrics["verdict"]
    impact = metrics.get("impact", {})
    row = {
        "env_index": case.get("env_index"), "cell_index": case.get("cell_index"),
        "column": case.get("column"), "row": case.get("row"),
        "grade_deg": case.get("grade_deg"), "velocity_mps": case["velocity_mps"],
        "connection": case["connection"], "cart_mass_kg": case["cart_mass_kg"],
        "connection_length_m": case.get("connection_length_m"),
        "verdict": verdict["code"], "reasons": "|".join(verdict["reasons"]),
        "startup_joint_rms_rad": startup.get("joint_rms_rad"),
        "startup_joint_max_rad": startup.get("joint_max_rad"),
        "startup_worst_joint": startup.get("worst_joint"),
        "startup_torque_sat_frac": startup.get("torque_saturated_frac"),
        "startup_vx_rms_err_mps": startup.get("body_vx_rms_err_mps"),
        "startup_time_to_90pct_s": startup.get("time_to_90pct_s"),
        "speed_mae_mps": speed.get("mae_mps"),
        "speed_rmse_mps": speed.get("rmse_mps"),
        "speed_ratio_mean": speed.get("ratio_mean"),
        "speed_steady_mae_mps": speed.get("steady_mae_mps"),
        "lane_mode": metrics.get("lane", {}).get("mode"),
        "lane_y_rms_m": metrics.get("lane", {}).get("y_rms_m"),
        "lane_y_max_abs_m": metrics.get("lane", {}).get("y_max_abs_m"),
        "lane_heading_rms_rad": metrics.get("lane", {}).get("heading_rms_rad"),
        "lane_heading_max_abs_rad": metrics.get("lane", {}).get("heading_max_abs_rad"),
        "lane_vy_saturated_frac": metrics.get("lane", {}).get("vy_saturated_frac"),
        "lane_wz_saturated_frac": metrics.get("lane", {}).get("wz_saturated_frac"),
        "stop_cart_coast_distance_m": stop.get("cart_coast_distance_m"),
        "stop_cart_coast_to_rest_m": stop.get("cart_coast_to_rest_m"),
        "stop_cart_speed_at_stop_mps": stop.get("cart_speed_at_stop_mps"),
        "stop_min_clearance_coast_m": stop.get("min_clearance_coast_m"),
        "stop_final_clearance_m": stop.get("final_clearance_m"),
        "stop_time_to_contact_s": stop.get("time_to_contact_after_stop_s"),
        "stop_contact": stop.get("contact"),
        "stop_joint_rms_rad": stop.get("joint_rms_rad"),
        "stop_joint_max_rad": stop.get("joint_max_rad"),
        "stop_settle_time_s": stop.get("settle_time_s"),
        "stop_robot_travel_m": stop.get("robot_travel_after_stop_m"),
        "min_robot_surface_height_m": stability.get("min_robot_surface_height_m"),
        "max_abs_pitch_rel_rad": stability.get("max_abs_pitch_rel_rad"),
        "fell": stability.get("fell"),
        "summarize_tow_valid": metrics.get("summarize_tow", {}).get("valid"),
        "summarize_tow_failures": "|".join(metrics.get("summarize_tow", {}).get("failures", [])),
    }
    # 冲击窗口独立统计（新增列，与既有 startup_/stop_ 列并存）。
    for name in ("startup", "takeup", "stop", "stop_contact", "steady"):
        block = impact.get(name, {})
        available = bool(block.get("available"))
        row[f"impact_{name}_available"] = available
        row[f"impact_{name}_samples"] = block.get("samples", 0)
        row[f"impact_{name}_time_s"] = block.get("time_s")
        row[f"impact_{name}_joint_rms_rad"] = block.get("joint_rms_rad")
        row[f"impact_{name}_joint_max_rad"] = block.get("joint_max_rad")
        row[f"impact_{name}_worst_joint"] = block.get("worst_joint")
        row[f"impact_{name}_torque_sat_frac"] = block.get("torque_saturated_frac")
        row[f"impact_{name}_rms_over_steady"] = block.get("rms_over_steady")
        row[f"impact_{name}_rms_delta_rad"] = block.get("rms_delta_rad")
    takeup = impact.get("takeup", {})
    row["impact_takeup_tension_n"] = takeup.get("tension_n")
    row["impact_takeup_vx_mps"] = takeup.get("vx_mps")
    row["impact_takeup_force_threshold_n"] = takeup.get("force_threshold_n")
    contact = impact.get("stop_contact", {})
    row["impact_stop_contact_hit"] = contact.get("contact")
    row["impact_stop_contact_channels"] = "|".join(contact.get("channels") or [])
    row["impact_stop_contact_deck_fx_n"] = contact.get("deck_fx_n")
    row["impact_window_s"] = impact.get("window_s")
    row["impact_steady_margin_s"] = impact.get("steady_margin_s")
    return row


def group_of(case: dict) -> str:
    """工况分组的 key：**按 lane 的坡度量级**（0 / 5 / 10°）。

    旧版按「平地/上坡/下坡」分组是因为每个 case 自己就是一个恒定坡度；现在每条 lane
    的剖面自带一段上坡和一段下坡，方向不再是 case 的属性，能分的只有量级。
    """
    return grade_group_key(case["grade_deg"])


def grade_group_key(grade_deg) -> str:
    return f"grade{float(grade_deg):g}"


def grade_group_label(grade_deg) -> str:
    grade = float(grade_deg)
    return "平地" if grade == 0.0 else f"{grade:g}° 坡"


def group_statistics(case_summaries) -> dict:
    """按坡度量级（0 / 5 / 10°）汇总判定与失败模式计数。"""
    grades = sorted({float(summary["metrics"]["case"]["grade_deg"])
                     for summary in case_summaries})
    groups = {}
    for grade in grades:
        name = grade_group_key(grade)
        groups[name] = {"grade_deg": grade, "label": grade_group_label(grade),
                        "cases": 0, "verdicts": {}, "reasons": {}, "worst_cases": []}
    for summary in case_summaries:
        metrics = summary["metrics"]
        group = group_of(metrics["case"])
        entry = groups[group]
        entry["cases"] += 1
        code = metrics["verdict"]["code"]
        entry["verdicts"][code] = entry["verdicts"].get(code, 0) + 1
        for reason in metrics["verdict"]["reasons"]:
            entry["reasons"][reason] = entry["reasons"].get(reason, 0) + 1
        entry["worst_cases"].append({
            "case": metrics["case"], "code": code, "reasons": metrics["verdict"]["reasons"],
            "speed_mae_mps": metrics["speed"].get("mae_mps"),
            "startup_joint_rms_rad": metrics["startup"].get("joint_rms_rad"),
            "stop_joint_rms_rad": metrics["stop"].get("joint_rms_rad"),
            "min_clearance_coast_m": metrics["stop"].get("min_clearance_coast_m"),
            "cart_coast_distance_m": metrics["stop"].get("cart_coast_distance_m"),
        })
    for entry in groups.values():
        entry["cases_ok"] = entry["verdicts"].get("OK", 0)
        entry["worst_cases"].sort(
            key=lambda item: (-VERDICT_SEVERITY[item["code"]],
                              -(item["speed_mae_mps"] or 0.0)))
        entry["worst_cases"] = entry["worst_cases"][:5]
    return groups


def case_metric_summary(case_summaries) -> dict:
    """把一轮的 case 摘要压成「通过数 + 判定码分布 + 五项指标中位数」。

    「策略 vs 基线」的对照列与参照列都走这一个函数，保证两边的口径、字段、样本计数完全一致
    （不给策略新造指标、不改阈值）。可直接吃 `report.json["cases"]`（结构相同）。
    """
    codes = Counter(summary["metrics"].get("verdict", summary.get("verdict", {})).get("code")
                    for summary in case_summaries)

    def median_of(reader):
        values = []
        for summary in case_summaries:
            value = reader(summary.get("metrics", {}))
            if isinstance(value, (int, float)) and not isinstance(value, bool) \
                    and math.isfinite(float(value)):
                values.append(float(value))
        return (float(statistics.median(values)) if values else None), len(values)

    medians, samples = {}, {}
    for group, field, _label in FIVE_METRIC_FIELDS:
        key = f"{group}.{field}"
        medians[key], samples[key] = median_of(
            lambda metrics, group=group, field=field: metrics.get(group, {}).get(field))
    # 冲击窗口独立统计：同样按逐 case 中位数汇总（空窗口/无绷直/无接触的 case 不计入样本数，
    # 但样本数如实报出来 —— 与 `stop.min_clearance_coast_m` 的处理一致）。
    impact_medians, impact_samples = {}, {}
    for key, _label in IMPACT_METRIC_FIELDS:
        _, _, tail = key.partition(".")
        _, _, field = tail.rpartition(".")
        impact_medians[key], impact_samples[key] = median_of(
            lambda metrics, tail=tail, field=field: (
                metrics.get("impact", {}).get(tail.rpartition(".")[0], {}) or {}).get(field))
    # 增量（RMS − 稳态 [rad]）对四个冲击窗口都出中位数：报告表的「RMS−稳态」列直接读它。
    for group in IMPACT_GROUPS:
        if group == "steady":
            continue
        key = f"impact.{group}.rms_delta_rad"
        impact_medians[key], impact_samples[key] = median_of(
            lambda metrics, group=group: (
                metrics.get("impact", {}).get(group, {}) or {}).get("rms_delta_rad"))
    groups = group_statistics(case_summaries)
    by_grade = {name: {"grade_deg": entry["grade_deg"], "cases": entry["cases"],
                       "ok": entry["cases_ok"]}
                for name, entry in groups.items()}
    # 每个冲击窗口里「哪个关节是实测最差」的众数（中位数会把它抹掉）。
    modal_worst_joint = {}
    for group in IMPACT_GROUPS:
        tally = Counter()
        for summary in case_summaries:
            block = (summary.get("metrics", {}).get("impact", {}) or {}).get(group, {}) or {}
            name = block.get("worst_joint")
            if block.get("available") and isinstance(name, str):
                tally[name] += 1
        modal_worst_joint[group] = (tally.most_common(1)[0][0] if tally else None)
    window_s = steady_margin_s = takeup_force_threshold_n = None
    for summary in case_summaries:
        impact = summary.get("metrics", {}).get("impact", {}) or {}
        window_s = window_s if window_s is not None else impact.get("window_s")
        steady_margin_s = (steady_margin_s if steady_margin_s is not None
                           else impact.get("steady_margin_s"))
        takeup_force_threshold_n = (takeup_force_threshold_n
                                    if takeup_force_threshold_n is not None
                                    else impact.get("takeup_force_threshold_n"))
    return {"envs": len(case_summaries), "ok": codes.get("OK", 0),
            "codes": dict(codes), "medians": medians, "samples": samples,
            "impact_medians": impact_medians, "impact_samples": impact_samples,
            "impact_modal_worst_joint": modal_worst_joint,
            "impact_window_s": window_s,
            "impact_steady_margin_s": steady_margin_s,
            "impact_takeup_force_threshold_n": takeup_force_threshold_n,
            "by_grade": by_grade}


def _fmt_comparison_summary(summary) -> list:
    """一行摘要 + 一行中位数（`case_metric_summary` 的输出）。"""
    if summary is None:
        return ["- （未提供）"]
    codes = "、".join(f"{code} {count}" for code, count in
                      sorted(summary["codes"].items(), key=lambda item: -item[1])) or "无"
    lines = [f"- 通过 **{summary['ok']}/{summary['envs']}**；判定码分布（每 case 取最严重项）：{codes}",
             "- 五项指标中位数：" + "；".join(
                 f"{label} {_fmt(summary['medians'].get(f'{group}.{field}'))}"
                 f"（n={summary['samples'].get(f'{group}.{field}', 0)}）"
                 for group, field, label in FIVE_METRIC_FIELDS)]
    grade = "；".join(f"{name} {entry['ok']}/{entry['cases']}"
                      for name, entry in sorted(summary["by_grade"].items()))
    if grade:
        lines.append(f"- 逐坡度量级通过数：{grade}")
    return lines


#: 「冲击窗口 vs 稳态」表里的行：`(metrics["impact"] 下的组, 显示标签, 对齐说明)`。
IMPACT_TABLE_ROWS = (
    ("startup", "起拖（tow 首行）", "t0 → +W"),
    ("takeup", "绷直/穿绳（首个 ‖F‖ 越阈）", "t ± W"),
    ("stop", "指令归零（coast 首行）", "t0 → +W"),
    ("stop_contact", "停车撞击（首个接触见证）", "t ± W"),
    ("steady", "稳态（牵引段去首尾 margin）", "去首尾 margin"),
)


def _impact_table_cell(impact_medians, group: str, field: str, samples=None) -> str:
    """表单元格：`n/a` 或 `值（n=样本数）`（None/NaN 与空样本都显示 `n/a`）。"""
    value = (impact_medians or {}).get(f"impact.{group}.{field}")
    count = (samples or {}).get(f"impact.{group}.{field}")
    if value is None:
        return "n/a"
    text = f"{float(value):.3f}"
    if count is not None:
        text += f"（n={count}）"
    return text


def format_impact_section(current: dict, *, reference=None, archived=None,
                          window_s=None, steady_margin_s=None,
                          takeup_force_threshold_n=None) -> list:
    """`report.md` 的「冲击窗口 vs 稳态（关节响应）」一节。

    一行一个时刻（起拖 / 绷直 / 指令归零 / 停车撞击）+ 稳态一行；每行的数字都是**逐 case
    中位数**（与「策略 vs 基线」同一套 `case_metric_summary` 汇总），另取 `case_summaries`
    里每个 case 的 `worst_joint` 众数（`modal_worst_joint`）报出来 —— 「哪个关节在扛冲击」
    是中位数掩盖掉的信息。比值列与增量列直接给出「冲击相对稳态猛多少倍 / 高出多少 rad」，
    不需要读者拿两个绝对量心算。

    `current` 是 `case_metric_summary(...)` 的输出；`reference`（同版本基线，
    `--compare-report`）与 `archived`（归档基线，**旧几何、无该字段**）只作对照列，
    缺字段时显示 `n/a` 而不是报错。
    """
    window = current.get("impact_window_s", window_s)
    margin = current.get("impact_steady_margin_s", steady_margin_s)
    threshold = current.get("impact_takeup_force_threshold_n", takeup_force_threshold_n)
    params = []
    if window is not None:
        params.append(f"W = {float(window):g} s")
    if margin is not None:
        params.append(f"稳态 margin = {float(margin):g} s")
    if threshold is not None:
        params.append(f"绷直阈值 = {float(threshold):g} N")
    lines = ["## 冲击窗口 vs 稳态（关节响应）", ""]
    if params:
        lines.append("窗口参数：" + "；".join(params) + "。")
        lines.append("")
    lines += [
        "**读法**：`joint_rms_rad` = 窗口内 12 关节跟踪误差 `q − q*` 的合并 RMS [rad]；"
        "`RMS/稳态` 是同一 case 先取比值再取中位数（>1 = 比稳态猛，≈1 = 与稳态同量级）；"
        "`RMS−稳态` 是差值 [rad]；`力矩饱和%` = `|τ| > 0.95·limit` 的步占比 ×100。"
        "四行分别对齐「起拖 / 绷直 / 指令归零 / 停车撞击」四个**不同**时刻"
        "（归零与撞击可能差数百毫秒），另有稳态一行作分母。"
        "**这些是新增的观测字段，不参与判定码**（`classify_case` 只读既有 startup/stop）。", ""]
    header = ("| 时刻（对齐） | 关节 RMS [rad] | 最差关节（众数） | 力矩饱和% | RMS/稳态 | "
              "RMS−稳态 [rad] | 样本 n |")
    lines += [header, "|---|---|---|---|---|---|---|"]
    for group, label, aligned in IMPACT_TABLE_ROWS:
        modal = (current.get("impact_modal_worst_joint") or {}).get(group) or "n/a"
        lines.append("| " + " | ".join([
            f"{label} `{aligned}`",
            _impact_table_cell(current.get("impact_medians"), group, "joint_rms_rad",
                               current.get("impact_samples")),
            modal,
            _impact_table_cell(current.get("impact_medians"), group, "torque_saturated_frac",
                               current.get("impact_samples")),
            _impact_table_cell(current.get("impact_medians"), group, "rms_over_steady",
                               current.get("impact_samples")) if group != "steady" else "1.000（分母）",
            _impact_table_cell(current.get("impact_medians"), group, "rms_delta_rad",
                               current.get("impact_samples")) if group != "steady" else "0.000（分母）",
            str((current.get("impact_samples") or {}).get(f"impact.{group}.joint_rms_rad", "n/a")),
        ]) + " |")
    lines += ["",
              "- `最差关节（众数）` 是本轮各 case `impact.<时刻>.worst_joint` 里出现最多的那个；"
              "逐 case 的 `worst_joint` 与逐关节 RMS（`per_joint_rms_rad`）在 "
              "`report.json`／`summaries/<case>.json` 里，本表只给跨 case 汇总，"
              "免得表格随关节名变宽。",
              "- `绷直` 一行的 `n` 是**有绷直样本**的 case 数：绳/连杆张力全程不越阈（例如 "
              "rigid 压缩、compliant 未拉直）时该 case 不进这一行的中位数（在 report.json 里是 "
              "`available=false` + `note`）。`停车撞击` 一行同理（没有接触见证的 case 不计入）。",
              "- 既有 `startup.*` / `stop.*`（`--transition-window` 默认 1.0 s）**语义未动**，"
              "与本表并存：前者是归档对照口径，本表是缩短到 `--impact-window` 的冲击窗口。"]
    if reference is not None or archived is not None:
        lines += ["", "**与基线对照（逐 case 中位数，同口径）**", "",
                  "| 时刻 | 本轮 关节 RMS [rad] | 本轮 RMS/稳态 | 本轮 RMS−稳态 [rad] |", "|---|---|---|---|"]
        for group, label, _aligned in IMPACT_TABLE_ROWS:
            lines.append("| " + " | ".join([
                label,
                _impact_table_cell(current.get("impact_medians"), group, "joint_rms_rad",
                                   current.get("impact_samples")),
                (_impact_table_cell(current.get("impact_medians"), group, "rms_over_steady",
                                    current.get("impact_samples"))
                 if group != "steady" else "1.000（分母）"),
                (_impact_table_cell(current.get("impact_medians"), group, "rms_delta_rad",
                                    current.get("impact_samples"))
                 if group != "steady" else "0.000（分母）"),
            ]) + " |")
        if reference is not None:
            lines += ["", "同版本基线（`--compare-report`）的同一张表：", "",
                      "| 时刻 | 基线 关节 RMS [rad] | 基线 RMS/稳态 | 基线 RMS−稳态 [rad] |", "|---|---|---|---|"]
            for group, label, _aligned in IMPACT_TABLE_ROWS:
                lines.append("| " + " | ".join([
                    label,
                    _impact_table_cell(reference.get("impact_medians"), group, "joint_rms_rad",
                                       reference.get("impact_samples")),
                    (_impact_table_cell(reference.get("impact_medians"), group, "rms_over_steady",
                                        reference.get("impact_samples"))
                     if group != "steady" else "1.000（分母）"),
                    (_impact_table_cell(reference.get("impact_medians"), group, "rms_delta_rad",
                                        reference.get("impact_samples"))
                     if group != "steady" else "0.000（分母）"),
                ]) + " |")
        else:
            lines += ["", "同版本基线：未提供 `--compare-report`（冲击窗口的有效对照同样要**同一版本**"
                          "跑两轮：一轮基线 + 一轮开开关，参数完全相同）。"]
        if archived is not None:
            lines += ["", f"归档基线参照（{archived.get('label', '归档')}）："
                          + ("该归档没有 `impact.*` 字段（本统计晚于它），"
                             "只能用既有 `startup.*` / `stop.*` 做同量级 sanity check。"
                             if not any("impact" in key for key in (archived.get("medians") or {}))
                             else "含 `impact.*` 字段。")]
    lines.append("")
    return lines


def format_baseline_comparison(current: dict, *, reference=None, archived=None,
                               upper_enabled: bool = False,
                               count_jnt: bool = DEFAULT_COUNT_JNT) -> list:
    """report.md 的「策略 vs 基线」一节（同网格、同指标、同阈值）。

    - `current`：本轮结果（`case_metric_summary`）；
    - `reference`：**同版本**基线那轮的结果（`--compare-report <基线>/report.json`，可为 None）；
    - `archived`：归档基线读数（默认 `ARCHIVED_BASELINE_2026_10_09`），只作参照；
    - `count_jnt`：本轮判定口径（默认 False = JNT 不计入），写在节标题下方。
    """
    archived = ARCHIVED_BASELINE_2026_10_09 if archived is None else archived
    switch = ("**开启**（上层策略驱动 12 维关节位置残差）" if upper_enabled
              else "**关闭**（基线，残差恒 0）")
    recount = JNT_EXCLUDED_RECOUNT_2026_10_10
    lines = ["## 策略 vs 基线（同网格、同指标、同阈值）", "",
             "判定口径：**JNT " + ("计入**（`--count-jnt` 已打开，与 2026-10-10 之前的旧口径逐位一致）"
                                   if count_jnt else
                                   "不计入**（默认；用户 2026-10-10 决定）——"
                                   "`startup_joint_error`/`stop_joint_error` 只作观测，"
                                   "下面两个关节响应 RMS 中位数行仍在，但不产生失败原因"),
             f"**本轮**（开关{switch}）："]
    lines += _fmt_comparison_summary(current)
    lines.append("")
    if reference is not None:
        lines.append("**同版本基线**（`--compare-report` 传入的上一轮结果）：")
        lines += _fmt_comparison_summary(reference)
    else:
        lines.append("**同版本基线**：未提供 `--compare-report`。有效对照必须**同一版本**跑两轮"
                     "（一轮基线 + 一轮开开关，参数完全相同），把基线那轮的 `report.json` 用 "
                     "`--compare-report` 传进来；下面那列归档基线只是参照。")
    lines += ["", f"**归档基线参照**（{archived['label']}，`{archived['report']}`；"
                  f"git `{archived['git_commit'][:7]}`"
                  + ("，**工作树脏**，复现请用 " + archived["reproduce_commit"]
                     if archived.get("working_tree_dirty") else "")
                  + f"，`--no-cart-fraction {archived['no_cart_fraction']:g}`）：",
              f"- 通过 **{archived['ok']}/{archived['num_envs']}**；判定码分布（每 case 取最严重项）："
              + "、".join(f"{code} {count}" for code, count in
                          sorted(archived["codes"].items(), key=lambda item: -item[1])),
              "- 五项指标中位数：" + "；".join(
                  f"{label} {_fmt(archived['medians'].get(f'{group}.{field}'))}"
                  f"（n={archived['median_samples'].get(f'{group}.{field}', 0)}）"
                  for group, field, label in FIVE_METRIC_FIELDS),
              "- ⚠ **不可逐格硬比**：" + archived["grid_note"] + "。",
              "- ⚠ **口径不可比（JNT）**：" + archived["caliber_note"] + "。",
              f"- **新口径重判（同一批 {recount['num_envs']} cell，仅把 JNT 剔除统计量）**："
              f"基线 **{recount['ok']['baseline']}/{recount['num_envs']}**、"
              f"策略 **{recount['ok']['policy']}/{recount['num_envs']}**"
              f"（旧口径下两轮都是 {recount['old_caliber_ok']['baseline']}/"
              f"{recount['num_envs']}）；剩余失败原因 基线 "
              + " / ".join(f"{name} {count}" for name, count in
                           sorted(recount["reasons"]["baseline"].items(),
                                  key=lambda item: -item[1]))
              + " → 策略 "
              + " / ".join(f"{name} {count}" for name, count in
                           sorted(recount["reasons"]["policy"].items(),
                                  key=lambda item: -item[1]))
              + f"。来源 `{recount['source_report']}`（git "
              + f"`{recount['git_commit'][:7]}`）；逐坡度/连接/负载与依据见 README TOW-23。",
              "",
              "**`JNT` 已移出判定统计量（用户 2026-10-10 决定）**：" + archived["jnt_note"] + "。"
              "因此默认口径下 `JNT` 不再出现为失败原因，验收读 `COL`/`SPD`/`LOW`"
              "（以及 `LAT`/`FALL`）的占比变化。另注意开了开关后 `JNT` 的参考量变成 "
              "`held = 冻结输出 + 残差`（训练侧 `reference=\"commanded\"`），与基线的"
              "「冻结输出」**不是同一个量**，这一项的差不能当改善看"
              "（只有 `--count-jnt` 追溯旧口径时才相关）。",
              ""]
    return lines


def necessity_conclusion(groups: dict, thresholds: dict) -> dict:
    """由分组统计给出「任务是否有必要」的判读（只依据本网格 + 本阈值）。

    `thresholds` 可带 `count_jnt`（`caliber_thresholds()` 落盘的那一位）：默认
    **JNT 不计入**，于是「全部通过」的判据列表里不出现关节 RMS 阈值（它只是观测）。
    """
    count_jnt = bool(thresholds.get("count_jnt", DEFAULT_COUNT_JNT))
    lines = []
    verdict = {}
    if not groups:
        return {"verdict": {}, "lines": ["- 没有任何 case（记录为空）。"]}
    joint_clause = ("关节 RMS ≤ " + f"{thresholds['joint_rms_limit_rad']:g} rad、"
                    if count_jnt else "")
    for name, entry in groups.items():
        label = entry.get("label") or grade_group_label(entry.get("grade_deg", 0.0))
        if entry["cases"] == 0:
            verdict[name] = "no_cases"
            lines.append(f"- {label}：本网格没有该组 case。")
            continue
        ok, total = entry["cases_ok"], entry["cases"]
        reasons = ", ".join(f"{reason}×{count}"
                            for reason, count in sorted(entry["reasons"].items(),
                                                        key=lambda item: -item[1])) or "无"
        if ok == total:
            verdict[name] = "baseline_sufficient"
            lines.append(f"- {label}：{ok}/{total} 全部通过阈值（{joint_clause}跟速 MAE ≤ "
                         f"{thresholds['speed_mae_ratio_limit']*100:g}% 指令、停车间隙 > "
                         f"{thresholds['gap_margin_limit_m']:g} m、无接触/跌倒）"
                         f"⇒ 该组不构成上层任务必要性的证据。")
        else:
            verdict[name] = "baseline_insufficient"
            failure_kinds = [reason for reason in entry["reasons"]
                             if reason != "invalid_record"]
            stop_dominated = all(reason in ("stop_collision", "stop_margin_low", "stop_joint_error")
                                 for reason in failure_kinds) and bool(failure_kinds)
            lane_dominated = all(reason in ("lane_deviation",) for reason in failure_kinds) \
                and bool(failure_kinds)
            hint = ("失败集中在**停车段**：上层任务的必要性主要来自停车时序与间隙维持，"
                    "需用 `--command-shaping ramp` 或更早的 STOP 调度做对照，判断纯脚本 shaping "
                    "是否已经够用。" if stop_dominated else
                    "失败集中在**横向/朝向保持**：先调 PD 增益与限幅（`--lane-kp-y` 等），"
                    "确认不是自动整定问题之后再谈学习——这类侧向扰动本身是经典反馈的短板。"
                    if lane_dominated else
                    "失败跨起步/全程/停车多相：脚本 shaping 不足以解释，属于上层残差/调度的目标。")
            lines.append(f"- {label}：{ok}/{total} 通过；失败模式 {reasons}。{hint}")
    return {"verdict": verdict, "lines": lines}


def format_verdict_matrix(case_summaries, *, grades) -> str:
    """逐**坡度量级**打印「质量 × (速度/连接)」判定矩阵（人读报告与终端共用）。

    格 = 该组（同一坡度量级 + 速度 + 连接 + 质量）里的**最严重判定码** + `×env 数`。
    为什么带 env 数：工作条件是逐 env 轮转的，同一个 (速度, 连接, 质量) 组合在这张表里
    对应多个 cell（800 环境 / 15 个工作条件组合 ≈ 53 个），只写一个码会掩盖「多数 OK、
    少数失败」的情况；`REF×3` 直读成「这一格 3 个 env，最严重的一个是 REF」。
    """
    blocks = []
    for grade in grades:
        cases = [summary["metrics"] for summary in case_summaries
                 if float(summary["metrics"]["case"]["grade_deg"]) == float(grade)]
        if not cases:
            continue
        velocities = sorted({case["case"]["velocity_mps"] for case in cases})
        connections = [name for name in CONNECTIONS
                       if any(case["case"]["connection"] == name for case in cases)]
        masses = sorted({case["case"]["cart_mass_kg"] for case in cases})
        header = "| 质量 kg | " + " | ".join(
            f"{velocity:g} m/s {connection}" for velocity in velocities
            for connection in connections) + " |"
        divider = "|" + "---|" * (len(velocities) * len(connections) + 1)
        lines = [f"### 坡度量级 {grade_group_label(grade)}"
                 f"（{len(cases)} env）", "", header, divider]
        index = {}
        for case in cases:
            key = (case["case"]["velocity_mps"], case["case"]["connection"],
                   case["case"]["cart_mass_kg"])
            index.setdefault(key, []).append(case["verdict"]["code"])
        for mass in masses:
            cells = []
            for velocity in velocities:
                for connection in connections:
                    codes = index.get((velocity, connection, mass), [])
                    if not codes:
                        cells.append("-")
                        continue
                    worst = max(codes, key=lambda code: VERDICT_SEVERITY[code])
                    cells.append(f"{worst}×{len(codes)}")
            lines.append(f"| {mass:g} | " + " | ".join(cells) + " |")
        lines.append("")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def build_markdown_report(*, case_summaries, groups, conclusion, args_dict, thresholds,
                          schedule, grades, git, comparison=None, impact=None) -> str:
    """人读报告：配置、判定矩阵、失败模式、冲击窗口、结论、「策略 vs 基线」、限制。"""
    count_jnt = bool(args_dict.get("count_jnt", DEFAULT_COUNT_JNT))
    lines = ["# 拖曳上层任务必要性：冻结策略基线测试（训练场景）", "",
             f"- 生成时间：{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
             f"- git：`{git.get('commit')}`（worktree {git.get('working_tree') or 'clean'}）",
             f"- 场景：**训练场景** `UpperTowingSceneCfg`（`connection_grid` 40 列 × 20 行；"
             f"{args_dict['num_envs']} 环境一次跑完）",
             f"- 工作条件：速度档 {list(args_dict['velocities'])} m/s × 质量档 "
             f"{list(args_dict['cart_masses'])} kg，**确定性轮转**（`slot = row + column`：质量 "
             f"`slot % {len(args_dict['cart_masses'])}`、速度每 {len(args_dict['cart_masses'])} 个 "
             f"slot 换一档）；连接/长度/坡度量级由 cell 决定，不可扫",
             f"- 判据阈值里的质量档与训练随机范围 {list(TRAINING_MASS_RANGE_KG)} kg 不同："
             f"超出范围的档位属于外推检查",
             f"- 地面摩擦固定 {args_dict['ground_friction']:g}（训练随机 0.4–1.2）；轮轴阻尼 "
             f"{args_dict['wheel_damping']:g} N·m·s/rad（训练随机 0.008–0.032）",
             f"- 指令整形：{args_dict['command_shaping']}"
             + (f"（ramp {args_dict['ramp_time_s']:g} s）" if args_dict["command_shaping"] == "ramp" else ""),
             "- 度量坐标系固定为 lane 系（出生在剖面平地段 ⇒ 切向 +x、法向 +z、重力竖直）："
             "`progress` = x 行程（**沿坡面的水平投影**，不是弧长）；`surface_height` = "
             "`z − profile_height(本 env 坡度量级, x)`；`pitch_rel` = 世界系俯仰 + 局部坡度",
             f"- 每 case：station {schedule.station_steps * schedule.dt:.2f} s + tow "
             f"{schedule.tow_steps * schedule.dt:.2f} s + coast "
             f"{schedule.coast_steps * schedule.dt:.2f} s，共 {schedule.total_steps} 物理步 "
             f"（dt={schedule.dt:g} s，记录每 {args_dict['record_every']} 步一行）",
             f"- 冲击窗口独立统计：`--impact-window` "
             f"{args_dict.get('impact_window', DEFAULT_IMPACT_WINDOW_S):g} s、稳态去首尾 "
             f"`--steady-margin-s` {args_dict.get('steady_margin_s', DEFAULT_STEADY_MARGIN_S):g} s、"
             f"绷直阈值 `--takeup-force-threshold` "
             f"{args_dict.get('takeup_force_threshold', DEFAULT_TAKEUP_FORCE_THRESHOLD_N):g} N"
             f"（只进 `metrics.impact` 与本节，**不改判据/判定码**）",
             f"- 阈值：关节 RMS ≤ {thresholds['joint_rms_limit_rad']:g} rad / 单关节 ≤ "
             f"{thresholds['joint_max_limit_rad']:g} rad、跟速 MAE ≤ "
             f"{thresholds['speed_mae_ratio_limit'] * 100:g}% 指令、停车几何间隙 > "
             f"{thresholds['gap_margin_limit_m']:g} m、跌倒 base 高 < "
             f"{thresholds['fall_height_limit_m']:g} m 或 |pitch| > "
             f"{thresholds['pitch_limit_rad']:g} rad 的样本 > "
             f"{thresholds['pitch_fraction_limit'] * 100:g}%",
             f"- 判定口径：**JNT "
             + ("计入**（`--count-jnt` 已打开）——`startup_joint_error`/`stop_joint_error` 按上面"
                "两个关节阈值产生失败原因与判定码 `JNT`，与 2026-10-10 之前的旧口径**逐位一致**"
                if count_jnt else
                "不计入**（默认；用户 2026-10-10 决定）——`startup.*`/`stop.*` 的关节跟踪误差"
                "只作**观测**（下面「策略 vs 基线」的 `起步/停车关节响应 RMS` 中位数行仍在），"
                "不产生失败原因；要复现旧口径（归档对齐/追溯）加 `--count-jnt`"),
             "", "判定码：`OK` 通过；`LOW` 停车余量低；`LAT` 横向/朝向保持超限；"
             "`SPD` 跟速超限；`COL` 追尾接触；`FALL` 跌倒；`INV` 记录不可用；"
             "`JNT` 关节响应超限（**默认不计入判定**，仅 `--count-jnt` 时评估）。", "",
             "横向/朝向保持："
             + ("**关闭**（指令只有 vx）" if args_dict["lane_keeping"] == "off" else
                f"PD（kp_y={args_dict['lane_kp_y']:g}、kd_y={args_dict['lane_kd_y']:g}、"
                f"kp_yaw={args_dict['lane_kp_yaw']:g}、kd_yaw={args_dict['lane_kd_yaw']:g}；"
                f"vy 限幅 {args_dict['lane_vy_limit']:g} m/s、wz 限幅 "
                f"{args_dict['lane_wz_limit']:g} rad/s）")
             + f"；判据 |y| ≤ {thresholds['lane_y_limit_m']:g} m 且 |yaw| ≤ "
               f"{thresholds['lane_heading_limit_deg']:g}°（**前进速度不参与 PD，只给指令**）",
             "上层网络："
             + ("**关闭**（基线；12 维关节位置残差恒 0，JNT 参考 = 冻结策略当拍输出）"
                if args_dict.get("upper_checkpoint") is None else
                f"**开启**（checkpoint `{args_dict['upper_checkpoint']}`；"
                + ("随机采样" if args_dict.get("upper_stochastic") else "确定性均值")
                + "；每 10 个物理步 = 50 ms 推理一次，两次之间残差不变；"
                  "`held = 冻结目标 + clamp(actor,±1)⊙action_scale` 既是下发目标也是 JNT 参考）"),
             "",
             "## 逐坡度量级判定矩阵", "",
             format_verdict_matrix(case_summaries, grades=grades), "",
             "## 分组统计（按坡度量级）", ""]
    for name, entry in groups.items():
        label = entry.get("label") or grade_group_label(entry.get("grade_deg", 0.0))
        if entry["cases"] == 0:
            continue
        reasons = ", ".join(f"{reason}×{count}" for reason, count in
                            sorted(entry["reasons"].items(), key=lambda item: -item[1])) or "无"
        lines.append(f"- **{label}**：{entry['cases_ok']}/{entry['cases']} 通过；失败模式：{reasons}")
    if impact:
        lines += ["", *impact]
    lines += ["", "## 结论（任务是否有必要）", ""]
    lines += conclusion["lines"]
    if comparison:
        lines += ["", *comparison]
    lines += ["", "## 限制", "",
              "- 只覆盖本网格与本阈值：连接类型/长度/坡度量级由训练网格决定，不能扫；"
              "未测训练侧的域随机化（摩擦 0.4–1.2、轮轴阻尼 0.008–0.032、"
              "12.5% 无小车锚点）、未测 breakaway/Coulomb 阻力、未验真机、未做跨环境隔离。",
              "- 质量档 20/25 kg 超出训练采样范围 [5, 15] kg：那两档是外推检查，"
              "不能当作「训练分布内基线够不够」的证据。",
              "- 工作条件轮转用的是 `slot = row + column`（不是 env 序号）：列数 40 是质量档数的"
              "整数倍，用序号轮转会让质量只由列决定、与连接类型/坡度量级混淆。当前分配下"
              "`(连接 × 质量)` 与 `(坡度量级 × 质量)` 严格均衡，但**每个 cell 仍然只落到一种 "
              "(速度, 质量) 组合**（800 env = 800 cell），所以判读要按分档聚合，不能当逐 cell 的"
              "完整响应面。",
              "- 跟速一律用**体系** vx（与冻结策略观测同口径）；`summarize_tow` 的 "
              "`steady_tracking_ratio` 是世界系口径，坡上不要混用。",
              "- `progress` 是 x 行程（水平投影），不是坡面弧长：10° 剖面上两者差 < 1%。",
              "- **JNT 已按用户决定（2026-10-10）移出判定统计量**：`startup_joint_error` /"
              "`stop_joint_error` 不再产生失败原因，`JNT` 从判定码分布里消失（要复现旧口径加 "
              "`--count-jnt`，逐位一致）。**理由与实测分布**：这两条在 800 cell 上基线 800/800、"
              "策略 800/800，零区分度；它们量的是 `q − q*` 的底层 PD 静差（≈ τ/kp，扛体重），"
              "不是上层任务的职责 —— `joint_rms_limit_rad=0.10` 连最好的 case 都超 1.6 倍"
              "（实测基线 startup RMS P1=0.161 / P50=0.221、stop RMS P1=0.126 / P50=0.154，"
              "800/800 全部超限）。两条既有观测字段（`startup.*` / `stop.*` 的 `joint_rms_rad`、"
              "`joint_max_rad`、`worst_joint`、`per_joint_rms_rad`、`torque_saturated_frac`）与"
              "阈值 `--joint-rms-limit-rad` / `--joint-max-limit-rad` **照旧保留并在本节上方打印**"
              "（见「策略 vs 基线」的 `起步关节响应 RMS` / `停车关节响应 RMS` 中位数行），"
              "只是不再进判据。",
              "- `JNT` 的参考量在两种开关下不是同一个东西（基线 = 冻结策略当拍输出、策略 = 下发的"
              "`held = 冻结目标 + 残差`），所以即使开 `--count-jnt`，两轮的 `JNT` 差也不能直接当"
              "改善读。归档对照（`docs/towingdata/2026-10-09_necessity_800*`）的通过数与判定码"
              "（0/800、`JNT` 395 等）是按**旧口径**算的，与本节口径**不可比**。",
              f"- {SUMMARIZE_TOW_NOTE}",
              "- 冲击窗口统计（`metrics.impact.*`）是**观测字段，不进判定码**："
              "`classify_case` 只读既有 `startup.*`/`stop.*`，`DEFAULT_THRESHOLDS` 也不含 "
              "`--impact-window`/`--steady-margin-s`/`--takeup-force-threshold` 任何一项；"
              "四行的样本数可能不同（无绷直/无接触的 case 不进那一行的中位数）。"
              "绷直判定用 `|rope_tension_n|`，rigid 连杆压缩（负张力）也算「穿绳」的充要见证。",
              "- 冲击窗口的 `1.0 s` 归档对照列（`startup.*`/`stop.*`）**没有** `impact.*` 字段："
              "本统计晚于归档，只能同版本跑两轮做有效对照。",
              "- 仿真相位（PhysX 步进、绳力/轮阻施加、重力写入）只在训练机实跑验证；"
              "本脚本的离线测试只覆盖纯逻辑与记录字段契约。",
              ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- CLI


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="拖曳上层任务必要性：冻结 AMP 策略 + 脚本指令，直接跑训练场景的网格基线测试",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--policy", default="amp", help="冻结底层策略名（见 policy_cfg.POLICIES）")
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS,
                        help=f"并行环境数（默认训练网格全集 {DEFAULT_NUM_ENVS} = 40 列 × 20 行；"
                             f"不是 {DEFAULT_NUM_ENVS} 的整数倍时只覆盖网格前缀，脚本会警告）")
    parser.add_argument("--velocities", type=float, nargs="+", default=list(DEFAULT_VELOCITIES),
                        help="速度档（m/s，体系 x 指令）；按 slot = row + column 轮转分配")
    parser.add_argument("--cart-masses", type=float, nargs="+", default=list(DEFAULT_CART_MASSES),
                        help="小车目标总质量档（kg，质量与惯量同比例缩放）；按 slot = row + column 轮转分配")
    parser.add_argument("--ground-friction", type=float, default=FACTORY_FLOOR_FRICTION,
                        help="地面静/动摩擦系数（固定值；工厂地面常规 0.8，训练侧随机 0.4–1.2）")
    parser.add_argument("--wheel-damping", type=float, default=TRAINING_WHEEL_DAMPING,
                        help="轮轴黏性阻尼 b（N·m·s/rad；训练名义值 0.032，训练随机 0.008–0.032）")
    parser.add_argument("--command-shaping", choices=("direct", "ramp"), default="direct",
                        help="速度指令整形：direct = 阶跃（默认）；ramp = 固定斜坡对照")
    parser.add_argument("--ramp-time-s", type=float, default=1.0, help="ramp 整形的上升时间（s）")
    # 时序
    parser.add_argument("--settle-time", type=float, default=1.0, help="站定时长（s）")
    parser.add_argument("--tow-duration", type=float, default=5.0, help="拖曳时长（s）")
    parser.add_argument("--coast-duration", type=float, default=5.0, help="STOP 后滑行观测时长（s）")
    parser.add_argument("--transition-window", type=float, default=1.0,
                        help="起步/停车瞬态窗口（s），用于关节响应与速度响应统计"
                             "（归档对照口径，语义不动）")
    # 冲击窗口独立统计（用户 2026-10-10 要求）：只新增字段，不改既有口径。
    parser.add_argument("--impact-window", type=float, default=DEFAULT_IMPACT_WINDOW_S,
                        help=f"冲击瞬态窗口宽度 W（s，默认 {DEFAULT_IMPACT_WINDOW_S:g}）："
                             "起拖/指令归零向未来取 [t0, t0+W]，绷直/停车撞击围绕时刻取 [t±W]；"
                             "各窗口的关节 RMS/max/逐关节/力矩饱和独立统计，另与稳态比比值")
    parser.add_argument("--steady-margin-s", type=float, default=DEFAULT_STEADY_MARGIN_S,
                        help=f"稳态基线在牵引段**首尾各**去掉多少秒（默认 "
                             f"{DEFAULT_STEADY_MARGIN_S:g} s），剩下的关节 RMS 作为冲击的分母")
    parser.add_argument("--takeup-force-threshold", type=float,
                        default=DEFAULT_TAKEUP_FORCE_THRESHOLD_N,
                        help=f"判定「绷直/穿绳」的绳张力**模长**阈值（N，默认 "
                             f"{DEFAULT_TAKEUP_FORCE_THRESHOLD_N:g}）：首个 "
                             f"|rope_tension_n| 超过它的样本即真实冲击时刻（rigid 连杆张力有符号，"
                             f"压缩为负，所以取模长）")
    parser.add_argument("--dt", type=float, default=0.005, help="物理步长（s）")
    parser.add_argument("--record-every", type=int, default=DEFAULT_RECORD_EVERY,
                        help="每 N 个物理步记一行（默认 5 = 25 ms = 冻结策略周期；"
                             "800 环境逐物理步记录约 176 万行/2 GB，不可行。"
                             ">1 会让接触的速度跃变见证失去 5 ms 标定，脚本内部已按步长放大阈值）")
    # 输出
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="输出目录（默认 imgo2_rl/logs/towing/play_test/<时间戳>_<随机>）；绝不覆盖已有目录")
    parser.add_argument("--write-csv", choices=("failed", "all", "none"), default="failed",
                        help="逐记录步 CSV 的写入范围（all = 800 环境全写，可能数百 MB）")
    parser.add_argument("--headless", action="store_true", help="无显示运行")
    parser.add_argument("--no-cart-fraction", type=float, default=0.0,
                        help="不拖车的 env 比例（0 = 全部拖车，默认；1 = 全部不拖车 ⇒ 测无负载参考）。"
                             "无小车 env 的小车会横向停到 2 m 外、绳力与轮阻置 0，判定时跳过"
                             "与小车有关的项；分配是确定性的（每 round(1/fraction) 个 env 抽 1 个）")
    parser.add_argument("--compact-log", action="store_true",
                        help="逐 case 明细最多打印 --compact-log-detail 条（默认 30），"
                             "每 100 个 case 打一条带判定码计数的进度；"
                             "不加则 800 环境会把 800 行逐 env 判读全部打到终端")
    parser.add_argument("--compact-log-detail", type=int, default=30,
                        help="--compact-log 下逐 case 明细的行数上限（默认 30；0 = 一条都不打，"
                             "逐 env 全量指标仍写进 summaries/<case>.json 与 report.*）")
    parser.add_argument("--device", default="cuda:0", help="仿真设备")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印将执行的网格、逐 env 分配与代价，不启动 Isaac Sim（标准库即可运行）")
    # 横向/朝向保持（PD）：默认开，前进速度不参与
    parser.add_argument("--lane-keeping", choices=("pd", "off"), default="pd",
                        help="横向/朝向保持：pd = 用 PD 生成 vy/wz 指令把机器人压在中线并保持超前；"
                             "off = 只给 vx 指令（旧行为）")
    for name, value in DEFAULT_LANE_OPTIONS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=value, dest=name,
                            help=f"横向/朝向 PD 参数（默认 {value}）")
    # 上层网络开关：默认关 = 现在的基线（关节位置残差恒 0）；给了 checkpoint = 打开开关，
    # 用同一轮 run / 同一网格 / 同一指标做「策略 vs 基线」对照（README TOW-19 的待办）。
    parser.add_argument("--upper-checkpoint", type=Path, default=None,
                        help="上层策略联合 checkpoint（towing runner 保存的 model_*.pt）。"
                             "默认 None = 基线（12 维关节位置残差恒 0，行为与加开关前逐位一致）；"
                             "给了路径就打开上层网络：delta = clamp(action,±1)⊙action_scale，"
                             "held = 冻结策略关节目标 + delta 既下发又作为 JNT 指标参考")
    parser.add_argument("--upper-stochastic", action="store_true",
                        help="上层动作按 actor 分布采样（默认取均值，与 rl_lab/towing/play.py 的"
                             "确定性口径一致）")
    parser.add_argument("--compare-report", type=Path, default=None,
                        help="上一轮（**同一版本、同一套参数**）的 report.json：在 report.md 的"
                             "「策略 vs 基线」一节里给出同版本对照列。默认 None = 只给归档基线"
                             "（docs/towingdata/2026-10-09_necessity_800_noload，旧几何）参照")
    # 阈值
    for name, value in DEFAULT_THRESHOLDS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=value,
                            dest=name, help=f"判定阈值（默认 {value}）")
    # 判定口径：JNT 默认**不计入**（用户 2026-10-10 决定）；打开 = 复现旧口径。
    parser.add_argument("--count-jnt", action="store_true", default=DEFAULT_COUNT_JNT,
                        dest="count_jnt",
                        help="把 JNT（startup/stop 关节跟踪误差）计入判定。默认**不计入**："
                             "`classify_case` 不再产生 `startup_joint_error`/`stop_joint_error`，"
                             "`JNT` 从判定码分布里消失；`startup.*`/`stop.*` 的观测值与 "
                             "`--joint-rms-limit-rad`/`--joint-max-limit-rad` 照旧保留。"
                             "打开后与 2026-10-10 之前的旧口径**逐位一致**，供与归档对齐/追溯")
    args = parser.parse_args(argv)

    if args.num_envs <= 0:
        parser.error(f"--num-envs 必须是正整数，收到 {args.num_envs!r}")
    if args.num_envs % int(connection_grid.GRID_SIZE) != 0:
        print(f"[warn] --num-envs {args.num_envs} 不是训练网格 {connection_grid.GRID_SIZE} 的"
              f"整数倍：只覆盖网格前缀（前 {args.num_envs} 个 cell），不是完整网格。",
              file=sys.stderr)
    for name, value in DEFAULT_LANE_OPTIONS.items():
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} 必须是非负有限数")
    for name in ("lane_vy_limit", "lane_wz_limit"):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} 必须是正数")
    if not args.velocities:
        parser.error("--velocities 不能为空")
    for velocity in args.velocities:
        if not math.isfinite(velocity) or not 0.0 < velocity <= 2.0:
            parser.error(f"--velocities 必须在 (0, 2] m/s 内，收到 {velocity!r}")
    if not args.cart_masses:
        parser.error("--cart-masses 不能为空")
    for mass in args.cart_masses:
        if not math.isfinite(mass) or not 2.0 <= mass <= 50.0:
            parser.error(f"--cart-masses 必须在 [2, 50] kg 内，收到 {mass!r}")
    if not math.isfinite(args.ground_friction) or not 0.0 < args.ground_friction <= 2.0:
        parser.error(f"--ground-friction 必须在 (0, 2] 内，收到 {args.ground_friction!r}")
    for name in ("settle_time", "tow_duration", "coast_duration", "transition_window", "dt",
                 "ramp_time_s", "wheel_damping", "impact_window"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} 必须是有限正数，收到 {value!r}")
    if not math.isfinite(args.steady_margin_s) or args.steady_margin_s < 0.0:
        parser.error(f"--steady-margin-s 必须是非负有限数，收到 {args.steady_margin_s!r}")
    if not math.isfinite(args.takeup_force_threshold) or args.takeup_force_threshold <= 0.0:
        parser.error("--takeup-force-threshold 必须是有限正数，收到 "
                     f"{args.takeup_force_threshold!r}")
    if args.transition_window > args.tow_duration:
        parser.error("--transition-window 不能超过 --tow-duration（窗口取不到记录）")
    if args.impact_window > args.tow_duration:
        parser.error("--impact-window 不能超过 --tow-duration（起拖窗口取不到记录）")
    # `--steady-margin-s` 故意**不设**「2·margin < tow_duration」的硬约束：短冒烟回合
    # （如 --tow-duration 2.0 用默认 margin 1.0）稳态窗会退化，`impact_stats` 会把它标成
    # `available=false` + `note` 并让比值变 None，而不是让 CLI 报错挡住整轮。
    if args.record_every < 1:
        parser.error("--record-every 必须是正整数")
    for name, value in DEFAULT_THRESHOLDS.items():
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} 必须是有限正数，收到 {value!r}")
    args.num_envs = int(args.num_envs)
    if not 0.0 <= args.no_cart_fraction <= 1.0:
        parser.error(f"--no-cart-fraction 必须在 [0, 1]，收到 {args.no_cart_fraction!r}")
    # 上层开关：默认关；给了 checkpoint 就必须真实存在（在 Isaac Sim 起来之前失败，
    # 别把几分钟的启动浪费在一个拼错的路径上）。--upper-stochastic 单独给没有意义。
    if args.upper_checkpoint is not None:
        args.upper_checkpoint = args.upper_checkpoint.expanduser().resolve()
        if not args.upper_checkpoint.is_file():
            parser.error(f"--upper-checkpoint 指向的文件不存在：{args.upper_checkpoint}")
    elif args.upper_stochastic:
        parser.error("--upper-stochastic 只在同时给了 --upper-checkpoint 时有意义"
                     "（默认是基线：残差恒 0，没有可采样的动作）")
    # 同版本对照：只在这里校验「存在 + 是本测试台的 report.json」；解析后的数据**不放进
    # args**（`experiment.json` 会把 vars(args) 整个 dump 出去，塞进去会变成几十 MB）。
    if args.compare_report is not None:
        args.compare_report = args.compare_report.expanduser().resolve()
        if not args.compare_report.is_file():
            parser.error(f"--compare-report 指向的文件不存在：{args.compare_report}")
        try:
            payload = json.loads(args.compare_report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"--compare-report 不是可读的 JSON：{type(exc).__name__}: {exc}")
        if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
            parser.error("--compare-report 必须是本测试台的 report.json（顶层要含 cases 列表）")
    args.cases = build_env_cases(args.num_envs, args.velocities, args.cart_masses,
                                 no_cart_fraction=args.no_cart_fraction)
    args.schedule = make_schedule(settle_steps=int(round(args.settle_time / args.dt)),
                                  tow_duration=args.tow_duration,
                                  coast_duration=args.coast_duration, dt=args.dt)
    return args


def upper_policy_record(args) -> dict:
    """`experiment.json.upper_policy` 段（纯逻辑：离线测试直接构造并逐字段核对）。

    记的是**期望契约**（v3 双头：帧 58 / explicit 6 / latent 16 / 动作 13 = 1 维 vx 偏移 +
    12 维关节残差）与偏移头的**两层限幅 + 门控边界**。仿真路径（`main()`）会把
    `towing_contract` / `iter` / 维数 / `residual_scale` 覆写成**实际加载到**的值，并把
    `towing_contract_expected` 换成 `expected_towing_contract(upper.spec)`（从实际 spec 派生）；
    开关关时这份期望值仍然留在产物里（TOW-26 的记录失真问题）。
    """
    return {
        "enabled": args.upper_checkpoint is not None,
        "checkpoint": (str(args.upper_checkpoint) if args.upper_checkpoint is not None
                       else None),
        "deterministic": not args.upper_stochastic,
        "stochastic": bool(args.upper_stochastic),
        "mode": "stochastic" if args.upper_stochastic else "deterministic",
        # 加载后填充（见 sim 路径）：`towing_contract` = checkpoint 里的 4 个字段，`iter` = 训练轮数
        "towing_contract": None,
        "iter": None,
        # 期望契约（离线镜像；实跑时由运行时的 spec 覆写，两者不一致会直接报错）
        "towing_contract_expected": dict(UPPER_TOWING_CONTRACT_EXPECTED),
        "frame_dim_expected": UPPER_FRAME_DIM,
        "action_dim_expected": UPPER_ACTION_DIM,
        "cmd_action_dim": UPPER_CMD_ACTION_DIM,
        "joint_action_dim": UPPER_JOINT_ACTION_DIM,
        # 节拍：训练侧 upper_control_dt（20 Hz）；实跑时按 `--dt` 换成物理步数
        "control_dt_s": UPPER_CONTROL_DT_S,
        # 速度头（第 1 维 vx 偏移）的边界：两层限幅 + 出生段门控
        "offset": {
            "cmd_offset_scale_mps": UPPER_OFFSET_SCALE_MPS,
            "min_mps": UPPER_OFFSET_RANGE_MPS[0],
            "max_mps": UPPER_OFFSET_RANGE_MPS[1],
            "bounded": True,
            "clip": "offset = clamp(clamp(u_cmd, ±1) × cmd_offset_scale_mps, min_mps, max_mps)",
            "amp_vx_range_mps": list(UPPER_AMP_VX_RANGE_MPS),
            "compose": "loco_vx = clamp(task_vx + offset, amp_vx_range_mps[0], "
                       "amp_vx_range_mps[1])",
            "gate": "elapsed_s >= tow_start_s",
            "gate_after_s": UPPER_GATE_AFTER_S,
            "gate_note": "出生段不叠加偏移；**STOP 之后仍然生效**"
                         "（task_vx = 0 ⇒ loco_vx = offset）",
            "applied_before_frozen_policy": True,
        },
        "note": "默认关（基线，残差与偏移都恒 0）。开启后每 50 ms（训练侧 upper_control_dt = "
                "0.05 s = 20 Hz；默认 --dt 5 ms 时即每 10 个物理步）推理一次上层，"
                "两次之间偏移与残差保持不变；速度头第 1 维 = "
                "vx 偏移，两层限幅（先限偏移 [min_mps, max_mps]，再把和裁进 amp_vx_range_mps）"
                "后合成 loco_command（门控 elapsed_s >= tow_start_s，STOP 之后仍生效），"
                "合成后的指令送**冻结策略**得到冻结关节目标，再加 12 维残差 ⇒ held，"
                "held 只在冻结策略刷新那一拍重算并下发（同训练侧 apply_actions），"
                "既是下发目标也是 JNT 参考。加载前硬校验 towing_contract"
                "（v3：帧 58 / explicit 6 / latent 16），旧 v2（帧 57 / actor 79 / 动作 12）"
                "与 56/63 维 checkpoint 会被拒绝。",
        "loaded": False,
    }


def planned_grid_lines(args) -> list:
    """`--dry-run` / 启动横幅：网格、逐 env 分配与代价（纯逻辑，标准库可跑）。"""
    schedule = args.schedule
    cases = args.cases
    summary = env_case_summary(cases)
    recorded = schedule.total_steps // args.record_every
    lines = [
        f"[plan] 场景 = **训练场景** `UpperTowingSceneCfg`：{NUM_COLUMNS} 列 × {NUM_ROWS} 行 = "
        f"{connection_grid.GRID_SIZE} 条 lane（env i 的连接/长度/坡度量级取 "
        f"`connection_grid.env_spec(i)`，与训练逐位一致）",
        f"[plan] 并行环境数 {summary['envs']}（网格全集 {DEFAULT_NUM_ENVS} 的 "
        f"{summary['cell_repeats']:g} 倍 ⇒ 覆盖 {summary['cells']} 个 cell）；"
        f"一次仿真跑完，不再按坡度分轮",
        f"[plan] 网格分布：连接 " + " / ".join(f"{name} {count}" for name, count
                                              in summary["connections"].items())
        + "；坡度量级 " + " / ".join(f"{grade}° {count}" for grade, count
                                    in summary["grades_deg"].items())
        + f"；连接长度 {summary['length_m']['min']:g}–{summary['length_m']['max']:g} m",
        f"[plan] 工作条件（确定性轮转）：速度 {', '.join(f'{v:g}' for v in args.velocities)} m/s × "
        f"质量 {', '.join(f'{m:g}' for m in args.cart_masses)} kg = "
        f"{len(args.velocities) * len(args.cart_masses)} 组；slot = row + column（质量 "
        f"`slot % {len(args.cart_masses)}`、速度每 {len(args.cart_masses)} 个 slot 换一档）⇒ 每组 "
        f"{min(summary['bucket_env_counts'])}–{max(summary['bucket_env_counts'])} 个 env",
        f"[plan] 「连接 × 质量」env 数（slot 轮转 ⇒ 严格均衡）："
        + "；".join(f"{connection} " + "/".join(f"{mass}:{count}" for mass, count in counts.items())
                    for connection, counts in summary["mass_counts_by_connection"].items()),
        f"[plan] 训练侧质量随机范围 [{TRAINING_MASS_RANGE_KG[0]:g}, "
        f"{TRAINING_MASS_RANGE_KG[1]:g}] kg："
        + (("超出该范围的质量档 " + ", ".join(
            f"{m:g} kg" for m in args.cart_masses if m > TRAINING_MASS_RANGE_KG[1])
            + " ⇒ 那几档是外推检查，判读要与分布内档位分开")
           if any(m > TRAINING_MASS_RANGE_KG[1] for m in args.cart_masses)
           else "本网格的质量档全部落在训练分布内"),
        f"[plan] 物理量：地面摩擦 {args.ground_friction:g}；轮轴阻尼 "
        f"{args.wheel_damping:g} N·m·s/rad；地面/重力/传感器/资产全部沿用训练配置",
        f"[plan] 每 env：station {schedule.station_steps * schedule.dt:.2f} s"
        f"（{schedule.station_steps} 步）+ tow {schedule.tow_steps * schedule.dt:.2f} s"
        f"（{schedule.tow_steps} 步）+ coast {schedule.coast_steps * schedule.dt:.2f} s"
        f"（{schedule.coast_steps} 步）= {schedule.total_steps} 步 / "
        f"{schedule.total_steps * schedule.dt:.2f} s",
        f"[plan] 记录：每 {args.record_every} 物理步一行（{args.record_every * args.dt * 1000:g} ms）"
        f" ⇒ 每 env 约 {recorded} 行，共约 {recorded * len(cases)} 行；CSV 策略 "
        f"{args.write_csv}",
        f"[plan] 冲击窗口独立统计（只新增字段，不改判据）：`--impact-window` "
        f"{args.impact_window:g} s（起拖/归零向未来取、绷直/停车撞击围绕时刻取 ±）"
        f"；稳态 = 牵引段去首尾各 `--steady-margin-s` {args.steady_margin_s:g} s"
        f"（≈ 每 env {int(round(args.steady_margin_s / (args.record_every * args.dt))) * 2} 行"
        f"不计入稳态）"
        f"；绷直判定 = 首个 |rope_tension_n| > `--takeup-force-threshold` "
        f"{args.takeup_force_threshold:g} N；另报与接触发生时刻对齐的 `stop_contact_*`",
        f"[plan] 冲击口径与既有 `--transition-window` {args.transition_window:g} s "
        f"（归档对照）**并存**：新增 `impact_startup_*`/`impact_takeup_*`/`impact_stop_*`/"
        f"`impact_stop_contact_*`/`impact_steady_*`，既有 `startup.*`/`stop.*` 语义逐位不变",
        (f"[plan] 判定口径：**JNT 不计入**（默认；用户 2026-10-10 决定）"
         f"——`classify_case` 不因 `startup_joint_error`/`stop_joint_error` 产生失败原因，"
         f"`JNT` 从判定码分布里消失；两条观测值与 `--joint-rms-limit-rad` "
         f"{args.joint_rms_limit_rad:g} rad / `--joint-max-limit-rad` "
         f"{args.joint_max_limit_rad:g} rad 照旧保留（观测，不是判据）。"
         f"要复现旧口径（归档对齐/追溯）加 `--count-jnt`")
        if not args.count_jnt else
        (f"[plan] 判定口径：**JNT 计入**（`--count-jnt` 已打开）"
         f"——`startup_joint_error`/`stop_joint_error` 按 `--joint-rms-limit-rad` "
         f"{args.joint_rms_limit_rad:g} rad / `--joint-max-limit-rad` "
         f"{args.joint_max_limit_rad:g} rad 产生失败原因与判定码 `JNT`，"
         f"与 2026-10-10 之前的旧口径**逐位一致**；不加 `--count-jnt` 时 `JNT` 只是观测"),
        (f"[plan] **无小车（无负载参考）env**：{sum(1 for case in cases if not case.cart_present)}"
         f"/{len(cases)}（`--no-cart-fraction {args.no_cart_fraction:g}`；小车横向停 "
         f"{NO_CART_LATERAL_OFFSET_M:g} m、绳力与轮阻置 0、跳过 COL/LOW 判定）"),
        f"[plan] 指令整形：{args.command_shaping}"
        + (f"（ramp {args.ramp_time_s:g} s）" if args.command_shaping == "ramp" else "（阶跃）"),
    ]
    if args.lane_keeping == "off":
        lines.append("[plan] 横向/朝向保持：**关闭**（指令只有 vx，机器人可能漂离中线）")
    else:
        lines.append(
            f"[plan] 横向/朝向保持：PD（kp_y={args.lane_kp_y:g} kd_y={args.lane_kd_y:g} "
            f"kp_yaw={args.lane_kp_yaw:g} kd_yaw={args.lane_kd_yaw:g}；"
            f"vy≤{args.lane_vy_limit:g} m/s、wz≤{args.lane_wz_limit:g} rad/s）"
            f"⇒ 目标 y=0（lane 中线）、yaw=0（超前）；**vx 只给指令、不参与 PD**"
            f"；判据 |y|≤{args.lane_y_limit_m:g} m、|yaw|≤{args.lane_heading_limit_deg:g}°")
    if args.upper_checkpoint is None:
        lines.append("[plan] 上层网络：**关闭**（基线；12 维关节位置残差恒 0、vx 偏移恒 0"
                     "（`loco_command` ≡ 任务指令）；JNT 参考 = 冻结策略当拍输出"
                     "——与加开关之前逐位一致）；打开后按 **v3 双头**契约："
                     f"帧 {UPPER_FRAME_DIM} / 动作 {UPPER_ACTION_DIM}"
                     f"（{UPPER_CMD_ACTION_DIM} 维 vx 偏移 + {UPPER_JOINT_ACTION_DIM} 维关节残差）、"
                     "offset 有界且门控（加 `--upper-checkpoint` 后打印完整边界/门控）")
    else:
        lines.append(
            f"[plan] 上层网络：**开启**（checkpoint={args.upper_checkpoint}；"
            f"{'随机采样' if args.upper_stochastic else '确定性均值'}；"
            f"契约 **v3 双头**：帧 {UPPER_FRAME_DIM} = loco_command 3 + last_action 13 "
            f"+ base_ang_vel 3 + projected_gravity 3 + last_loco_action 12 + joint_pos 12 "
            f"+ joint_vel 12，动作 {UPPER_ACTION_DIM} = "
            f"{UPPER_CMD_ACTION_DIM} 维 vx 偏移 + {UPPER_JOINT_ACTION_DIM} 维关节残差；"
            f"vx 偏移 = clamp(clamp(u_cmd,±1)·{UPPER_OFFSET_SCALE_MPS:g}, "
            f"{UPPER_OFFSET_RANGE_MPS[0]:g}, {UPPER_OFFSET_RANGE_MPS[1]:g}) m/s（**有界**）"
            f" ⇒ loco_vx = clamp(task_vx + offset, {UPPER_AMP_VX_RANGE_MPS[0]:g}, "
            f"{UPPER_AMP_VX_RANGE_MPS[1]:g})（AMP 训练包络，**两层限幅**）；"
            f"偏移**门控**在 elapsed_s ≥ tow_start_s = {UPPER_GATE_AFTER_S:g} s"
            f"（训练侧 SETTLE_TIME_S；出生段不叠加、**STOP 之后仍生效**）；"
            f"合成后的 loco_command 送**冻结策略** ⇒ 冻结关节目标，再加残差 "
            f"delta = clamp(action,±1)⊙action_scale（hip 0.125 / thigh·shank 0.25 rad）"
            f" ⇒ held = 冻结目标 + delta 既下发又作为 JNT 参考；"
            f"训练节拍 20 Hz（每 "
            f"{max(1, int(round(UPPER_CONTROL_DT_S / args.dt)))} 物理步 = "
            f"{UPPER_CONTROL_DT_S:g} s = 训练侧 `upper_control_dt`）推理一次、"
            f"两次之间偏移与残差都不变；"
            f"加载前硬校验 towing_contract：version=3 / frame_dim=58 / explicit_dim=6 / "
            f"latent_dim=16，**旧 v2（帧 57 / actor 79 / 动作 12）与 56/63 维 checkpoint "
            f"会被拒绝**）")
    if args.compare_report is not None:
        lines.append(f"[plan] 同版本基线对照：{args.compare_report}"
                     f"（report.md / report.json 的「策略 vs 基线」一节会给出同口径对照列；"
                     f"另附归档基线 2026-10-09（旧几何）参照列）")
    else:
        lines.append("[plan] 同版本基线对照：未提供 `--compare-report`（report.md 只给归档基线"
                     "参照列；有效对照需同一版本跑两轮并把基线那轮的 report.json 传进来）")
    preview = min(8, len(cases))
    lines.append(f"[plan] 前 {preview} 个 env 的分配（env → 列/行、连接、长度、坡度量级、"
                 f"速度、质量）：")
    for case in cases[:preview]:
        lines.append(f"        env{case.env_index:04d} → c{case.column:02d}r{case.row:02d} "
                     f"{case.connection} L={case.length_m:.3f} m g={case.grade_deg:g}° "
                     f"v={case.velocity_mps:g} m/s m={case.cart_mass_kg:g} kg")
    return lines


def git_info():
    def run(*command):
        try:
            result = subprocess.run(["git", *command], cwd=REPO, capture_output=True,
                                    text=True, encoding="utf-8", errors="replace", timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": run("rev-parse", "HEAD"), "working_tree": run("status", "--porcelain")}


# ---------------------------------------------------------------- 仿真


def main(args):
    """跑网格。仿真相关 import 全部放在这里，保证 `--dry-run`／`--help` 只用标准库。"""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output_dir or
              RL_ROOT / "logs/towing/play_test" / f"{stamp}_{uuid.uuid4().hex[:8]}")
    output = output.expanduser().resolve()

    git = git_info()
    thresholds = caliber_thresholds(args)
    cases = args.cases
    distribution = env_case_summary(cases)
    experiment = {
        "state": "starting",
        "script": str(Path(__file__).resolve()),
        "question": "冻结 AMP 策略 + 脚本速度指令的基线，在**训练场景的 40×20 网格**上是否已满足"
                    "起步关节响应 / 全程跟速 / 停车滑移 / 停车间距 / 停车关节响应五项指标",
        "scene": {
            "source": "imgo2_rl.tasks.manager_based.towing.upper_env_cfg.UpperTowingSceneCfg",
            "note": "直接构造训练场景（同一地形 mesh、机器人/小车资产、五个车体-机器人接触"
                    "传感器、世界竖直重力）；本脚本只覆盖地面摩擦与轮轴阻尼这两个标量",
            "num_envs": args.num_envs,
            "grid": {"columns": NUM_COLUMNS, "rows": NUM_ROWS,
                     "grid_size": int(connection_grid.GRID_SIZE)},
            "cell_assignment": "env i → connection_grid.env_spec(i)"
                               "（row-major：列 = i % 40、行 = i // 40）",
            "distribution": distribution,
            "spawn": "剖面平地段起点、姿态竖直（lane 系 = 世界系）；两挂点三维距 = "
                     "env_spec(i)['initial_distance']（绳 = 0.8·L0、刚体 = L）",
            "work_conditions": {
                "velocities_mps": list(args.velocities),
                "cart_masses_kg": list(args.cart_masses),
                "rotation": "质量 = cart_masses[i % len(cart_masses)]；"
                            "速度 = velocities[(i // len(cart_masses)) % len(velocities)]",
                "note": "确定性轮转：每个 cell 只落到一种 (速度, 质量) 组合，"
                        "各组合的 env 数只差 ±1",
                "bucket_env_counts": distribution["bucket_env_counts"],
            },
            "caveats": [
                "不使用训练侧 12.5% 的无小车锚点：本测试台所有 env 都拖车",
                "速度指令由确定性轮转固定（训练侧每回合随机重采样）：同一 cell 只有一种"
                " (速度, 质量) 组合，因此看不到「同一 cell 在不同工作条件下的离散度」",
                f"质量档里 > {TRAINING_MASS_RANGE_KG[1]:g} kg 的档位超出训练随机范围 "
                f"[{TRAINING_MASS_RANGE_KG[0]:g}, {TRAINING_MASS_RANGE_KG[1]:g}] kg，"
                f"属于外推检查",
                f"地面摩擦固定 {args.ground_friction:g}（训练随机 0.4–1.2）；轮轴阻尼固定 "
                f"{args.wheel_damping:g} N·m·s/rad（训练随机 0.008–0.032）",
                ("关节位置残差与 vx 偏移都为 0（`loco_command` ≡ 任务指令）：跑的是冻结策略 + "
                 "脚本指令的基线，**默认不加载**任何上层 checkpoint（可用 `--upper-checkpoint` "
                 "打开）"
                 if args.upper_checkpoint is None else
                 f"上层网络**开启**（checkpoint `{args.upper_checkpoint}`）：v3 双头动作 13 维"
                 f"（1 维 vx 偏移 + 12 维关节残差）；偏移经两层限幅"
                 f"（[{UPPER_OFFSET_RANGE_MPS[0]:g}, {UPPER_OFFSET_RANGE_MPS[1]:g}] m/s → "
                 f"clamp 进 [{UPPER_AMP_VX_RANGE_MPS[0]:g}, {UPPER_AMP_VX_RANGE_MPS[1]:g}]）"
                 f"合成 loco_command 后送冻结策略，门控 elapsed_s >= tow_start_s；残差 = "
                 f"clamp(actor,±1)⊙action_scale，叠加在冻结策略关节目标上；JNT 指标参考 = "
                 f"实际下发的 held（与训练侧 reference=\"commanded\" 同口径）"),
            ],
        },
        "schedule": args.schedule.to_dict(),
        "thresholds": thresholds,
        "ground_friction": args.ground_friction,
        "wheel_damping_nms_per_rad": args.wheel_damping,
        "lane_keeping": {
            "mode": args.lane_keeping,
            "note": "横向/朝向由 PD 生成送冻结策略的 vy/wz 指令把机器人压在中线并保持超前；"
                    "前进速度 vx 只给脚本指令，不参与 PD",
            "yaw_target_rad": 0.0,
            "gains": {name: getattr(args, name) for name in DEFAULT_LANE_OPTIONS},
            "limits": {name: getattr(args, name)
                       for name in ("lane_y_limit_m", "lane_heading_limit_deg")},
        },
        "upper_policy": upper_policy_record(args),
        "connection": {
            "note": "连接类型/长度/弹性逐 env 由训练网格给定（env_spec），不可扫；"
                    "三类模型都用逐 env 的 rest_length 构造，非本类型的 k/c 填 "
                    "connection_grid.ELASTIC_KC[0] 占位（MultiRopeModel 用掩码忽略）",
            "counts": distribution["connections"],
        },
        "arguments": vars(args).copy() | {"cases": None, "schedule": None},
        "python": platform.python_version(),
        "git": git,
    }
    experiment["arguments"] = {key: (str(value) if isinstance(value, Path) else value)
                               for key, value in experiment["arguments"].items()}
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=False)

    print("\n".join(planned_grid_lines(args)), flush=True)
    print(f"[plan] 输出目录：{output}", flush=True)
    if args.dry_run:
        print("[plan] --dry-run：不启动 Isaac Sim。", flush=True)
        return 0

    def write_json(path: Path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_finite_only(data), indent=2, ensure_ascii=False,
                                   allow_nan=False, default=str) + "\n", encoding="utf-8")

    write_json(output / "experiment.json", experiment)
    summaries_dir = output / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    application = None
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(headless=args.headless, device=args.device)
        application = launcher.app

        import isaaclab.sim as sim_utils
        import isaaclab.utils.math as math_utils
        import torch
        from isaaclab.scene import InteractiveScene
        # 上层网络运行时：模块只 import torch + 标准库，桥接 rl_lab 的 actor/decoder。
        # `sys.path` 与训练脚本同一约定（把 imgo2_rl/scripts/rl_lab 加进路径）。
        if str(RL_ROOT / "scripts" / "rl_lab") not in sys.path:
            sys.path.insert(0, str(RL_ROOT / "scripts" / "rl_lab"))
        from upper_policy_runtime import (
            AMP_VX_MAX, AMP_VX_MIN, CMD_ACTION_DIM, COMMAND_OFFSET_MAX, COMMAND_OFFSET_MIN,
            COMMAND_OFFSET_SCALE, JOINT_ACTION_DIM, UpperPolicyRuntime,
            expected_towing_contract)
        from imgo2_rl.assets.cart import resolve_cart_path
        from imgo2_rl.assets.cart_model import read_cart_model
        from imgo2_rl.tasks.manager_based.towing.agents.upper_ppo_cfg import (
            UpperTowingPPORunnerCfg)
        from imgo2_rl.tasks.manager_based.towing.mdp.profile_torch import (
            profile_height_tensor)
        from imgo2_rl.tasks.manager_based.towing.mdp.resistance import viscous_resistance
        from imgo2_rl.tasks.manager_based.towing.mdp.rope import point_velocity
        from imgo2_rl.tasks.manager_based.towing.mdp.rope_model import (
            BodyProperties, MultiRopeModel, make_rope_model, world_inverse_inertia)
        from imgo2_rl.tasks.manager_based.towing.mdp.slope_geometry import (
            profile_slope_degrees as profile_slope_fn)
        from imgo2_rl.tasks.manager_based.towing.upper_env_cfg import (
            UpperTowingEnvCfg, UpperTowingSceneCfg)
        from imgo2_rl.tasks.manager_based.towing.utils.low_level_policy import (
            FrozenLowLevelPolicy, parts_from_robot_state)
        from imgo2_rl.tasks.manager_based.towing.utils.policy_cfg import get_policy

        policy_cfg = get_policy(args.policy)
        num_envs = args.num_envs
        dt = args.dt
        record_dt = dt * args.record_every

        # ------------------------------------------------------------ 训练场景
        training_cfg = UpperTowingEnvCfg()
        action_cfg = training_cfg.actions.high_level_velocity
        # 偏移门控以**训练 cfg 的实际值**为准（`--dry-run`／加载前记的是离线镜像
        # `UPPER_GATE_AFTER_S = episode_geometry.SETTLE_TIME_S`，两者应当相等）。
        experiment["upper_policy"]["offset"]["gate_after_s"] = float(action_cfg.tow_start_s)
        if num_envs != int(training_cfg.scene.num_envs):
            print(f"[info] 训练场景默认 {int(training_cfg.scene.num_envs)} 环境，本次用 "
                  f"{num_envs}（逐 env 参数仍按 `env_spec` 分配，非整数倍时只覆盖网格前缀）",
                  flush=True)
        sim_cfg = training_cfg.sim
        sim_cfg.device = args.device
        sim = sim_utils.SimulationContext(sim_cfg)
        # 渲染节拍：与训练侧 `self.cfg.sim.render_interval` 同口径（物理步计数）。
        render_interval = max(1, int(getattr(sim_cfg, "render_interval", 1)))
        scene_cfg = UpperTowingSceneCfg(num_envs=num_envs,
                                        env_spacing=training_cfg.scene.env_spacing)
        material = scene_cfg.terrain.physics_material
        material.static_friction = args.ground_friction
        material.dynamic_friction = args.ground_friction
        scene_cfg.robot.init_state.joint_pos = dict(zip(policy_cfg.joint_names,
                                                        policy_cfg.default_dof_pos))
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        robot, cart = scene["robot"], scene["cart"]
        deck_sensor = scene[action_cfg.collision_sensor_names[0]]
        wheel_sensors = [scene[name] for name in action_cfg.collision_sensor_names[1:]]
        origins = scene.env_origins
        cart_model = read_cart_model(resolve_cart_path())
        dt = sim.get_physics_dt()
        if not math.isclose(dt, args.dt, rel_tol=1e-6):
            raise RuntimeError(f"Simulator dt {dt} 与请求的 {args.dt} 不一致")

        # ------------------------------------------------------------ 契约核对
        if robot.num_joints != policy_cfg.num_joints:
            raise RuntimeError(f"机器人有 {robot.num_joints} 个关节，契约要求 {policy_cfg.num_joints}")
        asset_perm = policy_cfg.asset_permutation(robot.joint_names)
        policy_to_asset = torch.tensor(asset_perm, dtype=torch.long, device=args.device)
        asset_to_policy = torch.empty_like(policy_to_asset)
        asset_to_policy[policy_to_asset] = torch.arange(policy_cfg.num_joints,
                                                        dtype=torch.long, device=args.device)
        reordered_default = [float(value) for value in robot.data.default_joint_pos[0, policy_to_asset]]
        if any(abs(a - b) > 1e-6 for a, b in zip(reordered_default, policy_cfg.default_dof_pos)):
            raise RuntimeError("关节置换核对失败：模型默认关节角与契约 default_dof_pos 不一致")
        # 上层帧里的 `joint_pos` 项 = joint_pos − default_joint_pos（**策略关节顺序**），
        # 与 `upper_mdp.joint_pos_rel_policy_order` 逐项同口径。
        default_joint_pos_policy = robot.data.default_joint_pos[:, policy_to_asset].clone()
        base_ids, _ = robot.find_bodies([action_cfg.robot_body_name])
        cart_base_ids, _ = cart.find_bodies([action_cfg.cart_body_name])
        cart_joint_ids, _ = cart.find_joints(list(action_cfg.cart_wheel_joint_names),
                                             preserve_order=True)
        cart_wheel_ids, _ = cart.find_bodies(list(action_cfg.cart_wheel_body_names),
                                             preserve_order=True)
        if (len(base_ids) != 1 or len(cart_base_ids) != 1
                or len(cart_joint_ids) != 4 or len(cart_wheel_ids) != 4):
            raise RuntimeError("base/base_link 或小车四轮关节/轮体的解析结果不符合预期")
        if len(action_cfg.collision_sensor_names) != 5:
            raise RuntimeError("训练侧碰撞传感器数量不是 5（车斗 + 四轮）")
        # 约定：第 0 个是车斗（Cart/base_link）、后 4 个是轮子。按名字守一道，避免训练侧
        # 调整顺序后本脚本静默把「轮子」当「车斗」记进 `cart_deck_fx_n`。
        if "deck" not in action_cfg.collision_sensor_names[0]:
            raise RuntimeError(
                f"碰撞传感器第 0 项不是车斗：{action_cfg.collision_sensor_names[0]!r}"
                f"（训练侧约定 {action_cfg.collision_sensor_names}）")
        for name in action_cfg.collision_sensor_names[1:]:
            if "wheel" not in name:
                raise RuntimeError(f"碰撞传感器 {name!r} 不是轮子（顺序约定被改了）")
        decimation = max(1, int(round(policy_cfg.control_dt / dt)))
        if not math.isclose(decimation * dt, policy_cfg.control_dt, rel_tol=1e-6):
            raise RuntimeError("policy_cfg.control_dt 必须是物理 dt 的整数倍")
        policy = FrozenLowLevelPolicy(policy_cfg, device=args.device)
        recommended_settle = policy.reset()
        if args.schedule.station_steps < recommended_settle:
            print(f"[warn] station {args.schedule.station_steps} 步（{args.settle_time:g} s）"
                  f"短于冻结策略 reset 契约建议的 {recommended_settle} 步"
                  f"（{policy_cfg.reset_settle_s:g} s）：起拖前机器人可能还没站定，"
                  f"起步指标会被这段未站定的瞬态污染。", flush=True)
        if not math.isclose(args.wheel_damping, float(action_cfg.initial_wheel_damping),
                            rel_tol=1e-9):
            print(f"[warn] 轮轴阻尼 {args.wheel_damping:g} ≠ 训练名义值 "
                  f"{float(action_cfg.initial_wheel_damping):g} N·m·s/rad（训练侧随机 "
                  f"0.008–0.032）：本轮的判读不再代表训练分布中心。", flush=True)
        if not math.isclose(args.ground_friction, float(action_cfg.initial_ground_friction),
                            rel_tol=1e-9):
            print(f"[warn] 地面摩擦 {args.ground_friction:g} ≠ 训练名义值 "
                  f"{float(action_cfg.initial_ground_friction):g}（训练侧随机 0.4–1.2）。",
                  flush=True)
        if args.lane_keeping == "pd":
            for name, cap in (("lane_vy_limit", 1.0), ("lane_wz_limit", 1.57)):
                if getattr(args, name) > cap:
                    raise RuntimeError(
                        f"--{name.replace('_', '-')} {getattr(args, name):g} 超出 AMP 训练过的"
                        f"指令范围 ±{cap:g}：会给冻结策略喂分布外指令")

        # ------------------------------------------------------------ 上层网络开关（默认关）
        # 关：`upper = None` ⇒ 主循环只走基线路径（残差与偏移都恒 0，与加开关之前逐位一致）。
        # 开：从 checkpoint 恢复 actor + decoder。网络超参从**注册的 agent cfg** 建，
        # 加载前硬校验 `towing_contract`（v3：58/6/16；旧 v2 57/79 与 56/63 维 checkpoint 直接抛错）。
        upper = None
        if args.upper_checkpoint is not None:
            upper = UpperPolicyRuntime(
                args.upper_checkpoint, num_envs=num_envs,
                action_scale=policy_cfg.action_scale, device=args.device,
                deterministic=not args.upper_stochastic, agent_cfg=UpperTowingPPORunnerCfg())
            upper_mode = "deterministic" if upper.deterministic else "stochastic"
            print(f"[plan] 上层网络（checkpoint 已加载）：checkpoint={upper.checkpoint_path} "
                  f"iter={upper.iteration} towing_contract={upper.contract} "
                  f"mode={upper_mode} frame={upper.spec.frame_dim} "
                  f"actor={upper.spec.actor_obs_dim} critic={upper.spec.num_critic_obs} "
                  f"action={upper.spec.num_actions}"
                  f"（= {CMD_ACTION_DIM} 维 vx 偏移 + {JOINT_ACTION_DIM} 维关节残差）"
                  f"；vx 偏移有界 [{COMMAND_OFFSET_MIN:g}, {COMMAND_OFFSET_MAX:g}] m/s"
                  f"（scale {COMMAND_OFFSET_SCALE:g}）⇒ loco_vx 裁进 AMP 包络 "
                  f"[{AMP_VX_MIN:g}, {AMP_VX_MAX:g}]；门控 elapsed_s >= "
                  f"{float(action_cfg.tow_start_s):g} s（STOP 之后仍生效）", flush=True)
            # 期望契约从**实际加载的 spec** 派生，并与离线镜像逐字段核对（防两处漂移）。
            expected = expected_towing_contract(upper.spec)
            if (expected != UPPER_TOWING_CONTRACT_EXPECTED
                    or (upper.spec.num_actions, CMD_ACTION_DIM, JOINT_ACTION_DIM)
                    != (UPPER_ACTION_DIM, UPPER_CMD_ACTION_DIM, UPPER_JOINT_ACTION_DIM)
                    or (COMMAND_OFFSET_SCALE, COMMAND_OFFSET_MIN, COMMAND_OFFSET_MAX)
                    != (UPPER_OFFSET_SCALE_MPS, *UPPER_OFFSET_RANGE_MPS)
                    or (AMP_VX_MIN, AMP_VX_MAX) != UPPER_AMP_VX_RANGE_MPS):
                raise RuntimeError(
                    f"运行时期望契约 {expected} / 动作 {upper.spec.num_actions} / 偏移边界"
                    f"（scale {COMMAND_OFFSET_SCALE:g}、[{COMMAND_OFFSET_MIN:g}, "
                    f"{COMMAND_OFFSET_MAX:g}]）⇒ 包络 [{AMP_VX_MIN:g}, {AMP_VX_MAX:g}] "
                    f"与测量台的离线镜像（{UPPER_TOWING_CONTRACT_EXPECTED} / "
                    f"{UPPER_ACTION_DIM} / scale {UPPER_OFFSET_SCALE_MPS:g} / "
                    f"{list(UPPER_OFFSET_RANGE_MPS)} ⇒ {list(UPPER_AMP_VX_RANGE_MPS)}）不一致："
                    f"先同步 play_towing_test.py 的 UPPER_* 常量")
            experiment["upper_policy"].update(
                loaded=True, towing_contract=upper.contract,
                towing_contract_expected=expected, iter=upper.iteration,
                frame_dim=upper.spec.frame_dim, actor_obs_dim=upper.spec.actor_obs_dim,
                critic_obs_dim=upper.spec.num_critic_obs, action_dim=upper.spec.num_actions,
                residual_scale=[float(value) for value in upper.action_scale.tolist()])
            experiment["upper_policy"]["offset"].update(
                cmd_offset_scale_mps=float(COMMAND_OFFSET_SCALE),
                min_mps=float(COMMAND_OFFSET_MIN), max_mps=float(COMMAND_OFFSET_MAX),
                amp_vx_range_mps=[float(AMP_VX_MIN), float(AMP_VX_MAX)],
                gate_after_s=float(action_cfg.tow_start_s))
            write_json(output / "experiment.json", experiment)
        # 上层推理节拍 = 训练侧 `upper_control_dt`（0.05 s = 10 个物理步）；两次之间偏移与残差
        # 都保持不变，冻结策略每次刷新（每 `decimation` 步）都重算 held = 冻结目标 + 残差
        # （同 apply_actions）。两者**独立**：上层 tick 可以落在两次冻结刷新之间。
        upper_control_decimation = max(1, int(round(float(action_cfg.upper_control_dt) / dt)))
        if not math.isclose(upper_control_decimation * dt, float(action_cfg.upper_control_dt),
                            rel_tol=1e-6):
            raise RuntimeError("upper_control_dt 必须是物理 dt 的整数倍")
        if upper is not None:
            print(f"[upper] 推理节拍：每 {upper_control_decimation} 个物理步"
                  f"（{upper_control_decimation * dt * 1000:g} ms，训练侧 upper_control_dt="
                  f"{float(action_cfg.upper_control_dt):g} s）；冻结策略每 {decimation} 步刷新",
                  flush=True)
            gate_s = float(action_cfg.tow_start_s)
            if not math.isclose(args.settle_time, gate_s, rel_tol=1e-9):
                print(f"[warn] --settle-time {args.settle_time:g} s ≠ 训练侧 `tow_start_s` "
                      f"{gate_s:g} s：偏移门控按训练侧阈值（elapsed_s ≥ {gate_s:g} s），"
                      f"与脚本速度指令的相位不完全重合（那一段 `task_vx = 0`、偏移直接"
                      f"变成 loco_vx）。要对齐就传 --settle-time {gate_s:g}。", flush=True)
            if max(args.velocities) > AMP_VX_MAX:
                print(f"[warn] --velocities 上限 {max(args.velocities):g} > 冻结 AMP 包络上界 "
                      f"{AMP_VX_MAX:g}：开开关后合成 vx 一律裁进包络（训练侧同口径），"
                      f"基线轮不裁 ⇒ 这一档的两轮不可逐位对照。", flush=True)

        # 相机：一次覆盖整张 40×20 网格（默认 800 环境时跨度约 230 m × 110 m）
        centre = origins.mean(dim=0)
        extent = (origins.max(dim=0).values - origins.min(dim=0).values)
        span = max(float(extent.max()), 10.0)
        sim.set_camera_view((float(centre[0]) + 1.1 * span, float(centre[1]) - 1.1 * span,
                             0.9 * span), tuple(float(value) for value in centre))

        # ------------------------------------------------------------ 逐 env 参数
        specs = [connection_grid.env_spec(index) for index in range(num_envs)]
        grade_deg = torch.tensor([float(case.grade_deg) for case in cases],
                                 dtype=torch.float32, device=args.device)
        if any(abs(float(spec["slope_degrees"]) - case.grade_deg) > 1e-9
               for spec, case in zip(specs, cases)):
            raise RuntimeError("case 的坡度量级与 env_spec 不一致（内部错误）")
        velocities = torch.tensor([float(case.velocity_mps) for case in cases],
                                  dtype=torch.float32, device=args.device)
        connection_length = torch.tensor([float(spec["length"]) for spec in specs],
                                         dtype=torch.float32, device=args.device)
        initial_distance = torch.tensor([float(spec["initial_distance"]) for spec in specs],
                                        dtype=torch.float32, device=args.device)
        model_ids = torch.tensor([int(spec["model_index"]) for spec in specs],
                                 dtype=torch.long, device=args.device)
        placeholder_k, placeholder_c = connection_grid.ELASTIC_KC[0]
        rope_stiffness = torch.tensor(
            [float(spec["stiffness"]) if spec["stiffness"] is not None else placeholder_k
             for spec in specs], dtype=torch.float32, device=args.device)
        rope_damping = torch.tensor(
            [float(spec["damping"]) if spec["damping"] is not None else placeholder_c
             for spec in specs], dtype=torch.float32, device=args.device)

        # ---- 逐 env 的小车质量/惯量（质量档轮转）
        nominal_masses = cart.root_physx_view.get_masses().clone()
        nominal_inertias = cart.root_physx_view.get_inertias().clone()
        nominal_total = nominal_masses.sum(dim=1)
        if float(nominal_total.min()) <= 0.0:
            raise RuntimeError("小车名义总质量非正，无法按档缩放")
        mass_scales = torch.tensor([float(case.cart_mass_kg) / float(nominal_total[index])
                                    for index, case in enumerate(cases)],
                                   dtype=torch.float32, device=args.device)
        cart_env_idx = torch.arange(num_envs, dtype=torch.int, device="cpu")
        # `get_masses()/get_inertias()` 是 **CPU** 缓冲（PhysX 视图约定），缩放系数必须同设备，
        # 否则 CPU×CUDA 直接报 "Expected all tensors to be on the same device"。
        scaled_masses, scaled_inertias = scale_cart_mass_inertia(
            nominal_masses, nominal_inertias, mass_scales.detach().to("cpu"))
        cart.root_physx_view.set_masses(scaled_masses, cart_env_idx)
        cart.root_physx_view.set_inertias(scaled_inertias, cart_env_idx)
        actual_masses = cart.root_physx_view.get_masses().sum(dim=1).tolist()
        for index, case in enumerate(cases):
            if abs(actual_masses[index] - case.cart_mass_kg) > 1e-3 + 1e-3 * case.cart_mass_kg:
                raise RuntimeError(
                    f"env {index} 的小车质量缩放失败：目标 {case.cart_mass_kg:g} kg，"
                    f"PhysX 里是 {actual_masses[index]:g} kg")
        torque_limits = robot.data.joint_effort_limits[0, policy_to_asset].tolist()

        # ---- 出生几何（逐 env）：目标挂点距 = env_spec 的 initial_distance
        spawn = spawn_offsets(initial_distance.tolist(),
                              robot_height=float(robot.data.default_root_state[0, 2]),
                              cart_height=float(cart.data.default_root_state[0, 2]),
                              robot_offset=action_cfg.robot_attachment,
                              cart_offset=action_cfg.cart_attachment)
        print(f"[info] 出生几何：挂点法向高差 {spawn['normal_difference_m']:+.3f} m；"
              f"沿 lane 切向 {min(spawn['along_m']):.3f}–{max(spawn['along_m']):.3f} m；"
              f"两挂点三维距 {min(spawn['target_distance_m']):.3f}–"
              f"{max(spawn['target_distance_m']):.3f} m（= env_spec 的 initial_distance）",
              flush=True)

        # ---- 连接模型：三套都按**逐 env** 的长度/弹性张量构造（训练侧同一写法）
        rope_model = MultiRopeModel(
            models=(
                make_rope_model("compliant", rest_length=connection_length,
                                stiffness=rope_stiffness, damping=rope_damping),
                make_rope_model("inextensible", rest_length=connection_length,
                                position_gain=float(action_cfg.rope_position_gain),
                                max_correction_rate=float(action_cfg.rope_max_correction_rate)),
                make_rope_model("rigid", rest_length=connection_length,
                                position_gain=float(action_cfg.rigid_position_gain),
                                max_correction_rate=float(action_cfg.rigid_max_correction_rate)),
            ), model_ids=model_ids)

        # 观测里的 `projected_gravity` 是**单位**重力方向（训练侧同口径）；重力恒为世界竖直。
        gravity = (0.0, 0.0, -GRAVITY_MPS2)
        gravity_world = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=args.device)

        def per_env(vector_1x3):
            return vector_1x3.view(1, 3).expand(num_envs, 3)

        robot_attach = torch.tensor(action_cfg.robot_attachment, dtype=torch.float32,
                                    device=args.device).view(1, 1, 3)
        cart_attach = torch.tensor(action_cfg.cart_attachment, dtype=torch.float32,
                                   device=args.device).view(1, 1, 3)
        robot_zero_torque = torch.zeros(num_envs, robot.num_bodies, 3,
                                        dtype=torch.float32, device=args.device)
        cart_zero_torque = torch.zeros(num_envs, cart.num_bodies, 3,
                                       dtype=torch.float32, device=args.device)

        def robot_contact_fx(sensor):
            """车体与**机器人**之间的法向接触力（世界系 x 分量，逐 env）。

            训练场景的接触传感器是按机器人 body 过滤的（`filter_prim_paths_expr` 逐个列出
            机器人 link），所以必须读 `force_matrix_w`（shape `(N, 1, M, 3)`，M = 机器人 link
            数）再对 M 求和，**不能**用 `net_forces_w`：那个是「传感器 body 受到的净法向力」，
            含车体与地面/自身部件的全部接触，不是「机器人碰车」的度量。训练侧的
            `cart_collision` 终止判据读的也是这张矩阵。
            """
            matrix = sensor.data.force_matrix_w
            if matrix is None:
                raise RuntimeError("过滤后的机器人—小车接触力矩阵不可用（filter 配置丢了？）")
            return matrix[:, 0].sum(dim=1)[:, 0]

        def link_frame_force(asset, body_id, force_world):
            local = math_utils.quat_apply_inverse(asset.data.body_quat_w[:, body_id], force_world)
            return local.unsqueeze(1)

        def _quat_to_matrix(quat):
            w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
            return ((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
                    (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
                    (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))

        def host_buffer(tensor):
            return tensor.to(args.device)

        def inverse_inertia_world(asset, body_id):
            flat = host_buffer(asset.root_physx_view.get_inertias())[:, body_id]
            rotation = _quat_to_matrix(asset.data.body_quat_w[:, body_id])
            return world_inverse_inertia((flat[:, 0], flat[:, 4], flat[:, 8]), rotation)

        robot_mass_kg = host_buffer(robot.root_physx_view.get_masses()).sum(dim=1)

        def cart_mass_effective_kg():
            masses = host_buffer(cart.root_physx_view.get_masses())
            inertias = host_buffer(cart.root_physx_view.get_inertias())
            spun = sum(inertias[:, body_id, 4] for body_id in cart_wheel_ids)
            return masses.sum(dim=1) + 4.0 * spun / float(cart_model["wheel_radius_m"]) ** 2

        def apply_rope_and_resistance(*, wheel_damping: float):
            robot_offset_w = math_utils.quat_apply(robot.data.body_quat_w[:, base_ids[0]],
                                                   per_env(robot_attach))
            cart_offset_w = math_utils.quat_apply(cart.data.body_quat_w[:, cart_base_ids[0]],
                                                  per_env(cart_attach))
            robot_p = robot.data.body_pos_w[:, base_ids[0]] + robot_offset_w
            cart_p = cart.data.body_pos_w[:, cart_base_ids[0]] + cart_offset_w
            robot_v = point_velocity(robot.data.body_lin_vel_w[:, base_ids[0]],
                                     robot.data.body_ang_vel_w[:, base_ids[0]], robot_offset_w)
            cart_v = point_velocity(cart.data.body_lin_vel_w[:, cart_base_ids[0]],
                                    cart.data.body_ang_vel_w[:, cart_base_ids[0]], cart_offset_w)
            robot_props = BodyProperties(
                mass=robot_mass_kg,
                inverse_inertia_world=inverse_inertia_world(robot, base_ids[0]),
                offset=(robot_offset_w[:, 0], robot_offset_w[:, 1], robot_offset_w[:, 2]))
            cart_props = BodyProperties(
                mass=cart_mass_effective_kg(),
                inverse_inertia_world=inverse_inertia_world(cart, cart_base_ids[0]),
                offset=(cart_offset_w[:, 0], cart_offset_w[:, 1], cart_offset_w[:, 2]))
            state = rope_model.update(robot_point=robot_p, cart_point=cart_p,
                                      robot_velocity=robot_v, cart_velocity=cart_v, dt=dt,
                                      robot=robot_props, cart=cart_props)
            force_robot = torch.stack([torch.as_tensor(component)
                                       for component in state.force_on_robot], dim=-1) \
                * present.unsqueeze(1)
            force_cart = torch.stack([torch.as_tensor(component)
                                      for component in state.force_on_cart], dim=-1) \
                * present.unsqueeze(1)
            robot.set_external_force_and_torque(link_frame_force(robot, base_ids[0], force_robot),
                                                robot_zero_torque[:, :1],
                                                positions=robot_attach.expand(num_envs, 1, 3),
                                                body_ids=base_ids)
            cart.set_external_force_and_torque(link_frame_force(cart, cart_base_ids[0], force_cart),
                                               cart_zero_torque[:, :1],
                                               positions=cart_attach.expand(num_envs, 1, 3),
                                               body_ids=cart_base_ids)
            effort = torch.zeros_like(cart.data.joint_pos)
            effort[:, cart_joint_ids] = viscous_resistance(
                cart.data.joint_vel[:, cart_joint_ids], wheel_damping) * present.unsqueeze(1)
            cart.set_joint_effort_target(effort)
            return state

        def lane_command(vx_command):
            """组装送冻结策略的 3 维速度指令 `(vx, vy, wz)`。

            `vx` 是**单纯给指令**（脚本调度，用户要求不参与 PD）；`vy`/`wz` 由横向/朝向 PD
            按当前状态算出（`--lane-keeping pd`），目标是 lane 中线（`y = 0`）与超前朝向
            （`yaw = 0`，即 lane 的 +x 方向）。PD 关掉时 vy = wz = 0，与旧行为一致。
            """
            zeros = torch.zeros_like(vx_command)
            if args.lane_keeping == "off":
                return torch.stack([vx_command, zeros, zeros], dim=1)
            local = robot.data.root_pos_w - origins
            lane_y = local[:, 1]
            quat = robot.data.root_quat_w
            qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
            lane_yaw = torch.atan2(2.0 * (qw * qz + qx * qy),
                                   1.0 - 2.0 * (qy * qy + qz * qz))
            body_vy = robot.data.root_lin_vel_b[:, 1]
            body_wz = robot.data.root_ang_vel_b[:, 2]
            # 与 `lane_keeping_command()` 同一公式（横向误差投到机体系横向轴：−y·cos(yaw)）
            vy_command = args.lane_kp_y * (-lane_y * torch.cos(lane_yaw)) - args.lane_kd_y * body_vy
            wz_command = args.lane_kp_yaw * (0.0 - lane_yaw) - args.lane_kd_yaw * body_wz
            return torch.stack([vx_command,
                                vy_command.clamp(-args.lane_vy_limit, args.lane_vy_limit),
                                wz_command.clamp(-args.lane_wz_limit, args.lane_wz_limit)], dim=1)

        def projected_gravity_body():
            """机体系重力方向（单位向量）——冻结策略与上层帧共用同一个量。"""
            return math_utils.quat_apply_inverse(robot.data.root_quat_w, per_env(gravity_world))

        def policy_step(command_tensor):
            """冻结策略一拍：下发它自己的关节位置目标，返回该目标（基线口径）。"""
            parts = parts_from_robot_state(
                base_ang_vel=robot.data.root_ang_vel_b,
                projected_gravity=projected_gravity_body(),
                velocity_command=command_tensor,
                joint_pos=robot.data.joint_pos[:, policy_to_asset],
                joint_vel=robot.data.joint_vel[:, policy_to_asset])
            out = policy.step(parts)
            if upper is not None:
                # 上层帧里的 `last_loco_action` 是**冻结策略自己**的动作（策略关节顺序）。
                last_loco_action.copy_(out.action)
            robot.set_joint_position_target(out.joint_targets[:, asset_to_policy])
            return out.joint_targets

        def pitch_offsets_for(progress):
            """逐 env 的局部坡度（rad）：相对剖面参考姿态的俯仰偏移。

            只有 0/5/10 三个档位，所以按档位分组做（每组一次 torch 赋值），避免逐 env 标量
            调用在 800 环境 × 440 个记录步上累计成几百万次 Python 调用。
            """
            offsets = torch.zeros_like(progress)
            for grade in torch.unique(grade_deg).tolist():
                if float(grade) == 0.0:
                    continue
                mask = grade_deg == grade
                values = [math.radians(profile_slope_fn(grade, x))
                          for x in progress[mask].tolist()]
                offsets[mask] = torch.tensor(values, dtype=torch.float32, device=args.device)
            return offsets

        # 交叉核对：张量版剖面高度必须与训练侧标量版逐点一致（新的 15 m 剖面同样检查）
        for grade in (0.0, 5.0, 10.0):
            xs = torch.linspace(0.0, 14.0, 29, device=args.device)
            got = profile_height_tensor(torch.full_like(xs, grade), xs).tolist()
            want = [slope_geometry.profile_height(grade, float(x)) for x in xs.tolist()]
            if any(abs(a - b) > 1e-5 for a, b in zip(got, want)):
                raise RuntimeError(f"profile_height_tensor 与标量版不一致（档位 {grade:g}°）")

        # 逐 env 的「是否拖车」掩码：无小车 env 的绳力、轮阻全部置 0（训练侧 `cart_present` 同义）。
        present = torch.tensor([1.0 if case.cart_present else 0.0 for case in cases],
                               dtype=torch.float32, device=args.device)
        present_mask = present > 0.5

        def reset_episode():
            """写死一个回合的初始状态（`Articulation.reset()` 不写位姿，必须显式重写）。

            出生在剖面平地段 ⇒ 姿态竖直（单位四元数）；小车根由 `spawn_offsets()` 逐 env
            解出，保证两挂点三维距 = `env_spec` 的目标值（spawn 第一拍不产生约束力）。
            """
            robot_root = robot.data.default_root_state.clone()
            cart_root = cart.data.default_root_state.clone()
            robot_root[:, :3] += origins
            cart_root[:, :3] = origins + torch.tensor(spawn["cart_root"],
                                                      dtype=torch.float32, device=args.device)
            # 无小车 env：小车沿 lane 横向停到 NO_CART_LATERAL_OFFSET_M 外（与训练侧锚点同做法），
            # 并且绳力/轮阻在 apply_rope_and_resistance 里按 present 置 0（只靠横向距离不够稳）。
            if not bool(present.all()):
                cart_root[~present_mask, 1] += NO_CART_LATERAL_OFFSET_M
            robot_root[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32,
                                              device=args.device)
            cart_root[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32,
                                             device=args.device)
            for asset, root in ((robot, robot_root), (cart, cart_root)):
                root[:, 7:] = 0.0
                asset.write_root_pose_to_sim(root[:, :7])
                asset.write_root_velocity_to_sim(root[:, 7:])
                asset.write_joint_state_to_sim(asset.data.default_joint_pos.clone(),
                                               torch.zeros_like(asset.data.default_joint_vel))
            cart.set_joint_effort_target(torch.zeros_like(cart.data.joint_pos))
            robot.set_external_force_and_torque(robot_zero_torque, robot_zero_torque)
            cart.set_external_force_and_torque(cart_zero_torque, cart_zero_torque)
            policy.reset()
            if upper is not None:
                # 上层 GRU/残差/偏移同样回到零初值（本测试台每 env 只跑一个 episode，
                # 正常路径不会二次调用；保留是为了复用与「回合边界清零」语义完整）。
                # `upper.reset()` 清 GRU 与上一拍 13 维动作；另三个是本循环持有的缓存
                # （12 维残差、上一拍 13 维动作副本、上一拍合成出的 loco_command）。
                upper.reset()
                upper_delta.zero_()
                upper_processed.zero_()
                upper_loco_command.zero_()
            scene.reset()
            scene.update(dt)

        def make_row(env_index: int, *, phase: str, step: int, command: float,
                     joint_targets, joint_err_rms: float, joint_err_max: float,
                     progress_robot, progress_load, robot_height, load_height,
                     pitch_offsets, command_vy: float, command_wz: float, lane_offset: float,
                     lane_heading: float, body_vy: float, body_wz: float, load_offset: float,
                     deck_fx: float, wheel_fx: float) -> dict:
            case = cases[env_index]
            quat = robot.data.root_quat_w[env_index]
            qw, qx, qy, qz = (float(value) for value in quat)
            pitch = math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx))))
            row = {
                "phase": phase,
                "time_s": (step + 1) * dt,
                "user_cmd_mps": command,
                "ref_cmd_mps": command,
                "robot_vx_mps": float(robot.data.root_lin_vel_w[env_index, 0]),
                "load_vx_mps": float(cart.data.root_lin_vel_w[env_index, 0]),
                "rope_tension_n": float(state.rope_tension[env_index]),
                "rope_extension_m": float(state.rope_extension[env_index]),
                "rope_length_rate_mps": float(state.rope_length_rate[env_index]),
                "rope_taut": float(state.is_taut[env_index]),
                "rope_impulse_ns": float(state.rope_impulse[env_index]),
                "rope_distance_m": float(state.rope_length[env_index]),
                "robot_x_m": float(robot.data.root_pos_w[env_index, 0]),
                "load_x_m": float(cart.data.root_pos_w[env_index, 0]),
                "robot_z_m": float(robot.data.root_pos_w[env_index, 2]),
                "load_z_m": float(cart.data.root_pos_w[env_index, 2]),
                "body_pitch_rad": pitch,
                "body_pitch_rate_radps": float(robot.data.root_ang_vel_b[env_index, 1]),
                # 接触见证（训练传感器按**机器人**过滤 ⇒ 读 `force_matrix_w`）：
                #   cart_deck_fx_n  = 车斗（Cart/base_link）与机器人各 link 的法向力之和的世界 x 分量
                #   cart_wheel_fx_n = 四个轮子与机器人各 link 的法向力之和的世界 x 分量之和
                # 轮地接触不在过滤列表里，所以这两列**只反映机器人—小车接触**，与训练侧
                # `cart_collision`（同样读 force_matrix_w）同一口径。
                "cart_deck_fx_n": float(deck_fx),
                "cart_wheel_fx_n": float(wheel_fx),
                "robot_quat_x": float(quat[1]), "robot_quat_y": float(quat[2]),
                "robot_quat_z": float(quat[3]), "robot_quat_w": float(quat[0]),
                "load_quat_x": float(cart.data.root_quat_w[env_index, 1]),
                "load_quat_y": float(cart.data.root_quat_w[env_index, 2]),
                "load_quat_z": float(cart.data.root_quat_w[env_index, 3]),
                "load_quat_w": float(cart.data.root_quat_w[env_index, 0]),
                "velocity_cmd_mps": command,
                "robot_vx_b_mps": float(robot.data.root_lin_vel_b[env_index, 0]),
                "joint_err_rms_rad": joint_err_rms,
                "joint_err_max_rad": joint_err_max,
                # lane 坐标：出生在平地段 ⇒ lane 系 = 世界系；逐 env 的坡度量级参与换算
                "robot_progress_m": progress_robot[env_index],
                "load_progress_m": progress_load[env_index],
                "robot_surface_height_m": robot_height[env_index],
                "load_surface_height_m": load_height[env_index],
                "body_pitch_rel_rad": pitch + pitch_offsets[env_index],
                "grade_deg": case.grade_deg,
                "connection_length_m": case.length_m,
                "gravity_x_mps2": float(gravity[0]),
                "gravity_z_mps2": float(gravity[2]),
                "velocity_cmd_vy_mps": command_vy,
                "velocity_cmd_wz_radps": command_wz,
                "lane_offset_m": lane_offset,
                "lane_heading_rad": lane_heading,
                "robot_vy_b_mps": body_vy,
                "robot_wz_b_radps": body_wz,
                "load_offset_m": load_offset,
                "connection": case.connection,
                "cart_mass_kg": actual_masses[env_index],
            }
            row.update(wheel_omega_fields(cart.data.joint_vel[env_index], cart_joint_ids))
            row.update(joint_state_fields(
                pos=robot.data.joint_pos[env_index, policy_to_asset],
                target=joint_targets[env_index],
                torque=robot.data.applied_torque[env_index, policy_to_asset]))
            if set(row) != set(TEST_FIELDS):
                missing = sorted(set(TEST_FIELDS) - set(row))
                extra = sorted(set(row) - set(TEST_FIELDS))
                raise RuntimeError(f"记录字段与契约不一致：缺 {missing}，多 {extra}")
            return row

        # ------------------------------------------------------------ 主循环（单轮跑完）
        case_rows = [[] for _ in range(num_envs)]
        joint_targets = torch.zeros(num_envs, policy_cfg.num_joints,
                                    dtype=torch.float32, device=args.device)
        # 冻结策略的关节位置目标（**不含**上层残差）与冻结策略自己的动作：
        # 只有开了上层网络才会被读/写；关时它们不参与任何计算。
        loco_joint_targets = torch.zeros_like(joint_targets)
        last_loco_action = torch.zeros(num_envs, policy_cfg.num_joints,
                                       dtype=torch.float32, device=args.device)
        # 上层残差（rad，策略关节顺序）：首拍为零，之后每 `upper_control_decimation` 步刷新。
        upper_delta = (torch.zeros(num_envs, policy_cfg.num_joints, dtype=torch.float32,
                                   device=args.device) if upper is not None else None)
        # 上一拍上层动作（13 维，clamp 后）与**上一拍合成出的** loco_command（送冻结策略、
        # 也进本拍帧）：只有开关开时才被读/写；关时它们不参与任何计算（基线路径原样）。
        upper_processed = (torch.zeros(num_envs, upper.spec.num_actions, dtype=torch.float32,
                                       device=args.device) if upper is not None else None)
        upper_loco_command = (torch.zeros(num_envs, 3, dtype=torch.float32,
                                          device=args.device) if upper is not None else None)
        command_tensor = torch.zeros(num_envs, 3, dtype=torch.float32, device=args.device)
        tangent_t = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=args.device)
        normal_t = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=args.device)
        reset_episode()
        print(f"\n[run] 训练场景 {num_envs} 环境一次跑完："
              f"{len(set(case.connection for case in cases))} 类连接、"
              f"{len({case.grade_deg for case in cases})} 个坡度量级、"
              f"{len({case.length_m for case in cases})} 种连接长度、"
              f"{len(args.velocities) * len(args.cart_masses)} 个工作条件组合；"
              f"重力 {gravity}（世界竖直）；上层网络"
              f"{'**开启**（v3 双头：帧 58 / 动作 13）' if upper is not None else '关闭（基线，残差与偏移恒 0）'}",
              flush=True)
        started = time.time()
        # 进度打印节拍：全程约 20 行，够看出在走又不刷屏（长跑的黑盒问题见循环内的注释）。
        progress_every = max(1, args.schedule.total_steps // 20)
        for step in range(args.schedule.total_steps):
            phase = args.schedule.phase_of(step)
            step_in_phase = args.schedule.step_in_phase(step)
            if phase == "tow":
                if args.command_shaping == "direct":
                    vx_command = velocities
                else:
                    vx_command = velocities * shaped_command(
                        phase="tow", step_in_phase=step_in_phase, velocity=1.0,
                        shaping="ramp", ramp_time_s=args.ramp_time_s, dt=dt)
            else:
                vx_command = torch.zeros_like(velocities)
            if upper is None:
                # ---- 基线路径（开关关）：与加开关之前**逐位一致** ----
                # 指令（含 PD 的 vy/wz）每控制步刷新一次，两次刷新之间保持不变；残差恒 0、
                # 偏移恒 0（`loco_command` ≡ 任务指令）⇒ 冻结策略拿到原样的指令。
                if step % decimation == 0:
                    command_tensor = lane_command(vx_command)
                    loco_joint_targets = policy_step(command_tensor)
                    joint_targets = loco_joint_targets
            else:
                # ---- 上层 v3 双头接线（开关开）----
                # 1) 上层一拍：训练侧 20 Hz（每 `upper_control_decimation` 个物理步），
                #    **与冻结策略的 `decimation` 步刷新独立**（训练里 `process_actions` 由 env
                #    step 驱动、`apply_actions` 由每个物理步驱动 ⇒ tick 会落在两次刷新之间）。
                #    帧里的量全部按 `upper_mdp.policy_frame` 的口径取：`last_action` 由 runtime
                #    自己持有（上一拍 clamp 后的 13 维动作）；`loco_command` 取**上一拍合成、
                #    这一拍仍在驱动底层**的那条指令（训练侧观测在 action 之前算，同口径）；
                #    关节位置/速度按**策略关节顺序**，joint_pos 还要减默认角。
                if step % upper_control_decimation == 0:
                    upper_delta, upper_processed = upper.act(
                        loco_command=upper_loco_command,
                        base_ang_vel=robot.data.root_ang_vel_b,
                        projected_gravity=projected_gravity_body(),
                        last_loco_action=last_loco_action,
                        joint_pos_rel=(robot.data.joint_pos[:, policy_to_asset]
                                       - default_joint_pos_policy),
                        joint_vel=robot.data.joint_vel[:, policy_to_asset])
                if step % decimation == 0:
                    command_tensor = lane_command(vx_command)
                    # 2) 速度头：动作第 1 维 = vx 偏移。**两层限幅**：先由 runtime 把偏移量本身
                    #    限进 [offset_min, offset_max]（`u_cmd` 先 clamp 到 ±1 再乘尺度），再把
                    #    「和」裁进冻结 AMP 策略的训练包络；门控 `elapsed_s >= tow_start_s`
                    #    （出生段不叠加；**STOP 之后仍然生效**）。**必须在冻结策略推理之前**
                    #    合成：偏移进的是底层观测（`velocity_command=self.loco_command`）。
                    offset_vx = upper.command_offset_vx(upper_processed)
                    loco_vx = compose_loco_vx_offset(
                        command_tensor[:, :1], offset_vx,
                        active=(step * dt) >= float(action_cfg.tow_start_s),
                        amp_vx_min=AMP_VX_MIN, amp_vx_max=AMP_VX_MAX)
                    # vy/wz 是 PD 外环给出的，原样透传（第 0 维才是 vx，同 `upper_mdp.CMD_ACTION_DIM`）
                    upper_loco_command = command_tensor.clone()
                    upper_loco_command[:, :1] = loco_vx
                    # 3) 合成后的 loco_command 喂**冻结策略** ⇒ 冻结关节目标；
                    # 4) 再加 12 维残差 ⇒ held。`held` 既下发、又作为 JNT 指标的参考量
                    #    （与训练侧 `low_position_error(reference="commanded")` 同口径）。
                    loco_joint_targets = policy_step(upper_loco_command)
                    # ⚠ 只在**冻结策略刷新**这一拍重算 held，与训练侧 `apply_actions` 一致
                    # （训练里 held 也只在 `_physics_step % low_level_decimation == 0` 时重算，
                    # 两次刷新之间即使 delta 刚变也不改下发目标）。
                    joint_targets = loco_joint_targets + upper_delta
                    robot.set_joint_position_target(joint_targets[:, asset_to_policy])
            state = apply_rope_and_resistance(wheel_damping=args.wheel_damping)
            scene.write_data_to_sim()
            # ⚠️ 物理步进**必须** `render=False`。`SimulationContext.step()` 的 render 默认是 True，
            # 那条分支走的是 `self._app.update()`：物理由 app/帧时序驱动，**一帧可能推进多个物理
            # tick**。实测（20261009T111331Z，--record-every 1）第一次调用就让 base 掉了 7 mm、
            # 小腿关节转了 0.1 rad —— 等效 dt ≈ 50 ms 而不是 5 ms。于是「每 4 次调用 = 一个
            # 20 ms 控制周期」这个前提整体失效，按 200 Hz 标定的 PD（kp 25 / kd 0.5，腿链惯量
            # ~6e-4 kg·m²）直接发散：关节正负交替打到 ±23.7 N·m、被推到自己限位（髋 ±0.523、
            # 小腿 −3.0），机器人在最初几十毫秒内塌掉（绳/杆与车斗接触都是**后果**）。
            # Isaac Lab 的 `ManagerBasedRLEnv` 物理步进一律 `sim.step(render=False)`，
            # 渲染单独放在 `sim_step_counter % render_interval == 0` 那一拍（`sim.render()`）。
            sim.step(render=False)
            scene.update(dt)
            if not args.headless and (step + 1) % render_interval == 0:
                sim.render()
            # 进度：800 环境 × 2200 步的长跑里，逐 case 的判读输出只在**全部步进结束之后**才打印，
            # 没有这条的话整段运行是黑盒（看不出在走、也估不出还要多久）。
            if (step + 1) % progress_every == 0 or step + 1 == args.schedule.total_steps:
                elapsed = time.time() - started
                rate = (step + 1) / max(1e-9, elapsed)
                eta = (args.schedule.total_steps - step - 1) / max(1e-9, rate)
                heights = robot.data.root_pos_w[:, 2]
                print(f"[progress] {step + 1}/{args.schedule.total_steps} 步（{phase}）"
                      f"用时 {elapsed:.0f}s、{rate:.1f} 步/s、ETA {eta:.0f}s；"
                      f"base z 均值 {float(heights.mean()):.3f} m / 最低 "
                      f"{float(heights.min()):.3f} m", flush=True)
            if step % args.record_every:
                continue
            joint_err = (robot.data.joint_pos[:, policy_to_asset] - joint_targets)
            err_rms = joint_err.pow(2).mean(dim=1).sqrt().tolist()
            err_max = joint_err.abs().amax(dim=1).tolist()
            commands = command_tensor.tolist()
            lane = (robot.data.root_pos_w - origins)[:, 1].tolist()
            quat = robot.data.root_quat_w
            qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
            headings = torch.atan2(2.0 * (qw * qz + qx * qy),
                                   1.0 - 2.0 * (qy * qy + qz * qz)).tolist()
            body_vy = robot.data.root_lin_vel_b[:, 1].tolist()
            body_wz = robot.data.root_ang_vel_b[:, 2].tolist()
            load_lane = (cart.data.root_pos_w - origins)[:, 1].tolist()
            deck_fx = robot_contact_fx(deck_sensor).tolist()
            wheel_fx = sum(robot_contact_fx(sensor) for sensor in wheel_sensors).tolist()
            # lane 坐标（一次张量运算，避免逐 env 做设备同步）
            relative_robot = robot.data.root_pos_w - origins
            relative_load = cart.data.root_pos_w - origins
            progress_robot = (relative_robot * tangent_t).sum(dim=1)
            progress_load = (relative_load * tangent_t).sum(dim=1)
            # 离面高度 = lane 系 z − 局部剖面高度（逐 env 的坡度量级参与；0° lane 恒 0）
            height_robot = (relative_robot * normal_t).sum(dim=1)
            height_load = (relative_load * normal_t).sum(dim=1)
            if float(grade_deg.max()) > 0.0:
                height_robot = height_robot - profile_height_tensor(grade_deg, progress_robot)
                height_load = height_load - profile_height_tensor(grade_deg, progress_load)
            height_robot = height_robot.tolist()
            height_load = height_load.tolist()
            pitch_offsets = pitch_offsets_for(progress_robot).tolist()
            progress_robot = progress_robot.tolist()
            progress_load = progress_load.tolist()
            for env_index in range(num_envs):
                case_rows[env_index].append(make_row(
                    env_index, phase=phase, step=step,
                    command=commands[env_index][0],
                    joint_targets=joint_targets,
                    joint_err_rms=err_rms[env_index], joint_err_max=err_max[env_index],
                    progress_robot=progress_robot, progress_load=progress_load,
                    robot_height=height_robot, load_height=height_load,
                    pitch_offsets=pitch_offsets,
                    command_vy=commands[env_index][1],
                    command_wz=commands[env_index][2],
                    lane_offset=lane[env_index], lane_heading=headings[env_index],
                    body_vy=body_vy[env_index], body_wz=body_wz[env_index],
                    load_offset=load_lane[env_index],
                    deck_fx=deck_fx[env_index], wheel_fx=wheel_fx[env_index]))
        elapsed = time.time() - started
        print(f"[run] 完成 {args.schedule.total_steps} 步 × {num_envs} 环境，用时 {elapsed:.1f} s",
              flush=True)

        # ------------------------------------------------------------ 逐 env 指标
        def write_case_artifacts(case, rows):
            """把一个 case 的原始轨迹 + config 落盘（`--write-csv` 的策略见调用处）。"""
            case_dir = output / case.slug
            case_dir.mkdir(parents=True, exist_ok=True)
            write_json(case_dir / "config.json", {
                "case": case.to_dict(), "schedule": args.schedule.to_dict(),
                "lane_frame": {"tangent": [1.0, 0.0, 0.0], "normal": [0.0, 0.0, 1.0],
                               "gravity_mps2": list(gravity)},
                "connection": experiment["connection"],
                "ground_friction": args.ground_friction,
                "wheel_damping_nms_per_rad": args.wheel_damping,
                "user_command_mps": case.velocity_mps,
                "policy_joint_names": list(policy_cfg.joint_names),
                "thresholds": thresholds,
            })
            with (case_dir / "tow.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=TEST_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            case_dirs.append(case.slug)

        case_summaries = []
        case_dirs = []
        compact_detail_printed = 0
        # 判读阶段的进度节拍：~20 行，够看出在走（每 100 个 case 一条的话，800 环境会有近 2 分钟
        # 完全没输出，实测被误判成「卡住」）。
        case_progress_every = max(1, len(cases) // 20)
        if args.compact_log:
            print(f"[cases] 开始逐 case 判读：{len(cases)} 个；明细最多打 "
                  f"{args.compact_log_detail} 条（`--compact-log-detail` 可调），"
                  f"之后每 {case_progress_every} 个 case 打一条带判定码计数的进度", flush=True)
        for env_index, case in enumerate(cases):
            rows = case_rows[env_index]
            spec = specs[env_index]
            # 原始轨迹**先落盘**：判读指标再出问题（例如 `summarize_tow` 在坏姿态下拿不到
            # 几何间隙）也不能把这一轮的证据一起丢掉。2026-10-09 首跑正是指标抛 TypeError，
            # 9 个 case 的 tow.csv 与摘要一个都没写出来，只能重跑。
            csv_written = False
            if args.write_csv == "all":
                write_case_artifacts(case, rows)
                csv_written = True
            tw_summary = summarize_tow(
                rows, user_command=case.velocity_mps, joint_names=policy_cfg.joint_names,
                config={
                    "rope": {"model": case.connection,
                             "rest_length_m": case.length_m,
                             "stiffness_n_per_m": spec["stiffness"],
                             "damping_ns_per_m": spec["damping"],
                             "initial_slack_m": case.length_m - float(spec["initial_distance"]),
                             "position_gain": float(action_cfg.rope_position_gain),
                             "max_correction_rate_mps": float(action_cfg.rope_max_correction_rate)},
                    "cart_model": cart_model,
                    "cart_mass_actual_kg": actual_masses[env_index],
                    "robot_mass_kg": float(robot_mass_kg[env_index]),
                    "dt_s": dt,
                })
            metrics = compute_case_metrics(
                rows, case=case.to_dict(), schedule=args.schedule, record_dt=record_dt,
                tow_summary=tw_summary, thresholds=thresholds,
                joint_names=list(policy_cfg.joint_names), torque_limits=torque_limits,
                transition_window_s=args.transition_window, lane_keeping=args.lane_keeping,
                lane_vy_limit=args.lane_vy_limit, lane_wz_limit=args.lane_wz_limit,
                cart_present=case.cart_present,
                impact_window_s=args.impact_window, steady_margin_s=args.steady_margin_s,
                takeup_force_threshold_n=args.takeup_force_threshold,
                count_jnt=args.count_jnt)
            if not case.cart_present:
                # `summarize_tow` 的间隙/接触都是「机器人 ↔ 小车」的量；小车停在 2 m 外时它们
                # 只是"很远"而不是"很好"，所以显式标注并清空，避免报告里出现误导性的读数。
                tw_summary["cart_present"] = False
                tw_summary["note_no_cart"] = ("无小车 env（--no-cart-fraction）：间隙/接触类字段不适用，"
                                              "已在测试台侧中性化")
                for key in ("min_clearance_m", "min_clearance_coast_m", "min_clearance_station_m",
                            "final_clearance_m", "clearance_at_stop_m", "contact",
                            "time_to_contact_after_stop_s"):
                    tw_summary[key] = None
            metrics["summarize_tow"] = tw_summary
            summary = {
                "case": case.to_dict(),
                "labels": {"grade_deg": case.grade_deg, "velocity_mps": case.velocity_mps,
                           "connection": case.connection,
                           "cart_mass_kg": actual_masses[env_index]},
                "schedule": args.schedule.to_dict(),
                "lane_frame": {"tangent": [1.0, 0.0, 0.0], "normal": [0.0, 0.0, 1.0],
                               "gravity_mps2": list(gravity)},
                "metrics": metrics,
                "verdict": metrics["verdict"],
                "summarize_tow_valid": tw_summary.get("valid"),
                "summarize_tow_note": SUMMARIZE_TOW_NOTE,
            }
            case_summaries.append(summary)
            write_json(summaries_dir / f"{case.slug}.json", summary)
            should_write = (not csv_written and args.write_csv == "failed"
                            and metrics["verdict"]["code"] != "OK")
            if should_write:
                write_case_artifacts(case, rows)
            verdict = metrics["verdict"]
            # 逐 case 判读行：800 环境完整网格会刷 800 行（每行还带 5 个指标）⇒ 提供与训练脚本
            # 同名的 `--compact-log`：只打非 OK 的行 + 每 100 个 case 一条进度。分项指标仍然逐 env
            # 完整写进 `summaries/<case>.json` 与 `report.*`，只是不再往终端倒。
            if not args.compact_log:
                show_detail = True
            else:
                show_detail = (verdict["code"] != "OK"
                               and compact_detail_printed < args.compact_log_detail)
                compact_detail_printed += int(show_detail)
            if show_detail:
                print(f"[case] env{case.env_index:04d} c{case.column:02d}r{case.row:02d} "
                      f"g{case.grade_deg:g} v{case.velocity_mps:g} {case.connection} "
                      f"L{case.length_m:.2f} m{actual_masses[env_index]:g}kg ⇒ {verdict['code']}"
                      f"{(' [' + ','.join(verdict['reasons']) + ']') if verdict['reasons'] else ''}"
                      f"  起步关节RMS={_fmt(metrics['startup']['joint_rms_rad'])} "
                      f"跟速MAE={_fmt(metrics['speed'].get('mae_mps'))} "
                      f"滑移={_fmt(metrics['stop'].get('cart_coast_distance_m'))} "
                      f"停车最小间隙={_fmt(metrics['stop'].get('min_clearance_coast_m'))} "
                      f"横向|y|max={_fmt(metrics['lane'].get('y_max_abs_m'))}", flush=True)
            if (args.compact_log and args.compact_log_detail > 0
                    and compact_detail_printed == args.compact_log_detail
                    and env_index + 1 == args.compact_log_detail):
                print(f"[cases] 明细已达上限 {args.compact_log_detail} 条 —— 其余只计数与落盘"
                      f"（只增不减的进度见下；逐 env 全量指标在 summaries/ 与 report.*）", flush=True)
            if args.compact_log and (env_index + 1) % case_progress_every == 0:
                # 进度里带判定码计数：明细被截断时也看得出整体分布（全量仍在 summaries/ 与 report.*）
                tally = Counter(summary["verdict"]["code"] for summary in case_summaries)
                elapsed = time.time() - started
                rate = (env_index + 1) / max(1e-9, elapsed)
                print(f"[cases] 判读 {env_index + 1}/{len(cases)}"
                      f"（{elapsed:.0f}s、{rate:.1f} 个/s、ETA "
                      f"{(len(cases) - env_index - 1) / max(1e-9, rate):.0f}s）：" +
                      "，".join(f"{code} {count}" for code, count in tally.most_common()),
                      flush=True)

        # ------------------------------------------------------------ 报告
        grades = sorted({case.grade_deg for case in cases})
        groups = group_statistics(case_summaries)
        conclusion = necessity_conclusion(groups, thresholds)
        # 「策略 vs 基线」：本轮 + 同版本基线（--compare-report）+ 归档基线参照。
        # 三项都用同一个 `case_metric_summary`（口径、字段、样本计数完全一致）。
        comparison_current = case_metric_summary(case_summaries)
        comparison_reference = None
        if args.compare_report is not None:
            payload = json.loads(args.compare_report.read_text(encoding="utf-8"))
            comparison_reference = case_metric_summary(payload["cases"])
            print(f"[cases] 同版本基线对照：{args.compare_report}"
                  f"（{comparison_reference['envs']} 个 case，通过 "
                  f"{comparison_reference['ok']}/{comparison_reference['envs']}）", flush=True)
        # 冲击窗口 vs 稳态：与「策略 vs 基线」同一套逐 case 中位数汇总（本轮 / 同版本基线）。
        # 归档那列的 `has_impact_fields` 是**算出来的**（不是硬写 False）：万一以后有人把带
        # `impact.*` 的一轮归档进来，报告与 JSON 会自动改口径。
        archived_has_impact = any(
            key.startswith("impact.") for key in (ARCHIVED_BASELINE_2026_10_09.get("medians") or {}))
        archived_impact = dict(ARCHIVED_BASELINE_2026_10_09)
        archived_impact["has_impact_fields"] = archived_has_impact
        impact_lines = format_impact_section(comparison_current,
                                            reference=comparison_reference,
                                            archived=archived_impact)
        impact_report = {
            "window_s": args.impact_window,
            "steady_margin_s": args.steady_margin_s,
            "takeup_force_threshold_n": args.takeup_force_threshold,
            "steady_definition": "牵引段（tow）去掉首尾各 steady_margin_s 后的关节跟踪误差 RMS",
            "startup_definition": "tow 段首行起 [t0, t0 + window_s]",
            "stop_definition": "coast 段首行（指令归零）起 [t0, t0 + window_s]",
            "takeup_definition": "tow 段首个 |rope_tension_n| > takeup_force_threshold_n 的样本 ± window_s",
            "stop_contact_definition": "coast 段首个接触见证样本 ± window_s（车斗接触力或负载速度跃变）",
            "current": {"medians": comparison_current.get("impact_medians"),
                        "samples": comparison_current.get("impact_samples"),
                        "modal_worst_joint": comparison_current.get("impact_modal_worst_joint"),
                        "window_s": comparison_current.get("impact_window_s"),
                        "steady_margin_s": comparison_current.get("impact_steady_margin_s"),
                        "takeup_force_threshold_n":
                            comparison_current.get("impact_takeup_force_threshold_n")},
            "reference": (None if comparison_reference is None else
                          {"medians": comparison_reference.get("impact_medians"),
                           "samples": comparison_reference.get("impact_samples"),
                           "modal_worst_joint":
                               comparison_reference.get("impact_modal_worst_joint"),
                           "window_s": comparison_reference.get("impact_window_s"),
                           "steady_margin_s": comparison_reference.get("impact_steady_margin_s"),
                           "takeup_force_threshold_n":
                               comparison_reference.get("impact_takeup_force_threshold_n")}),
            "reference_report": (str(args.compare_report)
                                 if args.compare_report is not None else None),
            "archived": {"label": ARCHIVED_BASELINE_2026_10_09["label"],
                         "report": ARCHIVED_BASELINE_2026_10_09["report"],
                         "has_impact_fields": archived_has_impact,
                         "note": "归档基线早于本统计，没有 impact.* 字段；只能用既有 "
                                 "startup.*/stop.* 做同量级 sanity check"},
            "note": "新增观测字段，不参与判定码（classify_case 只读既有 startup/stop）；"
                    "逐 case 的 impact.* 在 report.json.cases[].metrics.impact 与 "
                    "summaries/<case>.json；逐窗口的比值 rms_over_steady 与增量 rms_delta_rad "
                    "是同一 case 内先算再取中位数",
        }
        comparison_lines = format_baseline_comparison(
            comparison_current, reference=comparison_reference, archived=archived_impact,
            upper_enabled=args.upper_checkpoint is not None,
            count_jnt=args.count_jnt)
        baseline_comparison = {"current": comparison_current,
                               "reference": comparison_reference,
                               "reference_report": (str(args.compare_report)
                                                    if args.compare_report is not None else None),
                               "archived": ARCHIVED_BASELINE_2026_10_09,
                               # 本轮开关状态 + 上层策略的动作/契约口径（v3 双头，TOW-26）：
                               # 13 = 1 维 vx 偏移 + 12 维关节残差；关时策略轮不存在。
                               "upper_policy_enabled": args.upper_checkpoint is not None,
                               "upper_contract_version": UPPER_CONTRACT_VERSION,
                               "policy_action_dim": UPPER_ACTION_DIM,
                               "policy_action_dim_note": "上层策略的动作维数（13 = "
                                                         "1 维 vx 偏移 + 12 维关节残差）；"
                                                         "基线轮不加载上层策略（见 "
                                                         "upper_policy_enabled）",
                               # JNT 移出判定统计量后的离线重判读数（本次变更的依据与验收基线）
                               "jnt_excluded_recount": JNT_EXCLUDED_RECOUNT_2026_10_10,
                               "count_jnt": bool(args.count_jnt),
                               "note": "三项同口径（同网格/同指标/同阈值）；归档基线是旧几何"
                                       "**且是旧口径**（JNT 计入判定），只作参照，不可逐格硬比"}
        labels = {name: getattr(args, name) for name in
                  ("num_envs", "velocities", "cart_masses", "ground_friction", "wheel_damping",
                   "command_shaping", "ramp_time_s", "record_every", "write_csv", "lane_keeping",
                   "lane_kp_y", "lane_kd_y", "lane_kp_yaw", "lane_kd_yaw",
                   "lane_vy_limit", "lane_wz_limit", "upper_checkpoint", "upper_stochastic",
                   "compare_report", "impact_window", "steady_margin_s",
                   "takeup_force_threshold", "count_jnt")}
        report = {
            "question": experiment["question"],
            "scene": experiment["scene"], "thresholds": thresholds,
            "arguments": experiment["arguments"],
            "upper_policy": experiment["upper_policy"],
            "baseline_comparison": baseline_comparison,
            "impact_statistics": impact_report,
            "groups": groups, "conclusion": conclusion,
            "cases": case_summaries,
            "csv_cases": case_dirs,
            "elapsed_s": time.time() - started,
            "git": git,
        }
        write_json(output / "report.json", report)
        csv_rows = [case_report_row(summary["metrics"]) for summary in case_summaries]
        if csv_rows:
            with (output / "report.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
                writer.writeheader()
                writer.writerows(csv_rows)
        (output / "report.md").write_text(
            build_markdown_report(case_summaries=case_summaries, groups=groups,
                                  conclusion=conclusion, args_dict=labels,
                                  thresholds=thresholds, schedule=args.schedule,
                                  grades=grades, git=git, comparison=comparison_lines,
                                  impact=impact_lines),
            encoding="utf-8")

        print("\n" + format_verdict_matrix(case_summaries, grades=grades))
        print("[conclusion] 上层任务是否有必要（本网格 + 本阈值）")
        for line in conclusion["lines"]:
            print(line)
        print(f"\n[DONE] {len(case_summaries)} env，用时 {time.time() - started:.1f} s ⇒ {output}",
              flush=True)
        experiment.update(state="completed", env_count=len(case_summaries),
                          csv_cases=case_dirs, elapsed_s=time.time() - started)
        write_json(output / "experiment.json", experiment)
        sys.stdout.flush()
        sys.stderr.flush()
        # 与 tow_drag.py 同一约定：不依赖 Kit 的关停路径（它会吞掉异常与退出码）。
        os._exit(0)
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001
        experiment.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        try:
            write_json(output / "experiment.json", experiment)
        except OSError:
            pass
        print(f"[FAILED] {type(exc).__name__}: {exc}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(2)


def _finite_only(value):
    """把非有限浮点递归换成 `None`。

    `json.dumps(..., allow_nan=False)` 会因为**一个** NaN 让整轮 225 个 case 的产物
    一起写不出来（长跑最后一步失败最亏）。指标里 NaN 只该出现在「窗口为空」这类退化情形，
    换成 `None` 后仍然看得见（`available` 字段会说明），不会连累其它 case。
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _finite_only(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_only(item) for item in value]
    return value


def _fmt(value) -> str:
    """打印用：None/NaN 显示 n/a。"""
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(value) else f"{value:.3f}"


def wheel_omega_fields(joint_vel, joint_ids) -> dict:
    """四轮角速度列（列名只在这里生成，离线测试直接调用核对）。"""
    return {f"wheel_{leg}_omega_radps": float(joint_vel[joint_id])
            for leg, joint_id in zip(WHEEL_LEGS, joint_ids)}


def joint_state_fields(*, pos, target, torque) -> dict:
    """12 关节的「实测角 / 当拍目标角 / 实际力矩」三组列（策略顺序，逐腿 FL/FR/RL/RR）。"""
    fields = {}
    for index, name in enumerate(recording.ROBOT_JOINT_POSITION_FIELDS):
        fields[name] = float(pos[index])
        fields[f"robot_jt_{index:02d}"] = float(target[index])
        fields[f"robot_tau_{index:02d}"] = float(torque[index])
    return fields


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
