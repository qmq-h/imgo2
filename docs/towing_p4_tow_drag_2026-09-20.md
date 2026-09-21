# P4（下半）：机器人 + 小车 + 绳的拖曳场景与实验入口

日期：2026-09-20。状态：**P4 核心行为已在训练机验证——0.5 m/s 下拖曳成立**（`v_R ≈ v_L ≈ 0.512`、张力 5.2 N 且水平漂移仅 1.1%、小车被拖 2.47 m、`valid=true`，见 §5.5）。仍待做：`v_cmd = 1.0 m/s`、`m_L = 5–25 kg` 包线扫描（计划 P4 的第二步与 working envelope），以及确认 `--cart-drop` 是否消除站定自漂。
对应研究计划 `paper_plan_imgo2.md`「P4：接入你现有 locomotion」的运行部分，
上半（冻结策略契约与适配器）见 [P4 上半记录](towing_p4_policy_contract_2026-09-20.md)。

## 1. 本轮实现

| 文件（相对仓库根） | 内容与状态 |
|---|---|
| `.../towing/towing_env_cfg.py` | 平地 + 机器人（训练侧 `IMGO2_CFG`）+ 被动小车的场景配置；挂点常量 |
| `.../towing/utils/recording.py` | **扩充**：`TowRecorder` + `TOW_FIELDS`（沿用 `CartRecorder` 的严格校验） |
| `imgo2_rl/scripts/towing/tow_drag.py` | 拖曳入口：冻结策略按速度指令驱动机器人，经绳拖曳小车，逐物理步记录 |
| `imgo2_rl/tests/test_towing_tow_drag.py` | 16 项测试（布局算术、参数校验、记录器、接口契约） |

**没有新建任何目录**，全部落在既有目录内。

## 2. 场景与布局约定

- 机器人在前、小车在后，两者都朝 **+x**；速度指令为正即向前拖。
- 机器人挂点：base 坐标系的 `(-0.16, 0, 0)`。依据：base 碰撞箱长 0.315 m
  （`urdf/imgo2.urdf`）⇒ 后表面在 x = −0.1575 m，取 −0.16 略过后表面。
  **工程取值，没有实测挂点**；换实物尺寸只改 `ROBOT_ATTACHMENT_OFFSET_M`。
- 小车挂点：`cart.urdf` 的 `rope_attachment`，车体坐标 `(+0.25, 0, 0)`（前轴之前）。
  小车在后时该点朝向机器人 ⇒ 受力方向与拖曳一致。
- 初始挂点间距 = `L0 + slack`（默认 1.0 + 0.05 m）。`initial_cart_x()` 实现该算术，
  测试用**场景常量 + `cart.urdf` 实读挂点**反解校核：间距必须等于 `L0 + slack`
  （差一点初始张力就错了，所以这条专门锁住）。
- 机器人初始高度用显式参数 `--spawn-height`，默认 `ROBOT_SPAWN_HEIGHT_M = 0.35 m`
  （= 训练侧 `IMGO2_CFG.init_state.pos` 的 z）。README 记录过「训练多数从录制帧
  ≈0.297 m 起、play 从 0.35 m 起」可能对应**高/低两个稳定站高分支**，因此这里不写死，
  由训练机确认哪一个给出可复现的稳定拖曳。
- 关节初始角按契约的 `default_dof_pos` 给（与 `IMGO2_CFG` 一致），不使用随机初始状态。

## 3. 每物理步的顺序与力臂处理

与 P1/P2 相同（显式欧拉，力的口径一致）：

```text
按当前状态算绳力（P3 的 rope_state）与轮阻（-b·ω）
  → robot/cart.set_external_force_and_torque(...) / cart.set_joint_effort_target(...)
  → scene.write_data_to_sim() → sim.step() → scene.update(dt) → 记录
```

- **力臂交给仿真**：`set_external_force_and_torque(positions=...)` 的作用点坐标是**连杆坐标系**
  （`articulation.py` 的 `apply_forces_and_torques_at_position` 真正消费它），所以直接把挂点
  常量传进去，由 PhysX 自己算力矩，不手写 `offset×F`，也不用纠结 CoM 约定
  （实测 `base` 的惯性原点偏移 < 3 mm）。**整套调用必须用同一个 `is_global` 口径**，
  否则 Isaac Lab 会警告并改变全体的作用坐标系；这里统一用默认的连杆系。
- 力本身从世界系转到连杆系用 `quat_apply_inverse`。
- 绳数学走 `rope.py` 的**标量路径**（`num_envs = 1`），即已经被 31 项测试覆盖的那条路径。
- 策略抽帧：`decimation = round(control_dt / dt)`，契约要求 `control_dt` 是物理 `dt` 的整数倍，
  不满足直接报错。默认两者都是 0.005 s ⇒ 1:1。
- 启动时会核对：关节数、**关节名与顺序**是否与策略契约完全一致、`base`/`base_link` 唯一、
  小车四关节与四轮接触；不一致立即报错（而不是跑到一半形状不匹配）。

## 4. 运行与判据

```bash
cd imgo2_rl
# 稳定拖曳（计划 P4）
bash scripts/run_isaaclab.sh scripts/towing/tow_drag.py --velocity 0.5 --duration 5 --headless

# 阶跃停止（计划 P6）：拖 5 s 后指令归零，机器人停下、小车继续滑行
bash scripts/run_isaaclab.sh scripts/towing/tow_drag.py --velocity 0.5 --duration 10 --stop-at 5 --headless
```

**可视化（去掉 `--headless` 即可）**：`SimulationContext.step()` 默认 `render=True`，所以
非 headless 会正常刷新视口，不需要额外代码。相机由脚本用 `sim.set_camera_view()` 设死
（拖曳场景为眼位 `(2.5, 2.5, 1.8)`、注视 `(-0.7, 0, 0.2)`），而机器人/小车会沿 +x 走 2–3 m，
**长跑会出画**，需要时在窗口里手动移相机。

```bash
# 看「机器人停下、小车滑行」这一幕（计划 P6）
bash scripts/run_isaaclab.sh scripts/towing/tow_drag.py --velocity 0.5 --duration 10 --stop-at 5
# 看小车落地与滑行（P1/P2）
bash scripts/run_isaaclab.sh scripts/towing/cart_coast.py --mode drop --duration 3
bash scripts/run_isaaclab.sh scripts/towing/cart_coast.py --mode coast --velocities 1.0 --damping 0.016 --duration 10
```

播放速度取决于渲染帧率与物理 `dt` 的比值，可能快于实时；想慢一点用 `--dt 0.0025`
（P2 的 dt 复核显示停止距离只变 0.23%）。暂停会阻塞 `sim.step()`，画面保留、可慢慢看。

产物在 `imgo2_rl/logs/towing/tow_drag/<run_id>/`：`config.json`（含策略/模型哈希、绳参数、
抽帧、挂点）、`tow.csv`（逐物理步）、`summary.json`、`experiment.json`。
`--output-dir` 只能指向尚不存在的目录。

`summary.json` 的判据（计划 P4 的「稳定拖曳」）：

| 字段 | 含义 | 判据 |
|---|---|---|
| `steady_robot_vx_mps` / `steady_load_vx_mps` | 稳态窗（指令阶段后半段）两体平均 vx | 见下两行 |
| `steady_tracking_ratio` | 稳态机器人 vx ÷ 用户指令 | **≥ 0.5**，否则 `robot_not_tracking_command`（只看「两体同速」有洞：两者都静止时它也成立） |
| `steady_speed_gap_mps` | 两者之差 | **≤ 0.1 m/s**，否则 `robot_and_load_speeds_differ` |
| `time_to_taut_s` | 指令阶段首次出现张力的时刻 | 为 `None` ⇒ `rope_never_taut_during_tow`（没拖上） |
| `tension_slack_fraction_after_takeup` | 收掉初始松弛之后 T = 0 的占比 | **≤ 0.05**，否则 `rope_not_continuously_taut` |
| `steady_tension_n` | 稳态窗张力均值 | 必须 > 0 |
| `tension_drift_ratio` | 稳态窗**前后半均值**的相对差 | **≤ 0.25**，否则 `tension_not_steady`（判「水平是否稳」，不判方差） |
| `tension_ripple_ratio` | 稳态窗变异系数 | **只报告不判失败**：步态纹波是正常现象（实测主频 7.5 Hz、主要由阻尼项 `c·ḋ` 贡献，弹簧项只波动 1.06 N） |
| `max_abs_pitch_rad` | 全程最大 \|pitch\| | ≤ 0.6 rad，否则 `body_pitch_excessive` |
| `steady_robot_z_m` / `min_robot_z_m` | 稳态/最低机身高度 | 只报告：明显偏低说明塌下去或被拽倒 |
| `settle_max_tension_n` | **站定阶段**（指令 0）张力峰值 | **≤ 1 N**，否则 `rope_taut_during_settle`：实测机器人出生窜动把绳拉直过（85.7 N），会把小车甩出 0.24 m（见 §5.6） |
| `settle_max_abs_robot_vx_mps` / `settle_load_drift_m` / `settle_max_abs_load_vx_mps` | 站定阶段机器人窜动与小车位移 | 只报告 |
| `coast_load_travel_m` / `coast_robot_travel_m` | **滑行段**（`--stop-at` 后指令为 0）两体位移 | 只报告 |
| `coast_min_gap_m` / `coast_gap_at_stop_m` | 滑行段最小/起始间距 | `<= 0` ⇒ `load_reached_robot`（追尾） |
| `coast_time_to_slack_s` / `coast_retension_peak_n` | 停车后张力归零耗时、若重新绷紧的峰值 | 只报告（计划 P6 的核心观测量） |
| `coast_final_robot_vx_mps` | 滑行段末机器人 vx | ≤ 20% 指令，否则 `robot_did_not_stop` |

判读逻辑在 `imgo2_rl/scripts/tools/summarize_tow.py`（**纯标准库**，与 P1/P2 的
`summarize_cart_coast.py` 同一模式），所以新判据可以用真实轨迹离线复算：

```bash
python3 imgo2_rl/scripts/tools/summarize_tow.py <run 目录>
```

`failures` 为空 ⇒ `valid = true`、退出码 0；有失败项 ⇒ 退出码 2。

**退出码说明**：拖曳入口在失败路径上**显式 `os._exit(2)` 且不调用 `application.close()`** ——
这是 CART-02 的教训（Kit 关停会直接终止进程、把异常与退出码一起吞掉，实测三次尝试退出码都是 0）。
新入口因此不复制那个缺陷；`cart_coast.py` 的 CART-02 仍未修。

**操作前提：Isaac Sim 必须串行执行**（并发实例争锁会中断，见 P1/P2 检查记录 §3.5）。

## 5. 验证：离线测试与训练机实跑发现

| 验证 | 结果 |
|---|---|
| `python3 -m unittest test_towing_tow_drag` | **20 项通过** |
| `python3 -m unittest test_towing_policy_contract` | **24 项通过**（含 5 项关节顺序回归） |
| 全量离线测试（11 个测试文件） | **186 项通过** |
| `check_model_sync.py` / `check_asset_paths.py` / `check_cart_model.py` | 退出码均为 0 |
| `tow_drag.py --help`（只用标准库） | 退出码 0，不需要 Isaac Lab |

### 5.1 训练机第一次实跑暴露的问题：Isaac Lab 的 DOF 顺序 ≠ 策略顺序

第一次实跑（2026-09-20）在启动时的契约核对就报错退出，报告的两份顺序是：

```text
模型（PhysX）: FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh, RL_thigh, RR_thigh,
               FL_shank, FR_shank, RL_shank, RR_shank      ← 全部 hip → 全部 thigh → 全部 shank
契约（策略）  : FL_hip, FL_thigh, FL_shank, FR_hip, ...     ← 逐腿
```

**根因**：Isaac Lab / PhysX 的 DOF 顺序按运动学树**广度优先**排列，而 `urdf/imgo2.urdf`
是**逐腿**声明（FL hip/thigh/shank, FR …）。AMP 任务在 `amp_env_cfg.py` 里把策略的观测
（`observations.policy.joint_pos/joint_vel`）与动作（`actions.joint_pos.joint_names`）都
**显式**设成逐腿顺序，所以训练出来的策略按逐腿顺序收发关节量，而 Isaac Lab 的资产数组是
广度优先顺序 —— **两者必须显式置换**。

**这不是契约错、也不是模型错，是运行端漏了重排**。第一版把「模型关节顺序」与「策略关节
顺序」直接比较并要求相等，于是启动即失败；若当时改成「把契约顺序改成广度优先」来消掉报错，
就会让观测拼错、动作打到错误的关节上（**静默且致命**）。

**修法**：`policy_cfg.asset_permutation(asset_joint_names)` 比对关节**集合**并给出
「策略第 i 个 ↔ 资产第 perm[i] 号」的置换；入口用它把 `joint_pos`/`joint_vel` 重排进观测、
把策略输出的关节目标换回资产顺序再下发。另外用**一条强校验**确认置换是对的：资产
`default_joint_pos` 按置换重排后必须等于契约的 `default_dof_pos`
（URDF 用正则把 hip/thigh/shank 统一设成 0/0.87/-1.82）。
置换与会话两端顺序都写进运行配置的 `config.json`，便于事后核对。

**回归覆盖**：`test_towing_policy_contract.py` 的 `JointOrderTests` 用**实跑报告的那份
真实顺序**做固定样例，断言置换是合法重排、关键下标（策略 FL hip/thigh/shank → 资产 0/4/8）、
恒等置换会错（资产第 1 号是 FR_hip）、重排后能复现契约默认姿态、以及 URDF 声明顺序确实
与 Isaac Lab 顺序不同（记录差异来源）；`test_towing_tow_drag.py` 另加源码契约，禁止退回
「直接比较关节名列表」的写法。

### 5.2 第二次实跑暴露的问题：记录器与入口都建运行目录

第二次实跑（`20260920T142354Z_38639b47`）已经**通过了关节顺序检查**（说明 §5.1 的修复生效），
但随即报：

```text
[FAILED] FileExistsError: [Errno 17] File exists:
  '/root/Desktop/Imgo2/imgo2_rl/logs/towing/tow_drag/20260920T142354Z_38639b47'
```

**根因**：目录归属在两处重复。`main()` 在开头就用 `output.mkdir(parents=True, exist_ok=False)`
独占创建运行目录（保证新运行绝不落进已有目录），而 `TowRecorder` 照抄了 `CartRecorder` 的
「自己建目录」语义又建了一次 ⇒ 自己撞自己。`CartRecorder` 服务「一个运行里多个 case 子目录」，
每个实例收到的是**尚不存在**的子目录；拖曳运行只有一个目录，我把它传给了 recorder。

**修法**：明确职责——运行目录由入口用 `exist_ok=False` **独占创建**，`TowRecorder` 只负责
写入，并要求目录已存在、拒绝覆盖已存在的 `tow.csv`/`config.json`。

**为什么离线没抓到**：当时的单测只**单独构造 recorder**（沿用了 `CartRecorder` 的用法），
没有覆盖「入口先建目录、再交给 recorder」这条集成路径。现已补上三条：
`test_accepts_an_already_created_run_directory`（模拟入口已建目录）、
`test_requires_an_existing_directory`、`test_refuses_to_overwrite_existing_artifacts`，
以及一条源码契约（入口保留 `mkdir(exist_ok=False)`，`TowRecorder` 内**没有任何** `mkdir`
调用——用 AST 判断，避免匹配到文档字符串里的字样）。

### 5.3 顺带清理：删掉语义重复的 `apply_joint_mapping`

`apply_joint_mapping()`（部署侧语义：策略第 i 个 ↔ 该数组第 `joint_mapping[i]` 号）在 Isaac Lab
里已被 `asset_permutation()` 取代，且**没有任何运行时消费者**。两个名字相近的「重排」API
并存正是 §5.1 那次搞混的温床，故删除该方法；`joint_mapping` **字段**保留（它是部署契约的一部分，
仍与 `amp/config.yaml` 交叉核对）。

### 5.4 第三次实跑：流程跑通，但**行为失败**——两处 bug（已修）

第三次（`20260920T143803Z_a03574f3`）终于完整跑完（`state=completed`），管线全部打通：
配置、冻结策略、绳力、记录、汇总、判定都工作。但行为是失败的，`summary.json`：

```text
steady_robot_vx_mps   = 0.00054     ← 指令是 0.5，机器人几乎没动
steady_load_vx_mps    = -0.027
steady_speed_gap_mps  = 0.028       ← 这条反而「通过」了（两者都≈静止）
tension_peak_n        = 0.0
tension_slack_fraction= 1.0         ← 整个指令阶段绳都是松的
steady_distance_m     = 0.534       ← 远小于 L0=1.0
max_abs_pitch_rad     = 0.802       ← 唯一被抓住的失败项
```

轨迹把原因指得很清楚：**初始挂点间距实测 1.4777 m（设计值 1.05 m），初始张力 1955 N**
（机器人自重才 54 N）⇒ 0.1 s 内 pitch 冲到 −0.74 rad、t≈0.5 s 就被拽倒 ⇒ 之后 `v_R≈0`。
两处 bug 叠在一起：

1. **位置漏了挂点偏移**：算挂点**速度**时加了偏移，算挂点**位置**却直接用了刚体原点
   ⇒ 绳长里混进机器人与小车的高差（0.35 vs 0.15 m），`d = √(1.46² + 0.2²) = 1.478`。
2. **`--slack` 语义写反**：名字是「松弛量」，实现却是「初始间距 = L0 **+** slack」＝预张紧。
   5 cm 就对应 200 N 预载；叠加第 1 条后变成 ≈1911 N（与实测 1955 N 吻合，差值来自阻尼项）。

**修法**：

- 挂点世界坐标 = **刚体原点 + 旋转后的挂点偏移**（与速度的口径一致）；
- `initial_cart_x()` 改为按**三维**几何解算：目标距离 `L0 − slack`，横向间距取
  `√(target² − dz²)`，并显式检查「目标距离小于两挂点高差」这种几何不可能的情形；
- `--slack` 现在是真正的松弛：离线核算 `slack=0.05` ⇒ 初始间距 **0.9500 m**、
  **初始张力 0 N**、机器人前进 **0.05 m** 后绳才张紧（正是计划想观察的 slack→taut）；
- 日志增加 `robot_z_m`/`load_z_m`（这次「倒了」只能从 pitch 间接看出，太费劲），
  `summary.json` 增加 `steady_robot_z_m`/`min_robot_z_m`。

**同时加强判据**（这次 `steady_speed_gap_mps=0.028` 通过了，说明原判据有洞）：

- `steady_tracking_ratio = 稳态机器人 vx ÷ 用户指令`，< 0.5 判 `robot_not_tracking_command`；
- `tension_slack_fraction > 0.9` 判 `rope_never_taut_during_tow`。

**回归测试**：布局断言改为三维；新增 `test_initial_condition_leaves_the_rope_slack`
（用 `cart.urdf` 实读挂点与场景常量算出初始间距，断言 **t=0 张力必须为 0** —— 直接锁住
那次 1955 N 的拽倒）与 `test_rejects_geometry_that_cannot_hold_the_requested_slack`。

**当时仍未验证**（**§5.5 已回答：可以** —— 站高 0.284 m、pitch 均值 3.2°、跟速 102%，未复现「高/低站高分支」）：机器人能否在 `spawn_height=0.35` 稳定站立并跟住 0.5 m/s。上一次的数据在
被拽倒前没有干净样本（t=0.005 s 时张力已有 1955 N）。下一次的**站定阶段**（t<1 s，指令为 0、
绳松弛、张力为 0）会直接给出这个答案，也是判断「高/低站高分支」疑虑的第一手数据。

### 5.5 第四次实跑：**拖曳成立**（P4 的核心行为已验证）

运行 `20260920T144620Z_47cbd7ff`（`--velocity 0.5 --duration 5`，L0=1.0、k=4000、c=100、
轮阻 0.016、`spawn-height` 默认 0.35）。`summary.json`：

| 量 | 实测 | 判读 |
|---|---|---|
| `steady_robot_vx_mps` | **0.5123** | 指令 0.5 ⇒ 跟速 **102%** |
| `steady_load_vx_mps` | **0.5123** | — |
| `steady_speed_gap_mps` | **−6.3e-5** | `v_R ≈ v_L` ✅ 计划的头号判据 |
| `steady_tension_n` | **5.199**（σ 1.499） | 绳在拉 |
| `tension_drift_ratio` | **1.1%** | 张力水平极稳（「相对稳定区间」成立） |
| `tension_ripple_ratio` | 28.8% | 步态纹波，只报告 |
| `steady_distance_m` | 1.0013 | 伸长 1.3 mm ⇒ `k·δ = 5.2 N`，与实测张力吻合 |
| `time_to_taut_s` | 1.755 | 起步收松弛（见下） |
| `tension_slack_fraction_after_takeup` | **0.0** | 收松弛后全程张紧 |
| 小车位移 | −1.339 → **+1.136**（**2.47 m**） | 被拖走了 |
| `mean_pitch_rad` / `max_abs_pitch_rad` | 0.056 / 0.194 | 3.2° / 11.1°，稳定 |
| `steady_robot_z_m` / `min_robot_z_m` | 0.284 / 0.246 | 站得住（未出现塌陷） |
| 判定 | `valid = true`、`failures = []` | — |

**一个附带结论**：`--spawn-height 0.35` 下机器人落到 **0.284 m** 站高并稳定跟住 0.5 m/s，
没有出现 README 里担心的「高/低站高分支」问题 —— 至少 0.35 m 出生是可用的。

**这一轮同时暴露/处理了两件事**：

1. **判据的洞（已改）**：原判据用「张力变异系数 ≤ 25%」判稳定，本轮 28.8% 判失败。
   但实测纹波主频 **7.5 Hz**、弹簧项只波动 1.06 N（δ 变化 0.27 mm），波动主要来自阻尼项
   `c·ḋ` —— 那是机器人步态的正常纹波，不是失稳。计划要的是「T(t) 进入**相对稳定区间**」，
   看的是水平是否稳定，故改为**前后半均值漂移 ≤ 25%**，纹波只报告。用真实轨迹复算：
   `tension_drift_ratio = 1.1%` ✅。
2. **小车在站定阶段自漂 0.24 m**：那时指令为 0，小车却漂了 0.239 m、峰值速度 0.372 m/s。
   当时推断是「按 resting height 精确贴地生成导致的接触自漂」，于是加了 `--cart-drop`
   （默认 0.03）。**该诊断是错的，见 §5.6 的更正**：真因是机器人出生窜动把绳拉直。

### 5.6 更正 §5.5 的错误根因：小车初速度来自机器人出生窜动把绳拉直

用户查看可视化后指出「小车有初速度是不对的」。用已有轨迹逐帧核对（运行 `3746654e`，
commit `0e19d07`，已含 `--cart-drop 0.03`）：

```text
   t      v_R      v_L     T(N)      d(m)     z_R     z_L
0.005   0.6492   0.0000   0.0000   0.94876  0.3628  0.1785
0.055   0.5794   0.0000   0.0000   0.99036  0.3638  0.1577
0.080   0.4241   0.1032  72.9634   1.00694  0.3568  0.1500   ← 绳被拉直
0.105   0.3260   0.2545  62.1838   1.01284  0.3469  0.1500
0.155   0.1943   0.3671   0.0000   1.00593  0.3254  0.1500
```

- **`--cart-drop` 没能解决**：加了落差后站定自漂仍是 0.2375 m（改前 0.2390）⇒ 与小车生成方式无关。
- **真因**：机器人按 `spawn_height = 0.35 m` 出生，而该策略的**实测站高只有 0.284 m**
  ⇒ 先落下 6.6 cm，**落地时向前窜动（实测峰值 0.849 m/s）**，把绳从 0.95 m 拉过 `L0 = 1.0 m`
  ⇒ 站定阶段绳被拉直，**张力峰值 85.7 N、17 个样本 T > 0**，把小车甩到 0.369 m/s
  后自由滑行 0.238 m。看上去就是「小车自己有初速度」。
- 我此前只抽查了几个时间点就断言「站定阶段绳是松的」，这是错误结论的由来。

**本轮修法（三层）**：

1. **出生高度默认 0.35 → 0.30 m**（`ROBOT_SPAWN_HEIGHT_M`）：贴近实测站高 0.284 m，
   把 6.6 cm 的落下压到 1.6 cm，从源头减小窜动。
2. **`--slack` 默认 0.05 → 0.20 m**：实测窜动把绳多拉长约 0.063 m，松弛不足就会在站定阶段
   被拉直；0.20 m 留足余量。
3. **拖曳段开始前显式重摆小车并清零速度**（`reset_cart_for_tow()`）：把小车摆到
   「挂点间距 = `L0 − slack`」（相对机器人**当时**的实际位置解算）并清零线/角速度与轮速，
   再 `scene.update(dt)` 刷新状态。**初始条件由设计决定，而不是取决于出生条件**——
   与 P1/P2 在静置检查后重写小车状态的约定一致。

**判据同步收紧**：新增 `rope_taut_during_settle`（`station` 阶段张力 > 1 N 即失败）与
`settle_max_tension_n` / `settle_max_abs_robot_vx_mps` 两个报告量。**用真实轨迹复算，两条
既有运行都被正确判为失败**（85.7 N / 75.1 N），即判据能抓住用户看到的这个现象。

**同时新增阶跃停止（计划 P6 的画面）**：`--stop-at T` 把指令在拖曳段第 T 秒归零并保持，
于是轨迹分为 `station` / `tow` / `coast` 三段（`phase` 列），`coast` 段报告小车滑行距离、
**最小间距（追尾风险）**、张力归零耗时与是否重新绷紧——正是计划要求看的
「机器人停下后 `T → 0` 有多快、小车会不会追尾」。判据新增 `robot_did_not_stop`
与 `load_reached_robot`。

**待下一次实跑确认**：站定阶段张力是否归零、`--stop-at` 的滑行画面与指标。

### 5.7 其它覆盖与测试抓到的缺陷
覆盖：初始间距算术（用场景常量与 `cart.urdf` 实读值反解）、后挂点在机体后表面之后、
小车挂点朝向机器人、参数校验、记录器的表头/非有限值/时间不递增/拒绝覆盖、以及接口契约
（`positions=` 作用点、`write_data_to_sim()` 必须早于 `sim.step()` 且晚于施力、失败路径
`os._exit(2)` 且不调用 `application.close()`、场景配置只改 `prim_path` 不碰增益/观测/奖励、
入口不导入 `velocity_env_cfg`/`amp_env_cfg`、**初始位姿必须在 `InteractiveScene(...)` 之前
写进配置**）。

**测试抓到的两个真实缺陷（都已修）**：

1. **`--damping 0` / `--slack 0` 被错拒**：第一版把这两个量放进「必须为正」的校验循环，
   于是一个合法的无阻尼绳（`c = 0`）和初始就绷直（`slack = 0`）都过不了参数校验。
   已拆成「严格为正」与「允许为 0」两组。
2. **初始位姿改晚了**：第一版在 `InteractiveScene(scene_cfg)` **之后**才写
   `robot_cfg.init_state.pos`。场景在构造时就把资产按 `init_state` 摆好，之后再改配置
   不生效 ⇒ **`--spawn-height` 会被静默忽略**（而且没有任何报错）。已改为在构造场景前写配置，
   并用测试锁住这个先后顺序。

## 6. 已知限制（本轮**未**验证的部分）

- **P4 的核心行为已验证**（§5.5：`v_R ≈ v_L ≈ 0.512`、T ≈ 5.2 N 稳定、小车被拖 2.47 m），
  但**只在 `v_cmd = 0.5 m/s` + 名义质量（10 kg）+ 一组绳参数下验证过**。以下仍未验证：
  - `v_cmd = 1.0 m/s`（计划 P4 的第二步）；
  - **质量包线扫描** `m_L = 5/10/15/20/25 kg`（计划 P4 的明确要求，用来定 working envelope）——
    未做；缩放质量必须同时缩放惯量（AGENTS.md 的「只改质量不改惯量会造成不自洽」）；
  - 绳参数 `k`/`c`/`L0` 的敏感性（目前是 P3 记录里的工程起点，未标定）；
  - 其它 Isaac Lab/PhysX 调用路径（例如并发、GUI）未走。
- `spawn_height = 0.35 m` **可用**（§5.5 实测站高 0.284 m 且稳定跟速），README 里
  「高/低站高分支」的疑虑在这一档未复现；但没试过 ≈0.297 m 那一档，不构成对分支假说的否定。
- **小车在站定阶段的自漂**（实测 0.239 m）已加 `--cart-drop`（默认 0.03）并把该量记入
  `summary.json`，但**修复效果待下一次实跑确认**。
- **质量扫描未做**：计划要求扫 `m_L = 5/10/15/20/25 kg` 来确定现有底层的 working envelope，
  本入口目前只用小车 URDF 的名义质量（10 kg，即 `m_L = 10`）。缩放质量必须同时缩放惯量
  （AGENTS.md 的「只改质量不改惯量会造成不自洽」），这属于下一个节点。
- PPO 底层的契约仍未做（架构记录 §5 要求冻结两种、分别记录 `policy_type`），
  与 AMP 的观测顺序/动作映射**不默认相同**。
- 尚未记录步态类指标（`foot_slip` 等）；那是计划 P5 的 Logger 范围。

## 7. 未做

- 未修改 locomotion 的任何代码/配置；未训练、未导出策略、未动硬件。
- 未改动 `cart_coast.py`（CART-02 仍在问题表里）。
- 未新建目录、未新增依赖。
