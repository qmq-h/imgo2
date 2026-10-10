# 拖曳上层任务必要性：冻结策略基线测试（训练场景）

- 生成时间：2026-10-10T08:45:27Z
- git：`b650d553801d7e0dcb8bed66a3f8b785017c0aa9`（worktree ?? imgo2_rl/nohup.out）
- 场景：**训练场景** `UpperTowingSceneCfg`（`connection_grid` 40 列 × 20 行；800 环境一次跑完）
- 工作条件：速度档 [0.5, 1.0, 1.5, 2.0] m/s × 质量档 [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0] kg，**确定性轮转**（`slot = row + column`：质量 `slot % 8`、速度每 8 个 slot 换一档）；连接/长度/坡度量级由 cell 决定，不可扫
- 判据阈值里的质量档与训练随机范围 [5.0, 30.0] kg 不同：超出范围的档位属于外推检查
- 地面摩擦固定 0.8（训练随机 0.4–1.2）；轮轴阻尼 0.032 N·m·s/rad（训练随机 0.008–0.032）
- 指令整形：direct
- 度量坐标系固定为 lane 系（出生在剖面平地段 ⇒ 切向 +x、法向 +z、重力竖直）：`progress` = x 行程（**沿坡面的水平投影**，不是弧长）；`surface_height` = `z − profile_height(本 env 坡度量级, x)`；`pitch_rel` = 世界系俯仰 + 局部坡度
- 每 case：station 1.00 s + tow 5.00 s + coast 5.00 s，共 2200 物理步 （dt=0.005 s，记录每 5 步一行）
- 冲击窗口独立统计：`--impact-window` 0.2 s、稳态去首尾 `--steady-margin-s` 1 s、绷直阈值 `--takeup-force-threshold` 1 N（只进 `metrics.impact` 与本节，**不改判据/判定码**）
- 阈值：关节 RMS ≤ 0.1 rad / 单关节 ≤ 0.3 rad、跟速 MAE ≤ 20% 指令、停车几何间隙 > 0.1 m、跌倒 base 高 < 0.15 m 或 |pitch| > 0.8 rad 的样本 > 20%
- 判定口径：**JNT 不计入**（默认；用户 2026-10-10 决定）——`startup.*`/`stop.*` 的关节跟踪误差只作**观测**（下面「策略 vs 基线」的 `起步/停车关节响应 RMS` 中位数行仍在），不产生失败原因；要复现旧口径（归档对齐/追溯）加 `--count-jnt`

判定码：`OK` 通过；`LOW` 停车余量低；`LAT` 横向/朝向保持超限；`SPD` 跟速超限；`COL` 追尾接触；`FALL` 跌倒；`INV` 记录不可用；`JNT` 关节响应超限（**默认不计入判定**，仅 `--count-jnt` 时评估）。

横向/朝向保持：PD（kp_y=1、kd_y=0.3、kp_yaw=1.5、kd_yaw=0.3；vy 限幅 0.4 m/s、wz 限幅 0.8 rad/s）；判据 |y| ≤ 0.3 m 且 |yaw| ≤ 10°（**前进速度不参与 PD，只给指令**）
上层网络：**关闭**（基线；12 维关节位置残差恒 0，JNT 参考 = 冻结策略当拍输出）

## 逐坡度量级判定矩阵

### 坡度量级 平地（400 env）

| 质量 kg | 0.5 m/s compliant | 0.5 m/s inextensible | 0.5 m/s rigid | 1 m/s compliant | 1 m/s inextensible | 1 m/s rigid | 1.5 m/s compliant | 1.5 m/s inextensible | 1.5 m/s rigid | 2 m/s compliant | 2 m/s inextensible | 2 m/s rigid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | OK×1 | OK×4 | OK×3 | COL×8 | - | OK×1 | COL×8 | OK×1 | COL×8 | COL×3 | COL×4 | COL×8 |
| 10 | LOW×2 | OK×4 | OK×2 | COL×8 | - | COL×2 | OK×8 | COL×2 | COL×8 | COL×2 | COL×4 | COL×8 |
| 15 | COL×3 | OK×4 | OK×1 | COL×8 | - | OK×3 | SPD×8 | COL×3 | COL×8 | SPD×1 | COL×4 | COL×8 |
| 20 | COL×4 | OK×4 | - | COL×8 | - | OK×4 | COL×8 | COL×4 | SPD×8 | - | COL×4 | COL×8 |
| 25 | COL×5 | OK×3 | - | COL×8 | - | SPD×5 | COL×7 | COL×4 | SPD×8 | - | COL×4 | COL×7 |
| 30 | COL×6 | OK×2 | - | COL×8 | - | SPD×6 | COL×6 | COL×4 | SPD×8 | - | COL×4 | SPD×6 |
| 35 | COL×7 | OK×1 | - | COL×8 | - | SPD×7 | COL×5 | COL×4 | SPD×8 | - | COL×4 | SPD×5 |
| 40 | COL×8 | - | - | COL×8 | - | SPD×8 | COL×4 | COL×4 | SPD×8 | - | COL×4 | SPD×4 |

### 坡度量级 5° 坡（200 env）

| 质量 kg | 0.5 m/s compliant | 0.5 m/s inextensible | 0.5 m/s rigid | 1 m/s compliant | 1 m/s inextensible | 1 m/s rigid | 1.5 m/s compliant | 1.5 m/s inextensible | 1.5 m/s rigid | 2 m/s compliant | 2 m/s inextensible | 2 m/s rigid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | OK×4 | OK×2 | COL×4 | COL×3 | COL×2 | COL×4 | - | COL×1 | - | COL×4 | - | OK×1 |
| 10 | OK×4 | OK×2 | OK×4 | COL×2 | COL×2 | COL×4 | - | - | - | COL×4 | - | COL×2 |
| 15 | OK×4 | OK×2 | OK×4 | COL×1 | COL×2 | OK×4 | - | - | - | COL×4 | - | COL×3 |
| 20 | OK×4 | OK×2 | OK×4 | - | OK×2 | OK×4 | - | - | - | COL×4 | - | SPD×4 |
| 25 | OK×4 | OK×2 | OK×4 | - | COL×2 | SPD×3 | COL×1 | - | - | COL×4 | COL×1 | COL×4 |
| 30 | OK×4 | OK×2 | OK×4 | - | OK×2 | SPD×2 | COL×2 | - | - | COL×4 | COL×2 | SPD×4 |
| 35 | OK×4 | OK×2 | OK×4 | - | SPD×2 | SPD×1 | COL×3 | - | - | COL×4 | COL×2 | SPD×4 |
| 40 | OK×4 | OK×2 | OK×4 | - | SPD×2 | - | COL×4 | - | - | COL×4 | COL×2 | SPD×4 |

### 坡度量级 10° 坡（200 env）

| 质量 kg | 0.5 m/s compliant | 0.5 m/s inextensible | 0.5 m/s rigid | 1 m/s compliant | 1 m/s inextensible | 1 m/s rigid | 1.5 m/s compliant | 1.5 m/s inextensible | 1.5 m/s rigid | 2 m/s compliant | 2 m/s inextensible | 2 m/s rigid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | OK×3 | - | - | COL×4 | COL×2 | COL×4 | COL×4 | COL×2 | COL×4 | - | COL×2 | COL×1 |
| 10 | COL×4 | - | - | COL×4 | COL×2 | COL×4 | COL×4 | SPD×2 | COL×4 | - | COL×2 | - |
| 15 | COL×4 | - | OK×1 | COL×4 | SPD×2 | SPD×4 | COL×3 | COL×2 | COL×4 | - | SPD×1 | - |
| 20 | COL×4 | - | OK×2 | COL×4 | SPD×2 | SPD×4 | COL×2 | SPD×2 | SPD×4 | - | - | - |
| 25 | COL×4 | - | OK×3 | COL×4 | SPD×2 | SPD×4 | SPD×1 | SPD×2 | SPD×4 | - | - | - |
| 30 | COL×4 | - | OK×4 | COL×4 | SPD×2 | SPD×4 | - | SPD×2 | SPD×4 | - | - | - |
| 35 | COL×4 | COL×1 | OK×4 | COL×4 | SPD×2 | SPD×4 | - | SPD×2 | SPD×3 | SPD×1 | - | - |
| 40 | COL×4 | COL×2 | OK×4 | COL×4 | SPD×2 | SPD×4 | - | SPD×2 | SPD×2 | SPD×2 | - | - |


## 分组统计（按坡度量级）

- **平地**：119/400 通过；失败模式：speed_track_error×211, stop_collision×161, stop_margin_low×23
- **5° 坡**：101/200 通过；失败模式：speed_track_error×73, stop_collision×68, stop_margin_low×2
- **10° 坡**：42/200 通过；失败模式：speed_track_error×124, stop_collision×59, stop_margin_low×13

## 冲击窗口 vs 稳态（关节响应）

窗口参数：W = 0.2 s；稳态 margin = 1 s；绷直阈值 = 1 N。

**读法**：`joint_rms_rad` = 窗口内 12 关节跟踪误差 `q − q*` 的合并 RMS [rad]；`RMS/稳态` 是同一 case 先取比值再取中位数（>1 = 比稳态猛，≈1 = 与稳态同量级）；`RMS−稳态` 是差值 [rad]；`力矩饱和%` = `|τ| > 0.95·limit` 的步占比 ×100。四行分别对齐「起拖 / 绷直 / 指令归零 / 停车撞击」四个**不同**时刻（归零与撞击可能差数百毫秒），另有稳态一行作分母。**这些是新增的观测字段，不参与判定码**（`classify_case` 只读既有 startup/stop）。

| 时刻（对齐） | 关节 RMS [rad] | 最差关节（众数） | 力矩饱和% | RMS/稳态 | RMS−稳态 [rad] | 样本 n |
|---|---|---|---|---|---|---|
| 起拖（tow 首行） `t0 → +W` | 0.290（n=800） | RR_shank_joint | 0.000（n=800） | 1.323（n=800） | 0.069（n=800） | 800 |
| 绷直/穿绳（首个 ‖F‖ 越阈） `t ± W` | 0.207（n=800） | RL_shank_joint | 0.000（n=800） | 1.021（n=800） | 0.004（n=800） | 800 |
| 指令归零（coast 首行） `t0 → +W` | 0.171（n=800） | FR_thigh_joint | 0.000（n=800） | 0.880（n=800） | -0.024（n=800） | 800 |
| 停车撞击（首个接触见证） `t ± W` | 0.159（n=288） | RL_shank_joint | 0.000（n=288） | 0.812（n=288） | -0.040（n=288） | 288 |
| 稳态（牵引段去首尾 margin） `去首尾 margin` | 0.200（n=800） | RL_shank_joint | 0.000（n=800） | 1.000（分母） | 0.000（分母） | 800 |

- `最差关节（众数）` 是本轮各 case `impact.<时刻>.worst_joint` 里出现最多的那个；逐 case 的 `worst_joint` 与逐关节 RMS（`per_joint_rms_rad`）在 `report.json`／`summaries/<case>.json` 里，本表只给跨 case 汇总，免得表格随关节名变宽。
- `绷直` 一行的 `n` 是**有绷直样本**的 case 数：绳/连杆张力全程不越阈（例如 rigid 压缩、compliant 未拉直）时该 case 不进这一行的中位数（在 report.json 里是 `available=false` + `note`）。`停车撞击` 一行同理（没有接触见证的 case 不计入）。
- 既有 `startup.*` / `stop.*`（`--transition-window` 默认 1.0 s）**语义未动**，与本表并存：前者是归档对照口径，本表是缩短到 `--impact-window` 的冲击窗口。

**与基线对照（逐 case 中位数，同口径）**

| 时刻 | 本轮 关节 RMS [rad] | 本轮 RMS/稳态 | 本轮 RMS−稳态 [rad] |
|---|---|---|---|
| 起拖（tow 首行） | 0.290（n=800） | 1.323（n=800） | 0.069（n=800） |
| 绷直/穿绳（首个 ‖F‖ 越阈） | 0.207（n=800） | 1.021（n=800） | 0.004（n=800） |
| 指令归零（coast 首行） | 0.171（n=800） | 0.880（n=800） | -0.024（n=800） |
| 停车撞击（首个接触见证） | 0.159（n=288） | 0.812（n=288） | -0.040（n=288） |
| 稳态（牵引段去首尾 margin） | 0.200（n=800） | 1.000（分母） | 0.000（分母） |

同版本基线：未提供 `--compare-report`（冲击窗口的有效对照同样要**同一版本**跑两轮：一轮基线 + 一轮开开关，参数完全相同）。

归档基线参照（归档基线 2026-10-09（旧几何，仅参照））：该归档没有 `impact.*` 字段（本统计晚于它），只能用既有 `startup.*` / `stop.*` 做同量级 sanity check。


## 结论（任务是否有必要）

- 平地：119/400 通过；失败模式 speed_track_error×211, stop_collision×161, stop_margin_low×23。失败跨起步/全程/停车多相：脚本 shaping 不足以解释，属于上层残差/调度的目标。
- 5° 坡：101/200 通过；失败模式 speed_track_error×73, stop_collision×68, stop_margin_low×2。失败跨起步/全程/停车多相：脚本 shaping 不足以解释，属于上层残差/调度的目标。
- 10° 坡：42/200 通过；失败模式 speed_track_error×124, stop_collision×59, stop_margin_low×13。失败跨起步/全程/停车多相：脚本 shaping 不足以解释，属于上层残差/调度的目标。

## 策略 vs 基线（同网格、同指标、同阈值）

判定口径：**JNT 不计入**（默认；用户 2026-10-10 决定）——`startup_joint_error`/`stop_joint_error` 只作观测，下面两个关节响应 RMS 中位数行仍在，但不产生失败原因
**本轮**（开关**关闭**（基线，残差恒 0））：
- 通过 **262/800**；判定码分布（每 case 取最严重项）：COL 288、OK 262、SPD 223、LOW 27
- 五项指标中位数：起步关节响应 RMS 0.237（n=800）；全程跟速 MAE 0.225（n=800）；停车小车滑移 0.206（n=800）；停车最小几何间隙 0.361（n=688）；停车关节响应 RMS 0.159（n=800）
- 逐坡度量级通过数：grade0 119/400；grade10 42/200；grade5 101/200

**同版本基线**：未提供 `--compare-report`。有效对照必须**同一版本**跑两轮（一轮基线 + 一轮开开关，参数完全相同），把基线那轮的 `report.json` 用 `--compare-report` 传进来；下面那列归档基线只是参照。

**归档基线参照**（归档基线 2026-10-09（旧几何，仅参照），`docs/towingdata/2026-10-09_necessity_800_noload/report.json`；git `1bbb405`，**工作树脏**，复现请用 6fac89a，`--no-cart-fraction 0.125`）：
- 通过 **0/800**；判定码分布（每 case 取最严重项）：JNT 395、COL 276、SPD 129
- 五项指标中位数：起步关节响应 RMS 0.216（n=800）；全程跟速 MAE 0.161（n=800）；停车小车滑移 0.090（n=800）；停车最小几何间隙 0.490（n=691）；停车关节响应 RMS 0.154（n=800）
- ⚠ **不可逐格硬比**：旧几何：绳 0.6–1.2 m（未按类型解耦）、出生比 0.5；当前代码是绳 0.5–1.5 / 杆 0.5–1.0、出生比 0.8 ⇒ 不可逐格硬比，只能同版本内对照。
- ⚠ **口径不可比（JNT）**：归档的通过数（0/800）与判定码（JNT 395 / COL 276 / SPD 129）是按**旧口径**（JNT 计入判定）算的；2026-10-10 起默认把 JNT 移出判定统计量 ⇒ 两边的通过数与判定码分布**不可比**，要逐格对齐旧口径请用 `--count-jnt` 重跑。
- **新口径重判（同一批 800 cell，仅把 JNT 剔除统计量）**：基线 **356/800**、策略 **436/800**（旧口径下两轮都是 0/800）；剩余失败原因 基线 stop_collision 295 / speed_track_error 234 / stop_margin_low 28 → 策略 stop_collision 283 / speed_track_error 66 / stop_margin_low 36 / lane_deviation 1。来源 `imgo2_rl/logs/towing/play_test/upper_switch_{baseline,policy}/report.json`（git `df593f4`）；逐坡度/连接/负载与依据见 README TOW-23。

**`JNT` 已移出判定统计量（用户 2026-10-10 决定）**：归档基线里 JNT 在有负载（295/700）与无负载（100/100）两边都接近 100%，量的是 q − q* 的 PD 静差（≈ τ/kp），阈值未标定。因此默认口径下 `JNT` 不再出现为失败原因，验收读 `COL`/`SPD`/`LOW`（以及 `LAT`/`FALL`）的占比变化。另注意开了开关后 `JNT` 的参考量变成 `held = 冻结输出 + 残差`（训练侧 `reference="commanded"`），与基线的「冻结输出」**不是同一个量**，这一项的差不能当改善看（只有 `--count-jnt` 追溯旧口径时才相关）。


## 限制

- 只覆盖本网格与本阈值：连接类型/长度/坡度量级由训练网格决定，不能扫；未测训练侧的域随机化（摩擦 0.4–1.2、轮轴阻尼 0.008–0.032、12.5% 无小车锚点）、未测 breakaway/Coulomb 阻力、未验真机、未做跨环境隔离。
- 质量档 20/25 kg 超出训练采样范围 [5, 15] kg：那两档是外推检查，不能当作「训练分布内基线够不够」的证据。
- 工作条件轮转用的是 `slot = row + column`（不是 env 序号）：列数 40 是质量档数的整数倍，用序号轮转会让质量只由列决定、与连接类型/坡度量级混淆。当前分配下`(连接 × 质量)` 与 `(坡度量级 × 质量)` 严格均衡，但**每个 cell 仍然只落到一种 (速度, 质量) 组合**（800 env = 800 cell），所以判读要按分档聚合，不能当逐 cell 的完整响应面。
- 跟速一律用**体系** vx（与冻结策略观测同口径）；`summarize_tow` 的 `steady_tracking_ratio` 是世界系口径，坡上不要混用。
- `progress` 是 x 行程（水平投影），不是坡面弧长：10° 剖面上两者差 < 1%。
- **JNT 已按用户决定（2026-10-10）移出判定统计量**：`startup_joint_error` /`stop_joint_error` 不再产生失败原因，`JNT` 从判定码分布里消失（要复现旧口径加 `--count-jnt`，逐位一致）。**理由与实测分布**：这两条在 800 cell 上基线 800/800、策略 800/800，零区分度；它们量的是 `q − q*` 的底层 PD 静差（≈ τ/kp，扛体重），不是上层任务的职责 —— `joint_rms_limit_rad=0.10` 连最好的 case 都超 1.6 倍（实测基线 startup RMS P1=0.161 / P50=0.221、stop RMS P1=0.126 / P50=0.154，800/800 全部超限）。两条既有观测字段（`startup.*` / `stop.*` 的 `joint_rms_rad`、`joint_max_rad`、`worst_joint`、`per_joint_rms_rad`、`torque_saturated_frac`）与阈值 `--joint-rms-limit-rad` / `--joint-max-limit-rad` **照旧保留并在本节上方打印**（见「策略 vs 基线」的 `起步关节响应 RMS` / `停车关节响应 RMS` 中位数行），只是不再进判据。
- `JNT` 的参考量在两种开关下不是同一个东西（基线 = 冻结策略当拍输出、策略 = 下发的`held = 冻结目标 + 残差`），所以即使开 `--count-jnt`，两轮的 `JNT` 差也不能直接当改善读。归档对照（`docs/towingdata/2026-10-09_necessity_800*`）的通过数与判定码（0/800、`JNT` 395 等）是按**旧口径**算的，与本节口径**不可比**。
- summarize_tow.valid 是绳语义判据（要求 tow 段出现过 T>0）；rigid 连杆张力有符号，其失败项（如 rope_never_taut_during_tow）不代表工况失败。本脚本的 verdict 不依赖它。
- 冲击窗口统计（`metrics.impact.*`）是**观测字段，不进判定码**：`classify_case` 只读既有 `startup.*`/`stop.*`，`DEFAULT_THRESHOLDS` 也不含 `--impact-window`/`--steady-margin-s`/`--takeup-force-threshold` 任何一项；四行的样本数可能不同（无绷直/无接触的 case 不进那一行的中位数）。绷直判定用 `|rope_tension_n|`，rigid 连杆压缩（负张力）也算「穿绳」的充要见证。
- 冲击窗口的 `1.0 s` 归档对照列（`startup.*`/`stop.*`）**没有** `impact.*` 字段：本统计晚于归档，只能同版本跑两轮做有效对照。
- 仿真相位（PhysX 步进、绳力/轮阻施加、重力写入）只在训练机实跑验证；本脚本的离线测试只覆盖纯逻辑与记录字段契约。
