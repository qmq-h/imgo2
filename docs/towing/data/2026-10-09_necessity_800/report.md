# 拖曳上层任务必要性：冻结策略基线测试（训练场景）

- 生成时间：2026-10-09T11:50:06Z
- git：`3492e46f8fb3dc98f3eb5c3eb6061e4007e27682`（worktree ?? imgo2_rl/nohup.out）
- 场景：**训练场景** `UpperTowingSceneCfg`（`connection_grid` 40 列 × 20 行；800 环境一次跑完）
- 工作条件：速度档 [0.5, 1.0, 1.5] m/s × 质量档 [5.0, 10.0, 15.0, 20.0, 25.0] kg，**确定性轮转**（`slot = row + column`：质量 `slot % 5`、速度每 5 个 slot 换一档）；连接/长度/坡度量级由 cell 决定，不可扫
- 判据阈值里的质量档与训练随机范围 [5.0, 15.0] kg 不同：超出范围的档位属于外推检查
- 地面摩擦固定 0.8（训练随机 0.4–1.2）；轮轴阻尼 0.032 N·m·s/rad（训练随机 0.008–0.032）
- 指令整形：direct
- 度量坐标系固定为 lane 系（出生在剖面平地段 ⇒ 切向 +x、法向 +z、重力竖直）：`progress` = x 行程（**沿坡面的水平投影**，不是弧长）；`surface_height` = `z − profile_height(本 env 坡度量级, x)`；`pitch_rel` = 世界系俯仰 + 局部坡度
- 每 case：station 1.00 s + tow 5.00 s + coast 5.00 s，共 2200 物理步 （dt=0.005 s，记录每 5 步一行）
- 阈值：关节 RMS ≤ 0.1 rad / 单关节 ≤ 0.3 rad、跟速 MAE ≤ 20% 指令、停车几何间隙 > 0.1 m、跌倒 base 高 < 0.15 m 或 |pitch| > 0.8 rad 的样本 > 20%

判定码：`OK` 通过；`LOW` 停车余量低；`LAT` 横向/朝向保持超限；`JNT` 关节响应超限；`SPD` 跟速超限；`COL` 追尾接触；`FALL` 跌倒；`INV` 记录不可用。

横向/朝向保持：PD（kp_y=1、kd_y=0.3、kp_yaw=1.5、kd_yaw=0.3；vy 限幅 0.4 m/s、wz 限幅 0.8 rad/s）；判据 |y| ≤ 0.3 m 且 |yaw| ≤ 10°（**前进速度不参与 PD，只给指令**）

## 逐坡度量级判定矩阵

### 坡度量级 平地（400 env）

| 质量 kg | 0.5 m/s compliant | 0.5 m/s inextensible | 0.5 m/s rigid | 1 m/s compliant | 1 m/s inextensible | 1 m/s rigid | 1.5 m/s compliant | 1.5 m/s inextensible | 1.5 m/s rigid |
|---|---|---|---|---|---|---|---|---|---|
| 5 | JNT×9 | JNT×4 | COL×13 | COL×13 | COL×8 | COL×8 | COL×10 | COL×4 | COL×11 |
| 10 | JNT×10 | JNT×5 | JNT×12 | COL×13 | COL×7 | COL×8 | JNT×9 | JNT×4 | COL×12 |
| 15 | JNT×11 | JNT×6 | JNT×11 | COL×13 | COL×6 | JNT×8 | COL×8 | COL×4 | COL×13 |
| 20 | JNT×12 | JNT×7 | JNT×10 | COL×12 | COL×5 | JNT×9 | COL×8 | COL×4 | SPD×13 |
| 25 | JNT×13 | JNT×8 | JNT×9 | COL×11 | COL×4 | SPD×10 | COL×8 | COL×4 | SPD×13 |

### 坡度量级 5° 坡（200 env）

| 质量 kg | 0.5 m/s compliant | 0.5 m/s inextensible | 0.5 m/s rigid | 1 m/s compliant | 1 m/s inextensible | 1 m/s rigid | 1.5 m/s compliant | 1.5 m/s inextensible | 1.5 m/s rigid |
|---|---|---|---|---|---|---|---|---|---|
| 5 | JNT×4 | JNT×4 | COL×6 | COL×5 | COL×2 | COL×4 | COL×7 | COL×2 | COL×6 |
| 10 | JNT×4 | JNT×4 | JNT×5 | COL×6 | COL×2 | COL×4 | COL×6 | COL×2 | COL×7 |
| 15 | JNT×4 | JNT×4 | JNT×4 | COL×7 | COL×2 | JNT×4 | COL×5 | COL×2 | COL×8 |
| 20 | JNT×4 | JNT×3 | JNT×4 | COL×8 | COL×2 | JNT×4 | COL×4 | SPD×3 | SPD×8 |
| 25 | JNT×4 | JNT×2 | JNT×4 | COL×8 | COL×2 | SPD×5 | COL×4 | SPD×4 | SPD×7 |

### 坡度量级 10° 坡（200 env）

| 质量 kg | 0.5 m/s compliant | 0.5 m/s inextensible | 0.5 m/s rigid | 1 m/s compliant | 1 m/s inextensible | 1 m/s rigid | 1.5 m/s compliant | 1.5 m/s inextensible | 1.5 m/s rigid |
|---|---|---|---|---|---|---|---|---|---|
| 5 | JNT×5 | JNT×2 | COL×4 | COL×7 | COL×2 | COL×6 | COL×4 | COL×4 | COL×6 |
| 10 | JNT×6 | JNT×2 | JNT×4 | COL×6 | COL×2 | COL×7 | COL×4 | COL×4 | COL×5 |
| 15 | JNT×7 | JNT×2 | JNT×4 | COL×5 | COL×2 | COL×8 | COL×4 | COL×4 | COL×4 |
| 20 | COL×8 | JNT×2 | JNT×4 | COL×4 | SPD×3 | SPD×8 | SPD×4 | SPD×3 | SPD×4 |
| 25 | COL×8 | JNT×2 | JNT×5 | SPD×4 | COL×4 | SPD×7 | SPD×4 | SPD×2 | SPD×4 |


## 分组统计（按坡度量级）

- **平地**：0/400 通过；失败模式：startup_joint_error×400, stop_joint_error×400, stop_collision×146, speed_track_error×75, stop_margin_low×23
- **5° 坡**：0/200 通过；失败模式：startup_joint_error×200, stop_joint_error×200, stop_collision×88, speed_track_error×62, stop_margin_low×5
- **10° 坡**：0/200 通过；失败模式：startup_joint_error×200, stop_joint_error×200, speed_track_error×118, stop_collision×81, stop_margin_low×4

## 结论（任务是否有必要）

- 平地：0/400 通过；失败模式 startup_joint_error×400, stop_joint_error×400, stop_collision×146, speed_track_error×75, stop_margin_low×23。失败跨起步/全程/停车多相：脚本 shaping 不足以解释，属于上层残差/调度的目标。
- 5° 坡：0/200 通过；失败模式 startup_joint_error×200, stop_joint_error×200, stop_collision×88, speed_track_error×62, stop_margin_low×5。失败跨起步/全程/停车多相：脚本 shaping 不足以解释，属于上层残差/调度的目标。
- 10° 坡：0/200 通过；失败模式 startup_joint_error×200, stop_joint_error×200, speed_track_error×118, stop_collision×81, stop_margin_low×4。失败跨起步/全程/停车多相：脚本 shaping 不足以解释，属于上层残差/调度的目标。

## 限制

- 只覆盖本网格与本阈值：连接类型/长度/坡度量级由训练网格决定，不能扫；未测训练侧的域随机化（摩擦 0.4–1.2、轮轴阻尼 0.008–0.032、12.5% 无小车锚点）、未测 breakaway/Coulomb 阻力、未验真机、未做跨环境隔离。
- 质量档 20/25 kg 超出训练采样范围 [5, 15] kg：那两档是外推检查，不能当作「训练分布内基线够不够」的证据。
- 工作条件轮转用的是 `slot = row + column`（不是 env 序号）：列数 40 是质量档数的整数倍，用序号轮转会让质量只由列决定、与连接类型/坡度量级混淆。当前分配下`(连接 × 质量)` 与 `(坡度量级 × 质量)` 严格均衡，但**每个 cell 仍然只落到一种 (速度, 质量) 组合**（800 env = 800 cell），所以判读要按分档聚合，不能当逐 cell 的完整响应面。
- 跟速一律用**体系** vx（与冻结策略观测同口径）；`summarize_tow` 的 `steady_tracking_ratio` 是世界系口径，坡上不要混用。
- `progress` 是 x 行程（水平投影），不是坡面弧长：10° 剖面上两者差 < 1%。
- 关节跟踪误差阈值没有标定，首轮结果出来前不要把 `JNT` 当成定论。
- summarize_tow.valid 是绳语义判据（要求 tow 段出现过 T>0）；rigid 连杆张力有符号，其失败项（如 rope_never_taut_during_tow）不代表工况失败。本脚本的 verdict 不依赖它。
- 仿真相位（PhysX 步进、绳力/轮阻施加、重力写入）只在训练机实跑验证；本脚本的离线测试只覆盖纯逻辑与记录字段契约。
