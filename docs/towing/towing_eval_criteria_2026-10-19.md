# 拖曳验收口径收束（2026-10-19；含保留横向保持/关节静差）

> 用户第一轮原话：「目前似乎 towing 的训练结果并不是很好，但是我们首先要调整问题：
> 问题应该收束到，拉小车的情况下，速度跟随稳定性，启停（只考虑冲击，不需要刻意往前走
> 几步了）以及拉重车不翻车。按照这样调整我们的评估口径。」
>
> 第二轮补充：「横向保持和关节静差也还是要作为失败。」⇒ **最终口径 = 三条主问题 +
> 保留横向保持（`LAT`）/ 关节静差（`JNT`）两个失败项**。停车追尾接触与停车间隙仍只作诊断。
>
> 本文记录**口径本身、代码落点、离线验证与待训练机确认项**。这是一次**评测侧**变更：
> 不改训练奖励、不改终止条件、不改 checkpoint 契约、不重训。

## 1. 为什么改

2026-10-19 之前，`play_towing_test.py` 的 `verdict` 把「上层任务必要性」的五项指标
折成 `OK/SPD/COL/LOW/LAT/FALL/JNT`：

- `COL`：停车后小车追上机器人（接触）；
- `LOW`：停车几何间隙低于阈值；
- `LAT`：横向/朝向保持超限；
- `SPD`：全程跟速 MAE 超限（不分负载）；
- `JNT`：关节静差（2026-10-10 起默认不计入）。

最近两轮平坦实测（`model_1000` / `model_2000`，质量 10/20/30 kg、速度 0.5/1.0 m/s、
100 env、13 个无小车锚点）的旧口径读数是 **42/100** 与 **38/100**；失败几乎全是
`speed_track_error` 与 `stop_collision`，其中 `stop_collision` 是「机器人按指令停住、
小车靠惯性追上来」——这在小车物理上是正常的，旧口径却把它当失败，并要求机器人
停车后继续往前走几步来躲。第一轮因此把问题收束成三条，**停车接触/间隙、停车后位移
不再作为验收**；但用户随后明确 **横向保持与关节静差必须继续算失败**，所以最终保留这两项。

## 2. 新口径（`towing_task_2026-10-19_v2`）

逐 case 先按质量分档，再做五条检查：

| 检查 | 适用 | 通过条件 |
|---|---|---|
| `speed_tracking`（拉小车速度跟随稳定性） | 小车档（`cart_mass_kg <= 15 kg`）且拖车 | 稳态窗（牵引段后一半）的跟速 MAE ≤ 20% 指令、速度标准差 ≤ 10% 指令、\|误差\| p95 ≤ 40% 指令 |
| `lane_keeping`（横向/朝向保持，旧 `LAT`） | `--lane-keeping pd` 且拖车 | `y_max_abs_m ≤ 0.30 m` 且 `heading_max_abs_rad ≤ 10°` |
| `joint_static_error`（关节静差，旧 `JNT`） | 所有拖车 case | `startup` / `stop` 两窗口的 `joint_rms_rad ≤ 0.10 rad` 且 `joint_max_rad ≤ 0.30 rad` |
| `start_stop_impact`（启停只看冲击） | 所有拖车 case | `impact.startup/takeup/stop/stop_contact` 四个窗口中，可用的每个都满足 `RMS/稳态` ≤ 1.5、`RMS−稳态` ≤ 0.10 rad、力矩饱和步占比 ≤ 5% |
| `no_fall`（拉重车不翻车） | 所有拖车 case；重车档是主判据 | `stability.fell` 为假，且由 base 四元数解出的 `\|roll\|` ≤ 0.80 rad |

- **任务通过 = 全部适用检查通过**。小车档 = 速度 + 横向 + 关节 + 冲击 + 不跌倒；
  重车档 = 横向 + 关节 + 冲击 + 不翻车（速度只作观测）。
- 不适用的检查记 `N/A`，不算失败；无小车锚点（`cart_present=False`）整条记 `NA`，不进通过率。
- 判定码：`PASS / SPEED_UNSTABLE / JOINT_ERROR / LANE_DEVIATION / IMPACT_HIGH / FALL /
  INVALID / NA`，严重度 `NA < PASS < SPEED_UNSTABLE < JOINT_ERROR < LANE_DEVIATION <
  IMPACT_HIGH < FALL < INVALID`；一个 case 的 `reasons` 保留全部命中项
  （如 `speed_mae_high`、`speed_unstable`、`speed_dropout`、`lane_deviation`、
  `startup_joint_error`、`stop_joint_error`、`start_stop_impact`、`robot_fell`、
  `robot_rollover`、`invalid_record`）。

### 2.1 与旧判定码的映射

| 旧 | 新口径下的处理 |
|---|---|
| `COL` 停车追尾接触 | **不再是失败**。接触本身只作诊断；`impact.stop_contact` 的**冲击量**如果超限才进 `start_stop_impact`。 |
| `LOW` 停车间隙低 | **不再是失败**。`stop.clearance_*` / `min_clearance_coast_m` 保留观测。 |
| 停车后往前走几步 | **不再要求**。`stop.robot_travel_after_stop_m` 只作诊断；不参与任何检查。 |
| `LAT` 横向/朝向 | **恢复为失败**，进 `lane_keeping`；`--lane-keeping off` 时 N/A。 |
| `JNT` 关节静差 | **恢复为失败**，进 `joint_static_error`；与 `start_stop_impact` 分开：前者是 `q − q*` 的静差/跟踪误差，后者是相对稳态的冲击。 |
| `SPD` 全程跟速 | 只对**小车档**、且改用**稳态窗**判「跟随稳定性」；重车档的速度跟踪只作观测。 |
| `FALL` | 保留并加强：`fell` = 离面高度 <0.15 m **或** pitch >0.8 rad 样本占比 >20% **或** roll >0.8 rad 样本占比 >20%。 |

### 2.2 为什么是 15 kg 分档

训练侧质量随机范围是 **5–30 kg**，默认评测档是 `10/15/20/25/30 kg`。取
`light_max_kg = 15` 后：10/15 kg 为小车、20/25/30 kg 为重车；最近两轮实测用的
`10/20/30 kg` 也正好落成「10 小车、20/30 重车」。这个分界是**产品口径**，不是物理常数，
CLI 用 `--task-light-max-kg` 可改；改了会同时影响报告分档与通过率，需在产物里注明。

## 3. 代码落点

| 文件 | 变更 |
|---|---|
| [towing_task_criteria.py](../../imgo2_rl/scripts/towing/towing_task_criteria.py) | **新增**。纯标准库：`evaluate_task()` / `summarize_task()` / `task_conclusion_lines()` / `format_task_matrix()` / `_evaluate_report()`；`DEFAULT_TASK_THRESHOLDS` 与旧 `DEFAULT_THRESHOLDS` **分开存放**（旧阈值被归档对照依赖，字面量不动）。 |
| [play_towing_test.py](../../imgo2_rl/scripts/towing/play_towing_test.py) | `compute_case_metrics()` 增 `task_thresholds`，逐 case 写 `metrics["task"]`；`speed_track_stats()` 增 `steady_speed_std_mps` / `steady_abs_err_p95_mps` / `steady_frac_within_10pct`；`stability_stats()` 从 `robot_quat_*` 解 `max_abs_roll_rad`；`case_report_row()` 增 `task_*` 与 `max_abs_roll_rad` 列；`report.md` 增「任务口径（验收以此为准）」与任务矩阵，旧矩阵标「旧口径，仅追溯」；`report.json` 增 `criteria/task_thresholds/task_statistics/task_conclusion`；`experiment.json` 增 `criteria`；`[plan]` 打印任务口径；`--write-csv failed` 改为按**任务失败**留档；`--compact-log` 改为按任务码过滤。 |
| [towing_segment_report.py](../../imgo2_rl/scripts/tools/towing_segment_report.py) | 报表顶部注明「诊断，不是 pass/fail；停车接触/间隙不是失败判据」。 |
| [test_towing_task_criteria.py](../../imgo2_rl/tests/test_towing_task_criteria.py) | **新增** 51 项离线测试：分档、五条检查（含横向/关节）、严重度、无小车、阈值合并、汇总、矩阵、旧报告复判。 |
| [test_towing_play_test.py](../../imgo2_rl/tests/test_towing_play_test.py) | 新增 10 项接线测试：`metrics.task`、轻/重车判据、JNT 失败、阈值转发、CSV 列（含 `task_lane_ok`/`task_joint_ok`）、报告表/矩阵、分组计数。 |

### 3.1 新增 CLI

```
--task-light-max-kg                 小车/重车分界（默认 15 kg）
--speed-std-ratio-limit             稳态速度 std / 指令（默认 0.10）
--speed-p95-ratio-limit             稳态 |误差| p95 / 指令（默认 0.40）
--lane-y-limit-m                    横向 |y| 上限（沿用旧阈值，默认 0.30 m）
--lane-heading-limit-deg            朝向 |yaw| 上限（沿用旧阈值，默认 10°）
--joint-rms-limit-rad               关节静差 RMS 上限（沿用旧阈值，默认 0.10 rad）
--joint-max-limit-rad               关节静差单关节峰值上限（沿用旧阈值，默认 0.30 rad）
--impact-rms-ratio-limit            冲击窗口 RMS/稳态（默认 1.50）
--impact-rms-delta-limit-rad        冲击窗口 RMS−稳态（默认 0.10 rad）
--impact-torque-sat-frac-limit      冲击窗口力矩饱和步占比（默认 0.05）
--roll-limit-rad                    |roll| 上限（默认 0.80 rad）
```

旧 `--speed-mae-ratio-limit` 仍沿用为稳态 MAE 的比例阈值（默认 0.20）。

### 3.2 旧报告复判

不重跑仿真也能把旧 `report.json` 按新口径复判：

```bash
python3 imgo2_rl/scripts/towing/towing_task_criteria.py \
    imgo2_rl/logs/towing/play_test/<run>/report.json [<run2>/report.json ...] \
    [--task-light-max-kg 15] [--json /tmp/task_replay.json]
```

它只补 `metrics.task` 再汇总，**不写回原文件**。旧报告没有
`steady_speed_std_mps` / `steady_abs_err_p95_mps`，因此速度项只能按稳态 MAE 判，
std/p95 记 `None`（跳过），**不会退回到全程 `p95_abs_err_mps`**（那会把起步瞬态
算进「跟随稳定性」）。横向/关节两项的旧字段完整，可以完全复判。

## 4. 离线验证（本机）

- `python3 -m py_compile`：新增/改动的三个 Python 文件通过。
- `pytest imgo2_rl/tests -q` ⇒ **749 passed, 4 skipped, 1336 subtests**（基线 684；
  新增 51 项任务口径 + 10 项接线）。
- `play_towing_test.py --dry-run`（无 torch、无 Isaac Sim）退出 0，plan 打印任务口径与
  旧口径并存；`tests/test_towing_play_test.py` 的 DryRun/Cli 断言通过。
- `towing_task_criteria.py` 复判最近两轮平坦报告：

| run | 旧口径 | 新口径 | 小车 | 重车 | NA | 判定码 |
|---|---|---|---|---|---|---|
| `model1000_flat` | 42/100 | **0/87** | 0/29 | 0/58 | 13 | JOINT_ERROR 86、LANE_DEVIATION 1 |
| `model2000_flat` | 38/100 | **0/87** | 0/29 | 0/58 | 13 | JOINT_ERROR 86、LANE_DEVIATION 1 |

`NA` = 无小车锚点（`--no-cart-fraction 0.125` 的 13 个 env），不进通过率。

**关键读数**：恢复 `JNT` 后，旧报告的 **87 个拖车 case 全部**关节静差超限
（`startup_joint_error` 87、`stop_joint_error` 87；`startup` RMS 中位约 0.28 rad、
`stop` 约 0.20 rad，阈值 0.10/0.30 rad），另 1 个 case 横向保持超限。这不是新代码把
训练改坏了，而是**旧 JNT 阈值从未按新口径标定**：它一直是 800/800，用户现在要求它继续
算失败，所以旧 run 在新口径下必然几乎全灭。下一步要么按分布重标 `--joint-rms-limit-rad`
（例如先看 P50/P90），要么把「降低关节静差」正式作为训练目标；在标定前不要把 0/87
读成「策略变差」。

速度项仍是**部分口径**：旧报告没有稳态 std/p95，只按稳态 MAE 判（29/29 通过）；
要判「小车速度跟随稳定性」，必须在训练机用**新代码**重跑一轮，拿到
`steady_speed_std_mps` 与 `steady_abs_err_p95_mps`。

补充观察：旧报告里的**全程** `p95_abs_err_mps` 在 1.0 m/s × 10 kg 的 15 个 case 上约
**0.40–0.51 m/s**（旧 `SPD` 的一部分来源）；它包含起步瞬态，不能直接当稳态稳定性，
但提示重跑后 `speed_dropout` 有可能真的触发，应先看新产物再标定 `--speed-p95-ratio-limit`。

## 5. 限制与待办

### 已修（离线验证）

- 五条检查、质量分档、任务 verdict、报告/JSON/CSV 落盘、plan 说明、旧报告复判。
- roll 通道从既有 base 四元数解出，不改仿真、不改记录格式。
- 旧 `verdict`/`DEFAULT_THRESHOLDS` 字面量未动；归档对照仍可按旧口径追溯。

### 未验证 / 待训练机

1. **新代码实跑一轮**（必须）：旧报告缺稳态 std/p95，速度项仍是部分口径
   （横向/关节两项的旧字段完整，0/87 已可判）；当前 0/87 的主因是未标定的 JNT 阈值。
   建议先跑最近两轮同条件（平坦、100 env、质量 10/20/30、速度 0.5/1.0）：
   ```bash
   bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/towing/play_towing_test.py \
       --headless --columns 0-19 --num-envs 100 \
       --velocities 0.5 1.0 --cart-masses 10 20 30 --no-cart-fraction 0.125 \
       --tow-duration 10 --coast-duration 5 --write-csv none \
       --upper-checkpoint <run>/model_1000.pt
   ```
   看 `report.md` 的「任务口径」节与 `report.json.task_statistics`。**注意**：这条命令
   仍按旧报告的 100 env 口径；正式全网格去掉 `--columns/--num-envs` 即可。
2. **阈值标定（最优先是 JNT）**：`joint_rms_limit_rad=0.10` /
   `joint_max_limit_rad=0.30`（旧 JNT 口径）在最近两轮是 **87/87 全超限**
   （startup RMS 中位 0.28、stop 0.20），是当前 0/87 的主因；先看新产物的 P50/P90，
   再决定是「重标阈值」还是「把关节静差降下来当训练目标」。
   其余任务阈值 `speed_std_ratio_limit=0.10`、`speed_p95_ratio_limit=0.40`、
   `lane_y_limit_m=0.30`、`lane_heading_limit_deg=10`、
   `impact_rms_ratio_limit=1.50`、`impact_rms_delta_limit_rad=0.10`、
   `impact_torque_sat_frac_limit=0.05`、`roll_limit_rad=0.80` 也都是**未标定占位值**。
   标定前不要把「通过/失败」当最终结论。
3. **impact 阈值可能偏松**：最近两轮旧报告的 `impact.startup/stop` 的 `RMS/稳态`
   约 0.7–1.06、力矩饱和 0，全部低于 1.5/5% ⇒ 新口径的「启停冲击」当前没有区分度。
   重跑后若仍全过，应把阈值收紧到实测分布的 p90/p95，或增加绳力/力变化率冲击量。
4. **重车不翻车的 roll 判据**：只在新代码重跑后才有真实 `max_abs_roll_rad`；
   旧报告/合成轨迹四元数为 0 时该子项为 `None`（跳过），不会误判。
5. **训练奖励已按 2026-10-19 第二轮要求改**（见
   [towing_reward_2026-10-19.md](towing_reward_2026-10-19.md)）：`clearance` / `min_clearance`
   权重 **0.0**、`collision` **全域 −1.0**；**没有新增奖励项**。因此 JNT/横向/冲击仍只由
   评估判据约束，训练不会直接优化它们；若要加项，需先补观测/契约。`fall` 仍是高度
   <0.18 m（局部地形，含波浪）**或** |pitch|/|roll| >0.8 rad（2026-10-19 已加）；高度阈值
   0.18 vs 评估 0.15 的差异仍在。
6. **无 GPU**：本机 `torch.cuda.is_available() = False`、`nvidia-smi` 权限不足，
   Isaac Lab 仿真一次未跑；所有运行时结论待训练机。

## 6. 复现与验收命令

```bash
# 离线：口径单元测试
python3 -m pytest imgo2_rl/tests/test_towing_task_criteria.py -q

# 离线：全套
python3 -m pytest imgo2_rl/tests -q

# 离线：看 plan（不需要 GPU）
python3 imgo2_rl/scripts/towing/play_towing_test.py --dry-run --num-envs 1200

# 离线：旧报告复判（部分口径）
python3 imgo2_rl/scripts/towing/towing_task_criteria.py \
    imgo2_rl/logs/towing/play_test/<run>/report.json
```

## 7. 全列实测（2026-10-20，model_1300）

完整 1200 env 评测见 [全列结果记录](towing_eval_model1300_full_2026-10-20.md)。摘要：

- 任务 **0/1050**；速度 275/350、横向 1020/1050、JNT 0/1050、冲击 1048/1050、不翻车 1043/1050。
- **30 kg 342/349 不翻车**：平地 116/116、5° 57/57、10° 59/59；**波浪 3 + 减速带 4 次真翻车**（roll≈π，全部 30 kg）。
- 10 kg 速度失败集中 10° 35/61、波浪 24/61、减速带 43/55。
- JNT 仍 0/1050；任务 0/1050 的主因是旧 JNT 阈值，不是翻车回归。
