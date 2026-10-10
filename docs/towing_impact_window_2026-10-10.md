# 冲击窗口独立统计（`play_towing_test.py`，2026-10-10）

关联：[README](../README.md) 问题表 **TOW-22** 与维护记录（2026-10-10「冲击窗口」一行）；
需求来源见 [上层双头方案记录](towing_upper_two_head_2026-10-10.md) 的
「## 冲击窗口必须独立统计」一节。**本机没有 Isaac Lab（仿真一次都没跑）**，下面是离线结论；
运行期行为一律记为「已修，待训练机验证」。

## 交付内容

**改了什么**（只动 [play_towing_test.py](../imgo2_rl/scripts/towing/play_towing_test.py) 与
[test_towing_play_test.py](../imgo2_rl/tests/test_towing_play_test.py)，**只新增**）：新增纯函数
`impact_stats(rows, *, record_dt, impact_window_s, steady_margin_s, takeup_force_threshold_n,
deck_limit_n, joint_names, torque_limits)` 与两个定位助手 `_takeup_index`（首个
`|rope_tension_n| > 阈值` 的样本 ⇒ rigid 连杆负张力也算穿绳）/ `_contact_index`（coast 段首个
接触见证：`|cart_deck_fx_n| > 1 N` 或负载单步速度跃变，阈值按 `record_dt/5 ms` 放大，与
`contact_witness` 同口径）。CLI 新增 `--impact-window`（默认 **0.2 s**）、`--steady-margin-s`
（默认 **1.0 s**）、`--takeup-force-threshold`（默认 **1.0 N**，取模长）。`compute_case_metrics`
把结果挂到 `metrics["impact"]`，`report.json` 增顶层 `impact_statistics`（本轮 / 同版本基线的
逐 case 中位数 + 样本数 + 最差关节众数），`report.md` 增「## 冲击窗口 vs 稳态（关节响应）」一节
（起拖 / 绷直 / 指令归零 / 停车撞击四行 + 稳态一行 + `RMS/稳态`、`RMS−稳态` 两列），`report.csv`
增 `impact_<窗口>_*` 扁平列。

**字段名**（单位与读法）：

| 窗口（对齐） | 每个窗口都有的字段 |
|---|---|
| `metrics.impact.startup.*`（`impact_startup_*`）：tow 段首行起 `[t0, t0+W]` | `joint_rms_rad` [rad] 全 12 关节合并 RMS（`q − q*`）；`joint_max_rad` [rad] 单关节最大绝对误差；`worst_joint`；`per_joint_rms_rad`（12 项）；`torque_saturated_frac`（`\|τ\|>0.95·limit` 的步占比，0–1）；`samples`；`available`；`time_s`；`window_s` |
| `metrics.impact.takeup.*`（`impact_takeup_*`）：首个 `\|rope_tension_n\| > 阈值` 的样本 ± W | 同上，另带 `tension_n` [N]、`vx_mps` [m/s]（该样本体系 x 速）、`time_since_tow_start_s` [s]、`half_window_s` |
| `metrics.impact.stop.*`（`impact_stop_*`）：coast 段首行（**指令归零**）起 `[t0, t0+W]` | 同 `startup` |
| `metrics.impact.stop_contact.*`（`impact_stop_contact_*`）：coast 段首个接触见证样本 ± W | 同上，另带 `contact` [bool]、`channels`、`deck_fx_n` [N]、`load_vx_mps` [m/s]、`time_since_stop_s` [s] |
| `metrics.impact.steady.*`（`impact_steady_*`）：牵引段去掉首尾各 `--steady-margin-s` | 同 `startup`，另带 `margin_s` / `start_s` / `end_s` / `span_s`；**它是分母**，`rms_over_steady` 与 `rms_delta_rad` 恒 `null` |
| 四个冲击窗口各带 | `rms_over_steady` = `joint_rms_rad / steady.joint_rms_rad`（>1 = 比稳态猛，= 倍数）；`rms_delta_rad` [rad] = 冲击 − 稳态；`per_joint_rms_over_steady`（逐关节比值，12 项） |

时间字段 `time_s` 是**回合内的绝对时间**（与 `tow.csv` 的 `time_s` 同轴）：

- `impact_startup_time_s` = tow 段首行（起拖）；`impact_stop_time_s` = coast 段首行（指令归零）；
- `impact_takeup_time_s` = 真实绷直时刻（**不是**起拖时刻，两者可能差一整段松弛时间）；
- `impact_stop_contact_time_s` = 真实撞击时刻，与 `impact_stop_time_s` 的差就是
  `impact_stop_contact_time_since_stop_s`（**可能差数百毫秒，这是本节要分开统计的理由**）。

`report.csv` 里的扁平列名是 `impact_<窗口>_<字段>`（例如 `impact_takeup_rms_over_steady`、
`impact_stop_contact_time_s`），另有 `impact_window_s` / `impact_steady_margin_s` 记录当轮口径。

**口径冻结已核**：`DEFAULT_THRESHOLDS`、`VERDICT_CODES`、`classify_case`、既有 `startup.*` /
`stop.*` 全未改；`impact_stats` 里**不读 `thresholds`**、不含任何判定分支，`compute_case_metrics`
里新参数只出现 3 次（签名 + 一次转发）——这三条都由测试用 AST 钉住（见下）。

**验证方式**（本机无 Isaac Lab，全部离线）：`python3 -m pytest imgo2_rl/tests -q` ⇒
**501 passed**（会话开始时基线 473 + 本轮新增 28，全部在同一文件 `test_towing_play_test.py` 里；`ImpactWindowTests` 19 项覆盖四个时刻的对齐与区分、
`time_s` 闭区间取样（`samples` 与按行数取整不同）、稳态去首尾（79 行 / `start_s` / `end_s`）、
比值与增量、**尖峰不被稳态平均掉**、无绷直样本、rigid 负张力算绷直、无接触样本、
仅速度跃变的接触、窗口超出记录长度、单步记录、空记录、力矩饱和 0/1、
四个新参数不进判定（同轨迹换窗口后 `verdict` 与 `startup`/`stop` 逐位相同）、
`case_report_row` 新增列且不顶掉既有列；`BaselineComparisonTests` 新增 4 项覆盖
逐 case 中位数/样本数/最差关节众数、旧归档无 `impact.*` 时显示 `n/a`、报告表格与
`report.md` 小节顺序；`CliTests` 新增 3 项覆盖默认值/plan 行/非法值；
`DryRunTests` 用 `runpy` 在子进程里实跑 `--dry-run` 并断言 `torch` 不在 `sys.modules`）。
另：`python3 -m py_compile`、`git diff --check` 通过。

**未验证**：本机无 Isaac Lab，**仿真一次都没跑**——`rope_tension_n` 的实际量级（阈值 1 N 是否
合适）、绷直/接触时刻在真实轨迹里与起拖/归零各差多少、`--record-every 5`（25 ms）下 ±W 窗口
只有 8–9 个样本够不够稳、以及「比值是否真的能区分基线与策略」都要在训练机上看读数；
`impact_*` 的数值**不参与判定码**，所以本轮离线结论只是「统计口径正确」，不是「冲击被改善」。


## 重跑命令（训练机，基线 + 策略两份）

见 README TOW-20 / TOW-22：同一版本跑两轮，除开关（与 `--compare-report`）外参数完全相同。
第 2 轮的 `report.md` 会同时出现「本轮（开启）」「同版本基线（`--compare-report`）」
与「归档基线参照（无 `impact.*`，只有 `n/a`）」三列。要读的三行是
`绷直/穿绳`、`指令归零`、`停车撞击` 的 `RMS/稳态`（同一 case 先算比值再取中位数）。
