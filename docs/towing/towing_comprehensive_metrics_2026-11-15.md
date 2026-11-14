# 拖曳综合数据：地形 × 负载 × 速度 × 策略（2026-11-15）

> 这是把已有评测整理成**一份可筛选的总表**，不再按单轮 report 分开看。
> 所有数据都是历史 report 的统一字段抽取；不是重新跑一次仿真。

## 1. 文件

目录：[data/towing_comprehensive_2026-11-15](data/towing_comprehensive_2026-11-15)

| 文件 | 内容 |
|---|---|
| `cases.csv.gz` | **3600 行逐 case**：3 策略 × 5 地形 × 6 质量 × 2 速度 × 20 case |
| `summary.csv` | **180 行聚合**：策略 × 地形 × 质量 × 速度 |
| `metadata.json` | 源 report、覆盖范围、字段定义、限制 |

覆盖：

- 策略：
  - `baseline_oldppo_20hz`：老 PPO 无上层、20 Hz
  - `v6_m2000_50hz`：v6 model2000、fall −500 + roll −10、50 Hz
  - `v7_m2000_50hz`：v7 model2000、13 维、50 Hz
- 地形：`0deg / 5deg / 10deg / wave / bump`
- 质量：**5 / 10 / 15 / 20 / 25 / 30 kg**
- 速度：**0.5 / 1.0 m/s**
- 每格 20 case（1200/60）

15° / 20° 已从数据中排除；当前 60 列地形不再包含它们。

## 2. `summary.csv` 字段

| 字段 | 含义 |
|---|---|
| `policy`, `terrain`, `cart_mass_kg`, `velocity_mps`, `n` | 分组键与样本数 |
| `speed_applicable`, `speed_pass`, `speed_pass_frac` | 速度任务仅对 ≤15 kg 可判 |
| `speed_steady_mae_median` | 稳态 MAE 中位数（m/s） |
| `speed_steady_ratio_median` | 稳态实测/指令速度比中位数（1.0=完全跟上） |
| `speed_steady_std_median` | 稳态速度标准差中位数 |
| `speed_steady_p95_median` | 稳态绝对误差 p95 中位数 |
| `impact_applicable`, `impact_pass` | 启停冲击任务口径 |
| `impact_startup_rms_median` / `_rms_over_steady_median` | 起拖窗口 |
| `impact_takeup_rms_median` / `_rms_over_steady_median` | 绳绷直窗口 |
| `impact_stop_rms_median` / `_rms_over_steady_median` | 停车窗口 |
| `impact_stop_contact_rms_median` / `_rms_over_steady_median` | 停车接触窗口 |
| `fall`, `fall_rate` | `stability.fell` 次数/比例 |
| `no_fall_pass`, `lane_pass`, `joint_pass`, `task_pass` | 对应任务检查通过数 |
| `task_codes` | 该组任务判定码计数 |

## 3. `cases.csv.gz` 字段

逐 case 包含：

- 标识：`policy, source_run, frequency_hz, upper_policy, case_index, env_index, column, row`
- 工况：`terrain, grade_deg, cart_mass_kg, velocity_mps, connection, connection_length_m, cart_present`
- 速度：`speed_applicable, speed_ok, speed_steady_mae_mps, speed_steady_ratio_mean,
  speed_steady_std_mps, speed_steady_p95_mps, speed_frac_within_10pct`
- 冲击：每个窗口的 `impact_*_rms_rad / rms_over_steady / rms_delta_rad /
  torque_saturated_frac`
- 稳定性/任务：`fell, max_abs_roll_rad, max_abs_pitch_rel_rad,
  lane_ok, joint_ok, no_fall_ok, task_code, task_ok`

空单元格 = 该 case 没有可用数据（例如重车速度不判、无接触窗口）。

## 4. 查询示例

```bash
# 看 v6 的 25/30 kg、10 度、1.0 m/s
zcat docs/towing/data/towing_comprehensive_2026-11-15/cases.csv.gz | \
  awk -F, 'NR==1 || ($1=="v6_m2000_50hz" && $10=="25" && $11=="10deg" && $12=="1")' | head
```

PowerShell/Excel 可直接解压 `cases.csv.gz` 后筛选；`summary.csv` 也支持直接透视表。

## 5. 关键读法

- **速度**：
  - 只有 ≤15 kg 的 `speed_applicable=1` 才参与任务判定；
  - 重车看 `speed_steady_ratio_median` 作“能不能动/掉速多少”的诊断。
- **冲击**：
  - 任务通过看 `impact_pass`；
  - 绝对值 `rms` 受 JNT/稳态水平影响，比较时优先看 `rms_over_steady` 与 `rms_delta`。
- **任务通过**：
  - 三套策略当前 `task_pass` 都是 0，因为 `joint_static_error`（JNT）全失败；
  - 看速度/冲击/fall/lane 时不要被总任务 0 误导。

## 6. 限制

- 历史 report 抽取，不是一次统一的新仿真；
- 基线 20 Hz/无上层，v6/v7 50 Hz/上层，频率与架构同时变化；
- 只有 0.5/1.0 m/s；没有 1.5/2.0；
- 15°/20° 已排除；
- 无载组、质量突变组不在这份静态总表里，分别有单独记录。
