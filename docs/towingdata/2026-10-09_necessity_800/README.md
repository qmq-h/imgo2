# 拖曳上层任务必要性：800 环境完整网格的测试数据（2026-10-09）

> ⚠ **2026-10-10 奖励口径变更，LOW 计数同样不可比。** 用户 2026-10-10 批准：删除奖励项
> `extra_distance`（−0.1），`min_clearance` 阈值由 `ratio(0.25) × 连接长度` 改为
> `spawn_margin(0.85) × 出生间隙`、权重 −2.0 → −5.0。本目录的 `stop_margin_low`（LOW）
> 计数用的是测试台**自己的绝对 0.10 m 判据**（`gap_margin_limit_m`，本轮**未改**），
> 而**训练侧**阈值已换成逐 env 的出生几何口径（绳 0.310–1.012 m、杆 0.402–0.840 m）
> ⇒ **两侧不同口径，本目录的 LOW 计数不能与 2026-10-10 之后的训练侧读数对照**。
> 详见 [奖励改动记录](../../towing_reward_retune_2026-10-10.md) 与 README 问题表 TOW-24。

> ⚠ **归档口径（2026-10-10 起）不可比 —— JNT 已移出判定统计量。**
> 本目录的通过数（**0/800**）与判定码分布（`JNT` 345 / `COL` 315 / `SPD` 140 等）是按**旧口径**
> 算的：`startup_joint_error` / `stop_joint_error` 当时计入判定统计量。用户 2026-10-10 决定把
> 关节跟踪误差（JNT）**移出判定统计量**（依据：两条在 800 cell 上基线 800/800、策略 800/800
> 零区分度，量的是底层 PD 静差 `τ/kp`，阈值 `0.10 rad` 连最好的 case 都超 1.6 倍）。
> ⇒ 本目录的**通过数与判定码分布不能与 2026-10-10 之后的报告直接比**；
> 要逐格对齐旧口径，用 `play_towing_test.py --count-jnt` 重跑（判定逐位一致）。
> 新口径的重判读数（同一批 800 cell、同版本 `git df593f4`，仅剔除 JNT）：
> **基线 356/800、策略 436/800**；详见
> [JNT 移出判定统计量记录](../../towing_verdict_jnt_excluded_2026-10-10.md) 与
> [README TOW-23](../../../README.md)。下面各节的读数与结论均按**旧口径**保留（历史证据）。

这组数据是 [必要性测试台](../../../imgo2_rl/scripts/towing/play_towing_test.py) 在**训练场景本体**上跑完
**800 环境（40 列 × 20 行）× 2200 物理步**的那一轮产物，用来回答「纯脚本速度指令 + 冻结 AMP 策略
（即不做上层残差学习）**够不够**」。

- **来源 run**：`imgo2_rl/logs/towing/play_test/20261009T112519Z_2ead2ad7/`（`logs/` 已被 `.gitignore` 忽略，
  所以把可入库的部分（下表 5 个文件）复制到本目录归档；**原始逐物理步轨迹仍在原 run 里**，见「原始轨迹」）
- **代码版本**：`git commit 3492e46`，工作树干净（只有未跟踪的 `imgo2_rl/nohup.out`）——
  即**已含** `sim.step(render=False)` 的修复与进度打印，未含之后新增的 `--compact-log`
- **运行耗时**：`elapsed_s = 1471.2`（≈24.5 分钟，headless、3 张以下的 `--record-every 5` 记录）
- **设计与排查背景**：[必要性测试台记录](../../towing_necessity_test_2026-10-09.md)、问题表 TOW-10/TOW-11

## 结论（`report.md` 原文口径）

- 平地 `0/400` 通过、5° 坡 `0/200`、10° 坡 `0/200`（**总计 0/800**）；
- 失败模式跨起步/全程/停车多相 ⇒ 结论行给的是「脚本 shaping 不足以解释，属于上层残差/调度的目标」。

## 核心读数

| 项 | 值 |
|---|---|
| 判定码 | `COL` 315（39%）/ `JNT` 345（43%）/ `SPD` 140（18%） |
| 跌倒 `FALL` | **0**（最低 base 高 0.244–0.260 m ⇒ 无倒地、无塌陷） |
| 失败原因 | `startup_joint_error` **800/800**、`stop_joint_error` **800/800**、`stop_collision` 315、`speed_track_error` 255、`stop_margin_low` 32 |
| 起步关节 RMS (rad) | 中位 **0.218**（p25 0.174 / p75 0.243 / 范围 0.161–0.267） |
| 跟速 MAE (m/s) | 中位 **0.180**（范围 0.039–0.609） |
| 停车最小几何间隙 (m) | 中位 **0.490**（最小 −0.006 = 有一个 case 接触） |
| 横向 \|y\|max (m) / 朝向 max | 中位 0.029 / 2.26°（阈值 0.3 / 10°，**未触发**） |
| 几何间隙不可用样本非零 | 98/800 个 case（坏姿态被采到，非倒地） |

按工况切分（详见 `stats.txt`）：

- **追尾 `COL` 集中在「轻车 + 高速」**：5 kg 71%、1.5 m/s 60%（0.5 m/s 只有 9%）；
- **跟速 `SPD` 集中在「重车 + 陡坡 + 高速」**：25 kg 43%、10° 33%、1.5 m/s 33%；
- **`JNT`（起步/停车关节响应）与出生间隙无关**：行 0–19 的起步 RMS 中位全程 0.212–0.226，
  只有「停车最小间隙」随行号单调变大（连接越长初始间距越大）。

## 「起步/停车关节响应误差」的确切口径l

判 `JNT` 的两个量（`startup_joint_error` / `stop_joint_error`）比的是：

```
误差 = q − q*
  q  = robot_jp_NN  = 仿真里**实测的关节角**（robot.data.joint_pos，按策略顺序 FL/FR/RL/RR × hip/thigh/shank）
  q* = robot_jt_NN  = **当拍下发给关节的位置目标** = default_dof_pos + action_scale ⊙ clip(a, −3, +3)
                     （默认站姿 (0, 0.87, −1.82)；缩放：髋 0.125 rad、大腿/小腿 0.25 rad）
```

**不是**与默认站姿比、也**不是**与录制参考动作/专家数据比 —— 是"我们命令关节去的角度"与"关节实际
到达的角度"之差。语义（`joint_track_stats` 的注释）：上层任务的 12 维残差动作正是加在 `q*` 上，
所以 `q − q*` 就是**上层能修正的那部分偏差**（包含负载把腿压塌的量）。

- **窗口**：起步误差 = **拖曳相位（tow）开始后的第一个 `--transition-window`**（默认 **1.0 s**；
  `--record-every 5` 时 = 40 个记录行），**不含**前面 1 s 的站定段；停车误差 = coast 段第一个 1.0 s。
- **汇总**（`joint_track_stats`）：
  - `joint_rms_rad` = **12 关节 × 窗口内全部样本池化**的 RMS = `sqrt(Σ err² / (样本数 × 12))`
    （不是"逐关节 RMS 再取平均"）；阈值 0.10 rad；
  - `joint_max_rad` = 窗口内**单点最大** |err|（任意关节）；阈值 0.30 rad；
  - `worst_joint` / `per_joint_rms_rad` = 逐关节 RMS 与最大者；
  - `torque_saturated_frac` = `|τ| > 0.95 × 23.7 N·m` 的（关节, 样本）占比。
  - **任一超阈值即触发 `*_joint_error`**（两个原因串同名）。
- **同窗口的另外三个「起步」量**（不是关节误差，别混）：`startup_vx_rms_err_mps`
  = RMS(实测**体系** vx − 指令 vx)、`startup_time_to_90pct_s`（到 90% 指令并保持 0.25 s 的时刻）、
  `startup_overshoot_ratio`。而判 `SPD` 的 `speed_mae_mps` 是 **tow 段全程**的体系 vx 与指令的 MAE。

**本组数据的实测值**（800 case）：`joint_rms_rad` 中位 **0.218 rad（12.5°）**、p25 0.174 / p75 0.243 /
范围 0.161–0.267（阈值 0.10 ⇒ 800/800 超）；`joint_max_rad` 中位 **0.876 rad**（阈值 0.30 ⇒ 800/800 超）；
`worst_joint` 分布 **RR_shank 56% / RL_shank 29% / FL_shank 15%**（误差集中在**小腿**、后腿为主）；
这些 case 的 `torque_saturated_frac` 中位 **0.000** ⇒ 不是力矩不够，是**负载下的小腿跟踪滞后**。

具体一行（env0000，拖曳第一拍 t=1.005 s）：`FL_shank` 目标 `q* = −1.532`、实测 `q = −2.043` ⇒
误差 **−0.511 rad**（腿比命令**更折叠**）；起步 1 s 窗口内该关节 |err| 中位 0.293 rad、最大 0.511 rad，
而同一时刻的髋/大腿误差只有 0.05–0.12 rad —— 与"小腿比髋/大腿大 3–6 倍"的逐关节 RMS 一致。
字段位置：`report.json → cases[i].metrics.startup.*`；`report.csv` 的 `startup_joint_rms_rad` /
`startup_joint_max_rad` / `startup_worst_joint` / `startup_torque_sat_frac`；实现见
[`play_towing_test.py`](../../../imgo2_rl/scripts/towing/play_towing_test.py) 的 `joint_track_stats`（675-714 行）
与 `classify_case`（1034-1045 行）。

## 口径与限制（判读前必读）

1. **阈值是未标定的工程占位**：关节起步/停车 RMS ≤ 0.10 rad、单关节 ≤ 0.30 rad、跟速 MAE ≤ 指令 20%、
   停车间隙 > 0.10 m、跌倒 = base 高 < 0.15 m 或 |pitch| > 0.8 rad 的样本 > 20%（`report.json.thresholds`）。
   `startup_joint_error` / `stop_joint_error` 是 **800/800** ⇒ 这两条阈值对**任何**工况都不成立，
   「0/800 通过」不能直接当成「上层任务必要」的量化结论，应先按实测分布重新标定。
2. **质量档 20 / 25 kg 超出训练随机范围 [5, 15] kg**：那两档是**外推检查**，判读要与分布内档位分开。
3. **域随机化被有意固定**：地面摩擦固定 0.8（训练随机 0.4–1.2）、轮轴阻尼固定 0.032 N·m·s/rad
   （训练随机 0.008–0.032）、**不使用**训练侧的 12.5% 无小车锚点（所有 env 都拖车）；
   工作条件（速度 × 质量）由 `slot = row + column` **确定性轮转**：每个 cell 只落到一种组合。
4. **度量口径**：`progress` 是 x 行程（水平投影，10° 坡上与弧长差 < 1%）；跟速一律用**体系** vx；
   `summarize_tow.valid` 是**绳语义**判据（要求拖曳段出现过 `T > 0`），rigid 连杆张力有符号，
   它的失败项（如 `rope_never_taut_during_tow`）不代表工况失败；本测试台的判定不依赖它。
5. **不覆盖**：真机、跨环境隔离、breakaway/Coulomb 阻力、弹性绳另外 3 档 k/c、轮阻档、
   训练侧的逐回合重采样；一次仿真过程（不是 5 次分轮），网格与顺序见第 3 条。

## 文件清单

| 文件 | 内容 |
|---|---|
| `experiment.json` | 运行清单：参数、git 哈希、网格分布（连接/坡度/长度/工作条件分配）、出生几何、caveats |
| `report.md` | 人读报告：逐坡度量级判定矩阵、分组统计、结论、限制 |
| `report.csv` | **逐 case 一行**的关键指标（800 行，表工具友好） |
| `report.json` | **全量**：`question / scene / thresholds / arguments / groups / conclusion / cases(800，含逐 case 全部指标与 `summarize_tow`) / csv_cases / elapsed_s / git` |
| `stats.txt` | [`play_test_stats.py`](../../../imgo2_rl/scripts/tools/play_test_stats.py) 的完整交叉统计输出（判定码分布、指标分位数、按 坡度·连接·质量·速度·行 分组、四张交叉表、失败原因组合） |

## 复现

```bash
cd <repo>
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/towing/play_towing_test.py --headless --compact-log
# 更小/特定网格（示例）：前 20 个 cell = 第 0 列 row 0–19
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/towing/play_towing_test.py \
    --headless --num-envs 20 --record-every 1 --write-csv all --compact-log
```

产物落在 `imgo2_rl/logs/towing/play_test/<时间戳>_<随机>/`。**本目录的数据在跑完之后不需要重新生成**
（统计用纯标准库即可复算）：

```bash
python3 imgo2_rl/scripts/tools/play_test_stats.py docs/towingdata/2026-10-09_necessity_800   # 注意：该工具读的是 run 目录里的 summaries/
python3 - <<'PY'   # 直接复算 report.json 里的分组
import json; r = json.load(open('docs/towingdata/2026-10-09_necessity_800/report.json')); print(r['conclusion'])
PY
```

## 后续：同轮含无负载对照的那一轮

本目录是**全拖车**的一轮；用「同一轮里既有拖曳也有少量无负载 env」做的对照见
[../2026-10-09_necessity_800_noload/README.md](../2026-10-09_necessity_800_noload/README.md)
（结论摘要：拖曳的代价集中在**跟速**与**追尾**，`q − q*` 的静差部分与是否拖车无关），
三张对比表在 [no-load_comparison.md](no-load_comparison.md)。

## 下一步：无负载（不拖车）对照

`startup_joint_error` 是「实测关节角 `q` − 当拍下发的关节目标 `q*`」，实测**符号均值 = −τ/kp**
（PD 静差）⇒ 拖着负载必然有 ~0.2 rad，站着不动就有 0.212 rad（小腿）。所以要拿**完全不拖车**的一轮
做参考量（用户 2026-10-09 决定：不加"站定基线"判据，直接测无负载统计量再比较）：

```bash
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/towing/play_towing_test.py \
    --headless --compact-log --no-cart-fraction 1.0 --write-csv all
python3 imgo2_rl/scripts/tools/play_test_stats.py --latest --segments --segments-limit 100
```

同一套 `--segments` 口径分别跑本目录这轮（拖曳）与无负载那轮，**逐段逐格对比**；本目录的
`stats.txt` 里已有拖曳轮的分段结果（见「按剖面分段」一节）。

## 原始轨迹（未入库）

原 run 里还有 **800 个 `envXXXX_.../tow.csv`（合计 534 MB，逐记录步 2200/5=440 行 × ~120 列）**
以及 `summaries/*.json`（9.5 MB，内容已包含在 `report.json.cases` 里）。它们留在
`imgo2_rl/logs/towing/play_test/20261009T112519Z_2ead2ad7/`（`logs/` 被 git 忽略，**不在仓库里**）。
需要某个 case 的逐物理步轨迹（力矩/限位/张力/接触）时：

```bash
R=imgo2_rl/logs/towing/play_test/20261009T112519Z_2ead2ad7
ls $R/env0000*/tow.csv                      # 例：row0 compliant 5 kg
# 想要全网格的逐物理步轨迹：重跑并加 --record-every 1 --write-csv all（约 2 GB 内存 / 数百 MB 磁盘）
```
