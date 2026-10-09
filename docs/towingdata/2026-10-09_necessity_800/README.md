# 拖曳上层任务必要性：800 环境完整网格的测试数据（2026-10-09）

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
