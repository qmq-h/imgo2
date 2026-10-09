# 拖曳必要性：同轮含无负载对照的 800 环境网格（2026-10-09）

这是**新形态**的一轮：800 个环境里 **700 个拖车 + 100 个不拖车**（`--no-cart-fraction 0.125`），
于是「拖曳 vs 无负载」的对比在**同一轮 run、同一场景、同一指令、同一地形**下成立 —— 只差负载一个变量。
上一代（全部拖车）的数据在 [../2026-10-09_necessity_800/](../2026-10-09_necessity_800/README.md)。

- **来源 run**：`imgo2_rl/logs/towing/play_test/20261009T122044Z_b258d30d`（`state=completed`、`env_count=800`、
  `elapsed_s = 1467`；`--record-every 5`、`--write-csv all`、`--compact-log`）
- **代码版本（注意）**：`report.json.git.commit = 1bbb405` **但工作树是脏的** —— 当时未提交的是拖曳上层侧的
  `upper_mdp.py` / `upper_env_cfg.py` / `mdp/episode_geometry.py` / `test_towing_slope_geometry.py` /
  `test_towing_upper_rl_contract.py` / `docs/towing_slope_goal_vae_2026-10-09.md`（这些**测试台也用**，
  它们随后被提交为 **`6fac89a`**）。⇒ **要复现这一轮请用 `6fac89a`**，不要只用 `1bbb405`。
- 判读口径与阈值见 [../2026-10-09_necessity_800/README.md](../2026-10-09_necessity_800/README.md)
  的「起步/停车关节响应误差的确切口径」与「口径与限制」两节（同一套）。

## 核心读数（800 case）

| 分组 | n | 判定码 | 起步关节 RMS | 跟速 MAE | 停车间隙 | 横向 \|y\|max |
|---|---|---|---|---|---|---|
| 拖曳（有负载） | 700 | **COL 276（39%）/ JNT 295（42%）/ SPD 129（18%）** | 0.218 | **0.181** | 0.490 | 0.030 |
| 无负载（`_nocart`） | 100 | **JNT 100（100%）**，**COL 0、SPD 0** | 0.191 | **0.054** | n/a | 0.025 |
| 合计 | 800 | COL 276 / JNT 395 / SPD 129；`FALL` **0** | 0.216 | 0.161 | 0.490 | 0.029 |

⇒ **追尾（COL）与跟速超限（SPD）100% 来自负载**（无负载一个都没有，跟速 MAE 0.181 → 0.054 m/s）；
**`JNT` 两边都 100%** ⇒ 它测的是「PD 静差（≈ τ/kp，扛体重）vs 未标定阈值」，不是拖曳造成的能力缺陷。

## 分段 `q − q*`（拖曳段，按剖面位置切）

完整三张表（拖曳 / 无负载 / 差值）见同目录的 [no-load_comparison.md](../2026-10-09_necessity_800/no-load_comparison.md)
与 `stats.txt`；摘要：

| 量 | 拖曳（示例：10° 上坡） | 无负载（同段） | 差值 |
|---|---|---|---|
| 池化 RMS (rad) | 0.211 | 0.185 | **+0.026** |
| 静差 \|mean(e)\| (rad) | 0.086 | 0.080 | +0.006 |
| 动态 std (rad) | 0.144 | 0.122 | +0.023 |
| \|vx−cmd\| (m/s) | 0.403 | 0.079 | **+0.324** |
| 平均 \|τ\| (N·m) | 3.13 | 2.80 | +0.33 |

全段差值范围：`Δ池化RMS` **+0.003 … +0.054**（最大在 5° 坡顶 +0.054、10° 坡顶 +0.046、5° 上坡 +0.029），
其中 `Δ静差` 只有 **+0.001 … +0.018**、`Δ动态` +0.012 … +0.037，而 `Δ|vx−cmd|` 是 **+0.08 … +0.32 m/s**。

段 × 速度（池化 RMS）：拖曳 平地 0.161/0.205/0.236、上坡 0.164/0.198/0.222；
无负载 平地 0.153/0.186/0.210、上坡 0.152/0.177/0.187（坡顶/下坡只有 1.5 m/s 到得了）。

## 文件清单

| 文件 | 内容 |
|---|---|
| `experiment.json` | 运行清单（参数、网格与工作条件分配、出生几何、caveats、git 与工作树状态） |
| `report.md` | 人读报告：逐坡度量级判定矩阵 + 分组统计 + 结论 + 限制 |
| `report.csv` | 逐 case 一行关键指标（800 行） |
| `report.json` | 全量：`cases`（800 个 case 全部指标 + `summarize_tow`，含 `case.cart_present`）+ `groups` + `conclusion` + `thresholds` + `git`/`elapsed_s` |
| `stats.txt` | `play_test_stats.py <run> --segments --segments-limit 0 --list-codes` 的完整输出（判定码分布、指标分位数、按 是否拖车·坡度·连接·质量·速度·行 分组、四张交叉表、**拖曳/无负载/差值 三张分段表**、段 × 速度矩阵、失败原因组合） |

**未入库**（留在原 run，`logs/` 已被 git 忽略）：800 个 `env*_*/tow.csv`（逐记录步轨迹，`--write-csv all`
⇒ 约 534 MB）与 `summaries/*.json`（9.5 MB，内容已含在 `report.json.cases`）。

## 复现

```bash
# 跑（约 25 min 仿真 + 25 min 判读；磁盘需容纳 ~534 MB 轨迹）
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/towing/play_towing_test.py \
    --headless --compact-log --no-cart-fraction 0.125 --write-csv all

# 出统计（本目录 stats.txt 就是这条命令的输出）
python3 imgo2_rl/scripts/tools/play_test_stats.py --latest --segments --segments-limit 0 --list-codes
```

## 已知瑕疵（不影响上面的结论）

1. `_nocart` env 的**小车运动类字段**（`cart_coast_distance_m` / `cart_coast_to_rest_m` /
   `cart_speed_at_stop_mps`）仍照算但**没有意义**（小车停在 2 m 外、绳力为 0）；判定侧已中性化
   （无负载 env 不产出 `COL`/`LOW`），但报告里这些数要看 `case.cart_present=false` 就跳过。
   下一轮在 `compute_case_metrics` 里把它们一并置 `None`。
2. **「平地(出口)」段两轮都没有样本**：5 s 内只有 1.5 m/s 档接近 9 m ⇒ 想看出口平地段需
   `--tow-duration` 8–10 s。
3. 拖曳段的「坡顶/下坡」样本行数少（1.5 m/s 专属），按 `(段, 速度)` 单元比较才严格；
   本目录的 `stats.txt` 里有每一边的「段 × 速度」矩阵可用于该口径。
