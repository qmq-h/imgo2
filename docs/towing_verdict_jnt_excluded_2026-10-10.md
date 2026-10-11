# JNT 移出判定统计量（`play_towing_test.py`，2026-10-10）

关联：[README](../README.md) 问题表 **TOW-23** 与维护记录（2026-10-10「JNT 移出统计量」一行）。
代码：[play_towing_test.py](../imgo2_rl/scripts/towing/play_towing_test.py)、
测试：[test_towing_play_test.py](../imgo2_rl/tests/test_towing_play_test.py)。

**本机没有 Isaac Lab（仿真一次都没跑）**：下面是离线结论与对已有运行产物的复算；
运行期行为一律记为「已改，待训练机验证」。

## 决定与依据（用户 2026-10-10）

**关节跟踪误差（JNT）不进判定统计量** —— `startup_joint_error` / `stop_joint_error`
不再产生失败原因，`JNT` 从判定码分布里消失。

依据（两轮同版本、同几何、800 cell，来源见下节）：

1. **零区分度**：两条在 800 cell 上**基线 800/800、策略 800/800** 全部超限 ——
   通过数由它们决定时两边都是 0/800，`JNT` 只是把「所有 case 都失败」写了一遍。
2. **量的是底层 PD 静差，不是上层任务职责**：`q − q*` 的静差部分 ≈ `τ/kp`（扛体重），
   阈值 `joint_rms_limit_rad = 0.10 rad` **连最好的 case 都超 1.6 倍**：
   实测基线 `startup.joint_rms_rad` **P1 = 0.161 / P50 = 0.221**、
   `stop.joint_rms_rad` **P1 = 0.126 / P50 = 0.154**（策略轮同量级：startup 0.153/0.211、
   stop 0.126/0.159），两轮都是 800/800 超限。
3. 因此**字段与阈值保留**（观测），只有「是否据此判失败」由口径开关决定。

## 改动内容

| 层 | 改动 |
|---|---|
| 判定 | `classify_case(metrics, thresholds, *, count_jnt=False)`：**默认不计入**；只有 `count_jnt=True` 时才按 `joint_rms_limit_rad` / `joint_max_limit_rad` 产出 `startup_joint_error` / `stop_joint_error`（先后顺序与旧实现逐位一致） |
| CLI | 新增 `--count-jnt`（`action="store_true"`，默认关）；`--joint-rms-limit-rad` / `--joint-max-limit-rad` **保留**（仍参与观测标记与 `--count-jnt` 判定） |
| 转发 | `compute_case_metrics(..., count_jnt=False)` 把口径透传给 `classify_case`；`DEFAULT_COUNT_JNT = False` |
| 落盘 | `caliber_thresholds(args)` 在 `report.json` / `experiment.json` 的 `thresholds` 里加 `"count_jnt": bool` |
| plan | `[plan] 判定口径：**JNT 不计入**（默认…）` / `**JNT 计入**（--count-jnt 已打开）`，并打印两个关节阈值 |
| `report.md` | 配置区增「判定口径」行；判定码说明标注 `JNT` 默认不计入；「限制」节写明决定、理由、实测分布与归档不可比；「策略 vs 基线」的关节 RMS 中位数行**保留**（观测） |
| `report.json` | `thresholds.count_jnt`；`conclusion` 与 `groups[*].reasons` 反映新口径（不再有 JNT）；`baseline_comparison.archived` 仍按旧口径、报告里已标注不可比 |
| 归档 | `docs/towingdata/2026-10-09_necessity_800*/README.md` 增「归档口径（2026-10-10 起）不可比」一节 |

`DEFAULT_THRESHOLDS` 的键与数值**未改**（`VERDICT_CODES` 也保留 `JNT`，它只在
`--count-jnt` 时出现），所以「同阈值」这条对照前提不变。

## 「开 `--count-jnt` 旧口径可复现」的证据

复现方式是**逐位**的，不是"看起来一样"：把两轮 `report.json` 里每个 case 的
`metrics` 重新喂给判定函数，与归档里存的 `verdict` 对照。

```
classify_case(case["metrics"], thresholds, count_jnt=True)
  → code / reasons / severity  与归档 verdict 逐 case 完全相等：baseline 800/800、policy 800/800
classify_case(case["metrics"], thresholds, count_jnt=False)
  → reasons 恒等于「归档 reasons 剔除 startup_joint_error/stop_joint_error」，800/800
```

这条由 `imgo2_rl/tests/test_towing_play_test.py::JntExclusionRecountTests
::test_count_jnt_reproduces_the_archived_verdict_bit_for_bit` 在每次 `pytest` 里重跑
（缺 `report.json` 时 `skipTest`，不静默通过）。

## 实测重判（离线复算，本变更的依据与验收基线）

**数据来源**：两轮同版本、同几何、800 cell、`--no-cart-fraction 0.125`

- 基线：`imgo2_rl/logs/towing/play_test/upper_switch_baseline/report.json`
- 策略：`imgo2_rl/logs/towing/play_test/upper_switch_policy/report.json`
  （`--upper-checkpoint logs/towing_rl_lab/towing_upper/2026-10-09_22-18-13_rope05to15-rod05to10/model_1000.pt`；
  `upper_policy.enabled=true / iter=1000 / contract v2`）
- 两轮 `git.commit = df593f49d6469be0ec1bbd2637fa66bb5616cb3c`

**复算方式**：把 `cases[].verdict.reasons` 里的 `startup_joint_error` / `stop_joint_error`
剔除后重算通过数与分布（`JNT_EXCLUDED_RECOUNT_2026_10_10` 常量，测试逐项核对）。

| 维度 | 基线（旧口径 → 新口径） | 策略（旧口径 → 新口径） |
|---|---|---|
| 总通过 | 0/800 → **356/800** | 0/800 → **436/800** |
| 0° | 0/400 → **213/400** | 0/400 → **245/400** |
| 5° | 0/200 → **81/200** | 0/200 → **96/200** |
| 10° | 0/200 → **62/200** | 0/200 → **95/200** |
| compliant | 0/320 → **152/320** | 0/320 → **174/320** |
| inextensible | 0/160 → **69/160** | 0/160 → **88/160** |
| rigid | 0/320 → **135/320** | 0/320 → **174/320** |
| 有负载 | 0/700 → **256/700** | 0/700 → **336/700** |
| 无负载 | 0/100 → **100/100** | 0/100 → **100/100** |

**剩余失败原因**（剔除 JNT 后，一个 case 可命中多条）：

| 轮次 | `stop_collision` | `speed_track_error` | `stop_margin_low` | `lane_deviation` |
|---|---|---|---|---|
| 基线 | 295 | 234 | 28 | 0 |
| 策略 | 283 | 66 | 36 | 1 |

读法：剔除 JNT 后剩下的失败**全部与小车/停车/跟速有关**（正是上层任务该修的对象），
其中策略轮把 `speed_track_error` 234 → 66；`stop_collision` 几乎没动（295 → 283，
见下节「待定问题」）。

## 待定问题（本轮只记录、不实现）

`stop_collision` 的通道构成（基线轮 `cases[].metrics.impact.stop_contact.channels`，
即 coast 段**首个接触见证**的通道）：

- `load_velocity_jump` **254**、`deck_contact_force` **70**（并集口径的
  `stop.contact_channels` 是 264 / 71 —— 口径不同，不是矛盾）；
- `deck |fx|` **中位 0 N**（295 个 `stop_collision` 里真正有车斗接触力见证的是少数）。

⇒ **「负载是否发生速度跃变」这条判据可能也是过紧的占位阈值**：用绳拖 5–25 kg 车斗在
20 Hz 急停，速度在某一拍变化是**物理必然**。建议像冲击统计那样改成**相对量**
（负载减速度 / 与无负载对照的增量），是否实施**待用户决定**。

## 验证方式（本机无 Isaac Lab，全部离线）

| 检查 | 结果 |
|---|---|
| `python3 -m pytest imgo2_rl/tests -q` | **509 passed**（会话开始时 501） |
| `python3 imgo2_rl/scripts/towing/play_towing_test.py --dry-run` | 退出 0；plan 行含「判定口径：**JNT 不计入**」与 `--count-jnt`；`runpy` 子进程实测 `torch` 不在 `sys.modules` |
| `python3 ... --dry-run --count-jnt` | 退出 0；plan 行含「判定口径：**JNT 计入**…逐位一致」 |
| 归档复算 | 两次 `report.json` 的通过数/逐坡度/连接/负载/剩余原因逐项等于文档数字；`count_jnt=True` 对 800+800 case 逐位复现归档 verdict |
| `py_compile` / `git diff --check` / `git ls-files -i -c --exclude-standard` | 通过 / 无输出 / 空 |

新增/改写的测试（8 项）：

- `ClassificationTests::test_joint_error_is_not_counted_by_default`：合成 case 关节超限 ⇒
  默认 `OK` + `reasons == []`，观测值仍在；
- `ClassificationTests::test_count_jnt_reproduces_the_legacy_caliber_bit_for_bit`：`count_jnt=True`
  的原因顺序与判定码与旧实现逐位一致（含只超单关节阈值的情形）；
- `EpisodeMetricTests::test_slow_startup_raises_joint_flag`（改写）：默认不判 JNT、观测值超限；
  `count_jnt=True` 复现 `JNT`，两种口径的 `startup.joint_rms_rad` 相同；
- `ReportTests::test_markdown_report_excludes_jnt_but_keeps_the_observation_rows`：新口径报告不含
  `startup_joint_error×` / `stop_joint_error×` 失败行，但有「判定口径」「JNT 已按用户决定…移出判定
  统计量」与起步/停车关节响应 RMS 观测行；`--count-jnt` 渲染才出现 JNT 失败原因；
- `CliTests::test_count_jnt_defaults_off_and_plan_line_shows_the_caliber`：默认关、plan 口径行、
  两个关节阈值仍打印、`caliber_thresholds` 标记随开关翻面；
- `JntExclusionRecountTests` 4 项：来源核对（git/800 cell/0.125/策略 iter+契约）、
  重判数字逐项、`count_jnt=True` 逐位复现归档 verdict、待定问题的通道构成出处；
- `DryRunTests::test_dry_run_plan_lines_and_no_torch`（加强）：plan 里必须有默认口径行与 `--count-jnt`。

## 限制与未验证

- **重判不是重跑仿真**：它是把已归档 `report.json` 的 `reasons` 剔除 JNT 后重算分布。
  新口径在训练机上实跑出来的通过数、`report.md` / `report.json` 的实际渲染、
  以及 `--count-jnt` 对齐旧归档，都还没有运行证据。
- 归档 `docs/towingdata/2026-10-09_necessity_800*` 是**旧几何 + 旧口径**，与本轮
  不可逐格硬比；报告里已显式标注。
- 阈值 `joint_rms_limit_rad` / `joint_max_limit_rad` 仍是未标定的工程占位值：
  移出判定后它们只影响观测标记（`--count-jnt` 追溯旧口径时才有判定意义）。
- `stop_collision` 的速度跃变通道是否改相对量、以及「无负载对照」的增量口径，
  都**待用户决定**，本轮未实现。

## 训练机重跑命令（基线 + 策略，复核新口径）

```bash
# 第 1 轮：基线（默认新口径）
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/towing/play_towing_test.py \
    --headless --num-envs 800 --no-cart-fraction 0.125 \
    --output-dir imgo2_rl/logs/towing/play_test/jnt_excluded_baseline
# 第 2 轮：策略（同一套参数 + checkpoint + 基线 report.json）
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/towing/play_towing_test.py \
    --headless --num-envs 800 --no-cart-fraction 0.125 \
    --upper-checkpoint logs/towing_rl_lab/towing_upper/<run>/model_1000.pt \
    --compare-report imgo2_rl/logs/towing/play_test/jnt_excluded_baseline/report.json \
    --output-dir imgo2_rl/logs/towing/play_test/jnt_excluded_policy
# 第 3 轮（可选）：开旧口径，与 2026-10-09 归档对齐
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/towing/play_towing_test.py \
    --headless --num-envs 800 --no-cart-fraction 0.125 --count-jnt \
    --output-dir imgo2_rl/logs/towing/play_test/jnt_counted_baseline
```

验收：第 1/2 轮的 `report.md` 判定矩阵、`## 分组统计`、`groups[*].reasons` 里**不出现**
`JNT` 与 `*_joint_error`，「策略 vs 基线」的起步/停车关节响应 RMS 中位数行仍在；
`report.json.thresholds.count_jnt == false`；第 3 轮 `count_jnt == true` 且判定码分布回到旧口径。
