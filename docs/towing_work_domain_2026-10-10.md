# 拖曳工作域收紧：质量 5–20 kg、速度 0.5–1.5 m/s（2026-10-10）

## 背景与决定

用户 2026-10-10 确认拖曳上层任务的**工作域**为：

| 量 | 旧值 | 新值 |
|---|---|---|
| 小车质量 | 5–30 kg（2026-10-09 由 5–15 提到 5–30） | **5–20 kg** |
| 脚本速度 | 0.4–1.5 m/s | **0.5–1.5 m/s** |

这是**纯工作域对齐**：只改「训练域随机化」与「与之同步的测试台 / decoder 元数据」，
契约（帧 58 / actor 80 / critic 73 / 动作 13 / `towing_contract v3`）**一个字都不动**，
并新增守卫钉住这一点。

## 改动的 4 处常量（必须一起改，少一处就漂移）

| # | 文件 | 常量 | 旧 | 新 |
|---|---|---|---|---|
| 1 | `imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/upper_env_cfg.py` | `UpperEventsCfg.reset_work_condition.params["mass_range"]` | `(5.0, 30.0)` | **`(5.0, 20.0)`** |
| 2 | `.../towing/mdp/episode_geometry.py` | `SPEED_RANGE` | `(0.4, 1.5)` | **`(0.5, 1.5)`** |
| 3 | `.../towing/upper_logic.py`（任务书写的 `utils/upper_logic.py` 是笔误，实际不在 `utils/` 下） | `DecoderSpec.mass_range` | `(5.0, 30.0)` | **`(5.0, 20.0)`** |
| 4 | `imgo2_rl/scripts/towing/play_towing_test.py` | `TRAINING_MASS_RANGE_KG` | `(5.0, 30.0)` | **`(5.0, 20.0)`** |

第 3 处只作文档/派生用（**不参与任何张量归一化或损失门控**，`DecoderSpec.mass_range` 不进
`dim`），但训练域改了它必须同步，否则契约记录会与训练配置漂移。第 4 处决定测试台的
「域内 / 域外」判读：改前 25 kg 被算作域内，改后必须在 plan 行与 `experiment.json.caveats`
里标为**域外外推检查**。

## 耦合项：回合超时随最慢速度变短

`upper_env_cfg.__post_init__` 的 `episode_length_s` **不是常数**，而是由
`episode_timeout_s(坡面弧长上界, SPEED_RANGE[0], tow_start_s, margin=POST_STOP_WINDOW_S)`
推出来的，所以 `SPEED_RANGE[0]` 0.4 → 0.5 会**直接缩短回合**：

```
坡面弧长上界   = profile_arc_length(MAX_GRADE_DEG=10°, STOP_DISTANCE_M=10.0)
               = 10.09255967131447 m          （10 m 水平在最陡档上的弧长）
settle/tow_start = SETTLE_TIME_S = 1.0 s
停车窗口 margin  = POST_STOP_WINDOW_S = 3.0 s

旧（0.4 m/s）：1 + 10.09255967131447/0.4 + 3 = 29.231399178286175 s
新（0.5 m/s）：1 + 10.09255967131447/0.5 + 3 = 24.18511934262894  s
```

**步数**（`step_dt = sim.dt × decimation = 0.005 × 10 = 0.05 s`，20 Hz）：
Isaac Lab 0.45.9 的 `ManagerBasedRLEnv.max_episode_length` =
`math.ceil(episode_length_s / step_dt)`（`/root/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:104`）：

| | `episode_length_s` | `max_episode_length`（ceil，运行值） | 截断口径 |
|---|---|---|---|
| 旧（0.4） | 29.231399 s | **585 步** | 584 |
| 新（0.5） | 24.185119 s | **484 步** | 483 |

> ⚠ 2026-10-09 的记录（`towing_profile_terrain_2026-10-09.md`、README 顶部旧核对段）把
> 28.23/29.23 s 写成了 564/584 步，用的是**截断**；Isaac Lab 实际用 `math.ceil`，
> 因此当时的运行值是 565/585 步。本记录按运行口径给 484 步，并保留 483 作为截断对照。

**为什么 `decimation` / `num_steps_per_env` 不跟改**：

- `decimation = 10` 是**控制频率契约**：上层 20 Hz（`upper_control_dt = 0.05 s`）叠在冻结 AMP
  策略 50 Hz 之上（`sim.dt = 0.005`）。它由两个策略的接口决定，与回合能跑多久无关。
- `num_steps_per_env`（训练侧 rollout 长度，48）是**采样窗口**，与 `episode_length_s` 无关：
  一个 run 的 curve 只受「每轮采多少步」影响，不受回合上限影响；改它反而会改变训练动力学。
  因此两者都**保持原值**。

**自洽性检查（最慢速度 0.5 m/s 走 10 m + 余量）**：

- 水平 10 m @ 0.5 m/s = 20.0 s；坡面弧长 10.0926 m @ 0.5 m/s = 20.185 s。
- 加上 settle 1.0 s ⇒ 最慢速度抵达 STOP 点约 **21.185 s**，比 24.185 s 的超时早 3.0 s
  （正好等于 `POST_STOP_WINDOW_S`，见 `test_towing_slope_geometry.py` 的
  `test_training_timeout_uses_the_stop_window_margin`）。
- 0.5 m/s 走到坡面出口（9.0 m 水平）约 18 s，进度触发仍有充足余量保证「越过坡才置零指令」。
- 上方几何测试还逐 0.01 m/s 扫描整个 `SPEED_RANGE`，确认每个速度的到达时间 +0.1 s
  采样余量都严格小于 timeout。

## 契约不变守卫（新增）

`imgo2_rl/tests/test_towing_upper_rl_contract.py::test_work_domain_ranges_do_not_touch_the_contract`：

1. 训练域三处源码常量**四处一致**：env cfg 源码的 `"mass_range": (5.0, 20.0)` 与
   `"speed_range": SPEED_RANGE`、`DecoderSpec.mass_range == (5.0, 20.0)`、
   `episode_geometry.SPEED_RANGE == (0.5, 1.5)`，并用正则从 `play_towing_test.py` 读出
   `TRAINING_MASS_RANGE_KG` 与 `DecoderSpec.mass_range` 比对；
2. 契约维数/版本与工作域无关：`dataclasses.replace(DecoderSpec(), mass_range=(1.0, 99.0))`
   的 `dim` 不变，`(frame_dim, decoder_dim, actor_dim) == (58, 6, 80)`、
   `last_action == 13`、`CMD_ACTION_DIM == 1`、`CHECKPOINT_CONTRACT_VERSION = 3`。

## 其他被一并改掉的引用点

- 源码注释里的「脚本速度 0.4–1.5」：`upper_logic.py`（3 处）、`upper_mdp.py`（3 处）、
  `upper_env_cfg.py`/`episode_geometry.py` 的超时推导注释。
- 测试里的「0.4–1.5」注释：`test_towing_upper_rl_contract.py`（3 处）、
  `test_towing_upper_policy_runtime.py`（1 处）。
- `test_towing_upper_rl_contract.py` 的 `"mass_range": (5.0, 30.0)` 守卫 → `(5.0, 20.0)`。
- `test_towing_play_test.py`：`TRAINING_MASS_RANGE_KG`、dry-run plan 断言由「全部落在训练
  分布内」改为「超出该范围的质量档 25 kg ⇒ 外推检查」、默认速度下界 == 0.5。
- `test_towing_slope_geometry.py`：timeout 断言 28.2315 → 23.1851（默认 margin 2）、速度扫描
  改为由 `SPEED_RANGE` 驱动，并新增训练 cfg margin（3 s）下 24.185 s / 484 步的守卫。
- 文档变更注：`towing_slope_goal_vae_2026-10-09.md`、`towing_profile_terrain_2026-10-09.md`、
  `towing_upper_two_head_impl_2026-10-10.md`。

## 测试台：25 kg 现为域外

`play_towing_test.py` 的默认速度档 `0.5/1.0/1.5` 与新下界一致，**不改**；默认质量档
`5/10/15/20/25 kg` 中的 **25 kg 超出新训练域 5–20 kg**，按用户要求**保留**（允许外推检查），
并在 `[plan]`、`report`（`experiment.json.caveats`）与 docstring 里注明「25 kg 现为域外」。

## 验证

- `python3 -m pytest imgo2_rl/tests -q` ⇒ **553 passed**（改动前 551，本工作新增 2 条守卫）。
- `python3 -m py_compile` 全部改动文件通过。
- `git diff --check` 干净；`git ls-files -i -c --exclude-standard` 为空（仅未跟踪的
  `imgo2_rl/nohup.out`）。
- 离线复算（本机 Isaac Lab 版本 0.45.9，但只读源码/纯逻辑，**未起仿真**）：
  `episode_length_s` 29.231399 → 24.185119 s；`max_episode_length` 585 → 484 步（ceil）。
- 本记录的「待训练机」项：环境构造、`max_episode_length` 读回值、质量/速度采样张量、
  在 5–20 kg 与 0.5–1.5 m/s 下的实际动力学与停车窗口充分性。

## 未验证 / 待确认

- **运行期**：本机没有可跑的 Isaac Lab 仿真（无 GPU 视角），`episode_length_s`/`max_episode_length`
  的读回、两个域的采样、0.5 m/s 最慢档在坡上的表现都**待训练机验证**。
- **发现一处与本次改动无关的历史不一致（未改，供决定）**：`play_towing_test.py` 仍把
  上层 checkpoint 的期望契约写成 **v2 / frame_dim=57**（docstring 第 31–32 行、`[plan]` 打印、
  `experiment.json.upper_policy.towing_contract_expected`），而双头迁移后真实契约是
  **v3 / frame_dim=58**（`upper_policy_runtime.CHECKPOINT_CONTRACT_VERSION = 3`，加载用的是
  后者，所以**不影响运行**，只是记录/打印失真；`test_towing_play_test.py` 里那条
  `assertIn("frame_dim=57", ...)` 也钉住了旧值）。本轮按「只调工作域、不碰契约」未动，
  建议下一轮单独修并同步那条测试。
