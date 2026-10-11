# 滞空缺口奖励 `feet_air_time_deficit`：给 AMP 的 3 倍步频一个直接梯度（2026-10-10）

用户 2026-10-10 的决定：**加上 `feet_air_time` 作为奖励；退火（降低风格奖励占比）先不做。**
本记录写清楚「为什么是这一项」「阈值和系数怎么来的」「接线在哪」「验到了什么」「还缺什么」。

**本轮不动的东西**（都是刻意保留的变量）：`amp_task_reward_lerp` 仍 0.3、`λ_gp` 仍硬编码 10
（[amp_ppo.py:244](../imgo2_rl/scripts/rl_lab/rl_lab/algorithms/amp_ppo.py)，AMP-06 的 P1 仍未做）、
指令范围未动、动作数据未补录。

---

## 1. 为什么是这一项

| 事实 | 出处 |
|---|---|
| 策略的唯一实质缺陷是**周期 0.198 s 对参考 0.600 s（3.0 倍）** | [AMP 步态调整方案](amp_gait_adjust_plan_2026-09-18.md)、AMP-06 |
| 步幅偏小（0.61–0.74 倍）**只是周期的结果**：trot 里地面系步长 = `v × T / 2`，固定速度下周期拉长步长必然变长 | §4 的运动学推演 |
| 风格项（判别器）**在策略工作点上没有梯度**：局部 `|∂d/∂x|·σ ≈ 8e-5`，`d_policy` 20000 轮只动 0.07 | AMP-06 |
| 因此加风格权重、改风格占比都换不来步频；需要一项**对周期有直接梯度**的奖励 | [速度包线核算](amp_speed_coverage_2026-10-10.md) §9 |

滞空（swing time）是周期里唯一能被奖励直接观测到的那一半；另一半是支撑时间（`T = air + stance`）。
这一项**只压步频过快**，不承诺同时把支撑时间调对——见 §7 的失败线。

---

## 2. 为什么用「缺口」形式（新函数，而不是库里那个 `feet_air_time`）

```
feet_air_time（原有，未改） : Σ_落地足 (air − θ)      —— air > θ 之后**仍线性给奖**
feet_air_time_deficit（新） : Σ_落地足 relu(θ − air)  —— air ≥ θ 精确为 0，永不给奖
```

原始形式在滞空超过阈值后继续给奖，等于**奖励「滞空越长越好」**，策略可以用
「高抬腿 / 原地蹦 / 长时间单足悬空」换奖励——这与我们要的目标（周期回到参考值）不是一回事。
缺口形式：θ 处精确为 0、下界是每足每次落地 θ，只有「别迈太快」的梯度，没有「越慢越好」的捷径。
函数返回**非负缺口**、注册为**负权重**，与 `base_height_l2` / `lin_vel_z_l2` 的约定一致。

同时保留参考的两道门：指令范数 `> 0.1`（站定不给信号）与机身直立度门
（`clamp(-g_z, 0, 0.7)/0.7`，倒地时该项失效）。

---

## 3. 阈值 θ 怎么定的（实测，不是照抄）

方法：逐足取动作数据里「足端 base 系 z」，以 `min + 0.25 × (max − min)` 为腾空判据，
连续 ≥3 帧才算一段，取各段时长均值（纯标准库，`FrameDuration = 0.02 s`）。

| 片段 | FL 腾空 | FL 支撑 | RL 腾空 | RL 支撑 |
|---|---|---|---|---|
| `forward_0.3` | 0.191 s | 0.407 | 0.207 | 0.371 |
| `forward_0.6` | 0.193 | 0.362 | 0.207 | 0.390 |
| `forward_0.9` | 0.255 | 0.370 | 0.207 | 0.349 |
| `forward_1.2` | 0.235 | 0.212 | 0.275 | 0.328 |
| `backward_0.9` | 0.256 | 0.300 | 0.216 | 0.191 |
| `turn_left_1.0` | 0.191 | 0.362 | 0.202 | 0.373 |

**21 份动作的逐足腾空 = 0.19–0.28 s，中位 ≈ 0.21 s** ⇒ 取 **θ = 0.20 s**
（`FEET_AIR_TIME_THRESHOLD_S`）。**不要用库里的默认 0.5 s**：那比参考的整个周期（0.600 s）还长，
等于要求「滞空 ≥0.5 s / 周期 ≥1 s」，比参考还慢一倍。

---

## 4. 系数怎么定的（先说方法与推算，读数待训练机）

`c` 为「每步系数」（权重写 `c / step_dt`）时，该项每步的平均贡献

```
≈ c × (4 / T) × step_dt × (air − θ)        T = 周期，4 = 四足，step_dt = 0.02 s
```

| 步态 | T | air | 每步贡献（c = 1） |
|---|---|---|---|
| 当前策略 | 0.198 s | ≈0.09 s（占空比≈50%，由 T 估） | **−0.045** |
| 参考动作 | 0.600 s | 0.21 s | **+0.001** |

两种步态之差 ≈ **0.046 c / 步**；任务项每步 ≈ **1.402**（`0.3 × (4.0·track + 2.0·track_ang)`）。

| c | 差/步 | 占任务项 |
|---|---|---|
| 1 | 0.046 | 3%（等于没加） |
| **3** | **0.138** | **10%** |
| 5 | 0.23 | 16% |

⇒ 取 **c = −3.0**（`FEET_AIR_TIME_DEFICIT_PER_STEP`）。**有界性**（为什么敢给 3）：
缺口形式每足每次落地最多贡献 θ，即使周期掉到 0.1 s，每步量级也只有
`3 × 0.08 × 0.2 / 0.1 ≈ 0.48`，约为任务项的 1/3，压不垮跟踪。

### 4.1 历史注记（避免误读「把 amp_go2 的写法抄回来」）

被删掉的 `amp_go2` 配方里**确实有** `feet_air_time`（`AMP_GO2_RAW_WEIGHTS = {"feet_air_time": 1.0}`，
阈值 0.3 s，用户 2026-09-18 定）。但换算到同一「每步系数」口径它只有
`c = raw × dt_lg = 1.0 × 0.02 = 0.02`：因为 legged_gym/Isaac Lab 的 `weight` 都会被再乘一次 `step_dt`。
而 go2 配方的任务项（`track_lin_vel = 4.0 × 0.02`）每步系数也只有 0.08 —— 那套配方整体比
本仓库 AMP-only 配方（`TRACK_LIN_VEL_PER_STEP = 4.0`）小 **50 倍**。
所以照抄那个数字放进本任务，相当于 `c = 0.02`（任务项的 0.06%），**必然被淹没**。
本轮不是「把它加回来」，而是按本仓库奖励尺度重新标定。

---

## 5. 接线（改了哪四个地方）

| 文件 | 改动 |
|---|---|
| [mdp/rewards.py](../imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/mdp/rewards.py) | 新增 `feet_air_time_deficit`（放在 `feet_air_time` 之后；**原有函数一字未动**） |
| [base_move/amp_env_cfg.py](../imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/amp_env_cfg.py) | ① 模块级常量 `FEET_AIR_TIME_THRESHOLD_S = 0.20` / `FEET_AIR_TIME_DEFICIT_PER_STEP = -3.0`（含标定注释）；② `_keep_only_amp_task_rewards()` 里**新建** `RewTerm` 并加进 `keep_rewards`；③ 末尾按 `c / step_dt` 写权重；④ `import RewardTermCfg as RewTerm` |
| [tests/test_amp_alignment.py](../imgo2_rl/tests/test_amp_alignment.py) | 扩展 `test_amp_task_terms_match_reference_per_step_scale`（桩里加 `foot_body_names`、`mdp`、模块级常量从 **AST 取真实值**）；新增 `test_amp_feet_air_time_deficit_is_penalty_only` 与 `FeetAirTimeDeficitTests` 两项数值测试 |
| 本文 + README | 结论与状态 |

三个坑（都已在代码注释里写明）：

1. **该项在基类 `RewardsCfg` 里不存在**（是本任务专用的），所以必须先在 `__post_init__` 里建出来
   再让过滤循环保留；`keep_rewards` 里漏掉它就会在过滤时被置 `None`。
   **防御性**：即使真漏了也**不会静默失效**——紧接着的权重赋值会抛 `AttributeError`（§6 负向测试实测）。
2. **`sensor_cfg.body_names` 必须给四个足端**（`self.foot_body_names`）。给 `""` 会在环境构造时
   抛 `Not all regular expressions are matched!`，与 `base_height_l2` 是同一个坑。
3. `track_air_time=True` **已经开了**（[velocity_env_cfg.py:81](../imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py)），
   接触传感器不用动。

**与 fanziqi 任务的关系**：`Imgo2AmpRLAmpEnvCfg` 的 `apply_rlamp_task_rewards()` 只保留线/角速度两项，
会**把这一项清掉**（它是对照臂，忠于 rl_amp 配方）⇒ 本轮只有 45 维的 `flat-amp-height`
（可部署的那条）带上它。要 fanziqi 也带需要单独决定（它是「不加任何步态项」的基准）。

**标定用的现成仪表**：runner 早有 `AMP/mean_last_air_time_s` 与 `AMP/mean_air_time_fraction`
（`amp_on_policy_runner.py::_feet_air_metrics`），不用额外占 GPU 回放就能看滞空是否在往 0.20 s 走。

---

## 6. 验证（本机实跑）

* `pytest imgo2_rl/tests -q` ⇒ **567 passed**（同一工作区把本轮新增的 3 项 `--deselect` 后是
  **564 passed**，故净新增 3 项，全部通过）。解释器 = `/opt/conda/envs/isaaclab/bin/python3`
  （3.11.13），**有 torch/numpy ⇒ `FeetAirTimeDeficitTests` 的两项数值测试是真跑了、不是跳过**
  （HEAD 的全新 checkout 会因缺少被 gitignore 的本地 `logs/towing/.../report.json` 多 4 项跳过，与本轮无关）。
* 新增/强化的断言：缺口形式（`clamp(θ − air, min=0)`，且源码里不得出现无下界的 `air − θ`）；
  原有 `feet_air_time` 保持原样；阈值 0.20（且不得写回 `"threshold": 0.5`）；
  `sensor_cfg` 指 `contact_forces` 且 body 是四个足端；权重为负且 `weight × step_dt = −3.0`（对
  `step_dt ∈ {0.01, 0.02, 0.04}` 三个值）；数值行为：全达标 ⇒ **精确 0**（同一输入下原始形式是 +1.6 的
  正奖励）、低于阈值逐足求和、只看落地那一拍、站定与直立度两道门。
* **负向测试（本机实跑）**：① 把公式改回 `(air − θ)` ⇒ 3 项测试失败；② 把该项从 `keep_rewards`
  删掉 ⇒ 契约测试以 `AttributeError` 失败。两次都改回后全套恢复 567 passed。
* `git diff --check` 干净、`git ls-files -i -c --exclude-standard` = 0 条。

---

## 7. 未验证 / 训练机要做的事

**未验证（本机做不了）**：Isaac Lab **环境构造**跑不了——`isaaclab.envs` 在本机缺
`omni.log`、`omni.physics.tensors.impl.api`（无 Isaac Sim），所以**一次仿真都没跑**。
具体未知项：`SceneEntityCfg("contact_forces", body_names=[四个足端])` 在 AMP 场景下能否解析、
`compute_first_contact()` 的首次触地掩码是否符合预期、以及该项与
`reference_state_initialization`（reset 时把机器人瞬移到参考状态）在首拍的交互
（瞬移可能立刻产生一次「落地」，贡献一个小缺口）。系数 −3.0 是**推算值，不是读数**。

**R1 冒烟（只开这一项，256 环境 × ~300 轮）的读数顺序**：

1. 先对口径：读 `Episode_Reward/track_lin_vel_xy_exp`，若 ≈ `70/s`（= 1.402 × 50）说明 Isaac Lab
   的 `Episode_Reward/*` 是**按秒**记的（`episode_sums / max_episode_length_s`）。
2. 按同一口径读 `Episode_Reward/feet_air_time_deficit`：目标每步 ≈ 0.134 ⇒ 期望读数 ≈ **6.7/s**。
   若量级差一个数量级以上，按比例调 `FEET_AIR_TIME_DEFICIT_PER_STEP`（不要动阈值）。
3. 读 `AMP/mean_last_air_time_s`：期望从 ≈0.09 s 往 **0.20 s** 走；`AMP/mean_air_time_fraction` 配合看。
4. 用 `eval_gait.py` 看 **period（主指标，期望朝 0.600 s）** 与 stride（副指标，应随周期一起变长）。

**失败线（先说好，免得事后解释）**：

* `period` 不动 ⇒ 说明滞空不是瓶颈（`T = air + stance`，策略可能把支撑时间压短来抵消）⇒ 下一步应加
  **支撑时间/整周期**项，而不是继续加这一项的权重。
* tracking 掉到 **< 90%**，或根高掉出 **0.28–0.32 m**，或 `AMP/mean_air_time_fraction` 异常升高
  （长时间腾空）⇒ 系数过大。
* 一次只动这一个变量：**不要**同时改指令范围、`λ_gp` 或 `lerp`；退火仍按上一轮的分析留到
  「信号加够但步态还不动」之后（[速度包线核算](amp_speed_coverage_2026-10-10.md) §9）。
