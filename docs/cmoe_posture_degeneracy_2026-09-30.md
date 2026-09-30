# CMoE 姿态退化修复（平地蹲 7 cm + 膝盖蹭地 + 抬脚过高）

日期：2026-09-30
分支：`imgo2_CMoE`
状态：**已实现并离线验证；待训练机新 run 验证**（本容器无 GPU / 无 Isaac 环境构造）

---

## 1 现象与证据

用户回放 run `cmoe_v5_7_lv12cap` **@8500** 确认："**高度没保持住、膝盖往地、足端抬高**"。

分项奖励（`Episode_Reward/*` 是**每步加权值**；真实每步贡献 = 该值 × dt 0.02）：

| 分项 | @4000（退化前） | @8500 | 退化倍数 | 反解 |
|---|---|---|---|---|
| `track_world_vel_xy_exp` | —— | **+3.71** | —— | 任务主力 |
| `track_ang_vel_z_exp` | —— | **+1.37** | —— | 任务主力 |
| `base_height_l2` | −0.0030 | **−0.0484** | **14×** | 高度误差 RMS = √(0.0484/10) = **0.070 m** |
| `undesired_contacts` | −0.0008 | **−0.1019** | **127×** | ≈0.204 个违规接触/步（非足端 = 膝/小腿） |
| `lin_vel_z_l2` | —— | **−0.0585** | —— | vz RMS = √0.0585 ≈ **0.17 m/s** |

**关键失衡**：任务项合计 ≈ +5.1/步加权值，而高度误差 7 cm 的代价只有 0.0484 ⇒ **≈1%**。
策略没有理由站直；`undesired_contacts` 127× 说明"跪下用膝支撑"是一条被允许的省力路径。

## 2 根因（按四项改动对应）

1. **高度约束太弱且自带折扣**：`base_height_l2 −10` 是全地形唯一的高度项，且乘了直立门
   `clamp(−g_z, 0, 0.7)/0.7`。而"蹲 + 膝蹭"正是**俯仰/侧倾 + 压低**的组合姿态 ⇒ 门控让退化
   姿态**自己给自己打折**。
2. **接触约束被历史性放松**：2026-09-24 为治"停在沟前"，把 `undesired_contacts` 从 −5 降回
   −0.5、`contact_forces` 留 −0.02。跨沟的"轻擦"确实需要容忍，但当时没有区分"轻擦"与"称重跪地"。
3. **竖直速度罚只有 PPO 原值的一半**：−2.0（且 `boxes`/`gap` 豁免）压不住"耸一下再落"。
4. **缺少第一手读数**：既有 `gait_height_<地形>` 是 `base_height_l2` 的 `(误差)²` 反解，
   **丢符号** ⇒ 日志里分不出"蹲矮"和"抬高"，只能靠用户回放发现。

## 3 四项改动（权重按用户给定，未自行调整）

### ① 逐地形有符号高度误差探针 `diag_base_height`（1e-6，只记录）

* 代码**逐字取自 `foot_clearance` 分支**（`git show foot_clearance:.../mdp/rewards.py` 的
  `def diag_base_height`）；函数体一字未改，只删掉 docstring 里对同批次**未**移植的
  `diag_clearance_mean` / `gait_clearance_<地形>` 的交叉引用。
* 语义：基座相对**本环境局部地面**的高度误差（m，**正=偏高、负=偏矮**），与 `base_height_l2`
  同一套射线逻辑（逐环境只用有效射线；**全部落空 ⇒ 退回误差 0**），但**不平方、不乘权重、
  不乘重力门**。
* 接线：`CMoERewardsCfg` 里 `RewTerm(func=mdp.diag_base_height, weight=1e-6,
  params={target_height:0.30, sensor_cfg:SceneEntityCfg("height_scanner_base")})`；
  `terrain_levels_vel_logged` 的 `gait_metric_terms` 加 `("base_height", "diag_base_height")`
  ⇒ 逐地形读数 **`gait_base_height_<地形>`**。
* ⚠️ **只移植了这一项**：`feet_swing_clearance` / `clearance_math.py` / `FeetSwingClearance` /
  `feet_height_body` 改动 / 其它诊断项**一概未带进来**（该批已在 `0a5bf04` 从本分支下线，
  代码保留在 `foot_clearance` 分支）。

### ② 平地专用严格高度项 `base_height_flat_l2`（−35，flat-only、去重力门）

* 实现：新增 `mdp._local_ground_target_height`（**从 `base_height_l2` 逐字抽出**，无行为变化）
  ＋ `mdp.base_height_l2_strict`（同一局部地面逻辑，**只去掉重力门**）
  ＋ `mdp.MaskedBaseHeightL2Strict`（**白名单**掩码类）。
* 参数：与 `base_height_l2` 同 `target_height`、同 `height_scanner_base`、同基座 `asset_cfg`
  —— `__post_init__` 里 `target_height` **从 `base_height_l2.params` 现取**，避免两处漂移。
* 掩码语义与既有 `Masked*` 家族**方向相反**：那些是 `free_terrain_names`（豁免名单），本项是
  `active_terrain_names=("flat",)`（**白名单 ⇒ 只有平地生效**）。参数名刻意不叫
  `free_terrain_names`，免得"豁免 flat"与"只在 flat"看反（看反了会静默变成全地形 −35）。
* 权重：`CMoERewardsCfg` 默认 **0.0**（不影响其它任务），`Imgo2CMoERoughEnvCfg.__post_init__`
  设 **−35.0**；`Imgo2CMoEGaitFreeEnvCfg`（`-gaitfree`）**不归零**（它不是步态形状项）。
* 结果：flat 上 `base_height_l2 −10` 与 `base_height_flat_l2 −35` 同时生效 ⇒ 总强度 ≈ **−45**；
  7 cm 蹲姿的每步代价从 −0.049 变成 −0.049 −0.1715 = **−0.22**（≈ 任务项的 4.3%，翻 4.5×）。
  障碍地形只受 −10 约束（避免惩罚合法的越障姿态）。
* 回退：把 `self.rewards.base_height_flat_l2.weight = -35.0` 改回 `0.0`（然后它会因
  `disable_zero_weight_rewards()` 被移除）。

### ③ `lin_vel_z_l2` −2.0 → **−4.0**

* 掩码**保持 `free_terrain_names = ("boxes", "gap")` 不变** —— 跃起必须免费。
  2026-09-29 曾取消豁免，实测 L1~L4 的 gap 全无飞行相、boxes 前腿不承重 ⇒ 用户回放
  "gap/boxes 过不去" ⇒ 当日回退（`f7d1dc3`）。**不要再动掩码**；要压"雷霆大跳"应改用
  **滞空上限**（只罚长腾空，`foot_clearance` 分支的 `feet_air_time_over`，**待做**）。
* 回退：一行改回 `-2.0`。监控：`Episode_Reward/lin_vel_z_l2`、逐列 `gait_bounce_<地形>` 应降，
  而 `level_gap` / `level_boxes` / `tracking_pass_frac_*` 不得下滑。

### ④ 接触惩罚与终止（治"膝盖往地"）

| 项 | 旧 | 新 | 回退 |
|---|---|---|---|
| `undesired_contacts` | −0.5 | **−5.0** | 一行改回 −0.5 |
| `contact_forces` | −0.02 | **−0.1** | 一行改回 −0.02 |
| 新增 DoneTerm `illegal_contact_body` | 无 | `mdp.illegal_contact`、`body_names=[r"^(?!.*_FOOT).*"]`、**`threshold=50.0`** | 模块级 `ENABLE_ILLEGAL_CONTACT_BODY_TERMINATION = False`（**一行**） |

* 分工：`undesired_contacts` 罚**非足端**接触（膝/小腿/大腿/髋/基座，阈值 1 N）；
  `contact_forces` 罚**足端**接触力（阈值 100 N）；`illegal_contact_body` 只在**称重跪地**
  （瞬时 ≥ 50 N）时终止。阈值刻意取高：既有基座触地终止用 1 N，若这里也用 1 N，任何小腿轻擦
  都会立刻结束回合 —— 那正是 2026-09-24「罚太重 ⇒ 策略停在沟前」的同一个坑。
* ⚠️ **这是四项里最容易伤到 gap/stairs 的一项**：跨沟/上台阶时小腿/膝的**轻擦**常见且必要。
  若 `illegal_contact_body` 的终止占比明显上升、或 `level_gap`/`level_pyramid_stairs*` 下滑，
  **优先怀疑它**，先关开关再查其它三项。
* 与 `-gaitfree` 的关系：`undesired_contacts` / `contact_forces` 是安全/物理项，两条链同样生效。

## 4 验证（离线）

* `check_reward_overrides.py cmoe` ⇒ **29 项**（`base_height_flat_l2 −35`、`diag_base_height 1e-6`、
  `undesired_contacts −5`、`contact_forces −0.1`、`lin_vel_z_l2 −4`）；
  `cmoe-gaitfree` ⇒ **25 项**（`base_height_flat_l2` 仍是 −35，未被归零）。
* `check_terrain_columns.py` ⇒ 末尾"全部掩码引用的地形名都有 ≥1 列 ✅"（新增
  `base_height_flat_l2.active_terrain_names=("flat",)` 已纳入 `MASKED_NAMES`，flat 有 3 列）。
* 新增 `tests/test_cmoe_posture_penalties.py` **41 项**：
  * 掩码：桩 env ＋ AST 抽真实类源码 ⇒ **每个非 flat 地形恰好 0**、flat 上等于未加门的 L2 平方
    误差、`−35 × 0.07² = −0.1715`、逐环境掩码、局部地面偏移、整束落空 ⇒ 0、默认白名单是 flat；
  * 去门控：倾斜（−g_z=0.35 ⇒ 门=0.5）时严格项 = **2×** 带门项；直立时两者相等；
  * `diag_base_height`：抬高 ⇒ **+0.05**、蹲 ⇒ **−0.07**、有符号非平方、局部地面偏移、
    整束落空 ⇒ 0、逐环境互不影响、无扫描器 ⇒ 世界系目标、倒立 ⇒ 读数不变（无门控）、
    权重 1e-6（≤1e-5 ⇒ 不进梯度）；
  * 配置级：−5 / −0.1 / −4（掩码仍 `("boxes","gap")`）、DoneTerm 的 func/body_names/threshold=50、
    一行开关、`base_height_l2 −10` 保留、严格项只在 flat、
    `("base_height","diag_base_height")` 已接线、`-gaitfree` 无任何对严格项的赋值；
  * 契约级：`MaskedBaseHeightL2Strict.__call__` 与 `diag_base_height` 的参数名**覆盖** cfg 的
    全部 `params` 键（Isaac Lab 对 class-term 是 `cfg.func(env, **cfg.params)` ⇒ 少一个同名参数
    就是运行期 `TypeError`，离线 AST 审计看不到）；掩码类沿用 `_terrain_type_mask` 家法
    （`__init__` 缓存、`__call__` 乘掩码）。
* 全仓 `unittest discover -s tests` ⇒ **559 通过 / 8 跳过**（改前 **515 通过 / 8 跳过**）。

## 5 未验证 / 限制

* **没有起训练、没有构造环境**（本容器无 GPU：`NVIDIA_VISIBLE_DEVICES=void`；`import isaaclab`
  在裸解释器下缺 `omni.log`）。以下全部**未测**：
  * `MaskedBaseHeightL2Strict` 在真实 `RewardManager` 下的 `params` 解析（`active_terrain_names`
    是否被 `_resolve_common_term_cfg` 当成普通参数传入 —— 它是普通 tuple，应与 `free_terrain_names`
    同类，属**同类先例**但未实测）；
  * `illegal_contact_body` 的 50 N 阈值是否真的只命中"称重跪地"（需要真实接触力量级标定）；
    `.*_FOOT` 否定前瞻在真实 body 列表上的解析结果；
  * −35 / −45 的总强度是否"够用而不至于扼杀平地行走"（可能把策略推向另一种局部最优）；
  * `contact_forces −0.1`（×5）在 gap 边缘落地时是否过重；
  * `gait_base_height_<地形>` 的读数是否符合预期（需要新 run 的 TB 曲线）。
* 训练奖励改动**不能 resume**，必须起新 run。
* `diag_base_height` 自带一份与 `_local_ground_target_height` 等价的内联射线逻辑（为与
  `foot_clearance` 分支逐字一致）——**两份实现漂移的风险由测试
  `TestDiagBaseHeight.test_ground_offset_uses_the_local_ground` 与 `test_all_rays_miss_*` 部分锁住**，
  但若以后改 `base_height_l2` 的射线口径，**两个函数都要改**。

## 6 验收（训练机新 run）

起新 run（`--task=Imgo2-basemove-rough-cmoe-gaitfree`），确认：

1. TB 的 `Episode_Reward/` 里有 **`base_height_flat_l2`** 与 **`diag_base_height`**；
2. `Episode_Reward/base_height_l2` 与 `base_height_flat_l2` 的绝对值都从 −0.0484 明显下降；
3. `gait_base_height_flat` 的**有符号**读数从 −0.07 向 0 收敛（若变成明显正值 ⇒ 矫枉过正，
   把 ② 的 −35 减半）；
4. `Episode_Reward/undesired_contacts` 向 0 收敛，且 **`Episode_Termination/illegal_contact_body`
   占比不显著上升**（上升 ⇒ 先关 ④ 的终止项）；
5. `level_gap` / `level_boxes` / `level_pyramid_stairs*` / `tracking_pass_frac_*` **不得下滑**
   （这是"没把越障能力改坏"的硬指标）；
6. 逐列 `gait_bounce_<地形>`（vz RMS）下降，`gait_airtime_flat` 不塌到 0。

## 7 改动文件

| 文件 | 说明 |
|---|---|
| `imgo2_rl/source/.../mdp/rewards.py` | ＋`_local_ground_target_height`（抽出）、＋`base_height_l2_strict`、＋`MaskedBaseHeightL2Strict`、＋`diag_base_height`；`base_height_l2` 改为调用抽出的 helper（行为不变） |
| `imgo2_rl/source/.../base_move/CMoE_env_cfg.py` | ＋`diag_base_height` / `base_height_flat_l2` RewTerm；`__post_init__` 里四项权重与接线；＋`ENABLE_ILLEGAL_CONTACT_BODY_TERMINATION` 与 `illegal_contact_body` DoneTerm；`gait_metric_terms` 加 `("base_height","diag_base_height")` |
| `imgo2_rl/scripts/tools/check_terrain_columns.py` | `MASKED_NAMES` 加 `base_height_flat_l2.active_terrain_names` |
| `imgo2_rl/tests/test_cmoe_posture_penalties.py` | **新增 41 项** |
| `imgo2_rl/tests/test_check_reward_overrides.py` | 生效项数 27→29 / 23→25；`lin_vel_z_l2` −4；新增 3 项断言 |
| `imgo2_rl/tests/test_masked_terrain_terms.py` | `lin_vel_z_l2` 期望 −2.0 → −4.0 |
| `README.md` | 维护记录一行 ＋ 问题表 `CMOE-14` 状态更新 |

---

# 追加（2026-10-01）：膝关节**高度**软地板 `knee_height_flat`

状态：**已实现并离线验证；待训练机新 run 验证**（本容器无 GPU / 未构造环境）。

## 8 为什么接触口径不够

用户对 run `cmoe_v5_8b_posture` @4527 回放的判断："**对膝盖接触地面限制是没用的，他只是比较低**"。

新探针 `gait_base_height_flat = **−0.085 m**`（平地上基座仍比站姿低 8.5 cm），而 2026-09-30 加的
flat-only `base_height_flat_l2`（−35、去重力门）**没有压住**（蹲姿代价只占任务 ~5%）。
既有的两条膝相关口径都是**接触**口径：

* `undesired_contacts`（−5，非足端接触）—— 只有"碰到"才计数，"低但不碰"读数为 0；
* `illegal_contact_body`（50 N 终止）—— 已因 50 N 把 **100% 回合在 6 步内终止**而**默认关闭**。

⇒ 结论：必须补一条**高度**口径。

## 9 阈值标定（离线 FK，不许拍脑袋）

用仓库既有 FK（`imgo2_rl/scripts/tools/audit_amp_dataset.py` 的 `read_chain` / `forward_kinematics`，
stdlib only，不引入 isaaclab），源 `imgo2_description/urdf/imgo2.urdf`，
默认关节角 **hip 0 / thigh 0.87 / shank −1.82**（`assets/imgo2.py` 的 `init_state`），
基座 z = **0.30**（`base_height_l2` 的 `target_height`）：

| 量 | 值 |
|---|---|
| `FL/FR/RL/RR_SHANK` 原点相对基座的 z | **−0.141612 m** |
| **站姿膝高 `knee_z_stand` = 0.30 + (−0.141612)** | **0.158388 m** |
| `knee_min = 0.70 × knee_z_stand` | **0.110872 m** |
| 四舍五入到 mm ⇒ **cfg 里的 `min_height`** | **0.111 m** |
| 对照：同一算式的 `*_FOOT` 原点（踝）世界系 z | 0.038561 m |

链路依据：模型是 `*_HIP → *_THIGH → *_SHANK → *_FOOT`，关节 `*_shank_joint`（**膝**）的
`<origin xyz="0 0.0557 -0.22">` 就是 `*_SHANK` link 的原点 ⇒ 该 link 原点的世界系 z
**就是**膝关节高度，不需要额外偏移。四条腿同高（站姿对称，实测差 < 1e-9）。

`tests/test_cmoe_posture_penalties.py::TestKneeHeightThresholdFromFK` 用**同一套 FK** 复核
（容差 5e-3 m），并有负向对照（`*_THIGH` 原点比膝高 0.14 m ⇒ 证明取的是膝而不是大腿根）。

## 10 实现

* **内核** `mdp.knee_height_soft_floor(env, min_height, asset_cfg)`：
  `Σ_{4 条小腿} relu(min_height − body_pos_w[:, body_ids, 2])`（**世界系 z**，逐腿各自计一次，
  线性、不平方、不取均值）。取不到 4 条即报错（防"body_names 打错 ⇒ 静默少算"）。
* **掩码类** `mdp.MaskedKneeHeightFlat(ManagerTermBase)`：与 `MaskedBaseHeightL2Strict` 同一套家法
  （`__init__` 用 `_terrain_type_mask` 缓存白名单 `active_terrain_names`，`__call__` 乘
  `self._active_mask.float()`）。白名单而不是豁免名单：障碍地形（上箱/跨沟/上台阶）需要屈膝，
  膝高天然低于站姿，在那里压膝高等于惩罚合法动作。
* **权重**：`CMoERewardsCfg` 默认 **0.0**（不影响其它任务）；`Imgo2CMoERoughEnvCfg.__post_init__`
  一行启用 **−20.0**。它是**姿态**项、不是步态形状先验 ⇒ `-gaitfree` 子类**不归零**。
* **一行可关**：`self.rewards.knee_height_flat.weight = -20.0` ⇒ 改 `0.0`（或删掉该行 ⇒ 类体默认 0.0）。
* **一行可调阈值**：`CMoERewardsCfg.knee_height_flat` 的 `"min_height": 0.111`（算式与出处写在
  该 RewTerm 上方的注释里）。
* **探针** `mdp.diag_knee_height_min(env, asset_cfg)`（权重 1e-6、只记录、不进门控、不平方）：
  逐环境 **4 条小腿里最低那条**的 z（m，有符号）—— 地板项罚的就是最低那几条腿，取 mean 会被
  其余腿抬高（退化形态常是单侧/对角两条腿低）。接线
  `("knee_height", "diag_knee_height_min")` ⇒ 逐列 `gait_knee_height_<地形>`；
  与既有 `gait_height_*`（`base_height_l2` 的平方值）、`gait_base_height_*`（基座有符号误差）
  **label 不冲突**（有测试锁住 label 唯一）。

## 11 验证（离线）

* 新增/扩展 `tests/test_cmoe_posture_penalties.py`（**41 → 77 项**）：掩码（非 flat 严格 0、
  混合地形逐环境各算各的）、语义（全高 ⇒ 0；一条低 3 cm ⇒ 恰 `3 cm × weight = −0.6`；
  四条各低 2 cm ⇒ `4 × 2 cm × weight = −1.6`；线性 1:2；恰在阈值 ⇒ 0）、body_ids 生效、
  膝数 ≠4 报错、探针取 **min**（0.16/0.16/0.16/0.09 ⇒ 0.09）且有符号、权重 1e-6、
  FK 标定（站姿膝高 0.158388 m、四条腿对称、THIGH 负向对照、`min_height` == 0.70×FK 容差 5e-3、
  四舍五入到 mm）、配置级（−20、白名单 `("flat",)`、`-gaitfree` 无赋值、`__call__` 签名覆盖 params、
  注释含标定算式）。
* `tests/test_check_reward_overrides.py`：生效项数 **29 → 31**（`cmoe`）／**25 → 27**（`cmoe-gaitfree`），
  新增 `knee_height_flat −20` 与 `diag_knee_height_min 1e-6` 断言。
* 全仓 `unittest discover -s tests` ⇒ **596 通过 / 10 跳过**（改前 **559 / 10**）。
* `check_reward_overrides.py cmoe` = **31 项**（`knee_height_flat −20` 在列）、`cmoe-gaitfree` = **27 项**；
  `check_terrain_columns.py` 末行"全部掩码引用的地形名都有 ≥1 列 ✅"（新白名单已进 `MASKED_NAMES`）。

## 12 未验证 / 限制

* **没有构造环境、没有起训练**（本容器无 GPU，`import isaaclab` 缺 `omni.log`）⇒ 以下全部**未测**：
  * `SceneEntityCfg("robot", body_names=".*_SHANK")` 在真实 `InteractiveScene` 下的解析结果
    （是否恰好 4 条小腿、顺序如何 —— 顺序不影响 `Σ` 与 `min`，数量不对会被新加的守卫报错）；
  * `body_pos_w[:, SHANK, 2]` 的坐标约定（Isaac Lab 里确实是**世界系** `frame=world`）；
  * `gait_knee_height_<地形>` 的实际读数（站姿应 ≈0.158 m）、−20 的强度是否够/是否过强；
  * `knee_min = 0.111 m` 的**行为**含义：真实步态里支撑期膝会低于站姿多少（0.111 是
    "站姿的 70%"，属**先验判断**，只有新 run 的曲线能校准）；
  * 四条腿逐腿求和是否会让"单腿异常低"被别的腿的高值掩盖（不会掩盖 —— 求和只增不减，
    但四条腿同时略低与一条腿极低可以给出相同的值，需要 `gait_knee_height_<地形>` 的 min 读数配合判读）。
* 奖励改动**不能 resume**，必须起新 run。

## 13 验收（训练机新 run）

起新 run（`--task=Imgo2-basemove-rough-cmoe-gaitfree`），确认：

1. TB 的 `Episode_Reward/` 里有 **`knee_height_flat`** 与 **`diag_knee_height_min`**，
   并有逐列 **`gait_knee_height_flat`**（站姿应 ≈0.158 m）；
2. `gait_knee_height_flat` 不再长期低于 **0.111 m**（若一直贴 0 ⇒ 强度不够，−20 → −30；
   若平地上膝高被抬到明显超过 0.158 ⇒ 矫枉过正，−20 → −10）；
3. `gait_base_height_flat` 同步向 0 收敛（膝高的改善**不该**靠另一种蹲姿换来）；
4. `Episode_Reward/knee_height_flat` 的绝对值下降，且 **障碍列不受影响**
   （`level_gap`/`level_boxes`/`level_pyramid_stairs*`/`tracking_pass_frac_*` 不得下滑）。

## 14 改动文件（2026-10-01）

| 文件 | 说明 |
|---|---|
| `imgo2_rl/source/.../mdp/rewards.py` | ＋`knee_height_soft_floor`、＋`MaskedKneeHeightFlat`（含 4 膝守卫）、＋`diag_knee_height_min` |
| `imgo2_rl/source/.../base_move/CMoE_env_cfg.py` | ＋`knee_height_flat` RewTerm（默认 0、启用 −20，`min_height=0.111` 带 FK 标定注释）、＋`diag_knee_height_min` RewTerm、`gait_metric_terms` 加 `("knee_height","diag_knee_height_min")`、`-gaitfree` docstring 补"不得归零" |
| `imgo2_rl/scripts/tools/check_terrain_columns.py` | `MASKED_NAMES` 加 `knee_height_flat.active_terrain_names` |
| `imgo2_rl/tests/test_cmoe_posture_penalties.py` | **41 → 77 项**（＋36） |
| `imgo2_rl/tests/test_check_reward_overrides.py` | 生效项数 29→31 / 25→27；新增膝高项断言（＋1） |
| `README.md` | 维护记录一行 ＋ 问题表 `CMOE-14` 状态更新 |
