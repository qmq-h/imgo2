# 奖励改动记录：删除 `extra_distance` + `min_clearance` 改出生几何阈值并提权（2026-10-10）

用户 2026-10-10 批准的两项奖励改动，落地在
[`upper_env_cfg.py`](../imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/upper_env_cfg.py) 与
[`upper_mdp.py`](../imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/upper_mdp.py)。

**本机没有 Isaac Lab**：以下全部结论只由离线 `pytest` / AST / 纯标准库复算得出；
凡只有训练才能确认的一律标「**待训练机验证**」（见 §7）。README 问题表条目 = **TOW-24**。

---

## 1. 改动 ①：删除 `extra_distance`

| 项 | 变更 |
|---|---|
| `UpperRewardsCfg.extra_distance` | **删除**（原 `RewTerm(func=mdp.post_stop_distance, weight=-0.1, params={"post_stop_allowance_m": 0.0})`） |
| `mdp.post_stop_distance` | **保留**函数本体与 `post_stop_allowance_m` 形参（默认 0.0），docstring 写明「已不作为奖励项」 |
| `post_stop_allowance_m` | 随之**目前未接入奖励**；只作为该函数形参保留供复用（0.0 = 历史口径 `relu(x − x_stop)`，0.5–1.0 m = 允走量） |
| `--compact-log` 分项 picks | `extra_distance` 去掉，换成 `min_clearance`（见 §2 提权） |

**删除依据（用户决定）**：它与 `tracking_velocity` 在停车段**高度冗余**——指令归零后 `vx → 0`
已经隐含位移只剩不可避免的惯性滑行，再单独罚一次是同一个约束记两遍；而且**量级小 20 倍**
（`weight=-0.1` ⇒ 每步 −0.0025 量级，`tracking_velocity` 满额 +0.050/步），压不住任何东西，
却给「停车后按车重/坡度再走两步」的动作侧方案凭空加一个反向梯度（偏移头一让机器人前进就扣分）。

`stop_towing_force`（−1.0）**保留**：停车段现在只剩这一项奖励；「停车后继续走几步」改由
§2 的 `min_clearance` 梯度与动作侧 1 维 vx 偏移头表达。

---

## 2. 改动 ②：`min_clearance` 改成「出生几何」阈值 + 提权

### 2.1 新口径

```
threshold   = spawn_margin × spawn_clearance          # spawn_margin = 0.85（RewTerm params）
spawn_clearance = sqrt(initial_attachment_distance(类型, L)² − Δz²) + base_offset
```

| 量 | 取值 | 来源（源码，不手抄数字） |
|---|---|---|
| `spawn_margin` | **0.85** | `UpperRewardsCfg.min_clearance` 的 `params` |
| `softness` | 0.02（不变） | 同上 |
| `weight` | **−5.0**（原 −2.0） | 同上 |
| `initial_attachment_distance` | 绳 = `SLACK_RATIO(0.8) × L`；杆 = `L` | `mdp/connection_grid.py`；逐 env 已落在 `term.initial_distance`（`__init__` 里由 `env_spec` 算出） |
| `Δz` | **0.17 m** | 机器人 `init_state.pos[2] = 0.35` − cart `init_state.pos[2] = 0.18`（两个挂点的 z 偏移都是 0） |
| `base_offset` | **0.0025 m** | `−robot_rear_surface_x(0.1575) − robot_attachment_x(−0.16) + cart_attachment_x(0.25) − cart_front_surface_x(0.25)` |

实现：`HierarchicalVelocityAction.__init__` 里一次算好并缓存 `self.spawn_clearance`
（出生位姿固定 ⇒ 逐回合不变），奖励函数里只做 `threshold = spawn_margin * term.spawn_clearance`。
`spawn_clearance` 用的就是 `reset_towing_episode` 摆位几何的同一个解析解
（`attachment_horizontal_gap` 勾股解 + 四个表面常量）。

**为什么留 15% 余量（`spawn_margin = 0.85 < 1`）**：若取 1.0，出生瞬间缺口恰好为 0，
任何沉降/微动都会在出生那一拍开始扣分，破坏本仓库已被测试钉住的不变量
「`min_clearance` 在 spawn 精确为 0」（用户 2026-09-23 要求初始不生效）。

### 2.2 逐行数值表（离线复算，见 §5 的自动化核对）

绳行（`compliant` / `inextensible`，L 区间 0.5–1.5 m，spawn_ratio 0.8）：
**阈值 ≈ 0.62–0.67 · L**（L=0.5 → 0.310 m，L=1.5 → 1.012 m）。

| row | L (m) | 出生间隙 (m) | 阈值 = 0.85×间隙 (m) | 阈值/L |
|---|---|---|---|---|
| 0 | 0.5000 | 0.3646 | 0.3099 | 0.6198 |
| 1 | 0.5526 | 0.4106 | 0.3490 | 0.6316 |
| 2 | 0.6053 | 0.4559 | 0.3875 | 0.6402 |
| 3 | 0.6579 | 0.5006 | 0.4255 | 0.6468 |
| 4 | 0.7105 | 0.5449 | 0.4632 | 0.6519 |
| 5 | 0.7632 | 0.5889 | 0.5005 | 0.6559 |
| 6 | 0.8158 | 0.6326 | 0.5377 | 0.6591 |
| 7 | 0.8684 | 0.6761 | 0.5747 | 0.6618 |
| 8 | 0.9211 | 0.7195 | 0.6115 | 0.6640 |
| 9 | 0.9737 | 0.7627 | 0.6483 | 0.6658 |
| 10 | 1.0263 | 0.8058 | 0.6849 | 0.6673 |
| 11 | 1.0789 | 0.8488 | 0.7214 | 0.6687 |
| 12 | 1.1316 | 0.8917 | 0.7579 | 0.6698 |
| 13 | 1.1842 | 0.9345 | 0.7943 | 0.6708 |
| 14 | 1.2368 | 0.9773 | 0.8307 | 0.6716 |
| 15 | 1.2895 | 1.0200 | 0.8670 | 0.6724 |
| 16 | 1.3421 | 1.0626 | 0.9032 | 0.6730 |
| 17 | 1.3947 | 1.1053 | 0.9395 | 0.6736 |
| 18 | 1.4474 | 1.1478 | 0.9757 | 0.6741 |
| 19 | 1.5000 | 1.1904 | 1.0118 | 0.6746 |

杆行（`rigid`，L 区间 0.5–1.0 m，spawn_ratio **1.0**）：**永不触发**（见 §2.4）。

| row | L (m) | 出生间隙 (m) | 阈值 = 0.85×间隙 (m) | 阈值/L |
|---|---|---|---|---|
| 0 | 0.5000 | 0.4727 | 0.4018 | 0.8036 |
| 1 | 0.5263 | 0.5006 | 0.4255 | 0.8085 |
| 2 | 0.5526 | 0.5283 | 0.4491 | 0.8126 |
| 3 | 0.5789 | 0.5559 | 0.4725 | 0.8162 |
| 4 | 0.6053 | 0.5834 | 0.4959 | 0.8193 |
| 5 | 0.6316 | 0.6108 | 0.5192 | 0.8220 |
| 6 | 0.6579 | 0.6381 | 0.5423 | 0.8244 |
| 7 | 0.6842 | 0.6653 | 0.5655 | 0.8265 |
| 8 | 0.7105 | 0.6924 | 0.5885 | 0.8283 |
| 9 | 0.7368 | 0.7195 | 0.6115 | 0.8300 |
| 10 | 0.7632 | 0.7465 | 0.6345 | 0.8314 |
| 11 | 0.7895 | 0.7735 | 0.6574 | 0.8328 |
| 12 | 0.8158 | 0.8004 | 0.6803 | 0.8339 |
| 13 | 0.8421 | 0.8273 | 0.7032 | 0.8350 |
| 14 | 0.8684 | 0.8541 | 0.7260 | 0.8360 |
| 15 | 0.8947 | 0.8809 | 0.7488 | 0.8369 |
| 16 | 0.9211 | 0.9077 | 0.7716 | 0.8377 |
| 17 | 0.9474 | 0.9345 | 0.7943 | 0.8384 |
| 18 | 0.9737 | 0.9612 | 0.8170 | 0.8391 |
| 19 | 1.0000 | 0.9879 | 0.8398 | 0.8398 |

### 2.3 语义

- **绳行**：牵引段绳绷直 ⇒ 3D 挂点距 = L ⇒ 间隙 ≈ L ≫ 0.62–0.67·L，**不触发**；
  停车后车斗逼近、间隙穿过阈值时才开始线性出力 → 这正是用户要的「鼓励停车后继续走几步」
  的梯度（走得越近，缺口越大、惩罚越大）。
- **杆行**：连杆把挂点距固定在 `L`（双边约束）⇒ 间隙恒等于出生间隙 ⇒ 阈值 = 0.85 × 间隙
  < 间隙 ⇒ **永不触发**（物理正确：杆不会缩短，不存在「被拉太近」）。

### 2.4 权重的标定依据

`RewardManager.compute()` 每步代价 = `weight × func × step_dt`（`step_dt = 0.005 × 10 = 0.05 s`）：

| weight | 超出阈值 0.2 m 时每步代价 | 相对 `tracking_velocity` 满额（+1.0 × 0.05 = **+0.05/步**） |
|---|---|---|
| −2.0（旧） | **−0.02/步** | 40% —— 停车段惯性/车重面前太弱 |
| **−5.0（新）** | **−0.05/步** | **100%** —— 有动机但不压倒跟踪项 |

### 2.5 杆行「永不触发」的余量（离线核算）

| 量 | 数值 |
|---|---|
| 出生高差 `Δz` | 0.170 m |
| 杆行间隙在 `Δz ∈ [0.05, 0.25]` 上相对阈值的最小余量 | **+0.0337 m**（最短杆 L=0.5、Δz=0.25 处） |
| 触发所需 `Δz*`（解 `sqrt(L²−Δz²)+base_offset = 0.85×出生间隙`） | L=0.5 → **0.301 m**；L=1.0 → **0.547 m** |
| 即：最短杆也要 Δz 再大 | **+0.131 m**（杆越长要求越大） |

`Δz` 是两个挂点的高度差，由两体的地面接触与腿长决定；且刚性连杆自身会通过压小水平间隙来
抵抗 Δz 增大 ⇒ 上述 `Δz*` 实际不可达。

---

## 3. 与测试台口径的差异（**本轮不改测试台**）

`imgo2_rl/scripts/towing/play_towing_test.py` 的 `DEFAULT_THRESHOLDS["gap_margin_limit_m"] = 0.10`
是测试台**自己的判据**（`classify_case` 里「停车段最小几何间隙落在 `(0, 0.10] m` ⇒ `stop_margin_low`」），
量的是**绝对** 0.10 m，与训练侧新的「出生几何 × 0.85」阈值**不同口径**。
两者都不应互相代入：训练侧阈值逐 env 不同（绳 0.310–1.012 m、杆 0.402–0.840 m），
测试台仍是单值 0.10 m。用户明确本轮**不改**测试台侧判据。

---

## 4. 改动文件

| 文件 | 改了什么 |
|---|---|
| `imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/upper_env_cfg.py` | 删 `extra_distance` 注册；`min_clearance` → `weight=-5.0, params={"spawn_margin": 0.85, "softness": 0.02}`；相关 docstring/注释（含 `post_stop_allowance_m` 目前未接入奖励） |
| `imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/upper_mdp.py` | `min_clearance_violation(env, spawn_margin, softness=0.02)` 新阈值（含完整推导 docstring）；`__init__` 新增 `self.spawn_clearance`；`post_stop_distance` / `post_stop_towing_force` docstring 更新 |
| `imgo2_rl/scripts/rl_lab/rl_lab/runners/towing_on_policy_runner.py` | `--compact-log` 分项 picks：`extra_distance` → `min_clearance` |
| `imgo2_rl/tests/test_towing_upper_rl_contract.py` | 改写 2 条既有 `min_clearance` 测试、更新 2 条 post_stop 测试，新增 4 条（见 §5） |
| `README.md` | 问题表 TOW-24 + 维护记录 |
| `docs/towing_upper_two_head_impl_2026-10-10.md` | `post_stop_allowance_m` 说明改为「目前未接入奖励」 |
| 本文 + 8 份旧记录 | 加「2026-10-10 变更」注（口径变更提示） |

---

## 5. 验证（离线）

- `python3 -m pytest imgo2_rl/tests -q` ⇒ **526 passed**（基线 521 + 净新增 5 条）。
- 新增/改写 7 条契约测试（`imgo2_rl/tests/test_towing_upper_rl_contract.py`）：
  1. `test_min_clearance_reward_uses_spawn_geometry_and_is_gated` —— 新权重 `-5.0`、params 为
     `spawn_margin`/`softness`、`threshold = spawn_margin * term.spawn_clearance`、
     `spawn_clearance` 的构造式、无小车屏蔽；`ratio`/旧注册**不得回潮**；
     余量理由与权重标定依据必须在 docstring/注释里；
  2. `test_min_clearance_threshold_formula_matches_the_grid_row_by_row` —— **逐类型逐行**复算
     （绳/杆 20 行 + `inextensible`），阈值/L 必须落在 0.61–0.68（绳）/0.79–0.85（杆）；
     端点 0.310 / 1.012 / 0.402 / 0.840 m；
  3. `test_min_clearance_is_inactive_at_spawn_for_every_grid_row` —— 断言 `spawn_margin < 1`，
     且**所有类型 × 所有行**的出生间隙都高于阈值（spawn 处 `func` 精确为 0）；
  4. `test_min_clearance_threshold_increases_with_length` —— 阈值随 L 单调递增；
  5. `test_min_clearance_never_fires_for_rigid_rows` —— 杆行永不触发（恒等式 + Δz 扫描 +
     触发所需 Δz* 的二分核算）；
  6. `test_extra_distance_reward_term_is_removed_but_helper_survives` —— **AST** 断言
     `UpperRewardsCfg` 类体里没有 `extra_distance`、cfg 里没有 `post_stop_distance` 的 `RewTerm`；
     `upper_mdp` 里 `post_stop_distance(env, post_stop_allowance_m=0.0)` 仍在，并把函数体从源码里
     抽出来在 stub term 上**实跑**（`relu(x − x_stop − allowance)` 的 0.3 / 0.0 / 0.2 三个读数）；
  7. `test_min_clearance_doc_table_matches_the_source_constants` —— **把本文 §2.2 的 40 行表格解析出来，
     用同一套源码常量复算到 4 位小数**（文档漂移会被这条测试先抓到）。
- **文档表格与源码逐项一致（自动化）**：第 7 条测试解析本文 §2.2 的两张表并逐项复算（L / 出生间隙 /
  阈值 / 阈值÷L），另跑离线脚本核对 `spawn_margin=0.85, Δz=0.1700, base_offset=0.0025,
  绳 spawn_ratio=0.8, 杆=1.0`，40 行数值与表**逐项一致**。
- `python3 -m py_compile`：改动文件全部通过。
- `git diff --check` 干净；`git ls-files -i -c --exclude-standard` = 0 条。

---

## 6. 未验证（**待训练机验证**）

1. **仿真一次都没跑**：本机无 Isaac Lab。`spawn_clearance` 在 `__init__` 里读
   `self._asset.cfg.init_state.pos[2]` / `self._cart.cfg.init_state.pos[2]`（与
   `reset_towing_episode` 读 `default_root_state[:, 2]` 同值），但**运行时是否真的逐 env 等于
   出生那一拍的 `rope_state[:, 0]`** 必须实跑核对（建议在 reset 后打一拍
   `Episode_Reward/min_clearance`：若出生就非 0，说明缓存与实测几何有偏差）。
2. **训练行为**：权重 −5.0 与「0.85×出生间隙」阈值在停车段是否真的给出「继续走几步」的梯度、
   会不会在牵引段误触发（绳绷直时理论间隙 ≈L），只有 rollout / TensorBoard 能回答。
   要看的量：`Episode_Reward/min_clearance`（按相位分开更好）、`obs_stop_reached`、
   `stop_towing_force`、`tracking_velocity`。
3. **旧读数不可比（口径变更）**：`docs/towingdata/2026-10-09_*` 归档里的 LOW 计数
   （`stop_margin_low`）用的是测试台**绝对 0.10 m** 判据，且当时的训练侧 `min_clearance`
   是 `ratio × 连接长度`；**归档 LOW 计数与新口径不可比**，只能说"旧归档是旧口径"。同理
   `docs/towing_play_2026-09-25.md` 里 `Episode_Reward/min_clearance = −8.7e−5` 之类的读数
   是旧权重的，不能当新权重的基线。
4. **`extra_distance` 删除后的停车段**：`stop_towing_force` 是唯一停车项，其量级（−1.0）是否需要
   跟着调，需实跑读数再定。
5. **`post_stop_allowance_m` 复用**：函数与形参保留但未接入奖励，没有任何运行时证据；
   日后重新启用需一起看 `stop_command_ramp_s`（动作侧 ramp）。

---

## 7. 待用户决定

- 是否把测试台的 `gap_margin_limit_m`（绝对 0.10 m）也改成与训练侧同口径（本轮按用户要求不动）。
- `stop_towing_force`（−1.0）在删掉 `extra_distance` 之后是否要提权。
