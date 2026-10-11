# 奖励改动记录：删除 `extra_distance` + `min_clearance` 改出生几何阈值（绝对死区）并提权（2026-10-10）

用户 2026-10-10 批准的奖励改动，落地在
[`upper_env_cfg.py`](../imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/upper_env_cfg.py) 与
[`upper_mdp.py`](../imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/upper_mdp.py)。
`min_clearance` 的口径当天走了两步：**先**由 `ratio(0.25) × 连接长度` 改成 `spawn_margin(0.85) × 出生间隙`
（相对余量），**同日晚**按用户决定再微调为 **`出生间隙 − deadband_m`（绝对死区，默认 0.02 m）**。
本文只描述**当前口径**；`spawn_margin` 那一步只作历史出现在 §2.1 的沿革表里。

**本机没有 Isaac Lab**：以下全部结论只由离线 `pytest` / AST / 纯标准库复算得出；
凡只有训练才能确认的一律标「**待训练机验证**」（见 §6）。README 问题表条目 = **TOW-24**。

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

## 2. 改动 ②：`min_clearance` 改成「出生几何 − 绝对死区」阈值

### 2.1 新口径

```
threshold = spawn_clearance − deadband_m                # deadband_m = 0.02 m（RewTerm params）
spawn_clearance = sqrt(initial_attachment_distance(类型, L)² − Δz²) + base_offset
```

| 量 | 取值 | 来源（源码，不手抄数字） |
|---|---|---|
| `deadband_m` | **0.02 m**（绝对死区） | `UpperRewardsCfg.min_clearance` 的 `params` |
| `softness` | 0.02（不变） | 同上 |
| `weight` | **−5.0**（原 −2.0） | 同上 |
| `initial_attachment_distance` | 绳 = `SLACK_RATIO(0.8) × L`；杆 = `L` | `mdp/connection_grid.py`；逐 env 已落在 `term.initial_distance`（`__init__` 里由 `env_spec` 算出） |
| `Δz` | **0.17 m** | 机器人 `init_state.pos[2] = 0.35` − cart `init_state.pos[2] = 0.18`（两个挂点的 z 偏移都是 0） |
| `base_offset` | **0.0025 m** | `−robot_rear_surface_x(0.1575) − robot_attachment_x(−0.16) + cart_attachment_x(0.25) − cart_front_surface_x(0.25)` |

实现：`HierarchicalVelocityAction.__init__` 里一次算好并缓存 `self.spawn_clearance`
（出生位姿固定 ⇒ 逐回合不变），奖励函数里只做 `threshold = term.spawn_clearance - deadband_m`。
`spawn_clearance` 用的就是 `reset_towing_episode` 摆位几何的同一个解析解
（`attachment_horizontal_gap` 勾股解 + 四个表面常量）。

**口径沿革（都在 2026-10-10 当天）**：

| 时间 | 阈值 | 备注 |
|---|---|---|
| 2026-09-23 → 2026-10-09 | `ratio(0.25) × 连接长度`（`ratio` 一路从 0.6 降到 0.25） | 权重 −2.0；作废 |
| 2026-10-10 早 | `spawn_margin(0.85) × 出生间隙` | 相对余量；权重 −5.0；当天即被替代 |
| **2026-10-10 晚（现行）** | **`出生间隙 − deadband_m(0.02)`** | 绝对死区；权重 −5.0 |

**为什么从「相对余量 0.85」改成「绝对死区 2 cm」**（用户 2026-10-10 决定，三条理由）：

1. **语义更贴**：用户要的就是「**不许比出生时更近**」。`出生间隙 − 死区` 直接表达它；
   `0.85 × 出生间隙` 表达的是"比出生近 15%"，是个没有物理含义的相对量。
2. **量级正确**：`spawn_clearance` 是**解析**出生间隙（勾股解 + 四个表面常量），与仿真里的实际
   间隙差**几毫米**（落地/穿透/初始沉降）⇒ 绝对 2 cm 死区正好吸收这个量级。相对余量却随 L 放大：
   L=1.5 的绳行余量 `0.15 × 1.1904 = 0.179 m`，比需要的量大一个数量级，等于给长绳行白开几十厘米的
   "可以靠近"口子。
3. **不需要大余量（纠正旧顾虑）**：出生点在 lane 的**后向平段**上——`slope_geometry` 的剖面从
   **+2.25 m** 才起坡（平地 2.25 m → 上坡 3 m → 坡顶 0.75 m → 下坡 3 m → 平地 2.25 m）。
   所以**出生与停车都发生在平地、没有重力驱动** ⇒ 间隙只受机器人动作影响，**不存在被动的间隙漂移**。
   旧顾虑「坡上会溜车所以要留大余量」在本任务里不成立（停车滑行段在平地上由黏性轮阻耗散、车斗自行停住），
   因此 2 cm 的死区就够。

**「高于阈值精确为 0」的既有行为原样保留**：实现是**带截断的 softplus**，不是裸 softplus——
`bias = softplus(0) * softness`（= 0.6931471805599453 × 0.02），
`violation = relu(softplus((threshold − clearance)/softness) × softness − bias)`。
减去 `bias` 使缺口 ≤ 0 时**精确为 0**、阈值上方没有 softplus 尾巴；只在阈值下方做平滑过渡。
本次口径变更**只改阈值那一行**，这段实现一个字没动（有契约测试钉住）。

### 2.2 逐行数值表（离线复算，见 §5 的自动化核对）

绳行（`compliant` / `inextensible`，L 区间 0.5–1.5 m，spawn_ratio 0.8）：
**阈值 = 出生间隙 − 0.02 = 0.689–0.780 · L**（L=0.5 → 0.3446 m = 0.9451 × 出生间隙，
L=1.5 → 1.1704 m = 0.9832 × 出生间隙）。

| row | L (m) | 出生间隙 (m) | 阈值 = 出生间隙 − 0.02 (m) | 阈值/L |
|---|---|---|---|---|
| 0 | 0.5000 | 0.3646 | 0.3446 | 0.6892 |
| 1 | 0.5526 | 0.4106 | 0.3906 | 0.7068 |
| 2 | 0.6053 | 0.4559 | 0.4359 | 0.7202 |
| 3 | 0.6579 | 0.5006 | 0.4806 | 0.7305 |
| 4 | 0.7105 | 0.5449 | 0.5249 | 0.7388 |
| 5 | 0.7632 | 0.5889 | 0.5689 | 0.7454 |
| 6 | 0.8158 | 0.6326 | 0.6126 | 0.7509 |
| 7 | 0.8684 | 0.6761 | 0.6561 | 0.7555 |
| 8 | 0.9211 | 0.7195 | 0.6995 | 0.7594 |
| 9 | 0.9737 | 0.7627 | 0.7427 | 0.7627 |
| 10 | 1.0263 | 0.8058 | 0.7858 | 0.7656 |
| 11 | 1.0789 | 0.8488 | 0.8288 | 0.7681 |
| 12 | 1.1316 | 0.8917 | 0.8717 | 0.7703 |
| 13 | 1.1842 | 0.9345 | 0.9145 | 0.7722 |
| 14 | 1.2368 | 0.9773 | 0.9573 | 0.7740 |
| 15 | 1.2895 | 1.0200 | 1.0000 | 0.7755 |
| 16 | 1.3421 | 1.0626 | 1.0426 | 0.7769 |
| 17 | 1.3947 | 1.1053 | 1.0853 | 0.7781 |
| 18 | 1.4474 | 1.1478 | 1.1278 | 0.7792 |
| 19 | 1.5000 | 1.1904 | 1.1704 | 0.7803 |

杆行（`rigid`，L 区间 0.5–1.0 m，spawn_ratio **1.0**）：**在出生高差上永不触发**（见 §2.5），
阈值 = 出生间隙 − 0.02 = **0.905–0.968 · L**。

| row | L (m) | 出生间隙 (m) | 阈值 = 出生间隙 − 0.02 (m) | 阈值/L |
|---|---|---|---|---|
| 0 | 0.5000 | 0.4727 | 0.4527 | 0.9054 |
| 1 | 0.5263 | 0.5006 | 0.4806 | 0.9131 |
| 2 | 0.5526 | 0.5283 | 0.5083 | 0.9198 |
| 3 | 0.5789 | 0.5559 | 0.5359 | 0.9257 |
| 4 | 0.6053 | 0.5834 | 0.5634 | 0.9308 |
| 5 | 0.6316 | 0.6108 | 0.5908 | 0.9354 |
| 6 | 0.6579 | 0.6381 | 0.6181 | 0.9394 |
| 7 | 0.6842 | 0.6653 | 0.6453 | 0.9431 |
| 8 | 0.7105 | 0.6924 | 0.6724 | 0.9463 |
| 9 | 0.7368 | 0.7195 | 0.6995 | 0.9493 |
| 10 | 0.7632 | 0.7465 | 0.7265 | 0.9519 |
| 11 | 0.7895 | 0.7735 | 0.7535 | 0.9544 |
| 12 | 0.8158 | 0.8004 | 0.7804 | 0.9566 |
| 13 | 0.8421 | 0.8273 | 0.8073 | 0.9586 |
| 14 | 0.8684 | 0.8541 | 0.8341 | 0.9605 |
| 15 | 0.8947 | 0.8809 | 0.8609 | 0.9622 |
| 16 | 0.9211 | 0.9077 | 0.8877 | 0.9638 |
| 17 | 0.9474 | 0.9345 | 0.9145 | 0.9653 |
| 18 | 0.9737 | 0.9612 | 0.9412 | 0.9667 |
| 19 | 1.0000 | 0.9879 | 0.9679 | 0.9679 |

**40 行全部满足**：阈值 = 出生间隙 − 0.02（4 位小数逐项）、阈值严格小于出生间隙（spawn 处精确为 0，
余量恰为 `deadband_m`）、阈值随 L 单调递增、阈值恒 > 0（最小 = 最短绳行 0.3646 − 0.02 = **0.3446 m**）。
以上四条都有契约测试（§5）。

### 2.3 语义

- **绳行**：牵引段绳绷直 ⇒ 3D 挂点距 = L > 0.8·L ⇒ 间隙 = `sqrt(L²−Δz²)+base_offset` > 出生间隙
  > 阈值，**不触发**；绳一松、车斗/机器人逼近穿过阈值时才开始线性出力 → 这正是用户要的
  「不许比出生时更近」+「停车后继续走几步」的梯度（走得越近，缺口越大、惩罚越大）。
- **杆行**：连杆把挂点距固定在 `L`（双边约束）⇒ 在出生高差上间隙恒等于出生间隙 ⇒ 阈值 = 出生间隙
  − 0.02 < 间隙 ⇒ **永不触发**（物理正确：杆不会缩短，不存在「被拉太近」）。但与旧相对余量相比，
  对 Δz 漂移的稳健性明显变弱，见 §2.5。

### 2.4 停车段为什么**天然生效**（新事实，本轮最重要的一条）

用户给的标定：**停车瞬间的实测间隙 ≈ 0.64 · L**，而新阈值 ≈ **0.945 × 出生间隙 ≈ 0.69 · L**
⇒ **停车段的间隙天然落在阈值之下约 `0.05 · L`**，所以该项**在停车段自动生效**：

- L=1.0 时缺口约 0.05 m ⇒ 每步 `−5 × 0.05 × 0.05 ≈ −0.0125`（线性铰链值），
  与 `tracking_velocity` 满额 `+1.0 × 0.05 = +0.05/步` **同量级**；
- 因此**不需要**再单独做「用停车瞬间间隙当参考」的方案——「停车期间也保持间隙」已经由本项给出梯度。

**离线佐证（比用户给的标定更细，用归档实测数据复算）**：把
`docs/towingdata/2026-10-09_necessity_800{,_noload}/report.json` 的 `cases[].metrics.stop` 按
compliant 行聚合（每行 n≈14–16，共 316 / 277 个有效 case），用新阈值逐行比对：

| L (m) | 新阈值 (m) | 停车瞬间间隙中位 (m) | 缺口 (m) | 每步代价（线性铰链） | 滑行段最小间隙中位 (m) | 缺口 (m) | 每步代价 |
|---|---|---|---|---|---|---|---|
| 0.6000 | 0.4314 | 0.3273 | +0.1040 | −0.0260 | 0.1200 | +0.3113 | −0.0778 |
| 0.6947 | 0.5116 | 0.4228 | +0.0888 | −0.0222 | 0.1773 | +0.3343 | −0.0836 |
| 0.7895 | 0.5908 | 0.5142 | +0.0766 | −0.0191 | 0.2579 | +0.3329 | −0.0832 |
| 0.8842 | 0.6691 | 0.6175 | +0.0516 | −0.0129 | 0.4044 | +0.2647 | −0.0662 |
| 0.9789 | 0.7469 | 0.7145 | +0.0325 | −0.0081 | 0.4890 | +0.2579 | −0.0645 |
| 1.0737 | 0.8245 | 0.8108 | +0.0136 | −0.0034 | 0.6079 | +0.2166 | −0.0541 |
| 1.1684 | 0.9016 | 0.9031 | **−0.0015** | +0.0004 | 0.6675 | +0.2342 | −0.0585 |
| 1.2000 | 0.9273 | 0.9393 | **−0.0119** | +0.0030 | 0.7120 | +0.2153 | −0.0538 |

`min_clearance` 量的是 `rope_state[:, 0]`（绳语义间隙），所以第二组「滑行段最小间隙」才是本项
积分的对象；**它在全部 20 行上都低于阈值**（缺口 +0.19 … +0.35 m），停车瞬间那一拍也只在最长的
两行（L ≈ 1.17–1.20，本归档网格的上界）才略高于阈值。**触发 case 占比（停车瞬间判据）**：
`800` 有载 **284/316**、`800_noload` **251/277**（≈ 80–90%）。

口径提醒：

1. **归档网格是 L 0.6–1.2 m，现行网格是 0.5–1.5 m**，行号不对应，这里只作**量级佐证**、不做逐行核对；
2. 表里的「每步代价」是**线性铰链值**；`softness = 0.02` 的截断 softplus 会压掉一些
   （缺口 0.05 m 时实际 = 0.0377 m ≈ **0.75 ×** 线性值 ⇒ 每步 ≈ −0.0094；缺口 0.10 m 时 ≈0.87×、
   0.20 m 时 ≈0.93×、≥0.30 m 时 ≈0.95×）。量级结论不变；
3. 归档的 `stop_margin_low` 判据是测试台**绝对 0.10 m**，与本项训练侧阈值**不同口径**（§3），
   但归档里记录的 `clearance_at_stop_m` / `min_clearance_coast_m` 是**原始几何量**，可以拿来对
   本项阈值复算。

**备选方案（本轮不实现，只在训练机实测本项在停车段不生效时才启用）**：若训练机实测停车段间隙
反而**高于**阈值（例如最短绳行 L=0.5：阈值 0.3446 m，而 0.70 · L = 0.35 m 就高于阈值——归档里
最长两行已经出现这种"擦边不触发"），则改为在 `HierarchicalVelocityAction.process_actions`
（20 Hz）里记录**停车那一拍的间隙** `gap_at_stop`（需识别"停下"的时刻，例如 `obs_stop_reached`
或指令归零那一拍），把阈值改成 `gap_at_stop − deadband_m`。本机无 Isaac Lab，实测未做。

### 2.5 杆行「永不触发」的余量（离线核算）——本口径的**代价**

| 量 | 旧口径（0.85 相对余量） | **新口径（绝对死区 0.02）** |
|---|---|---|
| 出生高差 `Δz` | 0.170 m | 0.170 m |
| 杆行间隙恒比阈值大 | `0.15 × 间隙`（最短杆 **+0.071 m**） | **`deadband_m` = +0.02 m** |
| 触发所需 `Δz*`：L=0.5 / L=1.0 | 0.301 m / 0.547 m | **0.2175 m / 0.2606 m** |
| 即最短杆只需 Δz 再大 | +0.131 m | **+0.0475 m** |
| Δz 扫描（出生 ±0.04 m，即 0.13–0.21 m）最小余量 | +0.034 m（扫描到 0.25） | **+0.0036 m**（最短杆 L=0.5、Δz=0.21） |

结论不变但**稳健性变弱**：「杆行永不触发」在物理上仍成立（Δz 由两体的地面接触与腿长决定，
刚性连杆自身还会通过压小水平间隙来抵抗 Δz 增大），但新口径下最短杆只要 Δz 比出生值再大
**4.75 cm** 就会触发（旧口径是 13.1 cm）。**训练机若看到最短杆行 `min_clearance` 非 0，先查 Δz。**

### 2.6 权重的标定依据

`RewardManager.compute()` 每步代价 = `weight × func × step_dt`（`step_dt = 0.005 × 10 = 0.05 s`）：

| weight | 超出阈值 0.2 m 时每步代价（线性铰链） | 相对 `tracking_velocity` 满额（+1.0 × 0.05 = **+0.05/步**） |
|---|---|---|
| −2.0（旧） | **−0.02/步** | 40% —— 停车段惯性/车重面前太弱 |
| **−5.0（现行，未变）** | **−0.05/步** | **100%** —— 有动机但不压倒跟踪项 |

（截断 softplus 把 0.2 m 缺口压到 0.186 ⇒ 实际 ≈ −0.047/步，仍与满额同量级。）

---

## 3. 与测试台口径的差异（**本轮不改测试台**）

`imgo2_rl/scripts/towing/play_towing_test.py` 的 `DEFAULT_THRESHOLDS["gap_margin_limit_m"] = 0.10`
是测试台**自己的判据**（`classify_case` 里「停车段最小几何间隙落在 `(0, 0.10] m` ⇒ `stop_margin_low`」），
量的是**绝对** 0.10 m，与训练侧新的「出生间隙 − 0.02」阈值**不同口径**。
两者都不应互相代入：训练侧阈值逐 env 不同（绳 0.3446–1.1704 m、杆 0.4527–0.9679 m），
测试台仍是单值 0.10 m。用户明确本轮**不改**测试台侧判据。

---

## 4. 改动文件

| 文件 | 改了什么 |
|---|---|
| `imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/upper_env_cfg.py` | 删 `extra_distance` 注册；`min_clearance` → `weight=-5.0, params={"deadband_m": 0.02, "softness": 0.02}`（`spawn_margin` 删除）；相关注释/理由 |
| `imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/upper_mdp.py` | `min_clearance_violation(env, deadband_m, softness=0.02)`，函数内 `threshold = term.spawn_clearance - deadband_m`（其余实现未动）；`__init__` 的 `self.spawn_clearance` 缓存与注释；完整推导 docstring（含语义/几毫米解析差/后向平段无被动漂移/停车段天然生效 + `gap_at_stop` 备选） |
| `imgo2_rl/scripts/rl_lab/rl_lab/runners/towing_on_policy_runner.py` | `--compact-log` 分项 picks：`extra_distance` → `min_clearance`（**本轮未改**） |
| `imgo2_rl/tests/test_towing_upper_rl_contract.py` | 改写 6 条 `min_clearance` 契约测试（新增 1 条：真函数实跑 + `deadband_m > 0`），见 §5 |
| `imgo2_rl/tests/test_towing_force_rate.py` | 「既有项未动」断言跟随新 params（`deadband_m` 0.02） |
| `README.md` | 顶部核对行、问题表 TOW-24、维护记录追加一行 |
| `docs/towing_upper_two_head_impl_2026-10-10.md` | `post_stop_allowance_m` 说明（**本轮未改**） |
| 本文 + 10 份旧记录 | 顶部「2026-10-10 变更」注里的 `min_clearance` 口径改为「出生间隙 − `deadband_m`」 |

---

## 5. 验证（离线）

- `python3 -m pytest imgo2_rl/tests -q` ⇒ **551 passed**（上一基线 550 + 1 条新增）。
- `min_clearance` 契约测试（`imgo2_rl/tests/test_towing_upper_rl_contract.py`）覆盖用户点名的六条：
  1. **`deadband_m > 0`** —— `_spawn_gap_constants()` 从 `upper_env_cfg` 源码 regex 读出参数并断言
     `deadband_m == 0.02` 且 `> 0`（`test_min_clearance_deadband_positive_and_spawn_value_is_exactly_zero`）；
  2. **spawn 处精确为 0 且余量恰为 `deadband_m`** —— 把源码里的 `min_clearance_violation` **抽出来
     在 stub term 上实跑**（`torch` 可用时）：`clearance == spawn_clearance` 给 **0.0**、
     `clearance == 阈值`（用函数内同一步 float32 运算取边界）也给 **0.0**（无 softplus 尾巴）、
     再低 1 µm 才 > 0；另有 `test_min_clearance_is_inactive_at_spawn_for_every_grid_row` 对
     **3 类型 × 20 行**断言 `出生间隙 − 阈值 == deadband_m`；
  3. **逐行逐类型阈值 = 出生间隙 − 0.02** —— `test_min_clearance_threshold_formula_matches_the_grid_row_by_row`
     对绳/杆/`inextensible` 各 20 行断言 `threshold == spawn_gap − 0.02`（`places=12`）、
     阈值/L 落在 0.68–0.79（绳）/0.90–0.97（杆），端点 0.3446 / 1.1704 / 0.4527 / 0.9679 m；
  4. **单调递增** —— `test_min_clearance_threshold_increases_with_length`（20 档 + 端点）；
  5. **杆行永不触发** —— `test_min_clearance_never_fires_for_rigid_rows`（恒等式 + Δz ±0.04 扫描 +
     `Δz*` 二分：0.2175–0.2606 m，余量 +0.0475 … +0.0906 m 都断言在 0.045–0.10 之间）；
  6. **阈值恒 > 0** —— 第 3 条里逐行 `assertGreater(threshold, 0.0)`（最小 = 0.3446 m）。
- **文档表格与源码逐项一致（自动化）**：`test_min_clearance_doc_table_matches_the_source_constants`
  解析本文 §2.2 的两张表并逐项复算（L / 出生间隙 / 阈值 / 阈值÷L）到 4 位小数；
  离线脚本另核对 `deadband_m=0.02, Δz=0.1700, base_offset=0.0025, 绳 spawn_ratio=0.8, 杆=1.0`，
  40 行数值与表**逐项一致**。
- 源码守卫：`def min_clearance_violation(env, deadband_m, softness=0.02):`、
  `threshold = term.spawn_clearance - deadband_m`、`params={"deadband_m": 0.02, "softness": 0.02})`、
  `weight=-5.0`、截断 softplus 的 `bias = 0.6931471805599453 * softness` 必须在；
  `"spawn_margin"` 不得回潮（cfg 里连字符串都不许有）、`threshold = spawn_margin * ...` /
  `def min_clearance_violation(env, spawn_margin` / `"ratio": 0.25` / 旧 `weight=-2.0` 注册行不得回潮。
- `python3 -m py_compile`：改动文件全部通过。
- `git diff --check` 干净；`git ls-files -i -c --exclude-standard` = 0 条；工作区只有未跟踪的
  `imgo2_rl/nohup.out`。

---

## 6. 未验证（**待训练机验证**）

1. **仿真一次都没跑**：本机无 Isaac Lab。`spawn_clearance` 在 `__init__` 里读
   `self._asset.cfg.init_state.pos[2]` / `self._cart.cfg.init_state.pos[2]`（与
   `reset_towing_episode` 读 `default_root_state[:, 2]` 同值），但**运行时是否真的逐 env 等于
   出生那一拍的 `rope_state[:, 0]`** 必须实跑核对（建议在 reset 后打一拍
   `Episode_Reward/min_clearance`：若出生就非 0，说明缓存与实测几何有偏差，也说明 2 cm 死区
   是否够吸收那几毫米的解析-仿真差）。
2. **训练行为**：权重 −5.0 与「出生间隙 − 0.02」阈值在停车段是否真的给出「继续走几步」的梯度、
   会不会在牵引段误触发（绳绷直时理论间隙 > 出生间隙），只有 rollout / TensorBoard 能回答。
   要看的量：`Episode_Reward/min_clearance`（按相位分开更好）、`obs_stop_reached`、
   `stop_towing_force`、`tracking_velocity`。
3. **停车段「天然生效」待实测确认**：§2.4 的结论由**旧归档（L 0.6–1.2 m 网格、旧口径训练）**
   离线复算 + 用户给的 0.64·L 标定得出，不是新口径下的实跑读数。训练机要专门核对
   **最短绳行（L≈0.5–0.6）停车瞬间**是否落在阈值之上（归档里 0.70·L 已经高于 0.689·L）；
   若该项在停车段整体不生效，再按 §2.4 的**备选**改成记录 `gap_at_stop` 作参考（本轮不实现）。
4. **杆行余量变小**：Δz 从出生值漂 4.75 cm（最短杆）就会触发，只有实跑能给出真实 Δz 波动；
   若最短杆行 `min_clearance` 长期非 0，要么确认 Δz 真实漂移、要么把 `deadband_m` 调大。
5. **旧读数不可比（口径变更两次）**：`docs/towingdata/2026-10-09_*` 归档里的 LOW 计数
   （`stop_margin_low`）用的是测试台**绝对 0.10 m** 判据，且当时的训练侧 `min_clearance`
   是 `ratio × 连接长度`；**归档 LOW 计数与新口径不可比**。同理
   `docs/towing_play_2026-09-25.md` 里 `Episode_Reward/min_clearance = −8.7e−5` 之类的读数
   是旧权重的，不能当新权重的基线。
6. **`extra_distance` 删除后的停车段**：`stop_towing_force` 是唯一停车项，其量级（−1.0）是否需要
   跟着调，需实跑读数再定。
7. **`post_stop_allowance_m` 复用**：函数与形参保留但未接入奖励，没有任何运行时证据；
   日后重新启用需一起看 `stop_command_ramp_s`（动作侧 ramp）。

---

## 7. 待用户决定

- 是否把测试台的 `gap_margin_limit_m`（绝对 0.10 m）也改成与训练侧同口径（本轮按用户要求不动）。
- `stop_towing_force`（−1.0）在删掉 `extra_distance` 之后是否要提权。
- `deadband_m` 是否需要按行区分（现在 40 行统一 2 cm）：最短绳行的死区占出生间隙 5.5%、
  长绳行只占 1.7%——若训练机显示最短绳行误罚多，可考虑让它按 L 缩放（但那又回到"相对量"）。
