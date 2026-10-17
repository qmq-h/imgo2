# 奖励权重三处改动：`clearance` 提权、`action_rate` 减半、`collision` 下坡降权（2026-10-17）

用户 2026-10-17 的要求（原文）：

> `clearance` 权重提高到 −10，`action rate` 降低到 −0.05，`collision` 在下坡路段
> （只在下坡段落）降低到 −5.0 但是其他情况不变。

追问澄清后，`collision` 的下坡目标值确定为 **−1.0**（用户回复「下坡降低到 -1.0」）；
「下坡路段」的判定范围选的是 **只算坡面剖面的下坡段**（不含减速带列、不含波浪列）。

落地在 [`upper_env_cfg.py`](../../imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/upper_env_cfg.py)
与 [`upper_mdp.py`](../../imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/upper_mdp.py)，
离线守卫在 [`test_towing_reward_weights_2026_10_17.py`](../../imgo2_rl/tests/test_towing_reward_weights_2026_10_17.py)。
README 问题表条目 = **TOW-REW-01**。

**本机没有 Isaac Lab**（`upper_mdp` 顶层 import isaaclab、`isaaclab.envs` 缺 `omni.*` ⇒ 构造不了
环境）：以下量级全部是**按公式与源码常量的离线推算**，凡只有实跑才能确认的一律标「待训练机」。

---

## 0. 口径（读数字之前先看）

`RewardManager.compute()` 的每步值是

```
value = func() × weight × step_dt        step_dt = sim.dt × decimation = 0.005 × 10 = 0.05 s
```

TensorBoard 的 `Episode_Reward/<项>` 是**回合加权和 ÷ `episode_length_s`**（≈24.23 s / 485 步；训练侧 STOP 保持 10 m，2026-10-17 只改测评侧，见 TOW-EVAL-01），比每步值小约 20 倍，**不要直接与每步值比较**。
本仓库另有约定：`weight=0` 的项被
`RewardManager` 直接 `continue`（不调用、不记录），所以诊断项用 `1e-6` 而不是 0。

参照量：`tracking_velocity` 满额 = `1.0 × 1.0 × 0.05` = **+0.05/步**（唯一的正奖励）。

---

## 1. 改动总表

| 项 | 旧 | 新 | 每步值（含 step_dt） | 注册行 |
|---|---|---|---|---|
| `clearance` | `weight=-1.0` | **`weight=-10.0`** | 间隙 0.20 m ⇒ −0.347；0.10 ⇒ −1.06；0 ⇒ −2.01 | [`upper_env_cfg.py:294`](../../imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/upper_env_cfg.py#L294) |
| `action_rate` | `weight=-0.1` | **`weight=-0.05`** | 12 维各变 1 ⇒ −0.03；满幅翻转 ⇒ −0.12 | [`upper_env_cfg.py:420`](../../imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/upper_env_cfg.py#L420) |
| `collision` | `weight=-5.0`（全域） | **`weight=-5.0` + `params={"downhill_scale": 0.2}`** ⇒ 下坡有效 **−1.0** | 其他路段 −0.25；下坡段 −0.05 | [`upper_env_cfg.py:272`](../../imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/upper_env_cfg.py#L272) |

**没动的项**（用户只点了这三项）：`tracking_velocity` +1.0、`fall` −50.0、`min_clearance` −5.0、
`stop_towing_force` −1.0、`towing_force_y` −10.0、`towing_force_rate` −0.5、`feet_slide` −0.05、
`action_magnitude` −0.05、`low_level_pos_error` −0.1，以及 5 个 `1e-6` 诊断项。
（注意 `action_magnitude` 与新的 `action_rate` **恰好同为 −0.05**，别把两者的历史混在一起。）

---

## 2. `clearance`：−1.0 → −10.0

函数（`mdp.clearance_barrier`，未改）：

```
func = softplus((0.20 − 间隙) / 0.05) × cart_present        间隙 = term.rope_state[:, 0]
```

三个锚点（`step_dt = 0.05`，`weight = −10`）：

| 间隙（m） | func | 每步（新） | 每步（旧 −1.0） | 相对 `tracking_velocity` 满额 |
|---|---|---|---|---|
| 0.20（警戒线） | 0.6931 | **−0.3466** | −0.0347 | 6.9× |
| 0.10 | 2.1269 | **−1.0635** | −0.1063 | 21× |
| 0.00 | 4.0181 | **−2.0091** | −0.2009 | 40× |

**出生那一拍不会被扣分**：训练侧间距 `rope_state[:, 0]` 在出生时等于
`spawn_clearance = sqrt((spawn_ratio·L)² − Δz²) + base_offset`（`spawn_ratio` 绳 0.8 / 杆 1.0，
`Δz` = 0.17 m，`base_offset` = 0.0025 m，全部来自源码常量），最短绳行（L=0.5）**0.3646 m**、
最短杆行（L=0.5）**0.4725 m** ⇒ func 0.037 / 0.004 ⇒ 每步 **−0.019 / −0.002**。

**这条改动的副作用（明写，别事后才发现）**：`clearance` 是 softplus **软障碍、无上界**，
而 `min_clearance` 是「出生间隙 − 2 cm」的铰链（缺口 0.2 m 才 −0.05/步）。
提权 10 倍后，**间隙 < 0.18 m 时软障碍会压过** `min_clearance` 与 `tracking_velocity` 满额：

| 间隙 | `clearance` 每步 | `min_clearance` 每步（L=0.5 绳行，阈值 0.3446 m，缺口 = 0.3446−间隙） |
|---|---|---|
| 0.20 | −0.347 | 0（间隙高于阈值 ⇒ 精确 0） |
| 0.10 | −1.064 | −5 × 0.2446 × 0.05 = −0.061 |
| 0.00 | −2.009 | −5 × 0.3446 × 0.05 = −0.086 |

即接近碰撞时**真正起作用的会是 `clearance`（软障碍）**，而不是语义上更"硬"的 `min_clearance`。
这是用户「贴近就重罚」的取向，但两项权重的相对关系已与 2026-10-10 的设计说明不同 ——
**待训练机按 `Episode_Reward/clearance` 与 `min_clearance` 的实际读数决定是否回调**
（见 §5）。

---

## 3. `action_rate`：−0.1 → −0.05

函数（`mdp.action_rate_l2`，未改）：`func = Σ_i (a_i,t − a_i,t−1)²`，12 维关节残差（`clip_actions=1`
⇒ 每维 ∈ [−1, 1]，差值 ∈ [−2, 2]）。

| 情形 | func | 每步（新 −0.05） | 每步（旧 −0.1） |
|---|---|---|---|
| 12 维各变 1 | 12 | −0.030 | −0.060 |
| 从 +1 翻到 −1（Δ=±2，满幅） | 48 | **−0.120** | −0.240 |
| 各维抖动 0.3 | 1.08 | −0.0027 | −0.0054 |

旧的类注释里「±1 抖动时约 −0.24/步」对应的是**满幅翻转**那一行（func 48），不是各维变 1；
本轮注释已按这两档写清。减半的取向是让动作平滑项不再比跟踪项重 5 倍（−0.24 vs +0.05），
从而**不再顺带压制"残差去补偿底层下垂"**（`low_level_pos_error` 只有 −0.0017/步）。

---

## 4. `collision`：只在下坡段 −1.0，其他路段仍 −5.0

### 4.1 需求澄清与实现方式

用户原话是「collision 在下坡路段（只在下坡段落）降低到 −5.0」，但 `collision` **当时全域已经是
−5.0**，按字面等于不改。追问后用户给出真正想要的值：**下坡 −1.0**。

实现没有拆成第二条 `RewTerm`，而是**在 `func` 里按相位缩放**：

```python
# upper_env_cfg.py
collision = RewTerm(func=mdp.cart_collision_cost, weight=-5.0,
                    params={"downhill_scale": 0.2})

# upper_mdp.py
def cart_collision_cost(env, downhill_scale=1.0):
    cost = cart_collision(env).float()
    if downhill_scale == 1.0:
        return cost                       # 默认 ⇒ 与旧行为逐位一致
    scale = torch.where(downhill_segment(env),
                        cost.new_tensor(float(downhill_scale)), cost.new_tensor(1.0))
    return cost * scale
```

有效权重 `−5.0 × 0.2 = −1.0`（每步 −0.05 = `tracking_velocity` 满额 1 倍；其他路段 −0.25 = 5 倍）。

**为什么缩放 func 而不是再注册一条 `collision_downhill`**：

1. `Episode_Reward/collision` 保持**一条曲线**，与 2026-10-09 以来的历史 run 读数口径不变；
   拆成两条会让同一语义在 TB 与 `--compact-log` 的 picks 里分裂；
2. `weight` 仍是字面 −5.0 ⇒ 既有契约测试
   `test_collision_is_a_small_sustained_penalty_not_a_termination` 的语义不变
   （该测试本轮只把「整行字面量」断言改成「`weight=-5.0,` + `params`」两段，并新增
   `−5.0 × 0.2 == −1.0` 的数值断言，防止两个数被一起改掉）；
3. 与仓库既有的相位写法一致：`post_stop_towing_force` 也是**一项 + func 内的相位门控**
   （`elapsed_s >= stop_time_s`），而不是两条注册项。

### 4.2 「下坡段」的判定

`mdp.downhill_segment`（[`upper_mdp.py:974`](../../imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/upper_mdp.py#L974)）：

```
mask = (DOWN_START_M < lane_local_x <= FLAT_OUT_START_M) & ~terrain_speedbump & ~terrain_wave
```

- 剖面来源 `slope_geometry`：**平地 2.25 → 上坡 3 → 坡顶 0.75 → 下坡 3 → 平地 2.25**，
  所以下坡段是 lane 局部 x ∈ (6.0, 9.0] m；`DOWN_START_M`/`FLAT_OUT_START_M` 直接从
  `slope_geometry` import，**没有第二处硬编码**；
- `lane_local_x = (root_pos_w − env.scene.env_origins)[:, 0]`：出生点在 lane 原点平面上
  （`terrain_tangent_w` 恒为 +x）⇒ 第 0 分量就是 `profile_height(grade, x)` 的 `x`，
  不含弧长修正（与 `stop_reached` 用点积算 progress 不同，那里要的是行程）；
- **减速带列（列 40–49）与波浪列（列 50–59）恒为 False** ⇒ 全程 −5.0；
- 与几何定义的一致性由测试逐点核对：在 10° 剖面上，mask 必须**恒等于**
  `profile_slope_degrees(10, x) < 0`（−1.0…12.0 m、步长 0.01、1301 点）。

### 4.3 ⚠ rolling 档的已知近似（待用户决定要不要按真实坡度做）

`Imgo2-towing-upper-rolling-rl-lab` 把平滑列换成 `rolling_geometry` 的 4 座 raised-cosine 山
（宽度 1.5/2.0/1.25/2.0，起点 2.25 ⇒ 边界 2.25/3.75/5.75/7.0/9.0）。**同一个 x 窗口里
5.75–7.0 与 7.0–9.0 各含一个上坡半周期**，所以 rolling 档下这条 mask **不等于"真的在下坡"**。

- 现在按用户选的口径（只算坡面剖面的下坡段）落地，rolling 档沿用同一窗口；
- 如果要按真实坡度判定，改法是把 `downhill_segment` 换成对局部高度场做有限差分
  （`rolling_torch.profile_height_tensor` 在 x±ε 各算一次、取 `h(x+ε) − h(x−ε) < 0`），
  **代价**是每个控制步多两次高度场求值；
- 测试里有 `assertNotIn("terrain_profile", downhill_segment 源码)` 把当前选择钉住，
  换成坡度符号时该断言会失败 ⇒ 强迫同时更新本文与 README。

### 4.4 为什么下坡该降权（**推测，未验证**）

用户没有给理由。物理上最自然的解释是：下坡时车斗受重力分量驱动、会**主动加速追上机器人**，
碰撞在"机器人已经按指令减速/停住、车斗还在滑"的相位里几乎是必然事件，用与平地相同的
−5.0 去罚会让下坡段的回报被一项不可避免的代价主导。**这条只是解释，不是结论**：
本轮没有做任何"下坡碰撞是否真的不可避免"的测量（需要实跑）。

---

## 5. 验证

| 检查 | 结果 |
|---|---|
| `pytest imgo2_rl/tests -q`（`/opt/conda/envs/isaaclab/bin/python3`，Python 3.11.15、torch 2.7.0+cu128） | **657 passed, 4 skipped**（改动前基线 **639 passed, 4 skipped** ⇒ 净新增 **18 项**） |
| 新增文件单跑 | `test_towing_reward_weights_2026_10_17.py` ⇒ 18 passed |
| `py_compile`（`upper_env_cfg.py` / `upper_mdp.py` / 两个测试文件） | 通过 |
| `git diff --check` | 干净 |
| `git ls-files -i -c --exclude-standard` | 0 条 |

新测试覆盖的内容：

- **注册行**：`clearance` 权重 −10.0 与 `params` 逐项、旧 −1.0 字面量不得回潮；
  `action_rate` −0.05、旧 −0.1 不得回潮、`action_magnitude` 仍是 −0.05；
  `collision` 仍是 `weight=-5.0` + `params={"downhill_scale": 0.2}`，且
  `weight × scale == −1.0`（`places=12`）；
- **每步量级**：`clearance` 三个锚点的 func 与每步值、出生间距（0.365/0.473 m）下每步 > −0.02；
  `action_rate` 的 12/48 两档；
- **单条曲线**：`UpperRewardsCfg` 里名字含 `collision` 的 `RewTerm` **只有一条**；
- **掩码几何**：1301 点逐点等于 `profile_slope_degrees < 0`；边界（5.99/6.0/6.001/9.0/9.001）逐点；
  减速带/波浪列恒 False；源码读 `DOWN_START_M`/`FLAT_OUT_START_M`、不含 `terrain_profile`；
- **缩放语义**：`downhill_scale=0.2` 时下坡 cost=0.2、其他 1.0、无碰撞精确 0；
  **默认（不传参数）与旧实现逐位一致**且 dtype 仍 float32（退化性）。

**未验证（必须实跑才能回答）**：

1. 真实轨迹上 `downhill_segment` 的触发占比（下坡只有 3 m / 10 m 行程，且 STOP 点在出口平地）；
2. `Episode_Reward/clearance` 提权后的实际量级，以及"间隙 < 0.18 m 时它压过
   `min_clearance`/`tracking_velocity`"是否真的发生、是否导致策略为了避罚而后退/拒走；
3. 下坡 −1.0 是否足够（还是应当保持 −5.0 或取 −2.5）——本轮没有任何下坡碰撞率的前后对照；
4. `action_rate` 减半后 `Episode_Reward/action_rate` 与 `Train/mean_reward` 的相对变化。

**缺什么才能完成**：一台有 Isaac Lab 的训练机 + 一轮 64 环境 × ~20 轮的冒烟（同一 seed、
只换这三项权重的前后两轮），读 `Episode_Reward/{clearance, min_clearance, collision, action_rate,
tracking_velocity}`、`obs_cart_present`、`std` 与终止原因分布；下坡的相位读数需要
`downhill_segment` 的触发占比进诊断量（**本轮未加**，如需要可加一个 `1e-6` 诊断项或进
`--compact-log`）。
