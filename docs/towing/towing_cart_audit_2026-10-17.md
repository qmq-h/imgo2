# towing 一致性审计与离线工具修复（2026-10-17）

用户问「towing 侧是否基本准备好了 + 是否还有逻辑不自洽」，本记录保存审计结论与处置：
**P1 三项已修**（离线工具口径统一到新小车）、**P2 四项待实测**、**P3 十一处文字漂移待改**。

## 0. 已确认自洽（可以放心）

| 项 | 证据 |
|---|---|
| 契约 v5（frame 58 / explicit 5 / latent 16 / actor 79 / critic 73 / 动作 13） | runner、`upper_policy_runtime`、测试台三处版本常量同为 5；wrapper 断言 `58/5`；`validate_towing_contract` 拒 v4/v3 |
| CE Net 重建掩码 42 维 | `UpperObservationSpec.reconstruction_mask()` → cfg → runner → trainer；NaN 反证测试在 |
| 新小车几何（3.0 kg = 车斗 0.8 + 4×0.05 + 重物块 2.0，轮 r=0.03） | URDF / MJCF `cart.xml` / `check_towing_mjcf.py` 三处一致；C++ `ConfigureCartMass` 与 `assets/cart_mass.py` 同口径 |
| 轮阻不双计 | `cart.urdf` 轮关节 `damping="0"`；黏性轮阻只在 `_apply_towing_physics` 施加一次 |
| 质量/惯量/CoM 一起写 | `reset_towing_episode` 用 `cart_mass.body_properties` → `set_masses/set_inertias/set_coms`；测试台同一套 |
| 包络越界 | `effective_vx_clip(amp_vx_range, policy.vx_envelope_mps)` 超出低层训练包络**直接报错** |
| Δz | 由 `init_state` 自动派生 0.35 − 0.13 = **0.22**，min_clearance 契约测试已按 0.22 |

## 1. P1（已修）

### P1.1 `tow_drag.py` 的质量扫描模型

**现象**：`mass_scale_factor(target, nominal)` 给出一个比例，同时乘到 `masses` 与 `inertias` 上
（`set_masses(nominal_masses * scale)` / `set_inertias(nominal_inertias * scale)`）。新小车只有
重物块可调、轮子质量固定、CoM 随重物上移 ⇒ 这个模型会把**轮子也放大**、且 CoM 不动，
与训练侧 / 测试台（都用 `cart_mass.body_properties`）不一致。

**修法**：删 `mass_scale_factor`，改为 `cart_mass_properties(model, target_total_kg)` ——
按文件路径加载 `imgo2_rl/assets/cart_mass.py`（纯标准库，不拉 isaaclab 包），返回
`total/ballast/base_mass/base_com/base_inertia/fixed`；`reset_case` 只写 base_link 的
mass / 绕 CoM 惯量 / CoM 高度，四个轮子保持 URDF 名义值；写后与 PhysX 的 `get_masses()`
总和核对（误差 > 1e-3 相对值即报错）。`case_config` 记录改为
`cart_ballast_mass_kg` / `cart_mass_actual_kg` / `cart_base_mass_kg` / `cart_base_com_z_m`
（旧的 `cart_mass_scale` 删除）。

**数值自检**（真实 URDF，`read_cart_model`）：名义 3.0 kg ⇒ ballast 2.0、base 2.8、
`com_z = 0.1143`、`fixed = 1.0`；10 kg ⇒ ballast 9.0、base 9.8、`com_z = 0.1469`；
总重 ≤ 1.0 kg（车斗+四轮）时 `body_properties` 直接报错。

**测试**：`tests/test_towing_tow_drag.py::CoastPredictionTests::test_cart_mass_properties_follow_the_ballast_model`
（替换原 `test_mass_scale_factor`）；`test_default_configuration_keeps_a_margin` 改用新小车
夹具（3.0 kg / r=0.03 / 轮惯量 2.25e-5）并断言默认轮阻 0.0045。

### P1.2 轮阻「新域缩小」是不是问题

**不是**：`b` 是**力矩**阻尼，轮子稳态时 `b·ω = F·r`、`ω = v/r` ⇒ `F = b·v/r²`。因此
「同样的滚动阻力」要求 `b ∝ r²`：`0.032 × (0.03/0.08)² = 0.0045`、`0.008 × 0.140625 = 0.001125`。
两种轮径下**等效滚阻都是 `4b/r²` = 5–20 N/(m/s)**（旧 `4×0.008/0.08² = 5`、`4×0.032/0.08² = 20`；
新 `4×0.001125/0.03² = 5`、`4×0.0045/0.03² = 20`）⇒ **物理域没变，变的只是那一列数字的量纲基准**，
所以这次只能改数值、不能改行为。

**但有一个副作用要注意**：阻力不变、质量却从 10 kg 名义降到 3.0 kg 名义 ⇒ 同样 `b` 下
减速度约 3.5 倍（`m_eff` 10.8 → 3.1 kg），滑行距离只有旧车的 ~0.28 倍（0.5 m/s、名义 b：
0.077 m vs 0.27 m）。**结论**：若设计意图是「保持同样的阻力调参」——合适，本次按此落地
（`--wheel-damping` 默认 0.0045，`cart_coast --damping` 默认
`[0, 0.001125, 0.00225, 0.0045]`，`scan_towing_boundary.DEFAULT_DAMPINGS = (0.001125, 0.0045)`）；
若意图是「保持同样的滑行手感」，则 `b` 还要再按质量缩放（×3.1/10.8 ≈ 0.29），
**这是一个待定的设计选择**，见 §2 的 TOW-CAL-01。

### P1.3 统一到新模型车

- `towing_env_cfg.py`（P4 场景，`tow_drag.py` 与 cart 测试仍在用）的注释按新 URDF 更新：
  挂点/前表面 `+0.25 → +0.30`、车斗碰撞箱「0.60×0.45×0.02、底面离地 0.09」、
  轮 Ø6 cm（旧注释写 0.10–0.20 m、静止 0.15、r=0.08）；
- `tools/tow_clearance.py` 的说明改为「0.019r，r=0.03 ⇒ 0.57 mm」（半径本身从 URDF 读，逻辑未改）；
- `cart_coast.py` / `scan_towing_boundary.py` 的阻尼默认值与 `tow_drag.py` 统一到新域；
- 三个工具本来就通过 `read_cart_model` 从 URDF 取半径/质量，所以几何自动跟随，只有**默认的 b**
  与**质量施加方式**需要改。

## 2. P2（未修，待训练机实测）

| # | 项 | 现状与判据 |
|---|---|---|
| 1 | `min_clearance` 在停车段是否仍触发 | 阈值 = 出生间隙 − 0.02，Δz 0.17→0.22 后最短绳行 **0.3446 → 0.3166 m**（绳 0.3166–1.1622 = 0.633–0.775·L）；同时车前表面 0.25→0.30 会把实测间隙再压 0.05 m ⇒ 与「停车间隙 ≈0.64·L」的裕度只剩毫米级。**判据**：按连接类型/行读停车段 `Episode_Reward/min_clearance`，若整体为 0（尤其 L=0.5）就按 2026-10-10 记录 §2.4 的备选改用 `gap_at_stop − deadband_m` |
| 2 | `clearance` = −10 与新几何 | 预警线 0.20 m；间隙 0.10 m 时 −1.06/步。新小车前表面更靠前 ⇒ 可能比旧车更常触发；**无任何实跑读数** |
| 3 | `towing_force_rate` 的 200/600 N/s | 按旧 10 kg 车标定；新车名义 3 kg（靠重物块加到 5–20 kg），力尺度与瞬态都变了 ⇒ 阈值等于未标定 |
| 4 | `collision` 下坡 mask 在 rolling 档 | 同一 x 窗口在 5.75–7.0 / 7.0–9.0 各含上坡半周期 ⇒ mask ≠「真的在下坡」（2026-10-17 CE Net 记录 §5 已列） |

## 3. P3（文字漂移，代码正确；未修，逐条改即可）

1. Δz=0.17 残留：`upper_logic.py:255-256`（Fz 占比）、`upper_mdp.py:140-141`（Fz 占比）、
   `upper_mdp.py:1101`（min_clearance 推导）、`upper_mdp.py:1122-1123`（阈值表 0.3446–1.1704）、
   `upper_mdp.py:1132`（Δz* 0.218/0.261）、`upper_env_cfg.py:292`（阈值注释）、
   `connection_grid.py:94`（出生间隙公式 0.17/0.238）、`towing_env_cfg.py`（已随 P1.3 改）。
2. 包络注释仍写 (−1.0, 1.5)：`upper_mdp.py:6`、`upper_mdp.py:386-387`、`upper_logic.py:151`
   （实际 `amp_vx_range = (−1.0, 1.0)`）。
3. `docs/towing/towing_reward_weights_2026-10-17.md:77,81`（min_clearance 列用 0.3446）、
   README 2026-10-10 条目（历史可保留，但顶层「当前口径」应指新值）。
4. `play_towing_test.py:103` docstring 仍写「速度 0.5–1.5 m/s」（235 行已写 0.001125–0.0045）。
5. **未提交的并行会话把 `upper_env_cfg.py` 的模块级 docstring 整段删了**（原本写双头架构、
   `task_command` vs `loco_command` 的防作弊红线、契约版本；且那段停在 v4）⇒ 建议按 v5 补回，
   而不是删除。
6. `imgo2_deploy/policy/imgo2/towing/policy.pt` 是 v3 时代（含质量头、actor 80）的训练态
   checkpoint，v5 下 runner 直接拒载；目录里只有它，容易误当可部署策略。

## 4. 验证

| 检查 | 结果 |
|---|---|
| `pytest imgo2_rl/tests -q` | **683 passed, 4 skipped, 1336 subtests passed** |
| `tests/test_towing_tow_drag.py` | 62 passed（含新的质量属性测试） |
| `py_compile`（改动文件） | 通过 |
| 工具冒烟（无 Isaac Lab） | `cart_mass_properties` 对真实 URDF 输出：名义 3.0 kg → base 2.8 / com 0.1143；10 kg → base 9.8 / com 0.1469；≤1.0 kg 报错 |

**未验证**：本机无 Isaac Lab / CUDA ⇒ 三个工具的实跑（`tow_drag` 的质量/惯量/CoM 写入是否被
PhysX 接受、多 case 质量扫描的读数）与 §2 的全部标定都待训练机。

## 5. 2026-10-18 P3 清理（已完成）

用户「3 操作一下，其他不重要」⇒ 按 §3 逐条落地，**只动文字/注释与部署目录说明，不改逻辑**：

| # | 改动 | 位置 |
|---|---|---|
| 1 | Δz 0.17 → **0.22**；`min_clearance` 阈值重算为 **0.3166–1.1622 m**（= 0.633–0.775·L；0.941×/0.983× 出生间隙）；杆行 Δz* **0.2568/0.2950 m**，余量 **3.68/7.50 cm**（原 4.75/9.06 cm） | `upper_logic.py`（Fz 占比 15–44%）、`upper_mdp.py`（推导/阈值表/Δz*）、`upper_env_cfg.py`（注册注释）、`connection_grid.py`（出生间隙示例 206/429 mm @L=0.6） |
| 2 | 包络注释由「裁进 AMP (−1.0,1.5)」改为「裁进**任务包络** (−1.0,1.0)；它必须落在冻结低层训练包络内（AMP −1.0…1.5 / PPO −1.5…3.0），`effective_vx_clip` 越界报错」 | `upper_mdp.py`（模块/`process_actions`/cfg 字段）、`upper_env_cfg.py`（`__post_init__` 注释）、`upper_logic.py`（`loco_command_vx`） |
| 3 | 出生间距 **0.3366/0.4515 m**、func **0.063/0.0065**、每步 **−0.0315/−0.0033**、`min_clearance` 表阈值 **0.3166**（0.10 m ⇒ −0.054、0 ⇒ −0.079）；同日同步 `towing_reward_retune_2026-10-10.md` §2.4/§2.5 的新几何数字 | [奖励权重记录](towing_reward_weights_2026-10-17.md)、[阈值推导记录](towing_reward_retune_2026-10-10.md) |
| 4 | 速度域 0.5–1.5 → **0.5–1.0**（docstring/常量/`--velocities` 说明 3 处）；`tow_drag` 的旧 `b=0.016` 与滑行示例改为新小车（默认 0.0045、20 kg 约 0.50 m、出生间隙约 0.60 m） | `play_towing_test.py`、`tow_drag.py` |
| 5 | `upper_env_cfg.py` 模块 docstring 按 **v5/CE Net** 补回（58/5/16/79/73/13 + 42 维重建掩码；双头架构/防作弊红线保留） | `upper_env_cfg.py` |
| 6 | 部署目录新增 [README](../../imgo2_deploy/policy/imgo2/towing/README.md)：`policy.pt` 是旧契约训练态 checkpoint、v5 拒载、towing `play.py` 尚无导出分支、不要接线；**二进制保留**（历史权重） | `imgo2_deploy/policy/imgo2/towing/` |

**同时纠正一条旧结论**：§3 原文与 `upper_mdp` 都写「停车段天然生效」。按新几何重算后：

- 长绳行（L≥1.0）停车间隙 ≈0.64–0.70·L 低于阈值（0.752–0.775·L）⇒ **会触发**；
- 最短绳行 L=0.5 的停车间隙 ≈0.32 m 与阈值 0.3166 m 只差约 **3.4 mm** ⇒ **是否触发必须实测**，
  不能假定；已把 docstring 改成「待 TOW-CAL-01 #1 在训练机读 `Episode_Reward/min_clearance`」。

**验证**：`pytest imgo2_rl/tests -q` ⇒ **683 passed, 4 skipped, 1336 subtests passed**；
`py_compile` 改动文件通过；grep `0.3446|1.1704|0\.17 m|(-1\.0, 1\.5)` 在 towing 源码/脚本无命中
（`policy_cfg.py` 的 AMP 训练包络除外）。**未验证**：P2 四项（`min_clearance` 停车段、
`clearance −10` 量级、`towing_force_rate` 200/600、rolling 下坡 mask）仍需训练机
64 env × ~20 轮冒烟；见 §2 与 TOW-CAL-01。
