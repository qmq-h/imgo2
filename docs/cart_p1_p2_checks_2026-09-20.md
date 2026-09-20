# P1/P2 检查与验收记录（拉取 `f3657cc` 之后）

日期：2026-09-20。对象提交：`f3657cc`（本地 `HEAD` = 远端 `origin/main`）。
关联问题 ID：CART-01（验收，**已完整通过**）、CART-02（初始化失败退出码，待修复）、
CART-03（离线重算入口报错，待确认）。

**结论**：实现记录 §6 的 P1/P2 验收**全部完成并通过**——P1 落地、P2 单工况、`--dt 0.0025` 复核、
8 case 阻力扫描；离线 5 项检查两台机器一致通过。所有「已停止」工况与解析黏性模型吻合在 1% 以内。
尚未完成的是 P3 及以后。

本次是**执行检查与验收**，没有修改任何代码或配置：各验收运行的 `runtime.json` 里
`git.working_tree` 只有本文件与 README 的文档改动，`commit` 均为 `f3657cc`。

解释器与环境：`/opt/conda/envs/isaaclab/bin/python3`（Python 3.11.13）、torch 2.7.0+cu128、
Isaac Lab 0.45.9、Isaac Sim 5.0.0.0、GPU **RTX 4090 24 GB**、驱动 580.126.09。

## 1. 离线检查：5 项全部通过（两台机器一致）

从仓库根或 `imgo2_rl/` 执行，退出码均为 0。

| 检查 | 结果 |
|---|---|
| `check_asset_paths.py` | PASS：机器人 URDF 17 个 mesh 引用 0 缺失、动作数据 21 份、cart URDF 解析通过、无机器绝对路径 |
| `check_model_sync.py` | PASS：2 份已登记 URDF 一致且各自复现录制数据（FK 恒等 RMSE 0.00214 m）、网格指纹 `8dc5b5995a11`、3 个 URDF 全部登记、cart 独立结构检查通过 |
| `check_cart_model.py` | PASS：10 kg、四被动轮、实心惯量自洽，SHA256 `866934f60da3561a21124a4e2973bbac8552a26523bdb8d40f99c2bbb964090c` |
| `test_cart_coast_metrics.py` | `Ran 22 tests` OK |
| 全量离线测试 | `Ran 111 tests` OK（8 个测试文件） |

## 2. P1 落地验收：**通过**

运行 `20260920T135225Z_08c8774a`（`--mode drop --duration 3 --headless`）。

| 判据 | 实测 | 期望 |
|---|---|---|
| 运行/case 状态 | `experiment.json` `state=completed` `valid=true`；`status.json` `completed`；`summary.json` `valid=true` `failures=[]` | 无异常 |
| 四轮接触 | `contact_fraction` = **0.9734**，四轮数值完全相同 | ≥0.1 且四轮都有 |
| 静止高度 | `trajectory.csv` 末行 `z_m` = **0.15000001 m**，t=0.10 s 起恒定；起始 0.180 m（= 0.15 + 3 cm 掉落量） | ≈0.15 m |
| 姿态稳定 | `max_tilt_rad` = 9.8e-6；roll/pitch 全程 ~1e-7 | 小 |
| 横向/偏航 | 2.2e-6 m / 1.9e-6 rad | 无漂移 |
| 导入质量 | **9.99999964 kg**（8.4 + 4×0.4） | 10 kg |
| 关节与接触 | `wheel_{fl,fr,rl,rr}_joint` 四个自由轮、四个同名接触传感器 | 四 free wheel |
| 刚度/阻尼 | **恰好 0.0**（`cart_coast.py:129` 另有断言） | 全零（被动） |
| 模型哈希 | `866934f6…090c` | 与实现记录 §2 一致 |

补充判读：

- **2.66% 的「未接触」不是缺陷**：`fl_normal_n=0` 的样本恰为 16/601 = 0.0266，与 `1 − contact_fraction`
  逐位吻合，落地时刻 t≈0.08 s ⇒ 全部是 3 cm 掉落的下落瞬态，**落地后四轮全程接触**。
- 末步法向力 26.41 / 22.62 / 26.77 / 22.24 N，合计 **98.04 N ≈ 10 kg × 9.81**，前后各 49.0 N 对称。
  左右分到 53.2 / 44.9 N（差 ~8%）属四点静不定接触的求解器分配，**不是模型不对称**：
  `cart.urdf` 的 `base_link` 惯性原点即 `0 0 0`，质心无偏移，且机体保持水平静止（tilt 1e-7、横移 2e-6 m）。
- 两条警告均为预期：`link rope_attachment has no body properties … merged into base_link`（架构记录 §3
  的「空 link + fixed joint、不加虚假质量」）、`Not all actuators are configured! 0 != 4`（被动小车；§6 明确
  不要靠加伺服消警告）。

## 3. P2 滑行验收：**全部通过**（单工况、dt 复核、8 case 阻力扫描）

### 3.1 已跑：`--velocities 1.0 --damping 0.016 --duration 10`（运行 `20260920T135445Z_1f7366c6`）

| 判据 | 实测 | 期望 |
|---|---|---|
| 状态 | `experiment.json` `completed` `valid=true`；`summary.json` `failures=[]`，2001 样本 | 无异常 |
| 是否停止 | `stopped=true`，`stop_time_s` = **4.215 s** | 正 b 应收敛 |
| 停止距离 | **1.053 m**（`observed_distance_m` 10 s 内 1.074 m） | 工程目标 0.5–2.0 m |
| 初/末速度 | 1.0 → **0.00027 m/s** | 显著衰减 |
| 四轮接触 | `contact_fraction` = **1.0 ×4** | 全程着地 |
| 姿态/横向 | `max_tilt_rad` 1.1e-6、横移 4.0e-6 m | 直线滑行 |
| 滚动残差 | `rolling_residual_rms_mps` = **1.2e-4** | 轮子滚而非滑 |
| 阻力做功 | `positive_resistance_work_j` = **0.0** | 全程耗散 |
| `calibration.json` | `candidate_damping_nms_per_rad = [0.016]`；`distance_decreases_with_damping = null`（只有 1 个 b，无配对可判） | 候选仅供参考 |

**这是「力矩真的在 PhysX 中生效」的决定性证据**（替代后端单测不能替代这一步）。除读数外还做了两项独立交叉核对：

1. **与解析黏性衰减对比**。四轮各 `tau = -b·ω`，折算到车体 `F = -4b·v/r²`，
   等效质量 `m_eff = 10 + 4I/r² = 10.8 kg`（`I=0.00128 kg·m²`, `r=0.08 m`）：

   | 量 | 解析 | 实测 | 偏差 |
   |---|---|---|---|
   | 时间常数 `τ = m_eff r²/(4b)` | 1.0800 s | ≈**1.0769 s**（对数速度斜率 −0.9286 /s） | 0.3% |
   | 停止时刻 `τ·ln(v0/0.02)` | 4.225 s | **4.215 s** | 0.2% |
   | 停止距离 `v0·τ·(1−0.02/v0)` | 1.058 m | **1.053 m** | 0.5% |

2. **力矩方向**。逐 (样本, 轮) 检查 `tau·omega`：按**当前帧**轮速有 2 个正功样本，均在 t=9.87–9.875 s、
   rr 轮、`|tau|≈1.9e-7 N·m`、`|omega|≈1e-4 rad/s`，即静止边界上的微幅换向；按文档口径
   （力矩对应**结束于该采样时刻的上一物理步**，故 × **上一帧**轮速）正功样本为 **0**，与
   `positive_resistance_work_j = 0.0` 一致。逐点 `tau` 与 `-b·ω_当前` 相差 ≤0.4%，正是这一帧的时序差。
   `t=0` 行四轮 `tau` 均为 0，符合约定。

### 3.2 dt 敏感性复核：**通过**（`--dt 0.0025`，运行 `20260920T135727Z_1e534c59`）

同一工况（`v=1.0 b=0.016 10 s`）把物理步长减半，与 3.1 的 dt=0.005 对比：

| 量 | dt=0.005 | dt=0.0025 | 相对偏差 |
|---|---|---|---|
| 样本数 | 2001 | 4001 | — |
| `stop_time_s` | 4.215 | 4.220 | **0.12%** |
| `stop_distance_m` | 1.053125 | 1.055570 | **0.23%** |
| `observed_distance_m` | 1.074432 | 1.076994 | 0.24% |
| 四轮 `contact_fraction` | 1.0 ×4 | 1.0 ×4 | 无变化 |
| `max_tilt_rad` | 1.0e-6 | 1.0e-6 | 量级即噪声 |
| `rolling_residual_rms_mps` | 1.22e-4 | 8.4e-5 | 差 3.8e-5 m/s（绝对量微） |
| `positive_resistance_work_j` | 0.0 | 0.0 | 无变化 |

结论：**接触状态与停止距离对步长不敏感**（停止距离 0.23%、停止时刻 0.12%），dt 减半复核通过。

### 3.3 阻力扫描：**通过**（运行 `20260920T135948Z_16885db6`，8/8 个 case）

首两次尝试失败（见 §3.5），第三次**串行单独执行**成功。8 个 case 全部 `valid=true`、
`status="completed"`、2001 样本、`coast.svg` 齐全，四轮 `contact_fraction` 全为 1.0：

| case | v (m/s) | b (N·m·s/rad) | stopped | `t_stop` (s) | `stop_distance` (m) | v 末 (m/s) |
|---|---|---|---|---|---|---|
| `case_00_v0.5_b0` | 0.5 | 0 | **false** | — | —（10 s 观测 4.99990 m） | 0.49999 |
| `case_01_v0.5_b0.008` | 0.5 | 0.008 | true | 6.945 | 1.033949 | 0.00485 |
| `case_02_v0.5_b0.016` | 0.5 | 0.016 | true | 3.470 | 0.515817 | 0.00026 |
| `case_03_v0.5_b0.032` | 0.5 | 0.032 | true | 1.730 | 0.256703 | 0.00023 |
| `case_04_v1_b0` | 1.0 | 0 | **false** | — | —（10 s 观测 10.0000 m） | 1.00002 |
| `case_05_v1_b0.008` | 1.0 | 0.008 | true | 8.440 | 2.111118 | 0.00968 |
| `case_06_v1_b0.016` | 1.0 | 0.016 | true | 4.215 | 1.053125 | 0.00027 |
| `case_07_v1_b0.032` | 1.0 | 0.032 | true | 2.105 | 0.524179 | 0.00026 |

三条判据全部成立：

1. **`distance_decreases_with_damping = true`，两个速度组都是**（0.5 m/s：1.034 → 0.516 → 0.257 m；
   1.0 m/s：2.111 → 1.053 → 0.524 m），单调下降且分离度明显。
2. **`b=0` 对照干净**：两个速度下 10 s 内速度只变 3e-5 / 2e-5 m/s、`stopped=false`（与 §6 预期一致）
   ⇒ 接触/摩擦模型没有引入虚假拖滞。
3. **力矩在 PhysX 生效**（§3.1 已证）+ **工程候选合理**：`calibration.json` 给出
   `candidate_damping_nms_per_rad = [0.016, 0.032]`，即 1.0 m/s 下停止距离落在 0.5–2 m
   （0.008 的 2.111 m 超出上界，被正确排除）。

**6 个「已停止」case 逐个与解析黏性模型对照**（`m_eff = 10.8 kg`，`τ = m_eff r²/(4b)`，
`t_stop = τ·ln(v0/0.02)`，`D = v0·τ·(1−0.02/v0)`）：

| case | τ 解析 | `t_stop` 解析 | `t_stop` 实测 | `D` 解析 | `D` 实测 | 偏差 |
|---|---|---|---|---|---|---|
| `v0.5 b0.008` | 2.160 | 6.953 | 6.945 | 1.037 | 1.034 | 0.27% |
| `v0.5 b0.016` | 1.080 | 3.476 | 3.470 | 0.518 | 0.516 | 0.50% |
| `v0.5 b0.032` | 0.540 | 1.738 | 1.730 | 0.259 | 0.257 | 0.96% |
| `v1.0 b0.008` | 2.160 | 8.450 | 8.440 | 2.117 | 2.111 | 0.27% |
| `v1.0 b0.016` | 1.080 | 4.225 | 4.215 | 1.058 | 1.053 | 0.50% |
| `v1.0 b0.032` | 0.540 | 2.112 | 2.105 | 0.529 | 0.524 | 0.95% |

六个全部落在 **1% 以内**；偏差随 b 单调增大（0.27% → 0.96%）且实测恒略小于解析值，
与 dt=0.005 的离散化一阶误差一致（b=0.032 时 dt/τ≈0.9%）。**纯黏性阻力本来渐近趋零**，
所以停止时刻由「0.02 m/s 持续 0.5 s」的判据决定，这一点与实测完全吻合。

### 3.4 顺带验证：真实残缺运行的离线重算契约成立

`case_01_v0.5_b0.008` 是一个**真实的**截断产物（`status.json` 停在 `{"state":"running"}`、样本 1175/2001），
恰好用来验证实现记录 §3.4 那条「残缺运行不得被重新标为有效」的约定 —— 此前只有合成单测覆盖：

```text
python3 imgo2_rl/scripts/tools/summarize_cart_coast.py <该 case 目录>
→ valid=false, failures=["run_not_completed","incomplete_or_irregular_samples"], 退出码 1
```

即使它的速度曲线看起来正常衰减（0.5 → 0.033 m/s），也没有被判为有效。契约在真实数据上成立。

### 3.5 前两次扫描尝试失败（历史，用于说明「串行执行」这一操作前提）

| 运行 | 结果 |
|---|---|
| `20260920T135646Z_6ba3cab4` | 空，`experiment.json` 停在 `state:"starting"`，无 case 目录 |
| `20260920T135659Z_ef391b97` | 只跑完 2/8，中断于第 2 个 case 的 t≈5.87 s，`state:"starting"` |

这两条与 `20260920T135650Z_c2868729`（空）在 **13 秒内**先后启动，都没有正常收尾；
而随后**单独**运行的 `20260920T135727Z_1e534c59`（dt 复核）与 `20260920T135948Z_16885db6`
（完整扫描）都正常完成。本仓库此前已记录同一容器内并发 Kit 实例会争锁
（`another kit process is locking it` / `Failed to acquire exclusive lock to data store`）。
⇒ **操作前提：串行执行**，一次只跑一个 `cart_coast.py`，确认上一条命令退出后再起下一条。
（并发是中断的**推断**原因，未取得当时的控制台输出，故不作为已确认根因。）

被中断的 `ef391b97` 里 `case_00_v0.5_b0` 是完整的（10 s 内 4.99990 m、速度只掉 0.003%、
`stopped=false`），`case_01_v0.5_b0.008` 是截断的——正好用作 §3.4 的真实残缺样本。



## 4. 另一台容器（harness 沙箱）的记录：无 GPU，无法执行

同一批命令在 harness 所在容器（`NVIDIA_VISIBLE_DEVICES=void`、无 `/dev/nvidia*`、
`torch.cuda.is_available()=False`）中三次尝试（`20260920T132556Z_9c437998`、
`20260920T132609Z_eadaabd0`、`20260920T132637Z_c550537a`）**全部**停在 Isaac Sim 物理初始化：
`CUDA error: no CUDA-capable device is detected` → `Unable to create PxCudaContextManager!` →
`vkEnumeratePhysicalDevices failed`；`experiment.json` 记 `state:"failed"`、
`RuntimeError: No CUDA GPUs are available`，无任何轨迹产物。Isaac Sim 5.0 的 PhysX GPU 上下文
是硬依赖，没有 CPU 回退。此段保留为「同一命令在不同容器结果不同」的环境依据。

## 5. 新发现问题

### CART-02：初始化失败时入口退出码为 0

**现象**：上述三次失败的尝试真实退出码**全部为 0**，stderr 无 traceback；唯一失败信号是
`experiment.json` 的 `state: "failed"` ⇒ 调用方只看 `$?` 会把「Isaac Sim 根本没起来」当成成功。
两次成功的运行同样返回 0，**故退出码无法区分成功与初始化失败**。

**根因（源码依据 + 观察）**：`imgo2_rl/scripts/towing/cart_coast.py:224-230` 在异常时先记录
`failed` 再 `raise`，但 `finally: application.close()` **先于异常传播**执行；而
`SimulationApp.close()`（`isaacsim/exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py:626`）
内部调用 `self._app.shutdown()` 关闭 Kit/Carb，进程在 C++ 侧直接终止，Python 层再也拿不到控制权。
注意这与文档 §6 的「框架运行完成但数据异常返回 2」是两条不同路径（后者走
`return 0 if manifest["valid"] else 2`，未构造过该情形）。

**修法（未实施）**：失败路径显式带码退出（例如捕获异常后改用 `os._exit(2)`，或把
`application.close()` 挪到只关一次且不吞异常的位置）。
**验证方式**：无 GPU 环境下跑 `--mode drop`，断言退出码非 0 且 `experiment.json` 仍记 `failed`。

### CART-03：离线重算入口对失败运行目录抛原始 traceback

**现象**：`python3 imgo2_rl/scripts/tools/summarize_cart_coast.py <只含 experiment.json 的目录>`
抛未捕获的 `FileNotFoundError: .../config.json`，真实退出码 1。

**根因**：`summarize_cart_coast.py:13` 直接读 `config.json`，没有对「该目录不是合法 case 目录」做前置判断。
初始化失败时不会写 `config.json`，这种目录无法重算——本身合理，但报错不友好。
`--help` 正常；第 19-25 行「残缺/未完成运行不得被重标为有效」的行为由单测覆盖，不是本条问题。

**修法（未实施）**：入口处检查 `config.json`/`trajectory.csv` 是否存在，缺失时打印可读错误并非 0 退出。

## 6. 环境告警（未阻塞，记录备查）

成功运行的日志里 warp 报 CUDA 驱动入口错误：

```text
Warp CUDA error: Failed to get driver entry point 'cuDeviceGetUuid' (CUDA error 1)
Warp CUDA error 36: API call is not supported in the installed CUDA driver (cuda_init)
```

P1/P2 的确定性落地、四轮接触、无 NaN 与解析解 0.3% 吻合都正常，说明它未影响这两条路径
（PhysX 与 warp 是不同后端）。若后续出现异常读数，这是首要怀疑的环境项。

## 7. 产物清单

| 运行 | 结果 |
|---|---|
| `20260920T135225Z_08c8774a` | P1 drop，**通过**；含 `experiment.json`/`runtime.json`/`sweep.json`/`usd/` 与 `case_00_v0_b0/` |
| `20260920T135445Z_1f7366c6` | P2 coast 单工况 `v=1.0 b=0.016 dt=0.005`，**通过**；`case_00_v1_b0.016/` + `calibration.json` |
| `20260920T135727Z_1e534c59` | P2 dt 复核 `--dt 0.0025`，**通过**；`case_00_v1_b0.016/` |
| `20260920T135948Z_16885db6` | P2 阻力扫描，**通过（8/8）**；8 个 case + `sweep.json` + `calibration.json` |
| `20260920T135659Z_ef391b97` | 阻力扫描首两次尝试，**中断（2/8）**；`case_00_v0.5_b0`（完整）、`case_01_v0.5_b0.008`（截断） |
| `20260920T135646Z_6ba3cab4`、`20260920T135650Z_c2868729` | 空运行，`state:"starting"` 无产物 |
| 三个 `20260920T1325/1326…` 运行 | harness 容器无 GPU 的失败证据（只有 `experiment.json`） |

以上均在 `imgo2_rl/logs/towing/cart_coast/` 下，已被忽略规则覆盖。
`case_01_v0.5_b0.008` 的 `summary.json` 是 §3.4 离线重算生成的（`valid=false`）。

## 8. 未做

- 未修改任何代码、配置或模型；未提交、未推送。
- 未训练（P1/P2 不需要 RL）、未部署、未动硬件。
- 未做 P3 及以后（绳索张力、上层策略）；P1/P2 验收已按实现记录 §6 全部完成。
