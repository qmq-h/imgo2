# 拖曳「上层任务必要性」基线测试台（2026-10-09）

## 目的

在继续投入上层拖曳 RL（脚本速度指令 + 12 维关节位置残差）之前，先用**它的对照基线**
回答「这个任务有没有必要」：

- 基线 = **冻结 AMP 底层策略**（`imgo2_deploy/policy/imgo2/amp/policy.pt`，45 维观测，
  由 `FrozenLowLevelPolicy` 按部署契约喂观测）+ **脚本速度指令**（station 0 → tow v → STOP 0）
  + **横向/朝向 PD 保持**（把机器人压在 lane 中线、保持超前；前进速度 vx 不参与 PD）；
- 负载 = 仓库那台被动小车；
- **场景 = 训练场景**（用户 2026-10-09 决定「统一用训练场景，后续也是」）：
  `UpperTowingSceneCfg`，40 列 × 20 行 = 800 环境，逐 env 的连接类型 / 长度 / 坡度量级直接取
  `connection_grid.env_spec(i)`（不再有测试自己的地形、case 网格或第二套场景实现）；
- 测试只叠加**工作条件扫描 = 速度 × 质量**（确定性轮转，见下）。

如果基线在网格的全部指标上都通过，上层任务在这些工况上就没有必要性证据；如果它在某一相
系统性失败（起步关节响应、全程跟速、停车滑移、停车间距、停车关节响应），那些失败模式就是
上层任务（以及它的奖励项、STOP 调度）要修的对象。

**本测试不加载任何上层 checkpoint**：跑的是基线，不是策略回放。

## 交付物与命令

| 项 | 路径 |
|---|---|
| 测试脚本 | [play_towing_test.py](../imgo2_rl/scripts/towing/play_towing_test.py) |
| 离线测试（87 项，无需 GPU） | [test_towing_play_test.py](../imgo2_rl/tests/test_towing_play_test.py) |

```bash
# ① 只看网格与代价，不启动仿真（标准库即可）
python3 imgo2_rl/scripts/towing/play_towing_test.py --dry-run

# ② 冒烟：网格前缀 40 环境（三种连接 + 三个坡度量级都在），并缩短两段时长
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/towing/play_towing_test.py \
    --headless --num-envs 40 --tow-duration 2 --coast-duration 2 --write-csv none

# ③ 完整训练网格：800 环境 = 40 列 × 20 行，一次仿真跑完（不再按坡度分轮）
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/towing/play_towing_test.py --headless

# ④ 对照：把阶跃指令换成固定斜坡（「Fixed Ramp」基线），判断纯脚本 shaping 是否已经够用
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/towing/play_towing_test.py \
    --headless --command-shaping ramp --ramp-time-s 1.0

# ⑤ 对照：关掉横向/朝向 PD（只给 vx），看没有横向反馈时机器人自己会漂多少
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/towing/play_towing_test.py \
    --headless --lane-keeping off
```

`--dry-run` 的默认输出（脚本会打印实际值）：

```
[plan] 场景 = **训练场景** `UpperTowingSceneCfg`：40 列 × 20 行 = 800 条 lane（env i 的连接/长度/坡度量级取 env_spec(i)）
[plan] 并行环境数 800（网格全集 800 的 1 倍 ⇒ 覆盖 800 个 cell）；一次仿真跑完，不再按坡度分轮
[plan] 网格分布：连接 compliant 320 / inextensible 160 / rigid 320；坡度量级 0° 400 / 5° 200 / 10° 200；长度 0.6–1.2 m
[plan] 工作条件（确定性轮转）：速度 0.5, 1, 1.5 m/s × 质量 5, 10, 15, 20, 25 kg = 15 组；slot = row + column ⇒ 每组 50–55 个 env
[plan] 「连接 × 质量」env 数（slot 轮转 ⇒ 严格均衡）：compliant 5:64/…；inextensible 5:32/…；rigid 5:64/…
[plan] 训练侧质量随机范围 [5, 15] kg：20, 25 kg 超出该范围 ⇒ 那几档是外推检查
[plan] 物理量：地面摩擦 0.8；轮轴阻尼 0.032 N·m·s/rad；地面/重力/传感器/资产全部沿用训练配置
[plan] 每 env：station 1.00 s + tow 5.00 s + coast 5.00 s = 2200 步 / 11.00 s
[plan] 记录：每 5 物理步一行（25 ms）⇒ 每 env 约 440 行，共约 352000 行；CSV 策略 failed
```

## 工作条件怎么分配（`slot = row + column`）

逐 env 的 `(速度, 质量)` 由**确定性轮转**给出：`slot = row + column`，
质量 = `cart_masses[slot % len(masses)]`、速度 = `velocities[(slot // len(masses)) % len(velocities)]`。

**为什么不用 env 序号**：`COLUMNS = 40` 是质量档数的整数倍（40 % 5 == 0），按序号轮转会让
质量只由 `column % 5` 决定 —— 而列同时决定连接类型与坡度量级，于是 `(连接, 质量)`、
`(坡度量级, 质量)` 严重混淆（实测：inextensible 完全没有 5 kg，5°/10° 的 inextensible 只落到
重载，15 个组合里只有 14 个非空）。改成 `slot` 后固定一列时 `slot` 随 20 行取 20 个连续整数：

- 每个 (速度, 质量) 组合 50–55 个 env（`slot` 取值个数两头少中间多，不是严格 ±1）；
- **`(连接 × 质量)` 与 `(坡度量级 × 质量)` 严格均衡**：compliant/rigid 各档 64、inextensible
  各档 32；0° 各档 80、5°/10° 各档 40；每列都覆盖全部 15 个组合；
- **每个 cell 仍然只落到一种 (速度, 质量) 组合**（800 env = 800 cell）⇒ 判读要按分档聚合，
  不能当逐 cell 的完整响应面。

## 五项指标的定义

| 指标 | summary 字段 | 定义（口径写进产物，不做口头约定） |
|---|---|---|
| 起步关节响应误差 | `metrics.startup.joint_rms_rad` / `joint_max_rad` / `worst_joint` / `per_joint_rms_rad` / `torque_saturated_frac` | 起拖后 `--transition-window`（默认 1.0 s）内，`q − q*` 的逐关节统计；`q` 是实测关节角，`q*` 是**冻结策略当拍下发的关节位置目标**（上层残留动作改的正是它）。另给起步速度响应：`body_vx_rms_err_mps`、`time_to_90pct_s`、`overshoot_ratio` |
| 全程速度跟踪误差 | `metrics.speed.mae_mps` / `rmse_mps` / `bias_mps` / `ratio_mean` / `p95_abs_err_mps` / `frac_within_10pct`、稳态窗 `steady_*` | tow 段**全程**，**体系** x 速度与指令之差（与冻结策略观测同口径；坡上 ≠ 世界系速度） |
| 停止时小车滑移距离 | `metrics.stop.cart_coast_distance_m` / `cart_coast_to_rest_m` / `cart_coast_time_to_rest_s` / `cart_speed_at_stop_mps` | STOP 那一刻起小车的 x 位移；以及速度首次降到 0.02 m/s 以下时的距离与耗时（τ 由黏性轮阻决定，可用 `stop` 段解析式复算） |
| 停止时机器人—小车距离维持 | `metrics.stop.clearance_at_stop_m` / `min_clearance_coast_m` / `final_clearance_m` / `time_to_contact_after_stop_s` / `contact` / `contact_channels` | **车头到机器人后腿的几何间隙**（全腿 FK，复用 `summarize_tow.py`），挂点距 `rope_distance_m` 另列；接触由三路见证判定：车斗接触力、几何间隙 ≤ 0、负载单步速度跃变（阈值按记录步长放大） |
| 停止时的关节响应 | `metrics.stop.joint_rms_rad` / `joint_max_rad` / `worst_joint` / `torque_saturated_frac` / `settle_time_s` / `body_vx_rms_mps` | STOP 后 `--transition-window` 内的同一套关节跟踪误差，外加「指令归零到 `|vx| < 0.05 m/s` 的耗时」 |

| 横向/朝向保持（`--lane-keeping pd` 时） | `metrics.lane.y_rms_m` / `y_max_abs_m` / `heading_rms_rad` / `heading_max_abs_rad` / `vy_saturated_frac` / `wz_saturated_frac` / `tow_y_*` | lane 系横向偏移（目标 0 = 中线）与朝向误差（目标 0 = 超前），全回合与牵引段各一份；再报 PD 指令是否顶到限幅 |

判定码（阈值可用 CLI 覆盖，全部进产物）：`OK` / `LOW` 停车余量低 / `LAT` 横向或朝向保持超限 /
`JNT` 关节响应超限 / `SPD` 跟速超限 / `COL` 追尾接触 / `FALL` 跌倒 / `INV` 记录不可用；
一个 env 可命中多条，`code` 取最严重的一条，`reasons` 保留全部。

## 网格、摩擦与阈值默认值

- **场景与 cell 参数**：训练网格 40 列 × 20 行 = 800 条 lane；列决定连接类型与坡度量级
  （compliant 320 / inextensible 160 / rigid 320；0° 400 / 5° 200 / 10° 200），行决定连接长度
  （0.6–1.2 m，20 档）；env `i` 取 `connection_grid.env_spec(i)`；
- **只扫速度 × 质量**：速度 `0.5 / 1.0 / 1.5 m/s` × 质量 `5 / 10 / 15 / 20 / 25 kg`
  （质量与惯量按同一比例缩放，与 `tow_drag.py` 同口径），分配规则见上一节；
- 地面摩擦**固定 `0.8`**（训练名义值；训练随机 0.4–1.2）；
- 轮轴阻尼**固定 `0.032 N·m·s/rad`**（= 训练 `initial_wheel_damping`；训练随机 0.008–0.032）；
- 连接参数由网格给：弹性绳 `k`/`c` 取该 cell 的档位（`ELASTIC_KC`），绳的目标初始挂点距
  `0.5·L0`（留松弛），刚体杆的目标初始挂点距 = 杆长 `L`；
- 阈值初值（**未标定**）：关节 RMS ≤ `0.10 rad`、单关节 ≤ `0.30 rad`、跟速 MAE ≤ 指令的 `20%`、
  停车几何间隙 > `0.10 m`、跌倒判据 `base z < 0.15 m` 或 `|pitch| > 0.80 rad` 的样本 > 20%。

## 场景：直接用训练场景（2026-10-09 起，之前的两种后端已删除）

首版测试台自建了平地 + 旋转重力的 `gravity` 后端，以及把训练 tile 平移到紧凑网格的 `terrain`
后端。用户 2026-10-09 决定「统一用训练场景，后续也是」后，这两条自建路径**全部删除**，改为：

- 场景对象：`upper_env_cfg.UpperTowingSceneCfg(num_envs, env_spacing=训练值)` —— 与训练同一个
  `TowingSlopeTerrainImporter`（40×20 的连续剖面地形、相邻 lane 共边拼接）、同一个机器人
  `IMGO2_CFG`、同一个 `cart.urdf`、同一组**按机器人 body 过滤**的接触传感器；
- 物理量：`SimulationCfg`、地面材质、重力、时间步全部沿用训练配置（重力是世界竖直，地形才是
  坡；不再有 `set_gravity()` 这种「平地 + 旋转重力」的等价实现）；
- 逐 env 参数：连接类型 / 连接长度 / 坡度量级 / 目标初始挂点距 / 绳的 k·c 全部取
  `connection_grid.env_spec(i)`；绳模型按训练 `upper_mdp` 的写法用**逐 env 张量**构造
  （三套 `make_rope_model(rest_length=…)` + `MultiRopeModel(model_ids=逐env)`，非本类型的 k/c
  填 `ELASTIC_KC[0]` 占位）；
- 出生几何：出生在剖面的**平地段**（lane 原点），姿态竖直（单位四元数、无出生旋转），小车沿
  −x 退 `along = sqrt(d² − Δn²)`，使两挂点三维距 = 该 env 的 `initial_distance`
  （纯函数 `attachment_along` / `spawn_offsets`，逐 env 自检；第一拍不能有约束力）；
- 度量坐标系 = lane 系（切向 +x、法向 +z）：`progress` = x 行程；`surface_height` =
  `z − profile_height(逐 env 档位, x)`；`body_pitch_rel_rad` = 世界系俯仰 + 逐 env 局部坡度；
  启动时对 0/5/10° 做 `profile_height_tensor` ↔ 标量版的逐点交叉核对。

**与训练仍然不同的地方**（有意为之，写进产物 caveats）：本测试台**不加载上层 checkpoint**、
用脚本指令 + PD 代替上层策略；域随机化固定成确定性扫描（摩擦/轮轴阻尼固定、速度与质量由
`slot` 轮转而不是逐回合重采样、**不复现 12.5% 无小车锚点** —— 所有 env 都拖车）；
仿真循环是本脚本自己的「读状态 → 施力 → 步进 → 记录」，没有走 `ManagerBasedRLEnv`。

## 站定段：不再需要驻车制动仿真（`--slope-settle` 已删除）

首版在斜坡上出生，被动小车（只有轮轴黏性阻尼 `b = 0.016 N·m·s/rad`）在 1 s 站定段里会自己
溜坡（5° 时终端速度约 `m·g·sinθ·r²/(4b) ≈ 0.85 m/s`），把起拖瞬态盖掉，所以当时加了
「只在 station 段」的额外轮轴制动 `--slope-settle hold`（默认）作为建模选择。

2026-10-09 起出生点在剖面的**平地段**（lane 原点，剖面 `h(0)=0`）：机器人直立、小车在平地上
后方 `initial_distance` 处，站定段没有任何下滑分量 ⇒ 这个人为制动**不再需要**，选项与实现一起
删除。轮轴阻尼固定为训练名义值 `0.032 N·m·s/rad`（`initial_wheel_damping`），坡道出现在起拖之后。

## 产物

```
<output-dir>/                     # 默认 imgo2_rl/logs/towing/play_test/<UTC 时间戳>_<随机>
  experiment.json                 # 运行级清单：参数、git、python、训练场景来源、网格与工作条件分配、阈值、caveats
  report.json                     # 逐 env 摘要（含 summarize_tow 全量输出）+ 分档统计 + 结论
  report.csv                      # 逐 env 一行关键指标（表格工具友好）
  report.md                       # 人读报告：每个坡度量级一张判定矩阵、失败模式计数、结论、限制
  summaries/<env>.json            # 每 env 摘要（总是写）
  <env>/tow.csv + config.json     # 逐物理步原始记录（仅 --write-csv all|failed，默认 failed）
```

目录名 `env0000_col00_row00_compliant_L0.6_g0_v0.5_m5kg` 形式；`--write-csv` 默认 `failed`：只给非 `OK`
的 env 留原始轨迹（800 环境每 5 物理步一行 ≈ 35 万行，全写约 0.5 GB）。

## 判定与结论的写法

`report.md` 的「结论（任务是否有必要）」按 **平地 / 5° / 10°** 三组（= lane 的坡度量级）分别给：

- 全通过 ⇒ 明确写「该组不构成上层任务必要性的证据」；
- 有失败 ⇒ 列出失败模式计数；若失败**集中在停车段**（`COL`/`LOW`/`stop_joint_error`），
  提示「必要性主要来自停车时序与间隙维持，需用 `--command-shaping ramp` 或更早的 STOP 调度
  做对照，再判断纯脚本 shaping 是否够用」；若跨起步/全程/停车多相，则属于上层残差的目标。

结论只覆盖本网格 + 本阈值，脚本不替读者外推。

## 已做验证（本机 `/opt/conda/envs/isaaclab`，Python 3.11.13，torch 2.7.0+cu128）

- `--dry-run`（标准库即可运行）：打印训练场景网格（800 环境、逐 env 的 cell 与工作条件分配、
  「连接 × 质量」分布、物理量来源）、2200 步/env、PD 参数与阈值，以及命令行校验；
- `imgo2_rl/tests/test_towing_play_test.py` **87 项通过**（`/usr/bin/python3` 无 torch 时 87 项
  OK / 跳过 4 项，跳过项是与 `mdp/rope_model.py::CONNECTION_MODELS` 的交叉核对）；覆盖：
  `slot = row + column` 轮转的确定性、(速度×质量) 桶计数 50–55、**「连接 × 质量」「坡度量级 ×
  质量」均衡性的回归**（防止质量退化成只由列决定）、每列覆盖全部 15 个组合、逐 env 的
  `EnvCase` 与 `env_spec(i)` 逐位一致、网格分布（连接 320/160/320、坡度量级 400/200/200）、
  出生几何（勾股解、几何无解报错、逐 env 挂点距 = `initial_distance`）、阶段划分与 `phase_of`、
  指令整形、横向/朝向 PD（方向、阻尼、±π 归一化、限幅、机体系投影）、五项指标在合成轨迹上的
  数值、接触见证（`force_matrix_w` 口径 + `--record-every` 阈值放大）、判定码严重度、分组统计
  与结论文案、逐坡度量级判定矩阵、人读报告、CLI 校验（**断言旧开关 `--slopes/--slope-backend/
  --connections/--rope-length` 等已删除**）、记录字段契约（AST 抽取列集与 `TEST_FIELDS` 比对）；
- `imgo2_rl/tests` 全量 **419 项通过 0 失败**；
- `python -m compileall`、`git diff --check`、tracked-ignore 检查通过。

## 首跑定位并修复的一个阻断 bug（2026-10-09）

**现象**（用户默认网格首跑）：`[FAILED] RuntimeError: The size of tensor a (5) must match the
size of tensor b (45) at non-singleton dimension 2`。

**根因**：逐 env 的质量/惯量缩放写错了缓冲形状。PhysX tensor API 返回的是**整批**缓冲：
`get_masses()` → `(N, num_bodies)`、`get_inertias()` → `(N, num_bodies, 9)`；小车
`merge_fixed_joints=True`，无质量的挂点被并进 `base_link`，所以 `num_bodies = 5`
（base_link + 四轮）。原代码把已经带 env 维的缓冲又 `.unsqueeze(0)`：

```
nominal_inertias.unsqueeze(0) * scales_host     # (1, N, 5, 9) × (N, 1) → dim2: 5 vs N
```

`(N,1)` 广播成 `(1,1,N,1)`，于是 dim 2 上 `num_bodies`(5) 撞 `N`(45) —— 报错里的 5 和 45
正是这两个量。**质量那一路（2 维）单独乘不会报错**，只是形状变成 `(1, N, 5)`；N=1 时两路都
侥幸不炸，所以单环境/小网格冒烟抓不到它。

**修法**：抽成纯函数 `scale_cart_mass_inertia(nominal_masses, nominal_inertias, mass_scales)`，
质量乘 `(N,1)`、惯量乘 `(N,1,1)`，并在函数里显式校验输入形状（`(N,nb)` / `(N,nb,9)` / `(N,)`）。

**验证**：新增 `MassScalingTests` 5 项 —— N=1/8/45 的形状、逐 env 比例确实按 env 生效、
**把 bug 的确切形状写成负向断言**（`inertias.unsqueeze(0) * (N,1)` 必须在 dim 2 抛错）、
非法形状报错；另加静态守卫禁止 `nominal_masses.unsqueeze(0)` / `nominal_inertias.unsqueeze(0)`
再出现。该轮记录：`test_towing_play_test.py` 103 项通过（`/usr/bin/python3` 无 torch 时 98 通过
+ 5 跳过）、全量 `imgo2_rl/tests` 430 项通过 0 失败（当时的口径，`work_conditions` 质量轮转的
混淆与测试台改造见下一节）。

**顺带修正**：该 bug 的诱因之一是旧的 `terrain` 后端把 `terrain_origins` 只报第一块 tile；
2026-10-09 起测试台已不再自建地形（改用训练场景的 `TowingSlopeTerrainImporter`），这段历史只
保留在这里作为记录。

**限制**：这只是把首跑的报错原因消掉，**仿真仍未跑通**；同一条命令要重跑冒烟才能确认。

## 二跑：判读崩溃（已修）+「出生就贴上」的定量（2026-10-09 晚）

**现象**（用户实跑）：`--num-envs 9 --headless --write-csv all --tow-duration 1 --coast-duration 1`
主循环跑完，在**逐 case 判读**处 `[FAILED] TypeError: '<' not supported between instances of
'float' and 'NoneType'`；`experiment.json` 停在 `state=failed`，`summaries/` 与逐 case `tow.csv`
**一个都没有**（只能重跑）。同一配置开可视化时「机器人就在胡乱挣扎」。

### ① 崩溃根因（已修）

`tow_clearance.clearance()` 的「面对面」判据要求机器人后表面点与车头点在横向**与**竖向都落在
`FACING_TOL_M = 0.05 m` 内；机器人被拽翻 / 腾空 / 姿态离奇时该条件会整体不成立，函数**合法地**
返回 `(None, None)`。离线可复现：合成轨迹上 `robot_z_m ≥ 0.6 m`（机—车间距约 1 m 时）或
「俯仰 45° + 抬高」就一对点都对不上。而 `summarize_tow` 直接 `min(after_tow_clearance)` ⇒
`None` 与 `float` 不可比。

修法（`scripts/tools/summarize_tow.py`）：

- 逐相位统计不可用样本 → 新增 `clearance_unavailable_samples` / `clearance_unavailable_phases`（**不静默**，
  坏姿态看得见）；
- `min_clearance_m` / `min_clearance_coast_m` / `min_clearance_station_m` 只对**可用**样本取最小值，
  某相位全部不可用 ⇒ `None`；
- 判「几何接触」前守 `is not None`；`final_clearance_m` 只退回 **coast 段**最后一个可用样本
  （旧记录没有 coast 段时才退回全轨迹最后一个可用样本），**不让 station 段的数冒充「末端间隙」**。

同时把测试台的**原始轨迹提前到指标计算之前**落盘（`--write-csv all` 时逐 case 立即写 `tow.csv`）：
判读代码再出问题也不会把这一轮的证据一起丢掉（`--write-csv failed` 仍在判出非 `OK` 之后写）。

### ② 「出生就贴上」的定量

用 `tow_clearance` 的真实 FK（`imgo2_description/urdf/imgo2.urdf` 全腿链）逐行算**出生姿态**
（默认站姿、0.35 m 出生、竖直）下「机器人后表面 ↔ 车头」的几何间隙：

| 行 | L0 | compliant（`0.5·L0`） | 几何间隙 | rigid（`L0`） | 几何间隙 |
|---|---|---|---|---|---|
| 0 | 0.60 | 0.300 | **0.0088 m** | 0.600 | 0.3370 m |
| 1 | 0.63 | 0.316 | 0.0277 m | 0.632 | 0.3699 m |
| 2 | 0.66 | 0.332 | 0.0463 m | 0.663 | 0.4026 m |
| 3 | 0.69 | 0.347 | 0.0645 m | 0.695 | 0.4352 m |
| … | … | … | … | … | … |
| 19 | 1.20 | 0.600 | 0.3370 m | 1.200 | 0.9495 m |

⚠️ `upper_env_cfg.py` 注释里的「spawn 间隙 ≈ 0.250 m」是**挂点距**（`along = sqrt(d² − Δn²)`），
不是几何间隙：机器人后腿在默认站姿下伸到 base 后方 **0.398 m**，而挂点只在 −0.16 m，两者差 0.24 m。
当年平地单绳验证用的 0.4 m 挂点距对应几何间隙 **0.0955 m**（已记为「生成时就贴上」的观察值），
现在 row0 的 compliant 档只有 **8.8 mm**。

### ③ 实跑数据：9/9 都在**站定段**就塌（2026-10-09 18:06 run）

修好判读崩溃后同一条命令**跑完了**（`state=completed`，`imgo2_rl/logs/towing/play_test/20261009T100609Z_3b5735d0/`，
9 case + `report.md`），这次有数据：

| env | 连接/质量 | 判定 | 跌倒相位 | 最低 base 高 | 起步关节 RMS | 站定段最大张力 | 车斗接触峰值 |
|---|---|---|---|---|---|---|---|
| 0 | compliant/5 kg | FALL | **station** | 0.075 m | 0.611 rad | 130.7 N | 0 N |
| 1 | compliant/10 kg | FALL | tow | 0.076 m | 0.595 | 405.5 N | 14.5 N |
| 2 | compliant/15 kg | FALL | **station** | 0.075 m | 0.624 | 480.5 N | 0 |
| 3 | compliant/20 kg | FALL | **station** | 0.074 m | 0.627 | 267.2 N | 0 |
| 4 | compliant/25 kg | FALL | **station** | 0.075 m | 0.617 | 1004.9 N | 0 |
| 5 | compliant/5 kg | FALL | **station** | 0.074 m | 0.625 | 166.4 N | 0 |
| 6 | compliant/10 kg | FALL | **station** | 0.078 m | 0.635 | 1542.9 N | 472.9 N |
| 7 | compliant/15 kg | FALL | **station** | 0.079 m | 0.618 | 778.6 N | 0 |
| 8 | **rigid**/20 kg | FALL | **station** | 0.074 m | 0.587 | 1056.4 N | 0 |

读法（三条互斥情形里**第一条被排除、第三条被排除、只剩控制/驱动这一档**）：

1. **不是「绳/杆把机器人拽翻」**：`rope_tension_n` 在前 6 个记录点（含 t=0.03/0.055/0.105）**恒为 0**，
   张力要到 t≈0.33 s 才起跳（`settle_max_tension_n` 是站定段内的**最大值**，不是首拍值）；
   跌倒趋势（base 从 0.3425 → 0.1653 只用 25 ms）出现在任何张力之前。
2. **不是「出生间隙 8.8 mm 顶到小车」**：9 个 case 只有 2 个出现车斗接触（env1 14.5 N、env6 472.9 N），
   其余 7 个 `cart_deck_fx_n ≡ 0`；而 9 个全倒。⇒ 第二节的间隙表是**真实的设计余量问题**，但不是本次倒地的原因。
3. **不是「掉进地里」**：`robot_z_m` 不是单调下掉而是**弹跳式**（0.3425→0.1653→0.2443→0.2756→0.2339→0.2963→…→0.075），
   `invalid_samples = 0`（无 NaN/Inf）。
4. **落在「关节驱动」这一档**：`robot_tau_*` 从一开始就**正负交替、打到 ±23.7 N·m（= `effort_limit`）**，
   关节实测被压到极限（`robot_jp_02` 一度 = −3.0000），起步关节 RMS **0.59–0.64 rad（阈值 0.10 的 6 倍）**，
   同时 `settle_robot_travel_m = −1.099 m`、`settle_load_drift_m = −0.649 m` ⇒ 站定段整对儿被拖着倒退 1 m。
   **注意 t=0.005 那一拍是正常的**（目标 = 默认站姿 ± 小量：`jt=(−0.010, 0.912, −1.803)`，
   与 `default + action_scale×action` 逐项吻合），异常从**第二拍**开始（hip 目标 −2.1 ≈ 饱和）。
   即：**观测/置换在首拍是对的，但第二拍开始策略输出饱和、关节高速摆动**。

⇒ 下一步（判据已收窄）：把 `--record-every 1` 跑 3 环境，看**第 0–4 个物理步**（第二拍策略动作之前）
到底发生了什么；同时离线把「名义站姿」喂给导出的 `policy.pt`（`imgo2_deploy/policy/imgo2/amp/policy.pt`，
sha256 `cba59d44…`）验证契约：如果对名义站姿它就输出饱和动作，那问题在**观测组装/契约**而不在物理。

## 未验证（缺什么才能完成）

1. **9/9 倒地的根因还没定位到一行代码**：判读崩溃已修、同一条命令已跑完并落盘（见上一节），
   现在的判据收窄到「第二拍起策略输出饱和 + 关节高速摆动」这一档，但**还没区分**
   「观测组装/契约错」与「物理/驱动侧不稳」。需要：① `--record-every 1 --num-envs 3` 看第 0–4
   物理步；② 离线把名义站姿喂给 `policy.pt` 验契约；③ 与训练任务（同一 policy、同一 action term）
   做同条件对照。**本会话进程看不到 GPU**（无 `/dev/nvidia*`、`torch.cuda.is_available() == False`），
   ①③ 只能由用户终端执行。场景构造、逐 env 绳模型、逐 env 质量/惯量缩放、逐 env 出生几何
   （第一拍不能有约束力 → 实测首拍张力 0 ✓）、训练接触传感器的 `force_matrix_w` 读法与顺序守卫、
   记录列与 `TEST_FIELDS` 一致这些**已由这次实跑通过**。
2. **坡度曲线上没有任何数值**：出生下落、上坡绳被拽直、下坡小车自己溜向机器人这三类瞬态
   都只有推理，没有数据。
3. **阈值未标定**：关节 RMS/单关节上限是工程占位，首轮数据出来前 `JNT` 不能当定论。
4. **未覆盖**：训练侧的域随机化（摩擦 0.4–1.2、轮轴阻尼 0.008–0.032、12.5% 无小车锚点、
   逐回合重采样速度/质量）—— 本测试台有意固定成确定性扫描；breakaway/Coulomb 阻力、
   跨环境隔离、真机。
5. **逐 env 记录粒度**：`--record-every` 默认 5（25 ms = 冻结策略周期），否则 800 环境逐物理步
   是约 176 万行 / ~2 GB 内存。`time_to_contact_after_stop_s` 因此是 125 ms 粒度（contact 判据
   本身不受影响，速度跃变阈值已按步长放大并有测试）。
5. **「必要性」结论本身**：脚本只产出分组统计与措辞，是否真的砍掉/保留上层任务由用户读
   `report.md` 决定；本记录不预判结论。

## 与既有入口的关系

- `tow_drag.py`：单速度、单质量集合的**测量台**，逐 case 串行、带完整产物与解析预判；
  本脚本是它的网格化版本（逐 env 并行 + 坡度 + 关节响应指标），复用同一套绳模型、
  冻结策略适配器与 `summarize_tow` 判读；
- `scan_towing_boundary.py`：外层调度器，按速度/质量/轮阻/摩擦多次启动 `tow_drag.py`，
  产出 `boundary.json`；不做坡度、不做关节响应；
- `mdp/connection_grid.py` + `mdp/slope_geometry.py` + `slope_terrain.py`：训练场景的
  40×20 网格与连续剖面地形（类型 + 长度 + 坡度量级）；2026-10-09 起本脚本**直接使用**它们
  （场景 = `UpperTowingSceneCfg`，逐 env 参数 = `env_spec(i)`），不再有第二套 case 网格、
  自建地形或坡面 tile 平移。
