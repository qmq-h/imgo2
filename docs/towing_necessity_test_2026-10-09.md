# 拖曳「上层任务必要性」基线测试台（2026-10-09）

## 目的

在继续投入上层拖曳 RL（脚本速度指令 + 12 维关节位置残差）之前，先用**它的对照基线**
回答「这个任务有没有必要」：

- 基线 = **冻结 AMP 底层策略**（`imgo2_deploy/policy/imgo2/amp/policy.pt`，45 维观测，
  由 `FrozenLowLevelPolicy` 按部署契约喂观测）+ **脚本速度指令**（station 0 → tow v → STOP 0）；
- 负载 = 仓库那台被动小车；连接 = 三类（弹性绳 / 低弹性绳 / 刚体球铰连杆）；
- 网格 = 速度档 × 连接 × 质量档 × 坡度。

如果基线在网格的全部指标上都通过，上层任务在这些工况上就没有必要性证据；如果它在某一相
系统性失败（起步关节响应、全程跟速、停车滑移、停车间距、停车关节响应），那些失败模式就是
上层任务（以及它的奖励项、STOP 调度）要修的对象。

**本测试不加载任何上层 checkpoint**：跑的是基线，不是策略回放。

## 交付物与命令

| 项 | 路径 |
|---|---|
| 测试脚本 | [play_towing_test.py](../imgo2_rl/scripts/towing/play_towing_test.py) |
| 离线测试（67 项，无需 GPU） | [test_towing_play_test.py](../imgo2_rl/tests/test_towing_play_test.py) |

```bash
# ① 只看网格与代价，不启动仿真（标准库即可）
python3 imgo2_rl/scripts/towing/play_towing_test.py --dry-run

# ② 冒烟：1 坡度 × 2 速度 × 2 连接 × 2 质量 = 8 环境，先确认能跑起来
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/towing/play_towing_test.py \
    --headless --slopes 0 --velocities 0.5 1.0 --connections compliant rigid \
    --cart-masses 5 25 --write-csv none

# ③ 默认完整网格：225 case（5 坡度 × 3 速度 × 3 连接 × 5 质量），每坡度 45 环境并行、5 次仿真过程
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/towing/play_towing_test.py --headless

# ④ 对照：把阶跃指令换成固定斜坡（「Fixed Ramp」基线），判断纯脚本 shaping 是否已经够用
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/towing/play_towing_test.py \
    --headless --command-shaping ramp --ramp-time-s 1.0
```

`--dry-run` 的默认输出（脚本会打印实际值）：

```
[plan] 冻结底层策略：amp；地面摩擦固定 0.8；轮轴阻尼 0.016 N·m·s/rad
[plan] 速度档：0.5, 1, 1.5 m/s；连接：compliant, inextensible, rigid；质量：5, 10, 15, 20, 25 kg
[plan] 坡度：+0, +5, -5, +10, -10 deg（+ = 上坡）；后端 gravity；站定制动 hold（额外 5 N·m·s/rad，仅 station）
[plan] 网格：5 坡度 × 3 速度 × 3 连接 × 5 质量 = 225 case；每坡度 45 环境并行、5 次仿真过程
[plan] 每 case：station 1.00 s（200 步）+ tow 5.00 s（1000 步）+ coast 5.00 s（1000 步）= 2200 步 / 11.00 s
[plan] 记录：每 1 物理步一行 ⇒ 每 case 约 2200 行，共约 495000 行；CSV 策略 failed
```

## 五项指标的定义

| 指标 | summary 字段 | 定义（口径写进产物，不做口头约定） |
|---|---|---|
| 起步关节响应误差 | `metrics.startup.joint_rms_rad` / `joint_max_rad` / `worst_joint` / `per_joint_rms_rad` / `torque_saturated_frac` | 起拖后 `--transition-window`（默认 1.0 s）内，`q − q*` 的逐关节统计；`q` 是实测关节角，`q*` 是**冻结策略当拍下发的关节位置目标**（上层残留动作改的正是它）。另给起步速度响应：`body_vx_rms_err_mps`、`time_to_90pct_s`、`overshoot_ratio` |
| 全程速度跟踪误差 | `metrics.speed.mae_mps` / `rmse_mps` / `bias_mps` / `ratio_mean` / `p95_abs_err_mps` / `frac_within_10pct`、稳态窗 `steady_*` | tow 段**全程**，**体系** x 速度与指令之差（与冻结策略观测同口径；坡上 ≠ 世界系速度） |
| 停止时小车滑移距离 | `metrics.stop.cart_coast_distance_m` / `cart_coast_to_rest_m` / `cart_coast_time_to_rest_s` / `cart_speed_at_stop_mps` | STOP 那一刻起小车的 x 位移；以及速度首次降到 0.02 m/s 以下时的距离与耗时（τ 由黏性轮阻决定，可用 `stop` 段解析式复算） |
| 停止时机器人—小车距离维持 | `metrics.stop.clearance_at_stop_m` / `min_clearance_coast_m` / `final_clearance_m` / `time_to_contact_after_stop_s` / `contact` / `contact_channels` | **车头到机器人后腿的几何间隙**（全腿 FK，复用 `summarize_tow.py`），挂点距 `rope_distance_m` 另列；接触由三路见证判定：车斗接触力、几何间隙 ≤ 0、负载单步速度跃变（阈值按记录步长放大） |
| 停止时的关节响应 | `metrics.stop.joint_rms_rad` / `joint_max_rad` / `worst_joint` / `torque_saturated_frac` / `settle_time_s` / `body_vx_rms_mps` | STOP 后 `--transition-window` 内的同一套关节跟踪误差，外加「指令归零到 `|vx| < 0.05 m/s` 的耗时」 |

判定码（阈值可用 CLI 覆盖，全部进产物）：`OK` / `LOW` 停车余量低 / `JNT` 关节响应超限 /
`SPD` 跟速超限 / `COL` 追尾接触 / `FALL` 跌倒 / `INV` 记录不可用；一个 case 可命中多条，
`code` 取最严重的一条，`reasons` 保留全部。

## 网格、摩擦与阈值默认值

- 速度 `0.5 / 1.0 / 1.5 m/s`；连接 `compliant / inextensible / rigid`；质量 `5 / 10 / 15 / 20 / 25 kg`
  （质量与惯量按同一比例缩放，与 `tow_drag.py` 同口径）；坡度 `0 / +5 / −5 / +10 / −10 deg`
  （`+` = 沿 +x 上坡，机器人在前、小车在后）；
- 地面摩擦**固定 `0.8`**（工厂地面常规：混凝土/环氧地坪静动摩擦 0.6–0.9，本仓库历来用 0.8）；
- 连接参数取中间档：弹性绳 `k=4000 N/m`、`c=100 N·s/m`，`L0=0.8 m`、初始松弛 `0.4 m`；
  rigid 的杆长取初始挂点距 `L0 − slack = 0.4 m`（与测量台一致，保证三类初始间隙可比）；
- 阈值初值（**未标定**）：关节 RMS ≤ `0.10 rad`、单关节 ≤ `0.30 rad`、跟速 MAE ≤ 指令的 `20%`、
  停车几何间隙 > `0.10 m`、跌倒判据 `base z < 0.15 m` 或 `|pitch| > 0.80 rad` 的样本 > 20%。

## 坡度怎么实现：现有平地 + 旋转重力（物理等价）

**坡度地形资产还在实现中，本测试台不动地形**，默认用 `TowSceneCfg` 现有的那块平地，
把重力写成坡面分解：

```
g = (−g·sinθ, 0, −g·cosθ)        # +θ = 沿 +x 上坡
```

在随坡面倾斜的参考系里，「水平地面 + 倾斜重力」与「倾斜坡面 + 竖直重力」是**同一组方程**：
接触法向仍垂直于地面、法向力 `mg·cosθ`、沿坡下滑分量 `mg·sinθ` 逐项相同。因此这不是近似替代，
也不必新建几何。冻结策略的观测里本来就有 `projected_gravity`，它自然看到倾斜后的重力
（脚本按**单位**重力方向喂观测，`set_gravity()` 用的是带模长的完整向量，两处不能混）。

已知的唯一差别：出生高度是竖直 0.35 m（坡面上量是 `0.35/cosθ`），可忽略。

`--slope-backend terrain` 是给将来的坡面地形留的口子：目前 `slope_ground_plan()` 直接抛
`NotImplementedError`（CLI 会报错退出），而不是静默退回重力方案。接入时要改两处：
`slope_ground_plan()` 返回地面 spawn 配置，场景按坡度摆出坡面与出生点。

## 坡上站定段：驻车制动仿真（`--slope-settle hold`，默认）

被动小车只有轮轴黏性阻尼 `b = 0.016 N·m·s/rad`（训练侧同值），**没有驻车制动**。斜坡上
它在站定段就会自己溜：5° 时终端速度约 `m·g·sinθ·r²/(4b) ≈ 0.85 m/s`，1 s 站定能滚 0.8 m
—— 超过 0.4 m 的初始松弛，绳在起拖之前就被拽直，于是每个坡度 case 都被这个瞬态主导，
测不出控制器的差别。因此默认 `--slope-settle hold`：**只在 station 段**给四个轮子额外加
`--hold-damping`（默认 5 N·m·s/rad）的黏性制动，起拖瞬间释放（终端速度降到 ~3 mm/s）。
这是明确的建模选择，会写进 `experiment.json`、每个 case 的 `config.json`、`report.md` 与
`report.json`；要跑「无制动、真实溜坡」用 `--slope-settle free`。

## 产物

```
<output-dir>/                     # 默认 imgo2_rl/logs/towing/play_test/<UTC 时间戳>_<随机>
  experiment.json                 # 运行级清单：参数、git、python、网格、阈值、坡度/连接设置
  report.json                     # 逐 case 摘要（含 summarize_tow 全量输出）+ 分组统计 + 结论
  report.csv                      # 逐 case 一行关键指标（表格工具友好）
  report.md                       # 人读报告：逐坡度判定矩阵、失败模式计数、结论、限制
  summaries/<case>.json           # 每 case 摘要（总是写）
  <case>/tow.csv + config.json    # 逐物理步原始记录（仅 --write-csv all|failed，默认 failed）
```

目录名 `slope+5_v1_compliant_m20kg` 形式；`--write-csv` 默认 `failed`：只给非 `OK` 的 case
留原始轨迹（完整网格每步全写约 0.5 GB）。

## 判定与结论的写法

`report.md` 的「结论（任务是否有必要）」按 **平地 / 上坡 / 下坡** 三组分别给：

- 全通过 ⇒ 明确写「该组不构成上层任务必要性的证据」；
- 有失败 ⇒ 列出失败模式计数；若失败**集中在停车段**（`COL`/`LOW`/`stop_joint_error`），
  提示「必要性主要来自停车时序与间隙维持，需用 `--command-shaping ramp` 或更早的 STOP 调度
  做对照，再判断纯脚本 shaping 是否够用」；若跨起步/全程/停车多相，则属于上层残差的目标。

结论只覆盖本网格 + 本阈值，脚本不替读者外推。

## 已做验证（本机 `/opt/conda/envs/isaaclab`，Python 3.11.13，torch 2.7.0+cu128）

- `--dry-run`（标准库即可运行）：默认网格 225 case、每坡度 45 环境、2200 步/case、命令行校验
   （含 `--env-spacing` 下限、`--max-envs` 上限、坡度/质量/速度范围、`rope_length − slack` 几何）；
- `imgo2_rl/tests/test_towing_play_test.py` **67 项通过**（`/usr/bin/python3` 无 torch 时 66 通过
   + 1 跳过，跳过项是与 `mdp/rope_model.py::CONNECTION_MODELS` 的交叉核对）；覆盖：
  坡度→重力（含模长不变性与镜像）、网格与顺序、阶段划分与 `phase_of`、指令整形、
  五项指标在合成轨迹上的数值、接触三路见证（含 `--record-every` 的阈值放大）、判定码严重度、
  分组统计与结论文案、判定矩阵、人读报告、CLI 校验、记录字段契约（AST 抽取 `make_row`
  的列集与 `TEST_FIELDS` 逐项比对）；
- `imgo2_rl/tests` 全量 **390 项通过 0 失败**；
- `python -m compileall`、`git diff --check`、tracked-ignore 检查通过。

## 未验证（缺什么才能完成）

1. **仿真链路一次都没跑过**：本会话进程看不到 GPU（`NVIDIA_VISIBLE_DEVICES=void`、无
   `/dev/nvidia*`、`torch.cuda.is_available() == False`），Isaac Sim 起不来。要完成必须先跑
   上面命令 ② 的冒烟（8 环境），确认：环境构造、45/8 env 的逐 env 质量/连接/指令张量、
   `physics_sim_view.set_gravity()` 生效、记录列与 `TEST_FIELDS` 一致（运行时那条 `RuntimeError`
   是最后一道闸）、`summarize_tow` 能吃到新记录。
2. **坡度曲线上没有任何数值**：出生下落 + 倾斜重力下的站定、上坡绳被拽直、下坡小车自己
   溜向机器人这三类瞬态都只有推理，没有数据。
3. **阈值未标定**：关节 RMS/单关节上限是工程占位，首轮数据出来前 `JNT` 不能当定论。
4. **未覆盖**：弹性绳另外 3 档 k/c、轮阻档、breakaway/Coulomb 阻力、跨环境隔离、真机、
   `--slope-settle free` 的对比。
5. **「必要性」结论本身**：脚本只产出分组统计与措辞，是否真的砍掉/保留上层任务由用户读
   `report.md` 决定；本记录不预判结论。

## 与既有入口的关系

- `tow_drag.py`：单速度、单质量集合的**测量台**，逐 case 串行、带完整产物与解析预判；
  本脚本是它的网格化版本（逐 env 并行 + 坡度 + 关节响应指标），复用同一套绳模型、
  冻结策略适配器与 `summarize_tow` 判读；
- `scan_towing_boundary.py`：外层调度器，按速度/质量/轮阻/摩擦多次启动 `tow_drag.py`，
  产出 `boundary.json`；不做坡度、不做关节响应；
- `mdp/connection_grid.py`：训练场景的 20×20 列×行网格（类型 + 长度），本脚本的网格是
  另一套（速度 + 质量 + 坡度），只共用三类连接的定义。
