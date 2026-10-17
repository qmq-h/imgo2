# 混合地形：减速带 + 波浪拼进同一张 60 列网格（2026-10-16）

用户口径（当天多轮）：
1. 「目前的 towing 任务地形包括平地和上下坡地面，我希望添加一个减速带地面」；
2. 「直接把这个拼接到之前的设置中，直接增加 4 列来放这个，每个坡度两列」→ 追问连接分配后定为
   **「和 5° 那个块一样」= 10 列**；
3. 「没有两类了，都只有 1 类同参数」+「减速带就只设置 300 底，100 顶，50 高的一个等腰梯形形状，
   都是 mm 单位」；
4. 「最后添加一个波浪的起伏地形，**和上下坡占据同样的位置**，最后用一个平地收尾」+
   「振幅不要太大，先定义大概 10 cm」→ 选定 **3 周期等幅正弦、振幅 10 cm、两端平滑归零**。

结论：**地面只有一张按列分派的网格**，减速带与波浪各占一个 10 列的块拼进原网格；原 40 列的
列号与语义一字未动。第一步建的独立整块减速带任务 `Imgo2-towing-upper-speedbump-rl-lab` 已按
用户要求删除。

> 同期还有一项与地形无关的契约变更（decoder 去掉显式质量预测，v3 → v4），记录在
> [decoder v4 记录](towing_decoder_v4_2026-10-16.md)；两者都要求新建 run，旧 checkpoint 一律失效。

## 1. 网格（60 列 × 20 行 = 1200 环境）

| 块 | 列 | 地形 | 连接（弹性绳 4 档 k/c / 刚体杆 / 低弹性绳） |
|---|---|---|---|
| 平地 | 0–19 | 0° 剖面（矩形板） | 8 / 8 / 4（每档 2 列） |
| 5° 坡 | 20–29 | 平地 2.25 → 上坡 3 → 坡顶 0.75 → 下坡 3 → 平地 2.25 | 4 / 4 / 2（每档 1 列） |
| 10° 坡 | 30–39 | 同上、10° | 4 / 4 / 2 |
| **减速带** | **40–49** | **一条等腰梯形凸条**（§2.1） | 4 / 4 / 2（照 5° 块同构） |
| **波浪** | **50–59** | **3 周期等幅正弦**（§2.2） | 4 / 4 / 2（照 5° 块同构） |

行仍是 20 档长度（绳 0.5–1.5 m、杆 0.5–1.0 m，逐行一档）；`env i` 的列/行映射、出生与 STOP
口径都不变。全网格连接 **24 弹性 / 24 杆 / 12 低弹**（比例仍是 2 : 2 : 1，每类 ×20 行 =
480 / 480 / 240 环境）。

`--terrain-profile` 只决定**前 40 列的平滑段**用哪套高度场（`hill` 单坡包 / `rolling` 连续
起伏）；减速带块与波浪块**恒为各自地形**，与开关无关。两个任务 ID 不变：
`Imgo2-towing-upper-rl-lab`（hill）、`Imgo2-towing-upper-rolling-rl-lab`（rolling），各自都带
这两个块。

## 2. 两个新块的几何

### 2.1 减速带（列 40–49，10 列完全相同）

尺寸常量放在 `connection_grid`（最底层模块，因为 `slope_degrees` 要报出该列真实坡度）：

| 量 | 值 | 说明 |
|---|---|---|
| 底 | **300 mm** | 沿 x 总宽 0.30 m |
| 顶 | **100 mm** | 平顶长 0.10 m |
| 高 | **50 mm** | 平顶高度 |
| 单侧斜坡 | 水平跑 100 mm、升 50 mm | 长度 `hypot(0.1, 0.05) = 0.1118 m` |
| **侧坡** | **26.565°** | `atan(50/100)`，= 该列 `slope_degrees` 报出的值 |
| 位置 | 居中在剖面区 | `x = (2.25 + 9.0)/2 = 5.625 m`，凸条占 5.475–5.775 m |
| 横向 | `\|y\| ≤ 2 m` 满高、外缘 1 m `cos²` 收零 | 减速带块内部 10 列接缝无落差 |

每条 lane 只有 1 条凸条（用户选定），一个回合过一次冲击。

### 2.2 波浪（列 50–59，10 列完全相同）

| 量 | 值 | 说明 |
|---|---|---|
| 窗口 | **2.25–9.0 m** | **与平坡同一位置**（`slope_geometry.UP_START_M`…`FLAT_OUT_START_M`）|
| 形状 | `h(u) = A·sin(2πu/λ)·env(u)` | 3 个整周期，等幅段 4 个极值满幅 |
| 周期 λ | **2.25 m** | 3 × 2.25 = 6.75 = 窗口长度（导入时断言） |
| 振幅 A | **10 cm** | 峰 +0.10 / 谷 −0.10（峰谷差 20 cm；谷底仍在板厚 35 cm 内） |
| 最大侧坡 | **15.603°** | `atan(2πA/λ)`，= 该列 `slope_degrees` 报出的值 |
| 两端 | 各 1.125 m（λ/2）`cos²` 包络 | 高度与**一阶导**都归零（与前后平地 C1 衔接） |
| 收尾 | 9.0 m 之后是平地 | 用户要的「最后用一个平地收尾」；10 m 的 STOP 点仍在平地上 |
| 横向 | `\|y\| ≤ 2 m` 满幅、外缘 1 m `cos²` 收零 | 与 rolling / 减速带同一套因子 |

注意：收零段里的第一个波峰是**半幅**（0.05 m），等幅段（u ∈ [1.125, 5.625]）里的 4 个极值才是
±10 cm。正弦在整周期上均值为 0 ⇒ 波浪板的**体积仍等于平板 31.5 m³**（峰谷相抵）。

### 2.3 接缝

`10° 坡列 ↔ 减速带列`、`减速带列 ↔ 波浪列` 在坡顶/波峰附近仍有矮坎（hill 剖面与 y 无关），
与既有的「平地↔5°」「5°↔10°」接缝同性质，靠中线 PD 保持车道；减速带块内部、波浪块内部
（都在 |y| = 3 收到 0）无落差。rolling / 减速带 / 波浪三族在 |y| = 3 处都是 0 ⇒ 族内拼接连续。

## 3. 实现

| 文件 | 改动 |
|---|---|
| `mdp/connection_grid.py` | `COLUMNS` 60、五个块常量与起点、`SPEEDBUMP_*` 与 `WAVE_*`（周期/振幅/派生最大坡度）、`terrain_family()` / `is_speedbump()` / `is_wave()`、`slope_degrees` 对后两块报真实坡度、`env_spec` 增 `terrain_family` |
| `mdp/speedbump_geometry.py` | 单条梯形（高/坡度/精确弧长/单板闭合网格） |
| `mdp/speedbump_torch.py` | 梯形向量版（与标量版逐位一致） |
| `mdp/wave_geometry.py` | **新增**：3 周期正弦 + `cos²` 包络（解析高度/坡度）、数值弧长上界、单板闭合网格 |
| `mdp/wave_torch.py` | **新增**：正弦向量版（与标量版逐点一致） |
| `mdp/mixed_geometry.py` | 按列分派四类 tile（平坡 / 减速带 / 波浪），唯一的整块 `grid_mesh(smooth)`，弧长上界取三族最大 |
| `slope_terrain.py` | 生成器 `TowingMixedTerrainGenerator`（hill 平滑段）/ `TowingMixedRollingTerrainGenerator`（rolling） |
| `upper_env_cfg.py` | `towing_terrain_profile ∈ {hill, rolling}`；timeout 用 `mixed_geometry.arc_length_upper_bound` |
| `upper_mdp.py` | 逐 env `terrain_speedbump` / `terrain_wave` 掩码，`robot_fall` 三层 `torch.where` 分派 |
| `scripts/towing/play_towing_test.py` | 逐列掩码 + `surface_heights()` 三族分派二维离面高度与相对俯仰、启动时**三族**交叉核对 |
| `tools/play_test_stats.py` | `segment_of` 支持减速带（单条梯形）与波浪（用**解析坡度**定上/下坡）、`profile_for_column()` 按列分派 |
| `tools/check_towing_slope_grid.py` | 离线检查挂上 `test_towing_wave_geometry` |

**timeout**：`mixed_geometry.arc_length_upper_bound` 取三族最大 ⇒ 由**波浪**给出
（**2026-10-17 起 STOP = 9.5 m**（越过障碍 9.0 m 再走 0.5 m）⇒ **9.616044 m**；10 m 目标时是
10.116044 m。对照：10° 坡列 9.592560、rolling 9.6041、减速带 9.523607）⇒
`episode_length_s = 1 + 9.616044/0.5 + 3 = 23.232088 s`、`max_episode_length` =
`ceil(24.232088/0.05)` = **485 步**。

**网格规模**（离线实算）：混合 hill 网格 **347,000 顶点 / 689,200 面**，混合 rolling 网格
**775,800 / 1,546,800**。单板：平地 8 / 12、平坡 16 / 28、减速带 161 / 318、波浪 1526 / 3048
（波浪是光滑曲面，纵向按 0.1 m 采样；减速带分段线性，只需 4 个折点）。

## 4. 验证

解释器：`/opt/conda/envs/isaaclab/bin/python3`（Python 3.11.15，含 torch）。

- `pytest imgo2_rl/tests -q` ⇒ **661 passed, 4 skipped, 1336 subtests passed**（本轮新增
  `test_towing_wave_geometry.py` 10 项）。
- `/usr/bin/python3`（无 torch）单跑波浪测试 ⇒ 9 通过 + 1 跳过；`--dry-run`（hill/rolling）退出 0。
- `check_towing_slope_grid.py` ⇒ **42 项 OK**。
- 数值复核：窗口 2.25–9.0（3 × 2.25）、振幅 ±0.10、解析最大侧坡 15.602533°（数值扫描一致、且
  极值出现在包络到 1 的边界）、首尾 C1、10 m 目标的弧长上界 10.116044 m（数值积分 ≤ 上界且差 < 2 cm；
  STOP 保持 10 m）、
  波浪板体积 31.5 m³、顶面面内高度误差 ≤ 1 mm；混合网格逐块顶点/面数 = 四类之和。
- `compileall`、`git diff --check`、`git ls-files -i -c --exclude-standard`、
  `check_asset_paths.py`（PASS）通过。

## 5. 训练机短跑入口

```bash
# 测试台：只核对计划，不启动仿真（1200 环境 = 60 列 × 20 行）
python imgo2_rl/scripts/towing/play_towing_test.py --dry-run

# 新运行先做短冒烟；60 环境覆盖整行（含 10 列减速带 + 10 列波浪）
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/rl_lab/towing/train.py \
  --task Imgo2-towing-upper-rl-lab --num_envs 60 --max_iterations 10 --headless

# 冻结底层基线（测试台按时间停车；波浪占 2.25–9.0、10 m 目标仍按名义时长给）
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/towing/play_towing_test.py \
  --num-envs 1200 --tow-duration 20 --coast-duration 5 --write-csv all --headless
```

历史 800/1000 环境（40/50 列）的 run 与新网格**不再是同一张图**：加列本身改变了 env → cell
映射，不能再逐 cell 与旧报告配对。

以上训练／仿真命令**本轮未执行**（本机无 Isaac Lab 运行环境）。

## 6. 待修复／待确认（TOW-BUMP-01 / TOW-WAVE-01）

状态：两者都是**已修，待动态验证**。缺训练机 Isaac Lab：

1. 网格导入/碰撞：减速带有两条竖直棱与 5 cm 台阶；波浪是光滑曲面但纵向采样 0.1 m、横向
   收零段 0.2 m（要确认 PhysX 不报退化面、足端/车轮不打滑到异常）；
2. 60 环境短冒烟：读回 `max_episode_length = 485`、出生首拍、过带与过波时的接触、`robot_fall`
   的三族离面高度分派；
3. 1200 环境冻结基线一轮：比较四个地形族的速度/张力/碰撞/停车读数（波浪应给出**周期性**载荷，
   减速带给出**离散冲击**，两者与平坡列形成对照）；
4. hill / rolling 两档各自重跑同参数两轮才能互相配对（跨地形不可比，测试台与
   `compare_play_runs` 已按地形身份拦截）。

已知限制：本轮**没有任何动力学证据**；三族之间的接缝矮坎是混合地形的固有台阶。
