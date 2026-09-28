# CMoE 地形与参考实现的结构对齐（2026-09-28）

## 1. 起因与决定

用户原话：「我看了一下 cmoe 的地形设置，我们四足这个地形还是差点，感觉可以对齐一下。」
随后给出的参考是项目页 <https://hoshi-no-ai.github.io/CMoE/> 上的 Code 链接，即
[`Hoshi-No-Ai/CMoE`](https://github.com/Hoshi-No-Ai/CMoE)（ICRA 2026，G1 人形），
并按用户要求克隆到本机 `/root/Desktop/CMoE`（`4575d6ae`，94 MB）。
此前 `/root/Desktop/CMoE` 放的是另一个同名空仓库（`Fudan-MAGIC-Lab/CMoE`，只有一句
“Comming Soon”），已删除后重新克隆；那个空仓库随时可再 clone 回来。

用户对本次对齐**划定的范围**（三问三答）：

| 问题 | 用户选择 |
|---|---|
| 米制缩放口径 | **只对齐结构**：采纳参考的种类／比例／行列数／难度律形式／初始等级，米制难度区间沿用我们已验证的四足值 |
| `flat` 列（参考 plane 比例是 0.0） | **保留 flat 0.10**（2026-09-24 我们自己加的平地列） |
| `sub_terrains` 键名 | **沿用我方现有键名 + 只新增**（奖励掩码／课程名单／测试改动最小） |

## 2. 参考的地形规格（逐条从源码读出）

来源文件（克隆在本机 `/root/Desktop/CMoE`）：

* 比例／行列数／初始等级：`legged_gym/legged_gym/envs/g1/g1_cmoe_config.py::class terrain`
* 每类难度律：`legged_gym/legged_gym/utils/humanoid_terrain.py::Terrain.make_terrain`
* 纵向障碍图案：`legged_gym/legged_gym/utils/parkour_terrain_utils.py`
* 基类默认值：`legged_gym/legged_gym/envs/base/legged_robot_config.py::class terrain`

| 项 | 参考值 |
|---|---|
| 瓦片 | `terrain_length = terrain_width = 10 m`；`horizontal_scale = 0.05`、`vertical_scale = 0.005`；`mesh_type = trimesh`、`slope_treshold = 1.5` |
| 行列 | `num_rows = 10`（行＝难度，`difficulty = row / 10` → `0.0…0.9`）、`num_cols = 40`（列＝类型，按比例累积和切）、`max_init_terrain_level = 5`、`curriculum = True` |
| 比例 | plane **0.0**／rough slope 0.1／stairs up 0.1／stairs down 0.1／discrete 0.1／**parkour_gap 0.3**／parkour_step_up 0／parkour_step_down 0／parkour_hurdle 0.1／mix 0.1／narrow_stairs 0.1／composite 0.0 |
| 粗糙度叠加 | 除 plane 外**每一类都叠加**随机粗糙：`rough_height = [0.01, 0.03]`、`max_height = 0.01 + 0.02·d`、step 0.005、`downsampled_scale = 0.075` |
| 难度律 | 坡 `slope = 0.4·d`（同列前半随机翻符号）；台阶 `step_height = 0.05 + 0.18·d`、`step_width = 0.30`、platform 3.0；discrete `0.05 + 0.10·d`（20 个 1–2 m 方块）；沟 `gap_size = 0.1 + 0.7·d`、`num_gaps = 4`、`platform_len = 1.0`、间距 `x_range = [0.8, 1.4]`、`gap_depth = [0.5, 1.5]`；hurdle 高 `[0.2·d, 0.15 + 0.25·d]`、石长 `0.1 + 0.2·d`、4 个、间距 `[1.2, 2.0]`、整宽；mix 固定图案（见 §4）、`diff = 1.1·d`；narrow_stairs `step_height = 0.25·d`、24 级、步深固定 `0.30`、半宽 `1 − 0.5·d`、`platform_len = 2.5` |
| 出生点 | 列 `choice < non_parkour_terrain (=0.5)` → 瓦片中心、`z` 取中心 2×2 m 最高点；`choice ≥ 0.5`（parkour 列）→ `x = 0.75 m`、`y = 瓦片中心`、`z = 0` |
| 参考机型 | Unitree G1，`base_height_target = 0.75 m`、初始 `pos z = 0.8 m` |

## 3. 我们的落地映射

生成器级（`CMoE_env_cfg.py::Imgo2CMoERoughEnvCfg.__post_init__`）：

| 项 | 改前 | 改后 | 依据 |
|---|---|---|---|
| `size` | (8.0, 4.0) | **(8.0, 4.0) 不变** | 瓦片尺寸属米制；我们的赛道按 8×4 设计并已验证 |
| `num_rows` | 10 | 10 不变 | 参考 10 |
| `num_cols` | 训练 20／play 10 | **40／40** | 参考 40（旧 play=10 时 `hf_pyramid_slope_inv` 只有 0 列） |
| `max_init_terrain_level` | 5 | 5 不变 | 参考 5 |
| 难度律形式 | 每类线性插值 | 不变（线性插值） | 参考同类；Isaac Lab 另有每行 `η~U(0,1)` 抖动 |

比例与类型（11 类，`Σ = 1.15`，Isaac Lab 会先归一化再切列）：

| 键名 | 参考对应 | 比例 | 米制难度区间（来源） |
|---|---|---|---|
| `pyramid_stairs` | stairs up | 0.10 | 0.05–0.15／级（**我们的四足值**，2026-09-24） |
| `pyramid_stairs_inv` | stairs down | 0.10 | 同上 |
| `boxes` | discrete | 0.10 | 0.08–0.30 m 高（我们的四足值） |
| `gap` | parkour_gap | **0.30**（与参考一致） | 0.126–0.315 m＝0.4–1.0 体长（我们的四足值） |
| `hurdle` | parkour_hurdle | 0.10 | 新增：参考 ×0.4 |
| `mix` | mix | 0.10 | 新增：参考 ×0.4 |
| `narrow_stairs` | narrow_stairs | 0.10 | 新增：参考 ×0.4 |
| `hf_pyramid_slope` | rough slope（一半） | 0.05 | 0–0.4（与参考同值） |
| `hf_pyramid_slope_inv` | rough slope（另一半） | 0.05 | 0–0.4 |
| `random_rough` | —（我们保留） | 0.05 | 0.01–0.06 m 噪声（我们的四足值） |
| `flat` | plane（参考 0.0） | 0.10 | —（用户 2026-09-28 决定保留） |

`num_cols = 40` 时的实际列数（`scripts/tools/check_terrain_columns.py`，实跑输出）：
`gap 11／pyramid_stairs 4／boxes 4／mix 4／pyramid_stairs_inv 3／narrow_stairs 3／flat 3／
random_rough 2／hf_pyramid_slope_inv 2／hf_pyramid_slope 1／hurdle 3`，
**每一类都 ≥1 列**，奖励掩码引用的 `boxes`／`gap` 都有命中对象。

## 4. 新增三类的缩放推导（参考公式 → 我们的常量）

`REFERENCE_SCALE = 0.30 / 0.75 = 0.4`（我们站高 0.30 m ÷ 参考 G1 `base_height_target` 0.75 m），
写在 `cmoe_terrains.py` 顶部，并且**只用于参考里没有四足对应值的新增类型**。

**`track_hurdle_terrain`（整宽薄横栏）** —— 参考 `parkour_hurdle_terrain(num_stones=4,
stone_len=0.1+0.2d, x_range=[1.2,2], hurdle_height_range=[0.2d, 0.15+0.25d], half_valid_width=[4,4.5])`，
其中一个 `half_valid_width` 远大于瓦片半宽 ⇒ 参考里横栏也是整宽的：

| 项 | 参考 | ×0.4 后（我们的常量） |
|---|---|---|
| 横栏数量 | 4 | 4 |
| 厚度 | `0.1 + 0.2·d` | `stone_len_range = (0.04, 0.12)` |
| 高度 | `[0.2·d, 0.15 + 0.25·d]` | `min_slope 0.08`／`max_base 0.06`／`max_slope 0.10` |
| 间距 | `[1.2, 2.0]` | `spacing_range = (0.48, 0.80)` |
| 起步平台 | 2.0 m | `platform_length = 0.80` |

与已有 `boxes` 的区别是**更薄更高**（`boxes` 厚 0.18–0.30 m、高 0.08–0.30 m）⇒ 是"跨栏"而不是"踩台阶块"。

**`track_mix_terrain`（窄走廊混合障碍）** —— 参考 `mix_obstacles_terrain` 是**硬编码固定图案**
（不是按参数生成的），水平索引单位 0.05 m、高度索引单位 0.005 m、走廊半宽 20 索引（=1.0 m）、
走廊外下沉、高度整体乘 `diff = hurdle_height_range[0]·1.1 = 1.1·d`：

```
0:30 → 0      30:36 → 30    36:42 → 60    42:48 → 90    48:60 → 120
60:(72-k) → 深坑              (72-k):84 → 120   84:86 → 0（参考里未被赋值，照抄）
86:96 → 96    96:99 → 170    99:111 → 120    111:(123-k) → 深坑
(123-k):140 → 120            140:160 → 60      160:以后 → 0        k = round(10 − 10·d)
```

×0.4 后：`x_unit = 0.02 m`、`z_unit = 0.002 m`、`corridor_width = 2×20×0.05×0.4 = 0.80 m`。
d=1 时最高台面 `170×0.002×1.1 = 0.374 m`、`120` 系列 `0.264 m`；d=0 时 `diff=0` ⇒ 只剩走廊与深坑
（与参考一致：难度 0 的地形就是"桥 + 坑"）。

**`track_narrow_stairs_terrain`（窄走廊楼梯）** —— 参考 `narrow_stairs_terrain(num_stones=24,
step_height=0.25·d, x_range=[0.30,1.5], half_valid_width=[1−0.5·d, 1.5−0.5·d])`，
其中步深取 `x_range[0]`、半宽取 `half_valid_width[0]`：

| 项 | 参考 | ×0.4 后 |
|---|---|---|
| 级数 | 24（前 10 上行、10–14 保持、其后 9 下行） | 24（同结构） |
| 步深 | 0.30 m | `step_depth = 0.12` |
| 步高 | `0.25·d` | `step_height_max = 0.10` ⇒ d=1 时 0.09 m／级、净升高 1.0 m |
| 起步平台 | 2.5 m（**整宽**，`[0:platform_len, :] = 0`） | `platform_length = 1.00`（整宽） |
| 走廊半宽 | `1 − 0.5·d` | `0.40 − 0.20·d` ⇒ 全宽 0.44–0.80 m（我们髋距 0.134 m、车宽 0.243 m） |
| 坑深 | `−randint(10,300)` 索引 = 0.05–1.5 m | 固定 `pit_depth = 0.50`（见 §5） |

## 5. 已知偏离（有意为之，逐条写清）

1. **粗糙度叠加没有移植**：参考给除 plane 外的每一类都叠加 `rough_height` 随机粗糙；我们的
   地形是 trimesh 盒子拼的，叠加粗糙度需要整批改成 heightfield 生成。本轮没有做，属**待办**
   （见 §8）。新增三类的几何是干净的盒子面。
2. **`mix` 图案整体平移 +0.30 m**：参考起步平台缩比后只有 0.60 m，装不下我们 0.75 m 的出生点
   （会出生在 0.12 m 高的台阶上）⇒ 图案整体后移，起步平台变 0.90 m，其余形状逐段一致。
3. **坑深取固定 0.50 m**：参考 `mix` 的坑深是 0.5–1.5 m（缩比 0.02–0.60 m）、`narrow_stairs` 是
   0.05–1.5 m 随机（缩比 0.02–0.60 m）。同一条赛道里出现 1.5 m 级的深坑没有教学意义，取中值 0.5 m。
4. **瓦片仍是 8×4 m**（参考 10×10 m）：参考的非 parkour 类型（坡／楼梯／discrete）是在
   10×10 瓦片中央的四面金字塔／二维方块阵，我们保留"纵向单列赛道"这一既有设计（这也与参考的
   parkour 类型同为纵向 +x 走法一致）。
5. **出生点仍统一 0.75 m、朝向锁 +x**：参考对非 parkour 列用"瓦片中心 + 局部最高点"出生；我们所有
   类型都是纵向赛道，出生点都在赛道起点平台上（0.75 m），不采用中心出生。
6. **每行难度的 `η` 抖动保留**：Isaac Lab 的 `difficulty = (row + η)/num_rows`（`η~U(0,1)`），
   参考是严格 `row/10`。属框架实现细节，未改。
7. **77 维地形扫描的网格没有动**：参考 `measured_points_x = [-0.1…0.9]`（中心 +0.4 m）、
   `measured_points_y = [-0.3…0.3]`；我们是 `offset.pos=(0.25, 0, 20)` + `size=(1.0,0.6)` 的 11×7
   ⇒ 覆盖 `x ∈ [-0.25, 0.75]`，**比参考靠后 0.15 m**。扫描属观测契约（训练／MuJoCo 部署／
   `policy/imgo2/cmoe/config.yaml` 三处一致），用户本轮范围是地形生成器，故未动；见 §8。

## 6. 影响面（连带改动）

* **奖励掩码**：`joint_mirror.bound_terrain_names=("gap",)`、`free_terrain_names=("boxes",)`、
  `feet_air_time`／`feet_height_body`／`feet_air_time_variance`／`feet_gait` 的
  `free_terrain_names=("boxes","gap")` **一字未改**（键名沿用）⇒ 新类型目前**照常吃**步态 shaping。
  是否把 `hurdle`／`mix`／`narrow_stairs` 也加进豁免名单属**配方决定**，本轮未动，留作待办。
* **课程**：判据（`tracking_move_up 0.80`／`tracking_move_down 0.35`／放宽阈值 0.40）**未改**；
  只把三个新障碍列加入 `relaxed_terrain_names`——否则它们会像 2026-09-25 的 `gap` 一样落进
  `[0.35, 0.80)` 冻结带、永远停在最窄的等级。
* **命令**：`forward_only_terrain_names` 补齐到 11 类（漏项会静默退回全向命令，
  `check_terrain_columns.py` 会校验）。
* **play**：`num_cols` 10 → 40（与训练一致）。
* **旧 run／checkpoint**：观测维数（527）不变，但地形列数从 20 变 40、种类从 8 变 11 ⇒
  **地形分布已不同**，旧 CMoE run 不应据此对齐后的配置继续训练或据其回放结论；
  已有 checkpoint 仍然只能搭配各自 `params/CMoE_env_cfg.py` 快照理解。

## 7. 验证（离线，本机 `python3` = `/opt/conda/envs/isaaclab/bin/python3`）

| 检查 | 命令 | 结果 |
|---|---|---|
| 编译 | `python3 -m compileall -q <改动目录>` | 通过 |
| 全仓离线测试 | `cd imgo2_rl && python3 -m unittest discover -s tests -q` | **418 通过／8 跳过**（改前 401／8；新增 8 项列数测试 + 9 项对齐测试 + 3 项赛道几何测试） |
| 列数／掩码覆盖率 | `python3 imgo2_rl/scripts/tools/check_terrain_columns.py` | 11 类各 ≥1 列、`forward_only` 覆盖全部 11 类、掩码引用的名字都有列，退出码 0 |
| 资产路径 | `python3 imgo2_rl/scripts/tools/check_asset_paths.py` | PASS |
| 空白／忽略规则 | `git diff --check`、`git ls-files -i -c --exclude-standard` | 干净／空 |
| 新类型几何实跑（离线 stub 载入真实源码） | `tests/test_track_geometry.py` | 出生点在起始平面上、越界为 0、走廊宽度＝难度律、坑底下沉、横栏整宽 |

**未运行（本机没有 Isaac Lab 运行环境）**：`import isaaclab.terrains` 在本机报
`ModuleNotFoundError: No module named 'omni.log'`（与既往记录一致）⇒ 地形生成器、11 类地形网格、
课程与短训**都没有在 Isaac 里构造过**。

## 8. 待办（缺什么才能完成）

1. **在 Isaac Lab 里构造一次**：跑 `--num_envs=4 --max_iterations=2` 的短训或 `zero_agent.py`
   看 11 类地形是否都生成、`hurdle/mix/narrow_stairs` 的几何是否可走、出生点是否都在平台上。
   需要：有 Isaac Lab + GPU 的运行环境。
2. **粗糙度叠加**：参考的 `rough_height` 叠加尚未移植。需要先把这几类改成 heightfield 生成
   （Isaac Lab `height_field` 里已有 `random_uniform_terrain` 等原语），并重新核对赛道几何测试。
3. **新障碍列的步态 shaping 豁免**：`hurdle`／`mix`／`narrow_stairs` 目前和 `pyramid_stairs` 一样
   吃 `feet_gait`／`joint_mirror`／`feet_air_time` 的 shaping；是否照 `gap`／`boxes` 豁免，要由
   训练读数（逐列 `gait_*`／`tracking_*`）决定。
4. **77 维扫描前移量**：我们比参考靠后 0.15 m（§5.7）。若要对齐需同时改训练配置、MuJoCo 部署
   与 `imgo2_deploy/policy/imgo2/cmoe/config.yaml`，并重训；属观测契约变更，需用户单独决定。
