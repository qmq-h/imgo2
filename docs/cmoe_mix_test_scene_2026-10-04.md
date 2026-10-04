# `mix` 受控测试场景（`Imgo2-basemove-rough-cmoe-mix-test`）— 2026-10-04

本文件记录**两批**改动（同日）：

* **第一批**：新增评测任务类 `Imgo2CMoEMixTestEnvCfg`（继承 play）——只有 `mix` 地形、速度只给前进
  且默认恒定 1.0 m/s、横向与航向由**指令层 PD 外环**控制、heading 恒 0；
* **第二批**（用户："地形不要按照列排，放在行里面" ＋ "都固定到14难度" ＋ "mix 中每个地形间隔大一点
  ×1.5~2.0"）：排布改成 **20 条并排的 mix 道**（沿世界 Y）× 20 档难度（沿世界 X），**全部环境固定
  在第 14 行**（名义 `d = 0.70`，不再是随机的 0–5 行），并把 mix 图案的**障碍间距乘子**设为 **2.0**。

**这不是新策略任务**：动作空间、观测组与维度、奖励、终止**一字未改**，只是换了一个场景 + 一层指令
外环 ⇒ 既有 CMoE checkpoint（77 维地形 / 527 维 actor / 125 维 critic）**可直接加载**。

## 1. 目的与范围

用途：在受控条件下评估策略通过 `mix`（复合障碍：窄走廊 + 台阶上行 + 深坑 + 高台 + 高栏 + 深坑）
的能力，**把横向漂移与航向漂移从评测里剔除**，并且**难度固定**（不再受课程随机初始等级影响）。
第二批进一步要求"**每条道一个环境**、道沿 Y **并排**"和"**障碍之间更从容**（间距 ×1.5~2.0）"。

## 2. 改动清单

| 文件 | 改动 |
|---|---|
| `.../velocity/mdp/mix_test_pd.py` | **第一批新增**：纯数学（只依赖 `torch`）——`lateral_heading_pd`、`rotate_base_to_world` 与默认增益常量 |
| `.../velocity/mdp/mix_test_command.py` | **第一批新增**：`MixTestVelocityCommand`（继承 Isaac Lab `UniformVelocityCommand`）+ `MixTestVelocityCommandCfg`（PD 增益字段） |
| `.../velocity/mdp/__init__.py` | 加 `from .mix_test_command import *` |
| `.../velocity/base_move/cmoe_terrains.py` | **第二批新增** `CMoETrackMixTerrainCfg.pattern_spacing_scale: float = 1.0`（默认值 ⇒ 训练几何**逐位不变**）＋ `track_mix_terrain` 里按该乘子放大"图案之间的 X 推进量"＋**总长溢出保护**（超长直接 `raise`，对照问题表 CMOE-13） |
| `.../velocity/base_move/CMoE_env_cfg.py` | **第一批**新增 `Imgo2CMoEMixTestEnvCfg(Imgo2CMoERoughPlayEnvCfg)`；**第二批**加模块级可读常量 `MIX_TEST_LANES/MIX_TEST_LEVELS/MIX_TEST_PINNED_LEVEL/MIX_TEST_PATTERN_SPACING_SCALE` ＋ 新类 `Imgo2CMoEMixTestTerrainImporter`（钉死难度行 + 冻结课程升降级）＋ 网格改 `20 × 20` |
| `.../velocity/base_move/__init__.py` | 注册 `Imgo2-basemove-rough-cmoe-mix-test`（按 `-play` 同族） |
| `imgo2_rl/scripts/rl_lab/cmoe/play.py` | **第二批**修 `--terrain_level` 的**过期帮助文本**（原写 0–9）：实测它 `type=int` 且**无 `choices`**、代码直接 `terrain.terrain_levels[:] = level` ⇒ **不夹取**，`num_rows=20` 时 14 合法；帮助文本改成 "0..num_rows−1（本仓 0..19）、不会被夹到 9" 并写明 mix-test 固定 14 |
| `imgo2_rl/scripts/tools/check_terrain_columns.py` | 第一批加 `--task`（按任务区分）＋解析改**类作用域**；第二批：mix-test 分支期望列数 `1 → 20`（并检查"mix 占满 20 道"），解析新增**模块级常量求值**（`num_cols` 现在写成 `MIX_TEST_LANES`） |
| `imgo2_rl/tests/test_cmoe_mix_test_scene.py` | 第一批 35 项；第二批 → **56 项**（新增真几何组 `TestMixTerrainGeometry`、难度固定/道数/乘子/世界范围断言、`--terrain_level` CLI 断言） |
| `imgo2_rl/tests/test_check_terrain_columns.py` | `TestMixTestTerrainColumns` 14 → **17 项**（列数 20、常量解析、道数负向对照） |

## 3. 场景契约

由 `Imgo2CMoEMixTestEnvCfg.__post_init__`（继承 `Imgo2CMoERoughPlayEnvCfg`）设定：

* **地形**：`self.scene.terrain.terrain_generator.sub_terrains.clear()` 后只放一处 `CMoETrackMixTerrainCfg`：

  ```python
  CMoETrackMixTerrainCfg(
      proportion=1.0,
      x_unit=0.02, z_unit=0.002, height_scale=1.1, gap_shrink_units=10.0,
      corridor_width=0.80, pit_depth=0.50, pattern_start_x=0.30,
      pattern_spacing_scale=MIX_TEST_PATTERN_SPACING_SCALE,   # = 2.0（第二批）
      spawn_x=0.75,
  )
  ```

  除 `proportion` 与 `pattern_spacing_scale` 两处**有意偏离**外**逐字沿用训练实例化时的值**（训练侧是
  `CMoETrackMixTerrainCfg(proportion=0.10)`，其余字段全取类默认值）——`tests/test_cmoe_mix_test_scene.py`
  会把这些**字面量**与 `cmoe_terrains.py` 的类默认值**逐一比对**，改默认值而不同步这里就会红。
  规格原文写的是 `prop_start_x=0.30`，本仓的真实字段名是 **`pattern_start_x`**。
* **网格**：`num_cols = MIX_TEST_LANES = 20`、`num_rows = MIX_TEST_LEVELS = 20`、`scene.num_envs = 20`
  ⇒ **20 条并排的 mix 道（沿世界 Y）× 20 档难度（沿世界 X）**，默认一条道一个环境。
* **难度固定 14**（`MIX_TEST_PINNED_LEVEL`）：`max_init_terrain_level = 14` ＋
  `scene.terrain.class_type = Imgo2CMoEMixTestTerrainImporter` ⇒ **不依赖 CLI**。
* **障碍间距 ×2.0**（`MIX_TEST_PATTERN_SPACING_SCALE`）：只影响本评测场景；训练侧不传该字段。
* **继承来的确定性**：`pose_range` / `velocity_range` 全 0、全部域随机化事件置 `None`、观测噪声关闭
  （`observations.policy/terrain.enable_corruption = False`）。
* **契约不变**：新类体里**没有**任何对 `self.observations` / `self.actions` / `self.rewards` /
  `self.terminations` / `self.curriculum` / `self.scene.height_scanner` 的赋值（测试里用 AST 逐条断言），
  也没有出现 `ObsGroup` / `ObsTerm` / `ActionCfg`。

### 3.1 速度指令

| 项 | 值 |
|---|---|
| `ranges.lin_vel_x` | **`(1.0, 1.0)`** ⇒ 恒定 **1.0 m/s** 前进（2026-10-04 用户规格修正） |
| `ranges.lin_vel_y` | `(0.0, 0.0)`（横向由 PD 每步写，范围只作初值/占位） |
| `ranges.ang_vel_z` | `(0.0, 0.0)`（航向由 PD 每步写，范围只作初值/占位） |
| `ranges.heading` | `(0.0, 0.0)` ⇒ heading 目标恒 0 |
| `heading_command` | `True`（规格要求；也是 `ranges.heading` 的合法性前提） |
| `rel_heading_envs` / `rel_standing_envs` | `1.0` / `0.0` |
| `resampling_time_range` | `(1.0e9, 1.0e9)` ⇒ **不做指令重采样**，整个 episode 指令恒定 |

`self.commands.base_velocity` 是**整项替换**成 `mdp.MixTestVelocityCommandCfg`（不是逐字段覆盖）：
新命令项需要 PD 增益字段，语义与 `UniformThresholdVelocityCommandCfg` 不同，逐字段覆盖容易漏
（参见 CMOE-04 的"晚赋值覆盖"教训）。替换后 `forward_only_terrain_names` 等阈值命令专属字段不再存在，
**本命令项根本不读它们**；`export_deploy_cfg.py` 只读 `ranges`（本 cfg 有），不受影响。

## 4. 排布、世界坐标与"每条道一个环境"

**方向是 Isaac Lab 的源码事实，不是本仓的约定**（`isaaclab/terrains/terrain_generator.py`）：

* `:247-261`（课程模式）逐列逐行生成瓦片，难度 `difficulty = (sub_row + U(0,1)) / num_rows`；
* `:330` 里 `terrain_origins[row, col]` 的平移量是 `((row+0.5)·size[0], (col+0.5)·size[1])`
  ⇒ **行（＝难度）沿世界 +X，列（＝地形类型/道）沿世界 +Y**；
* `:176-182` 整片地形再按 `(-size[0]·num_rows/2, -size[1]·num_cols/2)` 居中。

本场景 `size = (8.0, 4.0)`（父类 `Imgo2CMoERoughEnvCfg` 设定）、`num_rows = num_cols = 20`：

| 轴 | 范围 | 单块 | 本例 |
|---|---|---|---|
| 世界 X（难度行） | `[-80, +80]` | 每行 8 m | **第 14 行占 `[+32, +40]`**（`14×8 − 80` 起） |
| 世界 Y（mix 道） | `[-40, +40]` | 每道 4 m | 第 i 道占 `[4i − 40, 4(i+1) − 40]`（第 0 道 `[-40, -36]`，第 19 道 `[+36, +40]`） |

排布图（俯视，X 向右＝前进方向，Y 向上＝道）：

```
           道0   道1   道2   ...            道19          ← num_cols=20（沿 Y，每道 4 m）
  X=+80 ┌──────┬──────┬──────┬─────┬──────┐
  行19  │ mix  │ mix  │ mix  │ ... │ mix  │
   ...  │ ...  │ ...  │ ...  │ ... │ ...  │
  行14  │ mix  │ mix  │ mix  │ ... │ mix  │  ← 全部环境固定在这一行（d=0.70 名义）
   ...  │ ...  │ ...  │ ...  │ ... │ ...  │
  行0   │ mix  │ mix  │ mix  │ ... │ mix  │
  X=-80 └──────┴──────┴──────┴─────┴──────┘
```

### 4.1 确定性分配（明确结论：**能**，且**不需要额外代码**）

Isaac Lab 的 `TerrainImporter._compute_env_origins_curriculum`
（`isaaclab/terrains/terrain_importer.py:342-344`）里道号是**确定性**公式：

```
terrain_types[i] = floor(i · num_cols / num_envs)
```

本场景 `num_envs == num_cols == 20` ⇒ `terrain_types = [0, 1, …, 19]`，即 **环境 i → 第 i 道**，
一格一个环境、**不存在两个环境挤在同一块瓦片/出生点重叠**。这是"用 Isaac Lab 现成机制"得到的，
**没有**为本场景写任何道号分配代码。

* **约束（必须遵守）**：该式只在 **`num_envs ≤ num_cols`** 时严格递增
  （`terrain_types[i+1] − terrain_types[i] = floor((i+1)C/N) − floor(iC/N) ≥ floor(C/N) ≥ 1`）⇒
  `--num_envs ≤ 20` 时"一道一个环境"；**`--num_envs > 20` 就会出现重复取值**（多个环境落在同一条道的
  同一行、出生点重叠）。子类 `Imgo2CMoEMixTestTerrainImporter` 在构造时对这种情况打印
  `[WARN] … num_envs=… > num_cols=… ⇒ terrain_types 会重复 …`（fail-soft，只告警不终止）。
* `num_envs < 20` 时（例如 10）道号是 `[0, 2, 4, …, 18]`：**仍然互不重叠**，只是道与道之间跳着用
  （不是"只占前 10 道"）。这是 Isaac Lab 的等比铺法，不是 bug。
* 未验证项：以上都是**源码级**结论（本容器无 Isaac 运行环境），没有在真仿真里 `print` 过
  `terrain_types`。训练机回放时可用 `--num_envs=20` 然后看 20 个出生点是否分居 20 条道来验收。

## 5. 难度固定 14：怎么固定、以及"名义 vs 实际"

用户要求"都固定到14难度"。分三层说明（**前两层已落地为配置默认值，不依赖 CLI**）：

| 层 | 原状（第一批） | 现在（第二批） |
|---|---|---|
| 初始**等级** | 继承 play 的 `max_init_terrain_level = 5` ⇒ `randint(0, 6)`，20 个环境**随机落在 0–5 行** | `max_init_terrain_level = 14` **且**由 `Imgo2CMoEMixTestTerrainImporter.configure_env_origins` 把 `terrain_levels[:]` 直接写成 `pinned_level = 14` ⇒ **全部环境都在第 14 行** |
| 课程**升降级** | 每回合按 `terrain_levels_vel_logged` 判据升降（到顶还会 `randint_like` 随机重开） | 子类把 `update_env_origins` 改成**空操作** ⇒ 一局之内难度与出生点都不变（等价于 `play.py --terrain_level` 的冻结逻辑，但不需要 CLI） |
| **难度** | `difficulty = row / num_rows` | 第 14 行 ⇒ **名义 `d = 14/20 = 0.70`**，但 Isaac Lab 给同一行内每块瓦片加 `U(0,1)` 抖动：`difficulty = (sub_row + U(0,1)) / num_rows` ⇒ **实际 `d ∈ [0.70, 0.75)`** |

**为什么需要子类**：`max_init_terrain_level` 在 Isaac Lab 里只是"**上限**"（`terrain_importer.py:334-341`：
`max_init_level = min(max_init_terrain_level, num_rows−1)`、`terrain_levels = randint(0, max_init_level+1)`），
**无法**表达"全部钉在第 N 行"；所以第二批加了 `Imgo2CMoEMixTestTerrainImporter`
（`scene.terrain.class_type`，**只在本评测 cfg 里设置**；训练/play 的 `class_type` 仍是 Isaac Lab 的默认
`TerrainImporter`，行为一字未改）。它只重写两个方法：

```python
def configure_env_origins(self, origins=None):
    super().configure_env_origins(origins)
    ...
    self.terrain_levels[:] = int(self.pinned_level)                       # ① 钉死等级
    self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]

def update_env_origins(self, env_ids, move_up, move_down):                 # ② 冻结升降级
    return
```

**`--terrain_level=N` 仍然可用**（而且现在**不夹取**）：`play.py` 的 argparse 是 `type=int` + **无
`choices`**，L254 起直接 `terrain.terrain_levels[:] = int(args_cli.terrain_level)` 并重算 `env_origins`、
把课程项换成 no-op ⇒ `N=14` 合法（`num_rows=20` ⇒ 合法区间 `0..19`；原帮助文本里"0–9"是**过期的写法**，
不是夹取，已在第二批修掉）。所以：

```bash
# 标准调用（难度已由 cfg 固定为 14，不需要 --terrain_level）
python scripts/rl_lab/cmoe/play.py --task=Imgo2-basemove-rough-cmoe-mix-test --num_envs=20 --headless \
    --checkpoint="/absolute/path/to/model.pt"
# 等价写法（显式钉 14；想扫别的档位时才需要传，例如 --terrain_level=0）
python scripts/rl_lab/cmoe/play.py --task=Imgo2-basemove-rough-cmoe-mix-test --num_envs=20 --headless \
    --terrain_level=14 --checkpoint="/absolute/path/to/model.pt"
```

**要不要把 `d` 精确到 0.70？** 要精确就得把 `terrain_generator.difficulty_range` 收成 `(0.70, 0.70)`
（那样 `difficulty = 0.70` 与行号无关）。代价：**20 行的难度会全部相同**、`--terrain_level` 失去"扫难度"
的意义 ⇒ 本类**有意不做**（用户没有要求"精确 0.70"，只要求"固定到 14 难度"；保留行内抖动也让评测不是
20 块完全相同的瓦片）。这是一条**待决定项**，改法是一行：
`self.scene.terrain.terrain_generator.difficulty_range = (0.70, 0.70)`。

## 6. 障碍间距乘子 `pattern_spacing_scale`（第二批）

`track_mix_terrain` 原本把图案逐段放在 `x0 = pattern_start_x + start_units · x_unit`。新字段把
"**图案之间的 X 推进量**"乘上该乘子，而**图案自身的宽度不动**：

```
shift(start) = (scale − 1) · start · x_unit
x0 = pattern_start_x + start · x_unit + shift(start)
x1 = min(pattern_start_x + end · x_unit + shift(start), size[0])      # 宽度 = (end − start)·x_unit，与 scale 无关
```

* **默认 `scale = 1.0` ⇒ `shift` 精确等于 `+0.0`** ⇒ 生成结果与加字段之前**逐位相同**（硬要求：
  训练配置不得受影响）。已用"拿 `b2caad1` 的源码与改后源码各跑一遍、逐块比较 `mesh.bounds`"验证：
  `d ∈ {0, 0.35, 0.70, 0.75, 1.0} × size ∈ {(8,4), (8,8)}` 共 10 组**全部逐位一致**。
* **不改**障碍自身的尺寸/高度/图案顺序（测试断言每块宽度、顶面高度、y 范围在 `scale=2.0` 下不变，
  且起点单调递增、互不重叠）。
* 训练侧**不传**该字段（`CMoETrackMixTerrainCfg(proportion=0.10)`）⇒ 走默认 1.0。**间距乘子只是评测用的
  放宽，训练仍用 1.0。**

### 6.1 总长核算（必须装得进 8 m 瓦片）

图案的最后一个索引是 **160**（末段 `140:160`），所以

```
图案 X 末端 = pattern_start_x + 160 · x_unit · scale = 0.30 + 3.20 · scale   [m]
```

| scale | 图案末端 | 尾廊（→8.00 m） | 装得下？ |
|---|---|---|---|
| 1.0（训练/改动前） | 3.50 m | 4.50 m | ✅ |
| 1.5（用户区间下限） | 5.10 m | 2.90 m | ✅ |
| **2.0（本场景选用）** | **6.70 m** | **1.30 m** | ✅ |
| 2.40625（**上限**） | 8.00 m | 0 m | ✅（等号边界） |
| 2.5 | 8.30 m | — | ❌ **raise** |

**结论（B3）**：`scale = 2.0` 时图案总长 **6.70 m ≤ 8 m 瓦片**，安全余量 1.30 m，**不需要**动
`terrain_generator.size[0]`、也不需要减图案数量。该瓦片上乘子的**最大值 = (8 − 0.30) / (160 × 0.02)
= 2.40625**（用户区间 1.5~2.0 完全在内）。

**溢出保护**：超过 `size[0]` 时 `track_mix_terrain` **直接 `raise ValueError`**，消息里给出实际总长、
瓦片长度与"该瓦片上的乘子上限"，并列出三个可选方案（减图案/加 `size[0]`/降乘子）——**不静默截断**
（这正是问题表 CMOE-13 在 `track_gap_terrain` 上缺失的保护；本次**没有**顺手改 `track_gap_terrain`，
它仍留在问题表里）。

### 6.2 ⚠️ 语义警告：被拉开的空间是**深坑**，不是平地跑道

`mix` 瓦片的**整宽底面本身就在 `-pit_depth = -0.50 m`**（`_platform(0, size[0], …, -pit_depth)`），
而走道（`_corridor`，宽 0.80 m）**只铺在 `segments` 那几段上**。因此把图案拉开后，**段与段之间新空出来的
X 区间由原来的坑底填充**，表现为**更长的整宽深坑**（不是"多一段平地"）。用真几何实测（桩 isaaclab +
真 trimesh，`d = 0.70`）：

| | scale = 1.0（训练原样） | scale = 2.0（本场景） |
|---|---|---|
| 图案内最大"缝"（＝深坑宽） | **0.18 m** | **0.60 m** |
| `d=0.70` 的坑数/位置 | 2 处（`1.50→1.68`、`2.52→2.70`） | 3 处（`0.90→1.50`、`2.46→3.06`、`4.50→5.10`） |
| 图案总长 | 3.50 m | 6.70 m |
| 尾廊 | 3.50→8.00 m | 6.70→8.00 m |

（第 3 处"新坑"出现在起步走廊的末端：段 `0:30` 的**宽度不缩放**（0.60 m），而到下一段的**推进量**
×2 ⇒ 原来与台阶相接的 `0.90 m` 处变成 0.60 m 的整宽坑。这是"只乘推进量、不乘宽度"的必然结果。）

**这条与用户的说法有出入，按要求如实报告**：用户把它描述为"模拟障碍之间更从容"，但按**字面**实现
（只乘 X 推进量、不改几何）得到的是**更长/更多、仍深 0.50 m 的整宽坑** ⇒ 地形**更难**，不是更从容。
本轮的取舍是"**严格照字面实现 + 如实报告**"，没有擅自改成"新空间铺平地"（那会新增几何、改变参考图案的
含义）。**待用户决定**的可选改法（都只改 `track_mix_terrain`，一行级别）：
① 把段与段之间空出来的区间补成 `height=0` 的走廊（真正"平地跑道"）；
② 只对**本来就相接**的段插入平地、保留原有两处深坑的难度律；
③ 把乘子降到 1.5（缝最大值 ≈0.39 m）看策略能否承受。当前 cfg 的乘子是一个数字：`MIX_TEST_PATTERN_SPACING_SCALE`。

## 7. 命令项设计（`mdp/mix_test_command.py`）

1. **继承 `UniformVelocityCommand`**：复用它的 `vel_command_b` / `heading_target` /
   `is_heading_env` / `is_standing_env` / `metrics` 与 `_update_metrics`，只重写
   `_resample_command` / `_update_command`。
2. **`vx_cmd` 恒定**：`super()._resample_command()` 采样**一次**（`lin_vel_x=(1.0,1.0)`），
   `_update_command` **不碰第 0 列**（测试断言 `vel_command_b[:,0]` 恒为 1.0）。
3. **横向 PD**（每步重算，PD 是时变的）：写 `vel_command_b[:,1]`。
4. **航向 PD**：写 `vel_command_b[:,2]`。
5. **写回 `vel_command_b` 与 `vel_command_w`**：奖励（`track_world_vel_xy_exp` 等）与观测读的
   仍是 `command`（＝`vel_command_b`）；`vel_command_w` 是**世界系镜像**，见 §9。
6. **纯函数化**：PD 数学在 `mdp/mix_test_pd.py`（只依赖 torch），可离线单测；命令项只负责
   把 `root_pos_w` / `root_lin_vel_w` / `heading_w` / `root_ang_vel_b` 接上去。

### 7.1 坐标系约定（**必读**）

| 变量 | 取法 | 为什么 |
|---|---|---|
| `y`（`y_local`） | **世界系**横向位置 − 本环境出生原点：`root_pos_w[:,1] − env.scene.env_origins[:,1]` | `env_origins` 是该 env 的**道中心** ⇒ 这个差值就是"离赛道中心线的横向距离"（出生在走廊中心 y=0） |
| `vy`（`vy_local`） | **世界系**横向线速度：`root_lin_vel_w[:,1]` | 与 `y` 同坐标系，也与评测奖励 `track_world_vel_xy_exp`（**世界系**，2026-09-24 起）一致 |
| `yaw` | `robot.data.heading_w` | Isaac Lab 由 **`root_quat_w`** 把机体系 +x 投到世界系后 `atan2(y, x)` ⇒ 就是偏航角，且已绕回 `(−π, π]` |
| `wz`（`wz_local`） | **机体系**偏航角速度：`root_ang_vel_b[:,2]` | 命令 `vel_command_b[:,2]` 写的就是机体系角速度；既有 `error_vel_yaw` 度量也用它 |

第二批之后 `env_origins` **一局内不再变化**（课程被冻结）⇒ 横向误差的参考点稳定（此前每回合升降级会
让 `env_origins` 跳到另一行/另一块瓦片）。

**备选（未采用，记录以便复核）**：把 `vy` 换成机体系 `root_lin_vel_b[:,1]`。在 yaw 已被航向 PD 压到
接近 0 时两者几乎等价；选世界系是为了与 `y`（世界系位置误差）和世界系速度奖励同口径。

### 7.2 为什么不用内置 heading 控制器

`UniformVelocityCommand._update_command` 在 `heading_command=True` 时用**内置 P 控制器**
（`vel_command_b[:,2] = clip(heading_control_stiffness · heading_error, ang_vel_z[0], ang_vel_z[1])`）
写 `wz`。本任务：

* `ranges.ang_vel_z = (0,0)` ⇒ 内置控制器会把 `wz` **恒夹成 0**，航向保持根本没有输出；
* 内置控制器只有 P 项，没有 `wz` 阻尼，压不住高速下的航向振荡。

因此本命令项**不使用**内置控制器，而是每步用自己的 PD 直接写 `vel_command_b[:,2]`。
**cfg 里 `heading_command` 仍保持 `True`**——规格要求它，而且 `UniformVelocityCommand.__init__` 用它做
`ranges.heading is not None` 的合法性检查。`heading_control_stiffness` 对本命令项**无作用**。

## 8. PD 公式、默认增益与调法

```
vy_cmd = clip( kp_y · (y_des − y)        + kd_y · (0 − vy),   −vy_max, +vy_max )
wz_cmd = clip( kp_h · (yaw_des − yaw)    + kd_h · (0 − wz),   −wz_max, +wz_max )
```

默认增益（**用户给定**，全部走 `MixTestVelocityCommandCfg` 的字段 ⇒ 可覆盖）：

| 参数 | 默认 | 单位 | 含义 |
|---|---|---|---|
| `kp_y` | **1.0** | 1/s | 横向位置比例增益（1 m 偏移 ⇒ 1 m/s 指令） |
| `kd_y` | **0.3** | — | 横向速度阻尼 |
| `vy_max` | **0.6** | m/s | 横向指令上限 |
| `y_des` | **0.0** | m | 横向目标（贴赛道中心线） |
| `kp_h` | **1.5** | 1/s | 航向比例增益（1 rad 误差 ⇒ 1.5 rad/s 指令） |
| `kd_h` | **0.3** | — | 航向阻尼 |
| `wz_max` | **1.0** | rad/s | 航向指令上限 |
| `yaw_des` | **0.0** | rad | 航向目标（始终朝世界 +x） |

### 怎么调

* **改前进速度**：`CMoE_env_cfg.py` 里 `ranges.lin_vel_x=(1.0, 1.0)` 那一行。想做速度扫描就把它改成
  区间（如 `(0.5, 1.0)`）**并**把 `resampling_time_range` 缩短（否则整回合只采样一次）。
* **改难度**：默认已是 14；要扫别的档位就传 `--terrain_level=N`（`0..19`，不夹取）。
* **改间距**：`MIX_TEST_PATTERN_SPACING_SCALE` 一个数字（≤ 2.40625；超过会 `raise`）。
* **改 PD 增益**：目前写在 `MixTestVelocityCommandCfg` 的字段默认值里；要按 run 调，在
  `Imgo2CMoEMixTestEnvCfg.__post_init__` 构造 cfg 时传参即可（例如 `kp_y=1.5, vy_max=0.8`）。
* **调参直觉**：`kp_y` 大了会来回摆（`kd_y` 要跟着加）；`vy_max` 是"允许用多大横移去纠偏"的上限；
  参考量级：站立高度 0.30 m、走廊半宽 0.40 m，而 `vy_max=0.6 m/s` 已接近行走速度（1.0 m/s）的 60%，
  因此**横向纠偏不能指望它救回大偏差**——本场景的前提是"横移本来就不该发生"。
* **关掉转向、只测横向**：把 `kp_h`/`kd_h` 置 0 即可。

## 9. `vel_command_w`（**与规格的一处冲突，已按"不改设计、只报告"处理**）

规格要求"把 `vy_cmd`/`wz_cmd` 写进该 command 的 `vel_command_b`/`vel_command_w`，
**与既有 `base_velocity` 的写法一致**"。但事实是：**基类 `UniformVelocityCommand` 与本仓都没有
`vel_command_w`**（全仓 `grep -rn vel_command_w` 在本次改动前为空）。处理方式：

* `vel_command_b` 是真正生效的缓冲（奖励、观测、度量都读它）；
* `vel_command_w` 由本命令项**新增**为纯附加缓冲，每步写入"机体系指令按当前 yaw 旋转到世界系"的镜像；
* 它**不进入观测、不进入动作空间、不参与奖励**，因此**不影响 checkpoint 兼容性**，仅供回放/审计读。

## 10. 运行示例

回放/评测（须替换 checkpoint 绝对路径；**难度默认已固定 14**）：

```bash
cd imgo2_rl
python scripts/rl_lab/cmoe/play.py \
    --task=Imgo2-basemove-rough-cmoe-mix-test \
    --num_envs=20 --headless \
    --checkpoint="/absolute/path/to/model.pt"

# 显式钉难度（等价；扫别的档位时才需要）
python scripts/rl_lab/cmoe/play.py \
    --task=Imgo2-basemove-rough-cmoe-mix-test \
    --num_envs=20 --headless --terrain_level=14 \
    --checkpoint="/absolute/path/to/model.pt"
```

* `--num_envs=20`（＝`MIX_TEST_LANES`）时**一道一个环境**（环境 i → 第 i 道，沿 Y 并排）；
  `--num_envs > 20` 会告警且出生点重叠。
* `--terrain_level=N` ⇒ 全部环境钉在第 N 行 ⇒ **名义难度 N/20**（`0..19`，**不会**被夹到 9）。
* 也可用 `--force_expert K` 等既有 CMoE play 开关（本任务不引入新开关）。

## 11. 地形列校验：按任务区分

`scripts/tools/check_terrain_columns.py` 的原判据是"**所有被掩码引用的地形名都必须 ≥1 列**"。
本测试场景**只有 `mix`** ⇒ 掩码引用的 `boxes` / `gap` / `flat` 在该场景**不存在**，引用它们的项
（`joint_mirror`、`feet_air_time`、`feet_height_body`、`feet_air_time_variance`、`feet_gait`、
`base_height_flat_l2`）在本场景**恒为 0**——这是**预期**，不是配置错误。

处理方式（用户要求"不削弱原判据"）：

* 新增 `--task`，默认 `cmoe-rough` **走原判据、一字未放宽**；
* `--task mix-test`（＝ `Imgo2-basemove-rough-cmoe-mix-test`）走**单独分支**：
  仍强制 `sub_terrains.clear()` + 只有 mix + **mix 占满 20 道** + `num_cols=20` + `num_rows=20`，
  并把"掩码引用但本场景不存在"的名字**逐条打印为预期**；
* 解析从"整文件 `ast.walk`"改成**类作用域**（只走指定类的 `__post_init__`）——否则 mix-test 类里的
  `sub_terrains["mix"] = ...(proportion=1.0)` 会**静默污染**训练侧的比例解析；
* **第二批**：`num_cols`/`num_rows` 现在写成**模块级可读常量**（`MIX_TEST_LANES` 等），工具的解析
  新增"顶层 `NAME = <字面量>` 常量求值"，否则读到的是名字字符串；
* 测试对 test 任务**单独断言**，并加**负向对照**（把"只有 mix"丢给原判据必须仍报错、存在但 0 列也必须
  报错、**期望道数被改成别的值时必须报 ❌**）证明判据没被放宽。

实测输出（`python3 scripts/tools/check_terrain_columns.py --task mix-test`，退出码 0）：

```
=== [Imgo2-basemove-rough-cmoe-mix-test] 受控测试场景（只 mix；cfg 类 Imgo2CMoEMixTestEnvCfg）===

=== Imgo2-basemove-rough-cmoe-mix-test｜只 mix（20 道并排）（num_cols=20，共 20 列）===
  mix                     20 列
  ✅ 唯一地形 mix 有 20 列（num_cols=20、num_rows=20）⇒ 网格 20 道 × 20 难度行，`--num_envs ≤ 20` 时环境 i → 第 i 道
  ⚠️ 本场景**只有 mix**；掩码引用但本场景**不存在**的地形名（原任务判据**不放宽**，这些项在本场景恒为 0 属**预期**）：
    - joint_mirror.free_terrain_names → boxes：本场景无该地形 ⇒ 该项本场景恒为 0（预期，不是配置错误）
    ...（共 11 条，逐条列出）
  ✅ `forward_only_terrain_names`（继承父类） 覆盖 mix；另含 10 个本场景不存在的地形名（...）—— 命令项
     `MixTestVelocityCommand` 根本不读这张表，且 `is_env_assigned_to_terrain` 对未登记的名字返回全 False
     ⇒ 在这里是**惰性**的，属预期

结论：test 任务只有 mix 一种地形；掩码引用的其它地形名在本场景恒为 0，属**预期** ✅
```

## 12. 未验证项与与设计冲突之处

**未验证（本容器无 GPU / 无 Isaac 运行环境：`NVIDIA_VISIBLE_DEVICES=void`、`import isaaclab` 缺 `omni.log`）**：

1. **未构造 Isaac 环境**：`scene.terrain.class_type` 换成子类后 `InteractiveScene` 的构造路径、
   `MixTestVelocityCommandCfg` 在真实 `CommandManager` 下的解析、`debug_vis=True` 在 20 环境下的
   marker 创建都未实测。
2. **未回放、未导出 `policy.pt`**：checkpoint 兼容性是**静态推理**（动作/观测契约未改）。
3. **PD 增益未标定**：`1.0/0.3/0.6`（横向）与 `1.5/0.3/1.0`（航向）**未在仿真里标定**。
4. **`terrain_types` 的"一道一个环境"未在真仿真里打印确认**（结论来自 Isaac Lab 源码公式，见 §4.1）。
5. **20 道 × 20 行的网格、`pattern_spacing_scale=2.0` 的真实瓦片未生成过**：几何结论来自**用桩
   `isaaclab` + 真 `trimesh` 实跑 `track_mix_terrain`**（不是 Isaac 的完整地形管线），
   因此"瓦片拼接、缓存、碰撞体"这些层未验证。
6. **奖励掩码在本场景恒为 0 的实际影响未测**：6 项掩码项恒 0 ⇒ 评测时 `Episode_Reward/*` 里这些键的
   读数不能与训练 run 直接对比（只是读数问题，不是 bug）。
7. **一处已知的读数副作用（有意保留）**：基类 `_update_metrics` 用
   `max_command_step = resampling_time_range[1] / step_dt` 做归一化；本任务
   `resampling_time_range = (1e9, 1e9)` ⇒ `error_vel_xy` / `error_vel_yaw` 两个**度量**读数≈0
   （只进日志）。要可读的跟踪误差请用奖励分项 `Episode_Reward/track_world_vel_xy_exp`。

**与设计/规格的冲突（未擅自改设计，按要求报告）**：

1. **`vel_command_w` 不存在**：规格假定它与 `vel_command_b` 一样是既有写法，实际基类与本仓都没有。
   已按 §9 作为**纯附加**缓冲实现。
2. **"每环境一行"在第一批不成立**（第二批已解决）：继承来的 `max_init_terrain_level = 5` 只表达上限
   ⇒ 第一批的默认是 **0–5 行随机抽**。第二批用 `Imgo2CMoEMixTestTerrainImporter`（钉死 14 + 冻结）
   ＋ Isaac Lab 现成的 `terrain_types` 确定性分配（环境 i → 道 i）解决；**代价**是新增一个
   `scene.terrain.class_type` 子类（只在本评测 cfg 里生效）。
3. **难度是"名义 0.70"**：第 14 行内每块瓦片的实际 `d ∈ [0.70, 0.75)`（Isaac Lab 的行内抖动）。
   要精确 0.70 就得把 `difficulty_range` 收成 `(0.70, 0.70)`（会让 20 行难度相同）——**待决定**。
4. **间距乘子把"缝"变成更长的深坑**（§6.2）：按字面实现 ⇒ 地形**更难**，与"更从容"的意图相反。
   三条可选改法已列在 §6.2，**待用户决定**；本轮没有擅自改几何。
5. **`track_gap_terrain` 的溢出保护仍未做**（问题表 CMOE-13）：本轮只给 `track_mix_terrain` 加了保护，
   没有顺手扩到 `track_gap_terrain`（超出本次范围）。
6. **`check_reward_overrides.py` 未加 `cmoe-mix-test` 链**：本场景**完全不改奖励**（生效项仍是 `cmoe`
   的 28 项），加一条与 `cmoe` 逐项相同的链没有信息量，故未加。

## 13. 验证与实测输出（本机离线）

```bash
cd imgo2_rl
/opt/conda/envs/isaaclab/bin/python -m unittest discover -s tests     # 632 通过 / 10 跳过
python3 scripts/tools/check_reward_overrides.py cmoe-gaitfree          # 24 项（cmoe = 28 项，与改前一致）
python3 scripts/tools/check_terrain_columns.py                         # 末行"全部掩码引用的地形名都有 ≥1 列 ✅"
python3 scripts/tools/check_terrain_columns.py --task mix-test         # "mix 20 列"、末行"…属预期 ✅"（退出码 0）
git diff --check                                                       # 干净
```

* **基线**：第二批改动前 `unittest discover -s tests` = **608 通过 / 10 跳过**（第一批的 35+14 项）；
  第二批后 = **632 / 10**（`test_cmoe_mix_test_scene.py` 35 → **56**、`TestMixTestTerrainColumns`
  14 → **17**）。
* **"默认 1.0 无差异"的验证方式**：把 `b2caad1` 的 `cmoe_terrains.py` 与改后版本分别 exec（桩
  `isaaclab` + 真 `trimesh`），对 `d ∈ {0, 0.35, 0.70, 0.75, 1.0} × size ∈ {(8,4), (8,8)}` 逐块比较
  `mesh.bounds` 与 `origin`：**10/10 逐位相同**；测试里把这条固化成了 `TestMixTerrainGeometry`
  （用改动前的公式表当参照，不依赖具体 `trimesh` 版本的哈希值）。
* **溢出保护实测**：`scale=2.5`（`size[0]=8`、`d=0.70`）抛
  `ValueError: track_mix_terrain: 图案总长 8.3000 m 超出瓦片长 size[0]=8.0000 m …本瓦片上 pattern_spacing_scale 最大可取 2.4062…`；
  `scale=2.40625` 不抛。
* **新增测试覆盖（第二批）**：`num_cols=20`/`num_rows=20`（读常量）、道数常量注释里写明"可同时评估的
  环境数上限"、`num_envs=20`、难度钉 14、`class_type` 指向子类且子类继承 `TerrainImporter`、子类
  `configure_env_origins` 写 `terrain_levels`＋`update_env_origins` 只有 `return`、`num_envs > num_cols`
  告警、训练/play 的 `class_type` 未被触碰、乘子 =2.0 且训练侧不传、总长 6.70 m（余 1.30 m）与上限
  2.40625、真几何（默认无差异 / 只改间距 / 顺序不重叠 / 总长核算 / 溢出 raise / 上限边界 / 保护公式
  源码级锁定）、`play.py --terrain_level` 无 `choices`＋帮助文本不再写 0–9。
