# `mix` 受控测试场景（`Imgo2-basemove-rough-cmoe-mix-test`）— 2026-10-04

本文件记录**同一天的三批**改动（前三批都在 `imgo2_CMoE` 分支上，HEAD `1839a12` 之后未提交）：

* **第一批**：新增评测任务类 `Imgo2CMoEMixTestEnvCfg`（继承 play）——只有 `mix` 地形、速度只给前进
  且默认恒定 1.0 m/s、横向与航向由**指令层 PD 外环**控制、heading 恒 0；
* **第二批**（用户："地形不要按照列排，放在行里面" ＋ "都固定到14难度" ＋ "mix 中每个地形间隔大一点
  ×1.5~2.0"）：排布改成 **20 条并排的 mix 道**（沿世界 Y）× **20 档难度**（沿世界 X），全部环境
  固定在第 14 行（名义 `d = 0.70`），并把 mix 图案的**障碍间距乘子**设为 **2.0**。
* **第三批（本轮）**（用户："分居了，但是我不需要还保持那么多行，我需要他们并列"；随后追加
  "各个难度间距先不要调整，回复吧"）：

  ① **只留 1 行难度 × 20 条并列的道** —— `num_rows = 1`、`num_cols = 20`；
  ② 世界由 **160 m(X) × 80 m(Y)** 缩到 **8 m(X) × 80 m(Y)**（19 行用不到的难度行被删掉）；
  ③ 难度不再靠"把课程等级钉在第 14 行"，改由 **`terrain_generator.difficulty_range = (0.70, 0.70)`
     精确固定**（旧方案的实际难度是 `d ∈ [0.70, 0.75)`，只是名义 0.70）；
  ④ **删掉钉等级的子类** `Imgo2CMoEMixTestTerrainImporter`，`class_type` 恢复为默认 `TerrainImporter`
     （`num_rows = 1` 时课程天然只有 0 级）；
  ⑤ 间距乘子 **2.0 → 1.0**（＝训练默认值）⇒ 评测几何与训练**逐位一致**；`pattern_spacing_scale`
     字段本体、默认值 1.0 与 `track_mix_terrain` 的**总长溢出保护**都保留（中性基础设施）。

**这不是新策略任务**：动作空间、观测组与维度、奖励、终止**一字未改**，只是换了一个场景 + 一层指令
外环 ⇒ 既有 CMoE checkpoint（77 维地形 / 527 维 actor / 125 维 critic）**可直接加载**。

## 1. 目的与范围

用途：在受控条件下评估策略通过 `mix`（复合障碍：窄走廊 + 台阶上行 + 深坑 + 高台 + 高栏 + 深坑）
的能力，**把横向漂移与航向漂移从评测里剔除**，并且**难度固定**（不再受课程随机初始等级影响）。
第三批进一步要求"**道并列、但不要那么多行**"，于是整个世界只剩**一行**难度、20 条道沿 Y 并排。

## 2. 改动清单（第三批）

| 文件 | 改动 |
|---|---|
| `.../velocity/base_move/CMoE_env_cfg.py` | 常量 `MIX_TEST_LEVELS = 20 → **1**`；新增 `MIX_TEST_DIFFICULTY = 0.70`、删 `MIX_TEST_PINNED_LEVEL`；`MIX_TEST_PATTERN_SPACING_SCALE = 2.0 → **1.0**`；**删除** `Imgo2CMoEMixTestTerrainImporter`；`Imgo2CMoEMixTestEnvCfg` 加 `difficulty_range = (0.70, 0.70)`、`max_init_terrain_level = 0`、不再设 `class_type`；头注释与世界坐标整段按新事实重写 |
| `.../velocity/base_move/cmoe_terrains.py` | **未改**（间距乘子语义与默认 1.0、溢出保护都保留；本轮**没有**实现"补空档为可走面"） |
| `imgo2_rl/scripts/rl_lab/cmoe/play.py` | 只改 `--terrain_level` 的**帮助文本**：写明 `num_rows=1` 的 mix-test 只有 `0` 合法、其难度来自 `difficulty_range=(0.70,0.70)`，该参数对本任务**不要传**；行为一行未改 |
| `imgo2_rl/scripts/tools/check_terrain_columns.py` | mix-test 分支：`MIX_TEST_LEVELS_EXPECTED = 1`；新增 `MIX_TEST_DIFFICULTY_EXPECTED = 0.70` 与**难度范围解析/检查**；成功行改写为"网格 **20 道 × 1 难度行**（世界 8 m(X) × 80 m(Y)）" |
| `imgo2_rl/tests/test_cmoe_mix_test_scene.py` | 56 → **72 项**：全套断言改单行事实（`num_rows=1`、难度来源 = `difficulty_range`、世界 X ∈ [−4,+4]、Y ∈ [−40,+40]）；新增 `TestMixTestDifficultyPinning`（难度确实 = 0.70，两条路径 + Isaac Lab 源码级依据）、`TestMixTestSpawnGeometry`（出生点真几何）、`TestTrainingSpawnHazard`（训练侧同类隐患量化） |
| `imgo2_rl/tests/test_check_terrain_columns.py` | `TestMixTestTerrainColumns` 17 → **20 项**（`num_rows=1`、难度范围被检查、行数/难度的负向对照） |

第二/第一批的改动清单（新增 `mdp/mix_test_pd.py`、`mdp/mix_test_command.py`、`pattern_spacing_scale`
字段、`--task` 分支、测试从 35 → 56 项等）见本文件的 git 历史与 README §8.3 的对应两行。

## 3. 场景契约

由 `Imgo2CMoEMixTestEnvCfg.__post_init__`（继承 `Imgo2CMoERoughPlayEnvCfg`）设定：

* **地形**：`self.scene.terrain.terrain_generator.sub_terrains.clear()` 后只放一处 `CMoETrackMixTerrainCfg`：

  ```python
  CMoETrackMixTerrainCfg(
      proportion=1.0,
      x_unit=0.02, z_unit=0.002, height_scale=1.1, gap_shrink_units=10.0,
      corridor_width=0.80, pit_depth=0.50, pattern_start_x=0.30,
      pattern_spacing_scale=MIX_TEST_PATTERN_SPACING_SCALE,   # = 1.0（第三批：训练同值）
      spawn_x=0.75,
  )
  ```

  除 `proportion` 一处**有意偏离**（本场景只有这一类，比例无意义）外**逐字沿用训练实例化时的值**
  （训练侧是 `CMoETrackMixTerrainCfg(proportion=0.10)`，其余字段全取类默认值）——
  `tests/test_cmoe_mix_test_scene.py` 会把这些**字面量**与 `cmoe_terrains.py` 的类默认值**逐一比对**，
  改默认值而不同步这里就会红。规格原文写的是 `prop_start_x=0.30`，本仓真实字段名是 `pattern_start_x`。
* **网格**：`num_cols = MIX_TEST_LANES = 20`、`num_rows = MIX_TEST_LEVELS = 1`、`scene.num_envs = 20`
  ⇒ **20 条并列的 mix 道（沿世界 Y 铺 80 m）× 唯一一行难度（沿世界 X 只有 8 m）**，默认一道一个环境。
* **难度精确 0.70**：`terrain_generator.difficulty_range = (MIX_TEST_DIFFICULTY, MIX_TEST_DIFFICULTY)`
  ＋ `max_init_terrain_level = 0`，且**不覆盖** `scene.terrain.class_type`（默认 `TerrainImporter`）
  ⇒ **不依赖 CLI、也不需要自定义子类**（详见 §5）。
* **障碍间距 ×1.0**（`MIX_TEST_PATTERN_SPACING_SCALE`）：与训练默认值相同 ⇒ 评测几何**逐位一致**；
  为什么不用 2.0 见 §6/§7。
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

* `:247-261`（课程模式）逐列逐行生成瓦片，
  `difficulty = lower + (upper − lower) · (sub_row + U(0,1)) / num_rows`；
* `:379-383` 每块瓦片网格先按 `(−size[0]/2, −size[1]/2)` 居中（**出生点因此落在瓦片中心系里**，
  `origin = (spawn_x − size[0]/2, 0, 0)`）；
* `:330` 里 `terrain_origins[row, col]` 的平移量是 `((row+0.5)·size[0], (col+0.5)·size[1])`
  ⇒ **行（＝难度）沿世界 +X，列（＝地形类型/道）沿世界 +Y**；
* `:176-182` 整片地形再按 `(−size[0]·num_rows/2, −size[1]·num_cols/2)` 居中。

本场景 `size = (8.0, 4.0)`（父类 `Imgo2CMoERoughEnvCfg` 设定）、`num_rows = 1`、`num_cols = 20`：

| 轴 | 范围 | 单块 | 本例 |
|---|---|---|---|
| 世界 X（难度行） | **`[-4, +4]`**（1 行 × 8 m） | 每行 8 m | **出生点 X = `spawn_x − 4 = −3.25 m`** |
| 世界 Y（mix 道） | **`[-40, +40]`**（20 道 × 4 m） | 每道 4 m | 第 i 道占 `[4i − 40, 4(i+1) − 40]`；道中心 `Y = 4i − 38`；道内 mix 走廊（宽 0.80 m）占 **`[4i − 38.4, 4i − 37.6]`**（第 0 道走廊 `[−38.4, −37.6]`，第 19 道 `[37.6, 38.4]`） |

排布图（俯视，X 向右＝前进方向，Y 向上＝道）：

```
           道0      道1      道2    ...            道19     ← num_cols=20（沿 Y，每道 4 m）
  X=+4  ┌────────┬────────┬────────┬─────┬────────┐
  唯一行 │  mix   │  mix   │  mix   │ ... │  mix   │        ← num_rows=1（沿 X 只有 8 m）
  X=-4  └────────┴────────┴────────┴─────┴────────┘
        世界 Y ∈ [−40, +40]；每道一条 0.80 m 宽的 mix 走廊，走廊外是 −0.50 m 坑底
```

**2026-10-04（第二批）的"20 行 × 160 m"是历史**：那版世界里 X ∈ [−80, +80]、全部环境钉在第 14 行
（占 X ∈ [+32, +40]），另外 19 行（152 m）永远用不到。第三批把它们删掉了。

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
  同一行、出生点重叠）。第三批**删掉了**旧子类的 `[WARN]` ⇒ 现在这条约束只由文档/帮助文本与测试
  （`test_num_envs_is_twenty`）保证，**请自觉不要传 > 20**。
* `num_envs < 20` 时（例如 10）道号是 `[0, 2, 4, …, 18]`：**仍然互不重叠**，只是道与道之间跳着用
  （不是"只占前 10 道"）。这是 Isaac Lab 的等比铺法，不是 bug。
* 未验证项：以上都是**源码级**结论（本容器无 Isaac 运行环境），没有在真仿真里 `print` 过
  `terrain_types`。训练机回放时可用 `--num_envs=20` 然后看 20 个出生点是否分居 20 条道来验收。

## 5. 难度精确固定 = 0.70：字段核实、语义、以及"抖动是否被消除"

用户要求"固定到 14 难度"（= 旧 20 行网格的第 14 行，名义 `14/20 = 0.70`）。第三批把它改成
**单行 + 精确 0.70**。

### 5.1 `difficulty_range` 的核实结论（字段名 / 语义 / 是否消除抖动）

| 问题 | 核实结果（依据 = 已安装的 Isaac Lab 源码） |
|---|---|
| 字段名 | **`TerrainGeneratorCfg.difficulty_range: tuple[float, float] = (0.0, 1.0)`** —— `isaaclab/terrains/terrain_generator_cfg.py:113`。字段名正确，无需自定义生成器子类 |
| 语义（课程分支） | `terrain_generator.py:230-261`（`_generate_curriculum_terrains`，由 `TerrainGeneratorCfg.curriculum=True` 走这条）逐行算：`lower, upper = self.cfg.difficulty_range`；`difficulty = (sub_row + self.np_rng.uniform()) / self.cfg.num_rows`；`difficulty = lower + (upper − lower) * difficulty` |
| 语义（随机分支） | `terrain_generator.py:209-228`（`curriculum=False`）：`difficulty = self.np_rng.uniform(*self.cfg.difficulty_range)` ⇒ `U(0.70, 0.70)` 也恒为 0.70 ⇒ **两条分支都被钉住** |
| 本仓走哪条 | 仓库在 `VelocityEnvCfg.__post_init__`（`velocity_env_cfg.py:729-736`）里：只要 `curriculum.terrain_levels` 存在就把 `terrain_generator.curriculum = True`。CMoE 训练/play/mix-test 都设了 `terrain_levels` ⇒ **走课程分支** |
| **是否消除行内抖动** | **是，且是双重消除**：`num_rows = 1`、`sub_row = 0` ⇒ 括号里 `(0 + U(0,1))/1 = U(0,1)`；但乘数 `upper − lower = 0.70 − 0.70 = 0` ⇒ 第二项**恒等于 0** ⇒ `difficulty ≡ 0.70`，**逐块精确**（测试断言 `η ∈ {0, 1e-9, 0.25, 0.5, 0.75, 1−1e-12}` 全部得 0.70，`places=15`） |
| 旧方案对比 | 第二批"钉第 14 行 + 默认 `(0,1)`"的**名义**难度是 0.70，**实际** `d = (14 + U(0,1))/20 ∈ [0.70, 0.75)`（测试里保留这条对照）|

**离线可核的两条路径**（都在 `tests/test_cmoe_mix_test_scene.py::TestMixTestDifficultyPinning`）：

1. **"生成器会传进去的 difficulty"**：用 cfg 里的 `num_rows = 1` 与 `difficulty_range = (0.70, 0.70)`
   逐字复现 Isaac Lab 的课程公式（把 `U(0,1)` 抽成变量 `η` 扫描）⇒ 精确 0.70；
2. **真跑 `track_mix_terrain`**：把该 difficulty 喂给 `cmoe_terrains.track_mix_terrain`（桩 `isaaclab`
   ＋**真 `trimesh`**），核对生成的几何就是 `d = 0.70` 那一套：高栏顶面 **0.2618 m**
   （`= 170 × 0.002 × 1.1 × 0.70`）、第一处原始深坑 **`[1.50, 1.68]`**
   （`gap_shrink = round(10 × 0.3) = 3`），且与 `d = 0.0 / 1.0` 明显不同。

### 5.2 为什么可以不要子类（`num_rows = 1` 时课程天然冻结）

* **初始等级**（`terrain_importer.py:334-341`）：
  `max_init_level = min(max_init_terrain_level, num_rows − 1) = min(0, 0) = 0`
  ⇒ `terrain_levels = randint(0, max_init_level + 1) = randint(0, 1)` ⇒ **恒 0**；
* **升降级**（`terrain_importer.py:308-323`）：`terrain_levels[env_ids] += move_up − move_down` 之后
  `torch.where(level >= max_terrain_level(= num_rows = 1), randint_like(level, 1), clip(level, 0))`
  ⇒ `randint_like(..., 1)` 的取值域是 `[0, 1)` ⇒ **只会得到 0**；
* 因此唯一的下标 `terrain_origins[0, :]` **恒合法、不会越界**，出生点一局内也不变。

结论：**选方案①（删掉子类、`class_type` 恢复默认）**。旧子类的第二个作用（`num_envs > num_cols`
的 `[WARN]`）随之消失，代价已在 §4.1 写明。**注意**：如果以后有人把 `num_rows` 调回 > 1，就必须
重新评估这件事（那时"钉等级"仍有意义）。

### 5.3 `--terrain_level` 现在对 mix-test 是什么含义

`play.py` 的 argparse 仍是 `type=int` + **无 `choices`**，代码直接 `terrain.terrain_levels[:] = level`
（**不夹取**，原帮助文本里的 "0–9" 是过期文档）。但本任务 `num_rows = 1` ⇒ **只有 `0` 合法**，
传 `--terrain_level=1` 会在 `origins[terrain_levels, terrain_types]` 处 `IndexError`。
**本任务的难度不用 CLI 控制**（已由 `difficulty_range` 固定），帮助文本已相应改写（第三批）。

## 6. 障碍间距乘子 `pattern_spacing_scale`：字段保留、本场景取 1.0

`track_mix_terrain` 把图案逐段放在 `x0 = pattern_start_x + start_units · x_unit`；可选的乘子把
"**图案之间的 X 推进量**"放大，而**图案自身宽度不动**：

```
shift(start) = (scale − 1) · start · x_unit
x0 = pattern_start_x + start · x_unit + shift(start)
x1 = min(pattern_start_x + end · x_unit + shift(start), size[0])   # 宽度 = (end − start)·x_unit，与 scale 无关
```

* **默认 `scale = 1.0` ⇒ `shift` 精确等于 `+0.0`** ⇒ 生成结果与加字段之前**逐位相同**
  （用真几何对 `d ∈ {0, 0.35, 0.70, 0.75, 1.0} × size ∈ {(8,4), (8,8)}` 共 10 组逐块比较过
  `mesh.bounds`，测试 `TestMixTerrainGeometry` 固化了这条）；
* **第三批把评测场景也设回 1.0**（用户："各个难度间距先不要调整"）⇒ **评测与训练的 mix 几何逐位一致**；
* 字段本体、默认值 1.0 与 `track_mix_terrain` 的**总长溢出保护**都保留（"中性基础设施"）：
  图案末端 = `pattern_start_x + 160 · x_unit · scale`，超过 `size[0]` 时**直接 raise `ValueError`**
  （消息给出实际总长、瓦片长与该瓦片上的乘子上限），不静默截断（对照问题表 CMOE-13）。
  `size[0] = 8 m` 时乘子上限 = `(8 − 0.30)/(160 × 0.02) = **2.40625**`；
* 训练侧**不传**该字段（`CMoETrackMixTerrainCfg(proportion=0.10)`）⇒ 走默认 1.0。

**本轮范围说明**：用户先追加过"把拉开的空档补成 `height=0` 可走面"（方案①），随后**撤回**
（"各个难度间距先不要调整"）⇒ 本轮**没有**实现任何补空档逻辑，`track_mix_terrain` 的间距语义
一字未动；§7 的实测数字就是"按字面实现"的现状。

### 6.1 为什么不用 2.0：被拉开的空间是**深坑**，而出生点正好紧贴它

`mix` 瓦片的**整宽底面本身就在 `-pit_depth = -0.50 m`**（`_platform(0, size[0], …, -pit_depth)`），
而走道（`_corridor`，宽 0.80 m）**只铺在 `segments` 那几段上**。因此把图案拉开后，**段与段之间新空出来的
X 区间由原来的坑底填充**，表现为**更长的整宽深坑**，不是"多一段平地"。真几何实测（`d = 0.70`）：

| scale | 可走面总长 | 坑总长 | 图案末端 | 尾廊 |
|---|---|---|---|---|
| **1.0（本场景选用）** | **7.64 m** | 0.36 m | 3.50 m | 3.50 → 8.00 m |
| 1.5 | 6.04 m | 1.96 m | 5.10 m | 5.10 → 8.00 m |
| 2.0（第二批用过） | 4.44 m | 3.56 m | 6.70 m | 6.70 → 8.00 m |

**这条与用户的说法有出入，按要求如实报告**：用户把它描述为"模拟障碍之间更从容"，但按**字面**实现
（只乘 X 推进量、不改几何）得到的是**更长/更多、仍深 0.50 m 的整宽坑** ⇒ 地形**更难**。
用户已决定**先不动间距**（第三批撤回），因此**三条可选改法**（① 把段与段之间空出来的区间补成
`height=0` 的走廊；② 只对**本来就相接**的段插平地、保留原有两处深坑的难度律；③ 把乘子降到 1.5）
**仍留在待决清单**，本轮一律未做。

## 7. 出生点真几何：`spawn_x = 0.75` 到底落在哪（用户实测反馈："出生点在一个很大的 gap 上"）

用**桩 `isaaclab` ＋ 真 `trimesh`** 真跑 `track_mix_terrain`，把返回的网格摊成"沿 X 的可走区间/坑区间"，
再取 `spawn_x = 0.75` 的读数（`d = 0.70`、`pattern_start_x = 0.30`、`size = (8,4)`）：

| scale | 出生点落在 | 该平台顶面 | **出生点前方还有多少实心地面** | 第一处坑 | 后面第一处**原始**坑 |
|---|---|---|---|---|---|
| **1.0（本场景）** | `[0.30, 0.90]`（0.60 m 起步段） | **0.000 m**（可走） | **0.75 m** | — | `[1.50, 1.68]`（宽 0.18 m） |
| 1.5 | `[0.30, 0.90]` | 0.000 m | **0.15 m** | `[0.90, 1.20]`（宽 0.30 m） | `[1.68, 1.74]` 等 |
| 2.0（第二批） | `[0.30, 0.90]` | 0.000 m | **0.15 m** | `[0.90, 1.50]`（宽 **0.60 m**） | `[2.46, 3.06]`（宽 0.60 m） |

**结论（如实）**：`scale = 2.0` 时出生点**仍然踩在实心面上**（不是"悬在坑里"），但它只是那块 0.60 m
平台上的一点，**离平台前缘只有 0.15 m**；机器人机身前缘/前足在基座前方约 0.2 m，**出生瞬间前足已经
探进 0.60 m 的整宽深坑** ⇒ 观感与实测都像"出生在一个很大的 gap 上"（本机另有一份函数级读数：
`walkable_total` 4.44 m / `pit_total` 3.56 m，即"可走面只剩一半"）。
`scale = 1.0` 时第一处坑在 **1.50 m**（前方 0.75 m 实心），前足落在平台上 ⇒ **这就是评测场景固定用
1.0 的直接原因**。（用户已决定不改几何 ⇒ 本轮**没有**加长走廊、也**没有**前移出生点。）

对应的回归锁定：`TestMixTestSpawnGeometry`（`scale ∈ {1.0, 1.5, 2.0}` 的出生点位置、前方实心长度、
坑宽），以及 `TestMixTerrainGeometry` 里对间距语义的既有断言。

## 8. 训练侧的同类隐患（**待决**，本轮不改训练；问题表 CMOE-17）

训练用的 `Imgo2-basemove-rough-cmoe` 继承同一套 mix 参数（`spawn_x = 0.75`、`pattern_start_x = 0.30`、
`scale = 1.0`），但它**开着 reset 平移**：`Imgo2CMoERoughEnvCfg` 把
`events.randomize_reset_base.params["pose_range"]` 设为 **`x ∈ (−0.5, 0.5)`、`y ∈ (−0.5, 0.5)`、`z = 0`**
（`CMoE_env_cfg.py:446-453`）。量化（真几何 + 源码，`TestTrainingSpawnHazard` 钉住）：

* **X 方向的阈值**：mix 的前 5 段（索引 `0→30→36→42→48→60`）**首尾相接** ⇒ 第一处深坑的左沿恒为
  `pattern_start_x + 60 · x_unit = **1.50 m**`（与难度 `d` 无关；已在 `d = 0, 0.70, 1.0` 上逐个核实）。
  `pose_range x = ±0.5` ⇒ 出生 x ∈ **[0.25, 1.25]**：
  * **不会出生在坑上** —— 要越界得 `spawn_x > 1.50 − 0.5 = 1.00`，现行 0.75 还差 0.25 m；
  * **但 x > 0.90（占 35%）会落在抬高的台阶段上**（顶面 0.0462 / 0.0924 / 0.1386 m @ d=0.70，
    d=1.0 时最高 **0.198 m**），而 `reset_root_state_uniform` 的高度只用
    `default_root_state.z + env_origins.z(=0) + pose_range.z(=0)`（`isaaclab/envs/mdp/events.py:1032`，
    **不看局部地形高度**）⇒ **足端会嵌进台阶**（最多 ~0.14~0.20 m）；最坏情况 `x = 1.25` 距坑沿只剩
    **0.25 m**（≈机身半长）。
* **Y 方向的隐患更大**：mix 是**窄走廊**，`corridor_width = 0.80 m` ⇒ 半宽只有 **0.40 m**，而
  `pose_range y = ±0.5` 超出半宽 ⇒ **|Δy| > 0.40 的 20% 重置会让机器人基座横向落在 −0.50 m 的坑底
  上方**（出生即下落 0.5 m / 一侧足悬空）。注意这条对**所有**用窄走廊的地形列（`mix`、
  `narrow_stairs`）都成立，不只 mix。

**建议改法（本轮不改，待训决）**：
① 最小改动、覆盖全部窄走廊：把训练 `pose_range.y` 从 `±0.5` 收到 **`±0.30`**（或按地形分列做掩码）；
② 辅助：把 `pose_range.x` 收到 `±0.3`（同时也消掉"落在台阶段上"的 35%）；③ 若想保留大抖动，可把
`spawn_x` 前移到 `≤1.00` 之前更靠后的位置并**加长起步走廊**（`pattern_start_x` ↑）——但那改的是
**地形几何**，必须重新验证训练侧几何不变性，因此**不属于本轮**。

## 9. 命令项设计（`mdp/mix_test_command.py`）

1. **继承 `UniformVelocityCommand`**：复用它的 `vel_command_b` / `heading_target` /
   `is_heading_env` / `is_standing_env` / `metrics` 与 `_update_metrics`，只重写
   `_resample_command` / `_update_command`。
2. **`vx_cmd` 恒定**：`super()._resample_command()` 采样**一次**（`lin_vel_x=(1.0,1.0)`），
   `_update_command` **不碰第 0 列**（测试断言 `vel_command_b[:,0]` 恒为 1.0）。
3. **横向 PD**（每步重算，PD 是时变的）：写 `vel_command_b[:,1]`。
4. **航向 PD**：写 `vel_command_b[:,2]`。
5. **写回 `vel_command_b` 与 `vel_command_w`**：奖励与观测读的仍是 `command`（＝`vel_command_b`）；
   `vel_command_w` 是**世界系镜像**，见 §11。
6. **纯函数化**：PD 数学在 `mdp/mix_test_pd.py`（只依赖 torch），可离线单测。

### 9.1 坐标系约定（**必读**）

| 变量 | 取法 | 为什么 |
|---|---|---|
| `y`（`y_local`） | **世界系**横向位置 − 本环境出生原点：`root_pos_w[:,1] − env.scene.env_origins[:,1]` | `env_origins` 是该 env 的**道中心** ⇒ 这个差值就是"离赛道中心线的横向距离"（出生在走廊中心 y=0） |
| `vy`（`vy_local`） | **世界系**横向线速度：`root_lin_vel_w[:,1]` | 与 `y` 同坐标系，也与评测奖励 `track_world_vel_xy_exp`（**世界系**）一致 |
| `yaw` | `robot.data.heading_w` | Isaac Lab 由 `root_quat_w` 把机体系 +x 投到世界系后 `atan2(y, x)` ⇒ 偏航角，已绕回 `(−π, π]` |
| `wz`（`wz_local`） | **机体系**偏航角速度：`root_ang_vel_b[:,2]` | 命令 `vel_command_b[:,2]` 写的就是机体系角速度；既有 `error_vel_yaw` 度量也用它 |

由于课程被"单行"固定，`env_origins` **一局内不再变化** ⇒ 横向误差的参考点稳定。

### 9.2 为什么不用内置 heading 控制器

`UniformVelocityCommand._update_command` 在 `heading_command=True` 时用**内置 P 控制器**
（`vel_command_b[:,2] = clip(heading_control_stiffness · heading_error, ang_vel_z[0], ang_vel_z[1])`）
写 `wz`。本任务 `ranges.ang_vel_z = (0,0)` ⇒ 内置控制器会把 `wz` **恒夹成 0**，且它只有 P 项、
没有 `wz` 阻尼。因此本命令项**不使用**内置控制器，而是每步用自己的 PD 写 `vel_command_b[:,2]`。
**cfg 里 `heading_command` 仍保持 `True`**（规格要求，也是 `ranges.heading` 的合法性前提）。

## 10. PD 公式、默认增益与调法

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
* **改难度**：改常量 `MIX_TEST_DIFFICULTY`（它同时写进 `difficulty_range` 的上下界）——**不是** CLI；
  `--terrain_level` 对本任务只有 0 合法。
* **改道数 / 环境数**：`MIX_TEST_LANES`（也是 `scene.num_envs`）；`--num_envs` 必须 ≤ 它。
* **改间距**：`MIX_TEST_PATTERN_SPACING_SCALE`（现值 1.0；上限 2.40625，超过会 raise）——
  改之前请先读 §6.1/§7：乘子 > 1 会把段间空档变成更长的**深坑**，且出生点前方实心地面会掉到 0.15 m。
* **改 PD 增益**：目前写在 `MixTestVelocityCommandCfg` 的字段默认值里；要按 run 调，在
  `Imgo2CMoEMixTestEnvCfg.__post_init__` 构造 cfg 时传参即可（例如 `kp_y=1.5, vy_max=0.8`）。
* **调参直觉**：`kp_y` 大了会来回摆（`kd_y` 要跟着加）；`vy_max` 是"允许用多大横移去纠偏"的上限；
  参考量级：站立高度 0.30 m、走廊半宽 0.40 m，而 `vy_max=0.6 m/s` 已接近行走速度（1.0 m/s）的 60%，
  因此**横向纠偏不能指望它救回大偏差**——本场景的前提是"横移本来就不该发生"。
* **关掉转向、只测横向**：把 `kp_h`/`kd_h` 置 0 即可。

## 11. `vel_command_w`（**与规格的一处冲突，已按"不改设计、只报告"处理**）

规格要求"把 `vy_cmd`/`wz_cmd` 写进该 command 的 `vel_command_b`/`vel_command_w`，
**与既有 `base_velocity` 的写法一致**"。但事实是：**基类 `UniformVelocityCommand` 与本仓都没有
`vel_command_w`**（全仓 `grep -rn vel_command_w` 在本次改动前为空）。处理方式：

* `vel_command_b` 是真正生效的缓冲（奖励、观测、度量都读它）；
* `vel_command_w` 由本命令项**新增**为纯附加缓冲，每步写入"机体系指令按当前 yaw 旋转到世界系"的镜像；
* 它**不进入观测、不进入动作空间、不参与奖励**，因此**不影响 checkpoint 兼容性**，仅供回放/审计读。

## 12. 运行示例

回放/评测（须替换 checkpoint 绝对路径；**难度默认已精确固定 d = 0.70，不要传 `--terrain_level`**）：

```bash
cd imgo2_rl
python scripts/rl_lab/cmoe/play.py \
    --task=Imgo2-basemove-rough-cmoe-mix-test \
    --num_envs=20 --headless \
    --checkpoint="/absolute/path/to/model.pt"
```

* `--num_envs=20`（＝`MIX_TEST_LANES`）时**一道一个环境**（环境 i → 第 i 道，沿 Y 并排，道中心
  `Y = 4i − 38`）；`--num_envs > 20` 会让多个环境落在同一条道的同一出生点（**不再有子类告警**）。
* 出生点在世界 **X = −3.25 m**（= `spawn_x − 4`）、道中心 Y；世界 **X ∈ [−4, +4]、Y ∈ [−40, +40]**。
* 也可用 `--force_expert K` 等既有 CMoE play 开关（本任务不引入新开关）。

## 13. 地形列校验：按任务区分

`scripts/tools/check_terrain_columns.py` 的原判据是"**所有被掩码引用的地形名都必须 ≥1 列**"。
本测试场景**只有 `mix`** ⇒ 掩码引用的 `boxes` / `gap` / `flat` 在该场景**不存在**，引用它们的项
（`joint_mirror`、`feet_air_time`、`feet_height_body`、`feet_air_time_variance`、`feet_gait`、
`base_height_flat_l2`）在本场景**恒为 0**——这是**预期**，不是配置错误。

处理方式（用户要求"不削弱原判据"）：

* 新增 `--task`，默认 `cmoe-rough` **走原判据、一字未放宽**；
* `--task mix-test` 走**单独分支**：仍强制 `sub_terrains.clear()` + 只有 mix + **mix 占满 20 道** +
  `num_cols=20` + **`num_rows=1`** + **`difficulty_range=(0.70, 0.70)`**，并把"掩码引用但本场景不存在"
  的名字**逐条打印为预期**；
* 解析从"整文件 `ast.walk`"改成**类作用域**（只走指定类的 `__post_init__`）；
* `num_cols`/`num_rows`/难度写成**模块级可读常量**，工具的解析支持"顶层 `NAME = <字面量>` 常量求值"；
* 测试对 test 任务**单独断言**，并加**负向对照**（把"只有 mix"丢给原判据必须仍报错、存在但 0 列也必须
  报错、**期望道数/行数/难度被改成别的值时必须报 ❌**）证明判据没被放宽。

实测输出（`python3 scripts/tools/check_terrain_columns.py --task mix-test`，退出码 0）：

```
=== [Imgo2-basemove-rough-cmoe-mix-test] 受控测试场景（只 mix；cfg 类 Imgo2CMoEMixTestEnvCfg）===

=== Imgo2-basemove-rough-cmoe-mix-test｜只 mix（20 道并列）（num_cols=20，共 20 列）===
  mix                     20 列
  ✅ 唯一地形 mix 有 20 列（num_cols=20、num_rows=1）⇒ 网格 **20 道 × 1 难度行**（世界 8 m(X) × 80 m(Y)），
     难度由 difficulty_range=(0.7, 0.7) 精确固定 ⇒ `--num_envs ≤ 20` 时环境 i → 第 i 道
  ⚠️ 本场景**只有 mix**；掩码引用但本场景**不存在**的地形名（原任务判据**不放宽**，这些项在本场景恒为 0 属**预期**）：
    - joint_mirror.free_terrain_names → boxes：本场景无该地形 ⇒ 该项本场景恒为 0（预期，不是配置错误）
    ...（共 11 条，逐条列出）
  ✅ `forward_only_terrain_names`（继承父类） 覆盖 mix；另含 10 个本场景不存在的地形名（...）

结论：test 任务只有 mix 一种地形；掩码引用的其它地形名在本场景恒为 0，属**预期** ✅
```

## 14. 未验证项、与设计冲突之处、待决项

**未验证（本容器无 GPU / 无 Isaac 运行环境：`NVIDIA_VISIBLE_DEVICES=void`、`import isaaclab` 缺 `omni.log`）**：

1. **未构造 Isaac 环境**：`difficulty_range` 在真实 `TerrainGenerator` 上的产出、`InteractiveScene`
   的构造路径、`MixTestVelocityCommandCfg` 在真实 `CommandManager` 下的解析、`debug_vis=True` 在
   20 环境下的 marker 创建都未实测。**难度公式是源码级复核 + 离线复现，不是运行期实测**。
2. **未回放、未导出 `policy.pt`**：checkpoint 兼容性是**静态推理**（动作/观测契约未改）。
3. **PD 增益未标定**：`1.0/0.3/0.6`（横向）与 `1.5/0.3/1.0`（航向）**未在仿真里标定**。
4. **`terrain_types` 的"一道一个环境"未在真仿真里打印确认**（结论来自 Isaac Lab 源码公式，见 §4.1）。
5. **1 行 × 20 道的真实地形网格未生成过**：几何结论来自**用桩 `isaaclab` + 真 `trimesh` 实跑**
   `track_mix_terrain`（函数级），不含 Isaac 的瓦片拼接、缓存与碰撞体。
6. **奖励掩码在本场景恒为 0 的实际影响未测**：6 项掩码项恒 0 ⇒ 评测时 `Episode_Reward/*` 里这些键的
   读数不能与训练 run 直接对比（只是读数问题，不是 bug）。
7. **一处已知的读数副作用（有意保留）**：基类 `_update_metrics` 用
   `max_command_step = resampling_time_range[1] / step_dt` 做归一化；本任务
   `resampling_time_range = (1e9, 1e9)` ⇒ `error_vel_xy` / `error_vel_yaw` 两个**度量**读数≈0
   （只进日志）。要可读的跟踪误差请用奖励分项 `Episode_Reward/track_world_vel_xy_exp`。

**与设计/规格的冲突（未擅自改设计，按要求报告）**：

1. **`vel_command_w` 不存在**（§11）：已作为**纯附加**缓冲实现，不进观测/动作。
2. **难度来源与用户原话的差异**：用户说"固定到 14 难度"，第三批落地为"**单行 + 精确 `d = 0.70`**"
   （等价于原第 14 行的**名义**值 14/20）。副作用：这唯一一行的实际难度由 `[0.70, 0.75)` 收紧为
   **精确 0.70**，且**不能再扫难度**（`--terrain_level` 已无意义）。这是用户"不要那么多行"要求的
   直接后果，已写进 cfg 注释与本文。
3. **删子类丢掉了 `num_envs > num_cols` 的 `[WARN]`**（§4.1/§5.2）：约束仍在文档、帮助文本与测试里，
   但运行期不再拦。若仍想要告警，需要重新引入一个只做检查的子类（本轮判定为不必要）。
4. **间距乘子 > 1 会把"缝"变成更长的深坑**（§6.1）⇒ 地形**更难**，与"更从容"的意图相反；
   三条可选改法仍**待用户决定**（用户第三批已明确"先不要调整"）。
5. **出生点问题只被"绕开"、没有被"修好"**（§7/§8）：评测场景靠 `scale = 1.0` 回到了安全布局，
   但训练侧 `spawn_x = 0.75` + `pose_range ±0.5` 的同类隐患**仍在**（问题表 **CMOE-17**，待决）。
6. **`track_gap_terrain` 的溢出保护仍未做**（问题表 CMOE-13）：本轮只保留 `track_mix_terrain` 的保护，
   没有顺手扩到 `track_gap_terrain`（超出本次范围）。
7. **`check_reward_overrides.py` 未加 `cmoe-mix-test` 链**：本场景**完全不改奖励**（生效项仍是 `cmoe`
   的 28 项），加一条与 `cmoe` 逐项相同的链没有信息量，故未加。

## 15. 验证与实测输出（本机离线）

```bash
cd imgo2_rl
/opt/conda/envs/isaaclab/bin/python -m unittest discover -s tests     # 651 通过 / 10 跳过
python3 scripts/tools/check_reward_overrides.py cmoe-gaitfree          # 24 项（cmoe = 28 项，与改前一致）
python3 scripts/tools/check_terrain_columns.py                         # 末行"全部掩码引用的地形名都有 ≥1 列 ✅"
python3 scripts/tools/check_terrain_columns.py --task mix-test         # "mix 20 列"、末行"…属预期 ✅"（退出码 0）
git diff --check                                                       # 干净
```

* **基线**：第三批改动前 `unittest discover -s tests` = **632 通过 / 10 跳过**；
  第三批后 = **651 / 10**（`test_cmoe_mix_test_scene.py` 56 → **72**、`TestMixTestTerrainColumns`
  17 → **20**）。
* **"难度确实 = 0.70"的两条路径**：见 §5.1（公式复现 `places=15`；真跑 `track_mix_terrain` 核对
  栏高 0.2618 m 与坑区间 `[1.50, 1.68]`）；另一条源码级断言在已安装的 Isaac Lab 里逐字核对
  `difficulty_range` / 两条分支公式 / 本仓 `curriculum = True` 的开启条件（缺 Isaac Lab 时该组 skip）。
* **出生点真几何实测**：见 §7 表（`d = 0.70`、`scale ∈ {1.0, 1.5, 2.0}`；读数由
  `TestMixTestSpawnGeometry` 锁定）。
* **训练侧隐患量化实测**：见 §8（`TestTrainingSpawnHazard` 锁定：第一处坑恒在 1.50 m、x 越界阈值
  `spawn_x > 1.00`、35% 落在台阶段、y 方向 20% 出走廊）。
* **溢出保护实测（沿用第二批）**：`scale=2.5` 抛
  `ValueError: track_mix_terrain: 图案总长 8.3000 m 超出瓦片长 size[0]=8.0000 m …本瓦片上
  pattern_spacing_scale 最大可取 2.4062…`；`scale=2.40625` 不抛。
* **"默认 1.0 无差异"（沿用第二批的验证方式）**：把 `b2caad1` 的 `cmoe_terrains.py` 与改后版本分别
  exec（桩 `isaaclab` + 真 `trimesh`），对 `d ∈ {0, 0.35, 0.70, 0.75, 1.0} × size ∈ {(8,4), (8,8)}`
  逐块比较 `mesh.bounds` 与 `origin`：**10/10 逐位相同**（`cmoe_terrains.py` 本轮**未改**）。
