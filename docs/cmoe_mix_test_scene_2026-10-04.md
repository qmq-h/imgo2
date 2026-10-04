# `mix` 受控测试场景（`Imgo2-basemove-rough-cmoe-mix-test`）— 2026-10-04

本文件记录**同一天的四批**改动（都在 `imgo2_CMoE` 分支上，HEAD `6d298b7` 之后未提交）：

* **第一批**：新增评测任务类 `Imgo2CMoEMixTestEnvCfg`（继承 play）——只有 `mix` 地形、速度只给前进
  且默认恒定 1.0 m/s、横向与航向由**指令层 PD 外环**控制、heading 恒 0；
* **第二批**（用户："地形不要按照列排，放在行里面" ＋ "都固定到14难度" ＋ "mix 中每个地形间隔大一点
  ×1.5~2.0"）：排布改成 **20 条并排的 mix 道**（沿世界 Y）× **20 档难度**（沿世界 X），全部环境
  固定在第 14 行（名义 `d = 0.70`），并把 mix 图案的**障碍间距乘子**设为 **2.0**。
* **第三批**（用户："分居了，但是我不需要还保持那么多行，我需要他们并列"；随后追加
  "各个难度间距先不要调整，回复吧"）：

  ① **只留 1 行难度 × 20 条并列的道** —— `num_rows = 1`、`num_cols = 20`；
  ② 世界由 **160 m(X) × 80 m(Y)** 缩到 **8 m(X) × 80 m(Y)**（19 行用不到的难度行被删掉）；
  ③ 难度不再靠"把课程等级钉在第 14 行"，改由 **`terrain_generator.difficulty_range = (0.70, 0.70)`
     精确固定**（旧方案的实际难度是 `d ∈ [0.70, 0.75)`，只是名义 0.70）；
  ④ **删掉钉等级的子类** `Imgo2CMoEMixTestTerrainImporter`，`class_type` 恢复为默认 `TerrainImporter`
     （`num_rows = 1` 时课程天然只有 0 级）；
  ⑤ 间距乘子 **2.0 → 1.0**（＝训练默认值）⇒ 评测几何与训练**逐位一致**；`pattern_spacing_scale`
     字段本体、默认值 1.0 与 `track_mix_terrain` 的**总长溢出保护**都保留（中性基础设施）。
* **第四批（本轮）**（用户："mix 还是太小了…课程长度太短：让 mix 占满整条道"）：

  ① `cmoe_terrains.CMoETrackMixTerrainCfg` 新增布尔字段 **`fill_stretched_gaps`（默认 `False`）**：
     为 `True` 时把**因拉开而多出来的空档**（段与段之间、最后一段末端 → 图案末端的"尾段"）铺成
     `height=0` 的可走面；**原图案本来就有的两处坑原样保留**（`d = 0.70` ⇒ 各 0.18 m），
     **障碍自身的宽度/高度/顺序一字不变**。默认 `False` ⇒ 训练与既有测试的几何**逐位不变**；
  ② 评测场景 `pattern_spacing_scale` 取 **反算的"刚好占满整条道"值 `2.25`** ＋ `fill_stretched_gaps
     = True` ⇒ 图案末端由 **3.50 m** 推到 **7.50 m**（占 8 m 道的 93.75 %，尾部留 0.50 m 平地）；
     乘子仍严格小于溢出上限 **2.40625**（保护**不放宽**，超限照旧直接 `ValueError`）；
  ③ 出生点前方的实心地面由 **0.75 m** 变成 **1.95 m**（第一处坑由 1.50 m 后移到 2.70 m）。

**这不是新策略任务**：动作空间、观测组与维度、奖励、终止**一字未改**，只是换了一个场景 + 一层指令
外环 ⇒ 既有 CMoE checkpoint（77 维地形 / 527 维 actor / 125 维 critic）**可直接加载**。

## 1. 目的与范围

用途：在受控条件下评估策略通过 `mix`（复合障碍：窄走廊 + 台阶上行 + 深坑 + 高台 + 高栏 + 深坑）
的能力，**把横向漂移与航向漂移从评测里剔除**，并且**难度固定**（不再受课程随机初始等级影响）。
第三批要求"**道并列、但不要那么多行**"，于是整个世界只剩**一行**难度、20 条道沿 Y 并排；
第四批要求"**让 mix 占满整条道**"（障碍尺寸不变、只把障碍之间拉开、把拉开的空档铺成可走面）。

## 2. 改动清单（第四批）

| 文件 | 改动 |
|---|---|
| `.../velocity/base_move/cmoe_terrains.py` | `CMoETrackMixTerrainCfg` 新增 `fill_stretched_gaps: bool = False`；`track_mix_terrain` 在 `fill_stretched_gaps and spacing > 1` 时把"段间/尾段"的拉开余量补成 `height=0` 的 `_corridor`（原有坑宽与障碍几何不变）；函数 docstring 补第四批说明；`pattern_spacing_scale` 注释里的场景取值 2.0 → 2.25 |
| `.../velocity/base_move/CMoE_env_cfg.py` | `MIX_TEST_PATTERN_SPACING_SCALE = 1.0 → **2.25**`（反算值，算式写在常量注释里）；新增 `MIX_TEST_TAIL_MARGIN = 0.50`、`MIX_TEST_PATTERN_END_X = 7.50`、`MIX_TEST_FILL_STRETCHED_GAPS = True`；`sub_terrains["mix"]` 加 `fill_stretched_gaps=MIX_TEST_FILL_STRETCHED_GAPS`；头注释、类 docstring、cfg 注释同步 |
| `imgo2_rl/scripts/tools/check_terrain_columns.py` | `scene_overrides` 额外记录每个 `sub_terrains[name] = <Cfg>(...)` 的**全部 kwargs**；新增 `terrain_size()`（真读 `terrain_generator.size`）；mix-test 分支新增第四批检查（乘子 = 反算值 2.25、`fill_stretched_gaps=True`、图案末端 ≥ 7.0 m 且 ≤ `size[0]`）并打印核算行 |
| `imgo2_rl/tests/test_cmoe_mix_test_scene.py` | 72 → **82 项**：新增 `TestMixTerrainGeometry` 的 6 条 fill 断言（默认中性、只加 height=0 补块、障碍几何与 scale 无关、图案区间可走面单调增而坑总长恒 0.36、图案末端 ≥ 7.0、溢出保护不放宽）、新增 `TestMixTestFilledSpawnGeometry`（4 条：反算前提、两 scale 出生点安全、不补空档的对照 0.15 m、真 trimesh 实测 X 区间表）；`TestMixTestEnvCfgSource` 的乘子/总长/参数一致性断言改第四批事实；第三批的"场景用 1.0"断言改成"`fill=False` 旧行为"回归对照 |
| `imgo2_rl/tests/test_check_terrain_columns.py` | `TestMixTestTerrainColumns` 20 → **26 项**：常量期望改 2.25/0.50/7.50/True、新增 mix kwargs 解析、`terrain_size()`、第四批核算行的正向 + 3 条负向对照（乘子改错 / 目标下界抬高 / `fill_stretched_gaps` 不为 True） |
| `README.md` | 任务表该行、CMOE-16/17 状态、§8.3 维护记录（第四批一行）同步 |

第一/二/三批的改动清单（新增 `mdp/mix_test_pd.py`、`mdp/mix_test_command.py`、`pattern_spacing_scale`
字段、`--task` 分支、难度钉死、删钉等级子类、测试从 35 → 56 → 72 项等）见本文件的 git 历史与
README §8.3 的对应三行。

## 3. 场景契约

由 `Imgo2CMoEMixTestEnvCfg.__post_init__`（继承 `Imgo2CMoERoughPlayEnvCfg`）设定：

* **地形**：`self.scene.terrain.terrain_generator.sub_terrains.clear()` 后只放一处 `CMoETrackMixTerrainCfg`：

  ```python
  CMoETrackMixTerrainCfg(
      proportion=1.0,
      x_unit=0.02, z_unit=0.002, height_scale=1.1, gap_shrink_units=10.0,
      corridor_width=0.80, pit_depth=0.50, pattern_start_x=0.30,
      pattern_spacing_scale=MIX_TEST_PATTERN_SPACING_SCALE,   # = 2.25（第四批：反算的"占满整条道"值）
      fill_stretched_gaps=MIX_TEST_FILL_STRETCHED_GAPS,       # = True（第四批：把拉开的空档铺成可走面）
      spawn_x=0.75,
  )
  ```

  除 `proportion`（本场景只有这一类，比例无意义）与第四批的 `pattern_spacing_scale` /
  `fill_stretched_gaps` 两处**有意偏离**外，其余字段**逐字沿用训练实例化时的值**
  （训练侧是 `CMoETrackMixTerrainCfg(proportion=0.10)`，其余字段全取类默认值）——
  `tests/test_cmoe_mix_test_scene.py` 会把这些**字面量**与 `cmoe_terrains.py` 的类默认值**逐一比对**，
  改默认值而不同步这里就会红。规格原文写的是 `prop_start_x=0.30`，本仓真实字段名是 `pattern_start_x`。
  **因此"障碍自身"的几何（每块宽/高/顺序）与训练仍逐字相同**，评测与训练的差异只在"障碍之间的间距"
  与"被拉开的空档是平地还是深坑"。
* **网格**：`num_cols = MIX_TEST_LANES = 20`、`num_rows = MIX_TEST_LEVELS = 1`、`scene.num_envs = 20`
  ⇒ **20 条并列的 mix 道（沿世界 Y 铺 80 m）× 唯一一行难度（沿世界 X 只有 8 m）**，默认一道一个环境。
* **难度精确 0.70**：`terrain_generator.difficulty_range = (MIX_TEST_DIFFICULTY, MIX_TEST_DIFFICULTY)`
  ＋ `max_init_terrain_level = 0`，且**不覆盖** `scene.terrain.class_type`（默认 `TerrainImporter`）
  ⇒ **不依赖 CLI、也不需要自定义子类**（详见 §5）。
* **课程占满整条道**（`MIX_TEST_PATTERN_SPACING_SCALE = 2.25` ＋ `fill_stretched_gaps = True`）：
  图案 X 总长 = `0.30 + 160×0.02×2.25 = 7.50 m`（占 8 m 道的 **93.75 %**，尾部留 0.50 m 平地）；
  反算式与铺平地规则见 §6.1/§6.2；为什么不再是 1.0、旧行为是什么见 §6.3。
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
2. **真跑 `track_mix_terrain`**：把该 difficulty 喂给 `cmoe_terrains.track_mix_terrain`
   （桩 `isaaclab` ＋**真 `trimesh`**，**用场景自己的乘子与补空档开关**），核对生成的几何就是
   `d = 0.70` 那一套：高栏顶面 **0.2618 m**（`= 170 × 0.002 × 1.1 × 0.70`）、两处原图案坑宽各
   **0.18 m**（`gap_shrink = round(10 × 0.3) = 3` ⇒ `(72 − 3 − 60) × 0.02`），占满值下第一处坑
   在 **`[2.70, 2.88]`**（`= 0.30 + 60×0.02 + (2.25−1)×48×0.02` 起）；且与 `d = 0.0 / 1.0` 明显不同。

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

## 6. 课程占满整条道：反算的间距乘子 ＋ `fill_stretched_gaps`

`track_mix_terrain` 把图案逐段放在 `x0 = pattern_start_x + start_units · x_unit`；可选的乘子把
"**图案之间的 X 推进量**"放大，而**图案自身宽度不动**：

```
shift(start) = (scale − 1) · start · x_unit
x0 = pattern_start_x + start · x_unit + shift(start)
x1 = min(pattern_start_x + end · x_unit + shift(start), size[0])   # 宽度 = (end − start)·x_unit，与 scale 无关
```

* **默认 `scale = 1.0` ⇒ `shift` 精确等于 `+0.0`** ⇒ 生成结果与加字段之前**逐位相同**
  （用真几何对 `d ∈ {0, 0.35, 0.70, 0.75, 1.0} × size ∈ {(8,4), (8,8)}` 共 10 组逐块比较过
  `mesh.bounds`，测试 `TestMixTerrainGeometry` 固化了这条；训练侧**两个字段都不传**）；
* 字段本体、默认值 1.0 与 `track_mix_terrain` 的**总长溢出保护**都保留：
  图案末端 = `pattern_start_x + 160 · x_unit · scale`，超过 `size[0]` 时**直接 raise `ValueError`**
  （消息给出实际总长、瓦片长与该瓦片上的乘子上限），不静默截断（对照问题表 CMOE-13）。
  `size[0] = 8 m` 时乘子上限 = `(8 − 0.30)/(160 × 0.02) = **2.40625**`。**第四批没有放宽这条保护**
  （`fill_stretched_gaps=True` 时超限仍 raise，测试 `test_overflow_guard_is_not_relaxed_by_fill_mode`）。

### 6.1 反算"刚好占满整条道"的乘子

用户第四批要求："让 mix 占满整条 8 m 道"且"障碍自身的尺寸/高度不变（只把障碍之间拉开）"。约束是
**图案末端 ≤ `size[0]` − 尾部余量**（尾部余量＝图案之后保留的可走平地段，本方案取 **0.50 m**）：

```
pattern_start_x + 160 · x_unit · scale ≤ size[0] − 尾部余量
⇒ scale ≤ (size[0] − pattern_start_x − 尾部余量) / (160 · x_unit)
        = (8.00 − 0.30 − 0.50) / (160 × 0.02)
        = 7.20 / 3.20
        = 2.25
```

取 **`scale = 2.25`** ⇒ 图案末端 = `0.30 + 160 × 0.02 × 2.25 = **7.50 m**`
（占 8 m 道的 **93.75 %**，≥ 用户给的 7.0 m / 85 % 目标），尾部平地 = `8.00 − 7.50 = 0.50 m`；
`2.25 < 2.40625` ⇒ **不需要**放宽溢出保护（也没有改图案数量或 `size[0]`）。
常量落点：`MIX_TEST_PATTERN_SPACING_SCALE = 2.25`、`MIX_TEST_TAIL_MARGIN = 0.50`、
`MIX_TEST_PATTERN_END_X = 7.50`（`CMoE_env_cfg.py`，注释里带着上面的算式）。

### 6.2 铺平地规则：`fill_stretched_gaps`

单靠乘子**不够**：`mix` 瓦片的**整宽底面本身就在 `-pit_depth = -0.50 m`**，而走道
（`_corridor`，宽 0.80 m）**只铺在 `segments` 那几段上** ⇒ 把图案拉开后，段与段之间新空出来的 X 区间
由原来的坑底填充，表现为**更长/更多的整宽深坑**（第二批就是这么"更难"的，见 §6.3 的对照表）。
第四批为此新增 `fill_stretched_gaps`：

* `False`（**默认**）＝ 上面的旧行为，**与改动前逐位相同**（训练侧不传该字段）；
* `True` ＝ 在 `scale > 1` 时，把**因拉开而多出来的空档**铺成 `height=0` 的可走面 `_corridor`：
  * **段与段之间**：空档 = `[上游块末端(拉伸后), 下游块起点(拉伸后)]`；其中前
    `(next_start − prev_end) · x_unit` 是**原图案本来就有的坑**（本图案两处：`60 → (72−k)`、
    `111 → (123−k)`，`k = round(10 − 10d)` ⇒ `d = 0.70` 时各 0.18 m），**原样保留且紧贴上游块末端**
    （⇒ 坑宽与"相对上游障碍的位置"都只由图案与难度决定，**与 scale 无关**）；余下的"拉开余量"铺平；
  * **尾段**：最后一段末端 → **图案末端（索引 160）**之间的空档也铺平（`d = 0.70, scale = 2.25` 时
    是 `[7.00, 7.50]` 这 0.50 m）；
  * **领先段**（`0 → pattern_start_x = 0.30`）与**原有尾廊**（图案末端 → `size[0]`）本来就在 0 高度，
    不需要额外处理；`scale = 1.0` 时没有任何可拉余量（补块列表为空）⇒ **开不开这个开关几何完全相同**；
  * ⇒ **障碍自身几何（每块宽度/顶面高度/顺序）与坑宽一字不变**，被拉开的只是"障碍之间的平地"。

### 6.3 真几何实测：`scale = 1.0` vs 占满值 `2.25`（`d = 0.70`、`fill_stretched_gaps=True`）

| scale | 图案末端 | 尾部平地 | **图案区间内**可走面 | 坑总长 | 补出的平地 | **整片**可走面 |
|---|---|---|---|---|---|---|
| 1.0 | 3.50 m | 4.50 m | 2.84 m | 0.36 m | 0.00 m | 7.64 m |
| 1.5 | 5.10 m | 2.90 m | 4.44 m | 0.36 m | 1.60 m | 7.64 m |
| 2.0 | 6.70 m | 1.30 m | 6.04 m | 0.36 m | 3.20 m | 7.64 m |
| **2.25（本场景）** | **7.50 m** | **0.50 m** | **6.84 m** | **0.36 m** | **4.00 m** | **7.64 m** |

* **坑总长恒为 0.36 m**（两处 0.18 m）⇒ 乘子**不会**再造深坑；补出的平地段总长 = `160·x_unit·(scale − 1)`
  = `3.2·(scale − 1)`；"图案区间内的可走面"随 scale **单调增**（2.84 → 6.84 m）。
* ⚠️ **一处与字面要求的冲突（如实报告）**：**整块瓦片**的可走面总长**恒为 7.64 m，不随 scale 变**
  —— 因为 `可走面 + 坑 = size[0]` 是恒等式，而坑总长恒定；图案越长，尾廊就越短（4.50 → 0.50 m），
  拉伸量被尾廊吸收。因此"**可走面总长随 scale 单调增**"只能对**图案区间内**（或"补出的平地段"）成立，
  本文与测试按后者表述（`TestMixTerrainGeometry.test_fill_mode_walkable_grows_in_the_pattern_and_pits_stay_constant`）。
* **对照：`fill_stretched_gaps=False`（第二/三批的旧行为）** —— 被拉开的空间由整宽 `-0.50 m` 坑底填充：

  | scale | 可走面总长 | 坑总长 | 图案末端 | 尾廊 |
  |---|---|---|---|---|
  | 1.0 | 7.64 m | 0.36 m | 3.50 m | 3.50 → 8.00 m |
  | 1.5 | 6.04 m | 1.96 m | 5.10 m | 5.10 → 8.00 m |
  | 2.0（第二批用过） | 4.44 m | 3.56 m | 6.70 m | 6.70 → 8.00 m |

  即：不补空档时"乘子越大 ⇒ 深坑越多越长 ⇒ 地形越难"，与"更从容"的意图相反（第二批的结论）。
  第四批的 `fill_stretched_gaps=True` 是**用户当时列出的三条候选改法中的方案①**（"把段与段之间空出来
  的区间补成 `height=0` 的走廊"），方案②/③（只对本来相接的段铺平地／把乘子降到 1.5）本方案没有采用
  （前者会保留一条"原生坑仍在间隙前端"的歧义版本，后者达不到"占满整条道"）。

### 6.4 障碍自身的尺寸与 scale 无关（逐块核对）

`d = 0.70`、任意 `scale ∈ {1.0, 1.25, 1.5, 2.0, 2.25}`：

| 指标 | 值 | 说明 |
|---|---|---|
| 图案段数 | 12 | 与 scale 无关（不新增/不丢图案） |
| 高栏（170 索引）顶面 | **0.2618 m** | `= 170 × 0.002 × 1.1 × 0.70`，`TestMixTerrainGeometry` 在 4 个 scale 上逐一断言 |
| 第一处原图案坑宽 | **0.18 m** | `= (72−3−60) × 0.02`（`k = round(10 − 10×0.70) = 3`） |
| 每段宽度 / 顶面高度 | 与 `scale = 1.0` 逐块相等 | 只有 `x0` 后移（`test_obstacle_geometry_is_scale_independent`） |
| 抬升台阶顶面（d=0.70） | 0.0462 / 0.0924 / 0.1386 / 0.1848 m | 与第二/三批记录一致 |

## 7. 出生点真几何与**可走面/坑 X 区间实测表**

用**桩 `isaaclab` ＋ 真 `trimesh`** 真跑 `track_mix_terrain`，把返回的网格摊成"沿 X 的可走区间/坑区间"，
再取 `spawn_x = 0.75` 的读数（`d = 0.70`、`pattern_start_x = 0.30`、`size = (8,4)`）。

### 7.1 第四批（`fill_stretched_gaps=True`）：出生点安全

| scale | 出生点落在 | 该平台顶面 | **出生点前方还有多少实心地面** | 第一处坑 | 坑总长 | 整片可走面 |
|---|---|---|---|---|---|---|
| 1.0 | `[0.30, 0.90]`（0.60 m 起步段） | **0.000 m**（可走） | 0.75 m ⚠️ | `[1.50, 1.68]`（宽 0.18 m） | 0.36 m | 7.64 m |
| **2.25（本场景）** | `[0.30, 0.90]` | 0.000 m | **1.95 m** ✅ | `[2.70, 2.88]`（宽 0.18 m） | 0.36 m | 7.64 m |

* 规格要求"`spawn_x = 0.75` 落在实心可走面上、前方实心 ≥ 1.0 m，覆盖 `scale ∈ {1.0, 反算值}`"：
  **占满值 2.25 满足（1.95 m）**；**`scale = 1.0` 不满足（只有 0.75 m）** —— 该数字只由图案与
  `pattern_start_x` 决定（第一处坑恒在 `0.30 + 60×0.02 = 1.50 m`），`scale = 1.0` 时没有任何可拉余量、
  补块列表为空 ⇒ **`fill_stretched_gaps` 对它没有任何影响**。要把它推到 ≥ 1.0 m 就得改
  `pattern_start_x`/图案本身 ⇒ 会破坏"默认路径逐位不变"的硬要求 ⇒ **本轮只记录，未做**（见 §14）。
* 对照（同一 `scale = 2.25` 但**不补空档**）：第一处"坑"紧贴起步平台末端 `[0.90, …]` ⇒ 前方只剩
  **0.15 m**（小于机身约 0.2 m 的半长 ⇒ 出生瞬间前足已探进 0.50 m 深坑）。这就是第二批/第三批记录的
  "出生点在一个很大的 gap 上"；**补空档同时把这个问题一并解决了**。

### 7.2 真 trimesh 实测的 X 区间表（`scale = 1.0` 对照 `scale = 2.25`）

`(x0, x1, 顶面高度)`，单位 m；`pits` 是相邻可走块之间**空出来**的区间（坑底 −0.50 m）：

| — | `scale = 1.0`（训练几何 / 对照） | `scale = 2.25`（本场景，占满整条道） |
|---|---|---|
| 起步平台 + 前 5 段 | `[0.00, 0.30]ᵖ`、`[0.30, 0.90]`、`[0.90, 1.02]`、`[1.02, 1.14]`、`[1.14, 1.26]`、`[1.26, 1.50]` | `[0.00, 0.30]ᵖ`、`[0.30, 0.90]`、`[0.90, 1.65]`、`[1.65, 1.77]`、`[1.77, 1.92]`、`[1.92, 2.04]`、`[2.04, 2.19]`、`[2.19, 2.31]`、`[2.31, 2.46]`、`[2.46, 2.70]` |
| 坑① | `[1.50, 1.68]`（宽 0.18） | `[2.70, 2.88]`（宽 0.18） |
| 中段 | `[1.68, 1.98]`、`[1.98, 2.02]`、`[2.02, 2.22]`、`[2.22, 2.28]`、`[2.28, 2.52]` | `[2.88, 3.405]`、`[3.405, 3.705]`、`[3.705, 4.08]`、`[4.08, 4.12]`、`[4.12, 4.17]`、`[4.17, 4.37]`、`[4.37, 4.62]`、`[4.62, 4.68]`、`[4.68, 4.755]`、`[4.755, 4.995]` |
| 坑② | `[2.52, 2.70]`（宽 0.18） | `[4.995, 5.175]`（宽 0.18） |
| 尾段 | `[2.70, 3.10]`、`[3.10, 3.50]`、`[3.50, 8.00]` | `[5.175, 5.70]`、`[5.70, 6.10]`、`[6.10, 6.60]`、`[6.60, 7.00]`、`[7.00, 7.50]`、`[7.50, 8.00]` |

（`ᵖ` = 起步走廊 `[0, pattern_start_x]`，其图案锚点恒为 0 ⇒ 与 scale 无关。顶面高度序列与 scale 无关：
`0.000 / 0.0462 / 0.0924 / 0.1386 / 0.1848 / 0.1848 / 0.000 / 0.1478 / 0.2618 / 0.1848 / 0.1848 / 0.0924`。
两张表都逐格钉在 `TestMixTestFilledSpawnGeometry.MEASURED` 与 `test_measured_x_interval_table` 里。）

对应的回归锁定：`TestMixTestFilledSpawnGeometry`（第四批：反算前提、两 scale 出生点安全、不补空档的
0.15 m 对照、上面两张区间表）、`TestMixTestSpawnGeometry`（第二/三批 `fill=False` 旧行为的回归对照）、
以及 `TestMixTerrainGeometry` 里对间距/补空档语义的全部断言。

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
* **改间距 / 占满程度**：`MIX_TEST_PATTERN_SPACING_SCALE`（现值 **2.25**＝反算的"占满整条道"值；
  上限 2.40625，超过会在 `track_mix_terrain` 里直接 raise）＋ `MIX_TEST_TAIL_MARGIN` /
  `MIX_TEST_PATTERN_END_X`（同一算式的两个中间量，只作文档/审计核对用，**不参与生成**）与
  `MIX_TEST_FILL_STRETCHED_GAPS`（现值 True；**必须与乘子一起改** —— 关掉它，拉开的空档会变回整宽
  −0.50 m 深坑、出生点前方实心地面会掉到 0.15 m）。改之前请读 §6.1–§6.3：乘子每 +0.05 ⇒ 图案末端
  +0.16 m、尾部平地 −0.16 m（`pattern_end = 0.30 + 3.2·scale`）。
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
* **第四批新增**：工具额外记录 mix 调用的**全部 kwargs**（`sub_terrain_calls`）、真读
  `terrain_generator.size`（新函数 `terrain_size()`），并对 mix-test 增加三项检查 —— 乘子必须 =
  反算值 **2.25**、mix 调用必须显式 **`fill_stretched_gaps=True`**、图案末端
  `pattern_start_x + 160·x_unit·scale` 必须 **≥ 7.0 m** 且 **≤ `size[0]`**（并打印乘子上限 2.40625）；
* 测试对 test 任务**单独断言**，并加**负向对照**（把"只有 mix"丢给原判据必须仍报错、存在但 0 列也必须
  报错、**期望道数/行数/难度/乘子被改成别的值时必须报 ❌**、**目标下界抬高 ⇒ 报"未占满整条道"**、
  **`fill_stretched_gaps` 不是 True ⇒ 报 ❌**）证明判据没被放宽也没被绕过。

实测输出（`python3 scripts/tools/check_terrain_columns.py --task mix-test`，退出码 0）：

```
=== [Imgo2-basemove-rough-cmoe-mix-test] 受控测试场景（只 mix；cfg 类 Imgo2CMoEMixTestEnvCfg）===

=== Imgo2-basemove-rough-cmoe-mix-test｜只 mix（20 道并列）（num_cols=20，共 20 列）===
  mix                     20 列
  ✅ 唯一地形 mix 有 20 列（num_cols=20、num_rows=1）⇒ 网格 **20 道 × 1 难度行**（世界 8 m(X) × 80 m(Y)），
     难度由 difficulty_range=(0.7, 0.7) 精确固定 ⇒ `--num_envs ≤ 20` 时环境 i → 第 i 道

  ── 第四批检查：`mix` 图案占满整条道（瓦片 size[0] = 8 m）──
  ✅ `pattern_spacing_scale` = 2.25（反算值：(8 − 0.3 − 0.50)/(160×0.02)）＋ `fill_stretched_gaps=True`
     ⇒ 图案末端 = 0.3 + 160×0.02×2.25 = 7.50 m（占 93.75 % 的 8 m 道），尾部平地 0.50 m；
     乘子上限 (8−0.3)/(160×0.02) = 2.40625（溢出保护**不放宽**：超限直接 raise）
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
4. **间距乘子 > 1 会把"缝"变成更长的深坑**（旧 §6.1，现为 §6.3 的对照表）⇒ 地形**更难**，与"更从容"
   的意图相反。**第四批已按用户列出的方案① 解决**（`fill_stretched_gaps=True` 把拉开余量铺成
   `height=0` 可走面，原有坑宽不变）；方案②/③ 未采用（见 §6.3 末段）。
5. **出生点问题只被"绕开"、没有被"修好"**（§7.1/§8）：评测场景靠"占满值 2.25 ＋ 补空档"把前方实心
   由 0.15 m 提到 **1.95 m**，但训练侧 `spawn_x = 0.75` + `pose_range ±0.5` 的同类隐患**仍在**
   （问题表 **CMOE-17**，待决）。
6. **`track_gap_terrain` 的溢出保护仍未做**（问题表 CMOE-13）：本轮只保留 `track_mix_terrain` 的保护，
   没有顺手扩到 `track_gap_terrain`（超出本次范围）。
7. **`check_reward_overrides.py` 未加 `cmoe-mix-test` 链**：本场景**完全不改奖励**（生效项仍是 `cmoe`
   的 28 项），加一条与 `cmoe` 逐项相同的链没有信息量，故未加。
8. **第四批的两处"与字面要求对不上"的地方（本轮只记录，未擅自扩范围）**：
   * **`scale = 1.0` 时"前方实心 ≥ 1.0 m"不成立**（实际 **0.75 m**）。规格要求"出生点安全……覆盖
     `scale ∈ {1.0, 反算值}`"，但 `scale = 1.0` 正是**训练几何本身**，没有任何可拉的余量（补块列表为空），
     第一处坑恒在 `0.30 + 60×0.02 = 1.50 m`。要满足它必须改 `pattern_start_x`/图案本身 ⇒ 会破坏
     "默认路径逐位不变"的硬要求 ⇒ 本轮**不动**；占满值 2.25 满足（1.95 m）。
   * **"可走面总长随 scale 单调增"在整块瓦片上不成立**：`可走面 + 坑 = size[0]`（8 m），坑总长恒定
     0.36 m ⇒ 整片可走面**恒为 7.64 m**（尾廊吸收拉伸量：4.50 → 0.50 m）。随 scale 单调增的是
     **图案区间内的可走面**（2.84 → 6.84 m）与**补出的平地段**（0 → 4.00 m）——本文件、cfg 注释与测试
     都按后者表述，并把"整片恒定"作为不变式一并锁住（§6.3 的 ⚠️）。
9. **尾段的归属**：本方案把"最后一段末端 → 图案末端（索引 160）"这段也视作"拉开的空档"并铺平
   （`scale = 2.25` 时是 `[7.00, 7.50]`）；参考图案在索引 140:160 之后本来就没有赋值 ⇒ 铺成 0 高度与
   参考语义一致（原来在 `scale = 1.0` 时这段是**尾廊**的一部分，也铺在 0 高度）。**四处坑/空档的
   归属选择**（坑紧贴**上游**块末端、余量铺平）是本方案的自定细节，参考实现里没有对应概念 ⇒
   若以后要按"坑紧贴下游"的语义改，只需改 `track_mix_terrain` 的 fill 分支与本文 §6.2。

## 15. 验证与实测输出（本机离线）

```bash
cd imgo2_rl
/opt/conda/envs/isaaclab/bin/python -m unittest discover -s tests     # 667 通过 / 10 跳过
python3 scripts/tools/check_reward_overrides.py cmoe-gaitfree          # 24 项（cmoe = 28 项，与改前一致）
python3 scripts/tools/check_terrain_columns.py                         # 末行"全部掩码引用的地形名都有 ≥1 列 ✅"
python3 scripts/tools/check_terrain_columns.py --task mix-test         # "mix 20 列" + 第四批核算行 + 末行"…属预期 ✅"（退出码 0）
git diff --check                                                       # 干净
```

* **基线**：第四批改动前 `unittest discover -s tests` = **651 通过 / 10 跳过**；
  第四批后 = **667 / 10**（`test_cmoe_mix_test_scene.py` 72 → **82**、`TestMixTestTerrainColumns`
  20 → **26**；共新增 16 项）。
* **"难度确实 = 0.70"的两条路径**：见 §5.1（公式复现 `places=15`；真跑 `track_mix_terrain` 核对
  栏高 0.2618 m、两处坑宽各 0.18 m 与第一处坑左沿 `2.70 m`（占满值））；另一条源码级断言在已安装的
  Isaac Lab 里逐字核对 `difficulty_range` / 两条分支公式 / 本仓 `curriculum = True` 的开启条件
  （缺 Isaac Lab 时该组 skip）。
* **"默认路径逐位不变"（硬要求）**：`fill_stretched_gaps=False` 时对
  `d ∈ {0, 0.35, 0.70, 0.75, 1.0} × size ∈ {(8,4), (8,8)}` 共 **10 组**逐块比较 `mesh.bounds`
  ⇒ **10/10 逐位相同**；`fill_stretched_gaps=True` 但 `scale = 1.0` 时补块列表为空 ⇒ 与 `False`
  **完全相同**（两条都钉在 `TestMixTerrainGeometry.test_fill_flag_and_defaults_are_neutral` 与
  `test_default_spacing_reproduces_the_pre_change_geometry` 里）。
* **"只拉间距、不多深坑"**：`scale ∈ {1.0, 1.25, 1.5, 2.0, 2.25}` 上坑总长恒 **0.36 m**、图案区间内
  可走面 `2.84 → 6.84 m` 单调增、补出平地 = `3.2·(scale−1)`（`0 → 4.00 m`）、障碍逐块宽/高与
  `scale = 1.0` 相等、栏高恒 0.2618 m（`test_fill_mode_walkable_grows_in_the_pattern_and_pits_stay_constant`
  / `test_obstacle_geometry_is_scale_independent`）。
* **出生点真几何实测**：见 §7.1/§7.2 两张表（`d = 0.70`；`scale = 1.0` 与占满值 **2.25**；
  读数由 `TestMixTestFilledSpawnGeometry` 逐格锁定；第二/三批 `fill=False` 的旧读数由
  `TestMixTestSpawnGeometry` 继续锁定）。
* **训练侧隐患量化实测**：见 §8（`TestTrainingSpawnHazard` 锁定：第一处坑恒在 1.50 m、x 越界阈值
  `spawn_x > 1.00`、35% 落在台阶段、y 方向 20% 出走廊）。**本轮未改训练侧任何参数**。
* **溢出保护实测（沿用第二批 + 第四批复核）**：`scale=2.5` 抛
  `ValueError: track_mix_terrain: 图案总长 8.3000 m 超出瓦片长 size[0]=8.0000 m …本瓦片上
  pattern_spacing_scale 最大可取 2.4062…`；`scale=2.40625` 不抛（`fill_stretched_gaps=True` 时同样成立，
  见 `test_overflow_guard_is_not_relaxed_by_fill_mode`）。
* **"默认 1.0 无差异"（沿用第二批的验证方式）**：把 `b2caad1` 的 `cmoe_terrains.py` 与改后版本分别
  exec（桩 `isaaclab` + 真 `trimesh`），对 `d ∈ {0, 0.35, 0.70, 0.75, 1.0} × size ∈ {(8,4), (8,8)}`
  逐块比较 `mesh.bounds` 与 `origin`：**10/10 逐位相同**（参照公式表 `_pattern_rows` 就来自改动前，
  第四批新增的 fill 分支在 `fill_stretched_gaps=False` 时**一行都不执行**）。
