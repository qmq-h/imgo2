# `mix` 受控测试场景（`Imgo2-basemove-rough-cmoe-mix-test`）— 2026-10-04

本文件记录**同一天的五批**改动（都在 `imgo2_CMoE` 分支上，HEAD `dd8b1f2` 之后未提交）：

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
* **第四批**（用户："mix 还是太小了…课程长度太短：让 mix 占满整条道"）：

  ① `cmoe_terrains.CMoETrackMixTerrainCfg` 新增布尔字段 **`fill_stretched_gaps`（默认 `False`）**：
     为 `True` 时把**因拉开而多出来的空档**（段与段之间、最后一段末端 → 图案末端的"尾段"）铺成
     `height=0` 的可走面；**原图案本来就有的两处坑原样保留**（`d = 0.70` ⇒ 各 0.18 m），
     **障碍自身的宽度/高度/顺序一字不变**。默认 `False` ⇒ 训练与既有测试的几何**逐位不变**；
  ② 评测场景 `pattern_spacing_scale` 取 **反算的"刚好占满整条道"值 `2.25`** ＋ `fill_stretched_gaps
     = True` ⇒ 图案末端由 **3.50 m** 推到 **7.50 m**（占 8 m 道的 93.75 %，尾部留 0.50 m 平地）；
     乘子仍严格小于溢出上限 **2.40625**（保护**不放宽**，超限照旧直接 `ValueError`）；
  ③ 出生点前方的实心地面由 **0.75 m** 变成 **1.95 m**（第一处坑由 1.50 m 后移到 2.70 m）。
* **第五批（本轮）**（用户原话："整体地形放大，原本是10m长就改成20m长，还是布满，但是障碍数量不变，
  设置不变，只把间隔改大"）：

  ① **单块瓦片的 X 由 8 m 放大到 20 m**（`terrain_generator.size = (20, 4)`；Y 仍 4 m）。`size` 是
     `terrain_generator` 的**共享字段** ⇒ 只有**本评测 cfg** 覆盖它，训练/play 两条链的
     `terrain_generator.size` **仍是 `(8, 4)`**（测试与 `check_terrain_columns.py` 都钉住）；
  ② **课程继续铺满整条 20 m**：乘子按同一套反算式（尾部余量仍 0.50 m）重算为
     `(20 − 0.30 − 0.50)/(160 × 0.02) = 19.20/3.20 = **6.00**` ⇒ 图案末端 =
     `0.30 + 160×0.02×6.00 = **19.50 m**`（占 97.5 %，尾部留 0.50 m 平地）；仍严格小于溢出上限
     `(20 − 0.30)/(160×0.02) = **6.15625**` ⇒ **溢出保护不放宽**；
  ③ **障碍数量 / 障碍自身尺寸与高度 / 坑深 / 走廊宽 / 难度(0.70) / 图案顺序全部不变** ——
     被改的只有障碍之间的间隔（真 trimesh 逐块核对：12 段宽度/顶面与 `scale = 1.0` 相等、
     高栏顶面恒 0.2618 m、第一处坑宽恒 0.18 m、坑总长恒 0.36 m）；
  ④ **单局时长由继承来的 20 s 提到 35 s**（道 20 m ÷ 1.0 m/s = 20 s ＋ 15 s 余量）——
     不改的话"刚好卡在走完就到时"；**只影响该评测任务**，训练/play 时长一字未改（见 §6.5）。
* **第六批（本轮）**（用户实测原话："**台阶似乎只有一级**"；随后追加
  "我似乎有看到一个**较高的台子后紧接 gap**"）：

  ① **拉伸/补平地的粒度从"每一段"改成"每个障碍组"**（run of contiguous segments）：`d = 0.70` 时
     12 段按"原始 units 上是否首尾相接"切成 **3 组**（`0→60` 起步平台 ＋ **4 级楼梯**、`69→111`、
     `120→160`），**组内整组共用一个平移量** ⇒ 段与段仍首尾相接、宽度/顶面/顺序不变 ⇒
     **4 级楼梯连成楼梯**（真 trimesh：`[0.90,1.02] 0.0462 / [1.02,1.14] 0.0924 / [1.14,1.26] 0.1386 /
     [1.26,1.50] 0.1848`，级间**无平地**）；
  ② **组间**把多出来的总长 `160·x_unit·(scale−1)` **均匀分给 3 个空档**（两处坑所在空档 ＋ 尾段；
     `scale = 6.00` 每处 **5.3333 m**）并铺 `height=0` 可走面；**坑宽恒 0.18 m**（各自的位置由
     "紧贴下游组起点"决定、与 `scale` 无关）；图案末端仍 **19.50 m**、可走面 19.64 m、坑总长 0.36 m；
  ③ **追加修正：补出的平地铺在"坑的上游"**（坑紧贴**下游组起点**）⇒ "从高处下来先有平地助跑、
     再遇到坑"——每处坑前 **5.3333 m** 平地（≥ 0.5 m）、坑后落点平地 **0.00 m**；
     第一处坑由 `[1.50, 1.68]`（紧贴楼梯末端）改为 `[6.8333, 7.0133]` ⇒ 出生点到第一处坑 **6.08 m**；
  ④ **默认路径逐位不变**：`fill_stretched_gaps=False`（默认，训练侧那条）仍走第四批的"逐段"公式，
     `scale = 1.0` 时 `shift` 精确为 `+0.0`；分组布局只在 `fill_stretched_gaps=True` **且** `scale > 1`
     时启用；**溢出保护不放宽**（仍按 `pattern_start_x + 160·x_unit·scale ≤ size[0]`，超限 `ValueError`）。
  详见 §6.6（教训、语义、几何）与 §7.3（实测表）。

* **第七批（本轮，2026-10-05）**（用户原话：「是有些障碍分到一起了，我说的是障碍本身。gap 和突台也
  在一起。我希望是和单独的那个几个场景类似的障碍放在这个整条的（道）上」；追加「每个地形要间隔开，
  而不是混合到一起放在一个位置」；再追加并确认顺序与楼梯形态）：
  **本场景的地形由"参考 `mix` 图案"改成"复合道"**
  （`cmoe_terrains.track_composite_terrain` ＋ `CMoETrackCompositeTerrainCfg`；`sub_terrains` 键名
  `composite`）—— 沿 +X 依次 **坑 → 楼梯（上 4 级 + 下 4 级紧贴）→ 箱子/台阶块 → 栏 → 坑**，
  每个障碍都用**它自己那类独立地形**的难度律与几何，**所有相邻障碍之间都是同一个
  `obstacle_spacing`（反算 2.3636 m ≥ 下界 1.50 m）** ⇒ 不再有"坑紧挨抬高块"、也不再有参考图案
  专有的 4 cm 窄槽 / 0.26 m 尖峰；总长刚好 20.00 m。详见 §16。

**这不是新策略任务**：动作空间、观测组与维度、奖励、终止**一字未改**，只是换了一个场景 + 一层指令
外环 ⇒ 既有 CMoE checkpoint（77 维地形 / 527 维 actor / 125 维 critic）**可直接加载**。

## 1. 目的与范围

用途：在受控条件下评估策略通过地形障碍的能力，**把横向漂移与航向漂移从评测里剔除**，并且
**难度固定**（不再受课程随机初始等级影响）。第三批要求"**道并列、但不要那么多行**"，于是整个世界
只剩**一行**难度、20 条道沿 Y 并排；第四批要求"**让 mix 占满整条道**"；第五批要求"**整体地形放大**：
单块瓦片 X 8 m → 20 m，还是布满、障碍数量不变、只把间隔改大"。

**第七批（本轮）改变了"地形是什么"**：参考实现 `mix_obstacles_terrain` 的小图案（窄走廊 + 台阶上行 +
深坑 + 高台 + 高栏 + 深坑）把坑与抬高块**挤在一起**、还自带 4 cm 窄槽与 0.26 m 长的尖峰 —— 用户
反馈「是有些障碍分到一起了，我说的是障碍本身。gap 和突台也在一起」，并要求「和单独的那个几个场景
类似的障碍放在这个整条的（道）上」「每个地形要间隔开，而不是混合到一起放在一个位置」。因此评测
场景的 `sub_terrains` 只剩**复合道**一类；`mix` 图案的字段与实现**原样保留**（训练侧仍在用），
只是不再被本场景引用。**训练/play 一字未改**。

## 2. 改动清单（第五批）

| 文件 | 改动 |
|---|---|
| `.../velocity/base_move/CMoE_env_cfg.py` | 新增常量 `MIX_TEST_TILE_SIZE = (20.0, 4.0)`、`MIX_TEST_EPISODE_LENGTH_S = 35.0`、`MIX_TEST_EPISODE_LENGTH_MIN_S = 25.0`、`MIX_TEST_FORWARD_SPEED = 1.0`；`MIX_TEST_PATTERN_SPACING_SCALE` **2.25 → 6.00**、`MIX_TEST_PATTERN_END_X` **7.50 → 19.50**（`MIX_TEST_TAIL_MARGIN` 仍 0.50）；`Imgo2CMoEMixTestEnvCfg.__post_init__` 新增 `self.scene.terrain.terrain_generator.size = MIX_TEST_TILE_SIZE` 与 `self.episode_length_s = MIX_TEST_EPISODE_LENGTH_S`；头注释/类 docstring/世界坐标注释（X ∈ [−10,+10]、出生点 X = **−9.25**）整段同步 |
| `.../velocity/base_move/cmoe_terrains.py` | **函数一行未改**（放大的是 `cfg.size`）；只同步注释/docstring：场景取值 2.25 → **6.00**、"8 m 道"→"20 m 道"、该瓦片乘子上限 2.40625 → **6.15625**、图案末端 7.50 → **19.50 m** |
| `imgo2_rl/scripts/tools/check_terrain_columns.py` | `terrain_size()` 由"全文件第一个匹配"改成**按类作用域**（`terrain_size(class_name)`；原来会先撞到训练类的 `(8, 4)`）；新增 `episode_length_s(class_name)` 与 `PLAY_CLASS`；mix-test 期望值改 **6.00 / 19.0 m / (20,4) / 35 s**；mix-test 分支新增两组检查与打印（**评测 size = 20×4 且训练/play size 仍 = 8×4**、**单局时长 ≥ 25 s**） |
| `imgo2_rl/tests/test_cmoe_mix_test_scene.py` | 82 → **84 项**：全部数值期望按新事实更新（`TILE_SIZE` 20 m、世界 X ∈ [−10,+10]、出生点 X = −9.25、乘子 6.00、图案末端 19.50、上限 6.15625、整片可走面 19.64 m、图案区间可走面 2.84 → 18.84 m、前方实心 5.55 m、`MEASURED` 表在 20 m 道上重跑）；新增 `test_tile_size_is_20_for_the_test_and_8_for_training`（**训练侧 size 未被改**）、`test_episode_length_is_long_enough_for_the_twenty_metre_lane`、`test_training_and_play_episode_length_are_untouched`；"契约不变"断言改成"非契约覆盖只允许 `size` / `episode_length_s` 两处" |
| `imgo2_rl/tests/test_check_terrain_columns.py` | `TestMixTestTerrainColumns` 26 → **31 项**（整个文件 34 → **39 项**）：常量期望改 6.00/19.50/0.50/True + 新常量；`terrain_size()` → **按类**断言（评测 (20,4)、训练 (8,4)、play 退化成 (8,4)）；新增 `episode_length_s()` 按类断言、瓦片尺寸正向/负向对照、**"训练侧被放大"负向对照**、单局时长正向/2 条负向对照 |
| `README.md` | 任务表该行、CMOE-16/17 状态、§8.3 维护记录（第五批一行）同步 |

### 2.1 第六批改动清单（本轮）

| 文件 | 改动 |
|---|---|
| `.../velocity/base_move/cmoe_terrains.py` | `track_mix_terrain` **真正改动**：① 新增**分组**（非空 `segments` 按"原始 units 上是否首尾相接"切成组）；② 组内**整组共用一个平移量**（`segment_shifts`），默认路径（`fill_stretched_gaps=False` **或** `scale = 1.0`）仍走原来的**逐段** `_shift(start_units)` 公式 ⇒ **逐位不变**；③ `fill_stretched_gaps=True` 且 `scale > 1` 时：可拉伸空档 = 原始就有空档的组间边界 ＋ 尾段，`extra = 160·x_unit·(scale−1)` **均匀分配**，**原有坑宽不变且紧贴下游组起点**，补出的 `height=0` 平地铺在**坑的上游**（`stretched_fills`）；④ 溢出保护与图案末端公式**未动**；⑤ 函数 docstring 新增第六批段落，第四批段落标注"逐段粒度只在 `fill_stretched_gaps=False` 时成立" |
| `imgo2_rl/scripts/tools/check_terrain_columns.py` | 新增 `MIX_CONTIGUOUS_GROUPS_EXPECTED / MIX_GROUP_SPANS_EXPECTED / MIX_STAIR_LEVELS_EXPECTED / MIX_STAIR_TOPS_EXPECTED / MIX_GAP_SLOTS_EXPECTED / MIX_PIT_WIDTH_EXPECTED / MIX_UPSTREAM_FLAT_MIN_S` 与 `CMOE_TERRAINS` 路径；新增**只用标准库**的 `mix_terrain_defaults()`（AST 读 cfg 默认值）、`mix_pattern_segments()`（AST 解析真 `segments` 表）、`mix_contiguous_groups()`、`mix_group_layout()`（独立几何核算）、`mix_hard_combinations()`（只报告清单）；mix-test 分支新增「**第六批检查：分组语义**」打印与检查（分组数/组内首尾相接/楼梯 4 级连续且级间无平地/坑宽 0.18/空档数 3/坑前平地 ≥ 0.5 m）＋「图案里其它『难组合』（只报告）」三行。**原任务与 mix-test 既有判据一字未动、默认任务输出逐字未变** |
| `imgo2_rl/tests/test_cmoe_mix_test_scene.py` | 84 → **92 项**：新增 `test_segment_table_matches_the_reference`、`test_grouping_is_implemented_in_the_source`、`test_contiguous_groups_are_three_and_include_the_staircase`、`test_staircase_is_four_contiguous_levels_at_full_scale`、`test_pit_width_is_constant_and_the_pit_abuts_the_downstream_group`、`test_slack_is_distributed_evenly_between_the_two_pit_gaps_and_the_tail`、`test_grouped_layout_matches_the_independent_reference`、`test_first_obstacle_group_is_bitwise_scale_invariant`、`test_fill_false_keeps_the_pre_change_per_segment_layout`（净 +8：删掉旧的 `test_fill_true_only_adds_flat_surface_on_the_stretched_slack`）；新增**独立参照** `_grouped_layout()` / `_pattern_piece_signature()` / `_fill_pieces()` / `_missing_signatures()`（删掉 `_extra_pieces()` / `_expected_slack_intervals()`）；`MEASURED` 表与出生点/尾部读数按 §7.2 重跑 |
| `imgo2_rl/tests/test_check_terrain_columns.py` | `TestMixTestTerrainColumns` 31 → **39 项**（整个文件 **39 → 47 项**）：新增 4 条正向（源码解析、分组、几何核算、报告文本）＋ 4 条负向对照（组数/楼梯级数/坑前平地下界/难组合只报告） |
| `README.md` | 顶部核对段、CMOE-16（状态"六批"＋第六批段落）、§8.3 维护记录新增第六批一行 |

### 2.2 第七批改动清单（本轮，2026-10-05）

| 文件 | 改动 |
|---|---|
| `.../velocity/base_move/cmoe_terrains.py` | **新增**（`track_mix_terrain` 一行未改）：① 模块级常量 `COMPOSITE_OBSTACLE_SEQUENCE = ("gap","stairs","boxes","hurdle","gap")`、`COMPOSITE_MIN_OBSTACLE_SPACING = 1.50`、`COMPOSITE_SUGGESTED_OBSTACLE_SPACING = (1.50, 2.50)`、`COMPOSITE_SPACING_AUTO = -1.0`；② `_composite_obstacle_units()`（按各类独立地形的难度律算障碍单元几何）；③ `composite_track_layout()`（纯算术布局：反算/显式间隔、逐对相邻间隔表、溢出与下界保护）；④ `composite_adjacency_violations()`（相邻间隔违规清单，供审计与负向对照）；⑤ `track_composite_terrain()`（真网格：整宽 −0.50 m 基座 ＋ 0.80 m 走廊 ＋ 各障碍抬高块，坑处**不铺走廊**）；⑥ `CMoETrackCompositeTerrainCfg`（字段默认值＝训练侧各类地形的实例值，`stairs_num_steps=4` 与栏的高度/间距分位是有意的确定性选择） |
| `.../velocity/base_move/CMoE_env_cfg.py` | `Imgo2CMoEMixTestEnvCfg.__post_init__` 的 `sub_terrains` 由 `["mix"] = CMoETrackMixTerrainCfg(...)` 换成 `["composite"] = CMoETrackCompositeTerrainCfg(...)`（字段**显式展开**）；常量 `MIX_TEST_PATTERN_SPACING_SCALE / MIX_TEST_TAIL_MARGIN / MIX_TEST_PATTERN_END_X / MIX_TEST_FILL_STRETCHED_GAPS` **删除**，替换为 `MIX_TEST_SUB_TERRAIN_KEY = "composite"`、`MIX_TEST_OBSTACLE_SPACING = 2.3636`、`MIX_TEST_MIN_OBSTACLE_SPACING = 1.50`、`MIX_TEST_SPAWN_CLEARANCE = 1.50`（含反算式注释）；头注释/类 docstring 整段同步 |
| `imgo2_rl/scripts/tools/check_terrain_columns.py` | mix-test 分支的 mix 图案检查（第四/五/六批）**整体换成复合道检查**：`_top_level_constants()`（同时收 `Assign` 与 `AnnAssign`）、`composite_terrain_defaults()`、`composite_units()`、`composite_layout()`（**只用标准库**的算术复算）、`_composite_detail()`、`COMPOSITE_*` 期望值；报告新增「第七批检查：复合道」＝障碍清单 ＋ **X 区间表** ＋ **逐对相邻障碍间隔表** ＋ `✅ 最小相邻间隔 2.3636 m ≥ 下界 1.5000 m` ＋ `✅ 楼梯上下紧贴 …` ＋ `✅ 楼梯与相邻障碍间隔 = 2.3636 m ≥ 下界` ＋ 占满核算 ＋ 出生点，并新增「训练侧 `mix` 未受影响」段落。**`--task cmoe-rough` 的判据与输出一字未改、一字未放宽** |
| `imgo2_rl/tests/test_cmoe_mix_test_scene.py` | 92 → **112 项**：新增 `TestCompositeTerrainGeometry`（24 项，桩 `isaaclab` ＋ **真 trimesh**：X 区间表、障碍顺序/宽度、均匀间隔反算、**逐对相邻间隔 ≥ 下界**、**坑不与抬高块相邻**、**负向对照**（人造贴在一起 / 下界调大）、**楼梯上下紧贴**与楼梯两侧有平地、占满 20 m、出生点、**与四类独立地形逐项尺寸对照**、尺寸随难度变、溢出与下界保护、边界值、训练侧未受影响、源码级接线、**与 checker 标准库算术交叉核对**）；`TestMixTestEnvCfgSource` 里与旧 mix 场景绑定的断言改写为复合道（`test_composite_is_the_only_sub_terrain`、`test_scene_is_a_composite_track_that_fills_the_lane`、`test_composite_call_expands_the_defaults_without_drift`、`test_composite_params_match_training_instances`）；`TestMixTestDifficultyPinning` 新增真跑复合道的路径②。**训练侧 `mix` 的几何类（`TestMixTerrainGeometry` / `TestMixTestSpawnGeometry` / `TestMixTestFilledSpawnGeometry` / 训练出生点隐患）原样保留**（"默认路径逐位不变"护栏未动） |
| `imgo2_rl/tests/test_check_terrain_columns.py` | `TestMixTestTerrainColumns` **整体改写为复合道**（47 项）：常量/键名/`composite` 唯一地形/`sub_terrain_calls` 解析、`composite_layout()` 的期望值自洽、逐对相邻间隔 ≥ 下界、楼梯上下紧贴、出生点/占满、报告文本，负向对照（错误顺序/错误间隔期望值/错误下界/错误楼梯级数/错误末端/缺少 `composite` 调用/训练侧 mix 默认值被改/瓦片尺寸/训练侧被放大/单局时长）；训练侧 `mix` 的纯函数复核（段表/分组/几何核算/难组合）**保留** |
| `README.md` | 任务表该行、CMOE-16 状态（"七批"）、§8.3 维护记录新增第七批一行 |

第一/二/三/四/五/六批的改动清单（新增 `mdp/mix_test_pd.py`、`mdp/mix_test_command.py`、
`pattern_spacing_scale` 与 `fill_stretched_gaps` 字段、`--task` 分支、难度钉死、删钉等级子类、
瓦片 8 → 20 m、mix 图案分组铺平地、测试从 35 → 56 → 72 → 82 → 84 → 92 项等）见本文件的 git 历史与
README §8.3 的对应六行。


## 3. 场景契约

由 `Imgo2CMoEMixTestEnvCfg.__post_init__`（继承 `Imgo2CMoERoughPlayEnvCfg`）设定：

* **地形**：`self.scene.terrain.terrain_generator.sub_terrains.clear()` 后只放一处 `CMoETrackMixTerrainCfg`：

  ```python
  CMoETrackMixTerrainCfg(
      proportion=1.0,
      x_unit=0.02, z_unit=0.002, height_scale=1.1, gap_shrink_units=10.0,
      corridor_width=0.80, pit_depth=0.50, pattern_start_x=0.30,
      pattern_spacing_scale=MIX_TEST_PATTERN_SPACING_SCALE,   # = 6.00（第四/五批：反算的"占满整条 20 m 道"值）
      fill_stretched_gaps=MIX_TEST_FILL_STRETCHED_GAPS,       # = True（第四批：把拉开的空档铺成可走面；
                                                              #   第六批：粒度＝连续段组、坑宽不变、平地铺在坑的上游）
      spawn_x=0.75,
  )
  ```

  除 `proportion`（本场景只有这一类，比例无意义）与第四/五批的 `pattern_spacing_scale` /
  `fill_stretched_gaps` / `terrain_generator.size` 三处**有意偏离**外，其余字段**逐字沿用训练实例化时的值**
  （训练侧是 `CMoETrackMixTerrainCfg(proportion=0.10)`，其余字段全取类默认值）——
  `tests/test_cmoe_mix_test_scene.py` 会把这些**字面量**与 `cmoe_terrains.py` 的类默认值**逐一比对**，
  改默认值而不同步这里就会红。规格原文写的是 `prop_start_x=0.30`，本仓真实字段名是 `pattern_start_x`。
  **因此"障碍自身"的几何（每块宽/高/顺序）与训练仍逐字相同**，评测与训练的差异只在"障碍之间的间距"、
  "被拉开的空档是平地还是深坑"与"道有多长"。
* **网格**：`num_cols = MIX_TEST_LANES = 20`、`num_rows = MIX_TEST_LEVELS = 1`、`scene.num_envs = 20`
  ⇒ **20 条并列的 mix 道（沿世界 Y 铺 80 m）× 唯一一行难度（沿世界 X，长度＝单块瓦片长）**，
  默认一道一个环境。
* **单块瓦片尺寸**（第五批）：`terrain_generator.size = MIX_TEST_TILE_SIZE = (20, 4)` ⇒ 世界
  **20 m(X) × 80 m(Y)**、X ∈ **[−10, +10]**、出生点 X = `spawn_x − 10 = **−9.25**`；
  **训练/play 的 `size` 仍是 `(8, 4)`**（共享字段，只有本评测 cfg 覆盖）。
* **难度精确 0.70**：`terrain_generator.difficulty_range = (MIX_TEST_DIFFICULTY, MIX_TEST_DIFFICULTY)`
  ＋ `max_init_terrain_level = 0`，且**不覆盖** `scene.terrain.class_type`（默认 `TerrainImporter`）
  ⇒ **不依赖 CLI、也不需要自定义子类**（详见 §5）。
* **课程占满整条 20 m 道**（`MIX_TEST_PATTERN_SPACING_SCALE = 6.00` ＋ `fill_stretched_gaps = True`）：
  图案 X 总长 = `0.30 + 160×0.02×6.00 = 19.50 m`（占 20 m 道的 **97.5 %**，尾部留 0.50 m 平地）；
  反算式与铺平地规则见 §6.1/§6.2（**第六批起粒度＝连续段组、平地铺在坑的上游**，见 §6.6）；
  为什么不再是 1.0、旧行为是什么见 §6.3。
* **单局时长 35 s**（第五批）：道 20 m ÷ 1.0 m/s = 20 s ⇒ `episode_length_s` 由继承来的 **20 s** 提到
  `MIX_TEST_EPISODE_LENGTH_S = 35 s`（只影响本评测任务）；理由、副作用与训练侧未改的证据见 §6.5。
* **继承来的确定性**：`pose_range` / `velocity_range` 全 0、全部域随机化事件置 `None`、观测噪声关闭
  （`observations.policy/terrain.enable_corruption = False`）。
* **契约不变**：新类体里**没有**任何对 `self.observations` / `self.actions` / `self.rewards` /
  `self.terminations` / `self.curriculum` / `self.scene.height_scanner` 的赋值（测试里用 AST 逐条断言），
  也没有出现 `ObsGroup` / `ObsTerm` / `ActionCfg`。**第五批新增的两处非契约覆盖
  （`terrain_generator.size`、`episode_length_s`）是全部**——测试用白名单把这件事锁住，
  以后顺手多改一处就会红。

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

本场景 `size = (20.0, 4.0)`（**第五批由评测 cfg 显式覆盖**；父类 `Imgo2CMoERoughEnvCfg` 与训练/play
链仍是 `(8.0, 4.0)`）、`num_rows = 1`、`num_cols = 20`：

| 轴 | 范围 | 单块 | 本例 |
|---|---|---|---|
| 世界 X（难度行） | **`[-10, +10]`**（1 行 × 20 m） | 每行 20 m | **出生点 X = `spawn_x − 10 = −9.25 m`** |
| 世界 Y（mix 道） | **`[-40, +40]`**（20 道 × 4 m） | 每道 4 m | 第 i 道占 `[4i − 40, 4(i+1) − 40]`；道中心 `Y = 4i − 38`；道内 mix 走廊（宽 0.80 m）占 **`[4i − 38.4, 4i − 37.6]`**（第 0 道走廊 `[−38.4, −37.6]`，第 19 道 `[37.6, 38.4]`） |

排布图（俯视，X 向右＝前进方向，Y 向上＝道）：

```
           道0      道1      道2    ...            道19     ← num_cols=20（沿 Y，每道 4 m）
  X=+10 ┌────────┬────────┬────────┬─────┬────────┐
  唯一行 │  mix   │  mix   │  mix   │ ... │  mix   │        ← num_rows=1（沿 X 只有 20 m）
  X=-10 └────────┴────────┴────────┴─────┴────────┘
        世界 Y ∈ [−40, +40]；每道一条 0.80 m 宽的 mix 走廊，走廊外是 −0.50 m 坑底
```

**2026-10-04（第二批）的"20 行 × 160 m"是历史**：那版世界里 X ∈ [−80, +80]、全部环境钉在第 14 行
（占 X ∈ [+32, +40]），另外 19 行（152 m）永远用不到。第三批把它们删掉了。
**第三批的"1 行 × 8 m"也是历史**（X ∈ [−4, +4]、出生点 X = −3.25）：第五批把单块瓦片 X 放大到 20 m，
现在 X ∈ [−10, +10]、出生点 X = −9.25。

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
   在 **`[6.8333, 7.0133]`**（第六批：`0.30 + 69×0.02 + 16/3 − 0.18`，即「下游组起点 − 0.18」，坑前 5.3333 m 平地；第五批逐段粒度时是 `[6.30, 6.48]`）；且与 `d = 0.0 / 1.0` 明显不同。

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
  **训练侧 `size[0] = 8 m`** 时乘子上限 = `(8 − 0.30)/(160 × 0.02) = **2.40625**`、
  **评测侧 `size[0] = 20 m`**（第五批）时 = `(20 − 0.30)/(160 × 0.02) = **6.15625**`。
  第四/五批都**没有放宽**这条保护（`fill_stretched_gaps=True` 时超限仍 raise，
  测试 `test_overflow_guard_is_not_relaxed_by_fill_mode`）。

### 6.1 反算"刚好占满整条道"的乘子

用户第四批要求："让 mix 占满整条道"且"障碍自身的尺寸/高度不变（只把障碍之间拉开）"；第五批要求
"**整体地形放大，原本是 10 m 长就改成 20 m 长，还是布满，但是障碍数量不变，设置不变，只把间隔改大**"。
约束是 **图案末端 ≤ `size[0]` − 尾部余量**（尾部余量＝图案之后保留的可走平地段，本方案取 **0.50 m**）：

```
pattern_start_x + 160 · x_unit · scale ≤ size[0] − 尾部余量
⇒ scale ≤ (size[0] − pattern_start_x − 尾部余量) / (160 · x_unit)
        = (20.00 − 0.30 − 0.50) / (160 × 0.02)      ← 第五批：只把 size[0] 从 8 换成 20
        = 19.20 / 3.20
        = 6.00
```

取 **`scale = 6.00`** ⇒ 图案末端 = `0.30 + 160 × 0.02 × 6.00 = **19.50 m**`
（占 20 m 道的 **97.5 %**，≥ 目标 19.0 m / 95 %），尾部平地 = `20.00 − 19.50 = 0.50 m`；
`6.00 < 6.15625` ⇒ **不需要**放宽溢出保护（也没有改图案数量）。
常量落点：`MIX_TEST_PATTERN_SPACING_SCALE = 6.00`、`MIX_TEST_TAIL_MARGIN = 0.50`、
`MIX_TEST_PATTERN_END_X = 19.50`、`MIX_TEST_TILE_SIZE = (20.0, 4.0)`
（`CMoE_env_cfg.py`，注释里带着上面的算式）。

对照（第四批在 8 m 道上的同一个式子）：`(8.00 − 0.30 − 0.50)/3.20 = 2.25` ⇒ 图案末端 7.50 m。
**两批只差 `size[0]` 一个数**，其余（`pattern_start_x`、`x_unit`、尾部余量、图案本身）一字未动。

### 6.2 铺平地规则：`fill_stretched_gaps`

单靠乘子**不够**：`mix` 瓦片的**整宽底面本身就在 `-pit_depth = -0.50 m`**，而走道
（`_corridor`，宽 0.80 m）**只铺在 `segments` 那几段上** ⇒ 把图案拉开后，段与段之间新空出来的 X 区间
由原来的坑底填充，表现为**更长/更多的整宽深坑**（第二批就是这么"更难"的，见 §6.3 的对照表）。
第四批为此新增 `fill_stretched_gaps`：

* `False`（**默认**）＝ 上面的旧行为，**与改动前逐位相同**（训练侧不传该字段）；
* `True` ＝ 在 `scale > 1` 时，把**因拉开而多出来的空档**铺成 `height=0` 的可走面 `_corridor`。
  **⚠️ 粒度在第六批由"每一段"改成了"每个障碍组"**（根因与教训见 §6.6）—— 现行规则是：
  * **组内（首尾相接的连续段）整组共用一个平移量** ⇒ 段与段仍然首尾相接、宽度/顶面不变
    （⇒ 4 级楼梯连成楼梯；旧规则按段各自平移会把楼梯拆成 4 个孤立小凸块）；
  * **组间**：可拉伸空档 = **原始 units 上本来就有空档的组间边界**（本图案两处：`60 → (72−k)`、
    `111 → (123−k)`，`k = round(10 − 10d)` ⇒ `d = 0.70` 时各 0.18 m）＋ **尾段**（最后一段末端 →
    图案末端索引 160）。每个空档里：**原有坑宽原样保留**（只由图案与难度决定，**与 scale 无关**）
    且**紧贴下游组起点**（第六批追加：补出的平地铺在**坑的上游** ⇒ "从高处下来先有平地助跑、
    再遇到坑"）；因乘子多出来的总长 `160·x_unit·(scale−1)` 在这些空档之间**均匀分配**并铺平
    （`d = 0.70, scale = 6.00`：每处 5.3333 m，坑前 5.3333 m、坑后 0.00 m，尾段 `[14.1667, 19.50]`）；
  * **领先段**（`0 → pattern_start_x = 0.30`）与**原有尾廊**（图案末端 → `size[0]`）本来就在 0 高度，
    不需要额外处理；`scale = 1.0` 时没有任何可拉余量（补块列表为空）⇒ **开不开这个开关几何完全相同**；
  * ⇒ **障碍自身几何（每块宽度/顶面高度/顺序）与坑宽一字不变**，被拉开的只是"障碍之间的平地"。

  📌 **第四批原文（"余下余量铺在坑之后、坑紧贴上游戏块末端"）只在 `fill_stretched_gaps=False` 的
  逐段路径上有意义**；第六批之后 `True` 走上面这套"分组 ＋ 坑前平地"规则。**历史读数**（第四/五批
  逐段铺平地，`scale = 6.00`）：第一处坑 `[6.30, 6.48]`、出生点前方实心 5.55 m、楼梯被 0.60 m 平地
  拆开 —— 这些数字已作废，仅作对照。

### 6.3 真几何实测：`scale = 1.0` vs 占满值 `6.00`（`d = 0.70`、`fill_stretched_gaps=True`、20 m 道）

| scale | 图案末端 | 尾部平地 | **图案区间内**可走面 | 坑总长 | 补出的平地 | **整片**可走面 |
|---|---|---|---|---|---|---|
| 1.0 | 3.50 m | 16.50 m | 2.84 m | 0.36 m | 0.00 m | 19.64 m |
| 1.5 | 5.10 m | 14.90 m | 4.44 m | 0.36 m | 1.60 m | 19.64 m |
| 2.0 | 6.70 m | 13.30 m | 6.04 m | 0.36 m | 3.20 m | 19.64 m |
| **6.00（本场景）** | **19.50 m** | **0.50 m** | **18.84 m** | **0.36 m** | **16.00 m** | **19.64 m** |

* **坑总长恒为 0.36 m**（两处 0.18 m）⇒ 乘子**不会**再造深坑；补出的平地段总长 = `160·x_unit·(scale − 1)`
  = `3.2·(scale − 1)`；"图案区间内的可走面"随 scale **单调增**（2.84 → 18.84 m）。
* ⚠️ **一处与字面要求的冲突（如实报告）**：**整块瓦片**的可走面总长**恒为 19.64 m，不随 scale 变**
  —— 因为 `可走面 + 坑 = size[0]` 是恒等式，而坑总长恒定；图案越长，尾廊就越短（16.50 → 0.50 m），
  拉伸量被尾廊吸收。因此"**可走面总长随 scale 单调增**"只能对**图案区间内**（或"补出的平地段"）成立，
  本文与测试按后者表述（`TestMixTerrainGeometry.test_fill_mode_walkable_grows_in_the_pattern_and_pits_stay_constant`）。
* **对照：`fill_stretched_gaps=False`（第二/三批的旧行为）** —— 被拉开的空间由整宽 `-0.50 m` 坑底填充
  （20 m 道；8 m 道上同一乘子的读数见 §15 的历史行）：

  | scale | 可走面总长 | 坑总长 | 图案末端 | 尾廊 |
  |---|---|---|---|---|
  | 1.0 | 19.64 m | 0.36 m | 3.50 m | 3.50 → 20.00 m |
  | 1.5 | 18.04 m | 1.96 m | 5.10 m | 5.10 → 20.00 m |
  | 2.0（第二批用过） | 16.44 m | 3.56 m | 6.70 m | 6.70 → 20.00 m |
  | 6.00（本场景若关掉 fill） | 3.64 m | 16.36 m | 19.50 m | 19.50 → 20.00 m |

  即：不补空档时"乘子越大 ⇒ 深坑越多越长 ⇒ 地形越难"，与"更从容"的意图相反（第二批的结论）。
  第四批的 `fill_stretched_gaps=True` 是**用户当时列出的三条候选改法中的方案①**（"把段与段之间空出来
  的区间补成 `height=0` 的走廊"），方案②/③（只对本来相接的段铺平地／把乘子降到 1.5）本方案没有采用
  （前者会保留一条"原生坑仍在间隙前端"的歧义版本，后者达不到"占满整条道"）。
  第五批沿用方案①：**只放大瓦片、只拉间距**；**第六批把方案①的粒度改成「每个连续段组」、并把补出的平地改铺在坑的上游**（第五批的逐段粒度会把 4 级楼梯拆散，见 §6.6.1 的教训）。

### 6.4 障碍自身的尺寸与 scale 无关（逐块核对）

`d = 0.70`、任意 `scale ∈ {1.0, 1.25, 1.5, 2.0, 6.00}`：

| 指标 | 值 | 说明 |
|---|---|---|
| 图案段数 | 12 | 与 scale 无关（不新增/不丢图案）—— 即"**障碍数量不变**" |
| 高栏（170 索引）顶面 | **0.2618 m** | `= 170 × 0.002 × 1.1 × 0.70`，`TestMixTerrainGeometry` 在 4 个 scale 上逐一断言 |
| 第一处原图案坑宽 | **0.18 m** | `= (72−3−60) × 0.02`（`k = round(10 − 10×0.70) = 3`） |
| 每段宽度 / 顶面高度 | 与 `scale = 1.0` 逐块相等 | 只有 `x0` 后移（`test_obstacle_geometry_is_scale_independent`） |
| 抬升台阶顶面（d=0.70） | 0.0462 / 0.0924 / 0.1386 / 0.1848 m | 与第二/三批记录一致 |
| 坑深 / 走廊宽 | −0.50 m / 0.80 m（`pit_depth` / `corridor_width`） | 逐字沿用训练默认值，与 scale、与 `size` 都无关 |

### 6.5 单局时长：20 m 道 × 1.0 m/s ⇒ 20 s，实测配置取 35 s

* **现值核对**：CMoE/rough 链路里**没有任何地方**写 `episode_length_s` ⇒ 一路继承到
  `LocomotionVelocityRoughEnvCfg`（`velocity_env_cfg.py:716`）的 **`self.episode_length_s = 20.0`**。
  步长 `step_dt = decimation × sim.dt = 4 × 0.005 = 0.02 s` ⇒ 单局 **1000 步**。
* **为什么不够**：道 20 m、恒定 1.0 m/s ⇒ 走完需要 **20 s**。20 s 的预算**刚好卡在"走完就到时"**，
  没有任何余量（起步、被坑/高栏拖慢后追赶、PD 横向修正都要时间）。
* **处理（按用户要求）**：`< 25 s` ⇒ 在**评测 cfg**里提到 **35 s**（`MIX_TEST_EPISODE_LENGTH_S = 35.0`，
  `Imgo2CMoEMixTestEnvCfg.__post_init__` 里赋值；常量旁写明 `MIX_TEST_EPISODE_LENGTH_MIN_S = 25.0`
  与 `MIX_TEST_FORWARD_SPEED = 1.0`）。算式：`20 m ÷ 1.0 m/s = 20 s`，再留 **15 s 余量** ⇒ 35 s。
* **副作用（用户已知）**：回合变长 ⇒ **单轮更慢**；按固定步数 dump/采样数据的工具需要把**步数上限
  相应调大**（单局步数 `1000 → 1750` 步 @ `step_dt = 0.02 s`）。
* **训练侧一字未改**：`Imgo2CMoERoughEnvCfg` 与 `Imgo2CMoERoughPlayEnvCfg` 的 `__post_init__` 里
  **都没有** `episode_length_s` 赋值（测试 `test_training_and_play_episode_length_are_untouched` +
  审计脚本的"训练侧未覆盖"打印一起钉住）⇒ 训练/play 仍是 **20 s**。

### 6.6 第六批：分组语义（组内连续、组间拉大、平地铺在坑的上游）

#### 6.6.1 教训：**上一批按"每一段"补平地会把楼梯拆散**

用户实测："**台阶似乎只有一级**"。函数级定位（桩 `isaaclab` ＋ 真 `trimesh` 真跑）：

* `segments` 里那段楼梯是 **4 个首尾相接的段**：`30:36 / 36:42 / 42:48 / 48:60`，每级只有
  `6 × 0.02 = 0.12 m` 宽、顶面 0.0462 / 0.0924 / 0.1386 / 0.1848 m；
* 第四/五批的 fill 实现按**每一段之间**铺 `height=0` 平地，而逐段公式给每段**各按自己的起点**加
  `(scale−1)·start·x_unit` ⇒ 相邻两级的 X 间距被撑成 `(scale−1)·6·x_unit`，`scale = 6.00` 时是
  **0.60 m**，正好被补平地规则铺成平地；
* 真 trimesh 实测（`scale = 6.00`、`fill_stretched_gaps=True`、第四/五批代码）：
  `[3.90,4.02] 0.0462 / [4.62,4.74] 0.0924 / [5.34,5.46] 0.1386 / [6.06,6.30] 0.1848`
  —— 相邻级之间全是平地 ⇒ **看起来只有一级**。
* **教训**：把"间距乘子"的粒度与"图案的语义单元"绑定。楼梯/平台这类**首尾相接的连续段**
  在语义上是一个障碍组，**任何按段施加的位移都会把它拆开**；粒度必须取"连续段组"。

#### 6.6.2 分组结果（`d = 0.70`，真源码 `segments` 表）

| 组 | 原始 units 区间 | 段数 | 组内段 | 组内是否首尾相接 |
|---|---|---|---|---|
| ① | `0 → 60` | 5 | `0:30`(h=0)、`30:36`、`36:42`、`42:48`、`48:60` | ✅（`30 == 30`） |
| ② | `69 → 111` | 5 | `69:84`、`84:86`、`86:96`、`96:99`、`99:111` | ✅ |
| ③ | `120 → 160` | 2 | `120:140`、`140:160` | ✅ |

* **实测 3 组**（没有空段）：把 12 段按"原始 units 上是否首尾相接"切开即是。
  ⚠️ **与用户预期"5 组"的差异（如实记录）**：用户把 `0..30` 起步段与 `30..60` 楼梯组分开数；
  但 `0:30` 与 `30:36` 在 `30` 处**首尾相接** ⇒ 按同一条定义属**同一组**。把它俩人为拆成两组
  **不改变任何几何**：二者之间的原始空档宽度恒为 **0** ⇒ 拿不到额外长度、平移量相同
  （测试 `test_first_obstacle_group_is_bitwise_scale_invariant` 锁定"前 7 块与 `scale = 1.0` 逐位相同"）。
* 组内**整组共用一个平移量** ⇒ 段与段仍首尾相接、段宽与顶面不变 ⇒ **4 级楼梯连成楼梯**
  （真 trimesh，`scale = 6.00`）：

  | 级 | X 区间 (m) | 顶面 (m) | 与上一级 |
  |---|---|---|---|
  | 1 | `[0.90, 1.02]` | 0.0462 | — |
  | 2 | `[1.02, 1.14]` | 0.0924 | 严丝合缝（级间**无平地**） |
  | 3 | `[1.14, 1.26]` | 0.1386 | 同上 |
  | 4 | `[1.26, 1.50]` | 0.1848 | 同上 |

#### 6.6.3 组间：把多出来的长度均匀分给 3 个空档，坑宽不变、平地铺在坑的上游

* **空档数 = 3**：两处坑所在空档（`60 → 69`、`111 → 120`，各 `9 units = 0.18 m`）＋ 尾段
  （最后一段末端 `160` → 图案末端 `160`，原始长度 0）。
* **总拉伸量** = `160 · x_unit · (scale − 1) = 3.2 · (scale − 1)`（`scale = 6.00` ⇒ **16.00 m**），
  **均匀分配**：每处 **5.3333 m**。
  * **为什么选"均匀"而不是"按原始空档宽度占比"**：两处坑所在空档的原始宽度各只有 **0.18 m**，
    按占比分配等于把 16 m 几乎全塞进这两处（各 ~8 m）、尾段分到 0 ⇒ 长距离平地被挤在两处坑附近、
    最后一组贴着图案末端。均匀分配让三处空档各承担等长平地，行程更均匀，且同样满足"坑宽不变 ＋
    图案末端 19.50 m"。
* **坑宽恒 0.18 m**（两处合计 0.36 m）：`= (next.start − prev.end) · x_unit`，只由图案与难度决定。
* **追加修正（本轮第二段要求）**：用户追加"我似乎有看到一个**较高的台子后紧接 gap**" ⇒ 把补出的平地
  改铺在**坑的上游**、**坑紧贴下游组起点**（原先坑紧贴**上游**组末端）：
  * `scale = 6.00` 实测：第一处坑前平地 `1.50 → 6.8333`（**5.3333 m**）、坑 `[6.8333, 7.0133]`，
    坑后**直接**是下游组第一块（0.1848 m）⇒ 落点平地 **0.00 m**；
  * 第二处同理：坑前平地 `7.8533 → 13.1867`（5.3333 m）、坑 `[13.1867, 13.3667]`；
  * ⇒ "从高处（楼梯顶 0.1848 m）下来"先有 5.3333 m 平地助跑，再遇到 0.18 m 的坑，而不是
    "刚下楼梯就踩空"；坑后紧接下游抬高块，与 `scale = 1.0` 的原始图案语义一致。
* **坑的位置与 scale 的关系**：坑宽恒 0.18 m；**第一处坑的位置**由"上游组末端 1.50 m ＋ 该空档分到的平地"
  决定（`scale = 1.0` ⇒ `[1.50, 1.68]`；`scale = 6.00` ⇒ `[6.8333, 7.0133]`）。
* **总长不变式（20 m 道）**：图案末端仍 `19.50 m`；整片可走面 `19.64 = 20 − 0.36`；补出平地 16.00 m；
  图案区间内可走面 18.84 m（与 §6.3 完全一致 —— 本批只改**图案区间内部**的分布，不改任何总量）。

#### 6.6.4 默认路径、溢出保护与其它「难组合」（后者**本轮只报告，不改**）

* **默认路径逐位不变（硬要求）**：`fill_stretched_gaps=False`（默认，训练侧那条）**仍走第四批的逐段公式**
  （多出来的空间由整宽 `-0.50 m` 坑底填充），`scale = 1.0` 时 `shift` 精确为 `+0.0`；分组布局只在
  `fill_stretched_gaps=True` **且** `scale > 1` 时启用。测试逐块锁：`scale=1.0 × d ∈ {0,0.35,0.70,0.75,1.0}
  × size ∈ {(8,4),(8,8)}`（10 组）＋ `scale=6.00, fill=False`（旧路径）。
* **溢出保护不放宽**：仍按 `pattern_start_x + 160·x_unit·scale ≤ size[0]` 判、超限 `ValueError`；
  该瓦片上限仍是 `(20 − 0.30)/(160×0.02) = 6.15625`，`size[0]=8 m` 的训练侧仍是 `2.40625`。
* **图案里其它"难组合"（函数级读数；本轮只报告）**：

  | # | 难组合 | 实测 | 结论 |
  |---|---|---|---|
  | ① | `84:86` 的 4 cm 窄凹口（`height=0`、宽 `2 units = 0.04 m`）夹在 0.1848 m（`69:84`）与 0.1478 m（`86:96`）两块之间 | 凹口比足端尺度还小 | 可能**卡脚**；本轮不改（要改就得动图案本身） |
  | ② | 最高块 `96:99`（`height=170` ⇒ **0.2618 m**）只有 `3 units = 0.06 m` 长，前接 0.1478 m、后接 0.1848 m | 上-下**尖峰**，横向尺度 6 cm | 冲击大；本轮不改 |
  | ③ | 末尾抬高段 `140:160`（0.0924 m）之后直接落到 0（尾段平地） | 落差 **0.0924 m** | 下台阶，属预期；本轮不改 |

  这三条由 `check_terrain_columns.py --task mix-test` 的"图案里其它「难组合」（只报告、本轮不改）"
  段落**每次实跑打印**，也在 `test_hard_combinations_are_report_only` 里钉住（只断言"报告了"，
  不断言几何该怎样）。

## 7. 出生点真几何与**可走面/坑 X 区间实测表**

用**桩 `isaaclab` ＋ 真 `trimesh`** 真跑 `track_mix_terrain`，把返回的网格摊成"沿 X 的可走区间/坑区间"，
再取 `spawn_x = 0.75` 的读数（**第五批在 20 m 道上重跑**：`d = 0.70`、`pattern_start_x = 0.30`、
`size = (20, 4)`、`pattern_spacing_scale = 6.00`、`fill_stretched_gaps = True`）。

### 7.1 第四/五批（`fill_stretched_gaps=True`，20 m 道）：出生点安全

| scale | 出生点落在 | 该平台顶面 | 到第一级台阶 | **到第一处坑（前方实心）** | 第一处坑 | 坑总长 | 整片可走面 |
|---|---|---|---|---|---|---|---|
| 1.0 | `[0.30, 0.90]`（0.60 m 起步段） | **0.000 m**（可走） | 0.15 m | 0.75 m ⚠️ | `[1.50, 1.68]`（宽 0.18 m） | 0.36 m | 19.64 m |
| **6.00（本场景）** | `[0.30, 0.90]` | 0.000 m | 0.15 m | **6.08 m** ✅ | `[6.8333, 7.0133]`（宽 0.18 m） | 0.36 m | 19.64 m |

* **第六批读数变化（如实记录）**：第四/五批的"逐段"铺平地会把 `0→30` 起步块与 `30→60` 楼梯之间的空档
  也铺平、并把楼梯整体推后 ⇒ 那时 `scale = 6.00` 的"到第一处坑"是 **5.55 m**（第一处坑 `[6.30, 6.48]`）。
  第六批改成"组内连续"后，**起步块与 4 级楼梯同属第 1 组**（`0:30` 与 `30:36` 在 units 上首尾相接）
  ⇒ 前 7 块与 `scale = 1.0` **逐位相同**；多出来的长度铺在**坑的上游**（§6.6.3 的追加修正）⇒ 第一处坑
  落到 `1.50 + 5.3333 = 6.8333 m`、"到第一处坑"= **6.08 m** ⇒ **第四批自定义的"前方实心 ≥ 1.0 m"
  继续成立**（这个数只由"平地铺在哪一侧 ＋ 该空档分到多少"决定）。
* ⚠️ **出生点到第一级台阶只有 0.15 m**（`spawn_x = 0.75` → 台阶左沿 `0.90`）：该数字**与 `scale = 1.0`
  （＝训练几何本身）完全一致**（起步块 `0:30` 只有 0.60 m），不是第六批新引入的隐患；训练侧同类隐患见 §8 / CMOE-17。
* 规格要求"`spawn_x = 0.75` 落在实心可走面上、前方实心 ≥ 1.0 m，覆盖 `scale ∈ {1.0, 反算值}`"：
  **占满值 6.00 满足（6.08 m）**；**`scale = 1.0` 不满足（只有 0.75 m）** —— 该数字只由图案与
  `pattern_start_x` 决定（第一处坑恒在 `0.30 + 60×0.02 = 1.50 m`），`scale = 1.0` 时没有任何可拉余量、
  补块列表为空 ⇒ **`fill_stretched_gaps` 对它没有任何影响**。要把它推到 ≥ 1.0 m 就得改
  `pattern_start_x`/图案本身 ⇒ 会破坏"默认路径逐位不变"的硬要求 ⇒ **本轮只记录，未做**（见 §14）。
* ⚠️ **与用户预期的一处历史差异（第五批读数，第六批已随"平地铺在坑上游"改变）**：第五批用户预期
  20 m 道下"前方实心地面 ~8 m 级"，当时实测 **5.55 m**（坑紧贴**上游**块末端 ⇒ 第一处坑
  `0.30 + 60×0.02 + (6.00−1)×48×0.02 = 6.30 m`）；第六批把平地改铺在**坑的上游**后实测
  **6.08 m**（第一处坑 `6.8333 m`）。两者都远大于规格的 1.0 m，但**都不等于 8 m** —— 以实测为准。
* 对照（同一 `scale = 6.00` 但**不补空档**）：第一处"坑"紧贴起步平台末端 `[0.90, …]` ⇒ 前方只剩
  **0.15 m**（小于机身约 0.2 m 的半长 ⇒ 出生瞬间前足已探进 0.50 m 深坑），整片可走面只有 **3.64 m**、
  坑总长 **16.36 m**。这就是第二批/第三批记录的"出生点在一个很大的 gap 上"；
  **补空档同时把这个问题一并解决了**。

### 7.2 真 trimesh 实测的 X 区间表（20 m 道；`scale = 1.0` 对照 `scale = 6.00`）

`(x0, x1, 顶面高度)`，单位 m；`pits` 是相邻可走块之间**空出来**的区间（坑底 −0.50 m）。
**这是第六批用真 trimesh 在 20 m 道上重跑的读数**（不是从旧 8 m/旧逐段表抄的；`scale = 1.0` 一侧与第五批逐字相同）：

| — | `scale = 1.0`（训练几何 / 对照） | `scale = 6.00`（本场景，占满整条 20 m 道） |
|---|---|---|
| 起步平台 + 4 级楼梯（**前 7 块与 `scale = 1.0` 逐位相同**） | `[0.00, 0.30]ᵖ`、`[0.30, 0.90]`、`[0.90, 1.02]`、`[1.02, 1.14]`、`[1.14, 1.26]`、`[1.26, 1.50]` | `[0.00, 0.30]ᵖ`、`[0.30, 0.90]`、`[0.90, 1.02]`、`[1.02, 1.14]`、`[1.14, 1.26]`、`[1.26, 1.50]`（**楼梯 4 级连续、级间无平地**） |
| 坑①前的助跑平地 | —（`scale = 1.0` 无拉伸量） | **`[1.50, 6.8333]`（5.3333 m，铺在坑的上游）** |
| 坑① | `[1.50, 1.68]`（宽 0.18） | **`[6.8333, 7.0133]`**（宽 0.18，**紧贴下游组**） |
| 组②（高台/平台/高栏/平台） | `[1.68, 1.98]`、`[1.98, 2.02]`、`[2.02, 2.22]`、`[2.22, 2.28]`、`[2.28, 2.52]` | `[7.0133, 7.3133]`、`[7.3133, 7.3533]`、`[7.3533, 7.5533]`、`[7.5533, 7.6133]`、`[7.6133, 7.8533]` |
| 坑②前的助跑平地 | — | **`[7.8533, 13.1867]`（5.3333 m）** |
| 坑② | `[2.52, 2.70]`（宽 0.18） | **`[13.1867, 13.3667]`**（宽 0.18） |
| 组③ + 尾段 | `[2.70, 3.10]`、`[3.10, 3.50]`、`[3.50, 20.00]` | `[13.3667, 13.7667]`、`[13.7667, 14.1667]`、`[14.1667, 19.50]`（补出 5.3333 m 尾段）、`[19.50, 20.00]`（原尾廊 0.50 m） |

* **可走面总长 19.64 m**（`= 20.00 − 坑总长 0.36`）；**坑总长恒 0.36 m**（两处 0.18 m）；
  **补出的平地总长 16.00 m**（`= 3.2 × (6.00 − 1)`，三处各 5.3333 m）。`scale = 1.0` 一侧 15 块网格
  （补块列表为空）、`scale = 6.00` 一侧 **18** 块（多出 3 块 `height=0` 的补块；第五批是 27 块 —— 逐段
  粒度会把每一段都拆开成"段 ＋ 补块"）。
* 顶面高度序列（两列完全相同，与 scale 无关）：
  `0.000 / 0.0462 / 0.0924 / 0.1386 / 0.1848 / 0.1848 / 0.000 / 0.1478 / 0.2618 / 0.1848 / 0.1848 / 0.0924`
  —— 即**障碍自身尺寸/高度/顺序一字未变**，变的只是"障碍之间的平地放在哪一侧、有多长"。
* `ᵖ` = 起步走廊 `[0, pattern_start_x]`，其图案锚点恒为 0 ⇒ 与 scale 无关。
* 两张表都逐格钉在 `TestMixTestFilledSpawnGeometry.MEASURED` 与 `test_measured_x_interval_table` 里
  （20 m 道上重跑；8 m 道与第五批"逐段"表都是旧读数，已作废）。

### 7.3 第六批新增的分组/连续性实测（`d = 0.70`、`scale = 6.00`、真 trimesh）

| 检查 | 实测 | 判据 |
|---|---|---|
| 分组数 | **3**（`0→60`、`69→111`、`120→160`） | 与"按原始 units 首尾相接切"一致；组内逐段相接 ✅ |
| 楼梯 4 级 | `[0.90,1.02] 0.0462 / [1.02,1.14] 0.0924 / [1.14,1.26] 0.1386 / [1.26,1.50] 0.1848` | 相邻级 `x1 == 下一级 x0`（级间无平地）✅ |
| 坑宽 | 两处各 **0.18 m**（合计 0.36 m），任意 `scale ∈ {1.0,1.25,1.5,2.0,6.00}` 不变 | 坑 = `(next.start − prev.end)·x_unit` ✅ |
| 坑的位置 | `scale = 6.00`：`[6.8333, 7.0133]`、`[13.1867, 13.3667]`；各自**紧贴下游组起点** | 补块 `x1 == 坑 x0`（平地铺在**坑的上游**）✅ |
| 每处坑前平地 | **5.3333 m**（＝`16.00/3`，≥ 0.5 m 建议值） | 均匀分配的一份 ✅ |
| 坑后落点平地 | **0.00 m**（坑后直接是下游组第一块） | 由"坑紧贴下游组"直接推出 ✅ |
| 补块总数 / 总长 | 3 块 / **16.00 m**（`= 3.2·(scale−1)`） | 两处坑所在空档 ＋ 尾段 ✅ |
| 组 1 与 scale 无关 | `scale = 1.0` 与 `6.00` 的**前 7 块逐位相同** | 组 1 平移量恒为 0 ✅ |
| 图案末端 / 可走面 / 坑总长 | `19.50 m` / `19.64 m` / `0.36 m` | 与 §6.3 的不变式一致 ✅ |

对应的测试：`TestMixTerrainGeometry` 的 `test_segment_table_matches_the_reference`、
`test_grouping_is_implemented_in_the_source`、`test_contiguous_groups_are_three_and_include_the_staircase`、
`test_staircase_is_four_contiguous_levels_at_full_scale`、
`test_pit_width_is_constant_and_the_pit_abuts_the_downstream_group`、
`test_slack_is_distributed_evenly_between_the_two_pit_gaps_and_the_tail`、
`test_grouped_layout_matches_the_independent_reference`、
`test_first_obstacle_group_is_bitwise_scale_invariant`、
`test_fill_false_keeps_the_pre_change_per_segment_layout`，以及 checker 侧
`TestMixTestTerrainColumns` 的 4 条正向 ＋ 4 条负向对照。

对应的回归锁定：`TestMixTestFilledSpawnGeometry`（第四/五/六批：反算前提、两 scale 出生点安全、不补空档的
0.15 m 对照、上面两张区间表）、`TestMixTestSpawnGeometry`（第二/三批 `fill=False` 旧行为的回归对照）、
以及 `TestMixTerrainGeometry` 里对间距/补空档/分组语义的全部断言（第六批新增的 9 条见 §7.3）。

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
* **改道长（放大/缩小瓦片）**：`MIX_TEST_TILE_SIZE`（现值 `(20, 4)`；**只本评测 cfg 覆盖
  `terrain_generator.size`**，训练/play 仍是 `(8, 4)`）。**改了它必须同步重算乘子**（下一条），
  否则图案要么占不满、要么超长 raise；同时**单局时长**也要按 `size[0] / 1.0` 重核（§6.5）。
* **改间距 / 占满程度**：`MIX_TEST_PATTERN_SPACING_SCALE`（现值 **6.00**＝反算的"占满整条 20 m 道"值；
  上限 6.15625，超过会在 `track_mix_terrain` 里直接 raise）＋ `MIX_TEST_TAIL_MARGIN` /
  `MIX_TEST_PATTERN_END_X`（同一算式的两个中间量，只作文档/审计核对用，**不参与生成**）与
  `MIX_TEST_FILL_STRETCHED_GAPS`（现值 True；**必须与乘子一起改** —— 关掉它，拉开的空档会变回整宽
  −0.50 m 深坑、出生点前方实心地面会掉到 0.15 m）。改之前请读 §6.1–§6.3：乘子每 +0.05 ⇒ 图案末端
  +0.16 m、尾部平地 −0.16 m（`pattern_end = 0.30 + 3.2·scale`）。
* **改单局时长**：`MIX_TEST_EPISODE_LENGTH_S`（现值 **35 s**，下界常量 `MIX_TEST_EPISODE_LENGTH_MIN_S
  = 25 s`；只在评测 cfg 赋值，训练/play 不覆盖 ⇒ 仍是父类的 20 s）。判据：`size[0] ÷ 1.0` ＋ 余量。
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
* 出生点在世界 **X = −9.25 m**（= `spawn_x − 10`，第五批：单块瓦片 X = 20 m）、道中心 Y；
  世界 **X ∈ [−10, +10]、Y ∈ [−40, +40]**。
* **单局时长 35 s**（第五批）：道 20 m ÷ 1.0 m/s = 20 s ＋ 15 s 余量；按固定步数 dump 的工具
  要把步数上限调到 **1750 步**（`step_dt = 0.02 s`）。
* 也可用 `--force_expert K` 等既有 CMoE play 开关（本任务不引入新开关）。

## 13. 地形列校验：按任务区分

`scripts/tools/check_terrain_columns.py` 的原判据是"**所有被掩码引用的地形名都必须 ≥1 列**"。
本测试场景**只有一类地形**（第一~六批是 `mix`；**2026-10-05 第七批起是复合道 `composite`**）⇒ 掩码引用的
`boxes` / `gap` / `flat` / `mix` 在该场景**不存在**，引用它们的项
（`joint_mirror`、`feet_air_time`、`feet_height_body`、`feet_air_time_variance`、`feet_gait`、
`base_height_flat_l2`）在本场景**恒为 0**——这是**预期**，不是配置错误。

处理方式（用户要求"不削弱原判据"）：

* 新增 `--task`，默认 `cmoe-rough` **走原判据、一字未放宽**；
* `--task mix-test` 走**单独分支**：仍强制 `sub_terrains.clear()` + 只有该类地形 + **它占满 20 道** +
  `num_cols=20` + **`num_rows=1`** + **`difficulty_range=(0.70, 0.70)`**，并把"掩码引用但本场景不存在"
  的名字**逐条打印为预期**；
* 解析从"整文件 `ast.walk`"改成**类作用域**（只走指定类的 `__post_init__`）；
* `num_cols`/`num_rows`/难度写成**模块级可读常量**，工具的解析支持"顶层 `NAME = <字面量>` 常量求值"；
* **第四批新增**：工具额外记录 mix 调用的**全部 kwargs**（`sub_terrain_calls`）、真读
  `terrain_generator.size`（新函数 `terrain_size()`），并对 mix-test 增加三项检查 —— 乘子必须 =
  反算值、mix 调用必须显式 **`fill_stretched_gaps=True`**、图案末端
  `pattern_start_x + 160·x_unit·scale` 必须 **≥ 目标下界** 且 **≤ `size[0]`**（并打印乘子上限）；
* **第五批新增**：① `terrain_size()` 由"全文件第一个匹配"改成**按类作用域**（原来会先撞到训练类的
  `(8, 4)`）⇒ `terrain_size(MIX_TEST_CLASS) = (20, 4)`、`terrain_size(TRAIN_CLASS) = (8, 4)`；
  ② mix-test 分支新增两组检查：**评测 cfg 的 size 必须是 (20, 4)** 且 **训练/play 的 size 必须仍是
  (8, 4)**（`size` 是共享字段，只有评测 cfg 能改）；**单局时长必须 ≥ 25 s**（新增 `episode_length_s()`
  读取器）；③ 期望值改 **乘子 6.00 / 目标下界 19.0 m / 上限 6.15625 / 评测 size (20,4) / 35 s**；
* **第七批新增（2026-10-05）**：地形换成复合道 ⇒ mix-test 分支的 mix 图案检查（第四/五/六批）**整体
  替换**为**复合道检查** —— 障碍清单 ＋ **X 区间表** ＋ **逐对相邻障碍间隔表**（打印
  `✅ 最小相邻间隔 2.3636 m ≥ 下界 1.5000 m`）＋ `✅ 楼梯上下紧贴 …` ＋
  `✅ 楼梯与相邻障碍间隔 = 2.3636 m ≥ 下界` ＋ 占满核算（最后障碍末端 17.6364 ＋ 尾段 2.3636 = 20.0000）
  ＋ 出生点（前方 1.5000 m），另加「训练侧 `mix` 未受影响」段落（默认 1.0/False、12 段图案仍 3 组、
  4 级楼梯仍连续）。几何复算**只用标准库**（`composite_layout()` 的算术与真生成器由测试交叉核对）。
  这一段是**新增**判据，原任务 `cmoe-rough` 的判据与输出**一字未改、一字未放宽**。
* 测试对 test 任务**单独断言**，并加**负向对照**（把"只有一类地形"丢给原判据必须仍报错、存在但 0 列也必须
  报错、**期望道数/行数/难度被改成别的值时必须报 ❌**、**复合道的障碍顺序/间隔/楼梯级数/末端期望值被改错
  ⇒ 报 ❌**、**`sub_terrains` 里没有 `composite` ⇒ 报 ❌**、**训练侧 `mix` 默认值被改 ⇒ 报 ❌**、
  **评测瓦片尺寸改错 ⇒ 报 ❌**、**训练侧尺寸被放大 ⇒ 报 ❌**、
  **单局时长下界抬高 / 评测 cfg 未覆盖时长 ⇒ 报 ❌**；另有生成器侧的"贴在一起"负向对照见 §16.8）
  证明判据没被放宽也没被绕过。

实测输出（`python3 scripts/tools/check_terrain_columns.py --task mix-test`，退出码 0）见 §15；
`--task cmoe-rough`（默认，原判据）的输出**逐字未变**。

## 14. 未验证项、与设计冲突之处、待决项

**未验证（本容器无 GPU / 无 Isaac 运行环境：`NVIDIA_VISIBLE_DEVICES=void`、`import isaaclab` 缺 `omni.log`）**：

1. **未构造 Isaac 环境**：`difficulty_range` 在真实 `TerrainGenerator` 上的产出、`InteractiveScene`
   的构造路径、`MixTestVelocityCommandCfg` 在真实 `CommandManager` 下的解析、`debug_vis=True` 在
   20 环境下的 marker 创建都未实测。**难度公式是源码级复核 + 离线复现，不是运行期实测**。
2. **未回放、未导出 `policy.pt`**：checkpoint 兼容性是**静态推理**（动作/观测契约未改）。
3. **PD 增益未标定**：`1.0/0.3/0.6`（横向）与 `1.5/0.3/1.0`（航向）**未在仿真里标定**。
4. **`terrain_types` 的"一道一个环境"未在真仿真里打印确认**（结论来自 Isaac Lab 源码公式，见 §4.1）。
5. **1 行 × 20 道 × 20 m 的真实地形网格未生成过**：几何结论来自**用桩 `isaaclab` + 真 `trimesh` 实跑**
   `track_mix_terrain`（函数级），不含 Isaac 的瓦片拼接、缓存与碰撞体；**20 m 单块的 mesh 规模
   （三角形数、PhysX 碰撞体）也未测**（8 m → 20 m 会放大瓦片面积，理论上要重新看一次生成时间与显存）。
6. **奖励掩码在本场景恒为 0 的实际影响未测**：6 项掩码项恒 0 ⇒ 评测时 `Episode_Reward/*` 里这些键的
   读数不能与训练 run 直接对比（只是读数问题，不是 bug）。
7. **一处已知的读数副作用（有意保留）**：基类 `_update_metrics` 用
   `max_command_step = resampling_time_range[1] / step_dt` 做归一化；本任务
   `resampling_time_range = (1e9, 1e9)` ⇒ `error_vel_xy` / `error_vel_yaw` 两个**度量**读数≈0
   （只进日志）。要可读的跟踪误差请用奖励分项 `Episode_Reward/track_world_vel_xy_exp`。
8. **35 s 的单局时长只是"够用"、未在仿真里标定**：结论是"20 m ÷ 1.0 m/s = 20 s ＋ 15 s 余量"的算术，
   **没有**在 Isaac 里跑满 35 s 看过终止分布/耗时；也没测过"20 m 道 + 27 块网格"下每步的实际耗时增幅。

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
   `height=0` 可走面，原有坑宽不变）；方案②/③ 未采用（见 §6.3 末段）。第五/六批沿用方案①。
5. **出生点问题只被"绕开"、没有被"修好"**（§7.1/§8）：评测场景靠"占满值 6.00 ＋ 分组铺平地（平地铺在
   坑的上游）"把到第一处坑的距离由 0.15 m 提到 **6.08 m**（第五批逐段粒度时是 5.55 m），但训练侧
   `spawn_x = 0.75` + `pose_range ±0.5` 的同类隐患**仍在**（问题表 **CMOE-17**，待决）；
   **`scale = 1.0`（＝训练几何本身）下到第一处坑仍只有 0.75 m、到第一级台阶只有 0.15 m**，结论不变。
6. **`track_gap_terrain` 的溢出保护仍未做**（问题表 CMOE-13）：本轮只保留 `track_mix_terrain` 的保护，
   没有顺手扩到 `track_gap_terrain`（超出本次范围）。
7. **`check_reward_overrides.py` 未加 `cmoe-mix-test` 链**：本场景**完全不改奖励**（生效项仍是 `cmoe`
   的 28 项），加一条与 `cmoe` 逐项相同的链没有信息量，故未加。
8. **第四/五批"与字面要求对不上"的地方（本轮只记录，未擅自扩范围）**：
   * **`scale = 1.0` 时"到第一处坑 ≥ 1.0 m"不成立**（实际 **0.75 m**）。规格要求"出生点安全……覆盖
     `scale ∈ {1.0, 反算值}`"，但 `scale = 1.0` 正是**训练几何本身**，没有任何可拉的余量（补块列表为空），
     第一处坑恒在 `0.30 + 60×0.02 = 1.50 m`。要满足它必须改 `pattern_start_x`/图案本身 ⇒ 会破坏
     "默认路径逐位不变"的硬要求 ⇒ 前五批**不动**；占满值 6.00 满足（第六批 **6.08 m**）。
   * **"前方实心地面 ~8 m 级"（第五批用户预期）与实测不符（第五批 5.55 m、第六批 6.08 m）**：
     第五批按逐段规则把坑紧贴**上游**块末端 ⇒ 第一处坑在 **6.30 m**；**第六批按用户追加反馈改成
     "平地铺在坑的上游"** ⇒ 第一处坑 **6.8333 m**、到第一处坑 **6.08 m**（6.08 而不是 ~8 的原因是
     16 m 拉伸量要**均分给 3 个空档**、这一处只拿到 5.3333 m）。两者都远大于规格的 1.0 m，但都
     **不等于 8 m** —— 以实测为准。
   * **"可走面总长随 scale 单调增"在整块瓦片上不成立**：`可走面 + 坑 = size[0]`（20 m），坑总长恒定
     0.36 m ⇒ 整片可走面**恒为 19.64 m**（尾廊吸收拉伸量：16.50 → 0.50 m）。随 scale 单调增的是
     **图案区间内的可走面**（2.84 → 18.84 m）与**补出的平地段**（0 → 16.00 m）——本文件、cfg 注释与测试
     都按后者表述，并把"整片恒定"作为不变式一并锁住（§6.3 的 ⚠️）。
   * **`size` 是 `terrain_generator` 的共享字段**：本仓**没有**"每个 sub_terrain 自己的尺寸"这种机制，
     所以"放大瓦片"只能写在 `terrain_generator` 上、并靠**类作用域覆盖**保证训练/play 不受影响
     （测试 + 审计双向钉住）。若以后新增别的评测场景也改 `size`，同样要显式覆盖并各自断言。
9. **尾段的归属**：本方案把"最后一段末端 → 图案末端（索引 160）"这段也视作"拉开的空档"并铺平
   （`scale = 6.00` 时是 `[14.1667, 19.50]` 这 5.3333 m）；参考图案在索引 140:160 之后本来就没有赋值 ⇒
   铺成 0 高度与参考语义一致（原来在 `scale = 1.0` 时这段是**尾廊**的一部分，也铺在 0 高度）。
10. **第六批（本轮）如实登记的四处**：
    * **分组数与用户预期不同**：用户预期"5 组"，按"原始 units 上是否首尾相接"实测 **3 组**
      （`0:30` 与 `30:36` 在 `30` 处相接 ⇒ 同组）。拆开也不改几何（该边界原始空档恒为 0）——
      本文与测试都按"3 组"记录，并在 §6.6.2 说明差异。
    * **出生点到第一级台阶只有 0.15 m**（`spawn_x = 0.75` → 台阶 `0.90`）：这是**训练几何本身**
      的读数（起步块 `0:30` = 0.60 m），第六批没有改它；要加大只能改图案/`pattern_start_x`，
      会破坏"默认路径逐位不变" ⇒ 本轮**不动**，训练侧隐患仍在 CMOE-17。
    * **坑前的平地长度随 scale 变化**：`scale = 1.25` 时只有 `0.2667 m`、`scale = 1.5` 时 `0.5333 m`，
      只有本场景的 `scale = 6.00` 才有 `5.3333 m`；"≥ 0.5 m"这条建议值作为**硬判据只对评测场景**
      成立（checker 与测试都只在 `scale = 6.00` 上断言），更小的 scale 只记录实际值。
    * **图案里其它"难组合"只报告不改**：4 cm 窄凹口（`84:86`）、0.2618 m 尖峰只有 0.06 m 长
      （`96:99`）、末尾 0.0924 m 落差（`140:160` → 尾段）—— 三者都是**参考图案自带**的结构，
      本轮除"坑前助跑"外**未改任何几何**（§6.6.4）。
11. **"坑紧贴哪一侧"已被第六批追加要求改过一次**：第四/五批是"坑紧贴**上游**块末端、平地在坑后"，
    第六批按用户追加反馈（"较高的台子后紧接 gap"）改成"平地铺在**坑的上游**、坑紧贴**下游**组起点"。
    这正是第五批 §14 里预留的那条改法（当时估计"第一处坑 ~8.40 m、前方实心 ~7.65 m"）—— 实测为
    **第一处坑 6.8333 m、前方实心 6.08 m**（因为平地还要与另两个空档**均分**，不是全给这一处）。

## 15. 验证与实测输出（本机离线）

```bash
cd imgo2_rl
/opt/conda/envs/isaaclab/bin/python -m unittest discover -s tests     # 709 通过 / 10 跳过（第七批后）
python3 scripts/tools/check_reward_overrides.py cmoe-gaitfree          # 24 项（与改前一致）
python3 scripts/tools/check_terrain_columns.py                         # 末行"全部掩码引用的地形名都有 ≥1 列 ✅"（逐字未变）
python3 scripts/tools/check_terrain_columns.py --task mix-test         # 见 §15.4（退出码 0）
git diff --check                                                       # 干净
```

**基线**：第五批后 `unittest discover -s tests` = **674 通过 / 10 跳过**；第六批后 = **690 / 10**
（`test_cmoe_mix_test_scene.py` 84 → **92**、`test_check_terrain_columns.py` 39 → **47**；共新增 **16** 项）；
**第七批后 = 709 / 10**（mix 场景 92 → **111**、列检查 47 → **48**；共新增 **19** 项）。

⚠️ 下面 §15.1–§15.3 的实测块是**第一~六批**（评测场景还是 `mix` 图案时）的记录；**第七批**改用复合道后的
实测输出见 **§15.4**。

`--task mix-test` 的实测输出（第五批；⚠️ 与"掩码引用但本场景不存在"的 11 条预期标注略去不重复）：

```
=== [Imgo2-basemove-rough-cmoe-mix-test] 受控测试场景（只 mix；cfg 类 Imgo2CMoEMixTestEnvCfg）===

=== Imgo2-basemove-rough-cmoe-mix-test｜只 mix（20 道并列）（num_cols=20，共 20 列）===
  mix                     20 列
  ✅ 唯一地形 mix 有 20 列（num_cols=20、num_rows=1）⇒ 网格 **20 道 × 1 难度行**（世界 20 m(X) × 80 m(Y)），难度由 difficulty_range=(0.7, 0.7) 精确固定 ⇒ `--num_envs ≤ 20` 时环境 i → 第 i 道

  ── 第五批检查：单块瓦片 `terrain_generator.size`（评测 20×4 m；训练 8×4 m）──
  ✅ 评测 cfg 的 `terrain_generator.size` = 20×4 m（第五批：X 由 8 m 放大到 20 m；Y 仍 4 m）
  ✅ **训练/play 侧**的 `terrain_generator.size` 仍是 8×4 m（共享字段未被评测场景带偏）

  ── 第四/五批检查：`mix` 图案占满整条道（瓦片 size[0] = 20 m）──
  ✅ `pattern_spacing_scale` = 6.0（反算值：(20 − 0.3 − 0.50)/(160×0.02)）＋ `fill_stretched_gaps=True` ⇒ 图案末端 = 0.3 + 160×0.02×6.0 = 19.50 m（占 97.50 % 的 20 m 道），尾部平地 0.50 m；乘子上限 (20−0.3)/(160×0.02) = 6.15625（溢出保护**不放宽**：超限直接 raise）

  ── 第六批检查：分组语义（组内连续、组间拉大、平地铺在坑的上游）──
  ✅ 分组：按原始 units 首尾相接切成 **3 组** —— 0→60、69→111、120→160（组内逐段首尾相接 ✅）
  ✅ 楼梯 4 级连续（同一组内 30:36 / 36:42 / 42:48 / 48:60，级间**无平地**）：[0.9000,1.0200] 0.0462 / [1.0200,1.1400] 0.0924 / [1.1400,1.2600] 0.1386 / [1.2600,1.5000] 0.1848
  ✅ 两处坑宽恒 0.18 m（合计 0.36 m）：[6.8333,7.0133]、[13.1867,13.3667]（各自**紧贴下游组起点** ⇒ 平地铺在坑的上游）
  ✅ 补出的平地 = 160×x_unit×(scale−1) 均匀分给 **3 个空档**：两处坑**上游**各 5.3333 m（建议 ≥ 0.5 m）、尾段 5.3333 m；坑后落点平地 **0.0000 m**（坑后直接是下游组第一块）
  ── 图案里其它「难组合」（**只报告、本轮不改**，供上层汇报）──
    ① 4 cm 窄凹口（0.04 m，位于 84:86）夹在 0.1848 m 与 0.1478 m 两块之间 ⇒ 可能卡脚（**只报告，本轮不改**）
    ② 最高块（96:99，顶面 0.2618 m）只有 0.06 m 长，前接 0.1478 m、后接 0.1848 m ⇒ 上-下尖峰（**只报告，本轮不改**）
    ③ 末尾抬高段（140:160）顶面 0.0924 m，之后直接落到 0 ⇒ 落差 0.0924 m（尾段平地，**只报告，本轮不改**）

  ── 第五批检查：单局时长（道 20 m ÷ 恒定 1 m/s = 20.0 s）──
  ✅ 评测 cfg 的 `episode_length_s` = 35 s ≥ 下界 25 s（走完 20 m 要 20.0 s，余量 15.0 s）
  ℹ️ 训练/play 链**未**覆盖 `episode_length_s`（沿用父类 20 s）—— 本批只改评测 cfg，训练时长一字未动
  ⚠️ 本场景**只有 mix**；掩码引用但本场景**不存在**的地形名（原任务判据**不放宽**，这些项在本场景恒为 0 属**预期**）：
    ...（共 11 条，逐条列出：boxes / gap / flat）
  ✅ `forward_only_terrain_names`（继承父类） 覆盖 mix；另含 10 个本场景不存在的地形名（...）

结论：test 任务只有 mix 一种地形；掩码引用的其它地形名在本场景恒为 0，属**预期** ✅
```

* **基线**：第五批改动前 `unittest discover -s tests` = **667 通过 / 10 跳过**；第五批后 = **674 / 10**
  （`test_cmoe_mix_test_scene.py` 82 → **84**、`TestMixTestTerrainColumns` 26 → **31**（整个
  `test_check_terrain_columns.py` 34 → **39**）；共新增 **7** 项）。
* **单局时长结论**：CMoE/rough 链路现值 = **20 s**（`velocity_env_cfg.py:716`，无人覆盖）< 25 s
  ⇒ 按用户要求只在**评测 cfg** 提到 **35 s**；训练/play 未覆盖、仍是 20 s（§6.5）。
* **"难度确实 = 0.70"的两条路径**：见 §5.1（公式复现 `places=15`；真跑 `track_mix_terrain` 核对
  栏高 0.2618 m、两处坑宽各 0.18 m 与第一处坑左沿 `6.8333 m`（第六批占满值；第五批是 `6.30 m`））；另一条源码级断言在已安装的
  Isaac Lab 里逐字核对 `difficulty_range` / 两条分支公式 / 本仓 `curriculum = True` 的开启条件
  （缺 Isaac Lab 时该组 skip）。
* **"默认路径逐位不变"（硬要求）**：`fill_stretched_gaps=False` 时对
  `d ∈ {0, 0.35, 0.70, 0.75, 1.0} × size ∈ {(8,4), (8,8)}` 共 **10 组**逐块比较 `mesh.bounds`
  ⇒ **10/10 逐位相同**；`fill_stretched_gaps=True` 但 `scale = 1.0` 时补块列表为空 ⇒ 与 `False`
  **完全相同**（两条都钉在 `TestMixTerrainGeometry.test_fill_flag_and_defaults_are_neutral` 与
  `test_default_spacing_reproduces_the_pre_change_geometry` 里）。**第五批 `track_mix_terrain` 一行未改**
  （只改了 `cfg.size`）⇒ 这条仍然成立。
* **"只拉间距、不多深坑"（20 m 道重跑，第六批后仍成立）**：`scale ∈ {1.0, 1.25, 1.5, 2.0, 6.00}` 上
  坑总长恒 **0.36 m**、图案区间内可走面 `2.84 → 18.84 m` 单调增、补出平地 = `3.2·(scale−1)`
  （`0 → 16.00 m`，第六批为** 3 块**各 `5.3333·(scale−1)`/4 m）、障碍逐块宽/高与 `scale = 1.0` 相等
  （12 段，按 `(宽度, 顶面)` 签名核对）、栏高恒 0.2618 m、第一处坑宽恒 0.18 m
  （`test_fill_mode_walkable_grows_in_the_pattern_and_pits_stay_constant`
  / `test_obstacle_geometry_is_scale_independent`）。
* **出生点真几何实测**：见 §7.1/§7.2/§7.3 三张表（**20 m 道**上重跑：`d = 0.70`、`scale = 1.0` 与占满值
  **6.00**；出生点仍在 `[0.30, 0.90]` 实心面上，到第一级台阶 **0.15 m**、到第一处坑
  **0.75 m（scale=1.0）→ 6.08 m（scale=6.00）**；第一处坑 `[1.50, 1.68] → [6.8333, 7.0133]`（平地铺在
  坑的上游）；读数由 `TestMixTestFilledSpawnGeometry` 逐格锁定；第二/三批 `fill=False` 的旧读数由
  `TestMixTestSpawnGeometry` 继续锁定）。
* **训练侧未受影响的证据**：① `Imgo2CMoERoughEnvCfg.__post_init__` 里仍是
  `self.scene.terrain.terrain_generator.size = (8.0, 4.0)`（测试 `test_tile_size_is_20_for_the_test_and_8_for_training`
  + 审计脚本的"训练/play 侧仍是 8×4 m ✅"）；② 两条链的 `__post_init__` 都没有 `episode_length_s` 赋值
  （测试 `test_training_and_play_episode_length_are_untouched` + 审计脚本"未覆盖"打印）；
  ③ `pattern_spacing_scale` 默认仍 **1.0**、`fill_stretched_gaps` 默认仍 **`False`**
  （`test_default_spacing_and_fill_stay_neutral_for_training`）⇒ **默认路径仍走第四批的逐段公式、
  与改动前逐位相同**（第六批新增的分组分支在该默认下**一行都不执行**）；④ `scale = 1.0` 时
  `shift` 精确为 `+0.0`、`fill=True` 也不产生任何补块（`test_fill_flag_and_defaults_are_neutral`）；
  ⑤ `scale = 6.00 + fill=False`（旧路径）逐块等于改动前公式（`_pattern_rows` 参照，
  `test_fill_false_keeps_the_pre_change_per_segment_layout`）。
* **溢出保护实测（第五批：20 m 瓦片）**：`scale=6.25` 抛
  `ValueError: track_mix_terrain: 图案总长 20.3000 m 超出瓦片长 size[0]=20.0000 m …本瓦片上
  pattern_spacing_scale 最大可取 6.1562…`；`scale=6.15625` 不抛（`fill_stretched_gaps=True` 时同样成立，
  见 `test_overflow_guard_is_not_relaxed_by_fill_mode`）。8 m 道的旧读数（`2.5` 抛、上限 `2.4062`）
  由 `test_mix_pattern_fits_the_tile` 里的对照断言保留。
* **"默认 1.0 无差异"（沿用第二批的验证方式）**：把 `b2caad1` 的 `cmoe_terrains.py` 与改后版本分别
  exec（桩 `isaaclab` + 真 `trimesh`），对 `d ∈ {0, 0.35, 0.70, 0.75, 1.0} × size ∈ {(8,4), (8,8)}`
  逐块比较 `mesh.bounds` 与 `origin`：**10/10 逐位相同**（参照公式表 `_pattern_rows` 就来自改动前，
  第四批新增的 fill 分支在 `fill_stretched_gaps=False` 时**一行都不执行**；第五批没有动这个函数）。

## 16. 第七批（2026-10-05）：地形改成**复合道**（放弃参考 `mix` 图案）

### 16.1 为什么放弃参考 mix 图案（用户原话 + 事实核对）

用户实测反馈（第七批，原话）：

> 「是有些障碍分到一起了，我说的是障碍本身。gap 和突台也在一起。我希望是和单独的那个几个场景
> 类似的障碍放在这个整条的（道）上」；追加「每个地形要间隔开，而不是混合到一起放在一个位置」。

事实核对（源码级）：参考实现 `parkour_terrain_utils.mix_obstacles_terrain` 是一张**硬编码的小图案**
（水平索引 × 0.05 m、高度索引 × 0.005 m，缩比 ×0.4 后 x 单位 0.02 m、z 单位 0.002 m），本仓照抄为
`track_mix_terrain` 的 12 段 `segments` 表。它的形状里**存在只有参考图案才有的结构**：

* 第一处坑 `60 → (72−k)` 与**紧邻其下游**的高台/高栏（`(72−k)` 起点）**贴在一起**（中间没有平地）；
* `84:86` 是一段 **0.04 m 长的窄凹口**，被两侧 0.1848 m / 0.1478 m 的抬高块夹住；
* `96:99` 是一段 **0.06 m 长、0.2618 m 高**的尖峰；
* 「坑 ↔ 抬高块」「窄槽 ↔ 抬高块」的间距只够一个脚掌，在视觉与落地力学上都属于**挤在一起**。

这些结构在任何"独立地形"（`gap`／`boxes`／`pyramid_stairs`／`hurdle`）里都**不存在** —— 它们是
参考图案为 G1（骨盆 0.75 m）设计的复合动作序列。用户要的是"**把各个独立地形的障碍各自成形、
按顺序铺在整条道上、彼此留开**"，因此本轮**不再使用**这套图案（字段与实现保留给训练侧）。

### 16.2 设计：一条 20 m 走廊上的 5 个独立障碍 ＋ 5 段等长平地

`cmoe_terrains.track_composite_terrain(difficulty, cfg)`（`CMoETrackCompositeTerrainCfg`）：

```
[整宽基座（顶面 −pit_depth = −0.50 m）]
出生平地 [0, first_obstacle_x]                       顶面 0
├ ① 坑（横向沟）                                     **不铺走廊** ⇒ 露出 −0.50 m 基座（＝坑底）
├ 平地间隔
├ ② 楼梯：上 4 级 + 下 4 级**首尾相接**（金字塔形）    逐级顶面 (1..4)·h、(3..0)·h
├ 平地间隔
├ ③ 箱子/台阶块：单元内平地 ＋ 2 块抬高块
├ 平地间隔
├ ④ 栏：单元内平地 ＋ 2 道薄横栏（**栏周围是 z=0 平地**，不是深坑）
├ 平地间隔
├ ⑤ 坑（横向沟，首尾各一）
└ 尾平地 → size[0]
```

* **障碍顺序**是模块级常量 `COMPOSITE_OBSTACLE_SEQUENCE = ("gap", "stairs", "boxes", "hurdle", "gap")`
  ⇒"按类型分组摆放、不把不同类型混在同一小段里"由构造保证；
* **每个障碍用它自己那类地形的难度律与几何**（`_composite_obstacle_units`，见 §16.5）——
  本函数**不新造任何尺寸**；
* 走廊宽 `corridor_width = 0.80 m`、坑底 `pit_depth = 0.50 m`、`spawn_x = 0.75`：与既有评测场景
  完全一致（**没有擅自改**）；
* **所有相邻障碍之间都是同一个 `obstacle_spacing`**（见 §16.3）⇒ 用户硬约束
  "任意两个相邻障碍之间都必须有 ≥ `obstacle_spacing` 的平地"由构造保证，而不是靠事后检查；
* 入口：`spawn_x = 0.75` 落在 `[0, 2.25]` 的**实心平地**上（顶面 0），**前方 1.50 m** 才是第一个障碍；
* 放不下时 `composite_track_layout` **直接 `ValueError`**（三选一：减障碍／增大 `size[0]`／降
  `min_obstacle_spacing`），**保护不放宽、不静默截断**。

### 16.3 间隔反算（"把余量按障碍个数均分"）

每个障碍之后各分到一段平地，**最后一段就是尾段** ⇒ 整条道刚好铺满：

```
spacing = (size[0] − first_obstacle_x − Σ 障碍宽) / N          # N ＝ 障碍个数 = 5
first_obstacle_x = spawn_x + spawn_clearance = 0.75 + 1.50 = 2.25 m
Σ 障碍宽（d = 0.70）= 0.26（坑）+ 2.40（楼梯 2×4×0.30）+ 2.18（箱子 2×0.44+1.30）
                     + 0.832（栏 2×0.096+0.64）+ 0.26（坑）= 5.932 m
spacing = (20.00 − 2.25 − 5.932) / 5 = 11.818 / 5 = **2.3636 m**   （≥ 下界 1.50 m，落在建议区间 1.5~2.5）
尾段结束 = 2.25 + 5.932 + 5 × 2.3636 = **20.0000 m**
```

* 若把尾段单独留 0.5 m 而只让**内部** 4 段平分，间隔会涨到 `(20 − 2.25 − 5.932 − 0.5)/4 = 2.8695 m`，
  **超出建议区间上限 2.5 m** ⇒ 采用"尾段也算一段"的均匀方案（用户要求"均匀"）。
* `obstacle_spacing` 字段的语义：**负值（默认 `COMPOSITE_SPACING_AUTO = −1.0`）＝自动反算**；
  **非负值＝显式指定**，且必须 `≥ min_obstacle_spacing`（默认 1.50 m），否则直接 `ValueError`
  ⇒"把 `obstacle_spacing` 设 0（＝两个障碍贴在一起）"是一个**会报错**的构造（负向对照见 §16.8）。
* 该瓦片能放下的**最低** `size[0]` = `2.25 + 5.932 + 5 × 1.50 = 15.682 m`；8 m 瓦片必然 `ValueError`
  （负向对照用它构造）。

### 16.4 障碍清单与 X 区间表（真 trimesh 实测，`d = 0.70`、`size = (20, 4)`）

| # | [x0, x1]（m） | 种类 | 尺寸（d = 0.70） | 之后平地 |
|---|---|---|---|---|
| — | [0.0000, 2.2500] | 出生平地 | 长 2.2500 m（顶面 0） | — |
| ① | [2.2500, 2.5100] | **坑**（横向沟） | 宽 **0.2600** = `0.12 + 0.70×0.20` | 2.3636 m |
| ② | [4.8736, 7.2736] | **楼梯**（上 4 级 + 下 4 级**紧贴**） | 级高 **0.1550**、步深 **0.30**、段长 **2.4000**、峰高 **0.6200**（8 级） | 2.3636 m |
| ③ | [9.6372, 11.8172] | **箱子/台阶块**（2 块） | 块高 **0.2340**、块长 **0.4400**、**单元内**间距 1.3000 | 2.3636 m |
| ④ | [14.1808, 15.0128] | **栏**（2 道薄横栏） | 栏高 **0.0930**（律区间 [0.0560, 0.1300] 的中点）、厚 **0.0960**、**单元内**间距 0.6400 | 2.3636 m |
| ⑤ | [17.3764, 17.6364] | **坑**（横向沟） | 宽 **0.2600** | 2.3636 m（尾段） |
| — | [17.6364, 20.0000] | 尾平地 | 长 2.3636 m（顶面 0） | — |

真 trimesh 读数（`mesh.bounds` 的 X 区间并集 / 补集）：

* **可走面 X 区间**（顶面 > −0.50 m 的并集）：`[0, 2.25] ∪ [2.51, 17.3764] ∪ [17.6364, 20.0]`
  ⇒ 总长 **19.48 m**；
* **坑**：`[2.25, 2.51]` 与 `[17.3764, 17.6364]`，各宽 0.26 m ⇒ **合计 0.52 m**（`19.48 + 0.52 = 20.00`）；
* **抬高块**（顶面 > 0，共 11 块）：楼梯 7 块 ＋ 箱子 2 块 ＋ 栏 2 道，逐块 `[x0,x1,顶面]` 见
  `tests/test_cmoe_mix_test_scene.py::COMPOSITE_MEASURED_RAISED`；
* 楼梯第 8 级（下行最后一级）顶面 = **0.00 m**，与走廊重合 ⇒ 不在"顶面 > 0"的表里，但**在 X 上仍然
  首尾相接**（`[6.9736, 7.2736]`），由布局的 8 级序列单独核对；
* ⚠️ 复合道**不能用**混合地形那套"按 x0 排序后相邻相减"的朴素算法找坑：抬高块与它所在单元的
  `height = 0` 走廊在 X 上**重叠**，朴素算法会把箱子/栏**单元内部**的平地误报成坑（实测会把
  `[10.0772, 11.3772]` 与 `[14.2768, 14.9168]` 误报为坑 ⇒ 坑总长 2.46 m）。测试里改用
  `_composite_walkable_union()`（区间并集 + 补集）。

### 16.5 与"每块障碍的独立地形"尺寸对照（同一难度 d = 0.70，真 trimesh）

| 障碍 | 本复合道 | 它在**自己那类地形**里的尺寸（训练实例参数，真跑） | 结论 |
|---|---|---|---|
| 坑 | 宽 0.2600 m × 2 处 | `track_gap_terrain`：`gap_width = 0.12 + 0.70×0.20 = 0.26`（三个坑全 0.26） | **逐项一致** |
| 坑前后平地 | 统一 2.3636 m | `platform_length_range=(0.9, 1.4)` ⇒ `1.40 − 0.70×0.50 = 1.05 m` | 本道**更宽松**（1.05 → 2.3636，且是"统一值"） |
| 楼梯（单级） | 级高 0.1550、步深 0.30 | `track_stairs_terrain`：`0.05 + 0.70×(0.20−0.05) = 0.155`、步深 0.30 | **逐项一致**（本道前 4 级与独立地形前 4 级重合） |
| 楼梯（级数） | 上 4 + 下 4（紧贴，共 8 级，段长 2.40 m） | 训练实例是**上行 6 级**（段长 1.80 m，之后是抬升平台） | **有意偏离**：用户确认"上 4 级 + 下 4 级紧贴"；**单级步高/步深逐字相同** |
| 箱子/台阶块 | 高 0.2340、长 0.4400、块数 2、单元内间距 1.30 | `track_step_terrain`：`0.08+0.70×0.22 = 0.234`、`0.30+0.70×0.20 = 0.44`、`num_steps=2`、`step_spacing=1.30` | **逐项一致** |
| 栏（厚度） | 0.0960 m | `stone_len = 0.04 + 0.70×0.08 = 0.096`（律的确定值） | **逐项一致** |
| 栏（高度） | 0.0930 m（律区间 [0.0560, 0.1300] 的**中点**） | 独立地形在 `[0.08·d, 0.06+0.10·d]` 内**逐道随机** | 取律区间中点（**确定性**要求；值落在律区间内） |
| 栏（间距） | 0.6400 m（`spacing_range` 中点） | 独立地形在 `[0.48, 0.80]` 内**随机** | 同上 |

⇒ 同一难度下，**坑宽、楼梯单级步高/步深、箱子高/长/块数/间距、栏厚**与各自独立地形**逐一相等**；
只有三处是**有意**的（都已在代码注释与 CMOE-16 里登记）：① 楼梯级数 4+4（用户指定，单级尺寸不变）；
② 栏高/栏间距取律区间中点（独立地形随机）；③ 障碍之间的平地由本道的统一 `obstacle_spacing` 决定
（独立地形用各自的 `platform_length` / `step_spacing` / `spacing_range`，本道一律 2.3636 m，**更宽**）。

### 16.6 逐对相邻间隔表（用户硬约束）与楼梯"上下紧贴"

| 相邻障碍对 | 间隔长度（m） | 下界（m） | 是否 ≥ 下界 |
|---|---|---|---|
| 坑 ① ↔ 楼梯 ② | 2.3636 | 1.50 | ✅ |
| 楼梯 ② ↔ 箱子 ③ | 2.3636 | 1.50 | ✅ |
| 箱子 ③ ↔ 栏 ④ | 2.3636 | 1.50 | ✅ |
| 栏 ④ ↔ 坑 ⑤ | 2.3636 | 1.50 | ✅ |

* **最小相邻间隔 2.3636 m ≥ 下界 1.5000 m** ⇒ 不存在任何"两个障碍贴在一起"（**不只是**"坑不紧贴
  抬高块"）；两处坑的上下游都先经过平地（`composite_adjacency_violations()` 返回空）。
* **楼梯的"紧贴"只指上/下楼梯之间**：8 级踏步 X 上严丝合缝
  （`[4.8736,5.1736] 0.155 / [5.1736,5.4736] 0.310 / [5.4736,5.7736] 0.465 / [5.7736,6.0736] 0.620 /
  [6.0736,6.3736] 0.465 / [6.3736,6.6736] 0.310 / [6.6736,6.9736] 0.155 / [6.9736,7.2736] 0.000`，
  **级间没有平地**）；而楼梯段与坑/箱子之间各是 **2.3636 m** 平地。
* 单元**内部**的分块间距沿用各自地形的律（箱子 1.30 m、栏 0.64 m）—— 它们同属**一个**障碍单元
  （用户确认的顺序里"【箱子/台阶块】""【栏】"各是一项），不参与"相邻障碍之间"的判据。

### 16.7 出生点、占满与保护

* **出生点**：`spawn_x = 0.75` 落在 `[0, 2.2500]` 的实心平地（顶面 0，且该 X 上没有抬高块）；
  到第一个障碍 **1.5000 m ≥ 1.50 m**（用户硬约束）。
* **占满**：最后一个障碍末端 **17.6364 m** ＋ 尾段平地 **2.3636 m** = **20.0000 m** = `size[0]`；
  真几何里最大的 X 也是 20.0（尾平地铺到瓦片末端）。
* **溢出/下界保护（不放宽）**：
  * `size = (8, 4)`（放不下）⇒ `ValueError`，消息含"三选一"与实际瓦片长；
  * 显式 `obstacle_spacing = 0.0` / `0.5`（< 下界）⇒ `ValueError`（"贴在一起"）；
  * 显式 `obstacle_spacing = 3.3636`（够大但溢出）⇒ `ValueError`；
  * 边界：`size[0] = 15.682`（＝最低可放长度）不抛，`−1e-3` 抛；显式间隔恰 1.50 m 不抛。

### 16.8 负向对照（"贴在一起"必须被判失败）

| 构造 | 期望 | 对应测试 |
|---|---|---|
| 显式 `obstacle_spacing = 0.0`（＝两个障碍贴在一起） | 生成器 `ValueError`（含"贴在一起"） | `TestCompositeTerrainGeometry.test_explicit_spacing_below_the_bound_raises` |
| 显式 `obstacle_spacing = 0.5`（< 下界 1.5） | `ValueError` | 同上 |
| 显式 `obstacle_spacing = 3.3636`（溢出） | `ValueError` | 同上 |
| `size = (8, 4)`（放不下） | `ValueError`（含"三选一"） | `test_overflow_guard_raises_instead_of_silently_clipping` |
| **人造布局**：把第一对相邻间隔改成 0 | `composite_adjacency_violations()` 返回 1 条违规，`pair == ("gap","stairs")` | `test_negative_control_a_tight_layout_is_flagged` |
| **下界调大**：`min_spacing = spacing + 1` | 4 对相邻全部违规 | 同上 |
| 审计脚本：期望顺序/间隔/楼梯级数/末端被改错 | 报告出现 ❌ | `test_check_terrain_columns.py` 的 4 条负向对照 |

### 16.9 训练侧未受影响（证据）

* `track_mix_terrain`、`CMoETrackMixTerrainCfg`、`track_gap_terrain` / `track_stairs_terrain` /
  `track_step_terrain` / `track_hurdle_terrain` **一行未改**；
* 训练类 `Imgo2CMoERoughEnvCfg` 里仍是 `CMoETrackMixTerrainCfg(proportion=0.10)`（不传
  `pattern_spacing_scale` / `fill_stretched_gaps`）、`size = (8.0, 4.0)`、`num_rows/num_cols = 20/40`，
  且**不覆盖** `episode_length_s`；
* 本文件既有的 `TestMixTerrainGeometry` / `TestMixTestSpawnGeometry` / `TestMixTestFilledSpawnGeometry` /
  `TestTrainingSpawnHazard` **原样保留**：其中"默认路径（`pattern_spacing_scale = 1.0`、
  `fill_stretched_gaps = False`）与改动前逐块相同"这条护栏仍在跑（10 组参数对照）；
* 审计脚本 `--task cmoe-rough` 的判据与输出**逐字未变**（11 类各 ≥1 列、比例表、`forward_only` 全覆盖）；
* 新测试 `test_training_uses_mix_and_the_default_path_is_untouched` 再钉一遍：评测 cfg 源码里
  **不再出现** `CMoETrackMixTerrainCfg`，而训练类里**仍有**；真跑 `track_mix_terrain(0.70, 默认 cfg)`
  与既有参照 `_pattern_rows` 逐块一致。

### 16.10 未验证项、限制与与设计的冲突（如实登记）

* **无 GPU / 无 Isaac Lab 运行环境**（本容器 `NVIDIA_VISIBLE_DEVICES=void`、`import isaaclab` 缺
  `omni.log`）⇒ **未构造 Isaac 环境、未回放、未导出 `policy.pt`**。本节所有几何都是"桩 `isaaclab` ＋
  **真 trimesh**"的**函数级**实跑，不含 Isaac 的瓦片拼接、碰撞体与真实 mesh 规模。
* **未验证**：20 m 瓦片的 mesh 规模（三角形数/PhysX 碰撞体/生成耗时）、20 条道的实际出生点分布、
  35 s 单局的实际终止分布、策略能否通过"0.26 m 宽 × 0.50 m 深的横向沟"与"0.62 m 高的锥形楼梯"
  （腿部最大伸展 0.426 m ⇒ **0.62 m 峰高对四足是硬障碍**，可能需要绕行或跳跃）、0.80 m 走廊对
  楼梯/栏是否够宽（本道未改走廊宽，**按用户要求"不要擅自改"**）。
* **与设计的一处待确认**：楼梯峰高 0.62 m 由"上 4 级 × 0.155 m"直接决定，**远高于**训练里 6 级
  （0.93 m 总升，但那是**坡道式**连续上升、不是锥形折返）；本道是"上 4 立即下 4"的锥形，机器人在
  峰顶要立刻反向。若实测发现策略在峰顶卡住，三个不改障碍尺寸的选项：① 减少级数（如 3+3 ⇒ 峰高
  0.465 m）；② 在峰顶插一段**平地**（会与"紧贴"要求冲突，需用户确认）；③ 把楼梯换成"上行 + 平台"
  （即回到独立地形的形态，但整条道后半段会被抬到 0.62 m）。**本轮按用户确认的"上下紧贴"实现**。
* **与用户具体措辞的一处差异**：用户举例的相邻对是"坑↔楼梯、楼梯↔箱子、箱子↔栏、栏↔坑"，本道完全
  按此实现；但"箱子两块之间 1.30 m""栏两道之间 0.64 m"是**单元内部**间距（沿用各自地形律），
  **小于** `obstacle_spacing = 2.3636 m` —— 若用户认为"同一类型的两块也必须拉开到 2.3636 m"，
  需要把 `box_spacing` / `hurdle_spacing_range` 也改成统一间隔（那会改变"尺寸与独立地形一致"这条
  要求），**本轮未擅自改**。
* **`corridor_width = 0.80 m` 未改**：四足机身约 0.24 m 宽，箱/栏/楼梯在 0.80 m 走廊内可放；若实测
  需要更宽（例如栏要起跳落点），按用户要求**只报告不建议擅自改**（改它会同时影响与训练侧的对照）。

### 15.4 第七批（复合道）的实测输出

```
$ cd imgo2_rl && /opt/conda/envs/isaaclab/bin/python -m unittest discover -s tests
Ran 709 tests in 36.9s
OK (skipped=10)

$ /opt/conda/envs/isaaclab/bin/python scripts/tools/check_reward_overrides.py cmoe-gaitfree
===== [cmoe-gaitfree] cmoe gaitfree（先验供步态）：生效 24 项 =====

$ /opt/conda/envs/isaaclab/bin/python scripts/tools/check_terrain_columns.py        # 退出码 0
... （与第一~六批逐字相同）
✅ forward_only_terrain_names 覆盖全部 11 类地形（全场景前向命令）
结论：全部掩码引用的地形名都有 ≥1 列 ✅

$ /opt/conda/envs/isaaclab/bin/python scripts/tools/check_terrain_columns.py --task mix-test   # 退出码 0
=== [Imgo2-basemove-rough-cmoe-mix-test] 受控测试场景（只**复合道**；cfg 类 Imgo2CMoEMixTestEnvCfg）===

=== Imgo2-basemove-rough-cmoe-mix-test｜只 composite（20 道并列）（num_cols=20，共 20 列）===
  composite               20 列
  ✅ 唯一地形 composite 有 20 列（num_cols=20、num_rows=1）⇒ 网格 **20 道 × 1 难度行**（世界 20 m(X) × 80 m(Y)）…

  ── 第五批检查：单块瓦片 `terrain_generator.size`（评测 20×4 m；训练 8×4 m）──
  ✅ 评测 cfg 的 `terrain_generator.size` = 20×4 m（第五批：X 由 8 m 放大到 20 m；Y 仍 4 m）
  ✅ **训练/play 侧**的 `terrain_generator.size` 仍是 8×4 m（共享字段未被评测场景带偏）

  ── 第七批检查：复合道（`composite`，瓦片 size[0] = 20 m）──
  ✅ 障碍按类型分组顺序摆放（不混在同一小段里）：坑（横向沟） → 楼梯（上下紧贴） → 箱子/台阶块 → 栏（薄横栏） → 坑（横向沟）
  ── 障碍清单与 X 区间表（d = 0.7，瓦片 20 m）──
    [0.0000, 2.2500]  平地（出生平地，长 2.2500 m）
    [2.2500, 2.5100]  坑（横向沟）     宽 0.2600 m（坑宽 0.2600 m）
    [2.5100, 4.8736]  平地间隔，长 2.3636 m
    [4.8736, 7.2736]  楼梯（上下紧贴）   宽 2.4000 m（上/下各 4 级，级高 0.1550 m、步深 0.30 m、峰高 0.6200 m）
    [7.2736, 9.6372]  平地间隔，长 2.3636 m
    [9.6372, 11.8172]  箱子/台阶块     宽 2.1800 m（2 块，块高 0.2340 m、块长 0.4400 m、块间距 1.30 m）
    [11.8172, 14.1808]  平地间隔，长 2.3636 m
    [14.1808, 15.0128]  栏（薄横栏）     宽 0.8320 m（2 道，栏高 0.0930 m（律区间 [0.0560, 0.1300] 的中点）、厚 0.0960 m、间距 0.64 m）
    [15.0128, 17.3764]  平地间隔，长 2.3636 m
    [17.3764, 17.6364]  坑（横向沟）     宽 0.2600 m（坑宽 0.2600 m）
    [17.6364, 20.0000]  尾平地，长 2.3636 m
  ── 逐对相邻障碍间隔表（硬约束：**所有**相邻对都必须 ≥ 下界）──
    障碍对                              间隔(m)   下界(m)   是否 ≥ 下界
    坑（横向沟） ↔ 楼梯（上下紧贴）         2.3636    1.5000   ✅
    楼梯（上下紧贴） ↔ 箱子/台阶块           2.3636    1.5000   ✅
    箱子/台阶块 ↔ 栏（薄横栏）           2.3636    1.5000   ✅
    栏（薄横栏） ↔ 坑（横向沟）           2.3636    1.5000   ✅
  ✅ 最小相邻间隔 2.3636 m ≥ 下界 1.5000 m（**每一对相邻障碍之间都有平地** ⇒ 各自独立、不混在一起）
  ✅ 均匀间隔 ＝ 反算值：(20 − 2.2500 − 5.9320) / 5 = **2.3636 m**（自动反算；下界 1.5000 m，建议区间 (1.50, 2.50)）
  ✅ 楼梯上下紧贴：上 4 级 + 下 4 级**首尾相接、级间无平地**（金字塔形），级高 0.155 m、步深 0.3 m、段长 2.40 m、峰高 0.62 m
      [4.8736,5.1736] 0.1550 / [5.1736,5.4736] 0.3100 / [5.4736,5.7736] 0.4650 / [5.7736,6.0736] 0.6200 / [6.0736,6.3736] 0.4650 / [6.3736,6.6736] 0.3100 / [6.6736,6.9736] 0.1550 / [6.9736,7.2736] 0.0000
  ✅ 楼梯与相邻障碍间隔 = 2.3636 m ≥ 下界 1.5000 m（「紧贴」只指上/下楼梯之间）
  ── 占满核算：最后一个障碍末端 17.6364 m ＋ 尾段平地 2.3636 m = 20.0000 m（瓦片 20 m）──
  ✅ 占满整条道：0 → 20.0000 m（最后一段间隔＝尾段 2.3636 m ≥ 下界 1.5000 m）
  ✅ 出生点：`spawn_x` = 0.75 落在 [0, 2.2500] 的实心平地上（顶面 0），到第一个障碍 **1.5000 m ≥ 1.5 m**

  ── 训练侧 `mix` 未受影响（本轮只换评测场景的地形类型）──
  ✅ `CMoETrackMixTerrainCfg` 默认仍是 `pattern_spacing_scale = 1.0`、`fill_stretched_gaps = False` ⇒ 训练几何逐位不变
  ✅ 训练侧 `mix` 图案 12 段、按首尾相接仍切成 3 组、4 级楼梯仍连续…
  ✅ 评测场景的 `sub_terrains` 只有 `composite`（**不再引用** `CMoETrackMixTerrainCfg`）
  ...
结论：test 任务只有复合道 `composite` 一种地形；掩码引用的其它地形名在本场景恒为 0，属**预期** ✅

$ git diff --check       # 干净
```

**真 trimesh 函数级实测（桩 `isaaclab` + 真 `trimesh`，`d = 0.70`、`size = (20, 4)`）**：

* 可走面并集 = `[(0.0, 2.25), (2.51, 17.3764), (17.6364, 20.0)]`，**总长 19.48 m**；
  坑 = `[(2.25, 2.51), (17.3764, 17.6364)]`，**总长 0.52 m**（19.48 + 0.52 = 20.00）；
* 抬高块（顶面 > 0）共 **11 块**：楼梯 7 块（第 8 级顶面 0 与走廊重合）＋ 箱子 2 块 ＋ 栏 2 道；
* 楼梯 8 级 X 上严丝合缝、级间**无平地**；而楼梯与坑/箱子之间各 **2.3636 m**；
* 与独立地形逐项对照：坑宽 `[0.26,0.26,0.26]`（独立）vs `[0.26,0.26]`（本道）；楼梯单级顶面
  `[0.155, 0.31, 0.465, 0.62, …]`（独立 6 级的前 4 级与本道前 4 级相同）；箱子
  `高 [0.234,0.234] / 长 [0.44,0.44] / 间距 1.3` = 本道；栏厚 `0.096` = 本道（栏高/间距本道取律中点
  `0.0930` / `0.6400`，独立地形为区间内随机）；
* 保护：`size=(8,4)`、显式 `obstacle_spacing = 0.0 / 0.5 / 3.3636` 全部 `ValueError`；
  边界 `size[0] = 15.682`（＝`2.25 + 5.932 + 5×1.50`）不抛。
