"""`check_terrain_columns.py` 的回归测试（只用标准库）。

锁住 **2026-09-28 与参考 CMoE 结构对齐后**的地形列分配（用户：「我看了一下 cmoe 的地形设置，
我们四足这个地形还是差点，感觉可以对齐一下」）：

* 比例按参考 `g1_cmoe_config.py::terrain.terrain_dict`（+ 我们保留的 `random_rough`／`flat`），
  `num_cols` 训练与 play 都是 **40**（参考值；原来是 20／10）；
* 11 类地形**每一类都 ≥1 列** —— 0 列时引用它的掩码会静默失效、不报错，这正是本工具存在的理由
  （旧配置在 `num_cols=10` 时 `hf_pyramid_slope_inv` 就是 0 列）。

2026-10-04 追加：工具新增 `--task`（按任务区分）。测试场景 `Imgo2-basemove-rough-cmoe-mix-test`
**只有 `mix` 一种地形** ⇒ 掩码引用的其它地形名在该场景恒为 0、属**预期**。本文件因此分两个类：
`TestTerrainColumns` = **原判据原样保留**（11 类各 ≥1 列、比例、40 列、forward_only 全覆盖）；
`TestMixTestTerrainColumns` = test 任务的**单独**断言（清空后只有 mix、**20 列**（20 条并列的 mix 道）
× **1 行难度**、`difficulty_range=(0.70, 0.70)` 精确固定难度、工具给出明确的"预期"标注），外加
**负向对照**证明原判据没有被放宽。

2026-10-04（第三批，用户："我不需要还保持那么多行，我需要他们并列"）：test 任务的网格由 20 行缩到
**1 行**（世界 X 160 m → 8 m），难度改由 `difficulty_range` 固定 ⇒ 本类的期望值同步改成
`num_rows=1`，并新增"难度固定被工具检查"的断言与**负向对照**。

逐项对照见 `docs/cmoe_terrain_alignment_2026-09-28.md` 与 `docs/cmoe_mix_test_scene_2026-10-04.md`。
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "scripts" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import check_terrain_columns as chk  # noqa: E402

# 2026-09-28 对齐后的 num_cols（训练与 play 同值）
NUM_COLS = 40


def _counts(num_cols: int) -> dict[str, int]:
    base = chk.base_sub_terrains()
    props, _ = chk.cmoe_overrides()
    merged = [(name, props.get(name, default)) for name, default in base]
    merged += [(name, value) for name, value in props.items() if name not in [n for n, _ in merged]]
    return chk.report(merged, num_cols, f"test/{num_cols}")


def _merged_proportions() -> dict[str, float]:
    base = chk.base_sub_terrains()
    props, _ = chk.cmoe_overrides()
    merged = {name: props.get(name, default) for name, default in base}
    for name, value in props.items():
        merged.setdefault(name, value)
    return merged


REPO = Path(__file__).resolve().parents[1]


class TestTerrainColumns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.counts = _counts(NUM_COLS)

    def test_total_columns(self):
        self.assertEqual(sum(self.counts.values()), NUM_COLS)

    def test_every_terrain_type_has_columns(self):
        for name, count in self.counts.items():
            self.assertGreater(count, 0, f"{name} 在 num_cols={NUM_COLS} 时是 0 列 ⇒ 引用它的掩码会静默失效")

    def test_reference_type_set_is_present(self):
        """参考有、我们原来缺的三类必须在配置里（2026-09-28 新增）。"""
        for name in ("hurdle", "mix", "narrow_stairs"):
            self.assertIn(name, self.counts, f"缺少参考地形类型 {name}")

    def test_reference_proportions(self):
        """比例＝参考 `terrain_dict`（+ 保留 random_rough/flat）；rough slope 0.10 拆给上/下坡各 0.05。"""
        props = _merged_proportions()
        expected = {
            "pyramid_stairs": 0.10,       # 参考 stairs up
            "pyramid_stairs_inv": 0.10,   # 参考 stairs down
            "boxes": 0.10,                # 参考 discrete
            "gap": 0.30,                  # 参考 parkour_gap
            "hurdle": 0.10,               # 参考 parkour_hurdle
            "mix": 0.10,                  # 参考 mix
            "narrow_stairs": 0.10,        # 参考 narrow_stairs
            "hf_pyramid_slope": 0.05,     # 参考 rough slope 的一半
            "hf_pyramid_slope_inv": 0.05,  # 参考 rough slope 的另一半
            "random_rough": 0.05,         # 我们保留的纯噪声面
            "flat": 0.10,                 # 我们保留的平地（参考 plane = 0.0）
        }
        for name, value in expected.items():
            self.assertAlmostEqual(props.get(name), value, places=6, msg=f"{name} 比例与对齐表不一致")

    def test_gap_keeps_the_largest_share(self):
        self.assertEqual(self.counts["gap"], 11)
        for name, count in self.counts.items():
            if name != "gap":
                self.assertLess(count, self.counts["gap"], f"{name} 的列数不应超过 gap")

    def test_forward_only_covers_every_terrain(self):
        """用户 2026-09-24："所有场景都只给超前的速度" ⇒ 该名单必须覆盖全部地形，漏项会静默退回全向命令。"""
        fwd = chk.forward_only_names()
        self.assertIsNotNone(fwd, "配置里没有设置 forward_only_terrain_names")
        names = [name for name, _ in chk.base_sub_terrains()]
        props, _ = chk.cmoe_overrides()
        names += [n for n in props if n not in names]
        self.assertEqual(sorted(fwd), sorted(names),
                         f"forward_only 名单与 sub_terrains 不一致：缺 {[n for n in names if n not in fwd]}")

    def test_every_masked_terrain_name_has_columns(self):
        for key, names in chk.MASKED_NAMES.items():
            for name in names:
                self.assertIn(name, self.counts, f"{key} 引用了不存在的地形名 {name}")
                self.assertGreater(self.counts[name], 0,
                                   f"{key} 引用的 {name} 是 0 列 ⇒ 掩码静默失效")

    def test_configured_num_cols_is_40_for_train_and_play(self):
        """训练 40（按比例，课程要每类多列）；**play 11**（2026-09-29 起改成"每类一列"、等比例）。

        直接读两段类体、不依赖工具的合并语义（play 的 11 会覆盖合并结果）。
        """
        cfg_src = (REPO / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/"
                   "base_move/CMoE_env_cfg.py").read_text(encoding="utf-8")
        train = cfg_src.split("class Imgo2CMoERoughEnvCfg", 1)[1].split(
            "class Imgo2CMoERoughPlayEnvCfg", 1)[0]
        play = cfg_src.split("class Imgo2CMoERoughPlayEnvCfg", 1)[1].split("\n@configclass", 1)[0]
        self.assertIn("num_cols = 40", train, "训练应为 40 列（按比例）")
        self.assertIn("num_cols = 11", play, "play 应为每类一列（11）")
        self.assertIn("_sub.proportion = 1.0", play, "play 应设等比例（否则会有 0 列的地形）")


class TestMixTestTerrainColumns(unittest.TestCase):
    """`Imgo2-basemove-rough-cmoe-mix-test`（**只有复合道**的受控测试场景）的单独断言。

    2026-10-04 用户要求：这个测试场景只有一种地形 ⇒ 掩码引用的其它地形名在本场景**恒为 0**（预期）。
    处理方式必须是"给工具加**按任务区分**的能力并**明确标注**"，而**不是**放宽原判据：
    ① 原任务（`--task cmoe-rough`）的"每一项 ≥1 列"判据在本文件里原样保留（见
    `TestTerrainColumns.test_every_masked_terrain_name_has_columns` 与下面的负向对照）；
    ② test 任务走**单独分支**，仍然强制 `sub_terrains.clear()` + 只有一种地形 + 占满全部列 +
    `num_cols=20`（＝`MIX_TEST_LANES`）+ `num_rows=1`（唯一一行难度）+ `difficulty_range=(0.70,0.70)`，
    并把"其它地形名 0 列"**逐条打印成预期**。

    **2026-10-05（第七批）**：唯一地形由 `mix` 换成**复合道**（键名 `composite`，
    `cmoe_terrains.track_composite_terrain`）⇒ 本类整体改写为复合道的断言：
    障碍清单/X 区间表、**逐对相邻间隔 ≥ 下界**、**楼梯上下紧贴**、占满 20 m、出生点前方 1.50 m，
    以及一段"训练侧 `mix` 未受影响"的复核。真 trimesh 侧的交叉核对在
    `tests/test_cmoe_mix_test_scene.py::TestCompositeTerrainGeometry`。
    """

    @classmethod
    def setUpClass(cls):
        cls.info = chk.scene_overrides(chk.MIX_TEST_CLASS)
        cls.props = cls.info["props"]
        cls.task = chk.MIX_TEST_TASK
        cls.key = chk.MIX_TEST_SUB_TERRAIN_KEY

    # ------------------------------------------------------------- test 任务自身
    def test_task_id_and_class_are_wired_to_the_tool(self):
        self.assertEqual(chk.TASK_ALIASES["mix-test"], chk.MIX_TEST_TASK)
        self.assertEqual(chk.TASK_CLASS[chk.MIX_TEST_TASK], chk.MIX_TEST_CLASS)

    def test_sub_terrains_are_cleared(self):
        self.assertTrue(self.info["cleared"], "test 任务必须先清空基类地形，否则会继承 11 类")

    def test_composite_is_the_only_sub_terrain(self):
        self.assertEqual(list(self.props.keys()), [self.key], f"只允许复合道，实测 {list(self.props)}")
        self.assertAlmostEqual(self.props[self.key], 1.0, places=9)
        self.assertEqual(self.key, "composite")

    def test_num_cols_twenty_and_num_rows_one(self):
        """20 条**并列**的道（列、沿世界 Y）× **唯一一行**难度（行、沿世界 X）。"""
        self.assertEqual(self.info["num_cols"], 20, "20 条并列的道（＝可同时评估的环境数上限）")
        self.assertEqual(self.info["num_rows"], 1, "只留唯一一行难度 ⇒ 世界 X 只有 20 m")
        self.assertEqual(chk.MIX_TEST_LANES_EXPECTED, 20)
        self.assertEqual(chk.MIX_TEST_LEVELS_EXPECTED, 1)

    def test_difficulty_range_is_exactly_pinned(self):
        """难度由 `difficulty_range = (0.70, 0.70)` 精确固定（＝旧"第 14 行"的名义难度）。"""
        self.assertEqual(self.info["difficulty_range"], (0.70, 0.70),
                         f"上下界必须同值（否则行内 U(0,1) 抖动会回来）：{self.info['difficulty_range']}")
        self.assertAlmostEqual(chk.MIX_TEST_DIFFICULTY_EXPECTED, 0.70, places=9)

    def test_num_cols_is_read_from_the_module_constant(self):
        """`num_cols`/`num_rows`/难度/复合道常量都写成可读常量 ⇒ 工具必须能求值模块常量。"""
        consts = chk.module_constants()
        self.assertEqual(consts["MIX_TEST_LANES"], 20)
        self.assertEqual(consts["MIX_TEST_LEVELS"], 1)
        self.assertAlmostEqual(consts["MIX_TEST_DIFFICULTY"], 0.70, places=9)
        self.assertEqual(consts["MIX_TEST_TILE_SIZE"], (20.0, 4.0))
        self.assertAlmostEqual(consts["MIX_TEST_EPISODE_LENGTH_S"], 35.0, places=9)
        self.assertAlmostEqual(consts["MIX_TEST_EPISODE_LENGTH_MIN_S"], 25.0, places=9)
        self.assertAlmostEqual(consts["MIX_TEST_FORWARD_SPEED"], 1.0, places=9)
        # 第七批新增的复合道常量
        self.assertEqual(consts["MIX_TEST_SUB_TERRAIN_KEY"], "composite")
        self.assertAlmostEqual(consts["MIX_TEST_OBSTACLE_SPACING"], 2.3636, places=3)
        self.assertAlmostEqual(consts["MIX_TEST_MIN_OBSTACLE_SPACING"], 1.50, places=9)
        self.assertAlmostEqual(consts["MIX_TEST_SPAWN_CLEARANCE"], 1.50, places=9)
        # 旧的 mix 图案常量必须随场景切换一起删掉（留着会让人以为场景还在用 mix）
        for gone in ("MIX_TEST_PATTERN_SPACING_SCALE", "MIX_TEST_TAIL_MARGIN",
                     "MIX_TEST_PATTERN_END_X", "MIX_TEST_FILL_STRETCHED_GAPS"):
            self.assertNotIn(gone, consts, f"评测场景已改用复合道 ⇒ 旧常量 {gone} 应删除")
        self.assertNotIn("MIX_TEST_PINNED_LEVEL", consts,
                         "旧的'钉第 14 行'常量必须随子类一起删掉（难度改由 difficulty_range 固定）")
        # 字面量与常量两种写法都要能读（`literal` 先字面量、后常量命名空间）
        self.assertEqual(chk.literal(ast.parse("7", mode="eval").body, consts), 7)
        self.assertEqual(chk.literal(ast.parse("MIX_TEST_LANES", mode="eval").body, consts), 20)
        self.assertEqual(chk.literal(ast.parse("MIX_TEST_SUB_TERRAIN_KEY", mode="eval").body, consts),
                         "composite")

    def test_terrain_size_is_read_from_the_source_per_class(self):
        """`terrain_generator.size` 由**各类的 `__post_init__`** 设定 ⇒ 工具要**按类**真读源码（不猜）。

        第五批：`size` 是共享字段 ⇒ 评测 cfg 覆盖成 `(20, 4)`、训练/play 仍是 `(8, 4)`。
        旧的"全文件扫描"会先撞到训练类，必须改成类作用域。
        """
        self.assertEqual(chk.terrain_size(chk.MIX_TEST_CLASS), (20.0, 4.0),
                         "评测 cfg 的单块瓦片应是 20 m(X) × 4 m(Y)")
        self.assertEqual(chk.terrain_size(chk.TRAIN_CLASS), (8.0, 4.0),
                         "训练/play 的共享 size 必须仍是 (8, 4)")
        self.assertEqual(chk.MIX_TEST_TILE_SIZE_EXPECTED, (20.0, 4.0))
        self.assertEqual(chk.TRAIN_TILE_SIZE_EXPECTED, (8.0, 4.0))
        # play 类不设 size ⇒ 落到"父类/训练值"的语义（这里退化成默认值 (8,4)）
        self.assertEqual(chk.terrain_size(chk.PLAY_CLASS), (8.0, 4.0))
        # 未知类名 ⇒ 退回默认值（不抛异常）
        self.assertEqual(chk.terrain_size("NoSuchClass"), (8.0, 4.0))

    def test_episode_length_is_read_from_the_source_per_class(self):
        """第五批：单局时长要**按类**解析 —— 评测 cfg = 35 s；训练/play 不覆盖（`None`＝沿用父类 20 s）。"""
        self.assertAlmostEqual(chk.episode_length_s(chk.MIX_TEST_CLASS), 35.0, places=9)
        self.assertIsNone(chk.episode_length_s(chk.TRAIN_CLASS),
                          "训练侧不得覆盖 episode_length_s")
        self.assertIsNone(chk.episode_length_s(chk.PLAY_CLASS),
                          "play 侧不得覆盖 episode_length_s")
        self.assertAlmostEqual(chk.MIX_TEST_EPISODE_LENGTH_MIN_S, 25.0, places=9)
        needed = chk.MIX_TEST_TILE_SIZE_EXPECTED[0] / chk.MIX_TEST_FORWARD_SPEED
        self.assertAlmostEqual(needed, 20.0, places=9, msg="走完 20 m 需要 20 s")
        self.assertGreaterEqual(chk.MIX_TEST_EPISODE_LENGTH_EXPECTED, chk.MIX_TEST_EPISODE_LENGTH_MIN_S)

    # ------------------------------------------- 2026-10-05（第七批）：复合道（离线算术复算）
    def test_composite_call_kwargs_are_parsed(self):
        """工具要能从 `sub_terrains['composite'] = <Cfg>(...)` 读出全部字段（含写成常量名的）。"""
        calls = self.info["sub_terrain_calls"]
        self.assertEqual(list(calls.keys()), [self.key])
        kwargs = calls[self.key]
        self.assertAlmostEqual(kwargs["proportion"], 1.0, places=9)
        self.assertLess(kwargs["obstacle_spacing"], 0.0, "必须是自动反算的哨兵")
        self.assertAlmostEqual(kwargs["spawn_clearance"], 1.50, places=9)
        self.assertAlmostEqual(kwargs["min_obstacle_spacing"], 1.50, places=9)
        self.assertEqual(tuple(kwargs["gap_width_range"]), (0.12, 0.32))
        self.assertEqual(int(kwargs["stairs_num_steps"]), 4)
        self.assertEqual(tuple(kwargs["stairs_step_height_range"]), (0.05, 0.20))
        self.assertEqual(int(kwargs["box_count"]), 2)
        self.assertEqual(tuple(kwargs["box_length_range"]), (0.30, 0.50))

    def _layout(self):
        return chk.composite_layout(chk.MIX_TEST_DIFFICULTY_EXPECTED,
                                    self.info["sub_terrain_calls"][self.key],
                                    chk.terrain_size(chk.MIX_TEST_CLASS))

    def test_composite_layout_matches_the_hardcoded_expectations(self):
        """障碍顺序/宽度/总宽/间隔必须与工具里**写死**的期望值一致（真几何用真 trimesh 另测）。"""
        layout = self._layout()
        self.assertEqual(tuple(layout["sequence"]), chk.COMPOSITE_SEQUENCE_EXPECTED)
        self.assertEqual([round(float(u["width"]), 6) for u in layout["units"]],
                         [round(w, 6) for w in chk.COMPOSITE_WIDTHS_EXPECTED])
        self.assertAlmostEqual(float(layout["total_width"]), chk.COMPOSITE_TOTAL_WIDTH_EXPECTED, places=9)
        self.assertAlmostEqual(float(layout["spacing"]), chk.COMPOSITE_SPACING_EXPECTED, places=9)
        # 反算式自洽：(20 − 2.25 − 5.932) / 5 = 2.3636
        self.assertAlmostEqual(
            (20.0 - chk.COMPOSITE_FIRST_OBSTACLE_X_EXPECTED - chk.COMPOSITE_TOTAL_WIDTH_EXPECTED) / 5,
            chk.COMPOSITE_SPACING_EXPECTED, places=9,
        )
        self.assertAlmostEqual(float(layout["last_obstacle_end"]),
                               chk.COMPOSITE_LAST_OBSTACLE_END_EXPECTED, places=9)
        self.assertAlmostEqual(float(layout["end_x"]), 20.0, places=9, msg="必须占满整条道")

    def test_adjacent_gaps_all_meet_the_lower_bound(self):
        """**逐对相邻障碍**：每一对（含尾段以外的所有相邻对）的平地都 ≥ 下界。"""
        layout = self._layout()
        pairs = tuple(tuple(g["pair"]) for g in layout["adjacent_gaps"])
        self.assertEqual(pairs, (("gap", "stairs"), ("stairs", "boxes"),
                                 ("boxes", "hurdle"), ("hurdle", "gap")))
        worst = min(float(g["length"]) for g in layout["adjacent_gaps"])
        self.assertGreaterEqual(worst, chk.COMPOSITE_MIN_SPACING_EXPECTED - 1.0e-9)
        self.assertAlmostEqual(worst, chk.COMPOSITE_SPACING_EXPECTED, places=9, msg="间隔等长")
        self.assertAlmostEqual(
            float(layout["spacing"]), (20.0 - float(layout["first_obstacle_x"])
                                      - float(layout["total_width"])) / len(layout["obstacles"]),
            places=12, msg="间隔必须等于反算式",
        )

    def test_stairs_up_and_down_are_contiguous(self):
        """楼梯：上 4 级 + 下 4 级**首尾相接**、级间无平地；峰高 0.62 m、段长 2.40 m。"""
        layout = self._layout()
        stairs = [o for o in layout["obstacles"] if o["kind"] == "stairs"]
        self.assertEqual(len(stairs), 1)
        params = stairs[0]["params"]
        self.assertEqual(int(params["num_steps"]), chk.COMPOSITE_STAIR_LEVELS_EXPECTED)
        self.assertEqual(int(params["step_count_total"]), 2 * chk.COMPOSITE_STAIR_LEVELS_EXPECTED)
        self.assertAlmostEqual(float(params["step_height"]),
                               chk.COMPOSITE_STAIR_STEP_HEIGHT_EXPECTED, places=9)
        self.assertAlmostEqual(float(params["peak_height"]),
                               chk.COMPOSITE_STAIR_PEAK_EXPECTED, places=9)
        self.assertAlmostEqual(float(stairs[0]["width"]),
                               2 * chk.COMPOSITE_STAIR_LEVELS_EXPECTED * 0.30, places=9)
        # 楼梯与相邻障碍之间的平地也必须 ≥ 下界（"紧贴"只指上/下楼梯之间）
        for gap in layout["adjacent_gaps"]:
            if "stairs" in gap["pair"]:
                self.assertGreaterEqual(float(gap["length"]),
                                        chk.COMPOSITE_MIN_SPACING_EXPECTED - 1.0e-9)

    def test_spawn_clearance_and_first_obstacle(self):
        """出生点：`spawn_x = 0.75` 前方 1.50 m 平地 ⇒ 第一个障碍左沿 = 2.25 m。"""
        layout = self._layout()
        self.assertAlmostEqual(float(layout["spawn_x"]), chk.COMPOSITE_SPAWN_X_EXPECTED, places=9)
        self.assertGreaterEqual(float(layout["spawn_clearance"]),
                                chk.COMPOSITE_SPAWN_CLEARANCE_MIN - 1.0e-9)
        self.assertAlmostEqual(float(layout["first_obstacle_x"]),
                               chk.COMPOSITE_FIRST_OBSTACLE_X_EXPECTED, places=9)

    # ------------------------------------------- 训练侧 mix 几何仍可离线复核（原判据保留）
    def test_pattern_segments_are_parsed_from_the_source(self):
        """`segments` 表必须能从 `cmoe_terrains.py` **AST 解析**出来（只用标准库 ⇒ 离线可复核）。"""
        segments = chk.mix_pattern_segments(chk.MIX_TEST_DIFFICULTY_EXPECTED)
        self.assertEqual(len(segments), 12, "训练侧图案 12 段")
        self.assertEqual(segments[0], (0.0, 30.0, 0.0))
        self.assertEqual(segments[5][0], 69.0, "第一处坑的下游段起点 = 72 − round(10·(1−0.70)) = 69")
        self.assertEqual(segments[10][0], 120.0, "第二处坑的下游段起点 = 123 − 3 = 120")
        self.assertEqual(segments[-1], (140.0, 160.0, 60.0))
        self.assertTrue(all(end > start for start, end, _height in segments))

    def test_contiguous_groups_are_three_and_include_the_staircase(self):
        """分组 = 按**原始 units 首尾相接**切；d = 0.70 时 3 组、组内逐段相接、楼梯 4 级在同一组。"""
        groups = chk.mix_contiguous_groups(chk.mix_pattern_segments(chk.MIX_TEST_DIFFICULTY_EXPECTED))
        spans = tuple((group[0][0], group[-1][1]) for group in groups)
        self.assertEqual(len(groups), chk.MIX_CONTIGUOUS_GROUPS_EXPECTED)
        self.assertEqual(spans, chk.MIX_GROUP_SPANS_EXPECTED)
        for group in groups:
            for before, after in zip(group, group[1:]):
                self.assertAlmostEqual(after[0], before[1], places=9, msg="组内必须首尾相接")
        stair_group = next(group for group in groups if group[0][0] <= 30.0 <= group[-1][1])
        raised = [segment for segment in stair_group if segment[2] > 0.0]
        self.assertEqual(len(raised), chk.MIX_STAIR_LEVELS_EXPECTED, "楼梯必须是同一组内首尾相接的 4 级")

    def test_group_layout_puts_the_flat_upstream_of_the_pits(self):
        """训练侧 `mix` 几何核算（**评测场景已不用它**，仍作为训练侧回归保留）：楼梯 4 级连续、
        两处坑紧贴**下游组起点**、平地铺在**坑的上游**且均匀分配。"""
        layout = chk.mix_group_layout(chk.MIX_TEST_DIFFICULTY_EXPECTED, chk.MIX_WIDE_SCALE_EXPECTED)
        stairs = layout["stairs"]
        self.assertEqual([round(top, 4) for _a, _b, top in stairs], list(chk.MIX_STAIR_TOPS_EXPECTED))
        self.assertEqual([round(a, 4) for a, _b, _t in stairs], [0.9, 1.02, 1.14, 1.26])
        for before, after in zip(stairs, stairs[1:]):
            self.assertAlmostEqual(before[1], after[0], places=9, msg="级间不得有空档")
        self.assertEqual([round(width, 9) for width in layout["pit_widths"]], [0.18, 0.18])
        self.assertEqual([(round(a, 4), round(b, 4)) for a, b in layout["pits"]],
                         [(6.8333, 7.0133), (13.1867, 13.3667)])
        fills = [fill for fill in layout["fills"] if fill[1] - fill[0] > 1.0e-12]
        self.assertEqual(len(fills), chk.MIX_GAP_SLOTS_EXPECTED)
        for fill, pit in zip(fills[:2], layout["pits"]):
            self.assertAlmostEqual(fill[1], pit[0], places=9, msg="平地必须正好铺到坑的左沿")
            self.assertGreaterEqual(fill[1] - fill[0], chk.MIX_UPSTREAM_FLAT_MIN_S)
        self.assertAlmostEqual(fills[0][1] - fills[0][0], 16.0 / 3.0, places=9)
        self.assertAlmostEqual(fills[2][1], 19.5, places=9, msg="尾段补块铺到图案末端")

    def test_hard_combinations_are_report_only(self):
        """图案里其它"难组合"只报告、不改几何：三行都在、都带"只报告"、且数与实测一致。

        （函数仍在：它是**训练侧 mix** 的只报告清单；评测场景已改用复合道、不再打印这三行。）
        """
        lines = chk.mix_hard_combinations(chk.MIX_TEST_DIFFICULTY_EXPECTED)
        self.assertEqual(len(lines), 3, lines)
        for line in lines:
            self.assertIn("只报告", line)
        self.assertIn("窄凹口", lines[0])
        self.assertIn("84:86", lines[0])
        self.assertIn("尖峰", lines[1])
        self.assertIn("落差", lines[2])

    # ------------------------------------------------------------- 报告（正向）
    def test_report_checks_the_composite_track(self):
        """工具必须打印并**检查**：障碍清单/X 区间表、逐对相邻间隔、楼梯上下紧贴、占满、出生点。"""
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            problems, lines = chk.mix_test_report()
        text = "\n".join(lines)
        self.assertEqual(problems, 0, text)
        self.assertIn("第七批检查：复合道", text)
        self.assertIn("障碍清单与 X 区间表", text)
        self.assertIn("逐对相邻障碍间隔表", text)
        self.assertIn("✅ 最小相邻间隔 2.3636 m ≥ 下界 1.5000 m", text)
        self.assertIn("楼梯上下紧贴", text)
        self.assertIn("楼梯与相邻障碍间隔 = 2.3636 m ≥ 下界", text)
        self.assertIn("占满整条道", text)
        self.assertIn("到第一个障碍 **1.5000 m ≥ 1.5 m**", text)
        self.assertIn("训练侧 `mix` 未受影响", text)
        self.assertIn("第五批检查：单块瓦片", text)
        self.assertIn("训练/play 侧**的 `terrain_generator.size` 仍是 8×4 m", text)
        self.assertIn("单局时长", text)
        self.assertIn("35 s", text)
        self.assertNotIn("❌", text)
        # 工具里写死的期望值也钉一遍（防止 cfg 常量被改错而工具沉默）
        self.assertEqual(chk.COMPOSITE_SEQUENCE_EXPECTED, ("gap", "stairs", "boxes", "hurdle", "gap"))
        self.assertAlmostEqual(chk.COMPOSITE_SPACING_EXPECTED, 2.3636, places=9)
        self.assertAlmostEqual(chk.COMPOSITE_TOTAL_WIDTH_EXPECTED, 5.932, places=9)
        self.assertAlmostEqual(chk.COMPOSITE_MIN_SPACING_EXPECTED, 1.50, places=9)

    def test_report_shows_twenty_columns_and_one_row(self):
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            chk.mix_test_report()
        text = buffer.getvalue()
        self.assertIn("num_cols=20，共 20 列", text)
        self.assertIn("composite               20 列", text)
        self.assertIn("网格 **20 道 × 1 难度行**", text)
        self.assertIn("世界 20 m(X) × 80 m(Y)", text)

    def test_mix_test_report_is_clean_and_annotates_absence(self):
        """test 任务判定必须通过（0 问题），并把"其它地形名恒为 0"明确标为预期。"""
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            problems, lines = chk.mix_test_report()
        text = "\n".join(lines)
        self.assertEqual(problems, 0, f"test 任务不应报错：{text}")
        self.assertIn("属**预期**", text)
        self.assertIn("复合道", text)
        self.assertNotIn("❌", text)
        # 每一条被引用的"本场景不存在"的地形名都要**逐条**标注（现在连 `mix` 也不存在了）
        absent = sorted({n for refs in chk.MASKED_NAMES.values() for n in refs}
                        - {chk.MIX_TEST_SUB_TERRAIN_KEY})
        self.assertTrue(absent, "本测试的前提是掩码引用了别的类地形名")
        for name in absent:
            self.assertIn(name, text, f"缺少对 '{name}' 的预期标注")

    def test_absent_terrain_names_are_all_annotated(self):
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            _, lines = chk.mix_test_report()
        text = "\n".join(lines)
        for key, refs in chk.MASKED_NAMES.items():
            for name in refs:
                if name != chk.MIX_TEST_SUB_TERRAIN_KEY:
                    self.assertIn(f"{key} → {name}", text)

    def test_forward_only_list_is_present_but_lazy_for_this_scene(self):
        """本场景的命令项**不读** `forward_only_terrain_names` ⇒ 只需该名单在继承链里存在。"""
        fwd = chk.forward_only_names(chk.MIX_TEST_CLASS) or chk.forward_only_names(chk.TRAIN_CLASS)
        self.assertIsNotNone(fwd)
        self.assertIn("mix", fwd, "训练侧名单里仍有 mix（训练未受影响）")

    # ------------------------------------------------------------- 负向对照（第七批）
    def _report(self):
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            problems, lines = chk.mix_test_report()
        return problems, "\n".join(lines)

    def test_report_flags_a_wrong_sequence(self):
        """负向对照：期望障碍顺序被改错时（或换成旧 mix 图案）必须报 ❌。"""
        original = chk.COMPOSITE_SEQUENCE_EXPECTED
        chk.COMPOSITE_SEQUENCE_EXPECTED = ("gap", "stairs", "gap")
        try:
            problems, text = self._report()
        finally:
            chk.COMPOSITE_SEQUENCE_EXPECTED = original
        self.assertGreater(problems, 0, text)
        self.assertIn("障碍顺序应为", text)

    def test_report_flags_a_wrong_spacing_expectation(self):
        """负向对照：期望间隔被改成别的值时必须报 ❌（"2.3636 m"是被**检查**的）。"""
        original = chk.COMPOSITE_SPACING_EXPECTED
        chk.COMPOSITE_SPACING_EXPECTED = 3.0
        try:
            problems, text = self._report()
        finally:
            chk.COMPOSITE_SPACING_EXPECTED = original
        self.assertGreater(problems, 0, text)
        self.assertIn("均匀间隔应为反算值", text)

    def test_report_flags_a_wrong_min_spacing(self):
        """负向对照：下界被抬到 3 m 时 2.3636 m 的间隔必须被判"小于下界"。"""
        original = chk.COMPOSITE_MIN_SPACING_EXPECTED
        chk.COMPOSITE_MIN_SPACING_EXPECTED = 3.0
        try:
            problems, text = self._report()
        finally:
            chk.COMPOSITE_MIN_SPACING_EXPECTED = original
        # 期望值只影响 `min_obstacle_spacing` 的核对；把下界抬到 3 会先报"应为 3.0 m"
        self.assertGreater(problems, 0, text)
        self.assertIn("`min_obstacle_spacing` 应为", text)

    def test_report_flags_a_wrong_stair_level_count(self):
        """负向对照：期望楼梯级数被改错时必须报 ❌（"上 4 + 下 4"是被检查的）。"""
        original = chk.COMPOSITE_STAIR_LEVELS_EXPECTED
        chk.COMPOSITE_STAIR_LEVELS_EXPECTED = 3
        try:
            problems, text = self._report()
        finally:
            chk.COMPOSITE_STAIR_LEVELS_EXPECTED = original
        self.assertGreater(problems, 0, text)
        self.assertIn("楼梯级数应为", text)

    def test_report_flags_a_lane_that_is_not_filled(self):
        """负向对照：末端期望值改错（或道变短）⇒ "未占满整条道"必须被报出来。"""
        original = chk.COMPOSITE_LAST_OBSTACLE_END_EXPECTED
        chk.COMPOSITE_LAST_OBSTACLE_END_EXPECTED = 19.0
        try:
            problems, text = self._report()
        finally:
            chk.COMPOSITE_LAST_OBSTACLE_END_EXPECTED = original
        self.assertGreater(problems, 0, text)
        self.assertIn("最后一个障碍末端应为", text)

    def test_report_flags_a_missing_composite_call(self):
        """负向对照：`sub_terrains` 里没有 `composite` 时必须报 ❌。"""
        import contextlib
        import io

        original = chk.scene_overrides

        def patched(class_name=chk.TRAIN_CLASS):
            info = original(class_name)
            if class_name == chk.MIX_TEST_CLASS:
                info["sub_terrain_calls"] = {}
            return info

        chk.scene_overrides = patched
        try:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                problems, lines = chk.mix_test_report()
            text = "\n".join(lines)
        finally:
            chk.scene_overrides = original
        self.assertGreater(problems, 0, text)
        self.assertIn("解析不到", text)

    def test_report_flags_a_too_short_episode(self):
        """负向对照：下界抬到 40 s 时 35 s 的单局时长必须被判"余量不足"。"""
        import contextlib
        import io

        original = chk.MIX_TEST_EPISODE_LENGTH_MIN_S
        chk.MIX_TEST_EPISODE_LENGTH_MIN_S = 40.0
        try:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                problems, lines = chk.mix_test_report()
            text = "\n".join(lines)
        finally:
            chk.MIX_TEST_EPISODE_LENGTH_MIN_S = original
        self.assertGreater(problems, 0, text)
        self.assertIn("下界 40 s", text)

    def test_report_flags_a_missing_episode_length_override(self):
        """负向对照：评测 cfg 没覆盖 `episode_length_s` 时必须报 ❌（继承来的 20 s 不够走 20 m）。"""
        import contextlib
        import io

        original = chk.episode_length_s

        def patched(class_name=chk.TRAIN_CLASS):
            if class_name == chk.MIX_TEST_CLASS:
                return None
            return original(class_name)

        chk.episode_length_s = patched
        try:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                problems, lines = chk.mix_test_report()
            text = "\n".join(lines)
        finally:
            chk.episode_length_s = original
        self.assertGreater(problems, 0, text)
        self.assertIn("未覆盖 `episode_length_s`", text)

    def test_report_flags_a_wrong_tile_size(self):
        """负向对照：评测瓦片期望值改错时必须报 ❌（"20 m"是被检查的）。"""
        import contextlib
        import io

        original = chk.MIX_TEST_TILE_SIZE_EXPECTED
        chk.MIX_TEST_TILE_SIZE_EXPECTED = (8.0, 4.0)
        try:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                problems, lines = chk.mix_test_report()
            text = "\n".join(lines)
        finally:
            chk.MIX_TEST_TILE_SIZE_EXPECTED = original
        self.assertGreater(problems, 0, text)
        self.assertIn("`terrain_generator.size` 应为", text)

    def test_report_flags_the_training_side_being_scaled_up(self):
        """负向对照：训练侧 `size` 若不是 (8,4)（＝被评测场景带偏）必须报 ❌。"""
        import contextlib
        import io

        original = chk.TRAIN_TILE_SIZE_EXPECTED
        chk.TRAIN_TILE_SIZE_EXPECTED = (20.0, 4.0)
        try:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                problems, lines = chk.mix_test_report()
            text = "\n".join(lines)
        finally:
            chk.TRAIN_TILE_SIZE_EXPECTED = original
        self.assertGreater(problems, 0, text)
        self.assertIn("训练/play 侧**的 `terrain_generator.size` 必须仍是", text)

    def test_report_flags_a_changed_training_mix_default(self):
        """负向对照：训练侧 `mix` 的默认值被改（`fill_stretched_gaps=True`）必须报 ❌。"""
        import contextlib
        import io

        original = chk.cfg_field_defaults

        def patched(class_name):
            defaults = original(class_name)
            if class_name == "CMoETrackMixTerrainCfg":
                defaults = dict(defaults)
                defaults["fill_stretched_gaps"] = True
            return defaults

        chk.cfg_field_defaults = patched
        try:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                problems, lines = chk.mix_test_report()
            text = "\n".join(lines)
        finally:
            chk.cfg_field_defaults = original
        self.assertGreater(problems, 0, text)
        self.assertIn("默认值被改了", text)

    # ------------------------------------------------- 原判据不放宽（负向对照）
    def test_original_criteria_still_fail_on_absent_names(self):
        """负向对照：把"只有复合道"的场景丢给**原判据**，必须仍然报错（证明没被放宽）。"""
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            problems = chk.check_mask_columns({"composite": 1}, 1)
        self.assertGreater(problems, 0, "原判据必须继续对'掩码引用的地形名不存在'报错")
        self.assertIn("不在 sub_terrains 里", buffer.getvalue())

    def test_original_criteria_still_fail_on_zero_columns(self):
        """负向对照：存在但 0 列时原判据也必须报错。"""
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            problems = chk.check_mask_columns({"composite": 1, "boxes": 0, "gap": 0, "flat": 0}, 1)
        self.assertGreater(problems, 0)
        self.assertIn("只有 0 列", buffer.getvalue())

    def test_masked_names_table_is_unchanged(self):
        """`MASKED_NAMES` 是本工具的原判据，按任务区分**不得**改它。"""
        self.assertEqual(chk.MASKED_NAMES["joint_mirror.free_terrain_names"], ("boxes",))
        self.assertEqual(chk.MASKED_NAMES["joint_mirror.bound_terrain_names"], ("gap",))
        self.assertEqual(chk.MASKED_NAMES["feet_gait.free_terrain_names"], ("boxes", "gap"))
        self.assertEqual(chk.MASKED_NAMES["base_height_flat_l2.active_terrain_names"], ("flat",))

    # ------------------------------------------------- 类作用域解析：不污染默认任务
    def test_default_task_parse_is_not_contaminated_by_the_test_scene(self):
        """mix-test 类里也写了 `sub_terrains[...]`；默认任务必须仍按训练值解析（比例 0.10、40 列）。"""
        props, cols = chk.cmoe_overrides()
        self.assertAlmostEqual(props["mix"], 0.10, places=9)
        self.assertEqual(cols["num_cols"], 40)
        merged, _info = chk.merged_scene(chk.TRAIN_CLASS)
        self.assertEqual(len(merged), 11, "默认任务仍是 11 类地形（不被 test 场景污染）")

    # ------------------------------------------------- CLI
    def test_cli_task_flag(self):
        import contextlib
        import io

        for task in ("cmoe-rough", "mix-test", chk.MIX_TEST_TASK):
            with self.subTest(task=task):
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    code = chk.main(["--task", task])
                self.assertEqual(code, 0, buffer.getvalue())

    def test_cli_rejects_unknown_task(self):
        with self.assertRaises(SystemExit):
            chk.main(["--task", "no-such-task"])


if __name__ == "__main__":
    sys.exit(unittest.main())
