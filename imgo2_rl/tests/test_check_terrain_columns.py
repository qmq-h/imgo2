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
    """`Imgo2-basemove-rough-cmoe-mix-test`（**只有 mix** 的受控测试场景）的单独断言。

    2026-10-04 用户要求：这个测试场景只有 `mix` ⇒ 掩码引用的其它地形名在本场景**恒为 0**（预期）。
    处理方式必须是"给工具加**按任务区分**的能力并**明确标注**"，而**不是**放宽原判据：
    ① 原任务（`--task cmoe-rough`）的"每一项 ≥1 列"判据在本文件里原样保留（见
    `TestTerrainColumns.test_every_masked_terrain_name_has_columns` 与下面的负向对照）；
    ② test 任务走**单独分支**，仍然强制 `sub_terrains.clear()` + 只有 mix + mix 占满全部列 +
    `num_cols=20`（＝`MIX_TEST_LANES`：20 条**并列**的 mix 道）+ `num_rows=1`（唯一一行难度）+
    `difficulty_range=(0.70, 0.70)`，并把"其它地形名 0 列"**逐条打印成预期**。
    2026-10-04（第二批，用户："地形不要按照列排，放在行里面"）：列数 **1 → 20**（20 条道沿世界 Y
    并列）⇒ 本类的期望值同步改成 20；同时加"模块级可读常量能被工具解析"的断言。
    2026-10-04（第三批，用户："我不需要还保持那么多行，我需要他们并列"）：行数 **20 → 1**、
    难度改由 `difficulty_range` 精确固定 ⇒ 本类期望改成 `num_rows=1` 并检查难度范围。
    """

    @classmethod
    def setUpClass(cls):
        cls.info = chk.scene_overrides(chk.MIX_TEST_CLASS)
        cls.props = cls.info["props"]
        cls.task = chk.MIX_TEST_TASK

    # ------------------------------------------------------------- test 任务自身
    def test_task_id_and_class_are_wired_to_the_tool(self):
        self.assertEqual(chk.TASK_ALIASES["mix-test"], chk.MIX_TEST_TASK)
        self.assertEqual(chk.TASK_CLASS[chk.MIX_TEST_TASK], chk.MIX_TEST_CLASS)

    def test_sub_terrains_are_cleared(self):
        self.assertTrue(self.info["cleared"], "test 任务必须先清空基类地形，否则会继承 11 类")

    def test_mix_is_the_only_sub_terrain(self):
        self.assertEqual(list(self.props.keys()), ["mix"], f"只允许 mix，实测 {list(self.props)}")
        self.assertAlmostEqual(self.props["mix"], 1.0, places=9)

    def test_num_cols_twenty_and_num_rows_one(self):
        """20 条**并列**的 mix 道（列、沿世界 Y）× **唯一一行**难度（行、沿世界 X）。"""
        self.assertEqual(self.info["num_cols"], 20, "20 条并列的 mix 道（＝可同时评估的环境数上限）")
        self.assertEqual(self.info["num_rows"], 1, "只留唯一一行难度 ⇒ 世界 X 只有 8 m（旧方案 20 行）")
        self.assertEqual(chk.MIX_TEST_LANES_EXPECTED, 20)
        self.assertEqual(chk.MIX_TEST_LEVELS_EXPECTED, 1)

    def test_difficulty_range_is_exactly_pinned(self):
        """难度由 `difficulty_range = (0.70, 0.70)` 精确固定（＝旧"第 14 行"的名义难度）。"""
        self.assertEqual(self.info["difficulty_range"], (0.70, 0.70),
                         f"上下界必须同值（否则行内 U(0,1) 抖动会回来）：{self.info['difficulty_range']}")
        self.assertAlmostEqual(chk.MIX_TEST_DIFFICULTY_EXPECTED, 0.70, places=9)

    def test_num_cols_is_read_from_the_module_constant(self):
        """`num_cols`/`num_rows`/难度/间距乘子都写成可读常量 ⇒ 工具必须能求值模块常量。"""
        consts = chk.module_constants()
        self.assertEqual(consts["MIX_TEST_LANES"], 20)
        self.assertEqual(consts["MIX_TEST_LEVELS"], 1)
        self.assertAlmostEqual(consts["MIX_TEST_DIFFICULTY"], 0.70, places=9)
        # 2026-10-04（第四批）：乘子改成"占满整条道"的反算值（原来 1.0 = 训练默认值）
        # 2026-10-04（第五批）：瓦片 X 8 m → 20 m ⇒ 反算值 2.25 → **6.00**、图案末端 7.50 → **19.50**
        self.assertAlmostEqual(consts["MIX_TEST_PATTERN_SPACING_SCALE"], 6.00, places=9)
        self.assertAlmostEqual(consts["MIX_TEST_TAIL_MARGIN"], 0.50, places=9)
        self.assertAlmostEqual(consts["MIX_TEST_PATTERN_END_X"], 19.50, places=9)
        self.assertIs(consts["MIX_TEST_FILL_STRETCHED_GAPS"], True)
        # 第五批新增常量
        self.assertEqual(consts["MIX_TEST_TILE_SIZE"], (20.0, 4.0))
        self.assertAlmostEqual(consts["MIX_TEST_EPISODE_LENGTH_S"], 35.0, places=9)
        self.assertAlmostEqual(consts["MIX_TEST_EPISODE_LENGTH_MIN_S"], 25.0, places=9)
        self.assertAlmostEqual(consts["MIX_TEST_FORWARD_SPEED"], 1.0, places=9)
        self.assertNotIn("MIX_TEST_PINNED_LEVEL", consts,
                         "旧的'钉第 14 行'常量必须随子类一起删掉（难度改由 difficulty_range 固定）")
        # 字面量与常量两种写法都要能读（`literal` 先字面量、后常量命名空间）
        self.assertEqual(chk.literal(ast.parse("7", mode="eval").body, consts), 7)
        self.assertEqual(chk.literal(ast.parse("MIX_TEST_LANES", mode="eval").body, consts), 20)
        self.assertEqual(chk.literal(ast.parse("MIX_TEST_FILL_STRETCHED_GAPS", mode="eval").body, consts), True)

    # ------------------------------------------- 2026-10-04（第四/五批）：图案占满整条道
    def test_mix_call_kwargs_are_parsed(self):
        """工具要能从 `sub_terrains['mix'] = <Cfg>(...)` 里读出 `pattern_spacing_scale` 与
        `fill_stretched_gaps`（新字段是常量名，必须走模块常量求值）。"""
        calls = self.info["sub_terrain_calls"]
        self.assertEqual(list(calls.keys()), ["mix"])
        kwargs = calls["mix"]
        self.assertAlmostEqual(kwargs["pattern_spacing_scale"], 6.00, places=9)
        self.assertIs(kwargs["fill_stretched_gaps"], True)
        self.assertAlmostEqual(kwargs["pattern_start_x"], 0.30, places=9)
        self.assertAlmostEqual(kwargs["x_unit"], 0.02, places=9)

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

    def test_report_checks_the_pattern_fills_the_lane(self):
        """第四/五批：工具必须核算"图案末端 ≥ 19.0 m"、乘子 = 反算值、`fill_stretched_gaps=True`。"""
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            problems, lines = chk.mix_test_report()
        text = "\n".join(lines)
        self.assertEqual(problems, 0, text)
        self.assertIn("第四/五批检查", text)
        self.assertIn("pattern_spacing_scale", text)
        self.assertIn("19.50 m", text)
        self.assertIn("97.50", text)
        self.assertIn("6.15625", text, "上限（溢出保护）必须打印出来")
        self.assertIn("fill_stretched_gaps=True", text)
        self.assertIn("第五批检查：单块瓦片", text)
        self.assertIn("训练/play 侧**的 `terrain_generator.size` 仍是 8×4 m", text)
        self.assertIn("单局时长", text)
        self.assertIn("episode_length_s", text)
        self.assertIn("35 s", text)
        self.assertNotIn("❌", text)
        # 常量本身的期望值也写死在工具里（防止 cfg 常量被改错而工具沉默）
        self.assertAlmostEqual(chk.MIX_TEST_PATTERN_SPACING_SCALE_EXPECTED, 6.00, places=9)
        self.assertAlmostEqual(chk.MIX_TEST_PATTERN_END_MIN_X, 19.0, places=9)
        self.assertAlmostEqual(chk.MIX_PATTERN_END_UNITS, 160.0, places=9)

    def test_report_flags_a_wrong_spacing_scale(self):
        """负向对照：期望乘子被改成别的值时必须报 ❌（"6.00"是被检查的，不是打印而已）。"""
        import contextlib
        import io

        original = chk.MIX_TEST_PATTERN_SPACING_SCALE_EXPECTED
        chk.MIX_TEST_PATTERN_SPACING_SCALE_EXPECTED = 2.0
        try:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                problems, lines = chk.mix_test_report()
            text = "\n".join(lines)
            self.assertGreater(problems, 0, text)
            self.assertIn("应为 2.0", text)
        finally:
            chk.MIX_TEST_PATTERN_SPACING_SCALE_EXPECTED = original

    def test_report_flags_a_pattern_that_does_not_fill_the_lane(self):
        """负向对照：目标下界抬到 19.6 m 时，19.50 m 的图案必须被判"未占满整条道"。"""
        import contextlib
        import io

        original = chk.MIX_TEST_PATTERN_END_MIN_X
        chk.MIX_TEST_PATTERN_END_MIN_X = 19.6
        try:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                problems, lines = chk.mix_test_report()
            text = "\n".join(lines)
            self.assertGreater(problems, 0, text)
            self.assertIn("未占满整条道", text)
        finally:
            chk.MIX_TEST_PATTERN_END_MIN_X = original

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
            self.assertGreater(problems, 0, text)
            self.assertIn("`terrain_generator.size` 应为", text)
        finally:
            chk.MIX_TEST_TILE_SIZE_EXPECTED = original

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
            self.assertGreater(problems, 0, text)
            self.assertIn("训练/play 侧**的 `terrain_generator.size` 必须仍是", text)
        finally:
            chk.TRAIN_TILE_SIZE_EXPECTED = original

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
            self.assertGreater(problems, 0, text)
            self.assertIn("下界 40 s", text)
        finally:
            chk.MIX_TEST_EPISODE_LENGTH_MIN_S = original

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
            self.assertGreater(problems, 0, text)
            self.assertIn("未覆盖 `episode_length_s`", text)
        finally:
            chk.episode_length_s = original

    def test_report_flags_a_missing_fill_flag(self):
        """负向对照：`fill_stretched_gaps` 不是 True 时必须报 ❌（"铺成可走面"同样被检查）。"""
        import contextlib
        import io

        original = chk.scene_overrides

        def patched(class_name=chk.TRAIN_CLASS):
            info = original(class_name)
            if class_name == chk.MIX_TEST_CLASS:
                info["sub_terrain_calls"]["mix"]["fill_stretched_gaps"] = False
            return info

        chk.scene_overrides = patched
        try:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                problems, lines = chk.mix_test_report()
            text = "\n".join(lines)
            self.assertGreater(problems, 0, text)
            self.assertIn("fill_stretched_gaps=True", text)
            self.assertIn("❌", text)
        finally:
            chk.scene_overrides = original

    def test_mix_gets_all_columns(self):
        """只 mix 一类 ⇒ 20 列**全部**是 mix（每道 1 列）。"""
        self.assertEqual(chk.allocate([("mix", 1.0)], 20), ["mix"] * 20)

    def test_report_shows_twenty_mix_columns_and_one_row(self):
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            chk.mix_test_report()
        text = buffer.getvalue()
        self.assertIn("num_cols=20，共 20 列", text)
        self.assertIn("mix                     20 列", text)
        self.assertIn("网格 **20 道 × 1 难度行**", text)
        self.assertIn("世界 20 m(X) × 80 m(Y)", text)

    def test_report_flags_a_wrong_lane_count(self):
        """负向对照：把期望道数改成别的值，判定必须报 ❌（证明"20 列"是被**检查**的，不是打印而已）。"""
        import contextlib
        import io

        original = chk.MIX_TEST_LANES_EXPECTED
        chk.MIX_TEST_LANES_EXPECTED = 1
        try:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                problems, lines = chk.mix_test_report()
            text = "\n".join(lines)
            self.assertGreater(problems, 0, text)
            self.assertIn("❌", text)
        finally:
            chk.MIX_TEST_LANES_EXPECTED = original

    def test_report_flags_a_wrong_row_count(self):
        """负向对照：期望行数被改错时必须报 ❌（"1 行"同样是被检查的）。"""
        import contextlib
        import io

        original = chk.MIX_TEST_LEVELS_EXPECTED
        chk.MIX_TEST_LEVELS_EXPECTED = 20
        try:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                problems, lines = chk.mix_test_report()
            text = "\n".join(lines)
            self.assertGreater(problems, 0, text)
            self.assertIn("num_rows 应为 20", text)
        finally:
            chk.MIX_TEST_LEVELS_EXPECTED = original

    def test_report_flags_a_wrong_difficulty(self):
        """负向对照：期望难度被改错时必须报 ❌（难度固定这一条同样是被检查的）。"""
        import contextlib
        import io

        original = chk.MIX_TEST_DIFFICULTY_EXPECTED
        chk.MIX_TEST_DIFFICULTY_EXPECTED = 0.5
        try:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                problems, lines = chk.mix_test_report()
            text = "\n".join(lines)
            self.assertGreater(problems, 0, text)
            self.assertIn("difficulty_range 应为", text)
        finally:
            chk.MIX_TEST_DIFFICULTY_EXPECTED = original

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
        self.assertIn("mix", text)
        self.assertNotIn("❌", text)
        # 每一条被引用的"本场景不存在"的地形名都要**逐条**标注
        absent = sorted({n for refs in chk.MASKED_NAMES.values() for n in refs} - {"mix"})
        self.assertTrue(absent, "本测试的前提是掩码引用了 mix 之外的地形名")
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
                if name != "mix":
                    self.assertIn(f"{key} → {name}", text)

    def test_forward_only_covers_the_only_terrain(self):
        fwd = chk.forward_only_names(chk.MIX_TEST_CLASS) or chk.forward_only_names(chk.TRAIN_CLASS)
        self.assertIn("mix", fwd)

    # ------------------------------------------------- 原判据不放宽（负向对照）
    def test_original_criteria_still_fail_on_absent_names(self):
        """负向对照：把"只有 mix"的场景丢给**原判据**，必须仍然报错（证明没被放宽）。"""
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            problems = chk.check_mask_columns({"mix": 1}, 1)
        self.assertGreater(problems, 0, "原判据必须继续对'掩码引用的地形名不存在'报错")
        self.assertIn("不在 sub_terrains 里", buffer.getvalue())

    def test_original_criteria_still_fail_on_zero_columns(self):
        """负向对照：存在但 0 列时原判据也必须报错。"""
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            problems = chk.check_mask_columns({"mix": 1, "boxes": 0, "gap": 0, "flat": 0}, 1)
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
        """mix-test 类里也写了 `sub_terrains["mix"]`；默认任务必须仍按训练值解析（比例 0.10、40 列）。"""
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
