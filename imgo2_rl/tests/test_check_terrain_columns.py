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
`TestMixTestTerrainColumns` = test 任务的**单独**断言（清空后只有 mix、**20 列**（20 条并排的 mix 道）
× 20 行难度、工具给出明确的"预期"标注），外加**负向对照**证明原判据没有被放宽。

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
    `num_cols=20`（＝`MIX_TEST_LANES`：20 条**并排**的 mix 道）+ `num_rows=20`，并把"其它地形名 0 列"
    **逐条打印成预期**。
    2026-10-04（第二批，用户："地形不要按照列排，放在行里面"）：列数 **1 → 20**（20 条道沿世界 Y
    并排）⇒ 本类的期望值同步改成 20；同时加"模块级可读常量能被工具解析"的断言。
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

    def test_num_cols_twenty_and_num_rows_twenty(self):
        """20 条**并排**的 mix 道（列、沿世界 Y）× 20 档难度（行、沿世界 X）。"""
        self.assertEqual(self.info["num_cols"], 20, "20 条并排的 mix 道（＝可同时评估的环境数上限）")
        self.assertEqual(self.info["num_rows"], 20, "行＝难度 level（沿世界 X）；固定用第 14 行")
        self.assertEqual(chk.MIX_TEST_LANES_EXPECTED, 20)
        self.assertEqual(chk.MIX_TEST_LEVELS_EXPECTED, 20)

    def test_num_cols_is_read_from_the_module_constant(self):
        """`num_cols` 现在写成可读常量 `MIX_TEST_LANES` ⇒ 工具必须能求值模块常量。"""
        consts = chk.module_constants()
        self.assertEqual(consts["MIX_TEST_LANES"], 20)
        self.assertEqual(consts["MIX_TEST_LEVELS"], 20)
        self.assertEqual(consts["MIX_TEST_PINNED_LEVEL"], 14)
        self.assertAlmostEqual(consts["MIX_TEST_PATTERN_SPACING_SCALE"], 2.0, places=9)
        # 字面量与常量两种写法都要能读（`literal` 先字面量、后常量命名空间）
        self.assertEqual(chk.literal(ast.parse("7", mode="eval").body, consts), 7)
        self.assertEqual(chk.literal(ast.parse("MIX_TEST_LANES", mode="eval").body, consts), 20)

    def test_mix_gets_all_columns(self):
        """只 mix 一类 ⇒ 20 列**全部**是 mix（每道 1 列）。"""
        self.assertEqual(chk.allocate([("mix", 1.0)], 20), ["mix"] * 20)

    def test_report_shows_twenty_mix_columns(self):
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            chk.mix_test_report()
        text = buffer.getvalue()
        self.assertIn("num_cols=20，共 20 列", text)
        self.assertIn("mix                     20 列", text)

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
