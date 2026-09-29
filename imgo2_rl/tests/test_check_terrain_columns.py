"""`check_terrain_columns.py` 的回归测试（只用标准库）。

锁住 **2026-09-28 与参考 CMoE 结构对齐后**的地形列分配（用户：「我看了一下 cmoe 的地形设置，
我们四足这个地形还是差点，感觉可以对齐一下」）：

* 比例按参考 `g1_cmoe_config.py::terrain.terrain_dict`（+ 我们保留的 `random_rough`／`flat`），
  `num_cols` 训练与 play 都是 **40**（参考值；原来是 20／10）；
* 11 类地形**每一类都 ≥1 列** —— 0 列时引用它的掩码会静默失效、不报错，这正是本工具存在的理由
  （旧配置在 `num_cols=10` 时 `hf_pyramid_slope_inv` 就是 0 列）。

逐项对照见 `docs/cmoe_terrain_alignment_2026-09-28.md`。
"""

from __future__ import annotations

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


if __name__ == "__main__":
    sys.exit(unittest.main())
