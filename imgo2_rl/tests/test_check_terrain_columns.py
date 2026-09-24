"""`check_terrain_columns.py` 的回归测试（只用标准库）。

锁住 2026-09-24 定下的地形列分配：`boxes` 只留 1 列、空出的 0.10 给 `gap`（4 → 6 列）、
其余列数量不变，且**所有奖励掩码引用的地形名在训练与 play 两套 `num_cols` 下都 ≥1 列**
（0 列时掩码会静默失效、不报错 —— 这正是本工具存在的理由）。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "scripts" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import check_terrain_columns as chk  # noqa: E402


def _counts(num_cols: int) -> dict[str, int]:
    base = chk.base_sub_terrains()
    props, _ = chk.cmoe_overrides()
    merged = [(name, props.get(name, default)) for name, default in base]
    merged += [(name, value) for name, value in props.items() if name not in [n for n, _ in merged]]
    return chk.report(merged, num_cols, f"test/{num_cols}")


class TestTerrainColumns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = _counts(20)
        cls.play = _counts(10)

    def test_train_total(self):
        self.assertEqual(sum(self.train.values()), 20)

    def test_boxes_gets_exactly_one_column(self):
        self.assertEqual(self.train["boxes"], 1, "用户要求 boxes 只给一个")
        self.assertEqual(self.play["boxes"], 1)

    def test_freed_share_went_to_gap(self):
        self.assertEqual(self.train["gap"], 6)
        self.assertEqual(self.play["gap"], 3)

    def test_other_columns_unchanged(self):
        # 2026-09-24 之前：stairs 4 / stairs_inv 2 / rough 2 / slope 2 / slope_inv 1 / flat 2
        self.assertEqual(self.train["pyramid_stairs"], 4)
        self.assertEqual(self.train["pyramid_stairs_inv"], 2)
        self.assertEqual(self.train["random_rough"], 2)
        self.assertEqual(self.train["hf_pyramid_slope"], 2)
        self.assertEqual(self.train["hf_pyramid_slope_inv"], 1)
        self.assertEqual(self.train["flat"], 2)

    def test_every_masked_terrain_name_has_columns(self):
        for num_cols, counts in ((20, self.train), (10, self.play)):
            for key, names in chk.MASKED_NAMES.items():
                for name in names:
                    self.assertIn(name, counts, f"{key} 引用了不存在的地形名 {name}")
                    self.assertGreater(counts[name], 0,
                                       f"num_cols={num_cols} 时 {key} 引用的 {name} 是 0 列 ⇒ 掩码静默失效")


if __name__ == "__main__":
    sys.exit(unittest.main())
