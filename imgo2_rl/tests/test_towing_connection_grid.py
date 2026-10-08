"""Offline tests for the deterministic towing scene grid (``mdp/connection_grid.py``).

用户 2026-10-08 指定：20 列 = 弹性绳 8 / 刚体 8 / 普通绳 4，20 行 = 长度 0.4–0.8 m；
弹性绳只在列上做弹性区分（4 档 k/c，每档 2 列）。本文件是纯标准库检查，不需要仿真器。
"""

import importlib.util
import math
from pathlib import Path
import sys
import unittest


RL = Path(__file__).resolve().parents[1]
MDP = RL / "source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/mdp"
sys.path.insert(0, str(MDP))          # 让 connection_grid 的顶层回退导入可用（如需要）

# 名义参数：小车 5 kg（质量域下界）+ 四轮滚动惯量折算 4I/r²，机器人整机 12.6996 kg。
CART_EFFECTIVE_MASS_KG = 5.0 + 4.0 * 0.00128 / 0.08 ** 2
ROBOT_MASS_KG = 12.6996
PHYSICS_DT_S = 0.005
MAX_ATTACHMENT_HEIGHT_DIFF_M = 0.17     # 机器人 base 0.35 − 小车 base_link 0.18


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


grid = load("towing_connection_grid_test", MDP / "connection_grid.py")
rope_model = load("towing_rope_model_grid_test", MDP / "rope_model.py")


class GridShapeTests(unittest.TestCase):
    def test_dimensions_and_row_lengths(self):
        self.assertEqual((grid.COLUMNS, grid.ROWS, grid.GRID_SIZE), (20, 20, 400))
        lengths = [grid.row_length(row) for row in range(grid.ROWS)]
        self.assertAlmostEqual(lengths[0], 0.4, places=12)
        self.assertAlmostEqual(lengths[-1], 0.8, places=12)
        # 20 档等距（含两端），步长 = 0.4 / 19
        step = (grid.LENGTH_MAX_M - grid.LENGTH_MIN_M) / (grid.ROWS - 1)
        for previous, current in zip(lengths, lengths[1:]):
            self.assertAlmostEqual(current - previous, step, places=12)
        # 每行长度互不相同（长度就是行索引）
        self.assertEqual(len(set(lengths)), grid.ROWS)

    def test_column_split_is_eight_eight_four(self):
        names = [grid.column_spec(column)[0] for column in range(grid.COLUMNS)]
        self.assertEqual(names.count("compliant"), 8)
        self.assertEqual(names.count("rigid"), 8)
        self.assertEqual(names.count("inextensible"), 4)
        # 列布局顺序：弹性（0-7）→ 刚体（8-15）→ 普通绳（16-19）
        self.assertEqual(names[:8], ["compliant"] * 8)
        self.assertEqual(names[8:16], ["rigid"] * 8)
        self.assertEqual(names[16:], ["inextensible"] * 4)

    def test_elasticity_is_differentiated_only_on_columns(self):
        """弹性绳 4 档 k/c、每档 2 列；刚体与普通绳的列是同参数重复列。"""
        self.assertEqual(len(grid.ELASTIC_KC), 4)
        elastic = [grid.column_spec(column)[1:] for column in range(8)]
        # 每档恰好 2 列、按给定顺序排列
        for level, (stiffness, damping) in enumerate(grid.ELASTIC_KC):
            self.assertEqual(elastic[2 * level], (stiffness, damping))
            self.assertEqual(elastic[2 * level + 1], (stiffness, damping))
        # 四档刚度严格递增，且都是有限正数
        stiffnesses = [value[0] for value in grid.ELASTIC_KC]
        self.assertEqual(stiffnesses, sorted(stiffnesses))
        self.assertEqual(len(set(stiffnesses)), 4)
        for stiffness, damping in grid.ELASTIC_KC:
            self.assertTrue(math.isfinite(stiffness) and stiffness > 0.0)
            self.assertTrue(math.isfinite(damping) and damping > 0.0)
        # 刚体/普通绳列不带 k/c
        for column in range(8, grid.COLUMNS):
            self.assertEqual(grid.column_spec(column)[1:], (None, None))

    def test_elastic_levels_stay_inside_the_explicit_spring_stability_limit(self):
        """最硬的弹性档在**最坏质量**（5 kg 小车）下仍需留在 dt=5 ms 的稳定域内。

        显式弹簧的稳定条件（弹簧-阻尼系统）：`dt < 2 / (ω(ζ + √(ζ²+1)))`，其中
        `ω = √(k/μ)`、`ζ = c / (2√(kμ))`、μ 为绳两端折合质量。文档结论是「再硬应改用约束
        模型或子步」，所以这里把边界锁进测试，避免以后悄悄把 k 提上去。
        """
        reduced_mass = 1.0 / (1.0 / ROBOT_MASS_KG + 1.0 / CART_EFFECTIVE_MASS_KG)
        for stiffness, damping in grid.ELASTIC_KC:
            omega = math.sqrt(stiffness / reduced_mass)
            zeta = damping / (2.0 * math.sqrt(stiffness * reduced_mass))
            dt_limit = 2.0 / (omega * (zeta + math.sqrt(zeta * zeta + 1.0)))
            self.assertGreater(dt_limit, PHYSICS_DT_S,
                               f"k={stiffness} 的步长上限 {dt_limit*1000:.2f} ms "
                               f"不高于物理步长 {PHYSICS_DT_S*1000:.1f} ms")

    def test_model_index_matches_connection_models_order(self):
        expected = {name: index for index, name in enumerate(rope_model.CONNECTION_MODELS)}
        self.assertEqual(grid.MODEL_INDEX, expected)


class GridSpawnTests(unittest.TestCase):
    def test_initial_distance_follows_type(self):
        for row in (0, grid.ROWS // 2, grid.ROWS - 1):
            length = grid.row_length(row)
            self.assertAlmostEqual(
                grid.initial_attachment_distance("rigid", length), length, places=12)
            for rope in ("compliant", "inextensible"):
                self.assertAlmostEqual(
                    grid.initial_attachment_distance(rope, length),
                    grid.SLACK_RATIO * length, places=12)

    def test_every_row_can_be_placed_above_the_attachment_height_difference(self):
        """三类、所有行的目标挂点距都必须大于两挂点高差，否则水平摆放无解。"""
        for model_name in ("compliant", "inextensible", "rigid"):
            for row in range(grid.ROWS):
                target = grid.initial_attachment_distance(model_name, grid.row_length(row))
                self.assertGreater(target, MAX_ATTACHMENT_HEIGHT_DIFF_M,
                                   f"{model_name} row={row} target={target}")
                # 勾股解应给出正的水平间距（与 upper_mdp 用的是同一个函数）
                horizontal = rope_model.attachment_horizontal_gap(
                    target, delta_z=MAX_ATTACHMENT_HEIGHT_DIFF_M)
                self.assertGreater(horizontal, 0.0)
                self.assertAlmostEqual(
                    math.hypot(horizontal, MAX_ATTACHMENT_HEIGHT_DIFF_M), target, places=12)


class GridIndexTests(unittest.TestCase):
    def test_env_spec_is_row_major_and_covers_the_grid_exactly_once(self):
        specs = [grid.env_spec(index) for index in range(grid.GRID_SIZE)]
        self.assertEqual(len({spec["grid_index"] for spec in specs}), grid.GRID_SIZE)
        # 行优先：col = i % 20、row = i // 20
        for index in (0, 1, 19, 20, 21, 399):
            spec = specs[index]
            self.assertEqual(spec["column"], index % grid.COLUMNS)
            self.assertEqual(spec["row"], index // grid.COLUMNS)
        # 同一行（同一长度）里三类都在；同一列（同一类型/弹性档）长度随行变化
        first_row = specs[:grid.COLUMNS]
        self.assertEqual(len({spec["length"] for spec in first_row}), 1)
        self.assertEqual({spec["model_name"] for spec in first_row},
                         {"compliant", "rigid", "inextensible"})
        column_zero = [specs[row * grid.COLUMNS] for row in range(grid.ROWS)]
        self.assertEqual({spec["model_name"] for spec in column_zero}, {"compliant"})
        self.assertEqual(len({spec["length"] for spec in column_zero}), grid.ROWS)

    def test_full_grid_and_wrapping(self):
        self.assertTrue(grid.is_full_grid(grid.GRID_SIZE))
        self.assertTrue(grid.is_full_grid(2 * grid.GRID_SIZE))
        self.assertFalse(grid.is_full_grid(4))          # 冒烟规模只覆盖前缀
        self.assertFalse(grid.is_full_grid(0))
        # 超出网格时取模：800 环境 = 两份相同网格
        wrapped = grid.env_spec(grid.GRID_SIZE)
        self.assertEqual(wrapped["grid_index"], grid.env_spec(0)["grid_index"])
        self.assertEqual(wrapped["model_name"], grid.env_spec(0)["model_name"])
        self.assertEqual(wrapped["length"], grid.env_spec(0)["length"])

    def test_rejects_out_of_range_arguments(self):
        for bad in (-1,):
            with self.assertRaises(ValueError):
                grid.env_spec(bad)
        with self.assertRaises(ValueError):
            grid.row_length(grid.ROWS)
        with self.assertRaises(ValueError):
            grid.column_spec(grid.COLUMNS)
        with self.assertRaises(ValueError):
            grid.initial_attachment_distance("rubber_band", 0.8)


if __name__ == "__main__":
    unittest.main()
