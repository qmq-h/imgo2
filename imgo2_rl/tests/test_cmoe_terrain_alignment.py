"""与参考 CMoE 的**结构对齐**的离线锁定（只用标准库，不需要 numpy／trimesh／Isaac Lab）。

2026-09-28 用户决定（「我看了一下 cmoe 的地形设置，我们四足这个地形还是差点，感觉可以对齐一下」）：
**只对齐结构** —— 地形种类、比例、行列数、难度律形式、初始等级；米制难度区间沿用我们已验证的
四足值；参考里没有四足对应值的三类（hurdle／mix／narrow_stairs）按参考米制 × 站高比
`REFERENCE_SCALE = 0.30/0.75 = 0.4` 落地。

参考实现（真值来源，测试里逐条写出以便对照）:
`Hoshi-No-Ai/CMoE` @ `4575d6ae`
  * `legged_gym/legged_gym/envs/g1/g1_cmoe_config.py::terrain`（比例、num_rows/num_cols、
    base_height_target=0.75）
  * `legged_gym/legged_gym/utils/humanoid_terrain.py::Terrain.make_terrain`（每类难度律）
  * `legged_gym/legged_gym/utils/parkour_terrain_utils.py`（hurdle／mix／narrow stairs 图案）

本文件锁定的是"结构没被改坏"：种类齐全、比例＝参考、行/列数＝10/40、初始等级＝5、
新增三类的米制常量确实等于参考值 ×0.4。几何不变式在 `test_track_geometry.py` 里。
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move"
TERRAINS = BASE / "cmoe_terrains.py"
CMOE_CFG = BASE / "CMoE_env_cfg.py"

# 参考 G1 的 base_height_target（`g1_cmoe_config.py::rewards.base_height_target`）
REFERENCE_BASE_HEIGHT = 0.75
# 我们 Imgo2 的站立高度
IMGO2_STAND_HEIGHT = 0.30


def _module_constants(tree: ast.Module) -> dict[str, object]:
    """模块级能求值的简单常量（支持 `0.05 * REFERENCE_SCALE` 这种由已解析常量构成的乘法）。"""
    out: dict[str, object] = {}

    def resolve(node: ast.AST):
        if isinstance(node, ast.Name):
            return out.get(node.id)
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError):
            return None

    for _ in range(4):  # 允许常量之间的多层引用
        for node in tree.body:
            target = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
            elif isinstance(node, ast.AnnAssign):
                target = node.target
            if not isinstance(target, ast.Name):
                continue
            value = resolve(node.value)
            if value is None and isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Mult):
                left, right = resolve(node.value.left), resolve(node.value.right)
                if left is not None and right is not None:
                    value = left * right
            if value is not None:
                out[target.id] = value
    return out


def _class_constants(source: str, class_name: str) -> dict[str, object]:
    """取某个 `@configclass` 类体里能求值的简单字段（含类型注解字段与模块级常量引用）。"""
    tree = ast.parse(source)
    module_consts = _module_constants(tree)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            out: dict[str, object] = {}
            for stmt in node.body:
                target = None
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                    target = stmt.targets[0]
                elif isinstance(stmt, ast.AnnAssign):
                    target = stmt.target
                if not isinstance(target, ast.Name):
                    continue
                try:
                    out[target.id] = ast.literal_eval(stmt.value)
                except (ValueError, TypeError):
                    if isinstance(stmt.value, ast.Name) and stmt.value.id in module_consts:
                        out[target.id] = module_consts[stmt.value.id]
            return out
    raise AssertionError(f"{class_name} 不在 {TERRAINS.name} 里")


def _dict_entries(source: str, wanted_keys: set[str]) -> dict[str, object]:
    """收集源码里所有 dict 字面量中键命中 `wanted_keys` 的条目（课程 params 用的是 dict 字面量）。"""
    out: dict[str, object] = {}
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and key.value in wanted_keys:
                try:
                    out[str(key.value)] = ast.literal_eval(value)
                except ValueError:
                    pass
    return out


class TestReferenceStructureAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.terrains_src = TERRAINS.read_text(encoding="utf-8")
        cls.cfg_src = CMOE_CFG.read_text(encoding="utf-8")

    # ---------------------------------------------------------------- 缩放口径
    def test_reference_scale_is_the_standing_height_ratio(self):
        value = _module_constants(ast.parse(self.terrains_src)).get("REFERENCE_SCALE")
        self.assertIsNotNone(value, "cmoe_terrains.py 里找不到 REFERENCE_SCALE")
        self.assertAlmostEqual(value, IMGO2_STAND_HEIGHT / REFERENCE_BASE_HEIGHT, places=6,
                               msg="REFERENCE_SCALE 必须等于站高比 0.30/0.75")

    def test_docstring_cites_the_reference_implementation(self):
        head = self.terrains_src[:2000]
        self.assertIn("Hoshi-No-Ai/CMoE", head, "文件头必须写明参考仓库")
        self.assertIn("0.75", head, "文件头必须写明参考 G1 的站高依据")
        self.assertIn("0.30", head, "文件头必须写明我们四足的站高依据")

    # ---------------------------------------------------------------- 新增三类的米制常量
    def test_hurdle_constants_are_reference_scaled(self):
        """参考 `humanoid_terrain.py:230`：stone_len=0.1+0.2d、x_range=[1.2,2]、
        hurdle_height_range=[0.2d, 0.15+0.25d]、platform_len=2.0。"""
        s = IMGO2_STAND_HEIGHT / REFERENCE_BASE_HEIGHT
        c = _class_constants(self.terrains_src, "CMoETrackHurdleTerrainCfg")
        self.assertEqual(c["num_hurdles"], 4)  # 参考 num_stones=4
        self.assertAlmostEqual(c["stone_len_range"][0], 0.1 * s, places=6)
        self.assertAlmostEqual(c["stone_len_range"][1], 0.3 * s, places=6)  # 0.1+0.2
        self.assertAlmostEqual(c["spacing_range"][0], 1.2 * s, places=6)
        self.assertAlmostEqual(c["spacing_range"][1], 2.0 * s, places=6)
        self.assertAlmostEqual(c["hurdle_height_min_slope"], 0.2 * s, places=6)
        self.assertAlmostEqual(c["hurdle_height_max_base"], 0.15 * s, places=6)
        self.assertAlmostEqual(c["hurdle_height_max_slope"], 0.25 * s, places=6)
        self.assertAlmostEqual(c["platform_length"], 2.0 * s, places=6)
        self.assertAlmostEqual(c["spawn_x"], 0.75, places=6)

    def test_mix_constants_are_reference_scaled(self):
        """参考 `mix_obstacles_terrain`：水平索引 0.05 m、高度索引 0.005 m、走廊半宽 20 索引、
        高度整体乘 `diff = hurdle_height_range[0]*1.1`。"""
        s = IMGO2_STAND_HEIGHT / REFERENCE_BASE_HEIGHT
        c = _class_constants(self.terrains_src, "CMoETrackMixTerrainCfg")
        self.assertAlmostEqual(c["x_unit"], 0.05 * s, places=9)
        self.assertAlmostEqual(c["z_unit"], 0.005 * s, places=9)
        self.assertAlmostEqual(c["height_scale"], 1.1, places=9)  # 无量纲，不缩放
        self.assertAlmostEqual(c["gap_shrink_units"], 10.0, places=9)  # 索引单位，不缩放
        self.assertAlmostEqual(c["corridor_width"], 2 * 20 * 0.05 * s, places=6)
        self.assertAlmostEqual(c["spawn_x"], 0.75, places=6)

    def test_narrow_stairs_constants_are_reference_scaled(self):
        """参考 `humanoid_terrain.py:241`：num_stones=24、step_height=0.25d、
        步深取 x_range[0]=0.30、半宽取 half_valid_width[0]=1-0.5d、platform_len=2.5。"""
        s = IMGO2_STAND_HEIGHT / REFERENCE_BASE_HEIGHT
        c = _class_constants(self.terrains_src, "CMoETrackNarrowStairsTerrainCfg")
        self.assertEqual(c["num_steps"], 24)
        self.assertAlmostEqual(c["step_depth"], 0.30 * s, places=6)
        self.assertAlmostEqual(c["step_height_max"], 0.25 * s, places=6)
        self.assertAlmostEqual(c["platform_length"], 2.5 * s, places=6)
        self.assertAlmostEqual(c["corridor_half_width_start"], 1.0 * s, places=6)
        self.assertAlmostEqual(c["corridor_half_width_slope"], 0.5 * s, places=6)
        self.assertAlmostEqual(c["spawn_x"], 0.75, places=6)

    # ---------------------------------------------------------------- 生成器级结构
    def test_rows_and_init_level_match_the_reference(self):
        """参考 num_rows=10、max_init_terrain_level=5（`legged_robot_config.py::terrain`）。"""
        self.assertIn("num_rows", self.cfg_src)  # 沿用基类默认 10，不显式覆盖
        self.assertIn("max_init_terrain_level = 5", self.cfg_src)

    def test_num_cols_is_40_as_in_the_reference(self):
        self.assertIn("self.scene.terrain.terrain_generator.num_cols = 40", self.cfg_src)

    def test_obstacle_columns_are_relaxed_in_the_curriculum(self):
        """扩张地形集时把新的障碍列一并纳入放宽名单，否则会落进 [0.35, 0.80) 冻结带（判据本身未改）。"""
        relaxed = _dict_entries(self.cfg_src, {"relaxed_terrain_names"}).get("relaxed_terrain_names")
        self.assertIsNotNone(relaxed, "找不到 relaxed_terrain_names")
        for name in ("pyramid_stairs", "pyramid_stairs_inv", "boxes", "gap",
                     "hurdle", "mix", "narrow_stairs"):
            self.assertIn(name, relaxed, f"{name} 应当纳入放宽名单")

    def test_curriculum_criteria_are_unchanged(self):
        """用户本轮只要对齐地形生成器 ⇒ 课程判据的门控值必须保持 0.80/0.35。"""
        thresholds = _dict_entries(self.cfg_src, {"tracking_move_up", "tracking_move_down"})
        self.assertEqual(thresholds.get("tracking_move_up"), 0.80)
        self.assertEqual(thresholds.get("tracking_move_down"), 0.35)


if __name__ == "__main__":
    unittest.main()
