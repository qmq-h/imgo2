"""Speed-bump trapezoid geometry, mixed-grid wiring and tensor equivalence, without Isaac Lab.

2026-10-16 口径（用户）：地面是**一张 50 列的混合网格** —— 列 0–19 平地、20–29 上/下坡 5°、
30–39 上/下坡 10°、**列 40–49 减速带块**（10 列地形完全相同：一条底 300 / 顶 100 / 高 50 mm
的等腰梯形凸条，居中在剖面区 x = 5.625 m；连接类型照 5° 坡块同构 4 弹性 / 4 杆 / 2 低弹）。
本文件核对：梯形分段与折点、横向收零、闭合实体网格（无退化面、绕向朝外、面内高度误差）、
精确弧长、张量↔标量等价、非法输入，以及混合网格的分派与任务/测试台/分段统计的接线。
"""
from collections import Counter
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
MDP = ROOT / 'imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/mdp'
sys.path.insert(0, str(MDP))
import connection_grid as grid
import speedbump_geometry as bump
import mixed_geometry as mixed
import slope_geometry as hill
import wave_geometry as wave
import wave_torch


class SpeedBumpGeometryTests(unittest.TestCase):
    def test_trapezoid_layout_and_flat_regions(self):
        # 尺寸与位置（用户口径：底 300 / 顶 100 / 高 50 mm，居中在剖面区）
        self.assertEqual((bump.BUMP_BOTTOM_M, bump.BUMP_TOP_M, bump.BUMP_HEIGHT_M),
                         (0.30, 0.10, 0.05))
        self.assertAlmostEqual(bump.RAMP_RUN_M, 0.10, places=12)
        self.assertAlmostEqual(bump.RAMP_DEG, 26.565051177077994, places=9)
        self.assertAlmostEqual(bump.BUMP_CENTER_X_M, 5.625, places=12)
        self.assertAlmostEqual(bump.BUMP_START_M, 5.475, places=12)
        self.assertAlmostEqual(bump.BUMP_TOP_START_M, 5.575, places=12)
        self.assertAlmostEqual(bump.BUMP_TOP_END_M, 5.675, places=12)
        self.assertAlmostEqual(bump.BUMP_END_M, 5.775, places=12)
        # 凸条整条落在剖面区（2.25–9.0）内，出生平地与出口平地（含 10 m STOP 点）无凸起
        self.assertGreater(bump.BUMP_START_M, bump.START_M)
        self.assertLess(bump.BUMP_END_M, bump.END_M)
        for grade in (0.0, 5.0, 10.0, bump.RAMP_DEG):
            for x in (-2.25, 0.0, 2.25, 5.4, bump.BUMP_START_M, bump.BUMP_END_M,
                      5.8, 7.0, bump.END_M, 10.0, 12.75):
                self.assertEqual(bump.profile_height(grade, x), 0.0, msg=f"grade={grade} x={x}")
                self.assertEqual(bump.profile_slope_degrees(grade, x), 0.0,
                                 msg=f"grade={grade} x={x}")

    def test_trapezoid_segments_heights_and_slopes(self):
        for grade in (0.0, 5.0, 10.0, bump.RAMP_DEG):
            # 上坡段：斜坡起点 0 → 坡顶左端 50 mm，线性；中点 25 mm
            self.assertAlmostEqual(bump.profile_height(grade, bump.BUMP_TOP_START_M),
                                   bump.BUMP_HEIGHT_M, places=12)
            self.assertAlmostEqual(bump.profile_height(grade, bump.BUMP_START_M + 0.05),
                                   bump.BUMP_HEIGHT_M / 2.0, places=12)
            # 平顶段：整段 50 mm
            for x in (bump.BUMP_TOP_START_M, bump.BUMP_CENTER_X_M, bump.BUMP_TOP_END_M):
                self.assertAlmostEqual(bump.profile_height(grade, x), bump.BUMP_HEIGHT_M,
                                       places=12)
                self.assertAlmostEqual(bump.profile_slope_degrees(grade, x), 0.0, places=12)
            # 下坡段：坡顶右端 50 mm → 斜坡终点 0；中点 25 mm
            self.assertAlmostEqual(bump.profile_height(grade, bump.BUMP_TOP_END_M + 0.05),
                                   bump.BUMP_HEIGHT_M / 2.0, places=12)
            # 坡度：上坡 +atan(0.05/0.1)、下坡 −、平顶与凸条外 0
            self.assertAlmostEqual(bump.profile_slope_degrees(grade, bump.BUMP_START_M + 0.05),
                                   bump.RAMP_DEG, places=9)
            self.assertAlmostEqual(bump.profile_slope_degrees(grade, bump.BUMP_TOP_END_M + 0.05),
                                   -bump.RAMP_DEG, places=9)
            # 与档位无关（几何固定）：四个档位给出同一组数值
            self.assertEqual(bump.profile_height(grade, bump.BUMP_CENTER_X_M),
                             bump.profile_height(bump.RAMP_DEG, bump.BUMP_CENTER_X_M))
        # 侧坡就是 connection_grid 报出的该列 `slope_degrees`，且 = atan(高/单侧跑动)
        self.assertAlmostEqual(
            math.tan(math.radians(bump.RAMP_DEG)), bump.BUMP_HEIGHT_M / bump.RAMP_RUN_M,
            places=12)
        self.assertAlmostEqual(grid.SPEEDBUMP_RAMP_DEG, bump.RAMP_DEG, places=12)
        self.assertAlmostEqual(grid.slope_degrees(grid.SPEEDBUMP_START_COLUMN, 0), bump.RAMP_DEG,
                               places=12)

    def test_lateral_crown_full_height_then_taper_to_zero(self):
        # 中央 ±2 m 满高、外侧 1 m 内 cos² 收零，±3 m 处精确为 0（相邻列接缝无落差）
        self.assertEqual(bump.lateral_factor(0.0), 1.0)
        for y in (-2.0, -1.0, 0.5, 2.0):
            self.assertEqual(bump.lateral_factor(y), 1.0)
        for y in (-3.0, 3.0, -4.0, 4.0):
            self.assertEqual(bump.lateral_factor(y), 0.0)
        self.assertAlmostEqual(bump.lateral_factor(2.5), 0.5, places=12)
        self.assertLess(bump.lateral_factor(2.9), bump.lateral_factor(2.1))
        self.assertAlmostEqual(bump.profile_height(bump.RAMP_DEG, bump.BUMP_CENTER_X_M, 2.0),
                               bump.BUMP_HEIGHT_M, places=12)
        self.assertEqual(bump.profile_height(bump.RAMP_DEG, bump.BUMP_CENTER_X_M, 3.0), 0.0)
        # 横向收零后纵向坡度也按同一因子变小
        self.assertLess(bump.profile_slope_degrees(bump.RAMP_DEG, bump.BUMP_START_M + 0.05, 2.5),
                        bump.RAMP_DEG)

    def test_arc_length_is_exact_and_conservative(self):
        ramp = math.hypot(bump.RAMP_RUN_M, bump.BUMP_HEIGHT_M)
        extra = 2.0 * ramp + bump.BUMP_TOP_M - bump.BUMP_BOTTOM_M
        self.assertAlmostEqual(extra, 0.023606797749978984, places=12)
        dx = 10.0 / 20000
        for grade in (0.0, 5.0, 10.0, bump.RAMP_DEG):
            actual = sum(dx * math.sqrt(1 + math.tan(math.radians(
                bump.profile_slope_degrees(grade, (index + .5) * dx))) ** 2)
                for index in range(20000))
            bound = bump.arc_length_upper_bound(grade, 10.0)
            self.assertGreaterEqual(bound + 1e-9, actual)
            self.assertAlmostEqual(bound, actual, places=5)
            self.assertAlmostEqual(bound, 10.0 + extra, places=12)
            self.assertEqual(bump.arc_length_upper_bound(grade, 1.0), 1.0)
        # 带内也单调递增、且不超过整条凸条的行程
        inside = [bump.arc_length_upper_bound(10.0, x)
                  for x in (bump.BUMP_START_M, bump.BUMP_TOP_START_M, bump.BUMP_TOP_END_M,
                            bump.BUMP_END_M)]
        self.assertEqual(inside, sorted(inside))
        self.assertAlmostEqual(inside[1] - inside[0], ramp, places=12)
        self.assertAlmostEqual(inside[2] - inside[1], bump.BUMP_TOP_M, places=12)

    def test_mesh_is_closed_outward_and_matches_surface(self):
        for column in (grid.SPEEDBUMP_START_COLUMN, 44, grid.COLUMNS - 1):
            vertices, faces, origin = bump.tile_mesh(0, column)
            local = [tuple(a - b for a, b in zip(p, origin)) for p in vertices]
            edges, directions = Counter(), Counter()
            for a, b, c in faces:
                p, q, r = local[a], local[b], local[c]
                u = [q[i] - p[i] for i in range(3)]
                v = [r[i] - p[i] for i in range(3)]
                normal = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                          u[0] * v[1] - u[1] * v[0])
                self.assertGreater(sum(x * x for x in normal), 1e-14)   # 无退化面
                for s, t in ((a, b), (b, c), (c, a)):
                    edges[tuple(sorted((s, t)))] += 1
                    directions[(s, t)] += 1
                if normal[2] > 0:
                    x, y, z = [sum(local[k][i] for k in (a, b, c)) / 3 for i in range(3)]
                    self.assertLess(abs(z - bump.profile_height(bump.RAMP_DEG, x, y)), .003)
            self.assertTrue(all(count == 2 for count in edges.values()))   # 闭合二维流形
            self.assertTrue(all(count == directions[(b, a)] for (a, b), count
                                in directions.items()))                      # 绕向一致
            # 朝外：体积 = 平板 15×6×0.35 + 凸条（横截面 0.01 m² × 横向积分 5 m）
            self.assertAlmostEqual(hill._signed_volume(local, faces), 31.5 + 0.05, places=6)
            # 上表面最高点 = 50 mm、且只在平顶段（梯形分段线性 ⇒ 顶点就在两个折点上）
            top = [p for p in local if p[2] > -0.5 * hill.THICKNESS_M]
            self.assertAlmostEqual(max(p[2] for p in top), bump.BUMP_HEIGHT_M, places=12)
            self.assertEqual({round(p[0], 9) for p in top if abs(p[2] - bump.BUMP_HEIGHT_M) < 1e-12},
                             {round(bump.BUMP_TOP_START_M, 9), round(bump.BUMP_TOP_END_M, 9)})

    def test_tensor_equivalence(self):
        if importlib.util.find_spec('torch') is None:
            self.skipTest('torch unavailable')
        import torch
        import speedbump_torch
        xs = torch.linspace(-3, 14, 1500, dtype=torch.float64)
        ys = torch.linspace(-4, 4, len(xs), dtype=torch.float64)
        for grade in (0.0, 5.0, 10.0, bump.RAMP_DEG):
            actual = speedbump_torch.profile_height_tensor(torch.full_like(xs, grade), xs, ys)
            expected = torch.tensor([bump.profile_height(grade, x, y)
                                     for x, y in zip(xs.tolist(), ys.tolist())],
                                    dtype=torch.float64)
            self.assertLess(float((actual - expected).abs().max()), 1e-12)
            centre = speedbump_torch.profile_height_tensor(torch.full_like(xs, grade), xs)
            expected = torch.tensor([bump.profile_height(grade, x) for x in xs.tolist()],
                                    dtype=torch.float64)
            self.assertLess(float((centre - expected).abs().max()), 1e-12)

    def test_invalid_inputs(self):
        for function in (bump.profile_height, bump.profile_slope_degrees):
            for args in ((46, 0.0), (10, float('nan')), (10, 5.6, float('inf')),
                         (float('inf'), 0.0)):
                with self.assertRaises(ValueError):
                    function(*args)
        with self.assertRaises(ValueError):
            bump.arc_length_upper_bound(10, float('nan'))
        with self.assertRaises(ValueError):
            bump.lateral_factor(float('nan'))


class MixedGridWiringTests(unittest.TestCase):
    """混合网格的分派、任务/cfg/测试台接线（源码级守卫，不需要 Isaac Lab）。"""

    def test_mixed_grid_dispatches_by_column(self):
        # 同一行里，平坡列 / 减速带列 / 波浪列取各自模块的 tile
        for row in (0, 3, grid.ROWS - 1):
            for column in (0, 20, 30, 39):
                self.assertEqual(mixed.tile_mesh(row, column, "hill"),
                                 hill.tile_mesh(row, column))
            for column in range(grid.SPEEDBUMP_START_COLUMN, grid.WAVE_START_COLUMN):
                self.assertEqual(mixed.tile_mesh(row, column, "hill"),
                                 bump.tile_mesh(row, column))
            for column in range(grid.WAVE_START_COLUMN, grid.COLUMNS):
                self.assertEqual(mixed.tile_mesh(row, column, "hill"),
                                 wave.tile_mesh(row, column))
        with self.assertRaises(ValueError):
            mixed.tile_mesh(0, 0, "moon")
        # 整块网格：四类 tile 的总和逐项对上（组装不丢块、不错块）
        vertices, faces, origins = mixed.grid_mesh("hill")
        lanes = Counter(grid.slope_degrees(c, r)
                        for r in range(grid.ROWS) for c in range(grid.COLUMNS))
        bump_verts, bump_faces, _ = bump.tile_mesh(0, grid.SPEEDBUMP_START_COLUMN)
        wave_verts, wave_faces, _ = wave.tile_mesh(0, grid.WAVE_START_COLUMN)
        hills = lanes[5.0] + lanes[10.0]
        self.assertEqual(len(vertices),
                         lanes[0.0] * 8 + hills * 16
                         + lanes[grid.SPEEDBUMP_RAMP_DEG] * len(bump_verts)
                         + lanes[grid.WAVE_MAX_SLOPE_DEG] * len(wave_verts))
        self.assertEqual(len(faces),
                         lanes[0.0] * 12 + hills * 28
                         + lanes[grid.SPEEDBUMP_RAMP_DEG] * len(bump_faces)
                         + lanes[grid.WAVE_MAX_SLOPE_DEG] * len(wave_faces))
        self.assertEqual(len(origins), grid.ROWS)
        self.assertEqual(len({o for row in origins for o in row}), grid.GRID_SIZE)

    def test_arc_length_bound_takes_the_worst_family(self):
        # 减速带列只有 10.023607 m、hill 10.092560 m、rolling 10.1041 m，
        # 而 2026-10-16 加入的波浪（3 周期正弦）是 10.116044 m ⇒ **波浪主导**（两种模式都一样）。
        wave_bound = wave.arc_length_upper_bound(10.0, 10.0)
        self.assertAlmostEqual(wave_bound, 10.11604389909993, places=9)
        for smooth in ("hill", "rolling"):
            self.assertAlmostEqual(mixed.arc_length_upper_bound(10.0, 10.0, smooth),
                                   wave_bound, places=12)
        self.assertGreater(wave_bound, bump.arc_length_upper_bound(10.0, 10.0))
        self.assertGreater(wave_bound, hill.profile_arc_length(10.0, 10.0))
        with self.assertRaises(ValueError):
            mixed.arc_length_upper_bound(10.0, 10.0, "moon")

    def test_task_wiring_carries_the_bump_block_in_every_task(self):
        package = ROOT / 'imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/towing'
        registry = (package / '__init__.py').read_text(encoding='utf-8')
        # 独立的全减速带任务已按用户决定删除；只有 hill / rolling 两个任务
        self.assertNotIn('speedbump', registry)
        self.assertIn('id="Imgo2-towing-upper-rl-lab"', registry)
        self.assertIn('id="Imgo2-towing-upper-rolling-rl-lab"', registry)
        env_cfg = (package / 'upper_env_cfg.py').read_text(encoding='utf-8')
        self.assertIn('if self.towing_terrain_profile not in ("hill", "rolling")', env_cfg)
        self.assertIn('TowingMixedRollingTerrainGenerator if self.towing_terrain_profile', env_cfg)
        self.assertIn('from .mdp.mixed_geometry import arc_length_upper_bound', env_cfg)
        self.assertNotIn('UpperSpeedBumpTowingEnvCfg', env_cfg)
        terrain = (package / 'slope_terrain.py').read_text(encoding='utf-8')
        self.assertIn('class TowingMixedTerrainGenerator:', terrain)
        self.assertIn('class TowingMixedRollingTerrainGenerator(TowingMixedTerrainGenerator):',
                      terrain)
        self.assertIn('smooth = "rolling"', terrain)
        self.assertIn('return grid_mesh(cls.smooth)', terrain)
        mdp = (package / 'upper_mdp.py').read_text(encoding='utf-8')
        self.assertIn('from .mdp.speedbump_torch import profile_height_tensor', mdp)
        self.assertIn('spec["terrain_family"] == SPEEDBUMP_TERRAIN', mdp)
        self.assertIn('torch.where(\n        term.terrain_speedbump,', mdp)

    def test_testbench_profile_choices_and_segment_stats(self):
        source = (ROOT / 'imgo2_rl/scripts/towing/play_towing_test.py').read_text(
            encoding='utf-8')
        self.assertIn('choices=("hill", "rolling")', source)
        # CLI 里不得再有 speedbump 选项（"speedbump" 只是 pitch 分派里的族名）
        terrain_line = next(line for line in source.splitlines() if "--terrain-profile" in line)
        self.assertNotIn("speedbump", terrain_line)
        self.assertNotIn("speedbump", terrain_line + source.split("--terrain-profile")[1][:200])
        for token in ('speedbump_mask', 'wave_mask', 'surface_heights', 'smooth_height_tensor',
                      'connection_grid.is_speedbump', 'connection_grid.is_wave',
                      'wave_height_tensor'):
            self.assertIn(token, source)
        # `--dry-run` 两种平滑段都能跑（纯标准库、不 import torch）
        for profile in ('hill', 'rolling'):
            result = subprocess.run(
                [sys.executable, '-X', 'utf8',
                 str(ROOT / 'imgo2_rl/scripts/towing/play_towing_test.py'),
                 '--dry-run', '--terrain-profile', profile],
                capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(profile, result.stdout)
        # 分段统计：减速带列按梯形切片，平滑列仍走原剖面
        path = ROOT / 'imgo2_rl/scripts/tools/play_test_stats.py'
        spec = importlib.util.spec_from_file_location('speedbump_stats', path)
        stats = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stats)
        self.assertEqual(stats.profile_for_column('hill', 0), 'hill')
        self.assertEqual(stats.profile_for_column('hill', grid.SPEEDBUMP_START_COLUMN),
                         'speedbump')
        self.assertEqual(stats.profile_for_column('hill', grid.WAVE_START_COLUMN), 'wave')
        self.assertEqual(stats.segment_of(2.0, 'wave'), '平地(出生)')
        self.assertEqual(stats.segment_of(2.8, 'wave'), '上坡')      # 波峰之前：正弦上升段
        self.assertEqual(stats.segment_of(3.2, 'wave'), '下坡')      # 波峰之后：下降段
        self.assertEqual(stats.segment_of(10.0, 'wave'), '平地(出口)')
        self.assertEqual(stats.segment_of(2.0, 'speedbump'), '平地(出生)')
        self.assertEqual(stats.segment_of(5.4, 'speedbump'), '平地(带前)')
        self.assertEqual(stats.segment_of(bump.BUMP_START_M + 0.05, 'speedbump'), '上坡')
        self.assertEqual(stats.segment_of(bump.BUMP_CENTER_X_M, 'speedbump'), '带顶')
        self.assertEqual(stats.segment_of(bump.BUMP_TOP_END_M + 0.05, 'speedbump'), '下坡')
        self.assertEqual(stats.segment_of(6.0, 'speedbump'), '平地(带后)')
        self.assertEqual(stats.segment_of(10.0, 'speedbump'), '平地(出口)')
        self.assertEqual(stats.segment_of(3.75, 'hill'), '上坡')       # 既有地形不受影响
        for name in ('平地(带前)', '带顶', '平地(带后)'):
            self.assertIn(name, stats.SEGMENT_ORDER)
        # 地形不同 ⇒ 不能标为同条件策略/基线配对
        path = ROOT / 'imgo2_rl/scripts/tools/compare_play_runs.py'
        spec = importlib.util.spec_from_file_location('speedbump_compare', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.paired_arguments(
            {'arguments': {'terrain_profile': 'hill'}},
            {'arguments': {'terrain_profile': 'hill'}}), [])
        self.assertEqual(len(module.paired_arguments(
            {'arguments': {'terrain_profile': 'hill'}},
            {'arguments': {'terrain_profile': 'rolling'}})), 1)


if __name__ == '__main__':
    unittest.main()
