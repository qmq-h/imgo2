"""Wave (3-period sine) geometry, mixed-grid dispatch and tensor equivalence, no Isaac Lab.

2026-10-16 口径（用户）：「最后添加一个波浪的起伏地形，**和上下坡占据同样的位置**，最后用一个
平地收尾」+「振幅不要太大，先定义大概 10 cm」。所以：3 周期等幅正弦，周期 2.25 m（正好铺满
与平坡相同的 2.25–9.0 m 窗口），振幅 **10 cm**，两端各用半个周期的 `cos²` 包络平滑归零
（与前后平地 C1 衔接），9.0 m 之后回到平地收尾 —— 10 m 的 STOP 点仍在平地上。

本文件核对：窗口/周期/包络、两端 C1、真实最大坡度 = 解析值、横向收零、闭合实体网格（正弦
均值为 0 ⇒ 体积 = 平板 31.5 m³）、弧长上界（10 m 目标 = 10.116044 m，是该网格最长的一族）、
张量↔标量等价、非法输入，以及混合网格/测试台/分段统计的接线。
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
import slope_geometry as hill
import wave_geometry as wave
import mixed_geometry as mixed


class WaveGeometryTests(unittest.TestCase):
    def test_window_period_and_envelope(self):
        # 与平坡**同一个窗口**；3 个周期正好铺满；两端各收半个周期
        self.assertAlmostEqual(wave.START_M, hill.UP_START_M, places=12)
        self.assertAlmostEqual(wave.END_M, hill.FLAT_OUT_START_M, places=12)
        self.assertAlmostEqual(wave.SPAN_M, 6.75, places=12)
        self.assertEqual((wave.PERIOD_COUNT, wave.PERIOD_M, wave.PEAK_M), (3, 2.25, 0.10))
        self.assertAlmostEqual(wave.TAPER_M, 1.125, places=12)
        self.assertAlmostEqual(wave.PERIOD_COUNT * wave.PERIOD_M, wave.SPAN_M, places=12)
        # 与 connection_grid 的常量/派生坡度一致（单一事实来源）
        self.assertEqual(group := (grid.WAVE_PERIOD_COUNT, grid.WAVE_PERIOD_M, grid.WAVE_PEAK_M),
                         (3, 2.25, 0.10))
        self.assertAlmostEqual(wave.MAX_SLOPE_DEG, grid.WAVE_MAX_SLOPE_DEG, places=12)
        self.assertAlmostEqual(
            math.tan(math.radians(wave.MAX_SLOPE_DEG)),
            2.0 * math.pi * wave.PEAK_M / wave.PERIOD_M, places=12)
        # 包络：0 与 1 的位置
        self.assertAlmostEqual(wave._envelope(0.0), 0.0, places=12)
        self.assertAlmostEqual(wave._envelope(wave.TAPER_M), 1.0, places=12)
        self.assertAlmostEqual(wave._envelope(wave.SPAN_M / 2), 1.0, places=12)
        self.assertAlmostEqual(wave._envelope(wave.SPAN_M), 0.0, places=12)
        # 导数为 0 的端点（与平地 C1 衔接）
        self.assertAlmostEqual(wave._envelope_derivative(0.0), 0.0, places=12)
        self.assertAlmostEqual(wave._envelope_derivative(wave.SPAN_M), 0.0, places=12)

    def test_flat_before_after_and_c1_junctions(self):
        """窗口外一律为平地；两端高度与一阶导都为 0（用户要的「平地收尾」）。"""
        for x in (-2.25, 0.0, wave.START_M, wave.END_M, 10.0, 11.25, 12.75):
            self.assertEqual(wave.profile_height(0.0, x), 0.0, msg=f"x={x}")
            self.assertEqual(wave.profile_slope_degrees(0.0, x), 0.0, msg=f"x={x}")
        for side in (-1, 1):
            probe = wave.START_M + side * 1e-6
            self.assertLess(abs(wave.profile_height(0.0, probe)), 1e-9)
            self.assertLess(abs(wave.profile_slope_degrees(0.0, probe)), 1e-3)
            probe = wave.END_M + side * 1e-6
            self.assertLess(abs(wave.profile_height(0.0, probe)), 1e-9)
            self.assertLess(abs(wave.profile_slope_degrees(0.0, probe)), 1e-3)

    def test_amplitude_and_slope_extremes(self):
        """等幅段 4 个极值满幅 ±10 cm（首尾各降为半幅，因为落在收零段里）；最大坡度 = 解析值。"""
        quarter = wave.PERIOD_M / 4.0
        # 每 1/4 周期取一点：0..12（0 与 12 = 两端；奇数是极值、偶数是过零点）
        heights = {index: wave.profile_height(0.0, wave.START_M + index * quarter)
                   for index in range(13)}
        for even in (0, 2, 4, 6, 8, 10, 12):       # 过零点
            self.assertAlmostEqual(heights[even], 0.0, places=12)
        # 收零段内（两端各一个极值）降为半幅，中间 4 个极值满幅 ±10 cm
        expected = {1: +0.05, 3: -0.10, 5: +0.10, 7: -0.10, 9: +0.10, 11: -0.05}
        for index, value in expected.items():
            self.assertAlmostEqual(heights[index], value, places=12,
                                   msg=f"1/4 周期序号 {index}")
        # 数值扫最大坡度：必须**不超过**解析值，且在中段取到
        worst, worst_x = 0.0, 0.0
        steps = 60000
        for index in range(steps + 1):
            x = wave.START_M + wave.SPAN_M * index / steps
            value = abs(wave.profile_slope_degrees(0.0, x))
            if value > worst:
                worst, worst_x = value, x
        self.assertLessEqual(worst, wave.MAX_SLOPE_DEG + 1e-9)
        self.assertAlmostEqual(worst, wave.MAX_SLOPE_DEG, places=6)
        # 极值出现在**包络到 1 的边界**（u = TAPER，即等幅段起点）与等幅段内
        self.assertGreaterEqual(worst_x, wave.START_M + wave.TAPER_M - 1e-9)
        self.assertLessEqual(worst_x, wave.END_M - wave.TAPER_M + 1e-9)
        # 谷底 −10 cm 仍在板厚 35 cm 之内（不会穿底）
        self.assertGreater(wave.PEAK_M, 0.0)
        self.assertLess(wave.PEAK_M, hill.THICKNESS_M)

    def test_lateral_crown_full_amplitude_then_taper_to_zero(self):
        self.assertEqual(wave.lateral_factor(0.0), 1.0)
        for y in (-2.0, -1.0, 0.5, 2.0):
            self.assertEqual(wave.lateral_factor(y), 1.0)
        for y in (-3.0, 3.0, -4.0, 4.0):
            self.assertEqual(wave.lateral_factor(y), 0.0)
        self.assertAlmostEqual(wave.lateral_factor(2.5), 0.5, places=12)
        trough = wave.START_M + 3 * wave.PERIOD_M / 4.0
        self.assertAlmostEqual(wave.profile_height(0.0, trough, 0.0), -wave.PEAK_M, places=12)
        self.assertAlmostEqual(wave.profile_height(0.0, trough, 2.5), -wave.PEAK_M / 2.0,
                               places=12)
        self.assertEqual(wave.profile_height(0.0, trough, 3.0), 0.0)

    def test_arc_length_bound_is_the_longest_family(self):
        dx = 10.0 / 40000
        for grade in (0.0, 5.0, 10.0, wave.MAX_SLOPE_DEG):
            actual = sum(dx * math.sqrt(1 + math.tan(math.radians(
                wave.profile_slope_degrees(grade, (index + .5) * dx))) ** 2)
                for index in range(40000))
            bound = wave.arc_length_upper_bound(grade, 10.0)
            self.assertGreaterEqual(bound + 1e-9, actual)
            self.assertLess(bound - actual, 0.02)          # 保守但不过分（<2 cm）
        self.assertAlmostEqual(wave.arc_length_upper_bound(0.0, 10.0), 10.11604389909993,
                               places=9)
        self.assertEqual(wave.arc_length_upper_bound(0.0, 1.0), 1.0)
        values = [wave.arc_length_upper_bound(0.0, x)
                  for x in (wave.START_M, 4.0, wave.END_M)]
        self.assertEqual(values, sorted(values))
        # 波浪比同窗口的其它族都长 ⇒ 混合网格的上界由它给出（两种平滑段一样）
        self.assertGreater(wave.arc_length_upper_bound(0.0, 10.0),
                           hill.profile_arc_length(10.0, 10.0))

    def test_mesh_is_closed_outward_and_matches_surface(self):
        for column in (grid.WAVE_START_COLUMN, 54, grid.COLUMNS - 1):
            vertices, faces, origin = wave.tile_mesh(0, column)
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
                    self.assertLess(abs(z - wave.profile_height(0.0, x, y)), .003)
            self.assertTrue(all(count == 2 for count in edges.values()))   # 闭合二维流形
            self.assertTrue(all(count == directions[(b, a)] for (a, b), count
                                in directions.items()))                      # 绕向一致
            self.assertGreater(hill._signed_volume(local, faces), 0)         # 朝外
            # 正弦在整周期上均值为 0 ⇒ 立体积仍是平板体积（谷与峰相抵）
            self.assertAlmostEqual(hill._signed_volume(local, faces), 31.5, places=3)
            # 上表面：波峰恰好 +10 cm、波谷恰好 −10 cm
            top = [p[2] for p in local if p[2] > -0.5 * hill.THICKNESS_M]
            self.assertAlmostEqual(max(top), wave.PEAK_M, places=9)
            self.assertAlmostEqual(min(top), -wave.PEAK_M, places=9)

    def test_tensor_equivalence(self):
        if importlib.util.find_spec('torch') is None:
            self.skipTest('torch unavailable')
        import torch
        import wave_torch
        xs = torch.linspace(-3, 14, 3000, dtype=torch.float64)
        ys = torch.linspace(-4, 4, len(xs), dtype=torch.float64)
        for grade in (0.0, grid.WAVE_MAX_SLOPE_DEG):
            actual = wave_torch.profile_height_tensor(torch.full_like(xs, grade), xs, ys)
            expected = torch.tensor([wave.profile_height(grade, x, y)
                                     for x, y in zip(xs.tolist(), ys.tolist())],
                                    dtype=torch.float64)
            self.assertLess(float((actual - expected).abs().max()), 1e-12)
            centre = wave_torch.profile_height_tensor(torch.full_like(xs, grade), xs)
            expected = torch.tensor([wave.profile_height(grade, x) for x in xs.tolist()],
                                    dtype=torch.float64)
            self.assertLess(float((centre - expected).abs().max()), 1e-12)

    def test_invalid_inputs(self):
        for function in (wave.profile_height, wave.profile_slope_degrees):
            for args in ((46, 0.0), (10, float('nan')), (10, 5.0, float('inf')),
                         (float('inf'), 0.0)):
                with self.assertRaises(ValueError):
                    function(*args)
        with self.assertRaises(ValueError):
            wave.arc_length_upper_bound(10, float('nan'))
        with self.assertRaises(ValueError):
            wave.lateral_factor(float('nan'))


class WaveGridWiringTests(unittest.TestCase):
    def test_grid_columns_and_mixed_dispatch(self):
        # 列 50–59 = 波浪块；档位报正弦最大坡度；连接照 5° 坡块同构 4/4/2
        self.assertEqual(grid.WAVE_START_COLUMN, 50)
        self.assertEqual(grid.COLUMNS, 60)
        self.assertEqual([grid.terrain_family(c) for c in range(50, 60)],
                         [grid.WAVE_TERRAIN] * 10)
        self.assertEqual([grid.slope_degrees(c, 0) for c in range(50, 60)],
                         [grid.WAVE_MAX_SLOPE_DEG] * 10)
        self.assertEqual([grid.column_spec(c)[0] for c in range(50, 60)],
                         ['compliant'] * 4 + ['rigid'] * 4 + ['inextensible'] * 2)
        for row in (0, grid.ROWS - 1):
            for column in (50, 54, 59):
                self.assertEqual(mixed.tile_mesh(row, column, "hill"), wave.tile_mesh(row, column))

    def test_testbench_and_stats_carry_the_wave_block(self):
        source = (ROOT / 'imgo2_rl/scripts/towing/play_towing_test.py').read_text(
            encoding='utf-8')
        for token in ('wave_mask', 'wave_height_tensor', 'wave_geometry.profile_slope_degrees',
                      'connection_grid.is_wave'):
            self.assertIn(token, source)
        path = ROOT / 'imgo2_rl/scripts/tools/play_test_stats.py'
        spec = importlib.util.spec_from_file_location('wave_stats', path)
        stats = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stats)
        self.assertEqual(stats.profile_for_column('rolling', 50), 'wave')
        self.assertEqual(stats.profile_for_column('rolling', 49), 'speedbump')
        self.assertEqual(stats.profile_for_column('rolling', 39), 'rolling')
        # 波浪分段用**解析坡度**定号：波峰前上坡、波峰后下坡、窗口外是平地
        self.assertEqual(stats.segment_of(1.0, 'wave'), '平地(出生)')
        self.assertEqual(stats.segment_of(2.25 + 0.3, 'wave'), '上坡')
        self.assertEqual(stats.segment_of(2.25 + 0.8, 'wave'), '下坡')
        self.assertEqual(stats.segment_of(2.25 + 1.3, 'wave'), '下坡')
        self.assertEqual(stats.segment_of(2.25 + 1.8, 'wave'), '上坡')
        self.assertEqual(stats.segment_of(10.0, 'wave'), '平地(出口)')
        # `--dry-run` 仍可在无 torch 的解释器上跑（纯标准库路径）
        result = subprocess.run(
            [sys.executable, '-X', 'utf8',
             str(ROOT / 'imgo2_rl/scripts/towing/play_towing_test.py'), '--dry-run'],
            capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('15.6025°', result.stdout)


if __name__ == '__main__':
    unittest.main()
