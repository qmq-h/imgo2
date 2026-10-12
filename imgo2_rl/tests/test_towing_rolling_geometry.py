"""Rolling mesh geometry and optional tensor equivalence, without Isaac Lab."""
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
import rolling_geometry as rolling


class RollingGeometryTests(unittest.TestCase):
    def test_smooth_junctions_and_spawn(self):
        junctions = [rolling.START_M]
        for width in rolling.HILL_WIDTHS_M:
            junctions.append(junctions[-1] + width)
        for grade in (0, 5, 10):
            for x in junctions:
                for y in (0, 1.2, 3):
                    self.assertAlmostEqual(rolling.profile_height(grade, x, y), 0, places=12)
                    self.assertAlmostEqual(rolling.profile_slope_degrees(grade, x, y), 0, places=12)
                    for side in (-1, 1):
                        self.assertLess(rolling.profile_height(grade, x + side * 1e-6, y), 1e-10)
                        self.assertLess(abs(rolling.profile_slope_degrees(grade, x + side * 1e-6, y)), 1e-4)
            for x in (-2.25, -1, 0, 1, 10, 12.75):
                self.assertEqual(rolling.profile_height(grade, x), 0)

    def test_shape_scale_and_lane_edges(self):
        start = rolling.START_M
        for width in rolling.HILL_WIDTHS_M:
            peak = start + width / 2
            self.assertAlmostEqual(rolling.profile_height(10, peak), math.tan(math.radians(10)) * width / math.pi)
            self.assertAlmostEqual(rolling.profile_slope_degrees(10, start + width / 4), 10)
            self.assertAlmostEqual(rolling.profile_slope_degrees(10, start + width * 3 / 4), -10)
            for y in (-3, 3, -4, 4):
                self.assertEqual(rolling.profile_height(10, peak, y), 0)
            start += width

    def test_deadline_is_conservative(self):
        for grade in (0, 5, 10):
            dx = 10 / 20000
            actual = sum(dx * math.sqrt(1 + math.tan(math.radians(
                rolling.profile_slope_degrees(grade, (i + .5) * dx))) ** 2) for i in range(20000))
            bound = rolling.arc_length_upper_bound(grade, 10)
            self.assertGreaterEqual(bound + 1e-10, actual)
            self.assertLess(bound, 10.11)

    def test_adjacent_tiles_have_matching_heights(self):
        for x in [rolling.START_M + i * .05 for i in range(136)]:
            self.assertEqual(rolling.profile_height(5, x, 3), rolling.profile_height(10, x, -3))
        for grade in (0, 5, 10):
            for y in (-3, -1.5, 0, 1.5, 3):
                self.assertEqual(rolling.profile_height(grade, rolling.layout.FORWARD_M, y),
                                 rolling.profile_height(grade, -rolling.layout.BACK_M, y))

    def test_mesh_is_closed_outward_and_matches_surface(self):
        for column in (0, 20, 30):
            vertices, faces, origin = rolling.tile_mesh(0, column)
            local = [tuple(a - b for a, b in zip(p, origin)) for p in vertices]
            edges, directions = Counter(), Counter()
            for a, b, c in faces:
                p, q, r = local[a], local[b], local[c]
                u = [q[i] - p[i] for i in range(3)]
                v = [r[i] - p[i] for i in range(3)]
                normal = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
                self.assertGreater(sum(x*x for x in normal), 1e-14)
                for s, t in ((a,b),(b,c),(c,a)):
                    edges[tuple(sorted((s,t)))] += 1
                    directions[(s,t)] += 1
                if normal[2] > 0:
                    x, y, z = [sum(local[k][i] for k in (a,b,c)) / 3 for i in range(3)]
                    grade = rolling.slope_degrees(column, 0)
                    self.assertLess(abs(z - rolling.profile_height(grade, x, y)), .003)
            self.assertTrue(all(n == 2 for n in edges.values()))
            self.assertTrue(all(n == directions[(b,a)] for (a,b),n in directions.items()))
            self.assertGreater(rolling.layout._signed_volume(local, faces), 0)

    def test_tensor_equivalence(self):
        if importlib.util.find_spec('torch') is None:
            self.skipTest('torch unavailable')
        import torch
        import rolling_torch
        xs = torch.linspace(-3, 14, 1500, dtype=torch.float64)
        ys = torch.linspace(-4, 4, len(xs), dtype=torch.float64)
        for grade in (0, 5, 10):
            actual = rolling_torch.profile_height_tensor(torch.full_like(xs, grade), xs, ys)
            expected = torch.tensor([rolling.profile_height(grade, x, y) for x,y in zip(xs.tolist(),ys.tolist())], dtype=torch.float64)
            self.assertLess(float((actual - expected).abs().max()), 1e-12)

    def test_invalid_inputs(self):
        for function in (rolling.profile_height, rolling.profile_slope_degrees):
            for args in ((46, 0), (10, float('nan')), (10, 3, float('inf'))):
                with self.assertRaises(ValueError):
                    function(*args)

    def test_testbench_dry_run_and_comparison(self):
        script = ROOT / 'imgo2_rl/scripts/towing/play_towing_test.py'
        result = subprocess.run([sys.executable, '-X', 'utf8', str(script), '--dry-run', '--terrain-profile', 'rolling'],
                                capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('rolling', result.stdout)
        path = ROOT / 'imgo2_rl/scripts/tools/compare_play_runs.py'
        spec = importlib.util.spec_from_file_location('rolling_compare', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.paired_arguments({'arguments': {}}, {'arguments': {'terrain_profile': 'hill'}}), [])
        self.assertEqual(len(module.paired_arguments({'arguments': {}}, {'arguments': {'terrain_profile': 'rolling'}})), 1)
        path = ROOT / 'imgo2_rl/scripts/tools/play_test_stats.py'
        spec = importlib.util.spec_from_file_location('rolling_stats', path)
        stats = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stats)
        # Repeated ascents/descents must not use the single-hill positional bins.
        self.assertEqual(stats.segment_of(4.0, 'rolling'), '上坡')
        self.assertEqual(stats.segment_of(3.5, 'rolling'), '下坡')
        self.assertEqual(stats.segment_of(10.0, 'rolling'), '平地(出口)')


if __name__ == '__main__':
    unittest.main()
