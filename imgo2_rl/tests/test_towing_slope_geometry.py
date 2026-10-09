"""Offline geometry/deadline checks against the actual terrain mesh generator.

2026-10-09：地形从「每格一个恒定坡度、按行列交替正负」改成**一条连续剖面**
（平地 3 m → 上坡 4 m → 坡顶 1 m → 下坡 4 m → 平地 3 m，见 `mdp/slope_geometry.py`），
所以这里检查的性质也跟着换：lane 的坡度量级（0/5/10）、闭合实体拓扑、剖面分段与弧长、
出生点仍在平地段且挂点距/安全边界成立、最慢速度仍能在 timeout 内到达目标。
"""

from collections import Counter
import importlib.util
import math
from pathlib import Path
import sys
import unittest

MDP = Path(__file__).resolve().parents[1] / \
    'source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/mdp'
sys.path.insert(0, str(MDP))
import connection_grid as grid
import slope_geometry as geometry
import episode_geometry as episode


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def _load_torch_profile():
    """`mdp/profile_torch.py`（向量化剖面高度）需要 torch；没有就跳过。"""
    try:
        import torch  # noqa: F401
    except ImportError:
        return None
    spec = importlib.util.spec_from_file_location("profile_torch_test", MDP / "profile_torch.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["profile_torch_test"] = module
    spec.loader.exec_module(module)
    return module


class SlopeGeometryTests(unittest.TestCase):
    def test_each_length_has_balanced_terrain_and_connection_types(self):
        for row in range(grid.ROWS):
            specs = [grid.env_spec(row*grid.COLUMNS+c) for c in range(grid.COLUMNS)]
            # 坡度量级：20 个平地 lane + 10 个 5° + 10 个 10°（不再有正负号）
            self.assertEqual(Counter(s['slope_degrees'] for s in specs), {0: 20, 5: 10, 10: 10})
            for start, end, expected in ((0, 20, (8, 8, 4)),
                                         (20, 30, (4, 4, 2)), (30, 40, (4, 4, 2))):
                counts = Counter(s['model_name'] for s in specs[start:end])
                self.assertEqual(tuple(counts[n] for n in ('compliant', 'rigid', 'inextensible')),
                                 expected)
        # 方向平衡是构造上的：同一列的坡度量级在 20 行里恒定，剖面自带一段上坡一段下坡
        for column in (20, 30):
            grade = grid.slope_degrees(column, 0)
            self.assertGreater(grade, 0.0)
            self.assertEqual([grid.slope_degrees(column, row) for row in range(grid.ROWS)],
                             [grade] * grid.ROWS)

    def test_profile_segments_and_local_slopes(self):
        for grade in (0.0, 5.0, 10.0):
            rise = math.tan(math.radians(grade))
            # 分段高度：平地 → 上坡 → 坡顶 → 下坡 → 平地
            self.assertAlmostEqual(geometry.profile_height(grade, 0.0), 0.0, places=12)
            self.assertAlmostEqual(geometry.profile_height(grade, 3.0), 0.0, places=12)
            self.assertAlmostEqual(geometry.profile_height(grade, 5.0), 2.0*rise, places=12)
            self.assertAlmostEqual(geometry.profile_height(grade, 7.0), 4.0*rise, places=12)
            self.assertAlmostEqual(geometry.profile_height(grade, 8.0), 4.0*rise, places=12)
            self.assertAlmostEqual(geometry.profile_height(grade, 12.0), 0.0, places=12)
            self.assertAlmostEqual(geometry.profile_height(grade, 20.0), 0.0, places=12)
            # 局部坡度：上坡 +grade、坡顶 0、下坡 −grade、两头平地 0
            self.assertEqual([geometry.profile_slope_degrees(grade, x)
                              for x in (1.0, 5.0, 7.5, 10.0, 13.0)],
                             [0.0, grade, 0.0, -grade, 0.0])
            self.assertAlmostEqual(geometry.profile_height(grade, 7.5), 4.0*rise, places=12)
        self.assertEqual(geometry.PROFILE_LENGTH_M, 15.0)
        self.assertLessEqual(geometry.PROFILE_LENGTH_M, geometry.FORWARD_M)
        with self.assertRaises(ValueError):
            geometry.profile_height(50.0, 5.0)
        with self.assertRaises(ValueError):
            geometry.profile_height(5.0, float('nan'))

    def test_arc_length_is_longer_than_the_plane_projection(self):
        for grade in (0.0, 5.0, 10.0):
            cosine = math.cos(math.radians(grade))
            self.assertAlmostEqual(geometry.profile_arc_length(grade, 3.0), 3.0, places=12)
            # 走过整条剖面后 = 平地 3 + 上坡 4/cos + 坡顶 1 + 下坡 4/cos + 出口平地
            self.assertAlmostEqual(geometry.profile_arc_length(grade, 18.0), 10.0 + 8.0/cosine,
                                   places=12)
            self.assertGreaterEqual(geometry.profile_arc_length(grade, 10.0), 10.0 - 1e-12)
        worst = geometry.profile_arc_length(geometry.MAX_GRADE_DEG, episode.GOAL_DISTANCE_M)
        # 10 m 水平目标在 10° 剖面上 = 3 + 6/cos10 + 1
        self.assertAlmostEqual(worst, 3.0 + 6.0/math.cos(math.radians(10.0)) + 1.0, places=9)
        self.assertLess(worst, 10.2)

    def test_closed_mesh_top_normal_and_origin_match_logical_cells(self):
        vertices, faces, origins = geometry.grid_mesh()
        # 平地 lane 的剖面退化成矩形（4 个剖面点 ⇒ 8 顶点 / 12 面），坡道 lane 是 8 点 / 16 顶点 / 28 面
        lanes = Counter(grid.slope_degrees(c, r)
                        for r in range(grid.ROWS) for c in range(grid.COLUMNS))
        slick = lanes[0.0]
        hills = lanes[5.0] + lanes[10.0]
        self.assertEqual(len(vertices), slick*8 + hills*16)
        self.assertEqual(len(faces), slick*12 + hills*28)
        self.assertEqual(len({o for row in origins for o in row}), grid.GRID_SIZE)
        for row in (0, 3, grid.ROWS - 1):
            for col in (0, 20, 25, 30, 39):
                verts, tris, origin = geometry.tile_mesh(row, col)
                self.assertEqual(origin, origins[row][col])
                grade = grid.slope_degrees(col, row)
                # 闭合二维流形：每条无向边恰好被两个面共享
                edge_counts = Counter(tuple(sorted((f[i], f[(i+1) % 3])))
                                      for f in tris for i in range(3))
                self.assertEqual(set(edge_counts.values()), {2})
                # 绕向朝外：有符号体积 = 剖面面积 × 板宽
                local = [(x-origin[0], y-origin[1], z-origin[2]) for x, y, z in verts]
                area = 20.0*geometry.THICKNESS_M + 5.0*4.0*math.tan(math.radians(grade))
                self.assertAlmostEqual(geometry._signed_volume(local, tris),
                                       area*2.0*geometry.HALF_WIDTH_M, places=6)
                # 上表面高度 = profile_height，最高点 = 坡顶高度
                top_z = {}
                for x, _, z in local:
                    top_z[round(x, 9)] = max(top_z.get(round(x, 9), z), z)
                # 每个上表面顶点都必须落在 profile_height 上（平地 lane 只有两端点）
                self.assertIn(round(-geometry.BACK_M, 9), top_z)
                self.assertIn(round(geometry.FORWARD_M, 9), top_z)
                for x, z in top_z.items():
                    self.assertAlmostEqual(z, geometry.profile_height(grade, x), places=10)
                self.assertAlmostEqual(max(z for _, _, z in local),
                                       4.0*math.tan(math.radians(grade)), places=10)
                # 上表面的面法向朝上（三个顶点都在上层）
                for face in tris:
                    points = [local[i] for i in face]
                    if all(p[2] > -0.5*geometry.THICKNESS_M for p in points):
                        a, b, c = points
                        u = [b[i]-a[i] for i in range(3)]
                        v = [c[i]-a[i] for i in range(3)]
                        cross = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
                        self.assertGreater(cross[2], 0.0)

    def test_spawn_clearance_attachment_distance_and_goal_stay_on_lane(self):
        for index in range(grid.GRID_SIZE):
            spec = grid.env_spec(index)
            grade = spec['slope_degrees']
            # 出生在剖面的**平地段**上：用 0° 的恒定坡面系解（姿态竖直、无出生旋转）
            tangent, normal = geometry.slope_frame(0.0)
            robot, cart, rp, cp = geometry.attachment_root_positions(
                0.0, spec['initial_distance'])
            self.assertAlmostEqual(dot(robot, normal), .35, places=12)
            self.assertAlmostEqual(dot(cart, normal), .18, places=12)
            self.assertAlmostEqual(math.dist(rp, cp), spec['initial_distance'], places=12)
            # 出生点与小车都在平地段内、且离 slab 边界有余量
            self.assertLess(robot[0], geometry.UP_START_M)
            self.assertGreaterEqual(robot[0], 0.0)
            for root in (robot, cart):
                self.assertGreater(root[0]-.03, -geometry.BACK_M+geometry.BOUNDARY_MARGIN_M)
                self.assertLess(root[0]+.03+episode.GOAL_DISTANCE_M,
                                geometry.FORWARD_M-geometry.BOUNDARY_MARGIN_M)
            # 目标点仍落在车道内（10 m 处已是下坡段），弧长给 timeout 用
            goal_x = robot[0] + 0.03 + episode.GOAL_DISTANCE_M
            self.assertLess(goal_x, geometry.FORWARD_M-geometry.BOUNDARY_MARGIN_M)
            self.assertAlmostEqual(geometry.profile_height(grade, 0.0), 0.0, places=12)
            self.assertGreaterEqual(geometry.profile_arc_length(grade, episode.GOAL_DISTANCE_M),
                                    episode.GOAL_DISTANCE_M - 1e-12)

    def test_every_speed_arrives_before_the_discretized_timeout(self):
        # timeout 用**最陡档的坡面弧长**（10 m 水平目标 → 10.093 m 弧长）
        worst = geometry.profile_arc_length(geometry.MAX_GRADE_DEG, episode.GOAL_DISTANCE_M)
        limit = episode.episode_timeout_s(worst, episode.SPEED_RANGE[0])
        self.assertLess(abs(limit - 28.2315), 0.01)
        for index in range(111):
            speed = .4 + index*.01
            arrival = episode.SETTLE_TIME_S + worst/speed
            # Includes one 50 ms action interval and one termination sampling interval.
            self.assertLess(arrival+.1, limit)
        self.assertEqual(episode.episode_timeout_s(12, .4), 33)
        for args in ((0,.4), (10,0), (10,-1), (float('inf'),.4)):
            with self.assertRaises(ValueError):
                episode.episode_timeout_s(*args)


class TorchProfileTests(unittest.TestCase):
    """向量化剖面高度必须与标量版逐点一致（奖励/终止用的就是它）。"""

    def test_vectorised_matches_scalar(self):
        module = _load_torch_profile()
        if module is None:
            self.skipTest("需要 torch 才能核对 profile_torch")
        import torch
        xs = [-1.0, 0.0, 1.5, 3.0, 3.5, 5.0, 6.999, 7.0, 7.5, 8.0, 8.001, 10.0, 12.0, 12.5, 17.0]
        for grade in (0.0, 5.0, 10.0):
            grades = torch.full((len(xs),), grade)
            values = module.profile_height_tensor(grades, torch.tensor(xs))
            for x, value in zip(xs, values.tolist()):
                self.assertAlmostEqual(value, geometry.profile_height(grade, x), delta=1e-6,
                                       msg=f"grade={grade} x={x}")

    def test_batched_grades_are_per_environment(self):
        module = _load_torch_profile()
        if module is None:
            self.skipTest("需要 torch 才能核对 profile_torch")
        import torch
        grades = torch.tensor([0.0, 5.0, 10.0])
        xs = torch.tensor([5.0, 5.0, 5.0])
        values = module.profile_height_tensor(grades, xs).tolist()
        self.assertAlmostEqual(values[0], 0.0, places=9)
        self.assertAlmostEqual(values[1], 2.0*math.tan(math.radians(5.0)), delta=1e-6)
        self.assertAlmostEqual(values[2], 2.0*math.tan(math.radians(10.0)), delta=1e-6)


if __name__ == '__main__':
    unittest.main()
