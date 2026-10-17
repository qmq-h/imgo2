"""Offline geometry/deadline checks against the actual terrain mesh generator.

2026-10-09：地形从「每格一个恒定坡度、按行列交替正负」改成**一条连续剖面**
（平地 2.25 m → 上坡 3 m → 坡顶 0.75 m → 下坡 3 m → 平地 2.25 m，跑道总长 15 m，
见 `mdp/slope_geometry.py`），
所以这里检查的性质也跟着换：lane 的坡度量级（0/5/10）、闭合实体拓扑、剖面分段与弧长、
出生点仍在平地段且挂点距/安全边界成立、最慢速度仍能在 timeout 内到达目标。

2026-10-16：地面变成**混合网格**（`mdp/mixed_geometry.py`）——列 0–39 是本文件检查的平坡地形、
列 40–49 是减速带块（由 `test_towing_speedbump_geometry.py` 检查），所以本文件的网格汇总检查
走 `mixed_geometry.grid_mesh("hill")`，减速带列的 tile 尺寸从 `speedbump_geometry` 取。
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
import mixed_geometry as mixed
import speedbump_geometry as speedbump
import wave_geometry as wave


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
            # 坡度量级：20 平地 + 10×5° + 10×10° + 10 减速带（26.565°）+ 10 波浪（15.603°）
            self.assertEqual(Counter(s['slope_degrees'] for s in specs),
                             {0: 20, 5: 10, 10: 10, grid.SPEEDBUMP_RAMP_DEG: 10,
                              grid.WAVE_MAX_SLOPE_DEG: 10})
            for start, end, expected in ((0, 20, (8, 8, 4)),
                                         (20, 30, (4, 4, 2)), (30, 40, (4, 4, 2)),
                                         (40, 50, (4, 4, 2)), (50, 60, (4, 4, 2))):
                counts = Counter(s['model_name'] for s in specs[start:end])
                self.assertEqual(tuple(counts[n] for n in ('compliant', 'rigid', 'inextensible')),
                                 expected)
            # 后 10 列是减速带块、最后 10 列是波浪块，各自地形相同（只有连接类型不同）
            self.assertEqual({s['terrain_family'] for s in specs[:40]},
                             {grid.SMOOTH_TERRAIN})
            self.assertEqual({s['terrain_family'] for s in specs[40:50]},
                             {grid.SPEEDBUMP_TERRAIN})
            self.assertEqual({s['terrain_family'] for s in specs[50:]},
                             {grid.WAVE_TERRAIN})
        # 方向平衡是构造上的：同一列的坡度量级在 20 行里恒定，剖面自带一段上坡一段下坡
        for column in (20, 30):
            grade = grid.slope_degrees(column, 0)
            self.assertGreater(grade, 0.0)
            self.assertEqual([grid.slope_degrees(column, row) for row in range(grid.ROWS)],
                             [grade] * grid.ROWS)

    def test_profile_segments_and_local_slopes(self):
        # 跑道 15 m（后 2.25 + 前 12.75），剖面 11.25 m：平地 2.25 / 上坡 3 / 坡顶 0.75 / 下坡 3 / 平地 2.25
        self.assertEqual((geometry.BACK_M, geometry.FORWARD_M), (2.25, 12.75))
        self.assertEqual((geometry.FLAT_IN_M, geometry.UP_M, geometry.CREST_M,
                          geometry.DOWN_M, geometry.EXIT_M), (2.25, 3.0, 0.75, 3.0, 2.25))
        self.assertEqual(geometry.PROFILE_LENGTH_M, 11.25)
        for grade in (0.0, 5.0, 10.0):
            rise = math.tan(math.radians(grade))
            # 分段高度：平地 → 上坡 → 坡顶 → 下坡 → 平地
            self.assertAlmostEqual(geometry.profile_height(grade, 0.0), 0.0, places=12)
            self.assertAlmostEqual(geometry.profile_height(grade, 2.25), 0.0, places=12)
            self.assertAlmostEqual(geometry.profile_height(grade, 3.75), 1.5*rise, places=12)
            self.assertAlmostEqual(geometry.profile_height(grade, 5.25), 3.0*rise, places=12)
            self.assertAlmostEqual(geometry.profile_height(grade, 6.0), 3.0*rise, places=12)
            self.assertAlmostEqual(geometry.profile_height(grade, 9.0), 0.0, places=12)
            self.assertAlmostEqual(geometry.profile_height(grade, 20.0), 0.0, places=12)
            # 局部坡度：上坡 +grade、坡顶 0、下坡 −grade、两头平地 0
            self.assertEqual([geometry.profile_slope_degrees(grade, x)
                              for x in (1.0, 3.75, 5.6, 7.5, 11.0)],
                             [0.0, grade, 0.0, -grade, 0.0])
            self.assertAlmostEqual(geometry.profile_height(grade, 5.6), 3.0*rise, places=12)
        self.assertLessEqual(geometry.PROFILE_LENGTH_M, geometry.FORWARD_M)
        with self.assertRaises(ValueError):
            geometry.profile_height(50.0, 5.0)
        with self.assertRaises(ValueError):
            geometry.profile_height(5.0, float('nan'))

    def test_arc_length_is_longer_than_the_plane_projection(self):
        for grade in (0.0, 5.0, 10.0):
            cosine = math.cos(math.radians(grade))
            self.assertAlmostEqual(geometry.profile_arc_length(grade, 2.25), 2.25, places=12)
            # 走过整条剖面后 = 平地 2.25 + 上坡 3/cos + 坡顶 0.75 + 下坡 3/cos + 出口平地
            self.assertAlmostEqual(geometry.profile_arc_length(grade, 18.0),
                                   2.25 + 6.0/cosine + 0.75 + (18.0 - geometry.FLAT_OUT_START_M),
                                   places=12)
            self.assertGreaterEqual(geometry.profile_arc_length(grade, 10.0), 10.0 - 1e-12)
        # 本模块的剖面弧长（hill 口径）：STOP 点 10 m 落在出口平地上
        hill_worst = geometry.profile_arc_length(geometry.MAX_GRADE_DEG,
                                                 episode.STOP_DISTANCE_M)
        self.assertAlmostEqual(hill_worst, 2.25 + 6.0/math.cos(math.radians(10.0)) + 0.75 + 1.0,
                               places=9)
        self.assertLess(hill_worst, 10.2)
        # 训练 cfg 用的是**混合网格**的上界（平滑段 / 减速带 / 波浪 三者取大）
        # 2026-10-16 起由**波浪**主导（3 周期正弦 ≈ 10.1160 m），见几何测试与记录。
        mixed_worst = mixed.arc_length_upper_bound(geometry.MAX_GRADE_DEG,
                                                   episode.STOP_DISTANCE_M)
        self.assertGreaterEqual(mixed_worst, hill_worst)
        self.assertAlmostEqual(mixed_worst, 10.11604389909993, places=9)

    def test_closed_mesh_top_normal_and_origin_match_logical_cells(self):
        """混合网格汇总（2026-10-16）：平坡 tile 逐块计入，减速带列与波浪列各按自己的 tile 计入。"""
        vertices, faces, origins = mixed.grid_mesh("hill")
        lanes = Counter(grid.slope_degrees(c, r)
                        for r in range(grid.ROWS) for c in range(grid.COLUMNS))
        slick = lanes[0.0]                       # 平地 lane 的剖面退化成矩形 ⇒ 8 顶点 / 12 面
        hills = lanes[5.0] + lanes[10.0]         # 坡道 lane 8 个剖面点 ⇒ 16 顶点 / 28 面
        bumps = lanes[grid.SPEEDBUMP_RAMP_DEG]   # 减速带列（梯形 tile，尺寸由该模块给出）
        waves = lanes[grid.WAVE_MAX_SLOPE_DEG]   # 波浪列（正弦 tile，尺寸由该模块给出）
        bump_verts, bump_faces, _ = speedbump.tile_mesh(0, grid.SPEEDBUMP_START_COLUMN)
        wave_verts, wave_faces, _ = wave.tile_mesh(0, grid.WAVE_START_COLUMN)
        self.assertEqual(len(vertices),
                         slick*8 + hills*16 + bumps*len(bump_verts) + waves*len(wave_verts))
        self.assertEqual(len(faces),
                         slick*12 + hills*28 + bumps*len(bump_faces) + waves*len(wave_faces))
        self.assertEqual(len({o for row in origins for o in row}), grid.GRID_SIZE)
        for row in (0, 3, grid.ROWS - 1):
            for col in (0, 20, 25, 30, 39):
                verts, tris, origin = geometry.tile_mesh(row, col)
                self.assertEqual(origin, origins[row][col])
                self.assertEqual((verts, tris), mixed.tile_mesh(row, col, "hill")[:2])
                grade = grid.slope_degrees(col, row)
                # 闭合二维流形：每条无向边恰好被两个面共享
                edge_counts = Counter(tuple(sorted((f[i], f[(i+1) % 3])))
                                      for f in tris for i in range(3))
                self.assertEqual(set(edge_counts.values()), {2})
                # 绕向朝外：有符号体积 = 剖面多边形（鞋带公式）面积 × 板宽
                local = [(x-origin[0], y-origin[1], z-origin[2]) for x, y, z in verts]
                polygon = geometry.profile_polyline(grade)
                area = 0.5*abs(sum(polygon[i][0]*polygon[(i+1) % len(polygon)][1]
                                   - polygon[(i+1) % len(polygon)][0]*polygon[i][1]
                                   for i in range(len(polygon))))
                self.assertGreater(area, 0.0)
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
                                       3.0*math.tan(math.radians(grade)), places=10)
                # 上表面的面法向朝上（三个顶点都在上层）
                for face in tris:
                    points = [local[i] for i in face]
                    if all(p[2] > -0.5*geometry.THICKNESS_M for p in points):
                        a, b, c = points
                        u = [b[i]-a[i] for i in range(3)]
                        v = [c[i]-a[i] for i in range(3)]
                        cross = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
                        self.assertGreater(cross[2], 0.0)

    def test_adjacent_lanes_are_spliced_into_one_continuous_ground(self):
        """相邻 lane **共边拼接**：间距等于板尺寸 ⇒ 板与板严格相接、行列之间不留虚空。

        2026-10-09 用户要求「训练场景拼接到一起」，起因是训练地形之外没有 ground，
        间距 2 m 时机器人/小车走出自己那块板或掉进缝里就会直接坠下去。
        """
        self.assertEqual(geometry.ROW_SPACING_M, geometry.BACK_M + geometry.FORWARD_M)
        self.assertEqual(geometry.COLUMN_SPACING_M, 2.0*geometry.HALF_WIDTH_M)
        for row, column in ((0, 0), (3, 20), (19, 39)):
            x, y, _ = geometry.tile_origin(row, column)
            # 板的足迹：[x − BACK, x + FORWARD] × [y ± HALF_WIDTH]
            if row + 1 < geometry.ROWS:
                nx, ny, _ = geometry.tile_origin(row + 1, column)
                self.assertAlmostEqual(x + geometry.FORWARD_M, nx - geometry.BACK_M, places=9)
            if column + 1 < geometry.COLUMNS:
                nx, ny, _ = geometry.tile_origin(row, column + 1)
                self.assertAlmostEqual(y + geometry.HALF_WIDTH_M, ny - geometry.HALF_WIDTH_M,
                                       places=9)
        # 行接缝处两侧都是平地：剖面在 x = ±(BACK/FORWARD) 上高度为 0 ⇒ 没有台阶
        for grade in (0.0, 5.0, 10.0):
            self.assertAlmostEqual(geometry.profile_height(grade, -geometry.BACK_M), 0.0, places=12)
            self.assertAlmostEqual(geometry.profile_height(grade, geometry.FORWARD_M), 0.0, places=12)

    def test_spawn_clearance_attachment_distance_and_stop_point_stay_on_lane(self):
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
                self.assertLess(root[0]+.03+episode.STOP_DISTANCE_M,
                                geometry.FORWARD_M-geometry.BOUNDARY_MARGIN_M)
            # STOP 触发点仍落在车道内、且**已越过坡面出口**（用户要求越过坡后才置零指令）
            stop_x = robot[0] + 0.03 + episode.STOP_DISTANCE_M
            self.assertLess(stop_x, geometry.FORWARD_M-geometry.BOUNDARY_MARGIN_M)
            self.assertGreater(episode.STOP_DISTANCE_M, geometry.FLAT_OUT_START_M)
            self.assertAlmostEqual(geometry.profile_height(grade, 0.0), 0.0, places=12)
            self.assertGreaterEqual(mixed.arc_length_upper_bound(grade, episode.STOP_DISTANCE_M),
                                    episode.STOP_DISTANCE_M - 1e-12)

    def test_every_speed_arrives_before_the_discretized_timeout(self):
        # timeout 用**混合地面最长行程上界**（10 m 水平 STOP 点 → 波浪给 10.1160 m）
        worst = mixed.arc_length_upper_bound(geometry.MAX_GRADE_DEG, episode.STOP_DISTANCE_M)
        limit = episode.episode_timeout_s(worst, episode.SPEED_RANGE[0])
        # 2026-10-10 工作域收紧：最慢速度 0.4 → 0.5 m/s；2026-10-16 加入波浪块后上界由波浪
        # 给出（10.1160 m）⇒ 1 + 10.1160/0.5 + 2（默认 margin TIMEOUT_MARGIN_S=2）= 23.232 s；
        # 训练 cfg 用 POST_STOP_WINDOW_S=3 ⇒ 24.2321 s / 485 步（ceil），见下一个测试。
        self.assertLess(abs(limit - 23.2321), 0.01)
        slowest = episode.SPEED_RANGE[0]
        for index in range(int(round((episode.SPEED_RANGE[1] - slowest) / .01)) + 1):
            speed = slowest + index*.01
            arrival = episode.SETTLE_TIME_S + worst/speed
            # Includes one 50 ms action interval and one termination sampling interval.
            self.assertLess(arrival+.1, limit)
        self.assertEqual(episode.episode_timeout_s(12, .4), 33)
        for args in ((0,.4), (10,0), (10,-1), (float('inf'),.4)):
            with self.assertRaises(ValueError):
                episode.episode_timeout_s(*args)

    def test_training_timeout_uses_the_stop_window_margin(self):
        """训练 cfg 的 `episode_length_s` = 最慢速度 + **停车窗口** margin（与 0.5 m/s 自洽）。

        `upper_env_cfg.__post_init__` 用的是 `margin=POST_STOP_WINDOW_S`（3.0 s），不是默认的
        `TIMEOUT_MARGIN_S`（2.0 s）⇒ 工作域收紧后

            1（settle/tow_start）+ 10.1160/0.5（最慢速度走完混合地面的最长行程上界）+ 3（停车窗口）
            = **24.2321 s** ⇒ `max_episode_length` = **485 步**（20 Hz；Isaac Lab 0.45.9 的
            `ManagerBasedRLEnv.max_episode_length` 用 `math.ceil(episode_length_s / step_dt)`，
            见 `manager_based_rl_env.py:104`；截断口径是 483 步）。

        旧值 0.4 m/s ⇒ 29.231 s / ceil ⇒ 585 步（2026-10-09 的记录写 564/584 是截断口径，
        比运行值少 0–1 步）。这里钉住耦合关系本身（数值改了必须一起改），避免只改
        `SPEED_RANGE` 而把超时/步数留在旧口径。
        """
        worst = mixed.arc_length_upper_bound(geometry.MAX_GRADE_DEG, episode.STOP_DISTANCE_M)
        timeout = episode.episode_timeout_s(worst, episode.SPEED_RANGE[0],
                                            episode.SETTLE_TIME_S,
                                            margin=episode.POST_STOP_WINDOW_S)
        self.assertLess(abs(timeout - 24.2321), 0.01)
        self.assertEqual(int(timeout / 0.05), 484)          # 截断
        self.assertEqual(math.ceil(timeout / 0.05), 485)    # Isaac Lab 运行值
        # 停车窗口自洽：最慢速度下到 STOP 点后仍剩约 POST_STOP_WINDOW_S 秒
        arrival = episode.SETTLE_TIME_S + worst / episode.SPEED_RANGE[0]
        self.assertLess(abs((timeout - arrival) - episode.POST_STOP_WINDOW_S), 0.01)
        # 水平 10 m 在 0.5 m/s 下 20 s < 24.2321 s（最慢速度也能走到 STOP 点）
        self.assertLess(episode.SETTLE_TIME_S + episode.STOP_DISTANCE_M
                        / episode.SPEED_RANGE[0], timeout)


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
        xs = torch.tensor([3.75, 3.75, 3.75])          # 上坡段中点 ⇒ 高度 = 1.5·tan(grade)
        values = module.profile_height_tensor(grades, xs).tolist()
        self.assertAlmostEqual(values[0], 0.0, places=9)
        self.assertAlmostEqual(values[1], 1.5*math.tan(math.radians(5.0)), delta=1e-6)
        self.assertAlmostEqual(values[2], 1.5*math.tan(math.radians(10.0)), delta=1e-6)


if __name__ == '__main__':
    unittest.main()
