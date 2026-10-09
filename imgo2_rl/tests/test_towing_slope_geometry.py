"""Offline geometry/deadline checks against the actual terrain mesh generator."""

from collections import Counter
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
    'source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/mdp'))
import connection_grid as grid
import slope_geometry as geometry
import episode_geometry as episode


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


class SlopeGeometryTests(unittest.TestCase):
    def test_each_length_has_balanced_terrain_and_connection_types(self):
        for row in range(grid.ROWS):
            specs = [grid.env_spec(row*grid.COLUMNS+c) for c in range(grid.COLUMNS)]
            self.assertEqual(Counter(s['slope_degrees'] for s in specs),
                             {0: 20, 5: 5, -5: 5, 10: 5, -10: 5})
            for start, end, expected in ((0, 20, (8, 8, 4)),
                                         (20, 30, (4, 4, 2)), (30, 40, (4, 4, 2))):
                counts = Counter(s['model_name'] for s in specs[start:end])
                self.assertEqual(tuple(counts[n] for n in ('compliant', 'rigid', 'inextensible')),
                                 expected)
        # Every stiffness level occurs on both slope signs over the length grid.
        for start in (20, 30):
            for level in range(4):
                self.assertEqual({grid.slope_degrees(start+level, r) for r in range(grid.ROWS)},
                                 {5, -5} if start == 20 else {10, -10})

    def test_closed_mesh_top_normal_and_origin_match_logical_cells(self):
        vertices, faces, origins = geometry.grid_mesh()
        self.assertEqual((len(vertices), len(faces)), (6400, 9600))
        self.assertEqual(len({o for row in origins for o in row}), grid.GRID_SIZE)
        for row in range(grid.ROWS):
            for col in range(grid.COLUMNS):
                verts, tris, origin = geometry.tile_mesh(row, col)
                self.assertEqual(origin, origins[row][col])
                tangent, normal = geometry.slope_frame(grid.slope_degrees(col, row))
                for v in verts[:4]:
                    self.assertAlmostEqual(dot(tuple(x-o for x,o in zip(v,origin)), normal), 0, places=10)
                edge_counts = Counter(tuple(sorted((f[i], f[(i+1)%3])))
                                      for f in tris for i in range(3))
                self.assertEqual(set(edge_counts.values()), {2})
                a,b,c = (verts[i] for i in tris[0])
                u,v = [b[i]-a[i] for i in range(3)], [c[i]-a[i] for i in range(3)]
                cross = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
                self.assertGreater(dot(cross, normal), 0)
                self.assertAlmostEqual(dot(tangent, normal), 0, places=12)

    def test_spawn_clearance_attachment_distance_and_goal_stay_on_lane(self):
        for index in range(grid.GRID_SIZE):
            spec = grid.env_spec(index)
            t,n = geometry.slope_frame(spec['slope_degrees'])
            robot,cart,rp,cp = geometry.attachment_root_positions(
                spec['slope_degrees'], spec['initial_distance'])
            self.assertAlmostEqual(dot(robot,n), .35, places=12)
            self.assertAlmostEqual(dot(cart,n), .18, places=12)
            self.assertAlmostEqual(math.dist(rp,cp), spec['initial_distance'], places=12)
            for root in (robot,cart):
                self.assertGreater(root[0]-.03, -geometry.BACK_M+geometry.BOUNDARY_MARGIN_M)
                self.assertLess(root[0]+.03+episode.GOAL_DISTANCE_M*t[0],
                                geometry.FORWARD_M-geometry.BOUNDARY_MARGIN_M)
            # The fall measure remains invariant while travelling on either ramp.
            endpoint = tuple(x+episode.GOAL_DISTANCE_M*d for x,d in zip(robot,t))
            self.assertAlmostEqual(dot(endpoint,n), .35, places=12)
            self.assertAlmostEqual(dot(endpoint,t)-dot(robot,t), episode.GOAL_DISTANCE_M, places=12)

    def test_every_speed_arrives_before_the_discretized_timeout(self):
        limit = episode.episode_timeout_s(episode.GOAL_DISTANCE_M, episode.SPEED_RANGE[0])
        self.assertEqual(limit, 28)
        for index in range(111):
            speed = .4 + index*.01
            arrival = episode.SETTLE_TIME_S + episode.GOAL_DISTANCE_M/speed
            # Includes one 50 ms action interval and one termination sampling interval.
            self.assertLess(arrival+.1, limit)
        self.assertEqual(episode.episode_timeout_s(12, .4), 33)
        for args in ((0,.4), (10,0), (10,-1), (float('inf'),.4)):
            with self.assertRaises(ValueError):
                episode.episode_timeout_s(*args)


if __name__ == '__main__':
    unittest.main()
