"""`cmoe_terrains.py` 赛道几何的离线不变式（**无侧墙**版本）。

2026-09-24 用户决定：**不给赛道加侧墙**（"侧墙不该有"），"横移绕开"改由**奖励**端解决
（世界系速度跟踪 + 全局的中心线/朝向罚，见 `test_world_vel_tracking.py` 与 §29.18）。
本文件因此锁定两件事：①**赛道是开放的**（没有沿 y 边缘的薄墙，防止以后又被"顺手"加回来）；
②赛道本身的不变式（平台数量、出生点、出生平面）没被改坏。

`cmoe_terrains.py` 顶层 `import isaaclab...`（缺 `omni.log`）无法整模块 import，所以用
`sys.modules` 里临时塞假模块的方式 exec **真实源码**（`trimesh`／`numpy` 是真的）。
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TERRAINS = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/cmoe_terrains.py"

try:
    import numpy as np
    import trimesh
except ModuleNotFoundError as error:
    np = trimesh = None
    IMPORT_ERROR: object = error
else:
    IMPORT_ERROR = None

CASES = (
    ("gap", "CMoETrackGapTerrainCfg", "track_gap_terrain"),
    ("step", "CMoETrackStepTerrainCfg", "track_step_terrain"),
    ("stairs_up", "CMoETrackStairsTerrainCfg", "track_stairs_terrain"),
    ("stairs_down", "CMoETrackStairsTerrainCfg", "track_stairs_terrain"),
)


def _load_module():
    """在桩命名空间里 exec 真实源码（临时替换 isaaclab 的两个子模块，exec 完还原）。"""
    import dataclasses
    import sys
    import types

    @dataclasses.dataclass
    class _StubSubTerrainBaseCfg:
        size: tuple = (8.0, 8.0)
        difficulty_range: tuple = (0.0, 1.0)
        proportion: float = 1.0

    src = TERRAINS.read_text(encoding="utf-8")
    module_name = "cmoe_terrains_under_test"
    keys = ("isaaclab", "isaaclab.terrains", "isaaclab.utils", module_name)
    saved = {k: sys.modules.get(k) for k in keys}
    pkg = types.ModuleType("isaaclab")
    pkg.__path__ = []
    terrains = types.ModuleType("isaaclab.terrains")
    terrains.SubTerrainBaseCfg = _StubSubTerrainBaseCfg
    utils = types.ModuleType("isaaclab.utils")
    utils.configclass = dataclasses.dataclass
    module = types.ModuleType(module_name)
    sys.modules.update(
        {"isaaclab": pkg, "isaaclab.terrains": terrains, "isaaclab.utils": utils, module_name: module}
    )
    try:
        ns = module.__dict__
        ns.update({"np": np, "trimesh": trimesh})
        exec(compile(src, str(TERRAINS), "exec"), ns)  # noqa: S102 - 只执行仓库自己的地形函数
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return ns


def _thin_y_meshes(meshes):
    """在 y 方向很薄的盒子 —— 之前那版"侧墙"就是这个形状。"""
    return [m for m in meshes if float(m.extents[1]) <= 0.1]


def _platforms(meshes):
    return [m for m in meshes if float(m.extents[1]) > 0.1]


@unittest.skipIf(trimesh is None, f"trimesh/numpy unavailable: {IMPORT_ERROR}")
class TestTrackGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _load_module()

    def _cases(self, difficulty):
        ns = self.ns
        out = []
        for name, cls_name, fn_name in CASES:
            kwargs = {"size": (8.0, 4.0), "difficulty_range": (0.0, 1.0)}
            if cls_name == "CMoETrackStairsTerrainCfg":
                kwargs["ascending"] = name == "stairs_up"
            cfg = ns[cls_name](**kwargs)
            meshes, origin = ns[fn_name](difficulty, cfg)
            out.append((name, cfg, meshes, origin))
        return out

    def test_no_side_walls(self):
        """用户 2026-09-24 决定：赛道两侧**不加墙**（开放赛道，绕开改由奖励端约束）。"""
        for difficulty in (0.0, 0.5, 1.0):
            for name, _cfg, meshes, _origin in self._cases(difficulty):
                self.assertEqual(
                    _thin_y_meshes(meshes), [],
                    f"{name}@d={difficulty} 出现了沿 y 边缘的薄网格（疑似又加了侧墙）",
                )

    def test_obstacle_geometry_unchanged(self):
        cases = {name: (cfg, meshes) for name, cfg, meshes, _o in self._cases(0.5)}
        gap_cfg, gap_meshes = cases["gap"]
        step_cfg, step_meshes = cases["step"]
        stairs_cfg, stairs_meshes = cases["stairs_up"]
        self.assertEqual(len(_platforms(gap_meshes)), gap_cfg.num_gaps + 1)
        self.assertEqual(len(_platforms(step_meshes)), 1 + step_cfg.num_steps)
        self.assertEqual(len(_platforms(stairs_meshes)), 1 + stairs_cfg.num_steps + 1)

    def test_track_tiles_are_full_width(self):
        """每个平台都必须横贯整个 tile 宽度（4 m）—— 障碍在赛道内不可绕过。"""
        for name, _cfg, meshes, _origin in self._cases(0.5):
            for m in _platforms(meshes):
                self.assertAlmostEqual(float(m.extents[1]), 4.0, places=6,
                                       msg=f"{name} 有平台没有横贯赛道宽度")

    def test_spawn_point_is_unchanged(self):
        for name, _cfg, _meshes, origin in self._cases(0.5):
            self.assertAlmostEqual(float(origin[0]), 0.75, places=6, msg=f"{name} 出生点 x 被移动了")
            self.assertAlmostEqual(float(origin[1]), 2.0, places=6, msg=f"{name} 出生点 y 被移动了")

    def test_spawn_sits_on_the_spawn_plane(self):
        for name, _cfg, meshes, origin in self._cases(0.5):
            x, y, z = float(origin[0]), float(origin[1]), float(origin[2])
            hits = [
                float(m.bounds[1][2]) for m in _platforms(meshes)
                if m.bounds[0][0] <= x <= m.bounds[1][0] and m.bounds[0][1] <= y <= m.bounds[1][1]
            ]
            self.assertTrue(hits, f"{name} 出生点下方没有地面")
            self.assertAlmostEqual(max(hits), z, places=6, msg=f"{name} 出生点不在起始平面上")


if __name__ == "__main__":
    unittest.main()
