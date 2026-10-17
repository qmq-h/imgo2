"""The training ground as one mesh: flat + slope + speed-bump + wave columns.

2026-10-16（用户要求「直接把这个拼接到之前的设置中」）：物理地面**只有一张网格**，按列分派
地形族：

    列 0–19   平地（`slope_geometry` 的 0° 剖面 ⇒ 就是一块矩形板）
    列 20–29  上/下坡 5°（`slope_geometry`，或 rolling 模式下换成 `rolling_geometry`）
    列 30–39  上/下坡 10°（同上）
    列 40–49  **减速带**（`speedbump_geometry`：一条 300/100/50 mm 等腰梯形凸条）
    列 50–59  **波浪**（`wave_geometry`：3 周期等幅正弦，振幅 10 cm，占与平坡相同的
              2.25–9.0 m 窗口，两端平滑归零、之后是平地收尾）

`smooth` 参数选择前 40 列用哪套平滑高度场（`"hill"` = 单坡包、`"rolling"` = 连续起伏）；
减速带块与波浪块**始终**是各自的地形，不受它影响。相邻板仍然共边拼接（间距 = 板尺寸，见
`slope_geometry`），板外没有 ground。

⚠ 列接缝的落差：hill 列的高度与 y 无关，所以「平地↔5°↔10°」「10° 坡列 ↔ 减速带列」
「减速带列 ↔ 波浪列」的接缝在坡顶/波峰附近仍有矮坎（`slope_geometry` 的既有说明 +
2026-10-16 记录）；rolling / 减速带 / 波浪三族在 |y| = 3 处都收到 0，族内接缝无落差。
"""

from __future__ import annotations

if __package__:
    from .connection_grid import (
        COLUMNS, ROWS, SPEEDBUMP_TERRAIN, WAVE_TERRAIN, terrain_family)
    from . import slope_geometry as hill
    from . import speedbump_geometry as speedbump
    from . import wave_geometry as wave
else:  # 按文件路径直接加载（离线测试的既有做法）时没有包上下文
    from connection_grid import (  # type: ignore[no-redef]
        COLUMNS, ROWS, SPEEDBUMP_TERRAIN, WAVE_TERRAIN, terrain_family)
    import slope_geometry as hill  # type: ignore[no-redef]
    import speedbump_geometry as speedbump  # type: ignore[no-redef]
    import wave_geometry as wave  # type: ignore[no-redef]

#: 平滑段可选的高度场实现（名字 → 取用的模块）。
SMOOTH_GEOMETRY = ("hill", "rolling")


def _smooth_module(smooth: str):
    if smooth == "hill":
        return hill
    if smooth == "rolling":
        if __package__:
            from . import rolling_geometry
            return rolling_geometry
        import rolling_geometry  # type: ignore[no-redef]
        return rolling_geometry
    raise ValueError(f"unknown smooth terrain {smooth!r}（可选 {SMOOTH_GEOMETRY}）")


def tile_mesh(row: int, column: int, smooth: str = "hill"):
    """一块板：减速带列 / 波浪列走各自模块，其余列走所选平滑高度场。"""
    family = terrain_family(column)
    if family == SPEEDBUMP_TERRAIN:
        return speedbump.tile_mesh(row, column)
    if family == WAVE_TERRAIN:
        return wave.tile_mesh(row, column)
    return _smooth_module(smooth).tile_mesh(row, column)


def arc_length_upper_bound(grade_deg: float, x: float, smooth: str = "hill") -> float:
    """整块地面的弧长上界：平滑段 / 减速带段 / 波浪段各取上界后取最大值（timeout 用）。

    三个族的行程上界**都要覆盖**：hill 的整段保守口径（10° 档 10.092560 m）、rolling
    （10.1041 m）、减速带梯形的精确弧长（10.023607 m）与波浪正弦的数值上界（≈10.116 m）。
    呼叫方传 `MAX_GRADE_DEG`（= 平滑段最陡档）即可 —— 后两者的弧长与档位无关。
    2026-10-16 加入波浪块后，最大值由**波浪**给出（≈10.116 m）⇒ timeout 略增。
    """
    module = _smooth_module(smooth)
    smooth_bound = (module.arc_length_upper_bound(grade_deg, x) if smooth == "rolling"
                    else module.profile_arc_length(grade_deg, x))
    return max(smooth_bound, speedbump.arc_length_upper_bound(grade_deg, x),
               wave.arc_length_upper_bound(grade_deg, x))


def grid_mesh(smooth: str = "hill"):
    """Return vertices/faces and origins[row][column]; no simulator imports."""
    vertices, faces, origins = [], [], []
    for row in range(ROWS):
        row_origins = []
        for column in range(COLUMNS):
            points, triangles, origin = tile_mesh(row, column, smooth)
            offset = len(vertices)
            vertices.extend(points)
            faces.extend(tuple(offset + index for index in face) for face in triangles)
            row_origins.append(origin)
        origins.append(row_origins)
    return vertices, faces, origins
