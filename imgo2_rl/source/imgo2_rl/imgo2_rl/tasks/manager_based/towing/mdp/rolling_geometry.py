"""Smooth, irregular rolling heightfield; standard-library geometry only.

Four unequal raised-cosine hills replace the single hill in an optional task.
Height and first derivatives join continuously; no steps or angular crests.
Lane edges taper to zero, so adjacent grade columns also join continuously.
"""

import math

if __package__:
    from . import slope_geometry as layout
    from .connection_grid import COLUMNS, ROWS, slope_degrees
else:
    import slope_geometry as layout
    from connection_grid import COLUMNS, ROWS, slope_degrees

START_M = layout.UP_START_M
END_M = layout.FLAT_OUT_START_M
HILL_WIDTHS_M = (1.5, 2.0, 1.25, 2.0)
X_STEP_M = 0.1
Y_STEP_M = 0.5
assert sum(HILL_WIDTHS_M) == END_M - START_M


def profile_height(grade_deg, x, y=0.0):
    rise = layout.profile_rise(grade_deg)
    if not all(math.isfinite(v) for v in (x, y)):
        raise ValueError('coordinates must be finite')
    if abs(y) >= layout.HALF_WIDTH_M:
        return 0.0
    start = START_M
    for width in HILL_WIDTHS_M:
        if start < x < start + width:
            # h'=rise*sin(phase), hence maximum longitudinal slope is grade_deg.
            phase = 2 * math.pi * (x - start) / width
            lateral = math.cos(math.pi * y / (2 * layout.HALF_WIDTH_M)) ** 2
            return rise * width / (2 * math.pi) * (1 - math.cos(phase)) * lateral
        start += width
    return 0.0


def profile_slope_degrees(grade_deg, x, y=0.0):
    rise = layout.profile_rise(grade_deg)
    if not all(math.isfinite(v) for v in (x, y)):
        raise ValueError('coordinates must be finite')
    if abs(y) >= layout.HALF_WIDTH_M:
        return 0.0
    start = START_M
    for width in HILL_WIDTHS_M:
        if start < x < start + width:
            lateral = math.cos(math.pi * y / (2 * layout.HALF_WIDTH_M)) ** 2
            return math.degrees(math.atan(rise * math.sin(2 * math.pi * (x - start) / width) * lateral))
        start += width
    return 0.0


def arc_length_upper_bound(grade_deg, x):
    """Conservative centreline distance for the existing episode deadline."""
    rise = layout.profile_rise(grade_deg)
    if not math.isfinite(x):
        raise ValueError('x must be finite')
    rough_distance = max(0.0, min(x, END_M) - START_M)
    return x + rough_distance * (math.sqrt(1 + rise * rise) - 1)


def tile_mesh(row, column):
    origin = layout.tile_origin(row, column)
    grade = slope_degrees(column, row)
    if grade == 0:
        return layout.tile_mesh(row, column)
    # Include every hill junction exactly, independent of the sampling step.
    xs = [-layout.BACK_M, START_M]
    start = START_M
    for width in HILL_WIDTHS_M:
        count = math.ceil(width / X_STEP_M)
        xs.extend(start + width * i / count for i in range(1, count + 1))
        start += width
    xs.append(layout.FORWARD_M)
    count_y = math.ceil(2 * layout.HALF_WIDTH_M / Y_STEP_M)
    ys = [-layout.HALF_WIDTH_M + 2 * layout.HALF_WIDTH_M * i / count_y
          for i in range(count_y + 1)]
    ny = len(ys)
    vertices = [(x, y, profile_height(grade, x, y)) for x in xs for y in ys]
    faces = []
    for i in range(len(xs) - 1):
        for j in range(ny - 1):
            a = i * ny + j
            faces.extend(((a, a + ny, a + ny + 1), (a, a + ny + 1, a + 1)))
    # Counterclockwise perimeter, viewed from above. All boundary heights are zero.
    perimeter = ([i * ny for i in range(len(xs))]
                 + [(len(xs) - 1) * ny + j for j in range(1, ny)]
                 + [i * ny + ny - 1 for i in range(len(xs) - 2, -1, -1)]
                 + [j for j in range(ny - 2, 0, -1)])
    centre = len(vertices)
    vertices.append((0.0, 0.0, -layout.THICKNESS_M))
    bottom = []
    for index in perimeter:
        bottom.append(len(vertices))
        x, y, _ = vertices[index]
        vertices.append((x, y, -layout.THICKNESS_M))
    for k, a in enumerate(perimeter):
        nxt = (k + 1) % len(perimeter)
        b, c, d = perimeter[nxt], bottom[k], bottom[nxt]
        faces.extend(((a, c, d), (a, d, b), (centre, d, c)))
    vertices = [(origin[0] + x, origin[1] + y, origin[2] + z) for x, y, z in vertices]
    return vertices, faces, origin


# 整块地面的组装在 `mixed_geometry.grid_mesh()`（2026-10-16 起地面是混合网格：平滑列走本模块、
# 减速带列走 `speedbump_geometry`），本模块只负责单块板的几何。
