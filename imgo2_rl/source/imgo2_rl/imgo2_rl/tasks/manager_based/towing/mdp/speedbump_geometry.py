"""One isosceles-trapezoid speed bump per lane; standard-library geometry only.

2026-10-16（用户口径）：「减速带就只设置 300 底、100 顶、50 高的一个等腰梯形形状，都是 mm 单位」
⇒ 每条 lane 只有**一条**梯形凸条，而且是**固定尺寸**（与列的量级档无关）：

    底 300 mm ┊ 斜坡 100 mm ┊ 平顶 100 mm ┊ 斜坡 100 mm ┊ 底 300 mm
    高度 0 → 50 mm → 保持 50 mm → 0；侧坡 = atan(50/100) = **26.565°**

尺寸常量在 `connection_grid`（最底层模块，`slope_degrees` 要报出这个真实坡度，见那里的注释），
本模块取用；`test_towing_speedbump_geometry` 交叉核对两处一致。

位置：剖面区（`slope_geometry` 的 2.25–9.0 m）**正中** x = 5.625 m，所以一个回合只过一次冲击，
出生点（x≈0）与 STOP 点（10 m）都在平地上。

横向：`|y| ≤ 2 m` 满高、外侧 1 m 用 `cos²` 平滑收到 0（`lateral_factor`）⇒ 同一列内横向连续，
且与本块两侧的平坡列在接缝处高度都是 0（平坡列在 |y| = 3 处高度也为 0？**不是** —— hill 剖面与
y 无关，因此「平地/5°/10° 坡列 ↔ 减速带列」的列接缝在坡顶附近仍有 0～0.53 m 的矮坎，这与既有的
「平地↔5°」「5°↔10°」接缝同性质，是混合地形的固有台阶，靠中线 PD 保持车道，2026-10-16 记录在案）。

mesh：梯形是**分段线性**的，所以纵向只需把 4 个折点（斜坡起点、坡顶两端、斜坡终点）显式采样，
不需要密采样；横向仍在收零段用 0.2 m 步长。闭合实体 = 上表面网格 + 平的底面 + 周边侧壁。
"""

from __future__ import annotations

import math

if __package__:
    from . import slope_geometry as layout
    from .connection_grid import (
        COLUMNS, ROWS, SPEEDBUMP_BOTTOM_M, SPEEDBUMP_HEIGHT_M, SPEEDBUMP_RAMP_DEG,
        SPEEDBUMP_RAMP_RUN_M, SPEEDBUMP_TOP_M, slope_degrees)
else:  # 按文件路径直接加载（离线测试的既有做法）时没有包上下文
    import slope_geometry as layout  # type: ignore[no-redef]
    from connection_grid import (  # type: ignore[no-redef]
        COLUMNS, ROWS, SPEEDBUMP_BOTTOM_M, SPEEDBUMP_HEIGHT_M, SPEEDBUMP_RAMP_DEG,
        SPEEDBUMP_RAMP_RUN_M, SPEEDBUMP_TOP_M, slope_degrees)

#: 减速带区（= 剖面区）在 lane 局部 x 上的范围；坡列与减速带列共用这两个边界。
START_M = layout.UP_START_M                    # 2.25
END_M = layout.FLAT_OUT_START_M                # 9.0

#: 梯形尺寸（m）：底/顶/高 + 单侧斜坡的水平跑动与坡度（用户口径见模块 docstring）。
BUMP_BOTTOM_M = SPEEDBUMP_BOTTOM_M             # 0.30
BUMP_TOP_M = SPEEDBUMP_TOP_M                   # 0.10
BUMP_HEIGHT_M = SPEEDBUMP_HEIGHT_M             # 0.05
RAMP_RUN_M = SPEEDBUMP_RAMP_RUN_M              # 0.10
RAMP_DEG = SPEEDBUMP_RAMP_DEG                  # 26.565…

#: 凸条在剖面区**正中**：x = (2.25 + 9.0)/2 = 5.625，四个折点依次排开。
BUMP_CENTER_X_M = (START_M + END_M) / 2.0
BUMP_START_M = BUMP_CENTER_X_M - BUMP_BOTTOM_M / 2.0
BUMP_TOP_START_M = BUMP_START_M + RAMP_RUN_M
BUMP_TOP_END_M = BUMP_TOP_START_M + BUMP_TOP_M
BUMP_END_M = BUMP_START_M + BUMP_BOTTOM_M

#: 横向：中央 `|y| ≤ LATERAL_FLAT_HALF_WIDTH_M` 满高，外侧 `LATERAL_TAPER_M` 内收到 0。
LATERAL_FLAT_HALF_WIDTH_M = 2.0
LATERAL_TAPER_M = layout.HALF_WIDTH_M - LATERAL_FLAT_HALF_WIDTH_M   # 1.0

#: mesh 采样步长（m）：纵向只需折点；横向收零段 0.2（弦高误差 ≤ 0.4 mm）、中央满高段 0.5。
Y_STEP_FLAT_M = 0.5
Y_STEP_TAPER_M = 0.2

if not START_M < BUMP_START_M < BUMP_TOP_START_M < BUMP_TOP_END_M < BUMP_END_M < END_M:
    raise ValueError("梯形凸条必须整条落在剖面区内，且四个折点严格递增")
if not math.isclose(BUMP_END_M - BUMP_START_M, BUMP_BOTTOM_M, abs_tol=1e-12):
    raise ValueError("梯形底宽 = 两条斜坡的水平跑动 + 平顶宽度")


def lateral_factor(y: float) -> float:
    """横向权重：中央满高 1、外侧 `cos²` 收到 0（`|y| ≥ HALF_WIDTH_M` 精确为 0）。"""
    if not math.isfinite(y):
        raise ValueError(f"y 必须是有限值，收到 {y!r}")
    distance = abs(y)
    if distance <= LATERAL_FLAT_HALF_WIDTH_M:
        return 1.0
    if distance >= layout.HALF_WIDTH_M:
        return 0.0
    phase = math.pi * (distance - LATERAL_FLAT_HALF_WIDTH_M) / (2.0 * LATERAL_TAPER_M)
    return math.cos(phase) ** 2


def _validate(grade_deg: float, x: float) -> None:
    """量程校验：档位必须是 [-45, 45] 内有限值、x 必须有限（与 hill/rolling 同口径）。"""
    layout.profile_rise(grade_deg)
    if not math.isfinite(x):
        raise ValueError(f"x 必须是有限值，收到 {x!r}")


def profile_height(grade_deg: float, x: float, y: float = 0.0) -> float:
    """`(x, y)` 处的减速带高度（lane 局部坐标；凸条外精确为 0）。

    **与档位无关**：几何是固定尺寸（用户口径），`grade_deg` 只做量程校验、并让调用签名与
    hill/rolling 一致（混合网格按列分派时会把该列的 `slope_degrees` 传进来）。
    """
    _validate(grade_deg, x)
    lateral = lateral_factor(y)
    if lateral == 0.0 or not BUMP_START_M < x < BUMP_END_M:
        return 0.0
    if x <= BUMP_TOP_START_M:
        height = BUMP_HEIGHT_M * (x - BUMP_START_M) / RAMP_RUN_M
    elif x <= BUMP_TOP_END_M:
        height = BUMP_HEIGHT_M
    else:
        height = BUMP_HEIGHT_M * (BUMP_END_M - x) / RAMP_RUN_M
    return height * lateral


def profile_slope_degrees(grade_deg: float, x: float, y: float = 0.0) -> float:
    """`(x, y)` 处的**纵向**局部坡度（deg，带符号：上坡为正、下坡为负、平地为 0）。

    斜坡段 = `atan(tan(26.565°) × lateral_factor(y))`（横向收零后坡度也变小）；
    平顶与凸条之外为 0。
    """
    _validate(grade_deg, x)
    lateral = lateral_factor(y)
    if lateral == 0.0 or not BUMP_START_M < x < BUMP_END_M:
        return 0.0
    rise = (BUMP_HEIGHT_M / RAMP_RUN_M) * lateral
    if x < BUMP_TOP_START_M:
        return math.degrees(math.atan(rise))
    if x <= BUMP_TOP_END_M:
        return 0.0
    return -math.degrees(math.atan(rise))


def profile_frame(grade_deg: float, x: float, y: float = 0.0):
    """`(x, y)` 处的坡面坐标系 (切向, 法向)（用纵向局部坡度，与 hill/rolling 同口径）。"""
    return layout.slope_frame(profile_slope_degrees(grade_deg, x, y))


def arc_length_upper_bound(grade_deg: float, x: float) -> float:
    """从 lane 原点沿坡面走到 `x` 的行程。

    梯形剖面**分段线性**，所以这就是精确弧长（不是上界）；仍沿用 hill/rolling 的函数名与
    签名，便于 `mixed_geometry` 取三者最大值。10 m 目标（凸条已整条走过）比水平多
    `2·hypot(0.1, 0.05) + 0.1 − 0.3 = 0.023607 m` ⇒ **10.023607 m**。
    """
    _validate(grade_deg, x)
    ramp = math.hypot(RAMP_RUN_M, BUMP_HEIGHT_M)
    if x <= BUMP_START_M:
        return x
    if x >= BUMP_END_M:
        return x + (2.0 * ramp + BUMP_TOP_M - BUMP_BOTTOM_M)
    if x <= BUMP_TOP_START_M:
        return BUMP_START_M + (x - BUMP_START_M) * ramp / RAMP_RUN_M
    if x <= BUMP_TOP_END_M:
        return BUMP_START_M + ramp + (x - BUMP_TOP_START_M)
    return BUMP_START_M + ramp + BUMP_TOP_M + (x - BUMP_TOP_END_M) * ramp / RAMP_RUN_M


def _sample_ys() -> list[float]:
    """横向采样点：中央满高段 0.5 m 步长、外缘收零段 0.2 m 步长（0 与 ±HALF 都取到）。"""
    taper_count = math.ceil(LATERAL_TAPER_M / Y_STEP_TAPER_M)
    taper = [LATERAL_FLAT_HALF_WIDTH_M
             + LATERAL_TAPER_M * index / taper_count for index in range(taper_count + 1)]
    crown_count = math.ceil(2.0 * LATERAL_FLAT_HALF_WIDTH_M / Y_STEP_FLAT_M)
    crown = [-LATERAL_FLAT_HALF_WIDTH_M
             + 2.0 * LATERAL_FLAT_HALF_WIDTH_M * index / crown_count
             for index in range(crown_count + 1)]
    ys = [-value for value in reversed(taper)] + crown[1:-1] + taper
    if any(b <= a for a, b in zip(ys, ys[1:])):
        raise ValueError("横向采样点必须严格递增")
    return ys


def _sample_xs() -> list[float]:
    """纵向采样点：平地只留端点，梯形显式取 4 个折点（分段线性 ⇒ 不需要密采样）。"""
    xs = [-layout.BACK_M, BUMP_START_M, BUMP_TOP_START_M, BUMP_TOP_END_M, BUMP_END_M,
          layout.FORWARD_M]
    if any(b <= a for a, b in zip(xs, xs[1:])):
        raise ValueError("纵向采样点必须严格递增")
    return xs


def tile_mesh(row: int, column: int):
    """一块闭合的减速带板（凸条 + 前后平地，底面 = −THICKNESS 的平底）。"""
    origin = layout.tile_origin(row, column)
    grade = slope_degrees(column, row)
    xs, ys = _sample_xs(), _sample_ys()
    ny = len(ys)
    vertices = [(x, y, profile_height(grade, x, y)) for x in xs for y in ys]
    faces = []
    for i in range(len(xs) - 1):
        for j in range(ny - 1):
            a = i * ny + j
            faces.extend(((a, a + ny, a + ny + 1), (a, a + ny + 1, a + 1)))
    # 俯视逆时针的周边一圈；边界高度全为 0（x 两端是平地、y 两端横向因子为 0）。
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
