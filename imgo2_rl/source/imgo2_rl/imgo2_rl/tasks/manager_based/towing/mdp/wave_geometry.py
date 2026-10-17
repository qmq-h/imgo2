"""Three-period equal-amplitude sine wave over the same window as the slope profile.

2026-10-16（用户口径）：「最后添加一个波浪的起伏地形，**和上下坡占据同样的位置**，最后用一个
平地收尾」+「振幅不要太大，先定义大概 10 cm」。

- **窗口与平坡完全相同**：占 lane 局部 x = 2.25–9.0 m（`slope_geometry.UP_START_M` …
  `FLAT_OUT_START_M`），出生平地（0–2.25）与出口平地（9.0–12.75）都不动 ⇒ 10 m 的 STOP 点
  仍落在平地上（用户要的「最后用一个平地收尾」）。
- **形状 = 3 周期等幅正弦**：周期 λ = 2.25 m（3 × 2.25 = 6.75 = 窗口长度），振幅 A = 10 cm。
  等幅段的最大纵向坡度 = `atan(2πA/λ)` = **15.603°**（`connection_grid.WAVE_MAX_SLOPE_DEG`，
  也是该列 `slope_degrees` 报出的值）。峰谷差 20 cm，谷底 −10 cm 仍在板厚 35 cm 之内。
- **两端平滑归零**：乘一个半个周期长的 `cos²` 包络（`TAPER_M = λ/2 = 1.125 m`）——
  在 u = 0 与 u = L 处高度的**一阶导也为 0**，与前后平地 C1 衔接（不会像纯截断正弦那样在
  接缝处留一个折角）。中间 4 个极值是满幅。
- **横向**：`|y| ≤ 2 m` 满幅、外侧 1 m 用 `cos²` 收到 0（与 rolling / 减速带同一套横向因子，
  保证相邻列（10° 坡列 ↔ 波浪列）接缝处高度都是 0）。

几何常量（周期/振幅/最大坡度）放在 `connection_grid`（最底层模块，`slope_degrees` 要用），
本模块取用；`test_towing_wave_geometry` 交叉核对两处一致，并逐个采样验证
「真实最大坡度 ≤ 解析值」。
"""

from __future__ import annotations

import math

if __package__:
    from . import slope_geometry as layout
    from .connection_grid import (
        COLUMNS, ROWS, WAVE_MAX_SLOPE_DEG, WAVE_PEAK_M, WAVE_PERIOD_COUNT,
        WAVE_PERIOD_M, slope_degrees)
else:  # 按文件路径直接加载（离线测试的既有做法）时没有包上下文
    import slope_geometry as layout  # type: ignore[no-redef]
    from connection_grid import (  # type: ignore[no-redef]
        COLUMNS, ROWS, WAVE_MAX_SLOPE_DEG, WAVE_PEAK_M, WAVE_PERIOD_COUNT,
        WAVE_PERIOD_M, slope_degrees)

#: 波浪窗口 = 与平坡相同的剖面窗口；波浪在该窗口内跑满 3 个整周期。
START_M = layout.UP_START_M                    # 2.25
END_M = layout.FLAT_OUT_START_M                # 9.0
SPAN_M = END_M - START_M                       # 6.75
PERIOD_M = WAVE_PERIOD_M                       # 2.25
PERIOD_COUNT = WAVE_PERIOD_COUNT               # 3
PEAK_M = WAVE_PEAK_M                           # 0.10（振幅，用户 2026-10-16 口径）
TAPER_M = PERIOD_M / 2.0                       # 两端各收一个半周期（1.125 m）

#: 波数 k = 2π/λ 与解析最大坡度（与 `connection_grid.WAVE_MAX_SLOPE_DEG` 同式）。
WAVE_NUMBER = 2.0 * math.pi / PERIOD_M
MAX_SLOPE_DEG = WAVE_MAX_SLOPE_DEG

#: 横向：中央 `|y| ≤ LATERAL_FLAT_HALF_WIDTH_M` 满幅，外侧 `LATERAL_TAPER_M` 内收到 0。
LATERAL_FLAT_HALF_WIDTH_M = 2.0
LATERAL_TAPER_M = layout.HALF_WIDTH_M - LATERAL_FLAT_HALF_WIDTH_M   # 1.0

#: mesh 采样步长（m）：纵向 0.1（弦高误差 ≤ 1 mm）、横向收零段 0.2、中央满幅段 0.5。
X_STEP_M = 0.1
Y_STEP_FLAT_M = 0.5
Y_STEP_TAPER_M = 0.2

#: 弧长积分的步长与安全余量（timeout 用；见 `arc_length_upper_bound`）。
ARC_STEP_M = 1.0e-3
ARC_SAFETY = 1.001
ARC_MARGIN_M = 0.001

if not math.isclose(PERIOD_COUNT * PERIOD_M, SPAN_M, abs_tol=1e-12):
    raise ValueError("3 个周期必须正好铺满与平坡相同的剖面窗口（2.25–9.0 m）")
if not 0.0 < 2.0 * TAPER_M < SPAN_M:
    raise ValueError("两端收零段之和必须小于窗口长度（否则没有等幅段）")
if math.isclose(PEAK_M, 0.0) or PEAK_M < 0.0:
    raise ValueError("振幅必须为正（否则不是起伏地形）")


def lateral_factor(y: float) -> float:
    """横向权重：中央满幅 1、外侧 `cos²` 收到 0（`|y| ≥ HALF_WIDTH_M` 精确为 0）。"""
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


def _envelope(u: float) -> float:
    """包络：两端各 `TAPER_M` 内 `sin²` 升/降，中间等幅 1（窗口外为 0）。"""
    if u <= 0.0 or u >= SPAN_M:
        return 0.0
    if u >= TAPER_M and u <= SPAN_M - TAPER_M:
        return 1.0
    edge = u if u < TAPER_M else SPAN_M - u
    return math.sin(0.5 * math.pi * edge / TAPER_M) ** 2


def _envelope_derivative(u: float) -> float:
    """包络的一阶导（两端为 0 ⇒ 与前后平地 C1 衔接）。"""
    if u <= 0.0 or u >= SPAN_M:
        return 0.0
    if TAPER_M <= u <= SPAN_M - TAPER_M:
        return 0.0
    if u < TAPER_M:
        return (0.5 * math.pi / TAPER_M) * math.sin(math.pi * u / TAPER_M)
    return -(0.5 * math.pi / TAPER_M) * math.sin(math.pi * (SPAN_M - u) / TAPER_M)


def profile_height(grade_deg: float, x: float, y: float = 0.0) -> float:
    """`(x, y)` 处的波面高度（lane 局部坐标；窗口外精确为 0）。

    **与档位无关**：几何是固定尺寸（用户口径），`grade_deg` 只做量程校验、并让调用签名与
    hill/rolling 一致（混合网格按列分派时会把该列的 `slope_degrees` 传进来）。
    """
    _validate(grade_deg, x)
    lateral = lateral_factor(y)
    if lateral == 0.0:
        return 0.0
    u = x - START_M
    if u <= 0.0 or u >= SPAN_M:
        return 0.0
    return PEAK_M * math.sin(WAVE_NUMBER * u) * _envelope(u) * lateral


def profile_slope_degrees(grade_deg: float, x: float, y: float = 0.0) -> float:
    """`(x, y)` 处的**纵向**局部坡度（deg，带符号：上坡为正、下坡为负、窗口外为 0）。

    解析导数 `∂h/∂x = A·[k·cos(k u)·env(u) + sin(k u)·env'(u)]·lateral(y)`。
    """
    _validate(grade_deg, x)
    lateral = lateral_factor(y)
    if lateral == 0.0:
        return 0.0
    u = x - START_M
    if u <= 0.0 or u >= SPAN_M:
        return 0.0
    slope = PEAK_M * (WAVE_NUMBER * math.cos(WAVE_NUMBER * u) * _envelope(u)
                      + math.sin(WAVE_NUMBER * u) * _envelope_derivative(u)) * lateral
    return math.degrees(math.atan(slope))


def profile_frame(grade_deg: float, x: float, y: float = 0.0):
    """`(x, y)` 处的坡面坐标系 (切向, 法向)（用纵向局部坡度，与 hill/rolling 同口径）。"""
    return layout.slope_frame(profile_slope_degrees(grade_deg, x, y))


def arc_length_upper_bound(grade_deg: float, x: float) -> float:
    """从 lane 原点沿波面走到 `x` 的行程**上界**（timeout 用）。

    波形是光滑正弦，没有闭式弧长，所以用固定步长数值积分 `∫√(1+h'²)du`（步长 1 mm，
    实测截断误差 ≪ 1 mm），再乘 `ARC_SAFETY`(1.001) 并加 1 mm ⇒ 稳定、保守、可复现。
    10 m 目标（走满 6.75 m 波浪 + 前后平地）约 **10.13 m**，是该网格里最长的行程上界
    （10° 坡列 10.0926 m、rolling 10.1041 m、减速带 10.0236 m）。
    """
    _validate(grade_deg, x)
    if x <= START_M:
        return x
    span = min(x, END_M) - START_M
    steps = max(1, int(math.ceil(span / ARC_STEP_M)))
    step = span / steps
    total = 0.0
    for index in range(steps):
        u = index * step + 0.5 * step
        slope = PEAK_M * (WAVE_NUMBER * math.cos(WAVE_NUMBER * u) * _envelope(u)
                          + math.sin(WAVE_NUMBER * u) * _envelope_derivative(u))
        total += math.sqrt(1.0 + slope * slope) * step
    return min(x, END_M) - span + total * ARC_SAFETY + ARC_MARGIN_M + max(0.0, x - END_M)


def _sample_ys() -> list[float]:
    """横向采样点：中央满幅段 0.5 m 步长、外缘收零段 0.2 m 步长（0 与 ±HALF 都取到）。"""
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
    """纵向采样点：窗口内等距密采样（含两端），窗口外只留板的两端。"""
    count = math.ceil(SPAN_M / X_STEP_M)
    xs = [-layout.BACK_M]
    xs.extend(START_M + SPAN_M * index / count for index in range(count + 1))
    xs.append(layout.FORWARD_M)
    if any(b <= a for a, b in zip(xs, xs[1:])):
        raise ValueError("纵向采样点必须严格递增")
    return xs


def tile_mesh(row: int, column: int):
    """一块闭合的波浪板（上表面 = 波面，底面 = −THICKNESS 的平底）。"""
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
