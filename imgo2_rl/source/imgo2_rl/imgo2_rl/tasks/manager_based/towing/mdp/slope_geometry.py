"""Standard-library geometry shared by the towing mesh and offline checks.

2026-10-09：坡面从「每格一个恒定坡度、按行列交替正负」改成**整合到一起的一条连续剖面**
（一整座小土包），每条 lane 沿 +X 依次是：

    平地 2.25 m → 上坡 3 m → 坡顶平段 0.75 m → 下坡 3 m → 平地 2.25 m
    x: 0 ──────── 2.25 ───── 5.25 ─────────── 6.0 ─────── 9.0 ─────── 11.25   (lane 局部坐标)

跑道总长也按同一比例从 **20 m 收到 15 m**（`BACK_M + FORWARD_M = 2.25 + 12.75`；用户
2026-10-09：「收窄的意思是不要 20 m，而是 15 m，对应缩小上下坡+平地的长度」）。机器人出生在
x=0（**平地上、姿态竖直**），走完 10 m 目标时落在**出口平地**上（剖面 11.25 m，出口平地
从 9.0 m 起）；上坡与下坡在**同一条 lane 内成对出现**，所以不再需要「坡度正负号交替」来
平衡方向。坡度量级由列决定（`connection_grid.slope_degrees` = 0 / 5 / 10 deg）。

**相邻 lane 共边拼接**（2026-10-09 用户要求「训练场景拼接到一起」）：
`ROW_SPACING_M = BACK_M + FORWARD_M`（15 m）、`COLUMN_SPACING_M = 2·HALF_WIDTH_M`（6 m），
即每块板的足迹正好等于网格间距 ⇒ 800 块板拼成**一整块连续地面**（板之间的内部侧面仍在，
但都落在整体内部、不可见）。此前间距各多 2 m、地形之外又没有 ground，机器人/小车走出自己
那块板或掉进缝里就会直接坠下去 —— 用户看到的「有些机器人直接到底」就是它。接缝处：行方向
（x）两侧都是平地、高度为 0 ⇒ 无落差；列方向只有「平地↔5°」「5°↔10°」两条分界有矮坎
（高 0～0.53 m、随 x 起伏），同档量的列之间剖面相同、接缝同样无落差。

坐标约定：`x` 沿坡前进，`y` 横向（±`HALF_WIDTH_M`），`z` 向上；lane 原点是剖面在
`x = 0` 处的表面点（`profile_height(·, 0) = 0`），因此 `body_z − origin_z − profile_height(x)`
就是「离坡面多高」，与 `slope_frame(局部坡度)` 一起构成坡面坐标系。
"""

from __future__ import annotations

import math

if __package__:
    from .connection_grid import COLUMNS, ROWS, slope_degrees
else:
    from connection_grid import COLUMNS, ROWS, slope_degrees

BACK_M = 2.25
FORWARD_M = 12.75
HALF_WIDTH_M = 3.0
# 行/列**共边拼接**（2026-10-09 用户要求「训练场景拼接到一起」）：间距 = 板尺寸，
# 不再留 2 m 间隙。此前板与板之间是空的（地形之外没有 ground），机器人/小车走出自己那块板、
# 或掉进缝里就会直接坠下去 —— 用户看到的「有些机器人直接到底」就是它。拼接后走出去会落到
# 相邻 lane 上：行方向（x）两侧都是平地、接缝处高度都是 0 ⇒ 无落差；列方向只有
# 「平地↔5°」「5°↔10°」两条分界会形成一道高 0～0.53 m、随 x 起伏的矮坎（同档量的列之间
# 剖面完全相同，接缝同样无落差）。
ROW_SPACING_M = BACK_M + FORWARD_M
COLUMN_SPACING_M = 2.0 * HALF_WIDTH_M
THICKNESS_M = 0.35
BOUNDARY_MARGIN_M = 0.6

# ---- 连续剖面（「平地 → 上坡 → 坡顶 → 下坡 → 平地」）各段边界（lane 局部 x） ---------------
# 各段长度（2026-10-09：跑道 20 m → 15 m，剖面按同一比例 0.75 缩放）
FLAT_IN_M = 2.25                # 出生点前方的平地：机器人 + 小车都在这一段
UP_M = 3.0                      # 上坡的水平长度
CREST_M = 0.75                  # 坡顶平段
DOWN_M = 3.0                    # 下坡的水平长度
EXIT_M = 2.25                   # 出坡后的平地（仍在 12.75 m 前向余量内）
UP_START_M = FLAT_IN_M                          # 2.25  上坡起点
CREST_START_M = UP_START_M + UP_M               # 5.25  坡顶起点
DOWN_START_M = CREST_START_M + CREST_M          # 6.0   下坡起点
FLAT_OUT_START_M = DOWN_START_M + DOWN_M        # 9.0   回到平地
PROFILE_LENGTH_M = FLAT_OUT_START_M + EXIT_M    # 11.25
MAX_GRADE_DEG = 10.0            # 剖面的最大坡度量级（= 网格里最陡的一档）

if PROFILE_LENGTH_M > FORWARD_M:
    raise ValueError("剖面长度超过 lane 的前向长度（FORWARD_M），出生/边界假设不再成立")
if BACK_M + FORWARD_M != 15.0:
    raise ValueError("跑道总长应为 15 m（2026-10-09 用户要求从 20 m 收到 15 m）")
if (ROW_SPACING_M, COLUMN_SPACING_M) != (BACK_M + FORWARD_M, 2.0 * HALF_WIDTH_M):
    raise ValueError("行/列间距必须等于板尺寸（相邻 lane 共边拼接）；改间距前先想清楚会不会又留出虚空")


def tile_origin(row, column):
    """True 20-by-40 mesh origins, independent of GridCloner's square layout."""
    if not 0 <= row < ROWS or not 0 <= column < COLUMNS:
        raise ValueError("invalid terrain cell")
    return ((row - (ROWS - 1) / 2) * ROW_SPACING_M,
            (column - (COLUMNS - 1) / 2) * COLUMN_SPACING_M, 0.0)


def slope_frame(degrees):
    """恒定坡度的 (切向, 法向)：`tangent = (cosθ, 0, sinθ)`、`normal = (−sinθ, 0, cosθ)`。"""
    angle = math.radians(degrees)
    return ((math.cos(angle), 0.0, math.sin(angle)),
            (-math.sin(angle), 0.0, math.cos(angle)))


# --------------------------------------------------------------------------- 连续剖面

def profile_rise(grade_deg):
    """坡度（deg）→ 每米水平前进的抬升（tan）。剖面总是「先上后下」，只用绝对值。"""
    if not math.isfinite(grade_deg) or abs(grade_deg) > 45.0:
        raise ValueError(f"坡度必须是 [-45, 45] 内的有限值，收到 {grade_deg!r}")
    return math.tan(math.radians(abs(grade_deg)))


def profile_height(grade_deg, x):
    """lane 局部坐标 `x` 处的坡面高度（相对 lane 原点；原点在平地上 ⇒ h(0)=0）。"""
    if not math.isfinite(x):
        raise ValueError(f"x 必须是有限值，收到 {x!r}")
    rise = profile_rise(grade_deg)
    if x <= UP_START_M:
        return 0.0
    if x <= CREST_START_M:
        return rise * (x - UP_START_M)
    if x <= DOWN_START_M:
        return rise * UP_M
    if x <= FLAT_OUT_START_M:
        return rise * (FLAT_OUT_START_M - x)
    return 0.0


def profile_slope_degrees(grade_deg, x):
    """`x` 处的**局部**坡度（带符号：上坡为正、下坡为负、平地 0）。"""
    if not math.isfinite(x):
        raise ValueError(f"x 必须是有限值，收到 {x!r}")
    profile_rise(grade_deg)              # 只做范围校验
    grade = abs(grade_deg)
    if x <= UP_START_M or x > FLAT_OUT_START_M:
        return 0.0
    if x <= CREST_START_M:
        return grade
    if x <= DOWN_START_M:
        return 0.0
    return -grade


def profile_frame(grade_deg, x):
    """`x` 处的坡面坐标系 (切向, 法向)（用局部坡度）。"""
    return slope_frame(profile_slope_degrees(grade_deg, x))


def profile_arc_length(grade_deg, x):
    """从 lane 原点沿**坡面**走到 `x` 的行程（平地段 = x，坡段要除 cosθ）。"""
    if not math.isfinite(x):
        raise ValueError(f"x 必须是有限值，收到 {x!r}")
    profile_rise(grade_deg)              # 只做范围校验
    grade = abs(grade_deg)
    cosine = math.cos(math.radians(grade))
    if x <= UP_START_M:
        return x
    if x <= CREST_START_M:
        return UP_START_M + (x - UP_START_M) / cosine
    if x <= DOWN_START_M:
        return UP_START_M + UP_M / cosine + (x - CREST_START_M)
    if x <= FLAT_OUT_START_M:
        return UP_START_M + UP_M / cosine + CREST_M + (x - DOWN_START_M) / cosine
    return (UP_START_M + (UP_M + DOWN_M) / cosine + CREST_M
            + (x - FLAT_OUT_START_M))


def profile_polyline(grade_deg):
    """剖面在 x–z 平面的闭合多边形（上表面 + 底面），供 mesh 与离线检查共用。

    上表面从 `-BACK_M` 到 `FORWARD_M`（两侧都是平地，z=0），底面比它低 `THICKNESS_M`。
    """
    # 只保留真正的折点：平地 lane（rise = 0）的中间三个点与两端共线，留它们会造出零面积
    # 三角形（PhysX 吃进去会报退化三角形），所以平地 lane 退化成简单矩形。
    xs = [-BACK_M]
    if profile_rise(grade_deg) != 0.0:
        xs += [UP_START_M, CREST_START_M, DOWN_START_M, FLAT_OUT_START_M]
    xs.append(FORWARD_M)
    top = [(x, profile_height(grade_deg, x)) for x in xs]
    # 底面必须**平**（z = −THICKNESS）：坡面是实心板上的凸起，不是等厚的弯曲薄壳。
    # 第一版把底面写成「上表面下移 THICKNESS」，结果整块板跟着坡走、凸起少了一半以上体积
    # （10° 档体积 42 m³ 而不是 63 m³，被 `_signed_volume` 与鞋带面积同时抓出来）。
    # 底面只留两个端点：中间那些点与两端共线，留着会在端面扇形三角化时造出零面积三角形。
    bottom = [(FORWARD_M, -THICKNESS_M), (-BACK_M, -THICKNESS_M)]
    return top + bottom


def _extrude(polygon, half_width):
    """把 x–z 平面的闭合多边形沿 y 挤出成闭合实体，返回 (vertices, faces)。

    两个端面用扇形三角化（本剖面多边形是凸的），侧壁每条边一个四边形。绕向这一步不保证，
    由调用方按有符号体积统一翻转。
    """
    count = len(polygon)
    vertices = [(x, -half_width, z) for x, z in polygon]
    vertices += [(x, half_width, z) for x, z in polygon]
    faces = []
    # 端面：从**最后一个顶点**（底面左下角）扇形三角化。不能从顶点 0（上表面最左点）扇：
    # 平地段两端与坡顶的 z 都是 0 或平台，会和顶点 0 连成共线三点 ⇒ 零面积三角形
    # （实测 5°/10° 档各 2 个；PhysX 吃退化三角形会报错）。底面角点天然离开 z=0 平面。
    # 端面法向必须朝 ±y 外侧：剖面多边形在 x–z 里是顺时针（上表面向右、底面折返），
    # 所以 y = −half 的端面要**反序**才是 −y 外法向，y = +half 的端面用原序得 +y。
    apex = count - 1
    for index in range(count - 2):
        faces.append((apex, index + 1, index))                           # y = −half 端面
        faces.append((count + apex, count + index, count + index + 1))   # y = +half 端面
    for index in range(count):
        nxt = (index + 1) % count          # 闭合边（index = count-1）必须回到 0，不能写 index+1
        faces.append((index, count + nxt, count + index))
        faces.append((index, nxt, count + nxt))
    return vertices, faces


def _signed_volume(vertices, faces):
    """闭合网格的有符号体积（> 0 表示面绕向朝外）。"""
    total = 0.0
    for a, b, c in faces:
        ax, ay, az = vertices[a]
        bx, by, bz = vertices[b]
        cx, cy, cz = vertices[c]
        total += (ax * (by * cz - bz * cy)
                  - ay * (bx * cz - bz * cx)
                  + az * (bx * cy - by * cx))
    return total / 6.0


def tile_mesh(row, column):
    """一块闭合的「小山」板：上表面 = `profile_height(该列坡度, x)`，底面低 THICKNESS。"""
    origin = tile_origin(row, column)
    grade = slope_degrees(column, row)
    local_vertices, faces = _extrude(profile_polyline(grade), HALF_WIDTH_M)
    if _signed_volume(local_vertices, faces) < 0.0:
        faces = [(a, c, b) for a, b, c in faces]
    vertices = [(origin[0] + x, origin[1] + y, origin[2] + z) for x, y, z in local_vertices]
    return vertices, faces, origin


def grid_mesh():
    """Return vertices/faces and origins[row][column]; no simulator imports."""
    vertices, faces, origins = [], [], []
    for row in range(ROWS):
        row_origins = []
        for column in range(COLUMNS):
            tile_vertices, tile_faces, origin = tile_mesh(row, column)
            offset = len(vertices)
            vertices.extend(tile_vertices)
            faces.extend(tuple(offset+index for index in face) for face in tile_faces)
            row_origins.append(origin)
        origins.append(row_origins)
    return vertices, faces, origins


def attachment_root_positions(degrees, distance, robot_height=0.35, cart_height=0.18):
    """平地上（或恒定坡度上）的名义出生位姿，用于检查离面净空与挂点距。"""
    tangent, normal = slope_frame(degrees)
    difference = robot_height-cart_height
    if distance <= abs(difference):
        raise ValueError("attachment distance must exceed normal height difference")
    along = math.sqrt(distance*distance-difference*difference)
    robot_root = tuple(robot_height*n for n in normal)
    robot_point = tuple(r-0.16*t for r, t in zip(robot_root, tangent))
    cart_point = tuple(p-along*t-difference*n for p, t, n in zip(robot_point, tangent, normal))
    cart_root = tuple(p-0.25*t for p, t in zip(cart_point, tangent))
    return robot_root, cart_root, robot_point, cart_point
