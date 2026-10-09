"""Standard-library geometry shared by the towing mesh and offline checks."""

from __future__ import annotations

import math

if __package__:
    from .connection_grid import COLUMNS, ROWS, slope_degrees
else:
    from connection_grid import COLUMNS, ROWS, slope_degrees

BACK_M = 3.0
FORWARD_M = 17.0
HALF_WIDTH_M = 3.0
ROW_SPACING_M = BACK_M + FORWARD_M + 2.0
COLUMN_SPACING_M = 2.0 * HALF_WIDTH_M + 2.0
THICKNESS_M = 0.35
BOUNDARY_MARGIN_M = 0.6


def tile_origin(row, column):
    """True 20-by-40 mesh origins, independent of GridCloner's square layout."""
    if not 0 <= row < ROWS or not 0 <= column < COLUMNS:
        raise ValueError("invalid terrain cell")
    return ((row - (ROWS - 1) / 2) * ROW_SPACING_M,
            (column - (COLUMNS - 1) / 2) * COLUMN_SPACING_M, 0.0)


def slope_frame(degrees):
    angle = math.radians(degrees)
    return ((math.cos(angle), 0.0, math.sin(angle)),
            (-math.sin(angle), 0.0, math.cos(angle)))


def tile_mesh(row, column):
    """Closed straight-ramp slab; top is z=tan(theta)*(x-origin_x)."""
    origin = tile_origin(row, column)
    slope = math.tan(math.radians(slope_degrees(column, row)))
    corners = ((-BACK_M, -HALF_WIDTH_M), (FORWARD_M, -HALF_WIDTH_M),
               (FORWARD_M, HALF_WIDTH_M), (-BACK_M, HALF_WIDTH_M))
    top = [(origin[0]+x, origin[1]+y, origin[2]+slope*x) for x, y in corners]
    vertices = top + [(x, y, z-THICKNESS_M) for x, y, z in top]
    faces = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
             (0, 5, 1), (0, 4, 5), (1, 6, 2), (1, 5, 6),
             (2, 7, 3), (2, 6, 7), (3, 4, 0), (3, 7, 4)]
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
    """Nominal yaw-zero reset used to check plane clearance and attachment distance."""
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
