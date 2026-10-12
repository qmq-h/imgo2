"""Vectorised rolling surface height, sharing all shape constants with geometry."""
import math
import torch

if __package__:
    from .rolling_geometry import START_M, HILL_WIDTHS_M, layout
else:
    from rolling_geometry import START_M, HILL_WIDTHS_M, layout


def profile_height_tensor(grade_deg, x, y=None):
    rise = torch.tan(torch.deg2rad(grade_deg.abs()))
    result = torch.zeros_like(x)
    start = START_M
    for width in HILL_WIDTHS_M:
        phase = 2 * math.pi * (x - start) / width
        height = rise * width / (2 * math.pi) * (1 - torch.cos(phase))
        result = result + torch.where((x > start) & (x < start + width), height, 0.0)
        start += width
    if y is not None:
        lateral = torch.cos(math.pi * y / (2 * layout.HALF_WIDTH_M)).square()
        result = result * torch.where(y.abs() < layout.HALF_WIDTH_M, lateral, 0.0)
    return result
