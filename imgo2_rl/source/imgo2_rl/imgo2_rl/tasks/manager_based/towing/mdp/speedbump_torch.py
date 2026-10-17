"""Vectorised trapezoid speed-bump height, sharing all shape constants with geometry.

与 `speedbump_geometry.profile_height` **同一分段/同一公式**（一条底 0.3 m、顶 0.1 m、
高 0.05 m 的等腰梯形，横向中央满高/外缘 cos² 收零，与列的量级档无关）；标量版保持纯标准库，
torch 版单独放这里，两边逐点等价由 `tests/test_towing_speedbump_geometry.py` 交叉核对。

奖励/终止里每个控制步都要对 N 个环境算离面高度（`robot_fall`），逐 env 调用标量函数太慢，
所以这里必须向量化；分段写成嵌套 `torch.where` 时容易漏边界，这正是要交叉核对的原因。
"""

from __future__ import annotations

import math

import torch

if __package__:
    from .speedbump_geometry import (
        BUMP_END_M, BUMP_HEIGHT_M, BUMP_START_M, BUMP_TOP_END_M, BUMP_TOP_START_M,
        LATERAL_FLAT_HALF_WIDTH_M, LATERAL_TAPER_M, RAMP_RUN_M, layout)
else:  # 按文件路径直接加载（离线测试的既有做法）时没有包上下文
    from speedbump_geometry import (  # type: ignore[no-redef]
        BUMP_END_M, BUMP_HEIGHT_M, BUMP_START_M, BUMP_TOP_END_M, BUMP_TOP_START_M,
        LATERAL_FLAT_HALF_WIDTH_M, LATERAL_TAPER_M, RAMP_RUN_M, layout)


def lateral_factor_tensor(y: torch.Tensor) -> torch.Tensor:
    """`lateral_factor` 的向量版：中央 `|y| ≤ 2.0` 为 1、外侧 1 m 内 `cos²`、`|y| ≥ 3.0` 为 0。"""
    distance = y.abs()
    phase = math.pi * (distance - LATERAL_FLAT_HALF_WIDTH_M) / (2.0 * LATERAL_TAPER_M)
    taper = torch.cos(phase).square()
    return torch.where(
        distance <= LATERAL_FLAT_HALF_WIDTH_M, torch.ones_like(distance),
        torch.where(distance >= layout.HALF_WIDTH_M, torch.zeros_like(distance), taper))


def profile_height_tensor(grade_deg: torch.Tensor, x: torch.Tensor,
                          y: torch.Tensor | None = None) -> torch.Tensor:
    """`x`（lane 局部坐标）处的减速带高度（形状 `(N,)`）。

    `grade_deg` 不影响结果（几何固定，用户口径），只为与 hill/rolling 的调用签名一致；
    `y` 省略时按中线（横向因子 1）算 —— 与标量版 `profile_height(grade, x)` 的默认一致。
    """
    height = torch.where(
        x <= BUMP_TOP_START_M,
        BUMP_HEIGHT_M * (x - BUMP_START_M) / RAMP_RUN_M,
        torch.where(x <= BUMP_TOP_END_M,
                    torch.full_like(x, BUMP_HEIGHT_M),
                    BUMP_HEIGHT_M * (BUMP_END_M - x) / RAMP_RUN_M))
    result = torch.where((x > BUMP_START_M) & (x < BUMP_END_M), height, torch.zeros_like(x))
    if y is not None:
        result = result * lateral_factor_tensor(y)
    return result
