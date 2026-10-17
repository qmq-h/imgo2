"""Vectorised three-period sine wave height, sharing all shape constants with geometry.

与 `wave_geometry.profile_height` **同一公式**（3 周期等幅正弦 × 两端 `cos²` 包络 × 横向
`cos²` 收零；几何与档位无关）；标量版保持纯标准库，torch 版单独放这里，两边逐点等价由
`tests/test_towing_wave_geometry.py` 交叉核对（有 torch 时）。

奖励/终止里每个控制步都要对 N 个环境算离面高度（`robot_fall`），逐 env 调用标量函数太慢，
所以这里必须向量化；包络的分段写成嵌套 `torch.where` 时容易漏边界，这正是要交叉核对的原因。
"""

from __future__ import annotations

import math

import torch

if __package__:
    from .wave_geometry import (
        END_M, LATERAL_FLAT_HALF_WIDTH_M, LATERAL_TAPER_M, PEAK_M, SPAN_M, START_M,
        TAPER_M, WAVE_NUMBER, layout)
else:  # 按文件路径直接加载（离线测试的既有做法）时没有包上下文
    from wave_geometry import (  # type: ignore[no-redef]
        END_M, LATERAL_FLAT_HALF_WIDTH_M, LATERAL_TAPER_M, PEAK_M, SPAN_M, START_M,
        TAPER_M, WAVE_NUMBER, layout)


def lateral_factor_tensor(y: torch.Tensor) -> torch.Tensor:
    """`wave_geometry.lateral_factor` 的向量版（中央 ±2 m 满幅、外缘 1 m `cos²` 收零）。"""
    distance = y.abs()
    phase = math.pi * (distance - LATERAL_FLAT_HALF_WIDTH_M) / (2.0 * LATERAL_TAPER_M)
    taper = torch.cos(phase).square()
    return torch.where(
        distance <= LATERAL_FLAT_HALF_WIDTH_M, torch.ones_like(distance),
        torch.where(distance >= layout.HALF_WIDTH_M, torch.zeros_like(distance), taper))


def envelope_tensor(u: torch.Tensor) -> torch.Tensor:
    """`wave_geometry._envelope` 的向量版：两端 `TAPER_M` 内 `sin²` 升/降，其余为 1。"""
    rising = torch.sin(0.5 * math.pi * u / TAPER_M).square()
    falling = torch.sin(0.5 * math.pi * (SPAN_M - u) / TAPER_M).square()
    ones = torch.ones_like(u)
    return torch.where(
        (u <= 0.0) | (u >= SPAN_M), torch.zeros_like(u),
        torch.where(u < TAPER_M, rising,
                    torch.where(u > SPAN_M - TAPER_M, falling, ones)))


def profile_height_tensor(grade_deg: torch.Tensor, x: torch.Tensor,
                          y: torch.Tensor | None = None) -> torch.Tensor:
    """`x`（lane 局部坐标）处的波面高度（形状 `(N,)`）。

    `grade_deg` 不影响结果（几何固定，用户口径），只为与 hill/rolling 的调用签名一致；
    `y` 省略时按中线（横向因子 1）算 —— 与标量版 `profile_height(grade, x)` 的默认一致。
    """
    u = x - START_M
    wave = PEAK_M * torch.sin(WAVE_NUMBER * u) * envelope_tensor(u)
    result = torch.where((u > 0.0) & (u < SPAN_M), wave, torch.zeros_like(x))
    if y is not None:
        result = result * lateral_factor_tensor(y)
    return result
