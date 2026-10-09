"""Torch-vectorised version of the continuous towing profile height.

与 `slope_geometry.profile_height` **同一分段**（平地 3 m → 上坡 4 m → 坡顶 1 m → 下坡 4 m →
平地 3 m）；`slope_geometry` 保持纯标准库（离线机器也能 import），torch 版本单独放这里，
两边的等价性由 `tests/test_towing_slope_geometry.py` 逐点交叉核对（有 torch 时）。

奖励/终止里每个控制步都要对 N 个环境算离面高度（`robot_fall`），逐 env 调用标量函数太慢，
所以这里必须向量化；把分段写成 `torch.where` 的嵌套时容易漏边界，这正是要交叉核对的原因。
"""

from __future__ import annotations

import torch

if __package__:
    from .slope_geometry import (
        CREST_START_M, DOWN_START_M, FLAT_IN_M, FLAT_OUT_START_M, UP_M)
else:  # 按文件路径直接加载（离线测试的既有做法）时没有包上下文
    from slope_geometry import (  # type: ignore[no-redef]
        CREST_START_M, DOWN_START_M, FLAT_IN_M, FLAT_OUT_START_M, UP_M)


def profile_height_tensor(grade_deg: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """`x`（lane 局部坐标）处的坡面高度；`grade_deg` 与 `x` 逐环境对应（形状 `(N,)`）。

    剖面总是「先上后下」，所以只用坡度量级的绝对值。`x` 在 [0, 3] 与 > 12 时高度 0。
    """
    rise = torch.tan(torch.deg2rad(grade_deg.abs()))
    zeros = torch.zeros_like(x)
    return torch.where(
        x <= FLAT_IN_M, zeros,
        torch.where(x <= CREST_START_M, rise * (x - FLAT_IN_M),
                    torch.where(x <= DOWN_START_M, rise * UP_M,
                                torch.where(x <= FLAT_OUT_START_M,
                                            rise * (FLAT_OUT_START_M - x), zeros))))
