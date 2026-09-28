"""地形分组助手（纯 torch / numpy，不依赖 Isaac Lab）——2026-09-28 v5 方案的"平地掩码"。

**用途**：v5 只在 `flat` 上对专家 0（AMP 先验）做**锚定**。锚是**逐样本**的损失项，所以训练时必须
知道"这一帧在不在 flat 列上"。本模块把这件事做成纯函数：

* `terrain_columns(keys, proportions, num_cols, names)`：按 **Isaac Lab 的列分配规则**（见
  `rl_lab.utils.gait_dump.expand_terrain_names` 的注释与 `terrain_generator.py:233-241`）算出
  指定地形名占用的**列索引集合**；
* `anchor_weights(terrain_types, ...)`：把逐环境的 `terrain_types` 映射成 `[N]` 的 float 权重
  （在目标地形上 = `scale`，其余 = 0）。

**为什么要独立成模块**：训练侧（`cmoe_on_policy_runner.py`）拿不到 reward 管理器里的地形掩码
（那些是 `Masked*` 奖励项内部的缓存），而它每个控制步都需要一个 `[N]` 权重喂给 storage。
放在这里既能被 runner 用，也能在没有 Isaac 的机器上单测（`tests/test_terrain_masks.py`）。
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch

from .gait_dump import expand_terrain_names


def terrain_columns(
    keys: Sequence[str],
    proportions: Iterable[float],
    num_cols: int,
    names: Iterable[str],
) -> list[int]:
    """返回 `names` 里每个地形名占用的**列索引**（按比例分配，可能连续多列）。

    名字不在 `keys` 里 ⇒ 抛 `ValueError`（打错字会让锚静默失效，必须显式报错）。
    """
    expanded = expand_terrain_names(keys, proportions, num_cols)
    wanted = {str(name) for name in names}
    missing = wanted - set(str(key) for key in keys)
    if missing:
        raise ValueError(f"这些地形名不在 sub_terrains 里：{sorted(missing)}（可用：{list(keys)}）")
    return [index for index, name in enumerate(expanded) if name in wanted]


def anchor_weights(
    terrain_types: torch.Tensor,
    *,
    keys: Sequence[str],
    proportions: Iterable[float],
    num_cols: int,
    names: Iterable[str] = ("flat",),
    scale: float = 1.0,
    columns: Sequence[int] | None = None,
) -> torch.Tensor:
    """逐环境的锚定权重：`terrain_types` 落在目标地形列上 ⇒ `scale`，否则 0。

    `columns` 可以预先算好并复用（每个控制步都算一遍列分配是浪费）。返回 `[N]` float32，
    **形状与 `terrain_types` 一致（都是一维、长度 = 环境数）**。
    """
    if terrain_types.ndim != 1:
        raise ValueError(f"terrain_types 应为一维 [N]，收到 {tuple(terrain_types.shape)}")
    if columns is None:
        columns = terrain_columns(keys, proportions, num_cols, names)
    if not columns:
        return torch.zeros_like(terrain_types, dtype=torch.float32)
    column_tensor = torch.as_tensor(sorted(columns), device=terrain_types.device, dtype=terrain_types.dtype)
    hit = (terrain_types.unsqueeze(-1) == column_tensor.unsqueeze(0)).any(dim=-1)
    return hit.to(torch.float32) * float(scale)
