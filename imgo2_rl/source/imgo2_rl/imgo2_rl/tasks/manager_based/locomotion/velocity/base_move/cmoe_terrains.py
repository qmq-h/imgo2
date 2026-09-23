"""Longitudinal CMoE obstacle courses scaled for the Imgo2 quadruped."""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass


_BASE_DEPTH = 1.0


def _platform(x0: float, x1: float, width: float, top_height: float = 0.0) -> trimesh.Trimesh:
    """Create one full-width platform segment whose top is at ``top_height``."""
    length = x1 - x0
    height = _BASE_DEPTH + top_height
    center = (0.5 * (x0 + x1), 0.5 * width, 0.5 * (top_height - _BASE_DEPTH))
    return trimesh.creation.box(
        (length, width, height), trimesh.transformations.translation_matrix(center)
    )


def track_gap_terrain(difficulty: float, cfg: CMoETrackGapTerrainCfg):
    """Generate the original CMoE-style +x course with several transverse gaps."""
    gap_width = cfg.gap_width_range[0] + difficulty * (
        cfg.gap_width_range[1] - cfg.gap_width_range[0]
    )
    spacing = cfg.platform_length_range[1] - difficulty * (
        cfg.platform_length_range[1] - cfg.platform_length_range[0]
    )

    meshes: list[trimesh.Trimesh] = []
    cursor = 0.0
    platform_end = cfg.first_gap_x
    for _ in range(cfg.num_gaps):
        meshes.append(_platform(cursor, platform_end, cfg.size[1]))
        cursor = platform_end + gap_width
        platform_end = cursor + spacing
    if cursor < cfg.size[0]:
        meshes.append(_platform(cursor, cfg.size[0], cfg.size[1]))

    origin = np.array([cfg.spawn_x, 0.5 * cfg.size[1], 0.0])
    return meshes, origin


@configclass
class CMoETrackGapTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a longitudinal course containing repeated gaps."""

    function = track_gap_terrain
    gap_width_range: tuple[float, float] = (0.08, 0.16)
    platform_length_range: tuple[float, float] = (0.65, 0.95)
    first_gap_x: float = 1.8
    num_gaps: int = 4
    spawn_x: float = 0.75


def track_step_terrain(difficulty: float, cfg: CMoETrackStepTerrainCfg):
    """Generate separated full-width steps along +x on an otherwise flat track."""
    step_height = cfg.step_height_range[0] + difficulty * (
        cfg.step_height_range[1] - cfg.step_height_range[0]
    )
    step_length = cfg.step_length_range[0] + difficulty * (
        cfg.step_length_range[1] - cfg.step_length_range[0]
    )

    meshes = [_platform(0.0, cfg.size[0], cfg.size[1])]
    x = cfg.first_step_x
    for _ in range(cfg.num_steps):
        if x + step_length >= cfg.size[0]:
            break
        meshes.append(_platform(x, x + step_length, cfg.size[1], step_height))
        x += step_length + cfg.step_spacing

    origin = np.array([cfg.spawn_x, 0.5 * cfg.size[1], 0.0])
    return meshes, origin


@configclass
class CMoETrackStepTerrainCfg(SubTerrainBaseCfg):
    """Configuration for separated step obstacles on a longitudinal track."""

    function = track_step_terrain
    step_height_range: tuple[float, float] = (0.04, 0.12)
    step_length_range: tuple[float, float] = (0.18, 0.30)
    step_spacing: float = 0.85
    first_step_x: float = 1.8
    num_steps: int = 4
    spawn_x: float = 0.75


def track_stairs_terrain(difficulty: float, cfg: CMoETrackStairsTerrainCfg):
    """Generate one compact multi-step staircase across a +x course."""
    step_height = cfg.step_height_range[0] + difficulty * (
        cfg.step_height_range[1] - cfg.step_height_range[0]
    )
    total_height = cfg.num_steps * step_height
    meshes: list[trimesh.Trimesh] = []

    if cfg.ascending:
        meshes.append(_platform(0.0, cfg.stairs_start_x, cfg.size[1]))
        for index in range(cfg.num_steps):
            x0 = cfg.stairs_start_x + index * cfg.step_depth
            x1 = x0 + cfg.step_depth
            meshes.append(_platform(x0, x1, cfg.size[1], (index + 1) * step_height))
        end_x = cfg.stairs_start_x + cfg.num_steps * cfg.step_depth
        meshes.append(_platform(end_x, cfg.size[0], cfg.size[1], total_height))
        spawn_height = 0.0
    else:
        meshes.append(_platform(0.0, cfg.stairs_start_x, cfg.size[1], total_height))
        for index in range(cfg.num_steps):
            x0 = cfg.stairs_start_x + index * cfg.step_depth
            x1 = x0 + cfg.step_depth
            meshes.append(_platform(x0, x1, cfg.size[1], (cfg.num_steps - index - 1) * step_height))
        end_x = cfg.stairs_start_x + cfg.num_steps * cfg.step_depth
        meshes.append(_platform(end_x, cfg.size[0], cfg.size[1]))
        spawn_height = total_height

    origin = np.array([cfg.spawn_x, 0.5 * cfg.size[1], spawn_height])
    return meshes, origin


@configclass
class CMoETrackStairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a single multi-level staircase obstacle."""

    function = track_stairs_terrain
    step_height_range: tuple[float, float] = (0.025, 0.08)
    step_depth: float = 0.30
    num_steps: int = 4
    stairs_start_x: float = 2.0
    spawn_x: float = 0.75
    ascending: bool = True
