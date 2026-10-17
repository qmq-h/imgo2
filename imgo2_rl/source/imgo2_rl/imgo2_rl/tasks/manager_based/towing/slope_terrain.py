"""Isaac Lab adapter for the deterministic mixed towing ground (slopes + speed bumps)."""

import numpy as np
import torch
import trimesh

from isaaclab.terrains import TerrainImporter

from .mdp.connection_grid import COLUMNS, ROWS, GRID_SIZE


class TowingMixedTerrainGenerator:
    """One ground mesh: flat + slope columns + the speed-bump block (2026-10-16).

    `smooth` 选择前 40 列（平地/平坡）用哪套高度场；列 40–49 恒为减速带（见
    `mdp/mixed_geometry.py`）。逻辑原点仍由 `env_spec` 的行优先映射给出。
    """

    #: 平滑段高度场：`"hill"`（单坡包，默认）或 `"rolling"`（连续起伏）。
    smooth = "hill"

    def __init__(self, cfg, device="cpu"):
        if (cfg.num_rows, cfg.num_cols) != (ROWS, COLUMNS):
            raise ValueError(f"towing terrain requires {ROWS} rows and {COLUMNS} columns")
        vertices, faces, origins = self.generate_mesh()
        self.terrain_mesh = trimesh.Trimesh(vertices=np.asarray(vertices),
                                          faces=np.asarray(faces), process=False)
        self.terrain_origins = np.asarray(origins, dtype=np.float32)
        self.flat_patches = {}

    @classmethod
    def generate_mesh(cls):
        from .mdp.mixed_geometry import grid_mesh
        return grid_mesh(cls.smooth)


class TowingMixedRollingTerrainGenerator(TowingMixedTerrainGenerator):
    """Same mixed ground with the smooth columns built from the rolling height field."""

    smooth = "rolling"


class TowingSlopeTerrainImporter(TerrainImporter):
    """Use env_spec's row-major mapping without random terrain levels."""

    def configure_env_origins(self, terrain_origins=None):
        if terrain_origins is None:
            raise ValueError("towing slopes require generated terrain origins")
        self.terrain_origins = torch.as_tensor(terrain_origins, device=self.device, dtype=torch.float32)
        if tuple(self.terrain_origins.shape) != (ROWS, COLUMNS, 3):
            raise ValueError("unexpected towing terrain origin shape")
        indices = torch.arange(self.cfg.num_envs, device=self.device) % GRID_SIZE
        self.terrain_levels = indices // COLUMNS
        self.terrain_types = indices % COLUMNS
        self.max_terrain_level = ROWS
        self.env_origins = self.terrain_origins[self.terrain_levels, self.terrain_types].clone()

    def update_env_origins(self, env_ids, move_up, move_down):
        raise RuntimeError("towing length/slope grid is fixed; curriculum is disabled")
