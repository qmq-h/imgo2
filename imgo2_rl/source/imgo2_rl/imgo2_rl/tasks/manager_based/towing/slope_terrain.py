"""Isaac Lab adapter for the deterministic straight-ramp towing grid."""

import numpy as np
import torch
import trimesh

from isaaclab.terrains import TerrainImporter

from .mdp.connection_grid import COLUMNS, ROWS, GRID_SIZE
from .mdp.slope_geometry import grid_mesh


class TowingSlopeTerrainGenerator:
    """Generate actual ramp geometry and matching logical cell origins."""

    def __init__(self, cfg, device="cpu"):
        if (cfg.num_rows, cfg.num_cols) != (ROWS, COLUMNS):
            raise ValueError(f"towing terrain requires {ROWS} rows and {COLUMNS} columns")
        vertices, faces, origins = grid_mesh()
        self.terrain_mesh = trimesh.Trimesh(vertices=np.asarray(vertices),
                                          faces=np.asarray(faces), process=False)
        self.terrain_origins = np.asarray(origins, dtype=np.float32)
        self.flat_patches = {}


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
