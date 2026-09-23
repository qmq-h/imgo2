"""CMoE rough-terrain task built on the existing Imgo2 PPO rough task."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import imgo2_rl.tasks.manager_based.locomotion.velocity.mdp as mdp
from imgo2_rl.tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg

from .rough_env_cfg import Imgo2RoughEnvCfg


@configclass
class CMoEObservationsCfg(ObservationsCfg):
    """Keep PPO's 45-D policy/235-D critic and expose height scan separately."""

    @configclass
    class TerrainCfg(ObsGroup):
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    terrain: TerrainCfg = TerrainCfg()


@configclass
class Imgo2CMoERoughEnvCfg(Imgo2RoughEnvCfg):
    """PPO rough task with policy/terrain/critic groups required by CMoE."""

    observations: CMoEObservationsCfg = CMoEObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        if self.observations.policy.base_lin_vel is not None:
            raise RuntimeError("CMoE policy must not receive privileged base linear velocity")
        if self.observations.policy.height_scan is not None:
            raise RuntimeError("CMoE height scan must only be exposed through the terrain group")
        # Imgo2RoughEnvCfg only performs this cleanup for its exact class name.
        self.disable_zero_weight_rewards()


@configclass
class Imgo2CMoERoughPlayEnvCfg(Imgo2CMoERoughEnvCfg):
    """Deterministic single-environment configuration used by CMoE play."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.terrain.terrain_generator.num_cols = 10
        self.scene.terrain.max_init_terrain_level = 5
        self.observations.policy.enable_corruption = False
        self.observations.terrain.enable_corruption = False
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.events.randomize_reset_base.params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                           "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                               "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
        }
        self.events.randomize_rigid_body_material = None
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_rigid_body_mass_others = None
        self.events.randomize_com_positions = None
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_reset_joints = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_push_robot = None
