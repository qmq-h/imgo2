"""CMoE rough-terrain task built on the existing Imgo2 PPO rough task."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import imgo2_rl.tasks.manager_based.locomotion.velocity.mdp as mdp
from imgo2_rl.tasks.manager_based.locomotion.velocity.velocity_env_cfg import MySceneCfg, ObservationsCfg, RewardsCfg

from .rough_env_cfg import Imgo2RoughEnvCfg
from .cmoe_terrains import CMoETrackGapTerrainCfg, CMoETrackStairsTerrainCfg, CMoETrackStepTerrainCfg


FOOT_EDGE_SENSOR_NAMES = (
    "foot_edge_scanner_fl",
    "foot_edge_scanner_fr",
    "foot_edge_scanner_rl",
    "foot_edge_scanner_rr",
)


def _foot_edge_scanner(foot_name: str) -> RayCasterCfg:
    """Create a compact downward ray grid centered on one foot."""
    return RayCasterCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{foot_name}",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.04, size=(0.12, 0.12)),
        max_distance=2.0,
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )


@configclass
class CMoESceneCfg(MySceneCfg):
    """PPO rough scene plus one local void-edge scanner for each foot."""

    foot_edge_scanner_fl = _foot_edge_scanner("FL_FOOT")
    foot_edge_scanner_fr = _foot_edge_scanner("FR_FOOT")
    foot_edge_scanner_rl = _foot_edge_scanner("RL_FOOT")
    foot_edge_scanner_rr = _foot_edge_scanner("RR_FOOT")


@configclass
class CMoERewardsCfg(RewardsCfg):
    """CMoE-specific addition for penalizing support at a gap edge."""

    feet_edge = RewTerm(
        func=mdp.feet_edge,
        weight=-1.0,
        params={
            "edge_sensor_names": FOOT_EDGE_SENSOR_NAMES,
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_FOOT"),
            "contact_force_threshold": 1.0,
        },
    )


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

    scene: CMoESceneCfg = CMoESceneCfg(num_envs=4096, env_spacing=2.5)
    observations: CMoEObservationsCfg = CMoEObservationsCfg()
    rewards: CMoERewardsCfg = CMoERewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        if self.observations.policy.base_lin_vel is not None:
            raise RuntimeError("CMoE policy must not receive privileged base linear velocity")
        if self.observations.policy.height_scan is not None:
            raise RuntimeError("CMoE height scan must only be exposed through the terrain group")

        # Reproduce the original +x obstacle-course layout at quadruped scale.
        self.scene.terrain.terrain_generator.size = (8.0, 4.0)
        sub_terrains = self.scene.terrain.terrain_generator.sub_terrains
        sub_terrains["pyramid_stairs"] = CMoETrackStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.025, 0.08),
            ascending=True,
        )
        sub_terrains["pyramid_stairs_inv"] = CMoETrackStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.025, 0.08),
            ascending=False,
        )
        sub_terrains["boxes"] = CMoETrackStepTerrainCfg(
            proportion=0.15,
            step_height_range=(0.04, 0.12),
        )
        sub_terrains["random_rough"].proportion = 0.20
        sub_terrains["hf_pyramid_slope"].proportion = 0.10
        sub_terrains["hf_pyramid_slope_inv"].proportion = 0.10
        sub_terrains["gap"] = CMoETrackGapTerrainCfg(
            proportion=0.20,
            gap_width_range=(0.08, 0.16),
            platform_length_range=(0.65, 0.95),
            first_gap_x=1.8,
            num_gaps=4,
        )

        # Match the original CMoE command split: easy terrain keeps small
        # omnidirectional commands, while obstacle terrain moves along world +x.
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-1.6, 1.6)
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.rel_heading_envs = 1.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.heading_control_stiffness = 0.5
        self.commands.base_velocity.forward_only_terrain_names = (
            "pyramid_stairs",
            "pyramid_stairs_inv",
            "boxes",
            "gap",
        )
        self.commands.base_velocity.forward_speed_range = (0.3, 1.0)
        self.commands.base_velocity.forward_heading_target = 0.0

        # Scale the original reset translation for the smaller course and keep its fixed orientation.
        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

        # Retain task/proprioceptive rewards, remove fixed-gait shaping, and strengthen safety.
        self.rewards.feet_stumble.weight = -1.0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.undesired_contacts.weight = -5.0
        self.rewards.feet_height.weight = 0.0
        self.rewards.feet_height_body.weight = 0.0
        self.rewards.feet_slide.weight = 0.0
        self.rewards.feet_air_time_variance.weight = 0.0
        self.rewards.joint_mirror.weight = 0.0

        # Falling on the base ends the episode; foot contacts remain legal.
        self.terminations.illegal_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.base_link_name]),
                "threshold": 1.0,
            },
        )

        edge_scan_period = self.decimation * self.sim.dt
        for sensor_name in FOOT_EDGE_SENSOR_NAMES:
            getattr(self.scene, sensor_name).update_period = edge_scan_period
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
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.rel_heading_envs = 1.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
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
