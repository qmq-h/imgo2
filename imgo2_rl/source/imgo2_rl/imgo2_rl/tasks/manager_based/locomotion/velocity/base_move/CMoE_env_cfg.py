"""CMoE rough-terrain task built on the existing Imgo2 PPO rough task."""

import math

import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
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
from .cmoe_terrains import (
    CMoETrackCompositeTerrainCfg,
    CMoETrackGapTerrainCfg,
    CMoETrackHurdleTerrainCfg,
    CMoETrackMixTerrainCfg,
    CMoETrackNarrowStairsTerrainCfg,
    CMoETrackStairsTerrainCfg,
    CMoETrackStepTerrainCfg,
)

LEVEL_ORIENTATION_TERRAIN_NAMES = ("flat", "random_rough")

EASY_TERRAIN_NAMES = ("flat", "hf_pyramid_slope", "hf_pyramid_slope_inv", "random_rough")
OBSTACLE_TERRAIN_NAMES = (
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "boxes",
    "gap",
    "hurdle",
    "mix",
    "narrow_stairs",
)

FOOT_EDGE_SENSOR_NAMES = (
    "foot_edge_scanner_fl",
    "foot_edge_scanner_fr",
    "foot_edge_scanner_rl",
    "foot_edge_scanner_rr",
)

ENABLE_ILLEGAL_CONTACT_BODY_TERMINATION = False

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

        weight=0.0,
        params={
            "edge_sensor_names": FOOT_EDGE_SENSOR_NAMES,
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_FOOT"),
            "contact_force_threshold": 1.0,
        },
    )

    # Nonzero diagnostic weights prevent removal by the reward manager.
    gait_metric_trot = RewTerm(
        func=mdp.GaitReward,
        weight=1e-6,
        params={
            "std": 0.2,
            "command_name": "base_velocity",
            "max_err": 0.5,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
            "synced_feet_pair_names": (("FL_FOOT", "RR_FOOT"), ("FR_FOOT", "RL_FOOT")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )
    gait_metric_bound = RewTerm(
        func=mdp.GaitReward,
        weight=1e-6,
        params={
            "std": 0.2,
            "command_name": "base_velocity",
            "max_err": 0.5,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
            "synced_feet_pair_names": (("FL_FOOT", "FR_FOOT"), ("RL_FOOT", "RR_FOOT")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )

    diag_bounce = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=1e-6,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    diag_air_time = RewTerm(
        func=mdp.diag_air_time,
        weight=1e-6,
        params={"sensor_cfg": SceneEntityCfg("contact_forces")},
    )
    diag_pair_mismatch = RewTerm(
        func=mdp.diag_pair_mismatch,
        weight=1e-6,
        params={"sensor_cfg": SceneEntityCfg("contact_forces")},
    )

    diag_base_height = RewTerm(
        func=mdp.diag_base_height,
        weight=1e-6,
        params={
            "target_height": 0.30,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),
        },
    )
    gait_metric_pace = RewTerm(
        func=mdp.GaitReward,
        weight=1e-6,
        params={
            "std": 0.2,
            "command_name": "base_velocity",
            "max_err": 0.5,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
            "synced_feet_pair_names": (("FL_FOOT", "RL_FOOT"), ("FR_FOOT", "RR_FOOT")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )

    track_world_vel_xy_exp = RewTerm(
        func=mdp.track_world_vel_xy_exp,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    lin_pos_y = RewTerm(
        func=mdp.lin_pos_y,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "terrain_names": (),
        },
    )
    yaw_abs = RewTerm(
        func=mdp.yaw_abs,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "terrain_names": (),
        },
    )

    base_height_flat_l2 = RewTerm(
        func=mdp.MaskedBaseHeightL2Strict,
        weight=0.0,
        params={
            "target_height": 0.30,
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            "active_terrain_names": ("flat",),
        },
    )

@configclass
class CMoEObservationsCfg(ObservationsCfg):
    """Policy 45-D, terrain 77-D, critic 125-D; actor uses 10 policy frames plus terrain."""

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

        # 11 x 7 rays, separate from policy history.
        scan = self.scene.height_scanner.pattern_cfg
        scan.resolution = 0.1
        scan.size = (1.0, 0.6)

        self.scene.height_scanner.offset.pos = (0.25, 0.0, 20.0)

        # Match grid counting without truncating floating-point 0.6 / 0.1.
        _rows = math.floor(scan.size[0] / scan.resolution + 1.0e-9) + 1
        _cols = math.floor(scan.size[1] / scan.resolution + 1.0e-9) + 1
        if _rows * _cols != 77:
            raise RuntimeError(
                f"CMoE terrain scan must be 77 rays (11x7), got {_rows}x{_cols}={_rows * _cols} "
                f"(size={tuple(scan.size)}, resolution={scan.resolution})"
            )

        self.scene.terrain.terrain_generator.size = (8.0, 4.0)

        self.scene.terrain.terrain_generator.num_cols = 40
        self.scene.terrain.terrain_generator.num_rows = 20
        sub_terrains = self.scene.terrain.terrain_generator.sub_terrains

        sub_terrains["pyramid_stairs"] = CMoETrackStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.05, 0.20),
            num_steps=6,
            ascending=True,
        )

        sub_terrains["pyramid_stairs_inv"] = CMoETrackStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.05, 0.20),
            num_steps=6,
            ascending=False,
        )

        sub_terrains["boxes"] = CMoETrackStepTerrainCfg(
            first_step_x=1.6,
            num_steps=2,
            step_spacing=1.3,
            step_length_range=(0.3, 0.5),
            proportion=0.10,
            step_height_range=(0.08, 0.30),
        )

        sub_terrains["hf_pyramid_slope"].proportion = 0.05
        sub_terrains["hf_pyramid_slope_inv"].proportion = 0.05

        sub_terrains["random_rough"].proportion = 0.05

        sub_terrains["gap"] = CMoETrackGapTerrainCfg(
            proportion=0.30,
            gap_width_range=(0.12, 0.32),
            platform_length_range=(0.9, 1.4),
            first_gap_x=1.6,
            num_gaps=3,
        )

        sub_terrains["hurdle"] = CMoETrackHurdleTerrainCfg(proportion=0.10)
        sub_terrains["mix"] = CMoETrackMixTerrainCfg(proportion=0.10)
        sub_terrains["narrow_stairs"] = CMoETrackNarrowStairsTerrainCfg(proportion=0.10)

        sub_terrains["flat"] = terrain_gen.MeshPlaneTerrainCfg(proportion=0.10)

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
            "random_rough",
            "hf_pyramid_slope",
            "hf_pyramid_slope_inv",
            "gap",
            "hurdle",
            "mix",
            "narrow_stairs",
            "flat",
        )
        self.commands.base_velocity.forward_speed_range = (0.3, 1.0)
        self.commands.base_velocity.forward_heading_target = 0.0

        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

        self.rewards.feet_stumble.weight = 0.0

        self.rewards.undesired_contacts.weight = -0.5

        self.rewards.contact_forces.weight = -0.02

        self.rewards.track_ang_vel_z_exp.weight = 2.0

        self.rewards.track_world_vel_xy_exp.weight = 5.0
        self.rewards.track_lin_vel_xy_exp.weight = 0.0

        self.rewards.lin_pos_y.weight = -0.4
        self.rewards.yaw_abs.weight = -0.2

        self.rewards.flat_orientation_l2.func = mdp.MaskedFlatOrientationL2
        self.rewards.flat_orientation_l2.weight = -5.0

        _terrain_keys = tuple(self.scene.terrain.terrain_generator.sub_terrains.keys())
        self.rewards.flat_orientation_l2.params["free_terrain_names"] = tuple(
            name for name in _terrain_keys if name not in LEVEL_ORIENTATION_TERRAIN_NAMES
        )

        self.rewards.lin_vel_z_l2.func = mdp.MaskedLinVelZ
        self.rewards.lin_vel_z_l2.weight = -2.0

        self.rewards.lin_vel_z_l2.params["free_terrain_names"] = ("boxes", "gap")

        self.rewards.base_height_flat_l2.func = mdp.MaskedBaseHeightL2Strict
        self.rewards.base_height_flat_l2.weight = 0.0
        self.rewards.base_height_flat_l2.params["target_height"] = self.rewards.base_height_l2.params[
            "target_height"
        ]
        self.rewards.base_height_flat_l2.params["sensor_cfg"] = SceneEntityCfg("height_scanner_base")
        self.rewards.base_height_flat_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=[self.base_link_name]
        )

        self.rewards.base_height_flat_l2.params["active_terrain_names"] = ("flat",)

        self.rewards.joint_mirror.func = mdp.MaskedJointMirror
        self.rewards.joint_mirror.weight = -1.0

        self.rewards.joint_mirror.params["free_terrain_names"] = ("boxes",)
        self.rewards.joint_mirror.params["bound_terrain_names"] = ("gap",)

        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|shank).*", "RL_(hip|thigh|shank).*"],
            ["FL_(hip|thigh|shank).*", "RR_(hip|thigh|shank).*"],
        ]
        self.rewards.joint_mirror.params["bound_mirror_joints"] = [
            ["FL_(hip|thigh|shank).*", "FR_(hip|thigh|shank).*"],
            ["RL_(hip|thigh|shank).*", "RR_(hip|thigh|shank).*"],
        ]

        self.rewards.feet_air_time.func = mdp.MaskedFeetAirTime

        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["free_terrain_names"] = ("boxes", "gap")

        self.rewards.feet_height_body.func = mdp.MaskedFeetHeightBody
        self.rewards.feet_height_body.weight = -5.0
        self.rewards.feet_height_body.params["free_terrain_names"] = ("boxes", "gap")

        self.rewards.feet_air_time_variance.func = mdp.MaskedFeetAirTimeVariance
        self.rewards.feet_air_time_variance.weight = -8.0
        self.rewards.feet_air_time_variance.params["free_terrain_names"] = ("boxes", "gap")

        # Phase shaping is disabled; gait diagnostics remain active.
        self.rewards.feet_gait.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.ang_vel_xy_l2.weight = -0.05

        self.rewards.feet_height.weight = 0.0

        self.rewards.feet_slide.weight = -0.05
        self.rewards.feet_slide.params["sensor_cfg"].body_names = ".*_FOOT"
        self.rewards.feet_slide.params["asset_cfg"].body_names = ".*_FOOT"

        self.terminations.illegal_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.base_link_name]),
                "threshold": 1.0,
            },
        )

        if ENABLE_ILLEGAL_CONTACT_BODY_TERMINATION:
            self.terminations.illegal_contact_body = DoneTerm(
                func=mdp.illegal_contact,
                params={
                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[r"^(?!.*_FOOT).*"]),
                    "threshold": 50.0,
                },
            )

        self.curriculum.terrain_levels = CurrTerm(
            func=mdp.terrain_levels_vel_logged,
            params={
                "tracking_term_name": "track_world_vel_xy_exp",
                "tracking_move_up": 0.80,
                "tracking_move_down": 0.35,

                "relaxed_terrain_names": (
                    "pyramid_stairs",
                    "pyramid_stairs_inv",
                    "boxes",
                    "gap",
                    "hurdle",
                    "mix",
                    "narrow_stairs",
                ),

                "gait_metric_terms": (
                    ("trot", "gait_metric_trot"),
                    ("bound", "gait_metric_bound"),
                    ("pace", "gait_metric_pace"),
                    ("bounce", "diag_bounce"),
                    ("height", "base_height_l2"),

                    ("airtime", "diag_air_time"),
                    ("mismatch", "diag_pair_mismatch"),

                    ("base_height", "diag_base_height"),
                ),

                "tracking_move_up_relaxed": 0.40,
            },
        )

        edge_scan_period = self.decimation * self.sim.dt
        for sensor_name in FOOT_EDGE_SENSOR_NAMES:
            getattr(self.scene, sensor_name).update_period = edge_scan_period

        self.disable_zero_weight_rewards()

@configclass
class Imgo2CMoERoughPlayEnvCfg(Imgo2CMoERoughEnvCfg):
    """Deterministic single-environment configuration used by CMoE play."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1

        self.scene.terrain.terrain_generator.num_cols = 11
        for _sub in self.scene.terrain.terrain_generator.sub_terrains.values():
            _sub.proportion = 1.0
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

MIX_TEST_LANES = 20

MIX_TEST_LEVELS = 1

MIX_TEST_DIFFICULTY = 0.70

MIX_TEST_TILE_SIZE = (20.0, 4.0)

MIX_TEST_SUB_TERRAIN_KEY = "composite"

MIX_TEST_OBSTACLE_SPACING = 2.3636
MIX_TEST_MIN_OBSTACLE_SPACING = 1.50

MIX_TEST_SPAWN_CLEARANCE = 1.50

MIX_TEST_EPISODE_LENGTH_S = 35.0

MIX_TEST_EPISODE_LENGTH_MIN_S = 25.0

MIX_TEST_FORWARD_SPEED = 1.0

@configclass
class Imgo2CMoEMixTestEnvCfg(Imgo2CMoERoughPlayEnvCfg):
    """Deterministic 20 m composite-course evaluation at difficulty 0.70."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_generator.sub_terrains.clear()
        self.scene.terrain.terrain_generator.sub_terrains["composite"] = (
            CMoETrackCompositeTerrainCfg(
                proportion=1.0,

                corridor_width=0.80,
                pit_depth=0.50,
                spawn_x=0.75,

                spawn_clearance=MIX_TEST_SPAWN_CLEARANCE,

                obstacle_spacing=-1.0,
                min_obstacle_spacing=MIX_TEST_MIN_OBSTACLE_SPACING,

                suggested_spacing_range=(1.50, 2.50),

                gap_width_range=(0.12, 0.32),

                stairs_num_steps=4,
                stairs_step_height_range=(0.05, 0.20),
                stairs_step_depth=0.30,

                box_count=2,
                box_height_range=(0.08, 0.30),
                box_length_range=(0.30, 0.50),
                box_spacing=1.30,

                hurdle_count=2,
                hurdle_len_range=(0.04, 0.12),
                hurdle_height_min_slope=0.08,
                hurdle_height_max_base=0.06,
                hurdle_height_max_slope=0.10,
                hurdle_spacing_range=(0.48, 0.80),
                hurdle_height_fraction=0.5,
                hurdle_spacing_fraction=0.5,
            )
        )

        self.scene.terrain.terrain_generator.size = MIX_TEST_TILE_SIZE

        self.scene.terrain.terrain_generator.num_cols = MIX_TEST_LANES
        self.scene.terrain.terrain_generator.num_rows = MIX_TEST_LEVELS

        self.scene.terrain.terrain_generator.difficulty_range = (
            MIX_TEST_DIFFICULTY,
            MIX_TEST_DIFFICULTY,
        )

        self.scene.terrain.max_init_terrain_level = 0

        self.scene.num_envs = MIX_TEST_LANES

        self.episode_length_s = MIX_TEST_EPISODE_LENGTH_S

        self.commands.base_velocity = mdp.MixTestVelocityCommandCfg(
            asset_name="robot",

            resampling_time_range=(1.0e9, 1.0e9),

            heading_command=True,
            rel_heading_envs=1.0,
            rel_standing_envs=0.0,

            debug_vis=True,
            ranges=mdp.MixTestVelocityCommandCfg.Ranges(

                lin_vel_x=(1.0, 1.0),

                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),

                heading=(0.0, 0.0),
            ),
        )

@configclass
class Imgo2CMoEGaitFreeEnvCfg(Imgo2CMoERoughEnvCfg):
    """Disable fixed-gait shaping when training from a pretrained locomotion prior."""

    def __post_init__(self):
        super().__post_init__()

        # The parent has already removed zero-weight terms.
        if self.rewards.joint_mirror is not None:
            self.rewards.joint_mirror.weight = 0.0
        if self.rewards.feet_air_time is not None:
            self.rewards.feet_air_time.weight = 0.0
        if self.rewards.feet_height_body is not None:
            self.rewards.feet_height_body.weight = 0.0
        if self.rewards.feet_air_time_variance is not None:
            self.rewards.feet_air_time_variance.weight = 0.0
        if self.rewards.feet_gait is not None:
            self.rewards.feet_gait.weight = 0.0

        self.disable_zero_weight_rewards()
