from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from imgo2_rl.assets.imgo2 import IMGO2_CFG, AMP_MOTION_FILES
import imgo2_rl.tasks.manager_based.locomotion.velocity.mdp as mdp
# `mdp/__init__.py` deliberately does not star-import amp_events: that module pulls in
# rl_lab -> torch -> pybullet_utils, which would otherwise be dragged into every task
# registration. Import the module directly here instead, so the AMP reference-reset term
# exists exactly when the AMP config is loaded. Without this, `mdp.reset_amp_reference_state`
# raises AttributeError at class-body evaluation and the AMP task cannot be created at all.
from imgo2_rl.tasks.manager_based.locomotion.velocity.mdp import amp_events as mdp_amp
from imgo2_rl.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    EventCfg,
    LocomotionVelocityRoughEnvCfg,
    ObservationsCfg,
)


@configclass
class AMPEventCfg(EventCfg):
    reference_state_initialization = EventTerm(
        func=mdp_amp.reset_amp_reference_state,
        mode="reset",
        params={
            "motion_files": AMP_MOTION_FILES,
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True),
            "reference_state_initialization_prob": 1.0,
            "velocity_frame": "base",
        },
    )


@configclass
class AMPObservationCfg(ObservationsCfg):
    @configclass
    class AMPGroupCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.amp_joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
        )
        foot_pos_base = ObsTerm(
            func=mdp.amp_foot_pos_base,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_FOOT", preserve_order=True)},
        )
        root_lin_vel_b = ObsTerm(func=mdp.base_lin_vel)
        root_ang_vel_b = ObsTerm(func=mdp.base_ang_vel)
        joint_vel = ObsTerm(
            func=mdp.amp_joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
        )
        root_z = ObsTerm(func=mdp.amp_root_z)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    amp: AMPGroupCfg = AMPGroupCfg()


@configclass
class Imgo2AmpMoveEnvCfg(LocomotionVelocityRoughEnvCfg):
    base_link_name = "base"
    foot_link_name = ".*_FOOT"
    foot_body_names = ["FL_FOOT", "FR_FOOT", "RL_FOOT", "RR_FOOT"]
    # The bundled imgo2_motion data already stores FL, FR, RL, RR blocks.
    # Keep observation and reference-reset mappings in the same order.
    # Verify a replacement dataset with scripts/tools/audit_amp_dataset.py.
    joint_mapping = list(range(12))
    joint_names = [
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_shank_joint",
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_shank_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_shank_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_shank_joint",
    ]

    observations: AMPObservationCfg = AMPObservationCfg()
    events: AMPEventCfg = AMPEventCfg()

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------Scene------------------------------
        self.scene.robot = IMGO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None

        # ------------------------------Observations------------------------------
        # Keep base_lin_vel for model_9000.pt (48 actor observations).
        # TODO AMP-05: remove it with a newly trained 45-input checkpoint and
        # update play/export/deployment together (README.md issue AMP-05).
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None

        # ------------------------------Curriculums------------------------------
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.57, 1.57)

        # ------------------------------Actions------------------------------
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.joint_pos.clip = {".*": (-3.0, 3.0)}
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------Observation Entities------------------------------
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.amp.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.amp.joint_pos.params["mapping"] = self.joint_mapping
        self.observations.amp.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.amp.joint_vel.params["mapping"] = self.joint_mapping
        self.observations.amp.foot_pos_base.params["asset_cfg"].body_names = self.foot_body_names
        self.observations.amp.foot_pos_base.params["mapping"] = self.joint_mapping

        # ------------------------------Events------------------------------
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_actuator_gains.params["asset_cfg"].joint_names = self.joint_names
        self.events.randomize_reset_base = None
        self.events.randomize_reset_joints = None
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.reference_state_initialization.params["asset_cfg"].joint_names = self.joint_names
        self.events.reference_state_initialization.params["joint_mapping"] = self.joint_mapping

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]

        # ------------------------------Rewards------------------------------
        self._keep_only_amp_task_rewards()

        if self.__class__.__name__ == "Imgo2AmpMoveEnvCfg":
            self.disable_zero_weight_rewards()

    def _keep_only_amp_task_rewards(self):
        keep_rewards = {"track_lin_vel_xy_exp", "track_ang_vel_z_exp", "base_height_l2"}
        for attr in dir(self.rewards):
            if attr.startswith("__"):
                continue
            reward_attr = getattr(self.rewards, attr)
            if hasattr(reward_attr, "weight") and attr not in keep_rewards:
                setattr(self.rewards, attr, None)

        # AMP style rewards are per policy step, while RewardManager multiplies
        # task terms by step_dt. Express these weights in per-step units too.
        step_dt = self.sim.dt * self.decimation
        self.rewards.track_lin_vel_xy_exp.weight = 1.0 / step_dt
        self.rewards.track_ang_vel_z_exp.weight = 0.3 / step_dt
        self.rewards.base_height_l2.weight = -10.0 / step_dt
        # Bundled reference motions have mean root heights around 0.30 m.
        # This is an explicit flat-ground height constraint, not frame tracking.
        self.rewards.base_height_l2.params["target_height"] = 0.30
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        # RewardsCfg declares base_height_l2 with body_names="" (velocity_env_cfg.py). Isaac Lab
        # resolves every SceneEntityCfg param at env construction and treats "" as a regex that
        # matches no body, then raises "Not all regular expressions are matched!". The reward
        # itself only reads root_pos_w, so naming the base body is enough to make it resolvable.
        # The sibling configs (rough_env_cfg.py, himloco_env_cfg.py) set this too.
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]


@configclass
class Imgo2AmpMovePlayEnvCfg(Imgo2AmpMoveEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ------------------------------Scene------------------------------
        self.scene.num_envs = 1

        # ------------------------------Events------------------------------
        self.events.reference_state_initialization = None
        self.events.randomize_rigid_body_material = None
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_rigid_body_mass_others = None
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None
        self.events.randomize_com_positions = None
        self.events.randomize_actuator_gains = None

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
