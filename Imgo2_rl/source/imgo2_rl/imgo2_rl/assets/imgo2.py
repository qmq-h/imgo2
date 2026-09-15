from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

import glob
AMP_MOTION_FILES = glob.glob(r"/root/gpufree-data/Imgo2_rl/datasets/imgo2_motion/*")

IMGO2_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path="/root/gpufree-data/Imgo2_rl/source/imgo2_rl/data/Imgo2/Imgo2_urdf/urdf/imgo2.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=1
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        joint_pos={
            ".*_hip_joint": 0.0,
            ".*_thigh_joint": 0.87,
            ".*_shank_joint": -1.82,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[".*_joint"],
            effort_limit=23.7,
            saturation_effort=23.7,
            velocity_limit=30.1,
            # stiffness=25.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)
