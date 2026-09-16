import glob
import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

# 以本文件位置推导路径，不再写死具体机器的绝对路径。
# 本文件位于 <repo>/imgo2_rl/source/imgo2_rl/imgo2_rl/assets/imgo2.py，
# 因此上溯 4 层即 imgo2_rl/ 项目根、5 层即仓库根。这样无论仓库被 clone 到哪里、
# 从哪个工作目录启动，路径都成立；前提是安装方式为 `pip install -e`（可编辑安装），
# 非可编辑安装会把包拷进 site-packages，届时数据与模型目录都不在上溯路径上。
_PROJECT_ROOT = Path(__file__).resolve().parents[4]  # <repo>/imgo2_rl/
_REPO_ROOT = Path(__file__).resolve().parents[5]     # <repo>/
_DEFAULT_MOTION_DIR = _PROJECT_ROOT / "datasets" / "imgo2_motion"
# 模型唯一源（2026-09-17 统一到 imgo2_description，见 README MODEL-02）：
# urdf/imgo2.urdf 是纯 URDF 生成物（core.xacro 的内核，无 Gazebo/transmission/IMU），
# 物理参数与训练侧的旧副本逐项相同，FK 复现录制数据。
_DEFAULT_URDF_PATH = _REPO_ROOT / "imgo2_description" / "urdf" / "imgo2.urdf"


def _resolve_motion_dir() -> Path:
    override = os.environ.get("IMGO2_AMP_MOTION_DIR")
    return Path(override).expanduser() if override else _DEFAULT_MOTION_DIR


def _resolve_urdf_path() -> Path:
    override = os.environ.get("IMGO2_URDF_PATH")
    return Path(override).expanduser() if override else _DEFAULT_URDF_PATH


# 环境变量 IMGO2_AMP_MOTION_DIR / IMGO2_URDF_PATH 可覆盖默认位置，便于数据或模型
# 放在别处时无需改代码。参考动作数据与训练用 URDF 都随仓库一起分发，默认即可用。
MOTION_DIR = _resolve_motion_dir()
URDF_PATH = _resolve_urdf_path()

# 注意：不在此处对缺失文件抛错。本模块在任务注册时即被导入，若在导入期报错会连带
# 影响 list_envs 等不需要 AMP 数据的入口。glob 为空时 AMPLoader 不会给出明确提示，
# 因此第一次训练前应确认这里确实读到了 21 份动作文件（见 README 的 ENV-01）。
AMP_MOTION_FILES = sorted(glob.glob(str(MOTION_DIR / "*")))

IMGO2_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=str(URDF_PATH),
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
