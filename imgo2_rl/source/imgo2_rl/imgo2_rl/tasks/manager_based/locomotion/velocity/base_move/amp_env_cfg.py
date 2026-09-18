from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

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
    MySceneCfg,
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
        # AMP-05（用户 2026-09-17 决定）：actor 不再观察 base_lin_vel —— 真机上基座线速度
        # 需要状态估计，难以可靠获得。移除后 actor 观察为 45 维：
        #   base_ang_vel 3 + projected_gravity 3 + velocity_commands 3 + joint_pos 12
        #   + joint_vel 12 + last_action 12 = 45
        # 注意：critic 仍保留 base_lin_vel（它不上真机，保留有利于价值估计）。
        # 影响：现有 model_9000.pt 的第一层是 [512, 48]，与新配置不兼容，必须重新训练；
        # 不得直接截掉旧网络输入。配套要同步的还包括回放、导出与部署侧 amp/config.yaml。
        self.observations.policy.base_lin_vel = None
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

        if self.__class__.__name__ in ("Imgo2AmpMoveEnvCfg", "Imgo2AmpGo2StyleEnvCfg",
                                       "Imgo2AmpRLAmpEnvCfg", "Imgo2AmpRoughEnvCfg"):
            self.disable_zero_weight_rewards()

    def _keep_only_amp_task_rewards(self):
        # AMP-06 任务项（2026-09-17 第二次调参）：
        #   速度项从 1.0 抬到 4.0（**只抬 4 倍**，不是上一轮的 50 倍），
        #   高度项从 -10 降到 -5（**仍强于单个速度项**，避免重演「趴低滑行」），
        #   并补三个轻量姿态约束，使姿态不再只由一个超强高度项独自承担。
        #   依据：a1 论文代码的真实每步系数是 1.67（不是 50——它配置里的 50 还会被
        #   legged_robot.py:710 再乘一次 dt），go2 用 4.0/2.0 + 一组辅助项 + 任务主导 lerp。
        keep_rewards = {
            "track_lin_vel_xy_exp", "track_ang_vel_z_exp", "base_height_l2",
            "lin_vel_z_l2", "ang_vel_xy_l2", "joint_pos_limits",
        }
        for attr in dir(self.rewards):
            if attr.startswith("__"):
                continue
            reward_attr = getattr(self.rewards, attr)
            if hasattr(reward_attr, "weight") and attr not in keep_rewards:
                setattr(self.rewards, attr, None)

        # 单位换算：AMP 风格奖励是**每步**（runner 里直接算，不进 RewardManager），
        # 而 RewardManager 计算 `term × weight × dt`，weight 的语义是「每秒」。
        # 因此 weight 一律写成 `每步系数 / step_dt`，由代码换算，不手写魔数。
        #
        # 历史（勿重复）：
        #   Run1（1.0 / 0.3 / -10）：策略只站不走（速度误差恒 1.65 m/s，1000 轮后全平）。
        #   Run2（50 / 17 / -1，5000 轮）：速度项生效但退化为贴地滑行（高度 0.172 m、
        #     贴地 0.89），且噪声 std 失控到 12、价值损失 ~5000、判别器饱和。
        #     根因之一是高度项相对速度项弱了约 3000 倍。
        TRACK_LIN_VEL_PER_STEP = 4.0      # Run1 1.0 / Run2 50；go2 用 4.0
        TRACK_ANG_VEL_PER_STEP = 2.0      # Run1 0.3 / Run2 17；go2 用 2.0
        BASE_HEIGHT_PER_STEP = -5.0       # Run1 -10 / Run2 -1；比单个速度项强、比两项之和小
        LIN_VEL_Z_PER_STEP = -1.0         # go2 的 lin_vel_z
        ANG_VEL_XY_PER_STEP = -0.05       # go2 的 ang_vel_xy
        JOINT_POS_LIMITS_PER_STEP = -2.0  # go2 的 dof_pos_limits

        step_dt = self.sim.dt * self.decimation
        self.rewards.track_lin_vel_xy_exp.weight = TRACK_LIN_VEL_PER_STEP / step_dt
        self.rewards.track_ang_vel_z_exp.weight = TRACK_ANG_VEL_PER_STEP / step_dt
        self.rewards.lin_vel_z_l2.weight = LIN_VEL_Z_PER_STEP / step_dt
        self.rewards.ang_vel_xy_l2.weight = ANG_VEL_XY_PER_STEP / step_dt
        self.rewards.joint_pos_limits.weight = JOINT_POS_LIMITS_PER_STEP / step_dt

        # 高度约束目标 0.30 m（参考动作的平均根高），平地不读高度扫描传感器。
        self.rewards.base_height_l2.weight = BASE_HEIGHT_PER_STEP / step_dt
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
        apply_amp_play_overrides(self)


##
# amp_go2 配方（2026-09-18）：把参考项目 ak1raljl/amp_go2 的奖励设计等价移植过来
##


def apply_amp_play_overrides(cfg) -> None:
    """`*-amp-play` 任务共用的回放覆盖（单环境、关域随机化、指令固定）。

    回放/评估（`play.py`、`eval_gait.py`）依赖这些覆盖：关掉参考状态初始化与全部随机化，
    把指令钉成常数，否则 8 个环境的统计不可比。
    """
    # ------------------------------Scene------------------------------
    cfg.scene.num_envs = 1

    # ------------------------------Events------------------------------
    cfg.events.reference_state_initialization = None
    cfg.events.randomize_rigid_body_material = None
    cfg.events.randomize_rigid_body_mass_base = None
    cfg.events.randomize_rigid_body_mass_others = None
    cfg.events.randomize_apply_external_force_torque = None
    cfg.events.randomize_push_robot = None
    cfg.events.randomize_com_positions = None
    cfg.events.randomize_actuator_gains = None
    # rlamp 任务会把这两项**重建**出来（见 `apply_rlamp_env_settings`）；回放/评估要确定性初始状态
    cfg.events.randomize_reset_base = None
    cfg.events.randomize_reset_joints = None

    # ------------------------------Commands------------------------------
    cfg.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


# amp_go2 的原始权重（`go2_amp_config.py` 的 `class scales`，11 个非零项，逐字抄录）。
# 键是我们 `RewardsCfg`（velocity_env_cfg.py）里的项名，值是 legged_gym 的**原始权重**。
#
# 单位语义（这是能与参考逐字对上的原因）：legged_gym 的 `_prepare_reward_function()` 会做
# `reward_scales[key] *= self.dt`，Isaac Lab 的 `RewardManager.compute(dt)` 做 `term*weight*dt`
# —— **两边 weight 的语义相同**，所以 `weight_il = weight_lg`（当 step_dt 都是 0.02 时）。
AMP_GO2_DT = 0.02  # amp_go2 的 step_dt = sim.dt 0.005 × decimation 4
AMP_GO2_RAW_WEIGHTS = {
    "track_lin_vel_xy_exp": 4.0,    # ← tracking_lin_vel
    "track_ang_vel_z_exp": 2.0,     # ← tracking_ang_vel
    "lin_vel_z_l2": -1.0,           # ← lin_vel_z
    "ang_vel_xy_l2": -0.05,         # ← ang_vel_xy
    "joint_acc_l2": -2.5e-7,        # ← dof_acc
    "joint_torques_l2": -1e-4,      # ← torques
    "base_height_l2": -1.0,         # ← base_height（比我们 AMP-only 配方的每步 -5.0 弱 250 倍）
    "action_rate_l2": -0.01,        # ← action_rate
    "undesired_contacts": -1.0,     # ← collision（amp_go2 只惩罚 thigh）
    "joint_pos_limits": -2.0,       # ← dof_pos_limits
    "feet_air_time": 1.0,           # ← feet_air_time
}
# 必须偏离参考的两处（理由见 Imgo2AmpGo2StyleEnvCfg 的 docstring）
AMP_GO2_BASE_HEIGHT_TARGET = 0.30       # amp_go2 用 0.38（Go2 站高）；Imgo2 参考动作是 0.297
AMP_GO2_FEET_AIR_TIME_THRESHOLD = 0.3   # 用户 2026-09-18 定 0.3 s（参考硬编码 0.5 更慢）
# amp_go2 `penalize_contacts_on = ["thigh"]`；a1 还含 calf 并把两者纳入终止
AMP_GO2_CONTACT_PENALTY_BODIES = ".*_THIGH"


def _amp_go2_term(name: str, foot_body_names):
    """重建被 `_keep_only_amp_task_rewards()` 清空的那 5 项（定义与 `RewardsCfg` 保持一致）。

    注意 `SceneEntityCfg` 的参数必须给出能匹配到 body/joint 的正则：Isaac Lab 在构造环境时解析
    每个 `SceneEntityCfg`，遇到 `""` 会报 "Not all regular expressions are matched!"。
    """
    if name == "joint_acc_l2":
        return RewTerm(func=mdp.joint_acc_l2, weight=0.0,
                       params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")})
    if name == "joint_torques_l2":
        return RewTerm(func=mdp.joint_torques_l2, weight=0.0,
                       params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")})
    if name == "action_rate_l2":
        return RewTerm(func=mdp.action_rate_l2, weight=0.0)
    if name == "undesired_contacts":
        return RewTerm(func=mdp.undesired_contacts, weight=0.0,
                       params={"sensor_cfg": SceneEntityCfg(
                           "contact_forces", body_names=AMP_GO2_CONTACT_PENALTY_BODIES),
                           "threshold": 1.0})
    if name == "feet_air_time":
        return RewTerm(func=mdp.feet_air_time, weight=0.0,
                       params={"command_name": "base_velocity",
                               "threshold": AMP_GO2_FEET_AIR_TIME_THRESHOLD,
                               "sensor_cfg": SceneEntityCfg("contact_forces", body_names=foot_body_names)})
    raise KeyError(f"没有为 {name} 定义 amp_go2 版的奖励项")


@configclass
class Imgo2AmpGo2StyleEnvCfg(Imgo2AmpMoveEnvCfg):
    """amp_go2（`ak1raljl/amp_go2`）配方的等价移植：**任务奖励负责步态，AMP 只做轻量先验**。

    与现有 `Imgo2AmpMoveEnvCfg`（AMP-only：6 项任务、每步 4.0/2.0/-5.0）的区别，就是参考项目
    与我们的差别 —— 参考在 `go2_amp_config.py` 里保留了 legged_gym 的整套步态奖励
    （`feet_air_time`、`collision`、`action_rate`、`dof_acc`、`torques`），并把风格权重压到很小
    （`coef 0.2` / `lerp 0.8` ⇒ 风格 ≤ 0.04/步）。依据见
    `docs/amp_gait_adjust_plan_2026-09-18.md` §7/§8：三个数据集的对照显示「节律」在这套判别器
    +损失+`λ_gp=10` 下天生是弱信号，所以参考能走的步态来自任务奖励，而不是风格项。

    与参考的**两处有意偏离**（必须写清楚，否则会被当成抄漏）：
      * `base_height` 权重照抄 -1.0（每步 -0.02，比我们原来的每步 -5.0 弱 250 倍），但目标高度
        用 Imgo2 的 0.30 m（参考是 Go2 的 0.38）；
      * `feet_air_time` 阈值用 **0.3 s**（用户 2026-09-18 决定；参考硬编码 0.5 s，等于奖励
        "滞空 >0.5 s / 周期 ≥1 s"，比我们参考动作的 0.600 s 周期还慢）。
        0.3 s 对应的目标周期：我们实测占空比 0.50–0.56（腾空比例 0.44–0.50），
        滞空 0.3 s ⇒ 周期 **0.60–0.69 s（1.45–1.67 Hz）**，正好落在参考的 0.600 s / 1.67 Hz 附近。
        该项每步奖励 = Σ_落地足 (滞空 − θ)，稳态下 ≈ (1−占空比) − θ/周期 ⇒ 对周期的推动 ∂/∂T = θ/T²，
        因此阈值越大推力越强（θ=0 时该项对步频完全没有激励）。
    """

    def __post_init__(self):
        super().__post_init__()
        self._apply_amp_go2_rewards()

    def _apply_amp_go2_rewards(self) -> None:
        step_dt = self.sim.dt * self.decimation
        for name, raw in AMP_GO2_RAW_WEIGHTS.items():
            term = getattr(self.rewards, name, None)
            if term is None:  # 被 _keep_only_amp_task_rewards() 清空的项，按参考重建
                term = _amp_go2_term(name, self.foot_body_names)
                setattr(self.rewards, name, term)
            # weight_il = raw × dt_lg / step_dt_il ⇒ 每步系数 = raw × dt_lg，与 legged_gym 等价
            term.weight = raw * AMP_GO2_DT / step_dt

        self.rewards.base_height_l2.params["target_height"] = AMP_GO2_BASE_HEIGHT_TARGET
        self.rewards.feet_air_time.params["threshold"] = AMP_GO2_FEET_AIR_TIME_THRESHOLD

        # 指令范围也照抄参考（`go2_amp_config.py` 的 `class commands.ranges`）：
        #   lin_vel_x [-1.2, 1.5]、lin_vel_y ±0.8、ang_vel_yaw ±1.0（我们原来 x (-1.0,1.5)、y ±1.0、yaw ±1.57）。
        # 注意：**x > ~0.9 m/s 段我们自己的录制数据覆盖不到**（参考动作实测最快 0.842 m/s），
        # 照抄参考的 1.5 上限会把"要求数据外的步态"这一点放大；评估时重点看 0.3–0.9。
        self.commands.base_velocity.ranges.lin_vel_x = (-1.2, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.8, 0.8)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class Imgo2AmpGo2StylePlayEnvCfg(Imgo2AmpGo2StyleEnvCfg):
    """`*-play` 版：与 `Imgo2AmpMovePlayEnvCfg` 相同的回放覆盖，用于 play/eval_gait。"""

    def __post_init__(self):
        super().__post_init__()
        apply_amp_play_overrides(self)


##
# rl_amp（fan-ziqi）配方：只留线/角速度跟踪奖励，风格由 AMP 负责
##

# 参考 `~/Desktop/AMP/rl_amp/legged_gym/legged_gym/envs/a1/a1_amp_config.py` 的
# `class scales`：除 `tracking_lin_vel = 1.5 * 1./(.005*6)`、`tracking_ang_vel = 0.5 * 1./(.005*6)`
# 外**全部为 0**（连 `base_height`/`lin_vel_z`/`ang_vel_xy`/`dof_pos_limits` 都是 0）。
# 这两个表达式算出的是 legged_gym 的**原始权重** 50.0 / 16.6667；
# legged_gym 的 `_prepare_reward_function()` 会再乘 dt=0.02 ⇒ **每步 1.0 / 0.3333**。
# 两边 weight 语义相同（都乘 dt），所以 `weight_il = raw × dt_lg / step_dt`。
RLAMP_DT = 0.02
RLAMP_TRACK_LIN_VEL_RAW = 1.5 * 1.0 / (0.005 * 6)
RLAMP_TRACK_ANG_VEL_RAW = 0.5 * 1.0 / (0.005 * 6)
# 参考的 AMP 侧：`amp_reward_coef = 2.0`、`amp_task_reward_lerp = 0.3`（= 我们的 `AMPRunnerCfg`），
# `λ_gp = 10`、判别器 `[1024,512]`、PPO 全套超参与我们逐项相同。按 `r = (1-lerp)·coef·style + lerp·task`
# 折算：风格上限 2.0×0.7 = **1.40/步**，任务上限 0.3×1.333 = **0.40/步**
# ⇒ 上限比 ≈1:1，而实测风格原始分约 0.39 时风格 ≈0.55/步、任务 ≈0.27/步 ⇒ **风格约占 2/3**。
RLAMP_KEEP_REWARDS = ("track_lin_vel_xy_exp", "track_ang_vel_z_exp")


def apply_rlamp_task_rewards(cfg) -> None:
    """把任务奖励换成 rl_amp（fan-ziqi）的配方：**只留线/角速度跟踪**，其余全为 0。

    与 `Imgo2AmpMoveEnvCfg._keep_only_amp_task_rewards()`（AMP-only 6 项、每步 4.0/2.0/−5.0）
    的区别就是"谁负责步态与姿态"：参考把这个责任完全交给 AMP 风格项，因此**没有高度项、
    没有姿态项、没有任何步态项**。这也是它唯一的防退化机制是「base 触地终止」的原因
    （rl_amp 的 `terminate_after_contacts_on = ["base"]`、`collision` 权重为 0）。
    """
    for attr in dir(cfg.rewards):
        if attr.startswith("__"):
            continue
        term = getattr(cfg.rewards, attr)
        if hasattr(term, "weight") and attr not in RLAMP_KEEP_REWARDS:
            setattr(cfg.rewards, attr, None)
    step_dt = cfg.sim.dt * cfg.decimation
    # 跟踪核：参考 `tracking_sigma = 0.25`（= std 0.5），与我们 `RewardsCfg` 的默认一致。
    cfg.rewards.track_lin_vel_xy_exp.weight = RLAMP_TRACK_LIN_VEL_RAW * RLAMP_DT / step_dt
    cfg.rewards.track_ang_vel_z_exp.weight = RLAMP_TRACK_ANG_VEL_RAW * RLAMP_DT / step_dt


# 参考的观测缩放与裁剪（`rl_amp` 基类 `legged_robot_config.py` 的 `class normalization`）：
#   obs_scales: lin_vel 2.0 / ang_vel 0.25 / dof_pos 1.0 / dof_vel 0.05；clip_observations 100.
#   `commands_scale = [lin_vel, lin_vel, ang_vel] = [2.0, 2.0, 0.25]`（`legged_robot.py:515`），
#   而且 **actor 与 critic 共用同一个缓冲区** ⇒ 两边的 commands 都带这个缩放。
#   `clip_actions = 100.`（≈不裁剪，见 `a1_amp_config.py` 的 `normalization`）。
RLAMP_OBS_SCALES = {"lin_vel": 2.0, "ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05}
RLAMP_COMMANDS_SCALE = (2.0, 2.0, 0.25)
RLAMP_CLIP_ACTIONS = (-100.0, 100.0)
RLAMP_ACTION_SCALE = 0.25   # a1_amp_config.py:72（基类默认是 0.5，参考覆盖成 0.25）


def apply_rlamp_obs_and_action_scales(cfg) -> None:
    """把观测缩放、动作缩放与裁剪统一到 rl_amp（fan-ziqi）的取值。

    与 `apply_rlamp_task_rewards()` 分开，是为了让"奖励配方"和"输入/动作接口"两个变量各自可查。

    * **critic**：`lin_vel 2.0 / ang_vel 0.25 / dof_pos 1.0 / dof_vel 0.05`
      （Isaac Lab 默认全是 1.0，我们此前的 AMP 任务也只缩放了 policy 组 ⇒ 这是真实偏差）。
    * **commands**（actor 与 critic 都改）：参考的 `commands_scale = [2.0, 2.0, 0.25]`；
      Isaac Lab 的 `ObsTerm.scale` 支持元组（`observation_manager.py:551-558` 会校验长度），
      正好可以逐分量表达。**注意这动到了 actor 的输入尺度**（不只是 critic）。
    * **动作**：`action_scale = 0.25` 统一（我们原来是髋 0.125 / 其余 0.25），
      `clip = ±100`（参考 `clip_actions = 100.`，即实际不裁剪；我们原来是 ±3）。

    ⚠️ 后两项属于**部署契约**：这两个任务的策略将来要导出到
    `imgo2_deploy/policy/imgo2/amp/` 时，`config.yaml` 必须同步
    （`action_scale` 全 0.25、`clip_actions_*` ±100、`commands_scale [2,2,0.25]`）。
    **现在不要改那份 yaml**——它服务的是已部署的 24500 策略（髋 0.125 / ±3）。
    """
    step_dt = cfg.sim.dt * cfg.decimation   # 未使用，保留以便将来按需换算
    assert step_dt > 0
    for group in ("policy", "critic"):
        obs = getattr(cfg.observations, group)
        obs.base_ang_vel.scale = RLAMP_OBS_SCALES["ang_vel"]
        obs.joint_pos.scale = RLAMP_OBS_SCALES["dof_pos"]
        obs.joint_vel.scale = RLAMP_OBS_SCALES["dof_vel"]
        obs.velocity_commands.scale = RLAMP_COMMANDS_SCALE
        if group == "critic":
            obs.base_lin_vel.scale = RLAMP_OBS_SCALES["lin_vel"]
    cfg.actions.joint_pos.scale = RLAMP_ACTION_SCALE
    cfg.actions.joint_pos.clip = {".*": RLAMP_CLIP_ACTIONS}


# ---------------------------------------------------------------------------------------
# 指令范围 / 域随机化 / reset 分布 / 观测噪声：同样对齐 rl_amp（fan-ziqi）的 `a1_amp_config.py`
# ---------------------------------------------------------------------------------------
# 参考的 `class commands.ranges`：`lin_vel_x [-1.0, 2.0]`、`lin_vel_y ±0.3`、
# `ang_vel_yaw ±1.57`（`resampling_time = 10.` 与 `heading_command = False` 我们本来就一致）。
RLAMP_COMMAND_RANGES = {
    "lin_vel_x": (-1.0, 2.0),
    "lin_vel_y": (-0.3, 0.3),
    "ang_vel_z": (-1.57, 1.57),
}
# 参考的 `class domain_rand`：**注意这个类没有继承 `LeggedRobotCfg.domain_rand`**，
# 只列了下面这些字段 ⇒ 基类里的其它随机化在参考里**根本不存在**（不是"关掉了"，是"没有"）：
#   randomize_friction / friction_range [0.25, 1.75]（legged_gym 的 friction 同时写进
#   `friction` 与 `friction`+`rolling_friction`；Isaac Lab 拆成 static/dynamic ⇒ 两个填同一范围）、
#   randomize_base_mass / added_mass_range [-1., 1.]（**只加在 base 上**，
#   `legged_robot.py:315-316` 只改 body 0）、
#   push_robots / push_interval_s 15 / max_push_vel_xy 1.0（`_push_robots` 把 x、y 速度都写成
#   ±max_push_vel_xy，其余四维为 0）、
#   randomize_gains / stiffness_multiplier_range [0.9, 1.1] / damping_multiplier_range [0.9, 1.1]
#   （`randomize_gains` 在 `_process_...`/`__init__` 期只做**一次**，不是每回合重采）。
# 参考**没有**：CoM 偏移、其它 body 的质量、外力/力矩、恢复系数随机化（legged_gym 的
# restitution 恒为 0，`_process_rigid_shape_props` 不写这个字段）、惯量随机化。
RLAMP_FRICTION_RANGE = (0.25, 1.75)
RLAMP_BASE_MASS_RANGE = (-1.0, 1.0)
RLAMP_GAIN_MULTIPLIER_RANGE = (0.9, 1.1)
RLAMP_PUSH_INTERVAL_S = (15.0, 15.0)
RLAMP_PUSH_VEL_XY = 1.0
# 参考 `class noise(noise_scales)` 相对 legged_gym 基类**覆盖**了两项：
#   `dof_pos = 0.03`（基类 0.01）、`ang_vel = 0.3`（基类 0.2）；其余（lin_vel 0.1 /
#   dof_vel 1.5 / gravity 0.05）与基类相同 ⇒ 我们只需改这两项。
# `legged_robot.py:466-478` 的 `noise_scale_vec` 在**缩放后**的观测上加
# `noise_scale × obs_scale` 的均匀噪声 ⇒ 换算成 Isaac Lab "raw 空间先加噪声再乘 scale"
# 就是 raw 噪声等于 `noise_scales` 的数值本身（我们的 `Unoise` 正是 raw 空间）。
# ⚠️ 注意 `ang_vel` 这一条在参考里其实是**死代码**：参考的 actor 观测是
# `privileged_obs_buf[:, 6:]`（`legged_robot_amp.py:269`，条件是
# `num_obs == num_privileged_obs - 6`）⇒ 参考 actor（42 维）**既不看线速度也不看角速度**，
# 加在 `[:, 3:6]` 上的噪声随即被切掉。我们保留 `base_ang_vel`（真机 IMU，用户决定），
# 所以这一条在我们这里**有效**：取参考声明的 0.3 是对齐它"声明值"的最直接做法。
RLAMP_NOISE_SCALES = {"ang_vel": 0.3, "dof_pos": 0.03}
# reset 分布（参考 `legged_robot.py` 的 `_reset_root_states` / `_reset_dofs`）：
#   * 平地（`custom_origins=False`）：姿态取 `init_state` 原值（**不加** xy/偏航扰动），
#     根速度 6 维全部 U(-0.5, 0.5)；
#   * 粗糙/高度场（`custom_origins=True`）：在上一行基础上，xy 再各加 U(-1, 1) m
#     （`_create_envs` 里的初始摆放也是 ±1 m，但那只影响第一个 episode）；
#   * 关节：`dof_pos = default_dof_pos × U(0.5, 1.5)`、`dof_vel = 0`（**乘默认角**，不是加偏移）。
RLAMP_RESET_JOINT_SCALE_RANGE = (0.5, 1.5)
RLAMP_RESET_VEL_RANGE = (-0.5, 0.5)
RLAMP_ROUGH_RESET_XY_RANGE = (-1.0, 1.0)
# 参考 `a1_amp_config.py` 的 `env`：`reference_state_initialization = True`、
# `reference_state_initialization_prob = 0.85` ⇒ **85% 的 reset 用参考动作状态、15% 走
# `_reset_dofs`/`_reset_root_states`**（就是上面那套 reset 分布）。我们原来是 1.0（全都用参考状态）
# ⇒ 若不改这一项，上面重建的 reset 随机化在平地版上会被参考状态初始化**完全覆盖**而失效。
RLAMP_REFERENCE_INIT_PROB = 0.85


def apply_rlamp_env_settings(cfg, *, custom_origins: bool = False) -> None:
    """指令范围、域随机化、reset 分布、观测噪声统一到 rl_amp（fan-ziqi）的 `a1_amp_config.py`。

    与 `apply_rlamp_task_rewards()`（奖励）/`apply_rlamp_obs_and_action_scales()`（观测缩放、
    动作缩放与裁剪）分开，是为了让三块变量各自可查、可单独回滚。**用户 2026-09-18 的决定**：
    除关节 kp/kd（参考 kp 20 / kd 0.5，我们 25 / 0.5，见 `IMGO2_CFG`）以外全部对齐。

    `custom_origins=True` 用于粗糙地形版：参考只在"地形由 heightfield/trimesh 提供"的
    分支里对初始 xy 加 ±1 m 扰动（`_get_env_origins` 的 `custom_origins`）。

    ⚠️ **reset 项是"重建"而不是"改值"**：`Imgo2AmpMoveEnvCfg.__post_init__` 把
    `randomize_reset_base` / `randomize_reset_joints` 设成了 `None`（AMP 的 reset 由参考状态
    初始化负责），所以这里必须**新建** `EventTerm`——直接写 `.params[...]` 会在第一次构造配置时
    抛 `AttributeError: 'NoneType' object has no attribute 'params'`（2026-09-18 冒烟训练实测）。
    `EventManager` 按 `cfg.__dict__` 顺序执行 reset 项，而 `reference_state_initialization` 是
    子类字段 ⇒ **排在最后**、会覆盖它前面那 85% 的 env ⇒ 净效果正是参考的"85% 参考状态 /
    15% 自然 reset"（粗糙版关掉了参考初始化，于是 100% 走这套 reset 分布，见类 docstring）。

    ⚠️ 对齐的代价（写进 `docs/amp_gait_adjust_plan_2026-09-18.md` §15.2）：
      * `lin_vel_x` 上限 2.0 m/s 超出我们录制动作数据覆盖的 0.842 m/s ⇒ 高速指令段没有
        专家参考可比，AMP 风格项在那一段只能靠插值外推；
      * 关掉 CoM/其它 body 质量/外力力矩随机化、把摩擦与 PD 增益范围收窄、去掉初始姿态
        与偏航扰动、关节初始角只按 ×[0.5,1.5] 缩放，都是**减少**随机化 ⇒ 学到的策略会更贴合
        标称动力学，sim2real 鲁棒性不如我们原来的配置；
      * 参考 `randomize_gains` 是**启动时**随机一次，我们原来是 `mode="reset"`（每回合重采）。

    参考源码行号：`legged_robot.py:266-269`（摩擦）、`315-316`（base 质量）、
    `334`/`417`（推力）、`466-478`（噪声）、`399-409`（reset 根状态）、
    `411-425`（reset 关节）、`738`（`push_interval` 换算）。
    """
    # ---------------------------------- 指令范围 ----------------------------------
    for name, value in RLAMP_COMMAND_RANGES.items():
        setattr(cfg.commands.base_velocity.ranges, name, value)

    # ---------------------------------- 接触/摩擦 ----------------------------------
    mat = cfg.events.randomize_rigid_body_material
    mat.params["static_friction_range"] = RLAMP_FRICTION_RANGE
    mat.params["dynamic_friction_range"] = RLAMP_FRICTION_RANGE
    # 参考没有恢复系数随机化（legged_gym 的 restitution 恒 0）；写成 (0,0) 而不是删掉该项，
    # 是为了保留"谁负责什么"的可追溯性（该项本身仍会把 restitution 显式写成 0）。
    mat.params["restitution_range"] = (0.0, 0.0)

    # ---------------------------------- 质量 ----------------------------------
    cfg.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = RLAMP_BASE_MASS_RANGE
    # 参考只随机化 base ⇒ 关掉"其它 body"的两项（质量缩放、CoM 偏移）
    cfg.events.randomize_rigid_body_mass_others = None
    cfg.events.randomize_com_positions = None

    # ---------------------------------- 外力/力矩 ----------------------------------
    # 参考没有这个随机化；原有的 ±10 N/Nm 是我们自己加的（非参考项）
    cfg.events.randomize_apply_external_force_torque = None

    # ---------------------------------- PD 增益 ----------------------------------
    gains = cfg.events.randomize_actuator_gains
    gains.mode = "startup"   # 参考在环境创建时随机一次（不是每回合重采）
    gains.params["stiffness_distribution_params"] = RLAMP_GAIN_MULTIPLIER_RANGE
    gains.params["damping_distribution_params"] = RLAMP_GAIN_MULTIPLIER_RANGE

    # ---------------------------------- 推力 ----------------------------------
    push = cfg.events.randomize_push_robot
    push.mode = "interval"
    push.interval_range_s = RLAMP_PUSH_INTERVAL_S
    push.params["velocity_range"] = {"x": (-RLAMP_PUSH_VEL_XY, RLAMP_PUSH_VEL_XY),
                                     "y": (-RLAMP_PUSH_VEL_XY, RLAMP_PUSH_VEL_XY)}

    # ---------------------------------- reset 分布 ----------------------------------
    # 2026-09-18 冒烟训练发现：这两项在 AMP 基类里是 `None`，必须**新建** EventTerm。
    # 语义核对：`reset_root_state_uniform` 把姿态写成 `default_root_state + env_origins + 采样`、
    # 速度写成 `default_root_vel + 采样`（`events.py`）；我们的默认根速度是 0 ⇒ 等价于参考的
    # "直接设成 U(-0.5,0.5)"。
    xy = RLAMP_ROUGH_RESET_XY_RANGE if custom_origins else (0.0, 0.0)
    cfg.events.randomize_reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": xy, "y": xy, "yaw": (0.0, 0.0)},
            "velocity_range": {key: RLAMP_RESET_VEL_RANGE
                               for key in ("x", "y", "z", "roll", "pitch", "yaw")},
        },
    )
    cfg.events.randomize_reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": RLAMP_RESET_JOINT_SCALE_RANGE, "velocity_range": (0.0, 0.0)},
    )
    # 参考的 85% 参考状态 / 15% 自然 reset（粗糙版没有参考初始化，100% 走上面这套）
    init = cfg.events.reference_state_initialization
    if init is not None:
        init.params["reference_state_initialization_prob"] = RLAMP_REFERENCE_INIT_PROB

    # ---------------------------------- 观测噪声（policy 组）----------------------------------
    # 参考只对 actor 观测加噪声（privileged 不加），我们的 `policy.enable_corruption = True` /
    # `critic.enable_corruption = False` 本来就是这个语义。
    cfg.observations.policy.base_ang_vel.noise = Unoise(
        n_min=-RLAMP_NOISE_SCALES["ang_vel"], n_max=RLAMP_NOISE_SCALES["ang_vel"])
    cfg.observations.policy.joint_pos.noise = Unoise(
        n_min=-RLAMP_NOISE_SCALES["dof_pos"], n_max=RLAMP_NOISE_SCALES["dof_pos"])


@configclass
class Imgo2AmpRLAmpEnvCfg(Imgo2AmpMoveEnvCfg):
    """**rl_amp（fan-ziqi）配方**的平地版：只有线/角速度跟踪奖励，风格占约 2/3。

    用途：与 `Imgo2AmpRoughEnvCfg`（粗糙地形、**同一配方**）组成一对，让"地形"成为单变量，
    对应 `paper_plan_imgo2.md` 里"比较平地训练、粗糙地形训练、动力学随机化训练的泛化能力"。
    AMP 侧沿用 `AMPRunnerCfg`（`coef 2.0 / lerp 0.3`），即参考的比例。

    2026-09-18 用户决定「除关节 kp/kd 以外全部对齐参考」，于是本类在三个 helper 里依次改：
    `apply_rlamp_task_rewards`（奖励）、`apply_rlamp_obs_and_action_scales`（观测缩放 + 动作
    缩放/裁剪）、`apply_rlamp_env_settings`（指令范围 + 域随机化 + reset 分布 + 噪声）。
    与参考仍**不一致**的只有：关节 kp/kd（用户排除）、actor 观测维数（我们 45 含
    `base_ang_vel`，参考 42）、以及 AMP 判别的数据来源（我们录制的真机动作 vs 参考的 mocap）。
    """

    def __post_init__(self):
        super().__post_init__()
        apply_rlamp_task_rewards(self)
        apply_rlamp_obs_and_action_scales(self)
        apply_rlamp_env_settings(self)


@configclass
class Imgo2AmpRLAmpPlayEnvCfg(Imgo2AmpRLAmpEnvCfg):
    """`*-play` 版：与其它 AMP play 相同的回放覆盖。"""

    def __post_init__(self):
        super().__post_init__()
        apply_amp_play_overrides(self)


##
# 粗糙地形版（2026-09-18）：**配方改为忠于 rl_amp**（只留速度跟踪），地形相关四处改动
##


@configclass
class Imgo2AmpRoughEnvCfg(Imgo2AmpMoveEnvCfg):
    """**rl_amp（fan-ziqi）配方**的粗糙地形版。

    **奖励 = 参考原样**：只留 `tracking_lin_vel`、`tracking_ang_vel`（每步 1.0 / 0.3333），
    `base_height`/`lin_vel_z`/`ang_vel_xy`/`dof_pos_limits`/`feet_air_time`/`collision`/
    `action_rate`/`dof_acc`/`torques` **全为 0**；AMP 侧 `coef 2.0 / lerp 0.3`（风格约占 2/3）。
    与它成对的平地版是 `Imgo2AmpRLAmpEnvCfg`（同一配方）⇒ 平地对粗糙是可解释的单变量对照。

    除配方之外，有地形与"因地形而必须改"的四处，外加一处**为学得动而加的特权输入**：

    1. 地形：恢复 `ROUGH_TERRAINS_CFG` 生成器 + `terrain_levels` 课程；子地形范围沿用仓库里
       已有的 PPO rough 任务（`rough_env_cfg.py`）—— 我们的机器人站高只有 0.30 m，
       用 Isaac Lab 默认的 boxes 0.05–0.2 m / stairs 0.05–0.15 m 太陡。
    2. 恢复 9 射线 `height_scanner_base`（**只给 AMP 观测用**，不进 actor；rl_amp 配方没有高度奖励）
    3. **AMP 观测的根高改成地形相对**：`observations.amp.root_z` 传同一个扫描器
       ⇒ 43 维 AMP 观测的最后一维是「离地形的高度」，与专家数据（平地 0.297 m）同口径；
       否则这一维会被地形起伏主导（见 `mdp/observations.py:amp_root_z` 的说明）。
    4. **关掉 AMP 参考状态初始化**：`reset_amp_reference_state()` 把根高写成参考动作的**绝对**
       高度（0.297 m）+ 一个常数偏移，**没有地形补偿**（`amp_events.py:60-61` 只把 x/y 加上
       env_origins）⇒ 在抬高的地形格子上会把机器人初始化到地面以下。粗糙地形下改为普通 reset。

    **刻意不做**（保持"简单配置"，一次只改一个变量）：
      * （已按用户决定加上）`height_scanner`（187 维网格）**只喂 critic**，actor 保持 **45 维盲走**，
        部署契约（`amp/deploy/config.yaml`、C++ 接口）不变 —— 非对称 actor-critic；
      * 不加 `feet_air_time` / `collision` / `action_rate` 等任务项（那是 amp_go2 路线，
        平地版已单独做成 `Imgo2AmpGo2StyleEnvCfg`）；
      * **域随机化与指令范围已按用户 2026-09-18 决定对齐参考**
        （`apply_rlamp_env_settings`：摩擦 [0.25,1.75]、base 质量 ±1 kg、PD 增益 ×[0.9,1.1]
        且仅启动时随机一次、推力 ±1.0 m/s 每 15 s、关掉 CoM/其它 body 质量/外力力矩；
        指令 x[-1.0,2.0]、y±0.3、yaw±1.57）。代价：对齐是**减少**随机化，鲁棒性不如原配置；
        且 2.0 m/s 超出我们录制数据覆盖的 0.842 m/s，那一段没有专家参考可比。

    **风险（必须知道）**：rl_amp 配方**没有任何姿态/高度约束**，参考项目靠"风格项 + base 触地终止"
    兜底；而我们的风格项实测**梯度≈0**（见 `docs/amp_gait_adjust_plan_2026-09-18.md` §1–§3），
    因此"趴地滑行"（Run2 实测 0.172 m、贴地率 0.89）的风险比平地版更高。训练时务必盯
    `AMP/mean_root_height_m` 与 `AMP/fraction_root_height_below_0_20m`；若趴地，就回到
    `Imgo2AmpMoveEnvCfg` 的 6 项配方（含 −5/步 高度项）或只把高度项单独加回（单变量）。
    """

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------Scene：地形------------------------------
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        # 187 维高度网格：**只喂 critic**（特权信息，见下面的 Observations 段）
        self.scene.height_scanner = MySceneCfg.height_scanner
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        gen = self.scene.terrain.terrain_generator
        gen.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        gen.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        gen.sub_terrains["random_rough"].noise_step = 0.01
        gen.sub_terrains["pyramid_stairs"].step_height_range = (0.025, 0.08)
        gen.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.025, 0.08)

        # 高度课程（`LocomotionVelocityRoughEnvCfg.__post_init__` 会在它非 None 时把
        # `terrain_generator.curriculum` 置 True）
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

        # ------------------------------扫描器（只给 AMP 观测用，不进 actor）------------------------------
        # 9 条射线、悬在 base 正上方；`MySceneCfg` 里已声明（prim_path 指 /Robot/base，
        # mesh_prim_paths 指地面）。注意 rl_amp 配方**没有高度奖励**，所以它只服务下面第 3 条。
        self.scene.height_scanner_base = MySceneCfg.height_scanner_base

        # ------------------------------Observations：critic 用特权高度扫描，actor 仍盲走------------------------------
        # 非对称 actor-critic：critic 多 187 维地形信息（48 → 235，正好等于 amp_go2 的
        # `num_privileged_obs = 235`），**actor 保持 45 维** ⇒ 部署契约（`amp/config.yaml`、
        # C++ 接口）不变。判别器吃的是 `observations.amp`（43 维），与此无关。
        # 代价：粗糙版因此比平地版多一个变量（"critic 是否看地形"），平地对粗糙的对照不再严格单变量。
        self.observations.critic.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        # actor 侧保持 None（45 维）；这两条是刻意的，别顺手改成一致
        assert self.observations.policy.height_scan is None, "actor 必须保持 45 维盲走"

        # ------------------------------AMP 观测：根高改地形相对------------------------------
        self.observations.amp.root_z.params["sensor_cfg"] = SceneEntityCfg("height_scanner_base")

        # ------------------------------Events：关掉参考状态初始化------------------------------
        # 理由见类 docstring 第 4 条：它写的是参考动作的绝对根高，没有地形补偿。
        self.events.reference_state_initialization = None

        # ------------------------------奖励与接口：全部统一到 rl_amp------------------------------
        # 必须在最后调用：`apply_rlamp_task_rewards` 会把
        # `base_height_l2`/`lin_vel_z_l2`/`ang_vel_xy_l2`/`joint_pos_limits` 一并清空
        # （参考里这些权重都是 0），只留线/角速度跟踪；后者统一观测缩放与动作缩放/裁剪。
        apply_rlamp_task_rewards(self)
        apply_rlamp_obs_and_action_scales(self)
        # 指令范围 + 域随机化 + reset 分布 + 噪声（含 `randomize_actuator_gains` 改成启动时随机）。
        # `custom_origins=True`：参考的地形分支里初始 xy 有 ±1 m 扰动。
        apply_rlamp_env_settings(self, custom_origins=True)


@configclass
class Imgo2AmpRoughPlayEnvCfg(Imgo2AmpRoughEnvCfg):
    """`*-play` 版：与平地 play 相同的回放覆盖（单环境、关域随机化、指令固定）。

    注意粗糙地形下回放仍会生成地形（保留课程配置），所以 `eval_gait.py` 的足端指标
    （占空比/步周期/相位）在斜坡上含义与平地不同 —— 评估时优先看 `base_height`（地形相对）
    与接触时序，别直接和 `docs/gait_reference_baseline.json`（平地录制）逐项比。
    """

    def __post_init__(self):
        super().__post_init__()
        apply_amp_play_overrides(self)
