"""Class-based configuration skeleton for the hierarchical towing RL environment.

This module intentionally defines the complete configuration contract but is not registered yet:
the ManagerBased per-physics-step rope/resistance adapter must first reproduce ``tow_drag.py``.
"""

from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from imgo2_rl.assets.cart import make_cart_cfg
from imgo2_rl.assets.imgo2 import IMGO2_CFG
import imgo2_rl.tasks.manager_based.towing.upper_mdp as mdp


# 仓库根。本文件在 `imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/` 下，
# 上溯 7 层才是仓库根（`assets/imgo2.py` 在浅 2 层的 `assets/` 下，那里是 parents[5]）。
# 2026-09-22 发现原先写的是 parents[6] ⇒ 缓存落到 `<repo>/imgo2_rl/logs/usd/upper_towing`。
_REPO_ROOT = Path(__file__).resolve().parents[7]
_USD_CACHE = _REPO_ROOT / "logs" / "usd" / "upper_towing"

def _robot_link_names_from_urdf() -> tuple[str, ...]:
    """从训练用 URDF 读出 link 名（纯文本解析，不需要导入 Isaac Sim）。

    2026-09-22 在训练机核对过：`scene["robot"].body_names` 与 URDF 的 link 名逐项一致
    （17 个，导入器保留原名），所以这里直接用 URDF 作权威来源，避免硬编码漂移。
    """
    import re
    urdf = _REPO_ROOT / "imgo2_description" / "urdf" / "imgo2.urdf"
    return tuple(re.findall(r'<link\s+name="([^"]+)"', urdf.read_text(encoding="utf-8")))


def _robot_body_filters() -> list[str]:
    """构造「每个 filter 项在每环境只解析出一个 prim」的过滤列表。

    不能用 `{ENV_REGEX_NS}/Robot/.*`：Isaac Lab 的 `ContactSensorCfg` 要求每个 filter 项在
    每个环境里只匹配一个 prim（`contact_sensor_cfg.py` 的 attention 段）。2026-09-22 实测
    `Robot/.*` 每环境展开 19 个 prim × 256 环境 = 4864，触发
    `Filter pattern ... did not match the correct number of entries (expected 256, found 4864)`，
    于是 `force_matrix_w` 拿不到真实接触力 ⇒ `cart_collision` 恒假 ⇒ 碰撞 reward（−50）与
    碰撞终止全部失效。逐个列出后每项恰好解析出 1 个 prim。

    注意 `{ENV_REGEX_NS}` 展开成 `/World/envs/env_.*`，之后被 Isaac Lab 换成 `env_*`；
    环境命名空间里的通配是预期行为，违规的只是 `Robot/` 后面那一段。
    """
    return [f"{{ENV_REGEX_NS}}/Robot/{name}" for name in _robot_link_names_from_urdf()]


_ROBOT_BODY_FILTERS = _robot_body_filters()


@configclass
class UpperTowingSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.8, restitution=0.0,
                friction_combine_mode="average", restitution_combine_mode="min")))
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000))
    robot: ArticulationCfg = IMGO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cart: ArticulationCfg = make_cart_cfg(_USD_CACHE)[0]
    cart.init_state.pos = (-0.7564, 0.0, 0.18)
    # 每个车体部件一个传感器，各自与**逐个列出的**机器人 body 过滤，避免把轮地接触算进碰撞。
    # filter 项必须每环境只匹配一个 prim，见 `_ROBOT_BODY_FILTERS` 的说明。
    cart_deck_robot_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Cart/base_link", update_period=0.0, history_length=1,
        filter_prim_paths_expr=_ROBOT_BODY_FILTERS)
    cart_wheel_fl_robot_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Cart/wheel_fl", update_period=0.0, history_length=1,
        filter_prim_paths_expr=_ROBOT_BODY_FILTERS)
    cart_wheel_fr_robot_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Cart/wheel_fr", update_period=0.0, history_length=1,
        filter_prim_paths_expr=_ROBOT_BODY_FILTERS)
    cart_wheel_rl_robot_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Cart/wheel_rl", update_period=0.0, history_length=1,
        filter_prim_paths_expr=_ROBOT_BODY_FILTERS)
    cart_wheel_rr_robot_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Cart/wheel_rr", update_period=0.0, history_length=1,
        filter_prim_paths_expr=_ROBOT_BODY_FILTERS)


@configclass
class UpperActionsCfg:
    high_level_velocity = mdp.HierarchicalVelocityActionCfg(
        asset_name="robot", cart_asset_name="cart", policy_name="amp",
        acceleration_min=(-1.0, -0.5, -1.0), acceleration_max=(0.5, 0.5, 1.0),
        reference_min=(0.0, -0.3, -1.0), reference_max=(1.0, 0.3, 1.0),
        upper_control_dt=0.05, low_level_control_dt=0.02,
    )


@configclass
class UpperObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        frame = ObsTerm(func=mdp.policy_frame)

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = True

    @configclass
    class CriticCfg(ObsGroup):
        policy_frame = ObsTerm(func=mdp.policy_frame)
        robot_velocity = ObsTerm(func=mdp.robot_velocity)
        cart_velocity = ObsTerm(func=mdp.cart_velocity)
        rope_state = ObsTerm(func=mdp.rope_privileged_state)
        towing_force = ObsTerm(func=mdp.towing_force)
        cart_parameters = ObsTerm(func=mdp.cart_privileged_parameters)

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    @configclass
    class DecoderCfg(ObsGroup):
        targets = ObsTerm(func=mdp.decoder_targets)
        mass_weight = ObsTerm(
            func=mdp.decoder_mass_weight,
            params={"minimum_force": 1.0, "force_scale": 10.0},
        )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    decoder: DecoderCfg = DecoderCfg()


@configclass
class UpperRewardsCfg:
    """上层拖曳策略的奖励。

    **单位约定**：`RewardManager.compute()` 计算的是
    `value = func() × weight × step_dt`（`step_dt = sim.dt × decimation = 0.005 × 10 = 0.05 s`），
    所以下面每个 `weight` 都被乘了 0.05。备注里的"每步"已含该系数。

    **记录口径**：TensorBoard 的 `Episode_Reward/<项>` = 该回合的加权和 ÷ `max_episode_length_s`，
    量级比"每步值"小约 5~10 倍，不要直接与每步值比较。

    **设计取向（截至 2026-09-23）**：跟踪类先有 `tracking_velocity`（正向驱动），
    另外两项 `reference_tracking`/`action_magnitude` 用于修「上层动作 ±1 饱和抖动、
    `ref` 追不上 `cmd`」这一具体失败。
    """

    # ============================ 跟踪（任务主目标）============================

    # 【实际速度 vs 命令期望】唯一的正奖励，是策略的主要驱动。
    # exp(−(Δlin/0.5)² − (Δyaw/1.0)²)：完全跟上给 1.0，误差到 0.5 m/s 掉到 0.37、到 1.0 掉到 0.018。
    # 全程生效（settle 与 STOP 段 `user_command` 为 0，即"保持零速"）。
    # ⚠ 已知缺陷：误差 >1 m/s 后梯度趋零（exp 的尾部），而牵引段起步瞬间正落在该区，
    #   属"探索与奖励脱钩"的来源之一，尚未修改。
    tracking_velocity = RewTerm(func=mdp.velocity_tracking_exp, weight=1.0,
                                params={"linear_std": 0.5, "yaw_std": 1.0})

    # 【上层指令 vs 命令期望】惩罚 ‖ref − user‖²（m²/s²），单位是"速度误差平方"。
    # 与 tracking_velocity 的关键区别：比的是**上层自己产生的 speed 指令**，不穿过底层动力学，
    # 因此误差归因清晰（是"上层没给对指令"还是"底层跟不上"一目了然）。
    # 平方形式**全域有梯度**（不像 exp 有死区），牵引段起步即有效。
    # 量级：牵引段实测 err_cmd≈0.425 m/s ⇒ −0.90/步，是当前量级最大的惩罚之一。
    # 若发现策略为压低它而不敢动，降到 −2.0 左右。
    reference_tracking = RewTerm(func=mdp.reference_tracking_l2, weight=-5.0)

    # ============================ 终止级（稀疏、致命）============================

    # 【撞车】车斗／四轮与机器人任一刚体的过滤接触力 > 1 N 时置 1，同时是终止条件。
    # 权重虽大（−50 ⇒ 每步 −2.5），但只在真撞上时才给，属稀疏信号。
    # 背景：修复 filter 通配前该判据恒假（碰撞完全不生效），详见 docs/towing_observability_2026-09-22.md。
    collision = RewTerm(func=mdp.cart_collision_cost, weight=-50.0)

    # 【跌倒】base 高度 < 0.18 m 时置 1，同时是终止条件。实测从未触发。
    fall = RewTerm(func=mdp.robot_fall_cost, weight=-50.0,
                   params={"minimum_height": 0.18})

    # ============================ 几何／安全约束 ============================

    # 【软间隙障碍】softplus((0.20 − 间隙)/0.05)，间隙趋向 0 时加大。
    # 0.20 m 是**绝对**警戒线，线性区在警戒线**下方**；间隙 0.2 时约 −0.69/步、0 时约 −0.2/步。
    # 无小车环境用 cart_present 屏蔽。
    clearance = RewTerm(func=mdp.clearance_barrier, weight=-1.0,
                        params={"warning_distance": 0.20, "scale": 0.05})

    # 【硬最小间距】间隙低于 `ratio × rope_length` 时**与缺口成正比**地惩罚，高于则**精确为 0**。
    # 与上面 clearance 的分工：clearance 是"近了要缓"的软障碍；本项是"不得拉太近"的硬约束，
    # 且按**绳长比例**给出阈值（不是绝对量）。weight 的单位是"每米缺口扣多少"。
    # ratio=0.40 ⇒ 阈值 0.32 m；初始间隙 0.349 m 高于它，故 **spawn 时精确为 0**
    # （用户 2026-09-23 要求初始不生效）。铰链已减掉 softplus 偏置以免阈值上方被误扣分。
    min_clearance = RewTerm(func=mdp.min_clearance_violation, weight=-2.0,
                            params={"rope_length": 0.8, "ratio": 0.40, "softness": 0.02})

    # 【朝向】惩罚偏离**初始 yaw** 的角度平方（rad²）。取相对值而非世界系 0：
    # 初始朝向含 ±0.03 rad 随机化，用绝对基准会把该偏移当成初始误差。
    # 存在原因：原奖励只惩罚 yaw **角速度**误差，匀速自转在 settle 段几乎不受罚
    # （角速度也≈0），导致"转着不动"成了不受罚的局部最优（用户报告"开始就在自转"）。
    # 量级：偏 11° ⇒ −0.074/步、29° ⇒ −0.51、90° ⇒ −4.9（刻意强于其余惩罚）。
    yaw_heading = RewTerm(func=mdp.yaw_heading_l2, weight=-2.0)

    # ============================ 停止阶段（仅 t ≥ t_stop 生效）============================

    # 【停车后卸力】‖F_tow‖/(‖F_tow‖+10)，有界归一化。表达"停车后应松绳"，
    # 与 extra_distance 配对，防止"为卸载拉力而继续前冲"或"停死后被追尾"两种极端。
    # 只在正式 STOP 之后计算；无小车环境屏蔽。实测每步约 −0.0045（绳力本就很小的必然结果）。
    stop_towing_force = RewTerm(func=mdp.post_stop_towing_force, weight=-1.0,
                                params={"force_scale": 10.0})

    # 【停车后额外位移】relu(x − x_stop)：越过停车点的距离。惩罚"停不住继续滑"。
    # 实测每步约 −0.003。
    extra_distance = RewTerm(func=mdp.post_stop_distance, weight=-0.1)

    # ============================ 动作平滑／幅值（治抖动）============================

    # 【动作变化率】‖a_t − a_{t−1}‖²，鼓励平滑。2026-09-23 由 −0.02 提到 −0.1：
    # 原值实测每步仅 −0.026，被跟踪项压住、基本没起作用。±1 抖动时现约 −0.24/步。
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)

    # 【动作幅值】‖a‖²（裁剪后，≤3）。与 action_rate 互补：一个压"变化量"、一个压"绝对值"。
    # 目的：策略长期输出 ±1 饱和随机方波时把它压回中间值，为 reference_tracking 留出
    # "用中等动作稳定跟踪"的空间。饱和时每步约 −0.15。
    # ⚠ 与"必须用饱和正向动作长时间加速"存在张力（+x 上限仅 0.5 m/s²，到 1.0 m/s 需 40 步饱和），
    #   若发现策略加速不足应下调。
    action_magnitude = RewTerm(func=mdp.action_magnitude_l2, weight=-0.05)

    # ============================ 诊断项（不塑造策略）============================

    # 只为把"验收时答不出的量"写进 TensorBoard，不参与策略优化。
    # **不能用 weight=0**：`RewardManager.compute()` 对 `weight == 0.0` 的项直接 `continue`，
    # 既不调用函数也不更新 `_episode_sums`，连日志都不会产生（2026-09-22 查源码确认）。
    # 用 1e-6 的极小权重：每步贡献 1e-6，重项（tracking 1.0、collision −50）完全占优。
    # ⚠ 因此这些项在 TensorBoard 里的值需要 ÷1e-6 才是物理量。
    obs_cart_present = RewTerm(func=mdp.cart_present_flag, weight=1.0e-6)
    obs_towing_force = RewTerm(func=mdp.towing_force_norm, weight=1.0e-6)
    obs_towing_force_active = RewTerm(func=mdp.towing_force_norm_active, weight=1.0e-6)


@configclass
class UpperTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    robot_fall = DoneTerm(func=mdp.robot_fall, params={"minimum_height": 0.18})
    cart_collision = DoneTerm(func=mdp.cart_collision)


@configclass
class UpperEventsCfg:
    reset_work_condition = EventTerm(
        func=mdp.reset_towing_episode,
        mode="reset",
        params={
            "speed_range": (0.2, 1.0),
            "stop_time_range": (4.0, 6.0),
            "mass_range": (5.0, 15.0),
            "friction_range": (0.4, 1.2),
            "wheel_damping_range": (0.008, 0.032),
            "robot_x_range": (-0.03, 0.03),
            "robot_y_range": (-0.02, 0.02),
            "robot_yaw_range": (-0.03, 0.03),
            # About 32 of 256 environments provide a zero-load anchor each reset batch.
            "no_cart_fraction": 0.125,
            "no_cart_lateral_offset": 2.0,
        },
    )


@configclass
class UpperTowingEnvCfg(ManagerBasedRLEnvCfg):
    scene: UpperTowingSceneCfg = UpperTowingSceneCfg(num_envs=256, env_spacing=6.0)
    observations: UpperObservationsCfg = UpperObservationsCfg()
    actions: UpperActionsCfg = UpperActionsCfg()
    rewards: UpperRewardsCfg = UpperRewardsCfg()
    terminations: UpperTerminationsCfg = UpperTerminationsCfg()
    events: UpperEventsCfg = UpperEventsCfg()
    commands = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 10              # upper policy: 0.005 * 10 = 0.05 s = 20 Hz
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.viewer.eye = (4.0, 4.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.2)
