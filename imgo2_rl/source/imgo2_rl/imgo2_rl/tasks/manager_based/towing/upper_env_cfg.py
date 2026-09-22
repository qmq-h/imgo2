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
    # Minimal v0 reward. Keep additional diagnostics out of the return until an observed failure
    # justifies adding a term.
    tracking_velocity = RewTerm(func=mdp.velocity_tracking_exp, weight=1.0,
                                params={"linear_std": 0.5, "yaw_std": 1.0})
    collision = RewTerm(func=mdp.cart_collision_cost, weight=-50.0)
    fall = RewTerm(func=mdp.robot_fall_cost, weight=-50.0,
                   params={"minimum_height": 0.18})
    clearance = RewTerm(func=mdp.clearance_barrier, weight=-1.0,
                        params={"warning_distance": 0.20, "scale": 0.05})
    stop_towing_force = RewTerm(func=mdp.post_stop_towing_force, weight=-1.0,
                                params={"force_scale": 10.0})
    extra_distance = RewTerm(func=mdp.post_stop_distance, weight=-0.1)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)

    # --- 诊断项：只为进 TensorBoard，不用于塑造策略 ---
    # 不能用 weight=0：`RewardManager.compute()` 对 `weight == 0.0` 的项直接 `continue`，
    # 既不调用函数也不更新 `_episode_sums`，于是连日志都不会产生（2026-09-22 查源码确认）。
    # 这里用 1e-6 的极小权重：每步贡献 1e-6，重项（tracking 1.0、collision −50）相对它
    # 完全占优，对总回报与策略梯度的影响可忽略。
    # 其余分项（tracking_velocity/collision/fall/clearance/stop_towing_force/
    # extra_distance/action_rate）由 RewardManager 自动记录为 Episode_* 统计。
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
