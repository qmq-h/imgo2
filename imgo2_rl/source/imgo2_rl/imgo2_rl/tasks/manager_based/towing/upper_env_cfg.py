"""Class-based configuration for the hierarchical towing RL environment.

架构（2026-10-10 起，**双头**）：送给冻结 AMP 底层策略的速度指令 = 脚本调度 + 上层
**1 维 vx 偏移**（``loco_command``）；上层网络另有 **12 维关节位置残差**叠加在冻结策略的
关节目标上。两个命令量**解耦**（防作弊红线）：

    task_command —— 脚本调度、到点归零 —— **只给奖励**（`tracking_velocity` 等）；
    loco_command —— task_command + 有界偏移 —— **给冻结策略 + 进 actor 帧**。

回合结构（2026-10-09 恢复三段制）：settle（指令 0，静止稳定）→ tow（指令 = tow_speed）
→ **STOP**（走到 `stop_distance_m`，**必须已越过坡**，指令归零）→ 继续跑到 timeout，
让 `stop_towing_force`／`extra_distance` 有真实相位可用。任务已注册为
``Imgo2-towing-upper-rl-lab``（见同目录 ``__init__.py``）；运行级验收仍未完成，
清单见 ``docs/towing_training_prep_2026-09-22.md``。

契约 v3（2026-10-10）：policy 帧 **58** = 57 + 1（`last_action` 12 → 13）、actor **80**、
critic **73**、decoder ``frame_dim`` **58**；动作 **13** = 1 + 12。
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
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass

from imgo2_rl.assets.cart import make_cart_cfg
from imgo2_rl.assets.imgo2 import IMGO2_CFG
import imgo2_rl.tasks.manager_based.towing.upper_mdp as mdp
from imgo2_rl.tasks.manager_based.towing.mdp.connection_grid import COLUMNS, ROWS
from imgo2_rl.tasks.manager_based.towing.mdp.episode_geometry import (
    POST_STOP_WINDOW_S, SPEED_RANGE, episode_timeout_s)
from imgo2_rl.tasks.manager_based.towing.mdp.slope_geometry import (
    BOUNDARY_MARGIN_M, FLAT_OUT_START_M, FORWARD_M, MAX_GRADE_DEG, profile_arc_length)
from imgo2_rl.tasks.manager_based.towing.utils.policy_cfg import get_policy
from imgo2_rl.tasks.manager_based.towing.slope_terrain import (
    TowingSlopeTerrainGenerator, TowingSlopeTerrainImporter,
)


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
    # Real mesh: 20 flat columns, 10 at |5 deg| and 10 at |10 deg|.
    # Exact origins come from the importer, not GridCloner's square arrangement.
    terrain = TerrainImporterCfg(
        prim_path="/World/Ground", terrain_type="generator",
        class_type=TowingSlopeTerrainImporter,
        terrain_generator=TerrainGeneratorCfg(
            class_type=TowingSlopeTerrainGenerator, size=(20.0, 6.0),
            num_rows=ROWS, num_cols=COLUMNS, sub_terrains={}, curriculum=False),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8, dynamic_friction=0.8, restitution=0.0,
            friction_combine_mode="average", restitution_combine_mode="min"))
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

    # 足端接触：供 `mdp.feet_slide`（支撑脚不许打滑）判着地。与 PPO rough 的写法一致，
    # 传感器自己的 `prim_path` 可匹配多个 prim（本模型 4 个 `FL/FR/RL/RR_FOOT`）；
    # 受「每项只解析一个 prim」约束的是 `filter_prim_paths_expr`，这里**不需要** filter
    # （用合力阈值判着地）。`history_length=3` 与 Isaac Lab 标准设置相同。
    foot_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_FOOT", update_period=0.0, history_length=3,
        # `track_air_time` 与 AMP 的 `contact_forces` 一致：终端要打「最近一次滞空时长」与
        # 「当前腾空足比例」（步频的观测口），`feet_slide` 只用 `net_forces_w_history`。
        track_air_time=True)


@configclass
class UpperActionsCfg:
    """13 维动作（2026-10-10 起）= 1 维 vx 偏移 + 12 维归一化关节残差。

    残差尺度取冻结策略契约的 `action_scale`，所以这里不再有 `acceleration_*`／
    `reference_*` 两组限制：送给冻结策略的速度指令由脚本调度产生，不由网络积分。
    偏移头（cmd vel 头）的三个尺度与两个「停机之后」开关都在
    `mdp.HierarchicalVelocityActionCfg` 上（默认值 = 用户 2026-10-10 建议值，
    `stop_command_ramp_s` / `post_stop_allowance_m` 默认 0.0 = 今天的行为）。
    """

    high_level_velocity = mdp.HierarchicalVelocityActionCfg(
        asset_name="robot", cart_asset_name="cart", policy_name="amp",
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

    **设计取向（截至 2026-10-10）**：回合是三段制（settle → tow → STOP/滑行），所以奖励按
    相位分四组——跟踪类只有 `tracking_velocity`（实测速度 vs **任务指令 `task_command`**
    ——脚本调度、不含策略偏移，防止策略自己给自己发目标；正向驱动，全程生效）；
    抖动/可控性类三项（`action_magnitude`／`action_rate` 罚**残差本身**，
    `low_level_pos_error` 罚**实际离底层期望多远**：冻结策略自己输出的关节位置 vs 实测（参考量不含残差；残差 δ 是补偿这项误差的执行器））；
    拉力方向一项（`towing_force_y`：机体系 y 分量占比，要求拉力落在矢状面内）；
    足端一项（`feet_slide`：着地脚的滑动速度，与 PPO rough 同式同权重）；
    停止类两项（`stop_towing_force`／`extra_distance`，只在指令归零之后生效）。
    **横向与朝向不在这里**：2026-10-09 用户决定用 PD 外环负责，`tracking_velocity` 也只算纵向 `vx`。
    原先与之并列的 `reference_tracking`（‖ref − user‖²）已删除：动作改成关节位置残差后不再有
    `reference_command`，该式恒为 0；其"起步即有效、全域有梯度"的作用由
    `tracking_velocity` 现在真的比较**实测速度**来承担。
    """

    # ============================ 跟踪（任务主目标）============================

    # 【实际速度 vs 命令期望】唯一的正奖励，是策略的主要驱动。
    # exp(−(Δlin/0.5)² − (Δyaw/1.0)²)：完全跟上给 1.0，误差到 0.5 m/s 掉到 0.37、到 1.0 掉到 0.018。
    # 全程生效（settle 段指令为 0，即"保持零速"）。
    # 2026-10-08 修正：线性项原先误用 `reference_command − user_command`（上层自己的积分指令
    # vs 任务指令），与函数名和奖励文档都不符，且与 reference_tracking 重复；现在比实测
    # 机体系线速度，口径与测量台的 `steady_tracking_ratio` 一致。
    # ⚠ 2026-10-10 防作弊红线：参考量是 **`task_command`**（脚本调度），**不是**
    #   `loco_command`（含策略自己的 1 维 vx 偏移）。若拿后者当参考，策略只要把偏移开大
    #   就能自己给自己发目标。有 AST 守卫（`test_reward_terms_never_read_loco_command`）。
    # ⚠ 已知缺陷：误差 >1 m/s 后梯度趋零（exp 的尾部），而牵引段起步瞬间正落在该区，
    #   属"探索与奖励脱钩"的来源之一，尚未修改。
    tracking_velocity = RewTerm(func=mdp.velocity_tracking_exp, weight=1.0,
                                params={"linear_std": 0.5, "yaw_std": 1.0,
                                        "use_lateral_and_heading": False})

    # ============================ 终止级（稀疏、致命）============================

    # 【撞车】车斗／四轮与机器人任一刚体的过滤接触力 > 1 N 时置 1，同时是终止条件。
    # 权重虽大（−50 ⇒ 每步 −2.5），但只在真撞上时才给，属稀疏信号。
    # 背景：修复 filter 通配前该判据恒假（碰撞完全不生效），详见 docs/towing_observability_2026-09-22.md。
    # 【碰撞】小车（车斗/四轮）与机器人 17 个 link 的接触力 > 1 N 即触发（判据见 `update_safety_state`）。
    # 2026-10-09 用户要求：**不再作为终止条件**，只保留惩罚，且幅度调小。
    # 标定逻辑：以前 −50 是"一次性"的（触发即终止、回合重置），每步 −2.5 只作用一步；
    # 现在碰撞可以持续整个回合 ⇒ 它是**持续惩罚**，必须小一个量级：
    # 每步 r = −5.0×0.05 = −0.25，约为 `tracking_velocity` 满额（+0.05/步）的 5 倍；
    # 顶住 1 s（20 步）约 −5，与一个回合的正奖励（≈+6）同量级 ⇒ 有动机脱离但不至于
    # 让早期学习被"一碰就废"支配。若日志里该项长期压过跟踪项再下调。
    collision = RewTerm(func=mdp.cart_collision_cost, weight=-5.0)

    # 【跌倒】base 高度 < 0.18 m 时置 1，同时是终止条件。实测从未触发。
    fall = RewTerm(func=mdp.robot_fall_cost, weight=-50.0,
                   params={"minimum_height": 0.18})

    # ============================ 几何／安全约束 ============================

    # 【软间隙障碍】softplus((0.20 − 间隙)/0.05)，间隙趋向 0 时加大。
    # 0.20 m 是**绝对**警戒线，线性区在警戒线**下方**；间隙 0.2 时约 −0.69/步、0 时约 −0.2/步。
    # 无小车环境用 cart_present 屏蔽。
    clearance = RewTerm(func=mdp.clearance_barrier, weight=-1.0,
                        params={"warning_distance": 0.20, "scale": 0.05})
    # 【硬最小间距】间隙低于 `ratio × 本 env 连接长度` 时**与缺口成正比**地惩罚，高于则**精确为 0**。
    # 与上面 clearance 的分工：clearance 是"近了要缓"的软障碍；本项是"不得拉太近"的硬约束，
    # 且按**连接长度比例**给出阈值（不是绝对量）。weight 的单位是"每米缺口扣多少"。
    # **ratio=0.25（2026-10-08 随 20 行长度网格由 0.40 下调）**：阈值逐 env 用
    # `term.connection_length`，不再是单一 `rope_length`。spawn 间隙 =
    # sqrt((0.5·L0)² − 0.17²) + 0.0025，最短行 L0=0.6 时约 0.250 m > 0.25×0.6=0.15 m，
    # 故**网格所有行 spawn 都精确为 0**（用户 2026-09-23 要求初始不生效）。
    min_clearance = RewTerm(func=mdp.min_clearance_violation, weight=-2.0,
                            params={"ratio": 0.25, "softness": 0.02})

    # 【朝向惩罚：2026-10-09 已关闭】横向与朝向改由 **PD 外环**负责（`upper_mdp` 的
    # `_lane_keeping_vy/_wz` 写 `loco_command[:,1:3]`，用户决定），所以这里**不再注册**
    # `yaw_heading = RewTerm(func=mdp.yaw_heading_l2, ...)`：PD 已经是逐拍闭环，再叠一个
    # 奖励等于同一目标约束两次，还会把"朝向误差"混进策略的回报、掩盖 PD 的效果。
    # 同时 `tracking_velocity` 也改成只用纵向 `vx`（见其 params 的 `use_lateral_and_heading=False`），
    # 横向/朝向的跟踪误差不再进回报。
    # helper 仍保留在 `upper_mdp`（`heading_deviation`／`yaw_heading_l2`），要回开就把下面一行
    # 取消注释；注意回开后与 PD 的基准相差 ≤0.03 rad（PD 目标是 0，helper 基准是 spawn 朝向）。
    # yaw_heading = RewTerm(func=mdp.yaw_heading_l2, weight=-2.0)

    # ============================ 停止阶段（仅 t ≥ t_stop 生效）============================

    # 【停车后卸力】`‖F‖/(‖F‖+10)`，有界归一化。表达"停车后应松绳"：别一直拽着、
    # 也别把小车当锚。门控是 `elapsed_s >= stop_time_s`；`stop_time_s` 在进度越过
    # `stop_distance_m`（**已越过坡**）那一拍写入，未停车前是 `+inf` ⇒ 本项精确为 0。
    # 与 `extra_distance` 配对，防"为卸载拉力而继续前冲"与"停死后被追尾"两种极端。
    stop_towing_force = RewTerm(func=mdp.post_stop_towing_force, weight=-1.0,
                                params={"force_scale": 10.0})

    # 【停车后额外位移】`relu(x − x_stop − allowance)`：越过停车点的距离，罚"停不住继续滑"。
    # 基线 `stop_origin_x` 是指令归零那一拍的 x，所以不含牵引段的前进。
    # `post_stop_allowance_m`（2026-10-10 新增，**默认 0.0 = 今天的行为**，公式退化为
    # `relu(x − x_stop)`）：要让策略学会「STOP 后按车重/坡度再走两步」就把 `extra_distance`
    # 的这个参数与 `actions.high_level_velocity.stop_command_ramp_s` 一起改成 0.5–1.0 m /
    # 1.0–2.0 s，否则本项（−0.1）与新增的 1 维 vx 偏移头对打 —— 偏移头一让机器人前进，
    # 本项立刻扣分，策略学到的最优解是"永远不碰偏移头"。
    extra_distance = RewTerm(func=mdp.post_stop_distance, weight=-0.1,
                             params={"post_stop_allowance_m": 0.0})

    # ============================ 拉力方向（机体系 y 分量为 0）============================

    # 【拉力方向偏离矢状面】`(F_y/‖F‖)² = sin²(偏离角)`，用户 2026-10-09 要求
    # 「运动过程中 3 维拉力在 y 维度保持为 0」。用**比值**而非原始 `F_y²`：与张力大小无关、
    # 有界 [0,1]，避免被起步绷直的百牛级峰值放大成"变相惩罚大张力"。
    # 量级：偏 10° → 0.030 ⇒ −10×0.030×0.05 = **−0.015/步**；偏 20° → −0.059/步。
    # ⚠ 注意它只保证"绳在机体矢状面内（小车正后方）"，**不等于**沿车道中线/朝向对齐：
    #   机器人整体偏航但小车也跟着偏到正后方时，本项仍可 ≈0。中线/朝向由 **PD 外环**负责
    #   （`_lane_keeping_vy/_wz`），不是奖励项。actor 目前**观测不到** F_y，只能靠 GRU 历史间接学；
    #   要真正闭环修正需把 F_y（或比值）加进 actor 帧，见 README 问题表 TOW-14。
    towing_force_y = RewTerm(func=mdp.towing_force_y_ratio_sq, weight=-10.0)

    # ============================ 支撑脚不许打滑 ============================

    # 【足端打滑】Σ_feet ‖v_foot,xy‖·1(着地)，着地判据 = 接触合力**历史最大值** > 1 N。
    # 公式与权重都与 PPO rough 一致（`feet_slide`，−0.05；足端正则 `.*_FOOT`），本仓库实现
    # 在 `upper_mdp.feet_slide`（按传感器 `body_names` 映射到资产 id，避免"两处同序"假设）。
    # 用户 2026-10-09 认可加入，用来回答「负载下速度跟踪变差、而角度跟踪没变差」是否由打滑
    # 贡献——这一项自己的 TB 曲线就是该问题的测量（着地脚静止时应 ≈0）。
    # ⚠ 它天然可被"走慢"降低，与 `tracking_velocity` 对冲；若日志里它压过跟踪项再下调。
    # 量级：单脚打滑 0.2 m/s × 2 只着地脚 ⇒ func=0.4 ⇒ 每步 −0.05×0.4×0.05 = −0.001
    # （约为 `tracking_velocity` 满额的 2%）。**权重沿用 PPO 值，待实跑标定。**
    feet_slide = RewTerm(func=mdp.feet_slide, weight=-0.05,
                         params={"sensor_name": "foot_contacts"})

    # ============================ 动作平滑／幅值（治抖动）============================

    # 【动作变化率】‖a_t − a_{t−1}‖²，鼓励平滑。2026-09-23 由 −0.02 提到 −0.1：
    # 原值实测每步仅 −0.026，被跟踪项压住、基本没起作用。±1 抖动时现约 −0.24/步。
    # ⚠ 2026-10-08 动作由 3 维变 12 维关节残差，同等逐维幅值下本项量级约 ×4，
    #   权重是否仍合适待实跑确认（见 README 问题表）。
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)

    # 【动作幅值】‖a‖²（裁剪后，3 维时 ≤3、12 维时 ≤12）。与 action_rate 互补：
    # 一个压"变化量"、一个压"绝对值"。目的是把残差压回小幅度、让冻结步态保持主导。
    # ⚠ 与"必须用饱和正向动作长时间加速"的旧张力已随动作语义变化：现在饱和的是**关节
    #   残差**，不再直接等于加速度上限。权重 −0.05 待实跑确认。
    action_magnitude = RewTerm(func=mdp.action_magnitude_l2, weight=-0.05)

    # ============================ 底层可控性（残差的后果）============================

    # 【底层跟踪误差】**底层期望 vs 实际**：参考量是冻结策略自己输出的关节位置（稳定步态），
    # 不是叠加残差后的下发值——用户判据「底层的期望值就是稳定步态的，残差是用来稳定角度跟踪的，
    # 所以奖励就应该是期望和实际的比较，这之中是根本不需要残差的」。
    # 数学上 `q_act − q_exp = δ + e_PD`：残差 δ 会出现在差里，但这不是缺陷而是**机制**——
    # 底层因负载下垂/坡度/接触产生系统性误差时，δ ≈ −e_PD 能把实测拉回期望，δ 就是执行器。
    # 与 `action_magnitude`/`action_rate`（罚 δ 本身）方向相反，三者权重需一起看。
    # 量级（2026-10-09 冒烟，64 环境 20 轮）：Σ≈0.34 rad²（每关节 RMS 0.17 rad；反推约 0.12 来自
    # 残差、0.22 来自 PD 误差）。权重演化：初值 −5.0 实测 −0.086/步 ≈ `tracking_velocity` 的 2 倍、
    # 主导整个回报 ⇒ 用户 2026-10-09 定为 **−0.1**（每步 ≈ −0.0017）。
    # ⚠ 若按"主目标"理解，−0.1 偏小：它只值跟踪项的约 4%，可能不足以驱动 δ 去补偿负载下垂；
    #   而 −5.0 又会主导回报。建议区间 −0.5 ~ −1.0，待用户定/实跑标定。
    # 用户 2026-10-09 定为 `reference="commanded"`：期望取**含残差**的下发目标，即
    # "底层的期望 = 底层输出 + 残差"，误差 = 纯 PD 跟踪误差；要换成不含残差的底层原始
    # 期望，把参数改成 "frozen" 即可（两条分支都有测试守卫）。
    low_level_pos_error = RewTerm(func=mdp.low_level_position_error_l2, weight=-0.1,
                                  params={"reference": "commanded"})

    # ============================ 诊断项（不塑造策略）============================

    # 只为把"验收时答不出的量"写进 TensorBoard，不参与策略优化。
    # **不能用 weight=0**：`RewardManager.compute()` 对 `weight == 0.0` 的项直接 `continue`，
    # 既不调用函数也不更新 `_episode_sums`，连日志都不会产生（2026-09-22 查源码确认）。
    # 用 1e-6 的极小权重：每步贡献 1e-6，重项（tracking 1.0、collision −50）完全占优。
    # ⚠ 因此这些项在 TensorBoard 里的值需要 ÷1e-6 才是物理量。
    obs_cart_present = RewTerm(func=mdp.cart_present_flag, weight=1.0e-6)
    obs_towing_force = RewTerm(func=mdp.towing_force_norm, weight=1.0e-6)
    obs_towing_force_active = RewTerm(func=mdp.towing_force_norm_active, weight=1.0e-6)
    # 【是否走到 STOP 点】0/1 粘性标志：回合一律由 timeout 收尾，所以"走没走到 STOP 点"
    # 只能靠这个诊断量回答，而不是靠终止原因。读法：`Episode_Reward/obs_stop_reached ÷ 1e-6`
    # = 处于 STOP 相位的步数占比（时间占比），**> 0 就是走到了**；恒 0 表示没走到
    # （那些回合全部由 `time_out` 收尾）。
    obs_stop_reached = RewTerm(func=mdp.stop_reached_flag, weight=1.0e-6)


@configclass
class UpperTerminationsCfg:
    """终止项（2026-10-09 用户确认：**正常回合一律由 timeout 收尾**）。

    到达 `stop_distance_m` 只把指令置零、进入 STOP 段，**不终止**（否则两条停车奖励没有相位）。
    因此没有 `goal_reached`／`stop_reached`／`post_stop_timeout` 终止项；"有没有走到 STOP 点"
    由诊断量 `obs_stop_reached` 回答。剩下的**只剩两条失败退出 + 超时**：
    `robot_fall`（离局部坡面高度 < 0.18 m ⇒ 倒地退出）、`terrain_exit`，以及 `time_out`。
    **`cart_collision` 自 2026-10-09 起不再是终止项**（用户要求改成惩罚），碰撞只由奖励项
    `collision`（−5.0）表达；`mdp.cart_collision` 函数保留（奖励与日志在用）。
    实测这样机器人**基本不避讳碰撞**（接触占用 0.39%/步、约 2.25 步/回合，而旧终止配置的
    危险率只有 0.082%/步、威慑力约 18 倍）——见 README TOW-19。
    """
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    robot_fall = DoneTerm(func=mdp.robot_fall, params={"minimum_height": 0.18})
    terrain_exit = DoneTerm(func=mdp.terrain_out_of_bounds)


@configclass
class UpperEventsCfg:
    reset_work_condition = EventTerm(
        func=mdp.reset_towing_episode,
        mode="reset",
        params={
            "speed_range": SPEED_RANGE,
            # 2026-10-09 用户要求上限提到 30 kg（原 5–15）：更接近"重载"工况。
            # 质量的**惯量按同一比例**缩放（见 `reset_towing_episode`），所以转动惯量自洽；
            # 注意轮轴阻尼范围 (0.008, 0.032) 未随质量缩放 ⇒ 30 kg 时"单位质量的滚动阻力"
            # 比 5 kg 小，这是刻意保留的建模选择（与测量台同一套参数）。
            "mass_range": (5.0, 30.0),
            "friction_range": (0.4, 1.2),
            "wheel_damping_range": (0.008, 0.032),
            # 出生**固定**（用户 2026-10-09）：x/y/yaw 三个抖动全部归零 ⇒ 每个 env 的出生
            # 位姿逐回合完全一致。保留参数形状（而不是删掉）是为了随时能调回去。
            "robot_x_range": (0.0, 0.0),
            "robot_y_range": (0.0, 0.0),
            "robot_yaw_range": (0.0, 0.0),
            # 约 12.5% 环境作为零负载锚点（0.125 × 800 = 100 个）。注意它是**随机**子集，
            # 会打破网格的 8/8/4 平衡——这是用户明确要求保留的域随机化。
            "no_cart_fraction": 0.125,
            "no_cart_lateral_offset": 2.0,
        },
    )


def _policy_joint_names() -> list[str]:
    """冻结 AMP 策略契约里的 12 个关节名（域随机化按它们作用，与 AMP 训练环境一致）。"""
    return list(get_policy("amp").joint_names)


@configclass
class UpperRobotDomainRandCfg:
    """**机器人侧**域随机化：参数逐项照抄 AMP vel 跟踪任务。

    来源：`locomotion/velocity/velocity_env_cfg.py` 的 `EventCfg`（基类范围）+
    `base_move/amp_env_cfg.py` 的 `__post_init__`（Imgo2 AMP 的覆盖：作用体/关节收窄）。对齐
    这几项是为了让**冻结底层策略处于它训练时见过的分布**（它是在这套 DR 下训出来的），
    同时给上层的 sim2real 留出机器人参数偏差。

    | 项 | 模式 | 作用对象 | 参数（与 AMP 相同） |
    |---|---|---|---|
    | 质量（base） | startup | `base` | **加** (−1, 3) kg，`recompute_inertia=True` |
    | 质量（其余） | startup | 除 base 外全部 | **乘** (0.7, 1.3)，`recompute_inertia=True` |
    | 质心 | startup | `base` | x/y/z 各 ±0.05 m |
    | 执行器增益 | reset | 12 个策略关节 | stiffness/damping **乘** (0.5, 2.0)，uniform |

    **刻意不恢复的两项（照抄会变成空操作，甚至误导）**：

    - `randomize_rigid_body_material`（AMP 是 startup、机器人全体、static 0.3–1.0 / dynamic
      0.3–0.8 / restitution 0–0.5）：本任务的 `reset_towing_episode` **每回合**都会把 robot 与
      cart 的 material 摩擦/恢复系数整体改写（`friction_range` 采样值），startup 的随机化在
      第一个回合就被覆盖 ⇒ 恒等于没加。
    - `randomize_apply_external_force_torque`（AMP 是 reset、作用 base、力/力矩 ±10）：
      本任务的取绳物理 `_apply_towing_physics` **每个物理步**都调用
      `set_external_force_and_torque(..., body_ids=[base])`，把 base 的外力通道整个覆盖 ⇒
      reset 时写进去的随机力在 5 ms 后就被冲掉，同样恒等于没加。要真正恢复它，必须把这份
      随机力**加进取绳物理的合力**里（另一处改动），是否要做待用户定。
    """
    randomize_robot_mass_base = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    randomize_robot_mass_others = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["^(?!.*base).*"]),
            "mass_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    randomize_robot_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    randomize_robot_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=_policy_joint_names()),
            "stiffness_distribution_params": (0.5, 2.0),
            "damping_distribution_params": (0.5, 2.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class UpperTowingEnvCfg(ManagerBasedRLEnvCfg):
    # 40 columns x 20 lengths (0.6--1.2 m) = 800 envs. 每条 lane 的剖面都是
    # 「平地 2.25 m → 上坡 3 m → 坡顶 0.75 m → 下坡 3 m → 平地 2.25 m」（`mdp/slope_geometry.py`，
    # 跑道总长 15 m）；
    # 列决定坡度量级：前 20 列是纯平地，之后 10 列 5°、10 列 10°（见 `connection_grid.py`）。
    # A non-multiple of 800 only covers a grid prefix, not all work conditions.
    scene: UpperTowingSceneCfg = UpperTowingSceneCfg(
        num_envs=COLUMNS * ROWS, env_spacing=6.0)
    observations: UpperObservationsCfg = UpperObservationsCfg()
    actions: UpperActionsCfg = UpperActionsCfg()
    rewards: UpperRewardsCfg = UpperRewardsCfg()
    terminations: UpperTerminationsCfg = UpperTerminationsCfg()
    events: UpperEventsCfg = UpperEventsCfg()
    robot_domain_rand: UpperRobotDomainRandCfg = UpperRobotDomainRandCfg()
    commands = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 10              # upper policy: 0.005 * 10 = 0.05 s = 20 Hz
        action = self.actions.high_level_velocity
        if action.stop_distance_m >= FORWARD_M - BOUNDARY_MARGIN_M - 0.1:
            raise ValueError("stop distance exceeds the safe lane length")
        # 用户要求「给 cmd vel 一定要在越过坡之后」：STOP 触发点必须落在坡面**出口平地**上。
        # 坡面出口 = `FLAT_OUT_START_M`（9.0 m），触发点 10.0 m 满足；写成断言防止以后调小。
        if action.stop_distance_m <= FLAT_OUT_START_M:
            raise ValueError(
                f"STOP 触发点 {action.stop_distance_m} m 必须越过坡面出口 "
                f"{FLAT_OUT_START_M} m，否则停车段落在坡上/坡中")
        minimum_speed = self.events.reset_work_condition.params["speed_range"][0]
        # 速度头把 `task_vx + 有界偏移` 裁进冻结 AMP 策略的**训练包络**（−1.0, 1.5，
        # 出处 `amp_env_cfg` 的 `commands.base_velocity.ranges.lin_vel_x`）。该包络必须
        # **覆盖整个脚本速度范围**：否则零偏移时脚本自己就被裁掉，退化性（偏移=0 时与旧口径
        # 逐位一致）直接失效，而且奖励参考量 `task_command` 与实际送策略的指令会分叉。
        amp_vx_min, amp_vx_max = action.amp_vx_range
        speed_min, speed_max = self.events.reset_work_condition.params["speed_range"]
        if not (amp_vx_min <= speed_min and speed_max <= amp_vx_max):
            raise ValueError(
                f"amp_vx_range {(amp_vx_min, amp_vx_max)} 必须覆盖脚本速度范围 "
                f"{(speed_min, speed_max)}，否则零偏移时脚本指令会被裁掉")
        # 正常回合一律由 `time_out` 收尾（用户 2026-10-09 确认）：timeout = settle +
        # （到 STOP 点的**坡面弧长**上界）/ 最小速度 + 停车窗口。弧长上界取最陡档：
        # 10 m 水平在 10° 剖面上是 `profile_arc_length(10, 10)=10.093 m`
        # ⇒ 1 + 10.093/0.4 + 3 = 29.23 s。`POST_STOP_WINDOW_S` 是最慢速度下的停车窗口；
        # 速度越快 STOP 越早、尾巴越长（已与用户确认接受；"有没有走到 STOP 点"看
        # `obs_stop_reached` 诊断量，不靠终止原因区分）。
        surface_distance = profile_arc_length(MAX_GRADE_DEG, action.stop_distance_m)
        self.episode_length_s = episode_timeout_s(
            surface_distance, minimum_speed, action.tow_start_s,
            margin=POST_STOP_WINDOW_S)
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.viewer.eye = (4.0, 4.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.2)
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0
