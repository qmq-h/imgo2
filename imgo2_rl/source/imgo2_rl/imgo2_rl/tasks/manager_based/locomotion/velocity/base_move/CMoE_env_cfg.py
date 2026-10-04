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
    CMoETrackGapTerrainCfg,
    CMoETrackHurdleTerrainCfg,
    CMoETrackMixTerrainCfg,
    CMoETrackNarrowStairsTerrainCfg,
    CMoETrackStairsTerrainCfg,
    CMoETrackStepTerrainCfg,
)


# 2026-09-28（用户决定）：**步态/姿态 shaping 的地形分区**。
# * `EASY_TERRAIN_NAMES`（无障碍）：平地、两种斜坡、普通粗糙 —— trot 与"机身水平"在这里既自然又省力；
# * `OBSTACLE_TERRAIN_NAMES`（障碍）：其余 7 类 —— 允许策略按地形自由选择步态与姿态（过沟/上箱需要俯仰）。
# ⚠️ 两张表必须**互斥且并集 = 全部 `sub_terrains` 键**；`tests/test_reward_terrain_partition.py` 会锁住这一点，
# 以后新增地形忘了归类就会红（历史上 `forward_only_terrain_names` 漏项曾静默退回全向命令）。
# `flat_orientation_l2`（机身水平罚）**只在这些地形上要求"机身保持水平"**（2026-09-28 用户：
# "斜坡和台阶都不需要保持水平"）：坡面要**贴坡**、台阶要**抬头/低头**、障碍要**俯仰** ⇒ 都不该被水平罚按住。
# 其余地形（含两种斜坡、两种台阶、boxes/gap/hurdle/mix/narrow_stairs）**自动豁免** —— 用"全部地形键减去本表"
# 算出来（见 `__post_init__`），这样以后新增地形默认**不受**水平约束（安全默认：不赌"新地形是平的"）。
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

# 2026-09-30（姿态退化修复 ④）：非足端**高力接触**终止项的**一行开关**。
# 关掉的方式（任选其一，都只影响本项）：
#   ① `ENABLE_ILLEGAL_CONTACT_BODY_TERMINATION = False`
#   ② 把 `__post_init__` 里那段 `if ENABLE_ILLEGAL_CONTACT_BODY_TERMINATION:` 整段注释掉。
# ⚠️ 这是四项改动里**最容易伤到 gap/stairs 的一项**：跨沟/上台阶时小腿/膝的**轻擦**是常见且必要的，
# 只有"称重跪地"（瞬时接触力 ≥ 50 N）才该终止。阈值因此取 **50 N**（远高于既有的基座触地终止
# 1 N，也高于 `undesired_contacts` 的判据阈值 1 N）——即"罚得动、但不到称重就不终止"。
# 2026-09-30 实测（run `cmoe_v5_8_posture`）：**50 N 阈值不可用** —— `Episode_Termination/
# illegal_contact_body = 1.0000`（100% 回合终止都是它）、回合长度掉到 **6.3 步（0.13 s）**、
# `mean_reward −0.058`、`level_mean 0` ⇒ 策略完全学不动（静止时每足就承重 ~13.5 N，
# 膝/小腿轻碰轻松超 50 N）。**默认关闭**；若将来重启该功能，阈值至少提到 150~250 N
# 并要求"持续接触 N 步"，且必须先离线标定真实接触力量级。
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
        # 2026-09-24：回放发现策略「停在沟前不敢迈」。该判据＝足底射线网格**部分命中**（实心/空洞
        # 交界）且该足接触 —— 正是跨沟必须摆出的「足踩边缘」姿态，等于惩罚过沟动作本身。
        # 先归零做单变量验证；若确认就是它，再考虑改成「整束射线全部落空才罚」的语义。
        weight=0.0,
        params={
            "edge_sensor_names": FOOT_EDGE_SENSOR_NAMES,
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_FOOT"),
            "contact_force_threshold": 1.0,
        },
    )

    # ---------------------------------------------------- 步态**度量**项（几乎不产生奖励，只为记录）
    # 2026-09-24（用户："是不是可以分列统计一下步态指标，在 curriculum 指标中"）：
    # 用同一个 `GaitReward`（6 核乘积）配**三种成对方式**做成一个**逐列步态分类器**：
    #   diagonal（FL↔RR、FR↔RL）= trot ／ left-right（FL↔FR、RL↔RR）= bound ／ same-side（FL↔RL、FR↔RR）= pace
    # 谁的值最高，那一列就是哪种步态；三者都贴近下界 `0.893^6 ≈ 0.508` ⇒ 既不是这三种（pronk／乱走）。
    # ⚠️ 权重必须是**非零**（否则 `disable_zero_weight_rewards()` 会把 term 整个移除、日志就没了），
    # 取 **1e-6**：`≤1e-6 × 1 = 1e-6/s`，相对整回合 ~4/s 完全可以忽略（占比 ~2.5e-7），不影响训练。
    # 这三项**不加地形掩码** —— 正是为了看清 `boxes`/`gap` 上"自己演化"成了什么步态。
    # ⚠️ 2026-09-24 夜：`std`/`max_err` 必须与 `feet_gait`（post_init 里那三项赋值）**保持一致**，
    # 否则分类器的读数不再代表奖励。改动理由见 docs §29.31/§29.32：旧值 `std=√0.5=0.7071` 让
    # 单核地板高达 `exp(−2·0.2²/0.7071)=0.893`（配错也拿 89 分）⇒ 6 核乘积只有 [0.508, 1]，
    # 实测 trot−bound 仅差 0.010（≈0.1 个核）⇒ 相位项的边际奖励只有 0.01/s，PPO 不会理它。
    # 新值让地板降到 `exp(−2·0.5²/0.2)=0.082`，trot/bound 溢价 0.26 → **0.70/s**（2.7×）。
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
    # 弹跳度量：同一个 `lin_vel_z_l2`（机体竖直速度²），权重 1e-6 只为记录 ⇒ 反解后
    # `gait_bounce_<地形>` ＝该列本回合的 **vz 均方**（开方即 vz RMS，单位 m/s）。
    diag_bounce = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=1e-6,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # 接触时序诊断（2026-09-24 夜，用户同意）：两个 1e-6 纯记录项，用来把"步子到底是长是短"和
    # "相位差到底多大"从日志里**直接读出来**，不再靠间接反解猜参数。
    # `gait_airtime_<地形>` ＝该列四足 `last_air_time` 均值（s，＝最近一次完整滞空时长）；
    #   与 `Episode_Reward/feet_air_time` 联立可解出**落地频率**（那个量只约束 `N落地·(ā−0.5)`，
    #   单看它"长步幅慢步"与"短步快蹭"不可分辨 —— 见 docs §29.31.2）。
    # `gait_mismatch_<地形>` ＝该列六对脚 `|Δair|+|Δcon|` 的均值（s）＝**相位核真正面对的时间差尺度**
    #   ⇒ 用来按实测选 `max_err`（若实测只有 0.1 s 级，0.5 就永远够不着地板、项又变回人人高分）。
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
    # 2026-09-30 新增（姿态退化排查的第一手读数，1e-6 纯记录）：`gait_base_height_<地形>`
    # ＝该列基座相对**局部地面**的**有符号**高度误差（m，正=偏高、负=偏矮）。
    # 为什么现有 `gait_height_<地形>`（`base_height_l2` 反解 √(Episode_Reward/10)）不够：
    # 那个量是 `(误差)²` ⇒ **丢符号**，分不出"蹲矮"与"抬高"（两者都让 −10 项变负）。
    # 本项不平方、不乘权重、不乘重力门 ⇒ 可直接读到 −0.070 m 这种"蹲 7 cm"。
    # 代码**逐字取自 `foot_clearance` 分支**（用户 2026-09-30 要求只移植这一项）。
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

    # ------------------------------------------------------------------ parkour 式"全球速度"约束
    # 2026-09-24（用户："该参考 parkour 用全局的速度来约束了"）。起因：回放发现策略**横移绕开障碍**
    # （§29.15/§29.16）。parkour 的 barrier/leap 配方在**世界系**上约束速度，并额外罚横向位置与朝向：
    #   `tracking_world_vel = 5.`、`lin_pos_y = -0.4`、`yaw_abs = -0.2`
    # （`legged_robot_field.py:476/493/496`、`go1_leap_config.py:93-100`）。三项一一对应如下。
    track_world_vel_xy_exp = RewTerm(
        func=mdp.track_world_vel_xy_exp,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),  # 与机体系版本同 σ²（parkour 的 leap 用 0.35，偏软一点）
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    lin_pos_y = RewTerm(
        func=mdp.lin_pos_y,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "terrain_names": (),  # __post_init__ 里填 forward_only_terrain_names
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

    # ---------------------------------------------------------------- ② 平地专用严格高度项
    # 2026-09-30（治"高度没保持住"）：`base_height_l2`（−10）在平地上**不够**——实测
    # run `cmoe_v5_7_lv12cap` @8500 反解高度误差 RMS **0.070 m**（蹲 7 cm），
    # 而该项每步代价只有 −0.0484（× dt 0.02 ≈ −9.7e-4/步），占任务项（+3.71/+1.37）的 ≈1%。
    # 本项 = `base_height_l2_strict`：**同一** `target_height`、**同一** `height_scanner_base`
    # 局部地面逻辑（共用 `_local_ground_target_height`），但**去掉重力门** ——
    # "蹲 + 膝蹭"本身是俯仰/侧倾姿态，门控 `clamp(−g_z,0,0.7)/0.7` 会让它自己给自己打折。
    # 掩码是**白名单** `active_terrain_names`（不是 `free_terrain_names` 豁免名单）：
    # **只在 flat 生效**，障碍地形恒 0（用户："避免惩罚合法的越障姿态"）。
    # ⚠️ `CMoERewardsCfg` 里权重必须留 **0.0**（不影响其它任务）；2026-09-30 曾由
    # `Imgo2CMoERoughEnvCfg.__post_init__` 设成 **−35.0** 启用，**2026-10-01 已还原为 0.0（禁用）**
    # （4000 轮那份存档里没有本项；见 `__post_init__` 内"还原项之四"的说明）。
    # **如需再启用：把 `__post_init__` 里那一行改回 −35 即可**（term 定义与类都保留）。
    # 它**不是**步态形状项，因此 `-gaitfree` 子类**不得**把它归零。
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
    """Keep PPO's 45-D policy and expose the terrain height scan (77-D since 2026-09-24) separately.

    critic 维度 = 3（base 线速度）+ 45（本体感知）+ 地形扫描维数（77，故为 125）。
    """

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

        # 2026-09-24（用户要求）：CMoE 的地形扫描改为**原版的 77 维**（11×7），而不是继承
        # `velocity_env_cfg` 的 187 维（17×11 @ 1.6×1.0 m）。只覆盖 CMoE 自己的场景，
        # 不动 `MySceneCfg.height_scanner`，因此 PPO/HIM/AMP 的观测契约不受影响。
        # 由此得到的新契约：terrain=77、actor 总输入 527（450+77）、expert/gate 157、critic 125。
        # ⚠️ 这与 2026-09-24 之前的 CMoE checkpoint 不兼容（load_state_dict 会尺寸不匹配）；
        # 旧 run 可用各自 `params/CMoE_env_cfg.py` 里保存的那份配置回放。
        scan = self.scene.height_scanner.pattern_cfg
        scan.resolution = 0.1
        scan.size = (1.0, 0.6)  # x: 1.0/0.1+1 = 11，y: 0.6/0.1+1 = 7 ⇒ 77
        # 前瞻：把 11×7 网格整体前移 0.25 m ⇒ x ∈ [−0.25, +0.75]（前瞻 0.75 m / 后视 0.25 m）。
        # 分辨率必须保持 0.1 m：当前课程沟宽约 0.13 m，间距 0.15 m 时沟可能整条落在两条射线之间
        # 而漏检（间距 0.1 m 时任意 0.13 m 区间必含至少一条射线）。z 仍保持 20 m 的射线起点高度。
        # 注意：只动 `height_scanner`（观测用），不动 `height_scanner_base`（base_height_l2 用，必须留在基座正下方）。
        self.scene.height_scanner.offset.pos = (0.25, 0.0, 20.0)
        # 射线计数必须镜像 `grid_pattern` 的 `arange(start, end + 1e-9, step)` 语义：
        # 写成 `int(span/res)+1` 会被浮点截断骗到（0.6/0.1 = 5.999999999999999 ⇒ 6 而不是 7，
        # 2026-09-24 实测报过 "got 11x6=66"），所以这里加同样的 1e-9 容差。
        _rows = math.floor(scan.size[0] / scan.resolution + 1.0e-9) + 1
        _cols = math.floor(scan.size[1] / scan.resolution + 1.0e-9) + 1
        if _rows * _cols != 77:
            raise RuntimeError(
                f"CMoE terrain scan must be 77 rays (11x7), got {_rows}x{_cols}={_rows * _cols} "
                f"(size={tuple(scan.size)}, resolution={scan.resolution})"
            )

        # ---------------------------------------------------------------------------------
        # 地形集：2026-09-28 与参考实现对齐（用户："我看了一下 cmoe 的地形设置，我们四足这个地形
        # 还是差点，感觉可以对齐一下"）。参考 = `Hoshi-No-Ai/CMoE` @ `4575d6ae`：
        #   * 比例／行列数／初始等级 → `legged_gym/legged_gym/envs/g1/g1_cmoe_config.py::terrain`
        #   * 每类难度律           → `legged_gym/legged_gym/utils/humanoid_terrain.py::make_terrain`
        #   * 纵向障碍图案         → `legged_gym/legged_gym/utils/parkour_terrain_utils.py`
        # **只对齐结构**（种类／比例／行列数／难度律形式／初始等级）；米制难度区间沿用我们已验证的
        # 四足值（台阶 0.05–0.15、块 0.08–0.30、沟 0.126–0.315、坡 0–0.4、噪声 0.01–0.06）；
        # 参考里没有四足对应值的三类（hurdle／mix／narrow_stairs）按参考米制 × 站高比 0.4
        # （`cmoe_terrains.REFERENCE_SCALE`）落地。逐项对照、已知偏离（粗糙度叠加未移植、
        # mix 图案平移 0.30 m）见 docs/cmoe_terrain_alignment_2026-09-28.md。
        #
        # 参考比例（G1 CMoE `terrain_dict`）：plane 0.0／rough slope 0.1／stairs up 0.1／
        # stairs down 0.1／discrete 0.1／parkour_gap 0.3／hurdle 0.1／mix 0.1／narrow_stairs 0.1。
        # 键名沿用我们的 + 只新增（2026-09-28 用户决定），所以：rough slope 的 0.10 拆给
        # `hf_pyramid_slope`（上坡）＋`hf_pyramid_slope_inv`（下坡）各 0.05；`random_rough` 0.05
        # 与 `flat` 0.10 是我们自己保留的两类（参考 plane 比例是 0.0）。合计 1.15，Isaac Lab
        # 会先按总和归一化再切列（见 `scripts/tools/check_terrain_columns.py`）。
        # ---------------------------------------------------------------------------------
        self.scene.terrain.terrain_generator.size = (8.0, 4.0)
        # 参考 num_rows=10（行＝难度 level）、num_cols=40（列＝地形类型）；初始等级 5 与本任务原值一致。
        # 2026-09-29（用户决定，重训前）：**抬高难度上限** —— gap 0.126~0.315 → **0.2~0.6 m**、
        # boxes 0.08~0.30 → **0.1~0.5 m**、stairs(_inv) 0.05~0.15 → **0.05~0.25 m**（步深 0.30 ⇒ 上限 39.8°）。
        # 同时 `num_rows 10 → 20`：难度按 `d = row/num_rows` 线性映射，只抬上限会让**中低难度一起变难**
        # （旧 level 6 的沟 0.25 m 会变成 ~0.35 m）⇒ 分级加倍后低难度基本维持、只有顶端变宽。
        # 未改动：hurdle（上限仍 ≈0.16 m）、narrow_stairs(0.10)、mix(1.1)、比例(gap 0.30)、初始等级 5。
        self.scene.terrain.terrain_generator.num_cols = 40
        self.scene.terrain.terrain_generator.num_rows = 20
        sub_terrains = self.scene.terrain.terrain_generator.sub_terrains
        # 上行台阶（参考 `stairs up` 0.1）。单级台阶 0.05–0.15 m 是我们 2026-09-24 定的四足区间：
        # 取 (0.05, 0.15) 而非更高——课程**下限必须可学**（level 0 就 10 cm 会直接卡死）；
        # 6 级 × step_depth 0.30 m = 1.8 m 跨度（x 2.0→3.8），难度 1.0 时总升高 0.9 m。
        # 腿部可行性：大腿 0.22 + 小腿 0.206 = 最大伸展 0.426 m、站立 0.30 m。
        sub_terrains["pyramid_stairs"] = CMoETrackStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.05, 0.20),
            num_steps=6,
            ascending=True,
        )
        # 下行台阶（参考 `stairs down` 0.1），参数与上行对称。
        sub_terrains["pyramid_stairs_inv"] = CMoETrackStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.05, 0.20),
            num_steps=6,
            ascending=False,
        )
        # 独立障碍块（参考 `discrete` 0.1）。整宽矮块，高 0.08–0.30 m（0.30 ≈ 0.95 体长 0.315 m，
        # 接近腿部最大伸展的可跃范围；下限同样为了 level 0 可学）；长 0.18–0.30、间距 0.85。
        sub_terrains["boxes"] = CMoETrackStepTerrainCfg(
            first_step_x=1.6,
            num_steps=2,
            step_spacing=1.3,
            step_length_range=(0.3, 0.5),
            proportion=0.10,
            step_height_range=(0.08, 0.30),
        )
        # 参考 `rough slope` 的 0.10 由我们的上/下坡各 0.05 承担（参考同一类里按列随机翻符号）。
        sub_terrains["hf_pyramid_slope"].proportion = 0.05
        sub_terrains["hf_pyramid_slope_inv"].proportion = 0.05
        # `random_rough`（纯噪声面 0.01–0.06 m）是我们保留的额外一类（2026-09-24 起的既有列）。
        sub_terrains["random_rough"].proportion = 0.05
        # 沟壑（参考 `parkour_gap` 0.30，比例完全一致）。沟宽按体长 0.4–1.0 L（躯干 0.315 m）
        # ＝0.126–0.315 m 是我们 2026-09-24 决定的四足区间；4 条沟、平台 0.65–0.95 m、首沟 x=1.8。
        sub_terrains["gap"] = CMoETrackGapTerrainCfg(
            proportion=0.30,
            gap_width_range=(0.12, 0.32),
            platform_length_range=(0.9, 1.4),
            first_gap_x=1.6,
            num_gaps=3,
        )
        # 2026-09-28 新增三类（参考有、我们缺）：横栏／混合障碍／窄楼梯。
        # 米制 = 参考 `parkour_hurdle_terrain` / `mix_obstacles_terrain` / `narrow_stairs_terrain`
        # × 0.4；各项常量与来源见 `cmoe_terrains.py` 对应配置类的注释。
        sub_terrains["hurdle"] = CMoETrackHurdleTerrainCfg(proportion=0.10)
        sub_terrains["mix"] = CMoETrackMixTerrainCfg(proportion=0.10)
        sub_terrains["narrow_stairs"] = CMoETrackNarrowStairsTerrainCfg(proportion=0.10)

        # 平地（2026-09-24 用户指出"似乎没有平地"后加入，2026-09-28 决定保留）。
        # 参考 plane 比例是 0.0；这块平地既是 trot 的"标称步态"学习面（真机录制基线也在平地），
        # 也方便部署／sim2sim 对照。MeshPlaneTerrainCfg 的 origin 是瓦片中心。
        sub_terrains["flat"] = terrain_gen.MeshPlaneTerrainCfg(proportion=0.10)

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
        # 2026-09-24（用户）：**所有地形列都只给"超前"的速度**——不再分"简单地形全向命令／障碍地形
        # 前向命令"，全场景统一为"沿世界 +x 前进 0.3–1.0 m/s、朝向锁 0"。
        # 起因：回放发现策略**横移绕开障碍**（§29.15/§29.16）。下面 `forward_only_terrain_names`
        # 因此列全**全部**地形键；`lin_pos_y`／`yaw_abs` 也改成**全局**（不再只作用于障碍列）。
        # 注意：本列表必须覆盖 `sub_terrains` 的**全部键**（漏项会静默退回全向命令）；
        # `scripts/tools/check_terrain_columns.py` 会做覆盖率校验。
        # 2026-09-28：新增 hurdle／mix／narrow_stairs 后同步补齐（11 类）。
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
        # 2026-09-24：以下三项（本提交前为 feet_stumble=-1、undesired_contacts=-5）都在惩罚
        # 「跨沟时足蹬对岸边沿／小腿擦对岸」这类必要动作，导致策略选择停在沟前。
        # 本次先降回 PPO rough 的量级做验证：feet_stumble 归零、undesired_contacts 回到 -0.5。
        self.rewards.feet_stumble.weight = 0.0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        # ---------------------------------------------------------------------------------
        # 2026-10-01（用户）：**只还原「奖励/权重」四项到 4000 轮那份存档配方**，其余一律不动。
        # 基准文件（逐字可信 —— 由 runner 落盘，是那次 run 实际使用的配置）：
        #   `imgo2_rl/logs/cmoe/base_move_cmoe_rough/2026-09-29_20-16-27_cmoe_v5_5_hard/params/CMoE_env_cfg.py`
        # 本段 = 还原项之一：`undesired_contacts` **−5.0 → −0.5**（与存档 L407 逐字一致）。
        # 2026-09-24 把它降回 −0.5 的理由是"跨沟时足蹬对岸边沿／小腿擦对岸是必要动作、惩罚会让策略
        # 停在沟前"；2026-09-30 曾为治"膝盖往地"加到 −5.0，现按"只还原奖励口径"的决定回退。
        # ⚠️ 注意：`ENABLE_ILLEGAL_CONTACT_BODY_TERMINATION = False`（开关）与其余地形/课程/终止项
        # **有意保留**，不随本次还原改动。
        self.rewards.undesired_contacts.weight = -0.5
        # 还原项之二：`contact_forces` **−0.1 → −0.02**（= 4000 时的父类默认值）。
        # 4000 那份存档里**没有**对本项的覆盖 ⇒ 实际生效值就是父类 `rough_env_cfg.py` 的 `-2e-2`。
        # 这里**显式写 −0.02**（而不是删掉这行去继承父类），好让 `check_reward_overrides.py`
        # 的 AST 审计能直接读到最终值，不落进"靠继承、静态看不出谁赢"的盲区。
        self.rewards.contact_forces.weight = -0.02

        # 2026-09-24：照搬 ZiwenZhuang/parkour 的 leap（跃沟）配方里**可平移**的权重。
        # 他们的 leap 技能：tracking_world_vel=+5.0、orientation=-0.1，且**完全没有**
        # lin_vel_z / ang_vel_xy / action_rate / base_height（见 docs §21）。
        # 目的：让「向前冲」的收益压倒「停在障碍前」。
        #
        # 2026-09-24 晚（用户："该参考 parkour 用全局的速度来约束了"）——**把机体系跟踪换成世界系**：
        # 回放确认策略「横移绕开障碍」（§29.15/§29.16）。根因之一是**速度奖励在机体系**：机体系版本
        # 配合 `heading_command`（yaw 指令由 heading 控制器按当前误差实时生成）时，机器人可以
        # "一边转身一边在机体系里前进"来拿满分（实测线速度核 0.88、偏航核仅 0.46）。parkour 用的是
        # **世界系** `tracking_world_vel`（`legged_robot_field.py:476`）⇒ 目标方向不随自身转动而变。
        # 2026-09-29（用户）：偏航角速度跟踪 0.6 → 0.8 → **2.0**（"转向直接给到 2.0"）。
        # 依据：命令里 `ang_vel_z ∈ (−1, 1)` 且 `heading_command=True`（航向由 0.5 刚度的控制器
        # 实时生成）＋ 全局 `yaw_abs −0.2`（别转身）；用户希望转向跟得更准。
        # ⚠️ 权衡（务必盯着看）：2.0 已是 `track_world_vel_xy_exp=5.0` 的 **40%**，不再是软项；
        #   本仓库其它任务里最高只用过 1.5（handstand/height），legged_gym 默认 0.5 ⇒ 这是**激进值**。
        #   风险：① 原地转向/蛇行时该项可能压过直线跟踪；② 强 yaw 率 + heading 控制器可能引起
        #   来回摆（`yaw_abs` 与它方向相反地拉扯）；③ 转向增多 ⇒ `feet_slide`/`illegal_contact` 变差。
        #   回退：一个数字（0.8~1.0）。监控读数：`Episode_Reward/track_ang_vel_z_exp`（应升）、
        #   `Episode_Reward/track_world_vel_xy_exp`（不应明显降）、`yaw_abs` 分项、`illegal_contact`。
        self.rewards.track_ang_vel_z_exp.weight = 2.0

        self.rewards.track_world_vel_xy_exp.weight = 5.0  # parkour 的 tracking_world_vel = 5.
        self.rewards.track_lin_vel_xy_exp.weight = 0.0  # 被世界系版本取代
        # parkour 同配方里另外两项"别绕开"的软约束（`go1_leap_config.py`：`lin_pos_y=-0.4`、
        # `yaw_abs=-0.2`；实现见 `legged_robot_field.py:493/496`）：
        #   `lin_pos_y` = |y − 出生点 y|＝**离赛道中心线的横向距离**；
        #   `yaw_abs`   = |yaw|（全场景 heading_target 都是 0 ⇒ 即"别转身"）。
        # 2026-09-24（用户）：**所有场景都给脱离中心的惩罚** ⇒ `terrain_names=()`（空＝全局生效），
        # 不再只作用于障碍列（那时普通列有 vy 指令与随机朝向，加全局罚会与命令打架；现在全场景
        # 都是前向命令，中心线/朝向罚与命令一致）。
        self.rewards.lin_pos_y.weight = -0.4
        self.rewards.lin_pos_y.params["terrain_names"] = ()
        self.rewards.yaw_abs.weight = -0.2
        self.rewards.yaw_abs.params["terrain_names"] = ()
        # 2026-09-28（用户）：**机身水平罚对齐 PPO rough 的 −5.0**（原 −0.1，相差 50×），
        # 但**不在障碍地形生效** —— 过沟/上箱/混合地形允许必要俯仰。PPO 是全地形 −5.0（它也有台阶），
        # 这里比 PPO 更宽松，属有意选择。实测依据：level 6 上 CMoE 的形态是"拖行/歪身"
        # （FL duty 0.80 / RR 0.57、速度 0.127 vs 指令 0.633），而 PPO 有 50× 强的水平罚把它按住。
        self.rewards.flat_orientation_l2.func = mdp.MaskedFlatOrientationL2
        self.rewards.flat_orientation_l2.weight = -5.0
        # 2026-09-28（用户："斜坡和台阶都不需要保持水平"）：豁免表 = **全部地形键 − 要求水平的地形**。
        # 于是在 `flat`/`random_rough` 上机身被显著压平（−5.0），而在两种斜坡（贴坡）、两种台阶与
        # 5 类障碍（俯仰）上完全不罚 ⇒ 让姿态跟着地形走。用现有地形键现算，避免两处硬写漂移。
        _terrain_keys = tuple(self.scene.terrain.terrain_generator.sub_terrains.keys())
        self.rewards.flat_orientation_l2.params["free_terrain_names"] = tuple(
            name for name in _terrain_keys if name not in LEVEL_ORIENTATION_TERRAIN_NAMES
        )
        # 2026-09-24 晚（用户："都还是蹦蹦跳跳的走的"）：**按地形豁免地恢复竖直速度罚**。
        # 形状：trot 列上恢复（压弹跳），`boxes`/`gap` 豁免（那里需要爆发式跃起，原清零理由是
        # "会与跃起对抗"——掩码后这个理由不再成立）。
        # 2026-09-30（姿态退化修复 ③）：权重曾 −2.0 → **−4.0**（治"抬脚过高/弹跳"）。
        # 2026-10-01（用户）：**还原项之三 ⇒ −4.0 → −2.0**（与 4000 存档 L461 逐字一致）。
        # 掩码**不动**：仍是 `("boxes", "gap")`（跃起仍必须免费）。2026-09-29 曾取消豁免，实测
        # L1~L4 的 gap 全无飞行相、boxes 出现前腿不承重 ⇒ 用户回放"gap/boxes 过不去" ⇒ 当日回退
        # （commit `f7d1dc3`）。**别再动掩码**；要压"雷霆大跳"应该用**滞空上限**（只罚长腾空）。
        self.rewards.lin_vel_z_l2.func = mdp.MaskedLinVelZ
        self.rewards.lin_vel_z_l2.weight = -2.0
        # 2026-09-29（用户："先退回到雷霆大跳版本"）：**恢复 boxes/gap 豁免**。
        # 依据（同日实测，见各档 dump `logs/gait_v54_L{1..4}.npz`）：
        #   * L1/L2 的 gap 是"走过去"（四足 duty 0.74~0.80，无腾空）；到 L3/L4 依然没有飞行相
        #     ⇒ 0.31 m 的宽沟（L6）**必须**短促腾空才过得去，而全地形 vz −2.0 让一次跃起
        #     （vz≈1.5 m/s）要付 `2×1.5²≈4.5/步` ⇒ 策略干脆不跳 ⇒ 用户回放确认"gap/boxes 过不去"；
        #   * 同时 boxes 在 L4 出现后腿 0.86~0.88 / 前腿 0.11 的"后腿蹬、前腿吊"姿态（占空比闸判 invalid）。
        # ⇒ 这是 09-24 记录过的同一失效模式（"罚 vz ⇒ 停在沟前"），在高难度上复现。
        # 真正的"压雷霆大跳"应该用**滞空上限**（只罚"长腾空"、不罚"短跃起"），
        # 即 `foot_clearance` 分支上的 `feet_air_time_over`——**待做**，别再动 vz 掩码。
        self.rewards.lin_vel_z_l2.params["free_terrain_names"] = ("boxes", "gap")
        self.rewards.ang_vel_xy_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = 0.0

        # ---------------------------------------------------------------------------------
        # ①/② 2026-09-30 「平地上蹲 7 cm + 膝盖蹭地 + 抬脚过高」姿态退化修复
        # 证据（run `cmoe_v5_7_lv12cap` @8500，`Episode_Reward/*` 是**每步加权值** ⇒ 真实每步贡献
        # = 该值 × dt 0.02；对照 @4000 是同一 run 退化前）：
        #   track_world_vel_xy_exp **+3.71**、track_ang_vel_z_exp **+1.37** ← 任务主力
        #   base_height_l2 **−0.0484** ⇒ 高度误差 RMS = √(0.0484/10) = **0.070 m（蹲 7 cm）**，
        #                              代价仅占任务的 ≈1%（@4000：−0.0030 ⇒ 退化 **14×**）
        #   undesired_contacts **−0.1019**（非足端接触 = 膝/小腿蹭地，@4000 −0.0008 ⇒ **127×**）
        #   lin_vel_z_l2 **−0.0585** ⇒ vz RMS ≈ **0.17 m/s**
        # ② 平地上把高度约束从"−10 带门控"变成"−10 带门控 + −35 去门控"：
        #    新增项只在 **flat** 列生效（白名单 `active_terrain_names`），障碍地形仍是 −10
        #    （用户："避免惩罚合法的越障姿态"）。⚠️ 它**不是**步态形状项 ⇒ `-gaitfree` 不归零。
        #    `target_height`/`sensor_cfg` 从既有 `base_height_l2` **现取**（不重写第二个 0.30 /
        #    第二个扫描器名），避免两处漂移 —— `asset_cfg` 同理复用基座 body_names。
        # **2026-10-01（用户）：还原项之四 ⇒ 启用权重 −35.0 → 0.0（禁用）。**
        #   4000 轮那份存档（见本函数上方基准文件说明）里**没有** `base_height_flat_l2` 这项；
        #   现按"只还原「奖励/权重」四项、其余一律不动"的决定把它移出生效表
        #   （靠本函数末尾的 `disable_zero_weight_rewards()`，权重 0 即移除）。
        #   term 定义与 `mdp.MaskedBaseHeightL2Strict` 类**保留**，作为"备用/可再启用"；
        #   **如需再启用：把下面这一行的权重改回 −35 即可**（参数/掩码都还在，唯一下一步）。
        self.rewards.base_height_flat_l2.func = mdp.MaskedBaseHeightL2Strict
        self.rewards.base_height_flat_l2.weight = 0.0
        self.rewards.base_height_flat_l2.params["target_height"] = self.rewards.base_height_l2.params[
            "target_height"
        ]
        self.rewards.base_height_flat_l2.params["sensor_cfg"] = SceneEntityCfg("height_scanner_base")
        self.rewards.base_height_flat_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=[self.base_link_name]
        )
        # 白名单语义：**只有 flat**（默认值也在 `CMoERewardsCfg` 里写了一遍，这里显式再写一次，
        # 好让"只在这一个地形生效"在审计源码里一眼可见）。
        self.rewards.base_height_flat_l2.params["active_terrain_names"] = ("flat",)

        # 未照搬（原因见 docs §21.3）：track_ang_vel_z_exp（我们命令里有 ±1.0 的 yaw，
        # 他们 leap 的 yaw 命令是 0，该项在其配方里近乎摆设）、base_height_l2（他们用
        # z_low 终止替代；直接删掉我们唯一的高度控制会重演 AMP 记录过的「贴地爬行」）、
        # feet_air_time（他们 leap 无步态塑形，但这是我方唯一正向步态项，先留）。
        # 2026-09-24（② 步态修复。起因：回放发现**所有地形**都塌缩到同一个 bound，见 docs §26）。
        # 现配方里没有任何项区分 trot 与 bound（五项 shaping 已清零，parkour 权重又关掉了
        # action_rate/ang_vel_xy/lin_vel_z）⇒ 对称弹跳步是免费的最优解。
        # `feet_gait`：2026-09-24 早先按用户决定**不用**（改用 PPO 那三项 shaping）；当晚用户改定
        # **"那就开 feet gait，同样加掩码"** ⇒ 已开启，见下方 ④（`TrotWithoutGapReward`，权重 1.0，
        # 掩码与其余四项步态 shaping 同一套）。注意 `synced_feet_pair_names` **必须给全**，
        # 给空字符串会在 `GaitReward.__init__` 直接抛 ValueError。
        # ② 恢复 PPO rough 原值 action_rate/ang_vel_xy，弱化弹跳的抖动与冲击（顺带抑制
        #    §23.3 里 mean_noise_std 涨到 1.5+ 的趋势）。
        # 刻意**不**恢复 lin_vel_z_l2：它直接惩罚竖直速度，会与过沟所需的爆发式跃起对抗。

        # 2026-09-24（③ 用户最终决定）：**照搬 PPO rough 的三项固定步态 shaping，都用地形掩码**。
        # 三项原值来自那次「trot 步态还行」的 PPO（复盘 docs §29.8）：
        #   `joint_mirror −1.0`（对角姿态一致，**含 hip**＝PPO 原配置）
        #   `feet_height_body −5.0`（强制抬脚，目标 −0.20 m ≈ 离地 0.10 m）
        #   `feet_air_time +1.0`（**阈值 0.5**＝PPO 原值）
        # 掩码：`feet_air_time`／`feet_height_body` 在障碍块（boxes）与沟槽（gap）上豁免，
        # 其余 13/20 列保留 trot 塑形；`joint_mirror` 的按地形行为见下方「分地形换对子」。
        # 相位问题现已由下方 ④ 的 `feet_gait` 负责（它显式要求对角反相 ⇒ 排除 pronk）。
        # 本段三项仍是"稠密先验 + 抬脚 + 时序均匀"，与相位项互补：`joint_mirror` 从第一步就有梯度，
        # 而 `feet_gait` 的 6 核乘积早期梯度≈0。
        self.rewards.joint_mirror.func = mdp.MaskedJointMirror
        self.rewards.joint_mirror.weight = -1.0
        # 分地形换对子（2026-09-24，用户要求「让 mirror 在沟壑变成 bound」）：
        #   trot 地形（13/20 列）＝**对角对**（FR↔RL、FL↔RR），巩固对角同相；
        #   `gap`（6 列）＝**换成正左右对 ⇒ bound 形状先验**（前腿一对同相、后腿一对同相），
        #        而不是单纯豁免——过沟要的就是前/后腿各自同步的 bound 式跃起；
        #   `boxes`（1 列）＝完全豁免（保持先前决定；若也想换成 bound，把它从 free 挪到 bound 即可）。
        # 对照 parkour：`_reward_sync_all_legs_cond`（注释即 "force same actuation on both front/rear
        # legs when jump"）比较的正是**右侧两腿 vs 左侧两腿**，且只在 engage `jump` 障碍时生效
        # ⇒ 本项与它同一机制。区别：他们 URDF 左右轴约定相反所以要翻肩关节符号，本 URDF 四腿轴
        # 完全相同，故左右对直接取**同号**。
        self.rewards.joint_mirror.params["free_terrain_names"] = ("boxes",)
        self.rewards.joint_mirror.params["bound_terrain_names"] = ("gap",)
        # **含 hip**（＝ PPO rough 原配置）。⚠️ 更正 2026-09-24 早先的一条错误注释：当时以为
        # mirror 对是"跨左右两侧"，于是删掉了 hip。实际 PPO 的对是 **FR↔RL、FL↔RR（对角对）**：
        # trot 里对角腿同相，而本 URDF **四条腿的关节轴完全相同**（hip=(1,0,0)、thigh/shank=(0,1,0)），
        # 所以"同相"就意味着**各关节同号**，不加符号翻转是对的。parkour 的 `_sync_legs_cond`
        # 比较的是 **RR↔RL（左右对）** 并翻转肩关节符号，那是**左右镜像**（轴约定左右相反）的算法，
        # 与对角对不是同一回事，不能直接类比。
        # 唯一残留的近似：hip 的"对称外展"是 `q_FR = −q_RL`，不是本项的最小点（`q_FR = q_RL`），
        # 所以该项严格说会轻微抑制髋外展、偏向"同向摆"。因为默认站姿 `hip=0`（最小值点正好是
        # 默认站姿）且 trot 中髋基本不动，实际影响有限；若日后看到机体歪斜/单侧铺腿的倾向，
        # 正确修法是**对 hip 翻转符号**（而不是删掉 hip）。
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|shank).*", "RL_(hip|thigh|shank).*"],
            ["FL_(hip|thigh|shank).*", "RR_(hip|thigh|shank).*"],
        ]
        self.rewards.joint_mirror.params["bound_mirror_joints"] = [
            ["FL_(hip|thigh|shank).*", "FR_(hip|thigh|shank).*"],
            ["RL_(hip|thigh|shank).*", "RR_(hip|thigh|shank).*"],
        ]
        # `feet_air_time` 本来是全局 +1.0；这里改为**按地形豁免**并回到 PPO 的阈值 0.5。
        # ⚠️ 更正 §29.7 的一处过度解读：展开 `Σ(last_air_time − c) = Σ last_air_time − c·N_落地`
        # 可见 c 只是"每次落地扣一个常数"⇒ 对滞空时间的**梯度方向与 c 无关**，c=0.5 与 0.25 的
        # 方向一致，差别在**"减少落地次数（步幅更长）"的压力（0.5 是 0.25 的 2 倍）**与日志偏移量
        # （0.5 时该分项通常为负）。PPO 用 0.5 且步态良好，故回到 0.5。
        self.rewards.feet_air_time.func = mdp.MaskedFeetAirTime
        # 2026-09-24 晚：1.0 → **0.3**。该式展开是 `4(1−d) − 2·N落地/T` ⇒ 对"滞空更久"的梯度恒为 +1
        # ⇒ 权重越大越奖励腾空/弹跳。降权后仍保留"别踩碎步"的作用（阈值仍 0.5 s），但不再主导步态。
        # 2026-09-28（用户）：**对齐 PPO rough 的 1.0**（原 0.3，只有 30% 的滞空/步幅压力）。
        # PPO 的"步态 shaping / 速度跟踪"比约 **7:1**，我们原配方只有 2.6:1（且 37.5% 环境被掩码豁免）
        # ⇒ 这是 CMoE 压不住步态的主因之一。掩码保持 `("boxes","gap")`（那两类允许爆发式跃起）。
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["free_terrain_names"] = ("boxes", "gap")

        # ③ 之三：足端抬升塑形 `feet_height_body`（摆动足机体系高度误差，目标 −0.20 m ≈ 离地 0.10 m）
        #    按地形豁免地恢复（障碍块/沟槽上放开）。依据：用户明确"我们从零训，需要 feet 相关奖励"
        #    （parkour 可以不要，因为它在已训好的行走策略上 fine-tune）；参考基线足端 z 峰峰中位 0.090 m。
        #    权重沿用 PPO rough 原值 −5.0；`target_height`/`tanh_mult`/link 过滤沿用 rough_env_cfg 的配置。
        self.rewards.feet_height_body.func = mdp.MaskedFeetHeightBody
        self.rewards.feet_height_body.weight = -5.0
        self.rewards.feet_height_body.params["free_terrain_names"] = ("boxes", "gap")
        # ③ 之四（2026-09-24 晚，用户："没看到在平坦地形的 trot 步态"）：把 PPO 那套里
        # **量级最大的步态项** `feet_air_time_variance` 加回来（−8.0＝PPO 原值），但按地形豁免。
        # 它是三项 shaping 里唯一管"**四足之间时序是否均匀**"的项：bound（前对/后对错开）会让
        # 前/后足的滞空与触地时长不一致而被罚。判据/门控沿用 `rough_env_cfg` 的配置（同一函数）。
        # ⚠️ 它排除 bound/pace，但 pronk（四足完全同步）满足它；要排除 pronk 得开相位项 `feet_gait`。
        self.rewards.feet_air_time_variance.func = mdp.MaskedFeetAirTimeVariance
        self.rewards.feet_air_time_variance.weight = -8.0
        self.rewards.feet_air_time_variance.params["free_terrain_names"] = ("boxes", "gap")
        # ④ 相位项 `feet_gait`（2026-09-24 用户："那就开 feet gait，同样加掩码"）。
        # 这是全配方里**唯一**显式要求"**对角对之间反相**"的项 ⇒ 排除 bound／pace／**pronk**
        # （`joint_mirror` 只压"对角对内相等"，pronk 满足它；`feet_air_time_variance` 罚四足时长方差，
        # pronk 也满足）。对角腿对＝trot 相位（`FL↔RR`、`FR↔RL`），权重 1.0（约占 `track_world_vel`
        # 5.0×~0.88≈4.4/s 的 20% 上限，属"温和但明确"的偏好）。掩码与其余四项一致：豁免 `boxes`/`gap`。
        self.rewards.feet_gait.func = mdp.TrotWithoutGapReward
        self.rewards.feet_gait.weight = 1.0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (
            ("FL_FOOT", "RR_FOOT"),
            ("FR_FOOT", "RL_FOOT"),
        )
        self.rewards.feet_gait.params["free_terrain_names"] = ("boxes", "gap")
        # ⚠️ 2026-09-24 夜（用户同意）：**把相位核变陡**。`std`/`max_err` 一直在沿用原版 PPO 的值
        # （`velocity_env_cfg.py:551` 的 `feet_gait`，`std=√0.5=0.7071`、`max_err=0.2`），
        # 于是单核地板＝`exp(−2·0.2²/0.7071)=0.893`：**配对完全错（差半个周期 ≈0.45 s）也拿 89 分**，
        # 6 核乘积只跨 [0.508, 1] ⇒ 这个项度量的是"有几对配上了"（近似二值），不是"差多少"。
        # run G @436 实测：`trot−bound` 只有 **−0.010**（≈0.1 个核），且从第 50 轮到 436 轮**完全平坦**
        # （期间速度核 0.20→0.62）⇒ 边际奖励 0.01/s（总量 ~5.5/s 的 0.2%），PPO 不会因它改步态。
        # 新值：地板降到 `exp(−2·0.5²/0.2)=0.082`，参考步态 trot 1.000 / bound 0.302
        # ⇒ 溢价 **0.264 → 0.698/s（2.7×）**。**必须与三个 `gait_metric_*` 的 std/max_err 同步**
        # （否则分类器读数不再代表奖励）。详见 docs §29.31.2 与 §29.32。
        self.rewards.feet_gait.params["std"] = 0.2
        self.rewards.feet_gait.params["max_err"] = 0.5
        # 2026-09-28（用户："相位核去掉"）：**步态不再靠相位核**，改为靠 PPO 的四项
        # （`joint_mirror`／`feet_air_time` 1.0／`feet_height_body` −5／`feet_air_time_variance` −8）
        # ＋ `feet_slide` −0.05 与 50× 强的 `flat_orientation_l2`。依据：相位核在这份高频步态上
        # **本身饱和**（实测溢价仅 0.116/s = 跟踪项的 2.3%，见 docs/cmoe_trot_warmstart §17），
        # 而 PPO 根本不用相位核也能练出干净 trot ⇒ 保留它只会多一个调不动的旋钮。
        # 三个 `gait_metric_*` 探针（1e-6）**保留**：它们是逐列步态的唯一读数来源。
        self.rewards.feet_gait.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.ang_vel_xy_l2.weight = -0.05

        self.rewards.feet_height.weight = 0.0
        # `feet_height_body` 同理不清零——已在上方设为 −5.0（MaskedFeetHeightBody）。
        # 2026-09-28（用户："feet_slide 对齐 PPO"）：PPO rough 是 **−0.05**，我们之前清零了。
        # 这一项是"**支撑脚不许打滑**"，正对着我们实测的拖行形态；PPO 的干净步态有它一份。
        # 足端 body 与 PPO 一致用 `.*_FOOT`（PPO 的 `foot_link_name` 就是这个正则）。
        # 注意：这是**全地形**生效（与 PPO 一致）；若发现 gap 上（落脚在对面边缘、滑动不可避免）
        # 惩罚过重，把 `("gap",)` 填进 `free_terrain_names` 即可（该 term 已支持掩码参数）。
        self.rewards.feet_slide.weight = -0.05
        self.rewards.feet_slide.params["sensor_cfg"].body_names = ".*_FOOT"
        self.rewards.feet_slide.params["asset_cfg"].body_names = ".*_FOOT"
        # `feet_air_time_variance` 不在这里清零——已在上方设为 −8.0（MaskedFeetAirTimeVariance，掩码）。
        # 注意：`joint_mirror` 不在这里清零——它在本函数上方被设为 −1.0（MaskedJointMirror，
        # 障碍块/沟槽豁免）。2026-09-24 曾因这一行在下方、晚赋值把它覆盖成 0，导致 mirror 静默失效。

        # Falling on the base ends the episode; foot contacts remain legal.
        self.terminations.illegal_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.base_link_name]),
                "threshold": 1.0,
            },
        )
        # ④之三 2026-09-30（姿态退化修复，「称重跪地」终止）：**非足端** body 的**高力**接触即终止。
        # 与 `undesired_contacts`（2026-10-01 还原为 **−0.5**）是同一现象的"罚 vs 终止"两级：
        #   * 轻擦（< 50 N）⇒ 只被接触罚容忍，跨沟/上台阶的常见动作仍容忍；
        #   * 称重跪地（瞬时接触力 ≥ 50 N）⇒ 直接终止（用户看到的"膝盖往地"就是这一档）。
        # ⚠️ 本项**不属于**本次（2026-10-01）要还原的"奖励/权重"四项 ⇒ 开关与代码**有意保留**。
        # 阈值 50 N 是**故意取高**的：既有的基座触地终止用 1 N，若这里也用 1 N，任何小腿轻擦都会
        # 立刻结束回合 —— 那正是 2026-09-24「罚太重 ⇒ 策略停在沟前」的同一个坑。
        # ⚠️ **这是四项改动里最容易伤到 gap/stairs 的一项**：若 `illegal_contact_body` 的终止占比
        # 明显上升（或 `level_gap`/`level_pyramid_stairs*` 下滑），优先怀疑它。回退方式：
        # 把模块级 `ENABLE_ILLEGAL_CONTACT_BODY_TERMINATION` 改成 `False`（**一行**，其余三项不受影响）。
        # 注意 body_names 的正则是在**全量 body 名**上做否定前瞻：`.*_FOOT` 精确匹配四个足端 link，
        # 故本项只匹配"非足端"（膝/小腿/大腿/髋/基座）。基座触地由上面的 `illegal_contact`（1 N）负责，
        # 两者重叠部分（基座 ≥ 50 N）会被任一项终止，语义不冲突。
        if ENABLE_ILLEGAL_CONTACT_BODY_TERMINATION:
            self.terminations.illegal_contact_body = DoneTerm(
                func=mdp.illegal_contact,
                params={
                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[r"^(?!.*_FOOT).*"]),
                    "threshold": 50.0,
                },
            )

        # 用仓库本地的带诊断副本替下上游 `terrain_levels_vel`。**注意：判据自 2026-09-24 起
        # 与上游有两处有意偏离**（见 `mdp/curriculums.py` 的 docstring 与 docs §29.20）：
        #   ① 晋级看**沿 +x 的前向进度**（不再用含横向分量的欧氏距离 ⇒ "横移绕开"不算通过）；
        #   ② 晋级还要求**本回合速度跟踪达标**（`track_avg > tracking_move_up`），
        #      跟踪太差（< `tracking_move_down`）直接降级 —— 用户："地形等级提升还是需要考虑速度跟踪效果"。
        # 另额外记录 move_up/down/frozen 比例、逐地形列等级均值与跟踪统计量（用于判断课程是否饱和）。
        self.curriculum.terrain_levels = CurrTerm(
            func=mdp.terrain_levels_vel_logged,
            params={
                "tracking_term_name": "track_world_vel_xy_exp",
                "tracking_move_up": 0.80,
                "tracking_move_down": 0.35,
                # 2026-09-24 晚（用户反馈"反楼梯不是很好"）：这三类只能**慢慢过**的地形在 0.80 下
                # 长期卡在 frozen 带（反楼梯 0.12、boxes 0.10，而 flat/slope 已 4–6 级）⇒ 单独放宽。
                # **2026-09-25 修**：`gap` 也纳入。依据 `2026-09-25_14-35-18_cmoe_gaitfree_amp24500_init`
                # 的实测：`level_gap` 从第 200 轮起**严格 0.000**（140+ 轮不动），同期 `tracking_gap`
                # = 0.486，而 gap 不在放宽名单里 ⇒ 需要 >0.80 ⇒ 落在 [0.35, 0.80) 的**冻结带**，
                # 既不上也不下 ⇒ 沟壑列永远停在最窄的 0.126 m，过沟能力根本测不到。
                # （`boxes` 虽在名单里但 `tracking_boxes`=0.467 < 0.50 同样被冻住 ⇒ 阈值一起下调。）
                # **2026-09-28 补**：新增的 hurdle／mix／narrow_stairs 同属"只能慢慢过"的障碍列
                # （窄走廊上速度跟踪天然偏低），不纳入同样会落进 [0.35, 0.80) 的冻结带 ⇒ 一并列入。
                # 这是**地形集扩张的连带修正**，不是重新设计课程判据（判据本身未改）。
                "relaxed_terrain_names": (
                    "pyramid_stairs",
                    "pyramid_stairs_inv",
                    "boxes",
                    "gap",
                    "hurdle",
                    "mix",
                    "narrow_stairs",
                ),
                # 逐列步态度量（三成对方式＝分类器）：日志会出 `gait_trot_<地形>` / `gait_bound_<地形>`
                # / `gait_pace_<地形>`，以及全局 `gait_<标签>_mean`。用来验证"除 boxes/gap 外倾向 trot"。
                "gait_metric_terms": (
                    ("trot", "gait_metric_trot"),
                    ("bound", "gait_metric_bound"),
                    ("pace", "gait_metric_pace"),
                    ("bounce", "diag_bounce"),        # 逐列 vz 均方（开方＝vz RMS）
                    ("height", "base_height_l2"),     # 逐列 (base 高度误差)²（开方＝RMS 误差）
                    # 2026-09-24 夜新增：把"步子长短"与"相位差尺度"直接读出来
                    ("airtime", "diag_air_time"),     # 逐列四足 last_air_time 均值（s）
                    ("mismatch", "diag_pair_mismatch"),  # 逐列六对 |Δair|+|Δcon| 均值（s）
                    # 2026-09-30 新增：逐列基座相对**局部地面**的**有符号**高度误差（m，
                    # 正=偏高、负=偏矮）⇒ `gait_base_height_<地形>`。与 `("height", "base_height_l2")`
                    # （只有平方值、丢符号）互补：蹲矮 ⇒ 负、抬高 ⇒ 正。
                    ("base_height", "diag_base_height"),
                ),
                # 2026-09-24（用户）：**台阶与 boxes 用 0.50、其余仍 0.80**。
                # ⚠️ 0.50 在这三类上**等价于取消跟踪门控**（整回合平均跟踪核实测 mean≈0.78、min≈0.69
                # ⇒ 没有任何回合会低于 0.50）⇒ 它们回到"只看前进进度（走够 4 m）"的旧判据。
                # 依据：旧判据在这三类上曾把 A 跑推到 level 6（反楼梯 6.06／boxes 5.99），而"能不能过障碍"
                # 本来就该由进度管；跟踪门控更适合容易地形（那里"糊弄过去"才是失败模式）。
                # **2026-09-25 修**：0.50 → **0.40**，并把 `gap` 纳入放宽名单。原因：`_init` run 里
                # `tracking_gap`=0.486、`tracking_boxes`=0.467，都在 [0.35, 0.50) 的冻结带里出不来；
                # 0.40 让这两列**都过门**（冻结带缩到 [0.35, 0.40)，即障碍列基本回到"只看进度"），
                # 而容易地形仍保持 0.80 的跟踪门控。逐列 `tracking_pass_frac_<地形>` 会显示实际过门比例，
                # 若发现障碍列"进度够但跟踪很差"仍被放行，回调到 0.45–0.50 即可。
                "tracking_move_up_relaxed": 0.40,
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
        # 2026-09-29（用户："play 的地形改一下，不用那么多，只要每种地形一行"）：
        # 回放网格缩到 **每类一列**：`num_cols = 11`（= sub_terrains 的 11 类）＋**等比例**（全 1.0，
        # Isaac 会归一化）⇒ 每类恰好占 1 列（按训练的比例会变成 gap 3 列、rough/slope 0 列，不行）。
        # 训练侧保持 40 列按比例（课程需要每类多列）；回放要的是"每类都看得到"。
        # 注意：`--terrain_level=N` 仍可把这一列的难度钉在 N（num_rows 现为 20 ⇒ 难度 = N/20）。
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


# ======================================================================================
# 2026-10-04（用户）：**只在 `mix` 一种地形上评测的受控测试场景**（**继承 play 任务**）。
#
# 用途：在受控条件下评估策略通过 `mix`（复合障碍：窄走廊+台阶+深坑+高台+高栏）的能力，把
# **横向漂移**与**航向漂移**从评测里剔除。本场景只改四件事：
#   ① 地形只留 `mix`；网格 = **20 条并排的 mix 道（沿世界 Y）× 20 档难度（沿世界 X）**，
#      且**全部环境固定在第 14 行 ⇒ 名义难度 d = 14/20 = 0.70**；
#   ② 速度指令**只给前进**（默认恒定 **1.0 m/s**）、`heading` 目标恒 0；
#   ③ 横向与航向改由**指令层 PD 外环**负责（`mdp.MixTestVelocityCommand`）；
#   ④ mix 图案的**障碍间距乘子 = 2.0**（`pattern_spacing_scale`，只影响本评测场景；训练侧不传
#      该字段 ⇒ 取默认 1.0 ⇒ 训练几何**逐位不变**）。
#
# 2026-10-04（用户第二批要求，原话）：「地形不要按照列排，放在行里面」「都固定到14难度」
# 「mix 中每个地形间隔大一点 ×1.5~2.0」。
#
# 网格方向的真值（来自 Isaac Lab 源码，不是约定）：`terrain_generator.py:247-261` 逐列逐行用
# `difficulty = (sub_row + U(0,1)) / num_rows` 生成瓦片；`:330` 里 `terrain_origins[row, col]`
# 的平移量是 `((row+0.5)·size[0], (col+0.5)·size[1])` ⇒ **行（难度）沿世界 +X、列（地形类型/道）
# 沿世界 +Y**；整片地形再按 `(-size[0]·num_rows/2, -size[1]·num_cols/2)` 居中（`:176-182`）。
# 本场景 size=(8,4)（父类设定）、num_rows=20、num_cols=20 ⇒
#   * 世界 X ∈ [-80, +80]：20 行 × 8 m；第 14 行占 [112, 120] − 80 = **[+32, +40]**；
#   * 世界 Y ∈ [-40, +40]：20 道 × 4 m；第 i 道占 [4i − 40, 4(i+1) − 40]（第 0 道 [-40,-36]）。
# **动作空间与观测契约一字未改** ⇒ 既有 CMoE checkpoint 可直接加载（这不是新策略任务）。
# ⚠️ 本类是"评测场景"，不参与训练；注册 id 见 `base_move/__init__.py`。
# ======================================================================================
# 道数 = `num_cols` = **可同时评估的环境数上限**（20 条并排的 mix 道；每条道一个环境 —— 见下面
# `Imgo2CMoEMixTestTerrainImporter` 里 `terrain_types` 的确定性分配说明）。
MIX_TEST_LANES = 20
# 难度档数 = `num_rows`（难度 = 行号 / num_rows）。行沿世界 +X 排。
MIX_TEST_LEVELS = 20
# **固定难度行**：全部环境钉在第 14 行 ⇒ 名义难度 d = MIX_TEST_PINNED_LEVEL / MIX_TEST_LEVELS = 0.70。
MIX_TEST_PINNED_LEVEL = 14
# mix 图案的**障碍间距乘子**（`CMoETrackMixTerrainCfg.pattern_spacing_scale`）：只影响本评测场景
# （训练侧不传该字段 ⇒ 取默认 1.0，训练几何逐位不变）。用户给的范围 1.5~2.0，取上限 2.0。
MIX_TEST_PATTERN_SPACING_SCALE = 2.0


class Imgo2CMoEMixTestTerrainImporter(terrain_gen.TerrainImporter):
    """把课程难度**钉死**在 :data:`MIX_TEST_PINNED_LEVEL` 行、并**冻结升降级**的地形导入器。

    为什么需要一个子类：Isaac Lab 的 `_compute_env_origins_curriculum`
    （`isaaclab/terrains/terrain_importer.py:329-348`）在课程模式下的初始等级是
    ``terrain_levels = randint(0, min(max_init_terrain_level, num_rows-1) + 1)`` ⇒
    `max_init_terrain_level` 只表达"**上限**"、无法表达"全部钉在第 N 行"；而
    `update_env_origins`（`:308-323`）每回合还会按课程判据升降级（到顶时甚至 `randint_like`
    随机重开）。本子类做两件事（**只用于 mix-test 任务**：训练/play 的 `class_type` 仍是
    `TerrainImporter`，行为一字未改）：

    ① `configure_env_origins` 之后把 `terrain_levels[:]` 全部设为 `pinned_level` 并重算 `env_origins`；
    ② `update_env_origins` 变成空操作 ⇒ 一局之内难度与出生点都不变。

    这等价于 `play.py` 的 `--terrain_level=N`（钉死 + 冻结课程），但**不依赖 CLI**：不传参时默认就是
    第 14 行；传 `--terrain_level=M` 仍可覆盖（CLI 在 env 建好后直接改 `terrain_levels`，本子类不再干预）。

    **道（`terrain_types`）不需要改**：Isaac Lab 的分配是**确定性**的
    ``terrain_types[i] = floor(i · num_cols / num_envs)``（同文件 `:342-344`）⇒ 本场景
    `num_envs == num_cols == 20` 时恰好 **环境 i → 第 i 道**，一条道一个环境、不会两块瓦片挤一起。
    ⚠️ 该式只在 `num_envs ≤ num_cols` 时严格递增（`terrain_types[i+1] ≥ terrain_types[i] + 1`）；
    一旦 `--num_envs > 20` 就会出现重复取值 ⇒ 多个环境落在**同一条道的同一行**、出生点重叠（构造时告警）。
    """

    pinned_level: int = MIX_TEST_PINNED_LEVEL

    def configure_env_origins(self, origins=None):
        super().configure_env_origins(origins)
        if self.terrain_origins is None:
            return  # 非课程地形（plane / usd）⇒ 没有"等级"可钉
        num_envs = int(self.cfg.num_envs)
        num_cols = int(self.terrain_origins.shape[1])
        if num_envs > num_cols:
            print(
                f"[WARN] {type(self).__name__}: num_envs={num_envs} > num_cols={num_cols} ⇒ "
                "terrain_types 会重复（多个环境挤在同一条道、同一行 ⇒ 出生点重叠）；"
                f"请让 --num_envs ≤ {num_cols}（= MIX_TEST_LANES）。"
            )
        self.terrain_levels[:] = int(self.pinned_level)
        self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]

    def update_env_origins(self, env_ids, move_up, move_down):
        """课程升降级对本场景是 **no-op**：难度固定 ⇒ 等级与出生点都不再变。"""
        return


@configclass
class Imgo2CMoEMixTestEnvCfg(Imgo2CMoERoughPlayEnvCfg):
    """`mix`-only 受控测试场景：**20 条并排的 mix 道 × 固定难度 14** + 前进 1.0 m/s + 横向/航向 PD。

    继承 `Imgo2CMoERoughPlayEnvCfg` ⇒ 自动保留它的确定性设定（`pose_range` 全 0、`velocity_range`
    全 0、关闭全部域随机化事件、关闭观测噪声），本类覆盖：

    * **网格**：`num_cols = MIX_TEST_LANES = 20`（20 条并排的 mix 道 ⇒ 沿世界 Y 展开）×
      `num_rows = MIX_TEST_LEVELS = 20`（难度 ⇒ 沿世界 X）；每道一个环境（`num_envs = 20`）；
    * **难度固定 = 14**：`max_init_terrain_level = MIX_TEST_PINNED_LEVEL = 14` ＋
      `class_type = Imgo2CMoEMixTestTerrainImporter`（钉死等级、冻结升降级）⇒ 名义 d = 14/20 = 0.70；
    * **障碍间距 ×2.0**：`pattern_spacing_scale = MIX_TEST_PATTERN_SPACING_SCALE = 2.0`
      ⇒ 图案 X 总长 = 0.30 + 160×0.02×2.0 = **6.70 m ≤ 8 m**（该瓦片上的乘子上限 2.40625）；
    * 速度指令 + 横向/航向 PD 外环（见下）。

    **运行示例**（回放，须替换 checkpoint 路径；难度已由 cfg 固定为 14，**不需要** `--terrain_level`
    —— 只有要扫别的难度时才传）::

        python scripts/rl_lab/cmoe/play.py \\
            --task=Imgo2-basemove-rough-cmoe-mix-test --num_envs=20 --headless \\
            --checkpoint="/absolute/path/to/model.pt"

        # 想覆盖难度（例如扫到最容易的一档）：--terrain_level=0
        # `--terrain_level=N` ⇒ 全部环境钉在第 N 行 ⇒ 名义难度 N/20（0 ≤ N ≤ 19，**不会**被夹到 9）。

    ⚠️ **名义 vs 实际难度**：Isaac Lab 的课程生成器给同一行内每个瓦片加 ``U(0,1)`` 抖动
    （`difficulty = (row + U(0,1)) / num_rows`）⇒ 第 14 行的实际 ``d ∈ [0.70, 0.75)``，0.70 是**下界/
    名义值**。要**精确** 0.70 需把 `terrain_generator.difficulty_range` 收成 `(0.70, 0.70)`，但那会让
    20 行的难度全部相同、`--terrain_level` 失去"扫难度"的意义 ⇒ 本类**有意不做**（见 docs 的
    "未验证项/与设计冲突之处"一节）。
    """

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------------------------------------ ① 地形：只留 `mix` 一类
        # 先清空（play 类刚把所有 proportion 设成 1.0），再只放 mix。**逐字沿用训练实例化时的 mix 值**
        # —— 训练侧是 `CMoETrackMixTerrainCfg(proportion=0.10)`，其余字段全部取类默认值，这里把它们
        # 显式展开写死（`tests/test_cmoe_mix_test_scene.py` 会断言这些字面量与 `cmoe_terrains.py` 的
        # 类默认值**逐一相等**，改默认值而不同步这里就会红）。
        # **有意偏离默认值的两处**（测试里单独断言）：
        #   * `proportion=1.0`（本场景只有这一类，比例无意义、Isaac 会归一化）；
        #   * `pattern_spacing_scale=2.0`（间距乘子，**评测专用**；训练取默认 1.0）。
        self.scene.terrain.terrain_generator.sub_terrains.clear()
        self.scene.terrain.terrain_generator.sub_terrains["mix"] = CMoETrackMixTerrainCfg(
            proportion=1.0,
            x_unit=0.02,            # = _MIX_X_UNIT = 0.05 × REFERENCE_SCALE(0.4)
            z_unit=0.002,           # = _MIX_Z_UNIT = 0.005 × REFERENCE_SCALE(0.4)
            height_scale=1.1,       # 参考 diff = hurdle_height_range[0] × 1.1
            gap_shrink_units=10.0,  # 参考 round(10 − 10·d)
            corridor_width=0.80,    # 参考走廊半宽 20 索引 = 1.0 m ⇒ ×0.4
            pit_depth=0.50,         # 参考坑深 0.5–1.5 m ⇒ ×0.4 后取固定值
            pattern_start_x=0.30,   # 见 `track_mix_terrain` docstring 的"已知偏离"（起步平台装得下 0.75 m 出生点）
            pattern_spacing_scale=MIX_TEST_PATTERN_SPACING_SCALE,  # 2.0：只放宽本评测场景的障碍间距
            spawn_x=0.75,
        )
        # **20 条并排的 mix 道 × 20 档难度**（列沿世界 Y、行沿世界 X）：
        # 道数 = 可同时评估的环境数上限 ⇒ 默认 20 个环境时"一道一个环境"，`terrain_types` 恰为 0..19。
        self.scene.terrain.terrain_generator.num_cols = MIX_TEST_LANES
        self.scene.terrain.terrain_generator.num_rows = MIX_TEST_LEVELS
        # ② **难度固定 14**（不依赖 CLI）：`max_init_terrain_level` 与 importer 的 `pinned_level`
        # 一致 ⇒ 初始等级就是 14、且课程升降级被冻结（见 `Imgo2CMoEMixTestTerrainImporter`）。
        self.scene.terrain.max_init_terrain_level = MIX_TEST_PINNED_LEVEL
        self.scene.terrain.class_type = Imgo2CMoEMixTestTerrainImporter
        # 默认 20 个环境（可被命令行 `--num_envs` 覆盖；⚠️ 超过 20 会与别的环境共享同一条道 ⇒ 出生点重叠）。
        self.scene.num_envs = MIX_TEST_LANES

        # ------------------------------------------------- ②③ 速度指令 + 横向/航向 PD 外环
        # **整项替换**（而不是只改 ranges）：新命令项类需要 PD 增益字段，量纲与语义都与
        # `UniformThresholdVelocityCommandCfg` 不同，逐字段覆盖容易漏（参见 CMOE-04 的"晚赋值覆盖"教训）。
        # 观察/动作契约不受影响：命令项只换 `class_type`，`vel_command_b` 仍是 (num_envs, 3)。
        self.commands.base_velocity = mdp.MixTestVelocityCommandCfg(
            asset_name="robot",
            # 不做指令重采样 ⇒ 整个 episode 指令恒定（vx 只在首次 resample 采样一次）。
            resampling_time_range=(1.0e9, 1.0e9),
            # 规格 ②：heading_command 必须为 True（也是 `ranges.heading` 的合法性前提）。
            # 但本命令项**不使用**内置 heading P 控制器，而是自己用 PD 写 wz（见 mix_test_command.py
            # docstring 第 4 点）—— 因为 `ranges.ang_vel_z=(0,0)` 会把内置控制器的输出恒夹成 0。
            heading_command=True,
            rel_heading_envs=1.0,
            rel_standing_envs=0.0,
            # 与 play 任务一致（`CommandsCfg` 里显式 `debug_vis=True`）；20 环境 ⇒ 40 个速度箭头 marker。
            debug_vis=True,
            ranges=mdp.MixTestVelocityCommandCfg.Ranges(
                # **前进速度**：默认恒定 1.0 m/s（2026-10-04 用户规格修正）。
                # 可调：改这一行即可（例如扫速度用 `(0.5, 1.0)` + 缩短 `resampling_time_range`）。
                lin_vel_x=(1.0, 1.0),
                # 横向与航向由 PD 外环写（此处只留 0 作初值/占位，PD 每步覆盖第 1/2 列）。
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                # heading 目标恒 0（航向保持）；本项不读这个范围，只用于合法性与自文档。
                heading=(0.0, 0.0),
            ),
        )


@configclass
class Imgo2CMoEGaitFreeEnvCfg(Imgo2CMoERoughEnvCfg):
    """**步态交给先验**的配方：五项手工步态 shaping 全部归零，其余奖励一项不动。

    用户 2026-09-25 决定（依据：`e6a1155` 把相位核变陡后，run H 1849 轮的三态步态读数仍是
    `trot 0.768 / bound 0.782 / pace 0.765`，离线反演（`scripts/tools/gait_kernel_probe.py`）
    显示该形态最接近**四足锁相**，即调核这条路已经走尽 —— 详见
    `docs/cmoe_trot_warmstart_2026-09-25.md` §12）。

    * 归零的五项＝`joint_mirror`／`feet_air_time`／`feet_height_body`／`feet_air_time_variance`／
      `feet_gait`。它们是"从零学 trot"的手工代理；步态改由 **45 维先验**（AMP 24500，
      `--init_experts_from`）提供后继续留着反而会与先验打架 —— 尤其 `feet_air_time` 的展开
      `4(1−d) − 2·N落地/T` 对"滞空更久"的梯度恒为 +1，是在**奖励腾空/弹跳**。
    * **刻意保留、不要跟着一起归零**：
      - 三个 `gait_metric_{trot,bound,pace}`（1e-6）＝步态分类器。权重 0 会被
        `disable_zero_weight_rewards()` 整个移除，就再也看不见"先验漂没漂"；
      - `diag_air_time`／`diag_bounce`／`diag_pair_mismatch`／`diag_base_height`（1e-6）＝接触时序、
        弹跳与**有符号**基座高度误差诊断；
      - `base_height_flat_l2`（`MaskedBaseHeightL2Strict`，**2026-10-01 起权重 0.0 = 已禁用**）：
        它曾是 flat-only 的**姿态**项（平地上别蹲，−35），**不是**步态形状先验；2026-10-01 按
        "只还原「奖励/权重」四项到 4000 轮存档口径"的决定移出生效表（term 定义与类保留、可再启用）。
        本类同样不动它（既不启用、也不额外归零 —— 父类已把它设为 0.0 并移除）；
      - `lin_vel_z_l2`（`MaskedLinVelZ`，**2026-10-01 还原为 −2**）：这是**物理平滑惩罚**
        （压竖直速度/弹跳），不是步态形状先验，先验的腾空很短，留作"别蹦"的兜底。
        若也要归零，删掉下面注释掉的那行即可。
    * 本类**只用于训练**：回放/判读用现有 `Imgo2-basemove-rough-cmoe-play`（奖励不参与回放行为）。
    * `check_reward_overrides.py` 的 `cmoe-gaitfree` 链会把这五项报成"func 是自定义类但权重 0 ⇒
      死代码"——这是**有意保留**：配方离"重新开 shaping"只差一个权重数字。
    """

    def __post_init__(self):
        super().__post_init__()

        # ⚠️ 2026-09-28 冒烟测试抓到的真 bug：v3 起 `feet_gait` 在**父类**就已归零，于是父类的
        # `disable_zero_weight_rewards()` 会把它 `setattr(..., None)`（实现见 `velocity_env_cfg.py:738-744`：
        # "If the weight of rewards is 0, set rewards to None"）⇒ 子类再写
        # `self.rewards.feet_gait.weight = 0.0` 就是 `AttributeError: 'NoneType' object has no attribute 'weight'`
        # （训练**启动即崩**，不是跑起来才出问题）。五项统一加 None 守卫：父类已移除的跳过、未移除的归零。
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
        # self.rewards.lin_vel_z_l2.weight = 0.0  # ← 若连"别蹦"的兜底也要去掉，取消注释这一行

        # 上面这些归零发生在 `super().__post_init__()` 的 `disable_zero_weight_rewards()` **之后**，
        # 所以必须再跑一次：否则这五项会以 0 权重留在奖励管理器里参与计算与分项日志，
        # 与"归零即移除"的既有约定不一致（也会让 `Episode_Reward/*` 多出一堆恒 0 的键）。
        self.disable_zero_weight_rewards()
