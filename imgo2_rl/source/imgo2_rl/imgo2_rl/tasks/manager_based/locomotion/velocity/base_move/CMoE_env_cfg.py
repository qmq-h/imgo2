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
    gait_metric_trot = RewTerm(
        func=mdp.GaitReward,
        weight=1e-6,
        params={
            "std": math.sqrt(0.5),
            "command_name": "base_velocity",
            "max_err": 0.2,
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
            "std": math.sqrt(0.5),
            "command_name": "base_velocity",
            "max_err": 0.2,
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
    gait_metric_pace = RewTerm(
        func=mdp.GaitReward,
        weight=1e-6,
        params={
            "std": math.sqrt(0.5),
            "command_name": "base_velocity",
            "max_err": 0.2,
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

        # Reproduce the original +x obstacle-course layout at quadruped scale.
        self.scene.terrain.terrain_generator.size = (8.0, 4.0)
        sub_terrains = self.scene.terrain.terrain_generator.sub_terrains
        # 2026-09-24（用户要求）：上行台阶**单独占更大的列**，并做成不可能看错的台阶。
        # 原来 proportion=0.15（num_cols=20 时 3 列）且单级只有 2.5–8 cm，低等级下视觉上像缓坡
        # （用户回放时"没看见上行台阶"，只看到整段塌下去的下行）。现在：
        #   * proportion 0.15 → 0.20（num_cols=20 时 3 → 4 列；num_cols=10 时 2 列），
        #     多出来的一份从 random_rough 的 0.20 → 0.15 扣；
        #   * num_steps 4 → 6、step_height_range (0.025,0.08) → (0.03,0.10) ⇒ 单级 3–10 cm、
        #     总升高 18–60 cm、跨度 1.8 m（从 x=2.0 到 3.8，8 m tile 内）。
        # 2026-09-24（用户要求）：单级台阶提高到 10–20 cm 量级。
        # 取 (0.05, 0.20) 而非 (0.10, 0.20)——课程**下限必须是可学的**：level 0 就是 10 cm 的话，
        # 全新策略很可能直接卡死在台阶列（课程只会把它降级到 level 0，那里仍太难）。
        # 用户随后把上限改回 **0.15**（`step_height_range=(0.05, 0.15)`）：d=0.5 → 10.0 cm、
        # d=0.62 → 11.2 cm、d=1.0 → 15.0 cm ⇒ 课程中上段落在 10–15 cm，等效倾角 ≤25.8°。
        # 腿部可行性：大腿 0.22 + 小腿 0.206 = 最大伸展 0.426 m，站立 0.30 m；踏上 20 cm 时该腿收缩到
        # 0.10 m（伸展率 23%），几何可达，但 20 cm 更接近"跃上"（关节上限 23.7 N·m、抬身 20 cm ≈ 25 J）；
        # 参考 parkour 的 jump 障碍是 height (0.2, 0.46) m。
        # 台阶跨度仍是 6 级 × step_depth 0.30 = 1.8 m（x 2.0→3.8）；难度 1.0 时总升高 1.2 m（≈34°）。
        sub_terrains["pyramid_stairs"] = CMoETrackStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.05, 0.15),
            num_steps=6,
            ascending=True,
        )
        sub_terrains["pyramid_stairs_inv"] = CMoETrackStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.05, 0.15),
            num_steps=6,
            ascending=False,
        )
        # 2026-09-24（用户要求）：独立障碍块（`boxes`，横贯赛道的整宽矮块，侧面像一排栏杆）
        # 高度提高到 0.1–0.3 m 量级。与台阶（0.05–0.20）、沟（0.126–0.315）同一量级。
        # 取 (0.08, 0.30) 而非 (0.10, 0.30)：理由同 §28.1——课程下限必须可学（level 0 就 10 cm 易卡死）。
        # 0.30 m ≈ 0.95 体长（躯干 0.315 m），接近腿部最大伸展 0.426 m 的可跃范围；
        # 参考 parkour 的 jump 障碍 height (0.2, 0.46) m。长度与间距维持 step_length (0.18,0.30)、spacing 0.85。
        # 2026-09-24（用户要求）：「boxes 只给一个」⇒ proportion 0.15 → **0.05**
        # （num_cols=20 时 **3 列 → 1 列**；num_cols=10 时 2 列 → 1 列；仍 >0，所以
        # `free_terrain_names=("boxes",)` 的掩码在训练与 play 上都还有命中对象）。空出的 0.10 给 `gap`
        # （见下），其余各列数量不变（离线核对：`scripts/tools/check_terrain_columns.py`）。
        sub_terrains["boxes"] = CMoETrackStepTerrainCfg(
            proportion=0.05,
            step_height_range=(0.08, 0.30),
        )
        # 2026-09-24：让出份额给上行台阶（0.05）与新增的平地（0.05）——总比例仍为 1.0。
        sub_terrains["random_rough"].proportion = 0.10
        sub_terrains["hf_pyramid_slope"].proportion = 0.10
        sub_terrains["hf_pyramid_slope_inv"].proportion = 0.05  # 让出 0.05 给平地
        sub_terrains["gap"] = CMoETrackGapTerrainCfg(
            # 2026-09-24（用户要求）：`boxes` 砍到 1 列后空出的 0.10 给沟壑 ⇒ 0.20 → **0.30**
            # （num_cols=20 时 **4 → 6 列**；num_cols=10 时 2 → 3 列）。理由：沟壑是历史上唯一
            # 持续降级、也是最吃"课时"的一列；其余列数量保持不变。
            proportion=0.30,
            # 2026-09-24（用户决定）：沟宽按**身体长度**给。躯干 0.315 m ⇒ 0.4 L–1.0 L = 0.126–0.315 m。
            # 旧值 (0.08, 0.16) 只有 0.25–0.5 L，属"地板缝"（不需要跃起）；现在最宽的沟等于一个躯干长。
            # 课程仍在区间内插值：level 0 ≈ 0.126 m（≈ 旧配方学到的 0.13 m），level 9 ≈ 0.306 m。
            # platform_length_range 保持 (0.65, 0.95)：4 条 0.315 m 沟 + 最长平台在 8 m tile 内末沟止于 x≈5.91 m。
            gap_width_range=(0.126, 0.315),
            platform_length_range=(0.65, 0.95),
            first_gap_x=1.8,
            num_gaps=4,
        )

        # 2026-09-24（用户指出"似乎没有平地"）：加入一列纯平地。
        # 原地形集里最缓的只有 random_rough（1–6 cm 噪声），没有真正的平地；
        # 平地既是 trot 的"标称步态"学习面（真机录制基线也是在平地上），也方便部署/ sim2sim 对照。
        # MeshPlaneTerrainCfg 的 origin 是瓦片中心，±1 m 的 reset 不会出界。
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
        # 因此列全 8 类地形；`lin_pos_y`／`yaw_abs` 也改成**全局**（不再只作用于障碍列）。
        # 注意：本列表必须覆盖 `sub_terrains` 的**全部键**（漏项会静默退回全向命令）；
        # `scripts/tools/check_terrain_columns.py` 会做覆盖率校验。
        self.commands.base_velocity.forward_only_terrain_names = (
            "pyramid_stairs",
            "pyramid_stairs_inv",
            "boxes",
            "random_rough",
            "hf_pyramid_slope",
            "hf_pyramid_slope_inv",
            "gap",
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
        self.rewards.undesired_contacts.weight = -0.5

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
        self.rewards.flat_orientation_l2.weight = -0.1
        # 2026-09-24 晚（用户："都还是蹦蹦跳跳的走的"）：**按地形豁免地恢复竖直速度罚**。
        # 形状：trot 列上恢复（压弹跳），`boxes`/`gap` 豁免（那里需要爆发式跃起，原清零理由是
        # "会与跃起对抗"——掩码后这个理由不再成立）。权重先取 **−2.0**（＝PPO rough 原值；探针
        # `gait_bounce_<地形>` 会给出逐列的 vz RMS，可按实测再调）。
        self.rewards.lin_vel_z_l2.func = mdp.MaskedLinVelZ
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.lin_vel_z_l2.params["free_terrain_names"] = ("boxes", "gap")
        self.rewards.ang_vel_xy_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = 0.0
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
        self.rewards.feet_air_time.weight = 0.3
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
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.ang_vel_xy_l2.weight = -0.05

        self.rewards.feet_height.weight = 0.0
        # `feet_height_body` 同理不清零——已在上方设为 −5.0（MaskedFeetHeightBody）。
        self.rewards.feet_slide.weight = 0.0
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
                "relaxed_terrain_names": ("pyramid_stairs", "pyramid_stairs_inv", "boxes"),
                # 逐列步态度量（三成对方式＝分类器）：日志会出 `gait_trot_<地形>` / `gait_bound_<地形>`
                # / `gait_pace_<地形>`，以及全局 `gait_<标签>_mean`。用来验证"除 boxes/gap 外倾向 trot"。
                "gait_metric_terms": (
                    ("trot", "gait_metric_trot"),
                    ("bound", "gait_metric_bound"),
                    ("pace", "gait_metric_pace"),
                    ("bounce", "diag_bounce"),        # 逐列 vz 均方（开方＝vz RMS）
                    ("height", "base_height_l2"),     # 逐列 (base 高度误差)²（开方＝RMS 误差）
                ),
                # 2026-09-24（用户）：**台阶与 boxes 用 0.50、其余仍 0.80**。
                # ⚠️ 0.50 在这三类上**等价于取消跟踪门控**（整回合平均跟踪核实测 mean≈0.78、min≈0.69
                # ⇒ 没有任何回合会低于 0.50）⇒ 它们回到"只看前进进度（走够 4 m）"的旧判据。
                # 依据：旧判据在这三类上曾把 A 跑推到 level 6（反楼梯 6.06／boxes 5.99），而"能不能过障碍"
                # 本来就该由进度管；跟踪门控更适合容易地形（那里"糊弄过去"才是失败模式）。
                "tracking_move_up_relaxed": 0.50,
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
