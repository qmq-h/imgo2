"""P4 robot + cart towing scene: frozen locomotion policy pulls a passive cart.

与 P1/P2 同一套风格（`SimulationContext + InteractiveScene`，不经 `ManagerBasedRLEnv`）：
拖曳实验只需要「读状态 → 施力 → 步进 → 记录」，不需要 RL 环境管理器。机器人本体沿用
训练侧配置 `IMGO2_CFG`（不改 locomotion 的任何参数），只把任务把它当冻结策略用。

布局约定（与绳的物理一致）：机器人在前、小车在后，两者都朝 +x。
`cart.urdf` 的挂点在车体 `+0.25 m`（前轴之前），小车在后时该点朝向机器人 ⇒ 绳从
机器人的后挂点连到小车的前挂点，受力方向与拖曳一致。
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from imgo2_rl.assets.imgo2 import IMGO2_CFG

# 机器人后挂点（base 坐标系）。base 的碰撞箱长 0.315 m（`urdf/imgo2.urdf`）⇒ 后表面在
# x = -0.1575 m，取 -0.16 略过后表面。**工程取值，没有实测挂点**；换成实物尺寸时只改这里。
# 注意：`set_external_force_and_torque(positions=...)` 的坐标系是**连杆坐标系**，所以
# 这个常量可以直接当作用点传进去，力臂由仿真自己算，不必手写 offset×F。
ROBOT_ATTACHMENT_OFFSET_M = (-0.16, 0.0, 0.0)

# 机器人初始高度。**回到训练侧 `IMGO2_CFG` 的 0.35 m** —— 2026-09-21 曾按「实测站高 0.284 m，
# 所以贴近它出生更稳」的推理把它改成 0.30 m，**结果是错的**，实跑直接把机器人拽翻：
#
#   出生高度   station 段机器人峰值 vx   station 段位移   结果
#   0.30 m     1.719 m/s                0.46 m          绳被拉直 → 236.7 N 猛拽 → pitch 1.31 rad 翻倒
#   0.35 m     0.849 m/s                0.063 m         绳未被拉直，拖曳验收通过
#
# 机理：出生后足端离地、机器人下落期间策略的启动动作会摆动腿部，落地时把腿的推力转成
# 前向窜动；0.30 m 下落时间更短，落地正好卡在启动动作中间，窜动反而大 7 倍。
#
# 供后续选值参考的模型量：名义站姿（hip 0 / thigh 0.87 / shank -1.82）下足端最低点比 base
# 低 **0.2685 m**（由 urdf/imgo2.urdf 的腿部链 FK 算得），所以 0.35 m 出生时足端离地
# 8.2 cm、0.30 m 时离地 3.2 cm —— 两者都是「落下」，差别在下落时长。要真做到「贴地出生」
# 应取 0.2685 + 少量余量，但**该档未验证**。`--spawn-height` 可覆盖，summary 里的
# `station_robot_travel_m` 用来比较各档的启动窜动。
ROBOT_SPAWN_HEIGHT_M = 0.35


@configclass
class TowSceneCfg(InteractiveSceneCfg):
    """平地 + 机器人 + 被动小车；机器人配置与训练侧一致。"""

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.8, restitution=0.0,
                friction_combine_mode="average", restitution_combine_mode="min")))
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000))
    robot: ArticulationCfg = IMGO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cart: ArticulationCfg | None = None
    wheel_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Cart/wheel_.*", update_period=0.0, history_length=1)
    # 车斗接触：`base_link` 的碰撞箱 z 区间是 0.10~0.20 m，**永远不会碰地**
    # （小车静止高度 0.15 m、轮半径 0.08 m），所以它的接触力非零只可能是机器人压上来。
    # 这是「小车是否追到机器人」的独立见证：不依赖 FK、不依赖挂点间距，直接测力。
    # 允许撞击发生是有意的设置（用户 2026-09-21 明确），这里只把它测出来、不阻止。
    deck_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Cart/base_link", update_period=0.0, history_length=1)
