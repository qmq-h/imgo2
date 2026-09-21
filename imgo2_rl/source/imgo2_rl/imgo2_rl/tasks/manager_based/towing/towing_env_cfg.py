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

# 机器人初始高度。训练侧的 `IMGO2_CFG.init_state.pos` 是 0.35 m，但**实测该策略的站高是
# 0.284 m**（2026-09-20 拖曳运行的 `steady_robot_z_m`），从 0.35 m 出生会先落下 6.6 cm，
# 落地时向前窜动 0.65 m/s —— 那一窜会把绳拉直（实测 73–86 N）并把小车甩出去，
# 看上去就像「小车自己有初速度」。故默认改为 0.30 m（贴近实测站高，留 1.6 cm 余量），
# 仍可用 `--spawn-height` 覆盖；README 里「高/低站高分支」的疑虑按 0.35 那一档未复现。
ROBOT_SPAWN_HEIGHT_M = 0.30


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
