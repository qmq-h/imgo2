"""Manager terms for upper towing RL.

架构（2026-10-10 起，**双头**）。链路顺序是**两层串联**，偏移进的是底层的**输入**，
残差加的是底层的**输出**：

    task_command ──(速度头: +有界偏移, 裁进 AMP 包络 −1.0…1.5)──→ loco_command
                                                                      │
                                                          （冻结 AMP 策略推理）
                                                                      ↓
                                                             冻结关节目标
                                                                      │
                                       (+ 12 维关节残差 → PD 位置目标) ┘
                                                                      ↓
                                                              set_joint_position_target

- ``task_command``：脚本调度（+ 可选 STOP ramp），**只给奖励**（`velocity_tracking_exp` 等）；
- ``loco_command``：``task_command + 有界偏移``，裁进冻结策略训练包络后**送策略 + 进 actor 帧**。

**防作弊红线**：奖励只能读 ``task_command``，**绝不能**读 ``loco_command``（含策略自己的偏移），
否则策略只要把偏移开大就能自己给自己发目标。

**两头不独立（动作空间存在别名/冗余）**：上层同时影响底层输入与输出，同一个关节目标变化
既可由偏移（改变步态）也可由残差（改变关节角）达成 ⇒ credit assignment 更难、两头可能互相打架。
缓解：偏移是"步态级"旋钮（秒级、建议低通/限速率），残差是"单步级"（20 Hz）；必要时只在特定相位
放开偏移。当前 v3 默认两条脚本开关全关，偏移在 `elapsed >= tow_start_s` 后全程可用。

观测／奖励函数对特权数据的取舍是刻意显式的。本文件是**目标函数与动作适配器**的唯一定义处；
任务注册见同目录 ``__init__.py``，运行验收清单见 ``docs/towing_training_prep_2026-09-22.md``。
"""

import math

import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

from .upper_logic import CMD_ACTION_DIM, UpperActionSpec
from .mdp.connection_grid import ELASTIC_KC, GRID_SIZE, env_spec, is_full_grid
from .mdp.episode_geometry import SETTLE_TIME_S, STOP_DISTANCE_M
from .mdp.profile_torch import profile_height_tensor
from .mdp.slope_geometry import (
    BACK_M, BOUNDARY_MARGIN_M, FORWARD_M, HALF_WIDTH_M)
from .mdp.rope import point_velocity
from .mdp.rope_model import (
    BodyProperties,
    MultiRopeModel,
    attachment_horizontal_gap,
    make_rope_model,
    world_inverse_inertia,
)
from .utils.low_level_policy import FrozenLowLevelPolicy, parts_from_robot_state
from .utils.policy_cfg import get_policy


def _term(env):
    return env.action_manager.get_term("high_level_velocity")


def yaw_from_quat(quat):
    """从四元数取**世界 yaw**（rad）。

    ⚠ **顺序陷阱**：Isaac Lab 的 `isaaclab.utils.math.euler_xyz_from_quat` 返回
    **(roll, pitch, yaw)**（源码 Returns 写明 roll-pitch-yaw），所以 yaw 是**第三个**元素。

    2026-10-09 修复：`heading_deviation` 原先写成 `yaw, _pitch, _roll = euler_xyz_from_quat(...)`，
    实际拿到的是 **roll** ⇒ `yaw_heading`（−2.0）一直在罚**横滚**偏差，而朝向只被
    `tracking_velocity` 的 yaw **角速度**项间接约束——角速度归零 ≠ 朝向不漂，所以
    "开始就在自转"那类问题不会被这一项真正压住。抽成单一入口避免再写错。
    """
    _roll, _pitch, yaw = math_utils.euler_xyz_from_quat(quat)
    return yaw


class HierarchicalVelocityAction(ActionTerm):
    cfg: "HierarchicalVelocityActionCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._cart = env.scene[cfg.cart_asset_name]
        self._collision_sensors = tuple(env.scene.sensors[name] for name in cfg.collision_sensor_names)
        # 冻结策略契约先加载：残差尺度直接取它的 `action_scale`（与底层动作同量纲），
        # 动作维数取它的关节数（AMP 为 12）。见 `UpperActionSpec` 的说明。
        self._policy_cfg = get_policy(cfg.policy_name)
        self._joint_action_dim = len(self._policy_cfg.joint_names)
        if len(self._policy_cfg.action_scale) != self._joint_action_dim:
            raise ValueError(
                f"{cfg.policy_name} 契约的 action_scale {len(self._policy_cfg.action_scale)} 项与 "
                f"joint_names {self._joint_action_dim} 项不一致")
        self._action_spec = UpperActionSpec(
            residual_scale=tuple(self._policy_cfg.action_scale),
            control_dt=cfg.upper_control_dt,
            cmd_offset_scale=cfg.cmd_offset_scale,
            offset_min=cfg.offset_min,
            offset_max=cfg.offset_max,
        )
        self._action_spec.validate()
        self._residual_scale = torch.tensor(
            self._policy_cfg.action_scale, dtype=torch.float32, device=env.device)
        # 动作 13 维 = 1 维 vx 偏移 + 12 维关节残差（`upper_logic.CMD_ACTION_DIM`）。
        # 这里展开写而不用 `self.action_dim`（属性在类上已定义，但那是 `init` 顺序守卫里的
        # 未赋值名字；展开可让守卫保持严格，不新增白名单）。
        self._raw = torch.zeros(env.num_envs, CMD_ACTION_DIM + self._joint_action_dim,
                                device=env.device)
        self._processed = torch.zeros_like(self._raw)
        self._previous = torch.zeros_like(self._raw)
        # 12 维关节位置残差（**策略关节顺序**）。`apply_actions` 每次刷新冻结策略输出时
        # 把它加到 `joint_targets` 上；两次上层更新之间保持不变。
        self.delta_joint_pos = torch.zeros(env.num_envs, self._joint_action_dim, device=env.device)
        # **任务指令**（脚本调度 + 可选 STOP ramp）：只给奖励当参考量（`velocity_tracking_exp`）。
        # 与 `loco_command` 分开是防作弊红线：策略的偏移只进 `loco_command`，不许污染奖励参考。
        self.task_command = torch.zeros(env.num_envs, 3, device=env.device)
        # 实际送给冻结底层策略的速度指令 = `task_command` + 有界 vx 偏移（出生段不叠加偏移）。
        self.loco_command = torch.zeros(env.num_envs, 3, device=env.device)
        self.tow_speed = torch.full((env.num_envs,), cfg.initial_tow_speed, device=env.device)
        self.tow_start_s = torch.full((env.num_envs,), cfg.tow_start_s, device=env.device)
        self.start_progress = torch.zeros(env.num_envs, device=env.device)
        # STOP 相位状态（2026-10-09 恢复）：走到 `stop_distance_m`（必须已越过坡）那一刻把指令
        # 置零，并记下**该时刻**与**该时刻的 x**——post_stop 奖励就靠这两个量定相位。
        # `stop_time_s` 初值 +inf ⇒ `elapsed_s >= stop_time_s` 恒假，未停车前该项精确为 0。
        # 2026-10-10：`post_stop_distance`（`extra_distance`）已从奖励表删除（见其 docstring），
        # `stop_origin_x` 保留——它仍是「停车那一拍」的唯一记录，供复用/诊断。
        self.stop_time_s = torch.full((env.num_envs,), math.inf, device=env.device)
        self.stop_origin_x = self._asset.data.root_pos_w[:, 0].clone()
        self._was_stopped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.last_loco_action = torch.zeros(env.num_envs, 12, device=env.device)
        # [clearance, tension, extension, taut]. Collision is deliberately separate: tautness is
        # a rope state and must never double as a contact flag.
        self.rope_state = torch.zeros(env.num_envs, 4, device=env.device)
        # 绳子对机器人的力，**base 机体系 3 维**（x 前后 / y 左右 / z 竖直）。
        # 2026-10-08 由 2 维改为 3 维：两挂点高差 Δz=0.17 m，绷紧时 |Fz| 占张力的
        # 0.17/L = 8.5%（绳 L=1.5）… 34%（杆 L=0.5），且 Fz 在 -0.16 m 挂点上产生俯仰力矩，丢掉它对 actor／惩罚都不可见。
        self.towing_force_b = torch.zeros(env.num_envs, 3, device=env.device)
        # 【拉力变化率惩罚的状态（2026-10-10，TOW-25；物理理由见 `towing_force_rate_penalty`）】
        # `prev_force_mag` = **上一个控制步**结束时的 ‖F‖（N）。只在 `process_actions`
        # 每个控制步（20 Hz）更新一次，**不在 200 Hz 物理子步里**——20 Hz 有限差分与
        # 奖励/动作同节拍，才有确定的「上一拍」。
        self.prev_force_mag = torch.zeros(env.num_envs, device=env.device)
        #: 本拍的「上一拍」样本是否真实存在：刚 `reset` 后的第一拍没有上一拍 ⇒ `False`
        #: ⇒ 该项精确为 0（与仓库其它项「spawn 精确为 0」的纪律一致）。它由
        #: `_force_rate_tick_seen` 决定，见 `process_actions` 里的注释。
        self.force_rate_has_prev = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        #: `process_actions` 内部用：本控制步是否已经开始过（决定**下一拍**的
        #: `force_rate_has_prev`）。`reset()` 清零 ⇒ 每回合的第一拍 rate 定义为 0。
        self._force_rate_tick_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        #: **200 Hz 诊断**：本控制步内物理子步的 max |Δ‖F‖| / dt_phys（N/s）。
        #: 每物理子步取最大、由 `obs_force_rate_max` **读后清零** ⇒ 每控制步一个值。
        #: 存在的理由：20 Hz 的有限差分看不到 5 ms 级的冲量尖峰（见 `obs_force_rate_max`）。
        self.force_rate_max = torch.zeros(env.num_envs, device=env.device)
        self._phys_prev_force_mag = torch.zeros(env.num_envs, device=env.device)
        self.cart_collision = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.cart_mass = torch.full((env.num_envs, 1), cfg.initial_cart_mass, device=env.device)
        self.ground_friction = torch.full((env.num_envs, 1), cfg.initial_ground_friction, device=env.device)
        self.wheel_damping = torch.full((env.num_envs, 1), cfg.initial_wheel_damping, device=env.device)
        # A subset of environments acts as a zero-load locomotion/stopping anchor. The cart
        # articulation still exists in the replicated scene but is parked laterally and masked
        # out of towing physics, safety costs, termination, and decoder supervision.
        self.cart_present = torch.ones(env.num_envs, 1, dtype=torch.bool, device=env.device)
        # 场景是「列 × 行」确定性网格（见 `mdp/connection_grid.py`）：列 = 连接类型/弹性档、
        # 行 = 连接长度。类型与长度不再逐 env 随机，整段训练按 env index 固定映射；因此这三
        # 个张量在 `__init__` 里一次算好，之后**不再改写**（模型直接持有它们的视图）。
        # 0 = compliant（弹性绳）, 1 = inextensible（低弹性绳）, 2 = rigid（双边球铰连杆）。
        if not is_full_grid(env.num_envs):
            # 允许非整倍数（例如 4 环境冒烟）跑通，但只覆盖网格前缀、不是平衡设计。
            print(f"[WARN] num_envs={env.num_envs} 不是网格 {GRID_SIZE} 的整数倍，"
                  f"只覆盖网格前缀；正式训练请用 {GRID_SIZE}（40 列 × 20 行）的整数倍")
        specs = [env_spec(index) for index in range(env.num_envs)]
        # 每条 lane 的坡度量级（0 / 5 / 10 deg）。**出生点在剖面的平地段起点**，所以出生
        # 坐标系就是世界系（+x 前、+z 上）：`terrain_tangent_w` / `terrain_normal_w` 不再是
        # 逐 env 的恒定坡面系，而是同一组基；lane 内部的上/下坡由 `slope_geometry.profile_*`
        # 按当前位置算（行程 / 离面高度 / 局部坡度）。
        self.hill_grade_deg = torch.tensor(
            [spec["slope_degrees"] for spec in specs], dtype=torch.float32, device=env.device)
        self.terrain_tangent_w = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(
            env.num_envs, 3).clone()
        self.terrain_normal_w = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(
            env.num_envs, 3).clone()
        self.rope_model_id = torch.tensor(
            [[spec["model_index"]] for spec in specs], dtype=torch.float32, device=env.device)
        # 连接长度：绳是 L0、刚体是杆长 L（同一行的数值相同）。
        self.connection_length = torch.tensor(
            [[spec["length"]] for spec in specs], dtype=torch.float32, device=env.device)
        # spawn 时的目标三维挂点距：绳 = SLACK_RATIO·L（留松弛）、刚体 = L。摆放小车时用。
        self.initial_distance = torch.tensor(
            [[spec["initial_distance"]] for spec in specs],
            dtype=torch.float32, device=env.device)
        # **出生瞬间的「后表面 → 车斗前表面」间隙**（逐 env，摆位几何的解析解）。
        # 与 `reset_towing_episode` 的摆位同源：三维挂点距被钉在 `initial_distance` 上，
        # 水平分量 = `attachment_horizontal_gap(target, Δz)`（Δz = 两挂点高差；两个挂点的
        # z 偏移都是 0，所以 Δz = 机器人/小车的出生高度之差，直接取各自的 `init_state.pos[2]`
        # ——与 `reset_towing_episode` 读 `default_root_state[:, 2]` 是同一个值，但不依赖数据
        # 缓冲区的初始化时机）；再减去两表面相对挂点的固定偏移
        # `−robot_rear_surface_x − robot_attachment_x + cart_attachment_x − cart_front_surface_x`
        # （= 0.0025 m，全部从 cfg 取，不手抄）。
        # 2026-10-10：`min_clearance_violation` 的阈值改成 `它 − deadband_m`（出生几何口径 +
        # 绝对死区），所以这里一次算好缓存；出生位姿固定（`robot_x/y/yaw_range` 全 0）⇒ 逐回合不变。
        attachment_height_difference = (
            self._asset.cfg.init_state.pos[2] - self._cart.cfg.init_state.pos[2])
        self.spawn_clearance = (
            attachment_horizontal_gap(self.initial_distance[:, 0],
                                      delta_z=attachment_height_difference)
            - cfg.robot_rear_surface_x - cfg.robot_attachment[0]
            + cfg.cart_attachment[0] - cfg.cart_front_surface_x)
        # 弹性档 k/c：非弹性环境填第一档占位值（会被 MultiRopeModel 的掩码忽略），
        # 但不能填 0——`CompliantRope` 要求 k > 0。
        placeholder_k, placeholder_c = ELASTIC_KC[0]
        self.rope_stiffness = torch.tensor(
            [[spec["stiffness"] if spec["stiffness"] is not None else placeholder_k]
             for spec in specs], dtype=torch.float32, device=env.device)
        self.rope_damping = torch.tensor(
            [[spec["damping"] if spec["damping"] is not None else placeholder_c]
             for spec in specs], dtype=torch.float32, device=env.device)
        if abs(cfg.physics_dt - env.physics_dt) > 1.0e-9:
            raise ValueError(
                f"physics_dt {cfg.physics_dt} does not match environment {env.physics_dt}")
        if abs(cfg.upper_control_dt - env.step_dt) > 1.0e-9:
            raise ValueError(
                f"upper_control_dt {cfg.upper_control_dt} does not match environment {env.step_dt}")
        if abs(cfg.low_level_control_dt - self._policy_cfg.control_dt) > 1.0e-9:
            raise ValueError(
                f"low_level_control_dt {cfg.low_level_control_dt} does not match frozen policy "
                f"period {self._policy_cfg.control_dt}")
        self._policy = FrozenLowLevelPolicy(self._policy_cfg, device=env.device)
        self._physics_step = 0
        self._held_joint_targets = self._asset.data.default_joint_pos.clone()
        asset_names = list(self._asset.joint_names)
        self._policy_to_asset = torch.tensor(
            [asset_names.index(name) for name in self._policy_cfg.joint_names],
            dtype=torch.long, device=env.device)
        # 冻结策略**自己**输出的关节位置目标（= 「底层期望」/ 稳定步态），供
        # `low_level_position_error_l2` 的 `reference="frozen"` 分支当参考量。
        # ⚠ 必须在 `_policy_to_asset` **之后**初始化（要用它取策略关节顺序）——
        # 2026-10-09 曾插到前面，训练一启动就 `AttributeError: ... has no attribute
        # '_policy_to_asset'`（本机无 Isaac Lab，离线测试没覆盖到这个顺序）。
        self.loco_joint_targets = self._asset.data.default_joint_pos[
            :, self._policy_to_asset].clone()
        robot_body_ids, _ = self._asset.find_bodies([cfg.robot_body_name])
        cart_body_ids, _ = self._cart.find_bodies([cfg.cart_body_name])
        wheel_joint_ids, _ = self._cart.find_joints(list(cfg.cart_wheel_joint_names), preserve_order=True)
        wheel_body_ids, _ = self._cart.find_bodies(list(cfg.cart_wheel_body_names), preserve_order=True)
        if (len(robot_body_ids) != 1 or len(cart_body_ids) != 1
                or len(wheel_joint_ids) != 4 or len(wheel_body_ids) != 4):
            raise RuntimeError("upper towing adapter requires one robot base, one cart base, and four wheels")
        # 足端 body：按**接触传感器**的 `body_names` 顺序映射到资产 body id。
        # Isaac Lab 的标准写法（`isaaclab_tasks/.../locomotion/velocity/mdp/rewards.py:feet_slide`）
        # 用两个 SceneEntityCfg 各自解析 body_ids 后按下标配对，隐含「传感器与资产同序」的假设；
        # 这里按名字映射，顺序不一致也能正确配对（配错会静默把 A 脚的速度算到 B 脚上）。
        if cfg.foot_contact_sensor_name not in env.scene.sensors:
            raise RuntimeError(
                f"场景里没有足端接触传感器 {cfg.foot_contact_sensor_name!r}"
                f"（现有：{sorted(env.scene.sensors.keys())}）")
        foot_sensor = env.scene.sensors[cfg.foot_contact_sensor_name]
        unknown = [name for name in foot_sensor.body_names if name not in set(self._asset.body_names)]
        if unknown:
            raise RuntimeError(f"足端接触传感器的 body {unknown} 不在机器人资产里")
        # 数量也必须是 4（四条腿的 `*_FOOT`）。少了不能静默通过：`body_lin_vel_w[:, []]`
        # 会得到 (N,0,2)、求和恒为 0，`feet_slide` 就变成一项永远为 0 的假奖励。
        if len(foot_sensor.body_names) != 4:
            raise RuntimeError(
                f"足端接触传感器 {cfg.foot_contact_sensor_name!r} 解析出 "
                f"{len(foot_sensor.body_names)} 个 body（{list(foot_sensor.body_names)}），"
                f"应为 4 个（FL/FR/RL/RR_FOOT）")
        self._foot_body_ids = torch.tensor(
            [self._asset.body_names.index(name) for name in foot_sensor.body_names],
            dtype=torch.long, device=env.device)
        ratio = cfg.low_level_control_dt / cfg.physics_dt
        if abs(ratio - round(ratio)) > 1.0e-6:
            raise ValueError("low_level_control_dt must be an integer multiple of physics_dt")
        self._robot_body_id = robot_body_ids[0]
        self._cart_body_id = cart_body_ids[0]
        self._wheel_joint_ids = wheel_joint_ids
        self._wheel_body_ids = wheel_body_ids
        self._robot_attach = torch.tensor(cfg.robot_attachment, device=env.device).view(1, 3)
        self._cart_attach = torch.tensor(cfg.cart_attachment, device=env.device).view(1, 3)
        # 三套模型都按**逐 env** 的长度/弹性张量构造（网格里每个 env 不同）。
        compliant = make_rope_model("compliant", rest_length=self.connection_length[:, 0],
                                    stiffness=self.rope_stiffness[:, 0],
                                    damping=self.rope_damping[:, 0])
        inextensible = make_rope_model("inextensible", rest_length=self.connection_length[:, 0],
                                       position_gain=cfg.rope_position_gain,
                                       max_correction_rate=cfg.rope_max_correction_rate)
        rigid = make_rope_model("rigid", rest_length=self.connection_length[:, 0],
                                position_gain=cfg.rigid_position_gain,
                                max_correction_rate=cfg.rigid_max_correction_rate)
        self._rope_model = MultiRopeModel(
            models=(compliant, inextensible, rigid), model_ids=self.rope_model_id[:, 0])

        # 质量与惯量只在 reset 时被随机化，但同一回合内的每个物理步都要用。原来的实现每个
        # 物理步都 `root_physx_view.get_masses()/get_inertias()` 回读一次——那是一次 GPU→CPU→
        # GPU 的往返，会打断流水。4096 环境下每轮 `480 步 × 4096 env`，这些回读各约 196 万次，
        # 实测把每轮固定开销推到约 4.6 s（与显卡算力无关，见 docs/towing_observability_2026-09-22.md）。
        # 这里缓存常量部分；`reset_towing_episode` 写完新质量后调用 `_invalidate_mass_cache()`。
        self._robot_inertia_diag = None
        self._robot_mass_total = None
        self._cart_inertia_diag = None
        self._cart_mass_total = None
        self._cart_wheel_inertia = None

    def _invalidate_mass_cache(self):
        """丢弃质量／惯量缓存。凡 `set_masses`／`set_inertias` 之后必须调用。"""
        self._robot_inertia_diag = None
        self._robot_mass_total = None
        self._cart_inertia_diag = None
        self._cart_mass_total = None
        self._cart_wheel_inertia = None

    def _refresh_mass_cache(self):
        """从 PhysX 读一次质量／惯量并缓存（每回合每环境约一次，而不是每物理步）。"""
        robot_inertias = self._asset.root_physx_view.get_inertias().to(self.device)
        self._robot_inertia_diag = (
            robot_inertias[:, self._robot_body_id, 0].clone(),
            robot_inertias[:, self._robot_body_id, 4].clone(),
            robot_inertias[:, self._robot_body_id, 8].clone(),
        )
        self._robot_mass_total = (
            self._asset.root_physx_view.get_masses().to(self.device).sum(dim=1)).clone()
        cart_inertias = self._cart.root_physx_view.get_inertias().to(self.device)
        self._cart_inertia_diag = (
            cart_inertias[:, self._cart_body_id, 0].clone(),
            cart_inertias[:, self._cart_body_id, 4].clone(),
            cart_inertias[:, self._cart_body_id, 8].clone(),
        )
        self._cart_mass_total = (
            self._cart.root_physx_view.get_masses().to(self.device).sum(dim=1)).clone()
        wheel_inertia = sum(cart_inertias[:, body_id, 4] for body_id in self._wheel_body_ids)
        self._cart_wheel_inertia = wheel_inertia.clone()

    @property
    def action_dim(self):
        """13：1 维 vx 偏移（cmd vel 头）+ 12 维关节残差（策略关节顺序）。"""
        return CMD_ACTION_DIM + self._joint_action_dim

    @property
    def raw_actions(self):
        return self._raw

    @property
    def processed_actions(self):
        return self._processed

    def process_actions(self, actions):
        """每上层控制步（50 ms）更新一次：脚本速度指令 + 1 维偏移 + 12 维关节残差。

        动作 ``u = cat(u_cmd(1), u_joint(12))``（`upper_logic.CMD_ACTION_DIM`）。相位
        （2026-10-09 恢复三段制，用户要求）：

            0 ── settle（指令 0，静止稳定）── tow（指令 = tow_speed）── STOP（指令 0）
                tow_start_s                                 stop_distance_m（**已越过坡**）

        STOP 由**进度**触发而不是时间：`progress` 沿 lane 累计，到达 `stop_distance_m`
        就把指令置零；同时记下 `stop_time_s`（该拍的时间）与 `stop_origin_x`（该拍的 x），
        供 `post_stop_towing_force`／`post_stop_distance` 定相位。用进度而非固定时间的原因：
        最慢速度下走到坡出口（9.0 m）就要约 23 s，固定时间阈值无法保证"越过坡之后"。

        **两个命令解耦（2026-10-10，防作弊红线）**：

        - ``task_command`` = 脚本调度（+ 可选 ramp）：**只给奖励**当参考量；
        - ``loco_command`` = ``clamp(task_command + 有界偏移, AMP 包络)``：给冻结策略、进 actor 帧。

        偏移只在 ``elapsed_s >= tow_start_s`` 后生效（出生段不许推机器人）；STOP 之后
        ``task_command`` 归零，但偏移头**仍然生效** —— 这正是「停机后继续走两步」的表达口。
        合成后的 vx 一律裁进冻结 AMP 策略的训练包络 ``amp_vx_range = (−1.0, 1.5)``
        （`amp_env_cfg` 的 `lin_vel_x`）：牵引段本来就顶在上界 1.5，超出即 OOD。
        """
        elapsed_s = self._env.episode_length_buf * self._env.step_dt
        progress = ((self._asset.data.root_pos_w - self._env.scene.env_origins)
                    * self.terrain_tangent_w).sum(dim=1) - self.start_progress
        crossed = progress >= self.cfg.stop_distance_m
        newly_stopped = crossed & ~self._was_stopped
        if bool(newly_stopped.any()):
            self.stop_time_s[newly_stopped] = elapsed_s[newly_stopped]
            self.stop_origin_x[newly_stopped] = self._asset.data.root_pos_w[newly_stopped, 0]
        self._was_stopped |= crossed
        # ---- 0) 拉力变化率（TOW-25）的「上一拍」缓存：与 `_previous/_processed` 同一处，
        #         每个控制步一次；**不在** `_apply_towing_physics` 的 200 Hz 子步里更新 ----
        # 顺序即语义：
        #   a) `force_rate_has_prev` 取**上一拍结束时**的 `_force_rate_tick_seen`——本拍能否
        #      算 rate 由它决定（刚 reset 的第一拍为 False ⇒ rate 定义 0，精确不罚）；
        #   b) `prev_force_mag` 取**当前** `towing_force_b`，即上一个控制步最后一个物理子步
        #      留下的力（刚 reset 时被清零 ⇒ 0）；
        #   c) 最后置位 `_force_rate_tick_seen`，让下一拍生效。
        # 注意 b) 必须在 c) 之前、且 a) 必须在 c) 之前：写反会让第一拍就拿到「上一拍」。
        self.force_rate_has_prev.copy_(self._force_rate_tick_seen)
        self.prev_force_mag.copy_(torch.linalg.vector_norm(self.towing_force_b, dim=1))
        self._force_rate_tick_seen.fill_(True)
        # ---- 1) 先更新本拍动作（偏移要读**本拍**的 u_cmd，不能读上一拍）----
        self._previous.copy_(self._processed)
        self._raw.copy_(actions)
        self._processed.copy_(actions.clamp(-1.0, 1.0))
        # 归一化关节残差 → 逐关节位置增量（rad）。尺度取冻结策略的 action_scale，
        # 因此等价于在底层动作空间上叠一个同量纲偏移。
        self.delta_joint_pos.copy_(
            self._processed[:, CMD_ACTION_DIM:] * self._residual_scale)
        # ---- 2) 任务指令：脚本调度 + 可选 STOP ramp（ramp_s = 0 时与旧口径逐位一致）----
        towing = (elapsed_s >= self.tow_start_s) & ~self._was_stopped
        scripted_vx = torch.where(towing, self.tow_speed, torch.zeros_like(self.tow_speed))
        if self.cfg.stop_command_ramp_s > 0.0:
            # STOP 之后 ramp_s 秒内把脚本 vx 从 tow_speed 线性降到 0（默认 0.0 = 关闭）。
            # `stop_time_s` 未停车时是 +inf，但这里被 `self._was_stopped` 门控住。
            fraction = (1.0 - (elapsed_s - self.stop_time_s) / self.cfg.stop_command_ramp_s)
            fraction = fraction.clamp(0.0, 1.0)
            scripted_vx = torch.where(self._was_stopped, self.tow_speed * fraction, scripted_vx)
        self.task_command.zero_()
        self.task_command[:, 0] = scripted_vx
        # 横向/朝向由 **PD 外环**给出（用户 2026-10-09 决定：保持中线和 heading 用 PD，
        # 不让策略学）。三个通道本来就是冻结策略的 (vx, vy, ω) 指令，所以不改任何观测/动作维数；
        # 而且 `loco_command` 是 actor 的观测项，PD 的意图对策略可见，它可以据此配合。
        if self.cfg.lane_keeping:
            self.task_command[:, 1] = self._lane_keeping_vy()
            self.task_command[:, 2] = self._lane_keeping_wz()
        # ---- 3) 送冻结策略的指令 = 任务指令 + 有界偏移，再裁进 AMP 训练包络 ----
        # 两层限幅，顺序不能反：
        #   a) 偏移本身限在头权限 [offset_min, offset_max]（**不是**把和裁到这个区间：
        #      脚本速度 0.5–1.5，裁和会把牵引段砍到 0.6 而奖励参考仍是 1.5）；
        #   b) **和**再限在冻结 AMP 策略的训练包络 amp_vx_range = (−1.0, 1.5)
        #      （`amp_env_cfg` 的 lin_vel_x）。超出即 OOD，步态会退化。
        # 因为包络覆盖整个脚本速度范围，u_cmd = 0 时 clamp 是恒等的 ⇒ 与旧口径逐位一致。
        self.loco_command.copy_(self.task_command)
        offset = self._processed[:, :CMD_ACTION_DIM] * self.cfg.cmd_offset_scale
        offset = offset.clamp(self.cfg.offset_min, self.cfg.offset_max)
        offset_active = (elapsed_s >= self.tow_start_s).unsqueeze(1)
        self.loco_command[:, :CMD_ACTION_DIM] += torch.where(
            offset_active, offset, torch.zeros_like(offset))
        amp_vx_min, amp_vx_max = self.cfg.amp_vx_range
        self.loco_command[:, :CMD_ACTION_DIM] = self.loco_command[:, :CMD_ACTION_DIM].clamp(
            amp_vx_min, amp_vx_max)

    def _lane_keeping_vy(self):
        """横向 PD → `vy` 指令：把机器人压在 lane 中线（lane 系 `y = 0`）。

        公式与测量台 `play_towing_test.lane_keeping_command` **逐项一致**，这样基线对照才可比：
        误差向量 `(0, −y, 0)`（lane 系）投到**机体系横向轴** ⇒ `−y·cos(yaw)`；D 项用**机体系**
        横向速度（冻结策略的 `velocity_commands` 就是机体系，两项必须同坐标系）。
        """
        local = self._asset.data.root_pos_w - self._env.scene.env_origins
        yaw = yaw_from_quat(self._asset.data.root_quat_w)
        lateral_error = -local[:, 1] * torch.cos(yaw)
        command = (self.cfg.lane_kp_y * lateral_error
                   - self.cfg.lane_kd_y * self._asset.data.root_lin_vel_b[:, 1])
        return torch.clamp(command, -self.cfg.lane_max_vy, self.cfg.lane_max_vy)

    def _lane_keeping_wz(self):
        """朝向 PD → `ω` 指令：把 yaw 拉回 `lane_yaw_target`（默认 0 = lane 的 +x 方向）。

        D 项用机体系 yaw 角速度；输出限幅在 AMP 训练分布的 `ang_vel_z ±1.57` 内。
        """
        yaw = yaw_from_quat(self._asset.data.root_quat_w)
        heading_error = math_utils.wrap_to_pi(self.cfg.lane_yaw_target - yaw)
        command = (self.cfg.lane_kp_yaw * heading_error
                   - self.cfg.lane_kd_yaw * self._asset.data.root_ang_vel_b[:, 2])
        return torch.clamp(command, -self.cfg.lane_max_wz, self.cfg.lane_max_wz)

    def update_safety_state(self):
        """Refresh the oriented body-surface gap and direct deck-contact collision witness."""
        rear = torch.tensor((-self.cfg.robot_rear_surface_x, 0.0, 0.0), device=self.device)
        front = torch.tensor((self.cfg.cart_front_surface_x, 0.0, 0.0), device=self.device)
        robot_surface = self._asset.data.root_pos_w + math_utils.quat_apply(
            self._asset.data.root_quat_w, rear.expand(self.num_envs, 3))
        cart_surface = self._cart.data.root_pos_w + math_utils.quat_apply(
            self._cart.data.root_quat_w, front.expand(self.num_envs, 3))
        forward = math_utils.quat_apply(
            self._asset.data.root_quat_w,
            torch.tensor((1.0, 0.0, 0.0), device=self.device).expand(self.num_envs, 3))
        self.rope_state[:, 0] = ((robot_surface - cart_surface) * forward).sum(dim=1)
        pair_forces = []
        for sensor in self._collision_sensors:
            force_matrix = sensor.data.force_matrix_w
            if force_matrix is None:
                raise RuntimeError("filtered cart/robot contact force matrix is unavailable")
            pair_forces.append(torch.linalg.vector_norm(force_matrix, dim=-1).reshape(self.num_envs, -1))
        maximum_force = torch.cat(pair_forces, dim=1).amax(dim=1)
        present = self.cart_present[:, 0]
        self.rope_state[~present, 0] = 0.0
        self.cart_collision.copy_((maximum_force > self.cfg.collision_force_threshold) & present)

    def apply_actions(self):
        # apply_actions is called every 5 ms physics step. Evaluate the frozen locomotion policy
        # every 20 ms and hold its joint targets between evaluations.
        low_level_decimation = round(self.cfg.low_level_control_dt / self.cfg.physics_dt)
        if self._physics_step % low_level_decimation == 0:
            gravity_world = torch.tensor((0.0, 0.0, -1.0), device=self.device).expand(self.num_envs, 3)
            projected_gravity = math_utils.quat_apply_inverse(
                self._asset.data.root_quat_w, gravity_world)
            output = self._policy.step(parts_from_robot_state(
                base_ang_vel=self._asset.data.root_ang_vel_b,
                projected_gravity=projected_gravity,
                # 冻结策略的输入只有**脚本调度**的速度指令；上层残差不进它的观测，
                # 否则 45 维冻结契约（含它自己的 last_action）就被污染了。
                velocity_command=self.loco_command,
                joint_pos=self._asset.data.joint_pos[:, self._policy_to_asset],
                joint_vel=self._asset.data.joint_vel[:, self._policy_to_asset]))
            self.last_loco_action.copy_(output.action)
            # 顺序不能反：先记下**底层期望**（不含残差），再叠加残差下发。反了参考量就含残差，
            # 该项就从"实际离期望多远"变成"PD 跟不跟得上下发目标"，是另一项语义。
            self.loco_joint_targets.copy_(output.joint_targets)
            # 残差加在**关节位置目标**上（`joint_targets` 与残差同为策略关节顺序），
            # 每次冻结策略刷新都要重算：残差在两次上层更新之间不变，但底层输出每 20 ms 变。
            self._held_joint_targets[:, self._policy_to_asset] = (
                output.joint_targets + self.delta_joint_pos)
        self._asset.set_joint_position_target(
            self._held_joint_targets[:, self._policy_to_asset], joint_ids=self._policy_to_asset)
        self._apply_towing_physics()
        self._physics_step += 1

    @staticmethod
    def _quat_to_matrix(quat):
        w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
        return ((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
                (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
                (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))

    def _body_properties(self, asset, body_id, offset_world, *, effective_mass=None,
                         inertia_diag=None, mass_total=None):
        """组装约束求解所需的刚体属性。

        `inverse_inertia_world` 依赖**当前姿态**，必须每步算；但机体对角惯量与质量是常量
        （只在 reset 随机化），由调用方通过 `inertia_diag` / `mass_total` 传入缓存值，
        避免每个物理步回读 PhysX。
        """
        if inertia_diag is None:
            inertias = asset.root_physx_view.get_inertias().to(self.device)
            inertia_diag = (inertias[:, body_id, 0], inertias[:, body_id, 4],
                            inertias[:, body_id, 8])
        rotation = self._quat_to_matrix(asset.data.body_quat_w[:, body_id])
        inverse = world_inverse_inertia(inertia_diag, rotation)
        if effective_mass is None:
            effective_mass = mass_total
        if effective_mass is None:
            effective_mass = asset.root_physx_view.get_masses().to(self.device).sum(dim=1)
        return BodyProperties(
            mass=effective_mass, inverse_inertia_world=inverse,
            offset=(offset_world[:, 0], offset_world[:, 1], offset_world[:, 2]))

    def _apply_towing_physics(self):
        robot_offset = self._robot_attach.expand(self.num_envs, 3)
        cart_offset = self._cart_attach.expand(self.num_envs, 3)
        robot_offset_w = math_utils.quat_apply(
            self._asset.data.body_quat_w[:, self._robot_body_id], robot_offset)
        cart_offset_w = math_utils.quat_apply(
            self._cart.data.body_quat_w[:, self._cart_body_id], cart_offset)
        robot_point = self._asset.data.body_pos_w[:, self._robot_body_id] + robot_offset_w
        cart_point = self._cart.data.body_pos_w[:, self._cart_body_id] + cart_offset_w
        robot_velocity = point_velocity(
            self._asset.data.body_lin_vel_w[:, self._robot_body_id],
            self._asset.data.body_ang_vel_w[:, self._robot_body_id], robot_offset_w)
        cart_velocity = point_velocity(
            self._cart.data.body_lin_vel_w[:, self._cart_body_id],
            self._cart.data.body_ang_vel_w[:, self._cart_body_id], cart_offset_w)

        if (self._robot_inertia_diag is None or self._cart_inertia_diag is None
                or self._cart_mass_total is None):
            self._refresh_mass_cache()
        wheel_inertia = self._cart_wheel_inertia
        cart_effective_mass = self._cart_mass_total + wheel_inertia / self.cfg.wheel_radius ** 2
        state = self._rope_model.update(
            robot_point=robot_point, cart_point=cart_point,
            robot_velocity=robot_velocity, cart_velocity=cart_velocity,
            dt=self.cfg.physics_dt,
            robot=self._body_properties(self._asset, self._robot_body_id, robot_offset_w,
                                        inertia_diag=self._robot_inertia_diag,
                                        mass_total=self._robot_mass_total),
            cart=self._body_properties(
                self._cart, self._cart_body_id, cart_offset_w,
                effective_mass=cart_effective_mass,
                inertia_diag=self._cart_inertia_diag))

        present = self.cart_present.to(dtype=robot_point.dtype)
        force_robot_w = torch.stack(tuple(state.force_on_robot), dim=-1) * present
        force_cart_w = torch.stack(tuple(state.force_on_cart), dim=-1) * present
        force_robot_b = math_utils.quat_apply_inverse(
            self._asset.data.body_quat_w[:, self._robot_body_id], force_robot_w).unsqueeze(1)
        force_cart_b = math_utils.quat_apply_inverse(
            self._cart.data.body_quat_w[:, self._cart_body_id], force_cart_w).unsqueeze(1)
        zero_torque = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._asset.set_external_force_and_torque(
            force_robot_b, zero_torque, positions=robot_offset.unsqueeze(1),
            body_ids=[self._robot_body_id])
        self._cart.set_external_force_and_torque(
            force_cart_b, zero_torque, positions=cart_offset.unsqueeze(1),
            body_ids=[self._cart_body_id])
        self.towing_force_b[:] = force_robot_b[:, 0, :3]
        # 200 Hz 诊断（TOW-25）：物理子步内的 max |Δ‖F‖| / dt_phys。
        # 只在这里累加、由 `obs_force_rate_max` 读后清零；**不参与** 20 Hz 的 rate 惩罚。
        force_mag = torch.linalg.vector_norm(self.towing_force_b, dim=1)
        self.force_rate_max.copy_(torch.maximum(
            self.force_rate_max,
            (force_mag - self._phys_prev_force_mag).abs() / self.cfg.physics_dt))
        self._phys_prev_force_mag.copy_(force_mag)

        effort = torch.zeros_like(self._cart.data.joint_pos)
        effort[:, self._wheel_joint_ids] = (
            -self.wheel_damping * self._cart.data.joint_vel[:, self._wheel_joint_ids]
            * present)
        self._cart.set_joint_effort_target(effort)
        self.rope_state[:, 1] = state.rope_tension * present[:, 0]
        self.rope_state[:, 2] = state.rope_extension * present[:, 0]
        self.rope_state[:, 3] = state.is_taut * present[:, 0]

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._raw[env_ids] = 0
        self._processed[env_ids] = 0
        self._previous[env_ids] = 0
        self.delta_joint_pos[env_ids] = 0
        self.task_command[env_ids] = 0
        self.loco_command[env_ids] = 0
        self.last_loco_action[env_ids] = 0
        self.rope_state[env_ids] = 0
        self.towing_force_b[env_ids] = 0
        # 拉力变化率（TOW-25）的全套状态：缓存、首拍门、子步诊断都必须复位。
        # `force_rate_has_prev` / `_force_rate_tick_seen` 置 False ⇒ reset 后第一拍
        # rate 定义为 0（精确不罚）；`prev_force_mag` 清零与 `towing_force_b` 同源。
        self.prev_force_mag[env_ids] = 0
        self.force_rate_has_prev[env_ids] = False
        self._force_rate_tick_seen[env_ids] = False
        self.force_rate_max[env_ids] = 0
        self._phys_prev_force_mag[env_ids] = 0
        self._held_joint_targets[env_ids] = self._asset.data.default_joint_pos[env_ids]
        self.loco_joint_targets[env_ids] = self._asset.data.default_joint_pos[
            env_ids][:, self._policy_to_asset]
        self.cart_collision[env_ids] = False
        progress = ((self._asset.data.root_pos_w - self._env.scene.env_origins)
                    * self.terrain_tangent_w).sum(dim=1)
        self.start_progress[env_ids] = progress[env_ids]
        # STOP 相位复位：`+inf` 让两条 post_stop 奖励在再次停车前恒为 0；
        # `stop_origin_x` 先落在当前 x，等 `process_actions` 触发时再写真实值。
        self.stop_time_s[env_ids] = math.inf
        self.stop_origin_x[env_ids] = self._asset.data.root_pos_w[env_ids, 0]
        self._was_stopped[env_ids] = False
        self._policy.reset(env_ids)


@configclass
class HierarchicalVelocityActionCfg(ActionTermCfg):
    """上层动作 = **1 维 vx 偏移 + 12 维归一化关节残差**（13 维，2026-10-10 起）。

    原先的 `acceleration_min/max` 与 `reference_min/max` 已删除：动作不再是加速度积分，
    残差尺度直接取冻结策略契约的 `action_scale`（见 `UpperActionSpec`）。

    双头分工（`docs/towing_upper_two_head_2026-10-10.md`）：

    - ``u_joint``（12 维）：关节位置残差，叠加在冻结策略的关节目标上；
    - ``u_cmd``（1 维）：vx 偏移，**加性 + 有界**地叠在脚本调度的 vx 上，只在
      ``elapsed_s >= tow_start_s`` 后生效。新字段默认值即用户建议值。
    """

    class_type: type[ActionTerm] = HierarchicalVelocityAction
    asset_name: str = "robot"
    cart_asset_name: str = "cart"
    policy_name: str = "amp"
    upper_control_dt: float = 0.05
    low_level_control_dt: float = 0.02
    physics_dt: float = 0.005
    initial_tow_speed: float = 0.5
    tow_start_s: float = SETTLE_TIME_S
    # ---- 2026-10-10 新增：1 维 vx 偏移头（cmd vel 头）----
    #: offset = clip(u_cmd, ±1) × cmd_offset_scale（m/s）。
    cmd_offset_scale: float = 0.5
    #: 偏移量本身的限幅（m/s）——**不是** `task_vx + offset` 的绝对限幅。理由见
    #: `upper_logic.UpperActionSpec`：脚本速度 0.5–1.5 m/s，裁和会把牵引指令砍到 0.6。
    offset_min: float = -0.2
    offset_max: float = 0.6
    #: **合成后 vx 的训练包络**（m/s），逐字取自冻结 AMP 策略的指令范围
    #: `amp_env_cfg.__post_init__`：`commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.5)`。
    #: 这不是"权限"，是**分布约束**：超出即 OOD，冻结策略的步态会退化。
    #: 它必须覆盖脚本速度范围（`episode_geometry.SPEED_RANGE = 0.5–1.5`），
    #: `upper_env_cfg.__post_init__` 有断言守着这条 —— 否则零偏移时脚本自己就被裁掉。
    amp_vx_range: tuple[float, float] = (-1.0, 1.5)
    # ---- 2026-10-10 新增：「停机之后」开关，默认关闭 = 今天的行为 ----
    # 同批落地的另一个「停机之后」旋钮 `post_stop_allowance_m` 是 `post_stop_distance` 的
    # **奖励 params**（不在本 cfg 上）；该奖励项已于 2026-10-10 删除（见 `post_stop_distance`
    # 的 docstring），形参保留供复用。
    #: STOP 之后脚本 vx 在多少秒内从 tow_speed 线性降到 0。**0.0 = 关闭**（今天的脚本调度，
    #: 到点直接归零）。要让策略学会「停机续走」就设成 1.0–2.0 s：脚本侧先给出可跟的
    #: 参考量，偏移头只做按车重/坡度的自适应修正，而不是从零学一个不存在的步态。
    stop_command_ramp_s: float = 0.0
    # 归零的沿 lane 水平距离（m）；必须 > `slope_geometry.FLAT_OUT_START_M`（坡面出口），
    # 即"越过坡之后"才 STOP。`upper_env_cfg.__post_init__` 有显式断言守着这条。
    stop_distance_m: float = STOP_DISTANCE_M
    # ---- 横向/朝向 PD（用户 2026-10-09 决定：保持中线与 heading 用 PD，不让策略学）----
    # 公式与测量台 `play_towing_test.lane_keeping_command` 逐项一致，默认增益也相同，
    # 这样"PD 基线 vs 上层策略"的对照才可比。
    lane_keeping: bool = True
    lane_yaw_target: float = 0.0       # 目标朝向：0 = lane 的 +x（车道轴向）
    lane_kp_y: float = 1.0             # 横向 P  [1/s]
    lane_kd_y: float = 0.3             # 横向 D  [-]
    lane_kp_yaw: float = 1.5           # 朝向 P  [1/s]
    lane_kd_yaw: float = 0.3           # 朝向 D  [-]
    # 限幅在**冻结 AMP 策略的训练分布**内（`amp_env_cfg`：lin_vel_y ±1.0、ang_vel_z ±1.57），
    # 否则指令出分布、下层跟踪会崩。
    lane_max_vy: float = 1.0
    lane_max_wz: float = 1.5708
    initial_cart_mass: float = 10.0
    initial_ground_friction: float = 0.8
    initial_wheel_damping: float = 0.032
    robot_body_name: str = "base"
    # 足端接触传感器（场景里由 `upper_env_cfg` 注册，`{ENV_REGEX_NS}/Robot/.*_FOOT`）。
    # `mdp.feet_slide` 用它判着地；body 顺序按该传感器的 `body_names` 映射到资产 id。
    foot_contact_sensor_name: str = "foot_contacts"
    cart_body_name: str = "base_link"
    cart_wheel_joint_names: tuple[str, ...] = (
        "wheel_fl_joint", "wheel_fr_joint", "wheel_rl_joint", "wheel_rr_joint")
    cart_wheel_body_names: tuple[str, ...] = (
        "wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr")
    robot_attachment: tuple[float, float, float] = (-0.16, 0.0, 0.0)
    cart_attachment: tuple[float, float, float] = (0.25, 0.0, 0.0)
    # 连接长度 L0（绳）/ L（刚体）与弹性绳的 k/c 都来自场景网格 `mdp/connection_grid.py`，
    # 逐 env 不同，不再是这里的标量配置。这里只留两类约束共用的回拉增益。
    rope_position_gain: float = 0.2
    rope_max_correction_rate: float = 0.2
    rigid_position_gain: float = 0.2
    rigid_max_correction_rate: float = 0.2
    wheel_radius: float = 0.08
    collision_sensor_names: tuple[str, ...] = (
        "cart_deck_robot_contacts",
        "cart_wheel_fl_robot_contacts", "cart_wheel_fr_robot_contacts",
        "cart_wheel_rl_robot_contacts", "cart_wheel_rr_robot_contacts",
    )
    collision_force_threshold: float = 1.0
    robot_rear_surface_x: float = 0.1575
    cart_front_surface_x: float = 0.25


def reset_towing_episode(
    env, env_ids, *, speed_range, mass_range, friction_range,
    wheel_damping_range, robot_x_range, robot_y_range, robot_yaw_range,
    no_cart_fraction, no_cart_lateral_offset,
):
    """Reset the per-episode random work condition；连接类型/长度来自确定性网格。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)
    if len(env_ids) == 0:
        return

    term = _term(env)
    count = len(env_ids)

    def sample(bounds):
        lo, hi = bounds
        return lo + (hi - lo) * torch.rand(count, device=env.device)

    speeds = sample(speed_range)
    masses_kg = sample(mass_range)
    friction = sample(friction_range)
    wheel_damping = sample(wheel_damping_range)
    term.tow_speed[env_ids] = speeds
    term.cart_mass[env_ids, 0] = masses_kg
    term.ground_friction[env_ids, 0] = friction
    term.wheel_damping[env_ids, 0] = wheel_damping
    if not 0.0 <= no_cart_fraction <= 1.0:
        raise ValueError("no_cart_fraction must be in [0, 1]")
    if no_cart_lateral_offset <= 0.0:
        raise ValueError("no_cart_lateral_offset must be positive")
    cart_present = torch.rand(count, device=env.device) >= no_cart_fraction
    term.cart_present[env_ids, 0] = cart_present

    # 连接类型与长度来自确定性网格（`term.rope_model_id` / `term.connection_length`，在
    # `__init__` 里按 env index 一次算好），**不再逐 env 随机**；这里只按 env 取回本回合
    # spawn 摆位需要的目标挂点距。
    target_distance = term.initial_distance[env_ids, 0]

    robot = term._asset
    cart = term._cart
    robot_state = robot.data.default_root_state[env_ids].clone()
    tangent = term.terrain_tangent_w[env_ids]
    normal = term.terrain_normal_w[env_ids]
    robot_height = robot.data.default_root_state[env_ids, 2]
    cart_height = cart.data.default_root_state[env_ids, 2]
    robot_state[:, :3] = (env.scene.env_origins[env_ids]
                         + sample(robot_x_range).unsqueeze(1) * tangent
                         + robot_height.unsqueeze(1) * normal)
    robot_state[:, 1] += sample(robot_y_range)
    yaw = sample(robot_yaw_range)
    yaw_delta = math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    # 出生在剖面的平地段上 ⇒ 姿态竖直，只叠一层采样的小偏航（不再有 R_y(−θ) 出生旋转；
    # 上/下坡在同一条 lane 内部，靠走过去而不是靠出生姿态对齐）。
    robot_state[:, 3:7] = math_utils.quat_mul(robot_state[:, 3:7], yaw_delta)
    robot_state[:, 7:13] = 0
    term.start_progress[env_ids] = ((robot_state[:, :3] - env.scene.env_origins[env_ids])
                                    * tangent).sum(dim=1)
    robot.write_root_pose_to_sim(robot_state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(robot_state[:, 7:13], env_ids=env_ids)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos[env_ids],
                                   torch.zeros_like(robot.data.default_joint_vel[env_ids]), env_ids=env_ids)

    cart_state = cart.data.default_root_state[env_ids].clone()
    # Solve in the (spawn) tangent/normal frame, not with a fixed world-Z difference.
    # Both default root heights are normal clearances; all attachment offsets are
    # transformed from the actual spawned orientation (including sampled yaw).
    # 出生在平地段 ⇒ 这里的 tangent/normal 就是 +x/+z，解退化成平地摆放；保留这套写法是为了
    # 挂点偏移随采样偏航旋转后仍然精确（两挂点三维距 = 目标值，第一拍没有约束力）。
    robot_attach_w = math_utils.quat_apply(
        robot_state[:, 3:7], term._robot_attach.expand(count, 3))
    cart_attach_w = math_utils.quat_apply(
        cart_state[:, 3:7], term._cart_attach.expand(count, 3))
    normal_difference = (robot_height - cart_height
                         + ((robot_attach_w-cart_attach_w)*normal).sum(dim=1))
    if bool((target_distance ** 2 <= normal_difference ** 2).any()):
        raise ValueError("connection distance must exceed normal attachment height difference")
    along = attachment_horizontal_gap(target_distance, delta_z=normal_difference)
    cart_state[:, :3] = (robot_state[:, :3] + robot_attach_w - cart_attach_w
                         - along.unsqueeze(1)*tangent - normal_difference.unsqueeze(1)*normal)
    # Isaac Lab replicates one cart articulation per environment. For zero-load environments,
    # park it inside the 6 m cell but well outside the robot's reachable path.
    # 注意顺序：必须在按连接长度摆放**之后**再加横移，否则会把无小车环境的横向停放覆盖掉。
    cart_state[~cart_present, 1] += no_cart_lateral_offset
    cart_state[:, 7:13] = 0
    cart.write_root_pose_to_sim(cart_state[:, :7], env_ids=env_ids)
    cart.write_root_velocity_to_sim(cart_state[:, 7:13], env_ids=env_ids)
    cart.write_joint_state_to_sim(cart.data.default_joint_pos[env_ids],
                                  torch.zeros_like(cart.data.default_joint_vel[env_ids]), env_ids=env_ids)

    # Scale every cart body's mass and inertia from immutable defaults, never from the prior reset.
    ids_cpu = env_ids.cpu()
    default_masses = cart.data.default_mass.detach().cpu()
    default_inertias = cart.data.default_inertia.detach().cpu()
    nominal_total = default_masses[ids_cpu].sum(dim=1)
    scale = masses_kg.detach().cpu() / nominal_total
    all_masses = cart.root_physx_view.get_masses().clone()
    all_inertias = cart.root_physx_view.get_inertias().clone()
    all_masses[ids_cpu] = default_masses[ids_cpu] * scale[:, None]
    all_inertias[ids_cpu] = default_inertias[ids_cpu] * scale[:, None, None]
    cart.root_physx_view.set_masses(all_masses, ids_cpu)
    cart.root_physx_view.set_inertias(all_inertias, ids_cpu)
    # 质量／惯量刚被改写，必须让每步物理用的缓存失效；否则 `_apply_towing_physics` 会继续用
    # 上一回合的质量（缓存只在 None 时重建，不会自己发现 PhysX 侧的变化）。
    term._invalidate_mass_cache()

    # Give robot and cart the same per-environment contact coefficient. This is terrain-like
    # domain information for critic/decoder, while the actor never receives the true value.
    for asset in (robot, cart):
        materials = asset.root_physx_view.get_material_properties().clone()
        materials[ids_cpu, :, 0] = friction.detach().cpu()[:, None]
        materials[ids_cpu, :, 1] = friction.detach().cpu()[:, None]
        materials[ids_cpu, :, 2] = 0.0
        asset.root_physx_view.set_material_properties(materials, ids_cpu)


def loco_command(env):
    """送给冻结底层策略的速度指令 = ``task_command`` + 有界 vx 偏移。

    ⚠ 这是**送策略 / 进 actor 帧**的量，含策略自己的偏移，**绝不能**当奖励参考量
    （否则策略把偏移开大就能自己给自己发目标）。奖励一律读 `task_command`。
    """
    return _term(env).loco_command
def task_command(env):
    """**任务指令**（脚本调度 + 可选 STOP ramp）：只给奖励当参考量，不含策略偏移。"""
    return _term(env).task_command
def upper_last_action(env):
    """上一拍 **13 维**上层动作（clamp 到 ±1，未缩放）：1 维 vx 偏移 + 12 维关节残差。"""
    return _term(env).processed_actions
def base_angular_velocity(env): return _term(env)._asset.data.root_ang_vel_b
def projected_gravity(env):
    term = _term(env)
    gravity_world = torch.tensor((0.0, 0.0, -1.0), device=term.device).expand(term.num_envs, 3)
    return math_utils.quat_apply_inverse(term._asset.data.root_quat_w, gravity_world)
def last_locomotion_action(env): return _term(env).last_loco_action
def joint_pos_rel_policy_order(env):
    term = _term(env); ids = term._policy_to_asset
    return term._asset.data.joint_pos[:, ids] - term._asset.data.default_joint_pos[:, ids]
def joint_vel_policy_order(env):
    term = _term(env); return term._asset.data.joint_vel[:, term._policy_to_asset]


def low_level_position_error_l2(env, reference="commanded"):
    """底层**期望关节位置**与实际关节位置的平方误差（rad²）。

    `reference` 决定"期望"取哪个（用户 2026-10-09 定为 `commanded`）：

    - `"commanded"`（当前）：期望 = **底层输出 + 上层残差** = 真正下发给 PD 的目标
      （`_held_joint_targets`）。残差被参考量抵消 ⇒ 误差 = 纯 PD 跟踪误差 `e_PD`，回答
      「底层达没达到要求的位置」。残差的作用是"稳住角度跟踪"时，它调整的正是**期望**，
      所以这里量的就是"调整后的期望达成了没有"。
    - `"frozen"`：期望 = 底层策略**自己**的输出（稳定步态、不含残差）。此时
      `q_act − q_exp = δ + e_PD`（含残差），语义变成"实际离底层意图多远"，会把 δ 一起罚。

    两处缓存都在 action term 里维护（`_held_joint_targets[:, _policy_to_asset]` 与
    `loco_joint_targets`），都是**策略关节顺序**，可直接与 `joint_pos[:, _policy_to_asset]` 相减。

    单位 rad²（12 关节求和，未取均值）。冒烟实测（2026-10-09，64 环境 20 轮）：`frozen` 口径
    Σ≈0.34 rad²，按 `action_magnitude` 反推其中约 0.12 来自残差、0.22 来自 PD 误差。
    """
    term = _term(env)
    actual = term._asset.data.joint_pos[:, term._policy_to_asset]
    if reference == "commanded":
        expected = term._held_joint_targets[:, term._policy_to_asset]
    elif reference == "frozen":
        expected = term.loco_joint_targets
    else:
        raise ValueError(f"reference 必须是 'commanded' 或 'frozen'，收到 {reference!r}")
    return (actual - expected).square().sum(dim=1)


def feet_slide(env, sensor_name, contact_threshold=1.0):
    """支撑脚不许打滑：**着地**足的机体（水平）速度模长之和（m/s）。

    公式与 PPO rough 的 `feet_slide` 一致（Isaac Lab
    `isaaclab_tasks/.../locomotion/velocity/mdp/rewards.py`，那边权重 −0.05、足端正则 `.*_FOOT`）：

        Σ_feet ‖v_foot,xy‖ · 1(net_forces_w_history 的历史最大合力 > contact_threshold)

    - 着地门控用 `net_forces_w_history` 的**历史最大合力**（≥1 N 视为着地），与正本一致：
      单拍的力抖动不会把着地误判成腾空。
    - body 顺序按**传感器**的 `body_names` 映射到资产 id（见 action term 的 `_foot_body_ids`），
      不依赖正本那种"传感器与资产同序"的隐含假设。
    - 它是**模长之和**（不是平方和、不是均值）：着地脚静止时 ≈0，打滑时线性增长。
      注意它天然可被"走慢"降低 ⇒ 与 `tracking_velocity` 对冲，权重需一起看。
    """
    term = _term(env)
    sensor = env.scene.sensors[sensor_name]
    contacts = (sensor.data.net_forces_w_history.norm(dim=-1).max(dim=1)[0]
                > contact_threshold)
    body_vel = term._asset.data.body_lin_vel_w[:, term._foot_body_ids, :2]
    return torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
def policy_frame(env):
    """One complete actor frame; history is applied once to preserve frame-major ordering.

    2026-10-10 起为 **58 维**（双头动作，v3 契约）：命令项是 ``loco_command``
    （= 任务指令 + 有界偏移；原来并列的 `cmd_vel`／`reference_command` 在残差方案下恒等，
    属冗余），``last_action`` 由 12 维变 **13 维**（1 维 vx 偏移 + 12 维关节残差）。
    """
    return torch.cat((loco_command(env), upper_last_action(env),
                      base_angular_velocity(env) * 0.25, projected_gravity(env),
                      last_locomotion_action(env), joint_pos_rel_policy_order(env),
                      joint_vel_policy_order(env) * 0.05), dim=1)
def robot_velocity(env): return _term(env)._asset.data.root_lin_vel_b[:, :2]
def cart_velocity(env): return _term(env)._cart.data.root_lin_vel_b[:, :2]
def rope_privileged_state(env):
    term = _term(env); term.update_safety_state(); return term.rope_state
def towing_force(env): return _term(env).towing_force_b
def cart_privileged_parameters(env):
    term = _term(env)
    return torch.cat((term.cart_mass, term.ground_friction, term.wheel_damping,
                      term.cart_present.float()), 1)
def decoder_targets(env):
    """decoder 的回归目标：**物理量**（m/s、kg、N），不归一化、不 clamp。

    2026-09-23 改为物理量：原先归一化到 [-1,1] 配合 head 的 tanh，但归一化尺度会按 s²
    压低 loss 的物理权重（力 s=10 ⇒ 0.01、质量 s=5 ⇒ 0.04），使这两项几乎训不动；而且
    `v/(1.0,0.5)` 的 clamp 会把超过 1.0/0.5 m/s 的真值截断。去掉后 head 也不带 tanh，
    稳定性由 `TowingDynamicsDecoder.loss()` 的 smooth_l1(β=1) 提供。
    """
    term = _term(env)
    robot_velocity_xy = term._asset.data.root_lin_vel_b[:, :2]
    mass = term.cart_mass
    force = term.towing_force_b
    return torch.cat((robot_velocity_xy, mass, force), dim=1)
def decoder_mass_weight(env, minimum_force, force_scale):
    force = torch.linalg.vector_norm(_term(env).towing_force_b, dim=1, keepdim=True)
    effective_force = (force - minimum_force).clamp_min(0.0)
    return effective_force / (effective_force + force_scale)


def cart_collision(env):
    term = _term(env); term.update_safety_state(); return term.cart_collision
def cart_collision_cost(env): return cart_collision(env).float()
def velocity_tracking_exp(env, linear_std, yaw_std, use_lateral_and_heading=False):
    """**实际速度** vs 命令期望的指数跟踪项（唯一的正奖励）。

    2026-10-08 修正：线性项原先比的是 ``reference_command − user_command``，也就是「上层
    自己积分出来的指令 vs 任务指令」——这与本函数的名字、以及 `upper_env_cfg` 奖励文档的
    描述（"实际速度 vs 命令期望"）都不符，而且和 `reference_tracking_l2` 重复。残差方案下
    指令由脚本给出、`reference_command` 不复存在，故直接改成实测机体系线速度。

    2026-10-09（用户决定）：**横向与朝向改由 PD 外环负责**（`_lane_keeping_vy/_wz`），
    所以本项默认**只用纵向 `vx`**：

        exp(−(vx_meas − vx_cmd)² / linear_std²)

    即"跟不跟得上横向/朝向指令"不再进回报——那是 PD 的职责；上层策略只对**前进速度**负责。
    `use_lateral_and_heading=True` 可恢复旧口径（含 `vy` 与 yaw 角速度误差），仅供对照。

    ⚠ **参考量是 `task_command`，不是 `loco_command`（2026-10-10 防作弊红线）**：
    `loco_command` 含策略自己的 1 维 vx 偏移，若拿它当参考，策略只要把偏移开大就能
    自己给自己发目标、白拿跟踪奖励。脚本调度（含 STOP 归零/ramp）只写在 `task_command`。
    """
    term = _term(env)
    forward_error = ((term._asset.data.root_lin_vel_b[:, 0] - term.task_command[:, 0])
                     / linear_std)
    if not use_lateral_and_heading:
        return torch.exp(-forward_error.square())
    lateral_error = ((term._asset.data.root_lin_vel_b[:, 1] - term.task_command[:, 1])
                     / linear_std)
    yaw_error = (term._asset.data.root_ang_vel_b[:, 2] - term.task_command[:, 2]) / yaw_std
    return torch.exp(-(forward_error.square() + lateral_error.square() + yaw_error.square()))
def clearance_barrier(env, warning_distance, scale):
    term = _term(env); term.update_safety_state(); clearance = term.rope_state[:, 0]
    return (torch.nn.functional.softplus((warning_distance - clearance) / scale)
            * term.cart_present[:, 0])
def min_clearance_violation(env, deadband_m, softness=0.02):
    """铰链式「最小间距」惩罚：间隙低于 `出生间隙 − deadband_m` 时与缺口成正比，高于则**精确为 0**。

    为什么单独加一项（而不是复用 `clearance_barrier`）：`clearance_barrier` 是 softplus
    软障碍，其"警戒距离"是绝对量（0.20 m）且线性区在警戒线**下方**；本项是**相对出生几何**的
    硬阈值：高于阈值恒为 0、低于阈值与缺口成正比，语义是「不得比出生时更近」，与「近了要缓」互补。
    实现是**带截断的 softplus**（`softplus 减去 softplus(0)` 再 `relu`）而不是纯铰链，但阈值
    **上方精确为 0、没有 softplus 尾巴**——这正是本项与 `clearance_barrier` 的分界。

    **2026-10-10 口径变更（用户批准，当天两次）**：阈值基准由「`ratio × 本 env 连接长度`」先改成
    「`spawn_margin(0.85) × 本 env 出生瞬间的间隙`」，同日再微调为**绝对死区**：`spawn_margin`
    形参删除、改传 `deadband_m`（默认口径 0.02 m），权重一直是 **−5.0**（由 −2.0 提上来）。

        threshold = spawn_clearance − deadband_m
        spawn_clearance = sqrt((spawn_ratio · L)² − Δz²) + base_offset

    其中（全部逐 env、从现有常量派生，不手抄数字）：

    - `spawn_ratio` = 绳 `connection_grid.SLACK_RATIO`（现 0.8）/ 杆 **1.0**（连杆出生即全长）
      —— 已经体现在 `term.initial_distance`（= `connection_grid.initial_attachment_distance`）里；
    - `Δz` = 两挂点高差 = 机器人/小车 `default_root_state` 的高度差（现 0.17 m）；
    - `base_offset = −robot_rear_surface_x − robot_attachment_x + cart_attachment_x
      − cart_front_surface_x`（现 0.0025 m）——即「后表面↔车斗前表面」比挂点距多出来的固定量；
    - `spawn_clearance` 在 `HierarchicalVelocityAction.__init__` 里一次算好并缓存。

    **为什么从「相对余量 0.85」改成「绝对死区 2 cm」**（用户 2026-10-10 决定，三条理由）：

    1. **语义更贴**：用户要的就是「**不许比出生时更近**」。`出生间隙 − 死区` 直接表达它；
       `0.85 × 出生间隙` 表达的是"比出生近 15%"，是个没有物理含义的相对量。
    2. **量级正确**：`spawn_clearance` 是**解析**出生间隙（勾股解 + 四个表面常量），与仿真里的
       实际间隙差**几毫米**（落地/穿透/初始沉降）⇒ 绝对 2 cm 死区正好吸收这个量级。相对余量却
       随 L 放大：L=1.5 的绳行余量 `0.15 × 1.1904 = 0.179 m`，比需要的量大一个数量级，等于给
       长绳行白开几十厘米的"可以靠近"口子。
    3. **不需要大余量（纠正旧顾虑）**：出生点在 lane 的**后向平段**上——`slope_geometry` 的剖面
       从 **+2.25 m** 才起坡（平地 2.25 m → 上坡 3 m → 坡顶 0.75 m → 下坡 3 m → 平地 2.25 m）。
       所以**出生与停车都发生在平地、没有重力驱动** ⇒ 间隙只受机器人动作影响，**不存在被动的
       间隙漂移**。旧顾虑「坡上会溜车所以要留大余量」在本任务里不成立（停车滑行段在平地上由
       黏性轮阻耗散、车斗自行停住），因此 2 cm 的死区就够。

    **逐行语义**（L = 连接长度，网格见 `mdp/connection_grid.py`）：

    - 绳行（L = 0.5–1.5 m，spawn_ratio 0.8）：阈值 **0.3446–1.1704 m**（= 0.689–0.780 · L；
      L=0.5 → 0.3446 = **0.945 × 出生间隙** = 0.689 · L，L=1.5 → 1.1704 = 0.983 × 出生间隙
      = 0.780 · L）。牵引段绳绷直 ⇒ 3D 挂点距 = L > 0.8·L ⇒ 间隙 > 出生间隙 > 阈值，**不触发**；
      绳一松、车斗逼近穿过阈值时才开始线性出力 —— 这正是用户要的「不许比出生时更近」+「停车后
      继续走几步」的梯度（走得越近，缺口越大、惩罚越大）；
    - 杆行（L = 0.5–1.0 m，spawn_ratio 1.0）：连杆把三维挂点距固定在 L ⇒ 间隙恒等于出生间隙
      ⇒ 阈值 = 出生间隙 − 0.02 < 间隙 ⇒ **在出生高差上永不触发**（物理正确：杆不会缩短，
      不存在"被拉太近"）。**注意余量比旧的相对口径小得多**：旧口径下间隙恒比阈值大
      `0.15 × 间隙`（最短杆 +0.071 m），新口径只大 `deadband_m = 0.02 m`；而间隙随 Δz 增大
      而变小（`sqrt(L²−Δz²)`）——解 `sqrt(L²−Δz²) + base_offset = 出生间隙 − 0.02` 得
      Δz* = **0.218 m（L=0.5）** … **0.261 m（L=1.0）**，即最短杆只要 Δz 比出生值 0.17 m
      再大 **4.75 cm** 就会触发（旧口径是 13.1 cm）。这是本次口径变更**换来的代价**，
      逐行数值见 `docs/towing_reward_retune_2026-10-10.md` §2.5。

    **停车段天然生效 ⇒ 不需要再单独做"停车参考间隙"**：停车瞬间的实测间隙 ≈ **0.64 · L**
    （用户给的标定值；离线用归档 `docs/towingdata/2026-10-09_necessity_800{,_noload}/report.json`
    复算 `stop.clearance_at_stop_m / L` 的中位数是 **0.700**（compliant 0.702）——两者同量级），
    而新阈值 ≈ **0.945 × 出生间隙 ≈ 0.69 · L**（最短绳行）⇒ 停车段间隙**天然落在阈值之下约
    `0.05 · L`**：L=1.0 时缺口约 0.05 m ⇒ 每步 `−5 × 0.05 × 0.05 ≈ −0.0125`（线性铰链值；
    `softness = 0.02` 的截断 softplus 会把它压掉一些——缺口 0.05 m 时实际 ≈ 0.038 ⇒ 每步
    ≈ **−0.0094**，缺口 ≥0.3 m 时 ≈0.96×线性值。量级结论不变），与 `tracking_velocity`
    满额 +0.05/步 **同量级**。这正是用户要的「停车期间也保持间隙」，
    因此**不需要**再单独做"用停车瞬间间隙当参考"的方案。**备选（本轮不实现）**：若训练机实测
    停车段间隙反而**高于**阈值、本项不生效，再在 `process_actions`（20 Hz）里记录停车那一拍的
    `gap_at_stop`（并识别"停下"时刻），把阈值改成 `gap_at_stop − deadband_m`；本机无 Isaac Lab，
    实测未做。

    **权重的标定依据**：`RewardManager` 每步代价 = `weight × func × step_dt`（`step_dt = 0.05 s`）
    ⇒ `w = −5` 时超出阈值 0.2 m 的违规 ≈ **−0.05/步**，恰等于 `tracking_velocity` 满额
    （+1.0 × 0.05 = +0.05/步）⇒ **有动机但不压倒**跟踪项；旧的 −2 只有 −0.02/步（满额的 40%），
    在停车段惯性/车重面前太弱，不足以改变「停车后是否再走两步」的取舍。

    历史（2026-09-23 → 2026-10-10）：初始「后表面→车斗」间隙约 0.349 m，`ratio` 由 0.6 降到 0.40
    才让 spawn 不触发；2026-10-08 随 20×20 网格改为逐 env + ratio 0.25；2026-10-09 长度区间解耦为
    绳 0.5–1.5 / 杆 0.5–1.0 并把出生比提到 0.8；2026-10-10 先用 `spawn_margin` 相对余量、同日
    改为 `deadband_m` 绝对死区。以上 ratio / spawn_margin 口径均已作废。

    spawn 处的余量恒等于 `deadband_m`（`出生间隙 − 阈值 = deadband_m > 0`）⇒ 该拍 `func`
    精确为 0，保持不变式「`min_clearance` 在 spawn 精确为 0」（用户 2026-09-23 要求初始不生效）。
    """
    term = _term(env)
    term.update_safety_state()
    clearance = term.rope_state[:, 0]
    # 出生几何阈值（逐 env 缓存）：`出生瞬间的间隙 − 绝对死区`。见上面 docstring。
    threshold = term.spawn_clearance - deadband_m
    gap = threshold - clearance
    if softness > 0:
        # 平滑只在**阈值下方**过渡：减去 softplus(0)*softness 使缺口 ≤ 0 时精确为 0，
        # 否则 softplus 在阈值处就有 0.0139 的偏置（间隙略高于阈值也会被扣分），
        # 与"初始不生效"的意图冲突。
        bias = 0.6931471805599453 * softness      # softplus(0)
        violation = torch.relu(
            torch.nn.functional.softplus(gap / softness) * softness - bias)
    else:
        violation = torch.relu(gap)              # 纯铰链
    return violation * term.cart_present[:, 0]


def action_magnitude_l2(env):
    """上层动作幅值平方（裁剪后），用于抑制动作抖动。

    与 `action_rate_l2` 的区别：后者罚"动作变化量"（平滑性），本项罚"动作绝对值"
    （幅值）。策略长期输出 ±1 饱和随机方波时，两项都会变大，但本项能直接压住幅值，
    把 12 维关节残差压回小幅度、让冻结步态保持主导。

    2026-10-08 语义变化：动作由 3 维速度指令增量变为 12 维关节残差，`Σ u²` 的上限从 3
    变成 12（同等逐维幅值下惩罚约 ×4），权重 `-0.05` 是否仍合适**尚未实跑验证**。
    2026-10-10 又新增 1 维 vx 偏移 ⇒ 上限 13（`Σ u² ≤ 13`；偏移头通常只用零点几，
    量级影响可忽略，但权重仍需实跑标定）。
    原先配套的 `reference_tracking_l2` 已随 `reference_command` 一起删除（残差方案下恒为 0）。
    """
    return _term(env).processed_actions.square().sum(dim=1)


def heading_deviation(env):
    """机体系朝向偏离「初始朝向」的 yaw 误差（rad）。

    2026-09-23 用户报告「机器人开始就在自转」：原奖励里**没有任何朝向约束**——
    `tracking_velocity` 只惩罚 yaw **角速度**误差（`loco_command` 的 yaw 恒为 0），
    因此「原地匀速自转」在 settle 阶段几乎不受罚（角速度也接近 0），
    而 STOP 后持续缓转同样不易被察觉。

    「正方向」取机器人**初始 yaw**（`default_root_state` 的四元数，即 spawn 朝向，
    随机化只加 ±0.03 rad）。取相对值而非世界系 0，是为了不依赖小车／世界坐标约定。
    """
    term = _term(env)
    # ⚠ 顺序：euler_xyz_from_quat 返回 (roll, pitch, yaw)，yaw 在第三个（见 yaw_from_quat）
    yaw = yaw_from_quat(term._asset.data.root_quat_w)
    yaw_init = yaw_from_quat(term._asset.data.default_root_state[:, 3:7])
    return math_utils.wrap_to_pi(yaw - yaw_init)


def yaw_heading_l2(env):
    """朝向误差平方。τ=1 时立姿为 0；偏 0.2 rad(11°) → 1，偏 0.5 rad(29°) → 6.25。

    平方形式在误差大时梯度更大（误差每增 1 rad 梯度增 2·误差），适合先把自转压住。
    """
    return heading_deviation(env).square()


def action_rate_l2(env):
    term = _term(env); return (term.processed_actions - term._previous).square().sum(1)
def robot_fall(env, minimum_height):
    """离**局部坡面**的高度低于阈值 ⇒ 判定为跌倒（坡上不能用绝对 z）。"""
    term = _term(env)
    local = term._asset.data.root_pos_w - env.scene.env_origins
    height = local[:, 2] - profile_height_tensor(term.hill_grade_deg, local[:, 0])
    return height < minimum_height
def robot_fall_cost(env, minimum_height): return robot_fall(env, minimum_height).float()


def stop_reached(env):
    """机器人沿 lane 的前进量是否到达 STOP 触发点（**不是**终止条件，只用来置零指令）。

    2026-10-09 由 `goal_reached` 改名并改语义：以前到达该点就**终止**回合；现在到达该点
    进入 STOP 段（指令 0），回合继续跑到 timeout，让 `post_stop_towing_force` /
    `post_stop_distance` 有真实相位可用。触发点必须已越过坡面
    （`stop_distance_m > FLAT_OUT_START_M = 9.0 m`）。

    `progress` 是 **lane 局部 x**（平面距离）：上/下坡段的弧长略长于水平投影
    （10° 档 10 m 对应 10.09 m 弧长，+0.9%），timeout 用的是弧长上界。
    """
    term = _term(env)
    progress = ((term._asset.data.root_pos_w - env.scene.env_origins)
                * term.terrain_tangent_w).sum(dim=1)
    return progress - term.start_progress >= term.cfg.stop_distance_m


def post_stop_towing_force(env, force_scale):
    """停车之后绳子还绷着就扣分（有界归一化），教「到点了就把拉力卸掉」。

    门控是 `elapsed_s >= term.stop_time_s`：`stop_time_s` 由 `process_actions` 在进度
    越过 `stop_distance_m`（已越坡）那一拍写入，未停车前是 `+inf` ⇒ 本项精确为 0
    （不是"一直生效"）。用有界归一化 `‖F‖/(‖F‖+force_scale)` 而不是原始范数，避免远端
    大拉力把回报尺度拉爆；无小车环境用 `cart_present` 屏蔽。

    与已删除的 `extra_distance`（2026-10-10）原是配对项，防的是两种相反的极端：「为卸载拉力而
    继续前冲」与「停死后被追尾」（后者由 `collision` 惩罚兜底）。现在这一侧只剩本项；「停车后
    继续走几步」由 `min_clearance` 的出生几何梯度（见其 docstring）与动作侧的 vx 偏移头表达。
    """
    term = _term(env)
    elapsed_s = env.episode_length_buf * env.step_dt
    post_stop = (elapsed_s >= term.stop_time_s).float()
    force = torch.linalg.vector_norm(term.towing_force_b, dim=1)
    normalized_force = force / (force + force_scale)
    return normalized_force * post_stop * term.cart_present[:, 0]


def post_stop_distance(env, post_stop_allowance_m=0.0):
    """停车之后还往前多走的距离（超过 `stop_origin_x + allowance` 的部分），教「说停就停」。

    距离取世界系 x（lane 切向就是 +x），基线 `stop_origin_x` 是**指令归零那一拍**的 x，
    所以只统计 STOP 之后的滑行量，不含牵引段的前进。同样由 `stop_time_s` 门控，
    未停车前精确为 0。

    **2026-10-10 用户决定：本函数已不作为奖励项**（`upper_env_cfg` 里的
    `extra_distance = RewTerm(func=mdp.post_stop_distance, ...)` 已删除），函数本体与
    `post_stop_allowance_m` 形参**保留**供复用/诊断。删除理由：

    - 与 `tracking_velocity` 在停车段**高度冗余**：指令归零后 `vx → 0` 已经隐含位移只剩
      不可避免的惯性滑行，再单独罚一次是同一个约束记两遍；
    - **量级小 20 倍**：`w = −0.1` ⇒ 每步最多 −0.005（实际 −0.0025 量级），而
      `tracking_velocity` 满额 +0.050/步 ⇒ 它压不住任何东西，还给「停车后按车重/坡度
      再走两步」的动作侧方案凭空加一个反向梯度（偏移头一让机器人前进，本项立刻扣分）。

    ``post_stop_allowance_m``（2026-10-10 TOW-21 落地时新增，默认 0.0）：允许停车后继续走的
    距离（m）；若日后要重新启用本项，设 0.5–1.0 m 可让策略学会「STOP 后按车重/坡度再走两步」。
    默认 0.0 时公式退化为 `relu(x − x_stop)`，与历史口径**逐位一致**。
    """
    term = _term(env)
    elapsed_s = env.episode_length_buf * env.step_dt
    post_stop = (elapsed_s >= term.stop_time_s).float()
    travelled = term._asset.data.root_pos_w[:, 0] - term.stop_origin_x - post_stop_allowance_m
    return torch.relu(travelled) * post_stop


def towing_force_y_ratio_sq(env):
    """拉力方向偏离机体系 **xz 平面**程度的平方：`(F_y / ‖F‖)² = sin²(偏离角)`。

    用户 2026-10-09 要求「运动过程中 3 维拉力在 y 维度保持为 0」。`towing_force_b` 是绳/杆
    作用在机器人挂点上的力（**base 机体系** 3 维，`T = tension·e` 转过来的），其中
    F_y 只有在**挂点连线不落在机体矢状面内**时才非零（小车横向偏置、机器人相对绳向有 yaw、
    或机体有 roll 把 F_z 漏进 y）。

    为什么用**比值**而不是原始 `F_y²`：
    - 与张力大小无关。原始 `F_y²` 会被起步绷直的百牛级峰值放大（最硬弹性档 + 1.5 m/s 时
      峰值可达 ~1 kN，见 README 起步峰值表），等价于"变相惩罚大张力"，和拖曳任务对冲；
    - 有界 `[0, 1]`，权重好标定：`sin²10° = 0.030`、`sin²20° = 0.117`；
    - 语义干净：**拉力方向必须落在机体系 xz 平面内**。

    绳松弛（`T = 0`）时整个力向量为 0 ⇒ 本项为 0（不会因 0/0 产生 NaN，分母有下限）。

    ⚠ **它不等于"沿车道中心线走"**：F_y ≈ 0 只说明绳在机体矢状面内（小车正后方）。
    若机器人整体偏航、但小车也跟着偏到正后方，F_y 仍可 ≈ 0。车道中线/朝向要靠**横向偏置与
    朝向**的观测＋对应奖励（或 PD 外环），二者互补。
    """
    term = _term(env)
    force = term.towing_force_b
    norm = torch.linalg.vector_norm(force, dim=1).clamp_min(1.0e-6)
    return (force[:, 1] / norm).square()


def towing_force_rate_penalty(env, rate_limit_n_per_s=200.0, release_limit_n_per_s=600.0,
                              slack_eps_n=1.0, clip_max=3.0):
    """**拉力幅值变化率**惩罚，但放行「卸载到松弛」那一下（TOW-25，用户 2026-10-10 要求）。
    参数全部是 `UpperRewardsCfg` 的 `RewTerm.params`（不写死在函数里）；默认值见下。

    ------------------------------------------------------------------
    为什么需要它 / 为什么掩码只放行一个方向（物理理由，本项存在的全部依据）
    ------------------------------------------------------------------
    **绳是单边约束**：只能拉、不能推 ⇒ **不存在"用绳缓慢刹车"**。绳行的停车机制不是靠绳
    把车斗拉住，而是 **绳松弛 + 车斗自由滑行 + 机器人往前走两步避让**；所以绳的
    **瞬间卸载（松弛）是正常且期望的**——那一下**必须允许突变**，否则奖励在惩罚一个
    正确的物理行为（而且那个突变不是策略能"缓"下来的：绳一松，力就只能是 0）。
    **刚性杆是双边约束**：可以主动管理力 ⇒ 杆行的目标是**缓慢卸载、避免冲击力过大**
    （力尖峰才要罚）。

    因此本项的语义 = **罚"拉力幅值的突变"，但放行"卸载到松弛"那一下**：

    - 绷紧那侧（`rate > 0`，快速加载 = 被猛拽 / 绳被绷直的冲击）**要罚**；
    - 卸载且**已经掉到 ≈0**（`mag_t < slack_eps_n`：绳松弛 / 杆卸载过零）**放行**
      ——`down` 那一支被同一个 `slack` 掩码关掉；
    - **仍是负载态（`mag_t ≥ slack_eps_n`）却断崖式掉力**才罚（`down` 那一支）：
      那是杆行/负载在"卸载到一半"时被抽掉力，对应真实的冲击。

    ------------------------------------------------------------------
    公式（`mag = ‖towing_force_b‖`，与 `obs_towing_force` 同量纲 N；`dt = env.step_dt`）
    ------------------------------------------------------------------
    ```
    rate    = (mag_t − mag_{t−1}) / dt                        # N/s，20 Hz 有限差分
    up      = relu(rate − rate_limit_n_per_s) / rate_limit_n_per_s
    slack   = mag_t < slack_eps_n                             # 卸载到 ≈0：绳松弛 / 杆卸载
    down    = relu(−rate − release_limit_n_per_s) / release_limit_n_per_s * (1 − slack)
    penalty = clamp(up + down, 0, clip_max)                   # 无量纲
    ```

    参数（默认值 = 用户 2026-10-10 建议值）：

    | 参数 | 默认 | 含义 |
    |---|---|---|
    | `rate_limit_n_per_s` | **200.0** | 加载侧的免罚速率上限（N/s）；**调参只需要改它** |
    | `release_limit_n_per_s` | **600.0** | 卸载侧的免罚速率上限（更宽松：绳松弛本身允许突变，这里只兜住"负载态断崖"） |
    | `slack_eps_n` | **1.0** | 与既有 `towing_force_norm_active` 的 1 N 阈值**同口径**（源码一致，不是另立数值） |
    | `clip_max` | **3.0** | 无量纲上限，防止极端峰值把回报尺度拉爆 |

    **标定依据**：`up` 在 `rate = 2 × rate_limit` 时 = 1.0 ⇒ 每步代价
    `−0.5 × 1.0 × step_dt(0.05) = −0.025`，即**约等于 `tracking_velocity` 满额（+0.05/步）的
    一半**——"一次明显猛拽"约等于半秒的跟速收益，有动机但不压倒跟踪项。默认行为：
    **"快速绷紧"必罚**；"卸载到 0"放行；"仍是负载态却断崖掉力"才罚。

    ------------------------------------------------------------------
    `mag_{t−1}` 从哪来 / 首拍为什么精确为 0
    ------------------------------------------------------------------
    `mag_{t−1}` 缓存在动作项 `HierarchicalVelocityAction.prev_force_mag` 上，**只在
    `process_actions`（每个控制步一次，与 `_previous/_processed` 同一处）更新**，
    不在 200 Hz 物理子步里更新——奖励与动作同节拍（20 Hz）才有唯一的「上一拍」。
    `reset()` 清 `prev_force_mag` / `force_rate_has_prev` / `_force_rate_tick_seen`
    ⇒ **每回合第一拍的 rate 精确为 0**（那一拍没有上一拍样本，`torch.where` 把它置 0），
    与仓库其它项「spawn 精确为 0」的纪律一致：出生那一拍不会因为"从 0 到出生力"被误罚。
    这**不是掩护真实猛拽的漏洞**：出生段是 settle（指令 0，`SETTLE_TIME_S = 1.0 s` ⇒
    20 个控制步），真正的起步绷直发生在第几十拍、早过了首拍门。

    ------------------------------------------------------------------
    ⚠ 已知限制（写进文档，不当缺陷）：20 Hz 采样看不到 5 ms 级的冲量尖峰
    ------------------------------------------------------------------
    控制步长 `step_dt = 0.05 s`，而物理步长只有 5 ms：**一次在一个控制步内完成
    "猛拽→回弹"的冲量在 20 Hz 的 `mag_t` 上可能与"缓慢加载到同一水平"完全一样**，
    本项对它没有梯度。这就是诊断项 `obs_force_rate_max`（200 Hz 子步的
    `max |Δmag| / dt_phys`）存在的理由：它用来量化"被 20 Hz 采样漏掉了多少"。

    ⚠ **符号约定（以源码为准）**：`towing_force_b` 是**作用在机器人挂点上的力**（base 机体系），
    `mag` 是它的**模长**，因此只对两套绳成立"力恒为拉力"。`RigidLink` 的
    `rope_tension` 是**有符号的**（`mdp/rope_model.py`：负值 = 推力），杆被压时
    `force_on_robot` 整个反向 ⇒ `mag` 无法区分"拉"与"推"。但杆**过零**（拉→推 / 推→拉）
    必然经过 `mag ≈ 0 < slack_eps_n` ⇒ **同一个 `slack` 掩码覆盖过零那一下**，
    不会把"杆卸载穿过零点"罚成突变；而穿过零点之后在另一侧重新快速加载，按上面的
    `up` 规则处理（那确实是"快速绷紧"）。**若将来要区分拉/推两向的加载**，需要改的是
    有符号投影而不是这个掩码——目前不做（用户要求量的是幅值）。

    ⚠ 无小车环境（`cart_present == False`）这里**不再乘掩码**：`_apply_towing_physics`
    已把 `towing_force_b` 整体乘 `present` ⇒ 那些 env 的 `mag ≡ 0`、rate ≡ 0、本项恒 0，
    与显式掩码等价（`reset` 也清零）。
    """
    term = _term(env)
    mag = torch.linalg.vector_norm(term.towing_force_b, dim=1)
    rate = (mag - term.prev_force_mag) / env.step_dt
    # 刚 reset 的第一拍没有「上一拍」样本 ⇒ rate 精确为 0（不是拿 0 当上一拍去算）。
    rate = torch.where(term.force_rate_has_prev, rate, torch.zeros_like(rate))
    up = torch.relu(rate - rate_limit_n_per_s) / rate_limit_n_per_s
    slack = mag < slack_eps_n
    down = (torch.relu(-rate - release_limit_n_per_s) / release_limit_n_per_s
            * (~slack).to(rate.dtype))
    return torch.clamp(up + down, 0.0, clip_max)


def stop_reached_flag(env):
    """本回合**是否走到过 STOP 点**（1.0 = 到过，0.0 = 没到过）。只进 TensorBoard。

    用户要求（2026-10-09）：回合一律由 timeout 收尾，所以"有没有走到 STOP 点"必须靠统计量
    回答，而不是靠终止原因。取的是 action term 里的**粘性标志** `_was_stopped`：它在进度
    越过 `stop_distance_m` 那一拍置位，直到本 env 复位才清零，所以

    - `Episode_Reward/obs_stop_reached ÷ 1e-6` = 该回合**处于 STOP 相位的步数占比**（时间占比）；
    - **占比 > 0 就等价于"走到了"**（到过之后才会计数）；恒 0 的回合就是没走到，
      它们全部由 `time_out` 收尾（`Episode_Termination/time_out` 里含这部分）。

    与 `cart_present_flag` 一致，用 1e-6 权重注册：不能填 0，否则 `RewardManager` 直接
    `continue`、连日志都不产生。
    """
    return _term(env)._was_stopped.float()


def terrain_out_of_bounds(env):
    """End the episode before either body reaches the disconnected slab edge."""
    term = _term(env)

    def outside(asset):
        local = asset.data.root_pos_w - env.scene.env_origins
        return ((local[:, 0] < -BACK_M + BOUNDARY_MARGIN_M)
                | (local[:, 0] > FORWARD_M - BOUNDARY_MARGIN_M)
                | (local[:, 1].abs() > HALF_WIDTH_M - BOUNDARY_MARGIN_M))

    return outside(term._asset) | (outside(term._cart) & term.cart_present[:, 0])


# ---------------------------------------------------------------------------
# 日志专用量（以 weight=0 的 reward term 注册，只进 TensorBoard 的 Episode_* 统计，
# 不影响总回报）。它们覆盖既有 reward 项无法回答的验收问题：无小车环境比例、绳力大小与
# 绳力实际作用的时刻。注意 `RewardManager` 会把每个 term 的回合和除以
# `max_episode_length_s`，所以这里取均值的量在日志里是「每步均值」。

def cart_present_flag(env):
    """1.0 = 本环境有小车，0.0 = 无小车零负载环境（期望均值≈0.875）。"""
    return _term(env).cart_present[:, 0].float()


def towing_force_norm(env):
    """机体系牵引力大小（N）。"""
    term = _term(env)
    return torch.linalg.vector_norm(term.towing_force_b, dim=1)


def towing_force_norm_active(env):
    """仅在绳力>1 N 时给值，其余置零，避免把「无拉力」稀释平均力度。

    用来区分两种情形：全程无拉力（该值也≈0 且 `towing_force_norm`≈0）与
    「有拉力但只有少数步有效」（该值明显>1 而 `towing_force_norm` 被稀释）。"""
    return torch.where(towing_force_norm(env) > 1.0, towing_force_norm(env),
                       torch.zeros_like(towing_force_norm(env)))


def obs_force_rate_max(env):
    """**诊断**：200 Hz 物理子步内的 `max |Δ‖F‖| / dt_phys`（N/s）；每控制步取最大、**读后清零**。

    存在的理由（TOW-25）：`towing_force_rate_penalty` 的 rate 是 **20 Hz** 的有限差分
    （`step_dt = 0.05 s`），而物理步长只有 **5 ms**——**一次在一个控制步内完成的
    "猛拽→回弹"冲量在 20 Hz 采样上可能与"缓慢加载到同一水平"完全一样**，惩罚项对它
    没有梯度。本诊断量用 200 Hz 的 `|Δmag| / dt_phys` 给出同一段过程里**最大的那一跳**，
    用来量化"被 20 Hz 采样漏掉了多少"：若它在 TensorBoard 上远大于 `up` 的
    `rate_limit_n_per_s`（200 N/s），说明真实冲量主要发生在子步尺度、只能靠提高
    控制频率（或改物理侧限速）解决，而不是继续加这个奖励的权重。

    实现：`_apply_towing_physics` 每物理子步 `max` 累积、本函数读取后把缓存清零
    （`clone()` 再 `zero_()`，返回的是清零前的值）⇒ `Episode_Reward/obs_force_rate_max ÷ 1e-6`
    是每控制步那个 max 的回合和。`reset()` 同步清零，出生那一拍不会带着上回合的尖峰。

    ⚠ 与所有 `1e-6` 诊断项一样：不参与策略优化，只进 TensorBoard；`weight` 不能填 0
    （`RewardManager` 对 0 权重直接 `continue`，连日志都不产生）。
    """
    term = _term(env)
    value = term.force_rate_max.clone()
    term.force_rate_max.zero_()
    return value



try:
    from isaaclab.envs.mdp import time_out
except ImportError:
    def time_out(env): return env.episode_length_buf >= env.max_episode_length - 1


# Isaac Lab 提供的事件函数（机器人侧域随机化，参数与 AMP vel 跟踪任务逐项一致）。
# 与上面的 `time_out` 同一处理：本模块被读取做静态检查时不需要 Isaac Lab，
# 缺依赖时留 None，只有真正构造环境（有 Isaac Lab）才会用到。
try:
    from isaaclab.envs.mdp import (
        apply_external_force_torque,
        randomize_actuator_gains,
        randomize_rigid_body_com,
        randomize_rigid_body_mass,
    )
except ImportError:  # pragma: no cover - 只在没有 Isaac Lab 的离线机器上走到
    apply_external_force_torque = None
    randomize_actuator_gains = None
    randomize_rigid_body_com = None
    randomize_rigid_body_mass = None
