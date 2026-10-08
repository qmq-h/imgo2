"""Manager terms for upper towing RL.

架构（2026-10-08 起）：冻结 AMP 底层策略的速度指令由**脚本调度**给出（``loco_command``），
上层网络输出 **12 维关节位置残差**，叠加在冻结策略的关节目标上：

    user cmd(脚本) → 冻结 locomotion → joint targets
                                          + deltapos(上层 20 Hz) → joint position command

观测／奖励函数对特权数据的取舍是刻意显式的。本文件是**目标函数与动作适配器**的唯一定义处；
任务注册见同目录 ``__init__.py``，运行验收清单见 ``docs/towing_training_prep_2026-09-22.md``。
"""

from dataclasses import MISSING

import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

from .upper_logic import UpperActionSpec
from .mdp.connection_grid import ELASTIC_KC, GRID_SIZE, env_spec, is_full_grid
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


class HierarchicalVelocityAction(ActionTerm):
    cfg: "HierarchicalVelocityActionCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._cart = env.scene[cfg.cart_asset_name]
        self._collision_sensors = tuple(env.scene.sensors[name] for name in cfg.collision_sensor_names)
        # 冻结策略契约先加载：残差尺度直接取它的 `action_scale`（与底层动作同量纲），
        # 动作维数取它的关节数（AMP 为 12）。见 `UpperActionSpec` 的说明。
        self._policy_cfg = get_policy(cfg.policy_name)
        self._action_dim = len(self._policy_cfg.joint_names)
        if len(self._policy_cfg.action_scale) != self._action_dim:
            raise ValueError(
                f"{cfg.policy_name} 契约的 action_scale {len(self._policy_cfg.action_scale)} 项与 "
                f"joint_names {self._action_dim} 项不一致")
        self._action_spec = UpperActionSpec(
            residual_scale=tuple(self._policy_cfg.action_scale),
            control_dt=cfg.upper_control_dt,
        )
        self._action_spec.validate()
        self._residual_scale = torch.tensor(
            self._policy_cfg.action_scale, dtype=torch.float32, device=env.device)
        self._raw = torch.zeros(env.num_envs, self._action_dim, device=env.device)
        self._processed = torch.zeros_like(self._raw)
        self._previous = torch.zeros_like(self._raw)
        # 12 维关节位置残差（**策略关节顺序**）。`apply_actions` 每次刷新冻结策略输出时
        # 把它加到 `joint_targets` 上；两次上层更新之间保持不变。
        self.delta_joint_pos = torch.zeros_like(self._raw)
        # 实际送给冻结底层策略的速度指令：脚本调度产生，不再由上层动作积分（v0.1 无 command shaping）。
        self.loco_command = torch.zeros(env.num_envs, 3, device=env.device)
        self.tow_speed = torch.full((env.num_envs,), cfg.initial_tow_speed, device=env.device)
        self.tow_start_s = torch.full((env.num_envs,), cfg.tow_start_s, device=env.device)
        self.stop_time_s = torch.full((env.num_envs,), cfg.initial_stop_time_s, device=env.device)
        self.last_loco_action = torch.zeros(env.num_envs, 12, device=env.device)
        # [clearance, tension, extension, taut]. Collision is deliberately separate: tautness is
        # a rope state and must never double as a contact flag.
        self.rope_state = torch.zeros(env.num_envs, 4, device=env.device)
        # 绳子对机器人的力，**base 机体系 3 维**（x 前后 / y 左右 / z 竖直）。
        # 2026-10-08 由 2 维改为 3 维：两挂点高差 Δz=0.17 m，绷紧时 |Fz| 占张力的
        # 0.17/L0 = 21–43%，且 Fz 在 -0.16 m 挂点上产生俯仰力矩，丢掉它对 actor／惩罚都不可见。
        self.towing_force_b = torch.zeros(env.num_envs, 3, device=env.device)
        self.cart_collision = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.stop_origin_x = self._asset.data.root_pos_w[:, 0].clone()
        self._was_stopped = torch.linalg.vector_norm(self.loco_command, dim=1) <= 1.0e-4
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
                  f"只覆盖网格前缀；正式训练请用 {GRID_SIZE}（20 列 × 20 行）的整数倍")
        specs = [env_spec(index) for index in range(env.num_envs)]
        self.rope_model_id = torch.tensor(
            [[spec["model_index"]] for spec in specs], dtype=torch.float32, device=env.device)
        # 连接长度：绳是 L0、刚体是杆长 L（同一行的数值相同）。
        self.connection_length = torch.tensor(
            [[spec["length"]] for spec in specs], dtype=torch.float32, device=env.device)
        # spawn 时的目标三维挂点距：绳 = 0.5·L0（留松弛）、刚体 = L。摆放小车时用。
        self.initial_distance = torch.tensor(
            [[spec["initial_distance"]] for spec in specs],
            dtype=torch.float32, device=env.device)
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
        robot_body_ids, _ = self._asset.find_bodies([cfg.robot_body_name])
        cart_body_ids, _ = self._cart.find_bodies([cfg.cart_body_name])
        wheel_joint_ids, _ = self._cart.find_joints(list(cfg.cart_wheel_joint_names), preserve_order=True)
        wheel_body_ids, _ = self._cart.find_bodies(list(cfg.cart_wheel_body_names), preserve_order=True)
        if (len(robot_body_ids) != 1 or len(cart_body_ids) != 1
                or len(wheel_joint_ids) != 4 or len(wheel_body_ids) != 4):
            raise RuntimeError("upper towing adapter requires one robot base, one cart base, and four wheels")
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
        """12：冻结策略每个关节一个归一化残差（策略关节顺序）。"""
        return self._action_dim

    @property
    def raw_actions(self):
        return self._raw

    @property
    def processed_actions(self):
        return self._processed

    def process_actions(self, actions):
        """每上层控制步（50 ms）更新一次：脚本速度指令 + 12 维关节残差。

        速度指令不再由网络积分产生——上层网络的输出**只是**残差。指令相位与测量台
        ``tow_drag.py`` 一致：settle 段 0、牵引段 ``tow_speed``、STOP 后 0。
        """
        elapsed_s = self._env.episode_length_buf * self._env.step_dt
        towing = (elapsed_s >= self.tow_start_s) & (elapsed_s < self.stop_time_s)
        self.loco_command.zero_()
        self.loco_command[:, 0] = torch.where(towing, self.tow_speed, 0.0)
        stopped = torch.linalg.vector_norm(self.loco_command, dim=1) <= 1.0e-4
        newly_stopped = stopped & ~self._was_stopped
        self.stop_origin_x[newly_stopped] = self._asset.data.root_pos_w[newly_stopped, 0]
        self._was_stopped.copy_(stopped)
        self._previous.copy_(self._processed)
        self._raw.copy_(actions)
        self._processed.copy_(actions.clamp(-1.0, 1.0))
        # 归一化残差 → 逐关节位置增量（rad）。尺度取冻结策略的 action_scale，
        # 因此等价于在底层动作空间上叠一个同量纲偏移。
        self.delta_joint_pos.copy_(self._processed * self._residual_scale)

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
        self.loco_command[env_ids] = 0
        self.last_loco_action[env_ids] = 0
        self.rope_state[env_ids] = 0
        self.towing_force_b[env_ids] = 0
        self._held_joint_targets[env_ids] = self._asset.data.default_joint_pos[env_ids]
        self.cart_collision[env_ids] = False
        self.stop_origin_x[env_ids] = self._asset.data.root_pos_w[env_ids, 0]
        self._was_stopped[env_ids] = False
        self._policy.reset(env_ids)


@configclass
class HierarchicalVelocityActionCfg(ActionTermCfg):
    """上层动作 = 12 维归一化关节残差，叠加在冻结策略的关节位置目标上。

    原先的 `acceleration_min/max` 与 `reference_min/max` 已删除：动作不再是加速度积分，
    残差尺度直接取冻结策略契约的 `action_scale`（见 `UpperActionSpec`）。
    """

    class_type: type[ActionTerm] = HierarchicalVelocityAction
    asset_name: str = "robot"
    cart_asset_name: str = "cart"
    policy_name: str = "amp"
    upper_control_dt: float = 0.05
    low_level_control_dt: float = 0.02
    physics_dt: float = 0.005
    initial_tow_speed: float = 0.5
    tow_start_s: float = 1.0
    initial_stop_time_s: float = 5.0
    initial_cart_mass: float = 10.0
    initial_ground_friction: float = 0.8
    initial_wheel_damping: float = 0.032
    robot_body_name: str = "base"
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
    env, env_ids, *, speed_range, stop_time_range, mass_range, friction_range,
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
    stop_times = sample(stop_time_range)
    masses_kg = sample(mass_range)
    friction = sample(friction_range)
    wheel_damping = sample(wheel_damping_range)
    term.tow_speed[env_ids] = speeds
    term.stop_time_s[env_ids] = stop_times
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
    robot_state[:, :3] += env.scene.env_origins[env_ids]
    robot_state[:, 0] += sample(robot_x_range)
    robot_state[:, 1] += sample(robot_y_range)
    yaw = sample(robot_yaw_range)
    yaw_delta = math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    robot_state[:, 3:7] = math_utils.quat_mul(robot_state[:, 3:7], yaw_delta)
    robot_state[:, 7:13] = 0
    robot.write_root_pose_to_sim(robot_state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(robot_state[:, 7:13], env_ids=env_ids)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos[env_ids],
                                   torch.zeros_like(robot.data.default_joint_vel[env_ids]), env_ids=env_ids)

    cart_state = cart.data.default_root_state[env_ids].clone()
    cart_state[:, :3] += env.scene.env_origins[env_ids]
    # 三类连接都按本 env 的目标三维挂点距摆放小车（**不再依赖场景里写死的 cart.init_state.pos**）：
    # 绳 = 0.5·L0（留松弛）、刚体 = L（杆正好是 L）。否则第一物理步就有初始约束力（绳预张紧、
    # 刚体初始压缩）。解析解：水平分量 = sqrt(target² − Δz²)，方向取机器人正后方；机器人挂点用
    # **实际** spawn 位姿（含 x/y/yaw 随机化）算，故初始违反量与抖动无关。
    robot_attach_w = math_utils.quat_apply(
        robot_state[:, 3:7], term._robot_attach.expand(count, 3))
    delta_z = robot_state[:, 2] - cart_state[:, 2]
    if bool((target_distance ** 2 <= delta_z ** 2).any()):
        raise ValueError("连接长度必须大于两挂点高差，否则水平摆放无解")
    horizontal = attachment_horizontal_gap(target_distance, delta_z=delta_z)
    cart_state[:, 0] = (robot_state[:, 0] + robot_attach_w[:, 0]
                        - horizontal - term.cfg.cart_attachment[0])
    cart_state[:, 1] = robot_state[:, 1] + robot_attach_w[:, 1]
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
    """送给冻结底层策略的速度指令（脚本调度；与任务指令同值，v0.1 无 command shaping）。"""
    return _term(env).loco_command
def upper_last_action(env): return _term(env).processed_actions
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
def policy_frame(env):
    """One complete actor frame; history is applied once to preserve frame-major ordering.

    2026-10-08 起为 **57 维**：命令项只留 `loco_command`（原来并列的 `cmd_vel`／
    `reference_command` 在残差方案下恒等，属冗余），`last_action` 由 3 维变 12 维。
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
def velocity_tracking_exp(env, linear_std, yaw_std):
    """**实际速度** vs 命令期望的指数跟踪项（唯一的正奖励）。

    2026-10-08 修正：线性项原先比的是 ``reference_command − user_command``，也就是「上层
    自己积分出来的指令 vs 任务指令」——这与本函数的名字、以及 `upper_env_cfg` 奖励文档的
    描述（"实际速度 vs 命令期望"）都不符，而且和 `reference_tracking_l2` 重复。残差方案下
    指令由脚本给出、`reference_command` 不复存在，故直接改成实测机体系线速度：

        exp(−‖v_meas,xy − cmd_xy‖²/0.5² − (ω_z − cmd_yaw)²/1.0²)

    这也是测量台的验收指标口径（``summarize_tow`` 的 ``steady_tracking_ratio`` =
    实测 vx ÷ 指令速度）。
    """
    term = _term(env)
    linear_error = (term._asset.data.root_lin_vel_b[:, :2] - term.loco_command[:, :2]) / linear_std
    yaw_error = (term._asset.data.root_ang_vel_b[:, 2] - term.loco_command[:, 2]) / yaw_std
    return torch.exp(-(linear_error.square().sum(1) + yaw_error.square()))
def clearance_barrier(env, warning_distance, scale):
    term = _term(env); term.update_safety_state(); clearance = term.rope_state[:, 0]
    return (torch.nn.functional.softplus((warning_distance - clearance) / scale)
            * term.cart_present[:, 0])
def min_clearance_violation(env, ratio, softness=0.02):
    """铰链式「最小间距」惩罚：间隙低于 `ratio × 本 env 的连接长度` 时线性加大。

    为什么单独加一项（而不是复用 `clearance_barrier`）：`clearance_barrier` 是
    softplus 软障碍，其"警戒距离"是绝对量（0.20 m）且线性区在警戒线**下方**；本项是
    按**连接长度比例**给出的硬阈值：高于阈值恒为 0、低于阈值与缺口成正比，语义是
    「不得拉得太近」，与「近了要缓」互补。

    **阈值必须逐 env 用连接长度**（2026-10-08 起）：场景是 20 行长度 0.4–0.8 m 的网格，
    写死 `rope_length=0.8` 会让短绳行（L0=0.4 时 spawn 间隙只有约 0.108 m）一开局就低于
    阈值 0.32 m 而满额惩罚。用逐 env 的 L0/L 后，ratio=0.25 在**所有行**的 spawn 都低于
    实际间隙（最小行 0.108 m > 0.1 m），保持「初始不生效」。

    历史（2026-09-23，单一 L0=0.8 时）：初始「后表面→车斗」间隙约 0.349 m，ratio 由
    0.6 降到 0.40 才让 spawn 不触发；这次随网格改为 0.25。
    """
    term = _term(env)
    term.update_safety_state()
    clearance = term.rope_state[:, 0]
    threshold = ratio * term.connection_length[:, 0]
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


def post_stop_towing_force(env, force_scale):
    term = _term(env)
    elapsed_s = env.episode_length_buf * env.step_dt
    post_stop = (elapsed_s >= term.stop_time_s).float()
    force = torch.linalg.vector_norm(term.towing_force_b, dim=1)
    normalized_force = force / (force + force_scale)
    return normalized_force * post_stop * term.cart_present[:, 0]
def post_stop_distance(env):
    term = _term(env)
    elapsed_s = env.episode_length_buf * env.step_dt
    post_stop = (elapsed_s >= term.stop_time_s).float()
    return torch.relu(term._asset.data.root_pos_w[:, 0] - term.stop_origin_x) * post_stop
def action_magnitude_l2(env):
    """上层动作幅值平方（裁剪后），用于抑制动作抖动。

    与 `action_rate_l2` 的区别：后者罚"动作变化量"（平滑性），本项罚"动作绝对值"
    （幅值）。策略长期输出 ±1 饱和随机方波时，两项都会变大，但本项能直接压住幅值，
    把 12 维关节残差压回小幅度、让冻结步态保持主导。

    2026-10-08 语义变化：动作由 3 维速度指令增量变为 12 维关节残差，`Σ u²` 的上限从 3
    变成 12（同等逐维幅值下惩罚约 ×4），权重 `-0.05` 是否仍合适**尚未实跑验证**。
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
    yaw, _pitch, _roll = math_utils.euler_xyz_from_quat(term._asset.data.root_quat_w)
    yaw_init, _pitch0, _roll0 = math_utils.euler_xyz_from_quat(
        term._asset.data.default_root_state[:, 3:7])
    return math_utils.wrap_to_pi(yaw - yaw_init)


def yaw_heading_l2(env):
    """朝向误差平方。τ=1 时立姿为 0；偏 0.2 rad(11°) → 1，偏 0.5 rad(29°) → 6.25。

    平方形式在误差大时梯度更大（误差每增 1 rad 梯度增 2·误差），适合先把自转压住。
    """
    return heading_deviation(env).square()


def action_rate_l2(env):
    term = _term(env); return (term.processed_actions - term._previous).square().sum(1)
def robot_fall(env, minimum_height): return _term(env)._asset.data.root_pos_w[:, 2] < minimum_height
def robot_fall_cost(env, minimum_height): return robot_fall(env, minimum_height).float()


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



try:
    from isaaclab.envs.mdp import time_out
except ImportError:
    def time_out(env): return env.episode_length_buf >= env.max_episode_length - 1
