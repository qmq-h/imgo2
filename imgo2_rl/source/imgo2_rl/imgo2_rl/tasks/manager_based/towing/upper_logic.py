"""Simulator-independent contracts for the upper towing policy."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class UpperActionSpec:
    """12 维归一化关节残差：``deltapos = residual_scale ⊙ clip(u, -1, 1)``。

    2026-10-08 由「3 维归一化加速度 → 积分成参考速度」改为**关节位置残差**。送给冻结
    底层策略的速度指令不再由上层网络积分产生，而是由脚本调度直接给出（与测量台
    ``tow_drag.py`` 的 ``user_cmd`` 逐位一致，其记录里 ``user_cmd == ref_cmd``），
    上层网络只输出叠加在冻结策略关节目标上的增量：

        joint_cmd = default_dof_pos + action_scale · (loco_action + u)

    残差尺度取 ``LowLevelPolicyCfg.action_scale``（12 个逐关节值），与底层动作同量纲。
    这样残余的物理幅值由冻结策略自己的动作范围界定，不需要另立一套尺度常数。
    """

    residual_scale: tuple[float, ...]
    control_dt: float = 0.05

    def validate(self):
        if not self.residual_scale:
            raise ValueError("residual_scale 不能为空")
        if not all(math.isfinite(value) for value in self.residual_scale):
            raise ValueError("residual_scale 必须全部为有限数")
        if any(value <= 0.0 for value in self.residual_scale):
            raise ValueError("residual_scale 必须全为正（否则该关节的残差通道被静默关闭）")
        if not math.isfinite(self.control_dt) or self.control_dt <= 0.0:
            raise ValueError("control_dt 必须为有限正数")

    def delta_joint_pos(self, action):
        """把 |u| ≤ 1 的归一化动作映射为逐关节位置增量（rad，策略关节顺序）。"""
        self.validate()
        if len(action) != len(self.residual_scale):
            raise ValueError(
                f"upper action 维数 {len(action)} != residual_scale 的 {len(self.residual_scale)}")
        return tuple(scale * min(1.0, max(-1.0, float(value)))
                     for value, scale in zip(action, self.residual_scale))


@dataclass(frozen=True)
class UpperObservationSpec:
    """Raw recurrent input plus the deployable decoder estimate.

    ``loco_command`` 是**脚本调度出来、实际送给冻结策略**的速度指令。2026-10-08 改残差
    方案后，它同时就是任务指令（v0.1 无 command shaping），所以只保留这一项；原先并列的
    ``cmd_vel``／``reference_command`` 已合并（两份的值恒等，属冗余观测）。
    """

    terms: tuple[tuple[str, int], ...] = (
        ("loco_command", 3),     # [vx, vy, yaw_rate]，送冻结策略
        ("last_action", 12),     # 上一拍 12 维关节残差（策略关节顺序）
        ("base_ang_vel", 3),     # body-frame IMU gyroscope
        ("projected_gravity", 3),
        ("last_loco_action", 12),
        ("joint_pos", 12),
        ("joint_vel", 12),
    )

    latent_dim: int = 16

    @property
    def decoder_dim(self):
        """decoder 输出维数。唯一定义在 ``DecoderSpec``，避免两处各自漂移。"""
        return DecoderSpec().dim

    @property
    def frame_dim(self):
        return sum(dim for _, dim in self.terms)

    @property
    def actor_dim(self):
        return self.frame_dim + self.decoder_dim + self.latent_dim


@dataclass(frozen=True)
class DecoderSpec:
    """GRU decoder targets used only during training.

    2026-10-08 用户决定：**牵引力 3 维（机体系 x/y/z）、机器人速度保持 2 维（x/y）**。
    竖直分力不是小量——两挂点高差 0.17 m，绷紧时方向向量 z 分量 = 0.17/L0，
    在 L0 = 0.4…0.8 m 上就是张力的 21–43%，且 Fz 在 0.16 m 挂点后臂上产生俯仰力矩。
    速度则只保留水平两轴：`root_lin_vel_b[:, 2]` 在 trot 步态下按步频大幅振荡，
    不是负载量，而上层也没有垂直控制通道。
    """

    terms: tuple[tuple[str, int], ...] = (
        ("robot_velocity_xy", 2),
        ("cart_mass", 1),
        ("towing_force_xyz", 3),
    )

    @property
    def dim(self):
        return sum(dim for _, dim in self.terms)

    # 2026-10-09 训练侧上限由 15 提到 30 kg（`UpperEventsCfg.reset_work_condition`），
    # 这里同步，避免契约与训练配置漂移（该字段只作文档/派生用，不参与张量归一化）。
    mass_range: tuple[float, float] = (5.0, 30.0)
    velocity_scale: tuple[float, float] = (1.0, 0.5)
    force_scale: float = 10.0


def normalize_decoder_targets(values, spec: DecoderSpec = DecoderSpec()):
    """已废弃：decoder target 改为直接回归物理量，不再归一化。

    2026-09-23 去掉 target 归一化与 head 的 tanh。保留此函数只为兼容旧调用点，
    它现在**原样返回**输入（仍做维数检查）。不要再新增调用。
    """
    if len(values) != spec.dim:
        raise ValueError(f"decoder target 维数 {len(values)} != {spec.dim}")
    return tuple(float(v) for v in values)


def denormalize_force(value, spec: DecoderSpec = DecoderSpec()):
    """已废弃：力 head 现在直接输出牛顿，无需反归一化。原样返回。"""
    return float(value)


def scheduled_command(elapsed_s, tow_speed, *, tow_start_s=1.0, stop_time_s=5.0):
    """Return the v0.1 longitudinal command for settle, tow, and zero-command phases.

    2026-10-08 起这是**送给冻结底层策略的指令本身**（不再是 network shaping 的目标）：
    ``upper_mdp.process_actions`` 用它逐拍写 ``loco_command``，与测量台 ``tow_drag.py``
    的 ``user_cmd`` 相位一致（settle 0 → tow 速度 → STOP 后 0）。
    """
    if not 0.0 <= tow_start_s < stop_time_s:
        raise ValueError("必须满足 0 <= tow_start_s < stop_time_s")
    if not math.isfinite(elapsed_s) or not math.isfinite(tow_speed) or tow_speed < 0.0:
        raise ValueError("elapsed_s 和 tow_speed 必须为有限非负数")
    return float(tow_speed) if tow_start_s <= elapsed_s < stop_time_s else 0.0


def classify_training_case(*, valid: bool, tracking_ratio: float, tracking_min: float = 0.8) -> str:
    """Training-domain gate shared with the boundary-scan semantics."""
    return "feasible" if valid and tracking_ratio >= tracking_min else "infeasible"
