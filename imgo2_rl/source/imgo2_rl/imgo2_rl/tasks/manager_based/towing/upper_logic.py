"""Simulator-independent contracts for the upper towing policy."""

from dataclasses import dataclass
import math


#: 双头动作布局的切分点：``u = cat(u_cmd(1), u_joint(12))``。
#: 切分点唯一定义在这里，`upper_mdp` 与 `upper_policy_runtime` 都引用它，避免两处各自漂移。
CMD_ACTION_DIM = 1


@dataclass(frozen=True)
class UpperActionSpec:
    """双头动作：**1 维 vx 偏移** + **12 维归一化关节残差**。

    2026-10-08 由「3 维归一化加速度 → 积分成参考速度」改为**关节位置残差**；2026-10-10
    再在其前面加一个 **cmd vel 头**（用户批准的双头方案），动作 12 → 13：

        u        = cat(u_cmd(1), u_joint(12))
        offset   = clip(u_cmd, ±1) × cmd_offset_scale，再限在 [offset_min, offset_max]
        loco_vx  = clamp(task_vx + offset, amp_vx_min, amp_vx_max)
        delta    = clip(u_joint, ±1) ⊙ residual_scale

    两头的语义与分工（`docs/towing_upper_two_head_2026-10-10.md`）：

    - ``u_joint``：叠加在冻结策略关节目标上的位置增量（力矩级、单步级），负责抗击冲击；
    - ``u_cmd``：叠加在**脚本调度**的 vx 上的偏移（行为级、秒级），负责「STOP 之后还走不走」。
      偏移**只在 `elapsed_s >= tow_start_s` 后生效**：出生段（settle）不许推机器人。

    **加性 + 有界，但限的两层是分开的**（2026-10-10 从源码核对后确定）：

    1. ``offset_min`` / ``offset_max`` 限的是**偏移量本身**（头权限，默认 −0.2 / +0.6）；
    2. ``amp_vx_range`` 限的是**合成后的 vx**，取冻结 AMP 策略的**训练指令包络**
       `lin_vel_x = (−1.0, 1.5)`（出处 `base_move/amp_env_cfg.py:123`）。超出该包络即
       OOD，冻结策略的步态会退化。

    ⚠ **绝不能**把"和"裁到偏移头的 `[−0.2, +0.6]`：拖曳脚本速度是 0.4–1.5 m/s
    （`episode_geometry.SPEED_RANGE`），牵引段本来就顶在 AMP 上界 1.5，那样会把正常牵引指令
    压成 0.6 而奖励参考量（`task_command`）仍是 1.5 ⇒ 策略被要求跟一个它根本发不出的速度。
    反过来，`amp_vx_range` 必须**覆盖整个脚本速度范围**（`upper_env_cfg.__post_init__` 有断言），
    这样 `u_cmd = 0` 时 `clamp(task_vx, ...) == task_vx`，与旧口径**逐位一致**。

    语义后果（写进文档与日志）：牵引段速度高时正半轴偏移会被包络钳掉（1.5 m/s 时 +0.5 无梯度），
    偏移实际只能**减小** vx —— 正好用于「绷直前降速、少灌冲击能量」；STOP 之后 `task_vx = 0`，
    偏移头在同一包络内可正可负（可倒走制造松弛）。因此偏移量建议**按相位分开记日志**。

    残差尺度取 ``LowLevelPolicyCfg.action_scale``（12 个逐关节值），与底层动作同量纲。
    这样残余的物理幅值由冻结策略自己的动作范围界定，不需要另立一套尺度常数。
    """

    residual_scale: tuple[float, ...]
    control_dt: float = 0.05
    # ---- 2026-10-10 新增：1 维 vx 偏移头（全部是 cfg 字段，默认值 = 用户建议值）----
    #: 偏移头的归一化尺度（m/s）：offset = clip(u_cmd, ±1) × cmd_offset_scale。
    cmd_offset_scale: float = 0.5
    #: 偏移下限（m/s）。scale 0.5 时负向先触界：u_cmd ≤ −0.4 就被压在 −0.2。
    offset_min: float = -0.2
    #: 偏移上限（m/s）。scale 0.5 时正向最多 +0.5，本值 0.6 是留出的富余。
    offset_max: float = 0.6
    #: **合成后 vx 的训练包络**（m/s）= 冻结 AMP 策略的 `commands.base_velocity.ranges.lin_vel_x`
    #: = (−1.0, 1.5)，出处 `tasks/manager_based/locomotion/velocity/base_move/amp_env_cfg.py`。
    #: 它必须覆盖脚本速度范围 0.4–1.5，否则零偏移时脚本自己就被裁掉（破坏退化性）。
    amp_vx_range: tuple[float, float] = (-1.0, 1.5)

    def validate(self):
        if not self.residual_scale:
            raise ValueError("residual_scale 不能为空")
        if not all(math.isfinite(value) for value in self.residual_scale):
            raise ValueError("residual_scale 必须全部为有限数")
        if any(value <= 0.0 for value in self.residual_scale):
            raise ValueError("residual_scale 必须全为正（否则该关节的残差通道被静默关闭）")
        if not math.isfinite(self.control_dt) or self.control_dt <= 0.0:
            raise ValueError("control_dt 必须为有限正数")
        if not math.isfinite(self.cmd_offset_scale) or self.cmd_offset_scale <= 0.0:
            raise ValueError("cmd_offset_scale 必须为有限正数")
        if not (math.isfinite(self.offset_min) and math.isfinite(self.offset_max)):
            raise ValueError("offset_min / offset_max 必须为有限数")
        if self.offset_min >= self.offset_max:
            raise ValueError("必须满足 offset_min < offset_max")
        self.validate_amp_vx_range()

    def validate_amp_vx_range(self):
        """训练包络必须是有限、有序、**包含 0** 的区间。"""
        low, high = self.amp_vx_range
        if not (math.isfinite(low) and math.isfinite(high)):
            raise ValueError("amp_vx_range 必须为有限数")
        if low >= high:
            raise ValueError("必须满足 amp_vx_range[0] < amp_vx_range[1]")
        if not (low <= 0.0 <= high):
            raise ValueError("amp_vx_range 必须包含 0（否则 settle 的零指令会被裁成非零）")

    @property
    def joint_action_dim(self):
        """关节残差头的维数（= 冻结策略关节数，当前 12）。"""
        return len(self.residual_scale)

    @property
    def action_dim(self):
        """动作总维数：1 维 vx 偏移 + 12 维关节残差 = 13。"""
        return CMD_ACTION_DIM + len(self.residual_scale)

    def split_action(self, action):
        """把 13 维归一化动作切成 ``(u_cmd(1), u_joint(12))``，各自 clamp 到 ±1。

        拆分的唯一定义处（`upper_mdp.process_actions` 的张量版与运行时都必须与此一致）。
        """
        self.validate()
        if len(action) != self.action_dim:
            raise ValueError(
                f"upper action 维数 {len(action)} != {self.action_dim}"
                f"（{CMD_ACTION_DIM} 维 vx 偏移 + {self.joint_action_dim} 维关节残差）")
        values = tuple(min(1.0, max(-1.0, float(value))) for value in action)
        return values[:CMD_ACTION_DIM], values[CMD_ACTION_DIM:]

    def command_offset(self, action):
        """``clip(u_cmd, ±1) × cmd_offset_scale``（m/s，未做 [min, max] 限幅）。"""
        command, _ = self.split_action(action)
        return command[0] * self.cmd_offset_scale

    def bounded_command_offset(self, action):
        """把 vx 偏移限在 ``[offset_min, offset_max]``（m/s）。"""
        offset = self.command_offset(action)
        return min(self.offset_max, max(self.offset_min, offset))

    def loco_command_vx(self, action, scripted_vx, *, apply_offset=True):
        """送给冻结策略的 vx = ``clamp(脚本 vx + 有界偏移, amp_vx_min, amp_vx_max)``。

        两层限幅（顺序不能反）：

        1. 偏移先限在头权限 ``[offset_min, offset_max]``；
        2. **和**再限在冻结 AMP 策略的训练包络 ``amp_vx_range``（−1.0, 1.5），超出即 OOD。

        ``apply_offset=False`` 即出生段（`elapsed_s < tow_start_s`）：偏移不生效，返回值等于
        ``clamp(scripted_vx, ...)``；由于 `amp_vx_range` 覆盖脚本速度范围（0.4–1.5），
        这就是脚本值本身，与旧口径**逐位一致**。
        """
        offset = self.bounded_command_offset(action) if apply_offset else 0.0
        low, high = self.amp_vx_range
        return min(high, max(low, float(scripted_vx) + offset))

    def delta_joint_pos(self, action):
        """把 |u| ≤ 1 的归一化**关节残差**映射为逐关节位置增量（rad，策略关节顺序）。

        入参是 12 维关节残差（不是 13 维完整动作）：本方法语义自 2026-10-08 起未变，
        新增的 cmd vel 头由 `split_action` / `bounded_command_offset` 单独处理。
        """
        self.validate()
        if len(action) != len(self.residual_scale):
            raise ValueError(
                f"upper action 维数 {len(action)} != residual_scale 的 {len(self.residual_scale)}")
        return tuple(scale * min(1.0, max(-1.0, float(value)))
                     for value, scale in zip(action, self.residual_scale))


@dataclass(frozen=True)
class UpperObservationSpec:
    """Raw recurrent input plus the deployable decoder estimate.

    ``loco_command`` 是**实际送给冻结策略**的速度指令 = ``task_command + 有界偏移``
    （2026-10-10 起是双头动作的第一个头写出来的，策略因此看得见自己下了什么指令）。

    ``last_action`` 2026-10-10 由 12 维变 **13 维**（1 维 vx 偏移 + 12 维关节残差），
    因此帧 57 → **58**、actor 79 → **80**、decoder ``frame_dim`` 57 → 58，
    checkpoint 契约 v2 → **v3**（旧 checkpoint 必须被拒，按仓库惯例新建 run）。

    **防作弊红线**：奖励**只能**读 ``task_command``（脚本调度、到点归零），绝不能读
    ``loco_command``（含策略自己的偏移），否则策略只要把偏移开大就能自己给自己发目标。
    """

    terms: tuple[tuple[str, int], ...] = (
        ("loco_command", 3),     # [vx, vy, yaw_rate]，送冻结策略 = task_command + 有界偏移
        ("last_action", 13),     # 上一拍 13 维动作（1 维 vx 偏移 + 12 维关节残差）
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


def scripted_vx(elapsed_s, tow_speed, *, tow_start_s=1.0, stop_time_s=math.inf, ramp_s=0.0):
    """``task_command[:, 0]``（脚本纵向指令）的**纯函数镜像**，含可选的 STOP ramp。

    `upper_mdp.process_actions` 的张量版逐拍写 ``task_command``，奖励只读它。本函数把
    同一套相位逻辑抽成纯函数，让「ramp 关闭时 == 旧口径」与「ramp 打开时线性降到 0」
    都能在没有 Isaac Lab 的机器上直接验算：

    - ``elapsed_s < tow_start_s``：0（settle，出生段）；
    - ``tow_start_s <= elapsed_s < stop_time_s``：``tow_speed``（tow）；
    - ``elapsed_s >= stop_time_s``：``ramp_s > 0`` 时从 ``tow_speed`` 在 ``ramp_s`` 秒内
      线性降到 0；``ramp_s = 0``（默认）时**立即**为 0 —— 与 2026-10-10 之前的脚本调度
      **逐位一致**。

    ⚠ 训练侧 STOP 由**进度**触发（`stop_distance_m`，逐 env 不同时刻），逐 env 的
    ``stop_time_s`` 是那一拍写入的；本函数只是同一逻辑的单 env 标量形式。
    """
    if not 0.0 <= tow_start_s < stop_time_s:
        raise ValueError("必须满足 0 <= tow_start_s < stop_time_s")
    if not math.isfinite(elapsed_s) or not math.isfinite(tow_speed) or tow_speed < 0.0:
        raise ValueError("elapsed_s 和 tow_speed 必须为有限非负数")
    if not math.isfinite(ramp_s) or ramp_s < 0.0:
        raise ValueError("ramp_s 必须为有限非负数")
    if elapsed_s < tow_start_s:
        return 0.0
    if elapsed_s < stop_time_s:
        return float(tow_speed)
    if ramp_s > 0.0:
        fraction = max(0.0, min(1.0, 1.0 - (elapsed_s - stop_time_s) / ramp_s))
        return float(tow_speed) * fraction
    return 0.0


def classify_training_case(*, valid: bool, tracking_ratio: float, tracking_min: float = 0.8) -> str:
    """Training-domain gate shared with the boundary-scan semantics."""
    return "feasible" if valid and tracking_ratio >= tracking_min else "infeasible"
