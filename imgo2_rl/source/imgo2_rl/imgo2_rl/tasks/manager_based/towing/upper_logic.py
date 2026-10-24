"""Simulator-independent contracts for the upper towing policy."""

from dataclasses import dataclass
import math


#: 2026-10-24 起动作只有 12 维关节残差；原 1 维 vx 偏移头已删除。
#: 保留常量名会让旧 import 静默变成 0 维切片，因此代码里不再定义 `CMD_ACTION_DIM`。


def effective_vx_clip(task_range, policy_range):
    """把「任务包络」裁进「冻结低层训练包络」，越界**直接报错**而不是静默放宽。

    2026-10-17 的教训：拖曳侧原先把 `amp_vx_range` 硬编码成 AMP 的 (−1.0, 1.5)，换了 PPO 之后
    仍用它当上限 ⇒ 2.0 m/s 的指令被静默裁成 1.5，跑出来「1.5 与 2.0 一模一样」其实是**同一个
    输入**。包络必须按 `policy_name` 取（`policy_cfg.LowLevelPolicyCfg.vx_envelope_mps`），任务
    包络只允许比它更窄。
    """
    task_lo, task_hi = (float(task_range[0]), float(task_range[1]))
    policy_lo, policy_hi = (float(policy_range[0]), float(policy_range[1]))
    if task_lo > task_hi or policy_lo > policy_hi:
        raise ValueError(f"包络上下界颠倒：task={task_range}, policy={policy_range}")
    if task_lo < policy_lo - 1e-9 or task_hi > policy_hi + 1e-9:
        raise ValueError(
            f"任务包络 {tuple(task_range)} 超出冻结低层的训练包络 {tuple(policy_range)}："
            f"超出部分会被低层当成分布外输入（实测会退化成站着不动），"
            f"请改 `amp_vx_range` 或换一条训练域更宽的冻结策略")
    return (task_lo, task_hi)


@dataclass(frozen=True)
class UpperActionSpec:
    """**12 维归一化关节位置残差**（2026-10-24 起去掉 1 维 vx 偏移头）。

    历史：2026-10-08 由 3 维加速度积分改为关节残差；2026-10-10 在残差前加 1 维 cmd vel
    偏移头（动作 12→13）；2026-10-24 用户要求去掉该偏移，动作回到 **12**。速度指令完全由
    脚本调度（+ 横向/朝向 PD）给出，不再允许 actor 改写 vx。

        u     = u_joint(12)
        delta = clip(u, ±1) ⊙ residual_scale

    残差尺度取 ``LowLevelPolicyCfg.action_scale``（12 个逐关节值），与底层动作同量纲。
    """

    residual_scale: tuple[float, ...]
    control_dt: float = 0.05
    #: 脚本速度的任务包络；仍是冻结低层训练包络的子区间（由 `effective_vx_clip` 校验）。
    #: 没有偏移头后它只用于给脚本 vx 做分布约束，不再与动作相加。
    amp_vx_range: tuple[float, float] = (-1.0, 1.0)

    def validate(self):
        if not self.residual_scale:
            raise ValueError("residual_scale 不能为空")
        if not all(math.isfinite(value) for value in self.residual_scale):
            raise ValueError("residual_scale 必须全部为有限数")
        if any(value <= 0.0 for value in self.residual_scale):
            raise ValueError("residual_scale 必须全为正（否则该关节的残差通道被静默关闭）")
        if not math.isfinite(self.control_dt) or self.control_dt <= 0.0:
            raise ValueError("control_dt 必须为有限正数")
        self.validate_amp_vx_range()

    def validate_amp_vx_range(self):
        """任务速度包络必须是有限、有序、**包含 0** 的区间。"""
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
        """动作总维数：只有 12 维关节残差（v6 起没有 vx 偏移头）。"""
        return len(self.residual_scale)

    def delta_joint_pos(self, action):
        """把 |u| ≤ 1 的 12 维归一化关节残差映射为逐关节位置增量（rad，策略关节顺序）。"""
        self.validate()
        if len(action) != len(self.residual_scale):
            raise ValueError(
                f"upper action 维数 {len(action)} != residual_scale 的 {len(self.residual_scale)}")
        return tuple(scale * min(1.0, max(-1.0, float(value)))
                     for value, scale in zip(action, self.residual_scale))

    def clip_loco_vx(self, scripted_vx):
        """把脚本 vx 裁进任务包络（没有偏移头，不再做加法）。"""
        low, high = self.amp_vx_range
        return min(high, max(low, float(scripted_vx)))


@dataclass(frozen=True)
class UpperObservationSpec:
    """Raw recurrent input plus the deployable decoder estimate.

    ``loco_command`` 是**实际送给冻结策略**的速度指令 = ``task_command``（2026-10-24 起
    不再叠加 actor 的 vx 偏移；vy/wz 仍由 PD 外环给出）。

    2026-10-24 去掉 1 维 vx 偏移头后，``last_action`` 回到 **12 维**、帧回到 **57**、
    decoder ``frame_dim`` 57、actor **78**，checkpoint 契约升 **v6**。旧 v5（58/13/79）
    及更早 checkpoint 一律失效，需重训。

    2026-10-16：decoder 去掉显式质量预测 ⇒ explicit 6 → **5**、契约 v4；2026-10-17 合并
    CE Net 后升 v5（显式 5 + 重建下一帧，42 维重建掩码）。2026-10-24 去掉 vx 偏移头后
    帧 58→57、actor 79→78，契约升 **v6**；v2/v3/v4/v5 checkpoint 一律失效，必须重训。

    **防作弊红线**：奖励**只能**读 ``task_command``（脚本调度、到点归零），绝不能读
    ``loco_command``（含策略自己的偏移），否则策略只要把偏移开大就能自己给自己发目标。
    """

    terms: tuple[tuple[str, int], ...] = (
        ("loco_command", 3),     # [vx, vy, yaw_rate]，送冻结策略 = task_command + 有界偏移
        ("last_action", 12),     # 上一拍 12 维关节残差（2026-10-24 去掉 vx 偏移头）
        ("base_ang_vel", 3),     # body-frame IMU gyroscope
        ("projected_gravity", 3),
        ("last_loco_action", 12),
        ("joint_pos", 12),
        ("joint_vel", 12),
    )

    latent_dim: int = 16
    #: 下一帧重建要**排除**的帧项（2026-10-17 CE Net 引入重建后新增）。
    #:
    #: 这两个项在 ``o[t+1]`` 里由**策略自己下一拍的动作**决定，而 decoder 在 t 时刻只看到
    #: ``o_t``（Isaac Lab 的 ``step()`` 顺序是 ``process_action → 物理 → reset → 算观测``，
    #: 所以 ``o_t`` 里的 ``last_action`` 是 ``a_t``，``o[t+1]`` 里的是 ``a_{t+1}``）：
    #: - ``last_action``（12 维）= ``a_{t+1}`` 本身 ⇒ 训练时是策略采样，**不可预测**；
    #: - ``loco_command``（3 维）= ``task_command``（2026-10-24 起无偏移），vx 可预测，
    #:   但整项一起排除更简单、也更保守。
    #: 只对剩下的 **42 维**（ang_vel/gravity/last_loco_action/joint_pos/joint_vel）算重建
    #: ⇒ 去掉不可消除的 MSE 地板（≈0.05）与那 16 维上的噪声梯度。
    #: 详见 docs/towing/towing_ce_net_2026-10-17.md §5①。
    reconstruction_excluded_terms: tuple[str, ...] = ("loco_command", "last_action")

    @property
    def decoder_dim(self):
        """decoder 输出维数。唯一定义在 ``DecoderSpec``，避免两处各自漂移。"""
        return DecoderSpec().dim

    @property
    def frame_dim(self):
        return sum(dim for _, dim in self.terms)

    def reconstruction_mask(self) -> tuple[bool, ...]:
        """逐维 bool：``True`` = 参与下一帧重建（排除 ``reconstruction_excluded_terms``）。"""
        excluded = set(self.reconstruction_excluded_terms)
        mask = []
        for name, dim in self.terms:
            mask.extend([name not in excluded] * int(dim))
        if len(mask) != self.frame_dim:
            raise ValueError(
                f"重建掩码长度 {len(mask)} != 帧维数 {self.frame_dim}（terms 定义被改坏？）")
        return tuple(mask)

    @property
    def reconstruction_dim(self) -> int:
        """参与下一帧重建的维数（当前 57 − 12 − 3 = **42**）。"""
        return sum(self.reconstruction_mask())

    @property
    def actor_dim(self):
        return self.frame_dim + self.decoder_dim + self.latent_dim


@dataclass(frozen=True)
class DecoderSpec:
    """GRU decoder targets used only during training（**5 维**，2026-10-17 契约 v5）。

    准则（用户 2026-10-16）：「decoder 只建模与四足机器人直接相关的量，质量用 latent 假定」
    ⇒ 只留**机器人自身速度**（机体系 x/y）与**作用在机器人上的牵引力**（机体系 x/y/z），
    **不再显式预测小车质量**（质量是负载的属性；actor 通过 latent 间接使用它）。

    - **牵引力 3 维**（2026-10-08 用户决定）：竖直分力不是小量——两挂点高差 Δz=0.22 m，绷紧时
      方向向量 z 分量 = Δz/L0，在 L0 = 0.5…1.5 m 上就是张力的 15–44%，且 Fz 在 0.16 m
      挂点后臂上产生俯仰力矩。
    - **速度 2 维**：`root_lin_vel_b[:, 2]` 在 trot 步态下按步频大幅振荡，不是负载量，
      而上层也没有垂直控制通道。
    """

    terms: tuple[tuple[str, int], ...] = (
        ("robot_velocity_xy", 2),
        ("towing_force_xyz", 3),
    )

    @property
    def dim(self):
        return sum(dim for _, dim in self.terms)

    # 2026-10-16：`mass_range` 字段随质量头一起删除——质量不再是 decoder 的目标，把工作域
    # 留在 decoder spec 里会误导。工作域（质量 5–30 kg / 速度 0.5–1.0 m/s）的权威位置是
    # `upper_env_cfg.UpperEventsCfg.reset_work_condition`（环境）与
    # `episode_geometry.SPEED_RANGE`（速度），测量台用 `TRAINING_MASS_RANGE_KG` 对齐。
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
