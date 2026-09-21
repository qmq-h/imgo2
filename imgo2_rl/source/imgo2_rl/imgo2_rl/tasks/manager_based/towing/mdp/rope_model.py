"""两套可切换的绳索模型：compliant（弹簧-阻尼）与 inextensible（单边距离约束）。

## 为什么需要两套，而不是把 k 调大

两者最大的区别**不是刚度大小**，而是物理机制：

* `CompliantRope`：单边弹簧-阻尼，绳可以**伸长并储存弹性能**，绷直靠 `k·δ + c·ḋ`；
* `InextensibleRope`：单边距离约束 `d ≤ L0, T ≥ 0, T(L0 − d) = 0`，**不允许明显伸长**，
  Slack→Taut 的速度突变由**约束冲量**处理。

把 compliant 的 `k` 调到 1×10⁶ N/m 并不能代替后者：实测（`docs/towing_p4_tow_drag_2026-09-20.md`
§5.19）在 dt=5 ms 下显式弹簧要么给出 373~1528 N 的猛拽、要么直接失稳（k ≥ 1.6×10⁶ 时
步长上限 3.8 ms），而且它仍然是「靠伸长储能」的机制。

## 统一接口

两种模型都实现 `RopeModel.update(...) -> RopeSample`，字段完全一致：

    rope_length / rope_extension / rope_tension / rope_length_rate
    / is_taut / rope_state / rope_impulse

于是 locomotion、上层策略、logger 与 reward 都不需要因换模型而改。力一律作用在
**实际挂点**（不是质心），以保留绳力对机器人 pitch 的影响 —— 与 P4 已有的施力方式一致。

## 逐环境分配（训练时 1:1）

为了让同一个训练里一半环境用 compliant、一半用 inextensible，用 `SplitRopeModel`：
它按一个 0/1 掩码把两套模型的结果**逐 env 拼起来**（纯算术混合，所以本模块仍然
不需要 torch）。示意：

    mask = torch.zeros(num_envs, 1)          # 0 = compliant
    mask[num_envs // 2:] = 1.0               # 1 = inextensible
    rope = SplitRopeModel(compliant=..., inextensible=..., inextensible_mask=mask)

本模块**只用算术**（不导入 torch / Isaac Lab）：标量、numpy 数组、torch 张量都能用。
约定：所有张量的 env 维在最前面（`(N,)` 或 `(N, 3)`），所以把惯量以「**世界系逆惯量**」
的形式传进来（`world_inverse_inertia` 负责由机体对角惯量与姿态旋转算出，元素可以是
`(N,)` 张量）。这样约束算法能在没有仿真器的机器上用小积分器离线复算（见
`imgo2_rl/tests/test_towing_rope_models.py` 的四个最小验证）。
"""

from __future__ import annotations

from typing import NamedTuple

if __package__:
    from .rope import (
        _components,
        _is_number,
        _maximum_zero,
        _norm,
        _validate,
        rope_extension,
        rope_tension,
    )
else:  # 按文件路径直接加载（离线测试的既有做法）时没有包上下文，退回顶层导入
    from rope import (  # type: ignore[no-redef]
        _components,
        _is_number,
        _maximum_zero,
        _norm,
        _validate,
        rope_extension,
        rope_tension,
    )

# rope_state 的取值；批量（多环境）混合模型下为 MIXED
SLACK = "slack"
TAUT = "taut"
MIXED = "mixed"

# 判「有没有产生冲量」用的小正数，避免浮点噪声被记成 taut
IMPULSE_EPSILON = 1e-9


# ------------------------------------------------------------------ 算术工具
# 这几个都写成纯算术，于是 float / numpy / torch 通用（不能用 math.* 或内建 max/min）
def _maximum(left, right):
    return (left + right + abs(left - right)) / 2


def _minimum(left, right):
    return (left + right - abs(left - right)) / 2


def _clamp(value, low, high):
    return _minimum(_maximum(value, low), high)


def _cross3(left, right):
    """3 元组叉积；元素可以是标量或 `(N,)` 张量。

    **不能复用 `rope.cross`**：那个会按标量逐个校验（`_validate`），批量时元素是数组会被拒。
    这里只用算术，因此 float / numpy / torch 的标量与批量都能用。
    """
    lx, ly, lz = left
    rx, ry, rz = right
    return (ly * rz - lz * ry, lz * rx - lx * rz, lx * ry - ly * rx)


def _quadratic_form(vector, matrix):
    """vᵀ M v（v 为 3 元组、M 为 3×3；元素可以是标量或 `(N,)` 张量）。"""
    return sum(vector[i] * matrix[i][j] * vector[j] for i in range(3) for j in range(3))


def _blend(mask, when_one, when_zero):
    """按掩码逐元素选：mask 为 1 取 `when_one`、为 0 取 `when_zero`（纯算术，可批量化）。"""
    return mask * when_one + (1.0 - mask) * when_zero


def _blend_vector(mask, when_one, when_zero):
    return tuple(_blend(mask, a, b) for a, b in zip(when_one, when_zero))


def _as_binary(flag):
    """把 taut 标志转成可参与算术混合的 0/1（标量 bool 或布尔张量都能乘 1.0）。"""
    return flag * 1.0


def _state_name(is_taut, numeric_source) -> str:
    """标量输入给 "slack"/"taut"；批量（张量）输入无法逐 env 给字符串，给 "mixed"。"""
    if not _is_number(numeric_source):
        return MIXED
    return TAUT if bool(is_taut) else SLACK


# ------------------------------------------------------------------ 数据
class BodyProperties(NamedTuple):
    """约束求解需要的一侧刚体属性（都在**世界系**）。

    * `mass`：标量或 `(N,)`；
    * `inverse_inertia_world`：世界系**逆**惯量的 3×3（元素为标量或 `(N,)`），
      用 `world_inverse_inertia(diagonal, rotation)` 从机体对角惯量与姿态旋转得到；
    * `offset`：挂点相对**质心**的世界系偏移（3 元组，元素为标量或 `(N,)`）。

    compliant 模型不需要这些量，可以不传。
    """

    mass: object
    inverse_inertia_world: tuple
    offset: tuple


def world_inverse_inertia(diagonal, rotation):
    """世界系**逆**惯量 `I_w⁻¹ = R · diag(1/I) · Rᵀ`（3×3，元素可为标量或 `(N,)` 张量）。

    `diagonal` = (Ixx, Iyy, Izz) 为**机体主轴系**下的对角惯量（PhysX 给的就是这个），
    `rotation` 为机体→世界的旋转矩阵（3×3）。先取逆再旋转最省事：对角阵求逆只是取倒数。
    """
    inverse_diagonal = tuple(1.0 / value for value in diagonal)
    return tuple(tuple(sum(rotation[i][k] * inverse_diagonal[k] * rotation[j][k] for k in range(3))
                       for j in range(3)) for i in range(3))


def effective_inverse_mass(direction, robot: BodyProperties, cart: BodyProperties):
    """沿绳方向的**等效逆质量** k_eff，使 Δḋ = k_eff · J。

    k_eff = 1/m_R + 1/m_L + (r_R×e)ᵀ I_R⁻¹ (r_R×e) + (r_L×e)ᵀ I_L⁻¹ (r_L×e)

    后两项是力作用在**挂点**（不是质心）带来的转动自由度：挂点离质心越远、且力方向与
    力臂不平行，物体越容易「转」而不是「平移」，等效质量因此更小。名义几何下
    （挂点与绳都在 x 轴上）`r×e = 0`，两项为零；俯仰/竖直偏移时才有贡献。
    """
    total = 1.0 / robot.mass + 1.0 / cart.mass
    for body in (robot, cart):
        arm = _cross3(body.offset, direction)
        total = total + _quadratic_form(arm, body.inverse_inertia_world)
    return total


class RopeSample(NamedTuple):
    """一次更新的完整结果；两种模型字段一致。"""

    rope_length: object
    rope_extension: object
    rope_tension: object
    rope_length_rate: object
    is_taut: object
    rope_state: str
    rope_impulse: object
    direction: tuple
    force_on_robot: tuple
    force_on_cart: tuple


# ------------------------------------------------------------------ 基类
class RopeModel:
    """绳索模型基类：只声明统一接口，具体物理在子类。"""

    name = "abstract"

    def __init__(self, *, rest_length: float):
        self.rest_length = _validate("rest_length", rest_length, minimum=0.0)

    def update(self, *, robot_point, cart_point, robot_velocity, cart_velocity, dt,
               robot: BodyProperties | None = None,
               cart: BodyProperties | None = None) -> RopeSample:
        raise NotImplementedError

    def _geometry(self, robot_point, cart_point, robot_velocity, cart_velocity):
        """共用几何：绳方向 e（机器人 → 负载）与伸长率 ḋ。两种模型共享，保证方向约定一致。"""
        rx, ry, rz = _components(robot_point, "robot_point")
        cx, cy, cz = _components(cart_point, "cart_point")
        vrx, vry, vrz = _components(robot_velocity, "robot_velocity")
        vcx, vcy, vcz = _components(cart_velocity, "cart_velocity")
        delta = (cx - rx, cy - ry, cz - rz)
        distance = _norm(delta)
        if _is_number(distance) and distance <= 0.0:
            raise ValueError("Rope attachment points must not coincide")
        direction = (delta[0] / distance, delta[1] / distance, delta[2] / distance)
        # ḋ = e·(v_L − v_R)：两挂点相对速度在绳方向上的投影
        rate = (direction[0] * (vcx - vrx)
                + direction[1] * (vcy - vry)
                + direction[2] * (vcz - vrz))
        return distance, direction, rate

    @staticmethod
    def _taut_flag(impulse) -> object:
        """由冲量判 taut：标量返回 bool，张量返回逐元素 bool 张量。"""
        if _is_number(impulse):
            return bool(impulse > IMPULSE_EPSILON)
        return impulse > IMPULSE_EPSILON


# ------------------------------------------------------------------ 方案 A
class CompliantRope(RopeModel):
    """方案 A：单边弹簧-阻尼（P3/P4 已有模型，**行为保持不变**）。

        T = 0                          d ≤ L0
          = max(0, k(d − L0) + c·ḋ)    d > L0
    """

    name = "compliant"

    def __init__(self, *, rest_length: float, stiffness: float, damping: float):
        super().__init__(rest_length=rest_length)
        self.stiffness = _validate("stiffness", stiffness, strictly_positive=True)
        self.damping = _validate("damping", damping, minimum=0.0)

    def update(self, *, robot_point, cart_point, robot_velocity, cart_velocity, dt,
               robot=None, cart=None) -> RopeSample:
        _validate("dt", dt, strictly_positive=True)
        distance, direction, rate = self._geometry(robot_point, cart_point,
                                                   robot_velocity, cart_velocity)
        tension = rope_tension(distance, rate, rest_length=self.rest_length,
                               stiffness=self.stiffness, damping=self.damping)
        extension = rope_extension(distance, rest_length=self.rest_length)
        taut = extension > 0
        force_on_robot = tuple(tension * component for component in direction)
        force_on_cart = tuple(-component for component in force_on_robot)
        return RopeSample(rope_length=distance,
                          # 松弛时不报「负伸长」：伸长量只取正部
                          rope_extension=_maximum_zero(extension),
                          rope_tension=tension, rope_length_rate=rate,
                          is_taut=taut,
                          rope_state=_state_name(taut, extension),
                          # J = ∫T dt：本步按力 T 作用 dt 计（与入口的显式欧拉一致）
                          rope_impulse=tension * dt, direction=direction,
                          force_on_robot=force_on_robot, force_on_cart=force_on_cart)


# ------------------------------------------------------------------ 方案 B
class InextensibleRope(RopeModel):
    """方案 B：单边距离约束（不可伸长、无质量）。

        d ≤ L0,  T ≥ 0,  T(L0 − d) = 0

    **做法（速度级约束 + 位置反馈，显式写成力）**：每步开始按当时状态算需要的冲量 `J`，
    再以 `J/dt` 的力作用在挂点上（与 compliant 走同一条施力管线，故力臂由 PhysX 算）。
    单边性由 `J ≥ 0` 保证：需要「推」时（两挂点在靠近）恒为 0。

    每步的算法：

        C = d − L0                                约束违反量（>0 表示已超长）
        ḋ_target = min(ḋ, max(−β·C/dt, −ṁax))      目标相对速率（只夹回拉侧；ṁax 见下）
        J = max(0, (ḋ − ḋ_target) / k_eff)         只移除**向外**的相对速度
        T = J / dt                                 （步平均张力，用于日志）

    力在**步开始时**写入、PhysX 再积分一步，所以这个 J 正是「一步后 ḋ 变成 ḋ_target」
    所需的一步隐式解，不会像弹簧那样来回振荡。`−β·C/dt` 这一项在 `C<0`（还没到 L0）
    但**本步就会超长**时也给出正的 correction，于是 `ḋ_target < ḋ` ⇒ 提前一步施加冲量
    —— 这就是「预测式（speculative）」绷直：`d` 不会真的越过 L0，`rope_extension`
    恒为 0。代价是 `is_taut` 可能在 `d` 略小于 L0 时已经为真（提前量 ≤ 一步），
    这与接触求解里的做法一致，也是「绷直由冲量处理」的正确含义。

    `position_gain` β（0~1）决定绷直接近 L0 的柔和程度与超长时的回拉强度；
    `max_correction_rate` 夹住回拉速度，避免深穿透时一脚踹回去。

    **峰值张力是步平均量**（`T = J/dt`），因此与 dt 相关；**冲量 J 才是稳健的不变量**
    （等于抹掉该相对速度所需的动量变化），日志与判据都以它为准。
    """

    name = "inextensible"

    def __init__(self, *, rest_length: float, position_gain: float = 0.2,
                 max_correction_rate: float = 0.2):
        super().__init__(rest_length=rest_length)
        self.position_gain = _validate("position_gain", position_gain, minimum=0.0)
        if self.position_gain > 1.0:
            raise ValueError("position_gain 必须 ≤ 1（否则会过冲）")
        self.max_correction_rate = _validate("max_correction_rate", max_correction_rate,
                                            minimum=0.0)

    def update(self, *, robot_point, cart_point, robot_velocity, cart_velocity, dt,
               robot: BodyProperties | None = None,
               cart: BodyProperties | None = None) -> RopeSample:
        dt = _validate("dt", dt, strictly_positive=True)
        if robot is None or cart is None:
            raise ValueError("inextensible 模型需要两侧的 mass / inverse_inertia_world / offset")
        distance, direction, rate = self._geometry(robot_point, cart_point,
                                                   robot_velocity, cart_velocity)
        violation = distance - self.rest_length

        # 目标速率：先按位置反馈回拉，夹住幅度，再不允许「比当前更向外」
        # `−β·C/dt` 就是「本步结束时刚好到达 L0」的分离速率（β<1 更柔和），
        # **但只能夹「回拉」那一侧**（负向）。深松弛时它是很大的正数，若一并夹到小正数，
        # 绳子会在完全松弛时就出力——第一版踩了这个：d=0.4、L0=0.8 时报出 350 N。
        allowed_rate = _maximum(-self.position_gain * violation / dt, -self.max_correction_rate)
        target_rate = _minimum(rate, allowed_rate)

        # 单边：只有需要**减小**相对速率（即拉）时才产生冲量
        impulse = _maximum_zero((rate - target_rate) / effective_inverse_mass(direction, robot, cart))
        # 低于阈值就**精确归零**：否则松弛时会留下 1e-14 量级的浮点残差，而离线统计正是用
        # `T == 0.0` 数松弛占比（`tension_slack_fraction`）——残差会让那些统计悄悄失效。
        impulse = impulse * (impulse > IMPULSE_EPSILON)
        tension = impulse / dt
        force_on_robot = tuple(tension * component for component in direction)
        force_on_cart = tuple(-component for component in force_on_robot)
        return RopeSample(rope_length=distance,
                          rope_extension=_maximum_zero(violation),
                          rope_tension=tension, rope_length_rate=rate,
                          is_taut=self._taut_flag(impulse),
                          rope_state=_state_name(self._taut_flag(impulse), impulse),
                          rope_impulse=impulse, direction=direction,
                          force_on_robot=force_on_robot, force_on_cart=force_on_cart)


# ------------------------------------------------------------------ 逐环境分配
class SplitRopeModel(RopeModel):
    """按 0/1 掩码把两套模型**逐环境**拼起来（训练时 1:1 分配用）。

    实现上是「两套都算一遍，再按掩码混合」：

        rope_impulse = mask · J_inextensible + (1 − mask) · J_compliant

    纯算术混合 ⇒ 本模块仍然不需要 torch；`mask` 可以是标量、numpy 数组或 torch 张量，
    形状按 env 维在最前面广播（`(N,)` 或 `(N, 1)`）。绳长/伸长率/方向这些**几何量两套
    完全一样**，所以直接用其中一套的值（不是混合）。

    `rope_state` 在批量下无法逐 env 给字符串，统一返回 `MIXED`；逐 env 判定请用
    `is_taut`（张量）。
    """

    name = "split"

    def __init__(self, *, compliant: CompliantRope, inextensible: InextensibleRope,
                 inextensible_mask):
        if compliant.rest_length != inextensible.rest_length:
            raise ValueError("两套模型的 rest_length 必须一致，否则 d≤L0 与 d>L0 的边界不统一")
        super().__init__(rest_length=compliant.rest_length)
        self.compliant = compliant
        self.inextensible = inextensible
        self.inextensible_mask = inextensible_mask

    def update(self, *, robot_point, cart_point, robot_velocity, cart_velocity, dt,
               robot=None, cart=None) -> RopeSample:
        soft = self.compliant.update(robot_point=robot_point, cart_point=cart_point,
                                     robot_velocity=robot_velocity,
                                     cart_velocity=cart_velocity, dt=dt, robot=robot, cart=cart)
        hard = self.inextensible.update(robot_point=robot_point, cart_point=cart_point,
                                        robot_velocity=robot_velocity,
                                        cart_velocity=cart_velocity, dt=dt, robot=robot, cart=cart)
        mask = self.inextensible_mask
        return RopeSample(
            rope_length=soft.rope_length,                 # 几何量两套一致
            rope_extension=soft.rope_extension,
            rope_length_rate=soft.rope_length_rate,
            direction=soft.direction,
            rope_tension=_blend(mask, hard.rope_tension, soft.rope_tension),
            rope_impulse=_blend(mask, hard.rope_impulse, soft.rope_impulse),
            is_taut=_blend(mask, _as_binary(hard.is_taut), _as_binary(soft.is_taut)),
            rope_state=MIXED,
            force_on_robot=_blend_vector(mask, hard.force_on_robot, soft.force_on_robot),
            force_on_cart=_blend_vector(mask, hard.force_on_cart, soft.force_on_cart),
        )


ROPE_MODELS = ("compliant", "inextensible")


def make_rope_model(name: str, *, rest_length: float, stiffness: float | None = None,
                    damping: float | None = None, position_gain: float = 0.2,
                    max_correction_rate: float = 0.2) -> RopeModel:
    """按 `rope_model = "compliant" | "inextensible"` 建模型（统一入口）。"""
    if name == "compliant":
        if stiffness is None or damping is None:
            raise ValueError("compliant 模型需要 stiffness 与 damping")
        return CompliantRope(rest_length=rest_length, stiffness=stiffness, damping=damping)
    if name == "inextensible":
        return InextensibleRope(rest_length=rest_length, position_gain=position_gain,
                                max_correction_rate=max_correction_rate)
    raise ValueError(f"未知的 rope_model {name!r}；可选 {ROPE_MODELS}")
