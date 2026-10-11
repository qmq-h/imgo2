"""三种可切换的拖曳连接模型：compliant（弹性绳）、inextensible（低弹性单边绳）、rigid（双边球铰连杆）。

## 为什么需要三种，而不是把 k 调大

三者最大的区别**不是刚度大小**，而是物理机制：

* `CompliantRope`：单边弹簧-阻尼，绳可以**伸长并储存弹性能**，绷直靠 `k·δ + c·ḋ`；
* `InextensibleRope`：单边距离约束 `d ≤ L0, T ≥ 0, T(L0 − d) = 0`，**不允许明显伸长**，
  Slack→Taut 的速度突变由**约束冲量**处理；
* `RigidLink`：**双边**距离约束 `d ≡ L0`，拉与推都传递，压缩时小车反向驱动机器人（反驱）。

把 compliant 的 `k` 调到 1×10⁶ N/m 并不能代替后者：实测（`docs/towing_p4_tow_drag_2026-09-20.md`
§5.19）在 dt=5 ms 下显式弹簧要么给出 373~1528 N 的猛拽、要么直接失稳（k ≥ 1.6×10⁶ 时
步长上限 3.8 ms），而且它仍然是「靠伸长储能」的机制。同理，把绳调硬也只得到「拉」，
永远得不到连杆的「推」——双边性是 `RigidLink` 独有的新物理。

## 统一接口

三种模型都实现 `RopeModel.update(...) -> RopeSample`，字段完全一致：

    rope_length / rope_extension / rope_tension / rope_length_rate
    / is_taut / rope_state / rope_impulse

于是 locomotion、上层策略、logger 与 reward 都不需要因换模型而改。力一律作用在
**实际挂点**（不是质心），以保留绳力对机器人 pitch 的影响 —— 与 P4 已有的施力方式一致。

`rope_extension` 对两套绳是 `max(0, d − L0)`（松弛不报负值），对 `RigidLink` **保留符号**
（负值 = 压缩量）；`rope_tension` 对两套绳恒 `≥ 0`，对 `RigidLink` 有符号（负值 = 推力）。

## 逐环境分配（训练时按类型分批）

为了让同一个训练里三类环境各占一批，用 `MultiRopeModel`：它按一个**整数 id 张量**把
各模型的结果**逐 env 拼起来**（纯算术混合，所以本模块仍然不需要 torch）。示意：

    ids = torch.randint(0, 3, (num_envs,))   # 0=compliant 1=inextensible 2=rigid
    rope = MultiRopeModel(models=(compliant, inextensible, rigid), model_ids=ids)

旧的 `SplitRopeModel` 保留为 `MultiRopeModel` 的两模型特例（两套绳长度必须一致）。

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
        config_value,
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
        config_value,
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
    """绳索模型基类：只声明统一接口，具体物理在子类。

    `rest_length` 可以是标量，也可以是逐环境数组（确定性网格里每个 env 的长度不同，
    见 `connection_grid.py`）；数组只做类型门、不逐元素扫，见 `rope.config_value`。
    """

    name = "abstract"

    def __init__(self, *, rest_length):
        self.rest_length = config_value("rest_length", rest_length, minimum=0.0)

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

    def __init__(self, *, rest_length, stiffness, damping):
        super().__init__(rest_length=rest_length)
        # 逐环境数组也接受（网格里同一列共用一档 k/c，但按 env 张量传入）
        self.stiffness = config_value("stiffness", stiffness, strictly_positive=True)
        self.damping = config_value("damping", damping, minimum=0.0)

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

    def __init__(self, *, rest_length, position_gain: float = 0.2,
                 max_correction_rate: float = 0.2):
        super().__init__(rest_length=rest_length)
        self.position_gain = config_value("position_gain", position_gain, minimum=0.0)
        if _is_number(self.position_gain) and self.position_gain > 1.0:
            raise ValueError("position_gain 必须 ≤ 1（否则会过冲）")
        self.max_correction_rate = config_value("max_correction_rate", max_correction_rate,
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


# ------------------------------------------------------------------ 方案 C
def attachment_horizontal_gap(target_distance, *, delta_z):
    """由**目标三维挂点距**反解 spawn 的**水平**间距 `h = sqrt(target² − Δz²)`（勾股解）。

    三类连接在 spawn 时都必须让两挂点的**三维**距离等于各自的目标值，否则第一物理步就有
    初始约束力（绳是预张紧、刚体是初始压缩，L=0.4 m 却按 0.8 m 布局时会把小车往后踹）。
    两挂点有竖直高差 `Δz`（机器人 base 与小车 base_link 的出生高度不同），所以水平间距不是
    目标距离本身。抽成纯算术函数，好让这条只有跑仿真才会执行的摆放几何能在离线测试里算到
    （`upper_mdp.reset_towing_episode` 对三类连接统一调用它）。

    `target_distance` / `delta_z` 可以是标量或张量；越界（target ≤ |Δz|）时钳到 0
    （调用方另做显式校验）。
    """
    return _maximum(target_distance * target_distance - delta_z * delta_z, 0.0) ** 0.5


class RigidLink(RopeModel):
    """方案 C：球铰刚性连杆（固定两挂点距离、允许绕挂点自由转动，**可拉可推**）。

    与两套绳模型最本质的差别是**双边性**：

        绳  ：d ≤ L0, T ≥ 0        只能拉，松弛后不再作用
        连杆：d ≡ L0, T ∈ ℝ        拉与推都传递；压缩时小车反向驱动机器人（反驱）

    实现沿用 inextensible 的「速度级约束 + 位置反馈，显式写成力」，但**不取正部**：

        C = d − L0
        target_rate = clamp(−β·C/dt, ±max_correction_rate)
        J = (ḋ − target_rate) / k_eff
        T = J / dt

    `C > 0`（被拉长）⇒ `target_rate < 0` ⇒ `J > 0` ⇒ 拉力；`C < 0`（被压缩）⇒ `target_rate > 0`
    ⇒ `J < 0` ⇒ 推力。力仍按 `e`（机器人 → 小车）给出：`F_R = T·e`，压缩时 `F_R` 指向机器人
    **前方** ⇒ 小车把机器人往前推，即反驱；`F_cart = −F_R` 把小车往后推。

    `is_taut` 恒为真：连杆不存在 slack，永远处于啮合状态。

    **已知限制**：它和 inextensible 一样靠外力等效约束，稳态下会留
    `C ≈ T·k_eff·dt²/β` 的柔度（名义参数、T=10 N 时约 7 mm），不是无穷刚度。要真正无穷刚性
    需要 PhysX 关节级约束，而机器人与小车是两个独立 articulation，当前架构只能用外力。
    本模型要的是**双边性（反驱）**，不是极限刚度；这一点在文档里必须写清，不能宣称「理想刚体」。

    `rest_length` 可以是标量，也可以是逐环境的 `(N,)` 张量/数组（训练里连杆长度按 env 随机化，
    见 `upper_mdp.py`）。张量形态不走标量校验，由调用方保证非负。
    """

    name = "rigid"

    def __init__(self, *, rest_length, position_gain: float = 0.2,
                 max_correction_rate: float = 0.2):
        super().__init__(rest_length=rest_length)          # 标量或逐环境数组
        self.position_gain = config_value("position_gain", position_gain, minimum=0.0)
        if _is_number(self.position_gain) and self.position_gain > 1.0:
            raise ValueError("position_gain 必须 ≤ 1（否则会过冲）")
        self.max_correction_rate = config_value("max_correction_rate", max_correction_rate,
                                               minimum=0.0)

    def update(self, *, robot_point, cart_point, robot_velocity, cart_velocity, dt,
               robot: BodyProperties | None = None,
               cart: BodyProperties | None = None) -> RopeSample:
        dt = _validate("dt", dt, strictly_positive=True)
        if robot is None or cart is None:
            raise ValueError("rigid 模型需要两侧的 mass / inverse_inertia_world / offset")
        distance, direction, rate = self._geometry(robot_point, cart_point,
                                                   robot_velocity, cart_velocity)
        violation = distance - self.rest_length
        # 双边：目标速率可以是正（推）也可以是负（拉），只受 max_correction_rate 夹幅
        target_rate = _clamp(-self.position_gain * violation / dt,
                             -self.max_correction_rate, self.max_correction_rate)
        impulse = (rate - target_rate) / effective_inverse_mass(direction, robot, cart)
        tension = impulse / dt
        force_on_robot = tuple(tension * component for component in direction)
        force_on_cart = tuple(-component for component in force_on_robot)
        return RopeSample(rope_length=distance,
                          # 与绳不同：连杆的伸长量**保留符号**，负值就是压缩量
                          rope_extension=violation,
                          rope_tension=tension, rope_length_rate=rate,
                          is_taut=abs(impulse) >= 0.0,          # 永远啮合，无 slack
                          rope_state=_state_name(True, impulse),
                          rope_impulse=impulse, direction=direction,
                          force_on_robot=force_on_robot, force_on_cart=force_on_cart)


# ------------------------------------------------------------------ 逐环境分配
def _masked_sum(masks, values):
    """`Σ_i mask_i · value_i`（纯算术，标量/numpy/torch 通用）。"""
    total = 0.0
    for mask, value in zip(masks, values):
        total = total + mask * value
    return total


class MultiRopeModel(RopeModel):
    """按**整数 id** 在 N 个模型间逐环境选择（N ≥ 2）。

        out = Σ_i [ (model_ids == i) · model_i(...) ]

    实现是「每个模型都算一遍，再按 id 掩码混合」，纯算术 ⇒ 本模块仍不需要 torch；
    `model_ids` 可以是标量、numpy 数组或 torch 张量（`(N,)` / `(N, 1)`）。

    与旧的 `SplitRopeModel` 的关键差别：**不要求各模型 `rest_length` 相同**（刚体连杆长度
    逐 env 随机化，与绳长 0.8 m 不同），因此 `rope_extension` 也必须混合，不能像以前那样
    「几何量取其中一套」——长度不同时两套的伸长量本就不同。`rope_length` /
    `rope_length_rate` / `direction` 只由挂点几何决定、与模型无关，取第一套。

    `rope_state` 在批量下无法逐 env 给字符串，统一返回 `MIXED`；逐 env 判定请用 `is_taut`。
    """

    name = "multi"

    def __init__(self, *, models, model_ids):
        models = tuple(models)
        if len(models) < 2:
            raise ValueError("MultiRopeModel 至少需要两个模型；单模型请直接用 make_rope_model")
        self.models = models
        self.model_ids = model_ids
        # 各模型长度可以不同，基类的标量 rest_length 在这里没有统一含义。
        self.rest_length = None

    def update(self, *, robot_point, cart_point, robot_velocity, cart_velocity, dt,
               robot=None, cart=None) -> RopeSample:
        samples = [model.update(robot_point=robot_point, cart_point=cart_point,
                                robot_velocity=robot_velocity, cart_velocity=cart_velocity,
                                dt=dt, robot=robot, cart=cart)
                   for model in self.models]
        masks = [(self.model_ids == index) * 1.0 for index in range(len(samples))]
        first = samples[0]
        return RopeSample(
            rope_length=first.rope_length,                 # 几何量只由挂点决定
            rope_length_rate=first.rope_length_rate,
            direction=first.direction,
            # 长度可能逐模型不同 ⇒ 伸长量必须混合
            rope_extension=_masked_sum(masks, [s.rope_extension for s in samples]),
            rope_tension=_masked_sum(masks, [s.rope_tension for s in samples]),
            rope_impulse=_masked_sum(masks, [s.rope_impulse for s in samples]),
            is_taut=_masked_sum(masks, [_as_binary(s.is_taut) for s in samples]),
            rope_state=MIXED,
            force_on_robot=tuple(_masked_sum(masks, [s.force_on_robot[axis] for s in samples])
                                 for axis in range(3)),
            force_on_cart=tuple(_masked_sum(masks, [s.force_on_cart[axis] for s in samples])
                                for axis in range(3)),
        )


class SplitRopeModel(MultiRopeModel):
    """两套绳索模型按 0/1 掩码逐环境拼接（历史接口，= `MultiRopeModel` 的两模型特例）。

    保留独立类名与构造签名是为了不改既有调用方与测试；两套绳的 `rest_length` 必须一致
    （`d ≤ L0` 与 `d > L0` 的边界统一）。长度不同的连接（刚体连杆）请用 `MultiRopeModel`。
    """

    name = "split"

    def __init__(self, *, compliant: CompliantRope, inextensible: InextensibleRope,
                 inextensible_mask):
        if compliant.rest_length != inextensible.rest_length:
            raise ValueError("两套模型的 rest_length 必须一致，否则 d≤L0 与 d>L0 的边界不统一")
        super().__init__(models=(compliant, inextensible), model_ids=inextensible_mask)
        self.compliant = compliant
        self.inextensible = inextensible
        self.inextensible_mask = inextensible_mask
        self.rest_length = compliant.rest_length


# 两套绳：都只有「拉」，`ROPE_MODELS` 的语义保持为绳，既有的单边性测试仍全部适用。
ROPE_MODELS = ("compliant", "inextensible")
# 刚体球铰连杆：双边（可反驱），不是绳。
RIGID_MODEL = "rigid"
# 三类可切换的拖曳连接模型；`make_rope_model` 与 `--rope-model` 都接受这三者。
CONNECTION_MODELS = ROPE_MODELS + (RIGID_MODEL,)


def make_rope_model(name: str, *, rest_length, stiffness: float | None = None,
                    damping: float | None = None, position_gain: float = 0.2,
                    max_correction_rate: float = 0.2) -> RopeModel:
    """按连接类型建模型：`"compliant" | "inextensible" | "rigid"`（统一入口）。

    `rest_length` 对前两者是标量，对 `rigid` 可以是逐环境张量（见 `RigidLink`）。
    """
    if name == "compliant":
        if stiffness is None or damping is None:
            raise ValueError("compliant 模型需要 stiffness 与 damping")
        return CompliantRope(rest_length=rest_length, stiffness=stiffness, damping=damping)
    if name == "inextensible":
        return InextensibleRope(rest_length=rest_length, position_gain=position_gain,
                                max_correction_rate=max_correction_rate)
    if name == RIGID_MODEL:
        return RigidLink(rest_length=rest_length, position_gain=position_gain,
                         max_correction_rate=max_correction_rate)
    raise ValueError(f"未知的连接模型 {name!r}；可选 {CONNECTION_MODELS}")
