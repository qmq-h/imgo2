"""Unilateral rope (spring-damper) between the robot rear attachment and the cart.

研究计划 P3 的实现（`paper_plan_imgo2.md`「P3：实现绳索」）：第一版**不做** rope USD /
多刚体绳，而是在物理步里直接计算虚拟绳力 —— 这样天然支持 slack ↔ taut，并避开地面
接触与弯曲带来的 PhysX 问题。

约定（与计划逐条对应）：

    d  = ||p_L - p_R||            机器人后挂点 p_R、小车挂点 p_L（世界系）
    δ  = d - L0                   绳长 L0
    T  = 0                        δ ≤ 0      （松弛：绳只能拉、不能推）
       = max(0, k·δ + c·ḋ)        δ > 0      （张紧）
    e  = (p_L - p_R) / d
    F_R = +T·e                    施加到机器人后挂点
    F_L = -T·e                    施加到小车挂点

挂点用「世界系点 + 相对质心的力臂」表达：`cart.urdf` 的 `rope_attachment` 是无质量帧
（`base_link` 下 `xyz="0.25 0 0"`、无旋转），Isaac Lab 导入时会把它合并进 `base_link`
（实测警告 `link rope_attachment has no body properties ... merged into base_link`），
所以不能依赖它作为独立 body 存在 —— 必须把力施加到 `base_link` 并补上力臂力矩，
见 `docs/cart_p1_p2_architecture_2026-09-20.md` §3。`offset_torque()` 就是这一步。

本模块只做算术、不导入 torch / Isaac Lab：Python 数、numpy 数组、torch 张量都可用
（与同目录 `resistance.py` 一致），因此能在没有仿真器的机器上复现验证。k / c / L0 是
配置标量、按标量严格校验；几何量可以是张量，此时非有限值仍由调用方在写入仿真前检查
（`scripts/towing/cart_coast.py` 已有同类检查）。
"""

from __future__ import annotations

import math
from typing import NamedTuple


class RopeState(NamedTuple):
    """一次绳力计算的完整结果；向量以 (x, y, z) 分量元组返回。"""

    distance: object
    extension: object
    distance_rate: object
    tension: object
    direction: tuple
    force_on_robot: tuple
    force_on_cart: tuple


def _validate(name, value, *, minimum=None, strictly_positive=False):
    """配置标量校验：必须是有限实数，并按需要限制符号。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if strictly_positive and value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    return value


def _is_number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _components(vector, name):
    """把 3 维向量拆成 (x, y, z) 分量；同时支持序列与 (…, 3) 数组/张量。"""
    if isinstance(vector, (tuple, list)):
        if len(vector) != 3:
            raise ValueError(f"{name} must have exactly 3 components, got {len(vector)}")
        return tuple(_validate(f"{name}[{i}]", c) for i, c in enumerate(vector))
    try:
        return vector[..., 0], vector[..., 1], vector[..., 2]
    except (TypeError, IndexError) as exc:  # pragma: no cover - defensive
        raise TypeError(f"{name} must be a 3-sequence or a (…, 3) array/tensor") from exc


def _maximum_zero(value):
    """max(value, 0)，用算术实现，于是 float / numpy / torch 都支持。"""
    return (value + abs(value)) / 2


def _norm(components):
    """欧氏范数；先夹到非负再做 **0.5，避免浮点误差造成的负数开方。"""
    total = 0
    for component in components:
        total = total + component * component
    return _maximum_zero(total) ** 0.5


def rope_extension(distance, *, rest_length):
    """δ = d - L0。松弛时为负，张紧时为正。"""
    rest_length = _validate("rest_length", rest_length, minimum=0.0)
    if _is_number(distance):
        distance = _validate("distance", distance, minimum=0.0)
    return distance - rest_length


def rope_tension(distance, distance_rate, *, rest_length, stiffness, damping):
    """单侧弹簧阻尼张力 T（N）；标量、数组、张量均可。

    松弛（δ ≤ 0）时恒为 0，**包括刚好绷直（δ = 0）而 ḋ > 0 的边界** —— 计划里的定义是
    `T = 0 if δ ≤ 0`，不连续点落在 δ = 0，故此处不引入阻尼项。张紧时取
    `max(0, k·δ + c·ḋ)`：绳不能推，因此负的阻尼贡献也被截到 0。
    """
    stiffness = _validate("stiffness", stiffness, strictly_positive=True)
    damping = _validate("damping", damping, minimum=0.0)
    extension = rope_extension(distance, rest_length=rest_length)
    if _is_number(distance_rate):
        distance_rate = _validate("distance_rate", distance_rate)
    tension = _maximum_zero(stiffness * extension + damping * distance_rate)
    # 与 taut 掩码相乘：松弛段强制为 0，且分支同时适用于标量与张量
    return tension * (extension > 0)


def cross(left, right):
    """left × right；力臂力矩、ω×r 都用它。"""
    lx, ly, lz = _components(left, "left")
    rx, ry, rz = _components(right, "right")
    return (ly * rz - lz * ry, lz * rx - lx * rz, lx * ry - ly * rx)


def offset_torque(force, offset):
    """把作用在挂点的力折算成相对质心的力矩：offset × force。

    挂点被固定关节合并进刚体后，力必须作用在刚体质心上并补这个力矩，否则会丢掉
    绳力对车体/躯干的转动效应。
    """
    return cross(offset, force)


def point_velocity(linear_velocity, angular_velocity, offset):
    """刚体上某点的速度：v_point = v_com + ω × offset。

    绳力的伸长率 ḋ 用的是**挂点**速度；用质心速度会把 ω×r 那一项漏掉。
    """
    vx, vy, vz = _components(linear_velocity, "linear_velocity")
    wx, wy, wz = cross(angular_velocity, offset)
    return (vx + wx, vy + wy, vz + wz)


def rope_state(robot_point, cart_point, robot_velocity, cart_velocity, *,
               rest_length, stiffness, damping):
    """由两个挂点的位置与**挂点速度**算出张力与一对等大反向的力。

    返回 `RopeState`：`tension`、`direction`（由机器人指向小车）、`force_on_robot`
    与 `force_on_cart`（= -force_on_robot，即牛顿第三定律由构造保证）。

    两个挂点重合（d = 0）时方向无定义，标量入参会直接报错；张量入参请保证挂点不重合，
    并像 `cart_coast.py` 那样在写进仿真缓冲前检查有限性。
    """
    rx, ry, rz = _components(robot_point, "robot_point")
    cx, cy, cz = _components(cart_point, "cart_point")
    vrx, vry, vrz = _components(robot_velocity, "robot_velocity")
    vcx, vcy, vcz = _components(cart_velocity, "cart_velocity")

    delta = (cx - rx, cy - ry, cz - rz)
    distance = _norm(delta)
    if _is_number(distance) and distance <= 0.0:
        raise ValueError("Rope attachment points must not coincide")

    direction = (delta[0] / distance, delta[1] / distance, delta[2] / distance)
    # ḋ = e·(v_L - v_R)：伸长率取两个挂点的相对速度在绳方向上的投影
    rate = (direction[0] * (vcx - vrx)
            + direction[1] * (vcy - vry)
            + direction[2] * (vcz - vrz))
    extension = rope_extension(distance, rest_length=rest_length)
    tension = rope_tension(distance, rate, rest_length=rest_length,
                           stiffness=stiffness, damping=damping)
    force_on_robot = tuple(tension * component for component in direction)
    force_on_cart = tuple(-component for component in force_on_robot)
    return RopeState(distance=distance, extension=extension, distance_rate=rate,
                     tension=tension, direction=direction,
                     force_on_robot=force_on_robot, force_on_cart=force_on_cart)
