"""Cart mass knob: the ballast cube is the only adjustable mass (stdlib only).

2026-10-17 用户口径：车斗 + 四轮的质量固定，**通过调整车斗中间的重物块质量**来得到所需总重。
重物块与 base_link 在导入时被 `merge_fixed_joints=True` 合并成**一个刚体**，所以运行时改的是
base_link 的质量、绕 CoM 的惯量与 CoM 高度——三者必须一起改，只改质量会让动力学不自洽
（AGENTS.md 的既有纪律）。

解析式（车斗 CoM 在 base_link 原点、重物块 CoM 在 +h）：

    m_base = m_deck + m_ballast
    z_com  = m_ballast·h / (m_deck + m_ballast)
    I_xx   = I_deck,xx + m_deck·z_com² + I_ballast,xx + m_ballast·(h − z_com)²
    I_yy   = 同上（y 方向）
    I_zz   = I_deck,zz + I_ballast,zz          # 偏移沿 z，不影响 zz

`I_ballast` 按质量线性缩放（立方体：`m·(b²+b²)/12`）；车斗惯量与轮子质量不随总重变化。
"""

from __future__ import annotations


def body_properties(model: dict, total_mass_kg: float) -> dict:
    """给定**整车总重**，返回合并后 base_link 的质量 / CoM / 惯量（关于 CoM）与轮子质量。

    `model` 是 `cart_model.read_cart_model()` 的返回；总重必须大于「车斗 + 四轮」。
    """
    deck = float(model["deck_mass_kg"])
    wheel = float(model["wheel_mass_kg"])
    count = len(model["wheel_names"])
    fixed = deck + count * wheel
    if not total_mass_kg > fixed:
        raise ValueError(
            f"整车总重 {total_mass_kg:g} kg 必须大于车斗 + {count} 轮 = {fixed:g} kg")
    ballast = total_mass_kg - fixed
    height = float(model["ballast_center_m"][2])
    base_mass = deck + ballast
    com_z = ballast * height / base_mass
    nominal_ballast = float(model["ballast_mass_kg"])
    scale = ballast / nominal_ballast
    ballast_inertia = tuple(float(v) * scale for v in model["ballast_inertia_kgm2"])
    deck_inertia = tuple(float(v) for v in model["base_inertia_kgm2"])
    shift_deck = deck * com_z * com_z
    shift_ballast = ballast * (height - com_z) ** 2
    inertia = (deck_inertia[0] + shift_deck + ballast_inertia[0] + shift_ballast,
               deck_inertia[1] + shift_deck + ballast_inertia[1] + shift_ballast,
               deck_inertia[2] + ballast_inertia[2])
    return {
        "total_mass_kg": float(total_mass_kg),
        "ballast_mass_kg": ballast,
        "base_mass_kg": base_mass,
        "base_com_m": (0.0, 0.0, com_z),
        "base_inertia_kgm2": inertia,
        "wheel_mass_kg": wheel,
        "fixed_mass_kg": fixed,
    }
