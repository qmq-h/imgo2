# 小车改版（2026-10-17）：低车斗 + 6 cm 小轮 + 重物块质量旋钮

用户口径（逐条）：

> 车斗改成 0.6m*0.45m*2cm，车斗在车轮上方高度 9cm，整体重量为 3kg，轮子间距对应调宽，
> 添加一个重物块在小车中间 fix 固定，通过调整这个重物块质量来调整需要的重量。
> 重物块设置成 0.3*0.3*0.3
>
> 轮子在车斗下方内侧，放置不超出车斗，设置为 6cm 直径，类似推车全向轮的形式。

澄清后确定的口径：**「9 cm」= 车斗底面离地高度**（不是相对轮顶/轮轴）；轮子缩到 **6 cm 直径**
才能塞进 9 cm 净空；轮子**完全在车斗下方内侧**、不超出 0.45 m 宽的投影。

## 1. 最终几何

| 项 | 旧 | 新 |
|---|---|---|
| 车斗（base_link） | 0.50 × 0.32 × 0.10 m，8.4 kg | **0.60 × 0.45 × 0.02 m，0.8 kg** |
| 车斗底面离地 | —（中心 0.15 m） | **0.09 m**（中心 0.10 m） |
| 车轮 | 4 × r=0.08、宽 0.04、各 0.4 kg | **4 × r=0.03（Ø6 cm）、宽 0.04、各 0.05 kg** |
| 轮距 / 轴距 | 0.40 / 0.36 m | **0.40 / 0.52 m**（x=±0.26、y=±0.20，全在 0.6×0.45 内） |
| 重物块（ballast） | 无 | **0.30³ m，fix 在车斗正中、坐在上表面**（中心 z=+0.16） |
| 名义总重 | 10.0 kg | **3.0 kg** = 车斗 0.8 + 4×0.05 + 重物块 **2.0** |
| 挂点 / 前表面 | 0.25 m | **0.30 m**（车斗前缘） |
| 静置高度 | 0.15 m | **0.10 m** |
| 轮轴阻尼 | 0.032 N·m·s/rad | **0.0045**（按 r² 重标定，保持等效滚阻 ≈ 20·v N） |

## 2. 质量旋钮（用户要求「通过重物块调重量」）

`imgo2_rl/source/imgo2_rl/imgo2_rl/assets/cart_mass.py::body_properties(model, total)`（纯标准库）：

    m_base = m_deck + m_ballast            # 重物块与 base_link 在导入时被合并
    z_com  = m_ballast·h / (m_deck + m_ballast)
    I_xx/yy = I_deck + m_deck·z_com² + I_ballast + m_ballast·(h − z_com)²
    I_zz    = I_deck + I_ballast
    m_ballast = total − m_deck − 4·m_wheel

- 训练侧 `upper_mdp.reset_towing_episode`：只写 base_link 的 mass/inertia/**CoM**（`set_coms`），
  轮子质量不动；
- 测试台 `play_towing_test.cart_mass_tensors` 用**同一个**函数；
- MuJoCo 侧 `mjcf/cart.xml` 保独立 `cart_ballast` body，C++ `ConfigureCartMass()` 只改它
  （`body_mass` + `0.015·m` 的惯量），车斗/轮子不再按比例缩放。

## 3. 连带改动

- `cart_model.py`：支持 7 link / 6 joint，解析 ballast；「轮子不得与车斗相交」改成**竖直间距**
  判据（轮顶 ≤ 车斗底面）＋ 车轮必须落在车斗投影内；
- `upper_mdp`：`initial_cart_mass 10 → 3.0`、`wheel_radius 0.08 → 0.03`、
  `initial_wheel_damping 0.032 → 0.0045`、`cart_front_surface_x 0.25 → 0.30`、
  `cart_attachment 0.25 → 0.30`；
- `upper_env_cfg`：`cart.init_state.pos` z 0.18 → **0.13**（静置 0.10 + drop 0.03）、
  `wheel_damping_range (0.008, 0.032) → (0.001125, 0.0045)`（同样按 r² 重标定）；
- `tow_clearance.py`：车斗前缘 0.30、静置 0.10；自检哨兵 0.5616 → **0.5937 m**（面面对点对随高度变化）；
- 奖励文档 `towing_reward_retune_2026-10-10.md` §2.2 两张阈值表按新 Δz=0.22 重算
  （绳 L=0.5 → 0.3166 m，L=1.5 → 1.1622 m；杆 0.4315 / 0.9580 m）。

## 4. 验证

- `pytest imgo2_rl/tests -q` ⇒ **672 passed, 4 skipped, 1336 subtests passed**；
- `python imgo2_rl/scripts/tools/check_towing_mjcf.py` ⇒ OK（名义 3.0 kg、阻尼 0.0045、ballast body）；
- `python3 imgo2_rl/scripts/towing/play_towing_test.py --dry-run` ⇒ 正常（打印轮轴阻尼 0.0045）；
- `tow_clearance.selftest()` ⇒ OK（0.5937 m 哨兵）。

## 5. 未验证 / 待确认

1. **本机没有 CUDA**，Isaac Sim 与 MuJoCo 都没实跑：物理行为（小轮 + 低车斗 + 高 CoM 重物块）
   待训练机首跑确认；建议先跑一轮 `--num-envs 20` 冒烟看有无穿透/翻车。
2. **重物块让 CoM 变高**（3 kg 名义 CoM 离地 0.114 + 车斗 0.10 ⇒ 全车 CoM ≈ 0.21 m；20 kg 时
   0.153 + 0.10 ≈ 0.25 m）⇒ 侧翻/急停点头风险上升，需实测。
3. **真全向轮没做**：几何上按推车轮外形（小轮、在车斗下），物理上仍是绕 y 轴的圆柱滚轮；
   要实现横向自由滚动需要滚子几何或各向异性摩擦，本轮未做（拖曳以纵向为主，影响有限）。
4. `rl_sim_mujoco.cpp` 的改动**未编译**（本机无 ROS/colcon），标记「已修，待验证」。
