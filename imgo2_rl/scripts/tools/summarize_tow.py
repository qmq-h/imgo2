"""Recompute a tow run's summary and verdict using only the Python standard library.

与 P1/P2 的 `summarize_cart_coast.py` 同一模式：入口只负责采样，离线重算负责判读，
于是判据可以在没有仿真器的机器上用**真实轨迹**复算（新判据就是拿实跑数据定的）。

三个阶段（`phase` 列；旧记录没有该列时按指令自动推断，见 `_phases`）：

- `station`：站定，指令 0，绳应当**全程松弛**；
- `tow`：指令 = `v_user`，判「稳定拖曳」；
- `coast`：阶跃把指令归零后的滑行段（计划 P6）：机器人停下、小车继续滑行，
  绳会松弛，甚至可能被追尾后重新绷紧。

判据对应计划 P4/P6：

- `v_R ≈ v_L`：`tow` 阶段稳态窗两体平均 vx 之差 ≤ 0.1 m/s；
- 机器人真的在按指令走：`steady_tracking_ratio ≥ 0.5`（只看「两体同速」有洞：两者都静止时
  它也成立，实测踩过——机器人被拽倒后 `speed_gap` 只有 0.028）；
- 绳真的在拉：`tow` 阶段必须出现过张力，且收掉初始松弛后不再频繁回到松弛；
- `T(t)` 稳定：用 `tow` 阶段稳态窗**前后半的均值漂移**判定，而不是方差。步态会给张力带来
  ~7.5 Hz 的纹波（实测主要由阻尼项 `c·ḋ` 贡献，弹簧项只波动 1.06 N），那是正常现象；
  计划要的是「相对稳定区间」，看的是水平是否稳定。纹波大小仍作为 `tension_ripple_ratio` 报告。
- `station` 阶段绳不得张紧：实测踩过——机器人按 0.35 m 出生（自然站高 0.284 m）会向前窜
  0.65 m/s，把绳多拉长约 0.06 m；初始松弛只有 0.05 m 时绳在站定阶段就被拉直，
  出现 73–86 N 猛拽，把小车甩出 0.24 m 自由滑行（看起来就像「小车自己会动」）。
- `coast` 阶段：报告小车滑行距离、最小间距（追尾风险）、张力归零耗时、是否重新绷紧。
"""

import csv
import json
import math
import sys
from pathlib import Path

TAKEUP_FRACTION = 0.2          # tow 阶段前 20% 视为「收初始松弛 + 起步」过渡
SPEED_GAP_LIMIT_MPS = 0.1
TRACKING_RATIO_MIN = 0.5
TENSION_DRIFT_LIMIT = 0.25
TAIL_SLACK_LIMIT = 0.05
PITCH_LIMIT_RAD = 0.6
SETTLE_TENSION_LIMIT_N = 1.0   # station 阶段张力超过此值 ⇒ 初始松弛量不够
STOP_ROBOT_VX_FRACTION = 0.2   # coast 段末机器人 vx 应降到指令的 20% 以下
SETTLE_ROBOT_TRAVEL_LIMIT_M = 0.15   # station 段机器人位移超过此值 ⇒ 启动窜动过大
SETTLE_LOAD_DRIFT_LIMIT_M = 0.02     # station 段小车位移超过此值 ⇒ 拖曳前它没静止
# ---- 「小车是否追到机器人」的三路见证（阈值由实跑数据定标，见 §5.17 与 §5.18）----
# 为什么不能只看挂点间距：碰撞时的挂点间距在 0.12~0.44 m 之间浮动（由机器人腿的位形决定），
# 实测四个真实撞上的 run 分别是 0.3631/0.3744/0.3649/0.3457 m —— 全都看不出碰撞。
CONTACT_DECK_LIMIT_N = 1.0           # 车斗接触力（车斗永不着地 ⇒ 非零即机器人压上来）
CONTACT_LOAD_DV_LIMIT_MPS = 0.015    # 负载单步速度跃变：无碰撞实测上限 0.0087（滑行起始
                                     # 张力卸载，a=T/m_eff 解析吻合），最小真实撞击 0.0233
CLEARANCE_CONTACT_M = 0.0            # 几何间隙 ≤ 此值 ⇒ 接触（采样误差上界 1.5 mm）
CAUGHT_UP_LIMIT_M = 0.0             # 旧记录（无关节角列）的回退判据：挂点间距 ≤ 此值
PREDICTION_TOLERANCE_M = 0.10       # 实测最小间距与启动预判的允许偏差


def _mean(values):
    return sum(values) / len(values)


def _pstdev(values):
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def _col(rows, key):
    return [float(r[key]) for r in rows]


def _rope_stats(rows, tow_rows, taut_rows, steady_tension, engagement_window_s=0.2):
    """绳索模型的通用离线统计（两套模型同一套字段，见 mdp/rope_model.py）。

    规格明确要求这些量，而它们**都能从记录复算**（不需要重跑）：

    * `rope_impulse_total_ns` = ∫T dt（全程、拖曳段、绷直窗口分别给），
      跨模型可比的不变量：compliant 靠弹簧逐步储能、inextensible 靠约束冲量一步抹掉速度差，
      但**抹掉的动量相同**，所以 ∫T dt 基本一致（实测 23.90 vs 23.93 N·s）；
    * `rope_peak_extension_mm` / `rope_steady_extension_mm`：compliant 是弹性伸长（≈T/k），
      inextensible 只剩数值穿透（mm 级）；
    * `max_tension_rate_n_per_s`：张力变化率峰值（纹波/冲击的强度指标）。
    """
    result = {"rope_impulse_total_ns": None, "rope_impulse_tow_ns": None,
              "rope_impulse_engagement_ns": None, "rope_peak_extension_mm": None,
              "rope_steady_extension_mm": None, "max_tension_rate_n_per_s": None,
              "rope_taut_fraction": None, "rope_dt_s": None}
    if not rows or "rope_impulse_ns" not in rows[0]:
        return result
    times = [float(r["time_s"]) for r in rows]
    steps = [b - a for a, b in zip(times, times[1:]) if b > a]
    dt = sum(steps) / len(steps) if steps else None
    result["rope_dt_s"] = dt
    impulses = [float(r["rope_impulse_ns"]) for r in rows]
    result["rope_impulse_total_ns"] = sum(impulses)
    result["rope_impulse_tow_ns"] = sum(float(r["rope_impulse_ns"]) for r in tow_rows)
    result["rope_peak_extension_mm"] = max(float(r["rope_extension_m"]) for r in rows) * 1000.0
    result["rope_taut_fraction"] = (sum(1 for r in rows if float(r.get("rope_taut", 0.0)) > 0.5)
                                   / len(rows))
    if taut_rows:
        # 绷直窗口：从首次张紧开始的一段固定时长（两套模型用同一口径才可比）
        start = float(taut_rows[0]["time_s"])
        window = [r for r in rows if start <= float(r["time_s"]) <= start + engagement_window_s]
        result["rope_impulse_engagement_ns"] = sum(float(r["rope_impulse_ns"]) for r in window)
    rates = []
    for a, b in zip(tow_rows, tow_rows[1:]):
        span = float(b["time_s"]) - float(a["time_s"])
        if span > 0:
            rates.append(abs(float(b["rope_tension_n"]) - float(a["rope_tension_n"])) / span)
    result["max_tension_rate_n_per_s"] = max(rates) if rates else None
    if steady_tension is not None and dt is not None:
        # 稳态伸长 = 稳态窗口内的平均伸长（compliant 下应 ≈ T_steady/k）
        window = [r for r in tow_rows
                  if float(r["time_s"]) >= float(tow_rows[-1]["time_s"]) - 1.0]
        if window:
            result["rope_steady_extension_mm"] = (
                sum(float(r["rope_extension_m"]) for r in window) / len(window) * 1000.0)
    return result


def _elasticity(tow_rows, taut_rows, steady_tension, config):
    """绳的弹性诊断：伸长、绷直过冲、固有频率/阻尼比，以及显式积分的步长上限。

    用户 2026-09-21 提出「照理说我们的绳子应该不是很弹性」。这些量由**配置常量**
    （`k`/`c`/`L0`、小车质量与轮惯量、机器人质量）与**记录里的张力**决定，所以
    `summarize_tow.py <run>` 就能对**已有 run 复算，不必重跑仿真**（见 docs §5.18）。

    为什么关心：`k=4000 N/m` 比 6 mm 涤纶绳（6×10⁴~1.2×10⁵ N/m）软 15~800 倍，
    稳态伸长只有零点几毫米（几乎不可伸长），但绷直瞬间是欠阻尼弹簧，过冲 6~25 倍。
    而显式弹簧的稳定条件 `dt < 2/(ω(ζ+√(ζ²+1)))` 限制了 k 能提到多大——所以
    `spring_dt_limit_ms` 必须与 `rope_stiffness_n_per_m` 一起读。

    缺 `config`（合成轨迹、旧调用方）时全部为 `None`，并写明缺什么。
    """
    result = {
        "rope_stiffness_n_per_m": None, "rope_damping_ns_per_m": None,
        "rope_stretch_steady_mm": None, "rope_stretch_peak_mm": None,
        "takeup_peak_tension_n": None, "takeup_peak_time_s": None, "takeup_peak_delay_s": None,
        "tension_overshoot_ratio": None, "takeup_closing_speed_mps": None,
        "rope_reduced_mass_kg": None, "rope_natural_freq_hz": None,
        "rope_damping_ratio": None, "spring_dt_limit_ms": None,
        "predicted_takeup_peak_n": None, "elasticity_note": None,
    }
    if not tow_rows:
        result["elasticity_note"] = "没有 tow 段样本"
        return result
    peak_row = max(tow_rows, key=lambda r: float(r["rope_tension_n"]))
    peak = float(peak_row["rope_tension_n"])
    result["takeup_peak_tension_n"] = peak
    result["takeup_peak_time_s"] = float(peak_row["time_s"])
    if taut_rows:
        result["takeup_peak_delay_s"] = (float(peak_row["time_s"]) - float(taut_rows[0]["time_s"]))
        # 绷直瞬间的相对接近速度必须取**前一个松弛步**：绳在上一个 5 ms 步内就绷直了，
        # 「首个 T>0」那行的速度已经被冲量改过。实测 k=1×10⁵ / c=1528 那次，首个 T>0 行的
        # v_R 已从 0.547 掉到 0.176、v_L 已从 0 跳到 0.431 —— 用它会把上界算成 195 N
        # 而实测是 921 N。力是在每步**开始时**按当时状态算的（见 config 的 torque_convention），
        # 所以正确输入就是最后一个松弛步的状态。
        first_taut = tow_rows.index(taut_rows[0])
        reference = tow_rows[first_taut - 1] if first_taut > 0 else taut_rows[0]
        result["takeup_closing_speed_mps"] = (float(reference["robot_vx_mps"])
                                              - float(reference["load_vx_mps"]))
    if steady_tension and steady_tension > 0:
        result["tension_overshoot_ratio"] = peak / steady_tension
    if not config:
        result["elasticity_note"] = "未提供 config（缺 rope/cart_model），只能报峰值张力"
        return result
    rope = config.get("rope") or {}
    stiffness = rope.get("stiffness_n_per_m")
    damping = rope.get("damping_ns_per_m")
    model = config.get("cart_model") or {}
    radius = model.get("wheel_radius_m")
    inertias = (model.get("inertias_kgm2") or {}).get("wheel_fl")
    mass_cart = config.get("cart_mass_actual_kg") or model.get("total_mass_kg")
    missing = [name for name, value in (("rope.stiffness_n_per_m", stiffness),
                                        ("rope.damping_ns_per_m", damping),
                                        ("cart_model.wheel_radius_m", radius),
                                        ("cart_model.inertias_kgm2.wheel_fl", inertias),
                                        ("cart_mass_actual_kg", mass_cart)) if value is None]
    if missing:
        result["elasticity_note"] = f"config 缺 {missing}"
        return result
    result["rope_stiffness_n_per_m"] = stiffness
    result["rope_damping_ns_per_m"] = damping
    if steady_tension:
        # 稳态伸长：绳在稳态张力下的静态伸长 δ = T/k
        result["rope_stretch_steady_mm"] = steady_tension / stiffness * 1000.0
    result["rope_stretch_peak_mm"] = peak / stiffness * 1000.0

    # 折合质量：绳两端的等效惯量。小车侧是 m_eff = m + 4I/r²（车轮滚动惯量折算），
    # 机器人侧取 URDF 总质量（config 没记时用 URDF 求和，见 tow_clearance.robot_mass_kg）
    mass_eff = mass_cart + 4.0 * float(inertias[1]) / radius ** 2
    mass_robot = config.get("robot_mass_kg")
    if mass_robot is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import tow_clearance
        mass_robot = tow_clearance.robot_mass_kg(tow_clearance.repo_root())
    reduced = 1.0 / (1.0 / mass_eff + 1.0 / mass_robot)
    omega = math.sqrt(stiffness / reduced)
    zeta = damping / (2.0 * math.sqrt(stiffness * reduced))
    result["rope_reduced_mass_kg"] = reduced
    result["rope_natural_freq_hz"] = omega / (2.0 * math.pi)
    result["rope_damping_ratio"] = zeta
    result["spring_dt_limit_ms"] = 2.0 / (omega * (zeta + math.sqrt(zeta ** 2 + 1.0))) * 1000.0
    if result["takeup_closing_speed_mps"] is not None:
        # 绷直能量界 ½μv² = ½kδ² ⇒ 峰值 ≈ v√(kμ)（无阻尼上界）
        result["predicted_takeup_peak_n"] = abs(result["takeup_closing_speed_mps"]) * math.sqrt(
            stiffness * reduced)
    return result


def _clearance_series(rows, joint_names, stride=5):
    """由记录关节角 FK 算出每步「小车车头 ↔ 机器人后表面」的纵向间隙。

    几何与 FK 实现在 `tow_clearance.py`（同样只用标准库），这里只做抽样与调用，
    免得判读工具里再长出一套 FK。`stride` 抽稀：间隙是连续量，5 步（25 ms）足够定位最小值。
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import tow_clearance
    robot, cart = tow_clearance.build_bodies(tow_clearance.repo_root())
    series = []
    for index, row in enumerate(rows):
        if index % stride and index != len(rows) - 1:
            continue
        angles, robot_pose, cart_pose = tow_clearance.row_state(row, joint_names)
        gap, _ = tow_clearance.clearance(
            robot.points(angles, *robot_pose),
            cart.points(tow_clearance.WHEEL_JOINT_ANGLES, *cart_pose))
        # 带时刻：判读要用它算「停止指令后多久接触」（训练目标要抢的时间窗）
        series.append((row["phase"], float(row["time_s"]), gap))
    return series


def _phases(rows):
    """按 `phase` 列分段；旧记录没有该列时按指令推断。"""
    if "phase" in rows[0]:
        phases = {name: [r for r in rows if r["phase"] == name]
                  for name in ("station", "tow", "coast")}
        if not phases["tow"]:
            raise ValueError("指令阶段一个样本都没有")
        return phases
    # 旧记录：开头指令 0 = 站定；指令 > 0 = 拖曳；拖曳之后又是 0 = 滑行
    commanded = [i for i, r in enumerate(rows) if float(r["user_cmd_mps"]) > 0.0]
    if not commanded:
        raise ValueError("指令阶段一个样本都没有")
    return {"station": rows[:commanded[0]], "tow": rows[commanded[0]:commanded[-1] + 1],
            "coast": rows[commanded[-1] + 1:]}


def summarize_tow(rows, *, user_command, takeup_fraction=TAKEUP_FRACTION, joint_names=None,
                  config=None):
    """由逐物理步记录算出汇总与判据；`rows` 为 `tow.csv` 的字典列表。"""
    if not rows:
        raise ValueError("拖动记录为空")
    if not math.isfinite(user_command) or user_command <= 0:
        raise ValueError("用户指令必须为有限正数")
    phases = _phases(rows)
    station_rows, tow_rows, coast_rows = phases["station"], phases["tow"], phases["coast"]

    window = tow_rows[len(tow_rows) // 2:]                   # tow 阶段后半段当稳态窗
    takeup = int(len(tow_rows) * takeup_fraction)
    tail = tow_rows[takeup:]
    tension = _col(tow_rows, "rope_tension_n")
    taut_rows = [r for r in tow_rows if float(r["rope_tension_n"]) > 0.0]
    half = len(window) // 2
    mean_first = _mean(_col(window[:half], "rope_tension_n"))
    mean_second = _mean(_col(window[half:], "rope_tension_n"))
    steady_tension = _mean(_col(window, "rope_tension_n"))

    summary = {
        "state": "completed",
        "samples": {"station": len(station_rows), "tow": len(tow_rows), "coast": len(coast_rows)},
        "steady_window_s": [window[0]["time_s"], window[-1]["time_s"]],
        "steady_robot_vx_mps": _mean(_col(window, "robot_vx_mps")),
        "steady_load_vx_mps": _mean(_col(window, "load_vx_mps")),
        "steady_tension_n": steady_tension,
        "steady_tension_std_n": _pstdev(_col(window, "rope_tension_n")),
        "steady_distance_m": _mean(_col(window, "rope_distance_m")),
        "mean_pitch_rad": _mean(_col(window, "body_pitch_rad")),
        "max_abs_pitch_rad": max(abs(float(r["body_pitch_rad"])) for r in rows),
        "tension_peak_n": max(tension),
        "tension_slack_fraction": sum(1 for t in tension if t == 0.0) / len(tension),
        "time_to_taut_s": float(taut_rows[0]["time_s"]) if taut_rows else None,
        "tension_slack_fraction_after_takeup": (
            sum(1 for r in tail if float(r["rope_tension_n"]) == 0.0) / len(tail) if tail else None),
        "final_load_x_m": float(rows[-1]["load_x_m"]),
        "final_robot_x_m": float(rows[-1]["robot_x_m"]),
        "steady_robot_z_m": _mean(_col(window, "robot_z_m")),
        "min_robot_z_m": min(_col(rows, "robot_z_m")),
        # station 阶段（指令 0、绳本应松弛）：出生窜动与「绳是否被拉直」都要报出来
        "settle_max_tension_n": (max(_col(station_rows, "rope_tension_n")) if station_rows else None),
        "settle_max_abs_robot_vx_mps": (max(abs(v) for v in _col(station_rows, "robot_vx_mps"))
                                        if station_rows else None),
        "settle_load_drift_m": (float(station_rows[-1]["load_x_m"]) - float(station_rows[0]["load_x_m"])
                                if station_rows else None),
        # station 段机器人自己走了多少：出生窜动超过松弛量就会把绳拉直（实测 0.46 m 时
        # 直接把机器人拽翻），所以这是选出生高度/松弛量的关键读数
        "settle_robot_travel_m": (float(station_rows[-1]["robot_x_m"]) - float(station_rows[0]["robot_x_m"])
                                  if station_rows else None),
        "settle_max_abs_load_vx_mps": (max(abs(v) for v in _col(station_rows, "load_vx_mps"))
                                       if station_rows else None),
    }
    # ---- 「会不会追到机器人」与「最终停在离机器人多远」 ------------------------------
    # 注意：station 段的间距（= L0 - slack）本来就很近，不能算作追尾，
    # 所以最小间距从**拖曳段开始**起算。
    #
    # 这里必须把「挂点间距」和「车头到机器人的间隙」区分开：`rope_distance_m` 是两个**挂点**
    # 的距离，碰撞时它在 0.12~0.44 m 之间浮动（取决于机器人腿伸到多后），所以它既不能判
    # 「有没有追上」，也不能当「停车距离」。判定用三路见证，并把每个见证量都报出来：
    # 判定用三路见证，并把每个见证量都报出来：
    #   ① 接触力（车斗永不着地 ⇒ 非零即机器人压上来；车轮那路的 x 分量**前后轮符号相反、
    #      稳态下互相抵消**（实测稳态和值 1.3e-5 N），所以不能拿它当「滚动基线」，
    #      反过来任何明显的非零和值都是异常——撞击只打一个轮子，不会成对抵消）；
    #   ② 负载单步速度跃变（黏性滑行的单步变化只有 1e-3 量级，撞击是 1e-2~1e-1）；
    #   ③ 由记录关节角 FK 算出的真实几何间隙 `min_clearance_m`（有该列时最可靠）。
    after_tow = tow_rows + coast_rows
    gaps = _col(after_tow, "rope_distance_m")
    summary["initial_gap_m"] = float(rows[0]["rope_distance_m"])
    summary["min_gap_after_tow_start_m"] = min(gaps)
    summary["final_gap_m"] = float(rows[-1]["rope_distance_m"])
    summary["gap_closed_m"] = summary["initial_gap_m"] - summary["final_gap_m"]

    witnesses = []
    coast_pairs = list(zip(coast_rows, coast_rows[1:])) if len(coast_rows) > 1 else []
    if coast_rows:
        summary["max_load_dv_mps"] = max(
            (abs(float(b["load_vx_mps"]) - float(a["load_vx_mps"])) for a, b in coast_pairs),
            default=0.0)
        # 机器人被撞的冲量特征：单步速度跃变。无碰撞时它是步态引起的 0.02~0.03，
        # 所以**不**用它判接触，只报告（撞击实测 0.037~0.092）
        summary["max_robot_dv_mps"] = max(
            (abs(float(b["robot_vx_mps"]) - float(a["robot_vx_mps"])) for a, b in coast_pairs),
            default=0.0)
    else:
        summary["max_load_dv_mps"] = None
        summary["max_robot_dv_mps"] = None
    if summary["max_load_dv_mps"] is not None and summary["max_load_dv_mps"] > CONTACT_LOAD_DV_LIMIT_MPS:
        witnesses.append("load_velocity_jump")

    if "cart_deck_fx_n" in (after_tow[0] if after_tow else {}):
        summary["deck_contact_peak_n"] = max(abs(float(r["cart_deck_fx_n"])) for r in after_tow)
        summary["wheel_fx_peak_n"] = max(abs(float(r["cart_wheel_fx_n"])) for r in after_tow)
        # station 段（拖曳之前）的接触单独报出来：出生时挂点间距只有 0.40 m，而机器人在
        # 下蹲姿态下后腿能伸到 base 后方 0.41 m —— 实测最小车头间隙只有 **0.0955 m**
        # （见 §5.19），所以「生成时就贴上」是需要单独盯的一种坏情况。
        summary["deck_contact_peak_station_n"] = (max(abs(float(r["cart_deck_fx_n"]))
                                                      for r in station_rows) if station_rows else None)
        if summary["deck_contact_peak_n"] > CONTACT_DECK_LIMIT_N:
            witnesses.append("deck_contact_force")
    else:
        summary["deck_contact_peak_n"] = None
        summary["wheel_fx_peak_n"] = None
        summary["deck_contact_peak_station_n"] = None

    if joint_names and rows and "robot_jp_00" in rows[0]:
        clearance = _clearance_series(rows, joint_names)
        after_tow_clearance = [gap for phase, _, gap in clearance if phase != "station"]
        coast_samples = [(time_s, gap) for phase, time_s, gap in clearance if phase == "coast"]
        station_clearance = [gap for phase, _, gap in clearance if phase == "station"]
        summary["min_clearance_m"] = min(after_tow_clearance)
        summary["min_clearance_coast_m"] = min(gap for _, gap in coast_samples) if coast_samples else None
        summary["min_clearance_station_m"] = min(station_clearance) if station_clearance else None
        summary["final_clearance_m"] = clearance[-1][2]
        # ---- 停止瞬态：训练目标（收到停止指令后机器人往前几步防追尾）要抢的时间窗 ----
        # `clearance_at_stop_m` = 指令归零那一刻的车头间隙；
        # `time_to_contact_after_stop_s` = 从那一刻到首次接触（间隙 ≤ 0）的时间 ⇒ 机器人
        #     「往前几步」必须在这么长时间内把间隙重新拉开；
        # `load_vx_at_contact_mps` = 接触瞬间负载的速度（撞击速度的实测值）。
        if coast_samples:
            stop_time, stop_gap = coast_samples[0]
            summary["clearance_at_stop_m"] = stop_gap
            contact = next(((time_s, gap) for time_s, gap in coast_samples
                            if gap <= CLEARANCE_CONTACT_M), None)
            summary["time_to_contact_after_stop_s"] = (contact[0] - stop_time) if contact else None
            if contact is not None:
                nearest = min(coast_rows, key=lambda r: abs(float(r["time_s"]) - contact[0]))
                summary["load_vx_at_contact_mps"] = float(nearest["load_vx_mps"])
            else:
                summary["load_vx_at_contact_mps"] = None
        else:
            summary["clearance_at_stop_m"] = None
            summary["time_to_contact_after_stop_s"] = None
            summary["load_vx_at_contact_mps"] = None
        summary["clearance_samples"] = len(clearance)
        if summary["min_clearance_m"] <= CLEARANCE_CONTACT_M:
            witnesses.append("geometric_clearance")
    else:
        summary["min_clearance_m"] = None
        summary["min_clearance_coast_m"] = None
        summary["min_clearance_station_m"] = None
        summary["final_clearance_m"] = None
        summary["clearance_at_stop_m"] = None
        summary["time_to_contact_after_stop_s"] = None
        summary["load_vx_at_contact_mps"] = None

    # 三路见证的可用性与结论。**注意别把「没有见证通道」当成「没追上」**：
    # 旧记录既没有接触力列也没有关节角列，只能退回挂点间距口径，并把这个不可靠性写进来源名。
    gap_fallback_reached = summary["min_gap_after_tow_start_m"] <= CAUGHT_UP_LIMIT_M
    have_witness_channel = (summary["min_clearance_m"] is not None
                            or summary["deck_contact_peak_n"] is not None)
    if witnesses:
        summary["reached_robot"] = True
        summary["reached_robot_source"] = "+".join(witnesses)
    elif have_witness_channel:
        summary["reached_robot"] = False
        summary["reached_robot_source"] = "witnesses_clear"
    else:
        summary["reached_robot"] = gap_fallback_reached
        summary["reached_robot_source"] = ("attachment_gap_fallback" if gap_fallback_reached
                                           else "attachment_gap_fallback_clear")
    # `caught_up` 保留为旧名（挂点间距口径），新代码请用 `reached_robot`
    summary["caught_up"] = summary["min_gap_after_tow_start_m"] <= CAUGHT_UP_LIMIT_M

    summary["steady_speed_gap_mps"] = summary["steady_robot_vx_mps"] - summary["steady_load_vx_mps"]
    summary["steady_tracking_ratio"] = summary["steady_robot_vx_mps"] / user_command
    summary["tension_drift_ratio"] = (abs(mean_second - mean_first) / steady_tension
                                     if steady_tension > 0 else None)
    summary["tension_ripple_ratio"] = (summary["steady_tension_std_n"] / steady_tension
                                      if steady_tension > 0 else None)
    summary.update(_elasticity(tow_rows, taut_rows, steady_tension, config))
    summary.update(_rope_stats(rows, tow_rows, taut_rows, steady_tension))

    # 收松弛的耗时与机器人走过多远才发力：设计上应当 ≈ `--slack`（可直接核对）
    if taut_rows:
        summary["takeup_time_s"] = float(taut_rows[0]["time_s"]) - float(tow_rows[0]["time_s"])
        summary["takeup_robot_travel_m"] = (float(taut_rows[0]["robot_x_m"])
                                            - float(tow_rows[0]["robot_x_m"]))
    else:
        summary["takeup_time_s"] = None
        summary["takeup_robot_travel_m"] = None

    if coast_rows:
        coast_tension = _col(coast_rows, "rope_tension_n")
        gaps = _col(coast_rows, "rope_distance_m")
        slack_at = next((i for i, t in enumerate(coast_tension) if t == 0.0), None)
        re_taut = ([t for t in coast_tension[slack_at + 1:] if t > 0.0]
                   if slack_at is not None else [])
        summary.update({
            "coast_duration_s": float(coast_rows[-1]["time_s"]) - float(coast_rows[0]["time_s"]),
            "coast_load_travel_m": float(coast_rows[-1]["load_x_m"]) - float(coast_rows[0]["load_x_m"]),
            "coast_robot_travel_m": float(coast_rows[-1]["robot_x_m"]) - float(coast_rows[0]["robot_x_m"]),
            "coast_gap_at_stop_m": gaps[0],
            "coast_min_gap_m": min(gaps),
            "coast_peak_tension_n": max(coast_tension),
            "coast_time_to_slack_s": (float(coast_rows[slack_at]["time_s"])
                                      - float(coast_rows[0]["time_s"])) if slack_at is not None else None,
            "coast_retension_peak_n": max(re_taut) if re_taut else 0.0,
            "coast_final_robot_vx_mps": float(coast_rows[-1]["robot_vx_mps"]),
            "coast_final_load_vx_mps": float(coast_rows[-1]["load_vx_mps"]),
        })

    failures = []
    # ---- station：绳不得被拉直（初始松弛量要够），且机器人不应自己窜出去
    if summary["settle_max_tension_n"] is not None and \
            summary["settle_max_tension_n"] > SETTLE_TENSION_LIMIT_N:
        failures.append("rope_taut_during_settle")
    if summary["settle_robot_travel_m"] is not None and \
            abs(summary["settle_robot_travel_m"]) > SETTLE_ROBOT_TRAVEL_LIMIT_M:
        failures.append("robot_lurches_during_settle")
    # 拖曳开始时小车必须已经静止在设计位置（不再用代码把它「摆正」，改为事后检查）
    if summary["settle_load_drift_m"] is not None and \
            abs(summary["settle_load_drift_m"]) > SETTLE_LOAD_DRIFT_LIMIT_M:
        failures.append("load_moved_during_settle")
    # 生成时就贴上机器人：station 段车斗接触力非零。挂点间距 0.40 m 听着安全，但实测
    # 车头间隙只有 0.0955 m（§5.19），所以这一档要单独判失败，而不是留到「追尾」里去
    if summary["deck_contact_peak_station_n"] is not None and \
            summary["deck_contact_peak_station_n"] > CONTACT_DECK_LIMIT_N:
        failures.append("load_touching_at_settle")
    # ---- tow
    if abs(summary["steady_speed_gap_mps"]) > SPEED_GAP_LIMIT_MPS:
        failures.append("robot_and_load_speeds_differ")
    if summary["steady_tracking_ratio"] < TRACKING_RATIO_MIN:
        failures.append("robot_not_tracking_command")
    if summary["time_to_taut_s"] is None:
        failures.append("rope_never_taut_during_tow")
    elif summary["tension_slack_fraction_after_takeup"] > TAIL_SLACK_LIMIT:
        failures.append("rope_not_continuously_taut")
    if steady_tension <= 0.0:
        failures.append("no_steady_tension")
    elif summary["tension_drift_ratio"] > TENSION_DRIFT_LIMIT:
        failures.append("tension_not_steady")
    if summary["max_abs_pitch_rad"] > PITCH_LIMIT_RAD:
        failures.append("body_pitch_excessive")
    # ---- coast：机器人要真的停下；小车不得到顶到机器人（追尾）
    if coast_rows and summary["coast_final_robot_vx_mps"] > STOP_ROBOT_VX_FRACTION * user_command:
        failures.append("robot_did_not_stop")
    # **不再是失败判据**：撞击是这套设置本来就允许、也希望出现的结果（用户 2026-09-21 明确），
    # 所以它只作为结果量 `reached_robot` 报告，不参与 `valid`。撞击把机器人弄坏仍然会被拦住：
    # `robot_did_not_stop`（被推着走）、`body_pitch_excessive`（被顶翻）、
    # `robot_not_tracking_command`（被拖停）三条判据都在。
    summary["failures"] = failures
    summary["valid"] = not failures
    return summary


def summarize_run(directory):
    """读一个运行目录里的 `tow.csv` 与 `config.json` 重算汇总。"""
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    with (directory / "tow.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    summary = summarize_tow(rows, user_command=float(config["user_command_mps"]),
                            joint_names=config.get("policy_joint_names"), config=config)
    # 与启动时的解析预判对照（预判只考虑滑行段的黏性衰减；实测还会受机器人减速影响）
    predicted = config.get("predicted_min_gap_m")
    if predicted is not None:
        summary["predicted_min_gap_m"] = predicted
        summary["prediction_error_m"] = summary["min_gap_after_tow_start_m"] - predicted
        summary["prediction_ok"] = abs(summary["prediction_error_m"]) <= PREDICTION_TOLERANCE_M
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = summarize_run(args.run_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["valid"] else 1)
