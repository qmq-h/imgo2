"""步态指标的纯 numpy 实现（不依赖 Isaac Lab / torch）。

之所以从 `eval_gait.py` 里抽出来：回放本身需要 GPU 与 Isaac Lab，但**口径**必须能在
任何机器上回归。这里只做数学，`eval_gait.py` 只负责采样与打印，
`tests/test_gait_metrics.py` 用合成信号锁定下面几条容易写错的地方：

  * 「逐环境占空比」是对**时间轴**求均值（`c[e].mean()`），不是对同一时刻的各环境求均值
    （`c[:, e].mean()`）。后者会得到 k/E 这种离散值，看起来像「有些环境整段不落地」——
    这正是 2026-09-18 那次回放 JSON 里 `contact_air_fraction_per_env` 出现 0.0/1.0 的原因。
  * 相位用**全部落地事件做圆周平均**，并给出集中度 R；只取「第一个落地事件、只取 0 号环境」
    会被单次抖动带偏（那次回放给出 FL-FR 相位 0.80 周期 ≈ 288°，而 Gazebo 侧同一份
    checkpoint 在 1.0 m/s 测到的是 176.9°）。

指标口径（与 `docs/gait_reference_baseline.json` 对照使用）：
  * duty factor — 支撑相时间占比；air fraction = 1 − duty factor
  * stride freq — 每秒落地次数（由 0→1 上升沿计数）；step period = 相邻落地间隔均值
  * phase       — 相对参考足（默认 FL）落地栅格的相位，0~1 周期；R 为圆周集中度
"""

from __future__ import annotations

import numpy as np


def landing_events(contact_env):
    """[T] 的 0/1 接触序列 → 落地时刻的下标（0→1 上升沿所在步）。"""
    c = np.asarray(contact_env).astype(np.int8)
    if c.size < 2:
        return np.empty(0, dtype=int)
    return np.flatnonzero(np.diff(c) > 0) + 1


def run_lengths(contact_env, dt):
    """[T] 的 0/1 序列 → (空中段时长列表, 支撑段时长列表)，单位秒。

    游程切分覆盖 [0,T) 全区间，因此 sum(air)+sum(stance) 恒等于 T*dt。
    """
    c = np.asarray(contact_env).astype(np.int8)
    if c.size == 0:
        return [], []
    bounds = np.concatenate(([0], np.flatnonzero(np.diff(c)) + 1, [c.size]))
    air, stance = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        (stance if c[a] else air).append(float(b - a) * dt)
    return air, stance


def circular_mean(angles):
    """弧度数组 → (相位 in [0,1), 集中度 R in [0,1])；空数组返回 (None, None)。"""
    a = np.asarray(angles, dtype=np.float64).ravel()
    if a.size == 0:
        return None, None
    c_mean, s_mean = float(np.cos(a).mean()), float(np.sin(a).mean())
    return float((np.arctan2(s_mean, c_mean) / (2.0 * np.pi)) % 1.0), float(np.hypot(c_mean, s_mean))


def interpolated_phase_angles(events, ref_events):
    """把事件相位**插值到相邻两个参考落地之间**，返回弧度数组。

    为什么不用 `(t - ref[0]) % period`：真实步周期几乎不可能正好是整数个仿真步
    （本例 0.193 s ÷ 0.02 s = 9.6 步），取模会把每周期 0.4 步的误差累积到上百个周期之后，
    圆周平均被抹平 —— 2026-09-18 那次实测集中度只剩 **0.17**，而 Gazebo 侧同一 checkpoint
    的相位是稳定的 ±180°。插值法对周期漂移免疫。

    落在参考落地区间之外的事件被丢弃（首尾各一小段）。
    """
    ev = np.asarray(events, dtype=np.float64).ravel()
    ref = np.asarray(ref_events, dtype=np.float64).ravel()
    if ev.size == 0 or ref.size < 2:
        return np.empty(0, dtype=np.float64)
    j = np.searchsorted(ref, ev, side="right") - 1      # ref[j] <= t < ref[j+1]
    ok = (j >= 0) & (j + 1 < ref.size)
    if not ok.any():
        return np.empty(0, dtype=np.float64)
    ev, j = ev[ok], j[ok]
    span = ref[j + 1] - ref[j]
    good = span > 0
    return 2.0 * np.pi * ((ev[good] - ref[j[good]]) / span[good])


def circular_phase(events, ref_events, period):
    """把 events 相对「固定周期参考栅格」的相位做圆周平均（`period` 与 events 同单位）。

    返回 (phase in [0,1), 集中度 R in [0,1])；样本不足返回 (None, None)。
    R 越接近 1 说明落地越规整（足端版的「周期强度」，与 Gazebo 侧用关节信号自相关算的
    周期强度不是同一个量，不要混用数值）。**周期会漂移时请用 `interpolated_phase_angles`。**
    """
    ev = np.asarray(events, dtype=np.float64).ravel()
    ref = np.asarray(ref_events, dtype=np.float64).ravel()
    if ev.size == 0 or ref.size < 2 or not np.isfinite(period) or period <= 0:
        return None, None
    ang = 2.0 * np.pi * (((ev - ref[0]) % period) / period)
    return circular_mean(ang)


def peak_to_peak_per_env(values):
    """[T,E]（或 [T,E,...] 展平后的 [T,E]）→ 逐环境峰峰值。

    返回 (均值, 最大值, 逐环境列表)。只报均值会掩盖「部分环境不动/被重置污染」，
    所以三个都给。
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 2:
        v = v.reshape(v.shape[0], -1)
    ptp = np.ptp(v, axis=0) if v.shape[0] else np.zeros(v.shape[1])
    return float(np.mean(ptp)), float(np.max(ptp)), [float(x) for x in ptp]


def per_env_mean(values):
    """[T,E] → 逐环境时间均值列表。"""
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 2:
        v = v.reshape(v.shape[0], -1)
    return [float(x) for x in np.mean(v, axis=0)] if v.shape[0] else [float("nan")] * v.shape[1]


def summarize_contact(contact, dt, foot_names, ref_index=0):
    """[E,T,F] 的 0/1 接触序列 → 逐足指标字典。

    逐足指标里同时给出「逐环境明细」与「全样本时间加权值」，两者必须自洽：
    mean(contact_air_fraction_per_env) == air_fraction（同一份数据，只是聚合顺序不同）。
    """
    c = np.asarray(contact).astype(np.int8)
    if c.ndim != 3:
        raise ValueError(f"contact 必须是 [E,T,F]，收到 {c.shape}")
    E, T, F = c.shape
    duration = T * dt

    # 参考足的落地栅格：周期先逐环境求，再对有效环境取均值
    ref_periods = []
    ref_landings = {}
    for e in range(E):
        ev = landing_events(c[e, :, ref_index])
        ref_landings[e] = ev
        if ev.size >= 2:
            ref_periods.append(float(np.mean(np.diff(ev))) * dt)
    period_s = float(np.mean(ref_periods)) if ref_periods else float("nan")

    per_foot = {}
    for f, name in enumerate(foot_names):
        air_runs, stance_runs, landings_per_env, air_frac_per_env = [], [], [], []
        angles = []
        for e in range(E):
            seq = c[e, :, f]
            a, s = run_lengths(seq, dt)
            air_runs += a
            stance_runs += s
            air_frac_per_env.append(float(1.0 - seq.mean()))   # 对时间轴求均值
            ev = landing_events(seq)
            landings_per_env.append(int(ev.size))
            # 相位必须以**该环境自己的**参考足落地栅格为零点；参考足在该环境里落地不足 2 次
            # 就没有周期可言，跳过（有测试专门喂这种「参考足整段不落地」的输入）。
            # 用插值相位（对周期漂移免疫），不用取模。
            ref_ev = ref_landings.get(e, np.empty(0, dtype=int))
            if ev.size and ref_ev.size >= 2:
                ang = interpolated_phase_angles(ev, ref_ev)
                if ang.size:
                    angles.append(ang)
        phase, concentration = (None, None)
        if angles:
            phase, concentration = circular_mean(np.concatenate(angles))

        duty = float(c[:, :, f].mean())
        per_foot[name] = {
            "duty_factor": duty,
            "air_fraction": float(1.0 - duty),
            "duty_factor_per_env": [float(1.0 - x) for x in air_frac_per_env],
            "contact_air_fraction_per_env": air_frac_per_env,
            "landings_per_env": landings_per_env,
            "stride_freq_hz": float(np.mean(landings_per_env) / duration) if duration > 0 else float("nan"),
            "step_period_s": float(duration / np.mean(landings_per_env)) if np.mean(landings_per_env) > 0 else float("nan"),
            "air_time_mean_s": float(np.mean(air_runs)) if air_runs else 0.0,
            "stance_time_mean_s": float(np.mean(stance_runs)) if stance_runs else 0.0,
            "phase_offset_vs_FL": phase if f != ref_index else 0.0,
            "phase_deg_vs_FL": (phase * 360.0) if (phase is not None and f != ref_index) else 0.0,
            "phase_concentration": concentration if f != ref_index else None,
        }

    return per_foot, {
        "gait_period_s": period_s,
        "gait_period_n_envs": len(ref_periods),
    }


def body_attitude_summary(quat_wxyz, lean_threshold_deg=2.0, dt=None):
    """[T,E,4] wxyz → 机身姿态统计（单位度）。

    参考基线（`docs/gait_reference_baseline.json`，21 份录制动作）：pitch 均值都在 ±1.6° 内、
    前进时略低头 ~0.5°、后退时略抬头 ~1°，roll 均值 ≈0、RMS 0.33°。明显偏离即姿态问题。

    给了 `dt` 就额外给出 yaw 的**累积漂移率**（首末差/时长，度/秒）——只报峰峰值分不清
    「单调漂移」与「来回摆」。
    """
    q = np.asarray(quat_wxyz, dtype=np.float64)
    if q.ndim != 3 or q.shape[-1] != 4:
        raise ValueError(f"quat 必须是 [T,E,4]，收到 {q.shape}")
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll = np.degrees(np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
    pitch = np.degrees(np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    # 偏航必须**先解缠（unwrap）再统计**：arctan2 给出的 yaw 落在 ±180°，机器人航向一旦跨过
    # ±180° 边界，「峰峰 = max − min」和「漂移率 = (末 − 首)/时长」都会凭空多（或少）360°。
    # 2026-09-20 实测到一次：`yaw 峰峰 359.95°` 而累积漂移率只有 −2.32 °/s（19 s 才 −44°）
    # —— 那个 360° 纯粹是解缠假象（旧 24500 那次没跨界，所以 24° 是对的）。
    yaw = np.degrees(np.unwrap(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)), axis=0))

    per_env_pitch = np.mean(pitch, axis=0)
    out = {
        "body_roll_deg_mean": float(np.mean(roll)),
        "body_roll_deg_rms": float(np.std(roll)),
        "body_pitch_deg_mean": float(np.mean(pitch)),
        "body_pitch_deg_rms": float(np.std(pitch)),
        "body_pitch_deg_ptp": float(np.max(pitch) - np.min(pitch)),
        # 解缠后的航向范围（跨 ±180° 不再产生 360° 假象）
        "body_yaw_drift_deg": float(np.max(yaw) - np.min(yaw)),
        "body_pitch_deg_mean_per_env": [float(v) for v in per_env_pitch],
        # 「恒定倾斜」：各环境的时间均值彼此接近且远离 0 —— 与「前后摆动」区分开
        "body_pitch_is_constant_lean": bool(
            abs(float(np.mean(per_env_pitch))) > lean_threshold_deg
            and abs(float(np.mean(per_env_pitch))) > 2.0 * float(np.std(per_env_pitch))
        ),
    }
    if dt is not None and yaw.shape[0] >= 2:
        span = (yaw.shape[0] - 1) * float(dt)
        rate = (yaw[-1] - yaw[0]) / span if span > 0 else np.zeros(yaw.shape[1])
        out["body_yaw_drift_rate_deg_s"] = float(np.mean(rate))
        out["body_yaw_drift_rate_deg_s_per_env"] = [float(v) for v in rate]
    return out


def _expect(name, arr, shape):
    """严格校验布局。**本模块所有采样输入的时间轴都在 0**，写错轴必须立刻报错。

    2026-09-18 的教训：`eval_gait.py` 里一处 `np.stack(..., axis=1)` 把 joint_pos 变成
    [E,T,J]，而 build_report 按 [T,E,J] 取 `[:, :, i]`，于是关节峰峰值算成了「同一时刻跨环境
    的散布」≈ 0，12 个关节全打印成 0.000 —— 而其余指标看起来完全正常，极易漏过。
    """
    a = np.asarray(arr)
    if a.shape == shape:
        return a
    hint = ""
    if a.ndim == len(shape) and a.shape[::-1][:2] == shape[:2]:
        hint = "（形状像是把时间轴与环境轴写反了：本模块一律要求时间在轴 0）"
    raise ValueError(f"build_report: {name} 形状应为 {tuple(shape)}，收到 {tuple(a.shape)}{hint}")


def build_report(contact, base_z, vel_err_xy, vel_err_yaw, quat, foot_pos_b, joint_pos,
                 ep_len, done, dt, foot_names, checkpoint="", iteration=-1,
                 warmup=0, t_series=None, dump_timeseries=False, meta=None):
    """把回放采样到的原始数组汇总成报告字典。

    `eval_gait.py` 只负责采样与打印，**所有口径与聚合都在这里**，因此这条链路可以在没有
    GPU / Isaac Lab 的机器上用合成数据回归（`tests/test_gait_metrics.py`）。

    入参都是**未裁剪**的整段数组（warmup 在这里切），且**时间轴一律在 0**：
      contact [T,E,F]、base_z/vel_err_xy/vel_err_yaw/ep_len/done [T,E]、
      quat [T,E,4]（wxyz）、foot_pos_b [T,E,F,3]、joint_pos [T,E,J]
    """
    contact = np.asarray(contact).astype(np.int8)
    if contact.ndim != 3:
        raise ValueError(f"build_report: contact 形状应为 [T,E,F]，收到 {contact.shape}")
    T_full, E, F = contact.shape
    base_z = _expect("base_z", base_z, (T_full, E))
    vel_err_xy = _expect("vel_err_xy", vel_err_xy, (T_full, E))
    vel_err_yaw = _expect("vel_err_yaw", vel_err_yaw, (T_full, E))
    quat = _expect("quat", quat, (T_full, E, 4))
    foot_pos_b = _expect("foot_pos_b", foot_pos_b, (T_full, E, F, 3))
    ep_len = _expect("ep_len", ep_len, (T_full, E))
    done = _expect("done", done, (T_full, E))
    if joint_pos.ndim != 3 or joint_pos.shape[:2] != (T_full, E):
        raise ValueError(f"build_report: joint_pos 形状应为 [T,E,J]，收到 {joint_pos.shape}"
                         "（把时间轴与环境轴写反是最容易犯的错）")

    warmup = int(max(0, min(warmup, T_full - 1)))
    sl = slice(warmup, None)

    contact_w = contact[sl].transpose(1, 0, 2)          # → [E,T,F]，交给 summarize_contact
    base_z = base_z[sl].astype(np.float64)
    vel_err_xy = vel_err_xy[sl].astype(np.float64)
    vel_err_yaw = vel_err_yaw[sl].astype(np.float64)
    quat = quat[sl].astype(np.float64)
    fp_all = foot_pos_b[sl].astype(np.float64)
    joint_pos = joint_pos[sl].astype(np.float64)
    ep_len = ep_len[sl]
    done = done[sl]
    T = contact_w.shape[1]

    metrics = {
        "checkpoint": str(checkpoint),
        "iter": int(iteration),
        "num_envs": E,
        "steps": T,
        "warmup_steps": warmup,
        "steps_recorded": T_full,
        "step_dt": dt,
        "feet": list(foot_names),
        "base_height_mean": float(np.mean(base_z)),
        "base_height_std": float(np.std(base_z)),
        "vel_err_xy_mean": float(np.mean(vel_err_xy)),
        "vel_err_yaw_mean": float(np.mean(vel_err_yaw)),
        # 回放期间的重置会打断逐足时序，峰峰/步频类指标会被污染 —— 显式报出，并把
        # 「末步的情节超时」与「窗口内的真重置」分开：前者每个环境必然各一次（片长=horizon），
        # 不污染统计；后者才会。
        "resets_total": int(done.sum()),
        "resets_per_env": [int(x) for x in done.sum(axis=0)],
        "envs_with_reset": int((done.sum(axis=0) > 0).sum()),
        "resets_at_last_step": int(done[-1].sum()) if T else 0,
        "resets_before_last_step": int(done[:-1].sum()) if T > 1 else 0,
        # 注意：episode_length_buf 在无重置时就是 0→T−1 的斜坡，它的**时间均值**恒为
        # (T−1)/2，不含任何重置信息（2026-09-18 v1 那份 JSON 里的 499.5 正是斜坡均值，
        # 不是「情节长度 500 步」）。判断有没有被重置要看 done 计数与**末步缓冲值**：
        # 末步缓冲≈steps_recorded 才说明整段没被重置（若末步刚好超时，则会是 0）。
        "episode_length_final_steps_per_env": [int(x) for x in ep_len[-1]],
        "episode_length_final_steps_mean": float(np.mean(ep_len[-1])),
    }
    if meta:
        metrics.update(meta)

    # ---- 机身姿态（前倾/侧倾），带 dt 以便区分「单调漂移」与「来回摆」----
    metrics.update(body_attitude_summary(quat, dt=dt))

    # ---- 逐足接触指标 ----
    per_foot, gait = summarize_contact(contact_w, dt, foot_names, ref_index=0)
    metrics.update(gait)

    # ---- 逐足行程/抬脚（base 系峰峰）----
    # 参考基线（docs/gait_reference_baseline.json）：1.0 m/s 附近 x 行程 ≈0.29 m、
    # 抬脚 z ≈0.115 m（forward_0.9 是 0.268/0.112，forward_1.2 是 0.366/0.121）。
    for f, name in enumerate(foot_names):
        fp = fp_all[:, :, f, :]                         # [T, E, 3]
        for axis, key in ((0, "foot_stride_x_m"), (2, "foot_lift_z_m"), (1, "foot_stride_y_m")):
            mean_v, max_v, per_env = peak_to_peak_per_env(fp[:, :, axis])
            per_foot[name][key] = mean_v
            per_foot[name][key + "_max"] = max_v
            per_foot[name][key + "_per_env"] = per_env
    metrics["per_foot"] = per_foot

    # ---- 关节幅度（顺序 = cfg.joint_names，每腿 3 个：hip/thigh/shank）----
    metrics["joint_peak_to_peak_rad"] = {
        f"leg{i}": peak_to_peak_per_env(joint_pos[:, :, i])[0] for i in range(joint_pos.shape[2])
    }
    # 兼容旧字段名：v1 曾把 12 个关节都叫 shank；真正的 shank 是每腿第 3 个（i % 3 == 2）
    metrics["shank_peak_to_peak_rad"] = {
        f"leg{i}": metrics["joint_peak_to_peak_rad"][f"leg{i}"] for i in range(joint_pos.shape[2])
    }

    # 自洽检查：逐环境空中占比（对时间轴求均值）与全样本必须一致。
    # 曾经写成 c[:, e].mean()（对同一时刻的各环境求均值），得到 k/E 的离散值，
    # 看起来像「有些环境的足整段不落地」—— 那是索引 bug，不是策略行为。
    for name, m in per_foot.items():
        per_env = m["contact_air_fraction_per_env"]
        lhs = sum(per_env) / len(per_env)
        if abs(lhs - m["air_fraction"]) > 1e-6:
            raise AssertionError(
                f"{name}: 逐环境空中占比均值 {lhs:.6f} != 全样本 {m['air_fraction']:.6f}"
                "（聚合顺序不一致，指标不可用）")

    if dump_timeseries:
        metrics["timeseries"] = {
            "t": (np.asarray(t_series)[sl].tolist() if t_series is not None
                  else (np.arange(T) * dt).tolist()),
            "contact": contact[sl].tolist(),   # 与输入同布局：[T,E,F]
            "base_z": base_z.tolist(),
        }
    return metrics
