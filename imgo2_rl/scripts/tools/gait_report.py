"""按地形分组的离线步态报告：从 `play.py` dump 的 npz 里直接读出「走成什么步态」。

**为什么要有它**：`gait_kernel_probe.py` 把 `gait_metric_{trot,bound,pace}` 三个核的读数反演成
合成步态（并能用真实接触序列回放），但它回答的是「这三**个核系数**读多少」；而核系数在
`duty ≈ 0.5` 时会**饱和并趋同**（该脚本自己的实测：四足锁相下三态完全相等）；
`docs/` 里 run H 的 `0.768/0.782/0.765` 就是这个签名。也就是说**三个系数不是可分量**。
真正可分的是**足间相位差**（占空比在 0.4~0.6 之间时它几乎不随 duty 变），以及由它派生的
周期与占空比。本脚本就是量这些量，并按地形列分组给出判定。

输入 schema（由 `scripts/rl_lab/cmoe/play.py` 在回放时 dump，本脚本**只读**）：

    contact       bool  [T, N, 4]  每步/每环境/每足是否触地，足序 FL, FR, RL, RR
    base_height   f32   [T, N]     base 高度（m）
    cmd           f32   [T, N, 3]  速度指令 (vx, vy, wz)
    base_lin_vel  f32   [T, N, 3]  实际 base 线速度（世界系 xy + z）
    terrain_type  i32   [N]        每个环境所在**地形列**索引（→ terrain_names）
    terrain_level i32   [N]        每个环境的课程等级
    terrain_names str   [n_cols]   列索引 → 地形名（flat / boxes / gap / random_rough …）
    dt            f32   标量       控制步长（秒）
    foot_names    str   [4]        足名（FL_FOOT …）

**边界一律 fail-soft**：老 npz 缺 `terrain_type`/`cmd`/`dt`… 就退化成「单个未知地形 / 不过滤 /
dt=0.02」并打印告警；样本不足、全腾空、某只足整段不触地都只产出 `NaN` + 标注，不抛异常。

用法：

    python scripts/tools/gait_report.py --npz logs/cmoe/.../play_dump.npz
    python scripts/tools/gait_report.py --npz x.npz --by-level --vx-min 0.3 --json out.json

指标定义（与 `gait_metrics.py` / `gait_kernel_probe.py` 的口径对齐）：

* ``duty[i]``  — 只统计「在走」的步（``|cmd.xy| > --vx-min``）：足 i 触地步数 / 有效步数。
* ``period``   — 对**单只参考足**的触地序列（默认 FL）做自相关，取**第一个显著峰**
  （峰高 > 0.2 且 lag > 2 步，并要求峰高 ≥ 0.8×最强候选峰，避免被占空比造成的次峰带偏），
  ``lag × dt``。**为什么不用四足之和**：trot/bound/pace 的「和」一个周期内有两个脉冲，
  自相关峰出现在 T/2 上，会把周期系统性低估一半；`--period-source sum` 保留该口径只为对照。
* 相位差 ``phase[i,j]`` — 两条触地序列的**归一化循环互相关**（Pearson：逐 lag 减均值、
  除以各自标准差，所以 duty 偏离 0.5 时峰高下降、峰位不受影响）取最大 lag，
  **以最接近 0 的循环 lag 报告**（±P/2 内唯一），``phase = (lag / period) mod 1``。
  0.0＝同相，0.5＝反相；**只有周期有效（非 NaN）时才计算**，否则相位全 NaN。
* 判定 — 把 6 对相位与 ``reference_phase_deg()`` 里四个参考模式相减（**圆周距离**，
  所以 0.95 与 0.05 的差是 0.1 而不是 0.9），取相对 RMS 残差最小的模式；最小残差 > 0.25
  给 ``无明确模式``。trot/bound/pace 的参考表是**结构性事实**（由 ``gait_offsets()`` 的足相位
  偏置合成后算出），lockstep 是四足同相的极限；``test_gait_report.py`` 用合成数据把
  「实测相位 == 参考表」这条不变量钉住。

**为什么是相位差、不是三个核系数**：三个核系数算的是「一足的滞空时间 vs 另一足的触地时间」
的六对时间差，当 duty≈0.5 时滞空≈触地，任何配对都近似满足 ⇒ 三态读数趋同、步态不可分；
相位差直接量的是**两条 0/1 序列的相对时移**，在同样的 duty 下 0.0 与 0.5 依然差满半个周期，
所以它是可分量。相位单独看也无法判断「是 trot 还是四足同时蹦」，所以本报告把周期、占空比、
相位集中度与各模式残差一起给出来。

纯离线：只用标准库 + numpy（无 torch / Isaac Lab / GPU）。足序与步态相位偏置从
`gait_kernel_probe.py` 经 ``sys.path`` 复用（`FEET`、`gait_offsets()`），不另抄一份。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# 足序是 npz schema 的一部分（play.py 的 contact 最后一维恒为 FL, FR, RL, RR），
# 足名与步态相位偏置都从 `gait_kernel_probe.py` 复用，避免两处各写一份后漂移。
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from gait_kernel_probe import FEET, gait_offsets  # noqa: E402

# 六对足（上三角，定义“谁相对谁”）：(FL,FR) 前沿对、(FL,RL) 同侧对、(FL,RR) 对角对、
# (FR,RL) 对角对、(FR,RR) 同侧对、(RL,RR) 后沿对。
PAIRS: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))

# 参考模式：`gait_offsets()` 给出的足相位偏置决定了每个 (i,j) 的期望相位
# （0.0＝同相，0.5＝反相）；数值由 `reference_phase_deg()` 用本文件的估计器算出，不是手抄：
#   * trot     offsets FL=RR=0, FR=RL=0.5  ⇒ 对角对 0.0、其余 0.5
#   * bound    offsets FL=FR=0, RL=RR=0.5  ⇒ 前后沿对 0.0、其余 0.5
#   * pace     offsets FL=RL=0, FR=RR=0.5  ⇒ 同侧对 0.0、其余 0.5
#   * lockstep 四足同相 ⇒ 全 0.0
# 「同相」是循环意义上的：0.95 与 0.05 同相，所以残差一律用圆周距离。
REFERENCE_GAITS: tuple[str, ...] = ("trot", "bound", "pace", "lockstep")

# 判定阈值：最小残差都大于它 ⇒ `无明确模式`。
# 0.25 ≈ 相位平均差 1/4 周期；合成 trot/bound/pace 的残差是 0，2% 观测噪声下约 0.05，
# 随机乱走的相位差 RMS 约 0.35~0.4，所以它能分开「有明确模式」与「乱走」。
VERDICT_RESIDUAL_MAX = 0.25

DEFAULT_DT = 0.02  # 训练侧 step_dt；老 npz 没存 dt 时按它算，并打印告警

# schema 白名单，仅用于报错提示（缺它不致命，`contact` 是唯一硬要求）
CONTRACT_KEYS = (
    "contact", "base_height", "cmd", "base_lin_vel",
    "terrain_type", "terrain_level", "terrain_names", "dt", "foot_names",
)


class SchemaError(ValueError):
    """npz 缺少必需字段或形状不可用（`contact` 是唯一的硬要求）。"""


# ---------------------------------------------------------------------------
# 载入
# ---------------------------------------------------------------------------

def load_npz(path: Path) -> dict:
    """读 npz，返回 ``{键: 数组}``，不做任何形状改写。

    ``allow_pickle=True`` 是必须的：`terrain_names` / `foot_names` 是 ``<U`` 字符串数组，
    老 npz 里偶尔会是 object 数组。
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"找不到 npz：{path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def prepare_arrays(data: dict, warn) -> dict:
    """把 npz 里的原始数组整理成分析用的规范形式，缺字段一律退化 + 告警。

    返回 dict：``contact[T,N,4] bool``、``cmd[T,N,3] 或 None``、``terrain_index[N]``
    （缺 ⇒ 全 -1）、``terrain_level[N]``（缺 ⇒ 0）、``terrain_names``、``base_height[T,N]``、
    ``base_lin_vel[T,N,3]``、``dt``、``foot_names``、``notes``。

    `contact` 是唯一硬要求：没有它无法做任何步态分析（不是「可以退化」的字段）。
    """
    notes: list[str] = []
    if not isinstance(data, dict) or "contact" not in data:
        available = sorted(data) if isinstance(data, dict) else "?"
        raise SchemaError(f"npz 里没有 `contact`（有 {available}）；契约字段：{', '.join(CONTRACT_KEYS)}")

    contact = np.asarray(data["contact"]).astype(bool)
    if contact.ndim == 2:                       # 容忍单环境的 [T, 4]
        warn("[WARN] contact 是 [T,4]，按 N=1 处理（schema 是 [T,N,4]）")
        contact = contact[:, None, :]
        notes.append("contact 输入为 [T,4]，已补环境轴")
    if contact.ndim != 3:
        raise SchemaError(f"contact 形状应为 [T,N,4]，收到 {contact.shape}")
    steps, num_envs, num_feet = contact.shape
    if num_envs == 0 or steps == 0:
        raise SchemaError(f"contact 是空的（形状 {contact.shape}），没有可分析的步")
    if num_feet < 4:
        # 少足也能算：缺的那只全程 NaN + 标注（不猜足名）
        warn(f"[WARN] contact 最后一维只有 {num_feet} 只足（schema 是 4：FL,FR,RL,RR），缺的足按 NaN 报")
        notes.append(f"contact 只有 {num_feet} 只足，缺少的足一律 NaN")

    # ---- dt ----
    dt = None
    if "dt" in data:
        try:
            dt = float(np.asarray(data["dt"]).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            dt = None
    if dt is None or not np.isfinite(dt) or dt <= 0.0:
        warn(f"[WARN] 缺 dt 或 dt 非法（{dt!r}），按默认 {DEFAULT_DT} s 计算")
        notes.append(f"dt 缺失或非法，按 {DEFAULT_DT} s 计算")
        dt = DEFAULT_DT

    # ---- cmd（只用于「是否在走」的过滤）----
    cmd = None
    if "cmd" in data:
        raw = np.asarray(data["cmd"], dtype=np.float64)
        if raw.ndim == 3 and raw.shape[:2] == (steps, num_envs) and raw.shape[2] >= 2:
            cmd = raw[:, :, :3] if raw.shape[2] >= 3 else raw
        else:
            warn(f"[WARN] cmd 形状 {raw.shape} 与 contact {(steps, num_envs)} 不匹配，忽略（不做过滤）")
            notes.append("cmd 形状不匹配，未做「在走」过滤")
    else:
        warn("[WARN] npz 里没有 cmd，无法只统计「在走」的步 ⇒ 全部步计入有效步")
        notes.append("缺 cmd ⇒ 未做「在走」过滤，全部步计入")

    # ---- 地形分组信息 ----
    terrain_index = np.full(num_envs, -1, dtype=np.int64)
    if "terrain_type" in data:
        raw = np.asarray(data["terrain_type"]).reshape(-1).astype(np.int64)
        if raw.size == num_envs:
            terrain_index = raw
        else:
            warn(f"[WARN] terrain_type 长度 {raw.size} != 环境数 {num_envs}，退化为单组")
            notes.append("terrain_type 长度不匹配 ⇒ 单组")
    else:
        warn("[WARN] npz 里没有 terrain_type（老 schema？）⇒ 所有环境归入同一组")
        notes.append("缺 terrain_type ⇒ 单组")

    terrain_level = np.zeros(num_envs, dtype=np.int64)
    if "terrain_level" in data:
        raw = np.asarray(data["terrain_level"]).reshape(-1).astype(np.int64)
        if raw.size == num_envs:
            terrain_level = raw
        else:
            warn(f"[WARN] terrain_level 长度 {raw.size} != 环境数 {num_envs}，忽略等级")
            notes.append("terrain_level 长度不匹配 ⇒ 忽略等级")

    terrain_names: tuple[str, ...] = ()
    if "terrain_names" in data:
        terrain_names = tuple(str(x) for x in np.asarray(data["terrain_names"]).reshape(-1))

    # ---- 顺带输出的标量 ----
    base_height = None
    if "base_height" in data:
        raw = np.asarray(data["base_height"], dtype=np.float64)
        if raw.shape == (steps, num_envs):
            base_height = raw
        else:
            warn(f"[WARN] base_height 形状 {raw.shape} != {(steps, num_envs)}，跳过该指标")
    else:
        warn("[WARN] 没有 base_height，平均高度报 NaN")

    base_lin_vel = None
    if "base_lin_vel" in data:
        raw = np.asarray(data["base_lin_vel"], dtype=np.float64)
        if raw.ndim == 3 and raw.shape[:2] == (steps, num_envs) and raw.shape[2] >= 2:
            base_lin_vel = raw[:, :, :2]      # 只用到 xy（水平速度）
        else:
            warn(f"[WARN] base_lin_vel 形状 {raw.shape} 与 {(steps, num_envs)} 不匹配，跳过该指标")
    else:
        warn("[WARN] 没有 base_lin_vel，实际速度报 NaN")

    foot_names = tuple(FEET[:num_feet]) if num_feet <= 4 else tuple(FEET)
    if "foot_names" in data:
        names = tuple(str(x) for x in np.asarray(data["foot_names"]).reshape(-1))
        if len(names) >= num_feet:
            foot_names = names[:num_feet]
        else:
            warn(f"[WARN] foot_names 只有 {len(names)} 个（contact 有 {num_feet} 只足），按 FL,FR,RL,RR 补齐")

    return {
        "contact": contact, "cmd": cmd,
        "terrain_index": terrain_index, "terrain_level": terrain_level,
        "terrain_names": terrain_names,
        "base_height": base_height, "base_lin_vel": base_lin_vel,
        "dt": dt, "foot_names": foot_names, "notes": notes,
        "steps": steps, "num_envs": num_envs, "num_feet": num_feet,
    }


# ---------------------------------------------------------------------------
# 周期：单足触地序列的自相关
# ---------------------------------------------------------------------------

def period_from_autocorr(signal, dt: float, min_peak: float = 0.2, min_lag: int = 2,
                         peak_frac: float = 0.8, smooth: int = 3) -> float:
    """单条 0/1 序列 → 步态周期（秒）；找不到显著峰返回 ``nan``。

    * 归一化：``r[k] = <x[t]·x[t+k]> / <x[t]²>``（**有偏**：分母恒为总能量）。无偏版本
      （逐 lag 除以重叠长度）在 duty≈0.5 的方波上会出现一个宽平台，平台右端会被误判成
      「第一个显著峰」—— 实测把 P=30 步读成 10 步；有偏版本只在真正的周期倍数上出峰。
    * ``smooth`` 点滑动平均抑制 50% 占空比方波的 1 步双峰（那种信号的自相关峰是双峰的，
      直接取局部极大可能提前 1 步）。
    * 候选峰必须 ``> min_peak`` 且 ``>= peak_frac × 最强候选峰``；在满足的里面取**最小 lag**，
      这样周期整数倍处的更高峰不会被选中。
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    if x.size < 2 * min_lag + 3:
        return float("nan")
    x = x - float(x.mean())
    energy = float(np.dot(x, x))
    if energy <= 0.0:                      # 整段不触地 / 整段触地：没有周期可言
        return float("nan")

    # 线性自相关用一次 2T 点 FFT 算（前 T 个 lag 就是无环绕的线性相关，不会被折回）
    length = x.size
    spectrum = np.fft.rfft(x, n=2 * length)
    acf = np.fft.irfft(spectrum * np.conj(spectrum), n=2 * length)[:length]
    r = acf / energy

    half = smooth // 2
    if half > 0 and r.size > 2 * half:
        r = np.convolve(np.concatenate([r[-half:], r, r[:half]]), np.ones(smooth) / smooth, mode="valid")

    total = r.size
    # 上限取 min(T-2, max(T//3, min_lag+2))：既保证有 ±1 邻居可比较，又不把 1/3 片长之后的
    # 长 lag（重叠样本少、噪声大）当候选。
    hi = min(total - 2, max(int(total // 3), min_lag + 2))
    if hi <= min_lag + 1:
        return float("nan")
    candidates = [lag for lag in range(min_lag + 1, hi)
                  if r[lag] > min_peak and r[lag] >= r[lag - 1] and r[lag] >= r[lag + 1]]
    if not candidates:
        return float("nan")
    best = max(float(r[lag]) for lag in candidates)
    first = min(lag for lag in candidates if float(r[lag]) >= peak_frac * best)
    return float(first) * float(dt)


def estimate_period(contact, dt: float, min_peak: float = 0.2, min_lag: int = 2,
                    source: str = "foot0") -> tuple[float, float, int]:
    """逐环境估计周期 → ``(中位数, 逐环境标准差, 有效环境数)``。

    * ``source="foot0"``：只用**第一只足**的触地序列。默认值，因为 trot/bound/pace 的
      「四足之和」一个周期内有两个脉冲，自相关会给出 T/2。
    * ``source="sum"``：四足之和（task 里提到的对照口径，结果通常只有真实周期的一半）。
    * ``source="any"``：四足各自估计后取中位数（对某只足整段不触地更稳健）。

    逐环境算再取中位数（不是拼成一长条），因为窗口边界会截断序列，拼接处的假边界会在
    自相关里引入额外的 lag。
    """
    frames = np.asarray(contact).astype(np.float64)
    if frames.ndim == 2:
        frames = frames[:, None, :]
    _steps, num_envs, num_feet = frames.shape
    per_env: list[float] = []
    for env in range(num_envs):
        if source == "sum":
            values = [period_from_autocorr(frames[:, env, :].sum(axis=1), dt, min_peak, min_lag)]
        elif source == "any":
            values = [period_from_autocorr(frames[:, env, foot], dt, min_peak, min_lag)
                      for foot in range(num_feet)]
        else:
            values = [period_from_autocorr(frames[:, env, 0], dt, min_peak, min_lag)]
        finite = [value for value in values if np.isfinite(value)]
        if finite:
            per_env.append(float(np.median(finite)))
    if not per_env:
        return float("nan"), float("nan"), 0
    return float(np.median(per_env)), float(np.std(per_env)), len(per_env)


# ---------------------------------------------------------------------------
# 相位：成对归一化循环互相关
# ---------------------------------------------------------------------------

def pair_phase(series_a, series_b, period_steps: int) -> tuple[float, float]:
    """两条等长 0/1 序列 → ``(相位 in [0,1), 峰值相关系数)``；不可算给 ``(nan, nan)``。

    做法：把 period 当环绕长度做**归一化循环互相关**（Pearson：逐 lag 减均值、除以各自
    标准差），取最大 lag，再折到最接近 0 的循环 lag（±P/2 内唯一）。

    相位定义与 ``gait_kernel_probe.gait_offsets()`` 的足偏置一致：那只足在周期内的**触地
    起点**相对参考足的偏移。用它反推的参考模式表见 ``reference_phase_deg()``。

    为什么以「最接近 0」报告：相位是 0.0（同相）/0.5（反相）两个物理量的判据，正负号只
    反映「谁先落地」这种约定。
    """
    a = np.asarray(series_a, dtype=np.float64).ravel()
    b = np.asarray(series_b, dtype=np.float64).ravel()
    size = int(period_steps)
    if size < 2 or a.size < 2 or b.size < 2 or min(a.size, b.size) < size:
        return float("nan"), float("nan")

    # 用「整数个周期」的长度做循环相关（整周期 ⇒ 环绕处没有拼接假象）。
    # 若序列有多个周期，把每个周期块的相关**取平均**（比只看第一块稳）。
    blocks = int(min(a.size, b.size) // size)
    correlations = np.zeros(size, dtype=np.float64)
    usable_blocks = 0
    for block in range(blocks):
        start, stop = block * size, (block + 1) * size
        piece_a = a[start:stop] - float(a[start:stop].mean())
        piece_b = b[start:stop] - float(b[start:stop].mean())
        norm_a = float(np.sqrt(np.dot(piece_a, piece_a)))
        norm_b = float(np.sqrt(np.dot(piece_b, piece_b)))
        if norm_a <= 0.0 or norm_b <= 0.0:      # 该块内某只足恒定 ⇒ 这块无相位可言
            continue
        piece_a, piece_b = piece_a / norm_a, piece_b / norm_b
        correlations += np.fft.irfft(np.fft.rfft(piece_a, n=size)
                                     * np.conj(np.fft.rfft(piece_b, n=size)), n=size)
        usable_blocks += 1
    if usable_blocks == 0:
        return float("nan"), float("nan")
    correlations /= usable_blocks

    lags = np.arange(size)
    wrapped = ((lags + size // 2) % size) - size // 2     # 折到 [-P/2, P/2)
    order = np.argsort(wrapped, kind="stable")            # 按 |lag| 升序（并列取小的）
    best = order[int(np.argmax(correlations[order]))]
    return float((int(wrapped[best]) / size) % 1.0), float(correlations[best])


def phase_table(contact_mask, period: float, dt: float
                ) -> tuple[dict[tuple[int, int], tuple[float, float]], dict[tuple[int, int], float]]:
    """``contact_mask[T, N, F]``（已按有效步掩码置零）→ 6 对的 ``(相位, 峰值相关)`` + 集中度。

    逐环境算相位再取**圆周平均**（峰值与集中度另算）：
      * 直接对相位取算术平均会在 0/1 边界上出错（0.95 与 0.05 的均值是 0.5 而不是 0.0）；
      * 集中度 ``R`` 低说明组内各环境的相位不一致（组里可能混着两种步态），判定要当心；
      * ``period`` 无效 ⇒ 全部 NaN（相位只有在已知周期时才有意义）。
    """
    frames = np.asarray(contact_mask).astype(np.float64)
    if frames.ndim == 2:
        frames = frames[:, None, :]
    _steps, num_envs, num_feet = frames.shape
    empty = {pair: (float("nan"), float("nan")) for pair in PAIRS}
    if not np.isfinite(period) or period <= 0 or not np.isfinite(dt) or dt <= 0:
        return empty, {pair: float("nan") for pair in PAIRS}
    period_steps = int(round(period / dt))

    phases: dict[tuple[int, int], tuple[float, float]] = {}
    concentration: dict[tuple[int, int], float] = {}
    for pair in PAIRS:
        i, j = pair
        if i >= num_feet or j >= num_feet:
            phases[pair] = (float("nan"), float("nan"))
            concentration[pair] = float("nan")
            continue
        env_phases: list[float] = []
        peaks: list[float] = []
        for env in range(num_envs):
            phase, peak = pair_phase(frames[:, env, i], frames[:, env, j], period_steps)
            if np.isfinite(phase):
                env_phases.append(phase)
            if np.isfinite(peak):
                peaks.append(peak)
        if not env_phases:
            phases[pair] = (float("nan"), float("nan"))
            concentration[pair] = float("nan")
            continue
        angles = np.asarray(env_phases, dtype=np.float64) * (2.0 * np.pi)
        mean_c, mean_s = float(np.cos(angles).mean()), float(np.sin(angles).mean())
        phases[pair] = (
            float((np.arctan2(mean_s, mean_c) / (2.0 * np.pi)) % 1.0),
            float(np.mean(peaks)) if peaks else float("nan"),
        )
        concentration[pair] = float(np.hypot(mean_c, mean_s))
    return phases, concentration


def circular_distance(a: float, b: float) -> float:
    """两个相位（周期单位）的圆周距离，落在 [0, 0.5]。0.95 与 0.05 的距离是 0.1。"""
    delta = (float(a) - float(b)) % 1.0
    return float(min(delta, 1.0 - delta))


def reference_phase_deg(period: float = 0.6, duty: float = 0.5, dt: float = 0.02,
                        steps: int = 2400) -> dict[str, dict[tuple[int, int], float]]:
    """四个参考模式在 ``PAIRS`` 上的相位表（由 ``gait_offsets()`` 的足偏置合成后**用本文件的
    相位估计器算出来**，不是手抄的常数）。

    自洽性由 ``tests/test_gait_report.py`` 锁定：合成 trot 的实测相位必须等于这里的
    ``trot`` 参考表。周期取 0.6 s（30 步，整步）以免舍入影响；相位对周期不敏感
    （已核对 0.34/0.5/0.6 s 给出同样的 0.0/0.5 结构）。
    """
    tables: dict[str, dict[tuple[int, int], float]] = {}
    for name in REFERENCE_GAITS:
        offsets = gait_offsets(name)
        frames = np.zeros((steps, 1, 4), dtype=bool)
        for index in range(steps):
            time = index * dt
            for foot_id, foot in enumerate(FEET):
                frames[index, 0, foot_id] = ((time / period + offsets[foot]) % 1.0) < duty
        table, _concentration = phase_table(frames, period, dt)
        tables[name] = {pair: value[0] for pair, value in table.items()}
    return tables


def _residuals(phases: dict[tuple[int, int], tuple[float, float]],
               references: dict[str, dict[tuple[int, int], float]]) -> dict[str, float]:
    """6 对相位的圆周残差 → 每个参考模式的相对 RMS（``sqrt(mean(Δ²))``，量纲仍是周期）。"""
    scores: dict[str, float] = {}
    for name, reference in references.items():
        terms = []
        for pair in PAIRS:
            measured = phases.get(pair, (float("nan"), float("nan")))[0]
            if not np.isfinite(measured):
                continue
            terms.append(circular_distance(measured, reference[pair]) ** 2)
        scores[name] = float(np.sqrt(np.mean(terms))) if terms else float("nan")
    return scores


def verdict(phases: dict[tuple[int, int], tuple[float, float]],
            references: dict[str, dict[tuple[int, int], float]] | None = None,
            threshold: float = VERDICT_RESIDUAL_MAX) -> tuple[str, str, dict[str, float]]:
    """按「最接近哪个参考模式」给判定 → ``(标签, 中文说明, 残差 dict)``。

    * 相位全 NaN（周期无效 / 某足整段不触地）⇒ ``无法判定``；
    * 最小残差 > ``threshold`` ⇒ ``无明确模式``（乱走或相位不规整，不硬套一个模式）；
    * 否则取最小残差的模式；lockstep 另给「四足近同时」的说明。

    残差是**相对 RMS**（圆周距离），两种模式只差半周期时数值差会被压小 —— 这是相位判定的
    固有性质（0.0 与 0.5 在圆周上是对称两端），所以残差一并打印，由人判断「领先是否显著」。
    """
    references = references if references is not None else reference_phase_deg()
    scores = _residuals(phases, references)
    finite = {name: value for name, value in scores.items() if np.isfinite(value)}
    if not finite:
        return "无法判定", "相位不可算（周期无效，或某只足整段不触地）", scores
    best = min(finite, key=lambda name: finite[name])
    if finite[best] > threshold:
        return "无明确模式", f"最接近的 {best} 残差也有 {finite[best]:.3f} > {threshold}", scores
    if best == "lockstep":
        spread = max(finite.values()) - min(finite.values())
        extra = "（四足近同时）" if spread < 0.15 else ""
        return "lockstep", f"四足近同时{extra}，残差 {finite[best]:.3f}", scores
    return best, f"最接近 {best}（残差 {finite[best]:.3f}）", scores


# ---------------------------------------------------------------------------
# 分组 + 指标
# ---------------------------------------------------------------------------

def group_environments(terrain_index, terrain_level, by_level: bool) -> list[dict]:
    """环境 → 分组（按 ``terrain_type``，``by_level`` 时再按等级细分）。

    返回按 (列索引, 等级) 排序的组列表，每组给 ``env_ids``、``terrain_index``、``level``；
    ``by_level`` 关闭时 ``level`` 是该组**唯一**等级，混杂则为 -1。
    """
    terrain_index = np.asarray(terrain_index).reshape(-1)
    terrain_level = np.asarray(terrain_level).reshape(-1)
    buckets: dict[tuple[int, int], list[int]] = {}
    for env in range(terrain_index.size):
        column = int(terrain_index[env])
        level = int(terrain_level[env]) if by_level else -1
        buckets.setdefault((column, level), []).append(env)
    groups = []
    for (column, level), env_ids in sorted(buckets.items()):
        if level < 0:
            unique = np.unique(terrain_level[env_ids])
            level = int(unique[0]) if unique.size == 1 else -1
        groups.append({"terrain_index": column, "level": int(level),
                       "env_ids": np.asarray(env_ids, dtype=np.int64)})
    return groups


def build_group_report(arrays: dict, group: dict, min_steps: int = 200, vx_min: float = 0.3,
                       period_source: str = "foot0", min_peak: float = 0.2,
                       min_lag: int = 2) -> dict:
    """单个地形组 → 指标 dict（**永不抛异常**：样本不足/全腾空都走标注分支）。"""
    contact = arrays["contact"]
    dt = float(arrays["dt"])
    steps = contact.shape[0]
    env_ids = group["env_ids"]

    column = group["terrain_index"]
    names = arrays["terrain_names"]
    name = names[column] if 0 <= column < len(names) else f"列{column}"
    report: dict = {
        "terrain": name, "terrain_index": int(column), "level": int(group["level"]),
        "envs": int(env_ids.size), "steps_total": int(env_ids.size * steps),
        "steps_valid": 0, "walk_fraction": float("nan"),
        "status": "ok", "notes": [], "metrics": None,
    }

    # ---- 有效步掩码：|cmd.xy| > vx_min ----
    cmd = arrays["cmd"]
    if cmd is None:
        valid = np.ones((steps, env_ids.size), dtype=bool)
    else:
        speed = np.linalg.norm(np.nan_to_num(cmd[:, env_ids, :2], nan=0.0), axis=-1)
        valid = speed > float(vx_min)
    report["steps_valid"] = int(valid.sum())
    report["walk_fraction"] = float(valid.mean()) if valid.size else float("nan")

    if report["steps_valid"] < int(min_steps):
        report["status"] = "insufficient"
        report["notes"].append(f"样本不足（有效步 {report['steps_valid']} < --min-steps {min_steps}）"
                               "⇒ 跳过指标计算")
        return report

    # ---- 掩码外的步置零：全腾空检测与逐足 duty 都只看有效步 ----
    sub = contact[:, env_ids, :].copy()
    sub[~valid] = False
    feet_present = int(min(sub.shape[2], 4))
    if not sub.any():
        report["status"] = "no_contact"
        report["notes"].append("组内所有足在有效步里**整段未触地**（一直腾空/摔倒）⇒ 指标全 NaN")
        return report

    # ---- 周期 ----
    period, period_std, period_envs = estimate_period(sub, dt, min_peak, min_lag, period_source)
    if not np.isfinite(period) and period_source == "foot0":
        # 2026-09-28 真实 dump 的经验：duty 高（足 80–90% 时间触地）时第一只足的触地序列几乎恒定，
        # 自相关没有显著峰 ⇒ foot0 给 NaN。这时**自动回退**到"四足各自估计取中位数"，能救回大部分列。
        fallback, fallback_std, fallback_envs = estimate_period(sub, dt, min_peak, min_lag, "any")
        if np.isfinite(fallback):
            period, period_std, period_envs = fallback, fallback_std, fallback_envs
            period_source = "foot0→any(回退)"
    period_valid = bool(np.isfinite(period) and period > 0.0)
    if not period_valid:
        report["notes"].append("未找到显著自相关峰（峰高 > 0.2 且 lag > 2）⇒ period 与相位全 NaN")

    # ---- duty（逐足；某足整段不触地 ⇒ NaN + 标注）----
    duty: list[float] = []
    for foot in range(feet_present):
        signal = sub[:, :, foot]
        duty.append(float(signal.mean()) if signal.any() else float("nan"))
        if not signal.any():
            foot_label = FEET[foot] if foot < 4 else f"足{foot}"
            report["notes"].append(f"{foot_label} 足整段不触地 ⇒ 该足 duty/相位 NaN")
    duty_full = duty + [float("nan")] * (4 - feet_present)

    # ---- 相位（只有周期有效时才算）----
    phase, phase_concentration = phase_table(sub, period, dt)
    if not period_valid:
        report["notes"].append("周期无效 ⇒ 相位未计算")

    label, reason, scores = verdict(phase)
    if not period_valid:
        label, reason = "无法判定", "周期无效 ⇒ 相位未计算"

    # ---- 顺带的标量 ----
    base_height = arrays["base_height"]
    base_lin_vel = arrays["base_lin_vel"]
    height_mean = (float(np.nanmean(base_height[:, env_ids][valid])) if base_height is not None
                   else float("nan"))
    speed_mean = (float(np.nanmean(np.linalg.norm(base_lin_vel[:, env_ids, :2], axis=-1)[valid]))
                  if base_lin_vel is not None else float("nan"))
    cmd_speed_mean = (float(np.nanmean(np.linalg.norm(cmd[:, env_ids, :2], axis=-1)[valid]))
                      if cmd is not None else float("nan"))

    def pair_key(pair: tuple[int, int]) -> str:
        return f"{FEET[pair[0]]}-{FEET[pair[1]]}"

    report["metrics"] = {
        "period_s": period if period_valid else float("nan"),
        "period_std_s": period_std if period_valid else float("nan"),
        "period_envs": int(period_envs),
        "period_source": period_source,
        "period_valid": period_valid,
        "duty": duty_full,
        "phase": {pair_key(pair): phase[pair][0] for pair in PAIRS},
        "phase_peak": {pair_key(pair): phase[pair][1] for pair in PAIRS},
        "phase_concentration": {pair_key(pair): phase_concentration[pair] for pair in PAIRS},
        "verdict": label,
        "verdict_reason": reason,
        "residuals": scores,
        "base_height_mean_m": height_mean,
        "base_speed_mean_mps": speed_mean,
        "cmd_speed_mean_mps": cmd_speed_mean,
    }
    return report


def build_report(arrays: dict, min_steps: int = 200, vx_min: float = 0.3, by_level: bool = False, by_name: bool = False,
                 period_source: str = "foot0", min_peak: float = 0.2, min_lag: int = 2) -> dict:
    """整个 npz → 报告 dict（含 metadata 与逐组结果），可直接 json.dump。"""
    if by_name:
        # 2026-09-28：`sub_terrains` 是"每类占连续若干列"（如 gap 占 10 列）⇒ 逐列输出会有几十行、
        # 名字还重复。`--by-name` 把 terrain_index 重映射成"按名字去重后的 id"，于是每个**地形类型**一行。
        names = arrays.get("terrain_names")
        if names is not None:
            import numpy as _np
            names = [str(x) for x in _np.asarray(names).reshape(-1)]
            index = _np.asarray(arrays["terrain_index"]).reshape(-1)
            unique = sorted({names[i] if 0 <= i < len(names) else f"列{i}" for i in index.tolist()})
            lookup = {name: k for k, name in enumerate(unique)}
            arrays = dict(arrays)
            arrays["terrain_index"] = _np.asarray(
                [lookup[names[i] if 0 <= i < len(names) else f"列{i}"] for i in index.tolist()],
                dtype=index.dtype)
            arrays["terrain_names"] = _np.asarray(unique, dtype="U")
            if arrays.get("terrain_level") is not None:
                pass          # 等级不受影响
    # ⚠️ 2026-09-28 修 bug：`groups` 必须在 `by_name` 重映射**之后**再算。
    # 原来算在重映射之前 ⇒ 分组仍按原始 40 个列索引 ⇒ `--by-name` 形同虚设（仍然 40 行），
    # 而且行标签会因为"名字数组已经换成 11 个去重名"而整体错位（实测："flat" 那一行其实是 column 10）。
    groups = group_environments(arrays["terrain_index"], arrays["terrain_level"], by_level)
    reports = [build_group_report(arrays, group, min_steps, vx_min, period_source, min_peak, min_lag)
               for group in groups]
    return {
        "metadata": {
            "steps": int(arrays["steps"]), "num_envs": int(arrays["num_envs"]),
            "num_feet": int(arrays["num_feet"]), "dt": float(arrays["dt"]),
            "foot_names": list(arrays["foot_names"]), "terrain_names": list(arrays["terrain_names"]),
            "min_steps": int(min_steps), "vx_min": float(vx_min), "by_level": bool(by_level),
            "warnings": list(arrays["notes"]),
        },
        "groups": reports,
    }


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def _cell(value, width: int = 8, places: int = 3) -> str:
    """数值 → 定宽单元格；NaN/inf 打印 ``NaN``（打印成 0.000 会被读成「真值 0」）。"""
    if isinstance(value, str):
        return value.rjust(width)
    if value is None:
        return f"{'NaN':>{width}}"
    number = float(value)
    if not np.isfinite(number):
        return f"{'NaN':>{width}}"
    return f"{number:>{width}.{places}f}"


def _row(cells: list[str], widths: list[int]) -> str:
    return "  ".join(cell.rjust(width) for cell, width in zip(cells, widths))


def format_report(report: dict) -> str:
    """报告 dict → 终端表格（中文，NaN 显式打印）。"""
    lines: list[str] = []
    meta = report["metadata"]
    lines.append(f"步态报告：{meta['steps']} 步 × {meta['num_envs']} 环境 × {meta['num_feet']} 足，"
                 f"dt={_cell(meta['dt'], 6, 4)} s（{meta['steps'] * meta['dt']:.1f} s/环境）"
                 + (f"；源 {meta['source']}" if meta.get("source") else ""))
    lines.append(f"过滤：|cmd.xy| > {_cell(meta['vx_min'], 4)}（只统计「在走」的步）；"
                 f"分组：terrain_type{' + terrain_level' if meta['by_level'] else ''}；"
                 f"样本不足阈值 {meta['min_steps']} 步")
    for warning in meta["warnings"]:
        lines.append(f"  ⚠ {warning}")
    if not meta["terrain_names"]:
        lines.append("  ⚠ npz 无 terrain_names，地形用「列N」显示")
    lines.append("")

    widths = [14, 5, 9, 8, 7, 7, 7, 7, 8, 8, 12, 8]
    header = ["地形", "等级", "有效步", "period", "dutyFL", "dutyFR", "dutyRL", "dutyRR",
              "FL-FR相", "FL-RL相", "判定", "残差"]
    lines.append(_row(header, widths))
    lines.append("-" * (sum(widths) + 2 * (len(widths) - 1)))

    for group in report["groups"]:
        level = "混合" if group["level"] < 0 else str(group["level"])
        metrics = group["metrics"]
        if metrics is None:
            label = "样本不足" if group["status"] == "insufficient" else "全腾空"
            lines.append(_row([group["terrain"], level, str(group["steps_valid"]),
                               "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", label, "NaN"], widths))
            continue
        residual = metrics["residuals"].get(metrics["verdict"], float("nan"))
        lines.append(_row([
            group["terrain"], level, str(group["steps_valid"]),
            _cell(metrics["period_s"], 8), _cell(metrics["duty"][0], 7), _cell(metrics["duty"][1], 7),
            _cell(metrics["duty"][2], 7), _cell(metrics["duty"][3], 7),
            _cell(metrics["phase"]["FL-FR"], 8), _cell(metrics["phase"]["FL-RL"], 8),
            metrics["verdict"], _cell(residual, 8),
        ], widths))

    lines.append("")
    lines.append("残差（圆周距离的相对 RMS，越小越像；最小残差 > 0.25 ⇒ 无明确模式）：")
    for group in report["groups"]:
        if not group["metrics"]:
            continue
        metrics = group["metrics"]
        level = "混合" if group["level"] < 0 else str(group["level"])
        scores = "  ".join(f"{name}={_cell(metrics['residuals'].get(name), 7)}"
                           for name in REFERENCE_GAITS)
        lines.append(f"  {group['terrain']}#{level:<4} {scores}  ⇒ {metrics['verdict']}")

    lines.append("")
    lines.append("参考模式相位表（0.0=同相 / 0.5=反相；由 gait_kernel_probe.gait_offsets 合成后算出）：")
    references = reference_phase_deg()
    for name in REFERENCE_GAITS:
        pairs = "  ".join(f"{FEET[i]}-{FEET[j]}={references[name][(i, j)]:.2f}" for i, j in PAIRS)
        lines.append(f"  {name:<9}{pairs}")

    lines.append("")
    lines.append("逐组明细：")
    for group in report["groups"]:
        level = "混合" if group["level"] < 0 else str(group["level"])
        lines.append(f"  [{group['terrain']} / 等级 {level}] 环境 {group['envs']} 个 / "
                     f"有效步 {group['steps_valid']}（占 {_cell(group['walk_fraction'], 5)} 的有效步比例）"
                     f" ⇒ {group['status']}")
        for note in group["notes"]:
            lines.append(f"      ⚠ {note}")
        metrics = group["metrics"]
        if not metrics:
            continue
        lines.append(f"      period={_cell(metrics['period_s'], 6)} s"
                     f"（逐环境 σ={_cell(metrics['period_std_s'], 6)}，"
                     f"{metrics['period_envs']} 个环境可用，周期源={metrics['period_source']}）"
                     f"  base 高 {_cell(metrics['base_height_mean_m'], 6)} m"
                     f"  实际速度 {_cell(metrics['base_speed_mean_mps'], 6)} m/s"
                     f"  指令速度 {_cell(metrics['cmd_speed_mean_mps'], 6)} m/s")
        lines.append("      " + "  ".join(
            f"{FEET[i]}-{FEET[j]}: 相位 {_cell(metrics['phase'][f'{FEET[i]}-{FEET[j]}'], 6)}"
            f" 峰 {_cell(metrics['phase_peak'][f'{FEET[i]}-{FEET[j]}'], 5, 2)}"
            f" R {_cell(metrics['phase_concentration'][f'{FEET[i]}-{FEET[j]}'], 5, 2)}"
            for i, j in PAIRS))
    return "\n".join(lines)


def _json_safe(value):
    """NaN/inf → None：``json.dump`` 默认会写出裸 `NaN`，那不是合法 JSON，别的工具会读崩。"""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (bool, str)) or value is None:
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def write_json(report: dict, path: Path) -> None:
    """报告 → JSON 文件（NaN 一律写 null，保证是合法 JSON）。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_json_safe(report), indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="按地形列分组的离线步态报告（读 play.py dump 的 npz；纯 numpy，无需 GPU / Isaac Lab）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--npz", type=Path, required=True, help="play.py 在回放时 dump 的 .npz")
    parser.add_argument("--min-steps", type=int, default=200,
                        help="组内有效步数少于它 ⇒ 标『样本不足』并跳过指标（不是报错）")
    parser.add_argument("--vx-min", type=float, default=0.3,
                        help="「在走」判据：cmd 的 xy 模长 > 它才算有效步")
    parser.add_argument("--json", type=Path, default=None, help="把机器可读结果写到这个路径")
    parser.add_argument("--by-level", action="store_true",
                        help="分组时把 terrain_level 也拼上（默认只按地形列）")
    parser.add_argument("--names", default=None,
                        help="逗号分隔的**逐列**地形名，覆盖 npz 里的 `terrain_names`（修旧 dump 的列名错位）")
    parser.add_argument("--by-name", action="store_true",
                        help="按**地形类型名**聚合（每类占连续若干列，如 gap 占 10 列）⇒ 每类一行")
    parser.add_argument("--period-source", choices=("foot0", "sum", "any"), default="foot0",
                        help="周期用哪条序列的自相关：foot0=第一只足（默认；四足之和会给出半周期）；"
                             "sum=四足之和（对照）；any=四足各自取中位数")
    parser.add_argument("--peak-min", type=float, default=0.2, help="自相关显著峰的最小峰高")
    parser.add_argument("--min-lag", type=int, default=2, help="自相关搜索的最小 lag（步）")
    args = parser.parse_args(argv)

    def warn(message: str) -> None:
        print(message, file=sys.stderr)

    try:
        data = load_npz(args.npz)
        if args.names:                      # 2026-09-28：修旧 dump 的列名错位（sub_terrains 键数 ≠ 列数）
            data = dict(data)
            data["terrain_names"] = [x.strip() for x in args.names.split(",") if x.strip()]
        arrays = prepare_arrays(data, warn)
    except (FileNotFoundError, SchemaError, OSError, ValueError) as error:
        print(f"❌ 读 npz 失败：{error}", file=sys.stderr)
        return 1

    report = build_report(arrays, min_steps=args.min_steps, vx_min=args.vx_min,
                          by_level=args.by_level, by_name=args.by_name, period_source=args.period_source,
                          min_peak=args.peak_min, min_lag=args.min_lag)
    report["metadata"]["source"] = str(args.npz)
    print(format_report(report))

    if args.json:
        try:
            write_json(report, args.json)
        except OSError as error:
            print(f"❌ 写 JSON 失败：{error}", file=sys.stderr)
            return 1
        print(f"\n已写出 JSON：{args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
