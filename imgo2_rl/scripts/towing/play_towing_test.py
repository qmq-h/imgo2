#!/usr/bin/env python3
"""拖曳「上层任务必要性」基线测试台（Isaac Sim play 仿真）。

## 这个脚本回答什么问题

上层拖曳 RL（脚本调度的速度指令 + 12 维关节位置残差）这个任务**有没有必要**？
必要性的下限来自它的对照基线：**冻结 AMP 底层策略 + 脚本速度指令**（settle 0 →
tow v → STOP 0），负载是仓库里那台被动小车，连接是三类（弹性绳 / 低弹性绳 / 刚体球铰
连杆）。如果这条基线在本脚本的网格上已经满足全部五项指标，那么上层任务在这些工况上
就没有必要性证据；如果它系统性地在某一相失败，失败模式就是上层任务要修的对象。

**本脚本不加载任何上层 checkpoint**：跑的是基线，不是策略回放。

## 五项指标（每一项在 summary 里都有明确字段）

| 指标 | 字段 | 口径 |
|---|---|---|
| 起步关节响应误差 | `startup.joint_rms_rad` / `joint_max_rad` / `worst_joint` / `torque_saturated_frac` | 起拖后 `--transition-window`（默认 1.0 s）内，12 关节 `q − q*` 的 RMS 与最大绝对值；`q*` 是**冻结策略当拍下发的关节位置目标**（上层残留动作改的就是它），另报力矩饱和（`|τ| > 0.95·limit`）步占比 |
| 全程速度跟踪误差 | `speed.mae_mps` / `rmse_mps` / `bias_mps` / `ratio_mean` / `p95_abs_err_mps` | tow 段**全程**，**体系** x 速度（与冻结策略的观测同口径；坡上 ≠ 世界系速度）与指令之差；另有后半段稳态窗 `speed.steady_*` |
| 停止时小车滑移距离 | `stop.cart_coast_distance_m` / `cart_coast_to_rest_m` / `cart_coast_time_to_rest_s` / `cart_speed_at_stop_mps` | 从 STOP 那一刻到回合结束小车沿 x 的位移；以及速度降到 0.02 m/s 以下那一刻的距离与耗时 |
| 停止时机器人—小车距离维持 | `stop.clearance_at_stop_m` / `min_clearance_coast_m` / `final_clearance_m` / `time_to_contact_after_stop_s` / `contact` | **车头到机器人后腿的真实几何间隙**（全腿 FK，复用 `summarize_tow.py`，挂点距单独报）；接触由「车斗接触力 / 几何间隙 / 负载单步速度跃变」三路见证判定 |
| 停止时的关节响应 | `stop.joint_rms_rad` / `joint_max_rad` / `worst_joint` / `settle_time_s` / `body_vx_rms_mps` | STOP 后 `--transition-window` 内的同一套关节跟踪误差 + 机器人从指令归零到 `|vx| < 0.05 m/s` 的耗时 |

判据（阈值可用 CLI 覆盖）落成逐 case 码：`OK` / `LOW`（停车余量低）/ `JNT`（关节响应超限）/
`SPD`（速度跟踪超限）/ `COL`（追尾接触）/ `FALL`（跌倒）/ `INV`（记录不可用）。

## 默认网格（完整分布）

- 速度档：`0.5, 1.0, 1.5 m/s`（阶跃指令，`--command-shaping ramp` 可换成固定斜坡对照）；
- 连接：`compliant, inextensible, rigid` 三类；
- 质量档：`5, 10, 15, 20, 25 kg`（质量与惯量按同一比例缩放，与测量台一致）；
- 坡度：`0, +5, −5, +10, −10 deg`（`+` = 沿 +x 上坡，机器人在前、小车在后）；
- 地面摩擦：**固定 `0.8`**（工厂地面常规值：混凝土/环氧地坪静动摩擦 0.6–0.9，本仓库
  历来用 0.8，`--ground-friction` 可改）；
- 共 3 × 3 × 5 × 5 = **225 case**。

同一个坡度上的 45 个（速度 × 连接 × 质量）组合**并行放在 45 个环境**里跑一遍，
逐坡度串行 ⇒ 总共 5 次仿真过程，而不是 225 次。每个环境 = 一个固定 case（逐 env 的
小车质量、连接类型、速度指令都不同），坡度在两次过程之间切换。

## 坡度怎么实现（现有平地 + 旋转重力，物理等价）

坡度地形**还在实现中**，本脚本默认不动地形：用 `TowSceneCfg` 现有的那块平地，
把**重力方向转成坡面的法向分解** `g = (−g·sinθ, 0, −g·cosθ)`。在随坡面倾斜的参考系里，
「水平地面 + 倾斜重力」与「倾斜坡面 + 竖直重力」是**同一组方程**（接触法向、法向力
`mg·cosθ`、下滑分量 `mg·sinθ` 逐项相同），因此这不是近似替代。冻结策略的观测里本来就
有 `projected_gravity`，它会自然看到倾斜后的重力。唯一差别是出生高度是竖直 0.35 m
（坡面上量是 0.35/cosθ），可忽略。

`--slope-backend terrain` 是给**将来**的坡面地形资产留的口子：目前直接报错而不是
静默退回重力方案（要接入时改 `slope_ground_plan()` 与场景的地面 spawn 两处即可）。

## 坡度上的站定段：驻车制动仿真（`--slope-settle`，默认 `hold`）

被动小车只有轮轴黏性阻尼（`b = 0.016 N·m·s/rad`，训练侧同值），**没有驻车制动**。
斜坡上它在站定段就会自己溜：5° 时终端速度约 0.85 m/s，1 s 站定能滚出 0.8 m ——
远超 0.4 m 的初始松弛，绳在起拖之前就被拽直，于是所有坡度 case 都被这个瞬态主导，
测不出控制器的差别。因此默认 `--slope-settle hold`：**只在 station 段**给四个轮子额外加
`--hold-damping`（默认 5 N·m·s/rad）的黏性制动，起拖瞬间释放。这是明确的建模选择，
会写进每个 case 的 `config.json` 与报告；要跑「无制动真实溜坡」用 `--slope-settle free`。

## 运行

```bash
# 只看将执行的网格，不启动仿真（标准库即可）
python3 imgo2_rl/scripts/towing/play_towing_test.py --dry-run

# 默认完整网格（225 case，5 次仿真过程；建议 headless）
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/towing/play_towing_test.py --headless

# 先小规模试跑（1 个坡度 × 2 速度 × 2 连接 × 2 质量 = 8 环境）
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/towing/play_towing_test.py \
    --headless --slopes 0 --velocities 0.5 1.0 --connections compliant rigid \
    --cart-masses 5 25
```

## 产物

```
<output-dir>/
  experiment.json     运行级清单（参数、git、python、网格、阈值）
  report.json         逐 case 摘要（含 summarize_tow 全量输出）+ 分组统计 + 结论
  report.csv          逐 case 一行的关键指标表（便于表格工具）
  report.md           人读报告：逐坡度判定矩阵、失败模式计数、结论与限制
  summaries/<case>.json   每个 case 的完整摘要（总是写）
  <case>/tow.csv + config.json   逐物理步原始记录（仅 --write-csv all|failed）
```

`--write-csv` 默认 `failed`：只给非 `OK` 的 case 留原始轨迹（完整网格每步全写约 0.5 GB）。

## 判读与限制

- 结论只由「本网格 + 本阈值」给出，不能外推：未测绳参数档（训练网格有 4 档 k/c，这里
  默认只跑中间档 `k=4000, c=100`）、未测轮阻档、未测 breakaway/Coulomb 阻力、
  未测跨环境隔离、未验真机。
- `summarize_tow` 的 `steady_tracking_ratio` 用的是**世界系** vx 除以指令，坡上口径不同，
  本脚本的跟速指标一律用**体系** vx；两者都写在产物里，不要混用。
- 速度指令是体系 x 速度（与冻结策略观测一致），不含转向；本测试只研究直线拖曳。
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

RL_ROOT = Path(__file__).resolve().parents[2]          # <repo>/imgo2_rl
REPO = RL_ROOT.parent                                  # <repo>
TOOLS_DIR = RL_ROOT / "scripts" / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from summarize_tow import summarize_tow  # noqa: E402  (纯标准库，离线工具)

# ---------------------------------------------------------------- 常量与默认值

GRAVITY_MPS2 = 9.81
#: 连接类型全集。与 `mdp/rope_model.py::CONNECTION_MODELS` 交叉核对（离线测试覆盖），
#: 这里不 import 那个模块：它 import torch，会让 `--dry-run`／`--help` 失去标准库可运行性。
CONNECTIONS = ("compliant", "inextensible", "rigid")
#: 工厂地面常规摩擦系数（混凝土/环氧地坪 0.6–0.9，本仓库历来固定 0.8）。
FACTORY_FLOOR_FRICTION = 0.8
DEFAULT_VELOCITIES = (0.5, 1.0, 1.5)
DEFAULT_CART_MASSES = (5.0, 10.0, 15.0, 20.0, 25.0)
DEFAULT_SLOPES_DEG = (0.0, 5.0, -5.0, 10.0, -10.0)
N_JOINTS = 12
PHASES = ("station", "tow", "coast")

_JOINT_NAMES_FALLBACK = tuple(
    f"{leg}_{part}_joint" for leg in ("FL", "FR", "RL", "RR")
    for part in ("hip", "thigh", "shank"))

# 记录列：`summarize_tow` 需要的列（TOW_FIELDS）+ 本测试新增的列。
# `recording.py` 是纯标准库模块，按文件路径加载，避免 import 重量级包 __init__。
def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块 {name}：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recording = _load_module(
    "imgo2_play_towing_test_recording",
    RL_ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/utils/recording.py")

JOINT_TARGET_FIELDS = tuple(f"robot_jt_{index:02d}" for index in range(N_JOINTS))
JOINT_TORQUE_FIELDS = tuple(f"robot_tau_{index:02d}" for index in range(N_JOINTS))
WHEEL_LEGS = ("fl", "fr", "rl", "rr")
#: 本测试新增列。`connection` 是字符串列，其余都是数值列。
EXTRA_FIELDS = (
    "velocity_cmd_mps",     # 本 env 当前的速度指令（体系 x）
    "robot_vx_b_mps",       # 体系 x 速度 = 跟速误差用的口径
    "joint_err_rms_rad",    # 本拍 12 关节跟踪误差的 RMS（在线算，便于画曲线）
    "joint_err_max_rad",
    "slope_deg", "gravity_x_mps2", "gravity_z_mps2",
    "connection", "cart_mass_kg",
)
EXTRA_NUMERIC_FIELDS = tuple(name for name in EXTRA_FIELDS if name != "connection")
TEST_FIELDS = ("phase", *recording.TOW_NUMERIC_FIELDS,
               *JOINT_TARGET_FIELDS, *JOINT_TORQUE_FIELDS, *EXTRA_FIELDS)

#: 逐 case 判定码。数值越大越严重（`classify_case` 取最严重的一条作为 `code`）。
VERDICT_CODES = ("OK", "LOW", "JNT", "SPD", "COL", "FALL", "INV")
VERDICT_SEVERITY = {code: index for index, code in enumerate(VERDICT_CODES)}
_REASON_TO_CODE = {
    "invalid_record": "INV",
    "robot_fell": "FALL",
    "stop_collision": "COL",
    "stop_margin_low": "LOW",
    "speed_track_error": "SPD",
    "startup_joint_error": "JNT",
    "stop_joint_error": "JNT",
}

#: `summarize_tow` 的 `valid/failures` 是**绳语义**判据（要求 tow 段出现过 `T > 0`）；
#: rigid 球铰连杆的张力有符号（压缩为负），因此 rigid case 可能报
#: `rope_never_taut_during_tow` 之类的失败 —— 那不代表工况失败。
#: 本脚本自己的判定（`verdict`）不依赖 `summarize_tow` 的 `valid`，只借用它的几何间隙。
SUMMARIZE_TOW_NOTE = (
    "summarize_tow.valid 是绳语义判据（要求 tow 段出现过 T>0）；rigid 连杆张力有符号，"
    "其失败项（如 rope_never_taut_during_tow）不代表工况失败。本脚本的 verdict 不依赖它。")

DEFAULT_THRESHOLDS = {
    # 关节跟踪误差（rad）：起步与停车两段的 RMS 上限、单关节绝对值上限。
    # 初值是工程占位（AMP 稳态跟踪误差量级 0.05 rad，加载后到 0.2 rad 量级都会超限），
    # 没有标定过；跑完首轮后应按实测分布重新设定并写进文档。
    "joint_rms_limit_rad": 0.10,
    "joint_max_limit_rad": 0.30,
    # 跟速：|vx_b − cmd| 的全程 MAE 相对指令的比例上限。
    "speed_mae_ratio_limit": 0.20,
    # 停车余量：车头几何间隙低于此值算「余量不足」（未接触）。
    "gap_margin_limit_m": 0.10,
    # 跌倒：base 高度低于此值，或 |pitch| 超限的样本占比超过 pitch_fraction_limit。
    "fall_height_limit_m": 0.15,
    "pitch_limit_rad": 0.80,
    "pitch_fraction_limit": 0.20,
}


# ---------------------------------------------------------------- 纯逻辑（离线可测）


@dataclass(frozen=True)
class TestCase:
    """一个 case：坡度 + 速度档 + 连接类型 + 小车质量。"""

    slope_deg: float
    velocity_mps: float
    connection: str
    cart_mass_kg: float

    @property
    def slug(self) -> str:
        """文件名安全的短标识（`+`/`.` 都合法，只有 `/` 需要避开）。"""
        return (f"slope{self.slope_deg:+g}_v{self.velocity_mps:g}_"
                f"{self.connection}_m{self.cart_mass_kg:g}kg")

    def to_dict(self) -> dict:
        return {"slope_deg": self.slope_deg, "velocity_mps": self.velocity_mps,
                "connection": self.connection, "cart_mass_kg": self.cart_mass_kg}


@dataclass(frozen=True)
class PhaseSchedule:
    """station / tow / coast 三段的步数划分（纯算术）。"""

    station_steps: int
    tow_steps: int
    coast_steps: int
    dt: float

    @property
    def total_steps(self) -> int:
        return self.station_steps + self.tow_steps + self.coast_steps

    def phase_of(self, step: int) -> str:
        if step < self.station_steps:
            return "station"
        return "tow" if step - self.station_steps < self.tow_steps else "coast"

    def step_in_phase(self, step: int) -> int:
        if step < self.station_steps:
            return step
        if step - self.station_steps < self.tow_steps:
            return step - self.station_steps
        return step - self.station_steps - self.tow_steps

    def to_dict(self) -> dict:
        return {"station_steps": self.station_steps, "tow_steps": self.tow_steps,
                "coast_steps": self.coast_steps, "dt_s": self.dt,
                "station_s": self.station_steps * self.dt,
                "tow_s": self.tow_steps * self.dt,
                "coast_s": self.coast_steps * self.dt,
                "total_s": self.total_steps * self.dt}


def slope_gravity(slope_deg: float, g: float = GRAVITY_MPS2) -> tuple:
    """坡度对应的重力向量（世界系）。

    约定 `+slope_deg` = 沿 **+x 上坡**（机器人在前、小车在后，两者都朝 +x）。
    上坡要求重力在 −x 方向有分量，因此 `gx = −g·sinθ`、`gz = −g·cosθ`。
    """
    if not math.isfinite(slope_deg) or abs(slope_deg) > 45.0:
        raise ValueError(f"坡度必须是 [-45, 45] 内的有限值，收到 {slope_deg!r}")
    if not math.isfinite(g) or g <= 0:
        raise ValueError(f"重力加速度必须是有限正数，收到 {g!r}")
    angle = math.radians(slope_deg)
    return (-g * math.sin(angle), 0.0, -g * math.cos(angle))


def slope_ground_plan(slope_deg: float, backend: str) -> dict:
    """坡度 → 场景设置方案（现有平地 + 倾斜重力 / 将来的坡面地形）。"""
    if backend == "terrain":
        raise NotImplementedError(
            "坡度地形（--slope-backend terrain）在仓库里还没有实现：本脚本默认用"
            " gravity 后端（现有平地 + 旋转重力，与真实坡面物理等价）。"
            "要接地形：改 slope_ground_plan() 返回地面 spawn 配置，并让场景按坡度摆出坡面与出生点。")
    if backend != "gravity":
        raise ValueError(f"未知的坡度后端 {backend!r}；可选 gravity / terrain")
    gravity = slope_gravity(slope_deg)
    return {"backend": backend, "gravity_mps2": gravity,
            "note": ("现有平地 + 旋转重力：随坡面倾斜的参考系里与真实坡度同解"
                     "（法向 mg·cosθ、下滑 mg·sinθ 逐项一致）")}


def build_case_grid(velocities, connections, cart_masses, slopes) -> list:
    """完整网格，顺序固定：坡度 → 速度 → 连接 → 质量（坡度在外层 = 仿真过程顺序）。"""
    return [TestCase(slope_deg=float(slope), velocity_mps=float(velocity),
                     connection=str(connection), cart_mass_kg=float(mass))
            for slope in slopes for velocity in velocities
            for connection in connections for mass in cart_masses]


def cases_for_slope(cases, slope_deg: float) -> list:
    """取出某个坡度的全部 case（保持网格顺序）。"""
    return [case for case in cases if case.slope_deg == slope_deg]


def make_schedule(*, settle_steps: int, tow_duration: float, coast_duration: float,
                  dt: float) -> PhaseSchedule:
    """由时长算出阶段划分；与测量台 `tow_drag.py` 的 station/tow/coast 语义一致。"""
    if settle_steps < 0:
        raise ValueError("station 步数不能为负")
    if not all(math.isfinite(v) and v > 0 for v in (tow_duration, coast_duration, dt)):
        raise ValueError("tow/coast 时长与 dt 必须是有限正数")
    tow_steps = int(round(tow_duration / dt))
    coast_steps = int(round(coast_duration / dt))
    if tow_steps <= 0 or coast_steps <= 0:
        raise ValueError("tow/coast 阶段的步数必须为正")
    if not math.isclose(tow_steps * dt, tow_duration, rel_tol=1e-6):
        raise ValueError(f"tow 时长 {tow_duration} 不是 dt={dt} 的整数倍")
    if not math.isclose(coast_steps * dt, coast_duration, rel_tol=1e-6):
        raise ValueError(f"coast 时长 {coast_duration} 不是 dt={dt} 的整数倍")
    return PhaseSchedule(station_steps=settle_steps, tow_steps=tow_steps,
                         coast_steps=coast_steps, dt=dt)


def shaped_command(*, phase: str, step_in_phase: int, velocity: float,
                   shaping: str = "direct", ramp_time_s: float = 1.0,
                   dt: float = 0.005) -> float:
    """脚本指令：station/coast 恒 0，tow 段按 shaping 给出速度指令。

    `direct` = 阶跃（本测试默认，与 `tow_drag.py` 的 `user_cmd` 同相位）；
    `ramp` = 在 `ramp_time_s` 内线性升到目标（「Fixed Ramp」对照基线）。
    """
    if phase != "tow":
        return 0.0
    if shaping == "direct":
        return float(velocity)
    if shaping != "ramp":
        raise ValueError(f"未知的指令整形 {shaping!r}；可选 direct / ramp")
    if ramp_time_s <= 0:
        return float(velocity)
    fraction = min(1.0, (step_in_phase + 1) * dt / ramp_time_s)
    return float(velocity) * fraction


def min_env_spacing(max_velocity_mps: float, *, tow_duration: float, coast_duration: float,
                    margin_m: float = 2.0) -> float:
    """并行环境的最小间距：机器人最长行程 + 小车滑行 + 余量。

    机器人只在 tow 段被指令驱动（coast 段指令为 0，但它仍可能被撞着走一点），
    所以行程上界取 `v·tow + v_robot_coast·coast` 里的保守近似：`v·(tow + 1.0) + margin`；
    小车滑行最多再叠 `v·coast` 的一半（黏性衰减，实测远小于线性）。这里取保守值。
    """
    if max_velocity_mps <= 0:
        return margin_m
    robot_travel = max_velocity_mps * (tow_duration + 1.0)
    cart_travel = 0.5 * max_velocity_mps * coast_duration
    return robot_travel + cart_travel + margin_m


def phase_rows(rows, phase: str) -> list:
    """取某一阶段的记录（`phase` 列）。"""
    return [row for row in rows if row["phase"] == phase]


def _col(rows, name: str) -> list:
    return [float(row[name]) for row in rows]


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def _rms(values) -> float:
    values = list(values)
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else float("nan")


def _percentile(values, fraction: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    index = min(len(values) - 1, max(0, int(round(fraction * (len(values) - 1)))))
    return values[index]


def _nan_summary(reason: str) -> dict:
    return {"available": False, "note": reason}


def joint_track_stats(rows, *, joint_names=None, torque_limits=None,
                      saturation_frac: float = 0.95) -> dict:
    """一段记录里 12 关节的位置跟踪误差 `q − q*` 统计（`q*` = 当拍下发的关节目标）。

    这是「关节响应误差」的口径：上层任务的残差动作正是加在 `q*` 上，
    所以 `q − q*` 就是上层能修正的那部分偏差（包含负载把腿压塌的量）。
    """
    if not rows:
        return _nan_summary("该窗口没有记录")
    names = list(joint_names) if joint_names else list(_JOINT_NAMES_FALLBACK)
    if len(names) != N_JOINTS:
        raise ValueError(f"关节名应为 {N_JOINTS} 个，收到 {len(names)}")
    limits = list(torque_limits) if torque_limits is not None else [23.7] * N_JOINTS
    if len(limits) != N_JOINTS:
        raise ValueError(f"力矩上限应为 {N_JOINTS} 个，收到 {len(limits)}")

    sumsq = [0.0] * N_JOINTS
    peak = [0.0] * N_JOINTS
    saturated = 0
    samples = 0
    for row in rows:
        for joint in range(N_JOINTS):
            error = float(row[f"robot_jp_{joint:02d}"]) - float(row[f"robot_jt_{joint:02d}"])
            sumsq[joint] += error * error
            peak[joint] = max(peak[joint], abs(error))
            if abs(float(row[f"robot_tau_{joint:02d}"])) > saturation_frac * limits[joint]:
                saturated += 1
        samples += 1
    per_joint_rms = [math.sqrt(total / samples) for total in sumsq]
    worst = max(range(N_JOINTS), key=lambda index: per_joint_rms[index])
    return {
        "available": True,
        "samples": samples,
        "joint_rms_rad": math.sqrt(sum(sumsq) / (samples * N_JOINTS)),
        "joint_max_rad": max(peak),
        "worst_joint": names[worst],
        "worst_joint_rms_rad": per_joint_rms[worst],
        "per_joint_rms_rad": {name: value for name, value in zip(names, per_joint_rms)},
        "torque_saturated_frac": saturated / (samples * N_JOINTS),
    }


def speed_track_stats(rows, command_mps: float, *, steady_fraction: float = 0.5) -> dict:
    """跟速误差：体系 x 速度与指令之差（tow 段全程 + 后半段稳态窗）。"""
    if not rows:
        return _nan_summary("tow 段没有记录")
    if not math.isfinite(command_mps) or command_mps <= 0:
        raise ValueError("速度指令必须是有限正数")
    errors = [float(row["robot_vx_b_mps"]) - command_mps for row in rows]
    speeds = _col(rows, "robot_vx_b_mps")
    start = int(len(rows) * (1.0 - steady_fraction))
    steady = errors[start:] or errors
    return {
        "available": True,
        "samples": len(rows),
        "mae_mps": _mean(abs(value) for value in errors),
        "rmse_mps": _rms(errors),
        "bias_mps": _mean(errors),
        "p95_abs_err_mps": _percentile([abs(value) for value in errors], 0.95),
        "max_abs_err_mps": max(abs(value) for value in errors),
        "ratio_mean": _mean(speeds) / command_mps,
        "frac_within_10pct": sum(1 for value in errors if abs(value) <= 0.1 * command_mps) / len(errors),
        "steady_mae_mps": _mean(abs(value) for value in steady),
        "steady_ratio_mean": _mean(_col(rows[start:] or rows, "robot_vx_b_mps")) / command_mps,
    }


def contact_witness(rows, *, record_dt: float, deck_limit_n: float = 1.0,
                    dv_limit_at_5ms: float = 0.015) -> dict:
    """追尾的三路见证（与 `summarize_tow.py` 同口径，但对记录步长做了标定）。

    `summarize_tow` 的 `CONTACT_LOAD_DV_LIMIT_MPS = 0.015` 是按 5 ms 记录定的；
    `--record-every > 1` 时单步速度跃变自然变大，所以这里把阈值按 `record_dt/5ms` 放大，
    避免采样变粗就误报接触。车斗接触力与几何间隙两路与采样无关。
    """
    if not rows:
        return {"available": False, "contact": None, "channels": []}
    channels = []
    peak_deck = max(abs(float(row["cart_deck_fx_n"])) for row in rows)
    if peak_deck > deck_limit_n:
        channels.append("deck_contact_force")
    limit = dv_limit_at_5ms * max(1.0, record_dt / 0.005)
    max_dv = max((abs(float(b["load_vx_mps"]) - float(a["load_vx_mps"]))
                  for a, b in zip(rows, rows[1:])), default=0.0)
    if max_dv > limit:
        channels.append("load_velocity_jump")
    return {"available": True, "contact": bool(channels), "channels": channels,
            "deck_contact_peak_n": peak_deck, "max_load_dv_mps": max_dv,
            "load_dv_limit_mps": limit}


def stop_stats(rows, *, record_dt: float, tow_summary: dict, command_mps: float,
               transition_window_s: float, gap_margin_limit_m: float,
               joint_names=None, torque_limits=None) -> dict:
    """停车段指标：小车滑移距离、间距维持、停车瞬态的关节响应。

    滑移距离用**世界系 x 位移**（坡度方向就是 x），与测量台的 `coast_load_travel_m` 同口径；
    间距维持优先用车头几何间隙（FK，来自 `summarize_tow`），没有时退回挂点距。
    """
    coast = phase_rows(rows, "coast")
    if not coast:
        return _nan_summary("coast 段没有记录")
    stop_speed = float(coast[0]["load_vx_mps"])
    start_x = float(coast[0]["load_x_m"])
    total = float(coast[-1]["load_x_m"]) - start_x
    to_rest = None
    time_to_rest = None
    for row in coast:
        if abs(float(row["load_vx_mps"])) < 0.02:
            to_rest = float(row["load_x_m"]) - start_x
            time_to_rest = float(row["time_s"]) - float(coast[0]["time_s"])
            break
    gaps = _col(coast, "rope_distance_m")
    window_steps = max(1, int(round(transition_window_s / record_dt)))
    # 追尾见证只看 **coast 段**：拖曳之前的接触是「生成/站定就贴上」，另一类问题。
    witness = contact_witness(coast, record_dt=record_dt)

    def summarize_tow_key(name, default=None):
        value = tow_summary.get(name, default)
        return value

    min_clearance = summarize_tow_key("min_clearance_coast_m")
    time_to_contact = summarize_tow_key("time_to_contact_after_stop_s")
    # 停车瞬态：指令归零后多久机器人真正停下来（|vx| < 0.05 m/s 并保持到窗口结束）
    settle_time = None
    for index, row in enumerate(coast[:window_steps]):
        if all(abs(float(later["robot_vx_mps"])) < 0.05 for later in coast[index:window_steps]):
            settle_time = float(row["time_s"]) - float(coast[0]["time_s"])
            break
    stop_joint = joint_track_stats(coast[:window_steps], joint_names=joint_names,
                                   torque_limits=torque_limits)
    contact = bool(witness["contact"]) if witness["contact"] is not None else None
    if min_clearance is not None and float(min_clearance) <= 0.0:
        contact = True
    margin_low = (min_clearance is not None and 0.0 < float(min_clearance) <= gap_margin_limit_m)
    return {
        "available": True,
        "cart_speed_at_stop_mps": stop_speed,
        "cart_coast_distance_m": total,
        "cart_coast_to_rest_m": to_rest,
        "cart_coast_time_to_rest_s": time_to_rest,
        "robot_travel_after_stop_m": float(coast[-1]["robot_x_m"]) - float(coast[0]["robot_x_m"]),
        "gap_at_stop_m": gaps[0],
        "min_gap_after_stop_m": min(gaps),
        "final_gap_m": gaps[-1],
        "clearance_at_stop_m": summarize_tow_key("clearance_at_stop_m"),
        "min_clearance_coast_m": min_clearance,
        "final_clearance_m": summarize_tow_key("final_clearance_m"),
        "time_to_contact_after_stop_s": time_to_contact,
        "contact": contact,
        "contact_channels": witness.get("channels", []),
        "deck_contact_peak_n": witness.get("deck_contact_peak_n"),
        "gap_margin_low": margin_low,
        "final_robot_vx_mps": float(coast[-1]["robot_vx_mps"]),
        "settle_time_s": settle_time,
        "joint_rms_rad": stop_joint.get("joint_rms_rad"),
        "joint_max_rad": stop_joint.get("joint_max_rad"),
        "worst_joint": stop_joint.get("worst_joint"),
        "torque_saturated_frac": stop_joint.get("torque_saturated_frac"),
        "per_joint_rms_rad": stop_joint.get("per_joint_rms_rad"),
        "body_vx_rms_mps": _rms(_col(coast[:window_steps], "robot_vx_b_mps")),
    }


def stability_stats(rows, *, transition_window_s: float, record_dt: float,
                    fall_height_limit_m: float, pitch_limit_rad: float,
                    pitch_fraction_limit: float) -> dict:
    """稳定性与记录有效性：跌倒判定、最大俯仰、是否出现非有限值。"""
    if not rows:
        return _nan_summary("没有记录")
    numeric_columns = [name for name in TEST_FIELDS
                       if name not in ("phase", "connection")]
    invalid_samples = 0
    for row in rows:
        for name in numeric_columns:
            if not math.isfinite(float(row[name])):
                invalid_samples += 1
                break
    heights = _col(rows, "robot_z_m")
    pitches = [abs(float(row["body_pitch_rad"])) for row in rows]
    pitch_fraction = sum(1 for value in pitches if value > pitch_limit_rad) / len(pitches)
    fell = (min(heights) < fall_height_limit_m) or (pitch_fraction > pitch_fraction_limit)
    fallen_phase = None
    if fell:
        for row in rows:
            if float(row["robot_z_m"]) < fall_height_limit_m or \
                    abs(float(row["body_pitch_rad"])) > pitch_limit_rad:
                fallen_phase = row["phase"]
                break
    return {
        "available": True,
        "invalid_samples": invalid_samples,
        "invalid": invalid_samples > 0,
        "min_robot_z_m": min(heights),
        "final_robot_z_m": heights[-1],
        "max_abs_pitch_rad": max(pitches),
        "final_abs_pitch_rad": pitches[-1],
        "pitch_over_limit_frac": pitch_fraction,
        "fell": bool(fell),
        "fell_phase": fallen_phase,
    }


def compute_case_metrics(rows, *, command_mps: float, slope_deg: float, connection: str,
                         cart_mass_kg: float, schedule: PhaseSchedule, record_dt: float,
                         tow_summary: dict, thresholds: dict, joint_names=None,
                         torque_limits=None, transition_window_s: float = 1.0) -> dict:
    """由逐物理步记录算出五项指标 + 判定。与仿真无关，可离线用合成轨迹复核。"""
    station = phase_rows(rows, "station")
    tow = phase_rows(rows, "tow")
    coast = phase_rows(rows, "coast")
    window_steps = max(1, int(round(transition_window_s / record_dt)))
    startup_rows = tow[:window_steps]
    startup_joint = joint_track_stats(startup_rows, joint_names=joint_names,
                                      torque_limits=torque_limits)
    startup = {
        "window_s": transition_window_s,
        "samples": len(startup_rows),
        "joint_rms_rad": startup_joint.get("joint_rms_rad"),
        "joint_max_rad": startup_joint.get("joint_max_rad"),
        "worst_joint": startup_joint.get("worst_joint"),
        "per_joint_rms_rad": startup_joint.get("per_joint_rms_rad"),
        "torque_saturated_frac": startup_joint.get("torque_saturated_frac"),
        # 起步的**速度**响应：体系 vx 的 RMS 误差、90% 到位时间、过冲
        "body_vx_rms_err_mps": _rms([float(row["robot_vx_b_mps"]) - command_mps
                                     for row in startup_rows]) if startup_rows else None,
        "time_to_90pct_s": _time_to_fraction(tow, command_mps, fraction=0.9, hold_s=0.25),
        "overshoot_ratio": (
            max((float(row["robot_vx_b_mps"]) for row in startup_rows), default=float("nan"))
            / command_mps - 1.0 if startup_rows else None),
    }
    speed = speed_track_stats(tow, command_mps)
    stop = stop_stats(rows, record_dt=record_dt, tow_summary=tow_summary,
                      command_mps=command_mps, transition_window_s=transition_window_s,
                      gap_margin_limit_m=thresholds["gap_margin_limit_m"],
                      joint_names=joint_names, torque_limits=torque_limits)
    stability = stability_stats(rows, transition_window_s=transition_window_s,
                                record_dt=record_dt,
                                fall_height_limit_m=thresholds["fall_height_limit_m"],
                                pitch_limit_rad=thresholds["pitch_limit_rad"],
                                pitch_fraction_limit=thresholds["pitch_fraction_limit"])
    metrics = {
        "case": {"slope_deg": slope_deg, "velocity_mps": command_mps,
                 "connection": connection, "cart_mass_kg": cart_mass_kg},
        "schedule": schedule.to_dict(),
        "samples": {"station": len(station), "tow": len(tow), "coast": len(coast)},
        "startup": startup,
        "speed": speed,
        "stop": stop,
        "stability": stability,
    }
    metrics["verdict"] = classify_case(metrics, thresholds)
    return metrics


def _time_to_fraction(rows, command_mps: float, *, fraction: float, hold_s: float) -> float:
    """tow 段开始后，|vx_b − cmd| ≤ (1−fraction)·cmd 并保持 `hold_s` 的首个时刻。"""
    if not rows:
        return None
    tolerance = (1.0 - fraction) * command_mps
    hold = max(1, int(round(hold_s / max(1e-9, float(rows[1]["time_s"]) - float(rows[0]["time_s"]))))) \
        if len(rows) > 1 else 1
    start_time = float(rows[0]["time_s"])
    for index in range(len(rows)):
        window = rows[index:index + hold]
        if len(window) < hold:
            break
        if all(abs(float(row["robot_vx_b_mps"]) - command_mps) <= tolerance for row in window):
            return float(rows[index]["time_s"]) - start_time
    return None


def classify_case(metrics: dict, thresholds: dict) -> dict:
    """把五项指标折成判定码（`OK` 或最严重的一条）+ 全部原因。"""
    reasons = []
    stability = metrics.get("stability", {})
    stop = metrics.get("stop", {})
    speed = metrics.get("speed", {})
    startup = metrics.get("startup", {})
    if stability.get("invalid"):
        reasons.append("invalid_record")
    if stability.get("fell"):
        reasons.append("robot_fell")
    if stop.get("contact"):
        reasons.append("stop_collision")
    elif stop.get("gap_margin_low"):
        reasons.append("stop_margin_low")
    if speed.get("available") and speed.get("mae_mps") is not None and \
            speed["mae_mps"] > thresholds["speed_mae_ratio_limit"] * metrics["case"]["velocity_mps"]:
        reasons.append("speed_track_error")
    for key, limit_key in (("startup", "joint_rms_limit_rad"), ("stop", "joint_rms_limit_rad")):
        section = metrics.get(key, {})
        value = section.get("joint_rms_rad")
        if value is not None and value > thresholds[limit_key]:
            reasons.append(f"{key}_joint_error")
    for key in ("startup", "stop"):
        section = metrics.get(key, {})
        value = section.get("joint_max_rad")
        if value is not None and value > thresholds["joint_max_limit_rad"]:
            reason = f"{key}_joint_error"
            if reason not in reasons:
                reasons.append(reason)
    codes = [_REASON_TO_CODE[reason] for reason in reasons]
    code = max(codes, key=lambda item: VERDICT_SEVERITY[item]) if codes else "OK"
    return {"code": code, "reasons": reasons, "severity": VERDICT_SEVERITY[code]}


# ---------------------------------------------------------------- 报告


def case_report_row(metrics: dict) -> dict:
    """逐 case 的扁平表行（report.csv 用）。"""
    case = metrics["case"]
    startup, speed, stop = metrics["startup"], metrics["speed"], metrics["stop"]
    stability = metrics["stability"]
    verdict = metrics["verdict"]
    return {
        "slope_deg": case["slope_deg"], "velocity_mps": case["velocity_mps"],
        "connection": case["connection"], "cart_mass_kg": case["cart_mass_kg"],
        "verdict": verdict["code"], "reasons": "|".join(verdict["reasons"]),
        "startup_joint_rms_rad": startup.get("joint_rms_rad"),
        "startup_joint_max_rad": startup.get("joint_max_rad"),
        "startup_worst_joint": startup.get("worst_joint"),
        "startup_torque_sat_frac": startup.get("torque_saturated_frac"),
        "startup_vx_rms_err_mps": startup.get("body_vx_rms_err_mps"),
        "startup_time_to_90pct_s": startup.get("time_to_90pct_s"),
        "speed_mae_mps": speed.get("mae_mps"),
        "speed_rmse_mps": speed.get("rmse_mps"),
        "speed_ratio_mean": speed.get("ratio_mean"),
        "speed_steady_mae_mps": speed.get("steady_mae_mps"),
        "stop_cart_coast_distance_m": stop.get("cart_coast_distance_m"),
        "stop_cart_coast_to_rest_m": stop.get("cart_coast_to_rest_m"),
        "stop_cart_speed_at_stop_mps": stop.get("cart_speed_at_stop_mps"),
        "stop_min_clearance_coast_m": stop.get("min_clearance_coast_m"),
        "stop_final_clearance_m": stop.get("final_clearance_m"),
        "stop_time_to_contact_s": stop.get("time_to_contact_after_stop_s"),
        "stop_contact": stop.get("contact"),
        "stop_joint_rms_rad": stop.get("joint_rms_rad"),
        "stop_joint_max_rad": stop.get("joint_max_rad"),
        "stop_settle_time_s": stop.get("settle_time_s"),
        "stop_robot_travel_m": stop.get("robot_travel_after_stop_m"),
        "min_robot_z_m": stability.get("min_robot_z_m"),
        "max_abs_pitch_rad": stability.get("max_abs_pitch_rad"),
        "fell": stability.get("fell"),
        "summarize_tow_valid": metrics.get("summarize_tow", {}).get("valid"),
        "summarize_tow_failures": "|".join(metrics.get("summarize_tow", {}).get("failures", [])),
    }


def group_of(case: dict) -> str:
    """工况分组：平地 / 上坡 / 下坡。"""
    slope = float(case["slope_deg"])
    if slope == 0.0:
        return "flat"
    return "uphill" if slope > 0 else "downhill"


def group_statistics(case_summaries) -> dict:
    """按 平地/上坡/下坡 汇总判定与失败模式计数。"""
    groups = {}
    for name in ("flat", "uphill", "downhill"):
        groups[name] = {"cases": 0, "verdicts": {}, "reasons": {}, "worst_cases": []}
    for summary in case_summaries:
        metrics = summary["metrics"]
        group = group_of(metrics["case"])
        entry = groups[group]
        entry["cases"] += 1
        code = metrics["verdict"]["code"]
        entry["verdicts"][code] = entry["verdicts"].get(code, 0) + 1
        for reason in metrics["verdict"]["reasons"]:
            entry["reasons"][reason] = entry["reasons"].get(reason, 0) + 1
        entry["worst_cases"].append({
            "case": metrics["case"], "code": code, "reasons": metrics["verdict"]["reasons"],
            "speed_mae_mps": metrics["speed"].get("mae_mps"),
            "startup_joint_rms_rad": metrics["startup"].get("joint_rms_rad"),
            "stop_joint_rms_rad": metrics["stop"].get("joint_rms_rad"),
            "min_clearance_coast_m": metrics["stop"].get("min_clearance_coast_m"),
            "cart_coast_distance_m": metrics["stop"].get("cart_coast_distance_m"),
        })
    for entry in groups.values():
        entry["cases_ok"] = entry["verdicts"].get("OK", 0)
        entry["worst_cases"].sort(
            key=lambda item: (-VERDICT_SEVERITY[item["code"]],
                              -(item["speed_mae_mps"] or 0.0)))
        entry["worst_cases"] = entry["worst_cases"][:5]
    return groups


def necessity_conclusion(groups: dict, thresholds: dict) -> dict:
    """由分组统计给出「任务是否有必要」的判读（只依据本网格 + 本阈值）。"""
    lines = []
    verdict = {}
    for name, label in (("flat", "平地"), ("uphill", "上坡"), ("downhill", "下坡")):
        entry = groups[name]
        if entry["cases"] == 0:
            verdict[name] = "no_cases"
            lines.append(f"- {label}：本网格没有该组 case。")
            continue
        ok, total = entry["cases_ok"], entry["cases"]
        reasons = ", ".join(f"{reason}×{count}"
                            for reason, count in sorted(entry["reasons"].items(),
                                                        key=lambda item: -item[1])) or "无"
        if ok == total:
            verdict[name] = "baseline_sufficient"
            lines.append(f"- {label}：{ok}/{total} 全部通过阈值（关节 RMS ≤ "
                         f"{thresholds['joint_rms_limit_rad']:g} rad、跟速 MAE ≤ "
                         f"{thresholds['speed_mae_ratio_limit']*100:g}% 指令、停车间隙 > "
                         f"{thresholds['gap_margin_limit_m']:g} m、无接触/跌倒）"
                         f"⇒ 该组不构成上层任务必要性的证据。")
        else:
            verdict[name] = "baseline_insufficient"
            failure_kinds = [reason for reason in entry["reasons"]
                             if reason != "invalid_record"]
            stop_dominated = all(reason in ("stop_collision", "stop_margin_low", "stop_joint_error")
                                 for reason in failure_kinds) and bool(failure_kinds)
            hint = ("失败集中在**停车段**：上层任务的必要性主要来自停车时序与间隙维持，"
                    "需用 `--command-shaping ramp` 或更早的 STOP 调度做对照，判断纯脚本 shaping "
                    "是否已经够用。" if stop_dominated else
                    "失败跨起步/全程/停车多相：脚本 shaping 不足以解释，属于上层残差/调度的目标。")
            lines.append(f"- {label}：{ok}/{total} 通过；失败模式 {reasons}。{hint}")
    return {"verdict": verdict, "lines": lines}


def format_verdict_matrix(case_summaries, *, slopes) -> str:
    """逐坡度打印「质量 × (速度/连接)」判定矩阵（人读报告与终端共用）。"""
    blocks = []
    for slope in slopes:
        cases = [summary["metrics"] for summary in case_summaries
                 if summary["metrics"]["case"]["slope_deg"] == slope]
        if not cases:
            continue
        velocities = sorted({case["case"]["velocity_mps"] for case in cases})
        connections = [name for name in CONNECTIONS
                       if any(case["case"]["connection"] == name for case in cases)]
        masses = sorted({case["case"]["cart_mass_kg"] for case in cases})
        header = "| 质量 kg | " + " | ".join(
            f"{velocity:g} m/s {connection}" for velocity in velocities
            for connection in connections) + " |"
        divider = "|" + "---|" * (len(velocities) * len(connections) + 1)
        lines = [f"### 坡度 {slope:+g}°", "", header, divider]
        index = {(case["case"]["velocity_mps"], case["case"]["connection"],
                  case["case"]["cart_mass_kg"]): case for case in cases}
        for mass in masses:
            cells = []
            for velocity in velocities:
                for connection in connections:
                    case = index.get((velocity, connection, mass))
                    cells.append(case["verdict"]["code"] if case else "-")
            lines.append(f"| {mass:g} | " + " | ".join(cells) + " |")
        lines.append("")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def build_markdown_report(*, case_summaries, groups, conclusion, args_dict, thresholds,
                          schedule, slopes, git) -> str:
    """人读报告：配置、判定矩阵、失败模式、结论、限制。"""
    lines = ["# 拖曳上层任务必要性：冻结策略基线测试", "",
             f"- 生成时间：{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
             f"- git：`{git.get('commit')}`（worktree {git.get('working_tree') or 'clean'}）",
             f"- 速度档：{list(args_dict['velocities'])} m/s；连接：{list(args_dict['connections'])}；"
             f"质量：{list(args_dict['cart_masses'])} kg",
             f"- 坡度：{list(slopes)} deg（+ = 沿 +x 上坡）；地面摩擦固定 "
             f"{args_dict['ground_friction']:g}",
             f"- 指令整形：{args_dict['command_shaping']}"
             + (f"（ramp {args_dict['ramp_time_s']:g} s）" if args_dict["command_shaping"] == "ramp" else "")
             + f"；坡度后端：{args_dict['slope_backend']}；站定驻车制动：{args_dict['slope_settle']}",
             f"- 每 case：station {schedule.station_steps * schedule.dt:.2f} s + tow "
             f"{schedule.tow_steps * schedule.dt:.2f} s + coast "
             f"{schedule.coast_steps * schedule.dt:.2f} s，共 {schedule.total_steps} 物理步 "
             f"（dt={schedule.dt:g} s，记录每 {args_dict['record_every']} 步一行）",
             f"- 阈值：关节 RMS ≤ {thresholds['joint_rms_limit_rad']:g} rad / 单关节 ≤ "
             f"{thresholds['joint_max_limit_rad']:g} rad、跟速 MAE ≤ "
             f"{thresholds['speed_mae_ratio_limit'] * 100:g}% 指令、停车几何间隙 > "
             f"{thresholds['gap_margin_limit_m']:g} m、跌倒 base 高 < "
             f"{thresholds['fall_height_limit_m']:g} m 或 |pitch| > "
             f"{thresholds['pitch_limit_rad']:g} rad 的样本 > "
             f"{thresholds['pitch_fraction_limit'] * 100:g}%",
             "", "判定码：`OK` 通过；`LOW` 停车余量低；`JNT` 关节响应超限；`SPD` 跟速超限；"
             "`COL` 追尾接触；`FALL` 跌倒；`INV` 记录不可用。", "",
             "## 逐坡度判定矩阵", "", format_verdict_matrix(case_summaries, slopes=slopes), "",
             "## 分组统计", ""]
    for name, label in (("flat", "平地"), ("uphill", "上坡"), ("downhill", "下坡")):
        entry = groups[name]
        if entry["cases"] == 0:
            continue
        reasons = ", ".join(f"{reason}×{count}" for reason, count in
                            sorted(entry["reasons"].items(), key=lambda item: -item[1])) or "无"
        lines.append(f"- **{label}**：{entry['cases_ok']}/{entry['cases']} 通过；失败模式：{reasons}")
    lines += ["", "## 结论（任务是否有必要）", ""]
    lines += conclusion["lines"]
    lines += ["", "## 限制", "",
              "- 只覆盖本网格与本阈值：未测弹性绳另外 3 档 k/c、未测轮阻档、未测 breakaway/"
              "Coulomb 阻力、未测真机、未做跨环境隔离。",
              "- 跟速一律用**体系** vx（与冻结策略观测同口径）；`summarize_tow` 的 "
              "`steady_tracking_ratio` 是世界系口径，坡上不要混用。",
              "- 坡度用「现有平地 + 旋转重力」实现（与真实坡面同解）；坡度地形资产尚未接入。",
              "- 关节跟踪误差阈值没有标定，首轮结果出来前不要把 `JNT` 当成定论。",
              f"- {SUMMARIZE_TOW_NOTE}",
              "- 仿真相位（PhysX 步进、绳力/轮阻施加、重力写入）只在训练机实跑验证；"
              "本脚本的离线测试只覆盖纯逻辑与记录字段契约。",
              ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- CLI


def _positive(value, name, *, allow_zero=False):
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise argparse.ArgumentTypeError(
            f"{name} 必须是有限{'非负' if allow_zero else '正'}数，收到 {value!r}")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="拖曳上层任务必要性：冻结 AMP 策略 + 脚本指令的网格基线测试",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--policy", default="amp", help="冻结底层策略名（见 policy_cfg.POLICIES）")
    parser.add_argument("--velocities", type=float, nargs="+", default=list(DEFAULT_VELOCITIES),
                        help="速度档（m/s，体系 x 指令）")
    parser.add_argument("--cart-masses", type=float, nargs="+", default=list(DEFAULT_CART_MASSES),
                        help="小车目标总质量档（kg，质量与惯量同比例缩放）")
    parser.add_argument("--connections", nargs="+", choices=list(CONNECTIONS),
                        default=list(CONNECTIONS), help="连接类型（三类）")
    parser.add_argument("--slopes", type=float, nargs="+", default=list(DEFAULT_SLOPES_DEG),
                        help="坡度（deg；+ = 沿 +x 上坡，0 = 平地）")
    parser.add_argument("--slope-backend", choices=("gravity", "terrain"), default="gravity",
                        help="坡度实现：gravity = 现有平地 + 旋转重力（默认，物理等价）；"
                             "terrain = 将来的坡面地形资产（尚未实现，会报错）")
    parser.add_argument("--slope-settle", choices=("hold", "free"), default="hold",
                        help="坡度站定段是否给小车加驻车制动（hold，起拖释放）/ 放任溜坡（free）")
    parser.add_argument("--hold-damping", type=float, default=5.0,
                        help="--slope-settle hold 时额外的轮轴黏性阻尼（N·m·s/rad，仅 station 段）")
    parser.add_argument("--ground-friction", type=float, default=FACTORY_FLOOR_FRICTION,
                        help="地面静/动摩擦系数（固定值；工厂地面常规 0.8）")
    parser.add_argument("--command-shaping", choices=("direct", "ramp"), default="direct",
                        help="速度指令整形：direct = 阶跃（默认）；ramp = 固定斜坡对照")
    parser.add_argument("--ramp-time-s", type=float, default=1.0, help="ramp 整形的上升时间（s）")
    # 负载与连接参数（与 tow_drag.py 同口径）
    parser.add_argument("--rope-length", type=float, default=0.8, help="绳 L0 / 连杆名义长度（m）")
    parser.add_argument("--slack", type=float, default=0.40, help="绳初始松弛量（m）")
    parser.add_argument("--stiffness", type=float, default=4000.0, help="弹性绳 k（N/m）")
    parser.add_argument("--damping", type=float, default=100.0, help="弹性绳 c（N·s/m）")
    parser.add_argument("--position-gain", type=float, default=0.2, help="inextensible/rigid 回拉增益")
    parser.add_argument("--max-correction-rate", type=float, default=0.2,
                        help="inextensible/rigid 回拉相对速度上限（m/s）")
    parser.add_argument("--wheel-damping", type=float, default=0.016,
                        help="轮轴黏性阻尼 b（N·m·s/rad，训练侧同值）")
    # 时序
    parser.add_argument("--settle-time", type=float, default=1.0, help="站定时长（s）")
    parser.add_argument("--tow-duration", type=float, default=5.0, help="拖曳时长（s）")
    parser.add_argument("--coast-duration", type=float, default=5.0, help="STOP 后滑行观测时长（s）")
    parser.add_argument("--transition-window", type=float, default=1.0,
                        help="起步/停车瞬态窗口（s），用于关节响应与速度响应统计")
    parser.add_argument("--dt", type=float, default=0.005, help="物理步长（s）")
    parser.add_argument("--record-every", type=int, default=1,
                        help="每 N 个物理步记一行（>1 会让接触的速度跃变见证失去 5 ms 标定，"
                             "脚本内部已按步长放大阈值）")
    # 场景
    parser.add_argument("--spawn-height", type=float, default=None, help="机器人初始高度（m）")
    parser.add_argument("--cart-drop", type=float, default=0.03, help="小车生成离地高度（m）")
    parser.add_argument("--env-spacing", type=float, default=16.0, help="并行环境间距（m）")
    parser.add_argument("--max-envs", type=int, default=128, help="单次仿真的最大环境数（保护）")
    # 输出
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="输出目录（默认 imgo2_rl/logs/towing/play_test/<时间戳>_<随机>）；绝不覆盖已有目录")
    parser.add_argument("--write-csv", choices=("failed", "all", "none"), default="failed",
                        help="原始逐物理步 CSV 的写入范围（full 网格全写约 0.5 GB）")
    parser.add_argument("--headless", action="store_true", help="无显示运行")
    parser.add_argument("--device", default="cuda:0", help="仿真设备")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印将执行的网格与命令，不启动 Isaac Sim（标准库即可运行）")
    # 阈值
    for name, value in DEFAULT_THRESHOLDS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=value,
                            dest=name, help=f"判定阈值（默认 {value}）")
    args = parser.parse_args(argv)

    if not args.velocities:
        parser.error("--velocities 不能为空")
    for velocity in args.velocities:
        if not math.isfinite(velocity) or not 0.0 < velocity <= 2.0:
            parser.error(f"--velocities 必须在 (0, 2] m/s 内，收到 {velocity!r}")
    if not args.cart_masses:
        parser.error("--cart-masses 不能为空")
    for mass in args.cart_masses:
        if not math.isfinite(mass) or not 2.0 <= mass <= 50.0:
            parser.error(f"--cart-masses 必须在 [2, 50] kg 内，收到 {mass!r}")
    if not args.connections:
        parser.error("--connections 不能为空")
    if not args.slopes:
        parser.error("--slopes 不能为空")
    for slope in args.slopes:
        if not math.isfinite(slope) or abs(slope) > 45.0:
            parser.error(f"--slopes 必须在 [-45, 45] deg 内，收到 {slope!r}")
    _positive(args.ground_friction, "--ground-friction")
    if args.ground_friction > 2.0:
        parser.error("--ground-friction 必须在 (0, 2] 内")
    _positive(args.rope_length, "--rope-length")
    _positive(args.slack, "--slack", allow_zero=True)
    if args.rope_length - args.slack <= 0.05:
        parser.error("--rope-length − --slack 必须 > 0.05 m（否则初始挂点距几何上不成立）")
    _positive(args.stiffness, "--stiffness")
    _positive(args.damping, "--damping", allow_zero=True)
    for name in ("settle_time", "tow_duration", "coast_duration", "transition_window", "dt",
                 "ramp_time_s", "hold_damping", "wheel_damping"):
        value = getattr(args, name)
        allow_zero = name in ("wheel_damping", "hold_damping")
        _positive(value, f"--{name.replace('_', '-')}", allow_zero=allow_zero)
    if args.dt > 0.01:
        parser.error("--dt 必须 ≤ 0.01 s")
    if args.record_every < 1:
        parser.error("--record-every 必须 ≥ 1")
    if args.spawn_height is not None:
        _positive(args.spawn_height, "--spawn-height")
    if not 0.0 <= args.cart_drop <= 0.1:
        parser.error("--cart-drop 必须在 [0, 0.1] m 内")
    if not math.isfinite(args.env_spacing) or args.env_spacing <= 0:
        parser.error("--env-spacing 必须是有限正数")
    if args.max_envs < 1:
        parser.error("--max-envs 必须 ≥ 1")
    for name in DEFAULT_THRESHOLDS:
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须是有限正数")

    cases = build_case_grid(args.velocities, args.connections, args.cart_masses, args.slopes)
    per_slope = len(cases) // len(args.slopes)
    if per_slope > args.max_envs:
        parser.error(
            f"单个坡度的 case 数 {per_slope} 超过 --max-envs {args.max_envs}。"
            f"减小网格（--velocities/--connections/--cart-masses/--slopes）或提高 --max-envs。")
    required_spacing = min_env_spacing(max(args.velocities), tow_duration=args.tow_duration,
                                       coast_duration=args.coast_duration)
    if args.env_spacing < required_spacing:
        parser.error(
            f"--env-spacing {args.env_spacing:g} m 小于并行环境所需的最小间距 "
            f"{required_spacing:.1f} m（最大速度 {max(args.velocities):g} m/s × "
            f"最长行程）；调大 --env-spacing 或减小速度/时长")
    try:
        slope_ground_plan(args.slopes[0], args.slope_backend)
    except NotImplementedError as exc:
        parser.error(str(exc))
    args.cases = cases
    args.schedule = make_schedule(settle_steps=int(round(args.settle_time / args.dt)),
                                  tow_duration=args.tow_duration,
                                  coast_duration=args.coast_duration, dt=args.dt)
    return args


def planned_grid_lines(args) -> list:
    """`--dry-run` / 启动横幅用的网格与代价说明（纯逻辑）。"""
    schedule = args.schedule
    cases = args.cases
    per_slope = len(cases) // len(args.slopes)
    recorded = schedule.total_steps // args.record_every
    lines = [
        f"[plan] 冻结底层策略：{args.policy}；地面摩擦固定 {args.ground_friction:g}；"
        f"轮轴阻尼 {args.wheel_damping:g} N·m·s/rad",
        f"[plan] 速度档：{', '.join(f'{v:g}' for v in args.velocities)} m/s；连接："
        f"{', '.join(args.connections)}；质量：{', '.join(f'{m:g}' for m in args.cart_masses)} kg",
        f"[plan] 坡度：{', '.join(f'{s:+g}' for s in args.slopes)} deg（+ = 上坡）；后端 "
        f"{args.slope_backend}；站定制动 {args.slope_settle}"
        + (f"（额外 {args.hold_damping:g} N·m·s/rad，仅 station）"
           if args.slope_settle == "hold" else ""),
        f"[plan] 网格：{len(args.slopes)} 坡度 × {len(args.velocities)} 速度 × "
        f"{len(args.connections)} 连接 × {len(args.cart_masses)} 质量 = {len(cases)} case；"
        f"每坡度 {per_slope} 环境并行、{len(args.slopes)} 次仿真过程",
        f"[plan] 每 case：station {schedule.station_steps * schedule.dt:.2f} s"
        f"（{schedule.station_steps} 步）+ tow {schedule.tow_steps * schedule.dt:.2f} s"
        f"（{schedule.tow_steps} 步）+ coast {schedule.coast_steps * schedule.dt:.2f} s"
        f"（{schedule.coast_steps} 步）= {schedule.total_steps} 步 / "
        f"{schedule.total_steps * schedule.dt:.2f} s",
        f"[plan] 记录：每 {args.record_every} 物理步一行 ⇒ 每 case 约 {recorded} 行，"
        f"共约 {recorded * len(cases)} 行；CSV 策略 {args.write_csv}",
        f"[plan] 指令整形：{args.command_shaping}"
        + (f"（ramp {args.ramp_time_s:g} s）" if args.command_shaping == "ramp" else "（阶跃）")
        + f"；环境间距 {args.env_spacing:g} m",
    ]
    return lines


def git_info():
    def run(*command):
        try:
            result = subprocess.run(["git", *command], cwd=REPO, capture_output=True,
                                    text=True, encoding="utf-8", errors="replace", timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": run("rev-parse", "HEAD"), "working_tree": run("status", "--porcelain")}


# ---------------------------------------------------------------- 仿真


def main(args):
    """跑网格。仿真相关 import 全部放在这里，保证 `--dry-run`／`--help` 只用标准库。"""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output_dir or
              RL_ROOT / "logs/towing/play_test" / f"{stamp}_{uuid.uuid4().hex[:8]}")
    output = output.expanduser().resolve()

    git = git_info()
    thresholds = {name: getattr(args, name) for name in DEFAULT_THRESHOLDS}
    grid = [case.to_dict() for case in args.cases]
    experiment = {
        "state": "starting",
        "script": str(Path(__file__).resolve()),
        "question": "冻结 AMP 策略 + 脚本速度指令的基线，在速度×连接×质量×坡度网格上是否已满足"
                    "起步关节响应 / 全程跟速 / 停车滑移 / 停车间距 / 停车关节响应五项指标",
        "grid": {"velocities_mps": list(args.velocities),
                 "connections": list(args.connections),
                 "cart_masses_kg": list(args.cart_masses),
                 "slopes_deg": list(args.slopes),
                 "cases": grid, "case_count": len(grid),
                 "envs_per_pass": len(grid) // len(args.slopes)},
        "schedule": args.schedule.to_dict(),
        "thresholds": thresholds,
        "ground_friction": args.ground_friction,
        "wheel_damping_nms_per_rad": args.wheel_damping,
        "slope": {"backend": args.slope_backend, "settle": args.slope_settle,
                  "hold_damping_nms_per_rad": args.hold_damping,
                  "realisation_note": "现有平地 + 旋转重力（与真实坡面同解）"},
        "connection": {"rope_length_m": args.rope_length, "slack_m": args.slack,
                       "stiffness_n_per_m": args.stiffness, "damping_ns_per_m": args.damping,
                       "position_gain": args.position_gain,
                       "max_correction_rate_mps": args.max_correction_rate},
        "arguments": vars(args).copy() | {"cases": None, "schedule": None},
        "python": platform.python_version(),
        "git": git,
    }
    experiment["arguments"] = {key: (str(value) if isinstance(value, Path) else value)
                               for key, value in experiment["arguments"].items()}
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=False)

    print("\n".join(planned_grid_lines(args)), flush=True)
    print(f"[plan] 输出目录：{output}", flush=True)
    if args.dry_run:
        print("[plan] --dry-run：不启动 Isaac Sim。", flush=True)
        return 0

    def write_json(path: Path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_finite_only(data), indent=2, ensure_ascii=False,
                                   allow_nan=False, default=str) + "\n", encoding="utf-8")

    write_json(output / "experiment.json", experiment)
    summaries_dir = output / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    application = None
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(headless=args.headless, device=args.device)
        application = launcher.app

        import carb
        import torch
        import isaaclab.sim as sim_utils
        import isaaclab.utils.math as math_utils
        from isaaclab.scene import InteractiveScene
        from imgo2_rl.assets.cart import make_cart_cfg
        from imgo2_rl.tasks.manager_based.towing.mdp.resistance import viscous_resistance
        from imgo2_rl.tasks.manager_based.towing.mdp.rope import point_velocity
        from imgo2_rl.tasks.manager_based.towing.mdp.rope_model import (
            RIGID_MODEL, BodyProperties, MultiRopeModel, make_rope_model,
            world_inverse_inertia)
        from imgo2_rl.tasks.manager_based.towing.towing_env_cfg import (
            ROBOT_ATTACHMENT_OFFSET_M, ROBOT_SPAWN_HEIGHT_M, TowSceneCfg)
        from imgo2_rl.tasks.manager_based.towing.utils.low_level_policy import (
            FrozenLowLevelPolicy, parts_from_robot_state)
        from imgo2_rl.tasks.manager_based.towing.utils.policy_cfg import get_policy

        policy_cfg = get_policy(args.policy)
        spawn_height = args.spawn_height if args.spawn_height is not None else ROBOT_SPAWN_HEIGHT_M
        slope_cases = cases_for_slope(args.cases, args.slopes[0])
        num_envs = len(slope_cases)
        if num_envs != len(args.cases) // len(args.slopes):
            raise RuntimeError("坡度分组后的 case 数与网格不一致（内部错误）")

        cart_cfg, model = make_cart_cfg(output / "usd", drop_height=args.cart_drop)
        cart_attachment = tuple(model["attachment_position_m"])
        robot_attachment = tuple(ROBOT_ATTACHMENT_OFFSET_M)
        cart_cfg.init_state.pos = (
            _initial_cart_x(args.rope_length, args.slack, spawn_height=spawn_height,
                            cart_height=model["resting_height_m"],
                            robot_offset=robot_attachment, cart_offset=cart_attachment),
            0.0, model["resting_height_m"] + args.cart_drop)

        sim_cfg = sim_utils.SimulationCfg(
            dt=args.dt, device=args.device,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=args.ground_friction, dynamic_friction=args.ground_friction,
                restitution=0.0, friction_combine_mode="average", restitution_combine_mode="min"))
        sim = sim_utils.SimulationContext(sim_cfg)
        grid_side = max(1, math.ceil(math.sqrt(num_envs)))
        span = args.env_spacing * max(1, grid_side - 1)
        if num_envs > 1:
            centre = 0.5 * span
            sim.set_camera_view((centre + 1.6 * span, centre - 1.6 * span, 1.1 * span),
                                (centre, centre, 0.2))
        else:
            sim.set_camera_view((2.5, 2.5, 1.8), (-0.7, 0.0, 0.2))
        scene_cfg = TowSceneCfg(num_envs=num_envs, env_spacing=args.env_spacing, cart=cart_cfg)
        ground = scene_cfg.ground.spawn.physics_material
        ground.static_friction = args.ground_friction
        ground.dynamic_friction = args.ground_friction
        scene_cfg.robot.init_state.pos = (0.0, 0.0, spawn_height)
        scene_cfg.robot.init_state.joint_pos = dict(zip(policy_cfg.joint_names,
                                                        policy_cfg.default_dof_pos))
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        robot, cart, contacts = scene["robot"], scene["cart"], scene["wheel_contacts"]
        deck_contacts = scene["deck_contacts"]

        # ------------------------------------------------------------ 契约核对
        if robot.num_joints != policy_cfg.num_joints:
            raise RuntimeError(f"机器人有 {robot.num_joints} 个关节，契约要求 {policy_cfg.num_joints}")
        asset_perm = policy_cfg.asset_permutation(robot.joint_names)
        policy_to_asset = torch.tensor(asset_perm, dtype=torch.long, device=args.device)
        asset_to_policy = torch.empty_like(policy_to_asset)
        asset_to_policy[policy_to_asset] = torch.arange(policy_cfg.num_joints,
                                                        dtype=torch.long, device=args.device)
        reordered_default = [float(value) for value in robot.data.default_joint_pos[0, policy_to_asset]]
        if any(abs(a - b) > 1e-6 for a, b in zip(reordered_default, policy_cfg.default_dof_pos)):
            raise RuntimeError("关节置换核对失败：模型默认关节角与契约 default_dof_pos 不一致")
        base_ids, _ = robot.find_bodies(["base"])
        cart_base_ids, _ = cart.find_bodies(["base_link"])
        cart_joint_ids, _ = cart.find_joints(list(model["joint_names"]), preserve_order=True)
        cart_wheel_ids, _ = cart.find_bodies(list(model["wheel_names"]), preserve_order=True)
        if len(base_ids) != 1 or len(cart_base_ids) != 1 or len(cart_joint_ids) != 4:
            raise RuntimeError("base/base_link 或小车四轮关节的解析结果不符合预期")
        decimation = max(1, int(round(policy_cfg.control_dt / args.dt)))
        if not math.isclose(decimation * args.dt, policy_cfg.control_dt, rel_tol=1e-6):
            raise RuntimeError("policy_cfg.control_dt 必须是物理 dt 的整数倍")
        policy = FrozenLowLevelPolicy(policy_cfg, device=args.device)
        # 冻结策略的 reset 契约建议的站定步数（`--settle-time` 比它短时起拖前可能还没站定）
        recommended_settle = policy.reset()
        if args.schedule.station_steps < recommended_settle:
            print(f"[warn] station {args.schedule.station_steps} 步（{args.settle_time:g} s）"
                  f"短于冻结策略 reset 契约建议的 {recommended_settle} 步"
                  f"（{policy_cfg.reset_settle_s:g} s）：起拖前机器人可能还没站定，"
                  f"起步指标会被这段未站定的瞬态污染。", flush=True)
        nominal_masses = cart.root_physx_view.get_masses().clone()
        nominal_inertias = cart.root_physx_view.get_inertias().clone()
        dt = sim.get_physics_dt()
        if not math.isclose(dt, args.dt, rel_tol=1e-6):
            raise RuntimeError("Simulator dt 与请求不一致")
        record_dt = dt * args.record_every

        # ------------------------------------------------------------ 逐 env 的 case 参数
        connection_ids = torch.tensor([CONNECTIONS.index(case.connection) for case in slope_cases],
                                      dtype=torch.long, device=args.device)
        mass_scales = torch.tensor([case.cart_mass_kg / model["total_mass_kg"]
                                    for case in slope_cases],
                                   dtype=torch.float32, device=args.device)
        velocities = torch.tensor([case.velocity_mps for case in slope_cases],
                                  dtype=torch.float32, device=args.device)
        cart_env_idx = torch.arange(num_envs, dtype=torch.int, device="cpu")
        # `get_masses()/get_inertias()` 是 **CPU** 缓冲（PhysX 视图约定），缩放系数必须同设备，
        # 否则 CPU×CUDA 直接报 "Expected all tensors to be on the same device"。
        scales_host = mass_scales.detach().to("cpu").unsqueeze(1)
        cart.root_physx_view.set_masses(nominal_masses.unsqueeze(0) * scales_host, cart_env_idx)
        cart.root_physx_view.set_inertias(nominal_inertias.unsqueeze(0) * scales_host, cart_env_idx)
        actual_masses = cart.root_physx_view.get_masses().sum(dim=1).tolist()
        torque_limits = robot.data.joint_effort_limits[0, policy_to_asset].tolist()

        def build_connection_model(names):
            """按逐 env 的连接类型建模型：全同单模型，混合用 MultiRopeModel。"""
            unique = list(dict.fromkeys(names))
            if len(unique) == 1:
                return _build_rope(unique[0], args, RIGID_MODEL, make_rope_model)
            indices = {name: index for index, name in enumerate(unique)}
            ids = torch.tensor([indices[name] for name in names], dtype=torch.long,
                               device=args.device)
            return MultiRopeModel(models=tuple(_build_rope(name, args, RIGID_MODEL, make_rope_model)
                                               for name in unique), model_ids=ids)

        rope_model = build_connection_model([case.connection for case in slope_cases])
        # 观测里的 `projected_gravity` 是**单位**重力方向（训练侧同口径）；`set_gravity()`
        # 要的是完整向量（含 9.81 的模长），所以两者分开：`gravity_world` 只给观测用。
        gravity_world = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=args.device)

        def per_env(vector_1x3):
            return vector_1x3.view(1, 3).expand(num_envs, 3)

        robot_attach = torch.tensor(robot_attachment, dtype=torch.float32,
                                    device=args.device).view(1, 1, 3)
        cart_attach = torch.tensor(cart_attachment, dtype=torch.float32,
                                   device=args.device).view(1, 1, 3)
        robot_zero_torque = torch.zeros(num_envs, robot.num_bodies, 3,
                                        dtype=torch.float32, device=args.device)
        cart_zero_torque = torch.zeros(num_envs, cart.num_bodies, 3,
                                       dtype=torch.float32, device=args.device)

        def link_frame_force(asset, body_id, force_world):
            local = math_utils.quat_apply_inverse(asset.data.body_quat_w[:, body_id], force_world)
            return local.unsqueeze(1)

        def _quat_to_matrix(quat):
            w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
            return ((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
                    (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
                    (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))

        def host_buffer(tensor):
            return tensor.to(args.device)

        def inverse_inertia_world(asset, body_id):
            flat = host_buffer(asset.root_physx_view.get_inertias())[:, body_id]
            rotation = _quat_to_matrix(asset.data.body_quat_w[:, body_id])
            return world_inverse_inertia((flat[:, 0], flat[:, 4], flat[:, 8]), rotation)

        robot_mass_kg = host_buffer(robot.root_physx_view.get_masses()).sum(dim=1)

        def cart_mass_effective_kg():
            masses = host_buffer(cart.root_physx_view.get_masses())
            inertias = host_buffer(cart.root_physx_view.get_inertias())
            spun = sum(inertias[:, body_id, 4] for body_id in cart_wheel_ids)
            return masses.sum(dim=1) + 4.0 * spun / model["wheel_radius_m"] ** 2

        def apply_rope_and_resistance(*, wheel_damping: float, hold_damping: float):
            robot_offset_w = math_utils.quat_apply(robot.data.body_quat_w[:, base_ids[0]],
                                                   per_env(robot_attach))
            cart_offset_w = math_utils.quat_apply(cart.data.body_quat_w[:, cart_base_ids[0]],
                                                  per_env(cart_attach))
            robot_p = robot.data.body_pos_w[:, base_ids[0]] + robot_offset_w
            cart_p = cart.data.body_pos_w[:, cart_base_ids[0]] + cart_offset_w
            robot_v = point_velocity(robot.data.body_lin_vel_w[:, base_ids[0]],
                                     robot.data.body_ang_vel_w[:, base_ids[0]], robot_offset_w)
            cart_v = point_velocity(cart.data.body_lin_vel_w[:, cart_base_ids[0]],
                                    cart.data.body_ang_vel_w[:, cart_base_ids[0]], cart_offset_w)
            robot_props = BodyProperties(
                mass=robot_mass_kg,
                inverse_inertia_world=inverse_inertia_world(robot, base_ids[0]),
                offset=(robot_offset_w[:, 0], robot_offset_w[:, 1], robot_offset_w[:, 2]))
            cart_props = BodyProperties(
                mass=cart_mass_effective_kg(),
                inverse_inertia_world=inverse_inertia_world(cart, cart_base_ids[0]),
                offset=(cart_offset_w[:, 0], cart_offset_w[:, 1], cart_offset_w[:, 2]))
            state = rope_model.update(robot_point=robot_p, cart_point=cart_p,
                                      robot_velocity=robot_v, cart_velocity=cart_v, dt=dt,
                                      robot=robot_props, cart=cart_props)
            force_robot = torch.stack([torch.as_tensor(component)
                                       for component in state.force_on_robot], dim=-1)
            force_cart = torch.stack([torch.as_tensor(component)
                                      for component in state.force_on_cart], dim=-1)
            robot.set_external_force_and_torque(link_frame_force(robot, base_ids[0], force_robot),
                                                robot_zero_torque[:, :1],
                                                positions=robot_attach.expand(num_envs, 1, 3),
                                                body_ids=base_ids)
            cart.set_external_force_and_torque(link_frame_force(cart, cart_base_ids[0], force_cart),
                                               cart_zero_torque[:, :1],
                                               positions=cart_attach.expand(num_envs, 1, 3),
                                               body_ids=cart_base_ids)
            damping = wheel_damping + hold_damping
            effort = torch.zeros_like(cart.data.joint_pos)
            effort[:, cart_joint_ids] = viscous_resistance(
                cart.data.joint_vel[:, cart_joint_ids], damping)
            cart.set_joint_effort_target(effort)
            return state

        def policy_step(command_tensor):
            parts = parts_from_robot_state(
                base_ang_vel=robot.data.root_ang_vel_b,
                projected_gravity=math_utils.quat_apply_inverse(
                    robot.data.root_quat_w, per_env(gravity_world)),
                velocity_command=torch.stack(
                    [command_tensor, torch.zeros_like(command_tensor),
                     torch.zeros_like(command_tensor)], dim=-1),
                joint_pos=robot.data.joint_pos[:, policy_to_asset],
                joint_vel=robot.data.joint_vel[:, policy_to_asset])
            out = policy.step(parts)
            robot.set_joint_position_target(out.joint_targets[:, asset_to_policy])
            return out.joint_targets

        def set_gravity(gravity):
            """把重力写进 PhysX 场景（Isaac Lab 官方 `randomize_physics_scene_gravity` 同一 API）。"""
            physics_sim_view = sim.physics_sim_view
            physics_sim_view.set_gravity(carb.Float3(*[float(value) for value in gravity]))

        def reset_episode():
            """把一个回合的初始状态写死（`Articulation.reset()` 不写位姿，必须显式重写）。"""
            for asset in (robot, cart):
                root = asset.data.default_root_state.clone()
                root[:, :3] += scene.env_origins
                root[:, 7:] = 0.0
                asset.write_root_pose_to_sim(root[:, :7])
                asset.write_root_velocity_to_sim(root[:, 7:])
                asset.write_joint_state_to_sim(asset.data.default_joint_pos.clone(),
                                               torch.zeros_like(asset.data.default_joint_vel))
            cart.set_joint_effort_target(torch.zeros_like(cart.data.joint_pos))
            robot.set_external_force_and_torque(robot_zero_torque, robot_zero_torque)
            cart.set_external_force_and_torque(cart_zero_torque, cart_zero_torque)
            policy.reset()
            scene.reset()
            scene.update(dt)

        def make_row(env_index: int, *, phase: str, step: int, command: float,
                     gravity, joint_targets, joint_err_rms: float, joint_err_max: float) -> dict:
            quat = robot.data.root_quat_w[env_index]
            qw, qx, qy, qz = (float(value) for value in quat)
            pitch = math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx))))
            row = {
                "phase": phase,
                "time_s": (step + 1) * dt,
                "user_cmd_mps": command,
                "ref_cmd_mps": command,
                "robot_vx_mps": float(robot.data.root_lin_vel_w[env_index, 0]),
                "load_vx_mps": float(cart.data.root_lin_vel_w[env_index, 0]),
                "rope_tension_n": float(state.rope_tension[env_index]),
                "rope_extension_m": float(state.rope_extension[env_index]),
                "rope_length_rate_mps": float(state.rope_length_rate[env_index]),
                "rope_taut": float(state.is_taut[env_index]),
                "rope_impulse_ns": float(state.rope_impulse[env_index]),
                "rope_distance_m": float(state.rope_length[env_index]),
                "robot_x_m": float(robot.data.root_pos_w[env_index, 0]),
                "load_x_m": float(cart.data.root_pos_w[env_index, 0]),
                "robot_z_m": float(robot.data.root_pos_w[env_index, 2]),
                "load_z_m": float(cart.data.root_pos_w[env_index, 2]),
                "body_pitch_rad": pitch,
                "body_pitch_rate_radps": float(robot.data.root_ang_vel_b[env_index, 1]),
                "cart_deck_fx_n": float(deck_contacts.data.net_forces_w[env_index, 0, 0]),
                "cart_wheel_fx_n": float(contacts.data.net_forces_w[env_index, :, 0].sum()),
                "robot_quat_x": float(quat[1]), "robot_quat_y": float(quat[2]),
                "robot_quat_z": float(quat[3]), "robot_quat_w": float(quat[0]),
                "load_quat_x": float(cart.data.root_quat_w[env_index, 1]),
                "load_quat_y": float(cart.data.root_quat_w[env_index, 2]),
                "load_quat_z": float(cart.data.root_quat_w[env_index, 3]),
                "load_quat_w": float(cart.data.root_quat_w[env_index, 0]),
                "velocity_cmd_mps": command,
                "robot_vx_b_mps": float(robot.data.root_lin_vel_b[env_index, 0]),
                "joint_err_rms_rad": joint_err_rms,
                "joint_err_max_rad": joint_err_max,
                "slope_deg": slope_cases[env_index].slope_deg,
                "gravity_x_mps2": float(gravity[0]),
                "gravity_z_mps2": float(gravity[2]),
                "connection": slope_cases[env_index].connection,
                "cart_mass_kg": actual_masses[env_index],
            }
            row.update(wheel_omega_fields(cart.data.joint_vel[env_index], cart_joint_ids))
            row.update(joint_state_fields(
                pos=robot.data.joint_pos[env_index, policy_to_asset],
                target=joint_targets[env_index],
                torque=robot.data.applied_torque[env_index, policy_to_asset]))
            if set(row) != set(TEST_FIELDS):
                missing = sorted(set(TEST_FIELDS) - set(row))
                extra = sorted(set(row) - set(TEST_FIELDS))
                raise RuntimeError(f"记录字段与契约不一致：缺 {missing}，多 {extra}")
            return row

        # ------------------------------------------------------------ 主循环
        case_summaries = []
        case_dirs = []
        started = time.time()
        for slope in args.slopes:
            plan = slope_ground_plan(slope, args.slope_backend)
            gravity = plan["gravity_mps2"]
            gravity_vector = torch.tensor(gravity, dtype=torch.float32, device=args.device)
            gravity_world = gravity_vector / gravity_vector.norm()
            set_gravity(gravity)
            reset_episode()
            print(f"\n[pass slope {slope:+g} deg] 重力 {tuple(round(v, 4) for v in gravity)} "
                  f"（{plan['note']}）；{num_envs} 环境并行", flush=True)
            case_rows = [[] for _ in range(num_envs)]
            joint_targets = torch.zeros(num_envs, policy_cfg.num_joints,
                                        dtype=torch.float32, device=args.device)
            command_tensor = torch.zeros(num_envs, dtype=torch.float32, device=args.device)
            pass_started = time.time()
            for step in range(args.schedule.total_steps):
                phase = args.schedule.phase_of(step)
                step_in_phase = args.schedule.step_in_phase(step)
                if phase == "tow":
                    if args.command_shaping == "direct":
                        command_tensor = velocities
                    else:
                        command_tensor = velocities * shaped_command(
                            phase="tow", step_in_phase=step_in_phase, velocity=1.0,
                            shaping="ramp", ramp_time_s=args.ramp_time_s, dt=dt)
                else:
                    command_tensor = torch.zeros_like(velocities)
                if step % decimation == 0:
                    joint_targets = policy_step(command_tensor)
                hold = args.hold_damping if (phase == "station" and args.slope_settle == "hold") else 0.0
                state = apply_rope_and_resistance(wheel_damping=args.wheel_damping,
                                                  hold_damping=hold)
                scene.write_data_to_sim()
                sim.step()
                scene.update(dt)
                if step % args.record_every:
                    continue
                joint_err = (robot.data.joint_pos[:, policy_to_asset] - joint_targets)
                err_rms = joint_err.pow(2).mean(dim=1).sqrt().tolist()
                err_max = joint_err.abs().amax(dim=1).tolist()
                commands = command_tensor.tolist()
                for env_index in range(num_envs):
                    case_rows[env_index].append(make_row(
                        env_index, phase=phase, step=step, command=commands[env_index],
                        gravity=gravity, joint_targets=joint_targets,
                        joint_err_rms=err_rms[env_index], joint_err_max=err_max[env_index]))
            print(f"[pass slope {slope:+g} deg] 完成，用时 {time.time() - pass_started:.1f} s",
                  flush=True)

            for env_index, case in enumerate(slope_cases):
                rows = case_rows[env_index]
                tw_summary = summarize_tow(
                    rows, user_command=case.velocity_mps, joint_names=policy_cfg.joint_names,
                    config={
                        "rope": {"model": case.connection,
                                 "rest_length_m": (args.rope_length - args.slack)
                                 if case.connection == RIGID_MODEL else args.rope_length,
                                 "stiffness_n_per_m": args.stiffness,
                                 "damping_ns_per_m": args.damping,
                                 "initial_slack_m": args.slack,
                                 "position_gain": args.position_gain,
                                 "max_correction_rate_mps": args.max_correction_rate},
                        "cart_model": model,
                        "cart_mass_actual_kg": actual_masses[env_index],
                        "robot_mass_kg": float(robot_mass_kg[env_index]),
                        "dt_s": dt,
                    })
                metrics = compute_case_metrics(
                    rows, command_mps=case.velocity_mps, slope_deg=case.slope_deg,
                    connection=case.connection, cart_mass_kg=actual_masses[env_index],
                    schedule=args.schedule, record_dt=record_dt, tow_summary=tw_summary,
                    thresholds=thresholds, joint_names=list(policy_cfg.joint_names),
                    torque_limits=torque_limits, transition_window_s=args.transition_window)
                metrics["summarize_tow"] = tw_summary
                summary = {
                    "case": case.to_dict(),
                    "labels": {"slope_deg": case.slope_deg, "velocity_mps": case.velocity_mps,
                               "connection": case.connection,
                               "cart_mass_kg": actual_masses[env_index]},
                    "schedule": args.schedule.to_dict(),
                    "slope": plan,
                    "metrics": metrics,
                    "verdict": metrics["verdict"],
                    "summarize_tow_valid": tw_summary.get("valid"),
                    "summarize_tow_note": SUMMARIZE_TOW_NOTE,
                }
                case_summaries.append(summary)
                write_json(summaries_dir / f"{case.slug}.json", summary)
                should_write = (args.write_csv == "all"
                                or (args.write_csv == "failed" and metrics["verdict"]["code"] != "OK"))
                if should_write:
                    case_dir = output / case.slug
                    case_dir.mkdir(parents=True, exist_ok=True)
                    write_json(case_dir / "config.json", {
                        "case": case.to_dict(), "schedule": args.schedule.to_dict(),
                        "slope": plan, "connection": experiment["connection"],
                        "ground_friction": args.ground_friction,
                        "wheel_damping_nms_per_rad": args.wheel_damping,
                        "slope_settle": {"mode": args.slope_settle,
                                         "hold_damping_nms_per_rad": args.hold_damping},
                        "user_command_mps": case.velocity_mps,
                        "policy_joint_names": list(policy_cfg.joint_names),
                        "thresholds": thresholds,
                    })
                    with (case_dir / "tow.csv").open("w", newline="", encoding="utf-8") as stream:
                        writer = csv.DictWriter(stream, fieldnames=TEST_FIELDS)
                        writer.writeheader()
                        writer.writerows(rows)
                    case_dirs.append(case.slug)
                verdict = metrics["verdict"]
                print(f"[case] slope{case.slope_deg:+g} v{case.velocity_mps:g} "
                      f"{case.connection} m{actual_masses[env_index]:g}kg ⇒ {verdict['code']}"
                      f"{(' [' + ','.join(verdict['reasons']) + ']') if verdict['reasons'] else ''}"
                      f"  起步关节RMS={_fmt(metrics['startup']['joint_rms_rad'])} "
                      f"跟速MAE={_fmt(metrics['speed'].get('mae_mps'))} "
                      f"滑移={_fmt(metrics['stop'].get('cart_coast_distance_m'))} "
                      f"停车最小间隙={_fmt(metrics['stop'].get('min_clearance_coast_m'))} "
                      f"停车关节RMS={_fmt(metrics['stop'].get('joint_rms_rad'))}", flush=True)

        # ------------------------------------------------------------ 报告
        groups = group_statistics(case_summaries)
        conclusion = necessity_conclusion(groups, thresholds)
        labels = {name: getattr(args, name) for name in
                  ("velocities", "connections", "cart_masses", "ground_friction",
                   "command_shaping", "ramp_time_s", "slope_backend", "slope_settle",
                   "record_every", "write_csv", "env_spacing")}
        report = {
            "question": experiment["question"],
            "grid": experiment["grid"], "thresholds": thresholds,
            "arguments": experiment["arguments"],
            "groups": groups, "conclusion": conclusion,
            "cases": case_summaries,
            "csv_cases": case_dirs,
            "elapsed_s": time.time() - started,
            "git": git,
        }
        write_json(output / "report.json", report)
        csv_rows = [case_report_row(summary["metrics"]) for summary in case_summaries]
        if csv_rows:
            with (output / "report.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
                writer.writeheader()
                writer.writerows(csv_rows)
        (output / "report.md").write_text(
            build_markdown_report(case_summaries=case_summaries, groups=groups,
                                  conclusion=conclusion, args_dict=labels,
                                  thresholds=thresholds, schedule=args.schedule,
                                  slopes=list(args.slopes), git=git),
            encoding="utf-8")

        print("\n" + format_verdict_matrix(case_summaries, slopes=list(args.slopes)))
        print("[conclusion] 上层任务是否有必要（本网格 + 本阈值）")
        for line in conclusion["lines"]:
            print(line)
        print(f"\n[DONE] {len(case_summaries)} case，用时 {time.time() - started:.1f} s ⇒ {output}",
              flush=True)
        experiment.update(state="completed", case_count=len(case_summaries),
                          csv_cases=case_dirs, elapsed_s=time.time() - started)
        write_json(output / "experiment.json", experiment)
        sys.stdout.flush()
        sys.stderr.flush()
        # 与 tow_drag.py 同一约定：不依赖 Kit 的关停路径（它会吞掉异常与退出码）。
        os._exit(0)
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001
        experiment.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        try:
            write_json(output / "experiment.json", experiment)
        except OSError:
            pass
        print(f"[FAILED] {type(exc).__name__}: {exc}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(2)


def _finite_only(value):
    """把非有限浮点递归换成 `None`。

    `json.dumps(..., allow_nan=False)` 会因为**一个** NaN 让整轮 225 个 case 的产物
    一起写不出来（长跑最后一步失败最亏）。指标里 NaN 只该出现在「窗口为空」这类退化情形，
    换成 `None` 后仍然看得见（`available` 字段会说明），不会连累其它 case。
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _finite_only(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_only(item) for item in value]
    return value


def _fmt(value) -> str:
    """打印用：None/NaN 显示 n/a。"""
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(value) else f"{value:.3f}"


def _build_rope(name, args, rigid_model, make_rope_model):
    """按连接类型建单个模型（rigid 的长度是初始挂点距 L0 − slack，与测量台一致）。"""
    rest_length = (args.rope_length - args.slack) if name == rigid_model else args.rope_length
    return make_rope_model(name, rest_length=rest_length, stiffness=args.stiffness,
                           damping=args.damping, position_gain=args.position_gain,
                           max_correction_rate=args.max_correction_rate)


def wheel_omega_fields(joint_vel, joint_ids) -> dict:
    """四轮角速度列（列名只在这里生成，离线测试直接调用核对）。"""
    return {f"wheel_{leg}_omega_radps": float(joint_vel[joint_id])
            for leg, joint_id in zip(WHEEL_LEGS, joint_ids)}


def joint_state_fields(*, pos, target, torque) -> dict:
    """12 关节的「实测角 / 当拍目标角 / 实际力矩」三组列（策略顺序，逐腿 FL/FR/RL/RR）。"""
    fields = {}
    for index, name in enumerate(recording.ROBOT_JOINT_POSITION_FIELDS):
        fields[name] = float(pos[index])
        fields[f"robot_jt_{index:02d}"] = float(target[index])
        fields[f"robot_tau_{index:02d}"] = float(torque[index])
    return fields


def _initial_cart_x(rope_length: float, slack: float, *, spawn_height: float,
                    cart_height: float, robot_offset, cart_offset) -> float:
    """机器人在原点出生时的小车初始 x：两挂点三维距离 = L0 − slack（绳是松的）。"""
    target = rope_length - slack
    dz = (spawn_height + robot_offset[2]) - (cart_height + cart_offset[2])
    if target * target < dz * dz:
        raise ValueError(f"绳长 {rope_length} m 减松弛 {slack} m 后小于挂点高差 {abs(dz):.3f} m")
    horizontal = math.sqrt(target * target - dz * dz)
    return 0.0 + robot_offset[0] - cart_offset[0] - horizontal


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
