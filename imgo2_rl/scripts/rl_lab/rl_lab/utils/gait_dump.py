"""回放时 dump「步态量测序列」，供 `scripts/tools/gait_report.py` 离线算相位（2026-09-28）。

**为什么需要它**：训练曲线里的三个相位核系数（`gait_trot/bound/pace_<地形>`）在 duty≈0.5 时
会**趋同**（滞空≈触地 ⇒ 反相核饱和，实测三者常常完全相等），所以"平地是不是 trot、箱/沟有没有换步态"
**在日志里看不出来**。相位（两条 0/1 触地序列的相对时移）在同样 duty 下 0.0 与 0.5 仍差满半周期，
是可分的量 —— 前提是拿到**回放时的足端接触序列**，本模块就是干这个的。

本文件**只依赖 numpy**（不 import torch / isaaclab），因此 `tests/test_gait_dump.py` 能在没有 GPU 的
机器上端到端验证「合成序列 → 本模块写出 npz → gait_report 读回并判定」这条链路。Isaac 侧的取数
（接触传感器、terrain_types…）留在 `scripts/rl_lab/cmoe/play.py` 的胶水里。

**npz 契约**（与 `gait_report.py` 的 `CONTRACT_KEYS` 一致，`contact` 是唯一硬要求）：

=================  ==========================  ==========================================
键                 形状/类型                   含义
=================  ==========================  ==========================================
`contact`          bool  [T, N, 4]              每步/每环境/每足是否触地；**足序恒为 FL,FR,RL,RR**
`base_height`      f32   [T, N]                 base 高度（米）
`cmd`              f32   [T, N, 3]              速度指令（vx, vy, wz）
`base_lin_vel`     f32   [T, N, 3]              base 线速度（本体系）
`terrain_type`     i32   [N]                    环境所在地形**列**索引（按 `terrain_names` 取名字）
`terrain_level`    i32   [N]                    课程等级（窗口内取**众数**）
`terrain_names`    U     [n_cols]               列索引 → 地形名（`sub_terrains` 的键顺序）
`dt`               f32   标量                   控制步长（秒）
`foot_names`       U     [4]                    实际 body 名（如 FL_FOOT），仅供人看
`gate`             f32   [T, N, K]             专家门控权重（softmax 后，K=专家数）；供
                                                 `scripts/tools/expert_report.py` 做「专家×地形」交叉表
=================  ==========================  ==========================================
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# 足序是**契约**的一部分：`gait_report.py` 按位置把 `contact` 最后一维读成 FL, FR, RL, RR。
FOOT_ORDER = ("FL", "FR", "RL", "RR")


def canonical_foot_indices(body_names: Sequence[str]) -> tuple[list[int], list[str]]:
    """把传感器给的 body 名映射成契约足序 FL,FR,RL,RR 对应的索引。

    接受 `FL_FOOT` / `fl_foot` / `FL` 这类命名（大小写不敏感；按"完全等于 / 以它结尾 / 以它开头"
    依次放宽）。任一只足找不到就抛 `ValueError` —— 调用侧应当 fail-soft（警告后放弃 dump，
    绝不能让量测把回放弄挂）。

    Returns:
        `(indices, names)`：`indices[i]` 是第 i 只契约足在 `body_names` 里的下标；
        `names[i]` 是它实际的 body 名。
    """
    lowered = [str(name).lower() for name in body_names]
    indices: list[int] = []
    names: list[str] = []
    for foot in FOOT_ORDER:
        key = foot.lower()
        hit: int | None = None
        for predicate in (
            lambda name, key=key: name == f"{key}_foot",
            lambda name, key=key: name.endswith(f"{key}_foot"),
            lambda name, key=key: name.startswith(key),
        ):
            for index, name in enumerate(lowered):
                if predicate(name):
                    hit = index
                    break
            if hit is not None:
                break
        if hit is None:
            raise ValueError(
                f"在 {list(body_names)} 里找不到 {foot} 足（期望形如 {foot}_FOOT）；"
                "无法确定契约足序，放弃步态 dump"
            )
        indices.append(hit)
        names.append(str(body_names[hit]))
    return indices, names


def expand_terrain_names(keys, proportions, num_cols) -> list[str]:
    """把 `sub_terrains` 的键按 **Isaac Lab 的列分配规则**铺成 `num_cols` 个逐列名。

    ⚠️ 踩过的坑（2026-09-28 真实 dump）：`terrain_generator.sub_terrains` 只有 **11** 个键，而地形有
    **40** 列 —— 直接把 `keys[i]` 当第 i 列的名字是**错的**（列 11 之后全变成"列N"，前 11 行也未必对）。
    Isaac 的规则在 `isaaclab/terrains/terrain_generator.py:233-241`：

        proportions /= sum(proportions)
        sub_indices[i] = min(where(i / num_cols + 0.001 < cumsum(proportions))[0])

    ⇒ 每个 sub-terrain 按比例占**连续若干列**（等比例 10 类 × 40 列 = 每类 4 列）。这里逐字复刻该规则，
    纯 numpy/列表实现（可离线单测，见 `tests/test_gait_dump.py`）。
    """
    names = [str(key) for key in keys]
    props = np.asarray([float(x) for x in proportions], dtype=np.float64)
    if props.size != len(names) or props.size == 0:
        raise ValueError(f"proportions 个数 {props.size} 与 sub_terrains 个数 {len(names)} 不一致")
    if not np.isfinite(props).all() or props.sum() <= 0:
        raise ValueError("proportions 必须是有限正数")
    props = props / props.sum()
    cumulative = np.cumsum(props)
    out: list[str] = []
    for index in range(int(num_cols)):
        target = index / int(num_cols) + 0.001
        hit = np.where(target < cumulative)[0]
        out.append(names[int(hit[0])] if hit.size else names[-1])
    return out


def _to_numpy(value: Any) -> np.ndarray:
    """torch tensor / 列表 / numpy 一律转 numpy（本模块不 import torch）。"""
    if value is None:
        return None  # type: ignore[return-value]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class GaitDumper:
    """按步累加足端接触等序列，最后写成一个 `gait_report.py` 能读的 npz。

    典型用法（见 `scripts/rl_lab/cmoe/play.py`）：构造一次 → 每个控制步 `record(...)` →
    退出前 `save()`（重复 `save()` 安全，第二次直接返回上次的 summary）。
    """

    def __init__(
        self,
        path: str | Path,
        dt: float,
        foot_names: Sequence[str] = FOOT_ORDER,
        terrain_names: Iterable[str] = (),
        max_steps: int = 0,
    ) -> None:
        self.path = Path(path)
        self.dt = float(dt)
        self.foot_names = [str(name) for name in foot_names]
        self.terrain_names = [str(name) for name in terrain_names]
        self.max_steps = int(max_steps)
        self._contact: list[np.ndarray] = []
        self._base_height: list[np.ndarray] = []
        self._cmd: list[np.ndarray] = []
        self._base_lin_vel: list[np.ndarray] = []
        self._terrain_type: list[np.ndarray] = []
        self._terrain_level: list[np.ndarray] = []
        self._gate: list[np.ndarray] = []
        self._num_envs: int | None = None
        self._shape_warnings = 0
        self._summary: dict[str, Any] | None = None

    # ------------------------------------------------------------------ 记录
    @property
    def steps(self) -> int:
        return len(self._contact)

    @property
    def num_envs(self) -> int:
        return int(self._num_envs or 0)

    def record(
        self,
        contact: Any,
        *,
        base_height: Any = None,
        cmd: Any = None,
        base_lin_vel: Any = None,
        terrain_type: Any = None,
        terrain_level: Any = None,
        gate: Any = None,
    ) -> None:
        """记录**一个控制步**的所有环境。

        `contact` 形状 `[N, 4]`（单环境可给 `[4]`），足序必须是 FL, FR, RL, RR。其余字段
        `[N]` / `[N, 3]`；给 `None` 就整段不记这个字段（报告侧 fail-soft 退化）。
        环境数 N 以**第一步**为准；后续步形状不合就丢弃该步并计数（`save()` 的 summary 里
        会报 `shape_warnings`，不抛异常 —— 量测不能把回放弄挂）。
        """
        contact_array = _to_numpy(contact)
        if contact_array is None:
            return
        contact_array = contact_array.astype(bool)
        if contact_array.ndim == 1:
            contact_array = contact_array[None, :]
        if contact_array.ndim != 2:
            raise ValueError(f"contact 形状应为 [N,4]（或 [4]），收到 {contact_array.shape}")
        if contact_array.shape[1] != len(FOOT_ORDER):
            raise ValueError(
                f"contact 最后一维应为 {len(FOOT_ORDER)}（契约足序 {list(FOOT_ORDER)}），"
                f"收到 {contact_array.shape[1]}"
            )
        num_envs = contact_array.shape[0]
        if self._num_envs is None:
            self._num_envs = num_envs
        elif num_envs != self._num_envs:
            self._shape_warnings += 1
            return
        self._contact.append(contact_array)
        for store, value in (
            (self._base_height, base_height),
            (self._cmd, cmd),
            (self._base_lin_vel, base_lin_vel),
            (self._terrain_type, terrain_type),
            (self._terrain_level, terrain_level),
            (self._gate, gate),
        ):
            array = _to_numpy(value)
            store.append(None if array is None else np.asarray(array))  # type: ignore[arg-type]

    # -------------------------------------------------------------------- 落盘
    def save(self) -> dict[str, Any]:
        """写出 npz 并返回 summary（可重复调用）。没有记录到任何步时抛 `RuntimeError`。"""
        if self._summary is not None:
            return self._summary
        if not self._contact:
            raise RuntimeError("没有记录到任何步（record 从未成功调用），不写 npz")
        contact = np.stack(self._contact, axis=0)                       # [T, N, 4]
        steps, num_envs, _ = contact.shape
        payload: dict[str, np.ndarray] = {
            "contact": contact.astype(bool),
            "dt": np.float32(self.dt),
            "terrain_names": np.asarray(self.terrain_names, dtype="U"),
            "foot_names": np.asarray(self.foot_names, dtype="U"),
        }
        for key, store, dtype in (
            ("base_height", self._base_height, np.float32),
            ("cmd", self._cmd, np.float32),
            ("base_lin_vel", self._base_lin_vel, np.float32),
            ("terrain_type", self._terrain_type, np.int32),
            ("terrain_level", self._terrain_level, np.int32),
            ("gate", self._gate, np.float32),
        ):
            if not store or any(item is None for item in store):
                continue        # 该字段整段没记（或记了一半）⇒ 干脆不写，报告侧会退化
            array = np.stack([np.asarray(item) for item in store], axis=0)
            if key in ("terrain_type", "terrain_level"):
                # 每个控制步一条**每环境标量**（形状 [N] 或 [N,1]）。窗口内可能发生过 reset
                # （等级会变）⇒ 取众数作为该环境的代表值。形状对不上就干脆不写这个字段。
                if array.size != steps * num_envs:
                    continue
                payload[key] = _per_env_mode(array.reshape(steps, num_envs)).astype(dtype)
                continue
            payload[key] = array.astype(dtype)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, **payload)
        if "gate" in payload and payload["gate"].shape[:2] != (steps, num_envs):
            del payload["gate"]          # 形状对不上就整段丢掉（报告侧会退化）
        stance_rate = float(contact.mean()) if contact.size else float("nan")
        other = [key for key in ("base_height", "cmd", "base_lin_vel", "terrain_type", "terrain_level", "gate")
                 if key in payload]
        self._summary = {
            "path": str(self.path),
            "steps": steps,
            "num_envs": num_envs,
            "stance_rate": stance_rate,
            "keys": ["contact", "dt", "terrain_names", "foot_names", *other],
            "terrain_names": list(self.terrain_names),
            "foot_names": list(self.foot_names),
            "shape_warnings": self._shape_warnings,
        }
        return self._summary


def _per_env_mode(values: np.ndarray) -> np.ndarray:
    """对 `[T, N]` 的整型/浮点序列，按列取众数（并列取较小值，保证确定性）。"""
    values = np.asarray(values)
    if values.ndim != 2:
        return values
    out = np.empty(values.shape[1], dtype=values.dtype)
    for column in range(values.shape[1]):
        counts = Counter(values[:, column].tolist())
        best = max(counts.items(), key=lambda item: (item[1], -float(item[0])))
        out[column] = best[0]
    return out


def format_gait_summary(summary: dict[str, Any], report_hint: str | None = None) -> str:
    """给回放终端用的一行摘要（含下一步该跑哪条命令）。"""
    parts = [
        f"[GAIT] 已写出 {summary['path']}",
        f"{summary['steps']} 步 × {summary['num_envs']} 环境",
        f"触地率 {summary['stance_rate'] * 100:.1f}%",
        f"地形列 {len(summary['terrain_names'])}",
        f"字段 {','.join(summary['keys'])}",
    ]
    if summary.get("shape_warnings"):
        parts.append(f"⚠️ 丢弃了 {summary['shape_warnings']} 个形状异常的步")
    text = "；".join(parts)
    if report_hint:
        text += f"\n[GAIT] 离线分析：{report_hint}"
    return text
