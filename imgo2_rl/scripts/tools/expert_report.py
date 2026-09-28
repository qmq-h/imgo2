#!/usr/bin/env python3
"""离线看「专家分布」：门控权重 × 地形（纯离线，只用标准库 + numpy）。

**为什么需要它**：训练里只记了 `Policy/expert_{i}_mean_weight`（**全体环境、全地形**的平均）与
`Policy/gate_entropy`，所以只能看出"有没有塌缩"，**答不了"哪个专家管哪类地形/哪种风格"**——
一个专家全局均值 0.2，完全可能是"flat 上 1.0、stairs 上 0.0"。要靠 `cmoe/play.py --dump_gait`
把**每步/每环境的门控权重**与所在**地形列**一起 dump 下来，才能做这张交叉表。

`cmoe/play.py --dump_gait` 写的 npz 里与本工具相关的键：

=================  ==================  ==========================================
键                 形状                含义
=================  ==================  ==========================================
`gate`             f32  [T, N, K]       门控权重（softmax 后的概率，和为 1）
`terrain_type`     i32  [N]             地形列索引（对应 `terrain_names`）
`terrain_names`    U    [n_cols]        列索引 → 地形名
`terrain_level`    i32  [N]（可选）     课程等级
`cmd`              f32  [T, N, 3]（可选）速度指令（用 `--vx-min` 过滤掉"没在走"的步）
`contact`          bool [T, N, 4]（可选）只用于报告里附带"该列实测步态"的一句话
=================  ==================  ==========================================

用法：
    python3 scripts/tools/expert_report.py --npz logs/gait_5000_l6.npz [--vx-min 0.3] [--json out.json]

输出两类表：
* **按地形列**：每个专家在该列上的平均门控权重、`argmax 占比`（该专家是"最大权重"的步数比例）、
  以及该列的"主责专家"（平均权重最大者）。
* **全局**：各专家平均权重、门控熵（nats）与均匀基线 `ln K`（熵≈基线 ⇒ 专家没分化；熵→0 ⇒ 塌缩）。

**判读要点**：`argmax 占比` 才是"分工"的证据 —— 平均权重高但 argmax 占比不高，说明只是"普遍参与"，
不是"负责这一列"。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def load_npz(path: Path) -> dict:
    """读 npz；`gate` 是本工具的硬要求（没有它做不了这个分析）。"""
    if not path.exists():
        raise FileNotFoundError(f"not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    if "gate" not in arrays:
        raise KeyError(
            "npz 里没有 `gate`（门控权重）。用带门控记录的 `cmoe/play.py --dump_gait` 重新 dump："
            f"现有键 {sorted(arrays)}"
        )
    gate = np.asarray(arrays["gate"], dtype=np.float64)
    if gate.ndim != 3:
        raise ValueError(f"gate 形状应为 [T,N,K]，收到 {gate.shape}")
    steps, num_envs, num_experts = gate.shape
    if steps == 0 or num_experts == 0:
        raise ValueError(f"gate 为空（形状 {gate.shape}）")
    return arrays


def prepare(arrays: dict, vx_min: float = 0.3) -> dict:
    """整理成 {gate[T,N,K], terrain_index[N], names, level[N]，valid[T,N] 掩码}。"""
    gate = np.asarray(arrays["gate"], dtype=np.float64)
    steps, num_envs, _num_experts = gate.shape
    names: tuple[str, ...] = ()
    if "terrain_names" in arrays:
        names = tuple(str(x) for x in np.asarray(arrays["terrain_names"]).reshape(-1))
    terrain_index = np.zeros(num_envs, dtype=np.int64)
    if "terrain_type" in arrays:
        raw = np.asarray(arrays["terrain_type"]).reshape(-1)
        if raw.size == num_envs:
            terrain_index = raw.astype(np.int64)
        else:
            _warn(f"terrain_type 长度 {raw.size} 与 gate 的环境数 {num_envs} 不符 ⇒ 按单列处理")
    levels = None
    if "terrain_level" in arrays:
        raw = np.asarray(arrays["terrain_level"]).reshape(-1)
        if raw.size == num_envs:
            levels = raw.astype(np.int64)
    valid = np.ones((steps, num_envs), dtype=bool)
    if "cmd" in arrays:
        cmd = np.asarray(arrays["cmd"], dtype=np.float64)
        if cmd.shape[:2] == (steps, num_envs) and cmd.shape[-1] >= 2:
            valid = np.linalg.norm(cmd[..., :2], axis=-1) > float(vx_min)
        else:
            _warn(f"cmd 形状 {cmd.shape} 与 gate 不匹配 ⇒ 不做速度过滤")
    return {"gate": gate, "terrain_index": terrain_index, "names": names,
            "levels": levels, "valid": valid}


def _terrain_name(names: tuple[str, ...], index: int) -> str:
    if 0 <= index < len(names):
        return names[index]
    return f"列{index}"


def regroup_by_name(prepared: dict) -> dict:
    """把 `terrain_index` 重映射成"按名字去重后的 id" ⇒ 每个**地形类型**一行（40 列 = 11 类）。"""
    names = prepared["names"]
    index = np.asarray(prepared["terrain_index"]).reshape(-1)
    label = [names[i] if 0 <= i < len(names) else f"列{i}" for i in index.tolist()]
    unique = sorted(set(label))
    lookup = {name: k for k, name in enumerate(unique)}
    out = dict(prepared)
    out["terrain_index"] = np.asarray([lookup[item] for item in label], dtype=index.dtype)
    out["names"] = tuple(unique)
    out["levels"] = prepared["levels"]
    return out


def build_report(prepared: dict, min_steps: int = 100) -> dict:
    """算全局 + 按地形的专家分布。样本不足的列仍然给出，但标注 `insufficient`。"""
    gate = prepared["gate"]
    terrain_index = prepared["terrain_index"]
    valid = prepared["valid"]
    steps, num_envs, num_experts = gate.shape
    reports: list[dict] = []
    for column in sorted(set(int(x) for x in terrain_index.tolist())):
        env_ids = np.flatnonzero(terrain_index == column)
        sub_valid = valid[:, env_ids]                                  # [T, n_col]
        mask = sub_valid[:, :, None]                                   # 广播到 [T, n_col, K]
        weights = gate[:, env_ids, :]
        count = int(sub_valid.sum())
        entry: dict = {
            "terrain_index": column,
            "terrain": _terrain_name(prepared["names"], column),
            "envs": int(len(env_ids)),
            "effective_steps": count,
            "level": None if prepared["levels"] is None else float(np.mean(prepared["levels"][env_ids])),
        }
        if count == 0:
            entry.update({"mean_weight": [None] * num_experts, "argmax_share": [None] * num_experts,
                          "owner": None, "entropy": None, "status": "no_data"})
            reports.append(entry)
            continue
        masked = np.where(mask, weights, np.nan)
        mean_weight = np.nanmean(masked.reshape(-1, num_experts), axis=0)
        argmax = np.argmax(weights, axis=-1)                           # [T, n_col]
        argmax_share = np.array([float(((argmax == k) & sub_valid).sum()) / count
                                 for k in range(num_experts)])
        entropy = float(np.nanmean(-np.sum(np.where(mask, weights * np.log(np.clip(weights, 1e-12, None)), 0.0),
                                           axis=-1)))
        entry.update({
            "mean_weight": [float(x) for x in mean_weight],
            "argmax_share": [float(x) for x in argmax_share],
            "owner": int(np.argmax(mean_weight)),
            "entropy": entropy,
            "status": "ok" if count >= int(min_steps) else "insufficient",
        })
        reports.append(entry)

    flat_valid = valid[:, :, None]
    global_masked = np.where(flat_valid, gate, np.nan).reshape(-1, num_experts)
    global_mean = np.nanmean(global_masked, axis=0)
    global_entropy = float(np.nanmean(-np.sum(
        np.where(flat_valid, gate * np.log(np.clip(gate, 1e-12, None)), 0.0), axis=-1)))
    return {
        "num_experts": num_experts,
        "num_envs": num_envs,
        "steps": steps,
        "effective_steps": int(valid.sum()),
        "global_mean_weight": [float(x) for x in global_mean],
        "global_entropy": global_entropy,
        "uniform_entropy": float(np.log(num_experts)),
        "groups": reports,
    }


def format_report(report: dict) -> str:
    lines: list[str] = []
    k = report["num_experts"]
    lines.append(f"# 专家分布（门控权重）：{report['steps']} 步 × {report['num_envs']} 环境，"
                 f"有效步 {report['effective_steps']}，K={k} 个专家")
    head = f"{'地形':>18} {'等级':>6} {'有效步':>8} " + " ".join(f"{'E' + str(i):>7}" for i in range(k)) \
           + "   " + " ".join(f"{'a' + str(i):>6}" for i in range(k)) + f" {'主责':>6} {'熵':>6}"
    lines.append(head)
    lines.append("-" * len(head))
    for group in report["groups"]:
        level = "-" if group["level"] is None else f"{group['level']:.2f}"
        cells = [f"{group['terrain']:>18}", f"{level:>6}", f"{group['effective_steps']:>8}"]
        for value in group["mean_weight"]:
            cells.append(f"{'-' if value is None else f'{value:.4f}':>7}")
        for value in group["argmax_share"]:
            cells.append(f"{'-' if value is None else f'{value:.3f}':>6}")
        owner = "-" if group["owner"] is None else f"E{group['owner']}"
        entropy = "-" if group["entropy"] is None else f"{group['entropy']:.3f}"
        cells += [f"{owner:>6}", f"{entropy:>6}"]
        lines.append(" ".join(cells))
        if group["status"] == "insufficient":
            lines[-1] += "   ← 样本不足"
        elif group["status"] == "no_data":
            lines[-1] += "   ← 无有效步"
    lines.append("")
    lines.append("全局平均权重 " + " ".join(f"E{i}={w:.4f}" for i, w in enumerate(report["global_mean_weight"])))
    lines.append(f"全局门控熵 {report['global_entropy']:.4f} nats（均匀基线 ln{k}={report['uniform_entropy']:.4f}；"
                 f"比值 {report['global_entropy'] / report['uniform_entropy']:.3f}）")
    lines.append("读法：`E*` 列是该列上的平均门控权重；`a*` 列是 argmax 占比（**分工的证据**）；"
                 "「主责」= 平均权重最大的专家。熵≈lnK ⇒ 没分化；熵→0 ⇒ 塌缩成单专家。")
    return "\n".join(lines)


def _json_safe(value):
    if isinstance(value, float):
        return None if value != value else value
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--npz", required=True, help="cmoe/play.py --dump_gait 写出的 npz")
    parser.add_argument("--vx-min", type=float, default=0.3, help="只统计 |cmd.xy| > 该值的步（默认 0.3）")
    parser.add_argument("--min-steps", type=int, default=100, help="样本不足的阈值（默认 100 个有效步）")
    parser.add_argument("--by-name", action="store_true",
                        help="按**地形类型名**聚合（`sub_terrains` 每类占连续若干列）⇒ 每类一行")
    parser.add_argument("--json", default=None, help="把结果写成 JSON")
    parser.add_argument("--names", default=None,
                        help="逗号分隔的**逐列**地形名，覆盖 npz 里的 `terrain_names`。"
                             "用于修旧 dump：`sub_terrains` 只有 ~11 个键却有 40 列时，npz 里的名字对不上"
                             "（列 11 之后显示成「列N」），可在这里按正确的 40 个名字覆盖。")
    args = parser.parse_args(argv)

    try:
        arrays = load_npz(Path(args.npz))
        if args.names:
            arrays = dict(arrays)
            arrays["terrain_names"] = np.asarray([x.strip() for x in args.names.split(",") if x.strip()],
                                                 dtype="U")
        prepared = prepare(arrays, vx_min=args.vx_min)
        if args.by_name:
            prepared = regroup_by_name(prepared)
        observed = int(np.max(prepared["terrain_index"])) + 1 if prepared["terrain_index"].size else 0
        if len(prepared["names"]) < observed:
            _warn(f"npz 的 terrain_names 只有 {len(prepared['names'])} 项，但出现了 {observed} 个地形列 "
                  f"⇒ 列名不可靠（用 --names 传逐列名覆盖；新 dump 已按 Isaac 的比例分配铺开）")
        report = build_report(prepared, min_steps=args.min_steps)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(format_report(report))
    if args.json:
        Path(args.json).write_text(json.dumps(_json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"# 已写出 {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
