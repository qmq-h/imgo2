#!/usr/bin/env python3
"""同一套条件下的两轮 `play_towing_test` 逐 cell 对照（纯标准库）。

用户 2026-10-11 的口径：**评价上层策略看"同条件下 v3 对基线的配对差"**，
不看基线的绝对值（基线差恰恰是好对照）。这个工具就是那个配对：
读两轮的 `report.json`（必须**只有 `--upper-checkpoint` 不同**），按 `cell_index` 逐格配对，给

1. **头条**：硬证据碰撞率（微平均 / 宏平均）与判定码，两轮并排 + delta；
2. **逐 cell 翻转表**：两轮都撞 / 只有基线撞（**救回**）/ 只有策略撞（**弄坏**）/ 都不撞；
3. **按轴 delta**：长度档（row）、坡度量级、速度档、质量档、连接类型；
4. **指令侧与 decoder**（策略轮有才打）：`stop.loco_cmd_vx_coast_mean_mps` 等新列的中位与"负偏移占比"，
   以及 `metrics["estimator"]` 的中位绝对误差（latent 聚类交给 `analyze_latent.py`）。

用法：

    python3 scripts/tools/compare_play_runs.py \\
        --baseline imgo2_rl/logs/towing/play_test/v3_800_base_c008/report.json \\
        --policy   imgo2_rl/logs/towing/play_test/v3_800_policy_c008/report.json

硬证据定义（与 `docs/towing/towing_play_test_2700_2026-10-11.md` 一致，**不看判定码**）：
`stop.deck_contact_peak_n > 0` 或 `stop.min_clearance_coast_m <= 0`。
判定码（`COL`/`LOW`/`SPD`）另列，因为 `load_velocity_jump` 代理通道会把改善读反。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

#: 影响物理的 CLI 参数：两轮只有 `--upper-checkpoint` 不同才算"同一套条件"
PAIRED_ARGUMENTS = (
    "terrain_profile",
    "num_envs", "velocities", "cart_masses", "ground_friction", "wheel_damping",
    "command_shaping", "ramp_time_s", "settle_time", "tow_duration", "coast_duration",
    "dt", "record_every", "no_cart_fraction", "lane_keeping", "device",
)
#: 逐 cell 对照时按哪些轴聚合
AXES = (
    ("row", "长度档 row", lambda case: case["row"]),
    ("grade_deg", "坡度量级", lambda case: case["grade_deg"]),
    ("velocity_mps", "速度档", lambda case: case["velocity_mps"]),
    ("cart_mass_kg", "质量档", lambda case: case["cart_mass_kg"]),
    ("connection", "连接类型", lambda case: case["connection"]),
)
#: 指令侧诊断列（`stop.*`，2026-10-11 新增）
COMMAND_FIELDS = (
    ("loco_cmd_vx_coast_mean_mps", "STOP 后合成指令 vx"),
    ("upper_offset_vx_coast_mean_mps", "STOP 后生效偏移"),
    ("upper_cmd_action_coast_mean", "STOP 后偏移头动作"),
    ("loco_cmd_vx_tow_mean_mps", "牵引段合成指令 vx"),
)
ESTIMATOR_ITEMS = ("vx_mps", "vy_mps", "mass_kg", "force_norm_n")
PHASES = ("station", "tow", "coast")


def stop_block(case):
    return ((case.get("metrics") or {}).get("stop") or {})


def deck_peak(case):
    value = stop_block(case).get("deck_contact_peak_n")
    return float(value) if isinstance(value, (int, float)) else 0.0


def min_clearance(case):
    value = stop_block(case).get("min_clearance_coast_m")
    return float(value) if isinstance(value, (int, float)) else None


def is_hard(case) -> bool:
    """硬证据碰撞：甲板接触力 > 0 或停车段最小间隙 ≤ 0（与判定码无关）。"""
    if deck_peak(case) > 0.0:
        return True
    clearance = min_clearance(case)
    return clearance is not None and clearance <= 0.0


def index_by_cell(cases):
    table = {}
    for case in cases:
        cell = case["case"]
        table[int(cell["cell_index"])] = case
    return table


def _rate(cases):
    if not cases:
        return None
    return sum(1 for case in cases if is_hard(case)) / len(cases) * 100.0


def _median(values):
    values = [value for value in values if isinstance(value, (int, float))]
    return statistics.median(values) if values else None


def macro_rate(cases, cells):
    """宏平均：按 (速度, 质量) 分格，每格等权（去掉两轮成分差异）。"""
    keys = sorted({(case["case"]["velocity_mps"], case["case"]["cart_mass_kg"]) for case in cells})
    rates = []
    for velocity, mass in keys:
        group = [case for case in cases
                 if case["case"]["velocity_mps"] == velocity
                 and case["case"]["cart_mass_kg"] == mass]
        rate = _rate(group)
        if rate is not None:
            rates.append(rate)
    return statistics.mean(rates) if rates else None


def paired_arguments(baseline_payload, policy_payload):
    """返回 [(参数, 基线值, 策略值)]，只列**不一致**的物理参数（上层开关允许不同）。"""
    left = baseline_payload.get("arguments") or {}
    right = policy_payload.get("arguments") or {}
    mismatches = []
    for name in PAIRED_ARGUMENTS:
        if name == "terrain_profile":
            # Historical reports predate the switch and always used the hill surface.
            if left.get(name, "hill") != right.get(name, "hill"):
                mismatches.append((name, left.get(name, "hill"), right.get(name, "hill")))
            continue
        if name not in left and name not in right:
            continue
        if left.get(name) != right.get(name):
            mismatches.append((name, left.get(name), right.get(name)))
    return mismatches


def flip_table(baseline_cells, policy_cells, shared):
    """逐 cell 翻转：返回 (都撞, 仅基线撞=救回, 仅策略撞=弄坏, 都不撞)。"""
    both = saved = broken = neither = 0
    for cell in shared:
        left = is_hard(baseline_cells[cell])
        right = is_hard(policy_cells[cell])
        if left and right:
            both += 1
        elif left:
            saved += 1
        elif right:
            broken += 1
        else:
            neither += 1
    return both, saved, broken, neither


def _sorted_values(values):
    """数值轴按数值排（否则 row 会排成 0,1,10,11,…,2），字符串轴按字符串排。"""
    values = list(values)
    if all(isinstance(value, (int, float)) for value in values):
        return sorted(values)
    return sorted(values, key=lambda value: str(value))


def axis_rows(baseline_cells, policy_cells, shared, key, label, getter):
    values = _sorted_values({getter(baseline_cells[cell]["case"]) for cell in shared})
    rows = []
    for value in values:
        cells = [cell for cell in shared if getter(baseline_cells[cell]["case"]) == value]
        left = [baseline_cells[cell] for cell in cells]
        right = [policy_cells[cell] for cell in cells]
        both, saved, broken, neither = flip_table(baseline_cells, policy_cells, cells)
        rows.append((label, value, len(cells), _rate(left), _rate(right), saved, broken))
    return rows


def command_diagnostics(cases):
    """指令侧新列的分布（策略轮）；基线轮的这些列按定义恒 0。"""
    out = {}
    for field, label in COMMAND_FIELDS:
        values = [stop_block(case).get(field) for case in cases]
        values = [float(value) for value in values if isinstance(value, (int, float))]
        out[label] = (statistics.median(values), len(values)) if values else (None, 0)
    by_velocity = {}
    for velocity in sorted({case["case"]["velocity_mps"] for case in cases}):
        group = [case for case in cases if case["case"]["velocity_mps"] == velocity]
        means = [stop_block(case).get("loco_cmd_vx_coast_mean_mps") for case in group]
        means = [float(value) for value in means if isinstance(value, (int, float))]
        if not means:
            continue
        negative = sum(1 for value in means if value < -0.02)
        by_velocity[velocity] = (statistics.median(means), negative / len(means) * 100.0, len(means))
    return out, by_velocity


def estimator_diagnostics(cases):
    """`metrics["estimator"]["abs_err"]` 的中位绝对误差（策略轮，需 `--dump-latent`）。"""
    rows = {}
    for phase in PHASES:
        per_item = {}
        for item in ESTIMATOR_ITEMS:
            values = []
            for case in cases:
                block = (((case.get("metrics") or {}).get("estimator") or {}).get("abs_err") or {}).get(phase)
                if block and isinstance(block.get(item), (int, float)):
                    values.append(float(block[item]))
            per_item[item] = statistics.median(values) if values else None
        rows[phase] = per_item
    return rows


def has_latent(cases):
    return any(((case.get("metrics") or {}).get("latent") or {}).get("tow") for case in cases)


def report(baseline_payload, policy_payload, stream=sys.stdout, labels=("基线", "策略")):
    baseline_cases = baseline_payload.get("cases") or []
    policy_cases = policy_payload.get("cases") or []
    baseline_cells = index_by_cell(baseline_cases)
    policy_cells = index_by_cell(policy_cases)
    shared = sorted(set(baseline_cells) & set(policy_cells))

    def out(line=""):
        print(line, file=stream)

    out(f"# 基线 {labels[0]}  （{len(baseline_cases)} case）")
    out(f"# 策略 {labels[1]}  （{len(policy_cases)} case）")
    mismatches = paired_arguments(baseline_payload, policy_payload)
    if mismatches:
        out("# ⚠ 两轮的物理参数不一致（这样比出来不是配对差）：")
        for name, left, right in mismatches:
            out(f"#     {name}: 基线 {left!r} ≠ 策略 {right!r}")
    else:
        physics = "\n#     ".join(f"{name} = {(baseline_payload.get('arguments') or {}).get(name)!r}"
                                 for name in ("wheel_damping", "ground_friction", "velocities", "cart_masses"))
        out("# 物理参数一致 ✓：")
        out(f"#     {physics}")
    out(f"# 可按 cell 配对：{len(shared)} 格"
        f"（基线独有 {len(set(baseline_cells) - set(shared))}、策略独有 {len(set(policy_cells) - set(shared))}）")
    out()
    if not shared:
        out("没有可配对的 cell，无法比较。")
        return 1

    both, saved, broken, neither = flip_table(baseline_cells, policy_cells, shared)
    out("## 1. 头条（硬证据 = 甲板力>0 或 间隙≤0；与判定码无关）")
    out(f"  逐 cell 配对 n = {len(shared)}")
    left = _rate([baseline_cells[cell] for cell in shared])
    right = _rate([policy_cells[cell] for cell in shared])
    out(f"  基线   硬证据 {left:5.1f}%   （宏平均 {macro_rate([baseline_cells[c] for c in shared], [baseline_cells[c] for c in shared]):5.1f}%）")
    out(f"  策略   硬证据 {right:5.1f}%   （宏平均 {macro_rate([policy_cells[c] for c in shared], [policy_cells[c] for c in shared]):5.1f}%）")
    delta = right - left
    extra = ""
    if right > 0 and left > 0:
        extra = f"（相对 {left / right:.2f}× 更少，策略/基线 = {right / left:.2f}）"
    elif right == 0 and left > 0:
        extra = "（策略轮 0 例硬碰撞）"
    out(f"  delta  {delta:+5.1f} 个百分点 {extra}")
    baseline_ok = sum(1 for cell in shared if baseline_cells[cell]["verdict"]["code"] == "OK")
    policy_ok = sum(1 for cell in shared if policy_cells[cell]["verdict"]["code"] == "OK")
    out(f"  【参考】判定码通过：基线 {baseline_ok}/{len(shared)}、策略 {policy_ok}/{len(shared)}"
        f"（代理通道 `load_velocity_jump` 会把改善读反，见文档 §8.1）")
    out()
    out("## 2. 逐 cell 翻转表（最关键的一行）")
    out(f"  两轮都撞      {both:4d}")
    out(f"  **只有基线撞（救回）** {saved:4d}")
    out(f"  **只有策略撞（弄坏）** {broken:4d}")
    out(f"  都不撞        {neither:4d}")
    total = both + saved + broken + neither
    out(f"  ⇒ 净救回 {saved - broken:+d} 格（救回 {saved / total * 100:.1f}% / 弄坏 {broken / total * 100:.1f}%）")
    out()
    out("## 3. 按轴 delta（硬证据率；救回/弄坏 = 该轴内的翻转计数）")
    for key, label, getter in AXES:
        out(f"  [{label}]")
        out(f"    {'值':>12s} {'n':>5s} {'基线':>8s} {'策略':>8s} {'delta':>8s} {'救回':>5s} {'弄坏':>5s}")
        for _, value, count, base, pol, saved_n, broken_n in axis_rows(
                baseline_cells, policy_cells, shared, key, label, getter):
            out(f"    {str(value):>12s} {count:>5d} {base:>7.1f}% {pol:>7.1f}% {pol - base:>+7.1f}pt"
                f" {saved_n:>5d} {broken_n:>5d}")
    out()
    out("## 4. 指令侧与 decoder（策略轮）")
    stats, by_velocity = command_diagnostics([policy_cells[cell] for cell in shared])
    for label, (value, count) in stats.items():
        out(f"  {label:24s} 中位 {'—' if value is None else format(value, '+.4f')}  (n={count})")
    if by_velocity:
        out("  按速度档：STOP 后合成指令 vx 中位 / 负偏移占比")
        for velocity, (value, negative, count) in by_velocity.items():
            out(f"    {velocity:>4.1f} m/s  {value:+.4f}  负偏移 {negative:5.1f}%  (n={count})")
    if has_latent([policy_cells[cell] for cell in shared]):
        rows = estimator_diagnostics([policy_cells[cell] for cell in shared])
        out("  decoder 显式估计中位绝对误差（需 --dump-latent；latent 聚类交给 analyze_latent.py）：")
        for phase in PHASES:
            cells_text = " ".join(f"{item}={rows[phase][item]:.4f}" if rows[phase][item] is not None else f"{item}=—"
                                  for item in ESTIMATOR_ITEMS)
            out(f"    {phase:>8s}  {cells_text}")
    else:
        out("  （这一轮没有 latent/estimator 数据；要看得用 `--dump-latent` 重跑）")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="同条件两轮 play_towing_test 的逐 cell 配对对照")
    parser.add_argument("--baseline", type=Path, required=True, help="基线轮 report.json")
    parser.add_argument("--policy", type=Path, required=True, help="策略轮 report.json（只应差 --upper-checkpoint）")
    args = parser.parse_args(argv)
    for name in ("baseline", "policy"):
        path = getattr(args, name)
        if not path.is_file():
            parser.error(f"--{name} 指向的文件不存在：{path}")
    return report(json.loads(args.baseline.read_text(encoding="utf-8")),
                  json.loads(args.policy.read_text(encoding="utf-8")),
                  labels=(str(args.baseline.parent), str(args.policy.parent)))


if __name__ == "__main__":
    sys.exit(main())
