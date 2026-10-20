#!/usr/bin/env python3
"""拖曳任务的**收束后验收口径**（2026-10-19，纯标准库、离线可测）。

## 为什么单独一层

`play_towing_test.py` 早先的判定码（`OK/SPD/COL/LOW/LAT/...`）是「上层任务必要性」的
基线口径：它把**停车是否追尾**、**停车几何间隙**、**横向保持**、**关节静差**都算进
通过/失败。2026-10-19 用户把问题收束成三件主事：

1. **拉小车：速度跟随稳定性**（只对小车档；轻载时上层的主要职责是把速度跟稳）；
2. **启停：只看冲击**（起拖、绷直、指令归零、停车撞击四个窗口的关节冲击/力矩饱和；
   **不再要求机器人停车后刻意往前走几步**，也不再把「小车追上来了」本身当失败）；
3. **拉重车：不翻车**（重载时上层的主要职责是安全，速度跟踪退为观测）。

同日第二轮，用户明确「**横向保持和关节静差也还是要作为失败**」⇒ 旧 `LAT`/`JNT` 两项
**恢复为任务失败项**，只是不再用旧 `verdict` 的统计口径。停车追尾/间隙仍只作诊断。

因此本模块在**原始指标之上**加一层 `task` 判定，不改任何观测字段、不改仿真：

- `metrics["task"]` 逐 case 落盘；
- `report.md` / `report.json` 的「任务口径」一节以它为准；
- 旧 `verdict`（`OK/SPD/COL/LOW/LAT/JNT/...`）保留，只作追溯与诊断，**不再作为验收结论**。

## 五条检查

| 检查 | 适用 | 通过条件 |
|---|---|---|
| `speed_tracking` | 小车档（`cart_mass_kg <= task_light_max_kg`）且拖车 | 稳态窗跟速 MAE、速度标准差、|误差| p95 都 ≤ 各自比例 × 指令 |
| `lane_keeping` | `--lane-keeping pd` 且拖车 | `y_max_abs_m ≤ lane_y_limit_m` 且 `heading_max_abs_rad ≤ lane_heading_limit_deg` |
| `joint_static_error` | 所有拖车 case | `startup`/`stop` 两窗口的 `joint_rms_rad ≤ joint_rms_limit_rad` 且 `joint_max_rad ≤ joint_max_limit_rad`（旧 JNT 口径） |
| `start_stop_impact` | 所有拖车 case | 起拖/绷直/归零/停车撞击四个窗口的 `RMS/稳态`、`RMS−稳态`、力矩饱和占比都不超阈值 |
| `no_fall` | 所有拖车 case（重车档是主判据） | `stability.fell` 为假，且由四元数解出的 `|roll|` 不超限 |

「任务通过」= 该 case 的**全部适用检查**通过。小车档 = 速度 + 横向 + 关节 + 冲击 + 不跌倒；
重车档 = 横向 + 关节 + 冲击 + 不翻车（速度只作观测）。**不适用**的检查记 `N/A`，不算失败。

## 与旧口径的映射（写进报告，避免读者混读）

- `COL`（追尾接触）/ `LOW`（停车间隙低）→ **不再是失败**；停车撞击的**冲击量**进
  `start_stop_impact`，接触本身只作诊断（`metrics.impact.stop_contact.*`、`metrics.stop.*`）。
- `LAT`（横向/朝向）→ **恢复为失败**，进 `lane_keeping`；`--lane-keeping off` 时 N/A。
- `JNT`（关节静差）→ **恢复为失败**，进 `joint_static_error`；它与
  `start_stop_impact` 分开：前者是 `q − q*` 的静差/跟踪误差，后者是相对稳态的冲击。
- `SPD`（全程跟速）→ 只对小车档、且改用**稳态窗**（后一半牵引段）判「跟随稳定性」；
  重车档的速度跟踪只作观测。
- `FALL` → 保留并加强：在原有 `fell` 之外加 roll 判据（重车不翻车）。

## 阈值来源

`DEFAULT_TASK_THRESHOLDS` 是**未标定占位值**，与旧 `DEFAULT_THRESHOLDS` **分开**存放
（旧阈值被归档对照依赖，不能改字面量）。实跑后应按分布重新标定，并在
`docs/towing/towing_eval_criteria_2026-10-19.md` 记录依据。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

#: 口径版本（写进 report.json / experiment.json；改语义必须升版本）。
TASK_VERSION = "towing_task_2026-10-19_v2"

#: 任务判定码（严重度从低到高；`NA` = 不适用，不参与通过率）。
TASK_CODES = ("NA", "PASS", "SPEED_UNSTABLE", "JOINT_ERROR", "LANE_DEVIATION",
              "IMPACT_HIGH", "FALL", "INVALID")
TASK_SEVERITY = {code: index for index, code in enumerate(TASK_CODES)}

#: 矩阵里用的短标签（完整码在 JSON 里）。
TASK_MATRIX_LABELS = {
    "NA": "NA",
    "PASS": "PASS",
    "SPEED_UNSTABLE": "SPD",
    "JOINT_ERROR": "JNT",
    "LANE_DEVIATION": "LAT",
    "IMPACT_HIGH": "IMP",
    "FALL": "FALL",
    "INVALID": "INV",
}

#: 检查顺序 = 报告顺序。2026-10-19 第二轮：用户明确「横向保持和关节静差也还是要
#: 作为失败」⇒ 这两项从旧口径的观测项恢复为任务失败项（`LAT`/`JNT`）。
TASK_CHECKS = ("speed_tracking", "lane_keeping", "joint_static_error",
               "start_stop_impact", "no_fall")
TASK_CHECK_LABELS = {
    "speed_tracking": "小车速度跟随稳定性",
    "lane_keeping": "横向/朝向保持",
    "joint_static_error": "关节静差",
    "start_stop_impact": "启停冲击",
    "no_fall": "不翻车/不跌倒",
}

#: 质量分档：默认「小车 = ≤ 15 kg，重车 = > 15 kg」。默认质量档
#: `10/15/20/25/30 kg` ⇒ 10/15 为小车、20/25/30 为重车；10/20/30 的实测网格也落在
#: 「10 小车、20/30 重车」上，与用户 2026-10-19 的三条口径一致。
DEFAULT_TASK_THRESHOLDS = {
    "task_light_max_kg": 15.0,
    # 小车速度跟随稳定性（全部是「占指令的比例」；稳态窗 = 牵引段后一半）。
    "speed_std_ratio_limit": 0.10,
    "speed_p95_ratio_limit": 0.40,
    # 启停冲击：窗口关节 RMS 相对稳态的比值、绝对增量、力矩饱和步占比。
    "impact_rms_ratio_limit": 1.50,
    "impact_rms_delta_limit_rad": 0.10,
    "impact_torque_sat_frac_limit": 0.05,
    # 「翻车」的侧倾上限（rad）；原有 `fell` 仍由俯仰/离面高度给出。
    "roll_limit_rad": 0.80,
}

#: 旧口径里跟速/横向/关节阈值仍在 `play_towing_test.DEFAULT_THRESHOLDS`；本模块不复制
#: 字面量，由 `resolve_thresholds()` 合并进来。独立 CLI 复判时用这些默认值。
TASK_LEGACY_THRESHOLD_DEFAULTS = {
    "speed_mae_ratio_limit": 0.20,
    "lane_y_limit_m": 0.30,
    "lane_heading_limit_deg": 10.0,
    "joint_rms_limit_rad": 0.10,
    "joint_max_limit_rad": 0.30,
}
DEFAULT_SPEED_MAE_RATIO_LIMIT = TASK_LEGACY_THRESHOLD_DEFAULTS["speed_mae_ratio_limit"]


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(float(value))


def mass_band(cart_mass_kg, light_max_kg: float = DEFAULT_TASK_THRESHOLDS["task_light_max_kg"]):
    """质量分档：`light` / `heavy`；质量缺失返回 `None`。"""
    if not _finite(cart_mass_kg):
        return None
    return "light" if float(cart_mass_kg) <= float(light_max_kg) else "heavy"


def band_label(band: str) -> str:
    return {"light": "小车", "heavy": "重车"}.get(band, str(band))


def resolve_thresholds(legacy_thresholds=None, task_thresholds=None) -> dict:
    """把旧速度/横向/关节阈值与任务阈值合成一份（不修改入参）。

    合并顺序：模块默认 → 旧 `DEFAULT_THRESHOLDS`（若提供）→ 任务阈值（显式覆盖）。
    `task_thresholds` 只接受已知键，防止拼写错误静默生效。
    """
    merged = {**TASK_LEGACY_THRESHOLD_DEFAULTS, **DEFAULT_TASK_THRESHOLDS}
    allowed = set(TASK_LEGACY_THRESHOLD_DEFAULTS) | set(DEFAULT_TASK_THRESHOLDS)
    if isinstance(legacy_thresholds, dict):
        merged.update({key: value for key, value in legacy_thresholds.items()
                       if key in allowed and _finite(value)})
    if isinstance(task_thresholds, dict):
        merged.update({key: value for key, value in task_thresholds.items()
                       if key in allowed and _finite(value)})
    return merged


def _na(reason: str) -> dict:
    return {"applicable": False, "ok": None, "reasons": [], "values": {}, "note": reason}


def _check(*, applicable: bool, reasons, values) -> dict:
    reasons = list(reasons)
    return {"applicable": bool(applicable), "ok": (not reasons) if applicable else None,
            "reasons": reasons, "values": values}


def _speed_check(metrics: dict, band: str | None, thresholds: dict) -> dict:
    speed = metrics.get("speed") or {}
    if band != "light":
        return _na("重车档不判速度跟随（只作观测）")
    if not speed.get("available") or not _finite(metrics.get("case", {}).get("velocity_mps")):
        return _na("没有可用的跟速记录")
    command = float(metrics["case"]["velocity_mps"])
    if command <= 0.0:
        return _na("速度指令不是正数")
    # 稳态窗优先。**旧产物**（2026-10-19 之前）没有 `steady_speed_std_mps` /
    # `steady_abs_err_p95_mps`：这时只用 `steady_mae_mps` 判，std/p95 记 None（跳过），
    # 不退回到全程 `p95_abs_err_mps`（那会把起步瞬态算进「跟随稳定性」）。
    mae = speed.get("steady_mae_mps")
    window = "steady"
    if mae is None:
        mae = speed.get("mae_mps")
        window = "full_tow_fallback"
    std = speed.get("steady_speed_std_mps")
    p95 = speed.get("steady_abs_err_p95_mps")
    reasons, values = [], {
        "window": window,
        "command_mps": command,
        "steady_mae_mps": mae,
        "steady_mae_ratio": (float(mae) / command) if _finite(mae) else None,
        "steady_speed_std_mps": std,
        "steady_speed_std_ratio": (float(std) / command) if _finite(std) else None,
        "steady_abs_err_p95_mps": p95,
        "steady_abs_err_p95_ratio": (float(p95) / command) if _finite(p95) else None,
        "mae_ratio_limit": thresholds["speed_mae_ratio_limit"],
        "std_ratio_limit": thresholds["speed_std_ratio_limit"],
        "p95_ratio_limit": thresholds["speed_p95_ratio_limit"],
    }
    if _finite(mae) and float(mae) > thresholds["speed_mae_ratio_limit"] * command:
        reasons.append("speed_mae_high")
    if _finite(std) and float(std) > thresholds["speed_std_ratio_limit"] * command:
        reasons.append("speed_unstable")
    if _finite(p95) and float(p95) > thresholds["speed_p95_ratio_limit"] * command:
        reasons.append("speed_dropout")
    return _check(applicable=True, reasons=reasons, values=values)




def _lane_check(metrics: dict, band: str | None, thresholds: dict) -> dict:
    """横向/朝向保持（旧 `LAT`，2026-10-19 第二轮恢复为失败项）。

    `--lane-keeping off` 时没有横向控制，记 N/A（不能把「没做」判成失败）；
    数据缺失也记 N/A，不伪造 0。
    """
    lane = metrics.get("lane") or {}
    if not lane.get("available") or lane.get("mode") == "off":
        return _na("横向保持未启用或没有记录")
    y_max = lane.get("y_max_abs_m")
    heading_max = lane.get("heading_max_abs_rad")
    if not _finite(y_max) and not _finite(heading_max):
        return _na("横向保持没有可用读数")
    y_limit = thresholds["lane_y_limit_m"]
    heading_limit = math.radians(thresholds["lane_heading_limit_deg"])
    reasons = []
    if _finite(y_max) and float(y_max) > y_limit:
        reasons.append("lane_deviation")
    if _finite(heading_max) and float(heading_max) > heading_limit:
        if "lane_deviation" not in reasons:
            reasons.append("lane_deviation")
    return _check(applicable=True, reasons=reasons, values={
        "y_max_abs_m": y_max,
        "y_limit_m": y_limit,
        "heading_max_abs_rad": heading_max,
        "heading_limit_rad": heading_limit,
        "y_rms_m": lane.get("y_rms_m"),
        "heading_rms_rad": lane.get("heading_rms_rad"),
        "mode": lane.get("mode"),
    })


def _joint_check(metrics: dict, band: str | None, thresholds: dict) -> dict:
    """关节静差（旧 `JNT`，2026-10-19 第二轮恢复为失败项）。

    口径与旧 `classify_case(count_jnt=True)` 一致：`startup` / `stop` 两个窗口的
    `q − q*` 的 RMS ≤ `joint_rms_limit_rad`、单关节峰值 ≤ `joint_max_limit_rad`。
    它是**静差/跟踪误差**，与 `start_stop_impact` 的「相对稳态冲击」是两件事。
    """
    startup = metrics.get("startup") or {}
    stop = metrics.get("stop") or {}
    reasons, values = [], {}
    for label, section in (("startup", startup), ("stop", stop)):
        rms = section.get("joint_rms_rad")
        peak = section.get("joint_max_rad")
        if not _finite(rms) and not _finite(peak):
            values[label] = {"available": False}
            continue
        bad = []
        if _finite(rms) and float(rms) > thresholds["joint_rms_limit_rad"]:
            bad.append("rms")
        if _finite(peak) and float(peak) > thresholds["joint_max_limit_rad"]:
            bad.append("max")
        values[label] = {"available": True, "joint_rms_rad": rms,
                         "joint_max_rad": peak, "failed": bad}
        if bad:
            reasons.append(f"{label}_joint_error")
    if not any(value.get("available") for value in values.values()):
        return _na("没有关节静差记录")
    return _check(applicable=True, reasons=reasons, values={
        "windows": values,
        "joint_rms_limit_rad": thresholds["joint_rms_limit_rad"],
        "joint_max_limit_rad": thresholds["joint_max_limit_rad"],
    })


def _impact_check(metrics: dict, band: str | None, thresholds: dict) -> dict:
    impact = metrics.get("impact") or {}
    if band is None:
        return _na("没有质量档，无法归入任务口径")
    windows = {}
    reasons, failed_windows = [], []
    for group in ("startup", "takeup", "stop", "stop_contact"):
        block = impact.get(group) or {}
        if not block.get("available"):
            continue
        ratio = block.get("rms_over_steady")
        delta = block.get("rms_delta_rad")
        saturated = block.get("torque_saturated_frac")
        bad = []
        if _finite(ratio) and float(ratio) > thresholds["impact_rms_ratio_limit"]:
            bad.append("rms_ratio")
        if _finite(delta) and float(delta) > thresholds["impact_rms_delta_limit_rad"]:
            bad.append("rms_delta")
        if _finite(saturated) and float(saturated) > thresholds["impact_torque_sat_frac_limit"]:
            bad.append("torque_saturation")
        windows[group] = {
            "joint_rms_rad": block.get("joint_rms_rad"),
            "rms_over_steady": ratio,
            "rms_delta_rad": delta,
            "torque_saturated_frac": saturated,
            "failed": bad,
        }
        if bad:
            failed_windows.append(group)
    if not windows:
        return _na("没有可用的冲击窗口（旧产物或空轨迹）")
    if failed_windows:
        reasons.append("start_stop_impact")
    return _check(applicable=True, reasons=reasons, values={
        "windows": windows,
        "failed_windows": failed_windows,
        "rms_ratio_limit": thresholds["impact_rms_ratio_limit"],
        "rms_delta_limit_rad": thresholds["impact_rms_delta_limit_rad"],
        "torque_sat_frac_limit": thresholds["impact_torque_sat_frac_limit"],
    })


def _fall_check(metrics: dict, band: str | None, thresholds: dict) -> dict:
    stability = metrics.get("stability") or {}
    if band is None:
        return _na("没有质量档，无法归入任务口径")
    if not stability:
        return _na("没有稳定性记录")
    reasons = []
    if stability.get("fell"):
        reasons.append("robot_fell")
    roll = stability.get("max_abs_roll_rad")
    if _finite(roll) and abs(float(roll)) > thresholds["roll_limit_rad"]:
        reasons.append("robot_rollover")
    return _check(applicable=True, reasons=reasons, values={
        "fell": bool(stability.get("fell")),
        "fell_phase": stability.get("fell_phase"),
        "max_abs_roll_rad": roll,
        "roll_limit_rad": thresholds["roll_limit_rad"],
        "min_robot_surface_height_m": stability.get("min_robot_surface_height_m"),
        "max_abs_pitch_rel_rad": stability.get("max_abs_pitch_rel_rad"),
    })


def evaluate_task(metrics: dict, *, legacy_thresholds=None, task_thresholds=None) -> dict:
    """按收束后的五条检查给一个 case 判 `PASS/SPD/JNT/LAT/IMP/FALL/INV/NA`。

    `metrics` 是 `play_towing_test.compute_case_metrics()` 的返回值（或 `report.json` 里
    同一结构）。函数**只读**指标，不改任何字段，也不依赖 torch / Isaac Sim。
    """
    thresholds = resolve_thresholds(legacy_thresholds, task_thresholds)
    case = metrics.get("case") or {}
    stability = metrics.get("stability") or {}
    band = mass_band(case.get("cart_mass_kg"), thresholds["task_light_max_kg"])
    cart_present = bool(metrics.get("cart_present", case.get("cart_present", True)))

    if not cart_present:
        return {
            "version": TASK_VERSION, "band": band, "applicable": False,
            "code": "NA", "ok": None, "reasons": [],
            "checks": {name: _na("无小车 env，任务口径不适用") for name in TASK_CHECKS},
            "thresholds": thresholds,
            "note": "无小车参考 env：不参与任务通过率",
        }

    checks = {
        "speed_tracking": _speed_check(metrics, band, thresholds),
        "lane_keeping": _lane_check(metrics, band, thresholds),
        "joint_static_error": _joint_check(metrics, band, thresholds),
        "start_stop_impact": _impact_check(metrics, band, thresholds),
        "no_fall": _fall_check(metrics, band, thresholds),
    }
    reasons = []
    if stability.get("invalid"):
        reasons.append("invalid_record")
    for name in TASK_CHECKS:
        check = checks[name]
        if check.get("applicable") and check.get("ok") is False:
            reasons.extend(check["reasons"])

    if "invalid_record" in reasons:
        code = "INVALID"
    elif any(reason in ("robot_fell", "robot_rollover") for reason in reasons):
        code = "FALL"
    elif "start_stop_impact" in reasons:
        code = "IMPACT_HIGH"
    elif "lane_deviation" in reasons:
        code = "LANE_DEVIATION"
    elif any(reason.endswith("_joint_error") for reason in reasons):
        code = "JOINT_ERROR"
    elif any(reason.startswith("speed_") for reason in reasons):
        code = "SPEED_UNSTABLE"
    else:
        code = "PASS"

    applicable = any(check.get("applicable") for check in checks.values())
    return {
        "version": TASK_VERSION,
        "band": band,
        "band_label": band_label(band),
        "applicable": bool(applicable),
        "code": code if applicable else "NA",
        "ok": (code == "PASS") if applicable else None,
        "reasons": reasons if applicable else [],
        "checks": checks,
        "thresholds": thresholds,
        "note": ("任务通过 = 全部适用检查通过；速度只对小车档，横向/关节/冲击/不翻车"
                 "对所有拖车 case" if applicable else "没有可适用的任务检查"),
    }


def summarize_task(case_summaries) -> dict:
    """把逐 case 的 `metrics.task` 汇总成「小车/重车 × 五条检查」的通过率。

    只统计带 `task` 字段的记录（旧 `report.json` 没有该字段 ⇒ `with_task=0`），
    不会把旧产物误算成通过或失败。
    """
    codes, reasons = {}, {}
    bands = {
        "light": {"cases": 0, "applicable": 0, "pass": 0,
                  "checks": {name: {"applicable": 0, "pass": 0, "fail": 0} for name in TASK_CHECKS}},
        "heavy": {"cases": 0, "applicable": 0, "pass": 0,
                  "checks": {name: {"applicable": 0, "pass": 0, "fail": 0} for name in TASK_CHECKS}},
    }
    grades, records, with_task, applicable, passed = {}, 0, 0, 0, 0
    for summary in case_summaries or []:
        metrics = summary.get("metrics", summary) if isinstance(summary, dict) else {}
        task = (metrics.get("task") or {}) if isinstance(metrics, dict) else {}
        records += 1
        if not task.get("version"):
            continue
        with_task += 1
        code = task.get("code") or "NA"
        codes[code] = codes.get(code, 0) + 1
        for reason in task.get("reasons") or []:
            reasons[reason] = reasons.get(reason, 0) + 1
        if task.get("applicable"):
            applicable += 1
            if task.get("ok"):
                passed += 1
        band = task.get("band")
        if band in bands:
            entry = bands[band]
            entry["cases"] += 1
            if task.get("applicable"):
                entry["applicable"] += 1
                if task.get("ok"):
                    entry["pass"] += 1
            for name in TASK_CHECKS:
                check = (task.get("checks") or {}).get(name) or {}
                if check.get("applicable"):
                    entry["checks"][name]["applicable"] += 1
                    if check.get("ok"):
                        entry["checks"][name]["pass"] += 1
                    else:
                        entry["checks"][name]["fail"] += 1
        case = metrics.get("case") or {}
        grade = case.get("grade_deg")
        if _finite(grade):
            key = f"grade{float(grade):g}"
            entry = grades.setdefault(key, {"grade_deg": float(grade), "cases": 0,
                                            "applicable": 0, "pass": 0})
            entry["cases"] += 1
            if task.get("applicable"):
                entry["applicable"] += 1
                if task.get("ok"):
                    entry["pass"] += 1
    return {
        "version": TASK_VERSION,
        "records": records,
        "with_task": with_task,
        "applicable": applicable,
        "pass": passed,
        "codes": codes,
        "reasons": reasons,
        "bands": bands,
        "grades": grades,
    }


def task_conclusion_lines(summary: dict) -> list:
    """`report.md` 的任务结论行（纯文本，不含 markdown 标题）。"""
    if not summary or not summary.get("with_task"):
        return ["- 本轮产物没有 `metrics.task` 字段（旧版本报告）⇒ 任务口径不可用；"
                "重跑或对旧 `report.json` 用 `python3 imgo2_rl/scripts/towing/towing_task_criteria.py` 复判。"]
    lines = []
    total, ok = summary["applicable"], summary["pass"]
    lines.append(f"- 总任务通过 **{ok}/{total}**（可判 case；无小车锚点不计）；"
                 f"判定码 " + "、".join(f"{code} {count}" for code, count in
                                       sorted(summary["codes"].items(), key=lambda item: -item[1])))
    for band in ("light", "heavy"):
        entry = summary["bands"][band]
        if not entry["cases"]:
            continue
        checks = entry["checks"]
        lane = (f"横向保持 {checks['lane_keeping']['pass']}/"
                f"{checks['lane_keeping']['applicable']}")
        joint = (f"关节静差 {checks['joint_static_error']['pass']}/"
                 f"{checks['joint_static_error']['applicable']}")
        impact = (f"启停冲击 {checks['start_stop_impact']['pass']}/"
                  f"{checks['start_stop_impact']['applicable']}")
        fall = (f"不跌倒 {checks['no_fall']['pass']}/"
                f"{checks['no_fall']['applicable']}")
        if band == "light":
            detail = (f"速度跟随稳定 {checks['speed_tracking']['pass']}/"
                      f"{checks['speed_tracking']['applicable']}、{lane}、{joint}、"
                      f"{impact}、{fall}")
        else:
            detail = (f"{lane}、{joint}、{impact}、"
                      f"不翻车 {checks['no_fall']['pass']}/"
                      f"{checks['no_fall']['applicable']}"
                      f"（速度跟踪只作观测，不判）")
        lines.append(f"- **{band_label(band)}档**：任务通过 **{entry['pass']}/{entry['applicable']}**；"
                     f"{detail}。")
    if summary["reasons"]:
        lines.append("- 失败原因：" + "、".join(
            f"{reason}×{count}" for reason, count in
            sorted(summary["reasons"].items(), key=lambda item: -item[1])) + "。")
    lines.append("- **横向保持（`LAT`）与关节静差（`JNT`）仍是失败项**（用户 2026-10-19 第二轮明确）；"
                 "`COL`（追尾接触）/`LOW`（停车间隙）仍只作诊断，停车撞击只看 "
                 "`impact.stop_contact` 的冲击量，不要求停车后刻意往前走。")
    return lines


def format_task_matrix(case_summaries, *, grades) -> str:
    """按坡度量级打印「质量 × (速度/连接)」的任务判定矩阵（短标签 + env 数）。"""
    blocks = []
    for grade in grades:
        cases = [summary["metrics"] for summary in case_summaries
                 if float((summary.get("metrics", {}).get("case") or {}).get("grade_deg", 0.0))
                 == float(grade)]
        if not cases:
            continue
        velocities = sorted({case["case"]["velocity_mps"] for case in cases
                             if (case.get("case") or {}).get("velocity_mps") is not None})
        connections = [name for name in ("compliant", "inextensible", "rigid")
                       if any((case.get("case") or {}).get("connection") == name for case in cases)]
        masses = sorted({case["case"]["cart_mass_kg"] for case in cases
                         if (case.get("case") or {}).get("cart_mass_kg") is not None})
        header = "| 质量 kg | " + " | ".join(
            f"{velocity:g} m/s {connection}" for velocity in velocities
            for connection in connections) + " |"
        divider = "|" + "---|" * (len(velocities) * len(connections) + 1)
        lines = [f"### 坡度量级 {('平地' if float(grade) == 0.0 else f'{float(grade):g}° 坡')}"
                 f"（{len(cases)} env）", "", header, divider]
        index = {}
        for metrics in cases:
            case = metrics.get("case") or {}
            task = metrics.get("task") or {}
            key = (case.get("velocity_mps"), case.get("connection"), case.get("cart_mass_kg"))
            index.setdefault(key, []).append(task.get("code") or "NA")
        for mass in masses:
            cells = []
            for velocity in velocities:
                for connection in connections:
                    codes = index.get((velocity, connection, mass), [])
                    if not codes:
                        cells.append("-")
                        continue
                    worst = max(codes, key=lambda code: TASK_SEVERITY.get(code, 0))
                    cells.append(f"{TASK_MATRIX_LABELS.get(worst, worst)}×{len(codes)}")
            lines.append(f"| {mass:g} | " + " | ".join(cells) + " |")
        lines.append("")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def format_task_summary_table(summary: dict, *, light_max_kg=None) -> list:
    """`report.md` 的「档 × 检查 × 通过率」表（纯 markdown 行）。"""
    if not summary or not summary.get("with_task"):
        return []
    light_max = (DEFAULT_TASK_THRESHOLDS["task_light_max_kg"]
                 if light_max_kg is None else float(light_max_kg))

    def cell(entry, name):
        check = entry["checks"][name]
        if not check["applicable"]:
            return "—"
        return f"{check['pass']}/{check['applicable']}"

    def total(band):
        entry = summary["bands"][band]
        return f"{entry['pass']}/{entry['applicable']}" if entry["applicable"] else "—"

    def sum_check(name):
        applicable = sum(summary["bands"][band]["checks"][name]["applicable"]
                         for band in ("light", "heavy"))
        passed = sum(summary["bands"][band]["checks"][name]["pass"]
                     for band in ("light", "heavy"))
        return f"{passed}/{applicable}" if applicable else "—"

    lines = ["| 档 | 速度跟随稳定性 | 横向/朝向保持 | 关节静差 | 启停冲击 | 不翻车 | 任务通过 |",
             "|---|---|---|---|---|---|---|"]
    lines.append(f"| 小车（≤ {light_max:g} kg） | {cell(summary['bands']['light'], 'speed_tracking')} | "
                 f"{cell(summary['bands']['light'], 'lane_keeping')} | "
                 f"{cell(summary['bands']['light'], 'joint_static_error')} | "
                 f"{cell(summary['bands']['light'], 'start_stop_impact')} | "
                 f"{cell(summary['bands']['light'], 'no_fall')} | {total('light')} |")
    lines.append(f"| 重车（> {light_max:g} kg） | —（不判） | "
                 f"{cell(summary['bands']['heavy'], 'lane_keeping')} | "
                 f"{cell(summary['bands']['heavy'], 'joint_static_error')} | "
                 f"{cell(summary['bands']['heavy'], 'start_stop_impact')} | "
                 f"{cell(summary['bands']['heavy'], 'no_fall')} | {total('heavy')} |")
    lines.append(f"| 合计 | {sum_check('speed_tracking')} | {sum_check('lane_keeping')} | "
                 f"{sum_check('joint_static_error')} | {sum_check('start_stop_impact')} | "
                 f"{sum_check('no_fall')} | "
                 f"{summary['pass']}/{summary['applicable']} |")
    return lines


def _evaluate_report(report: dict, *, legacy_thresholds=None, task_thresholds=None) -> dict:
    """给旧 `report.json` 的 cases 补 `metrics.task` 并汇总（不写文件）。"""
    cases = report.get("cases") or []
    for summary in cases:
        metrics = summary.get("metrics")
        if isinstance(metrics, dict) and (metrics.get("task") or {}).get("version") != TASK_VERSION:
            metrics["task"] = evaluate_task(metrics, legacy_thresholds=legacy_thresholds,
                                            task_thresholds=task_thresholds)
    return summarize_task(cases)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="按 2026-10-19 收束后的任务口径复判 play_towing_test 的 report.json")
    parser.add_argument("report", type=Path, nargs="+", help="report.json（可多个）")
    for name, value in DEFAULT_TASK_THRESHOLDS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=value,
                            dest=name)
    for name in ("lane_y_limit_m", "lane_heading_limit_deg",
                 "joint_rms_limit_rad", "joint_max_limit_rad"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=float,
                            default=TASK_LEGACY_THRESHOLD_DEFAULTS[name], dest=name)
    parser.add_argument("--speed-mae-ratio-limit", type=float,
                        default=DEFAULT_SPEED_MAE_RATIO_LIMIT,
                        help="稳态 MAE / 指令（沿用旧 play_towing_test 的同名阈值）")
    parser.add_argument("--json", type=Path, default=None, help="把复判汇总写成 JSON")
    args = parser.parse_args(argv)
    task_thresholds = {name: getattr(args, name) for name in DEFAULT_TASK_THRESHOLDS}
    task_thresholds.update({name: getattr(args, name) for name in
                            ("lane_y_limit_m", "lane_heading_limit_deg",
                             "joint_rms_limit_rad", "joint_max_limit_rad")})
    task_thresholds["speed_mae_ratio_limit"] = args.speed_mae_ratio_limit
    payloads = []
    for path in args.report:
        report = json.loads(path.read_text(encoding="utf-8"))
        thresholds = report.get("thresholds") or {}
        summary = _evaluate_report(report, legacy_thresholds=thresholds,
                                   task_thresholds=task_thresholds)
        payloads.append({"report": str(path), "summary": summary})
        print(f"\n=== {path} ===")
        print("\n".join(task_conclusion_lines(summary)))
    if args.json is not None:
        args.json.write_text(json.dumps(payloads, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
        print(f"\n[out] {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
