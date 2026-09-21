#!/usr/bin/env python3
"""Run a coarse towing grid and classify steady-tow / direct-stop boundaries.

One Isaac Sim process is launched per velocity.  Inside that process ``tow_drag.py``
already sweeps mass and wheel damping without restarting the simulator.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys


RL_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = RL_ROOT.parent
LAUNCHER = RL_ROOT / "scripts/run_isaaclab.sh"
TOW_DRAG = RL_ROOT / "scripts/towing/tow_drag.py"
DEFAULT_VELOCITIES = (0.2, 0.4, 0.6, 0.8, 1.0)
DEFAULT_MASSES = (5.0, 10.0, 15.0, 20.0, 25.0)
DEFAULT_DAMPINGS = (0.008, 0.032)
DEFAULT_FRICTIONS = (0.8,)


def finite_values(values, *, name: str, lower: float, upper: float) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name} 不能为空")
    if any(not math.isfinite(value) or value < lower or value > upper for value in result):
        raise ValueError(f"{name} 必须在 [{lower:g}, {upper:g}] 内")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} 不能包含重复值")
    return tuple(sorted(result))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="扫描拖曳能力与 Direct Stop 边界")
    parser.add_argument("--velocities", nargs="+", type=float, default=list(DEFAULT_VELOCITIES),
                        help="速度网格，m/s；默认 0.2 0.4 0.6 0.8 1.0")
    parser.add_argument("--cart-masses", nargs="+", type=float, default=list(DEFAULT_MASSES),
                        help="小车质量网格，kg")
    parser.add_argument("--wheel-dampings", nargs="+", type=float,
                        default=list(DEFAULT_DAMPINGS), help="单轮黏性阻尼网格，N m s/rad")
    parser.add_argument("--ground-frictions", nargs="+", type=float,
                        default=list(DEFAULT_FRICTIONS),
                        help="机器人/小车共享地面摩擦；默认先固定 0.8，边界复核可用 0.4 0.8 1.2")
    parser.add_argument("--rope-model", choices=("compliant", "inextensible"),
                        default="compliant", help="一次扫描一种绳，便于解释边界")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--stop-at", type=float, default=5.0)
    parser.add_argument("--tracking-min", type=float, default=0.8,
                        help="稳态跟速比低于该值，归类为 tow_infeasible")
    parser.add_argument("--clearance-margin", type=float, default=0.10,
                        help="未碰撞但最终车头间隙不大于该值，归类为 stop_margin_low")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true", help="只打印将执行的命令")
    parser.add_argument("--summarize", type=Path,
                        help="不运行仿真，只重新汇总已有扫描根目录")
    args = parser.parse_args(argv)
    try:
        args.velocities = finite_values(args.velocities, name="velocities", lower=0.2, upper=1.0)
        args.cart_masses = finite_values(args.cart_masses, name="cart-masses", lower=2.0, upper=50.0)
        args.wheel_dampings = finite_values(
            args.wheel_dampings, name="wheel-dampings", lower=0.000001, upper=0.1)
        args.ground_frictions = finite_values(
            args.ground_frictions, name="ground-frictions", lower=0.01, upper=2.0)
    except ValueError as exc:
        parser.error(str(exc))
    if not math.isfinite(args.duration) or not math.isfinite(args.stop_at):
        parser.error("duration/stop-at 必须是有限数")
    if not 0.0 < args.stop_at < args.duration:
        parser.error("必须满足 0 < --stop-at < --duration")
    if not 0.5 <= args.tracking_min <= 1.2:
        parser.error("--tracking-min 必须在 [0.5, 1.2]")
    if not 0.0 <= args.clearance_margin <= 0.5:
        parser.error("--clearance-margin 必须在 [0, 0.5] m")
    return args


def velocity_label(velocity: float) -> str:
    return f"v{velocity:.2f}".replace(".", "p")


def scan_label(velocity: float, friction: float) -> str:
    return f"{velocity_label(velocity)}_mu{friction:.2f}".replace(".", "p")


def child_command(args, velocity: float, friction: float, output: Path) -> list[str]:
    return [
        "bash", str(LAUNCHER), str(TOW_DRAG), "--headless",
        "--device", args.device,
        "--velocity", f"{velocity:g}",
        "--cart-mass", *(f"{value:g}" for value in args.cart_masses),
        "--wheel-damping", *(f"{value:g}" for value in args.wheel_dampings),
        "--ground-friction", f"{friction:g}",
        "--rope-model", args.rope_model,
        "--duration", f"{args.duration:g}", "--stop-at", f"{args.stop_at:g}",
        "--output-dir", str(output),
    ]


def classify(summary: dict, *, tracking_min: float, clearance_margin: float) -> str:
    """Separate pullability from the outcome of the unshaped direct stop."""
    tracking = summary.get("steady_tracking_ratio")
    if not summary.get("valid", False) or tracking is None or tracking < tracking_min:
        return "tow_infeasible"
    if summary.get("reached_robot") is True:
        return "stop_collision"
    clearance = summary.get("final_clearance_m")
    if clearance is None:
        return "unknown"
    if clearance <= clearance_margin:
        return "stop_margin_low"
    return "direct_stop_safe"


def load_rows(root: Path, *, tracking_min: float, clearance_margin: float) -> list[dict]:
    rows = []
    for summary_path in sorted(root.glob("*/case_*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config_path = summary_path.with_name("config.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        row = {
            "velocity_mps": float(config["user_command_mps"]),
            "cart_mass_kg": float(summary["cart_mass_kg"]),
            "wheel_damping": float(summary["wheel_damping"]),
            "ground_friction": float(config["ground_friction"]),
            "rope_model": summary["rope_model"],
            "classification": classify(summary, tracking_min=tracking_min,
                                       clearance_margin=clearance_margin),
            "valid": bool(summary["valid"]),
            "steady_tracking_ratio": summary.get("steady_tracking_ratio"),
            "reached_robot": summary.get("reached_robot"),
            "final_clearance_m": summary.get("final_clearance_m"),
            "min_clearance_m": summary.get("min_clearance_m"),
            "time_to_contact_after_stop_s": summary.get("time_to_contact_after_stop_s"),
            "load_vx_at_contact_mps": summary.get("load_vx_at_contact_mps"),
            "failures": summary.get("failures", []),
            "case_dir": str(summary_path.parent.relative_to(root)),
        }
        rows.append(row)
    return sorted(rows, key=lambda row: (row["rope_model"], row["ground_friction"],
                                         row["wheel_damping"],
                                         row["cart_mass_kg"], row["velocity_mps"]))


def boundary_table(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        key = (row["rope_model"], row["ground_friction"],
               row["wheel_damping"], row["cart_mass_kg"])
        groups.setdefault(key, []).append(row)
    result = []
    for (rope, friction, damping, mass), items in sorted(groups.items()):
        safe = [r["velocity_mps"] for r in items if r["classification"] == "direct_stop_safe"]
        towable = [r["velocity_mps"] for r in items if r["classification"] != "tow_infeasible"]
        unsafe = [r["velocity_mps"] for r in items
                  if r["classification"] in ("stop_margin_low", "stop_collision")]
        result.append({
            "rope_model": rope, "ground_friction": friction,
            "wheel_damping": damping, "cart_mass_kg": mass,
            "max_tested_tow_feasible_velocity_mps": max(towable) if towable else None,
            "max_direct_stop_safe_velocity_mps": max(safe) if safe else None,
            "first_tested_stop_risk_velocity_mps": min(unsafe) if unsafe else None,
        })
    return result


def write_report(root: Path, rows: list[dict], args) -> None:
    boundaries = boundary_table(rows)
    payload = {
        "definition": {
            "tow_infeasible": f"summary invalid or steady_tracking_ratio < {args.tracking_min:g}",
            "stop_collision": "steady tow feasible, but Direct Stop reaches the robot",
            "stop_margin_low": f"no collision, final_clearance_m <= {args.clearance_margin:g} m",
            "direct_stop_safe": "steady tow feasible and clearance above the configured margin",
            "warning": "Direct Stop risk is not proof that an upper controller cannot solve the case",
        },
        # Derive the realized grid from artifacts.  In --summarize mode the CLI defaults need not
        # match the original run, so copying args here would silently write false provenance.
        "grid": {
            "velocities_mps": sorted({row["velocity_mps"] for row in rows}),
            "cart_masses_kg": sorted({row["cart_mass_kg"] for row in rows}),
            "wheel_dampings": sorted({row["wheel_damping"] for row in rows}),
            "ground_frictions": sorted({row["ground_friction"] for row in rows}),
            "rope_models": sorted({row["rope_model"] for row in rows}),
        },
        "cases": rows, "boundaries": boundaries,
    }
    (root / "boundary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fields = list(rows[0]) if rows else ["velocity_mps", "cart_mass_kg", "wheel_damping",
                                         "rope_model", "classification"]
    with (root / "boundary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "failures": ";".join(row.get("failures", []))})


def main(args) -> int:
    if args.summarize:
        root = args.summarize.expanduser().resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        root = (args.output_dir or RL_ROOT / "logs/towing/boundary" / stamp).expanduser().resolve()
        if root.exists():
            raise FileExistsError(f"输出目录已存在，拒绝覆盖：{root}")
        if not args.dry_run:
            root.mkdir(parents=True)
        for friction in args.ground_frictions:
            for velocity in args.velocities:
                output = root / scan_label(velocity, friction)
                command = child_command(args, velocity, friction, output)
                print(" ".join(command), flush=True)
                if args.dry_run:
                    continue
                result = subprocess.run(command, cwd=REPO_ROOT, check=False)
                # tow_drag uses 2 when at least one case is invalid.  That is expected in a boundary
                # scan; missing output is the actual orchestration failure.
                if result.returncode not in (0, 2) or not (output / "sweep.json").is_file():
                    raise RuntimeError(f"v={velocity:g} μ={friction:g} 扫描失败，"
                                       f"退出码 {result.returncode}")
        if args.dry_run:
            launches = len(args.velocities) * len(args.ground_frictions)
            cases = launches * len(args.cart_masses) * len(args.wheel_dampings)
            print(f"[DRY-RUN] {launches} 次 Isaac Sim 启动，{cases} 个 case")
            return 0
    rows = load_rows(root, tracking_min=args.tracking_min,
                     clearance_margin=args.clearance_margin)
    if not rows:
        raise RuntimeError(f"没有找到 */case_*/summary.json：{root}")
    write_report(root, rows, args)
    counts = {name: sum(row["classification"] == name for row in rows) for name in
              ("direct_stop_safe", "stop_margin_low", "stop_collision", "tow_infeasible", "unknown")}
    print(f"[BOUNDARY] {len(rows)} cases: {counts} ⇒ {root / 'boundary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
