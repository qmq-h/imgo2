"""Recompute a cart run's summary and SVG using only the Python standard library."""

import argparse
import csv
import json
import math
from pathlib import Path

from cart_coast_metrics import summarize, svg_plot


def summarize_run(directory: Path):
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    with (directory / "trajectory.csv").open(encoding="utf-8", newline="") as stream:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(stream)]
    result = summarize(rows, radius=config["model"]["wheel_radius_m"],
                       resting_height=config["model"]["resting_height_m"], mode=config["mode"],
                       stop_speed=config["stop_speed_mps"], stop_hold=config["stop_hold_s"])
    # Preserve abnormal termination instead of reclassifying a partial CSV as a valid run.
    status_path = directory / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {"state": "unknown"}
    result["run_state"] = status["state"]
    if status["state"] != "completed":
        result["valid"] = False
        result["failures"].append("run_not_completed")
    if "dt_s" in config and "requested_duration_s" in config:
        dt = config["dt_s"]
        duration = config["requested_duration_s"]
        if not all(math.isfinite(v) and v > 0 for v in (dt, duration)):
            raise ValueError("Invalid recorded timestep/duration")
        expected_count = math.ceil(duration / dt) + 1
        complete = len(rows) == expected_count and all(
            math.isclose(row["time_s"], i * dt, abs_tol=1e-7, rel_tol=1e-7)
            for i, row in enumerate(rows))
        if not complete:
            result["valid"] = False
            result["failures"].append("incomplete_or_irregular_samples")
    (directory / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (directory / "coast.svg").write_text(svg_plot(rows), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = summarize_run(args.run_dir)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["valid"] else 1)
