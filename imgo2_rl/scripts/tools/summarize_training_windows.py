"""Summarize TensorBoard scalar windows using only the standard library."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from read_tfevents import CLOCK_AXIS_SUFFIX, read_scalars, iteration_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event_file", type=Path)
    parser.add_argument("--window", type=int, default=2000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.window <= 0:
        parser.error("--window must be positive")
    series = read_scalars(str(args.event_file))
    last, axis = iteration_step(series)
    windows = []
    for start in range(0, last + 1, args.window):
        end = min(start + args.window, last + 1)
        metrics = {}
        for tag, points in series.items():
            if tag.endswith(CLOCK_AXIS_SUFFIX):
                continue
            values = [value for step, value in points if start <= step < end]
            finite = [value for value in values if math.isfinite(value)]
            if values:
                metrics[tag] = {
                    "count": len(values), "nonfinite_count": len(values) - len(finite),
                    "mean": statistics.mean(finite) if finite else None,
                    "std": statistics.pstdev(finite) if finite else None,
                }
        windows.append({"start": start, "end_exclusive": end, "metrics": metrics})
    report = {
        "event_filename": args.event_file.name,
        "event_sha256": hashlib.sha256(args.event_file.read_bytes()).hexdigest(),
        "last_iteration": last, "iteration_axis": axis, "window_size": args.window,
        "windows": windows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"iterations={last}; windows={len(windows)}; output={args.out}")


if __name__ == "__main__":
    main()
