"""Measure raised-foot intervals in reference kinematics, NOT contact air time.

Run from any directory with Python 3; only the standard library is required.
The ground proxy is each foot's fifth percentile world height. Intervals cut
by a clip boundary are excluded. No contact labels exist in these motion files.
"""

import hashlib
import json
import statistics
from pathlib import Path


def world_foot_z(frame, leg):
    x, y, z, w = frame[3:7]  # xyzw
    px, py, pz = frame[19 + 3 * leg:22 + 3 * leg]
    norm = x * x + y * y + z * z + w * w
    return frame[2] + (
        2 * (x * z - w * y) * px + 2 * (y * z + w * x) * py
        + (w * w - x * x - y * y + z * z) * pz
    ) / norm


def complete_intervals(mask, dt):
    start = None
    spans = []
    for i, raised in enumerate(mask):
        if raised and start is None:
            start = i
        if not raised and start is not None:
            if start > 0:
                spans.append((i - start) * dt)
            start = None
    return spans


def main():
    root = Path(__file__).resolve().parents[3]
    results = {}
    for path in sorted((root / "imgo2_dataset/datasets/imgo2_motion").glob("*.txt")):
        raw = path.read_bytes()
        data = json.loads(raw)
        per_threshold = {}
        for threshold in (0.01, 0.02, 0.03):
            spans = []
            for leg in range(4):
                heights = [world_foot_z(frame, leg) for frame in data["Frames"]]
                baseline = sorted(heights)[int(0.05 * (len(heights) - 1))]
                spans.extend(complete_intervals(
                    [h > baseline + threshold for h in heights], data["FrameDuration"]
                ))
            per_threshold[str(threshold)] = {
                "complete_intervals": len(spans),
                "median_s": round(statistics.median(spans), 6) if spans else None,
            }
        results[path.stem] = {"sha256": hashlib.sha256(raw).hexdigest(),
                              "raised_height_threshold_m": per_threshold}
    print(json.dumps({"warning": "Kinematic raised-foot proxy, not measured contact air time; "
                      "baseline=per-foot 5th percentile world z; boundary intervals excluded.",
                      "motions": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
