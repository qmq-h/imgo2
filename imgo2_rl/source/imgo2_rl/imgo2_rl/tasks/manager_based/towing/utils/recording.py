"""Small, stdlib-only record writer for reproducible cart experiments."""

import csv
import json
import math
from pathlib import Path

LEGS = ("fl", "fr", "rl", "rr")
FIELDS = (
    "time_s", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps",
    "roll_rad", "pitch_rad", "yaw_rad", "wx_radps", "wy_radps", "wz_radps",
    *(f"{leg}_omega_radps" for leg in LEGS),
    *(f"{leg}_tau_nm" for leg in LEGS),
    *(f"{leg}_normal_n" for leg in LEGS),
)


def write_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


class CartRecorder:
    def __init__(self, directory: Path, config: dict):
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        write_json(directory / "config.json", config)
        self._stream = (directory / "trajectory.csv").open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._stream, fieldnames=FIELDS)
        self._writer.writeheader()
        self._last_time = -1.0

    def append(self, row: dict):
        if set(row) != set(FIELDS) or not all(math.isfinite(float(row[k])) for k in FIELDS):
            raise ValueError("Invalid/non-finite cart sample")
        if row["time_s"] <= self._last_time:
            raise ValueError("Cart sample times must strictly increase")
        self._writer.writerow(row)
        self._stream.flush()
        self._last_time = row["time_s"]

    def close(self):
        self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
