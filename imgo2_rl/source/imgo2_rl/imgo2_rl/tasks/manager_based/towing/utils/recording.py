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


# 拖曳记录（计划 P5 的字段表，P4 先只填其中不需要步态指标的部分）。
# `ref_cmd_mps` 在 P4 恒等于用户指令（还没有 command shaping），留列是为了 P6/P8。
TOW_FIELDS = (
    "time_s", "user_cmd_mps", "ref_cmd_mps", "robot_vx_mps", "load_vx_mps",
    "rope_tension_n", "rope_distance_m", "robot_x_m", "load_x_m",
    "body_pitch_rad", "body_pitch_rate_radps",
)


class TowRecorder:
    """逐物理步记录机器人/负载/绳状态；与 CartRecorder 同样的严格校验。"""

    def __init__(self, directory: Path, config: dict):
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        write_json(directory / "config.json", config)
        self._stream = (directory / "tow.csv").open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._stream, fieldnames=TOW_FIELDS)
        self._writer.writeheader()
        self._last_time = -1.0

    def append(self, row: dict):
        if set(row) != set(TOW_FIELDS) or not all(math.isfinite(float(row[k])) for k in TOW_FIELDS):
            raise ValueError("Invalid/non-finite tow sample")
        if row["time_s"] <= self._last_time:
            raise ValueError("Tow sample times must strictly increase")
        self._writer.writerow(row)
        self._stream.flush()
        self._last_time = row["time_s"]

    def close(self):
        self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
