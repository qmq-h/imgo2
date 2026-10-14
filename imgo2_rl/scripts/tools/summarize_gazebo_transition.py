#!/usr/bin/env python3
"""Summarize full transition telemetry recorded by eval_gazebo_policy (stdlib)."""
import argparse
import json
from pathlib import Path


def summarize(path):
    data = json.loads(path.read_text())
    commands = data["motor_commands_wall"]
    first = next((i for i, (_, c) in enumerate(commands)
                  if c and 0 < c[0][2] < 59.999), None)
    if first is None or first == 0:
        return {"path": str(path), "error": "No stand-to-policy gain transition recorded"}
    start = commands[first][0]
    before, after = commands[first - 1][1], commands[first][1]
    jump = max(abs(a[0] - b[0]) for a, b in zip(after, before))
    early = [commands[first - 1]] + [row for row in commands[first:] if row[0] <= start + 0.5]
    steps = [max(abs(a[0] - b[0]) for a, b in zip(right[1], left[1]))
             for left, right in zip(early, early[1:])]
    feedback = [row for t, row in data["motor_feedback_wall"] if start <= t <= start + 1]
    pose = [row for t, row in zip(data["odom_wall"], data["odom"]) if start <= t <= start + 1]
    stable = [row for t, row in commands if t >= start + 4 and row and abs(row[0][2] - 25) < 1e-4]
    defaults, scales = [0, 0.87, -1.82] * 4, [0.125, 0.25, 0.25] * 4
    saturations = sum(abs((row[i][0] - defaults[i]) / scales[i]) >= 2.999 for row in stable for i in range(12))
    return {"path": str(path), "first_policy_target_jump_rad": jump,
            "first_policy_kp_change": after[0][2] - before[0][2],
            "max_target_step_first_half_second_rad": max(steps, default=0),
            "peak_joint_velocity_first_second": max((abs(m[1]) for row in feedback for m in row), default=None),
            "peak_feedback_effort_first_second": max((abs(m[2]) for row in feedback for m in row), default=None),
            "first_second_z_min": min((row[2] for row in pose), default=None),
            "stable_raw_action_boundary_fraction": saturations / max(1, len(stable) * 12),
            "limits": "Boundary fraction inferred from transmitted targets; not exact unclipped network output. Commands/feedback use wall timestamps; pose rows use simulation stamps."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    rows = [summarize(path) for path in args.paths]
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
