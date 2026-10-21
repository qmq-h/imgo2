"""Audit AMP timing declarations and motion metadata without Isaac Lab."""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
VELOCITY = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity"


def literal_assignments(path, class_name):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    result = {}
    for node in ast.walk(cls):
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                result[ast.unparse(target)] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--motion-dir", type=Path, default=ROOT / "datasets/imgo2_motion",
                        help="Motion directory to audit; pass the actual AMP dataset explicitly.")
    args = parser.parse_args()
    base = literal_assignments(VELOCITY / "velocity_env_cfg.py", "LocomotionVelocityRoughEnvCfg")
    height = literal_assignments(VELOCITY / "base_move/amp_env_cfg.py", "Imgo2AmpMoveEnvCfg")
    runner = literal_assignments(VELOCITY / "base_move/agents/amp_rsl_rl_cfg.py", "AMPHeightRunnerCfg")
    physics_dt = height.get("self.sim.dt", base["self.sim.dt"])
    decimation = height.get("self.decimation", base["self.decimation"])
    control_dt = physics_dt * decimation
    rows = []
    for path in sorted(args.motion_dir.glob("*.txt")):
        data = json.loads(path.read_text())
        count, dt = len(data["Frames"]), float(data["FrameDuration"])
        if count < 2 or dt <= 0:
            raise ValueError(f"Invalid timing metadata: {path}")
        rows.append({"file": path.name, "frame_count": count, "frame_dt_s": dt,
                     "duration_s": (count - 1) * dt,
                     "old_index_speedup_pct": 100 / (count - 1)})
    if not rows:
        raise ValueError("No motion files found")
    deploy = (ROOT.parent / "imgo2_deploy/ros2_ws/policy/imgo2/base.yaml").read_text()
    deploy_dt = float(re.search(r"^  dt:\s*([\d.]+)", deploy, re.M)[1])
    deploy_decimation = int(re.search(r"^  decimation:\s*(\d+)", deploy, re.M)[1])
    report = {
        "motion_dir": str(args.motion_dir.resolve()),
        "physics_dt_s": physics_dt, "physics_hz": 1 / physics_dt,
        "decimation": decimation, "control_dt_s": control_dt, "policy_hz": 1 / control_dt,
        "rollout_steps": runner["num_steps_per_env"],
        "rollout_duration_s": runner["num_steps_per_env"] * control_dt,
        "deploy_policy_dt_s": deploy_dt * deploy_decimation,
        "motion_frame_dt_counts": dict(Counter(str(row["frame_dt_s"]) for row in rows)),
        "motions": rows,
        "scope": "Local source and bundled metadata; no server runtime verification",
    }
    if abs(report["deploy_policy_dt_s"] - control_dt) > 1e-9:
        raise ValueError("Deployment and training policy dt differ")
    output = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.write_text(output)
    print(f"Physics {1/physics_dt:g} Hz; policy {1/control_dt:g} Hz; rollout {runner['num_steps_per_env']} steps = {report['rollout_duration_s']:g} s")
    print(f"Motion metadata: {len(rows)} files, dt counts {report['motion_frame_dt_counts']}; deploy policy dt {report['deploy_policy_dt_s']:g} s")


if __name__ == "__main__":
    main()
