#!/usr/bin/env python3
"""Run the controlled cart-mass step experiment with play_towing_test.py.

Default matrix (flat 0 deg, 1.0 m/s, 160 env per connection):
  compliant (columns 0-7): fixed 10 kg, fixed 30 kg, 10 -> 30 kg @ tow+5 s
  rigid     (columns 8-15): fixed 10 kg, fixed 30 kg, 10 -> 30 kg @ tow+5 s
Optional: 30 -> 10 kg unload for both connections.

Each run enables --dump-latent (phase means) and --latent-trace (per-record 16-D
latent + current mass).  Run this script under the Isaac Lab launcher, e.g.:

  bash imgo2_rl/scripts/run_isaaclab.sh \
      imgo2_rl/scripts/towing/mass_step_sweep.py --dry-run
  bash imgo2_rl/scripts/run_isaaclab.sh \
      imgo2_rl/scripts/towing/mass_step_sweep.py
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PLAY = ROOT / "imgo2_rl/scripts/towing/play_towing_test.py"

CONNECTIONS = {
    "compliant": "0-7",
    "rigid": "8-15",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upper-checkpoint", type=Path,
                        default=ROOT / "logs/towing_rl_lab/towing_upper/2026-11-13_15-23-45_v6_50hz_roll10_fall500_70col_20261113/model_2000.pt")
    parser.add_argument("--policy-checkpoint", type=Path,
                        default=ROOT / "imgo2_deploy/policy/imgo2/ppo/policy.pt")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--num-envs", type=int, default=160,
                        help="flat 0-7 / 8-15 各 8 列 ×20 行 = 160；默认每连接 160")
    parser.add_argument("--mass-step-at-s", type=float, default=5.0,
                        help="tow 段开始后的秒数（1.0 m/s 时约 5 m）")
    parser.add_argument("--include-reverse", action="store_true",
                        help="额外跑 30 -> 10 kg 卸载")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for path in (PLAY, args.upper_checkpoint, args.policy_checkpoint):
        if not path.is_file():
            parser.error(f"文件不存在：{path}")
    if args.num_envs != 160:
        parser.error("--num-envs 必须为 160（每连接 8 列 × 20 行）")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = (args.output_root or
            ROOT / f"imgo2_rl/logs/towing/play_test/mass_step_70col_flat_v1_{stamp}")
    root.mkdir(parents=True, exist_ok=True)

    common = [
        sys.executable, str(PLAY), "--headless",
        "--num-envs", "160", "--velocities", "1.0",
        "--tow-duration", "24", "--coast-duration", "5",
        "--stop-progress", "10", "--write-csv", "none",
        "--compact-log", "--dump-latent", "--latent-trace",
        "--policy", "ppo", "--policy-checkpoint", str(args.policy_checkpoint),
        "--upper-checkpoint", str(args.upper_checkpoint),
    ]
    jobs = []
    for connection, columns in CONNECTIONS.items():
        base = [*common, "--columns", columns]
        for label, mass, step in (
            ("fixed10", 10, None),
            ("fixed30", 30, None),
            ("step10to30", 10, (10, 30)),
        ):
            cmd = [*base, "--cart-masses", str(mass),
                   "--output-dir", str(root / f"{connection}_{label}")]
            if step is not None:
                cmd += ["--mass-step-at-s", str(args.mass_step_at_s),
                        "--mass-step-from", str(step[0]),
                        "--mass-step-to", str(step[1])]
            jobs.append((f"{connection}/{label}", cmd))
        if args.include_reverse:
            cmd = [*base, "--cart-masses", "30",
                   "--mass-step-at-s", str(args.mass_step_at_s),
                   "--mass-step-from", "30", "--mass-step-to", "10",
                   "--output-dir", str(root / f"{connection}_step30to10")]
            jobs.append((f"{connection}/step30to10", cmd))

    print(f"[mass-step] output root: {root}")
    for label, cmd in jobs:
        print(f"\n[mass-step] {label}\n  {shlex.join(cmd)}")
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True, cwd=str(ROOT))
    if args.dry_run:
        print(f"\n[mass-step] dry-run only; {len(jobs)} commands")
    else:
        print(f"\n[mass-step] done: {len(jobs)} runs -> {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
