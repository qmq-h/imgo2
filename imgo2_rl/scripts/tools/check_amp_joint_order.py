"""Independent checks that the AMP dataset's joint-angle blocks are correctly ordered.

Complements audit_amp_dataset.py. The bundled audit compares joint blocks against
foot blocks using forward kinematics and the training URDF only. This script adds:

1. FK against the unified model (`imgo2_description/urdf/imgo2.urdf`, which since the
   2026-09-17 unification is the same robot as the training URDF), with a cross-leg
   swapped-order control to show the pairing is not accidental.
   Note: before the unification this used the separate LF/LH/RF/RH recording copy and
   also contrasted its declaration order; the model is now FL/FR/RL/RR, so that
   particular sub-test no longer exists and checks 2 and 3 carry the URDF-free evidence.
2. A URDF-free check: in a quasi-symmetric stance, hip abduction angles of the left
   and right leg of the same pair must have opposite signs. This identifies which
   blocks are left legs without any model, FK or recording-argument assumption.
3. A joint-velocity block check: central-difference the joint-position block and
   correlate with the recorded joint-velocity block per channel.

All three support the identity mapping `joint_mapping = list(range(12))` for the
current `datasets/imgo2_motion` data, i.e. file order FL,FR,RL,RR (== LF,RF,LH,RH).

Stdlib only. Run from imgo2_rl:
    python scripts/tools/check_amp_joint_order.py
"""

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_amp_dataset import audit  # noqa: E402

LEGS = ("FL", "FR", "RL", "RR")
DOFS = ("hip", "thigh", "shank")


def pearson(xs, ys):
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else float("nan")


def check_recording_urdf(motion_dir, recording_urdf):
    """FK with the unified model, plus the audit's cross-leg swapped-order control."""
    import audit_amp_dataset as a

    original = a.LEGS
    a.LEGS = ("FL", "FR", "RL", "RR")
    try:
        result = a.audit(motion_dir, recording_urdf)
    finally:
        a.LEGS = original

    def stats(key):
        values = [m["fk_coordinate_rmse_m_by_order"][key] for m in result["motions"]]
        return statistics.mean(values), max(values)

    id_mean, id_max = stats("FL_FR_RL_RR")
    sw_mean, sw_min = stats("FL_RL_FR_RR")
    print("1) FK against the unified model imgo2_description/urdf/imgo2.urdf")
    print(f"   file order FL,FR,RL,RR: mean={id_mean:.5f} m max={id_max:.5f} m")
    print(f"   swapped    FL,RL,FR,RR: mean={sw_mean:.5f} m min={sw_min:.5f} m")
    print(f"   files={result['file_count']} frames={result['frame_count']} "
          f"urdf_sha256={result['urdf_sha256'][:16]}")
    return id_mean < 0.005 and id_max < 0.005 and sw_min > 0.20


def check_hip_symmetry(files):
    """Left/right hip abduction antisymmetry; needs no URDF and no FK."""
    identity, swapped = [], []
    for frames in files.values():
        block = [statistics.mean(f[7 + 3 * i] for f in frames) for i in range(4)]
        identity += [abs(block[0] + block[1]), abs(block[2] + block[3])]
        swapped += [abs(block[0] + block[2]), abs(block[1] + block[3])]
    id_mean, sw_mean = statistics.mean(identity), statistics.mean(swapped)
    print("\n2) Hip abduction left/right antisymmetry (no URDF involved)")
    print(f"   identity FL,FR,RL,RR: mean |pair sum| = {id_mean:.4f} rad")
    print(f"   swapped  FL,RL,FR,RR: mean |pair sum| = {sw_mean:.4f} rad")
    return id_mean < sw_mean / 2.0


def check_joint_velocity_block(files, dt):
    """Same-leg pos/vel correlation vs a cross-leg control."""
    matched, cross = [], []
    for frames in files.values():
        n = len(frames)
        for leg in range(4):
            for k in range(3):
                qcol, vcol = 7 + 3 * leg + k, 37 + 3 * leg + k
                fd = [(frames[i + 1][qcol] - frames[i - 1][qcol]) / (2 * dt) for i in range(1, n - 1)]
                obs = [frames[i][vcol] for i in range(1, n - 1)]
                matched.append(pearson(fd, obs))
                qcol_o = 7 + 3 * ((leg + 1) % 4) + k
                fd_o = [(frames[i + 1][qcol_o] - frames[i - 1][qcol_o]) / (2 * dt) for i in range(1, n - 1)]
                cross.append(pearson(fd_o, obs))
    m_mean, c_mean = statistics.mean(matched), statistics.mean(cross)
    print("\n3) Joint-position block vs joint-velocity block alignment")
    print(f"   matched same-leg channels: mean r = {m_mean:.4f}")
    print(f"   cross-leg control        : mean r = {c_mean:.4f}")
    return m_mean > 0.90 and c_mean < 0.60


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-dir", type=Path, default=root / "datasets/imgo2_motion")
    parser.add_argument(
        "--recording-urdf",
        type=Path,
        default=root.parent / "imgo2_description/urdf/imgo2.urdf",
        help="The model the motion data was recorded from. Since the 2026-09-17 unification "
             "this is the generated core URDF in imgo2_description (same robot as the training URDF).",
    )
    args = parser.parse_args()

    if not args.motion_dir.is_dir():
        raise SystemExit(f"Motion directory not found: {args.motion_dir}")
    if not args.recording_urdf.is_file():
        raise SystemExit(
            f"Recording URDF not found: {args.recording_urdf}. "
            "Pass --recording-urdf pointing at the model used for data collection."
        )

    files = {p.name: json.loads(p.read_text(encoding="utf-8-sig"))["Frames"]
             for p in sorted(args.motion_dir.glob("*.txt"))}
    if not files:
        raise SystemExit(f"No motion .txt files in {args.motion_dir}")
    durations = {json.loads(p.read_text(encoding="utf-8-sig"))["FrameDuration"]
                 for p in sorted(args.motion_dir.glob("*.txt"))}
    if len(durations) != 1:
        raise SystemExit(f"Expected one FrameDuration across files, got {sorted(durations)}")
    dt = durations.pop()

    results = [
        ("recording-URDF FK", check_recording_urdf(args.motion_dir, args.recording_urdf)),
        ("hip left/right symmetry", check_hip_symmetry(files)),
        ("joint-velocity block", check_joint_velocity_block(files, dt)),
    ]

    print("\nSummary")
    for name, passed in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    if not all(passed for _, passed in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
