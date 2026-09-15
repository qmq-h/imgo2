"""Verify that Imgo2's asset paths resolve on this machine, without Isaac Lab.

`assets/imgo2.py` derives the URDF and AMP motion paths from its own location
(`Path(__file__)`), so they follow the checkout instead of a hardcoded machine
path. This script replicates that derivation, checks the targets exist, and
reports any machine-specific absolute path left in the sources.

Run this on a new machine (especially the training server) before training: an
empty motion glob makes AMPLoader fail without a clear message.

Stdlib only. Run from imgo2_rl:
    python scripts/tools/check_asset_paths.py
Exit code 0 only when every check passes.
"""

import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]  # imgo2_rl/
ASSET_FILES = [
    ROOT / "source/imgo2_rl/imgo2_rl/assets/imgo2.py",
    ROOT / "source/imgo2_rl/imgo2_rl/assets/amp_motions.py",
]
MACHINE_PATH_RX = re.compile(r"/root/|/home/|[A-Za-z]:\\\\Users")


def resolve_project_root(asset_file: Path) -> tuple[Path | None, int | None]:
    """Read the parents[N] depth out of the asset file and apply it."""
    text = asset_file.read_text(encoding="utf-8")
    match = re.search(r"parents\[(\d+)\]", text)
    if not match:
        return None, None
    depth = int(match.group(1))
    return asset_file.resolve().parents[depth], depth


def main() -> int:
    failures = []

    asset = ASSET_FILES[0]
    if not asset.is_file():
        print(f"FAIL  asset config not found: {asset}")
        return 1

    project_root, depth = resolve_project_root(asset)
    if project_root is None:
        print(f"FAIL  {asset.name} does not derive paths from Path(__file__)")
        return 1

    print(f"asset config      : {asset.relative_to(ROOT.parent)}")
    print(f"parents[{depth}]          : {project_root}")
    ok_root = project_root.name == "imgo2_rl" and (project_root / "source").is_dir()
    print(f"  looks like imgo2_rl root: {ok_root}")
    if not ok_root:
        failures.append("project root derivation")

    urdf = project_root / "source/imgo2_rl/data/imgo2_model/imgo2_urdf/urdf/imgo2.urdf"
    print(f"\nURDF              : {urdf}")
    print(f"  exists          : {urdf.is_file()}")
    if not urdf.is_file():
        failures.append("URDF missing")

    motion_dir = Path(os.environ.get("IMGO2_AMP_MOTION_DIR") or project_root / "datasets/imgo2_motion")
    motions = sorted(motion_dir.glob("*"))
    print(f"\nmotion dir        : {motion_dir}")
    print(f"  override in use : {bool(os.environ.get('IMGO2_AMP_MOTION_DIR'))}")
    print(f"  file count      : {len(motions)} (expected 21)")
    if len(motions) != 21:
        failures.append(f"motion file count {len(motions)} != 21")

    urdf_override = os.environ.get("IMGO2_URDF_PATH")
    print(f"\nIMGO2_URDF_PATH    : {urdf_override or '(unset, using default)'}")
    print(f"IMGO2_AMP_MOTION_DIR: {os.environ.get('IMGO2_AMP_MOTION_DIR') or '(unset, using default)'}")

    print("\nmachine-specific absolute paths left in asset configs:")
    leftovers = 0
    for path in ASSET_FILES:
        if not path.is_file():
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if MACHINE_PATH_RX.search(line):
                leftovers += 1
                print(f"  {path.name}:{lineno}: {line.strip()}")
    if not leftovers:
        print("  none")
    else:
        failures.append(f"{leftovers} machine-specific path(s)")

    print("\nSummary")
    if failures:
        for item in failures:
            print(f"  FAIL  {item}")
        return 1
    print("  PASS  asset paths resolve on this machine")
    return 0


if __name__ == "__main__":
    sys.exit(main())
