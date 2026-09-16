"""Verify that Imgo2's asset paths resolve on this machine, without Isaac Lab.

`assets/imgo2.py` derives the URDF and AMP motion paths from its own location
(`Path(__file__)`), so they follow the checkout instead of a hardcoded machine
path. This script replicates that derivation, checks the targets exist, and
reports any machine-specific absolute path left in the sources.

Since the 2026-09-17 model unification the URDF lives outside `imgo2_rl/`
(`<repo>/imgo2_description/urdf/imgo2.urdf`), so the config declares two roots:
`_PROJECT_ROOT` (parents[4] -> `imgo2_rl/`) and `_REPO_ROOT` (parents[5] ->
`<repo>/`). The checker reads whichever root each declared path uses.

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


def root_vars(text: str) -> dict[str, int]:
    """{variable -> parents[N]} for every `VAR = Path(__file__).resolve().parents[N]`."""
    return {m.group(1): int(m.group(2))
            for m in re.finditer(r"(\w+)\s*=\s*Path\(__file__\)\.resolve\(\)\.parents\[(\d+)\]", text)}


def declared_path(text: str, name: str):
    """(root variable, literal components) declared for `name`, or None.

    Reads e.g. `_DEFAULT_URDF_PATH = _REPO_ROOT / "imgo2_description" / ...`.
    This must be read from the source rather than duplicated here: an earlier
    version of this script hardcoded the expected relative path, so it kept
    passing even when the config's own path was wrong.
    """
    match = re.search(rf"{name}\s*=\s*\(([^)]*)\)", text)
    if match is None:
        match = re.search(rf"{name}\s*=\s*(.+)", text)
    if match is None:
        return None
    body = match.group(1)
    roots = re.findall(r"\b(_[A-Z][A-Z_]*)\b", body)
    if not roots:
        return None
    return roots[0], re.findall(r'"([^"]*)"', body)


def mesh_refs(urdf: Path) -> list[Path]:
    """Mesh files referenced by the URDF, resolved relative to it."""
    text = urdf.read_text(encoding="utf-8", errors="replace")
    return [(urdf.parent / ref).resolve() for ref in re.findall(r'filename="([^"]+)"', text)]


def main() -> int:
    failures = []

    asset = ASSET_FILES[0]
    if not asset.is_file():
        print(f"FAIL  asset config not found: {asset}")
        return 1

    asset_text = asset.read_text(encoding="utf-8")
    depths = root_vars(asset_text)
    if not depths:
        print(f"FAIL  {asset.name} does not derive paths from Path(__file__)")
        return 1
    roots = {name: asset.resolve().parents[d] for name, d in depths.items()}

    print(f"asset config      : {asset.relative_to(ROOT.parent)}")
    for name, depth in sorted(depths.items()):
        print(f"  {name:16s} parents[{depth}] -> {roots[name]}")
    project_root = roots.get("_PROJECT_ROOT")
    ok_root = project_root is not None and project_root.name == "imgo2_rl" and (project_root / "source").is_dir()
    print(f"  looks like imgo2_rl root: {ok_root}")
    if not ok_root:
        failures.append("project root derivation")

    # --- read the paths the config actually declares, do not duplicate them here ---
    print("\ndeclared by the asset config:")
    resolved = {}
    for name, label in (("_DEFAULT_URDF_PATH", "URDF"), ("_DEFAULT_MOTION_DIR", "motion dir")):
        declared = declared_path(asset_text, name)
        if declared is None:
            failures.append(f"{name} not found / not derived from a Path(__file__) root")
            print(f"  FAIL  {name}: not found or not derived from a root variable")
            continue
        root_var, parts = declared
        if root_var not in roots:
            failures.append(f"{name}: uses unknown root variable {root_var}")
            print(f"  FAIL  {name}: unknown root variable {root_var}")
            continue
        target = roots[root_var].joinpath(*parts)
        resolved[name] = target
        print(f"  {name}  ({label}, via {root_var})")
        print(f"    components : {parts}")
        print(f"    resolves to: {target}")

    urdf_env = os.environ.get("IMGO2_URDF_PATH")
    urdf = Path(urdf_env).expanduser() if urdf_env else resolved.get("_DEFAULT_URDF_PATH")
    if urdf is None:
        failures.append("URDF path unresolved")
    else:
        print(f"\nURDF in effect    : {urdf}")
        print(f"  exists          : {urdf.is_file()}")
        if not urdf.is_file():
            failures.append("URDF missing")
        else:
            refs = mesh_refs(urdf)
            missing = [r for r in refs if not r.is_file()]
            print(f"  mesh refs       : {len(refs)} referenced, {len(missing)} missing")
            for m in missing[:5]:
                print(f"    MISSING {m}")
            if missing:
                failures.append(f"{len(missing)} mesh reference(s) unresolvable")

    motion_env = os.environ.get("IMGO2_AMP_MOTION_DIR")
    motion_dir = Path(motion_env).expanduser() if motion_env else resolved.get("_DEFAULT_MOTION_DIR")
    if motion_dir is None:
        failures.append("motion dir unresolved")
    else:
        motions = sorted(p for p in motion_dir.glob("*") if p.is_file())
        print(f"\nmotion dir        : {motion_dir}")
        print(f"  override in use : {bool(motion_env)}")
        print(f"  file count      : {len(motions)} (expected 21)")
        if len(motions) != 21:
            failures.append(f"motion file count {len(motions)} != 21")

    print(f"\nIMGO2_URDF_PATH      : {urdf_env or '(unset, using default)'}")
    print(f"IMGO2_AMP_MOTION_DIR : {motion_env or '(unset, using default)'}")

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
