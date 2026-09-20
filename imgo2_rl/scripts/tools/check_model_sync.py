"""Check that every Imgo2 model agrees on joint convention, and that each URDF
still reproduces the recorded motion data.

Why this exists: the deploy URDF was once fully sign-inverted on all 12 leg
joints (q_deploy = -q_train), which silently disagreed with the real-robot data
the AMP reference motions were recorded from. Signs and limits must therefore be
checked across every model, not just the one being edited.

Comparison is by *logical joint* (leg + joint role) and *logical link*, so it does
not depend on the declaration order inside a file.

Model layout since 2026-09-17 (user decision, README MODEL-02): `imgo2_description/`
is the single source. `xacro/core.xacro` holds the physics (FL/FR/RL/RR naming,
identical to the training URDF), and `xacro/robot.xacro` assembles the committed
artifacts `urdf/imgo2.urdf` (core only) and `urdf/imgo2.gazebo.urdf`
(core + transmission + gazebo + imu). The former training / deploy / `imgo2_model/`
copies have been deleted; this script now guards the two generated artifacts and
the single mesh directory.

Fails when any of these breaks:
  1. a leg joint's axis or limits differs between any two complete URDFs;
  2. a link's mass, centre of mass, inertia tensor or collision geometry differs;
  3. a base link inertial block drifts from the agreed canonical values
     (user decision: the recording model's values);
  4. the model mesh directory drifts from the recorded fingerprint, or a URDF
     references a mesh file that does not exist;
  5. any URDF stops reproducing `datasets/imgo2_motion` through forward kinematics;
  6. a tracked or non-ignored untracked URDF has no registered asset family;
  7. the independent cart model fails its structural/physical checks.

Stdlib only. Run from imgo2_rl:
    python scripts/tools/check_model_sync.py
"""

import hashlib
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_amp_dataset as audit_mod  # noqa: E402
from check_cart_model import DEFAULT_CART, check_cart  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]   # imgo2_rl/
REPO = ROOT.parent
MOTION_DIR = ROOT / "datasets/imgo2_motion"

# Agreed canonical base-link values (the recording model's).
CANONICAL_BASE_MASS = "5.53394020"
CANONICAL_BASE_INERTIA = ("0.03866860", "0.10411461", "0.12554111")

# Complete URDFs that must all agree. foot_legs is the audit script's LEGS tuple,
# i.e. the foot link name prefixes used by that file.
URDFS = {
    "imgo2_description (core)": {
        "path": REPO / "imgo2_description/urdf/imgo2.urdf",
        "foot_legs": ("FL", "FR", "RL", "RR"),
    },
    "imgo2_description (gazebo)": {
        "path": REPO / "imgo2_description/urdf/imgo2.gazebo.urdf",
        "foot_legs": ("FL", "FR", "RL", "RR"),
        # imu.xacro adds base_imu on top of the 17 core links
        "extra_links": ("base_imu",),
    },
}

# The single model mesh directory (10 files, shared by every consumer since the
# 2026-09-17 unification; LF_hip->FL_hip / L_calf->L_shank renames only).
MESH_DIR = REPO / "imgo2_description/meshes"
EXPECTED_MESH_FINGERPRINT = "8dc5b5995a11"
SHARED_MESH_DIRS = {"imgo2_description/meshes": MESH_DIR}


def logical_joints(path: Path) -> dict:
    """{logical joint -> (axis xyz, (lower, upper))} for the 12 leg joints."""
    root = ET.parse(path).getroot()
    out = {}
    for j in root.findall("joint"):
        name = j.get("name")
        if name and name.endswith(("_hip_joint", "_thigh_joint", "_shank_joint")):
            logical = name[: -len("_joint")]
        else:
            continue
        ax, lim = j.find("axis"), j.find("limit")
        out[logical] = (
            ax.get("xyz") if ax is not None else None,
            (lim.get("lower"), lim.get("upper")) if lim is not None else None,
        )
    return out


def base_inertial(path: Path):
    root = ET.parse(path).getroot()
    blk = root.find("link[@name='base']/inertial")
    if blk is None:
        return None, None
    i = blk.find("inertia")
    return (blk.find("mass").get("value").strip(),
            (i.get("ixx").strip(), i.get("iyy").strip(), i.get("izz").strip()))


# The description copy names legs LF/LH/RF/RH and calls the shank "calf".
_LINK_PREFIX = {"LF": "FL", "LH": "RL", "RF": "FR", "RH": "RR"}
_LINK_ROLE = {"hip": "HIP", "thigh": "THIGH", "calf": "SHANK", "FOOT": "FOOT"}


def logical_link(name: str) -> str:
    if name == "base":
        return "base"
    prefix, _, role = name.partition("_")
    return f"{_LINK_PREFIX.get(prefix, prefix)}_{_LINK_ROLE.get(role, role)}"


def link_physics(path: Path) -> dict:
    """{logical link -> mass/CoM/inertia/collision signature} for every link."""
    root = ET.parse(path).getroot()
    out = {}
    for link in root.findall("link"):
        parts = []
        inertial = link.find("inertial")
        if inertial is not None:
            origin = inertial.find("origin")
            inertia = inertial.find("inertia")
            parts.append("m=" + inertial.find("mass").get("value").strip())
            parts.append("o=" + (origin.get("xyz") if origin is not None else "-"))
            parts.append("i=" + ",".join(inertia.get(k).strip()
                                         for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")))
        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            for shape in geometry:
                parts.append(f"c={shape.tag}:" + ",".join(f"{k}={v}" for k, v in sorted(shape.attrib.items())))
        out[logical_link(link.get("name"))] = "|".join(parts)
    return out


def mesh_fingerprint(d: Path):
    # Path ordering folds case on Windows, but not on Linux. Hash the same
    # bytewise filename order on both hosts; do not change the reference hash.
    files = sorted((p for p in d.glob("*") if p.is_file()), key=lambda p: p.name)
    entries = [(p.name, hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
    fp = hashlib.sha256("\n".join(f"{n}:{h}" for n, h in entries).encode()).hexdigest()[:12]
    return fp, len(entries)


def main() -> int:
    failures = []

    # ---- 1. joint convention across every model ----
    print("1) leg joint axes/limits, by logical joint (order-independent)")
    conventions = {}
    for label, spec in URDFS.items():
        if not spec["path"].is_file():
            failures.append(f"{label}: file missing")
            print(f"   FAIL  {label}: not found")
            continue
        conventions[label] = logical_joints(spec["path"])
        print(f"   {label:28s} leg joints={len(conventions[label])}")

    if len(conventions) >= 2:
        ref_label = "imgo2_description (core)"
        ref = conventions.get(ref_label)
        if ref is None:
            failures.append("core URDF missing; cannot establish reference")
        else:
            for label, conv in conventions.items():
                if label == ref_label:
                    continue
                if set(conv) != set(ref):
                    failures.append(f"{label}: logical joint set differs")
                    print(f"   FAIL  {label}: joint set differs {sorted(set(conv) ^ set(ref))}")
                    continue
                bad = [k for k in sorted(ref) if ref[k] != conv[k]]
                if bad:
                    failures.append(f"{label}: {len(bad)} joint(s) differ from training")
                    for k in bad:
                        print(f"   FAIL  {k}: training={ref[k]}  {label}={conv[k]}")
                else:
                    print(f"   PASS  {label:28s} all {len(ref)} joints match the core URDF")

    # ---- 2. per-link physical parameters, by logical link ----
    print("\n2) link mass / CoM / inertia / collision, by logical link")
    physics = {}
    for label, spec in URDFS.items():
        if spec["path"].is_file():
            physics[label] = link_physics(spec["path"])
    if "imgo2_description (core)" in physics:
        ref_label = "imgo2_description (core)"
        ref = physics[ref_label]
        print(f"   reference: {ref_label} ({len(ref)} links)")
        for label, got in physics.items():
            if label == ref_label:
                continue
            # The Gazebo assembly adds an IMU frame (imu.xacro) and the legs keep the
            # 17 core links; anything else is a real divergence.
            allowed_extra = set(URDFS[label].get("extra_links", ()))
            missing = set(ref) - set(got)
            extra = set(got) - set(ref)
            if missing or (extra - allowed_extra):
                failures.append(f"{label}: link set differs")
                print(f"   FAIL  {label:28s} link set differs: missing={sorted(missing)} "
                      f"unexpected={sorted(extra - allowed_extra)}")
                continue
            if extra:
                print(f"   note  {label:28s} expected extra link(s): {sorted(extra)}")
            bad = [k for k in sorted(ref) if ref[k] != got[k]]
            if bad:
                failures.append(f"{label}: {len(bad)} link(s) differ physically")
                for k in bad[:6]:
                    print(f"   FAIL  {k}: core={ref[k]}")
                    print(f"   {'':4s}  {'':28s} {label}={got[k]}")
                if len(bad) > 6:
                    print(f"   FAIL  ... and {len(bad) - 6} more links")
            else:
                print(f"   PASS  {label:28s} all {len(ref)} links identical")
    else:
        failures.append("core URDF missing; cannot establish reference")
        print("   FAIL  core URDF not found")

    # ---- 3. base link inertial vs the agreed canonical values ----
    print("\n3) base link inertial vs canonical")
    for label, spec in URDFS.items():
        if not spec["path"].is_file():
            continue
        mass, inertia = base_inertial(spec["path"])
        ok = mass == CANONICAL_BASE_MASS and inertia == CANONICAL_BASE_INERTIA
        print(f"   {'PASS' if ok else 'FAIL'}  {label:28s} mass={mass} ixx/iyy/izz={inertia}")
        if not ok:
            failures.append(f"{label}: base inertial differs from canonical")

    # ---- 4. the single model mesh directory ----
    print("\n4) model mesh directory")
    for label, d in SHARED_MESH_DIRS.items():
        if not d.is_dir():
            failures.append(f"mesh dir missing: {label}")
            print(f"   FAIL  missing dir {label}")
            continue
        fp, n = mesh_fingerprint(d)
        ok = (n == 10 and fp == EXPECTED_MESH_FINGERPRINT)
        print(f"   {'PASS' if ok else 'FAIL'}  {label:34s} files={n:2d} fingerprint={fp} "
              f"(expected 10 / {EXPECTED_MESH_FINGERPRINT})")
        if not ok:
            failures.append(f"{label}: mesh set drifted")
    # every mesh referenced by the URDFs must exist next to them
    # (only <mesh ... file="..."> inside <asset>; Gazebo plugins also use filename=)
    # 纯 URDF 用相对路径 ../meshes/...；Gazebo 版用 package://imgo2_description/meshes/...
    # （Gazebo 经 ament 解析，见 xacro/robot.xacro 的 mesh_prefix），两种都要能解析。
    missing_refs = []
    for label, spec in URDFS.items():
        if not spec["path"].is_file():
            continue
        root = ET.parse(spec["path"]).getroot()
        for mesh in root.iter("mesh"):
            ref = mesh.get("file") or mesh.get("filename")
            if not ref:
                continue
            if ref.startswith("package://"):
                rest = ref[len("package://"):]
                pkg, _, rel = rest.partition("/")
                if pkg != "imgo2_description" or not rel:
                    missing_refs.append(f"{label}: {ref}")
                    continue
                target = (REPO / "imgo2_description" / rel).resolve()
            else:
                target = (spec["path"].parent / ref).resolve()
            if not target.is_file():
                missing_refs.append(f"{label}: {ref}")
    if missing_refs:
        failures.append(f"{len(missing_refs)} unresolvable mesh reference(s)")
        for item in missing_refs[:5]:
            print(f"   FAIL  unresolved mesh reference {item}")
    else:
        print("   PASS  every mesh referenced by the URDFs exists")

    # ---- 5. forward kinematics against the recorded data ----
    print("\n5) forward kinematics against datasets/imgo2_motion")
    if not MOTION_DIR.is_dir():
        failures.append("motion dir missing")
        print(f"   FAIL  {MOTION_DIR} not found")
    else:
        original_legs = audit_mod.LEGS
        try:
            for label, spec in URDFS.items():
                if not spec["path"].is_file():
                    continue
                audit_mod.LEGS = spec["foot_legs"]
                result = audit_mod.audit(MOTION_DIR, spec["path"])
                ident = [m["fk_coordinate_rmse_m_by_order"]["FL_FR_RL_RR"] for m in result["motions"]]
                swap = [m["fk_coordinate_rmse_m_by_order"]["FL_RL_FR_RR"] for m in result["motions"]]
                worst, best_swap = max(ident), min(swap)
                ok = worst < 0.005 and best_swap > 0.20
                print(f"   {'PASS' if ok else 'FAIL'}  {label:28s} worst identity RMSE={worst:.5f} m, "
                      f"best swapped={best_swap:.5f} m")
                if not ok:
                    failures.append(f"{label}: no longer reproduces the recorded data")
        finally:
            audit_mod.LEGS = original_legs

    # ---- 6. every URDF in the tree must be a registered one ----
    # The checks above only cover URDFS keys, so a NEW copy appearing elsewhere
    # would go unnoticed — that is precisely how the sign-inverted deploy copy
    # survived. Discover all tracked URDFs and require each to be known.
    print("\n6) coverage: tracked and non-ignored untracked URDFs are registered")
    known = {p.resolve() for p in (spec["path"] for spec in URDFS.values())}
    # Separate asset family: the cart is not an Imgo2 physics/FK copy.
    known.add(DEFAULT_CART.resolve())
    inventory = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "*.urdf"],
                               capture_output=True, text=True, cwd=REPO)
    if inventory.returncode:
        failures.append("Cannot enumerate URDF inventory")
    tracked = [name for name in inventory.stdout.split("\0") if name]
    print(f"   discovered URDF files: {len(tracked)}")
    unregistered = []
    for rel in tracked:
        if (REPO / rel).resolve() not in known:
            unregistered.append(rel)
    if unregistered:
        failures.append(f"{len(unregistered)} unregistered URDF file(s)")
        for rel in unregistered:
            print(f"   FAIL  not registered: {rel}")
        print("         register its asset family and validation; do not exempt unknown models")
    else:
        print(f"   PASS  all {len(tracked)} URDF files accounted for ({len(URDFS)} Imgo2 models + cart)")

    print("\n7) independent passive cart model")
    try:
        cart_model = check_cart(DEFAULT_CART)
        print(f"   PASS  cart: {cart_model['total_mass_kg']:g} kg, four passive wheels, valid solid inertias")
    except (ValueError, OSError, ET.ParseError, AttributeError, TypeError) as exc:
        failures.append(f"cart model: {exc}")
        print(f"   FAIL  cart: {exc}")

    # ---- informational ----
    print("\n8) notes (reported, not failures)")
    print("   - imgo2_description/ is the single model source: xacro/core.xacro holds the")
    print("     physics; urdf/imgo2.urdf and urdf/imgo2.gazebo.urdf are generated, do not edit")
    print("     (regenerate with `xacro xacro/robot.xacro [transmission:=true gazebo:=true imu:=true]`)")
    if (REPO / "imgo2_description/mjcf/scene.xml").is_file():
        print("   - imgo2_description/mjcf/{imgo2.xml,scene.xml} is the MuJoCo model used by rl_sim_mujoco")
    if (REPO / "imgo2_description/mjcf/imgo2.xml").is_file():
        print("   - the MuJoCo model carries a framelinvel sensor (adr 43) that the C++ does not read yet")

    print("\nSummary")
    if failures:
        for item in failures:
            print(f"  FAIL  {item}")
        return 1
    print(f"  PASS  all {len(URDFS)} registered URDFs agree and each reproduces the recorded data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
