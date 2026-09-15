"""Check that every Imgo2 model agrees on joint convention, and that each URDF
still reproduces the recorded motion data.

Why this exists: the deploy URDF was once fully sign-inverted on all 12 leg
joints (q_deploy = -q_train), which silently disagreed with the real-robot data
the AMP reference motions were recorded from. Signs and limits must therefore be
checked across every model, not just the one being edited.

Comparison is by *logical joint* (leg + joint role), so the deliberate leg-order
difference of `imgo2_description/` is tolerated: that copy names its joints
LF_HAA/LH_HFE/... and orders legs LF, LH, RF, RH, while the others use
FL_hip_joint/... ordered FL, FR, RL, RR.

Fails when any of these breaks:
  1. a leg joint's axis or limits differs between any two complete URDFs;
  2. a link's mass, centre of mass, inertia tensor or collision geometry differs
     (compared by logical link, so the description copy's LF_/LH_ naming and its
     leg order are tolerated);
  3. a base link inertial block drifts from the agreed canonical values
     (user decision: the recording model's values);
  4. the mesh set shared by the training, deploy and `imgo2_model/` copies diverges;
  5. any URDF stops reproducing `datasets/imgo2_motion` through forward kinematics.

Stdlib only. Run from imgo2_rl:
    python scripts/tools/check_model_sync.py
"""

import hashlib
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_amp_dataset as audit_mod  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]   # imgo2_rl/
REPO = ROOT.parent
MOTION_DIR = ROOT / "datasets/imgo2_motion"

# Agreed canonical base-link values (the recording model's).
CANONICAL_BASE_MASS = "5.53394020"
CANONICAL_BASE_INERTIA = ("0.03866860", "0.10411461", "0.12554111")

# Complete URDFs that must all agree. foot_legs is the audit script's LEGS tuple,
# i.e. the foot link name prefixes used by that file.
URDFS = {
    "imgo2_rl (training)": {
        "path": ROOT / "source/imgo2_rl/data/imgo2_model/imgo2_urdf/urdf/imgo2.urdf",
        "foot_legs": ("FL", "FR", "RL", "RR"),
    },
    "imgo2_deploy": {
        "path": REPO / "imgo2_deploy/robot_description/imgo2_urdf/urdf/imgo2.urdf",
        "foot_legs": ("FL", "FR", "RL", "RR"),
    },
    "imgo2_description.urdf": {
        "path": REPO / "imgo2_description/urdf/imgo2_description.urdf",
        "foot_legs": ("LF", "RF", "LH", "RH"),
    },
    "imgo2_description/imgo2.urdf": {
        "path": REPO / "imgo2_description/urdf/imgo2.urdf",
        "foot_legs": ("FL", "FR", "RL", "RR"),
    },
}

# description joint name -> training joint name (leg order differs; roles map 1:1)
DESC_TO_TRAIN = {
    "LF_HAA": "FL_hip_joint", "LF_HFE": "FL_thigh_joint", "LF_KFE": "FL_shank_joint",
    "LH_HAA": "RL_hip_joint", "LH_HFE": "RL_thigh_joint", "LH_KFE": "RL_shank_joint",
    "RF_HAA": "FR_hip_joint", "RF_HFE": "FR_thigh_joint", "RF_KFE": "FR_shank_joint",
    "RH_HAA": "RR_hip_joint", "RH_HFE": "RR_thigh_joint", "RH_KFE": "RR_shank_joint",
}

SHARED_MESH_DIRS = {
    "imgo2_rl/.../imgo2_urdf/meshes": ROOT / "source/imgo2_rl/data/imgo2_model/imgo2_urdf/meshes",
    "imgo2_deploy/.../meshes": REPO / "imgo2_deploy/robot_description/imgo2_urdf/meshes",
    "imgo2_model/imgo2_urdf/meshes": REPO / "imgo2_model/imgo2_urdf/meshes",
}


def logical_joints(path: Path) -> dict:
    """{logical joint -> (axis xyz, (lower, upper))} for the 12 leg joints."""
    root = ET.parse(path).getroot()
    out = {}
    for j in root.findall("joint"):
        name = j.get("name")
        if name in DESC_TO_TRAIN:
            logical = DESC_TO_TRAIN[name][: -len("_joint")]
        elif name and name.endswith(("_hip_joint", "_thigh_joint", "_shank_joint")):
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
    files = sorted(p for p in d.glob("*") if p.is_file())
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
        ref_label = "imgo2_rl (training)"
        ref = conventions.get(ref_label)
        if ref is None:
            failures.append("training URDF missing; cannot establish reference")
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
                    print(f"   PASS  {label:28s} all {len(ref)} joints match training")

    # ---- 2. per-link physical parameters, by logical link ----
    print("\n2) link mass / CoM / inertia / collision, by logical link")
    physics = {}
    for label, spec in URDFS.items():
        if spec["path"].is_file():
            physics[label] = link_physics(spec["path"])
    if "imgo2_rl (training)" in physics:
        ref_label = "imgo2_rl (training)"
        ref = physics[ref_label]
        print(f"   reference: {ref_label} ({len(ref)} links)")
        for label, got in physics.items():
            if label == ref_label:
                continue
            if set(got) != set(ref):
                failures.append(f"{label}: link set differs")
                print(f"   FAIL  {label:28s} link set differs: {sorted(set(got) ^ set(ref))}")
                continue
            bad = [k for k in sorted(ref) if ref[k] != got[k]]
            if bad:
                failures.append(f"{label}: {len(bad)} link(s) differ physically")
                for k in bad[:6]:
                    print(f"   FAIL  {k}: training={ref[k]}")
                    print(f"   {'':4s}  {'':28s} {label}={got[k]}")
                if len(bad) > 6:
                    print(f"   FAIL  ... and {len(bad) - 6} more links")
            else:
                print(f"   PASS  {label:28s} all {len(ref)} links identical")
    else:
        failures.append("training URDF missing; cannot establish reference")
        print("   FAIL  training URDF not found")

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

    # ---- 4. shared mesh set ----
    print("\n4) mesh set shared by three copies")
    prints = {}
    for label, d in SHARED_MESH_DIRS.items():
        if not d.is_dir():
            failures.append(f"mesh dir missing: {label}")
            print(f"   FAIL  missing dir {label}")
            continue
        fp, n = mesh_fingerprint(d)
        prints[label] = fp
        print(f"   {label:34s} files={n:2d} fingerprint={fp}")
    if prints:
        if len(set(prints.values())) > 1:
            failures.append("shared mesh copies differ")
            print("   FAIL  fingerprints differ")
        else:
            print("   PASS  all three copies byte-identical")

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
    print("\n6) coverage: every tracked URDF is a registered one")
    known = {p.resolve() for p in (spec["path"] for spec in URDFS.values())}
    known.add((REPO / "imgo2_model/imgo2_urdf/urdf/imgo2.urdf").resolve())  # del leg-only fragment
    tracked = subprocess.run(["git", "ls-files", "*.urdf"], capture_output=True,
                             text=True, cwd=REPO).stdout.split()
    print(f"   tracked URDF files: {len(tracked)}")
    unregistered = []
    for rel in tracked:
        if (REPO / rel).resolve() not in known:
            unregistered.append(rel)
    if unregistered:
        failures.append(f"{len(unregistered)} unregistered URDF file(s)")
        for rel in unregistered:
            print(f"   FAIL  not registered: {rel}")
        print("         add it to URDFS (if it must agree) or to `known` above")
    else:
        print(f"   PASS  all {len(tracked)} URDF files are accounted for "
              f"({len(URDFS)} checked + fragment)")

    # ---- informational ----
    print("\n7) known-intentional / orphan items (reported, not failures)")
    print("   - imgo2_description/ leg order LF,LH,RF,RH differs by design (another implementation)")
    frag = REPO / "imgo2_model/imgo2_urdf/urdf/imgo2.urdf"
    if frag.is_file():
        print(f"   - {frag.relative_to(REPO)} is a leg-only fragment (no base link, no <robot>)")
    for rel in ("imgo2_description/xacro/common/leg.xacro",
                "imgo2_description/urdf/imgo2.urdf"):
        if (REPO / rel).is_file():
            print(f"   - {rel} is not referenced by any file in the workspace")
    mjcf = REPO / "imgo2_model/imgo2_mjcf/meshes"
    if mjcf.is_dir():
        fp, n = mesh_fingerprint(mjcf)
        print(f"   - imgo2_model/imgo2_mjcf/meshes uses MuJoCo naming: files={n} fingerprint={fp}")
    if (REPO / "imgo2_description/mjcf/scene.xml").is_file():
        print("   - imgo2_description/mjcf/scene.xml exists (lead for README DEPLOY-02, which needs a MuJoCo scene)")

    print("\nSummary")
    if failures:
        for item in failures:
            print(f"  FAIL  {item}")
        return 1
    print("  PASS  all four URDFs agree and each reproduces the recorded data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
