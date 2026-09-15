import os
from pathlib import Path


def _glob_motion_files() -> list[str]:
    motion_dirs = [
        os.environ.get("IMGO2_AMP_MOTION_DIR"),
        Path.cwd() / "datasets" / "mocap_motions",
        Path.cwd().parent / "AMP_for_hardware-main" / "datasets" / "mocap_motions",
        Path("/root/gpufree-data/RL/AMP/AMP_for_hardware-main/datasets/mocap_motions"),
        Path("/root/gpufree-data/AMP_for_hardware-main/datasets/mocap_motions"),
        Path(r"C:\Users\qmq\Desktop\RL\AMP\AMP_for_hardware-main\datasets\mocap_motions"),
    ]
    for motion_dir in motion_dirs:
        if motion_dir is None:
            continue
        motion_path = Path(motion_dir).expanduser()
        motion_files = sorted(str(path) for path in motion_path.glob("*") if path.is_file())
        if motion_files:
            return motion_files
    return []


AMP_MOTION_FILES = _glob_motion_files()
