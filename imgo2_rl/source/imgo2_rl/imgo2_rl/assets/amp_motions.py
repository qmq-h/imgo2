import os
from pathlib import Path

# 以本文件位置推导路径，去掉原先写死的机器绝对路径。
# 本文件位于 <imgo2_rl>/source/imgo2_rl/imgo2_rl/assets/amp_motions.py。
_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _glob_motion_files() -> list[str]:
    """Locate mocap motion files by trying several candidate directories.

    候选顺序：环境变量、imgo2_rl 内的 datasets/、仓库同级的 A1 参考项目、当前目录。
    注意当前 AMP 环境与 runner 导入的是 `assets.imgo2.AMP_MOTION_FILES`，不经过本模块；
    这里的 mocap 数据属于另一个参考项目，见 README 相应说明。
    """
    motion_dirs = [
        os.environ.get("IMGO2_AMP_MOTION_DIR"),
        _PROJECT_ROOT / "datasets" / "mocap_motions",
        _PROJECT_ROOT.parent / "AMP_for_hardware-main" / "datasets" / "mocap_motions",
        Path.cwd() / "datasets" / "mocap_motions",
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
