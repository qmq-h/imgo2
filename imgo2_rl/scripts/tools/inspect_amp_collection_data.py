"""Inspect stored AMP clips; timestamps/contact cannot be recovered from 61 columns."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def inspect(motion_dir):
    rows = []
    for path in sorted(motion_dir.glob('*.txt')):
        raw = path.read_bytes()
        data = json.loads(raw)
        frames = data['Frames']
        if not frames or any(len(f) != 61 or not all(math.isfinite(v) for v in f) for f in frames):
            raise ValueError(f'Invalid 61-column frames: {path}')
        rows.append({
            'file': path.name, 'frames': len(frames), 'frame_dt': data['FrameDuration'],
            'motion_weight': data['MotionWeight'],
            'root_speed_mean': statistics.mean(math.hypot(f[31], f[32]) for f in frames),
            'yaw_mean': statistics.mean(f[36] for f in frames),
            'adjacent_equal_joint_positions': sum(a[7:19] == b[7:19] for a, b in zip(frames, frames[1:])),
            'adjacent_equal_foot_positions': sum(a[19:31] == b[19:31] for a, b in zip(frames, frames[1:])),
            'sha256': hashlib.sha256(raw).hexdigest(),
        })
    if not rows:
        raise ValueError(f'No clips in {motion_dir}')
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--motion-dir', type=Path, default=Path(__file__).resolve().parents[2] / 'datasets/imgo2_motion')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    rows = inspect(args.motion_dir)
    text = json.dumps(rows, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
