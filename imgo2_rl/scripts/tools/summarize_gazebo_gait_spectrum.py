"""Offline joint-waveform cross-check; standard library only, no contact claim."""
import argparse
import ast
import cmath
import hashlib
import json
import math
import statistics
from pathlib import Path


def fft(values):
    n = len(values)
    if n == 1:
        return values
    even, odd = fft(values[::2]), fft(values[1::2])
    factors = [cmath.exp(-2j * math.pi * k / n) * odd[k] for k in range(n // 2)]
    return [even[k] + factors[k] for k in range(n // 2)] + [even[k] - factors[k] for k in range(n // 2)]


def analyze(path, autocorr):
    raw = path.read_bytes()
    data = json.loads(raw)
    joints = data['joint_states']
    dt = statistics.median(b[0] - a[0] for a, b in zip(joints, joints[1:]) if b[0] > a[0])
    result = {'raw_sha256': hashlib.sha256(raw).hexdigest(), 'sample_dt_s': dt, 'per_leg': {}}
    for leg in ['FL', 'FR', 'RL', 'RR']:
        y = [entry[1][leg + '_thigh_joint'][0] for entry in joints]
        mean = statistics.mean(y)
        y = [v - mean for v in y]
        period, score, frequency = autocorr(y, dt)
        size = 1 << (len(y) - 1).bit_length()
        windowed = [v * (0.5 - 0.5 * math.cos(2 * math.pi * i / (len(y) - 1))) for i, v in enumerate(y)]
        transformed = fft(windowed + [0.0] * (size - len(y)))
        power = [abs(v) ** 2 for v in transformed[:size // 2 + 1]]
        peaks = [i for i in range(1, len(power) - 1) if 0.7 < i / (size * dt) < 15
                 and power[i] > power[i - 1] and power[i] > power[i + 1]]
        peaks.sort(key=lambda i: power[i], reverse=True)
        result['per_leg'][leg] = {
            'autocorr_period_s': period if math.isfinite(period) else None,
            'autocorr_score': score if math.isfinite(score) else None,
            'autocorr_frequency_hz': frequency if math.isfinite(frequency) else None,
            'fft_top_peaks_hz': [round(i / (size * dt), 6) for i in peaks[:3]],
        }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('raw_files', nargs='+', type=Path)
    ap.add_argument('--json', type=Path)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[3]
    source = root / 'imgo2_deploy/ros2_ws/scripts/eval_gazebo_policy.py'
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'autocorr_peak')
    namespace = {'math': math}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    results = {str(p): analyze(p, namespace['autocorr_peak']) for p in args.raw_files}
    text = json.dumps(results, indent=2, allow_nan=False)
    if args.json:
        args.json.write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
