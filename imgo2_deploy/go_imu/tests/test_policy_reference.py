"""Compare deployment C++ inference with the same existing TorchScript model, never export a model."""
import pathlib
import random
import subprocess
import sys
import tempfile
import torch
exe, directory = sys.argv[1:]
torch.set_num_threads(1)
model = torch.jit.load(str(pathlib.Path(directory) / 'policy.pt'), map_location='cpu').eval()
rng = random.Random(17)
inputs = [[rng.uniform(-2, 2) for _ in range(45)] for _ in range(64)]
with tempfile.TemporaryDirectory(prefix='go_imu_policy_check_') as tmp:
    source = pathlib.Path(tmp) / 'inputs.txt'
    source.write_text('\n'.join(' '.join(format(v, '.9g') for v in row) for row in inputs))
    result = subprocess.run([exe, directory, str(source)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    outputs = [[float(v) for v in line.split()] for line in result.stdout.splitlines()]
    assert len(outputs) == len(inputs)
    worst = 0.0
    with torch.no_grad():
        for row, out in zip(inputs, outputs):
            # Same decimal text parsed by the C++ client, then float32.
            tensor = torch.tensor([[float(format(v, '.9g')) for v in row]], dtype=torch.float32)
            reference = model(tensor).flatten().tolist()
            assert len(reference) == len(out) == 12
            worst = max(worst, max(abs(a-b) for a,b in zip(reference,out)))
    assert worst < 1e-5, worst
    print('64 deterministic inputs C++ / Python max diff:', worst)
