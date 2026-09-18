"""同一套「只差节律」的对照，换不同数据集跑：判别器能拉开多大跨度？

动机：rl_amp（a1 数据）用了 **13 段 / 1274 帧 / 21.4 s**、覆盖 trot/pace/canter 三种步态；
我们的 imgo2_motion 是 **21 段 / 5097 帧 / 101.9 s**，但**只有一种步态节律**（0.600 s）。
数据量我们反而多 4 倍，所以「判别器退化」如果与数据有关，更可能是**模态数**而不是帧数。

做法（与 probe_gp_effect.py 同构，但每个数据集**各自标准化**，避免用我们的 normalizer 去套他们的数据）：
  正类 = 该数据集正常速度的相邻帧对 (s_t, s_{t+1})
  负类 = 同一份数据 3 倍速的帧对 (s_t, s_{t+3})
  网络 [1024,512]+ReLU、MSE(±1)、λ_gp=10、同预算
比较：d(正)/d(负) 的跨度、以及按 AMP 奖励公式折算的「走得对 vs 走快 3 倍」奖励差。
"""

import argparse
import glob
import importlib.util
import sys
import types

import numpy as np
import torch
import torch.nn as nn

_RL = "imgo2_rl/scripts/rl_lab"
sys.path.insert(0, _RL)
sys.path.insert(0, ".local-py")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_pkg = types.ModuleType("rl_lab")
_pkg.__path__ = [f"{_RL}/rl_lab"]
sys.modules["rl_lab"] = _pkg
for _sub in ("utils", "datasets", "algorithms"):
    _m = types.ModuleType(f"rl_lab.{_sub}")
    _m.__path__ = [f"{_RL}/rl_lab/{_sub}"]
    sys.modules[f"rl_lab.{_sub}"] = _m
    setattr(_pkg, _sub, _m)
sys.modules["rl_lab.utils"].utils = _load("rl_lab.utils.utils", f"{_RL}/rl_lab/utils/utils.py")
sys.modules["rl_lab.datasets"].pose3d = _load("rl_lab.datasets.pose3d", f"{_RL}/rl_lab/datasets/pose3d.py")
sys.modules["rl_lab.datasets"].motion_util = _load("rl_lab.datasets.motion_util", f"{_RL}/rl_lab/datasets/motion_util.py")
AMPLoader = _load("rl_lab.datasets.motion_loader", f"{_RL}/rl_lab/datasets/motion_loader.py").AMPLoader

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="动作目录（*.txt）")
ap.add_argument("--label", default="")
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--lam", type=float, default=10.0)
ap.add_argument("--n", type=int, default=8000)
args = ap.parse_args()

torch.manual_seed(0)
np.random.seed(0)
files = sorted(glob.glob(args.data + "/*.txt"))
loader = AMPLoader(device="cpu", time_between_frames=0.02, preload_transitions=True,
                   num_preload_transitions=200000, motion_files=files)
lens = np.asarray(loader.trajectory_lens)
n_frames = sum(int(t.shape[0]) for t in loader.trajectories)


def amp_obs(fr):
    return torch.cat([fr[:, 7:49], fr[:, 2:3]], dim=-1)


def sample(scale, n):
    traj = np.random.randint(0, len(loader.trajectories), n)
    t = np.random.rand(n) * np.maximum(lens[traj] - 0.02 * scale - 0.1, 1e-3)
    return (amp_obs(loader.get_full_frame_at_time_batch(traj, t)),
            amp_obs(loader.get_full_frame_at_time_batch(traj, t + 0.02 * scale)))


real_s, real_n = sample(1.0, args.n)
fake_s, fake_n = sample(3.0, args.n)
# 每个数据集各自标准化（用正类样本的统计）
mean = real_s.mean(0, keepdim=True)
std = real_s.std(0, keepdim=True).clamp_min(1e-3)


def norm(x):
    return torch.clamp((x - mean) / std, -10, 10)


real_x = torch.cat([norm(real_s), norm(real_n)], -1)
fake_x = torch.cat([norm(fake_s), norm(fake_n)], -1)


class Disc(nn.Module):
    def __init__(self, dim=86, hidden=(1024, 512)):
        super().__init__()
        layers, cur = [], dim
        for h in hidden:
            layers += [nn.Linear(cur, h), nn.ReLU()]
            cur = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(cur, 1)

    def forward(self, x):
        return self.head(self.trunk(x))


def grad_pen(model, x, lam):
    x = x.clone().requires_grad_(True)
    d = model(x)
    g = torch.autograd.grad(d, x, torch.ones_like(d), create_graph=True)[0]
    return lam * (g.norm(2, dim=1) - 0).pow(2).mean()


model = Disc()
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
for step in range(args.steps):
    ib = torch.randint(0, args.n, (256,))
    d_r, d_f = model(real_x[ib]), model(fake_x[ib])
    loss = 0.5 * (nn.functional.mse_loss(d_r, torch.ones_like(d_r))
                  + nn.functional.mse_loss(d_f, -torch.ones_like(d_f)))
    if args.lam > 0:
        loss = loss + grad_pen(model, real_x[ib], args.lam)
    opt.zero_grad()
    loss.backward()
    opt.step()
model.eval()
with torch.no_grad():
    dr = float(model(real_x[:4096]).mean())
    df = float(model(fake_x[:4096]).mean())


def style(d):
    return 2.0 * max(0.0, 1.0 - (d - 1.0) ** 2 / 4.0)


print(f"[{args.label or args.data}] 文件 {len(files)} 个 / 帧 {n_frames} / "
      f"时长 {float(np.sum(lens)):.1f}s / λ_gp={args.lam:g} / {args.steps} 步", flush=True)
print(f"   d(1×)={dr:+.4f}  d(3×)={df:+.4f}  跨度={dr - df:.4f}  "
      f"风格奖励差/步={(style(dr) - style(df)) * 0.7:.4f}", flush=True)
