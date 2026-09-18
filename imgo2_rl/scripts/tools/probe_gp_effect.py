"""离线验证「梯度惩罚压平判别器」这一假设。

背景：24500 轮的判别器在**专家点**与**策略点**上的局部梯度都只有 ~1e-4（1σ 全维扰动只能
移动 d 约 5e-4），而它同时又能把策略判成 d≈−0.15（专家 +0.57）。也就是说风格奖励是个
**台阶**：告诉策略"你不在流形上"，但**不给方向**。可疑项是 `compute_grad_pen(..., lambda_=10)`：
它显式把 d 在专家数据上的梯度压向 0。

本脚本用一个**合成但同构**的判别问题直接测这件事：
  正类 = 录制动作按正常速度的相邻帧对 (s_t, s_{t+1})
  负类 = 同一份动作按 3 倍速播放的帧对 (s_t, s_{t+3})   ← 正是我们实测的步频倍数
用与训练完全相同的网络 [1024,512]+ReLU、相同的 MSE(±1) 损失、相同的归一化，
只改 lambda_gp ∈ {10, 1, 0}，然后比较：
  * d(正类) / d(负类)（判别器能不能分开）
  * 两侧的局部梯度 |∂d/∂x|·σ（策略能不能顺着它爬）
  * 按 AMP 风格奖励公式折算出的奖励差
结论若为「λ=10 梯度≈0、λ=0 有明显梯度」，则修判别器就有据可依。
"""

import glob
import importlib.util
import sys
import types

import argparse

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
ap.add_argument("--steps", type=int, default=1500, help="每个 λ 的判别器训练步数")
ap.add_argument("--lams", type=float, nargs="+", default=[10.0, 1.0, 0.0])
args = ap.parse_args()

CKPT = "imgo2_rl/logs/amp_rsl_rl/base_move_amp/2026-09-17_21-25-57/model_24500.pt"
torch.manual_seed(0)
np.random.seed(0)

loader = AMPLoader(device="cpu", time_between_frames=0.02, preload_transitions=True,
                   num_preload_transitions=400000,
                   motion_files=sorted(glob.glob("imgo2_rl/datasets/imgo2_motion/*.txt")))
traj_lens = np.asarray(loader.trajectory_lens)


def amp_obs(frames):
    return torch.cat([frames[:, 7:49], frames[:, 2:3]], dim=-1)


def sample(scale, n):
    """scale=1 → 正类（正常速度）；scale=3 → 负类（3 倍速）。"""
    traj = np.random.randint(0, len(loader.trajectories), n)
    t = np.random.rand(n) * (traj_lens[traj] - 0.02 * scale - 0.1)
    a = amp_obs(loader.get_full_frame_at_time_batch(traj, t))
    b = amp_obs(loader.get_full_frame_at_time_batch(traj, t + 0.02 * scale))
    return a, b


n = 8000
real_s, real_n = sample(1.0, n)
fake_s, fake_n = sample(3.0, n)

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
nrm = ck["amp_normalizer"]
mean = np.asarray(nrm["mean"], np.float32).reshape(1, -1)
std = np.sqrt(np.asarray(nrm["var"], np.float32).reshape(1, -1) + float(nrm["epsilon"]))
mt, st = torch.from_numpy(mean), torch.from_numpy(std)


def norm(a):
    return torch.clamp((a - mt) / st, -10, 10)


R = norm(real_s); RN = norm(real_n)
F = norm(fake_s); FN = norm(fake_n)
real_x = torch.cat([R, RN], -1)
fake_x = torch.cat([F, FN], -1)


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


def local_grad_sum(model, x):
    """|∂d/∂x|·σ 在归一化空间的合计（σ 在归一化空间里恒为 1，所以就是 |∂d/∂x_norm| 之和）。"""
    x = x.clone().requires_grad_(True)
    d = model(x).mean()
    g = torch.autograd.grad(d, x)[0]
    return float(g.abs().sum(dim=1).mean())


def style(d):
    return 2.0 * max(0.0, 1.0 - (d - 1.0) ** 2 / 4.0)


print(f"正类(1×) {real_x.shape}  负类(3×) {fake_x.shape}", flush=True)
print(f"\n{'λ_gp':>5} {'d(正类)':>10} {'d(负类)':>10} {'d 跨度':>9} "
      f"{'梯度(正)':>11} {'梯度(负)':>11} {'风格差/步':>11}")
for lam in args.lams:
    torch.manual_seed(0)
    model = Disc()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for step in range(args.steps):
        ib = torch.randint(0, n, (256,))
        d_r = model(real_x[ib])
        d_f = model(fake_x[ib])
        loss = (0.5 * (nn.functional.mse_loss(d_r, torch.ones_like(d_r))
                       + nn.functional.mse_loss(d_f, -torch.ones_like(d_f))))
        if lam > 0:
            loss = loss + grad_pen(model, real_x[ib], lam)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        dr = float(model(real_x[:8192]).mean())
        df = float(model(fake_x[:8192]).mean())
    gr = local_grad_sum(model, real_x[:8192])
    gf = local_grad_sum(model, fake_x[:8192])
    print(f"{lam:5.0f} {dr:+10.4f} {df:+10.4f} {dr - df:9.4f} "
          f"{gr:11.3f} {gf:11.3f} {(style(dr) - style(df)) * 0.7:11.4f}", flush=True)

print("\n（参考：现网判别器在专家/策略点上的梯度合计都只有 ~1e-4；"
      "正类=1×帧对、负类=3×帧对，只差节律）")
