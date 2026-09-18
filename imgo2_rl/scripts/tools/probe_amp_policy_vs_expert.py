"""策略侧 AMP 观测 vs 专家：判别器到底靠哪一块把策略判成"假"。

配合 `eval_gait.py --dump-npz` 产出的 .npz 使用（里面含策略侧 43 维 `amp_obs`）。
全部在 CPU 上跑，不需要 GPU / Isaac Lab。

三类对照：
  A. 策略自己的 (s, s_next) 对 → 复现训练日志里的 d_policy；
  B. **把专家观测的某一块替换成策略的对应块** → 看哪一块单独就能把 d 拉下来
     （这是"判别器指纹在哪"的直接答案）；
  C. 反向：把策略的某一块替换成专家块 → 看哪一块能救回 d。
"""

import argparse
import glob
import importlib.util
import sys
import types

import numpy as np
import torch

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
AMPDiscriminator = _load("rl_lab.algorithms.amp_discriminator",
                         f"{_RL}/rl_lab/algorithms/amp_discriminator.py").AMPDiscriminator
AMPLoader = _load("rl_lab.datasets.motion_loader", f"{_RL}/rl_lab/datasets/motion_loader.py").AMPLoader

ap = argparse.ArgumentParser()
ap.add_argument("--npz", default="imgo2_rl/logs/gait_24500_vx1.0.npz")
ap.add_argument("--checkpoint",
                default="imgo2_rl/logs/amp_rsl_rl/base_move_amp/2026-09-17_21-25-57/model_24500.pt")
ap.add_argument("--batch", type=int, default=8192)
ap.add_argument("--warmup", type=int, default=50)
args = ap.parse_args()

BLOCKS = {"joint_pos": (0, 12), "foot_pos": (12, 24), "lin_vel": (24, 27),
          "ang_vel": (27, 30), "joint_vel": (30, 42), "root_z": (42, 43)}
torch.manual_seed(0)

# ---- 策略侧观测 ----
z = np.load(args.npz)
obsT = z["amp_obs"][args.warmup:]                         # [T,E,43]，与训练同口径的原始观测
T, E, _ = obsT.shape
cmdT = z["cmd"][args.warmup:]                             # [T,E,3]
nb = min(args.batch, (T - 1) * E)
tt = np.random.randint(0, T - 1, size=nb)
ee = np.random.randint(0, E, size=nb)
pol_s = torch.from_numpy(obsT[tt, ee])                    # (s_t, s_{t+1}) 必须取**同一环境**的相邻帧
pol_n = torch.from_numpy(obsT[tt + 1, ee])
print(f"策略侧: {args.npz} → {T}×{E} 帧；指令均值 vx={cmdT[..., 0].mean():.3f} "
      f"vy={cmdT[..., 1].mean():.3f} yaw={cmdT[..., 2].mean():.3f}")

# ---- 专家侧 ----
loader = AMPLoader(device="cpu", time_between_frames=0.02, preload_transitions=True,
                   num_preload_transitions=400000,
                   motion_files=sorted(glob.glob("imgo2_rl/datasets/imgo2_motion/*.txt")))
gen = loader.feed_forward_generator(num_mini_batch=1, mini_batch_size=nb)   # 与策略侧同 batch
exp_s, exp_n = next(gen)

# ---- 判别器 ----
ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
dsd = ck["discriminator_state_dict"]
hid = [dsd[f"trunk.{i}.weight"].shape[0] for i in range(0, len(dsd), 2) if f"trunk.{i}.weight" in dsd]
disc = AMPDiscriminator(dsd["trunk.0.weight"].shape[1], 2.0, hid, "cpu", task_reward_lerp=0.3)
disc.load_state_dict(dsd)
disc.eval()
nrm = ck["amp_normalizer"]
mean = np.asarray(nrm["mean"], np.float32).reshape(1, -1)
std = np.sqrt(np.asarray(nrm["var"], np.float32).reshape(1, -1) + float(nrm["epsilon"]))
mt, st = torch.from_numpy(mean), torch.from_numpy(std)


def d_of(a, b):
    with torch.no_grad():
        na = torch.clamp((a - mt) / st, -10, 10)
        nb = torch.clamp((b - mt) / st, -10, 10)
        return float(disc(torch.cat([na, nb], -1)).mean())


d_exp = d_of(exp_s, exp_n)
d_pol = d_of(pol_s, pol_n)
print(f"\nA. d(专家) = {d_exp:+.4f}   d(策略) = {d_pol:+.4f}   跨度 {d_exp - d_pol:.4f}")


def style(d):
    return 2.0 * max(0.0, 1.0 - (d - 1.0) ** 2 / 4.0)


print(f"   → 风格项/步（未加权）: 专家 {style(d_exp):.3f} / 策略 {style(d_pol):.3f}；"
      f"混合后（×0.7）差距 {(style(d_exp) - style(d_pol)) * 0.7:.3f}/步")

print("\nB. 把**专家**观测的某一块换成策略的对应块（哪一块单独就能把 d 拉下来）：")
print(f"{'换入策略的块':14s} {'d':>9} {'Δ vs 专家':>12} {'风格差(×0.7)':>14}")
for name, (a, b) in BLOCKS.items():
    m = exp_s.clone(); mn = exp_n.clone()
    m[:, a:b] = pol_s[:, a:b]; mn[:, a:b] = pol_n[:, a:b]
    d = d_of(m, mn)
    print(f"{name:14s} {d:+9.4f} {d - d_exp:+12.4f} {(style(d_exp) - style(d)) * 0.7:14.3f}")
m = exp_s.clone(); mn = exp_n.clone()
m[:, 0:12] = pol_s[:, 0:12]; mn[:, 0:12] = pol_n[:, 0:12]
m[:, 12:24] = pol_s[:, 12:24]; mn[:, 12:24] = pol_n[:, 12:24]
d = d_of(m, mn)
print(f"{'joint+foot':14s} {d:+9.4f} {d - d_exp:+12.4f} {(style(d_exp) - style(d)) * 0.7:14.3f}")

print("\nC. 把**策略**观测的某一块换成专家的对应块（哪一块能救回 d）：")
print(f"{'换入专家的块':14s} {'d':>9} {'Δ vs 策略':>12}")
for name, (a, b) in BLOCKS.items():
    m = pol_s.clone(); mn = pol_n.clone()
    m[:, a:b] = exp_s[:, a:b]; mn[:, a:b] = exp_n[:, a:b]
    d = d_of(m, mn)
    print(f"{name:14s} {d:+9.4f} {d - d_pol:+12.4f}")
m = pol_s.clone(); mn = pol_n.clone()
m[:, 0:12] = exp_s[:, 0:12]; mn[:, 0:12] = exp_n[:, 0:12]
m[:, 12:24] = exp_s[:, 12:24]; mn[:, 12:24] = exp_n[:, 12:24]
d = d_of(m, mn)
print(f"{'joint+foot':14s} {d:+9.4f} {d - d_pol:+12.4f}")

print("\nE. 逐步把策略的块换成专家（看「步态块能救回多少」与「命令/域块占多少」）：")
sets = [("只 joint_vel", [BLOCKS["joint_vel"]]),
        ("joint_pos+foot_pos", [BLOCKS["joint_pos"], BLOCKS["foot_pos"]]),
        ("步态三块(姿+足+关节速度)", [BLOCKS["joint_pos"], BLOCKS["foot_pos"], BLOCKS["joint_vel"]]),
        ("+lin_vel", [BLOCKS["joint_pos"], BLOCKS["foot_pos"], BLOCKS["joint_vel"], BLOCKS["lin_vel"]]),
        ("+ang_vel", [BLOCKS["joint_pos"], BLOCKS["foot_pos"], BLOCKS["joint_vel"], BLOCKS["lin_vel"],
                      BLOCKS["ang_vel"]]),
        ("+root_z（=全换成专家）", list(BLOCKS.values()))]
for label, blks in sets:
    m, mn = pol_s.clone(), pol_n.clone()
    for a, b in blks:
        m[:, a:b] = exp_s[:, a:b]
        mn[:, a:b] = exp_n[:, a:b]
    d = d_of(m, mn)
    print(f"  {label:26s} d = {d:+.4f}  风格项/步 {(style(d)):.3f}  "
          f"（占专家可达 {100 * (style(d) - style(d_pol)) / max(style(d_exp) - style(d_pol), 1e-9):.0f}%）")

print("\nF. 策略点上还有没有局部梯度（决定「加权重」能不能把 d 推上去）：")
for label, base_s, base_n in (("专家", exp_s, exp_n), ("策略", pol_s, pol_n)):
    x = base_s.clone().requires_grad_(True)
    y = base_n.clone().requires_grad_(True)
    dmean = disc(torch.cat([torch.clamp((x - mt) / st, -10, 10),
                            torch.clamp((y - mt) / st, -10, 10)], -1)).mean()
    g = torch.autograd.grad(dmean, [x, y])
    imp = (g[0].abs().mean(0) * st.squeeze(0))
    print(f"  {label}点: |∂d/∂x|·σ 合计 = {imp.sum():.2e}（s 段）"
          f"  最大单维 {imp.max():.2e}  1σ 全维扰动可移动 d ≈ {imp.sum() * np.sqrt(len(imp)):.2e}")

print("\nD. 归一化空间里各块的均值/标准差（策略 vs 专家）—— 直接看差在哪：")
pe = torch.clamp((exp_s - mt) / st, -10, 10)
pp = torch.clamp((pol_s - mt) / st, -10, 10)
print(f"{'块':12s} {'专家均值':>10s} {'策略均值':>10s} {'专家σ':>8s} {'策略σ':>8s} {'均值偏移σ':>10s}")
for name, (a, b) in BLOCKS.items():
    e_mu, p_mu = pe[:, a:b].mean().item(), pp[:, a:b].mean().item()
    e_sd, p_sd = pe[:, a:b].std().item(), pp[:, a:b].std().item()
    print(f"{name:12s} {e_mu:10.3f} {p_mu:10.3f} {e_sd:8.3f} {p_sd:8.3f} {abs(p_mu - e_mu):10.3f}")
