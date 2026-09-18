"""离线判别器探针：用已存的 discriminator 直接回答「风格项到底在看什么」。

为什么需要它：24500 轮训练里 `AMP/disc_policy_pred` 从 −0.42 变到 −0.57 就再也不动了，
而速度误差同时从 1.11 降到 0.47 —— 说明**任务项有方向、风格项几乎是个常数**。
如果判别器根本没在看步态（步频/步幅）相关的输入块，那么「提高风格权重」只会加一个常数，
不会改变步态；要改的是观测/数据/判别器，而不是权重。

做法（全部在 CPU 上，不需要 GPU / Isaac Lab）：
  1. 用 rl_lab 自己的 AMPLoader 取真实专家 (s, s_next)（43 维 ×2，与训练完全同源）；
  2. 载入 model_24500.pt 的 discriminator + amp_normalizer（与 runner 的 predict_amp_reward 同口径）；
  3. 对输入做「破坏性对照」，看 d 掉不掉：
       - 打乱 s_next（切断时序一致性）    → 若 d 不掉，判别器没在用「转移/节律」
       - (s, s) 零时间差 / 时间反序
       - 逐块置零（关节角/足端/线速度/角速度/关节速度/根高）→ 看它依赖哪一块
       - 只把根高整体平移 +1.5 cm（我们策略 0.312 vs 录制 0.297）→ 量化「平凡域差」的代价
"""

import glob
import importlib.util
import sys
import types

import numpy as np
import torch

# 只按文件路径加载需要的模块：`rl_lab/__init__.py` 与 `rl_lab/utils/__init__.py` 会 import
# isaaclab（导出配置模块），在没有 Isaac Sim 的机器上会 ModuleNotFoundError: omni.log。
# 这里给 rl_lab 系列建空壳包，再逐个加载 .py，绕开那些 __init__ 的副作用。
_RL = "imgo2_rl/scripts/rl_lab/rl_lab"
sys.path.insert(0, "imgo2_rl/scripts/rl_lab")
sys.path.insert(0, ".local-py")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_pkg = types.ModuleType("rl_lab")
_pkg.__path__ = [_RL]
sys.modules["rl_lab"] = _pkg
for _sub in ("utils", "datasets", "algorithms"):
    _m = types.ModuleType(f"rl_lab.{_sub}")
    _m.__path__ = [f"{_RL}/{_sub}"]
    sys.modules[f"rl_lab.{_sub}"] = _m
    setattr(_pkg, _sub, _m)
sys.modules["rl_lab.utils"].utils = _load("rl_lab.utils.utils", f"{_RL}/utils/utils.py")
sys.modules["rl_lab.datasets"].pose3d = _load("rl_lab.datasets.pose3d", f"{_RL}/datasets/pose3d.py")
sys.modules["rl_lab.datasets"].motion_util = _load("rl_lab.datasets.motion_util", f"{_RL}/datasets/motion_util.py")
AMPDiscriminator = _load("rl_lab.algorithms.amp_discriminator",
                         f"{_RL}/algorithms/amp_discriminator.py").AMPDiscriminator
AMPLoader = _load("rl_lab.datasets.motion_loader", f"{_RL}/datasets/motion_loader.py").AMPLoader

CKPT = ("imgo2_rl/logs/amp_rsl_rl/base_move_amp/2026-09-17_21-25-57/model_24500.pt")
MOTIONS = sorted(glob.glob("imgo2_rl/datasets/imgo2_motion/*.txt"))
BLOCKS = {"joint_pos": (0, 12), "foot_pos": (12, 24), "lin_vel": (24, 27),
          "ang_vel": (27, 30), "joint_vel": (30, 42), "root_z": (42, 43)}

torch.manual_seed(0)
np.random.seed(0)

# ---------------- 1. 专家数据 ----------------
loader = AMPLoader(device="cpu", time_between_frames=0.02, preload_transitions=True,
                   num_preload_transitions=400000, motion_files=MOTIONS)
gen = loader.feed_forward_generator(num_mini_batch=1, mini_batch_size=8192)
s, s_next = next(gen)
print(f"专家 batch: s{s.shape} s_next{s_next.shape}  obs_dim={loader.observation_dim}")

# ---------------- 2. 判别器 + 归一化 ----------------
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
dsd = ckpt["discriminator_state_dict"]
hidden = []
i = 0
while f"trunk.{i}.weight" in dsd:
    hidden.append(dsd[f"trunk.{i}.weight"].shape[0])
    i += 2
in_dim = dsd["trunk.0.weight"].shape[1]
print(f"discriminator: input_dim={in_dim} hidden={hidden}")
disc = AMPDiscriminator(in_dim, 2.0, hidden, "cpu", task_reward_lerp=0.3)
disc.load_state_dict(dsd)
disc.eval()

nrm = ckpt["amp_normalizer"]
mean = np.asarray(nrm["mean"], dtype=np.float32).reshape(1, -1)
var = np.asarray(nrm["var"], dtype=np.float32).reshape(1, -1)
eps = float(nrm["epsilon"])
clip = float(nrm["clip_obs"])
print(f"normalizer: mean[:3]={np.round(mean[0, :3], 3)} std[:3]={np.round(np.sqrt(var[0, :3] + eps), 3)} "
      f"count={nrm['count']} clip={clip}")


def norm(x):
    return torch.from_numpy(np.clip((x.numpy() - mean) / np.sqrt(var + eps), -clip, clip))


def d_of(a, b):
    """与 predict_amp_reward 同口径：先归一化，再拼 s||s_next 过判别器。"""
    with torch.no_grad():
        return disc(torch.cat([norm(a), norm(b)], dim=-1)).squeeze(-1).mean().item()


base = d_of(s, s_next)
print(f"\n=== 基线 ===")
print(f"真实专家 (s, s_next)                     d = {base:+.4f}")

print(f"\n=== 时序一致性对照（若 d 不掉，说明判别器没在用节律/时间结构）===")
perm = torch.randperm(s.shape[0])
print(f"打乱 s_next（跨样本拼接）                d = {d_of(s, s_next[perm]):+.4f}")
print(f"零时间差 (s, s)                          d = {d_of(s, s):+.4f}")
print(f"时间反序 (s_next, s)                     d = {d_of(s_next, s):+.4f}")

print(f"\n=== 逐块置零（在归一化空间置 0 = 取专家均值）===")
for name, (a, b) in BLOCKS.items():
    za, zb = s.clone(), s_next.clone()
    na, nb = norm(za), norm(zb)
    na[:, a:b] = 0.0
    nb[:, a:b] = 0.0
    with torch.no_grad():
        d = disc(torch.cat([na, nb], dim=-1)).squeeze(-1).mean().item()
    print(f"置零 {name:9s} ({a:2d}:{b:2d})            d = {d:+.4f}   (Δ {d - base:+.4f})")
with torch.no_grad():
    d = disc(torch.cat([norm(s) * 0, norm(s_next) * 0], dim=-1)).squeeze(-1).mean().item()
print(f"全部置零（常数输入）                     d = {d:+.4f}   (Δ {d - base:+.4f})")

print(f"\n=== 平凡域差：只平移根高（我们 0.312 m vs 录制 0.297 m）===")
for dz in (0.005, 0.015, 0.030):
    za, zb = s.clone(), s_next.clone()
    za[:, 42] += dz
    zb[:, 42] += dz
    print(f"root_z +{dz:.3f} m                          d = {d_of(za, zb):+.4f}   (Δ {d_of(za, zb) - base:+.4f})")

print(f"\n=== 归一化后的专家分布（看哪些维度的量纲被归一化放大了）===")
ns = norm(s)
std_after = ns.std(dim=0).numpy()
order = np.argsort(-std_after)
names = []
for k, (a, b) in BLOCKS.items():
    names += [f"{k}[{j}]" for j in range(b - a)]
print("  归一化后方差最大的 12 维:", ", ".join(f"{names[i]}={std_after[i]:.2f}" for i in order[:12]))
print("  归一化后方差最小的 6 维:", ", ".join(f"{names[i]}={std_after[i]:.2f}" for i in order[-6:]))


# ---------------- 3. 把专家动作「按倍速播放」，量判别器对节律的敏感度 ----------------
# 这是本探针最有信息量的一步：把同一份录制动作在时间轴上压缩 k 倍（姿态分布不变、
# 关节速度与相邻帧增量变大），正好模拟「同样的风格、不同的步频」。
# 如果 d 随 k 单调下降、且在 k≈3（我们实测的步频倍数）落到 −0.5 附近，
# 说明判别器**看得见步频**，问题只是它的动态范围太小（风格项竞争不过任务项）；
# 如果 d 对 k 不敏感，说明它根本没在看节律，加权重也不会改变步态。


def amp_obs(frames):
    """[B,49]（root_pos 0:3 / root_rot 3:7 / amp 7:49）→ [B,43]，与 loader 同口径。"""
    return torch.cat([frames[:, 7:49], frames[:, 2:3]], dim=-1)


traj = np.random.randint(0, len(loader.trajectories), 8192)
t0 = np.random.rand(8192) * (np.asarray(loader.trajectory_lens)[traj] - 1.0)
print("\n=== 倍速播放专家的判别器打分（k = 时间压缩倍数）===")
print(f"{'k':>6} {'d':>10} {'style_raw':>11} {'混合后风格项/步':>16}")
for k in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0):
    fa = loader.get_full_frame_at_time_batch(traj, t0)
    fb = loader.get_full_frame_at_time_batch(traj, t0 + 0.02 * k)
    d = d_of(amp_obs(fa), amp_obs(fb))
    style_raw = max(0.0, 1.0 - (d - 1.0) ** 2 / 4.0)
    print(f"{k:6.1f} {d:+10.4f} {style_raw:11.4f} {0.7 * 2.0 * style_raw:16.4f}")

print("\n参考：实测策略 d_policy≈-0.567 → style_raw≈0.386，混合后风格项≈0.540/步；"
      "任务项（lerp 后）≈1.402/步")

# ---------------- 4. 判别器对输入的**局部敏感度**（autograd）----------------
# d 对每个输入维度的平均 |梯度| × 该维在专家数据里的标准差 = 「这一维动 1σ 能让 d 动多少」。
# 用它判断判别器到底在看哪一块，以及要产生实测的 1.14 跨度需要在哪些维度偏离多少 σ。
xs = s.clone()
xn = s_next.clone()
xs.requires_grad_(True)
xn.requires_grad_(True)
_mean_t = torch.from_numpy(mean)
_std_t = torch.from_numpy(np.sqrt(var + eps))
_nt = lambda x: torch.clamp((x - _mean_t) / _std_t, -clip, clip)   # noqa: E731  可微版归一化
d_mean = disc(torch.cat([_nt(xs), _nt(xn)], dim=-1)).mean()
g = torch.autograd.grad(d_mean, [xs, xn])
imp = torch.cat([gi.abs().mean(dim=0) * torch.from_numpy(np.sqrt(var + eps)).squeeze(0)
                 for gi in g])          # 两段各 43 维，已乘上各自的 σ
print("\n=== 判别器局部敏感度：|∂d/∂x| × σ(x)（按块求和，s 段与 s_next 段各 43 维）===")
print(f"{'块':10s} {'s 段':>10s} {'s_next 段':>10s}")
for name, (a, b) in BLOCKS.items():
    print(f"{name:10s} {imp[a:b].sum():10.4f} {imp[43 + a:43 + b].sum():10.4f}")
print(f"{'合计':10s} {imp[:43].sum():10.4f} {imp[43:].sum():10.4f}")

# ---------------- 5. 平坦性直接检验 ----------------
# 「局部梯度≈0」是判别器退化的典型特征：它只记住了一个粗分区（专家区/策略区），
# 区域内处处平坦 —— 于是策略无法通过小改动把风格分推上去，风格项实际变成一个常数。
print("\n=== 平坦性直接检验（专家输入上叠加噪声 / ReLU 活跃率）===")
sig = torch.from_numpy(np.sqrt(var + eps)).squeeze(0)
for scale in (0.1, 0.5, 1.0, 2.0):
    n = torch.randn_like(s) * scale * sig
    print(f"  全维加 {scale:.1f}σ 噪声: d = {d_of(s + n, s_next + n):+.4f}  (基线 {base:+.4f})")
with torch.no_grad():
    full_in = torch.cat([norm(s), norm(s_next)], dim=-1)   # 归一化是逐 43 维做的，再拼成 86
    pre = disc.trunk[0](full_in)
    act = (pre > 0).float().mean().item()
    pre2 = disc.trunk[2](torch.relu(pre))
    act2 = (pre2 > 0).float().mean().item()
print(f"  第一层 ReLU 活跃率 {act:.3f} / 第二层 {act2:.3f}（专家输入）")
print(f"  trunk 输出范数 {disc.trunk(full_in).norm(dim=1).mean():.4f} / "
      f"amp_linear.bias={float(disc.amp_linear.bias):+.4f}（trunk 输出≈0 时 d≈bias，等于没在看输入）")

# ---------------- 6. 速度块放大：模拟「同样的步态、走得更快」----------------
print("\n=== 只放大速度块（姿态不变），看 d 对「速度超出专家分布」有多敏感 ===")
for blk, (a, b) in (("lin_vel", BLOCKS["lin_vel"]), ("ang_vel", BLOCKS["ang_vel"]),
                    ("joint_vel", BLOCKS["joint_vel"]), ("joint_pos", BLOCKS["joint_pos"])):
    row = []
    for f in (1.0, 1.5, 2.0, 3.0):
        za, zb = s.clone(), s_next.clone()
        za[:, a:b] *= f
        zb[:, a:b] *= f
        row.append(f"×{f:.1f}→{d_of(za, zb):+.3f}")
    print(f"  {blk:10s} " + "  ".join(row))
