# towing 冻结低层选定 PPO@2500（2026-10-18）

用户 2026-10-18 决定：**towing 基线测试的低层统一用 PPO**（不再混用 AMP），依据是
`2026-10-17_19-11-01_cart_v2_curriculum` 这轮训练「大结果还行」。本文记录选型依据、
可用窗口、使用约定与已知风险。

## 1. 选中的权重

| 项 | 值 |
|---|---|
| checkpoint | `logs/rsl_rl/base_move_rough/2026-10-17_19-11-01_cart_v2_curriculum/model_2500.pt` |
| 导出件 | 同目录 `exported/policy.pt`（`play.py --headless` 导出；`torch.jit.load` 可载，输出 `(1,12)`） |
| 体检 | 零观测 `\|a\|max = 1.864`（clip ±3，未饱和）、std 均值 **0.74**、actor 45→512→256→128→12、`empirical_normalization: false` |
| 训练时课程 | 速度上界 **1.5 m/s**、角速度 0.9 rad/s、地形均值 **6.0**（10 行网格，起点 5） |
| 训练指标（ITER 2500） | `track_lin_vel_xy_exp` ≈1.00（等价 RMS 跟速误差 ≈0.33 m/s）、`time_out` 99.9%（不摔）、`mean_reward` ≈1.7 |

## 2. 为什么不是更晚的 checkpoint

| iter | std 均值 | 零观测 \|a\|max | 判断 |
|---|---|---|---|
| 1000 | 0.61 | 1.27 | 可用（速度域只到 1.3） |
| 2000 | 0.72 | 1.16 | 可用 |
| **2500** | **0.74** | **1.86** | **选用** |
| 3000 | 0.81 | 2.78 | 可用（接近饱和边缘） |
| 3500 | 0.95 | 5.03 | 边缘，动作会被 clip |
| 4000 | 1.22 | 8.36 | 发散开始，关节基本全饱和 |
| 5000 | 2.41 | 23.9 | 不可用 |
| 6000 | 16.86 | 44.9 | **全倒**（用户实测「直接全部倒下」） |

发散机制：PPO 的 `std` **没有上界**，从 0.5 一路涨到 17；动作采样随之爆炸，策略均值也被带崩
（`track_lin_vel_xy_exp` 0.0016、地形课程被回退到 0）。训练停在 **iter 6034**（未跑满 10000）。

## 3. 使用约定（踩过的坑）

1. **`--policy-checkpoint` 只接受导出件**，不能传 `model_*.pt` 训练态 checkpoint —— 后者会报
   `RuntimeError: PytorchStreamReader failed locating file constants.pkl`（`FrozenLowLevelPolicy`
   用 `torch.jit.load`）。
2. **`exported/` 是「每个 run 目录一份」**：换 checkpoint 重新导出会**覆盖**同一份
   `policy.pt`。用不同 checkpoint 做对照时，必须先把导出件复制到独立路径，否则会静默用错权重
   （第一次用 model_6000 时就是这样：`exported/policy.pt` 是 6000 的，|a|=45）。
3. 导出写法：`play.py --task=Imgo2-basemove-rough-ppo --num_envs 16 --headless --checkpoint=<模型>`
   （导出发生在回放循环之前，`timeout 90` 兜住不退出的循环即可）。
4. 测试台一律写 `--policy ppo --policy-checkpoint <导出的 policy.pt>`；不要传 `model_*.pt`。

## 4. 待验证

- **towing 口径的实跑**：平坦（`--columns 0-19 --stop-progress 4.5`）先把 0.5/1.0 的跟速比测出来，
  与归档 PPO@4000 的 0.058/0.905、0.196/0.804 对照（那版 4000 已崩，只作历史参照）。
- **质量上限扫描**：见 [奖励权重/审计记录](towing_cart_audit_2026-10-17.md) 之后的质量扫描计划
  （阶段 1：列 0–39、masses 20/25/30/40/50/60；阶段 2：全网格、20/30/40/50）。判据取
  跟速比 <0.8（≈ `SPD`）与 `FALL`/`COL`/力矩饱和/张力。
- **风险**：`base_lin_vel` 观测仍为 `null`（策略看不到自身速度），是跟踪误差压不下去的结构性
  原因之一；要提速度或抬质量上限，应先修它 + 给 std 加上界 + 放宽 `track_lin_vel_xy_exp` 的
  std（0.5 → 0.8），再重训。
