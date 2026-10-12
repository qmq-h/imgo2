# AMP 策略「站着不动」排查清单（发给有参考项目的机器执行）

**目的**：定位我们 Imgo2 的 AMP 策略为什么只站不走，而参考项目的 AMP 能走。
**方法**：在参考项目里读 5 处代码，填下面的对照表，把结果发回。

---

## 0. 背景（我方现状，供对照）

- 代码：`imgo2_rl/`（Isaac Lab 移植版），任务 `Imgo2-basemove-flat-amp`，参考数据 21 份 `imgo2_motion`。
- 现象：训练 9496 轮，基座高度 0.306 m（达标）、贴地比例 0.5%、**但线速度误差恒为 1.65 m/s**（指令范围 x∈(-1,1.5)、y∈(-1,1)），关节幅值仅 0.014 rad ≈ 姿态锁死。
- 训练日志特征：**第 ~1000 轮后所有指标完全平坦**；`mean_disc_pred ≈ -0.52`（判别器对策略数据打负分）；`mean_noise_std` 0.13（接近下限 0.05）。
- 已排除：缺 `lin_vel` 观测、传感器装错、指令未下发、地形限制误伤、模型/物理错误（参考策略在同一模型上能走 0.454 m/s）。
- **首要怀疑**：风格奖励里的 `clamp(min=0)` 在 `d ≤ -1` 时把奖励和梯度一起截成 0，形成零梯度死锁。

---

## 1. 必查的 5 处（按优先级）

### ① 风格奖励公式 —— 最关键

我方实现（`scripts/rl_lab/rl_lab/algorithms/amp_discriminator.py`）：

```python
d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
reward = self.amp_reward_coef * torch.clamp(1 - (1/4) * torch.square(d - 1), min=0)
```

**要找**：参考项目里从判别器输出算 reward 的函数。

```bash
cd <参考项目根>
grep -rn "amp_reward_coef\|reward_coef" --include=*.py . | head
grep -rn "1 - (1/4)\|0.25\|clamp\|square" --include=*.py . | grep -i "reward\|disc" | head
```

**要回答**：
1. 公式是否也是 `coef * clamp(1 - (d-1)²/4, min=0)`？**有没有 `min=0` 这个地板？**
2. 若有地板，`d` 的取值区间是多少（`d` 是裸 logit 还是有 sigmoid/tanh）？
3. 若**没有**地板，替代形式是什么（`-log(1-d)`、`softplus(-d)`、`coef*(1-d)`…）？残留梯度在 `d≪-1` 时是否仍非零？

---

### ② 判别器损失形式

我方（`scripts/rl_lab/rl_lab/algorithms/amp_ppo.py`）：

```python
expert_loss = MSELoss(expert_d, +1)      # 真实参考动作 → +1
policy_loss = MSELoss(policy_d, -1)      # 策略动作     → -1
amp_loss = 0.5 * (expert_loss + policy_loss)
+ grad_pen = 10 * ||∇_expert d||²
```

**要找**：参考项目判别器的 loss。

```bash
grep -rn "expert_loss\|policy_loss\|amp_loss\|grad_pen\|least_squares\|softplus\|binary_cross" --include=*.py . | head -20
```

**要回答**：
1. 也是 MSE 到 ±1（最小二乘 GAN），还是 BCE / softplus / WGAN 形式？
2. 梯度惩罚系数与作用对象（只对 expert，还是 expert+policy）？
3. 判别器与策略**是否共用同一个 optimizer / 同一个学习率**？（我方是共用一个 Adam，`lr=1e-3`，trunk `weight_decay=1e-3`、head `1e-1`）

---

### ③ 风格与任务的权重配比

我方（`agents/amp_rsl_rl_cfg.py`）：

```python
amp_reward_coef     = 2.0
amp_task_reward_lerp = 0.3      # r = 0.7*风格 + 0.3*任务
```
实测每步：风格贡献 ≈0.467、任务贡献 ≈0.078 → **风格:任务 ≈ 6:1**。

**要找**：

```bash
grep -rn "task_reward_lerp\|style_weight\|task_weight\|amp_reward_coef\|lerp" --include=*.py . | head -20
```

**要回答**：风格与任务各占多少？任务项内部（速度跟踪 / 高度 / 其他）各自的权重是多少？

---

### ④ 速度跟踪奖励的核宽与权重

我方：`track_lin_vel_xy_exp = exp(-error²/0.25)`，`std=√0.25=0.5`，每步权重 1.0（配置里写 `1.0/step_dt`）。
→ 误差 1.65 m/s 时奖励 `exp(-10.9) ≈ 1e-15`，**几乎零梯度**。

**要找**：

```bash
grep -rn "track_lin_vel\|lin_vel_xy\|tracking_lin_vel" --include=*.py . | head -10
```

**要回答**：奖励形式（指数核 / 平方误差 / 别的）、`std`（或 `sigma`）取值、权重。同样看 `track_ang_vel_z`。

---

### ⑤ 探索与初始化

我方：`init_noise_std=1.0`、`min_normalized_std=[0.05, 0.02, 0.05]*4`、参考状态初始化概率 1.0。
实测 `mean_noise_std` 从 0.32 快速掉到 0.13。

**要找**：

```bash
grep -rn "init_noise_std\|min_normalized_std\|fixed_std\|log_std\|noise_std" --include=*.py . | head -15
```

**要回答**：
1. 初始 std 与下限分别是多少？是学习 `log_std` 还是 `std`？
2. 有没有额外的探索机制（熵系数、动作噪声）？熵系数多少？

---

## 2. 顺手一起看（次要但有用）

```bash
# 判别器网络结构与输入维度
grep -rn "hidden_layer_sizes\|hidden_dims\|discriminator" --include=*.py . | grep -i "1024\|512\|256\|Linear" | head

# AMP 观察由哪些量组成（是否含 root 高度、线速度）
grep -rn "amp_obs\|amp_observation\|AMPObs" --include=*.py . | head -15

# termination 条件（有无基座触地终止）
grep -rn "termination\|illegal_contact\|base_contact" --include=*.py . | head -10

# 速度指令范围 / 站立比例
grep -rn "lin_vel_x\|rel_standing_envs\|command" --include=*.py . | grep -i "range\|standing" | head
```

---

## 3. 回报格式（照这个填，发回即可）

```text
参考项目路径：
代码版本（git log -1 或日期）：

① 风格奖励公式：
   原文一行：
   有/无 clamp(min=0)：
   d 的取值范围：

② 判别器 loss：
   形式（MSE±1 / BCE / softplus / 其他）：
   梯度惩罚系数与对象：
   是否与策略共用 optimizer 与 lr：

③ 风格:任务 配比：
   task_reward_lerp 或等效：
   任务项各自权重：

④ 速度奖励：
   形式与 std：
   权重：
   角速度同样：

⑤ 探索：
   init std / 下限 / 学 log_std 还是 std：
   熵系数：

其他补充（观察组成、termination、指令范围）：
```

---

## 4. 我们最想先确认的一条

**① 里「`d ≤ -1` 时风格奖励是否还有非零梯度」。**

- 若参考项目**没有 floor**（或用的是残留梯度的形式），则基本可以确定：我方移植时引入的 `clamp(min=0)` 是零梯度死锁的根源，改动方向就是换掉这个 clamp / 改判别器 loss 形式；
- 若参考项目**也有**同样的 clamp，那问题在别处（配比、探索、判别器训练节奏），需要按 ③④⑤ 继续对比。

**最小回报**：其实只要 ① 和 ② 两处的**原文代码**，就能判断大半。
