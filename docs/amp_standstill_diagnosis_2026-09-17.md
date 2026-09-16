# AMP「站着不动」对照报告（2026-09-17）

对象：`imgo2_rl` 的 `Imgo2-basemove-flat-amp`（训练 9496 轮；基座高度 0.306 m、贴地 0.5%，
但线速度误差恒为 1.65 m/s、关节幅值 0.014 rad ≈ 姿态锁死；~1000 轮后指标全平）。
参考实现（本机都有）：

- `~/RL/isaac/AMP/AMP_for_hardware-main`（文件名与我们的 `amp_discriminator.py`/`amp_ppo.py` 一一对应，
  是移植来源）
- `~/RL/isaac/AMP/amp_go2-main`（legged_gym + rsl_rl 的 Go2 AMP，能走）

## 结论（先说答案）

**① `clamp(min=0)` 不是差异**：两份参考的公式与我们的**逐字相同**，都有这个地板。
真正差异在 **③④ 的任务奖励量级**：参考 a1 的 AMP 任务**只有速度跟踪两项、每步权重 50 / 16.7、
其余全 0**；我们等效只有 **1.0 / 0.3**（小 50 倍），却**额外留了一个 −10/步的 `base_height_l2`**。
于是速度误差一大（`exp(-(1.65/0.5)^2) ≈ 2e-5`）速度项梯度≈0，而高度项梯度把策略按在
「精确站在 0.30 m」这个局部最优上 → 锁死站姿 → 判别器对冻结策略给出近乎常数的风格分
（`mean_disc_pred≈-0.52`，风格≈0.467/步）→ PPO 优势≈0 → 全指标变平。与观测（高度达标、
关节不动、速度误差恒定、1000 轮后全平）完全一致。

## 逐条对照

### ① 风格奖励（`predict_amp_reward`）

- 参考两份**完全相同**，也**有** `min=0` 地板，`d` 是**裸 logit**（`amp_linear` 输出 1 维，无
  sigmoid/tanh）：
  `reward = amp_reward_coef * torch.clamp(1 - (1/4) * torch.square(d - 1), min=0)`
- 我们与之逐字相同 ⇒ **d≤-1 时同样没有梯度，但这不是差异**。

### ② 判别器 loss

- 两份参考相同：`expert_loss = MSE(expert_d, +1)`、`policy_loss = MSE(policy_d, -1)`、
  `amp_loss = 0.5*(expert+policy)`、`grad_pen = compute_grad_pen(*expert, lambda_=10)`（**只对 expert**）、
  `loss = surrogate + value_coef*value − entropy_coef*entropy + amp_loss + grad_pen`；
  **与策略共用一个 Adam**（`lr=1e-3`，trunk `weight_decay=10e-4`、head `10e-2`，KL 自适应缩放 lr），
  另有 `std.data.clamp(min=min_std)`。
- 我们相同 ⇒ 也不是差异。

### ③ 风格:任务 配比

| | amp_reward_coef | amp_task_reward_lerp | 任务项（每步权重） |
|---|---|---|---|
| 我们 | 2.0 | 0.3 | track_lin_vel 等效 **1.0**、track_ang_vel 等效 **0.3**、**base_height_l2 −10** |
| `AMP_for_hardware` a1 | 2.0 | 0.3 | tracking_lin_vel **50**、tracking_ang_vel **16.67**、**其余全 0**（base_height 0） |
| `amp_go2`（能走） | **0.2** | **0.8** | tracking_lin_vel 4.0、tracking_ang_vel 2.0 + lin_vel_z −1、ang_vel_xy −0.05、<br>dof_acc −2.5e-7、torques −1e-4、base_height −1、action_rate −0.01、collision −1、<br>dof_pos_limits −2、feet_air_time 1.0 |

### ④ 速度/角速度奖励

- 形式（两份参考与我们都一样）：`exp(-error²/tracking_sigma)`（或等价的 `std=sqrt(0.25)`），
  `tracking_sigma = 0.25` ⇒ σ=0.5；角速度同式。
- **权重差异是关键**：参考 a1 写 `1.5 / (0.005*6) = 50`、`0.5 / (0.005*6) ≈ 16.7`，而
  legged_gym 的 `compute_reward` **不再乘 dt**（`self.rew_buf += rew`，见
  `legged_gym/envs/base/legged_robot.py:245-256`）⇒ 参考的**每步**任务奖励是 `50*exp`。
  我们写 `1.0/step_dt`、`0.3/step_dt`，但 Isaac Lab 的 RewardManager **会乘 step_dt**
  （`sim.dt*decimation=0.02`）⇒ 约掉后每步只有 `1.0*exp`、`0.3*exp`，**比参考小 50 倍**。
  这也解释了实测的「风格 0.467 / 任务 0.078 ≈ 6:1」，而参考在同一误差下任务项约 0.117×50=13
  （⇒ 任务主导）。

### ⑤ 探索与初始化

- `init_noise_std = 1.0`、`min_normalized_std = [0.05, 0.02, 0.05] * 4`、`entropy_coef = 0.01`、
  `learning_rate = 1e-3`、学 `log_std`（rsl_rl 的 `std` + `clamp(min=min_std)`）——**与 a1 逐字相同**
  ⇒ 不是差异。

### 其他（都一致或无害）

- 判别器结构 `[1024, 512]`（ReLU）、`amp_num_preload_transitions=2000000`、
  `amp_replay_buffer_size=1000000`：与 a1 相同。
- **AMP 观测组成相同**：joint_pos + foot_pos_base + root_lin_vel_b + root_ang_vel_b + joint_vel + root_z
  （参考：joint_pos、foot_pos、base_lin_vel、base_ang_vel、joint_vel、z_pos；**都不含 commands**）。
- 复位时 commands 是随机采样（参考 `legged_robot.py:409-417`，并把 |v|<0.2 置零）——两边一致。
- 终止：我们保留 `illegal_contact`（base 触地终止）；参考的 `termination` 奖励权重为 0。
- 指令范围：我们 x∈(−1,1.5)、y∈(−1,1)、yaw ±1.57，且 `command_levels_* = None`
  （**无课程**，一开始就满量程）；参考 a1 `curriculum=False`，范围默认 ±1。

## 建议的最小改动（按优先级）

1. `amp_env_cfg.py`：把速度跟踪权重提到与参考同量级——`track_lin_vel_xy_exp.weight = 50/step_dt`、
   `track_ang_vel_z_exp.weight = 17/step_dt`（即比现在大 50 倍）。
2. **把 `base_height_l2` 从 AMP 任务里去掉**（移出 `keep_rewards` 或权重设 0）：a1 的 AMP 没有任何
   高度项，留着 −10/步会把策略按在「精确站 0.30 m」的局部最优上。
3. 想更稳可同时照 `amp_go2`（能走的那份）改配比：`amp_reward_coef=0.2`、`amp_task_reward_lerp=0.8`。
4. 加指令课程（例如先 |v|≤0.5 再放开到 1.5）：σ=0.5 的指数核在 1.5 m/s 命令下几乎恒为 0，
   一开始就给满量程不利于起步。
5. ⑤ 探索保持现状（与参考一致）。

改完建议用同一脚本/指标复查：基座高度、速度误差、关节幅值是否随训练变化（而不是 1000 轮后变平）。

## 8. 参考侧可核实的回答（2026-09-18 补，回应对方 §5 的五个问题）

在 `~/RL/isaac/AMP/AMP_for_hardware-main` 与 `~/RL/isaac/AMP/amp_go2-main` 里逐条核对：

| 对方的问题 | 参考侧的实际情况 | 含义 |
|---|---|---|
| Q1 参考 a1 真能跟踪速度吗？ | 仓库里**没有训练曲线**：`logs/a1_amp_example/Apr25_18-17-33_/` 只有 `model_36450.pt`（训了 **36450 轮**，我们 9496）与 `exported/policies/policy_1.pt`，**没有 events 文件** | 拿不到曲线证据。可替代证据：a1 的任务项**只有速度跟踪**（每步 50 / 16.7）、**其余全 0**，任务定义本身就是"纯速度跟踪"；且是有 paper/仓库背书的 a1 AMP 示例。若要硬证据，需在训练机用其 play 跑 `model_36450.pt` 量速度（本机无 GPU/IsaacGym） |
| Q2 参考的指令范围是否更小？ | a1：`lin_vel_x = [-1.0, 2.0]`、`lin_vel_y = [-0.3, 0.3]`、`ang_vel_yaw = ±1.57`；go2：`x = [-1.2, 1.5]`、`y = ±0.8` | **不比我们窄**（a1 的 x 上限还更大）⇒「范围小掩盖了 exp 核的远场无梯度」不成立 |
| Q3 参考是否靠探索噪声点火 / 有没有课程？ | **两份参考的 `commands.curriculum` 都是 `False`**（没有速度课程）；`init_noise_std = 1.0`（与我们相同）；`min_normalized_std`：a1 = `[0.05, 0.02, 0.05]*4`（**与我们逐字相同**）、go2 = `[0.01, 0.01, 0.01]*4`（**比我们还小**） | 探索超参**不是差异**；**课程不是参考的做法**，是我们要新加的（可以作为 D 组，但别写成"对齐参考"） |
| Q4 参考的 base_height 真为 0？ | a1：`base_height = 0.0`（`base_height_target = 0.25` 未被使用）；go2：`base_height = -1.0`（target 0.38） | a1（我们的移植来源）**完全没有高度项** ✓；go2 有但只有 **−1**，我们是 **−10**（差 10 倍） |
| Q5 A 还是 B 是主因？ | A 是唯一被实测到的**结构差异**（任务项 1.0 vs 50，且我们多一个 −10/步高度项）；B（exp 核远场无梯度）是**双方共有**的性质（a1 同核同 σ，命令上限 2.0） | B 只能当"放大器"，解释不了"为什么参考动、我们不动"；A 才是可归因差异 ⇒ **A 组必须排第一，且必须包含"去掉 height 项"**，而不是只提权重 |

### 8.1 另外两处核对结论

- **我们的 `amp_normalizer` 更新与参考有忠实性偏差**（低优先）：参考在
  `amp_ppo.py:224-227` 已把 `policy_state/expert_state` 覆盖成**归一化后**的张量，随后 254-256 行
  用它们 `update()`；我们的 `rl_lab/.../amp_ppo.py:258-260` 用的是 `raw_policy_state/raw_expert_state`
  （**原始**统计）。两者最终都会收敛到 ~N(0,1)，差别在瞬态；若 A/D 都无效，值得对齐后再试一次。
- **`amp_go2` 的奖励是"任务主导"的另一套**：`coef = 0.2`、`lerp = 0.8`，且任务项是完整的
  （tracking 4.0/2.0 + lin_vel_z −1、ang_vel_xy −0.05、dof_acc −2.5e-7、torques −1e-4、
  base_height −1、action_rate −0.01、collision −1、dof_pos_limits −2、feet_air_time 1.0）。
  a1 则相反（风格主导 2.0/0.3、任务只有两项）。**两条路线都能走** ⇒ 说明关键不是"风格/任务谁主导"，
  而是**任务项必须真有量级、且不能被高度项之类的强整形项抢走梯度**。

### 8.2 对实验流程的建议（回应对方 §6）

1. **先验证权重真值**：改完后在环境构造处以代码打印 `weight * step_dt`（或 reward_manager 的最终
   权重），确认"每步 50 / 16.7"真的生效。本次根因就是单位语义（legged_gym 不乘 dt、Isaac Lab 乘
   `step_dt`），这个检查只要 1 分钟，应放在所有实验之前。
2. **A 组的最小正确版本**：`track_lin_vel_xy_exp.weight = 50/step_dt`、
   `track_ang_vel_z_exp.weight = 17/step_dt`，并**把 `base_height_l2` 从 `keep_rewards` 里移除**
   （现在 `keep_rewards` 有三个键，去掉后只留两个）；若想保留高度约束，参照 go2 用 **−1** 而不是 −10。
3. **判据补两条**：除 `error_vel_xy` 外记录「风格奖励:任务奖励 的每步均值比」（A 生效时应从 6:1 翻到
   约 1:5 量级）与「关节活动幅度」（是否 >0.1 rad）。只看 `error_vel_xy` 分不清"高度崩了"和"真的在学走"。
4. **轮数与种子**：500 轮筛选可行（死锁 ~1000 轮出现，500 轮能看到趋势），但决赛组用
   1500–2000 轮 × ≥2 个种子（AMP 方差大）。
5. **顺序建议（与对方倾向相反）**：A（含去掉 height）→ D（课程，绕开远场无梯度）→ C（抬高探索下限）
   → B（换核，最后；因为参考同核能走，改核不是"对齐"而是"绕路"）。
6. **若 A/D 都无效**，下一个嫌疑依次是：① `amp_normalizer` 的 update 口径（见 8.1）；② 参考动作数据
   的**速度场**（`imgo2_motion` 的 root lin vel / joint vel 的坐标系与尺度是否与仿真一致——已知
   运动学 FK 对齐，但速度场未见核对记录）；③ 判别器输入里 `joint_pos`/`foot_pos_base` 是否需要
   `default_dof_pos` 相对化（参考用绝对量，我们也是绝对量，此项一致）。
