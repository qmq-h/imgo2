# AMP 步态怎么调：先证明「风格项还有没有梯度」（2026-09-18）

问题：24500 轮策略的**步频是录制参考的 3.1 倍**（[足端回放记录](gait_eval_amp_24500_2026-09-18.md)）。
用户提问：**是不是风格占比不够？**

本记录用四份证据回答，全部可在无 GPU 的机器上复现：
① 训练日志的回报分解；② 判别器探针（离线，专家侧）；
③ **策略侧 AMP 观测的块级归因**（用 `eval_gait.py --dump-npz` 产出的数据）；
④ 合成对照实验：**梯度惩罚 λ_gp 对判别器分辨能力的影响**。

---

## 1. 结论先说

1. **不是占比不够，是风格项在策略工作点上没有梯度。**
   终局每步混合回报 **1.97** = 任务 **1.402（71%）** + 风格 **0.540（29%）**；风格项按判别器
   自己的标定可达 **1.335**，所以「理论上有 0.8/步可以争」。但实测判别器在**策略点**的局部梯度
   `|∂d/∂x|·σ` 合计只有 **8.0e-5**（1σ 全维扰动只能移动 d 约 **5e-4**）——风格奖励是个**台阶**：
   它说"你不在录制流形上"，却不给方向。此时加大 `lerp`/`coef` 只是把一个没有梯度的常数放大，
   同时把 1.4/步的任务项压低。
2. **判别器看得见缺陷，只是给不出梯度。** 把策略观测的三块「关节角 + 足端 + 关节速度」换成
   专家的，能补回 **86%** 的风格差距；其中**关节速度块单独就占 53%**：策略 `joint_vel` 的标准差
   是专家的 **1.7 倍**（1.113 vs 0.650）——这正是 3 倍步频的印记。
   而 `joint_pos`/`foot_pos` 的边缘分布几乎与专家一致（均值偏移 0.08σ/0.11σ，σ 比 1.00）
   ⇒ **逐帧姿态是对的，差的是时间尺度**。与「干净对角步、关节幅度达标」的足端结论完全自洽。
3. **「指令超出数据覆盖」不是原因**：在 0.6 m/s（数据里就有 forward_0.6 这一档）步频仍是
   **5.03 Hz = 3.0 倍**。压指令范围解决不了节律问题（仍值得做，但属于部署包线治理）。
4. **λ_gp=10 是让风格项失去梯度的头号嫌疑，且已用合成对照证实**：同样训练预算下，
   λ_gp 10 → 1 → 0 时，判别器对「正常节律 vs 3 倍速节律」的**风格奖励差**从 **0.105/步**
   升到 **0.564/步**、**1.262/步**（5.4× / 12×）。见 §4。

**因此调整方向是**：先修判别器（λ_gp 等），或直接加一个针对节律的项（`feet_air_time`），
**而不是**单纯加大风格权重。

---

## 2. 证据一：训练日志的回报分解

来源（`logs/amp_rsl_rl/base_move_amp/2026-09-17_21-25-57/events.*`）：
`Train/mean_amp_reward_step`（混合后总回报/步）、`Train/weighted_task_reward_step`（= lerp×任务）、
`AMP/disc_{expert,policy}_pred`（判别器裸打分）。

| 迭代 | d_expert | d_policy | 总回报/步 | 任务(lerp后) | 风格 | 风格占比 | 速度误差 |
|---|---|---|---|---|---|---|---|
| 0 | +0.194 | −0.423 | 1.240 | 0.184 | 1.056 | 85% | 0.023 |
| 1000 | +0.598 | −0.599 | 1.303 | 0.808 | 0.495 | 39% | 1.106 |
| 2000 | +0.626 | −0.626 | 1.529 | 1.067 | 0.462 | 31% | 0.795 |
| 5000 | +0.657 | −0.655 | 1.695 | 1.257 | 0.438 | 26% | 0.596 |
| 12000 | +0.550 | −0.549 | 1.920 | 1.372 | 0.548 | 29% | 0.507 |
| 24500 | +0.571 | −0.567 | 1.931 | 1.402 | 0.540 | 28% | 0.468 |

**任务项一路单调涨（0.18→1.40），风格项在 5000 轮后就在 0.44–0.56 打转**；
判别器两边始终停在 ±0.55～0.66，既不朝 ±1 饱和、也不随策略变化移动 ⇒
「任务项有梯度、风格项近似常数」。

公式（`amp_discriminator.py:65,71-72`）：`disc_r = coef × clamp(1 − (d−1)²/4, min=0)`，
`混合 = (1−lerp)·disc_r + lerp·task`，本项目 `coef=2.0`、`lerp=0.3`。

---

## 3. 证据二：判别器的块级归因（专家侧 + 策略侧）

脚本：[`probe_amp_discriminator.py`](../imgo2_rl/scripts/tools/probe_amp_discriminator.py)（专家侧）、
[`probe_amp_policy_vs_expert.py`](../imgo2_rl/scripts/tools/probe_amp_policy_vs_expert.py)（策略侧，
需 `eval_gait.py --dump-npz` 的产物）。

### 3.1 专家侧：判别器对什么敏感

| 对照（专家输入） | d | Δ |
|---|---|---|
| 真实专家对 | +0.571 | — |
| **全部输入置零** | +0.527 | −0.044 |
| 打乱 s_next / 零时间差 / 时间反序 | +0.299 / +0.562 / +0.521 | −0.27 / −0.01 / −0.05 |
| **3 倍速播放专家动作**（= 我们的步频倍数） | **+0.504** | **−0.067** |
| 全维加 0.1σ / 0.5σ / 1σ / 2σ 噪声 | +0.537 / +0.275 / −0.095 / −0.771 | −0.03 / −0.30 / −0.67 / −1.34 |
| 只放大 lin_vel ×3 / joint_vel ×3 | +0.466 / +0.510 | −0.11 / −0.06 |

⇒ 单看专家邻域，判别器更像「离录制流形有多远」的探测器；**在专家点上单独改节律只值 0.013/步**。
但把它放到策略的真实分布上，结论更细致，见 3.2。

### 3.2 策略侧：风格差距的块级归因（vx=1.0，8 环境 × 950 步）

`d(专家)=+0.572`，`d(策略)=−0.145`（固定 1.0 m/s 指令下；训练日志里的 −0.57 是在
**混合指令分布**下测的）。

**把策略的某一块换成专家的**（能救回多少）：

| 换入专家的块 | d | Δ |
|---|---|---|
| **joint_vel** | **+0.157** | **+0.302（占风格差距 53%）** |
| joint_pos | −0.014 | +0.131 |
| foot_pos | −0.053 | +0.092 |
| lin_vel | −0.095 | +0.050 |
| ang_vel | −0.109 | +0.036 |
| root_z | −0.145 | 0.000 |
| joint_pos+foot_pos | +0.103 | +0.248 |
| **关节角+足端+关节速度（"步态三块"）** | **+0.413** | **+0.558（86%）** |
| 上述 + lin_vel | +0.475 | 92% |
| 上述 + ang_vel | +0.552 | 99% |
| 全部（=专家） | +0.571 | 100% |

**归一化空间里的分布对比**：

| 块 | 专家均值 | 策略均值 | 专家 σ | 策略 σ | 均值偏移 |
|---|---|---|---|---|---|
| joint_pos | 0.000 | 0.078 | 0.923 | 0.918 | 0.08σ |
| foot_pos | −0.021 | −0.128 | 0.884 | 0.877 | 0.11σ |
| **joint_vel** | 0.022 | −0.056 | **0.650** | **1.113** | 0.08σ |
| lin_vel | −0.067 | 0.577 | 0.623 | 0.685 | **0.64σ** |
| ang_vel | −0.001 | 0.004 | 0.364 | 0.476 | 0.01σ |
| root_z | −0.115 | 0.416 | 0.425 | 0.200 | **0.53σ** |

**读法**：姿态与足端的**边缘分布**几乎重合（σ 比 1.00），唯一明显异常的是
**关节速度的标准差 1.7 倍** ⇒ 判别器确实抓到了「时间尺度」这个缺陷；
`lin_vel`/`root_z` 的均值偏移是**指令驱动的**（我们以 1.0 m/s 走、专家平均约 0.5 m/s、
站高 0.312 vs 0.297），它们合计只占风格差距的 14%。

### 3.3 关键：局部梯度

| 观测点 | `|∂d/∂x|·σ` 合计（s 段） | 1σ 全维扰动可移动 d |
|---|---|---|
| 专家点 | 6.5e-05 | ~4e-04 |
| **策略点** | **8.1e-05** | **~5e-04** |

⇒ **在策略自己的工作点上，风格奖励几乎是平的**：策略哪怕整体往专家方向移动 1σ，d 只动 5e-4
（对应风格奖励 <0.001/步）。这与训练日志「20000 轮里 d_policy 只动了 0.07」完全吻合。

---

## 4. 证据三：λ_gp 的合成对照实验（离线，CPU）

脚本：[`probe_gp_effect.py`](../imgo2_rl/scripts/tools/probe_gp_effect.py)。
构造一个**只差节律**的判别问题：正类 = 录制动作正常速度的相邻帧对 `(s_t, s_{t+1})`，
负类 = 同一份动作 **3 倍速**的帧对 `(s_t, s_{t+3})`；网络与损失完全照抄本项目
（`[1024,512]+ReLU`、`MSE(±1)`、`grad_pen` 只作用于正类），只改 `λ_gp`：

| λ_gp | d(正类) | d(负类) | d 跨度 | 风格奖励差/步 |
|---|---|---|---|---|
| **10（本项目现用）** | +0.072 | −0.077 | **0.149** | **0.105** |
| 1 | +0.413 | −0.399 | 0.812 | **0.564**（5.4×） |
| 0 | +0.898 | −0.901 | 1.799 | **1.262**（12×） |

**结论**：λ_gp=10 会把判别器的输出幅度整体压扁（d 跨度只有 0.15），于是「走得对」与
「走快 3 倍」在风格奖励上只差 0.105/步；λ_gp=1 时同样的区分值 0.564/步。
这正是 `compute_grad_pen` 的设计意图（让 d 在专家数据上平坦）在**小数据集**上的副作用。

> **边界（不夸大）**：① 这是一个**合成**的两类问题（只差节律、没有域差），绝对数值不能直接
> 搬到真实训练；② 这组对照的预算是 1500 步，比真实训练（24000 次迭代 × 每次多个 epoch）小得多，
> 现网判别器在真实训练里达到了 ±0.57 的跨度，说明长预算下 λ=10 也能学到东西。
> 因此上表能支持的结论是**趋势与量级**（同样预算下 λ 越小、判别器对节律的响应幅度越大），
> 而不是「λ=10 一定学不会」。要做成配置改动，建议先跑 6000 步的同预算横向对照
> （脚本已支持 `--steps 6000 --lams 10 1`），再决定 λ 取值。

---

## 5. 怎么调（新的优先级）

| # | 动作 | 成本 | 为什么 / 判据 |
|---|---|---|---|
| **P1** | **降 `λ_gp`（10 → 1~2）或改其作用方式**，让判别器恢复对节律的响应幅度 | 一次训练 | 唯一被实验直接指向的风格侧原因。判据：`disc_expert/policy` 重新拉开到 ±1 附近、**风格项随训练单调上升**（现在 5000 轮后就是常数），同时 `Loss/AMP` 不能塌 |
| **P2** | **加一个针对节律的项：`feet_air_time`**（legged_gym 里治「高频小步」的标准手段）。当前滞空 **0.067–0.085 s**；参考周期 0.600 s、占空比 ~0.6 ⇒ 目标滞空 ≈0.24 s；**用户 2026-09-18 定为 0.3 s**（见 §9） | 一次训练 | **直接命中唯一缺陷**（步频 3.1 倍），且**不依赖判别器健康度**。预期：步频下降、步幅上升、速度误差略升；判据：步周期 ≥0.45 s 且线速度误差 ≤0.6 m/s |
| **P3** | 可选兜底：单变量 `amp_task_reward_lerp` 0.3 → 0.5 | 一次训练 | 若 P1/P2 都不理想，用它确认「风格项到底有没有用」；只抬权重不动判别器时，预期是「整体变保守、步频不变」——那本身就是一个干净的反证 |
| **P4** | 恒定抬头 +2.1°（1.0 m/s）/ +2.9°（0.6 m/s）：把现在 `weight=0.0`、被过滤掉的 `flat_orientation_l2` 打开做单变量消融 | 一次训练 | 与高度项耦合，注意别把「站得住」改坏 |
| ~~P5~~ | ~~压指令范围到数据覆盖内~~ | — | **降级**：0.6 m/s 在覆盖内、步频仍是 3.0 倍 ⇒ 不是原因。仍值得为部署包线做，但不解决节律 |

**不建议**：只加大 `amp_reward_coef`/`lerp` 而不动判别器（第 1、3 节）。

---

## 6. 缺什么才能完成

- **P1 定档**：6000 步同预算的 λ_gp 对照（`--steps 6000 --lams 10 1`，本机 CPU 可跑，
  约 10 分钟/组）⇒ 给 λ 取值一个直接的横向比较。
- **P1–P4 落地**：各自一次完整训练（4096 环境、10000 轮 ≈ 4 h ≈ 6.5 元；用户历史口径 12 h ≈ 20 元）。
  **一次只改一个变量**，判据统一为：步周期、步幅、滞空时间、线速度误差 +
  `AMP/disc_{expert,policy}_pred` + 风格项/步。
- **未做**：真机、GUI、MuJoCo 验证；本记录只动评估/诊断工具与文档，**没有改任何训练配置**。
  按工作区约定，P1/P2 目前都还是「有证据支持的建议」，不是已完成的修复。

---

## 7. 与两份参考项目逐项对照（2026-09-18 补，**更正 §4/§5 里对 λ_gp 的过度归因**）

两份参考都在本机 `~/Desktop/AMP/`：`AMP_for_hardware-main`（a1，**纯 AMP 配方**）与
`amp_go2-main`（go2，**任务主导配方**）。逐项对照后发现：

| | a1（AMP-only） | go2（任务主导） | **我们** |
|---|---|---|---|
| `lambda_`（梯度惩罚） | **10** | **10** | **10 ⇒ 与参考一致，不是配方差异** |
| 网络 / 优化器 | `[1024,512]`、Adam lr 1e-3、trunk wd 1e-3、head wd 1e-2 | 同 | **完全相同** |
| AMP 观测 | 43 维（同一布局：关节角/足端/线速度/角速度/关节速度/根高） | 43 维 | 43 维，相同 |
| `amp_reward_coef` / `lerp` | 2.0 / 0.3 | 0.2 / 0.8 | 2.0 / 0.3（= a1） |
| 判别器批大小/步数 | `num_learning_epochs 5 × num_mini_batches 4` | 同 | 相同 |
| **每步任务奖励**（`dt=0.02`，legged_gym 会再乘 dt） | 速度 1.0–1.5 + 角 0.33–0.5，**其余全 0** ⇒ lerp 后 ≈**0.3–0.45** | 速度 0.08 + 角 0.04 + 一组步态项 ⇒ lerp 后 ≈**0.12** | **4.678 ⇒ lerp 后 1.402** |
| 风格:任务（实测/上限） | ≈**70:30**（风格占优） | ≈25:75 | **28:72** |
| **步态项** | 全 0 | **有**：`feet_air_time 1.0`、`action_rate −0.01`、`dof_acc`、`torques`、`collision` | **全 0**（被 `_keep_only_amp_task_rewards()` 删掉，只留 6 项） |

**由此更正两条：**

1. **λ_gp=10 不是差异**：两份参考都用 10，且在这个 λ 下都能走出干净步态 ⇒
   §4/§5 把「降 λ_gp」排在 P1 属于**过度归因**（§4 的合成对照只说明「λ 越小判别器响应幅度越大」
   这个单调趋势，不能证明参考配方有问题）。降 λ 可以作为一个**有意的偏离**去试，但要按实验对待。
2. **真正与参考不一致的是任务侧**：我们等于「像 a1 一样纯靠 AMP 管步态」**加上**「任务权重按每步
   4.0/2.0 写」（a1 等效每步 1.0–1.5、go2 更小），于是风格只占 28%（a1 约 70%），
   **同时一个步态塑造项都没有**（go2 有一整套）。这是一个两边都不像的第三种配置。

**因此方案收敛为两条（二选一，别混）：**

- **方案 A（抄 a1，回应「调比例」）**：把任务项降到 a1 量级（每步 1.5/0.5 左右，甚至考虑去掉高度项），
  让 AMP 真正主导。**风险已知且明确**：我们自己的 Run1（每步 1.0/0.3 + 高度 −10）就是
  「只站不走」——纯降任务项有复现该退化解的高风险。
- **方案 B（抄 go2，推荐先做）**：保留现有任务尺度，**把 legged_gym 的步态项加回来**，
  首推 `feet_air_time = 1.0`（原始权重；它直接奖励更长的滞空时间 ⇒ 降步频、增步幅，
  正是我们唯一的缺陷），可选再加 `action_rate −0.01`。这一条**既不减少已有压力**，又命中缺陷。

---

## 8. 数据不是原因：三份参考数据集在同一探针下的对照（2026-09-18 补）

新增 [`probe_dataset_effect.py`](../imgo2_rl/scripts/tools/probe_dataset_effect.py)：同一套「只差节律」的
合成判别问题（正类 = 正常速度相邻帧对，负类 = **3 倍速**帧对），网络 `[1024,512]+ReLU`、`MSE(±1)`、
**λ_gp=10**、每份 1200 步、各数据集**各自标准化**，只换数据：

| 数据集 | 文件 | 帧 | 时长 | 步态模态 | d 跨度 | 风格奖励差/步 |
|---|---|---|---|---|---|---|
| **我们 imgo2_motion** | 21 | 5097 | 101.5 s | 单一 0.600 s 节律 | 0.2506 | 0.178 |
| rl_amp（a1 mocap） | 13 | 1274 | 21.1 s | **trot/pace/canter 三种** | 0.3168 | 0.227 |
| **amp_go2** | 17 | 3740 | 74.5 s | 单一 0.60–0.63 s（+2 段 0.39–0.42 s） | **0.2407** | **0.168** |

**读法**：
1. **数据量与模态都不是我们判别器退化的原因**：我们帧数最多；单节律的 amp_go2 数据甚至比我们
   更"难分开"（0.2407 < 0.2506）；三种步态的 a1 数据也只把跨度从 0.25 提到 0.32（+26%）。
2. ⇒ **「节律」在这套判别器 + 损失 + λ_gp=10 下天生是弱信号**（三个数据集一致）。
   这也解释了为什么 rl_amp / a1 的 AMP-only 配方能走：它们只要求"像数据"，不要求某个特定步频；
   而我们要的是**特定 1.67 Hz**，这个信息在这条通路里本来就传不出来。
3. ⇒ 反过来印证 **amp_go2 的步态不是风格项做出来的**：它的风格上限只有
   `coef 0.2 ×(1−lerp 0.8) = 0.04/步`，真正塑形的是 `feet_air_time` 那一套任务奖励。

**amp_go2 数据实测（全部 17 段，逐足 FFT）**：15 段周期 **0.596–0.625 s（≈1.6 Hz）**
（含全部转弯段），2 段 `faster` 为 **0.385–0.417 s（≈2.5 Hz）**，`stance` 0.490 s（准静止）；
合计 **3740 帧 / 74.8 s**。
**测量陷阱**：用四足 z 的**均值**做 FFT 会因 trot 两对反相而抵消基频、只剩二次谐波
（第一遍算出 3.36 Hz，逐足后是 1.68 Hz）。`docs/gait_reference_baseline.json` 用的是逐足 FFT，没踩这个坑。

**对方案选择的意义**：想在单节律数据上拿到**特定**步频，只能像 go2 那样把它写进任务奖励；
想靠 AMP 出步态，就得按 rl_amp/a1 的配置（任务项近乎清零、风格占优），并接受"AMP 只保证像数据、
不保证节律"。我们（单节律数据 + 目标 1.67 Hz + 要上真机）属于前者。

---

## 9. 落地：amp_go2 配方的移植（2026-09-18 已实现，**待训练验证**）

用户决定「尽可能参考 amp_go2 操作试试看」，于是**新增**一个任务而不是改动原配方：

| | 任务 ID | 环境配置 | agent 配置 |
|---|---|---|---|
| 原（AMP-only） | `Imgo2-basemove-flat-amp` | `Imgo2AmpMoveEnvCfg` | `AMPRunnerCfg`（coef 2.0 / lerp 0.3） |
| **新（amp_go2 配方）** | **`Imgo2-basemove-flat-amp-go2`** | `Imgo2AmpGo2StyleEnvCfg` | `AMPGo2RunnerCfg`（**coef 0.2 / lerp 0.8**） |
| 回放 | `Imgo2-basemove-flat-amp-go2-play` | `Imgo2AmpGo2StylePlayEnvCfg` | 同上 |

> 注意别忘了：**amp_go2 自己也是在平地训的**（`go2_amp_config.py` 里 `mesh_type = 'plane'`），
> 所以这次移植是「平地 + amp_go2 奖励设计」，不是粗糙地形。要走粗糙地形，还要叠加上一轮说的
> 地形相对 height / `amp_root_z` / `terrain_levels` 那几处（另开一次改动与训练）。

**奖励逐字照抄**（`AMP_GO2_RAW_WEIGHTS`）。之所以能照抄：legged_gym 的
`_prepare_reward_function()` 会 `scales[key] *= self.dt`，Isaac Lab 的 `RewardManager.compute(dt)`
也做 `term × weight × dt` —— **两边 `weight` 语义相同**。

| 我们的项 | amp_go2 项 | 原始权重 | 每步系数（×0.02） |
|---|---|---|---|
| `track_lin_vel_xy_exp` | `tracking_lin_vel` | 4.0 | 0.08 |
| `track_ang_vel_z_exp` | `tracking_ang_vel` | 2.0 | 0.04 |
| `lin_vel_z_l2` | `lin_vel_z` | −1.0 | −0.02 |
| `ang_vel_xy_l2` | `ang_vel_xy` | −0.05 | −0.001 |
| `joint_acc_l2` | `dof_acc` | −2.5e-7 | −5e-9 |
| `joint_torques_l2` | `torques` | −1e-4 | −2e-6 |
| `base_height_l2` | `base_height` | −1.0 | **−0.02**（原配方 −5.0，弱 250 倍） |
| `action_rate_l2` | `action_rate` | −0.01 | −2e-4 |
| `undesired_contacts` | `collision` | −1.0 | −0.02（只惩罚 `.*_THIGH`，同参考） |
| `joint_pos_limits` | `dof_pos_limits` | −2.0 | −0.04 |
| `feet_air_time` | `feet_air_time` | 1.0 | 0.02 |

风格侧：`coef 0.2` / `lerp 0.8` ⇒ 混合后风格上限 **0.2 × 0.2 = 0.04/步**，即"任务奖励负责步态、
AMP 只做轻量风格先验"（与我们原配方的 0.540/步 差 13 倍）。

**两处有意偏离参考**（已写进代码注释与测试）：
1. `base_height` 目标 **0.30 m**（参考 0.38 是 Go2 的站高；Imgo2 参考动作的根高是 0.297）；
2. `feet_air_time` 阈值 **0.3 s**（**用户 2026-09-18 决定**；参考硬编码 0.5 s，语义是奖励
   "滞空 >0.5 s / 周期 ≥1 s"，比我们参考动作的 0.600 s 周期还慢）。
   0.3 s 对应的目标周期：实测占空比 0.50–0.56（腾空 0.44–0.50）⇒ 滞空 0.3 s 对应周期
   **0.60–0.69 s（1.45–1.67 Hz）**，正好落在参考的 0.600 s / 1.67 Hz 附近。
   机制上：每步该项 ≈ (1−占空比) − θ/周期，对周期的推动 `∂/∂T = θ/T²` ⇒ **阈值越大推力越强**，
   θ=0 时它对步频完全没有激励；阈值只改变"常数项 + 推力强度"，不改变"每多 1 秒滞空值多少"的
   单位斜率（后者由权重 0.02 决定）。因此若滞空不动，第一顺位是放大权重，第二顺位才是再调阈值。

**离线验证（已做）**：`tests/test_amp_go2_recipe.py` 9 项，其中一项**直接读本机参考项目
`~/Desktop/AMP/amp_go2-main` 的 `class scales` 逐项比对**（数值一致、参考里没有漏抄的非零项）；
另校验 11 个项名真实存在于 `RewardsCfg`、重建项的 `SceneEntityCfg` 正则非空、每步换算等价、
runner 权重、任务注册且旧任务未被改动。全仓 40 项测试通过。

**没做/必须由训练验证**：Isaac Lab 的配置类无法在无 Isaac Sim 的沙箱里实例化
（`import omni.log` 失败），所以**一次都没跑过**。按工作区约定先跑短训练做前置检查：

```bash
cd <repo>/imgo2_rl
python scripts/rl_lab/amp/train.py --task=Imgo2-basemove-flat-amp-go2 \
    --num_envs=256 --max_iterations=100 --seed=42 --headless     # 前置检查
python scripts/rl_lab/amp/train.py --task=Imgo2-basemove-flat-amp-go2 --headless   # 正式训练
```

**风险与判据（重点盯三条）**：
1. **`base_height` 弱了 250 倍** ⇒ 我们自己的 Run2 就是在弱高度项下退化成贴地滑行（0.172 m、贴地率 0.89）。
   短训练里看 `AMP/mean_root_height_m`、`AMP/fraction_root_height_below_0_20m`；若爬行，**只把
   `base_height` 单独调回**（单变量），其它照抄参考。
2. ~~回报量级缩小约 12 倍会拖慢 critic~~ —— **2026-09-18 当晚更正：这条是过度担心**。
   两条理由：① advantage 在 `rollout_storage.py:138` 被归一化 `(adv-mean)/std`，策略梯度对整体尺度不变；
   ② Adam 逐参数尺度不变（梯度缩 c 倍 → m 缩 c、v 缩 c²，`m/√v` 不变），而 actor / critic 参数集不相交，
   共享优化器也不会让两个损失互相压制。唯一实际差异是 `use_clipped_value_loss` 的 `clip_param=0.2`
   （单位是回报）在微小尺度下不再生效 ≈ 退回普通 MSE。⇒ **照抄 amp_go2 的绝对数值无害，不必刻意放大**；
   `Loss/value_function` 从 ~29 掉到 ~1 属预期，不是异常。
   顺带更正一条相关的误读（2026-09-18）：amp_go2 与我们原来的**风格占比其实几乎一样**
   （它 style ≤0.04 / task ≈0.12 ≈ 25:75；我们 0.540 / 1.402 ≈ 28:72），
   差的是**绝对尺度**与**任务奖励的内容**，不是占比。
3. **`feet_air_time` 阈值 0.3 s + 权重 0.02 是否够力**：该项在每步系数 0.02 下只能给到百分之几每步的激励，
   若步频不动，先确认这一项确实被计算（`Episode_Reward/feet_air_time` 应出现在日志里且非零），
   再考虑把它的系数往上调（仍是单变量）。

**成功判据**（用修好的回放工具：`eval_gait.py --task=Imgo2-basemove-flat-amp-go2-play`）：
步周期 ≥0.45 s（现在 0.19）、`air_time_mean_s` 升向 0.3 s、线速度误差 ≤0.6 m/s、
相位仍是 FL-FR≈±180° 的对角步、`mean_root_height_m` 别掉到 0.25 以下。

---

## 10. 澄清：`rl_amp` 里没有 go2 部分（2026-09-18 补充）

用户一度说「主要参考 fan-ziqi 的设置」，随后明确「go2 部分的」。**核实结果：`rl_amp`
（fan-ziqi）里没有 go2**——

* `legged_gym/legged_gym/envs/` 只有 `a1 / anymal_b / anymal_c / base / cassie`；
* 全仓 `find -iname "*go2*"`（排除 `.git`）**为空**；
* 注册任务只有 `anymal_c_rough`、`anymal_c_flat`、`anymal_b`、`a1`、`a1_amp`、`cassie`。

它的 AMP 骨架与 a1 同源、**奖励是 AMP-only**（除 `tracking_lin_vel 1.5/(.005*6)`、
`tracking_ang_vel 0.5/(.005*6)` 外全部为 0，连 `base_height`、`lin_vel_z`、`dof_pos_limits`
都是 0），地形 `mesh_type='plane'`、`measure_heights=False`，`coef 2.0 / lerp 0.3`，
`λ_gp=10`，`terminate_after_contacts_on=["base"]`、`penalize_contacts_on=["thigh"]`。

⇒ 所以"go2 那部分"只有 `ak1raljl/amp_go2` 有，而它已经在 §9 移植完成。
本轮补齐了它最后一处未对齐的 go2 专属设置：**指令范围**
`lin_vel_x [-1.2, 1.5]`、`lin_vel_y ±0.8`、`ang_vel_yaw ±1.0`
（我们原来是 x `(-1.0, 1.5)`、y `±1.0`、yaw `±1.57`）。
⚠️ 注意 `x > ~0.9 m/s` 段**我们的录制数据覆盖不到**（参考动作实测最快 0.842 m/s），
照抄参考的 1.5 上限会放大"要求数据外步态"的矛盾；评估时以 0.3–0.9 为准。

**已核对为"本来就一致"的 go2 设置**（无需改）：`env.num_observations = 45`（actor 无 `lin_vel`）、
`obs_scales`（ang_vel 0.25 / dof_pos 1.0 / dof_vel 0.05）、`tracking_sigma 0.25`、
`soft_dof_pos_limit 0.9`（我们的 `soft_joint_pos_limit_factor=0.9`）、
`reference_state_initialization_prob`（他们都是"全部环境用参考帧初始化"，我们 1.0 等价）、
PPO 全套超参（`init_noise_std 1.0`、`value_loss_coef 1.0`、`clip_param 0.2`、epochs 5 / minibatches 4、
`lr 1e-3` adaptive、`gamma 0.99`、`lam 0.95`、`desired_kl 0.01`、`max_grad_norm 1.0`、
`num_steps_per_env 24`、`entropy_coef 0.01`、replay buffer 1e6）、`λ_gp=10`、判别器 `[1024,512]`。
**明确不一致且【不建议动】的**：`num_privileged_obs`（他们 235 含 187 维高度扫描在 critic 里；
我们 48，actor 侧不受影响）、`clip_observations/clip_actions = 100`（我们的动作 clip ±3 是
部署契约里冻结的值）、`self_collisions`、`max_iterations`（他们写 500000，a1 实际 checkpoint 是
36450；我们保持 40000）。

---

## 11. 关于「1000 轮就能走」这个量级（2026-09-18 用户提供视频）

用户找到参考项目的视频：**1000 轮即可在 Isaac Sim 里正常行走**。与本机实测**同量级**：
我们那次 AMP-only run 在 1000 轮的线速度误差是 1.106（已经在走），最终 24500 轮 0.468；
按 1.25–1.4 s/轮算，1000 轮 ≈ 20 分钟。

**但要把两个门槛分开**：

| 门槛 | 何时达到 | 判据 |
|---|---|---|
| 能走（视频里"看着正常"） | **1000–2000 轮**（20–45 min） | 线速度误差开始下降、不摔 |
| 步态干净（步频/相位/步幅） | **≥10000 轮**（我们那次要到 24500 才干净） | `eval_gait.py` 的步周期、FL-FR 相位、集中度 R |

证据：我们 5000 轮那个 checkpoint 在 Isaac 里看同样是"在走"，但 Gazebo 周期强度只有 0.17–0.20、
足端回放显示是 5 Hz 高频小步；到 24500 轮才变成 0.71–0.83 的干净对角步。
⇒ **1000 轮适合当"任务/奖励没写错、能起来"的冒烟检查，不能当步态合格判据。**

---

## 12. 从 24500 续训 vs 从零：具体做法与判据（2026-09-18 补）

用户问「要不要从 24500 开始而不是重头」。技术上可行，但有三处必须先知道：

1. **`--resume` 的查找路径是新实验自己的 log 根**：`get_checkpoint_path()` 只在
   `logs/amp_rsl_rl/<experiment_name>/` 下扫子目录（**不支持绝对路径**），而新任务的
   `experiment_name = base_move_amp_go2`、旧 checkpoint 在 `base_move_amp/2026-09-17_21-25-57/`。
   ⇒ 先把 checkpoint 放进新实验的 run 目录（复制或软链），再 `--resume --load_run <那个目录>`。
2. **`max_iterations` 在 resume 时是「额外轮数」**：`learn()` 里是
   `tot_iter = self.current_learning_iteration + num_learning_iterations`
   ⇒ 从 24500 续训 3000 轮就写 `--max_iterations=3000`，产物会是 `model_27500.pt`。
3. **resume 会恢复 Adam 状态与当时已衰减到 `1e-4` 的自适应学习率**（新奖励若想从 `1e-3` 起跑需要额外开关），
   并且 **critic 是"过期"的**（它预测的是旧配方的 1.97/步，新配方只有 ~0.16/步）⇒ 前几百轮价值损失有瞬态；
   因为 Adam 逐参数尺度不变，重拟合很快，但看曲线时别误判。

**更关键的是"能不能撼动旧解"**：从 24500 起跑 = 从「5 Hz 小步」这个局部最优出发，而新配方里真正能压步频的项
权重都很小——按每步边际收益估算，滞空从 0.085 s 拉到 0.2 s 只能拿回 **≈ +0.001/步**，
而速度跟踪项对 0.03 m/s 误差的敏感度就有 **≈0.004/步** ⇒ **照抄 amp_go2 的系数很可能推不动它**。
但这件事**值得用一次便宜实验问清楚**：如果旧解推不动，就说明必须"从零 + 放大 `feet_air_time` 系数"。

**因此分三步（信息量优先，成本递增）：**

| 步 | 内容 | 成本 | 判据 |
|---|---|---|---|
| 1 | 两条各跑 **100 轮冒烟**（`--num_envs=256`）：① 从零 `flat-amp-go2`；② 从 24500 续训 | 各 2–3 min | 任务能起、`Episode_Reward/feet_air_time` 非零、`AMP/mean_last_air_time_s` 出现在日志里、高度不崩 |
| 2 | **从 24500 续训 3000 轮** | ~1 h / ~1.7 元 | `AMP/mean_last_air_time_s` 是否从 0.07–0.09 **升向 0.2**、`error_vel_xy` 是否变差。这是"旧解可撼动性"的直接判据 |
| 3 | 若第 2 步没动 ⇒ **从零训练**，并把 `feet_air_time` 系数放大 **20–50 倍**（单变量）；若动了 ⇒ 继续续训到 5–10k | 3.5–4 h | 步周期 ≥0.45 s 且线速度误差 ≤0.6 m/s |

**已加的训练期指标（2026-09-18，不必再跑回放就能盯步频）**：
`AMPOnPolicyRunner.log()` 现在写两条与 `eval_gait.py` 同口径的标量——
`AMP/mean_last_air_time_s`（四足最近一次滞空时长均值；参考 0.20–0.24 s，我们 24500 轮实测 0.067–0.085 s）
与 `AMP/mean_air_time_fraction`（当前腾空足比例 ≈ 1 − 占空比；实测 0.40–0.50），
并在控制台按行打印；取不到接触传感器时安全降级为 `n/a`。
`Episode_Reward/feet_air_time`（奖励值本身）由 RewardManager 自动记录，一直都有。

---

## 13. 粗糙地形版 AMP（2026-09-18 实现，**待训练验证**）

用户决定「着手一个 rough 版本，用简单配置、先对齐 AMP 原仓库」。落地为**新任务**
`Imgo2-basemove-rough-amp`（回放 `-play`），用 `AMPRunnerCfg`（`coef 2.0 / lerp 0.3`）。

**核心原则：配方一行都不改。** 奖励（6 项、每步 4.0/2.0/−5.0/−1.0/−0.05/−2.0）、AMP 超参、
PPO、动作/观测口径全部与平地版 `Imgo2-basemove-flat-amp` 相同 —— 这样"地形"才是**单变量**，
平地对粗糙地形才是可解释的对照。只做**四处因地形而必须**的改动：

| # | 改动 | 为什么必须 |
|---|---|---|
| 1 | 恢复 `ROUGH_TERRAINS_CFG` 生成器 + `terrain_levels` 课程；子地形范围**沿用仓库已有 PPO rough 任务**（boxes 0.025–0.1 m、stairs 0.025–0.08 m、noise 0.01–0.06） | 平地版把 `terrain_generator`/课程都关掉了；Isaac Lab 默认子地形（boxes 0.05–0.2 m、stairs 0.05–0.15 m）对站高只有 0.30 m 的机器人太陡 |
| 2 | 恢复 9 射线 `height_scanner_base`（**只给奖励**），`base_height_l2` 改成**地形相对** | 平地版目标 0.30 m 是世界系；斜坡/台阶上世界系目标自相矛盾，改地形相对才能继续承担"防趴滑" |
| 3 | `mdp.amp_root_z` 新增可选 `sensor_cfg`（默认 `None` ⇒ **平地行为不变**），粗糙版传同一扫描器 | 专家数据永远是平地 0.297 m；粗糙地形上世界系根高随地形起伏，会把 43 维 AMP 观测的最后一维变成地形噪声 |
| 4 | **关掉参考状态初始化** | `reset_amp_reference_state()` 写的是参考动作的**绝对**根高 + 常数偏移（`amp_events.py:60-61` 只给 x/y 加 `env_origins`），**没有地形补偿** ⇒ 在抬高的地形格子上会把机器人初始化到地面以下 |

**刻意不做**（保持"简单配置"）：不把 187 维高度网格加进 actor（**保持 45 维盲走**，
`amp/deploy/config.yaml` 与 C++ 接口不变）；不引入 `feet_air_time`/`collision`/`action_rate`
等 go2 任务项（那是 `flat-amp-go2` 路线，要混就另开一次实验）；不动域随机化（与平地版、
以及 a1 原仓库一致：摩擦/质量/质心/PD/外力都开着）。

**离线验证**：`tests/test_amp_rough_recipe.py` 9 项 AST 契约（地形与课程恢复、子地形范围与
PPO rough 一致、高度奖励与 AMP 根高均为地形相对、参考初始化已关、actor 仍盲走、
配方未被改动、任务注册齐全）→ 全仓 **52 项通过**。

**未验证 / 风险**：
1. Isaac Lab 配置类在本机沙箱无法实例化（`import omni.log`），**一次都没跑过**；
2. **本仓库的 rough 任务（含 PPO rough）从来没有在本机跑过的记录**，
   `height_scanner_base` 对生成地形的射线命中（`mesh_prim_paths=["/World/ground"]`）
   也只按上游惯例配置 ⇒ 短训练要确认：`mean_root_height_m` 的量级是否合理（地形相对后应在 0.30 附近，
   而不是被地形高度抬到 0.4+）、`terrain_levels` 是否在涨；
3. **blind + 单帧无历史**在粗糙地形上明显更难（要重试的可能方向：给 critic 加高度扫描做
   非对称 actor-critic、或加观测历史），预期需要比平地更多的轮次才能站住；
4. 评估口径：`eval_gait.py` 的足端指标在斜坡上含义与平地不同，别直接与
   `docs/gait_reference_baseline.json`（平地录制）逐项比 —— 先看 `base_height`（地形相对）与接触时序。

**命令**：

```bash
cd <repo>/imgo2_rl
python scripts/rl_lab/amp/train.py --task=Imgo2-basemove-rough-amp \
    --num_envs=256 --max_iterations=100 --seed=42 --headless     # 前置检查
python scripts/rl_lab/amp/train.py --task=Imgo2-basemove-rough-amp --headless
```

---

## 14. 更正 §13：粗糙版改为**忠于 rl_amp**的配方（2026-09-18 晚，用户决定）

用户明确要求：「忠于 AMP，使用 fan-ziqi 版本的设置，**只有 lin vel 和 ang 的速度跟踪奖励**，
amp 风格和这个奖励的比例也是。」⇒ §13 里"配方与平地 AMP-only 相同（6 项 4.0/2.0/−5.0）"作废，
粗糙版改为 rl_amp 的原样配方，并**补一个同配方的平地版**，让"地形"成为单变量。

**参考数值**（`rl_amp/legged_gym/legged_gym/envs/a1/a1_amp_config.py` 的 `class scales`）：

| | 参考原样 | 我们的实现 | 每步系数 |
|---|---|---|---|
| `tracking_lin_vel` | `1.5 * 1./(.005*6)` = **50.0** | `RLAMP_TRACK_LIN_VEL_RAW` | **1.0** |
| `tracking_ang_vel` | `0.5 * 1./(.005*6)` = **16.6667** | `RLAMP_TRACK_ANG_VEL_RAW` | **0.3333** |
| 其余 14 项（`lin_vel_z`/`ang_vel_xy`/`orientation`/`torques`/`dof_vel`/`dof_acc`/`base_height`/`feet_air_time`/`collision`/`feet_stumble`/`action_rate`/`stand_still`/`dof_pos_limits`/`termination`） | **0** | `apply_rlamp_task_rewards()` 一律清空 | 0 |
| AMP 侧 | `coef 2.0 / lerp 0.3`（= 我们的 `AMPRunnerCfg`） | 同 | 风格上限 **1.40/步**、任务上限 **0.40/步** ⇒ 上限比 ≈3.5:1；按实测 `style_raw≈0.39` 折算，实际 **风格 ≈2/3** |

**新增任务**（成对，配方完全相同）：`Imgo2-basemove-flat-amp-rlamp`、`Imgo2-basemove-rough-amp-rlamp`
（各带 `-play`）；`Imgo2AmpRLAmpEnvCfg` / `Imgo2AmpRoughEnvCfg`。

**粗糙版保留的四处地形相关改动**（与 §13 相同）：① terrain generator + `terrain_levels` 课程
② 恢复 9 射线 `height_scanner_base`（**只给 AMP 观测用**，因为 rl_amp 没有高度奖励）
③ `mdp.amp_root_z` 传该扫描器 ⇒ AMP 观测最后一维是"离地形高度" ④ 关掉无地形补偿的参考状态初始化。

**不照抄参考的两处**（已在代码注释标注）：指令范围保持我们原来的
`x(-1.0,1.5) / y±1.0 / yaw±1.57`（rl_amp 是 `x[-1.0,2.0] / y±0.3`，其中 2.0 m/s 超出我们录制数据
覆盖的 0.842 m/s）；`only_positive_rewards`（参考基类默认 `True`，会把总回报在 0 处截断，
我们的框架没有这一层，属已知差异）。

**风险（比 §13 更高，必须盯）**：rl_amp 配方**没有任何姿态/高度约束**（连 `lin_vel_z`/`ang_vel_xy`
都是 0），参考项目靠"风格项 + base 触地终止"兜底；而我们的风格项实测**梯度≈0**（§1–§3），
所以「趴地滑行」的风险显著上升（Run2 实测高度 0.172 m、贴地率 0.89 就是在弱高度项下出现的）。
训练时**必须**盯 `AMP/mean_root_height_m` 与 `AMP/fraction_root_height_below_0_20m`；
一旦趴地，退回 `flat-amp`/`rough` 的 6 项配方（含 −5/步高度项），或只把高度项单独加回（单变量）。

**离线验证**：新增 `tests/test_amp_rlamp_recipe.py` **8 项**（只保留两项、helper 清空其余、
项名存在于 `RewardsCfg`、**直接读参考源码比对**原始权重 50/16.667 与非零项集合、
每步 1.0/0.3333 与风格:任务比例、runner 用 2.0/0.3、平地与粗糙同配方、原 6 项配方未被改动），
并同步更新 `test_amp_rough_recipe.py` ⇒ 全仓 **60 项通过**。

### 14.1 粗糙版加特权 height scan（critic 48 → 235，actor 仍 45）（2026-09-18 晚）

用户决定「就给粗糙版加」。落地：`Imgo2AmpRoughEnvCfg` 里恢复 187 维 `height_scanner`
（`GridPatternCfg(resolution=0.1, size=[1.6,1.0])` → 17×11）并**只**接到 critic：

| | 维度 | 说明 |
|---|---|---|
| actor（policy） | **45**（不变） | `observations.policy.height_scan` 保持 `None`，并在代码里用 `assert` 守住 |
| critic | **48 → 235** | 多出的 187 维 = 地形高度网格；**正好等于 amp_go2 的 `num_privileged_obs = 235`** |
| 判别器 | 43（不变） | 它吃 `observations.amp`，与 critic 无关 |
| 部署契约 | **不变** | `amp/deploy/config.yaml` 的 `num_observations: 45` 与 C++ 接口都不受影响 |

**代价（必须记住）**：粗糙版从此比平地版多一个变量（"critic 是否看地形"），
所以**平地对粗糙不再是严格的单变量对照**。若以后要做干净的泛化对比，要么给平地版也加一个
（平地上扫描是常数，等于给 critic 187 个常量输入，无害但无信息），要么跑一个"粗糙 − 特权 critic"的消融。
静态核算写进测试：`test_critic_dim_contract_235` 会从 `GridPatternCfg` 反算网格维度并断言 235。

---

## 15. actor / critic 与 rl_amp 的逐项对照（2026-09-18 晚，用户提问后核对）

**结论：网络结构完全一致，critic 维度与组成也一致，但 actor 差一个 3 维块。**

| | rl_amp（fan-ziqi，`a1_amp`） | 我们（`flat-amp` / `flat-amp-rlamp`） | 一致？ |
|---|---|---|---|
| 网络 | `actor/critic_hidden_dims = [512,256,128]`、`activation='elu'`、`init_noise_std=1.0` | 同 | ✅ |
| **actor 维度** | **42** | **45** | ❌ |
| actor 组成 | `projected_gravity 3 + commands 3 + dof_pos(相对) 12 + dof_vel 12 + actions 12` | `base_ang_vel 3 + projected_gravity 3 + commands 3 + joint_pos 12 + joint_vel 12 + last_action 12` | ❌ **差 `base_ang_vel` 这 3 维** |
| **critic 维度** | **48** | **48** | ✅ |
| critic 组成与顺序 | `lin_vel 3 + ang_vel 3 + gravity 3 + commands 3 + dof_pos 12 + dof_vel 12 + actions 12` | **完全相同（含顺序）** | ✅ |
| critic 各项缩放 | `lin_vel 2.0 / ang_vel 0.25 / dof_pos 1.0 / dof_vel 0.05` | **全 1.0**（Isaac Lab 默认；我们只缩放了 policy 组） | ❌ |
| actor 各项缩放 | `ang_vel 0.25 / dof_pos 1.0 / dof_vel 0.05`（gravity/commands 不缩放） | `ang_vel 0.25 / joint_pos 1.0 / joint_vel 0.05` | ✅（共同项一致） |
| 动作 | `action_scale = 0.25`（统一）、`clip_actions = 100`（≈不裁剪） | 髋 `0.125` / 其余 `0.25`、`clip = ±3`（部署契约冻结值） | ❌（执行层差异） |
| PD 增益 | `stiffness 20.0 / damping 0.5` | `stiffness 25.0 / damping 0.5` | ❌（物理差异，机器人不同） |

**关键依据**：`rl_amp/legged_gym/legged_gym/envs/base/legged_robot_amp.py:250-271`——
先把 48 维 privileged obs 拼好（含 `base_lin_vel`、`base_ang_vel`），然后
`if self.num_obs == self.num_privileged_obs - 6: self.obs_buf = self.privileged_obs_buf[:, 6:]`
**切掉前 6 维 = 线速度 3 + 角速度 3** ⇒ 参考的 actor **既不看线速度也不看角速度**，
只靠 `projected_gravity` 感知姿态。我们（AMP-05）只去掉了线速度、保留了 `base_ang_vel`
（真机 IMU 本来就有陀螺仪）。

**要不要为了"忠于 fanziqi"把 actor 也改成 42？** 我的建议是**不改**，并把差异写清楚：
1. 代价：`base_ang_vel` 是部署契约里的一项（`amp/deploy/policy/imgo2/amp/config.yaml` 的
   `observations` 列表 + C++ 观测拼接都要同步），改动会连带导出与部署复核；而且丢掉真机已有的
   陀螺仪信息会让学习更难（参考去掉它是为了"完全不依赖速度估计"的硬件通用性，不是硬件限制）。
2. 本轮要检验的是**配方**（2 项奖励 + `coef/lerp`），观测集差异是另一个变量；混在一起改，
   结果又说不清。
3. 若确实要严格复现参考的 42 维：作为**独立单变量**做（改 actor 观测 → 新任务 → 重训 → 重导出 →
   复核三条契约），别和这次配方实验叠在一起。

**另一处可以顺手对齐但需要重训才能验证的**：critic 各项缩放（我们全 1.0，参考 2.0/0.25/1.0/0.05）。
它只影响 critic 的输入尺度（不影响部署），但改了就等于换了价值网络的输入分布，必须重训才算验证。

### 15.1 按用户要求统一：critic 缩放 + commands 缩放 + 动作缩放/裁剪（2026-09-18 晚）

用户决定「critic 缩放和动作缩放 + clip 和 fanziqi 统一一下」。落地为
`apply_rlamp_obs_and_action_scales()`（只作用于两个 rlamp 任务，不动其它变体）：

| 项 | 参考（rl_amp） | 我们原来 | 现在（rlamp 任务） | 影响面 |
|---|---|---|---|---|
| critic `lin_vel` 缩放 | **2.0** | 1.0（Isaac Lab 默认） | 2.0 | 仅 critic |
| critic `ang_vel` / `dof_pos` / `dof_vel` | **0.25 / 1.0 / 0.05** | 1.0 / 1.0 / 1.0 | 同参考 | 仅 critic |
| **commands 缩放**（actor+critic） | **`[2.0, 2.0, 0.25]`**（`legged_robot.py:515` 的 `commands_scale`） | `1.0` | `(2.0, 2.0, 0.25)` | **动到 actor 输入尺度**（用户只点名了 critic；我按"统一"一并处理，若要改回只需一行） |
| 动作缩放 | **0.25 统一**（`a1_amp_config.py:72`） | 髋 0.125 / 其余 0.25 | 0.25 统一 | **部署契约** |
| 动作裁剪 | **±100**（≈不裁剪） | ±3 | ±100 | **部署契约** |

**代价与后续必做（重要）**：动作缩放与裁剪、commands 缩放都属于**部署契约**。
这两个任务的策略将来若导出到 `imgo2_deploy/policy/imgo2/amp/`，`config.yaml` 必须同步：
`action_scale`（全 0.25）、`clip_actions_lower/upper`（±100）、`commands_scale`（`[2.0, 2.0, 0.25]`）
—— 都是 yaml 字段，C++ 不用改（`rl_sdk.cpp` 按 yaml 取值）。**现在不要改那份 yaml**：
它服务的是已部署的 24500 策略（髋 0.125 / ±3），改了会让当前部署件行为改变。
`base_ang_vel` 的 actor 侧缩放（0.25）本来就与参考一致，未动。

**验证**：`test_amp_rlamp_recipe.py` 新增 3 项（helper 覆盖 policy+critic 且 critic 额外 lin_vel、
**直接读参考基类源码**比对 `obs_scales`/`clip_actions` 与 a1 的 `action_scale`、其它变体不得被动到）；
全仓 **65 项通过**。

### 15.2 按用户要求继续对齐：指令范围 + 域随机化 + reset 分布 + 噪声（2026-09-18 晚）

用户决定「**都对齐。除了关节 kp/kd，其它都对齐一下**」。落地为
`apply_rlamp_env_settings(cfg, *, custom_origins=False)`（同样只作用于
`flat-amp-rlamp` / `rough-amp-rlamp` 两个任务及它们的 play 版），
另加 `AMPRLAmpRunnerCfg`（`min_normalized_std`）。逐项依据都来自
`rl_amp/legged_gym/legged_gym/envs/{base/legged_robot.py,base/legged_robot_config.py}` 与
`envs/a1/a1_amp_config.py`：

| 项 | 参考（rl_amp / a1） | 我们原来 | 现在（rlamp 任务） | 依据 |
|---|---|---|---|---|
| `lin_vel_x` | `[-1.0, 2.0]` | `(-1.0, 1.5)` | `(-1.0, 2.0)` | `a1_amp_config.py` `commands.ranges` |
| `lin_vel_y` | `±0.3` | `±1.0` | `±0.3` | 同上 |
| `ang_vel_yaw` | `±1.57` | `±1.57` | 不变 | 同上（本来就一致） |
| 指令重采样 / heading | `10 s` / `False` | 同 | 不变 | 同上（本来就一致） |
| 摩擦 | 单一 `[0.25, 1.75]`（`randomize_friction`） | static `(0.3,1.0)`、dynamic `(0.3,0.8)` | static/dynamic 都 `(0.25, 1.75)` | `legged_robot.py:266-269` |
| 恢复系数 | 不随机化（legged_gym 恒 0） | `(0.0, 0.5)` | `(0.0, 0.0)` | 参考无此项 |
| base 质量 | `+= U(-1, 1)` kg | `(-1.0, 3.0)` | `(-1.0, 1.0)` | `legged_robot.py:315-316`（只改 body 0） |
| 其它 body 质量缩放 | **没有** | `(0.7, 1.3)` scale | **关掉** | 参考 `domain_rand` 无此字段 |
| CoM 偏移 | **没有** | `±0.05 m` | **关掉** | 同上 |
| 外力/力矩 | **没有** | `±10 N / ±10 Nm`（reset） | **关掉** | 同上 |
| PD 增益 | `×U(0.9, 1.1)`，**启动时随机一次** | `×U(0.5, 2.0)`，`mode="reset"` | `×U(0.9, 1.1)`、`mode="startup"` | `randomize_gains` |
| 推力 | `±1.0 m/s`（x,y），每 `15 s` | `±0.5 m/s`，每 `10–15 s` | `±1.0 m/s`，每 `15 s` | `legged_robot.py:334/417`、`push_interval_s` |
| reset 根姿态 | 平地：**不加** xy/偏航扰动；地形：xy `±1 m` | `±0.5 m` xy、`yaw ±3.14` | 平地 `(0,0)`、粗糙 xy `±1 m`、都不随机偏航 | `legged_robot.py:399-409` |
| reset 根速度 | 6 维全 `U(-0.5, 0.5)` | 同 | 不变（本来就一致） | 同上 |
| reset 关节角 | `default_dof_pos × U(0.5, 1.5)`、`dof_vel=0` | 无随机（`×1.0`） | `×U(0.5, 1.5)`、`dof_vel=0` | `legged_robot.py:411-425` |
| actor 噪声 | `dof_pos 0.03` / `ang_vel 0.3`（覆盖基类 0.01/0.2）；其余同基类 | `0.01` / `0.2` | `0.03` / `0.3` | `a1_amp_config.py` `noise.noise_scales` |
| `min_normalized_std` | `[0.01]*12` | `[0.05,0.02,0.05]*4` | `[0.01]*12`（`AMPRLAmpRunnerCfg`） | `a1_amp_config.py` runner |

**关于 `ang_vel` 噪声的诚实说明**：参考的 actor 是 `privileged_obs_buf[:, 6:]`，**线速度与角速度
都被切掉** ⇒ 加在 `[:, 3:6]` 上的噪声随即被丢弃，那条 `ang_vel = 0.3` 在参考里是**死代码**。
我们保留 `base_ang_vel`（AMP-05 的决定，真机 IMU），所以它在**我们这里有效**；取参考声明的 0.3
是对齐"声明值"的最直接做法，但严格说参考的 actor 行为里没有这一项。

**关于地形/恢复系数**：`velocity_env_cfg.py` 的地形材质是
`restitution=1.0` + `restitution_combine_mode="multiply"`（Isaac Lab 模板默认）。**没有改它**，
原因是 `randomize_rigid_body_material` 会把**机器人**每个形状的 restitution 显式写成 0，
multiply 组合下净恢复系数必然为 0；改共享地形材质会牵动 PPO/AMP 全部任务与已部署策略。
（注意 play 配置会关掉该项随机化，所以**回放/评估**时机器人拿到的是资产默认材质 —— 这是
改动前就存在的行为，不是本轮引入的。）

**对齐的代价（减少随机化 ⇒ 更贴近标称动力学，鲁棒性下降）**：关掉 CoM/其它 body 质量/外力力矩、
去掉初始姿态与偏航扰动、摩擦与 PD 增益范围收窄、关节初始角只按 ×[0.5,1.5] 缩放。
另外 `lin_vel_x` 上限 2.0 m/s **超出我们录制动作数据覆盖的 0.842 m/s**，那一段没有专家参考可比。

**仍未对齐（明确记录）**：

1. **actor 观测 45 vs 42**（我们多 `base_ang_vel`）：与 §15 的建议一致，先不动；本轮实验只改配方/接口，
   要严格 42 维请作为独立单变量另开任务。
2. **关节 kp/kd**：`25.0/0.5`（我们）vs `20.0/0.5`（参考）—— 用户本轮明确排除。
3. **共享资产/求解器**：`solver_velocity_iteration_count=1`（参考 `num_velocity_iterations=0`）、
   接触 `contact_offset`、`armature`/`friction` 等 `IMGO2_CFG` 里的量。它们被所有任务（含已部署的
   24500 策略）共用，改动会让既有基线与部署件不可比，故不动。
4. **AMP 数据来源**：我们用的是自录真机动作（21 段），参考用 mocap；43 维观测的组成与顺序一致
   （§3 已核对），但分布不同。

**验证**：`test_amp_rlamp_recipe.py` 新增 6 项——
常量值、helper 逐字段覆盖 + 两个任务的接线（粗糙版 `custom_origins=True`）、
**把 helper 的 AST 抽出来用鸭子类型 cfg 真跑一遍**（Isaac Lab 的 configclass 在无 GPU 机器上
无法实例化）、**直接读参考源码**比对 commands/domain_rand/noise（含"参考 `domain_rand` 不继承基类
⇒ 确实没有 CoM/惯量/外力"的判据）、参考 reset 分布（正则抓 `torch_rand_float` 端点）、
`AMPRLAmpRunnerCfg` 与任务注册（4 个 rlamp 任务都指向它）。全仓 **73 项通过**（原 65 项）。

**状态**：**已实现，待训练验证**（离线只能证明"配置写成了参考的数值"，不能证明学出来的步态更好）。
