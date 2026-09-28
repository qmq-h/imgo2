# MoE 用于腿足运动：两条参考项目 vs 我们的 CMoE port（2026-09-28）

用户问："CMoE 原本是用在人形上的，是否有用 CMoE 在四足上的项目参考"。本节记录查证结果与对照结论。

## 1. 两条最相关的参考

### ① CMoE 原文（人形，ICRA 2026，复旦）——**我们的直接来源**
* 项目页 <https://hoshi-no-ai.github.io/CMoE/>、代码 <https://github.com/Hoshi-No-Ai/CMoE>、arXiv 2603.03067。
* 机器人：**Unitree G1 人形**；单阶段端到端；含状态/地形两个估计器。
* **原文摘要里点明的核心问题**："Vanilla MoE ... **in practice, the gating network exhibits nearly uniform
  expert activations across different terrains, weakening the expert specialization**"。
* **他们的解法**：加**对比学习目标**——"maximizes the consistency of expert activations **within the same
  terrain** while minimizing their similarity **across different terrains**"。
* 效果：连续 20 cm 台阶、80 cm 沟壑，混合地形上的稳健自然步态。
* ⇒ **重点**：CMoE 的全部贡献就是"把趋均匀的门控压成地形专精"。**我们这份 port 已经实现了这个对比损失**
  （`cmoe_actor_critic.py:186-206 compute_contrastive_loss`，Sinkhorn + prototypes，
  `cmoe_ppo.py:29/56/147/170` 里 `contrastive_loss_coef = 1.0` 加进总损失）。

### ② MoE-Loco（**四足**，arXiv 2503.08564，清华/上海期智）——**四足上最直接的参考**
* 项目页 <https://moe-loco.github.io/>、arXiv <https://arxiv.org/abs/2503.08564>。
* 机器人：**Unitree Go2 四足**；真机零样本部署；50 Hz；Kp 40 / Kd 0.5。
* MoE 设计：**actor 与 critic 都带 MoE，共享同一个门控网络**（`a_t = Σ ĝ_i f_i(h_t)`，`ĝ = softmax(g(h_t))`），
  **6 个专家**；训练目标只有 `L_surro + L_value + L_recon`（**没有额外的负载均衡项**）。
* 任务：**9 个** —— 四足步态 5 个（横杆/沟/挡板爬行/上楼梯/斜坡）+ **双足步态 4 个**（站起/平地/斜坡/下楼梯），
  即同一个 Go2 既四足走又双足站走；观测里含**步态 one-hot**；奖励按步态分族（`r_quad = r_track + r_reg`、
  `r_bip = r_track + r_stand + r_reg`）。
* 训练：**两阶段 + 适配**——40k 平地（两种步态）→ 80k 复杂地形 → 10k PAS（改成纯本体感知，
  估计器/LSTM/MoE 权重从第一阶段复制）。
* **他们给出的关键证据：梯度冲突**。同一框架换成同参数量的普通 MLP（"Ours w/o MoE"）后，
  跨任务梯度余弦相似度出现**负值**（如 Bar(q)↔SlopeUp(b)：MoE **+0.278** vs 无 MoE **−0.132**；
  Bar(q)↔SlopeDown(b)：+0.091 vs −0.128），负梯度占比也更高；成功率对比悬殊
  （Stair(q) **0.924** vs 0.264；Mix **0.879** vs 0.571）。
* 他们还验证：**专家自发分化**（平衡／爬行／越障），并可以**手工改门控权重 `w[i]` 组合新技能**。

### ③ 其他被提到的（非 CMoE、部分非四足）
* MELA：用预训练专家组合运动策略（偏基础技能）。
* ManyQuadrupeds：跨机型统一策略（不是 MoE）。

## 2. 为什么它们成立、我们的不成立（对照）

| 维度 | MoE-Loco（四足） | CMoE（人形） | **我们（四足 CMoE port）** |
|---|---|---|---|
| 任务冲突 | **极大**：双足 ↔ 四足（同两条腿要当腿/当手臂）+ 5 类障碍 | **大**：20 cm 连续台阶 / 80 cm 沟（步态必须不同） | **小**：单一"向前走 + 越障"，命令族单一、地形是难度渐变 |
| 专家初始 | 随机（**天然彼此不同**） | 随机 | **5 份完全相同**（AMP 先验零填充 + jitter 0.01 ⇒ 实测逐位相同） |
| 门控训练 | 纯 PPO，靠冲突自然分化 | **对比损失**（核心贡献） | 有对比损失（coef 1.0），但门控熵只从 1.609 → **1.29**（压得不够狠） |
| 梯度冲突度量 | 有定量表（无 MoE 时余弦为负） | — | **没量过** |
| 结果 | 专家自发分化 + 可手工重组 | 门控地形专精 | 门控**确实**地形条件化（gap→E1/E2、mix→E0、flat→E4），但**步态全同**（11 类地形全 lockstep，FL-FR 0.04–0.13） |

**结论**：MoE 的收益来自"**任务之间存在真冲突**"，不是"专家多"。我们的任务里没有可分的冲突
（PPO 单策略 + 27 项奖励就能在 rough 上走出稳定步态 ⇒ 这个任务不需要 MoE 的容量），
所以门控只能学出"地形条件化的软混合"，专家全都长成通才。

## 3. 如果还想让 MoE 生效：三个具体旋钮（按性价比）

1. **把对比损失调狠**（这是 CMoE 原文的**唯一核心机制**，我们只做到"温和"）：
   `contrastive_loss_coef`、`temperature`、prototype 数量/维度。验收读数：`Loss/Contrastive*` 曲线下降、
   门控熵下降、`expert_report.py` 的逐地形 argmax 集中度上升。
2. **恢复专家的初始多样性**：`--init_experts_jitter` **0.01 → 0.05~0.1**；或
   `--init_experts_mode=first`（专家 0 ＝ AMP 的 trot，其余随机）⇒ 门控**有理由**把 flat 路由到专家 0。
   代价：混合里若有 80% 随机专家，前几百轮会摔得厉害（离线实测 `mode=first` 的初始动作 RMSE 2.33）。
3. **引入真冲突**：给观测加**显式步态 token**（trot / bound / 爬行 …）或把"平地优雅 trot"与"越障"
   当成两个对立任务 ⇒ 冲突出现，MoE 才有可分的东西。代价：观测契约变更（要重训 + 部署同步）。

**或者**接受"这个任务不需要 MoE 的容量"，走单策略 + AMP 先验 + 平地教师锚（见
`docs/parkour_reference_2026-09-28.md` §7.2 的 v4 提案）。

## 4. 判决性实验（便宜，先做）

* `cmoe/play.py --force_expert k`（**待实现**，~20 行 + 测试）：把门控 one-hot 到第 k 个专家，逐地形量相位与任务指标；
* `scripts/tools/expert_report.py`（已有）：逐地形**argmax 占比**（分工证据）；
* `Loss/Contrastive*` 曲线（已有读数）。
⇒ 三条一起看：若 5 个专家表现几乎一致、且对比损失已收敛而熵不降，就说明**MoE 在我们的任务里没有功能分工**，
此时继续调门控/奖励不如直接放弃 MoE 容量。

*（限制：以上论文信息来自项目页/arXiv 摘要与正文；未运行其代码，未核对它们的超参与实现细节与我们的差异。）*
