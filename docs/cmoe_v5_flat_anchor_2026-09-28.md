# CMoE v5（2026-09-28 用户决定）：**只有一个专家装先验 + 只在 flat 上锚定**

用户原话依次为："我也倾向于，5个专家只有一个用 amp 预训练结果初始化"、"合理。只在 flat 上锚定"。
本记录写清**做了什么、为什么这样做、怎么验证、还差什么**。

## 1. 为什么"只在 flat 上锚"是必须的（不是优化）

核实 AMP 任务的真实契约（`amp_env_cfg.py:98-113`，任务 id `Imgo2-basemove-flat-amp-height`）：

* **平地**：任务名与地形都是 flat；
* **地形盲**：`self.scene.height_scanner = None`、`height_scanner_base = None`（场景里**没有射线扫描器**），
  `observations.policy.height_scan = None`、`critic.height_scan = None`；
* **45 维** = `base_ang_vel 3 + projected_gravity 3 + velocity_commands 3 + joint_pos 12 + joint_vel 12 + last_action 12`
  （policy 连 `base_lin_vel` 都没有）。

⇒ 在斜坡/台阶/沟壑上，**最优动作本身依赖地形**；把一个地形盲策略锚在那儿等于**强迫策略忽略地形**
（也正是我们实测"先验在极端地形上 OOD、RMSE 被放大到 15+"的成因）。放在 `flat` 上则恰好就是
"**平地始终 trot**"这一条需求。

架构上的分工因此是自洽的：
| 模块 | 角色 |
|---|---|
| 专家 0 | 地形盲的**平地 trot 参考**（被锚住；它对后 112 维输入的权重保持在 0 附近） |
| 专家 1–4 | 自由演化（各自去适应台阶/沟壑/混合障碍…） |
| **门控** | 输入是 157 维（含 77 维地形扫描）⇒ **唯一能"看地形决定用谁"的模块** |

## 2. 实现（四处，全部可离线单测）

| # | 文件 | 内容 |
|---|---|---|
| ① | `utils/pretrained_prior.py::init_gate_bias` | 把 `gating_network` **末层**的 bias 设成 `bias[k]=margin`、其余 0（可选再缩小末层权重 ⇒ 初始门控几乎只由 bias 决定）。**只改门控，不动任何专家** |
| ② | `utils/terrain_masks.py`（新） | `terrain_columns()` / `anchor_weights()`：按 **Isaac 的按比例列分配**（同 `gait_dump.expand_terrain_names`）算出 `flat` 的列索引（本任务 = **[37, 38, 39]**），并把逐环境 `terrain_types` 映成 `[N]` 权重 |
| ③ | `storage/cmoe_rollout_storage.py` + `algorithms/cmoe_ppo.py` | storage 多一个**逐样本** `anchor_weight`（默认 0 ⇒ 与旧行为完全一致）；算法侧 `teacher_policy` / `anchor_expert`，锚损失 = `Σ w·‖f_k(x) − a_teacher(x[:45])‖² / Σ w` |
| ④ | `runners/cmoe_on_policy_runner.py` | 启动时算锚定列；每个控制步算 `w = 地形掩码 × 线性衰减`（`anchor_coef → anchor_coef_final`，`anchor_decay_iters` 轮）交给算法；日志 `Loss/anchor_prior`、`Policy/anchor_weight_mean` |
| ⑤ | `cmoe/play.py --force_expert K` + `modules/cmoe_actor_critic.py::forced_expert` | 把门控**强制 one-hot** 到第 K 个专家 ⇒ 混合输出严格等于该专家（逐个专家回放的判决性诊断） |

两个关键的正确性细节：
* 锚的梯度**只**进被锚的那个专家 ⇒ `actor_input` 显式 `.detach()`，否则它会顺带把状态/地形**估计器**
  也拉走（估计器有自己的损失，会互相打架）；
* 未启用（没有教师 / 权重全 0）时不写 `Loss/anchor_prior` 标签（不会留一条恒 0 的曲线）。

## 3. 验证（本机离线，`tests/test_cmoe_v5_anchoring.py` **11 项**）

* **门控偏置**：偏置后目标专家的门控权重 `> 0.95`、混合输出≈该专家（相对容差 5%）；`shrink` 的语义
  被钉成"初始门控几乎不随输入变化"（std < 1e-3）——**它不会让目标专家更占优**（这点容易搞反）；
  非法 expert 下标 / 没有 gating_network 报错。
* **强制专家**：对 k=0,1,2 逐个验证 `mixed == experts[k](build_actor_input(obs))`（1e-6），门控 one-hot。
* **只在 flat 上锚**：`flat` 的列 = `[37, 38, 39]`（按比例分配）；地形名打错**显式报错**（防静默失效）；
  权重按 `terrain_types` 正确置 0/scale；storage 往返 + 未设置的步必须为 0；`mini_batch_generator`
  多出第 11 项且 batch 维度一致。
* **锚损失语义**：只有被锚专家有梯度（其它专家、门控、共享编码器**全为 None**）；权重全 0 或没有教师
  ⇒ 损失恒 0；只有权重非零的样本参与（与手算的 masked MSE 一致）。

全仓离线 **494 通过 / 8 跳过**；`py_compile`、`check_reward_overrides`、`git diff --check` 干净。
**未验证**：没有起训练（本机无 GPU）——门控偏置/锚定在真实 Isaac 环境里的行为需要训练机确认。

## 3.5 冒烟测试抓到的启动 bug（已修）

第一次冒烟（`--num_envs=64 --max_iterations=2`）**启动即崩**：

```
File ".../CMoE_env_cfg.py", line 717, in __post_init__
    self.rewards.feet_gait.weight = 0.0
AttributeError: 'NoneType' object has no attribute 'weight'
```

* **根因**：v3 起 `feet_gait` 在**父类** `Imgo2CMoERoughEnvCfg` 就已归零，父类的
  `disable_zero_weight_rewards()` 会把 0 权重项 `setattr(..., None)`
  （实现 `velocity_env_cfg.py:738-744`："If the weight of rewards is 0, set rewards to None"）
  ⇒ 子类 `Imgo2CMoEGaitFreeEnvCfg` 再裸赋值就撞 `None`。
* **修法**：五项归零统一加 **None 守卫**；审计工具用 `ast.walk` 收集赋值，`if` 包裹后仍能正确识别
  （复核：`cmoe-gaitfree` 仍是 23 项、五项都在"权重 0 ⇒ 移除"列表）。
* **回归测试**：`tests/test_reward_terrain_partition.py::test_gaitfree_zeroing_is_none_safe`。
* **教训**：改奖励权重会让父类把项从配置里**移除**，任何子类的后续赋值都必须容 `None`；
  这类错误只在**真实构造 cfg** 时暴露（离线 AST 审计看不到）⇒ 每次改奖励后**先跑 2 轮冒烟**。

## 4. 训练机命令（A＝用户决定的版本；B＝更保守的变体）

```bash
cd <repo>/imgo2_rl
PRIOR="$(pwd)/logs/amp_rsl_rl/base_move_amp/2026-09-17_21-25-57/model_24500.pt"

# A：只有一个专家装 AMP（专家 0＝纯 AMP，其余 4 个随机）；门控初始偏向专家 0；锚只在 flat
PYTHONUNBUFFERED=1 setsid nohup bash scripts/run_isaaclab.sh scripts/rl_lab/cmoe/train.py \
  --task=Imgo2-basemove-rough-cmoe-gaitfree --num_envs=4096 --headless --max_iterations=2000 --seed 1 \
  --init_experts_from="$PRIOR" --init_experts_mode=first --init_experts_std=true \
  --init_experts_critic=false --init_gate_bias=0 --init_gate_bias_margin=4.0 \
  --anchor_coef=0.25 --anchor_coef_final=0.08 --anchor_decay_iters=1000 \
  --anchor_terrain_names=flat --anchor_expert=0 \
  --run_name=cmoe_v5_flathero </dev/null > logs/run_v5.log 2>&1 &

# B（更保守）：5 个专家都起自 AMP（专家 0 纯净、其余只抖动"地形/估计器"那 112 列）
#   …把 --init_experts_mode=first 换成 --init_experts_mode=all --init_experts_jitter=0.05
```

启动时应看到：
```
[INFO]   门控初始偏置到专家 0（margin=4.0, shrink=0.0） ⇒ 第 0 步的混合动作 ≈ 该专家…
[INFO]   先验锚定：专家 0；地形 ('flat',)（3 列）；权重 0.25 → 0.08（1000 轮线性衰减）
```

## 5. 验收标准

| 读数 | 期望 |
|---|---|
| `Policy/anchor_weight_mean` | 启动 ≈ 0.25×flat 环境占比，随轮数线性降到 ≈0.08×占比 |
| `Loss/anchor_prior` | 有界、随训练下降（专家 0 一直贴着教师）；**若它上升**说明任务梯度在把专家 0 拉走（锚太弱） |
| `--force_expert 0` 在 flat 上回放 | 与 `--prior` 回放（纯 AMP）行为接近 ⇒ 专家 0 没被改写 |
| `expert_report.py` 的 flat 列 `a0`（argmax 占比） | 显著高于其它地形列（门控学会"平地→专家 0"） |
| `flat` 的 `gait_report.py` 判定 | **trot**（FL-FR ≈ 0.5、对角 ≈ 0） |
| 障碍地形的步态 | **不作要求**（允许 bound/lockstep，由专家 1–4 自由演化） |
| `illegal_contact`／速度跟踪 | 不劣于同配方无锚的对照 |

## 6. v5.1（2026-09-28 当晚追加）：锚"最终输出"，不只锚专家

### 6.1 实测暴露的问题（同一条 run 内的时间序列）

| 轮次 | `gate_entropy` | 反解专家 0 权重 w₀ | `gait_airtime_flat`（v5，mode=first） | 同一项（旧 run，mode=all＝5 个专家全是先验） |
|---|---|---|---|---|
| 5 | 0.274 | ≈0.95 | 0.0013（热身，两者都不可信） | 0.0148 |
| **20** | 0.362 | **≈0.93** | **0.0059** | **0.0837** |
| 60 | 0.349 | ≈0.94 | 0.0128 | 0.0886 |
| 120 | 0.388 | ≈0.93 | 0.0147 | 0.0634 |
| 200 | 0.458 | ≈0.91 | 0.0192 | 0.0652 |
| 258 | 0.478 | **≈0.90** | 0.0175 | 0.0677（@250） |

**关键一行是 @20**：门控那时还有 **93% 的权重在专家 0**，而 v5 的平地滞空已经只有 **0.006 s**，
旧 run（混合≙纯先验）是 **0.084 s** ⇒ **差 14 倍**。

结论（两条，都重要）：

1. **只锚"专家"不够**：专家 0 完好（`Loss/anchor_prior` = 0.0074）不等于平地在 trot —— 门控可以绕过它。
2. **随机专家的污染阈值极低**：只要门控给出 **5~7%** 的权重给随机初始化的专家，trot 就被毁掉
   （脚不离地、拖行）。所以 `mode="first"`（4 个随机专家）从一开始就注定守不住平地，
   与门控是否漂移无关。

### 6.2 修法

| # | 改动 | 说明 |
|---|---|---|
| ① | 初始化换成 `--init_experts_mode=all --init_experts_jitter=0.05` | 5 个专家都起自 AMP（专家 0 纯净、其余只抖动"地形/估计器"那 112 列）⇒ 消掉污染悬崖；专家之间靠**怎么用地形信息**分化（这才是 MoE 该分化的地方） |
| ② | 新增 `anchor_target`：`expert` / **`mixture`** / `both` | **`mixture` 锚的是最终混合输出**（"平地上整体必须像 AMP"）。梯度会同时进到**门控**与全部专家 ⇒ 门控自己学会"平地该把权重给谁"；而专家 0 本来就满足教师，网络满足约束最省力的方式就是把平地路由给它，所以**其它专家在 flat 上几乎收不到锚梯度**，分化不受损 |
| ③ | 保留 `expert` 锚（`both`） | 随时留一个纯净的 trot 专家可回放对照 |

实现要点（都已在代码里）：
* 算式抽到 `scripts/rl_lab/rl_lab/utils/anchor.py`（纯 torch ⇒ 可离线单测）：`weighted_mse` / `resolve_targets` / `mix_experts`；
* 混合输出**用 detach 过的 `build_actor_input` 重算**，不用更新循环里的 `mu_batch`（后者的图里带状态/地形**估计器**，会把它们也拉走 ⇒ 与估计器自己的损失打架）；多一次小前向可忽略；
* 非法 `anchor_target` 在算法构造时与 runner 启动时**都**报错（防静默退化）；
* 默认 `expert` 保持向后兼容（旧的 `--anchor_coef` 配方行为不变）。

### 6.3 用这条 run 就能白拿的验收指标

`anchor_target=mixture`（或 `both`）且掩码为 flat 时，**`Loss/anchor_prior` 就是"平地上混合动作与 AMP 的 MSE"**
⇒ 直接当验收线：

| 读数 | 期望 |
|---|---|
| `Loss/anchor_prior` | **< 0.01** 并随训练下降（现在 0.0074 是"专家 0 vs AMP"，换了模式后这个数才有真实含义） |
| `Policy/anchor_weight_mean` | 与 `退火系数 × flat 环境占比` 吻合（本例 0.25→0.08 × 7.5%） |
| `gait_report.py` 在 `flat` | **trot**（FL-FR≈0.5、对角≈0）；`--force_expert 0` 与正常混合的平地步态应接近 |
| `gait_airtime_flat` | 回到先验量级 **0.06~0.09 s**（现在 0.018） |

### 6.4 v5.1 启动命令（替代 §4 的 A 版）

```bash
cd <repo>/imgo2_rl
PRIOR="$(pwd)/logs/amp_rsl_rl/base_move_amp/2026-09-17_21-25-57/model_24500.pt"
LOG="logs/run_v5_1_$(date +%m%d_%H%M).log"

PYTHONUNBUFFERED=1 setsid nohup bash scripts/run_isaaclab.sh scripts/rl_lab/cmoe/train.py \
  --task=Imgo2-basemove-rough-cmoe-gaitfree \
  --num_envs=4096 --headless --max_iterations=2000 --seed 1 \
  --init_experts_from="$PRIOR" --init_experts_mode=all --init_experts_jitter=0.05 \
  --init_experts_critic=false --init_gate_bias=0 --init_gate_bias_margin=4.0 \
  --anchor_target=both --anchor_coef=0.25 --anchor_coef_final=0.08 --anchor_decay_iters=1000 \
  --anchor_terrain_names=flat --anchor_expert=0 \
  --run_name=cmoe_v5_1_outanchor </dev/null > "$LOG" 2>&1 &
```
启动时应多出一行 `…；锚的对象：both`。

**待决定（未实施）**：掩码是否扩到 `random_rough`（AMP 是平面策略，0.01–0.06 m 噪声下"对齐 AMP"的语义存疑）；
退火终值是否降到 0（现在是 0.08 的永久弱约束）。

### 6.5 最终定稿（2026-09-28 用户拍定，已写进配置默认值）

用户原话链："只在 flat 上锚定" → "让最终的输出对齐 amp，在平地上的时候" →
"就只设置，在平地的那几列添加 amp 教师，是针对最终输出的" →
"这样似乎 amp 初始化都不是很需要了，但是还是先初始化一下，加速训练"。

| 项 | 定稿 | 写在哪 |
|---|---|---|
| 掩码 | **只 `flat`**（40 列里的 3~4 列），`random_rough` 暂不纳入 | `anchor_terrain_names: tuple = ("flat",)` |
| 锚的对象 | **最终混合输出**（`mixture`），不是某个专家 | `anchor_target: str = "mixture"` |
| 权重 | **常数 0.2，不退火** | `anchor_coef = 0.2`、`anchor_coef_final = 0.2`、`anchor_decay_iters = 1` |
| AMP 初始化 | **保留，但角色降级为"热启动"**：正确性由锚保证，先验只负责起步快 | `--init_experts_mode=all --init_experts_jitter=0.05` |

**"不退火"的理由**：锚是"平地行为约束"而非课程；flat 列占比全程恒定；且我们的任务项对偷懒步态
几乎免费（`action_rate −0.01`、碰撞 −0.5、无 `lazy_stop`）⇒ 地板太低会重新滑回拖行。

**"初始化不再是必需"的含义**：现在即使不给 `--init_experts_from`，平地也会被锚拉向 AMP 的行为
（只是没有热启动、收敛更慢）；反过来，**没有先验时锚恒为 0**（代码里 `_anchor_weight_now` 同时
守住"系数>0 / 有教师 / 有可用地形列"三个条件，已有测试锁定）⇒ 默认值不会让无先验的 run 出错。

**判据**（`mixture` 模式下 `Loss/anchor_prior` **就是**"平地混合动作 vs AMP 的 MSE"）：
- 稳在 **< 0.01** ⇒ 约束在起作用；训练中爬升 >0.02 ⇒ 把常数提到 0.3；
- 一路 < 0.002 ⇒ 压得过死，可降到 0.15；
- 回放：`gait_report.py` 在 flat 判 **trot**、`gait_airtime_flat` 回到 **0.06~0.09 s**（v5 首跑只有 0.018）。
