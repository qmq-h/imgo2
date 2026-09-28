# CMoE trot 先验移植核对（2026-09-25）

本轮起因：用户 2026-09-25 反馈「感觉还是很难直接学会 trot」，提出
**「考虑新增一个预训练，用之前的 PPO 调整 obs 来操作，然后直接替换一个专家」**。
用户随后选定先验用 **AMP 24500**（本机有完整 checkpoint），并要求先做
「①离线移植校验」。本记录只覆盖**第一步：先验装得进去吗**——不训练、不改训练配置。

## 1. 结论摘要

| 问题 | 结论 | 依据 |
|---|---|---|
| 45 维先验能装进 157 维专家吗 | **能，而且不需要调 obs** | 契约逐项一致（§4）；零填充后**逐位相同**（§5，实测 max\|diff\| = 0） |
| 那 77 维地形怎么办 | **照常喂进去，只是初始权重为 0**（不是"没有这个输入"） | §3；梯度实测流到新增列 |
| 「只替换一个专家」合适吗 | **是最弱的三种做法之一** | 门控随机时先验只占约 1/5；expert 照吃梯度会漂走；`last_action` 变成混合动作后先验输入分布漂移（§6） |
| 推荐写法 | **全专家初始化 + 独立残差 adapter + 掩码 KL/BC 锚** | 全专家初始化时混合**恒等于先验、与门控无关**（§5 已实测），地形通路靠 adapter 从零学，锚只在 45 维上算且 boxes/gap 豁免 |
| 用哪份先验 | **AMP 24500**（用户 2026-09-25 选定） | 本机有完整 checkpoint：`actor 45→12`、`critic 48→1`、`std (12,)`，可继续微调；PPO 只有部署 TorchScript（无 critic、无 std），但步频更接近参考基线 |
| 先验在 Isaac 里能走吗 | **未知，没测** | 本轮零 GPU；入口与指令见 §9，**还没跑** |
| 值得做的依据 | run H（相位核变陡版）1849 轮**仍未 trot**，实测形态最接近「四足锁相」 | §12（离线反演；调核空间有限 ⇒ 直接钉相位更对症） |

## 2. 两份候选先验的实际差别

| | AMP 24500（选定） | PPO 导出件 |
|---|---|---|
| 文件 | `imgo2_rl/logs/amp_rsl_rl/base_move_amp/2026-09-17_21-25-57/model_24500.pt` | `imgo2_deploy/policy/imgo2/ppo/policy.pt` |
| 形态 | 训练 checkpoint：`actor 45→512→256→128→12`、`critic 48→…→1`、`std (12,)`、`iter 24500` | TorchScript：`mlp 45→512→256→128→12`，`obs_normalizer = Identity` |
| 能否微调 | **能**（含优化器以外的全部网络参数） | 不能（只有推理图） |
| 步态实测 | FL-FR **+174.2°**、集中度 R 0.86–0.91 的干净 trot；**步频 5.2 Hz 偏高**（[足端回放记录](gait_eval_amp_24500_2026-09-18.md)） | 周期强度 0.95、FL-FR **+175°**、步频接近参考 1.67 Hz；`vx=0.5` 实测 0.428 m/s（**低 14%**） |
| 代价 | 训练环境是 `terrain_type=plane`、`height_scanner=None` ⇒ 先验**天然不含地形** | 6 月外部仓库训出、无训练日志（[来源复核](ppo_policy_provenance_2026-09-19.md)） |

## 3. 关键事实：专家确实有 77 维地形，零填充 ≠ 把这个输入拿掉

`CMoEExpertActorCritic` 的输入是 157 维，切片如下（`modules/cmoe_actor_critic.py::actor_input_dim`）：

| 切片 | 内容 | 维数 |
|---|---|---|
| `[0:45]` | 当前帧本体感知（ang_vel, gravity, commands, dof_pos, dof_vel, last_action） | 45 |
| `[45:48]` | `explicit`（状态估计器输出） | 3 |
| `[48:64]` | `state_latent` | 16 |
| **`[64:141]`** | **77 维地形扫描** | 77 |
| `[141:157]` | `terrain_latent` | 16 |

移植只改第一层权重矩阵的形状：`(512,45) → (512,157)`，**新增的 112 列权重置 0**。含义是
「通路接进来了、初始权重为零」：

* 地形**照常喂进去**，只是初始化时对输出没有贡献（先验本来是平地策略，不可能"用过"地形）；
* 梯度不会被零权重挡住（`∂L/∂w = ∂L/∂y · x`，地形输入非零就有梯度）⇒ 地形通路从零学；
* 本轮实测：把后 112 维换成**随机非零**输入，专家输出仍与先验**逐位相同** ⇒ 确实是"零权重"而非"丢弃输入"；
  反向传播后 `max|grad[:, 45:]| = 7.1`（AMP）/ `3.7`（PPO）⇒ 通路是通的。

三种写法（供下一步选）：

| 写法 | 初始行为 | 地形通路 | 评价 |
|---|---|---|---|
| (a) 纯零填充 | 精确等于先验 | 训练从零长 | 最忠实，地形起步慢 |
| **(b) 残差 adapter**：`expert_out = trunk(45) + adapter(112)`，adapter 末层零初始化 | 精确等于先验 | 有独立容量、不扰动主干 | **推荐**；"调 obs"就落在这里（例如只喂前沿 21 条射线、逐射线归一化） |
| (c) 新增 112 列随机初始化 | 立刻地形敏感，但输出 ≠ 先验 | 立刻可用 | 不适合"精确复现先验" |

## 4. 契约核对（本轮从**在跑 run 的 `params/env.yaml`**＋源码＋部署 yaml 读出）

真值来源：`logs/cmoe/base_move_cmoe_rough/2026-09-24_22-08-42_cmoe_H_sharpkernel/params/env.yaml`
（`resolution=0.1`、`size=(1.0,0.6)` ⇒ 11×7=**77** 条射线）＋ 同目录 `agent.yaml`
（`history_steps=10`、`num_experts=5`、`explicit_dim=3`、`state_latent_dim=16`、`terrain_latent_dim=16`）。

| 项 | 训练侧（`rough_env_cfg.py` / `assets/imgo2.py` / 在跑 run 的 env.yaml） | 部署 `amp/config.yaml` | 部署 `ppo/config.yaml` |
|---|---|---|---|
| policy obs 顺序 | ang_vel, gravity, commands, dof_pos, dof_vel, actions | 同 | 同 |
| policy 组尺度 | ang_vel **0.25**、joint_vel **0.05**、其余 1.0 | 0.25 / 0.05 / 1.0 | 0.25 / 0.05 / 1.0 |
| policy 组不含 | `base_lin_vel`、`height_scan`（显式 `null`） | —— | —— |
| 动作 | 髋 0.125、其余 0.25，clip ±3，`use_default_offset=True` | 同 | 同 |
| 默认姿态 / 增益 | 0 / 0.87 / −1.82，kp 25、kd 0.5 | 同 | 同 |
| 关节顺序 | `FL,FR,RL,RR × hip,thigh,shank` | `joint_mapping` 恒等（12） | 同 |
| critic 组 | `base_lin_vel 3 + 45 本体 + 77 地形 = 125`（全 scale 1.0） | AMP critic 48 维 = 前 48 维（`amp_env_cfg.py` 置 `critic.height_scan = None`） | 不导出 critic |

派生契约（全部由上面这些数算出来，不写死）：`one_step=45`、`terrain=77`、`critic=125`、
`expert/gate=157`、`actor=527`。⇒ **前 45 维在顺序、尺度、动作定义上逐项一致，移植不需要改观测。**

## 5. 数值核对结果（本机实测，纯 torch、无 GPU）

用**真实的** `CMoEActorCritic` / `CMoEExpertActorCritic` 建模型后装入先验：

| 检查 | AMP 24500 | PPO `policy.pt` |
|---|---|---|
| expert 输出 == 先验输出（真实估计器/地形尾输入） | **max\|diff\| = 0.0** | **0.0** |
| 后 112 维填随机非零后仍不变 | **0.0** | **0.0** |
| 新增 112 列权重 | `max\|w[:,45:]\| = 0.0` | `0.0` |
| 新增列梯度（通路是否接上） | `max\|grad[:,45:]\| = 7.1e0` | `3.7e0` |
| critic 零填充等价（48 → 125） | **0.0** | 无 critic |
| 噪声 `std` | 拷入后均值 **0.334**（默认 `init_noise_std=1.0`） | 无 std |
| **5 个专家同一先验 ⇒ 混合恒等于先验、与门控无关** | 两次随机门控 max\|diff\| ≤ **4.8e-7** | ≤ 4.8e-7 |

最后一项的意义：全专家初始化时，**混合输出与门控权重无关**（5 个相同专家的加权和；残差 1e-7 来自
float32 softmax 权重和 = 1 ± 1e-7）。⇒ 「第 0 步就是先验步态」是可证的，不需要给门控加偏置。
**反面对照**：`--mode first`（只装第 0 个专家）时混合与先验差 > 1e-3（测试里锁死），说明上面的等价性
确实来自"全专家初始化"而不是别的原因。

## 6. 为什么「只替换一个专家」是最弱的写法

1. **占比**：门控随机初始化时该专家权重约 1/5，其余 4 个随机专家把 trot 糊掉；想救要额外给门控加偏置。
2. **会漂走**：expert 照常吃 PPO 梯度，几百轮内先验就没了。而 `README` §29.16/§29.19/§29.21 的结论是
   **奖励本身偏好 bound/pronk** ⇒ 没有锚它就会漂回 bound。
3. **输入分布漂移**：expert 看到的 `actions`（以及历史 9 帧）都是**混合策略**的输出；门控一动，
   被锚定专家的输入分布就偏，"先验"很快名不副实。全专家初始化在早期没有这个问题（混合 ≡ 先验）。
4. **它不能替代奖励侧修复**：run H 正在跑的 `e6a1155`（相位核 `std 0.7071→0.2`、`max_err 0.2→0.5`，
   溢价 2.7×）与本方案互补，不是替代关系。

## 7. 本轮新增文件与复现

| 文件 | 作用 |
|---|---|
| [pretrained_prior.py](../imgo2_rl/scripts/rl_lab/rl_lab/utils/pretrained_prior.py) | 只依赖 torch 的移植函数：`load_prior`（AMP checkpoint / TorchScript）、`zero_pad_first_layer`、`install_prior(actor_critic, prior, mode="all"\|"first")`、`teacher_actor/critic`，以及 `prior_actions`（step-0 的"取前 45 维 + 裁 ±3"）。将来训练期初始化钩子直接复用它 |
| [check_cmoe_expert_init.py](../imgo2_rl/scripts/tools/check_cmoe_expert_init.py) | 离线核对工具：契约层（纯标准库）+ 数值层（torch）。数值一律从来源文件读取，不复制预期值 |
| [test_cmoe_expert_init.py](../imgo2_rl/tests/test_cmoe_expert_init.py) | **14 项**离线测试（解析层 4 + 真实 run 契约 1 + 移植等价 6 + `prior_actions` 3） |
| [cmoe/play.py](../imgo2_rl/scripts/rl_lab/cmoe/play.py) | 新增 `--prior`／`--prior_clip`／`--steps`／`--prior_report_every`（§9）；**默认行为不变**，只有 `--prior` 时跳过 CMoE runner 与 `policy.pt` 导出 |

```bash
cd imgo2_rl
# 契约层 + 数值层（用有 torch 的解释器）
python scripts/tools/check_cmoe_expert_init.py                 # 69/69，退出码 0
python scripts/tools/check_cmoe_expert_init.py --skip-torch    # 50/50：纯契约层，不需要 torch
python -m unittest discover -s tests -p "test_cmoe_expert_init.py" -v   # 14 项
```

## 8. 验证记录（做了什么、判据、结果）

* **工具**：`--skip-torch` 只跑契约层 → **50/50 通过、退出码 0**；完整跑 →
  **69/69 通过、退出码 0**；`--mode first` → 68/68。
* **torch 缺失时 fail-closed**：用假 `torch.py`（`PYTHONPATH` 前置一个 `raise ModuleNotFoundError`）模拟无 torch 环境 →
  **50/51、`❌ torch 不可用`、退出码 1**（不会静默降级成"通过"）；同一模拟下 `--skip-torch` 仍 50/50 退出 0。
* **负向对照（证明检查不是空转）**：拿**旧 187 维** run 的
  `logs/cmoe/base_move_cmoe_rough/2026-09-23_22-18-33_cmoe_obstacle_4096_seed1/params/env.yaml`
  当真值 → 恰好 4 项 ❌（射线数 ≠77、critic 维数、expert 维数、actor 维数），退出码 1。
* **测试**：新文件 **14 项通过**（含 `prior_actions` 的 3 项）；与 `test_cmoe*.py` 同进程跑
  **20 项通过/3 跳过**（确认私有包名导入没有污染真实 `rl_lab.*`）；`unittest discover` 全仓
  **371 通过 / 8 跳过**。
* **其它**：`py_compile`、`check_asset_paths.py`（PASS）、`git diff --check` 全部通过；
  `git ls-files -i -c --exclude-standard` 为空。
* **环境边界**：本机 torch 来自 `/opt/conda/envs/isaaclab/bin/python`（`torch 2.7.0+cu128`），而 PATH 上的
  `python3` **就是**这个解释器（`which python3` → `/opt/conda/envs/isaaclab/bin/python3`）⇒ 本机
  **不能靠"`python3` 能不能 import torch"判断解释器**，缺 torch 的情形用上面的假 `torch.py` 验证。
  工具用**私有包名**（`_rl_lab_offline`）导入 `rl_lab` 子模块，绕开
  `rl_lab/modules/__init__.py → actor_critic_recurrent → rl_lab.utils → export_deploy_cfg → isaaclab
  → omni.log` 这条"只有 Isaac Sim 启动后才存在"的链 —— 否则离线脚本在裸解释器里根本 import 不进来
  （这也是既有 `test_cmoe_stack.py` 在本机会被 skip 的原因）。

## 9. step-0 教师回放入口（已接线，**未运行**）

入口复用现成的 `cmoe/play.py`（而不是另写脚本 —— 它的环境构造路径在训练机上已经跑通过），
新增三个开关，**默认行为不变**：

| 开关 | 作用 |
|---|---|
| `--prior <权重>` | 不加载 CMoE checkpoint，改把一份 **45 维先验**（AMP 训练 checkpoint 或 play.py 导出的 TorchScript）当固定策略：只喂 CMoE 观测的**前 45 维（当前帧本体感知）**，动作裁 ±`--prior_clip`（默认 3）。⚠️ 此时**不导出** `policy.pt`（先验不是 CMoE 网络） |
| `--steps N` | 跑满 N 步自动退出并打印汇总（否则只能关窗口／`--video_length` 结束） |
| `--prior_report_every N` | 每 N 步打印一次**逐地形列终止次数**（`--prior` 时默认开启本统计） |

指令（在训练机、有 GPU 的那份仓库里跑；`cd <repo>/imgo2_rl`）：

```bash
# ⓪ 前置（不需要 GPU）：确认契约与零填充等价性
bash scripts/run_isaaclab.sh scripts/tools/check_cmoe_expert_init.py

# ① step-0：把 AMP 24500 当固定策略塞进 CMoE 环境；10 列各 1 个环境，跑 3000 步后自停并打印逐列终止次数
bash scripts/run_isaaclab.sh scripts/rl_lab/cmoe/play.py \
  --task=Imgo2-basemove-rough-cmoe-play --num_envs=10 --headless \
  --steps 3000 --prior_report_every 500 \
  --prior="$(pwd)/logs/amp_rsl_rl/base_move_amp/2026-09-17_21-25-57/model_24500.pt"

# ② 要看画面（录像）：产物在 logs/cmoe/base_move_cmoe_rough/prior_play/videos/play/*.mp4
... --video --video_length 600

# ③ 或直播（客户端连 <主机IP>:49100；WebRTC 媒体是 UDP，只转发 TCP 会有信令没画面）
... --livestream 2

# ④ 对照：PPO 那份导出件也能当先验（它没有 critic/std，步频更接近参考基线）
... --prior="$(pwd)/../imgo2_deploy/policy/imgo2/ppo/policy.pt"

# ⑤ 顺便看现成 CMoE checkpoint（第一步那条指令；此路径会导出 exported/policy.pt）
bash scripts/run_isaaclab.sh scripts/rl_lab/cmoe/play.py \
  --task=Imgo2-basemove-rough-cmoe-play --num_envs=1 --headless \
  --checkpoint="$(pwd)/../logs/cmoe/base_move_cmoe_rough/2026-09-24_22-08-42_cmoe_H_sharpkernel/model_1500.pt"
```

### 9.1 首跑结果（**用户报告** 2026-09-25，细节待补）

用户报告：**在训练机上跑 AMP 先验，策略运行正常**（无报错）。这一条把 §4 的契约核对从
"源码级推断"升级为"实跑通过"——先验与 CMoE 的动作语义（scale/clip、默认姿态、kp/kd、关节顺序）
在真环境里确实对得上，也排除了最坏的分支 B（先验与 CMoE 动力学/控制频率不匹配 ⇒ 连平地都站不住）。

**但还缺三样东西才能定锚的地形掩码**（缺了就只能按最保守的分支 A 准备）：

1. `[REPORT] step=... 各列终止次数: …` 那一行 —— 到底哪几列掉、掉多少次（`--num_envs=10` 时一列一个环境）；
2. `--terrain_level=9` 的同一跑（最难几何：台阶 14.5 cm、boxes 28.9 cm）——play 默认只从 0–5 级起步，
   单环境几乎不晋级，所以"看着正常"很可能只是**0–5 级的温和地形**；
3. 步态形态（相位/周期/占空比）：肉眼只能看个大概，要量就得给 `eval_gait.py` 补 CMoE 适配。

⇒ 在补齐之前，按"分支 A 与 C 之间"准备：**锚默认只作用在"先验能活"的列**，adapter 末层用**小随机**
而不是全零（保留地形响应），等逐列读数回来再把掩码放大到全地形。

读法：`[REPORT] step=... 各列终止次数: 列0=.. 列9=..`。预期**列 6–8（`gap`）与列 3（`boxes`）先出问题**——
先验的地形通路权重为 0，它"看不见"沟，会按平地 trot 直接走进去；列 9（`flat`）／列 4（`rough`）应当能走。
若列 0–2（楼梯）也很快终止，说明这份平地先验连缓坡都接不住，锚就该按更窄的地形掩码给。

## 10. 未做 / 待确认（缺什么才能完成）

1. **step-0 教师回放已接线但没跑过**：入口＝`cmoe/play.py --prior=<先验>`（§9）。
   **本机起不了 Isaac Sim**（容器 `NVIDIA_VISIBLE_DEVICES=void`、没有 `/dev/nvidia*` 设备节点，
   `torch.cuda.is_available()=False`）⇒ **未运行**；只有 `py_compile` 与 `prior_actions` 的离线测试
   （3 项：取当前帧、裁剪等价于部署语义、过短观测要报错）。**缺什么**：训练机跑一次。
   定量步态（FL-FR 相位／步周期／抬脚高度）仍要 `eval_gait.py` 的 CMoE 适配。
2. **训练期钩子没实现**：`--init_experts`、残差 adapter、掩码 KL/BC 锚与衰减、门控处理、日志都还没写；
   `pretrained_prior.install_prior()` 已可直接复用。**缺什么**：用户对写法(a)/(b)/(c)与锚系数的决定。
3. **地形盲的代价没量化**：5 个专家全零填充 ⇒ 初始策略完全地形盲，早期可能在台阶/沟壑上摔。
   adapter 末层从"全零"改成"小随机"时，初始输出与先验的偏差**可以离线算**（成本极低），但本轮没算。
4. **"同一动作数值 ⇒ 同一关节目标"仍属源码级推断**：动作 scale/clip、默认姿态、kp/kd、关节顺序已逐项
   对齐（§4），但没有在 Isaac 里实跑过；step-0 回放会一并验证。
5. **run H 尚未入档**：本轮只把
   `2026-09-24_22-08-42_cmoe_H_sharpkernel/params/env.yaml` 当作契约真值，**它的训练曲线与逐列步态指标
   （`gait_trot_*`/`gait_bounce_*`/`tracking_*`）没人看过**；README 维护记录里也还没有 run H 的行。
6. **换机器的先验路径**：默认先验发现依赖 `docs/gait_eval_amp_24500.json` 里记录的绝对路径与本机 `logs/`；
   在别的机器上跑要用 `--prior` 显式指定。

## 11. 下一步的判据

* **step-0**：给出先验在 8 类地形列的存活时长／跟速核／FL-FR 相位（含"地形盲策略在 gap 列会直接走进去"的
  定量证据）⇒ 据此决定锚的地形掩码与 adapter 的处理。
* **训练期**（若采纳初始化+锚）：1500–2000 轮内 `gait_trot_*` 保持领先、`level_*`／`tracking_*` 不退化，
  且 `Policy/expert_*_mean_weight` 显示被初始化专家确实被门控使用。

## 12. run H（1849 轮）读数：相位核变陡**没有**把步态推向 trot

零 GPU，只读盘：`logs/cmoe/base_move_cmoe_rough/2026-09-24_22-08-42_cmoe_H_sharpkernel`
（`e6a1155` 那一版：`std` 0.7071→**0.2**、`max_err` 0.2→**0.5**，四处同参）。

### 12.1 三态步态读数（三个 1e-6 记录项，无地形掩码）

| 迭代 | trot | bound | pace | trot−bound |
|---|---|---|---|---|
| 250 | 0.7392 | 0.7621 | 0.7458 | −0.023 |
| 1000 | 0.7708 | 0.7902 | 0.7728 | −0.019 |
| 1849 | **0.7684** | **0.7820** | **0.7647** | **−0.014** |

逐列 @1849（trot/bound）：`flat 0.8475/0.8567`、`random_rough 0.8403/0.8501`、`gap 0.7254/0.7443`、
`boxes 0.7112/0.7294`。`Episode_Reward/feet_gait` 0.38 → 0.42/s 几乎不动；
`mean_reward` 49.6、`ep_len` 823/1000、`gate_entropy` 1.40。

### 12.2 为什么"三态接近相等"等于"不是 trot"（离线反演，非猜测）

新增 [gait_kernel_probe.py](../imgo2_rl/scripts/tools/gait_kernel_probe.py)：核公式逐字取自
`GaitReward._sync_reward_func` / `_async_reward_func`，`std`/`max_err` 从 `CMoE_env_cfg.py` 读
（并检查三处 `gait_metric_*` ＋ `feet_gait` 的 post_init **四处一致**）。

```
python scripts/tools/gait_kernel_probe.py --period 0.34 --observed 0.768,0.782,0.765

步态            trot   bound    pace   形状判定
trot         1.000   0.676   0.676   领先 trot +0.324
bound        0.676   1.000   0.676   领先 bound +0.324
pace         0.676   0.676   1.000   领先 pace +0.324
lockstep     0.676   0.676   0.676   三态相等（不可分辨）
⇒ 实测最接近：lockstep（形状距离 0.0166；trot/bound 0.34/0.32，差 20 倍）
```

**机制**：`_async_reward_func` 比较的是"一只脚的**滞空时间** vs 另一只脚的**触地时间**"。
当 duty≈0.5（滞空≈触地）时，这四个"反相核"对**任何**配对都近似满足 ⇒ 只剩 2 个同步核在分辨，
所以四种步态里三种配对读数会趋同，**"四足锁相"与"干净 trot"的差别被压到 0.32/s**（T=0.93 s 时 0.72/s）。
`gait_airtime_mean` 0.169 s（高步频小腾空）与 duty≈0.5 一致。⇒ **继续调核参数的空间有限**
（G→H 的 2.7× 变陡换来的是 trot−bound 从 −0.010 到 −0.014，等于没动），
把相位**直接钉住**（先验＋掩码 BC 锚）比继续调核更对症。

### 12.3 同一读数里的另外两处红旗

* `level_boxes` **1.23**（`gap` 已 6.49、`level_mean` 5.71）⇒ `boxes` 列几乎没在晋级；
* `Episode_Termination/illegal_contact` **0.32**（README §29.13 的老配方是 0.0356）、
  `tracking_mean` 0.755（低于课程 0.80 晋级门）。

这两条与步态无关，但说明"这一轮不只是步态问题"，做先验实验时要**分开记账**。

### 12.4 边界

周期/占空比用的是反推值（空时 0.169 s、duty≈0.5），不是实测；要钉死相位/周期必须给
`eval_gait.py` 补 CMoE 适配（它现在硬编码 amp wrapper/runner）。本轮**没有运行任何仿真**。

## 13. 新方向（用户 2026-09-25 决定）：从 AMP 策略起步 ＋ 手工步态 shaping 归零

用户原话：「考虑直接从 amp 那个策略恢复，然后之前步态相关权重至 0。刚刚跑的就是 cmoe 那条。
**死掉是正常的，本就就要后训练调整**」。即：**步态由先验提供，训练只负责地形与跟速**；
step-0 里先验在难地形上摔下去是预期内的事，不该因此否决这条路（§9.1 已确认 step-0 跑通）。

### 13.1 落地了什么（全部离线验证，**训练路径未跑**）

| 改动 | 位置 | 默认行为 |
|---|---|---|
| 新任务 `Imgo2-basemove-rough-cmoe-gaitfree`：**五项手工步态 shaping 归零**（`joint_mirror`／`feet_air_time`／`feet_height_body`／`feet_air_time_variance`／`feet_gait`），其余奖励一项不动；归零后**再跑一次** `disable_zero_weight_rewards()`（否则它们会以 0 权重留在管理器里）| `CMoE_env_cfg.py::Imgo2CMoEGaitFreeEnvCfg`、`base_move/__init__.py` | 新任务，现有 `Imgo2-basemove-rough-cmoe` 配方**一字未改**（留作对照） |
| 从先验初始化：`--init_experts_from=<AMP ckpt 或 TorchScript>` ＋ `--init_experts_mode=all\|first`；零填充装进专家（+critic+std），并在 `learn()` 开头打印"混合 vs 先验"的动作 RMSE 自检 | `cmoe/train.py`、`runners/cmoe_on_policy_runner.py`、`config/cmoe_algorithm_cfg.py` | `init_experts_from=None` ⇒ **默认不装**，现有 run 行为不变 |
| 漂移度量 `Policy/prior_action_rmse`（每轮记录，**不裁剪**）＋ 纯 torch 函数 `prior_action_rmse()` | `utils/pretrained_prior.py`、runner `log()` | 只在装了先验时写 |
| 审计链 `cmoe-gaitfree`（生效 **27 → 22 项**；五项归零会**被报出来**，不是静默）＋ 测试 | `scripts/tools/check_reward_overrides.py`、`tests/test_check_reward_overrides.py`、`tests/test_cmoe_expert_init.py` | 新增链，`cmoe` 链不变 |

**刻意保留、不要跟着归零**：三个 `gait_metric_{trot,bound,pace}`（步态分类器）＋ 三个
`diag_*`（时序/弹跳诊断）——权重 0 会被 `disable_zero_weight_rewards()` 整个移除，就再也看不见
"先验漂没漂"了；`lin_vel_z_l2`（`MaskedLinVelZ`，−2）也留着，它是**物理平滑惩罚**而不是步态形状先验
（要连它一起去掉，`Imgo2CMoEGaitFreeEnvCfg` 里有一行注释掉的赋值）。

### 13.2 训练机指令

```bash
cd <repo>/imgo2_rl
PRIOR="$(pwd)/logs/amp_rsl_rl/base_move_amp/2026-09-17_21-25-57/model_24500.pt"

# ⓪ 冒烟（1–2 分钟）：先确认构造、先验装入与自检打印都对
bash scripts/run_isaaclab.sh scripts/rl_lab/cmoe/train.py \
  --task=Imgo2-basemove-rough-cmoe-gaitfree --num_envs=4 --max_iterations=2 --headless \
  --init_experts_from="$PRIOR" --run_name=cmoe_gaitfree_smoke

# ① 正式（4096 env，先 2000 轮 ≈2.3–2.6 h）
bash scripts/run_isaaclab.sh scripts/rl_lab/cmoe/train.py \
  --task=Imgo2-basemove-rough-cmoe-gaitfree --num_envs=4096 --headless \
  --max_iterations=2000 --seed 1 \
  --init_experts_from="$PRIOR" --run_name=cmoe_gaitfree_amp24500
```

### 13.3 判据（按重要性）

1. **启动三行打印**：`[INFO] 先验初始化：source=amp-checkpoint, actor 45->12, critic 48->1, std=yes；
   已装进 5/5 个专家（mode=all），actor 第一层 (512, 157)（置零列 112）`；
   `[INFO]   噪声 std 随先验装入：均值 0.3340`；
   `[INFO] 先验初始化自检：混合策略 vs 先验 动作 RMSE = ~0`（≈0 才是"装对了"的证明：全专家同一先验
   ⇒ 混合恒等于先验、与门控无关，见 §5）。
2. `Episode/Curriculum/terrain_levels/gait_trot_flat` 应接近 **1.0**（先验是干净 trot；run H 的旧配方
   在同一指标上是 0.8475）——这是"步态有没有被保住"最直接的读数。
3. `Policy/prior_action_rmse` 的**上升速度**决定要不要加锚：若 2000 轮内升到 >0.5 且 `gait_trot_*`
   掉到 `gait_bound_*`/`gait_pace_*` 之下 ⇒ 先验被改写、必须加**掩码 BC/KL 锚**（备份方案，仍未实现）；
   若升得慢且 `gait_trot_*` 保持领先 ⇒ 先验自己扛得住，可以往下加长训练。
4. 地形/跟速不能退化：`level_mean`、`level_gap`、`tracking_mean` 不低于同轮数的 run H；`illegal_contact`
   终止占比（run H 1849 轮是 **0.32**）应下降。

### 13.4 边界与风险

* **验证范围**：本容器无 GPU，agent 侧只有 `py_compile`、**386 项离线测试**、审计工具与
  "cfg 默认值＝关"的测试；**训练路径一次都没跑过** ⇒ ⓪ 冒烟必须先做。
* **五项归零后没有任何手工项约束 trot**：先验是唯一来源。这也是为什么同时加了
  `prior_action_rmse`（量化漂移）——没有它就只能靠肉眼判断"步态是不是变回去了"。
* 回放判读仍用现有 `Imgo2-basemove-rough-cmoe-play`（奖励不参与回放行为），
  或 `play.py --prior=<先验>` 对照"先验原样"与"训练后"。
* `boxes` 列（`level_boxes` 1.23）与 `illegal_contact` 0.32 是这一轮里与步态无关的两个已知短板，
  换配方不会自动修好，要单独盯。

## 14. 实跑首查（2026-09-25 14:22，`cmoe_gaitfree_amp24500`）：**奖励侧生效，但先验没装**

第一次真跑发生在 `imgo2_rl/logs/cmoe/base_move_cmoe_rough/2026-09-25_14-22-41_cmoe_gaitfree_amp24500`
（注意落在 **`imgo2_rl/` 下的 `logs/`**，因为命令是从 `imgo2_rl/` 目录起的；repo 根另有一份 `logs/`）。
14:31–14:32 仍在写 tfevents、约 **4.3 s/轮**（与 run H 同量级）。

### 14.1 两个确认

| 项 | 实测 | 结论 |
|---|---|---|
| 奖励侧 | 该 run 自带 `params/env.yaml` 的 `rewards` 组**恰 22 项**：五项手工步态项（`joint_mirror`／`feet_air_time`／`feet_height_body`／`feet_air_time_variance`／`feet_gait`）**不在**，三个 `gait_metric_*`＋三个 `diag_*`＋`lin_vel_z_l2`（−2）／`base_height_l2`（−10）等在 | **§13 的离线预测（27→22 项）在真 run 里成立** ✅ |
| 先验初始化 | `params/agent.yaml` 里 **`init_experts_from: ''`**（空串），且 tfevents 里 **没有** `Policy/prior_action_rmse` | **先验一次都没装** ❌ —— 这是"**既无手工 shaping、也无先验**"的**零约束对照**，不是 §13 的目标实验 |

**根因**：命令里 `--init_experts_from` 传成了空串，落到 cfg 是 `''`，而 runner 里 `if init_from:` 判假
⇒ 静默跳过。训练照跑、只少一个标量，很难发现。

### 14.2 已修（本轮）

* `utils/pretrained_prior.py::normalize_prior_path()`：`None`／空串／纯空白 ⇒ 视为没给；
* `cmoe/train.py`：`--init_experts_from` 传空串直接 `parser.error`；任务名带 `gaitfree` 又没给先验时
  打印 `[WARN] …步态目前没有任何约束…`；
* runner：cfg 里是空串时打印 `[WARN] …若本意是「从先验起步」，请给真实路径…`；
* 测试：全仓离线 **387 通过／8 跳过**（新增 `normalize_prior_path` 一例）。

### 14.3 这个 run 能回答什么、不能回答什么

* **能**：手工步态项全去掉、无先验时，策略会演化成什么样（纯对照）。
* **不能**：先验能否扛住训练 —— 那条必须重跑并带上 `--init_experts_from`（**构造期初始化，中途补不了**）。

@111 轮的早期读数（**太早，不能据此下结论**）：`mean_reward` 16.2、`ep_len` 503/1000、
`illegal_contact` 0.4545、`level_mean` 0.265、`gait_trot_flat` 0.7911、`gait_bounce_mean` 0.1099、
三态 `trot 0.7254 / bound 0.7436 / pace 0.7263`（仍近相等、bound 略高，与 run H 同型）。

## 15. 实跑现状（2026-09-25 14:43）＋ 修掉一个度量缺陷 ＋ 系数怎么算

### 15.1 两条 run 的对比：**"有先验"这条目前反而更差**

| run | 起跑 | 先验 | 状态 |
|---|---|---|---|
| `2026-09-25_14-22-41_cmoe_gaitfree_amp24500` | 14:22 | ❌ 没装（空串） | 14:34 停 ⇒ **无先验对照** |
| `2026-09-25_14-35-18_cmoe_gaitfree_amp24500_init` | 14:35 | ✅ 装了（`agent.yaml` 路径正确、有 `Policy/prior_action_rmse` 标签、`mean_noise_std` 0.335≈先验的 0.334） | 在跑 |

| 指标（同轮数附近） | 有先验 @97 | 无先验 @100 |
|---|---|---|
| `Episode_Reward/track_world_vel_xy_exp` | **0.58** | 1.77 |
| `Episode_Termination/illegal_contact` | **0.72** | 0.45 |
| `Train/mean_reward` | **−3.0** | ≈ +16 |
| `Train/mean_episode_length` | 370 | ≈ 500 |
| `level_mean` | 0.085 | 0.27 |
| `Policy/gate_entropy` | 0.737 | 1.288 |
| `Policy/expert_2_mean_weight` | 0.34（0/1 号各 0.05） | — |

**步态**：只有平坦/粗糙两列 trot 略高于 bound（`flat 0.627 vs 0.610`、`random_rough 0.620 vs 0.618`），
`gap 0.453 vs 0.449`、`boxes 0.421 vs 0.441`、**全地形均值 `trot 0.4929 vs bound 0.4964`**
⇒ **还谈不上"trot 占主导"**，而且绝对值离"干净 trot"（该 duty 下 trot 应≈1.0）很远。

**首要嫌疑**：先验的 **critic 被一起搬了过来** —— 它是在 AMP 平地的另一套奖励尺度下训的，
早期 advantage 尺度错配，足以把刚装好的 actor 迅速改写。次因：先验原始输出在 OOD 观测下极大
（实测 `|obs|~0.5` 时 `|a|max≈17`、`|obs|~10` 时 `≈193`），且 AMP 是 5.2 Hz 的平地 trot。

### 15.2 修掉度量缺陷：`prior_action_rmse` 原来量的是"摔得多凶"

它是**当轮观测**上的**绝对**动作 RMSE ⇒ 随观测幅度线性放大；同一份权重实测：

| 观测尺度 | 已训 1 轮 | 刚装好（应为 0） |
|---|---|---|
| \|obs\|~0.5 | 0.66 | 1.8e-07 |
| \|obs\|~5 | 8.03 | 1.4e-06 |
| \|obs\|~10 | **15.55** | 2.9e-06 |
| \|obs\|~20 | 29.93 | 5.6e-06 |

⇒ run 里那个 ≈17–18 读的是"**机器人摔得有多凶**"，不是"先验漂了多少"。
**离线取证确认安装无误**：该 run 的 `model_0.pt`（＝第 0 轮**更新后**落盘：第一层前 45 列与 AMP actor
差 0.0042、尾部 0.0041，即"装好后只动了 0.004"）装回模型后，与"刚装好"的差距只有 0.004 量级。

**修法（已落地）**：
1. `prior_action_rmse` 改用**第一轮抓下的固定 1024 条参考观测**（后续每轮在同一批输入上重算 ⇒ 只反映策略漂移）；
2. 新增 `Policy/prior_weight_rel_drift`＝专家 actor 参数相对"刚装好那一刻"的**相对 L2 漂移**；
3. 新增 `Policy/prior_new_col_mass`＝专家第一层权重落在**地形/估计器那 112 列**上的份额（装好时＝0，上升＝地形通路在被打开）。
4. 新增 `--init_experts_critic=true|false`（默认 true）——下一轮建议试 **false**（只搬 actor）。

### 15.3 三个步态系数到底怎么算（完整版）

`gait_metric_{trot,bound,pace}` 是**同一个** `GaitReward`，只换 `synced_feet_pair_names`：
trot＝对角 `(FL,RR)(FR,RL)`、bound＝前后 `(FL,FR)(RL,RR)`、pace＝同侧 `(FL,RL)(FR,RR)`；
权重 **1e-6**（纯记录：权重 0 会被 `disable_zero_weight_rewards()` 整个移除，就再也看不见了）。

每次调用（核公式逐字取自 `mdp/rewards.py::GaitReward`，`std=0.2`、`max_err=0.5`）：

```
air[f] = contact_sensor.data.current_air_time[f]      # 当前这段腾空已持续多久
con[f] = contact_sensor.data.current_contact_time[f]  # 当前这段触地已持续多久

sync(f0,f1)  = exp( −[ clip((air0−air1)², max=max_err²) + clip((con0−con1)², max=max_err²) ] / std )
async(f0,f1) = exp( −[ clip((air0−con1)², max=max_err²) + clip((con0−air1)², max=max_err²) ] / std )

value = sync(声明对1) · sync(声明对2) · async(4 个跨对)        # 6 核乘积 ∈ [3e-7, 1]
value ×= 1[ ‖cmd‖>0.1 或 ‖v_xy‖>0.5 ]                          # 站着/没命令时记 0
value ×= clamp(−g_z, 0, 0.7) / 0.7                             # 越倾斜越接近 0（倒地时不记分）
```

日志里的逐列读数＝**刚结束那一回合的时间平均**，权重被除掉：
`_episode_sums[term] / weight / (回合步数 · step_dt)`（见 `mdp/curriculums.py::_episode_term_average`）
⇒ 读数就是**核乘积本身**（0–1），与 1e-6 无关。

**怎么读**（`gait_kernel_probe.py` 的离线对照）：

| 真实步态 | trot | bound | pace |
|---|---|---|---|
| 干净 trot | **1.000** | 0.676（T=0.34 s）／0.280（T=0.93 s） | 同上 |
| 干净 bound | 0.676／0.280 | **1.000** | 同上 |
| 四足锁相（pronk 型） | 0.676 | 0.676 | 0.676（**三者相等**） |

⇒ 判读看**相对次序与领先量**：`trot` 明显最高且接近 1 ＝ 真 trot；三者几乎相等 ＝ 锁相/乱走
（run H 的 0.768/0.782/0.765 就是后者）。**绝对值会被"摔倒稀释"**（倒地时直立门＝0），
所以摔率不同的两条 run 不能直接比绝对值。

## 16. 实测基准修正：**先验自己的三态读数就是"接近"的**（用户质疑成立一半）

用户 2026-09-25 质疑：「在有先验的情况下，3 个差不多反而是接近先验」。这个假设**可验证** ——
`logs/gait_24500_vx{1.0,0.6}.npz` 里存着 AMP 24500 策略**自己**的逐步四足接触序列（`contact`
1000×8×4 @0.02 s）。把它过一遍同一套核（新增 `gait_kernel_probe.py --npz`）：

| | trot | bound | pace | **trot−bound** | 六对 \|Δair\|+\|Δcon\| | 完整腾空时长 |
|---|---|---|---|---|---|---|
| **AMP 先验 @vx=1.0** | **0.947** | 0.831 | 0.828 | **+0.116** | 0.0854 s | 0.0868 s |
| **AMP 先验 @vx=0.6** | **0.841** | 0.747 | 0.744 | **+0.094** | 0.0903 s | 0.0684 s |
| run H @1849（旧配方） | 0.768 | 0.782 | 0.765 | −0.014 | （4.32，被"悬空腿"污染） | 0.1688 s |
| `_init` @124（先验起步） | 0.426 | 0.427 | 0.412 | −0.002 | — | 0.116 s |

**结论（两半）**：

1. **用户对了一半**：先验自己的三态**也确实是"接近"的**（bound/pace 都有 0.74–0.83，绝不是
   §12 里那张**理想化**表说的 0.28/0.676）。所以"看三态接近"这个直觉方向是对的 —— 我之前那张
   理想表**不适用于这份高频短相位步态**，拿它当"干净 trot 应该 ≈1.000 vs 0.676"的尺子会误导。
2. **关键差别仍在**：先验自己的 **trot 始终领先 +0.09~+0.12**；两条 run 的领先量是 −0.014 与 −0.002
   ⇒ 现在**都不在先验的步态上**。另一个独立佐证：完整腾空时长 先验 **0.087 s** vs run H **0.169 s**、
   `_init` **0.116 s** ⇒ 训练出来的步子腾空时间是先验的 1.3–2 倍，不是同一套步态。
3. **判据应改成"领先量"**：`trot − bound > 0.05` ⇒ 与先验同类；`|三态极差| < 0.02` ⇒ 锁相/乱走；
   介于两者 ⇒ 混合。别再看绝对高度。

### 16.1 顺带查实：这套核参数对这份步态**几乎饱和**

先验的六对时间差只有 **0.085–0.090 s**（95 分位 0.20 s，最大 0.30–0.52 s），而
`max_err=0.5` ⇒ `clip(Δ², max=0.25)` **永远够不着**；再加上 `std=0.2`，单个核在 Δ=0.085 s 时是
`exp(−2·0.085²/0.2)=0.93`。于是三个声明的读数被整体挤进 **0.74–0.95**，动态范围只剩 ~0.2。
⇒ §29.31 里"按实测选 `max_err`"这条现在有了实测值：**该取 0.05–0.1 s**（`std` 相应取 ~0.005–0.01），
否则这个分类器/塑形项对**快而浅的 trot** 分不出与 bound 的差别。**这解释了为什么 run G→H 的"变陡"
没换来任何变化**：0.5 与 0.2 都远大于这份步态的真实时间尺度。

### 16.2 工具修了两个 bug（本轮）

1. `--npz`/`simulate` 传进核的是 `max_err` 而非 `max_err²`（`cap` 语义错）⇒ 修正后按
   `clip(Δ², max=max_err²)` 计算；对 Δ<0.5 s 的理想化表无影响（Δ²<0.25 ⇒ 不触发截断），但打印的
   "当前 max_err" 之前是错的（显示 0.707）。
2. `last_air_time` 原来在**每个触地步**都被覆盖成 0，与 Isaac 语义（记**刚结束**的那段腾空）不符
   ⇒ 改成只在**落地沿**更新；修正后先验的读数是 0.087 s（与 duty 0.56 × period 0.195 一致）。

**验证**：`gait_kernel_probe.py --npz` 两张表实跑；`test_gait_kernel_probe.py` 扩到 **10 项**
（含合成 trot/锁相回放、`last_air_time` 落地沿、以及**有 npz 才跑**的先验基准用例）；全仓离线
**392 通过／8 跳过**。

## 17. 相位核的参数：`std` 与 `max_err` 各管什么（`std` **不是必须保留**的）

问题：把 `max_err` 从 0.5 调到 0.1 就够了吗？在先验自己的接触序列上扫一遍（`--npz --sweep`）：

| 核 | trot | bound | pace | **溢价（trot−bound，＝奖励 /s）** | 单核地板 |
|---|---|---|---|---|---|
| 高斯 `std=0.2, max_err=0.5`（**现参数**） | 0.947 | 0.831 | 0.828 | **+0.116** | 0.082 |
| 高斯 `std=0.2, max_err=0.1` | 0.956 | 0.839 | 0.837 | **+0.116** | 0.905 |
| 高斯 `std=0.01, max_err=0.1` | 0.651 | 0.126 | 0.123 | **+0.526** | 0.135 |
| 高斯 `std=0.005, max_err=0.1` | 0.554 | 0.049 | 0.049 | +0.504 | 0.018 |
| 高斯 `std=0.02, max_err=0.3` | 0.730 | 0.260 | 0.255 | +0.470 | 0.000 |
| **hinge**（无 `std`）`max_err=0.15` | 0.723 | 0.240 | 0.235 | **+0.483** | 0 |
| hinge（无 `std`）`max_err=0.30` | 0.889 | 0.657 | 0.652 | +0.231 | 0 |

**结论**：
* **分辨率完全由 `std` 决定，`max_err` 只管"地板/截断"**：只把 `max_err` 0.5→0.1 而 `std` 保持 0.2，
  溢价**纹丝不动**（+0.116），地板反而升到 0.905（更没区分度）。
* 现参数下溢价只有 **0.116/s**（跟踪项是 5.0 ⇒ 2.3%），**这就是相位项一直没有梯度的原因**。
* `std` **不是必须保留**：改用归一化二次 hinge 核 `max(0, 1 − (Δa²+Δc²)/max_err²)`（只留 `max_err=0.15`）
  就得到 +0.483/s，与最好的高斯（+0.526/s）相当，而且少一个旋钮、地板天然为 0。
* 若坚持高斯形式：`std ≈ 0.01`（配 `max_err 0.1`）或 `std ≈ 0.02`（配 `max_err 0.3`）。
* 改核 = 改奖励 ⇒ **必须新开 run**；且 `trot` 绝对分会从 0.947 掉到 0.65–0.73，若要维持同等压力，
  可把 `feet_gait` 权重从 1.0 提到 2–3。

## 18. 初始化的 `std` 与「只装一个专家」（用户 2026-09-25 追问）

用户问：①「`std` 应该保留吗」（指**初始化的噪声 `std`**）；②「是不是还是要只给 1 个专家来初始化」。

先验的噪声 `std`（学出来的 12 维）：`[0.365, 0.303, 0.360, 0.381, 0.295, 0.328, 0.365, 0.284, 0.349, 0.359, 0.270, 0.348]`
⇒ 均值 **0.334**、范围 0.270–0.381（CMoE 默认 `init_noise_std=1.0`）。

**三种初始化在真实模块上的实测**（256 条随机观测，`CMoEActorCritic` 5 专家）：

| 方案 | 混合 vs 先验 RMSE | 地形列权重份额 | 含义 |
|---|---|---|---|
| `mode="all"`（当前） | **0.000** | 0.0000 | 初始行为**严格等于先验**、与门控无关；但 5 个专家完全相同、门控无理由分化 |
| `mode="first"`（只装 1 个） | **2.327** | 0.5537 | 混合被 4 个随机专家稀释 80% ⇒ **不是"从先验起步"**；地形列已有 55% 权重是随机的 |
| `all` + 新增列抖动 `σ=0.01` | **0.028** | 0.0897 | **两头兼顾**：仍≈先验，但专家一开始就不同 |

**结论**：
1. **噪声 `std` 应当保留（搬）**：它是"这份行为"的一部分 —— CMoE 默认的 1.0 会让动作噪声**盖过**
   先验的动作均值，第 0 步就不再是先验步态；而且它是**可学习参数**，训练中自己会调
   （`_init` run 实测 0.334→0.353）。已加 `--init_experts_std=true|false` 便于 A/B。
2. **不建议"只装 1 个"**：那样初始行为不是先验（RMSE 2.33，相当于 4/5 是随机策略），
   除非同时加**门控偏置**＋**锚/冻结**；否则"纯 trot 专家"几分钟内就会被 PPO 改写。
3. **推荐 `all` + `--init_experts_jitter=0.01`**：实测混合仍≈先验（RMSE 0.028）、第 0 个专家保持纯净、
   其余 4 个已经分化 ⇒ 既拿到"第 0 步就是先验步态"，又不浪费 5 个专家的容量。
4. 已加开关：`--init_experts_std`、`--init_experts_jitter`（默认 true / 0.0 ⇒ 现行为不变）、
   启动打印里会显示 `噪声 std 随先验装入 / 沿用 init_noise_std`。
   **验证**：`tests/test_cmoe_expert_init.py` 扩到 **23 项**（`include_std=False` ⇒ std 仍 1.0；
   抖动后 RMSE<0.1 且第 0 个专家新增列仍全 0、前 45 列仍等于先验；`first` 模式 RMSE>1.0 的负向对照）；
   全仓离线 **396 通过／8 跳过**。

## 19. `_init` run 的课程走势（15:08，@440）：**多数列在涨，两条列被"冻结带"钉住**

用户问"目前各个地形等级都在或多或少的增长了吗"。逐列读数（`Episode/Curriculum/terrain_levels/level_<地形>`）：

| 列 | @50 | @100 | @200 | @300 | @400 | @437 | 同期 `tracking_<地形>` @442 | 过门条件 |
|---|---|---|---|---|---|---|---|---|
| `flat` | 0.797 | 0.264 | 0.216 | 1.257 | 3.205 | **3.923** ↑ | 0.903 | >0.80 ✅ |
| `hf_pyramid_slope` | 0.646 | 0.217 | 0.140 | 1.102 | 2.639 | **3.100** ↑ | 0.829 | >0.80 ✅ |
| `hf_pyramid_slope_inv` | 0.331 | 0.039 | 0.180 | 1.414 | 3.226 | **3.853** ↑ | 0.813 | >0.80 ✅ |
| `pyramid_stairs_inv` | 0.317 | 0.135 | 0.707 | 1.785 | 2.887 | **3.240** ↑ | 0.639 | 放宽 0.50 ✅ |
| `pyramid_stairs` | 0.139 | 0.002 | 0.002 | 0.125 | 0.629 | **0.863** ↑ | 0.546 | 放宽 0.50 ✅ |
| `random_rough` | 0.623 | 0.152 | 0.010 | 0.264 | 1.296 | **1.813** ↑ | 0.829 | >0.80 ✅ |
| **`gap`** | 0.174 | 0.011 | **0.000** | 0.000 | 0.000 | **0.000** ✗ | **0.486** | >0.80（**未放宽**）❌ |
| **`boxes`** | 0.014 | 0.000 | 0.000 | 0.005 | 0.033 | **0.052** ✗ | **0.467** | 放宽 0.50 ❌（差 0.03） |
| **均值** | 0.336 | 0.082 | 0.117 | 0.536 | 1.291 | **1.575** ↑ | 0.646（全局） | pass_frac 0.521 |

健康度同步改善（同 run）：`mean_reward` 9.6(@200)→**31.2**、`illegal_contact` 0.589→**0.359**、
`ep_len` 481→**737**、`move_up_frac` 0.071→**0.409**、`move_down_frac` 0.814→0.531、
`distance_mean` 3.39→4.75（峰值 5.66@400）。

**读法（重要）**：这是"**先崩后爬**"——@100–200 是谷底（`level_mean` 0.082–0.117，与 §15 的
`illegal_contact` 0.72–0.80 同期），之后才回升。所以"在涨"= 从自己造成的坑里爬出来，不是新配方天然更好。

### 19.1 两条列的卡死：**"冻结带"机制**

课程判据是**按回合**的（`track_avg > 门槛` 才晋级、`< 0.35` 降级），所以：
`gap`（0.486）与 `boxes`（0.467）都落在 **[0.35, 门槛)** 的**冻结带**里 ⇒ **既不上也不下**，
于是钉在 0/0.05 —— 策略在进步，但这两列**永远不动**。

**但有一个口径陷阱不能忽略**：run H 的 `gap` 列在 tracking 均值只有 **0.62–0.78**（同样低于 0.8 门）
时，照样从 0.017 涨到 **6.49**（@250→1849：0.62/0.73/0.75/0.78/0.77）⇒ **列均值低于门槛 ≠ 没有回合过门**
（均值掩盖分布）。所以"`gap` 卡 0"到底是"大部分回合过不了门"还是"能过却被别的条件挡住"，
**当前日志答不了**。

### 19.2 已补的读数（本轮）

`mdp/curriculums.py::terrain_levels_vel_logged` 新增逐列 **`tracking_pass_frac_<地形>`** 与
**`tracking_fail_frac_<地形>`**（纯记录、不参与任何判据）⇒ 下一轮就能直接看到每一列"有多少回合
真的过了门"。**验证**：`tests/test_terrain_curriculum.py` 扩到 **10 项**（逐列过门比例，断言在放宽
阈值取 0.50 或 0.65 时都成立）；全仓离线 **397 通过／8 跳过**。

### 19.3 同一读数里两个比"等级涨不涨"更重要的发现

1. **MoE 门控已塌缩**：`expert_4` 独占 **0.951**，其余四个 0.007–0.024（`gate_entropy` 0.414→0.241→**0.162**，
   `ln5=1.609`）⇒ 5 专家**实际退化成单专家策略**，分工从未发生。在"全专家用同一先验初始化"的前提下
   这几乎是必然（5 个完全相同 + 没有任何分化压力）⇒ 正是 §18 那条 `--init_experts_jitter=0.01` 要解的。
2. **先验的步态没保住**：`trot−bound` 只有 **+0.025（mean）／+0.036（flat）**，而先验自己
   **+0.116／+0.094**；完整腾空时长 **0.144 s**（先验 0.087 s，还在变长）⇒ 这条 run 学会的是
   "它自己长出来的近-trot"，不是先验那个 trot。
   ⇒ **等级在涨 ≠ 先验路线成功**；两件事必须分开记账。

## 20. 控制台过滤：逐列 `gait_*` 不再打印（用户 2026-09-25 要求）

**现象**：`CMoEOnPolicyRunner.log()` 会把每个 `Episode/...` 键都打一行
`Mean episode <key>: <value>`。逐列指标是 `7 组 × 8 地形`（`gait_{trot,bound,pace,bounce,height,airtime,mismatch}_<地形>`）
＋`level_<地形>`／`tracking_<地形>`／新增的 `tracking_{pass,fail}_frac_<地形>` ⇒ 一轮近百行，控制台被刷没。

**做法**：新增 `CMoEOnPolicyRunnerCfg.console_skip_columns`（默认 `"gait"`）与纯函数
`cmoe_on_policy_runner._console_visible(key, mode)`。**过滤只作用于控制台，TensorBoard 一个 tag 都不少**
（`writer.add_scalar` 在过滤之前，见 `log()` 里的顺序）。

| 模式 | 控制台打印行数（按真实 100 个 tag 计） | 静音内容 |
|---|---|---|
| `none` | 100 | 旧行为（全打印） |
| **`gait`（默认）** | **44** | 逐列 `gait_*_<地形>`（7×8=56 行），保留 7 个 `gait_*_mean` |
| `all` | 12 | 另含 `level_<地形>`／`tracking_<地形>`／`tracking_{pass,fail}_frac_<地形>`（再省 32 行） |

汇总量（`*_mean`／`*_min`／`*_max`、`move_up_frac`／`frozen_frac`／`tracking_pass_frac` 等）在两种静音模式下都照旧打印。

**验证**：`tests/test_cmoe_runner_logging.py` 扩到 **6 项** —— 新增
"逐列 gait 写 TB 但不出现在 stdout、`gait_*_mean` 仍打印"、
"`all` 模式下写 TB 的键数不变（5/5）且逐列 level_/tracking_ 被静音"、
"`none` 回退到旧行为"；`_log` 助手改为同时返回 writer 标量与 stdout 文本。
全仓离线 **400 通过／8 跳过**，`py_compile`／表格校验／`git diff --check` 通过。

## 21. 2026-09-25 重启训练：修掉课程"冻结带"＋换用两个初始化开关

### 21.1 修了什么

| 改动 | 位置 | 依据 | 影响 |
|---|---|---|---|
| `gap` 纳入**放宽名单** | `CMoE_env_cfg.py`：`relaxed_terrain_names` = `pyramid_stairs, pyramid_stairs_inv, boxes, **gap**` | `_init` run 的 `level_gap` 从第 200 轮起**严格 0.000**（140+ 轮不动），同期 `tracking_gap` = **0.486**；gap 未放宽 ⇒ 需 >0.80 ⇒ 落进 [0.35, 0.80) 冻结带 ⇒ 永远停在最窄的 0.126 m，过沟能力测不到 | 沟壑列恢复"进度驱动"（走够 4 m 且跟踪 > 阈值即晋级） |
| 放宽阈值 **0.50 → 0.40** | 同上：`tracking_move_up_relaxed` | `boxes` 虽在名单里，但 `tracking_boxes` = **0.467** < 0.50 同样被冻住；0.40 让 0.486/0.467 都能过门（冻结带缩到 [0.35, 0.40)） | 障碍列基本回到"只看进度"；容易地形仍保持 0.80 的跟踪门控 |
| 新增逐列过门比例 | `mdp/curriculums.py`（§19.2） | 列均值掩盖分布：run H 的 gap 在均值 0.62–0.78（<0.8）时照样涨到 6.49 | 下一轮可直接看 `tracking_pass_frac_<地形>` 判定"是不是真过门" |
| 控制台过滤逐列 `gait_*` | `console_skip_columns="gait"`（§20） | 一轮近百行把控制台刷没 | 控制台 100 → 44 行，TB 不变 |

⚠️ **不确定性要说清**：0.40 是"实测值 0.486/0.467 再留 0.05 余量"的选择；若发现障碍列"进度够、跟踪很差"仍被放行，
把 `tracking_move_up_relaxed` 回到 0.45–0.50 即可（只改一个数）。

### 21.2 重启命令（训练机）

```bash
cd <repo>/imgo2_rl
pkill -f "cmoe/train.py" || true            # 停掉在跑的 _init（它用的是旧课程与旧日志代码）
PRIOR="$(pwd)/logs/amp_rsl_rl/base_move_amp/2026-09-17_21-25-57/model_24500.pt"
LOG="logs/train_$(date +%m%d_%H%M)_gaitfree_ampactor_jitter.log"

PYTHONUNBUFFERED=1 nohup bash scripts/run_isaaclab.sh scripts/rl_lab/cmoe/train.py \
  --task=Imgo2-basemove-rough-cmoe-gaitfree \
  --num_envs=4096 --headless --max_iterations=2000 --seed 1 \
  --init_experts_from="$PRIOR" \
  --init_experts_critic=false \
  --init_experts_jitter=0.01 \
  --run_name=cmoe_gaitfree_ampactor_jitter_v2 > "$LOG" 2>&1 &
echo "日志: $(pwd)/$LOG"
```

* `PYTHONUNBUFFERED=1` 必须加（README 记录过：不加会丢控制台输出）。
* 想同时保留手工步态 shaping（"双保险"）：把 `--task` 换成 `Imgo2-basemove-rough-cmoe` 即可，其余不变。
  注意那条路上的 `feet_gait` 相位核**对这份高频步态几乎饱和**（§17：溢价只有 0.116/s），指望它压住步态不现实。
* 先跑 2000 轮（≈2.5 h，4.5 s/轮）确认，再决定是否加长（`--max_iterations=60000`）。

**后台化的正确姿势（2026-09-25 首次重启踩过坑）**：`nohup ... &` 如果把 `&`／重定向写丢，进程会**在前台**跑，
终端只显示 `nohup: 忽略输入并把输出追加到 'nohup.out'` 然后看起来"卡住" —— 其实**训练正常**，只是没后台化、
输出进了 `imgo2_rl/nohup.out`。当时误判成卡死、Ctrl-C 掉了第一次（`2026-09-25_15-25-44_...`，只留下 `model_0.pt`
与 30 KB tfevents，属废 run），随后重跑才有了现在这条。稳妥写法：

```bash
setsid nohup bash scripts/run_isaaclab.sh scripts/rl_lab/cmoe/train.py ... </dev/null > "$LOG" 2>&1 &
# 或者： nohup ... </dev/null >"$LOG" 2>&1 & disown
```
终端已被占住又不想丢训练：**`Ctrl-Z` → `bg` → `disown`**（**别按 Ctrl-C**，那是杀训练）。最省心是用 `tmux`。

**重复启动的检查与日志轮转（同日踩过三次启动的坑）**：启动前先 `pgrep -af "cmoe/train.py"`
（应只有一条）；判断"某条 run 是死是活"最快的方法是看它 `events.out.tfevents.*` 的 **mtime 是否还在推进**
（15:25:44 与 15:26:36 两条分别停在 15:26:19 / 15:28:22 ⇒ 已死）。`nohup.out` 会把**每次**启动的输出累积在
同一个文件里，可以在**不杀进程**的前提下轮转：`mv nohup.out nohup.<时间>.out`（进程的 stdout 仍指向同一个
inode，照常写入新名字）。

**当前在跑的 run**（第 3 次启动）：`imgo2_rl/logs/cmoe/base_move_cmoe_rough/2026-09-25_15-30-26_cmoe_gaitfree_ampactor_jitter_v2`
（Isaac 报 ETA ≈ 9400 s ≈ 2.6 h）；启动四行核对全部通过（`critic **保持随机初始化**`、`已给 4 个专家…N(0, 0.01)`、
`std 均值 0.3340`、自检 RMSE `1.214e-02`），课程修复也已在它自带的 `env.yaml` 里生效
（`relaxed_terrain_names` 含 `gap`、`tracking_move_up_relaxed: 0.4`）。

### 21.3 启动后核对清单（3 分钟）

```
[INFO] 先验初始化：... 已装进 5/5 个专家（mode=all），actor 第一层 (512, 157)（置零列 112）；
       critic **保持随机初始化**；噪声 std 随先验装入
[INFO]   噪声 std 随先验装入：均值 0.3340
[INFO]   已给 4 个专家（第 1 个保持纯净）的新增 112 列加 N(0, 0.01) 扰动以打破对称
[INFO] 先验初始化自检：固定参考批次（1024 条）上 混合策略 vs 先验 动作 RMSE = ~0
```
任一行不对（尤其 `critic 随先验装入` 或"已给 0 个专家"）就先别走。

### 21.4 与 §19 基线对比时看什么

| 读数 | 上一轮（`_init` @440） | 这一轮的判据 |
|---|---|---|
| `level_gap` | **0.000**（钉死） | **开始 > 0** —— 这是本次修复的直接验证 |
| `level_boxes` | 0.052 | 开始爬 |
| `tracking_pass_frac_gap`（新） | 无 | 若 >0 说明确实有回合过门；若仍 0 ⇒ 0.40 还不够，要看进度 |
| `Policy/expert_*_mean_weight` | `expert_4` **0.951**（塌缩） | 应**分散**（jitter 生效）；仍塌缩 ⇒ 加锚 |
| `Policy/prior_new_col_mass` | 0（旧代码没这项） | 从 0 上升 ⇒ 地形通路在被打开 |
| `Policy/prior_weight_rel_drift` | 无 | 上升速度 ⇒ 先验被改写多快（决定要不要加 BC/KL 锚） |
| `gait_trot_mean − gait_bound_mean` | **−0.002** | 参考先验自己的 **+0.116**；保持 >+0.05 ＝步态没漂 |
| `illegal_contact` | 0.311 | 不上升 |

## 22. 重启后首次读数（`2026-09-25_15-30-26_..._ampactor_jitter_v2`）

状态：@43 轮、Iteration time 4.55 s、Isaac 报 ETA ≈ 9410 s（≈2.6 h）；只有这一条在跑
（前两次启动的 tfevents 分别停在 15:26:19 / 15:28:22）。启动四行核对全过（`critic **保持随机初始化**`、
`已给 4 个专家…N(0, 0.01)`、`std 0.3340`、自检 RMSE `1.214e-02`）。

### 22.1 与上一条（`_init`，搬 critic）同轮数对比 @34

| 指标 | 新（actor-only + jitter） | 旧（`_init`，搬 critic） | 读法 |
|---|---|---|---|
| `Policy/gate_entropy` | **1.5935** | 1.1303 | ✅ jitter 起效：门控保持近均匀（ln5=1.609），不再往塌缩走 |
| `gait_airtime_mean` | **0.0811** | 0.0665 | 先验自己是 **0.087** ⇒ 新的更接近先验的步频/腾空 |
| `Episode_Termination/illegal_contact` | **0.9409** | 0.7542 | ⚠️ 摔得更凶（早期） |
| `level_mean` / `level_gap` / `level_boxes` | 0.196 / 0.050 / 0.010 | 0.514 / 0.377 / 0.054 | ⚠️ 早期摔多 ⇒ 课程更低（判不了修复） |
| `tracking_pass_frac_gap` | **0.0879** | —（上一条没这项） | ✅ 沟壑列**确实有回合过门**（放宽生效），但进度不够 ⇒ level 还没起来 |
| `gait_trot_mean − gait_bound_mean` | −0.006 | +0.001 | 都还没偏向 trot（variance 大，@34 太早） |

### 22.2 先验漂移：**开局一跳，之后慢漂**

| 轮 | 1 | 8 | 16 | 24 | 34 | 43 |
|---|---|---|---|---|---|---|
| `Policy/prior_weight_rel_drift` | **0.0338** | 0.0343 | 0.0354 | 0.0370 | 0.0394 | **0.0417** |

* **第一次更新就贡献了当前漂移的 ~80%**（0.0338/0.0417）；离线取证 `model_0.pt`（`iter=0`，即第 0 轮
  更新后落盘）：专家 0 的第一层前 45 列已偏离 AMP actor **4.05e-03**、尾部 std 从 0 → **0.0029**
  ⇒ 确实是"一轮就跳"。
* 之后速率降到 **≈0.0002/轮**，按此外推 2000 轮 ≈ **0.4 相对漂移**。
* `Policy/prior_action_rmse` 从自检 **0.0121** 跳到 **1.09** —— 但这条**不可全信**：参考批次是**回合初始**的
  1024 条观测，对先验是 OOD（先验在极端观测下原始输出可达 ±17），绝对 RMSE 被放大。
  ⇒ **判漂移以 `prior_weight_rel_drift`（与观测无关）为准**；`prior_action_rmse` 只当相对趋势看。

### 22.2b @142 更新（15:43）：**课程被全场降到 0，瓶颈是策略不是门控**

| 读数 @142 | 值 | 对照 |
|---|---|---|
| `level_mean` | **0.004**（各列 ≈0：flat/rough/slope 全 0，stairs_inv 0.039） | 旧 `_init` @200 是 0.117 ⇒ 新的谷更深 |
| `move_down_frac` / `move_up_frac` | **0.9972** / 0.0000 | 99.7% 的回合在降级 |
| `progress_mean` | **2.61 m**（晋级要 >4 m） | 连"走够 4 m"都不到 ⇒ **放宽门控不是瓶颈** |
| `illegal_contact` / `ep_len` | 0.884 / 158 | 旧 `_init` @200：0.589 / 481 |
| `tracking_pass_frac_gap` | **0.155** | ✅ 放宽生效（15.5% 的沟壑回合过跟踪门） |
| `gate_entropy` | 1.5864 | ✅ jitter 仍有效 |
| `gait_airtime_mean` | 0.0752 | 先验 0.087 ⇒ 仍接近先验 |
| `gait_trot − gait_bound` | **+0.004**（0.6019 vs 0.5978） | 首次转正，但幅度远小于先验的 +0.116 |
| `prior_weight_rel_drift` | 0.0589 | 与 §22.2 的 ≈0.0002/轮 外推一致 |

**结论**：门控修复、jitter、airtime 都按预期工作，但**策略本身在早期被打进了一个比先验差得多的区域**
（`progress_mean` 2.6 m、88% 摔率、全场降级）。结合 §22.2 的"开局一跳"，最对症的短期对策是
**critic warmup**（前 N 轮冻结 actor、只训 critic/estimator），把"随机 critic 的噪声 advantage 把先验打飞"
这一步消掉；其次是退回"先验 + 手工 shaping"的双保险配方。

### 22.3 下一步要盯的三件事

1. **@100–200 轮的 `illegal_contact`**：必须从 0.94 降下来（旧配方 @200 是 0.589、@437 是 0.359）。
   若仍 >0.9 ⇒ "随机 critic + 大步长"把先验在开局打飞了，对策是 **critic warmup**（前 N 轮冻结 actor，
   只让 critic 追奖励尺度）——这是 §22.2 那条"开局一跳"的直接对策，尚未实现。
2. **`level_gap` 是否稳定离开 0**：现在 0.050，且 `tracking_pass_frac_gap` 0.088 > 0 ⇒ 门已开，差进度。
3. **`gait_trot − gait_bound` 与 `gait_airtime_mean`**：前者应向 +0.05 以上走（先验自己 +0.116），
   后者应停在 0.09 附近（先验 0.087）；若 `airtime` 往 0.15+ 走 ⇒ 步态又在往"大腾空"漂。

## 23. 实时看 iter 数据：`read_tfevents.py --follow`（用户 2026-09-25 要求"想要看到 log、有 iter 数据"）

控制台（`nohup.out`）里只有 Isaac 的周期表和 runner 的 `Mean episode ...` 汇总，**逐轮次的量在
TensorBoard event 文件里**。为此给 `scripts/tools/read_tfevents.py` 加了 `--follow`：
像 `tail -f` 一样盯着 event 文件，**每出现一个新轮次就打一行**（默认从当前最新轮次开始，不重放历史）。

```bash
cd <repo>/imgo2_rl
python3 -u scripts/tools/read_tfevents.py \
  --run logs/cmoe/base_move_cmoe_rough/2026-09-25_15-30-26_cmoe_gaitfree_ampactor_jitter_v2 \
  --follow 20 --total 2000 \
  --columns mean_reward,illegal_contact,level_mean,level_gap,tracking_pass_frac_gap,gate_entropy,prior_weight_rel_drift,gait_trot_mean,gait_bound_mean
```

输出形如（实测）：

```
# 从 iter 264 开始盯（历史用 --steps 查）
iter    265 | mean_reward 13.9818 | illegal_contact 0.4798 | level_mean 0.0444 | level_gap 0.0000 |
             tracking_pass_frac_gap 0.8147 | gate_entropy 1.5407 | prior_weight_rel_drift 0.1080 | ...
[progress] 265/2000 轮（13.3%），13.6 轮/分钟，ETA ≈ 127.6 分钟
```

要点：`--follow [SECONDS]`（默认 10 s 轮询）、`--columns` 选列、`--total N` 打印进度与 ETA
（速率用**滑动窗口**、≥60 s 样本才算，避免前几次乱跳）、**Ctrl-C 退出且不影响训练**。

**开发中踩到并修掉的两个坑**（都已加测试）：
1. 列解析原来取"第一个包含该子串的 tag"，于是 `mean_reward` 命中了 `Train/mean_reward/time`
   —— 而该 tag 的 step 是**墙钟秒** ⇒ 打出一堆"没有数据的假轮次"（265/273/280…）。
   修法：优先"以该名字结尾且不是 `*/time`"的 tag；`resolve_columns` 找不到的列原样保留（打印 `-`）。
2. 新轮次原来从**所有** tag 的 step 里取 ⇒ 又把墙钟秒混进来。修法：只用 `iteration_step()` 选出的
   **迭代轴 tag**（这里是 `Train/mean_reward`）的 step。

**验证**：`tests/test_read_tfevents.py` 扩到 **9 项**（列解析不命中 `*/time`、行格式化含缺失列、
`--follow` 不重放历史＋新轮次逐行打印＋Ctrl-C 返回 0）；另有对**正在跑的 run** 的实跑读数（见 §22/§22.2b）。
