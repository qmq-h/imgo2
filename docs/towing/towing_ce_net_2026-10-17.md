# 拖曳 CE Net 结构调整与合并（2026-10-17）

> 本文有两个来源：① 远端 `817d44e` 的实现记录（原文已随本合并改写）；② 本机合并时的适配
> 与评审，见 §4/§5。远端那台的同步背景见
> [同步记录](../maintenance/main_sync_decoder_2026-10-17.md)（其起点是 `4d66f2c`，不含本机的
> 「decoder 去质量」）。

## 1. 用户要求与最终结构

用户要求（远端记录原文）：「观测历史经 GRU 并列得到显式动力学估计与变分隐向量，两者共同
重建下一时刻的 58 维观测」。合并后（**保留本机 2026-10-16 的「去质量」准则**）：

```
o_t (58) ─ encoder(58→128)+GRU(128) ─┬─ velocity_head(128→2) ┐ 显式 5 维（有监督）
                                      ├─ force_head(128→3)    ┘
                                      ├─ fc_mu / fc_log_var → z(16)   （KL 只在这里）
                                      └─ cat(显式5, z16)=21 → MLP → ô_{t+1} (58)
actor 输入 = 58 + 5 + 16 = 79（后两者 detach，PPO 不训练 CE Net）
L = L_est + reconstruction_coef·MSE(ô_{t+1}, o_{t+1}) + kld_weight·KL
     L_est = 5.0·SmoothL1(vel) + 10.0·SmoothL1(force)      （物理单位 m/s、N，不归一化）
     reconstruction_coef = 1.0，kld_weight = 0.005（**均未实跑标定**）
```

- 显式量**不经过**变分瓶颈（直接读 GRU hidden）；质量**不显式预测**（只留给 latent 与重建）。
  与远端原实现的差别正是这一点：远端保留 `[vx, vy, mass, Fx, Fy, Fz]` 6 维、actor 80。
- 推理只前向编码分支（`forward_with_latent`，`sample=False` 取 μ），**不调用重建头**；
  重建只用于训练。
- 布局常量集中在 `towing_decoder.py`（`VELOCITY_DIM`/`FORCE_DIM`/`OUTPUT_DIM`/`FORCE_SLICE`），
  `force_newtons` 由它们推导。

## 2. 契约版本：v5（不是 v4）

三条线都曾用「v4」表示不同的网：

| 来源 | v4 的含义 |
|---|---|
| 本机 2026-10-16 | 去质量、无重建：58 / **5** / 16、actor 79 |
| 远端 2026-10-17 | CE Net、含质量：58 / **6** / 16、actor 80 |
| 合并结果（本文） | CE Net、去质量：58 / **5** / 16、actor **79** |

三者权重互不兼容，因此合并后把契约升为 **v5**：`TOWING_CONTRACT_VERSION = 5`（runner）、
`CHECKPOINT_CONTRACT_VERSION = 5`（runtime）、`UPPER_CONTRACT_VERSION = 5`（测试台镜像）。
`version=4` 的 checkpoint 一律拒绝（`explicit_dim` 也会挡 5≠6）。**不要改旧 checkpoint 的
元数据冒充 v5**，必须新建 run。

## 3. 时序与回合边界

runner 在 `env.step` **前**快照 `o_t` 与同拍物理监督，**后**快照返回的 `o[t+1]`；包括 rollout
最后一拍、不 shift 丢尾；所有缓存 `.clone()`（Isaac Lab 会复用 observation buffer）。
`dones` 为真的那一步，返回值可能是自动 reset 后的新回合首帧 ⇒ **只对 `~dones` 计算重建损失**
（当前时刻的物理监督与 KL 保留）；`update` 重算序列时按 `dones[step-1]` 把已结束环境的 GRU
隐状态清零。整批全终止时重建损失精确为 0，仍可训练估计与 KL。

## 4. 本机适配清单（合并时改的）

| 文件 | 改动 |
|---|---|
| `scripts/rl_lab/rl_lab/modules/towing_decoder.py` | 取远端 CE Net；去掉 `mass_head`/`MASS_DIM`/`mass_supervision_weight`；显式头合并为 `velocity_head`+`force_head`；`latent_decoder` 输入 `OUTPUT_DIM+latent_dim=21`；`loss()` 只留 vel/force；`update(frames, targets, *, dones, hidden_state, next_frames)` 关键字专用 |
| `scripts/rl_lab/rl_lab/runners/towing_on_policy_runner.py` | 契约 **v5**；`next_frame_rollout` + `Loss/decoder_reconstruction(_weighted)` + 控制台重建行；**去掉** `weight_rollout`/`mass_weights`/`mass_coef`；分项日志只留 velocity/force |
| `scripts/rl_lab/rl_lab/config/towing_algorithm_cfg.py`、`agents/upper_ppo_cfg.py` | 加 `reconstruction_coef=1.0`；仍**没有** `mass_coef` |
| `scripts/towing/upper_policy_runtime.py`、`play_towing_test.py` | 版本 5；显式维保持 5；`estimate`/`latent` 观测口径不变 |
| `tests/test_towing_ce_net.py` | 远端新增文件，按 5 维改写（显式头、targets 5、`update(..., next_frames=)`、真实 runner 循环的帧/下一帧配对），外加一条「合并不得把质量头带回来」的源码守卫 |
| `tests/test_towing_upper_rl_contract.py`、`test_towing_upper_policy_runtime.py`、`test_towing_play_test.py` | 版本号 4 → 5；trainer 用例改为传 `next_frames` |

## 5. 评审发现（结构方向对，但下面三条要在训练机上确认）

**① 重建目标里有 ~13–16 维不可观测的策略输出（结构性问题，最值得先处理）**

`upper_last_action` 返回 `processed_actions`（**当前**动作）；Isaac Lab `step()` 的顺序是
`process_action → 物理 → reset → compute observations`（`manager_based_rl_env.py`），所以
`o_t` 里是 `a_t`，而 `o[t+1]` 里的 `last_action` 是 **`a_{t+1}`**——decoder 在 t 时刻只看到
`o_t`，拿不到 `a_{t+1}`。同理 `loco_command` 的 3 维里含偏移头对 `a_{t+1}` 的输出。
训练时动作用策略分布采样（`init_noise_std=0.5`、clip ±1），这部分**只能预测均值**：

- 不可消除的 MSE 下限 ≈ 13/58 × Var(a) ≈ **0.05**（RMSE ≈ 0.23/维），外加 3 维偏移；
- 重建梯度在这些维度上退化为噪声。

可选修法（**尚未实现，等用户定**）：只对「外生可观测切片」算重建
（`projected_gravity`/`base_ang_vel`/`joint_pos`/`joint_vel`/`last_loco_action` = 42 维）、
或把这些切片 mask 掉、或把 `a_t` 一并喂进重建分支。

**② 重建项与 KL 相对 force 项小 2~3 个数量级（标定风险）**

`force_coef=10` 且目标未归一化（N）：误差几牛时 smooth_l1 ≈ 2.5 ⇒ 贡献 ≈ 25；重建是 58 维
原始帧的 MSE（≈0.1~0.3，coef 1），KL ≈ 43×0.005 ≈ 0.22。对共享 GRU 而言重建的单维梯度
（≈2·e/58 ≈ 0.01）比 force（≈10）小约 **10²~10³ 倍** ⇒ **latent 有塌缩成常数的风险**，
"上下文编码"可能名不副实。修法：逐维标准化重建目标、或调 `reconstruction_coef`、或把 force
也归一化。**必须用 `Loss/decoder_reconstruction` 与 `Loss/decoder_kl` 的实测读数定权重。**

**③ 显式头绕开瓶颈**（设计后果，不是 bug）：变分部分不再约束估计量，质量从「latent 假定」
变成「只由重建隐式影响」。训练后需用 `analyze_latent*.py` 复核质量在 latent 上是否仍可读。

另外：远端原实现把 `mass_supervision_weight` 放回 `update()` 的第 3 个位置参数，本机合并时
**没有**恢复它（保持关键字专用），避免「监督权重被当成 dones」的静默错位。

## 6. 验证

| 检查 | 结果 |
|---|---|
| 远端树（`git archive origin/main` 到独立目录）全量 `pytest` | **604 passed, 4 skipped**；其中 `test_towing_ce_net.py` **7/7 通过**（远端那台无 torch，6 项是跳过的 ⇒ 这些张量用例**第一次真正执行**） |
| 合并后本机全量 `pytest imgo2_rl/tests -q` | **670 passed, 4 skipped, 1336 subtests passed** |
| `py_compile`（全部改动的 .py） | 通过 |
| `git diff --check` / `git ls-files -i -c --exclude-standard` | 干净 / 0 条 |

覆盖到的行为：并列头梯度（重建梯度回传显式头与 latent）、显式量不依赖 latent 采样、
下一帧损失与 done 掩码（含「把 reset 帧替换成 NaN 也不影响损失」）、全终止 batch、GRU 按环境
重置、真实 runner 循环的 `frames[t] → next_frames[t]` 配对与缓冲区快照、契约 v5 三处常量一致。

**未验证（缺什么才能完成）**：本机无 Isaac Lab ⇒ 仿真一次没跑。需要训练机：① 一轮 64 环境
× ~20 轮冒烟，确认三类损失有限、有梯度且量级排序可解释（读 `Loss/decoder`、
`Loss/decoder_velocity|force`、`Loss/decoder_reconstruction`、`Loss/decoder_kl`）；
② 重建确实对应**下一帧**（对比 `next_frames` 与 `ô_{t+1}` 的逐维误差，按切片分组看
`last_action` 那 13 维是否显著更差——这正是 §5① 的可测证据）；③ done/timeout 后 hidden
重置与掩码在真实环境里的行为；④ v5 checkpoint 的 save/load 往返；⑤ 据读数重标
`reconstruction_coef` / `kld_weight` / force 归一化。
