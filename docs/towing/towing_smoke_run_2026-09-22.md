# 拖曳上层 RL 冒烟实跑结果（2026-09-22）

## 运行信息

| 项 | 值 |
|---|---|
| 命令 | `run_isaaclab.sh imgo2_rl/scripts/rl_lab/towing/train.py --task=Imgo2-towing-upper-rl-lab --num_envs=4 --max_iterations=10 --headless` |
| 代码版本 | 工作树含本轮改动（`--agent` 已修为 `rl_lab_cfg_entry_point`，注册块已加入） |
| GPU | 可用（用户实测 `cuda available: True`、`CUDA solve: OK`） |
| 日志目录 | `logs/towing_rl_lab/towing_upper/2026-09-22_21-33-23/`，文件名内主机名 `gpufree-container` |
| 是否跑完 | 是，进程已退出，无残留 `train.py` |

## 结论：冒烟通过，管线闭环可用

10 轮 × 4 环境跑完并正常落盘。**这是拖曳上层 RL 链路第一次真实运行**，此前所有结论都是离线
静态核对。

### 已验证（本次实跑的直接证据）

- **任务注册生效**：第一次实跑的错误栈来自 `load_cfg_from_registry`，说明 gym spec 已找到
  `Imgo2-towing-upper-rl-lab`；修正 `--agent` 后不再报错。
- **环境构造成功**：4 环境、`decimation: 10`、`episode_length_s: 10.0`、`sim.dt: 0.005`、
  `device: cuda:0`，`env.yaml` 落盘的配置与设计一致。
- **51 维帧 + 5 维 estimate = 56 维 actor 契约在运行时成立**：checkpoint 里
  `memory_a.rnn.weight_ih_l0` 形状 `(768, 56)`（actor GRU 输入 56 维，3×256=768）；
  `memory_c` 为 `(768, 65)`（critic 65 维）。这是 TOW-03 验收项里的「wrapper 构造」实跑证据。
- **三套 GRU 都在 checkpoint 里**：`model_state_dict` 含 actor／critic 两套（`memory_a`／`memory_c`），
  `decoder_state_dict` 含 decoder 自己的 GRU（`gru.weight_ih_l0 (384, 128)`，3×128）。
- **critic-only normalizer 已保存**：`critic_normalizer_state_dict` 为 `mean (65,)`、`var (65,)`、
  `count`，维度与 critic 观测一致。
- **两个优化器状态都在**：`optimizer_state_dict` 与 `decoder_optimizer_state_dict`。
- **`iter` 写入正确**：`model_0.pt` 的 `iter = 0`、`model_10.pt` 的 `iter = 10`（对应
  `max_iterations=10`）。这是 AMP-08 修好后**第一次在真实 checkpoint 上确认**该字段随迭代推进。
- **decoder 可独立加载并推理**：按 checkpoint 权重 strict 加载成功；单步输入 `(6,51)` →
  `pred (6,5)`、hidden `(1,6,128)`；序列输入 `(4,3,51)` → `(4,3,5)`；
  `augment_actor_observation` 输出 `(6,56)`；`reset_gru_hidden` 按 done 掩码工作；
  `force_newtons` 反归一化正常（示例输出质量 10.80 kg，落在 5–15 kg 训练域内）。

### 训练数值（`events.out.tfevents...`，8 个标量）

| step | mean_reward | mean_episode_length | surrogate | value_function | decoder | mean_noise_std |
|---|---|---|---|---|---|---|
| 2 | −5.51 | 110.3 | 4.70 | 0.103 | 0.0187 | 0.5019 |
| 3 | −14.37 | 120.0 | 3.15 | 0.479 | 0.0062 | 0.5028 |
| 4 | −14.37 | 120.0 | 1.61 | 35.36 | 0.0496 | 0.5040 |
| 5 | −14.37 | 120.0 | 3.17 | 4.148 | 0.0236 | 0.5051 |
| 6 | −20.97 | 154.3 | 2.29 | 23.22 | 0.211 | 0.5059 |
| 7 | −29.46 | 160.0 | 0.848 | 18.01 | 0.300 | 0.5068 |
| 8 | −29.46 | 160.0 | 1.48 | 34.77 | 0.134 | 0.5082 |
| 9 | −29.46 | 160.0 | 0.412 | 2.231 | 0.123 | 0.5096 |

**不要从这张表读学习效果。** 只有 10 轮 × 4 环境，且这几个量都不是拖曳的验收指标：

- `mean_reward` 从 −5.5 降到 −29.5，主要因为 `mean_episode_length` 从 110 涨到 160（上限 200），
  而 v0 reward 以惩罚项为主（collision/fall 各 `−50`、clearance/stop_force 等），回合越长累计越负。
  这不是「策略变差」，是回合变长的算术结果。
- `mean_episode_length` 上升说明回合多跑到超时（`episode_length_s=10.0`、50 Hz → 上限 200 步），
  即**大部分回合没有被提前终止**——这是好消息（没在一开始就撞车或跌倒），但 10 轮内不能当结论。
- `value_function` 在 0.1–35 之间大幅震荡、`surrogate` 在 0.4–4.7 之间震荡，是 critic 尚未拟合、
  优势估计噪声大的表现，早期正常。
- `decoder` loss 在 0.006–0.66 之间无趋势。**无法据此判断 decoder 是否在学**：本次没有跑
  「无 decoder／constant prior／Oracle 真值」对照，而这是设计文档要求的判据。

## 本次冒烟没有覆盖的（因此 TOW-03 验收项仍未完成）

- TensorBoard 里**没有记录**拖曳专用诊断量：`cart_present` 比例、绳力／张力、真实间隙、
  轮阻、碰撞与终止原因。所以**无法从本次产物核对**「`cart_present` 是否约 12.5%」「横向停放的
  小车是否跨环境接触」「STOP 后是否有为卸载拉力而逼近的策略捷径」这三条关键验收项。
- 未跑 256 环境，未验证异步 reset 在大量环境下的比例、跨环境写入与 NaN。
- 未做 checkpoint 恢复。`--resume` 的目录逻辑已于同日修正（恢复时写回被恢复的 run 目录），但**修正后尚未实跑验证**；另外 `learn()` 把 `max_iterations` 当「本次还要跑多少轮」而非总轮数目标，恢复时总轮数会超出，恢复前需据此计算 `--max_iterations`。
- 未做 estimator 精度评估（需要真值对照）。
- 未做 scripted policy 复现 `tow_drag.py`。

## 下一步建议

1. 先加诊断量记录（reward 分项、终止原因、`cart_present`、绳力），否则正式训练跑完也无法判断
   奖励与终止是否按设计工作；这是把「能跑」变成「能评价」的前提。
2. 再跑 256 环境正式训练。
3. 若要判断 decoder 是否有效，必须先建立设计文档要求的四组对照（无 decoder／constant prior／
   shuffled label／Oracle 真值）。

## 产物清单

```
logs/towing_rl_lab/towing_upper/2026-09-22_21-33-23/
├── model_0.pt          iter=0
├── model_10.pt         iter=10
├── events.out.tfevents.1790084008.gpufree-container.127109.0
└── params/{env.yaml, agent.yaml}
```
