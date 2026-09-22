# 拖曳 Dynamics Decoder 与 recurrent PPO（2026-09-22）

## 结论

该结构可行，且比从机器人侧本体感知强行重建小车速度、位置和完整绳参数更合适。环境每个 20 Hz 控制步输出当前 51 维可部署本体帧；decoder 使用 `51→128→GRU(128)`，显式估计机器人机体系 `vx/vy`、小车质量和机器人所受机体系 `Fx/Fy`。5 维估计 detach 后与原始帧组成 56 维 actor 输入。PPO actor、critic 继续使用各自独立的 GRU。

训练期 critic 和 decoder label 可以读取机器人／小车真值速度、质量、牵引力、绳状态、摩擦和轮阻；部署只保留 51 维输入、decoder、actor 和两套 hidden state。

## Decoder 监督

- `L_v`：机器人 `vx/vy` Huber loss，全程监督；
- `L_F`：机体系 `Fx/Fy` Huber loss，全程监督，松绳时零力也是有效标签；
- `L_m`：质量 Huber loss，使用 `s=max(||F_GT||-1 N,0)`、`w=s/(s+10 N)` 连续加权；无有效拉力时权重为零。

总损失为 `lambda_v L_v + lambda_F L_F + lambda_m w_t L_m`。权重由 detach 的 GT force 计算，不能使用预测力。预测误差不进入 PPO reward；PPO 梯度也不进入 decoder。Decoder 只在 PPO 使用完当前 rollout 后更新，rollout 保存采样时的 5 维估计。松绳后的 persistent 质量监督只作为遗忘问题的备选消融。

Target 使用 `vx/1.0`、`vy/0.5`、质量 5–15 kg 线性映射以及逐分量 `F/(|F|+10 N)`；training-only observation 单独输出由物理牵引力计算的质量权重。

## 可行性边界

- 机器人速度可由 IMU、关节状态、底层动作和 GRU 历史学习，但无足端接触标志时会受打滑影响，必须在未见摩擦条件报告 RMSE。
- 质量和牵引力不是无条件可辨识量。不同质量、轮阻和绳参数可能产生相似本体响应；结论只能限定在训练域及有效牵引发生之后。
- Unknown `L/k/c` 可由 hidden state 吸收其历史效应，但这不代表网络恢复了真实绳参数。需要跨绳模型和参数分布验证控制收益。
- Decoder 更新会改变 actor observation 分布。第一版应先预训练 decoder，再交替进行 PPO rollout 与 decoder 更新，并保留无 decoder、constant prior、shuffled label 和 Oracle 真值输入对照。
- 当前停车 reward 同时要求跟踪零速度、卸载绳子拉力并限制停车后继续前进；拉力项只在正式 STOP 后生效，使用 `||F||/(||F||+10 N)` 有界归一化。它能表达“只前移必要距离并逐步卸载”的折中，但必须用 scripted rollout 验证安全延迟停车的 return 高于立即停死后追尾。

## 已实现

- PPO 使用仓库 `rl_lab` 的 `ActorCriticRecurrent`，actor／critic 各自维护单层 256 维 GRU，不再调用外部 RSL-RL runner 或配置类；
- policy 原始输入从 102 维两帧展平改为 51 维单帧；
- decoder 建成 `51→128→GRU(128)` 和 velocity／mass／force 三个 head；
- 牵引力使用机器人机体系 `Fx/Fy`，不预测小车速度、位置、绳长或绳参数；
- 提供 56 维 actor 拼接、PPO 梯度隔离、force-weighted mass supervision 和逐环境 hidden reset 的模块契约。
- `TowingOnPolicyRunner` 已把 decoder／actor／critic 三套 GRU、critic-only normalizer、rollout 固化、PPO 后 decoder 更新及联合 checkpoint 接成闭环；`TowingVecEnvWrapper` 负责读取 `policy/critic/decoder` 三组 observation。

## 待实现／待验证

- 在训练机运行自有 runner，验证三套 hidden state、按 done 切断 decoder 序列、critic-only normalizer以及 decoder／optimizer checkpoint 恢复；
- decoder＋actor 的联合导出及部署逐环境 reset；
- Isaac Lab 环境构造、scripted policy、4／256 环境 rollout 和张量接口验证；
- 根据真实 `v/F/m` 分布确定归一化尺度、`F_min` 和三项 loss 权重。

当前任务仍不注册，不把静态配置改造记为运行通过。

## 离线验证

- `compileall` 通过；
- 上层拖曳契约 16 项通过、2 项因当前解释器没有 PyTorch 跳过；
- 全部 `test_towing*.py` 为 191 项通过、12 项按可选运行环境跳过；
- `git diff --check` 通过。

Decoder 的张量数值、RSL-RL recurrent minibatch 和 Isaac Lab rollout 仍须在带 PyTorch／Isaac Lab 的训练环境验证。
