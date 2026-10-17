# 拖曳 CE Net 结构调整（2026-10-17）

## 现象与根因

用户明确要求 CE Net 的上下文编码：观测历史经 GRU 并列得到显式动力学估计与变分隐向量，两者共同重建下一时刻的 58 维观测。原实现却将 GRU 输出全部压到 16 维变分 latent，再从 latent 回归物理量；没有下一帧重建分支，显式量也不是独立编码头。

## 已修复的代码／待验证的行为

- `TowingDynamicsDecoder.encode` 返回 `estimate, mu, log_var, hidden`。显式头直接读取 GRU hidden，保持 `[vx, vy, mass, Fx, Fy, Fz]` 6 维；16 维 latent 的均值与方差由另两个并列头输出。
- `decode(estimate, latent)` 拼接 22 维编码，经 MLP 输出下一时刻完整的 58 维 policy 帧。重建梯度不 detach，回传显式分支和 latent 分支；KL 只作用在 16 维 latent。
- 推理仍通过 `forward_with_latent` 返回 6 维显式量、16 维 latent 和 GRU 状态。rollout 与确定性推理使用 latent 均值，估计网络训练使用重参数采样；推理不调用重建分支。
- Actor 继续拼接 `frame(58) + estimate(6) + latent(16) = 80`，后两者 detach，PPO 不训练 CE Net；critic 73、动作 13、控制频率及物理监督定义保持原契约。

损失为：

```text
L_CE = L_est + reconstruction_coef * MSE(reconstruct(e_t, z_t), o[t+1])
             + kld_weight * KL(q(z_t | history) || N(0,I))
```

`L_est` 沿用速度／拉力／质量三项 Smooth L1：注册配置权重分别为 5／10／1，质量沿用已有交互监督权重。重建使用现有 policy 帧缩放后的全部 58 维，不另加归一化；MSE 对有效 transition 和维度取均值。KL 对 latent 维求和，再对时间／环境取均值。`reconstruction_coef=1.0`、`kld_weight=0.005`；前者在通用与注册配置中均显式提供。这些权重尚未实跑标定。

## 时序与回合边界

runner 在 `env.step` 前保存 `o_t` 和同拍的物理监督，step 后保存返回的 `o[t+1]`。包括 rollout 最后一拍，不通过 shift 丢弃尾部。所有缓存 clone，防止环境复用 observation／supervision buffer 后覆盖历史。

当该步 terminated 或 truncated 时，返回值可能是自动 reset 后的新回合首帧。因此只对 `~dones[t]` 计算该步重建损失；当前时刻物理监督与 KL 保留。重算序列时，在读取 `o[t+1]` 前按 `dones[t]` 将结束环境的 GRU hidden 清零。整批全为终止步时重建损失精确为零，仍可训练估计与 KL。

## checkpoint 与日志

训练 runner、纯 torch 推理运行时及必要性测试台统一为 `towing_contract.version=4`。维数仍是 58／6／16；由于权重结构改变，旧 v3 即使维数相同也会在契约校验处拒绝，必须新建 run，不允许修改旧 checkpoint 元数据冒充 v4。新 v4 权重的实际 save/load 数值检查待训练机。

新增 TensorBoard `Loss/decoder_reconstruction` 与 `Loss/decoder_weighted_reconstruction`；标准控制台显示重建损失，原三项估计损失和 KL 日志保留。历史 v3 实验与聚类产物仍是旧网络结果，不能当作新 CE Net 的验证依据。

## 验证与限制

本机解释器：`C:\Users\qmq\AppData\Local\Programs\Python\Python311\python.exe`，缺 torch、NumPy、pytest 和 Isaac Lab。用户要求完成代码即可，不在本机安装训练依赖或运行训练。

- 修改的 Python 源码与测试共 10 个文件 `compileall` 通过；`git diff --check` 通过；`git ls-files -i -c --exclude-standard` 为空。
- `python -X utf8 -m unittest discover -s imgo2_rl/tests -p test_towing_ce_net.py -q`：7 项，1 通过、6 项缺 torch 跳过。
- 同样执行 `test_towing_upper_rl_contract.py`：63 项，55 通过、8 跳过；`test_towing_play_test.py`：151 项，144 通过、7 跳过。合计 **200 通过／21 跳过**，不能把跳过项记作网络数值验证。
- 早先尝试全量 unittest 收集得到 556 项，5 个错误、72 跳过：NumPy 缺失引发 2 处错误、torch 缺失使 runtime 的 2 条未跳过检查访问 None、默认 GBK 控制台导致聚类输出编码错误。后续相关检查使用 `-X utf8`；没有据此修改无关代码或宣称全量通过。
- 新增 6 项张量回归测试覆盖：并列头与重建梯度、显式量不依赖 latent 采样、下一帧损失与 reset 目标屏蔽、全终止 batch、GRU 按环境重置、真实 runner loop 对共享缓冲区的时序快照。均留待训练机运行。

## 待修复／待确认（TOW-DEC-01）

**代码已调整，网络数值与运行验证待完成。** 需要训练机 torch 执行上述张量测试与 `test_towing_upper_policy_runtime.py`（包括 v3 拒绝和 v4 加载），再在 Isaac Lab 做短训练，核对三类损失有限且有梯度、重建确实对应下一帧、done／timeout 后 hidden 重置与重建掩码、新 checkpoint 保存恢复。最后根据真实读数标定重建与 KL 权重；本轮不启动短训练、长训练或硬件任务。
