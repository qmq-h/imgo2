# 上层拖曳 RL 链路修复（2026-09-22）

## 已修复

- 安全信号不再是常量：用机器人 base 后表面到车斗前表面的有向纵向间隙写入 `rope_state[:,0]`。车斗和四个车轮各用一个 one-to-many ContactSensor，只读取对机器人各刚体的 `force_matrix_w`，因此既覆盖车斗／车轮撞腿，又排除正常轮地接触；最大配对接触力超过 1 N 时触发碰撞 reward 和 termination。
- actor observation 保留完整 51 维单帧，并新增部署可得的 `reference_command`；PPO actor／critic 改为各自独立的单层 256 维 GRU，不再由环境展平两帧历史。
- 非对称 action 使用以零为中心的分段缩放；归一化零动作现在对应零加速度。前 1 s settle 阶段强制参考指令为零。
- 子环境 reset 只清对应环境的底层 `last_action`，同时清 `rope_state`；绳模型改为逐环境 Bernoulli 采样，singleton 异步 reset 不再固定落入 compliant。
- PPO 最初按本机较新源码补了 `obs_groups` 和 `RslRlRNNModelCfg`；用户随后确认训练机是 Isaac Lab 2.2.1／RSL-RL 2.3.3，并决定沿用 AMP 的仓库自有算法模式。现已改由 `rl_lab.TowingOnPolicyRunner` 管理 recurrent PPO、decoder 与 critic-only normalizer，上层配置不再导入外部 RSL-RL runner/config；尚未在训练机运行。
- `extra_distance` 改为只在 `t ≥ t_stop` 后生效，不再把最初 1 s 站定阶段误判成停车后滑行。
- 修正冻结 AMP 的频率契约：训练与部署均为物理／关节控制 `0.005 s`、`decimation=4`，所以策略推理周期是 `0.02 s`（50 Hz），不是 `0.005 s`（200 Hz）。`tow_drag.py` 现在会按 4 个物理步调用一次策略；上层 action term 启动时还会交叉核对环境物理周期、上层 20 Hz 周期和底层 50 Hz 周期。
- 增加无小车零负载环境：每次 reset 默认以 `12.5%` 概率采样，256 环境时期望约 32 个。由于 Isaac Lab scene 仍会复制 cart articulation，无小车实例把小车横向停放 2 m，并通过 `cart_present` 屏蔽绳力、轮阻、碰撞／间隙 reward 和碰撞 termination。decoder 的速度／零力监督保留，质量监督权重自然为零；critic 读取存在标志，actor 不读取。
- decoder 改为 `51→128→GRU(128)`，使用 velocity／mass／force 三个 head 预测机器人 `vx/vy`、小车质量和机体系牵引力 `Fx/Fy`。5 维 estimate detach 后与原始帧组成 56 维 actor 输入；预测误差不进入 reward。

## 验证

使用 Windows Python 3.14.6：

- `test_towing*.py`：199 通过，12 项因缺 PyTorch／Isaac Lab 等可选环境跳过；
- 上层契约专项：14 项中 12 通过，2 项因缺 PyTorch 跳过；
- `compileall`、`git diff --check`、资源路径检查、模型同步检查通过；
- `git ls-files -i -c --exclude-standard` 无文件输出，仅因沙箱无法读取用户级 ignore 文件打印 warning。

## 待修复／待确认

- 当前机器没有 Isaac Lab／PyTorch 运行环境，尚未构造环境、执行单步 reward／termination，也未核对 ContactSensor 的实际张量和终止步计奖顺序。因此代码状态是“已修，待训练机验证”，任务继续保持未注册。
- 连续间隙是 base／车斗表面代理，不包含腿部 FK 几何。训练机需与 `tow_clearance.py` 的全腿 FK 间隙和 `tow_drag.py` 的车斗／车轮接触记录交叉验证；若误差会改变 reward 排序，应把完整几何 producer 迁入环境。
- `TowingDynamicsDecoder`、GT-force weighted mass supervision 和 optimizer 仍是独立模块，普通 `OnPolicyRunner` 不会调用。需要实现 rollout estimate 固化、三套 hidden state 的逐环境 reset、episode 序列切分和 checkpoint 保存／恢复。
- 注册前仍需单环境 scripted policy 复现 `tow_drag.py`，再做 4／256 环境短 rollout，检查异步 reset、NaN、绳模型比例和 GPU/CPU view 写入。
- 无小车实现仍需在训练机确认横向停放的小车不会跨环境接触，并统计 `cart_present` 比例、零绳力、零碰撞／间隙回报和零质量监督权重；若场景隔离不足，应改为运行时 collision disable，而不是继续增大停放距离。
- 当前 `curriculum=None`，速度、质量、摩擦、轮阻和两类绳从第一回合就覆盖完整随机域。先用单一绳和窄参数域完成短训练，再依据边界扫描逐步放宽；否则首次训练失败时难以区分环境错误、奖励问题和域随机化过强。
- `collision`／`fall` 的配置权重会被 RewardManager 再乘 `step_dt=0.05`，当前单次实际贡献各为 `-2.5`。这个量级是否足以压住提前终止，只能结合 scripted rollout 的逐项 episode return 判断；在获得该记录前不盲目改权重。
