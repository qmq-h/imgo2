# AMP 配置收缩记录（2026-09-22）

## 结论

当前只注册两套平地 AMP 任务：

1. `Imgo2-basemove-flat-amp-height`：保留 Imgo2 的高度和轻量姿态奖励，使用 `AMPHeightRunnerCfg`。
2. `Imgo2-basemove-flat-amp-fanziqi`：对齐 Fanziqi A1 AMP 配方，使用 `FanziqiAMPRunnerCfg`。

两套任务各保留一个 `-play` 入口。原 amp_go2、粗糙地形 rl_amp 任务及其环境类、runner 和专用测试已删除。

## Fanziqi 配方

Runner 按本机参考 `C:/Users/qmq/Desktop/RL/AMP/AMP_for_hardware-main/legged_gym/envs/a1/a1_amp_config.py` 对齐：24 步 rollout、500000 轮、每 50 轮保存、PPO 网络 `[512,256,128]`、`coef=2.0`、`lerp=0.3`、判别器 `[1024,512]`、预载 2000000、`min_normalized_std=[0.05,0.02,0.05]×4`，并启用 std 下限。

环境使用 `sim.dt=0.005 s`、`decimation=6`，只保留参考原始权重 50／16.6667 的线／角速度任务奖励；关闭后来加入的持续外力。actor 同参考移除基座线速度和角速度，变为 42 维。参考状态初始化概率保持 0.85，剩余 0.15 使用普通 reset。

机器人模型、关节名称、运动数据和执行器物理仍使用 Imgo2；它们不是 A1 配置可以原样替换的部分。该任务必须新训，45 维旧 checkpoint 不能加载到 42 维 actor，也不应覆盖当前部署的 45 维策略配置。

## 验证与限制

- AMP 离线测试：15 项中 14 通过、1 项因缺训练环境依赖跳过。
- 新的两套配置契约测试：5 项全部通过。
- `compileall` 与 `git diff --check` 通过。
- 尚未在 Isaac Lab 中构造两套环境或运行短训练，因此状态为“配置已清理，待训练机验证”。

训练机应分别运行 256 环境、100 轮短训练，确认 observation 维数为 45／42、Fanziqi 任务没有持续外力、reset 比例正确，并检查保存间隔和 std 下限日志。
