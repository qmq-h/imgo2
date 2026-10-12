# AMP 采样与时间步长检查（2026-10-10）

## 范围与结论

针对当前本机 AMP Height 源码、21 个内置动作文件、旧 24000 轮保存配置和部署配置检查。未访问服务器，不能用本机源码代替服务器正在运行的 env.yaml、已加载模块或新 checkpoint 对应配置。

主要时间链路一致，未发现把 200 Hz 物理步长当作 50 Hz 策略间隔传给专家采样的问题：

| 环节 | 实际设置／源码关系 |
|---|---|
| Isaac Lab 物理步长 | sim.dt=0.005 s，200 Hz |
| AMP Height 动作保持 | decimation=4，每个动作施加 4 个物理步 |
| 策略与环境步长 | step_dt=0.02 s，50 Hz |
| AMP policy 相邻状态 | runner 每次 env.step 取前后 AMP 状态，隔 0.02 s |
| AMP expert 相邻状态 | AMPLoader(time_between_frames=self.env.dt)，wrapper.dt→env.dt→step_dt，隔 0.02 s |
| 参考状态初始化 loader | 使用 env.step_dt，不使用 physics_dt |
| 动作文件 | 21/21 的 FrameDuration=0.02 s，50 Hz |
| 接触传感器更新 | update_period=sim.dt，即 200 Hz；track_air_time=True |
| PPO/AMP rollout | 每环境 24 步=0.48 s，不是 24 Hz |
| 部署策略周期 | base.yaml dt=0.005、decimation=4，即 0.02 s/50 Hz |

旧 `2026-09-24_00-57-38_xiaoji_real1/params` 同样记录 dt=0.005、decimation=4、num_steps_per_env=24。Fanziqi 任务单独覆写 decimation=6，策略 0.03 s，其专家对按对应 env.dt 在 0.02 s 数据中插值，不能把这两个任务的设置混用。

`AmpManagerBasedRLEnv.step()` 在 decimation 循环中每次 scene.update(physics_dt)，奖励按 step_dt 结算，终止 AMP 状态在 reset 前捕获。runner 用 terminal_amp_states 替换终止环境的 next_amp_obs，未发现把 reset 后的随机参考姿态直接作为该步真实后继送给判别器。对目标 0.5–0.6 秒周期，50 Hz 给每周期 25–30 个策略步；目前 0.29 秒周期约有 14.5 个策略步，没有观察到足以证明控制采样率导致该高步频的时间尺度错误。

## 已发现并修正：专家帧索引的 N / (N−1) 错误

`rl_lab/datasets/motion_loader.py` 用 `(N−1)×FrameDuration` 定义轨迹时长，四个按时间取帧的方法却使用 `frame_position = time / length × N`。正确帧位置为 `time / FrameDuration`（等价于 `time / length × (N−1)`）。

旧结果：

- 250 帧运动片段有 249 个时间间隔，专家动作加速 250/249，即约 0.402%；249 帧片段约 0.403%；100 帧站立片段约 1.010%。
- 0.02 秒专家时间差实际对应约 0.0200803 秒的录制相位推进（250 帧）；不会产生 2–3 倍的步频变化。
- 在最后时刻索引会到 N，超出有效末帧 N−1；此前时间采样人为多避开一帧，掩盖了常规预加载中的端点越界，但公共取帧 API 仍不正确。
- 随机时间原用 `max(0, uniform×length−(transition_dt+frame_dt))`，人为在零时刻堆积概率质量，约为运动片段 0.8%、站立片段 2.0%（dt=0.02）；并非真正的可用时间区间均匀采样。

当前统一使用 `_frame_indices()`，按每条轨迹 FrameDuration 计算帧位置，夹紧到 [0,N−1]，标量／批量／完整帧／AMP帧共用。采样改为可用起点 `[0,max(length−transition_dt,0)]` 内均匀抽取，后继时刻不越过末帧。支持 Fanziqi 的 0.03 s 相邻时间差；不会重写数据、checkpoint 或改变动作网络结构。

## 验证与复现

标准库脚本 [check_amp_timing.py](../imgo2_rl/scripts/tools/check_amp_timing.py) 从实际配置 AST 读取数值、汇总动作元数据并比对部署周期，输出 [JSON](amp_timing_audit_2026-10-10.json)：

```bash
python3 imgo2_rl/scripts/tools/check_amp_timing.py \
  --out docs/amp_timing_audit_2026-10-10.json
```

本机 CPU 测试直接执行真实 loader 类与 quaternion helper（只绕过 Isaac Sim 包初始化，不替代插值算法），使用 Python 3.11 的 env_isaaclab、NumPy/PyTorch/pybullet：

```bash
/home/tommy0929/miniconda3/envs/env_isaaclab/bin/python -m unittest discover \
  -s imgo2_rl/tests -p test_amp_motion_timing.py -v
```

4 项通过：标量／批量的整数帧、半帧和边界；1000 个预加载 ramp 状态对相差精确 0.02 s；均匀时间采样没有零点钳制峰；全部 21 条实际动作的第 0/1/末帧正确取回。针对 HEAD 原版 loader 的负向运行出现 23 个断言失败和 22 个子测试错误（包含末帧越界），确认回归能检出原问题。既有 AMP 配方契约 6 项通过，git diff --check 通过。

## 待运行验证与建议

已修复并 CPU 验证：专家时间插值与边界、采样分布；主频率链路已完成源码／元数据检查。尚未用修正后的 loader 重新跑 Isaac Lab 环境或短训练，也未证明修正能改善步态，需要服务器同步后重启进程、重建 expert 预加载，并跑 4 环境×10轮验证。当前正在运行的训练不会自动载入本机修复。

不建议为了减慢步频直接改变 sim.dt 或 decimation，这会同时改变动作保持、专家状态对、奖励换算及折扣的物理时域。rollout 的 0.48 s 小于目标单周期不构成必须加长的错误：价值函数跨 rollout bootstrap，24 步不是步态周期上限。之后可以把 num_steps_per_env=48 作为单独训练实验，但它同时改变每轮样本量和优化批量，不属于本轮必要修复。

判别器只看两个相邻状态（0.02 s 间隔），没有显式观察完整 0.5–0.6 s 周期。这是对节律约束可能偏弱的结构因素，而不是已证实的采样频率错误；与滞空项增权、gradient penalty 消融分开评估。当前先保持物理 200 Hz、策略 50 Hz，确认修正后的奖励快照和实际接触周期，再决定训练调参。没有启动长期训练、部署或真机操作。
