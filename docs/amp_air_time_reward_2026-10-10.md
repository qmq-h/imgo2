# AMP Height 滞空奖励启用记录（2026-10-10）

## 同日全部奖励与速度范围复核

核对当前源码中的 AMP Height（不是 Fanziqi、不是 height_move PPO），控制步长 0.005 × 4 = 0.02 s。任务共 7 项：

| 项 | RewardManager 权重 | 每步系数 | 参数／意义 |
|---|---:|---:|---|
| track_lin_vel_xy_exp | 200 | 4 | `std=0.5`；平面速度误差的指数核，直立门控 |
| track_ang_vel_z_exp | 100 | 2 | `std=0.5`；偏航角速度误差的指数核，直立门控 |
| base_height_l2 | -250 | -5 | 机身高度目标 0.30 m；平方误差 |
| lin_vel_z_l2 | -50 | -1 | 垂直速度平方 |
| ang_vel_xy_l2 | -2.5 | -0.05 | 横滚／俯仰角速度平方，直立门控 |
| joint_pos_limits | -100 | -2 | 超出软关节限位的量；模型 soft_joint_pos_limit_factor=0.9 |
| feet_air_time | 50 | 1 | 落地事件 ×（滞空时间−0.3 s），详见下文 |

AMP 风格奖励是 `2 × max(1−(d−1)^2/4, 0)`，每步范围 0–2；混合为 `0.7 × 风格 + 0.3 × 任务`。0.7/0.3 是系数，不代表实际回报占比。动作变化、关节力矩／加速度、足端高度／滑动／滞空方差／相位、平躺姿态和终止惩罚等任务奖励均被保留名单过滤；base 触地依然是终止条件。

训练指令为身体坐标系 `vx∈[-1.0,1.5] m/s`、`vy∈[-1.0,1.0] m/s`、`wz∈[-1.57,1.57] rad/s`。每 10 s 重新采样，站立采样比例 0.02；自定义指令生成器把平面指令范数 ≤0.2 m/s 的 vx/vy 清零（可保留转向），速度课程关闭。平地不受坑地形的额外限制。play 配置固定为 vx=1.0 m/s、vy=0、wz=0，专用评估脚本可再次覆盖。

已确认：7 项保留名单与参数、继承的时间步长和命令生成逻辑、play 的最终覆盖、风格混合公式；6 项配置检查通过。用户指出滞空与 tracking 差距后，发现初始实现中 tracking 抵消了 RewardManager 的 dt，而滞空未抵消；已将滞空也换算为每步／事件系数。在当前步长下扩大 50 倍。例：两足同时落地且各自滞空 0.1 s 时，混合后的事件贡献由 `−0.0024` 变为 `0.3×2×(0.1−0.3)=−0.12`，两项完美速度跟踪合计为 1.8。事件只在落地时发生，连续跟踪每步发生，平均贡献仍不可按事件峰值等同；训练效果待确认。

## 现象与目标

用户训练的是 `base_move_amp_height`，希望启用滞空时间奖励以缓解高频小步，并希望完整步态周期接近 0.5–0.6 s（约 1.67–2 Hz）。这与单腿的滞空时间不同：周期还包含支撑时间。

## 已修改

`imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/amp_env_cfg.py` 的 `Imgo2AmpMoveEnvCfg._keep_only_amp_task_rewards()` 原先过滤掉了共用的 `feet_air_time`。本轮将该项加入保留名单，并设置：

```python
FEET_AIR_TIME_PER_STEP = 1.0
self.rewards.feet_air_time.weight = FEET_AIR_TIME_PER_STEP / step_dt
self.rewards.feet_air_time.params["threshold"] = 0.3
self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
```

阈值沿用 [历史方案](amp_gait_adjust_plan_2026-09-18.md) §9 记录的用户选择。历史初版直接沿用 PPO 原始权重 1.0，未与本 AMP 的每步 tracking 系数统一，导致额外乘 0.02；本轮修正这一尺度遗漏。当前 raw weight=50，RewardManager 乘 dt 后有效系数=1，AMP 混合再乘任务比例 0.3。这里的 1.0 是新的事件系数初值，不是已经验证的最优权重；改变 dt 时该换算保持单次事件系数，但不意味着每秒奖励或步态天然与 dt 无关。

沿用 `mdp/rewards.py::feet_air_time`：仅在首次落地时结算，短于阈值给负值，长于阈值给正值；低速命令时关闭，并带直立姿态门控。这是鼓励较长滞空的软奖励，没有周期误差项，因此不会精确锁定目标周期。站立占空比约 0.5 时，0.3 s 滞空对应约 0.6 s 周期；占空比改变时这一关系也改变。

修改同时适用于 AMP Height 的 play 配置。Fanziqi 子类后续调用 `apply_rlamp_task_rewards()`，仍只保留速度跟踪项。共用奖励公式未改变。

## 宇树公开代码核对

2026-10-10 读取 [宇树官方 unitree_rl_lab 的 Go2 环境配置](https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py)：`feet_air_time.weight=0.1`、`threshold=0.5`、足端正则 `.*_foot`。该值是滞空阈值，不是 0.5 s 的完整步态周期；该配置是 Go2 速度任务，不是当前 Imgo2 AMP Height 的奖励尺度，不能把绝对权重当成通用最优值。本轮选择 1.0/0.3 是本项目的实验起点，并非照搬宇树官方参数。

## 验证与待确认

已修复并离线验证：奖励保留名单包含滞空项、正权重与正阈值生效，足端正则在当前训练 URDF 中恰好匹配 FL/FR/RL/RR 四个 FOOT link；后续 Fanziqi 筛选只保留两项速度奖励。新增标准库配置执行检查，直接执行源码中的筛选函数（奖励项用轻量对象代替），并解析真实 URDF 校验足端名称。

使用系统 `python3`（3.10）执行：

```bash
python3 -m unittest discover -s imgo2_rl/tests -p test_amp_recipe_contract.py -v
```

6 项通过；修改后的 AMP 配置通过 Python `compile()` 语法检查；`git diff --check` 通过。初版启用前曾 `git pull --ff-only` 确认当时 HEAD `1c6f6c9`，本次尺度修正基于当前本地 `fc8b341`，未 fetch/pull。尺度回归检查直接执行实际配置，在控制步长 0.01、0.02、0.04 s 验证同一双足落地事件贡献为 −0.4、完美平面 tracking 为 4.0，比例 −0.1；旧版在此检查中失败。

待修复／待确认：尚未启动 Isaac Lab 环境、短训练或完整重训，未证明滞空统计在运行中正常，也未证明周期达到 0.5–0.6 s。需要在训练机重启训练进程，确认保存的 `params/env.yaml` 和 TensorBoard 的 `Episode_Reward/feet_air_time` 包含新项；再在固定速度（建议先 vx=0.5 m/s）下回放统计周期、滞空时间、步幅、速度误差及机身高度。已运行的训练进程和现有导出策略不会因源码改变而自动更新。建议新建训练运行与旧奖励作对照，正式导出继续使用对应 `play.py`。
