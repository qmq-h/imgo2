# IMU 启动零位偏置 + 策略输入对应性核对（2026-10-23）

> 需求（用户）：通读 IMU 代码，确认策略输入的状态/单位是否完全对应；然后让程序每次启动连续读 50 次
> IMU，把这段当成偏置消掉，使每次上电 RPY 接近 0，并应用到 `rl_real_imgo2` 与 `go_monitor`。
>
> 本轮改了 `ImuMonitor`（两条入口的公共路径）与一条显示器回归断言；实机未运行，PTY 端到端已验证。
>
> **2026-11-08 修正**：原 `q_corr = q ⊗ conj(q_bias)` 是机体系旋转，会把启动 roll/pitch 混入策略 gravity；
> 已改为 quaternion 保持原始、只对显示 Euler 归零。见下面“2026-11-08 修正（乘法顺序）”。

---

## 1. 策略输入链路（代码位置）

| 步骤 | 位置 | 内容 |
|---|---|---|
| 1 | `Imu/src/yesense_parser.cpp` | 0x20 角速度 deg/s → rad/s；0x41 四元数 4×int32×1e-6 → 归一化 wxyz；0x40 Euler 线序 pitch/roll/yaw → 存 (roll,pitch,yaw) rad |
| 2 | `Imu/src/imu_monitor.cpp` | 后台线程读串口、发布 `MonitorSnapshot`（本轮在此加启动偏置） |
| 3 | `hardware/real_hardware.cpp` | `snapshotLocked()` 取 `imu_->Snapshot()`，判 quat+gyro 同帧、age<`stale_after_ms` |
| 4 | `hardware/robot_adapter.cpp` | `toPolicy()`：quat/gyro 原样进 `RobotState`，关节按 `policy_to_hardware` 重排 |
| 5 | `sdk/rl_sdk.cpp` `buildObservation()` | 拼 45 维观测并裁剪 |
| 6 | `sdk/policy_runner.cpp` `infer()` | 前向 + `clipActions` + `convertAction`（`default_pose + action_scale·a`） |

45 维顺序（与 `policy/ppo/config.yaml` 的 `observations` 一致）：

| 项 | 维数 | 来源 | 部署缩放 | 训练侧对照 |
|---|---:|---|---:|---|
| `ang_vel` | 3 | IMU gyro，rad/s，机体系 | 0.25 | 参考项目 base_move `0.25` |
| `gravity_vec` | 3 | `projectedGravity(quat)` = R(q)ᵀ·(0,0,−1) | 1.0 | Isaac Lab `quat_rotate_inverse(quat,(0,0,−1))`，逐项一致 |
| `commands` | 3 | vx,vy (m/s), wz (rad/s) | [1,1,1] | 同 |
| `dof_pos` | 12 | (q − default_pose)，rad | 1.0 | 同 |
| `dof_vel` | 12 | dq，rad/s | 0.05 | 同 |
| `last_action` | 12 | 上一拍裁剪后动作 | 1.0 | 同 |
| 总裁剪 | | `clip_obs = 100` | | 观测项 clip ±100 |

**结论：状态、单位、顺序、缩放都对应**（`policy_to_hardware` 恒等，硬件顺序 FL/FR/BL/BR = 训练的 FL/FR/RL/RR）。
以下是**不对应/未覆盖**的点，均未在本轮改：

1. `ImuConfig::mount_quaternion` 只做校验、**从不施加**；gyro 没有任何安装旋转。283 s 日志里上电 yaw 偏置约
   **+111.8°**、roll/pitch 约 +0.9°/−1.6°——yaw 这么大只可能是传感器参考系（磁/陀螺积分起点），不是物理装歪，
   所以**不能**把这段旋转施加到 gyro（否则 x/y 角速度会被转 111.8°）。本轮只消显示 Euler，quaternion 保留原始，不动 gyro。
2. gyro 零偏未消除（用户本次只要求 RPY）。`go_monitor` 面板能看到静止时的 gyro 值，若明显非零可另加。
3. 加速度计解析但策略不用；Euler 只给面板显示。
4. `captured_at` 是主环时刻而非传感器采样时刻；`obs_age` 结构上恒 0（已知 DEPLOY-DIAG-01）。
5. 训练侧 `velocity_env_cfg.py` 的 `base_ang_vel/joint_vel` scale 是 1.0，与部署的 0.25/0.05 不同——因为部署的
   `policy_flat.pt` 来自参考项目（25/0.5、0.25/0.05）。**换成本仓库训练的策略必须同步换这组 scale**（README 5.3 已记）。

---

## 2. 启动偏置（本轮改动）

`ImuMonitor` 是 `rl_real_imgo2`（经 `RealHardware`）与 `go_monitor`（直接）共用的唯一 IMU 入口，所以改这里
两条入口同时生效，不需要分别改。

- 头 `Imu/inc/imu_sensor/imu_monitor.hpp`：`kBiasSamples = 50`，累加/校正成员，`MonitorSnapshot` 增加
  `bias_samples` / `bias_ready` 两个只读状态。
- 源 `Imu/src/imu_monitor.cpp`：
  - 前 50 个带四元数的有效帧：统计并按帧求 Euler 均值；**quaternion 不修改**。
  - 之后每帧：只对显示 Euler 逐项减均值并按 ±π 回绕。**quaternion 和 gyro 都不修改**。
  - 50 帧之前**照常发布原始帧**（因此 `RealHardware` 的 `imu_ready_timeout_ms` 与 PTY 测试都不被拖慢）；
    第 50 帧起发布已校正帧。
- 测试 `tests/test_combined_monitor.py`：假 IMU 发恒定 Euler (1.182, −0.307, −57.670)°，校正后应显示
  `0.000 0.000 0.000`，断言已同步。

**2026-11-08 修正（乘法顺序）**：原先的 `q ⊗ conj(q_bias)` 是**机体系**旋转，不是世界航向归零。
数值验证（初始 yaw −64.615°，随后机体 roll 10°）：

| 处理 | 机体系 gravity |
|---|---|
| 原始 q / 正确左乘 `conj(q_yaw)⊗q` | `[0.000, −0.1736, −0.9848]` |
| 原右乘 `q⊗conj(q_bias)` | `[−0.1569, −0.0744, −0.9848]` |

右乘把启动 roll/pitch（且 `q_bias` 是完整姿态均值，不只是 yaw）混进了重力方向，而 gyro 仍是原始轴；
策略的 gravity 与 gyro 因此不自洽。正确做法是：**quaternion 保持原始给策略**；显示 Euler 单独归零；
gyro 不旋转。如果以后确实需要 quaternion 本体的航向归零，只能左乘 `conj(q_yaw0)`（仅 yaw），
不能用完整 `q_bias`。

**为什么 warm-up 期间发布原始帧**：`RealHardware::start()` 在 `imu_ready_timeout_ms`（默认 2 s、PTY 测试
180 ms）内要有有效帧；若先阻塞采集 50 帧（200 Hz 下 250 ms）再发布，PTY 测试会直接判 IMU 超时。
warm-up 期间机器人处于 InitialHold、策略未激活，原始姿态不影响控制。

---

## 3. 验证

- 构建：`cmake --build build -j16`，无 error；另用 `-Wall -Wextra -Wpedantic` 单独编译 `imu_monitor.cpp` 干净。
- 全量 `ctest --test-dir build -j1`：**28/29 通过**；唯一失败 `rl_relocation` 是既有问题
  （`cmake --install` 不打包 libtorch `.so`），与本轮无关。`fsm_check` 本轮通过（历史上高负载时会偶发）。
- 端到端（PTY，假 IMU 恒发 roll=10°、pitch=−5°、yaw=30°，gyro=(0.1,0.2,0.3) rad/s）：
  `rl_real_imgo2_pty --log` 的 `robot.csv` 显示

  | 行 | RPY(°) | gravity | gyro |
  |---|---|---|---|
  | 0–10（前 49 帧） | 10.000 / −5.000 / 30.000 | (−0.0872, −0.1730, −0.9811) | 0.100/0.200/0.300 |
  | 60 起（第 50 帧后） | **0.000 / 0.000 / 0.000（显示）** | **(−0.0872, −0.1730, −0.9811)（raw q 不变）** | **0.100/0.200/0.300（未变）** |

  退出码 0，无 Fault。⇒ **显示 Euler** 被归零；quaternion 保持原始（测试里 raw q 对应上面的 gravity），
  gyro 未被误转。

---

## 4. 限制与待确认

- **显示 Euler 把上电姿态当零，策略 gravity 用原始 q**：如果上电不水平，面板 RPY 会被显示成 0，
  但策略 gravity 仍是真实姿态；这是有意的（策略不依赖显示零点）。283 s 日志里 InitialHold 与 StandHold 相差
  roll ≈0.5°、pitch ≈0.7°，站起后显示残余 <1°。
- **前 50 帧（200 Hz 下约 250 ms）内必须静止**；期间抬动机器人会把运动当作零位。
- 实机未运行：真实 IMU 的 50 帧耗时、上电残余倾角、PPO 静态/行走表现都待实机确认。
- gyro 零偏、安装旋转仍未处理（见 §1 第 1、2 条）。
