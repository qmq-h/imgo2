# 部署线程/观测时序复审：5 ms 主环到底加了多少延迟（2026-10-23）

> 触发问题（用户）：主线程 5 ms 更新，也在主线程给策略更新观测；观测不是应该直接拿最新值吗，
> 这 5 ms 会不会造成延迟？另外线程安全有没有问题？仿真不抖、实机抖，跑一段时间后关节抖得更厉害。
>
> 本轮**只读代码 + 读已有实测记录**，未改代码、未编译、未跑硬件/仿真。文中"确认"=源码可直接读出；
> "推断/待测"=需要下面的最小验证集才能落定。基线：`d30af52`（最新一次真机日志对应的流水分支）。

---

## 0. 结论（先看这三条）

1. **不是"额外 5 ms 排队"**。`InferenceWorker` 每次醒来取的是**最新一次控制拍提交的快照**，没有积压；
   所以 5 ms 主环给观测带来的是**采样相位**（0.2–4.9 ms，均值 2.5），不是固定多 5 ms。
2. **但两个 5 ms 周期相等的时钟自由运行、相位不受控**，于是"反馈解析完成 → 主环读到"与
   "主环发布目标 → 总线取走"这两段**反相关，二者之和二峰分布**：要么约 **2 ms**、要么约 **7 ms**
   （§2 有推导）。再加上异步推理线程固定要**跨一拍交接（约 4.8 ms）**，整条
   "状态 → 力矩" 的调度开销就是 **9–14 ms**；而策略零阶保持本身是 **0–20 ms（均值 10）**。
   这和已量化的 `q_des`↔反馈估算力矩 7.5–10 ms 对齐（[关节动态](joint_feedback_dynamics_2026-10-21.md) §4）
   是同一量级，但**不能把两边数字直接相加**（口径不同）。
3. **线程安全没有找到会直接引起抖动的数据竞争**；确定的问题集中在诊断路径（trace 环形缓冲）
   和"新鲜度信号被丢掉"这条逻辑上。后者会把每 20–70 s 一次的 15–25 ms 停顿**变成策略看不见的开环一拍**，
   这是仿真里绝不会出现的东西，优先级高于 5 ms 本身。

---

## 1. 数据链与代码位置（确认）

主环 5 ms（`control_ms=5`，[deployment_config.h:8](../../imgo2_deploy/real/config/deployment_config.h#L8)）：

| 顺序 | 代码 | 内容 |
|---|---|---|
| 1 | [rl_real_imgo2.cpp:303](../../imgo2_deploy/real/rl_real_imgo2.cpp#L303) | `now = Clock::now()`，本拍时基 |
| 2 | [rl_real_imgo2.cpp:308](../../imgo2_deploy/real/rl_real_imgo2.cpp#L308) | `raw = hardware.readState()`（拷贝总线线程发布的反馈 + IMU 快照） |
| 3 | [rl_real_imgo2.cpp:337](../../imgo2_deploy/real/rl_real_imgo2.cpp#L337) | `latest = worker.latest()`（**拷一份**推理结果） |
| 4 | [rl_real_imgo2.cpp:338](../../imgo2_deploy/real/rl_real_imgo2.cpp#L338) | `fsm.update(raw, latest, now)` 得到目标 |
| 5 | [rl_real_imgo2.cpp:353](../../imgo2_deploy/real/rl_real_imgo2.cpp#L353) | `hardware.sendCommand(target)` → 写总线命令槽 |
| 6 | [rl_real_imgo2.cpp:356-357](../../imgo2_deploy/real/rl_real_imgo2.cpp#L356-L357) | `adapter.toPolicy(raw)` → `captured_at=now` → `worker.submit(...)` |

推理线程 20 ms（`policy_ms=20`，[inference_worker.cpp:25-55](../../imgo2_deploy/real/sdk/inference_worker.cpp#L25-L55)）：
醒来 → 锁内拷 `state_` → 判 `age<source_timeout_ms_`（=100 ms）→ `infer_()` → 发布 `result_` → 等到 `next`。

总线线程 5 ms（`period=5ms`，[real_config.h:64](../../imgo2_deploy/real/config/real_config.h#L64)）：
每轮取目标序号 → `setPD` → `cycle()`（一轮约 2.0–2.2 ms）→ 发布反馈 + `cycle_count`。

**关键事实**：`submit()` 每拍都推一份新快照，但 `InferenceWorker::run()` 里的
`cv_.wait_until(..., [&]{return stopping_||!active_||generation_!=generation;})`（第 52 行）
**谓词不含"有新状态"**，所以 `submit()` 的 `notify_all()` 只是把等待者唤醒再睡回去。
推理节拍严格由它自己的 `next += 20ms` 决定，**与控制拍不同相**。

---

## 2. 5 ms 到底加了多少：延迟预算（全是"由频率/相位推出的量"，本轮没有新测）

设：总线线程在 `n·5 ms` 醒来，`2.2 ms` 后**发布**反馈；主环在 `n·5 ms + φ` 醒来（`φ` 由开机决定，
`0 ≤ φ < 5`）；主环 body（读状态→FSM→提交）约 `0.9 ms`（**估计值**，未单独测）。

| 段 | 值 | 性质 |
|---|---:|---|
| 电机应答在适配器里攒包 | 1.59 ms | 固定，实测 13708 次（[项目验证总结 §8.7](../../imgo2_deploy/real/docs/xiaoji/项目验证总结.md)） |
| 解析完成 → 主环读到（`seg1`） | 0.2–4.9 ms | 相位决定 |
| 主环采样 → 推理线程取走 → 下一拍被消费（`seg2+seg3`） | **≈4.8 ms（正好一拍）** | 异步线程交接，固定 |
| 主环发布 → 总线取走（`seg4`） | 0.2–4.9 ms | 相位决定，与 `seg1` 反相关 |
| **`seg1+seg4`** | **≈2 ms 或 ≈7 ms（二峰）** | **差 5 ms，只由开机相位决定** |
| 推理本身 | 0.2 ms | [deployment.yaml](../../imgo2_deploy/real/config/deployment.yaml) 注释 |
| 主环 body | ~0.9 ms | 估计 |
| 电机内部 PD / 力矩建立 | 未标定 | 大反转实测 ~15 ms（§8.8），不是代码能定的 |

**推导**（`φ` 为相位，`2.2` 为一轮发布时间）：

- `seg1 = (φ − 2.2) mod 5`，`seg4 = 5 − ((φ + 0.9) mod 5)`
  ⇒ 两者之和 = `5 − 3.1 = 1.9 ms`（不跨发布时间）或 `5 − 3.1 + 5 = 6.9 ms`（跨）。
- `seg2 + seg3 ≈ 5 − 0.2 = 4.8 ms`：观测在拍 `k` 采样、提交，结果最早只能在拍 `k+1` 被 `latest()` 读到
  （本拍的 `latest()` 在 `submit()` 之前，见 [rl_real_imgo2.cpp:337-357](../../imgo2_deploy/real/rl_real_imgo2.cpp#L337-L357)），
  所以**这一项与相位无关，是一整拍**。

因此"状态 → 力矩"的代码侧开销 ≈ **1.6 + 0.9 + 4.8 + (1.9 或 6.9) ≈ 9.2 ms 或 14.2 ms**，
另有 20 ms 策略零阶保持和电机力矩建立。已量化的 `q_des(t−L)↔tau(t)` 的 L=7.5–10 ms
大约对应 `seg4 + 1.6 + 电机建立`，与 `seg4∈[1.9,6.9]` 相容；但
[关节动态](joint_feedback_dynamics_2026-10-21.md) 已说明不能把 L 逐项拆开。

**结论**：5 ms 主环本身**不是**"加 5 ms"，但它和总线同为 5 ms 且不同相，**值 5 ms 的反馈延迟是白送的**
（好相位 1.9、坏相位 6.9）；异步推理线程又白送一整拍 4.8 ms。

### 2.1 为什么"直接拿最新值"不会更快

`readState()` 是唯一读口，而总线线程本身就是 200 Hz 发布。**推理线程就算自己直接去读硬件，
拿到的也是同一份 200 Hz 快照**，最多省掉 `seg2` 的相位；而且会让 FSM 的安全判断和策略看到的
不是同一份快照（现在至少是同一份 `raw`）。所以"观测延迟"的根不在"通过主环转一手"，
而在**三个时钟没有对齐**。

### 2.2 真正值得做的（都不改策略、不改训练契约）

1. **推理改由控制拍触发 / 直接内联在控制拍里**：观测年龄 → 0，`seg2+seg3` 从 4.8 ms → 0.2 ms，
   一次性省掉 **约 4.6 ms**；代价是控制拍多 ~0.2 ms（预算 5 ms 里够）。
   这是对"5 ms 是否造成延迟"最直接的判决实验。
2. **控制拍由总线发布驱动**（等 `cycleCount` 变化再算，见 §3.2）：`seg1` → ≈0，
   把 `seg1+seg4` 从"1.9 或 6.9 的抽签"变成**确定**的 ~1.9 ms，消掉那 5 ms 抖动。
3. 剩下的 `seg4`≈1.9 ms 需要总线线程一有新目标就立刻补跑一轮，改动大，先不做。

两项都做：`1.6 + 0.9 + 0.2 + 1.9 ≈ 4.6 ms`（确定性），比坏相位的 14.2 ms 少约 **10 ms**。

## 3. 线程/逻辑问题清单（按能否解释"仿真不抖、实机抖"排序）

### 3.1 【逻辑】三个 5 ms/20 ms 时钟自由运行，相位是"开机抽签"（确认，影响可量化）

- 总线线程与控制线程都是 `next += period; if(next<now) next=now; sleep_until(next)`
  （[go_loop.cpp:154-165](../../imgo2_deploy/real/Motor/src/go_loop.cpp#L154-L165)、
  [rl_real_imgo2.cpp:471-473](../../imgo2_deploy/real/rl_real_imgo2.cpp#L471-L473)）。
- 两者周期**相等**，所以相对相位在启动后基本锁死，但**锁在哪个值取决于开机时两者谁先醒**：
  好相位时 `seg1+seg4≈1.9 ms`，坏相位时 `≈6.9 ms`（§2 推导）。**同一个二进制、同一份配置，
  换一次开机就换一个 5 ms 量级的反馈延迟**。
- 而且任何一次迟到都会把相位永久平移（`next=now`），长跑下来相位会慢慢走。
  ⇒ 这正好能解释"有时好、有时抖""跑一段时间后变差"这类**没有改任何参数却改变表现**的观察。
  **待测**：用 `robot.csv` 的 `age_ms[12]` 与 `imu_age_ms` 分布判断——
  窄峰 = 相位锁住；宽/双峰 = 在漂。

### 3.2 【逻辑】总线层算出了"新鲜度/失联/被夹"信号，但高层全部丢掉（确认）

`GoMotorArray::state()` 填了 `cycleCount / targetSeq / targetAgeMs / stale / clampedCount`
（[go_motor_array.cpp:528-571](../../imgo2_deploy/real/Motor/src/go_motor_array.cpp#L528-L571)），
但 `RealHardware::snapshotLocked()` 只取 `faulted / feedback / receive_time_ns`
（[real_hardware.cpp:115-165](../../imgo2_deploy/real/hardware/real_hardware.cpp#L115-L165)）。
全仓 grep 确认 `cycleCount/targetSeq/targetAgeMs/clampedCount/stale` 在 Motor 层与测试之外**零消费者**。

后果：
- 控制环**无法知道这一拍拿到的是不是新反馈**。停顿期间它会把同一份 20 ms 前的反馈当成新状态，
  继续算目标、继续 `sendCommand`、继续 `submit` 给推理线程。
- `stale`（看门狗 80 ms 主动卸力）也不上报，FSM 不会知道"总线刚刚把力矩归零了"。
- 日志里的 `clamped` 是主环自己再跑一遍 `Safety::clamp` 得到的（[rl_real_imgo2.cpp:367](../../imgo2_deploy/real/rl_real_imgo2.cpp#L367)），
  不是总线真实的 `clampedCount`。

### 3.3 【逻辑】推理线程是"时间驱动"而不是"状态驱动"，停顿时会拿旧观测重复推理（确认）

`run()` 不检查快照是否已消费过（`RobotState` 里没有序号），只判 `age<100 ms`。
当主环停顿 15–25 ms（[项目验证总结 §3.4](../../imgo2_deploy/real/docs/xiaoji/项目验证总结.md)）时，
推理线程会**用同一份冻结观测再多推 1 次**，并且 `PolicyRunner::previous_` 照常前进
（[policy_runner.cpp:40](../../imgo2_deploy/real/sdk/policy_runner.cpp#L40)），
于是 `last_action` 这 12 维输入"自己往前走了一步"——**仿真里 `last_action` 永远对应上一个真实步**。

### 3.4 【逻辑】观测由两个异步 200 Hz 源拼成，关节与 IMU 的相对时刻不受控（确认）

- 关节：总线 5 ms 一轮，`receive_time_ns` 是**解析完成**时刻；IMU：独立线程 200 Hz 阻塞读，
  `receive_time_ns` 同理是解析时刻。
- 两者各自有 0–5 ms 的采样相位，且**互不相关**，所以进入同一条 45 维观测时相对偏差 0–10 ms 在漂。
- 仿真里 45 维是**同一时刻的完整状态**（`projected_gravity` 与 `dof_pos/dof_vel` 严格同拍）。
- `captured_at` 被主环覆盖成"本拍 `now`"（[rl_real_imgo2.cpp:356](../../imgo2_deploy/real/rl_real_imgo2.cpp#L356)），
  所以 `policy.csv` 的 `obs_age_ms` **结构上恒为 0**，不能用来量这件事（与 DEPLOY-DIAG-01 一致）。
  每关节真实年龄在 `robot.csv` 的 `age_ms[]` 里，IMU 的是 `imu_age_ms`。

### 3.5 【契约】`torque_limits` 不参与控制，仿真与实机的总力矩上限不同（确认；旧结论的补充）

- `PolicyConfig::torque_limits` 只被解析/校验并写进 `meta.json`（[policy_config.cpp:36](../../imgo2_deploy/real/sdk/policy_config.cpp#L36)、
  [rl_real_imgo2.cpp:143](../../imgo2_deploy/real/rl_real_imgo2.cpp#L143)），控制路径不读。
- `Safety::clamp` 夹的是 `tau`（前馈），而本链路 `tau_ff ≡ 0`；**kp/kd 产生的 PD 力矩不经过 23.5/23.7**。
- 仿真侧 `effort_limit` 是真裁的。⇒ 大误差瞬态下实机可以打到比仿真更大的力矩。
  （日志里 PPO 段反馈估算 |tau| 最大 22.67 N·m，尚未看到 ≥23.5 的样本；不能说已经顶到限，但契约确实没对齐。）

### 3.6 【逻辑】20 ms 零阶保持是最大单项，5 ms 不是

`q_des` 每 20 ms 才变一次，日志里能看到 4 拍同值（§8.8）。5 ms 总线买到的是
"反馈更新鲜、安全反应更快"，**不是"指令更快"**。要压这一项只能 `policy_ms=5`（改训练契约）。

---

## 4. 线程安全清单

### 4.1 【确定问题】`TransactionTrace` 是"多线程写、单线程读"的无锁环形缓冲，读侧会撕裂

- `record()` 直接 `ring_[slot] = r`（24 字节结构体，非原子）见 [go_trace.h:40-42](../../imgo2_deploy/real/Motor/inc/go_trace.h#L40-L42)；
  `snapshot()` 在总线线程仍在写时逐条读（[go_trace.h:52-59](../../imgo2_deploy/real/Motor/inc/go_trace.h#L52-L59)）。
- `RealHardware::transactionTrace()` 返回**裸指针**，出了锁之后调用方才去 `snapshot()`
  （[real_hardware.cpp:252-256](../../imgo2_deploy/real/hardware/real_hardware.cpp#L252-L256)）。
- 退出路径先 `dumpTrace()` 再 `hardware.stop()`（[rl_real_imgo2.cpp:476](../../imgo2_deploy/real/rl_real_imgo2.cpp#L476)、
  [rl_real_imgo2.cpp:484](../../imgo2_deploy/real/rl_real_imgo2.cpp#L484)），所以退出时这份快照**一定**和 4 条总线线程并发。
- 影响：只影响故障现场可读性（可能读到半条记录），不影响控制。修法：改成 seqlock，或返回 `std::vector` 且在锁内拷贝，
  或 `stop()` 之后再 dump。与 DEPLOY-DIAG-01「trace 并发快照需修订」是同一条。

### 4.2 【确定问题·低危】200 Hz 的 supervisor 与控制线程抢同一把锁，且锁内分配

`RealHardware::supervise()` 每 5 ms 调一次 `checkLocked()` → `snapshotLocked()`，全程持 `mutex_`，
里面构造 `vector<JointState>` 并拷字符串（[real_hardware.cpp:257-267](../../imgo2_deploy/real/hardware/real_hardware.cpp#L257-L267)）。
控制线程的 `readState()` 与 `sendCommand()`（各自再跑一次 `checkLocked`）抢的是同一把锁。
⇒ 控制拍理论上可能被 supervisor 让出（µs 级），并多出一次 12 关节拷贝。降到 20–50 Hz 或改成无分配快照即可，
不是抖动主因，但属于"控制临界区里有堆分配"。

### 4.3 【确定问题·低危】`stop()` 持 `RealHardware::mutex_` join 总线线程

[real_hardware.cpp:268-282](../../imgo2_deploy/real/hardware/real_hardware.cpp#L268-L282)：`motors_->stop()`
（内部 join 4 条总线线程 + 它们的 `onExit` 卸力）在持锁状态下执行。
现在不死锁，因为总线线程不碰 `RealHardware::mutex_`；但这是一条**只靠现状成立的锁序约束**——
以后谁在总线线程里加一个 `RealHardware` 调用就是死锁。建议 join 前放锁。

### 4.4 【无正确性问题】`InferenceWorker` 的共享状态是对的

`state_/velocity_/generation_/active_/result_` 全部在 `mutex_` 内读写，`PolicyRunner`（含 `previous_`）
只在推理线程里被摸（warmup 在 `start()` 之前），`stop()` 先置位再 join。唯一的小浪费是
`submit()` 每 5 ms `notify_all()` 唤醒一个**谓词为假**的 `wait_until`（空转一次系统调用），
正确性没问题，改成只在 `generation/active` 变化时 notify 即可。

### 4.5 【开销】控制线程每拍两次深拷贝

`latest()` 返回 `PolicyResult`（5 个 `vector<float>`）、`fsm.update` 里再 `c_.command = c_.result.command` 拷一次，
外加日志 `push()` 的 ~1 KB memcpy（[log_writer](../../imgo2_deploy/real/apps/log_writer.h) 队列非阻塞、满则丢）。
都是 µs 级、不影响 5 ms，但都在"每拍都要走"的路径上，属于可消项。

### 4.6 小结

**没有找到会让机器人抖的数据竞争。** 确定的并发缺陷都在诊断/退出路径（4.1）、
临界区设计（4.2/4.3）与可消开销（4.4/4.5）。真正与"抖"相关的在 §3 的时序逻辑，不在锁。

---

## 5. 最小验证集（不改策略、不重训，按顺序做）

1. **先量相位**（只读日志，零风险）：用最近一次流水线日志的 `robot.csv` 统计
   `age_ms[0..11]` 的分布 + `imu_age_ms - age_ms` 的分布 + `bus*_total_us`。
   - 窄峰（σ<0.5 ms）⇒ 相位锁住，延迟是该次运行的常数；
   - 宽峰/双峰 ⇒ §3.1 的漂移成立，需要把相位钉住。
2. **透出 `cycleCount`/`targetAgeMs`/`stale`**（10 行左右）：主环据此区分"新反馈/重复反馈"，
   并让日志记下"这一拍有没有错过新轮"。这是后面所有改动的观测前提，本身不改控制。
3. **A/B：推理内联到控制拍**（每 4 拍一次，或由 `cycleCount` 变化触发）：其余不变，比
   `imu_hip_new_log` 里那颗 7–8 Hz 峰的幅值与 `tau` RMS。这是对"5 ms 是否造成延迟"最直接的判决实验。
4. **IMU 姿态群延迟台架**：仿真里 `projected_gravity` 是零延迟；实机四元数来自芯片融合，
   群延迟未标定。用"快速敲击/倾斜"同时记 `gyro` 积分角与四元数角，看后者滞后多少 ms。
   若 ≥20 ms，它会和 §2 的 9–14 ms 叠加，直接吃掉相位裕度。
5. **力矩契约**：确认电机固件实际的总力矩/电流上限，再决定 `torque_limits` 要不要真正生效（§3.5）。

> 顺序理由：1 和 2 只加观测不碰控制；3 是唯一能直接回答用户问题、且可回滚的实验；
> 4/5 是并行的物理侧标定，不依赖前面。

---

## 6. 与"实机抖、仿真不抖、越跑越抖"的对应

| 现象 | §2 的 9–14 ms 调度开销 | §3.1 相位漂移 | §3.3 旧观测重复推理 | §3.4 关节/IMU 不同拍 | §3.5 力矩上限 | IMU 融合群延迟 | 温度/机械 |
|---|---|---|---|---|---|---|---|
| 仿真完全不抖 | 不适用（0 延迟） | 不适用 | 不适用 | 不适用 | 仿真真裁 | 不适用 | 不适用 |
| 实机震 | 直接吃掉相位裕度 | 改延迟 → 时好时坏 | 停顿后一拍开环 | 观测不同步 | 瞬态力矩更大 | 直接吃掉相位裕度 | — |
| 越跑越抖 | — | 迟到重置让相位走 | 停顿越多越明显 | 拍频漂 | 发热后力矩常数变化 | 偏置漂移 | 关节/减速器间隙、温升 |

**判断**：单靠 5 ms 主环不足以解释，但"三个自由时钟（9–14 ms 且带 5 ms 相位抽签）+ 20 ms 零阶保持
+ 未标定的 IMU 群延迟"合起来就是一条**相位裕度被吃掉十几到几十 ms** 的闭环，这在 CLASSIC 的延迟-失稳机制里足以让一个
在仿真里零延迟整定的 50 Hz 步态进入极限环；而 §3.1/§3.4 的相位漂移正好给出"时好时坏"。
**这些都是可测的**，不需要先猜是谁。

---

## 7. 本轮边界

- 已做：读 `rl_real_imgo2.cpp / sdk / hardware / Motor / config / fsm` 全链路、核对两条实测记录
  （`joint_feedback_dynamics_2026-10-21`、`项目验证总结 §8`）、逐条标注代码位置。
- 未做：未改任何代码/配置，未编译、未跑硬件或仿真，未新增日志分析（本机没有 `robot.csv`）。
- 因此本文所有结论按"源码可确认"与"待实测"分列，**不把推断记为已验证**，也不把"改法"当成已修复。
