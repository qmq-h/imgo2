# 三个策略的 sim2sim 运行期排查（2026-09-17）

关联问题：MODEL-03（新增）、AMP-06、DEPLOY-08、DEPLOY-02、DEPLOY-06、AMP-05。
本文只记这一轮排查的现象、根因、修法与验证数值；结论与状态同步进根 `README.md` 第 7 节。

## 1. 现象

用户反馈「目前运行都有点问题」，要求先看 PPO（键 1）。本机没有可用显示与 NVIDIA
驱动（`glfw` 建窗失败），因此用**仓库外临时 harness** 复现：直接复用仓库真实的
`RL`/`FSM`/`librl_sdk`/libtorch/MuJoCo 与 `policy/`，只重写 `GetState`/`SetCommand`/
`RunModelStep`/`Forward`，按键序列与 GUI 相同（`0` 起身，3.5 s 进策略）。

修传感器之前，三个键在 harness 里的表现（vx=0，测量窗口取策略接管后 2 s 起）：

| 键 | 策略 | `z_final` | 现象 |
|---|---|---|---|
| 1 | PPO（flat run 导出，45 维） | 0.154 | 接管瞬间就输出饱和动作（`act_max=3.93`，后续 ±7.5），塌成深蹲 |
| 2 | himloco（Go2 占位） | 0.188 | 侧倾约 36° 后卡住 |
| 3 | AMP（`model_9000.pt` 导出，48 维） | 0.2663 | 能站住 |

参考项目那两份 45 维策略（`base_move/policy.pt`、`amp/policy.pt`）同样塌到 0.14–0.16。

## 2. 根因：MuJoCo 的 frame 传感器挂在 `body` 上，姿态被惯量主轴旋转污染

### 2.1 决定性观测

策略刚接管的瞬间，机器人正处于 `GetUp` 结束的直立姿态，此时 45 维观测打印为

```
obs[0..8] = 0.000 -0.000 -0.000 | -1.000 0.004 -0.025 | 0.000 0.000 0.000
             ang_vel(3)          gravity_vec(3)          commands(3)
```

`gravity_vec` 应该是 `(0, 0, -1)`，实际是 `(-1, 0.004, -0.025)`——**恰好是被旋转 90° 的"翻倒"姿态**。
`QuatRotateInverse(·, (0,0,-1))` 的第三分量恒为 `-1 - 2q_z² ≤ -1`（见
`library/core/vector_math/vector_math.hpp`），所以 `-0.025` 出现在第三槽在数学上不可能；
说明读到的三元组并不是 `(g_x, g_y, g_z)` 而是另一段数据。

时间线打印（每 0.25 s 一次 `D->xquat` 与 `D->sensordata[36..39]` 对照）：

```
T 0.00  z=0.500 xquat=(1.000 0.000 0.000 0.000) sensq=(-0.002 0.707 -0.002 0.707)
T 3.50  z=0.287 xquat=(1.000 -0.000 -0.012 0.002) sensq=(-0.003 0.698 -0.000 0.716)
```

即：**机器人一直直立（`xquat` 恒为 ±(1,0,0,0)），但 `framequat` 传感器报的是
`xquat ⊗ 常量偏移`**，偏移 ≈ `(-0.002, 0.707, -0.002, 0.707)`（≈ 180° 绕 `(1,0,1)/√2`），
正是 base 的惯量主轴旋转 `iquat`（base 的 `fullinertia` 三个主惯量不是按 `ixx<iyy<izz`
排列，MuJoCo 因此把主轴做了转置）。

### 2.2 为什么参考项目没这个问题

参考项目 `~/RL/sim2sim/Imgo2_deploy/imgo2_description/mjcf/imgo2.xml` 的 IMU 传感器挂的是
**site**：

```xml
<site name="base_site" pos="0.0 0.0 0.0" />
<framequat name="imu_quat" objtype="site" objname="base_site" />
<gyro name="imu_gyro" site="base_site" />
<framelinvel name="frame_vel" objtype="site" objname="base_site" />
```

我们 2026-09-17 用训练 URDF 重写 MJCF 时写成了 `objtype="body" objname="base"`，
于是姿态观测整体偏转 90°。策略把"站直"当成"已翻倒"，PPO 这类 45 维策略直接崩，
AMP 只是恰好鲁棒。

### 2.3 修法（`imgo2_description/mjcf/imgo2.xml`）

```xml
<framequat name="orientation" objtype="site" objname="imu" />
<gyro name="angular_velocity" site="imu" />
<framelinvel name="base_lin_vel" objtype="site" objname="imu" />
```

`imu` 是 base 里原有的 site（`pos` 缺省为原点、无 `quat`），所以 site 坐标系 = link 坐标系。
**声明顺序不变**，`sensordata` 偏移因此也不变（0–11 关节位置、12–23 关节速度、
24–35 关节力矩、36–39 姿态、40–42 陀螺、43–45 线速度，`nsensordata=46`），
`rl_sim_mujoco.cpp` 的 `[3n..3n+3]`／`[3n+4..3n+6]` 读取处无需改动。

同一轮把文件末尾那条陈旧 `<keyframe>`（`base z=0.35`，即当初"一启动就飞"的穿地参数）
注释掉：C++ 路径不应用 keyframe，但 MuJoCo `simulate` GUI 的 Key 下拉框会应用它，
参考项目同样把它注释掉了。

## 3. 修后验证

`gravity_vec` 立刻变成 `(0.00, -0.00, -1.00)`，三个键全部正常（vx 为指令线速度，
`dx` 是策略接管后到 14 s 的位移，`thigh_range` 是同期 FL_thigh 关节角极差）：

| 键 | 策略 | vx | `z_mean` | `z_final` | `dx` | 实测速度 | `thigh_range` | max\|dq\| |
|---|---|---|---|---|---|---|---|---|
| 1 | PPO | 0.0 | 0.3212 | 0.3212 | 0.075 | — | 0.003 | 0.03 |
| 1 | PPO | 0.5 | 0.1276 | 0.1271 | 5.343 | **0.509 m/s** | 1.114 | 11.91 |
| 1 | PPO | 1.0 | 0.1413 | 0.1384 | 10.240 | **0.975 m/s** | 1.359 | 13.61 |
| 2 | himloco | 0.0 | 0.1877 | 0.1881 | −0.060 | — | 1.866 | 12.51 |
| 3 | AMP | 0.0 | 0.3015 | 0.3015 | 0.060 | — | 0.014 | 0.03 |
| 3 | AMP | 0.5 | 0.3032 | 0.3033 | 0.076 | **0.007 m/s** | 0.024 | 0.07 |

结论：

- **PPO 成立**：静止时纹丝不动（关节角极差 0.003 rad），给指令后跟踪误差 <2%
  （0.5→0.509、1.0→0.975），同时 FL_thigh 摆幅 1.1–1.4 rad，是真步态而非滑行。
  唯一观察点：行走时基座高度只有 0.13 m（站姿 0.32 m），步态偏低。
- **AMP 只站不走**：站姿稳定（0.3015）、四腿几乎不动（极差 0.014 rad），
  `vx=0.5` 时 10.5 s 只走 0.076 m，等于完全忽略速度指令。
- **himloco 依旧不可用**：这是 Go2 参考占位策略（`himloco.pt` 与
  `~/RL/sim2sim/rl_sar/policy/go2/himloco/himloco.pt` 同哈希），且配置里的
  `joint_mapping [3,4,5,0,1,2,9,10,11,6,7,8]` 是硬件置换，在 MuJoCo 路径上会索引错腿。
- **按键 1→2→3 连续切换不再崩溃**（详见第 5 节 DEPLOY-08）。

### 3.1 对照实验：参考项目的模型与策略

同一个 harness、同一套参考配置，只换模型：

| 模型 | 策略 | vx | `z_final` | `dx` |
|---|---|---|---|---|
| 我们的（训练侧物理，base `5.53394020`、图元碰撞、`timestep=0.005`） | 参考 `base_move/policy.pt` | 0.0 | **0.2887** | 0.065 |
| 我们的 | 参考 `base_move/policy.pt` | 0.5 | **0.2796** | **+4.770（0.454 m/s）** |
| 参考自己的（base `6.53394`、mesh 碰撞、没写 `timestep`=2 ms） | 参考 `base_move/policy.pt` | 0.0 | 0.0742 | +1.149 |
| 参考自己的（强制 `timestep=0.005`） | 参考 `base_move/policy.pt` | 0.0 | 0.0771 | −0.053 |
| 参考自己的（强制 `timestep=0.005`） | 参考 `base_move/policy.pt` | 0.5 | 0.1114 | −0.494 |

**参考策略在参考自己的 MJCF 上站不起来，在我们的模型上反而站得稳、走得准。**
这正面支持"物理参数以训练侧为准"（MODEL-01/MODEL-02）这一决定：参考那份 MJCF 的
base 质量 6.53 kg（比训练侧重 1 kg）与 mesh 碰撞体并不是训练时的受控对象，
参考项目的策略本身是为训练侧物理训练的。两份模型的 `framequat` 都挂在 site 上，
所以在同一 harness 里 `gravity_vec` 都是 `(0,0,-1)`，可排除传感器差异。

## 4. 更正此前记录里的三条结论

1. **「用同一 harness 跑参考项目 45 维 `policy.pt`，结果四位小数完全相同，故非映射问题」无效。**
   `POLICY_DIR` 是编译期 `-D` 烘焙进 `librl_sdk.a` 的（`strings` 可见
   `/home/qmq/RL/imgo2/imgo2_deploy/policy`），当时链接的是仓库自带库，因此换目录并没有
   真正换策略，那次比较实际上是"自己和自己比"。本轮改为把 `rl_sdk.cpp` 直接编进
   harness 可执行文件并用 `-DPOLICY_DIR=<临时目录>`，才真正换到了参考策略。
2. **DEPLOY-08 的崩溃是 harness 假象。** 我第一版 harness 的 `Forward()` 简化成
   `model->forward({ComputeObservation()})`，漏掉了真实 `RL_Sim::Forward()` 里的历史分支
   （`history_obs_buf.insert` → `get_obs_vec(observations_history)`）。himloco 需要 6 帧
   历史，于是报 `mat1 and mat2 shapes cannot be multiplied (1x45 and 270x128)`。
   把 `Forward()` 逐行照搬真实实现后，**单键与 1→2→3 连续切换都不再崩溃**。
   结论：真实 deploy 代码没有这个问题；GUI 复现仍可作为独立确认项保留。
3. **AMP 不跟踪速度不是"缺 `lin_vel` 观测"。** 我们 `amp/config.yaml` 声明了 `lin_vel`，
   而 `rl_sim_mujoco.cpp` 的 `GetState` 从未读 `sensordata[43..45]`（参考项目的
   `GetState` 也不读），所以该项在部署侧恒为 0。本轮在 harness 里把
   `frame lin vel`（世界系）用 `QuatRotateInverse(base_quat, v)` 旋进体系后实测：
   `obs.lin_vel` 能正确读出 `(0.087, 0.000, 0.012)`，但 AMP 结果不变
   （vx=0.5：`dx` 0.0612 对 0.0759、`thigh_range` 0.002 对 0.024）。
   **所以没有改 deploy 代码**，AMP-06 的"不跟踪"归因为策略性质。

## 5. 复现方法（临时 harness 的要点）

1. 必须在**同一次 shell 调用**里完成建目录、编译、运行：`/tmp` 是本 harness 每次调用的私有目录。
2. 要换策略目录必须把 `rl_sdk.cpp` 一起编进来并 `-DPOLICY_DIR=<目录>`，不能只链接 `librl_sdk.a`。
3. 临时策略目录要带 `policy/imgo2/base.yaml`：`ReadYaml("imgo2","base.yaml")` 在
   `InitRL` 之前执行，缺它会让 `num_of_dofs` 为 0 并以 SIGFPE 退出（136）。
4. `Forward()` 必须逐行照搬 `RL_Sim::Forward()`（含历史分支），否则 himloco 必然误报崩溃。
5. 模型时间步决定控制周期：harness 用 `decimation × M->opt.timestep`，
   所以对没有写 `timestep` 的 MJCF（参考项目那份 = 2 ms）要显式设为 0.005 才与部署等价；
   真实部署由 `PhysicsThread` 按模型时间步在独立线程步进、`LoopFunc` 按 `dt×decimation` 跑策略。
6. 测量窗口要跳过进入策略后的前 2 s（PD 收敛与姿态过渡）。

## 6. 限制与下一步

**未运行**：GUI（本机无可用显示/驱动）、Isaac Lab 训练与回放、ROS/Gazebo、真机。
因此"修好了"的结论目前只覆盖无头物理回放；GUI 内仍需人工确认一次按键 1/2/3 与
`1→2→3` 切换。

待决定：

- **PPO 用哪份权重（已决定，见第 7 节）**：行走时基座高度是主要区别 —— `imgo2_flat/2026-06-21_23-24-09`
  是 0.13 m，rough 系列是 0.25–0.30 m。
- **AMP 是否要换 checkpoint**：现有 `model_9000.pt` 只站不走；若目标是速度跟踪，
  需要一份会走的 AMP/运动模仿 checkpoint（也与 AMP-05 的 45/48 维问题相关）。
- **himloco 是否要替换**：需要 Imgo2 的 HIM-Loco 导出（参考项目那份还有
  `encoder.onnx`，即历史编码器），并同时改 `joint_mapping` 与默认姿态。
- **是否要在 `GetState` 里补 `lin_vel`**：本轮证明它对 AMP 现状没有影响，
  参考项目也不读；若将来有依赖 `base_lin_vel` 的策略再补，注意 framelinvel 是世界系速度，
  训练侧 `base_lin_vel` 是体系速度，需要旋转。

## 7. 后续（同日）：五份 PPO 导出的对比、来源核对与键位最终分配

### 7.1 五份导出的实测（同一 harness、同一模型，各自 run 的 kp/kd）

窗口 = 策略接管后到 12.5 s（`dx` 除以 12.5 即平均速度；`thigh_range` 是同期 FL_thigh 极差）：

| run（导出） | kp/kd | vx=0 | vx=0.5 | vx=1.0 |
|---|---|---|---|---|
| flat/2026-06-21_23-24-09（2000 it） | 20/1.0 | z 0.321 | z 0.128，0.43 m/s | z 0.14，0.82 m/s |
| rough/2026-06-19_19-58-19（900 it） | 25/0.5 | z 0.322 | z 0.301，0.47 m/s | z 0.307，0.74 m/s |
| rough/2026-06-21_08-18-30（4400 it） | 25/1.0 | z 0.307 | z 0.256，0.54 m/s | z 0.288，0.80 m/s |
| **rough/2026-06-21_16-37-38（4400 it）** | **20/0.2** | **z 0.327** | **z 0.259，0.44 m/s** | **z 0.261，0.89 m/s** |
| rough/2026-06-21_23-23-13（4999 it） | 20/1.0 | z 0.324 | z 0.167，0.22 m/s | z 0.216，0.58 m/s |

（表内速度按 12.5 s 归一；用 14 s 窗口的原始 `dx` 见前文，两种口径都指向同一结论。）
能"一边走一边维持高度"的是 **rough 系列**（z 0.26–0.31），flat 那份虽然也跟速度但只有 0.13 m。

### 7.2 来源核对：用 checkpoint storage 散列给导出"验明正身"

方法（不依赖 Python torch，只用标准库）：`torch.save`/`torch.jit.script` 的 `.pt` 都是 zip，
成员形如 `<名字>/data.pkl` 与 `<名字>/data/<key>`；每个 tensor 的原始字节就是一个 storage 成员。
于是把每个文件的 storage 取 md5 成集合，两两求交集，就能判断某个导出是不是某个 checkpoint 的 actor。

结果（交集个数 / 8）：

| | flat/…23-24-09 | rough 19-58-19 | rough 08-18-30 | rough 16-37-38 | rough 23-23-13 | amp/model_9000 |
|---|---|---|---|---|---|---|
| `ppo/policy.pt`（换之前 = flat 导出） | **8** | 0 | 0 | 0 | 0 | 0 |
| `amp/policy.pt` | 0 | 0 | 0 | 0 | 0 | **8** |
| 各 run 自己的 `exported/policy.pt` | 8（自身） | 0（自身的末号 checkpoint 是 model_900，未命中，应来自更早的号） | 8（自身） | 8（自身） | 8（自身） | — |

另外用 `strings` 看 state-dict 键名可以一眼分开两类：五个 run 的 checkpoint 只有
`actor/critic/normalizer`；`amp/model_9000.pt` 还含 `discriminator`，且体积 11.9 MB（三个 2 MB 层）
对 4.6–5.7 MB。五个 run 的 `agent.yaml` 全是 `OnPolicyRunner` + `class_name: PPO`，env 里没有任何
motion/AMP 项，仓库的 AMP 训练用的是 `AMPOnPolicyRunner`。

**结论：`ppo/` 原来那份是 `imgo2_flat/2026-06-21_23-24-09/model_1999.pt` 的 actor（普通 PPO，
flat 地形），不是 AMP 训练产物；全机唯一的 AMP 训练产物是 `amp/model_9000.pt`。**

### 7.3 最终键位与权重（2026-09-17 按用户决定执行）

- **键 1（PPO）** ← `imgo2_rough/2026-06-21_16-37-38/exported/policy.pt`
  （sha256 `618cc4ea4737c343…`，来源已由 7.2 的散列比对确认 8/8 命中该 run 的 `model_4400.pt`）；
  `ppo/config.yaml` 的 `rl_kp/rl_kd` 由 `20/1.0` 改为该 run 的 `20/0.2`，其余（45 维观测、
  `default_dof_pos 0/0.8/-1.5`、`action_scale 0.125/0.25`、`clip ±100`、观测缩放）与该 run 的
  `env.yaml` 逐项一致。
- **键 3（AMP）** 不变：`amp/policy.pt`（`model_9000.pt` 的 actor，48 维，唯一 AMP 产物）。
  该 checkpoint 本身（`imgo2_deploy/policy/imgo2/amp/model_9000.pt`，11.9 MB）被 `.gitignore`
  的 `*.pt` 排除、**未入库**，只在本机；入库的产物是它的导出 `policy.pt`。换机器重建时需要另外
  取得该 checkpoint。
- flat 那份导出**从仓库移除**（`ppo/policy.pt` 已被替换；训练日志 `~/RL/isaac/...` 里那份仍保留，
  那是训练产物，不属于仓库）。

**换后复测**（16 s 仿真，窗口 = 接管后 12.5 s）：键 1 `vx=0` 站姿 0.327 且基座位移 0.023 m；
`vx=0.5` → z 0.259、**0.51 m/s**、FL_thigh 极差 0.88 rad；`vx=1.0` → z 0.264、**1.05 m/s**、
极差 1.96 rad；键 3 站姿 0.3015 不变；`1→3` 与 `1→2→3` 连续切换均不崩溃（himloco 仍会把机器人
掀一下，AMP 这次能接住，见 DEPLOY-01/DEPLOY-06）。
