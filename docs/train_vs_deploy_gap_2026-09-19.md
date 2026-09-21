# 训练侧 ↔ 部署侧还有哪些差异（不含仿真器物理）（2026-09-19）

起因：用户问「在不考虑仿真器的情况下，训练侧和仿真侧还有啥区别」。
本文只做**源码逐项对照**，不涉及 Gazebo/MuJoCo 的求解器、接触模型、步长等仿真器物理
（那部分见 [两仿真器差异记录](sim2sim_mujoco_vs_gazebo_2026-09-19.md)）。本轮未改任何代码。

对照对象：

- 训练侧 = 本仓库 `imgo2_rl` 的 `base_move`（`velocity_env_cfg.py` 公共配置 + `rough_env_cfg.py` 覆盖 +
  `assets/imgo2.py` + `agents/rsl_rl_ppo_cfg.py`）。
- 部署侧 = `imgo2_deploy`（`policy/imgo2/base.yaml` + `policy/imgo2/ppo/config.yaml` +
  `library/core/rl_sdk/rl_sdk.cpp` + `fsm_robot/fsm_imgo2.hpp`）。
- **前提与限制**：键 1 的权重（`base_move/policy_flat.pt`）来自参考项目，参考项目的奖励/终止/随机化配置
  我们看不到；下面的"训练侧"是**本仓库镜像的那套接口**（9.5 节离线核对过接口一致），
  不是参考项目训练时的全部设置。

## 1. 已经逐项一致（这部分不用再怀疑）

| 项 | 训练侧 | 部署侧 | 证据 |
|---|---|---|---|
| actor 观测项与顺序 | ang_vel, gravity_vec, commands, joint_pos, joint_vel, actions（`base_lin_vel`/`height_scan` 置 None） | `observations` 同名同序 | `rough_env_cfg.py:43-47` / `ppo/config.yaml` |
| 观测缩放 | ang_vel 0.25、joint_pos 1.0、joint_vel 0.05、其余 1.0 | 同 | 同上 |
| 观测裁剪 | 每项 clip(-100,100) | 拼接后整体 clamp ±100 | `rl_sdk.cpp:181` |
| joint_pos 观测 | `joint_pos_rel` = q − 默认 | (q − `default_dof_pos`) | `rl_sdk.cpp:98` |
| gravity_vec | `projected_gravity` = Rᵀ(0,0,−1) | `QuatRotateInverse(base_quat,(0,0,-1))` | `rl_sdk.cpp:88-91` |
| ang_vel 坐标系 | body | body（ROS 2 / MuJoCo 路径） | `rl_sdk.cpp:77-86` + `ang_vel_axis="body"` |
| 动作裁剪 | action clip ±3 | `Forward()` clamp ±3 | `rough_env_cfg.py:54` / `rl_sim_mujoco.cpp:403-406` |
| 动作 → 目标 | `use_default_offset=True`，scale hip 0.125 / 其余 0.25 | `default_dof_pos + action*action_scale` | `rl_sdk.cpp:267` |
| PD 公式 | `kp(q_des−q) + kd(dq_des−dq) + τ_ff` | 同（且 τ_ff = 0，见下） | `actuator_pd.py:150-161` / `rl_sdk.cpp:269`、`rl_sim_mujoco.cpp:184-187` |
| 增益 | kp 25 / kd 0.5（标称） | `rl_kp 25` / `rl_kd 0.5` | `assets/imgo2.py:82-84` / `ppo/config.yaml` |
| 默认关节角 | 0 / 0.87 / −1.82 | 同 | `assets/imgo2.py` init_state / `base.yaml` |
| 关节顺序 | `FL,FR,RL,RR` × (hip,thigh,shank)，`preserve_order=True` | 恒等 `joint_mapping` | `rough_env_cfg.py:18-25` |
| 无归一化 | `empirical_normalization=False`、actor/critic_obs_normalization=False | 导出用 `normalizer=None`，部署不归一化 | `rsl_rl_ppo_cfg.py` |
| 观测历史 | 空 | `observations_history: []` | — |
| 策略周期 | `dt=0.005 × decimation=4` = 50 Hz | 同（MuJoCo 路径） | `velocity_env_cfg.py:715,718` / `rl_sim_mujoco.cpp:109-110` |
| `last_action` | 裁剪后的动作，episode 开始为 0 | `obs.actions` 为裁剪后动作，`InitObservations()` 置 0 | `rl_sdk.cpp:195-196` |

**一个容易误读、但核对后没问题的地方**：`RL::ComputeOutput()` 会把
`kp(q_des−q) − kd·dq` 算成 `output_dof_tau`（`rl_sdk.cpp:269`），而两条部署路径的 plant
（`rl_sim_mujoco.cpp:184-187` 的 `mj_data->ctrl`、ROS 2 的 `RobotJointControllerGroup::UpdateFunc`）
**又算了一遍 PD**。看起来像"PD 被算了两次"，但 `RLFSMState::RLControl()` 明确把
`motor_command.tau[i] = 0`（`rl_sdk.cpp:606`），所以 feedforward 力矩根本没进 plant，
两边的 PD 都只生效一次。

## 2. 训练有、部署没有（域随机化与课程）

全部来自 `velocity_env_cfg.py` 的 `EventCfg`（`base_move` 只关了外力/推力两项，其余保留）：

| # | 训练侧 | 部署侧 | 影响 |
|---|---|---|---|
| 2.1 | **PD 增益每个 episode reset 随机 ×(0.5, 2.0)**（`randomize_actuator_gains`，`:332-342`）⇒ 训练见过 kp 12.5–50 / kd 0.25–1.0 | 固定 25 / 0.5 | 策略被训成对增益不敏感；反过来也说明"换增益/换权重"未必改变行为 |
| 2.2 | 质量/惯量：base `+(−1,3)` kg、其它 `×(0.7,1.3)`、`recompute_inertia=True`（`:269-289`） | 名义值 | 部署是分布里的一个点，不是中心 |
| 2.3 | 质心 `±0.05 m`（`:302-309`） | 名义 CoM | 同上 |
| 2.4 | 地面/机体摩擦材料随机：static `0.3–1.0`、dynamic `0.3–0.8`、restitution `0–0.5`（`:257-267`，`base_move` 未关） | MJCF 足端摩擦 0.4、Gazebo 足端 `mu1/mu2=0.6` | 部署落在训练分布偏低的一段 |
| 2.5 | 初始状态随机：x/y `±1.0`、yaw `±π`、roll/pitch `±0.3`、速度 `±0.2`/`±0.05`（`rough_env_cfg.py:75-90`） | 固定从默认姿态起身 | 接管时状态分布完全不同（见 3.1） |
| 2.6 | 观测噪声（`enable_corruption=True`，`:133-182`）：ang_vel ±0.2、gravity ±0.05、joint_pos ±0.01、joint_vel ±1.5（原始单位） | 不注入噪声（只有真实传感器噪声） | 仿真对比里部署观测"更干净" |
| 2.7 | 指令每 10 s 重采样、2% 站定、**线性指令模长 ≤0.2 直接置 0**（`mdp/commands.py:48`）、yaw 由 heading 控制器生成（stiffness 0.5、`rel_heading_envs=1.0`）；范围 x `±1.0`、y `±0.8`、yaw `±1.5`（`rough_env_cfg.py:186-188`） | 手柄/键盘直接给 (x,y,yaw)，**无死区、无范围限制** | 部署可以下发训练里不存在的指令（小速度、超范围、纯 yaw） |
| 2.8 | episode 20 s、基座触地等终止、自动复位 | 无终止、无自动复位（只有手动 `R`） | 部署没有"失败"概念，退化姿态会一直持续 |
| 2.9 | 地形课程与粗糙地形/台阶 | 平地 + 单一固定场景 | 已测动作范围之外 |
| — | 随机推力/外力：`base_move` 里**已置 None**（`rough_env_cfg.py:100-101`） | — | 这条其实两边一致 |

## 3. 部署有、训练没有

| # | 部署侧 | 训练侧 | 备注 |
|---|---|---|---|
| 3.1 | **GetUp 相**：先用固定增益 `fixed_kp/kd = 60 / 2.0` 插值到默认姿态（从 Passive 起：1 s "pre getup" + 2 s "getup"；从其它姿态起 1 s），再切策略，并把 `episode_length_buf` 归零（`fsm_imgo2.hpp:81-93,172-190`、`rl_sdk.cpp:531-587`） | 直接从随机初始状态开始，一开始就是策略 | 策略接管瞬间：部署≈静止精确默认姿态，训练≈随机姿态+速度 |
| 3.2 | **Passive 相**：kp=0、kd=8、tau=0（`fsm_imgo2.hpp:25-35`） | 没有这一相 | — |
| 3.3 | 安全网 `TorqueProtect` / `AttitudeProtect` **两条路径都注释掉**（`rl_sim.cpp:728-729`、`rl_sim_mujoco.cpp:364-365`） | 训练侧没有对应机制（靠奖励/终止） | 真机上少一层保护 |
| 3.4 | 手柄有 `axis_deadzone`；键盘每次按键步进 0.1（`rl_sdk.cpp:27-56`） | — | 指令来源差异 |
| 3.5 | `base.yaml` 的 `pre_dof_pos = 0/1.2/−2.65` 是**死配置**——GetUp 用的是类里硬编码的 `1.36`（`fsm_imgo2.hpp:56-62`） | — | 文档/配置一致性问题 |
| 3.6 | 力矩上限的实际生效值不统一：`torque_limits 23.5` 只用在（被丢弃的）feedforward tau 与 `TorqueProtect`；真正 clamp 的是 plant 的 **23.7**（URDF `<limit effort>` / MJCF `ctrlrange`） | 23.7 | 见 4.1 |

## 4. 同一件事、实现不同（最值得注意的几条）

### 4.1 力矩包线：训练是转速下垂，部署是恒定 clamp

训练用 `DCMotorCfg`（`assets/imgo2.py:82-84`，`effort_limit=saturation_effort=23.7`、`velocity_limit=30.1`），
它的 `_clip_effort` 是**线性四象限扭矩-转速曲线**（`actuator_pd.py:295-307`）：

```
τ_max = 23.7 × (1 − dq/30.1)      # dq = 10 rad/s 时只有 ≈15.8 N·m
τ_min = 23.7 × (−1 − dq/30.1)
```

部署两条路径都是**恒定** `±23.7`（MuJoCo `ctrlrange`、ROS 2 控制器的 `EffortLimit` + URDF `effort="23.7"`），
而 config 里的 `torque_limits: 23.5` 实际没进 plant。
⇒ 高速摆腿时**部署比训练"更有力"**，这是分布外输入，也可能是 sim2real 的一个隐患。

### 4.2 观测裁剪的先后顺序

训练是「每项先加噪声 → clip(-100,100) → 再乘 scale」，部署是「先乘 scale → 拼接后整体 clamp ±100」。
对 ≤1 的缩放和物理量级内的值两者等价，只有原始值 >100/scale（如 ang_vel > 400 rad/s）时才不同，**可忽略**，
但口径上确实不同。

### 4.3 默认姿态的来源

训练取 asset 的 `init_state`，部署取 `base.yaml`/`config.yaml` 里的副本。当前数值一致，
但它是**两份独立文本**，改一边不会自动同步（这也是换来路不明的 checkpoint 时必须连配置一起换的原因）。

### 4.4 指令语义

训练 `velocity_commands` 里的 yaw 分量是 heading P 控制器的输出（只在非站定 env 上生效），
部署直接给手柄的 yaw 轴；训练还有 0.2 的线性死区。同一个数值在两边的"含义分布"不同。

## 5. 与 SIM-01 的关系

上面 2.1（增益随机化）、2.6（观测噪声）、2.8（无终止）、3.1（GetUp 接管）都不解释
"Gazebo 更抖还是 MuJoCo 更抖"，但它们决定了**策略本身有多鲁棒**：
一个被训练成对 kp/kd ×(0.5,2.0)、对质量/摩擦/CoM 随机都成立的策略，
在两个仿真器里表现差异大，更多要往**仿真器/控制实现**（步长、PD 频率、接触、关节阻尼）
去找，而不是往训练侧找。这也是 SIM-01 的完成标准里要先做同口径实测的原因。

## 6. 本轮性质与未做

- 只读源码与配置，**没有跑训练、没有跑仿真、没有改任何文件**（用户本轮只问差异）。
- 未核对：参考项目训练时实际的奖励权重、终止条件、随机化范围（本仓库 `base_move` 只是它的接口镜像）；
  真机链路的传感器噪声与延迟、电机内环（`rl_real_imgo2.cpp` 走 Unitree SDK，本次未展开）。
- 相关：SIM-01（两仿真器抖动差异，待量化）、PPO-01（键 1 权重来源与选型）。
