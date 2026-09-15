# Imgo2 研究探索路线图：面向 10 月真机的前期准备

## 0. 基本判断

现有项目中的模型、Isaac Lab 环境、PPO/HIM-Loco/AMP、部署框架，大部分来自开源方案或开源方案改造。它们本身更适合作为实验平台，而不是直接作为论文贡献。

因此，小论文不建议写成：

> 我搭建了一个 Imgo2 强化学习训练与部署系统。

更建议写成：

> 我利用 Imgo2 平台，围绕一个明确问题，提出并验证一种能在真机上产生差异的控制/训练/迁移方法。

你的现实条件是：

- 10 月计划上真机。
- 当前擅长 RL locomotion 和 RL tracking。
- RL tracking 暂时缺少视觉、深度、激光等外部传感器技术储备。
- 现阶段最可控的传感器主要是：IMU、关节位置、关节速度、电机电流/力矩估计、足端接触或接触估计。

所以近期最值得探索的是：

> 不依赖复杂外部感知，仅依靠本体感知和运动数据，让策略在真机上更稳、更可迁移、更像目标动作。

## 1. 选题方向总览

### 方向 A：面向真机迁移的本体感知鲁棒运动控制

核心问题：

在没有视觉/深度传感器的情况下，如何利用关节、IMU、历史动作等本体信息，提高四足机器人从仿真到真机的稳定性？

这条线最稳，因为它和 10 月真机直接相关。

可探索内容：

- actuator delay 随机化。
- PD 参数随机化。
- 电机力矩限制与饱和建模。
- 关节摩擦、阻尼、反电动势近似建模。
- IMU 噪声、关节编码器噪声、动作延迟。
- 历史观察用于隐式估计地形/接触/动力学误差。
- teacher-student：teacher 在仿真中看 privileged state，student 只看真机可获得的 proprioception。

潜在论文题目：

- 面向自研四足机器人的本体感知鲁棒强化学习运动控制方法
- 基于动力学随机化与历史观测的四足机器人 sim-to-real 运动控制
- 面向低成本四足机器人的无外部感知强化学习运动控制

论文贡献可以写成：

1. 构建面向真机迁移的 actuator-aware 随机化训练框架。
2. 设计只依赖 IMU 与关节状态的历史观测策略。
3. 在仿真和真机上验证速度跟踪、抗扰动和部署稳定性。

这是我最推荐的主线。

### 方向 B：RL tracking 的“本体感知动作跟踪”

核心问题：

在没有 mocap、视觉或外部定位的情况下，能否让 Imgo2 通过本体感知稳定跟踪参考步态或参考关节轨迹？

这条线和你的 RL tracking 技术栈匹配，但要避开对传感器的依赖。

可探索内容：

- 关节级 tracking：策略跟踪参考关节位置/速度。
- 足端相位 tracking：不要求知道世界坐标足端轨迹，只约束相位、触地时序、关节形态。
- command-conditioned motion tracking：输入速度指令，自动选择或插值参考动作。
- residual RL：用开源/手写 gait 或参考动作作为基准，RL 输出 residual correction。
- AMP/判别器只作为风格约束，不要求严格追踪世界系轨迹。

为什么适合你：

- 不需要额外视觉传感器。
- 现有 `imgo2_dataset/datasets/imgo2_motion/` 已有多速度动作。
- 可以先在仿真中做完整，然后 10 月上机做低速直线、转向、站立恢复。

潜在论文题目：

- 基于本体感知的四足机器人参考步态跟踪强化学习方法
- 融合参考动作先验与残差策略的四足机器人运动控制
- 面向真机部署的四足机器人低维运动跟踪策略

关键创新点不应是“我用了 AMP”，而是：

> 在传感受限条件下，把动作数据转换为真机可用的低维 tracking 约束，并通过 residual/RL 提高迁移稳定性。

### 方向 C：actuator-aware RL：把电机真实特性放进训练

核心问题：

四足机器人 sim2real 失败很多时候不是算法问题，而是电机、减速器、PD 控制、延迟、限幅和摩擦没有建模。能否围绕 Imgo2 使用 Go2 同类电机这一条件，做一个更贴近真机执行层的训练方法？

这条线很有论文味，因为它不是简单复现 locomotion，而是针对真实硬件瓶颈做方法。

可探索内容：

- 动作延迟随机化：1-3 个 control step。
- 控制频率随机化：例如 50 Hz、100 Hz、200 Hz 对比。
- PD gain 随机化。
- torque saturation。
- motor strength randomization。
- joint friction/damping identification。
- 简化 actuator network 或 actuator residual model。

可验证指标：

- 仿真中加入不同电机误差后的速度跟踪误差。
- 随机外力扰动后的恢复时间。
- 真机上电机温升、力矩峰值、步态抖动、跌倒次数。

潜在论文题目：

- 面向四足机器人真机迁移的执行器感知强化学习控制
- 考虑电机延迟与限幅的四足机器人强化学习运动控制
- 基于执行器随机化的低成本四足机器人 sim-to-real 方法

这条线推荐作为方向 A 的核心机制。

### 方向 D：低传感器条件下的 RL tracking + 鲁棒恢复

核心问题：

真机上最有价值的不一定是跑得很快，而是能在扰动、滑移、低电量、电机误差下不摔，并能恢复稳定步态。

可探索内容：

- standing recovery：从小角度倾斜、推搡后恢复站立。
- gait recovery：跟踪参考步态过程中被扰动后恢复相位。
- command smoothing：速度指令突变时动作平滑过渡。
- contact-aware reward：不用真实足端传感器，基于仿真接触训练，真机部署只用本体观测。
- fall prevention：引入 roll/pitch safety critic 或终止风险估计。

潜在论文题目：

- 面向扰动恢复的四足机器人本体感知运动控制策略
- 结合参考步态与恢复奖励的四足机器人鲁棒 tracking 方法

这条线很适合做真机视频和毕设展示。

### 方向 E：从开源复现到自研平台的“可复现实验基准”

核心问题：

如果想把工程价值转为论文价值，可以不直接声称算法原创，而是做一个 Imgo2 平台上的系统性基准：不同训练策略在同一自研机器人上的可迁移性对比。

可探索内容：

- PPO locomotion。
- HIM-Loco/history observation。
- AMP/style reward。
- residual tracking。
- actuator-aware randomization。

论文贡献：

- 搭建统一训练/部署/评测流程。
- 给出不同策略在同一硬件平台上的仿真和真机对比。
- 总结哪些因素对低成本四足机器人最关键。

风险：

- 如果没有真机结果，这条线像 benchmark，创新性一般。
- 如果有真机结果和充分消融，会比较扎实。

## 2. 最推荐的组合路线

建议不要把论文押在单一算法名上，而是组合成一个清晰问题：

> 传感受限条件下，自研四足机器人如何通过本体感知和执行器感知训练，实现稳定的 sim-to-real locomotion/tracking？

推荐主线：

1. 本体感知策略：actor 只使用 IMU、关节状态、上一时刻动作、速度指令。
2. privileged teacher：critic 或 teacher 使用仿真中的地形高度、真实速度、接触、动力学参数。
3. 执行器感知随机化：延迟、PD、力矩限幅、摩擦、质量、质心。
4. 参考动作或 tracking 先验：用于改善步态自然性和相位稳定性。
5. 10 月真机验证：低速前进、转向、推扰恢复、不同地面测试。

一句话题目：

> 面向真机迁移的本体感知四足机器人鲁棒运动控制方法

如果强调 tracking：

> 融合参考动作先验与执行器随机化的四足机器人本体感知运动跟踪方法

## 3. 近期应该补的技术储备

### 3.1 不急着补的内容

短期不建议把大量时间投入：

- 视觉 SLAM。
- 深度相机地形感知。
- 激光雷达建图。
- 复杂落足点规划。
- 全身 MPC + RL 混合框架。

这些当然有价值，但会显著拉长学习链路，而且和 10 月真机风险叠加。

### 3.2 最值得补的内容

优先补这些：

- 状态估计基础：IMU 姿态、base angular velocity、重力投影。
- 电机控制链路：PD 控制、控制频率、延迟、限幅、保护。
- sim2real 随机化：mass、COM、friction、motor strength、action delay。
- ONNX/TorchScript 部署一致性：训练输入输出和 C++ 推理完全对齐。
- 数据记录：真机 rosbag/log、关节位置速度、电流、IMU、指令、策略动作。
- 安全状态机：站立、策略控制、急停、软限位、跌倒检测。

## 4. 10 月前的准备工作

### 阶段 1：7 月下旬到 8 月中旬，确定研究变量

目标：

- 不再泛泛调 locomotion，而是确定一个变量：执行器随机化、历史观测、residual tracking 或参考动作先验。

建议实验：

- Baseline PPO。
- PPO + dynamics randomization。
- PPO + action delay。
- PPO + motor strength/PD randomization。
- PPO + history observation。
- PPO + reference gait/residual。

输出：

- 一张主表：不同训练设置在仿真扰动下的表现。
- 一组曲线：速度误差、姿态误差、力矩峰值、跌倒率。

### 阶段 2：8 月中旬到 9 月上旬，建立部署一致性

目标：

- 让训练策略可以稳定导出，并在 deploy 侧完全复现 observation/action。

必须检查：

- joint order。
- default joint position。
- action scale。
- PD gains。
- control decimation。
- observation normalization。
- history buffer。
- quaternion/gravity projection。
- command scaling。

输出：

- `policy_export_checklist.md`。
- sim2sim 对比视频。
- 训练端和部署端同一输入下输出一致性测试。

### 阶段 3：9 月，准备真机安全链路

目标：

- 不是一上来跑策略，而是先能安全站立、切换、急停、记录数据。

必须完成：

- 电机零位与方向确认。
- 关节限位。
- 电机通信测试。
- PD 站立。
- 策略低幅值输出。
- 急停和跌倒检测。
- 数据记录脚本。

输出：

- 真机实验 SOP。
- safety checklist。
- 第一版真机部署参数表。

### 阶段 4：10 月，上真机最小实验

优先级：

1. 原地站立 30 秒。
2. 低速前进 0.2-0.4 m/s。
3. 原地转向。
4. 速度指令阶跃。
5. 轻微推扰恢复。
6. 不同地面：地砖、薄地毯、橡胶垫。

不要一开始测试：

- 高速奔跑。
- 台阶。
- 大坡度。
- 跳跃。
- 复杂 tracking 动作。

## 5. 论文可用的实验矩阵

### 主实验

| 方法 | 本体观测 | 历史观测 | 执行器随机化 | 参考动作 | 目标 |
|---|---|---|---|---|---|
| M1 PPO baseline | yes | no | no | no | 基线 |
| M2 PPO + DR | yes | no | yes | no | 鲁棒性 |
| M3 PPO + history | yes | yes | yes | no | 隐式估计 |
| M4 residual tracking | yes | yes | yes | yes | tracking 与自然步态 |
| M5 teacher-student | student yes / teacher privileged | yes | yes | optional | sim2real |

### 仿真指标

- 速度跟踪误差。
- roll/pitch RMS。
- base height RMS。
- torque RMS / peak torque。
- action rate。
- feet slip。
- contact force。
- episode failure rate。
- 扰动后恢复时间。

### 真机指标

- 连续运行时间。
- 跌倒次数。
- 急停次数。
- 电机峰值电流或峰值力矩。
- 速度指令响应时间。
- 姿态角 RMS。
- 视频定性对比。

## 6. 和传感器短板的关系

你现在缺传感器相关技术储备，这不是完全坏事。它可以反过来变成研究边界：

> 在不依赖外部地形传感器和全局定位的条件下，只使用低成本四足机器人天然具备的本体传感信息完成鲁棒运动控制。

这样写有两个好处：

- 避免被问“为什么没有视觉/激光/深度”。
- 把问题聚焦到 RL 运控、执行器建模、sim2real，这正好是你的技术栈。

论文里可以明确说明：

- 本文不处理全局导航。
- 本文不依赖外部感知地形。
- 本文关注低层运动控制器的鲁棒性和可部署性。

## 7. 最小可发表故事线

如果 10 月真机成功，最小故事线是：

1. 开源 PPO/HIM/AMP 方案直接迁移到 Imgo2 存在部署不稳定问题。
2. 分析原因：执行器延迟、动作尺度、PD 参数、观测噪声、动力学偏差。
3. 提出 actuator-aware proprioceptive RL/tracking 方法。
4. 在仿真中通过消融证明每个设计有效。
5. 在真机上完成低速运动和扰动恢复验证。

如果 10 月真机结果一般，仍可保留：

1. 完整 sim2sim。
2. 执行器误差注入实验。
3. 部署一致性测试。
4. 真机站立和低幅值策略初步验证。

但投稿说服力会明显低一些。

## 8. 当前最该做的 5 件事

1. 确定 actor 的真机可用观测，不要在训练中误用真机没有的量。
2. 给环境加入 action delay、PD gain、motor strength、friction 等随机化。
3. 做一个 policy export / deploy observation 对齐测试。
4. 把参考动作从“AMP 数据”转成“低维 tracking 约束”，例如关节相位、默认步态、残差动作。
5. 为 10 月真机准备安全状态机和数据记录，而不是只准备训练脚本。

## 9. 推荐最终方向

最推荐方向：

> 融合执行器随机化与参考动作先验的本体感知四足机器人运动控制

它的优点：

- 不完全依赖开源算法本身。
- 不要求复杂外部传感器。
- 和你的 RL 运控、RL tracking 技术栈匹配。
- 10 月真机结果可以直接成为论文核心证据。
- 就算 tracking 效果一般，也能退回鲁棒 locomotion 主线。

