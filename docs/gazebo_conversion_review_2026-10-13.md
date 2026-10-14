# Gazebo sim2sim 转换复核

日期：2026-10-13。用户追问部署代码和转换是否可以确认没有问题。关联AMP-17/20、DEPLOY-14。

后续状态：Gazebo AMP策略裁剪/last_action已修复并通过编译、数值与四方向实际回放；未改善整体贴地，controller额外限位等仍待确认，见 [修正对照](gazebo_amp_clip_fix_comparison_2026-10-13.md)。以下保留本次修复前的复核依据。

## 已验证与未验证的边界

**不能确认整条Gazebo sim2sim链路没有问题。** 策略文件导出一致、原Isaac也有贴地行为，只能排除对应的部分解释，不能据此证明部署转换正确。

| 转换环节 | 依据与状态 |
|---|---|
| checkpoint actor → policy.pt | 上轮128组确定性输入max diff=0，部署libtorch加载/有限值推理通过；部署文件与官方导出逐字节一致。没有发现网络权重导出错误 |
| 观测拼接 | 源码/保存配置核对45维顺序：身体角速度、投影重力、命令、相对关节角、关节速度、上一动作；缩放一致。last_action的裁剪语义存在下述差异 |
| 四元数/重力 | ROS orientation按w/x/y/z送入部署逆旋转；real模型base_imu固定姿态rpy=0、initial_orientation_as_reference=false。C++真实QuatRotateInverse对独立Rodrigues矩阵500姿态/1000向量检查，最大误差2.35202e−7；yaw/pitch90°符号检查通过。只验证数学转换，不等同于运行时传感器时序对齐 |
| 关节顺序 | 当前模型12活动关节为FL/FR/RL/RR逐腿hip/thigh/shank；ROS2 controller通过OrderJointsByModelOrder重排名字，恒等mapping与该顺序匹配。当前未发现左右腿交换；未对所有平台/模型覆盖保证 |
| 动作仿射转换 | action_scale=.125/.25/.25、default q=0/.87/−1.82、策略PD25/.5与训练配方一致；裁剪阶段不一致 |
| 执行与切换 | PD速率、执行器限幅模型、线程时钟/反馈以及站立→AMP切换还存在差异；切换目标跳变已实测，过渡候选未稳定改善抖动，默认关闭 |

来源/逐项结果沿用 [主排查](gazebo_amp_motion_new_25000_2026-10-13.md)，足端问题见 [接触与高度检查](amp_foot_lift_review_2026-10-13.md)。本轮没有重新导出策略，也没有再次启动仿真。

## 明确的转换问题：AMP-20

训练原生JointAction先`q_target=raw_action*scale+default_q`，然后裁剪q_target到±3 rad；AMP wrapper不裁剪raw，last_action观察保留原始网络动作。

Gazebo当前`RL_Sim::Forward()`先把raw_action裁剪到±3，再由`RL::ComputeOutput()`缩放/偏置；下一步`obs.actions`也是裁剪后的值。因此动作执行和反馈两处都不完全等价。

以小腿通道raw=−5、scale=.25、default=−1.82为例：

- 训练：clip(−5×.25−1.82)=−3.00 rad；下一步last_action=−5。
- Gazebo：clip(−5)=−3，目标−3×.25−1.82=−2.57 rad；下一步last_action=−3。

相差0.43 rad。这是可由源码直接确定的差异，不是模型惯量问题。上轮从执行目标反推的稳定窗口边界比例约0–.67%，因此不能断言该差异解释了全部贴地和切换抖动；需要原始网络动作记录及修正后单变量运行，包含起始瞬态。

优先对齐该任务的裁剪阶段和last_action，避免破坏PPO/HIM/MuJoCo等既有配方，再判断Gazebo剩余差异与训练奖励。原Isaac也贴地说明策略品质同样需要改进，不能把全部问题推给一侧。

## 本轮完成与待完成

已修复/完成：补修amp/config.yaml仍把数值相同描述为匹配训练的旧注释；新增真实逆旋转数学回归，c++17/O2编译并运行通过；本轮git diff检查通过。配置数值、部署控制实现、模型和policy没有修改。

待修复/确认：AMP-20裁剪/last_action代码修复、端到端同状态观测/目标对照、原始网络动作及反馈延迟测量、修正后Gazebo单变量方向与切换对照。DEPLOY-14整体切换抖动仍未修好。**本轮复核不把整条转换链路标为通过。**

可复现数学检查（仓库根）：

```bash
c++ -std=c++17 -O2 imgo2_deploy/tests/test_amp_quaternion_conversion.cpp \
  -o /tmp/imgo2_amp_quaternion_conversion_test
/tmp/imgo2_amp_quaternion_conversion_test
```
