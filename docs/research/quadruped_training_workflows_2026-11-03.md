# 开源四足训练流程：从零、课程与续训（2026-11-03核对）

## 核心结论及证据边界

公开实现中明确存在“从零启动一个run，训练过程中自动调整地形/指令分布”的完整流程，不要求人工先训静站、再低速、再高速，每阶段换checkpoint续训。也存在固定速度范围配地形课程，以及命令分布课程而不启用地形课程的任务，不能把两类课程等同。

依据是作者仓库README、具体机器人配置、运行时采样/课程代码和论文。仅有resume开关或从零默认值，不能证明作者研发时从未续训或改奖励；公开默认配置也不是完整私人调参流水账。以下为核对日main/master内容，后续分支可能改变。未复跑这些外部项目，不宣称其参数适用于Imgo2。

## 1. ETH legged_gym：从零rough + 在线地形课程

[项目与运行说明](https://github.com/leggedrobotics/legged_gym)、[基础配置](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py)、[ANYmal C rough配置](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/anymal_c/mixed_terrains/anymal_c_rough_config.py)。

rough继承terrain.curriculum=True，命令课程False，前后/横移配置均±1m/s，max_init_terrain_level=5，runner.resume=False。因此该公开配置允许从零在rough任务中训练，地形等级在运行内更新，无需先训练flat checkpoint。初始不是全部level0；课程会按各环境运动距离升降级。flat任务是另一独立配置，不能据其存在推断所有rough都必须先flat预训练。

[环境代码](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py)中reset_idx触发地形课程：走过地形长度一半升一级，完成距离不足相对指令预期且未升则降级；达到最高等级会重新分配等级。这不是训练到固定1000轮就统一跳到下一关。README明确提供--resume，表示支持恢复，不是要求必走流程。

## 2. HIMLoco A1：地形 + 速度课程，同run保留低速样本

[README](https://github.com/InternRobotics/HIMLoco)、[A1配置](https://github.com/InternRobotics/HIMLoco/blob/main/legged_gym/legged_gym/envs/a1/a1_config.py)、[基础配置](https://github.com/InternRobotics/HIMLoco/blob/main/legged_gym/legged_gym/envs/base/legged_robot_config.py)。

A1启用命令课程、max_curriculum=2m/s，初始vx范围±1m/s；地形课程继承开启，resume默认False。README直接给train.py入口，未要求先完成静止/低速checkpoint。

[实际采样与更新](https://github.com/InternRobotics/HIMLoco/blob/main/legged_gym/legged_gym/envs/base/legged_robot.py)：_resample_commands先给所有重采样环境vx∈[-1,1]，再给环境编号前20%覆盖为当前command_ranges vx。其余约80%维持[-1,1]，不是始终随着上限一起扩大。两组tracking_lin_vel均超过各自80%理论最大贡献门槛时，vx边界各扩.2至A1设置的±2。高速度环境|vx|>=1时横向指令置0；小xy指令范数<=.2也置0。此速度课程主要扩展vx，不能泛称vx/vy/yaw都自动扩展。heading_command=True时yaw还由heading误差重算，因此配置yaw范围不等于独立均匀角速度采样。

这给Imgo2的启发是：可以同一run连续训练，并保留基础运动样本，避免训练分布只朝高速度迁移。其低速区是±1m/s，不等于适合用户目标的±.3区间；比例和阈值也不能未经验证照搬。

## 3. Rapid Locomotion：速度网格课程，不一定开地形课程

[作者仓库](https://github.com/Improbable-AI/rapid-locomotion-rl)、[论文](https://arxiv.org/abs/2205.02824)、[Mini Cheetah配置](https://github.com/Improbable-AI/rapid-locomotion-rl/blob/main/mini_gym/envs/mini_cheetah/mini_cheetah_config.py)。论文将自适应速度课程列为核心方法之一。

公开Mini Cheetah配置关闭terrain.curriculum、terrain_noise_magnitude=0，启用command_curriculum，初始vx/vy±.6、yaw±1；[RewardThresholdCurriculum](https://github.com/Improbable-AI/rapid-locomotion-rl/blob/main/mini_gym/envs/base/curriculum.py)按线/角速度奖励阈值增加成功速度格及邻近格的采样权重。速度极限/网格定义与初始采样支持集不同，不能把limit_vel_x的±10看成第一轮就均匀采±10。

[训练入口](https://github.com/Improbable-AI/rapid-locomotion-rl/blob/main/scripts/train.py)创建环境/runner后learn(4000)；这是公开单次运行预算，不代表所有实验都4000轮。[Runner](https://github.com/Improbable-AI/rapid-locomotion-rl/blob/main/mini_gym_learn/ppo/__init__.py)默认resume=False。README还介绍teacher-student方案，这是特权信息/适应模块训练安排，不等同于手工先低速再高速换奖励。

## 4. 对Imgo2当前工作的含义（本项目建议，不是论文结论）

1. 不应说“四足必须人工分几轮训练”。固定清晰目标、合理采样和自动课程，可以从零连续完成rough训练。
2. 也不应说“别人都是开始就最大速度/最难地形，然后一直等”。地形分级、速度支持集、采样权重与低速保留均可能动态变化。
3. 当前最新9999保存配置是地形课程开启、vx±1.2/vy±.8/heading控制、静止2%、速度课程关闭，属于较广范围rough单run，路线本身有公开先例。问题仍在目标是否符合用户慢步/稳定/高机身要求，以及采样是否充分覆盖低速场景；不能只凭课程开启判断配置正确。
4. 前面建议短续训/单项对照，是针对已有可用checkpoint的调参验证、节约重复长训，并非PPO/四足训练的必需步骤。短对照确认方向后，可以定一套完整配方再从零训，自动课程也可以贯穿这个新run。
5. 续训可用于中断恢复、增加预算或适配目标；奖励/采样变化会改变优化目标，不能保证保留原步态优点。算法/观测/模型/归一化契约改变时另需评估加载兼容性。
6. 不直接拿外部max_iterations和本项目10000比较，需看num_envs、rollout、物理/控制dt及实际评测。HIM公开rollout100，本项目PPO rollout24，同轮数数据量不同；大max_iterations字段可能只是上限。

已完成：核对上述作者仓库/论文与配置继承、在线采样代码，澄清从零课程与手工续训的区别。未改项目训练/部署配置、未下载运行外部项目、未启动新训练。待确认：适合Imgo2的课程/采样配方及奖励权重仍需短对照和实际轨迹验证，不把研究建议标为已解决。
