# PPO 上机前接口核对（2026-10-14）

本轮只读源码、配置与权重哈希，不启动硬件、不修改go_imu或策略。git pull因.git/FETCH_HEAD只读失败，依据当前本地版本。

## 已核对／已澄清

- PPO policy.pt SHA256为53a57f909b61d2a708d24d6c08a90e086409b6d4fce4fdd3e6f3e1849abce7cf，与[历史来源核对](ppo_policy_provenance_2026-09-19.md)记录的policy_flat.pt一致，不是参考侧另一个policy.pt。历史Gazebo验证不等于真机验证。
- go_stand四腿目标0/0.87/-1.82 rad，与ppo default_dof_pos一致；go_stand输出侧kp25/kd0.5与ppo rl_kp/rl_kd一致。PPO fixed_kp60/fixed_kd2为另一组，不能当go_stand同参数。
- 启动姿态3.49/67.43/-165.01度（部分hip负号）用于机械初始姿态归零，与最终站姿是两件事；运行过程不应再次归零。
- 电机接口输出侧q rad、dq rad/s、tau N·m，协议处理6.33减速比、PD增益除6.33²及方向/offset。tau为理想减速比换算估计，不是独立测得足端力或真实关节力矩。
- IMU解析保持wxyz、归一化四元数、gyro deg/s→rad/s、Euler pitch/roll/yaw→roll/pitch/yaw rad，加速度m/s²；数据仍为传感器坐标。策略用q+gyro生成投影重力，不使用显示RPY替代四元数。
- 当前路径为/dev/robot/leg_fl、leg_fr、leg_bl、leg_br和/dev/robot/imu；不是旧的默认CH343/ACM路径。本会话未观察到/dev/robot，设备可用性须在硬件主机确认。

## 待修复／待确认

- rl_real_imgo2仍为Unitree后端；go_imu没有策略运行入口。需要在同一进程中一次start→平滑站起→站姿保持→PPO，不先退出go_stand再重启归零。
- hardware配置joint_mapping_confirmed/imu_mount_confirmed仍false，映射为空；leg_*名称/注释不能代替物理接线验证。IMU旋转须按安装和设备姿态约定实测后应用。
- config.limits为空：clip_actions±3仅是网络动作裁剪。髋目标可到±0.375rad，大腿0.12..1.62rad，小腿-2.57..-1.07rad，不能据此认定机械安全。tau_max仅限制前馈，不能独自限制电机内部PD力矩。
- 控制总线目标200Hz、策略目标50Hz；实际频率、延迟、坏帧和丢帧须在实机记录。反馈成功时间是主机时间，不是设备同步采样。
- RealHardware具有电机100ms/IMU500ms过期和整机锁定；有效标志与新鲜度不等于安装轴、物理符号及四元数方向正确。ImuData.CopyTo额外要求加速度同帧，而RealHardware就绪只要求q+gyro；未来适配不要混用这两个条件。
- 保留前次DEPLOY-13的共享状态同步、完整命令交接、旧动作清理、故障/退出卸力，以及DEPLOY-12力矩队列未消费问题。

完成条件：确认硬件实际站起/方向及反馈；水平和单轴转动验证机身角速度、投影重力；确认限幅/急停/故障卸力；无下发策略观察与输出检查后再受控接入零速度策略。零速度指令仍可能使机器人调整或抬脚，不保证输出全零动作。

验证范围：配置/源码静态核对、policy字节哈希、文档检查；本轮未重新加载推理模型、构建、跑测试或进行实机试验。仅更新维护文档，未宣称任何真机问题已修复。
