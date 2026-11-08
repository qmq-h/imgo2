# 旧9299 Gazebo站立→PPO前进→站立全过程采集（2026-11-07）

## 已完成与范围

按用户最终要求，实际启动无界面Gazebo/ROS2：先起身并站立，切入PPO直接请求vx=.5/vy=0/yaw=0，观察到PPO PD命令后计10秒，再请求GetUp/站立并继续采集5秒。不是MuJoCo或Isaac替代，也不是只保存回放摘要。隔离domain67/master11377，仅启动并关闭本次拥有的进程，结束后gzserver/rl_sim无残留，未打开硬件、未训练或替换策略。

已修复/完成：原IMGO2_POLICY_TRACE只记录AMP，现扩展为PPO/AMP可选记录；保留原71列并追加状态/时间/实际处理后动作/PD字段。未改Forward输出、裁剪、关节映射或PD计算。rl_sim编译通过，真实PPO采集和CPU重放通过，状态/命令/IMU/odom完整保存并整理。AMP新增列未在本轮重跑闭环，不冒称AMP回归已完成。

待修复/待确认：实机相同阶段数据尚未提供、实机日志实际输入与输出配对问题仍待DEPLOY-DIAG-01；仿真与实机的采样/执行时序、反馈年龄需使用各自时标对照。本轮没有任何硬件验证。

## 输出位置

数据目录：`/home/tommy0929/Documents/imgo2/imgo2_deploy/logs/gazebo_old9299_forward05_20261107_172338`。

单文件压缩包：`/home/tommy0929/Documents/imgo2/imgo2_deploy/logs/gazebo_old9299_forward05_20261107_172338.tar.gz`（约10.6MiB，原始数据约27.3MiB）。日志目录由根.gitignore的logs规则忽略，不随源码Git上传；需要跨机对比请自行传目录/压缩包。机器可读证据另存[验证摘要](gazebo_forward05_full_capture_2026-11-07.json)，无需把原始大型日志入库。

| 文件 | 内容 |
|---|---|
| policy_raw.csv | 原始182列、482次实际推理，成对45维输入/12维原始输出/处理后动作/关节目标/输入反馈快照/PD |
| policy.csv | 同一数据改为obs_00…44、act_raw_00…11、act_00…11、q_des等一致命名，时间以PPO开始为0 |
| robot.csv | 3721行实际命令，加此前最近到达的反馈/IMU/odom；各数据源匹配年龄单独保留，不伪装同时采样 |
| state.csv | 20571行原始12关节q/dq/tau_est反馈 |
| command.csv | 3721行实际12关节q/dq/tau/kp/kd下发 |
| imu.csv | 4137行带消息时标的四元数/角速度/加速度 |
| odom.csv | 2076行真值位姿、四元数及插件twist；身体前进速度另由相邻位姿差分投影计算 |
| velocity.csv / events.log | 请求指令、阶段与切换时刻 |
| clock.csv / feedback_age.csv | ROS时钟与主机单调时钟映射、控制反馈年龄 |
| policy_used.pt / policy_config.yaml / base_config.yaml | 使用的策略与配置快照；没有复制机器人URDF模型 |
| metadata.json / collection.json / schema.json / manifest.json | 策略/配置/程序哈希，阶段、字段单位/顺序/限制、数据文件哈希 |
| replay_check.json / summary.json | CPU重放和目标/last_action核对、统计与实际记录频率 |
| gazebo.log / rl_sim.log / recorder.log / ros_logs | 本次控制台/ROS日志 |

关节/策略顺序FL,FR,RL,RR，每腿hip/thigh/shank。角度rad、角速度rad/s、力矩N·m、位置m。45维actor为gyro*.25、投影重力、commands、q-q0、dq*.05、上一版裁剪动作。原始/裁剪动作均保存，不能混用。

## 策略与阶段核对

旧目标-.20组`2026-11-06_15-10-00/model_9299.pt`，policy SHA256=d2334e9845c237062fc0671e6df8e79440efbfa035c0c9a3c2c884daac40cfea。当前PPO PD25/.5、默认角0/.87/-1.82，额外电机/IMU延迟0，earth平地，Gazebo步长与原生产世界相同。policy/config/model/binary哈希前后未变。

仿真采集原点2.2秒，请求PPO约8.2秒，首次PPO PD命令8.3秒，返回站立请求18.3秒，继续记录至23.2秒左右。PPO PD阶段恰10.0秒，期间482次推理，其中切换初始2帧命令0，其余480帧命令.5；保留零指令过渡而不裁掉。没有5秒加速斜坡，也没有用“PPO零指令”代替最后的固定站姿。最后实际命令kp60/kd2，目标0/.87/-1.82；反馈关节速度最终最大约.000314rad/s，固定站姿目标与真实关节角存在PD静差，不声称完全重合。

CPU在所有记录输入上重算：raw最大差2.384e-6、目标转换差0、last_action反馈差0；无动作±3裁剪，无额外URDF目标限位触发。全部182列有限、列数一致、vy/yaw命令均0。

## 本次前进稳定段

采用仿真10.3–18.1秒，排除初始2秒与切回前.2秒，不用它代替整个切换段：

- 实际身体前进速度约.5049m/s。
- 横滚/俯仰角标准差约.548°/.289°。
- 横滚/俯仰角速度RMS约.291rad/s。
- 平均机身高约29.43cm，高度标准差约3.76mm。

支持当前平地前进没有大幅失稳，但不能说完全不抖或真机也同幅度。切换数据完整保留，可另行与实机做瞬态对照。

## 时间、力矩与实机对照注意点

1. policy.csv的obs与act_raw来自同一次真实推理，act为裁剪后值；实机当前logger的obs由消费时反馈重建，搭配的act是已消费的较早结果，不能直接逐行做网络重放；不要随意假设固定移一行就对齐。已有DEPLOY-DIAG-01保留待修，本轮没改实机代码。
2. 实机robot.csv是硬件顺序，按实机meta.json的joints/映射转到FL/FR/RL/RR再比较；实机act是裁剪后动作，对应仿真act列，不是act_raw。
3. command tau/tau_ff是实际前馈，下发PD关节位置/速度/kp/kd另存；diagnostic_tau是代码算出的参考PD力矩，不是实际前馈或独立实测力矩。tau/tau_est是反馈估计，仿真与驱动估计口径仍需核对。
4. /clock约10Hz，policy sim_s和无header电机消息的sim_s是最近收到的粗时钟；host_s为精确单调时钟，IMU/odom用消息自身仿真时戳。不要把同一个sim_s的多个推理样本当重复行删除。robot.csv只做向后主机到达匹配，保留age，不能声称精确同步采样。
5. 本次实际策略trace约48.5Hz主机时间；ROS命令到达率约178.6Hz，不能把名义200Hz当实测执行率，也不能凭到达率区分控制漏拍与DDS丢帧。全部来源实际率见summary.json。
6. policy_raw中的accel来自控制器输入状态结构，当前未填充而为0；物理加速度使用imu.csv或robot.csv的独立IMU流。加速度不是45维actor输入。没有伪造电机温度、错误码或总线RTT。
7. 对比两台机器以模式切换/命令边沿和相对时长对齐，不拿两机单调时钟绝对数相减；先比较同阶段速度、姿态、关节q/dq/q_des与PD，不通过任意相位平移掩盖时延差。

## 复现与实机采集方式

仿真采集脚本（已实际执行）为`imgo2_deploy/ros2_ws/scripts/record_ppo_forward_comparison.py --output <新的目录> --hold-seconds 10`，需已编译本轮rl_sim并具备正常ROS/Gazebo环境；会自行准备ROS环境和隔离进程，脚本不连接硬件。汇总使用系统python3的`imgo2_rl/scripts/tools/summarize_forward_capture.py <数据目录>`。

重放校验使用env_isaaclab Python运行`imgo2_deploy/ros2_ws/scripts/check_policy_trace.py <目录>/policy_raw.csv --policy <目录>/policy_used.pt --config <目录>/policy_config.yaml --config-key imgo2/ppo --json <目录>/replay_check.json`。

实机由用户自行操作：在原有已验证启动命令后增加日志参数（以下目录为本机路径示例，程序运行时创建时间子目录）：

```bash
--log --log-every 1 --log-dir /home/tommy0929/Documents/imgo2/imgo2_deploy/logs/real_forward05
```

使用完全相同旧-.20/9299及配置，站立稳定后切PPO请求vx=.5、vy=0、yaw=0，保持10秒，再切固定站立，继续记录至少5秒后正常退出。保留robot.csv、policy.csv、events.log、meta.json、trace.*.csv及策略哈希；相同指令不保证两边实际速度相同，实际速度无法测量时如实标记。硬件命令没有由本轮agent执行。
