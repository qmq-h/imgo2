# MuJoCo逐物理步PD、反馈延迟与IMU噪声（2026-10-23）

## 完成结果

已按用户确认实施两个连续需求：①电机PD按MuJoCo动力学计算更新，5ms循环仅发布目标/增益；②增加独立电机/IMU反馈附加延迟和MJCF可配置IMU噪声。当前入口已编译，3/3 CTest与10/10模型回归通过，默认及非零延迟/噪声的实际图形程序短程操作已通过。没有修改真实硬件工程、URDF/Xacro、策略权重、action_scale或状态增益。

验证摘要与源文件hash见[JSON](mujoco_pd_feedback_validation_2026-10-23.json)。此前[启动抽搐排查](mujoco_passive_timing_2026-10-23.md)中的“生产PD待修改”已由本次实现取代；长时步态质量与真实时序仍待量化。

## 1. 控制与反馈分层

```text
策略名义50Hz → FSM/目标发布名义200Hz → 保存整份q_des/dq_des/Kp/Kd/tau_ff
                                                ↓
每个物理步/动力学求值：当前qpos/qvel → PD → ctrl力矩限幅 → 物理推进
                                                ↓
物理步完成 → 采集电机与IMU → IMU噪声 → 各自延迟缓冲 → GetState → 上层
```

MujocoMotorPd在进程启动物理线程前安装mjcb_control回调。SetCommand发布完整命令，回调直接用当前qpos/qvel计算PD；不使用延迟观测或sensordata里的旧关节反馈。当前物理步长.5ms，对应每仿真秒2000步；这不是独立墙钟2000Hz线程。mj_forward或其它积分器也可能调用回调，因此“回调次数”不能当电机收到的新目标次数。被动0/8、固定60/2、PPO25/.5及原motor ctrlrange±23.7不变。

完整命令先验证尺寸/有限值/非负增益，再在短互斥量内原子发布。回调不推理、不打印、不等待I/O。映射经actuator→joint解析qpos/qvel地址，不把带自由关节模型的qpos槽位等同电机序号。

物理线程和应用循环退出时join，再卸载全局回调；LoopFunc新增可选join模式，仅MuJoCo入口使用start(false)，其它调用默认行为不变。初始化失败、SIGINT/SIGTERM、正常关闭均回收；信号函数只设置sig_atomic_t标志。等待viewer加载的条件变量增加退出检测，避免初始化失败时join永久等待RenderLoop。模型重载时忽略未绑定的新模型回调，主循环绑定新索引；不兼容IMU/关节布局会结束程序。实际GUI热重载仍未做专项操作，unit只验证不同model的忽略/重新绑定。

## 2. 延迟在base.yaml设置

文件：imgo2_deploy/ros2_ws/policy/imgo2/base.yaml，顶层imgo2下：

```yaml
mujoco_motor_feedback_delay_ms: 0.0
mujoco_imu_feedback_delay_ms: 0.0
```

例如电机增加1.5ms、IMU增加10ms：

```yaml
mujoco_motor_feedback_delay_ms: 1.5
mujoco_imu_feedback_delay_ms: 10.0
```

允许有限值0..1000，0关闭附加等待，支持小数；启动读取，编辑后重启，不需要重新编译。电机q/dq/tau同帧延迟，IMU quaternion/gyro同帧延迟；二者可用不同延迟，不承诺跨设备同步。

缓冲按仿真时间、因果零阶保持交付：只取时间不晚于now-delay的最新已到期源样本，不插值、不sleep、不提前释放。在初始或reset后尚无到期数据时，上层等待，PPO不推理，内部PD照常处理已有完整目标；启动先发布Passive命令，避免未初始化命令。改变映射、重载模型或时间倒退会清理历史，reset同时重置噪声随机序列。

这是**附加反馈交付延迟**，不是控制目标发送延迟；内部电机PD永远不使用这份延迟状态。源采样周期、5ms主循环和策略周期仍会增加实际观测年龄；设置1.5ms不保证策略输入总年龄恰好1.5ms。

物理推进结束时采集电机源数据；默认IMU独立200Hz采样保持。时间戳是仿真步完成的交付时标，并非保证所有sensordata严格同时采样（MuJoCo传感器求值/积分阶段本身也参与年龄）。暂停时仿真时间不前进，不虚构新样本或重复抽噪声；GUI慢放/加速下仍按仿真毫秒解释，不当作USB墙钟时间的完整模拟。程序RunModel先在短锁内取得完整robot_state副本，再锁外构造输入。

## 3. IMU噪声在MJCF设置

文件：imgo2_description/mjcf/imgo2.xml，在底部sensor段：

```xml
<framequat name="orientation" objtype="site" objname="imu" noise="0" />
<gyro name="angular_velocity" site="imu" noise="0.0002" />
```

- orientation的noise：每轴旋转向量的高斯标准差，单位rad；例如.001rad约.0573°的小角度标准差。通过小旋转四元数右乘到原姿态并归一化，不给wxyz四分量独立相加。
- gyro的noise：每轴角速度高斯标准差，单位rad/s；例如.005。均值0、三轴独立。
- 两者设0则关闭人为噪声。当前默认姿态0、gyro.0002；没有擅自把高噪声当默认。
- 本功能是测量白噪声，不是物理机体振动、偏置漂移、姿态融合滤波或真实安装杆臂模型。它会影响策略看到的数据，不直接向机器人施加扰动力。

文件上部custom段：

```xml
<numeric name="imu_sample_rate_hz" data="200" />
<numeric name="imu_noise_seed" data="12345" />
```

采样率必须为正且不超过物理步频，seed为32位非负整数。程序重启/reset时按seed重现随机序列。同一个IMU源样本在两次主循环读取之间保持相同噪声，避免把轮询频率当采样频率。改XML后关闭并重启仿真即可，无需重新编译。

**版本区别**：MuJoCo自3.1.4移除了内置传感器加噪，noise字段只是参数存储。已在MujocoFeedback中实现读取和扰动，不能只靠原版3.2.7执行mj_step自动产生噪声。[官方变更说明](https://mujoco.readthedocs.io/en/3.2.7/changelog.html#version-3-1-4-april-10th-2024)

## 4. 实际验证

- 工程内rl_sim_mujoco和新增两个test目标构建通过；CTest3/3：mujoco_motor_pd、mujoco_feedback、原feedback_delay。
- 正式PD类验证：无新目标时根据新速度翻转力矩、限幅、NaN拒绝且不部分发布、并发整命令一致性、reset、不同模型忽略/重绑、回调卸载及再次安装。5秒Passive末1秒关节速度RMS约.05127rad/s（先前200Hz采样力矩约10.24），无数值警告。
- 反馈类验证：1.5ms电机/3.3ms IMU分离交付、warmup不泄露新数据、q/dq/tau一致、reset、非法延迟、零延迟、200Hz保持、暂停无新噪声、单位四元数、可重现seed与gyro统计。给定.005rad/s标准差，12000个轴样本RMS在.0047..0053范围。
- 现有模型检查10/10、同步幂等检查及diff格式通过。XML新增sensor noise/custom没有改变18body惯性/关节/碰撞。
- 实际程序PTY键盘＋本机图形会话两次短跑：默认0/0ms；临时电机1.5ms、IMU10ms、gyro.005、姿态.001。两次均完成0→站起→1 PPO→W→Space→暂停/继续→R→SIGINT，退出0、线程回收；非零配置启动行确认读取1.5/10ms。测试结束YAML/XML逐字节恢复默认。
- 最终二进制SIGTERM正常退出0；模型已加载但机器人配置缺失时退出1，无join挂死。受限会话第一次无法初始化GLFW，随后在允许访问本机图形会话的环境完成上述测试，不把该环境失败记为模型故障。
- GUI短跑只验证状态与操作链路，不是长时间站立/行走稳定性、实测闭环频率或真实USB延迟识别；没有接触真实电机/IMU。

复现：

```bash
cd /home/tommy0929/Documents/imgo2/imgo2_deploy/ros2_ws
cmake --build cmake_build_mujoco --target rl_sim_mujoco test_mujoco_motor_pd test_mujoco_feedback test_feedback_delay -j2
ctest --test-dir cmake_build_mujoco --output-on-failure
./cmake_build_mujoco/bin/rl_sim_mujoco imgo2 scene
```

## 5. 后续待确认

长时PPO/转向/横移、延迟梯度与实机振荡频率对照、接触求解/自碰撞/执行器包络差异、GUI热重载、FIFO旧目标及遗留共享输入/参数的全面并发审查仍独立存在。不要把本次PD修复当实机抖动根因已定；也不要把模拟反馈延迟等同真实USB单一固定等待。根README保留这些待项。

## 同日核对：用户改成10ms但感觉不明显

当前用户YAML为mujoco_motor_feedback_delay_ms=10.0、imu=0.0，本轮保持不变。只启动工程cmake_build_mujoco/bin/rl_sim_mujoco imgo2 scene，实际日志反馈motor=10 ms/imu=0，退出0；未进入PPO或执行运动。新增正式feedback测试以变化的源值断言50ms读取40ms的关节帧，目标测试1/1通过，说明10ms队列交付确实生效；不是本轮实机延迟测量。

需完整退出后重新启动该部署入口，不能只在GUI重置或单独用MuJoCo自带simulate打开scene.xml：后者不调用项目的YAML读取/反馈适配器。启动日志是核对本次进程读取值的依据。当前没有保留用户上一轮进程，不能保证其此前使用的启动路径与配置版本。

窗口渲染当前物理真值，电机内部PD读取当前qpos/qvel；上层FSM/策略才读取延迟帧。固定站姿反馈近似不变，延迟几乎不改变数值；零速PPO也可能不敏感。运动时10ms会改变策略输入时刻，但不保证出现抽搐；不能用是否失稳当功能开关检测，也不能说策略50Hz所以小于20ms的延迟就无效。后续应在同一指令历史下0/10/20ms做轨迹/策略输入年龄对照，当前尚未完成这种步态定量比较。
