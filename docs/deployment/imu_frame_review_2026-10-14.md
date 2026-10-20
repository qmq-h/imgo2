# IMU安装位置与sim2real坐标核对（2026-10-14）

## 现象与结论

用户询问仿真IMU位置及真机是否需要坐标转换，并确认真实安装轴为X向前、Y向左、Z向上。按此说明，传感器轴与策略机身轴一致，安装旋转为单位矩阵；无需换轴或翻转符号。安装方向由用户确认，本轮未测真实设备。

已澄清：仿真IMU设在base坐标原点，非严格的机身质心；硬件接口现返回sensor frame数据，mount_quaternion没有实际应用。待确认：真实设备四元数的世界参考、旋转方向与协议顺序；这些不由安装箭头单独决定。策略真机后端尚未接入，不能将此次静态核对视为真机部署通过。

## 仿真源码和生成物

- `imgo2_description/xacro/robot.xacro:48` 调用IMU宏：parent=base、xyz="0 0 0"、rpy="0 0 0"。`imu.xacro` 生成固定base_imu_joint。
- 当前Gazebo launch默认读 `imgo2_description/urdf/imgo2_real.gazebo.urdf`，可由IMGO2_GAZEBO_URDF覆盖。XML中固定关节同样为零位置/零旋转，sensor没有额外pose。此前留下的 `/tmp/imgo2_gazebo_resolved.urdf` 也一致；它不证明当前有仿真进程运行。
- Gazebo IMU发布100Hz，initial_orientation_as_reference=false；官方插件使用世界参考并直接发布sensor的姿态、角速度与加速度。[ROS2插件源码](https://github.com/ros-simulation/gazebo_ros_pkgs/blob/ros2/gazebo_plugins/src/gazebo_ros_imu_sensor.cpp)
- `rl_sim.cpp` 的ROS2分支设ang_vel_axis=body，将消息四元数整理成wxyz，角速度直接赋值。`rl_sdk.cpp` 用四元数逆旋转世界向下单位向量，得到机身重力投影。
- Isaac训练纯URDF不包含独立IMU link；actor用base_ang_vel和projected_gravity，即机身角速度和机身重力投影，不通过Gazebo传感器。
- MuJoCo的base/imu site省略pos和姿态，即默认零位置/单位旋转；本轮只核对XML，不宣称当前MuJoCo运行或其它模型同步已经通过。
- base惯性原点为(-0.00002223,-0.00136579,0.00287896)m，故base原点与质心有毫米级差别。

## PPO配置补充核对

用户明确使用PPO。读取 `imgo2_deploy/policy/imgo2/ppo/config.yaml`：num_observations=45，顺序为ang_vel(3)、gravity_vec(3)、commands(3)、dof_pos(12)、dof_vel(12)、actions(12)。IMU相关输入仅为机身角速度与由四元数计算的重力投影，不直接输入加速度、欧拉角或四元数四个数本身。ang_vel_scale=.25，由策略观测构造应用一次；真机parser输出仍保持rad/s，不能提前再次乘.25。

当前PPO rough_env_cfg.py删除actor的base_lin_vel和height_scan，base_ang_vel.scale=.25；flat_env_cfg继承该配置。与部署列出的45维IMU契约一致。四元数仍必须符合机身到世界的方向以正确计算gravity_vec；PPO算法名称不会改变安装旋转要求。因此用户确认的X前/Y左/Z上仍可采用单位安装旋转，保留设备四元数方向/世界参考的待确认项。

本补充仅配置/源码核对，修正上轮按AMP描述的适用范围；未改算法、权重、驱动或硬件，未运行策略。真机后端移植仍由DEPLOY-13跟踪。

## 真机接口与转换约定

`go_imu/Imu/inc/imu_sensor/imu_data.hpp` 明确数据仍在传感器轴，安装转换由调用方负责。yesense_parser把角速度由deg/s转rad/s、欧拉角转rad、四元数按现有wxyz接口读入并归一化。硬件snapshot没有应用安装旋转；ImuConfig的mount_quaternion={1,0,0,0}为元数据，mount_confirmed目前默认false。本轮未修改这些字段。

策略需要右手机身轴X前/Y左/Z上，与[REP103](https://github.com/ros-infrastructure/rep/blob/master/rep-0103.rst)一致。角速度必须为机身轴rad/s；四元数应表示机身到Z向上世界的旋转，内部顺序wxyz。传感器轴与世界参考是两个独立概念，NED/ENU、四元数是否表示逆方向仍须依据型号/配置或实测确认。[REP145](https://github.com/ros-infrastructure/rep/blob/master/rep-0145.rst)

一般情况下定义R_BS将sensor向量转为body向量，则omega_B=R_BS*omega_S；若q_WS表示sensor到同一世界的旋转，q_WB=q_WS*inverse(q_BS)。世界约定不同还需先转换世界参考。此次用户确认轴一致，R_BS=I，安装部分无旋转。不要为了“sim2real”额外旋转已经在body坐标的数据，也不要直接对欧拉角逐项加减安装角。

IMU装在机身偏离原点处但轴一致，刚体角速度及姿态不需要平移补偿。线加速度则会有角加速度/向心加速度的杆臂项；未来若使用加速度或估计base速度，应单独考虑。用户本次使用PPO；当前部署PPO为45维actor，不直接输入accelerometer，不能因IMU离中心几厘米就断定当前策略必须加杆臂补偿。

## 不控制电机的实测确认办法（待执行）

1. 机身水平静止：gyro三轴接近零；用输出q计算R(q)^T*[0,0,-1]，应接近[0,0,-1]，yaw可任意，不要求q恰好[1,0,0,0]。
2. 手动绕机身各轴旋转：gyro通道对应同一轴、符号遵循右手规则；从上方看向左转时gyro_z应为正。
3. 缓慢倾斜机身，比较由q得到的重力方向与实际倾斜；不能仅在水平位置判定q与其逆谁正确，也不要把开机倾斜直接当水平归零。
4. 原始加速度若为包含重力的比力，水平静止Z应约+9.81m/s²；它与策略向下的单位重力向量不是同一量。若设备输出已经去重力的数据，不能套用该检查。

完成上述检查并明确设备四元数协议后，才在策略硬件适配层确认body坐标和数据有效性。若发现世界参考或方向不匹配，在适配层一次性修正并离线/实测验证；无需盲改模型IMU安装角。

## 验证与限制

已完成：使用 `/usr/bin/python3` 解析实际Gazebo生成物和MuJoCo XML，断言base挂载、零位姿、无sensor额外pose、世界参考配置；逐段核对launch、训练观测、ROS2读取、数学函数、真机parser与hardware snapshot。机器可读证据见 [结果](imu_frame_review_2026-10-14.json)。

待确认：真实设备世界参考/四元数方向与顺序、实际倾斜/角速度符号、策略硬件适配接入。缺设备型号/协议或上述静态测量才能关闭这些待办。本轮只更新维护文档与离线结果，未改模型/配置/IMU驱动，未启动仿真、硬件或训练。
