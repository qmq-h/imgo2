# Gazebo / ROS 2 部署链路打通记录（2026-09-17）

关联问题：DEPLOY-05（原先标记"链路不完整"）、DEPLOY-02、MODEL-02。
目标：让 `imgo2_deploy` 的 Gazebo 入口能在本机（ROS 2 Humble + Gazebo 11）跑起来，
以便在另一个仿真器里看策略步态。

## 1. 原先缺什么

`imgo2_deploy` 的 Gazebo 链路来自上游 rl_sar：URDF 侧只有 `<transmission>`
（ROS 1 `hardware_interface/EffortJointInterface`），`gazebo.xacro` 里挂的是
**`liblegged_hw_sim.so`**（`gazebo_ros_control` + `legged_gazebo/LeggedHWSim` 插件）。
该库不在仓库内，本机也没有，ROS 2 下更不存在 `gazebo_ros_control` —— 所以
`rl_sim.cpp` 启动的 controller spawner 找不到 controller_manager。

## 2. 改成了什么（对照参考项目）

参考项目 `~/RL/sim2sim/Imgo2_deploy/imgo2_description/xacro/gazebo.xacro` 走的是
**ROS 2 的 `gazebo_ros2_control`**，本机 `/opt/ros/humble/lib/libgazebo_ros2_control.so`
与 `ros-humble-gazebo-ros2-control 0.4.10` 都在，于是照它改：

| 文件 | 改动 |
|---|---|
| `imgo2_description/xacro/gazebo.xacro` | 加 IMU 传感器（`libgazebo_ros_imu_sensor.so`，挂 `base_imu`，remap `~/out:=imu`）、`<ros2_control name="GazeboSystem">`（12 关节：effort 命令 + position/velocity/effort 状态）、`libgazebo_ros2_control.so` 插件（`<parameters>` 指向本包 config）；删掉不可用的 `liblegged_hw_sim.so` 与 ROS 1 的 p3d 插件 |
| `imgo2_description/config/robot_control_ros2.yaml` | 新增：`controller_manager`(`update_rate 1000`)、`joint_state_broadcaster`、`robot_joint_controller: robot_joint_controller/RobotJointControllerGroup`（仓库内那个控制器插件） |
| `imgo2_description/{CMakeLists.txt,package.xml}` | 做成标准包：ament（`config launch meshes urdf xacro mjcf` 全装）+ `package.ros1.xml`/`package.ros2.xml` + 由 `build.sh` 替换的 `package.xml` 软链（与 `src/` 下其它包同一套机制） |
| `imgo2_description/xacro/{core.xacro,robot.xacro}` | 网格前缀改为 xacro 属性 `mesh_prefix`：纯 URDF 仍 `../meshes/`，**Gazebo 组装改用 `package://imgo2_description/meshes/`**（Gazebo 经 ament 解析，不必依赖 URDF 文件位置） |
| `imgo2_deploy/build.sh` | 包扫描改 `find -L src`，这样软链进来的 `imgo2_description` 也会被替换 `package.xml` |
| `imgo2_deploy/src/imgo2_deploy/launch/gazebo.launch.py` | 见下 |
| `imgo2_rl/scripts/tools/check_model_sync.py` | 网格引用检查同时认 `../meshes/...` 与 `package://imgo2_description/meshes/...` |

工作区里 `imgo2_deploy/src/imgo2_description -> ../../imgo2_description` 是软链（不复制模型，
模型唯一源仍在仓库根）。

## 3. 两个真正的坑（都实测定位）

1. **`<parameters>` 必须是真实文件路径。** `gazebo_ros2_control` 0.4.10 不解析 `package://`：
   写 `package://imgo2_description/config/robot_control_ros2.yaml` 时插件 `Load()` 抛异常
   （Gazebo 只打印 `Exception occured in the Load function of plugin [gazebo_ros2_control]`），
   换成绝对路径后 0 异常、`/gazebo_ros2_control` 节点出现。→ launch 里把
   `package://imgo2_description` 一律替换成实际 share 路径，并写一份临时 URDF 给 Gazebo。
   相对路径（`../config/...`）同样不行。
2. **URDF 传给 controller_manager 时必须压成单行。** 插件把 URDF 以
   `--param robot_description:=<xml>` 的形式交给 controller_manager，带换行/XML 声明时 rcl 报
   `Couldn't parse parameter override rule`，controller_manager 拿不到 URDF，控制器因此起不来
   （现象是 spawner 一直 `Could not contact service /controller_manager/list_controllers`）。
   → launch 里 `" ".join(text.split())` 去掉换行与 `<?xml ...?>` 声明。

另外：不要用 `gazebo_ros` 自带 launch。本机实测它起的 gzserver 没有加载
`libgazebo_ros_factory.so`（`/spawn_entity` 永远不出现），手动 `gzserver -s ...` 却正常，
所以我们的 launch 直接 `ExecuteProcess(["gzserver", "-s", "libgazebo_ros_init.so",
"-s", "libgazebo_ros_factory.so", world])`，`gui:=true` 时另起 `gzclient`。

## 4. 怎么跑

```bash
cd ~/RL/imgo2/imgo2_deploy
source /opt/ros/humble/setup.bash

# 编译（含 imgo2_description / robot_msgs / robot_joint_controller / imgo2_deploy）
bash build.sh                 # 或 colcon build --merge-install --symlink-install
source install/setup.bash

# 终端 A：Gazebo（GUI 窗口）
ros2 launch imgo2_deploy gazebo.launch.py                 # 默认 earth 世界
#   wname:=stairs 换楼梯世界；gui:=false 无头

# 终端 B：策略节点（键盘在这里按）
ros2 run imgo2_deploy rl_sim
#   0 起身 → 1 PPO（新权重）→ 2 himloco（Go2 占位）→ 3 AMP → 9 下蹲 → P 回 Passive
#   W/S/A/D/Q/E 给速度指令，Space 清零
```

`rl_sim` 自己会用 `controller_manager` 的 spawner 起
`robot_joint_controller`（关节名单由 `base.yaml` 的 `joint_names` 经临时 yaml 传入），
所以不需要手动 spawn 它。

## 5. 已验证到什么程度

无头单次调用内跑通（`gui:=false`，`DISPLAY` 未设）：

- `SpawnEntity: Successfully spawned entity [imgo2_gazebo]`，网格 0 处报错；
- 节点出现 `/gazebo`、`/gazebo_ros2_control`、`/imu_plugin`、`/robot_state_publisher`；
- `ros2 control list_controllers` → `joint_state_broadcaster ... active`；
- `/joint_states` 与 `/imu`（`frame_id: base_imu`）都在发布；
- 启动 `./install/lib/imgo2_deploy/rl_sim` 后控制器链路正常；用
  `ros2 topic pub -r 10 /joy sensor_msgs/Joy "{buttons:[1,...]}"`（A=起身）再
  `"{buttons:[0,0,0,0,0,1,...], axes:[...,1]}"`（RB+DPadUp）后，`rl_sim` 打印
  **`RL Controller [ppo]`**，说明已进入键 1 的 PPO 状态、策略在闭环里跑。

**未验证**：GUI 画面（本机 `DISPLAY=:1` 建 GL 上下文失败：`X_GLXCreateContext ... BadValue`，
gzclient 大概率也渲染不了，需要可用显示或 `LIBGL_ALWAYS_SOFTWARE=1`）、步态观感、
himloco/AMP 在本链路的长时间行为、真机。

**限制**：以上验证都在 harness 的沙箱 shell 里完成（每次调用独立 PID/`/tmp` 命名空间，
后台进程会被收掉），所以只能"单次调用内"跑完整链路；用户自己的终端不受此限。
`~/.ros` 与 `~/.gazebo` 需可写（沙箱里用 `HOME`/`ROS_LOG_DIR` 指到工作区）。
