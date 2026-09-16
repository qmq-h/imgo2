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

# 0) 关键：用「干净」的 Python 环境。若 shell 里 source 过 Isaac Lab / 有 conda 环境，
#    它们的 PYTHONPATH 会带进一份给 Python 3.11 编的 numpy，把系统 Python 3.10 的 numpy 顶掉，
#    于是 launch 和 rl_sim 内部调用的 Python `spawner` 会崩（见第 6 节）。
unset PYTHONPATH PYTHONHOME          # 若激活了 conda：conda deactivate
python3 -c "import numpy, rclpy; print(numpy.__file__)"   # 不应指向 isaac 的 pip_prebundle

source /opt/ros/humble/setup.bash    # 顺序很重要：先清干净，再 source ROS（它会加回自己的路径）
bash build.sh                        # 或 colcon build --merge-install --symlink-install
source install/setup.bash

# 终端 A：Gazebo（GUI 窗口）；这个终端也要同样清掉 PYTHONPATH
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

## 6. 环境陷阱：Isaac Lab 的 `PYTHONPATH` 会让 Python `spawner` 崩

**现象**（用户实跑）：`ros2 run imgo2_deploy rl_sim` 在 `Get param` / `Entered passive mode` 之后
抛一长串 Python traceback，最后：

```
ImportError: Error importing numpy: you should not try to import numpy from
        its source directory; ...
[ros2run]: Process exited with failure 1
terminate called after throwing an instance of 'std::runtime_error'
  what():  Failed to start joint controller
```

**根因**：`rl_sim.cpp` 的 `StartJointController()` 是用 `sh -c "ros2 run controller_manager spawner
robot_joint_controller -p <临时yaml>"` 起控制器的，子进程继承当前环境。若这个 shell 里 source 过
Isaac Lab（或挂着 conda 环境），`PYTHONPATH` 里会有
`/home/qmq/isaac/IsaacLab/_isaac_sim/extscache/omni.kit.pip_archive-*/pip_prebundle`——那是一份给
**Python 3.11** 编的 numpy 1.26.0。`spawner` 跑的是系统 **Python 3.10**，import numpy 时先命中它，
C 扩展对不上就崩；spawner 非 0 退出 → `rl_sim` 抛 `Failed to start joint controller`。

**本机复现**（同一台机器、同一份 ROS 2）：

```bash
source /opt/ros/humble/setup.bash
ISAAC=/home/qmq/isaac/IsaacLab/_isaac_sim/extscache/omni.kit.pip_archive-*/pip_prebundle
/usr/bin/python3 -c "import numpy; print(numpy.__file__)"                     # /usr/lib/python3/dist-packages/numpy（1.21.5，正常）
PYTHONPATH="$PYTHONPATH:$ISAAC" /usr/bin/python3 -c "import numpy; print(numpy.__file__)"   # 报 "its source directory"（复现）
PYTHONPATH="$PYTHONPATH:$ISAAC" ros2 run controller_manager spawner --help    # 同样报错（复现）
ros2 run controller_manager spawner --help                                    # 正常
```

**反向的同一个坑**：如果**先** `source /opt/ros/humble/setup.bash`、**后**才 `unset PYTHONPATH`，
就把 ROS 自己的 `site-packages` 也删掉了，`ros2` 立刻变成

```
importlib.metadata.PackageNotFoundError: No package metadata was found for ros2cli
```

因为 `ros2cli` 的 dist-info 就在 ROS 的 python 路径里，而那条路径是 `setup.bash` 追加到
`PYTHONPATH` 的。此时只要**重新** `source /opt/ros/humble/setup.bash`（再 source 一次工作区
`install/setup.bash`）即可恢复。

**污染源**：Isaac Sim 的 `~/isaac/IsaacLab/_isaac_sim/setup_python_env.sh` 会把
`.../extscache/omni.kit.pip_archive-*/pip_prebundle` 写进 `PYTHONPATH`。所以规矩是
**Isaac Lab 训练与 ROS 2 部署分用不同终端**，ROS 终端里不要 source 这套脚本、也不要 activate
`isaaclab` conda 环境。

**验证过的正确做法**（本机实测，numpy 落在系统路径、`ros2cli` 元数据正常、包能解析）：

```bash
# 终端 A / B 都这样起（不依赖 .bashrc 是否已 source 过 ROS）
env -u PYTHONPATH bash -c 'source /opt/ros/humble/setup.bash &&   source ~/RL/imgo2/imgo2_deploy/install/setup.bash && exec ros2 run imgo2_deploy rl_sim'
# 启动 Gazebo 的那个终端把最后一句换成：
#   exec ros2 launch imgo2_deploy gazebo.launch.py
```

**解决**：跑 Gazebo/ROS 2 的终端要干净——先 `unset PYTHONPATH PYTHONHOME`（有 conda 就
`conda deactivate`），**然后**才 `source /opt/ros/humble/setup.bash`（它会加回 ROS 自己的路径；
顺序反了会把 ROS 的路径也清掉）。launch 那个终端同样要清，因为它也要起
`joint_state_broadcaster`（同样是 Python spawner）。Isaac Lab 训练与 ROS 2 部署请用不同终端，
不要共用一套环境变量。

## 7. 与 MuJoCo 的数值对比：判据必须看"位姿"，不能只看姿态

用户反馈"Gazebo 完全不行、四脚朝天也显得稳定"。**只看姿态/关节会被骗过**（躺平、四脚朝天都能保持
"稳定"），所以必须同时看**基座高度与位移**。为此在 `gazebo.xacro` 里加回了基座真值插件
`libgazebo_ros_p3d.so`（发布 `/odom`，`nav_msgs/Odometry`，100 Hz，body `base`，world 系）：

```bash
ros2 topic echo /odom --field pose.pose.position --once     # 看 x（位移）与 z（高度）
ros2 topic hz /odom                                          # 应为 ~100 Hz
```

**实测对比**（同一策略 `ppo`=rough/16-37-38，同一台机；Gazebo 稳态窗口取后 7 s，MuJoCo 为无头
harness 结果）：

| 指标 | MuJoCo | Gazebo |
|---|---|---|
| `vx=0` 基座高度 z | 0.327 m | 0.333 m（0.326–0.339） |
| `vx=0` 位移 | 0.023 m | −0.001 m |
| `vx=0` 大腿摆幅 | 0.001 rad | 0.079 rad |
| `vx=0.5` 基座高度 z | 0.26 m | 0.276 m（0.244–0.332） |
| `vx=0.5` 实测速度 | **0.51 m/s** | **0.502 m/s** |
| `vx=0.5` 大腿摆幅 | 0.88 rad | 0.89–1.20 rad |
| `vx=0.5` roll 范围 | — | ±6° |

→ **两条链路在高度（±0.02 m）、速度（±0.01 m/s）、步态幅度（±0.3 rad）上一致，Gazebo 侧没有
"完全不行"**。GetUp 阶段同样正常（roll/pitch ≈0.1°、关节跟到 `0/0.87/-1.82`、命令 kp=60/kd=2）。

### 7.1 为什么会被看成"完全不行"——两条路径的指令来源不同

- **`/cmd_vel` 默认不生效**：`rl_sim.cpp:465` 只有 `control.navigation_mode` 为真时，`cmd_vel`
  才会写进 `obs.commands`。默认 OFF，此时给 `/cmd_vel` 完全无效果（机器人只会站着）。
- **ROS 路径的速度指令有三个来源**：① 手柄（`/joy` 的 `axes[1]`=vx、`axes[0]`=vy、`axes[3]`=yaw）；
  ② 键盘 `W/S/A/D/Q/E`（`RL::KeyboardInterface`，0.1/次步进，**要求 `rl_sim` 的 stdin 是终端**，
  无 TTY 时收不到按键）；③ 先按 `N`（或手柄 `X`）打开 navigation mode，再用 `/cmd_vel`。
- **键盘 `W/S` 在这两条路径都能用**，但 MuJoCo 窗口用的是 GLFW 键盘、ROS 路径用的是 stdin 的
  `kbhit`：从别的程序/无 TTY 启动 `rl_sim` 时按什么都没反应，看起来就像"策略坏了"。
- **别按错键**：键 `2` 是 himloco（Go2 参考占位），会把机器人掀翻；键 `3` 是 AMP（只站不走）；
  `vx` 建议留在训练范围 ±1.0 m/s 内。

无手柄时的最小验证（本机实测可用，`vx=0.5` 稳定走到 0.502 m/s）：

```bash
# 进 PPO 并给 vx=0.5（axes[1]=LY=vx；buttons[5]+axes[7]>0 = RB+DPadUp = 键1）
ros2 topic pub -r 20 /joy sensor_msgs/Joy \
  "{buttons: [0,0,0,0,0,1,0,0,0,0,0], axes: [0,0.5,0,0,0,0,0,1]}"
```

判据建议固定成三件套：**基座 z**（是否站立高度）、**x 位移/时间**（是否按指令速度走）、
**大腿摆幅**（是步态还是站着抖），单看任何一个都会误判。
