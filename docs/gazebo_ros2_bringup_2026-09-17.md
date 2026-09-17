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

## 8. 关节顺序：三条路径的"数组"不同（Gazebo 会翻车，MuJoCo 正常）

### 8.1 现象与根因

Gazebo 里按 `1` 进 PPO 后，机器人**翻成四脚朝天**并保持不动（`/odom`：z=0.073 m、roll=180°、
dx=0）；同一策略在 MuJoCo 里站得住、按指令前进。原因不是物理，而是**关节顺序错位**。

`joint_mapping` 的语义是"策略第 i 个关节 ↔ **该路径数组**的第 `joint_mapping[i]` 号"，但三条
路径的"数组"根本不是同一个东西：

| 路径 | 被索引的数组 | 顺序由谁决定 | 恒等映射是否正确 |
|---|---|---|---|
| MuJoCo（`rl_sim_mujoco.cpp`） | `mjData.sensordata` / `ctrl` 原始数组 | **MJCF 声明顺序** = 生成时按训练 URDF = FL,FR,RL,RR = 策略顺序 | ✅ 正确 |
| ROS 2 / Gazebo（`rl_sim.cpp`） | `robot_msgs/RobotCommand`、`RobotState` 的 12 个槽位 | **controller 的 `joints` 参数顺序**，由 `rl_sim` 从 `base.yaml` 的 `joint_names` 传入（原为 Unitree SDK 的 FR,FL,RR,RL） | ❌ 策略的 FL 接到物理 FR |
| 真机（`rl_real_imgo2.cpp`） | Unitree SDK `motor_state()` 数组 | 硬件/SDK 固定 FR,FL,RR,RL | ❌ 同上 |

关键：**ROS 2 那边完全不看 URDF 顺序**——`robot_joint_controller` 按名字解析 ros2_control
接口，槽位顺序只取决于我们传给它的 `joints` 名单。所以两条路径"都从 `base.yaml` 取映射"，
但被映射的数组一个来自 MJCF、一个来自 `joint_names`，结果就不同。

### 8.2 修法（只动 ROS 路径与数据，模型/URDF 与 MuJoCo 路径不动）

1. **数据**：`base.yaml` 的 `joint_names` / `joint_controller_names` 改为**模型顺序**
   （FL,FR,RL,RR）。它们只被 ROS 路径（`rl_sim.cpp`）与 ROS1 的按名控制器使用，真机路径不读，
   所以改动不影响 MuJoCo 与真机的 `joint_mapping` 语义。
2. **代码（兜底）**：`rl_sim.cpp` 的 ROS2 分支新增 `OrderJointsByModelOrder()`，启动控制器前
   按 **URDF（模型唯一源 `IMGO2_MODEL_DIR/urdf/imgo2.urdf`）里的关节声明顺序**重排名单，
   读不到或数量不符则回退并告警。这样即使 `joint_names` 写错，ROS 路径也不会错位；
   MuJoCo 路径的代码（`rl_sim_mujoco.cpp`）与 `joint_mapping` 语义完全未改。
3. 仅针对 PPO（键 1）：它的 `joint_mapping` 是恒等；himloco 的 `[3,4,5,0,1,2,9,10,11,6,7,8]`
   在两种数组顺序下都仍然正确（该置换自逆，且 Go2 策略本身就是 SDK 顺序）。

**验证**（Gazebo，`/odom` 真值）：`vx=0` → z 0.328–0.332 m、roll ±2.9°、dx≈0；`vx=0.5` →
**向前 8.20 m / 12 s（0.56 m/s）**、z 0.26–0.27、大腿摆幅 0.99–1.08 rad。修前同一条命令是
z 0.073 m + roll 180°（四脚朝天）。

### 8.3 顺带查清的两件事

- **IMU 没有问题**：把 `/imu` 的 `angular_velocity.z` 与 `/odom` 的 yaw 速率对比，两者同号同量级
  （+0.059/+0.058、−0.055/−0.050、−0.065/−0.054 rad/s），说明 Gazebo 给的是体系角速度、约定与
  策略一致。
- **偏航/转圈是策略自身的性质，不是 Gazebo 特有**：同一策略在 MuJoCo 里 `vx=0.5` 也会持续偏航
  （12 s 内 yaw 到 −61.9°，同时仍前进 6.18 m），`vx=0` 时只有 ~0.7°/s；Gazebo 侧量级相同但
  抖动更大（`vx=0` 某些 run 到 ~7°/s）。所以"步态奇怪/走一会儿朝反方向"要往策略权重与
  `kd=0.2`（阻尼偏低）上找，而不是继续查 Gazebo 物理。下一步可对比另外几份 PPO 导出
  （flat/23-24-09 或 rough 系列其它 checkpoint），或用 `axes[3]` 给 yaw 指令看它是否真在闭环控航向。

## 9. 键 3（AMP `model_5000`）四项指标首测 + 部署导出约定（2026-09-18 补）

### 9.1 被测对象与安装动作

| 项目 | 值 |
|---|---|
| 训练 checkpoint | `policy/imgo2/amp/model_5000.pt`，sha256 `bb399eb2519fc3f1…`（文件名 5000 轮；文件内 `iter: 0`，无 `infos`，故轮次只能靠文件名） |
| actor | **45 维输入** → 512 → 256 → 128 → 12，ELU（第一层 `[512,45]`） |
| critic | 仍 48 维（`[512,48]`，保留 `base_lin_vel`，只用于训练） |
| 部署接口 | `amp/config.yaml`：`num_observations: 45`，`observations: [ang_vel, gravity_vec, commands, dof_pos, dof_vel, actions]`（即原 48 维去掉开头的 `lin_vel`，与参考 `amp_go2` 的 45 维 actor 组成一致） |
| 导出 | `amp/policy.pt`，sha256 `0e5bd661b1591a87…`，770931 B（旧 48 维那份 sha256 `e5a8bfec742038e0…`，可从 git 历史取回） |

安装动作只有两处：换 `amp/policy.pt`、把 `amp/config.yaml` 的 `num_observations`/`observations` 改成 45 维。
C++ 侧不用改（`rl_sdk.cpp` 按 `observations` 名单逐个取项，名单里没有 `lin_vel` 就自然拼出 45 维）。

### 9.2 四项指标实测（Gazebo 无头，`--key 3`，窗口 = 进入策略后 3 s 起，24 s）

| vx 命令 | z_mean（min–max） | roll / pitch 范围 | yaw 漂移 | dx / 实测速度 | 抖动 hip/thigh/shank | 周期 / 强度 / FL-FR 相位 |
|---|---|---|---|---|---|---|
| **0.0** | 0.270（0.262–0.280） | 2.55° / 2.39° | −3.75°/s | 0.584 m（0.039 m/s） | 31.7 / 36.5 / 67.0 | 0.192 s / 0.45 / −96° |
| **0.5** | 0.282（0.269–0.296） | 6.93° / 4.38° | +2.89°/s | 2.271 m（**0.151 m/s**） | 46.7 / 70.7 / 120.2 | 0.296 s / 0.21 / 249° |

对照（同一脚本、同一世界）：键 1 参考 `base_move/policy_flat.pt` 在 `vx=0.5` 是 **0.428 m/s**、
z 0.288、roll 2.5°、thigh 抖动 45.3、周期强度 0.95、相位 175°；旧的 `model_9000.pt`（48 维）在
`vx=0.5` 只有 **0.057 m/s**、z 0.182。

**读法**：

1. **管线是通的**：`rl_sim` 打印 `Successfully loaded Torch model: …/amp/policy.pt`，FSM 进入
   `RLFSMStateAMPLocomotion`；45 维观测下正指令给正向前进（0.151 对 0.039 m/s），站姿高度正常
   （0.27–0.28 m）、不翻车。所以"45 维 + 去掉 `lin_vel`"这条接口是自洽的——若观测顺序错位，
   `gravity_vec`/`commands` 会乱，机器人不可能保持高度并按指令方向走。
2. **但速度只跟到 30%**（0.5 命令 → 0.151 m/s），比旧的 48 维 AMP 好 2.6 倍，仍远低于 PPO（0.428）。
3. **步态是高频抖**：周期 0.296 s ≈ 3.4 Hz、周期强度只有 0.21（PPO 是 0.95）、FL-FR 相位 249°
   （trot 应接近 180°），shank 抖动 120 rad/s²（PPO 45）。`vx=0` 时也不安静（shank 67、漂 ~0.04 m/s、
   yaw −3.75°/s）。⇒ 5000 轮还停在"原地快速倒腿/滑行"，尚未形成清晰步态，与
   [AMP 对照报告](amp_standstill_diagnosis_2026-09-17.md) 的结论（风格奖励没在塑形、缺低姿态终止）
   一致，属于"训练轮数与奖励结构"问题，不是部署接口问题。

### 9.3 导出约定（用户 2026-09-18 决定）

**以后导出部署用的 `policy.pt` 一律走对应算法目录下的 `play.py`，headless 模式**，不要手写导出脚本、
不要手工拼 TorchScript：

```bash
cd ~/RL/imgo2                 # 训练机上跑（要有 NVIDIA 驱动 / Isaac Sim）
unset PYTHONPATH              # 先清干净再 source，见第 6 节
source ~/isaac/IsaacLab/isaaclab.sh -p $(which python3)   # 或该机既有的 Isaac Lab 启动方式
python imgo2_rl/scripts/rl_lab/amp/play.py \
    --task Imgo2-basemove-flat-amp --headless --num_envs 1 \
    --load_run <run 目录名> --checkpoint model_5000.pt
```

`play.py`（`imgo2_rl/scripts/rl_lab/amp/play.py:117`）自己调用
`export_policy_as_jit(runner.alg.actor_critic, normalizer=None, path=<run>/exported, filename="policy.pt")`，
产物落在 checkpoint 同级的 `exported/policy.pt`（还有 `policy.onnx`）。

**前提**：

- **配置必须与 checkpoint 的观测维数一致**。2026-09-18 拉取 `b9a4913` 后这一条**已满足**：
  `f028802` 已把 `observations.policy.base_lin_vel = None`（AMP-05）落进仓库，actor 观测为
  `base_ang_vel 3 + projected_gravity 3 + velocity_commands 3 + joint_pos 12 + joint_vel 12 + last_action 12 = 45`。
  本机已离线核对部署侧 `amp/config.yaml` 与训练侧配置**四处一致**（逐项见 9.5 节）；
  在此之前（部署接口还是 48 维）拿 45 维 checkpoint 跑 `play.py` 会在 `runner.load()` 的
  `load_state_dict` 处尺寸不匹配报错。
- **本机没有 NVIDIA 驱动**（`nvidia-smi` 报无法与驱动通信），Isaac Sim 起不来，所以上面这条命令
  只能在训练机执行。
- 本次的 `policy.pt` 是用 **Isaac Lab `_TorchPolicyExporter` 的等价复刻**（同一个
  `actor + Identity normalizer` 包装 + `torch.jit.script`，torch 2.7.0 CPU）导出的，并用两条数值契约
  验证：① 对 `model_9000.pt` 的 actor 用同一流程重导出，与仓库里既有的 48 维 `policy.pt` 在 128 组
  随机输入上 **max|diff| = 0.0**；② 新 45 维导出与 `model_5000.pt` 的 actor 也是 **0.0**；
  ③ 两个文件都能被部署自己的 **libtorch 2.3.0**（`torch::jit::load`）加载，且同一确定性探针输入的
  12 维输出与 Python 侧一致到 1e-6。⇒ 接口等价、可直接用于 sim2sim；正式物证仍应由训练机上的
  `play.py` 产出，替换后按同样的三条契约复核一遍即可。

### 9.4 复现这次测试的要点（本机受限环境）

1. **必须在同一次 shell 调用里跑完 Gazebo + `rl_sim` + 评测**。分成多个后台进程/job 时 ROS 2 的
   发现会失效：`rl_sim` 报 `Failed to call param_node service`、`Failed getting a result from calling
   /controller_manager/list_controllers`，FSM 拿不到 `robot_name` 直接退。仓库里临时脚本见
   `imgo2_deploy/build/run_gz_test.sh`（`build/` 被忽略，属复现脚手架，不入库）。
2. `~/.ros`、`~/.gazebo` 只读 ⇒ 用工作区做假 HOME：`HOME=<repo>/imgo2_deploy/build/fakehome
   ROS_LOG_DIR=$HOME/.ros/log`。
3. `DISPLAY` **必须 unset**：本机 `DISPLAY=:1` 的 GLX 是坏的，gzserver 会以
   `X_GLXCreateContext BadValue` 崩掉，`gui:=false` 也救不了。
4. 评测脚本新增 `--key {1,2,3}`（1=PPO/`RB+DPadUp`、2=himloco/`RB+DPadRight`、3=amp/`RB+DPadDown`，
   见 `fsm_robot/fsm_imgo2.hpp`）；`rl_sim` 每个控制周期都打印 `RL Controller [amp] x:…`，
   日志几百 KB 是正常的，抓关键行用 `grep -v "RL Controller"`。

### 9.5 部署接口 ↔ 训练配置的离线核对（2026-09-18 拉取 `b9a4913` 后补）

`f028802` 落地后，训练侧 actor 观测与部署侧 `amp/config.yaml` 已可逐项对照（本机只用
标准库读源码，不需要 GPU）。核对方法：从 `velocity_env_cfg.py` 的 `PolicyCfg` 按**声明顺序**
取 ObsTerm 列表，减去 `amp_env_cfg.py` 里被置 `None` 的项，再映射到部署侧的观测名。

| 项目 | 训练侧 | 部署侧 | 结论 |
|---|---|---|---|
| 观测项与顺序 | `base_ang_vel, projected_gravity, velocity_commands, joint_pos, joint_vel, actions`（`base_lin_vel`/`height_scan` 已置 None） | `observations: [ang_vel, gravity_vec, commands, dof_pos, dof_vel, actions]` | **逐项同序** |
| 观测维数 | 3+3+3+12+12+12 = **45** | `num_observations: 45`（`rl_sdk.cpp` 按名单取项，没有 `lin_vel` 分支） | **一致** |
| 观测缩放 | `base_ang_vel 0.25`、`joint_pos 1.0`、`joint_vel 0.05`；`projected_gravity`/`velocity_commands` 未设 scale（=1.0） | `ang_vel_scale 0.25`、`dof_pos_scale 1.0`、`dof_vel_scale 0.05`、`commands_scale [1,1,1]`、`gravity_vec` 无 scale | **一致**（`lin_vel_scale` 字段留着但已无消费者） |
| 动作空间 | `clip {".*": (-3, 3)}`、`scale {".*_hip_joint": 0.125, 其余 0.25}` | `clip_actions ±3.0`、`action_scale 0.125/0.25` | **一致** |
| 默认姿态 / 增益 | `assets/imgo2.py` 默认 `0.87 / −1.82`、`DCMotorCfg 25 / 0.5` | `default_dof_pos 0/0.87/−1.82`、`rl_kp 25 / rl_kd 0.5` | **一致** |

另外在本机复核了训练侧这次提交自带的测试（`python3 -m unittest discover -s tests -p test_amp_alignment.py`）：
**6 项通过 + 1 项跳过**（跳过的是需要 torch 的 CPU 回归），其中新增的
`test_amp_task_terms_match_reference_per_step_scale` 通过——即"每步系数 = weight × step_dt"这条契约
在**没有 Isaac Lab 的机器上**也能独立复现。

训练机上正式导出（导出后替换 `imgo2_deploy/policy/imgo2/amp/policy.pt`，并按 9.3 的三条契约复核）：

```bash
cd <工作区根>/imgo2_rl
python scripts/rl_lab/amp/play.py --task Imgo2-basemove-flat-amp-play --headless --num_envs 1 \
    --load_run <run 目录名> --checkpoint model_5000.pt
```

注意 `Imgo2AmpMovePlayEnvCfg` 会把命令固定成 `lin_vel_x = 1.0`、`y = 0`、`yaw = 0`
（便于回放对比），所以 `play.py` 里看到的步态是 1.0 m/s 下的；要与部署侧四项指标对照，
把同一个 checkpoint 放到 Gazebo 里用 `--key 3 --vx 1.0` 再跑一遍即可。
