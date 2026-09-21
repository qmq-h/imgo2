# rl_sim 启动期连不上 `/controller_manager`：排查与加固（2026-09-18）

对应问题表 **DEPLOY-09**。只改了 ROS 2 路径（`rl_sim.cpp`）与 `base.yaml`，未动模型、策略、launch。

## 1. 现象（用户实跑）

`ros2 launch imgo2_deploy gazebo.launch.py` 已经在跑（用户确认 Gazebo 正常起来了），另一终端
`ros2 run imgo2_deploy rl_sim` 的输出：

```
[INFO]    [FSMManager] FSM created for type: imgo2
[INFO]    Joint order for the ROS 2 controller follows the URDF: FL_hip_joint FL_thigh_joint ... RR_shank_joint
[WARN] [1789714718.445904331] [spawner_robot_joint_controller]: Failed getting a result from
       calling /controller_manager/list_controllers in 10.0. (Attempt 1 of 3.)
^C[INFO] [1789714722.630605479] [rclcpp]: signal_handler(signum=2)
terminate called after throwing an instance of 'rclcpp::exceptions::RCLError'
  what():  could not create publisher: rcl node's context is invalid, at ./src/rcl/node.c:428
[ros2run]: Aborted
```

另一次运行的日志里还有一条 `[RTPS_TRANSPORT_SHM Error] Failed init_port fastrtps_port7419:
open_and_lock_file failed`（见 §6）。

## 2. 定位过程

1. **不是 README §6.2 记录的 numpy 坑**。那个坑的报错是
   `ImportError: Error importing numpy: you should not try to import numpy from its source directory`
   加 `[ros2run]: Process exited with failure 1`；本次是 spawner **跑起来了但服务调用超时**。
   （本机复现过该坑仍在：只要 `PYTHONPATH` 里带 Isaac Lab 的 `pip_prebundle`，系统 Python 3.10 的
   `spawner` 就 import numpy 失败。）
2. **失败点在启动顺序**。`rl_sim.cpp` 构造函数顺序是：等 `param_node` → `ReadYaml` → 建 FSM
   （打印 `Entered passive mode`）→ `InitControl` → `StartJointController()`（先
   `OrderJointsByModelOrder()` 打印 `Joint order ...`，再 fork `ros2 run controller_manager spawner`
   并 `waitpid`）→ 建 publisher/subscriber → 起三个 LoopFunc → `RL_Sim start`。
   用户日志正好停在 `Joint order ...` 之后，即卡在 spawner。
3. **`/controller_manager` 不是 gzserver 一起步就有的**：它是 `gazebo_ros2_control` 插件在机器人
   实体 spawn 成功之后才创建的节点。Gazebo 还在加载/还没 spawn 完就起控制器，spawner 会
   重试 3 次×10 s 后非 0 退出；`StartJointController()` 看到非 0 就
   `throw std::runtime_error("Failed to start joint controller")`，于是 `terminate` → Aborted。
4. **两次 abort 是次生现象**：Ctrl+C 让 rclcpp 先把 context 置为失效，而
   `waitpid()` 被 `EINTR` 打断后原代码不重试、`status` 是**未初始化**的（`WIFEXITED` 判定属 UB），
   构造函数于是继续往下走到 `create_publisher`，在失效 context 上抛 `RCLError` → 线程外
   `std::terminate` → `Aborted`。
5. **对照实验（本机沙箱，无头）**：按仓库自己的 launch 跑
   `gzserver + gazebo_ros2_control`，`/controller_manager`、
   `/controller_manager/list_controllers` 都在，`joint_state_broadcaster` 为 active；
   `install/` 是 `--symlink-install`（`launch`/`urdf`/`config` 全是指向源文件的软链，不是陈旧副本），
   Gazebo 版 URDF 的 `<parameters>` 会被 launch 改写成真实路径。⇒ 仓库代码与 launch 无缺陷，
   失败来自运行现场（时序或 DDS/插件）。

## 3. 修法（`imgo2_deploy/src/imgo2_deploy/src/rl_sim.cpp`，只动 ROS 2 分支）

| 位置 | 改动 |
|---|---|
| `StartJointController()` | spawner 之前先等 `/controller_manager/get_parameters` 服务出现（`wait_for_service`，沿用 `param_client` 同样的用法），最多等 `base.yaml` 的 `controller_manager_timeout` 秒（默认 60 s）。等的时候打印一次 `Waiting for /controller_manager (max 60 s); is Gazebo running? ...`；超时打印 `/controller_manager did not show up within ... Check the Gazebo terminal for 'Loaded gazebo_ros2_control' and 'Successfully spawned entity ...'` 后退出。 |
| `StartJointController()` | `waitpid` 按 `EINTR` 重试；被打断时给出明确信息。spawner 非 0 退出时打印退出码和临时参数文件路径（失败时不再删该文件，便于复查）。 |
| `RobotControl()` / `SetCommand()` | `!rclcpp::ok()` 时直接返回，不在失效 context 上 publish / 发服务请求。 |
| `main()` | 用 `try/catch(std::exception)` 包住构造 + spin：启动期失败打印一行原因并 `return 1`，不再 uncaught exception → `terminate called` → Aborted。 |
| `policy/imgo2/base.yaml` | 新增 `controller_manager_timeout: 60.0`（ROS1/MuJoCo 路径不读）。 |

## 4. 验证（本机 `colcon build --merge-install --symlink-install --packages-select imgo2_deploy`，35 s 通过）

| 场景 | 结果 |
|---|---|
| 正向：先起 Gazebo（无头）再跑 `rl_sim` | 打印 `controller_manager is up; spawning robot_joint_controller` → `Configured and activated robot_joint_controller` → `RL_Sim start`；abort 标记 0 |
| 正向运行中 SIGINT（直接发给 `rl_sim` 进程） | `[Loop] Loop end` ×3 → `RL_Sim exit`；**abort/RCLError 标记 0**（修前实测会 `could not create publisher` → Aborted） |
| 负向：只起 `param_node`（无 Gazebo），等 8 s 后 SIGINT | `[WARNING] Waiting for /controller_manager (max 60 s); is Gazebo running? ...` → `[ERROR] Interrupted while waiting for /controller_manager` → `[ros2run] Process exited with failure 1`；abort 标记 0 |
| 负向：只起 `param_node`，等满超时 | 62 s 时 `[ERROR] /controller_manager did not show up within 60 s; ...` → exit code 1；abort 标记 0 |
| 离线检查 | `check_model_sync.py` / `check_asset_paths.py` / `check_amp_joint_order.py` 全 PASS（本次未动模型） |

## 5. 没修掉的部分（待用户现场确认）

本次改动**只把"含糊的 abort"变成"能指向 Gazebo 侧的可读报错"**，并没有证明用户现场
`/controller_manager` 为什么不在。仍需用户在 Gazebo 正在跑时执行：

```bash
echo "DOMAIN=$ROS_DOMAIN_ID RMW=$RMW_IMPLEMENTATION LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY"
ros2 node list | sort
ros2 service list | grep controller_manager
ros2 control list_controllers
```

- 若**没有** `/controller_manager`：Gazebo 侧插件没加载或实体没 spawn，需要 Gazebo 终端的
  `Loaded gazebo_ros2_control` / `Successfully spawned entity` / `Exception occured in the Load function` 附近日志。
- 若**有** `/controller_manager`：属 DDS 发现问题（两个终端的 domain id / RMW / SHM 残留），
  §6 的清理与 `ROS_DOMAIN_ID` 隔离可试。
- 若按顺序（等 Gazebo 就绪再起 `rl_sim`）就正常：属启动时序，本次的显式等待已覆盖。

**未验证**：真实 GUI 画面、真机链路、ROS 1 分支（本机没有 ROS 1，本次改动在 ROS 1 分支内没有代码变更）。

## 6. 顺带记录的 `RTPS_TRANSPORT_SHM` 告警

`Failed init_port fastrtps_port7419: open_and_lock_file failed` 是 Fast DDS 共享内存端口初始化失败
（`/dev/shm` 里有上一个进程留下的 `fastrtps_*` / `sem.fastrtps_*`，或另一个用户/进程占着同号端口）。
它通常只是让 SHM 传输退到 UDP，不是本次失败的直接原因，但会在同机发现异常时添乱。惯常处理：
停掉所有 ROS/Gazebo 进程后 `rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_*`，或给两端设同一个
非 0 `ROS_DOMAIN_ID` 绕开残留端口。

## 7. 第二轮：用户改报「卡死在 `[FSMManager] Registered type: imgo2`」

### 7.1 现象

修好上一轮之后，用户重跑得到的是**卡住**（不是 abort）：终端停在
`[INFO] [FSMManager] Registered type: imgo2` 之后没有下文。

### 7.2 卡在哪：`param_node` 等待，且这是没被改过的老代码

`[FSMManager] Registered type: imgo2` 是静态注册（`dlopen` 时）打印的，它之后的第一件事就是
`RL_Sim` 构造函数里等 `/param_node/get_parameters`（`rl_sim.cpp:24-33`）：`param_node` 由
`gazebo.launch.py` 起的 `demo_nodes_cpp` 的 `parameter_blackboard` 提供。等不到它 →
`wait_for_service` 一直返回 false → 卡住。

**git 证据（用户要求核对「以前的版本能跑」）**：

- 等 `param_node` 这段代码来自 `8dcb9d5`（整合为单一项目时），此后没有改动；
- 本轮未提交的改动全部在它**之后**：`git diff -U0 HEAD -- rl_sim.cpp` 的 hunk 头是
  `@@ -8,0 +9,2 @@`（include）与 `@@ -273,0 +276,41 @@` 起（`StartJointController` 及以后），
  没有一行落在 `param_node` 等待之前；
- 最新的 `dc21184`（2026-09-18 15:18）只改了 `README.md`、`docs/gazebo_ros2_bringup_2026-09-17.md`
  与 `amp/policy.pt`，与启动链路无关；
- `build.sh` 的 ROS 分支就是 `colcon build --merge-install --symlink-install`
  （`build.sh:119-131`），与手工 `colcon build ... --packages-select imgo2_deploy` 等价，
  所以本轮重建没有引入编译期差异；
- 也就是说：**这个卡点不存在代码回归**，是运行时 `param_node` 不可见。

`param_node` 不可见只有两种可能：① Gazebo 那条 launch 当时没在跑（或已经退出）；② 两个终端的
DDS 发现不一致。别忘了一个环境级可能：本机还存在参考项目的
`~/RL/sim2sim/Imgo2_deploy/install`，若那个终端 source 的是它，`ros2 run imgo2_deploy rl_sim`
跑的就是**参考项目那份二进制**（可用 `ros2 pkg prefix imgo2_deploy` 确认到底解析到哪一个）。

### 7.3 加固：把「无声卡死」变成一行带 DDS 环境的报错

`rl_sim.cpp` 的 `param_node` 等待改为：最多等 `kParamNodeTimeoutS = 60 s`（这里读不到
`base.yaml`，因为它下面才 `ReadYaml`，所以写常量），等的时候**只打印一次**可操作提示并附带当前
`ROS_DOMAIN_ID` / `RMW_IMPLEMENTATION` / `ROS_LOCALHOST_ONLY`；超时打印
`/param_node did not show up within 60 s; giving up. Start Gazebo first ... and check both terminals
share ROS_DOMAIN_ID/RMW_IMPLEMENTATION and can see each other (ros2 node list).` 后 `exit 1`。
原来那个分支里的 `return`（构造到一半返回）改成 `throw`，由 `main()` 的 `catch` 收尾。

### 7.4 验证（重建 32.7 s 通过）

- 无 Gazebo：立刻打印
  `Waiting for /param_node (max 60 s). It is started by gazebo.launch.py; is that launch still running?
  Current DDS: ROS_DOMAIN_ID=<unset> RMW_IMPLEMENTATION=<unset> ROS_LOCALHOST_ONLY=0`，
  61 s 时打印超时原因并 `exit 1`，abort 标记 0；
- 先起 Gazebo（无头）再跑：`Get param robot_name: imgo2` → `controller_manager is up` →
  控制器激活 → `RL_Sim start`；SIGINT 后 `[Loop] Loop end` → `RL_Sim exit`，abort 标记 0。

### 7.5 仍待用户确认

在跑 `rl_sim` 的那一刻，Gazebo 终端是否还活着；`ros2 pkg prefix imgo2_deploy` 指向哪个工作区；
以及新提示行里的 `ROS_DOMAIN_ID` / `RMW_IMPLEMENTATION` 与 Gazebo 终端里的是否一致。

## 8. 第三轮：用户的 Gazebo 是好的，坏的是**同机 DDS 的数据通路**

用户回执：`ros2 pkg prefix imgo2_deploy` → `/home/qmq/RL/imgo2/imgo2_deploy/install`（工作区正确）；
终端 A 的 `param_node` 明确打印 `Parameter blackboard node named '/param_node' ready`；
但终端 C 的 `ros2 node list` 与 `ros2 service list | grep param_node` **都是空的**；
终端 B 只有 `[INFO] [FSMManager] Registered type: imgo2` 一行。

本机沙箱与宿主**共享网络命名空间**（`bwrap` 只 `--unshare-pid`），所以可以直接探现场：

1. **Gazebo 侧没问题，发现通路也没问题**：新起的 rclpy 进程用默认配置能列出
   `/param_node/get_parameters` 等 6 个服务（`ROS_LOCALHOST_ONLY=1` 时列不到，与"两端配置不一致
   就互相看不见"一致）。
2. **但真实调用回不来**：用仓库自己的二进制、对着用户正在跑的 Gazebo 执行
   `ros2 run imgo2_deploy rl_sim`，实际得到的是
   `Creating ROS 2 node ...` → `ROS 2 node created; rmw=rmw_fastrtps_cpp` →
   `[ERROR] Failed to call param_node service` → `[ERROR] The file '.../policy//base.yaml' does not
   exist` → `[ERROR] [FSM] No FSM registered for robot:` → 空关节名单 →
   `Waiting for /controller_manager`。
   即：**服务能被发现（多播），请求却回不来（单播/共享内存）**——同机 DDS 两条通路只通了前一条。
   这与用户最早那条 `[RTPS_TRANSPORT_SHM Error] Failed init_port fastrtps_port7419:
   open_and_lock_file failed` 指向同一个东西：共享内存端口不可用。
3. 终端 C 的 `ros2 node list` 为空则是另一件事：该命令走 **ROS 2 daemon**，
   daemon 的图缓存过期时会看到空列表（`ros2 daemon stop` 后再试即可）；判断真实发现情况要用
   `scripts/check_ros2_dds.py`（直接建 rclpy 节点，不经 daemon）。

### 8.1 新增的三处加固（`rl_sim.cpp`，重建 35 s 通过）

| 位置 | 改动 |
|---|---|
| `RL_Sim` 构造函数开头 | 在 `rclcpp::Node` 创建前后各打印一行（`Creating ROS 2 node (rl_sim_node) ...` / `ROS 2 node created; rmw=...`），这样"卡在 DDS 参与者初始化"与"卡在等 param_node"能一眼分开（只有 `Registered type` 一行、这两行都不出现的报告就是前者）。 |
| `param_node` 取值 | 从"失败只打印一行就继续"改成**重试 3 次×5 s，仍失败即带 DDS 线索退出**：原文会带着空 `robot_name` 去读 `policy//base.yaml`，再报 FSM 未注册、空关节名单，把根因埋掉。 |
| `ReadYaml` 之后 | 增加 `num_of_dofs` 存在性检查，`base.yaml` 没读到时立即报出期望路径并退出。 |

### 8.2 新增两个工具

- `imgo2_deploy/scripts/check_ros2_dds.py`：不经 daemon，直接建 rclpy 节点，分开报告
  **发现**（节点/服务）与**数据**（真实 `GetParameters` 调用耗时与结果），并打印四个相关环境变量；
  退出码 0/1 可直接用于判断。默认查 `/param_node`，可传别的目标（如 `/controller_manager`）。
  输出带时间戳且 `flush=True`（重定向/GUI 终端下不会因为行缓冲看起来"卡住"），并且把
  `rclpy.init()` 与 `create_node()` 各自计时——这两行如果不出现，就是"连 DDS 参与者都建不起来"，
  正好对应 `rl_sim` 只打印 `Registered type` 的那种报告。参考值：`init 0.00s`、`create_node 0.05s`、
  有目标时立即出结果、没有目标时等满 10 s 后报 `discovery FAILED`。
- `imgo2_deploy/scripts/fastdds_udp_only.xml`：Fast DDS 兜底配置，**只走 UDPv4、禁用共享内存**。
  两端都 `export FASTRTPS_DEFAULT_PROFILES_FILE=<该文件>` 后可绕开 SHM 通路（Humble 默认
  `rmw_fastrtps_cpp` 有效）。

### 8.3 验证

- `check_ros2_dds.py` 在隔离 domain（`ROS_DOMAIN_ID=42`）下对自建 Gazebo 栈：
  默认配置与 UDP-only 配置均 `RESULT: OK`（发现 9 个节点、6 个服务，取参 `['imgo2','imgo2_gazebo']`，
  0.00 s 返回），XML 无解析错误。
- `rl_sim`：无 Gazebo → 60 s 后明确超时退出；对着用户正在跑的 Gazebo（发现得到、数据不通）→
  3 次 `service call timed out` 后一行线索化报错并 `exit 1`（abort 0）；隔离 domain 下先起 Gazebo →
  `Get param` → `controller_manager is up` → 控制器激活 → `RL_Sim start`，SIGINT 后
  `[Loop] Loop end` ×3 + `RL_Sim exit`（abort 0）。
- **未做**：user 现场「为什么数据通路不通」的最终归因（`ROS_LOCALHOST_ONLY` 是否两端一致、
  `/dev/shm` 是否有残留、两个终端是否在同一份 `/dev/shm` 视图里）需要用户按 §8.4 的步骤回执。

### 8.4 给用户的现场步骤

```bash
# 1) Gazebo 正在跑时，在另一个终端（与 rl_sim 同环境）直接测两条通路
python3 imgo2_deploy/scripts/check_ros2_dds.py
#    RESULT: OK          → DDS 没问题，看 rl_sim 的完整输出
#    RESULT: ... FAILED  → 按提示处理：两端 ROS_LOCALHOST_ONLY 必须一致；清 /dev/shm 残留；
#                          两个终端要用同一种方式启动（都原生或都同一个 GUI/沙箱，别一个原生一个沙箱）

# 2) ros2 node list 为空多半是 daemon 缓存过期
ros2 daemon stop && ros2 node list

# 3) 彻底清一次遗留进程与共享内存（顺序不能反）
pkill -f gzserver; pkill -f gzclient; pkill -f rl_sim; pkill -f parameter_blackboard; sleep 2
rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_*

# 4) 仍不通时的兜底：两端都设同一个 profile，然后重启两端
export FASTRTPS_DEFAULT_PROFILES_FILE=$PWD/imgo2_deploy/scripts/fastdds_udp_only.xml
```

## 9. 第四轮：定位到 Fast DDS 的**共享内存通路**（DDS 数据面），并给出可用绕法

用户回执 `strace` 抓不到（`ptrace_scope=1` + 跨终端），但 `ps` 与 `/dev/shm` 给出了关键证据：

- `ps -ef` 里有 **2026-09-18 14:58 起一直在跑的遗留 spawner**：
  `sh -c ros2 run controller_manager spawner robot_joint_controller -p /tmp/robot_joint_controller_params.yaml`
  （`spawner` 默认无限等 `controller_manager`，所以从第一次失败那次到现在都没退出）；
- `/dev/shm` 里堆了 **十几组** `fastrtps_<hostid>` / `..._el` 文件，mtime 覆盖 14:58、16:19、16:21、
  16:28、16:29、16:30、16:52——正是这些天所有启动过的参与者留下的。

### 9.1 A/B 实验（同一时刻、同一个目标，只差共享内存开关）

在用户 Gazebo（16:52 启动）正在运行时，用 `scripts/check_ros2_dds.py` 对比：

| 配置 | 发现 | 真实调用 |
|---|---|---|
| 默认（SHM+UDP） | 6 个 `/param_node/*` 服务全部列出 | **FAILED（8 s 内没回来）** |
| `FASTRTPS_DEFAULT_PROFILES_FILE=scripts/fastdds_udp_only.xml`（禁用 SHM，只走 UDP） | 同样全部列出 | **OK（0.00 s）`['imgo2','imgo2_gazebo']`** |

⇒ **UDP 通路一直是好的，坏的是 Fast DDS 的共享内存通路**。这与用户最早那条
`[RTPS_TRANSPORT_SHM Error] Failed init_port fastrtps_port7419: open_and_lock_file failed` 同源。
用真二进制验证：设了该 profile 后 `rl_sim` **顺利过了 `param_node`**
（`Get param robot_name: imgo2` / `Get param gazebo_model_name: imgo2_gazebo`），不再出现
`Failed to call param_node service` 那串次生错误。

### 9.2 顺带查出第二件事：用户当前这个 Gazebo 里**没有机器人**

对用户 16:52 起的会话做全量探测（UDP-only 下）：

- 节点：`gazebo`、`joy_node`、`param_node`、`robot_state_publisher`——**没有 `gazebo_ros2_control`、
  没有 `controller_manager`、没有 `imu_plugin`、没有 `p3d_base_controller`**；
- 服务：只有 `/spawn_entity`、`/delete_entity`（factory 插件在）；
- 话题：`/clock`、`/joint_states` 在，但 **`/odom`、`/imu` 不在**。

⇒ `imgo2_gazebo` 实体没有 spawn 成功（所以 `rl_sim` 等到超时也见不到 `/controller_manager`）。
而 `spawn_entity.py → /spawn_entity` 是**同一个 launch 内的两个宿主进程之间**的调用；它失败说明
共享内存通路的问题**不只影响跨沙箱调用，也影响宿主内部**（这正是判断"不是沙箱假象"的依据）。
另一种可能是 spawn 因模型/网格原因失败，需要终端 A 的 `Successfully spawned entity` /
`Exception occured in the Load function` 日志来区分。

### 9.3 结论与用户端操作

**结论**：这台机器上 Fast DDS 的共享内存通路已损坏（遗留进程 + `/dev/shm` 残留端口锁定），
于是所有跨进程服务调用（`param_node` 取参、`spawn_entity`、控制器 spawn）都会超时，而**发现**依旧
正常，所以现象看起来像"服务在、就是不动"。之前两轮加固只是把这种失败从"静默卡死/abort"变成
可读报错，真正让它跑起来需要绕开共享内存。

**操作（两个终端都要做，且顺序是先清后设）**：

```bash
# 清掉遗留进程与共享内存（顺序不能反）
pkill -f gzserver; pkill -f gzclient; pkill -f rl_sim; pkill -f parameter_blackboard
pkill -f "controller_manager spawner"        # ← 那个 14:58 起挂到现在的进程
sleep 2
rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_*

cd ~/RL/imgo2/imgo2_deploy
source /opt/ros/humble/setup.bash && source install/setup.bash
export FASTRTPS_DEFAULT_PROFILES_FILE=$PWD/scripts/fastdds_udp_only.xml   # ← 两个终端都设

# 终端 A：等到这两行都出现
ros2 launch imgo2_deploy gazebo.launch.py
#   Successfully spawned entity [imgo2_gazebo]
#   Configured and activated joint_state_broadcaster

# 终端 B
ros2 run imgo2_deploy rl_sim
```

**验证方式**：`python3 -u scripts/check_ros2_dds.py`——不设 profile 应报 `data FAILED`，设了应报
`RESULT: OK`；终端 A 应重新出现 `Successfully spawned entity` 与 `controller_manager`。

**状态**：SHM 损坏的**具体成因**（哪个进程/文件锁住了哪个端口）未最终归因；profile 是已验证的
绕法，不是根治。用户可以先用它恢复运行，再决定是否深挖（例如逐次 `rm /dev/shm/fastrtps_*` 后
不设 profile 复测，确认是残留文件还是某个常驻进程）。

## 10. 第五轮：遗留 spawner 的来源查清并修掉（回答"是不是有进程卡住导致的"）

用户确认按 §9.3 操作后可以运行，并问「感觉就是有进程卡住导致的？」。答案是**部分成立**，
而且其中一环确实是代码缺陷导致的进程泄漏，本轮把它修了。

### 10.1 遗留进程是怎么来的（机制）

旧版 `rl_sim` 在 `StartJointController()` 里 `fork()` 一个子进程跑
`sh -c "ros2 run controller_manager spawner robot_joint_controller -p <yaml>"`，然后**阻塞**
`waitpid()` 等它。这里有两处问题：

1. `spawner` 的 `--controller-manager-timeout` 默认是 **0 = 无限等**（源码
   `/opt/ros/humble/local/lib/python3.10/dist-packages/controller_manager/spawner.py:130`），
   所以只要 `controller_manager` 一直没出现，它**永远不会自己退出**；
2. `fork()` 出来的是**三层进程树**（`sh` → `python3 ros2 run` → `python3 spawner`）。旧代码在
   Ctrl+C 时抛出异常并 abort，**一个子进程都没收**；而且 `server` 三层的直接父进程只是 `sh`，
   就算只杀直接子进程，下面两层也会变成孤儿继续跑。

于是 14:58 那次失败留下了一个**一直重试到 16:5x 的常驻 spawner**；它自己又是一个 DDS 参与者，
不断新增 `/dev/shm/fastrtps_*`，和"共享内存端口被占/被锁"互相加重。

### 10.2 修法（`rl_sim.cpp`）

| 改动 | 作用 |
|---|---|
| 子进程 `setpgid(0, 0)`（父进程也补一次 `setpgid(pid, pid)`） | 让 spawner 自成进程组，父进程可以 `kill(-pgid, SIGTERM)` 整棵收掉；同时也避免终端 Ctrl+C 越过父进程直接打到子进程 |
| 阻塞 `waitpid` 改成 `WNOHANG` 轮询（50 ms 一次，带 60 s 上限） | 原来的阻塞等待在信号竞态下会一直等下去（**实测**：SIGINT 后 `rl_sim` 仍活着，直到 spawner 自己超时），轮询才能立刻响应 Ctrl+C；超上限则主动收掉，杜绝无限挂起 |
| spawner 命令行加 `--controller-manager-timeout 30` | 即使 `rl_sim` 被 `SIGKILL`（没法清理），spawner 也最多再活 30 s，不会永久常驻 |

### 10.3 验证（重建 31 s 通过）

复现手法：起 Gazebo（隔离 domain）→ 等 `controller_manager is up` 后 `kill -STOP gzserver`，让
spawner 卡在服务调用上 → 对 `rl_sim` 发 SIGINT。

- 修前：`rl_sim` **不退出**（阻塞在 `waitpid`），spawner 三层进程 3 个全部遗留；
- 修后：立即打印 `Terminated the joint controller spawner process group (pgid 238)` →
  `Interrupted while waiting for the joint controller spawner` → `exit 1`，
  **遗留 spawner = 0**；`ps` 也确认 `--controller-manager-timeout 30` 已带上；
- 正常路径复测：`Get param` → `controller_manager is up` → `Configured and activated
  robot_joint_controller` → `RL_Sim start`；SIGINT 后 `[Loop] Loop end` ×3 + `RL_Sim exit`，
  abort 标记 0、遗留 spawner 0。

### 10.4 给用户的结论

「有进程卡住」是**加重因素**，不是全部：真正让"服务在、调用回不来"的是共享内存通路（§9）；
而那个从 14:58 挂到 16:5x 的 spawner 是**旧代码泄漏出来的**，它又反过来占用 DDS/SHM 资源。
现在两边都堵上了：用 profile 绕开 SHM（§9.3），以及 `rl_sim` 被打断时按进程组收掉 spawner、
spawner 自身也有 30 s 上限。
**未做**：`ROS_LOCALHOST_ONLY` 两端不一致的具体情形、`/dev/shm` 里究竟是哪一组文件锁住端口，
均未最终归因；建议用户先稳定用 profile 跑，若想确认是否还需要它，可在**完全清理后不设 profile**
复测一次 `scripts/check_ros2_dds.py`。
