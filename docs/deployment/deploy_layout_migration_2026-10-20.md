# real / ros2_ws 目录迁移后的路径修复（2026-10-20）

用户已移动目录，本轮修复随之失效的路径、命令和忽略规则。没有再次改名/搬动源码，没有修改关节宏、控制参数数值、模型物理、策略权重或控制语义；没有运行电机/IMU，也没有启动Gazebo动态仿真。以下是本轮实际验证，不沿用历史测试数字。

## 1. 当前布局与入口

```text
imgo2_deploy/
├── real/                       原go_imu独立真机工程
│   ├── CMakeLists.txt
│   ├── Motor/ Imu/ serialport/ hardware/
│   ├── sdk/ fsm/ config/ policy/
│   ├── apps/ display/ function/ tests/
│   └── library/                本机运行依赖，忽略
└── ros2_ws/                    后续sim2sim工作区
    ├── build.sh
    ├── src/
    │   ├── imgo2_deploy/        ROS策略节点、launch、Gazebo/MuJoCo入口
    │   ├── robot_msgs/
    │   ├── robot_joint_controller/
    │   └── imgo2_description → ../../../imgo2_description
    ├── policy/imgo2/            仿真策略
    ├── scripts/                仿真检查与依赖下载
    └── library/                本机下载依赖，忽略
```

ROS包名、节点名、可执行文件名和真机CMake选项没有改。`ros2_ws/build.sh --cmake`仍是原ROS工程的非ROS参考后端构建选项，**不是real工程的构建命令**；两套同名`rl_real_imgo2`不可混淆。

## 2. 已修复内容

| 项目 | 根因与修复 |
|---|---|
| 模型软链 | 原相对层数在移动后落到部署子目录；改为回到仓库根模型唯一源，不复制模型 |
| ROS/MuJoCo模型根 | PROJECT_ROOT_DIR现在是ros2_ws；CMake模型目录及build.sh预检多退一层 |
| Python根目录 | ros2_ws/scripts下相关脚本的REPO由parents[2]改为parents[3]；同步policy/install/scripts路径和工作目录 |
| 外部分析与回归 | RL侧调用Gazebo脚本及读取部署policy的工具/测试更新到ros2_ws；保留原算法与判据 |
| 旧图表数据 | 绘图从元数据中按策略路径后缀找哈希，兼容搬迁前/后的记录键名 |
| real安装运行时 | 初轮28/29，安装搬迁测试报libtorch_cpu.so找不到；CMake会过滤源码树内运行库的隐式安装RPATH，现显式保留所选LibTorch/lib并优先查相邻lib。重测搬迁及最终29/29通过 |
| 旧构建缓存 | 发现缓存含旧仓库路径及小电脑/home/rcer路径；不编辑生成文件。build.sh检测移动缓存并提示备份后重建，不自动删除产物/日志 |
| Git忽略 | 删除迁移来的real/.gitignore，集中根规则；忽略两工程运行库、构建/安装/日志；放行real PPO及ros2_ws部署权重，保留源码library/core |
| 文档 | 根README、根command.md、docs/xiaoji_cmd仿真页、当前架构/真机手册及相关链接同步；AGENTS只更新资源路径说明，维护约定不变 |

原Git索引尚含旧路径的已跟踪权重，根规则暂保留旧权重例外，以便用户暂存迁移之前`git ls-files -i -c --exclude-standard`也为空。新位置的权重已确认不被忽略，包括现存policy_xiaoji.pt；没有替用户执行git add/commit/push。

real安装后仍需要所选外部LibTorch存在，或自行把兼容运行库打包到安装目录的lib；本轮没有将数百MB运行库加入Git，也不声称只搬可执行文件到完全没有依赖的新电脑即可运行。

## 3. 后续如何构建与运行

根[command.md](../../command.md)及[策略导出与ROS2仿真](../xiaoji_cmd/策略导出与ROS2仿真.md)已给出完整命令。关键是先备份旧缓存再重建：

```bash
cd /home/tommy0929/Documents/imgo2/imgo2_deploy/ros2_ws
unset PYTHONHOME PYTHONPATH
source /opt/ros/humble/setup.bash

# 在没有加载旧overlay的干净终端执行；仅迁移后的首次重建需要。
backup_dir="build_before_layout_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"
for dir in build install cmake_build; do
  if [ -e "$dir" ]; then mv "$dir" "$backup_dir/"; fi
done

bash build.sh
source install/setup.bash
ros2 launch imgo2_deploy gazebo.launch.py gui:=false
```

另开终端，进入同一ros2_ws，source ROS和新install，再`ros2 run imgo2_deploy rl_sim`；ROS仿真键盘仍为0起身、1 PPO、2 HIM-Loco、3 AMP，不能与real的状态键规则混用。本轮没有执行上述launch启动。

真机离线构建使用新的构建目录，避开用户保留的旧小电脑build：

```bash
# 仓库根；只编译/离线检测，不启动硬件。
cmake -S imgo2_deploy/real -B imgo2_deploy/real/build_layout \
  -DGO_IMU_BUILD_RL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build imgo2_deploy/real/build_layout -j2
ctest --test-dir imgo2_deploy/real/build_layout --output-on-failure
```

## 4. 实际验证与限制

本机`python3`/系统`/usr/bin/python3`、GCC11.4、ROS2 Humble、工程内LibTorch2.3。所有本轮构建在/tmp；旧build/install/log保留。没有为修路径下载依赖或启动后台硬件进程。

- 新增标准库检查`python3 imgo2_rl/scripts/tools/check_deploy_layout.py`：模型软链、资源、脚本根、命令页与Git忽略检查0失败；`git ls-files -i -c --exclude-standard`为空。
- real：临时目录完整构建通过；RPATH修复后`ctest` **29/29通过**，含外部硬件库消费、伪终端、FSM、策略预检、复制构建/安装搬迁。没有真实串口访问。
- ROS：colcon list正确发现四包。临时colcon build停滞在命令环境采集，没有进入包编译，已Ctrl+C中断，**不计colcon构建通过**。改用相同Humble依赖，在/tmp按robot_msgs→imgo2_description→robot_joint_controller→imgo2_deploy逐包CMake构建安装，全部通过并生成rl_sim。
- 使用新安装share路径构造`LaunchDescription`，10项action成功；未运行LaunchService，未启动任何节点。ROS日志设/tmp，避开受限环境不可写的~/.ros。
- Gazebo相关Python回归 **7/7通过**；四元数检查独立编译运行，500姿态最大误差`2.35202e-07`。Python源码AST、Shell脚本及两份命令页Bash块语法通过；旧缓存检测已用保留的旧缓存验证。
- `check_asset_paths.py`通过；既有ROS部署权重与迁移前HEAD对应文件逐字节一致；real权重仍53a57f…。
- `check_model_sync.py`仍失败：网格14文件/指纹66c496556453，不同于登记10文件/8dc5b5995a11；四份real/real_wrong URDF未登记。此为本轮前已存在的模型契约差异，不以移动目录为由改模型或绕过检查。

机器可读汇总：[deploy_layout_validation_2026-10-20.json](deploy_layout_validation_2026-10-20.json)。临时日志位置为`/tmp/imgo2_real_layout_*`、`/tmp/imgo2_layout_*`，不是长期归档依据；可复现检查脚本在仓库内。

## 5. 待确认

- 用户在正常终端备份旧输出后，以colcon/build.sh重建，并进行Gazebo实际起身与策略运动、GUI验证；本轮仅离线。
- 确认real/real_wrong模型与新增网格的登记、来源和物理契约；不删除用户模型，也不把自检失败标通过。
- 1.65ms通信与落地颤抖沿用原问题，目录迁移没有解决它们。
- Git当前仍存在用户迁移造成的旧路径删除/新路径未跟踪，需要用户审阅并暂存提交；本轮未解决原分支分叉、未推送。

## 6. 2026-10-21：四个local_setup缺失的实际修复

现象：用户进入ros2_ws后source install/setup.bash，四个包的local_setup.bash均报not found。根因已实查：install/share内软链接仍指向迁移前的imgo2_deploy/build，四个CMakeCache的源码和安装前缀也都为旧位置。这是迁移后尚未原位重建，不能只改生成setup文件或只重新source。

### 已修复并验证

- 将旧build/install保留到 `/home/tommy0929/Documents/imgo2/imgo2_deploy/ros2_ws/build_before_rebuild_20261021_173237`；旧log、运行库、策略及模型未删除。备份路径属于本机恢复记录，不是运行依赖。
- 在新ros2_ws路径运行build.sh；正常本机进程环境中colcon实际完成四包（44.3秒）。受限环境首次尝试停在colcon安装后处理，已中断本轮进程再完成构建，不将首次停滞计为通过。
- 所有local_setup软链现指向ros2_ws/build，四个CMake缓存源码与安装前缀均为新路径。
- 干净终端环境依次source ROS与新install通过，无not found；四个ros2 pkg prefix正确，gazebo.launch.py --show-args通过，rl_sim动态库无缺失。受限环境launch检查使用/tmp的ROS_LOG_DIR，正常用户终端无需照搬。
- 短暂实跑独立domain63/master11373、gui=false：robot成功spawn，rl_sim及关节控制器启动到Passive，随后本次进程全部停止。没有注入起身或策略按键，没有运行电机/IMU。
- 构建仅警告参考Unitree SDK2缺失，跳过其rl_real_imgo2目标；本次需要的ROS rl_sim已成功生成。这与同级real工程无关。

### 当前启动（本机路径）

已经完成本次重建，用户不必再次备份/清理。新开终端：

```bash
cd /home/tommy0929/Documents/imgo2/imgo2_deploy/ros2_ws
unset PYTHONHOME PYTHONPATH
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch imgo2_deploy gazebo.launch.py gui:=false
```

另开终端进入同一工作区并加载相同两份setup，然后运行 `ros2 run imgo2_deploy rl_sim`。使用GUI时去掉gui:=false；本轮只验证无界面启动。

### 待确认与限制

目录迁移后的source/编译/无界面启动故障已修复并运行验证。完整起身、PPO/AMP运动与GUI尚未在本次新构建下复测；既有模型同步差异、航向/抖动问题未因重建而解决。需后续按需要做策略运动和GUI验证。没有修改构建脚本、控制参数、策略或物理模型。

机器可读依据：[重建结果](ros2_workspace_rebuild_2026-10-21.json)。本机过程日志 `/tmp/imgo2_ros2_rebuild_2026-10-21.log`、`/tmp/imgo2_ros2_source_check_2026-10-21.log`、`/tmp/imgo2_gazebo_startup_2026-10-21/` 是临时辅助；持久依据为本记录和JSON。

## 7. 2026-10-21：修改PPO的rl_kp/rl_kd如何生效

已核对：用户当前 `ros2_ws/policy/imgo2/ppo/config.yaml` 为12关节rl_kp=20、rl_kd=.5。实际新构建的POLICY_DIR指向ros2_ws/policy。PPO状态Enter每次调用InitRL，后者ReadYaml读取config.yaml；运行中的PPO不持续监视文件。

无需因这两项YAML数值修改重新编译。推荐保留Gazebo，Ctrl+C停止rl_sim，再运行 `ros2 run imgo2_deploy rl_sim`，按0起身、1进入PPO读取新配置；也可退出PPO再重新进入，Enter会重新读取。仅重启Gazebo不是重新读取策略配置的可靠操作，关键是rl_sim重新进入PPO。重复按1但状态未退出，不会重走Enter。

rl_kp/rl_kd用于PPO运动状态，起身/固定姿态还使用fixed_kp/fixed_kd，不能用起身阶段判定PPO增益是否变化。

本次只核对源码、实际编译定义和YAML，没有改用户数值、重新构建或启动仿真。已澄清加载方式；20/.5的真实下发与步态效果待用户重新进入PPO后确认，本次没有这组增益的运动验证。相关动态品质继续由DEPLOY-LAYOUT-01/PPO-RAMP-01跟踪。
