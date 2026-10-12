# Imgo2 整体项目架构、入口与移植指南

源码核对日期：2026-10-12。根 README 是总入口；本文用于从整体理解项目，细节链接到各模块指南。本轮只更新文档，没有修改代码、运行构建或启动机器人。go_imu 保持冻结待实机测试。

## 1. 先看这一张图

```text
imgo2/                          一个 Git 项目，训练和部署都在这里
│
├─ imgo2_description/           机器人长什么样：模型、物理参数、网格
│         │                     模型唯一源，不往其他目录复制
│         ├─────────────────────→ Isaac Lab
│         └─────────────────────→ Gazebo / MuJoCo
│
├─ imgo2_dataset/               AMP 用的参考运动数据
│         │
├─ imgo2_rl/                    Python：环境 → 算法 → 训练 → 回放 → 导出
│         └─ checkpoint/model_*.pt
│                     └─ 对应 play.py 正式导出 → exported/policy.pt
│
├─ imgo2_deploy/                C++：读传感器 → 拼观察 → 推理 → 下发目标
│   ├─ policy/imgo2/            base.yaml + 各策略 config.yaml / policy.pt
│   ├─ src/imgo2_deploy/        FSM、推理核心、Gazebo/MuJoCo/宇树真机入口
│   ├─ src/robot_msgs/          ROS 关节状态／命令消息
│   ├─ src/robot_joint_controller/  Gazebo 控制器
│   ├─ go_imu/                 你的串口电机 + Yesense IMU + 整机硬件接口
│   ├─ go/                     原已成功站起代码，保留
│   └─ imu_sensor/             原 IMU 调试代码，保留
│
├─ docs/                       架构、故障与验证细节
├─ README.md                   总入口、当前状态、问题表和维护记录
└─ AGENTS.md                   工作区维护约定
```

**已经存在的关系：**模型→训练→导出→仿真部署；go_imu→串口电机／IMU。
**尚未接通的关系：**rl_real_imgo2→go_imu::hardware。现在真机入口仍是宇树 DDS 后端，不能直接控制你的机器人。

训练程序和部署程序是不同进程、不同环境。部署只需要导出策略与观察／动作契约，不需要把训练算法、判别器或 Isaac Lab 一起搬到机器人上。

## 2. 目录职责：什么内容到哪里找

| 目录／文件 | 用途 | 修改时注意 |
|---|---|---|
| imgo2_description/xacro/ | 物理模型源及 Gazebo 模块 | 修改源后重新生成 URDF，不手改生成物 |
| imgo2_description/urdf/ | 训练／Gazebo 使用的已生成模型 | 实际使用哪份模型必须明确；不同实验模型不自动等价 |
| imgo2_description/mjcf/ | MuJoCo 机器人和场景 | scene.xml 包含 imgo2.xml；拖曳场景另含 cart.xml |
| imgo2_description/meshes/ | 全仓唯一机器人网格 | 不复制进训练包或部署包 |
| imgo2_dataset/datasets/ | 参考动作数据 | 本地与服务器选择的数据源须记录，不只看文件夹名 |
| imgo2_rl/source/imgo2_rl/imgo2_rl/assets/ | Isaac Lab 机器人资产、执行器及初始状态 | 配合模型、部署默认姿态核对 |
| …/tasks/manager_based/locomotion/velocity/ | 底层运动任务：环境、奖励、观察和动作 | base_move 注册任务，mdp 提供计算函数 |
| …/tasks/manager_based/towing/ | 上层拖曳任务 | 与底层步态策略分开，不当普通12关节actor直接部署 |
| imgo2_rl/scripts/rl_lab/ | train/play、算法、网络、runner、包装器 | 修改算法找 rl_lab/algorithms；改训练参数找任务 agents |
| imgo2_rl/scripts/tools/ | 离线检查、数据与步态分析 | 不能把离线通过当真机验证 |
| imgo2_deploy/build.sh | ROS／普通 CMake／MuJoCo 构建入口 | 可下载依赖、处理软链及构建产物；不是启动机器人命令 |
| imgo2_deploy/policy/imgo2/base.yaml | 公共 dt、关节数、固定增益、默认姿态和映射 | 改动会影响多个入口，仿真映射不自动适用于串口硬件 |
| imgo2_deploy/policy/imgo2/{ppo,amp,himloco}/ | 策略配置和部署模型 | 每份模型必须匹配观察顺序、缩放、历史和动作定义 |
| imgo2_deploy/src/imgo2_deploy/src/ | 三个主要部署入口 | rl_sim、rl_sim_mujoco、rl_real_imgo2 分别构建运行 |
| …/include/ | 上述入口的类声明／后端成员 | 真机头文件仍引入 Unitree SDK |
| …/fsm_robot/fsm_imgo2.hpp | Imgo2 状态与按键切换 | 没有同名 cpp；状态机不负责串口协议 |
| …/library/core/rl_sdk/ | 公共机器人状态／命令、观察构造、策略输出、FSM 辅助 | rl_sdk 是项目自己的 C++ 核心，区别于训练侧 RSL-RL 与宇树 SDK |
| …/library/core/inference_runtime/ | 部署模型加载和推理封装 | 与外层 library/inference_runtime 的运行库依赖区分 |
| …/library/core/observation_buffer/ | 历史观察缓存 | 只对要求历史帧的策略使用 |
| …/library/core/{fsm,loop,vector_math}/ | 状态机调度、循环、向量／四元数数学 | 公共机制，避免写入具体电机接线 |
| …/library/thirdparty/robot_sdk/ | Unitree 等参考硬件 SDK | 自己的串口后端不应该继续依赖宇树 DDS |
| imgo2_deploy/library/ | 推理运行库／MuJoCo等外部依赖位置 | 是否已下载以文件和构建结果为准 |
| imgo2_deploy/config、launch、worlds/ | ROS 控制配置、启动和世界 | 不等同于 go_imu/config |
| imgo2_deploy/go_imu/ | 独立硬件工具，详情见使用手册 | 本轮保持不变；没有策略推理依赖 |

外层 imgo2_deploy 是**工作区**；内层 src/imgo2_deploy 才是 ROS 功能包。src/imgo2_description 为指向模型唯一源的软链。当前部署包 package.xml 指向 package.ros2.xml；仍保留 ROS1 描述及条件代码，不表示本机已验证 ROS1。

## 3. 主函数有哪些，各自做什么

| 主入口 | 可执行程序／用途 | main 和初始化负责什么 | 运行期间的主链路 |
|---|---|---|---|
| scripts/rl_lab/{ppo,amp,himloco}/train.py | Python 训练 | 解析任务／环境／runner 参数，启动 Isaac Lab、构建环境、加载或新建训练 | 环境采样→奖励与终止→rollout→算法更新→保存checkpoint |
| 同目录 play.py | Python 回放／正式导出 | 加载匹配任务和checkpoint，调用正式导出工具 | 策略推理→环境动作→回放；导出actor供C++使用 |
| src/rl_sim.cpp | rl_sim：Gazebo ROS部署 | 初始化ROS，创建RL_Sim节点、控制器与订阅发布，启动循环，spin | GetState→StateController→SetCommand；独立RunModel做推理 |
| src/rl_sim_mujoco.cpp | rl_sim_mujoco：MuJoCo部署 | 解析机器人／scene、加载模型、建立模拟器、循环和键盘／手柄 | MuJoCo反馈→共同FSM／观察／模型→MuJoCo控制量 |
| src/rl_real_imgo2.cpp | rl_real_imgo2：当前宇树参考后端 | networkInterface→DDS→LowCmd/LowState→运动服务释放→循环 | Unitree反馈→共同FSM／推理→Unitree LowCmd |
| go_imu/go_probe.cpp | go_probe：有限电机扫描 | 薄main→RunProbe：参数、扫描与自检 | 全零FOC请求→反馈检查→有限报告，结束 |
| go_imu/go_monitor.cpp | go_monitor：电机＋IMU监测 | 薄main→RunMonitor：四总线诊断、IMU线程和面板 | 取独立快照→单线程固定刷新；Ctrl+C汇总 |
| go_imu/imu_reader.cpp | imu_reader：独立IMU | 薄main→RunImuReader：设备参数、采集和显示 | 连续解析→RPY/gyro/accel/RX面板 |
| go_imu/pd_demo_main.cpp | go_pd_demo：保持 | 薄main→RunPdDemo：严格电机启动、归零、日志 | 不断提交当前位置保持目标，退出卸力 |
| go_imu/stand_main.cpp | go_stand：站起 | 薄main→RunStand：严格启动、启动位置保持、等待0 | 8秒五次平滑轨迹→站姿保持，退出卸力 |

C++ main 通常只是初始化和等待；真正循环在构造函数创建的线程、RobotControl/RunModel 或 apps 函数中。不要只读 main 就以为找到所有控制逻辑。

## 4. 部署时每一层传什么

```text
                         每约20ms（当前base dt=5ms, decimation=4）
反馈快照 ──────────────→ RunModel
   │                        ├─角速度、四元数、逻辑关节角／速度
   │                        ├─ComputeObservation：缩放、重力投影、观察拼接
   │                        ├─Forward：模型推理／可选历史
   │                        └─ComputeOutput：action → q目标／dq／计算力矩
   │                                  │
   ↓ 每约5ms                          ↓ 动作交接
RobotControl → GetState → StateController/FSM → SetCommand
                                               ↓
                   Gazebo消息 / MuJoCo ctrl / 当前Unitree DDS
```

FSM按键：P→Passive；0→起身；起身结束后1→PPO、2→HIM-Loco、3→AMP；9→下蹲。**按键2仍对应参考占位策略，不作为Imgo2已确认可用策略。**不同后端也有自己的输入限制，Gazebo键盘需要stdin终端，导航cmd_vel须开启navigation mode。

PPO/AMP当前配置的单帧actor为45维：角速度3、投影重力3、速度指令3、相对默认关节角12、关节速度12、上次动作12。ComputeObservation用四元数逆旋转世界重力，四元数4维与重力3维各有用途。其他策略不能只凭维数相同就复用顺序。

标准位置动作是 default_dof_pos + action_scale×action，之后通过配置PD控制。实际后端是否下发位置或力矩由SetCommand决定，不能因为ComputeOutput计算过力矩就认为真实电机力矩被夹住。

真机应接入的链路（**建议，尚未实现**）：

```text
RealHardware.start 一次完成机械启动姿态归零与IMU就绪
  └─readState → Imgo2适配：硬件索引→训练顺序，传感器坐标→机身坐标
       └─RobotState → FSM / ComputeObservation / 推理
            └─RobotCommand → 逆映射为硬件Targets → sendCommand
                   └─go_imu安全监督／四总线线程／串口电机
结束或故障 → 禁止新运动目标 → RealHardware.stop
```

## 5. 如何运行：先选链路，不要全部一起启动

以下命令均标明工作目录；用户按机器实际环境执行。本轮没有运行这些命令。

### 5.1 训练／回放（需要Isaac Lab和可用GPU环境）

从仓库根进入imgo2_rl。包装脚本默认解释器/opt/conda/envs/isaaclab/bin/python若不存在，设置IMGO2_ISAACLAB_PYTHON为本机实际安装的解释器；不能直接照抄别的机器路径。

```bash
cd imgo2_rl
bash scripts/run_isaaclab.sh --check
# 小规模前置训练；完成后才考虑长训练：
bash scripts/run_isaaclab.sh scripts/rl_lab/amp/train.py \
  --task=Imgo2-basemove-flat-amp-height --num_envs=4 \
  --max_iterations=10 --headless
```

回放与正式导出入口：

```bash
# IMGO2_CHECKPOINT由用户设置为实际存在、与任务配置匹配的checkpoint绝对路径
bash scripts/run_isaaclab.sh scripts/rl_lab/amp/play.py \
  --task=Imgo2-basemove-flat-amp-height-play \
  --checkpoint="$IMGO2_CHECKPOINT" --num_envs=1 --headless
```

产物为checkpoint同级exported/policy.pt及policy.onnx；核对与actor一致、C++可加载、观察／动作契约后才替换部署策略。不要用手写TorchScript代替play.py正式导出。其他算法用各自play.py和任务，不照搬AMP任务。详细环境、配置继承和长训练见[RL指南](rl_structure_guide_2026-10-10.md)，具体实验命令见根[command.md](../command.md)。

### 5.2 Gazebo sim2sim（ROS2 Humble链路）

两个终端都在imgo2_deploy工作目录；先安装相应ROS/Gazebo依赖，避免conda的Python路径干扰。

```bash
cd imgo2_deploy
unset PYTHONPATH PYTHONHOME
source /opt/ros/humble/setup.bash
bash build.sh
source install/setup.bash
# 终端A
ros2 launch imgo2_deploy gazebo.launch.py
# 终端B：同样source ROS及install环境
ros2 run imgo2_deploy rl_sim
```

Gazebo启动实体与控制器，rl_sim读状态并下发控制。先检查初始姿态及有效反馈，再0起身、选择已经核对过的策略；不能把关节变化当身体真实行走。默认／real实验模型选择见[Gazebo代码指南](sim2sim_gazebo_code_guide_2026-10-12.md)和根README §6.2；启动姿态问题见[排查](gazebo_startup_posture_2026-10-12.md)。

### 5.3 MuJoCo sim2sim（独立CMake）

```bash
cd imgo2_deploy
bash build.sh --mujoco
./cmake_build/bin/rl_sim_mujoco imgo2 scene
```

加载imgo2_description/mjcf/scene.xml。依赖下载和显示环境须可用。拖曳scene_tow_compliant／scene_tow_inextensible是另一条场景链路，上层拖曳策略不能据此认定已经接通。

### 5.4 你的硬件工具（无需ROS、策略或宇树SDK）

从仓库根执行，构建放/tmp以保持go_imu文件夹不变：

```bash
cmake -S imgo2_deploy/go_imu -B /tmp/imgo2_go_imu_user -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/imgo2_go_imu_user -j4
/tmp/imgo2_go_imu_user/imu_reader
# 退出reader，机械架空／托住后再运行联合监测：
/tmp/imgo2_go_imu_user/go_monitor
```

go_probe、go_pd_demo、go_stand位于同一构建目录。monitor/probe发送卸力命令，不能与站姿保持程序同时运行。demo/stand启动即可能主动保持；控制初始化必须处于规定机械启动姿态。全部参数、错误与操作次序见[硬件手册](go_imu_usage_2026-10-11.md)。

### 5.5 当前真机策略入口

`bash build.sh --cmake`是宇树参考真机目标的构建路径。缺Unitree SDK2时CMake可跳过rl_real_imgo2；整体构建成功不说明真机目标存在。即使生成了，它仍要求宇树网络接口／DDS，并非你的串口机器人。本指南不提供把它直接启动到Imgo2的命令。

## 6. 为移植保留什么，替换什么

| 层 | 可以复用 | 必须适配／验证 |
|---|---|---|
| 模型与训练 | 模型源、正式导出actor | 选定checkpoint和训练契约，核对实际机械参数 |
| 推理 | InferenceRuntime、观察数学、历史缓存 | 观察顺序／缩放、四元数约定、模型维数及非法输出处理 |
| FSM | 起身／保持／策略／下蹲的状态结构 | 沿用已确认的go_stand轨迹与增益，区别Passive阻尼和全零卸力，增加故障联动 |
| 状态／命令 | RobotState/RobotCommand形式 | 一个完整快照、完整命令，序号／时间／过期；统一硬件与策略映射 |
| 硬件 | go_imu::hardware | 实机验证、限幅、安装方向及接线确认 |
| 输入 | 键盘、可选速度指令 | 移除宇树遥控器；需要手柄时另定义独立输入后端 |
| 构建 | 现有独立CMake模式 | 新真机目标链接hardware，解除Unitree及Gazebo依赖 |
| ROS | 可选cmd_vel、状态与诊断发布 | 保持核心控制可独立运行，ROS回调不直接竞争串口 |

建议继续放在现有仓库和部署工作区，**独立真机可执行目标＋共用推理／FSM＋硬件库＋可选ROS适配**。无需现在复制出第二套项目。物理目录在ROS包里不等于程序必须依赖ROS；实现时以CMake依赖分离为准，尚未实现的目录拆分不能写成事实。

外部工程可直接链接现有硬件库：

```cmake
set(GO_IMU_BUILD_APPS OFF CACHE BOOL "" FORCE)
# 建议同时关闭测试，避免给消费者引入工具测试目标
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory(imgo2_deploy/go_imu)
target_link_libraries(your_target PRIVATE go_imu::hardware)
```

硬件对象采用start/readState/sendCommand/stop；不用复制Motor或Imu实现。具体公开接口、单位和线程约束见[重构记录](go_imu_refactor_2026-10-11.md)。安装旋转和policy_to_hardware目前只是保留元数据，仍需未来适配层实际应用。

## 7. 当前阻碍真机部署的问题与完成条件

根据前次源码复核：

- 宇树后端尚未替换，且没有首帧有效反馈和过期门控：需接RealHardware严格启动与故障锁定。
- 回调、控制、推理及模型切换共享状态／参数的同步不完整：需完整快照与受保护的切换过程，模型锁不能代替整个运行链路同步。
- 力矩队列写入但RLControl不消费，位置／速度分开排队且缺生成时间：需完整最新命令交接，丢弃旧动作，避免持续积压。
- 当前析构只停循环，缺明确末尾卸力；推理失败、退出和信号应统一走硬件停止。
- FSM的预起身{0,1.36,-2.65}、1＋2秒线性轨迹、60/2固定增益与已验证go_stand不同；需按自己的机械流程确认。
- 当前Passive为kd=8阻尼，不等同全零FOC卸力；需明确状态语义。
- 当前计算力矩裁剪不代表实际电机PD力矩受限；go_imu限幅配置仍为空：需确认硬件力矩限制和每关节边界。
- 仿真恒等joint_mapping不能证明硬件映射正确，原始IMU不等于机身坐标：需接线／安装实测和数学转换验证。

顺序建议：硬件工具实测→映射与安装标定→统一接口接入→无策略FSM起身保持→策略观察／动作离线对照→选定单策略并受控实机测试。每阶段记录结果，不因代码完成而跳过验证。

## 8. 阅读顺序与维护记录

第一次阅读：本文→硬件手册→fsm_imgo2.hpp→rl_sdk.cpp的观察／输出→rl_real_imgo2.cpp的GetState/SetCommand/RobotControl/RunModel。需要仿真时读Gazebo指南，需要训练时读RL指南。

已完成：整体目录分层、主入口、数据流、运行路径和移植边界的文档整理，依据当前源码及已有专项记录。
待修复／待确认：第7节全部真机接入条件，当前硬件实测；没有把建议架构当已完成实现。
本轮验证仅为路径／链接／入口静态核对和文档差异检查，没有构建、重新测试、下载依赖或启动硬件。此前17/17硬件离线测试属于历史证据，不等于本轮实机通过。
