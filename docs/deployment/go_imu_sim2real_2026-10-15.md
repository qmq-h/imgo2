# go_imu 独立 Sim2Real：结构、运行与验证

核对日期：2026-10-15。本轮按用户批准的计划增加独立部署入口；没有启动真实电机、IMU或训练，没有导出、替换策略。**30/30 离线测试通过，不等于真机 PPO 已验收。** 测试结论、策略身份与源码指纹见 [核对数据](go_imu_sim2real_checks_2026-10-15.json)。

当前离线入口为`fsm_check`，复用正式FSM/SDK、键盘切换及固定位置关节目标显示，见 [测试指南](fsm_check_2026-10-15.md)。旧ppo_check、imu_reader、go_probe已按用户要求删除；以下历史30项验证指最初部署落地版本，当前入口及回归结果见 [入口精简记录](go_imu_entry_cleanup_2026-10-15.md)。

当前状态实现已拆分，Passive改为趴下实际到位后卸力，站姿/趴下角在deployment内独立设置；以 [状态拆分记录](go_imu_state_split_2026-10-15.md) 为准。

## 1. 已完成与待确认

已完成：在现有串口硬件库上接入 45 维 PPO SDK、异步推理、状态机和键盘控制；固定姿态用 60/2，PPO 用 25/0.5；初次/Passive后站起、趴下默认8秒五次轨迹，PPO回站默认2秒；状态切换和故障卸力离线通过。原 `go_stand`、demo 的默认增益和流程保持原样。

待确认：12 个策略关节和实际总线 ID 的物理对应、四元数参考与倾斜时投影重力方向、真实推理/通信频率、站起/趴下裕量、增益交接和卸力效果。用户已确认 udev、monitor 和 IMU 轴一致，程序不额外旋转 IMU。未确认映射时 `mapping_confirmed: false`，按 1 不进入 PPO。**AMP 只留接口，按 2 不切换、不加载模型。**

本轮保留并核对了 17 个已有文件 SHA256：关节宏所在 `config/real_config.h/.cpp`、两处策略文件、参考 FSM、原 ROS 包真机 cpp/hpp 均无变化。开始时原 `go/` 与 `imu_sensor/` 已不存在，本轮未重建或删除。工作区还有用户此前的修改，不能把全部 Git 差异算成本轮改动。受当前权限限制，`git pull --ff-only` 无法写 `.git/FETCH_HEAD`，本轮未同步远端、未提交。

## 2. 目录和入口

以下为新增部署涉及的目录；硬件目录继续沿用，避免改动已有工具的包含路径。

```text
imgo2_deploy/real/
├── CMakeLists.txt
├── rl_real_imgo2.cpp           新的独立真机主入口
├── fsm_check.cpp               离线键盘FSM与PPO测试
├── stand_main.cpp             原 go_stand：0+回车站起
├── pd_demo_main.cpp            原 go_pd_demo
├── go_monitor.cpp             电机+IMU诊断面板
├── config/
│   ├── real_config.h/.cpp      原关节宏、MotorConfig、ImuConfig、HardwareConfig
│   └── deployment_config.h/.cpp + deployment.yaml
├── hardware/
│   ├── real_hardware.h/.cpp    电机+IMU生命周期、安全监督与卸力
│   └── robot_adapter.h/.cpp    策略顺序↔硬件顺序，只做排列
├── sdk/
│   ├── robot_types.h          完整观测快照、推理结果、时间戳
│   ├── policy_config.h/.cpp    验证并加载PPO YAML
│   ├── rl_sdk.h/.cpp           拼观测、裁剪动作、动作转位置
│   ├── policy_runner.h/.cpp    libtorch CPU加载/预热/推理
│   └── inference_worker.h/.cpp 独立推理线程与最新结果
├── fsm/
│   ├── fsm.h                  共用State与FSM
│   ├── deployment_states.h/.cpp 总调度及切换条件
│   ├── deployment_context.h   共享数据，无具体状态类
│   └── states/                 stand/passive/ppo/hold/fault独立文件，amp抽象接口
├── apps/
│   ├── runtime.h              原公共信号、参数与输入辅助
│   └── deployment_input.h     即时速度键、整行状态键、终端恢复
├── function/my_myth.h/.cpp     数学常量、五次曲线、投影重力
├── Motor/ serialport/ Imu/     原电机、共用串口与IMU实现
├── display/                   原诊断显示，新部署不依赖面板
├── policy/ppo/config.yaml + policy.pt  用户已有策略
└── tests/                     原21项+新增9项离线测试
```

原 `imgo2_deploy/ros2_ws/src/imgo2_deploy/src/rl_real_imgo2.cpp` 仍为参考的 Unitree/ROS 入口，本轮没替换它。**运行本篇的可执行文件必须由 `go_imu` 构建，不使用 `ros2 run` 启动参考入口。**

主函数直接显示顺序：解析 CLI → 加载部署/PPO 配置 → CPU 模型加载和 16 次预热 → 校验固定姿态/轨迹 → 配置信号与键盘 → 一次启动硬件 → 构造 FSM/推理线程 → 控制循环 → 请求卸力 → 回收推理线程 → 停止硬件 → 恢复终端。异常路径先请求卸力，再回收资源。模型与 YAML 不支持热切换，修改后需重新启动。

## 3. 依赖、构建与搬出使用

需要 Linux、C++17、CMake、线程库，以及启用 RL 时的 **LibTorch CPU C++ 接口和 yaml-cpp**。无需 ROS 2、Unitree SDK、训练用 Python rsl_rl、Isaac Lab 或参考项目的 rl_sdk 安装包。本工程的 `sdk/` 就是本机器人需要的部署 SDK。

默认 `GO_IMU_BUILD_RL=OFF`，原五个工具和硬件库不查找 Torch/yaml-cpp。系统需要 yaml-cpp 开发文件（Ubuntu 可安装 `libyaml-cpp-dev`）。LibTorch 应选与系统 yaml-cpp 相容的 **C++11 ABI** 版本；ABI=0 会在配置时明确拒绝。

从仓库根目录构建（本机已存在的依赖路径示例；搬到另一台机器时换成该机实际 LibTorch 路径）：

```bash
cmake -S imgo2_deploy/real -B /tmp/imgo2_rl_deploy \
  -DCMAKE_BUILD_TYPE=Release -DGO_IMU_BUILD_RL=ON \
  -DCMAKE_PREFIX_PATH="$PWD/imgo2_deploy/ros2_ws/library/inference_runtime/libtorch"
cmake --build /tmp/imgo2_rl_deploy -j2
/tmp/imgo2_rl_deploy/rl_real_imgo2 --check-policy
ctest --test-dir /tmp/imgo2_rl_deploy --output-on-failure
```

`--check-policy` **不打开任何设备**，验证 YAML、模型输出、预热和固定轨迹。构建后自动把本工程 `policy/` 与 `config/deployment.yaml` 复制到可执行文件旁边。模型不存在会在硬件启动前报错。当前 `go_imu/policy/ppo/policy.pt` 受原忽略规则影响，不能假定Git clone会带上；本轮未改忽略规则。新机器需把已核对的模型随工程搬运，或从仓库已跟踪的 `imgo2_deploy/ros2_ws/policy/imgo2/ppo/policy.pt` 复制并检查SHA256为 `53a57f909b61d2a708d24d6c08a90e086409b6d4fce4fdd3e6f3e1849abce7cf`，配置也须匹配。只修改资源时应确保同步到运行目录，可显式指定：

```bash
/tmp/imgo2_rl_deploy/rl_real_imgo2 --check-policy \
  --policy-dir "$PWD/imgo2_deploy/real/policy" \
  --config "$PWD/imgo2_deploy/real/config/deployment.yaml"
```

`--policy-dir` 指向**包含 ppo 子目录**的根目录；不是直接指向 `.pt`。默认路径从可执行文件位置推导，与当前工作目录无关。硬件运行命令为 `/tmp/imgo2_rl_deploy/rl_real_imgo2`，本轮未执行该命令连接真实设备。

可以把完整 `go_imu/` 搬出仓库，保留其 policy/config，再用相同 CMake 命令独立构建。其他 C++ 工程可：

```cmake
set(GO_IMU_BUILD_APPS OFF CACHE BOOL "" FORCE)
set(GO_IMU_BUILD_RL ON CACHE BOOL "" FORCE)
add_subdirectory(go_imu)
target_link_libraries(my_deployment PRIVATE go_imu::sdk)
```

只要硬件层则链接 `go_imu::hardware`，不启用 RL。SDK 的公开头文件不要求调用方包含 torch/script.h。

安装：`cmake --install /tmp/imgo2_rl_deploy --prefix /tmp/imgo2_rl_install`，入口位于 `bin/rl_real_imgo2`，资源在 `bin/policy/` 和 `bin/config/`。安装后可移动整个安装目录。**第三方共享库并不打包进安装目录**：安装 RPATH 保留配置时的外部库路径，并支持安装目录的 `lib/`；换机器仍需提供兼容的 LibTorch/yaml-cpp，必要时设置 `LD_LIBRARY_PATH` 为该机器的 LibTorch `lib/`。本轮已验证复制工程→仓库外构建→安装→移动安装目录→删除源码/build→从无关工作目录加载策略；这不代表脱离第三方运行时可运行。

## 4. 参数与单位

| 配置位置 | 负责什么 |
|---|---|
| `config/real_config.h/.cpp` | 关节宏、设备/ID/方向、输出侧启动角、位置/速度/力矩/增益安全上限、总线周期、看门狗、IMU和硬件新鲜度 |
| `policy/ppo/config.yaml` | 当前模型的45维观测顺序和缩放、动作裁剪、action_scale、默认站姿、fixed和rl增益 |
| `config/deployment.yaml` | 控制/推理周期、交接时间、站起/趴下时间、速度指令范围、稳定条件、物理关节映射和独立趴下目标 |

原关节宏不删除、不改值。硬件展开顺序为 FL、FR、BL、BR，各 ID0/1/2，均使用逻辑输出端 rad、rad/s、N·m。启动只进行一次原有编码器零位计算；**不能先用 go_stand 站起，再保持站姿重启本程序按趴下姿态归零**。应从配置对应的机械初始姿态启动本程序，再按 0 站起。

`policy_to_hardware[i]` 表示“策略第 i 个关节对应硬件展开索引”。这是唯一应用的物理重排；YAML 原 `joint_mapping` 会校验，但作为旧部署字段不再叠加一次。确认后填写真实排列并显式设置 `mapping_confirmed: true`。该标志不验证接线，也不自动推断。修改排列会重排策略关节及逐关节增益；独立stand_pose和crouch_pose均明确使用硬件顺序，不再重排。

当前配置显式列出了可独立修改的 `stand_pose` 和 `crouch_pose`。趴下初始值与原启动角一致：FL/BL 髋 +3.49°、FR/BR 髋 −3.49°，所有腿大腿67.43°、小腿−165.01°。这沿用用户确认的启动 offset，**没有再次计算/修改 offset**。如要修改，编辑 deployment 下各自12项 rad；上电归零仍使用原宏。

新入口固定姿态 `fixed_kp/fixed_kd=60/2`，PPO `rl_kp/rl_kd=25/0.5`。位置/增益交接默认0.5秒，进入 PPO 前保留原目标直到首个有效结果。返回固定姿态以测得位置作为新轨迹起点，增益平滑回到60/2。正常Passive到位后KP/KD/tau/dq清零，不再保持60/2；到位超时进入Fault。原工具的25/0.5默认值不跟着改变。

五次曲线：`q=q0+(q1-q0)(10u³−15u⁴+6u⁵)`，`u=t/T`，目标速度为其时间导数；起止速度和加速度为零，关节角度随时间缓起缓停。启动前检查端点、增益与规划峰值速度，越界轨迹拒绝，不通过裁剪改变计划。PPO 有限目标由 Motor 最终裁剪；NaN/Inf 拒绝并锁定。

策略 YAML 的 `torque_limits=23.5` 被解析校验；本轮位置策略发送 `tau_ff=0`，该字段**没有替换**用户 Motor 的23.7上限，也不等于限制电机内部 PD 总力矩。改变底层安全参数须单独确认。

## 5. 状态与键盘

| 当前状态 | 有效指令 | 结果 |
|---|---|---|
| InitialHold / Passive | `0`+回车 | 8秒站起→StandHold |
| StandHold | `1`+回车 | 映射已确认，且误差≤0.1rad、速度≤0.5rad/s持续0.5秒时进入PPO |
| StandHold / PPO | `9`+回车 | 8秒趴下→实际反馈连续到位→Passive卸力 |
| PPO | `0`+回车 | 从当前反馈规划回站姿 |
| 非Fault | `2`+回车 | AMP提示占位，保持当前状态 |
| Stand / LieDown | 其他状态请求 | 不打断轨迹 |
| Fault | 任何普通状态键 | 不恢复；需要退出、排查并回到正确初始机械姿态重新启动 |

状态只接受完整单字符行 `0/1/2/9`，`10`、其他文字等不处理。W/S 前后速度、A/D 横移、Q/E 转向立即生效，不需回车；每次0.1，默认范围分别±0.5m/s、±0.2m/s、±0.5rad/s。只在 PPO 中有效；松键不自动归零，**空格立即清零**，离开 PPO 也清零。方向键/功能键的转义序列不会产生速度命令。Ctrl+C 或 SIGTERM 停止并卸力；输入EOF/断开也停止。终端 RAII 恢复原有回显/行模式，输入不是 TTY 时不改终端设置。

本入口只打印操作、状态变化和必要错误，不写电机文件日志，不持续打印反馈。需要完整反馈使用现有 monitor，但不要同时占用同一串口。

## 6. 数据流和故障条件

控制线程默认5ms：读取安全快照→处理键盘→FSM判断→整份12关节目标发布→把完整观测快照交给推理线程。每条电机总线有自己的线程，内部继续逐电机请求应答；5ms是设定节拍，实际成功反馈频率需实测。

推理线程默认20ms：取最新输入→组装观测→推理→裁剪动作→转完整目标→发布最新结果。不积压输入队列；每次进入策略重置 last_action 和序号。generation隔离状态，旧状态正在执行的推理结果不会用于新状态。结果携带生成时间与输入快照时间，两者均检查100ms；控制线程不等待推理。

45维顺序：角速度3（×0.25）、投影重力3、速度指令3、`q-default`12（×1）、dq12（×0.05）、上次裁剪动作12；整体观测裁剪±100。输入四元数 wxyz（body→world）转换成 `R(q)^T[0,0,-1]` 三维重力，零/非有限/单位范数偏差超过0.01拒绝。IMU位置偏移不影响本策略使用的角速度和姿态；没有使用加速度估算机身速度，也没有添加线加速度补偿。主循环在固定状态也检查四元数和角速度有效性。

12维输出先逐项裁剪±3，再 `q_target=default_dof_pos+action_scale*action`，髋0.125、其他0.25；dq和前馈tau为0，增益25/0.5，然后一次重排成硬件顺序。方向、减速比和offset仅在已有Motor层处理，SDK不重复处理。

| 现象/错误 | 原因与处理 |
|---|---|
| CMake找不到Torch/yaml-cpp | RL打开但依赖路径或开发库未提供；只调硬件可保持RL关闭 |
| `deployment error` / policy shape/order/history错误 | YAML、模型或部署参数不符合当前45/12无历史的位置策略；打开硬件前退出 |
| `start failed` | 原电机启动健康检查失败、IMU打开失败，或2秒内没有含四元数+角速度的有效帧；停止已启动资源 |
| `PPO锁定` | mapping_confirmed=false；不进入策略 |
| `尚未稳定站立` | 位置/速度未满足稳定时间；保持站姿 |
| `轨迹拒绝` | 固定姿态、速度峰值或增益超出Motor配置；保持现状态 |
| `Fault: ...motor...` / `joint ... stale` | 电机故障、反馈过期默认100ms等；所有总线撤销活动目标、发送零控制量 |
| `Fault: ...IMU...` | IMU缺字段、断连、默认500ms过期或非有限/无效四元数；卸力并锁定 |
| `Fault: inference...` | 模型运行异常、无效观测/输出；卸力并锁定 |
| `policy result timeout/expired` | 首结果、最近结果或其输入快照达到100ms；卸力并锁定 |
| `libtorch_cpu.so`找不到 | 外部运行库未安装到可搜索路径；提供兼容LibTorch，配置RPATH或LD_LIBRARY_PATH |

Fault不自动恢复；结果正常或数据恢复不会重新使能。退出先请求卸力，**再等待推理线程退出**，避免慢模型阻塞卸力；不可强行取消正在执行的LibTorch调用，因此进程最终退出时间仍受该调用影响。串口物理断连时“发送零控制量”不代表断连电机一定收到了，实际断连卸力要依靠硬件侧保护验证。

接收时间戳是本机steady clock，不是设备测量时间。不同总线与IMU不是同步采样；程序返回原始有效性/年龄，不能把控制周期或显示刷新率当成真实反馈频率。

## 7. 离线验证依据

环境：Linux；CMake Release；现有C++11 ABI LibTorch路径 `imgo2_deploy/ros2_ws/library/inference_runtime/libtorch`；yaml-cpp系统开发库。测试脚本用 `/usr/bin/python3`；Python数值参照使用本机 `/home/tommy0929/miniconda3/envs/env_isaaclab/bin/python`（torch2.7.0+cu128，CPU推理）。路径是本机验证环境示例。

启用 `GO_IMU_TORCH_PYTHON` 指向有torch的解释器后，**30/30 CTest通过**（原21项+新增9项）：

- 原协议、方向/单位、轨迹、限幅、看门狗、通信、monitor宽/窄终端/重定向、无文件日志、站立FSM及硬件库导入回归通过。
- policy_core：45维缩放、投影重力、动作转换、映射往返、无效输入及完整行键盘解析。
- robot_fsm：60/2与25/0.5交接、入口锁、站起/趴下、AMP占位、旧结果与超时锁定、无效IMU拒绝。
- inference_worker：旧generation在途结果丢弃、重新进入重置、异常快照和重复退出。
- policy_preflight：现有模型加载/预热，硬件不打开。
- policy_reference：同一个已有TorchScript文件，64组固定seed17观测，C++与Python直接推理最大差 **3.820483398087049e-06**（要求<1e-5）；没有与checkpoint actor另做比较，也没有重新导出。
- terminal_input：即时速度、回车状态、Ctrl+C/SIGTERM/异常退出后原termios恢复，管道EOF识别。
- rl_deployment_pty：四个模拟电机总线+模拟IMU，完整主流程、入口锁、PPO、趴下、即时W/空格；IMU停流与200ms慢推理触发故障，全部总线线上的控制量清零，Fault不恢复。
- rl_invalid_preflight：错误观测维数、历史、非有限尺度、缺模型、错误映射在启动硬件前拒绝。
- rl_relocation：复制到仓库外编译/安装/移动并加载策略，修复首次安装后LibTorch搜索失败。

硬件-only构建单独以 `GO_IMU_BUILD_RL=OFF` 编译并运行 **21/21测试通过**。伪终端测试中的短轨迹/映射确认/模拟设备覆盖只存在测试配置或专门的 `_pty` 测试目标；生产入口不读取这些测试环境变量，不自动选择真实设备。

## 8. 下一步验收

先完成重构版本固定姿态实机检查，再确认策略索引→FL/FR/BL/BR和ID的对应，实测静止/倾斜四元数投影重力与角速度。确认默认初始姿态、60/2固定增益、8秒站起/趴下的真实裕量和退出卸力。最后才显式解除PPO映射锁，验证真实50Hz推理、各总线成功反馈频率、100ms超时是否有足够裕量，以及25/0.5运动控制效果。AMP需要独立导出且核对观测/动作契约，不能把其模型替换进当前PPO路径。
