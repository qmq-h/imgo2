# MuJoCo sim2sim现状复审（2026-10-23）

> 后续更新：逐物理步PD和反馈延迟/IMU噪声已实现并完成编译、组件测试与GUI短流程；本文件保留当时排查。当前设置与限制见[实现记录](mujoco_pd_feedback_2026-10-23.md)。

> 更新：本文件保留修改前排查。现有MJCF已按用户授权同步Gazebo参数，8body差异已修；当前验证与未完成项以[同步记录](mjcf_gazebo_sync_2026-10-23.md)为准。

## 结论与范围

架构支持独立MuJoCo sim2sim：MJCF场景、C++入口、共享FSM/推理/观测与动作转换均存在。当前不能标为“本机已能跑通”或“与实机/Gazebo严格对齐”：本轮临时CMake配置实际因缺MuJoCo开发库失败，GLFW开发包亦未找到；模型和控制时序有下述差异。未下载安装依赖、未运行MuJoCo动力学或真实硬件，未修改模型、策略、配置、控制代码。本轮只补维护文档与静态核对结果。

## 已核对无明显错位的基础接口

- 场景入口imgo2_description/mjcf/scene.xml包含imgo2.xml，CMake的IMGO2_MODEL_DIR正确推导到仓库根模型源，POLICY_DIR指向ros2_ws/policy；10个网格引用均存在。
- 当前12个MJCF关节轴/限位与训练声明实际使用的imgo2_real.urdf一致，顺序FL/FR/RL/RR、每腿hip/thigh/shank；base.yaml与ppo/config.yaml映射均恒等。
- 传感器前12项位置、再12项速度、再12项执行器力；四元数adr36/dim4、gyro adr40/dim3、线速度adr43/dim3，与当前GetState硬编码位置一致。总46维。使用base上的零旋转imu site；gyro为局部坐标，wxyz与共享重力投影处理相容。新增/重排传感器会破坏硬编码读取，后续宜按名称解析地址。
- PPO是45维观测、12维动作；gyro*.25、重力投影、指令、相对关节位置、速度*.05、上一动作。动作裁剪±3后按[.125,.25,.25]逐腿缩放，再加[0,.87,-1.82]。PPO的raw-action裁剪行为与Gazebo PPO相容，不能把AMP的问题说成PPO已错。
- ros2_ws与real的PPO权重SHA256均为53a57f909b61d2a708d24d6c08a90e086409b6d4fce4fdd3e6f3e1849abce7cf。用DL环境Python/Torch2.5.1 CPU加载TorchScript，对1x45静站输入（gravity_z=-1，其余0）推理，返回有限1x12；这只验证Python加载与维数，不替代C++/MuJoCo闭环。
- PPO YAML两边当前均25/.5、固定保持60/2；MuJoCo使用ros2_ws的参考FSM，并不调用real的FSM/硬件SDK。

## 待修复／待确认

1. **模型8个body未同步当前训练模型（MODEL-MJCF-01）**。4腿HIP和THIGH的质量、COM及显式惯量不同；例HIP .66982842对当前.66kg、THIGH .96995719对当前1.044kg。基础关节轴/限位一致并不等于动力学模型一致。以确认的模型源同步，不修改轴符号或静默覆盖用户物理参数。结果见[machine-readable audit](mujoco_sim2sim_review_2026-10-23.json)。
2. **被动动力学额外参数**。每个关节damping=1、armature=.1。前者是额外被动阻尼，后者为附加转动惯量，不是policy的Kp/Kd。在无速度目标、未饱和且采样状态相同时，主动Kd=.5与被动阻尼1共同产生约-1.5*dq的阻尼项；离散计算、被动隐式积分与饱和使其不能简单当成统一Kd=1.5。它们会改变/抑制振荡，不能直接删除，需明确对照基线。
3. **物理/PD/策略时序不等价**。MJCF timestep=.005，即物理名义200Hz；控制LoopFunc名义200Hz，SetCommand显式计算tau_ff+Kp*(q_des-q)+Kd*(dq_des-dq)再保持ctrl；策略LoopFunc名义50Hz。没有在每个mj_step回调内重算PD。Gazebo现有内部控制反馈约1000Hz、物理2000Hz，不能仅用“策略50/外层200”就认定相同。物理与控制线程按墙钟分别运行，暂停/慢放/加速可能改变每个仿真步的指令刷新数；名义频率未在本轮实测。
4. **反馈延迟功能尚未覆盖MuJoCo**。base.yaml的gazebo_motor_feedback_delay_ms/gazebo_imu_feedback_delay_ms仅Gazebo使用；MuJoCo GetState直接取sensordata。不能在现有YAML改10ms就声称完成MuJoCo延迟实验。若后续实现，须分开策略观测延迟与内部PD反馈，避免把外层通信延迟错误施加到驱动内部编码器反馈。
5. **状态快照存在并发缺口**。RobotControl持sim->mtx更新robot_state，RunModel却不持同一状态互斥量，逐字段读同一个vector状态；不能保证同一时刻完整快照，存在C++数据竞争风险。model_mutex只保护Forward的模型访问，不保护上述robot_state。应先在短锁中复制状态，再锁外推理；输入/FSM初始化标志也需按同一并发设计复核。未运行线程检测，未据此认定它已造成抖动。
6. **队列与退出问题**。RunModel向三个队列发布，RLControl只消费q/dq而不消费tau队列；该路径持续增长。q/dq分别排队，存在跨代次/阻塞后旧目标积压风险，宜单个带代次的最新完整命令。LoopFunc start后detach、shutdown无法join，不能证明退出前线程结束，需回收线程；本轮未改公共类以免影响Gazebo。
7. **启动失败处理不足**。构造函数等待全局d的while(1)无超时/失败检查；XML加载失败时PhysicsThread可能已记录错误而d始终空，入口仍等待。当前入口总是创建GLFW窗口，没有现成headless参数；依赖补齐也不等于本机图形环境可运行。
8. **AMP另有旧契约差异（DEPLOY-MJCF-01）**。MuJoCo仍裁剪raw动作并将裁剪值作为last_action；Gazebo AMP现为缩放/加默认角后裁剪关节目标且保留raw last_action。PPO当前不因此判错；AMP切换前必须单独同步与验证。himloco等历史策略也不可默认套PPO关节映射。
9. **模型总检查仍有既有失败**。check_model_sync.py的核心/标准Gazebo关节与惯量/FK通过，但网格14文件指纹66c496556453不符登记10文件指纹，另4份real/real_wrong URDF未登记。不能把它说成全项目模型已同步，也不能未经用户决定删除模型或资源。

MuJoCo力矩限制还应区别：ComputeOutput限制的是诊断估计力矩，RLControl实际tau_ff置0；最终控制由SetCommand重算，MJCF motor ctrlrange±23.7约束。PPO YAML torque_limits±23.5不等于最终驱动力矩强制±23.5。关节硬限位也不等于目标角已裁剪。需与实机Motor层目标限幅和真实驱动力矩限制分开核对。

## 本轮验证与复现

1. `/usr/bin/python3 imgo2_rl/scripts/tools/check_mjcf_current_robot.py --json /tmp/imgo2_mjcf_audit_20261023.json`：实际退出1，8body不同（预期发现，不记通过）。附加标准库XML核对轴/限位、mesh与sensor布局，结果归档本目录JSON。
2. `/usr/bin/python3 imgo2_rl/scripts/tools/check_asset_paths.py`：通过。`check_model_sync.py`：实际退出1，原因如上。
3. 临时配置实际执行并停在“MuJoCo not found”，LibTorch2.3.0已找到，不是缺整个推理库：

```bash
cmake -S imgo2_deploy/ros2_ws/src/imgo2_deploy \
  -B /tmp/imgo2_mujoco_review_20261023 \
  -DUSE_CMAKE=ON -DUSE_MUJOCO=ON -DBUILD_REAL_IMGO2=OFF
```

CMake固定查ros2_ws/library/mujoco并链接libmujoco.so.3.2.7。pkg-config查glfw3失败；当前未找到其开发配置。未下载或构建可执行文件，未有本轮MuJoCo步态结果。
4. `/home/tommy0929/miniconda3/envs/DL/bin/python`，Torch2.5.1 CPU执行上述PPO形状/有限值测试通过。系统/usr/bin/python3没有torch/mujoco；环境差异不等于项目LibTorch缺失。

## 推荐后续顺序及运行位置

先确认模型基线并同步8body，明确被动参数与PD计算步长；修复快照/线程/命令队列，补齐MuJoCo3.2.7与GLFW；先固定目标与零指令PPO短测，再+.5/横移/转向，最后增加分离的观测延迟扫描和IMU对照。需要与Gazebo/real相同策略、PD、指令历史和日志时标；只观察画面不能鉴定电机或IMU根因。

准备完成后，仓库现有入口位置是ros2_ws（10-14文档中的旧目录仅是历史说明）：

```bash
cd /home/tommy0929/Documents/imgo2/imgo2_deploy/ros2_ws
bash build.sh -mj
./cmake_build/bin/rl_sim_mujoco imgo2 scene
```

上述是将来的构建/运行方式，不是本轮已通过命令；build.sh可能下载依赖并复用cmake_build，本轮没有运行它。GUI仿真Enter暂停/继续；终端0起身、完成后1 PPO，WS/AD/QE增减指令、Space清零、9趴下、P被动、R重置。参考FSM不使用真机“状态数字+回车”的行命令机制，起身也不是real的8秒五次曲线（当前参考首次1秒预动作+2秒插值）。键盘要在程序终端输入；暂停/GUI焦点操作须实测。

来源：[MuJoCo3.2.7 XML参数说明](https://mujoco.readthedocs.io/en/3.2.7/XMLreference.html)，本仓库rl_sim_mujoco.cpp、rl_sdk.cpp、fsm_imgo2.hpp、loop.hpp、MJCF和当前YAML。原10-14记录仍作为历史，当前结论以本次核对为准。

## 同日补充：用户现有MuJoCo2.3.7可用范围

用户提供/home/tommy0929/Documents/mujoco-2.3.7-linux-x86_64，实际开发库根在其下mujoco-2.3.7/，include/mujoco与lib/libmujoco.so.2.3.7存在。修正前述范围：本机并非完全没有MuJoCo，而是工程固定查找位置/版本缺失，且已有2.3.7与现有viewer源码不兼容。

已用该目录原版头文件/库编译/tmp/imgo2_mujoco237_check.cpp并加载仓库scene.xml成功：header237、runtime2.3.7，nq19/nv18/nu12/nsensordata46、dt=.005；mj_forward及20次无控制mj_step完成（0.1秒，base z=.44631）。这证明2.3.7能解析并推进当前场景，不是PPO站立/步态或物理对齐验证。

对工程现有simulate.cc使用2.3.7头文件执行C++17 syntax-only，实际失败：mjPRESERVE未定义、mjData::solver_nisland不存在（设置最多5个错误，非完整不兼容清单）。因此不能只换路径、改库名或把2.3.7软链冒充3.2.7来构建现有程序；头库版本检查也不能绕过API差异。建议保持2.3.7安装不动，为本工程使用匹配的3.2.7与GLFW；若坚持2.3.7，需要适配viewer或独立无界面入口，再重新构建/闭环验证。未改CMake/库/模型/控制，未安装依赖；仅临时独立烟雾验证。

## 同日补充：3.2.7安装与旧版本清理说明（尚未执行）

已核对官方3.2.7发布页和本仓库scripts/download_mujoco.sh：脚本固定下载3.2.7到imgo2_deploy/ros2_ws/library/mujoco，符合当前CMake固定路径/库名；无需全局LD_LIBRARY_PATH，也无需覆盖已有2.3.7安装。建议先安装依赖并下载，在独立cmake_build_mujoco中配置/仅构建rl_sim_mujoco，验证后再删除旧目录。命令为：

```bash
sudo apt update
sudo apt install build-essential cmake curl libglfw3-dev libgl1-mesa-dev
cd /home/tommy0929/Documents/imgo2/imgo2_deploy/ros2_ws
bash scripts/download_mujoco.sh
grep mjVERSION_HEADER library/mujoco/include/mujoco/mujoco.h
ls -l library/mujoco/lib/libmujoco.so.3.2.7
cmake -S src/imgo2_deploy -B cmake_build_mujoco -DUSE_CMAKE=ON -DUSE_MUJOCO=ON -DBUILD_REAL_IMGO2=OFF
cmake --build cmake_build_mujoco --target rl_sim_mujoco -j4
./cmake_build_mujoco/bin/rl_sim_mujoco imgo2 scene
```

以上是用户后续执行建议，不是已安装/编译通过；模型/线程/PD差异依然存在。旧2.3.7是解压目录，可在关闭相关程序、确认无自写文件后用rm -rI -- /home/tommy0929/Documents/mujoco-2.3.7-linux-x86_64清理（GNU rm保留确认）；删除不自动清除用户shell配置中的旧环境变量。先查看.bashrc/.profile/.zshrc中的mujoco/MUJOCO引用，若有再按实际条目处理；不要删除不相关配置。无自动安装、删除、修改shell环境或运行硬件。

## 同日更新：用户已完成安装构建并报告启动

现场只读确认cmake_build_mujoco/bin/rl_sim_mujoco存在，ldd找到工程libmujoco.so.3.2.7、系统GLFW和工程LibTorch，无缺库记录。用户报告程序可启动；前面“缺依赖/未生成入口”仅适用于安装前检查。该结果不等同本轮代为运行PPO或完成步态验证。操作与调用链见[学习说明](mujoco_code_and_usage_2026-10-23.md)，模型/并发/时序待修项保持。新构建目录未跟踪且未忽略，后续需处理忽略规则，本轮未改。
