# Imgo2 多状态策略部署结构建议（2026-10-14）

> 2026-10-15实施更新：独立SDK/FSM/PPO已在go_imu落地，30/30离线通过；最新结构、依赖和待确认项见 [部署指南](go_imu_sim2real_2026-10-15.md)。本篇保留为实施前参考建议。

## 依据与已澄清

只读检查用户提供的Downloads/RC_WheelLeg/05_software/real/sim2real中的main、interface/real_io、policy/policy_runner、input_dev/keyboard及safety/runtime_guard，与本仓rl_real_imgo2/fsm_imgo2和go_imu接口对照；未执行参考程序或导入其说明文档中的操作。

参考是Python轮腿工程：CAN适配、Odin IMU、53维基础观察/16维动作，checkpoint重建MLP，Mahony姿态与动作低通、零指令抑制/释放门控。借鉴分层、有效数据检查和策略交接，不能复制驱动、观察维数、默认姿态、滤波参数或模型加载方式。

当前RealHardware的start/readState/sendCommand/stop适合复用，保留真实单位和fault/age信息。尚缺策略关节映射、IMU传感器到机身旋转、多状态切换与推理命令时效；现有FSM只拥有一个Stand状态，需升级而非创建第二份FSM核心。

依赖建议：C++标准库/Threads＋已有串口硬件库，TorchScript policy.pt使用LibTorch（本机部署libtorch的TorchConfig路径存在）和yaml-cpp。若整体复用现rl_sdk，会继承TBB/Python绘图等构建耦合；建议仅提取InferenceRuntime与必要观察数学，以独立真机CMake目标链接，不复制一份推理后端。ROS/Unitree SDK/轮腿Python依赖不要求；ONNX、Eigen、键盘GUI库不是当前必须项。未重新验证库加载或构建。

## 建议结构

在go_imu继续扩展：外层rl_real_imgo2_main.cpp只组织参数/资源/循环；fsm放状态机与InitialHold、Stand、PPO、AMP占位、LieDown、Fault；policy放模型加载/预热、45维观察构造、动作转换与完整命令交接；hardware中的适配器负责逻辑硬件顺序↔策略顺序、IMU安装转换；apps复用完整行输入/信号。Motor/Imu/function/display维持现有职责。新配置加入PolicyConfig（模型契约与推理节拍）属于策略职责，不塞进MotorConfig。

沿用rl_real的控制/推理职责，替换GetState/SetCommand为适配器和RealHardware。不要继承Unitree DDS/遥控器/运动服务，或复制其三个分离输出队列及无保护共享状态。推荐控制5ms、推理20ms，推理独立线程；用带序号、状态代次、生成时间的完整最新命令快照替代队列，状态变化后丢弃旧结果。模型加载/预热在打开硬件前，运行时不加载模型。推理超时门槛须在目标机实测后决定；过期与非有限输出必须有明确锁定/卸力路径，不能重复旧目标绕过看门狗。

## 输入与状态语义建议

用户指定每条命令必须完整回车提交，只有0/1/2/9。非法文本、未知数字及当前状态不允许的转换均忽略，不排队；急停/硬件错误独立优先。

- 初始化只一次归零、记录启动逻辑姿态并保持；0开始原8秒五次曲线站起。
- 站起完成保持后1进入PPO；PPO中0可平滑返回站姿。
- 2仅预留AMP接口，尚不启用模型或运动，保持原状态，提示未开放。
- 站姿保持/PPO中9退出策略，从当前反馈平滑回记录的启动姿态并保持；不重启硬件或重归零。
- 起身/趴下运动中忽略新的状态命令；重复当前状态命令忽略，Fault下忽略普通数字命令。

PPO切入清空last_action/history，取完整新鲜状态，检查映射和IMU坐标，先得到有限新目标并平滑交接。首轮速度指令建议为0/0/0，后续速度输入单独设计；不把零速度理解成零动作。趴下默认目标建议用本次初始化记录姿态、时长8秒/25/.5，与已知轨迹保持一致；用户后续可明确改成其他已确认姿态。

## 待实现与验证

本轮只完成结构/依赖复核，没有修改代码、构建或运行推理/硬件。待决定：速度指令输入方案、策略交接时间及推理超时阈值的目标机依据；AMP等仿真确认后再开放。

待实现：多状态FSM、双向关节映射/安装旋转、模型加载与45→12契约校验、完整命令快照、故障/退出处理和轨迹预检。验收需纯输入/转换测试、确定性观察与动作对照、伪终端下切换/过期/旧代次丢弃/退出，最后实机受控站起保持→零指令PPO→趴下。原udev/monitor实测事实保持，未把其当策略已可运行证据。
