# MuJoCo sim2sim 启动核对（2026-10-08）

初次核对回答如何跑通现有部署入口，未修改代码、下载运行库或重新训练。随后针对用户“按键没反应”反馈尝试一次有超时限制的 PTY 启动，结果见末节。操作步骤统一放在根 README §6.1。

## 已修复与已确认

- 现象：README 写“窗口内按键”，但控制输入实际从启动终端读取。
- 根因：`library/core/rl_sdk/rl_sdk.cpp` 的 `KeyboardInterface()` 调用 `kbhit()`，Linux 实现使用 `read(STDIN_FILENO, ...)`；MuJoCo GUI 自身也有快捷键，不能混为部署控制输入。
- 修法：README 明确终端输入，并补充先 `0`、等待起身完成、再 `3`、两次 `W`、`Space` 停止的顺序。FSM 起身插值为 1 秒预起身加 2 秒起身，起身完成后才允许切换策略；`W` 每次增加 0.1 指令，AMP 的 commands_scale 为 1。
- `build.sh -mj` 会自行准备 inference_runtime 和 MuJoCo；CMake 的 `USE_CMAKE=ON` 使用非 ROS 模式。模型读取仓库唯一源 `imgo2_description/mjcf/scene.xml`，策略已随仓库提供，首次行走不需要 Isaac Lab。
- AMP 实际配置为 45 维观察、12 维动作、恒等 joint_mapping。现有权重 SHA256：`cba59d44e387834ea11983b7489f4fd0351ddba8e8eba3c41e6713e33b520c54`，与 README §5.4 相符；config 顶部仍有旧 model_5000 注释，不能据此判断实际权重来源。

## 本轮验证

- `git pull --ff-only`：已是最新（受限环境 `.git` 只读，放宽沙箱后完成）。
- `/usr/bin/python3 imgo2_rl/scripts/tools/check_model_sync.py`：通过，模型物理、网格、FK 与 URDF 覆盖一致。
- `/usr/bin/python3 imgo2_rl/scripts/tools/check_asset_paths.py`：通过。
- `/usr/bin/python3 imgo2_rl/scripts/tools/check_towing_mjcf.py`：静态契约通过。
- 源码复核按键映射、状态切换、命令步进及构建脚本；没有运行按键控制或加载策略。

## 待修复／待确认及完成条件

- 初次核对时工作区没有 `imgo2_deploy/library/` 与 `cmake_build/bin/rl_sim_mujoco`；后续排查时二者已存在，不能继续判断本机缺这些依赖。仍需在可用显示环境运行起身、AMP 前进与停止，才能记录本机动态通过；`pip install mujoco` 不能替代 CMake 预期的 C++ 运行库布局。
- HIM-Loco 是参考占位策略，关节映射风险继续由 DEPLOY-01／DEPLOY-06 跟踪，不用于首次验收。
- 拖曳场景可供底层 AMP 试跑，但上层 `towing/policy.pt` 是训练 checkpoint，缺导出路径、config 与 FSM。完整拖曳闭环仍需 TOW-05 的 recurrent actor 导出和部署接线，以及 5／10／15 kg 牵引停车验证。
- 本轮静态检查不证明当前主机已能编译、GUI 可用、libtorch 可加载或动态控制有效。此前的行为验证以 README 链接的历史记录为准。

## 后续：用户反馈按键无响应（INPUT-01）

- 用户报告按键没反应，尚未说明输入位于 GUI 或终端，也未提供状态切换日志。当前不能确定根因或标为已修复。
- 复核 `glfw_adapter.cc`：窗口键盘回调只调用 `OnKey()`，没有部署 `Control::SetKeyboard()` 的转发。用户若在 GUI 输入部署按键，就不会送到 FSM；此项是源码行为，尚不证明用户实际操作如此。
- 复核 `fsm_imgo2.hpp`：Passive 只接受 `0` 起身；起身完成后才接受 `3`。终端读入为非规范模式，不需要按回车；回车本身会切换仿真暂停。
- 排查时已发现 `cmake_build/bin/rl_sim_mujoco` 和运行库；`ldd` 所有依赖解析成功。这只证明动态库可解析，不证明策略能加载。
- 在 agent 的 `DISPLAY=:1`、PTY 中执行 `timeout --signal=INT --kill-after=3s 25s ./cmake_build/bin/rl_sim_mujoco imgo2 scene`，打印 MuJoCo 3.2.7 后报 `ERROR: could not initialize GLFW`，提示按 Enter 退出；发送 Enter 后退出码 1。没有启动到键盘线程和 `RL_Sim start`，不能复现用户已打开窗口后的按键现象。
- 已修复：维护文档更新本机依赖状态与终端输入说明；没有改控制代码。
- 缺什么才能完成：用户说明按键焦点与启动方式，给出 `0` 后是否出现 `Getting up`／FSM 状态切换，以及启动日志是否到 `RL_Sim start`。在用户可用 GUI 环境验证前台终端 `0 → 3 → W → Space`，才能关闭 INPUT-01。

## 相机跟随操作复核

- 现象：用户询问为什么 MuJoCo 操作时视角不跟随。入口只调用 `mjv_defaultCamera()`，没有设置 tracking camera 或绑定 base，因而默认自由相机。
- 已修复：README 补充手动跟踪操作，无代码修改。
- 源码依据：`simulate.cc` 的帮助表将 `Ctrl Right double-click` 对应到 `Tracking camera`；选择动态 body 后设置 `cam.type = mjCAMERA_TRACKING`、`trackbodyid = selbody` 和 UI `camera = 1`。普通右键双击仅更新 lookat；`Esc` 返回自由相机。
- 验证与限制：完成源码复核，未实测 GUI。需要在用户可用显示环境中 Ctrl＋右键双击机器人躯干并移动机器人，确认画面持续跟踪。此操作指南不表示 INPUT-01 的控制输入问题已解决。

## 倒地乱踢与物理重置（FALL-01）

- 现象：用户反馈机器人倒地后乱踢。
- 源码事实：`rl_sim_mujoco.cpp::RunModel()` 只按 `rl_init_done && simulation_running` 决定推理，`AttitudeProtect(..., 75, 75)` 被注释；`RLFSMStateAMPLocomotion::CheckChange()` 只有手动按键／手柄切换，没有倒地条件。因此倒地不会自动退出策略，仍输出关节目标并由 PD 执行。
- 训练侧 `base_move/amp_env_cfg.py` 保留 base 接触终止项。倒地接触机身时训练回合会终止，不能把行走策略视为倒地恢复策略；超出正常行走状态时动作可能失去有效性。此项解释风险，不是对当前翻倒原因的动态诊断。
- `RobotControl()` 的 `R` 仅调用 `mj_resetData()`／`mj_forward()`，没有退出 FSM、清零命令或重置策略缓存；若 locomotion 仍在，重置后会继续输出。
- 已修复：README 补充启动终端 `P → Space → R → 0 → 等待起身完成 → 3` 的人工恢复顺序。Passive 是 `kp=0, kd=8, tau=0` 的阻尼状态，并非彻底零力矩。未修改保护代码。
- 待修复与完成条件：在独立变更中接入倒地检测、退出策略及完整控制状态重置，并验证翻倒后停止策略动作、正常行走不误触发、重新进入策略不消费旧动作。当前 agent 的 GLFW 限制仍在，本轮仅源码复核，未实测该恢复顺序，也未定位为何先翻倒。

## 自主起身：建议方案，尚未实施

用户询问是否通过新奖励训练自主起身，本轮仅讨论设计，不视为授权启动训练，也没有修改训练任务。

- 已澄清：行走 AMP 配置的 base 接触终止会截断恢复过程；参考状态初始化和行走动作先验也不等于覆盖侧躺／仰躺。只修改奖励不能使这些状态可学习。
- 建议独立恢复 PPO，部署 FSM 为行走→确认倒地→恢复→连续稳定站立→行走。行为分开训练再协调切换有研究先例：[Lee 等，2019](https://arxiv.org/abs/1901.07517)；本项目的具体分阶段设计是建议，尚无验证结果。
- 起身任务初始状态从腹部着地、左右侧躺逐步扩展到仰躺与随机腿姿，确保不穿地、不超限；命令置零。允许恢复期间机身接触，保留超时、数值异常等终止。观察包含姿态／重力、角速度、关节位置速度、上一动作；部署输入来源与维数必须一致。
- 奖励建议按姿态改善→扶正后抬身→稳定支撑组织：连续直立度奖励，近直立时启用目标高度和站姿奖励，稳定站立保持一段时间后给成功奖励；力矩、动作变化、关节限位、冲击为约束。不能全程强罚机身接触或大幅动作，以免阻止翻身；不能只奖励瞬时高度，以免学成弹跳。
- 当前行走动作尺度和裁剪是否允许足够的翻身／撑地动作需按关节工作空间检查，扩大策略输出范围也必须保持真实关节限位、力矩限制与模型轴约定。
- 若仍用 AMP，需新增恢复参考动作，或在恢复阶段关闭／重新设计行走风格先验；独立 PPO 可减少对行走数据的依赖。
- 待决定／待补：是否建立独立恢复任务、成功阈值与保持时间、各类倒地采样比例、动作尺度及课程安排。先做训练前置和短冒烟，再另行授权长训练；按左右侧躺／仰躺分别评估成功率、恢复时长、峰值力矩与冲击，完成 MuJoCo sim2sim 和策略切换验证后才能写“已具备恢复能力”。
