# Gazebo 无头步频测量（2026-10-09）

## 目的与运行条件

在 Ubuntu 22.04 / ROS 2 Humble / Gazebo 11 下，无 GUI 测量部署侧 PPO 策略的稳定步频。用户的 `DISPLAY=:1` 无法创建 GLX 上下文，`gzclient` 无法显示；Gazebo 无头服务端可以运行。

本次复用机器上已存在的 Gazebo 服务端；`gzclient` 不在运行，且 `DISPLAY=:1` 的 GLX 初始化失败，因此按无 UI 仿真采样。初次重复启动时发现机器人实体已存在，且控制器处于 active、原 `rl_sim` 已退出；停用并卸载遗留控制器后，重新启动 `rl_sim` 成功。

## 测量方法

启动 `rl_sim` 后运行：

```bash
python3 imgo2_deploy/scripts/eval_gazebo_policy.py --vx 0.5 --duration 22
```

脚本自动发送 6 秒起身指令，进入键 1 PPO 策略并给出 0.5 m/s 前进命令；进入策略后跳过 3 秒稳定时间。以四条腿的大腿关节位置自相关估算周期，统计 `/odom` 与 `/joint_states`。

## 结果

- 单腿完整关节周期：**0.4270 s**；对应步频：**2.342 Hz**（约 2.34 个周期/秒）。
- 周期性分数：**0.962**；FL 与 FR 大腿相位差：**175.36°**，与对角小跑相符。
- 指令速度 0.5 m/s，测得平均速度 **0.469 m/s**（误差 6.19%）；基座平均高度 **0.289 m**，roll 范围 **2.06°**，pitch 范围 **1.37°**。
- 有效样本：`/odom` 1298 条，`/joint_states` 12987 条。

这里的 2.34 Hz 指**每条腿的完整摆动周期频率**，不是把四只脚的触地事件相加后的计数。该测量是一次 22 秒仿真运行，作为本策略在本工况下的观测值；它不代表对其他速度、策略或模型版本的验收。

## 限制与后续

- GUI 尚未验证成功：在 `DISPLAY=:1` 下 `glxinfo` 报 `X_GLXCreateNewContext` / `BadValue`，表明当前显示服务无法创建 GLX 上下文。无头 Gazebo 和 PPO 闭环已运行，但图形显示仍需在可用的本地图形桌面或提供 GLX 的远程桌面验证。
- 当前 `gazebo.launch.py` 使用 `imgo2_description/urdf/imgo2.gazebo.urdf`；本次结果不代表 `imgo2_description_real` 模型或 real AMP 策略。
- 脚本基于大腿角周期估频，未直接从足端接触信号计步；需跨速度、重复运行或按触地事件统计时，应增加对应数据采集。
