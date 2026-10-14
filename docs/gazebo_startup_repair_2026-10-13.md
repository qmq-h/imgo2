# Gazebo启动失败：外层description引用与重复服务

日期：2026-10-13。状态：模型资源引用已恢复、启动与AMP短测通过；GUI画面和硬件GL仍待用户确认。模型唯一源保持仓库根 `imgo2_description/`。

## 现象与根因

用户将description整理到外层后反馈sim2sim不能启动。现场 `imgo2_deploy/src/imgo2_description` 不存在，git显示该已跟踪软链被删除；merged install下URDF、网格、config等软链指向这一缺失入口。外层目录本身存在且位置正确，需要恢复ROS工作区的引用，不需要复制模型。

用户18:58那次日志还显示gzserver退出255、/spawn_entity等待30s后不可用。Gazebo服务日志明确为 `bind: Address already in use`；18:06的旧gzserver PID91329占11345，18:58的重复launch PID92952及gzclient仍在。因此不能把该次服务退出全部归因于模型路径。

## 修复

恢复原有相对软链：

```bash
cd /home/tommy0929/Documents/imgo2
ln -s ../../imgo2_description imgo2_deploy/src/imgo2_description
```

此命令用于当时入口缺失的现场；链接已恢复，当前无需重复运行。实际模型仍是外层唯一源。安装资源软链恢复可读，本次无需重编译。未修改URDF、惯量、policy、训练奖励或PD。

确认/proc命令属于本项目后，向旧gzserver91329和重复launch92952发送SIGINT，释放默认端口；没有全局pkill。测试只管理自身创建的进程组并正常关闭，结束后11345/11369均无监听、无gzserver/gzclient/rl_sim残留。

## 已完成验证

- 解释器：ROS2 CLI `/usr/bin/python3`（系统Python3.10），工具用 `/usr/bin/python3`；清除训练PYTHONPATH/PYTHONHOME，source Humble与部署merged install。
- `colcon list`识别外层软链引用的description包；install/share/imgo2_description所有已安装软链检查，失效数0。
- 默认launch参数解析通过，默认URDF最终指向仓库根 `imgo2_description/urdf/imgo2_real.gazebo.urdf`。
- 独立domain59/master11369，软件GL、gui=false，Gazebo生成机器人、robot_joint_controller加载激活、rl_sim循环与键3 AMP闭环26s短测完成；稳定窗口z约0.2852m、vx约0.0047m/s。当前策略SHA为旧24000的21e7ae...，未替换。
- 默认模型路径与默认master11345，domain59、软件GL，默认gui=true启动18s：spawn成功、gazebo_ros2_control加载、joint_state_broadcaster active，gzclient未在启动期退出。结束时SIGINT后GUI由launch升级SIGTERM关闭，这是测试清理阶段，非启动故障。

证据：[指标与身份](gazebo_startup_repair_2026-10-13.json)、[默认启动日志](gazebo_startup_repair_default_2026-10-13.log)、[控制启动日志](gazebo_startup_repair_control_2026-10-13.log)。启动验证不替代步态、切换抖动或长期运行品质评估。

## 当前启动方式

终端1：

```bash
cd /home/tommy0929/Documents/imgo2/imgo2_deploy
unset PYTHONPATH PYTHONHOME
source /opt/ros/humble/setup.bash
source install/setup.bash
LIBGL_ALWAYS_SOFTWARE=1 ros2 launch imgo2_deploy gazebo.launch.py
```

终端2：

```bash
cd /home/tommy0929/Documents/imgo2/imgo2_deploy
unset PYTHONPATH PYTHONHOME
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run imgo2_deploy rl_sim
```

控制窗口0站起、3切AMP。每次仅保留一套同master的仿真；先Ctrl+C退出旧launch再重启。当前使用软件GL规避历史硬件GLX问题，GUI画面未通过屏幕观察确认。

## 状态与限制

已修复并验证：外层模型ROS引用、旧实例默认端口占用、默认模型/控制插件与AMP短测启动。

待确认：DEPLOY-05的实际GUI画面与硬件GL；DEPLOY-14/AMP-23的切换和持续静止抖动、AMP-22步态品质仍是独立待改进项。此次没有实现上一轮低速奖励方案，也未启动训练。
