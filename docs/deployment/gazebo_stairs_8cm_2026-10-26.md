# Gazebo楼梯降为8cm（2026-10-26）

用户将每阶高度从15cm改为8cm。当前入口为 `wname:=stairs_8cm`，world源文件在 `imgo2_deploy/ros2_ws/src/imgo2_deploy/worlds/stairs_8cm.world`。8阶总高.64m；踏面.30m、宽2m、第一阶迎面x=1.5m、沿+X上升，顶部平台长1.2m，均与上一版布局相同。每阶和平台碰撞及视觉同步降高，实体仍填充到地面。保留之前15cm场景作为已验证的历史版本；当前低楼梯使用8cm入口。

## 启动

本机安装目录已刷新，无需重新编译。按原流程加载ROS/ros2_ws环境后：

```bash
ros2 launch imgo2_deploy gazebo.launch.py wname:=stairs_8cm
```

无界面加 `gui:=false`；另一终端同样加载环境后运行 `ros2 run imgo2_deploy rl_sim`，0起身、1选PPO、W向前。完整环境加载步骤沿用 [上一版记录](gazebo_stairs_15cm_2026-10-26.md#当前本机启动)，其中world名改为stairs_8cm。

## 已验证和待确认

已完成：系统Python检查逐阶高度.08*i、底面到地面、视觉/碰撞同步与安装软链，Gazebo SDF检查通过。独立domain65/master11375实际无界面启动，机器人spawn成功，Gazebo transport能查询到8个台阶和平台；测试进程已停止。没有改机器人、策略、PD或物理步长。

待确认：尚未进入策略或尝试登阶，不能把加载成功记为爬楼成功；GUI画面和当前PPO是否能爬完仍需实际操作/测量。本轮没有重新验证控制器或启动硬件。机器可读依据见 [核对结果](gazebo_stairs_8cm_2026-10-26.json)。

## 同日后续策略实测

已随后实际运行当前部署PPO：Isaac/Gazebo均未完成首阶登阶，原先“尚未测试”仅描述地形添加时的范围。结果与数据统一见 [双引擎确认](ppo_stairs_confirmation_2026-10-26.md)；登阶能力待改善，地形加载通过不等于策略成功。
