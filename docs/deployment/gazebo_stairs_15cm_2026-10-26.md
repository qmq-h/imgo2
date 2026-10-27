# Gazebo 15cm八阶楼梯（2026-10-26）

> 历史版本：用户随后改用每阶8cm，当前启动入口及验证见 [8cm记录](gazebo_stairs_8cm_2026-10-26.md)。本文件保留15cm场景的原始验证。

## 已完成

按用户要求新增 `stairs_15cm.world`，通过现有wname参数选择。保留旧earth/stairs，默认仍为earth。原stairs为10阶，不能作为本次8阶的结果。

| 项目 | 数值 |
|---|---|
| 阶数/每阶净高 | 8 / 0.15m |
| 每阶踏面深/宽 | 0.30m / 2.0m |
| 第一阶迎面 | 世界x=1.5m，横向y=−1…+1m |
| 上升方向 | +X，即机器人默认朝向 |
| 第八阶顶面 | z=1.20m，x=3.60…3.90m |
| 顶部平台 | x=3.90…5.10m，长1.20m、宽2.0m、高1.20m |

机器人仍出生于x=0、y=0、z=.5m，前方有1.5m平地接近区。每个台阶是从地面填满的独立静态长方体，视图和碰撞尺寸相同，踏面之间无水平缺口，顶部平台连续；台阶摩擦mu/mu2=1，接触组合仍取决于机器人材料。未添加栏杆或下降楼梯，平台远端为落差，操作时应停在平台内。

ODE物理步长/求解参数与earth一致（.0005s、2000Hz），没有修改机器人模型、策略或PD增益。launch仅补充可选world的描述。文件：`imgo2_deploy/ros2_ws/src/imgo2_deploy/worlds/stairs_15cm.world`。

## 当前本机启动

本轮已刷新安装目录，新world已安装为指向源码的软链接，不需要用户再次编译。

终端A：

```bash
cd /home/tommy0929/Documents/imgo2/imgo2_deploy/ros2_ws
unset PYTHONHOME PYTHONPATH
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch imgo2_deploy gazebo.launch.py wname:=stairs_15cm
```

无界面时在launch命令后加 `gui:=false`。终端B进入同一目录，执行相同unset和两条source后：

```bash
ros2 run imgo2_deploy rl_sim
```

按0起身、1切入PPO，然后W增加前进指令接近楼梯。这里是ROS2仿真入口，不是real真机程序。启动地形不会自动使平地策略具备爬楼梯能力。

其它主机同步新增world后，需在已正常构建的ros2_ws执行 `bash build.sh`（或重新安装该包），使新文件进入share/worlds；老build/install若仍为迁移前路径，按迁移记录先备份重建。

## 验证与待确认

已验证：系统/usr/bin/python3解析8阶几何、逐阶高度/落地实心体、踏面连续、视图/碰撞一致、平台接续、物理设置继承与安装软链；Gazebo自带 `gz sdf -k` 输出Check complete。实际独立domain64/master11374无界面启动，通过Gazebo transport查询到楼梯模型全部8个台阶及top_landing；机器人spawn成功，rl_sim与关节控制器进入Passive。随后停止本次启动的全部进程。

待确认：本轮未注入策略按键/速度，未测试PPO/AMP实际登阶或是否能爬完，未验证GUI画面；需要受控前进试验，记录接触、base高度、姿态与完成阶数。当前完成的是地形与启动链路，不能记为爬楼成功。没有运行真机或训练。

持久结果：[几何与运行核对JSON](gazebo_stairs_15cm_2026-10-26.json)。临时过程日志 `/tmp/imgo2_stairs_startup_2026-10-26/`（含Gazebo查询原文）、安装日志 `/tmp/imgo2_stairs_install_2026-10-26.log`。
