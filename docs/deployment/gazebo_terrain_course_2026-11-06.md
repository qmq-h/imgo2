# Gazebo多难度地形观察场（2026-11-06）

## 已完成与边界

新增可选世界terrain_course，含3条难度通道、每条4类地形。world源位于imgo2_deploy/ros2_ws/src/imgo2_deploy/worlds/terrain_course.world；现有Gazebo launch的wname说明新增此项。默认earth、原楼梯、机器人模型/PD/策略均未修改；ODE参数逐字段继承stairs_8cm。各障碍静态、摩擦mu/mu2=1，与旧楼梯一致；有效接触仍取决于机器人材料和求解器。

已修复/完成：原观察场地形不足，新增12块分级地形及本机安装软链。系统python3解析视觉/碰撞一致、正尺寸、重复生成SHA一致、原物理块一致；gz sdf -k通过，Gazebo11独立master11386无界面实跑并逐一gz model -i查询12模型/408碰撞通过，进程随后正常停止。查询早期误用gz model -l已修正，最终证据为逐模型-i，不以空列表判通过。

待确认：GUI相机实际观感、加载耗时/实时率、机器人ROS启动/策略通过各地形与动态切换；本轮未运行机器人或策略、未训练或实机。不能把场景加载通过写成策略爬坡/爬楼验收完成。证据见[gazebo_terrain_course_2026-11-06.json](gazebo_terrain_course_2026-11-06.json)。

## 布局

机器人仍在(0,0,.5)出生，朝+X。通道宽2.4m、中心y=0/5/10，之间保留空地可绕行/转向；世界地面仍是ground_plane。初级绿色、中级黄色、高级红色；Gazebo模型名含level和尺寸，便于左侧实体树定位。无自动课程，三档同时存在，用户手动选择。

| 类型 | X区间(m) | 初级 y=0 | 中级 y=5 | 高级 y=10 |
|---|---|---|---|---|
| 上/下楼梯 | 3–7.8 | 单阶2.5cm | 单阶5cm | 单阶8cm |
| 上/下坡 | 10–17 | 5° | 10° | 15° |
| 分散方块 | 20–23.6 | 高1.25–2.5cm | 高2.5–5cm | 高5–10cm |
| 随机起伏块面 | 27–30.6 | 高1–2cm | 高1–3cm | 高1–6cm |

楼梯6阶上升、1.2m平台、6阶下降；踏面.30m，总顶高15/30/48cm。坡道水平上坡长度3m、平台1m、下坡3m；顶高约26.2/52.9/80.4cm。通道侧边未设置防坠边界，转向/退出可通过空地绕行。

方块45cm见方，中心间隔60cm，间隙为地面；随机起伏由连续30cm方形小块拼接、每区12×8块，底部落地、高度固定种子42。它是带离散台沿的观察用近似，不是Isaac插值高度场，也不等同Isaac允许正负扰动的噪声定义；本场景所有块顶面高度非负。阶数/平台/坡道也不是Isaac金字塔地形逐顶点复刻。三档用于重复观察，不能当跨引擎严格同地形量化对照。

## 本机启动

在未启用IsaacLab Conda环境的ROS终端执行（原有conda环境需先退出）：

```bash
cd /home/tommy0929/Documents/imgo2/imgo2_deploy/ros2_ws
unset PYTHONHOME PYTHONPATH
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch imgo2_deploy gazebo.launch.py wname:=terrain_course
```

另一ROS终端执行相同cd、unset及两条source，再运行：

```bash
ros2 run imgo2_deploy rl_sim
```

沿用原键盘/FSM操作（0起身、1选PPO、W/S等调速度），先走初级。启动场地不自动改变策略，也不使平地策略获得障碍能力。无界面加载可加gui:=false。机器人后续FSM重置仍沿用原逻辑，不新增跨通道重置参数。

本机新world已安装为源码软链，无需重编译。其他机器同步后需按原已可用ROS工作区安装流程刷新worlds；已有build可在ros2_ws执行cmake --install build/imgo2_deploy，或原build.sh。先确认build目录与当前源码路径一致，勿复用其他机器的绝对路径缓存。

## 后续调整

只改生成器中的3档参数，再重新生成：

```bash
cd /home/tommy0929/Documents/imgo2
python3 imgo2_rl/scripts/tools/generate_gazebo_terrain_world.py
```

生成器只用标准库；默认输出到既有worlds目录，world生成物入库。重新生成后关闭旧Gazebo并重新启动才能载入新几何。难度参数在generate()的三档元组，布局/阶数在对应循环；当前固定seed42。改变参数后应重新做SDF及加载核对，旧JSON不能代表修改后的场景。
