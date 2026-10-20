# 按修改后的core生成real URDF

日期：2026-10-13。请求：用户修改core.xacro中的符号后，生成纯real与Gazebo real URDF。

## 改动与产物

用户源文件相对HEAD的变更是FL_THIGH、RL_THIGH惯量交叉项ixz从 `+0.000245651` 改为 `−0.000245651`，不是关节轴符号。本轮不再修改core源参数，仅由robot.xacro生成：

- `imgo2_description/urdf/imgo2_real.urdf`：17 link、16 joint；SHA256 `1d97de7806341b376df619c48c5ce463f72e16d87f47211bf74c7acfddfc30a2`。
- `imgo2_description/urdf/imgo2_real.gazebo.urdf`：18 link、17 joint，包含transmission/Gazebo/IMU；SHA256 `e8021e5f94108ccc2e441c605420f5dcb87d84e7aaaafe5db4484452aee1086f`。

core SHA256 `6d760f088835949359dcd5d09c6b2c9f4f43d7a3b7bc5da52ea3dc1837f64029`。用户原文件已命名为real_wrong两份，保留；未修改原有imgo2.urdf/imgo2.gazebo.urdf或MJCF。

生成命令（ROS2 Humble xacro，清除训练Python环境覆盖）：

```bash
cd /home/tommy0929/Documents/imgo2/imgo2_description/xacro
unset PYTHONPATH PYTHONHOME
source /opt/ros/humble/setup.bash
xacro robot.xacro -o ../urdf/imgo2_real.urdf
xacro robot.xacro transmission:=true gazebo:=true imu:=true -o ../urdf/imgo2_real.gazebo.urdf
```

实际先生成到/tmp，结构/惯量与相对HEAD变更核对后写入目标；期间用户重命名备份，故最终比较基线采用git HEAD内容，不依赖备份名称。

## 已完成验证

- `/opt/ros/humble/bin/check_urdf`：两份均成功解析，树结构有效。
- `/usr/bin/python3`独立物理核对：17个共有link惯量/碰撞一致，全部质量正、惯量正定且满足三角不等式；Gazebo只多base_imu。见 [物理结果](urdf_real_regeneration_physics_2026-10-13.json)。
- 两份相对HEAD只改变FL_THIGH/RL_THIGH惯量，所有joint结构/轴/限位不变，见 [差异核对](urdf_real_regeneration_checks_2026-10-13.json)。
- 当前motion_new 21段在两份real模型上的identity FK RMSE均为0，和HEAD基线完全相同，见 [FK结果](urdf_real_regeneration_fk_2026-10-13.json)。惯量变更不改变几何FK，不能由此推断动力学表现相同。
- `check_asset_paths.py`通过：默认real URDF存在，17个网格引用全部可读、21段motion存在；ROS部署install软链也仍指向这两份产物。
- 按约定执行 `check_model_sync.py`，全仓守门仍失败：14网格指纹与旧登记不符，real及用户保留real_wrong四份URDF未登记。该脚本主要比较旧登记imgo2两份，不能将其通过的旧模型项当成本轮real验证。

## 状态与限制

已修复/完成：两份real生成物恢复并同步用户的ixz符号修改，生成/解析、物理有效性、真实数据FK和资源路径离线验证通过。

待确认（MODEL-05）：旧imgo2生成物/MJCF与新core的惯量版本关系、模型族登记/备份URDF管理和网格指纹；本轮没有扩大到这些模型。新惯量下的Isaac/Gazebo实际动力学与服务器USD缓存未运行验证，旧模型上的历史回放不能直接作为新惯量运行结果。

Gazebo重新启动后读取新URDF；本机采用symlink install，无需仅为这次URDF内容修改重编译。当前运行中的世界不会自动更新。未运行训练或替换policy。
