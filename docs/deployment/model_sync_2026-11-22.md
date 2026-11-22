# 关节几何修订同步与服务器训练

新版模型包曾多套两层同名目录，导致 Isaac Lab 的默认 URDF 和 Gazebo 安装软链接失效。已把原包原样移回仓库根的 `imgo2_description/`，恢复 `package.xml -> package.ros2.xml`。用户的 `imgo2_description_old/` 备份保留。

本次几何变更来自 [`core.xacro`](../../imgo2_description/xacro/core.xacro) 和两份用户更新过的 real URDF。修改了四髋及四大腿的轴心位置、四髋和右侧足端的碰撞体位置，以及两个后髋的视觉偏移。关节名、转轴方向、限位、惯量和 STL 未改。对应几何已同步到 [MuJoCo XML](../../imgo2_description/mjcf/imgo2.xml)。旧 `imgo2.urdf` 与 `imgo2.gazebo.urdf` 仍是历史变体。

## 服务器上的 Isaac Lab PPO

如果服务器已有同一项目扩展和相同的网格，只需将 [新版 imgo2_real.urdf](../../imgo2_description/urdf/imgo2_real.urdf) 覆盖到服务器 checkout 的 `imgo2_description/urdf/imgo2_real.urdf`，然后重新启动训练。 [资产入口](../../imgo2_rl/source/imgo2_rl/imgo2_rl/assets/imgo2.py#L20)默认读取这一文件，`IMGO2_URDF_PATH` 可以覆盖它。URDF 使用相对路径 `../meshes/`，目录布局应保留。如果从该服务器继续生成模型文件，应同时同步新版 Xacro 源。

替换后从目标 checkout 根目录检查：

```bash
python3 imgo2_rl/scripts/tools/check_asset_paths.py
sha256sum imgo2_description/urdf/imgo2_real.urdf
```

本地新版 SHA256 为 `0d19ce0efcf37c65b17f984f53c10a74a473d4319f445a77b4999da3ce28aabf`。检查输出中的 `URDF in effect` 要指向更新的文件，网格应为 17 个引用、0 缺失。正常配置没有固定 `usd_dir`，重新创建环境时由 Isaac Lab 的 URDF 导入器生成 USD。已有进程中的物理模型需要重启。几何更新可继续用原训练入口，先跑短训练确认环境与 rollout，再启正式新跑。几何变更未改变策略接口或权重形状。

## 同步验证和待确认

- **Isaac Lab** 用原资产入口在 CPU 上无头导入并执行一步，原 DCMotor 绑定 12 轴。实际左前髋相对 base 坐标为 `[0.15910003,0.06200000,-0.00016999]m`，确认加载新版。导入和物理断言完成后，Kit 退出超时，本轮没有在服务器运行短训练。
- **Gazebo** 用当前 ROS2 overlay 的 launch 参数预检，确认生成的临时描述与新版 Gazebo URDF 一致，17 个网格引用和控制器配置均可解析。当前安装是软链接，重启 Gazebo 即可读到。未运行 Gazebo 世界或闭环，`gz sdf` 在本环境的网络接口枚举失败。现有 Gazebo URDF 的 IMU 仍为 200Hz，Xacro 中为 100Hz，盲目重新生成会改变 IMU 频率，本轮保留用户已生成的文件。
- **MuJoCo** 与新版 Gazebo URDF 的 12 个活动关节、17 个连接、33 个碰撞体和惯量核对通过。MuJoCo 3.2.7 对普通及两个拖曳场景均通过加载、定姿及接触检查。重启 MuJoCo 进程会读取新版 XML。
- **仍待确认** `check_model_sync.py` 的全仓门禁仍报告网格历史指纹和旧模型登记问题（MODEL-05），包含用户保留的备份目录。服务器的新版模型短训练、三端步态及实机横滚影响仍待验证。实机俯仰模式的生成运动学表已被 `--check` 报为过期，实机控制有独立的更新和验证流程。
