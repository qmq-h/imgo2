# CMoE MuJoCo sim2sim 接线（2026-09-24）

## 范围与训练契约

在 `imgo2_CMoE` 分支 rebase 到远端 `bac18d6` 后接入部署。该提交的 `Imgo2CMoERoughEnvCfg` 将 height scanner 改为 `size=(1.0,0.6)`、`resolution=0.1`、`offset.pos=(0.25,0,20)`；`GridPatternCfg` 的 xy 展平顺序产生 11×7=77 个高度点，覆盖 base 坐标 x∈[−0.25,0.75]、y∈[−0.3,0.3]。`CMoEVecEnvWrapper` 组成 10×45 本体历史（当前帧在前）加当前 77 点地形，总计 527 维。这个配置与旧的 187 点 checkpoint 不兼容；部署权重必须由 77 点运行对应的 `cmoe/play.py` 导出。

训练地形观测为 `base_z - hit_z - 0.5`，20 m 只用于射线起点；无命中经裁剪为 −1。训练 terrain term 在裁剪前加均匀噪声 ±0.1，本体项也加噪声：角速度 ±0.2、重力 ±0.05、关节角 ±0.01、关节速度 ±1.5。`-play` 配置关闭两组观测噪声。

## 已接代码

- FSM 键 4 进入 `RLFSMStateCMoELocomotion`，加载 `policy/imgo2/cmoe/config.yaml` 和 `policy.pt`；缺文件时报错并返回 passive。ROS/Gazebo/真机暂不提供同等地形扫描，状态只允许 MuJoCo。
- MuJoCo 每步按 11×7、yaw 对齐生成向下射线，只命中 `group=5` 的静态地形；平地 `scene.xml` 已标组 5。首次进入用当前 45 维本体观测填满 10 帧历史，之后每步滑动，追加当前 77 维地形；尺寸检查为 527。
- `observation_noise: false` 默认匹配 `-play`；改为 `true` 可按训练侧各项幅值加均匀噪声。未命中射线仍为 −1，噪声在裁剪前加入。
- `scene_cmoe_gap.xml` 是四条 0.12 m 横向沟壑的 MuJoCo 样例，平台与训练 `CMoETrackGapTerrainCfg` 中等难度对应，出生点坐标平移到 MuJoCo 的 x=0。它只覆盖一种地形，不代表完整随机地形课程。

## 验证状态与待完成项

本轮 `git diff --check`、tracked-ignore 检查及两个场景 XML 解析通过；没有本轮 77 维 checkpoint、可用的 C++/MuJoCo 构建环境，**未编译、未加载 libtorch、未运行 sim2sim**。模型同步与资源路径 Python 自检因本机 WindowsApps Python 别名不可运行而未执行。代码状态为“已接线，待验证”，不能记为 sim2sim 已完成。

训练机应先用 `scripts/rl_lab/cmoe/play.py` 对应 checkpoint 导出 `exported/policy.pt`，比较导出件和 checkpoint actor 在确定性 527 维输入上的输出，再由部署 libtorch 加载。之后在 Linux 构建 `cd imgo2_deploy && bash build.sh -mj`，运行 `rl_sim_mujoco imgo2 scene` 验证平地，再运行 `rl_sim_mujoco imgo2 scene_cmoe_gap` 验证缺测 −1、站立、跟速、跨沟成功率和跌倒原因；最后可分别用无噪与训练噪声模式复测。

**缺什么才能完成：** 77 维运行的 checkpoint/导出件、MuJoCo/libtorch 构建机和实际回放记录。当前无 `policy.pt`，按键 4 会安全退回 passive。
