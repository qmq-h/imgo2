本文命令针对 **AMP Height**：`Imgo2-basemove-flat-amp-height`。本机已核对日期：2026-10-11。

**每个执行训练／导出／Isaac Sim 回放的终端，先运行以下准备命令：**

```bash
# 本机路径示例；服务器上改成实际仓库路径与虚拟环境解释器。
export IMGO2_REPO=/home/tommy0929/Documents/imgo2
export IMGO2_ISAACLAB_PYTHON=/home/tommy0929/miniconda3/envs/env_isaaclab/bin/python
export PYTHONPATH="$IMGO2_REPO/imgo2_rl/source/imgo2_rl:$IMGO2_REPO/imgo2_rl/scripts/rl_lab"
export IMGO2_RUN_NAME=2026-10-11_00-00-01_xiaoji_real1_air_time_finetune
export IMGO2_RUN_DIR="$IMGO2_REPO/imgo2_rl/logs/amp_rsl_rl/base_move_amp_height/$IMGO2_RUN_NAME"
cd "$IMGO2_REPO/imgo2_rl"
```

新数据默认目录现为 `imgo2_rl/datasets/imgo2_motion_new`；若曾设置 `IMGO2_AMP_MOTION_DIR`，需确认该变量没有仍指向旧数据。

本机的最新运行已位于 `logs/amp_rsl_rl/base_move_amp_height/` 下。切换到其他运行时，修改 `IMGO2_RUN_NAME` 并重新设置 `IMGO2_RUN_DIR`；切换 checkpoint 时修改命令中的 `model_25000.pt`。安装配置见 [Ubuntu 安装流程](docs/training/ubuntu_amp_install_2026-10-10.md)。

# 本机：从 checkpoint 导出到 ROS 2 仿真

完整顺序命令见 [策略导出与ROS2仿真](docs/xiaoji_cmd/策略导出与ROS2仿真.md)，采用当前存在的2026-10-14_00-20-45_motion_new_model_new/model_25000.pt作为路径示例：导出 → 备份/替换AMP policy → Gazebo real模型 → rl_sim → 按0/3。这是sim2sim；下面旧25000段落保留为其它checkpoint的模板。

# 导出 pt 文件为 policy

使用 **AMP 官方入口 `play.py`**，输入训练 checkpoint，生成部署用 TorchScript 和 ONNX：

```bash
bash scripts/run_isaaclab.sh scripts/rl_lab/amp/play.py   --task=Imgo2-basemove-flat-amp-height-play   --checkpoint="$IMGO2_RUN_DIR/model_25000.pt"   --num_envs=1 --device=cuda:0 --headless
```

导出完成后脚本继续回放；看到生成文件后可用 `Ctrl+C` 停止。

- 默认输出：`$IMGO2_RUN_DIR/exported/policy.pt` 与 `policy.onnx`。
- 同目录另一个 checkpoint 再导出会覆盖 `exported/`，因此要区分轮数时及时保存副本：

```bash
mkdir -p "$IMGO2_RUN_DIR/exported_25000"
cp "$IMGO2_RUN_DIR/exported/policy.pt" "$IMGO2_RUN_DIR/exported_25000/policy.pt"
cp "$IMGO2_RUN_DIR/exported/policy.onnx" "$IMGO2_RUN_DIR/exported_25000/policy.onnx"
```

将导出的策略安装到 AMP 部署入口：

```bash
cp "$IMGO2_RUN_DIR/exported_25000/policy.pt"   "$IMGO2_REPO/imgo2_deploy/ros2_ws/policy/imgo2/amp/policy.pt"
```

`model_25000.pt` 是训练 checkpoint，不能直接用它替换部署 `policy.pt`。当前默认部署已核对为2026-10-14_00-20-45_motion_new_model_new/model_25000.pt对应的正式导出；其它历史策略的备份路径以各自记录为准。更换策略后重启 `rl_sim`，无需重新编译。

## 目录调整后的首次 ROS 2 构建（2026-10-20）

仿真工作区已改为 `imgo2_deploy/ros2_ws`，真机工程在同级 `real`。移动前的build/install含旧绝对路径，不能直接source。首次执行下面步骤；旧产物只备份，不删除日志或运行库。以下路径是本机示例，其他电脑改成自己的仓库位置。

```bash
cd /home/tommy0929/Documents/imgo2/imgo2_deploy/ros2_ws
unset PYTHONHOME PYTHONPATH
source /opt/ros/humble/setup.bash

# 仅目录迁移后的首次重建需要；后续可以省略这一段。
backup_dir="build_before_layout_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"
for dir in build install cmake_build; do
  if [ -e "$dir" ]; then mv "$dir" "$backup_dir/"; fi
done

bash build.sh
source install/setup.bash
```

在新的干净终端构建，避免继承旧工作区overlay；Isaac/Conda环境与ROS运行环境分开。`build.sh`编译的是ROS仿真链路，**不会编译或启动real中的电机控制程序**。结构/路径离线验证见根README；Gazebo实际起身及策略运动仍需另测。

# Gazebo sim2sim 命令

以下两段分别在 **两个终端**执行。使用 ROS 2 Humble、对应的 real 模型。

**终端 A：启动 Gazebo。**

```bash
cd /home/tommy0929/Documents/imgo2/imgo2_deploy/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

export IMGO2_GAZEBO_URDF=/home/tommy0929/Documents/imgo2/imgo2_description/urdf/imgo2_real.gazebo.urdf
ros2 launch imgo2_deploy gazebo.launch.py gui:=true
```

无界面测试时，把最后一个参数改为 `gui:=false`。历史 GLX/OpenGL 问题尚未确认解决，无界面实测可运行不代表 GUI 已验证。

**终端 B：启动部署策略。**

```bash
cd /home/tommy0929/Documents/imgo2/imgo2_deploy/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run imgo2_deploy rl_sim
```

在终端 B 按 `0` 起身，再按 `3` 进入 AMP；`1` 是 PPO、`2` 是 HIM-Loco、`9` 下蹲、`P` 回 Passive。前进等操作由部署键盘／手柄输入控制。

**可选：第三个终端自动发送起身和 AMP 命令，测量 0.5 m/s 步态。** 在新启动、尚未手动进入策略的仿真中执行一次：

```bash
cd /home/tommy0929/Documents/imgo2/imgo2_deploy/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

python3 scripts/eval_gazebo_policy.py   --key 3 --vx 0.5 --duration 30 --skip 4 --sim-time   --json /tmp/imgo2_amp_vx05.json   --raw-json /tmp/imgo2_amp_vx05_raw.json
```

所有终端须使用相同的 `ROS_DOMAIN_ID` 与 `GAZEBO_MASTER_URI`（如自行设置）。测量输出包含身体速度、姿态、偏航和关节周期；不直接测量接触滞空。两次真实对比结果见 [25000 sim2sim 记录](docs/deployment/gazebo_amp_25000_2026-10-11.md)。

# 开始训练命令

本节为 **从头训练**。当前新数据实验已关闭滞空奖励（`feet_air_time=None`）。下方命令保留历史任务混合比例 `0.7` 的显式覆盖；若使用源码默认 `0.3`，删除 `agent.amp_task_reward_lerp=0.7`。不要再添加 `env.rewards.feet_air_time.weight=100.0`。

先做 4 环境、10 轮短训练，核对导入、CUDA、参考数据和奖励是否正常：

```bash
bash scripts/run_isaaclab.sh scripts/rl_lab/amp/train.py   --task=Imgo2-basemove-flat-amp-height   --num_envs=4 --max_iterations=10 --seed=1   --run_name=amp_height_smoke   --device=cuda:0 --headless   agent.amp_task_reward_lerp=0.7
```

短训练正常后，再按显存容量选择环境数量。下面示例使用 1024 环境、24000 轮：

```bash
bash scripts/run_isaaclab.sh scripts/rl_lab/amp/train.py   --task=Imgo2-basemove-flat-amp-height   --num_envs=1024 --max_iterations=24000 --seed=1   --run_name=amp_height_new   --device=cuda:0 --headless   agent.amp_task_reward_lerp=0.7
```

`num_envs` 需按实际显存调整；1024 是示例，不是已验证的本机容量上限。输出目录为 `logs/amp_rsl_rl/base_move_amp_height/<时间戳>_<run_name>/`，保存 checkpoint、TensorBoard 与 `params/env.yaml`／`agent.yaml`。

# 续训练命令

下面示例从当前已有的 **25000 轮 checkpoint** 继续 **1000 轮**，新建日志运行，显式使用普通奖励比例0.7；当前缺口奖励开启，旧feet_air_time关闭。新数据更换后，不建议直接恢复旧判别器与归一化统计；此段仅展示兼容配置的续训操作。它是操作模板，不表示当前配方已达到期望步态。

```bash
bash scripts/run_isaaclab.sh scripts/rl_lab/amp/train.py   --task=Imgo2-basemove-flat-amp-height   --experiment_name=base_move_amp_height   --resume --load_run="$IMGO2_RUN_NAME"   --checkpoint=model_25000.pt   --num_envs=1024 --max_iterations=1000 --seed=1   --run_name=air_time_continue_from_25000   --device=cuda:0 --headless   agent.amp_task_reward_lerp=0.7
```

- `--load_run` 是 experiment 目录下的运行文件夹名；本例不是绝对路径。
- 续训 `--max_iterations=1000` 表示**额外**1000轮，25000继续到26000，不能填26000来表示终点。
- train.py 从当前任务配置启动，并不自动恢复 checkpoint 同目录的所有环境／奖励参数；保存的 yaml 用于核对。修改奖励后应另起运行，检查新 yaml。
- checkpoint 观测、动作、网络结构和模型契约需一致；改判别器特征或归一化输入尺度不能当作普通兼容续训。
- 准备长训练前，用本命令的4环境／10轮版本确认恢复成功。本次文档补齐未运行新训练。

# Isaac Sim 看训练效果命令

使用同一个 AMP `play.py`，去掉 `--headless`，开启实时回放。以下固定前进0.5 m/s，便于与Gazebo测试命令对照：

```bash
bash scripts/run_isaaclab.sh scripts/rl_lab/amp/play.py   --task=Imgo2-basemove-flat-amp-height-play   --checkpoint="$IMGO2_RUN_DIR/model_25000.pt"   --num_envs=1 --device=cuda:0 --real-time   'env.commands.base_velocity.ranges.lin_vel_x=[0.5,0.5]'
```

play任务默认横向速度与偏航命令为0，关闭参考状态初始化和相关域随机化。省略最后一行覆盖时，当前默认前进速度为1.0 m/s。此入口也会重新导出到 `exported/`。

需要可用的本地图形显示；服务器无桌面时可加 `--headless` 做无界面回放。GUI尚未在本轮运行验证。奖励及速度配置位置见 [AMP 调参入口](docs/amp/amp_air_time_style_2026-10-10.md#5-amp-height-调参入口复核)。

# 服务器从安装 imgo2 到重新训练

现在统一阅读 [服务器训练.md](docs/xiaoji_cmd/服务器训练.md)，按“系统Python已装Sim/Lab”与“空白环境新建Conda”两种情况执行，项目固定在/root/Desktop/imgo2。当前AMP使用工程算法，不要求RSL-RL5；标准Lab2.3.2仍需配套外部包装依赖。旧服务器步骤保留作历史记录，不作为当前操作入口。
