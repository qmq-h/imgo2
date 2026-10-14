# AMP 策略导出与 ROS 2 Gazebo 仿真

本机路径示例，核对日期2026-10-14。用户称“sim2real”，但此处要求的是导出后启动ROS2仿真，实际为 **sim2sim**；本文不包含真机电机启动。

当前示例使用用户已复制到本机、路径已确认存在的 `2026-10-14_00-20-45_motion_new_model_new/model_25000.pt`。换策略时需同步修改checkpoint、导出目录和复制源；不同运行都会在各自checkpoint同级的exported目录产生文件。它不是2026-10-11滞空微调目录中的同名25000文件。

## 1. model_25000.pt 导出为 policy.pt

在终端A执行；如果已显示 `(env_isaaclab)`，不必重复激活。下面的变量设置都是独立的export命令，可以逐条执行：

```bash
cd /home/tommy0929/Documents/imgo2/imgo2_rl
conda activate env_isaaclab

export IMGO2_ISAACLAB_PYTHON="$CONDA_PREFIX/bin/python"
export PYTHONPATH="$PWD/source/imgo2_rl:$PWD/scripts/rl_lab${PYTHONPATH:+:$PYTHONPATH}"

bash scripts/run_isaaclab.sh scripts/rl_lab/amp/play.py \
  --task=Imgo2-basemove-flat-amp-height-play \
  --checkpoint=/home/tommy0929/Documents/imgo2/imgo2_rl/logs/amp_rsl_rl/base_move_amp_height/2026-10-14_00-20-45_motion_new_model_new/model_25000.pt \
  --num_envs=1 --device=cuda:0 --headless
```

使用本机已验证的Isaac环境解释器，以及当前项目两份源码。这是回放/导出，不是训练。生成：

```text
/home/tommy0929/Documents/imgo2/imgo2_rl/logs/amp_rsl_rl/base_move_amp_height/2026-10-14_00-20-45_motion_new_model_new/exported/policy.pt
/home/tommy0929/Documents/imgo2/imgo2_rl/logs/amp_rsl_rl/base_move_amp_height/2026-10-14_00-20-45_motion_new_model_new/exported/policy.onnx
```

确认加载的是本次25000文件且两个导出文件已生成后，按Ctrl+C停止回放，再执行下一步。另一checkpoint在同目录导出会覆盖exported，所以拷贝前确认本次导出已完成。headless不显示Isaac窗口；export只在当前终端生效，新终端需重新设置。

看到 `[INFO] Starting playback...` 表示JIT和ONNX导出调用已结束，程序进入持续回放循环。默认没有自动结束条件，也没有每步输出；headless下只停留在这一行是正常表现。只需要导出时，此时确认文件生成后按Ctrl+C即可。这个日志不单独证明后续每一步仿真都正常。

原写法把两个变量赋值和bash用反斜杠连成一条命令：完整执行时有效；若单独运行赋值再另起一条bash，未export的变量不会传给程序。现已改成上述分步写法。启动器仍需显式指定解释器，单独激活Conda不会覆盖其默认路径。

## 2. 备份并替换部署 AMP 策略

先停止已有rl_sim，再执行：

```bash
cd /home/tommy0929/Documents/imgo2/imgo2_rl

mkdir -p logs/amp_rsl_rl/base_move_amp_height/2026-10-14_00-20-45_motion_new_model_new/deployment_backup
cp ../imgo2_deploy/policy/imgo2/amp/policy.pt \
  "logs/amp_rsl_rl/base_move_amp_height/2026-10-14_00-20-45_motion_new_model_new/deployment_backup/policy_before_copy_$(date +%Y%m%d_%H%M%S).pt"

cp logs/amp_rsl_rl/base_move_amp_height/2026-10-14_00-20-45_motion_new_model_new/exported/policy.pt \
  ../imgo2_deploy/policy/imgo2/amp/policy.pt
```

model_25000.pt是训练checkpoint，导出的policy.pt才是部署TorchScript。本轮只修正文档，未导出或替换当前部署4000策略；成功执行以上cp后才会换成新示例25000。复制不会自动更改amp/config.yaml，需要与所导出checkpoint的观测、动作、默认姿态和PD对应。更换策略不需重新编译，已有rl_sim需重启；本步骤不会改变训练奖励或模型文件。

## 3. 终端A：启动 ROS 2 Gazebo

```bash
cd /home/tommy0929/Documents/imgo2/imgo2_deploy
unset PYTHONHOME PYTHONPATH
source /opt/ros/humble/setup.bash
source install/setup.bash

export IMGO2_GAZEBO_URDF=/home/tommy0929/Documents/imgo2/imgo2_description/urdf/imgo2_real.gazebo.urdf
ros2 launch imgo2_deploy gazebo.launch.py gui:=true
```

先等待机器人生成。gui:=false为无界面；本机历史GLX问题未确认解决，有界面失败时需按显示环境排查，不把策略导出成功当作GUI成功。unset仅清掉当前终端给ROS Python带来的路径覆盖，不删除文件；它只在结束Isaac回放后用于ROS终端。

## 4. 终端B：启动策略节点

新开终端执行：

```bash
cd /home/tommy0929/Documents/imgo2/imgo2_deploy
unset PYTHONHOME PYTHONPATH
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run imgo2_deploy rl_sim
```

保持终端B键盘焦点，按 **0起身 → 3进入AMP**。0对应起身，3是AMP，不要按1（PPO）或2（HIM-Loco）。这时加载imgo2_deploy/policy/imgo2/amp/policy.pt，即刚复制的导出文件。

两ROS终端若自行设置ROS_DOMAIN_ID或GAZEBO_MASTER_URI，需要值一致。若终端激活了Conda导致ROS Python/动态库问题，先退出Conda再运行上述ROS命令；Isaac导出与ROS仿真是两个运行环境。

## 核对与限制

已修复：环境变量设置改为独立export，避免拆开执行后子进程收不到变量；checkpoint、两个导出文件与备份/复制源统一改为本次motion_new_resume的25000。旧24000示例及“需重新提供新策略路径”的过时说明已更新。

验证范围：本机Python、新25000 checkpoint、real模型、ROS setup和launch路径均存在；5段Bash语法及git diff --check通过。提取第一段中的独立cd/export命令，经实际启动器执行轻量Python检查，解释器选中env_isaaclab且两份项目源码路径均传入sys.path。未重新运行Isaac导出/策略替换/Gazebo。先前4000策略导出契约已通过，其Gazebo效果待验证；新25000的实际导出、配置契约及Gazebo效果同样待确认（AMP-17/18）。真机sim2real需要另行确认整机硬件入口与观测/关节映射，不能把本页rl_sim命令当成真机启动。

用户随后实际运行至Starting playback（2026-10-13），本机只读确认该新25000运行的exported中两个文件已生成：policy.pt约752 KiB、policy.onnx约741 KiB，文件修改时间显示2026-10-13 11:40。已修复“等待回放是否等于导出卡住”的说明；生成产物已确认，尚未核对actor数值一致性/部署libtorch与配置契约，也未替换部署或运行Gazebo。上述“实际导出待确认”状态现更新为“文件生成已确认，契约与效果待确认”（AMP-17/18）；回放后续的持续运行不能仅由这一行日志验证。

## 为什么第一个launch就出现机器人

源码核对：gazebo.launch.py第82行定义gazebo_ros/spawn_entity.py，读取已解析URDF、entity imgo2_gazebo、出生z=.5；它在LaunchDescription中直接启动。所以Gazebo launch负责生成模型，不需等rl_sim加载策略。rl_sim.cpp的StartJointController只启动关节控制器spawner，不是机器人spawn；按0/3才进入起身和AMP控制。机器人显示不代表policy已经执行。

git HEAD中的launch也已有同一spawn_entity链路，本轮未新增它，最近4000策略替换也不能决定模型是否生成。此前用户提供过gzserver/gzclient GLX BadValue及spawn_entity等待日志：服务失败可导致当时无模型/无显示；若原来使用另一launch，也可能仅开世界。仅凭当前现象不能断言某次历史启动的唯一原因，本轮未复现GUI。已完成源码/HEAD核对，无代码改动。

## 2026-10-13：已激活env_isaaclab仍提示找不到解释器

启动器未设置IMGO2_ISAACLAB_PYTHON时仍选择固定默认 `/opt/conda/envs/isaaclab/bin/python`，不会自动使用已激活Conda的Python。在已激活env_isaaclab的终端先执行：

```bash
export IMGO2_ISAACLAB_PYTHON="$CONDA_PREFIX/bin/python"
```

然后重新执行原play命令即可。export对当前终端后续命令生效，新终端需重新设置；无需重新安装。play.py支持checkpoint绝对路径，与train.py按load_run/文件名定位的行为不同。

本次用户使用的文件已确认存在：

```text
/home/tommy0929/Documents/imgo2/imgo2_rl/logs/amp_rsl_rl/base_move_amp_height/2026-10-14_00-20-45_motion_new_model_new/model_25000.pt
```

已修复：补齐激活环境与启动器解释器选择之间的说明。验证：指定本机env_isaaclab解释器经真实启动器执行轻量Python检查，sys.executable为 `/home/tommy0929/miniconda3/envs/env_isaaclab/bin/python`，上述checkpoint存在。未启动Isaac Sim或play，未导出/替换策略，启动器源码未改。待确认：实际play加载/导出及新策略效果，沿用AMP-17/18验证事项。
# 2026-10-14 checkpoint路径拼写核对

用户play报错 `Unable to find the file`，命令目录开头重复写成`20262026-10-14_00-20-45_motion_new_model_new`。本机只读检查确认正确目录为`2026-10-14_00-20-45_motion_new_model_new`，其中model_25000.pt真实存在；删除重复的2026即可修正此次文件查找错误。CPU powersave/IOMMU提示不是这次异常的触发点。

已完成：路径错误归因及正确checkpoint存在性核对；本轮未运行play、导出或替换policy。待确认：修正命令后的加载、回放与部署效果，沿用AMP-17/21/22。
