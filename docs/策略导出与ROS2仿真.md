# AMP 策略导出与 ROS 2 Gazebo 仿真

本机路径示例，核对日期2026-10-12。用户称“sim2real”，但此处要求的是导出后启动ROS2仿真，实际为 **sim2sim**；本文不包含真机电机启动。

最终路径检查发现先前motion_new运行目录已不在原位置，未找到迁移目的地；使用目前仍存在的原始 `2026-09-24_00-57-38_xiaoji_real1/model_24000.pt` 作为可执行路径例子。换策略时修改checkpoint路径；其它运行会把导出文件写到各自checkpoint同级的exported目录。

## 1. model_24000.pt 导出为 policy.pt

在终端A执行：

```bash
cd /home/tommy0929/Documents/imgo2/imgo2_rl

IMGO2_ISAACLAB_PYTHON=/home/tommy0929/miniconda3/envs/env_isaaclab/bin/python \
PYTHONPATH="$PWD/source/imgo2_rl:$PWD/scripts/rl_lab${PYTHONPATH:+:$PYTHONPATH}" \
bash scripts/run_isaaclab.sh scripts/rl_lab/amp/play.py \
  --task=Imgo2-basemove-flat-amp-height-play \
  --checkpoint=/home/tommy0929/Documents/imgo2/imgo2_rl/logs/amp_rsl_rl/base_move_amp_height/2026-09-24_00-57-38_xiaoji_real1/model_24000.pt \
  --num_envs=1 --device=cuda:0 --headless
```

使用本机已验证的Isaac环境解释器，以及当前项目两份源码。这是回放/导出，不是训练。生成：

```text
/home/tommy0929/Documents/imgo2/imgo2_rl/logs/amp_rsl_rl/base_move_amp_height/2026-09-24_00-57-38_xiaoji_real1/exported/policy.pt
/home/tommy0929/Documents/imgo2/imgo2_rl/logs/amp_rsl_rl/base_move_amp_height/2026-09-24_00-57-38_xiaoji_real1/exported/policy.onnx
```

确认两个文件生成后按Ctrl+C停止回放，再执行下一步。另一checkpoint在同目录导出会覆盖exported，所以拷贝前确认本次导出已完成。

## 2. 备份并替换部署 AMP 策略

先停止已有rl_sim，再执行：

```bash
cd /home/tommy0929/Documents/imgo2/imgo2_rl

mkdir -p logs/amp_rsl_rl/base_move_amp_height/2026-09-24_00-57-38_xiaoji_real1/deployment_backup
cp ../imgo2_deploy/policy/imgo2/amp/policy.pt \
  "logs/amp_rsl_rl/base_move_amp_height/2026-09-24_00-57-38_xiaoji_real1/deployment_backup/policy_before_copy_$(date +%Y%m%d_%H%M%S).pt"

cp logs/amp_rsl_rl/base_move_amp_height/2026-09-24_00-57-38_xiaoji_real1/exported/policy.pt \
  ../imgo2_deploy/policy/imgo2/amp/policy.pt
```

model_24000.pt是训练checkpoint，导出的policy.pt才是部署TorchScript。本轮只写指令，不替换当前部署4000策略；执行以上cp后才会换成示例24000。原始24000曾完成一致性核对与真实Gazebo基准，见 [基准记录](amp_collection_review_2026-10-11.md#原始-xiaoji_real1-24000-同条件补测已完成)。更换策略不需重新编译，已有rl_sim需重启；本步骤不会改变奖励或模型文件。

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

本轮只写命令，未重新运行导出/替换/仿真。已确认示例checkpoint、本机Python、real模型、ROS安装setup和launch存在；Bash语法核对通过。先前4000策略导出契约已通过、其Gazebo效果仍待验证（AMP-17/18）；本例24000历史基准不替代用户新checkpoint的验证。新motion_new原目录缺失事实已记录，若要看新策略需提供其现位置或重新同步。真机sim2real需要另行确认整机硬件入口与观测/关节映射，不能把本页rl_sim命令当成真机启动。

## 为什么第一个launch就出现机器人

源码核对：gazebo.launch.py第82行定义gazebo_ros/spawn_entity.py，读取已解析URDF、entity imgo2_gazebo、出生z=.5；它在LaunchDescription中直接启动。所以Gazebo launch负责生成模型，不需等rl_sim加载策略。rl_sim.cpp的StartJointController只启动关节控制器spawner，不是机器人spawn；按0/3才进入起身和AMP控制。机器人显示不代表policy已经执行。

git HEAD中的launch也已有同一spawn_entity链路，本轮未新增它，最近4000策略替换也不能决定模型是否生成。此前用户提供过gzserver/gzclient GLX BadValue及spawn_entity等待日志：服务失败可导致当时无模型/无显示；若原来使用另一launch，也可能仅开世界。仅凭当前现象不能断言某次历史启动的唯一原因，本轮未复现GUI。已完成源码/HEAD核对，无代码改动。
