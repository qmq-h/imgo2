# 新策略无头 Gazebo 站立/行走验证（2026-11-06）

## 目的

用仓库自带的 ROS 2 Gazebo 链路运行 `imgo2_deploy/real/policy/ppo/policy.pt`。
本轮检查两个候选：`b25b48c8…`（文件 `policy_lowspeed.pt`，上一版低速）和
`1199a85e…`（当前 `policy.pt`，部署候选，`rl_kp=28`），确认它们至少能加载、
站立和低速行走，而不是只在 `--check-policy` 里“形状对”。

## 环境与构建

- ROS 2 Humble / Gazebo（`gazebo_ros`、`gazebo_ros2_control` 已安装），系统 Python 3.10。
- **必须用系统 Python**：`colcon` 默认会取 `~/miniconda3/bin/python3`，`rosidl_adapter`
  报 `ModuleNotFoundError: No module named 'em'`。首次失败后还要删掉 `build/`，
  否则 CMake 缓存仍指向 conda。实际命令：

```bash
cd imgo2_deploy/ros2_ws
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/ros/humble/bin
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE
source /opt/ros/humble/setup.bash
bash build.sh          # 4 个包全部构建通过
```

> 改 `policy/ppo/{config.yaml,policy.pt}` 后，`cmake --build build`/`make` 会通过新增的
> `sync_policy` 目标同步到 `build/policy/ppo/`；`make rl_real_imgo2`/`fsm_check`/
> `rl_real_imgo2_pty` 有显式依赖。旧的 `POST_BUILD` 只在可执行文件重新链接时拷，
> 单换策略不会触发。CMakeLists 改动后先 `cmake -S . -B build` 重新配置一次。

- 推理库直接软链到 `real/library/inference_runtime/{libtorch,onnxruntime}`，
  `download_inference_runtime.sh` 校验通过，未重复下载。
- 沙箱下 `~/.ros`、`~/.gazebo` 只读：运行时把 `ROS_HOME`、`HOME` 指到工作区
  （`.ros*`、`.gazebo_home`），并在同一个 bash 任务里起 Gazebo + `rl_sim` + 评估脚本，
  避免不同 DDS/共享内存命名空间导致 `/param_node` 调用超时。
- **MuJoCo 预编译二进制没有使用**：`cmake_build_mujoco/bin/rl_sim_mujoco` 的
  `POLICY_DIR` 编译成 `/home/tommy0929/Documents/imgo2/...`，本机不存在；且缺
  `libmujoco.so.3.2.7`、`libglfw.so.3`。强行跑会加载错误路径的策略，所以本轮走 Gazebo。

## 策略与配置

- 当前部署候选：`policy.pt`，md5 `1199a85e26f0fac2bd98433650ee8959`，`rl_kp=28`、`rl_kd=1.0`。
- 上一版低速：`policy_lowspeed.pt`，md5 `b25b48c8a9aa5d3d806bef87ecc5a487`，`rl_kp=25`、`rl_kd=1.0`。
- 旧仿真策略：`ros2_ws/policy/imgo2/ppo/policy.pt`，md5 `e083a385…`（已恢复，可作基线）。
- 仿真配置：复制 real 的 `config.yaml`，末尾补 `joint_mapping: [0..11]`
  （ROS 2 仿真 `rl_sdk` 需要；real 配置已移除该键）。
- 测完已用 `git checkout --` 恢复仿真目录的原策略/配置；仓库里只保留 real 的新权重。

## 方法

```bash
# 同一个任务内：
ros2 launch imgo2_deploy gazebo.launch.py gui:=false   # 无头
ros2 run imgo2_deploy rl_sim                           # PTY 保持键盘循环
python3 scripts/eval_gazebo_policy.py --vx 0    --duration 14 --skip 4
python3 scripts/eval_gazebo_policy.py --vx 0.3  --duration 16 --skip 4
```

`eval_gazebo_policy.py` 自己发布 `/joy`（A 起身 → RB+DPadUp 进 PPO），并从
`/odom`、`/joint_states` 统计位姿、速度跟随、抖动和周期性。旧策略先跑 `vx=0`
作为链路基线。

> `1199` 复测时 Gazebo 的 RTF 较慢，`eval` 固定 6 s 起身窗口不够；改为手动以 20 Hz
> 发 A（`/joy`），等日志出现 `Getting up completed`，再跑
> `eval_gazebo_policy.py --no-getup ...`。b25b 用原来的 `--vx` 流程即可。

## 结果

| 运行 | vx 命令 | z 均值 (m) | roll/pitch 范围 (°) | 实测速度 (m/s) | 周期/相位 | 抖动 hip/thigh/shank (rad/s²) | 摔倒 |
|---|---:|---:|---:|---:|---|---:|---|
| 旧策略基线 | 0 | 0.3020 | 0.024 / 0.025 | 0 | — | 0.014 / 0.033 / 0.025 | 否 |
| **b25b 站立** | 0 | 0.2957 | **0.0010 / 0.0022** | 0 | — | **0.006 / 0.008 / 0.010** | 否 |
| **b25b 行走** | 0.3 | 0.3022（0.2982–0.3048） | 1.014 / 0.902 | **0.2723**（误差 9.2%） | 0.595 s / 1.68 Hz；FL-FR 167° | 10.71 / 20.96 / **45.09**；hip 换向 14.1/s | 否 |

b25b 行走额外指标：yaw 漂移 **−1.85°/s**，横向速度误差 0.005 m/s，步态周期强度 0.898。

### 最新 `1199`（当前 `policy.pt`，`rl_kp=28`）复测

| 运行 | vx 命令 | z 均值 (m) | roll/pitch 范围 (°) | 实测速度 (m/s) | 周期/相位 | 抖动 hip/thigh/shank (rad/s²) | 摔倒 |
|---|---:|---:|---:|---:|---|---:|---|
| **1199 站立** | 0 | 0.2952 | **0.0048 / 0.0158** | 0 | — | **0.0085 / 0.0085 / 0.0114** | 否 |
| **1199 行走** | 0.3 | 0.2974（0.2923–0.3040） | 2.833 / 1.710 | **0.2941**（误差 2.0%） | 0.597 s / 1.674 Hz；FL-FR −177.3° | 18.31 / 29.37 / **73.23**；hip 换向 21.5/s | 否 |

1199 行走额外指标：yaw 漂移 **+0.878°/s**，横向速度误差 0.016 m/s，步态周期强度 0.935。

与 b25b（0.3 行走）相比：速度跟随更好（2.0% vs 9.2%）、yaw 漂移更小（+0.88 vs −1.85），
但 **shank 抖动 73 vs 45、hip 换向 21.5 vs 14.1、roll 2.83° vs 1.01°**，明显更激进。

作为参考，旧策略在仿真配置注释里的 `vx=0.5` 记录为：z 0.288 m、roll 2.5°、
yaw −0.5°/s、大腿抖动 45 rad/s²、实测 0.428 m/s（低 14%）。两者不是同一 `vx`
的严格 A/B，只能看量级。

## 判读（b25b / policy_lowspeed）

- b25b 能被部署侧 libtorch 加载，Gazebo 里能起身、进 PPO、站立和行走，**没有摔倒**。
- 站立时 roll/pitch 几乎为零，关节抖动比旧策略低 2–4 倍；说明至少不是“权重坏了/动作尺度完全不对”。
- 行走速度跟随不错（0.3 命令 → 0.272，低 9.2%），trot 相位 167°接近 180°。
- 需要关注的是 **shank 抖动 45 rad/s²、hip 换向 14 次/s、yaw 漂移 −1.85°/s**。

## 判读（1199 / 当前 policy.pt）

- 1199 能被部署侧 libtorch 加载，站立 z=0.2952 m、roll/pitch≤0.016°、抖动与 b25b 同量级，
  Gazebo 站立很稳。
- 行走 0.3 命令实测 0.2941（误差 2.0%），无摔倒，trot 相位 −177°、周期强度 0.935，
  速度跟随和 yaw 漂移都优于 b25b。
- 但 **shank 抖动 73.2 rad/s²、hip 换向 21.5 次/s、roll 2.83°**，明显比 b25b 激进。
  结合实机已有的 7–9 Hz 抖动问题，这是部署前最需要重点验证的一项；不能只看“没摔倒”。
- 结论：**可以进入实机低速验证，但不建议直接按 b25b 的步态稳定性预期去跑**；
  先吊住/托住、低速、短时，重点记录 shank 抖动、7–9 Hz 和 yaw 漂移。

## 限制/待办

- Gazebo 模型与 real 的 8-body/接触/延迟/噪声不同；本轮没有真实电机、IMU、足地接触。
- 1199 训练时的 kp/kd、default pose、action_scale 是否与部署 `config.yaml`
  （`rl_kp=28`、`rl_kd=1.0`）一致，权重本身无法证明；需要按训练配置核对。
- MuJoCo 预编译二进制的 `POLICY_DIR` 写死用户机器路径且缺库；要在本机跑 MuJoCo
  需重新构建或改路径，本轮未做。
- 实机建议先吊住/托住，`vx=0` 站立、再 0.1–0.2 低速，记录 7–9 Hz、yaw 漂移和 shank 抖动；
  1199 的 shank 抖动在仿真里已比 b25b 高约 60%，实机要优先确认是否放大。
