# go_imu入口精简与下一步测试（2026-10-15）

## 删除与保留范围

按用户要求删除三个主程序：`go_probe.cpp`、`imu_reader.cpp`、`ppo_check.cpp`，同时删除CMake目标、资源复制/安装声明和工程build中的同名旧可执行文件，避免误运行残留版本。

删除前已检查源码/CMake和文档引用。两个仅服务于已删除入口的脚本`tests/test_probe_pty.py`、`tests/test_ppo_check.py`一并删除；`application_preflight`取消这两个硬件入口的参数断言，保留monitor与stand预检。底层协议、串口、Motor、IMU、硬件接口、策略SDK、状态类及配置/关节宏不改；对应通信、解析、安全、数值与故障测试仍保留。monitor帮助/面板文字取消对已删除probe的引用，采集行为不变。

原PPO固定反馈数值记录和历史源码哈希JSON保留，反映当时测试，不将历史文件名当作现用命令。已同步当前控制手册、架构、使用手册和根README，历史整理记录增加入口退役提示。

当前用户入口：

| 可执行文件 | 用途 | 构建条件 |
|---|---|---|
| go_monitor | 电机与IMU联合诊断 | GO_IMU_BUILD_APPS=ON（默认） |
| go_stand | 原独立站起FSM，只需电机 | GO_IMU_BUILD_APPS=ON |
| go_pd_demo | 原独立电机保持 | GO_IMU_BUILD_APPS=ON |
| rl_real_imgo2 | 正式串口硬件部署FSM与PPO | GO_IMU_BUILD_RL=ON |
| fsm_check | 相同FSM/SDK的模拟反馈键盘测试 | GO_IMU_BUILD_RL=ON |

测试辅助目标（如rl_real_imgo2_pty）不是用户硬件启动入口。

## 编译需要什么开关

**新测试程序和正式部署程序需要`-DGO_IMU_BUILD_RL=ON`，没有额外测试宏或ROS命令。**默认RL选项为OFF。CMake会将选项与依赖位置缓存到指定build目录；已有build已设置ON，后续只改C++可直接build。换新build目录时必须重新配置ON和LibTorch位置。

从仓库根目录，使用本机已存在的LibTorch：

```bash
cmake -S imgo2_deploy/real -B imgo2_deploy/real/build_layout \
  -DCMAKE_BUILD_TYPE=Release -DGO_IMU_BUILD_RL=ON \
  -DCMAKE_PREFIX_PATH="$PWD/imgo2_deploy/ros2_ws/library/inference_runtime/libtorch"
cmake --build imgo2_deploy/real/build_layout -j2
```

已有配置时，仅构建离线入口：

```bash
cmake --build imgo2_deploy/real/build_layout --target fsm_check -j2
```

运行时不需要再加`-D`参数。`-D`是CMake配置选项。本机依赖已由构建确认；其他机器仍需兼容C++11 ABI的LibTorch和yaml-cpp，不能据此认定复制工程后无需提供动态库。

## 建议的测试次序

### 1. 先测离线FSM与输出

```bash
./imgo2_deploy/real/build_layout/fsm_check \
  --policy-dir ./imgo2_deploy/real/policy \
  --config ./imgo2_deploy/real/config/deployment.yaml
```

依次确认：

1. 初始显示InitialHold；只输入0但不回车，状态不变。
2. 回车进入Stand，四腿目标在8秒内连续变化，结束为StandHold；默认站姿为0/0.87/−1.82rad。
3. 完成站起再等约0.5秒，1＋回车进入PPO；0.5秒交接后观察目标及25/0.5增益。
4. 先保持零速度观察，再单次按w/a/q看速度指令即时变化，空格归零；2＋回车显示AMP未开放，状态不变。额外测0＋回车从PPO用2秒回Stand，再连续稳定0.5秒后重新1＋回车；提前输入只提示、不排队。
5. 9＋回车，目标8秒变化到crouch_pose，满足到位条件后显示Passive/Unloaded和0/0增益；0＋回车可再站起。
6. Ctrl+C退出，光标恢复。终端至少53列、10行；不足时扩大窗口。

这个入口不会打开硬件。PPO反馈固定站姿、零dq、零IMU；输出不回灌，不能据此测试动力学稳定性。当前小腿目标相对站姿偏差约7.17°–20.05°，详细实测数据在 [FSM测试记录](fsm_check_2026-10-15.md)。进入PPO后不会强制保持配置站姿。

### 2. 模型与配置预检

```bash
./imgo2_deploy/real/build_layout/rl_real_imgo2 --check-policy \
  --policy-dir ./imgo2_deploy/real/policy \
  --config ./imgo2_deploy/real/config/deployment.yaml
```

这一步只加载/预热模型并预检配置与固定轨迹，不打开设备。通过不代表机身能站稳。

### 3. 用户执行真实反馈与固定姿态验证

先用go_monitor看12电机与IMU有效性、数据年龄和真实接收频率；monitor会发送零控制量，机械需要支撑。退出monitor后再运行控制，不能同时占用串口。

正式FSM从规定的机械启动姿态进行一次初始化；先只测试0＋回车站起、60/2保持、9＋回车趴下并卸力。站起后不能重启另一个控制程序再按启动姿态归零。再次测试应先回到规定启动姿态，或在同一运行实例内按状态切换。

PPO真实闭环建议放在固定姿态和映射验证之后：确认实际关节排列、方向、角度反馈、限位、IMU、停止方式及受控支撑条件；当前0.35rad小腿偏差需要结合真实动力学/闭环确认，先不要把“离线成功”当成直接落地行走的依据。不自动修改mapping_confirmed，目前正式配置保持false；离线fsm_check的内存解锁不影响真机入口。

Ctrl+C在正式入口请求卸力，不会自动执行8秒趴下；正常趴下需先9＋回车，完成卸力后再退出。这一行为与离线测试面板一致，但硬件卸力效果仍需实测。

## 验证结果与待确认

工程内重新配置/编译后，剩余 **30/30离线CTest通过（65.76秒）**。数量由32降到30是删除probe_pty与ppo_check两项入口专用测试，其他测试保留。包含FSM键盘测试、原状态、安全、协议/IMU、硬件伪终端以及复制工程构建/安装移动。monitor文字调整后重编译，combined_monitor/monitor/application_preflight **3/3通过（4.85秒）**。

已完成：三个入口、专用测试/目标和旧二进制删除，当前使用文档与编译开关说明同步；底层和配置策略指纹未变。待确认：真实闭环下PPO小腿偏移、映射与频率、轨迹和限位裕量、到位卸力效果仍按README DEPLOY-13跟踪，需要真实机械验证。本轮未启动真实硬件。
