# 前部摄像头俯仰状态（2026-11-13）

## 目的与范围

用户需要通过躯干俯仰，让四足前部摄像头观察更高处的仪表。新增独立 `Pitch` 和 `PitchReturn` 状态，接入实机 `rl_real_imgo2` 与离线 `fsm_check`。不改原 Stand/PPO/Passive/Hold/Fault 状态实现、SDK、策略、Motor/IMU、零位与关节方向；原有首次站起/趴下 4 s、PPO 回站 2 s、固定 PD 60/2、PPO PD 25/0.8 保留。开始时工作树已有修改和大量诊断产物，保留原状；基线见本轮此前只读复审（本机 `real_control_review_2026-11-13.md`）。沿用本地 HEAD `e9c2719687f393db555b9f3c04a817d0e96201f2`，此前同步受 `.git/FETCH_HEAD` 只读限制，本轮未合并、提交或推送。

**已完成：**新状态、X/右摇杆、固定角速度与角度边界、保持/回站/PPO 门禁、两个主程序接入、离线键盘覆盖、运动学审计及动画。**待确认：**真实承重状态下的姿态、足端打滑、机械/力矩裕量与摄像头实际覆盖；记为 README `PITCH-HW-01`。本轮没有真实电机、IMU、相机运行，也没有物理引擎仿真或训练。

## 使用

当前已构建到 `/tmp/imgo2_real_review_2026-11-13`。在本机可直接运行：

```bash
/tmp/imgo2_real_review_2026-11-13/fsm_check --keyboard --imu-rpy 0,0,0
```

`--keyboard` 仅覆盖离线程序本次输入源，不改源 YAML 默认手柄配置，不需要手柄/电机/IMU。这个离线程序会加载真实策略，仍需现有 LibTorch；俯仰时不做 PPO 推理。

| 步骤 | 无手柄键盘 | 手柄（两个主程序共用） | 行为 |
|---|---|---|---|
| 站起 | `0`+回车 | A | 原 4 s 站起，StandHold 后实际/模拟反馈连续稳定 0.5 s |
| 进入俯仰 | `3`+回车 | X | 仅接受站稳保持状态；从 0° 开始 |
| 仰身 | D | 右摇杆水平向右 | 固定 10°/s，最大 +30° |
| 俯身 | A | 右摇杆水平向左 | 固定 10°/s，最小 −20° |
| 保持 | 空格 | 松开右摇杆回死区 | 停止角度积分，保持当前关节目标；不是回零 |
| 回站 | `0`或`3`+回车 | A 或再次 X | `PitchReturn` 用 1 s 身体空间五次轨迹回到 0°，然后 StandHold |
| PPO | `1`+回车 | Y | 仅 StandHold 且再次连续站稳后接受；Pitch 和 PitchReturn 都拒绝，也不排队 |
| 趴下 | `9`+回车 | B | Pitch 也可进入原趴下流程，按实测位置开始原 4 s 轨迹 |

键盘不能检测松键，D/A 为持续方向命令，必须按空格保持；不是键盘松开自动停止。手柄右摇杆已过原死区，之后只取正/负方向，不随推杆幅度改变角速度。左右摇杆其他轴在 Pitch 下不控制平移/转向。手柄断开即停止积分、保持当前角度；实机入口沿用原重连机制。A/B/X/Y 同拍时按此顺序处理一个状态请求；保留原 A/B 优先级。

从 PPO 按 X 不会直接转俯仰，需先 A 回站并站稳；从俯仰按 Y 不会进入 PPO。回站途中不能重新进入俯仰；等到站稳后再按 X。

重新构建（从仓库根运行；`/tmp` 是本轮独立构建目录，不覆盖用户旧 build 缓存）：

```bash
python3 imgo2_rl/scripts/tools/export_pitch_geometry.py --check
cmake -S imgo2_deploy/real -B /tmp/imgo2_real_review_2026-11-13 \
  -DCMAKE_BUILD_TYPE=Release -DGO_IMU_BUILD_RL=ON \
  -DGO_IMU_BUILD_APPS=ON -DPYTHON_EXECUTABLE=/usr/bin/python3
cmake --build /tmp/imgo2_real_review_2026-11-13 -j2
ctest --test-dir /tmp/imgo2_real_review_2026-11-13 --output-on-failure -j2
```

实机入口仍为同目录 `rl_real_imgo2`；其正常启动会打开真实设备。本轮只运行了正式入口的 `--check-policy` 与编译时隔离的 `rl_real_imgo2_pty`，未执行正式硬件控制。默认从可执行文件旁读取资源，源 YAML/策略在 build 时同步；不改已运行进程。

## 参数与文件

`config/deployment.yaml` 新增：

```yaml
pitch_speed_deg_s: 10.0
pitch_min_deg: -20.0
pitch_max_deg: 30.0
pitch_return_stand_s: 1.0
```

速度和返回时间必须为有限正数；范围可收窄，不能突破 −20°/+30°，且必须跨过 0°。启动预检扫描全范围的可达性、模型/硬件关节限位、所配速度和回站速度；每拍再次确认整个目标不需要限幅，避免单独截断某个关节后破坏固定足端约束。严重调度停顿后最多积分两个主控拍的角度，避免一次补上很大的角度跳变。

| 文件 | 实现职责 |
|---|---|
| `real/function/pitch_kinematics.{h,cpp}` | 四腿正/逆运动学、全范围预检 |
| `real/function/pitch_geometry_generated.h` | 从唯一模型源导出的最小运动学表，不是新的 URDF/模型副本 |
| `imgo2_rl/scripts/tools/export_pitch_geometry.py` | 标准库生成/校验运动学表，修改模型后运行；生成物不可手改 |
| `real/fsm/states/pitch_state.{h,cpp}` | Pitch 定速积分、保持；PitchReturn 五次身体角轨迹与关节速度前馈 |
| `real/fsm/deployment_states.*` | 注册新状态、X/3/A 及 PPO 入口门禁；旧模式编号 0…6 不变，新 Pitch/PitchReturn 为 7/8 |
| `real/input/controller_input.*` | X 的 evdev `BTN_NORTH` 按下沿；原 Y=`BTN_WEST` 保留 |
| `real/fsm_check.cpp` | `--keyboard`、3 事件、目标俯仰角显示；原 PPO 固定反馈逻辑保留 |
| `real/rl_real_imgo2.cpp` | 同一 FSM/输入链路；帮助及日志 metadata 增补俯仰参数/模式名，CSV 原列布局保留 |

表中 `real/` 均指 `imgo2_deploy/real/`。原电机状态、零位、方向与传动换算不在新运动学层重复应用。

## 为什么能够绕中心俯仰

枢轴取 `base` 原点 `(0,0,0)`，也是当前主躯干碰撞盒的几何中心，**不是整机随腿运动而变化的质心**。世界坐标原点选在初始站姿的这个中心。初始四个足端接触球球心由 `stand_pose` 和当前 `imgo2_description/urdf/imgo2_real.urdf` 正运动学得到并固定。

每拍给定机身角度后，将固定世界足端反变换回机体坐标，再逐腿逆解髋、大腿、小腿；机身平移恒为零。大腿和小腿共同变化，前后腿解不同，不是单改大腿把机身前后挪动。回站也是角度五次插值后重新逆解，不复用关节线性插值。正常 Pitch 用固定 PD 60/2，目标速度由逆解对身体角的导数换算；松杆/限位时目标速度归零。

模型中的 `*_Ankle` 固定旋转和 FOOT 球心偏移已计入，不能只把 shank 长度当接触点。几何表记录当前 URDF SHA256 `1d97de7806341b376df619c48c5ce463f72e16d87f47211bf74c7acfddfc30a2`，没有改模型、关节轴或限位。需要改几何时先改唯一源，再重新生成本表并校验。

**正角度专指仰身/机头抬起**；在 X 前、Y 左、Z 上的右手坐标中，对应数学旋转 `Ry(-θ)`，不能直接当 IMU Euler pitch 的正角。摄像头安装在中心前方会沿圆弧抬高且视线向上，不是保持摄像头自身位置不动。

## 验证与限制

环境：`/usr/bin/cmake`、GNU C++ 11.4.0、`/usr/bin/python3`，现有 LibTorch/yaml-cpp。新 Release 编译通过；最终全量 CTest 结果见同目录机器可读验收摘要（本机 `pitch_control_validation_2026-11-13.json`）。

- `pitch`：1001 个角度的 C++ FK/IK、模型/硬件限位、非法配置/不可达拒绝，固定速率/摇杆幅度无关、松杆/断线保持、上下限、X/A 返回、返回 0.995 s 仍在途中/1.000 s 完成、回站后重新站稳、禁止 Pitch/PitchReturn→PPO、保留 PPO 原 2 s 回站与趴下流程。
- `controller_input`：FIFO 模拟 evdev，核对真实 X/Y/A/B 编码、按下沿、重复事件不重触发、轴正号/死区与断线。
- `pitch_app`：运行真实 `fsm_check` 主程序，分别测试 `--keyboard` 和模拟手柄；覆盖目标角变化、保持、X 回站约 1 s、Y 被拒绝及回站后才可 PPO。
- `rl_deployment_pty`：新增 Pitch 场景，真实应用主循环经 4 路伪电机串口 + 伪 IMU 下发命令；验证四路大腿和小腿目标均变化、禁止直接 PPO、A/0 约 1 s 回站，并保留原故障/卸力场景。没有访问 `/dev/robot/*`。
- 独立 Python 审计：对同一个 C++ FSM 导出的 2190 拍关节目标，用 XML 关节变换逐级计算 FK，与 C++ 解析逆解独立；全程最大球心位置误差约 `4.2131e-8 m`，俯仰实到 −20°/+30°，关节位置/目标速度未越模型限位。详见数值与动画数据（本机 `pitch_control_kinematics_2026-11-13.json`）及逐拍目标（本机 `pitch_control_trace_2026-11-13.csv`）。
- 原 Stand/PPO/Passive/Hold/Fault 状态文件、Motor/IMU、SDK、策略及 real_config 共 49 个既有受保护文件与开始基线逐文件 SHA256 一致；新需求只扩展调度、输入和配置等接入处。

全量初跑 31/32 通过，`rl_relocation` 找不到 `libtorch_cpu.so`。用 `git archive HEAD imgo2_deploy/real` 的旧版本在同样存在本地 library 的条件下复现相同错误：测试把运行库复制入临时源码，安装时引用该路径，随后删除源码。只修测试复制夹具，保留通过 `Torch_DIR` 提供的外部运行库；复测安装搬移/策略预检通过。此测试证明独立源码配外部 LibTorch 的模式，不代表已经把 LibTorch 打包进可搬移分发目录；生产加载/控制实现未因此修改。

仍待 `PITCH-HW-01`：当前是基于已标定关节与固定足端假设的运动学位置控制，未新增 IMU 姿态闭环。实际地面、负载、PD 跟踪误差、足端打滑会导致机身中心/角度偏差；需真实设备小角度起步，再逐步核对全范围、1 s 回站和实际相机画面。`fsm_check` 是理想关节跟随，IMU 为原注入值，面板 `Pitch nose-up` 表示目标角，不能作为实测姿态或动力学通过。相机外参/视场未提供，动画安装点与仪表仅为示意，不验收真实可见高度。

## 动画与复现

动画来自上述同一 C++ 状态机目标，不另写一套假运动。当前序列为仰身到 +30°、保持、俯身到 −20°、保持、1 s 回站。可直接查看GIF 动画（本机 `pitch_control_preview_2026-11-13.gif`）与+30° 静帧（本机 `pitch_control_preview_2026-11-13.png`）。

```bash
/tmp/imgo2_real_review_2026-11-13/test_pitch \
  imgo2_deploy/real/policy/ppo imgo2_deploy/real/config/deployment.yaml \
  docs/deployment/pitch_control_trace_2026-11-13.csv
python3 imgo2_rl/scripts/tools/audit_pitch_trace.py \
  docs/deployment/pitch_control_trace_2026-11-13.csv \
  --output docs/deployment/pitch_control_kinematics_2026-11-13.json
# 下列仅为绘制，依赖现有 Pillow；上面的检查只需标准库
python3 imgo2_rl/scripts/tools/render_pitch_animation.py \
  docs/deployment/pitch_control_kinematics_2026-11-13.json \
  --output docs/deployment/pitch_control_preview_2026-11-13.gif
```

交互版本另在本聊天显示，使用同一审计数据；JS 语法及 320/736 px 的轻量 DOM 播放/暂停/时间拖动检查通过。静帧已实际查看并修正视线裁切。环境禁止本地 HTTP 监听、浏览器拒绝 file 协议，未做真实浏览器截图验收，不把轻量 DOM 检查称为完整浏览器渲染测试。
