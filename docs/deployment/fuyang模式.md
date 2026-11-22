# fuyang 模式：原理与跨电脑移植说明

核对日期：2026-11-13。用途：明天在另一台电脑中，给结构相似的四足控制程序加入同效果的俯仰功能。本文解释当前实现的依据及移植契约；当前项目的完整验收、动画和历史问题统一见[俯仰实现记录](pitch_control_2026-11-13.md)，不把本机测试结果当作另一份代码已通过。

## 1. 这套功能是按照什么写的

依据是**当前机器人的 URDF 几何、刚体坐标变换、四腿逆运动学和现有电机位置 PD 接口**。它不使用新策略，也没有用经验角度表单独改变大腿角度。

需求是让安装在躯干前部的摄像头看向高处仪表：控制躯干绕身体中心抬头/低头，同时让四个足端保持原位。前腿和后腿分别求解，大腿、小腿联动；摄像头随躯干转动，前部安装点还会沿圆弧抬高。

实现分成四层：

```text
手柄方向 → 目标机身俯仰角 → 固定足端的四腿 IK → 各关节 q_des / dq_des → 原电机 PD
```

当前是**运动学位置控制**，没有新增 IMU 姿态闭环。保持不变的是计算目标中的机身中心和世界足端；实机上还取决于零位、关节跟踪和接触是否成立。目标角不是实际测得角，base 几何中心也不是整机质心。

## 2. 迁移后应保留的操作契约

| 输入/条件 | 必须实现的行为 |
|---|---|
| 站稳后按 X | 进入独立俯仰状态，从 0° 相对站姿开始 |
| 右摇杆水平正方向 | 以固定角速度仰身，最大 +30° |
| 右摇杆水平负方向 | 以相同固定角速度俯身，最小 −20° |
| 摇杆回死区 | 停止角度积分，保持当前角度；不是自动回站 |
| 俯仰中再次 X 或按 A | 经独立身体角轨迹在 1 s 回到站姿，再进入站稳保持 |
| 俯仰中或回站中按 Y | 拒绝进入 PPO，不排队 |
| 回站且反馈重新站稳后按 Y | 才允许 PPO |
| 手柄断开 | 停止俯仰积分并保持当前角度；其他硬件故障仍走原故障处理 |

当前实现仅允许从 `StandHold` 且连续站稳进入俯仰；PPO 中按 X 也拒绝，需先回站。俯仰中 B 可转入原趴下流程，保持原流程的到位卸力规则。首次站起、PPO 回站、俯仰回站是不同流程，不能把它们都改成 1 s。

默认参数为下列值，不代表另一台机器已完成实机验收：

```yaml
deployment:
  pitch_speed_deg_s: 10.0
  pitch_min_deg: -20.0
  pitch_max_deg: 30.0
  pitch_return_stand_s: 1.0
```

原项目主控周期为 5 ms；俯仰直接随主控运行，不需要等 20 ms 的策略推理拍。范围可以收窄，当前代码不允许扩到 −20°/+30°以外。

## 3. 首先统一坐标、单位和顺序

移植前在目标代码上逐项确认，不要按变量名字猜：

| 项目 | 当前项目约定 | 目标代码需要确认的内容 |
|---|---|---|
| 机体坐标 | X 前、Y 左、Z 上 | 是否相同；若不同先转换 |
| 用户正俯仰 | 机头抬起 | 不直接套 IMU Euler pitch 的正号 |
| 数学旋转 | 仰身 θ 对应 `Ry(-θ)` | 目标库的旋转方向和主动/被动变换约定 |
| 位置/速度单位 | 输出轴 rad、rad/s | 是否仍是逻辑输出轴，还是电机转子单位 |
| 硬件关节顺序 | FL/FR/BL/BR，每腿髋/大腿/小腿 | 是否使用 RL/RR 名称、腿块顺序或关节角色顺序不同 |
| 几何腿名 | 模型使用 FL/FR/RL/RR | 当前代码 BL/BR 对应模型 RL/RR，不能误换左右后腿 |
| 方向、零点、减速比 | 原硬件层处理 | IK 层不能重复做一遍 |
| 身体中心 | `base` 原点，主躯干碰撞盒几何中心 | 目标模型 base 原点是否位于想要的中心 |

“左右腿使用同一套正负号”只在模型的逻辑关节轴约定相同时成立。不要为迁移方便修改已确认的关节轴/限位，也不要改上电归零流程。

## 4. 核心几何：足端不动，反算腿应该怎么摆

### 4.1 初始足端锚点

先由固定站姿 `q_stand` 做 FK，求每条腿的足端点 `p0[i]`。当前选择的是**足端接触球的球心**，不是小腿末端、脚 link 原点或网格原点。保持球心不动，球面接触点才具有一致的几何依据。

只在建立这次俯仰基准时生成足端锚点；不要每拍根据新目标重新生成，否则会丢失“固定世界足端”的约束。

当前模型来源为 `imgo2_description/urdf/imgo2_real.urdf`。几何表自动包含：髋安装位置、髋到大腿偏移、大腿到膝偏移、膝到足端球心的等效向量、关节轴和限位所要求的几何条件。

特别注意：`*_Ankle` 有固定旋转，FOOT 球心有非零偏移。等效膝到球心向量必须是：

```text
v_knee_to_contact = ankle_origin_xyz + R_ankle_origin * contact_sphere_origin_xyz
```

不能直接把 `0.206 m` 当作到接触球心的完整小腿长度。当前导出器会校验所支持的轴、原点旋转、链路和球心，再生成最小几何表；它不是第二份 URDF。

### 4.2 绕机身中心转动

本实现以初始水平机身为参考，中心为 `(0,0,0)`。给定仰身角 θ：

```text
机体 → 世界：p_world = Ry(-θ) * p_body
因此：      p_body_des[i] = Ry(+θ) * p0[i]
```

θ 用 rad。第二个公式就是每拍四腿 IK 的足端目标：先让“世界足端固定”成立，再反求相对旋转后机体的足端坐标。

如果目标程序的旋转中心不在 base 原点，要显式加入枢轴平移。通用写法如下，不能只改一个角度符号：

```text
c_body  = 机体坐标中的目标枢轴
c_world = t_initial + R_initial * c_body
p_world_fixed[i] = t_initial + R_initial * FK(q_stand[i])
R_des = R_initial * Ry(-θ)
p_body_des[i] = c_body + transpose(R_des) * (p_world_fixed[i] - c_world)
```

这个写法保证枢轴在世界中固定；若 `c_body != 0`，base 原点本身会沿圆弧移动，这是绕指定枢轴转动应有的结果。本项目取 `R_initial=I`、`t_initial=0`、`c_body=0`，不使用实时 IMU 来修正这个参考。

### 4.3 四腿逆运动学

优先复用目标程序经过验证的腿部 IK。如果模型几何不同，不要直接拷当前解析公式。

当前解析 IK 的适用结构是：髋绕局部 X，大腿和小腿绕局部 Y；前三个关节 origin 无固定旋转；大腿到膝主要沿负 Z，同时允许模型中的侧向偏移；膝到接触球心允许 X/Y/Z 偏移。导出脚本会检查这些前提，正常使用弯膝分支。

每条腿求解后，要同时核对：可达性、接近奇异位形、模型限位和硬件限位。不能为了得到结果，把明显不可达目标的余弦直接夹到 ±1 后继续执行；也不能单独把某个关节截到限位后继续声称足端固定。

## 5. 每拍控制逻辑

下面是移植所需的控制骨架，不是可以直接覆盖目标项目的完整源文件。`solve_ik_all_legs()` 必须使用目标机器人的几何与顺序。

```cpp
// 状态入口：从已站稳的目标站姿建立/校验基准。
theta = 0.0;                  // 目标仰身角，rad
axis_direction = 0;            // -1 / 0 / +1
last_time = steady_clock_now();

// 每个主控拍：输入轴已经过手柄死区。
axis_direction = !isfinite(right_axis) || right_axis == 0
                   ? 0 : (right_axis > 0 ? +1 : -1);

dt = seconds(now - last_time);
if (dt < 0) { enter_fault(); return; }
dt = min(dt, 2.0 * control_period_s); // 不补偿很久的调度停顿
last_time = now;

theta_dot = axis_direction * deg_to_rad(pitch_speed_deg_s);
theta = clamp(theta + theta_dot * dt,
              deg_to_rad(pitch_min_deg), deg_to_rad(pitch_max_deg));
// 若已经顶到本方向角度边界，令theta_dot=0；反方向仍可离开边界。

q_des = solve_ik_all_legs(theta);
dq_des = zeros(12);
if (theta_dot != 0) {
    eps = 1e-4; // rad，当前实现用数值导数换算关节速度
    dq_des = (solve_ik_all_legs(theta + eps)
              - solve_ik_all_legs(theta - eps)) / (2 * eps) * theta_dot;
}

kp = fixed_stand_kp;
kd = fixed_stand_kd;
tau_ff = zeros(12);
// 整份目标必须合法；需要改变单关节才能合法时拒绝并走原故障流程。
validate_whole_target(q_des, dq_des, kp, kd, tau_ff);
send_through_existing_hardware_interface();
```

当前 `fixed_stand_kp/kd` 为 60/2，沿用原固定姿态增益。不要因为俯仰就改原 PPO 增益或导入策略的 default pose。PPO 下右摇杆仍控制原转向；只有 Pitch 状态把同一根轴解释为身体俯仰方向。

最终使用的是关节位置与速度目标，电机内部等效 PD 为：

```text
tau_total = Kp * (q_des - q_feedback)
          + Kd * (dq_des - dq_feedback)
          + tau_ff
```

原硬件层负责单位换算和通信。`tau_ff=0` 不等于没有力矩；夹住前馈力矩也不等于限制了内部 PD 总力矩。保持角度时仍然主动支撑，不应清零增益。

## 6. 返回站立必须对身体角插值

单独增加 `PitchReturn`，不要直接把各关节从当前角插值到站姿，这样才能在回站计算中继续保留固定足端和中心约束。

进入回站时保存 `theta_start`，设 `T=pitch_return_stand_s`。每拍计算：

```text
u = clamp(elapsed / T, 0, 1)
s(u) = 10u³ - 15u⁴ + 6u⁵
theta = theta_start * (1 - s(u))
theta_dot = -theta_start * 30u²(1-u)² / T
q_des = IK(theta)
dq_des = dIK/dtheta * theta_dot
```

到 T 时目标角为 0，速度为 0，然后进入 `StandHold`。必须重新满足原位置/速度/连续稳定判据才能接受 Y；不要自动排队一个之前被拒绝的 PPO 请求。当前回站是从保存的**指令角**回零，不是从 IMU 实测角估计当前姿态。

速度需要分开理解：摇杆固定 10°/s，30°仰身约 3 s、20°俯身约 2 s；回站是 1 s 五次轨迹，峰值为 `1.875 * abs(theta_start) / T`。从 30°回站峰值约 **56.25°/s**，从 −20°约 **37.5°/s**。平滑并不等于很慢，不能把 10°/s 当作回站的限速。

## 7. 在另一份相似代码中修改哪些位置

不要整目录覆盖另一台电脑已测试的工程。建议按以下顺序逐项接入：

1. **配置**：新增四个 pitch 字段，补读取和有限值/范围校验。确认旧配置缺字段时有合理默认值，区分俯仰回站和 PPO 回站时长。
2. **运动学**：增加独立模块，只接收逻辑关节/几何。对零角做 `IK(FK(q_stand))` 一致性检查，再扫描全范围，不能等运行到 30°才发现不可达。
3. **状态**：新增 Pitch/PitchReturn、目标角、方向和时间。退出时清方向，非 PPO 状态不提交策略推理，沿用原故障/看门狗处理。
4. **状态切换**：站稳→Pitch；Pitch→PitchReturn；PitchReturn→StandHold。PPO 的允许起点仍只保留 StandHold。
5. **手柄**：增加 X 的按下沿事件，长按/重复事件不能反复切状态。实际键码先用目标电脑的手柄诊断工具核对；当前工程 X=`BTN_NORTH`、Y=`BTN_WEST`，不是所有输入库都这样命名。
6. **主程序**：让实机与离线入口调用同一套 FSM/IK，不要分别写两套俯仰算法。输入源、状态显示、命令发送、推理启停都要接通。
7. **构建/资源**：增加新 cpp 编译项，确保实际读取的部署 YAML 是更新后的那份。当前程序读可执行文件旁资源；只改源码 YAML 而没有 build 同步，可能仍跑旧配置。
8. **测试/记录**：先无设备回归，再伪硬件；保持原状态测试，并把目标电脑的构建与限制单独记录。

原日志中若用整数 mode，新增编号应追加，不要插入旧模式中间。本项目保留 0…6，追加 Pitch=7、PitchReturn=8；目标项目应先查自己的旧编号。

## 8. 可以带走的当前源码

下列路径均相对仓库根；这些文件已实际存在。若另一台电脑是同一项目，可用它们作参照；若是另一个项目，只迁移职责和数学契约，不照抄原项目路径/构造函数。

| 类型 | 文件 |
|---|---|
| 新运动学 | `imgo2_deploy/real/function/pitch_kinematics.h`、`pitch_kinematics.cpp`、`pitch_geometry_generated.h` |
| 新状态 | `imgo2_deploy/real/fsm/states/pitch_state.h`、`pitch_state.cpp` |
| 运动学表生成 | `imgo2_rl/scripts/tools/export_pitch_geometry.py` |
| 原模型源 | `imgo2_description/urdf/imgo2_real.urdf`，只在唯一源维护；不要给新程序再养一份模型副本 |
| 配置接入 | `imgo2_deploy/real/config/deployment_config.h`、`deployment_config.cpp`、`deployment.yaml` |
| 状态调度接入 | `imgo2_deploy/real/fsm/deployment_context.h`、`deployment_states.h`、`deployment_states.cpp` |
| 输入接入 | `imgo2_deploy/real/input/controller_input.h`、`controller_input.cpp`、`imgo2_deploy/real/apps/deployment_input.h` |
| 两个主程序 | `imgo2_deploy/real/rl_real_imgo2.cpp`、`fsm_check.cpp` |
| 构建接入 | `imgo2_deploy/real/CMakeLists.txt` |
| 状态/运动学测试 | `imgo2_deploy/real/tests/test_pitch.cpp` |
| 手柄/主程序测试 | `imgo2_deploy/real/tests/test_controller_input.cpp`、`test_pitch_app.py`；四电机总线场景见 `test_rl_deployment_pty.py` |
| 独立数值检查 | `imgo2_rl/scripts/tools/audit_pitch_trace.py` |
| 动画绘制 | `imgo2_rl/scripts/tools/render_pitch_animation.py`，仅绘图依赖 Pillow，不参与控制 |

生成器的默认仓库路径从脚本位置推导，依赖当前项目结构；另一个工程要相应调整模型输入/输出位置。生成表只适用于生成它的那份模型，不能把它当作所有四足通用几何。

本轮没有提交或推送；**只在另一台电脑 `git pull` 不保证拿到这些工作树修改**。需要实际传递文档及所需源文件/补丁，或另行完成版本同步。`/tmp` 下的可执行文件和会话记录也不会自动跨电脑同步。

## 9. 无设备验证怎么做

本机现有 `real/build` 的跨机旧缓存已于 2026-11-13 备份重建，现在可在该目录直接 `make -j2`，并运行 `./fsm_check --keyboard`。根因、实际路径与验证见构建缓存修复（本机 `real_build_cache_2026-11-13.md`）。下面使用独立临时目录的命令仍可用于另一台电脑首次配置；不要把本机 build 缓存当作可直接跨机使用的资源。

若另一台是相同仓库布局，且已经携带新增文件，可从仓库根运行下列命令。它们是本项目入口，不是任意工程通用命令；LibTorch/yaml-cpp 等依赖需在目标电脑实际存在。构建目录为目标电脑本地临时目录，不沿用本机旧二进制或 CMake 缓存。

```bash
python3 imgo2_rl/scripts/tools/export_pitch_geometry.py --check
cmake -S imgo2_deploy/real -B /tmp/imgo2_fuyang_build \
  -DCMAKE_BUILD_TYPE=Release -DGO_IMU_BUILD_RL=ON \
  -DGO_IMU_BUILD_APPS=ON
cmake --build /tmp/imgo2_fuyang_build -j2
ctest --test-dir /tmp/imgo2_fuyang_build --output-on-failure
/tmp/imgo2_fuyang_build/rl_real_imgo2 --check-policy
/tmp/imgo2_fuyang_build/fsm_check --keyboard --imu-rpy 0,0,0
```

`--check-policy` 是不开硬件的预检。`fsm_check --keyboard` 不需要手柄/电机/IMU；键盘 `0+回车` 站起、站稳后 `3+回车` 俯仰，D 仰身/A 俯身/空格保持，`0或3+回车` 回站。键盘不能检测松键，所以它的保持方式与手柄松杆不同。

最低验收项：

- θ=0 时 IK 复现原站姿，无入口跳变。
- 全范围 FK 验证：`Ry(-θ) * FK(IK(θ))` 与初始足端锚点一致；同时检查身体枢轴世界位置不变。
- 前后腿大腿/小腿确实联动；左右与逻辑轴一致；无单关节夹限后冒充求解成功。
- 两种非零推杆幅度产生相同角速度；松杆/断线保持；到上下限停止，并可反向离开。
- X 是按下沿；重复按 X 回站；回站 1 s 完成；途中 1/Y 被拒绝；回站后需重新站稳。
- 实机和离线主程序都走同一实现，四条模拟总线均收到对应关节目标。
- 原站立/PPO/趴下、故障与退出回归仍通过，PPO 原回站时长没有变成俯仰回站时长。

本机历史验收和证据见机器可读摘要（本机 `pitch_control_validation_2026-11-13.json`）；不要把其中的测试数、时间或哈希直接登记为目标电脑通过。

## 10. 待确认与首次实机边界

文档已完成；目标电脑的代码尚未提供，也没有在目标电脑构建或运行。需要当地核对：模型/枢轴/接触点、关节顺序/单位/方向、手柄映射、原 FSM/PPO 门禁与时效处理、编译依赖和实际读取的 YAML。只有这些完成并通过当地测试后，才能称为移植完成。

实机验证仍待 `PITCH-HW-01`：从小角度验证承重、中心偏差、足端打滑、关节跟踪与力矩裕量，再核对全范围和回站。没有相机外参/视场和实际画面，不能保证某个仪表一定进入视野。建议首测降低速度并收窄角度，这只是测试建议，**本次没有因此改动当前 10°/s、−20°～30°、1 s 参数**。

还要确认目标程序的停机语义：当前 Ctrl+C/硬件故障是卸力，机器人可能塌下；它不是保持站立的姿态急停。关节限幅与离线测试不等同于实际接触、力矩和防跌验收。

## 11. 给另一台电脑上的编程助手的任务说明

可以把本文和下列文字一起交给对方，要求先读目标源码再实施：

> 请依据本文，在当前四足工程中新增独立俯仰/俯仰回站状态。用途是抬高前部摄像头视线。保留已测试的原站立、PPO、趴下、零位/方向/传动换算和故障逻辑；先核对当前模型、关节顺序、单位、足端接触点及手柄事件。X 从站稳进入俯仰，右摇杆左右只取方向，以固定速度在 −20°～+30°之间改变目标角，松杆保持；再次 X 或 A 用独立身体角五次轨迹 1 s 回站，重新站稳后才允许 PPO。四足固定世界接触点、机身绕指定几何中心转动，用四腿 IK 联动关节，不只改大腿角度，也不把回站直接写成关节插值。若当前腿部几何不满足现有解析 IK 前提，使用当前工程正确的 IK。实机与离线主程序共用实现，先验证运动学、切换、限位和原功能回归。无设备时不启动硬件，不把离线通过写成实机验收，待确认项写入维护文档。

本次文档整理只做源码/参数与相对引用静态核对，没有重跑训练、硬件或全量回归，也没有修改生产代码。后续若改变限速、PD、角度范围、回站时长或增加 IMU 闭环，应作为新的控制改动单独验证。
