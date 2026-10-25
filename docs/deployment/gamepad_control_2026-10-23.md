# 手柄控制接入（rl_real_imgo2 / fsm_check，2026-10-23）

> 目标：在键盘之外增加 Xbox 兼容手柄（黑武士4 Pro 2.4G）控制；输入源在 `deployment.yaml` 选，
> 设备名/死区在 `real_config` 配；状态映射与摇杆速度映射放在 `RobotFSM`（`deployment_states.cpp`）里，
> FSM、安全层、日志都不另起一套。

## 1. 配置

### 1.1 选键盘还是手柄（`config/deployment.yaml`）

```yaml
deployment:
  # 1 = 键盘；2 = 手柄
  input_source: 1
```

- `1`：保持原来的 0/1/2/9 + WASDQE。
- `2`：读手柄；键盘的状态/速度不再生效（Ctrl+C 仍可退出）。

> 为兼容旧配置，`keyboard`/`gamepad` 字符串仍然接受；新配置统一写 `1`/`2`。

> 改完 `config/deployment.yaml` 后，`cmake --build build` 会自动把它同步到
> `build/config/deployment.yaml`（新增 `sync_config` 目标）；也可以直接
> `--config config/deployment.yaml` 读源文件，完全不依赖 build 里的副本。

### 1.2 手柄设备名与死区（`config/real_config.h`）

```cpp
class ControllerConfig {
public:
    std::string device = "/dev/imgo2_joystick"; // udev 别名，见 xbox_controller_demo 文档
    float deadzone = 0.08f;                     // 0..0.95
};
```

- 推荐把 udev 别名配成 `/dev/imgo2_joystick`（见 [手柄核对 §8](xbox_controller_demo_2026-10-23.md)）；
  也可以用 `/dev/input/by-id/...-event-joystick`。
- 两个入口都支持 `--controller-device <path>` 临时覆盖（`fsm_check` 测试 FIFO 就靠它）。

## 2. 映射规则

`input/controller_input.cpp` 读 evdev，换算成 `GamepadCommand`（-1..+1、已过死区）；
`fsm/deployment_states.cpp` 的 `RobotFSM::gamepad()` 再做状态和极限映射。

### 2.1 摇杆 → 速度

| 输入 | 轴 | 方向 | 映射 |
|---|---|---|---|
| 左摇杆 前后 | `ABS_Y` | 上 = 负，下 = 正 | `vx = -LY`（**上推为正**） |
| 左摇杆 左右 | `ABS_X` | 左 = 负，右 = 正 | `vy = -LX`（**左推为正**） |
| 右摇杆 水平 | `ABS_RX` | 左 = 负，右 = 正 | `wz = RX`（**右推为正**） |

- 死区：`|s| <= 0.08` 归零；死区外按 `(|s|-dz)/(1-dz)` 重新拉满，**整个行程映射到
  `deployment.yaml` 的 `velocity_limits`**：`vx ∈ ±limits[0]`、`vy ∈ ±limits[1]`、`wz ∈ ±limits[2]`。
- 摇杆只在 **PPO** 下生效；Standing/StandHold/Crouching 下不动速度（与键盘一致）。
- 进入 PPO 时速度会先清零；建议按 Y 前把左/右摇杆回中，避免进入瞬间跳速度。

### 2.2 按键 → 状态

| 物理键 | evdev | 动作 | 等价键盘 |
|---|---|---|---|
| A | `BTN_SOUTH` (0x130) | 站起 | `0` |
| B | `BTN_EAST` (0x131) | 趴下 | `9` |
| Y | `BTN_WEST` (0x134) | 进入 PPO | `1` |

- 只在**按下沿**触发一次，长按/重复不重复触发。
- 状态切换走现有 `RobotFSM::request(digit, ...)`，所以站起/趴下的轨迹检查、PPO 就绪判据、
  Fault/安全限幅全部复用，没有绕开任何保护。
- 同时按 A/B/Y 时优先级：站起 > 趴下 > PPO。

## 3. 断线/启动

- `rl_real_imgo2`：启动时打开手柄失败会直接报错退出（不会带着“没有输入源”跑）；
  运行中断线会把速度清零、打印一次原因，并**每秒尝试重连**，插回来自动恢复。
- `fsm_check`：测试工具，断线只把速度清零，不自动重连。
- 手柄模式不读键盘；Ctrl+C 始终可用（信号处理，与输入源无关）。

## 4. 构建与运行

```bash
cd imgo2_deploy/real
cmake -S . -B build
cmake --build build -j16

# 手柄模式（先在 deployment.yaml 把 input_source 改成 2）
sudo ./build/rl_real_imgo2
# 或临时覆盖设备名：
sudo ./build/rl_real_imgo2 --controller-device /dev/input/by-id/...-event-joystick

# 纯软件 FSM 测试（不打开电机/IMU）：
./build/fsm_check --config config/deployment.yaml
```

## 5. 验证

- **FSM 单测**（`tests/test_robot_fsm.cpp`）：
  - A → Standing；8 s 后 StandHold；Y → PPO；
  - 摇杆 `vx=1, vy=-0.5, wz=0.25` → `velocityCommand()` = `(limit0, -0.5·limit1, 0.25·limit2)`；
  - `connected=false` → 三项速度清零；B → Crouching。
- **`fsm_check` + FIFO 假手柄端到端**：A 站起 → Y 进 PPO → 左摇杆上+左、右摇杆右，
  在 `velocity_limits: [0.8, 0.4, 1.0]` 下实测面板 `cmd: 0.80 0.40 1.00`。
- **`rl_real_imgo2_pty` + FIFO 假手柄冒烟**：启动打印 `手柄: <path>（死区 0.08）`；
  按 A 打印 `开始站起` / `状态: Stand`；`meta.json` 记录 `input_source: gamepad` 与设备名。
- **全量 ctest：28/29**，唯一失败是既有的 `rl_relocation`（`cmake --install` 不打包 libtorch `.so`）。
- `policy_core` / `test_fsm_check.py` / `test_robot_fsm.cpp` 里写死的 `kd=0.5` 已改为从
  `policy/ppo/config.yaml` 读取，因为用户把 `rl_kd` 调成了 0.8。

## 6. 限制与待确认

- **按键码按标准 Xbox 布局**（A=South、B=East、Y=West）；尚未在真实 2.4G dongle 上逐键核对。
  如果实际码值不同，用 `controller_demo --list` / `--events` 确认后改
  `input/controller_input.cpp` 里的三个 `BTN_*`（目前没有做成配置项）。
- 只用了左/右摇杆和 A/B/Y；LB/RB、扳机、十字键、背键都未绑定。
- 摇杆是**速度指令**，不是“按一下走一步”；松手回中即零速。
- 实机未运行：真实手柄的摇杆手感、死区、最大速度、断线重连、PPO 下的稳定性都待上车确认。
- `fsm_check` 的假手柄测试是 FIFO（没有 absinfo，走默认 -32768..32767）；真实设备的范围由
  `EVIOCGABS` 读取，换算公式相同。
