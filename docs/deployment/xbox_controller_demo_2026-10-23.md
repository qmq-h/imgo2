# 手柄输入核对 demo（Xbox / 黑武士4 Pro，2026-10-23）

> 目的：在把键盘控制扩成手柄控制之前，先确认「手柄的每个键/摇杆/扳机，到 Linux 里到底是什么」。
> 本工具只读事件，不打开电机、不发任何控制指令；`rl_real_imgo2` 未改动。

## 1. 做什么

- 扫描 `/dev/input/event*`，按**能力**（有 `ABS_X`/`ABS_RX` + `BTN_*`）识别手柄，不只看名字。
- 打印设备信息：路径、名称、物理口、`bus/vendor/product/version`、**支持的全部按键/轴及轴范围**。
- 每按一个键/动一次摇杆，打印一行：`evdev 名称 + 编号 + 原始值 + 归一化值 + 范围`。
- `--state`：每 200 ms 打一行实时状态（组合键、摇杆、扳机、十字键）。
- 断线后（2.4G dongle 拔出/手柄休眠）自动等待重连，不用重启程序。

## 2. 构建与运行

```bash
cd imgo2_deploy/real
cmake -S . -B build            # 新增了 target，需要重新配置一次
cmake --build build -j16 --target controller_demo
sudo ./build/controller_demo           # 自动找手柄
./build/controller_demo --list         # 先看有哪些 event 设备
./build/controller_demo -d 5           # 指定 event5（event 号会变）
./build/controller_demo -d /dev/input/event5
./build/controller_demo -d vader       # 按名字子串
./build/controller_demo                # 默认：摇杆值在最后一行原地刷新，按键另起一行
./build/controller_demo --events       # 每个轴事件单独一行（老行为，便于看时序）
./build/controller_demo --raw          # 只打数字，便于脚本
```

**权限**：`/dev/input/event*` 通常是 `root:input 0660`。二选一：

```bash
sudo ./build/controller_demo
# 或者
sudo usermod -aG input $USER    # 重新登录后生效
```

程序打不开设备时会把 errno 打出来并给这两条提示。

## 3. 输出怎么读

**默认界面（推荐）**：

- **最后一行**是实时状态，原地刷新（每 100 ms 最多一次，不闪也不刷屏）：
  ```
  LX+0.03 LY-0.12 RX+0.00 RY+0.00  LT0.00 RT0.00  D+0,+0  [A][LB]
  ```
  `LX/LY/RX/RY` 是 **-1..+1、中点为 0**；`LT/RT` 是 0..1；`D` 是十字键；后面是当前按下的键。
- **按键另起一行**，不会被摇杆刷掉：
  ```
  1.234  KEY  BTN_SOUTH(A)           code=0x130 value=1  [按下]
  1.567  KEY  BTN_SOUTH(A)           code=0x130 value=0  [松开]
  ```
- 重定向/管道（非 TTY）时自动退回逐行模式；想要逐行看轴用 `--events`。

**逐行模式（`--events`）**：

```
设备: /dev/input/event5
名称: Microsoft X-Box 360 pad
ID:   bus=0x0003 vendor=0x045e product=0x028e version=0x0110
  按键: BTN_SOUTH(A) BTN_EAST(B) BTN_NORTH(X) BTN_WEST(Y) BTN_TL(LB) ...
  轴:   ABS_X(左摇杆X)[-32768,32767] ABS_Y(左摇杆Y)[-32768,32767] ABS_Z(...)[0,255] ...
   1.234  KEY  BTN_SOUTH(A)           code=0x130 value=1  [按下]
   1.567  KEY  BTN_SOUTH(A)           code=0x130 value=0  [松开]
   1.890  ABS  ABS_X(左摇杆X)          code=0x00 value=  1234  range=[-32768,32767] norm=+0.538
```

- **能力行**很重要：它列出内核认为这个手柄**有**哪些键/轴。某个键按了没反应，先看它是否在能力列表里——
  不在 = 驱动/模式没把它暴露出来；在 = 事件应该会打出来。
- `code` 是内核编号（十六进制），`value` 是原始值；对照 `linux/input-event-codes.h` 可以查任意编号。
- 未命名的码会显示 `(未命名)`，把编号发我即可补进名称表。

**类型与范围（2026-10-23 补）**：evdev 的 `input_event.value` 是 `__s32`（32 位有符号），本工具按 int32
打印/保存，**不会在 32767 处截断**；`range=[min,max]` 来自 `EVIOCGABS`。某值超出声明范围会标
`[超出声明范围!]`，退出时还会打印「本次观测到的轴原始范围」；摇杆额外打印以中点为 0 的 `signed`（-1..+1）。
注意：`/dev/input/js*` 的 `struct js_event.value` 是 `__s16`，用它读超过 32767 的值会回绕；控制请用 evdev。

## 4. 常见 Xbox 映射（仅对照，实际以打印为准）

| 物理键 | 常见 evdev |
|---|---|
| A / B / X / Y | `BTN_SOUTH` / `BTN_EAST` / `BTN_NORTH` / `BTN_WEST` |
| LB / RB | `BTN_TL` / `BTN_TR` |
| LT / RT | `ABS_Z` / `ABS_RZ`，部分驱动是 `ABS_GAS` / `ABS_BRAKE` |
| Back / Start / Guide | `BTN_SELECT` / `BTN_START` / `BTN_MODE` |
| L3 / R3（按摇杆） | `BTN_THUMBL` / `BTN_THUMBR` |
| 左摇杆 | `ABS_X` / `ABS_Y` |
| 右摇杆 | `ABS_RX` / `ABS_RY` |
| 十字键 | `ABS_HAT0X` / `ABS_HAT0Y`（-1/0/1） |

**不同模式映射会变**：同一个手柄在 XInput / DInput / Switch 模式下，按键编号和扳机轴都可能不同；
2.4G dongle、蓝牙、有线三种连接也可能不同。所以**以实际打印为准**，并把结果记下来给下一步接线用。

## 5. 黑武士4 Pro（Flydigi Vader 4 Pro）注意事项

- 2.4G dongle 一般会被识别成 Xbox 360 兼容手柄（`xpad` 驱动），名字可能是 `Microsoft X-Box 360 pad`。
  如果 `--list` 里看不到，先确认 dongle 插好、手柄开机、模式拨到 XInput（X 灯）。
- 如果名字是 `Flydigi ...` 或 `Gamepad`，本工具也能按能力识别。
- **背键 / C、Z 等额外键**：如果厂商软件把它们映射成键盘按键，会以 `KEY_*` 事件出现；如果完全没出现，
  说明当前模式没有把它们暴露给 Linux（换模式或换厂商软件配置）。
- 多个手柄同时插着时，`-d` 指定；程序会把候选列出来。

## 6. 本轮验证边界

- 已做：`cmake --build` 成功；`-Wall -Wextra -Wpedantic` 单独编译无 warning；`--list` / 无设备报错 /
  `-d 99` 路径正确；用 FIFO 喂**真实 evdev 字节流**做了端到端验证——`BTN_SOUTH` 按下/松开、
  `ABS_X` 值、`ABS_HAT0X/Y`、未知码 `0x2ff`、`--state` 组合状态、Ctrl+C 退出都正确。
- **未做**：本机没有 `/dev/input`、没有手柄，也没有 `/dev/uinput`，所以真实设备的
  `EVIOCGNAME/EVIOCGBIT/EVIOCGABS` 与 2.4G 实际映射**未验证**，需要在有手柄的小电脑上跑。
- 未接线：`rl_real_imgo2` 仍然是键盘控制；等映射确认后再做「按钮 → FSM 状态、摇杆 → 速度」的接入。

## 7. 下一步（映射确认后）

1. 把每个键的 `code` 固化成一张表（含背键/C/Z）。
2. 在 `rl_real_imgo2` 里加一个 `SDL`/`evdev` 输入线程或轮询器，把「状态键」和「速度轴」翻译成
   现有 `fsm.request(digit, ...)` / `fsm.velocity(key)` 的同一接口——这样 FSM、安全层、日志都不用改。
3. 死区、归一化、扳机阈值、摇杆到速度的斜率要单独标定；按键去抖和长按语义（站立/趴下）也要定。
