# 串口配置改动（系统侧）

本文件只覆盖系统侧的改动（`/etc/...`）。工程代码里的改动（`config/real_config.cpp` 的设备路径）不在内。

一共 3 个文件：

| 文件 | 作用 |
|---|---|
| `/etc/modprobe.d/blacklist-cdc-acm.conf` | 挡住 `cdc_acm` 自动加载 |
| `/etc/modules` | 开机按固定顺序加载 `ch343`、`cdc_acm` |
| `/etc/udev/rules.d/99-robot-serial.rules` | 建 5 个固定软链接 + 挡住 ModemManager |

---

## 1. `/etc/modprobe.d/blacklist-cdc-acm.conf`

### 命令

```bash
sudo tee /etc/modprobe.d/blacklist-cdc-acm.conf >/dev/null <<'EOF'
blacklist cdc_acm
EOF
```

**这条命令做什么**：把 `blacklist cdc_acm` 写进该文件。
`sudo tee <文件>` 是以 root 身份把内容写进文件；`>/dev/null` 只是不让内容再回显到屏幕上，不起别的作用。

### 为什么

CH343 和 IMU 都能被 `cdc_acm` 这个驱动绑定，而 `modules.alias` 里 `cdc_acm` 排在 `ch343` 前面，所以开机时它会被抢先自动加载，把 4 个 485 口变成 `/dev/ttyACM*`。

`blacklist` 的作用是让 udev/kmod 在**按别名自动加载**时忽略掉 `cdc_acm`。

注意：它**不影响** `modprobe cdc_acm` 按名字显式加载 —— 这正是下面第 2 步还要显式加载它的原因。

### 验证

```bash
cat /etc/modprobe.d/blacklist-cdc-acm.conf
```

---

## 2. `/etc/modules`

### 命令

```bash
sudo tee /etc/modules >/dev/null <<'EOF'
ch343
cdc_acm
EOF
```

**这条命令做什么**：把这个文件重写成两行。开机时 `systemd-modules-load.service` 会按文件里的**行序**依次加载这两个模块。

> 文件里以 `#` 开头的注释行可以保留，不影响功能。

### 为什么

第 1 步把 `cdc_acm` 屏蔽掉之后，它就不会自动加载了，但 **IMU 需要它**，所以必须在这里显式加载一次。

**顺序不能反**：内核给设备绑驱动时是"先注册的先占"，`ch343` 写在前面，才能先占住 4 个 485 口；`cdc_acm` 写在后面，只能接管剩下没被占用的设备，也就是 IMU。

### 验证

```bash
grep -n . /etc/modules        # 结尾必须是 ch343 在 cdc_acm 前面
```

> ⚠️ 以后重装 ch343 驱动（在驱动源码目录跑 `sudo make install`）时，那个 Makefile 会自动往这个文件**末尾追加一行 `ch343`**，把顺序弄反。装完要跑上面这条命令检查。

---

## 3. `/etc/udev/rules.d/99-robot-serial.rules`

### 命令一：创建规则文件

```bash
sudo tee /etc/udev/rules.d/99-robot-serial.rules >/dev/null <<'EOF'
# CH9114 四路 485 盒子（序列号 BC4E6C65ABCDD480）：按接口号固定成腿名
SUBSYSTEM=="tty", KERNEL=="ttyCH343USB*", \
  ENV{ID_SERIAL_SHORT}=="BC4E6C65ABCDD480", ENV{ID_USB_INTERFACE_NUM}=="00", \
  SYMLINK+="robot/leg_bl", MODE="0666", ENV{ID_MM_DEVICE_IGNORE}="1"

SUBSYSTEM=="tty", KERNEL=="ttyCH343USB*", \
  ENV{ID_SERIAL_SHORT}=="BC4E6C65ABCDD480", ENV{ID_USB_INTERFACE_NUM}=="02", \
  SYMLINK+="robot/leg_fl", MODE="0666", ENV{ID_MM_DEVICE_IGNORE}="1"

SUBSYSTEM=="tty", KERNEL=="ttyCH343USB*", \
  ENV{ID_SERIAL_SHORT}=="BC4E6C65ABCDD480", ENV{ID_USB_INTERFACE_NUM}=="04", \
  SYMLINK+="robot/leg_fr", MODE="0666", ENV{ID_MM_DEVICE_IGNORE}="1"

SUBSYSTEM=="tty", KERNEL=="ttyCH343USB*", \
  ENV{ID_SERIAL_SHORT}=="BC4E6C65ABCDD480", ENV{ID_USB_INTERFACE_NUM}=="06", \
  SYMLINK+="robot/leg_br", MODE="0666", ENV{ID_MM_DEVICE_IGNORE}="1"

# IMU（ZJZ CMSIS-DAP，序列号 201408080001）：这块板有两个 CDC-ACM 口，
# 按接口号钉死真正出数据的那个（01）
SUBSYSTEM=="tty", KERNEL=="ttyACM*", \
  ENV{ID_SERIAL_SHORT}=="201408080001", ENV{ID_USB_INTERFACE_NUM}=="01", \
  SYMLINK+="robot/imu", MODE="0666", \
  ENV{ID_MM_DEVICE_IGNORE}="1", ENV{ID_MM_PORT_IGNORE}="1"
EOF
```

**这条命令做什么**：把规则文件写进 `/etc/udev/rules.d/`。

规则里每个字段的意思：

| 字段 | 作用 |
|---|---|
| `SUBSYSTEM=="tty"` | 只对 tty 类设备生效 |
| `KERNEL=="ttyCH343USB*"` | 只对设备名是 `ttyCH343USB` 开头的生效 |
| `ENV{ID_SERIAL_SHORT}=="..."` | 匹配**盒子序列号**，用来区分是哪一颗盒子 |
| `ENV{ID_USB_INTERFACE_NUM}=="00"` | 匹配**接口号**，即盒子上的哪一个口 |
| `SYMLINK+="robot/leg_bl"` | 建软链接 `/dev/robot/leg_bl`（`robot/` 目录会自动创建） |
| `MODE="0666"` | 把设备权限设为所有用户可读写 |
| `ENV{ID_MM_DEVICE_IGNORE}="1"` | 告诉 ModemManager 不要碰这个口 |

**为什么用 `ENV{}` 而不是 `ATTRS{}`**：`man 7 udev` 规定，同一条规则里写多个 `ATTRS{}` 时，它们**必须匹配同一个设备**。而在这台机器上 `idVendor`/`serial` 在 USB 设备层（`3-7`），`bInterfaceNumber` 在接口层（`3-7:1.0`），分属两层，写在一起**永远不会匹配，而且不报错**。`ENV{}` 环境变量没有这个限制。

### 命令二：让规则生效

```bash
sudo udevadm control --reload-rules
```

**这条命令做什么**：让 udevd 重新读取 `/etc/udev/rules.d/` 下所有规则文件。
不做这一步，新写的规则不会被加载。

```bash
sudo udevadm trigger --subsystem-match=tty --action=add
```

**这条命令做什么**：对**已经插着的** tty 设备重新"播放"一遍 add 事件，让规则立刻执行一次。
不做这一步，就得拔插设备或重启才能建出软链接。
`--subsystem-match=tty` 表示只处理 tty 类设备，`--action=add` 表示按"设备插入"来处理。

### 为什么需要这个文件

1. **拿不到持久名**：系统自带的 `/lib/udev/rules.d/60-serial.rules` 第 11 行是
   `KERNEL!="ttyUSB[0-9]*|ttyACM[0-9]*", GOTO="serial_end"`
   —— 它只给 `ttyUSB*` 和 `ttyACM*` 建 `by-id`/`by-path` 持久名，`ttyCH343USB*` 不在名单里，所以只能自己写规则建。

2. **ModemManager 会来捣乱**：`udevadm test` 显示这些口带着 `ID_MM_CANDIDATE=1`，即 ModemManager 会把**电机总线口**当成"可能插了调制解调器"，自己打开串口发 AT 指令探测，这会往 485 总线上注入垃圾帧、和电机通信撞车。加 `ID_MM_DEVICE_IGNORE=1` 让它彻底不碰。

3. **用固定名字代替会变的编号**：`ttyCH343USB` 后面的数字是驱动分配顺序，`/dev/robot/*` 是按芯片物理接口固定的。用后者，拔插、换 USB 口都不会把腿搞反。

### 软链接对应关系

| 软链接 | 指向 | 接口号 | 腿 |
|---|---|---|---|
| `/dev/robot/leg_fl` | `ttyCH343USB1` | 02 | 前左 |
| `/dev/robot/leg_fr` | `ttyCH343USB2` | 04 | 前右 |
| `/dev/robot/leg_bl` | `ttyCH343USB0` | 00 | 后左 |
| `/dev/robot/leg_br` | `ttyCH343USB3` | 06 | 后右 |
| `/dev/robot/imu` | `ttyACM0` | 01 | IMU |

### 验证

```bash
ls -l /dev/robot/
```

**这条命令做什么**：列出建出来的 5 个软链接。看不到就是规则没匹配上。

```bash
sudo udevadm test /sys/class/tty/ttyCH343USB0 2>&1 | grep -iE 'leg_|LINK'
```

**这条命令做什么**：模拟 udev 处理这块设备并打印过程，用来查"规则到底匹配没有"。匹配成功会打印类似：

```
ttyCH343USB0: /etc/udev/rules.d/99-robot-serial.rules:23 LINK 'robot/leg_bl'
DEVLINKS=/dev/robot/leg_bl
```

没有 `LINK` / `DEVLINKS` 行就是没匹配上。

---

## 全部改动重做一遍

新机器上按顺序跑，第 3 步的完整内容请直接复制上面第 3 节「命令一」那条 `sudo tee`：

```bash
# 1. 挡住 cdc_acm 的自动加载（内容见第 1 节）
sudo tee /etc/modprobe.d/blacklist-cdc-acm.conf >/dev/null <<'EOF'
blacklist cdc_acm
EOF

# 2. 开机按顺序显式加载两个模块（内容见第 2 节）
sudo tee /etc/modules >/dev/null <<'EOF'
ch343
cdc_acm
EOF

# 3. 写 udev 规则建软链接 —— 直接复制第 3 节「命令一」的完整 sudo tee，不要手打

# 4. 重新加载规则并立刻生效（见第 3 节「命令二」）
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=tty --action=add

# 5. 验证
ls -l /dev/robot/
grep -n . /etc/modules
```
