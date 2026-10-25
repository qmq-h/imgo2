/* ----------------------------------------------------------------------------
 * controller_demo —— 手柄/控制器输入核对工具（只读，不发任何控制指令）
 *
 * 用途：检测 Linux 下的 Xbox / 黑武士4 Pro（2.4G dongle）/ 其它 evdev 手柄，
 *       打印设备信息、内核暴露的全部按键与轴，以及每次按键/摇杆事件的
 *       「evdev 名称 + 编号 + 原始值」，用来核对每个按键到底对应什么。
 *
 * 用法：
 *   ./controller_demo              # 自动找手柄（按能力判断，不只看名字）
 *   ./controller_demo --list       # 列出 /dev/input/event*（不需要手柄）
 *   ./controller_demo -d 5         # 指定 event5
 *   ./controller_demo -d /dev/input/event5
 *   ./controller_demo -d vader     # 按设备名子串匹配
 *   ./controller_demo --events     # 每个轴事件单独一行（默认：最后一行原地刷新）
 *   ./controller_demo --raw        # 不打印中文说明，只打事件（便于脚本）
 *   ./controller_demo -h
 *
 * 权限：
 *   /dev/input/event* 一般属于 root:input，普通用户需要：
 *     sudo ./controller_demo
 *   或把用户加入 input 组（重新登录生效）：
 *     sudo usermod -aG input $USER
 *
 * 注意：
 *   - 只读打开，不 EVIOCGRAB、不抢占；Steam 等程序同时占用时两边都能收到。
 *   - 手柄的按键编号随模式（XInput / DInput / Switch）、连接方式（2.4G / 蓝牙 / 有线）
 *     和驱动（xpad / xpadneo / hid-generic）不同，**以下面实际打印的事件为准**。
 *   - 背键/宏如果在厂商软件里映射成键盘按键，会以 KEY_* 事件出现，本工具同样会打印。
 * ---------------------------------------------------------------------------- */
#include "apps/runtime.h"

#include <linux/input.h>
#include <sys/ioctl.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <dirent.h>
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>

#ifndef BTN_SOUTH
#define BTN_SOUTH 0x130
#define BTN_EAST  0x131
#define BTN_NORTH 0x133
#define BTN_WEST  0x134
#endif
#ifndef BTN_GAMEPAD
#define BTN_GAMEPAD BTN_SOUTH
#endif
#ifndef BTN_C
#define BTN_C 0x132
#endif
#ifndef BTN_Z
#define BTN_Z 0x135
#endif

namespace {

constexpr const char* kInputDir = "/dev/input";
constexpr int kEventStruct = static_cast<int>(sizeof(input_event));

struct Options {
    bool list = false;
    bool live = true;   // 摇杆值在最后一行原地刷新（非 TTY 自动关掉）
    bool raw = false;
    bool wait = false;  // 启动时没手柄就等，而不是直接退出
    bool help = false;
    std::string device; // 空 = 自动
};

struct Device {
    std::string path;
    std::string name;
    std::string phys;
    int index = -1;
    int fd = -1;
    input_id id{};
    std::array<unsigned char, (EV_MAX + 7) / 8 + 1> ev_bits{};
    std::array<unsigned char, (KEY_MAX + 7) / 8 + 1> key_bits{};
    std::array<unsigned char, (ABS_MAX + 7) / 8 + 1> abs_bits{};
    std::array<input_absinfo, ABS_MAX + 1> abs_info{};
};

bool bitTest(const unsigned char* bits, int code) {
    return (bits[code / 8] >> (code % 8)) & 1;
}

std::string toLower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

bool nameLooksLikeGamepad(const std::string& name) {
    const std::string n = toLower(name);
    for (const char* hint : {"xbox", "x-box", "gamepad", "controller", "joystick", "flydigi",
                             "vader", "apex", "8bitdo", "dualshock", "dualsense", "switch",
                             "pro controller", "hid"}) {
        if (n.find(hint) != std::string::npos) return true;
    }
    return false;
}

bool capsLookLikeGamepad(const Device& d) {
    const bool axes = bitTest(d.abs_bits.data(), ABS_X) || bitTest(d.abs_bits.data(), ABS_RX) ||
                      bitTest(d.abs_bits.data(), ABS_HAT0X);
    const bool buttons = bitTest(d.key_bits.data(), BTN_SOUTH) ||
                         bitTest(d.key_bits.data(), BTN_EAST) ||
                         bitTest(d.key_bits.data(), BTN_TRIGGER) ||
                         bitTest(d.key_bits.data(), BTN_JOYSTICK) ||
                         bitTest(d.key_bits.data(), BTN_TOP);
    return axes && buttons;
}

const char* keyName(int code) {
    switch (code) {
    case BTN_SOUTH: return "BTN_SOUTH(A)";
    case BTN_EAST: return "BTN_EAST(B)";
    case BTN_C: return "BTN_C";
    case BTN_NORTH: return "BTN_NORTH(X)";
    case BTN_WEST: return "BTN_WEST(Y)";
    case BTN_Z: return "BTN_Z";
    case BTN_TL: return "BTN_TL(LB)";
    case BTN_TR: return "BTN_TR(RB)";
    case BTN_TL2: return "BTN_TL2(LT)";
    case BTN_TR2: return "BTN_TR2(RT)";
    case BTN_SELECT: return "BTN_SELECT(Back)";
    case BTN_START: return "BTN_START(Start)";
    case BTN_MODE: return "BTN_MODE(Guide)";
    case BTN_THUMBL: return "BTN_THUMBL(L3)";
    case BTN_THUMBR: return "BTN_THUMBR(R3)";
    case BTN_TRIGGER: return "BTN_TRIGGER";
    case BTN_THUMB: return "BTN_THUMB";
    case BTN_THUMB2: return "BTN_THUMB2";
    case BTN_TOP: return "BTN_TOP";
    case BTN_TOP2: return "BTN_TOP2";
    case BTN_PINKIE: return "BTN_PINKIE";
    case BTN_BASE: return "BTN_BASE";
    case BTN_BASE2: return "BTN_BASE2";
    case BTN_BASE3: return "BTN_BASE3";
    case BTN_BASE4: return "BTN_BASE4";
    case BTN_BASE5: return "BTN_BASE5";
    case BTN_BASE6: return "BTN_BASE6";
    // 厂商软件把背键映射成键盘时会出现这些
    case KEY_ESC: return "KEY_ESC";
    case KEY_ENTER: return "KEY_ENTER";
    case KEY_SPACE: return "KEY_SPACE";
    case KEY_TAB: return "KEY_TAB";
    case KEY_BACKSPACE: return "KEY_BACKSPACE";
    case KEY_LEFTSHIFT: return "KEY_LEFTSHIFT";
    case KEY_LEFTCTRL: return "KEY_LEFTCTRL";
    case KEY_LEFTALT: return "KEY_LEFTALT";
    default: return nullptr;
    }
}

const char* absName(int code) {
    switch (code) {
    case ABS_X: return "ABS_X(左摇杆X)";
    case ABS_Y: return "ABS_Y(左摇杆Y)";
    case ABS_Z: return "ABS_Z(常见:LT/左扳机 或 右摇杆Z)";
    case ABS_RX: return "ABS_RX(右摇杆X)";
    case ABS_RY: return "ABS_RY(右摇杆Y)";
    case ABS_RZ: return "ABS_RZ(常见:RT/右扳机 或 右摇杆Z)";
    case ABS_THROTTLE: return "ABS_THROTTLE";
    case ABS_RUDDER: return "ABS_RUDDER";
    case ABS_WHEEL: return "ABS_WHEEL";
    case ABS_GAS: return "ABS_GAS(油门,常见:LT)";
    case ABS_BRAKE: return "ABS_BRAKE(刹车,常见:RT)";
    case ABS_HAT0X: return "ABS_HAT0X(十字键X)";
    case ABS_HAT0Y: return "ABS_HAT0Y(十字键Y)";
    case ABS_MISC: return "ABS_MISC";
    default: return nullptr;
    }
}

std::string hex(unsigned value, int width) {
    char buf[16];
    std::snprintf(buf, sizeof buf, "0x%0*x", width, value);
    return buf;
}

void usage(const char* argv0) {
    std::printf(
        "controller_demo —— 手柄输入核对（只读，不发控制指令）\n"
        "\n"
        "用法:\n"
        "  %s              自动找手柄\n"
        "  %s --list       列出 /dev/input/event*（不需要手柄）\n"
        "  %s -d 5         指定 event5\n"
        "  %s -d /dev/input/event5\n"
        "  %s -d vader     按设备名子串匹配\n"
        "  %s --events     每个轴事件都单独打一行（默认是最后一行原地刷新）\n"
        "  %s --raw        只打事件行（数字）\n"
        "  %s --wait       启动时没手柄就等待（插上自动接）\n"
        "\n"
        "权限: 普通用户可能读不了 /dev/input/event*，需要 sudo，或把用户加入 input 组。\n",
        argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0);
}

Options parse(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") opt.help = true;
        else if (arg == "--list") opt.list = true;
        else if (arg == "--events") opt.live = false;
        else if (arg == "--state") opt.live = true; // 兼容旧参数
        else if (arg == "--raw") opt.raw = true;
        else if (arg == "--wait") opt.wait = true;
        else if ((arg == "-d" || arg == "--device") && i + 1 < argc) opt.device = argv[++i];
        else if (!arg.empty() && arg[0] == '-') throw std::invalid_argument("未知参数: " + arg);
        else opt.device = arg; // 允许直接写设备
    }
    return opt;
}

bool openDevice(Device& d) {
    d.fd = ::open(d.path.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    if (d.fd < 0) return false;
    char name[256] = {};
    if (ioctl(d.fd, EVIOCGNAME(sizeof name - 1), name) >= 0) d.name = name;
    char phys[256] = {};
    if (ioctl(d.fd, EVIOCGPHYS(sizeof phys - 1), phys) >= 0) d.phys = phys;
    ioctl(d.fd, EVIOCGID, &d.id);
    ioctl(d.fd, EVIOCGBIT(0, d.ev_bits.size()), d.ev_bits.data());
    if (bitTest(d.ev_bits.data(), EV_KEY))
        ioctl(d.fd, EVIOCGBIT(EV_KEY, d.key_bits.size()), d.key_bits.data());
    if (bitTest(d.ev_bits.data(), EV_ABS))
        ioctl(d.fd, EVIOCGBIT(EV_ABS, d.abs_bits.size()), d.abs_bits.data());
    for (int code = 0; code <= ABS_MAX; ++code) {
        if (!bitTest(d.abs_bits.data(), code)) continue;
        ioctl(d.fd, EVIOCGABS(code), &d.abs_info[code]);
    }
    return true;
}

void closeDevice(Device& d) {
    if (d.fd >= 0) ::close(d.fd);
    d.fd = -1;
}

// 扫描 /dev/input/event*。打不开的记到 failed（权限/占用时给用户看）。
std::vector<Device> scan(std::vector<std::pair<std::string, int>>* failed) {
    std::vector<Device> devices;
    DIR* dir = opendir(kInputDir);
    if (!dir) {
        if (failed) failed->emplace_back(kInputDir, errno);
        return devices;
    }
    while (dirent* entry = readdir(dir)) {
        if (std::strncmp(entry->d_name, "event", 5) != 0) continue;
        Device d;
        d.path = std::string(kInputDir) + "/" + entry->d_name;
        d.index = std::atoi(entry->d_name + 5);
        if (!openDevice(d)) {
            if (failed) failed->emplace_back(d.path, errno);
            continue;
        }
        devices.push_back(std::move(d));
    }
    closedir(dir);
    std::sort(devices.begin(), devices.end(),
              [](const Device& a, const Device& b) { return a.index < b.index; });
    return devices;
}

void printCapabilities(const Device& d) {
    std::printf("  按键: ");
    bool any = false;
    for (int c = 0; c <= KEY_MAX; ++c) {
        if (!bitTest(d.key_bits.data(), c)) continue;
        const char* n = keyName(c);
        if (n) { std::printf("%s ", n); any = true; }
    }
    if (!any) std::printf("(无)");
    std::printf("\n  轴:   ");
    any = false;
    for (int c = 0; c <= ABS_MAX; ++c) {
        if (!bitTest(d.abs_bits.data(), c)) continue;
        const char* n = absName(c);
        const input_absinfo& ai = d.abs_info[c];
        if (n) std::printf("%s[%d,%d] ", n, ai.minimum, ai.maximum);
        else std::printf("ABS_0x%02x[%d,%d] ", c, ai.minimum, ai.maximum);
        any = true;
    }
    if (!any) std::printf("(无)");
    std::printf("\n");
}

void printDevice(const Device& d) {
    std::printf("设备: %s\n", d.path.c_str());
    std::printf("名称: %s\n", d.name.empty() ? "(未知)" : d.name.c_str());
    if (!d.phys.empty()) std::printf("物理: %s\n", d.phys.c_str());
    std::printf("ID:   bus=%s vendor=%s product=%s version=%s\n",
                hex(d.id.bustype, 4).c_str(), hex(d.id.vendor, 4).c_str(),
                hex(d.id.product, 4).c_str(), hex(d.id.version, 4).c_str());
    printCapabilities(d);
    std::printf("提示: 按 Ctrl+C 退出。下面每个事件就是“这个键到电脑里是什么”。\n");
    std::fflush(stdout);
}

void printLegend() {
    std::printf(
        "常见 Xbox 映射（仅作对照，实际以下面打印为准）:\n"
        "  A/B/X/Y   = BTN_SOUTH / BTN_EAST / BTN_NORTH / BTN_WEST\n"
        "  LB/RB     = BTN_TL / BTN_TR\n"
        "  LT/RT     = ABS_Z / ABS_RZ（部分驱动是 ABS_GAS / ABS_BRAKE）\n"
        "  Back/Start/Guide = BTN_SELECT / BTN_START / BTN_MODE\n"
        "  L3/R3     = BTN_THUMBL / BTN_THUMBR\n"
        "  左摇杆    = ABS_X / ABS_Y，右摇杆 = ABS_RX / ABS_RY\n"
        "  十字键    = ABS_HAT0X / ABS_HAT0Y\n"
        "  背键/宏   = 可能是 BTN_*，也可能被厂商软件映射成 KEY_*\n"
        "\n");
    std::fflush(stdout);
}

void printFailed(const std::vector<std::pair<std::string, int>>& failed) {
    if (failed.empty()) return;
    bool permission = false;
    for (const auto& item : failed) {
        std::printf("打不开: %s  (%s)\n", item.first.c_str(), std::strerror(item.second));
        if (item.second == EACCES || item.second == EPERM) permission = true;
        if (item.first == kInputDir && item.second == ENOENT)
            std::printf("  这台机器没有 /dev/input（不是目标主机），请在有手柄的机器上运行。\n");
    }
    if (permission) {
        std::printf("  权限不足，二选一：\n"
                    "    sudo ./controller_demo\n"
                    "    sudo usermod -aG input $USER   # 重新登录后生效\n");
    }
}

bool matchSelector(const Options& opt, const Device& d) {
    if (opt.device.empty()) return false;
    const std::string& want = opt.device;
    if (!want.empty() && want[0] == '/') return want == d.path;
    if (std::all_of(want.begin(), want.end(), [](unsigned char c) { return std::isdigit(c); }))
        return std::atoi(want.c_str()) == d.index;
    if (want.rfind("event", 0) == 0)
        return std::atoi(want.c_str() + 5) == d.index;
    return toLower(d.name).find(toLower(want)) != std::string::npos;
}

// 选一个设备：显式 -d 优先；否则按能力挑手柄，再按名字兜底。
bool selectDevice(const Options& opt, Device& out, go_app::StopSignals& signals,
                  bool wait_forever) {
    bool announced = false;
    for (;;) {
        // 显式给了路径就直接开，不要求它在 /dev/input 下（便于用 FIFO/自定义节点测试）。
        if (!opt.device.empty() && opt.device[0] == '/') {
            Device direct;
            direct.path = opt.device;
            if (openDevice(direct)) {
                out = std::move(direct);
                return true;
            }
            if (!wait_forever) {
                std::printf("打不开 %s: %s\n", opt.device.c_str(), std::strerror(errno));
                return false;
            }
        }
        std::vector<std::pair<std::string, int>> failed;
        std::vector<Device> devices = scan(&failed);
        int chosen = -1;
        if (!opt.device.empty() && opt.device[0] != '/') {
            for (size_t i = 0; i < devices.size(); ++i)
                if (matchSelector(opt, devices[i])) { chosen = static_cast<int>(i); break; }
            if (chosen < 0 && !devices.empty()) {
                std::printf("没有匹配 '%s' 的设备。当前可用:\n", opt.device.c_str());
                for (const Device& d : devices)
                    std::printf("  %s  %s\n", d.path.c_str(), d.name.c_str());
            }
        } else {
            std::vector<int> by_caps;
            std::vector<int> by_name;
            for (size_t i = 0; i < devices.size(); ++i) {
                if (capsLookLikeGamepad(devices[i])) by_caps.push_back(static_cast<int>(i));
                if (nameLooksLikeGamepad(devices[i].name)) by_name.push_back(static_cast<int>(i));
            }
            const std::vector<int>& pool = !by_caps.empty() ? by_caps : by_name;
            if (pool.size() == 1) chosen = pool[0];
            else if (pool.size() > 1) {
                std::printf("找到多个像手柄的设备，请用 -d 指定:\n");
                for (int i : pool)
                    std::printf("  -d %d   %s  %s\n", devices[i].index,
                                devices[i].path.c_str(), devices[i].name.c_str());
            }
        }
        if (chosen >= 0) {
            out = std::move(devices[chosen]);
            for (size_t i = 0; i < devices.size(); ++i)
                if (static_cast<int>(i) != chosen) closeDevice(devices[i]);
            return true;
        }
        for (Device& d : devices) closeDevice(d);
        if (!announced && devices.empty() && !failed.empty()) printFailed(failed);
        if (!wait_forever) {
            if (devices.empty() && failed.empty())
                std::printf("没有找到 %s/event*。\n", kInputDir);
            std::printf("可用 --list 查看，或用 -d 指定设备。\n");
            return false;
        }
        if (signals.requested()) return false;
        if (!announced) {
            std::printf("等待手柄连接（Ctrl+C 退出）...\n");
            std::fflush(stdout);
            announced = true;
        }
        for (int i = 0; i < 10 && !signals.requested(); ++i) usleep(100 * 1000);
    }
}

struct LiveState {
    std::array<bool, KEY_MAX + 1> pressed{};
    std::array<int, ABS_MAX + 1> axis{};
    std::array<int, ABS_MAX + 1> observed_min{};
    std::array<int, ABS_MAX + 1> observed_max{};
    std::array<bool, ABS_MAX + 1> observed{};
    bool changed = false;
};

void renderLive(const Device& d, const LiveState& s) {
    const auto raw = [&](int code) { return code <= ABS_MAX ? s.axis[code] : 0; };
    const auto norm = [&](int code) {
        const input_absinfo& ai = d.abs_info[code];
        const int span = ai.maximum - ai.minimum;
        return span > 0 ? double(raw(code) - ai.minimum) / double(span) : 0.0;
    };
    const auto stick = [&](int code) { return 2.0 * norm(code) - 1.0; };
    // 扳机轴：xpad 常见 ABS_Z/ABS_RZ，部分驱动是 ABS_GAS/ABS_BRAKE。
    const int lt = bitTest(d.abs_bits.data(), ABS_Z) ? ABS_Z : ABS_GAS;
    const int rt = bitTest(d.abs_bits.data(), ABS_RZ) ? ABS_RZ : ABS_BRAKE;

    std::printf("\rLX%+.2f LY%+.2f RX%+.2f RY%+.2f  LT%.2f RT%.2f  D%+d,%+d  ",
                stick(ABS_X), stick(ABS_Y), stick(ABS_RX), stick(ABS_RY),
                norm(lt), norm(rt), raw(ABS_HAT0X), raw(ABS_HAT0Y));
    const std::pair<int, const char*> keys[] = {
        {BTN_SOUTH, "A"}, {BTN_EAST, "B"}, {BTN_NORTH, "X"}, {BTN_WEST, "Y"},
        {BTN_TL, "LB"}, {BTN_TR, "RB"}, {BTN_SELECT, "Back"}, {BTN_START, "Start"},
        {BTN_MODE, "Guide"}, {BTN_THUMBL, "L3"}, {BTN_THUMBR, "R3"},
        // 有些手柄把扳机做成数字键而不是 ABS_Z/RZ 轴
        {BTN_TL2, "LTd"}, {BTN_TR2, "RTd"},
        // 背键/额外键常见映射
        {BTN_C, "C"}, {BTN_Z, "Z"}};
    std::array<bool, KEY_MAX + 1> shown{};
    for (const auto& key : keys) {
        if (key.first <= KEY_MAX) {
            shown[key.first] = true;
            if (s.pressed[key.first]) std::printf("[%s]", key.second);
        }
    }
    // 其它 BTN_*（背键/厂商自定义）直接打编号，便于识别
    for (int c = BTN_MISC; c <= BTN_GEAR_UP && c <= KEY_MAX; ++c)
        if (!shown[c] && s.pressed[c]) std::printf("[0x%03x]", c);
    std::printf("\x1b[K"); // 清掉本行剩余字符，避免旧值残留
    std::fflush(stdout);
}

void printKeyEvent(const input_event& e, double seconds, bool raw) {
    const char* n = keyName(e.code);
    const char* state = e.value == 0 ? "松开" : e.value == 1 ? "按下" : "重复";
    if (raw)
        std::printf("%8.3f KEY 0x%03x %d\n", seconds, e.code, e.value);
    else if (n)
        std::printf("%8.3f  KEY  %-22s code=0x%03x value=%d  [%s]\n",
                    seconds, n, e.code, e.value, state);
    else
        std::printf("%8.3f  KEY  %-22s code=0x%03x value=%d  [%s]\n",
                    seconds, "(未命名)", e.code, e.value, state);
}

void printAbsEvent(const input_absinfo& ai, const input_event& e, double seconds,
                    bool raw, bool out_of_range) {
    const int span = ai.maximum - ai.minimum;
    const double norm = span > 0 ? double(e.value - ai.minimum) / double(span) : 0.0;
    const bool stick = e.code == ABS_X || e.code == ABS_Y || e.code == ABS_RX || e.code == ABS_RY;
    const char* n = absName(e.code);
    const char* flag = out_of_range ? "  [超出声明范围!]" : "";
    if (raw) {
        std::printf("%8.3f ABS 0x%02x %d%s\n", seconds, e.code, e.value, flag);
    } else if (stick && span > 0) {
        // 摇杆额外给一个以中点为 0 的 -1..+1，便于直接映射到速度
        std::printf("%8.3f  ABS  %-26s code=0x%02x value=%6d  range=[%d,%d] norm=%+.3f signed=%+.3f%s\n",
                    seconds, n ? n : "(未命名)", e.code, e.value, ai.minimum, ai.maximum,
                    norm, 2.0 * norm - 1.0, flag);
    } else {
        std::printf("%8.3f  ABS  %-26s code=0x%02x value=%6d  range=[%d,%d] norm=%+.3f%s\n",
                    seconds, n ? n : "(未命名)", e.code, e.value, ai.minimum, ai.maximum, norm, flag);
    }
}

void printObserved(const LiveState& s) {
    bool any = false;
    for (int c = 0; c <= ABS_MAX; ++c) any = any || s.observed[c];
    if (!any) return;
    std::printf("\n本次观测到的轴原始范围（对照声明范围）:\n");
    for (int c = 0; c <= ABS_MAX; ++c) {
        if (!s.observed[c]) continue;
        std::printf("  %-26s 观测 [%d, %d]%s\n", absName(c) ? absName(c) : "(未命名)",
                    s.observed_min[c], s.observed_max[c],
                    (s.observed_min[c] < -32768 || s.observed_max[c] > 32767)
                        ? "   <-- 超出 int16 范围，任何 int16_t 都会截断" : "");
    }
    std::fflush(stdout);
}

void runLoop(Device& d, const Options& opt, go_app::StopSignals& signals) {
    const auto start = std::chrono::steady_clock::now();
    auto last_state = start;
    bool live_line = false;
    LiveState state;
    std::array<unsigned char, kEventStruct * 64> buffer{};
    if (opt.live) { // 先画一行，用户一进来就能看到摇杆读数
        renderLive(d, state);
        live_line = true;
    }
    for (;;) {
        if (signals.requested()) break;
        pollfd pfd{d.fd, POLLIN, 0};
        const int ready = ::poll(&pfd, 1, 200);
        if (ready < 0) {
            if (errno == EINTR) continue;
            std::printf("poll 失败: %s\n", std::strerror(errno));
            break;
        }
        if (ready == 0) continue;
        if (pfd.revents & (POLLHUP | POLLERR)) {
            std::printf("设备已断开。\n");
            break;
        }
        const ssize_t bytes = ::read(d.fd, buffer.data(), buffer.size());
        if (bytes < 0) {
            if (errno == EAGAIN || errno == EINTR) continue;
            std::printf("读设备失败: %s\n", std::strerror(errno));
            break;
        }
        if (bytes == 0) {
            std::printf("设备已断开（EOF）。\n");
            break;
        }
        const int count = static_cast<int>(bytes) / kEventStruct;
        for (int i = 0; i < count; ++i) {
            input_event e{};
            std::memcpy(&e, buffer.data() + i * kEventStruct, sizeof e);
            const auto now = std::chrono::steady_clock::now();
            const double seconds = std::chrono::duration<double>(now - start).count();
            if (e.type == EV_KEY) {
                if (opt.live && live_line) {
                    std::printf("\r\x1b[K"); // 先把状态行清掉，按键另起一行
                    live_line = false;
                }
                printKeyEvent(e, seconds, opt.raw);
                if (e.code <= KEY_MAX) state.pressed[e.code] = e.value != 0;
                state.changed = true;
            } else if (e.type == EV_ABS) {
                const bool known = e.code <= ABS_MAX;
                const input_absinfo ai = known ? d.abs_info[e.code] : input_absinfo{};
                const bool out_of_range = e.value < ai.minimum || e.value > ai.maximum;
                if (known) {
                    state.axis[e.code] = e.value;
                    if (!state.observed[e.code]) {
                        state.observed[e.code] = true;
                        state.observed_min[e.code] = e.value;
                        state.observed_max[e.code] = e.value;
                    } else {
                        state.observed_min[e.code] = std::min(state.observed_min[e.code], e.value);
                        state.observed_max[e.code] = std::max(state.observed_max[e.code], e.value);
                    }
                }
                if (!opt.live) printAbsEvent(ai, e, seconds, opt.raw, out_of_range);
                state.changed = true;
            } else if (e.type == EV_SYN && e.code == SYN_REPORT && opt.live) {
                const double since = std::chrono::duration<double>(now - last_state).count();
                if (state.changed && since >= 0.1) { // 10 Hz 刷新，够看又不闪
                    renderLive(d, state);
                    live_line = true;
                    last_state = now;
                    state.changed = false;
                }
            }
            std::fflush(stdout);
        }
    }
    if (opt.live && live_line) std::printf("\r\x1b[K");
    printObserved(state);
}

} // namespace

int main(int argc, char** argv) try {
    const Options opt = parse(argc, argv);
    if (opt.help) {
        usage(argv[0]);
        return 0;
    }
    if (opt.list) {
        std::vector<std::pair<std::string, int>> failed;
        std::vector<Device> devices = scan(&failed);
        std::printf("共找到 %zu 个可打开的 /dev/input/event*:\n", devices.size());
        for (const Device& d : devices) {
            std::printf("\n%s  %s%s%s\n", d.path.c_str(), d.name.c_str(),
                        capsLookLikeGamepad(d) ? "   <-- 像手柄" : "",
                        nameLooksLikeGamepad(d.name) ? " (名字像手柄)" : "");
            printCapabilities(d);
        }
        printFailed(failed);
        for (Device& d : devices) closeDevice(d);
        return 0;
    }

    bool live = opt.live;
    if (live && !isatty(STDOUT_FILENO)) live = false; // 重定向/管道时退回逐行
    Options opts = opt;
    opts.live = live;

    // --wait 也等不到“没有 /dev/input 的机器”；显式给了路径则按路径等。
    if (opts.wait && (opts.device.empty() || opts.device[0] != '/') &&
        access(kInputDir, F_OK) != 0) {
        std::printf("%s 不存在，--wait 不会等到手柄；请在有手柄的机器上运行。\n", kInputDir);
        return 1;
    }

    go_app::StopSignals signals;
    Device device;
    if (!selectDevice(opts, device, signals, opts.wait))
        return signals.requested() ? 0 : 1; // Ctrl+C 中断等待不算失败
    if (!opts.raw) printLegend();
    printDevice(device);
    if (opts.live) std::printf("摇杆值在最后一行原地刷新；按键会另起一行打印。\n");
    for (;;) {
        runLoop(device, opts, signals);
        closeDevice(device);
        if (signals.requested()) break;
        usleep(300 * 1000); // 设备还在但读失败时，避免重连空转
        if (!selectDevice(opts, device, signals, true)) break;
        std::printf("\n重新连接：\n");
        printDevice(device);
    }
    std::printf("退出。\n");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "controller_demo: %s\n", e.what());
    return 1;
}
