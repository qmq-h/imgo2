/* ----------------------------------------------------------------------------
 * 用法
 *   ./single_monitor [串口] [ID] [-t ms] [-b baud]
 *   直接运行 = 用下面 [配置] 区的 kDevice / kMotorId（默认 /dev/robot/leg_fl, ID 0）。
 *   每按一次 1 发一条全零 FOC 指令（卸力），打印：
 *     t_send / 写完→首字节 / 首字节→收完 / 发完合计 / 按键→结束 / 反馈。
 *   换串口或 ID：改 [配置] 区，然后重新编译即可。
 * ---------------------------------------------------------------------------- */
// single_monitor：单线程"按一次键 → 发一条全零(FOC 卸力)指令 → 等应答 → 报耗时"。
//
// 用法：直接运行 ./single_monitor
//       要换串口/电机，改下面 [配置] 区里的常量后重新编译（也支持命令行参数临时覆盖）。
//
// 说明：
//   * 只注册**一台**电机，所以 GoBus::cycle() 每轮正好发一条命令、收一条应答；
//   * 从不提交目标 ⇒ 发出的永远是全零指令（FOC、kp=kd=tau=0，即卸力，不会出力）；
//   * 不用线程：阻塞等键 → 一次事务 → 打印 → 再等键；Ctrl+C 或 EOF 退出；
//   * 打印三段耗时（flush/write/reply）——与部署程序日志里的 b*_* 列同一套计时。
#include "go_bus.h"

#include <chrono>
#include <filesystem>
#include <cstdio>
#include <cstring>
#include <poll.h>
#include <string>
#include <vector>
#include <unistd.h>

// ===================== [配置] 改这里 =====================
namespace config {
constexpr const char* kDevice    = "/dev/robot/leg_fr"; // 串口设备（leg_fl/leg_fr/leg_bl/leg_br）
constexpr int         kMotorId   = 1;                  // 该总线上的电机 ID：0/1/2
constexpr int         kTimeoutMs = 20;                 // 单事务等待预算(ms)
constexpr unsigned    kBaud      = 4000000;            // 波特率
}
// ========================================================

namespace {
double ms(std::chrono::steady_clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
}
// 列出常见的串口候选，方便直接复制
void listPorts() {
    std::vector<std::string> found;
    std::error_code ec;
    if (std::filesystem::is_directory("/dev/robot", ec)) {
        for (const auto& e : std::filesystem::directory_iterator("/dev/robot", ec)) {
            if (ec) break;
            found.push_back(e.path().string());
        }
    }
    ec.clear();
    if (std::filesystem::is_directory("/dev", ec)) {
        for (const auto& e : std::filesystem::directory_iterator("/dev", ec)) {
            if (ec) break;
            const std::string name = e.path().filename().string();
            if (name.rfind("ttyCH", 0) == 0 || name.rfind("ttyUSB", 0) == 0 || name.rfind("ttyACM", 0) == 0)
                found.push_back(e.path().string());
        }
    }
    if (found.empty()) { std::printf("  （没找到 /dev/robot/* 或 /dev/tty{CH,USB,ACM}*）\n"); return; }
    std::printf("候选串口：\n");
    for (const auto& p : found) std::printf("  %s\n", p.c_str());
    std::printf("例：%s %s 0\n", "./single_monitor", found.front().c_str());
}
void usage(const char* argv0) {
    std::printf(
        "用法: %s            （直接运行，串口/ID 读源码顶部 [配置] 区）\n"
        "      %s [串口] [ID] [-t ms] [-b baud]   （命令行临时覆盖，可选）\n"
        "  每按一次 1（无需回车）发送一条全零 FOC 指令并等待应答，随即打印：\n"
        "    t_send         发送时刻（相对程序启动，µs 精度）\n"
        "    写完→首字节    命令写进驱动后，到收到第一个字节的间隔\n"
        "    首字节→收完    从第一个字节到凑齐/解析出该 ID 的合法帧\n"
        "    发完合计       写完命令到收完（= 首字节 + 首字节→收完）\n"
        "    按键→结束      从按 1 到打印的总耗时（含编码/清缓冲/写）\n"
        "    反馈            q/dq/tau/mode/err/temp（全零指令下电机应进入 FOC 待机）\n"
        "  Ctrl+C 或输入结束退出。该工具只发全零指令，不给电机出力。\n"
        "  设备名就是部署用的那四个：/dev/robot/leg_fl、leg_fr、leg_bl、leg_br（udev 规则见 docs/xiaoji/udev.md）。\n",
        argv0, argv0);
}
} // namespace

int main(int argc, char** argv) try {
    std::string device = config::kDevice;
    int id = config::kMotorId;
    int timeout_ms = config::kTimeoutMs;
    unsigned baud = config::kBaud;
    // 命令行只做"临时覆盖"：第一个非选项参数是串口，第二个是电机 ID；不带任何参数就用 config 里的值
    int argi = 1;
    if (argi < argc && argv[argi][0] != '-') device = argv[argi++];
    if (argi < argc && argv[argi][0] != '-') id = std::atoi(argv[argi++]);
    for (; argi < argc; ++argi) {
        const std::string a = argv[argi];
        if (a == "--help" || a == "-h") { usage(argv[0]); listPorts(); return 0; }
        else if (a == "-t" && argi + 1 < argc) timeout_ms = std::atoi(argv[++argi]);
        else if (a == "-b" && argi + 1 < argc) baud = static_cast<unsigned>(std::strtoul(argv[++argi], nullptr, 10));
        else { std::fprintf(stderr, "无法识别的参数: %s\n", a.c_str()); return 1; }
    }
    if (id < 0 || id > 14) { std::fprintf(stderr, "电机 ID 必须在 0..14\n"); return 1; }
    if (timeout_ms <= 0) { std::fprintf(stderr, "超时必须是正数(ms)\n"); return 1; }

    go::GoBus bus(device, baud, static_cast<size_t>(timeout_ms) * 1000, 100);
    go::GoMotor& motor = bus.addMotor(id, 0.0f, 1.0f);
    if (!bus.open()) {
        std::fprintf(stderr, "打开 %s 失败: %s\n", device.c_str(), bus.lastError().c_str());
        std::fprintf(stderr, "请改源码顶部 [配置] 区的 config::kDevice（或命令行传第一个参数）。\n");
        listPorts();
        return 1;
    }
    std::printf("single_monitor: %s  id=%d  波特率=%u  单事务预算=%dms\n", device.c_str(), id, baud, timeout_ms);
    std::printf("每按一次 1 发一条全零指令（卸力），Ctrl+C 退出。\n");
    std::fflush(stdout);

    const auto t_start = std::chrono::steady_clock::now();
    unsigned long long press = 0;
    for (;;) {
        struct pollfd p {STDIN_FILENO, POLLIN, 0};
        const int ready = ::poll(&p, 1, -1);
        if (ready < 0) { if (errno == EINTR) continue; std::perror("poll"); return 1; }
        char key = 0;
        const ssize_t got = ::read(STDIN_FILENO, &key, 1);
        if (got <= 0) break;                       // EOF：退出
        if (key != '1') continue;                  // 只认 1

        const auto before = bus.transportStats();
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = bus.cycle();
        const auto t1 = std::chrono::steady_clock::now();
        const auto after = bus.transportStats();
        ++press;

        const auto flush = bus.roundFlushUs();
        const auto write = bus.roundWriteUs();
        const auto reply = bus.roundReplyUs();
        const auto first = bus.roundFirstUs();
        const double since_start_us = ms(t1 - t_start) * 1000.0;
        const double t_send_s = (since_start_us - double(reply)) / 1e6;   // 写完命令的时刻
        std::printf("[%llu] t_send=+%.6f s | 写完→首字节 %7u us | 首字节→收完 %6u us | 发完合计 %7u us | 按键→结束 %8.0f us"
                    " | ok=%d 收 %zuB 坏 %zu 串 %zu\n",
                    press, t_send_s, first,
                    reply >= first ? reply - first : 0u, reply,
                    ms(t1 - t0) * 1000.0, ok ? 1 : 0,
                    after.bytesIn - before.bytesIn, after.badFrames - before.badFrames,
                    after.strayFrames - before.strayFrames);
        const auto& fb = motor.feedback();
        if (fb.valid)
            std::printf("    反馈 q=%+8.5f dq=%+8.4f tau=%+7.4f mode=%u err=%u temp=%d\n",
                        double(fb.q), double(fb.dq), double(fb.tau),
                        unsigned(fb.mode), unsigned(fb.error), int(fb.temp));
        else
            std::printf("    无有效反馈: %s\n", motor.lastError().c_str());
        if (flush || write)
            std::printf("    （发送前: 清缓冲 %u us + 写命令 %u us）\n", flush, write);
        if (!ok && !bus.lastError().empty())
            std::printf("    总线: %s\n", bus.lastError().c_str());
        // 故障会被锁存；能恢复就恢复，恢复不了也不影响"一次一条"的语义
        if (!ok && bus.clearFault()) std::printf("    （已清除锁存故障）\n");
        std::fflush(stdout);
    }
    std::printf("single_monitor: 退出（共 %llu 次）\n", press);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "single_monitor error: %s\n", e.what());
    return 1;
}
