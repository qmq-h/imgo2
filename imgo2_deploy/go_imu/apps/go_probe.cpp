// ============================================================================
//  go_probe.cpp —— CH343/CH9114 串口自检工具
//
//  作用：对每个串口、每个电机 ID 发送一条 FOC 全零控制量，找出哪些 ID 真的
//        应答，并统计线路噪声；最后打印可直接粘贴进 config/real_config.cpp 的配置。
//
//  注意：全零控制量会让电机卸力（失去保持力矩），运行前请确保机械悬空或有人扶住。
//        本工具只发零指令，不发任何运动指令。
// ============================================================================
#include "real_config.h"
#include "go_protocol.h"

#include <cctype>
#include <algorithm>
#include <fstream>
#include "serial_port.h"
#include "motor_transport.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace
{

const uint32_t kBaud = go::makeDefaultConfig().baud;
constexpr int kMinId = 0;
constexpr int kMaxId = 14;
constexpr size_t kRxBufSize = 512;

struct MotorHit
{
    int id = -1;
    go::Feedback feedback;
    double rtt_ms = 0.0;
};

struct PortReport
{
    std::string device;
    bool opened = false;
    std::string openError;
    uint32_t baud = 0;
    std::vector<MotorHit> hits;
    std::vector<int> missed;
    size_t bytesIn = 0;
    size_t badFrames = 0;
    size_t strayFrames = 0;
};

void usage(const char *argv0)
{
    std::printf(
        "go_probe —— CH343/CH9114 串口自检：找出每个口上应答的电机\n"
        "\n"
        "用法:\n"
        "  %s [选项] [设备...]\n"
        "\n"
        "选项:\n"
        "  -t <毫秒>  每次请求的等待上限（默认 40）\n"
        "  -n <次数>  每个 ID 的尝试次数（默认 2）\n"
        "  -i <列表>  每路一行汇总里要显示的 ID，逗号分隔（默认 0,1,2）\n"
        "  -h         显示本帮助\n"
        "\n"
        "默认设备与波特率: config/real_config.cpp\n"
        "默认扫描 ID %d..%d，每条命令都是 FOC 全零控制量（4 Mbps）。\n"
        "\n"
        "警告: 全零命令会让电机卸力、失去保持力矩，运行前请确保机械悬空或有人扶住。\n",
        argv0, kMinId, kMaxId);
}

// 在 end 之前读取并匹配 id 的反馈。返回 true 表示收到该 ID 的合法反馈。
// bytesIn / badFrames / strayFrames 用于观察线路质量：
//   badFrames   = 找到反馈帧头但 CRC/格式非法（线被干扰的典型症状）
//   strayFrames = 合法帧但 ID 不是本次请求的（ID 配错或总线串扰）
bool awaitReply(go::SerialPort& port, int id, go::SerialPort::Deadline end,
                go::Feedback& out, size_t& bytesIn, size_t& badFrames,
                size_t& strayFrames, std::string& error) {
    go::TransportStats stats{bytesIn,badFrames,strayFrames};
    const bool ok=go::MotorTransport::receive(port,id,end,out,stats,error);
    bytesIn=stats.bytesIn; badFrames=stats.badFrames; strayFrames=stats.strayFrames;
    return ok;
}

// 向 id 发一条全零 FOC 命令并等它的反馈。
bool probeOnce(go::SerialPort &port, int id, size_t timeout_us, go::Feedback &out,
               double &rtt_ms, size_t &bytesIn, size_t &badFrames,
               size_t &strayFrames, std::string &error)
{
    go::Command cmd;
    cmd.reset(); // 全零控制量 + FOC 模式（卸力）
    cmd.id = static_cast<uint8_t>(id);

    const auto start = go::SerialPort::Clock::now();
    const auto end = start + std::chrono::microseconds(timeout_us);
    go::TransportStats stats{bytesIn,badFrames,strayFrames};
    const bool ok=go::MotorTransport::exchange(port,cmd,end,out,stats,error);
    bytesIn=stats.bytesIn; badFrames=stats.badFrames; strayFrames=stats.strayFrames;
    if (!ok) return false;
    rtt_ms = std::chrono::duration<double, std::milli>(go::SerialPort::Clock::now() - start).count();
    return true;
}

// 第一阶段：逐个串口、逐个 ID 扫描。
PortReport scanPort(const std::string &device, size_t timeout_us, int attempts)
{
    PortReport report;
    report.device = device;

    go::SerialPort port;
    if (!port.open(device, kBaud, timeout_us))
    {
        report.openError = port.lastError();
        return report;
    }
    report.opened = true;
    report.baud = port.baudrate();
    std::printf("  已打开，波特率回读 %u\n", report.baud);

    for (int id = kMinId; id <= kMaxId; ++id)
    {
        go::Feedback feedback;
        double rtt_ms = 0.0;
        std::string error;
        bool ok = false;
        for (int attempt = 0; attempt < attempts && !ok; ++attempt)
            ok = probeOnce(port, id, timeout_us, feedback, rtt_ms, report.bytesIn,
                           report.badFrames, report.strayFrames, error);
        if (ok)
        {
            report.hits.push_back(MotorHit{id, feedback, rtt_ms});
            std::string err_text;
            if (feedback.error)
                err_text = std::string(" (") + go::motorErrorText(feedback.error) + ")";
            std::printf("  ID %2d  应答  %6.2f ms  mode=%u  q=%+7.2f dq=%+7.2f tau=%+6.2f  "
                        "T=%d  err=%u%s\n",
                        id, rtt_ms, unsigned(feedback.mode), double(feedback.q),
                        double(feedback.dq), double(feedback.tau), int(feedback.temp),
                        unsigned(feedback.error), err_text.c_str());
        }
        else
        {
            report.missed.push_back(id);
        }
    }

    if (!report.missed.empty())
    {
        std::printf("  未应答:");
        for (int id : report.missed)
            std::printf(" %d", id);
        std::printf("\n");
    }
    std::printf("  线路统计: 收到 %zu 字节，坏帧 %zu，非请求 ID 帧 %zu\n",
                report.bytesIn, report.badFrames, report.strayFrames);
    port.close();
    return report;
}

// 每路一行汇总：按给定 ID 顺序列出“是否应答 + 当前位置 q + 可直接填进 addMotor() 的转子侧 offset”。
// offset = q * kGearRatio：q 是输出侧角度（协议原始值已除以 6.33），而
// addMotor(id, offset_rad) 要的是转子侧角度，两者差的正是这个减速比。
// 填上它以后，该电机当前位置即成为逻辑零点（再发送 q=0 就是保持当前位置）。
void printBusLine(const PortReport &report, const std::vector<int> &show_ids)
{
    if (!report.opened)
    {
        std::printf("%-22s | 打开失败: %s\n", report.device.c_str(), report.openError.c_str());
        return;
    }
    std::printf("%-22s", report.device.c_str());
    for (int id : show_ids)
    {
        const MotorHit *hit = nullptr;
        for (const MotorHit &candidate : report.hits)
            if (candidate.id == id)
                hit = &candidate;
        if (hit)
            std::printf(" | ID%d 应答 off=%+7.3f err=%u", id, double(hit->feedback.q),
                        unsigned(hit->feedback.error));
        else
            std::printf(" | ID%d 无应答", id);
    }
    std::printf("\n");
}

// 第二阶段：让所有"有电机的口"几乎同时下发零命令，再逐个收回复。
// 如果这些口其实并在同一条 485 总线上，两个发送器会互相冲掉帧，回复会消失。
void simultaneousProbe(const std::vector<PortReport> &reports, size_t timeout_us)
{
    std::vector<std::string> active;
    std::vector<int> activeId;
    for (const PortReport &report : reports)
    {
        if (report.opened && !report.hits.empty())
        {
            active.push_back(report.device);
            activeId.push_back(report.hits.front().id);
        }
    }
    if (active.size() < 2)
    {
        std::printf("\n========== 并发发送自检 ==========\n");
        std::printf("  可用的口少于 2 个（或只有一个口上有电机），跳过。\n");
        return;
    }

    std::printf("\n========== 并发发送自检：%zu 个口同时发 ==========\n", active.size());
    std::printf("  （用于判断这些口的 485 是否并在同一条物理总线上）\n");

    std::vector<std::unique_ptr<go::SerialPort>> ports;
    std::vector<int> ids;
    for (size_t i = 0; i < active.size(); ++i)
    {
        std::unique_ptr<go::SerialPort> port(new go::SerialPort());
        if (!port->open(active[i], kBaud, timeout_us))
        {
            std::printf("  %s 打开失败: %s\n", active[i].c_str(), port->lastError().c_str());
            continue;
        }

        ports.push_back(std::move(port));
        ids.push_back(activeId[i]);
    }
    if (ports.size() < 2)
    {
        std::printf("  实际可用端口不足 2 个，跳过。\n");
        return;
    }

    // 先把所有零命令背靠背写出去，中间不等待。
    for (size_t i = 0; i < ports.size(); ++i)
    {
        go::Command cmd;
        cmd.reset();
        cmd.id = static_cast<uint8_t>(ids[i]);
        std::string error;
        if (!go::MotorTransport::send(*ports[i],cmd,ports[i]->deadline(),error))
            std::printf("  发送失败: %s\n",error.c_str());
    }

    // 再逐个收回复。
    for (size_t i = 0; i < ports.size(); ++i)
    {
        go::Feedback feedback;
        size_t bytesIn = 0, badFrames = 0, strayFrames = 0;
        std::string error;
        const auto end = go::SerialPort::Clock::now() + std::chrono::microseconds(timeout_us);
        const bool ok = awaitReply(*ports[i], ids[i], end, feedback, bytesIn, badFrames,
                                   strayFrames, error);
        std::printf("  %-22s ID %2d  %s", active[i].c_str(), ids[i],
                    ok ? "应答正常" : "无应答");
        if (!ok)
            std::printf("  (%s)", error.c_str());
        else
        {
            std::string err_text;
            if (feedback.error)
                err_text = std::string(" (") + go::motorErrorText(feedback.error) + ")";
            std::printf("  q=%+7.2f err=%u%s", double(feedback.q), unsigned(feedback.error),
                        err_text.c_str());
        }
        std::printf("   [收到 %zu 字节，坏帧 %zu，非请求 ID 帧 %zu]\n",
                    bytesIn, badFrames, strayFrames);
    }
}

} // namespace

// 读回 config/real_config.cpp 里该设备那一行的**表达式形状**（如 "0.35299f - HIP_START_OFFSET"），
// 返回第 index 项"数字之后"的那一段（含前导空格，例如 " - HIP_START_OFFSET"；纯数字则返回空）。
// 宏名与符号由现有代码决定，工具不需要知道起始姿态、也不需要任何命令行参数。
// 路径从 __FILE__ 推出（tools/../config/real_config.cpp）；解析不到就返回空 → 退回纯数字。
bool g_configReadOk = false; // 有没有成功读到 go_config.cpp（用于给出提示）

std::string offsetSuffixAt(const std::string &device, size_t index)
{
    // 候选路径：__FILE__ 推出的（源码目录）、相对当前目录的、以及从 build/ 往上一层的。
    // 用户在哪个目录跑都能找到。
    const std::string candidates[] = {std::string(__FILE__).substr(0, std::string(__FILE__).find_last_of("/\\") + 1) + "../config/real_config.cpp",
                                      "config/real_config.cpp", "../config/real_config.cpp"};
    std::string text;
    for (const std::string &path : candidates)
    {
        std::ifstream in(path);
        if (in)
        {
            text.assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            g_configReadOk = true;
            break;
        }
    }
    if (text.empty())
        return {};
    const std::string key = "BusConfig{\"" + device + "\"";
    size_t at = text.find(key);
    if (at == std::string::npos)
        return {};
    size_t pos = at + key.size(), begin = std::string::npos;
    for (int depth = 0; depth < 2; ++depth) // key 里已含 BusConfig{ → 第 2 个 '{' 才是 offset 列表
    {
        begin = text.find('{', pos);
        if (begin == std::string::npos)
            return {};
        pos = begin + 1;
    }
    size_t end = begin + 1;
    for (int depth = 1; depth > 0 && end < text.size(); ++end)
    {
        if (text[end] == '{') ++depth;
        else if (text[end] == '}') --depth;
    }
    if (end <= begin)
        return {};
    const std::string block = text.substr(begin + 1, end - begin - 2);
    size_t k = 0;
    for (size_t i = 0, start = 0; i <= block.size(); ++i)
    {
        if (i != block.size() && block[i] != ',')
            continue;
        std::string term = block.substr(start, i - start);
        start = i + 1;
        size_t p = 0;
        while (p < term.size() && std::isspace(uint8_t(term[p]))) // 项之间是 ", "，有前导空格
            ++p;
        while (p < term.size() && (std::isdigit(uint8_t(term[p])) || term[p] == '.' ||
                                   term[p] == '-' || term[p] == '+' || term[p] == 'e'))
            ++p;
        if (p < term.size() && term[p] == 'f')
            ++p;
        if (k++ == index)
            return term.substr(p); // " - HIP_START_OFFSET" 或空
    }
    return {};
}

int RunProbe(int argc, char **argv)
{
    size_t timeout_us = 40000;
    int attempts = 2;
    std::vector<int> show_ids{0, 1, 2}; // 每路一行汇总里显示的 ID，可用 -i 覆盖
    std::vector<std::string> devices;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help")
        {
            usage(argv[0]);
            return 0;
        }
        if (arg == "-t" && i + 1 < argc)
        {
            timeout_us = size_t(std::strtoul(argv[++i], nullptr, 10)) * 1000;
            if (timeout_us == 0 || timeout_us > 10000000)
            {
                std::fprintf(stderr, "-t 必须在 1..10000 毫秒之间\n");
                return 1;
            }
            continue;
        }
        if (arg == "-n" && i + 1 < argc)
        {
            attempts = std::atoi(argv[++i]);
            if (attempts < 1 || attempts > 10)
            {
                std::fprintf(stderr, "-n 必须在 1..10 之间\n");
                return 1;
            }
            continue;
        }
        if (arg == "-i" && i + 1 < argc)
        {
            show_ids.clear();
            const std::string list = argv[++i];
            size_t pos = 0;
            while (pos <= list.size())
            {
                const size_t comma = list.find(',', pos);
                const std::string token = list.substr(
                    pos, comma == std::string::npos ? std::string::npos : comma - pos);
                if (!token.empty())
                {
                    const int id = std::atoi(token.c_str());
                    if (id < kMinId || id > kMaxId)
                    {
                        std::fprintf(stderr, "-i 里的 ID 必须在 %d..%d 之间\n", kMinId, kMaxId);
                        return 1;
                    }
                    show_ids.push_back(id);
                }
                if (comma == std::string::npos)
                    break;
                pos = comma + 1;
            }
            if (show_ids.empty())
            {
                std::fprintf(stderr, "-i 至少需要一个 ID\n");
                return 1;
            }
            continue;
        }
        devices.push_back(arg);
    }
    if (devices.empty()) {
        for (const auto& bus : go::makeDefaultConfig().buses) devices.push_back(bus.device);
        std::sort(devices.begin(), devices.end());
    }

    std::printf("go_probe: 扫描 %zu 个串口 × ID %d..%d，每次等待 %zu ms，每个 ID 尝试 %d 次\n",
                devices.size(), kMinId, kMaxId, timeout_us / 1000, attempts);
    std::printf("警告: 发送的是 FOC 全零控制量，电机会卸力（失去保持力矩）。\n");

    std::vector<PortReport> reports;
    for (const std::string &device : devices)
    {
        std::printf("\n[%s]\n", device.c_str());
        std::fflush(stdout);
        PortReport report = scanPort(device, timeout_us, attempts);
        if (!report.opened)
            std::printf("  打开失败: %s\n", report.openError.c_str());
        reports.push_back(report);
    }

    // ---------------- 每路一行汇总（用于填 offset） ----------------
    std::printf("\n========== 每路一行汇总（每行一条总线） ==========\n");
    std::printf("  q = 输出侧位置 rad；off = 当前位置（**输出侧** rad，可直接填 offsets_rad）（填上后当前位置即逻辑零点）\n");
    for (const PortReport &report : reports)
        printBusLine(report, show_ids);

    // ---------------- 汇总 ----------------
    std::printf("\n================ 汇总 ================\n");
    size_t found = 0;
    for (const PortReport &report : reports)
    {
        if (!report.opened)
        {
            std::printf("  %-22s 打开失败\n", report.device.c_str());
            continue;
        }
        if (report.hits.empty())
        {
            std::printf("  %-22s 未发现应答电机\n", report.device.c_str());
            continue;
        }
        std::printf("  %-22s 应答 ID:", report.device.c_str());
        for (const MotorHit &hit : report.hits)
            std::printf(" %d", hit.id);
        std::printf("   (共 %zu 台)\n", report.hits.size());
        found += report.hits.size();
    }
    std::printf("  合计发现 %zu 台电机\n", found);

    if (found > 0)
    {
        // 按 go_config.cpp 里现有的总线顺序输出：整段替换后关节编号不会变
        const go::RobotConfig config = go::makeDefaultConfig();
        std::vector<const PortReport *> ordered;
        for (const go::BusConfig &bus : config.buses)
            for (const PortReport &report : reports)
                if (report.opened && !report.hits.empty() && report.device == bus.device)
                    ordered.push_back(&report);
        for (const PortReport &report : reports)
        {
            if (!report.opened || report.hits.empty())
                continue;
            bool known = false;
            for (const go::BusConfig &bus : config.buses)
                if (bus.device == report.device)
                    known = true;
            if (!known)
                ordered.push_back(&report); // 配置里没有的口，按扫描顺序补在后面
        }

        std::printf("\n建议写入 config/real_config.cpp 的 makeDefaultConfig()"
                    "（offset = 当前位置（输出侧 rad），使当前位置成为逻辑零点；未应答的 ID 已略去）：\n");
        for (const PortReport *report : ordered)
        {
            const go::BusConfig *cfg = nullptr;
            for (const go::BusConfig &bus : config.buses)
                if (bus.device == report->device)
                    cfg = &bus;

            std::printf("    config.buses.push_back(BusConfig{\"%s\", {", report->device.c_str());
            for (size_t i = 0; i < report->hits.size(); ++i)
                std::printf("%s%d", i ? ", " : "", report->hits[i].id);
            std::printf("},\n                                     {");
            for (size_t i = 0; i < report->hits.size(); ++i)
                std::printf("%s%.5ff%s", i ? ", " : "",
                            report->hits[i].feedback.q,
                            offsetSuffixAt(report->device, i).c_str());
            std::printf("},\n                                     {");
            for (size_t i = 0; i < report->hits.size(); ++i)
            {
                double dir = 1.0; // 新口 / 配置里查不到时按 +1
                if (cfg)
                    for (size_t m = 0;
                         m < cfg->motor_ids.size() && m < cfg->motor_directions.size(); ++m)
                        if (cfg->motor_ids[m] == report->hits[i].id)
                            dir = cfg->motor_directions[m];
                std::printf("%s%+.1ff", i ? ", " : "", dir);
            }
            std::printf("}});\n");
        }
        if (!g_configReadOk)
            std::printf("    （注意：没读到 config/real_config.cpp，所以只输出了纯数字；"
                        "在项目根目录或 build/ 下跑都能读到）\n");
        std::printf("    （offset 是**输出侧** rad、与 q 同值、不需要乘 6.33；"
                    "手上要按起始姿态加修正项的话，直接在这一行上加 `± XXX_START_OFFSET` 即可）\n");
    }
    else
    {
        std::printf("\n一个电机都没应答。请依次检查：\n"
                    "  1) 电机是否上电（电机完全不回复最常见的原因）；\n"
                    "  2) 485 的 A/B 是否接对、是否与适配器同一个口；\n"
                    "  3) 是否两个口并在同一条总线上（跑一次上面的并发自检）；\n"
                    "  4) 电机 ID 是否在 0..14 之内。\n");
    }

    simultaneousProbe(reports, timeout_us);
    return found > 0 ? 0 : 2;
}
