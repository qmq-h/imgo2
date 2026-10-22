#include "display/motor_panel.h"
#include <algorithm>
#include <cstdarg>
#include <cstdio>
namespace go_display {
using PortState=go::DiagnosticBus;
using MotorState=go::DiagnosticMotor;
double toLogical(const MotorState& motor,double raw) { return motor.direction*(raw-motor.offset); }
namespace {
long long sumErrorRounds(const std::vector<PortState>& states) {
    long long n = 0;
    for (const auto& s : states) n += s.errorRounds;
    return n;
}
long long sumErrorCount(const std::vector<PortState>& states) {
    long long n = 0;
    for (const auto& s : states) n += static_cast<long long>(s.errorSeq);
    return n;
}
// 最近一条错误（来自错误序号最大的那条总线），按长度截断
std::string latestError(const std::vector<PortState>& states, size_t max_len) {
    const PortState* worst = nullptr;
    for (const auto& s : states)
        if (!worst || s.errorSeq > worst->errorSeq) worst = &s;
    if (!worst || worst->errors.empty()) return {};
    std::string text = worst->device + " " + worst->errors.back();
    if (text.size() > max_len) text = text.substr(0, max_len) + "...";
    return text;
}
} // namespace
void append(std::string& output, const char* format, ...) {
    va_list args, copy;
    va_start(args, format);
    va_copy(copy, args);
    const int size = std::vsnprintf(nullptr, 0, format, copy);
    va_end(copy);
    if (size > 0) {
        std::vector<char> buffer(static_cast<size_t>(size) + 1);
        std::vsnprintf(buffer.data(), buffer.size(), format, args);
        output.append(buffer.data(), static_cast<size_t>(size));
    }
    va_end(args);
}

void printBusLine(std::string& screen, const PortState &state)
{
    if (!state.opened)
    {
        append(screen, "%-22s | 打开失败: %s\n", state.device.c_str(), state.error.c_str());
        return;
    }
    append(screen, "%-22s", state.device.c_str());
    for (const MotorState &motor : state.motors)
    {
        if (!motor.feedback.valid)
        {
            if (motor.timeoutCount)
                append(screen, " | ID%d 无应答×%llu(%s)", motor.id,
                       (unsigned long long)motor.timeoutCount, motor.error.c_str());
            else
                append(screen, " | ID%d 无应答", motor.id);
            continue;
        }
        // q  = 原始输出侧角度（不受 offset/direction 影响）
        // qL = 逻辑角度 = direction × (q − offset) —— 姿态、轨迹、日志用的就是这个
        append(screen, " | ID%d qOut=%+5.2f qL=%+5.2f", motor.id, double(motor.feedback.q),
                    toLogical(motor, double(motor.feedback.q)));
        append(screen, " err=%u", unsigned(motor.feedback.error));
        if (motor.feedback.error)
            append(screen, "(%s)", go::motorErrorText(motor.feedback.error));
        if (motor.timeoutCount)
            append(screen, " TO%llu", (unsigned long long)motor.timeoutCount);
    }
    if (state.roundSent > 0)
        append(screen, "  [流水线 发%.2f/收%.2fms 收齐%d/%d 缺%d 重复%d 错轮%lld]\n",
               state.sendMs, state.replyMs, state.roundGot, state.roundSent,
               state.roundMissing, state.roundDuplicate, state.errorRounds);
    else
        append(screen, "  [轮 %.1f/%.1fms TO=%llu]\n", state.sweepMs, state.sweepMaxMs,
               (unsigned long long)state.timeoutTotal);
}

void RenderMonitor(const std::vector<PortState> &states, long long refresh,
            const imu_sensor::TerminalDisplay& display,
            const imu_sensor::MonitorSnapshot& imu, const real_config::ImuConfig& config,
            size_t timeout_us)
{
    std::string screen;
    if (display.Columns() < 200 || display.Rows() < 24) {
        append(screen, "go_monitor | zero commands | refresh %lld | Ctrl+C exits\n", refresh);
        append(screen, "Compact: ID qOut/qL (rad), err; widen for line statistics.\n");
        for (const auto& state : states) {
            append(screen, "%s", state.device.c_str());
            if (!state.opened) append(screen, " OPEN FAILED");
            append(screen, "\n");
            for (const auto& motor : state.motors) {
                if (motor.feedback.valid)
                    append(screen, " ID%d %+.2f/%+.2f e%u", motor.id,
                           double(motor.feedback.q), toLogical(motor, motor.feedback.q),
                           unsigned(motor.feedback.error));
                else append(screen, " ID%d --x%llu", motor.id, (unsigned long long)motor.timeoutCount);
            }
            if (state.roundSent > 0)
                append(screen, " [发%.1f/收%.1f 缺%d 错%d]\n", state.sendMs, state.replyMs,
                       state.roundMissing, state.errorRounds);
            else
                append(screen, " [轮 %.1f/%.1fms TO=%llu]\n", state.sweepMs, state.sweepMaxMs,
                       (unsigned long long)state.timeoutTotal);
        }
        {
            uint64_t timeouts = 0; size_t bad = 0; double worst = 0;
            for (const auto& state : states) {
                timeouts += state.timeoutTotal;
                bad += state.badFrames;
                worst = std::max(worst, state.sweepMaxMs);
            }
            append(screen, "合计 超时=%llu 坏帧=%zu 最坏轮=%.2fms 错轮=%lld",
                   (unsigned long long)timeouts, bad, worst, sumErrorRounds(states));
            const std::string err = latestError(states, 36);
            if (!err.empty()) append(screen, " 最后: %s", err.c_str());
            append(screen, "\n");
        }
        display.Show(screen + imu_sensor::MonitorPanel(imu, config));
        return;
    }
    append(screen, "go_monitor —— 持续发送 FOC 全零命令（卸力），实时查看电机反馈与线路质量"
                "      第 %lld 次刷新\n",
                refresh);
    append(screen, "  核对正方向：机械架空后用手慢慢搬动关节，看该台的 qL 往哪边变；\n");
    append(screen, "  方向据此填 config/real_config.cpp 的 motor_directions（+1 原样 / -1 取反）。\n");
    append(screen, "  qOut = 输出端角度（协议层已经除以减速比 6.33；未减 offset）\n");
    append(screen, "  qL   = 逻辑角度 = direction × (qOut − offset) —— 姿态/轨迹/日志用的都是它\n");
    append(screen, "  qOut 与 qL 都是输出端 rad；控制目标与反馈采用逻辑角度 qL。\n");
    append(screen, "  范围在退出时按 qL 汇总，可直接填 config.limits 的 q_min/q_max。Ctrl+C 退出。\n\n");

    for (const PortState &state : states)
        printBusLine(screen, state);

    size_t bytes = 0, bad = 0, stray = 0; uint64_t timeouts = 0; double worst = 0;
    for (const PortState &state : states)
    {
        bytes += state.bytesIn;
        bad += state.badFrames;
        stray += state.strayFrames;
        timeouts += state.timeoutTotal;
        worst = std::max(worst, state.sweepMaxMs);
    }
    append(screen, "\n  合计: 收到 %zu 字节，坏帧 %zu，非请求 ID 帧 %zu | 超时 %llu 次"
                   "（单次预算 %.1f ms，最坏单轮 %.2f ms）| 流水线错轮 %lld，错误 %lld 条\n",
           bytes, bad, stray, (unsigned long long)timeouts, timeout_us / 1000.0, worst,
           sumErrorRounds(states), sumErrorCount(states));
    const std::string last = latestError(states, display.Columns() > 30 ? display.Columns() - 12 : 40);
    if (!last.empty()) append(screen, "  最后错误: %s\n", last.c_str());
    else append(screen, "  最后错误: 无\n");
    display.Show(screen + imu_sensor::MonitorPanel(imu, config));
}

}
