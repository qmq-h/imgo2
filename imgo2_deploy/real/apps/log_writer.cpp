#include "apps/log_writer.h"

#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>

namespace go_app {
namespace {

constexpr int kRobotColumns = 153;
constexpr int kPolicyColumns = 67;

// 快速数字格式化（只在写线程里用）。
void num(std::string& out, float v, int precision = 5) {
    char buf[64];
    const auto result = std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::fixed, precision);
    if (result.ec == std::errc()) out.append(buf, result.ptr);
    else out += "nan";
}
void num(std::string& out, double v, int precision = 3) {
    char buf[64];
    const auto result = std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::fixed, precision);
    if (result.ec == std::errc()) out.append(buf, result.ptr);
    else out += "nan";
}
void inum(std::string& out, long long v) {
    char buf[32];
    const auto result = std::to_chars(buf, buf + sizeof(buf), v);
    out.append(buf, result.ptr);
}
// 简单的 JSON 字符串转义（路径里的引号/反斜杠）。
std::string jsonEscape(const std::string& in) {
    std::string out;
    out.reserve(in.size() + 8);
    for (char c : in) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\t': out += "\\t"; break;
        default: out += c;
        }
    }
    return out;
}
void headerGroup(std::string& out, const char* prefix, int count) {
    char buf[48];
    for (int i = 0; i < count; ++i) {
        std::snprintf(buf, sizeof(buf), ",%s_%02d", prefix, i);
        out += buf;
    }
}
void writeRobotHeader(std::FILE* f) {
    std::string h = "t_mono_ms,t_wall_ms,tick,gen,mode,clamped,valid_mask,err_mask,cmd_vx,cmd_vy,cmd_wz";
    headerGroup(h, "kp", 12);
    headerGroup(h, "kd", 12);
    headerGroup(h, "q_des", 12);
    headerGroup(h, "dq_des", 12);
    headerGroup(h, "tau_ff", 12);
    headerGroup(h, "q", 12);
    headerGroup(h, "dq", 12);
    headerGroup(h, "tau", 12);
    headerGroup(h, "temp", 12);
    headerGroup(h, "age", 12);
    h += ",quat_w,quat_x,quat_y,quat_z,gyro_x,gyro_y,gyro_z,imu_age_ms,imu_valid,imu_hz";
    headerGroup(h, "act", 12);
    for (int b = 0; b < 4; ++b) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), ",b%d_total_us,b%d_flush_us,b%d_write_us,b%d_reply_us", b, b, b, b);
        h += buf;
    }
    h += '\n';
    if (f) { std::fwrite(h.data(), 1, h.size(), f); std::fflush(f); }
}
void writePolicyHeader(std::FILE* f) {
    std::string h = "t_mono_ms,t_wall_ms,tick,gen,mode,cmd_vx,cmd_vy,cmd_wz";
    headerGroup(h, "obs", 45);
    headerGroup(h, "act", 12);
    h += ",result_age_ms,obs_age_ms\n";
    if (f) { std::fwrite(h.data(), 1, h.size(), f); std::fflush(f); }
}
} // namespace

RunLogger::~RunLogger() { close(); }

bool RunLogger::open(const std::string& dir, const LogMeta& meta, int max_mb, std::string& error) {
    error.clear();
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) { error = "cannot create log directory: " + ec.message(); return false; }
    dir_ = dir;
    meta_ = meta;
    max_bytes_ = max_mb > 0 ? max_mb * 1024 * 1024 : 0;
    robot_path_ = dir_ + "/robot.csv";
    robot_file_ = std::fopen(robot_path_.c_str(), "wb");
    policy_file_ = std::fopen((dir_ + "/policy.csv").c_str(), "wb");
    events_file_ = std::fopen((dir_ + "/events.log").c_str(), "wb");
    if (!robot_file_ || !policy_file_ || !events_file_) {
        error = "cannot open log files under " + dir;
        close();
        return false;
    }
    writeRobotHeader(robot_file_);
    writePolicyHeader(policy_file_);

    if (std::FILE* m = std::fopen((dir_ + "/meta.json").c_str(), "wb")) {
        std::fprintf(m, "{\n");
        std::fprintf(m, "  \"created\": \"%s\",\n", jsonEscape(meta.created).c_str());
        std::fprintf(m, "  \"exe\": \"%s\",\n", jsonEscape(meta.exe).c_str());
        std::fprintf(m, "  \"args\": \"%s\",\n", jsonEscape(meta.args).c_str());
        std::fprintf(m,
                     "  \"log\": {\"every\": %d, \"control_ms\": %d, \"policy_ms\": %d,"
                     " \"robot_columns\": %d, \"policy_columns\": %d},\n",
                     meta.every, meta.control_ms, meta.policy_ms, kRobotColumns, kPolicyColumns);
        std::fprintf(m,
                     "  \"units\": {\"angle\": \"rad (输出侧、逻辑方向)\", \"velocity\": \"rad/s\","
                     " \"torque\": \"N.m\", \"kp\": \"N.m/rad\", \"kd\": \"N.m/(rad/s)\","
                     " \"temp\": \"degC\", \"time\": \"ms\"},\n");
        std::fprintf(m,
                     "  \"orders\": {\"robot\": \"硬件顺序（总线顺序 x 电机顺序）\","
                     " \"policy\": \"策略顺序（policy_to_hardware 的逆）\","
                     " \"policy_csv\": \"obs 为策略顺序、act 为策略顺序\"},\n");
        std::fprintf(m, "  \"obs_layout\": {\"0-2\": \"gyro*ang_vel_scale\", \"3-5\": \"projected_gravity\","
                        " \"6-8\": \"cmd*commands_scale\", \"9-20\": \"(q-default)*dof_pos_scale\","
                        " \"21-32\": \"dq*dof_vel_scale\", \"33-44\": \"上一版动作\","
                        " \"note\": \"统一 clamp 到 +-clip_obs\"},\n");
        std::fprintf(m, "  \"joints\": %s,\n", meta.joints_json.c_str());
        std::fprintf(m, "  \"config\": %s\n", meta.config_json.c_str());
        std::fprintf(m, "}\n");
        std::fclose(m);
    }

    robot_ring_.resize(kRobotRing);
    policy_ring_.resize(kPolicyRing);
    stop_.store(false);
    enabled_ = true;
    writer_ = std::thread([this] { run(); });
    return true;
}

void RunLogger::push(const LogSample& sample) {
    if (!enabled_) return;
    const auto head = robot_head_.load(std::memory_order_relaxed);
    const auto tail = robot_tail_.load(std::memory_order_acquire);
    if (head - tail >= kRobotRing) { dropped_.fetch_add(1, std::memory_order_relaxed); return; }
    robot_ring_[head % kRobotRing] = sample;
    robot_head_.store(head + 1, std::memory_order_release);
}

void RunLogger::pushPolicy(const PolicySample& sample) {
    if (!enabled_) return;
    const auto head = policy_head_.load(std::memory_order_relaxed);
    const auto tail = policy_tail_.load(std::memory_order_acquire);
    if (head - tail >= kPolicyRing) { dropped_.fetch_add(1, std::memory_order_relaxed); return; }
    policy_ring_[head % kPolicyRing] = sample;
    policy_head_.store(head + 1, std::memory_order_release);
}

void RunLogger::event(const std::string& line) {
    if (!enabled_ || !events_file_) return;
    std::fprintf(events_file_, "%s\n", line.c_str());
    std::fflush(events_file_);
}

void RunLogger::row(const LogSample& s, std::string& o) {
    o.clear();
    num(o, s.t_mono_ms, 3); o += ',';
    inum(o, static_cast<long long>(s.t_wall_ms)); o += ',';
    inum(o, static_cast<long long>(s.tick)); o += ',';
    inum(o, static_cast<long long>(s.generation)); o += ',';
    inum(o, s.mode); o += ',';
    inum(o, s.clamped); o += ',';
    inum(o, s.valid_mask); o += ',';
    inum(o, s.err_mask); o += ',';
    for (int i = 0; i < 3; ++i) { num(o, s.cmd[i], 4); o += ','; }
    for (int i = 0; i < 12; ++i) { num(o, s.kp[i], 3); o += ','; }
    for (int i = 0; i < 12; ++i) { num(o, s.kd[i], 4); o += ','; }
    for (int i = 0; i < 12; ++i) { num(o, s.q_des[i], 5); o += ','; }
    for (int i = 0; i < 12; ++i) { num(o, s.dq_des[i], 4); o += ','; }
    for (int i = 0; i < 12; ++i) { num(o, s.tau_ff[i], 4); o += ','; }
    for (int i = 0; i < 12; ++i) { num(o, s.q[i], 5); o += ','; }
    for (int i = 0; i < 12; ++i) { num(o, s.dq[i], 5); o += ','; }
    for (int i = 0; i < 12; ++i) { num(o, s.tau[i], 4); o += ','; }
    for (int i = 0; i < 12; ++i) { inum(o, s.temp[i]); o += ','; }
    for (int i = 0; i < 12; ++i) { inum(o, s.age_ms[i]); o += ','; }
    for (int i = 0; i < 4; ++i) { num(o, s.quat[i], 6); o += ','; }
    for (int i = 0; i < 3; ++i) { num(o, s.gyro[i], 5); o += ','; }
    inum(o, s.imu_age_ms); o += ',';
    inum(o, s.imu_valid); o += ',';
    num(o, s.imu_hz, 1); o += ',';
    for (int i = 0; i < 12; ++i) { num(o, s.act[i], 4); o += ','; }
    for (int b = 0; b < 4; ++b) {
        inum(o, s.bus_total_us[b]); o += ',';
        inum(o, s.bus_flush_us[b]); o += ',';
        inum(o, s.bus_write_us[b]); o += ',';
        inum(o, s.bus_reply_us[b]); o += (b == 3 ? '\n' : ',');
    }
}

void RunLogger::prow(const PolicySample& s, std::string& o) {
    o.clear();
    num(o, s.t_mono_ms, 3); o += ',';
    inum(o, static_cast<long long>(s.t_wall_ms)); o += ',';
    inum(o, static_cast<long long>(s.tick)); o += ',';
    inum(o, static_cast<long long>(s.generation)); o += ',';
    inum(o, s.mode); o += ',';
    for (int i = 0; i < 3; ++i) { num(o, s.cmd[i], 4); o += ','; }
    for (int i = 0; i < 45; ++i) { num(o, s.obs[i], 6); o += ','; }
    for (int i = 0; i < 12; ++i) { num(o, s.act[i], 5); o += ','; }
    num(o, s.result_age_ms, 2); o += ',';
    num(o, s.obs_age_ms, 2); o += '\n';
}

void RunLogger::rotateIfNeeded() {
    if (max_bytes_ <= 0 || !robot_file_ || robot_bytes_ < max_bytes_) return;
    std::fclose(robot_file_);
    ++file_index_;
    char name[64];
    std::snprintf(name, sizeof(name), "/robot_%03d.csv", file_index_);
    robot_path_ = dir_ + name;
    robot_file_ = std::fopen(robot_path_.c_str(), "wb");
    robot_bytes_ = 0;
    if (robot_file_) writeRobotHeader(robot_file_);
}

void RunLogger::run() {
    std::string text;
    text.reserve(2048);
    int flush_counter = 0;
    for (;;) {
        bool any = false;
        for (std::size_t n = 0; n < 256; ++n) {
            const auto tail = robot_tail_.load(std::memory_order_relaxed);
            if (tail == robot_head_.load(std::memory_order_acquire)) break;
            row(robot_ring_[tail % kRobotRing], text);
            if (robot_file_) robot_bytes_ += static_cast<long long>(std::fwrite(text.data(), 1, text.size(), robot_file_));
            robot_tail_.store(tail + 1, std::memory_order_release);
            any = true;
        }
        for (std::size_t n = 0; n < 64; ++n) {
            const auto tail = policy_tail_.load(std::memory_order_relaxed);
            if (tail == policy_head_.load(std::memory_order_acquire)) break;
            prow(policy_ring_[tail % kPolicyRing], text);
            if (policy_file_) std::fwrite(text.data(), 1, text.size(), policy_file_);
            policy_tail_.store(tail + 1, std::memory_order_release);
            any = true;
        }
        rotateIfNeeded();
        if (stop_.load(std::memory_order_relaxed) && !any) break;
        if (!any) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } else if ((++flush_counter & 0x3F) == 0) {
            if (robot_file_) std::fflush(robot_file_);
            if (policy_file_) std::fflush(policy_file_);
        }
    }
    if (robot_file_) std::fflush(robot_file_);
    if (policy_file_) std::fflush(policy_file_);
}

void RunLogger::close() {
    if (!enabled_) return;
    enabled_ = false;
    stop_.store(true);
    if (writer_.joinable()) writer_.join();
    if (events_file_) {
        std::fprintf(events_file_, "dropped=%llu (队列满丢弃的样本数)\n",
                     static_cast<unsigned long long>(dropped()));
    }
    for (std::FILE** f : {&robot_file_, &policy_file_, &events_file_}) {
        if (*f) { std::fflush(*f); std::fclose(*f); *f = nullptr; }
    }
}
} // namespace go_app
