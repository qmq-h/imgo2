#include "go_log.h"

#include <ctime>
#include <sys/stat.h>

namespace go
{

namespace
{

// 文件是否已存在（用来给同一秒内的重复运行让路）
bool fileExists(const std::string &path)
{
    std::FILE *f = std::fopen(path.c_str(), "r");
    if (!f)
        return false;
    std::fclose(f);
    return true;
}

// 目录不存在就建（只建最后一级；更上层的父目录需要已经存在）
void ensureDir(const std::string &dir)
{
    if (dir.empty())
        return;
    struct stat info;
    if (::stat(dir.c_str(), &info) == 0)
        return;
    ::mkdir(dir.c_str(), 0755);
}

} // namespace

Logger::~Logger()
{
    close();
}

bool Logger::open(const std::string &path, std::string &error)
{
    close();
    ensureDir(path.substr(0, path.find_last_of('/')));

    file_ = std::fopen(path.c_str(), "w");
    if (!file_)
    {
        error = "无法创建日志文件 " + path;
        return false;
    }
    path_ = path;
    columns_ = 0;
    with_targets_ = false;
    return true;
}

void Logger::writeHeader(const std::vector<std::string> &joints, bool with_targets)
{
    if (!file_)
        return;

    std::fprintf(file_, "# 电机角度反馈日志\n");
    std::fprintf(file_, "# t_s = 从程序启动算起的秒数\n");
    std::fprintf(file_, "# qN  = 第 N 个关节的输出侧逻辑位置 (rad)\n");
    if (!joints.empty())
    {
        std::fprintf(file_, "# 关节对应关系:");
        for (size_t i = 0; i < joints.size(); ++i)
            std::fprintf(file_, " %zu=%s", i, joints[i].c_str());
        std::fprintf(file_, "\n");
    }

    std::fprintf(file_, "t_s");
    for (size_t i = 0; with_targets && i < joints.size(); ++i)
        std::fprintf(file_, ",q_des%zu", i);
    for (size_t i = 0; i < joints.size(); ++i)
        std::fprintf(file_, ",q%zu", i);
    std::fprintf(file_, "\n");

    columns_ = joints.size();
    with_targets_ = with_targets;
    std::fflush(file_);
}

void Logger::writeFeedback(double seconds, const std::vector<go::Feedback> &feedback,
                           const std::vector<float> &targets_q)
{
    if (!file_)
        return;
    // 列数对不上就整行不写，避免日志里出现半截数据还看不出来
    if (columns_ == 0 || feedback.size() != columns_)
        return;
    if (with_targets_ && targets_q.size() != columns_)
        return;
    if (!with_targets_ && !targets_q.empty())
        return; // 表头没声明 q_des* 列，却给了目标：宁可不写

    std::fprintf(file_, "%.6f", seconds);
    if (with_targets_)
        for (float value : targets_q)
            std::fprintf(file_, ",%.6f", double(value));
    for (const go::Feedback &item : feedback)
        std::fprintf(file_, ",%.6f", double(item.q));
    std::fprintf(file_, "\n");
    std::fflush(file_); // 立刻落盘：Ctrl+C 或异常退出也不丢已写的行
}

void Logger::close()
{
    if (file_)
    {
        std::fclose(file_);
        file_ = nullptr;
    }
}

std::string Logger::makeRunPath(const std::string &dir)
{
    std::time_t now = std::time(nullptr);
    std::tm local{};
#if defined(_WIN32)
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    char stamp[32] = {0};
    std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", &local);

    const std::string prefix = dir.empty() ? std::string() : dir + "/";
    std::string path = prefix + "run_" + stamp + ".log";
    for (int suffix = 2; fileExists(path) && suffix < 1000; ++suffix)
        path = prefix + "run_" + stamp + "_" + std::to_string(suffix) + ".log";
    return path;
}

} // namespace go
