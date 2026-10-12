#pragma once

#include "go_protocol.h" // Feedback

#include <cstdio>
#include <string>
#include <vector>

namespace go
{

/**
 * 把电机反馈写进文件，**每次运行新建一个文件**。
 *
 * 每行是（targets 可选，给了就多一组列）：
 *     t_s, [q_des0 ... q_desN,] q0, q1, ... qN
 * 文件开头有几行 `#` 注释，说明列的含义以及“第几列对应哪个设备/哪个 ID”，
 * 这样过几天再看这个文件也不用猜。数值用 6 位小数（终端显示才用两位，
 * 日志要留精度——标定 offset 时要靠它）。
 *
 * 用法：
 *     go::Logger logger;
 *     std::string error;
 *     if (logger.open(go::Logger::makeRunPath("log"), error)) {
 *         logger.writeHeader(multi.motorLabels());
 *         ...
 *         logger.writeFeedback(t_seconds, multi.feedback());   // 每行一次采样
 *     }
 *     // 析构自动关闭；也可以显式 logger.close()
 *
 * 写日志失败**不应该**影响控制：`open()` 返回 false 时调用方可以选择继续跑。
 */
class Logger
{
public:
    Logger() = default;
    ~Logger();

    Logger(const Logger &) = delete;
    Logger &operator=(const Logger &) = delete;

    /**
     * 打开一个新文件（覆盖同名文件）。目录不存在时会尽力创建**最后一级**。
     * 成功返回 true；失败返回 false 并把原因写进 error。
     */
    bool open(const std::string &path, std::string &error);

    /**
     * 写开头的注释和列名。joints 是每个关节的说明（例如 "/dev/ttyCH343USB0/ID0"），
     * 顺序必须与之后 writeFeedback() 传进来的 feedback 顺序一致。
     * with_targets = true 时多写一组 q_des* 列（"指令 vs 实际"对照用）。
     */
    void writeHeader(const std::vector<std::string> &joints, bool with_targets = false);

    /**
     * 写一行采样：seconds 是从程序启动算起的秒数，feedback 按关节顺序排列。
     * targets_q 是可选的目标角度（输出侧 rad，同 feedback 的顺序）；留空表示不写这一组列，
     * 但表头声明了 with_targets 时必须给。
     * 文件没打开、或长度与表头对不上时什么都不写。
     * 每次写完立刻 flush，这样被 Ctrl+C 打断也不会丢已写的内容。
     */
    void writeFeedback(double seconds, const std::vector<go::Feedback> &feedback,
                       const std::vector<float> &targets_q = {});

    /**
     * 文件是否已打开。
     */
    bool opened() const { return file_ != nullptr; }

    /**
     * 当前文件名（未打开时为空串）。
     */
    const std::string &path() const { return path_; }

    /**
     * 关闭文件；可重复调用，析构时也会调用。
     */
    void close();

    /**
     * 生成“每次运行一个”的文件名：<dir>/run_YYYYmmdd_HHMMSS.log。
     * 同一秒内重复调用会退化成加后缀 _2、_3…，保证不覆盖已有文件。
     */
    static std::string makeRunPath(const std::string &dir);

private:
    std::FILE *file_ = nullptr;
    std::string path_;
    size_t columns_ = 0;       // 表头声明的关节数，用来挡住列数不一致的写入
    bool with_targets_ = false; // 表头是否声明了 q_des* 列
};

} // namespace go
