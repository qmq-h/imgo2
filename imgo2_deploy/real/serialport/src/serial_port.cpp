// ============================================================================
//  serial_port.cpp —— 串口传输层实现
// ============================================================================
#include "serial_port.h"

#include <cerrno>
#include <cstdio>
#include <cstring>

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <poll.h>
#include <algorithm>
#include <termios.h>

namespace go
{

    // ---------------------------------------------------------------------------
    // struct termios2 与 BOTHER
    // ---------------------------------------------------------------------------
    // 为什么不直接 #include <asm/termbits.h>？
    //   它会与 <termios.h> 冲突（重复定义 termios 结构体与常量）。
    //   所以这里自行声明一份布局完全一致的结构体。
    //
    // 为什么需要 termios2？
    //   struct termios 里波特率只能存 Bxxxx 常量，表内最大值是 B4000000，
    //   没有 4800000 / 6000000。termios2 多了 c_ispeed / c_ospeed 两个
    //   unsigned int 字段，配合 BOTHER(0x1000) 哨兵值即可设置任意速率。
    //
    // x86-64 布局（共 44 字节）：
    //   偏移  0  c_iflag   (4)
    //   偏移  4  c_oflag   (4)
    //   偏移  8  c_cflag   (4)   <- CBAUD(0x100F) 位域在此
    //   偏移 12  c_lflag   (4)
    //   偏移 16  c_line    (1)
    //   偏移 17  c_cc[19]  (19)  -> 到偏移 35
    //   偏移 36  c_ispeed  (4)
    //   偏移 40  c_ospeed  (4)
    // ---------------------------------------------------------------------------
    namespace
    {

        struct termios2_compat
        {
            unsigned int c_iflag;
            unsigned int c_oflag;
            unsigned int c_cflag;
            unsigned int c_lflag;
            unsigned char c_line;
            unsigned char c_cc[19]; // NCCS = 19
            unsigned int c_ispeed;
            unsigned int c_ospeed;
        };

        static_assert(sizeof(termios2_compat) == 44, "unsupported termios2 layout");

// _IOR/_IOW 会把结构体尺寸编码进请求码：44 字节 => 0x802C542A / 0x402C542B
#define TCGETS2_COMPAT _IOR('T', 0x2A, struct termios2_compat)
#define TCSETS2_COMPAT _IOW('T', 0x2B, struct termios2_compat)

#ifndef BOTHER
#define BOTHER 0010000 // = 0x1000，Baud OTHER 哨兵值
#endif

        // 标准波特率常量表，仅用于 BOTHER 不可用时的降级
        /**
         * 内部降级查询：将数值 baud 映射到标准 Bxxxx 常量 out。
         * 返回 true 时 out 才有效；非标准速率返回 false，由调用方报告不支持。
         */
        bool stdBaudConstant(uint32_t baud, speed_t &out)
        {
            switch (baud)
            {
            case 9600:
                out = B9600;
                return true;
            case 19200:
                out = B19200;
                return true;
            case 38400:
                out = B38400;
                return true;
            case 57600:
                out = B57600;
                return true;
            case 115200:
                out = B115200;
                return true;
            case 230400:
                out = B230400;
                return true;
            case 460800:
                out = B460800;
                return true;
            case 921600:
                out = B921600;
                return true;
            case 1000000:
                out = B1000000;
                return true;
            case 2000000:
                out = B2000000;
                return true;
            case 3000000:
                out = B3000000;
                return true;
            case 4000000:
                out = B4000000;
                return true;
            default:
                return false; // 4800000 / 6000000 等非标值走这里
            }
        }

    } // namespace
    /**
     * 系统调用失败后立即调用，保存操作名 what、当前 errno 数字和错误描述到本实例。
     */
    void SerialPort::setError(const std::string &what)
    {
        const int e = errno;
        error_ = what + ": " + std::strerror(e) + " (errno=" + std::to_string(e) + ")";
    }

    /**
     * 析构时关闭文件描述符。不发送电机卸力指令；卸力由 GoBus::close() 负责。
     */
    SerialPort::~SerialPort()
    {
        close();
    }

    // ---------------------------------------------------------------------------
    // 波特率：优先 TCSETS2 + BOTHER（支持任意速率），失败则降级到标准常量
    // ---------------------------------------------------------------------------
    /**
     * 内部设置 baud：优先 termios2/BOTHER，设置接口不可用时尝试标准波特率表。
     * 设置后必须回读匹配；回读失败或不匹配直接返回 false，不视为配置成功。
     */
    bool SerialPort::applyBaudrate(uint32_t baud)
    {
        struct termios2_compat t2;
        if (::ioctl(fd_, TCGETS2_COMPAT, &t2) == 0)
        {
            t2.c_cflag &= (unsigned int)~CBAUD; // 清掉波特率位域
            t2.c_cflag |= BOTHER;               // 告诉内核：速率看 c_ispeed/c_ospeed
            t2.c_ispeed = baud;
            t2.c_ospeed = baud;
            if (::ioctl(fd_, TCSETS2_COMPAT, &t2) == 0)
            {
                // 回读确认驱动真的接受了该速率 —— 有些驱动会静默改成别的值
                struct termios2_compat rb;
                if (::ioctl(fd_, TCGETS2_COMPAT, &rb) != 0)
                {
                    setError("baud readback");
                    return false;
                }
                if (rb.c_ispeed != baud || rb.c_ospeed != baud)
                {
                    setErrorMsg("baud readback mismatch");
                    std::fprintf(stderr,
                                 "[SerialPort] 警告: 请求波特率 %u，驱动回读为 %u/%u\n",
                                 baud, rb.c_ispeed, rb.c_ospeed);
                    return false;
                }
                baud_ = baud;
                return true;
            }
            std::fprintf(stderr, "[SerialPort] TCSETS2 失败: %s，尝试降级\n",
                         std::strerror(errno));
        }
        else
        {
            std::fprintf(stderr, "[SerialPort] TCGETS2 失败: %s，尝试降级\n",
                         std::strerror(errno));
        }

        speed_t sp;
        if (!stdBaudConstant(baud, sp))
        {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                          "波特率 %u 不是标准值，且本驱动不支持 BOTHER", baud);
            setErrorMsg(buf);
            return false;
        }
        struct termios tio;
        if (::tcgetattr(fd_, &tio) != 0)
        {
            setError("tcgetattr");
            return false;
        }
        if (::cfsetispeed(&tio, sp) != 0 || ::cfsetospeed(&tio, sp) != 0)
        {
            setError("set speed");
            return false;
        }
        if (::tcsetattr(fd_, TCSANOW, &tio) != 0)
        {
            setError("tcsetattr");
            return false;
        }
        if (::tcgetattr(fd_, &tio) != 0)
        {
            setError("baud readback");
            return false;
        }
        if (::cfgetispeed(&tio) != sp || ::cfgetospeed(&tio) != sp)
        {
            setErrorMsg("standard baud readback mismatch");
            return false;
        }
        baud_ = baud;
        return true;
    }

    // ---------------------------------------------------------------------------
    // 配置：原始模式 8N1，无流控
    // ---------------------------------------------------------------------------
    /**
     * 内部初始化：设置原始二进制 8N1、无流控，配置波特率并清理旧输入。
     * 仅在 open() 获得有效描述符后调用；任一步失败返回 false，由 open() 释放资源。
     */
    bool SerialPort::configure()
    {
        struct termios tio;
        // 先读再改：不要凭空构造 termios，会踩到未知默认值
        if (::tcgetattr(fd_, &tio) != 0)
        {
            setError("tcgetattr");
            return false;
        }

        // 关键：切成原始模式。
        // cfmakeraw 关闭 ICRNL(CR->NL) / ECHO / ICANON(行缓冲) / ISIG / OPOST ...
        // 不关的话二进制协议数据会被内核静默改写，极难排查。
        ::cfmakeraw(&tio);

        tio.c_cflag |= (CLOCAL | CREAD); // 忽略调制解调器控制线；使能接收
        tio.c_cflag &= (tcflag_t)~CSIZE;
        tio.c_cflag |= CS8; // 8 数据位（必须先清 CSIZE 再设）
#ifdef CRTSCTS
        tio.c_cflag &= (tcflag_t)~CRTSCTS; // 无硬件流控
#endif
        tio.c_cflag &= (tcflag_t)~PARENB;                  // 无校验
        tio.c_cflag &= (tcflag_t)~CSTOPB;                  // 1 位停止位
        tio.c_iflag &= (tcflag_t) ~(IXON | IXOFF | IXANY); // 无软件流控

        tio.c_cc[VMIN] = 0; // 配合 O_NONBLOCK
        tio.c_cc[VTIME] = 0;

        if (::tcsetattr(fd_, TCSANOW, &tio) != 0)
        {
            setError("tcsetattr");
            return false;
        }

        if (!applyBaudrate(baud_))
            return false;

        return discardInput();
    }

    // ---------------------------------------------------------------------------
    /**
     * 打开并配置 device（如 /dev/ttyCH343USB0），baud 为请求波特率。
     * timeout_us 为配置的等待预算，单位微秒，范围 1～10000000，默认 20000。
     * 每次调用先关闭旧连接；成功返回 true，失败返回 false 并保存 lastError()。
     * 成功只代表串口配置及速率回读通过，不代表已收到电机反馈。
     */
    bool SerialPort::open(const std::string &device, uint32_t baudrate, size_t timeout_us)
    {
        close();
        error_.clear();
        if (!baudrate || !timeout_us || timeout_us > 10000000)
        {
            setErrorMsg("invalid baud/timeout (1..10000000 us)");
            return false;
        }
        baud_ = baudrate;
        timeout_us_ = timeout_us;

        // O_NOCTTY : 不要把该 tty 当作本进程的控制终端，否则 Ctrl+C 等信号会乱窜
        // O_NONBLOCK: 避免 open 在某些情况下阻塞
        fd_ = ::open(device.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK | O_CLOEXEC);
        if (fd_ < 0)
        {
            baud_ = 0;
            setError(("打开 " + device).c_str());
            return false;
        }
        if (!::isatty(fd_))
        {
            setErrorMsg(device + " 不是 tty 设备");
            close();
            return false;
        }
        if (!configure())
        {
            close();
            return false;
        }
        return true;
    }

    /**
     * 关闭串口并把 baudrate() 置 0；可重复调用，保留最近错误信息。
     * 不发送卸力报文，也不确认输出队列已发完；控制程序应通过 GoBus::close() 退出。
     */
    void SerialPort::close()
    {
        baud_ = 0;
        if (fd_ >= 0)
        {
            ::close(fd_);
            fd_ = -1;
        }
    }

    /**
     * 丢弃内核尚未读取的输入字节，成功 true，失败 false 并记录错误；不清空输出队列。
     * 仅在新事务发送前调用；发送后调用可能丢掉电机立即返回的应答。
     */
    bool SerialPort::discardInput()
    {
        error_.clear();
        if (fd_ < 0)
        {
            setErrorMsg("port closed");
            return false;
        }
        if (::tcflush(fd_, TCIFLUSH) != 0)
        {
            setError("discard input");
            return false;
        }
        return true;
    }
    /**
     * 内部使用 poll 等待 events（POLLIN/POLLOUT），直到 end；就绪 true，超时/断开/错误 false。
     * EINTR 后沿用原截止时间；毫秒取整和系统调度可能使实际返回略晚，不提供硬实时保证。
     */
    bool SerialPort::wait(short events, Deadline end)
    {
        while (true)
        {
            auto remaining = std::chrono::duration_cast<std::chrono::microseconds>(end - Clock::now()).count();
            if (remaining <= 0)
            {
                read_status_ = ReadStatus::Timeout;
                setErrorMsg("I/O timeout");
                return false;
            }
            pollfd p{fd_, events, 0};
            int ms = int(std::min<long long>((remaining + 999) / 1000, 10000));
            int n = ::poll(&p, 1, ms);
            if (n < 0)
            {
                if (errno == EINTR)
                    continue;
                setError("poll");
                return false;
            }
            if (!n)
                continue;
            if (p.revents & (POLLERR | POLLHUP | POLLNVAL))
            {
                setErrorMsg("serial device disconnected/error");
                return false;
            }
            if (p.revents & events)
                return true;
        }
    }
    /**
     * 使用共同截止时间 end 写 data 的 len 字节，处理部分写入和可重试中断。
     * 返回实际写入数，必须等于 len 才算写完整；失败时可能已经写出一部分。
     * 完整写入只说明内核接收了数据，不等于电机已经执行；data 在调用期间必须有效。
     */
    size_t SerialPort::writeUntil(const uint8_t *data, size_t len, Deadline end)
    {
        error_.clear();
        if (fd_ < 0 || !data)
        {
            setErrorMsg("invalid write/closed port");
            return 0;
        }
        size_t off = 0;
        // wait()是等待串口写入，最多等到end，然后使用linux系统中的系统函数write，从data+off字节发送到len-off字节，
        // while是为了保险，防止write只写了一部分数据，一般不会发生，运行一次就结束，除非设备延时等原因。
        while (off < len && wait(POLLOUT, end))
        {
            auto n = ::write(fd_, data + off, len - off);
            if (n < 0)
            {
                if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)
                    continue;
                setError("write");
                break;
            }
            if (!n)
            {
                setErrorMsg("zero length write");
                break;
            }
            off += size_t(n);
        }
        return off;
    }
    /**
     * 在截止时间 end 前等待可读，最多读 len 字节到 data（需有足够空间）。
     * 第一次读到正数字节就返回，不等满 len、不检查帧头/CRC/ID；0 表示超时或错误。
     * 组帧时重复调用并保留半帧，各次调用继续使用同一个 end。
     */
    size_t SerialPort::readUntil(uint8_t *data, size_t len, Deadline end)
    {
        error_.clear();
        read_status_ = ReadStatus::Error;
        if (fd_ < 0 || !data || !len)
        {
            setErrorMsg("invalid read/closed port");
            return 0;
        }
        while (wait(POLLIN, end))
        {
            auto n = ::read(fd_, data, len);
            if (n < 0)
            {
                if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)
                    continue;
                setError("read");
                return 0;
            }
            if (!n)
            {
                setErrorMsg("serial EOF");
                return 0;
            }
            read_status_ = ReadStatus::Success;
            return size_t(n);
        }
        return 0;
    }
    /**
     * 在新建的 deadline() 内尝试写完 data 的 len 字节，返回实际写入数。
     * 调用方必须与 len 比较；小于 len 表示未完成，可查看 lastError()。
     */
    size_t SerialPort::write(const uint8_t *p, size_t n) { return writeUntil(p, n, deadline()); }
    /**
     * 在新建的 deadline() 内等待一批输入，最多读取 len 字节到 data，返回实际字节数。
     * 返回正数也可能只有半帧；返回 0 表示失败。需要协议解析时使用 GoBus。
     */
    size_t SerialPort::read(uint8_t *p, size_t n) { return readUntil(p, n, deadline()); }
    /**
     * 原始字节便捷接口：清理旧输入、写 txlen 字节、读取最多 rxmax 字节，共用一个截止时间。
     * 返回读到的字节数，0 表示失败；rx 需有 rxmax 字节空间。
     * 它不保证完整反馈帧，也不验证 CRC/ID；电机控制应调用 GoBus::cycle()。
     */
    size_t SerialPort::sendRecv(const uint8_t *tx, size_t tn, uint8_t *rx, size_t rn)
    {
        auto end = deadline();
        if (!discardInput() || writeUntil(tx, tn, end) != tn)
            return 0;
        return readUntil(rx, rn, end);
    }
} // namespace go

namespace go {
SerialPort::ReadResult SerialPort::readSome(uint8_t* data, size_t len, int timeout_ms) {
    if (timeout_ms < 0 || fd_ < 0 || !data || !len) {
        setErrorMsg("invalid read/closed port");
        return {ReadStatus::Error, 0};
    }
    if (timeout_ms > 0) {
        const size_t count = readUntil(data, len, Clock::now() +
                                      std::chrono::milliseconds(timeout_ms));
        return {read_status_, count};
    }
    // Immediate check is supported for IMU; do not flush streaming input.
    error_.clear();
    pollfd event{fd_, POLLIN, 0};
    const int ready = ::poll(&event, 1, 0);
    if (ready < 0) {
        if (errno == EINTR) return {ReadStatus::Timeout, 0};
        setError("poll"); return {ReadStatus::Error, 0};
    }
    if (event.revents & (POLLERR | POLLHUP | POLLNVAL)) {
        setErrorMsg("serial device disconnected/error");
        return {ReadStatus::Error, 0};
    }
    if (!ready || !(event.revents & POLLIN)) return {ReadStatus::Timeout, 0};
    const ssize_t count = ::read(fd_, data, len);
    if (count > 0) return {ReadStatus::Success, static_cast<size_t>(count)};
    if (count < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK))
        return {ReadStatus::Timeout, 0};
    if (count == 0) setErrorMsg("serial EOF"); else setError("read");
    return {ReadStatus::Error, 0};
}
} // namespace go
