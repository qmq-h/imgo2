#pragma once
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
namespace go {
// 底层 Linux 字节传输，不理解电机协议。同一实例的全部操作由一个线程串行执行。
class SerialPort {
public:
    enum class ReadStatus { Success, Timeout, Error };
    struct ReadResult { ReadStatus status; size_t count; };
    ReadResult readSome(uint8_t* data, size_t len, int timeout_ms);
    using Clock=std::chrono::steady_clock;
    using Deadline=Clock::time_point;
    /**
     * 创建未打开的串口对象，不访问设备。通常由 GoBus 持有，随后调用 open()。
     */
    SerialPort()=default;
    /**
     * 析构时关闭文件描述符。不发送电机卸力指令；卸力由 GoBus::close() 负责。
     */
    ~SerialPort();
    /**
     * 禁止复制/赋值：对象由单一所有者管理，避免重复持有设备或电机状态。
     */
    SerialPort(const SerialPort&)=delete;
    /**
     * 禁止复制/赋值：对象由单一所有者管理，避免重复持有设备或电机状态。
     */
    SerialPort& operator=(const SerialPort&)=delete;
    /**
     * 打开并配置 device（如 /dev/ttyCH343USB0），baud 为请求波特率。
     * timeout_us 为配置的等待预算，单位微秒，范围 1～10000000，默认 20000。
     * 每次调用先关闭旧连接；成功返回 true，失败返回 false 并保存 lastError()。
     * 成功只代表串口配置及速率回读通过，不代表已收到电机反馈。
     */
    bool open(const std::string& device,uint32_t baud,size_t timeout_us=20000);
    /**
     * 关闭串口并把 baudrate() 置 0；可重复调用，保留最近错误信息。
     * 不发送卸力报文，也不确认输出队列已发完；控制程序应通过 GoBus::close() 退出。
     */
    void close();
    /**
     * 返回文件描述符是否已打开。不能据此判断电机在线或线缆连接正常。
     */
    bool isOpen() const { return fd_>=0; }
    /**
     * 返回成功配置并经驱动回读匹配的波特率；未打开或打开失败后为 0。
     * 这是驱动配置值，不是仪器测得的物理线速。
     */
    uint32_t baudrate() const { return baud_; }
    /**
     * 返回配置的超时预算（微秒），不是上次收发的实际耗时。
     */
    size_t timeoutUs() const { return timeout_us_; }
    /**
     * 返回借用的 Linux 文件描述符，未打开为 -1。用于底层诊断；调用方不要自行关闭或并发读写。
     */
    int fd() const { return fd_; }
    /**
     * 返回本对象保存的最近错误。收发失败后立即查看；后续操作可能清空或覆盖它。
     */
    const std::string& lastError() const { return error_; }
    /**
     * 生成“单调时钟当前时刻 + timeoutUs()”的截止时间。
     * 一组请求和应答应共用同一个值传给 writeUntil/readUntil，避免每次重试重新计时。
     */
    Deadline deadline() const { return Clock::now()+std::chrono::microseconds(timeout_us_); }
    /**
     * 丢弃内核尚未读取的输入字节，成功 true，失败 false 并记录错误；不清空输出队列。
     * 仅在新事务发送前调用；发送后调用可能丢掉电机立即返回的应答。
     */
    bool discardInput();
    /**
     * 在新建的 deadline() 内尝试写完 data 的 len 字节，返回实际写入数。
     * 调用方必须与 len 比较；小于 len 表示未完成，可查看 lastError()。
     */
    size_t write(const uint8_t* data,size_t len);
    /**
     * 在新建的 deadline() 内等待一批输入，最多读取 len 字节到 data，返回实际字节数。
     * 返回正数也可能只有半帧；返回 0 表示失败。需要协议解析时使用 GoBus。
     */
    size_t read(uint8_t* data,size_t len);
    /**
     * 使用共同截止时间 end 写 data 的 len 字节，处理部分写入和可重试中断。
     * 返回实际写入数，必须等于 len 才算写完整；失败时可能已经写出一部分。
     * 完整写入只说明内核接收了数据，不等于电机已经执行；data 在调用期间必须有效。
     */
    size_t writeUntil(const uint8_t* data,size_t len,Deadline end);
    /**
     * 在截止时间 end 前等待可读，最多读 len 字节到 data（需有足够空间）。
     * 第一次读到正数字节就返回，不等满 len、不检查帧头/CRC/ID；0 表示超时或错误。
     * 组帧时重复调用并保留半帧，各次调用继续使用同一个 end。
     */
    size_t readUntil(uint8_t* data,size_t len,Deadline end);
    /**
     * 原始字节便捷接口：清理旧输入、写 txlen 字节、读取最多 rxmax 字节，共用一个截止时间。
     * 返回读到的字节数，0 表示失败；rx 需有 rxmax 字节空间。
     * 它不保证完整反馈帧，也不验证 CRC/ID；电机控制应调用 GoBus::cycle()。
     */
    size_t sendRecv(const uint8_t* tx,size_t txlen,uint8_t* rx,size_t rxmax);
private:
    /**
     * 内部初始化：设置原始二进制 8N1、无流控，配置波特率并清理旧输入。
     * 仅在 open() 获得有效描述符后调用；任一步失败返回 false，由 open() 释放资源。
     */
    bool configure();
    /**
     * 内部设置 baud：优先 termios2/BOTHER，设置接口不可用时尝试标准波特率表。
     * 设置后必须回读匹配；回读失败或不匹配直接返回 false，不视为配置成功。
     */
    bool applyBaudrate(uint32_t baud);
    /**
     * 内部使用 poll 等待 events（POLLIN/POLLOUT），直到 end；就绪 true，超时/断开/错误 false。
     * EINTR 后沿用原截止时间；毫秒取整和系统调度可能使实际返回略晚，不提供硬实时保证。
     */
    bool wait(short events,Deadline end);
    /**
     * 系统调用失败后立即调用，保存操作名 what、当前 errno 数字和错误描述到本实例。
     */
    void setError(const std::string& what);
    /**
     * 保存不依赖 errno 的自定义错误 what，例如超时或参数非法；内部使用。
     */
    void setErrorMsg(const std::string& what) { error_=what; }
    ReadStatus read_status_=ReadStatus::Error;
    int fd_=-1;
    uint32_t baud_=0;
    size_t timeout_us_=20000;
    std::string error_;
};
}
