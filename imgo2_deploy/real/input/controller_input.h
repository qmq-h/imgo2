#pragma once
#include <memory>
#include <string>
namespace imgo2_input {

/**
 * 一帧手柄命令（已从原始 evdev 值换算好）。
 *
 * 轴值都是 -1..+1、已过死区：
 *   vx = 左摇杆前后（推上 = +1）
 *   vy = 左摇杆左右（推左 = +1）
 *   wz = 右摇杆水平（推右 = +1）
 * 按钮只在“按下沿”的那一帧为 true，长按不会重复触发。
 */
struct GamepadCommand {
    bool connected = false;
    bool stand = false;      // A
    bool lie_down = false;   // B
    bool enter_ppo = false;  // Y
    float vx = 0.f;
    float vy = 0.f;
    float wz = 0.f;
};

enum class ControllerPoll { Ok, Disconnected, Error };

/**
 * 只读 evdev 手柄输入（/dev/input/event* 或 udev 别名）。
 * 非阻塞：poll() 每个控制拍调用一次，没事件时返回当前轴值。
 * 线程约定：只在一个线程里使用（与主控制环同线程）。
 */
class ControllerInput {
public:
    ControllerInput();
    ~ControllerInput();
    ControllerInput(const ControllerInput&) = delete;
    ControllerInput& operator=(const ControllerInput&) = delete;

    // deadzone 是 0..1（推荐 0.08）。失败时 error 写原因。
    bool open(const std::string& device, float deadzone, std::string& error);
    void close();
    bool isOpen() const;
    const std::string& device() const { return device_; }
    const std::string& lastError() const;

    // 读取“当前这一拍”的命令；没有事件也会填当前轴值。
    ControllerPoll poll(GamepadCommand& command);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string device_;
};
} // namespace imgo2_input
