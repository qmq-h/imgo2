#include "input/controller_input.h"

#include <linux/input.h>
#include <sys/ioctl.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <string>

#include <fcntl.h>
#include <poll.h>
#include <unistd.h>

namespace imgo2_input {
namespace {

bool bitTest(const unsigned char* bits, int code) {
    return (bits[code / 8] >> (code % 8)) & 1;
}

struct AxisRange {
    int minimum;
    int maximum;
};

// 设备没有上报 EVIOCGABS（例如测试用的 FIFO）时，按 evdev 的常见默认值。
AxisRange defaultRange(int code) {
    switch (code) {
    case ABS_X:
    case ABS_Y:
    case ABS_RX:
    case ABS_RY:
        return {-32768, 32767};
    case ABS_Z:
    case ABS_RZ:
    case ABS_GAS:
    case ABS_BRAKE:
        return {0, 255};
    case ABS_HAT0X:
    case ABS_HAT0Y:
        return {-1, 1};
    default:
        return {0, 0};
    }
}

float applyDeadzone(float value, float deadzone) {
    const float magnitude = std::fabs(value);
    if (magnitude <= deadzone) return 0.f;
    const float scaled = std::min((magnitude - deadzone) / (1.f - deadzone), 1.f);
    return std::copysign(scaled, value);
}
} // namespace

struct ControllerInput::Impl {
    int fd = -1;
    float deadzone = 0.08f;
    std::array<input_absinfo, ABS_MAX + 1> abs_info{};
    std::array<bool, ABS_MAX + 1> has_abs_info{};
    std::array<int, ABS_MAX + 1> axis{};
    std::array<bool, KEY_MAX + 1> pressed{};
    std::string error;

    float signedValue(int code) const {
        AxisRange range;
        if (code <= ABS_MAX && has_abs_info[code]) {
            range = {abs_info[code].minimum, abs_info[code].maximum};
        } else {
            range = defaultRange(code);
        }
        if (range.maximum <= range.minimum) return 0.f;
        const float normalized =
            float(axis[code] - range.minimum) / float(range.maximum - range.minimum);
        return std::clamp(2.f * normalized - 1.f, -1.f, 1.f);
    }

    void fillAxes(GamepadCommand& command) const {
        // xpad 约定：X 左负右正，Y 上负下正，RX 左负右正。
        // 需求：上推/左推为速度正，右推角速度为正。
        command.vx = applyDeadzone(-signedValue(ABS_Y), deadzone);
        command.vy = applyDeadzone(-signedValue(ABS_X), deadzone);
        command.wz = applyDeadzone(signedValue(ABS_RX), deadzone);
    }
};

ControllerInput::ControllerInput() : impl_(new Impl) {}
ControllerInput::~ControllerInput() { close(); }

bool ControllerInput::open(const std::string& device, float deadzone, std::string& error) {
    close();
    device_ = device;
    impl_->error.clear();
    if (device.empty()) {
        error = "controller device is empty";
        return false;
    }
    if (!std::isfinite(deadzone) || deadzone < 0.f || deadzone >= 0.95f) {
        error = "controller deadzone must be in [0, 0.95)";
        return false;
    }
    impl_->fd = ::open(device.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    if (impl_->fd < 0) {
        error = "open " + device + ": " + std::strerror(errno);
        return false;
    }
    impl_->deadzone = deadzone;
    impl_->axis.fill(0);
    impl_->pressed.fill(false);
    impl_->has_abs_info.fill(false);

    std::array<unsigned char, (EV_MAX + 7) / 8 + 1> ev_bits{};
    std::array<unsigned char, (ABS_MAX + 7) / 8 + 1> abs_bits{};
    ioctl(impl_->fd, EVIOCGBIT(0, ev_bits.size()), ev_bits.data());
    const bool have_abs_bits =
        bitTest(ev_bits.data(), EV_ABS) &&
        ioctl(impl_->fd, EVIOCGBIT(EV_ABS, abs_bits.size()), abs_bits.data()) >= 0;

    if (have_abs_bits) {
        if (!bitTest(abs_bits.data(), ABS_X)) {
            error = device + " has no ABS_X (not a gamepad?)";
            close();
            return false;
        }
        for (int code = 0; code <= ABS_MAX; ++code) {
            if (!bitTest(abs_bits.data(), code)) continue;
            if (ioctl(impl_->fd, EVIOCGABS(code), &impl_->abs_info[code]) >= 0)
                impl_->has_abs_info[code] = true;
        }
    }
    return true;
}

void ControllerInput::close() {
    if (impl_ && impl_->fd >= 0) ::close(impl_->fd);
    if (impl_) {
        impl_->fd = -1;
        impl_->axis.fill(0);
        impl_->pressed.fill(false);
    }
}

bool ControllerInput::isOpen() const { return impl_ && impl_->fd >= 0; }
const std::string& ControllerInput::lastError() const { return impl_->error; }

ControllerPoll ControllerInput::poll(GamepadCommand& command) {
    command = GamepadCommand{};
    if (!isOpen()) return ControllerPoll::Disconnected;

    pollfd pfd{impl_->fd, POLLIN, 0};
    const int ready = ::poll(&pfd, 1, 0);
    if (ready < 0) {
        if (errno == EINTR) {
            impl_->fillAxes(command);
            command.connected = true;
            return ControllerPoll::Ok;
        }
        impl_->error = std::string("controller poll: ") + std::strerror(errno);
        close();
        return ControllerPoll::Error;
    }

    if (ready > 0) {
        if (pfd.revents & (POLLHUP | POLLERR | POLLNVAL)) {
            impl_->error = "controller disconnected";
            close();
            return ControllerPoll::Disconnected;
        }
        if (pfd.revents & POLLIN) {
            std::array<unsigned char, sizeof(input_event) * 64> buffer{};
            const ssize_t bytes = ::read(impl_->fd, buffer.data(), buffer.size());
            if (bytes < 0) {
                if (errno != EAGAIN && errno != EINTR) {
                    impl_->error = std::string("controller read: ") + std::strerror(errno);
                    const bool gone = errno == ENODEV || errno == EIO;
                    close();
                    return gone ? ControllerPoll::Disconnected : ControllerPoll::Error;
                }
            } else if (bytes == 0) {
                impl_->error = "controller disconnected";
                close();
                return ControllerPoll::Disconnected;
            } else {
                const int count = static_cast<int>(bytes) / static_cast<int>(sizeof(input_event));
                for (int i = 0; i < count; ++i) {
                    input_event event{};
                    std::memcpy(&event, buffer.data() + i * sizeof(input_event), sizeof event);
                    if (event.type == EV_KEY) {
                        if (event.code <= KEY_MAX) impl_->pressed[event.code] = event.value != 0;
                        if (event.value == 1) { // 只认按下沿，长按/重复不重复触发
                            if (event.code == BTN_SOUTH) command.stand = true;
                            else if (event.code == BTN_EAST) command.lie_down = true;
                            else if (event.code == BTN_WEST) command.enter_ppo = true;
                        }
                    } else if (event.type == EV_ABS && event.code <= ABS_MAX) {
                        impl_->axis[event.code] = event.value;
                    }
                }
            }
        }
    }

    if (!isOpen()) return ControllerPoll::Disconnected;
    impl_->fillAxes(command);
    command.connected = true;
    return ControllerPoll::Ok;
}
} // namespace imgo2_input
