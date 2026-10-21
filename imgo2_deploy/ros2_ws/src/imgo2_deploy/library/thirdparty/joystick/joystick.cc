#include "joystick.hh"

#include <cerrno>
#include <cstring>

#if defined(__linux__)
#include <fcntl.h>
#include <linux/joystick.h>
#include <unistd.h>
#endif

namespace {
#if !defined(__linux__)
struct js_event {
    std::uint32_t time;
    std::int16_t value;
    std::uint8_t type;
    std::uint8_t number;
};

constexpr std::uint8_t JS_EVENT_BUTTON = 0x01;
constexpr std::uint8_t JS_EVENT_AXIS = 0x02;
constexpr std::uint8_t JS_EVENT_INIT = 0x80;
#endif
}  // namespace

JoystickEvent::JoystickEvent() : time(0), value(0), type(0), number(0) {}

JoystickEvent::JoystickEvent(const js_event& event)
    : time(event.time), value(event.value), type(static_cast<std::uint8_t>(event.type & ~JS_EVENT_INIT)), number(event.number) {}

bool JoystickEvent::isButton() const {
    return type == JS_EVENT_BUTTON;
}

bool JoystickEvent::isAxis() const {
    return type == JS_EVENT_AXIS;
}

Joystick::Joystick(const std::string& device_path) : fd_(-1) {
#if defined(__linux__)
    fd_ = open(device_path.c_str(), O_RDONLY | O_NONBLOCK);
#else
    (void)device_path;
#endif
}

Joystick::~Joystick() {
#if defined(__linux__)
    if (fd_ >= 0) {
        close(fd_);
    }
#endif
}

bool Joystick::isFound() const {
    return fd_ >= 0;
}

bool Joystick::sample(JoystickEvent* event) const {
#if defined(__linux__)
    if (fd_ < 0 || event == nullptr) {
        return false;
    }

    js_event raw_event;
    const ssize_t bytes = read(fd_, &raw_event, sizeof(raw_event));
    if (bytes == static_cast<ssize_t>(sizeof(raw_event))) {
        *event = JoystickEvent(raw_event);
        return true;
    }

    if (bytes < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        return false;
    }
    return false;
#else
    (void)event;
    return false;
#endif
}
