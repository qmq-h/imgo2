#ifndef THIRD_PARTY_JOYSTICK_HH_
#define THIRD_PARTY_JOYSTICK_HH_

#include <cstdint>
#include <string>

class JoystickEvent {
public:
    JoystickEvent();
    explicit JoystickEvent(const struct js_event& event);

    bool isButton() const;
    bool isAxis() const;

    std::uint32_t time;
    std::int16_t value;
    std::uint8_t type;
    std::uint8_t number;
};

class Joystick {
public:
    explicit Joystick(const std::string& device_path);
    ~Joystick();

    bool isFound() const;
    bool sample(JoystickEvent* event) const;

private:
    int fd_;
};

#endif  // THIRD_PARTY_JOYSTICK_HH_
