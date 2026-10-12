#include "serial_port.h"
#include <pty.h>
#include <unistd.h>
#include <stdexcept>
#include <iostream>
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)
int main() {
    int master = -1, slave = -1;
    try {
        char name[128];
        CHECK(openpty(&master, &slave, name, nullptr, nullptr) == 0);
        go::SerialPort port;
        CHECK(port.open(name, 460800));
        CHECK(port.baudrate() == 460800);
        close(slave); slave = -1;
        uint8_t bytes[32];
        CHECK(port.readSome(bytes, sizeof(bytes), 0).status == go::SerialPort::ReadStatus::Timeout);
        CHECK(port.readSome(bytes, sizeof(bytes), 10).status == go::SerialPort::ReadStatus::Timeout);
        const uint8_t data[] = {0x59, 0x53, 0x00};
        CHECK(write(master, data, sizeof(data)) == sizeof(data));
        auto result = port.readSome(bytes, sizeof(bytes), 0);
        // PTY delivery may not be visible to a zero-time poll immediately.
        if (result.status == go::SerialPort::ReadStatus::Timeout)
            result = port.readSome(bytes, sizeof(bytes), 100);
        CHECK(result.status == go::SerialPort::ReadStatus::Success && result.count == 3);
        CHECK(bytes[0] == 0x59 && bytes[1] == 0x53);
        close(master); master = -1;
        CHECK(port.readSome(bytes, sizeof(bytes), 100).status == go::SerialPort::ReadStatus::Error);
        CHECK(!port.lastError().empty());
        port.close();
        CHECK(port.readSome(bytes, sizeof(bytes), 0).status == go::SerialPort::ReadStatus::Error);
        CHECK(openpty(&master, &slave, name, nullptr, nullptr) == 0);
        CHECK(port.open(name, 4000000));
        CHECK(port.baudrate() == 4000000);
        port.close();
        close(master); close(slave);
        std::cout << "Shared serial statuses and baud rates passed\n";
    } catch (const std::exception& error) {
        if (master >= 0) close(master);
        if (slave >= 0) close(slave);
        std::cerr << error.what() << '\n'; return 1;
    }
}
