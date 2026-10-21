#pragma once
#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cwchar>
#include <sstream>
#include <string>
#include <sys/ioctl.h>
#include <unistd.h>
namespace imu_sensor {
// Only the main thread renders. No escape sequences in redirected output.
class TerminalDisplay {
public:
    TerminalDisplay() : terminal_(::isatty(STDOUT_FILENO)) {
        std::setlocale(LC_CTYPE, "");
        if (terminal_) std::fputs("\033[?1049h\033[2J\033[H\033[?25l", stdout);
    }
    ~TerminalDisplay() { Restore(); }
    void Restore() {
        if (terminal_) {
            std::fputs("\033[?25h\033[?1049l", stdout);
            std::fflush(stdout);
            terminal_ = false;
        }
    }
    int Columns() const { return Size().ws_col; }
    int Rows() const { return Size().ws_row; }
    void Show(const std::string& text) const {
        if (!terminal_) { std::fputs(text.c_str(), stdout); std::fflush(stdout); return; }
        const auto size = Size();
        const int lines = static_cast<int>(std::count(text.begin(), text.end(), '\n'));
        std::string screen = text;
        if (size.ws_row <= lines || size.ws_col < 53)
            screen = "Terminal too small; enlarge to at least 53 columns and " +
                std::to_string(lines + 1) + " rows. Ctrl+C exits.\n";
        std::fputs("\033[H", stdout);
        std::istringstream input(screen);
        std::string line;
        int row = 0;
        while (std::getline(input, line) && row < std::max(1, int(size.ws_row) - 1)) {
            // Clip by display columns (including CJK), never split UTF-8 bytes.
            std::mbstate_t state{};
            size_t pos = 0, end = 0;
            int width = 0;
            while (pos < line.size()) {
                wchar_t character;
                size_t count = std::mbrtowc(&character, line.data() + pos, line.size() - pos, &state);
                if (count == size_t(-1) || count == size_t(-2) || count == 0) {
                    state = {}; count = 1; character = L'?';
                }
                const int columns = std::max(0, ::wcwidth(character));
                if (width + columns > std::max(1, int(size.ws_col) - 1)) break;
                width += columns; pos += count; end = pos;
            }
            std::fwrite(line.data(), 1, end, stdout);
            std::fputs("\033[K\n", stdout);
            ++row;
        }
        std::fputs("\033[J", stdout);
        std::fflush(stdout);
    }
private:
    winsize Size() const {
        winsize size{};
        if (::ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) < 0 || !size.ws_col || !size.ws_row)
            size = {24, 80, 0, 0};
        return size;
    }
    bool terminal_;
};
} // namespace imu_sensor
