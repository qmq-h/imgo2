#pragma once
#include <cctype>
#include <string>
#include <termios.h>
#include <unistd.h>
#include <stdexcept>
#include <poll.h>
#include <cerrno>
namespace go_app {
struct KeyEvent {
    enum class Kind {None,State,Velocity};
    Kind kind=Kind::None;
    int digit=-1;
    char key=0;
};
class DeploymentInput {
public:
    KeyEvent feed(char c) {
        // Ignore terminal escape sequences (arrows/function keys), including their final A/D.
        if(escape_) {
            if(escape_==1 && (c=='[' || c=='O')) escape_=2;
            else if(escape_==1 || (c>='@' && c<='~')) escape_=0;
            return {};
        }
        if(c==27) {escape_=1;return {};}
        if(c=='\r') c='\n';
        if(c=='\n') {
            int digit=-1;
            if(!overflow_ && line_.size()==1 && (line_[0]=='0' || line_[0]=='1' || line_[0]=='2' || line_[0]=='9')) digit=line_[0]-'0';
            line_.clear();overflow_=false;
            return digit<0?KeyEvent{}:KeyEvent{KeyEvent::Kind::State,digit,0};
        }
        if(c=='\b' || c==127) {if(!line_.empty()) line_.pop_back();return {};}
        const char key=char(std::tolower(static_cast<unsigned char>(c)));
        if(key=='w' || key=='s' || key=='a' || key=='d' || key=='q' || key=='e' || key==' ') {
            if(!line_.empty()) append(c); // Mixed candidate cannot become a state request.
            return {KeyEvent::Kind::Velocity,-1,key};
        }
        append(c);return {};
    }
private:
    void append(char c) {if(line_.size()<32) line_+=c;else overflow_=true;}
    std::string line_;
    bool overflow_=false;
    int escape_=0;
};
enum class InputStatus {Waiting, Character, Closed};
inline InputStatus readDeploymentChar(char& c) {
    pollfd input{STDIN_FILENO,POLLIN,0};
    const int ready=::poll(&input,1,0);
    if(ready<0) {if(errno==EINTR) return InputStatus::Waiting;throw std::runtime_error("terminal poll failed");}
    if(!ready) return InputStatus::Waiting;
    if(input.revents&POLLIN) {
        const auto n=::read(STDIN_FILENO,&c,1);
        if(n==1) return InputStatus::Character;
        if(n<0 && (errno==EINTR || errno==EAGAIN)) return InputStatus::Waiting;
        return InputStatus::Closed;
    }
    return input.revents&(POLLHUP|POLLERR|POLLNVAL)?InputStatus::Closed:InputStatus::Waiting;
}
class TerminalInput {
public:
    explicit TerminalInput(int fd=STDIN_FILENO):fd_(fd) {
        if(!::isatty(fd_)) return;
        if(::tcgetattr(fd_,&saved_)<0) throw std::runtime_error("cannot read terminal settings");
        auto raw=saved_;raw.c_lflag&=static_cast<tcflag_t>(~(ICANON|ECHO));
        raw.c_lflag|=ISIG;raw.c_cc[VMIN]=0;raw.c_cc[VTIME]=0;
        if(::tcsetattr(fd_,TCSANOW,&raw)<0) throw std::runtime_error("cannot configure terminal");
        changed_=true;
    }
    ~TerminalInput() {if(changed_) ::tcsetattr(fd_,TCSANOW,&saved_);}
    TerminalInput(const TerminalInput&)=delete;
    TerminalInput& operator=(const TerminalInput&)=delete;
private:
    int fd_;bool changed_=false;termios saved_{};
};
}
