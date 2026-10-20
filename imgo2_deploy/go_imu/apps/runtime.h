#pragma once
#include <csignal>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/select.h>
#include <unistd.h>
namespace go_app {
inline volatile std::sig_atomic_t& stopFlag() {static volatile std::sig_atomic_t flag=0;return flag;}
inline void onSignal(int) {stopFlag()=1;}
class StopSignals {
public:
    StopSignals() {stopFlag()=0;old_int_=std::signal(SIGINT,onSignal);old_term_=std::signal(SIGTERM,onSignal);}
    ~StopSignals() {std::signal(SIGINT,old_int_);std::signal(SIGTERM,old_term_);}
    bool requested() const {return stopFlag()!=0;}
private:
    using Handler=void(*)(int);
    Handler old_int_,old_term_;
};
inline bool readChar(char& c) {
    fd_set set;FD_ZERO(&set);FD_SET(STDIN_FILENO,&set);timeval timeout{0,0};
    return ::select(STDIN_FILENO+1,&set,nullptr,nullptr,&timeout)>0 && ::read(STDIN_FILENO,&c,1)==1;
}
inline long integer(const std::string& value,long low,long high,const char* option) {
    size_t used=0;long result=0;
    try {result=std::stol(value,&used);} catch(const std::exception&) {throw std::invalid_argument(std::string(option)+": invalid integer");}
    if(used!=value.size() || result<low || result>high) throw std::invalid_argument(std::string(option)+": value out of range");
    return result;
}
inline std::vector<int> ids(const std::string& list) {
    std::vector<int> result;
    for(size_t pos=0;;) {
        const size_t comma=list.find(',',pos);
        result.push_back(static_cast<int>(integer(list.substr(pos,comma==std::string::npos?comma:comma-pos),0,14,"-i")));
        if(comma==std::string::npos) return result;
        pos=comma+1;
    }
}
}
