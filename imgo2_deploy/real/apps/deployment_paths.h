#pragma once
#include <filesystem>
#include <stdexcept>
#include <unistd.h>
namespace go_app {
inline std::filesystem::path executableDirectory() {
    char path[4096];
    const auto count=::readlink("/proc/self/exe",path,sizeof(path)-1);
    if(count<=0 || count==sizeof(path)-1) throw std::runtime_error("cannot locate executable");
    path[count]=0;
    return std::filesystem::path(path).parent_path();
}
}
