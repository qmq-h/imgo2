#include "apps/deployment_input.h"
#include "apps/runtime.h"
#include <cstdio>
#include <thread>
int main(int argc,char** argv) try {
    go_app::StopSignals signals;go_app::TerminalInput terminal;go_app::DeploymentInput input;
    std::puts("READY");std::fflush(stdout);
    if(argc>1 && std::string(argv[1])=="throw") throw std::runtime_error("test exception");
    while(!signals.requested()) {
        char c=0;
        const auto status=go_app::readDeploymentChar(c);
        if(status==go_app::InputStatus::Closed) return 0;
        if(status==go_app::InputStatus::Character) {
            const auto event=input.feed(c);
            if(event.kind==go_app::KeyEvent::Kind::Velocity) std::printf("KEY %d\n",int(event.key));
            if(event.kind==go_app::KeyEvent::Kind::State) std::printf("STATE %d\n",event.digit);
            std::fflush(stdout);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return 0;
}catch(const std::exception&){return 1;}
