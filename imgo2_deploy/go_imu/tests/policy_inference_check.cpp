#include "sdk/policy_runner.h"
#include <fstream>
#include <iostream>
#include <iomanip>
int main(int argc,char** argv) try {
    if(argc!=3) return 1;
    imgo2_sdk::PolicyRunner runner(imgo2_sdk::PolicyConfig::load(argv[1]));runner.load();runner.warmup();
    std::ifstream input(argv[2]);if(!input) return 1;
    std::cout<<std::setprecision(9);
    std::vector<float> obs(45);
    while(input>>obs[0]) {
        for(size_t i=1;i<45;++i) if(!(input>>obs[i])) return 1;
        const auto out=runner.forward(obs);for(float x:out) std::cout<<x<<' ';std::cout<<'\n';
    }
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
