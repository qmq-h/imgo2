#include "go_safety.h"
#include "apps/runtime.h"
#include <iostream>
#include <limits>
#define CHECK(x) do{if(!(x)){std::cerr<<#x<<'\n';return 1;}}while(0)
int main() {
    go::MotorConfig config;go::Targets start,goal;
    for(const auto& b:config.buses) start.q.insert(start.q.end(),b.start_pose_rad.begin(),b.start_pose_rad.end());
    for(int i=0;i<4;++i) goal.q.insert(goal.q.end(),{0,.87f,-1.82f});
    for(auto* t:{&start,&goal}) {t->dq.assign(12,0);t->tau.assign(12,0);t->kp.assign(12,25);t->kd.assign(12,.5f);}
    std::string error;CHECK(go::Safety::trajectoryValid(start,goal,8,config.limits,error));
    CHECK(!go::Safety::trajectoryValid(start,goal,.001,config.limits,error));
    goal.q[0]=1;CHECK(!go::Safety::trajectoryValid(start,goal,8,config.limits,error));goal.q[0]=0;
    goal.kp[0]=71;CHECK(!go::Safety::trajectoryValid(start,goal,8,config.limits,error));goal.kp[0]=25;
    start.q[0]=std::numeric_limits<float>::quiet_NaN();CHECK(!go::Safety::trajectoryValid(start,goal,8,config.limits,error));
    CHECK(go_app::ids("0,1,2")==std::vector<int>({0,1,2}));
    for(const char* bad:{"a","0,","15"}) {bool rejected=false;try{go_app::ids(bad);}catch(const std::invalid_argument&){rejected=true;}CHECK(rejected);}
    std::cout<<"trajectory validation and CLI passed\n";
}
